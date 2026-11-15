"""FastAPI server for the image-to-3D benchmark.

Run it either way:

    # as a module (relative imports resolve)
    uvicorn src.api.server:app --host 127.0.0.1 --port 8000

    # directly
    python src/api/server.py

API-18: the ``from ..config import ...`` statements need package context, so
running this file as a script used to raise ImportError before anything else
could run. The bootstrap below re-execs as a module in that case.
"""

import sys as _sys

# API-18: must run before the relative imports below. When this file is
# executed as a script, __package__ is empty and `from ..config import ...`
# fails; re-exec as a module so both entry points work.
if __package__ in (None, ""):
    from pathlib import Path as _Path

    _repo_root = str(_Path(__file__).resolve().parents[2])
    if _repo_root not in _sys.path:
        _sys.path.insert(0, _repo_root)
    import runpy as _runpy

    _runpy.run_module("src.api.server", run_name="__main__")
    raise SystemExit(0)

import asyncio
import io
import logging
import os
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Security, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from PIL import Image
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

# SEC-29: accept only formats this service actually supports. PIL will happily
# decode many other formats, and some of those parsers have a history of
# memory-safety bugs, so the allow-list is checked against both the declared
# Content-Type and the format PIL actually decoded.
ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG", "WEBP", "BMP"}
ALLOWED_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/bmp",
}

from ..config import EngineType  # noqa: E402  (bootstrap above must run first)
from ..engines import (  # noqa: E402  (bootstrap above must run first)
    AbstractEngine,
    LGMEngine,
    SF3DEngine,
    Spar3dEngine,
    TrellisEngine,
    TriposrEngine,
)
from ..export.glb_exporter import export_glb  # noqa: E402
from ..export.ply_exporter import export_ply  # noqa: E402
from ..preprocessing.background_removal import BackgroundRemover  # noqa: E402

app = FastAPI(title="Image-to-3D Benchmark API")

# SECURITY: API key authentication
API_KEY = os.environ.get("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    if not API_KEY:
        return True  # No auth required if API_KEY not set
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True


# SECURITY: Simple rate limiting (requests per minute per IP)
_rate_limits: dict[str, list] = defaultdict(list)
RATE_LIMIT = 10  # requests per minute

# API-09: nothing bounded concurrent inference, so N simultaneous requests
# meant N models resident on one GPU. A semaphore caps in-flight predictions;
# the timeout stops one slow request from blocking the queue behind it.
MAX_CONCURRENT_INFERENCE = int(os.environ.get("MAX_CONCURRENT_INFERENCE", "1"))
INFERENCE_TIMEOUT_S = float(os.environ.get("INFERENCE_TIMEOUT_S", "600"))
_inference_slots = asyncio.Semaphore(MAX_CONCURRENT_INFERENCE)


async def rate_limit(request):
    import time

    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    _rate_limits[client_ip] = [t for t in _rate_limits[client_ip] if now - t < 60]
    if len(_rate_limits[client_ip]) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    _rate_limits[client_ip].append(now)
    return True


# Lazy engine cache
_engines: dict[EngineType, AbstractEngine] = {}
_bg_remover = None  # Lazy background remover
_bg_lock = threading.Lock()  # API-07: guards _bg_remover creation

# API-07: get_engine is a check-then-construct. Two concurrent first-requests
# for the same engine could both see the cache empty and each load the
# multi-GB model, OOMing the GPU. A per-engine lock makes construction
# happen exactly once; the outer check keeps the common case lock-free.
_engine_locks: dict[EngineType, threading.Lock] = {}
_engine_locks_guard = threading.Lock()


def _lock_for(engine_type: EngineType) -> threading.Lock:
    with _engine_locks_guard:
        lock = _engine_locks.get(engine_type)
        if lock is None:
            lock = threading.Lock()
            _engine_locks[engine_type] = lock
        return lock


def get_engine(engine_type: EngineType) -> AbstractEngine:
    engine = _engines.get(engine_type)
    if engine is not None:
        return engine

    with _lock_for(engine_type):
        # Re-check inside the lock: another thread may have built it while we
        # were waiting.
        engine = _engines.get(engine_type)
        if engine is not None:
            return engine

        if engine_type == EngineType.TRIPOSR:
            engine = TriposrEngine()
        elif engine_type == EngineType.LGM:
            engine = LGMEngine()
        elif engine_type == EngineType.TRELLIS:
            engine = TrellisEngine()
        elif engine_type == EngineType.SPAR3D:
            engine = Spar3dEngine()
        elif engine_type == EngineType.SF3D:
            engine = SF3DEngine()
        else:
            raise ValueError(f"Unknown engine: {engine_type}")

        _engines[engine_type] = engine
        return engine


# API-05: ExportConfig.output_dir was never read; the API hardcoded ./outputs.
# Load AppConfig at import time and honour an env override for deployments that
# mount a scratch volume.
def _resolve_output_dir() -> str:
    from ..config import AppConfig

    return os.environ.get("IMAGE_TO_3D_OUTPUT_DIR") or AppConfig().export.output_dir


OUTPUT_DIR = _resolve_output_dir()
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _cleanup_file(path: str) -> None:
    """Delete a generated artefact after it has been streamed to the client."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError:
        # Never let cleanup failure turn a successful response into a 500.
        pass


def _sweep_output_dir(max_age_s: float = 3600.0) -> int:
    """Delete artefacts left behind by a request that never completed.

    API-06: per-request BackgroundTask cleanup covers the normal path, but if
    the process dies mid-response the file is orphaned. Sweep stale files on
    startup so disk cannot grow without bound across restarts.

    Returns the number of files removed.
    """
    now = time.time()
    removed = 0
    try:
        entries = list(Path(OUTPUT_DIR).iterdir())
    except OSError:
        return 0

    for entry in entries:
        try:
            if not entry.is_file():
                continue
            if now - entry.stat().st_mtime < max_age_s:
                continue
            entry.unlink()
            removed += 1
        except OSError:
            # A file we cannot remove must not stop the sweep or the server.
            continue
    if removed:
        logger.info("Swept %d stale artefact(s) from %s", removed, OUTPUT_DIR)
    return removed


_sweep_output_dir()


async def _run_inference(call):
    """Run a blocking engine call under the concurrency cap and timeout.

    API-09: without this, every request ran its own inference immediately, so
    N simultaneous requests meant N models resident on one GPU.
    """
    async with _inference_slots:
        try:
            return await asyncio.wait_for(run_in_threadpool(call), timeout=INFERENCE_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.error("Inference exceeded %.0fs", INFERENCE_TIMEOUT_S)
            raise HTTPException(
                status_code=504,
                detail=f"Inference timed out after {INFERENCE_TIMEOUT_S:.0f}s",
            ) from None


@app.get("/health")
def health_check():
    # API-05: expose the resolved output dir so a deployment's config is
    # observable without reading the source.
    return {"status": "healthy", "output_dir": OUTPUT_DIR}


@app.get("/engines")
def list_engines():
    return [
        {
            "id": EngineType.TRIPOSR,
            "info": {
                "name": "TripoSR",
                "paper": "arXiv:2403.02151",
                "latency_approx": "<0.5s (A100) / ~1-2s (T4)",
                "vram_approx": "4–6 GB (fp16)",
                "output_format": "mesh (.glb)",
            },
            # API-03: what /predict returns when output_format is omitted.
            "default_output_format": "mesh",
            "supported_output_formats": ["mesh"],
        },
        {
            "id": EngineType.LGM,
            "info": {
                "name": "LGM",
                "paper": "arXiv:2402.05054",
                "latency_approx": "~5s total",
                "vram_approx": "~10 GB (fp16)",
                "output_format": "gaussians (.ply)",
            },
            "default_output_format": "gaussians",
            "supported_output_formats": ["gaussians", "mesh"],
        },
        {
            "id": EngineType.TRELLIS,
            "info": {
                "name": "TRELLIS",
                "paper": "arXiv:2412.01506",
                "latency_approx": "~6.5–10s (A100)",
                "vram_approx": "~16–24 GB (fp16)",
                "output_format": "mesh (.glb) / gaussians (.ply)",
            },
            "default_output_format": "mesh",
            "supported_output_formats": ["mesh", "gaussians"],
        },
        {
            "id": EngineType.SPAR3D,
            "info": {
                "name": "SPAR3D",
                "paper": "arXiv:2501.04689",
                "latency_approx": "~37.5s (T4)",
                "vram_approx": "~9.2 GB (T4)",
                "output_format": "mesh (.glb) — point diffusion + DMTet + PBR",
            },
            "default_output_format": "mesh",
            "supported_output_formats": ["mesh"],
        },
        {
            "id": EngineType.SF3D,
            "info": {
                "name": "SF3D",
                "paper": "arXiv:2408.00653",
                "latency_approx": "~0.5s (A100) / ~1.5-2s (T4)",
                "vram_approx": "~6–7 GB (low_vram_mode=True)",
                "output_format": "mesh (.glb) — textured UV mesh (DMTet)",
            },
            "default_output_format": "mesh",
            "supported_output_formats": ["mesh"],
        },
    ]


@app.post("/predict")
async def predict(
    request,
    file: UploadFile = File(...),
    engine_type: EngineType = Form(...),
    output_format: str = Form(None),
    _api_key: bool = Depends(verify_api_key),
    _rate_limit: bool = Depends(rate_limit),
):
    # API-03: /engines advertises a default artefact per engine. Honour an
    # explicit request, otherwise produce what the advertisement promises.
    requested_format = (output_format or "").lower() or None
    if requested_format is not None and requested_format not in ("mesh", "gaussians"):
        raise HTTPException(
            status_code=400,
            detail=f"output_format must be 'mesh' or 'gaussians', got '{output_format}'",
        )

    try:
        engine = get_engine(engine_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None

    # Initialize background remover (lazy)
    global _bg_remover
    if _bg_remover is None:
        # API-07: double-checked, so concurrent first-requests cannot each
        # build their own session.
        with _bg_lock:
            if _bg_remover is None:
                try:
                    _bg_remover = BackgroundRemover()
                except Exception as e:
                    print(f"[API] Warning: Could not initialize background remover: {e}")

    try:
        contents = await file.read()

        # SECURITY: Validate image size to prevent decompression bombs
        if len(contents) > 10 * 1024 * 1024:  # 10 MB limit
            raise HTTPException(status_code=413, detail="Image too large (max 10 MB)")

        # SEC-29: reject anything that is not one of the formats we support,
        # based on the declared Content-Type.
        if file.content_type and file.content_type.lower() not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported media type '{file.content_type}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_MIME_TYPES))}",
            )

        image = Image.open(io.BytesIO(contents))

        # SEC-29: and again on what PIL actually decoded — a client can lie
        # about Content-Type, so the decoded format is authoritative.
        if (image.format or "").upper() not in ALLOWED_IMAGE_FORMATS:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported image format '{image.format}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_IMAGE_FORMATS))}",
            )

        # SECURITY: Validate image dimensions to prevent decompression bombs
        if image.width > 4096 or image.height > 4096:
            raise HTTPException(
                status_code=413, detail="Image dimensions too large (max 4096x4096)"
            )

        # Keep RGBA to preserve alpha channel for foreground cropping
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        # Apply background removal if available
        if _bg_remover is not None:
            try:
                image = _bg_remover.remove(image)
            except Exception as e:
                print(f"[API] Warning: Background removal failed: {e}")

        # Run blocking GPU inference in threadpool to avoid blocking the event loop
        # API-03: LGM's predict() takes extract_mesh; every other engine is
        # fixed at construction, so the request is only honoured where it can be.
        if engine_type == EngineType.LGM:
            want_mesh = requested_format != "gaussians"

            def call(_e=engine, _img=image, _m=want_mesh):
                return _e.predict(_img, _m)
        else:
            mesh_only = {
                EngineType.TRIPOSR: "TripoSR",
                EngineType.SPAR3D: "SPAR3D",
                EngineType.SF3D: "SF3D",
            }
            if requested_format == "gaussians" and engine_type in mesh_only:
                raise HTTPException(
                    status_code=400,
                    detail=f"{mesh_only[engine_type]} only produces a mesh; "
                    "use output_format='mesh'.",
                )

            def call(_e=engine, _img=image):
                return _e.predict(_img)

        # API-09: bound concurrency and wait no longer than the timeout.
        result = await _run_inference(call)

        # API-03: prefer whichever artefact the caller asked for, then fall
        # back to whatever the engine produced.
        want_gaussians = requested_format == "gaussians"
        if want_gaussians and result.gaussians is not None:
            out_filename = f"{uuid.uuid4().hex}.ply"
            out_path = os.path.join(OUTPUT_DIR, out_filename)
            export_ply(result.gaussians, out_path)
            return FileResponse(
                path=out_path,
                filename=out_filename,
                media_type="application/octet-stream",
                background=BackgroundTask(_cleanup_file, out_path),
            )

        if result.mesh is not None:
            out_filename = f"{uuid.uuid4().hex}.glb"
            out_path = os.path.join(OUTPUT_DIR, out_filename)
            export_glb(result.mesh, out_path)
            # SECURITY: delete the exported artefact once the response is sent so
            # predictions cannot accumulate on disk (disk exhaustion / info leak).
            return FileResponse(
                path=out_path,
                filename=out_filename,
                media_type="model/gltf-binary",
                background=BackgroundTask(_cleanup_file, out_path),
            )
        elif result.gaussians is not None:
            out_filename = f"{uuid.uuid4().hex}.ply"
            out_path = os.path.join(OUTPUT_DIR, out_filename)
            export_ply(result.gaussians, out_path)
            return FileResponse(
                path=out_path,
                filename=out_filename,
                media_type="application/octet-stream",
                background=BackgroundTask(_cleanup_file, out_path),
            )
        else:
            raise HTTPException(
                status_code=500, detail="Engine returned neither mesh nor gaussians."
            )

    except HTTPException:
        # Preserve deliberate 4xx/5xx responses (bad input, missing engine output).
        # Without this, they would be swallowed and rewritten as a generic 500.
        raise
    except Exception:
        # SECURITY: never return exception text to the client (paths, model and
        # library internals). Log the detail server-side with a request ID the
        # operator can correlate.
        request_id = uuid.uuid4().hex
        logger.exception("Prediction failed (request_id=%s)", request_id)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error (request_id={request_id})",
        ) from None


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.server:app",
        host=os.environ.get("API_HOST", "127.0.0.1"),
        port=int(os.environ.get("API_PORT", "8000")),
    )
