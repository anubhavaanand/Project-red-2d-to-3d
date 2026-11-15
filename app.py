"""
Production App — Image to 3D Model
====================================
Upload any photo → background removed → 3D model generated → download GLB.

Paper: Benchmarking Feed-Forward Single-Image 3D Reconstruction Across
       Triplane, Gaussian, Point-Aware, and Structured Latent Paradigms

Verified T4 benchmark numbers (n=100, code 09131e6):
  | Engine | Latency | VRAM | CD   | F@0.1 |
  |--------|---------|------|------|-------|
  | TripoSR | 3.66 s | 2.32 GB | 0.137 | 0.790 |
  | LGM     | 11.69 s| 5.19 GB | 0.128 | 0.796 |
  | SPAR3D  | 37.54 s| 9.17 GB | 0.114 | 0.837 |
  | TRELLIS | 20.46 s| 6.65 GB | 0.079 | 0.937 |

Engines available:
  • TripoSR  (3.66 s, 2.32 GB VRAM) — fastest, lowest VRAM
  • TRELLIS  (20.46 s, 6.65 GB VRAM) — highest quality (best CD, F@0.1, PSNR)
  • LGM      (11.69 s, 5.19 GB VRAM) — Gaussian output (PLY)
  • SPAR3D   (37.54 s, 9.17 GB VRAM) — most compact meshes

Run locally:
    pip install gradio
    python app.py

Security:
    - Default: binds to 127.0.0.1 only (local access)
    - Set GRADIO_USERNAME + GRADIO_PASSWORD for auth
    - Set GRADIO_SERVER_NAME=0.0.0.0 for PaaS (requires auth)

Deploy to HuggingFace Spaces:
    1. Create a new Space (Gradio SDK, GPU T4 or A100)
    2. Push this file + requirements.txt to the Space repo
    3. Done — it auto-runs on HF infrastructure

HuggingFace Space URL (after deploy):
    https://huggingface.co/spaces/YOUR_USERNAME/image-to-3d
"""

from __future__ import annotations

import os
import sys
import time
import tempfile
import traceback
import uuid
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

# APP-06: import the package absolutely. When the project is installed
# (`pip install -e .`) or app.py is run as a module (`python -m app`), these
# resolve without touching sys.path. The fallback keeps `python app.py`
# working from a source checkout without installing; it appends rather than
# prepends (SEC-16) so a same-named module here cannot shadow an installed one.
try:
    from src.preprocessing.background_removal import BackgroundRemover
except ModuleNotFoundError:  # pragma: no cover - source checkout only
    if str(Path(__file__).parent) not in sys.path:
        sys.path.append(str(Path(__file__).parent))
    from src.preprocessing.background_removal import BackgroundRemover

from src.engines.triposr_engine import TriposrEngine
from src.engines.trellis_engine import TrellisEngine
from src.engines.lgm_engine import LGMEngine
from src.engines.spar3d_engine import Spar3dEngine
from src.export.glb_exporter import export_glb
from src.export.ply_exporter import export_ply

# ---------------------------------------------------------------------------
# Global state — engines loaded lazily on first use to save startup time
# ---------------------------------------------------------------------------

_engines: dict = {}
_bg_remover: BackgroundRemover | None = None

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# SEC-27: /tmp is world-writable and the old fixed subdirectory name was
# predictable, so any local user could read (or pre-create) generated models.
# Use a per-user directory and restrict it to 0o700.
OUTPUT_DIR = Path(tempfile.gettempdir()) / f"image_to_3d_outputs-{os.getuid()}"
OUTPUT_DIR.mkdir(exist_ok=True, mode=0o700)
try:
    OUTPUT_DIR.chmod(0o700)
except OSError:
    # Some filesystems (e.g. certain mounted volumes) reject chmod; the
    # per-user path already prevents cross-user name collisions.
    pass


def _get_bg_remover() -> BackgroundRemover:
    global _bg_remover
    if _bg_remover is None:
        _bg_remover = BackgroundRemover()
    return _bg_remover


def _get_engine(name: str):
    global _engines
    if name not in _engines:
        if name == "TripoSR":
            e = TriposrEngine(device=DEVICE)
        elif name == "TRELLIS":
            e = TrellisEngine(device=DEVICE, output_format="mesh", slat_steps=8, sparse_steps=8)
        elif name == "LGM":
            e = LGMEngine(device=DEVICE)
        elif name == "SPAR3D":
            e = Spar3dEngine(device=DEVICE)
        else:
            raise ValueError(f"Unknown engine: {name}")
        e.load_model()
        _engines[name] = e
    return _engines[name]


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------

# SEC-22: upload limits. Gradio hands us an already-decoded PIL image, so the
# only defence left is to bound what we accept before it reaches a model.
MAX_IMAGE_PIXELS = 4096 * 4096  # ~16.7 MP; 4096x4096 RGBA decodes to ~67 MB
ALLOWED_IMAGE_MODES = {"RGB", "RGBA", "L", "LA", "P"}


def validate_upload(image: Image.Image | None) -> str | None:
    """Return an error message if ``image`` is unacceptable, else ``None``.

    Kept separate from :func:`generate_3d` so the limits can be exercised
    without importing the Gradio UI.
    """
    if image is None:
        return "⚠️ Please upload an image first."

    # SEC-22: reject unsupported modes rather than letting an exotic format
    # (CMYK, 16-bit, palette oddities) reach the engines.
    if image.mode not in ALLOWED_IMAGE_MODES:
        return f"❌ Unsupported image mode '{image.mode}'. Use RGB or RGBA."

    # SECURITY: bound width/height (decompression bombs)
    if image.width > 4096 or image.height > 4096:
        return "❌ Image too large (max 4096x4096)"

    # SEC-22: cap total pixel count too. A 4096x4096 RGBA upload decodes to
    # ~67 MB in memory before any processing starts.
    if image.width * image.height > MAX_IMAGE_PIXELS:
        return "❌ Image has too many pixels (max 16.7 MP)"

    return None


def generate_3d(
    image: Image.Image,
    engine_name: str,
    remove_bg: bool,
) -> tuple[str | None, str, Image.Image | None]:
    """
    Returns
    -------
    (glb_path_or_none, status_message, preview_image_or_none)
    """
    error = validate_upload(image)
    if error is not None:
        return None, error, None

    try:
        t0 = time.time()

        # 1. Background removal
        status = "Removing background…"
        if remove_bg:
            remover = _get_bg_remover()
            image = remover.remove(image)
        else:
            # Ensure white background if RGBA
            if image.mode == "RGBA":
                white = Image.new("RGB", image.size, "WHITE")
                white.paste(image, mask=image.split()[3])
                image = white

        # 2. Run engine
        status = f"Running {engine_name}…"
        engine = _get_engine(engine_name)
        result = engine.predict(image)
        latency = time.time() - t0

        # 3. Export
        # APP-03: a millisecond timestamp collides when two requests land in
        # the same millisecond (or on a coarse clock); uuid4 is unique.
        out_file = OUTPUT_DIR / f"output_{uuid.uuid4().hex}"

        if result.mesh is not None:
            glb_path = str(out_file.with_suffix(".glb"))
            export_glb(result.mesh, glb_path)
            vertex_count = len(result.mesh.vertices)
            face_count = len(result.mesh.faces)
            status_msg = (
                f"✅ Done in {latency:.1f}s\n"
                f"Engine: {engine_name}\n"
                f"Vertices: {vertex_count:,}  |  Faces: {face_count:,}\n"
                f"Format: GLB mesh"
            )
            # Quick 3-view thumbnail
            preview = _render_preview(result.mesh)
            return glb_path, status_msg, preview

        elif result.gaussians is not None:
            ply_path = str(out_file.with_suffix(".ply"))
            export_ply(result.gaussians, ply_path)
            g_count = len(result.gaussians.positions)
            status_msg = (
                f"✅ Done in {latency:.1f}s\n"
                f"Engine: {engine_name}\n"
                f"Gaussians: {g_count:,}\n"
                f"Format: PLY point cloud\n"
                f"Note: Use Gaussian viewer (e.g. SuperSplat) to open .ply files."
            )
            return ply_path, status_msg, None

        else:
            return None, "❌ Engine returned no output.", None

    except Exception as exc:
        return None, f"❌ Error: {type(exc).__name__}: {exc}", None


def _render_preview(mesh) -> Image.Image | None:
    """Generate a simple 3-view preview image using trimesh."""
    try:
        import trimesh

        tm = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            vertex_colors=(
                (mesh.vertex_colors * 255).astype(np.uint8)
                if mesh.vertex_colors is not None else None
            ),
            process=False,
        )
        # trimesh scene rendering (headless, no display needed)
        scene = tm.scene()
        frames = []
        for azim in [0, 90, 270]:
            scene.set_camera(angles=[0.3, 0, np.radians(azim)], distance=2.0)
            try:
                png = scene.save_image(resolution=(256, 256), visible=True)
                from io import BytesIO
                frames.append(Image.open(BytesIO(png)).convert("RGB"))
            except Exception:
                pass

        if not frames:
            return None

        # Stitch frames side by side
        total_w = sum(f.width for f in frames)
        combined = Image.new("RGB", (total_w, frames[0].height), "WHITE")
        x = 0
        for f in frames:
            combined.paste(f, (x, 0))
            x += f.width
        return combined
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

DESCRIPTION = """
# 🎯 Image → 3D Model

Upload any photo of an object and get a downloadable 3D model.

| Engine | Latency (T4) | VRAM (T4) | CD | F@0.1 |
|--------|-------------|-----------|-----|-------|
| **TripoSR** | 3.66 s | 2.32 GB | 0.137 | 0.790 |
| **LGM** | 11.69 s | 5.19 GB | 0.128 | 0.796 |
| **SPAR3D** | 37.54 s | 9.17 GB | 0.114 | 0.837 |
| **TRELLIS** | 20.46 s | 6.65 GB | 0.079 | 0.937 |

> **Tip:** For best results, use a clean image with the object centred and well-lit.  
> Background removal is applied automatically.
"""

# APP-07: resolve examples relative to this file, not the process CWD, so the
# UI works when started from another directory or imported by a server.
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"
EXAMPLES = [
    [str(EXAMPLES_DIR / "rubber_duck.jpg"), "TripoSR", True],
    [str(EXAMPLES_DIR / "sneaker.jpg"), "TRELLIS", True],
    [str(EXAMPLES_DIR / "vase.jpg"), "LGM", True],
]


def build_ui() -> gr.Blocks:
    # APP-07: create here, not in __main__, so a server that imports build_ui
    # also gets the directory.
    EXAMPLES_DIR.mkdir(exist_ok=True)

    with gr.Blocks(title="Image → 3D Model", theme=gr.themes.Soft()) as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Input Image",
                    type="pil",
                    height=320,
                )
                engine_choice = gr.Radio(
                    choices=["TripoSR", "TRELLIS", "LGM", "SPAR3D"],
                    value="TripoSR",
                    label="3D Engine",
                )
                remove_bg_toggle = gr.Checkbox(
                    value=True,
                    label="Auto-remove background",
                )
                run_btn = gr.Button("🚀 Generate 3D Model", variant="primary", size="lg")

            with gr.Column(scale=1):
                status_box = gr.Textbox(
                    label="Status",
                    lines=5,
                    interactive=False,
                    placeholder="Status will appear here after generation…",
                )
                preview_img = gr.Image(
                    label="3-View Preview",
                    type="pil",
                    interactive=False,
                    height=200,
                )
                output_file = gr.File(
                    label="⬇️ Download 3D Model (.glb / .ply)",
                    interactive=False,
                )

        run_btn.click(
            fn=generate_3d,
            inputs=[input_image, engine_choice, remove_bg_toggle],
            outputs=[output_file, status_box, preview_img],
            show_progress=True,
        )

        gr.Markdown("""
---
### 📖 How to view your 3D model
- **GLB files**: Open in [gltf.report](https://gltf.report), [3D Viewer (Windows)](https://apps.microsoft.com/detail/9nblggh42ths), or drag into [Blender](https://blender.org)
- **PLY files** (Gaussians): Open in [SuperSplat](https://supersplat.xyz) or [Gaussian Splatting Viewer](https://antimatter15.com/splat/)

### 📄 Paper
This app is the production demo for:  
*Triplanes, Gaussians, or Structured Latents? A Comparative Evaluation of Feed-Forward Paradigms for Single-Image 3D Reconstruction*
        """)

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_ui()

    # Security: require authentication for public deployments
    # Set GRADIO_USERNAME and GRADIO_PASSWORD env vars to enable auth
    # Set GRADIO_SERVER_NAME=0.0.0.0 for PaaS deployments (requires auth)
    auth_username = os.environ.get("GRADIO_USERNAME")
    auth_password = os.environ.get("GRADIO_PASSWORD")

    # Fail closed: both credentials must be set together
    if bool(auth_username) != bool(auth_password):
        sys.exit("[SECURITY] Both GRADIO_USERNAME and GRADIO_PASSWORD must be set (got only one). Refusing to start.")

    auth = None
    if auth_username and auth_password:
        auth = (auth_username, auth_password)
        print(f"[SECURITY] Authentication enabled for user: {auth_username}")
    else:
        print("[SECURITY] No credentials set — running in local-only mode (127.0.0.1)")

    app.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,         # SECURITY FIX: never auto-create public tunnel
        auth=auth,           # SECURITY FIX: require auth if credentials provided
        show_error=False,    # SECURITY FIX: never expose tracebacks to users
    )
