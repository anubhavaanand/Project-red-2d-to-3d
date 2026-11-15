"""
TRELLIS Engine — audited wrapper for microsoft/TRELLIS-image-large.

Fixes applied from repo_3_trellis.txt audit:
1. Built-in T4 single-GPU memory hygiene (`staged_offload=True`): keeps unused
   GS/RF decoders on CPU, runs under `@torch.no_grad()`, and offloads flow models
   + DINOv2 to CPU before 256^3 FlexiCubes decoding.
2. Uses `raw.vertices` (fixes `raw.vertex` typo) and preserves 3-channel RGB
   per-vertex colors from `raw.vertex_attrs[:, :3]`.
3. Rotates from TRELLIS's Blender Z-up coordinate system (`Rx(-90)`) to standard
   OpenGL Y-up coordinates.
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image

from ._mask import alpha_is_usable, derive_foreground_mask
from ._memory import free_cuda_cache
from ._paths import extend_sys_path
from .base import AbstractEngine, GaussianCloud, Mesh3D, ReconstructionResult

# ENG-29: TRELLIS reads these at import time, so they must be set before the
# engine is imported — but setting them at module scope mutated global state
# for every importer. load_model() now sets them just before the import.

# ENG-27: TRELLIS's pipeline can decode a radiance field, but predict() only
# consumes mesh and gaussian, so only those are accepted.
OutputFormat = Literal["mesh", "gaussian"]


class TrellisOutputError(RuntimeError):
    """ENG-22: raised when the TRELLIS pipeline returns an unexpected payload."""


def _first_of(value, label: str):
    """Return the first element of a pipeline output, whatever container it is.

    ENG-22: TRELLIS returns a list for most decoders, but indexing
    ``outputs["mesh"][0]`` blindly assumed both the key and the sequence type.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _normalize_vertex_colors(raw) -> tuple[np.ndarray | None, str]:
    """Return float RGB in [0, 1] plus a note describing the source range.

    ENG-23: TRELLIS vertex_attrs were clipped to [0, 1] without checking the
    decoder's actual range. A 0..255 decoder therefore produced an all-black
    mesh (everything clipped to 1 after scaling was skipped) or a saturated
    one. Detect the range from the data and scale accordingly.
    """
    if raw is None:
        return None, "absent"

    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    rgb = arr[:, :3]
    if rgb.shape[1] == 0:
        return None, "empty"

    peak = float(np.nanmax(rgb)) if rgb.size else 0.0
    if peak > 1.5:
        # 0..255 (or similar) — scale into [0, 1]
        return np.clip(rgb / 255.0, 0.0, 1.0), f"0..255 (peak {peak:.1f})"
    return np.clip(rgb, 0.0, 1.0), f"0..1 (peak {peak:.3f})"


class TrellisEngine(AbstractEngine):
    def __init__(
        self,
        device: str = "cuda",
        output_format: OutputFormat = "mesh",
        slat_steps: int = 12,
        sparse_steps: int = 12,
        sparse_cfg_strength: float = 7.5,
        slat_cfg_strength: float = 3.0,
        cfg_strength: float | None = None,
        staged_offload: bool = True,
    ):
        self.device = device
        self.dev = torch.device(device if ":" in device else f"{device}:0")
        self.output_format = output_format
        self.slat_steps = slat_steps
        self.sparse_steps = sparse_steps
        # ENG-26: cfg_strength used to be stored as self.cfg_strength and then
        # never read, so callers thought they were changing CFG. It is now an
        # explicit alias for sparse_cfg_strength only; slat_cfg_strength is
        # configured separately because the two samplers use different scales.
        if cfg_strength is not None:
            sparse_cfg_strength = cfg_strength
        self.sparse_cfg_strength = float(sparse_cfg_strength)
        self.slat_cfg_strength = float(slat_cfg_strength)
        self.staged_offload = staged_offload
        self.pipeline = None

    def _ensure_trellis_imports(self) -> None:
        try:
            import trellis  # noqa: F401
        except ImportError:
            # SEC-16: append (never prepend) so a planted module in a
            # world-writable clone dir cannot shadow a real dependency.
            extend_sys_path(
                *[
                    c
                    for c in [
                        Path("/tmp/TRELLIS"),
                        Path("/kaggle/working/TRELLIS"),
                        Path(__file__).resolve().parents[2] / "TRELLIS",
                        Path(__file__).resolve().parents[3] / "TRELLIS",
                    ]
                    if (c / "trellis").is_dir()
                ]
            )

    def set_pipeline(self, pipeline, dev: str | None = None) -> None:
        """Attach an already-loaded pipeline, skipping ``load_model()``.

        ENG-28: replaced the untyped, uncalled ``attach()``. Useful for tests
        and for sharing one pipeline across engine instances.
        """
        if pipeline is None:
            raise ValueError("pipeline must not be None")
        self.pipeline = pipeline
        if dev is not None:
            self.dev = torch.device(dev)

    def load_model(self) -> None:
        self._ensure_trellis_imports()
        # ENG-29: set the TRELLIS backend variables here, immediately before the
        # import that reads them, rather than mutating os.environ at module scope.
        os.environ.setdefault("SPCONV_ALGO", "native")
        os.environ.setdefault("ATTN_BACKEND", "xformers")
        from huggingface_hub import snapshot_download
        from trellis.pipelines import TrellisImageTo3DPipeline

        print("[TRELLIS] Fetching microsoft/TRELLIS-image-large snapshot …")
        local_dir = snapshot_download(
            "microsoft/TRELLIS-image-large",
            revision="25e0d31ffbebe4b5a97464dd851910efc3002d96",  # pinned SHA
        )
        pipe = TrellisImageTo3DPipeline.from_pretrained(local_dir)

        if self.staged_offload:
            for name, mod in pipe.models.items():
                target = "cpu" if name in ("slat_decoder_gs", "slat_decoder_rf") else self.dev
                mod.to(target)
            self.pipeline = pipe
        else:
            moved = pipe.to(self.device)
            self.pipeline = moved if moved is not None else pipe

        free_cuda_cache()
        print("[TRELLIS] Pipeline loaded.")

    @torch.no_grad()
    def predict(self, image: Image.Image) -> ReconstructionResult:
        if self.pipeline is None:
            self.load_model()

        p = self.pipeline
        rgba = image.convert("RGBA")
        arr = np.array(rgba)
        # ENG-05: use the shared thresholded + speckle-filtered mask instead of
        # treating every pixel below 248 as foreground (JPEG noise became mask).
        if not alpha_is_usable(arr):
            fg = derive_foreground_mask(arr)
            if fg.any():
                arr[..., 3] = fg.astype(np.uint8) * 255
                rgba = Image.fromarray(arr, mode="RGBA")

        t0 = time.perf_counter()
        decode_mode = self.output_format

        # ENG-27: "radiance_field" is in the Literal but neither predict branch
        # implements it — it would fall through to the gaussian path and return
        # the wrong artefact. Reject it explicitly.
        if decode_mode not in ("mesh", "gaussian"):
            raise TrellisOutputError(
                f"output_format={decode_mode!r} is not implemented; use 'mesh' or 'gaussian'"
            )

        if self.staged_offload and self.output_format == "mesh":
            img = p.preprocess_image(rgba)
            cond = p.get_cond([img])
            torch.manual_seed(42)
            coords = p.sample_sparse_structure(
                cond, 1, {"steps": self.sparse_steps, "cfg_strength": self.sparse_cfg_strength}
            )
            slat = p.sample_slat(
                cond, coords, {"steps": self.slat_steps, "cfg_strength": self.slat_cfg_strength}
            )
            del cond, coords
            gc.collect()
            free_cuda_cache()

            offload_names = (
                "sparse_structure_flow_model",
                "slat_flow_model",
                "image_cond_model",
                "sparse_structure_decoder",
            )
            for n in offload_names:
                if n in p.models:
                    p.models[n].to("cpu")
            free_cuda_cache()
            gc.collect()

            out_gs_color = None
            # ENG-21: models that must stay on CPU after this block. Moving a
            # decoder back to the GPU right after it OOM'd would immediately
            # OOM again, and the original error would be masked.
            keep_on_cpu: set = set()
            try:
                out_mesh = p.models["slat_decoder_mesh"](slat)
                outputs = {"mesh": out_mesh}
                if "slat_decoder_gs" in p.models:
                    try:
                        p.models["slat_decoder_mesh"].to("cpu")
                        free_cuda_cache()
                        p.models["slat_decoder_gs"].to(self.dev)
                        with torch.cuda.device(self.dev):
                            out_gs_color = p.models["slat_decoder_gs"](slat)
                    except Exception as gs_err:
                        print(f"[TRELLIS] GS color decoder fallback ({gs_err})")
                    finally:
                        p.models["slat_decoder_gs"].to("cpu")
                        p.models["slat_decoder_mesh"].to(self.dev)
                        free_cuda_cache()
            except torch.OutOfMemoryError as mesh_oom:
                print("[TRELLIS] mesh decode OOM -> gaussian fallback")
                # the mesh decoder is the thing that just failed: keep it off the GPU
                keep_on_cpu.add("slat_decoder_mesh")
                if "slat_decoder_mesh" in p.models:
                    try:
                        p.models["slat_decoder_mesh"].to("cpu")
                    except Exception:
                        pass
                free_cuda_cache()
                gc.collect()
                try:
                    p.models["slat_decoder_gs"].to(self.dev)
                    with torch.cuda.device(self.dev):
                        out_gs = p.models["slat_decoder_gs"](slat)
                    p.models["slat_decoder_gs"].to("cpu")
                    outputs = {"gaussian": out_gs}
                    decode_mode = "gaussian"
                except Exception as gs_oom:
                    # ENG-21: do not let the fallback failure mask the original.
                    raise RuntimeError(
                        f"TRELLIS gaussian fallback failed after a mesh-decode OOM: {gs_oom}"
                    ) from mesh_oom
            finally:
                for n in offload_names:
                    if n in p.models and n not in keep_on_cpu:
                        p.models[n].to(self.dev)
                free_cuda_cache()
        else:
            outputs = p.run(
                rgba,
                seed=42,
                formats=[self.output_format],
                sparse_structure_sampler_params={
                    "steps": self.sparse_steps,
                    "cfg_strength": self.sparse_cfg_strength,
                },
                slat_sampler_params={
                    "steps": self.slat_steps,
                    "cfg_strength": self.slat_cfg_strength,
                },
                preprocess_image=True,
            )
            out_gs_color = outputs.get("gaussian")

        latency = time.perf_counter() - t0
        print(f"[TRELLIS] Inference done in {latency:.2f}s (mode={decode_mode})")

        metadata = {
            "engine": "TRELLIS",
            "output_format": decode_mode,
            "latency_s": round(latency, 3),
            "slat_steps": self.slat_steps,
            "sparse_steps": self.sparse_steps,
            "sparse_cfg_strength": self.sparse_cfg_strength,
            "slat_cfg_strength": self.slat_cfg_strength,
        }

        if decode_mode == "mesh":
            # ENG-22: outputs["mesh"] is indexed without checking the key or the
            # returned type, so a gaussian-only response raised KeyError.
            raw = _first_of(outputs.get("mesh"), "mesh")
            if raw is None:
                raise TrellisOutputError(
                    f"TRELLIS returned no 'mesh' entry (keys: {sorted(outputs)})"
                )
            verts_raw = raw.vertices.detach().cpu().numpy().astype(np.float32)
            faces = raw.faces.detach().cpu().numpy().astype(np.int32)

            vcols = None
            attrs = getattr(raw, "vertex_attrs", None)
            if attrs is not None:
                try:
                    # ENG-23: detect the decoder's colour range instead of
                    # assuming [0, 1], and record what was found.
                    vcols, color_range = _normalize_vertex_colors(attrs.detach().cpu().numpy())
                    metadata["vertex_color_range"] = color_range
                except Exception as ce:
                    print(f"[TRELLIS] vertex color warning: {ce}")

            # Transfer high-fidelity appearance colors from SLatGaussianDecoder in the shared
            # pre-rotated [-0.5, 0.5]^3 coordinate frame via opacity-weighted 3D k-NN.
            if out_gs_color is not None and len(out_gs_color) > 0 and len(verts_raw) > 0:
                try:
                    from scipy.spatial import cKDTree

                    gs0 = out_gs_color[0]
                    gs_xyz = gs0.get_xyz.detach().cpu().numpy().astype(np.float32)
                    gs_opa = gs0.get_opacity.detach().cpu().numpy().astype(np.float32).reshape(-1)
                    # Store raw SH DC (consistent with PLY exporter convention)
                    gs_rgb = np.clip(
                        gs0._features_dc.detach().cpu().numpy().astype(np.float32)[:, 0, :3],
                        -1.0,
                        1.0,
                    )
                    mask = gs_opa > 0.10
                    if int(mask.sum()) >= 64:
                        gs_xyz, gs_opa, gs_rgb = gs_xyz[mask], gs_opa[mask], gs_rgb[mask]
                    tree = cKDTree(gs_xyz)
                    k = min(8, len(gs_xyz))
                    dists, idx = tree.query(verts_raw, k=k)
                    if k == 1 or idx.ndim == 1:
                        gs_vcols = gs_rgb[idx]
                    else:
                        w = (gs_opa[idx] ** 2) / (dists**2 + 1e-6)
                        w /= w.sum(axis=1, keepdims=True)
                        gs_vcols = (gs_rgb[idx] * w[..., None]).sum(axis=1).astype(np.float32)
                    vcols = np.clip(gs_vcols, 0.0, 1.0).astype(np.float32)
                except Exception as gce:
                    print(f"[TRELLIS] GS color transfer warning: {gce}")

            # Convert Blender Z-up (x, y, z) -> OpenGL Y-up (x, z, -y) via Rx(-90)
            rx_neg90 = np.array(
                [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float32
            )
            verts = verts_raw @ rx_neg90.T

            mesh = Mesh3D(vertices=verts, faces=faces, vertex_colors=vcols)
            # ENG-42: validate at the source rather than in export/rendering.
            mesh.validate()
            metadata.update(
                {
                    "vertex_count": len(mesh.vertices),
                    "face_count": len(mesh.faces),
                    "has_vertex_colors": vcols is not None,
                }
            )
            free_cuda_cache()
            gc.collect()
            return ReconstructionResult(mesh=mesh, metadata=metadata)

        gs = outputs["gaussian"][0]
        gaussians = GaussianCloud(
            positions=gs.get_xyz.detach().cpu().numpy().astype(np.float32),
            scales=gs.get_scaling.detach().cpu().numpy().astype(np.float32),
            rotations=gs.get_rotation.detach().cpu().numpy().astype(np.float32),
            opacities=gs.get_opacity.detach().cpu().numpy().astype(np.float32),
            sh_coeffs=gs.get_features[:, 0, :3].detach().cpu().numpy().astype(np.float32),
        )
        metadata.update({"gaussian_count": len(gaussians.positions)})
        # ENG-42: validate at the source rather than in export/rendering.
        gaussians.validate()
        free_cuda_cache()
        gc.collect()
        return ReconstructionResult(gaussians=gaussians, metadata=metadata)

    def unload_model(self) -> None:
        """ENG-09: TRELLIS keeps staged CPU/GPU offload state; drop it all."""
        import gc

        super().unload_model()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def get_engine_info(self) -> dict[str, Any]:
        return {
            "name": "TRELLIS",
            "paper": "arXiv:2412.01506",
            "hf_repo": "microsoft/TRELLIS-image-large",
            "output_formats": ["mesh (FlexiCubes @ 256^3)", "gaussian (SLAT)"],
        }
