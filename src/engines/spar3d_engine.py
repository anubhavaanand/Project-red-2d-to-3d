"""
SPAR3D Engine — audited wrapper for Stability-AI/stable-point-aware-3d (arXiv:2501.04689).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from PIL import Image

from .base import AbstractEngine, ReconstructionResult, Mesh3D
from ._compat import spar3d_compat_shims
from ._paths import extend_sys_path
from ._memory import free_cuda_cache
from ._mask import alpha_is_usable, derive_foreground_mask
from ._outputs import EngineOutputError, extract_mesh_object


class Spar3dOutputError(EngineOutputError):
    """ENG-36: raised when SPAR3D returns a payload we cannot read a mesh from."""


def _extract_mesh_object(mesh_out):
    """Return SPAR3D's mesh object, validating the payload (ENG-36)."""
    return extract_mesh_object(mesh_out, engine="SPAR3D")


def _to_rgb_array(tex_img) -> np.ndarray | None:
    """Return a texture as float RGB in [0, 1], whatever container it arrives in.

    ENG-35: the code called tex_img.convert("RGB") unconditionally, which raises
    AttributeError for a NumPy texture and silently skipped colour extraction.
    """
    if tex_img is None:
        return None

    # PIL image
    if hasattr(tex_img, "convert"):
        try:
            return np.asarray(tex_img.convert("RGB"), dtype=np.float32) / 255.0
        except Exception:
            return None

    # NumPy array / array-like
    arr = np.asarray(tex_img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None

    rgb = arr[..., :3].astype(np.float32)
    if rgb.size and float(np.nanmax(rgb)) > 1.5:
        rgb = rgb / 255.0
    return rgb


def _ensure_outward_normals(tm) -> str:
    """Make ``tm`` face outward, reporting what was done.

    ENG-33: the code called ``tm.invert()`` inside a bare ``try/except``. If
    upstream stopped inverting, that would flip correct normals; if it raised,
    the failure vanished. Use the signed volume instead: a negative volume means
    inward-facing winding, which ``fix_normals()`` repairs.
    """
    try:
        volume = float(tm.volume)
    except Exception as e:
        return f"volume unavailable ({type(e).__name__})"

    if volume < 0:
        try:
            tm.fix_normals()
            return "inverted (negative volume) -> fixed"
        except Exception as e:
            return f"inverted (negative volume), fix failed ({type(e).__name__})"

    return "already outward"


def _vertex_colors_from_to_color(vis, n_verts: int) -> np.ndarray | None:
    """Read vertex colours via ``vis.to_color()``, tolerating its absence.

    ENG-34: ``vis.to_color()`` may not exist on every mesh type. Probe for it and
    skip (returning None) rather than letting the AttributeError propagate to the
    outer handler and silently discard the colours.
    """
    to_color = getattr(vis, "to_color", None)
    if not callable(to_color):
        print("[SPAR3D] vertex colors: visual has no to_color(); skipping")
        return None

    vc_vis = to_color()
    if getattr(vc_vis, "vertex_colors", None) is None:
        return None

    vc = np.asarray(vc_vis.vertex_colors, dtype=np.float32)
    if len(vc) != n_verts:
        return None
    return vc[:, :3] / (255.0 if vc.max() > 1.05 else 1.0)


class Spar3dEngine(AbstractEngine):
    def __init__(
        self,
        device: str = "cuda",
        texture_resolution: int = 1024,
        low_vram_mode: bool = False,
        foreground_ratio: float = 1.3,
    ):
        self.device = device
        self.texture_resolution = texture_resolution
        self.low_vram_mode = low_vram_mode
        self.foreground_ratio = foreground_ratio
        self.model = None
        self._foreground_crop = None

    def _locate_repo(self) -> Path:
        for c in [
            Path("/tmp/stable-point-aware-3d"),
            Path("/kaggle/working/stable-point-aware-3d"),
            Path(__file__).resolve().parents[2] / "stable-point-aware-3d",
            Path(__file__).resolve().parents[3] / "stable-point-aware-3d",
            Path(__file__).resolve().parents[3] / "SPAR3D",
        ]:
            if (c / "spar3d").is_dir():
                return c
        raise RuntimeError(
            "stable-point-aware-3d not found. Clone https://github.com/Stability-AI/stable-point-aware-3d."
        )

    def load_model(self) -> None:
        # SEC-15: shims are scoped to this block and undone on exit.
        with spar3d_compat_shims():
            repo_dir = self._locate_repo()
            # SEC-16: append (never prepend) so a planted module in a
            # world-writable clone dir cannot shadow a real dependency.
            extend_sys_path(
                *[
                    p
                    for p in [
                        repo_dir,
                        Path("/tmp/CLIP"),
                        Path("/tmp/AlphaCLIP"),
                        Path("/kaggle/working/CLIP"),
                        Path("/kaggle/working/AlphaCLIP"),
                    ]
                    if p.is_dir()
                ]
            )

            from spar3d.system import SPAR3D
            from spar3d.utils import foreground_crop

            print(f"[SPAR3D] Loading model (low_vram_mode={self.low_vram_mode}) …")
            try:
                self.model = SPAR3D.from_pretrained(
                    "stabilityai/stable-point-aware-3d",
                    config_name="config.yaml",
                    weight_name="model.safetensors",
                    low_vram_mode=self.low_vram_mode,
                    revision="5699918cb34f55cd7d828493d2725f3038313761",  # pinned SHA
                )
            except Exception as e:
                # SECURITY: Only fall back to mirror if explicitly allowed
                import os
                if os.environ.get("SPAR3D_ALLOW_MIRROR", "0") == "1":
                    print(f"[SPAR3D] WARNING: Primary repo failed ({e}). Falling back to third-party mirror (zimengxiong/Modelr-SPAR3D)")
                    print("[SPAR3D] WARNING: Third-party code will be executed. Set SPAR3D_ALLOW_MIRROR=0 to disable.")
                    self.model = SPAR3D.from_pretrained(
                        "zimengxiong/Modelr-SPAR3D",
                        config_name="config.yaml",
                        weight_name="model.safetensors",
                        low_vram_mode=self.low_vram_mode,
                        revision="af033552e8c7d25971c4fca69ab4c926f43b2ec2",  # pinned SHA
                    )
                else:
                    raise RuntimeError(
                        f"SPAR3D primary repo failed ({e}). "
                        "Set SPAR3D_ALLOW_MIRROR=1 to allow fallback to third-party mirror."
                    ) from e
            self.model.to(self.device).eval()
            self._foreground_crop = foreground_crop
            print("[SPAR3D] Model loaded.")

    @staticmethod
    def _extract_spar3d_vertex_colors(tm, verts: np.ndarray) -> np.ndarray | None:
        """
        Extract per-vertex RGB colors from SPAR3D's baked PBRMaterial.baseColorTexture
        and repair any unbaked black UV-seam texels via 3D k-NN interpolation.
        """
        try:
            # ENG-45: scipy is only needed when baking vertex colours.
            from scipy.spatial import cKDTree

            vis = getattr(tm, "visual", None)
            if vis is None:
                return None
            uv = getattr(vis, "uv", None)
            mat = getattr(vis, "material", None)
            tex_img = None
            if mat is not None:
                tex_img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)

            vcols = None
            if uv is not None and tex_img is not None and len(uv) == len(verts):
                rgb = _to_rgb_array(tex_img)
                if rgb is None:
                    return None
                h, w = rgb.shape[:2]
                u = np.clip(np.nan_to_num(np.asarray(uv[:, 0], dtype=np.float64), nan=0.5), 0.0, 1.0)
                v = np.clip(np.nan_to_num(np.asarray(uv[:, 1], dtype=np.float64), nan=0.5), 0.0, 1.0)
                # SPAR3D's baker_kernel.cu writes UV v = 1.0 - row/height (OpenGL bottom-left origin).
                # Bilinearly sample at x = u * (w - 1), y = (1.0 - v) * (h - 1).
                fx = u * (w - 1)
                fy = (1.0 - v) * (h - 1)
                x0 = np.clip(np.floor(fx).astype(int), 0, w - 1)
                x1 = np.clip(x0 + 1, 0, w - 1)
                y0 = np.clip(np.floor(fy).astype(int), 0, h - 1)
                y1 = np.clip(y0 + 1, 0, h - 1)
                wx = (fx - x0)[..., None].astype(np.float32)
                wy = (fy - y0)[..., None].astype(np.float32)
                c00 = rgb[y0, x0]
                c10 = rgb[y0, x1]
                c01 = rgb[y1, x0]
                c11 = rgb[y1, x1]
                vcols = np.clip(
                    (1.0 - wy) * ((1.0 - wx) * c00 + wx * c10)
                    + wy * ((1.0 - wx) * c01 + wx * c11),
                    0.0,
                    1.0,
                ).astype(np.float32)
            else:
                # ENG-34: vis.to_color() may not exist on this mesh type.
                vcols = _vertex_colors_from_to_color(vis, len(verts))

            if vcols is not None and len(vcols) == len(verts):
                # Repair unbaked black UV chart padding texels (brightness < 0.04) via 3D k-NN
                valid = vcols.max(axis=-1) >= 0.04
                if valid.any() and (~valid).any():
                    tree = cKDTree(verts[valid])
                    _, idx = tree.query(verts[~valid], k=min(4, int(valid.sum())))
                    if idx.ndim == 1:
                        vcols[~valid] = vcols[valid][idx]
                    else:
                        vcols[~valid] = vcols[valid][idx].mean(axis=1)
            return vcols
        except Exception as e:
            print(f"[SPAR3D] Vertex color extraction warning: {e}")
            return None

    @torch.no_grad()
    def predict(self, image: Image.Image) -> ReconstructionResult:
        if self.model is None:
            self.load_model()

        t0 = time.perf_counter()

        rgba = image.convert("RGBA")
        arr = np.array(rgba)
        # ENG-05: use the shared thresholded + speckle-filtered mask instead of
        # treating every pixel below 248 as foreground (JPEG noise became mask).
        if not alpha_is_usable(arr):
            fg = derive_foreground_mask(arr)
            if fg.any():
                arr[..., 3] = fg.astype(np.uint8) * 255
                rgba = Image.fromarray(arr, mode="RGBA")

        # ENG-32: a failed crop was printed and ignored, so an uncropped image
        # silently degraded results with no structured signal. Record it.
        crop_error = None
        try:
            rgba = self._foreground_crop(
                rgba, crop_ratio=self.foreground_ratio, newsize=(512, 512)
            )
        except Exception as crop_err:
            crop_error = f"{type(crop_err).__name__}: {crop_err}"
            print(f"[SPAR3D] foreground_crop failed, using uncropped input ({crop_error})")

        torch.manual_seed(42)
        np.random.seed(42)

        mesh_out, _ = self.model.run_image(
            rgba,
            bake_resolution=self.texture_resolution,
            remesh="none",
            estimate_illumination=False,
        )
        latency = time.perf_counter() - t0

        tm = _extract_mesh_object(mesh_out)
        # SPAR3D's system.py:917 calls tmesh.invert(), so the mesh arrives with
        # inward-facing winding. ENG-33: rather than inverting blind inside a
        # bare except, check the signed volume and fix normals if needed, then
        # record what was done.
        normal_fix = _ensure_outward_normals(tm)

        verts = np.asarray(tm.vertices, dtype=np.float32)
        faces = np.asarray(tm.faces, dtype=np.int64)
        vcols = self._extract_spar3d_vertex_colors(tm, verts)

        uv_coords = None
        tex_map = None
        try:
            vis = getattr(tm, "visual", None)
            if vis is not None and getattr(vis, "uv", None) is not None:
                uv = np.asarray(vis.uv, dtype=np.float32)
                if len(uv) == len(verts):
                    uv_coords = uv
            mat = getattr(vis, "material", None) if vis is not None else None
            if mat is not None:
                img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
                if isinstance(img, Image.Image):
                    tex_map = img.copy()
                elif isinstance(img, np.ndarray):
                    tex_map = Image.fromarray(img)
        except Exception:
            pass

        # Note: spar3d/system.py:705-707 & 915 ALREADY applies output_rotation = Ry(90) @ Rx(-90)
        # to tmesh inside run_image(), placing vertices in OpenGL Y-up (+Y up, +Z front).
        # Do NOT apply a second Rx(-90) rotation here.

        print(f"[SPAR3D] Inference done in {latency:.2f}s ({len(verts)} verts)")

        mesh = Mesh3D(
            vertices=verts,
            faces=faces,
            vertex_colors=vcols,
            uv_coords=uv_coords,
            texture_map=tex_map,
        )
        # ENG-42: validate at the source rather than in export/rendering.
        mesh.validate()
        free_cuda_cache()

        return ReconstructionResult(
            mesh=mesh,
            metadata={
                "engine": "SPAR3D",
                "latency_s": round(latency, 3),
                "vertex_count": len(mesh.vertices),
                "face_count": len(mesh.faces),
                "has_vertex_colors": vcols is not None,
                "normal_fix": normal_fix,
                "crop_error": crop_error,
            },
        )

    def get_engine_info(self) -> Dict[str, Any]:
        return {
            "name": "SPAR3D",
            "paper": "arXiv:2501.04689",
            "hf_repo": "stabilityai/stable-point-aware-3d",
            "output_format": "mesh (Point Diffusion + DMTet + PBR)",
        }
