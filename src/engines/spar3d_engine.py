"""
SPAR3D Engine — audited wrapper for Stability-AI/stable-point-aware-3d (arXiv:2501.04689).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree

from .base import AbstractEngine, ReconstructionResult, Mesh3D
from ._compat import spar3d_compat_shims
from ._paths import extend_sys_path
from ._memory import free_cuda_cache


class Spar3dOutputError(RuntimeError):
    """ENG-36: raised when SPAR3D returns a payload we cannot read a mesh from."""


def _extract_mesh_object(mesh_out):
    """Return the mesh object from SPAR3D's run_image() output.

    ENG-36: the previous code did
    ``mesh_out[0] if isinstance(mesh_out, (list, tuple)) else mesh_out``, which
    silently accepted a dict or scene (indexing a dict by 0 raises KeyError, and
    a scene is not a mesh). Validate instead.
    """
    if isinstance(mesh_out, (list, tuple)):
        if not mesh_out:
            raise Spar3dOutputError("SPAR3D returned an empty sequence")
        return mesh_out[0]

    if isinstance(mesh_out, dict):
        raise Spar3dOutputError(
            f"SPAR3D returned a dict (keys: {sorted(mesh_out)}); expected a mesh"
        )

    if not hasattr(mesh_out, "vertices") or not hasattr(mesh_out, "faces"):
        raise Spar3dOutputError(
            f"SPAR3D returned {type(mesh_out).__name__}, which has no "
            "vertices/faces"
        )

    return mesh_out


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
                rgb = np.asarray(tex_img.convert("RGB"), dtype=np.float32) / 255.0
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
                vc_vis = vis.to_color()
                if getattr(vc_vis, "vertex_colors", None) is not None:
                    vc = np.asarray(vc_vis.vertex_colors, dtype=np.float32)
                    if len(vc) == len(verts):
                        vcols = vc[:, :3] / (255.0 if vc.max() > 1.05 else 1.0)

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
        if arr[..., 3].min() > 250:
            non_white = np.any(arr[..., :3] < 248, axis=-1)
            if non_white.any():
                arr[..., 3] = non_white.astype(np.uint8) * 255
                rgba = Image.fromarray(arr, mode="RGBA")

        try:
            rgba = self._foreground_crop(
                rgba, crop_ratio=self.foreground_ratio, newsize=(512, 512)
            )
        except Exception as crop_err:
            print(f"[SPAR3D] foreground_crop fallback ({crop_err})")

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
        # Undo spar3d/system.py:917's `tmesh.invert()` so triangle winding & normals face OUTWARD!
        try:
            tm.invert()
        except Exception:
            pass

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
        free_cuda_cache()

        return ReconstructionResult(
            mesh=mesh,
            metadata={
                "engine": "SPAR3D",
                "latency_s": round(latency, 3),
                "vertex_count": len(mesh.vertices),
                "face_count": len(mesh.faces),
                "has_vertex_colors": vcols is not None,
            },
        )

    def get_engine_info(self) -> Dict[str, Any]:
        return {
            "name": "SPAR3D",
            "paper": "arXiv:2501.04689",
            "hf_repo": "stabilityai/stable-point-aware-3d",
            "output_format": "mesh (Point Diffusion + DMTet + PBR)",
        }
