from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
from PIL import Image

@dataclass
class Mesh3D:
    vertices: np.ndarray
    faces: np.ndarray
    vertex_colors: Optional[np.ndarray] = None
    uv_coords: Optional[np.ndarray] = None
    texture_map: Optional[Image.Image] = None

    def validate(self) -> None:
        """Raise ValueError if this mesh is structurally inconsistent.

        ENG-42: the dataclass accepted any arrays, so mismatched lengths or
        non-finite coordinates failed later, far from the cause.
        """
        v = np.asarray(self.vertices)
        f = np.asarray(self.faces)

        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError(f"vertices must be (N, 3), got {v.shape}")
        if f.ndim != 2 or f.shape[1] != 3:
            raise ValueError(f"faces must be (M, 3), got {f.shape}")
        if len(v) == 0:
            raise ValueError("mesh has no vertices")
        if len(f) and int(f.max()) >= len(v):
            raise ValueError(
                f"face index {int(f.max())} out of range for {len(v)} vertices"
            )
        if not np.isfinite(v).all():
            raise ValueError("vertices contain non-finite values")

        if self.vertex_colors is not None:
            vc = np.asarray(self.vertex_colors)
            if len(vc) != len(v):
                raise ValueError(
                    f"vertex_colors length {len(vc)} != vertices length {len(v)}"
                )
            if vc.ndim != 2 or vc.shape[1] < 3:
                raise ValueError(f"vertex_colors must be (N, >=3), got {vc.shape}")
            if not np.isfinite(vc).all():
                raise ValueError("vertex_colors contain non-finite values")

        if self.uv_coords is not None:
            uv = np.asarray(self.uv_coords)
            if len(uv) != len(v):
                raise ValueError(
                    f"uv_coords length {len(uv)} != vertices length {len(v)}"
                )


@dataclass
class GaussianCloud:
    positions: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    opacities: np.ndarray
    sh_coeffs: np.ndarray
    sh_convention: str = "sh_dc"  # "sh_dc" (raw SH DC) or "rgb" (display RGB 0..1)

    def validate(self) -> None:
        """Raise ValueError if this cloud is structurally inconsistent (ENG-42)."""
        arrays = {
            "positions": np.asarray(self.positions),
            "scales": np.asarray(self.scales),
            "rotations": np.asarray(self.rotations),
            "opacities": np.asarray(self.opacities).reshape(-1),
            "sh_coeffs": np.asarray(self.sh_coeffs),
        }

        n = len(arrays["positions"])
        if n == 0:
            raise ValueError("gaussian cloud has no points")

        for name, arr in arrays.items():
            if name in ("sh_coeffs",):
                continue
            if len(arr) != n:
                raise ValueError(
                    f"{name} length {len(arr)} != positions length {n}"
                )
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} contains non-finite values")

        if not np.isfinite(arrays["sh_coeffs"]).all():
            raise ValueError("sh_coeffs contains non-finite values")

@dataclass
class ReconstructionResult:
    mesh: Optional[Mesh3D] = None
    gaussians: Optional[GaussianCloud] = None
    metadata: Optional[Dict[str, Any]] = None

class AbstractEngine(ABC):
    @abstractmethod
    def load_model(self) -> None:
        pass
        
    @abstractmethod
    def predict(self, image: Image.Image) -> ReconstructionResult:
        pass
        
    @abstractmethod
    def get_engine_info(self) -> Dict[str, Any]:
        pass

    def unload_model(self) -> None:
        """Release the model and any GPU memory it holds.

        ENG-09: BenchmarkRunner calls this between engines when present, but
        engines did not implement it, so models stayed resident in VRAM/RAM
        and long runs OOMed. Subclasses with heavier state should override,
        free CUDA explicitly, then call ``super().unload_model()``.
        """
        for name in list(vars(self)):
            # Keep the scalar configuration; drop anything holding model state.
            if name in ("device", "dev", "output_format", "chunk_size", "resolution",
                        "texture_resolution", "low_vram_mode", "foreground_ratio",
                        "mv_guidance_scale", "mv_num_inference_steps", "elevation",
                        "slat_steps", "sparse_steps", "sparse_cfg_strength",
                        "slat_cfg_strength", "cfg_strength", "staged_offload"):
                continue
            setattr(self, name, None)
