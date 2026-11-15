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

@dataclass
class GaussianCloud:
    positions: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    opacities: np.ndarray
    sh_coeffs: np.ndarray
    sh_convention: str = "sh_dc"  # "sh_dc" (raw SH DC) or "rgb" (display RGB 0..1)

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
