from __future__ import annotations

from pathlib import Path

import numpy as np

from ..engines.base import GaussianCloud
from ._validate import require_finite


def export_ply(gaussians: GaussianCloud, filepath: str) -> None:
    """
    Exports a GaussianCloud to a standard 3DGS PLY format.
    Compatible with 3DGS visualizers (e.g. SuperSplat, MeshLab, WebGL 3DGS).
    """
    if gaussians is None:
        raise ValueError("GaussianCloud is None, cannot export.")

    pos = np.asarray(gaussians.positions, dtype=np.float32)
    n = len(pos)

    # EXP-04: fail fast on non-finite splats rather than writing a broken file.
    require_finite("positions", gaussians.positions)
    require_finite("scales", gaussians.scales)
    require_finite("rotations", gaussians.rotations)
    require_finite("opacities", gaussians.opacities)
    require_finite("sh_coeffs", gaussians.sh_coeffs)

    # Scale attributes
    if gaussians.scales is not None:
        scales = np.asarray(gaussians.scales, dtype=np.float32)
        if scales.ndim == 1:
            scales = np.tile(scales[:, None], (1, 3))
    else:
        scales = np.zeros((n, 3), dtype=np.float32)

    # Rotation attributes (quaternion)
    if gaussians.rotations is not None:
        rot = np.asarray(gaussians.rotations, dtype=np.float32)
    else:
        rot = np.zeros((n, 4), dtype=np.float32)

    # Opacity
    if gaussians.opacities is not None:
        opacities = np.asarray(gaussians.opacities, dtype=np.float32)
        if opacities.ndim == 1:
            opacities = opacities[:, None]
    else:
        opacities = np.ones((n, 1), dtype=np.float32)

    # SH DC colors
    if gaussians.sh_coeffs is not None:
        sh = np.asarray(gaussians.sh_coeffs, dtype=np.float32)
        if sh.ndim == 3:
            sh = sh[:, 0, :3]
        # Convert display RGB to SH DC if needed
        if gaussians.sh_convention == "rgb":
            # RGB (0..1) -> SH DC: (rgb - 0.5) / 0.28209479
            sh = (sh - 0.5) / 0.28209479177387814
    else:
        sh = np.zeros((n, 3), dtype=np.float32)

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header\n",
    ]

    # EXP-10: build the body with numpy and write once. One f.write per
    # Gaussian took seconds for 65k-500k splats and blocked the API loop.
    normals = np.zeros((n, 3), dtype=np.float32)
    rows = np.concatenate([pos, normals, sh, opacities.reshape(n, 1), scales, rot], axis=1)
    with open(filepath, "w") as f:
        f.write("\n".join(header))
        np.savetxt(f, rows, fmt="%.6f", delimiter=" ")
