import numpy as np
import trimesh

from ..engines.base import Mesh3D
from ._validate import require_finite


def export_glb(mesh: Mesh3D, filepath: str) -> None:
    """Exports a Mesh3D object to a GLB file."""
    if mesh is None:
        raise ValueError("Mesh3D is None, cannot export.")

    # EXP-04: fail fast on non-finite geometry rather than writing a broken file.
    require_finite("vertices", mesh.vertices)
    require_finite("faces", mesh.faces)
    if mesh.vertex_colors is not None:
        require_finite("vertex_colors", mesh.vertex_colors)

    # Prepare vertex colors if available.
    # EXP-05: decide by dtype, not by max(). A uint8 array holding only 0s and
    # 1s (a nearly-black mesh) has max() <= 1.05 and would be misread as float
    # [0,1], then scaled to near-white.
    vertex_colors = None
    if mesh.vertex_colors is not None:
        vc = np.asarray(mesh.vertex_colors)
        if np.issubdtype(vc.dtype, np.floating):
            rgb = np.clip(vc.astype(np.float32), 0.0, 1.0) * 255.0
        else:
            rgb = np.clip(vc, 0, 255)
        vertex_colors = rgb.astype(np.uint8)

    # Prepare visual with UV and texture if available
    visual = None
    if mesh.uv_coords is not None and mesh.texture_map is not None:
        visual = trimesh.visual.TextureVisuals(
            uv=np.asarray(mesh.uv_coords, dtype=np.float32), image=mesh.texture_map
        )
    elif vertex_colors is not None:
        visual = trimesh.visual.ColorVisuals(mesh=None, vertex_colors=vertex_colors)

    t_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, visual=visual, process=False)
    t_mesh.export(filepath, file_type="glb")
