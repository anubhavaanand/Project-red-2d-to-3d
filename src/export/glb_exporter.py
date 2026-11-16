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

    # Prepare vertex colors if available
    vertex_colors = None
    if mesh.vertex_colors is not None:
        vc = np.asarray(mesh.vertex_colors, dtype=np.float32)
        if vc.max() <= 1.05:
            vc = (vc * 255).astype(np.uint8)
        else:
            vc = np.clip(vc, 0, 255).astype(np.uint8)
        vertex_colors = vc

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
