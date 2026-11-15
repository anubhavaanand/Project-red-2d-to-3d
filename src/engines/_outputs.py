"""Shared validation for engine output payloads."""

from __future__ import annotations


class EngineOutputError(RuntimeError):
    """Raised when an engine returns a payload this project cannot read."""


def extract_mesh_object(mesh_out, engine: str = "engine"):
    """Return the mesh object from an engine's ``run_image()`` output.

    ENG-36 / ENG-38: both SPAR3D and SF3D did ``mesh_out[0]`` (or a conditional
    variant) without checking the payload. A bare ``Trimesh`` return was indexed
    as a vertex array, and a dict raised ``KeyError``. Validate instead: accept
    a non-empty list/tuple, or a bare object exposing ``vertices`` and ``faces``.
    """
    if isinstance(mesh_out, (list, tuple)):
        if not mesh_out:
            raise EngineOutputError(f"{engine} returned an empty sequence")
        return mesh_out[0]

    if isinstance(mesh_out, dict):
        raise EngineOutputError(
            f"{engine} returned a dict (keys: {sorted(mesh_out)}); expected a mesh"
        )

    if not hasattr(mesh_out, "vertices") or not hasattr(mesh_out, "faces"):
        raise EngineOutputError(
            f"{engine} returned {type(mesh_out).__name__}, which has no vertices/faces"
        )

    return mesh_out
