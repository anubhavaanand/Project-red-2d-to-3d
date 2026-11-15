"""Engines module with lazy imports to avoid heavyweight engine loads at import time."""

import importlib
from typing import Any

from .base import AbstractEngine, GaussianCloud, Mesh3D, ReconstructionResult

__all__ = [
    "AbstractEngine",
    "Mesh3D",
    "GaussianCloud",
    "ReconstructionResult",
    "TriposrEngine",
    "LGMEngine",
    "TrellisEngine",
    "Spar3dEngine",
    "SF3DEngine",
]

# ENG-44: name -> (module, attribute). Resolved classes are cached so repeated
# attribute access does not re-import, and every access returns the same object.
_LAZY: dict[str, tuple] = {
    "TriposrEngine": ("triposr_engine", "TriposrEngine"),
    "LGMEngine": ("lgm_engine", "LGMEngine"),
    "TrellisEngine": ("trellis_engine", "TrellisEngine"),
    "Spar3dEngine": ("spar3d_engine", "Spar3dEngine"),
    "SF3DEngine": ("sf3d_engine", "SF3DEngine"),
}

_cache: dict[str, Any] = {}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from None

    if name not in _cache:
        module = importlib.import_module(f".{module_name}", __name__)
        _cache[name] = getattr(module, attr)
    return _cache[name]


def __dir__() -> list:
    return sorted(set(list(globals()) + list(_LAZY)))
