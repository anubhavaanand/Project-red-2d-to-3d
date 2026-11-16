"""Shared validation for exported geometry.

EXP-04: NaN/inf coordinates were written straight into GLB/PLY files without
checks, producing artefacts that viewers reject or render as garbage.
"""

from __future__ import annotations

import numpy as np


def require_finite(name: str, *arrays) -> None:
    """Raise ValueError if any array holds a non-finite value.

    Args:
        name: human-readable label used in the error message.
        *arrays: arrays to check; ``None`` entries are skipped.
    """
    for i, arr in enumerate(arrays):
        if arr is None:
            continue
        a = np.asarray(arr)
        if a.size == 0:
            continue
        if not np.isfinite(a).all():
            label = name if len(arrays) == 1 else f"{name}[{i}]"
            raise ValueError(f"{label} contains NaN or infinite values; refusing to export")
