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


def require_matching_length(name: str, expected: int, *arrays) -> None:
    """Raise ValueError if any array's first axis is not ``expected``.

    EXP-11: the PLY writer concatenated every attribute array into one row
    matrix, so a single mismatched length raised ``ValueError`` *after* the
    header had already been written — leaving a truncated, unreadable file on
    disk. Validating up front turns that into a clean refusal before any I/O.

    Args:
        name: human-readable label used in the error message.
        expected: the required length (normally the number of positions).
        *arrays: arrays to check; ``None`` entries are skipped.
    """
    for i, arr in enumerate(arrays):
        if arr is None:
            continue
        a = np.asarray(arr)
        if a.ndim == 0:
            continue
        if len(a) != expected:
            label = name if len(arrays) == 1 else f"{name}[{i}]"
            raise ValueError(
                f"{label} has length {len(a)}, expected {expected}; refusing to export"
            )
