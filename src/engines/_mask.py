"""Shared foreground-mask derivation for the engine wrappers.

ENG-05: four engines each had their own inline heuristic for "the alpha channel
is missing or useless, guess the foreground from the background colour":

    if arr[..., 3].min() > 250:              # triposr / trellis / spar3d
        non_white = np.any(arr[..., :3] < 248, axis=-1)

That treats any pixel below 248 as foreground, so JPEG noise speckles the mask
and a white object on a white background loses its mask entirely.

This module provides one thresholded, speckle-filtered implementation.
"""

from __future__ import annotations

import numpy as np

# Alpha at or above this is treated as "fully opaque" — i.e. no usable alpha.
ALPHA_OPAQUE = 250
# Alpha above this is foreground.
ALPHA_FOREGROUND = 10
# A pixel brighter than this in every channel is background (white studio bg).
WHITE_BACKGROUND_LEVEL = 248
# Connected components smaller than this are discarded as speckle.
MIN_COMPONENT_PX = 64
# ...and so are components below this fraction of the largest component, which
# keeps genuinely separate parts (a mug and its handle) while dropping noise.
MIN_COMPONENT_FRACTION = 0.01


def alpha_is_usable(arr: np.ndarray) -> bool:
    """True when the alpha channel carries real transparency information."""
    return np.asarray(arr)[..., 3].min() < ALPHA_OPAQUE


def derive_foreground_mask(rgb: np.ndarray) -> np.ndarray:
    """Return a boolean foreground mask for a white-background image.

    Pixels are foreground when they are not near-white, then small connected
    components are dropped so JPEG speckle does not become a mask.
    """
    rgb = np.asarray(rgb)[..., :3]
    mask = np.any(rgb < WHITE_BACKGROUND_LEVEL, axis=-1)
    return _drop_speckle(mask)


def _drop_speckle(mask: np.ndarray) -> np.ndarray:
    """Remove small connected components from a boolean mask."""
    if not mask.any():
        return mask

    try:
        from scipy.ndimage import label
    except ImportError:  # pragma: no cover - scipy is a declared dependency
        return mask

    labeled, count = label(mask)
    if count <= 1:
        return mask

    sizes = np.bincount(labeled.ravel(), minlength=count + 1)
    sizes[0] = 0  # label 0 is the background
    largest = int(sizes.max())
    if largest == 0:
        return mask

    keep = sizes >= max(MIN_COMPONENT_PX, int(largest * MIN_COMPONENT_FRACTION))
    keep[0] = False
    if keep.sum() == count:
        return mask  # nothing would be removed
    return keep[labeled]
