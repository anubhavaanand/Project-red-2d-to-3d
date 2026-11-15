"""Shared helpers for the engine wrappers."""

from __future__ import annotations


def free_cuda_cache() -> bool:
    """Release cached CUDA blocks, if CUDA is actually available.

    ENG-08: engines called ``torch.cuda.empty_cache()`` unconditionally. On a
    CPU-only torch build that is at best a no-op and at worst an error, and it
    reads as if GPU memory was freed when it was not. Returns whether the call
    did anything, so callers can log honestly.
    """
    try:
        import torch
    except ImportError:
        return False

    if not torch.cuda.is_available():
        return False

    torch.cuda.empty_cache()
    return True