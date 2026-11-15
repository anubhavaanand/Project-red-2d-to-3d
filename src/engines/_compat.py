"""Scoped compatibility shims for third-party model repositories.

SPAR3D and LGM vendor code written against APIs that are absent from the
pinned dependency set: ``pkg_resources``, ``xformers.ops`` and a few
``transformers`` helpers. Without a shim the import fails; with one installed
into ``sys.modules`` or onto a class it becomes process-global, so every module
imported afterwards — including unrelated libraries — sees the fake.

Each helper here is a context manager that installs the shim for the duration
of a block and restores the previous state on exit, so a failed or successful
model load never leaves a half-fake API behind.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from contextlib import contextmanager
from typing import Iterator

_MISSING = object()


@contextmanager
def lgm_xformers_shim() -> Iterator[None]:
    """Expose ``xformers.ops`` backed by PyTorch SDPA while the block runs.

    Installed only when a real ``xformers`` with ``memory_efficient_attention``
    is unavailable. The fake modules are removed again on exit.
    """
    if importlib.util.find_spec("xformers") is not None:
        try:
            import xformers.ops as real_ops

            if hasattr(real_ops, "memory_efficient_attention"):
                yield
                return
        except Exception:
            pass

    import torch
    import torch.nn.functional as F

    def _memory_efficient_attention(q, k, v, attn_bias=None, op=None, **_kw):
        # xformers' attn_bias is not a dense mask, so it cannot be translated to
        # PyTorch's attn_mask. Reject rather than silently ignore it.
        if attn_bias is not None:
            raise NotImplementedError(
                "xformers attention bias is not supported in the PyTorch SDPA shim. "
                "Use dense boolean masks instead."
            )
        if q.ndim == 3:
            return F.scaled_dot_product_attention(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)).squeeze(1)
        # xformers 4D tensors are (B, N, H, D); PyTorch SDPA wants (B, H, N, D)
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2).contiguous()

    def _unbind(tensor, dim: int = 0):
        return torch.unbind(tensor, dim=dim)

    saved = {k: v for k, v in sys.modules.items() if k == "xformers" or k.startswith("xformers.")}

    xformers = types.ModuleType("xformers")
    ops = types.ModuleType("xformers.ops")
    ops.memory_efficient_attention = _memory_efficient_attention
    ops.unbind = _unbind
    xformers.ops = ops
    # diffusers calls importlib.util.find_spec("xformers") at import time, which
    # raises on a fake module without __spec__.
    xformers.__spec__ = importlib.machinery.ModuleSpec("xformers", None, is_package=True)
    xformers.__path__ = []
    ops.__spec__ = importlib.machinery.ModuleSpec("xformers.ops", None)

    sys.modules["xformers"] = xformers
    sys.modules["xformers.ops"] = ops
    try:
        yield
    finally:
        for name in [k for k in sys.modules if k == "xformers" or k.startswith("xformers.")]:
            del sys.modules[name]
        sys.modules.update(saved)


@contextmanager
def spar3d_compat_shims() -> Iterator[None]:
    """Install SPAR3D's compatibility shims for the duration of the block.

    Covers a missing ``pkg_resources.packaging`` and three ``transformers``
    helpers that were removed or renamed in newer releases. Everything the shim
    added is undone on exit, including a fake ``pkg_resources`` module.
    """
    import packaging

    installed_pkg_resources = False
    patched: list[tuple[object, str, object]] = []

    def _patch(obj, name, value):
        patched.append((obj, name, getattr(obj, name, _MISSING)))
        setattr(obj, name, value)

    try:
        # SPAR3D's vendored code reads pkg_resources.packaging. Prefer the real
        # setuptools module; only fabricate one when it is genuinely absent, and
        # drop the fake on exit so a later import gets the real thing.
        if "pkg_resources" not in sys.modules:
            try:
                import pkg_resources  # noqa: F401
            except ImportError:
                fake = types.ModuleType("pkg_resources")
                fake.packaging = packaging
                sys.modules["pkg_resources"] = fake
                installed_pkg_resources = True

        try:
            import torch
            import transformers.pytorch_utils as tpu

            if not hasattr(tpu, "find_pruneable_heads_and_indices"):

                def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
                    heads = set(heads) - already_pruned_heads
                    mask = torch.ones(n_heads, head_size)
                    for head in heads:
                        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                        mask[head] = 0
                    mask = mask.view(-1).contiguous().eq(1)
                    index = torch.arange(len(mask))[mask].long()
                    return heads, index

                _patch(tpu, "find_pruneable_heads_and_indices", _find_pruneable_heads_and_indices)

            if not hasattr(tpu, "prune_linear_layer"):

                def _prune_linear_layer(layer, index, dim=0):
                    index = index.to(layer.weight.device)
                    weight = layer.weight.index_select(dim, index).clone().detach()
                    bias = (
                        (layer.bias.clone().detach() if dim == 1 else layer.bias[index].clone().detach())
                        if layer.bias is not None
                        else None
                    )
                    new_size = list(layer.weight.size())
                    new_size[dim] = len(index)
                    new_layer = torch.nn.Linear(
                        new_size[1], new_size[0], bias=layer.bias is not None
                    ).to(layer.weight.device)
                    new_layer.weight.requires_grad = False
                    new_layer.weight.copy_(weight.contiguous())
                    new_layer.weight.requires_grad = True
                    if layer.bias is not None:
                        new_layer.bias.requires_grad = False
                        new_layer.bias.copy_(bias.contiguous())
                        new_layer.bias.requires_grad = True
                    return new_layer

                _patch(tpu, "prune_linear_layer", _prune_linear_layer)

            from transformers.modeling_utils import PreTrainedModel

            if not hasattr(PreTrainedModel, "get_head_mask"):

                def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
                    if head_mask is not None:
                        head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
                        if is_attention_chunked is True:
                            head_mask = head_mask.unsqueeze(-1)
                    else:
                        head_mask = [None] * num_hidden_layers
                    return head_mask

                _patch(PreTrainedModel, "get_head_mask", _get_head_mask)
        except Exception:
            # Missing/incompatible transformers is not fatal here: SPAR3D only
            # needs the shims when its vendored imports reach them.
            pass

        # SECURITY: never fabricate transparent_background — silently swapping in
        # a no-op would change what SPAR3D considers foreground.
        if "transparent_background" not in sys.modules:
            try:
                import transparent_background  # noqa: F401
            except ImportError:
                raise ImportError(
                    "transparent_background is required for SPAR3D. "
                    "Install it with: pip install transparent-background"
                ) from None

        yield
    finally:
        for obj, name, previous in reversed(patched):
            if previous is _MISSING:
                if getattr(obj, name, _MISSING) is not _MISSING:
                    delattr(obj, name)
            else:
                setattr(obj, name, previous)
        if installed_pkg_resources:
            sys.modules.pop("pkg_resources", None)