"""Tests for the SPAR3D prune shim (ENG-30)."""

import ast
from pathlib import Path

import pytest

COMPAT = Path(__file__).parent.parent / "src" / "engines" / "_compat.py"

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402  (must follow importorskip)


def _prune_fn():
    """Compile _prune_linear_layer from _compat.py."""
    src = COMPAT.read_text()
    tree = ast.parse(src)
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_prune_linear_layer"
    )
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(COMPAT), "exec"), ns)
    return ns["_prune_linear_layer"]


class TestPruneLinearLayer:
    def test_pruned_shape(self):
        layer = nn.Linear(8, 4)
        out = _prune_fn()(layer, torch.tensor([0, 2]), dim=0)
        assert out.weight.shape == (2, 8)

    def test_values_are_selected(self):
        layer = nn.Linear(4, 2)
        with torch.no_grad():
            layer.weight.copy_(torch.arange(8, dtype=torch.float32).reshape(2, 4))
        out = _prune_fn()(layer, torch.tensor([1, 3]), dim=0)
        assert torch.allclose(out.weight, torch.tensor([[4.0, 5.0, 6.0, 7.0]]))

    def test_requires_grad_is_preserved_when_true(self):
        """ENG-30: the shim must not force requires_grad on or off."""
        layer = nn.Linear(4, 2)
        layer.weight.requires_grad = True
        out = _prune_fn()(layer, torch.tensor([0]), dim=0)
        assert out.weight.requires_grad is True

    def test_requires_grad_is_preserved_when_false(self):
        layer = nn.Linear(4, 2)
        layer.weight.requires_grad = False
        layer.bias.requires_grad = False
        out = _prune_fn()(layer, torch.tensor([0]), dim=0)
        assert out.weight.requires_grad is False
        assert out.bias.requires_grad is False

    def test_bias_requires_grad_is_preserved(self):
        layer = nn.Linear(4, 2)
        layer.bias.requires_grad = True
        out = _prune_fn()(layer, torch.tensor([0, 1]), dim=0)
        assert out.bias.requires_grad is True

    def test_no_bias_layer(self):
        layer = nn.Linear(4, 2, bias=False)
        out = _prune_fn()(layer, torch.tensor([0]), dim=0)
        assert out.bias is None

    def test_result_is_detached_from_graph(self):
        """The copy must not carry autograd history."""
        layer = nn.Linear(4, 2)
        out = _prune_fn()(layer, torch.tensor([0]), dim=0)
        assert out.weight.grad_fn is None

    def test_no_requires_grad_toggling_in_source(self):
        """ENG-30: the off/on toggle must be gone."""
        src = COMPAT.read_text()
        assert "requires_grad = False" not in src
        assert "requires_grad = True" not in src

    def test_uses_no_grad(self):
        src = COMPAT.read_text()
        assert "with torch.no_grad():" in src

    def test_shim_is_installed_once(self):
        src = COMPAT.read_text()
        assert src.count("_prune_linear_layer") >= 2
