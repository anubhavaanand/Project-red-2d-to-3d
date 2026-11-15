"""Tests for the TripoSR marching-cubes helper (ENG-04)."""

import ast
from pathlib import Path

import numpy as np
import pytest

ENGINE = Path(__file__).parent.parent / "src" / "engines" / "triposr_engine.py"

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _mc_func():
    """Compile _skimage_mc_func without importing the engine module."""
    src = ENGINE.read_text()
    tree = ast.parse(src)
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_skimage_mc_func"
    )
    ns = {"np": np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(ENGINE), "exec"), ns)
    return ns["_skimage_mc_func"]


def _sphere_volume(n=16):
    zz, yy, xx = np.meshgrid(*[np.linspace(-1, 1, n)] * 3, indexing="ij")
    return ((xx**2 + yy**2 + zz**2) < 0.5).astype(np.float32)


class TestSourceContract:
    def test_dim_call_is_guarded(self):
        """ENG-04: level.dim() must not run unconditionally."""
        src = ENGINE.read_text()
        assert 'hasattr(level, "dim")' in src

    def test_accepts_array_like(self):
        src = ENGINE.read_text()
        assert "np.asarray(level, dtype=np.float32)" in src

    def test_engine_parses(self):
        ast.parse(ENGINE.read_text())


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestWithTorch:
    def test_numpy_input_does_not_raise(self):
        fn = _mc_func()
        vol = _sphere_volume()
        v, f = fn(vol, 0.5)
        assert v.shape[1] == 3
        assert f.shape[1] == 3

    def test_torch_input_still_works(self):
        import torch

        fn = _mc_func()
        vol = torch.from_numpy(_sphere_volume())
        v, f = fn(vol, 0.5)
        assert v.shape[1] == 3

    def test_batched_tensor_is_squeezed(self):
        import torch

        fn = _mc_func()
        vol = torch.from_numpy(_sphere_volume())[None, ...]  # (1, N, N, N)
        v, f = fn(vol, 0.5)
        assert v.shape[1] == 3

    def test_empty_input_returns_empty(self):

        fn = _mc_func()
        v, f = fn(np.zeros((8, 8, 8), dtype=np.float32), 0.5)
        assert v.shape[0] == 0
