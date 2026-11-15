"""Tests for the CUDA cache helper (ENG-08)."""

import ast
import sys
import types
from pathlib import Path

import pytest

MEMORY = Path(__file__).parent.parent / "src" / "engines" / "_memory.py"
ENGINES = Path(__file__).parent.parent / "src" / "engines"


def _free_cuda_cache():
    """Compile free_cuda_cache with a stubbed torch module."""
    tree = ast.parse(MEMORY.read_text())
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "free_cuda_cache"
    )
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(MEMORY), "exec"), ns)
    return ns["free_cuda_cache"]


class TestFreeCudaCache:
    def test_no_op_without_cuda(self, monkeypatch):
        """ENG-08: a CPU-only host must not touch the CUDA cache."""
        calls = []
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(
            is_available=lambda: False,
            empty_cache=lambda: calls.append(1),
        )
        monkeypatch.setitem(sys.modules, "torch", torch)

        assert _free_cuda_cache()() is False
        assert calls == []

    def test_calls_empty_cache_when_cuda_present(self, monkeypatch):
        calls = []
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: calls.append(1),
        )
        monkeypatch.setitem(sys.modules, "torch", torch)

        assert _free_cuda_cache()() is True
        assert calls == [1]

    def test_missing_torch_is_tolerated(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        # force ImportError on `import torch`
        import builtins

        real_import = builtins.__import__

        def _fail(name, *a, **k):
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", _fail)
        assert _free_cuda_cache()() is False


class TestCallSites:
    @pytest.mark.parametrize(
        "engine", ["triposr_engine.py", "lgm_engine.py", "trellis_engine.py",
                   "spar3d_engine.py", "sf3d_engine.py"]
    )
    def test_engine_uses_the_helper(self, engine):
        src = (ENGINES / engine).read_text()
        assert "from ._memory import free_cuda_cache" in src
        assert "free_cuda_cache()" in src

    @pytest.mark.parametrize(
        "engine", ["triposr_engine.py", "lgm_engine.py", "trellis_engine.py",
                   "spar3d_engine.py", "sf3d_engine.py"]
    )
    def test_no_unguarded_cuda_call_remains(self, engine):
        """ENG-08: raw empty_cache() calls must be gone from engine code."""
        src = (ENGINES / engine).read_text()
        code_lines = [
            l for l in src.splitlines() if not l.lstrip().startswith("#")
        ]
        for i, line in enumerate(code_lines):
            if line.strip() == "torch.cuda.empty_cache()":
                prev = code_lines[i - 1].strip() if i else ""
                assert prev.startswith(
                    "if torch.cuda.is_available()"
                ), f"{engine}: unguarded call -> {line.strip()!r}"


class TestSourceStructure:
    def test_helper_parses(self):
        ast.parse(MEMORY.read_text())

    def test_all_engines_parse(self):
        for f in ENGINES.glob("*_engine.py"):
            ast.parse(f.read_text())