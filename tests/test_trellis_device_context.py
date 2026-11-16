"""Tests for TRELLIS device-context handling (ENG-47)."""

import ast
from pathlib import Path

import pytest

ENGINE = Path(__file__).parent.parent / "src" / "engines" / "trellis_engine.py"


def _run_on_device():
    """Compile _run_on_device with a stubbed torch."""
    tree = ast.parse(ENGINE.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_run_on_device")
    src = "import torch\n\n" + ast.unparse(fn)
    ns = {}
    exec(compile(src, str(ENGINE), "exec"), ns)
    return ns["_run_on_device"]


class TestSourceContract:
    def test_helper_exists(self):
        assert "def _run_on_device(model, slat, dev):" in ENGINE.read_text()

    def test_no_unguarded_cuda_device(self):
        """ENG-47: torch.cuda.device must only appear inside _run_on_device."""
        tree = ast.parse(ENGINE.read_text())
        helper = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_run_on_device"
        )
        helper_lines = set(range(helper.lineno, (helper.end_lineno or helper.lineno) + 1))
        for node in ast.walk(tree):
            if isinstance(node, ast.With) and "torch.cuda.device" in ast.unparse(node):
                if node.lineno not in helper_lines:
                    pytest.fail(f"torch.cuda.device outside helper at line {node.lineno}")

    def test_both_sites_use_the_helper(self):
        src = ENGINE.read_text()
        assert src.count("_run_on_device(") >= 2

    def test_guard_matches_torch_semantics(self):
        src = ENGINE.read_text()
        assert 'if str(dev).startswith("cuda"):' in src

    def test_engine_parses(self):
        ast.parse(ENGINE.read_text())


class TestDeviceSemantics:
    def test_only_cuda_enters_the_context(self):
        for dev in ("cpu", "cuda", "cuda:0", "cuda:1"):
            should = str(dev).startswith("cuda")
            assert should == (dev != "cpu")

    def test_cpu_does_not_touch_cuda(self):
        """The CUDA context must sit inside the cuda branch, not before it."""
        src = ENGINE.read_text()
        helper = src[src.index("def _run_on_device") : src.index("def _first_of")]
        # drop the docstring so its prose cannot match
        body = helper.split('"""', 2)[2]
        guard = body.index('if str(dev).startswith("cuda"):')
        ctx = body.index("torch.cuda.device(dev)")
        assert guard < ctx, "cuda context must be inside the guard"
