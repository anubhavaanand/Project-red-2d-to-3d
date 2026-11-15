"""Tests for engine unload_model (ENG-09)."""

import ast
from pathlib import Path

import pytest

BASE = Path(__file__).parent.parent / "src" / "engines" / "base.py"


class _Engine:
    """Stand-in shaped like a real engine (mirrors AbstractEngine)."""

    def __init__(self):
        self.device = "cuda"
        self.output_format = "mesh"
        self.model = object()
        self.mv_pipeline = object()
        self.rays_embeddings = object()
        self._resize_foreground = object()


@pytest.fixture
def unload_model():
    """Compile AbstractEngine.unload_model with a stubbed ABC."""
    src = BASE.read_text()
    tree = ast.parse(src)
    fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "unload_model"
    )
    body = ast.unparse(fn)
    # drop the decorator-free def, exec in a throwaway namespace
    ns = {}
    exec(body, ns)
    return ns["unload_model"]


class TestUnloadModel:
    def test_model_state_is_cleared(self, unload_model):
        e = _Engine()
        unload_model(e)
        assert e.model is None
        assert e.mv_pipeline is None
        assert e.rays_embeddings is None

    def test_private_helpers_are_cleared(self, unload_model):
        """_resize_foreground holds a function reference; it must go too."""
        e = _Engine()
        unload_model(e)
        assert e._resize_foreground is None

    def test_scalar_config_is_preserved(self, unload_model):
        """Config must survive, otherwise a reload cannot be configured."""
        e = _Engine()
        unload_model(e)
        assert e.device == "cuda"
        assert e.output_format == "mesh"

    def test_is_idempotent(self, unload_model):
        e = _Engine()
        unload_model(e)
        unload_model(e)  # must not raise
        assert e.model is None

    def test_missing_attributes_do_not_raise(self, unload_model):
        class Bare:
            device = "cpu"

        unload_model(Bare())  # no model attrs at all


class TestInterface:
    def test_base_defines_unload_model(self):
        src = BASE.read_text()
        assert "def unload_model(self) -> None:" in src

    def test_unload_model_is_not_abstract(self):
        """It must be concrete, so existing engines inherit it for free."""
        tree = ast.parse(BASE.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "unload_model":
                decorators = [ast.unparse(d) for d in node.decorator_list]
                assert "abstractmethod" not in decorators

    def test_runner_hook_exists(self):
        runner = Path(__file__).parent.parent / "src" / "evaluation" / "benchmark_runner.py"
        assert 'hasattr(engine, "unload_model")' in runner.read_text()

    @pytest.mark.parametrize("engine_file", ["lgm_engine.py", "trellis_engine.py"])
    def test_heavy_engines_override_and_free_cuda(self, engine_file):
        src = (BASE.parent / engine_file).read_text()
        assert "def unload_model(self) -> None:" in src
        assert "super().unload_model()" in src
        assert "torch.cuda.empty_cache()" in src

    def test_all_engines_import_cleanly(self):
        """Every engine module must still parse after the change."""
        for f in BASE.parent.glob("*_engine.py"):
            ast.parse(f.read_text())
