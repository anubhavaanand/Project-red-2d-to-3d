"""Tests for TRELLIS's pipeline setter (ENG-28)."""

import ast
from pathlib import Path

ENGINE = Path(__file__).parent.parent / "src" / "engines" / "trellis_engine.py"


class TestSetPipeline:
    def test_attach_is_gone(self):
        src = ENGINE.read_text()
        assert "def attach(self" not in src

    def test_set_pipeline_exists(self):
        src = ENGINE.read_text()
        assert "def set_pipeline(self" in src

    def test_dev_is_typed(self):
        src = ENGINE.read_text()
        assert "def set_pipeline(self, pipeline, dev: str | None = None) -> None:" in src

    def test_none_pipeline_is_rejected(self):
        src = ENGINE.read_text()
        assert 'raise ValueError("pipeline must not be None")' in src

    def test_dev_converted_to_torch_device(self):
        src = ENGINE.read_text()
        set_block = src[src.index("def set_pipeline") : src.index("def load_model")]
        assert "self.dev = torch.device(dev)" in set_block

    def test_pipeline_is_stored(self):
        src = ENGINE.read_text()
        set_block = src[src.index("def set_pipeline") : src.index("def load_model")]
        assert "self.pipeline = pipeline" in set_block

    def test_engine_parses(self):
        ast.parse(ENGINE.read_text())


class TestNoDanglingReferences:
    def test_no_attach_callers_anywhere(self):
        """No production code may still call the removed attach()."""
        base = ENGINE.parent.parent.parent
        here = Path(__file__).resolve()
        for py in base.rglob("*.py"):
            if ".venv" in str(py) or py.resolve() == here:
                continue
            for line in py.read_text(errors="ignore").splitlines():
                if ".attach(" in line and "def attach" not in line:
                    raise AssertionError(f"{py}: dangling .attach( call")
