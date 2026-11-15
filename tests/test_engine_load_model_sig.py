"""Tests for the uniform load_model() signature (ENG-43)."""

import ast
from pathlib import Path

ENGINES = Path(__file__).parent.parent / "src" / "engines"

ALL_ENGINES = [
    "triposr_engine.py",
    "lgm_engine.py",
    "trellis_engine.py",
    "spar3d_engine.py",
    "sf3d_engine.py",
]


def _load_model_args(file: str) -> list:
    tree = ast.parse((ENGINES / file).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "load_model":
                    return [a.arg for a in item.args.args]
    raise AssertionError(f"load_model not found in {file}")


class TestUniformSignature:
    def test_base_takes_only_self(self):
        assert _load_model_args("base.py") == ["self"]

    def test_every_engine_takes_only_self(self):
        """ENG-43: no engine may extend the abstract signature."""
        for f in ALL_ENGINES:
            assert _load_model_args(f) == ["self"], f"{f} has extra params"

    def test_lgm_has_no_big_parameter(self):
        src = (ENGINES / "lgm_engine.py").read_text()
        assert "def load_model(self, big" not in src


class TestLgmBigMovedToInit:
    def test_big_is_an_init_parameter(self):
        tree = ast.parse((ENGINES / "lgm_engine.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LGMEngine")
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        assert "big" in [a.arg for a in init.args.args]

    def test_big_defaults_to_true(self):
        tree = ast.parse((ENGINES / "lgm_engine.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LGMEngine")
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        names = [a.arg for a in init.args.args]
        idx = names.index("big")
        offset = len(names) - len(init.args.defaults)
        default = init.args.defaults[idx - offset]
        assert isinstance(default, ast.Constant) and default.value is True

    def test_stored_on_self(self):
        src = (ENGINES / "lgm_engine.py").read_text()
        assert "self.big = big" in src

    def test_used_instead_of_the_parameter(self):
        src = (ENGINES / "lgm_engine.py").read_text()
        assert 'config_defaults["big"] if self.big else Options()' in src

    def test_internal_call_uses_no_args(self):
        src = (ENGINES / "lgm_engine.py").read_text()
        assert "self.load_model(big=True)" not in src
        assert "self.load_model()" in src


class TestRunnerContract:
    def test_runner_calls_load_model_without_args(self):
        runner = (ENGINES.parent / "evaluation" / "benchmark_runner.py").read_text()
        assert "engine.load_model()" in runner
        assert "engine.load_model(big" not in runner

    def test_app_calls_load_model_without_args(self):
        app = ENGINES.parent.parent / "app.py"
        src = app.read_text()
        assert "e.load_model()" in src
