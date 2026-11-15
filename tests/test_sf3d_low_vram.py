"""Tests for SF3D's configurable low_vram_mode (ENG-37)."""

import ast
from pathlib import Path

ENGINE = Path(__file__).parent.parent / "src" / "engines" / "sf3d_engine.py"


def _engine_class_source():
    src = ENGINE.read_text()
    tree = ast.parse(src)
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SF3DEngine"
    )
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    return init


class TestInitSignature:
    def test_low_vram_mode_is_a_parameter(self):
        args = {a.arg for a in _engine_class_source().args.args}
        assert "low_vram_mode" in args

    def test_defaults_to_true(self):
        """Preserves the previous T4-safe behaviour by default."""
        init = _engine_class_source()
        defaults = init.args.defaults
        names = [a.arg for a in init.args.args]
        idx = names.index("low_vram_mode")
        # defaults align to the tail of args
        default_offset = len(names) - len(defaults)
        assert default_offset <= idx
        default = defaults[idx - default_offset]
        assert ast.literal_eval(default) is True

    def test_stored_on_self(self):
        body = ast.unparse(_engine_class_source())
        assert "self.low_vram_mode = low_vram_mode" in body


class TestCallSite:
    def test_call_site_uses_the_attribute(self):
        src = ENGINE.read_text()
        assert "low_vram_mode=self.low_vram_mode" in src

    def test_no_hardcoded_true_remains(self):
        """ENG-37: the literal keyword argument must be gone from the call."""
        src = ENGINE.read_text()
        load = src[src.index("def load_model") : src.index("def predict")]
        assert "low_vram_mode=True" not in load

    def test_documented_default_in_info(self):
        src = ENGINE.read_text()
        assert "low_vram_mode=True" in src  # in the vram_approx note


class TestSourceStructure:
    def test_engine_parses(self):
        ast.parse(ENGINE.read_text())

    def test_spar3d_already_had_it(self):
        """SPAR3D exposed the knob before this fix; SF3D was the outlier."""
        spar3d = (ENGINE.parent / "spar3d_engine.py").read_text()
        assert "low_vram_mode: bool = False" in spar3d
        assert "low_vram_mode=self.low_vram_mode" in spar3d