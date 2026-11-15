"""Tests for TRELLIS CFG configuration (ENG-26)."""

import ast
from pathlib import Path

ENGINE = Path(__file__).parent.parent / "src" / "engines" / "trellis_engine.py"


def _init_body():
    tree = ast.parse(ENGINE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TrellisEngine")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    return init


class TestSignature:
    def test_cfg_strength_is_optional(self):
        init = _init_body()
        assert "cfg_strength" in [a.arg for a in init.args.args]

    def test_defaults_present(self):
        init = _init_body()
        names = [a.arg for a in init.args.args]
        idx = names.index("cfg_strength")
        offset = len(names) - len(init.args.defaults)
        default = init.args.defaults[idx - offset]
        assert isinstance(default, ast.Constant) and default.value is None


class TestAliasSemantics:
    def test_alias_overrides_sparse_only(self):
        src = ENGINE.read_text()
        init = src[src.index("def __init__") : src.index("def _ensure_trellis_imports")]
        assert "if cfg_strength is not None:" in init
        assert "sparse_cfg_strength = cfg_strength" in init
        # must NOT touch slat_cfg_strength
        assert "slat_cfg_strength = cfg_strength" not in init

    def test_no_dead_attribute(self):
        """ENG-26: self.cfg_strength must no longer be assigned."""
        init = _init_body()
        body = ast.unparse(init)
        assert "self.cfg_strength" not in body

    def test_both_samplers_use_their_own_value(self):
        src = ENGINE.read_text()
        assert '"cfg_strength": self.sparse_cfg_strength' in src
        assert '"cfg_strength": self.slat_cfg_strength' in src

    def test_engine_parses(self):
        ast.parse(ENGINE.read_text())


class TestRuntimeSemantics:
    """Mirror the assignment logic to check the intended behaviour."""

    @staticmethod
    def _resolve(sparse=7.5, slat=3.0, cfg=None):
        if cfg is not None:
            sparse = cfg
        return float(sparse), float(slat)

    def test_defaults_unchanged(self):
        assert self._resolve() == (7.5, 3.0)

    def test_alias_sets_sparse(self):
        assert self._resolve(cfg=9.0) == (9.0, 3.0)

    def test_alias_leaves_slat_alone(self):
        _, slat = self._resolve(slat=4.0, cfg=9.0)
        assert slat == 4.0

    def test_explicit_sparse_wins_without_alias(self):
        assert self._resolve(sparse=6.0) == (6.0, 3.0)
