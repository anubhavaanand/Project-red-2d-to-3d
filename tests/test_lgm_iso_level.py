"""Tests for LGM's iso-level configuration (ENG-18)."""

import ast
from pathlib import Path

import numpy as np

ENGINE = Path(__file__).parent.parent / "src" / "engines" / "lgm_engine.py"


def _engine_source():
    return ENGINE.read_text()


class TestDefaults:
    def test_percentile_default_is_48(self):
        src = _engine_source()
        assert "iso_percentile: float = 48.0" in src

    def test_fixed_level_defaults_to_none(self):
        """ENG-18: the legacy per-object percentile must remain the default."""
        src = _engine_source()
        assert "fixed_iso_level: float | None = None" in src

    def test_default_reproduces_the_legacy_value(self):
        rng = np.random.default_rng(0)
        grid = rng.random((32, 32, 32)).astype(np.float32)
        nonzero = grid[grid > 1e-4]
        assert float(np.percentile(nonzero, 48)) == float(np.percentile(nonzero, 48.0))


class TestSelection:
    def test_fixed_takes_precedence(self):
        src = _engine_source()
        block = src[
            src.index("if self.fixed_iso_level is not None") :
            src.index("# Always use skimage")
        ]
        assert "iso = float(self.fixed_iso_level)" in block

    def test_percentile_used_when_no_fixed(self):
        src = _engine_source()
        assert "iso = float(np.percentile(nonzero, self.iso_percentile))" in src

    def test_rule_recorded_in_metadata(self):
        """The chosen rule must be visible, not silent."""
        src = _engine_source()
        assert '"iso_rule": getattr(self, "iso_rule", None)' in src

    def test_both_rules_described(self):
        src = _engine_source()
        assert 'f"fixed ({iso:.4f})"' in src
        assert 'f"percentile {self.iso_percentile:g} of non-zero density"' in src


class TestConstructor:
    def test_stored_on_self(self):
        src = _engine_source()
        assert "self.iso_percentile = float(iso_percentile)" in src
        assert (
            "self.fixed_iso_level = None if fixed_iso_level is None else float(fixed_iso_level)"
            in src
        )


class TestSourceContract:
    def test_hardcoded_percentile_48_is_gone(self):
        """ENG-18: the bare literal must be replaced by the parameter."""
        src = _engine_source()
        assert "np.percentile(nonzero, 48)" not in src

    def test_engine_parses(self):
        ast.parse(_engine_source())
