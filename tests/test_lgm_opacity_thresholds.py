"""Tests for LGM's opacity-threshold selection (ENG-17)."""

import ast
from pathlib import Path

import numpy as np
import pytest

ENGINE = Path(__file__).parent.parent / "src" / "engines" / "lgm_engine.py"


def _select():
    tree = ast.parse(ENGINE.read_text())
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_select_foreground"
    )
    ns = {
        "np": np,
        "DEFAULT_MIN_FOREGROUND_POINTS": 200,
        "DEFAULT_OPACITY_THRESHOLDS": (0.10, 0.08, 0.03),
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(ENGINE), "exec"), ns)
    return ns["_select_foreground"]


def _old_chain(opa):
    """The previous hardcoded fallback chain, for equivalence checking."""
    valid = opa > 0.10
    if valid.sum() < 200:
        valid = opa > 0.08
    if valid.sum() < 100:
        valid = opa > 0.03
    if valid.sum() < 50:
        valid = np.ones_like(opa, dtype=bool)
    return valid


class TestDefaultEquivalence:
    """ENG-17 must not change the numbers the paper reports."""

    @pytest.mark.parametrize(
        "name,opa",
        [
            ("plenty", np.random.default_rng(0).uniform(0, 1, 5000)),
            ("sparse", np.random.default_rng(1).uniform(0, 0.09, 5000)),
            ("very sparse", np.random.default_rng(2).uniform(0, 0.02, 5000)),
            ("empty", np.zeros(0, dtype=np.float32)),
        ],
    )
    def test_matches_old_chain(self, name, opa):
        mask, _thr = _select()(opa.astype(np.float32))
        assert np.array_equal(mask, _old_chain(opa)), name

    def test_first_threshold_is_reported_when_it_suffices(self):
        opa = np.full(1000, 0.5, dtype=np.float32)
        _mask, thr = _select()(opa)
        assert thr == pytest.approx(0.10)

    def test_second_threshold_when_first_is_too_spare(self):
        opa = np.full(1000, 0.09, dtype=np.float32)
        _mask, thr = _select()(opa)
        assert thr == pytest.approx(0.08)

    def test_third_threshold_when_second_is_too_spare(self):
        opa = np.full(1000, 0.05, dtype=np.float32)
        _mask, thr = _select()(opa)
        assert thr == pytest.approx(0.03)

    def test_all_points_fallback_reports_none(self):
        opa = np.full(1000, 0.01, dtype=np.float32)
        mask, thr = _select()(opa)
        assert thr is None
        assert mask.all()


class TestConfigurability:
    def test_custom_thresholds_are_honoured(self):
        opa = np.full(1000, 0.5, dtype=np.float32)
        mask, thr = _select()(opa, min_points=200, thresholds=(0.9,))
        assert thr is None  # nothing survives 0.9
        assert mask.all()

    def test_custom_min_points(self):
        opa = np.full(100, 0.09, dtype=np.float32)
        _mask, thr = _select()(opa, min_points=10, thresholds=(0.10, 0.08))
        assert thr == pytest.approx(0.08)

    def test_thresholds_are_tried_in_order(self):
        opa = np.full(1000, 0.5, dtype=np.float32)
        _mask, thr = _select()(opa, thresholds=(0.2, 0.1))
        assert thr == pytest.approx(0.2)


class TestWiring:
    def test_defaults_documented(self):
        src = ENGINE.read_text()
        assert "DEFAULT_OPACITY_THRESHOLDS = (0.10, 0.08, 0.03)" in src
        assert "DEFAULT_MIN_FOREGROUND_POINTS = 200" in src

    def test_engine_exposes_the_knobs(self):
        src = ENGINE.read_text()
        assert "opacity_thresholds: tuple = DEFAULT_OPACITY_THRESHOLDS" in src
        assert "min_foreground_points: int = DEFAULT_MIN_FOREGROUND_POINTS" in src

    def test_chosen_branch_recorded_in_metadata(self):
        src = ENGINE.read_text()
        assert '"opacity_threshold_used": self.last_opacity_threshold' in src

    def test_old_hardcoded_chain_is_gone(self):
        src = ENGINE.read_text()
        assert "valid = opa_all > 0.10" not in src

    def test_engine_parses(self):
        ast.parse(ENGINE.read_text())
