"""Tests for TripoSR's interior colour offsets (ENG-07)."""

import ast
from pathlib import Path

import pytest

ENGINE = Path(__file__).parent.parent / "src" / "engines" / "triposr_engine.py"

NOMINAL_RADIUS = 0.5
BASE_OFFSETS = [0.0, -0.002, -0.005]


def _offsets_for(radius: float) -> list:
    """Mirror the production expression."""
    scale = radius / NOMINAL_RADIUS if NOMINAL_RADIUS else 1.0
    return [0.0, BASE_OFFSETS[1] * scale, BASE_OFFSETS[2] * scale]


class TestDefaultUnchanged:
    def test_nominal_radius_reproduces_original_offsets(self):
        """ENG-07 must not change the default configuration's numbers."""
        assert _offsets_for(0.5) == pytest.approx([0.0, -0.002, -0.005])

    def test_first_offset_is_always_surface(self):
        assert _offsets_for(0.5)[0] == 0.0
        assert _offsets_for(2.0)[0] == 0.0


class TestScaling:
    def test_larger_radius_scales_offsets_up(self):
        offsets = _offsets_for(1.0)
        assert offsets[1] == pytest.approx(-0.004)
        assert offsets[2] == pytest.approx(-0.010)

    def test_smaller_radius_scales_offsets_down(self):
        offsets = _offsets_for(0.25)
        assert offsets[1] == pytest.approx(-0.001)
        assert offsets[2] == pytest.approx(-0.0025)

    def test_offsets_stay_ordered(self):
        for radius in (0.1, 0.25, 0.5, 1.0, 2.0):
            o = _offsets_for(radius)
            assert o[0] > o[1] > o[2], f"offsets not ordered at radius={radius}"

    def test_offsets_remain_small_relative_to_radius(self):
        """The deepest probe must stay well inside the object."""
        for radius in (0.25, 0.5, 1.0):
            assert abs(_offsets_for(radius)[2]) < 0.05 * radius


class TestSourceContract:
    def test_offsets_are_scaled_not_literal(self):
        src = ENGINE.read_text()
        assert "offsets = [0.0, -0.002 * offset_scale, -0.005 * offset_scale]" in src

    def test_old_literal_list_is_gone(self):
        src = ENGINE.read_text()
        assert "offsets = [0.0, -0.002, -0.005]" not in src

    def test_nominal_radius_is_documented(self):
        src = ENGINE.read_text()
        assert "nominal_radius = 0.5" in src

    def test_engine_parses(self):
        ast.parse(ENGINE.read_text())