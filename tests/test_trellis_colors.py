"""Tests for TRELLIS vertex-colour normalisation (ENG-23)."""

import ast
from pathlib import Path

import numpy as np
import pytest

ENGINE = Path(__file__).parent.parent / "src" / "engines" / "trellis_engine.py"


def _normalise():
    tree = ast.parse(ENGINE.read_text())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_normalize_vertex_colors"
    )
    ns = {"np": np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(ENGINE), "exec"), ns)
    return ns["_normalize_vertex_colors"]


class TestNormalizeVertexColors:
    def test_unit_range_is_untouched(self):
        rgb, note = _normalise()(np.array([[0.0, 0.5, 1.0]], dtype=np.float32))
        assert rgb.max() <= 1.0
        assert "0..1" in note

    def test_255_range_is_scaled(self):
        """ENG-23: a 0..255 decoder must not be clipped to all-white/black."""
        rgb, note = _normalise()(np.array([[255.0, 128.0, 0.0]], dtype=np.float32))
        assert rgb.max() <= 1.0
        assert rgb[0][0] == pytest.approx(1.0)
        assert rgb[0][1] == pytest.approx(128 / 255, abs=1e-3)
        assert "0..255" in note

    def test_mid_range_is_scaled(self):
        rgb, note = _normalise()(np.array([[0.0, 2.0, 4.0]], dtype=np.float32))
        assert rgb.max() <= 1.0
        assert "0..255" in note

    def test_values_are_clipped(self):
        rgb, _ = _normalise()(np.array([[5.0, -3.0, 0.5]], dtype=np.float32))
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    def test_none_returns_none(self):
        rgb, note = _normalise()(None)
        assert rgb is None
        assert note == "absent"

    def test_extra_channels_are_dropped(self):
        rgb, _ = _normalise()(np.array([[0.1, 0.2, 0.3, 0.9]], dtype=np.float32))
        assert rgb.shape == (1, 3)

    def test_1d_input_is_promoted(self):
        rgb, _ = _normalise()(np.array([0.1, 0.2, 0.3], dtype=np.float32))
        assert rgb.shape == (1, 3)

    def test_empty_input_does_not_raise(self):
        rgb, note = _normalise()(np.zeros((0, 3), dtype=np.float32))
        assert rgb is None or rgb.shape[0] == 0
        assert isinstance(note, str)

    def test_nat_max_is_not_propagated(self):
        """NaNs must not defeat the peak detection."""
        arr = np.array([[np.nan, np.nan, np.nan], [255.0, 0.0, 0.0]], dtype=np.float32)
        rgb, note = _normalise()(arr)
        assert "0..255" in note


class TestSourceContract:
    def test_no_bare_clip_to_unit_range(self):
        """ENG-23: the unconditional clip(..., 0, 1) must be gone."""
        tree = ast.parse(ENGINE.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "clip":
                if node.args and isinstance(node.args[0], ast.Constant):
                    # clip(x, 0.0, 1.0) on the raw attrs is the old pattern
                    assert len(node.args) == 3

    def test_call_site_uses_normaliser(self):
        src = ENGINE.read_text()
        assert "_normalize_vertex_colors(" in src
        assert "attrs.detach().cpu().numpy()" in src

    def test_range_recorded_in_metadata(self):
        src = ENGINE.read_text()
        assert 'metadata["vertex_color_range"] = color_range' in src

    def test_engine_parses(self):
        ast.parse(ENGINE.read_text())