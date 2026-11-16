"""Tests for GLB vertex-colour conversion (EXP-05)."""

import ast
from pathlib import Path

import numpy as np
import pytest
import trimesh

from src.engines.base import Mesh3D
from src.export.glb_exporter import export_glb

ENGINE = Path(__file__).parent.parent / "src" / "engines" / "spar3d_engine.py"  # placeholder
GLB = Path(__file__).parent.parent / "src" / "export" / "glb_exporter.py"


def _export_colours(vertex_colors, tmp_path):
    """Export a tiny mesh with the given colours and read them back."""
    mesh = Mesh3D(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
        faces=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64),
        vertex_colors=vertex_colors,
    )
    out = tmp_path / "m.glb"
    export_glb(mesh, str(out))
    loaded = trimesh.load(out, process=False)
    # GLB loads as a Scene; take the single geometry inside it
    if hasattr(loaded, "geometry"):
        assert len(loaded.geometry) == 1
        return next(iter(loaded.geometry.values()))
    return loaded


class TestUint8NotRescaled:
    def test_nearly_black_uint8_stays_dark(self, tmp_path):
        """EXP-05: a uint8 mesh with max()==1 must not be scaled to white."""
        vc = np.zeros((4, 3), dtype=np.uint8)
        vc[0] = 1
        loaded = _export_colours(vc, tmp_path)
        assert loaded.visual.vertex_colors[..., :3].max() <= 2

    def test_uint8_is_passed_through(self, tmp_path):
        vc = np.full((4, 3), 200, dtype=np.uint8)
        loaded = _export_colours(vc, tmp_path)
        # 200 stays 200 — it must not be rescaled to 255
        assert loaded.visual.vertex_colors[..., :3].max() == 200

    def test_uint8_black_stays_black(self, tmp_path):
        vc = np.zeros((4, 3), dtype=np.uint8)
        loaded = _export_colours(vc, tmp_path)
        assert loaded.visual.vertex_colors[..., :3].max() <= 2


class TestFloatScaled:
    def test_float_unit_range_is_scaled(self, tmp_path):
        vc = np.full((4, 3), 0.5, dtype=np.float32)
        loaded = _export_colours(vc, tmp_path)
        assert loaded.visual.vertex_colors[..., :3].max() == pytest.approx(127, abs=3)

    def test_float_one_is_white(self, tmp_path):
        vc = np.ones((4, 3), dtype=np.float32)
        loaded = _export_colours(vc, tmp_path)
        assert loaded.visual.vertex_colors[..., :3].max() == pytest.approx(255, abs=2)

    def test_float_above_one_is_clipped(self, tmp_path):
        vc = np.full((4, 3), 300.0, dtype=np.float32)
        loaded = _export_colours(vc, tmp_path)
        assert loaded.visual.vertex_colors[..., :3].max() <= 255


class TestSourceContract:
    def test_max_based_heuristic_is_gone(self):
        """EXP-05: the vc.max() <= 1.05 check must be gone."""
        assert "vc.max() <= 1.05" not in GLB.read_text()

    def test_uses_dtype_check(self):
        assert "np.issubdtype(vc.dtype, np.floating)" in GLB.read_text()

    def test_glb_parses(self):
        ast.parse(GLB.read_text())
