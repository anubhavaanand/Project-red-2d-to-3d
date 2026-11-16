"""Tests for export payload validation (EXP-04)."""

from pathlib import Path

import numpy as np
import pytest

from src.engines.base import GaussianCloud, Mesh3D
from src.export._validate import require_finite

ENGINE_DIR = Path(__file__).parent.parent / "src" / "export"


class TestRequireFinite:
    def test_finite_passes(self):
        require_finite("v", np.zeros((3, 3), dtype=np.float32))

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_raises(self, bad):
        arr = np.zeros((3, 3), dtype=np.float32)
        arr[0, 0] = bad
        with pytest.raises(ValueError, match="refusing to export"):
            require_finite("v", arr)

    def test_none_is_skipped(self):
        require_finite("v", None)

    def test_empty_is_skipped(self):
        require_finite("v", np.zeros((0, 3), dtype=np.float32))

    def test_error_names_the_array(self):
        a = np.zeros((2, 2), dtype=np.float32)
        b = np.zeros((2, 2), dtype=np.float32)
        b[0, 0] = np.nan
        with pytest.raises(ValueError, match=r"\[1\]"):
            require_finite("v", a, b)


class TestGLBExport:
    def test_rejects_nan_vertices(self, tmp_path):
        from src.export.glb_exporter import export_glb

        mesh = Mesh3D(
            vertices=np.array([[0, 0, 0], [np.nan, 0, 0], [1, 1, 1]], dtype=np.float32),
            faces=np.array([[0, 1, 2]], dtype=np.int64),
        )
        with pytest.raises(ValueError, match="refusing to export"):
            export_glb(mesh, str(tmp_path / "out.glb"))

    def test_rejects_inf_vertices(self, tmp_path):
        from src.export.glb_exporter import export_glb

        mesh = Mesh3D(
            vertices=np.array([[0, 0, 0], [np.inf, 0, 0], [1, 1, 1]], dtype=np.float32),
            faces=np.array([[0, 1, 2]], dtype=np.int64),
        )
        with pytest.raises(ValueError):
            export_glb(mesh, str(tmp_path / "out.glb"))

    def test_valid_mesh_exports(self, tmp_path):
        from src.export.glb_exporter import export_glb

        mesh = Mesh3D(
            vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
            faces=np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64),
        )
        out = tmp_path / "out.glb"
        export_glb(mesh, str(out))
        assert out.exists() and out.stat().st_size > 0


class TestPLYExport:
    def _cloud(self, pos):
        n = len(pos)
        return GaussianCloud(
            positions=pos,
            scales=np.ones((n, 3), dtype=np.float32),
            rotations=np.zeros((n, 4), dtype=np.float32),
            opacities=np.ones((n, 1), dtype=np.float32),
            sh_coeffs=np.zeros((n, 3), dtype=np.float32),
        )

    def test_rejects_nan_positions(self, tmp_path):
        from src.export.ply_exporter import export_ply

        cloud = self._cloud(np.array([[0, 0, 0], [np.nan, 0, 0]], dtype=np.float32))
        with pytest.raises(ValueError, match="refusing to export"):
            export_ply(cloud, str(tmp_path / "out.ply"))

    def test_valid_cloud_exports(self, tmp_path):
        from src.export.ply_exporter import export_ply

        cloud = self._cloud(np.zeros((4, 3), dtype=np.float32))
        out = tmp_path / "out.ply"
        export_ply(cloud, str(out))
        assert out.exists() and out.stat().st_size > 0


class TestSourceContract:
    def test_both_exporters_validate(self):
        for f in ("glb_exporter.py", "ply_exporter.py"):
            src = (ENGINE_DIR / f).read_text()
            assert "require_finite(" in src, f

    def test_engine_parses(self):
        import ast

        for f in ("glb_exporter.py", "ply_exporter.py", "_validate.py"):
            ast.parse((ENGINE_DIR / f).read_text())
