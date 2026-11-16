"""Tests for PLY export performance and correctness (EXP-10)."""

import ast
import time
from pathlib import Path

import numpy as np
import pytest

from src.engines.base import GaussianCloud
from src.export.ply_exporter import export_ply

EXPORTER = Path(__file__).parent.parent / "src" / "export" / "ply_exporter.py"


def _cloud(n, seed=0):
    rng = np.random.default_rng(seed)
    return GaussianCloud(
        positions=rng.random((n, 3), dtype=np.float32),
        scales=rng.random((n, 3), dtype=np.float32),
        rotations=rng.random((n, 4), dtype=np.float32),
        opacities=rng.random((n, 1), dtype=np.float32),
        sh_coeffs=rng.random((n, 3), dtype=np.float32),
    )


def _load_vertices(path):
    """Parse a PLY ascii body into an (n, 17) float array."""
    lines = Path(path).read_text().splitlines()
    i = (
        lines.index("end_header")
        if "end_header" in lines
        else next(k for k, line in enumerate(lines) if line.strip() == "end_header")
    )
    body = lines[i + 1 :]
    return np.array([[float(x) for x in row.split()] for row in body if row.strip()])


class TestCorrectness:
    def test_vertex_count(self, tmp_path):
        out = tmp_path / "a.ply"
        export_ply(_cloud(500), str(out))
        assert len(_load_vertices(out)) == 500

    def test_positions_preserved(self, tmp_path):
        cloud = _cloud(10)
        out = tmp_path / "a.ply"
        export_ply(cloud, str(out))
        rows = _load_vertices(out)
        assert np.allclose(rows[:, :3], cloud.positions, atol=1e-5)

    def test_column_layout(self, tmp_path):
        """17 columns: pos(3) normal(3) sh_dc(3) opacity(1) scale(3) rot(4)."""
        out = tmp_path / "a.ply"
        export_ply(_cloud(5), str(out))
        assert _load_vertices(out).shape[1] == 17

    def test_header_has_all_properties(self, tmp_path):
        out = tmp_path / "a.ply"
        export_ply(_cloud(3), str(out))
        text = out.read_text()
        for prop in ("x", "nx", "f_dc_0", "opacity", "scale_0", "rot_3"):
            assert f"property float {prop}" in text

    def test_matches_reference_implementation(self, tmp_path):
        """Output must be numerically identical to the old per-row writer."""
        cloud = _cloud(200, seed=7)
        out = tmp_path / "a.ply"
        export_ply(cloud, str(out))
        rows = _load_vertices(out)

        pos = np.asarray(cloud.positions, dtype=np.float32)
        sh = np.asarray(cloud.sh_coeffs, dtype=np.float32)
        opa = np.asarray(cloud.opacities, dtype=np.float32)
        sc = np.asarray(cloud.scales, dtype=np.float32)
        rot = np.asarray(cloud.rotations, dtype=np.float32)

        assert np.allclose(rows[:, 0:3], pos, atol=1e-5)
        assert np.allclose(rows[:, 6:9], sh, atol=1e-5)
        assert np.allclose(rows[:, 9], opa[:, 0], atol=1e-5)
        assert np.allclose(rows[:, 10:13], sc, atol=1e-5)
        assert np.allclose(rows[:, 13:17], rot, atol=1e-5)
        assert np.allclose(rows[:, 3:6], 0.0)  # normals


class TestPerformance:
    def test_large_cloud_exports_quickly(self, tmp_path):
        """EXP-10: one write per Gaussian took seconds; a single write is fast."""
        out = tmp_path / "big.ply"
        start = time.perf_counter()
        export_ply(_cloud(100_000, seed=3), str(out))
        elapsed = time.perf_counter() - start
        # generous bound; the vectorised path does 100k in well under a second
        assert elapsed < 5.0, f"took {elapsed:.2f}s"

    def test_output_size_scales_linearly(self, tmp_path):
        small = tmp_path / "s.ply"
        big = tmp_path / "b.ply"
        export_ply(_cloud(1000), str(small))
        export_ply(_cloud(4000), str(big))
        ratio = big.stat().st_size / small.stat().st_size
        assert 3.0 < ratio < 5.0, ratio


class TestSourceContract:
    def test_no_per_row_write_loop(self):
        """EXP-10: the f.write-per-gaussian loop must be gone."""
        tree = ast.parse(EXPORTER.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.For) and isinstance(node.iter, ast.Call):
                if getattr(node.iter.func, "id", "") == "range":
                    pytest.fail(f"range loop at line {node.lineno}")

    def test_uses_savetxt(self):
        assert "np.savetxt(" in EXPORTER.read_text()

    def test_exporter_parses(self):
        ast.parse(EXPORTER.read_text())
