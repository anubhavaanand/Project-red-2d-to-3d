"""Tests for the Gaussian scale convention (EXP-07)."""

from pathlib import Path

import numpy as np
import pytest

from src.engines.base import GaussianCloud
from src.export.ply_exporter import export_ply

PLY = Path(__file__).parent.parent / "src" / "export" / "ply_exporter.py"
BASE = Path(__file__).parent.parent / "src" / "engines" / "base.py"

# Log-space scales: exp() of these is a sane splat size in world units.
SCALES = np.array([[-1.0, -1.0, -1.0], [-2.0, -2.0, -2.0]], dtype=np.float32)
N = len(SCALES)


def _cloud(**overrides) -> GaussianCloud:
    cloud = GaussianCloud(
        positions=np.zeros((N, 3), dtype=np.float32),
        scales=SCALES.copy(),
        rotations=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (N, 1)),
        opacities=np.ones((N, 1), dtype=np.float32),
        sh_coeffs=np.zeros((N, 3), dtype=np.float32),
    )
    for key, value in overrides.items():
        setattr(cloud, key, value)
    return cloud


def _read_scale_column(tmp_path) -> np.ndarray:
    """Read the scale_0..2 columns back out of an exported PLY."""
    out = tmp_path / "cloud.ply"
    export_ply(_cloud(), str(out))
    header, body = out.read_text().split("end_header\n")
    # Column position = index among the "property" lines, in file order.
    properties = [line.split()[-1] for line in header.splitlines() if line.startswith("property ")]
    scale_idx = [i for i, name in enumerate(properties) if name.startswith("scale_")]
    assert len(scale_idx) == 3, f"expected 3 scale columns, got {scale_idx}"
    rows = [line.split() for line in body.strip().splitlines()]
    return np.array([[float(r[i]) for i in scale_idx] for r in rows], dtype=np.float32)


class TestScalesAreWrittenVerbatim:
    def test_log_scales_are_not_exponentiated(self, tmp_path):
        """EXP-07: the exporter must not apply exp() — viewers do that."""
        written = _read_scale_column(tmp_path)
        assert np.allclose(written, SCALES, atol=1e-6), (
            "scales must be written verbatim; an exp() here would render splats "
            "at the wrong size in every 3DGS viewer"
        )

    def test_written_values_match_the_input_exactly(self, tmp_path):
        written = _read_scale_column(tmp_path)
        assert written.shape == SCALES.shape
        for i in range(N):
            assert written[i][0] == pytest.approx(SCALES[i][0], abs=1e-6)

    def test_negative_scales_survive(self, tmp_path):
        """Log-scales are typically negative; a clamp or exp() would destroy them."""
        written = _read_scale_column(tmp_path)
        assert (written < 0).all()


class TestConventionIsDocumented:
    def test_exporter_docstring_states_the_convention(self):
        doc = export_ply.__doc__ or ""
        assert "log-space" in doc.lower() or "log space" in doc.lower()
        assert "EXP-07" in doc

    def test_docstring_warns_against_exponentiating(self):
        doc = export_ply.__doc__ or ""
        assert "do **not** exponentiate" in doc.lower()

    def test_dataclass_documents_the_convention(self):
        """Discoverable at the source, not only in the writer."""
        src = BASE.read_text()
        assert "EXP-07" in src
        assert "log-space" in src.lower() or "log space" in src.lower()

    def test_no_exp_is_applied_to_scales(self):
        """Guard the actual behaviour, not just the prose."""
        import inspect

        src = inspect.getsource(export_ply)
        # np.exp on the scales array is what we must never do.
        assert "np.exp(scales)" not in src
        assert "np.exp(gaussians.scales)" not in src
