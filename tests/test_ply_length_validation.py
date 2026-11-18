"""Tests for PLY export length validation (EXP-11)."""

import numpy as np
import pytest

from src.engines.base import GaussianCloud
from src.export._validate import require_matching_length
from src.export.ply_exporter import export_ply


def _cloud(n=5, **overrides) -> GaussianCloud:
    cloud = GaussianCloud(
        positions=np.zeros((n, 3), dtype=np.float32),
        scales=np.ones((n, 3), dtype=np.float32),
        rotations=np.zeros((n, 4), dtype=np.float32),
        opacities=np.ones((n, 1), dtype=np.float32),
        sh_coeffs=np.zeros((n, 3), dtype=np.float32),
    )
    for key, value in overrides.items():
        setattr(cloud, key, value)
    return cloud


class TestHelper:
    def test_matching_lengths_pass(self):
        require_matching_length("x", 3, np.zeros(3), np.zeros(3))

    def test_mismatch_raises_with_the_expected_length(self):
        with pytest.raises(ValueError, match="expected 5"):
            require_matching_length("rotations", 5, np.zeros((3, 4)))

    def test_error_message_names_the_field(self):
        with pytest.raises(ValueError, match="rotations"):
            require_matching_length("rotations", 5, np.zeros((3, 4)))

    def test_none_is_skipped(self):
        """Absent optional attributes are legal and must not be checked."""
        require_matching_length("scales", 5, None)

    def test_zero_dimensional_array_is_skipped(self):
        """A scalar has no row axis; length checks do not apply."""
        require_matching_length("x", 5, np.float32(1.0))


class TestExportValidation:
    def test_mismatched_scales_are_rejected(self, tmp_path):
        cloud = _cloud(scales=np.ones((3, 3), dtype=np.float32))
        with pytest.raises(ValueError, match="scales"):
            export_ply(cloud, str(tmp_path / "out.ply"))

    def test_mismatched_rotations_are_rejected(self, tmp_path):
        cloud = _cloud(rotations=np.zeros((3, 4), dtype=np.float32))
        with pytest.raises(ValueError, match="rotations"):
            export_ply(cloud, str(tmp_path / "out.ply"))

    def test_mismatched_opacities_are_rejected(self, tmp_path):
        cloud = _cloud(opacities=np.ones((2, 1), dtype=np.float32))
        with pytest.raises(ValueError, match="opacities"):
            export_ply(cloud, str(tmp_path / "out.ply"))

    def test_mismatched_sh_are_rejected(self, tmp_path):
        cloud = _cloud(sh_coeffs=np.zeros((1, 3), dtype=np.float32))
        with pytest.raises(ValueError, match="sh_coeffs"):
            export_ply(cloud, str(tmp_path / "out.ply"))

    def test_error_names_both_lengths(self, tmp_path):
        """The message must let the caller find the bug without a debugger."""
        cloud = _cloud(rotations=np.zeros((3, 4), dtype=np.float32))
        with pytest.raises(ValueError) as exc:
            export_ply(cloud, str(tmp_path / "out.ply"))
        assert "3" in str(exc.value) and "5" in str(exc.value)

    def test_no_file_is_written_on_a_length_error(self, tmp_path):
        """Refusing up front must not leave a half-written PLY behind."""
        out = tmp_path / "out.ply"
        cloud = _cloud(rotations=np.zeros((3, 4), dtype=np.float32))
        with pytest.raises(ValueError):
            export_ply(cloud, str(out))
        assert not out.exists()

    def test_a_pre_existing_file_is_left_untouched(self, tmp_path):
        """A bad export must not clobber a good file from an earlier run."""
        out = tmp_path / "out.ply"
        out.write_text("PREVIOUS GOOD CONTENT\n")
        cloud = _cloud(rotations=np.zeros((3, 4), dtype=np.float32))
        with pytest.raises(ValueError):
            export_ply(cloud, str(out))
        assert out.read_text() == "PREVIOUS GOOD CONTENT\n"

    def test_validation_runs_before_the_file_is_opened(self):
        """Guard the ordering: validation must precede any I/O."""
        import inspect

        src = inspect.getsource(export_ply)
        validate_at = src.index("require_matching_length")
        mkdir_at = src.index("mkdir")
        open_at = src.index("with open(filepath")
        assert validate_at < mkdir_at, "validate before creating directories"
        assert validate_at < open_at, "validate before opening the file"


class TestHappyPathStillWorks:
    def test_valid_cloud_exports(self, tmp_path):
        out = tmp_path / "ok.ply"
        export_ply(_cloud(), str(out))
        assert out.exists()
        assert out.read_text().startswith("ply")

    def test_optional_attributes_may_be_none(self, tmp_path):
        """The exporter fills defaults for absent optional fields."""
        cloud = _cloud(scales=None, rotations=None, opacities=None, sh_coeffs=None)
        out = tmp_path / "minimal.ply"
        export_ply(cloud, str(out))
        assert out.exists()

    def test_single_scale_per_row_is_broadcast(self, tmp_path):
        cloud = _cloud(scales=np.ones(5, dtype=np.float32))
        out = tmp_path / "broadcast.ply"
        export_ply(cloud, str(out))
        assert out.exists()
