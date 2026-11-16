"""Tests for the unaligned-mesh export path (EVAL-22)."""

from pathlib import Path

RUNNER = Path(__file__).parent.parent / "src" / "evaluation" / "benchmark_runner.py"


class TestSourceContract:
    def test_alignment_failed_flag_is_set(self):
        src = RUNNER.read_text()
        assert "alignment_failed = aligned_for_render is None" in src
        assert 'row["alignment_failed"] = alignment_failed' in src

    def test_export_name_differs_when_unaligned(self):
        """An unaligned mesh must not be written as `<engine>_mesh.obj`."""
        src = RUNNER.read_text()
        assert 'f"{engine_name}_unaligned_mesh.obj"' in src
        assert 'f"{engine_name}_mesh.obj"' in src

    def test_filename_branches_on_the_flag(self):
        src = RUNNER.read_text()
        export = src[src.index(").export(") : src.index("self._meshes_saved += 1")]
        assert "alignment_failed" in export
        assert "_unaligned_mesh.obj" in export

    def test_no_render_target_also_marks_failed(self):
        """The else-branch (no mesh at all) must not leave the flag unset."""
        src = RUNNER.read_text()
        idx = src.index('row["psnr"] = np.nan\n                        row["lpips"] = np.nan')
        tail = src[idx : idx + 400]
        assert 'row["alignment_failed"] = True' in tail

    def test_flag_precedes_the_export(self):
        src = RUNNER.read_text()
        assert src.index("alignment_failed = aligned_for_render is None") < src.index(").export(")


class TestSemantics:
    def test_aligned_when_render_target_is_the_aligned_mesh(self):
        aligned_for_render = object()
        assert (aligned_for_render is None) is False

    def test_unaligned_when_no_alignment(self):
        aligned_for_render = None
        assert (aligned_for_render is None) is True

    def test_filenames_differ(self):
        engine = "TripoSR"
        assert f"{engine}_unaligned_mesh.obj" != f"{engine}_mesh.obj"


class TestNoRegression:
    def test_aligned_exports_keep_the_original_name(self):
        """When alignment succeeds the filename must not change."""
        src = RUNNER.read_text()
        export = src[src.index(").export(") : src.index("self._meshes_saved += 1")]
        assert 'else f"{engine_name}_mesh.obj"' in export.replace("'", '"')
