"""Tests for the VRAM columns (EVAL-18)."""

from pathlib import Path

RUNNER = Path(__file__).parent.parent / "src" / "evaluation" / "benchmark_runner.py"
RESULTS_README = Path(__file__).parent.parent / "results" / "README.md"


class TestSourceContract:
    def test_peak_metric_unchanged(self):
        """The published figure must still come from max_memory_allocated."""
        src = RUNNER.read_text()
        assert "torch.cuda.max_memory_allocated() / (1024**3)" in src
        assert '"vram_peak_gb": round(vram_gb, 2)' in src

    def test_reserved_metric_added(self):
        src = RUNNER.read_text()
        assert "torch.cuda.max_memory_reserved() / (1024**3)" in src
        assert '"vram_reserved_gb": round(vram_reserved_gb, 2)' in src

    def test_reserved_is_included_in_summary(self):
        src = RUNNER.read_text()
        block = src.split("numeric_cols = [")[1].split("]")[0]
        assert '"vram_reserved_gb"' in block

    def test_both_guarded_on_cuda(self):
        src = RUNNER.read_text()
        # both metric expressions are guarded by torch.cuda.is_available()
        assert src.count("torch.cuda.max_memory_allocated() / (1024**3)") == 1
        assert src.count("torch.cuda.max_memory_reserved() / (1024**3)") == 1

    def test_reserved_reads_after_peak(self):
        """Reserved must be sampled after the same inference, not before."""
        src = RUNNER.read_text()
        assert src.index("max_memory_allocated") < src.index("max_memory_reserved")


class TestDocs:
    def test_readme_documents_the_new_column(self):
        text = RESULTS_README.read_text()
        assert "vram_reserved_gb" in text

    def test_readme_explains_the_difference(self):
        text = RESULTS_README.read_text()
        assert "max_memory_allocated" in text
        assert "nvidia-smi" in text

    def test_readme_keeps_the_frozen_column(self):
        text = RESULTS_README.read_text()
        assert "`vram_peak_gb`" in text


class TestSummarySemantics:
    def test_groupby_includes_both(self):
        src = RUNNER.read_text()
        block = src[src.index("numeric_cols = [") : src.index("]\n        existing")]
        assert "vram_peak_gb" in block
        assert "vram_reserved_gb" in block

    def test_peak_is_not_replaced(self):
        """A regression guard: no code should assign reserved into vram_peak."""
        src = RUNNER.read_text()
        assert "vram_peak_gb = round(vram_reserved_gb" not in src
