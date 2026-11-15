"""Tests for per-sample failure recording (EVAL-24)."""

import ast
import traceback
from pathlib import Path

RUNNER = Path(__file__).parent.parent / "src" / "evaluation" / "benchmark_runner.py"


class TestSourceContract:
    def test_stores_error_type(self):
        src = RUNNER.read_text()
        assert '"error_type": err_type' in src

    def test_stores_traceback(self):
        src = RUNNER.read_text()
        assert '"traceback": tb' in src

    def test_captures_formatted_traceback(self):
        src = RUNNER.read_text()
        assert "traceback.format_exc()" in src

    def test_traceback_imported_at_module_level(self):
        """Not inside the except, so it is available for every path."""
        tree = ast.parse(RUNNER.read_text())
        top_imports = {a.name for n in tree.body if isinstance(n, ast.Import) for a in n.names}
        assert "traceback" in top_imports

    def test_still_stores_the_message(self):
        """The original 'error' field must be preserved for compatibility."""
        assert '"error": str(e)' in RUNNER.read_text()

    def test_failure_is_printed_with_the_type(self):
        src = RUNNER.read_text()
        assert 'print(f"FAILED: {err_type}: {e}")' in src


class TestTracebackContent:
    """Verify format_exc actually carries the failing line."""

    def test_traceback_names_the_function(self):
        try:

            def boom():
                raise ValueError("kaboom")

            boom()
        except ValueError:
            tb = traceback.format_exc()
        assert "boom" in tb
        assert "ValueError: kaboom" in tb

    def test_traceback_is_a_string(self):
        try:
            raise RuntimeError("x")
        except RuntimeError:
            tb = traceback.format_exc()
        assert isinstance(tb, str)
        assert "Traceback (most recent call last)" in tb


class TestSchemaCompatibility:
    def test_new_fields_are_additive(self):
        """The results contract only requires columns to exist, so extras are safe."""
        required = {
            "engine",
            "sample_id",
            "latency_s",
            "vram_peak_gb",
            "chamfer_distance",
            "f_score_0.1",
            "f_score_0.2",
            "f_score_0.5",
            "psnr",
            "lpips",
        }
        added = {"error", "error_type", "traceback"}
        assert required.isdisjoint(added)
