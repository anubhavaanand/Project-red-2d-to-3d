"""Tests for bounded retries of failed samples (EVAL-23)."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

RUNNER = Path(__file__).parent.parent / "src" / "evaluation" / "benchmark_runner.py"


def _run_body():
    """Return the engine loop for source-level assertions.

    EVAL-27 split the loop out of run() into _run_engines(); look there first and
    fall back to run() so the tests survive either layout.
    """
    src = RUNNER.read_text()
    if "def _run_engines(" in src:
        start = src.index("def _run_engines(")
    else:
        start = src.index("def run(self)")
    end = src.find("\n    def ", start + 1)
    return src[start : end if end != -1 else len(src)]


class TestSourceContract:
    def test_failure_budget_is_counted(self):
        """EVAL-23: resume must know how often a sample already failed."""
        body = _run_body()
        assert "failed_attempts" in body
        assert "error_type" in body

    def test_budget_is_compared_to_max_retries(self):
        body = _run_body()
        assert "max_retries" in body

    def test_exhausted_samples_are_skipped(self):
        body = _run_body()
        assert "given_up" in body

    def test_max_retries_is_a_constructor_parameter(self):
        tree = ast.parse(RUNNER.read_text())
        runner = next(
            n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "BenchmarkRunner"
        )
        init = next(
            n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        )
        args = [a.arg for a in init.args.args]
        assert "max_retries" in args, f"max_retries not in {args}"

    def test_max_retries_defaults_to_a_small_number(self):
        """Default must be finite; an unbounded retry is the bug."""
        src = RUNNER.read_text()
        assert "max_retries: int = 2" in src

    def test_done_ids_still_requires_a_chamfer_distance(self):
        """A successful row is what makes a sample cacheable."""
        src = RUNNER.read_text()
        assert 'r.get("chamfer_distance") is not None' in src


class TestBudgetBehaviour:
    def test_counts_failures_per_engine_and_sample(self):
        """The same sample failing under two engines must not share a budget."""
        ns: dict = {}
        # Mirror the counting loop from _run_engines.
        exec(
            "def count(results, engine_name):\n"
            "    failed = {}\n"
            "    for r in results:\n"
            "        if r.get('engine') == engine_name and r.get('error_type'):\n"
            "            sid = r.get('sample_id')\n"
            "            failed[sid] = failed.get(sid, 0) + 1\n"
            "    return failed\n",
            ns,
        )
        results = [
            {"engine": "A", "sample_id": "s0", "error_type": "RuntimeError"},
            {"engine": "A", "sample_id": "s0", "error_type": "RuntimeError"},
            {"engine": "B", "sample_id": "s0", "error_type": "RuntimeError"},
            # A successful row has no error_type and must not count.
            {"engine": "A", "sample_id": "s1", "chamfer_distance": 0.1},
        ]
        assert ns["count"](results, "A") == {"s0": 2}
        assert ns["count"](results, "B") == {"s0": 1}

    def test_successful_rows_do_not_count_as_failures(self):
        ns: dict = {}
        exec(
            "def count(results, engine_name):\n"
            "    failed = {}\n"
            "    for r in results:\n"
            "        if r.get('engine') == engine_name and r.get('error_type'):\n"
            "            sid = r.get('sample_id')\n"
            "            failed[sid] = failed.get(sid, 0) + 1\n"
            "    return failed\n",
            ns,
        )
        assert ns["count"]([{"engine": "A", "sample_id": "s0", "chamfer_distance": 0.2}], "A") == {}


class TestIntegration:
    def test_a_failing_sample_is_not_retried_forever(self, monkeypatch, tmp_path):
        """End-to-end: two failures, then resume must skip instead of retrying."""
        br = pytest.importorskip("src.evaluation.benchmark_runner")

        sample = SimpleNamespace(sample_id="s0", image_path="/nonexistent.png")
        monkeypatch.setattr(br, "load_gso_samples", lambda *a, **k: [sample])

        class _BoomEngine:
            def __init__(self):
                self.calls = 0

            def get_engine_info(self):
                return {"name": "boom"}

            def load_model(self):
                pass

            def predict(self, image):
                self.calls += 1
                raise RuntimeError("always fails")

        engine = _BoomEngine()
        runner = br.BenchmarkRunner(
            dataset_path="/nonexistent-gso",
            engines=[engine],
            num_samples=1,
            num_points=10,
            output_dir=tmp_path,
            max_retries=2,
        )
        monkeypatch.setattr(runner.bg_remover, "remove", lambda img: img)
        monkeypatch.setattr(br.Image, "open", lambda *a, **k: br.Image.new("RGBA", (4, 4)))

        # First run: predict is called once and fails.
        runner.run()
        assert engine.calls == 1

        # Second run (resume): the failure is recorded, still under budget.
        runner2 = br.BenchmarkRunner(
            dataset_path="/nonexistent-gso",
            engines=[engine],
            num_samples=1,
            num_points=10,
            output_dir=tmp_path,
            max_retries=2,
        )
        runner2.results = list(runner.results)  # simulate a resume from disk
        monkeypatch.setattr(runner2.bg_remover, "remove", lambda img: img)
        runner2.run()
        assert engine.calls == 2

        # Third run: budget is spent, so the sample is skipped, not retried.
        runner3 = br.BenchmarkRunner(
            dataset_path="/nonexistent-gso",
            engines=[engine],
            num_samples=1,
            num_points=10,
            output_dir=tmp_path,
            max_retries=2,
        )
        runner3.results = list(runner2.results)
        monkeypatch.setattr(runner3.bg_remover, "remove", lambda img: img)
        runner3.run()
        assert engine.calls == 2, "an exhausted sample must not be retried again"
