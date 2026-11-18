"""Tests that BenchmarkRunner.run() restores signal handlers (EVAL-27)."""

import ast
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

RUNNER = Path(__file__).parent.parent / "src" / "evaluation" / "benchmark_runner.py"


class TestSourceContract:
    def test_run_saves_the_previous_handlers(self):
        """EVAL-27: the host's handlers must be read before being replaced."""
        src = RUNNER.read_text()
        assert "signal.getsignal(sig)" in src

    def test_run_restores_handlers_in_a_finally(self):
        """A finally block is what guarantees restoration on exception paths."""
        src = RUNNER.read_text()
        run_at = src.index("def run(self)")
        # The next method after run() bounds the block we inspect.
        next_def = src.find("\n    def ", run_at + 1)
        run_body = src[run_at : next_def if next_def != -1 else len(src)]
        assert "finally:" in run_body, "restoration must be in a finally block"
        assert "signal.signal(sig, previous)" in run_body

    def test_module_imports_signal_at_top_level(self):
        """A function-local `import signal` next to the mutation is easy to miss."""
        tree = ast.parse(RUNNER.read_text())
        top_level = {
            alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names
        }
        assert "signal" in top_level

    def test_getsignal_returns_none_is_handled(self):
        """getsignal returns None when no handler is set; signal() would reject it."""
        src = RUNNER.read_text()
        assert "if previous is None" in src


class TestBehaviour:
    def test_handlers_are_restored_after_run(self, monkeypatch, tmp_path):
        """Exercise the real install/restore pair, not just the early-exit path.

        run() installs the handlers only after loading samples, so the test
        supplies one sample and an engine that raises. That reaches the mutation
        and lets us assert it was undone — which is exactly the regression this
        issue describes.

        Skipped without torch: importing benchmark_runner pulls in torch.
        """
        br = pytest.importorskip("src.evaluation.benchmark_runner")

        before = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

        # One sample so run() gets past the loader.
        sample = SimpleNamespace(sample_id="s0", image_path="/nonexistent.png")
        monkeypatch.setattr(br, "load_gso_samples", lambda *a, **k: [sample])

        # An engine whose predict() fails, so run() finishes quickly but does
        # execute the engine loop under the installed handlers.
        class _BoomEngine:
            def get_engine_info(self):
                return {"name": "boom"}

            def load_model(self):
                pass

            def predict(self, image):
                raise RuntimeError("boom")

        runner = br.BenchmarkRunner(
            dataset_path="/nonexistent-gso",
            engines=[_BoomEngine()],
            num_samples=1,
            num_points=10,
            output_dir=tmp_path,
        )
        # Do not let background removal or image IO run.
        monkeypatch.setattr(runner.bg_remover, "remove", lambda img: img)
        monkeypatch.setattr(br.Image, "open", lambda *a, **k: Image.new("RGBA", (4, 4)))

        runner.run()

        after = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        assert after == before, "run() left its signal handlers installed"

    def test_no_handler_is_left_bound_to_the_runner(self):
        """The specific leak: a bound method of the runner surviving run()."""
        src = RUNNER.read_text()
        # The handler must be a local closure, never stored on self.
        assert "_sig_handler" in src
        assert "self._sig_handler" not in src
