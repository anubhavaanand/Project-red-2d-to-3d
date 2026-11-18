"""Tests for the Gradio queue wiring (APP-14)."""

import ast
from pathlib import Path

import pytest

APP_PY = Path(__file__).parent.parent / "app.py"


def _app_tree():
    return ast.parse(APP_PY.read_text())


def _queue_keywords():
    """Return the keyword arguments passed to app.queue(...) as source text."""
    for node in ast.walk(_app_tree()):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "queue"
        ):
            return {kw.arg for kw in node.keywords if kw.arg}
    return set()


class TestQueueIsConfigured:
    def test_queue_is_called(self):
        assert "app.queue(" in APP_PY.read_text(), "APP-14: the app must enable a queue"

    def test_queue_precedes_launch(self):
        """queue() must be called before launch(), or it has no effect."""
        src = APP_PY.read_text()
        assert src.index("app.queue(") < src.index("app.launch(")

    def test_concurrency_is_bounded(self):
        """A cap is what stops one user's inference blocking everyone else."""
        assert "default_concurrency_limit" in _queue_keywords()

    def test_queue_size_is_bounded(self):
        """An unbounded waiting line is its own memory/DoS problem."""
        assert "max_size" in _queue_keywords()


class TestConfiguration:
    def test_limits_come_from_the_environment(self):
        src = APP_PY.read_text()
        assert "GRADIO_CONCURRENCY" in src
        assert "GRADIO_QUEUE_SIZE" in src

    def test_concurrency_defaults_to_one(self):
        """Default to serial: engines share one GPU and must not oversubscribe VRAM."""
        src = APP_PY.read_text()
        assert 'GRADIO_CONCURRENCY", "1"' in src


class TestSignatureCompatibility:
    def test_queue_keywords_match_gradio(self):
        """The kwargs passed must exist on gradio.Blocks.queue.

        Skipped where gradio is not installed (the core CI job); the torch-cpu
        job and local runs cover it.
        """
        gradio = pytest.importorskip("gradio")
        import inspect

        valid = set(inspect.signature(gradio.Blocks.queue).parameters)
        unknown = _queue_keywords() - valid
        assert not unknown, (
            f"queue() does not accept {sorted(unknown)} in gradio {gradio.__version__}"
        )
