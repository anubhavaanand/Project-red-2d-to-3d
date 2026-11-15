"""Tests for renderer cleanup (EVAL-15).

The runner imports torch, so the context manager is lifted out by AST and run
against a fake pyrender module.
"""

import ast
import sys
import types
from pathlib import Path

import pytest

RUNNER = Path(__file__).parent.parent / "src" / "evaluation" / "benchmark_runner.py"


def _offscreen_renderer():
    """Compile _offscreen_renderer with a stubbed pyrender."""
    tree = ast.parse(RUNNER.read_text())
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_offscreen_renderer"
    )
    src = "from contextlib import contextmanager\n\n" + ast.unparse(fn)
    ns = {}
    exec(compile(src, str(RUNNER), "exec"), ns)
    return ns["_offscreen_renderer"]


class _FakeRenderer:
    instances = []

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.deleted = False
        self.render_calls = 0
        _FakeRenderer.instances.append(self)

    def render(self, scene, flags=None):
        self.render_calls += 1
        return "color", "depth"

    def delete(self):
        self.deleted = True


@pytest.fixture
def fake_pyrender(monkeypatch):
    mod = types.ModuleType("pyrender")
    mod.OffscreenRenderer = _FakeRenderer
    _FakeRenderer.instances.clear()
    monkeypatch.setitem(sys.modules, "pyrender", mod)
    return mod


class TestNormalPath:
    def test_renderer_is_deleted_on_exit(self, fake_pyrender):
        with _offscreen_renderer()(64, 64) as r:
            r.render(None)
        assert r.deleted

    def test_dimensions_passed_through(self, fake_pyrender):
        with _offscreen_renderer()(128, 256) as r:
            pass
        assert (r.width, r.height) == (128, 256)


class TestExceptionPath:
    def test_renderer_deleted_when_render_raises(self, fake_pyrender):
        """EVAL-15: the leak happened exactly here."""
        renderer = _offscreen_renderer()

        class Boom(_FakeRenderer):
            def render(self, scene, flags=None):
                raise RuntimeError("render failed")

        fake_pyrender.OffscreenRenderer = Boom
        with pytest.raises(RuntimeError, match="render failed"):
            with renderer(64, 64) as r:
                r.render(None)
        assert r.deleted, "GL context leaked on exception"

    def test_delete_failure_does_not_mask_the_original(self, fake_pyrender):
        class BadDelete(_FakeRenderer):
            def delete(self):
                raise RuntimeError("delete failed")

        fake_pyrender.OffscreenRenderer = BadDelete
        # delete failure is reported, not raised, so it cannot mask the result
        with _offscreen_renderer()(32, 32) as r:
            pass
        assert r.deleted is False  # delete raised, but we did not propagate


class TestSourceContract:
    def test_helper_exists(self):
        assert "def _offscreen_renderer(width: int, height: int):" in RUNNER.read_text()

    def test_delete_is_in_finally(self):
        src = SERVER_SRC()
        helper = src[
            src.index("def _offscreen_renderer") : src.index("def _ensure_pyopengl_compat")
        ]
        assert "finally:" in helper
        assert "renderer.delete()" in helper

    def test_no_bare_delete_left(self):
        """The old 'r.delete()' after render() must be gone."""
        src = SERVER_SRC()
        assert "        r.delete()\n" not in src

    def test_both_call_sites_use_the_helper(self):
        src = SERVER_SRC()
        assert src.count("with _offscreen_renderer(resolution * 2, resolution * 2) as r:") == 2


def SERVER_SRC():
    return RUNNER.read_text()
