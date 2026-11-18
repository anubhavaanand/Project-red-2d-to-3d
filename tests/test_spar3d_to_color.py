"""Behavioural tests for SPAR3D's optional vis.to_color() path (ENG-34)."""

import types

import numpy as np
import pytest
from conftest import TorchStub


@pytest.fixture(scope="module")
def spar3d():
    """Import spar3d_engine with a minimal torch stub (restored on teardown)."""
    stub = TorchStub()
    stub()
    try:
        from src.engines import spar3d_engine

        return spar3d_engine
    finally:
        stub.restore()


class _Vis:
    """Stand-in for a trimesh visual object."""

    def __init__(self, vertex_colors=None, with_to_color=True):
        self._vc = vertex_colors
        if with_to_color:
            self.to_color = self._to_color

    def _to_color(self):
        return types.SimpleNamespace(vertex_colors=self._vc)


class TestToColorPresent:
    def test_reads_vertex_colors(self, spar3d):
        vc = np.array([[1.0, 0.0, 0.0]] * 4, dtype=np.float32)
        out = spar3d._vertex_colors_from_to_color(_Vis(vc), 4)
        assert out is not None
        assert out.shape == (4, 3)

    def test_uint8_range_is_scaled(self, spar3d):
        vc = np.full((4, 3), 255, dtype=np.uint8)
        out = spar3d._vertex_colors_from_to_color(_Vis(vc), 4)
        assert out.max() == pytest.approx(1.0)

    def test_float_range_is_not_scaled(self, spar3d):
        vc = np.full((4, 3), 0.5, dtype=np.float32)
        out = spar3d._vertex_colors_from_to_color(_Vis(vc), 4)
        assert out.max() == pytest.approx(0.5)

    def test_rgba_is_truncated_to_rgb(self, spar3d):
        vc = np.zeros((4, 4), dtype=np.float32)
        out = spar3d._vertex_colors_from_to_color(_Vis(vc), 4)
        assert out.shape == (4, 3)


class TestToColorAbsent:
    def test_missing_method_returns_none(self, spar3d):
        """ENG-34: no to_color must not raise."""
        assert spar3d._vertex_colors_from_to_color(_Vis(with_to_color=False), 4) is None

    def test_object_without_the_attribute_returns_none(self, spar3d):
        assert spar3d._vertex_colors_from_to_color(object(), 4) is None

    def test_none_visual_returns_none(self, spar3d):
        assert spar3d._vertex_colors_from_to_color(None, 4) is None


class TestMalformedResults:
    def test_no_vertex_colors_attribute_returns_none(self, spar3d):
        vis = types.SimpleNamespace(to_color=lambda: types.SimpleNamespace())
        assert spar3d._vertex_colors_from_to_color(vis, 4) is None

    def test_length_mismatch_returns_none(self, spar3d):
        vc = np.zeros((3, 3), dtype=np.float32)
        assert spar3d._vertex_colors_from_to_color(_Vis(vc), 4) is None

    def test_matching_length_succeeds(self, spar3d):
        vc = np.zeros((4, 3), dtype=np.float32)
        assert spar3d._vertex_colors_from_to_color(_Vis(vc), 4) is not None


class TestNonCallableAttribute:
    def test_non_callable_to_color_is_treated_as_absent(self, spar3d):
        """A property that isn't callable must not be invoked."""
        vis = types.SimpleNamespace(to_color="not callable")
        assert spar3d._vertex_colors_from_to_color(vis, 4) is None


class TestWiring:
    def test_helper_exists_and_is_used(self, spar3d):
        import inspect

        src = inspect.getsource(spar3d.Spar3dEngine._extract_spar3d_vertex_colors)
        assert "_vertex_colors_from_to_color(vis, len(verts))" in src

    def test_no_bare_to_color_call_left(self, spar3d):
        """The blind vis.to_color() call must be gone from the engine method."""
        import ast
        import inspect
        import textwrap

        src = textwrap.dedent(inspect.getsource(spar3d.Spar3dEngine._extract_spar3d_vertex_colors))
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "to_color":
                if not isinstance(node.func.value, ast.Name):
                    pytest.fail(
                        f"bare vis.to_color() at line {node.lineno} — "
                        "should go through _vertex_colors_from_to_color"
                    )
