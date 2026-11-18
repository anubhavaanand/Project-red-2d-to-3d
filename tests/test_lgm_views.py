"""Behavioural tests for LGM's ImageDream view reordering (ENG-14).

Exercises the real reorder_mv_views() function rather than asserting on source.
"""

import numpy as np
import pytest
from conftest import TorchStub


@pytest.fixture(scope="module")
def lgm():
    """Import lgm_engine with a minimal torch stub (restored on teardown)."""
    stub = TorchStub()
    stub()
    try:
        from src.engines import lgm_engine

        return lgm_engine
    finally:
        stub.restore()


def _views(n=4, size=2):
    return [np.full((size, size, 3), i, dtype=np.float32) for i in range(n)]


class TestReordering:
    def test_output_shape(self, lgm):
        out = lgm.reorder_mv_views(_views())
        assert out.shape == (4, 2, 2, 3)

    def test_front_is_image_dreams_second_view(self, lgm):
        """LGM front (index 0) comes from ImageDream index 1."""
        out = lgm.reorder_mv_views(_views())
        assert np.all(out[0] == 1)

    def test_back_is_image_dreams_first_view(self, lgm):
        """LGM back (index 3) comes from ImageDream index 0."""
        out = lgm.reorder_mv_views(_views())
        assert np.all(out[3] == 0)

    def test_full_permutation(self, lgm):
        """The whole mapping is (1, 2, 3, 0)."""
        out = lgm.reorder_mv_views(_views())
        assert [float(out[i].flat[0]) for i in range(4)] == [1.0, 2.0, 3.0, 0.0]

    def test_accepts_a_tuple(self, lgm):
        out = lgm.reorder_mv_views(tuple(_views()))
        assert out.shape == (4, 2, 2, 3)

    def test_extra_views_are_ignored(self, lgm):
        """Five views still work; the first four are reordered."""
        out = lgm.reorder_mv_views(_views(5))
        assert out.shape == (4, 2, 2, 3)
        assert float(out[0].flat[0]) == 1.0

    def test_single_channel_views(self, lgm):
        views = [np.full((2, 2, 1), i, dtype=np.float32) for i in range(4)]
        assert lgm.reorder_mv_views(views).shape == (4, 2, 2, 1)


class TestRejectsBadPayloads:
    def test_too_few_views_raises(self, lgm):
        with pytest.raises(lgm.LgmViewError, match="expected 4 views"):
            lgm.reorder_mv_views(_views(3))

    def test_empty_raises(self, lgm):
        with pytest.raises(lgm.LgmViewError):
            lgm.reorder_mv_views([])

    def test_none_raises(self, lgm):
        with pytest.raises(lgm.LgmViewError):
            lgm.reorder_mv_views(None)

    def test_dict_raises(self, lgm):
        with pytest.raises(lgm.LgmViewError):
            lgm.reorder_mv_views({"front": 1})

    def test_generator_raises(self, lgm):
        """A generator has no len() — it must not be indexed."""
        with pytest.raises(lgm.LgmViewError):
            lgm.reorder_mv_views(iter(_views()))

    def test_error_names_the_type(self, lgm):
        with pytest.raises(lgm.LgmViewError, match="dict"):
            lgm.reorder_mv_views({"a": 1})

    def test_error_is_a_runtime_error(self, lgm):
        assert issubclass(lgm.LgmViewError, RuntimeError)


class TestWiring:
    def test_predict_uses_the_helper(self, lgm):
        """The reordering logic lives in one place, called from predict."""
        import inspect

        src = inspect.getsource(lgm.LGMEngine.predict)
        assert "reorder_mv_views(mv_image)" in src

    def test_no_blind_indexing_left_in_predict(self, lgm):
        import ast
        import inspect
        import textwrap

        src = textwrap.dedent(inspect.getsource(lgm.LGMEngine.predict))
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                if node.value.id == "mv_image" and isinstance(node.slice, ast.Constant):
                    pytest.fail(f"bare mv_image[{node.slice.value}] still in predict")

    def test_engine_parses(self, lgm):
        assert lgm.LGMEngine is not None
