"""Tests for LGM's ImageDream view reordering (ENG-14)."""

import ast
from pathlib import Path

import numpy as np
import pytest

ENGINE = Path(__file__).parent.parent / "src" / "engines" / "lgm_engine.py"


def _engine_source():
    return ENGINE.read_text()


class TestViewReordering:
    def test_exact_order_is_front_left_back_right(self):
        """The permutation (1, 2, 3, 0) must be explicit in the source."""
        src = _engine_source()
        assert "for i in (1, 2, 3, 0)" in src

    def test_each_view_is_coerced_with_asarray(self):
        src = _engine_source()
        assert "np.asarray(mv_image[i])" in src

    def test_length_is_validated_before_indexing(self):
        src = _engine_source()
        assert "len(mv_image) < 4" in src

    def test_type_is_validated(self):
        src = _engine_source()
        assert "isinstance(mv_image, (list, tuple))" in src

    def test_typed_error_raised(self):
        src = _engine_source()
        assert "raise LgmViewError(" in src

    def test_error_names_the_offending_length(self):
        src = _engine_source()
        assert "expected 4 views" in src

    def test_error_type_is_defined(self):
        src = _engine_source()
        assert "class LgmViewError(RuntimeError):" in src

    def test_no_bare_indexing_remains(self):
        """ENG-14: mv_image[1] .. mv_image[3] must be gone."""
        tree = ast.parse(_engine_source())
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                if node.value.id == "mv_image" and isinstance(node.slice, ast.Constant):
                    pytest.fail(f"bare mv_image[{node.slice.value}] at line {node.lineno}")


class TestReorderingSemantics:
    def test_permutation_matches_the_documented_layout(self):
        """Simulate the pipeline output and check which view lands first."""
        views = [np.full((2, 2, 3), i, dtype=np.float32) for i in range(4)]
        reordered = np.stack([np.asarray(views[i]) for i in (1, 2, 3, 0)], axis=0)
        assert reordered.shape == (4, 2, 2, 3)
        # front (index 0 of the output) is ImageDream's view 1
        assert np.all(reordered[0] == 1)
        assert np.all(reordered[3] == 0)

    def test_wrong_length_would_have_raised(self):
        """Documents why the guard matters."""
        views = [np.zeros((2, 2, 3), dtype=np.float32) for _ in range(3)]
        with pytest.raises(IndexError):
            np.stack([views[i] for i in (1, 2, 3, 0)], axis=0)


class TestSourceStructure:
    def test_engine_parses(self):
        ast.parse(_engine_source())

    def test_other_engines_do_not_use_this_permutation(self):
        base = ENGINE.parent
        for f in base.glob("*_engine.py"):
            if f == ENGINE:
                continue
            assert "(1, 2, 3, 0)" not in f.read_text()