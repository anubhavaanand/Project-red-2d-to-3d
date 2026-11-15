"""Tests for NaN-safe engine selection (TOOL-05)."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "rebuild_figures.py"


def _best_engine():
    tree = ast.parse(SCRIPT.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "best_engine")
    ns = {"np": np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SCRIPT), "exec"), ns)
    return ns["best_engine"]


def _sub(cd, f1):
    return pd.DataFrame(
        {"engine": ["A", "B", "C"], "chamfer_distance": cd, "f_score_0.1": f1}
    ).set_index("engine")


class TestAllNaN:
    def test_returns_none_instead_of_raising(self):
        """TOOL-05: idxmin on an all-NaN column raised."""
        best = _best_engine()
        assert best(_sub([np.nan] * 3, [1, 2, 3]), "chamfer_distance") is None

    def test_all_nan_fscore(self):
        best = _best_engine()
        assert best(_sub([1, 2, 3], [np.nan] * 3), "f_score_0.1", largest=True) is None


class TestPartialNaN:
    def test_ignores_nan_rows(self):
        best = _best_engine()
        assert best(_sub([0.5, np.nan, 0.2], [1, 2, 3]), "chamfer_distance") == "C"

    def test_normal_selection(self):
        best = _best_engine()
        assert best(_sub([0.5, 0.2, 0.9], [1, 2, 3]), "chamfer_distance") == "B"

    def test_largest_flag(self):
        best = _best_engine()
        assert best(_sub([1, 2, 3], [0.1, 0.9, 0.5]), "f_score_0.1", largest=True) == "B"


class TestEdgeCases:
    def test_infinite_values_are_treated_as_missing(self):
        best = _best_engine()
        assert best(_sub([np.inf, 0.2, np.inf], [1, 2, 3]), "chamfer_distance") == "B"

    def test_missing_column_returns_none(self):
        best = _best_engine()
        assert best(_sub([1, 2, 3], [1, 2, 3]), "nope") is None

    def test_empty_frame(self):
        best = _best_engine()
        empty = pd.DataFrame({"chamfer_distance": []})
        assert best(empty, "chamfer_distance") is None


class TestSourceContract:
    def test_no_direct_idxmin_on_a_raw_frame(self):
        """TOOL-05: direct idxmin/idxmax on a raw column must be gone.

        Calls on the dropna()'d `series` inside best_engine are fine.
        """
        tree = ast.parse(SCRIPT.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") in (
                "idxmin",
                "idxmax",
            ):
                inner = node.func.value
                if isinstance(inner, ast.Name) and inner.id == "series":
                    continue
                pytest.fail(f"direct {node.func.attr}() at line {node.lineno}")

    def test_helper_is_used(self):
        src = SCRIPT.read_text()
        assert src.count("best_engine(sub,") >= 2

    def test_script_parses(self):
        ast.parse(SCRIPT.read_text())
