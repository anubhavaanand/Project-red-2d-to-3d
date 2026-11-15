"""Tests for duplicate-safe engine indexing (TOOL-11)."""

import ast
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "rebuild_figures.py"


def _by_engine():
    tree = ast.parse(SCRIPT.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "by_engine")
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SCRIPT), "exec"), ns)
    return ns["by_engine"]


def _df(rows):
    return pd.DataFrame(rows)


def _row(sid, engine, cd=0.1):
    return {"sample_id": sid, "engine": engine, "chamfer_distance": cd}


class TestByEngine:
    def test_unique_rows_index_cleanly(self):
        by_engine = _by_engine()
        df = _df([_row("o1", "TripoSR"), _row("o1", "LGM")])
        sub = by_engine(df, "o1")
        assert set(sub.index) == {"TripoSR", "LGM"}
        assert isinstance(sub.loc["TripoSR", "chamfer_distance"], float)

    def test_duplicate_engine_raises(self):
        """TOOL-11: duplicates made .loc return a Series."""
        by_engine = _by_engine()
        df = _df([_row("o1", "TripoSR"), _row("o1", "TripoSR")])
        with pytest.raises(SystemExit) as exc:
            by_engine(df, "o1")
        assert "duplicate engine" in str(exc.value)
        assert "TripoSR" in str(exc.value)

    def test_other_samples_are_unaffected(self):
        by_engine = _by_engine()
        df = _df(
            [
                _row("o1", "TripoSR"),
                _row("o2", "TripoSR"),
                _row("o2", "TripoSR"),  # duplicate only for o2
            ]
        )
        assert by_engine(df, "o1") is not None
        with pytest.raises(SystemExit):
            by_engine(df, "o2")

    def test_missing_sample_returns_empty(self):
        by_engine = _by_engine()
        df = _df([_row("o1", "TripoSR")])
        sub = by_engine(df, "nope")
        assert len(sub) == 0


class TestSourceContract:
    def test_no_direct_set_index_outside_the_helper(self):
        """TOOL-11: inline set_index("engine") sites must be gone.

        The single call inside by_engine is the point of the helper.
        """
        tree = ast.parse(SCRIPT.read_text())
        helper = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "by_engine"
        )
        helper_lines = set(range(helper.lineno, (helper.end_lineno or helper.lineno) + 1))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "set_index":
                if node.lineno not in helper_lines:
                    pytest.fail(f"direct set_index at line {node.lineno}")

    def test_helper_is_used(self):
        src = SCRIPT.read_text()
        assert src.count("by_engine(df, sid)") == 3

    def test_script_parses(self):
        ast.parse(SCRIPT.read_text())
