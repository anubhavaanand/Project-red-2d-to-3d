"""Tests for the figure builder's engine validation (TOOL-04)."""

import ast
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "rebuild_figures.py"

ENGINES = ("TripoSR", "LGM", "SPAR3D", "TRELLIS")


def _load_engine_means():
    """Extract engine_means without importing the module (needs trimesh)."""
    tree = ast.parse(SCRIPT.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "engine_means")
    ns = {}
    # REQUIRED_ENGINES is a module constant
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", "") == "REQUIRED_ENGINES" for t in node.targets
        ):
            exec(ast.unparse(node), ns)
            break
    exec(ast.unparse(fn), ns)
    return ns["engine_means"]


def _df(engines=ENGINES):
    rows = []
    for e in engines:
        for i in range(10):
            rows.append(
                {
                    "engine": e,
                    "sample_id": f"{e}_{i}",
                    "latency_s": 1.0 + i,
                    "vram_peak_gb": 2.0 + i,
                }
            )
    return pd.DataFrame(rows)


class TestEngineMeans:
    def test_returns_latency_and_vram(self):
        fn = _load_engine_means()
        out = fn(_df())
        assert set(out) == {"lat", "vram"}
        assert out["lat"]["TripoSR"] == pytest.approx(5.5)
        assert out["vram"]["LGM"] == pytest.approx(6.5)

    def test_missing_engine_fails_clearly(self):
        fn = _load_engine_means()
        with pytest.raises(SystemExit) as exc:
            fn(_df(engines=("TripoSR", "LGM", "SPAR3D")))
        message = str(exc.value)
        assert "TRELLIS" in message
        assert "missing engine" in message

    def test_missing_engine_lists_what_was_found(self):
        fn = _load_engine_means()
        with pytest.raises(SystemExit) as exc:
            fn(_df(engines=("TripoSR",)))
        assert "Found:" in str(exc.value)

    def test_empty_dataframe_fails(self):
        fn = _load_engine_means()
        with pytest.raises(SystemExit):
            fn(pd.DataFrame(columns=["engine", "latency_s", "vram_peak_gb"]))


class TestSourceContract:
    def test_required_engines_are_pinned(self):
        src = SCRIPT.read_text()
        assert 'REQUIRED_ENGINES = ("TripoSR", "LGM", "SPAR3D", "TRELLIS")' in src

    def test_main_uses_the_helper(self):
        src = SCRIPT.read_text()
        main = src[src.index("def main():") :]
        assert "engine_means(df)" in main

    def test_direct_groupby_is_gone(self):
        """TOOL-04: the unguarded lat/vram groupby in main must be gone."""
        src = SCRIPT.read_text()
        main = src[src.index("def main():") :]
        assert 'df.groupby("engine")["latency_s"].mean()' not in main

    def test_script_parses(self):
        ast.parse(SCRIPT.read_text())
