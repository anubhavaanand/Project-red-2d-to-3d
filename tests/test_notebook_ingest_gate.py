"""Tests that the Kaggle kernels fail loudly on missing objects (NB-06)."""

import ast
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
BATCH1 = ROOT / "kaggle" / "batch1" / "kernel_batch1_triposr_lgm.ipynb"
BATCH2 = ROOT / "kaggle" / "batch2" / "kernel_batch2_trellis_spar3d.ipynb"
KERNELS = {"batch1": BATCH1, "batch2": BATCH2}


def _cell_sources(path: Path) -> dict[int, str]:
    nb = json.loads(path.read_text())
    return {i: "".join(c["source"]) for i, c in enumerate(nb["cells"])}


def _python_only(src: str) -> str:
    """Drop IPython magics and shell escapes so the cell can be parsed."""
    return "\n".join(line for line in src.splitlines() if not line.strip().startswith(("!", "%")))


class TestNotebooksAreValid:
    @pytest.mark.parametrize("name,path", sorted(KERNELS.items()))
    def test_notebook_is_valid_json(self, name, path):
        nb = json.loads(path.read_text())
        assert "cells" in nb and nb.get("nbformat") == 4

    @pytest.mark.parametrize("name,path", sorted(KERNELS.items()))
    def test_ingest_cell_parses(self, name, path):
        """Cell 5 is the ingest cell this issue touches; it must stay valid."""
        src = _cell_sources(path)[5]
        ast.parse(_python_only(src))


class TestNoSilentSampleReduction:
    @pytest.mark.parametrize("name,path", sorted(KERNELS.items()))
    def test_target_objects_is_never_shrunk(self, name, path):
        """NB-06: the bug was `TARGET_OBJECTS = [n for n in TARGET_OBJECTS if ...]`."""
        for cell_id, src in _cell_sources(path).items():
            for line in src.splitlines():
                stripped = line.strip()
                if stripped.startswith("TARGET_OBJECTS = [n for n in TARGET_OBJECTS"):
                    raise AssertionError(
                        f"{name} cell {cell_id} shrinks TARGET_OBJECTS: {stripped}"
                    )

    @pytest.mark.parametrize("name,path", sorted(KERNELS.items()))
    def test_missing_objects_raise(self, name, path):
        """A missing object must abort, not warn."""
        src = _cell_sources(path)[5]
        assert "raise RuntimeError" in src
        assert "still_missing" in src

    @pytest.mark.parametrize("name,path", sorted(KERNELS.items()))
    def test_missing_objects_are_recorded(self, name, path):
        """Diagnostics must survive the failure."""
        src = _cell_sources(path)[5]
        assert "ingest_missing.json" in src

    @pytest.mark.parametrize("name,path", sorted(KERNELS.items()))
    def test_failure_message_names_the_count(self, name, path):
        src = _cell_sources(path)[5]
        assert "NUM_SAMPLES" in src
        assert "FATAL" in src or "missing" in src.lower()

    @pytest.mark.parametrize("name,path", sorted(KERNELS.items()))
    def test_assert_on_target_count_is_kept(self, name, path):
        """The pre-flight assert that pins len(TARGET_OBJECTS) == NUM_SAMPLES."""
        src = _cell_sources(path)[5]
        assert "assert len(TARGET_OBJECTS) == NUM_SAMPLES" in src


class TestRowCountGate:
    """NB-10: the aggregation cell must FAIL on a short table, not just warn."""

    def _agg(self, path):
        return next(src for src in _cell_sources(path).values() if "EXPECTED_ENGINES" in src)

    @pytest.mark.parametrize("name,path", sorted(KERNELS.items()))
    def test_aggregation_raises_on_a_row_count_mismatch(self, name, path):
        agg = self._agg(path)
        assert "raise RuntimeError" in agg, "a mismatch must fail the kernel"
        assert "row-count mismatch" in agg

    @pytest.mark.parametrize("name,path", sorted(KERNELS.items()))
    def test_aggregation_checks_every_expected_engine(self, name, path):
        """A missing engine must be caught, not only a short one."""
        agg = self._agg(path)
        assert "_missing_engines" in agg
        assert "set(EXPECTED_ENGINES) - set(_counts)" in agg

    @pytest.mark.parametrize("name,path", sorted(KERNELS.items()))
    def test_expected_engines_use_the_reported_names(self, name, path):
        """EXPECTED_ENGINES must use get_engine_info()['name'], not the class id."""
        line = next(line for line in self._agg(path).splitlines() if "EXPECTED_ENGINES =" in line)
        declared = set(re.findall(r'"([^"]+)"', line))
        expected = {"TripoSR", "LGM"} if name == "batch1" else {"TRELLIS", "SPAR3D"}
        assert declared == expected, f"{name} declares {declared}, expected {expected}"

    @pytest.mark.parametrize("name,path", sorted(KERNELS.items()))
    def test_incomplete_run_is_recorded(self, name, path):
        assert "incomplete_run.json" in self._agg(path)
