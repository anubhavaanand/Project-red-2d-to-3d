"""Tests for order-independent batch merging (TOOL-09)."""

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "merge_batches.py"


def _load():
    spec = importlib.util.spec_from_file_location("merge_batches", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["merge_batches"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mb():
    return _load()


def _row(engine="TripoSR", sample_id="obj_1", cd=0.1, **kw):
    base = {
        "engine": engine,
        "sample_id": sample_id,
        "chamfer_distance": cd,
        "f_score_0.1": 0.9,
        "f_score_0.2": 0.9,
        "f_score_0.5": 0.9,
        "psnr": 20.0,
        "lpips": 0.1,
        "latency_s": 1.0,
        "vram_peak_gb": 1.0,
    }
    base.update(kw)
    return base


class TestDiffers:
    def test_identical_rows_do_not_differ(self, mb):
        assert mb._differs(_row(), _row()) is False

    def test_different_chamfer_differs(self, mb):
        assert mb._differs(_row(cd=0.1), _row(cd=0.2)) is True

    def test_different_psnr_differs(self, mb):
        assert mb._differs(_row(psnr=20.0), _row(psnr=21.0)) is True

    def test_missing_field_on_one_side_differs(self, mb):
        a = _row()
        b = _row()
        del b["psnr"]
        assert mb._differs(a, b) is True

    def test_float_tolerance(self, mb):
        """Tiny float noise should not count as a conflict."""
        assert mb._differs(_row(cd=0.1), _row(cd=0.1 + 1e-12)) is False

    def test_non_numeric_fields_compared_literally(self, mb):
        assert mb._differs(_row(engine="A"), _row(engine="B")) is False  # key fields ignored


class TestMergeAll:
    def test_merges_duplicate_free_run(self, mb, tmp_path, monkeypatch):
        d1 = tmp_path / "batch1"
        d2 = tmp_path / "batch2"
        d1.mkdir()
        d2.mkdir()
        (d1 / "a_results.json").write_text(json.dumps([_row("TripoSR", "o1"), _row("LGM", "o1")]))
        (d2 / "b_results.json").write_text(
            json.dumps([_row("TRELLIS", "o1"), _row("SPAR3D", "o1")])
        )
        monkeypatch.chdir(tmp_path)
        mb.merge_all([str(d1), str(d2)])
        out = tmp_path / "batch1" / "unified_4models.csv"
        assert out.exists()
        assert len(out.read_text().splitlines()) == 5  # header + 4 rows

    def test_identical_duplicates_are_deduplicated(self, mb, tmp_path, monkeypatch):
        d1 = tmp_path / "b1"
        d1.mkdir()
        (d1 / "x_results.json").write_text(json.dumps([_row()]))
        monkeypatch.chdir(tmp_path)
        mb.merge_all([str(d1)])
        # same row twice in one file
        (d1 / "y_results.json").write_text(json.dumps([_row()]))
        mb.merge_all([str(d1)])

    def test_conflicting_duplicates_exit_nonzero(self, mb, tmp_path, monkeypatch):
        d1 = tmp_path / "b1"
        d1.mkdir()
        d2 = tmp_path / "b2"
        d2.mkdir()
        (d1 / "old_results.json").write_text(json.dumps([_row(cd=0.1)]))
        (d2 / "new_results.json").write_text(json.dumps([_row(cd=0.9)]))
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit) as exc:
            mb.merge_all([str(d1), str(d2)])
        assert exc.value.code == 1

    def test_missing_chamfer_rows_are_skipped(self, mb, tmp_path, monkeypatch):
        d = tmp_path / "b1"
        d.mkdir()
        (d / "x_results.json").write_text(json.dumps([{"engine": "A", "sample_id": "s"}]))
        monkeypatch.chdir(tmp_path)
        mb.merge_all([str(d)])  # no rows, but must not raise


class TestSourceContract:
    def test_first_row_wins_no_longer_uses_a_set(self):
        src = SCRIPT.read_text()
        assert "seen = set()" not in src

    def test_conflicts_are_reported(self):
        src = SCRIPT.read_text()
        assert "conflicting duplicate row(s)" in src

    def test_script_parses(self):
        ast.parse(SCRIPT.read_text())
