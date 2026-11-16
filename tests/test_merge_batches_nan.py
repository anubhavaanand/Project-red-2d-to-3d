"""Tests for NaN filtering during batch merge (TOOL-10)."""

import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
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


class TestFinite:
    def test_accepts_numbers(self, mb):
        assert mb._finite(0.1) is True
        assert mb._finite(0) is True
        assert mb._finite(-1.5) is True

    def test_rejects_none(self, mb):
        assert mb._finite(None) is False

    def test_rejects_nan(self, mb):
        assert mb._finite(float("nan")) is False
        assert mb._finite(np.nan) is False

    def test_rejects_inf(self, mb):
        assert mb._finite(float("inf")) is False
        assert mb._finite(float("-inf")) is False

    def test_rejects_non_numeric(self, mb):
        assert mb._finite("abc") is False
        assert mb._finite({}) is False

    def test_accepts_numeric_strings(self, mb):
        """JSON may carry numbers as strings."""
        assert mb._finite("0.25") is True


def _row(engine="TripoSR", sample_id="obj_1", cd=0.1):
    return {
        "engine": engine,
        "sample_id": sample_id,
        "chamfer_distance": cd,
        "f_score_0.1": 0.9,
        "psnr": 20.0,
        "lpips": 0.1,
        "latency_s": 1.0,
        "vram_peak_gb": 1.0,
    }


class TestMergeSkipsFailedRows:
    def _run(self, mb, tmp_path, monkeypatch, rows):
        d1 = tmp_path / "b1"
        d1.mkdir()
        (d1 / "x_results.json").write_text(json.dumps(rows))
        monkeypatch.chdir(tmp_path)
        mb.merge_all([str(d1)])
        import pandas as pd

        return pd.read_csv(tmp_path / "b1" / "unified_4models.csv")

    def test_nan_cd_is_excluded(self, mb, tmp_path, monkeypatch):
        df = self._run(
            mb,
            tmp_path,
            monkeypatch,
            [
                _row("TripoSR", "o1", cd=0.1),
                _row("TripoSR", "o2", cd=float("nan")),
            ],
        )
        assert len(df) == 1
        assert df["sample_id"].iloc[0] == "o1"

    def test_none_cd_is_excluded(self, mb, tmp_path, monkeypatch):
        df = self._run(
            mb,
            tmp_path,
            monkeypatch,
            [
                _row("TripoSR", "o1", cd=0.1),
                _row("TripoSR", "o2", cd=None),
            ],
        )
        assert len(df) == 1

    def test_missing_cd_is_excluded(self, mb, tmp_path, monkeypatch):
        bad = _row("TripoSR", "o2")
        del bad["chamfer_distance"]
        df = self._run(mb, tmp_path, monkeypatch, [_row("TripoSR", "o1", 0.1), bad])
        assert len(df) == 1

    def test_all_failed_yields_no_rows(self, mb, tmp_path, monkeypatch):
        """When every row fails the finite check, no CSV is written."""
        d1 = tmp_path / "b1"
        d1.mkdir()
        (d1 / "x_results.json").write_text(json.dumps([_row("TripoSR", "o1", cd=float("nan"))]))
        monkeypatch.chdir(tmp_path)
        mb.merge_all([str(d1)])  # returns early, no SystemExit
        assert not (tmp_path / "b1" / "unified_4models.csv").exists()

    def test_inf_cd_is_excluded(self, mb, tmp_path, monkeypatch):
        df = self._run(
            mb,
            tmp_path,
            monkeypatch,
            [
                _row("TripoSR", "o1", cd=0.1),
                _row("TripoSR", "o2", cd=float("inf")),
            ],
        )
        assert len(df) == 1


class TestSourceContract:
    def test_only_key_presence_is_gone(self):
        src = SCRIPT.read_text()
        assert '"chamfer_distance" in r' not in src

    def test_finite_check_used(self):
        src = SCRIPT.read_text()
        assert '_finite(r.get("chamfer_distance"))' in src

    def test_script_parses(self):
        ast.parse(SCRIPT.read_text())
