"""Tests for the results CSV contract validator (DATA-10)."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "validate_results.py"


def _load():
    spec = importlib.util.spec_from_file_location("validate_results", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["validate_results"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def vr():
    return _load()


def _valid_df(vr, engines=("TripoSR", "LGM", "TRELLIS", "SPAR3D"), per_engine=100):
    rows = []
    for engine in engines:
        for i in range(per_engine):
            rows.append(
                {
                    "engine": engine,
                    "sample_id": f"{engine}_{i}",
                    "latency_s": 1.0,
                    "vram_peak_gb": 1.0,
                    "chamfer_distance": 0.1,
                    "f_score_0.1": 0.9,
                    "f_score_0.2": 0.9,
                    "f_score_0.5": 0.9,
                    "psnr": 20.0,
                    "lpips": 0.1,
                }
            )
    return pd.DataFrame(rows)


class TestValidInput:
    def test_complete_run_passes(self, vr):
        assert vr.validate(_valid_df(vr)) == []

    def test_partial_mode_skips_counts(self, vr):
        df = _valid_df(vr).head(10)
        assert vr.validate(df, expect_complete=False) == []


class TestRowCount:
    def test_too_few_rows_fails(self, vr):
        problems = vr.validate(_valid_df(vr).head(399))
        assert any("expected 400 rows" in p for p in problems)

    def test_too_many_rows_fails(self, vr):
        df = _valid_df(vr)
        extra = df.iloc[[0]].copy()
        problems = vr.validate(pd.concat([df, extra], ignore_index=True))
        assert any("expected 400 rows" in p for p in problems)


class TestEngines:
    def test_missing_engine_fails(self, vr):
        df = _valid_df(vr, engines=("TripoSR", "LGM", "TRELLIS"))
        # pad to 400 so only the engine check trips
        pad = _valid_df(vr, engines=("TRELLIS",), per_engine=100)
        df = pd.concat([df, pad], ignore_index=True)
        problems = vr.validate(df)
        assert any("expected engines" in p for p in problems)

    def test_unexpected_engine_fails(self, vr):
        df = _valid_df(vr, engines=("TripoSR", "LGM", "TRELLIS", "BOGUS"))
        problems = vr.validate(df)
        assert any("expected engines" in p for p in problems)

    def test_uneven_counts_fail(self, vr):
        df = _valid_df(vr, per_engine=100)
        df = pd.concat([df, df[df["engine"] == "LGM"].head(5)], ignore_index=True)
        problems = vr.validate(df)
        assert any("expected 400 rows" in p for p in problems)


class TestNaN:
    def test_nan_in_key_metric_fails(self, vr):
        df = _valid_df(vr)
        df.loc[0, "chamfer_distance"] = float("nan")
        problems = vr.validate(df)
        assert any("NaN" in p and "chamfer_distance" in p for p in problems)

    def test_nan_in_psnr_fails(self, vr):
        df = _valid_df(vr)
        df.loc[5, "psnr"] = float("nan")
        problems = vr.validate(df)
        assert any("psnr" in p for p in problems)

    def test_nan_in_non_key_column_is_allowed(self, vr):
        df = _valid_df(vr)
        df.loc[0, "lpips"] = float("nan")  # lpips is not a KEY_METRIC
        assert vr.validate(df) == []


class TestSchema:
    def test_missing_column_fails(self, vr):
        df = _valid_df(vr).drop(columns=["psnr"])
        problems = vr.validate(df)
        assert any("missing required columns" in p for p in problems)

    def test_missing_column_short_circuits(self, vr):
        """Without required columns nothing else can be checked."""
        df = _valid_df(vr).drop(columns=["psnr", "engine"])
        problems = vr.validate(df)
        assert len(problems) == 1

    def test_duplicate_rows_fail(self, vr):
        df = _valid_df(vr)
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        problems = vr.validate(pd.concat([_valid_df(vr), df.iloc[[400]]], ignore_index=True))
        assert any("duplicate" in p for p in problems)


class TestCli:
    def test_passes_on_the_real_results(self):
        script = SCRIPT
        import subprocess

        r = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            cwd=str(SCRIPT.parents[1]),
        )
        assert r.returncode == 0, r.stdout + r.stderr

    def test_missing_file_exits_nonzero(self):
        import subprocess

        r = subprocess.run(
            [sys.executable, str(SCRIPT), "/tmp/does-not-exist.csv"],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 2
