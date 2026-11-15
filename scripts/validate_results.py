#!/usr/bin/env python3
"""Validate a results CSV against the benchmark's publication contract.

DATA-10: nothing checked row counts, engine names or required columns before
publication, so a partial run could be merged and published as complete.

Usage:
    python scripts/validate_results.py [path/to/results.csv]

Exits non-zero on any violation, so it can gate CI or a release.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "engine",
    "sample_id",
    "latency_s",
    "vram_peak_gb",
    "chamfer_distance",
    "f_score_0.1",
    "f_score_0.2",
    "f_score_0.5",
    "psnr",
    "lpips",
]

EXPECTED_ENGINES = {"TripoSR", "LGM", "TRELLIS", "SPAR3D"}
EXPECTED_ROWS = 400
# key metric columns that must not be NaN in a complete run
KEY_METRICS = ["chamfer_distance", "f_score_0.1", "f_score_0.2", "psnr"]


def validate(df: pd.DataFrame, *, expect_complete: bool = True) -> list[str]:
    """Return a list of contract violations (empty means valid)."""
    problems: list[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"missing required columns: {missing}")
        return problems  # nothing else can be checked reliably

    if expect_complete:
        if len(df) != EXPECTED_ROWS:
            problems.append(f"expected {EXPECTED_ROWS} rows, found {len(df)}")

        found = set(df["engine"].unique())
        if found != EXPECTED_ENGINES:
            problems.append(f"expected engines {sorted(EXPECTED_ENGINES)}, found {sorted(found)}")

        for engine, count in df["engine"].value_counts().items():
            if count != EXPECTED_ROWS // len(EXPECTED_ENGINES):
                problems.append(
                    f"engine {engine} has {count} rows, expected {EXPECTED_ROWS // len(EXPECTED_ENGINES)}"
                )

        for col in KEY_METRICS:
            n_nan = int(df[col].isna().sum())
            if n_nan:
                problems.append(f"{n_nan} NaN value(s) in {col}")

    dupes = df.duplicated(subset=["engine", "sample_id"]).sum()
    if dupes:
        problems.append(f"{dupes} duplicate (engine, sample_id) row(s)")

    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default="results/unified_results_4engines_100obj.csv")
    parser.add_argument(
        "--partial",
        action="store_true",
        help="skip row-count/engine/NaN checks (for an intentionally incomplete run)",
    )
    args = parser.parse_args(argv)

    path = Path(args.csv)
    if not path.exists():
        print(f"FAIL: {path} not found")
        return 2

    df = pd.read_csv(path)
    problems = validate(df, expect_complete=not args.partial)

    if problems:
        print(f"FAIL: {path}")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(f"PASS: {path} ({len(df)} rows, {df['engine'].nunique()} engines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
