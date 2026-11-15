#!/usr/bin/env python3
"""
Merge Batch 1 (TripoSR + LGM) and Batch 2 (TRELLIS + SPAR3D) results into:
1. unified_4models.csv
2. unified_4models_table.tex (IEEE publication table)
3. pareto_4models.png (Speed vs F-Score & VRAM Pareto plot)
"""

import json
import sys
from pathlib import Path

import pandas as pd

# TOOL-09: fields that must agree when the same (engine, sample_id) appears twice.
_COMPARED_FIELDS = (
    "chamfer_distance",
    "f_score_0.1",
    "f_score_0.2",
    "f_score_0.5",
    "psnr",
    "lpips",
    "latency_s",
    "vram_peak_gb",
)


def _differs(a, b) -> bool:
    """True when two duplicate rows disagree on a metric that matters."""
    for field in _COMPARED_FIELDS:
        va, vb = a.get(field), b.get(field)
        if va is None or vb is None:
            if va is not vb:
                return True
            continue
        try:
            if abs(float(va) - float(vb)) > 1e-9:
                return True
        except (TypeError, ValueError):
            if va != vb:
                return True
    return False


def merge_all(search_dirs):
    rows = []
    seen = {}
    conflicts = []
    for d in search_dirs:
        for p in sorted(Path(d).rglob("*_results.json")):
            try:
                data = json.load(open(p))
                for r in data:
                    key = (r.get("engine"), r.get("sample_id"))
                    if "chamfer_distance" not in r:
                        continue
                    if key in seen:
                        # TOOL-09: keeping the first row made the winner depend on
                        # directory order, so a stale run could override the final
                        # one. Compare instead of silently dropping.
                        if _differs(seen[key], r):
                            conflicts.append((key, p))
                        continue
                    seen[key] = r
                    rows.append(r)
                print(f"[loaded] {p} ({len(data)} rows)")
            except Exception as e:
                print(f"[skip] {p}: {e}")

    if conflicts:
        print(f"[error] {len(conflicts)} conflicting duplicate row(s):")
        for (engine, sample_id), path in conflicts[:10]:
            print(f"  - {engine}/{sample_id} also in {path}")
        sys.exit(1)
    if not rows:
        print("No result JSONs found.")
        return

    out_dir = Path(search_dirs[0])
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "unified_4models.csv", index=False)

    cols = [
        "chamfer_distance",
        "f_score_0.1",
        "f_score_0.2",
        "f_score_0.5",
        "psnr",
        "lpips",
        "latency_s",
        "vram_peak_gb",
    ]
    avail = [c for c in cols if c in df.columns]
    summary = df.groupby("engine")[avail].agg(["mean", "std"]).round(4)
    print("\n=== 4-MODEL UNIFIED BENCHMARK SUMMARY (mean +/- std) ===")
    print(summary.to_string())

    tex_path = out_dir / "unified_4models_table.tex"
    tex_path.write_text(summary.reset_index().to_latex(index=False, na_rep="--"))
    print(f"\nSaved CSV and LaTeX table to {out_dir}")


if __name__ == "__main__":
    dirs = sys.argv[1:] or ["./outputs"]
    merge_all(dirs)
