#!/usr/bin/env python3
"""
Comprehensive analysis of all 400 benchmark runs (4 engines × 100 objects).

For each run, this script:
1. Reads the result JSON from both Kaggle batches
2. Checks if input and output images exist in qualitative/ directories
3. Compares input vs output images pixel-by-pixel
4. Flags runs where output == input (model just passed through the input)
5. Writes a per-object analysis file with all details

Outputs:
    - analysis_400runs.csv        : one row per run with full analysis
    - analysis_400runs_summary.md : markdown summary
    - same_as_input_runs.txt      : list of runs where output==input
    - per_object/<engine>/<sample_id>/analysis.json : per-run detail
"""

import csv
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path("/home/anubhavanand/Desktop/reasearch papers/image-to-3d-benchmark")
KAGGLE_RUNS = Path("/home/anubhavanand/Desktop/reasearch papers/kaggle_runs")
OUTPUT_DIR = Path("/home/anubhavanand/Desktop/reasearch papers/analysis_400runs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Expected engine directories per batch (lowercase = dir name, capitalized = file prefix)
BATCH_ENGINES = {
    "batch1_100obj_v35": [
        ("triposr", "TripoSR"),
        ("lgm", "LGM"),
    ],
    "batch2_100obj_v35": [
        ("trellis", "TRELLIS"),
        ("spar3d", "SPAR3D"),
    ],
}

def load_json_results(batch_dir, engine_dir_name):
    """Load results JSON for a given batch and engine."""
    outputs_dir = KAGGLE_RUNS / batch_dir / "outputs"
    # Try both flat and nested paths
    paths = [
        outputs_dir / f"{engine_dir_name}_results.json",
        outputs_dir / engine_dir_name / f"{engine_dir_name}_results.json",
    ]
    for p in paths:
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None

def image_hash(img_path):
    """Compute MD5 hash of an image file."""
    if not img_path.exists():
        return None
    with open(img_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def images_identical(path1, path2):
    """Check if two images are pixel-identical."""
    if not path1.exists() or not path2.exists():
        return False, None, None
    h1 = image_hash(path1)
    h2 = image_hash(path2)
    if h1 is None or h2 is None:
        return False, h1, h2
    return h1 == h2, h1, h2

def images_similar(path1, path2, threshold=0.99):
    """Check if two images are visually similar (pixel correlation)."""
    if not path1.exists() or not path2.exists():
        return False, 0.0
    try:
        img1 = np.asarray(Image.open(path1).convert("RGB")).astype(np.float64)
        img2 = np.asarray(Image.open(path2).convert("RGB")).astype(np.float64)
        if img1.shape != img2.shape:
            return False, 0.0
        img1 = (img1 - img1.mean()) / (img1.std() + 1e-8)
        img2 = (img2 - img2.mean()) / (img2.std() + 1e-8)
        corr = np.mean(img1 * img2)
        return corr > threshold, float(corr)
    except Exception:
        return False, 0.0

def find_file_case_insensitive(directory, filename_pattern):
    """Find a file matching pattern (case-insensitive) in directory."""
    if not directory.exists():
        return None
    for f in directory.iterdir():
        if f.name.lower() == filename_pattern.lower():
            return f
    return None

def analyze_run(batch_dir, engine_dir_name, engine_display_name, result_row):
    """Analyze a single run result."""
    sample_id = result_row["sample_id"]
    
    # Find qualitative directory
    outputs_dir = KAGGLE_RUNS / batch_dir / "outputs"
    qual_dir = None
    for pattern in [
        outputs_dir / engine_dir_name / "qualitative" / sample_id,
        outputs_dir / "qualitative" / sample_id,
        outputs_dir / "qualitative" / engine_dir_name / sample_id,
    ]:
        if pattern.exists():
            qual_dir = pattern
            break
    
    analysis = {
        "engine": engine_display_name,
        "engine_dir": engine_dir_name,
        "sample_id": sample_id,
        "batch": batch_dir,
        "latency_s": result_row.get("latency_s"),
        "vram_peak_gb": result_row.get("vram_peak_gb"),
        "chamfer_distance": result_row.get("chamfer_distance"),
        "f_score_0.1": result_row.get("f_score_0.1"),
        "f_score_0.2": result_row.get("f_score_0.2"),
        "f_score_0.5": result_row.get("f_score_0.5"),
        "psnr": result_row.get("psnr"),
        "lpips": result_row.get("lpips"),
        "vertex_count": result_row.get("vertex_count"),
        "face_count": result_row.get("face_count"),
        "has_vertex_colors": None,
        "qual_dir_exists": qual_dir is not None,
        "qual_dir": str(qual_dir) if qual_dir else None,
        "input_exists": False,
        "novel_exists": False,
        "gt_novel_exists": False,
        "mesh_exists": False,
        "input_hash": None,
        "output_hash": None,
        "output_equals_input": False,
        "output_similarity_to_input": 0.0,
        "comment": "",
        "input_used": None,
    }
    
    # Extract has_vertex_colors from metadata
    metadata = result_row.get("metadata")
    if metadata:
        if isinstance(metadata, str):
            try:
                import ast
                metadata = ast.literal_eval(metadata)
            except:
                pass
        if isinstance(metadata, dict):
            analysis["has_vertex_colors"] = metadata.get("has_vertex_colors")
    
    if qual_dir:
        input_path = find_file_case_insensitive(qual_dir, "input.png")
        output_path = find_file_case_insensitive(qual_dir, f"{engine_display_name}_novel.png")
        gt_path = find_file_case_insensitive(qual_dir, "gt_novel.png")
        mesh_path = find_file_case_insensitive(qual_dir, f"{engine_display_name}_mesh.obj")
        
        analysis["input_exists"] = input_path is not None
        analysis["novel_exists"] = output_path is not None
        analysis["gt_novel_exists"] = gt_path is not None
        analysis["mesh_exists"] = mesh_path is not None
        analysis["input_hash"] = image_hash(input_path) if input_path else None
        analysis["output_hash"] = image_hash(output_path) if output_path else None
        analysis["input_used"] = str(input_path) if input_path else None
        
        # Check if output equals input
        if output_path and input_path:
            identical, h1, h2 = images_identical(input_path, output_path)
            analysis["output_equals_input"] = identical
            similar, corr = images_similar(input_path, output_path)
            analysis["output_similarity_to_input"] = corr
            
            if identical:
                analysis["comment"] = "OUTPUT IS IDENTICAL TO INPUT - model may have passed through input unchanged"
            elif corr > 0.95:
                analysis["comment"] = f"Output very similar to input (corr={corr:.3f}) - possible degenerate reconstruction"
            elif corr < 0.1:
                analysis["comment"] = "Output completely different from input - expected for 3D reconstruction"
            else:
                analysis["comment"] = f"Normal reconstruction (input-output corr={corr:.3f})"
    
    return analysis

def main():
    print("=== Comprehensive Analysis of 400 Benchmark Runs ===")
    print()
    
    all_analyses = []
    
    for batch_dir, engines in BATCH_ENGINES.items():
        for engine_dir_name, engine_display_name in engines:
            print(f"Loading {engine_display_name} results from {batch_dir}...", end=" ", flush=True)
            results = load_json_results(batch_dir, engine_dir_name)
            if results is None:
                print("NOT FOUND")
                continue
            print(f"{len(results)} rows")
            
            for row in results:
                analysis = analyze_run(batch_dir, engine_dir_name, engine_display_name, row)
                all_analyses.append(analysis)
                
                # Write per-object analysis
                obj_dir = OUTPUT_DIR / "per_object" / engine_display_name / row["sample_id"]
                obj_dir.mkdir(parents=True, exist_ok=True)
                with open(obj_dir / "analysis.json", "w") as f:
                    json.dump(analysis, f, indent=2)
                
                # Copy images if they exist
                if analysis["qual_dir_exists"]:
                    qual_dir = Path(analysis["qual_dir"])
                    for img_name in ["input.png", f"{engine_display_name}_novel.png", "gt_novel.png"]:
                        src = find_file_case_insensitive(qual_dir, img_name)
                        if src:
                            shutil.copy2(src, obj_dir / img_name)
    
    print(f"\nTotal runs analyzed: {len(all_analyses)}")
    
    # Save CSV
    csv_path = OUTPUT_DIR / "analysis_400runs.csv"
    if all_analyses:
        fieldnames = all_analyses[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_analyses)
        print(f"Saved CSV to {csv_path}")
    
    # Find same-as-input runs
    same_as_input = [a for a in all_analyses if a["output_equals_input"]]
    if same_as_input:
        same_path = OUTPUT_DIR / "same_as_input_runs.txt"
        with open(same_path, "w") as f:
            for a in same_as_input:
                f.write(f"{a['engine']}\t{a['sample_id']}\tbatch={a['batch']}\tPSNR={a['psnr']}\tCD={a['chamfer_distance']}\n")
        print(f"\n⚠️  {len(same_as_input)} runs have output identical to input:")
        for a in same_as_input[:20]:
            print(f"   {a['engine']}/{a['sample_id']} (PSNR={a['psnr']}, CD={a['chamfer_distance']})")
    else:
        print("\n✅ No runs have output identical to input")
    
    # Summary statistics
    print("\n=== PER-ENGINE SUMMARY ===")
    for engine_display_name in ["TripoSR", "LGM", "TRELLIS", "SPAR3D"]:
        er = [a for a in all_analyses if a["engine"] == engine_display_name]
        if not er:
            continue
        cds = [a["chamfer_distance"] for a in er if a.get("chamfer_distance") is not None]
        psnrs = [a["psnr"] for a in er if a.get("psnr") is not None]
        latencies = [a["latency_s"] for a in er if a.get("latency_s") is not None]
        same_count = sum(1 for a in er if a["output_equals_input"])
        n_input = sum(1 for a in er if a["input_exists"])
        n_novel = sum(1 for a in er if a["novel_exists"])
        n_mesh = sum(1 for a in er if a["mesh_exists"])
        print(f"  {engine_display_name}: n={len(er)}, CD_mean={sum(cds)/len(cds):.4f}, PSNR_mean={sum(psnrs)/len(psnrs):.2f}, Lat_mean={sum(latencies)/len(latencies):.2f}s")
        print(f"    Files: input={n_input}, novel={n_novel}, mesh={n_mesh}, same_as_input={same_count}")
    
    # Save markdown summary
    md_path = OUTPUT_DIR / "analysis_400runs_summary.md"
    lines = ["# 400-Run Benchmark Analysis", ""]
    lines.append(f"Total runs: {len(all_analyses)}")
    lines.append(f"Same-as-input runs: {len(same_as_input)}")
    lines.append("")
    lines.append("## Per-Engine Summary")
    lines.append("")
    lines.append("| Engine | N | Mean CD | Mean PSNR | Mean Latency (s) | Same-as-Input |")
    lines.append("|--------|---|---------|-----------|------------------|---------------|")
    for engine_display_name in ["TripoSR", "LGM", "TRELLIS", "SPAR3D"]:
        er = [a for a in all_analyses if a["engine"] == engine_display_name]
        if not er:
            continue
        cds = [a["chamfer_distance"] for a in er if a.get("chamfer_distance") is not None]
        psnrs = [a["psnr"] for a in er if a.get("psnr") is not None]
        latencies = [a["latency_s"] for a in er if a.get("latency_s") is not None]
        same_count = sum(1 for a in er if a["output_equals_input"])
        lines.append(f"| {engine_display_name} | {len(er)} | {sum(cds)/len(cds):.4f} | {sum(psnrs)/len(psnrs):.2f} | {sum(latencies)/len(latencies):.2f} | {same_count} |")
    
    md_path.write_text("\n".join(lines))
    print(f"\nSaved markdown summary to {md_path}")
    print(f"\nAll output in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()