#!/usr/bin/env python3
"""
Visual analysis of all 400 runs without requiring vision model.
Uses image processing to detect:
1. PASS_THROUGH: Output is identical or near-identical to input (SSIM > 0.95)
2. DEGENERATE: Output is blank, grey, or corrupted (low variance, low color)
3. GOOD: Output has 3D shading cues (depth gradients, different viewpoint)
4. PARTIAL: Some 3D structure but with artifacts

Also checks:
- Color histogram correlation between input and output
- Edge density (3D renders have different edge patterns than flat input)
- Shading gradient (3D renders have smooth lighting gradients)
"""

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("[WARN] scikit-image not installed, using fallback SSIM")

KAGGLE_RUNS = Path("/home/anubhavanand/Desktop/reasearch papers/kaggle_runs")
OUTPUT_DIR = Path("/home/anubhavanand/Desktop/reasearch papers/analysis_400runs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

def find_file_case_insensitive(directory, filename_pattern):
    if not directory or not directory.exists():
        return None
    for f in directory.iterdir():
        if f.name.lower() == filename_pattern.lower():
            return f
    return None

def load_image(path, size=(256, 256)):
    """Load image as RGB numpy array."""
    if path is None or not path.exists():
        return None
    img = Image.open(path).convert("RGB").resize(size, Image.LANCZOS)
    return np.array(img, dtype=np.float64)

def compute_ssim(img1, img2):
    """Compute SSIM between two images."""
    if img1 is None or img2 is None:
        return 0.0
    if HAS_SKIMAGE:
        return float(ssim(img1.astype(np.uint8), img2.astype(np.uint8), channel_axis=2))
    else:
        # Fallback: normalized cross-correlation
        i1 = (img1 - img1.mean()) / (img1.std() + 1e-8)
        i2 = (img2 - img2.mean()) / (img2.std() + 1e-8)
        return float(np.mean(i1 * i2))

def compute_histogram_correlation(img1, img2):
    """Compute color histogram correlation."""
    if img1 is None or img2 is None:
        return 0.0
    h1 = np.histogram(img1[:, :, 0], bins=32, range=(0, 255))[0].astype(float)
    h2 = np.histogram(img2[:, :, 0], bins=32, range=(0, 255))[0].astype(float)
    h1 /= (h1.sum() + 1e-8)
    h2 /= (h2.sum() + 1e-8)
    return float(np.corrcoef(h1, h2)[0, 1])

def compute_shading_gradient(img):
    """Compute average shading gradient magnitude (3D renders have smooth gradients)."""
    if img is None:
        return 0.0
    gray = np.mean(img, axis=2)
    dy = np.abs(np.diff(gray, axis=0)).mean()
    dx = np.abs(np.diff(gray, axis=1)).mean()
    return float((dx + dy) / 2.0)

def compute_edge_density(img):
    """Compute edge density (Canny-like approximation)."""
    if img is None:
        return 0.0
    gray = np.mean(img, axis=2)
    dy = np.abs(np.diff(gray, axis=0))  # shape (H-1, W)
    dx = np.abs(np.diff(gray, axis=1))  # shape (H, W-1)
    # Pad to same shape for comparison
    dy_padded = np.zeros_like(gray)
    dy_padded[:-1, :] = dy
    dx_padded = np.zeros_like(gray)
    dx_padded[:, :-1] = dx
    edges = (dy_padded > 15) | (dx_padded > 15)
    return float(edges.mean())

def compute_colorfulness(img):
    """Compute colorfulness metric (Hasler & Süsstrunk 2003)."""
    if img is None:
        return 0.0
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    rg = R - G
    yb = 0.5 * (R + G) - B
    return float(np.sqrt(rg.std()**2 + yb.std()**2) + 0.3 * np.sqrt(rg.mean()**2 + yb.mean()**2))

def classify_run(input_img, output_img):
    """Classify a run based on visual characteristics."""
    if input_img is None or output_img is None:
        return "MISSING", 0.0, 0.0, 0.0
    
    ssim_val = compute_ssim(input_img, output_img)
    hist_corr = compute_histogram_correlation(input_img, output_img)
    output_colorfulness = compute_colorfulness(output_img)
    output_shading = compute_shading_gradient(output_img)
    output_edges = compute_edge_density(output_img)
    
    # Check for degenerate (blank/grey) output
    if output_colorfulness < 2.0 and output_shading < 1.0:
        return "DEGENERATE", ssim_val, hist_corr, output_colorfulness
    
    # Check for pass-through (output very similar to input)
    if ssim_val > 0.90:
        return "PASS_THROUGH", ssim_val, hist_corr, output_colorfulness
    
    # Check for good 3D reconstruction (different from input but has 3D cues)
    if ssim_val < 0.70 and output_colorfulness > 5.0 and output_shading > 1.5:
        return "GOOD", ssim_val, hist_corr, output_colorfulness
    
    # Default: partial
    return "PARTIAL", ssim_val, hist_corr, output_colorfulness

def analyze_all_runs():
    """Analyze all 400 runs."""
    results = []
    categories = defaultdict(lambda: defaultdict(int))
    
    for batch_dir, engines in BATCH_ENGINES.items():
        for engine_dir_name, engine_display_name in engines:
            outputs_dir = KAGGLE_RUNS / batch_dir / "outputs"
            qual_base = outputs_dir / engine_dir_name / "qualitative"
            
            if not qual_base.exists():
                print(f"[SKIP] {qual_base} not found")
                continue
            
            obj_dirs = sorted([d for d in qual_base.iterdir() if d.is_dir()])
            print(f"Analyzing {engine_display_name} ({len(obj_dirs)} objects)...")
            
            for obj_dir in obj_dirs:
                sample_id = obj_dir.name
                
                input_path = find_file_case_insensitive(obj_dir, "input.png")
                output_path = find_file_case_insensitive(obj_dir, f"{engine_display_name}_novel.png")
                gt_path = find_file_case_insensitive(obj_dir, "gt_novel.png")
                
                input_img = load_image(input_path)
                output_img = load_image(output_path)
                
                category, ssim_val, hist_corr, colorfulness = classify_run(input_img, output_img)
                categories[engine_display_name][category] += 1
                
                results.append({
                    "engine": engine_display_name,
                    "sample_id": sample_id,
                    "batch": batch_dir,
                    "category": category,
                    "ssim": round(ssim_val, 4),
                    "hist_correlation": round(hist_corr, 4),
                    "output_colorfulness": round(colorfulness, 2),
                    "input_exists": input_path is not None,
                    "output_exists": output_path is not None,
                    "gt_exists": gt_path is not None,
                })
    
    return results, categories

def main():
    print("=== Visual Analysis of 400 Runs (Programmatic) ===")
    print()
    
    results, categories = analyze_all_runs()
    
    # Save CSV
    csv_path = OUTPUT_DIR / "visual_analysis_400runs.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {len(results)} results to {csv_path}")
    
    # Print summary
    print("\n=== CATEGORY SUMMARY ===")
    for engine in ["TripoSR", "LGM", "TRELLIS", "SPAR3D"]:
        counts = categories.get(engine, {})
        total = sum(counts.values())
        if total == 0:
            continue
        print(f"\n  {engine} ({total} objects):")
        for cat in ["GOOD", "PARTIAL", "PASS_THROUGH", "DEGENERATE", "MISSING"]:
            count = counts.get(cat, 0)
            if count > 0:
                print(f"    {cat}: {count} ({100*count/total:.1f}%)")
    
    # List problematic runs
    problematic = [r for r in results if r["category"] in ("PASS_THROUGH", "DEGENERATE", "MISSING")]
    if problematic:
        print(f"\n⚠️  {len(problematic)} problematic runs found:")
        for r in problematic[:30]:
            print(f"   {r['engine']}/{r['sample_id']}: {r['category']} (SSIM={r['ssim']:.3f})")
    else:
        print("\n✅ No problematic runs found - all outputs are valid 3D reconstructions")
    
    # Save markdown report
    md_path = OUTPUT_DIR / "VISION_ANALYSIS_REPORT.md"
    lines = ["# Visual Analysis Report — 400 Benchmark Runs", ""]
    lines.append("## Methodology")
    lines.append("")
    lines.append("Each run was analyzed programmatically using:")
    lines.append("- **SSIM** (Structural Similarity Index) — measures if output is identical to input")
    lines.append("- **Color histogram correlation** — measures color distribution similarity")
    lines.append("- **Colorfulness** — detects blank/grey/degenerate outputs")
    lines.append("- **Shading gradient** — detects 3D lighting cues (smooth gradients)")
    lines.append("- **Edge density** — measures structural complexity")
    lines.append("")
    lines.append("## Classification Criteria")
    lines.append("")
    lines.append("| Category | Criteria |")
    lines.append("|----------|----------|")
    lines.append("| `PASS_THROUGH` | SSIM > 0.90 (output nearly identical to input) |")
    lines.append("| `DEGENERATE` | Colorfulness < 2.0 AND shading < 1.0 (blank/grey) |")
    lines.append("| `GOOD` | SSIM < 0.70 AND colorfulness > 5.0 AND shading > 1.5 |")
    lines.append("| `PARTIAL` | Everything else (some 3D structure but artifacts) |")
    lines.append("")
    lines.append("## Results Summary")
    lines.append("")
    lines.append("| Engine | Total | GOOD | PARTIAL | PASS_THROUGH | DEGENERATE |")
    lines.append("|--------|-------|------|---------|--------------|------------|")
    for engine in ["TripoSR", "LGM", "TRELLIS", "SPAR3D"]:
        counts = categories.get(engine, {})
        total = sum(counts.values())
        if total == 0:
            continue
        lines.append(f"| {engine} | {total} | {counts.get('GOOD',0)} | {counts.get('PARTIAL',0)} | {counts.get('PASS_THROUGH',0)} | {counts.get('DEGENERATE',0)} |")
    
    lines.append("")
    lines.append("## Problematic Runs")
    lines.append("")
    if problematic:
        lines.append(f"**{len(problematic)} problematic runs found:**")
        lines.append("")
        for r in problematic:
            lines.append(f"- `{r['engine']}/{r['sample_id']}` — {r['category']} (SSIM={r['ssim']:.3f})")
    else:
        lines.append("✅ **No problematic runs found** — all 400 outputs are valid 3D reconstructions")
    
    lines.append("")
    lines.append("## Key Findings")
    lines.append("")
    
    # Calculate overall stats
    all_counts = defaultdict(int)
    for engine_counts in categories.values():
        for cat, count in engine_counts.items():
            all_counts[cat] += count
    
    total_runs = sum(all_counts.values())
    lines.append(f"- **Total runs analyzed:** {total_runs}")
    lines.append(f"- **GOOD (proper 3D reconstruction):** {all_counts.get('GOOD',0)} ({100*all_counts.get('GOOD',0)/total_runs:.1f}%)")
    lines.append(f"- **PARTIAL (some artifacts):** {all_counts.get('PARTIAL',0)} ({100*all_counts.get('PARTIAL',0)/total_runs:.1f}%)")
    lines.append(f"- **PASS_THROUGH (identical to input):** {all_counts.get('PASS_THROUGH',0)} ({100*all_counts.get('PASS_THROUGH',0)/total_runs:.1f}%)")
    lines.append(f"- **DEGENERATE (blank/grey):** {all_counts.get('DEGENERATE',0)} ({100*all_counts.get('DEGENERATE',0)/total_runs:.1f}%)")
    
    md_path.write_text("\n".join(lines))
    print(f"\nSaved report to {md_path}")
    print(f"\nAll output in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()