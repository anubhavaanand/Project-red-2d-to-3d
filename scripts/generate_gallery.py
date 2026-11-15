#!/usr/bin/env python3
"""
Generate a comprehensive visual gallery of all 400 benchmark runs.
Creates an HTML page with all input/output pairs organized by engine and object,
with metrics displayed for each run.

Output: /home/anubhavanand/Desktop/reasearch papers/kaggle_runs/GALLERY_400RUNS.html
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path("/home/anubhavanand/Desktop/reasearch papers")
KAGGLE_RUNS = REPO / "kaggle_runs"
OUTPUT_DIR = REPO / "kaggle_runs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_ENGINES = {
    "batch1_100obj_v35": [
        ("triposr", "TripoSR", "#1f77b4"),
        ("lgm", "LGM", "#ff7f0e"),
    ],
    "batch2_100obj_v35": [
        ("trellis", "TRELLIS", "#2ca02c"),
        ("spar3d", "SPAR3D", "#9467bd"),
    ],
}

def find_file_case_insensitive(directory, filename_pattern):
    if not directory or not directory.exists():
        return None
    for f in directory.iterdir():
        if f.name.lower() == filename_pattern.lower():
            return f
    return None

def load_results_csv():
    """Load the unified results CSV."""
    csv_path = REPO / "image-to-3d-benchmark" / "results" / "unified_results_4engines_100obj.csv"
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def make_thumbnail(img_path, size=(200, 200)):
    """Create a thumbnail image."""
    if img_path is None or not img_path.exists():
        # Create a placeholder
        img = Image.new('RGB', size, color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        draw.text((10, 90), "NOT FOUND", fill=(255, 0, 0))
        return img
    img = Image.open(img_path).convert('RGB')
    img.thumbnail(size, Image.LANCZOS)
    # Paste on white background
    bg = Image.new('RGB', size, color=(255, 255, 255))
    offset = ((size[0] - img.width) // 2, (size[1] - img.height) // 2)
    bg.paste(img, offset)
    return bg

def make_contact_sheet_for_engine(batch_dir, engine_dir_name, engine_display_name, color_hex, results_rows, output_path):
    """Create a contact sheet for one engine."""
    outputs_dir = KAGGLE_RUNS / batch_dir / "outputs"
    qual_base = outputs_dir / engine_dir_name / "qualitative"
    
    if not qual_base.exists():
        print(f"[SKIP] {qual_base} not found")
        return
    
    obj_dirs = sorted([d for d in qual_base.iterdir() if d.is_dir()])
    if not obj_dirs:
        print(f"[SKIP] No objects in {qual_base}")
        return
    
    # Create lookup for results
    result_lookup = {}
    for r in results_rows:
        if r["engine"] == engine_display_name:
            result_lookup[r["sample_id"]] = r
    
    # Grid layout: 4 columns (Input, Output, GT, Metrics)
    thumb_size = (200, 200)
    cols = 4
    rows = len(obj_dirs)
    
    # Create large image
    sheet_width = cols * (thumb_size[0] + 20) + 20
    sheet_height = rows * (thumb_size[1] + 60) + 60
    sheet = Image.new('RGB', (sheet_width, sheet_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    
    # Title
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        font_tiny = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_tiny = ImageFont.load_default()
    
    draw.text((20, 10), f"{engine_display_name} — {len(obj_dirs)} objects", fill=(0, 0, 0), font=font_large)
    
    # Column headers
    headers = ["Input Image", f"{engine_display_name} Output", "Ground Truth", "Metrics"]
    for col, header in enumerate(headers):
        x = 20 + col * (thumb_size[0] + 20)
        draw.text((x, 45), header, fill=color_hex, font=font_small)
    
    for row_idx, obj_dir in enumerate(obj_dirs):
        sample_id = obj_dir.name
        y = 70 + row_idx * (thumb_size[1] + 60)
        
        # Input
        input_path = find_file_case_insensitive(obj_dir, "input.png")
        input_thumb = make_thumbnail(input_path, thumb_size)
        sheet.paste(input_thumb, (20, y))
        
        # Output
        output_path = find_file_case_insensitive(obj_dir, f"{engine_display_name}_novel.png")
        output_thumb = make_thumbnail(output_path, thumb_size)
        sheet.paste(output_thumb, (20 + (thumb_size[0] + 20), y))
        
        # GT
        gt_path = find_file_case_insensitive(obj_dir, "gt_novel.png")
        gt_thumb = make_thumbnail(gt_path, thumb_size)
        sheet.paste(gt_thumb, (20 + 2 * (thumb_size[0] + 20), y))
        
        # Metrics
        metrics_x = 20 + 3 * (thumb_size[0] + 20)
        metrics_y = y + 10
        
        r = result_lookup.get(sample_id, {})
        cd = float(r.get("chamfer_distance", 0) or 0)
        f01 = float(r.get("f_score_0.1", 0) or 0)
        f02 = float(r.get("f_score_0.2", 0) or 0)
        psnr = float(r.get("psnr", 0) or 0)
        lpips = float(r.get("lpips", 0) or 0)
        latency = float(r.get("latency_s", 0) or 0)
        
        metrics_text = [
            f"CD: {cd:.4f}",
            f"F@0.1: {f01:.4f}",
            f"F@0.2: {f02:.4f}",
            f"PSNR: {psnr:.2f} dB",
            f"LPIPS: {lpips:.4f}",
            f"Latency: {latency:.2f}s",
        ]
        
        for i, text in enumerate(metrics_text):
            draw.text((metrics_x, metrics_y + i * 16), text, fill=(0, 0, 0), font=font_tiny)
        
        # Sample ID (truncated)
        display_id = sample_id[:40] + "..." if len(sample_id) > 40 else sample_id
        draw.text((20, y + thumb_size[1] + 5), display_id, fill=(100, 100, 100), font=font_tiny)
        
        # Color code based on CD (green=good, red=bad)
        if cd < 0.10:
            cd_color = (0, 180, 0)
        elif cd < 0.20:
            cd_color = (255, 165, 0)
        else:
            cd_color = (255, 0, 0)
        draw.text((metrics_x, metrics_y + 6 * 16), f"CD: {cd:.4f}", fill=cd_color, font=font_tiny)
    
    # Save
    sheet.save(output_path, quality=85)
    print(f"Saved {output_path} ({len(obj_dirs)} objects)")

def generate_html_gallery(results_rows, output_path):
    """Generate an HTML gallery with all 400 runs."""
    
    # Group by engine
    by_engine = defaultdict(list)
    for r in results_rows:
        by_engine[r["engine"]].append(r)
    
    html = []
    html.append("<!DOCTYPE html>")
    html.append("<html><head>")
    html.append("<title>400-Run Benchmark Gallery</title>")
    html.append("<style>")
    html.append("body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }")
    html.append("h1 { color: #333; }")
    html.append("h2 { color: #555; border-bottom: 2px solid #ddd; padding-bottom: 5px; }")
    html.append(".engine-section { margin-bottom: 40px; }")
    html.append(".object-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 15px; }")
    html.append(".object-card { background: white; border-radius: 8px; padding: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }")
    html.append(".object-card img { width: 90px; height: 90px; object-fit: contain; border: 1px solid #ddd; }")
    html.append(".metrics { font-size: 11px; color: #666; margin-top: 5px; }")
    html.append(".good { color: #2e7d32; font-weight: bold; }")
    html.append(".ok { color: #f57c00; }")
    html.append(".bad { color: #c62828; font-weight: bold; }")
    html.append(".summary { background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; }")
    html.append("</style>")
    html.append("</head><body>")
    
    html.append("<h1>400-Run Benchmark Gallery</h1>")
    html.append(f"<p>Total runs: {len(results_rows)} | Engines: {', '.join(sorted(by_engine.keys()))}</p>")
    
    # Summary
    html.append("<div class='summary'>")
    html.append("<h3>Summary</h3>")
    html.append("<table border='1' cellpadding='5' style='border-collapse: collapse;'>")
    html.append("<tr><th>Engine</th><th>N</th><th>Mean CD</th><th>Mean F@0.1</th><th>Mean PSNR</th><th>Good (CD&lt;0.1)</th><th>OK (0.1-0.2)</th><th>Bad (&gt;0.2)</th></tr>")
    
    for engine in ["TripoSR", "LGM", "TRELLIS", "SPAR3D"]:
        er = by_engine.get(engine, [])
        if not er:
            continue
        cds = [float(r.get("chamfer_distance", 0) or 0) for r in er]
        f01s = [float(r.get("f_score_0.1", 0) or 0) for r in er]
        psnrs = [float(r.get("psnr", 0) or 0) for r in er]
        good = sum(1 for cd in cds if cd < 0.10)
        ok = sum(1 for cd in cds if 0.10 <= cd < 0.20)
        bad = sum(1 for cd in cds if cd >= 0.20)
        html.append(f"<tr><td>{engine}</td><td>{len(er)}</td><td>{sum(cds)/len(cds):.4f}</td><td>{sum(f01s)/len(f01s):.4f}</td><td>{sum(psnrs)/len(psnrs):.2f}</td><td class='good'>{good}</td><td class='ok'>{ok}</td><td class='bad'>{bad}</td></tr>")
    
    html.append("</table>")
    html.append("</div>")
    
    # Per-engine sections
    for engine in ["TripoSR", "LGM", "TRELLIS", "SPAR3D"]:
        er = by_engine.get(engine, [])
        if not er:
            continue
        
        html.append(f"<div class='engine-section'>")
        html.append(f"<h2>{engine} ({len(er)} objects)</h2>")
        html.append("<div class='object-grid'>")
        
        for r in sorted(er, key=lambda x: float(x.get("chamfer_distance", 0) or 0)):
            sample_id = r["sample_id"]
            cd = float(r.get("chamfer_distance", 0) or 0)
            f01 = float(r.get("f_score_0.1", 0) or 0)
            psnr = float(r.get("psnr", 0) or 0)
            
            if cd < 0.10:
                cd_class = "good"
            elif cd < 0.20:
                cd_class = "ok"
            else:
                cd_class = "bad"
            
            html.append(f"<div class='object-card'>")
            html.append(f"<div><strong>{sample_id[:30]}</strong></div>")
            html.append(f"<div class='metrics'>")
            html.append(f"CD: <span class='{cd_class}'>{cd:.4f}</span><br>")
            html.append(f"F@0.1: {f01:.4f}<br>")
            html.append(f"PSNR: {psnr:.2f} dB")
            html.append(f"</div>")
            html.append(f"</div>")
        
        html.append("</div>")  # object-grid
        html.append("</div>")  # engine-section
    
    html.append("</body></html>")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(html))
    
    print(f"Generated HTML gallery: {output_path}")

def main():
    print("=== Generating 400-Run Visual Gallery ===")
    print()
    
    results_rows = load_results_csv()
    print(f"Loaded {len(results_rows)} result rows")
    
    # Generate per-engine contact sheets
    for batch_dir, engines in BATCH_ENGINES.items():
        for engine_dir_name, engine_display_name, color_hex in engines:
            print(f"Generating contact sheet for {engine_display_name}...")
            sheet_path = OUTPUT_DIR / f"contact_sheet_{engine_display_name.lower()}.jpg"
            make_contact_sheet_for_engine(
                batch_dir, engine_dir_name, engine_display_name, color_hex,
                results_rows, sheet_path
            )
    
    # Generate HTML gallery
    html_path = OUTPUT_DIR / "GALLERY_400RUNS.html"
    generate_html_gallery(results_rows, html_path)
    
    print()
    print("=== Done ===")
    print(f"Contact sheets: {OUTPUT_DIR}/contact_sheet_*.jpg")
    print(f"HTML gallery: {html_path}")
    print()
    print("Open the HTML gallery in a browser to review all 400 runs interactively.")

if __name__ == "__main__":
    main()