#!/usr/bin/env python3
"""
Local re-evaluation of all exported *_mesh.obj files from Kaggle Batch 1 & 2.
Re-renders each mesh under the NEW upright camera (GSO Z-up -> Y-up, az=20/65,
el=15, fovy=40, dist=2.4) and computes updated PSNR/LPIPS vs GT novel view.

Self-contained - does not import torch or benchmark_runner.

Usage:
    python3 local_reevaluate.py [--output-dir /path/to/output]

Outputs:
    - local_reevaluation.csv  : per-object metrics under new camera
    - local_reevaluation_summary.tex : LaTeX summary table
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

REPO = Path("/home/anubhavanand/Desktop/reasearch papers")
KAGGLE_RUNS = REPO / "kaggle_runs"
OUTPUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "local_reeval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# GSO Z-up -> OpenGL Y-up rotation (Rx -90)
GSO_ZUP_TO_YUP_4X4 = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float32)

def bake_texture_to_vertices(tm):
    """Sample a trimesh object's UV texture map into per-vertex ColorVisuals."""
    try:
        from PIL import Image as _PILImage
        uv = getattr(tm.visual, "uv", None)
        material = getattr(tm.visual, "material", None)
        img = (
            getattr(material, "image", None) or getattr(material, "baseColorTexture", None)
        ) if material is not None else None
        if uv is None or img is None:
            return tm.visual
        if not isinstance(img, _PILImage.Image):
            img = _PILImage.fromarray(np.array(img))
        img = np.array(img.convert("RGB"))
        h, w = img.shape[:2]
        u = np.mod(np.asarray(uv[:, 0], dtype=np.float64), 1.0)
        v = 1.0 - np.mod(np.asarray(uv[:, 1], dtype=np.float64), 1.0)
        x = np.clip((u * (w - 1)).round().astype(int), 0, w - 1)
        y = np.clip((v * (h - 1)).round().astype(int), 0, h - 1)
        vcol = img[y, x].astype(np.float32) / 255.0
        import trimesh as _trimesh
        return _trimesh.visual.ColorVisuals(tm, vertex_colors=vcol)
    except Exception:
        return tm.visual

def render_mesh_upright(tm, azimuth_deg, elevation_deg, distance=2.4, resolution=256):
    """Render a trimesh under pyrender with upright camera."""
    import pyrender

    try:
        tm.fix_normals()
    except Exception:
        pass

    # Extract vertex colors
    vcols_uint8 = None
    if getattr(tm.visual, "vertex_colors", None) is not None:
        vc = np.asarray(tm.visual.vertex_colors, dtype=np.float32)
        if vc.max() <= 1.05:
            vc = np.clip(vc, 0.0, 1.0) * 255.0
        vcols_uint8 = np.clip(vc, 0, 255).astype(np.uint8)

    tm_render = trimesh.Trimesh(
        vertices=tm.vertices,
        faces=tm.faces,
        vertex_colors=vcols_uint8,
        process=False,
    )
    tm_render.fix_normals()

    scene = pyrender.Scene(
        bg_color=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        ambient_light=np.array([0.65, 0.65, 0.65], dtype=np.float32),
    )
    scene.add(pyrender.Mesh.from_trimesh(tm_render))

    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    cam_pos = np.array([
        distance * np.cos(el) * np.sin(az),
        distance * np.sin(el),
        distance * np.cos(el) * np.cos(az),
    ], dtype=np.float32)
    forward = -cam_pos / (np.linalg.norm(cam_pos) + 1e-8)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right /= (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = cam_pos

    camera = pyrender.PerspectiveCamera(yfov=np.radians(40.0))
    scene.add(camera, pose=pose)
    scene.add(
        pyrender.DirectionalLight(color=np.array([1.0, 1.0, 1.0]), intensity=1.8),
        pose=pose,
    )

    renderer = pyrender.OffscreenRenderer(resolution * 2, resolution * 2)
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
    renderer.delete()

    color = np.array(Image.fromarray(color).resize((resolution, resolution), Image.LANCZOS))
    return color

def compute_psnr(img1, img2):
    """Compute PSNR between two images."""
    if img1 is None or img2 is None:
        return float('nan')
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def find_qualitative_dir(batch_dir, engine_dir_name, engine_display_name, sample_id):
    """Find the qualitative directory for a given object."""
    outputs_dir = KAGGLE_RUNS / batch_dir / "outputs"
    for pattern in [
        outputs_dir / engine_dir_name / "qualitative" / sample_id,
        outputs_dir / "qualitative" / sample_id,
        outputs_dir / "qualitative" / engine_dir_name / sample_id,
    ]:
        if pattern.exists():
            return pattern
    return None

def find_file_case_insensitive(directory, filename_pattern):
    """Find a file matching pattern (case-insensitive) in directory."""
    if not directory or not directory.exists():
        return None
    for f in directory.iterdir():
        if f.name.lower() == filename_pattern.lower():
            return f
    return None

def main():
    print("=== Local Re-Evaluation of Exported Meshes ===")
    print()

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

    results = []

    for batch_dir, engines in BATCH_ENGINES.items():
        for engine_dir_name, engine_display_name in engines:
            outputs_dir = KAGGLE_RUNS / batch_dir / "outputs"
            qual_base = outputs_dir / engine_dir_name / "qualitative"
            if not qual_base.exists():
                print(f"[SKIP] {qual_base} not found")
                continue

            obj_dirs = sorted([d for d in qual_base.iterdir() if d.is_dir()])
            print(f"Processing {engine_display_name} ({len(obj_dirs)} objects)...")

            for obj_dir in obj_dirs:
                sample_id = obj_dir.name

                # Load the mesh
                mesh_path = find_file_case_insensitive(obj_dir, f"{engine_display_name}_mesh.obj")
                if mesh_path is None:
                    print(f"  [SKIP] {sample_id}: no mesh file")
                    continue

                try:
                    tm = trimesh.load(str(mesh_path), force="mesh")
                    if tm is None or len(tm.faces) == 0:
                        print(f"  [SKIP] {sample_id}: empty mesh")
                        continue

                    tm.visual = bake_texture_to_vertices(tm)
                    tm.apply_transform(GSO_ZUP_TO_YUP_4X4)

                    pts = np.asarray(tm.vertices, dtype=np.float32)
                    c = pts.mean(axis=0)
                    s = np.linalg.norm(pts - c, axis=1).max()
                    if s < 1e-8:
                        s = 1.0
                    tm.apply_translation(-c)
                    tm.apply_scale(1.0 / (s + 1e-8))

                    # Render input view (az=20, el=15)
                    img_input = render_mesh_upright(tm, 20.0, 15.0)

                    # Render novel view (az=65, el=15)
                    img_novel = render_mesh_upright(tm, 65.0, 15.0)

                    # Find GT novel view for PSNR comparison
                    gt_path = find_file_case_insensitive(obj_dir, "gt_novel.png")
                    img_gt = None
                    if gt_path:
                        img_gt = np.array(Image.open(gt_path).convert("RGB").resize((256, 256)))

                    psnr = compute_psnr(img_novel, img_gt) if img_gt is not None else float('nan')

                    result = {
                        "engine": engine_display_name,
                        "sample_id": sample_id,
                        "batch": batch_dir,
                        "mesh_path": str(mesh_path),
                        "psnr_new_camera": round(psnr, 4),
                        "has_input_render": img_input is not None,
                        "has_novel_render": img_novel is not None,
                        "has_gt_render": img_gt is not None,
                    }
                    results.append(result)

                    # Save renders
                    out_dir = OUTPUT_DIR / "renders" / engine_display_name / sample_id
                    out_dir.mkdir(parents=True, exist_ok=True)
                    if img_input is not None:
                        Image.fromarray(img_input).save(out_dir / "input_new_camera.png")
                    if img_novel is not None:
                        Image.fromarray(img_novel).save(out_dir / "novel_new_camera.png")
                    if img_gt is not None:
                        Image.fromarray(img_gt).save(out_dir / "gt_novel.png")

                    print(f"  {sample_id}: PSNR={psnr:.2f} dB")

                except Exception as e:
                    print(f"  [ERROR] {sample_id}: {e}")

    # Save CSV
    results_csv = OUTPUT_DIR / "local_reevaluation.csv"
    with open(results_csv, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print(f"\nSaved {len(results)} results to {results_csv}")

    # Save summary
    by_engine = defaultdict(list)
    for r in results:
        by_engine[r["engine"]].append(r)

    summary_lines = ["\\begin{table}[ht]", "\\centering",
                     "\\caption{Local re-evaluation under new upright camera}",
                     "\\label{tab:local_reeval}",
                     "\\begin{tabular}{lcccc}",
                     "\\toprule",
                     "Engine & N & Mean PSNR (dB) & With GT & % Complete \\\\",
                     "\\midrule"]

    for engine in ["TripoSR", "LGM", "TRELLIS", "SPAR3D"]:
        er = by_engine.get(engine, [])
        if not er:
            continue
        psnrs = [r["psnr_new_camera"] for r in er if r["psnr_new_camera"] == r["psnr_new_camera"]]
        n_gt = sum(1 for r in er if r["has_gt_render"])
        pct = 100.0 * len(er) / 100.0
        mean_psnr = sum(psnrs) / len(psnrs) if psnrs else float('nan')
        summary_lines.append(
            f"{engine} & {len(er)} & {mean_psnr:.2f} & {n_gt} & {pct:.0f}\\% \\\\"
        )

    summary_lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    summary_tex = OUTPUT_DIR / "local_reevaluation_summary.tex"
    summary_tex.write_text("\n".join(summary_lines))
    print(f"Saved LaTeX summary to {summary_tex}")
    print(f"\nOutput directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
