# Results

This directory holds the measured benchmark data. Keep the canonical files; treat the rest as run history.

## Canonical files

| File | Rows | What it is |
|---|---:|---|
| `unified_results_4engines.csv` | 40 | Canonical-10 subset: 10 GSO objects × 4 engines, filtered from the 400-row run. Not 40 additional evaluations. |
| `unified_results_4engines_100obj.csv` | 400 | Full run: 100 objects × 4 engines (the paper's n = 100 block) |
| `unified_table_4engines.tex` | — | LaTeX table generated from the canonical-10 CSV |
| `unified_table_4engines_100obj.tex` | — | LaTeX table generated from the n = 100 CSV |

Both CSVs are derived from the same final 400 evaluations; the 40-row file is a filtered canonical-10 subset. They are the source of every number in the manuscript.

## Mesh manifest

`mesh_manifest.csv` lists every exported `*_mesh.obj`: 400 rows (100 objects × 4 engines) with size, SHA-256, vertex and face counts.

**The `path` column is relative to the Kaggle run root, which is not in this repository.** Paths look like `batch1_100obj_v35/outputs/...`, and neither `batch1_100obj_v35/` nor `batch2_100obj_v35/` exists here. To resolve a path, download the run tree and join it:

```python
run_root = Path("/path/to/kaggle_runs")   # contains batch1_100obj_v35/ and batch2_100obj_v35/
mesh = run_root / row["path"]
```

The `run_root` column names the top-level directory each row belongs to, so the expected root is explicit without opening the CSV.

## Provenance of the n = 100 block

Both final kernels ran at code commit `09131e61568d98401e71861bc27fdce6637b2f20` (each output directory contains a `code_sha.txt`). The run directories are named `batch1_100obj_v35` and `batch2_100obj_v35`; both produced 100 rows per engine (400 unique evaluations, 0 failures). The 10-object block in the paper is the canonical subset of those 400 rows, not a separate 40-evaluation run.

## Reproducing

The canonical CSVs were built from the final Kaggle run outputs, which live outside this repository:

```bash
python scripts/merge_batches.py ../kaggle_runs/batch1_100obj_v35/outputs \
                                 ../kaggle_runs/batch2_100obj_v35/outputs
python scripts/rebuild_figures.py
```

`merge_batches.py` collects `*_results.json` files, de-duplicates by `(engine, sample_id)`, and writes the unified CSV/TeX plus the Pareto figure. The canonical `results/` directory now contains only the final data and the derived canonical-10 subset; superseded run dumps were removed on 2026-10-31.

## Columns

`engine`, `sample_id`, `latency_s`, `vram_peak_gb`, `vram_reserved_gb`, `alignment_failed`, `chamfer_distance`, `f_score_0.1`, `f_score_0.2`, `f_score_0.5`, `psnr`, `lpips`, `metadata`, `vertex_count`, `face_count`, `mesh_size_mb`.

`vram_peak_gb` is `torch.cuda.max_memory_allocated()` — what the published v1.0 table reports. `vram_reserved_gb` is `torch.cuda.max_memory_reserved()`, which includes the CUDA context and allocator fragmentation and so runs higher; it is provided for comparison with `nvidia-smi` and does not replace the frozen figure.

`metadata` is a JSON object (not a Python repr), so consumers can `json.loads(row["metadata"])`.

`alignment_failed` is `true` when the mesh could not be aligned to the ground truth (no GT points, or alignment failed). In that case the exported OBJ is written as `<engine>_unaligned_mesh.obj` — it is in the engine's raw frame and scale and is **not comparable across engines**.
