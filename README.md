# Image-to-3D Unified Benchmark

[![CI](https://github.com/anubhavaanand/Project-red-2d-to-3d/actions/workflows/ci.yml/badge.svg)](https://github.com/anubhavaanand/Project-red-2d-to-3d/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A reproducible, ground-truth-referenced benchmark for feed-forward single-image 3D reconstruction. Four representative paradigms are evaluated on Google Scanned Objects (GSO) under one locked protocol, on a single 16 GB NVIDIA Tesla T4: **400 evaluations** (100 objects × 4 engines), **0 failures**.

The benchmark measures geometry (Chamfer Distance, F-Score), appearance (PSNR, LPIPS), cost (latency, peak VRAM), and output topology — then tests the ranking with paired significance tests rather than reporting means alone.

> Manuscript: *Benchmarking Feed-Forward Single-Image 3D Reconstruction Across Triplane, Gaussian, Point-Aware, and Structured Latent Paradigms* (under review).

## Evaluated paradigms

| Engine | Representation | Decoder | Reference |
|---|---|---|---|
| **TripoSR** | Implicit triplane NeRF | 256³ Marching Cubes | [arXiv:2403.02151](https://arxiv.org/abs/2403.02151) |
| **LGM** | Multi-view 3D Gaussian Splatting | Foreground-filtered density Marching Cubes | [arXiv:2402.05054](https://arxiv.org/abs/2402.05054) |
| **SPAR3D** | Point-aware diffusion + deformable tetrahedra | DMTet + PBR UV baking | [arXiv:2501.04689](https://arxiv.org/abs/2501.04689) |
| **TRELLIS** | Sparse structured latents + rectified flow | FlexiCubes | [arXiv:2412.01506](https://arxiv.org/abs/2412.01506) |

**Also implemented, not part of the headline run:**

| Engine | Representation | Decoder | Status | Reference |
|---|---|---|---|---|
| **SF3D** | DINOv2 + LRM-style transformer | DMTet + spherical-Gaussian delighting | Paradigm-1 ablation only — **no** rows in `unified_results_4engines_100obj.csv` | [arXiv:2408.00653](https://arxiv.org/abs/2408.00653) |

SF3D exists as a first-class engine (`src/engines/sf3d_engine.py`, `EngineType.SF3D`, `SF3DEngine` exported from `src.engines`) and shares TripoSR's paradigm with a different decoder, which is why it is kept for the DMTet-vs-Marching-Cubes ablation. It is deliberately **excluded** from every headline number: the 400 evaluations, the significance tests and the figures are all TripoSR / LGM / SPAR3D / TRELLIS. Installing it needs a CUDA build — see `docs/COMPILE.md` and `pip install -e ".[sf3d]"`.

## Headline results (n = 100)

Measured on a Tesla T4 (15.6 GB) with the protocol below. Best per column in **bold**.

| Engine | CD ↓ | F@0.1 ↑ | PSNR ↑ | LPIPS ↓ | Latency (s) | Peak VRAM (GB) | Median CD |
|---|---:|---:|---:|---:|---:|---:|---:|
| TripoSR | 0.1373 ± 0.0802 | 0.7900 ± 0.2070 | 19.03 ± 2.77 | 0.2334 ± 0.0803 | **3.66 ± 0.25** | **2.32** | 0.1135 |
| LGM | 0.1279 ± 0.0588 | 0.7960 ± 0.1789 | 17.64 ± 2.35 | 0.2832 ± 0.0761 | 11.69 ± 1.12 | 5.19 | 0.1075 |
| SPAR3D | 0.1142 ± 0.0528 | 0.8368 ± 0.1588 | 19.17 ± 2.93 | 0.2126 ± 0.0707 | 37.54 ± 0.53 | 9.17 | 0.1028 |
| TRELLIS | **0.0786 ± 0.0366** | **0.9374 ± 0.1028** | **20.51 ± 2.89** | **0.1849 ± 0.0605** | 20.46 ± 6.78 | 6.65 mean / 10.45 max | **0.0695** |

TRELLIS is best on 73/100 objects by CD and 72/100 by F@0.1 (ties included). SPAR3D is second on every quality metric and produces the most compact meshes (2.16 ± 1.08 MB).

## Statistical significance

Two-sided paired Wilcoxon signed-rank tests over the 100 shared objects:

- TRELLIS vs SPAR3D on CD: **p = 3.18 × 10⁻¹²**
- TRELLIS vs LGM on CD: **p = 5.31 × 10⁻¹⁶**
- TRELLIS vs TripoSR on CD: **p = 5.64 × 10⁻¹⁴**
- SPAR3D vs LGM on CD: **p = 1.54 × 10⁻³**
- LGM vs TripoSR on CD: p = 0.443 — **not significant**, consistent with the 52/48 per-object split; they differ on PSNR (p = 8.39 × 10⁻¹⁰) and LPIPS (p = 1.90 × 10⁻¹⁴)

95% bootstrap confidence intervals for mean CD (10,000 resamples): TRELLIS [0.0718, 0.0860], SPAR3D [0.1046, 0.1249], LGM [0.1170, 0.1398], TripoSR [0.1227, 0.1535].

## Protocol

- **Sampling** — 10,000 surface points per mesh, centroid-centered, normalized to the unit sphere at the 99.5th-percentile radius. Note the two-stage rule: mesh vertices are normalised by **bounding-box centre**, then the points sampled from that mesh are re-normalised by **centroid**. The rendered/exported mesh therefore follows the bounding-box convention while CD/F-score use the centroid convention.
- **Alignment** — 72 candidate rotations (6 principal axis-up orientations × 12 yaw steps) followed by 5-iteration Kabsch ICP, scored by symmetric mean Chamfer Distance. Both clouds are subsampled to **2,048 points** for the ICP search (seed 0) while the reported metrics use the full 10,000; the subsample size is configurable via `alignment_subsample_points` on `BenchmarkRunner` (default 2048).
- **Rendering** — one novel view at 65° azimuth / 15° elevation, rendered at 2× and Lanczos-downsampled to 256², under three-point studio lighting (ambient 0.45, key 0.85, fill 0.35, rim 0.25, distance 2.45, yfov 49.1°) with linear-to-sRGB encoding.
- **LGM proxy mesh** — opacity filter α > 0.10, 98th-percentile spatial trim, 3D Gaussian smoothing (k = 7, σ = 1.10), 256³ Marching Cubes at the 48th percentile, floater pruning, and k = 6 opacity-weighted color transfer from core Gaussians (α > 0.35, fallback α > 0.20).
- **TRELLIS** — 8 + 8 sampling steps (CFG 7.5 / 3.0) with staged CPU–GPU offloading.
- **PSNR** — computed on the 256² novel-view renders. A pixel-perfect match returns the documented cap of **100 dB** rather than infinity, so aggregates stay finite; the cap is recorded in `metadata` per sample.

## Repository layout

```text
image-to-3d-benchmark/
├── README.md
├── LICENSE                       # MIT (code only; weights keep their own terms)
├── CITATION.cff
├── AGENTS.md                     # technical context for contributors/agents
├── app.py                        # Gradio demo
├── handpicked.txt                # curated 100-object GSO catalog
├── gso_catalog.txt               # full GSO identifier list
├── docs/
│   ├── COMPILE.md                # CUDA extension build guide
│   └── PROJECT_STATUS.md         # historical run log
├── kaggle/
│   ├── batch1/                   # TripoSR + LGM kernel (final v35 run)
│   ├── batch2/                   # TRELLIS + SPAR3D kernel (final v35 run)
│   ├── triposr_only/             # isolated TripoSR run
│   └── verify/                   # render verification kernel
├── results/
│   ├── README.md                 # which files are canonical
│   ├── unified_results_4engines.csv        # canonical-10 subset (40 rows)
│   └── unified_results_4engines_100obj.csv # n = 100
├── scripts/
│   ├── merge_batches.py          # merge per-engine JSON into the unified CSV/TeX
│   └── rebuild_figures.py
├── datasets/                     # GSO evaluation manifests
├── examples/                     # sample input images
├── notebooks/                    # Colab demo
├── src/
│   ├── engines/                  # one module per paradigm
│   ├── evaluation/               # CD, F-score, PSNR/LPIPS, runner
│   ├── preprocessing/            # rembg foreground segmentation
│   ├── export/                   # GLB / PLY writers
│   └── api/server.py
└── gso1000/                      # 1000-object catalog selection tooling
```

## Installation

```bash
# Core: evaluation harness, metrics, result CSV tooling (no torch, no GPU stack)
pip install -e .

# Inference stack (torch + engine-side deps, CPU wheels by default)
pip install -e ".[engines]"

# Optional slices
pip install -e ".[app]"        # Gradio demo (engines + gradio)
pip install -e ".[api]"        # FastAPI server (engines + fastapi/uvicorn)
pip install -e ".[notebooks]"  # Kaggle/Colab kernel glue
pip install -e ".[sf3d]"       # SF3D prerequisites (see COMPILE.md for CUDA build)
pip install -e ".[dev]"        # lint/test/security tooling

# Everything above
pip install -e ".[all]"
```

**Pick one, do not run both.** `pip install -e ".[...]"` resolves through
`pyproject.toml` and is the canonical install. `requirements.txt` is the exact
pinned set used for the published runs, kept for reproducibility; installing it
*after* the extras can downgrade a package the extras resolved, so use it only
inside a clean environment.

See [docs/COMPILE.md](docs/COMPILE.md) for the CUDA extensions
(`diff-gaussian-rasterization`, the `spconv` wheels, the SF3D rasterizer) that
must be built or installed separately.

## Reproducing the benchmark

Local run over a few objects:

```python
from src.engines.triposr_engine import TriposrEngine
from src.evaluation.benchmark_runner import BenchmarkRunner

runner = BenchmarkRunner(
    dataset_path="./data/gso",          # GSO meshes + rendered conditioning views
    engines=[TriposrEngine()],
    num_samples=10,
    num_points=10_000,
    output_dir="outputs",
)
df = runner.run()
print(runner.get_summary_df())
```

Full runs are executed on Kaggle T4 kernels:

```bash
kaggle kernels push -p kaggle/batch1   # TripoSR + LGM
kaggle kernels push -p kaggle/batch2   # TRELLIS + SPAR3D
```

Upload `handpicked.txt` as a kernel input to reproduce the exact 100-object catalog. Kernel outputs are downloaded into `kaggle_runs/batch*_100obj_v35/outputs`; figure regeneration expects those directories plus `data/gso` alongside the repository:

```bash
python scripts/merge_batches.py ../kaggle_runs/batch1_100obj_v35/outputs \
                                 ../kaggle_runs/batch2_100obj_v35/outputs
python scripts/rebuild_figures.py        # set BENCH_ROOT if the repo is not in the default location
```

## Data

- **GSO** — Google Scanned Objects, CC-BY 4.0. Download instructions follow the original release; meshes are read from `data/gso/<object_id>/meshes/model.obj`.
- **Canonical 10** — the cross-paradigm subset used for the paper's n = 10 block.
- **100-object catalog** — `handpicked.txt`; the wider 1,000-object selection tooling lives in `gso1000/`.

## Results

See [results/README.md](results/README.md) for the canonical CSV/TeX files and the difference between the canonical-10 and n = 100 runs. Large per-run render dumps are kept out of git.

## Citation

```bibtex
@misc{anand2026image3d,
  title  = {Benchmarking Feed-Forward Single-Image 3D Reconstruction Across
            Triplane, Gaussian, Point-Aware, and Structured Latent Paradigms},
  author = {Anand, Anubhav},
  year   = {2026},
  note   = {Manuscript under review},
  url    = {https://github.com/anubhavaanand/Project-red-2d-to-3d}
}
```

This repository is tagged [`v1.0.0`](https://github.com/anubhavaanand/Project-red-2d-to-3d/releases/tag/v1.0.0). The measurements were produced at code commit `09131e61568d98401e71861bc27fdce6637b2f20`; use that SHA, not the Kaggle kernel version number, when referring to the exact run.

## License and model terms

The benchmark harness in this repository is released under the [MIT License](LICENSE). That license does not extend to third-party model weights, third-party source code, or the GSO dataset. Evaluated models retain their original licenses — several are non-commercial — so check each model card before redistributing weights or derivative meshes. GSO assets are CC-BY 4.0.
