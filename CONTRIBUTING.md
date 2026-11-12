# Contributing to Image-to-3D Benchmark

This page is the **newcomer guide**. It explains what the project is, which files to read
first, how the pieces fit together, and how to propose a change. If you are new, start at
§1 and do not read the whole repository up front.

For the full list of known bugs, security findings and suggested fixes, see
[`docs/ISSUES.md`](docs/ISSUES.md). For the label definitions used by that register, see
[`docs/LABELS.md`](docs/LABELS.md).

---

## 1. What this project is

`image-to-3d-benchmark` is a **reproducible benchmark for feed-forward single-image 3D
reconstruction**. Given one RGB photo of an object, each engine predicts a 3D mesh (or a
Gaussian cloud). The benchmark compares those predictions against a ground-truth GSO
(Google Scanned Objects) mesh using:

- **Geometry:** symmetric Chamfer Distance (CD) and F-Score at thresholds 0.1 / 0.2 / 0.5.
- **Appearance:** PSNR and LPIPS against a re-rendered ground-truth novel view.
- **Cost:** wall-clock latency and peak GPU memory.
- **Protocol:** a locked, documented pipeline (sampling, alignment, rendering) so numbers
  are comparable across engines.

Four paradigms are evaluated in the headline run:

| Engine | Representation | Decoder |
|---|---|---|
| TripoSR | Triplane NeRF | 256^3 Marching Cubes |
| LGM | 4-view 3D Gaussian Splatting | Foreground-filtered density Marching Cubes |
| SPAR3D | Point diffusion + deformable tetrahedra | DMTet + PBR |
| TRELLIS | Sparse structured latents + rectified flow | FlexiCubes |

`SF3D` (`src/engines/sf3d_engine.py`) also exists as a paradigm-1 ablation. It is exported
from the engines package and selected in `EngineType`, but it is not in the README headline
table; see `DOC-03` in `docs/ISSUES.md`.

The paper's headline run is **100 objects x 4 engines = 400 evaluations**. The canonical
10-object subset is a filtered view of those same 400 rows (it is not a separate run).

---

## 2. The 20-minute reading path

Read these in order. Each step says why it matters and what to look for.

| Step | File | Why read it | Time |
|---:|---|---|---:|
| 1 | [`README.md`](README.md) | The claims, protocol summary, headline numbers and repo layout. | 4 min |
| 2 | [`AGENTS.md`](AGENTS.md) | Dense technical context: engine internals, locked protocol, disk hygiene. | 4 min |
| 3 | [`src/engines/base.py`](src/engines/base.py) | The contract every engine implements: `Mesh3D`, `GaussianCloud`, `ReconstructionResult`, `AbstractEngine`. | 2 min |
| 4 | [`src/engines/triposr_engine.py`](src/engines/triposr_engine.py) | The simplest engine. Shows preprocessing, marching cubes, vertex colours, coordinate rotation. | 4 min |
| 5 | [`src/evaluation/benchmark_runner.py`](src/evaluation/benchmark_runner.py) | The heart of the benchmark: loading, alignment, rendering, metrics, resume, export. Start with `BenchmarkRunner.run`. | 8 min |
| 6 | [`src/evaluation/metrics.py`](src/evaluation/metrics.py), [`chamfer_distance.py`](src/evaluation/chamfer_distance.py), [`f_score.py`](src/evaluation/f_score.py), [`psnr_lpips.py`](src/evaluation/psnr_lpips.py) | The exact metric definitions. Small files; read them before touching numbers. | 3 min |
| 7 | [`src/preprocessing/background_removal.py`](src/preprocessing/background_removal.py) | Alpha/foreground handling. Many downstream bugs start here. | 2 min |
| 8 | [`src/export/glb_exporter.py`](src/export/glb_exporter.py), [`src/export/ply_exporter.py`](src/export/ply_exporter.py) | Output formats. Small but easy to get wrong (colour space, SH coefficients). | 2 min |
| 9 | [`app.py`](app.py), [`src/api/server.py`](src/api/server.py) | The two user-facing front ends. Read after the core so you understand what they call. | 4 min |
| 10 | [`kaggle/`](kaggle/) notebooks | How the 400-evaluation run is actually executed on Kaggle T4. Skim the setup and runner cells; do not start here. | 5 min |
| 11 | [`docs/ISSUES.md`](docs/ISSUES.md) | Pick one `good first issue` and read its full entry. | 5 min |

If you only have five minutes: read `README.md` §Protocol, `src/engines/base.py`, and the
first 120 lines of `src/evaluation/benchmark_runner.py`.

---

## 3. File map: "I want to work on X, where do I look?"

| If you want to work on... | Start here | Usually also touch |
|---|---|---|
| A new 3D engine | `src/engines/base.py`, `src/engines/triposr_engine.py` | `src/engines/__init__.py`, `src/config.py`, `src/api/server.py`, `app.py` |
| Geometry metrics | `src/evaluation/chamfer_distance.py`, `f_score.py`, `metrics.py` | `src/evaluation/benchmark_runner.py` |
| Rendering / PSNR / LPIPS | `src/evaluation/benchmark_runner.py` (`render_novel_view_same_bg`, `render_gso_mesh_view`) | `src/evaluation/psnr_lpips.py`, `scripts/rebuild_figures.py` |
| Alignment / normalisation | `src/evaluation/benchmark_runner.py` (`align_mesh_to_gt`, `_normalize_vertices_robust`) | `src/evaluation/metrics.py` |
| Background removal | `src/preprocessing/background_removal.py` | every engine's `predict` |
| GLB / PLY export | `src/export/` | `app.py`, `src/api/server.py` |
| Gradio demo | `app.py` | `src/engines/*`, `src/export/*` |
| REST API | `src/api/server.py` | `src/config.py`, `src/engines/*` |
| Kaggle run / reproducibility | `kaggle/*/kernel-metadata.json`, kernel notebooks | `scripts/merge_batches.py`, `results/README.md` |
| Figures for the paper | `scripts/rebuild_figures.py` | `results/*.csv` |
| 1000-object selection | `gso1000/select_1000.py`, `gso1000/SELECTION_SPEC.md` | `gso_catalog.txt`, `handpicked.txt` |
| Packaging / dependencies | `pyproject.toml`, `requirements.txt` | all imports |
| Docs | `README.md`, `AGENTS.md`, `docs/` | code you document |

### What each top-level file is

- `app.py` — Gradio demo. Upload image -> background removal -> engine -> GLB/PLY.
- `src/config.py` — Pydantic configuration models. Currently mostly declarative; not wired
  into the engines or runner (see `ISSUES.md` `PKG-*`/`CFG-*`).
- `src/engines/` — one module per paradigm, all implementing `AbstractEngine`.
- `src/evaluation/` — metrics and the locked `BenchmarkRunner` protocol.
- `src/preprocessing/` — `rembg` foreground segmentation.
- `src/export/` — GLB and 3DGS PLY writers.
- `src/api/server.py` — FastAPI inference service (separate from the Gradio app).
- `scripts/merge_batches.py` — merge per-engine `*_results.json` into CSV/TeX.
- `scripts/rebuild_figures.py` — rebuild paper figures from run outputs.
- `kaggle/` — executable Kaggle kernels that produced the reported numbers.
- `gso1000/` — deterministic 1000-object manifest builder.
- `datasets/`, `handpicked.txt`, `gso_catalog.txt` — evaluation metadata and object lists.
- `results/` — canonical CSVs, TeX tables and mesh manifest.
- `examples/` — sample input images used by the demo.
- `notebooks/colab_demo.ipynb` — lightweight TripoSR-only Colab demo.

---

## 4. Core concepts (plain language)

- **Engine** — a wrapper around a third-party 3D reconstruction model. It exposes
  `load_model()`, `predict(image) -> ReconstructionResult`, and `get_engine_info()`.
- **Mesh3D** — vertices `(N, 3)`, faces `(M, 3)`, optional per-vertex RGB colours in
  `[0, 1]`, optional UVs and texture.
- **GaussianCloud** — positions, scales, rotations, opacities, SH/DC colours for 3DGS.
- **GSO** — Google Scanned Objects, the ground-truth dataset. Each object has `meshes/`
  and rendered `images/` views.
- **Canonical 10** — the 10 GSO objects used for the paper's small table. They are a subset
  of the 100-object run.
- **CD (Chamfer Distance)** — symmetric mean nearest-neighbour distance between predicted
  and GT surface samples, after alignment. Lower is better.
- **F-Score** — harmonic mean of precision and recall of surface points within a distance
  threshold, after alignment. Higher is better.
- **PSNR / LPIPS** — 2D image similarity after rendering both meshes from the same camera.
  Higher PSNR / lower LPIPS is better.
- **Staged offload** — TRELLIS-specific memory strategy that keeps unused submodels on the
  CPU and moves them to the GPU only when needed. Needed for a 16 GB T4.
- **Locked protocol** — the exact sampling, alignment and rendering settings. Do not change
  them casually: published numbers depend on them. If you must change them, treat it as a
  new protocol version and document the old one.

---

## 5. Setup

The benchmark targets Linux + NVIDIA GPU (a 16 GB T4 is the reference). CPU-only is possible
for TripoSR and the metrics, but the diffusion-based engines effectively require CUDA.

```bash
git clone <your-fork-url>
cd image-to-3d-benchmark
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

Heavy CUDA extensions (`diff-gaussian-rasterization`, `spconv`, SPAR3D rasterizers,
`diffoctreerast`) are documented in [`docs/COMPILE.md`](docs/COMPILE.md). You do **not**
need them to work on metrics, export, docs, API or packaging.

Verify the import path:

```bash
python -c "from src.evaluation.metrics import normalize_point_cloud; print('ok')"
```

There is currently **no test suite** (see `ISSUES.md` `CI-*`). Until one exists, verify
changes with the focused commands in §7.

---

## 6. How to contribute

### Workflow

1. **Find or open an issue.** Every entry in `docs/ISSUES.md` has an ID (for example
   `EVAL-03`) and labels. If you want to work on one, comment on the issue with the ID and
   what you plan to change.
2. **Create a branch** from `main` (or the session branch in a Cloud workspace). Keep the
   branch focused: one issue per branch.
3. **Make the smallest correct change.** Do not reformat unrelated files.
4. **Add or update a regression check** where one is feasible. If no test framework exists,
   add a small script under `scripts/` or a notebook cell and say so in the PR.
5. **Run the verification commands** in §7 that apply to your change.
6. **Open a pull request** using `.github/pull_request_template.md`. In the description,
   include the issue ID, what changed, how you verified it, and any protocol/result impact.
7. **Do not commit generated artefacts** (`outputs/`, `paper/`, large renders, `*_mesh.obj`).
   They are gitignored on purpose.

### Rules that apply to every contribution

- **Do not change the locked protocol silently.** Sampling counts, rotation search, ICP
  iterations, lighting, camera, resolution and colour handling are part of the published
  method. A change requires a new protocol label and a note in `results/README.md`.
- **Do not edit `results/*.csv` by hand.** They are generated from run JSON. Fix the
  generator or the runner instead.
- **Do not add model weights, GSO meshes or textures to git.** They are large and have
  separate licences.
- **Do not introduce a new third-party dependency** without a strong reason; the CUDA stack
  is already fragile. If you must, add it to the smallest extra in `pyproject.toml` and to
  `requirements.txt`.
- **Keep import side effects out of library modules.** Environment variables, `sys.path`
  edits, monkeypatches and global caches make the benchmark non-reproducible. Prefer an
  explicit `configure()` function called by the entry point.
- **Never log secrets** (HF tokens, API keys) or full tracebacks to end users.
- **Use `file.py:line` references** in issues and PR comments.

### Commit and PR style

Short imperative subject, for example:

```text
fix(evaluation): align rendered mesh with the transform used for CD
```

Use the scopes `engines`, `evaluation`, `preprocessing`, `export`, `api`, `app`, `scripts`,
`docs`, `kaggle`, `packaging`. Reference the issue ID in the PR body, not necessarily in
every commit.

### Review expectations

A maintainer will check:

- Does the change preserve the locked protocol (or explicitly version it)?
- Is there a focused verification (unit test, script, or before/after numbers)?
- Does it avoid new global state, unpinned downloads or silent `except` blocks?
- Are docs/issue labels updated when behaviour changes?

---

## 7. Verification commands

There is no lint/typecheck/test command configured yet. Until §CI issues are resolved, use
these focused checks. Do not run repository-wide installs in a sandbox without a GPU.

```bash
# Syntax / import sanity for a file you changed
python -m py_compile src/evaluation/benchmark_runner.py

# Import the package (will fail if a core dependency is missing)
python -c "import src.evaluation.benchmark_runner"

# Metric sanity on a tiny synthetic pair
python - <<'PY'
import numpy as np
from src.evaluation.metrics import normalize_point_cloud
from src.evaluation.chamfer_distance import compute_chamfer_distance
from src.evaluation.f_score import compute_f_score
a = normalize_point_cloud(np.random.default_rng(0).normal(size=(1000, 3)))
print("CD", compute_chamfer_distance(a, a))
print("F", compute_f_score(a, a))
PY

# Export round-trip (needs trimesh)
python - <<'PY'
import numpy as np, trimesh
from src.engines.base import Mesh3D
from src.export.glb_exporter import export_glb
m = Mesh3D(vertices=np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float32),
           faces=np.array([[0,1,2]], dtype=np.int64))
export_glb(m, "/tmp/roundtrip.glb")
print(trimesh.load("/tmp/roundtrip.glb"))
PY
```

If you add a test framework, document the exact command here and in `AGENTS.md`.

---

## 8. Good first issues

These are small, well-scoped and do not require CUDA. The full text is in
`docs/ISSUES.md`; search for the ID.

| ID | One-line summary | Why it is a good start |
|---|---|---|
| `DOC-01` | README says 3 ICP iterations; code uses 5 | Docs-only, touches one line |
| `DOC-06` | `docs/COMPILE.md` duplicate section number + stray backtick | Docs-only |
| `EXP-03` | GLB exporter does not create the parent directory | Two-line fix, easy to test |
| `EXP-05` | GLB exporter passes float vertex colours | Small, testable with trimesh |
| `EVAL-32` | `gt_novel_Views` is not snake_case | Mechanical rename, compiler catches uses |
| `EVAL-29` | Duplicate `compute_chamfer_distance_symmetric` wrapper | Delete-only |
| `TOOL-05` | `merge_batches.py` docstring promises a Pareto PNG it never writes | Docs/behaviour mismatch |
| `TOOL-06` | `merge_batches.py` writes into the run directory | One-line change |
| `PKG-13` | `pyproject.toml` has no `license` field | Packaging metadata |
| `DOC-11` | No `CHANGELOG.md` | Create from tags/releases |
| `DOC-12` | No `CODE_OF_CONDUCT.md` | Add the standard Contributor Covenant |
| `CI-01` | No `tests/` directory | Add the first metric/export tests |

Before starting, read the full entry for the ID and the files it references. If the issue
turns out to be larger than described, comment on it instead of silently expanding scope.

---

## 9. Where to ask

- Open a GitHub issue using `.github/ISSUE_TEMPLATE/bug_report.yml` for a defect, or
  `feature_request.yml` for an enhancement.
- For a security issue, **do not open a public issue**. Follow the repository's private
  reporting process (to be added in `SECURITY.md`) or contact the maintainer.
- When asking about a metric, include the object ID, engine, code SHA and the exact command.

---

## 10. Newcomer checklist

- [ ] I read `README.md` and `AGENTS.md`.
- [ ] I read `src/engines/base.py` and `src/engines/triposr_engine.py`.
- [ ] I skimmed `src/evaluation/benchmark_runner.py`.
- [ ] I read the full entry for the issue I am claiming in `docs/ISSUES.md`.
- [ ] I know whether my change touches the locked protocol.
- [ ] I know which verification command in §7 applies.
- [ ] I will not commit generated outputs, weights or GSO assets.