# Label taxonomy

This repository uses a small, consistent label set so that issues and pull requests can be
triaged without reading the whole thread. Every entry in [`ISSUES.md`](ISSUES.md) carries
labels from this page.

The labels below are the **recommended GitHub labels**. If they do not exist yet in the
repository, create them with the descriptions shown (Settings -> Labels), or apply the same
names in the issue title/body when filing.

## 1. Type labels

| Label | Meaning |
|---|---|
| `bug` | Something is wrong: incorrect result, crash, silent fallback, wrong number. |
| `security` | Confidentiality, integrity or availability risk, including supply chain. |
| `documentation` | Docs are missing, wrong or inconsistent with code/data. |
| `dependencies` | Packaging, version pins, missing or conflicting requirements. |
| `performance` | Latency, memory, disk or renderer-context cost. |
| `maintainability` | Duplication, dead code, global state, confusing API. |
| `testing` | Missing tests, CI, regression gates or reproducibility checks. |
| `methodology` | A change to the measurement protocol or a metric/render inconsistency that can move published numbers. |
| `good first issue` | Small, well-scoped, no CUDA required. |
| `help wanted` | A maintainer wants community input. |
| `blocked` | Waiting on a decision, dataset access or another issue. |

## 2. Severity labels

| Label | Meaning | Expected response |
|---|---|---|
| `severity:critical` | Remote code execution, unauthenticated destructive/abusive access, or wrong published numbers. | Fix before next release. |
| `severity:high` | Security weakness, silent data corruption, crash on a supported path, or major reproducibility break. | Fix in the current milestone. |
| `severity:medium` | Incorrect only in an edge case, or a real defect with a workaround. | Schedule. |
| `severity:low` | Cosmetic, docs, style or minor robustness. | Backlog. |

## 3. Component labels

Use one or more.

| Label | Area |
|---|---|
| `component:api` | `src/api/server.py` |
| `component:app` | `app.py` (Gradio) |
| `component:engines` | `src/engines/` |
| `component:triposr` | `src/engines/triposr_engine.py` |
| `component:lgm` | `src/engines/lgm_engine.py` |
| `component:trellis` | `src/engines/trellis_engine.py` |
| `component:spar3d` | `src/engines/spar3d_engine.py` |
| `component:sf3d` | `src/engines/sf3d_engine.py` |
| `component:evaluation` | `src/evaluation/` |
| `component:metrics` | `chamfer_distance.py`, `f_score.py`, `psnr_lpips.py`, `metrics.py` |
| `component:rendering` | `render_novel_view_same_bg`, `render_gso_mesh_view` |
| `component:preprocessing` | `src/preprocessing/` |
| `component:export` | `src/export/` |
| `component:scripts` | `scripts/` |
| `component:kaggle` | `kaggle/` |
| `component:notebooks` | `notebooks/` |
| `component:packaging` | `pyproject.toml`, `requirements.txt` |
| `component:docs` | `README.md`, `AGENTS.md`, `docs/` |
| `component:data` | `datasets/`, `handpicked.txt`, `gso_catalog.txt`, `results/` |
| `component:ci` | `.github/`, tests, automation |

## 4. Status labels

| Label | Meaning |
|---|---|
| `status:triage` | Not yet assessed by a maintainer. |
| `status:confirmed` | Reproduced or verified from source. |
| `status:needs-repro` | Needs a minimal reproduction or exact run context. |
| `status:in-progress` | Someone is working on it. |
| `status:wontfix` | Intentional; the reason is recorded in the issue. |
| `status:duplicate` | Duplicate of another ID. |

## 5. How to label a new issue

1. Pick exactly one `severity:*`.
2. Pick one or more `component:*`.
3. Add `bug`, `security`, `documentation`, `methodology` or `dependencies` as appropriate.
4. Add `good first issue` only if the full entry shows a small, isolated change.
5. If it changes a measured number, also add `methodology` and link the protocol section in
   `README.md` and `results/README.md`.

## 6. ID prefixes used in `ISSUES.md`

| Prefix | Area |
|---|---|
| `SEC` | Security and supply chain |
| `API` | REST API (`src/api/server.py`) |
| `APP` | Gradio app (`app.py`) |
| `ENG` | Engine correctness |
| `EVAL` | Evaluation, metrics, rendering, alignment |
| `DATA` | Dataset and results data |
| `EXP` | Export |
| `TOOL` | Scripts and tooling |
| `PKG` | Packaging and dependencies |
| `DOC` | Documentation and provenance |
| `NB` | Notebooks |
| `CI` | Tests, CI and automation |

IDs are stable. Do not renumber an existing entry; add a new one and mark the old one
`status:duplicate` if needed.