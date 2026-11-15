# Issue Triage Report

**Total issues analyzed:** 249

## Summary

| Decision | Count |
|----------|-------|
| GENUINE | 243 |
| NEEDS_REPRO | 6 |
| FALSE_ALARM | 0 |
| WONT_FIX | 0 |

## Issues by Category

### API (18 issues)
- GENUINE: 18
- NEEDS_REPRO: 0

### APP (16 issues)
- GENUINE: 16
- NEEDS_REPRO: 0

### CI (10 issues)
- GENUINE: 10
- NEEDS_REPRO: 0

### DATA (13 issues)
- GENUINE: 13
- NEEDS_REPRO: 0

### DOC (15 issues)
- GENUINE: 14
- NEEDS_REPRO: 1

### ENG (46 issues)
- GENUINE: 46
- NEEDS_REPRO: 0

### EVAL (40 issues)
- GENUINE: 40
- NEEDS_REPRO: 0

### EXP (14 issues)
- GENUINE: 13
- NEEDS_REPRO: 1

### NB (12 issues)
- GENUINE: 11
- NEEDS_REPRO: 1

### PKG (15 issues)
- GENUINE: 14
- NEEDS_REPRO: 1

### SEC (30 issues)
- GENUINE: 28
- NEEDS_REPRO: 2

### TOOL (20 issues)
- GENUINE: 20
- NEEDS_REPRO: 0

## Detailed Triage

| ID | Title | Decision | Reason |
|----|-------|----------|--------|
| API-01 | API-01: API bypasses background removal | GENUINE | No specific rule — defaults to genuine (conservative) |
| API-02 | API-02: Alpha channel is discarded | GENUINE | No specific rule — defaults to genuine (conservative) |
| API-03 | API-03: LGM returns a mesh, contradicting `/engines` | GENUINE | No specific rule — defaults to genuine (conservative) |
| API-04 | API-04: SPAR3D and SF3D are missing from the API | GENUINE | No specific rule — defaults to genuine (conservative) |
| API-05 | API-05: Output directory ignores configuration | GENUINE | No specific rule — defaults to genuine (conservative) |
| API-06 | API-06: No background cleanup task | GENUINE | No specific rule — defaults to genuine (conservative) |
| API-07 | API-07: Engine cache is not thread-safe | GENUINE | No specific rule — defaults to genuine (conservative) |
| API-08 | API-08: Blocking inference in an async endpoint | GENUINE | No specific rule — defaults to genuine (conservative) |
| API-09 | API-09: No timeout or concurrency cap | GENUINE | No specific rule — defaults to genuine (conservative) |
| API-10 | API-10: Inconsistent error status for invalid engine | GENUINE | No specific rule — defaults to genuine (conservative) |
| API-11 | API-11: No CORS policy documented | GENUINE | No specific rule — defaults to genuine (conservative) |
| API-12 | API-12: Health check does not report readiness | GENUINE | No specific rule — defaults to genuine (conservative) |
| API-13 | API-13: Engine metadata is hardcoded and can drift | GENUINE | No specific rule — defaults to genuine (conservative) |
| API-14 | API-14: All engine failures become 500 | GENUINE | No specific rule — defaults to genuine (conservative) |
| API-15 | API-15: Uploaded image is not closed | GENUINE | No specific rule — defaults to genuine (conservative) |
| API-16 | API-16: Output directory created as an import side effect | GENUINE | No specific rule — defaults to genuine (conservative) |
| API-17 | API-17: No API tests | GENUINE | No specific rule — defaults to genuine (conservative) |
| API-18 | API-18: Relative imports break direct execution | GENUINE | No specific rule — defaults to genuine (conservative) |
| APP-01 | APP-01: UI omits SPAR3D although the README advertises it | GENUINE | No specific rule — defaults to genuine (conservative) |
| APP-02 | APP-02: `_get_engine` cannot construct SPAR3D | GENUINE | No specific rule — defaults to genuine (conservative) |
| APP-03 | APP-03: Output filename collisions | GENUINE | No specific rule — defaults to genuine (conservative) |
| APP-04 | APP-04: Preview renderer silently returns nothing | GENUINE | No specific rule — defaults to genuine (conservative) |
| APP-05 | APP-05: Device chosen at import time | GENUINE | No specific rule — defaults to genuine (conservative) |
| APP-06 | APP-06: `sys.path.insert(0, ...)` | GENUINE | No specific rule — defaults to genuine (conservative) |
| APP-07 | APP-07: Example paths are CWD-dependent | GENUINE | No specific rule — defaults to genuine (conservative) |
| APP-08 | APP-08: No cleanup of the app output directory | GENUINE | No specific rule — defaults to genuine (conservative) |
| APP-09 | APP-09: Models are never unloaded | GENUINE | No specific rule — defaults to genuine (conservative) |
| APP-10 | APP-10: Latency includes background removal | GENUINE | No specific rule — defaults to genuine (conservative) |
| APP-11 | APP-11: Vertex colours multiplied without range check | GENUINE | No specific rule — defaults to genuine (conservative) |
| APP-12 | APP-12: Description numbers can drift | GENUINE | No specific rule — defaults to genuine (conservative) |
| APP-13 | APP-13: No input validation before inference | GENUINE | No specific rule — defaults to genuine (conservative) |
| APP-14 | APP-14: Long inference blocks the Gradio worker | GENUINE | No specific rule — defaults to genuine (conservative) |
| APP-15 | APP-15: Heavy imports at module import | GENUINE | No specific rule — defaults to genuine (conservative) |
| APP-16 | APP-16: No model-load error handling | GENUINE | No specific rule — defaults to genuine (conservative) |
| CI-01 | CI-01: No `tests/` directory | GENUINE | Real testing/CI infrastructure gap that should be addressed |
| CI-02 | CI-02: No CI workflow | GENUINE | Real testing/CI infrastructure gap that should be addressed |
| CI-03 | CI-03: No lint/format gate | GENUINE | Real testing/CI infrastructure gap that should be addressed |
| CI-04 | CI-04: No benchmark regression test | GENUINE | Real testing/CI infrastructure gap that should be addressed |
| CI-05 | CI-05: No metric unit tests | GENUINE | Real testing/CI infrastructure gap that should be addressed |
| CI-06 | CI-06: No export round-trip tests | GENUINE | Real testing/CI infrastructure gap that should be addressed |
| CI-07 | CI-07: No API smoke test | GENUINE | Real testing/CI infrastructure gap that should be addressed |
| CI-08 | CI-08: No dependency audit / pin check | GENUINE | Real testing/CI infrastructure gap that should be addressed |
| CI-09 | CI-09: No results-contract validation in CI | GENUINE | Real testing/CI infrastructure gap that should be addressed |
| CI-10 | CI-10: No notebook execution smoke test | GENUINE | Real testing/CI infrastructure gap that should be addressed |
| DATA-01 | DATA-01: Handpicked loader semantics differ across call site... | GENUINE | Handpicked loader semantics differ across call sites — confirmed |
| DATA-02 | DATA-02: Dataset manifest key mismatch | GENUINE | Dataset manifest key mismatch — confirmed |
| DATA-03 | DATA-03: Preview manifest is tracked like a run manifest | GENUINE | Preview manifest tracked like run manifest — confirmed |
| DATA-04 | DATA-04: Mesh manifest paths are relative to a missing run d... | GENUINE | Mesh manifest paths relative to missing run directory — confirmed |
| DATA-05 | DATA-05: `metadata` column is a Python repr, not JSON | GENUINE | metadata column is Python repr, not JSON — confirmed |
| DATA-06 | DATA-06: Results README column list does not match the CSV | GENUINE | Results README column list doesn't match CSV — confirmed |
| DATA-07 | DATA-07: `datasets/gso_eval_v1.json` is not used by the load... | GENUINE | gso_eval_v1.json not used by loader — confirmed |
| DATA-08 | DATA-08: Catalog verification date is hardcoded | GENUINE | Catalog verification date hardcoded — confirmed |
| DATA-09 | DATA-09: Mesh manifest has no run/commit column | GENUINE | Mesh manifest has no run/commit column — confirmed |
| DATA-10 | DATA-10: No schema/contract test for result CSVs | GENUINE | No schema/contract test for result CSVs — confirmed (we're adding one) |
| DATA-11 | DATA-11: `handpicked.txt` rationale is free-text only | GENUINE | handpicked.txt rationale is free-text only — confirmed |
| DATA-12 | DATA-12: No checksum/verification script for `gso_catalog.tx... | GENUINE | No checksum/verification script for gso_catalog.txt — confirmed |
| DATA-13 | DATA-13: `mesh_manifest.csv` hashes cover files not in git | GENUINE | mesh_manifest.csv hashes cover files not in git — confirmed |
| DOC-01 | DOC-01: README ICP iteration count is wrong | GENUINE | README ICP iteration count is wrong — confirmed, says 3 but code uses 3 iteratio... |
| DOC-02 | DOC-02: Repository URL is inconsistent | GENUINE | Repository URL inconsistent — confirmed |
| DOC-03 | DOC-03: SF3D is implemented but absent from the README parad... | GENUINE | SF3D implemented but absent from README — confirmed, it's an ablation engine |
| DOC-04 | DOC-04: Run directory names disagree across docs and scripts | GENUINE | Run directory names disagree across docs and scripts — confirmed |
| DOC-05 | DOC-05: `PROJECT_STATUS.md` contains superseded numbers | GENUINE | PROJECT_STATUS.md contains superseded numbers — confirmed |
| DOC-06 | DOC-06: `COMPILE.md` section numbering and stray backtick | GENUINE | COMPILE.md section numbering and stray backtick — confirmed |
| DOC-07 | DOC-07: Paper title differs between `app.py` and README | GENUINE | Paper title differs between app.py and README — confirmed |
| DOC-08 | DOC-08: `CITATION.cff` release date may precede the tag | NEEDS_REPRO | CITATION.cff release date may precede tag — needs git tag verification |
| DOC-09 | DOC-09: "0 failures" is not independently verifiable | GENUINE | 0 failures is not independently verifiable — confirmed, needs log evidence |
| DOC-10 | DOC-10: `AGENTS.md` hardcodes local paths | GENUINE | AGENTS.md hardcodes local paths — confirmed |
| DOC-11 | DOC-11: No `CHANGELOG.md` | GENUINE | No CHANGELOG.md — confirmed |
| DOC-12 | DOC-12: No `CODE_OF_CONDUCT.md` | GENUINE | No CODE_OF_CONDUCT.md — confirmed |
| DOC-13 | DOC-13: No `SECURITY.md` | GENUINE | No SECURITY.md — confirmed |
| DOC-14 | DOC-14: `COMPILE.md` contradicts `pyproject.toml` on spconv | GENUINE | COMPILE.md contradicts pyproject.toml on spconv — confirmed |
| DOC-15 | DOC-15: README install order can produce a broken environmen... | GENUINE | README install order can produce broken environment — confirmed |
| ENG-01 | ENG-01: TripoSR wraps `load_state_dict` repeatedly | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-02 | ENG-02: Remap wrapper ignores the caller's `strict` | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-03 | ENG-03: Marching-cubes failure becomes an empty mesh | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-04 | ENG-04: Marching-cubes helper assumes a torch tensor | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-05 | ENG-05: Foreground-mask heuristic is brittle | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-06 | ENG-06: Orientation test assumes a closed surface | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-07 | ENG-07: Interior colour offsets are absolute | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-08 | ENG-08: Unconditional CUDA cache calls | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-09 | ENG-09: No `unload_model` despite the runner hook | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-10 | ENG-10: LGM xformers shim detection is wrong | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-11 | ENG-11: Shim `attn_bias` semantics are wrong | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-12 | ENG-12: Global `xformers` replacement | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-13 | ENG-13: LGM silently drops mismatched checkpoint keys | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-14 | ENG-14: LGM view reordering assumes a 4-tuple | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-15 | ENG-15: Unconditional CUDA autocast | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-16 | ENG-16: Batch dimension assumed to be 1 | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-17 | ENG-17: Opacity thresholds are arbitrary and unvalidated | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-18 | ENG-18: Per-object percentile iso-level | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-19 | ENG-19: `trimesh.split` on an empty mesh | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-20 | ENG-20: Broad `except` in LGM mesh extraction | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-21 | ENG-21: TRELLIS OOM fallback can re-OOM in `finally` | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-22 | ENG-22: TRELLIS assumes the `mesh` key exists | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-23 | ENG-23: TRELLIS vertex colours assumed `[0,1]` | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-24 | ENG-24: TRELLIS uses two different DC conventions | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-25 | ENG-25: Mesh and Gaussian outputs use different coordinate f... | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-26 | ENG-26: Duplicate/unused `cfg_strength` parameters | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-27 | ENG-27: Unsupported `radiance_field` output format | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-28 | ENG-28: `attach()` is untyped and unused | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-29 | ENG-29: Import-time environment mutation | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-30 | ENG-30: SPAR3D prune shim re-enables gradients | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-31 | ENG-31: Dummy `transparent_background` silently disables rem... | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-32 | ENG-32: Silent foreground-crop fallback | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-33 | ENG-33: Unconditional `tm.invert()` | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-34 | ENG-34: `vis.to_color()` may not exist | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-35 | ENG-35: Texture assumed to be a PIL image | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-36 | ENG-36: `mesh_out[0]` list assumption (SPAR3D) | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-37 | ENG-37: SF3D hardcodes `low_vram_mode=True` | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-38 | ENG-38: SF3D `mesh_out[0]` list assumption | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-39 | ENG-39: SF3D drops UVs and texture | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-40 | ENG-40: SF3D `glob_dict` unused | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-41 | ENG-41: SF3D CUDA-only device context | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-42 | ENG-42: No shape/dtype validation in the engine contract | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-43 | ENG-43: `load_model` signature mismatch | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-44 | ENG-44: `engines/__init__.__getattr__` does not cache | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-45 | ENG-45: Heavy imports at engine module import | GENUINE | No specific rule — defaults to genuine (conservative) |
| ENG-46 | ENG-46: No engine smoke tests | GENUINE | No specific rule — defaults to genuine (conservative) |
| EVAL-01 | EVAL-01: README says 3 ICP iterations; code uses 5 | GENUINE | Real evaluation methodology issue |
| EVAL-02 | EVAL-02: Rendered mesh uses a different transform than the m... | GENUINE | No specific rule — defaults to genuine (conservative) |
| EVAL-03 | EVAL-03: Vertex-colour gamma is not linearised despite the c... | GENUINE | No specific rule — defaults to genuine (conservative) |
| EVAL-04 | EVAL-04: Mesh and point normalisation use different centres | GENUINE | No specific rule — defaults to genuine (conservative) |
| EVAL-05 | EVAL-05: Render normalisation uses a magic radius window | GENUINE | No specific rule — defaults to genuine (conservative) |
| EVAL-06 | EVAL-06: PSNR returns 100 dB for identical images | GENUINE | No specific rule — defaults to genuine (conservative) |
| EVAL-07 | EVAL-07: LPIPS model is not moved to a device | GENUINE | No specific rule — defaults to genuine (conservative) |
| EVAL-08 | EVAL-08: LPIPS tensor path skips normalisation | GENUINE | No specific rule — defaults to genuine (conservative) |
| EVAL-09 | EVAL-09: Texture lookup can select a novel-view PNG | GENUINE | No specific rule — defaults to genuine (conservative) |
| EVAL-10 | EVAL-10: Silent grey fallback when texture baking fails | GENUINE | No specific rule — defaults to genuine (conservative) |
| EVAL-11 | EVAL-11: Default sample selection is alphabetical, not the c... | GENUINE | No specific rule — defaults to genuine (conservative) |
| EVAL-12 | EVAL-12: `num_samples` is ignored with `object_ids` | GENUINE | No specific rule — defaults to genuine (conservative) |
| EVAL-13 | EVAL-13: Alignment uses a 2048-point subsample | GENUINE | No specific rule — defaults to genuine (conservative) |
| EVAL-14 | EVAL-14: Empty mesh/cloud crashes alignment | GENUINE | No specific rule — defaults to genuine (conservative) |
| EVAL-15 | EVAL-15: Per-render `OffscreenRenderer` leak | GENUINE | Per-render OffscreenRenderer leak — confirmed, performance issue |
| EVAL-16 | EVAL-16: GT texture is not attached to the trimesh visual | GENUINE | GT texture not attached to trimesh visual — confirmed, high severity |
| EVAL-17 | EVAL-17: GT mesh is reloaded for every engine | GENUINE | GT mesh reloaded for every engine — confirmed, performance |
| EVAL-18 | EVAL-18: VRAM metric underestimates real usage | GENUINE | VRAM metric underestimates real usage — confirmed |
| EVAL-19 | EVAL-19: Latency excludes preprocessing | GENUINE | Latency excludes preprocessing — confirmed |
| EVAL-20 | EVAL-20: PSNR/LPIPS size mismatch is unguarded | GENUINE | PSNR/LPIPS size mismatch unguarded — confirmed edge case |
| EVAL-21 | EVAL-21: Stale qualitative artefacts are not refreshed | GENUINE | Stale qualitative artefacts not refreshed — confirmed |
| EVAL-22 | EVAL-22: Mesh export can use an unaligned mesh | GENUINE | Mesh export can use unaligned mesh — confirmed |
| EVAL-23 | EVAL-23: Failed rows are never cached on resume | GENUINE | Failed rows never cached on resume — confirmed |
| EVAL-24 | EVAL-24: Per-sample broad `except` hides the traceback | GENUINE | Per-sample broad except hides traceback — confirmed |
| EVAL-25 | EVAL-25: `save_results` swallows errors | GENUINE | save_results swallows errors — confirmed |
| EVAL-26 | EVAL-26: Summary includes error rows | GENUINE | Summary includes error rows — confirmed |
| EVAL-27 | EVAL-27: Library code installs global signal handlers | GENUINE | Library code installs global signal handlers — confirmed |
| EVAL-28 | EVAL-28: No result schema validation | GENUINE | No result schema validation — confirmed |
| EVAL-29 | EVAL-29: Duplicate Chamfer wrapper | GENUINE | Duplicate Chamfer wrapper — confirmed |
| EVAL-30 | EVAL-30: F-score tree naming is confusing | GENUINE | F-score tree naming confusing — confirmed |
| EVAL-31 | EVAL-31: No metric unit tests | GENUINE | No metric unit tests — confirmed (we're adding them) |
| EVAL-32 | EVAL-32: `gt_novel_Views` is not snake_case | GENUINE | gt_novel_Views not snake_case — confirmed |
| EVAL-33 | EVAL-33: Legacy alias `render_novel_view` | GENUINE | Legacy alias render_novel_view — confirmed |
| EVAL-34 | EVAL-34: sRGB encoding only on the native-UV path | GENUINE | sRGB encoding only on native-UV path — confirmed, high severity |
| EVAL-35 | EVAL-35: OpenGL compatibility patch runs at import | GENUINE | OpenGL compatibility patch runs at import — confirmed |
| EVAL-36 | EVAL-36: Missing dataset path returns an empty list, then a ... | GENUINE | Missing dataset path returns empty list then generic error — confirmed |
| EVAL-37 | EVAL-37: GT colour length is not validated | GENUINE | GT colour length not validated — confirmed |
| EVAL-38 | EVAL-38: Upright threshold is a magic number | GENUINE | Upright threshold is a magic number — confirmed |
| EVAL-39 | EVAL-39: Colour penalty weight is a magic number | GENUINE | Colour penalty weight is a magic number — confirmed |
| EVAL-40 | EVAL-40: Alignment parameters are not exposed on the runner | GENUINE | Alignment parameters not exposed on runner — confirmed |
| EXP-01 | EXP-01: GLB exporter drops UVs and textures | GENUINE | GLB exporter drops UVs and textures — confirmed, high severity |
| EXP-02 | EXP-02: GLB export merges vertices by default | GENUINE | GLB export merges vertices by default — confirmed |
| EXP-03 | EXP-03: GLB export does not create the parent directory | GENUINE | GLB export doesn't create parent directory — confirmed |
| EXP-04 | EXP-04: No finite/NaN validation before export | GENUINE | No finite/NaN validation before export — confirmed |
| EXP-05 | EXP-05: Float vertex colours passed to trimesh | GENUINE | Float vertex colors passed to trimesh — confirmed |
| EXP-06 | EXP-06: PLY export assumes one SH convention for all engines | GENUINE | PLY export assumes one SH convention for all engines — confirmed |
| EXP-07 | EXP-07: Scale convention is undocumented | NEEDS_REPRO | Scale convention undocumented — needs verification |
| EXP-08 | EXP-08: Rotation quaternions are not normalised | GENUINE | Rotation quaternions not normalised — confirmed |
| EXP-09 | EXP-09: PLY lacks higher-order SH | GENUINE | PLY lacks higher-order SH — confirmed, by design (degree 0 only) |
| EXP-10 | EXP-10: ASCII PLY writer is slow for large clouds | GENUINE | ASCII PLY writer slow for large clouds — confirmed |
| EXP-11 | EXP-11: PLY arrays are not length-checked | GENUINE | PLY arrays not length-checked — confirmed |
| EXP-12 | EXP-12: PLY writes zero normals | GENUINE | PLY writes zero normals — confirmed |
| EXP-13 | EXP-13: Config allows `obj` but there is no OBJ exporter | GENUINE | Config allows obj but no OBJ exporter — confirmed |
| EXP-14 | EXP-14: No export round-trip tests | GENUINE | No export round-trip tests — confirmed |
| NB-01 | NB-01: Colab demo uses a different segmentation model | GENUINE | Colab demo uses different segmentation model (rembg vs ISNet) — confirmed in cod... |
| NB-02 | NB-02: Colab demo composites on white and feeds RGB | GENUINE | Colab demo composites on white instead of 0.5 grey — confirmed in code |
| NB-03 | NB-03: Colab demo likely calls `extract_mesh` incorrectly | NEEDS_REPRO | Colab demo extract_mesh call — need to verify if API changed upstream |
| NB-04 | NB-04: Unused imports/helpers in batch kernels | GENUINE | Unused imports/helpers in batch kernels — confirmed by code inspection |
| NB-05 | NB-05: `RUN_TRELLIS` is a dead branch | GENUINE | RUN_TRELLIS is a dead branch — confirmed in notebook code |
| NB-06 | NB-06: Batch 2 silently reduces the sample count | GENUINE | Batch 2 silently reduces sample count — confirmed in code |
| NB-07 | NB-07: Notebooks use IPython `display` | GENUINE | Notebooks use IPython display — confirmed, minor issue |
| NB-08 | NB-08: `Tee` log file is never closed | GENUINE | Tee log file never closed — confirmed in code |
| NB-09 | NB-09: Ingest and aggregation code is duplicated across kern... | GENUINE | Ingest/aggregation code duplicated across kernels — confirmed |
| NB-10 | NB-10: No row-count/failure gate in the kernels | GENUINE | No row-count/failure gate in kernels — confirmed |
| NB-11 | NB-11: `open(...).write_text` without encoding | GENUINE | open(...).write_text without encoding — confirmed, minor |
| NB-12 | NB-12: Verify kernel's markdown describes the wrong engines | GENUINE | Kernel markdown describes wrong engines — confirmed |
| PKG-01 | PKG-01: Top-level package is named `src` | GENUINE | Top-level package named 'src' — confirmed, non-standard but intentional for rese... |
| PKG-02 | PKG-02: `pyrender` is not a declared core dependency | GENUINE | pyrender not declared as core dependency — confirmed |
| PKG-03 | PKG-03: `gradio` is not a declared dependency | GENUINE | gradio not declared as dependency — confirmed |
| PKG-04 | PKG-04: `scikit-image` is not a declared core dependency | GENUINE | scikit-image not declared — confirmed (added by our fix) |
| PKG-05 | PKG-05: `packaging` is imported but not declared | GENUINE | packaging imported but not declared — confirmed |
| PKG-06 | PKG-06: `numpy<2.0` conflicts with the Kaggle runtime | GENUINE | numpy<2.0 conflicts with Kaggle runtime — confirmed |
| PKG-07 | PKG-07: Duplicate/conflicting ONNX runtimes | GENUINE | Duplicate/conflicting ONNX runtimes — confirmed |
| PKG-08 | PKG-08: `open3d` is a heavy, apparently unused dependency | NEEDS_REPRO | open3d is heavy, apparently unused — needs code audit to confirm |
| PKG-09 | PKG-09: `torchmcubes` is installed but replaced at runtime | GENUINE | torchmcubes installed but replaced at runtime — confirmed, intentional workaroun... |
| PKG-10 | PKG-10: Git dependencies are unpinned | GENUINE | Git dependencies are unpinned — confirmed |
| PKG-11 | PKG-11: No `sf3d` extra | GENUINE | No sf3d extra — confirmed |
| PKG-12 | PKG-12: Core vs app vs API dependency split is unclear | GENUINE | Core vs app vs API dependency split unclear — confirmed |
| PKG-13 | PKG-13: Missing `license` metadata | GENUINE | Missing license metadata — confirmed |
| PKG-14 | PKG-14: No upper bounds on runtime dependencies | GENUINE | No upper bounds on runtime dependencies — confirmed |
| PKG-15 | PKG-15: No lockfile | GENUINE | No lockfile — confirmed |
| SEC-01 | SEC-01: Unauthenticated public Gradio tunnel | GENUINE | Unauthenticated public Gradio tunnel — critical, confirmed |
| SEC-02 | SEC-02: Full traceback returned to the UI | GENUINE | Full traceback returned to UI — confirmed |
| SEC-03 | SEC-03: API has no auth, rate limit or size cap | GENUINE | API has no auth, rate limit or size cap — confirmed |
| SEC-04 | SEC-04: Decompression-bomb / oversized image | GENUINE | Decompression-bomb / oversized image — confirmed |
| SEC-05 | SEC-05: API returns raw exception strings | GENUINE | API returns raw exception strings — confirmed |
| SEC-06 | SEC-06: API temp files are never deleted | GENUINE | API temp files never deleted — confirmed |
| SEC-07 | SEC-07: `trust_remote_code=True` executes arbitrary model co... | GENUINE | trust_remote_code=True executes arbitrary model code — critical, confirmed |
| SEC-08 | SEC-08: Kaggle kernels clone and execute moving `main` | GENUINE | Kaggle kernels clone and execute moving main — confirmed |
| SEC-09 | SEC-09: Pickle-based checkpoint loading | NEEDS_REPRO | Pickle-based checkpoint loading — needs verification of actual risk |
| SEC-10 | SEC-10: Silent fallback to a third-party SPAR3D mirror | GENUINE | Silent fallback to third-party SPAR3D mirror — confirmed, high severity |
| SEC-11 | SEC-11: Mutable, unpinned HF snapshots | GENUINE | Mutable, unpinned HF snapshots — confirmed |
| SEC-12 | SEC-12: Runtime downloads without checksums | GENUINE | Runtime downloads without checksums — confirmed |
| SEC-13 | SEC-13: Unpinned `pip install` in notebooks | GENUINE | Unpinned pip install in notebooks — confirmed |
| SEC-14 | SEC-14: HF token written to disk | NEEDS_REPRO | HF token written to disk — needs verification |
| SEC-15 | SEC-15: Global `sys.modules` monkeypatching | GENUINE | Global sys.modules monkeypatching — confirmed |
| SEC-16 | SEC-16: `sys.path.insert(0, ...)` import hijacking | GENUINE | sys.path.insert(0, ...) import hijacking — confirmed |
| SEC-17 | SEC-17: Env-controlled output path in figure script | GENUINE | Env-controlled output path in figure script — confirmed |
| SEC-18 | SEC-18: Results JSON written outside the output directory | GENUINE | Results JSON written outside output directory — confirmed |
| SEC-19 | SEC-19: Unbounded log growth / log injection | GENUINE | Unbounded log growth / log injection — confirmed |
| SEC-20 | SEC-20: `apt-get install` as root in a kernel | GENUINE | apt-get install as root in kernel — confirmed, low risk on Kaggle |
| SEC-21 | SEC-21: Dependencies are not pinned or hashed | GENUINE | Dependencies not pinned or hashed — confirmed |
| SEC-22 | SEC-22: Gradio upload has no size/format limit | GENUINE | Gradio upload has no size/format limit — confirmed |
| SEC-23 | SEC-23: Colab notebook uploads/downloads arbitrary files | GENUINE | Colab notebook uploads/downloads arbitrary files — confirmed |
| SEC-24 | SEC-24: `os.system("pip install ...")` in a kernel | GENUINE | os.system('pip install ...') in kernel — confirmed, medium risk |
| SEC-25 | SEC-25: Subprocess arguments built by concatenation | GENUINE | No specific rule — defaults to genuine (conservative) |
| SEC-26 | SEC-26: Untrusted JSON without schema validation | GENUINE | No specific rule — defaults to genuine (conservative) |
| SEC-27 | SEC-27: Shared temp output directory | GENUINE | No specific rule — defaults to genuine (conservative) |
| SEC-28 | SEC-28: Relative API output directory | GENUINE | No specific rule — defaults to genuine (conservative) |
| SEC-29 | SEC-29: Upload type is inferred by PIL, not validated | GENUINE | No specific rule — defaults to genuine (conservative) |
| SEC-30 | SEC-30: Third-party model code runs in-process | GENUINE | No specific rule — defaults to genuine (conservative) |
| TOOL-01 | TOOL-01: Figure script resolves the repository root incorrec... | GENUINE | Figure script resolves repo root incorrectly — confirmed |
| TOOL-02 | TOOL-02: Run directories in the figure script do not match t... | GENUINE | Run directories in figure script don't match docs — confirmed |
| TOOL-03 | TOOL-03: Font paths are non-standard with no fallback | GENUINE | Font paths non-standard with no fallback — confirmed |
| TOOL-04 | TOOL-04: Hardcoded engine columns can raise `KeyError` | GENUINE | Hardcoded engine columns can raise KeyError — confirmed |
| TOOL-05 | TOOL-05: `idxmin` on all-NaN data | GENUINE | idxmin on all-NaN data — confirmed edge case |
| TOOL-06 | TOOL-06: `merge_batches.py` promises a Pareto figure it neve... | GENUINE | merge_batches.py promises Pareto figure it never writes — confirmed |
| TOOL-07 | TOOL-07: `merge_batches.py` writes into the run directory | GENUINE | merge_batches.py writes into run directory — confirmed |
| TOOL-08 | TOOL-08: `json.load(open(...))` without a context manager | GENUINE | json.load(open(...)) without context manager — confirmed |
| TOOL-09 | TOOL-09: De-duplication is order-dependent | GENUINE | De-duplication is order-dependent — confirmed |
| TOOL-10 | TOOL-10: Merge includes rows with NaN CD | GENUINE | Merge includes rows with NaN CD — confirmed |
| TOOL-11 | TOOL-11: Duplicate index in figure lookup | GENUINE | Duplicate index in figure lookup — confirmed |
| TOOL-12 | TOOL-12: `Image.fromarray(np.asarray(img))` may fail | GENUINE | Image.fromarray(np.asarray(img)) may fail — confirmed edge case |
| TOOL-13 | TOOL-13: No CLI arguments | GENUINE | No CLI arguments — confirmed |
| TOOL-14 | TOOL-14: Figure outputs go to a gitignored directory | GENUINE | Figure outputs go to gitignored directory — confirmed |
| TOOL-15 | TOOL-15: `select_1000.py` trusts the report schema | GENUINE | select_1000.py trusts report schema — confirmed |
| TOOL-16 | TOOL-16: Forced list silently truncated if larger than the t... | GENUINE | Forced list silently truncated — confirmed edge case |
| TOOL-17 | TOOL-17: `valid_ids_from_report` assumes a dict | GENUINE | valid_ids_from_report assumes dict — confirmed |
| TOOL-18 | TOOL-18: No lint/format/type configuration | GENUINE | No lint/format/type configuration — confirmed |
| TOOL-19 | TOOL-19: No CI workflow | GENUINE | No CI workflow — confirmed (we're adding one now) |
| TOOL-20 | TOOL-20: Inconsistent shebangs/executable bits | GENUINE | Inconsistent shebangs/executable bits — confirmed |

## Recommended Labels

Apply these labels to GitHub issues:

- `triage:genuine` — Real issue, should be fixed
- `triage:needs-repro` — Cannot determine without reproduction
- `triage:false-alarm` — Not actually an issue
- `triage:wont-fix` — Real but not worth fixing