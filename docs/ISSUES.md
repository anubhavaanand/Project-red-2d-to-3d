# Issue register: security, correctness, packaging and documentation

Full-read audit of the repository at commit `66a0b23` (branch
`kilo/analog-crane-ucm`). This file is a local register: nothing here is filed on GitHub
yet. Each entry has a stable ID, labels, exact file/line evidence, impact and concrete
guidance.

**How to use**

1. Pick an ID.
2. Read the full entry and the referenced code.
3. Claim it on the GitHub issue (after filing) or in the PR description.
4. Follow [`CONTRIBUTING.md`](../CONTRIBUTING.md) §6-§7 for workflow and verification.

**Severity**

- `critical` — RCE, unauthenticated abuse, or published numbers are wrong.
- `high` — exploitable weakness, silent corruption, crash on a supported path, or a
  reproducibility break.
- `medium` — edge-case defect or real bug with a workaround.
- `low` — docs, style, minor robustness.

**Label format**

Each entry lists labels as `` `type` `` · `` `severity:*` `` · `` `component:*` `` ·
`` `status:*` ``. Definitions are in [`LABELS.md`](LABELS.md).

**Scope note**

Findings are based on source reading, not on a GPU execution of every path. Entries marked
`status:needs-repro` require a runtime check; the rest are confirmed from the code or data.

---

## 1. Security and supply chain (`SEC`)

### `SEC-01` Unauthenticated public Gradio tunnel
`security` · `severity:critical` · `component:app` · `status:confirmed`
- **Where:** `app.py:303-313`
- **Finding:** `app.launch(server_name="0.0.0.0", share=True, show_error=True)` creates a
  public `gradio.live` tunnel with no auth or quota.
- **Impact:** Anyone with the link can trigger GPU inference, causing cost/DoS and exposing
  the demo beyond the operator's intent.
- **Fix:** Default to `share=False` and `127.0.0.1`; require an explicit env flag for public
  sharing; put auth/proxy in front of any public deployment.
- **Good first issue:** no

### `SEC-02` Full traceback returned to the UI
`security` · `severity:high` · `component:app` · `status:confirmed`
- **Where:** `app.py:165-167`, `app.py:312`
- **Finding:** `generate_3d` returns `f"Error: {exc}\n\n{tb}"` (the source includes a leading
  emoji); `show_error=True` also renders exception detail.
- **Impact:** Leaks absolute paths, versions and model internals.
- **Fix:** Log the traceback server-side; return a stable code and short message; set
  `show_error=False` publicly.
- **Good first issue:** yes

### `SEC-03` API has no auth, rate limit or size cap
`security` · `severity:high` · `component:api` · `status:confirmed`
- **Where:** `src/api/server.py:73-104`
- **Finding:** `POST /predict` accepts any upload and runs a full engine; no auth, no rate
  limit, no body-size or concurrency limit.
- **Impact:** Unauthenticated GPU/disk exhaustion.
- **Fix:** Add API-key/mTLS, a body-size limit (for example 10 MB), per-client rate limiting
  and a bounded inference queue; return 413/429.
- **Good first issue:** no

### `SEC-04` Decompression-bomb / oversized image
`security` · `severity:high` · `component:api` · `status:confirmed`
- **Where:** `src/api/server.py:84-85`, `app.py:244-248`
- **Finding:** `Image.open(...).convert("RGB")` runs on untrusted bytes with no
  `Image.MAX_IMAGE_PIXELS` or dimension check.
- **Impact:** A small file can expand to gigabytes and OOM the process.
- **Fix:** Cap bytes, set `Image.MAX_IMAGE_PIXELS`, reject large dimensions, verify magic
  bytes.
- **Good first issue:** no

### `SEC-05` API returns raw exception strings
`security` · `severity:medium` · `component:api` · `status:confirmed`
- **Where:** `src/api/server.py:102-103`
- **Finding:** `raise HTTPException(500, detail=str(e))`.
- **Impact:** Internal paths, model errors and library versions are exposed.
- **Fix:** Log the detail, return a generic message plus a request ID.
- **Good first issue:** yes

### `SEC-06` API temp files are never deleted
`security` · `severity:medium` · `component:api` · `status:confirmed`
- **Where:** `src/api/server.py:31-32,90-98`
- **Finding:** Every prediction writes a new `.glb`/`.ply` into `./outputs` and returns it;
  nothing removes it.
- **Impact:** Disk exhaustion; local users can read generated models.
- **Fix:** Per-request temp dir + `BackgroundTask` cleanup, or stream bytes without disk.
- **Good first issue:** yes

### `SEC-07` `trust_remote_code=True` executes arbitrary model code
`security` · `severity:critical` · `component:lgm` · `status:confirmed`
- **Where:** `src/engines/lgm_engine.py:129-133`
- **Finding:** `MVDreamPipeline.from_pretrained(..., trust_remote_code=True)` downloads and
  executes Python from the model repo.
- **Impact:** A compromised/typosquatted repo gives RCE.
- **Fix:** Vendor/pin the pipeline code, load with `trust_remote_code=False`, pin a revision
  and verify hashes; document the trust model.
- **Good first issue:** no

### `SEC-08` Kaggle kernels clone and execute moving `main`
`security` · `severity:high` · `component:kaggle` · `status:confirmed`
- **Where:** `kaggle/batch1/kernel_batch1_triposr_lgm.ipynb`,
  `kaggle/batch2/kernel_batch2_trellis_spar3d.ipynb`,
  `kaggle/triposr_only/kernel_triposr_only.ipynb`,
  `kaggle/verify/kernel_verify_render.ipynb`
- **Finding:** Each kernel runs `git clone --depth 1 -b main .../image-to-3d.git` and imports
  `src` from it.
- **Impact:** A force-push changes the code behind published numbers; reruns are not
  reproducible.
- **Fix:** Clone a tag/SHA, `git checkout <sha>`, record it in `code_sha.txt` and verify it.
  Also fix the URL: the configured `origin` is `Project-RED-image-to-3d`.
- **Good first issue:** no

### `SEC-09` Pickle-based checkpoint loading
`security` · `severity:high` · `component:triposr` · `status:needs-repro`
- **Where:** `src/engines/triposr_engine.py:140-144` (TripoSR `model.ckpt`)
- **Finding:** `TSR.from_pretrained` loads a `.ckpt` via `torch.load`; pickle executes code on
  deserialization.
- **Impact:** A tampered checkpoint/mirror runs arbitrary code.
- **Fix:** Prefer `safetensors`; otherwise verify a pinned hash and load in an isolated worker.
- **Good first issue:** no

### `SEC-10` Silent fallback to a third-party SPAR3D mirror
`security` · `severity:high` · `component:spar3d` · `status:confirmed`
- **Where:** `src/engines/spar3d_engine.py:128-142`
- **Finding:** `except Exception` falls back to `zimengxiong/Modelr-SPAR3D` for any error,
  including network/auth.
- **Impact:** A different, unvetted model can be reported as SPAR3D.
- **Fix:** Fall back only on an explicit gated-repo error, log loudly, record the resolved
  repo/revision in metadata, and let callers opt out.
- **Good first issue:** no

### `SEC-11` Mutable, unpinned HF snapshots
`security` · `severity:high` · `component:trellis` · `component:lgm` · `status:confirmed`
- **Where:** `src/engines/trellis_engine.py:82-84`,
  `src/engines/lgm_engine.py:116-118`
- **Finding:** `snapshot_download(...)`/`hf_hub_download(...)` do not pin `revision`.
- **Impact:** Upstream changes weights; reruns differ and a compromised revision executes.
- **Fix:** Pin a SHA/tag, record it, verify hashes.
- **Good first issue:** no

### `SEC-12` Runtime downloads without checksums
`security` · `severity:medium` · `component:preprocessing` · `component:metrics` · `status:confirmed`
- **Where:** `src/preprocessing/background_removal.py:12`,
  `src/evaluation/psnr_lpips.py:35`
- **Finding:** `rembg.new_session` and `lpips.LPIPS(net="vgg")` fetch weights on first use
  with no hash check.
- **Impact:** A compromised CDN silently changes segmentation or perceptual metrics.
- **Fix:** Pre-download in setup, pin versions/hashes, fail closed on mismatch.
- **Good first issue:** no

### `SEC-13` Unpinned `pip install` in notebooks
`security` · `severity:high` · `component:kaggle` · `component:notebooks` · `status:confirmed`
- **Where:** `kaggle/*/*.ipynb`, `notebooks/colab_demo.ipynb`
- **Finding:** Packages are installed by name, often `--no-deps`; git URLs are unpinned.
- **Impact:** Dependency confusion, API drift, non-reproducible runs.
- **Fix:** Commit a hashed lockfile, pin git deps to SHAs, record `pip freeze`.
- **Good first issue:** no

### `SEC-14` HF token written to disk
`security` · `severity:medium` · `component:kaggle` · `status:needs-repro`
- **Where:** `kaggle/*/*.ipynb` — `login(token=_hf_token, write_to_disk=True)`
- **Finding:** The token is persisted to the HF cache.
- **Impact:** Leak if the home/cache directory is archived.
- **Fix:** `write_to_disk=False`; never copy dotfiles into `/kaggle/working`.
- **Good first issue:** yes

### `SEC-15` Global `sys.modules` monkeypatching
`security` · `severity:medium` · `component:lgm` · `component:spar3d` · `status:confirmed`
- **Where:** `src/engines/lgm_engine.py:35-73`, `src/engines/spar3d_engine.py:22-85`
- **Finding:** Fake `xformers`, `pkg_resources`, `transparent_background` and patched
  `transformers` internals are installed globally.
- **Impact:** Other libraries get partial/fake APIs; silent feature loss; order dependence.
- **Fix:** Scope shims to a subprocess or a dedicated compatibility module; restore
  `sys.modules` afterwards; never fake a failed dependency.
- **Good first issue:** no

### `SEC-16` `sys.path.insert(0, ...)` import hijacking
`security` · `severity:medium` · `component:app` · `component:engines` · `status:confirmed`
- **Where:** `app.py:50`, `src/engines/triposr_engine.py:115-123`,
  `src/engines/lgm_engine.py:93-103`, `src/engines/trellis_engine.py:58-70`,
  `src/engines/spar3d_engine.py:120-122`
- **Finding:** Runtime prepends of `/tmp` clones and repo roots to `sys.path`.
- **Impact:** A file in a writable directory can shadow a real dependency.
- **Fix:** Install packages properly, use explicit importlib paths, append rather than
  prepend, validate the resolved module file.
- **Good first issue:** no

### `SEC-17` Env-controlled output path in figure script
`security` · `severity:low` · `component:scripts` · `status:confirmed`
- **Where:** `scripts/rebuild_figures.py:43-44,403-408`
- **Finding:** `BENCH_ROOT` is joined and written to without validation.
- **Impact:** A compromised CI environment can write outside the checkout.
- **Fix:** Validate that the output path is inside the repo, or require an explicit CLI
  output directory.
- **Good first issue:** yes

### `SEC-18` Results JSON written outside the output directory
`security` · `severity:low` · `component:evaluation` · `status:confirmed`
- **Where:** `src/evaluation/benchmark_runner.py:909-910`
- **Finding:** Writes `<output_dir>/<engine>_results.json` and
  `<output_dir>/../<engine>_results.json`.
- **Impact:** Surprising writes outside the requested directory.
- **Fix:** Write only inside `output_dir`; make the parent copy an explicit option.
- **Good first issue:** yes

### `SEC-19` Unbounded log growth / log injection
`security` · `severity:low` · `component:kaggle` · `status:confirmed`
- **Where:** `kaggle/*/*.ipynb` (`Tee`, `benchmark_log.txt`),
  `src/evaluation/benchmark_runner.py:771-777`
- **Finding:** Logs append forever and include raw sample IDs.
- **Impact:** Disk growth; IDs containing newlines can forge log lines.
- **Fix:** Rotate/size-cap logs, escape newlines, sanitise IDs.
- **Good first issue:** yes

### `SEC-20` `apt-get install` as root in a kernel
`security` · `severity:low` · `component:kaggle` · `status:confirmed`
- **Where:** `kaggle/triposr_only/kernel_triposr_only.ipynb` (CPU fallback)
- **Finding:** Runs `apt-get update` and `apt-get install -y libosmesa6...`.
- **Impact:** Mutates the shared image; may fail or pull unexpected packages.
- **Fix:** Use a prebuilt image/userspace wheel.
- **Good first issue:** yes

### `SEC-21` Dependencies are not pinned or hashed
`security` · `severity:high` · `component:packaging` · `status:confirmed`
- **Where:** `requirements.txt`, `pyproject.toml:7-74`
- **Finding:** Mostly `>=`; git deps have no `ref`; no lockfile/hash checking.
- **Impact:** Non-reproducible builds and exposure to a compromised release.
- **Fix:** Add a lockfile, pin the CUDA stack, use `--require-hashes` in CI.
- **Good first issue:** no

### `SEC-22` Gradio upload has no size/format limit
`security` · `severity:medium` · `component:app` · `status:confirmed`
- **Where:** `app.py:244-248`
- **Finding:** `gr.Image(type="pil")` accepts arbitrary uploads; no dimension/byte limit.
- **Impact:** Memory exhaustion (same class as `SEC-04`).
- **Fix:** Validate dimensions and file size in `generate_3d`.
- **Good first issue:** yes

### `SEC-23` Colab notebook uploads/downloads arbitrary files
`security` · `severity:low` · `component:notebooks` · `status:confirmed`
- **Where:** `notebooks/colab_demo.ipynb` (`files.upload()`, `files.download()`)
- **Finding:** Any file is accepted; a generated file is served for download.
- **Impact:** Low (ephemeral), but normalises an unsafe pattern.
- **Fix:** Validate uploads are images, cap size, label the notebook a demo.
- **Good first issue:** yes

### `SEC-24` `os.system("pip install ...")` in a kernel
`security` · `severity:medium` · `component:kaggle` · `status:confirmed`
- **Where:** `kaggle/batch2/kernel_batch2_trellis_spar3d.ipynb` (xformers fallback)
- **Finding:** Shell command built as a string; failures are silent.
- **Impact:** Shell metacharacters in the environment can alter the command.
- **Fix:** `subprocess.run([...], check=True)` with captured output.
- **Good first issue:** yes

### `SEC-25` Subprocess arguments built by concatenation
`security` · `severity:low` · `component:kaggle` · `status:needs-repro`
- **Where:** `kaggle/*/*.ipynb`
- **Finding:** Some URLs/paths are concatenated before being passed to subprocesses.
- **Impact:** A malformed path can clone/build in the wrong directory.
- **Fix:** Use `pathlib`, validate destinations, avoid shell strings.
- **Good first issue:** yes

### `SEC-26` Untrusted JSON without schema validation
`security` · `severity:low` · `component:scripts` · `status:confirmed`
- **Where:** `gso1000/select_1000.py:70-80`, `datasets/gso_eval_v1.json`
- **Finding:** `json.loads` output is indexed without type/size validation.
- **Impact:** A malformed report can crash or over-allocate.
- **Fix:** Validate with a Pydantic model and cap file size.
- **Good first issue:** yes

### `SEC-27` Shared temp output directory
`security` · `severity:medium` · `component:app` · `status:confirmed`
- **Where:** `app.py:66-67`
- **Finding:** Outputs go to `tempfile.gettempdir()/image_to_3d_outputs` with default
  permissions.
- **Impact:** Other local users can read generated models; filenames are predictable.
- **Fix:** Per-user directory with `0o700`, or a per-session temp dir removed on exit.
- **Good first issue:** no

### `SEC-28` Relative API output directory
`security` · `severity:low` · `component:api` · `status:confirmed`
- **Where:** `src/api/server.py:31-32`
- **Finding:** `OUTPUT_DIR = "./outputs"` is resolved against the process CWD.
- **Impact:** Writes may land in an unexpected place or fail depending on launch.
- **Fix:** Resolve an absolute configured path and fail fast if not writable.
- **Good first issue:** yes

### `SEC-29` Upload type is inferred by PIL, not validated
`security` · `severity:medium` · `component:app` · `component:api` · `status:confirmed`
- **Where:** `app.py:244-248`, `src/api/server.py:84-85`
- **Finding:** Any bytes PIL can decode are accepted.
- **Impact:** Parser-level memory corruption on old Pillow versions; unexpected formats.
- **Fix:** Allow-list extensions/MIME and magic bytes; keep Pillow patched.
- **Good first issue:** no

### `SEC-30` Third-party model code runs in-process
`security` · `severity:high` · `component:engines` · `status:confirmed`
- **Where:** `src/engines/*`
- **Finding:** All engines import/execute third-party Python/CUDA with the benchmark's
  privileges.
- **Impact:** Any compromised engine dependency compromises the host.
- **Fix:** Isolate engines in subprocesses/containers, drop privileges, document trust.
- **Good first issue:** no

## 2. REST API (`API`)

### `API-01` API bypasses background removal
`bug` · `severity:high` · `component:api` · `status:confirmed`
- **Where:** `src/api/server.py:83-87`
- **Finding:** `predict` converts the upload to RGB and calls the engine; `BackgroundRemover`
  is never used.
- **Impact:** Engines see a different input distribution than the benchmark; quality drops and
  API results are not comparable to the paper.
- **Fix:** Lazily create a shared `BackgroundRemover`, call `remove()` before the engine, and
  record `preprocessing` in the response metadata.
- **Good first issue:** no

### `API-02` Alpha channel is discarded
`bug` · `severity:high` · `component:api` · `status:confirmed`
- **Where:** `src/api/server.py:85`
- **Finding:** `convert("RGB")` drops the mask used for foreground cropping.
- **Impact:** Incorrect centring/cropping (see `API-01`).
- **Fix:** Keep RGBA through preprocessing; composite only at the engine boundary if needed.
- **Good first issue:** yes

### `API-03` LGM returns a mesh, contradicting `/engines`
`bug` · `severity:medium` · `component:api` · `status:confirmed`
- **Where:** `src/api/server.py:38-71` vs `src/api/server.py:89-98`
- **Finding:** `/engines` advertises LGM as `gaussians (.ply)`, but `LGMEngine.predict`
  defaults to `extract_mesh=True`, so the API exports `.glb`.
- **Impact:** Clients get a different artefact than documented.
- **Fix:** Add an explicit `output_format` form field or pass `extract_mesh=False`.
- **Good first issue:** yes

### `API-04` SPAR3D and SF3D are missing from the API
`bug` · `severity:medium` · `component:api` · `status:confirmed`
- **Where:** `src/api/server.py:19-29,38-71`
- **Finding:** `get_engine` handles only TripoSR/LGM/TRELLIS; SPAR3D/SF3D are in
  `EngineType` and `src/engines/__init__.py`.
- **Impact:** `EngineType.SPAR3D` returns 400; the API is incomplete.
- **Fix:** Add both engines and metadata, or document the supported subset and remove them
  from the enum.
- **Good first issue:** yes

### `API-05` Output directory ignores configuration
`bug` · `severity:medium` · `component:api` · `status:confirmed`
- **Where:** `src/api/server.py:31-32` vs `src/config.py:26-29`
- **Finding:** `ExportConfig.output_dir` is never read; the API hardcodes `./outputs`.
- **Impact:** Configuration has no effect.
- **Fix:** Load `AppConfig` at startup and use `export.output_dir`.
- **Good first issue:** yes

### `API-06` No background cleanup task
`bug` · `severity:medium` · `component:api` · `status:confirmed`
- **Where:** `src/api/server.py:90-98`
- **Finding:** Files are returned but never deleted (see `SEC-06`).
- **Impact:** Disk growth.
- **Fix:** Attach `BackgroundTask(Path.unlink)` or use a swept temp directory.
- **Good first issue:** yes

### `API-07` Engine cache is not thread-safe
`bug` · `severity:medium` · `component:api` · `status:confirmed`
- **Where:** `src/api/server.py:17-29`
- **Finding:** Check-then-construct without a lock.
- **Impact:** Two concurrent first-requests can load the same multi-GB model twice and OOM.
- **Fix:** Double-checked `threading.Lock`, or preload at startup.
- **Good first issue:** no

### `API-08` Blocking inference in an async endpoint
`performance` · `severity:high` · `component:api` · `status:confirmed`
- **Where:** `src/api/server.py:73-104`
- **Finding:** `async def predict` calls the synchronous GPU-bound `engine.predict`.
- **Impact:** The event loop is blocked; health checks and all other requests stall.
- **Fix:** Use a sync `def` (threadpool) or `run_in_threadpool`; bound concurrency.
- **Good first issue:** no

### `API-09` No timeout or concurrency cap
`performance` · `severity:medium` · `component:api` · `status:confirmed`
- **Where:** `src/api/server.py:73-104`
- **Finding:** Nothing limits simultaneous inferences or per-request duration.
- **Impact:** GPU memory exhaustion and head-of-line blocking.
- **Fix:** Bounded executor/semaphore plus a per-request timeout.
- **Good first issue:** no

### `API-10` Inconsistent error status for invalid engine
`bug` · `severity:low` · `component:api` · `status:confirmed`
- **Where:** `src/api/server.py:75-81`
- **Finding:** Invalid enum form values yield 422; `get_engine`'s `ValueError` yields 400.
- **Impact:** Clients cannot rely on one status for "bad engine".
- **Fix:** Validate explicitly and return 400 for all invalid values.
- **Good first issue:** yes

### `API-11` No CORS policy documented
`documentation` · `severity:low` · `component:api` · `status:confirmed`
- **Where:** `src/api/server.py:14`
- **Finding:** No CORS middleware.
- **Impact:** Browser clients cannot call the API cross-origin.
- **Fix:** Decide and document a policy; add `CORSMiddleware` only if needed.
- **Good first issue:** yes

### `API-12` Health check does not report readiness
`bug` · `severity:low` · `component:api` · `status:confirmed`
- **Where:** `src/api/server.py:34-36`
- **Finding:** `/health` always returns healthy even if no engine can load.
- **Impact:** Orchestrators route traffic to a broken instance.
- **Fix:** Add `/ready` that checks engine load status.
- **Good first issue:** yes

### `API-13` Engine metadata is hardcoded and can drift
`maintainability` · `severity:low` · `component:api` · `status:confirmed`
- **Where:** `src/api/server.py:38-71`
- **Finding:** Names/papers/latency/VRAM are duplicated from `get_engine_info()`/`README`.
- **Impact:** API and docs disagree after a model change.
- **Fix:** Build `list_engines` from `get_engine_info()`.
- **Good first issue:** yes

### `API-14` All engine failures become 500
`bug` · `severity:low` · `component:api` · `status:confirmed`
- **Where:** `src/api/server.py:102-103`
- **Finding:** Load failures, OOM and bad input are indistinguishable.
- **Impact:** Clients cannot retry intelligently.
- **Fix:** Map known failures to 503/507/422; keep 500 for unexpected errors.
- **Good first issue:** no

### `API-15` Uploaded image is not closed
`maintainability` · `severity:low` · `component:api` · `status:confirmed`
- **Where:** `src/api/server.py:84-85`
- **Finding:** `Image.open(...)` is used without a context manager.
- **Impact:** File-descriptor pressure under load.
- **Fix:** `with Image.open(...) as im:` and copy if needed.
- **Good first issue:** yes

### `API-16` Output directory created as an import side effect
`maintainability` · `severity:low` · `component:api` · `status:confirmed`
- **Where:** `src/api/server.py:31-32`
- **Finding:** Importing the module creates a directory.
- **Impact:** Surprising side effects in tests/library use.
- **Fix:** Create it in a startup hook/lifespan.
- **Good first issue:** yes

### `API-17` No API tests
`testing` · `severity:high` · `component:api` · `component:ci` · `status:confirmed`
- **Where:** repository root (no `tests/`)
- **Finding:** No `TestClient` smoke test for `/health`, `/engines` or `/predict`.
- **Impact:** The API can break undetected.
- **Fix:** Add `tests/test_api.py` with a mocked engine and a tiny PNG.
- **Good first issue:** no

### `API-18` Relative imports break direct execution
`bug` · `severity:medium` · `component:api` · `status:confirmed`
- **Where:** `src/api/server.py:9-12`
- **Finding:** `from ..config import ...` requires `uvicorn src.api.server:app`;
  `python src/api/server.py` raises `ImportError`.
- **Impact:** The obvious run path fails.
- **Fix:** Add a `__main__` block and document the module path, or use absolute imports.
- **Good first issue:** yes

## 3. Gradio app (`APP`)

### `APP-01` UI omits SPAR3D although the README advertises it
`bug` · `severity:high` · `component:app` · `status:confirmed`
- **Where:** `app.py:249-253`, `README.md:9-16`
- **Finding:** Radio choices are `["TripoSR","TRELLIS","LGM"]`.
- **Impact:** Users cannot run the advertised fourth engine from the demo.
- **Fix:** Add SPAR3D to the choices and `_get_engine`, or remove the claim.
- **Good first issue:** yes

### `APP-02` `_get_engine` cannot construct SPAR3D
`bug` · `severity:high` · `component:app` · `status:confirmed`
- **Where:** `app.py:77-90`
- **Finding:** The `if/elif` chain handles only TripoSR/TRELLIS/LGM.
- **Impact:** `APP-01` cannot be fixed by editing the radio choices alone.
- **Fix:** Use a registry mapping names to constructors.
- **Good first issue:** no

### `APP-03` Output filename collisions
`bug` · `severity:medium` · `component:app` · `status:confirmed`
- **Where:** `app.py:132`
- **Finding:** `output_{int(time.time()*1000)}` can collide.
- **Impact:** Concurrent requests overwrite each other's files.
- **Fix:** Use `uuid4().hex` or a `NamedTemporaryFile`.
- **Good first issue:** yes

### `APP-04` Preview renderer silently returns nothing
`bug` · `severity:low` · `component:app` · `status:confirmed`
- **Where:** `app.py:170-208`
- **Finding:** `_render_preview` swallows all exceptions and returns `None`.
- **Impact:** No preview and no explanation (common on headless hosts).
- **Fix:** Log the exception and return a placeholder with a reason.
- **Good first issue:** yes

### `APP-05` Device chosen at import time
`bug` · `severity:low` · `component:app` · `status:confirmed`
- **Where:** `app.py:65`
- **Finding:** `DEVICE = ... __import__("torch").cuda.is_available()` runs once at import.
- **Impact:** Changing `CUDA_VISIBLE_DEVICES` after import has no effect; obscure form.
- **Fix:** Import torch normally and resolve the device in a function.
- **Good first issue:** yes

### `APP-06` `sys.path.insert(0, ...)`
`maintainability` · `severity:medium` · `component:app` · `status:confirmed`
- **Where:** `app.py:50`
- **Finding:** Repo root is prepended at import.
- **Impact:** Can shadow installed packages; order-dependent imports.
- **Fix:** Run as a module or install the package.
- **Good first issue:** no

### `APP-07` Example paths are CWD-dependent
`bug` · `severity:medium` · `component:app` · `status:confirmed`
- **Where:** `app.py:231-235,305`
- **Finding:** `EXAMPLES` uses relative `examples/*.jpg`; the directory is created only in
  `__main__`.
- **Impact:** Broken images or errors when started elsewhere/imported by a server.
- **Fix:** Resolve from `Path(__file__).parent`; create in `build_ui`.
- **Good first issue:** yes

### `APP-08` No cleanup of the app output directory
`performance` · `severity:medium` · `component:app` · `status:confirmed`
- **Where:** `app.py:66-67`
- **Finding:** `OUTPUT_DIR` is never pruned.
- **Impact:** Disk growth; stale models remain downloadable if exposed.
- **Fix:** Retention policy or per-session temp dirs.
- **Good first issue:** yes

### `APP-09` Models are never unloaded
`performance` · `severity:medium` · `component:app` · `status:confirmed`
- **Where:** `app.py:62,77-90`
- **Finding:** Every selected engine stays in `_engines` and on the GPU.
- **Impact:** VRAM exhaustion when trying all engines.
- **Fix:** LRU cache with `unload_model()`, or a clear-models action.
- **Good first issue:** no

### `APP-10` Latency includes background removal
`bug` · `severity:low` · `component:app` · `status:confirmed`
- **Where:** `app.py:111-129`
- **Finding:** `t0` starts before `BackgroundRemover.remove`.
- **Impact:** Displayed latency is not comparable to the benchmark's engine-only number.
- **Fix:** Time preprocessing and inference separately.
- **Good first issue:** yes

### `APP-11` Vertex colours multiplied without range check
`bug` · `severity:low` · `component:app` · `status:confirmed`
- **Where:** `app.py:178-181`
- **Finding:** `(mesh.vertex_colors * 255).astype(np.uint8)` assumes `[0,1]`.
- **Impact:** A `0..255` engine wraps around and renders noise.
- **Fix:** Normalise based on `max()` before scaling.
- **Good first issue:** yes

### `APP-12` Description numbers can drift
`documentation` · `severity:low` · `component:app` · `status:confirmed`
- **Where:** `app.py:215-229`
- **Finding:** The latency/VRAM/CD/F table is hardcoded.
- **Impact:** The demo can advertise stale numbers.
- **Fix:** Load the canonical CSV/JSON at startup.
- **Good first issue:** yes

### `APP-13` No input validation before inference
`bug` · `severity:medium` · `component:app` · `status:confirmed`
- **Where:** `app.py:244-248,107-109`
- **Finding:** Only `image is None` is checked; size/mode/content are not.
- **Impact:** Memory abuse and confusing engine errors (see `SEC-22`, `SEC-29`).
- **Fix:** Validate dimensions/file size and convert to RGBA explicitly.
- **Good first issue:** yes

### `APP-14` Long inference blocks the Gradio worker
`performance` · `severity:medium` · `component:app` · `status:confirmed`
- **Where:** `app.py:278-283`
- **Finding:** The click handler is synchronous with no queue/cancel.
- **Impact:** One user's 30 s inference blocks others; no cancellation.
- **Fix:** Queue with a concurrency limit and a cancel event.
- **Good first issue:** no

### `APP-15` Heavy imports at module import
`performance` · `severity:medium` · `component:app` · `status:confirmed`
- **Where:** `app.py:51-56`
- **Finding:** All engine modules are imported at startup.
- **Impact:** Slow cold start; unnecessary CUDA/diffusers imports.
- **Fix:** Import engines lazily inside `_get_engine`.
- **Good first issue:** no

### `APP-16` No model-load error handling
`bug` · `severity:low` · `component:app` · `status:confirmed`
- **Where:** `app.py:77-90`
- **Finding:** `e.load_model()` is called without a try/except.
- **Impact:** Users see a traceback rather than "engine unavailable".
- **Fix:** Catch load errors, return a clear status, keep the failed engine out of the cache.
- **Good first issue:** yes

## 4. Engines (`ENG`)

### `ENG-01` TripoSR wraps `load_state_dict` repeatedly
`bug` · `severity:high` · `component:triposr` · `status:confirmed`
- **Where:** `src/engines/triposr_engine.py:133-137`
- **Finding:** Each `load_model()` captures the current `TSR.load_state_dict` (possibly the
  wrapper) and installs a new wrapper.
- **Impact:** Calling `load_model()` twice nests remappers and applies the key remap twice.
- **Fix:** Install the patch once (idempotent guard) or patch a bound instance.
- **Good first issue:** no

### `ENG-02` Remap wrapper ignores the caller's `strict`
`bug` · `severity:medium` · `component:triposr` · `status:confirmed`
- **Where:** `src/engines/triposr_engine.py:134-136`
- **Finding:** The wrapper accepts `strict` but always calls the original with `strict=False`.
- **Impact:** A caller requesting strict loading cannot detect missing keys.
- **Fix:** Forward the caller's `strict`.
- **Good first issue:** yes

### `ENG-03` Marching-cubes failure becomes an empty mesh
`bug` · `severity:high` · `component:triposr` · `status:confirmed`
- **Where:** `src/engines/triposr_engine.py:35-39`
- **Finding:** Any exception returns zero-length vertices/faces.
- **Impact:** The engine reports success with an empty mesh; metrics silently degrade.
- **Fix:** Let the exception propagate or return a typed failure the runner records.
- **Good first issue:** no

### `ENG-04` Marching-cubes helper assumes a torch tensor
`bug` · `severity:low` · `component:triposr` · `status:confirmed`
- **Where:** `src/engines/triposr_engine.py:28-30`
- **Finding:** `level.dim()` is outside the `try`; a NumPy array raises `AttributeError`.
- **Impact:** Fragile drop-in replacement for `torchmcubes`.
- **Fix:** Accept array-likes or document the tensor-only contract.
- **Good first issue:** yes

### `ENG-05` Foreground-mask heuristic is brittle
`bug` · `severity:medium` · `component:triposr` · `component:trellis` · `component:spar3d` · `status:confirmed`
- **Where:** `src/engines/triposr_engine.py:209-215`,
  `src/engines/trellis_engine.py:104-110`, `src/engines/spar3d_engine.py:218-224`
- **Finding:** `alpha.min() > 250` treats any lower alpha as a real mask;
  `np.any(rgb < 248)` treats JPEG noise as foreground.
- **Impact:** White objects lose their mask; noisy images gain speckle masks.
- **Fix:** Use a proper alpha threshold plus largest-connected-component filtering; share one
  helper across engines.
- **Good first issue:** no

### `ENG-06` Orientation test assumes a closed surface
`bug` · `severity:medium` · `component:triposr` · `status:confirmed`
- **Where:** `src/engines/triposr_engine.py:232-240`
- **Finding:** `np.sum(v0 * np.cross(v1, v2))` is the signed volume of tetrahedra with the
  world origin. For a closed, consistently wound surface it is translation-invariant; for an
  open/non-manifold marching-cubes mesh it is not.
- **Impact:** If the extracted mesh has holes or inconsistent winding, the sign can be wrong,
  and the code flips a correct mesh.
- **Fix:** Check watertightness/winding (`trimesh.Trimesh.is_watertight`,
  `fix_normals()`) before using the sign, or compute the volume about the centroid.
- **Good first issue:** no

### `ENG-07` Interior colour offsets are absolute
`bug` · `severity:medium` · `component:triposr` · `status:confirmed`
- **Where:** `src/engines/triposr_engine.py:160-162`
- **Finding:** Offsets `0.0, -0.002, -0.005` are fixed in model units.
- **Impact:** Large/small objects sample the wrong depth; thin shells get grey interiors.
- **Fix:** Scale offsets by the renderer radius or bbox diagonal.
- **Good first issue:** no

### `ENG-08` Unconditional CUDA cache calls
`bug` · `severity:low` · `component:triposr` · `component:lgm` · `component:trellis` · `component:spar3d` · `status:confirmed`
- **Where:** `src/engines/triposr_engine.py:263`, `lgm_engine.py:215`,
  `trellis_engine.py:95`, `spar3d_engine.py:286`
- **Finding:** `torch.cuda.empty_cache()` is called regardless of device.
- **Impact:** Fails on a CPU-only torch build; misleading.
- **Fix:** Guard with `torch.cuda.is_available()`.
- **Good first issue:** yes

### `ENG-09` No `unload_model` despite the runner hook
`bug` · `severity:medium` · `component:engines` · `component:evaluation` · `status:confirmed`
- **Where:** `src/evaluation/benchmark_runner.py:934-939`; no engine defines it
- **Finding:** The runner checks `hasattr(engine, "unload_model")`, but no engine implements it.
- **Impact:** Models stay in VRAM/RAM between engines; long runs OOM.
- **Fix:** Implement `unload_model()` on each engine and add it to the interface.
- **Good first issue:** no

### `ENG-10` LGM xformers shim detection is wrong
`bug` · `severity:high` · `component:lgm` · `status:confirmed`
- **Where:** `src/engines/lgm_engine.py:35-41`
- **Finding:** The guard requires `memory_efficient_attention` **and** `unbind` on the real
  package; `unbind` is not a stable public attribute.
- **Impact:** The fake shim can override a working xformers install.
- **Fix:** Detect with `importlib.util.find_spec` and only shim when the import actually fails.
- **Good first issue:** no

### `ENG-11` Shim `attn_bias` semantics are wrong
`bug` · `severity:high` · `component:lgm` · `status:needs-repro`
- **Where:** `src/engines/lgm_engine.py:45-55`
- **Finding:** `attn_bias` is passed as `attn_mask` to `scaled_dot_product_attention`; xformers
  bias objects are not dense masks.
- **Impact:** Crash or silently wrong attention when a bias is used.
- **Fix:** Convert the bias to a dense mask or reject non-`None` bias explicitly.
- **Good first issue:** no

### `ENG-12` Global `xformers` replacement
`bug` · `severity:high` · `component:lgm` · `status:confirmed`
- **Where:** `src/engines/lgm_engine.py:60-72`
- **Finding:** A fake `xformers` module is inserted into `sys.modules`.
- **Impact:** Other libraries get a partial API; `find_spec` can fail; debugging is hard.
- **Fix:** Keep the shim subprocess-local or restore the previous modules in `finally`.
- **Good first issue:** no

### `ENG-13` LGM silently drops mismatched checkpoint keys
`bug` · `severity:high` · `component:lgm` · `status:confirmed`
- **Where:** `src/engines/lgm_engine.py:120-125`
- **Finding:** Only keys present with matching shape are kept, then
  `load_state_dict(..., strict=False)`.
- **Impact:** Missing weights stay at random init; outputs degrade without warning.
- **Fix:** Log dropped/missing counts, fail below a match threshold, record in metadata.
- **Good first issue:** no

### `ENG-14` LGM view reordering assumes a 4-tuple
`bug` · `severity:medium` · `component:lgm` · `status:needs-repro`
- **Where:** `src/engines/lgm_engine.py:169-177`
- **Finding:** `mv_image[1..3]` assumes at least four arrays in a known order.
- **Impact:** `IndexError` or silently wrong view assignment after a dependency update.
- **Fix:** Validate the return type/length and document the order; add a stub-pipeline test.
- **Good first issue:** no

### `ENG-15` Unconditional CUDA autocast
`bug` · `severity:medium` · `component:lgm` · `status:confirmed`
- **Where:** `src/engines/lgm_engine.py:192-194`
- **Finding:** `torch.autocast(device_type="cuda")` is used even on CPU.
- **Impact:** CPU-only inference raises.
- **Fix:** Select the device type from `self.device`.
- **Good first issue:** yes

### `ENG-16` Batch dimension assumed to be 1
`bug` · `severity:high` · `component:lgm` · `status:needs-repro`
- **Where:** `src/engines/lgm_engine.py:198-211`
- **Finding:** `squeeze(0)` and `opa_all.reshape(-1)` assume one Gaussian set; a batch > 1
  misaligns positions and opacities.
- **Impact:** Shape mismatch or silently wrong flattening.
- **Fix:** Assert the output shape and flatten all Gaussian dimensions consistently.
- **Good first issue:** no

### `ENG-17` Opacity thresholds are arbitrary and unvalidated
`methodology` · `severity:medium` · `component:lgm` · `status:confirmed`
- **Where:** `src/engines/lgm_engine.py:244-250`
- **Finding:** The fallback chain `0.10 -> 0.08 -> 0.03 -> all` is hardcoded.
- **Impact:** Objects with unusual opacity distributions get very different foreground sets.
- **Fix:** Document thresholds, record the chosen branch in metadata, make them configurable.
- **Good first issue:** no

### `ENG-18` Per-object percentile iso-level
`methodology` · `severity:medium` · `component:lgm` · `status:confirmed`
- **Where:** `src/engines/lgm_engine.py:291-294`
- **Finding:** `iso = np.percentile(nonzero, 48)` adapts to each object.
- **Impact:** Surface position is not comparable across objects; harder to reproduce.
- **Fix:** Use a fixed iso or document/record the percentile rule.
- **Good first issue:** no

### `ENG-19` `trimesh.split` on an empty mesh
`bug` · `severity:low` · `component:lgm` · `status:confirmed`
- **Where:** `src/engines/lgm_engine.py:309-317`
- **Finding:** `max(...)` over an empty `parts` raises.
- **Impact:** Caught by the outer handler and returns `None` with no diagnostic.
- **Fix:** Guard `if not parts: return None` with a log line.
- **Good first issue:** yes

### `ENG-20` Broad `except` in LGM mesh extraction
`bug` · `severity:medium` · `component:lgm` · `status:confirmed`
- **Where:** `src/engines/lgm_engine.py:350-352`
- **Finding:** Every error prints a message and returns `None`.
- **Impact:** The runner records a NaN CD with no structured error.
- **Fix:** Raise a typed exception or return `(mesh, error)`.
- **Good first issue:** no

### `ENG-21` TRELLIS OOM fallback can re-OOM in `finally`
`bug` · `severity:medium` · `component:trellis` · `status:confirmed`
- **Where:** `src/engines/trellis_engine.py:158-172`
- **Finding:** The `finally` moves all `offload_names` back to the GPU, including models just
  offloaded to recover from OOM.
- **Impact:** The fallback can immediately OOM again; the original error is masked.
- **Fix:** Keep the failed decoder on CPU; restructure with explicit stages.
- **Good first issue:** no

### `ENG-22` TRELLIS assumes the `mesh` key exists
`bug` · `severity:medium` · `component:trellis` · `status:confirmed`
- **Where:** `src/engines/trellis_engine.py:203-204`
- **Finding:** `outputs["mesh"][0]` is indexed without checking the returned format.
- **Impact:** `KeyError` if the pipeline returns only gaussians.
- **Fix:** Branch on `decode_mode` and validate `outputs` keys.
- **Good first issue:** yes

### `ENG-23` TRELLIS vertex colours assumed `[0,1]`
`bug` · `severity:medium` · `component:trellis` · `status:confirmed`
- **Where:** `src/engines/trellis_engine.py:208-216`
- **Finding:** `np.clip(attrs[..., :3], 0, 1)` without checking `0..255`.
- **Impact:** Colours can be all-black or all-white depending on the decoder.
- **Fix:** Inspect `max()` and normalise; record the source range.
- **Good first issue:** yes

### `ENG-24` TRELLIS uses two different DC conventions
`bug` · `severity:high` · `component:trellis` · `status:confirmed`
- **Where:** `src/engines/trellis_engine.py:265-272` vs `229-234`
- **Finding:** The color-transfer path converts `dc` to display RGB with
  `rgb = dc * 0.28209479 + 0.5`, but the Gaussian output writes raw
  `get_features[:, 0, :3]`. The engine therefore emits display RGB in one place and raw SH DC
  in another, with no marker on `GaussianCloud`.
- **Impact:** The PLY exporter (which assumes one convention) can produce too-dark or
  double-converted gaussians depending on which path produced the cloud.
- **Fix:** Pick one convention for `GaussianCloud.sh_coeffs` (prefer raw SH DC), convert only
  for display, and add the convention to the dataclass. See `EXP-06`.
- **Good first issue:** no

### `ENG-25` Mesh and Gaussian outputs use different coordinate frames
`bug` · `severity:medium` · `component:trellis` · `status:confirmed`
- **Where:** `src/engines/trellis_engine.py:251-253` vs `265-272`
- **Finding:** The mesh is rotated `Rx(-90)`; the Gaussian cloud is not.
- **Impact:** The two output formats are not aligned with each other.
- **Fix:** Apply the same rotation to both, or document the frame per format.
- **Good first issue:** no

### `ENG-26` Duplicate/unused `cfg_strength` parameters
`maintainability` · `severity:low` · `component:trellis` · `status:confirmed`
- **Where:** `src/engines/trellis_engine.py:44,52-54`
- **Finding:** `cfg_strength` is accepted, collapsed into `self.cfg_strength`, and never used.
- **Impact:** Callers think they are changing CFG.
- **Fix:** Remove the alias or wire it to both samplers.
- **Good first issue:** yes

### `ENG-27` Unsupported `radiance_field` output format
`bug` · `severity:low` · `component:trellis` · `status:confirmed`
- **Where:** `src/engines/trellis_engine.py:32`
- **Finding:** `OutputFormat` includes `"radiance_field"`, but `predict` handles only mesh and
  gaussian.
- **Impact:** Selecting it raises or silently returns nothing.
- **Fix:** Remove it or implement it.
- **Good first issue:** yes

### `ENG-28` `attach()` is untyped and unused
`maintainability` · `severity:low` · `component:trellis` · `status:confirmed`
- **Where:** `src/engines/trellis_engine.py:72-75`
- **Finding:** No type, no caller.
- **Impact:** Dead API surface; unclear ownership.
- **Fix:** Remove or add a typed `set_pipeline` used by the runner.
- **Good first issue:** yes

### `ENG-29` Import-time environment mutation
`maintainability` · `severity:low` · `component:trellis` · `status:confirmed`
- **Where:** `src/engines/trellis_engine.py:29-30`
- **Finding:** `os.environ.setdefault` runs at import.
- **Impact:** Importing the module changes global state; order-dependent behaviour.
- **Fix:** Set variables in `load_model()` before importing TRELLIS.
- **Good first issue:** yes

### `ENG-30` SPAR3D prune shim re-enables gradients
`bug` · `severity:medium` · `component:spar3d` · `status:confirmed`
- **Where:** `src/engines/spar3d_engine.py:44-59`
- **Finding:** `_prune_linear_layer` toggles `requires_grad` off/on and always creates a new
  `nn.Linear`.
- **Impact:** Changes training semantics for any model loaded after the shim; unnecessary
  autograd state.
- **Fix:** Use `torch.no_grad()` and preserve the original `requires_grad`.
- **Good first issue:** no

### `ENG-31` Dummy `transparent_background` silently disables removal
`bug` · `severity:high` · `component:spar3d` · `status:confirmed`
- **Where:** `src/engines/spar3d_engine.py:76-85`
- **Finding:** If the import fails, a dummy module is installed whose `process()` returns the
  input unchanged.
- **Impact:** SPAR3D may receive a background; quality drops with no warning.
- **Fix:** Do not fake a failed dependency; fail with an actionable message or implement a real
  fallback and log it.
- **Good first issue:** no

### `ENG-32` Silent foreground-crop fallback
`bug` · `severity:medium` · `component:spar3d` · `status:confirmed`
- **Where:** `src/engines/spar3d_engine.py:226-231`
- **Finding:** Any `foreground_crop` exception is printed and ignored.
- **Impact:** Uncropped input degrades results; no structured error.
- **Fix:** Record the fallback in metadata and let the caller decide.
- **Good first issue:** yes

### `ENG-33` Unconditional `tm.invert()`
`bug` · `severity:medium` · `component:spar3d` · `status:confirmed`
- **Where:** `src/engines/spar3d_engine.py:246-249`
- **Finding:** Inversion is inside a bare `try/except`.
- **Impact:** If upstream changes, normals are flipped; failures are swallowed.
- **Fix:** Check winding explicitly (`fix_normals`) and log the decision.
- **Good first issue:** no

### `ENG-34` `vis.to_color()` may not exist
`bug` · `severity:low` · `component:spar3d` · `status:confirmed`
- **Where:** `src/engines/spar3d_engine.py:190-194`
- **Finding:** The fallback calls `vis.to_color()` without checking for the method.
- **Impact:** `AttributeError` caught by the outer handler; colours silently missing.
- **Fix:** Use `hasattr`/`getattr` and return `None` with a log.
- **Good first issue:** yes

### `ENG-35` Texture assumed to be a PIL image
`bug` · `severity:medium` · `component:spar3d` · `status:confirmed`
- **Where:** `src/engines/spar3d_engine.py:164-168`
- **Finding:** `tex_img.convert("RGB")` assumes PIL; a NumPy texture raises.
- **Impact:** Colour extraction silently returns `None`.
- **Fix:** Normalise both PIL and ndarray inputs.
- **Good first issue:** yes

### `ENG-36` `mesh_out[0]` list assumption (SPAR3D)
`bug` · `severity:medium` · `component:spar3d` · `status:needs-repro`
- **Where:** `src/engines/spar3d_engine.py:244`
- **Finding:** `mesh_out[0] if isinstance(mesh_out, (list, tuple)) else mesh_out`.
- **Impact:** A dict or scene return type yields the wrong object.
- **Fix:** Validate the return type and raise a clear error.
- **Good first issue:** yes

### `ENG-37` SF3D hardcodes `low_vram_mode=True`
`bug` · `severity:medium` · `component:sf3d` · `status:confirmed`
- **Where:** `src/engines/sf3d_engine.py:63-67`
- **Finding:** Not an `__init__` parameter.
- **Impact:** Users with more VRAM cannot disable it.
- **Fix:** Add the parameter and pass it through.
- **Good first issue:** yes

### `ENG-38` SF3D `mesh_out[0]` list assumption
`bug` · `severity:medium` · `component:sf3d` · `status:needs-repro`
- **Where:** `src/engines/sf3d_engine.py:89`
- **Finding:** `tm = mesh_out[0]` with no type check.
- **Impact:** A `Trimesh` return is indexed as a vertex array.
- **Fix:** Validate the return type.
- **Good first issue:** yes

### `ENG-39` SF3D drops UVs and texture
`bug` · `severity:high` · `component:sf3d` · `status:confirmed`
- **Where:** `src/engines/sf3d_engine.py:90-93`
- **Finding:** `Mesh3D` is built from vertices/faces only; `uv_coords`/`texture_map` are never
  read from `tm.visual`.
- **Impact:** SF3D's textured output is lost in exports and renders.
- **Fix:** Extract `visual.uv` and `visual.material.image` and pass them to `Mesh3D`.
- **Good first issue:** no

### `ENG-40` SF3D `glob_dict` unused
`maintainability` · `severity:low` · `component:sf3d` · `status:confirmed`
- **Where:** `src/engines/sf3d_engine.py:83`
- **Finding:** The second return value is assigned but never used.
- **Impact:** Dead code; no model metadata.
- **Fix:** Record useful values in metadata or drop the variable.
- **Good first issue:** yes

### `ENG-41` SF3D CUDA-only device context
`bug` · `severity:low` · `component:sf3d` · `status:confirmed`
- **Where:** `src/engines/sf3d_engine.py:82`
- **Finding:** `with torch.cuda.device(self.device)` fails on CPU.
- **Impact:** CPU-only runs raise.
- **Fix:** Guard with `if self.device.startswith("cuda")`.
- **Good first issue:** yes

### `ENG-42` No shape/dtype validation in the engine contract
`bug` · `severity:medium` · `component:engines` · `status:confirmed`
- **Where:** `src/engines/base.py:7-27`
- **Finding:** `Mesh3D`/`GaussianCloud` accept any arrays; mismatched lengths fail later.
- **Impact:** Silent corruption or late crashes in export/rendering.
- **Fix:** Add `__post_init__` validation (finite, shapes, colour range) and `validate()`.
- **Good first issue:** no

### `ENG-43` `load_model` signature mismatch
`maintainability` · `severity:low` · `component:engines` · `status:confirmed`
- **Where:** `src/engines/base.py:29-32`, `src/engines/lgm_engine.py:105`
- **Finding:** `LGMEngine.load_model(self, big=True)` extends the abstract signature.
- **Impact:** Callers cannot rely on a uniform interface; type checkers complain.
- **Fix:** Move `big` to `__init__` or add a separate method.
- **Good first issue:** yes

### `ENG-44` `engines/__init__.__getattr__` does not cache
`maintainability` · `severity:low` · `component:engines` · `status:confirmed`
- **Where:** `src/engines/__init__.py:17-32`
- **Finding:** Each attribute access re-imports the module.
- **Impact:** Minor repeated import cost; inconsistent identity in edge cases.
- **Fix:** Cache resolved classes in a module-level dict.
- **Good first issue:** yes

### `ENG-45` Heavy imports at engine module import
`performance` · `severity:medium` · `component:engines` · `status:confirmed`
- **Where:** `src/engines/lgm_engine.py:21-25`, `trellis_engine.py:23-26`,
  `spar3d_engine.py:13-17`
- **Finding:** `torch`, `safetensors`, `huggingface_hub`, `scipy` are imported at module
  import.
- **Impact:** Importing one engine pulls in heavy dependencies; slows tests/startup.
- **Fix:** Move heavy imports into `load_model`/`predict`.
- **Good first issue:** no

### `ENG-46` No engine smoke tests
`testing` · `severity:high` · `component:engines` · `component:ci` · `status:confirmed`
- **Where:** repository root (no `tests/`)
- **Finding:** No test constructs an engine with stub weights or validates output shapes.
- **Impact:** Contract regressions are invisible.
- **Fix:** Add `tests/engines/test_contract.py` with fake models and tiny arrays.
- **Good first issue:** no

## 5. Evaluation, metrics and rendering (`EVAL`)

### `EVAL-01` README says 3 ICP iterations; code uses 5
`documentation` · `severity:medium` · `component:evaluation` · `status:confirmed`
- **Where:** `README.md:46` vs `src/evaluation/benchmark_runner.py:317`
- **Finding:** `align_mesh_to_gt(..., n_iter: int = 5)` is called without an override.
- **Impact:** The documented protocol does not match the code that produced the numbers.
- **Fix:** Set the default to 3 or update the README/results provenance.
- **Good first issue:** yes

### `EVAL-02` Rendered mesh uses a different transform than the metrics
`methodology` · `severity:critical` · `component:evaluation` · `status:confirmed`
- **Where:** `src/evaluation/benchmark_runner.py:384-436,807-832`
- **Finding:** `align_mesh_to_gt` computes `best_R/best_t` from the 72-pose CD search and
  `best_R_vis/best_t_vis` from a separate upright/colour-aware search. `pred_pts` (CD/F-score)
  uses the first; `aligned_mesh.vertices` (rendering) uses the second.
- **Impact:** The rendered mesh is not the mesh that was scored. PSNR/LPIPS and qualitative
  figures can correspond to a different alignment than the geometry metrics. The README says
  the optimal transform is applied to the vertices, so this also contradicts the documented
  protocol.
- **Fix:** Use one transform for both, or report both explicitly and apply the upright
  transform only to a clearly separated figure copy.
- **Good first issue:** no

### `EVAL-03` Vertex-colour gamma is not linearised despite the comment
`bug` · `severity:high` · `component:evaluation` · `status:confirmed`
- **Where:** `src/evaluation/benchmark_runner.py:504-514`
- **Finding:** The comment says sRGB vertex colours are pre-converted to linear with
  `pow(c, 2.2)`, but the code only does `vc_srgb.astype(np.float32) / 255.0`. There is no
  `** 2.2`, even though pyrender applies an output gamma conversion.
- **Impact:** Vertex-colour predictions are effectively treated as linear and then
  gamma-encoded, so they render too bright/washed out relative to the native-UV GT path.
  PSNR/LPIPS compare different transfer functions for vertex-colour engines.
- **Fix:** Apply `vc_srgb ** 2.2` after the `/255` (or remove the gamma claim and document the
  actual pipeline); add a render regression test comparing a known colour through both paths.
- **Good first issue:** no

### `EVAL-04` Mesh and point normalisation use different centres
`methodology` · `severity:medium` · `component:evaluation` · `status:confirmed`
- **Where:** `src/evaluation/benchmark_runner.py:257-267` vs `src/evaluation/metrics.py:7-18`
- **Finding:** `_normalize_vertices_robust` centres on the bounding-box centre;
  `normalize_point_cloud` centres on the mean. The predicted mesh is normalised with the
  former; its sampled points are then re-normalised with the latter. The GT points use the
  latter only.
- **Impact:** The rendered/exported mesh scale and the metric point scale are not the same.
  CD/F-score are computed on consistently normalised points, but the qualitative render and
  the exported OBJ can be framed/scaled differently from the scored cloud.
- **Fix:** Use one normalisation function for vertices and sampled points, or explicitly
  document the two-stage rule; add a scale assertion in the render path.
- **Good first issue:** no

### `EVAL-05` Render normalisation uses a magic radius window
`bug` · `severity:medium` · `component:evaluation` · `status:confirmed`
- **Where:** `src/evaluation/benchmark_runner.py:489-494`
- **Finding:** `if not already_normalized or max_r < 0.6 or max_r > 1.6:` silently
  re-normalises.
- **Impact:** Some meshes are framed differently, changing PSNR/LPIPS for reasons unrelated to
  reconstruction quality.
- **Fix:** Normalise once, explicitly, and trust the flag; remove the heuristic.
- **Good first issue:** no

### `EVAL-06` PSNR returns 100 dB for identical images
`bug` · `severity:medium` · `component:metrics` · `status:confirmed`
- **Where:** `src/evaluation/psnr_lpips.py:13-14`
- **Finding:** `if mse < 1e-10: return 100.0`.
- **Impact:** A perfect prediction is reported as 100 dB, skewing means and hiding
  duplicate-image bugs.
- **Fix:** Return `float("inf")` and document how means handle it, or use a clearly named
  capped convention.
- **Good first issue:** yes

### `EVAL-07` LPIPS model is not moved to a device
`bug` · `severity:low` · `component:metrics` · `status:confirmed`
- **Where:** `src/evaluation/psnr_lpips.py:34-39`
- **Finding:** `_lpips_vgg` is created on CPU and never `.to(...)`; the helper accepts CUDA
  tensors.
- **Impact:** Device mismatch if callers pass GPU tensors.
- **Fix:** Move the model to the input device or document CPU-only inputs.
- **Good first issue:** yes

### `EVAL-08` LPIPS tensor path skips normalisation
`bug` · `severity:low` · `component:metrics` · `status:confirmed`
- **Where:** `src/evaluation/psnr_lpips.py:18-28`
- **Finding:** Tensor inputs are returned as-is; NumPy inputs are scaled to `[-1,1]`.
- **Impact:** The same image gives different LPIPS depending on the container type.
- **Fix:** Normalise both paths identically.
- **Good first issue:** yes

### `EVAL-09` Texture lookup can select a novel-view PNG
`bug` · `severity:high` · `component:evaluation` · `status:confirmed`
- **Where:** `src/evaluation/benchmark_runner.py:102-123`
- **Finding:** The fallback `rglob("*.png")` excludes names containing `000`, `001`, `novel`
  or `input`, but `002.png`/`003.png` is accepted as the texture.
- **Impact:** GT renders can use a rendered view as the texture, corrupting PSNR/LPIPS and
  figures.
- **Fix:** Accept only known texture names or the material referenced by the `.obj`/`.mtl`;
  never glob arbitrary numbered views.
- **Good first issue:** no

### `EVAL-10` Silent grey fallback when texture baking fails
`bug` · `severity:high` · `component:evaluation` · `status:confirmed`
- **Where:** `src/evaluation/benchmark_runner.py:138-142`
- **Finding:** Missing UVs/image return uniform grey `ColorVisuals`.
- **Impact:** GT renders become grey; PSNR/LPIPS compare colour against grey without error.
- **Fix:** Return `None`/raise and mark the sample "GT render failed" with a reason.
- **Good first issue:** no

### `EVAL-11` Default sample selection is alphabetical, not the curated set
`bug` · `severity:high` · `component:evaluation` · `status:confirmed`
- **Where:** `src/evaluation/benchmark_runner.py:177-185`
- **Finding:** Without `object_ids`, the loader takes `sorted(dirs)[:num_samples]`.
- **Impact:** `BenchmarkRunner(num_samples=100)` does not reproduce the paper's
  `handpicked.txt` catalog; it evaluates the first 100 IDs alphabetically.
- **Fix:** Default `object_ids` to `handpicked.txt` when present, or make it required and
  document the difference.
- **Good first issue:** no

### `EVAL-12` `num_samples` is ignored with `object_ids`
`bug` · `severity:low` · `component:evaluation` · `status:confirmed`
- **Where:** `src/evaluation/benchmark_runner.py:178-183`
- **Finding:** The `object_ids` branch ignores `num_samples`.
- **Impact:** A caller can request 10 and get 100 with no warning.
- **Fix:** Validate lengths and raise on mismatch, or truncate explicitly.
- **Good first issue:** yes

### `EVAL-13` Alignment uses a 2048-point subsample
`methodology` · `severity:medium` · `component:evaluation` · `status:confirmed`
- **Where:** `src/evaluation/benchmark_runner.py:336-348`
- **Finding:** Both clouds are subsampled to 2048 for ICP, while metrics use 10,000.
- **Impact:** Alignment can be suboptimal for fine geometry; the protocol does not mention it.
- **Fix:** Document the subsample/seed or align on the full clouds if affordable.
- **Good first issue:** no

### `EVAL-14` Empty mesh/cloud crashes alignment
`bug` · `severity:medium` · `component:evaluation` · `status:confirmed`
- **Where:** `src/evaluation/benchmark_runner.py:325-348`
- **Finding:** Empty points make `sub_n == 0`; `cKDTree` raises.
- **Impact:** A degenerate prediction produces an unhandled exception instead of a recorded
  failure.
- **Fix:** Check for empty inputs and return a typed failure.
- **Good first issue:** yes

### `EVAL-15` Per-render `OffscreenRenderer` leak
`performance` · `severity:medium` · `component:evaluation` · `status:confirmed`
- **Where:** `src/evaluation/benchmark_runner.py:559-561,648-650`
- **Finding:** A new renderer is created per render; `delete()` is not in `finally`.
- **Impact:** EGL/GL context exhaustion on long runs; slow renders.
- **Fix:** Reuse a renderer per resolution or wrap creation in `try/finally`.
- **Good first issue:** no

### `EVAL-16` GT texture is not attached to the trimesh visual
`bug` · `severity:high` · `component:evaluation` · `status:confirmed`
- **Where:** `src/evaluation/benchmark_runner.py:605-614`
- **Finding:** `uv` is read, but `tm.visual` is not replaced with
  `TextureVisuals(uv=uv, image=tex_img)` before `pyrender.Mesh.from_trimesh(tm, material=mat)`.
- **Impact:** pyrender may ignore the UVs and render a flat/white texture, corrupting GT
  appearance metrics.
- **Fix:** Set `tm.visual` before constructing the pyrender mesh; add a render assertion.
- **Good first issue:** no

### `EVAL-17` GT mesh is reloaded for every engine
`performance` · `severity:medium` · `component:evaluation` · `status:confirmed`
- **Where:** `src/evaluation/benchmark_runner.py:590`
- **Finding:** `render_gso_mesh_view` calls `trimesh.load` each time.
- **Impact:** Repeated I/O and parsing for the same GT across four engines.
- **Fix:** Cache the loaded/normalised GT render per sample.
- **Good first issue:** no

### `EVAL-18` VRAM metric underestimates real usage
`methodology` · `severity:medium` · `component:evaluation` · `status:confirmed`
- **Where:** `src/evaluation/benchmark_runner.py:782-796`
- **Finding:** `max_memory_allocated()` excludes the CUDA context, allocator fragmentation and
  other processes.
- **Impact:** Reported peak VRAM is lower than `nvidia-smi`; cross-engine comparisons may be
  unfair.
- **Fix:** Also record `max_memory_reserved()`/`nvidia-smi` and document which is reported.
- **Good first issue:** no

### `EVAL-19` Latency excludes preprocessing
`methodology` · `severity:low` · `component:evaluation` · `status:confirmed`
- **Where:** `src/evaluation/benchmark_runner.py:779-790`
- **Finding:** `t_start` begins after background removal.
- **Impact:** End-to-end latency is understated; the column is engine-only but not labelled
  as such.
- **Fix:** Record `preprocess_s` and `engine_s`; label the reported column.
- **Good first issue:** yes

### `EVAL-20` PSNR/LPIPS size mismatch is unguarded
`bug` · `severity:low` · `component:evaluation` · `status:confirmed`
- **Where:** `src/evaluation/benchmark_runner.py:860-862`
- **Finding:** `compute_psnr` subtracts arrays without checking shapes.
- **Impact:** A resolution mismatch raises a broadcast error.
- **Fix:** Validate shapes in the metric functions.
- **Good first issue:** yes

### `EVAL-21` Stale qualitative artefacts are not refreshed
`bug` · `severity:low` · `component:evaluation` · `status:confirmed`
- **Where:** `src/evaluation/benchmark_runner.py:871-877`
- **Finding:** `gt_novel.png`/`input.png` are written only if absent.
- **Impact:** A rerun with a new renderer keeps old images, mixing provenance.
- **Fix:** Write to a run-specific directory or overwrite atomically.
- **Good first issue:** yes

### `EVAL-22` Mesh export can use an unaligned mesh
`bug` · `severity:medium` · `component:evaluation` · `status:confirmed`
- **Where:** `src/evaluation/benchmark_runner.py:821-824,882-899`
- **Finding:** `render_target` falls back to `recon.mesh` when GT points are missing; the
  exported OBJ is in the engine's raw frame/scale.
- **Impact:** Downloaded meshes are not comparable across engines.
- **Fix:** Always export the normalised mesh, or mark `aligned: false` in the filename.
- **Good first issue:** no

### `EVAL-23` Failed rows are never cached on resume
`bug` · `severity:medium` · `component:evaluation` · `status:confirmed`
- **Where:** `src/evaluation/benchmark_runner.py:919-927,699-709`
- **Finding:** Failed rows lack `chamfer_distance`; the resume filter drops them, so they are
  retried forever.
- **Impact:** A permanently failing object blocks completion and inflates runtime.
- **Fix:** Record failures with a stable error code and let resume skip them (bounded retries).
- **Good first issue:** no

### `EVAL-24` Per-sample broad `except` hides the traceback
`bug` · `severity:medium` · `component:evaluation` · `status:confirmed`
- **Where:** `src/evaluation/benchmark_runner.py:919-927`
- **Finding:** `str(e)` is stored; the traceback is not saved.
- **Impact:** Hard to diagnose failures in a 400-row run.
- **Fix:** Log `traceback.format_exc()` and store a stable error type.
- **Good first issue:** yes

### `EVAL-25` `save_results` swallows errors
`bug` · `severity:low` · `component:evaluation` · `status:confirmed`
- **Where:** `src/evaluation/benchmark_runner.py:929-932`
- **Finding:** `try: save_results(...) except Exception: pass`.
- **Impact:** A full disk/permission error is silent; results are lost.
- **Fix:** Log/retry and surface the error.
- **Good first issue:** yes

### `EVAL-26` Summary includes error rows
`bug` · `severity:low` · `component:evaluation` · `status:confirmed`
- **Where:** `src/evaluation/benchmark_runner.py:943-963`
- **Finding:** Rows with only `engine/sample_id/error` are included; means skip NaNs but the
  row count is misleading.
- **Impact:** Summary tables can disagree with the number of successful evaluations.
- **Fix:** Separate successful/failed rows and report counts.
- **Good first issue:** yes

### `EVAL-27` Library code installs global signal handlers
`maintainability` · `severity:medium` · `component:evaluation` · `status:confirmed`
- **Where:** `src/evaluation/benchmark_runner.py:736-744`
- **Finding:** `signal.signal` is called and never restored.
- **Impact:** Embedding the runner changes Ctrl-C/SIGTERM behaviour process-wide.
- **Fix:** Install handlers only in `__main__`, or save/restore and make it opt-in.
- **Good first issue:** no

### `EVAL-28` No result schema validation
`maintainability` · `severity:low` · `component:evaluation` · `status:confirmed`
- **Where:** `src/evaluation/benchmark_runner.py:943-944`
- **Finding:** `get_results_df` returns whatever list of dicts exists.
- **Impact:** Downstream scripts assume columns that may be absent.
- **Fix:** Define a row schema and validate before writing.
- **Good first issue:** no

### `EVAL-29` Duplicate Chamfer wrapper
`maintainability` · `severity:low` · `component:metrics` · `status:confirmed`
- **Where:** `src/evaluation/chamfer_distance.py:18-19`
- **Finding:** `compute_chamfer_distance_symmetric` is a one-line alias.
- **Impact:** Dead API surface.
- **Fix:** Remove it or re-export the main function.
- **Good first issue:** yes

### `EVAL-30` F-score tree naming is confusing
`maintainability` · `severity:low` · `component:metrics` · `status:confirmed`
- **Where:** `src/evaluation/f_score.py:9-21`
- **Finding:** `tree1` is built from `p1` but `d1` queries `p2`; precision uses `d2` (pred to
  GT), recall uses `d1` (GT to pred).
- **Impact:** Easy to invert precision/recall in future edits; current code is correct but
  fragile.
- **Fix:** Rename to `pred_tree`/`gt_tree` and `d_pred_to_gt`/`d_gt_to_pred`.
- **Good first issue:** yes

### `EVAL-31` No metric unit tests
`testing` · `severity:high` · `component:metrics` · `component:ci` · `status:confirmed`
- **Where:** repository root (no `tests/`)
- **Finding:** No tests assert known CD/F-score/PSNR values on simple shapes.
- **Impact:** A sign flip or threshold change can alter every published number unnoticed.
- **Fix:** Add `tests/evaluation/test_metrics.py` with analytic cases.
- **Good first issue:** no

### `EVAL-32` `gt_novel_Views` is not snake_case
`maintainability` · `severity:low` · `component:evaluation` · `status:confirmed`
- **Where:** `src/evaluation/benchmark_runner.py:46`
- **Finding:** The dataclass field is `gt_novel_Views`.
- **Impact:** Inconsistent style; easy to typo.
- **Fix:** Rename to `gt_novel_views` and update references.
- **Good first issue:** yes

### `EVAL-33` Legacy alias `render_novel_view`
`maintainability` · `severity:low` · `component:evaluation` · `status:confirmed`
- **Where:** `src/evaluation/benchmark_runner.py:669`
- **Finding:** `render_novel_view = render_novel_view_same_bg` keeps an old name alive.
- **Impact:** Two names for one function; unclear canonical API.
- **Fix:** Remove after updating callers, or mark deprecated.
- **Good first issue:** yes

### `EVAL-34` sRGB encoding only on the native-UV path
`methodology` · `severity:high` · `component:evaluation` · `status:confirmed`
- **Where:** `src/evaluation/benchmark_runner.py:568-569,653-654`
- **Finding:** `_srgb_encode_u8` is applied only when `use_native_uv` is true; vertex-colour
  renders return linear-ish values.
- **Impact:** Predicted and GT renders use different transfer functions, so PSNR/LPIPS are not
  measuring the same appearance space.
- **Fix:** Define one colour pipeline for both paths and add a render test.
- **Good first issue:** no

### `EVAL-35` OpenGL compatibility patch runs at import
`maintainability` · `severity:low` · `component:evaluation` · `status:confirmed`
- **Where:** `src/evaluation/benchmark_runner.py:65-89`
- **Finding:** `_ensure_pyopengl_compat()` executes at module import.
- **Impact:** Import side effect; can mask a genuinely broken OpenGL install.
- **Fix:** Call it explicitly before the first render.
- **Good first issue:** yes

### `EVAL-36` Missing dataset path returns an empty list, then a generic error
`bug` · `severity:low` · `component:evaluation` · `status:confirmed`
- **Where:** `src/evaluation/benchmark_runner.py:172-175,733-734`
- **Finding:** `load_gso_samples` prints and returns `[]`; the runner raises a generic
  `RuntimeError`.
- **Impact:** The root cause (typo in the path) is only in stdout.
- **Fix:** Raise `FileNotFoundError` with the path.
- **Good first issue:** yes

### `EVAL-37` GT colour length is not validated
`bug` · `severity:low` · `component:evaluation` · `status:confirmed`
- **Where:** `src/evaluation/benchmark_runner.py:342-347`
- **Finding:** `gt_colors` is silently ignored if the length mismatches.
- **Impact:** Colour-aware alignment is disabled without warning.
- **Fix:** Log when `gt_colors` is dropped.
- **Good first issue:** yes

### `EVAL-38` Upright threshold is a magic number
`methodology` · `severity:low` · `component:evaluation` · `status:confirmed`
- **Where:** `src/evaluation/benchmark_runner.py:405`
- **Finding:** `if R_cand[1, 1] < 0.82: break`.
- **Impact:** The upright constraint is undocumented and not configurable.
- **Fix:** Document and expose the threshold.
- **Good first issue:** yes

### `EVAL-39` Colour penalty weight is a magic number
`methodology` · `severity:low` · `component:evaluation` · `status:confirmed`
- **Where:** `src/evaluation/benchmark_runner.py:416-421`
- **Finding:** `col_pen = 0.12 * (...)`.
- **Impact:** The vis alignment objective is not reproducible from the README.
- **Fix:** Document and expose the weight.
- **Good first issue:** yes

### `EVAL-40` Alignment parameters are not exposed on the runner
`maintainability` · `severity:medium` · `component:evaluation` · `status:confirmed`
- **Where:** `src/evaluation/benchmark_runner.py:672-684`
- **Finding:** `n_iter`, yaw count and pitch candidates are hardcoded in `align_mesh_to_gt`.
- **Impact:** A new protocol requires editing library code; old runs cannot be reproduced by
  configuration.
- **Fix:** Add a frozen `ProtocolConfig` dataclass and pass it to the runner/aligner.
- **Good first issue:** no

## 6. Dataset and results data (`DATA`)

### `DATA-01` Handpicked loader semantics differ across call sites
`bug` · `severity:medium` · `component:data` · `status:confirmed`
- **Where:** `handpicked.txt:1-3`, `gso1000/select_1000.py:58-66`, `kaggle/*/*.ipynb`
- **Finding:** `select_1000.load_forced` only skips blank lines, while the notebooks skip `#`
  comments; `BenchmarkRunner` never reads the file.
- **Impact:** A `#` line can be treated as an object ID in one path and ignored in another.
- **Fix:** Centralise `load_object_ids()` and use it everywhere.
- **Good first issue:** no

### `DATA-02` Dataset manifest key mismatch
`documentation` · `severity:low` · `component:data` · `status:confirmed`
- **Where:** `datasets/gso_eval_v1.json` (`object_ids`) vs
  `gso1000/SELECTION_SPEC.md` (`objects`)
- **Finding:** Two key names for the same concept.
- **Impact:** Tooling expecting `objects` silently sees none.
- **Fix:** Standardise and validate with a schema.
- **Good first issue:** yes

### `DATA-03` Preview manifest is tracked like a run manifest
`documentation` · `severity:medium` · `component:data` · `status:confirmed`
- **Where:** `gso1000/manifest_1000.preview.json`
- **Finding:** `validated_ok: 1046` assumes every object validates; it is a dry-run artefact.
- **Impact:** Running it directly will raise on the first invalid object.
- **Fix:** Keep the `.preview` suffix and add a README warning; never consume without a
  validation report.
- **Good first issue:** yes

### `DATA-04` Mesh manifest paths are relative to a missing run directory
`documentation` · `severity:medium` · `component:data` · `status:confirmed`
- **Where:** `results/mesh_manifest.csv`
- **Finding:** Paths start with `batch1_100obj_v35/outputs/...`, but the repository has no such
  directory.
- **Impact:** The manifest cannot locate meshes without the external run tree.
- **Fix:** Document the expected root or store a `run_root` column.
- **Good first issue:** yes

### `DATA-05` `metadata` column is a Python repr, not JSON
`bug` · `severity:medium` · `component:data` · `status:confirmed`
- **Where:** `results/unified_results_4engines_100obj.csv`, `scripts/merge_batches.py:35`
- **Finding:** The CSV shows `"{'engine': 'TripoSR', ...}"` with single quotes (`str(dict)`),
  not JSON.
- **Impact:** Consumers cannot `json.loads(row["metadata"])`.
- **Fix:** Serialise with `json.dumps` before adding it to the row, or add a
  `metadata_json` column.
- **Good first issue:** yes

### `DATA-06` Results README column list does not match the CSV
`documentation` · `severity:low` · `component:data` · `status:confirmed`
- **Where:** `results/README.md:32-34`, `results/unified_results_4engines_100obj.csv`
- **Finding:** The README list omits `sample_id` and uses a different order.
- **Impact:** Minor confusion for downstream parsers.
- **Fix:** Generate the column list from the CSV header.
- **Good first issue:** yes

### `DATA-07` `datasets/gso_eval_v1.json` is not used by the loader
`maintainability` · `severity:medium` · `component:data` · `status:confirmed`
- **Where:** `datasets/gso_eval_v1.json`, `src/evaluation/benchmark_runner.py:164-254`
- **Finding:** The loader scans directories; it never reads the manifest.
- **Impact:** The manifest can drift from the actual dataset with no check.
- **Fix:** Accept a manifest path and validate IDs against it.
- **Good first issue:** no

### `DATA-08` Catalog verification date is hardcoded
`documentation` · `severity:low` · `component:data` · `status:confirmed`
- **Where:** `gso1000/select_1000.py:153`, `gso1000/SELECTION_SPEC.md:12`
- **Finding:** `catalog_verified_date` is a constant, not a checked fact.
- **Impact:** The manifest can claim a verification that never happened.
- **Fix:** Read the date from the validation report, or omit it in preview mode.
- **Good first issue:** yes

### `DATA-09` Mesh manifest has no run/commit column
`maintainability` · `severity:medium` · `component:data` · `status:confirmed`
- **Where:** `results/mesh_manifest.csv`
- **Finding:** It records paths/hashes but not the code SHA.
- **Impact:** A mesh cannot be traced to a protocol version.
- **Fix:** Add `code_sha` and `protocol_version`.
- **Good first issue:** yes

### `DATA-10` No schema/contract test for result CSVs
`testing` · `severity:medium` · `component:data` · `component:ci` · `status:confirmed`
- **Where:** `results/`, `scripts/merge_batches.py`
- **Finding:** Nothing validates row counts, engine names or required columns before
  publication.
- **Impact:** A partial run can be merged and published as complete.
- **Fix:** Add `scripts/validate_results.py` (400 rows, 4 engines, no NaN in key columns) and
  run it in CI.
- **Good first issue:** no

### `DATA-11` `handpicked.txt` rationale is free-text only
`maintainability` · `severity:low` · `component:data` · `status:confirmed`
- **Where:** `handpicked.txt:1-3`
- **Finding:** Why 26 duplicates and 8 degenerate objects were replaced is only in comments.
- **Impact:** Future maintainers cannot reconstruct the decision.
- **Fix:** Move the rationale to `results/README.md` or a sidecar.
- **Good first issue:** yes

### `DATA-12` No checksum/verification script for `gso_catalog.txt`
`maintainability` · `severity:low` · `component:data` · `status:confirmed`
- **Where:** `gso_catalog.txt`, `gso1000/SELECTION_SPEC.md`
- **Finding:** Verification is a manual curl one-liner that is not stored in the repo.
- **Impact:** The claim is not reproducible.
- **Fix:** Store the command/script and the API response hash.
- **Good first issue:** yes

### `DATA-13` `mesh_manifest.csv` hashes cover files not in git
`documentation` · `severity:low` · `component:data` · `status:confirmed`
- **Where:** `results/mesh_manifest.csv`
- **Finding:** Integrity is only checkable if the external run tree is available.
- **Impact:** Published meshes cannot be verified from the repository alone.
- **Fix:** State this in `results/README.md` and link the run archive.
- **Good first issue:** yes

## 7. Export (`EXP`)

### `EXP-01` GLB exporter drops UVs and textures
`bug` · `severity:high` · `component:export` · `status:confirmed`
- **Where:** `src/export/glb_exporter.py:9-13`
- **Finding:** Only vertices, faces and vertex colours are passed; `Mesh3D.uv_coords` and
  `texture_map` are ignored.
- **Impact:** SPAR3D/SF3D textured meshes are exported untextured.
- **Fix:** Build `TextureVisuals(uv=..., image=...)` when both are present and export the
  material; add a round-trip test.
- **Good first issue:** no

### `EXP-02` GLB export merges vertices by default
`bug` · `severity:medium` · `component:export` · `status:confirmed`
- **Where:** `src/export/glb_exporter.py:9-13`
- **Finding:** `trimesh.Trimesh(...)` defaults to `process=True`.
- **Impact:** Per-vertex colours/UV seams can be lost or mismatched; face indices change.
- **Fix:** Pass `process=False` and validate.
- **Good first issue:** yes

### `EXP-03` GLB export does not create the parent directory
`bug` · `severity:low` · `component:export` · `status:confirmed`
- **Where:** `src/export/glb_exporter.py:9-13`
- **Finding:** `t_mesh.export(filepath)` fails if the parent does not exist.
- **Impact:** First-run failures in the API/app.
- **Fix:** `Path(filepath).parent.mkdir(parents=True, exist_ok=True)`.
- **Good first issue:** yes

### `EXP-04` No finite/NaN validation before export
`bug` · `severity:medium` · `component:export` · `status:confirmed`
- **Where:** `src/export/glb_exporter.py:9-13`
- **Finding:** NaN/inf vertices are written without checks.
- **Impact:** Invalid GLB that viewers reject or render as garbage.
- **Fix:** Validate `np.isfinite(...).all()` and raise a clear error.
- **Good first issue:** yes

### `EXP-05` Float vertex colours passed to trimesh
`bug` · `severity:medium` · `component:export` · `status:confirmed`
- **Where:** `src/export/glb_exporter.py:12`
- **Finding:** `Mesh3D.vertex_colors` is `[0,1]` float, but `trimesh` expects uint8 or handles
  floats inconsistently.
- **Impact:** Colours can be black/white or raise, depending on the version.
- **Fix:** Convert to uint8 (`clip(c,0,1)*255`) before constructing the mesh.
- **Good first issue:** yes

### `EXP-06` PLY export assumes one SH convention for all engines
`bug` · `severity:high` · `component:export` · `status:confirmed`
- **Where:** `src/export/ply_exporter.py:41-47,81`; `lgm_engine.py:183`;
  `trellis_engine.py:265-272`
- **Finding:** `GaussianCloud.sh_coeffs` has no declared convention. LGM stores display RGB
  (`rgb`, 0..1); TRELLIS's Gaussian output stores raw SH DC (`get_features[:,0,:3]`). The PLY
  writer treats every value as 3DGS `f_dc`, which expects
  `f_dc = (rgb - 0.5) / 0.28209479`.
- **Impact:** LGM gaussians are wrong (too dark/clipped) in standard viewers; TRELLIS is only
  correct if the raw DC is written unchanged. The same exporter cannot serve both conventions.
- **Fix:** Add a `colorspace`/`dc_convention` field to `GaussianCloud`, convert display RGB to
  SH DC on export, and document the convention. See also `ENG-24`.
- **Good first issue:** no

### `EXP-07` Scale convention is undocumented
`bug` · `severity:medium` · `component:export` · `status:needs-repro`
- **Where:** `src/export/ply_exporter.py:82-83`
- **Finding:** `GaussianCloud.scales` is written directly; 3DGS PLY stores log-scale.
- **Impact:** If the engine outputs linear scales, exported splats are the wrong size.
- **Fix:** Document linear vs log and convert explicitly.
- **Good first issue:** no

### `EXP-08` Rotation quaternions are not normalised
`bug` · `severity:low` · `component:export` · `status:confirmed`
- **Where:** `src/export/ply_exporter.py:83-84`
- **Finding:** Quaternions are written as-is.
- **Impact:** Viewers assuming unit quaternions can distort splats.
- **Fix:** Normalise each quaternion and document the component order.
- **Good first issue:** yes

### `EXP-09` PLY lacks higher-order SH
`maintainability` · `severity:low` · `component:export` · `status:confirmed`
- **Where:** `src/export/ply_exporter.py:51-73`
- **Finding:** Only `f_dc_*` is written; no `f_rest_*`.
- **Impact:** View-dependent colour is lost; acceptable but should be stated.
- **Fix:** Document DC-only export.
- **Good first issue:** yes

### `EXP-10` ASCII PLY writer is slow for large clouds
`performance` · `severity:medium` · `component:export` · `status:confirmed`
- **Where:** `src/export/ply_exporter.py:75-85`
- **Finding:** One `f.write` per Gaussian with Python f-strings.
- **Impact:** 65k-500k gaussians take seconds and block the API event loop.
- **Fix:** Use `plyfile`/`numpy.savetxt` or a buffered join.
- **Good first issue:** no

### `EXP-11` PLY arrays are not length-checked
`bug` · `severity:medium` · `component:export` · `status:confirmed`
- **Where:** `src/export/ply_exporter.py:16-47`
- **Finding:** Positions/scales/rotations/opacities/colours are indexed independently.
- **Impact:** A mismatch raises mid-write, leaving a truncated file.
- **Fix:** Validate all lengths equal `n` before opening the file.
- **Good first issue:** yes

### `EXP-12` PLY writes zero normals
`maintainability` · `severity:low` · `component:export` · `status:confirmed`
- **Where:** `src/export/ply_exporter.py:58-60`
- **Finding:** `nx, ny, nz` are hardcoded to zero.
- **Impact:** Some viewers shade poorly; acceptable but undocumented.
- **Fix:** Omit the normal properties or document why they are zero.
- **Good first issue:** yes

### `EXP-13` Config allows `obj` but there is no OBJ exporter
`bug` · `severity:low` · `component:export` · `component:packaging` · `status:confirmed`
- **Where:** `src/config.py:27`, `src/export/`
- **Finding:** `ExportConfig.format` pattern is `^(glb|ply|obj)$`; only GLB/PLY exist.
- **Impact:** Selecting `obj` fails at runtime.
- **Fix:** Remove `obj` or implement an exporter.
- **Good first issue:** yes

### `EXP-14` No export round-trip tests
`testing` · `severity:high` · `component:export` · `component:ci` · `status:confirmed`
- **Where:** repository root (no `tests/`)
- **Finding:** Nothing verifies that an exported GLB/PLY reloads with the expected geometry and
  colours.
- **Impact:** Silent data loss in the primary deliverable.
- **Fix:** Add `tests/export/test_roundtrip.py` with a tetrahedron and a tiny Gaussian cloud.
- **Good first issue:** no

## 8. Scripts and tooling (`TOOL`)

### `TOOL-01` Figure script resolves the repository root incorrectly
`bug` · `severity:high` · `component:scripts` · `status:confirmed`
- **Where:** `scripts/rebuild_figures.py:43-44`
- **Finding:** `ROOT = parents[2]` and `BENCH = ROOT / "image-to-3d-benchmark"`. In this
  checkout `parents[2]` is `/workspace`, so `BENCH` becomes `/workspace/image-to-3d-benchmark`
  (does not exist).
- **Impact:** The script cannot find `results/`/`data/` unless the checkout is named exactly
  as assumed under the expected parent.
- **Fix:** `BENCH = Path(__file__).resolve().parents[1]` and remove the hardcoded directory
  name; add `--root`.
- **Good first issue:** yes

### `TOOL-02` Run directories in the figure script do not match the docs
`bug` · `severity:high` · `component:scripts` · `status:confirmed`
- **Where:** `scripts/rebuild_figures.py:77-78` vs `results/README.md:18`
- **Finding:** `FINAL_B1/B2` use `final_batch1_100obj`/`final_batch2_100obj`; docs say
  `batch1_100obj_v35`/`batch2_100obj_v35`.
- **Impact:** The script looks in the wrong place and fails or uses stale data.
- **Fix:** One source of truth for run directory names.
- **Good first issue:** yes

### `TOOL-03` Font paths are non-standard with no fallback
`bug` · `severity:high` · `component:scripts` · `status:confirmed`
- **Where:** `scripts/rebuild_figures.py:90-96`
- **Finding:** `/usr/share/fonts/TTF/DejaVuSans.ttf` is not the usual Debian path
  (`/usr/share/fonts/truetype/dejavu/...`).
- **Impact:** `get_font` raises on most Linux images; figure generation fails.
- **Fix:** Search candidate paths, fall back to `ImageFont.load_default()`, allow an env
  override.
- **Good first issue:** yes

### `TOOL-04` Hardcoded engine columns can raise `KeyError`
`bug` · `severity:medium` · `component:scripts` · `status:confirmed`
- **Where:** `scripts/rebuild_figures.py:340-348`
- **Finding:** `lat['TripoSR']`, `vram['LGM']`, etc. assume every engine is present.
- **Impact:** A partial run crashes instead of reporting the missing engine.
- **Fix:** Validate the engine set and fail with a clear message.
- **Good first issue:** yes

### `TOOL-05` `idxmin` on all-NaN data
`bug` · `severity:medium` · `component:scripts` · `status:confirmed`
- **Where:** `scripts/rebuild_figures.py:358-359`
- **Finding:** `sub["chamfer_distance"].idxmin()` raises if all values are NaN.
- **Impact:** Figure generation fails on a failed subset.
- **Fix:** Drop NaN rows and handle the empty case.
- **Good first issue:** yes

### `TOOL-06` `merge_batches.py` promises a Pareto figure it never writes
`documentation` · `severity:low` · `component:scripts` · `status:confirmed`
- **Where:** `scripts/merge_batches.py:1-7`
- **Finding:** The docstring lists `pareto_4models.png`, but `merge_all` only writes CSV/TeX.
- **Impact:** Users look for a file that is never produced.
- **Fix:** Implement the figure or remove the claim.
- **Good first issue:** yes

### `TOOL-07` `merge_batches.py` writes into the run directory
`bug` · `severity:low` · `component:scripts` · `status:confirmed`
- **Where:** `scripts/merge_batches.py:33`
- **Finding:** `out_dir = Path(search_dirs[0])` writes outputs into the first input run.
- **Impact:** Mutates inputs; surprising for read-only archives.
- **Fix:** Add an explicit `--out` directory.
- **Good first issue:** yes

### `TOOL-08` `json.load(open(...))` without a context manager
`maintainability` · `severity:low` · `component:scripts` · `status:confirmed`
- **Where:** `scripts/merge_batches.py:20`
- **Finding:** The file handle is not closed deterministically.
- **Impact:** ResourceWarning; on Windows, the file can stay locked.
- **Fix:** `with open(p) as f: json.load(f)`.
- **Good first issue:** yes

### `TOOL-09` De-duplication is order-dependent
`bug` · `severity:medium` · `component:scripts` · `status:confirmed`
- **Where:** `scripts/merge_batches.py:21-25`
- **Finding:** `seen` keeps the first `(engine, sample_id)` row; the winner depends on
  directory order.
- **Impact:** A stale row can override the final run.
- **Fix:** Prefer the newest run (`run`/`code_sha`) or fail on conflicting values.
- **Good first issue:** no

### `TOOL-10` Merge includes rows with NaN CD
`bug` · `severity:medium` · `component:scripts` · `status:confirmed`
- **Where:** `scripts/merge_batches.py:23`
- **Finding:** Only the presence of the key is checked, not that the value is finite.
- **Impact:** Summary means mix successful and failed rows.
- **Fix:** Require finite `chamfer_distance`, `f_score_0.1`, `psnr`, `lpips`.
- **Good first issue:** yes

### `TOOL-11` Duplicate index in figure lookup
`bug` · `severity:medium` · `component:scripts` · `status:confirmed`
- **Where:** `scripts/rebuild_figures.py:357,387`
- **Finding:** `set_index("engine")` then `sub.loc[eng]` returns a Series if duplicates exist.
- **Impact:** `r['chamfer_distance']` is a Series and the f-string fails.
- **Fix:** Drop duplicates or assert uniqueness before indexing.
- **Good first issue:** yes

### `TOOL-12` `Image.fromarray(np.asarray(img))` may fail
`bug` · `severity:low` · `component:scripts` · `status:confirmed`
- **Where:** `scripts/rebuild_figures.py:139-143`
- **Finding:** Palette/16-bit textures are converted without mode handling.
- **Impact:** Figure generation fails for some GSO textures.
- **Fix:** Convert PIL images with `.convert("RGB")`; handle ndarray dtypes.
- **Good first issue:** yes

### `TOOL-13` No CLI arguments
`maintainability` · `severity:low` · `component:scripts` · `status:confirmed`
- **Where:** `scripts/rebuild_figures.py:301-587`
- **Finding:** Input/output paths are constants/env only.
- **Impact:** Cannot run against an alternate checkout/output without editing code.
- **Fix:** Add `argparse` for root, CSV, run dirs and output.
- **Good first issue:** yes

### `TOOL-14` Figure outputs go to a gitignored directory
`documentation` · `severity:low` · `component:scripts` · `status:confirmed`
- **Where:** `scripts/rebuild_figures.py:403-408`, `.gitignore:35`
- **Finding:** `paper/figures/` is ignored, so generated figures are not versioned.
- **Impact:** Reviewers cannot see the exact figure used in the paper.
- **Fix:** Document where canonical figures live, or commit a hash manifest.
- **Good first issue:** yes

### `TOOL-15` `select_1000.py` trusts the report schema
`bug` · `severity:low` · `component:scripts` · `status:confirmed`
- **Where:** `gso1000/select_1000.py:70-80`
- **Finding:** `rep.get("objects", [])` assumes a dict; a list raises.
- **Impact:** Cryptic failure on a malformed report.
- **Fix:** Validate the top-level type and object fields.
- **Good first issue:** yes

### `TOOL-16` Forced list silently truncated if larger than the target
`bug` · `severity:medium` · `component:scripts` · `status:confirmed`
- **Where:** `gso1000/select_1000.py:89-91`
- **Finding:** `manifest = sorted(forced_valid)[:TARGET]` drops objects without warning.
- **Impact:** A forced object can disappear from the manifest.
- **Fix:** Raise if `len(forced_valid) > TARGET`, or record the dropped IDs.
- **Good first issue:** yes

### `TOOL-17` `valid_ids_from_report` assumes a dict
`bug` · `severity:low` · `component:scripts` · `status:confirmed`
- **Where:** `gso1000/select_1000.py:70-80`
- **Finding:** `rep.get` fails if the JSON root is a list.
- **Impact:** Unclear error.
- **Fix:** Type-check the root and report the expected schema.
- **Good first issue:** yes

### `TOOL-18` No lint/format/type configuration
`maintainability` · `severity:high` · `component:ci` · `status:confirmed`
- **Where:** repository root
- **Finding:** No `ruff`, `black`, `flake8`, `mypy`, `pre-commit` or `setup.cfg`.
- **Impact:** Style/type regressions are not caught; review is slower.
- **Fix:** Add `ruff` + `black` (line length 100) and a `pre-commit` config; run on changed
  files first.
- **Good first issue:** no

### `TOOL-19` No CI workflow
`testing` · `severity:high` · `component:ci` · `status:confirmed`
- **Where:** `.github/` (absent)
- **Finding:** No GitHub Actions runs tests, lint or a smoke import.
- **Impact:** Broken `main` is possible.
- **Fix:** Minimal workflow: install core deps, `py_compile`, metric/export tests.
- **Good first issue:** no

### `TOOL-20` Inconsistent shebangs/executable bits
`maintainability` · `severity:low` · `component:scripts` · `status:confirmed`
- **Where:** `scripts/merge_batches.py`, `scripts/rebuild_figures.py`,
  `gso1000/select_1000.py`
- **Finding:** Some scripts have shebangs, one does not; executable bits are inconsistent.
- **Impact:** `./scripts/rebuild_figures.py` fails.
- **Fix:** Add a shebang and `chmod +x` to all CLI scripts, or document `python`.
- **Good first issue:** yes

## 9. Packaging and dependencies (`PKG`)

### `PKG-01` Top-level package is named `src`
`bug` · `severity:high` · `component:packaging` · `status:confirmed`
- **Where:** `pyproject.toml:1-22`, `src/__init__.py`
- **Finding:** The import package is literally `src`; setuptools will install a top-level
  `src` package into site-packages.
- **Impact:** Generic name collision with any other project; unsafe in a shared environment.
- **Fix:** Rename to `image_to_3d_benchmark` (or `image3dbench`) with a `src/` layout and
  update imports; keep a compatibility shim if needed.
- **Good first issue:** no

### `PKG-02` `pyrender` is not a declared core dependency
`dependencies` · `severity:high` · `component:packaging` · `status:confirmed`
- **Where:** `pyproject.toml:7-22` vs `src/evaluation/benchmark_runner.py:487`
- **Finding:** Rendering imports `pyrender`, but only `requirements.txt` lists it.
- **Impact:** `pip install -e .` alone cannot render; PSNR/LPIPS silently become NaN via the
  broad `except`.
- **Fix:** Add `pyrender` (and `PyOpenGL`) to core dependencies or a `render` extra, and fail
  loudly if missing.
- **Good first issue:** yes

### `PKG-03` `gradio` is not a declared dependency
`dependencies` · `severity:high` · `component:packaging` · `status:confirmed`
- **Where:** `pyproject.toml:7-22`, `app.py:45`
- **Finding:** The app imports `gradio`, but package metadata does not require it.
- **Impact:** `pip install -e .` then `python app.py` fails.
- **Fix:** Add an `app` extra with `gradio`; document `pip install -e ".[app]"`.
- **Good first issue:** yes

### `PKG-04` `scikit-image` is not a declared core dependency
`dependencies` · `severity:high` · `component:packaging` · `status:confirmed`
- **Where:** `pyproject.toml:7-22`, `src/engines/triposr_engine.py:25-26`
- **Finding:** `_skimage_mc_func` imports `skimage.measure`; `scikit-image` is only in the
  LGM extra.
- **Impact:** TripoSR fails on a core install.
- **Fix:** Add `scikit-image` to core dependencies.
- **Good first issue:** yes

### `PKG-05` `packaging` is imported but not declared
`dependencies` · `severity:medium` · `component:packaging` · `status:confirmed`
- **Where:** `src/engines/spar3d_engine.py:14`
- **Finding:** `import packaging` at module import; not in `pyproject`/`requirements`.
- **Impact:** ImportError on minimal environments.
- **Fix:** Add `packaging` to core dependencies.
- **Good first issue:** yes

### `PKG-06` `numpy<2.0` conflicts with the Kaggle runtime
`dependencies` · `severity:high` · `component:packaging` · `status:confirmed`
- **Where:** `requirements.txt:4` vs `kaggle/*/*.ipynb` (`numpy==2.1.3`)
- **Finding:** Kernels pin NumPy 2.1.3; `requirements.txt` forbids NumPy 2.
- **Impact:** `pip install -r requirements.txt` downgrades NumPy and can break the torch/scipy
  ABI; both installation paths cannot be satisfied.
- **Fix:** Choose one supported range, update extras, test the CUDA stack against it.
- **Good first issue:** no

### `PKG-07` Duplicate/conflicting ONNX runtimes
`dependencies` · `severity:medium` · `component:packaging` · `status:confirmed`
- **Where:** `requirements.txt:17-18`
- **Finding:** `rembg[gpu]` pulls an ONNX runtime; `onnxruntime` is also listed.
- **Impact:** pip may install CPU and GPU runtimes; ambiguous provider selection.
- **Fix:** Choose `rembg[gpu]` or `rembg[cpu]`, drop the explicit runtime, or pin deliberately
  with a comment.
- **Good first issue:** yes

### `PKG-08` `open3d` is a heavy, apparently unused dependency
`dependencies` · `severity:low` · `component:packaging` · `status:needs-repro`
- **Where:** `requirements.txt:9`, `pyproject.toml:18`
- **Finding:** No `import open3d` in `src/` or `scripts/`; only notebooks may use it.
- **Impact:** Large install, slower CI, more supply-chain surface.
- **Fix:** Move to a `notebooks` extra or remove.
- **Good first issue:** yes

### `PKG-09` `torchmcubes` is installed but replaced at runtime
`dependencies` · `severity:medium` · `component:triposr` · `status:confirmed`
- **Where:** `requirements.txt:36`, `pyproject.toml:26`,
  `src/engines/triposr_engine.py:19-40`
- **Finding:** The code monkeypatches `marching_cubes` with a scikit-image CPU version and never
  uses `torchmcubes`; it is still a hard dependency.
- **Impact:** Unnecessary CUDA build failures for CPU-only users.
- **Fix:** Make it optional; document that skimage is the default.
- **Good first issue:** yes

### `PKG-10` Git dependencies are unpinned
`dependencies` · `severity:high` · `component:packaging` · `status:confirmed`
- **Where:** `pyproject.toml:39,42-45,64-70`, `requirements.txt:36,49,52,55`
- **Finding:** `spconv`, `flash-attn`, `diff-gaussian-rasterization`, `diffoctreerast` and
  `torchmcubes` are installed from moving git branches.
- **Impact:** A new upstream commit can break the build or change results.
- **Fix:** Pin every git dependency to a tag/SHA; record resolved SHAs in run metadata.
- **Good first issue:** no

### `PKG-11` No `sf3d` extra
`dependencies` · `severity:medium` · `component:packaging` · `status:confirmed`
- **Where:** `pyproject.toml:24-74`, `src/engines/sf3d_engine.py`
- **Finding:** `SF3DEngine` exists, but its CUDA extensions/checkout have no extra.
- **Impact:** Users cannot install SF3D support with the documented pattern.
- **Fix:** Add a `sf3d` extra mirroring the module docstring.
- **Good first issue:** yes

### `PKG-12` Core vs app vs API dependency split is unclear
`dependencies` · `severity:medium` · `component:packaging` · `status:confirmed`
- **Where:** `pyproject.toml:7-22`
- **Finding:** `fastapi`/`uvicorn`/`lpips`/`rembg` are core, but `gradio`/`pyrender` are not.
- **Impact:** `pip install -e .` is either too heavy or too light.
- **Fix:** Define `core`, `app`, `api`, `notebooks`, `all` extras and document them.
- **Good first issue:** no

### `PKG-13` Missing `license` metadata
`documentation` · `severity:low` · `component:packaging` · `status:confirmed`
- **Where:** `pyproject.toml:1-6`
- **Finding:** No `license` field despite the MIT `LICENSE`.
- **Impact:** Incomplete package metadata; some tools cannot detect the licence.
- **Fix:** Add `license = { text = "MIT" }` or the SPDX string.
- **Good first issue:** yes

### `PKG-14` No upper bounds on runtime dependencies
`dependencies` · `severity:medium` · `component:packaging` · `status:confirmed`
- **Where:** `requirements.txt`, `pyproject.toml`
- **Finding:** `torch>=2.1.0`, `trimesh>=4.0.0`, `transformers>=4.38.0` allow breaking majors.
- **Impact:** A fresh install can pull an incompatible release.
- **Fix:** Add tested upper bounds or a lockfile.
- **Good first issue:** no

### `PKG-15` No lockfile
`dependencies` · `severity:medium` · `component:packaging` · `status:confirmed`
- **Where:** repository root
- **Finding:** Only `requirements.txt`/`pyproject.toml`; no hashed lockfile.
- **Impact:** Two installs on the same day can differ.
- **Fix:** Generate and commit a lockfile for core + CUDA extras.
- **Good first issue:** no

## 10. Documentation and provenance (`DOC`)

### `DOC-01` README ICP iteration count is wrong
`documentation` · `severity:medium` · `component:docs` · `status:confirmed`
- **Where:** `README.md:46` vs `src/evaluation/benchmark_runner.py:317`
- **Finding:** README says "3-iteration Kabsch ICP"; the default is 5.
- **Impact:** The published protocol is misstated.
- **Fix:** Correct one side; if 5 was used, update the README and results provenance.
- **Good first issue:** yes

### `DOC-02` Repository URL is inconsistent
`documentation` · `severity:high` · `component:docs` · `status:confirmed`
- **Where:** `README.md:151`, `CITATION.cff`, `kaggle/*/*.ipynb`, `git remote -v`
- **Finding:** README/CITATION/kernels use `anubhavaanand/image-to-3d`; the configured `origin`
  is `anubhavaanand/Project-RED-image-to-3d`.
- **Impact:** Kernels clone a different repo (or 404); citation resolves elsewhere;
  reproduction fails.
- **Fix:** Choose the canonical URL and update README, CITATION, kernels and the remote; verify
  the clone in CI.
- **Good first issue:** no

### `DOC-03` SF3D is implemented but absent from the README paradigm table
`documentation` · `severity:medium` · `component:docs` · `status:confirmed`
- **Where:** `README.md:9-16`, `AGENTS.md`, `src/config.py:11`,
  `src/engines/__init__.py:13`
- **Finding:** SF3D is a fifth engine in code/config but not in the README's evaluated table.
- **Impact:** Readers cannot tell whether SF3D is part of the benchmark.
- **Fix:** Add a row with its status (ablation/not in headline run) or remove it from the
  enum/exports.
- **Good first issue:** yes

### `DOC-04` Run directory names disagree across docs and scripts
`documentation` · `severity:high` · `component:docs` · `status:confirmed`
- **Where:** `results/README.md:18`, `scripts/rebuild_figures.py:77-78`
- **Finding:** Docs say `batch1_100obj_v35`/`batch2_100obj_v35`; the script uses
  `final_batch1_100obj`/`final_batch2_100obj`.
- **Impact:** Reproduction instructions do not work as written.
- **Fix:** Standardise the names and reference one constant.
- **Good first issue:** yes

### `DOC-05` `PROJECT_STATUS.md` contains superseded numbers
`documentation` · `severity:low` · `component:docs` · `status:confirmed`
- **Where:** `docs/PROJECT_STATUS.md:1-20`
- **Finding:** It documents a 36-pose/σ=1.5/k=4 intermediate state; it self-labels as
  historical but the numbers are easy to mistake for current.
- **Impact:** A new contributor may implement the old protocol.
- **Fix:** Move to `docs/history/` or add a stronger "do not use these numbers" banner.
- **Good first issue:** yes

### `DOC-06` `COMPILE.md` section numbering and stray backtick
`documentation` · `severity:low` · `component:docs` · `status:confirmed`
- **Where:** `docs/COMPILE.md:27-45`
- **Finding:** Two `## 2` sections; line 38 has a stray backtick.
- **Impact:** Confusing navigation; broken Markdown.
- **Fix:** Renumber sections and fix the code fence.
- **Good first issue:** yes

### `DOC-07` Paper title differs between `app.py` and README
`documentation` · `severity:low` · `component:docs` · `status:confirmed`
- **Where:** `app.py:292-293` vs `README.md:7`
- **Finding:** The app uses an older title ("Triplanes, Gaussians, or Structured Latents?") than
  the README/CITATION.
- **Impact:** Citation confusion.
- **Fix:** Use the canonical title from `CITATION.cff` everywhere.
- **Good first issue:** yes

### `DOC-08` `CITATION.cff` release date may precede the tag
`documentation` · `severity:low` · `component:docs` · `status:needs-repro`
- **Where:** `CITATION.cff:13-14`
- **Finding:** `date-released: 2026-10-28`; the tag `v1.0.0` date is not recorded here.
- **Impact:** Citation metadata can be inconsistent with the repository.
- **Fix:** Generate `CITATION.cff` from the release, or verify the tag date.
- **Good first issue:** yes

### `DOC-09` "0 failures" is not independently verifiable
`documentation` · `severity:low` · `component:docs` · `status:confirmed`
- **Where:** `README.md:3`
- **Finding:** The 400-row CSV has no failures, but failed rows are dropped by the resume filter
  and never written to the canonical CSV.
- **Impact:** The claim is true for the published CSV but not provable for the raw run.
- **Fix:** Publish a run manifest with attempted/failed counts.
- **Good first issue:** no

### `DOC-10` `AGENTS.md` hardcodes local paths
`documentation` · `severity:low` · `component:docs` · `status:confirmed`
- **Where:** `AGENTS.md:7-9`
- **Finding:** Workspace/repo roots are the author's absolute paths.
- **Impact:** Cloud agents and new contributors see paths that do not exist.
- **Fix:** Use repository-relative paths and a "your checkout may differ" note.
- **Good first issue:** yes

### `DOC-11` No `CHANGELOG.md`
`documentation` · `severity:low` · `component:docs` · `status:confirmed`
- **Where:** repository root
- **Finding:** Protocol changes and fixes are only in git history.
- **Impact:** Users cannot tell which protocol version a result belongs to.
- **Fix:** Add `CHANGELOG.md` with protocol versions and keep it updated.
- **Good first issue:** yes

### `DOC-12` No `CODE_OF_CONDUCT.md`
`documentation` · `severity:low` · `component:docs` · `status:confirmed`
- **Where:** repository root
- **Finding:** No community guidelines.
- **Impact:** Unclear expectations for contributors.
- **Fix:** Add the Contributor Covenant.
- **Good first issue:** yes

### `DOC-13` No `SECURITY.md`
`documentation` · `severity:low` · `component:docs` · `status:confirmed`
- **Where:** repository root
- **Finding:** No private vulnerability-reporting process.
- **Impact:** Security issues may be opened publicly.
- **Fix:** Add `SECURITY.md` with a contact and supported versions.
- **Good first issue:** yes

### `DOC-14` `COMPILE.md` contradicts `pyproject.toml` on spconv
`documentation` · `severity:medium` · `component:docs` · `status:confirmed`
- **Where:** `docs/COMPILE.md:27-45` vs `pyproject.toml:42`
- **Finding:** The guide says "do not `pip install .`" from the spconv repo and recommends
  prebuilt wheels; `pyproject.toml` still declares `spconv @ git+...` as an extra.
- **Impact:** `pip install -e ".[trellis]"` follows the discouraged path.
- **Fix:** Remove the git spconv dependency and document the wheel install.
- **Good first issue:** yes

### `DOC-15` README install order can produce a broken environment
`documentation` · `severity:medium` · `component:docs` · `status:confirmed`
- **Where:** `README.md:89-94`
- **Finding:** `pip install -e .` then `pip install -r requirements.txt` can resolve conflicting
  versions (see `PKG-06`).
- **Impact:** Users get a broken environment after following the README.
- **Fix:** Provide one canonical install command and test it in CI.
- **Good first issue:** no

## 11. Notebooks (`NB`)

### `NB-01` Colab demo uses a different segmentation model
`bug` · `severity:high` · `component:notebooks` · `status:confirmed`
- **Where:** `notebooks/colab_demo.ipynb` (`birefnet-general`) vs
  `src/preprocessing/background_removal.py:10` (`isnet-general-use`)
- **Finding:** The demo uses `birefnet-general`; the benchmark uses `isnet-general-use`.
- **Impact:** Different alpha quality and therefore different reconstructions from the same
  image.
- **Fix:** Use the benchmark's model/config, or state that the demo is not comparable.
- **Good first issue:** no

### `NB-02` Colab demo composites on white and feeds RGB
`bug` · `severity:high` · `component:notebooks` · `status:confirmed`
- **Where:** `notebooks/colab_demo.ipynb` (background-removal cell)
- **Finding:** It pastes the RGBA result onto white and converts to RGB, then calls TripoSR
  directly; the benchmark keeps RGBA and lets the engine crop.
- **Impact:** Different foreground handling and results.
- **Fix:** Use `TriposrEngine.predict` on the RGBA image, or document the difference.
- **Good first issue:** no

### `NB-03` Colab demo likely calls `extract_mesh` incorrectly
`bug` · `severity:high` · `component:notebooks` · `status:needs-repro`
- **Where:** `notebooks/colab_demo.ipynb` (TripoSR cell)
- **Finding:** `model.extract_mesh(scene_codes, resolution=256, threshold=25.0)` omits the
  `has_vertex_color` positional argument that `TriposrEngine` passes.
- **Impact:** Likely `TypeError` or a different default (no vertex colours).
- **Fix:** Call `extract_mesh(scene_codes, True, resolution=256)` as the engine does.
- **Good first issue:** yes

### `NB-04` Unused imports/helpers in batch kernels
`maintainability` · `severity:low` · `component:kaggle` · `status:confirmed`
- **Where:** `kaggle/batch1/kernel_batch1_triposr_lgm.ipynb` (`hashlib`, `random`,
  `count_valid`)
- **Finding:** Several imports/functions are defined but never used.
- **Impact:** Noise; makes review harder.
- **Fix:** Remove dead code.
- **Good first issue:** yes

### `NB-05` `RUN_TRELLIS` is a dead branch
`maintainability` · `severity:low` · `component:kaggle` · `status:confirmed`
- **Where:** `kaggle/batch2/kernel_batch2_trellis_spar3d.ipynb`
- **Finding:** `RUN_TRELLIS = True` and the `if not RUN_TRELLIS: raise` branch is unreachable.
- **Impact:** Confusing control flow.
- **Fix:** Remove the flag or make it a real CLI parameter.
- **Good first issue:** yes

### `NB-06` Batch 2 silently reduces the sample count
`bug` · `severity:medium` · `component:kaggle` · `status:confirmed`
- **Where:** `kaggle/batch2/kernel_batch2_trellis_spar3d.ipynb` (ingest cell)
- **Finding:** On missing objects it sets `TARGET_OBJECTS = [...]` and
  `NUM_SAMPLES = len(TARGET_OBJECTS)` and continues.
- **Impact:** A run can finish with fewer than 100 objects and still be treated as the
  benchmark.
- **Fix:** Fail the kernel or write a prominent `ingest_missing.json` and a non-zero exit.
- **Good first issue:** no

### `NB-07` Notebooks use IPython `display`
`maintainability` · `severity:low` · `component:notebooks` · `status:confirmed`
- **Where:** `kaggle/*/*.ipynb`, `notebooks/colab_demo.ipynb`
- **Finding:** `display(...)` is not defined outside IPython.
- **Impact:** `jupyter nbconvert --execute` works, but `python -c` extraction fails.
- **Fix:** Use `print` for headless execution or guard the import.
- **Good first issue:** yes

### `NB-08` `Tee` log file is never closed
`maintainability` · `severity:low` · `component:kaggle` · `status:confirmed`
- **Where:** `kaggle/*/*.ipynb` (first cell)
- **Finding:** `log_f = open(...)` with no close/flush on exit.
- **Impact:** Truncated last lines; ResourceWarning.
- **Fix:** Register `atexit` close or use a context manager.
- **Good first issue:** yes

### `NB-09` Ingest and aggregation code is duplicated across kernels
`maintainability` · `severity:medium` · `component:kaggle` · `status:confirmed`
- **Where:** `kaggle/batch1`, `kaggle/batch2`, `kaggle/triposr_only`, `kaggle/verify`
- **Finding:** The GSO download/ingest and results-aggregation cells are copy-pasted with
  small differences.
- **Impact:** Fixes must be applied four times; drift is likely (for example the batch-2
  silent-reduction path).
- **Fix:** Move shared cells into a small `kaggle/common.py` or a dataset utility.
- **Good first issue:** no

### `NB-10` No row-count/failure gate in the kernels
`testing` · `severity:medium` · `component:kaggle` · `status:confirmed`
- **Where:** `kaggle/*/*.ipynb` (final aggregation cell)
- **Finding:** The kernels print a mismatch warning but do not fail the run.
- **Impact:** A partial run can be downloaded as if complete.
- **Fix:** `assert` the expected row count per engine, or exit non-zero.
- **Good first issue:** yes

### `NB-11` `open(...).write_text` without encoding
`maintainability` · `severity:low` · `component:kaggle` · `status:confirmed`
- **Where:** `kaggle/*/*.ipynb`
- **Finding:** Several `Path.write_text` calls omit `encoding="utf-8"`.
- **Impact:** Locale-dependent failures on non-UTF-8 hosts.
- **Fix:** Pass `encoding="utf-8"`.
- **Good first issue:** yes

### `NB-12` Verify kernel's markdown describes the wrong engines
`documentation` · `severity:low` · `component:notebooks` · `status:confirmed`
- **Where:** `kaggle/verify/kernel_verify_render.ipynb` (markdown says "Evaluates SPAR3D and
  TRELLIS"; the code also includes TripoSR/LGM imports)
- **Finding:** The markdown is accurate for the run but the setup imports all four engines.
- **Impact:** Minor confusion about what is evaluated.
- **Fix:** Clarify in the markdown.
- **Good first issue:** yes

## 12. Tests, CI and automation (`CI`)

### `CI-01` No `tests/` directory
`testing` · `severity:high` · `component:ci` · `status:confirmed`
- **Where:** repository root
- **Finding:** There are no unit tests at all.
- **Impact:** Every correctness fix in this register is unverified.
- **Fix:** Add `tests/` with metric, export, API and engine-contract tests; run on PR.
- **Good first issue:** no

### `CI-02` No CI workflow
`testing` · `severity:high` · `component:ci` · `status:confirmed`
- **Where:** `.github/` (absent)
- **Finding:** No GitHub Actions.
- **Impact:** Broken `main` is possible.
- **Fix:** Add a minimal workflow (lint, `py_compile`, tests, smoke import).
- **Good first issue:** no

### `CI-03` No lint/format gate
`maintainability` · `severity:high` · `component:ci` · `status:confirmed`
- **Where:** repository root
- **Finding:** No `ruff`/`black` config or CI step.
- **Impact:** Style drift and noisy reviews.
- **Fix:** Add `ruff` + `black`, run on changed files.
- **Good first issue:** no

### `CI-04` No benchmark regression test
`testing` · `severity:high` · `component:ci` · `status:confirmed`
- **Where:** `src/evaluation/`
- **Finding:** No tiny synthetic mesh is run end-to-end through load -> predict (stub) ->
  align -> metrics -> export.
- **Impact:** Protocol changes can silently alter numbers.
- **Fix:** Add a 1-triangle/1-cube end-to-end test with a stub engine.
- **Good first issue:** no

### `CI-05` No metric unit tests
`testing` · `severity:high` · `component:ci` · `status:confirmed`
- **Where:** `src/evaluation/metrics.py`
- **Finding:** No known-value tests for CD, F-score, PSNR or LPIPS.
- **Impact:** A sign flip or threshold change can alter published numbers.
- **Fix:** Add analytic cases (identical clouds, translated cube, known radii).
- **Good first issue:** no

### `CI-06` No export round-trip tests
`testing` · `severity:medium` · `component:ci` · `status:confirmed`
- **Where:** `src/export/`
- **Finding:** Nothing checks that exported files reload.
- **Impact:** Silent data loss in deliverables.
- **Fix:** Add GLB/PLY round-trip tests.
- **Good first issue:** yes

### `CI-07` No API smoke test
`testing` · `severity:medium` · `component:ci` · `status:confirmed`
- **Where:** `src/api/server.py`
- **Finding:** No `TestClient` test.
- **Impact:** API regressions go unnoticed.
- **Fix:** Add `tests/test_api.py` with a mocked engine.
- **Good first issue:** yes

### `CI-08` No dependency audit / pin check
`testing` · `severity:medium` · `component:ci` · `status:confirmed`
- **Where:** `requirements.txt`, `pyproject.toml`
- **Finding:** No `pip-audit`, no lockfile check, no `pip check`.
- **Impact:** Vulnerable or inconsistent environments can be published.
- **Fix:** Add a scheduled dependency audit and a lockfile check.
- **Good first issue:** no

### `CI-09` No results-contract validation in CI
`testing` · `severity:medium` · `component:ci` · `status:confirmed`
- **Where:** `results/`
- **Finding:** Nothing checks the committed CSVs against the documented schema/counts.
- **Impact:** A partial or malformed CSV can be committed.
- **Fix:** Add `scripts/validate_results.py` and run it in CI.
- **Good first issue:** no

### `CI-10` No notebook execution smoke test
`testing` · `severity:low` · `component:ci` · `status:confirmed`
- **Where:** `kaggle/`, `notebooks/`
- **Finding:** Notebooks are never executed or parsed in CI.
- **Impact:** Syntax/API drift in the reproduction path.
- **Fix:** At minimum, parse notebooks and run a `--dry-run` import cell; full execution needs
  a GPU runner.
- **Good first issue:** no

---

## 13. Suggested filing order

If you are triaging this register, file in this order:

1. `SEC-07`, `SEC-01`, `EVAL-02`, `SEC-08`, `SEC-10`, `SEC-11` — critical/high impact.
2. `EVAL-03`, `EVAL-04`, `EVAL-09`, `EVAL-16`, `EVAL-34`, `EXP-01`, `EXP-06` — numbers/deliverables.
3. `EVAL-11`, `EVAL-23`, `DATA-05`, `DOC-02`, `DOC-04`, `TOOL-01`, `TOOL-02`, `TOOL-03` —
   reproducibility.
4. `PKG-01`..`PKG-15`, `CI-01`..`CI-05` — install/test health.
5. Remaining medium/low entries.

When filing, copy the entry verbatim (ID, labels, where, finding, impact, fix) and add a
minimal reproduction if you have one. Do not renumber IDs.