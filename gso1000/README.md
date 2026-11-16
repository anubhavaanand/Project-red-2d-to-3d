# gso1000 — object selection

`select_1000.py` picks the 1,000 GSO objects used by the benchmark and writes
a manifest.

## Modes

| Mode | Trigger | Meaning |
|---|---|---|
| `final` | `--validation-report <report.json>` | Objects were checked against a validation report; `counts.validated_ok` is real. |
| `preview` | *(no report)* | **Dry run.** Nothing was validated — every object in the pool is assumed valid, so `counts.validated_ok == counts.pool`. |

## ⚠️ Do not consume a preview manifest

`manifest_1000.preview.json` is a **dry-run artefact**. Its
`counts.validated_ok` (1046) equals the whole pool because no validation
happened — it is *not* evidence that 1,046 objects are usable.

Consuming it directly will fail on the first object that does not actually
validate. Before using a manifest, either:

1. re-run with `--validation-report` to produce `manifest_1000.json`, or
2. check that the manifest has `\"mode\": \"final\"` and a `validation_report` key.

Preview manifests now carry a `warnings` array saying the same thing, so the
caveat travels with the file rather than living only in this README.

## Files

| File | Purpose |
|---|---|
| `select_1000.py` | Selector / manifest generator |
| `handpicked.txt` | The 100 curated objects (incl. `CANONICAL_10`) |
| `manifest_1000.preview.json` | Dry-run manifest — **not for consumption** |
| `manifest_1000.json` | Validated manifest (produced with `--validation-report`) |
| `manifest_1000.preview.md` | Human-readable preview |

## Object-ID parsing

`handpicked.txt` is parsed by `src.data.object_ids.load_object_ids` (DATA-01):
blank lines and `#` comments (indented included) are skipped, order is
preserved, duplicates removed.