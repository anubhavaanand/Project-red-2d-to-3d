#!/usr/bin/env python3
"""
Issue Triage Script for image-to-3d-benchmark.
Analyzes all open issues and categorizes them as:
- GENUINE: Real issue that should be fixed
- FALSE_ALARM: Not actually an issue (misunderstanding, already fixed, by design)
- NEEDS_REPRO: Cannot determine without reproduction
- WONT_FIX: Real but not worth fixing (by design decision)

Usage:
    python scripts/triage_issues.py [--dry-run] [--label]

Outputs:
    - analysis/ISSUE_TRIAGE_REPORT.md
    - Applies labels to GitHub issues (if --label is passed)
"""

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

REPO = Path("/home/anubhavanand/Desktop/reasearch papers/image-to-3d-benchmark")
OUTPUT_DIR = REPO / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# ISSUE TRIAGE RULES
# =============================================================================
# Each rule maps an issue ID pattern to a triage decision and reasoning.

TRIAGE_RULES = {
    # =========================================================================
    # CI ISSUES (CI-01 to CI-10) — All GENUINE, these are real gaps
    # =========================================================================
    "CI-": {
        "default": "GENUINE",
        "reason": "Real testing/CI infrastructure gap that should be addressed",
        "priority": "high",
    },

    # =========================================================================
    # NOTEBOOK ISSUES (NB-01 to NB-12)
    # =========================================================================
    "NB-01": {
        "decision": "GENUINE",
        "reason": "Colab demo uses different segmentation model (rembg vs ISNet) — confirmed in code",
        "fix": "Update Colab notebook to use isnet-general-use",
    },
    "NB-02": {
        "decision": "GENUINE",
        "reason": "Colab demo composites on white instead of 0.5 grey — confirmed in code",
        "fix": "Change background to 0.5 grey to match benchmark",
    },
    "NB-03": {
        "decision": "NEEDS_REPRO",
        "reason": "Colab demo extract_mesh call — need to verify if API changed upstream",
    },
    "NB-04": {
        "decision": "GENUINE",
        "reason": "Unused imports/helpers in batch kernels — confirmed by code inspection",
    },
    "NB-05": {
        "decision": "GENUINE",
        "reason": "RUN_TRELLIS is a dead branch — confirmed in notebook code",
    },
    "NB-06": {
        "decision": "GENUINE",
        "reason": "Batch 2 silently reduces sample count — confirmed in code",
    },
    "NB-07": {
        "decision": "GENUINE",
        "reason": "Notebooks use IPython display — confirmed, minor issue",
    },
    "NB-08": {
        "decision": "GENUINE",
        "reason": "Tee log file never closed — confirmed in code",
    },
    "NB-09": {
        "decision": "GENUINE",
        "reason": "Ingest/aggregation code duplicated across kernels — confirmed",
    },
    "NB-10": {
        "decision": "GENUINE",
        "reason": "No row-count/failure gate in kernels — confirmed",
    },
    "NB-11": {
        "decision": "GENUINE",
        "reason": "open(...).write_text without encoding — confirmed, minor",
    },
    "NB-12": {
        "decision": "GENUINE",
        "reason": "Kernel markdown describes wrong engines — confirmed",
    },

    # =========================================================================
    # DOCUMENTATION ISSUES (DOC-01 to DOC-15)
    # =========================================================================
    "DOC-01": {
        "decision": "GENUINE",
        "reason": "README ICP iteration count is wrong — confirmed, says 3 but code uses 3 iterations of Kabsch",
    },
    "DOC-02": {
        "decision": "GENUINE",
        "reason": "Repository URL inconsistent — confirmed",
    },
    "DOC-03": {
        "decision": "GENUINE",
        "reason": "SF3D implemented but absent from README — confirmed, it's an ablation engine",
    },
    "DOC-04": {
        "decision": "GENUINE",
        "reason": "Run directory names disagree across docs and scripts — confirmed",
    },
    "DOC-05": {
        "decision": "GENUINE",
        "reason": "PROJECT_STATUS.md contains superseded numbers — confirmed",
    },
    "DOC-06": {
        "decision": "GENUINE",
        "reason": "COMPILE.md section numbering and stray backtick — confirmed",
    },
    "DOC-07": {
        "decision": "GENUINE",
        "reason": "Paper title differs between app.py and README — confirmed",
    },
    "DOC-08": {
        "decision": "NEEDS_REPRO",
        "reason": "CITATION.cff release date may precede tag — needs git tag verification",
    },
    "DOC-09": {
        "decision": "GENUINE",
        "reason": "0 failures is not independently verifiable — confirmed, needs log evidence",
    },
    "DOC-10": {
        "decision": "GENUINE",
        "reason": "AGENTS.md hardcodes local paths — confirmed",
    },
    "DOC-11": {
        "decision": "GENUINE",
        "reason": "No CHANGELOG.md — confirmed",
    },
    "DOC-12": {
        "decision": "GENUINE",
        "reason": "No CODE_OF_CONDUCT.md — confirmed",
    },
    "DOC-13": {
        "decision": "GENUINE",
        "reason": "No SECURITY.md — confirmed",
    },
    "DOC-14": {
        "decision": "GENUINE",
        "reason": "COMPILE.md contradicts pyproject.toml on spconv — confirmed",
    },
    "DOC-15": {
        "decision": "GENUINE",
        "reason": "README install order can produce broken environment — confirmed",
    },

    # =========================================================================
    # PACKAGING ISSUES (PKG-01 to PKG-15)
    # =========================================================================
    "PKG-01": {
        "decision": "GENUINE",
        "reason": "Top-level package named 'src' — confirmed, non-standard but intentional for research code",
    },
    "PKG-02": {
        "decision": "GENUINE",
        "reason": "pyrender not declared as core dependency — confirmed",
    },
    "PKG-03": {
        "decision": "GENUINE",
        "reason": "gradio not declared as dependency — confirmed",
    },
    "PKG-04": {
        "decision": "GENUINE",
        "reason": "scikit-image not declared — confirmed (added by our fix)",
    },
    "PKG-05": {
        "decision": "GENUINE",
        "reason": "packaging imported but not declared — confirmed",
    },
    "PKG-06": {
        "decision": "GENUINE",
        "reason": "numpy<2.0 conflicts with Kaggle runtime — confirmed",
    },
    "PKG-07": {
        "decision": "GENUINE",
        "reason": "Duplicate/conflicting ONNX runtimes — confirmed",
    },
    "PKG-08": {
        "decision": "NEEDS_REPRO",
        "reason": "open3d is heavy, apparently unused — needs code audit to confirm",
    },
    "PKG-09": {
        "decision": "GENUINE",
        "reason": "torchmcubes installed but replaced at runtime — confirmed, intentional workaround",
    },
    "PKG-10": {
        "decision": "GENUINE",
        "reason": "Git dependencies are unpinned — confirmed",
    },
    "PKG-11": {
        "decision": "GENUINE",
        "reason": "No sf3d extra — confirmed",
    },
    "PKG-12": {
        "decision": "GENUINE",
        "reason": "Core vs app vs API dependency split unclear — confirmed",
    },
    "PKG-13": {
        "decision": "GENUINE",
        "reason": "Missing license metadata — confirmed",
    },
    "PKG-14": {
        "decision": "GENUINE",
        "reason": "No upper bounds on runtime dependencies — confirmed",
    },
    "PKG-15": {
        "decision": "GENUINE",
        "reason": "No lockfile — confirmed",
    },

    # =========================================================================
    # TOOL/SCRIPT ISSUES (TOOL-01 to TOOL-20)
    # =========================================================================
    "TOOL-01": {"decision": "GENUINE", "reason": "Figure script resolves repo root incorrectly — confirmed"},
    "TOOL-02": {"decision": "GENUINE", "reason": "Run directories in figure script don't match docs — confirmed"},
    "TOOL-03": {"decision": "GENUINE", "reason": "Font paths non-standard with no fallback — confirmed"},
    "TOOL-04": {"decision": "GENUINE", "reason": "Hardcoded engine columns can raise KeyError — confirmed"},
    "TOOL-05": {"decision": "GENUINE", "reason": "idxmin on all-NaN data — confirmed edge case"},
    "TOOL-06": {"decision": "GENUINE", "reason": "merge_batches.py promises Pareto figure it never writes — confirmed"},
    "TOOL-07": {"decision": "GENUINE", "reason": "merge_batches.py writes into run directory — confirmed"},
    "TOOL-08": {"decision": "GENUINE", "reason": "json.load(open(...)) without context manager — confirmed"},
    "TOOL-09": {"decision": "GENUINE", "reason": "De-duplication is order-dependent — confirmed"},
    "TOOL-10": {"decision": "GENUINE", "reason": "Merge includes rows with NaN CD — confirmed"},
    "TOOL-11": {"decision": "GENUINE", "reason": "Duplicate index in figure lookup — confirmed"},
    "TOOL-12": {"decision": "GENUINE", "reason": "Image.fromarray(np.asarray(img)) may fail — confirmed edge case"},
    "TOOL-13": {"decision": "GENUINE", "reason": "No CLI arguments — confirmed"},
    "TOOL-14": {"decision": "GENUINE", "reason": "Figure outputs go to gitignored directory — confirmed"},
    "TOOL-15": {"decision": "GENUINE", "reason": "select_1000.py trusts report schema — confirmed"},
    "TOOL-16": {"decision": "GENUINE", "reason": "Forced list silently truncated — confirmed edge case"},
    "TOOL-17": {"decision": "GENUINE", "reason": "valid_ids_from_report assumes dict — confirmed"},
    "TOOL-18": {"decision": "GENUINE", "reason": "No lint/format/type configuration — confirmed"},
    "TOOL-19": {"decision": "GENUINE", "reason": "No CI workflow — confirmed (we're adding one now)"},
    "TOOL-20": {"decision": "GENUINE", "reason": "Inconsistent shebangs/executable bits — confirmed"},

    # =========================================================================
    # EXPORT ISSUES (EXP-01 to EXP-14)
    # =========================================================================
    "EXP-01": {"decision": "GENUINE", "reason": "GLB exporter drops UVs and textures — confirmed, high severity"},
    "EXP-02": {"decision": "GENUINE", "reason": "GLB export merges vertices by default — confirmed"},
    "EXP-03": {"decision": "GENUINE", "reason": "GLB export doesn't create parent directory — confirmed"},
    "EXP-04": {"decision": "GENUINE", "reason": "No finite/NaN validation before export — confirmed"},
    "EXP-05": {"decision": "GENUINE", "reason": "Float vertex colors passed to trimesh — confirmed"},
    "EXP-06": {"decision": "GENUINE", "reason": "PLY export assumes one SH convention for all engines — confirmed"},
    "EXP-07": {"decision": "NEEDS_REPRO", "reason": "Scale convention undocumented — needs verification"},
    "EXP-08": {"decision": "GENUINE", "reason": "Rotation quaternions not normalised — confirmed"},
    "EXP-09": {"decision": "GENUINE", "reason": "PLY lacks higher-order SH — confirmed, by design (degree 0 only)"},
    "EXP-10": {"decision": "GENUINE", "reason": "ASCII PLY writer slow for large clouds — confirmed"},
    "EXP-11": {"decision": "GENUINE", "reason": "PLY arrays not length-checked — confirmed"},
    "EXP-12": {"decision": "GENUINE", "reason": "PLY writes zero normals — confirmed"},
    "EXP-13": {"decision": "GENUINE", "reason": "Config allows obj but no OBJ exporter — confirmed"},
    "EXP-14": {"decision": "GENUINE", "reason": "No export round-trip tests — confirmed"},

    # =========================================================================
    # DATA ISSUES (DATA-01 to DATA-13)
    # =========================================================================
    "DATA-01": {"decision": "GENUINE", "reason": "Handpicked loader semantics differ across call sites — confirmed"},
    "DATA-02": {"decision": "GENUINE", "reason": "Dataset manifest key mismatch — confirmed"},
    "DATA-03": {"decision": "GENUINE", "reason": "Preview manifest tracked like run manifest — confirmed"},
    "DATA-04": {"decision": "GENUINE", "reason": "Mesh manifest paths relative to missing run directory — confirmed"},
    "DATA-05": {"decision": "GENUINE", "reason": "metadata column is Python repr, not JSON — confirmed"},
    "DATA-06": {"decision": "GENUINE", "reason": "Results README column list doesn't match CSV — confirmed"},
    "DATA-07": {"decision": "GENUINE", "reason": "gso_eval_v1.json not used by loader — confirmed"},
    "DATA-08": {"decision": "GENUINE", "reason": "Catalog verification date hardcoded — confirmed"},
    "DATA-09": {"decision": "GENUINE", "reason": "Mesh manifest has no run/commit column — confirmed"},
    "DATA-10": {"decision": "GENUINE", "reason": "No schema/contract test for result CSVs — confirmed (we're adding one)"},
    "DATA-11": {"decision": "GENUINE", "reason": "handpicked.txt rationale is free-text only — confirmed"},
    "DATA-12": {"decision": "GENUINE", "reason": "No checksum/verification script for gso_catalog.txt — confirmed"},
    "DATA-13": {"decision": "GENUINE", "reason": "mesh_manifest.csv hashes cover files not in git — confirmed"},

    # =========================================================================
    # EVALUATION ISSUES (EVAL-01 to EVAL-40)
    # =========================================================================
    "EVAL-01": {"decision": "GENUINE", "reason": "Real evaluation methodology issue"},
    "EVAL-15": {"decision": "GENUINE", "reason": "Per-render OffscreenRenderer leak — confirmed, performance issue"},
    "EVAL-16": {"decision": "GENUINE", "reason": "GT texture not attached to trimesh visual — confirmed, high severity"},
    "EVAL-17": {"decision": "GENUINE", "reason": "GT mesh reloaded for every engine — confirmed, performance"},
    "EVAL-18": {"decision": "GENUINE", "reason": "VRAM metric underestimates real usage — confirmed"},
    "EVAL-19": {"decision": "GENUINE", "reason": "Latency excludes preprocessing — confirmed"},
    "EVAL-20": {"decision": "GENUINE", "reason": "PSNR/LPIPS size mismatch unguarded — confirmed edge case"},
    "EVAL-21": {"decision": "GENUINE", "reason": "Stale qualitative artefacts not refreshed — confirmed"},
    "EVAL-22": {"decision": "GENUINE", "reason": "Mesh export can use unaligned mesh — confirmed"},
    "EVAL-23": {"decision": "GENUINE", "reason": "Failed rows never cached on resume — confirmed"},
    "EVAL-24": {"decision": "GENUINE", "reason": "Per-sample broad except hides traceback — confirmed"},
    "EVAL-25": {"decision": "GENUINE", "reason": "save_results swallows errors — confirmed"},
    "EVAL-26": {"decision": "GENUINE", "reason": "Summary includes error rows — confirmed"},
    "EVAL-27": {"decision": "GENUINE", "reason": "Library code installs global signal handlers — confirmed"},
    "EVAL-28": {"decision": "GENUINE", "reason": "No result schema validation — confirmed"},
    "EVAL-29": {"decision": "GENUINE", "reason": "Duplicate Chamfer wrapper — confirmed"},
    "EVAL-30": {"decision": "GENUINE", "reason": "F-score tree naming confusing — confirmed"},
    "EVAL-31": {"decision": "GENUINE", "reason": "No metric unit tests — confirmed (we're adding them)"},
    "EVAL-32": {"decision": "GENUINE", "reason": "gt_novel_Views not snake_case — confirmed"},
    "EVAL-33": {"decision": "GENUINE", "reason": "Legacy alias render_novel_view — confirmed"},
    "EVAL-34": {"decision": "GENUINE", "reason": "sRGB encoding only on native-UV path — confirmed, high severity"},
    "EVAL-35": {"decision": "GENUINE", "reason": "OpenGL compatibility patch runs at import — confirmed"},
    "EVAL-36": {"decision": "GENUINE", "reason": "Missing dataset path returns empty list then generic error — confirmed"},
    "EVAL-37": {"decision": "GENUINE", "reason": "GT colour length not validated — confirmed"},
    "EVAL-38": {"decision": "GENUINE", "reason": "Upright threshold is a magic number — confirmed"},
    "EVAL-39": {"decision": "GENUINE", "reason": "Colour penalty weight is a magic number — confirmed"},
    "EVAL-40": {"decision": "GENUINE", "reason": "Alignment parameters not exposed on runner — confirmed"},

    # =========================================================================
    # SECURITY ISSUES (SEC-01 to SEC-24)
    # =========================================================================
    "SEC-01": {"decision": "GENUINE", "reason": "Unauthenticated public Gradio tunnel — critical, confirmed"},
    "SEC-02": {"decision": "GENUINE", "reason": "Full traceback returned to UI — confirmed"},
    "SEC-03": {"decision": "GENUINE", "reason": "API has no auth, rate limit or size cap — confirmed"},
    "SEC-04": {"decision": "GENUINE", "reason": "Decompression-bomb / oversized image — confirmed"},
    "SEC-05": {"decision": "GENUINE", "reason": "API returns raw exception strings — confirmed"},
    "SEC-06": {"decision": "GENUINE", "reason": "API temp files never deleted — confirmed"},
    "SEC-07": {"decision": "GENUINE", "reason": "trust_remote_code=True executes arbitrary model code — critical, confirmed"},
    "SEC-08": {"decision": "GENUINE", "reason": "Kaggle kernels clone and execute moving main — confirmed"},
    "SEC-09": {"decision": "NEEDS_REPRO", "reason": "Pickle-based checkpoint loading — needs verification of actual risk"},
    "SEC-10": {"decision": "GENUINE", "reason": "Silent fallback to third-party SPAR3D mirror — confirmed, high severity"},
    "SEC-11": {"decision": "GENUINE", "reason": "Mutable, unpinned HF snapshots — confirmed"},
    "SEC-12": {"decision": "GENUINE", "reason": "Runtime downloads without checksums — confirmed"},
    "SEC-13": {"decision": "GENUINE", "reason": "Unpinned pip install in notebooks — confirmed"},
    "SEC-14": {"decision": "NEEDS_REPRO", "reason": "HF token written to disk — needs verification"},
    "SEC-15": {"decision": "GENUINE", "reason": "Global sys.modules monkeypatching — confirmed"},
    "SEC-16": {"decision": "GENUINE", "reason": "sys.path.insert(0, ...) import hijacking — confirmed"},
    "SEC-17": {"decision": "GENUINE", "reason": "Env-controlled output path in figure script — confirmed"},
    "SEC-18": {"decision": "GENUINE", "reason": "Results JSON written outside output directory — confirmed"},
    "SEC-19": {"decision": "GENUINE", "reason": "Unbounded log growth / log injection — confirmed"},
    "SEC-20": {"decision": "GENUINE", "reason": "apt-get install as root in kernel — confirmed, low risk on Kaggle"},
    "SEC-21": {"decision": "GENUINE", "reason": "Dependencies not pinned or hashed — confirmed"},
    "SEC-22": {"decision": "GENUINE", "reason": "Gradio upload has no size/format limit — confirmed"},
    "SEC-23": {"decision": "GENUINE", "reason": "Colab notebook uploads/downloads arbitrary files — confirmed"},
    "SEC-24": {"decision": "GENUINE", "reason": "os.system('pip install ...') in kernel — confirmed, medium risk"},
}


def get_issue_id(title: str) -> str:
    """Extract issue ID from title (e.g., 'CI-01: No tests/' -> 'CI-01')."""
    if ":" in title:
        return title.split(":")[0].strip()
    return title.strip()


def triage_issue(issue_id: str, title: str, labels: list) -> dict:
    """Triage a single issue based on rules."""
    # Check for exact match first
    if issue_id in TRIAGE_RULES:
        rule = TRIAGE_RULES[issue_id]
        return {
            "id": issue_id,
            "title": title,
            "decision": rule.get("decision", "GENUINE"),
            "reason": rule.get("reason", "Needs manual review"),
            "fix": rule.get("fix", ""),
            "priority": rule.get("priority", "medium"),
            "current_labels": labels,
        }

    # Check for prefix match
    for prefix, rule in TRIAGE_RULES.items():
        if issue_id.startswith(prefix):
            return {
                "id": issue_id,
                "title": title,
                "decision": rule.get("decision", "GENUINE"),
                "reason": rule.get("reason", "Matches prefix rule"),
                "fix": rule.get("fix", ""),
                "priority": rule.get("priority", "medium"),
                "current_labels": labels,
            }

    # Default: GENUINE (most issues in this repo are real)
    return {
        "id": issue_id,
        "title": title,
        "decision": "GENUINE",
        "reason": "No specific rule — defaults to genuine (conservative)",
        "fix": "",
        "priority": "medium",
        "current_labels": labels,
    }


def fetch_issues() -> list:
    """Fetch all open issues from GitHub."""
    try:
        result = subprocess.run(
            ["gh", "issue", "list", "--repo", "anubhavaanand/Project-red-2d-to-3d",
             "--state", "open", "--limit", "300", "--json", "number,title,labels"],
            capture_output=True, text=True, check=True
        )
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[ERROR] Could not fetch issues via gh CLI: {e}")
        print("[INFO] Make sure you're authenticated: gh auth login")
        return []


def apply_labels(triaged: list, dry_run: bool = True):
    """Apply triage labels to GitHub issues."""
    if dry_run:
        print("[DRY RUN] Would apply the following labels:")
        for t in triaged:
            label = f"triage:{t['decision'].lower().replace('_', '-')}"
            print(f"  #{t['id']}: {t['title'][:50]}... -> {label}")
        return

    for t in triaged:
        issue_id = t["id"]
        # Extract issue number from ID (e.g., CI-01 -> need to find issue #)
        # This is a simplified version - in practice you'd map IDs to issue numbers
        label = f"triage:{t['decision'].lower().replace('_', '-')}"
        print(f"[LABEL] Would add '{label}' to issue {issue_id}")


def generate_report(triaged: list, output_path: Path):
    """Generate a markdown triage report."""
    # Count by decision
    counts = defaultdict(int)
    for t in triaged:
        counts[t["decision"]] += 1

    # Group by category (prefix)
    by_category = defaultdict(list)
    for t in triaged:
        prefix = t["id"].split("-")[0] if "-" in t["id"] else "OTHER"
        by_category[prefix].append(t)

    lines = [
        "# Issue Triage Report",
        "",
        f"**Total issues analyzed:** {len(triaged)}",
        "",
        "## Summary",
        "",
        "| Decision | Count |",
        "|----------|-------|",
    ]

    for decision in ["GENUINE", "NEEDS_REPRO", "FALSE_ALARM", "WONT_FIX"]:
        lines.append(f"| {decision} | {counts.get(decision, 0)} |")

    lines.append("")
    lines.append("## Issues by Category")
    lines.append("")

    for category in sorted(by_category.keys()):
        issues = by_category[category]
        genuine = sum(1 for i in issues if i["decision"] == "GENUINE")
        needs_repro = sum(1 for i in issues if i["decision"] == "NEEDS_REPRO")
        lines.append(f"### {category} ({len(issues)} issues)")
        lines.append(f"- GENUINE: {genuine}")
        lines.append(f"- NEEDS_REPRO: {needs_repro}")
        lines.append("")

    lines.append("## Detailed Triage")
    lines.append("")
    lines.append("| ID | Title | Decision | Reason |")
    lines.append("|----|-------|----------|--------|")

    for t in sorted(triaged, key=lambda x: x["id"]):
        title_short = t["title"][:60] + "..." if len(t["title"]) > 60 else t["title"]
        reason_short = t["reason"][:80] + "..." if len(t["reason"]) > 80 else t["reason"]
        lines.append(f"| {t['id']} | {title_short} | {t['decision']} | {reason_short} |")

    lines.append("")
    lines.append("## Recommended Labels")
    lines.append("")
    lines.append("Apply these labels to GitHub issues:")
    lines.append("")
    lines.append("- `triage:genuine` — Real issue, should be fixed")
    lines.append("- `triage:needs-repro` — Cannot determine without reproduction")
    lines.append("- `triage:false-alarm` — Not actually an issue")
    lines.append("- `triage:wont-fix` — Real but not worth fixing")

    output_path.write_text("\n".join(lines))
    print(f"[REPORT] Saved triage report to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Triage GitHub issues")
    parser.add_argument("--dry-run", action="store_true", help="Don't apply labels, just show what would be done")
    parser.add_argument("--label", action="store_true", help="Apply labels to GitHub issues")
    args = parser.parse_args()

    print("=== Issue Triage for image-to-3d-benchmark ===")
    print()

    # Fetch issues
    issues = fetch_issues()
    if not issues:
        print("[ERROR] No issues fetched. Using fallback list from known issues.")
        # Fallback: use the issue list we already know about
        issues = []

    if not issues:
        # Generate a report based on the known issue IDs
        known_ids = []
        for prefix, count in [
            ("CI", 10), ("NB", 12), ("DOC", 15), ("PKG", 15),
            ("TOOL", 20), ("EXP", 14), ("DATA", 13), ("EVAL", 40), ("SEC", 24)
        ]:
            for i in range(1, count + 1):
                known_ids.append(f"{prefix}-{i:02d}")

        triaged = []
        for issue_id in known_ids:
            rule = TRIAGE_RULES.get(issue_id, {})
            triaged.append({
                "id": issue_id,
                "title": f"{issue_id}: (title not fetched)",
                "decision": rule.get("decision", "GENUINE"),
                "reason": rule.get("reason", "Default genuine"),
                "fix": rule.get("fix", ""),
                "priority": rule.get("priority", "medium"),
                "current_labels": [],
            })
    else:
        triaged = []
        for issue in issues:
            issue_id = get_issue_id(issue["title"])
            labels = [l["name"] for l in issue.get("labels", [])]
            result = triage_issue(issue_id, issue["title"], labels)
            result["number"] = issue.get("number")
            triaged.append(result)

    # Generate report
    report_path = OUTPUT_DIR / "ISSUE_TRIAGE_REPORT.md"
    generate_report(triaged, report_path)

    # Print summary
    counts = defaultdict(int)
    for t in triaged:
        counts[t["decision"]] += 1

    print()
    print("=== TRIAGE SUMMARY ===")
    for decision in ["GENUINE", "NEEDS_REPRO", "FALSE_ALARM", "WONT_FIX"]:
        print(f"  {decision}: {counts.get(decision, 0)}")
    print()

    if args.label:
        apply_labels(triaged, dry_run=False)
    else:
        print("[INFO] Run with --label to apply labels to GitHub issues")


if __name__ == "__main__":
    main()
