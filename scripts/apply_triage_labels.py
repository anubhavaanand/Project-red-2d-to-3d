#!/usr/bin/env python3
"""
Apply triage labels to GitHub issues based on the triage report.
Requires: gh CLI authenticated (gh auth login)

Usage:
    python scripts/apply_triage_labels.py [--dry-run]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/anubhavanand/Desktop/reasearch papers/image-to-3d-benchmark")
REPORT_PATH = REPO / "analysis" / "ISSUE_TRIAGE_REPORT.md"

# Triage results (hardcoded from the analysis)
TRIAGE_RESULTS = {
    # CI issues (all genuine)
    "CI-01": "genuine", "CI-02": "genuine", "CI-03": "genuine", "CI-04": "genuine",
    "CI-05": "genuine", "CI-06": "genuine", "CI-07": "genuine", "CI-08": "genuine",
    "CI-09": "genuine", "CI-10": "genuine",
    # Notebook issues
    "NB-01": "genuine", "NB-02": "genuine", "NB-03": "needs-repro", "NB-04": "genuine",
    "NB-05": "genuine", "NB-06": "genuine", "NB-07": "genuine", "NB-08": "genuine",
    "NB-09": "genuine", "NB-10": "genuine", "NB-11": "genuine", "NB-12": "genuine",
    # Documentation issues
    "DOC-01": "genuine", "DOC-02": "genuine", "DOC-03": "genuine", "DOC-04": "genuine",
    "DOC-05": "genuine", "DOC-06": "genuine", "DOC-07": "genuine", "DOC-08": "needs-repro",
    "DOC-09": "genuine", "DOC-10": "genuine", "DOC-11": "genuine", "DOC-12": "genuine",
    "DOC-13": "genuine", "DOC-14": "genuine", "DOC-15": "genuine",
    # Packaging issues
    "PKG-01": "genuine", "PKG-02": "genuine", "PKG-03": "genuine", "PKG-04": "genuine",
    "PKG-05": "genuine", "PKG-06": "genuine", "PKG-07": "genuine", "PKG-08": "needs-repro",
    "PKG-09": "genuine", "PKG-10": "genuine", "PKG-11": "genuine", "PKG-12": "genuine",
    "PKG-13": "genuine", "PKG-14": "genuine", "PKG-15": "genuine",
    # Tool/script issues (all genuine)
    **{f"TOOL-{i:02d}": "genuine" for i in range(1, 21)},
    # Export issues
    "EXP-01": "genuine", "EXP-02": "genuine", "EXP-03": "genuine", "EXP-04": "genuine",
    "EXP-05": "genuine", "EXP-06": "genuine", "EXP-07": "needs-repro", "EXP-08": "genuine",
    "EXP-09": "genuine", "EXP-10": "genuine", "EXP-11": "genuine", "EXP-12": "genuine",
    "EXP-13": "genuine", "EXP-14": "genuine",
    # Data issues (all genuine)
    **{f"DATA-{i:02d}": "genuine" for i in range(1, 14)},
    # Evaluation issues (all genuine)
    **{f"EVAL-{i:02d}": "genuine" for i in range(1, 41)},
    # Security issues
    "SEC-01": "genuine", "SEC-02": "genuine", "SEC-03": "genuine", "SEC-04": "genuine",
    "SEC-05": "genuine", "SEC-06": "genuine", "SEC-07": "genuine", "SEC-08": "genuine",
    "SEC-09": "needs-repro", "SEC-10": "genuine", "SEC-11": "genuine", "SEC-12": "genuine",
    "SEC-13": "genuine", "SEC-14": "needs-repro", "SEC-15": "genuine", "SEC-16": "genuine",
    "SEC-17": "genuine", "SEC-18": "genuine", "SEC-19": "genuine", "SEC-20": "genuine",
    "SEC-21": "genuine", "SEC-22": "genuine", "SEC-23": "genuine", "SEC-24": "genuine",
}


def fetch_issues():
    """Fetch all open issues with their numbers and titles."""
    try:
        result = subprocess.run(
            ["gh", "issue", "list", "--repo", "anubhavaanand/Project-red-2d-to-3d",
             "--state", "open", "--limit", "300", "--json", "number,title,labels"],
            capture_output=True, text=True, check=True
        )
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[ERROR] gh CLI not available or not authenticated: {e}")
        print("[INFO] Run: gh auth login")
        return []


def extract_issue_id(title: str) -> str:
    """Extract issue ID like 'CI-01' from title."""
    if ":" in title:
        return title.split(":")[0].strip()
    return ""


def apply_label(issue_number: int, label: str, dry_run: bool = True):
    """Apply a label to a GitHub issue."""
    if dry_run:
        print(f"  [DRY RUN] Would add '{label}' to issue #{issue_number}")
        return True

    try:
        subprocess.run(
            ["gh", "issue", "edit", str(issue_number),
             "--repo", "anubhavaanand/Project-red-2d-to-3d",
             "--add-label", label],
            capture_output=True, text=True, check=True
        )
        print(f"  [OK] Added '{label}' to issue #{issue_number}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] Failed to label issue #{issue_number}: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Apply triage labels to GitHub issues")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without applying")
    args = parser.parse_args()

    print("=== Applying Triage Labels to GitHub Issues ===")
    print()

    issues = fetch_issues()
    if not issues:
        print("[ERROR] Could not fetch issues. Make sure gh CLI is authenticated.")
        sys.exit(1)

    print(f"Found {len(issues)} open issues")
    print()

    labeled = 0
    skipped = 0
    errors = 0

    for issue in issues:
        number = issue["number"]
        title = issue["title"]
        issue_id = extract_issue_id(title)

        if not issue_id or issue_id not in TRIAGE_RESULTS:
            print(f"  [SKIP] #{number}: {title[:50]}... (no triage rule)")
            skipped += 1
            continue

        label = f"triage:{TRIAGE_RESULTS[issue_id]}"

        # Check if label already exists
        existing_labels = [l["name"] for l in issue.get("labels", [])]
        if label in existing_labels:
            print(f"  [SKIP] #{number}: Already has '{label}'")
            skipped += 1
            continue

        if apply_label(number, label, dry_run=args.dry_run):
            labeled += 1
        else:
            errors += 1

    print()
    print(f"=== SUMMARY ===")
    print(f"  Labeled: {labeled}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")

    if args.dry_run:
        print()
        print("[INFO] Run without --dry-run to apply labels")


if __name__ == "__main__":
    main()