"""Tests for the mesh manifest layout (DATA-04)."""

import csv
from pathlib import Path

import pytest

MANIFEST = Path(__file__).parent.parent / "results" / "mesh_manifest.csv"
README = Path(__file__).parent.parent / "results" / "README.md"

EXPECTED_ENGINES = {"triposr", "lgm", "trellis", "spar3d"}


@pytest.fixture(scope="module")
def rows():
    with MANIFEST.open(newline="") as f:
        return list(csv.DictReader(f))


class TestColumns:
    def test_has_run_root(self):
        """DATA-04: the manifest must state which run root its paths belong to."""
        with MANIFEST.open(newline="") as f:
            fields = csv.DictReader(f).fieldnames
        assert "run_root" in fields

    def test_required_columns_present(self):
        with MANIFEST.open(newline="") as f:
            fields = csv.DictReader(f).fieldnames
        for col in (
            "engine",
            "sample_id",
            "run_root",
            "path",
            "file_size_bytes",
            "sha256",
            "vertex_count",
            "face_count",
        ):
            assert col in fields, col


class TestContent:
    def test_400_rows(self, rows):
        assert len(rows) == 400

    def test_run_root_matches_path_prefix(self, rows):
        """run_root must be derivable from path, not arbitrary."""
        for r in rows:
            assert r["path"].startswith(r["run_root"] + "/"), r["path"]

    def test_two_run_roots(self, rows):
        roots = {r["run_root"] for r in rows}
        assert roots == {"batch1_100obj_v35", "batch2_100obj_v35"}

    def test_four_engines(self, rows):
        assert {r["engine"] for r in rows} == EXPECTED_ENGINES

    def test_paths_are_relative(self, rows):
        """Absolute paths would be machine-specific and useless in git."""
        for r in rows:
            assert not r["path"].startswith("/"), r["path"]

    def test_hashes_are_sha256(self, rows):
        for r in rows:
            assert len(r["sha256"]) == 64, r["sha256"]


class TestDocs:
    def test_readme_documents_the_missing_root(self):
        text = README.read_text()
        assert "Mesh manifest" in text
        assert "not in this repository" in text

    def test_readme_shows_how_to_resolve(self):
        text = README.read_text()
        assert "run_root = Path(" in text
        assert 'row["path"]' in text

    def test_readme_mentions_run_root_column(self):
        assert "`run_root` column" in README.read_text()


class TestConsistency:
    def test_csv_matches_the_csv_contract_validator(self):
        """The results contract validator must still accept the CSVs."""
        import subprocess
        import sys

        r = subprocess.run(
            [sys.executable, "scripts/validate_results.py"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert r.returncode == 0, r.stdout + r.stderr
