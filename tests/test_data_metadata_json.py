"""Tests for JSON metadata in result rows (DATA-05)."""

import ast
import json
from pathlib import Path

import pytest

RUNNER = Path(__file__).parent.parent / "src" / "evaluation" / "benchmark_runner.py"


class TestSourceContract:
    def test_metadata_is_json_serialised(self):
        src = RUNNER.read_text()
        assert "json.dumps(recon.metadata, default=str)" in src

    def test_no_bare_metadata_assignment(self):
        src = RUNNER.read_text()
        assert 'row["metadata"] = recon.metadata\n' not in src

    def test_none_metadata_is_preserved(self):
        src = RUNNER.read_text()
        assert "if recon.metadata is not None" in src


class TestJsonRoundTrip:
    def test_single_quoted_repr_is_not_json(self):
        """The old format, for the record."""
        old = "{'engine': 'TripoSR', 'latency_s': 5.922}"
        with pytest.raises(json.JSONDecodeError):
            json.loads(old)

    def test_serialised_metadata_parses(self):
        payload = {"engine": "TripoSR", "latency_s": 5.922, "vertex_count": 74603}
        assert json.loads(json.dumps(payload)) == payload

    def test_nested_values_survive(self):
        payload = {"engine": "TRELLIS", "metadata": {"steps": 8}, "tags": ["a", "b"]}
        assert json.loads(json.dumps(payload, default=str)) == payload

    def test_non_serialisable_values_are_stringified(self):
        """default=str keeps exotic values (Path, numpy scalars) from crashing."""
        import datetime

        payload = {"when": datetime.datetime(2026, 1, 1)}
        out = json.loads(json.dumps(payload, default=str))
        assert isinstance(out["when"], str)


class TestCsvCompatibility:
    def test_metadata_column_still_exists(self):
        src = RUNNER.read_text()
        assert 'row["metadata"]' in src

    def test_results_readme_documents_the_format(self):
        readme = Path(__file__).parent.parent / "results" / "README.md"
        text = readme.read_text()
        assert "metadata" in text


class TestSourceStructure:
    def test_engine_parses(self):
        ast.parse(RUNNER.read_text())
