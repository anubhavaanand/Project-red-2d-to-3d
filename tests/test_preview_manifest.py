"""Tests for preview-manifest labelling (DATA-03)."""

import ast
import json
from pathlib import Path

SELECT = Path(__file__).parent.parent / "gso1000" / "select_1000.py"
PREVIEW = Path(__file__).parent.parent / "gso1000" / "manifest_1000.preview.json"
README = Path(__file__).parent.parent / "gso1000" / "README.md"


class TestPreviewManifest:
    def test_is_marked_preview(self):
        d = json.loads(PREVIEW.read_text())
        assert d["mode"] == "preview"

    def test_has_a_warning(self):
        """DATA-03: the dry-run caveat must travel with the file."""
        d = json.loads(PREVIEW.read_text())
        assert d.get("warnings"), "preview manifest has no warnings"

    def test_warning_mentions_validation(self):
        d = json.loads(PREVIEW.read_text())
        text = " ".join(d["warnings"]).lower()
        assert "valid" in text
        assert "preview" in text

    def test_has_no_validation_report_key(self):
        """A preview must not claim to have been validated."""
        d = json.loads(PREVIEW.read_text())
        assert "validation_report" not in d

    def test_validated_ok_equals_pool(self):
        """The exact confusion the issue describes: it looks like a result."""
        d = json.loads(PREVIEW.read_text())
        assert d["counts"]["validated_ok"] == d["counts"]["pool"]

    def test_still_has_1000_objects(self):
        d = json.loads(PREVIEW.read_text())
        assert len(d["objects"]) == 1000

    def test_valid_json(self):
        json.loads(PREVIEW.read_text())


class TestGenerator:
    def test_emits_warnings_for_preview(self):
        src = SELECT.read_text()
        assert '"warnings": (' in src
        assert "PREVIEW: no validation report was supplied" in src

    def test_warning_only_for_preview_mode(self):
        src = SELECT.read_text()
        block = src[src.index('"warnings": (') : src.index("**extras,")]
        assert 'if mode == "preview"' in block
        assert "[]" in block  # empty list otherwise

    def test_readme_warns(self):
        text = README.read_text()
        assert "Do not consume a preview manifest" in text
        assert "preview" in text.lower()


class TestSourceStructure:
    def test_selector_parses(self):
        ast.parse(SELECT.read_text())
