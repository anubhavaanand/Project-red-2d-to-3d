"""Tests for the Gradio output directory (SEC-27)."""

import ast
import os
import shutil
import stat
import tempfile
from pathlib import Path

APP_PY = Path(__file__).parent.parent / "app.py"


def _output_dir_source():
    """Pull just the OUTPUT_DIR block out of app.py (no gradio/torch import).

    Extracted by line markers rather than AST because the surrounding
    try/except belongs to a wider scope than the statements we need.
    """
    lines = APP_PY.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("OUTPUT_DIR ="))
    end = next(i for i in range(start, len(lines)) if lines[i].strip() == "pass" and lines[i].startswith("    "))
    return "\n".join(lines[start : end + 1])


class TestOutputDir:
    def test_output_dir_is_uid_scoped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
        ns = {"os": os, "tempfile": tempfile, "Path": Path}
        exec(_output_dir_source(), ns)
        out = ns["OUTPUT_DIR"]
        assert str(os.getuid()) in out.name
        assert out.parent == tmp_path

    def test_output_dir_is_owner_only(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
        ns = {"os": os, "tempfile": tempfile, "Path": Path}
        exec(_output_dir_source(), ns)
        mode = stat.S_IMODE(ns["OUTPUT_DIR"].stat().st_mode)
        assert mode == 0o700, f"expected 0o700, got {oct(mode)}"
        shutil.rmtree(ns["OUTPUT_DIR"])

    def test_existing_dir_is_reused_and_restricted(self, tmp_path, monkeypatch):
        """A pre-existing dir (possibly created with loose perms) is tightened."""
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
        ns = {"os": os, "tempfile": tempfile, "Path": Path}
        exec(_output_dir_source(), ns)
        out = ns["OUTPUT_DIR"]
        os.chmod(out, 0o755)  # simulate a loosely-created directory
        exec(_output_dir_source(), ns)  # re-run the block
        assert stat.S_IMODE(out.stat().st_mode) == 0o700
        shutil.rmtree(out)

    def test_app_py_is_importable_source(self):
        ast.parse(APP_PY.read_text())