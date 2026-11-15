"""Tests for the API's output-directory configuration (API-05)."""

import ast
import os
from pathlib import Path

SERVER_PY = Path(__file__).parent.parent / "src" / "api" / "server.py"


def _resolver_source():
    lines = SERVER_PY.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("def _resolve_output_dir"))
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("OUTPUT_DIR ="))
    return "\n".join(lines[start:end])


def _resolve():
    """Execute the real resolver with a stubbed AppConfig import."""
    src = _resolver_source()
    # replace the package-relative import with a direct one
    src = src.replace("from ..config import AppConfig", "from src.config import AppConfig")
    ns = {"os": os}
    exec(compile(src, str(SERVER_PY), "exec"), ns)
    return ns["_resolve_output_dir"]


class TestOutputDirResolution:
    def test_defaults_to_export_config(self, monkeypatch):
        monkeypatch.delenv("IMAGE_TO_3D_OUTPUT_DIR", raising=False)
        from src.config import AppConfig

        assert _resolve()() == AppConfig().export.output_dir

    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("IMAGE_TO_3D_OUTPUT_DIR", "/scratch/renders")
        assert _resolve()() == "/scratch/renders"

    def test_empty_env_falls_back_to_config(self, monkeypatch):
        """An empty variable must not produce an empty path."""
        monkeypatch.setenv("IMAGE_TO_3D_OUTPUT_DIR", "")
        from src.config import AppConfig

        assert _resolve()() == AppConfig().export.output_dir

    def test_output_dir_is_not_hardcoded(self):
        """API-05 regression: the literal './outputs' must not be assigned."""
        tree = ast.parse(SERVER_PY.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if getattr(t, "id", "") == "OUTPUT_DIR":
                        assert not (
                            isinstance(node.value, ast.Constant) and node.value.value == "./outputs"
                        ), "OUTPUT_DIR is still hardcoded"

    def test_health_exposes_output_dir(self):
        src = SERVER_PY.read_text()
        assert '"output_dir": OUTPUT_DIR' in src
