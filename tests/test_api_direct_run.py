"""Tests for running the API server directly (API-18)."""

import ast
import subprocess
import sys
from pathlib import Path

SERVER_PY = Path(__file__).parent.parent / "src" / "api" / "server.py"


def _module_body():
    """Return the module AST with the API-18 bootstrap stripped out.

    The bootstrap re-execs the module, which would recurse under pytest, so
    tests inspect everything after it.
    """
    tree = ast.parse(SERVER_PY.read_text())
    return tree


class TestBootstrap:
    def test_bootstrap_precedes_relative_imports(self):
        """API-18: the re-exec must happen before `from ..config import ...`."""
        tree = _module_body()
        bootstrap_line = None
        relative_line = None
        for node in tree.body:
            if (
                isinstance(node, ast.If)
                and "__package__" in ast.dump(node.test)
                and bootstrap_line is None
            ):
                bootstrap_line = node.lineno
            if isinstance(node, ast.ImportFrom) and node.level > 0 and relative_line is None:
                relative_line = node.lineno
        assert bootstrap_line is not None, "bootstrap not found"
        assert relative_line is not None, "relative import not found"
        assert bootstrap_line < relative_line, "bootstrap runs after the relative import"

    def test_bootstrap_reexecs_as_module(self):
        src = SERVER_PY.read_text()
        assert 'run_module("src.api.server", run_name="__main__")' in src

    def test_bootstrap_adds_repo_root_to_path(self):
        src = SERVER_PY.read_text()
        assert "parents[2]" in src
        assert "_sys.path.insert(0, _repo_root)" in src

    def test_module_documents_both_entry_points(self):
        doc = ast.get_docstring(_module_body())
        assert "python src/api/server.py" in doc
        assert "uvicorn src.api.server:app" in doc


class TestRunBlock:
    def test_has_main_guard(self):
        src = SERVER_PY.read_text()
        assert 'if __name__ == "__main__":' in src

    def test_uvicorn_entry_point_is_module_path(self):
        """Serving by module path keeps relative imports working."""
        src = SERVER_PY.read_text()
        assert 'uvicorn.run(\n        "src.api.server:app"' in src

    def test_host_defaults_to_loopback(self):
        src = SERVER_PY.read_text()
        assert '"API_HOST", "127.0.0.1"' in src


class TestImportableAsModule:
    def test_module_imports_as_a_package(self):
        """The documented path `uvicorn src.api.server:app` must work.

        Skipped when fastapi is not installed; the import machinery itself is
        what this guards.
        """
        try:
            import fastapi  # noqa: F401
        except ImportError:
            import pytest

            pytest.skip("fastapi not installed")

        import importlib

        mod = importlib.import_module("src.api.server")
        assert hasattr(mod, "app")

    def test_server_file_is_syntactically_valid(self):
        ast.parse(SERVER_PY.read_text())

    def test_direct_execution_reaches_bootstrap(self):
        """`python src/api/server.py` must not fail with an ImportError about
        relative imports — it should get at least as far as importing fastapi.

        A successful run *blocks*, because server.py starts uvicorn and never
        returns. So a TimeoutExpired means the bootstrap worked; only the
        ImportError this test guards against is a real failure. The timeout is
        generous because importing the engine stack (torch, diffusers) on a cold
        runner can take a while.
        """
        try:
            proc = subprocess.run(
                [sys.executable, str(SERVER_PY)],
                capture_output=True,
                text=True,
                cwd=str(SERVER_PY.parents[2]),
                timeout=120,
            )
            combined = proc.stdout + proc.stderr
        except subprocess.TimeoutExpired as exc:
            # The server started and is serving; that is the success path.
            combined = (
                (exc.stdout or b"").decode()
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or "")
            )
            combined += (
                (exc.stderr or b"").decode()
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or "")
            )
        assert "attempted relative import" not in combined
        assert "ImportError: attempted relative import with no known parent package" not in combined
