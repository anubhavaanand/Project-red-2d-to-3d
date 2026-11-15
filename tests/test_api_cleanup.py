"""Tests for API artefact cleanup (API-06)."""

import ast
import os
import time
from pathlib import Path

SERVER_PY = Path(__file__).parent.parent / "src" / "api" / "server.py"


def _sweep_fn(output_dir: str):
    """Compile _sweep_output_dir with OUTPUT_DIR bound to a temp directory."""
    src = SERVER_PY.read_text()
    lines = src.splitlines()
    tree = ast.parse(src)
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_sweep_output_dir"
    )
    body = "\n".join(lines[fn.lineno - 1 : fn.end_lineno])
    ns = {
        "time": time,
        "Path": Path,
        "logger": type("L", (), {"info": staticmethod(lambda *a, **k: None)})(),
        "OUTPUT_DIR": output_dir,
    }
    exec(compile(body, str(SERVER_PY), "exec"), ns)
    return ns["_sweep_output_dir"]


class TestSweepOutputDir:
    def test_removes_stale_files(self, tmp_path):
        old = tmp_path / "stale.glb"
        old.write_bytes(b"x")
        os.utime(old, (time.time() - 7200, time.time() - 7200))
        assert _sweep_fn(str(tmp_path))() == 1
        assert not old.exists()

    def test_keeps_fresh_files(self, tmp_path):
        fresh = tmp_path / "fresh.glb"
        fresh.write_bytes(b"y")
        assert _sweep_fn(str(tmp_path))() == 0
        assert fresh.exists()

    def test_keeps_subdirectories(self, tmp_path):
        sub = tmp_path / "nested"
        sub.mkdir()
        os.utime(sub, (time.time() - 99999, time.time() - 99999))
        assert _sweep_fn(str(tmp_path))() == 0
        assert sub.exists()

    def test_empty_directory(self, tmp_path):
        assert _sweep_fn(str(tmp_path))() == 0

    def test_missing_directory_does_not_raise(self, tmp_path):
        assert _sweep_fn(str(tmp_path / "nope"))() == 0

    def test_respects_max_age(self, tmp_path):
        f = tmp_path / "recent.glb"
        f.write_bytes(b"z")
        os.utime(f, (time.time() - 100, time.time() - 100))
        assert _sweep_fn(str(tmp_path))(max_age_s=3600) == 0
        assert _sweep_fn(str(tmp_path))(max_age_s=10) == 1
        assert not f.exists()

    def test_runs_at_import(self):
        """The sweep must actually be invoked once, at startup."""
        src = SERVER_PY.read_text()
        tree = ast.parse(src)
        calls = [
            n
            for n in tree.body
            if isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call)
            and getattr(n.value.func, "id", "") == "_sweep_output_dir"
        ]
        assert calls, "_sweep_output_dir() is never called at module level"

    def test_per_request_cleanup_still_present(self):
        """API-06 keeps SEC-06's per-request BackgroundTask cleanup."""
        src = SERVER_PY.read_text()
        assert src.count("BackgroundTask(_cleanup_file, out_path)") >= 2
