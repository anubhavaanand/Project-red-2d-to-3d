"""Tests for app output-directory retention (APP-08)."""

import ast
import os
import time
from pathlib import Path

import pytest

APP_PY = Path(__file__).parent.parent / "app.py"


def _prune_fn():
    """Compile _prune_output_dir without importing app.py (which needs gradio)."""
    tree = ast.parse(APP_PY.read_text())
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_prune_output_dir"
    )
    # The defaults reference module-level OUTPUT_DIR / MAX_OUTPUT_FILES /
    # OUTPUT_MAX_AGE_S; tests always pass them explicitly, so placeholders suffice.
    ns = {
        "Path": Path,
        "time": time,
        "OUTPUT_DIR": Path("/nonexistent"),
        "MAX_OUTPUT_FILES": 200,
        "OUTPUT_MAX_AGE_S": 24 * 60 * 60,
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(APP_PY), "exec"), ns)
    return ns["_prune_output_dir"]


@pytest.fixture
def prune(tmp_path):
    return _prune_fn()


def _touch(directory: Path, name: str, age_s: float) -> Path:
    path = directory / name
    path.write_bytes(b"x")
    old = time.time() - age_s
    os.utime(path, (old, old))
    return path


class TestPruneByAge:
    def test_removes_files_older_than_max_age(self, prune, tmp_path):
        old = _touch(tmp_path, "old.glb", age_s=25 * 3600)
        fresh = _touch(tmp_path, "fresh.glb", age_s=10)
        removed = prune(tmp_path, max_files=100, max_age_s=24 * 3600)
        assert removed == 1
        assert not old.exists()
        assert fresh.exists()

    def test_keeps_files_under_the_age_limit(self, prune, tmp_path):
        _touch(tmp_path, "a.glb", age_s=23 * 3600)
        _touch(tmp_path, "b.glb", age_s=60)
        assert prune(tmp_path, max_files=100, max_age_s=24 * 3600) == 0
        assert len(list(tmp_path.iterdir())) == 2


class TestPruneByCount:
    def test_evicts_oldest_when_over_the_cap(self, prune, tmp_path):
        # Oldest first, so the 3 newest must survive a cap of 2.
        for i in range(5):
            _touch(tmp_path, f"f{i}.glb", age_s=(5 - i) * 60)
        removed = prune(tmp_path, max_files=2, max_age_s=10**9)
        assert removed == 3
        remaining = sorted(p.name for p in tmp_path.iterdir())
        assert remaining == ["f3.glb", "f4.glb"]

    def test_no_op_when_under_the_cap(self, prune, tmp_path):
        for i in range(2):
            _touch(tmp_path, f"f{i}.glb", age_s=60)
        assert prune(tmp_path, max_files=5, max_age_s=10**9) == 0
        assert len(list(tmp_path.iterdir())) == 2


class TestEdgeCases:
    def test_missing_directory_is_not_an_error(self, prune, tmp_path):
        assert prune(tmp_path / "does-not-exist", max_files=5, max_age_s=60) == 0

    def test_subdirectories_are_left_alone(self, prune, tmp_path):
        (tmp_path / "nested").mkdir()
        _touch(tmp_path, "a.glb", age_s=10**6)
        assert prune(tmp_path, max_files=100, max_age_s=60) == 1
        assert (tmp_path / "nested").is_dir()

    def test_both_limits_apply_together(self, prune, tmp_path):
        # One old file (age) plus enough fresh ones to exceed the count cap.
        _touch(tmp_path, "stale.glb", age_s=25 * 3600)
        for i in range(4):
            _touch(tmp_path, f"fresh{i}.glb", age_s=30)
        removed = prune(tmp_path, max_files=2, max_age_s=24 * 3600)
        assert removed == 3  # 1 stale + 2 oldest fresh
        assert not (tmp_path / "stale.glb").exists()


class TestWiring:
    def test_module_sweeps_on_import(self):
        """APP-08: a leftover sweep must run at import, not only per request."""
        src = APP_PY.read_text()
        assert "_prune_output_dir()" in src

    def test_generation_prunes_before_writing(self):
        """The prune must run before the output path is chosen."""
        src = APP_PY.read_text()
        prune_at = src.index("_prune_output_dir()")
        out_at = src.index("out_file = OUTPUT_DIR")
        assert prune_at < out_at

    def test_defaults_are_declared(self):
        src = APP_PY.read_text()
        assert "MAX_OUTPUT_FILES" in src
        assert "OUTPUT_MAX_AGE_S" in src
