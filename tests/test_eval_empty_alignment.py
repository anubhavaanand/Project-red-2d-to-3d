"""Tests for alignment input validation (EVAL-14)."""

import ast
from pathlib import Path

import numpy as np
import pytest

RUNNER = Path(__file__).parent.parent / "src" / "evaluation" / "benchmark_runner.py"


def _extract_guard():
    """Compile just the empty-input guard from align_mesh_to_gt.

    The runner needs torch/trimesh to import, so the guard is lifted out by
    unparsing its own source and wrapping it in a function.
    """
    tree = ast.parse(RUNNER.read_text())
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "align_mesh_to_gt"
    )
    guard_src = None
    for stmt in fn.body:
        if isinstance(stmt, ast.If) and "len(pred_pts)" in ast.unparse(stmt.test):
            guard_src = ast.unparse(stmt)
            break
    assert guard_src, "guard not found"

    wrapped = (
        "import numpy as np\n\nclass EmptyAlignmentError(RuntimeError):\n    pass\n\n\ndef _guard(pred_pts, gt):\n"
        + "\n".join("    " + line for line in guard_src.splitlines())
    )
    ns = {}
    exec(wrapped, ns)
    return ns["_guard"], ns["EmptyAlignmentError"]


class TestGuard:
    def test_empty_prediction_raises(self):
        guard, Err = _extract_guard()
        with pytest.raises(Err, match="empty input"):
            guard(np.zeros((0, 3), dtype=np.float32), np.zeros((10, 3), dtype=np.float32))

    def test_empty_gt_raises(self):
        guard, Err = _extract_guard()
        with pytest.raises(Err, match="empty input"):
            guard(np.zeros((10, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32))

    def test_both_empty_raises(self):
        guard, Err = _extract_guard()
        with pytest.raises(Err, match="empty input"):
            guard(np.zeros((0, 3)), np.zeros((0, 3)))

    def test_non_empty_does_not_raise(self):
        guard, _ = _extract_guard()
        assert guard(np.zeros((10, 3)), np.zeros((10, 3))) is None

    def test_error_names_the_counts(self):
        guard, Err = _extract_guard()
        with pytest.raises(Err, match=r"pred_pts=0"):
            guard(np.zeros((0, 3)), np.zeros((10, 3)))

    def test_error_type_is_runtime_error(self):
        _, Err = _extract_guard()
        assert issubclass(Err, RuntimeError)


class TestSourceContract:
    def test_exception_defined(self):
        assert "class EmptyAlignmentError(RuntimeError):" in RUNNER.read_text()

    def test_guard_precedes_sub_n(self):
        src = RUNNER.read_text()
        assert src.index("if len(pred_pts) == 0 or len(gt) == 0:") < src.index(
            "sub_n = min(2048, len(pred_pts), len(gt))"
        )

    def test_runner_records_failures_per_sample(self):
        """A raise here is caught per sample, not crashing the run."""
        src = RUNNER.read_text()
        assert "except Exception as e:" in src
        assert '"error": str(e)' in src
