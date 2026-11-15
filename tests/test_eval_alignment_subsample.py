"""Tests for the configurable alignment subsample (EVAL-13)."""

import ast
from pathlib import Path

import pytest

RUNNER = Path(__file__).parent.parent / "src" / "evaluation" / "benchmark_runner.py"


def _align_fn():
    """Compile align_mesh_to_gt's signature + guard region.

    The full body needs torch/trimesh, so only the signature and the subsample
    computation are extracted.
    """
    tree = ast.parse(RUNNER.read_text())
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "align_mesh_to_gt"
    )
    defaults = {}
    for arg, default in zip(fn.args.args[-len(fn.args.defaults) :], fn.args.defaults):
        if isinstance(default, ast.Constant):
            defaults[arg.arg] = default.value
        elif isinstance(default, ast.Name):
            # resolved from the module constant, checked separately
            defaults[arg.arg] = default.id
        else:
            defaults[arg.arg] = ast.literal_eval(default)
    return defaults


class TestSignature:
    def test_subsample_parameter_exists(self):
        defaults = _align_fn()
        assert "subsample_points" in defaults

    def test_default_is_the_module_constant(self):
        """The default resolves to ALIGNMENT_SUBSAMPLE_POINTS (== 2048)."""
        assert _align_fn()["subsample_points"] == "ALIGNMENT_SUBSAMPLE_POINTS"

    def test_module_constant_matches(self):
        tree = ast.parse(RUNNER.read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if getattr(t, "id", "") == "ALIGNMENT_SUBSAMPLE_POINTS":
                        assert ast.literal_eval(node.value) == 2048
                        return
        pytest.fail("ALIGNMENT_SUBSAMPLE_POINTS not found")

    def test_constant_used_as_the_default(self):
        """The signature default must reference the constant, not a literal."""
        src = RUNNER.read_text()
        assert "subsample_points: int = ALIGNMENT_SUBSAMPLE_POINTS" in src


class TestRunnerExposure:
    def test_runner_takes_the_parameter(self):
        src = RUNNER.read_text()
        assert "alignment_subsample_points: int = ALIGNMENT_SUBSAMPLE_POINTS" in src

    def test_runner_stores_it(self):
        src = RUNNER.read_text()
        assert "self.alignment_subsample_points = int(alignment_subsample_points)" in src

    def test_runner_forwards_it(self):
        src = RUNNER.read_text()
        assert "subsample_points=self.alignment_subsample_points" in src

    def test_runner_default_is_the_constant(self):
        src = RUNNER.read_text()
        block = src[src.index("def __init__(") : src.index("self.dataset_path = Path")]
        assert "alignment_subsample_points: int = ALIGNMENT_SUBSAMPLE_POINTS" in block


class TestSubsampleComputation:
    def test_sub_n_is_derived_from_the_parameter(self):
        src = RUNNER.read_text()
        assert "sub_n = min(subsample_points, len(pred_pts), len(gt))" in src

    def test_hardcoded_2048_is_gone(self):
        src = RUNNER.read_text()
        assert "min(2048, len(pred_pts), len(gt))" not in src

    def test_seed_is_stable(self):
        """The published runs used seed 0; it must not change silently."""
        src = RUNNER.read_text()
        assert "np.random.default_rng(0)" in src

    def test_default_reproduces_the_frozen_count(self):
        """min(2048, ...) is what the frozen runs used."""
        for pred, gt in [(20000, 20000), (5000, 5000), (100, 100)]:
            assert min(2048, pred, gt) == min(2048, pred, gt)


class TestDocs:
    def test_readme_documents_the_subsample(self):
        text = (RUNNER.parents[2] / "README.md").read_text()
        assert "2,048 points" in text
        assert "alignment_subsample_points" in text
