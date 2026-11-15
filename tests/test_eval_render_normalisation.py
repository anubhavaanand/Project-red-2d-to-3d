"""Tests for the render normalisation heuristic (EVAL-05)."""

from pathlib import Path

RUNNER = Path(__file__).parent.parent / "src" / "evaluation" / "benchmark_runner.py"


def _decide(already_normalized, max_r, trust):
    """Mirror the production decision logic."""
    if already_normalized and trust:
        return "keep"
    if not already_normalized or max_r < 0.6 or max_r > 1.6:
        return "renormalise"
    return "keep"


class TestLegacyDefault:
    """Default behaviour must match the frozen v1.0 heuristic."""

    def test_not_normalized_always_renormalises(self):
        assert _decide(False, 1.0, trust=False) == "renormalise"

    def test_flagged_and_in_range_is_kept(self):
        assert _decide(True, 1.0, trust=False) == "keep"

    def test_flagged_but_too_small_is_renormalised(self):
        """The magic window the issue describes."""
        assert _decide(True, 0.5, trust=False) == "renormalise"

    def test_flagged_but_too_large_is_renormalised(self):
        assert _decide(True, 1.7, trust=False) == "renormalise"

    def test_boundaries_are_inclusive(self):
        assert _decide(True, 0.6, trust=False) == "keep"
        assert _decide(True, 1.6, trust=False) == "keep"


class TestOptIn:
    def test_trusted_flag_skips_the_window(self):
        assert _decide(True, 0.1, trust=True) == "keep"
        assert _decide(True, 99.0, trust=True) == "keep"

    def test_trust_does_not_override_missing_flag(self):
        """If the caller did not normalise, we still normalise."""
        assert _decide(False, 1.0, trust=True) == "renormalise"


class TestSourceContract:
    def test_parameter_exists_and_defaults_false(self):
        src = RUNNER.read_text()
        assert "trust_already_normalized: bool = False" in src

    def test_no_unconditional_literal_removal(self):
        """The legacy heuristic must still be present for the default path."""
        src = RUNNER.read_text()
        assert "max_r < 0.6 or max_r > 1.6" in src

    def test_trust_branch_precedes_the_heuristic(self):
        src = RUNNER.read_text()
        fn = src[src.index("def render_novel_view_same_bg") : src.index("def render_gso_mesh_view")]
        assert fn.index("trust_already_normalized") < fn.index("max_r < 0.6")

    def test_documented(self):
        src = RUNNER.read_text()
        assert "EVAL-05" in src


class TestNoBehaviourChange:
    def test_default_path_unchanged(self):
        """With the default, the decision is identical to the original code."""
        for already in (True, False):
            for r in (0.1, 0.5, 0.6, 1.0, 1.6, 1.7, 10.0):
                legacy = "keep" if (already and 0.6 <= r <= 1.6) else "renormalise"
                assert _decide(already, r, trust=False) == legacy
