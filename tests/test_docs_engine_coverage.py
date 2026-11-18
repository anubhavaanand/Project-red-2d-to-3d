"""Tests that every registered engine is documented (DOC-03)."""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
README = ROOT / "README.md"
CONFIG = ROOT / "src" / "config.py"


def _engine_types() -> list[str]:
    """Read EngineType members from config.py without importing it."""
    tree = ast.parse(CONFIG.read_text())
    enum = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "EngineType")
    names = []
    for node in enum.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            names.append(node.value.value)
    return names


ENGINE_TYPES = _engine_types()

# Engines that appear in the headline 400-run benchmark.
HEADLINE = {"TRIPOSR", "LGM", "SPAR3D", "TRELLIS"}
# Documented, but explicitly out of the headline run.
ABLATION_ONLY = {"SF3D"}


class TestEveryEngineIsDocumented:
    @pytest.mark.parametrize("engine", sorted(ENGINE_TYPES))
    def test_engine_appears_in_the_readme(self, engine):
        """DOC-03: a registered engine must not be invisible to readers."""
        readme = README.read_text()
        # Match the engine id as a word, case-insensitively.
        assert re.search(rf"\b{re.escape(engine)}\b", readme, re.IGNORECASE), (
            f"{engine} is in EngineType but never mentioned in README.md"
        )

    @pytest.mark.parametrize("engine", sorted(ENGINE_TYPES))
    def test_engine_appears_in_agents_md(self, engine):
        agents = ROOT / "AGENTS.md"
        if not agents.exists():
            pytest.skip("AGENTS.md not present")
        assert re.search(rf"\b{re.escape(engine)}\b", agents.read_text(), re.IGNORECASE), (
            f"{engine} is missing from AGENTS.md"
        )


class TestHeadlineSetIsUnchanged:
    """The headline table must stay the four engines behind the published numbers."""

    def test_headline_engines_are_exactly_the_published_four(self):
        assert HEADLINE == {"TRIPOSR", "LGM", "SPAR3D", "TRELLIS"}

    def test_results_csv_has_only_headline_engines(self):
        """DOC-03 guard: adding an engine to the CSV would change the headline."""
        csv = ROOT / "results" / "unified_results_4engines_100obj.csv"
        if not csv.exists():
            pytest.skip("results CSV not present")
        engines = set()
        with csv.open() as f:
            next(f)  # header
            for line in f:
                if line.strip():
                    engines.add(line.split(",")[0].strip())
        # The CSV spells TripoSR in mixed case; compare case-insensitively.
        normalised = {e.upper() for e in engines}
        assert normalised == HEADLINE, f"results CSV engines changed: {engines}"


class TestSf3dIsMarkedAsAblation:
    """SF3D must not be presented as part of the headline run."""

    def test_sf3d_row_says_it_is_ablation_only(self):
        readme = README.read_text()
        # Find the SF3D table row and check its status cell.
        row = next(
            (line for line in readme.splitlines() if line.strip().startswith("| **SF3D**")),
            None,
        )
        assert row is not None, "no SF3D row in README"
        assert "ablation" in row.lower(), f"SF3D row does not say ablation: {row}"

    def test_sf3d_is_not_in_the_results_csv(self):
        csv = ROOT / "results" / "unified_results_4engines_100obj.csv"
        if not csv.exists():
            pytest.skip("results CSV not present")
        assert "SF3D" not in csv.read_text()


class TestInstallPointer:
    def test_sf3d_extra_is_documented(self):
        """Readers must be told how to install SF3D prerequisites."""
        readme = README.read_text()
        assert "sf3d" in readme
        assert "sf3d" in (ROOT / "pyproject.toml").read_text()
