"""Tests that the spconv install guidance is consistent (DOC-14)."""

from pathlib import Path

ROOT = Path(__file__).parent.parent
COMPILE = ROOT / "docs" / "COMPILE.md"
REQS = ROOT / "requirements.txt"
PYPROJECT = ROOT / "pyproject.toml"


class TestNoGitSpconv:
    def test_requirements_does_not_install_spconv_from_git(self):
        """DOC-14: the traveller59 recipe no longer builds on modern Python."""
        for line in REQS.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "traveller59/spconv" not in stripped, (
                "requirements.txt must not install spconv from the git repo; "
                "use a prebuilt wheel per docs/COMPILE.md"
            )

    def test_pyproject_does_not_install_spconv_from_git(self):
        """The engines extra must not reintroduce the discouraged path."""
        assert "traveller59/spconv" not in PYPROJECT.read_text()

    def test_requirements_points_at_the_wheel_instead(self):
        text = REQS.read_text()
        assert "spconv-cu126" in text
        assert "spconv-cu120" in text


class TestCompileGuide:
    def test_compile_md_recommends_the_wheels(self):
        text = COMPILE.read_text()
        assert "spconv-cu126" in text

    def test_compile_md_warns_against_the_old_recipe(self):
        text = COMPILE.read_text()
        assert "Do not clone" in text

    def test_code_fences_are_balanced(self):
        """An unbalanced fence silently swallows the rest of the document."""
        fences = [
            line for line in COMPILE.read_text().splitlines() if line.strip().startswith("```")
        ]
        assert len(fences) % 2 == 0, f"{len(fences)} fence markers is not even"

    def test_no_prose_leaked_into_a_code_block(self):
        """The cumm sentence used to sit inside the pip install block."""
        text = COMPILE.read_text()
        # The sentence must be prose, not a line inside a fenced block.
        lines = text.splitlines()
        inside = False
        for line in lines:
            if line.strip().startswith("```"):
                inside = not inside
                continue
            if inside and "only practical route to the" in line:
                raise AssertionError(f"prose trapped in a code block: {line!r}")


class TestDocsAgreeWithEachOther:
    def test_mentions_python_and_torch_floors(self):
        """The guide must say which Python/torch the wheels target."""
        text = COMPILE.read_text()
        assert "3.13" in text or "3.12" in text
        assert "torch 2.11" in text or "2.11" in text
