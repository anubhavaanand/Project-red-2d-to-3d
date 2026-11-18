"""Tests that the documented install path is coherent (DOC-15)."""

import re
from pathlib import Path

import pytest

# tomllib is 3.11+; the project supports 3.10, so fall back to the backport.
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).parent.parent
README = ROOT / "README.md"
PYPROJECT = ROOT / "pyproject.toml"
REQS = ROOT / "requirements.txt"


def _pyproject_deps() -> list[str]:
    data = tomllib.loads(PYPROJECT.read_text())
    deps = list(data["project"]["dependencies"])
    for group in data["project"]["optional-dependencies"].values():
        deps.extend(group)
    return deps


def _core_extra() -> list[str]:
    data = tomllib.loads(PYPROJECT.read_text())
    return list(data["project"]["optional-dependencies"]["core"])


def _req_pins() -> dict[str, str]:
    """Map package name -> pin spec from requirements.txt."""
    pins = {}
    for line in REQS.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[<>=!\[@; ]", line, maxsplit=1)[0].lower()
        pins[name] = line
    return pins


class TestCanonicalInstallIsDocumented:
    def test_readme_names_the_extras_as_canonical(self):
        readme = README.read_text()
        assert "canonical install" in readme.lower()

    def test_readme_warns_against_using_both(self):
        """DOC-15: the two paths must not be presented as sequential steps."""
        readme = README.read_text()
        assert "do not run both" in readme.lower()

    def test_readme_does_not_show_a_bare_install_then_requirements_sequence(self):
        """The original bug: `pip install -e .` immediately followed by
        `pip install -r requirements.txt`."""
        readme = README.read_text()
        # Allow the two commands to be mentioned, but not back-to-back.
        pattern = r"pip install -e \.\s*\n\s*pip install -r requirements\.txt"
        assert not re.search(pattern, readme), (
            "README still shows `pip install -e .` followed by `pip install -r requirements.txt`"
        )


class TestNoContradictionOnCoreBounds:
    """pyproject and requirements must not disagree about the same package."""

    @pytest.mark.parametrize("pkg", ["numpy", "scipy", "pydantic"])
    def test_lower_bounds_agree(self, pkg):
        pins = _req_pins()
        if pkg not in pins:
            pytest.skip(f"{pkg} not pinned in requirements.txt")
        deps = " ".join(_pyproject_deps()).lower()
        # The requirements floor must not be below the pyproject floor.
        req_floor = re.search(rf"{pkg}[>=]*([0-9]+(?:\.[0-9]+)*)", pins[pkg])
        py_floor = re.search(rf"{pkg}[>=]+([0-9]+(?:\.[0-9]+)*)", deps)
        if not (req_floor and py_floor):
            pytest.skip(f"could not parse floors for {pkg}")
        req_v = tuple(int(x) for x in req_floor.group(1).split("."))
        py_v = tuple(int(x) for x in py_floor.group(1).split("."))
        assert req_v >= py_v, (
            f"requirements.txt floors {pkg} at {req_floor.group(1)} but "
            f"pyproject.toml at {py_floor.group(1)} — installing both downgrades"
        )

    def test_pillow_is_not_pinned_to_a_vulnerable_release(self):
        """The CVE-2026 fix must be reflected in requirements too."""
        pins = _req_pins()
        pillow = pins.get("pillow", "")
        assert "13.0" in pillow or "12.3" in pillow, pillow

    def test_core_extra_is_not_empty(self):
        assert _core_extra()


class TestPinnedSetIsReproducible:
    def test_requirements_pins_the_four_core_packages(self):
        pins = _req_pins()
        for pkg in ("numpy", "pillow", "scipy", "pydantic"):
            assert pkg in pins, f"{pkg} missing from requirements.txt"

    def test_requirements_has_no_bare_git_spconv(self):
        """DOC-14 regression guard, restated here because it breaks installs."""
        text = REQS.read_text()
        for line in text.splitlines():
            if line.strip().startswith("#"):
                continue
            assert "traveller59/spconv" not in line


class TestReadmeInstallBlockIsValid:
    def test_extras_named_in_readme_exist_in_pyproject(self):
        """Every `.[extra]` the README shows must be a real extra."""
        data = tomllib.loads(PYPROJECT.read_text())
        extras = set(data["project"]["optional-dependencies"])
        readme = README.read_text()
        # Matches pip install -e ".[a,b]" and pip install -e ".[all]"
        referenced = set(re.findall(r"\.\[([a-z0-9,_ ]+)\]", readme))
        referenced = {e.strip() for group in referenced for e in group.split(",")}
        referenced.discard("")
        assert referenced, "README shows no extras at all"
        missing = referenced - extras
        assert not missing, f"README references undefined extras: {sorted(missing)}"
