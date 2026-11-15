"""Tests for app.py import hygiene (APP-06)."""

import ast
from pathlib import Path

import tomllib

APP_PY = Path(__file__).parent.parent / "app.py"
PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"


class TestPackaging:
    def test_pyproject_declares_a_build_system(self):
        data = tomllib.loads(PYPROJECT.read_text())
        assert "build-system" in data
        assert "setuptools" in data["build-system"]["requires"][0]

    def test_project_metadata_exists(self):
        data = tomllib.loads(PYPROJECT.read_text())
        assert data["project"]["name"] == "image-to-3d-benchmark"
        assert data["project"]["requires-python"]

    def test_all_subpackages_are_declared(self):
        """Every directory with an __init__.py must be listed for installation."""
        data = tomllib.loads(PYPROJECT.read_text())
        declared = set(data["tool"]["setuptools"]["packages"])
        root = PYPROJECT.parent
        found = set()
        for init in (root / "src").rglob("__init__.py"):
            rel = init.parent.relative_to(root)
            found.add(".".join(rel.parts))
        assert found == declared, f"undeclared: {found - declared}"


class TestAppImports:
    def test_does_not_prepend_repo_root(self):
        """APP-06: the import must not put the repo root at sys.path[0]."""
        src = APP_PY.read_text()
        assert "sys.path.insert(0, str(Path(__file__).parent))" not in src

    def test_relative_imports_are_absolute(self):
        """app.py must import src.* absolutely, not relatively."""
        tree = ast.parse(APP_PY.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.level == 0, f"relative import at line {node.lineno}"

    def test_sys_path_mutation_is_append_only_and_guarded(self):
        """Any sys.path change must be an append inside a fallback."""
        tree = ast.parse(APP_PY.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Attribute)
                    and func.value.attr == "path"
                    and func.value.value.id == "sys"
                ):
                    assert func.attr == "append", f"sys.path.{func.attr} is not append-only"

    def test_fallback_is_tied_to_import_failure(self):
        """The sys.path fallback must sit in an except block, not at module top."""
        src = APP_PY.read_text()
        try_block = src.split("except ModuleNotFoundError")[0]
        assert "sys.path.append" not in try_block, "sys.path mutated before the try"

    def test_app_parses(self):
        ast.parse(APP_PY.read_text())
