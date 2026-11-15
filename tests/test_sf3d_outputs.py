"""Tests for SF3D's mesh payload handling (ENG-38)."""

import ast
from pathlib import Path

import numpy as np
import pytest

from src.engines._outputs import EngineOutputError, extract_mesh_object

ENGINE = Path(__file__).parent.parent / "src" / "engines" / "sf3d_engine.py"


class _Mesh:
    def __init__(self):
        self.vertices = np.zeros((3, 3), dtype=np.float32)
        self.faces = np.zeros((1, 3), dtype=np.int64)


class TestSharedValidator:
    def test_bare_mesh_returned(self):
        m = _Mesh()
        assert extract_mesh_object(m, engine="SF3D") is m

    def test_list_returns_first(self):
        m = _Mesh()
        assert extract_mesh_object([m], engine="SF3D") is m

    def test_empty_list_raises(self):
        with pytest.raises(EngineOutputError, match="empty sequence"):
            extract_mesh_object([], engine="SF3D")

    def test_dict_raises_naming_keys(self):
        """A Trimesh is not a dict — but if one ever is, say so."""
        with pytest.raises(EngineOutputError, match=r"keys: \['mesh'\]"):
            extract_mesh_object({"mesh": _Mesh()}, engine="SF3D")

    def test_object_without_vertices_raises(self):
        with pytest.raises(EngineOutputError, match="SF3D"):
            extract_mesh_object(object(), engine="SF3D")

    def test_error_names_the_engine(self):
        with pytest.raises(EngineOutputError, match="SF3D"):
            extract_mesh_object(None, engine="SF3D")


class TestSourceContract:
    def test_bare_index_is_gone(self):
        tree = ast.parse(ENGINE.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                if node.value.id == "mesh_out":
                    pytest.fail(f"bare mesh_out[...] at line {node.lineno}")

    def test_uses_the_shared_helper(self):
        src = ENGINE.read_text()
        assert "extract_mesh_object(mesh_out, engine=\"SF3D\")" in src

    def test_imports_the_helper(self):
        src = ENGINE.read_text()
        assert "from ._outputs import extract_mesh_object" in src

    def test_engine_parses(self):
        ast.parse(ENGINE.read_text())


class TestNoRegression:
    def test_spar3d_still_uses_the_shared_validator(self):
        """The SPAR3D wrapper must delegate to the shared helper."""
        src = (ENGINE.parent / "spar3d_engine.py").read_text()
        assert "return extract_mesh_object(mesh_out, engine=\"SPAR3D\")" in src

    def test_spar3d_error_subclasses_engine_output_error(self):
        src = (ENGINE.parent / "spar3d_engine.py").read_text()
        assert "class Spar3dOutputError(EngineOutputError):" in src

    def test_spar3d_call_site_uses_helper(self):
        src = (ENGINE.parent / "spar3d_engine.py").read_text()
        assert "tm = _extract_mesh_object(mesh_out)" in src