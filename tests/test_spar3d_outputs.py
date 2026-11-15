"""Tests for SPAR3D's mesh payload extraction (ENG-36)."""

import ast
from pathlib import Path

import numpy as np
import pytest

from src.engines._outputs import EngineOutputError
from src.engines._outputs import extract_mesh_object as _shared_extract

ENGINE = Path(__file__).parent.parent / "src" / "engines" / "spar3d_engine.py"


def _extract():
    """Compile SPAR3D's thin wrapper, backed by the real shared validator."""
    tree = ast.parse(ENGINE.read_text())
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_extract_mesh_object"
    )
    ns = {
        "Spar3dOutputError": EngineOutputError,
        "extract_mesh_object": _shared_extract,
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(ENGINE), "exec"), ns)
    return ns["_extract_mesh_object"], EngineOutputError


class _Mesh:
    def __init__(self):
        self.vertices = np.zeros((3, 3), dtype=np.float32)
        self.faces = np.zeros((1, 3), dtype=np.int64)


class TestExtractMeshObject:
    def test_bare_mesh_is_returned(self):
        fn, _ = _extract()
        m = _Mesh()
        assert fn(m) is m

    def test_list_returns_first(self):
        fn, _ = _extract()
        m = _Mesh()
        assert fn([m, "extra"]) is m

    def test_tuple_returns_first(self):
        fn, _ = _extract()
        m = _Mesh()
        assert fn((m, None)) is m

    def test_empty_list_raises(self):
        fn, Err = _extract()
        with pytest.raises(Err, match="empty sequence"):
            fn([])

    def test_dict_raises_with_keys(self):
        """ENG-36: a dict return must not be indexed by 0."""
        fn, Err = _extract()
        with pytest.raises(Err, match=r"dict \(keys: \['mesh'\]\)"):
            fn({"mesh": _Mesh()})

    def test_object_without_vertices_raises(self):
        fn, Err = _extract()
        with pytest.raises(Err, match="no vertices/faces"):
            fn(object())

    def test_error_names_the_type(self):
        fn, Err = _extract()
        with pytest.raises(Err, match="str"):
            fn("not a mesh")

    def test_none_raises(self):
        fn, Err = _extract()
        with pytest.raises(Err):
            fn(None)


class TestSourceContract:
    def test_old_expression_is_gone(self):
        """The old conditional-index expression must not appear in real code."""
        tree = ast.parse(ENGINE.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.IfExp):
                text = ast.unparse(node)
                if "mesh_out" in text:
                    pytest.fail(f"old conditional still present: {text}")

    def test_call_site_uses_helper(self):
        src = ENGINE.read_text()
        assert "tm = _extract_mesh_object(mesh_out)" in src

    def test_error_type_defined(self):
        src = ENGINE.read_text()
        assert "class Spar3dOutputError(EngineOutputError):" in src

    def test_engine_parses(self):
        ast.parse(ENGINE.read_text())

    def test_run_image_is_unpacked_as_a_pair(self):
        """Both engines unpack (mesh, aux) — pins the real contract."""
        for name in ("spar3d_engine.py", "sf3d_engine.py"):
            src = (ENGINE.parent / name).read_text()
            assert (
                "mesh_out, _ = self.model.run_image(" in src
                or "mesh_out, glob_dict = self.model.run_image(" in src
            )
