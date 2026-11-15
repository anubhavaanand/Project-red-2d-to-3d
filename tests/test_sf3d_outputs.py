"""Behavioural tests for SF3D's mesh payload handling (ENG-38)."""

import ast
import inspect
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from src.engines._outputs import EngineOutputError, extract_mesh_object

ENGINE = Path(__file__).parent.parent / "src" / "engines" / "sf3d_engine.py"


@pytest.fixture(scope="module")
def sf3d():
    """Import sf3d_engine with a minimal torch stub."""
    t = types.ModuleType("torch")
    t.__version__ = "0.0-stub"
    t.float32, t.int64 = "float32", "int64"
    t.device = lambda x: x
    t.cuda = types.SimpleNamespace(
        is_available=lambda: False, empty_cache=lambda: None, device=lambda d: _NoCtx()
    )

    class _NoCtx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def __call__(self, fn):
            return fn

    t.no_grad = lambda: _NoCtx()
    t.nn = types.ModuleType("torch.nn")
    t.nn.functional = types.ModuleType("torch.nn.functional")
    sys.modules["torch"] = t
    sys.modules["torch.nn"] = t.nn
    sys.modules["torch.nn.functional"] = t.nn.functional

    from src.engines import sf3d_engine

    return sf3d_engine


class _Mesh:
    def __init__(self):
        self.vertices = np.zeros((3, 3), dtype=np.float32)
        self.faces = np.zeros((1, 3), dtype=np.int64)
        self.visual = types.SimpleNamespace(uv=None, material=None)


class TestSharedValidator:
    def test_bare_mesh_returned(self):
        m = _Mesh()
        assert extract_mesh_object(m, engine="SF3D") is m

    def test_list_returns_first(self):
        m = _Mesh()
        assert extract_mesh_object([m], engine="SF3D") is m

    def test_tuple_returns_first(self):
        m = _Mesh()
        assert extract_mesh_object((m,), engine="SF3D") is m

    def test_empty_list_raises(self):
        with pytest.raises(EngineOutputError, match="empty sequence"):
            extract_mesh_object([], engine="SF3D")

    def test_dict_raises_naming_keys(self):
        with pytest.raises(EngineOutputError, match=r"keys: \['mesh'\]"):
            extract_mesh_object({"mesh": _Mesh()}, engine="SF3D")

    def test_object_without_vertices_raises(self):
        with pytest.raises(EngineOutputError, match="SF3D"):
            extract_mesh_object(object(), engine="SF3D")

    def test_error_names_the_engine(self):
        with pytest.raises(EngineOutputError, match="SF3D"):
            extract_mesh_object(None, engine="SF3D")


class TestPredictUsesTheValidator:
    """Run the real predict() with a stubbed model and payload."""

    def _engine(self, sf3d, run_image_return):
        e = sf3d.SF3DEngine(device="cpu")
        e.model = types.SimpleNamespace(
            run_image=lambda *a, **k: (run_image_return, {}),
            to=lambda d: None,
        )
        return e

    def test_valid_mesh_produces_a_result(self, sf3d):
        from PIL import Image

        e = self._engine(sf3d, _Mesh())
        result = e.predict(Image.new("RGBA", (32, 32), (10, 20, 30, 255)))
        assert result.mesh is not None
        assert len(result.mesh.vertices) == 3

    def test_bad_payload_raises_engine_output_error(self, sf3d):
        """ENG-38: a dict return must not be indexed as a mesh."""
        from PIL import Image

        e = self._engine(sf3d, {"mesh": _Mesh()})
        with pytest.raises(EngineOutputError, match="dict"):
            e.predict(Image.new("RGBA", (32, 32), (10, 20, 30, 255)))

    def test_bare_object_without_vertices_raises(self, sf3d):
        from PIL import Image

        e = self._engine(sf3d, object())
        with pytest.raises(EngineOutputError, match="no vertices/faces"):
            e.predict(Image.new("RGBA", (32, 32), (10, 20, 30, 255)))

    def test_none_payload_raises(self, sf3d):
        from PIL import Image

        e = self._engine(sf3d, None)
        with pytest.raises(EngineOutputError):
            e.predict(Image.new("RGBA", (32, 32), (10, 20, 30, 255)))


class TestWiring:
    def test_no_bare_index_remains(self):
        tree = ast.parse(ENGINE.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                if node.value.id == "mesh_out":
                    pytest.fail(f"bare mesh_out[...] at line {node.lineno}")

    def test_uses_the_shared_helper(self):
        src = inspect.getsource(
            __import__("src.engines.sf3d_engine", fromlist=["x"]).SF3DEngine.predict
        )
        assert 'extract_mesh_object(mesh_out, engine="SF3D")' in src

    def test_engine_parses(self):
        ast.parse(ENGINE.read_text())


class TestSpar3dStillUsesTheSharedValidator:
    def test_spar3d_delegates(self):
        src = (Path(ENGINE).parent / "spar3d_engine.py").read_text()
        assert 'return extract_mesh_object(mesh_out, engine="SPAR3D")' in src

    def test_spar3d_error_subclasses_engine_output_error(self):
        src = (Path(ENGINE).parent / "spar3d_engine.py").read_text()
        assert "class Spar3dOutputError(EngineOutputError):" in src
