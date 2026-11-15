"""Tests for SPAR3D normal orientation handling (ENG-33)."""

import ast
from pathlib import Path

import numpy as np
import pytest

ENGINE = Path(__file__).parent.parent / "src" / "engines" / "spar3d_engine.py"

trimesh = pytest.importorskip("trimesh")


def _ensure():
    tree = ast.parse(ENGINE.read_text())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_ensure_outward_normals"
    )
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(ENGINE), "exec"), ns)
    return ns["_ensure_outward_normals"]


def _mesh(invert=False):
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    f = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    if invert:
        f = f[:, ::-1]
    return trimesh.Trimesh(vertices=v, faces=f, process=False)


class TestEnsureOutwardNormals:
    def test_outward_mesh_is_left_alone(self):
        tm = _mesh()
        note = _ensure()(tm)
        assert note == "already outward"
        assert tm.volume > 0

    def test_inverted_mesh_is_fixed(self):
        """ENG-33: a negative signed volume must be repaired, not silently kept."""
        tm = _mesh(invert=True)
        assert tm.volume < 0
        note = _ensure()(tm)
        assert "fixed" in note
        assert tm.volume > 0

    def test_decision_is_reported(self):
        tm = _mesh(invert=True)
        note = _ensure()(tm)
        assert "inverted" in note

    def test_failure_is_reported_not_swallowed(self):
        class Broken:
            @property
            def volume(self):
                raise RuntimeError("no volume")

        note = _ensure()(Broken())
        assert "volume unavailable" in note

    def test_fix_failure_is_reported(self):
        class BadFix(_mesh().__class__):
            def fix_normals(self):
                raise RuntimeError("cannot fix")

        tm = _mesh(invert=True)
        tm.fix_normals = lambda: (_ for _ in ()).throw(RuntimeError("cannot fix"))
        note = _ensure()(tm)
        assert "fix failed" in note


class TestSourceContract:
    def test_no_bare_invert_call(self):
        """ENG-33: tm.invert() must not be called directly."""
        tree = ast.parse(ENGINE.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "invert":
                pytest.fail(f"bare tm.invert() at line {node.lineno}")

    def test_uses_the_helper(self):
        src = ENGINE.read_text()
        assert "normal_fix = _ensure_outward_normals(tm)" in src

    def test_decision_recorded_in_metadata(self):
        src = ENGINE.read_text()
        assert '"normal_fix": normal_fix' in src

    def test_engine_parses(self):
        ast.parse(ENGINE.read_text())


class TestUnderlyingAssumption:
    def test_signed_volume_distinguishes_winding(self):
        """Pins why the signed volume is a valid check."""
        assert _mesh().volume > 0
        assert _mesh(invert=True).volume < 0

    def test_fix_normals_repairs_winding(self):
        tm = _mesh(invert=True)
        tm.fix_normals()
        assert tm.volume > 0
