"""Tests for TripoSR's mesh orientation check (ENG-06)."""

import ast
from pathlib import Path

import numpy as np
import pytest

ENGINE = Path(__file__).parent.parent / "src" / "engines" / "triposr_engine.py"

trimesh = pytest.importorskip("trimesh")


def _helpers():
    tree = ast.parse(ENGINE.read_text())
    fns = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in ("_signed_volume_about", "_orient_outward")
    ]
    ns = {"np": np}
    exec(compile(ast.Module(body=fns, type_ignores=[]), str(ENGINE), "exec"), ns)
    return ns["_signed_volume_about"], ns["_orient_outward"]


class _Mesh:
    """Minimal stand-in for the trimesh object TripoSR hands us."""

    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces
        self.inverted = False

    def invert(self):
        self.inverted = not self.inverted


@pytest.fixture(scope="module")
def sphere():
    tm = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    tm.merge_vertices()
    return np.asarray(tm.vertices, dtype=np.float64), np.asarray(tm.faces, dtype=np.int64)


class TestSignedVolumeAbout:
    def test_volume_is_positive_for_outward_sphere(self, sphere):
        v, f = sphere
        vol, _ = _helpers()
        assert vol(v, f, v.mean(axis=0)) > 0

    def test_volume_is_negative_when_inverted(self, sphere):
        v, f = sphere
        vol, _ = _helpers()
        assert vol(v, f[:, ::-1], v.mean(axis=0)) < 0

    def test_translation_invariant_about_centroid(self, sphere):
        """ENG-06: the origin-based form was not translation invariant."""
        v, f = sphere
        vol, _ = _helpers()
        c = v.mean(axis=0)
        assert vol(v, f, c) == pytest.approx(vol(v + 100.0, f, c + 100.0), rel=1e-6)

    def test_origin_form_is_unstable_for_open_meshes(self):
        """The actual ENG-06 failure: an open mesh's sign depends on origin.

        A single flat sheet has no volume at all, but its origin-based sum is
        not translation invariant — move it and the sign can change.
        """
        vol, _ = _helpers()
        sheet = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

        near = vol(sheet, faces, np.zeros(3))
        far = vol(sheet + 100.0, faces, np.zeros(3))
        assert near != pytest.approx(far, rel=1e-6)

        # while the centroid form is stable for the same sheet
        c = sheet.mean(axis=0)
        assert vol(sheet, faces, c) == pytest.approx(vol(sheet + 100.0, faces, c + 100.0), rel=1e-6)

    def test_watertight_mesh_is_translation_invariant_either_way(self, sphere):
        """Documents why the old code usually worked: closed meshes are fine."""
        v, f = sphere
        vol, _ = _helpers()
        origin = np.zeros(3)
        assert vol(v, f, origin) == pytest.approx(vol(v + 100.0, f, origin), rel=1e-6)


class TestOrientOutward:
    def test_outward_sphere_left_alone(self, sphere):
        v, f = sphere
        _, orient = _helpers()
        assert orient(_Mesh(v, f)) == "already outward"

    def test_inward_sphere_is_flipped(self, sphere):
        v, f = sphere
        _, orient = _helpers()
        m = _Mesh(v, f[:, ::-1])
        assert "inverted" in orient(m)
        assert m.inverted

    def test_open_mesh_is_left_alone(self, sphere):
        """ENG-06: an open mesh has no meaningful sign, so do not flip it."""
        v, f = sphere
        _, orient = _helpers()
        m = _Mesh(v, f[: len(f) // 2])
        note = orient(m)
        assert "not watertight" in note
        assert not m.inverted

    def test_inconsistent_winding_is_left_alone(self, sphere):
        v, f = sphere
        _, orient = _helpers()
        f_mixed = f.copy()
        f_mixed[0] = f_mixed[0][::-1]
        m = _Mesh(v, f_mixed)
        assert "not watertight" in orient(m)

    def test_empty_mesh_is_handled(self):
        _, orient = _helpers()
        assert orient(_Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))) == "no faces"


class TestSourceContract:
    def test_origin_based_check_is_gone(self):
        src = ENGINE.read_text()
        assert "np.sum(v0 * np.cross(v1, v2))" not in src.split("def _signed_volume_about")[0]

    def test_uses_the_helper(self):
        src = ENGINE.read_text()
        assert "_orient_outward(mesh_obj)" in src

    def test_bare_except_pass_is_gone(self):
        src = ENGINE.read_text()
        orient_call = src[
            src.index("_orient_outward(mesh_obj)") : src.index("vertex_colors = None")
        ]
        assert "except Exception:\n            pass" not in orient_call

    def test_engine_parses(self):
        ast.parse(ENGINE.read_text())
