"""Tests for the engine dataclass validation (ENG-42)."""

import numpy as np
import pytest

from src.engines.base import GaussianCloud, Mesh3D


def _mesh(**kw):
    base = {
        "vertices": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
        "faces": np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64),
    }
    base.update(kw)
    return Mesh3D(**base)


def _cloud(n=4):
    return GaussianCloud(
        positions=np.zeros((n, 3), dtype=np.float32),
        scales=np.ones((n, 3), dtype=np.float32),
        rotations=np.zeros((n, 4), dtype=np.float32),
        opacities=np.ones((n, 1), dtype=np.float32),
        sh_coeffs=np.zeros((n, 3), dtype=np.float32),
    )


class TestMesh3DValidate:
    def test_valid_mesh_passes(self):
        _mesh().validate()

    def test_empty_mesh_rejected(self):
        with pytest.raises(ValueError, match="no vertices"):
            _mesh(
                vertices=np.zeros((0, 3), dtype=np.float32),
                faces=np.zeros((0, 3), dtype=np.int64),
            ).validate()

    def test_bad_vertex_shape_rejected(self):
        with pytest.raises(ValueError, match=r"vertices must be \(N, 3\)"):
            _mesh(vertices=np.zeros((4, 2), dtype=np.float32)).validate()

    def test_bad_face_shape_rejected(self):
        with pytest.raises(ValueError, match=r"faces must be \(M, 3\)"):
            _mesh(faces=np.zeros((2, 4), dtype=np.int64)).validate()

    def test_out_of_range_face_index_rejected(self):
        with pytest.raises(ValueError, match="out of range"):
            _mesh(faces=np.array([[0, 1, 99]], dtype=np.int64)).validate()

    def test_non_finite_vertices_rejected(self):
        v = np.array([[0, 0, 0], [np.nan, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        with pytest.raises(ValueError, match="non-finite"):
            _mesh(vertices=v).validate()

    def test_color_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="vertex_colors length"):
            _mesh(vertex_colors=np.zeros((3, 3), dtype=np.float32)).validate()

    def test_uv_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="uv_coords length"):
            _mesh(uv_coords=np.zeros((2, 2), dtype=np.float32)).validate()

    def test_matching_colors_and_uvs_pass(self):
        _mesh(
            vertex_colors=np.zeros((4, 3), dtype=np.float32),
            uv_coords=np.zeros((4, 2), dtype=np.float32),
        ).validate()


class TestGaussianCloudValidate:
    def test_valid_cloud_passes(self):
        _cloud().validate()

    def test_empty_cloud_rejected(self):
        with pytest.raises(ValueError, match="no points"):
            _cloud(n=0).validate()

    def test_length_mismatch_rejected(self):
        c = _cloud(n=4)
        c.scales = np.ones((3, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="scales length"):
            c.validate()

    def test_non_finite_positions_rejected(self):
        c = _cloud()
        c.positions = np.array([[np.inf, 0, 0]] * 4, dtype=np.float32)
        with pytest.raises(ValueError, match="positions contains non-finite"):
            c.validate()

    def test_non_finite_sh_rejected(self):
        c = _cloud()
        c.sh_coeffs = np.array([[np.nan, 0, 0]] * 4, dtype=np.float32)
        with pytest.raises(ValueError, match="sh_coeffs contains non-finite"):
            c.validate()


class TestWiredIntoEngines:
    def test_every_engine_validates(self):
        from pathlib import Path

        base = Path(__file__).parent.parent / "src" / "engines"
        for f in (
            "triposr_engine.py",
            "sf3d_engine.py",
            "spar3d_engine.py",
            "trellis_engine.py",
            "lgm_engine.py",
        ):
            assert ".validate()" in (base / f).read_text(), f"{f} does not validate"

    def test_validate_is_not_post_init(self):
        """ENG-42 asks for an explicit validate(); __post_init__ would run on
        every construction and could break callers that build partial meshes."""
        import ast
        from pathlib import Path

        src = (Path(__file__).parent.parent / "src" / "engines" / "base.py").read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "__post_init__":
                pytest.fail("__post_init__ added; validation should be explicit")
