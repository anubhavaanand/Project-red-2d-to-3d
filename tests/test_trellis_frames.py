"""Tests for TRELLIS coordinate-frame alignment (ENG-25)."""

import ast
from pathlib import Path

import numpy as np
import pytest

ENGINE = Path(__file__).parent.parent / "src" / "engines" / "trellis_engine.py"


def _rx_neg90() -> np.ndarray:
    """Read _RX_NEG90 from the source (the module needs torch to import)."""
    tree = ast.parse(ENGINE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", "") == "_RX_NEG90" for t in node.targets
        ):
            return np.array(ast.literal_eval(node.value.args[0]), dtype=np.float32)
    pytest.fail("_RX_NEG90 not found")


class TestRotationMatrix:
    def test_matrix_is_rx_neg90(self):
        assert np.allclose(_rx_neg90(), [[1, 0, 0], [0, 0, 1], [0, -1, 0]])

    def test_z_up_becomes_y_up(self):
        up_zup = np.array([[0, 0, 1]], dtype=np.float32)
        assert np.allclose((up_zup @ _rx_neg90().T)[0], [0, 1, 0])


class TestOptInFlag:
    def test_flag_reads_env(self):
        src = ENGINE.read_text()
        assert 'os.environ.get("TRELLIS_ROTATE_GAUSSIANS", "0") == "1"' in src

    def test_default_is_off(self):
        """ENG-25 must not change the legacy Gaussian output by default."""
        src = ENGINE.read_text()
        assert '"TRELLIS_ROTATE_GAUSSIANS", "0"' in src

    def test_mesh_uses_the_same_matrix(self):
        src = ENGINE.read_text()
        assert "verts = verts_raw @ _RX_NEG90.T" in src

    def test_gaussians_use_it_conditionally(self):
        src = ENGINE.read_text()
        assert "if _ROTATE_GAUSSIANS_TO_MATCH_MESH:" in src
        assert "gs_pos = gs_pos @ _RX_NEG90.T" in src

    def test_flag_is_evaluated_once_at_import(self):
        """Reading env at call time would make behaviour order-dependent."""
        src = ENGINE.read_text()
        assert src.count('os.environ.get("TRELLIS_ROTATE_GAUSSIANS"') == 1


class TestRotationSemantics:
    def test_matrix_is_orthogonal(self):
        rx = _rx_neg90()
        assert np.allclose(rx @ rx.T, np.eye(3), atol=1e-6)

    def test_determinant_is_plus_one(self):
        """A rotation, not a reflection."""
        assert np.linalg.det(_rx_neg90()) == pytest.approx(1.0)


class TestSourceStructure:
    def test_no_inline_matrix_duplication(self):
        """The matrix should be defined once, not rebuilt inline."""
        src = ENGINE.read_text()
        assert src.count("[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]") == 1

    def test_engine_parses(self):
        ast.parse(ENGINE.read_text())
