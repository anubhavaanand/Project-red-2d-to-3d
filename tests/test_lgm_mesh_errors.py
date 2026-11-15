"""Tests for LGM mesh extraction error handling (ENG-20)."""

import ast
from pathlib import Path

ENGINE = Path(__file__).parent.parent / "src" / "engines" / "lgm_engine.py"


class TestTypedException:
    def test_exception_is_defined(self):
        src = ENGINE.read_text()
        assert "class LgmMeshExtractionError(RuntimeError):" in src

    def test_predict_records_the_error(self):
        """A mesh failure must surface in metadata, not vanish."""
        src = ENGINE.read_text()
        assert '"mesh_error": mesh_error' in src

    def test_predict_does_not_swallow_the_exception(self):
        src = ENGINE.read_text()
        assert "except LgmMeshExtractionError as e:" in src
        assert "mesh_error = str(e)" in src

    def test_gaussians_still_returned_on_mesh_failure(self):
        """The fallback path must keep gaussians available."""
        src = ENGINE.read_text()
        predict = src[src.index("def predict(") : src.index("def _gaussians_to_mesh")]
        assert "gaussians=gaussians_out" in predict

    def test_no_bare_return_none_on_failure(self):
        """ENG-20: the silent 'print then return None' pattern must be gone."""
        src = ENGINE.read_text()
        mesh_fn = src[src.index("def _gaussians_to_mesh") :]
        assert "return None" not in mesh_fn


class TestFailurePathsRaise:
    def test_no_foreground_raises(self):
        src = ENGINE.read_text()
        assert "nothing to mesh" in src
        assert "LgmMeshExtractionError(" in src

    def test_marching_cubes_failure_is_chained(self):
        src = ENGINE.read_text()
        assert "raise LgmMeshExtractionError(f\"Marching cubes failed: {e}\") from e" in src

    def test_empty_mesh_raises(self):
        src = ENGINE.read_text()
        assert "Marching cubes produced an empty mesh" in src


class TestSourceStructure:
    def test_engine_parses(self):
        ast.parse(ENGINE.read_text())

    def test_outer_except_raises_instead_of_returning_none(self):
        src = ENGINE.read_text()
        mesh_fn = src[src.index("def _gaussians_to_mesh") :]
        assert "raise LgmMeshExtractionError(" in mesh_fn
        assert "Mesh extraction error" not in mesh_fn  # old print message gone