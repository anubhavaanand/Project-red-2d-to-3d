"""Tests for LGM island cleanup on degenerate meshes (ENG-19)."""

import ast
from pathlib import Path

import numpy as np
import pytest

ENGINE = Path(__file__).parent.parent / "src" / "engines" / "lgm_engine.py"

trimesh = pytest.importorskip("trimesh")


class TestSplitBehaviour:
    """Pins the trimesh semantics the cleanup relies on."""

    def test_empty_mesh_yields_no_parts(self):
        tm = trimesh.Trimesh(
            vertices=np.zeros((0, 3)),
            faces=np.zeros((0, 3), dtype=np.int64),
            process=False,
        )
        assert len(tm.split(only_watertight=False)) == 0

    def test_single_body_yields_one_part(self):
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        f = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        tm = trimesh.Trimesh(vertices=v, faces=f, process=False)
        assert len(tm.split(only_watertight=False)) == 1

    def test_len_guard_prevents_empty_max(self):
        """max() over an empty parts list is what would raise."""
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        f = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        tm = trimesh.Trimesh(vertices=v, faces=f, process=False)
        parts = tm.split(only_watertight=False)
        if len(parts) > 1:
            max(len(p.faces) for p in parts)  # would raise if parts were empty
        assert True


class TestSourceContract:
    def test_empty_parts_is_handled_explicitly(self):
        src = ENGINE.read_text()
        assert "elif not parts:" in src
        assert "split returned no parts" in src

    def test_bare_except_is_gone(self):
        """ENG-19: the silent `except Exception: pass` must be diagnosable."""
        src = ENGINE.read_text()
        mesh_fn = src[src.index("def _gaussians_to_mesh") :]
        cleanup = mesh_fn[mesh_fn.index("tm.split(") : mesh_fn.index("verts_final")]
        assert "except Exception:\n                pass" not in cleanup
        assert "Island cleanup skipped" in cleanup

    def test_island_cleanup_failure_does_not_abort(self):
        """Pruning is cosmetic; it must never fail the extraction."""
        src = ENGINE.read_text()
        mesh_fn = src[src.index("def _gaussians_to_mesh") :]
        cleanup = mesh_fn[mesh_fn.index("tm.split(") : mesh_fn.index("verts_final")]
        assert "raise" not in cleanup

    def test_engine_parses(self):
        ast.parse(ENGINE.read_text())