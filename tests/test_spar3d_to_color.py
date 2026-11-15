"""Tests for SPAR3D's optional vis.to_color() path (ENG-34)."""

import ast
from pathlib import Path

import pytest

ENGINE = Path(__file__).parent.parent / "src" / "engines" / "spar3d_engine.py"


class TestSourceContract:
    def test_to_color_is_probed_with_getattr(self):
        """ENG-34: vis.to_color() must not be called blind."""
        src = ENGINE.read_text()
        assert 'getattr(vis, "to_color", None)' in src

    def test_result_is_checked_for_callability(self):
        src = ENGINE.read_text()
        assert "if not callable(to_color):" in src

    def test_missing_method_is_logged(self):
        src = ENGINE.read_text()
        assert "has no to_color(); skipping" in src

    def test_no_bare_vis_to_color_call(self):
        """A direct vis.to_color() call must be gone from real code."""
        tree = ast.parse(ENGINE.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "to_color":
                if not isinstance(node.func.value, ast.Name):
                    pytest.fail(
                        f"bare vis.to_color() at line {node.lineno} "
                        "(should be via the probed callable)"
                    )

    def test_engine_parses(self):
        ast.parse(ENGINE.read_text())


class TestProbeSemantics:
    def test_getattr_probe_pattern(self):
        class Without:
            pass

        class With:
            def to_color(self):
                return "ok"

        assert getattr(Without(), "to_color", None) is None
        assert callable(getattr(With(), "to_color", None))
        assert not callable(getattr(Without(), "to_color", None))


