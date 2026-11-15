"""Tests for TRELLIS output payload handling (ENG-22)."""

import ast
from pathlib import Path

ENGINE = Path(__file__).parent.parent / "src" / "engines" / "trellis_engine.py"


def _first_of():
    """Compile _first_of without importing the engine (it needs torch)."""
    tree = ast.parse(ENGINE.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_first_of")
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(ENGINE), "exec"), ns)
    return ns["_first_of"]


class TestFirstOf:
    def test_list_returns_first(self):
        assert _first_of()(["a", "b"], "mesh") == "a"

    def test_tuple_returns_first(self):
        assert _first_of()(("a", "b"), "mesh") == "a"

    def test_empty_list_returns_none(self):
        assert _first_of()([], "mesh") is None

    def test_none_returns_none(self):
        assert _first_of()(None, "mesh") is None

    def test_bare_object_returned_as_is(self):
        sentinel = object()
        assert _first_of()(sentinel, "mesh") is sentinel

    def test_single_element_list(self):
        assert _first_of()(["only"], "mesh") == "only"


class TestSourceContract:
    def test_mesh_key_is_not_indexed_directly(self):
        """ENG-22: outputs["mesh"][0] must be gone from real code."""
        tree = ast.parse(ENGINE.read_text())
        offenders = []
        for node in ast.walk(tree):
            # a Subscript whose value is a Subscript on outputs["mesh"]
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Subscript):
                inner = node.value
                if (
                    isinstance(inner.value, ast.Name)
                    and inner.value.id == "outputs"
                    and isinstance(inner.slice, ast.Constant)
                    and inner.slice.value == "mesh"
                ):
                    offenders.append(node.lineno)
        assert not offenders, f'outputs["mesh"][...] indexed at lines {offenders}'

    def test_uses_get_with_helper(self):
        src = ENGINE.read_text()
        assert '_first_of(outputs.get("mesh"), "mesh")' in src

    def test_missing_mesh_raises_typed_error(self):
        src = ENGINE.read_text()
        assert "class TrellisOutputError(RuntimeError):" in src
        assert "raise TrellisOutputError(" in src

    def test_error_lists_available_keys(self):
        """A diagnostic should say what *was* returned."""
        src = ENGINE.read_text()
        assert "sorted(outputs)" in src

    def test_gaussian_path_still_guarded(self):
        src = ENGINE.read_text()
        assert "if out_gs_color is not None and len(out_gs_color) > 0" in src

    def test_engine_parses(self):
        ast.parse(ENGINE.read_text())
