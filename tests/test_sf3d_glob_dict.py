"""Tests for SF3D's glob_dict handling (ENG-40)."""

import ast
from pathlib import Path

ENGINE = Path(__file__).parent.parent / "src" / "engines" / "sf3d_engine.py"


def _summarise():
    tree = ast.parse(ENGINE.read_text())
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_summarize_glob_dict"
    )
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(ENGINE), "exec"), ns)
    return ns["_summarize_glob_dict"]


class TestSummarizeGlobDict:
    def test_scalars_are_kept(self):
        out = _summarise()({"steps": 50, "cfg": 7.5, "name": "x", "flag": True})
        assert out == {"steps": 50, "cfg": 7.5, "name": "x", "flag": True}

    def test_short_lists_are_kept(self):
        out = _summarise()({"sizes": [1, 2, 3]})
        assert out["sizes"] == [1, 2, 3]

    def test_long_lists_are_dropped(self):
        out = _summarise()({"big": list(range(100))})
        assert "big" not in out

    def test_tensors_and_objects_are_dropped(self):
        out = _summarise()({"tensor": object(), "nested": {"a": 1}})
        assert out == {}

    def test_none_value_is_kept(self):
        out = _summarise()({"maybe": None})
        assert out == {"maybe": None}

    def test_non_dict_returns_empty(self):
        assert _summarise()(None) == {}
        assert _summarise()([1, 2]) == {}

    def test_keys_are_stringified(self):
        out = _summarise()({1: "a"})
        assert out == {"1": "a"}

    def test_result_is_json_safe(self):
        import json

        out = _summarise()({"steps": 50, "sizes": [1, 2], "name": "x"})
        json.dumps(out)  # must not raise


class TestSourceContract:
    def test_glob_dict_is_recorded(self):
        src = ENGINE.read_text()
        assert '"glob_dict": _summarize_glob_dict(glob_dict)' in src

    def test_helper_defined(self):
        src = ENGINE.read_text()
        assert "def _summarize_glob_dict(glob_dict)" in src

    def test_both_unpack_sites_present(self):
        src = ENGINE.read_text()
        assert src.count("mesh_out, glob_dict = self.model.run_image(") == 2

    def test_engine_parses(self):
        ast.parse(ENGINE.read_text())
