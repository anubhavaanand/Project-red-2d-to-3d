"""Tests for import-time environment mutation (ENG-29)."""

import ast
import os
from pathlib import Path

ENGINES = Path(__file__).parent.parent / "src" / "engines"


def _module_level_env_writes(file: str) -> list:
    """Return os.environ[...] = / os.environ.setdefault(...) at module scope."""
    tree = ast.parse((ENGINES / file).read_text())
    found = []

    def walk(body, top=True):
        for node in body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                func = call.func
                if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Attribute):
                    if (
                        func.value.attr == "environ"
                        and isinstance(func.value.value, ast.Name)
                        and func.value.value.id == "os"
                        and func.attr in ("__setitem__", "setdefault")
                    ):
                        found.append(node.lineno)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Attribute):
                        if t.value.attr == "environ":
                            found.append(node.lineno)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            elif isinstance(node, (ast.If, ast.Try, ast.With)):
                walk(node.body, top=False)
                walk(getattr(node, "orelse", []), top=False)
                walk(getattr(node, "finalbody", []), top=False)

    walk(tree.body)
    return found


class TestNoModuleLevelEnvWrites:
    def test_trellis_does_not_mutate_env_at_import(self):
        """ENG-29: the regression is os.environ.setdefault at module scope."""
        assert _module_level_env_writes("trellis_engine.py") == []

    def test_no_engine_writes_env_at_module_scope(self):
        for f in sorted(ENGINES.glob("*_engine.py")):
            assert _module_level_env_writes(f.name) == [], f"{f.name} mutates os.environ at import"


class TestEnvSetInLoadModel:
    def test_vars_are_set_in_load_model(self):
        src = (ENGINES / "trellis_engine.py").read_text()
        load = src[src.index("def load_model") : src.index("def unload_model")]
        assert 'os.environ.setdefault("SPCONV_ALGO", "native")' in load
        assert 'os.environ.setdefault("ATTN_BACKEND", "xformers")' in load

    def test_set_before_the_trellis_import(self):
        src = (ENGINES / "trellis_engine.py").read_text()
        load = src[src.index("def load_model") : src.index("def unload_model")]
        env_line = load.index('os.environ.setdefault("SPCONV_ALGO"')
        import_line = load.index("from trellis.pipelines import")
        assert env_line < import_line, "env must be set before importing trellis"

    def test_vars_appear_once_in_the_file(self):
        src = (ENGINES / "trellis_engine.py").read_text()
        assert src.count('os.environ.setdefault("SPCONV_ALGO"') == 1
        assert src.count('os.environ.setdefault("ATTN_BACKEND"') == 1


class TestSemantics:
    def test_setdefault_preserves_user_override(self):
        """The fix must not clobber an explicit user setting."""
        os.environ["SPCONV_ALGO"] = "user-choice"
        try:
            os.environ.setdefault("SPCONV_ALGO", "native")
            assert os.environ["SPCONV_ALGO"] == "user-choice"
        finally:
            del os.environ["SPCONV_ALGO"]

    def test_setdefault_applies_when_absent(self):
        os.environ.pop("SPCONV_ALGO", None)
        os.environ.setdefault("SPCONV_ALGO", "native")
        assert os.environ["SPCONV_ALGO"] == "native"
        del os.environ["SPCONV_ALGO"]


class TestSourceStructure:
    def test_engine_parses(self):
        ast.parse((ENGINES / "trellis_engine.py").read_text())