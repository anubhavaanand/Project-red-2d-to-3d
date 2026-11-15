"""Tests for deferred heavy imports in the engine wrappers (ENG-45)."""

import ast
from pathlib import Path

ENGINES = Path(__file__).parent.parent / "src" / "engines"

# (file, module, symbol) — heavy things that must not be imported at module scope
DEFERRED = [
    ("lgm_engine.py", "safetensors.torch", "load_file"),
    ("lgm_engine.py", "huggingface_hub", "hf_hub_download"),
    ("lgm_engine.py", "scipy.spatial", "cKDTree"),
    ("spar3d_engine.py", "scipy.spatial", "cKDTree"),
]


def _module_level_imports(file: str) -> set:
    """Return modules imported at module scope (not inside a function)."""
    tree = ast.parse((ENGINES / file).read_text())
    found = set()

    def walk(body):
        for node in body:
            if isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    found.add(alias.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue  # nested scope
            elif isinstance(node, (ast.If, ast.Try, ast.With)):
                walk(node.body)
                walk(getattr(node, "orelse", []))
                walk(getattr(node, "finalbody", []))

    walk(tree.body)
    return found


class TestModuleLevelImports:
    def test_heavy_modules_are_not_imported_at_module_scope(self):
        for file, module, _sym in DEFERRED:
            top = _module_level_imports(file)
            assert module not in top, f"{file} imports {module} at module scope"

    def test_heavy_symbols_appear_somewhere(self):
        """They must still be imported — just later."""
        for file, module, sym in DEFERRED:
            src = (ENGINES / file).read_text()
            assert f"from {module} import {sym}" in src, f"{sym} import missing from {file}"


class TestImportPlacement:
    def test_lgm_defers_into_load_model(self):
        src = (ENGINES / "lgm_engine.py").read_text()
        load = src[src.index("def load_model") : src.index("def predict")]
        assert "from safetensors.torch import load_file" in load
        assert "from huggingface_hub import hf_hub_download" in load

    def test_spar3d_defers_scipy_into_the_colour_helper(self):
        src = (ENGINES / "spar3d_engine.py").read_text()
        helper = src[
            src.index("def _extract_spar3d_vertex_colors") : src.index("def predict")
        ]
        assert "from scipy.spatial import cKDTree" in helper


class TestTorchIsStillEager:
    def test_torch_stays_at_module_scope(self):
        """ENG-45 must not make torch lazy — the @torch.no_grad() decorators
        are evaluated at class-definition time, so a lazy torch would break."""
        for f in ("lgm_engine.py", "trellis_engine.py", "spar3d_engine.py"):
            tree = ast.parse((ENGINES / f).read_text())
            decorators = [
                ast.unparse(d)
                for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef)
                for item in n.body
                if isinstance(n, ast.ClassDef)
                for n2 in [item]
                if isinstance(n2, ast.FunctionDef)
                for d in n2.decorator_list
            ]
            if any("torch" in d for d in decorators):
                assert "torch" in _module_level_imports(f)


class TestStartupCost:
    def test_importing_engines_module_is_cheap(self):
        """The package __init__ must not pull in the engine modules."""
        src = (ENGINES / "__init__.py").read_text()
        for f, _m, _s in DEFERRED:
            pass  # covered above; this test guards the package import path
        assert "from .lgm_engine import" not in src
        assert "from .spar3d_engine import" not in src
        assert "from .trellis_engine import" not in src

    def test_engines_init_imports_only_base(self):
        tree = ast.parse((ENGINES / "__init__.py").read_text())
        eager = [
            n.module
            for n in tree.body
            if isinstance(n, ast.ImportFrom) and n.level == 1
        ]
        assert eager == ["base"], f"unexpected eager imports: {eager}"