"""Tests for lazy engine imports in app.py (APP-15)."""

import ast
from pathlib import Path

APP_PY = Path(__file__).parent.parent / "app.py"


def _app_tree():
    return ast.parse(APP_PY.read_text())


def _top_level_imports(tree):
    """Return the set of module names imported at module scope."""
    modules = set()
    for node in tree.body:  # only direct children = module scope
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


class TestEngineImportsAreLazy:
    def test_no_engine_module_is_imported_at_module_scope(self):
        """APP-15: engine modules must not be imported at module import."""
        modules = _top_level_imports(_app_tree())
        engine_imports = {m for m in modules if m.startswith("src.engines.")}
        assert not engine_imports, f"eager engine imports: {sorted(engine_imports)}"

    def test_export_modules_are_still_eager(self):
        """glb/ply exporters are light and used by every response; keep them eager."""
        modules = _top_level_imports(_app_tree())
        assert "src.export.glb_exporter" in modules
        assert "src.export.ply_exporter" in modules

    def test_engine_registry_covers_every_engine(self):
        """Every engine the UI offers must be in the lazy registry."""
        tree = _app_tree()
        registry = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "_ENGINE_MODULES":
                        registry = ast.literal_eval(node.value)
        assert registry is not None, "_ENGINE_MODULES not found"
        assert set(registry) == {"TripoSR", "TRELLIS", "LGM", "SPAR3D"}
        for engine, (module_path, class_name) in registry.items():
            assert module_path.startswith("src.engines."), engine
            assert class_name, engine

    def test_get_engine_loads_the_class_lazily(self):
        """_get_engine must call the loader rather than use a top-level name."""
        src = APP_PY.read_text()
        assert "_load_engine_class(name)" in src

    def test_module_getattr_resolves_engine_classes(self):
        """PEP 562 __getattr__ keeps `app.TriposrEngine` working for callers."""
        tree = _app_tree()
        has_getattr = any(
            isinstance(node, ast.FunctionDef) and node.name == "__getattr__" for node in tree.body
        )
        assert has_getattr, "app.py must define __getattr__ to resolve engine classes lazily"

    def test_importlib_is_imported(self):
        tree = _app_tree()
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert "importlib" in imported


class TestNoRegressionOnEngineUsage:
    def test_get_engine_still_constructs_with_device(self):
        src = APP_PY.read_text()
        assert "engine_cls(device=DEVICE)" in src

    def test_trellis_keeps_its_specific_kwargs(self):
        src = APP_PY.read_text()
        assert 'output_format="mesh"' in src
        assert "slat_steps=8" in src

    def test_engine_is_cached(self):
        """APP-09 depends on _engines caching; _get_engine must still populate it."""
        src = APP_PY.read_text()
        assert "_engines[name] = e" in src
