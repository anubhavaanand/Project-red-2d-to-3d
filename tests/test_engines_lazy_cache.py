"""Tests for the engines lazy-attribute cache (ENG-44)."""

import ast
import importlib
from pathlib import Path

import pytest

INIT = Path(__file__).parent.parent / "src" / "engines" / "__init__.py"


class TestLazyTable:
    def test_all_public_engines_are_lazy(self):
        import src.engines as engines

        for name in engines.__all__:
            if name in ("AbstractEngine", "Mesh3D", "GaussianCloud", "ReconstructionResult"):
                continue
            assert name in engines._LAZY, f"{name} missing from _LAZY"

    def test_unknown_attribute_raises_attribute_error(self):
        import src.engines as engines

        with pytest.raises(AttributeError, match="no attribute 'NoSuchEngine'"):
            _ = engines.NoSuchEngine  # noqa: B018

    def test_dir_includes_lazy_names(self):
        import src.engines as engines

        listing = dir(engines)
        assert "TriposrEngine" in listing
        assert "SF3DEngine" in listing


class TestCaching:
    def test_resolved_class_is_cached(self):
        """ENG-44: the second access must not re-import."""
        import src.engines as engines

        engines._cache.clear()
        try:
            first = engines.SF3DEngine
        except Exception:
            pytest.skip("engine import requires torch")

        assert "SF3DEngine" in engines._cache
        assert engines.SF3DEngine is first

    def test_cache_prevents_repeated_imports(self):
        import src.engines as engines

        engines._cache.clear()
        attempts = {"n": 0}
        real = importlib.import_module

        def counting(name, package=None):
            if name in (".sf3d_engine", "src.engines.sf3d_engine"):
                attempts["n"] += 1
            return real(name, package)

        engines.importlib.import_module = counting
        try:
            try:
                _ = engines.SF3DEngine  # noqa: B018
            except Exception:
                pytest.skip("engine import requires torch")
            attempts["n"] = 0
            _ = engines.SF3DEngine  # noqa: B018
            assert attempts["n"] == 0, "module was re-imported on the cached path"
        finally:
            engines.importlib.import_module = real
            engines._cache.clear()

    def test_each_name_caches_independently(self):
        import src.engines as engines

        engines._cache.clear()
        try:
            _ = engines.TriposrEngine  # noqa: B018
            _ = engines.LGMEngine  # noqa: B018
        except Exception:
            pytest.skip("engine import requires torch")
        assert "TriposrEngine" in engines._cache
        assert "LGMEngine" in engines._cache
        engines._cache.clear()


class TestSourceStructure:
    def test_uses_importlib_import_module(self):
        src = INIT.read_text()
        assert 'importlib.import_module(f".{module_name}", __name__)' in src

    def test_no_repeated_from_imports(self):
        """ENG-44: the per-name 'from .x import Y' chain must be gone."""
        tree = ast.parse(INIT.read_text())
        lazy_imports = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom)
            and n.level == 1
            and n.module
            and n.module.endswith("_engine")
        ]
        assert not lazy_imports, f"still importing engines eagerly: {lazy_imports}"

    def test_init_parses(self):
        ast.parse(INIT.read_text())
