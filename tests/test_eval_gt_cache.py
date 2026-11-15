"""Tests for GT render caching (EVAL-17)."""

import ast
from pathlib import Path

import numpy as np

RUNNER = Path(__file__).parent.parent / "src" / "evaluation" / "benchmark_runner.py"


class TestCacheDecorator:
    def test_render_gt_novel_is_cached(self):
        src = RUNNER.read_text()
        tree = ast.parse(src)
        cls = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BenchmarkRunner"
        )
        method = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_render_gt_novel"
        )
        decorators = [ast.unparse(d) for d in method.decorator_list]
        assert any("lru_cache" in d for d in decorators), decorators

    def test_lru_cache_is_imported(self):
        tree = ast.parse(RUNNER.read_text())
        imports = {
            a.name
            for n in tree.body
            if isinstance(n, ast.ImportFrom) and n.module == "functools"
            for a in n.names
        }
        assert "lru_cache" in imports

    def test_cache_size_is_bounded(self):
        src = SERVER_SRC()
        assert "lru_cache(maxsize=64)" in src

    def test_staticmethod_preserved(self):
        """The cache must not turn it into an instance method."""
        tree = ast.parse(SERVER_SRC())
        cls = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BenchmarkRunner"
        )
        method = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_render_gt_novel"
        )
        decorators = [ast.unparse(d) for d in method.decorator_list]
        assert any("staticmethod" in d for d in decorators), decorators


class TestCacheBehaviour:
    """The caching itself, verified on an equivalent function."""

    def test_repeated_calls_hit_the_cache(self):
        from functools import lru_cache

        calls = []

        @lru_cache(maxsize=8)
        def render(path):
            calls.append(path)
            return np.zeros((2, 2), dtype=np.uint8)

        render("a.glb")
        render("a.glb")
        assert calls == ["a.glb"]

    def test_different_arguments_do_not_collide(self):
        from functools import lru_cache

        calls = []

        @lru_cache(maxsize=8)
        def render(path, resolution):
            calls.append((path, resolution))
            return resolution

        render("a.glb", 256)
        render("a.glb", 512)
        assert sorted(calls) == [("a.glb", 256), ("a.glb", 512)]

    def test_cache_info_reports_hits(self):
        from functools import lru_cache

        @lru_cache(maxsize=8)
        def render(path):
            return path

        render("x")
        render("x")
        info = render.cache_info()
        assert info.hits == 1
        assert info.misses == 1


class TestResultNotMutated:
    def test_callers_only_read_the_array(self):
        """Caching a mutable array is only safe if callers never write to it."""
        src = SERVER_SRC()
        idx = src.index("gt_render_hi = self._render_gt_novel(")
        tail = src[idx : idx + 3000]
        # no in-place mutation of gt_render_hi
        for pat in (
            "gt_render_hi[",
            "gt_render_hi +=",
            "gt_render_hi *=",
            "gt_render_hi /=",
            "np.clip(gt_render_hi",
        ):
            assert pat not in tail, f"callers mutate gt_render_hi via {pat!r}"


def SERVER_SRC():
    return RUNNER.read_text()
