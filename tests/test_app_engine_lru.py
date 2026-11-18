"""Tests for the app's engine LRU cache (APP-09)."""

import ast
from collections import OrderedDict
from pathlib import Path

APP_PY = Path(__file__).parent.parent / "app.py"


def _app_tree():
    return ast.parse(APP_PY.read_text())


class _FakeEngine:
    """Stand-in shaped like AbstractEngine."""

    def __init__(self, name):
        self.name = name
        self.loaded = True
        self.unloaded = False

    def unload_model(self):
        self.unloaded = True
        self.loaded = False


class _ExplodingEngine(_FakeEngine):
    def unload_model(self):
        raise RuntimeError("cuda oom")


def _make_evict(engines, cap):
    """Compile _evict_lru_engine against a supplied engine cache and cap.

    app.py cannot be imported without gradio, so the function is exec'd with an
    explicit namespace instead of monkeypatching a module attribute.
    """
    tree = _app_tree()
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_evict_lru_engine"
    )
    ns = {"_engines": engines, "MAX_LOADED_ENGINES": cap}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(APP_PY), "exec"), ns)
    return ns["_evict_lru_engine"]


class TestEviction:
    def test_no_eviction_below_the_cap(self):
        engine = _FakeEngine("a")
        engines = OrderedDict(a=engine)
        evict = _make_evict(engines, cap=2)
        assert evict() is None
        assert list(engines) == ["a"]
        assert not engine.unloaded

    def test_evicts_least_recently_used(self):
        first, second = _FakeEngine("first"), _FakeEngine("second")
        engines = OrderedDict([("first", first), ("second", second)])
        evict = _make_evict(engines, cap=2)
        assert evict() == "first"
        assert first.unloaded
        assert list(engines) == ["second"]
        assert not second.unloaded

    def test_eviction_calls_unload_model(self):
        engine = _FakeEngine("only")
        engines = OrderedDict([("only", engine)])
        evict = _make_evict(engines, cap=1)
        evict()
        assert engine.unloaded is True

    def test_unload_failure_does_not_propagate(self):
        engines = OrderedDict([("boom", _ExplodingEngine("boom"))])
        evict = _make_evict(engines, cap=1)
        # Must not raise, and the entry must still be dropped so the caller can
        # proceed rather than looping forever on a full cache.
        assert evict() == "boom"
        assert "boom" not in engines

    def test_returns_none_on_an_empty_cache(self):
        evict = _make_evict(OrderedDict(), cap=1)
        assert evict() is None


class TestWiring:
    def test_engines_is_an_ordered_dict(self):
        src = APP_PY.read_text()
        assert "OrderedDict()" in src, "engine cache must be an OrderedDict for LRU"

    def test_eviction_runs_before_loading_a_new_engine(self):
        src = APP_PY.read_text()
        assert src.index("_evict_lru_engine()") < src.index("engine_cls = _load_engine_class(name)")

    def test_recency_is_refreshed_on_access(self):
        """A cache hit must move the engine to the most-recently-used end."""
        src = APP_PY.read_text()
        assert "_engines.move_to_end(name)" in src

    def test_cap_is_declared(self):
        assert "MAX_LOADED_ENGINES" in APP_PY.read_text()

    def test_cap_is_smaller_than_the_engine_count(self):
        """The cap must be low enough to matter, but high enough to avoid thrash."""
        cap = None
        for node in ast.walk(_app_tree()):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "MAX_LOADED_ENGINES":
                        cap = ast.literal_eval(node.value)
        assert cap is not None
        assert 1 <= cap < 4, f"cap={cap} should be 1..3 for four engines"
