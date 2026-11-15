"""Tests for the API's thread-safe engine cache (API-07).

The real engines load multi-GB models, so these tests re-implement the exact
double-checked-locking structure from server.py against a counting fake, and
assert the source keeps the lock.
"""

import ast
import threading
import time
from pathlib import Path

import pytest

SERVER_PY = Path(__file__).parent.parent / "src" / "api" / "server.py"


class _FakeEngine:
    instances = 0
    _gate = threading.Lock()

    def __init__(self):
        with _FakeEngine._gate:
            _FakeEngine.instances += 1
        time.sleep(0.05)  # simulate a slow model load


def _build_get_engine():
    """Extract the cache machinery from server.py, retargeted at _FakeEngine."""
    src = SERVER_PY.read_text()
    lines = src.splitlines()
    tree = ast.parse(src)
    names = {"_lock_for", "get_engine"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    body = "\n".join("\n".join(lines[n.lineno - 1 : n.end_lineno]) for n in nodes)

    ns = {
        "threading": threading,
        "Dict": dict,
        "EngineType": type(
            "E", (), {n: n for n in ("TRIPOSR", "LGM", "TRELLIS", "SPAR3D", "SF3D")}
        ),
        "AbstractEngine": object,
        "TriposrEngine": _FakeEngine,
        "LGMEngine": _FakeEngine,
        "TrellisEngine": _FakeEngine,
        "Spar3dEngine": _FakeEngine,
        "SF3DEngine": _FakeEngine,
        "_engines": {},
        "_engine_locks": {},
        "_engine_locks_guard": threading.Lock(),
    }
    exec(compile(body, str(SERVER_PY), "exec"), ns)
    return ns


class TestGetEngineThreadSafety:
    def test_concurrent_first_requests_build_one_engine(self):
        ns = _build_get_engine()
        get_engine = ns["get_engine"]
        EngineType = ns["EngineType"]
        _FakeEngine.instances = 0

        results = []
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()  # maximise contention on the first request
            results.append(get_engine(EngineType.TRIPOSR))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert _FakeEngine.instances == 1, "engine was constructed more than once"
        assert len({id(r) for r in results}) == 1, "callers got different instances"

    def test_cached_engine_is_returned_without_reconstruction(self):
        ns = _build_get_engine()
        get_engine = ns["get_engine"]
        EngineType = ns["EngineType"]
        _FakeEngine.instances = 0

        first = get_engine(EngineType.TRIPOSR)
        second = get_engine(EngineType.TRIPOSR)
        assert first is second
        assert _FakeEngine.instances == 1

    def test_different_engines_are_built_separately(self):
        ns = _build_get_engine()
        get_engine = ns["get_engine"]
        EngineType = ns["EngineType"]
        _FakeEngine.instances = 0

        a = get_engine(EngineType.TRIPOSR)
        b = get_engine(EngineType.LGM)
        assert a is not b
        assert _FakeEngine.instances == 2

    def test_lock_is_per_engine(self):
        ns = _build_get_engine()
        EngineType = ns["EngineType"]
        assert ns["_lock_for"](EngineType.TRIPOSR) is not ns["_lock_for"](EngineType.LGM)
        # same engine returns the same lock
        assert ns["_lock_for"](EngineType.TRIPOSR) is ns["_lock_for"](EngineType.TRIPOSR)


class TestSourceStructure:
    def test_get_engine_uses_double_checked_locking(self):
        src = SERVER_PY.read_text()
        assert "with _lock_for(engine_type):" in src
        # re-check inside the lock
        assert "engine = _engines.get(engine_type)" in src

    def test_unknown_engine_still_raises(self):
        ns = _build_get_engine()
        with pytest.raises(ValueError):
            ns["get_engine"]("NOPE")

    def test_bg_remover_init_is_guarded(self):
        src = SERVER_PY.read_text()
        assert "_bg_lock" in src
        assert "with _bg_lock:" in src
