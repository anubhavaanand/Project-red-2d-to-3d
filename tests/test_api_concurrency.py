"""Tests for API inference concurrency limiting and timeout (API-09)."""

import ast
import asyncio
import threading
import time
from pathlib import Path

import pytest

SERVER_PY = Path(__file__).parent.parent / "src" / "api" / "server.py"


class _HTTPException(Exception):
    def __init__(self, status_code, detail):
        self.status_code = status_code
        self.detail = detail


async def _run_in_threadpool(fn):
    return await asyncio.to_thread(fn)


def _build(semaphore_size=2, timeout_s=0.2):
    """Compile _run_inference from server.py with the given limits."""
    src = SERVER_PY.read_text()
    lines = src.splitlines()
    tree = ast.parse(src)
    fn = next(
        n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "_run_inference"
    )
    body = "\n".join(lines[fn.lineno - 1 : fn.end_lineno])
    ns = {
        "asyncio": asyncio,
        "run_in_threadpool": _run_in_threadpool,
        "HTTPException": _HTTPException,
        "logger": type("L", (), {"error": staticmethod(lambda *a, **k: None)})(),
        "_inference_slots": asyncio.Semaphore(semaphore_size),
        "INFERENCE_TIMEOUT_S": timeout_s,
    }
    exec(compile(body, str(SERVER_PY), "exec"), ns)
    return ns["_run_inference"]


class TestInferenceConcurrency:
    def test_concurrency_is_capped(self):
        async def scenario():
            # generous timeout: this test measures concurrency, not timeouts,
            # and the 0.2s default is too tight on a loaded CI runner
            run = _build(semaphore_size=2, timeout_s=30.0)
            active = 0
            peak = 0
            lock = threading.Lock()

            def slow():
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep(0.1)
                with lock:
                    active -= 1
                return "ok"

            await asyncio.gather(*[run(slow) for _ in range(6)])
            return peak

        assert asyncio.run(scenario()) <= 2

    def test_results_are_returned_in_order(self):
        async def scenario():
            run = _build(semaphore_size=3)
            return await asyncio.gather(*[run(lambda i=i: i) for i in range(6)])

        assert asyncio.run(scenario()) == [0, 1, 2, 3, 4, 5]

    def test_single_slot_serialises_everything(self):
        async def scenario():
            run = _build(semaphore_size=1, timeout_s=30.0)
            active = 0
            peak = 0
            lock = threading.Lock()

            def slow():
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep(0.05)
                with lock:
                    active -= 1

            await asyncio.gather(*[run(slow) for _ in range(4)])
            return peak

        assert asyncio.run(scenario()) == 1


class TestInferenceTimeout:
    def test_slow_call_raises_504(self):
        async def scenario():
            run = _build(timeout_s=0.05)
            with pytest.raises(_HTTPException) as exc:
                await run(lambda: time.sleep(1.0))
            return exc.value

        err = asyncio.run(scenario())
        assert err.status_code == 504
        assert "timed out" in err.detail

    def test_fast_call_is_not_timed_out(self):
        async def scenario():
            run = _build(timeout_s=5.0)
            return await run(lambda: "done")

        assert asyncio.run(scenario()) == "done"


class TestSourceStructure:
    def test_limits_are_configurable_by_env(self):
        src = SERVER_PY.read_text()
        assert "MAX_CONCURRENT_INFERENCE" in src
        assert "INFERENCE_TIMEOUT_S" in src
        assert "os.environ.get" in src

    def test_predict_uses_the_bounded_runner(self):
        src = SERVER_PY.read_text()
        assert "result = await _run_inference(call)" in src
        # the raw unbounded call must no longer be used directly in predict
        assert "await run_in_threadpool(engine.predict" not in src
