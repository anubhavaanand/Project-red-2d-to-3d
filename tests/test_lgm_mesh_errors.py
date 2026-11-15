"""Behavioural tests for LGM mesh extraction error handling (ENG-20).

These exercise the real code paths with a stubbed torch so a regression in the
error handling is caught even where torch is not installed.
"""

import sys
import types

import numpy as np
import pytest


@pytest.fixture(scope="module")
def lgm():
    """Import lgm_engine with a minimal torch stub (torch is not installed here)."""
    if "torch" not in sys.modules:
        t = types.ModuleType("torch")
        t.__version__ = "0.0-stub"
        t.float32, t.float16, t.int64 = "float32", "float16", "int64"
        t.device = lambda x: x

        class _NoCtx:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        t.no_grad = _NoCtx
        t.cuda = types.SimpleNamespace(is_available=lambda: False)
        t.nn = types.ModuleType("torch.nn")
        t.nn.functional = types.ModuleType("torch.nn.functional")
        sys.modules["torch"] = t
        sys.modules["torch.nn"] = t.nn
        sys.modules["torch.nn.functional"] = t.nn.functional

    for name in ("safetensors", "safetensors.torch", "huggingface_hub"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["safetensors.torch"].load_file = lambda *a, **k: {}
    sys.modules["huggingface_hub"].hf_hub_download = lambda *a, **k: ""
    sys.modules["huggingface_hub"].snapshot_download = lambda *a, **k: ""

    from src.engines import lgm_engine

    return lgm_engine


def _cloud(opa_values, n_colour=3):
    lgm = pytest.importorskip("src.engines.lgm_engine")
    n = len(opa_values)
    return lgm.GaussianCloud(
        positions=np.zeros((n, 3), dtype=np.float32),
        scales=np.ones((n, 3), dtype=np.float32),
        rotations=np.zeros((n, 4), dtype=np.float32),
        opacities=np.asarray(opa_values, dtype=np.float32).reshape(-1, 1),
        sh_coeffs=np.zeros((n, n_colour), dtype=np.float32),
    )


class TestExceptionIsReal:
    def test_is_a_runtime_error(self, lgm):
        assert issubclass(lgm.LgmMeshExtractionError, RuntimeError)

    def test_can_be_raised_and_caught(self, lgm):
        with pytest.raises(lgm.LgmMeshExtractionError, match="boom"):
            raise lgm.LgmMeshExtractionError("boom")


class TestNoForegroundRaises:
    """ENG-20: 'nothing to mesh' must raise, not return None."""

    def test_all_opacities_below_every_threshold_raises(self, lgm):
        engine = lgm.LGMEngine(device="cpu")
        engine.opacity_thresholds = (0.10, 0.08, 0.03)
        engine.min_foreground_points = 200

        # every opacity is 0.0 -> the all-points fallback fires, but there is no
        # density, so marching cubes gets an empty grid
        cloud = _cloud(np.zeros(500, dtype=np.float32))
        with pytest.raises(lgm.LgmMeshExtractionError):
            engine._gaussians_to_mesh(cloud, resolution=16)

    def test_raises_rather_than_returning_none(self, lgm):
        """The regression: the old code returned None here."""
        engine = lgm.LGMEngine(device="cpu")
        cloud = _cloud(np.zeros(500, dtype=np.float32))
        result = None
        try:
            result = engine._gaussians_to_mesh(cloud, resolution=16)
        except lgm.LgmMeshExtractionError:
            result = "raised"
        assert result == "raised", "did not raise and did not return a mesh"


class TestOuterExceptChains:
    def test_unexpected_failure_is_wrapped(self, lgm, monkeypatch):
        """Any unexpected error becomes LgmMeshExtractionError with a cause."""
        engine = lgm.LGMEngine(device="cpu")
        cloud = _cloud(np.full(500, 0.9, dtype=np.float32))

        def _boom(*a, **k):
            raise ValueError("synthetic failure")

        # _select_foreground is a module function, so patch it there
        monkeypatch.setattr(lgm, "_select_foreground", _boom)

        with pytest.raises(lgm.LgmMeshExtractionError) as exc:
            engine._gaussians_to_mesh(cloud, resolution=16)
        # the original exception must be preserved as __cause__
        assert exc.value.__cause__ is not None
        assert "synthetic failure" in str(exc.value.__cause__)


class TestSelectForegroundHelper:
    """ENG-17's helper is the one pure function worth testing directly."""

    def test_picks_first_sufficient_threshold(self, lgm):
        opa = np.full(1000, 0.5, dtype=np.float32)
        mask, thr = lgm._select_foreground(opa)
        assert thr == pytest.approx(0.10)
        assert mask.all()

    def test_falls_through_to_lower_thresholds(self, lgm):
        opa = np.full(1000, 0.09, dtype=np.float32)
        _mask, thr = lgm._select_foreground(opa)
        assert thr == pytest.approx(0.08)

    def test_all_points_fallback_reports_none(self, lgm):
        opa = np.full(1000, 0.01, dtype=np.float32)
        mask, thr = lgm._select_foreground(opa)
        assert thr is None
        assert mask.all()

    def test_empty_input_is_handled(self, lgm):
        mask, thr = lgm._select_foreground(np.zeros(0, dtype=np.float32))
        assert thr is None
        assert len(mask) == 0
