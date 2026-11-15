"""Behavioural tests for SF3D's configurable low_vram_mode (ENG-37)."""

import inspect
import sys
import types

import pytest


@pytest.fixture(scope="module")
def sf3d():
    """Import sf3d_engine with a minimal torch stub."""
    t = types.ModuleType("torch")
    t.__version__ = "0.0-stub"
    t.float32, t.int64 = "float32", "int64"
    t.device = lambda x: x
    t.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        empty_cache=lambda: None,
        device=lambda d: _NoCtx(),
    )

    class _NoCtx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def __call__(self, fn):
            return fn

    t.no_grad = lambda: _NoCtx()
    t.nn = types.ModuleType("torch.nn")
    t.nn.functional = types.ModuleType("torch.nn.functional")
    sys.modules["torch"] = t
    sys.modules["torch.nn"] = t.nn
    sys.modules["torch.nn.functional"] = t.nn.functional

    from src.engines import sf3d_engine

    return sf3d_engine


class TestDefaults:
    def test_low_vram_mode_defaults_to_true(self, sf3d):
        """ENG-37 must preserve the previous T4-safe default."""
        assert sf3d.SF3DEngine(device="cpu").low_vram_mode is True

    def test_can_be_disabled(self, sf3d):
        assert sf3d.SF3DEngine(device="cpu", low_vram_mode=False).low_vram_mode is False

    def test_can_be_enabled_explicitly(self, sf3d):
        assert sf3d.SF3DEngine(device="cpu", low_vram_mode=True).low_vram_mode is True

    def test_other_defaults_unchanged(self, sf3d):
        e = sf3d.SF3DEngine(device="cpu")
        assert e.texture_resolution == 1024
        assert e.device == "cpu"


class TestPassedToFromPretrained:
    """The attribute must actually reach SF3D.from_pretrained."""

    def test_load_model_forwards_the_attribute(self, sf3d):
        src = inspect.getsource(sf3d.SF3DEngine.load_model)
        assert "low_vram_mode=self.low_vram_mode" in src

    def test_no_hardcoded_true_in_load_model(self, sf3d):
        """ENG-37: the literal keyword argument must be gone."""
        src = inspect.getsource(sf3d.SF3DEngine.load_model)
        assert "low_vram_mode=True" not in src

    def test_call_is_keyword_form(self, sf3d):
        """Guards against accidentally passing it positionally."""
        src = inspect.getsource(sf3d.SF3DEngine.load_model)
        assert "low_vram_mode=self.low_vram_mode" in src


class TestEndToEndForwarding:
    """Run load_model with a stubbed SF3D to prove the value is forwarded."""

    @staticmethod
    def _stub_sf3d(captured):
        class _Model:
            def to(self, device):
                return self

            def eval(self):
                return self

        class _FakeSF3D:
            @staticmethod
            def from_pretrained(repo, config_name=None, weight_name=None, low_vram_mode=None):
                captured["low_vram_mode"] = low_vram_mode
                return _Model()

        return _FakeSF3D

    def test_false_is_forwarded(self, sf3d, monkeypatch):
        captured = {}
        fake_module = types.ModuleType("sf3d.system")
        fake_module.SF3D = self._stub_sf3d(captured)
        monkeypatch.setitem(sys.modules, "sf3d", types.ModuleType("sf3d"))
        monkeypatch.setitem(sys.modules, "sf3d.system", fake_module)
        monkeypatch.setattr(
            sf3d.SF3DEngine, "_locate_repo", lambda self: __import__("pathlib").Path("/tmp")
        )

        sf3d.SF3DEngine(device="cpu", low_vram_mode=False).load_model()
        assert captured["low_vram_mode"] is False

    def test_true_is_forwarded(self, sf3d, monkeypatch):
        captured = {}
        fake_module = types.ModuleType("sf3d.system")
        fake_module.SF3D = self._stub_sf3d(captured)
        monkeypatch.setitem(sys.modules, "sf3d", types.ModuleType("sf3d"))
        monkeypatch.setitem(sys.modules, "sf3d.system", fake_module)
        monkeypatch.setattr(
            sf3d.SF3DEngine, "_locate_repo", lambda self: __import__("pathlib").Path("/tmp")
        )

        sf3d.SF3DEngine(device="cpu", low_vram_mode=True).load_model()
        assert captured["low_vram_mode"] is True
