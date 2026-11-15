"""Tests for scoped third-party compatibility shims (SEC-15)."""

import builtins
import sys
import types

import pytest


@pytest.fixture
def stub_torch(monkeypatch):
    """Minimal torch stub: the shim only needs nn.Linear and a few tensor ops."""
    torch = types.ModuleType("torch")
    torch.ones = lambda *shape: _FakeTensor(list(shape))
    torch.arange = lambda n: _FakeTensor(list(range(n)))
    torch.nn = types.ModuleType("torch.nn")
    torch.nn.functional = types.ModuleType("torch.nn.functional")
    torch.nn.functional.scaled_dot_product_attention = lambda q, k, v: _FakeTensor([1])
    torch.nn.functional.__spec__ = None

    class _Linear:
        def __init__(self, in_f, out_f, bias=True):
            self.in_features, self.out_features = in_f, out_f
            self.weight = _FakeTensor([out_f, in_f])
            self.bias = _FakeTensor([out_f]) if bias else None

        def to(self, device):
            return self

    torch.nn.Linear = _Linear
    torch.nn.functional.__spec__ = None
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.nn", torch.nn)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", torch.nn.functional)
    return torch


class _FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)
        self._data = list(shape)

    def view(self, *shape):
        return _FakeTensor(shape)

    def contiguous(self):
        return self

    def eq(self, other):
        return self

    def __getitem__(self, item):
        return _FakeTensor([1])

    def to(self, device):
        return self

    def index_select(self, dim, index):
        return _FakeTensor(self.shape)

    def clone(self):
        return self

    def detach(self):
        return self

    def copy_(self, other):
        return self

    def unsqueeze(self, dim):
        return self


@pytest.fixture
def stub_transparent_background(monkeypatch):
    mod = types.ModuleType("transparent_background")
    monkeypatch.setitem(sys.modules, "transparent_background", mod)
    return mod


class TestXformersShimScope:
    """The xformers shim must not outlive the block that installs it."""

    @pytest.fixture
    def no_real_xformers(self, monkeypatch):
        import src.engines._compat as compat

        real_find_spec = compat.importlib.util.find_spec

        def _find_spec(name, *a, **k):
            if name == "xformers":
                return None
            return real_find_spec(name, *a, **k)

        monkeypatch.setattr(compat.importlib.util, "find_spec", _find_spec)
        monkeypatch.setitem(sys.modules, "xformers", None)
        monkeypatch.setitem(sys.modules, "xformers.ops", None)

    def test_shim_present_inside_and_absent_after(self, stub_torch, no_real_xformers):
        from src.engines._compat import lgm_xformers_shim

        saved = {k: v for k, v in sys.modules.items() if k.split(".")[0] == "xformers"}
        with lgm_xformers_shim():
            assert sys.modules.get("xformers") is not None
            ops = sys.modules["xformers.ops"]
            assert hasattr(ops, "memory_efficient_attention")
            assert hasattr(ops, "unbind")
            # diffusers calls find_spec() at import time; a fake module needs __spec__
            assert sys.modules["xformers"].__spec__ is not None
        for key in ("xformers", "xformers.ops"):
            assert sys.modules.get(key) is None, f"{key} leaked"
        sys.modules.update(saved)

    def test_shim_removed_even_on_exception(self, stub_torch, no_real_xformers):
        from src.engines._compat import lgm_xformers_shim

        with pytest.raises(RuntimeError):
            with lgm_xformers_shim():
                raise RuntimeError("boom")
        assert sys.modules.get("xformers") is None
        assert sys.modules.get("xformers.ops") is None

    def test_real_xformers_is_left_alone(self, monkeypatch):
        """A real xformers with memory_efficient_attention must not be replaced."""
        import src.engines._compat as compat

        fake_ops = types.ModuleType("xformers.ops")
        fake_ops.memory_efficient_attention = lambda *a, **k: None
        fake_ops.__spec__ = None
        fake_pkg = types.ModuleType("xformers")
        fake_pkg.ops = fake_ops
        fake_pkg.__spec__ = None

        real_find_spec = compat.importlib.util.find_spec
        monkeypatch.setattr(
            compat.importlib.util,
            "find_spec",
            lambda name, *a, **k: object() if name == "xformers" else real_find_spec(name, *a, **k),
        )
        monkeypatch.setitem(sys.modules, "xformers", fake_pkg)
        monkeypatch.setitem(sys.modules, "xformers.ops", fake_ops)

        from src.engines._compat import lgm_xformers_shim

        with lgm_xformers_shim():
            assert sys.modules["xformers"] is fake_pkg
        assert sys.modules["xformers"] is fake_pkg

    def test_attn_bias_is_rejected(self, stub_torch, no_real_xformers):
        """The shim must not silently ignore an xformers attn_bias."""
        from src.engines._compat import lgm_xformers_shim

        with lgm_xformers_shim():
            ops = sys.modules["xformers.ops"]
            with pytest.raises(NotImplementedError, match="attention bias"):
                ops.memory_efficient_attention(None, None, None, attn_bias=object())


class TestSpar3dShimScope:
    """SPAR3D shims must be undone, including a fabricated pkg_resources."""

    @pytest.fixture
    def fake_transformers(self, monkeypatch):
        tpu = types.ModuleType("transformers.pytorch_utils")
        tpu.__spec__ = None
        modeling_utils = types.ModuleType("transformers.modeling_utils")
        modeling_utils.__spec__ = None

        class PreTrainedModel:
            pass

        modeling_utils.PreTrainedModel = PreTrainedModel
        transformers = types.ModuleType("transformers")
        transformers.__path__ = []

        monkeypatch.setitem(sys.modules, "transformers", transformers)
        monkeypatch.setitem(sys.modules, "transformers.pytorch_utils", tpu)
        monkeypatch.setitem(sys.modules, "transformers.modeling_utils", modeling_utils)
        return tpu, PreTrainedModel

    def test_transformers_patches_are_reverted(
        self, stub_torch, stub_transparent_background, fake_transformers
    ):
        from src.engines._compat import spar3d_compat_shims

        tpu, PreTrainedModel = fake_transformers
        assert not hasattr(tpu, "find_pruneable_heads_and_indices")

        with spar3d_compat_shims():
            assert hasattr(tpu, "find_pruneable_heads_and_indices")
            assert hasattr(tpu, "prune_linear_layer")
            assert hasattr(PreTrainedModel, "get_head_mask")

        assert not hasattr(tpu, "find_pruneable_heads_and_indices")
        assert not hasattr(tpu, "prune_linear_layer")
        assert not hasattr(PreTrainedModel, "get_head_mask")

    def test_existing_helpers_are_not_overwritten(
        self, stub_torch, stub_transparent_background, fake_transformers
    ):
        from src.engines._compat import spar3d_compat_shims

        tpu, _ = fake_transformers

        def sentinel(*a, **k):
            return "original"

        tpu.find_pruneable_heads_and_indices = sentinel
        with spar3d_compat_shims():
            assert tpu.find_pruneable_heads_and_indices is sentinel
        assert tpu.find_pruneable_heads_and_indices is sentinel

    def test_fabricated_pkg_resources_is_removed(
        self, monkeypatch, stub_torch, stub_transparent_background
    ):
        from src.engines._compat import spar3d_compat_shims

        monkeypatch.delitem(sys.modules, "pkg_resources", raising=False)
        real_import = builtins.__import__

        def _no_pkg_resources(name, *a, **k):
            if name == "pkg_resources":
                raise ImportError("no pkg_resources")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", _no_pkg_resources)
        with spar3d_compat_shims():
            fake = sys.modules.get("pkg_resources")
            assert fake is not None
            assert hasattr(fake, "packaging")
        assert sys.modules.get("pkg_resources") is None

    def test_shims_removed_even_on_exception(
        self, stub_torch, stub_transparent_background, fake_transformers
    ):
        from src.engines._compat import spar3d_compat_shims

        tpu, PreTrainedModel = fake_transformers
        with pytest.raises(RuntimeError):
            with spar3d_compat_shims():
                raise RuntimeError("boom")
        assert not hasattr(tpu, "find_pruneable_heads_and_indices")
        assert not hasattr(PreTrainedModel, "get_head_mask")

    def test_missing_transparent_background_raises(self, monkeypatch, stub_torch):
        from src.engines._compat import spar3d_compat_shims

        monkeypatch.delitem(sys.modules, "transparent_background", raising=False)
        real_import = builtins.__import__

        def _fail(name, *a, **k):
            if name == "transparent_background":
                raise ImportError("nope")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", _fail)
        with pytest.raises(ImportError, match="transparent-background"):
            with spar3d_compat_shims():
                pass
