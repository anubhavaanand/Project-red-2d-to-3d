"""Behavioural tests for TRELLIS output-format handling (ENG-27).

Exercises the real guard in predict() rather than asserting on source text.
"""

import sys
import types

import pytest


@pytest.fixture(scope="module")
def trellis_mod():
    """Import trellis_engine with a minimal torch stub.

    Installs unconditionally: other test modules may have registered a different
    stub, and this one must be the one trellis_engine sees.
    """
    t = types.ModuleType("torch")
    t.__version__ = "0.0-stub"
    t.float32, t.float16, t.int64 = "float32", "float16", "int64"
    t.device = lambda x: x
    t.manual_seed = lambda s: None

    class _NoCtx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def __call__(self, fn):
            # supports use as a decorator: @torch.no_grad()
            return fn

    t.no_grad = lambda: _NoCtx()
    t.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        empty_cache=lambda: None,
        ipc_collect=lambda: None,
        device=lambda d: _NoCtx(),
    )
    t.nn = types.ModuleType("torch.nn")
    t.nn.functional = types.ModuleType("torch.nn.functional")
    t.OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
    sys.modules["torch"] = t
    sys.modules["torch.nn"] = t.nn
    sys.modules["torch.nn.functional"] = t.nn.functional

    from src.engines import trellis_engine

    return trellis_engine


class TestSupportedFormats:
    def test_literal_has_exactly_mesh_and_gaussian(self, trellis_mod):
        import typing

        args = typing.get_args(trellis_mod.OutputFormat)
        assert set(args) == {"mesh", "gaussian"}

    def test_radiance_field_is_not_accepted(self, trellis_mod):
        import typing

        assert "radiance_field" not in typing.get_args(trellis_mod.OutputFormat)


class TestPredictRejectsUnsupported:
    """ENG-27: an unimplemented format must raise before any decoding."""

    @pytest.fixture
    def engine(self, trellis_mod):
        """An engine with a stub pipeline, so the guard is what gets exercised."""
        e = trellis_mod.TrellisEngine(device="cpu", output_format="mesh")
        e.pipeline = types.SimpleNamespace(models={})
        return e

    @pytest.mark.parametrize("bad", ["radiance_field", "pointcloud", ""])
    def test_bad_format_raises_typed_error(self, trellis_mod, engine, bad):
        engine.output_format = bad
        with pytest.raises(trellis_mod.TrellisOutputError) as exc:
            engine.predict(_fake_image())
        assert "not implemented" in str(exc.value)

    def test_error_names_the_offending_value(self, trellis_mod, engine):
        engine.output_format = "radiance_field"
        with pytest.raises(trellis_mod.TrellisOutputError, match="radiance_field"):
            engine.predict(_fake_image())

    def test_no_decoding_is_attempted(self, trellis_mod, engine):
        """The guard must fire before any pipeline model is touched."""
        engine.output_format = "radiance_field"

        class _ExplodingPipeline:
            models = {}

            def __getattr__(self, item):
                raise AssertionError(f"pipeline.{item} touched for a bad format")

        engine.pipeline = _ExplodingPipeline()
        with pytest.raises(trellis_mod.TrellisOutputError):
            engine.predict(_fake_image())


class TestValidFormatsReachThePipeline:
    @pytest.mark.parametrize("fmt", ["mesh", "gaussian"])
    def test_valid_format_is_not_rejected_by_the_guard(self, trellis_mod, fmt):
        """A valid format must get past the guard and fail later, on the stub
        pipeline — not on the format check."""
        e = trellis_mod.TrellisEngine(device="cpu", output_format=fmt)
        e.pipeline = types.SimpleNamespace(models={})
        with pytest.raises(Exception) as exc:
            e.predict(_fake_image())
        assert not isinstance(exc.value, trellis_mod.TrellisOutputError), (
            f"{fmt} was rejected by the format guard"
        )


class TestEngineInfo:
    def test_engine_info_lists_supported_formats(self, trellis_mod):
        info = trellis_mod.TrellisEngine(device="cpu").get_engine_info()
        assert set(info["output_formats"]) == {
            "mesh (FlexiCubes @ 256^3)",
            "gaussian (SLAT)",
        }


def _fake_image():
    from PIL import Image

    return Image.new("RGBA", (32, 32), (255, 255, 255, 0))