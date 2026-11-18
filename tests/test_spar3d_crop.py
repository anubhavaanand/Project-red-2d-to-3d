"""Behavioural tests for SPAR3D's foreground-crop fallback (ENG-32)."""

import pytest
from conftest import TorchStub
from PIL import Image


@pytest.fixture(scope="module")
def spar3d():
    """Import spar3d_engine with a minimal torch stub (restored on teardown)."""
    stub = TorchStub()
    stub()
    try:
        from src.engines import spar3d_engine

        return spar3d_engine
    finally:
        stub.restore()


class _Engine:
    """Minimal engine with the attributes the helper reads."""

    def __init__(self, crop=None, ratio=1.3):
        self._crop = crop
        self.foreground_ratio = ratio
        self.calls = []

    def _foreground_crop(self, image, crop_ratio=None, newsize=None):
        self.calls.append((crop_ratio, newsize))
        if self._crop is None:
            raise RuntimeError("no cropper configured")
        return self._crop(image)


def _img():
    return Image.new("RGBA", (64, 64), (10, 20, 30, 255))


class TestSuccessfulCrop:
    def test_returns_cropped_image_and_no_error(self, spar3d):
        cropped = _img()
        engine = _Engine(crop=lambda im: cropped)
        out, err = spar3d._apply_foreground_crop(engine, _img())
        assert out is cropped
        assert err is None

    def test_passes_ratio_and_size_through(self, spar3d):
        engine = _Engine(crop=lambda im: im)
        spar3d._apply_foreground_crop(engine, _img())
        assert engine.calls == [(1.3, (512, 512))]

    def test_uses_the_engine_ratio(self, spar3d):
        engine = _Engine(crop=lambda im: im, ratio=1.75)
        spar3d._apply_foreground_crop(engine, _img())
        assert engine.calls[0][0] == 1.75


class TestFailedCrop:
    def test_returns_original_image(self, spar3d):
        """ENG-32: a failed crop must fall back to the uncropped input."""
        original = _img()
        engine = _Engine(crop=None)
        out, err = spar3d._apply_foreground_crop(engine, original)
        assert out is original

    def test_error_is_returned_not_raised(self, spar3d):
        engine = _Engine(crop=None)
        _out, err = spar3d._apply_foreground_crop(engine, _img())
        assert err is not None

    def test_error_names_the_exception_type(self, spar3d):
        engine = _Engine(crop=None)
        _out, err = spar3d._apply_foreground_crop(engine, _img())
        assert err.startswith("RuntimeError:")

    def test_error_includes_the_message(self, spar3d):
        engine = _Engine(crop=None)
        _out, err = spar3d._apply_foreground_crop(engine, _img())
        assert "no cropper configured" in err

    def test_arbitrary_exception_types_are_captured(self, spar3d):
        class Boom:
            foreground_ratio = 1.3

            def _foreground_crop(self, *a, **k):
                raise ValueError("bad crop")

        _out, err = spar3d._apply_foreground_crop(Boom(), _img())
        assert err.startswith("ValueError:")
        assert "bad crop" in err


class TestWiring:
    def test_predict_uses_the_helper(self, spar3d):
        import inspect

        src = inspect.getsource(spar3d.Spar3dEngine.predict)
        assert "_apply_foreground_crop(self, rgba)" in src

    def test_crop_error_recorded_in_metadata(self, spar3d):
        import inspect

        src = inspect.getsource(spar3d.Spar3dEngine.predict)
        assert '"crop_error": crop_error' in src

    def test_crop_is_still_attempted(self, spar3d):
        """The fallback must not skip cropping altogether."""
        import inspect

        src = inspect.getsource(
            __import__("src.engines.spar3d_engine", fromlist=["x"])._apply_foreground_crop
        )
        assert "engine._foreground_crop(" in src
