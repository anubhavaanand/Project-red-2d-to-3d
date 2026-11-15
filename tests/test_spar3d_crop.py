"""Tests for SPAR3D's foreground-crop fallback (ENG-32)."""

import ast
from pathlib import Path

ENGINE = Path(__file__).parent.parent / "src" / "engines" / "spar3d_engine.py"


class TestSourceContract:
    def test_crop_error_is_initialised(self):
        src = ENGINE.read_text()
        assert "crop_error = None" in src

    def test_crop_failure_is_captured(self):
        src = ENGINE.read_text()
        assert "crop_error = f\"{type(crop_err).__name__}: {crop_err}\"" in src

    def test_crop_error_recorded_in_metadata(self):
        """ENG-32: the fallback must surface to the caller."""
        src = ENGINE.read_text()
        assert '"crop_error": crop_error' in src

    def test_message_says_uncropped(self):
        src = ENGINE.read_text()
        assert "using uncropped input" in src

    def test_engine_parses(self):
        ast.parse(ENGINE.read_text())


class TestSemantics:
    def test_crop_error_defaults_to_none(self):
        """The happy path must report no error."""
        src = ENGINE.read_text()
        predict = src[src.index("def predict(") :]
        assert predict.index("crop_error = None") < predict.index("self._foreground_crop(")

    def test_metadata_is_built_after_the_crop(self):
        """crop_error must be assigned before it is read into metadata."""
        src = ENGINE.read_text()
        assign = src.index("crop_error = f\"")
        read = src.index('"crop_error": crop_error')
        assert assign < read

    def test_crop_still_attempted_first(self):
        """The fallback must not skip cropping altogether."""
        src = ENGINE.read_text()
        assert "rgba = self._foreground_crop(" in src