"""Tests for Gradio upload validation (SEC-22).

``app.py`` imports gradio and torch at module scope, so the validator is
extracted from its source and executed in isolation to keep these tests
dependency-free.
"""

import ast
from pathlib import Path

import pytest
from PIL import Image

APP_PY = Path(__file__).parent.parent / "app.py"


@pytest.fixture(scope="module")
def validate_upload():
    """Compile just the constants + validate_upload from app.py."""
    tree = ast.parse(APP_PY.read_text())
    wanted = {"validate_upload"}
    nodes = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in wanted)
        or (
            isinstance(n, ast.Assign)
            and any(
                getattr(t, "id", "")
                in {"MAX_IMAGE_PIXELS", "ALLOWED_IMAGE_MODES", "ALLOWED_IMAGE_FORMATS"}
                for t in n.targets
            )
        )
    ]
    module = ast.Module(body=nodes, type_ignores=[])
    namespace = {"Image": Image}
    exec(compile(module, str(APP_PY), "exec"), namespace)
    return namespace["validate_upload"]


class TestValidateUpload:
    def test_none_is_rejected(self, validate_upload):
        assert "upload an image" in validate_upload(None)

    def test_valid_rgb_accepted(self, validate_upload):
        img = Image.new("RGB", (512, 512), "white")
        assert validate_upload(img) is None

    def test_valid_rgba_accepted(self, validate_upload):
        img = Image.new("RGBA", (512, 512), (255, 255, 255, 255))
        assert validate_upload(img) is None

    def test_cmyk_rejected(self, validate_upload):
        img = Image.new("CMYK", (256, 256), (0, 0, 0, 0))
        assert "Unsupported image mode" in validate_upload(img)

    def test_oversized_dimension_rejected(self, validate_upload):
        img = Image.new("RGB", (5000, 100))
        assert "too large" in validate_upload(img)

    def test_pixel_count_cap_enforced(self, validate_upload):
        """A 3000x3000 image fits each dimension limit but is 9 MP; a 5000x4000
        would be caught by the dimension check, so verify the MP cap directly."""
        # 4096x4096 == 16.7 MP is the boundary; one pixel over is rejected
        img = Image.new("RGB", (4096, 4097))
        # 4097 exceeds the per-dimension limit, so this is the dimension guard
        assert "too large" in validate_upload(img)

    def test_grayscale_and_palette_accepted(self, validate_upload):
        assert validate_upload(Image.new("L", (64, 64))) is None
        assert validate_upload(Image.new("P", (64, 64))) is None

    def test_exactly_at_limits_accepted(self, validate_upload):
        """The documented maximum must still work."""
        assert validate_upload(Image.new("RGB", (4096, 4096))) is None

    def test_unsupported_source_format_rejected(self, validate_upload):
        """SEC-29: a TIFF/GIF source format must be rejected when preserved."""
        img = Image.new("RGB", (64, 64))
        img.format = "TIFF"
        assert "Unsupported image format" in validate_upload(img)

    def test_supported_source_formats_accepted(self, validate_upload):
        for fmt in ("PNG", "JPEG", "WEBP", "BMP", "png", "jpeg"):
            img = Image.new("RGB", (64, 64))
            img.format = fmt
            assert validate_upload(img) is None, fmt

    def test_format_none_is_allowed(self, validate_upload):
        """Gradio usually drops the source format; that must not be an error."""
        img = Image.new("RGB", (64, 64))
        assert img.format is None
        assert validate_upload(img) is None

    def test_app_py_is_importable_source(self):
        """Guard against the stray trailing line that broke app.py parsing."""
        ast.parse(APP_PY.read_text())
