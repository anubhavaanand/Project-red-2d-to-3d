"""Tests for SPAR3D texture normalisation (ENG-35)."""

import ast
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ENGINE = Path(__file__).parent.parent / "src" / "engines" / "spar3d_engine.py"


def _to_rgb_array():
    tree = ast.parse(ENGINE.read_text())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_to_rgb_array"
    )
    ns = {"np": np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(ENGINE), "exec"), ns)
    return ns["_to_rgb_array"]


class TestToRgbArray:
    def test_pil_rgb(self):
        img = Image.new("RGB", (4, 4), (255, 128, 0))
        out = _to_rgb_array()(img)
        assert out.shape == (4, 4, 3)
        assert out.dtype == np.float32
        assert out[0, 0][0] == pytest.approx(1.0)
        assert out[0, 0][1] == pytest.approx(128 / 255, abs=1e-3)

    def test_pil_rgba_drops_alpha(self):
        img = Image.new("RGBA", (4, 4), (10, 20, 30, 255))
        out = _to_rgb_array()(img)
        assert out.shape == (4, 4, 3)

    def test_pil_l_is_promoted(self):
        img = Image.new("L", (4, 4), 128)
        out = _to_rgb_array()(img)
        assert out.shape == (4, 4, 3)

    def test_numpy_uint8_is_scaled(self):
        arr = np.full((4, 4, 3), 255, dtype=np.uint8)
        out = _to_rgb_array()(arr)
        assert out.max() <= 1.0
        assert out[0, 0][0] == pytest.approx(1.0)

    def test_numpy_float_is_kept(self):
        arr = np.full((4, 4, 3), 0.5, dtype=np.float32)
        out = _to_rgb_array()(arr)
        assert out.max() <= 1.0
        assert out[0, 0][0] == pytest.approx(0.5)

    def test_numpy_rgba_drops_alpha(self):
        arr = np.zeros((4, 4, 4), dtype=np.float32)
        assert _to_rgb_array()(arr).shape == (4, 4, 3)

    def test_grayscale_array_is_promoted(self):
        arr = np.full((4, 4), 0.25, dtype=np.float32)
        out = _to_rgb_array()(arr)
        assert out.shape == (4, 4, 3)

    def test_none_returns_none(self):
        assert _to_rgb_array()(None) is None

    def test_bad_shape_returns_none(self):
        assert _to_rgb_array()(np.zeros((4,), dtype=np.float32)) is None

    def test_all_black_array_is_not_scaled(self):
        arr = np.zeros((2, 2, 3), dtype=np.float32)
        out = _to_rgb_array()(arr)
        assert np.all(out == 0.0)


class TestSourceContract:
    def test_texture_convert_is_guarded(self):
        """ENG-35: the *texture* conversion must live inside the helper.

        A blind \`tex_img.convert(...)\` on the material is the bug; converting
        the incoming PIL image in \`predict\` is fine.
        """
        tree = ast.parse(ENGINE.read_text())
        helper = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_to_rgb_array"
        )
        helper_lines = set(range(helper.lineno, (helper.end_lineno or helper.lineno) + 1))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "convert":
                if node.lineno in helper_lines:
                    continue
                # only the texture variable is the concern
                src = ast.unparse(node.func.value)
                if "tex" in src or "img" == src:
                    pytest.fail(f"texture .convert() outside helper at line {node.lineno}")

    def test_call_site_uses_helper(self):
        src = ENGINE.read_text()
        assert "rgb = _to_rgb_array(tex_img)" in src

    def test_helper_defined(self):
        src = ENGINE.read_text()
        assert "def _to_rgb_array(tex_img)" in src

    def test_engine_parses(self):
        ast.parse(ENGINE.read_text())