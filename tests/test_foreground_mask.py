"""Tests for the shared foreground-mask derivation (ENG-05)."""

import numpy as np
import pytest

from src.engines._mask import (
    ALPHA_OPAQUE,
    WHITE_BACKGROUND_LEVEL,
    alpha_is_usable,
    derive_foreground_mask,
)


class TestAlphaIsUsable:
    def test_transparent_alpha_is_usable(self):
        arr = np.full((8, 8, 4), 255, np.uint8)
        arr[..., 3] = 0
        assert alpha_is_usable(arr)

    def test_partially_transparent_alpha_is_usable(self):
        arr = np.full((8, 8, 4), 255, np.uint8)
        arr[0, 0, 3] = 0  # a single transparent pixel is enough
        assert alpha_is_usable(arr)

    def test_fully_opaque_alpha_is_not_usable(self):
        arr = np.full((8, 8, 4), 255, np.uint8)
        assert not alpha_is_usable(arr)

    def test_opaque_at_the_boundary_is_not_usable(self):
        arr = np.full((8, 8, 4), 255, np.uint8)
        arr[..., 3] = ALPHA_OPAQUE
        assert not alpha_is_usable(arr)

    def test_just_below_the_boundary_is_usable(self):
        arr = np.full((8, 8, 4), 255, np.uint8)
        arr[..., 3] = ALPHA_OPAQUE - 1
        assert alpha_is_usable(arr)


class TestDeriveForegroundMask:
    def test_object_on_white_background(self):
        img = np.full((64, 64, 3), 255, np.uint8)
        img[20:44, 20:44] = 30
        mask = derive_foreground_mask(img)
        assert mask[30, 30]
        assert not mask[5, 5]
        assert mask.sum() == pytest.approx(576, abs=5)

    def test_all_white_yields_empty_mask(self):
        img = np.full((64, 64, 3), 255, np.uint8)
        assert not derive_foreground_mask(img).any()

    def test_near_white_is_background(self):
        img = np.full((64, 64, 3), WHITE_BACKGROUND_LEVEL, np.uint8)
        assert not derive_foreground_mask(img).any()

    def test_just_below_threshold_is_foreground(self):
        img = np.full((64, 64, 3), WHITE_BACKGROUND_LEVEL - 1, np.uint8)
        assert derive_foreground_mask(img).all()

    def test_isolated_speckle_is_removed(self):
        """ENG-05: JPEG noise must not become mask."""
        img = np.full((128, 128, 3), 255, np.uint8)
        img[40:88, 40:88] = 30
        rng = np.random.default_rng(1)
        for _ in range(200):
            y, x = rng.integers(0, 128, 2)
            img[y, x] = 200
        mask = derive_foreground_mask(img)
        assert mask.sum() == pytest.approx(2304, abs=10)

    def test_separate_parts_of_similar_size_are_kept(self):
        """Two blobs of comparable size are both real objects, not speckle."""
        img = np.full((128, 128, 3), 255, np.uint8)
        img[10:50, 10:50] = 30  # 1600 px
        img[70:110, 70:110] = 30  # 1600 px
        mask = derive_foreground_mask(img)
        assert mask.sum() == pytest.approx(3200, abs=10)

    def test_rgba_input_uses_only_colour(self):
        rgba = np.full((64, 64, 4), 255, np.uint8)
        rgba[20:44, 20:44, :3] = 30
        rgba[20:44, 20:44, 3] = 128
        mask = derive_foreground_mask(rgba)
        assert mask[30, 30]

    def test_mask_shape_matches_image(self):
        img = np.full((32, 48, 3), 255, np.uint8)
        assert derive_foreground_mask(img).shape == (32, 48)


class TestWiredIntoEngines:
    @pytest.mark.parametrize(
        "engine",
        ["triposr_engine.py", "trellis_engine.py", "spar3d_engine.py", "lgm_engine.py"],
    )
    def test_engine_uses_the_shared_helper(self, engine):
        from pathlib import Path

        src = (Path(__file__).parent.parent / "src" / "engines" / engine).read_text()
        assert "derive_foreground_mask(" in src, f"{engine} does not use the helper"

    @pytest.mark.parametrize(
        "engine",
        ["triposr_engine.py", "trellis_engine.py", "spar3d_engine.py", "lgm_engine.py"],
    )
    def test_old_inline_heuristic_is_gone(self, engine):
        from pathlib import Path

        src = (Path(__file__).parent.parent / "src" / "engines" / engine).read_text()
        assert "np.any(arr[..., :3] < 248" not in src
        assert "np.any(image_np[..., :3] < 248" not in src
