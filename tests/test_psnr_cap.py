"""Tests for PSNR edge cases (EVAL-06)."""

import numpy as np
import pytest

from src.evaluation.psnr_lpips import PSNR_CAP_DB, compute_psnr


class TestIdenticalImages:
    def test_returns_the_documented_cap(self):
        """EVAL-06: a perfect match must not silently report a real score."""
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        assert compute_psnr(img, img) == PSNR_CAP_DB

    def test_cap_is_finite(self):
        """Returning inf would poison every plain sum()/len() mean downstream."""
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        assert np.isfinite(compute_psnr(img, img))

    def test_cap_is_explicit(self):
        assert PSNR_CAP_DB == 100.0

    def test_all_black_identical(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        assert compute_psnr(img, img) == PSNR_CAP_DB

    def test_all_white_identical(self):
        img = np.full((32, 32, 3), 255, dtype=np.uint8)
        assert compute_psnr(img, img) == PSNR_CAP_DB


class TestDifferentImages:
    def test_below_the_cap(self):
        a = np.zeros((32, 32, 3), dtype=np.uint8)
        b = np.full((32, 32, 3), 128, dtype=np.uint8)
        assert compute_psnr(a, b) < PSNR_CAP_DB

    def test_small_difference_is_high_but_below_cap(self):
        a = np.zeros((32, 32, 3), dtype=np.uint8)
        b = np.full((32, 32, 3), 1, dtype=np.uint8)
        psnr = compute_psnr(a, b)
        assert psnr < PSNR_CAP_DB
        assert psnr > 30

    def test_symmetric(self):
        a = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        b = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        assert compute_psnr(a, b) == pytest.approx(compute_psnr(b, a), rel=1e-5)

    def test_alpha_channel_is_ignored(self):
        """Only the first three channels are compared."""
        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        a = np.dstack([rgb, np.zeros((16, 16), dtype=np.uint8)])
        b = np.dstack([rgb, np.full((16, 16), 255, dtype=np.uint8)])
        assert compute_psnr(a, b) == PSNR_CAP_DB


class TestAggregationSafety:
    def test_mean_is_finite_even_for_perfect_matches(self):
        """The reason for the cap: aggregation must not produce inf."""
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        scores = [compute_psnr(img, img) for _ in range(10)]
        assert np.isfinite(np.mean(scores))
