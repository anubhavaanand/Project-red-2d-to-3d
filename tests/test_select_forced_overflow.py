"""Tests for forced-object overflow handling (TOOL-16)."""

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import gso1000.select_1000 as mod  # noqa: E402


@pytest.fixture
def select_mod(monkeypatch):
    """select_1000 with a small TARGET so overflow is easy to trigger."""

    importlib.reload(mod)
    return mod


def _run(mod, forced, valid=None, target=3):
    mod.TARGET = target
    valid = valid or {f"o{i}" for i in range(10)}
    return mod.select(sorted(valid), valid, forced, seed=0)


class TestNoOverflow:
    def test_forced_within_target_is_kept(self, select_mod):
        manifest, stats, extras = _run(select_mod, ["o0", "o1"], target=3)
        assert set(manifest) >= {"o0", "o1"}
        assert extras["forced_overflow_ids"] == []

    def test_fewer_forced_than_target_fills(self, select_mod):
        manifest, stats, extras = _run(select_mod, ["o0"], target=3)
        assert len(manifest) == 3
        assert "o0" in manifest


class TestOverflow:
    def test_overflow_is_recorded(self, select_mod):
        """TOOL-16: forced objects beyond TARGET must not vanish silently."""
        manifest, stats, extras = _run(select_mod, ["o0", "o1", "o2", "o3", "o4"], target=3)
        assert len(manifest) == 3
        assert extras["forced_overflow_ids"] == ["o3", "o4"]
        assert stats["forced_overflow"] == 2

    def test_overflow_ids_are_exact(self, select_mod):
        forced = [f"f{i}" for i in range(6)]
        valid = set(forced) | {f"o{i}" for i in range(10)}
        manifest, stats, extras = _run(select_mod, forced, valid=valid, target=3)
        assert extras["forced_overflow_ids"] == ["f3", "f4", "f5"]
        assert not (set(extras["forced_overflow_ids"]) & set(manifest))

    def test_manifest_is_capped(self, select_mod):
        _, _, extras = _run(select_mod, [f"o{i}" for i in range(10)], target=3)
        assert len(extras["forced_overflow_ids"]) == 7


class TestInvalidForced:
    def test_invalid_forced_recorded_separately(self, select_mod):
        """Forced objects that fail validation are a different bucket."""
        manifest, stats, extras = _run(select_mod, ["o0", "ghost"], target=5)
        assert "ghost" in extras["forced_dropped_ids"]
        assert stats["forced_dropped_invalid"] == 1


class TestContract:
    def test_extras_has_overflow_keys(self, select_mod):
        _, _, extras = _run(select_mod, ["o0"], target=3)
        assert "forced_overflow_ids" in extras

    def test_stats_has_overflow_count(self, select_mod):
        _, stats, _ = _run(select_mod, ["o0"], target=3)
        assert "forced_overflow" in stats

    def test_overflow_is_captured_not_silently_dropped(self):
        """TOOL-16: the cap may stay, but the remainder must be recorded."""
        src = (ROOT / "gso1000" / "select_1000.py").read_text()
        assert "ordered_forced[:TARGET]" in src
        assert "ordered_forced[TARGET:]" in src
        assert "forced_overflow_ids" in src
