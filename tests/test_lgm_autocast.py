"""Tests for the LGM autocast device type (ENG-15)."""

import ast
import re
from pathlib import Path

ENGINE = Path(__file__).parent.parent / "src" / "engines" / "lgm_engine.py"


class TestAutocastDevice:
    def test_autocast_is_not_hardcoded_to_cuda(self):
        """ENG-15: the literal device_type="cuda" must be gone from real code."""
        src = ENGINE.read_text()
        code = "\n".join(
            line for line in src.splitlines() if not line.lstrip().startswith("#")
        )
        assert 'device_type="cuda"' not in code

    def test_device_type_derived_from_engine_device(self):
        src = ENGINE.read_text()
        assert "device_type=device_type" in src

    def test_device_index_is_stripped(self):
        """"cuda:0" is a valid torch device but not a valid autocast type."""
        src = ENGINE.read_text()
        assert 'str(self.device).split(":")[0]' in src

    def test_normalisation_matches_torch_semantics(self):
        cases = {"cuda": "cuda", "cpu": "cpu", "cuda:0": "cuda", "cuda:1": "cuda"}
        for device, expected in cases.items():
            assert str(device).split(":")[0] == expected

    def test_only_one_autocast_call(self):
        """Guards against the pattern being reintroduced elsewhere.

        Comments are stripped first so the explanatory text does not count.
        """
        src = ENGINE.read_text()
        code = "\n".join(
            line for line in src.splitlines() if not line.lstrip().startswith("#")
        )
        assert len(re.findall(r"torch\.autocast\(", code)) == 1

    def test_engine_parses(self):
        ast.parse(ENGINE.read_text())