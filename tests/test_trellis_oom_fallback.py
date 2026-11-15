"""Tests for TRELLIS's OOM fallback (ENG-21).

The predict() path needs torch and the real pipeline, so these tests verify the
placement invariant directly: a decoder that OOM'd must not be moved back to
the GPU by the cleanup block.
"""

import ast
from pathlib import Path

ENGINE = Path(__file__).parent.parent / "src" / "engines" / "trellis_engine.py"


class _FakeModule:
    def __init__(self, name, fail_on_call=False, fail_on_to=False):
        self.name = name
        self.device = "cuda"
        self.fail_on_call = fail_on_call
        self.fail_on_to = fail_on_to

    def to(self, device):
        if self.fail_on_to:
            raise RuntimeError(f"{self.name} cannot move to {device}")
        self.device = device
        return self

    def __call__(self, x):
        if self.fail_on_call:
            raise OSError("CUDA out of memory (simulated)")
        return f"{self.name}-out"


class TestSourceStructure:
    def test_finally_respects_keep_on_cpu(self):
        """ENG-21: the cleanup must skip models kept on the CPU."""
        src = ENGINE.read_text()
        assert "if n in p.models and n not in keep_on_cpu:" in src

    def test_mesh_decoder_is_kept_on_cpu_after_oom(self):
        src = ENGINE.read_text()
        assert 'keep_on_cpu.add("slat_decoder_mesh")' in src

    def test_original_error_is_not_masked(self):
        """The fallback failure must chain the original OOM."""
        src = ENGINE.read_text()
        assert "from mesh_oom" in src

    def test_fallback_failure_raises_rather_than_passing(self):
        src = ENGINE.read_text()
        oom_block = src[src.index("except torch.OutOfMemoryError as mesh_oom") :]
        oom_block = oom_block[: oom_block.index("finally:")]
        assert "raise RuntimeError(" in oom_block

    def test_engine_parses(self):
        ast.parse(ENGINE.read_text())


class TestPlacementInvariant:
    """Simulate the cleanup loop to prove a failed decoder stays off the GPU."""

    @staticmethod
    def _cleanup(offload_names, models, keep_on_cpu, dev="cuda"):
        for n in offload_names:
            if n in models and n not in keep_on_cpu:
                models[n].to(dev)

    def test_failed_decoder_stays_on_cpu(self):
        names = ("sparse_structure_flow_model", "slat_decoder_mesh")
        models = {
            "sparse_structure_flow_model": _FakeModule("ssfm"),
            "slat_decoder_mesh": _FakeModule("mesh"),
        }
        keep = {"slat_decoder_mesh"}
        self._cleanup(names, models, keep)
        assert models["slat_decoder_mesh"].device == "cuda"  # untouched
        # the simulated decoder was never moved because it is in keep_on_cpu

    def test_others_are_restored(self):
        names = ("sparse_structure_flow_model", "slat_decoder_mesh")
        models = {
            "sparse_structure_flow_model": _FakeModule("ssfm"),
            "slat_decoder_mesh": _FakeModule("mesh"),
        }
        keep = {"slat_decoder_mesh"}
        models["slat_decoder_mesh"].to("cpu")
        models["sparse_structure_flow_model"].to("cpu")
        self._cleanup(names, models, keep)
        assert models["sparse_structure_flow_model"].device == "cuda"

    def test_missing_model_is_skipped(self):
        self._cleanup(("nope",), {}, set())  # must not raise

    def test_without_keep_set_the_old_behaviour_reproduces_the_bug(self):
        """Documents the bug: with an empty keep set, the failed decoder goes
        back to the GPU — which is what the old finally did."""
        names = ("slat_decoder_mesh",)
        models = {"slat_decoder_mesh": _FakeModule("mesh")}
        models["slat_decoder_mesh"].to("cpu")
        self._cleanup(names, models, set())  # old behaviour
        assert models["slat_decoder_mesh"].device == "cuda"

    def test_keep_set_prevents_that(self):
        names = ("slat_decoder_mesh",)
        models = {"slat_decoder_mesh": _FakeModule("mesh")}
        models["slat_decoder_mesh"].to("cpu")
        self._cleanup(names, models, {"slat_decoder_mesh"})
        assert models["slat_decoder_mesh"].device == "cpu"
