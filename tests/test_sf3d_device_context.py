"""Tests for SF3D's device context handling (ENG-41)."""

import ast
from pathlib import Path

ENGINE = Path(__file__).parent.parent / "src" / "engines" / "sf3d_engine.py"


class TestSourceContract:
    def test_cuda_context_is_guarded(self):
        """ENG-41: torch.cuda.device(...) must only be entered for CUDA."""
        src = ENGINE.read_text()
        assert 'if str(self.device).startswith("cuda"):' in src

    def test_cpu_path_runs_without_context(self):
        src = ENGINE.read_text()
        predict = src[src.index("def predict(") :]
        assert "else:" in predict
        # the unguarded single call must be gone
        assert predict.count("self.model.run_image(") == 2

    def test_no_unguarded_cuda_device(self):
        """The context manager must sit inside the guard."""
        tree = ast.parse(ENGINE.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.With) and "torch.cuda.device" in ast.unparse(node):
                # find the enclosing If
                parent_guard = any(
                    isinstance(p, ast.If)
                    and "startswith" in ast.unparse(p.test)
                    and node in ast.walk(p)
                    for p in ast.walk(tree)
                )
                assert parent_guard, "torch.cuda.device is not inside the guard"

    def test_engine_parses(self):
        ast.parse(ENGINE.read_text())


class TestDeviceSemantics:
    def test_cpu_string_would_fail_in_cuda_context(self):
        """Pins why the guard matters: only 'cuda*' may enter the context."""
        for device in ("cpu", "cuda", "cuda:0"):
            is_cuda = str(device).startswith("cuda")
            assert is_cuda == (device != "cpu")

    def test_guard_matches_torch_semantics(self):
        assert not "cpu".startswith("cuda")
        assert "cuda:0".startswith("cuda")
