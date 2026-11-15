"""Tests for TRELLIS's supported output formats (ENG-27)."""

import ast
from pathlib import Path

ENGINE = Path(__file__).parent.parent / "src" / "engines" / "trellis_engine.py"


class TestOutputFormatLiteral:
    def test_radiance_field_is_removed(self):
        src = ENGINE.read_text()
        assert '"radiance_field"' not in src.split("class TrellisOutputError")[0]

    def test_literal_has_mesh_and_gaussian(self):
        src = ENGINE.read_text()
        assert 'OutputFormat = Literal["mesh", "gaussian"]' in src

    def test_literal_is_documented(self):
        src = ENGINE.read_text()
        assert "TRELLIS's pipeline can decode a radiance field" in src


class TestPredictRejectsUnknown:
    def test_predict_guards_decode_mode(self):
        src = ENGINE.read_text()
        assert 'if decode_mode not in ("mesh", "gaussian"):' in src

    def test_guard_raises_typed_error(self):
        src = ENGINE.read_text()
        assert "raise TrellisOutputError(" in src
        assert "is not implemented" in src

    def test_guard_precedes_the_mesh_branch(self):
        src = ENGINE.read_text()
        guard = src.index('if decode_mode not in ("mesh", "gaussian"):')
        mesh_branch = src.index('if decode_mode == "mesh":')
        assert guard < mesh_branch

    def test_no_implementation_for_radiance_field(self):
        """The guard must be the only handling — no silent fallback."""
        src = ENGINE.read_text()
        predict = src[src.index("def predict(") : src.index("def unload_model")]
        assert "radiance" not in predict.lower() or "not implemented" in predict

    def test_engine_parses(self):
        ast.parse(ENGINE.read_text())


class TestCallersUnaffected:
    def test_app_uses_mesh(self):
        app = ENGINE.parent.parent.parent / "app.py"
        assert 'TrellisEngine(device=DEVICE, output_format="mesh"' in app.read_text()

    def test_notebooks_using_trellis_ask_for_mesh(self):
        checked = 0
        for nb in (ENGINE.parent.parent.parent / "kaggle").rglob("*.ipynb"):
            text = nb.read_text()
            if "TrellisEngine(" in text:
                checked += 1
                # notebooks store source as JSON strings, so quotes are escaped
                assert 'output_format=\\"mesh\\"' in text, f"{nb.name} does not request mesh"
        assert checked >= 1, "no notebook constructs TrellisEngine"