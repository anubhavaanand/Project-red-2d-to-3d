"""Tests for the API's output-format contract (API-03).

These exercise the pure decision logic without importing FastAPI, torch or any
engine, so they run in a bare environment.
"""

import ast
from pathlib import Path

SERVER_PY = Path(__file__).parent.parent / "src" / "api" / "server.py"


def _tree():
    return ast.parse(SERVER_PY.read_text())


def _engines_payload():
    """Extract the list returned by /engines as plain Python.

    ``ast.literal_eval`` cannot resolve ``EngineType.X`` attribute references,
    so those are rewritten to their string names first.
    """
    tree = _tree()

    class _EnumToStr(ast.NodeTransformer):
        def visit_Attribute(self, node):
            self.generic_visit(node)
            if isinstance(node.value, ast.Name) and node.value.id == "EngineType":
                return ast.Constant(value=node.attr)
            return node

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "list_engines":
            ret = next(n for n in ast.walk(node) if isinstance(n, ast.Return))
            return ast.literal_eval(_EnumToStr().visit(ret.value))
    raise AssertionError("list_engines not found")


class TestEnginesAdvertisement:
    def test_every_engine_declares_a_default_format(self):
        engines = _engines_payload()
        assert engines, "no engines advertised"
        for e in engines:
            assert "default_output_format" in e
            assert "supported_output_formats" in e
            assert e["default_output_format"] in e["supported_output_formats"]

    def test_lgm_defaults_to_gaussians(self):
        """API-03: LGM is advertised as gaussians (.ply), so it must default there."""
        lgm = next(e for e in _engines_payload() if e["id"] == "LGM")
        assert lgm["default_output_format"] == "gaussians"
        assert "gaussians" in lgm["supported_output_formats"]

    def test_triposr_is_mesh_only(self):
        triposr = next(e for e in _engines_payload() if e["id"] == "TRIPOSR")
        assert triposr["supported_output_formats"] == ["mesh"]

    def test_trellis_supports_both(self):
        trellis = next(e for e in _engines_payload() if e["id"] == "TRELLIS")
        assert set(trellis["supported_output_formats"]) == {"mesh", "gaussians"}

    def test_spar3d_and_sf3d_are_advertised(self):
        """API-04: both engines are in EngineType and must be served."""
        engines = {e["id"]: e for e in _engines_payload()}
        assert "SPAR3D" in engines
        assert "SF3D" in engines
        for name in ("SPAR3D", "SF3D"):
            assert engines[name]["default_output_format"] == "mesh"
            assert engines[name]["supported_output_formats"] == ["mesh"]

    def test_all_five_engine_types_are_advertised(self):
        """EngineType has five members; /engines must list all of them."""
        advertised = {e["id"] for e in _engines_payload()}
        assert advertised == {"TRIPOSR", "LGM", "TRELLIS", "SPAR3D", "SF3D"}

    def test_get_engine_handles_every_engine_type(self):
        """Every EngineType must have a construction branch in get_engine."""
        import src.config as config

        src = SERVER_PY.read_text()
        for member in config.EngineType:
            assert f"EngineType.{member.name}" in src, member


class TestPredictFormatHandling:
    def test_output_format_is_a_form_field(self):
        tree = _tree()
        predict = next(
            n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "predict"
        )
        names = {a.arg for a in predict.args.args}
        assert "output_format" in names

    def test_invalid_output_format_is_rejected(self):
        src = SERVER_PY.read_text()
        assert "output_format must be 'mesh' or 'gaussians'" in src

    def test_lgm_passes_extract_mesh(self):
        """LGM must be told which artefact to build."""
        src = SERVER_PY.read_text()
        assert "engine.predict, image, want_mesh" in src

    def test_gaussian_request_is_preferred_over_mesh(self):
        """A gaussians request must not be silently served as .glb."""
        src = SERVER_PY.read_text()
        assert "want_gaussians = requested_format == \"gaussians\"" in src
        # the gaussians branch must come before the mesh branch
        assert src.index("if want_gaussians and result.gaussians is not None") < src.index(
            "if result.mesh is not None"
        )