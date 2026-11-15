"""Tests for the v1.0 benchmark-freeze opt-in flags.

The published numbers are frozen, so the new ENG-05 and ENG-06 behaviour must
be opt-in and the default must reproduce legacy v1.0 exactly.
"""

import ast
from pathlib import Path

import numpy as np
import pytest

ENGINES = Path(__file__).parent.parent / "src" / "engines"
MASK = ENGINES / "_mask.py"
TRIPOSR = ENGINES / "triposr_engine.py"


def _mask_ns():
    """Exec _mask.py's functions with its module constants."""
    src = MASK.read_text()
    tree = ast.parse(src)

    # module-level constants
    ns = {"np": np}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                exec(ast.unparse(node), ns)
            except Exception:
                pass

    # the two functions under test
    for name in ("derive_foreground_mask", "_drop_speckle"):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(ast.unparse(fn), ns)
    return ns["derive_foreground_mask"]


class TestMaskFlagExists:
    def test_filter_speckle_parameter_defaults_to_true(self):
        src = MASK.read_text()
        assert "def derive_foreground_mask(rgb: np.ndarray, filter_speckle: bool = True)" in src

    def test_module_level_flag_reads_env(self):
        src = MASK.read_text()
        assert 'os.environ.get("IMAGE_TO_3D_FILTER_SPECKLE", "0") == "1"' in src

    def test_default_is_legacy(self):
        """The shipped default must be the legacy behaviour (no speckle filter)."""
        src = MASK.read_text()
        assert 'FILTER_SPECKLE = os.environ.get("IMAGE_TO_3D_FILTER_SPECKLE", "0") == "1"' in src


class TestMaskBehaviour:
    def test_legacy_path_matches_the_old_expression(self):
        derive = _mask_ns()
        rng = np.random.default_rng(0)
        for _ in range(50):
            img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
            legacy = np.any(img[..., :3] < 248, axis=-1)
            assert np.array_equal(derive(img, filter_speckle=False), legacy)

    def test_filter_path_removes_speckle(self):
        derive = _mask_ns()
        img = np.full((128, 128, 3), 255, np.uint8)
        img[40:88, 40:88] = 30  # real object
        rng = np.random.default_rng(1)
        for _ in range(200):
            y, x = rng.integers(0, 128, 2)
            img[y, x] = 200  # speckle
        assert derive(img, filter_speckle=True).sum() < derive(img, filter_speckle=False).sum()

    def test_filter_path_keeps_the_object(self):
        derive = _mask_ns()
        img = np.full((128, 128, 3), 255, np.uint8)
        img[40:88, 40:88] = 30
        assert derive(img, filter_speckle=True)[60, 60]


class TestEnginesPassTheFlag:
    @pytest.mark.parametrize(
        "engine", ["triposr_engine.py", "trellis_engine.py", "spar3d_engine.py", "lgm_engine.py"]
    )
    def test_engine_forwards_the_flag(self, engine):
        src = (ENGINES / engine).read_text()
        assert "filter_speckle=FILTER_SPECKLE" in src

    @pytest.mark.parametrize(
        "engine", ["triposr_engine.py", "trellis_engine.py", "spar3d_engine.py", "lgm_engine.py"]
    )
    def test_engine_imports_the_flag(self, engine):
        src = (ENGINES / engine).read_text()
        assert "FILTER_SPECKLE" in src.split("class ")[0]


class TestOrientOutwardFlag:
    def test_check_winding_parameter_defaults_to_false(self):
        src = TRIPOSR.read_text()
        assert "def _orient_outward(mesh_obj, check_winding: bool = False) -> str:" in src

    def test_module_flag_reads_env(self):
        src = TRIPOSR.read_text()
        assert 'os.environ.get("IMAGE_TO_3D_CHECK_WINDING", "0") == "1"' in src

    def test_call_site_forwards_the_flag(self):
        src = TRIPOSR.read_text()
        assert "_orient_outward(mesh_obj, check_winding=CHECK_WINDING)" in src

    def test_legacy_path_uses_origin(self):
        src = TRIPOSR.read_text()
        legacy = src[src.index("if not check_winding:") : src.index("import trimesh")]
        assert "np.zeros(3)" in legacy

    def test_new_path_uses_centroid(self):
        src = TRIPOSR.read_text()
        assert "centroid = vertices.mean(axis=0)" in src

    def test_new_path_checks_watertight(self):
        src = TRIPOSR.read_text()
        assert "if not tm.is_watertight or not tm.is_winding_consistent:" in src


class TestLegacyEquivalence:
    def test_origin_sign_matches_the_original_expression(self):
        """The v1.0 code used sum(v0 * cross(v1, v2)) about the origin."""

        def signed(v, f, origin):
            v0 = v[f[:, 0]] - origin
            v1 = v[f[:, 1]] - origin
            v2 = v[f[:, 2]] - origin
            return float(np.sum(v0 * np.cross(v1, v2)) / 6.0)

        rng = np.random.default_rng(0)
        for _ in range(100):
            v = rng.normal(size=(40, 3))
            f = rng.integers(0, 40, size=(20, 3))
            legacy = float(np.sum(v[f[:, 0]] * np.cross(v[f[:, 1]], v[f[:, 2]])))
            assert np.sign(legacy) == np.sign(signed(v, f, np.zeros(3)))
