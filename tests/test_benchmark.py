"""
Test suite for the image-to-3d-benchmark project.
Covers: metrics, engines (import/smoke), evaluation runner, export, data contracts.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# =============================================================================
# METRICS TESTS (EVAL-31)
# =============================================================================

class TestChamferDistance:
    """Test Chamfer Distance computation."""

    def test_identical_point_clouds(self):
        """CD between identical clouds should be ~0."""
        from src.evaluation.chamfer_distance import compute_chamfer_distance
        pts = np.random.rand(100, 3).astype(np.float32)
        cd = compute_chamfer_distance(pts, pts)
        assert cd < 1e-6, f"Expected ~0, got {cd}"

    def test_symmetric_property(self):
        """CD(A, B) should equal CD(B, A)."""
        from src.evaluation.chamfer_distance import compute_chamfer_distance
        a = np.random.rand(50, 3).astype(np.float32)
        b = np.random.rand(60, 3).astype(np.float32)
        cd_ab = compute_chamfer_distance(a, b)
        cd_ba = compute_chamfer_distance(b, a)
        assert abs(cd_ab - cd_ba) < 1e-5

    def test_different_clouds(self):
        """CD between different clouds should be > 0."""
        from src.evaluation.chamfer_distance import compute_chamfer_distance
        a = np.zeros((100, 3), dtype=np.float32)
        b = np.ones((100, 3), dtype=np.float32)
        cd = compute_chamfer_distance(a, b)
        assert cd > 0.5

    def test_empty_cloud(self):
        """Empty cloud should return inf or handle gracefully."""
        from src.evaluation.chamfer_distance import compute_chamfer_distance
        a = np.random.rand(10, 3).astype(np.float32)
        with pytest.raises((ValueError, IndexError)):
            compute_chamfer_distance(a, np.array([]).reshape(0, 3))


class TestFScore:
    """Test F-Score computation."""

    def test_identical_clouds(self):
        """F-Score of identical clouds should be 1.0."""
        from src.evaluation.f_score import compute_f_score
        pts = np.random.rand(100, 3).astype(np.float32)
        scores = compute_f_score(pts, pts, thresholds=[0.1, 0.2, 0.5])
        for t in [0.1, 0.2, 0.5]:
            assert scores[f"f_score_{t}"] > 0.99

    def test_distant_clouds(self):
        """F-Score of distant clouds should be < 0.5."""
        from src.evaluation.f_score import compute_f_score
        a = np.zeros((100, 3), dtype=np.float32)
        b = np.ones((100, 3), dtype=np.float32) * 10
        scores = compute_f_score(a, b, thresholds=[0.1])
        assert scores["f_score_0.1"] < 0.5

    def test_threshold_types(self):
        """Test with different threshold sets."""
        from src.evaluation.f_score import compute_f_score
        a = np.random.rand(100, 3).astype(np.float32)
        b = a + np.random.randn(100, 3).astype(np.float32) * 0.05
        scores = compute_f_score(a, b, thresholds=[0.01, 0.05, 0.1])
        assert "f_score_0.01" in scores
        assert "f_score_0.05" in scores


class TestPSNR:
    """Test PSNR computation."""

    def test_identical_images(self):
        """PSNR of identical images should be inf or very high."""
        from src.evaluation.psnr_lpips import compute_psnr
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        psnr = compute_psnr(img, img)
        assert psnr > 50 or np.isinf(psnr)

    def test_different_images(self):
        """PSNR of different images should be < 30."""
        from src.evaluation.psnr_lpips import compute_psnr
        img1 = np.zeros((256, 256, 3), dtype=np.uint8)
        img2 = np.ones((256, 256, 3), dtype=np.uint8) * 128
        psnr = compute_psnr(img1, img2)
        assert psnr < 30


# =============================================================================
# ENGINE IMPORT TESTS
# =============================================================================

class TestEngineImports:
    """Verify all engines can be imported without CUDA."""

    def test_triposr_import(self):
        from src.engines.triposr_engine import TriposrEngine
        engine = TriposrEngine(device="cpu")
        assert engine.device == "cpu"

    def test_lgm_import(self):
        from src.engines.lgm_engine import LGMEngine
        engine = LGMEngine(device="cpu")
        assert engine.device == "cpu"

    def test_trellis_import(self):
        from src.engines.trellis_engine import TrellisEngine
        engine = TrellisEngine(device="cpu")
        assert engine.device == "cpu"

    def test_spar3d_import(self):
        from src.engines.spar3d_engine import Spar3dEngine
        engine = Spar3dEngine(device="cpu")
        assert engine.device == "cpu"

    def test_lazy_init(self):
        """Engines should not load models on __init__."""
        from src.engines.triposr_engine import TriposrEngine
        engine = TriposrEngine(device="cpu")
        assert engine.model is None


# =============================================================================
# EVALUATION RUNNER TESTS
# =============================================================================

class TestBenchmarkRunner:
    """Test BenchmarkRunner with mock engines."""

    def test_runner_init(self):
        from src.evaluation.benchmark_runner import BenchmarkRunner
        runner = BenchmarkRunner(
            dataset_path="/tmp/fake",
            engines=[],
            num_samples=10,
            output_dir="/tmp/test_outputs",
        )
        assert runner.num_samples == 10

    def test_empty_engines(self):
        """Runner with no engines should handle gracefully."""
        from src.evaluation.benchmark_runner import BenchmarkRunner
        runner = BenchmarkRunner(
            dataset_path="/tmp/fake",
            engines=[],
            num_samples=10,
            output_dir="/tmp/test_outputs",
        )
        # Should not crash on init
        assert runner.engines == []


# =============================================================================
# BENCHMARK REGRESSION TESTS (CI-04)
# =============================================================================

class TestBenchmarkRegression:
    """Regression tests to ensure benchmark produces consistent results."""

    def test_benchmark_runner_returns_dataframe(self):
        """BenchmarkRunner.get_summary_df should return a DataFrame."""
        import pandas as pd
        from src.evaluation.benchmark_runner import BenchmarkRunner
        runner = BenchmarkRunner(
            dataset_path="/tmp/fake",
            engines=[],
            num_samples=10,
            output_dir="/tmp/test_outputs",
        )
        df = runner.get_summary_df()
        assert isinstance(df, pd.DataFrame)

    def test_benchmark_runner_empty_results(self):
        """Runner with no results should return empty DataFrame."""
        from src.evaluation.benchmark_runner import BenchmarkRunner
        runner = BenchmarkRunner(
            dataset_path="/tmp/fake",
            engines=[],
            num_samples=10,
            output_dir="/tmp/test_outputs",
        )
        df = runner.get_summary_df()
        assert df.empty

    def test_benchmark_runner_export_latex(self):
        """export_latex_table should return a string with table markup."""
        from src.evaluation.benchmark_runner import BenchmarkRunner
        runner = BenchmarkRunner(
            dataset_path="/tmp/fake",
            engines=[],
            num_samples=10,
            output_dir="/tmp/test_outputs",
        )
        latex = runner.export_latex_table()
        assert isinstance(latex, str)
        assert "tabular" in latex

    def test_benchmark_runner_save_results(self):
        """save_results should write a JSON file."""
        import json
        import tempfile
        from src.evaluation.benchmark_runner import BenchmarkRunner
        runner = BenchmarkRunner(
            dataset_path="/tmp/fake",
            engines=[],
            num_samples=10,
            output_dir="/tmp/test_outputs",
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            runner.save_results(path)
            with open(path) as f:
                data = json.load(f)
            assert isinstance(data, list)
        finally:
            os.unlink(path)


# =============================================================================
# DATA CONTRACT TESTS (DATA-10)
# =============================================================================

class TestResultsContract:
    """Validate results CSV schema."""

    @pytest.fixture
    def results_csv(self):
        csv_path = Path(__file__).parent.parent / "results" / "unified_results_4engines_100obj.csv"
        if not csv_path.exists():
            pytest.skip("Results CSV not found")
        import pandas as pd
        return pd.read_csv(csv_path)

    def test_row_count(self, results_csv):
        assert len(results_csv) == 400, f"Expected 400 rows, got {len(results_csv)}"

    def test_engine_coverage(self, results_csv):
        engines = set(results_csv["engine"].unique())
        expected = {"TripoSR", "LGM", "TRELLIS", "SPAR3D"}
        assert engines == expected

    def test_required_columns(self, results_csv):
        required = [
            "engine", "sample_id", "latency_s", "vram_peak_gb",
            "chamfer_distance", "f_score_0.1", "f_score_0.2", "f_score_0.5",
            "psnr", "lpips"
        ]
        for col in required:
            assert col in results_csv.columns, f"Missing column: {col}"

    def test_no_nan_in_critical_columns(self, results_csv):
        for col in ["chamfer_distance", "f_score_0.1", "f_score_0.2", "psnr"]:
            assert not results_csv[col].isna().any(), f"NaN in {col}"

    def test_sample_id_format(self, results_csv):
        """Sample IDs should be non-empty strings."""
        assert results_csv["sample_id"].str.len().min() > 0

    def test_latency_positive(self, results_csv):
        assert (results_csv["latency_s"] > 0).all()

    def test_vram_positive(self, results_csv):
        assert (results_csv["vram_peak_gb"] > 0).all()


# =============================================================================
# EXPORT TESTS (EXP-14)
# =============================================================================

class TestExport:
    """Test mesh export functionality."""

    def test_glb_export_creates_file(self):
        """GLB export should create a valid file."""
        import trimesh
        tm = trimesh.creation.box()
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
            path = f.name
        try:
            tm.export(path)
            assert Path(path).exists()
            assert Path(path).stat().st_size > 0
        finally:
            os.unlink(path)

    def test_ply_export_creates_file(self):
        """PLY export should create a valid file."""
        import trimesh
        tm = trimesh.creation.box()
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            path = f.name
        try:
            tm.export(path)
            assert Path(path).exists()
            assert Path(path).stat().st_size > 0
        finally:
            os.unlink(path)


# =============================================================================
# CONFIG TESTS
# =============================================================================

class TestConfig:
    """Test configuration loading."""

    def test_config_exists(self):
        config_path = Path(__file__).parent.parent / "src" / "config.py"
        assert config_path.exists()

    def test_pyproject_exists(self):
        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        assert pyproject.exists()


# =============================================================================
# RUN AS MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])