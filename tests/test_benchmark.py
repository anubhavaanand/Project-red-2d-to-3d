"""
Test suite for the image-to-3d-benchmark project.
Covers: metrics, engines (import/smoke), evaluation runner, export, data contracts.
"""

import io
import importlib.util
import json
import os
import tempfile
from pathlib import Path

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
    """Test PSNR computation (requires torch)."""

    pytestmark = pytest.mark.skipif(
        importlib.util.find_spec("torch") is None,
        reason="torch not installed",
    )

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
    """Verify all engines can be imported (requires torch)."""

    pytestmark = pytest.mark.skipif(
        importlib.util.find_spec("torch") is None,
        reason="torch not installed",
    )

    """Verify all engines can be imported without CUDA."""

    @pytest.mark.gpu
    def test_triposr_import(self):
        from src.engines.triposr_engine import TriposrEngine
        engine = TriposrEngine(device="cpu")
        assert engine.device == "cpu"

    @pytest.mark.gpu
    def test_lgm_import(self):
        from src.engines.lgm_engine import LGMEngine
        engine = LGMEngine(device="cpu")
        assert engine.device == "cpu"

    @pytest.mark.gpu
    def test_trellis_import(self):
        from src.engines.trellis_engine import TrellisEngine
        engine = TrellisEngine(device="cpu")
        assert engine.device == "cpu"

    @pytest.mark.gpu
    def test_spar3d_import(self):
        from src.engines.spar3d_engine import Spar3dEngine
        engine = Spar3dEngine(device="cpu")
        assert engine.device == "cpu"

    @pytest.mark.gpu
    def test_lazy_init(self):
        """Engines should not load models on __init__."""
        from src.engines.triposr_engine import TriposrEngine
        engine = TriposrEngine(device="cpu")
        assert engine.model is None


# =============================================================================
# EVALUATION RUNNER TESTS
# =============================================================================

class TestBenchmarkRunner:
    """BenchmarkRunner construction (requires torch)."""

    pytestmark = pytest.mark.skipif(
        importlib.util.find_spec("torch") is None,
        reason="torch not installed",
    )

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
# BENCHMARK REGRESSION TESTS (CI-04) hehe
# =============================================================================

class TestBenchmarkRegression:
    """Regression tests (requires torch)."""

    pytestmark = pytest.mark.skipif(
        importlib.util.find_spec("torch") is None,
        reason="torch not installed",
    )

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


# API SMOKE TESTS (CI-07)
# =============================================================================

class TestAPISmoke:
    """Smoke tests for the FastAPI server endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client for the API."""
        try:
            from fastapi.testclient import TestClient

            from src.api.server import app
            return TestClient(app)
        except ImportError:
            pytest.skip("fastapi/testclient not installed")

    def test_health_check(self, client):
        """Health endpoint should return 200 and healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_list_engines(self, client):
        """Engines endpoint should return list of 3 engines."""
        response = client.get("/engines")
        assert response.status_code == 200
        engines = response.json()
        assert len(engines) == 3
        engine_ids = [e["id"] for e in engines]
        assert "triposr" in engine_ids
        assert "lgm" in engine_ids
        assert "trellis" in engine_ids

    def test_predict_missing_file(self, client):
        """Predict endpoint should return 400/422 without file."""
        response = client.post("/predict", data={"engine_type": "triposr"})
        assert response.status_code in (400, 422)

    def test_predict_invalid_engine(self, client):
        """Predict endpoint should return 400 for invalid engine."""
        import io
        fake_image = io.BytesIO(b"fake image data")
        response = client.post(
            "/predict",
            files={"file": ("test.png", fake_image, "image/png")},
            data={"engine_type": "invalid_engine"},
        )
        assert response.status_code in (400, 422)


# =============================================================================
# NOTEBOOK SMOKE TESTS (CI-10)
# =============================================================================

class TestNotebookStructure:
    """Validate Kaggle notebook structure without executing them."""

    @pytest.fixture
    def batch1_notebook(self):
        path = Path(__file__).parent.parent / "kaggle" / "batch1" / "kernel_batch1_triposr_lgm.ipynb"
        if not path.exists():
            pytest.skip("Batch 1 notebook not found")
        with open(path) as f:
            return json.load(f)

    @pytest.fixture
    def batch2_notebook(self):
        path = Path(__file__).parent.parent / "kaggle" / "batch2" / "kernel_batch2_trellis_spar3d.ipynb"
        if not path.exists():
            pytest.skip("Batch 2 notebook not found")
        with open(path) as f:
            return json.load(f)

    def test_notebook_valid_json(self, batch1_notebook, batch2_notebook):
        """Notebooks should be valid JSON with required fields."""
        for nb in [batch1_notebook, batch2_notebook]:
            assert "cells" in nb
            assert "metadata" in nb
            assert "nbformat" in nb
            assert isinstance(nb["cells"], list)
            assert len(nb["cells"]) > 0

    def test_notebook_has_code_cells(self, batch1_notebook, batch2_notebook):
        """Notebooks should have code cells."""
        for nb in [batch1_notebook, batch2_notebook]:
            code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
            assert len(code_cells) > 0

    def test_notebook_code_cells_valid_python(self, batch1_notebook, batch2_notebook):
        """All code cells should contain valid Python syntax.

        IPython magics and shell escapes (``!pip install``, ``%matplotlib``) are
        not valid Python, so each line is replaced by ``pass`` before parsing.
        """
        import ast
        for nb in [batch1_notebook, batch2_notebook]:
            for i, cell in enumerate(nb["cells"]):
                if cell["cell_type"] != "code":
                    continue
                lines = []
                for raw in cell["source"]:
                    stripped = raw.lstrip()
                    if stripped.startswith(("!", "%")):
                        # keep indentation so the block structure survives
                        indent = raw[: len(raw) - len(stripped)]
                        lines.append(f"{indent}pass\n")
                    else:
                        lines.append(raw)
                src = "".join(lines)
                try:
                    ast.parse(src)
                except SyntaxError as e:
                    pytest.fail(f"Syntax error in cell {i}: {e}")

    def test_notebook_has_markdown_cells(self, batch1_notebook, batch2_notebook):
        """Notebooks should have markdown cells for documentation."""
        for nb in [batch1_notebook, batch2_notebook]:
            md_cells = [c for c in nb["cells"] if c["cell_type"] == "markdown"]
            assert len(md_cells) > 0

    def test_batch1_has_required_engines(self, batch1_notebook):
        """Batch 1 notebook should reference TripoSR and LGM."""
        all_source = " ".join(
            "".join(c["source"]) for c in batch1_notebook["cells"]
        )
        assert "TripoSR" in all_source or "triposr" in all_source
        assert "LGM" in all_source or "lgm" in all_source

    def test_batch2_has_required_engines(self, batch2_notebook):
        """Batch 2 notebook should reference TRELLIS and SPAR3D."""
        all_source = " ".join(
            "".join(c["source"]) for c in batch2_notebook["cells"]
        )
        assert "TRELLIS" in all_source or "trellis" in all_source
        assert "SPAR3D" in all_source or "spar3d" in all_source

# =============================================================================
# MODEL WEIGHT PROVENANCE (SEC-12)
# =============================================================================

class TestModelWeightProvenance:
    """Pinned SHA-256 verification for implicitly downloaded weights."""

    def _asset(self, tmp_path, name="model.bin", content=b"hello"):
        from src.model_weights import Asset
        path = tmp_path / name
        path.write_bytes(content)
        import hashlib
        return Asset(path=path, url="https://example.invalid/" + name,
                     sha256=hashlib.sha256(content).hexdigest())

    def test_matching_hash_returns_path(self, tmp_path):
        from src.model_weights import ensure
        a = self._asset(tmp_path)
        assert ensure(a) == a.path

    def test_corrupt_cache_raises(self, tmp_path):
        from src.model_weights import ChecksumMismatchError, ensure
        a = self._asset(tmp_path, content=b"good")
        a.path.write_bytes(b"tampered")
        with pytest.raises(ChecksumMismatchError):
            ensure(a)

    def test_download_verifies_and_replaces(self, tmp_path):
        import hashlib

        from src.model_weights import Asset, ensure
        dest = tmp_path / "sub" / "w.bin"
        a = Asset(path=dest, url="https://example.invalid/w.bin",
                  sha256=hashlib.sha256(b"payload").hexdigest())

        class _Resp:
            def __init__(self, data): self._b = io.BytesIO(data)
            def read(self, n=-1): return self._b.read(n)
            def __enter__(self): return self
            def __exit__(self, *a): return False

        import src.model_weights as mw
        orig = mw.urllib.request.urlopen
        try:
            mw.urllib.request.urlopen = lambda url: _Resp(b"payload")
            assert ensure(a) == dest
        finally:
            mw.urllib.request.urlopen = orig
        assert dest.read_bytes() == b"payload"
        assert not dest.with_name(dest.name + ".part").exists()

    def test_download_mismatch_leaves_nothing(self, tmp_path):
        import hashlib

        from src.model_weights import Asset, ChecksumMismatchError, ensure
        dest = tmp_path / "w.bin"
        a = Asset(path=dest, url="https://example.invalid/w.bin",
                  sha256=hashlib.sha256(b"expected").hexdigest())

        class _Resp:
            def __init__(self, data): self._b = io.BytesIO(data)
            def read(self, n=-1): return self._b.read(n)
            def __enter__(self): return self
            def __exit__(self, *a): return False

        import src.model_weights as mw
        orig = mw.urllib.request.urlopen
        try:
            mw.urllib.request.urlopen = lambda url: _Resp(b"evil")
            with pytest.raises(ChecksumMismatchError):
                ensure(a)
        finally:
            mw.urllib.request.urlopen = orig
        assert not dest.exists()
        assert not dest.with_name(dest.name + ".part").exists()

    def test_pins_exist_for_required_assets(self):
        from src.model_weights import asset
        for name in ("isnet-general-use.onnx", "vgg16-397923af.pth"):
            assert len(asset(name).sha256) == 64

    def test_unknown_asset_rejected(self):
        from src.model_weights import asset
        with pytest.raises(KeyError):
            asset("no-such-weights.bin")


# =============================================================================
# RUN AS MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
