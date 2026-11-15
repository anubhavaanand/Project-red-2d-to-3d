"""Tests for the normalisation-centre documentation (EVAL-04).

EVAL-04 is methodology: changing the normalisation would move PSNR/LPIPS, so
the fix documents the two-stage rule and adds a render-scale guard rather than
changing the numbers.
"""

from pathlib import Path

RUNNER = Path(__file__).parent.parent / "src" / "evaluation" / "benchmark_runner.py"
README = Path(__file__).parent.parent / "README.md"
METRICS = Path(__file__).parent.parent / "src" / "evaluation" / "metrics.py"


class TestTwoNormalisers:
    def test_vertices_use_bbox_centre(self):
        src = RUNNER.read_text()
        fn = src[
            src.index("def _normalize_vertices_robust") : src.index("def _candidate_rotations")
        ]
        assert "bbox_center = 0.5 * (v.min(axis=0) + v.max(axis=0))" in fn

    def test_points_use_mean_centre(self):
        src = METRICS.read_text()
        fn = src[src.index("def normalize_point_cloud") : src.index("__all__")]
        assert "centroid = pts.mean(axis=0)" in fn

    def test_both_use_the_995th_percentile_radius(self):
        verts_fn = RUNNER.read_text()
        pts_fn = METRICS.read_text()
        assert verts_fn.count("np.percentile(radii, 99.5)") == 1
        assert pts_fn.count("np.percentile(radii, 99.5)") == 1

    def test_centres_genuinely_differ(self):
        """Pins why this matters: bbox centre != mean for an asymmetric cloud."""
        import numpy as np

        pts = np.array(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [10, 0, 0]],
            dtype=np.float32,
        )
        bbox_centre = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
        mean_centre = pts.mean(axis=0)
        assert not np.allclose(bbox_centre, mean_centre)


class TestRenderScaleGuard:
    def test_guard_exists(self):
        src = RUNNER.read_text()
        assert (
            "render_radius = float(np.linalg.norm(verts - verts.mean(axis=0), axis=1).max())" in src
        )

    def test_guard_warns_on_mis_scale(self):
        src = ENGINE_SRC()
        assert "outside the unit sphere" in src
        assert "render_radius > 10.0" in src

    def test_guard_is_inside_the_render_function(self):
        src = ENGINE_SRC()
        fn = src[src.index("def render_novel_view_same_bg") : src.index("def render_gso_mesh_view")]
        assert "render_radius" in fn


class TestDocumentation:
    def test_readme_documents_the_two_stage_rule(self):
        text = README.read_text()
        assert "two-stage rule" in text

    def test_readme_names_both_conventions(self):
        text = README.read_text()
        assert "bounding-box centre" in text
        assert "centroid convention" in text

    def test_readme_keeps_the_sampling_line(self):
        text = README.read_text()
        assert "10,000 surface points per mesh" in text


class TestNoBehaviourChange:
    def test_normalisation_functions_unchanged(self):
        """The fix must not alter either normaliser."""
        verts_fn = ENGINE_SRC()[
            ENGINE_SRC().index("def _normalize_vertices_robust") : ENGINE_SRC().index(
                "def _candidate_rotations"
            )
        ]
        assert "vc = v - bbox_center" in verts_fn
        assert "return vc / (scale + 1e-8), bbox_center, scale" in verts_fn

    def test_point_normaliser_unchanged(self):
        src = METRICS.read_text()
        fn = src[src.index("def normalize_point_cloud") : src.index("__all__")]
        assert "return (centered / (scale + 1e-8)).astype(np.float32)" in fn


def ENGINE_SRC():
    return RUNNER.read_text()
