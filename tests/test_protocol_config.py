"""Tests for the frozen protocol configuration (EVAL-40)."""

import ast
import dataclasses
from pathlib import Path

import pytest

RUNNER = Path(__file__).parent.parent / "src" / "evaluation" / "benchmark_runner.py"


def _protocol_config():
    """Exec just the ProtocolConfig dataclass."""
    src = RUNNER.read_text()
    tree = ast.parse(src)
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ProtocolConfig":
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            getattr(t, "id", "") == "DEFAULT_PROTOCOL" for t in node.targets
        ):
            nodes.append(node)
    ns = {
        "dataclass": __import__("dataclasses").dataclass,
        "field": __import__("dataclasses").field,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(RUNNER), "exec"), ns)
    return ns["ProtocolConfig"], ns["DEFAULT_PROTOCOL"]


class TestDefaults:
    def test_defaults_match_the_published_protocol(self):
        """EVAL-40 must not change the frozen numbers."""
        PC, default = _protocol_config()
        assert default.num_surface_points == 10000
        assert default.icp_iterations == 5
        assert default.alignment_yaw_steps == 12
        assert default.visual_yaw_steps == 24

    def test_is_frozen(self):
        PC, default = _protocol_config()
        with pytest.raises(dataclasses.FrozenInstanceError):
            default.icp_iterations = 9

    def test_configurable(self):
        PC, _ = _protocol_config()
        custom = PC(icp_iterations=3, alignment_yaw_steps=6)
        assert custom.icp_iterations == 3
        assert custom.alignment_yaw_steps == 6
        # defaults untouched
        assert PC().icp_iterations == 5


class TestWiring:
    def test_align_uses_the_protocol_default(self):
        src = RUNNER.read_text()
        assert "protocol: ProtocolConfig = DEFAULT_PROTOCOL" in src

    def test_align_resolves_none_to_the_protocol(self):
        src = RUNNER.read_text()
        assert "if num_points is None:" in src
        assert "num_points = protocol.num_surface_points" in src
        assert "if n_iter is None:" in src
        assert "n_iter = protocol.icp_iterations" in src

    def test_yaw_counts_come_from_the_protocol(self):
        src = RUNNER.read_text()
        assert "_candidate_rotations(protocol.alignment_yaw_steps)" in src
        assert "_upright_yaw_rotations(protocol.visual_yaw_steps)" in src

    def test_hardcoded_yaw_counts_are_gone(self):
        src = RUNNER.read_text()
        assert "_candidate_rotations(12)" not in src
        assert "_upright_yaw_rotations(24)" not in src

    def test_runner_exposes_the_protocol(self):
        src = RUNNER.read_text()
        assert "protocol: ProtocolConfig = DEFAULT_PROTOCOL" in src
        assert "self.protocol = protocol" in src
        assert "protocol=self.protocol" in src


class TestSourceStructure:
    def test_explicit_arguments_still_win(self):
        """Passing num_points/n_iter explicitly must override the protocol."""
        src = RUNNER.read_text()
        body = src[src.index("def align_mesh_to_gt") :]
        assert "if num_points is None:" in body[:2000]

    def test_engine_parses(self):
        ast.parse(RUNNER.read_text())
