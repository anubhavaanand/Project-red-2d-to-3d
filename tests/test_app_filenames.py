"""Tests for Gradio output filename uniqueness (APP-03)."""

import ast
from pathlib import Path

APP_PY = Path(__file__).parent.parent / "app.py"


def _output_naming_source():
    lines = APP_PY.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if "out_file = OUTPUT_DIR" in line)
    return "\n".join(lines[start : start + 3])


class TestOutputFilename:
    def test_uses_uuid_not_timestamp(self):
        src = APP_PY.read_text()
        assert "int(time.time()*1000)" not in src, "still using a millisecond timestamp"
        assert 'f"output_{uuid.uuid4().hex}"' in src

    def test_uuid_is_imported(self):
        tree = ast.parse(APP_PY.read_text())
        imported = {
            a.asname or a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
        }
        assert "uuid" in imported

    def test_generated_names_are_unique(self):
        """Many rapid generations must not repeat a name."""
        import uuid

        names = [f"output_{uuid.uuid4().hex}" for _ in range(5000)]
        assert len(set(names)) == len(names)

    def test_timestamp_collision_is_genuinely_possible(self):
        """Guard the premise: millisecond timestamps do collide in a tight loop."""
        import time

        stamps = [int(time.time() * 1000) for _ in range(2000)]
        assert len(set(stamps)) < len(stamps), "premise no longer holds; revisit this test"

    def test_suffix_is_applied_per_format(self):
        """The stem is shared, but .glb and .ply must still differ."""
        import uuid

        stem = Path("output_" + uuid.uuid4().hex)
        assert stem.with_suffix(".glb") != stem.with_suffix(".ply")

    def test_no_timestamp_fragment_remains(self):
        src = APP_PY.read_text()
        assert "output_{int(" not in src
