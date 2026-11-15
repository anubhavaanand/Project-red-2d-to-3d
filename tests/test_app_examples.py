"""Tests for CWD-independent example paths (APP-07)."""

import ast
import os
from pathlib import Path

APP_PY = Path(__file__).parent.parent / "app.py"


def _examples_block():
    lines = APP_PY.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("EXAMPLES_DIR ="))
    end = next(i for i in range(start, len(lines)) if lines[i].strip() == "]")
    return "\n".join(lines[start : end + 1])


class TestExamplePaths:
    def test_examples_dir_is_anchored_to_the_file(self):
        src = _examples_block()
        assert 'Path(__file__).resolve().parent / "examples"' in src

    def test_examples_are_absolute(self):
        ns = {"Path": Path, "__file__": str(APP_PY)}
        exec(_examples_block(), ns)
        for entry in ns["EXAMPLES"]:
            p = Path(entry[0])
            assert p.is_absolute(), f"{p} is not absolute"

    def test_examples_do_not_depend_on_cwd(self):
        """Resolving the block from a different CWD must give the same paths."""
        ns = {"Path": Path, "__file__": str(APP_PY)}
        exec(_examples_block(), ns)
        first = [e[0] for e in ns["EXAMPLES"]]

        cwd = os.getcwd()
        try:
            os.chdir("/")
            ns2 = {"Path": Path, "__file__": str(APP_PY)}
            exec(_examples_block(), ns2)
            assert [e[0] for e in ns2["EXAMPLES"]] == first
        finally:
            os.chdir(cwd)

    def test_example_files_exist(self):
        ns = {"Path": Path, "__file__": str(APP_PY)}
        exec(_examples_block(), ns)
        for entry in ns["EXAMPLES"]:
            assert Path(entry[0]).exists(), f"missing example: {entry[0]}"

    def test_no_relative_examples_literal_remains(self):
        src = APP_PY.read_text()
        assert '"examples/' not in src

    def test_dir_is_created_in_build_ui(self):
        """APP-07: creation must happen in build_ui, not only in __main__."""
        tree = ast.parse(APP_PY.read_text())
        build_ui = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_ui"
        )
        body = ast.dump(build_ui)
        assert "EXAMPLES_DIR" in body and "mkdir" in body

    def test_main_block_no_longer_creates_the_dir(self):
        """The old __main__ mkdir would hide the bug for direct runs only."""
        src = APP_PY.read_text()
        main_idx = src.index('if __name__ == "__main__":')
        main_block = src[main_idx:]
        assert 'Path("examples").mkdir' not in main_block

    def test_app_parses(self):
        ast.parse(APP_PY.read_text())