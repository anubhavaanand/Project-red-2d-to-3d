"""Tests for the shared object-id loader (DATA-01)."""

from pathlib import Path

from src.data.object_ids import load_object_ids


class TestParsing:
    def test_reads_the_real_file(self):
        """The shipped handpicked.txt must yield 100 clean IDs."""
        ids = load_object_ids(Path(__file__).parent.parent / "handpicked.txt")
        assert len(ids) == 100
        assert not any(i.startswith("#") for i in ids)

    def test_skips_comments(self, tmp_path):
        p = tmp_path / "ids.txt"
        p.write_text("# a comment\nobj_a\nobj_b\n")
        assert load_object_ids(p) == ["obj_a", "obj_b"]

    def test_skips_indented_comments(self, tmp_path):
        """DATA-01: the runner checked the unstripped line, so these leaked."""
        p = tmp_path / "ids.txt"
        p.write_text("   # indented comment\nobj_a\n")
        assert load_object_ids(p) == ["obj_a"]

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "ids.txt"
        p.write_text("\n\nobj_a\n\n")
        assert load_object_ids(p) == ["obj_a"]

    def test_preserves_order(self, tmp_path):
        p = tmp_path / "ids.txt"
        p.write_text("z\na\nm\n")
        assert load_object_ids(p) == ["z", "a", "m"]

    def test_deduplicates_first_wins(self, tmp_path):
        p = tmp_path / "ids.txt"
        p.write_text("a\nb\na\n")
        assert load_object_ids(p) == ["a", "b"]

    def test_strips_whitespace(self, tmp_path):
        p = tmp_path / "ids.txt"
        p.write_text("  obj_a  \n")
        assert load_object_ids(p) == ["obj_a"]

    def test_empty_file(self, tmp_path):
        p = tmp_path / "ids.txt"
        p.write_text("")
        assert load_object_ids(p) == []


class TestSingleImplementation:
    def test_selector_uses_it(self):
        src = (ENGINES := Path(__file__).parent.parent / "gso1000" / "select_1000.py").read_text()
        assert "load_object_ids(handpicked_path)" in src
        assert "[l.strip() for l in handpicked_path" not in src

    def test_runner_uses_it(self):
        src = (
            Path(__file__).parent.parent / "src" / "evaluation" / "benchmark_runner.py"
        ).read_text()
        assert "object_ids = load_object_ids(handpicked_path)" in src
        # the old unstripped-line check must be gone
        assert 'line.startswith("#")' not in src

    def test_selector_removed_its_local_parser(self):
        src = (Path(__file__).parent.parent / "gso1000" / "select_1000.py").read_text()
        assert "def load_forced" in src  # still exists, but delegates
        assert ".splitlines() if l.strip()" not in src
