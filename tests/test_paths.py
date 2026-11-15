"""Tests for safe sys.path handling of vendored repos (SEC-16)."""

import sys

from src.engines._paths import extend_sys_path


class TestExtendSysPath:
    def test_appends_instead_of_prepending(self, tmp_path):
        """A vendored dir must not take priority over installed packages."""
        before = list(sys.path)
        try:
            assert extend_sys_path(tmp_path) is True
            assert sys.path[-1] == str(tmp_path)
            # must be at the end, not the front
            assert sys.path[0] == before[0]
        finally:
            sys.path[:] = before

    def test_missing_directory_is_skipped(self, tmp_path):
        missing = tmp_path / "does-not-exist"
        before = list(sys.path)
        try:
            assert extend_sys_path(missing) is False
            assert sys.path == before
        finally:
            sys.path[:] = before

    def test_first_existing_candidate_wins(self, tmp_path):
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        before = list(sys.path)
        try:
            assert extend_sys_path(tmp_path / "nope", first, second) is True
            assert str(first) in sys.path
            assert str(second) not in sys.path
        finally:
            sys.path[:] = before

    def test_idempotent(self, tmp_path):
        before = list(sys.path)
        try:
            extend_sys_path(tmp_path)
            extend_sys_path(tmp_path)
            assert sys.path.count(str(tmp_path)) == 1
        finally:
            sys.path[:] = before

    def test_no_candidates_returns_false(self):
        before = list(sys.path)
        try:
            assert extend_sys_path() is False
            assert sys.path == before
        finally:
            sys.path[:] = before

    def test_file_path_is_not_added(self, tmp_path):
        """Only directories belong on sys.path."""
        a_file = tmp_path / "not-a-dir.txt"
        a_file.write_text("x")
        before = list(sys.path)
        try:
            assert extend_sys_path(a_file) is False
            assert str(a_file) not in sys.path
        finally:
            sys.path[:] = before

    def test_installed_package_keeps_priority(self, tmp_path):
        """The regression this guards: a vendored dir must not shadow stdlib."""
        vendored = tmp_path / "vendored"
        vendored.mkdir()
        # a planted json.py that would break stdlib json if it won
        (vendored / "json.py").write_text("raise RuntimeError('shadowed!')\n")

        before = list(sys.path)
        try:
            extend_sys_path(vendored)
            # stdlib json still imports from its real location
            import json

            assert json.loads("{}") == {}
            assert not json.__file__.startswith(str(vendored))
        finally:
            sys.path[:] = before
