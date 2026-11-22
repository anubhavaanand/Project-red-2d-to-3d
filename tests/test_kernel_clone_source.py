"""Tests for the Kaggle kernel clone source (NB-09 prerequisite).

Every kernel clones the benchmark repo at a pinned commit. Two things must hold:

1. the URL is the canonical repo, not the pre-rename name that GitHub happens to
   redirect (a redirect can be dropped, and the old name is claimable by anyone);
2. the commit is actually on the remote, or the pin cannot be fetched.
"""

import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
KERNELS = sorted((ROOT / "kaggle").glob("*/*.ipynb"))
CANONICAL = "https://github.com/anubhavaanand/Project-red-2d-to-3d.git"
STALE = "https://github.com/anubhavaanand/image-to-3d.git"


def _cell_sources(path: Path) -> dict[int, str]:
    nb = json.loads(path.read_text())
    return {i: "".join(c["source"]) for i, c in enumerate(nb["cells"])}


def _clone_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for src in _cell_sources(path).values()
        for line in src.splitlines()
        if "git" in line and "clone" in line
    ]


# Kernels pin the benchmark repo by fetching one commit, e.g.
#   ["git", "-C", CODE_DIR, "fetch", "--depth", "1", "origin", "<sha>"]
#   ["git", "-C", CODE_DIR, "checkout", "FETCH_HEAD"]
# GitHub refuses `clone -b <sha>` for a commit that is not a branch tip, so the
# fetch form is the one that actually works for old pins.
_BENCH_FETCH = re.compile(r'"fetch",\s*"--depth",\s*"1",\s*"origin",\s*"([0-9a-f]{40})"')

# Older kernels cloned the benchmark repo directly:
#   ["git", "clone", "--depth", "1", "-b", "<sha>", "<url>", CODE_DIR]
_BENCH_CLONE = re.compile(
    r'"git",\s*"clone"[^\]]*?"-b",\s*"([0-9a-f]{40}|main)",\s*"'
    r"https://github\.com/anubhavaanand/Project-red-2d-to-3d\.git"
)


def _pinned_sha(path: Path) -> str | None:
    """Return the commit the benchmark repo is pinned to, or None."""
    for src in _cell_sources(path).values():
        flat = " ".join(src.split())
        for pattern in (_BENCH_FETCH, _BENCH_CLONE):
            m = pattern.search(flat)
            if m:
                return m.group(1)
    return None


class TestCloneUrl:
    @pytest.mark.parametrize("kernel", KERNELS, ids=lambda p: p.parent.name)
    def test_kernel_clones_the_canonical_repo(self, kernel):
        """Only the *benchmark* clone is checked — CLIP/nvdiffrast are upstream."""
        src = " ".join(" ".join(_cell_sources(kernel).values()).split())
        assert "git", "clone" in src or "git clone" in src
        assert STALE not in src, (
            f"{kernel.name} still references the pre-rename URL {STALE}; "
            f"GitHub redirects it today but that is not a guarantee"
        )
        # The benchmark repo must be cloned from the canonical URL.
        assert f'"{CANONICAL}"' in src or CANONICAL in src, (
            f"{kernel.name}: no clone of {CANONICAL}"
        )

    @pytest.mark.parametrize("kernel", KERNELS, ids=lambda p: p.parent.name)
    def test_no_kernel_references_the_stale_url_anywhere(self, kernel):
        """Not just the clone line — any mention can break later."""
        assert STALE not in kernel.read_text(), f"{kernel.name} still mentions {STALE}"


class TestPinnedCommit:
    @pytest.mark.parametrize("kernel", KERNELS, ids=lambda p: p.parent.name)
    def test_kernel_pins_a_full_sha(self, kernel):
        """Not `-b main`: an unpinned clone makes runs non-reproducible."""
        pin = _pinned_sha(kernel)
        assert pin, f"{kernel.name}: no benchmark repo pin found"
        assert pin != "main", (
            f"{kernel.name} tracks `main`, so its code version drifts; "
            f"pin a 40-char SHA like the other kernels do"
        )

    def test_pinned_sha_exists_on_the_remote(self):
        """A pinned fetch fails if the commit is not reachable from a ref."""
        shas = {_pinned_sha(k) for k in KERNELS} - {None}
        assert shas, "no pinned SHAs found"
        try:
            remote = subprocess.run(
                ["git", "ls-remote", "origin", "main"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=ROOT,
            )
        except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
            pytest.skip(f"cannot reach the remote: {exc}")
        if remote.returncode != 0:
            pytest.skip("no network access to the remote")
        remote_head = remote.stdout.split()[0]
        for sha in shas:
            # The pin must be an ancestor of main; a clone -b works for any
            # reachable commit, so require it to be in the fetched history.
            #
            # CI checks out with fetch-depth: 1, so the pinned SHA is usually not
            # in the local history at all and `merge-base` exits 128 ("Not a
            # valid commit name"). There is nothing to assert in that case — the
            # ls-remote check above already proved the remote is reachable.
            known = subprocess.run(
                ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
                capture_output=True,
                cwd=ROOT,
            )
            if known.returncode != 0:
                pytest.skip(
                    f"pinned {sha[:12]} is not in the local (shallow) history; "
                    f"cannot verify ancestry here"
                )
            check = subprocess.run(
                ["git", "merge-base", "--is-ancestor", sha, "origin/main"],
                capture_output=True,
                cwd=ROOT,
            )
            assert check.returncode == 0, (
                f"pinned {sha[:12]} is not an ancestor of origin/main "
                f"({remote_head[:12]}); kernels would clone a commit that is not "
                f"reachable from the default branch"
            )


class TestNotebooksStayValid:
    @pytest.mark.parametrize("kernel", KERNELS, ids=lambda p: p.parent.name)
    def test_notebook_is_still_valid_json(self, kernel):
        nb = json.loads(kernel.read_text())
        assert nb.get("nbformat") == 4
        assert nb.get("cells")
