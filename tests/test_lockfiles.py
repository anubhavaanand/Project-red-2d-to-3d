"""Tests for the hashed lockfile setup (PKG-15)."""

import re
from pathlib import Path

import pytest

# tomllib is 3.11+; the project supports 3.10, so fall back to the backport that
# pytest already pulls in there (pytest>=8 requires tomli on 3.10).
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).parent.parent
LOCKS = ROOT / "requirements"
MAKEFILE = ROOT / "Makefile"
PYPROJECT = ROOT / "pyproject.toml"
README = ROOT / "README.md"

CORE_LOCK = LOCKS / "lock-core.txt"
LOCKS_PRESENT = [CORE_LOCK]


def _extras() -> set[str]:
    data = tomllib.loads(PYPROJECT.read_text())
    return set(data["project"]["optional-dependencies"])


class TestLockfilesExist:
    @pytest.mark.parametrize("lock", LOCKS_PRESENT, ids=lambda p: p.name)
    def test_lockfile_is_committed(self, lock):
        assert lock.exists(), f"{lock.name} is missing; run `make lock`"

    @pytest.mark.parametrize("lock", LOCKS_PRESENT, ids=lambda p: p.name)
    def test_lockfile_records_its_command(self, lock):
        """The regeneration command must be in the header, or the lock rots."""
        head = lock.read_text().splitlines()[:3]
        assert any("uv pip compile" in line for line in head), head

    @pytest.mark.parametrize("lock", LOCKS_PRESENT, ids=lambda p: p.name)
    def test_every_pin_carries_a_hash(self, lock):
        """An unhashed lockfile is just requirements.txt with extra steps."""
        text = lock.read_text()
        pins = re.findall(r"^([A-Za-z0-9_.\-]+)==(\S+)", text, re.MULTILINE)
        assert pins, "no pinned packages found"
        hashes = re.findall(r"--hash=sha256:[0-9a-f]{64}", text)
        assert len(hashes) >= len(pins), (
            f"{lock.name}: {len(pins)} pins but only {len(hashes)} hashes"
        )


class TestLockCoverage:
    def test_core_lock_covers_the_base_dependencies(self):
        """Every mandatory dependency must appear in the core lock."""
        data = tomllib.loads(PYPROJECT.read_text())
        base = data["project"]["dependencies"]
        text = CORE_LOCK.read_text().lower()
        for dep in base:
            name = re.split(r"[<>=!\[@; ]", dep, maxsplit=1)[0].lower()
            assert name in text, f"{name} (a base dependency) is not in the core lock"

    def test_core_lock_excludes_torch(self):
        """The core lock is the CPU/no-torch slice; torch lives in [engines]."""
        assert not re.search(r"^torch==", CORE_LOCK.read_text(), re.MULTILINE | re.IGNORECASE), (
            "the core lock must not pin torch"
        )

    def test_no_lock_pins_torch(self):
        """No lockfile should drag in the CUDA tree."""
        for lock in LOCKS.glob("lock-*.txt"):
            assert not re.search(r"^torch==", lock.read_text(), re.MULTILINE | re.IGNORECASE), (
                f"{lock.name} pins torch"
            )

    def test_cuda_only_extras_are_not_locked(self):
        """Documented limitation: git deps cannot be hashed."""
        readme = README.read_text()
        assert "not** locked" in readme.lower() or "not locked" in readme.lower()
        for name in ("flash-attn", "diff-gaussian-rasterization", "diffoctreerast"):
            assert not (LOCKS / f"lock-{name}.txt").exists()


class TestMakefile:
    def test_lock_target_regenerates_the_core_lock(self):
        assert "requirements/lock-core.txt" in MAKEFILE.read_text()

    def test_lock_target_uses_generate_hashes(self):
        assert "--generate-hashes" in MAKEFILE.read_text()

    def test_install_target_uses_the_lock(self):
        assert "-r $(LOCK)" in MAKEFILE.read_text()
        assert "lock-core.txt" in MAKEFILE.read_text()

    def test_help_target_exists(self):
        assert "\nhelp:" in MAKEFILE.read_text()


class TestDocs:
    def test_readme_documents_the_lockfile(self):
        readme = README.read_text()
        assert "lock-core.txt" in readme
        assert "make lock" in readme

    def test_readme_states_the_cuda_limitation(self):
        """Readers must know why the heavy extras are not locked."""
        readme = README.read_text()
        assert "cannot be hashed" in readme


class TestLocksAreFresh:
    """The lock must match pyproject, not a stale resolution."""

    def test_core_lock_has_a_pillow_at_or_above_the_pyproject_floor(self):
        pyproject = PYPROJECT.read_text()
        floor = re.search(r'"pillow>=\s*([0-9.]+)', pyproject)
        assert floor, "pillow floor not found in pyproject"
        pinned = re.search(r"^pillow==([0-9.]+)", CORE_LOCK.read_text(), re.MULTILINE)
        assert pinned, "pillow not pinned in the core lock"
        pin_v = tuple(int(x) for x in pinned.group(1).split("."))
        floor_v = tuple(int(x) for x in floor.group(1).split("."))
        assert pin_v >= floor_v, f"lock pins pillow {pinned.group(1)} < floor {floor.group(1)}"
