"""Tests for manifest-driven GSO loading (DATA-07)."""

import json
from pathlib import Path

import pytest

# benchmark_runner imports torch at module scope, so these tests only run where
# torch is installed (the torch-cpu CI job). The pure-JSON manifest checks would
# work anywhere, but they live here to keep the file in one place.
pytest.importorskip("torch")

from src.evaluation.benchmark_runner import _validate_against_manifest, load_gso_samples

MANIFEST = Path(__file__).parent.parent / "datasets" / "gso_eval_v1.json"


def _fake_object(root: Path, name: str) -> Path:
    """Create the minimum GSO layout the loader needs."""
    d = root / name
    (d / "images").mkdir(parents=True)
    (d / "meshes").mkdir()
    (d / "meshes" / "m.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    (d / "images" / "000.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    return d


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "gso"
    for name in ("A_obj", "B_obj", "C_obj"):
        _fake_object(root, name)
    return root


def _manifest(root: Path, ids: list[str], count: int | None = None, **extra) -> Path:
    payload = {
        "dataset": "test",
        "version": "1.1",
        "count": len(ids) if count is None else count,
        "object_ids": ids,
    }
    payload.update(extra)
    path = root / "manifest.json"
    path.write_text(json.dumps(payload))
    return path


class TestValidator:
    def test_matching_manifest_returns_ids(self, dataset):
        man = _manifest(dataset, ["A_obj", "B_obj"])
        assert _validate_against_manifest(dataset, man) == ["A_obj", "B_obj"]

    def test_missing_manifest_file_raises(self, dataset):
        with pytest.raises(RuntimeError, match="manifest not found"):
            _validate_against_manifest(dataset, dataset / "nope.json")

    def test_unreadable_manifest_raises(self, dataset):
        bad = dataset / "bad.json"
        bad.write_text("{not json")
        with pytest.raises(RuntimeError, match="cannot read manifest"):
            _validate_against_manifest(dataset, bad)

    def test_manifest_without_object_ids_raises(self, dataset):
        man = dataset / "empty.json"
        man.write_text(json.dumps({"count": 0, "object_ids": []}))
        with pytest.raises(RuntimeError, match="no object_ids list"):
            _validate_against_manifest(dataset, man)

    def test_declared_count_must_match(self, dataset):
        man = _manifest(dataset, ["A_obj"], count=5)
        with pytest.raises(RuntimeError, match="declares count=5"):
            _validate_against_manifest(dataset, man)

    def test_absent_object_raises(self, dataset):
        man = _manifest(dataset, ["A_obj", "ZZZ"])
        with pytest.raises(RuntimeError, match="absent"):
            _validate_against_manifest(dataset, man)

    def test_unlisted_objects_are_reported_not_fatal(self, dataset, capsys):
        """Extra dataset objects are allowed — the manifest is the selection."""
        man = _manifest(dataset, ["A_obj"])
        assert _validate_against_manifest(dataset, man) == ["A_obj"]
        assert "not in the" in capsys.readouterr().out


class TestLoaderUsesManifest:
    def test_manifest_drives_the_selection(self, dataset):
        """DATA-07: without this, the loader scanned directories instead."""
        man = _manifest(dataset, ["A_obj"])
        samples = load_gso_samples(dataset, num_samples=10, manifest_path=man)
        assert [s.sample_id for s in samples] == ["A_obj"]

    def test_manifest_limits_the_count(self, dataset):
        """The dataset has 3 objects; the manifest must cap it at 2."""
        man = _manifest(dataset, ["A_obj", "B_obj"])
        samples = load_gso_samples(dataset, num_samples=10, manifest_path=man)
        assert len(samples) == 2

    def test_explicit_object_ids_win_over_the_manifest(self, dataset):
        """An explicit request is more specific than the manifest."""
        man = _manifest(dataset, ["A_obj", "B_obj"])
        samples = load_gso_samples(dataset, num_samples=10, object_ids=["C_obj"], manifest_path=man)
        assert [s.sample_id for s in samples] == ["C_obj"]

    def test_invalid_manifest_fails_the_load(self, dataset):
        man = _manifest(dataset, ["A_obj", "ZZZ"])
        with pytest.raises(RuntimeError, match="absent"):
            load_gso_samples(dataset, manifest_path=man)

    def test_no_manifest_keeps_the_old_behaviour(self, dataset, monkeypatch):
        """Default execution must not change — the benchmark numbers are frozen.

        The loader falls back to the repo's handpicked.txt when no manifest is
        given; point that at this temp dataset's object_ids so the test isolates
        itself from the real 100-object catalog.
        """
        import src.evaluation.benchmark_runner as br

        monkeypatch.setattr(br, "load_object_ids", lambda _path: ["A_obj", "B_obj", "C_obj"])
        samples = load_gso_samples(dataset, num_samples=10)
        assert len(samples) == 3, "without a manifest the loader scans directories"

    def test_num_samples_still_caps_a_manifest_run(self, dataset):
        man = _manifest(dataset, ["A_obj", "B_obj", "C_obj"])
        samples = load_gso_samples(dataset, num_samples=1, manifest_path=man)
        assert len(samples) == 1


class TestShippedManifest:
    def test_shipped_manifest_is_valid_json(self):
        data = json.loads(MANIFEST.read_text())
        assert data["object_ids"]
        assert isinstance(data["count"], int)

    def test_shipped_count_matches_the_id_list(self):
        """A drift here is exactly what DATA-07 was about."""
        data = json.loads(MANIFEST.read_text())
        assert data["count"] == len(data["object_ids"]), (
            f"count={data['count']} but {len(data['object_ids'])} object_ids"
        )

    def test_shipped_ids_are_unique(self):
        data = json.loads(MANIFEST.read_text())
        assert len(set(data["object_ids"])) == len(data["object_ids"])

    def test_shipped_ids_match_handpicked(self):
        """The manifest and handpicked.txt must describe the same selection."""
        repo = MANIFEST.parent.parent
        handpicked = repo / "handpicked.txt"
        if not handpicked.exists():
            pytest.skip("handpicked.txt not present")
        from src.data.object_ids import load_object_ids

        ids = load_object_ids(handpicked)
        manifest_ids = json.loads(MANIFEST.read_text())["object_ids"]
        assert set(ids) == set(manifest_ids), (
            "manifest and handpicked.txt disagree on the 100-object selection"
        )
