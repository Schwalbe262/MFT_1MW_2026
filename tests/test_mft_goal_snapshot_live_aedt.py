from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import mft_goal_snapshot_live_aedt as snapshot


SHA_A = "a" * 40
SHA_B = "b" * 64
SHA_C = "c" * 64


def _invoke(tmp_path: Path, source: Path, **overrides):
    values = {
        "source_root": source,
        "output_relative_root": "mft_goal_20260726/live_aedt_snapshots_v1",
        "lane": "symmetric",
        "source_task_id": 96313,
        "expected_node": "n111",
        "expected_slurm_job_id": "838708",
        "expected_project_filename": "simulation.aedt",
        "maximum_bytes": 1024 * 1024,
        "source_solver_revision": SHA_A,
        "snapshot_tool_revision": "d" * 40,
        "candidate_sha256": SHA_B,
        "fixed_physics_sha256": SHA_C,
        "environ": {
            "SLURM_SCHED_TASK_ID": "96314",
            "SLURM_JOB_ID": "838708",
            "SLURM_CPUS_PER_TASK": "1",
        },
        "cwd": tmp_path,
        "hostname": "n111",
        "uid": source.stat().st_uid,
    }
    values.update(overrides)
    return snapshot.snapshot_live_aedt(**values)


def test_snapshot_publishes_exact_immutable_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "enroot" / "run"
    source.mkdir(parents=True)
    project = source / "simulation.aedt"
    project.write_bytes(b"AEDT" * 4096)
    real_resolve = Path.resolve

    def fake_safe_root(path, *, expected_uid):
        resolved = real_resolve(path, strict=True)
        assert resolved == real_resolve(source, strict=True)
        assert expected_uid == source.stat().st_uid
        return resolved

    monkeypatch.setattr(snapshot, "_safe_source_root", fake_safe_root)

    result = _invoke(tmp_path, source)

    assert result["status"] == "published"
    destination = Path(result["destination"])
    artifact = destination / "symmetric.aedt"
    manifest = json.loads(
        (destination / "manifest.json").read_text(encoding="utf-8")
    )
    assert artifact.read_bytes() == project.read_bytes()
    assert manifest["source_changed_during_snapshot"] is False
    assert manifest["solver_result_truth_included"] is False
    assert manifest["artifact_sha256"] == snapshot.sha256_file(artifact)
    assert manifest["payload_sha256"] == snapshot.canonical_sha256(
        {
            key: value
            for key, value in manifest.items()
            if key != "payload_sha256"
        }
    )
    if snapshot.os.name == "posix":
        assert artifact.stat().st_mode & 0o777 == 0o400
    else:
        assert artifact.stat().st_mode & 0o222 == 0


def test_snapshot_existing_publication_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "enroot" / "run"
    source.mkdir(parents=True)
    (source / "simulation.aedt").write_bytes(b"stable")
    monkeypatch.setattr(
        snapshot,
        "_safe_source_root",
        lambda path, *, expected_uid: path.resolve(strict=True),
    )

    first = _invoke(tmp_path, source)
    second = _invoke(tmp_path, source, environ={
        "SLURM_SCHED_TASK_ID": "96315",
        "SLURM_JOB_ID": "838708",
        "SLURM_CPUS_PER_TASK": "1",
    })

    assert first["status"] == "published"
    assert second["status"] == "existing_authenticated"
    assert second["artifact_sha256"] == first["artifact_sha256"]


def test_snapshot_rejects_nonexact_aedt_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "enroot" / "run"
    source.mkdir(parents=True)
    (source / "simulation.aedt").write_bytes(b"one")
    (source / "other.aedt").write_bytes(b"two")
    monkeypatch.setattr(
        snapshot,
        "_safe_source_root",
        lambda path, *, expected_uid: path.resolve(strict=True),
    )

    with pytest.raises(snapshot.SnapshotError, match="not exact: 2"):
        _invoke(tmp_path, source)


def test_snapshot_rejects_wrong_allocation(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        snapshot.SnapshotError, match="same-allocation placement contract"
    ):
        _invoke(
            tmp_path,
            tmp_path,
            environ={
                "SLURM_SCHED_TASK_ID": "96314",
                "SLURM_JOB_ID": "wrong",
                "SLURM_CPUS_PER_TASK": "1",
            },
        )


def test_snapshot_rejects_job_id_path_traversal(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        snapshot.SnapshotError,
        match="expected Slurm job ID",
    ):
        _invoke(
            tmp_path,
            tmp_path,
            expected_slurm_job_id="../../escape",
            environ={
                "SLURM_SCHED_TASK_ID": "96314",
                "SLURM_JOB_ID": "../../escape",
                "SLURM_CPUS_PER_TASK": "1",
            },
        )


def test_snapshot_rejects_existing_artifact_path_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "enroot" / "run"
    source.mkdir(parents=True)
    (source / "simulation.aedt").write_bytes(b"stable")
    monkeypatch.setattr(
        snapshot,
        "_safe_source_root",
        lambda path, *, expected_uid: path.resolve(strict=True),
    )
    first = _invoke(tmp_path, source)
    manifest_path = Path(first["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_filename"] = "../../outside.aedt"
    manifest.pop("payload_sha256")
    manifest["payload_sha256"] = snapshot.canonical_sha256(manifest)
    manifest_path.chmod(0o600)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(snapshot.SnapshotError, match="artifact name drifted"):
        _invoke(tmp_path, source)


def test_snapshot_rejects_source_replaced_between_inventory_and_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "enroot" / "run"
    source.mkdir(parents=True)
    project = source / "simulation.aedt"
    project.write_bytes(b"before")
    monkeypatch.setattr(
        snapshot,
        "_safe_source_root",
        lambda path, *, expected_uid: path.resolve(strict=True),
    )
    real_open = snapshot.os.open
    replaced = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal replaced
        if Path(path) == project and not replaced:
            replaced = True
            replacement = source / "replacement"
            replacement.write_bytes(b"after!")
            snapshot.os.replace(replacement, project)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(snapshot.os, "open", racing_open)

    with pytest.raises(snapshot.SnapshotError, match="before snapshot"):
        _invoke(tmp_path, source)
