from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import stat
import sys
import time
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "mft_goal_corrected_thermal_continuation.py"
SPEC = importlib.util.spec_from_file_location(
    "corrected_thermal_checkpoint", MODULE_PATH
)
assert SPEC and SPEC.loader
checkpoint = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checkpoint
SPEC.loader.exec_module(checkpoint)


def _source_tree(root: Path) -> Path:
    source = root / "simulation_824575_2786652"
    source.mkdir()
    stem = source.name
    results = source / f"{stem}.aedtresults"
    thermal = results / "icepak_thermal.results"
    thermal.mkdir(parents=True)
    (source / f"{stem}.aedt").write_bytes(b"sealed-aedt")
    (results / "ManagedFiles_Design7.asol").write_bytes(b"managed")
    (results / "icepak_thermal.asol").write_bytes(b"public-asol")
    (thermal / "DV274_S271_V0.profile").write_bytes(b"profile-zero")
    (thermal / "DV274_S271_V275.profile").write_bytes(b"profile-mesh")
    for family, suffix in checkpoint.MESH_FAMILIES:
        for index in checkpoint.MESH_INDICES:
            mesh = thermal / f"{family}{index}{suffix}"
            mesh.mkdir()
            (mesh / "grid_mapping").write_bytes(f"map-{family}-{index}".encode())
            (mesh / "grid_output").write_bytes(f"grid-{family}-{index}".encode())
    return source


def _no_writers(*_args: Any, **_kwargs: Any) -> tuple[Any, ...]:
    return ()


def _snapshot(monkeypatch: pytest.MonkeyPatch, source: Path) -> Any:
    monkeypatch.setattr(checkpoint, "_open_writers", _no_writers)
    return checkpoint.snapshot_static_source(source)


def _quota() -> dict[str, int | str]:
    gib = 1024**3
    return {
        "filesystem": "gpfs",
        "quota_type": "USR",
        "uid": 0,
        "name": "test",
        "usage_bytes": 90 * gib,
        "soft_limit_bytes": 190 * gib,
        "hard_limit_bytes": 200 * gib,
        "in_doubt_bytes": 5 * gib,
        "files_used": 900_000,
        "files_soft_limit": 10_240_000,
        "files_hard_limit": 11_264_000,
        "files_in_doubt": 100,
    }


def _quota_evidence(observed_at_epoch: float | None = None) -> tuple[str, str]:
    quota = _quota()
    evidence = {
        key: value
        for key, value in quota.items()
        if key
        in {
            "filesystem",
            "quota_type",
            "uid",
            "usage_bytes",
            "soft_limit_bytes",
            "hard_limit_bytes",
            "in_doubt_bytes",
            "files_used",
            "files_soft_limit",
            "files_hard_limit",
            "files_in_doubt",
        }
    }
    evidence["observed_at_epoch"] = (
        time.time() if observed_at_epoch is None else observed_at_epoch
    )
    evidence["source"] = "gate2:mmlsquota-Y"
    canonical = checkpoint._sha256_bytes(checkpoint._canonical_json(evidence))
    evidence["canonical_sha256"] = canonical
    encoded = base64.b64encode(checkpoint._canonical_json(evidence)).decode()
    return encoded, canonical


def _portable_noreplace(source: Path, destination: Path) -> None:
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    os.rename(source, destination)


def _portable_verify(root: Path, expected_hashes: dict[str, str]) -> dict[str, Any]:
    actual = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    assert actual == set(expected_hashes) | {".checkpoint_manifest.json"}
    for relative, expected in expected_hashes.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
    return {
        "directory_count": sum(path.is_dir() for path in root.rglob("*")) + 1,
        "file_count": len(actual),
        "manifest_sha256": hashlib.sha256(
            (root / ".checkpoint_manifest.json").read_bytes()
        ).hexdigest(),
    }


def _stage_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    allowed = tmp_path / "gpfs" / "immutable_sources_v1"
    allowed.parent.mkdir(parents=True)
    monkeypatch.setattr(checkpoint, "ALLOWED_DESTINATION_ROOT", allowed)
    monkeypatch.setattr(checkpoint, "_open_writers", _no_writers)
    monkeypatch.setattr(
        checkpoint,
        "_filesystem_capacity",
        lambda path, **kwargs: {
            "anchor_device": 1,
            "bavail": 1_000_000,
            "block_size": 4096,
            "free_bytes": 1_000_000 * 4096,
            "fsid": 2,
            "readonly": False,
            "required_free_bytes": kwargs["required_free_bytes"],
        },
    )
    monkeypatch.setattr(checkpoint, "_assert_private_directory", lambda path: None)
    monkeypatch.setattr(checkpoint, "_rename_noreplace", _portable_noreplace)
    monkeypatch.setattr(checkpoint, "_fsync_directory", lambda path: None)
    monkeypatch.setattr(checkpoint, "_fsync_file", lambda path: None)
    monkeypatch.setattr(checkpoint, "_verify_protected_tree", _portable_verify)
    monkeypatch.setattr(checkpoint.os, "getuid", lambda: 0, raising=False)
    monkeypatch.setattr(checkpoint.os, "getgid", lambda: 0, raising=False)
    return allowed


def _stage(
    source: Path,
    destination: Path,
    sealed: Any,
) -> dict[str, Any]:
    aedt = next(row for row in sealed.files if row.path.endswith(".aedt"))
    aedt_sha = hashlib.sha256((source / aedt.path).read_bytes()).hexdigest()
    quota_encoded, quota_sha = _quota_evidence()
    return checkpoint.stage_static_source(
        source=source,
        destination=destination,
        expected_metadata_sha256=sealed.metadata_sha256,
        expected_count=len(sealed.files),
        expected_directory_count=len(sealed.directories),
        expected_logical_bytes=sealed.logical_bytes,
        expected_allocated_bytes=sealed.allocated_bytes,
        expected_aedt_sha256=aedt_sha,
        checkpoint_tool_payload_sha256="a" * 64,
        login_quota_evidence_base64=quota_encoded,
        login_quota_evidence_sha256=quota_sha,
        login_quota_evidence_max_age_seconds=120,
        minimum_soft_headroom_bytes=50 * 1024**3,
        minimum_hard_headroom_bytes=60 * 1024**3,
        minimum_soft_inode_headroom=1_000_000,
        minimum_hard_inode_headroom=1_000_000,
    )


def test_exact_allowlist_prunes_unrelated_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _source_tree(tmp_path)
    unrelated = source / f"{source.name}.aedtresults" / "maxwell_matrix.results"
    unrelated.mkdir()
    (unrelated / "live-or-irrelevant.bin").write_bytes(b"ignore")

    value = _snapshot(monkeypatch, source)

    assert len(value.files) == 25
    assert len(value.directories) == 13
    assert len([row for row in value.files if row.path.endswith("/grid_mapping")]) == 10
    assert len([row for row in value.files if row.path.endswith("/grid_output")]) == 10
    assert not any("maxwell_matrix" in row.path for row in value.files)
    assert value.required_inodes == 39
    assert value.required_budget_bytes > max(value.logical_bytes, value.allocated_bytes)


def test_mesh_extra_child_and_hardlink_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _source_tree(tmp_path)
    thermal = source / f"{source.name}.aedtresults" / "icepak_thermal.results"
    mesh = thermal / "DV274_Meshes0_V213.sd"
    (mesh / "unexpected").write_bytes(b"x")
    monkeypatch.setattr(checkpoint, "_open_writers", _no_writers)
    with pytest.raises(checkpoint.CheckpointError, match="non-contract children"):
        checkpoint.snapshot_static_source(source)

    (mesh / "unexpected").unlink()
    os.link(mesh / "grid_mapping", tmp_path / "second-link")
    with pytest.raises(checkpoint.CheckpointError, match="st_nlink"):
        checkpoint.snapshot_static_source(source)


def test_symlink_directory_stat_is_rejected() -> None:
    class FakeSymlink:
        def lstat(self) -> os.stat_result:
            return os.stat_result((stat.S_IFLNK | 0o777,) + (0,) * 9)

        def __str__(self) -> str:
            return "fake-symlink"

    with pytest.raises(checkpoint.CheckpointError, match="real directory"):
        checkpoint._lstat_real_directory(FakeSymlink(), "test")


def test_proc_absence_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(checkpoint.CheckpointError, match="proc filesystem"):
        checkpoint._open_writers(source, set(), proc_root=tmp_path / "missing-proc")


def test_included_writer_rejects_sealed_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _source_tree(tmp_path)
    writer = checkpoint.Writer(
        pid=10,
        fd=4,
        command="ansysedt.exe",
        path=str(source / f"{source.name}.aedt"),
        included=True,
        flags_octal="0o2",
    )
    monkeypatch.setattr(checkpoint, "_open_writers", lambda *_a, **_k: (writer,))
    value = checkpoint.snapshot_static_source(source)

    with pytest.raises(checkpoint.CheckpointError, match="open for write"):
        checkpoint._assert_expected_snapshot(
            value,
            metadata_sha256=value.metadata_sha256,
            count=len(value.files),
            directory_count=len(value.directories),
            logical_bytes=value.logical_bytes,
            allocated_bytes=value.allocated_bytes,
        )


def test_destination_requires_literal_direct_child_and_no_overlap(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    allowed = tmp_path / "gpfs" / "immutable_sources_v1"
    allowed.parent.mkdir()

    destination, root = checkpoint._validate_destination(
        source, allowed / "sealed", allowed_root=allowed
    )
    assert destination == allowed / "sealed"
    assert root == allowed
    with pytest.raises(checkpoint.CheckpointError, match="direct child"):
        checkpoint._validate_destination(
            source, allowed / "nested" / "sealed", allowed_root=allowed
        )
    with pytest.raises(checkpoint.CheckpointError, match="normalized"):
        checkpoint._validate_destination(
            source,
            Path(os.fspath(allowed / ".." / allowed.name / "sealed")),
            allowed_root=allowed,
        )

    overlapping_root = source / "checkpoints"
    with pytest.raises(checkpoint.CheckpointError, match="overlap"):
        checkpoint._validate_destination(
            source,
            overlapping_root / "sealed",
            allowed_root=overlapping_root,
        )


def test_rename_noreplace_refuses_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.write_bytes(b"a")
    destination.write_bytes(b"b")

    with pytest.raises(FileExistsError):
        checkpoint._rename_noreplace(source, destination)
    assert source.read_bytes() == b"a"
    assert destination.read_bytes() == b"b"


def test_quota_checks_soft_hard_bytes_and_file_inodes() -> None:
    gib = 1024**3
    value = checkpoint._assert_quota(
        _quota(),
        budget_bytes=26 * gib,
        budget_inodes=39,
        minimum_soft_headroom_bytes=50 * gib,
        minimum_hard_headroom_bytes=60 * gib,
        minimum_soft_inode_headroom=1_000_000,
        minimum_hard_inode_headroom=1_000_000,
    )
    assert value["soft_headroom_after_bytes"] == 69 * gib
    assert value["hard_headroom_after_bytes"] == 79 * gib
    assert value["soft_inode_headroom_after"] == 9_339_861
    assert value["hard_inode_headroom_after"] == 10_363_861

    quota = _quota()
    quota["hard_limit_bytes"] = 160 * gib
    with pytest.raises(checkpoint.CheckpointError, match="hard_headroom"):
        checkpoint._assert_quota(
            quota,
            budget_bytes=26 * gib,
            budget_inodes=39,
            minimum_soft_headroom_bytes=0,
            minimum_hard_headroom_bytes=40 * gib,
            minimum_soft_inode_headroom=0,
            minimum_hard_inode_headroom=0,
        )


def test_login_quota_evidence_is_canonical_fresh_and_uid_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 1_000_000.0
    monkeypatch.setattr(checkpoint.os, "getuid", lambda: 0, raising=False)
    encoded, canonical = _quota_evidence(now - 30)

    value = checkpoint._decode_login_quota_evidence(
        encoded,
        expected_canonical_sha256=canonical,
        maximum_age_seconds=120,
        now=now,
    )

    assert value["canonical_sha256"] == canonical
    assert value["age_seconds_at_validation"] == 30
    with pytest.raises(checkpoint.CheckpointError, match="stale"):
        checkpoint._decode_login_quota_evidence(
            encoded,
            expected_canonical_sha256=canonical,
            maximum_age_seconds=20,
            now=now,
        )
    with pytest.raises(checkpoint.CheckpointError, match="canonical SHA"):
        checkpoint._decode_login_quota_evidence(
            encoded,
            expected_canonical_sha256="0" * 64,
            maximum_age_seconds=120,
            now=now,
        )
    with pytest.raises(checkpoint.CheckpointError, match=r"\[1, 120\]"):
        checkpoint._decode_login_quota_evidence(
            encoded,
            expected_canonical_sha256=canonical,
            maximum_age_seconds=121,
            now=now,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_login_quota_evidence_rejects_nonfinite_json_numbers(
    monkeypatch: pytest.MonkeyPatch,
    value: float,
) -> None:
    monkeypatch.setattr(checkpoint.os, "getuid", lambda: 0, raising=False)
    encoded, _ = _quota_evidence()
    evidence = json.loads(base64.b64decode(encoded))
    evidence["usage_bytes"] = value
    evidence.pop("canonical_sha256")
    canonical = checkpoint._sha256_bytes(checkpoint._canonical_json(evidence))
    evidence["canonical_sha256"] = canonical
    tampered = base64.b64encode(checkpoint._canonical_json(evidence)).decode()

    with pytest.raises(checkpoint.CheckpointError, match="valid base64 JSON"):
        checkpoint._decode_login_quota_evidence(
            tampered,
            expected_canonical_sha256=canonical,
            maximum_age_seconds=120,
        )


@pytest.mark.parametrize("value", [True, 1.0, -1])
def test_login_quota_evidence_rejects_noninteger_or_negative_integer_fields(
    monkeypatch: pytest.MonkeyPatch,
    value: bool | float | int,
) -> None:
    monkeypatch.setattr(checkpoint.os, "getuid", lambda: 0, raising=False)
    encoded, _ = _quota_evidence()
    evidence = json.loads(base64.b64decode(encoded))
    evidence["files_used"] = value
    evidence.pop("canonical_sha256")
    canonical = checkpoint._sha256_bytes(checkpoint._canonical_json(evidence))
    evidence["canonical_sha256"] = canonical
    tampered = base64.b64encode(checkpoint._canonical_json(evidence)).decode()

    with pytest.raises(checkpoint.CheckpointError, match="exact integer"):
        checkpoint._decode_login_quota_evidence(
            tampered,
            expected_canonical_sha256=canonical,
            maximum_age_seconds=120,
        )


def test_login_quota_evidence_rejects_authority_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(checkpoint.os, "getuid", lambda: 0, raising=False)
    encoded, _ = _quota_evidence()
    evidence = json.loads(base64.b64decode(encoded))
    evidence["source"] = "n114:mmlsquota-Y"
    evidence.pop("canonical_sha256")
    canonical = checkpoint._sha256_bytes(checkpoint._canonical_json(evidence))
    evidence["canonical_sha256"] = canonical
    tampered = base64.b64encode(checkpoint._canonical_json(evidence)).decode()

    with pytest.raises(checkpoint.CheckpointError, match="authority"):
        checkpoint._decode_login_quota_evidence(
            tampered,
            expected_canonical_sha256=canonical,
            maximum_age_seconds=120,
        )


@pytest.mark.parametrize("value", [True, "not-a-time", -1])
def test_login_quota_evidence_rejects_invalid_observation_time(
    monkeypatch: pytest.MonkeyPatch,
    value: bool | str | int,
) -> None:
    monkeypatch.setattr(checkpoint.os, "getuid", lambda: 0, raising=False)
    encoded, _ = _quota_evidence()
    evidence = json.loads(base64.b64decode(encoded))
    evidence["observed_at_epoch"] = value
    evidence.pop("canonical_sha256")
    canonical = checkpoint._sha256_bytes(checkpoint._canonical_json(evidence))
    evidence["canonical_sha256"] = canonical
    tampered = base64.b64encode(checkpoint._canonical_json(evidence)).decode()

    with pytest.raises(checkpoint.CheckpointError, match="observed_at_epoch"):
        checkpoint._decode_login_quota_evidence(
            tampered,
            expected_canonical_sha256=canonical,
            maximum_age_seconds=120,
        )


def test_destination_filesystem_capacity_checks_free_readonly_and_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    anchor = tmp_path / "gpfs"
    anchor.mkdir()
    actual_device = anchor.stat().st_dev
    values = SimpleNamespace(
        f_bavail=1000,
        f_frsize=4096,
        f_flag=0,
        f_fsid=77,
    )
    monkeypatch.setattr(checkpoint.os, "statvfs", lambda path: values, raising=False)
    monkeypatch.setattr(checkpoint.os, "ST_RDONLY", 1, raising=False)

    result = checkpoint._filesystem_capacity(
        anchor,
        required_free_bytes=4_000_000,
        expected_device=actual_device,
        expected_fsid=77,
    )
    assert result["free_bytes"] == 4_096_000

    values.f_flag = 1
    with pytest.raises(checkpoint.CheckpointError, match="read-only"):
        checkpoint._filesystem_capacity(anchor, required_free_bytes=1)
    values.f_flag = 0
    with pytest.raises(checkpoint.CheckpointError, match="physical free"):
        checkpoint._filesystem_capacity(anchor, required_free_bytes=5_000_000)
    with pytest.raises(checkpoint.CheckpointError, match="device changed"):
        checkpoint._filesystem_capacity(
            anchor,
            required_free_bytes=1,
            expected_device=actual_device + 1,
        )


def test_open_fd_copy_hashes_source_before_during_after(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _source_tree(tmp_path)
    sealed = _snapshot(monkeypatch, source)
    row = next(item for item in sealed.files if item.path.endswith(".asol"))
    destination = tmp_path / "destination"
    destination.mkdir()

    result = checkpoint._copy_stable_file(source, destination, row)

    expected = hashlib.sha256((source / row.path).read_bytes()).hexdigest()
    assert result == expected
    assert hashlib.sha256((destination / row.path).read_bytes()).hexdigest() == expected


def test_protection_fsyncs_after_each_chmod(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path("/fake/root")
    child = root / "child"
    file_path = child / "file"
    events: list[tuple[str, str, int | None]] = []
    monkeypatch.setattr(
        checkpoint,
        "_walk_real_tree",
        lambda path: ((root, child), (file_path,)),
    )
    monkeypatch.setattr(
        checkpoint,
        "_chmod_nofollow",
        lambda path, mode: events.append(("chmod", str(path), mode)),
    )
    monkeypatch.setattr(
        checkpoint,
        "_fsync_file",
        lambda path: events.append(("fsync-file", str(path), None)),
    )
    monkeypatch.setattr(
        checkpoint,
        "_fsync_directory",
        lambda path: events.append(("fsync-dir", str(path), None)),
    )

    checkpoint._fsync_and_protect_tree(root)

    assert events == [
        ("chmod", str(file_path), 0o400),
        ("fsync-file", str(file_path), None),
        ("chmod", str(child), 0o500),
        ("fsync-dir", str(child), None),
        ("chmod", str(root), 0o500),
        ("fsync-dir", str(root), None),
    ]


def test_stage_success_with_mocked_quota(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _source_tree(tmp_path)
    allowed = _stage_environment(monkeypatch, tmp_path)
    sealed = checkpoint.snapshot_static_source(source)
    destination = allowed / "sealed"

    result = _stage(source, destination, sealed)

    assert result["status"] == "complete"
    assert destination.is_dir()
    assert not os.path.lexists(Path(str(destination) + ".incoming"))
    manifest = json.loads(
        (destination / ".checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["source_provenance"]["solver_revision"] == (
        "a1e4f70cefa1af04673c73a6131bf490c0cc14b5"
    )
    assert manifest["checkpoint_creator"]["tool_payload_sha256"] == "a" * 64
    assert "checkpoint_executor" not in manifest
    assert manifest["post_login_quota_reauthentication_required"] is True
    assert result["post_login_quota_reauthentication_required"] is True
    assert len(manifest["files"]) == 25
    assert manifest["diagnostic_only"] is True
    assert manifest["canonical"] is False


def test_changed_source_is_quarantined_and_final_is_not_published(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _source_tree(tmp_path)
    allowed = _stage_environment(monkeypatch, tmp_path)
    sealed = checkpoint.snapshot_static_source(source)
    destination = allowed / "sealed"
    original_copy = checkpoint._copy_stable_file
    calls = 0

    def changing_copy(source_root: Path, destination_root: Path, row: Any) -> str:
        nonlocal calls
        value = original_copy(source_root, destination_root, row)
        calls += 1
        if calls == 1:
            target = source / f"{source.name}.aedtresults" / "icepak_thermal.asol"
            target.write_bytes(target.read_bytes() + b"-changed")
        return value

    monkeypatch.setattr(checkpoint, "_copy_stable_file", changing_copy)
    with pytest.raises(checkpoint.CheckpointError):
        _stage(source, destination, sealed)

    assert not destination.exists()
    assert not os.path.lexists(Path(str(destination) + ".incoming"))
    quarantines = list(allowed.glob("sealed.incomplete.*"))
    assert len(quarantines) == 1
    error = json.loads(capsys.readouterr().err)
    assert error["quarantine"] == str(quarantines[0])
    assert "audit-incomplete" in error["audit_command"]


def test_collision_fails_before_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _source_tree(tmp_path)
    allowed = _stage_environment(monkeypatch, tmp_path)
    allowed.mkdir()
    destination = allowed / "sealed"
    destination.mkdir()
    sealed = checkpoint.snapshot_static_source(source)

    with pytest.raises(checkpoint.CheckpointError, match="collision"):
        _stage(source, destination, sealed)


def test_quarantine_uses_next_deterministic_noncolliding_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "sealed"
    partial = tmp_path / "sealed.incoming"
    partial.mkdir()
    monkeypatch.setattr(checkpoint, "_rename_noreplace", _portable_noreplace)
    monkeypatch.setattr(checkpoint, "_fsync_directory", lambda path: None)
    monkeypatch.setattr(checkpoint.time, "strftime", lambda *_a, **_k: "STAMP")
    monkeypatch.setattr(checkpoint.os, "getpid", lambda: 123)
    first = tmp_path / "sealed.incomplete.STAMP.123.00000"
    first.mkdir()

    result = checkpoint._quarantine_partial(partial, destination)

    assert result.name == "sealed.incomplete.STAMP.123.00001"
    assert result.is_dir()
    assert not partial.exists()


def test_post_login_gate_can_quarantine_published_checkpoint_without_delete(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _source_tree(tmp_path)
    allowed = _stage_environment(monkeypatch, tmp_path)
    sealed = checkpoint.snapshot_static_source(source)
    destination = allowed / "sealed"
    _stage(source, destination, sealed)
    manifest = destination / ".checkpoint_manifest.json"
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert checkpoint._full_sha256(manifest) == manifest_sha
    monkeypatch.setattr(
        checkpoint,
        "_lstat_real_directory",
        lambda path, label: SimpleNamespace(
            st_uid=0,
            st_gid=0,
            st_mode=stat.S_IFDIR | 0o500,
        ),
    )

    result = checkpoint.quarantine_published_checkpoint(
        destination,
        expected_manifest_sha256=manifest_sha,
        reason="post_login_quota_gate_failed",
    )

    assert result["status"] == "quarantined-no-delete"
    assert not destination.exists()
    quarantine = Path(result["quarantine"])
    assert quarantine.is_dir()
    assert (quarantine / ".checkpoint_manifest.json").is_file()
