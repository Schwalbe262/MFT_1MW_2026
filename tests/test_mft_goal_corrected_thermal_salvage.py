from __future__ import annotations

import errno
import base64
import hashlib
import importlib.util
import json
import os
import stat
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "mft_goal_corrected_thermal_continuation.py"
SPEC = importlib.util.spec_from_file_location(
    "corrected_thermal_checkpoint_salvage_tests", MODULE_PATH
)
assert SPEC and SPEC.loader
checkpoint = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checkpoint
SPEC.loader.exec_module(checkpoint)


def _identity(
    *,
    mode: int,
    inode: int = 22,
    device: int = 11,
    uid: int = 7,
    gid: int = 8,
    nlink: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        st_dev=device,
        st_gid=gid,
        st_ino=inode,
        st_mode=mode,
        st_nlink=nlink,
        st_uid=uid,
    )


def test_fd_protection_uses_fchmod_then_fsync_and_rechecks_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _identity(mode=stat.S_IFREG | 0o600)
    after = _identity(mode=stat.S_IFREG | 0o400)
    events: list[tuple[str, Any]] = []

    class FakePath:
        def __init__(self) -> None:
            self.calls = 0

        def lstat(self) -> SimpleNamespace:
            self.calls += 1
            return before if self.calls == 1 else after

        def __fspath__(self) -> str:
            return "/fake/incoming/sealed.aedt"

        def __str__(self) -> str:
            return self.__fspath__()

    path = FakePath()
    fstats = iter((before, before, after))
    descriptors = iter((91, 92))
    fake_os = SimpleNamespace(
        O_DIRECTORY=0x10000,
        O_NOFOLLOW=0x20000,
        O_RDONLY=os.O_RDONLY,
        O_RDWR=os.O_RDWR,
        close=lambda descriptor: events.append(("close", descriptor)),
        fchmod=lambda descriptor, mode: events.append(("fchmod", mode)),
        fstat=lambda descriptor: next(fstats),
        fsync=lambda descriptor: events.append(("fsync", descriptor)),
        getgid=lambda: 8,
        getuid=lambda: 7,
        name="posix",
        open=lambda target, flags: (
            events.append(("open", flags)) or next(descriptors)
        ),
    )
    monkeypatch.setattr(checkpoint, "os", fake_os)

    checkpoint._protect_path_fd(path, 0o400, directory=False)

    assert ("fchmod", 0o400) in events
    assert events.count(("fsync", 92)) == 2
    fsync_indices = [
        index for index, event in enumerate(events) if event == ("fsync", 92)
    ]
    assert fsync_indices[0] < events.index(("fchmod", 0o400)) < fsync_indices[1]
    assert events[-2:] == [("close", 92), ("close", 91)]


def _prepare_guarded_rename(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    parent = tmp_path / "private"
    parent.mkdir(mode=0o700)
    parent.chmod(0o700)
    source = parent / "sealed.incoming"
    source.mkdir(mode=0o700)
    destination = parent / "sealed"
    parent_stat = parent.lstat()
    monkeypatch.setattr(
        checkpoint.os,
        "getuid",
        lambda: parent_stat.st_uid,
        raising=False,
    )
    monkeypatch.setattr(
        checkpoint.os,
        "getgid",
        lambda: parent_stat.st_gid,
        raising=False,
    )
    monkeypatch.setattr(
        checkpoint,
        "_acl_lines",
        lambda path: ("user::rwx", "group::---", "other::---"),
    )
    # Windows does not preserve POSIX 0700 mode bits.  The fallback-specific
    # tests mock this already separately tested gate and assert it is invoked.
    monkeypatch.setattr(checkpoint, "_assert_private_directory", lambda path: None)
    monkeypatch.setattr(
        checkpoint,
        "_gpfs_mount_identity",
        lambda path: {
            "filesystem_type": "gpfs",
            "mount_point": str(path),
            "mount_source": "test",
        },
    )
    monkeypatch.setattr(
        checkpoint,
        "_finalize_rename_claim",
        lambda path, payload: path.write_bytes(checkpoint._canonical_json(payload)),
    )
    monkeypatch.setattr(checkpoint, "_fsync_directory", lambda path: None)
    monkeypatch.setattr(
        checkpoint,
        "_full_sha256",
        lambda path: "a" * 64,
    )
    return parent, source, destination


def test_gpfs_einval_fallback_is_claim_guarded_and_noreplacing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parent, source, destination = _prepare_guarded_rename(monkeypatch, tmp_path)
    monkeypatch.setattr(
        checkpoint,
        "_rename_noreplace",
        lambda *_args: (_ for _ in ()).throw(
            OSError(errno.EINVAL, os.strerror(errno.EINVAL))
        ),
    )
    exclusive_calls: list[Path] = []
    original_write_exclusive = checkpoint._write_exclusive

    def record_exclusive(path: Path, payload: bytes) -> None:
        exclusive_calls.append(path)
        original_write_exclusive(path, payload)

    monkeypatch.setattr(checkpoint, "_write_exclusive", record_exclusive)

    result = checkpoint._guarded_rename_noreplace(
        source,
        destination,
        operation="resume-protect-publish",
        tool_payload_sha256="a" * 64,
    )

    assert result["method"] == "gpfs-einval-guarded-plain-rename"
    assert exclusive_calls == [
        parent / ".sealed.rename.claim.resume-protect-publish.aaaaaaaaaaaa"
    ]
    assert destination.is_dir()
    assert not source.exists()
    assert exclusive_calls[0].is_file()
    assert json.loads(exclusive_calls[0].read_bytes())["status"] == "complete"


def test_gpfs_einval_fallback_refuses_final_collision_without_deleting_either(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parent, source, destination = _prepare_guarded_rename(monkeypatch, tmp_path)
    destination.mkdir()
    marker = destination / "existing"
    marker.write_bytes(b"do-not-replace")
    monkeypatch.setattr(
        checkpoint,
        "_rename_noreplace",
        lambda *_args: (_ for _ in ()).throw(
            OSError(errno.EINVAL, os.strerror(errno.EINVAL))
        ),
    )

    with pytest.raises(FileExistsError):
        checkpoint._guarded_rename_noreplace(
            source,
            destination,
            operation="resume-protect-publish",
            tool_payload_sha256="a" * 64,
        )

    assert source.is_dir()
    assert marker.read_bytes() == b"do-not-replace"
    claims = list(parent.glob(".sealed.rename.claim.*"))
    assert len(claims) == 1
    assert json.loads(claims[0].read_bytes())["status"] == "failed-no-delete"


def test_gpfs_einval_fallback_requires_owner_only_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parent, source, destination = _prepare_guarded_rename(monkeypatch, tmp_path)
    monkeypatch.setattr(
        checkpoint,
        "_assert_private_directory",
        lambda path: (_ for _ in ()).throw(
            checkpoint.CheckpointError("destination root mode is not 0700")
        ),
    )
    monkeypatch.setattr(
        checkpoint,
        "_rename_noreplace",
        lambda *_args: (_ for _ in ()).throw(
            OSError(errno.EINVAL, os.strerror(errno.EINVAL))
        ),
    )

    with pytest.raises(checkpoint.CheckpointError, match="mode is not 0700"):
        checkpoint._guarded_rename_noreplace(
            source,
            destination,
            operation="resume-protect-publish",
            tool_payload_sha256="a" * 64,
        )

    assert source.is_dir()
    assert not destination.exists()
    assert not list(parent.glob(".sealed.rename.claim.*"))


def test_final_exact_verification_rejects_extra_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "sealed"
    child = root / "nested"
    child.mkdir(parents=True)
    payload = child / "grid_output"
    manifest = root / ".checkpoint_manifest.json"
    payload.write_bytes(b"grid")
    manifest.write_bytes(b"manifest")
    expected_hash = hashlib.sha256(payload.read_bytes()).hexdigest()
    directories = (root, child)
    files = (manifest, payload)
    owner = root.stat()

    def fake_lstat(path: Path) -> SimpleNamespace:
        is_directory = path in directories
        return _identity(
            mode=(stat.S_IFDIR if is_directory else stat.S_IFREG)
            | (0o500 if is_directory else 0o400),
            inode=hash(str(path)),
            device=owner.st_dev,
            uid=owner.st_uid,
            gid=owner.st_gid,
        )

    monkeypatch.setattr(
        checkpoint.os, "getuid", lambda: owner.st_uid, raising=False
    )
    monkeypatch.setattr(
        checkpoint.os, "getgid", lambda: owner.st_gid, raising=False
    )
    monkeypatch.setattr(checkpoint, "_walk_real_tree", lambda path: (directories, files))
    monkeypatch.setattr(Path, "lstat", fake_lstat)

    verified = checkpoint._verify_protected_tree(
        root, {"nested/grid_output": expected_hash}
    )
    assert verified["file_count"] == 2

    extra = root / "unexpected.bin"
    extra.write_bytes(b"not-allowlisted")
    monkeypatch.setattr(
        checkpoint,
        "_walk_real_tree",
        lambda path: (directories, files + (extra,)),
    )
    with pytest.raises(checkpoint.CheckpointError, match="children mismatch"):
        checkpoint._verify_protected_tree(
            root, {"nested/grid_output": expected_hash}
        )


def test_quarantine_failure_never_deletes_partial(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "sealed"
    partial = tmp_path / "sealed.incoming"
    partial.mkdir()
    marker = partial / "copied.aedt"
    marker.write_bytes(b"recoverable")
    monkeypatch.setattr(checkpoint, "_full_sha256", lambda path: "a" * 64)
    monkeypatch.setattr(
        checkpoint,
        "_guarded_rename_noreplace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            checkpoint.CheckpointError("rename unavailable")
        ),
    )

    with pytest.raises(checkpoint.CheckpointError, match="rename unavailable"):
        checkpoint._quarantine_partial(partial, destination)

    assert marker.read_bytes() == b"recoverable"
    assert partial.is_dir()


def _source_tree(root: Path) -> Path:
    source = root / "simulation_824575_2786652"
    source.mkdir()
    results = source / f"{source.name}.aedtresults"
    thermal = results / "icepak_thermal.results"
    thermal.mkdir(parents=True)
    (source / f"{source.name}.aedt").write_bytes(b"sealed-aedt")
    (results / "ManagedFiles_Design7.asol").write_bytes(b"managed")
    (results / "icepak_thermal.asol").write_bytes(b"public-asol")
    (thermal / "DV274_S271_V0.profile").write_bytes(b"profile-zero")
    (thermal / "DV274_S271_V275.profile").write_bytes(b"profile-mesh")
    for family, suffix in checkpoint.MESH_FAMILIES:
        for index in checkpoint.MESH_INDICES:
            mesh = thermal / f"{family}{index}{suffix}"
            mesh.mkdir()
            (mesh / "grid_mapping").write_bytes(
                f"map-{family}-{index}".encode()
            )
            (mesh / "grid_output").write_bytes(
                f"grid-{family}-{index}".encode()
            )
    return source


def _quota_evidence(*, observed_at: float | None = None) -> tuple[str, str]:
    gib = 1024**3
    evidence = {
        "filesystem": "gpfs",
        "files_hard_limit": 11_264_000,
        "files_in_doubt": 0,
        "files_soft_limit": 10_240_000,
        "files_used": 1,
        "hard_limit_bytes": 200 * gib,
        "in_doubt_bytes": 0,
        "observed_at_epoch": time.time() if observed_at is None else observed_at,
        "quota_type": "USR",
        "soft_limit_bytes": 190 * gib,
        "source": "gate2:mmlsquota-Y",
        "uid": 0,
        "usage_bytes": gib,
    }
    canonical = checkpoint._sha256_bytes(checkpoint._canonical_json(evidence))
    evidence["canonical_sha256"] = canonical
    encoded = base64.b64encode(checkpoint._canonical_json(evidence)).decode()
    return encoded, canonical


def _portable_verify(
    root: Path,
    expected_hashes: dict[str, str],
) -> dict[str, Any]:
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    expected = set(expected_hashes) | {".checkpoint_manifest.json"}
    if actual != expected:
        raise checkpoint.CheckpointError("checkpoint children mismatch")
    for relative, expected_hash in expected_hashes.items():
        if (
            hashlib.sha256((root / relative).read_bytes()).hexdigest()
            != expected_hash
        ):
            raise checkpoint.CheckpointError(
                f"checkpoint content mismatch: {relative}"
            )
    manifest_hash = hashlib.sha256(
        (root / ".checkpoint_manifest.json").read_bytes()
    ).hexdigest()
    return {
        "directory_count": sum(path.is_dir() for path in root.rglob("*")) + 1,
        "file_count": len(actual),
        "manifest_sha256": manifest_hash,
    }


def _portable_rename(source: Path, destination: Path) -> None:
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    os.rename(source, destination)


def _make_salvage_case(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> SimpleNamespace:
    source = _source_tree(tmp_path)
    allowed = tmp_path / "gpfs" / "immutable_sources_v1"
    allowed.parent.mkdir(parents=True)
    monkeypatch.setattr(checkpoint, "ALLOWED_DESTINATION_ROOT", allowed)
    monkeypatch.setattr(checkpoint, "_open_writers", lambda *_a, **_k: ())
    monkeypatch.setattr(checkpoint.os, "getuid", lambda: 0, raising=False)
    monkeypatch.setattr(checkpoint.os, "getgid", lambda: 0, raising=False)
    monkeypatch.setattr(checkpoint, "_assert_private_directory", lambda path: None)
    monkeypatch.setattr(checkpoint, "_fsync_directory", lambda path: None)
    monkeypatch.setattr(checkpoint, "_fsync_and_protect_tree", lambda path: None)
    monkeypatch.setattr(checkpoint, "_verify_protected_tree", _portable_verify)
    monkeypatch.setattr(checkpoint, "_rename_noreplace", _portable_rename)
    monkeypatch.setattr(
        checkpoint,
        "_filesystem_capacity",
        lambda path, **kwargs: {
            "anchor_device": 1,
            "bavail": 1_000_000,
            "block_size": 4096,
            "free_bytes": 4_096_000_000,
            "fsid": 2,
            "readonly": False,
            "required_free_bytes": kwargs["required_free_bytes"],
        },
    )
    sealed = checkpoint.snapshot_static_source(source)
    aedt = next(row for row in sealed.files if row.path.endswith(".aedt"))
    aedt_sha = hashlib.sha256((source / aedt.path).read_bytes()).hexdigest()
    encoded, quota_sha = _quota_evidence()
    destination = allowed / "sealed"
    checkpoint.stage_static_source(
        source=source,
        destination=destination,
        expected_metadata_sha256=sealed.metadata_sha256,
        expected_count=len(sealed.files),
        expected_directory_count=len(sealed.directories),
        expected_logical_bytes=sealed.logical_bytes,
        expected_allocated_bytes=sealed.allocated_bytes,
        expected_aedt_sha256=aedt_sha,
        checkpoint_tool_payload_sha256="b" * 64,
        login_quota_evidence_base64=encoded,
        login_quota_evidence_sha256=quota_sha,
        login_quota_evidence_max_age_seconds=120,
        minimum_soft_headroom_bytes=0,
        minimum_hard_headroom_bytes=0,
        minimum_soft_inode_headroom=0,
        minimum_hard_inode_headroom=0,
    )
    incoming = destination.with_name(destination.name + ".incoming")
    os.rename(destination, incoming)
    manifest_path = incoming / ".checkpoint_manifest.json"
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    return SimpleNamespace(
        aedt_sha=aedt_sha,
        allowed=allowed,
        destination=destination,
        encoded=encoded,
        incoming=incoming,
        manifest=manifest,
        manifest_payload_sha=manifest["manifest_payload_sha256"],
        manifest_sha=hashlib.sha256(manifest_raw).hexdigest(),
        quota_sha=quota_sha,
        sealed=sealed,
        source=source,
        tool_sha=hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest(),
    )


def _resume(case: SimpleNamespace, **overrides: Any) -> dict[str, Any]:
    values = {
        "source": case.source,
        "destination": case.destination,
        "expected_metadata_sha256": case.sealed.metadata_sha256,
        "expected_count": len(case.sealed.files),
        "expected_directory_count": len(case.sealed.directories),
        "expected_logical_bytes": case.sealed.logical_bytes,
        "expected_allocated_bytes": case.sealed.allocated_bytes,
        "expected_aedt_sha256": case.aedt_sha,
        "expected_manifest_sha256": case.manifest_sha,
        "expected_manifest_payload_sha256": case.manifest_payload_sha,
        "expected_checkpoint_tool_payload_sha256": "b" * 64,
        "resume_tool_payload_sha256": case.tool_sha,
        "login_quota_evidence_base64": case.encoded,
        "login_quota_evidence_sha256": case.quota_sha,
        "login_quota_evidence_max_age_seconds": 120,
        "minimum_soft_headroom_bytes": 0,
        "minimum_hard_headroom_bytes": 0,
        "minimum_soft_inode_headroom": 0,
        "minimum_hard_inode_headroom": 0,
    }
    values.update(overrides)
    return checkpoint.resume_protect_publish(**values)


def test_resume_authenticates_exact_25_plus_manifest_source_and_publishes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_salvage_case(monkeypatch, tmp_path)
    assert len(case.manifest["files"]) == 25
    monkeypatch.setattr(
        checkpoint,
        "_guarded_rename_noreplace",
        lambda source, destination, **_kwargs: (
            _portable_rename(source, destination)
            or {"method": "portable-noreplace-test"}
        ),
    )

    result = _resume(case)

    assert result["status"] == (
        "complete-awaiting-controller-login-quota-reauthentication"
    )
    assert result["incoming_preprotect_verification"]["file_count"] == 26
    assert result["post_publish_verification"]["file_count"] == 26
    assert result["manifest_sha256"] == case.manifest_sha
    assert case.destination.is_dir()
    assert not case.incoming.exists()


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("extra", "incoming children mismatch"),
        ("manifest", "incoming manifest SHA-256 mismatch"),
        ("content", "incoming content hash mismatch"),
        ("source", "source differs from sealed admission"),
        ("writer", "open for write"),
    ],
)
def test_resume_fails_closed_on_bad_tree_manifest_hash_source_or_writer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fault: str,
    message: str,
) -> None:
    case = _make_salvage_case(monkeypatch, tmp_path)
    if fault == "extra":
        (case.incoming / "unexpected.bin").write_bytes(b"extra")
    elif fault == "manifest":
        manifest = case.incoming / ".checkpoint_manifest.json"
        manifest.write_bytes(manifest.read_bytes() + b" ")
    elif fault == "content":
        relative = case.manifest["files"][0]["path"]
        target = case.incoming / relative
        original = target.read_bytes()
        target.write_bytes(bytes((original[0] ^ 1,)) + original[1:])
    elif fault == "source":
        target = case.source / f"{case.source.name}.aedt"
        target.write_bytes(target.read_bytes() + b"changed")
    elif fault == "writer":
        included = checkpoint.Writer(
            pid=1,
            fd=2,
            command="ansysedt",
            path=str(case.source / f"{case.source.name}.aedt"),
            included=True,
            flags_octal="0o2",
        )
        monkeypatch.setattr(
            checkpoint,
            "_open_writers",
            lambda *_a, **_k: (included,),
        )

    with pytest.raises(checkpoint.CheckpointError, match=message):
        _resume(case)

    assert case.incoming.is_dir()
    assert not case.destination.exists()


def test_resume_rejects_stale_quota_and_statvfs_failure_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_salvage_case(monkeypatch, tmp_path)
    stale_encoded, stale_sha = _quota_evidence(observed_at=time.time() - 500)
    with pytest.raises(checkpoint.CheckpointError, match="stale"):
        _resume(
            case,
            login_quota_evidence_base64=stale_encoded,
            login_quota_evidence_sha256=stale_sha,
        )
    assert case.incoming.is_dir()
    assert not case.destination.exists()

    monkeypatch.setattr(
        checkpoint,
        "_filesystem_capacity",
        lambda *_a, **_k: (_ for _ in ()).throw(
            checkpoint.CheckpointError("destination physical free below contract")
        ),
    )
    with pytest.raises(checkpoint.CheckpointError, match="physical free"):
        _resume(case)
    assert case.incoming.is_dir()
    assert not case.destination.exists()


def test_resume_final_collision_and_post_protect_failure_preserve_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_salvage_case(monkeypatch, tmp_path)
    case.destination.mkdir()
    marker = case.destination / "existing"
    marker.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        _resume(case)
    assert case.incoming.is_dir()
    assert marker.read_bytes() == b"keep"

    marker.unlink()
    case.destination.rmdir()
    monkeypatch.setattr(
        checkpoint,
        "_verify_protected_tree",
        lambda *_a, **_k: (_ for _ in ()).throw(
            checkpoint.CheckpointError("post-protect exact verification failed")
        ),
    )
    with pytest.raises(checkpoint.CheckpointError, match="post-protect"):
        _resume(case)
    assert case.incoming.is_dir()
    assert not case.destination.exists()
