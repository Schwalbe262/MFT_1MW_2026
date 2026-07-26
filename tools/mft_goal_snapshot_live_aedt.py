#!/usr/bin/env python3
"""Atomically preserve one saved AEDT project from a live Slurm node.

The worker is intentionally solver-read-only.  It copies exactly one regular
``.aedt`` file from an exact node-local source root, proves that the source did
not change during the copy, and atomically publishes a sealed diagnostic
snapshot below the task account's durable workspace.  It never reads or edits
AEDT through automation and never treats the snapshot as solver-result truth.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import stat
import sys
from typing import Any, Mapping


SCHEMA = "mft-live-aedt-snapshot-v1"
RESULT_SCHEMA = "mft-live-aedt-snapshot-result-v1"
MAX_MANIFEST_BYTES = 1024 * 1024
COPY_CHUNK_BYTES = 8 * 1024 * 1024
SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_SHA1_RE = re.compile(r"[0-9a-f]{40}")
LANE_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
SLURM_JOB_ID_RE = re.compile(r"[1-9][0-9]{0,19}")
SAFE_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
NODE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class SnapshotError(RuntimeError):
    """Raised when a live project cannot be snapshotted safely."""


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(COPY_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise SnapshotError(f"{label} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SnapshotError(f"{label} must be a positive integer") from exc
    if parsed <= 0:
        raise SnapshotError(f"{label} must be a positive integer")
    return parsed


def _sha(value: Any, label: str) -> str:
    normalized = str(value or "").strip().casefold()
    if not SHA256_RE.fullmatch(normalized):
        raise SnapshotError(f"{label} must be one lowercase SHA-256")
    return normalized


def _git_revision(value: Any, label: str) -> str:
    normalized = str(value or "").strip().casefold()
    if not GIT_SHA1_RE.fullmatch(normalized):
        raise SnapshotError(f"{label} must be one lowercase Git SHA-1")
    return normalized


def _slurm_job_id(value: Any) -> str:
    normalized = str(value or "").strip()
    if not SLURM_JOB_ID_RE.fullmatch(normalized):
        raise SnapshotError("expected Slurm job ID must be one positive integer")
    return normalized


def _identity_from_stat(
    metadata: os.stat_result,
    *,
    expected_uid: int,
    path: Path,
) -> dict[str, int]:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != expected_uid
        or metadata.st_nlink != 1
    ):
        raise SnapshotError(f"unsafe live AEDT source metadata: {path}")
    return {
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "size_bytes": int(metadata.st_size),
        "mtime_ns": int(metadata.st_mtime_ns),
        "ctime_ns": int(metadata.st_ctime_ns),
        "uid": int(metadata.st_uid),
        "mode": int(stat.S_IMODE(metadata.st_mode)),
        "nlink": int(metadata.st_nlink),
    }


def _regular_identity(path: Path, *, expected_uid: int) -> dict[str, int]:
    metadata = path.lstat()
    if path.is_symlink():
        raise SnapshotError(f"unsafe live AEDT source metadata: {path}")
    return _identity_from_stat(
        metadata,
        expected_uid=expected_uid,
        path=path,
    )


def _safe_source_root(path: Path, *, expected_uid: int) -> Path:
    raw_metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISDIR(raw_metadata.st_mode):
        raise SnapshotError("live AEDT source root is not a real directory")
    root = path.resolve(strict=True)
    metadata = root.lstat()
    if (
        root.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != expected_uid
        or root == Path("/enroot")
        or Path("/enroot") not in root.parents
    ):
        raise SnapshotError("live AEDT source root is outside a safe /enroot child")
    return root


def _safe_relative_output(path: str) -> PurePosixPath:
    raw = str(path or "")
    pure = PurePosixPath(raw)
    if (
        not raw
        or "\\" in raw
        or "\x00" in raw
        or pure.is_absolute()
        or ".." in pure.parts
        or len(pure.parts) < 2
        or any(part in {"", "."} for part in pure.parts)
        or any(not SAFE_COMPONENT_RE.fullmatch(part) for part in pure.parts)
    ):
        raise SnapshotError("output root must be a scoped relative path")
    return pure


def _find_exact_project(
    root: Path,
    *,
    expected_name: str,
    expected_uid: int,
    maximum_bytes: int,
) -> tuple[Path, dict[str, int]]:
    if (
        not expected_name.casefold().endswith(".aedt")
        or Path(expected_name).name != expected_name
        or "/" in expected_name
        or "\\" in expected_name
        or "\x00" in expected_name
    ):
        raise SnapshotError("expected AEDT filename is unsafe")
    candidates: list[Path] = []

    def fail_walk(error: OSError) -> None:
        raise SnapshotError(
            f"live AEDT source inventory could not be enumerated: {error}"
        ) from error

    for directory, directory_names, file_names in os.walk(
        root,
        followlinks=False,
        onerror=fail_walk,
    ):
        base = Path(directory)
        directory_names[:] = [
            name for name in directory_names if not (base / name).is_symlink()
        ]
        for name in file_names:
            if name.casefold().endswith(".aedt"):
                candidates.append(base / name)
    if len(candidates) != 1:
        raise SnapshotError(
            f"live AEDT source inventory is not exact: {len(candidates)}"
        )
    candidate = candidates[0]
    candidate_identity = _regular_identity(
        candidate,
        expected_uid=expected_uid,
    )
    source = candidate.resolve(strict=True)
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise SnapshotError("live AEDT source escaped its root") from exc
    if source.name != expected_name:
        raise SnapshotError(
            f"live AEDT filename drifted: {source.name!r} != {expected_name!r}"
        )
    identity = _regular_identity(source, expected_uid=expected_uid)
    if identity != candidate_identity:
        raise SnapshotError("live AEDT source changed during inventory")
    if not 0 < identity["size_bytes"] <= maximum_bytes:
        raise SnapshotError("live AEDT source size is outside its sealed bound")
    return source, identity


def _atomic_json(path: Path, value: Mapping[str, Any]) -> Path:
    payload = (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    if len(payload) > MAX_MANIFEST_BYTES:
        raise SnapshotError("snapshot manifest exceeds its bounded size")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def _validate_existing(
    destination: Path,
    *,
    expected: Mapping[str, Any],
    expected_uid: int,
) -> dict[str, Any]:
    manifest_path = destination / "manifest.json"
    if (
        destination.is_symlink()
        or not destination.is_dir()
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.stat().st_size > MAX_MANIFEST_BYTES
    ):
        raise SnapshotError("existing live AEDT snapshot is incomplete")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SnapshotError("existing live AEDT snapshot manifest is invalid") from exc
    if not isinstance(manifest, dict):
        raise SnapshotError("existing live AEDT snapshot manifest is not an object")
    claimed = str(manifest.get("payload_sha256") or "")
    unsigned = dict(manifest)
    unsigned.pop("payload_sha256", None)
    if (
        manifest.get("schema") != SCHEMA
        or not SHA256_RE.fullmatch(claimed)
        or canonical_sha256(unsigned) != claimed
        or any(manifest.get(key) != value for key, value in expected.items())
        or manifest.get("diagnostic_only") is not True
        or manifest.get("canonical") is not False
        or manifest.get("production_truth_eligible") is not False
        or manifest.get("solver_result_truth_included") is not False
    ):
        raise SnapshotError("existing live AEDT snapshot identity drifted")
    artifact_filename = f"{expected['lane']}.aedt"
    if manifest.get("artifact_filename") != artifact_filename:
        raise SnapshotError("existing live AEDT snapshot artifact name drifted")
    artifact = destination / artifact_filename
    if (
        artifact.is_symlink()
        or not artifact.is_file()
        or artifact.stat().st_size != manifest.get("artifact_size_bytes")
        or sha256_file(artifact) != manifest.get("artifact_sha256")
    ):
        raise SnapshotError("existing live AEDT snapshot artifact drifted")
    _regular_identity(manifest_path, expected_uid=expected_uid)
    _regular_identity(artifact, expected_uid=expected_uid)
    return {
        "schema": RESULT_SCHEMA,
        "status": "existing_authenticated",
        "destination": str(destination),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "artifact_path": str(artifact),
        "artifact_sha256": manifest["artifact_sha256"],
        "artifact_size_bytes": manifest["artifact_size_bytes"],
        "payload_sha256": manifest["payload_sha256"],
    }


def snapshot_live_aedt(
    *,
    source_root: Path,
    output_relative_root: str,
    lane: str,
    source_task_id: int,
    expected_node: str,
    expected_slurm_job_id: str,
    expected_project_filename: str,
    maximum_bytes: int,
    source_solver_revision: str,
    snapshot_tool_revision: str,
    candidate_sha256: str,
    fixed_physics_sha256: str,
    environ: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    hostname: str | None = None,
    uid: int | None = None,
) -> dict[str, Any]:
    env = dict(os.environ if environ is None else environ)
    actual_cwd = Path.cwd() if cwd is None else cwd
    actual_hostname = (
        socket.gethostname().split(".", 1)[0] if hostname is None else hostname
    )
    source_task = _positive_int(source_task_id, "source task ID")
    snapshot_task = _positive_int(
        env.get("SLURM_SCHED_TASK_ID"), "snapshot task ID"
    )
    maximum = _positive_int(maximum_bytes, "maximum AEDT bytes")
    expected_job = _slurm_job_id(expected_slurm_job_id)
    if not LANE_RE.fullmatch(str(lane or "")):
        raise SnapshotError("snapshot lane is unsafe")
    if (
        not NODE_RE.fullmatch(str(expected_node or ""))
        or actual_hostname != expected_node
        or str(env.get("SLURM_JOB_ID") or "") != expected_job
    ):
        raise SnapshotError(
            "same-allocation placement contract failed: "
            f"host={actual_hostname!r}, job={env.get('SLURM_JOB_ID')!r}"
        )
    if _positive_int(
        env.get("SLURM_CPUS_PER_TASK"), "Slurm CPUs per task"
    ) < 1:
        raise SnapshotError("snapshot task has no Slurm CPU")

    if uid is None:
        getuid = getattr(os, "getuid", None)
        if not callable(getuid):
            raise SnapshotError("live AEDT snapshot requires a POSIX runtime")
        uid = int(getuid())
    source_base = _safe_source_root(source_root, expected_uid=uid)
    source, source_before = _find_exact_project(
        source_base,
        expected_name=expected_project_filename,
        expected_uid=uid,
        maximum_bytes=maximum,
    )
    source_revision = _git_revision(
        source_solver_revision, "source solver revision"
    )
    tool_revision = _git_revision(
        snapshot_tool_revision, "snapshot tool revision"
    )
    candidate = _sha(candidate_sha256, "candidate SHA")
    physics = _sha(fixed_physics_sha256, "fixed-physics SHA")

    relative_output = _safe_relative_output(output_relative_root)
    workspace = actual_cwd.resolve(strict=True)
    output_root = workspace.joinpath(*relative_output.parts)
    output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    output_root = output_root.resolve(strict=True)
    if workspace == output_root or workspace not in output_root.parents:
        raise SnapshotError("snapshot output escaped the task workspace")
    destination = output_root / (
        f"{lane}-from-t{source_task}-j{expected_job}"
    )
    expected_existing = {
        "lane": lane,
        "source_task_id": source_task,
        "source_slurm_job_id": expected_job,
        "source_node": expected_node,
        "source_project_filename": expected_project_filename,
        "source_solver_revision": source_revision,
        "snapshot_tool_revision": tool_revision,
        "candidate_sha256": candidate,
        "fixed_physics_sha256": physics,
    }
    if destination.exists():
        return _validate_existing(
            destination,
            expected=expected_existing,
            expected_uid=uid,
        )

    staging = output_root / (
        f".{destination.name}.tmp-snapshot-t{snapshot_task}"
    )
    staging.mkdir(mode=0o700)
    artifact_name = f"{lane}.aedt"
    artifact = staging / artifact_name
    try:
        digest = hashlib.sha256()
        copied = 0
        open_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        open_flags |= getattr(os, "O_NOFOLLOW", 0)
        source_fd = os.open(source, open_flags)
        with os.fdopen(source_fd, "rb") as reader, artifact.open("xb") as writer:
            opened_identity = _identity_from_stat(
                os.fstat(reader.fileno()),
                expected_uid=uid,
                path=source,
            )
            if opened_identity != source_before:
                raise SnapshotError("live AEDT source changed before snapshot")
            while True:
                chunk = reader.read(COPY_CHUNK_BYTES)
                if not chunk:
                    break
                writer.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
                if copied > maximum:
                    raise SnapshotError(
                        "live AEDT source grew beyond its sealed bound"
                    )
            writer.flush()
            os.fsync(writer.fileno())
            descriptor_after = _identity_from_stat(
                os.fstat(reader.fileno()),
                expected_uid=uid,
                path=source,
            )
        source_after = _regular_identity(source, expected_uid=uid)
        if (
            descriptor_after != source_before
            or source_after != source_before
            or copied != source_before["size_bytes"]
        ):
            raise SnapshotError("live AEDT source changed during snapshot")
        artifact_sha = digest.hexdigest()
        if (
            artifact.stat().st_size != copied
            or sha256_file(artifact) != artifact_sha
        ):
            raise SnapshotError("live AEDT destination verification failed")

        manifest = {
            "schema": SCHEMA,
            "diagnostic_only": True,
            "canonical": False,
            "production_truth_eligible": False,
            "solver_result_truth_included": False,
            "snapshot_semantics": (
                "saved-project bytes captured while the source solver task "
                "remained active; no AEDT automation or source mutation"
            ),
            **expected_existing,
            "snapshot_task_id": snapshot_task,
            "snapshot_slurm_job_id": str(env["SLURM_JOB_ID"]),
            "snapshot_slurm_step_id": (
                str(env.get("SLURM_STEP_ID") or "") or None
            ),
            "artifact_filename": artifact_name,
            "artifact_size_bytes": copied,
            "artifact_sha256": artifact_sha,
            "source_path": str(source),
            "source_identity_before": source_before,
            "source_identity_after": source_after,
            "source_changed_during_snapshot": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest["payload_sha256"] = canonical_sha256(manifest)
        manifest_path = _atomic_json(staging / "manifest.json", manifest)
        if os.name == "posix":
            staging_handle = os.open(
                staging,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(staging_handle)
            finally:
                os.close(staging_handle)
        os.chmod(artifact, 0o400)
        os.chmod(manifest_path, 0o400)
        os.chmod(staging, 0o500)
        os.replace(staging, destination)
        if os.name == "posix":
            directory_handle = os.open(output_root, os.O_RDONLY)
            try:
                os.fsync(directory_handle)
            finally:
                os.close(directory_handle)
    except BaseException:
        if staging.exists():
            try:
                os.chmod(staging, 0o700)
            except OSError:
                pass
            shutil.rmtree(staging)
        raise

    published_manifest = destination / "manifest.json"
    published_artifact = destination / artifact_name
    return {
        "schema": RESULT_SCHEMA,
        "status": "published",
        "diagnostic_only": True,
        "canonical": False,
        "production_truth_eligible": False,
        "destination": str(destination),
        "manifest_path": str(published_manifest),
        "manifest_sha256": sha256_file(published_manifest),
        "artifact_path": str(published_artifact),
        "artifact_sha256": sha256_file(published_artifact),
        "artifact_size_bytes": published_artifact.stat().st_size,
        "payload_sha256": manifest["payload_sha256"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-relative-root", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--source-task-id", required=True, type=int)
    parser.add_argument("--expected-node", required=True)
    parser.add_argument("--expected-slurm-job-id", required=True)
    parser.add_argument("--expected-project-filename", required=True)
    parser.add_argument("--maximum-bytes", required=True, type=int)
    parser.add_argument("--source-solver-revision", required=True)
    parser.add_argument("--snapshot-tool-revision", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--fixed-physics-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = snapshot_live_aedt(
            source_root=args.source_root,
            output_relative_root=args.output_relative_root,
            lane=args.lane,
            source_task_id=args.source_task_id,
            expected_node=args.expected_node,
            expected_slurm_job_id=args.expected_slurm_job_id,
            expected_project_filename=args.expected_project_filename,
            maximum_bytes=args.maximum_bytes,
            source_solver_revision=args.source_solver_revision,
            snapshot_tool_revision=args.snapshot_tool_revision,
            candidate_sha256=args.candidate_sha256,
            fixed_physics_sha256=args.fixed_physics_sha256,
        )
    except (OSError, SnapshotError) as exc:
        print(
            f"LIVE_AEDT_SNAPSHOT_ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
