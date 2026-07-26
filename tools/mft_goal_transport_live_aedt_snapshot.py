#!/usr/bin/env python3
"""Publish bounded text chunks for one authenticated live AEDT snapshot."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import sys
from typing import Any, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import mft_goal_snapshot_live_aedt as snapshot


SCHEMA = "mft-live-aedt-snapshot-transport-v1"
RESULT_SCHEMA = "mft-live-aedt-snapshot-transport-result-v1"
RAW_CHUNK_BYTES = 384_000
MAX_ENCODED_CHUNK_BYTES = 512_000


class TransportError(RuntimeError):
    """Raised when a snapshot transport cannot be authenticated."""


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > snapshot.MAX_MANIFEST_BYTES
    ):
        raise TransportError(f"{label} is not one bounded regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TransportError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise TransportError(f"{label} is not an object")
    return value


def _validate_seal(
    value: Mapping[str, Any],
    *,
    schema: str,
) -> dict[str, Any]:
    claimed = str(value.get("payload_sha256") or "")
    unsigned = dict(value)
    unsigned.pop("payload_sha256", None)
    if (
        value.get("schema") != schema
        or not snapshot.SHA256_RE.fullmatch(claimed)
        or snapshot.canonical_sha256(unsigned) != claimed
    ):
        raise TransportError(f"{schema} seal is invalid")
    return dict(value)


def _safe_snapshot_directory(
    cwd: Path,
    relative: str,
    *,
    expected_uid: int,
) -> Path:
    pure = snapshot._safe_relative_output(relative)
    raw = cwd.joinpath(*pure.parts)
    metadata = raw.lstat()
    if raw.is_symlink() or not raw.is_dir() or metadata.st_uid != expected_uid:
        raise TransportError("snapshot directory metadata is unsafe")
    resolved_cwd = cwd.resolve(strict=True)
    resolved = raw.resolve(strict=True)
    if resolved_cwd == resolved or resolved_cwd not in resolved.parents:
        raise TransportError("snapshot directory escaped the task workspace")
    return resolved


def _snapshot_contract(
    directory: Path,
    *,
    lane: str,
    source_node: str,
    source_task_id: int,
    source_slurm_job_id: str,
    artifact_sha256: str,
    artifact_size_bytes: int,
    expected_uid: int,
) -> tuple[dict[str, Any], Path, str]:
    manifest_path = directory / "manifest.json"
    snapshot._regular_identity(manifest_path, expected_uid=expected_uid)
    manifest_sha = snapshot.sha256_file(manifest_path)
    manifest = _validate_seal(
        _read_json(manifest_path, "snapshot manifest"),
        schema=snapshot.SCHEMA,
    )
    expected_artifact = f"{lane}.aedt"
    if (
        manifest.get("diagnostic_only") is not True
        or manifest.get("canonical") is not False
        or manifest.get("production_truth_eligible") is not False
        or manifest.get("solver_result_truth_included") is not False
        or manifest.get("lane") != lane
        or manifest.get("source_node") != source_node
        or manifest.get("source_task_id") != source_task_id
        or manifest.get("source_slurm_job_id") != source_slurm_job_id
        or manifest.get("artifact_filename") != expected_artifact
        or manifest.get("artifact_sha256") != artifact_sha256
        or manifest.get("artifact_size_bytes") != artifact_size_bytes
        or manifest.get("source_changed_during_snapshot") is not False
    ):
        raise TransportError("snapshot manifest contract drifted")
    artifact = directory / expected_artifact
    identity = snapshot._regular_identity(
        artifact, expected_uid=expected_uid
    )
    if (
        identity["size_bytes"] != artifact_size_bytes
        or snapshot.sha256_file(artifact) != artifact_sha256
    ):
        raise TransportError("snapshot artifact identity drifted")
    return manifest, artifact, manifest_sha


def _write_chunk(path: Path, payload: bytes) -> dict[str, Any]:
    encoded = base64.b64encode(payload)
    if len(encoded) > MAX_ENCODED_CHUNK_BYTES:
        raise TransportError("encoded snapshot chunk exceeded its bound")
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "filename": path.name,
        "raw_size_bytes": len(payload),
        "raw_sha256": hashlib.sha256(payload).hexdigest(),
        "encoded_size_bytes": len(encoded),
        "encoded_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _validate_existing(
    transport: Path,
    *,
    expected: Mapping[str, Any],
    expected_uid: int,
) -> dict[str, Any]:
    receipt_path = transport / "transport_receipt.json"
    snapshot._regular_identity(receipt_path, expected_uid=expected_uid)
    receipt = _validate_seal(
        _read_json(receipt_path, "transport receipt"),
        schema=SCHEMA,
    )
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise TransportError("existing transport receipt identity drifted")
    rows = receipt.get("chunks")
    if (
        not isinstance(rows, list)
        or len(rows) != receipt.get("chunk_count")
        or len(rows)
        != math.ceil(
            receipt["artifact_size_bytes"] / receipt["raw_chunk_bytes"]
        )
    ):
        raise TransportError("existing transport chunk inventory drifted")
    digest = hashlib.sha256()
    total = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TransportError("existing transport chunk row is invalid")
        expected_name = f"{index:08d}.b64"
        if row.get("filename") != expected_name:
            raise TransportError("existing transport chunk ordering drifted")
        path = transport / expected_name
        snapshot._regular_identity(path, expected_uid=expected_uid)
        encoded = path.read_bytes()
        if (
            len(encoded) != row.get("encoded_size_bytes")
            or hashlib.sha256(encoded).hexdigest()
            != row.get("encoded_sha256")
            or len(encoded) > MAX_ENCODED_CHUNK_BYTES
        ):
            raise TransportError("existing encoded chunk drifted")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise TransportError("existing encoded chunk is invalid") from exc
        if (
            len(raw) != row.get("raw_size_bytes")
            or hashlib.sha256(raw).hexdigest() != row.get("raw_sha256")
        ):
            raise TransportError("existing raw chunk drifted")
        digest.update(raw)
        total += len(raw)
    if (
        total != receipt["artifact_size_bytes"]
        or digest.hexdigest() != receipt["artifact_sha256"]
    ):
        raise TransportError("existing transport reconstruction drifted")
    return {
        "schema": RESULT_SCHEMA,
        "status": "existing_authenticated",
        "transport_directory": str(transport),
        "receipt_path": str(receipt_path),
        "receipt_sha256": snapshot.sha256_file(receipt_path),
        "chunk_count": len(rows),
        "artifact_size_bytes": total,
        "artifact_sha256": digest.hexdigest(),
        "payload_sha256": receipt["payload_sha256"],
    }


def transport_snapshot(
    *,
    snapshot_relative_directory: str,
    lane: str,
    source_task_id: int,
    source_slurm_job_id: str,
    artifact_sha256: str,
    artifact_size_bytes: int,
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
    task_id = snapshot._positive_int(
        env.get("SLURM_SCHED_TASK_ID"), "transport task ID"
    )
    source_task = snapshot._positive_int(source_task_id, "source task ID")
    source_job = snapshot._slurm_job_id(source_slurm_job_id)
    size = snapshot._positive_int(
        artifact_size_bytes, "artifact size bytes"
    )
    artifact_sha = snapshot._sha(artifact_sha256, "artifact SHA")
    if not snapshot.LANE_RE.fullmatch(str(lane or "")):
        raise TransportError("snapshot lane is unsafe")
    if (
        not snapshot.NODE_RE.fullmatch(actual_hostname)
        or str(env.get("SLURM_JOB_ID") or "") != source_job
    ):
        raise TransportError("transport same-allocation contract failed")
    if uid is None:
        getuid = getattr(os, "getuid", None)
        if not callable(getuid):
            raise TransportError("snapshot transport requires POSIX")
        uid = int(getuid())

    directory = _safe_snapshot_directory(
        actual_cwd, snapshot_relative_directory, expected_uid=uid
    )
    manifest, artifact, manifest_sha = _snapshot_contract(
        directory,
        lane=lane,
        source_node=actual_hostname,
        source_task_id=source_task,
        source_slurm_job_id=source_job,
        artifact_sha256=artifact_sha,
        artifact_size_bytes=size,
        expected_uid=uid,
    )
    transport = directory / "transport"
    expected = {
        "diagnostic_only": True,
        "canonical": False,
        "production_truth_eligible": False,
        "lane": lane,
        "source_task_id": source_task,
        "source_slurm_job_id": source_job,
        "snapshot_manifest_sha256": manifest_sha,
        "snapshot_manifest_payload_sha256": manifest["payload_sha256"],
        "artifact_filename": f"{lane}.aedt",
        "artifact_size_bytes": size,
        "artifact_sha256": artifact_sha,
        "raw_chunk_bytes": RAW_CHUNK_BYTES,
        "maximum_encoded_chunk_bytes": MAX_ENCODED_CHUNK_BYTES,
    }
    if transport.exists():
        return _validate_existing(
            transport, expected=expected, expected_uid=uid
        )

    staging = directory / f".transport.tmp-t{task_id}"
    staging.mkdir(mode=0o700)
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    total = 0
    try:
        source_before = snapshot._regular_identity(
            artifact, expected_uid=uid
        )
        with artifact.open("rb") as stream:
            while True:
                raw = stream.read(RAW_CHUNK_BYTES)
                if not raw:
                    break
                row = _write_chunk(
                    staging / f"{len(rows):08d}.b64", raw
                )
                rows.append(row)
                digest.update(raw)
                total += len(raw)
        source_after = snapshot._regular_identity(
            artifact, expected_uid=uid
        )
        if (
            source_after != source_before
            or total != size
            or digest.hexdigest() != artifact_sha
        ):
            raise TransportError("snapshot changed during chunk publication")
        receipt = {
            "schema": SCHEMA,
            **expected,
            "transport_task_id": task_id,
            "transport_slurm_job_id": str(env["SLURM_JOB_ID"]),
            "transport_slurm_step_id": (
                str(env.get("SLURM_STEP_ID") or "") or None
            ),
            "transport_node": actual_hostname,
            "chunk_count": len(rows),
            "chunks": rows,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        receipt["payload_sha256"] = snapshot.canonical_sha256(receipt)
        snapshot._atomic_json(staging / "transport_receipt.json", receipt)
        if os.name == "posix":
            handle = os.open(
                staging,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(handle)
            finally:
                os.close(handle)
        for path in staging.iterdir():
            os.chmod(path, 0o400)
        os.chmod(staging, 0o500)
        os.replace(staging, transport)
        if os.name == "posix":
            handle = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(handle)
            finally:
                os.close(handle)
    except BaseException:
        if staging.exists():
            try:
                os.chmod(staging, 0o700)
            except OSError:
                pass
            shutil.rmtree(staging)
        raise

    published_receipt = transport / "transport_receipt.json"
    return {
        "schema": RESULT_SCHEMA,
        "status": "published",
        "transport_directory": str(transport),
        "receipt_path": str(published_receipt),
        "receipt_sha256": snapshot.sha256_file(published_receipt),
        "chunk_count": len(rows),
        "artifact_size_bytes": total,
        "artifact_sha256": digest.hexdigest(),
        "payload_sha256": receipt["payload_sha256"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-relative-directory", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--source-task-id", required=True, type=int)
    parser.add_argument("--source-slurm-job-id", required=True)
    parser.add_argument("--artifact-sha256", required=True)
    parser.add_argument("--artifact-size-bytes", required=True, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = transport_snapshot(
            snapshot_relative_directory=args.snapshot_relative_directory,
            lane=args.lane,
            source_task_id=args.source_task_id,
            source_slurm_job_id=args.source_slurm_job_id,
            artifact_sha256=args.artifact_sha256,
            artifact_size_bytes=args.artifact_size_bytes,
        )
    except (OSError, snapshot.SnapshotError, TransportError) as exc:
        print(
            f"LIVE_AEDT_TRANSPORT_ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
