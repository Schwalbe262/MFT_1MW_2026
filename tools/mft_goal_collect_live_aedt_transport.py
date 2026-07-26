#!/usr/bin/env python3
"""Collect and authenticate a chunked live AEDT snapshot transport."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import mft_goal_snapshot_live_aedt as snapshot
from tools import mft_goal_transport_live_aedt_snapshot as transport


SCHEMA = "mft-live-aedt-snapshot-local-collection-v1"
RESULT_SCHEMA = "mft-live-aedt-snapshot-local-collection-result-v1"
CHUNK_NAME_RE = re.compile(r"^[0-9]{8}[.]b64$")
MAX_RECEIPT_BYTES = 1_048_576
MAX_WORKERS = 8
MAX_DOWNLOAD_ATTEMPTS = 8


class CollectionError(RuntimeError):
    """Raised when an AEDT transport cannot be collected safely."""


def _safe_remote_directory(value: str) -> str:
    raw = str(value or "")
    pure = PurePosixPath(raw)
    if (
        not raw
        or "\\" in raw
        or "\x00" in raw
        or pure.is_absolute()
        or ".." in pure.parts
        or len(pure.parts) < 2
        or any(part in {"", "."} for part in pure.parts)
    ):
        raise CollectionError("remote transport directory is unsafe")
    return pure.as_posix()


def _scheduler_url(value: str) -> str:
    raw = str(value or "").rstrip("/")
    parsed = urlsplit(raw)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise CollectionError("scheduler URL is invalid")
    return raw


def _bounded_file(path: Path, *, maximum_bytes: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise CollectionError(f"local file metadata is unsafe: {path.name}")
    size = path.stat().st_size
    if not 0 < size <= maximum_bytes:
        raise CollectionError(f"local file size is unsafe: {path.name}")
    return path.read_bytes()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        if temporary.is_symlink() or not temporary.is_file():
            raise CollectionError("local temporary path is unsafe")
        temporary.unlink()
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _fetch_remote_bytes(
    *,
    scheduler_url: str,
    task_id: int,
    relative_path: str,
    maximum_bytes: int,
) -> bytes:
    query = urlencode(
        {"path": relative_path, "max_bytes": maximum_bytes}
    )
    url = (
        f"{scheduler_url}/api/tasks/{task_id}/remote-file?{query}"
    )
    last_error: BaseException | None = None
    for attempt in range(MAX_DOWNLOAD_ATTEMPTS):
        try:
            request = Request(
                url,
                headers={
                    "Accept": "text/plain",
                    "Cache-Control": "no-cache",
                    "User-Agent": "mft-goal-aedt-collector/1",
                },
            )
            with urlopen(request, timeout=20) as response:
                payload = response.read(maximum_bytes + 1)
            if not 0 < len(payload) <= maximum_bytes:
                raise CollectionError("remote file size is outside its bound")
            return payload
        except (
            CollectionError,
            HTTPError,
            TimeoutError,
            URLError,
            OSError,
        ) as exc:
            last_error = exc
            if attempt + 1 < MAX_DOWNLOAD_ATTEMPTS:
                time.sleep(min(0.5 * (attempt + 1), 3.0))
    raise CollectionError(
        f"remote file download failed: {relative_path}"
    ) from last_error


def _receipt_contract(
    receipt_bytes: bytes,
    *,
    expected_receipt_sha256: str,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    if receipt_sha != expected_receipt_sha256:
        raise CollectionError("transport receipt SHA drifted")
    try:
        value = json.loads(receipt_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CollectionError("transport receipt is invalid JSON") from exc
    if not isinstance(value, dict):
        raise CollectionError("transport receipt is not an object")
    try:
        receipt = transport._validate_seal(
            value,
            schema=transport.SCHEMA,
        )
    except transport.TransportError as exc:
        raise CollectionError("transport receipt seal is invalid") from exc
    if any(receipt.get(key) != item for key, item in expected.items()):
        raise CollectionError("transport receipt contract drifted")
    rows = receipt.get("chunks")
    if not isinstance(rows, list) or len(rows) != receipt.get("chunk_count"):
        raise CollectionError("transport receipt chunk inventory drifted")
    if not 0 < len(rows) <= 100_000:
        raise CollectionError("transport receipt chunk count is unsafe")
    for index, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or row.get("filename") != f"{index:08d}.b64"
            or not CHUNK_NAME_RE.fullmatch(str(row.get("filename") or ""))
            or not 0 < int(row.get("encoded_size_bytes") or 0)
            <= transport.MAX_ENCODED_CHUNK_BYTES
            or not 0 < int(row.get("raw_size_bytes") or 0)
            <= transport.RAW_CHUNK_BYTES
            or not snapshot.SHA256_RE.fullmatch(
                str(row.get("encoded_sha256") or "")
            )
            or not snapshot.SHA256_RE.fullmatch(
                str(row.get("raw_sha256") or "")
            )
        ):
            raise CollectionError("transport receipt chunk row is unsafe")
    return receipt


def _authenticated_chunk(
    encoded: bytes,
    *,
    row: Mapping[str, Any],
) -> bytes:
    if (
        len(encoded) != row["encoded_size_bytes"]
        or hashlib.sha256(encoded).hexdigest()
        != row["encoded_sha256"]
    ):
        raise CollectionError("encoded chunk identity drifted")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise CollectionError("encoded chunk is invalid base64") from exc
    if (
        len(raw) != row["raw_size_bytes"]
        or hashlib.sha256(raw).hexdigest() != row["raw_sha256"]
    ):
        raise CollectionError("raw chunk identity drifted")
    return raw


def collect_transport(
    *,
    scheduler_url: str,
    task_id: int,
    remote_transport_relative_directory: str,
    local_directory: Path,
    expected_lane: str,
    expected_source_task_id: int,
    expected_source_slurm_job_id: str,
    expected_transport_task_id: int,
    expected_transport_node: str,
    expected_artifact_sha256: str,
    expected_artifact_size_bytes: int,
    expected_receipt_sha256: str,
    expected_snapshot_manifest_sha256: str,
    expected_snapshot_manifest_payload_sha256: str,
    max_workers: int = 4,
) -> dict[str, Any]:
    base_url = _scheduler_url(scheduler_url)
    remote_directory = _safe_remote_directory(
        remote_transport_relative_directory
    )
    task = snapshot._positive_int(task_id, "collector task ID")
    source_task = snapshot._positive_int(
        expected_source_task_id, "source task ID"
    )
    transport_task = snapshot._positive_int(
        expected_transport_task_id, "transport task ID"
    )
    source_job = snapshot._slurm_job_id(expected_source_slurm_job_id)
    size = snapshot._positive_int(
        expected_artifact_size_bytes, "artifact size"
    )
    artifact_sha = snapshot._sha(
        expected_artifact_sha256, "artifact SHA"
    )
    receipt_sha = snapshot._sha(
        expected_receipt_sha256, "transport receipt SHA"
    )
    snapshot_manifest_sha = snapshot._sha(
        expected_snapshot_manifest_sha256, "snapshot manifest SHA"
    )
    snapshot_manifest_payload = snapshot._sha(
        expected_snapshot_manifest_payload_sha256,
        "snapshot manifest payload SHA",
    )
    if (
        task != transport_task
        or not snapshot.LANE_RE.fullmatch(str(expected_lane or ""))
        or not snapshot.NODE_RE.fullmatch(
            str(expected_transport_node or "")
        )
        or not 1 <= int(max_workers) <= MAX_WORKERS
    ):
        raise CollectionError("local collection contract is unsafe")

    output = local_directory.resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    if output.is_symlink() or not output.is_dir():
        raise CollectionError("local collection directory is unsafe")
    receipt_path = output / "transport_receipt.json"
    if receipt_path.exists():
        receipt_bytes = _bounded_file(
            receipt_path, maximum_bytes=MAX_RECEIPT_BYTES
        )
    else:
        receipt_bytes = _fetch_remote_bytes(
            scheduler_url=base_url,
            task_id=task,
            relative_path=f"{remote_directory}/transport_receipt.json",
            maximum_bytes=MAX_RECEIPT_BYTES,
        )
        if hashlib.sha256(receipt_bytes).hexdigest() != receipt_sha:
            raise CollectionError("downloaded transport receipt SHA drifted")
        _atomic_bytes(receipt_path, receipt_bytes)

    expected = {
        "diagnostic_only": True,
        "canonical": False,
        "production_truth_eligible": False,
        "lane": expected_lane,
        "source_task_id": source_task,
        "source_slurm_job_id": source_job,
        "transport_task_id": transport_task,
        "transport_node": expected_transport_node,
        "transport_slurm_job_id": source_job,
        "snapshot_manifest_sha256": snapshot_manifest_sha,
        "snapshot_manifest_payload_sha256": snapshot_manifest_payload,
        "artifact_filename": f"{expected_lane}.aedt",
        "artifact_size_bytes": size,
        "artifact_sha256": artifact_sha,
        "raw_chunk_bytes": transport.RAW_CHUNK_BYTES,
        "maximum_encoded_chunk_bytes": (
            transport.MAX_ENCODED_CHUNK_BYTES
        ),
    }
    receipt = _receipt_contract(
        receipt_bytes,
        expected_receipt_sha256=receipt_sha,
        expected=expected,
    )
    chunks_directory = output / "transport_chunks"
    chunks_directory.mkdir(mode=0o700, exist_ok=True)

    def obtain(row: Mapping[str, Any]) -> str:
        filename = str(row["filename"])
        local_path = chunks_directory / filename
        if local_path.exists():
            encoded = _bounded_file(
                local_path,
                maximum_bytes=transport.MAX_ENCODED_CHUNK_BYTES,
            )
            _authenticated_chunk(encoded, row=row)
            return filename
        last_error: BaseException | None = None
        for attempt in range(MAX_DOWNLOAD_ATTEMPTS):
            try:
                encoded = _fetch_remote_bytes(
                    scheduler_url=base_url,
                    task_id=task,
                    relative_path=f"{remote_directory}/{filename}",
                    maximum_bytes=transport.MAX_ENCODED_CHUNK_BYTES,
                )
                _authenticated_chunk(encoded, row=row)
                _atomic_bytes(local_path, encoded)
                return filename
            except CollectionError as exc:
                last_error = exc
                if attempt + 1 < MAX_DOWNLOAD_ATTEMPTS:
                    time.sleep(min(0.5 * (attempt + 1), 3.0))
        raise CollectionError(
            f"chunk download could not be authenticated: {filename}"
        ) from last_error

    with ThreadPoolExecutor(max_workers=int(max_workers)) as executor:
        downloaded = list(executor.map(obtain, receipt["chunks"]))
    if downloaded != [row["filename"] for row in receipt["chunks"]]:
        raise CollectionError("local chunk ordering drifted")

    artifact = output / receipt["artifact_filename"]
    if artifact.exists():
        if (
            artifact.is_symlink()
            or not artifact.is_file()
            or artifact.stat().st_size != size
            or snapshot.sha256_file(artifact) != artifact_sha
        ):
            raise CollectionError("existing local AEDT identity drifted")
        status = "existing_authenticated"
    else:
        temporary = artifact.with_name(f".{artifact.name}.tmp")
        if temporary.exists():
            if temporary.is_symlink() or not temporary.is_file():
                raise CollectionError("local AEDT temporary path is unsafe")
            temporary.unlink()
        digest = hashlib.sha256()
        total = 0
        with temporary.open("xb") as stream:
            for row in receipt["chunks"]:
                encoded = _bounded_file(
                    chunks_directory / row["filename"],
                    maximum_bytes=transport.MAX_ENCODED_CHUNK_BYTES,
                )
                raw = _authenticated_chunk(encoded, row=row)
                stream.write(raw)
                digest.update(raw)
                total += len(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if total != size or digest.hexdigest() != artifact_sha:
            temporary.unlink()
            raise CollectionError("reconstructed local AEDT identity drifted")
        os.replace(temporary, artifact)
        status = "collected"

    collection = {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "canonical": False,
        "production_truth_eligible": False,
        "solver_result_truth_included": False,
        "status": status,
        "lane": expected_lane,
        "source_task_id": source_task,
        "source_slurm_job_id": source_job,
        "transport_task_id": transport_task,
        "transport_node": expected_transport_node,
        "remote_transport_relative_directory": remote_directory,
        "transport_receipt_sha256": receipt_sha,
        "transport_receipt_payload_sha256": receipt["payload_sha256"],
        "snapshot_manifest_sha256": snapshot_manifest_sha,
        "snapshot_manifest_payload_sha256": snapshot_manifest_payload,
        "chunk_count": len(receipt["chunks"]),
        "artifact_filename": artifact.name,
        "artifact_size_bytes": artifact.stat().st_size,
        "artifact_sha256": snapshot.sha256_file(artifact),
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    collection["payload_sha256"] = snapshot.canonical_sha256(collection)
    collection_path = output / "local_collection_receipt.json"
    _atomic_bytes(
        collection_path,
        (
            json.dumps(
                collection,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8"),
    )
    return {
        "schema": RESULT_SCHEMA,
        "status": status,
        "artifact_path": str(artifact),
        "artifact_size_bytes": collection["artifact_size_bytes"],
        "artifact_sha256": collection["artifact_sha256"],
        "chunk_count": collection["chunk_count"],
        "collection_receipt_path": str(collection_path),
        "collection_receipt_sha256": snapshot.sha256_file(
            collection_path
        ),
        "payload_sha256": collection["payload_sha256"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler-url", required=True)
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument(
        "--remote-transport-relative-directory", required=True
    )
    parser.add_argument("--local-directory", required=True, type=Path)
    parser.add_argument("--expected-lane", required=True)
    parser.add_argument("--expected-source-task-id", required=True, type=int)
    parser.add_argument("--expected-source-slurm-job-id", required=True)
    parser.add_argument(
        "--expected-transport-task-id", required=True, type=int
    )
    parser.add_argument("--expected-transport-node", required=True)
    parser.add_argument("--expected-artifact-sha256", required=True)
    parser.add_argument(
        "--expected-artifact-size-bytes", required=True, type=int
    )
    parser.add_argument("--expected-receipt-sha256", required=True)
    parser.add_argument(
        "--expected-snapshot-manifest-sha256", required=True
    )
    parser.add_argument(
        "--expected-snapshot-manifest-payload-sha256", required=True
    )
    parser.add_argument("--max-workers", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = collect_transport(
            scheduler_url=args.scheduler_url,
            task_id=args.task_id,
            remote_transport_relative_directory=(
                args.remote_transport_relative_directory
            ),
            local_directory=args.local_directory,
            expected_lane=args.expected_lane,
            expected_source_task_id=args.expected_source_task_id,
            expected_source_slurm_job_id=(
                args.expected_source_slurm_job_id
            ),
            expected_transport_task_id=args.expected_transport_task_id,
            expected_transport_node=args.expected_transport_node,
            expected_artifact_sha256=args.expected_artifact_sha256,
            expected_artifact_size_bytes=(
                args.expected_artifact_size_bytes
            ),
            expected_receipt_sha256=args.expected_receipt_sha256,
            expected_snapshot_manifest_sha256=(
                args.expected_snapshot_manifest_sha256
            ),
            expected_snapshot_manifest_payload_sha256=(
                args.expected_snapshot_manifest_payload_sha256
            ),
            max_workers=args.max_workers,
        )
    except (
        CollectionError,
        OSError,
        snapshot.SnapshotError,
        transport.TransportError,
        ValueError,
    ) as exc:
        print(
            f"LIVE_AEDT_COLLECTION_ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
