from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from tools import mft_goal_collect_live_aedt_transport as collector
from tools import mft_goal_snapshot_live_aedt as snapshot
from tools import mft_goal_transport_live_aedt_snapshot as transport


def test_collect_transport_round_trip_is_idempotent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw = (b"AEDT-local-collection" * 30_000) + b"tail"
    rows = []
    remote: dict[str, bytes] = {}
    remote_directory = "campaign/transports/full-v1"
    for index, offset in enumerate(
        range(0, len(raw), transport.RAW_CHUNK_BYTES)
    ):
        chunk = raw[offset : offset + transport.RAW_CHUNK_BYTES]
        encoded = base64.b64encode(chunk)
        filename = f"{index:08d}.b64"
        rows.append(
            {
                "filename": filename,
                "raw_size_bytes": len(chunk),
                "raw_sha256": hashlib.sha256(chunk).hexdigest(),
                "encoded_size_bytes": len(encoded),
                "encoded_sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
        remote[f"{remote_directory}/{filename}"] = encoded
    receipt = {
        "schema": transport.SCHEMA,
        "diagnostic_only": True,
        "canonical": False,
        "production_truth_eligible": False,
        "lane": "full",
        "source_task_id": 96307,
        "source_slurm_job_id": "829952",
        "transport_task_id": 96318,
        "transport_node": "n116",
        "transport_slurm_job_id": "829952",
        "snapshot_manifest_sha256": "1" * 64,
        "snapshot_manifest_payload_sha256": "2" * 64,
        "artifact_filename": "full.aedt",
        "artifact_size_bytes": len(raw),
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_chunk_bytes": transport.RAW_CHUNK_BYTES,
        "maximum_encoded_chunk_bytes": (
            transport.MAX_ENCODED_CHUNK_BYTES
        ),
        "chunk_count": len(rows),
        "chunks": rows,
    }
    receipt["payload_sha256"] = snapshot.canonical_sha256(receipt)
    receipt_bytes = (
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    remote[f"{remote_directory}/transport_receipt.json"] = receipt_bytes
    calls = []

    def fake_fetch(**kwargs):
        calls.append(kwargs["relative_path"])
        return remote[kwargs["relative_path"]]

    monkeypatch.setattr(collector, "_fetch_remote_bytes", fake_fetch)
    local = tmp_path / "collected"
    values = {
        "scheduler_url": "http://127.0.0.1:8002",
        "task_id": 96318,
        "remote_transport_relative_directory": remote_directory,
        "local_directory": local,
        "expected_lane": "full",
        "expected_source_task_id": 96307,
        "expected_source_slurm_job_id": "829952",
        "expected_transport_task_id": 96318,
        "expected_transport_node": "n116",
        "expected_artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "expected_artifact_size_bytes": len(raw),
        "expected_receipt_sha256": hashlib.sha256(
            receipt_bytes
        ).hexdigest(),
        "expected_snapshot_manifest_sha256": "1" * 64,
        "expected_snapshot_manifest_payload_sha256": "2" * 64,
        "max_workers": 2,
    }
    first = collector.collect_transport(**values)
    first_calls = len(calls)
    second = collector.collect_transport(**values)

    assert first["status"] == "collected"
    assert second["status"] == "existing_authenticated"
    assert (local / "full.aedt").read_bytes() == raw
    assert first_calls == len(rows) + 1
    assert len(calls) == first_calls
