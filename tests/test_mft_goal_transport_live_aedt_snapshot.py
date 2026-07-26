from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

from tools import mft_goal_snapshot_live_aedt as snapshot
from tools import mft_goal_transport_live_aedt_snapshot as transport


def _fixture(tmp_path: Path, *, lane: str = "full"):
    root = (
        tmp_path
        / "mft_goal_20260726"
        / "live_aedt_snapshots_v1"
        / f"{lane}-from-t96307-j829952"
    )
    root.mkdir(parents=True)
    raw = (b"AEDT-transport" * 40_000) + b"tail"
    artifact = root / f"{lane}.aedt"
    artifact.write_bytes(raw)
    manifest = {
        "schema": snapshot.SCHEMA,
        "diagnostic_only": True,
        "canonical": False,
        "production_truth_eligible": False,
        "solver_result_truth_included": False,
        "lane": lane,
        "source_node": "n116",
        "source_task_id": 96307,
        "source_slurm_job_id": "829952",
        "artifact_filename": artifact.name,
        "artifact_size_bytes": len(raw),
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "source_changed_during_snapshot": False,
    }
    manifest["payload_sha256"] = snapshot.canonical_sha256(manifest)
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return root, raw, manifest


def _invoke(tmp_path: Path, root: Path, raw: bytes, **overrides):
    values = {
        "snapshot_relative_directory": root.relative_to(tmp_path).as_posix(),
        "lane": "full",
        "source_task_id": 96307,
        "source_slurm_job_id": "829952",
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "artifact_size_bytes": len(raw),
        "environ": {
            "SLURM_SCHED_TASK_ID": "96316",
            "SLURM_JOB_ID": "829952",
            "SLURM_STEP_ID": "125",
        },
        "cwd": tmp_path,
        "hostname": "n116",
        "uid": root.stat().st_uid,
    }
    values.update(overrides)
    return transport.transport_snapshot(**values)


def test_transport_round_trips_and_is_idempotent(tmp_path: Path) -> None:
    root, raw, _manifest = _fixture(tmp_path)

    first = _invoke(tmp_path, root, raw)
    second = _invoke(
        tmp_path,
        root,
        raw,
        environ={
            "SLURM_SCHED_TASK_ID": "96317",
            "SLURM_JOB_ID": "829952",
        },
    )

    assert first["status"] == "published"
    assert second["status"] == "existing_authenticated"
    receipt = json.loads(
        (root / "transport" / "transport_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    rebuilt = b"".join(
        base64.b64decode(
            (root / "transport" / row["filename"]).read_bytes(),
            validate=True,
        )
        for row in receipt["chunks"]
    )
    assert rebuilt == raw
    assert receipt["chunk_count"] == 2
    assert receipt["artifact_sha256"] == hashlib.sha256(raw).hexdigest()


def test_transport_rejects_snapshot_artifact_drift(tmp_path: Path) -> None:
    root, raw, _manifest = _fixture(tmp_path)
    (root / "full.aedt").write_bytes(raw + b"changed")

    with pytest.raises(
        transport.TransportError, match="artifact identity drifted"
    ):
        _invoke(tmp_path, root, raw)


def test_transport_rejects_wrong_allocation(tmp_path: Path) -> None:
    root, raw, _manifest = _fixture(tmp_path)

    with pytest.raises(
        transport.TransportError, match="same-allocation contract"
    ):
        _invoke(
            tmp_path,
            root,
            raw,
            environ={
                "SLURM_SCHED_TASK_ID": "96316",
                "SLURM_JOB_ID": "838708",
            },
        )


def test_transport_rejects_wrong_source_node(tmp_path: Path) -> None:
    root, raw, _manifest = _fixture(tmp_path)

    with pytest.raises(
        transport.TransportError, match="manifest contract drifted"
    ):
        _invoke(tmp_path, root, raw, hostname="n111")
