from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools import tier1_final1000_multiseed_contract as contract
from tools import tier1_final1000_multiseed_monitor as monitor
from tools import tier1_final1000_multiseed_status as status_adapter


def _lane(
    task_id: int,
    *,
    protocol: str = contract.PROTOCOL_VERSION,
    state: str = "running",
    batch_length: int = 4,
    sealed: int = 2,
) -> dict:
    return {
        "task_id": task_id,
        "stage_id": "entry-1200-t125",
        "protocol_version": protocol,
        "state": state,
        "account_name": "fixture",
        "batch_length": batch_length,
        "current_seed": 101,
        "sealed_child_count": sealed,
    }


def _record(seed: int, *, parent: int | None = None, ordinal: int | None = None):
    value = {
        "bundle_id": "bundle-a",
        "seed": seed,
        "terminal_state": "completed",
    }
    if parent is not None:
        value["physical_parent_task_id"] = parent
    if ordinal is not None:
        value["batch_ordinal"] = ordinal
    return value


def _published_snapshot(tmp_path: Path) -> tuple[Path, dict, dict]:
    lanes = [
        _lane(1001),
        _lane(
            1002,
            protocol=status_adapter.SINGLE_SEED_PROTOCOL,
            batch_length=99,
            sealed=99,
        ),
    ]
    records = [
        _record(100, parent=1001, ordinal=0),
        _record(101, parent=1001, ordinal=1),
        # The receipt may exist in the crash window, but the running parent's
        # mutable cursor has not sealed ordinal 2.  It must not reach 8010.
        _record(102, parent=1001, ordinal=2),
        # Immutable history survives after its physical parent drains out of
        # the bounded hot inventory.
        _record(90, parent=999, ordinal=3),
        _record(200),
    ]
    snapshot = status_adapter.build_compact_snapshot(
        stage_id="entry-1200-t125",
        physical_lanes=lanes,
        seed_records=records,
        frontend_static={"cohort_id": "fixture"},
        shard_size=2,
    )
    status, _manifest, _shards, index = snapshot
    assert status_adapter.publish_compact_snapshot(tmp_path, *snapshot) > 0
    return tmp_path / "index.json", status, index


def test_running_parent_exposes_only_sealed_prefix_and_separates_counts(
    tmp_path: Path,
):
    index_path, status, index = _published_snapshot(tmp_path)

    assert status["physical_lane_count"] == 2
    assert status["logical_seed_count"] == 5
    assert status["logical_sealed_seed_count"] == 2
    assert status["logical_unsealed_seed_count"] == 3
    assert status["hidden_unsealed_seed_record_count"] == 1
    assert status["authenticated_terminal_seed_count"] == 4
    assert index["count_contract"] == {
        "scheduler_task_count_field": "physical_lane_count",
        "logical_seed_count_field": "logical_seed_count",
        "sealed_prefix_rule": "batch_ordinal < sealed_child_count",
    }

    page = monitor.adapt_condition_index(index_path, limit=256)
    assert page["schema_version"] == monitor.COHORT_STATUS_SCHEMA
    assert page["scheduler_task_count"] == 2
    assert page["physical_lane_count"] == 2
    assert page["logical_seed_count"] == 5
    assert page["logical_sealed_seed_count"] == 2
    assert page["logical_unsealed_seed_count"] == 3
    assert page["hidden_unsealed_seed_record_count"] == 1
    assert {record["seed"] for record in page["terminal_results"]} == {
        90,
        100,
        101,
        200,
    }
    assert len(contract.json_bytes(page)) < status_adapter.MAX_HOT_WIRE_BYTES
    assert page["virtual_scheduler_task_ids_created"] is False


def test_batch_record_without_a_provable_ordinal_fails_closed(tmp_path: Path):
    with pytest.raises(RuntimeError, match="ordinal"):
        status_adapter.build_compact_snapshot(
            stage_id="entry-1200-t125",
            physical_lanes=[_lane(1001)],
            seed_records=[_record(100, parent=1001)],
        )

    with pytest.raises(RuntimeError, match="cursor"):
        status_adapter.build_compact_snapshot(
            stage_id="entry-1200-t125",
            physical_lanes=[_lane(1001, batch_length=4, sealed=5)],
            seed_records=[],
        )


def test_read_only_index_adapter_refuses_count_and_reference_drift(tmp_path: Path):
    index_path, _status, index = _published_snapshot(tmp_path)
    loaded = monitor.load_compact_condition_index(index_path)
    assert loaded["index"] == index

    status_path = loaded["status_path"]
    tampered = copy.deepcopy(loaded["status"])
    tampered["logical_seed_count"] += 1
    unsigned = {key: value for key, value in tampered.items() if key != "status_sha256"}
    tampered["status_sha256"] = monitor.canonical_sha256(unsigned)
    status_path.chmod(0o644)
    status_path.write_bytes(contract.json_bytes(tampered))
    with pytest.raises(RuntimeError, match="file/reference seal"):
        monitor.adapt_condition_index(index_path)


def test_read_only_index_adapter_rejects_a_fully_resealed_manifest_gap(
    tmp_path: Path,
):
    snapshot = status_adapter.build_compact_snapshot(
        stage_id="entry-1200-t125",
        physical_lanes=[_lane(1001)],
        seed_records=[
            _record(100, parent=1001, ordinal=0),
            _record(101, parent=1001, ordinal=1),
        ],
        shard_size=2,
    )
    status, manifest, _shards, index = copy.deepcopy(snapshot)
    manifest["shards"][0]["record_offset"] = 1
    manifest_unsigned = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    manifest["manifest_sha256"] = monitor.canonical_sha256(manifest_unsigned)
    status["terminal_results_manifest_sha256"] = manifest["manifest_sha256"]
    status_unsigned = {
        key: value for key, value in status.items() if key != "status_sha256"
    }
    status["status_sha256"] = monitor.canonical_sha256(status_unsigned)
    status_payload = contract.json_bytes(status)
    manifest_payload = contract.json_bytes(manifest)
    (tmp_path / "status.json").write_bytes(status_payload)
    (tmp_path / "manifest.json").write_bytes(manifest_payload)
    index["status"].update(
        path="status.json",
        sha256=monitor._sha256(status_payload),
        size=len(status_payload),
    )
    index["seed_result_shards"].update(
        path="manifest.json",
        sha256=monitor._sha256(manifest_payload),
        size=len(manifest_payload),
    )
    index_unsigned = {
        key: value for key, value in index.items() if key != "index_sha256"
    }
    index["index_sha256"] = monitor.canonical_sha256(index_unsigned)
    index_path = tmp_path / "index.json"
    index_path.write_bytes(contract.json_bytes(index))

    with pytest.raises(RuntimeError, match="range/reference"):
        monitor.load_compact_condition_index(index_path)


def _receipt_inputs(tmp_path: Path, index_path: Path) -> dict:
    backend = tmp_path / "backend_reader.py"
    backend.write_text("CAPABILITY = 'compact-v2'\n", encoding="utf-8")
    evidence = tmp_path / "pytest-evidence.json"
    evidence.write_text(
        json.dumps({"passed": True, "tests": 7}, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "index_path": index_path,
        "code_files": {
            "tools/tier1_final1000_multiseed_monitor.py": Path(monitor.__file__),
            "tools/tier1_final1000_multiseed_status.py": Path(status_adapter.__file__),
        },
        "code_revision": "a" * 40,
        "backend_files": {"regression_260707/monitoring/readers.py": backend},
        "backend_revision": "b" * 40,
        "test_evidence_files": {"tests/pytest-evidence.json": evidence},
        "test_revision": "c" * 40,
    }


def test_capability_receipt_requires_exact_code_backend_index_and_test_seals(
    tmp_path: Path,
):
    index_path, _status, _index = _published_snapshot(tmp_path / "snapshot")
    inputs = _receipt_inputs(tmp_path, index_path)
    receipt = monitor.build_backend_capability_receipt(**inputs)
    assert monitor.validate_backend_capability_receipt(receipt) == receipt
    assert receipt["capability_contract"] == monitor.CAPABILITY_CONTRACT
    assert receipt["scheduler_mutation_performed"] is False
    assert receipt["remote_write_performed"] is False
    assert receipt["aedt_used"] is False
    assert receipt["fea_submission_performed"] is False

    receipt_path = tmp_path / "backend-capability.json"
    receipt_path.write_bytes(contract.json_bytes(receipt))
    assert (
        monitor.require_backend_capability(receipt_path, **inputs)["receipt_sha256"]
        == receipt["receipt_sha256"]
    )

    with pytest.raises(RuntimeError, match="receipt is required"):
        monitor.require_backend_capability(tmp_path / "missing.json", **inputs)

    evidence_path = next(iter(inputs["test_evidence_files"].values()))
    evidence_path.write_text('{"passed": false}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="test_seals drifted"):
        monitor.require_backend_capability(receipt_path, **inputs)


def test_capability_receipt_and_cli_refuse_index_or_backend_drift(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    index_path, _status, _index = _published_snapshot(tmp_path / "snapshot")
    inputs = _receipt_inputs(tmp_path, index_path)
    receipt = monitor.build_backend_capability_receipt(**inputs)
    receipt_path = tmp_path / "backend-capability.json"
    receipt_path.write_bytes(contract.json_bytes(receipt))

    backend_path = next(iter(inputs["backend_files"].values()))
    backend_path.write_text("CAPABILITY = 'drifted'\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="backend_identity drifted"):
        monitor.require_backend_capability(receipt_path, **inputs)
    backend_path.write_text("CAPABILITY = 'compact-v2'\n", encoding="utf-8")

    # Publishing a new coherent pointer is legitimate, but it invalidates an
    # exact cutover receipt until the backend gate is re-run and resealed.
    status, manifest, shards, index = status_adapter.build_compact_snapshot(
        stage_id="entry-1200-t125",
        physical_lanes=[_lane(1001, sealed=3)],
        seed_records=[
            _record(100, parent=1001, ordinal=0),
            _record(101, parent=1001, ordinal=1),
            _record(102, parent=1001, ordinal=2),
        ],
        shard_size=2,
    )
    status_adapter.publish_compact_snapshot(
        index_path.parent, status, manifest, shards, index
    )
    with pytest.raises(RuntimeError, match="condition_index_identity drifted"):
        monitor.require_backend_capability(receipt_path, **inputs)

    assert monitor.main(["adapt", "--index", str(index_path), "--limit", "1"]) == 0
    cli_page = json.loads(capsys.readouterr().out)
    assert cli_page["scheduler_task_count"] == 1
    assert cli_page["logical_seed_count"] == 4
    assert cli_page["terminal_results_returned_count"] == 1


def test_capability_cli_seals_and_revalidates_exact_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    index_path, _status, _index = _published_snapshot(tmp_path / "snapshot")
    inputs = _receipt_inputs(tmp_path, index_path)
    backend_name, backend_path = next(iter(inputs["backend_files"].items()))
    evidence_name, evidence_path = next(iter(inputs["test_evidence_files"].items()))
    receipt_path = tmp_path / "cli-capability.json"
    identity_args = [
        "--index",
        str(index_path),
        "--code-root",
        str(Path(monitor.__file__).parents[1]),
        "--code-revision",
        inputs["code_revision"],
        "--backend-file",
        f"{backend_name}={backend_path}",
        "--backend-revision",
        inputs["backend_revision"],
        "--test-evidence",
        f"{evidence_name}={evidence_path}",
        "--test-revision",
        inputs["test_revision"],
    ]
    assert (
        monitor.main(["seal-capability", *identity_args, "--output", str(receipt_path)])
        == 0
    )
    sealed = json.loads(capsys.readouterr().out)
    assert sealed == json.loads(receipt_path.read_text(encoding="utf-8"))
    assert (
        monitor.main(
            ["validate-capability", *identity_args, "--receipt", str(receipt_path)]
        )
        == 0
    )
    validation = json.loads(capsys.readouterr().out)
    assert validation == {
        "valid": True,
        "receipt_sha256": sealed["receipt_sha256"],
        "condition_index_identity_sha256": sealed["condition_index_identity"][
            "identity_sha256"
        ],
    }
