from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from tools import mft_campaign_atomic_claim as atomic_claim
from tools import mft_goal_full_fastlane_watcher as fastlane
from tools import mft_goal_terminal_success_watcher as terminal


def _gpfs(account: str = "dhj02") -> dict[str, Any]:
    return {
        "account_name": account,
        "filesystem_type": "gpfs",
        "fileset_name": "root",
        "user_quota_scope": "filesystem",
        "quota_type": "USR",
        "block_used_gb": 40.0,
        "block_in_doubt_gb": 0.0,
        "block_limit_gb": 110.0,
    }


def _task(
    task_id: int,
    *,
    account: str = "dhj02",
    timeout: int = 28800,
    status: str = "running",
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "id": task_id,
        "name": f"mft-goal-diag-standard-{task_id}",
        "status": status,
        "state": status,
        "project": "MFT_1MW_2026v1",
        "dedupe_key": f"mft-test:{task_id}",
        "account_name": account,
        "cpus": 8,
        "memory_mb": 32768,
        "timeout_seconds": timeout,
        "aedt_backend": "standalone",
        "created_at": "2026-07-25 14:00:00",
        "started_at": "2026-07-25 14:01:00",
        "finished_at": None,
    }


def _capacity(account: str = "dhj02") -> dict[str, Any]:
    return {
        "fit_slots": 1,
        "ready_fit_slots": 1,
        "pending_fit_slots": 0,
        "inflight_fit_slots": 1,
        "memory_pressure_state": "ok",
        "allocations": [
            {
                "allocation_id": 1,
                "account_name": account,
                "slurm_job_id": "123",
                "state": "warm",
                "partition": "cpu",
                "node_name": "n116",
                "free_cpus": 16,
                "free_memory_mb": 98304,
                "free_gpus": 0,
                "gpu_model": "",
                "fit_slots": 1,
                "memory_pressure_state": "ok",
                "node_memory_free_percent": 60.0,
            }
        ],
    }


def _mock_active_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tasks: list[dict[str, Any]],
    *,
    bound_gib: list[int],
) -> list[Path]:
    values: dict[Path, dict[str, Any]] = {}
    for task, gib in zip(tasks, bound_gib, strict=True):
        path = (tmp_path / f"bound-{task['task_id']}.json").resolve()
        path.write_text("{}\n", encoding="utf-8")
        values[path] = {
            "payload_sha256": f"{task['task_id']:064x}",
            "task_id": task["task_id"],
            "task_name": task["name"],
            "dedupe_key": task["dedupe_key"],
            "standard_storage_bound_bytes": gib * 1024**3,
            "standard_storage_bound_gib": float(gib),
        }

    def load(path: Path, *, task: dict[str, Any] | None = None):
        value = values[Path(path).resolve()]
        if task is not None and (
            value["task_id"] != task["task_id"]
            or value["task_name"] != task["name"]
            or value["dedupe_key"] != task["dedupe_key"]
        ):
            raise fastlane.HandoffContractError("bound mismatch")
        return value

    monkeypatch.setattr(fastlane, "_load_standard_storage_bound", load)
    return list(values)


def test_full_profile_keeps_reviewed_physics_and_resources() -> None:
    profile, _record = fastlane.promotion._full_profile()
    assert profile["mem_mb"] == 98304
    assert profile["cpus"] == 16
    assert profile["timeout_seconds"] == 43200
    assert profile["param_overrides"]["full_model"] == 1
    assert profile["param_overrides"]["thermal_symmetry"] == "full"
    assert profile["param_overrides"]["fan_velocity"] == 1.5
    assert profile["param_overrides"]["fan_config"] == "dual"
    assert profile["fixed_boundary_contract"] == {
        "thermal_pad_conductivity_W_mK": 0.2,
        "core_plate_pad_t_mm": 2.0,
        "wcp_pad_t_mm": 2.0,
        "fan_velocity_m_s": 1.5,
        "fan_config": "dual",
        "core_plate_on": 1,
        "wcp_on": 1,
    }


def test_terminal_authority_requires_immutable_nds_receipt_and_rank0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "truth_pareto_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    receipt = tmp_path / "truth.receipt.json"
    receipt_value = terminal._sealed(
        {
            "schema_version": terminal.NDS_RECEIPT_SCHEMA,
            "campaign_id": "mft-goal-20260726",
            "authority_key": "a" * 16,
            "logical_authority_task_ids": [96212],
            "input_collections": [],
            "truth_manifest": fastlane.production._file_record(manifest),
            "global_nondominated_sort_performed": True,
            "scheduler_mutation_performed": False,
            "created_at_utc": "2026-07-25T14:00:00+00:00",
        }
    )
    fastlane.production._write_immutable_json(receipt, receipt_value)
    rank0 = {
        "candidate_physics_sha256": "1" * 64,
        "truth_non_dominated_rank": 0,
    }
    monkeypatch.setattr(
        fastlane.promotion,
        "_load_truth_manifest",
        lambda _path: (
            {"payload_sha256": "2" * 64, "rank0_count": 1},
            [rank0],
            {},
        ),
    )
    authenticated = fastlane._authenticate_terminal_authority(
        manifest_path=manifest, receipt_path=receipt
    )
    assert authenticated["rank0_rows"] == [rank0]


def test_terminal_authority_rejects_mutating_promotion_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "truth_pareto_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    receipt = tmp_path / "truth.receipt.json"
    fastlane.production._write_immutable_json(
        receipt,
        terminal._sealed(
            {
                "schema_version": terminal.NDS_RECEIPT_SCHEMA,
                "campaign_id": "mft-goal-20260726",
                "truth_manifest": fastlane.production._file_record(manifest),
                "global_nondominated_sort_performed": True,
                "scheduler_mutation_performed": True,
            }
        ),
    )
    monkeypatch.setattr(
        fastlane.promotion,
        "_load_truth_manifest",
        lambda _path: (
            {"payload_sha256": "2" * 64, "rank0_count": 1},
            [
                {
                    "candidate_physics_sha256": "1" * 64,
                    "truth_non_dominated_rank": 0,
                }
            ],
            {},
        ),
    )
    with pytest.raises(
        fastlane.HandoffContractError,
        match="rank-0 promotion authority drifted",
    ):
        fastlane._authenticate_terminal_authority(
            manifest_path=manifest, receipt_path=receipt
        )


def test_gpfs_audit_uses_per_task_authenticated_bounds_without_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = [_task(1), _task(2)]
    authorities = _mock_active_bounds(
        tmp_path, monkeypatch, tasks, bound_gib=[19, 23]
    )
    audit = fastlane.build_gpfs_audit(
        observed_at_utc="2026-07-25T23:00:00+09:00",
        account_observations=[_gpfs()],
        rows=tasks,
        active_task_storage_authority_paths=authorities,
    )
    account = audit["account_observations"][0]
    assert account["active_bounded_task_count"] == 2
    assert account["active_storage_bound_gib"] == 42.0
    assert account["free_after_active_gb"] == 28.0
    assert account["minimum_free_floor_gb"] == 10.0
    assert audit["candidate_storage_bound_included"] is False
    assert audit["generic_per_task_storage_reservation_used"] is False
    assert audit["retained_bytes_used"] is False
    assert audit["active_unbounded_storage_conflict_count"] == 0
    assert audit["arithmetic_passed"] is True


@pytest.mark.parametrize(
    ("account", "timeout"),
    [("", 28800), ("unknown", 28800), ("dhj02", 0)],
)
def test_gpfs_audit_blocks_unbounded_or_unaudited_active_work(
    account: str, timeout: int
) -> None:
    audit = fastlane.build_gpfs_audit(
        observed_at_utc="2026-07-25T23:00:00+09:00",
        account_observations=[_gpfs()],
        rows=[_task(1, account=account, timeout=timeout)],
    )
    assert audit["active_unbounded_storage_conflict_count"] == 1
    assert audit["arithmetic_passed"] is False


def test_gpfs_audit_blocks_active_task_without_storage_authority() -> None:
    audit = fastlane.build_gpfs_audit(
        observed_at_utc="2026-07-25T23:00:00+09:00",
        account_observations=[_gpfs()],
        rows=[_task(1)],
    )
    assert audit["active_unbounded_storage_conflict_count"] == 1
    assert (
        audit["active_unbounded_storage_conflicts"][0]["reason"]
        == "active_task_storage_authority_missing"
    )
    assert audit["arithmetic_passed"] is False


def test_gpfs_audit_isolates_missing_bound_to_its_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocked = _task(96294, account="dhj02")
    bounded = _task(96298, account="r1jae262")
    authorities = _mock_active_bounds(
        tmp_path, monkeypatch, [bounded], bound_gib=[29]
    )
    observed = datetime(2026, 7, 25, 15, 0, tzinfo=timezone.utc)
    audit = fastlane.build_gpfs_audit(
        observed_at_utc=observed.isoformat(),
        account_observations=[
            _gpfs("dhj02"),
            {
                **_gpfs("r1jae262"),
                "block_used_gb": 100.0,
                "block_limit_gb": 200.0,
            },
        ],
        rows=[blocked, bounded],
        active_task_storage_authority_paths=authorities,
    )
    accounts = {
        item["account_name"]: item
        for item in audit["account_observations"]
    }
    assert accounts["dhj02"]["active_only_arithmetic_passed"] is False
    assert (
        accounts["dhj02"]["active_unbounded_storage_conflict_count"]
        == 1
    )
    assert accounts["r1jae262"]["active_only_arithmetic_passed"] is True
    assert (
        accounts["r1jae262"]["active_unbounded_storage_conflict_count"]
        == 0
    )
    assert audit["admissible_account_names"] == ["r1jae262"]
    assert audit["blocked_account_names"] == ["dhj02"]
    assert audit["active_unbounded_storage_conflict_count"] == 1
    assert audit["arithmetic_passed"] is True
    path = tmp_path / "account-isolated-audit.json"
    fastlane.production._write_immutable_json(path, audit)
    assert fastlane._load_gpfs_audit(
        path, require_fresh=True, now=observed + timedelta(seconds=30)
    ) == audit


def test_gpfs_audit_is_fresh_and_fails_closed_when_stale(
    tmp_path: Path,
) -> None:
    observed = datetime(2026, 7, 25, 14, 0, tzinfo=timezone.utc)
    audit = fastlane.build_gpfs_audit(
        observed_at_utc=observed.isoformat(),
        account_observations=[_gpfs()],
        rows=[],
    )
    path = tmp_path / "audit.json"
    fastlane.production._write_immutable_json(path, audit)
    assert fastlane._load_gpfs_audit(
        path, require_fresh=True, now=observed + timedelta(seconds=90)
    ) == audit
    with pytest.raises(fastlane.HandoffContractError, match="GPFS audit is stale"):
        fastlane._load_gpfs_audit(
            path, require_fresh=True, now=observed + timedelta(seconds=121)
        )


def test_source_grid_provenance_is_reauthenticated_from_timeout_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "diagnostic_timeout12h_retry_plan.json"
    source.write_text(
        fastlane.json.dumps(
            {"schema_version": fastlane.timeout12h.PLAN_SCHEMA}
        ),
        encoding="utf-8",
    )
    grid_bytes = 27 * 1024**3
    stream = {"fresh_grid_output_bytes": grid_bytes}
    storage = {"prospective_grid_gb": 27.0}
    plan = {
        "payload_sha256": "1" * 64,
        "candidate_physics_sha256": "2" * 64,
        "retry_of_timeout12h": {
            "logical_authority_task_id": 96218,
            "stream_evidence": stream,
            "stream_evidence_sha256": fastlane.canonical_sha256(stream),
            "storage_audit": storage,
            "storage_audit_sha256": fastlane.canonical_sha256(storage),
        },
        "stage": {
            "name": "standard",
            "task_name": "mft-goal-diag-timeout12h-r1",
            "full_model": 0,
            "thermal_symmetry": "eighth",
            "retained_aedt_bundle": {"dedupe_key": "mft:test"},
        },
    }
    monkeypatch.setattr(
        fastlane.timeout12h,
        "_load_plan",
        lambda _path: (plan, {}, {}, {}),
    )
    evidence = fastlane._authenticated_standard_grid_evidence(source)
    assert evidence["fresh_grid_output_bytes"] == grid_bytes
    assert evidence["source_stage_thermal_symmetry"] == "eighth"
    assert evidence["candidate_physics_sha256"] == "2" * 64


@pytest.mark.parametrize(
    ("execution_task_id", "schema", "candidate"),
    [
        (96294, fastlane.diagnostic.PLAN_SCHEMA, "692c1a03e5fd"),
        (96295, fastlane.diagnostic.PLAN_SCHEMA, "c3195a79f5b2"),
        (96296, fastlane.diagnostic.PLAN_SCHEMA, "628c9fcfec1e"),
        (
            96298,
            fastlane.diagnostic._dependency_retry_module().PLAN_SCHEMA,
            "ab33ef0ba3ef",
        ),
    ],
)
def test_current_successor_without_terminal_stream_authority_is_unsupported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution_task_id: int,
    schema: str,
    candidate: str,
) -> None:
    source = tmp_path / f"task-{execution_task_id}-plan.json"
    source.write_text(
        fastlane.json.dumps({"schema_version": schema}),
        encoding="utf-8",
    )
    plan = {
        "schema_version": schema,
        "payload_sha256": "1" * 64,
        "candidate_physics_sha256": candidate.ljust(64, "0"),
        "stage": {
            "name": "standard",
            "full_model": 0,
            "thermal_symmetry": "eighth",
            "retained_aedt_bundle": {"dedupe_key": "mft:test"},
        },
    }
    monkeypatch.setattr(
        fastlane.diagnostic,
        "_load_collectible_plan",
        lambda _path: (plan, {}, {}),
    )
    with pytest.raises(
        fastlane.HandoffContractError,
        match="no terminal-success stream authority",
    ):
        fastlane._authenticated_standard_grid_evidence(source)


def test_successful_standard_stream_grid_is_actual_premesh_not_retained() -> None:
    task_id = 96294
    result = {
        "solver_core_scheduler_task_id_readback": str(task_id),
        "solver_num_cores_requested": 8,
        "solver_num_cores_effective": 8,
        "full_model": 0,
        "thermal_symmetry": "eighth",
    }
    artifacts = [
        {
            "directory": "/enroot/mft_campaign-test/results",
            "name": "Mesh0.sd",
            "grid_output_size": 11 * 1024**3,
            "grid_output_sha256_sample": "a" * 64,
        },
        {
            "directory": "/enroot/mft_campaign-test/results",
            "name": "Mesh1.sd",
            "grid_output_size": 8 * 1024**3,
            "grid_output_sha256_sample": "b" * 64,
        },
    ]
    preflight = {
        "schema": "thermal-mesh-preflight-v2",
        "passed": True,
        "status": "passed_standalone_native_premesh",
        "mesh_mapping_coverage_passed": True,
        "mesh_artifact_readback_passed": True,
        "native_operation_readback_passed": True,
        "standalone_idle_barrier_passed": True,
        "postflight_error": "",
        "native_errors": [],
        "fresh_mesh_artifact_count": 2,
        "fresh_mesh_artifacts": artifacts,
    }
    stdout = (
        "RESULT_JSON "
        + fastlane.json.dumps(result, sort_keys=True)
        + "\n"
    ).encode()
    stderr = (
        "[thermal] native mesh preflight: "
        + fastlane.json.dumps(preflight, sort_keys=True)
        + "\n"
    ).encode()
    collection = {
        "task_id": task_id,
        "scheduler_status": "completed",
        "scheduler_task_execution": {
            "state": "succeeded",
            "exit_code": 0,
        },
        "result": result,
        "result_sha256": fastlane.canonical_sha256(result),
    }
    evidence = fastlane._successful_standard_grid_stream_evidence(
        stdout=stdout, stderr=stderr, collection=collection
    )
    assert evidence["fresh_grid_output_bytes"] == 19 * 1024**3
    assert evidence["solver_transient_root"] == "/enroot"
    assert "retained" not in evidence


def test_dependency_predecessor_grid_authority_uses_stable_get_streams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "dependency-plan.json"
    source.write_text("{}\n", encoding="utf-8")
    dependency = fastlane.diagnostic._dependency_retry_module()
    predecessor = dependency._normalized_task_snapshot(
        {
            **_task(96279, account="jji0930", status="failed"),
            "exit_code": None,
            "failure_message": "same_node_as task 96262 is failed",
            "finished_at": "2026-07-25 07:08:46",
            "same_node_as_task_id": 96262,
        }
    )
    candidate = "a" * 64
    immediate = {
        "task_id": 96279,
        "payload_sha256": "1" * 64,
        "plan_payload_sha256": "2" * 64,
        "candidate_physics_sha256": candidate,
    }
    plan = {
        "payload_sha256": "3" * 64,
        "candidate_physics_sha256": candidate,
        "stage": {
            "name": "standard",
            "scheduler_url": "http://127.0.0.1:8002",
            "full_model": 0,
            "thermal_symmetry": "eighth",
        },
        "retry_of_dependency_failure": {
            "retry_of_task_id": 96279,
            "original_plan": {"path": "parent.json"},
            "original_plan_payload_sha256": "2" * 64,
            "original_submission": {"path": "submission.json"},
            "original_submission_payload_sha256": "1" * 64,
            "dependency_failure_evidence": {"task": predecessor},
        },
    }
    artifacts = [
        {
            "directory": "/enroot/mft_campaign-test/results",
            "name": f"Mesh{index}.sd",
            "grid_output_size": size,
            "grid_output_sha256_sample": f"{index + 1:x}" * 64,
        }
        for index, size in enumerate((11 * 1024**3, 17 * 1024**3))
    ]
    preflight = {
        "schema": "thermal-mesh-preflight-v2",
        "passed": True,
        "status": "passed_standalone_native_premesh",
        "mesh_mapping_coverage_passed": True,
        "mesh_artifact_readback_passed": True,
        "native_operation_readback_passed": True,
        "standalone_idle_barrier_passed": True,
        "postflight_error": "",
        "native_errors": [],
        "fresh_mesh_artifact_count": len(artifacts),
        "fresh_mesh_artifacts": artifacts,
    }
    stdout = b"dependency predecessor ran without final result\n"
    stderr = (
        "[thermal] native mesh preflight: "
        + fastlane.json.dumps(preflight, sort_keys=True)
        + "\n"
    ).encode()
    monkeypatch.setattr(
        fastlane,
        "_dependency_predecessor_context",
        lambda _path: (plan, immediate, predecessor),
    )
    task_reads = 0
    stream_reads = {"stdout": 0, "stderr": 0}

    def task_reader(**_kwargs: Any) -> dict[str, Any]:
        nonlocal task_reads
        task_reads += 1
        return dict(predecessor)

    def stream_reader(*, stream: str, **_kwargs: Any) -> bytes:
        stream_reads[stream] += 1
        return stdout if stream == "stdout" else stderr

    output = tmp_path / "predecessor-grid.json"
    fastlane.create_dependency_predecessor_grid_authority(
        source_plan_path=source,
        output=output,
        task_reader=task_reader,
        stream_reader=stream_reader,
    )
    authority = fastlane._load_dependency_predecessor_grid_authority(
        output
    )
    assert task_reads == 2
    assert stream_reads == {"stdout": 2, "stderr": 2}
    assert authority["predecessor_task_id"] == 96279
    assert authority["fresh_grid_output_bytes"] == 28 * 1024**3
    assert authority["scheduler_methods_used"] == ["GET"]
    assert authority["scheduler_stream_reads_per_stream"] == 2
    assert authority["scheduler_mutation_performed"] is False
    assert authority["retained_bytes_used"] is False
    assert authority["generic_reservation_used"] is False


def test_full_storage_bound_is_eight_sector_source_grid_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection_path = tmp_path / "selection.json"
    full_plan_path = tmp_path / "full_plan.json"
    collection_path = tmp_path / "collection.json"
    source_plan_path = tmp_path / "source_plan.json"
    for path in (
        selection_path,
        full_plan_path,
        collection_path,
        source_plan_path,
    ):
        path.write_text("{}\n", encoding="utf-8")
    candidate = "3" * 64
    selection = {
        "payload_sha256": "4" * 64,
        "candidate_physics_sha256": candidate,
        "full_plan": fastlane.production._file_record(full_plan_path),
    }
    collection_record = fastlane.production._file_record(collection_path)
    source_record = fastlane.production._file_record(source_plan_path)
    full_plan = {
        "payload_sha256": "5" * 64,
        "candidate_physics_sha256": candidate,
        "standard_collection": collection_record,
        "standard_collection_payload_sha256": "6" * 64,
        "standard_task_id": 96300,
    }
    grid_bytes = 27 * 1024**3
    evidence = {
        "source_standard_plan": source_record,
        "source_standard_plan_payload_sha256": "7" * 64,
        "candidate_physics_sha256": candidate,
        "logical_authority_task_id": 96218,
        "task_name": "standard",
        "dedupe_key": "mft:standard",
        "fresh_grid_output_bytes": grid_bytes,
        "fresh_grid_output_gib": 27.0,
        "stream_evidence_sha256": "8" * 64,
        "storage_audit_sha256": "9" * 64,
        "source_stage_full_model": 0,
        "source_stage_thermal_symmetry": "eighth",
    }
    monkeypatch.setattr(
        fastlane, "_load_selection", lambda _path: selection
    )
    monkeypatch.setattr(
        fastlane, "_selection_plan_path", lambda _selection: full_plan_path
    )
    monkeypatch.setattr(
        fastlane.promotion,
        "_load_full_plan",
        lambda _path: (full_plan, {}, {}, {}, {}),
    )
    monkeypatch.setattr(
        fastlane.diagnostic,
        "authenticate_collection",
        lambda _path: {
            "collection": {
                "task_id": 96300,
                "payload_sha256": "6" * 64,
                "plan": source_record,
            },
            "plan": {
                "payload_sha256": "7" * 64,
                "candidate_physics_sha256": candidate,
            },
        },
    )
    monkeypatch.setattr(
        fastlane,
        "_authenticated_standard_grid_evidence",
        lambda *_args, **_kwargs: evidence,
    )
    payload = fastlane._full_storage_authority_payload(
        selection_path=selection_path, selection=selection
    )
    assert payload["full_symmetry_expansion_factor"] == 8
    assert payload["full_prospective_storage_bound_bytes"] == 216 * 1024**3
    assert payload["full_prospective_storage_bound_gib"] == 216.0
    assert payload["minimum_post_reservation_free_floor_gib"] == 10.0
    assert payload["solver_transient_root"] == "/enroot"
    assert payload["solver_transient_route_minimum_gib"] == 200.0
    assert payload["static_transient_route_bound_passed"] is False
    assert payload["retained_bytes_used"] is False
    assert payload["generic_reservation_used"] is False


def test_full_storage_authority_missing_or_mismatched_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = {"payload_sha256": "1" * 64}
    with pytest.raises(
        fastlane.HandoffContractError,
        match="exactly one matching Full prospective-storage authority",
    ):
        fastlane._latest_full_storage_authority(
            tmp_path, selection=selection
        )
    selection_path = tmp_path / "selection.json"
    selection_path.write_text("{}\n", encoding="utf-8")
    expected = {
        "schema_version": fastlane.FULL_STORAGE_AUTHORITY_SCHEMA,
        "selection": fastlane.production._file_record(selection_path),
        "selection_payload_sha256": "1" * 64,
        "candidate_physics_sha256": "2" * 64,
    }
    monkeypatch.setattr(
        fastlane,
        "_full_storage_authority_payload",
        lambda **_kwargs: expected,
    )
    bad = fastlane._sealed(
        {
            **expected,
            "candidate_physics_sha256": "3" * 64,
            "created_at_utc": "2026-07-25T14:00:00+00:00",
        }
    )
    bad_path = tmp_path / "bad.json"
    fastlane.production._write_immutable_json(bad_path, bad)
    with pytest.raises(
        fastlane.HandoffContractError,
        match="prospective-storage authority drifted",
    ):
        fastlane._load_full_storage_authority(
            bad_path, selection=selection
        )


def test_full_storage_gate_rejects_insufficient_quota() -> None:
    audit = {
        "account_observations": [
            {
                "account_name": "dhj02",
                "free_after_active_gb": 100.0,
                "active_only_arithmetic_passed": True,
            }
        ]
    }
    source_bytes = 12 * 1024**3
    full = {
        "fresh_grid_output_bytes": source_bytes,
        "full_symmetry_expansion_factor": 8,
        "standard_symmetry_denominator": 8,
        "full_prospective_storage_bound_bytes": source_bytes * 8,
        "full_prospective_storage_bound_gib": 96.0,
        "minimum_post_reservation_free_floor_gib": 10.0,
        "solver_transient_root": "/enroot",
        "solver_transient_route_minimum_gib": 200.0,
        "full_bound_plus_working_floor_gib": 106.0,
        "static_transient_route_bound_passed": True,
        "retained_bytes_used": False,
        "generic_reservation_used": False,
    }
    evaluated, safe = fastlane._storage_admission_accounts(
        audit=audit, full_storage=full
    )
    assert evaluated[0]["free_after_active_and_full_gib"] == 4.0
    assert evaluated[0]["full_arithmetic_passed"] is False
    assert safe == []


def test_full_storage_gate_rejects_bound_above_static_enroot_route() -> None:
    audit = {
        "account_observations": [
            {
                "account_name": "large",
                "free_after_active_gb": 500.0,
                "active_only_arithmetic_passed": True,
            }
        ]
    }
    source_bytes = 27 * 1024**3
    full = {
        "fresh_grid_output_bytes": source_bytes,
        "full_symmetry_expansion_factor": 8,
        "standard_symmetry_denominator": 8,
        "full_prospective_storage_bound_bytes": source_bytes * 8,
        "full_prospective_storage_bound_gib": 216.0,
        "minimum_post_reservation_free_floor_gib": 10.0,
        "solver_transient_root": "/enroot",
        "solver_transient_route_minimum_gib": 200.0,
        "full_bound_plus_working_floor_gib": 226.0,
        "static_transient_route_bound_passed": False,
        "retained_bytes_used": False,
        "generic_reservation_used": False,
    }
    evaluated, safe = fastlane._storage_admission_accounts(
        audit=audit, full_storage=full
    )
    assert evaluated[0]["free_after_active_and_full_gib"] == 284.0
    assert evaluated[0]["full_arithmetic_passed"] is False
    assert safe == []


def test_capacity_must_be_ready_on_exact_audited_account() -> None:
    snapshot = _capacity()
    assert (
        fastlane._require_account_capacity(
            scheduler_url="http://127.0.0.1:8002",
            account_name="dhj02",
            live_reader=lambda **_kwargs: snapshot,
        )
        == snapshot
    )
    wrong = _capacity(account="other")
    with pytest.raises(
        fastlane.HandoffContractError,
        match="no ready 16-core/98304MB capacity",
    ):
        fastlane._require_account_capacity(
            scheduler_url="http://127.0.0.1:8002",
            account_name="dhj02",
            live_reader=lambda **_kwargs: wrong,
        )


def test_capacity_probe_uses_the_actual_full_submission_profile() -> None:
    endpoint = fastlane._capacity_endpoint("dhj02")
    query = fastlane.urllib.parse.parse_qs(
        fastlane.urllib.parse.urlparse(endpoint).query
    )
    assert query["scheduling_profile"] == [
        fastlane.scheduler_client.FEA_SCHEDULING_PROFILE
    ]
    assert query["scheduling_profile"] == ["fea_bursty"]


def test_capacity_zero_fails_closed() -> None:
    with pytest.raises(
        fastlane.HandoffContractError,
        match="no ready 16-core/98304MB capacity",
    ):
        fastlane._require_account_capacity(
            scheduler_url="http://127.0.0.1:8002",
            account_name="dhj02",
            live_reader=lambda **_kwargs: {
                "fit_slots": 0,
                "ready_fit_slots": 0,
                "memory_pressure_state": "ok",
                "allocations": [],
            },
        )


def test_priority_fence_binds_manifest_rank0_and_selection(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    promotion_receipt = tmp_path / "promotion.json"
    manifest.write_text("{}\n", encoding="utf-8")
    promotion_receipt.write_text("{}\n", encoding="utf-8")
    selection = fastlane._sealed(
        {
            "schema_version": fastlane.SELECTION_SCHEMA,
            "truth_manifest": fastlane.production._file_record(manifest),
            "truth_manifest_payload_sha256": "1" * 64,
            "promotion_receipt": fastlane.production._file_record(
                promotion_receipt
            ),
            "promotion_receipt_payload_sha256": "2" * 64,
            "candidate_physics_sha256": "3" * 64,
            "standard_task_id": 96212,
        }
    )
    selection_path = tmp_path / "selection.json"
    fastlane.production._write_immutable_json(selection_path, selection)
    plan = {
        "output_root": str(tmp_path),
        "priority_fence_path": str(tmp_path / "shared" / "fence.json"),
    }
    fence = fastlane._ensure_priority_fence(
        plan=plan, selection=selection
    )
    assert fence["selection"] == fastlane.production._file_record(
        selection_path
    )
    assert fence["selection_payload_sha256"] == selection["payload_sha256"]
    assert fence["truth_non_dominated_rank"] == 0
    assert fence["safe_refill_next_post_blocked"] is True
    assert fence["safe_refill_logical_authority_task_ids"] == [
        96223,
        96224,
        96230,
    ]


def test_active_safe_refill_yields_to_priority_full() -> None:
    active = [
        _task(1),
        {
            **_task(2),
            "name": "mft-goal-diag-timeout-r4-l96230-b7c30cb70b95",
        },
    ]
    assert [
        row["task_id"]
        for row in fastlane._active_safe_refill_tasks(active)
    ] == [2]


def test_latest_license_snapshot_requires_fresh_full_headroom(
    tmp_path: Path,
) -> None:
    checked = datetime(2026, 7, 25, 14, 0, tzinfo=timezone.utc)
    directory = tmp_path / "capture"
    directory.mkdir()
    snapshot = {
        "schema": "mft-aedt-license-headroom-snapshot-v1",
        "server_up": True,
        "server": "1055@172.16.10.81",
        "checked_at": checked.isoformat(),
        "features": {
            "anshpc": {"total": 32, "used": 0},
            "elec_solve_maxwell": {"total": 2, "used": 0},
            "electronics_desktop": {"total": 2, "used": 0},
            "electronics3d_gui": {"total": 2, "used": 0},
        },
    }
    path = directory / "snapshot.json"
    path.write_text(
        fastlane.json.dumps(snapshot, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    assert fastlane._latest_license_snapshot(
        tmp_path, now=checked + timedelta(seconds=300)
    ) == path.resolve()
    with pytest.raises(
        fastlane.HandoffContractError,
        match="no fresh passing 16-core license snapshot",
    ):
        fastlane._latest_license_snapshot(
            tmp_path, now=checked + timedelta(seconds=601)
        )


def test_without_activation_cycle_cannot_reach_submit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = {
        "payload_sha256": "1" * 64,
        "output_root": str(tmp_path),
    }
    selection = {
        "payload_sha256": "2" * 64,
        "candidate_physics_sha256": "3" * 64,
    }
    gates = {"selected_account_name": "dhj02"}
    monkeypatch.setattr(fastlane, "_load_watch_plan", lambda _path: plan)
    monkeypatch.setattr(fastlane, "_prepare_selection", lambda _plan: selection)
    monkeypatch.setattr(
        fastlane,
        "_ensure_full_storage_authority",
        lambda *_args, **_kwargs: tmp_path / "full-storage.json",
    )
    monkeypatch.setattr(fastlane, "_fresh_gates", lambda *_args, **_kwargs: gates)

    def forbidden(**_kwargs):
        raise AssertionError("submission path was reached without activation")

    monkeypatch.setattr(fastlane, "_execute_submission", forbidden)
    state = fastlane.process_cycle(watch_plan_path=tmp_path / "watch_plan.json")
    assert state["status"] == "ready_live_post_disabled_pending_root_review"
    assert state["scheduler_submission_performed"] is False
    assert state["live_post_default_enabled"] is False


def test_submission_adapter_posts_once_and_injects_locked_guard(
    tmp_path: Path,
) -> None:
    calls = []
    guards = []

    class Delegate:
        @staticmethod
        def submit_verification(*args, **kwargs):
            calls.append((args, kwargs))
            kwargs["pre_submit_guard"]()
            return 96310

    adapter = fastlane._SubmissionAdapter(
        delegate=Delegate,
        account_name="dhj02",
        guard=lambda: guards.append("guarded"),
        post_result_path=tmp_path / "post_result.json",
    )
    assert adapter.submit_verification("name", "work", {}) == 96310
    assert calls[0][1]["account_name"] == "dhj02"
    assert guards == ["guarded"]
    assert adapter.scheduler_post_count == 1
    with pytest.raises(fastlane.HandoffContractError, match="re-POST"):
        adapter.submit_verification("name", "work", {})


def test_recovery_adapter_never_calls_scheduler_delegate(
    tmp_path: Path,
) -> None:
    class Forbidden:
        @staticmethod
        def submit_verification(*_args, **_kwargs):
            raise AssertionError("recovery attempted a Scheduler POST")

    adapter = fastlane._SubmissionAdapter(
        delegate=Forbidden,
        account_name="dhj02",
        guard=lambda: None,
        post_result_path=tmp_path / "post_result.json",
        recovered_task_id=96310,
    )
    assert adapter.submit_verification("name", "work", {}) == 96310
    assert adapter.scheduler_post_count == 0
    assert not (tmp_path / "post_result.json").exists()


def test_singleton_claim_rejects_a_second_full_winner(tmp_path: Path) -> None:
    authority = fastlane._initialize_claim_root(tmp_path / "claims")
    reference = fastlane._claim_reference(authority)
    winner = {
        "immediate_task_id": 1,
        "immediate_retry_kind": "none",
        "plan_payload_sha256": "1" * 64,
        "plan_file_sha256": "2" * 64,
        "profile_sha256": "3" * 64,
        "resources": dict(fastlane.FULL_RESOURCES),
        "task_name": "mft-goal-truth-full-first",
        "dedupe_key": "mft:test:first",
    }
    first = atomic_claim.acquire_claim(
        tmp_path / "claims", reference, winner
    )
    assert first["status"] == "fresh_pending"
    second_winner = dict(winner)
    second_winner["task_name"] = "mft-goal-truth-full-second"
    second_winner["dedupe_key"] = "mft:test:second"
    with pytest.raises(
        atomic_claim.ClaimContractError,
        match="different ancestry",
    ):
        atomic_claim.acquire_claim(
            tmp_path / "claims", reference, second_winner
        )


def test_active_4fac_is_mandatory_at_watch_plan_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_parent = tmp_path / "terminal"
    state_parent.mkdir()
    cutover = tmp_path / "cutover.json"
    cutover.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        fastlane,
        "_initialize_claim_root",
        lambda _root: {"resolved_root": str(tmp_path / "claims")},
    )
    monkeypatch.setattr(
        fastlane,
        "_claim_reference",
        lambda _authority: {"payload_sha256": "1" * 64},
    )
    monkeypatch.setattr(
        fastlane.diagnostic,
        "_validate_scheduler_cutover_receipt",
        lambda *_args, **_kwargs: (
            {
                "candidate_revision": "0" * 40,
                "pin_generation": "scheduler-strict-node-legacy",
                "payload_sha256": "2" * 64,
            },
            None,
        ),
    )
    with pytest.raises(
        fastlane.HandoffContractError,
        match="requires active Scheduler 4fac",
    ):
        fastlane.initialize_watch_plan(
            terminal_state_path=state_parent / "state.json",
            scheduler_cutover_receipt_path=cutover,
            license_snapshot_directory=tmp_path / "licenses",
            gpfs_audit_directory=tmp_path / "gpfs",
            output_root=tmp_path / "fastlane",
            claim_root=tmp_path / "claims",
        )
