import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import regression_260707.monitoring.deadline_design as deadline_design_module
import regression_260707.monitoring.release_canary as release_canary_module
from regression_260707.monitoring.local_aedt_gui import (
    DEFAULT_RUNNER_SHA256,
    DEFAULT_SOLVER_REVISION,
    GUI_RESULT_SCHEMA,
)
from regression_260707.monitoring.deadline_design import (
    DEADLINE_LOCAL_GUI_RUNNER_SHA256,
    EXPECTED_LOCAL_GUI_SOLVER,
)
from regression_260707.monitoring.release_canary import (
    BLOCKER_HPO_V2_STATUS_SCHEMA,
    CanaryFailure,
    CURRENT7_AGGREGATE_SCHEMA,
    CURRENT7_COMPATIBILITY_SCHEMA,
    CURRENT7_CONDITION_STATUS_MAX_BYTES,
    CURRENT7_CONSTRAINT_NAMES,
    CURRENT7_HARD_SPEC,
    CURRENT7_HARD_SPEC_SHA256,
    CURRENT7_INDEX_SCHEMA,
    CURRENT7_STATUS_SCHEMA,
    CURRENT7_TEMPERATURE_TARGETS,
    EXPECTED_DUAL_STANDARD_FEA_TASKS,
    EXPECTED_TIER1_TEMPERATURE_TARGETS,
    _authenticate_blocker_hpo_v2_status,
    _authenticate_current7_condition_indexes,
    _authenticate_current7_index,
    _current7_constraint_identity_sha256,
    _dual_release_configuration,
    _refresh_blocker_hpo_v2_status,
    _refresh_current7_index,
    _verify_blocker_hpo_v2_api,
    _verify_current7_api,
    _verify_current7_condition_archive_nsga_api,
    _verify_current7_condition_searches,
    _verify_dual_standard_fea,
    _verify_local_gui_launches,
    _verify_nsga_api,
    _verify_nsga_progress,
    _wait_for_blocker_hpo_v2_api,
    _wait_for_stable_nsga_generation,
    _require_utf8_display_text,
    _write_new_json,
)
from regression_260707.monitoring.runtime_transition import (
    _read_sealed_json,
    _require_process_location,
    _verify_optional_runtime_artifacts,
    _write_sidecar,
)


def _nsga_payload():
    hard_spec = {"T_limit_C": 120.0, "resonance_min_Hz": 10_000.0}
    temperature_contract = {
        "robust_upper_bound_C": 120.0,
        "target_count": len(EXPECTED_TIER1_TEMPERATURE_TARGETS),
        "targets": list(EXPECTED_TIER1_TEMPERATURE_TARGETS),
    }
    return {
        "schema_version": 1,
        "available": True,
        "note": "현재 모델의 NSGA-II 상태입니다.",
        "candidate_count": 1,
        "candidates": [{"id": "candidate-1"}],
        "tier1_feedback_search": {
            "available": True,
            "integrity_verified": True,
            "running_count": 4,
            "queued_count": 5,
            "attaching_count": 1,
            "active_plus_queued": 10,
            "completed_count": 3,
            "terminal_results_verified": 3,
            "feasible_pareto_count": 1,
            "candidate_preview_count": 1,
            "candidate_preview_limit": 128,
            "candidate_preview_truncated": False,
            "near_feasible_count": 7,
            "near_feasible_preview_count": 7,
            "near_feasible_preview_limit": 64,
            "near_feasible_preview_truncated": False,
            "coherent_snapshot_verified": True,
            "coherent_snapshot_attempts": 1,
            "constraint_version": "constraint-v2",
            "constraints": hard_spec,
            "temperature_constraint_contract": temperature_contract,
            "model_manifest_sha256": "1" * 64,
            "deployment_model_manifest_sha256": "2" * 64,
            "updated_at": "2026-07-18T15:00:00+09:00",
        },
    }


def _progress_payload(nsga_payload):
    search = nsga_payload["tier1_feedback_search"]
    return {
        "schema_version": 1,
        "available": search["available"],
        "integrity_verified": search["integrity_verified"],
        "counters_consistent": True,
        "candidate_preview_consistent": True,
        "source_endpoint": "/api/nsga2",
        **{
            name: search[name]
            for name in (
                "running_count",
                "queued_count",
                "attaching_count",
                "active_plus_queued",
                "completed_count",
                "terminal_results_verified",
                "feasible_pareto_count",
                "candidate_preview_count",
                "candidate_preview_limit",
                "candidate_preview_truncated",
                "near_feasible_count",
                "near_feasible_preview_count",
                "near_feasible_preview_limit",
                "near_feasible_preview_truncated",
                "coherent_snapshot_verified",
                "coherent_snapshot_attempts",
                "constraint_version",
                "constraints",
                "temperature_constraint_contract",
                "model_manifest_sha256",
                "deployment_model_manifest_sha256",
                "updated_at",
            )
        },
    }


def _write_canonical_json(path: Path, value: dict) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _current7_index_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = (tmp_path / "current7").resolve()
    status_path = root / "cohorts" / "cohort-1" / "status.json"
    compatibility_path = root / "cohorts" / "cohort-1" / "compatibility.json"
    snapshot_sha = "9" * 64
    bundle_sha = "8" * 64
    status_event_at = "2026-07-20T06:00:00+00:00"
    harvest_observed_at = "2026-07-20T06:00:04+00:00"
    constraint_identity = {
        "constraint_version": "current7-constraint-v1",
        "hard_spec": dict(CURRENT7_HARD_SPEC),
        "hard_spec_sha256": CURRENT7_HARD_SPEC_SHA256,
        "hard_constraint_contract_sha256": "c" * 64,
        "temperature_contract_sha256": "d" * 64,
        "constraint_names": list(CURRENT7_CONSTRAINT_NAMES),
        "temperature_targets": list(CURRENT7_TEMPERATURE_TARGETS),
    }
    status_sha = _write_canonical_json(
        status_path,
        {
            "schema_version": CURRENT7_STATUS_SCHEMA,
            "cohort_id": "cohort-1",
            "bundle_id": "bundle-1",
            "bundle_manifest_sha256": bundle_sha,
            "snapshot_sha256": snapshot_sha,
            "updated_at": status_event_at,
            "healthy": True,
            "refused_terminal_count": 0,
            "production_eligible": False,
            "fea_submission_approved": False,
            "fea_submission_performed": False,
            "aedt_used": False,
            "automatic_promotion_allowed": False,
            **constraint_identity,
            "aggregate": {
                "schema_version": CURRENT7_AGGREGATE_SCHEMA,
                "production_eligible": False,
                "fea_submission_approved": False,
                "fea_submission_performed": False,
                **constraint_identity,
            },
        },
    )
    compatibility_sha = _write_canonical_json(
        compatibility_path,
        {
            "schema_version": CURRENT7_COMPATIBILITY_SCHEMA,
            "compatible": False,
        },
    )
    index_path = root / "canonical" / "current7-index.json"
    _write_canonical_json(
        index_path,
        {
            "schema_version": CURRENT7_INDEX_SCHEMA,
            "active_cohort_id": "cohort-1",
            "bundle_id": "bundle-1",
            "bundle_manifest_sha256": bundle_sha,
            "path_containment_root": str(root),
            "snapshot_sha256": snapshot_sha,
            "updated_at": status_event_at,
            "status_event_at": status_event_at,
            "harvest_observed_at": harvest_observed_at,
            **constraint_identity,
            "status": {
                "path": str(status_path),
                "schema_version": CURRENT7_STATUS_SCHEMA,
                "sha256": status_sha,
            },
            "compatibility": {
                "path": str(compatibility_path),
                "schema_version": CURRENT7_COMPATIBILITY_SCHEMA,
                "sha256": compatibility_sha,
            },
            "production_eligible": False,
            "fea_submission_approved": False,
            "fea_submission_performed": False,
            "aedt_used": False,
            "automatic_promotion_allowed": False,
        },
    )
    return index_path, status_path


def _advance_current7_snapshot(
    index_path: Path,
    status_path: Path,
    snapshot_sha256: str,
) -> None:
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["snapshot_sha256"] = snapshot_sha256
    status_sha = _write_canonical_json(status_path, status)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["snapshot_sha256"] = snapshot_sha256
    index["status"]["sha256"] = status_sha
    _write_canonical_json(index_path, index)


def _current7_api_payload(index_path: Path) -> dict:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index_sha = hashlib.sha256(index_path.read_bytes()).hexdigest()
    return {
        "schema_version": 1,
        "available": True,
        "note": "Current7 NSGA-II status",
        "candidate_count": 0,
        "candidates": [],
        "search_authority": {
            "configured": True,
            "kind": "current7",
            "available": True,
            "integrity_verified": True,
            "legacy_role": "archive",
        },
        "tier1_feedback_search": {
            "available": False,
            "integrity_verified": False,
            "authority_role": "archive",
            "archived": True,
        },
        "tier1_current7_search": {
            "available": True,
            "integrity_verified": True,
            "healthy": True,
            "source": str(index_path),
            "pointer_verified": True,
            "gui_launch_eligible": False,
            "warnings": [],
            "search_count": 2,
            "running_count": 1,
            "queued_count": 1,
            "attaching_count": 0,
            "completed_count": 0,
            "failed_count": 0,
            "cancelled_count": 0,
            "timeout_count": 0,
            "active_plus_queued": 2,
            "authenticated_terminal_seed_count": 0,
            "refused_terminal_count": 0,
            "feasible_pareto_count": 0,
            "near_feasible_count": 0,
            "candidate_preview_count": 0,
            "candidate_preview_limit": 256,
            "candidate_preview_truncated": False,
            "freshness_required": True,
            "freshness_ok": True,
            "freshness_age_seconds": 4.0,
            "harvest_observed_at": index["harvest_observed_at"],
            "status_event_at": index["status_event_at"],
            "state_counts": {"queued": 1, "running": 1},
            "snapshot_sha256": index["snapshot_sha256"],
            "index_file_sha256": index_sha,
            "snapshot_file_sha256": index["status"]["sha256"],
            "snapshot_attempts": 1,
            "bundle_manifest_sha256": index["bundle_manifest_sha256"],
            "cohort_id": index["active_cohort_id"],
            "bundle_id": index["bundle_id"],
            "constraints": dict(CURRENT7_HARD_SPEC),
            "constraint_names": list(CURRENT7_CONSTRAINT_NAMES),
            "temperature_targets": list(CURRENT7_TEMPERATURE_TARGETS),
            "constraint_version": "current7-constraint-v1",
            "hard_spec_sha256": CURRENT7_HARD_SPEC_SHA256,
            "temperature_contract_sha256": index[
                "temperature_contract_sha256"
            ],
            "constraint_identity_sha256": (_current7_constraint_identity_sha256(index)),
            "constraint_contract_verified": True,
            "authority_eligible": True,
            "updated_at": "2026-07-20T06:00:00+00:00",
        },
    }


def _current7_progress_payload(nsga_payload: dict) -> dict:
    search = nsga_payload["tier1_current7_search"]
    bundle_sha = search["bundle_manifest_sha256"]
    return {
        "schema_version": 1,
        "available": True,
        "integrity_verified": True,
        "counters_consistent": True,
        "source_endpoint": "/api/nsga2",
        "authority_kind": "current7",
        "authority_identity_sha256": bundle_sha,
        "authority_index_file_sha256": search["index_file_sha256"],
        "authority_snapshot_sha256": search["snapshot_file_sha256"],
        "authority_snapshot_identity_sha256": search["snapshot_sha256"],
        "freshness_required": search["freshness_required"],
        "freshness_ok": search["freshness_ok"],
        "freshness_age_seconds": search["freshness_age_seconds"],
        "harvest_observed_at": search["harvest_observed_at"],
        "status_event_at": search["status_event_at"],
        "search_count": search["search_count"],
        "running_count": search["running_count"],
        "queued_count": search["queued_count"],
        "attaching_count": search["attaching_count"],
        "active_plus_queued": search["active_plus_queued"],
        "completed_count": search["completed_count"],
        "failed_count": search["failed_count"],
        "cancelled_count": search["cancelled_count"],
        "timeout_count": search["timeout_count"],
        "terminal_results_verified": search["authenticated_terminal_seed_count"],
        "refused_terminal_count": search["refused_terminal_count"],
        "feasible_pareto_count": search["feasible_pareto_count"],
        "near_feasible_count": search["near_feasible_count"],
        "state_counts": search["state_counts"],
        "candidate_preview_consistent": True,
        "candidate_preview_count": search["candidate_preview_count"],
        "candidate_preview_limit": search["candidate_preview_limit"],
        "candidate_preview_truncated": search["candidate_preview_truncated"],
        "near_feasible_preview_count": 0,
        "near_feasible_preview_limit": 1,
        "near_feasible_preview_truncated": False,
        "constraint_version": search["constraint_version"],
        "constraints": search["constraints"],
        "constraint_names": search["constraint_names"],
        "temperature_targets": search["temperature_targets"],
        "temperature_constraint_contract": {
            "schema_version": (
                "mft-tier1-current7-temperature-reference-v1"
            ),
            "sha256": search["temperature_contract_sha256"],
            "robust_upper_bound_C": 110.0,
            "target_count": len(CURRENT7_TEMPERATURE_TARGETS),
            "targets": list(CURRENT7_TEMPERATURE_TARGETS),
        },
        "constraint_identity_sha256": search["constraint_identity_sha256"],
        "model_manifest_sha256": bundle_sha,
        "deployment_model_manifest_sha256": bundle_sha,
        "coherent_snapshot": True,
        "coherent_snapshot_verified": True,
        "coherent_snapshot_attempts": search["snapshot_attempts"],
        "updated_at": search["updated_at"],
    }


def _hpo_v2_status_fixture(tmp_path: Path) -> Path:
    path = (tmp_path / "hpo-v2" / "status.json").resolve()
    _write_canonical_json(
        path,
        {
            "schema_version": BLOCKER_HPO_V2_STATUS_SCHEMA,
            "phase": "optimizing",
            "heartbeat_at": "2026-07-20T00:00:00+00:00",
            "config_sha256": "a" * 64,
            "dataset_sha256": "b" * 64,
            "selected_cumulative_trials_per_job": 20,
            "jobs": {
                "target-a:model-a": {
                    "total": 20,
                    "complete": 7,
                    "running": 1,
                    "failed": 0,
                },
                "target-b:model-b": {
                    "total": 20,
                    "complete": 5,
                    "running": 1,
                    "failed": 1,
                },
            },
            "trials": {
                "total": 40,
                "complete": 12,
                "running": 2,
                "failed": 1,
            },
            "production_eligible": False,
            "production_model_eligible": False,
            "fea_submission_approved": False,
            "promotion_approved": False,
        },
    )
    return path


def _authenticated_data_cohort_fixture(
    tmp_path: Path,
) -> tuple[Path, dict[str, object]]:
    import pyarrow as pa
    import pyarrow.parquet as parquet

    pipeline_root = tmp_path / "pipeline"
    generation_id = "3" * 64
    parquet_path = (
        pipeline_root / "artifacts" / "dataset" / generation_id
        / "train.parquet"
    )
    parquet_path.parent.mkdir(parents=True)
    parquet.write_table(pa.table({
        name: [0]
        for name in release_canary_module.PRIMARY_PARQUET_COLUMNS
    }), parquet_path)
    cohort = {
        "available": True,
        "counts_available": True,
        "authority_verified": True,
        "manifest_matches_audit": True,
        "raw_rows": 1,
        "strict_em_rows": 1,
        "strict_full_rows": 1,
        "counts_source": "authenticated_train.parquet_quality_contract",
        "generation_id": generation_id,
        "generation": f"dataset:{generation_id}",
        "controller_generation": f"dataset:{generation_id}",
        "is_controller_generation": True,
        "artifact_sha256": hashlib.sha256(parquet_path.read_bytes()).hexdigest(),
        "solver_revision": "a" * 40,
        "library_revision": "b" * 40,
        "quality_contract_sha256": "c" * 64,
        "profile_sha256": "d" * 64,
    }
    return pipeline_root, cohort


def test_release_canary_accepts_only_authenticated_sealed_data_fallback(
    tmp_path,
):
    pipeline_root, sealed = _authenticated_data_cohort_fixture(tmp_path)
    sealed = {
        **sealed,
        "kind": "latest_completed_training_snapshot",
    }
    payload = {
        "schema_version": 1,
        "active_cohort": {"label": "활성 코호트"},
        "latest_eligible_cohort": {
            "available": False,
            "counts_available": False,
            "authority_verified": False,
            "manifest_matches_audit": False,
            "scan_truncated": False,
            "error": "controller role is not alive; controller status is stale",
            "counts_error": (
                "controller role is not alive; controller status is stale"
            ),
        },
        "latest_completed_training": {"snapshot": sealed, "job": None},
    }

    cohort, parquet_read = release_canary_module._verify_data_api(
        payload, pipeline_root
    )

    assert cohort["authority_source"] == "latest_completed_training.snapshot"
    assert cohort["strict_full_rows"] == 1
    assert parquet_read["validated_full"] is True


def test_release_canary_rejects_unverified_live_and_sealed_data(tmp_path):
    pipeline_root, sealed = _authenticated_data_cohort_fixture(tmp_path)
    sealed = {
        **sealed,
        "kind": "latest_completed_training_snapshot",
        "authority_verified": False,
    }
    payload = {
        "schema_version": 1,
        "active_cohort": {"label": "활성 코호트"},
        "latest_eligible_cohort": {
            "available": False,
            "counts_available": False,
            "authority_verified": False,
            "scan_truncated": False,
            "error": "controller role is not alive; controller status is stale",
            "counts_error": (
                "controller role is not alive; controller status is stale"
            ),
        },
        "latest_completed_training": {"snapshot": sealed, "job": None},
    }

    with pytest.raises(CanaryFailure, match="no fully authenticated"):
        release_canary_module._verify_data_api(payload, pipeline_root)


def test_release_canary_forbids_sealed_fallback_on_live_identity_drift(
    tmp_path,
):
    pipeline_root, sealed = _authenticated_data_cohort_fixture(tmp_path)
    sealed = {
        **sealed,
        "kind": "latest_completed_training_snapshot",
    }
    payload = {
        "schema_version": 1,
        "active_cohort": {"label": "활성 코호트"},
        "latest_eligible_cohort": {
            "available": False,
            "counts_available": False,
            "authority_verified": False,
            "scan_truncated": False,
            "error": "artifact hash identity diverged",
            "counts_error": "artifact hash identity diverged",
        },
        "latest_completed_training": {"snapshot": sealed, "job": None},
    }

    with pytest.raises(CanaryFailure, match="non-stale reason"):
        release_canary_module._verify_data_api(payload, pipeline_root)


def _advance_hpo_v2_status(path: Path) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    value.update(
        {
            "phase": "optimizing",
            "heartbeat_at": "2026-07-20T00:01:00+00:00",
            "selected_cumulative_trials_per_job": 50,
            "jobs": {
                "target-a:model-a": {
                    "total": 50,
                    "complete": 20,
                    "running": 1,
                    "failed": 0,
                },
                "target-b:model-b": {
                    "total": 50,
                    "complete": 18,
                    "running": 1,
                    "failed": 1,
                },
            },
            "trials": {
                "total": 100,
                "complete": 38,
                "running": 2,
                "failed": 1,
            },
        }
    )
    _write_canonical_json(path, value)


def _hpo_v2_dashboard(status: dict) -> dict:
    return {
        "continuous_pipeline": {
            "blocker_hpo_v2": {
                "available": True,
                "validated_running": status["phase"] == "optimizing",
                "state": status["phase"],
                "source": status["path"],
                "config_sha256": status["config_sha256"],
                "dataset_sha256": status["dataset_sha256"],
                "job_count": status["job_count"],
                "stage_trials_per_job": status["stage_trials_per_job"],
                "trial_total": status["trial_total"],
                "trial_complete": status["trial_complete"],
                "trial_running": status["trial_running"],
                "trial_failed": status["trial_failed"],
            },
        },
    }


def _dual_expected(tmp_path: Path):
    allowed_root = tmp_path / "allowed"
    runtime_root = allowed_root / "dual-runtime"
    runtime_root.mkdir(parents=True)
    hashes = {
        "handoff": "a" * 64,
        "plan": "b" * 64,
        "ui": "c" * 64,
        "receipt": "d" * 64,
        "manifest": "e" * 64,
        "execution": "f" * 64,
        "source": "1" * 64,
    }
    return {
        "sealed_successor_handoff": {
            "path": "C:/sealed/handoff.json",
            "sha256": hashes["handoff"],
        },
        "dual_successor_plan": {
            "path": "C:/sealed/plan.json",
            "sha256": hashes["plan"],
        },
        "dual_successor_ui": {"path": "C:/sealed/ui.json", "sha256": hashes["ui"]},
        "dual_successor_receipt": {
            "path": "C:/sealed/receipt.json",
            "sha256": hashes["receipt"],
        },
        "standard_fea_validation": {
            "runtime_root": str(runtime_root.resolve()),
            "allowed_root": str(allowed_root.resolve()),
            "runtime_manifest": {
                "path": str(runtime_root / "manifest.json"),
                "sha256": hashes["manifest"],
            },
            "execution_plan": {
                "path": "C:/sealed/execution-plan.json",
                "sha256": hashes["execution"],
            },
            "sealed_plan_source": {
                "path": str(
                    runtime_root / "sealed-plan-source" / "experimental_source.json"
                ),
                "sha256": hashes["source"],
            },
            "expected_tasks": [],
        },
    }


def _dual_payload(expected):
    validation_integrity = {
        "manifest_sha256": expected["standard_fea_validation"]["runtime_manifest"][
            "sha256"
        ],
        "sealed_plan_source_sha256": expected["standard_fea_validation"][
            "sealed_plan_source"
        ]["sha256"],
        "execution_plan_sha256": expected["standard_fea_validation"]["execution_plan"][
            "sha256"
        ],
        "state_sha256": "2" * 64,
        "status_sha256": "3" * 64,
        "allowed_root": expected["standard_fea_validation"]["allowed_root"],
    }
    overlays = {}
    candidates = []
    for candidate_sha, identity in sorted(EXPECTED_DUAL_STANDARD_FEA_TASKS.items()):
        overlay = {
            "schema_version": "mft-monitor-standard-fea-validation-v1",
            "configured": True,
            "available": True,
            "integrity_verified": True,
            "read_only": True,
            "runtime_state": "active",
            "submission_state": "submitted",
            "task_status": "running",
            "collection_state": "not_terminal",
            "task": {
                "id": identity["task_id"],
                "name": identity["task_name"],
                "status": "running",
            },
            "candidate_digest": identity["candidate_digest"],
            "result_identity": {
                "available": False,
                "path": None,
                "sha256": None,
                "candidate_digest": identity["candidate_digest"],
                "task_id": identity["task_id"],
                "contract_valid": False,
                "candidate_identity_matches": False,
            },
            "measurements": None,
            "collected": False,
            "valid": False,
            "actual_gates": None,
            "actual_hard_gates_pass": None,
            "full_model_validation_candidate_eligible": False,
            "pareto_eligible": False,
            "production_eligible": False,
            "integrity": {
                "manifest_sha256": validation_integrity["manifest_sha256"],
                "execution_plan_sha256": validation_integrity["execution_plan_sha256"],
                "state_sha256": validation_integrity["state_sha256"],
                "status_sha256": validation_integrity["status_sha256"],
                "candidate_record_sha256": "4" * 64,
            },
            "warning": None,
        }
        overlays[candidate_sha] = overlay
        candidates.append(
            {
                "schema_version": "mft-monitor-sealed-successor-v1",
                "configured": True,
                "available": True,
                "status": "validate_only",
                "lifecycle_state": "sealed_successor",
                "source_kind": "sealed_successor_evidence",
                "integrity_verified": True,
                "display_only": True,
                "read_only": True,
                "canonical_candidate": False,
                "terminal_result": False,
                "pareto": False,
                "production": False,
                "production_eligible": False,
                "production_submission_enabled": False,
                "fea": False,
                "fea_submission_enabled": False,
                "fea_submission_approved": False,
                "fea_submission_performed": False,
                "fea_status": "not_submitted",
                "scheduler_task_id": None,
                "candidate": {"candidate_sha256": candidate_sha},
                "standard_fea_validation": overlay,
            }
        )
    return {
        "schema_version": "mft-monitor-dual-sealed-successors-v1",
        "configured": True,
        "available": True,
        "status": "validate_only",
        "source_kind": "sealed_successor_evidence",
        "integrity_verified": True,
        "display_only": True,
        "read_only": True,
        "candidate_count": 2,
        "canonical_candidate": False,
        "terminal_result": False,
        "pareto": False,
        "production": False,
        "production_eligible": False,
        "fea": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "scheduler_mutation_performed": False,
        "integrity": {
            "validateonly_plan_sha256": expected["dual_successor_plan"]["sha256"],
            "read_only_ui_sha256": expected["dual_successor_ui"]["sha256"],
            "validation_receipt_sha256": expected["dual_successor_receipt"]["sha256"],
        },
        "candidates": candidates,
        "standard_fea_validation": {
            "schema_version": "mft-monitor-dual-standard-fea-runtime-v1",
            "configured": True,
            "available": True,
            "integrity_verified": True,
            "read_only": True,
            "candidate_count": 2,
            "candidates": overlays,
            "runtime_root": expected["standard_fea_validation"]["runtime_root"],
            "runtime_state": "active",
            "updated_at": "2026-07-18T20:00:00+09:00",
            "integrity": validation_integrity,
            "warning": None,
        },
    }


def _actual_t120_gate():
    return {
        "contract": "mft-tier1-actual-Llt-and-conditional-all11-T120-v1",
        "Llt_target_uH": 27.5,
        "Llt_tolerance_uH": 0.55,
        "temperature_limit_C": 120.0,
        "temperature_contract": {
            "robust_upper_bound_C": 120.0,
            "target_count": len(EXPECTED_TIER1_TEMPERATURE_TARGETS),
            "targets": list(EXPECTED_TIER1_TEMPERATURE_TARGETS),
        },
        "pass": True,
        "reasons": [],
        "actual": {
            "Llt_full_uH": 27.5,
            "N2_side": 21,
            "temperatures": {
                name: {"applicable": True, "value_C": 105.0, "pass": True}
                for name in EXPECTED_TIER1_TEMPERATURE_TARGETS
            },
        },
    }


def test_release_canary_verifies_nsga_and_progress_contracts():
    nsga = _verify_nsga_api(_nsga_payload())
    progress = {
        "schema_version": 1,
        "available": True,
        "integrity_verified": True,
        "counters_consistent": True,
        "candidate_preview_consistent": True,
        "source_endpoint": "/api/nsga2",
        "active_plus_queued": 10,
        "running_count": 4,
        "queued_count": 5,
        "attaching_count": 1,
        "terminal_results_verified": 3,
        "completed_count": 3,
        "feasible_pareto_count": 1,
        "candidate_preview_count": 1,
        "candidate_preview_limit": 128,
        "candidate_preview_truncated": False,
        "near_feasible_count": 7,
        "near_feasible_preview_count": 7,
        "near_feasible_preview_limit": 64,
        "near_feasible_preview_truncated": False,
        "coherent_snapshot_verified": True,
        "coherent_snapshot_attempts": 1,
        "constraint_version": "constraint-v2",
        "constraints": nsga["constraints"],
        "temperature_constraint_contract": nsga["temperature_constraint_contract"],
        "model_manifest_sha256": "1" * 64,
        "deployment_model_manifest_sha256": "2" * 64,
    }
    verified = _verify_nsga_progress(progress, nsga)
    assert verified["integrity_verified"] is True
    assert verified["active_plus_queued"] == 10


def test_release_canary_rejects_nsga_counter_mismatch():
    payload = _nsga_payload()
    payload["tier1_feedback_search"]["active_plus_queued"] = 11
    with pytest.raises(CanaryFailure, match="counters disagree"):
        _verify_nsga_api(payload)


def _condition_projection(path: Path, configured: dict) -> dict:
    return {
        "configured": True,
        "available": True,
        "integrity_verified": True,
        "healthy": False,
        "constraint_contract_verified": True,
        "authority_eligible": False,
        "display_only": True,
        "read_only": True,
        "pointer_verified": True,
        "gui_launch_eligible": False,
        "source": str(path.resolve()),
        "lanes": [],
        "warnings": ["authenticated terminal refusals remain read-only"],
        "index_file_sha256": configured["sha256"],
        "snapshot_file_sha256": configured["status"]["sha256"],
        "snapshot_sha256": configured["snapshot_sha256"],
        "constraint_version": configured["constraint_version"],
        "hard_spec_sha256": configured["hard_spec_sha256"],
        "hard_constraint_contract_sha256": configured[
            "hard_constraint_contract_sha256"
        ],
        "temperature_contract_sha256": configured[
            "temperature_contract_sha256"
        ],
        "bundle_manifest_sha256": configured["bundle_manifest_sha256"],
        "state_counts": {"completed": 3, "running": 1, "queued": 2},
        "search_count": 6,
        "running_count": 1,
        "queued_count": 2,
        "attaching_count": 0,
        "active_plus_queued": 3,
        "completed_count": 3,
        "failed_count": 0,
        "refused_terminal_count": 1,
        "feasible_pareto_count": 0,
        "near_feasible_count": 1,
        "freshness_required": True,
        "freshness_ok": False,
    }


def test_release_canary_authenticates_current7_condition_index_and_api(tmp_path):
    index_path, _ = _current7_index_fixture(tmp_path)
    configured = _authenticate_current7_condition_indexes([index_path])
    assert len(configured) == 1
    assert configured[0]["status_max_bytes"] == 64 * 1024 * 1024
    assert CURRENT7_CONDITION_STATUS_MAX_BYTES == 64 * 1024 * 1024

    payload = {
        "schema_version": 1,
        "available": True,
        "source_kind": "current7_condition_archive",
        "search_authority": {
            "configured": False,
            "kind": "condition_archive",
            "available": False,
            "integrity_verified": False,
            "source": None,
            "legacy_role": "archive",
        },
        "tier1_feedback_search": {
            "authority_role": "archive",
            "archived": True,
        },
        "constraint_version": configured[0]["constraint_version"],
        "constraints": dict(CURRENT7_HARD_SPEC),
        "tier1_current7_condition_searches": [
            _condition_projection(index_path, configured[0])
        ],
        "near_feasible_preview": [],
        "near_feasible_preview_count": 0,
    }
    verified = _verify_current7_condition_searches(payload, configured)
    assert verified[0]["read_only"] is True
    assert verified[0]["authority_eligible"] is False
    summary = _verify_current7_condition_archive_nsga_api(payload, configured)
    assert summary["mode"] == "current7_condition_archive"
    assert summary["condition_count"] == 1
    assert summary["integrity_verified"] is True


def test_release_canary_accepts_deadline_design_over_condition_archive(
    tmp_path,
):
    index_path, _ = _current7_index_fixture(tmp_path)
    configured = _authenticate_current7_condition_indexes([index_path])
    payload = {
        "schema_version": 1,
        "available": True,
        "source_kind": "deadline_design_with_condition_archive",
        "search_authority": {
            "configured": False,
            "kind": "condition_archive",
            "available": False,
            "integrity_verified": False,
            "source": None,
            "legacy_role": "archive",
        },
        "tier1_feedback_search": {
            "authority_role": "archive",
            "archived": True,
        },
        "constraint_version": "deadline-20260724-0900-kst-v1",
        "constraints": {"T_limit_C": 100.0},
        "tier1_current7_condition_searches": [
            _condition_projection(index_path, configured[0])
        ],
        "near_feasible_preview": [],
        "near_feasible_preview_count": 0,
    }

    summary = _verify_current7_condition_archive_nsga_api(
        payload, configured
    )

    assert summary["mode"] == "deadline_design_with_condition_archive"
    assert summary["integrity_verified"] is True


def test_release_canary_requires_coherent_condition_archive_progress(tmp_path):
    index_path, _ = _current7_index_fixture(tmp_path)
    configured = _authenticate_current7_condition_indexes([index_path])
    nsga_payload = {
        "schema_version": 1,
        "available": True,
        "source_kind": "current7_condition_archive",
        "search_authority": {
            "configured": False,
            "kind": "condition_archive",
            "available": False,
            "integrity_verified": False,
            "source": None,
            "legacy_role": "archive",
        },
        "tier1_feedback_search": {
            "authority_role": "archive",
            "archived": True,
        },
        "constraint_version": configured[0]["constraint_version"],
        "constraints": dict(CURRENT7_HARD_SPEC),
        "tier1_current7_condition_searches": [
            _condition_projection(index_path, configured[0])
        ],
        "near_feasible_preview": [],
        "near_feasible_preview_count": 0,
    }
    nsga = _verify_current7_condition_archive_nsga_api(
        nsga_payload, configured
    )
    progress_payload = {
        "schema_version": 1,
        "available": True,
        "integrity_verified": True,
        "source_endpoint": "/api/nsga2",
        "source_kind": "current7_condition_archive",
        "authority_kind": "none",
        "condition_count": 1,
        "source_count": 1,
        "configured_source_count": 1,
        "rejected_source_count": 0,
        "all_configured_sources_verified": True,
        "counters_consistent": True,
        "candidate_preview_consistent": True,
        "search_count": 6,
        "running_count": 1,
        "queued_count": 2,
        "attaching_count": 0,
        "active_plus_queued": 3,
        "completed_count": 3,
        "failed_count": 0,
        "feasible_pareto_count": 0,
        "candidate_count": 0,
        "near_feasible_count": 1,
        "near_feasible_preview_count": 0,
        "status": "running",
    }

    verified = release_canary_module._verify_current7_condition_archive_progress(
        progress_payload, nsga
    )
    assert verified["available"] is True
    assert verified["condition_count"] == 1

    progress_payload["source_count"] = 0
    with pytest.raises(CanaryFailure, match="counters are inconsistent"):
        release_canary_module._verify_current7_condition_archive_progress(
            progress_payload, nsga
        )


def test_release_canary_rejects_legacy_authority_in_condition_archive(tmp_path):
    index_path, _ = _current7_index_fixture(tmp_path)
    configured = _authenticate_current7_condition_indexes([index_path])
    payload = {
        "schema_version": 1,
        "available": True,
        "source_kind": "current7_condition_archive",
        "search_authority": {
            "configured": False,
            "kind": "legacy",
            "available": True,
            "integrity_verified": True,
            "source": "legacy.json",
            "legacy_role": "primary",
        },
        "tier1_feedback_search": {
            "authority_role": "primary",
            "archived": False,
        },
        "constraint_version": configured[0]["constraint_version"],
        "constraints": dict(CURRENT7_HARD_SPEC),
        "tier1_current7_condition_searches": [
            _condition_projection(index_path, configured[0])
        ],
        "near_feasible_preview": [],
        "near_feasible_preview_count": 0,
    }

    with pytest.raises(CanaryFailure, match="authority boundary diverged"):
        _verify_current7_condition_archive_nsga_api(payload, configured)


def test_release_canary_rejects_condition_projection_authority(tmp_path):
    index_path, _ = _current7_index_fixture(tmp_path)
    configured = _authenticate_current7_condition_indexes([index_path])
    search = _condition_projection(index_path, configured[0])
    search["authority_eligible"] = True
    with pytest.raises(CanaryFailure, match="asserted authority"):
        _verify_current7_condition_searches(
            {"tier1_current7_condition_searches": [search]}, configured
        )


def test_release_canary_authenticates_current7_index_and_api(tmp_path):
    index_path, _ = _current7_index_fixture(tmp_path)
    identity = _authenticate_current7_index(index_path)

    verified = _verify_current7_api(_current7_api_payload(index_path), identity)

    assert identity["nested_sha256_verified"] is True
    assert verified["integrity_verified"] is True
    assert verified["active_plus_queued"] == 2
    assert verified["configured_index_sha256"] == identity["sha256"]
    assert identity["status_event_at"] == "2026-07-20T06:00:00+00:00"
    assert identity["harvest_observed_at"] == "2026-07-20T06:00:04+00:00"


def test_release_canary_rejects_current7_status_event_identity_drift(tmp_path):
    index_path, _ = _current7_index_fixture(tmp_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["status_event_at"] = "2026-07-20T06:00:01+00:00"
    _write_canonical_json(index_path, index)

    with pytest.raises(CanaryFailure, match="status event identity diverged"):
        _authenticate_current7_index(index_path)


def test_release_canary_rejects_current7_api_heartbeat_identity_drift(tmp_path):
    index_path, _ = _current7_index_fixture(tmp_path)
    identity = _authenticate_current7_index(index_path)
    payload = _current7_api_payload(index_path)
    payload["tier1_current7_search"]["harvest_observed_at"] = (
        "2026-07-20T06:00:05+00:00"
    )

    with pytest.raises(CanaryFailure, match="freshness contract"):
        _verify_current7_api(payload, identity)


def test_release_canary_refreshes_mutable_current7_snapshot_identity(tmp_path):
    index_path, status_path = _current7_index_fixture(tmp_path)
    configured = _authenticate_current7_index(index_path)

    _advance_current7_snapshot(index_path, status_path, "7" * 64)
    latest = _refresh_current7_index(configured)

    assert latest["bundle_manifest_sha256"] == configured["bundle_manifest_sha256"]
    assert (
        latest["constraint_identity_sha256"] == configured["constraint_identity_sha256"]
    )
    assert latest["snapshot_sha256"] == "7" * 64
    assert latest["index_file_sha256"] != configured["index_file_sha256"]


def test_release_canary_enforces_active_current7_freshness_and_terminal_exemption(
    tmp_path,
):
    index_path, _ = _current7_index_fixture(tmp_path)
    identity = _authenticate_current7_index(index_path)
    stale_active = _current7_api_payload(index_path)
    stale_active["tier1_current7_search"].update(
        {
            "freshness_ok": False,
            "freshness_age_seconds": 301.0,
            "authority_eligible": False,
        }
    )
    with pytest.raises(CanaryFailure, match="freshness contract"):
        _verify_current7_api(stale_active, identity)

    sealed = _current7_api_payload(index_path)
    search = sealed["tier1_current7_search"]
    search.update(
        {
            "running_count": 0,
            "queued_count": 0,
            "active_plus_queued": 0,
            "completed_count": 2,
            "authenticated_terminal_seed_count": 2,
            "state_counts": {"completed": 2},
            "freshness_required": False,
            "freshness_ok": True,
            "freshness_age_seconds": 7_200.0,
        }
    )

    verified = _verify_current7_api(sealed, identity)
    assert verified["freshness_required"] is False
    assert verified["freshness_ok"] is True


def test_release_canary_uses_current7_as_authority_with_stale_legacy_archive(
    tmp_path,
):
    index_path, _ = _current7_index_fixture(tmp_path)
    identity = _authenticate_current7_index(index_path)
    payload = _current7_api_payload(index_path)
    payload["tier1_feedback_search"]["running_count"] = "stale-not-a-count"

    nsga = _verify_nsga_api(
        payload,
        expected_current7_index=identity,
    )
    progress = _verify_nsga_progress(_current7_progress_payload(payload), nsga)

    assert nsga["authority_kind"] == "current7"
    assert nsga["authority_identity_sha256"] == "8" * 64
    assert nsga["legacy_authority_role"] == "archive"
    assert nsga["terminal_results_verified"] == 0
    assert nsga["authority_eligible"] is True
    assert nsga["constraints"] == CURRENT7_HARD_SPEC
    assert nsga["temperature_targets"] == list(CURRENT7_TEMPERATURE_TARGETS)
    assert nsga["model_manifest_sha256"] == "8" * 64
    assert nsga["deployment_model_manifest_sha256"] == "8" * 64
    assert progress["authority_identity_sha256"] == "8" * 64


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("healthy", False, "failed closed"),
        ("index_file_sha256", "0" * 64, "configured identity"),
        ("snapshot_sha256", "0" * 64, "configured identity"),
        ("constraint_contract_verified", False, "hard-spec identity"),
        ("authority_eligible", False, "hard-spec identity"),
    ),
)
def test_release_canary_rejects_explicit_current7_projection_drift(
    tmp_path, field, value, message
):
    index_path, _ = _current7_index_fixture(tmp_path)
    identity = _authenticate_current7_index(index_path)
    payload = _current7_api_payload(index_path)
    payload["tier1_current7_search"][field] = value

    with pytest.raises(CanaryFailure, match=message):
        _verify_nsga_api(payload, expected_current7_index=identity)


def test_release_canary_rejects_missing_current7_or_unmarked_legacy(tmp_path):
    index_path, _ = _current7_index_fixture(tmp_path)
    identity = _authenticate_current7_index(index_path)
    payload = _current7_api_payload(index_path)
    payload.pop("tier1_current7_search")
    with pytest.raises(CanaryFailure, match="projection is missing"):
        _verify_nsga_api(payload, expected_current7_index=identity)

    payload = _current7_api_payload(index_path)
    payload["tier1_feedback_search"]["archived"] = False
    with pytest.raises(CanaryFailure, match="not marked archive"):
        _verify_nsga_api(payload, expected_current7_index=identity)

    payload = _current7_api_payload(index_path)
    payload["search_authority"]["kind"] = "legacy"
    with pytest.raises(CanaryFailure, match="search authority diverged"):
        _verify_nsga_api(payload, expected_current7_index=identity)


def test_release_canary_rejects_resealed_current7_constraint_drift(tmp_path):
    index_path, status_path = _current7_index_fixture(tmp_path)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["hard_spec"]["T_limit_C"] = 109.0
    status_sha = _write_canonical_json(status_path, status)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["status"]["sha256"] = status_sha
    _write_canonical_json(index_path, index)

    with pytest.raises(CanaryFailure, match="status constraint identity"):
        _authenticate_current7_index(index_path)


def test_release_canary_rejects_current7_nested_sha_tamper(tmp_path):
    index_path, status_path = _current7_index_fixture(tmp_path)
    status_path.write_text('{"tampered":true}\n', encoding="utf-8")

    with pytest.raises(CanaryFailure, match="status SHA-256 mismatch"):
        _authenticate_current7_index(index_path)


def test_release_canary_authenticates_blocker_hpo_v2_status_and_api(tmp_path):
    status_path = _hpo_v2_status_fixture(tmp_path)
    identity = _authenticate_blocker_hpo_v2_status(status_path)
    dashboard = {
        "continuous_pipeline": {
            "blocker_hpo_v2": {
                "available": True,
                "validated_running": True,
                "state": "optimizing",
                "source": str(status_path),
                "config_sha256": "a" * 64,
                "dataset_sha256": "b" * 64,
                "job_count": 2,
                "stage_trials_per_job": 20,
                "trial_total": 40,
                "trial_complete": 12,
                "trial_running": 2,
                "trial_failed": 1,
            },
        },
    }

    verified = _verify_blocker_hpo_v2_api(dashboard, identity)

    assert identity["authority_flags_verified_false"] is True
    assert verified["validated_running"] is True
    assert verified["trial_total"] == 40


def test_release_canary_reauthenticates_live_hpo_stage_without_identity_drift(
    tmp_path, monkeypatch
):
    status_path = _hpo_v2_status_fixture(tmp_path)
    configured = _authenticate_blocker_hpo_v2_status(status_path)
    first_dashboard = _hpo_v2_dashboard(configured)
    calls = 0

    def getter(base_url, endpoint, timeout):
        nonlocal calls
        assert base_url == "http://candidate.test"
        assert endpoint == "/api/dashboard"
        calls += 1
        if calls == 1:
            _advance_hpo_v2_status(status_path)
            return first_dashboard
        return _hpo_v2_dashboard(_refresh_blocker_hpo_v2_status(configured))

    monkeypatch.setattr(release_canary_module, "_get_json", getter)
    _, verified, latest = _wait_for_blocker_hpo_v2_api(
        "http://candidate.test",
        configured,
        timeout_seconds=1,
        poll_seconds=0,
    )

    assert verified["stability_attempts"] == 2
    assert verified["stage_trials_per_job"] == 50
    assert verified["trial_complete"] == 38
    assert latest["job_identity_sha256"] == configured["job_identity_sha256"]


def test_release_canary_rejects_hpo_v2_authority_flag(tmp_path):
    status_path = _hpo_v2_status_fixture(tmp_path)
    value = json.loads(status_path.read_text(encoding="utf-8"))
    value["promotion_approved"] = True
    _write_canonical_json(status_path, value)

    with pytest.raises(CanaryFailure, match="forbidden authority"):
        _authenticate_blocker_hpo_v2_status(status_path)


def test_release_canary_accepts_bounded_candidate_preview_and_rejects_drift():
    payload = _nsga_payload()
    search = payload["tier1_feedback_search"]
    search.update(
        {
            "feasible_pareto_count": 151,
            "candidate_preview_count": 128,
            "candidate_preview_limit": 128,
            "candidate_preview_truncated": True,
        }
    )
    verified = _verify_nsga_api(payload)
    assert verified["feasible_pareto_count"] == 151
    assert verified["candidate_preview_count"] == 128
    assert verified["candidate_preview_truncated"] is True
    progress = _progress_payload(payload)
    assert _verify_nsga_progress(progress, verified)["candidate_preview_count"] == 128

    for field, value in (
        ("candidate_preview_count", 127),
        ("candidate_preview_limit", 127),
        ("candidate_preview_truncated", False),
    ):
        drifted = _nsga_payload()
        drifted["tier1_feedback_search"].update(
            {
                "feasible_pareto_count": 151,
                "candidate_preview_count": 128,
                "candidate_preview_limit": 128,
                "candidate_preview_truncated": True,
                field: value,
            }
        )
        with pytest.raises(CanaryFailure, match="candidate-preview"):
            _verify_nsga_api(drifted)


def test_release_canary_authenticates_running_dual_standard_fea_overlay(
    tmp_path,
):
    expected = _dual_expected(tmp_path)
    payload = _nsga_payload()
    payload["sealed_successors"] = _dual_payload(expected)

    verified = _verify_nsga_api(payload, expected_dual_validation=expected)[
        "dual_standard_fea_validation"
    ]

    assert verified["sealed_authority_flags_verified_false"] is True
    assert verified["task_ids"] == [53840, 53841]
    assert {item["task_status"] for item in verified["tasks"]} == {"running"}
    assert all(item["valid"] is False for item in verified["tasks"])
    assert verified["runtime_manifest_sha256"] == "e" * 64
    assert verified["execution_plan_sha256"] == "f" * 64
    assert verified["sealed_plan_source_sha256"] == "1" * 64


def test_release_canary_accepts_only_completed_plus_collected_valid_result(
    tmp_path,
):
    expected = _dual_expected(tmp_path)
    collection = _dual_payload(expected)
    candidate_sha, identity = next(
        iter(sorted(EXPECTED_DUAL_STANDARD_FEA_TASKS.items()))
    )
    overlay = collection["standard_fea_validation"]["candidates"][candidate_sha]
    result_root = Path(expected["standard_fea_validation"]["runtime_root"]) / "results"
    result_root.mkdir()
    result_path = result_root / f"task-{identity['task_id']}.json"
    result_path.write_text('{"fixture":"authenticated"}\n', encoding="utf-8")
    result_sha = hashlib.sha256(result_path.read_bytes()).hexdigest()
    overlay.update(
        {
            "task_status": "completed",
            "collection_state": "collector_succeeded",
            "collected": True,
            "valid": True,
            "actual_gates": _actual_t120_gate(),
            "actual_hard_gates_pass": True,
            "full_model_validation_candidate_eligible": True,
        }
    )
    overlay["task"]["status"] = "completed"
    overlay["result_identity"].update(
        {
            "available": True,
            "path": str(result_path),
            "sha256": result_sha,
            "contract_valid": True,
            "candidate_identity_matches": True,
        }
    )

    verified = _verify_dual_standard_fea({"sealed_successors": collection}, expected)
    completed = next(
        item for item in verified["tasks"] if item["task_id"] == identity["task_id"]
    )
    assert completed["valid"] is True
    assert completed["actual_t120"]["pass"] is True
    assert completed["result_sha256"] == result_sha


@pytest.mark.parametrize(
    ("drift", "match"),
    (
        ("collection_authority", "attempted authority"),
        ("candidate_authority", "attempted authority"),
        ("task_identity", "task identity/status drifted"),
        ("runtime_identity", "runtime/plan/source identity drifted"),
        ("overlay_rejected", "overlay failed closed"),
        ("active_collector", "active task exposes a terminal result"),
    ),
)
def test_release_canary_rejects_dual_authority_identity_and_lifecycle_drift(
    tmp_path,
    drift,
    match,
):
    expected = _dual_expected(tmp_path)
    collection = _dual_payload(expected)
    candidate_sha = sorted(EXPECTED_DUAL_STANDARD_FEA_TASKS)[0]
    candidate = next(
        item
        for item in collection["candidates"]
        if item["candidate"]["candidate_sha256"] == candidate_sha
    )
    card_overlay = candidate["standard_fea_validation"]
    top_overlay = collection["standard_fea_validation"]["candidates"][candidate_sha]
    if drift == "collection_authority":
        collection["production"] = True
    elif drift == "candidate_authority":
        candidate["fea"] = True
    elif drift == "task_identity":
        card_overlay["task"]["id"] = 99999
        top_overlay["task"]["id"] = 99999
    elif drift == "runtime_identity":
        collection["standard_fea_validation"]["integrity"]["manifest_sha256"] = "9" * 64
    elif drift == "overlay_rejected":
        collection["standard_fea_validation"]["available"] = False
    elif drift == "active_collector":
        card_overlay["collection_state"] = "collector_succeeded"
        card_overlay["collected"] = True
        top_overlay["collection_state"] = "collector_succeeded"
        top_overlay["collected"] = True
    with pytest.raises(CanaryFailure, match=match):
        _verify_dual_standard_fea({"sealed_successors": collection}, expected)


def test_release_canary_recomputes_projected_actual_t120_gate(tmp_path):
    expected = _dual_expected(tmp_path)
    collection = _dual_payload(expected)
    candidate_sha, identity = next(
        iter(sorted(EXPECTED_DUAL_STANDARD_FEA_TASKS.items()))
    )
    overlay = collection["standard_fea_validation"]["candidates"][candidate_sha]
    result_root = Path(expected["standard_fea_validation"]["runtime_root"]) / "results"
    result_root.mkdir()
    result_path = result_root / f"task-{identity['task_id']}.json"
    result_path.write_bytes(b"{}\n")
    overlay.update(
        {
            "task_status": "completed",
            "collection_state": "collector_succeeded",
            "collected": True,
            "valid": True,
            "actual_gates": _actual_t120_gate(),
            "actual_hard_gates_pass": False,
        }
    )
    overlay["task"]["status"] = "completed"
    overlay["result_identity"].update(
        {
            "available": True,
            "path": str(result_path),
            "sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            "contract_valid": True,
            "candidate_identity_matches": True,
        }
    )
    with pytest.raises(CanaryFailure, match="actual gate projection drifted"):
        _verify_dual_standard_fea({"sealed_successors": collection}, expected)


def test_dual_release_configuration_hashes_every_cli_input(tmp_path):
    allowed = (tmp_path / "allowed").resolve()
    runtime = allowed / "runtime"
    (runtime / "sealed-plan-source").mkdir(parents=True)

    def write(relative, content=b"{}\n"):
        path = (tmp_path / relative).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path, hashlib.sha256(content).hexdigest()

    handoff, handoff_sha = write("sealed/handoff.json")
    plan, plan_sha = write("sealed/dual-plan.json")
    ui, ui_sha = write("sealed/dual-ui.json")
    receipt, receipt_sha = write("sealed/receipt.json")
    execution, execution_sha = write("sealed/execution.json")
    manifest, manifest_sha = write("allowed/runtime/manifest.json")
    source, source_sha = write(
        "allowed/runtime/sealed-plan-source/experimental_source.json"
    )
    args = SimpleNamespace(
        sealed_successor_handoff=handoff,
        sealed_successor_handoff_sha256=handoff_sha,
        dual_successor_plan=plan,
        dual_successor_plan_sha256=plan_sha,
        dual_successor_ui=ui,
        dual_successor_ui_sha256=ui_sha,
        dual_successor_receipt=receipt,
        dual_successor_receipt_sha256=receipt_sha,
        dual_fea_runtime_root=runtime,
        dual_fea_allowed_root=allowed,
        dual_fea_manifest_sha256=manifest_sha,
        dual_fea_plan=execution,
        dual_fea_plan_sha256=execution_sha,
        dual_fea_source_sha256=source_sha,
    )
    configuration = _dual_release_configuration(args)
    validation = configuration["standard_fea_validation"]
    assert validation["runtime_manifest"]["path"] == str(manifest)
    assert validation["execution_plan"]["sha256"] == execution_sha
    assert validation["sealed_plan_source"]["path"] == str(source)
    assert [item["task_id"] for item in validation["expected_tasks"]] == [
        53841,
        53840,
    ]

    args.dual_fea_manifest_sha256 = "0" * 64
    with pytest.raises(CanaryFailure, match="runtime manifest SHA-256 mismatch"):
        _dual_release_configuration(args)


def test_nsga_switch_gate_waits_for_two_reads_of_one_fresh_generation(monkeypatch):
    stale_nsga = json.loads(json.dumps(_nsga_payload()))
    stale_nsga["tier1_feedback_search"]["available"] = False
    stale_nsga["tier1_feedback_search"]["integrity_verified"] = False
    stale_progress = _progress_payload(stale_nsga)

    generation_one = _nsga_payload()
    generation_two = json.loads(json.dumps(_nsga_payload()))
    generation_two["tier1_feedback_search"]["updated_at"] = "2026-07-18T15:00:10+09:00"
    progress_two = _progress_payload(generation_two)
    pairs = [
        (stale_nsga, stale_progress),
        (generation_one, progress_two),
        (generation_two, progress_two),
        (generation_two, progress_two),
    ]
    attempt = 0

    def getter(base_url, endpoint, timeout):
        nonlocal attempt
        assert base_url == "http://candidate.test"
        pair = pairs[min(attempt, len(pairs) - 1)]
        if endpoint == "/api/nsga2":
            return pair[0]
        assert endpoint == "/api/nsga2/progress"
        attempt += 1
        return pair[1]

    monkeypatch.setattr(release_canary_module, "_get_json", getter)
    _, _, nsga, progress, gate = _wait_for_stable_nsga_generation(
        "http://candidate.test", timeout_seconds=1, poll_seconds=0
    )
    assert gate["verified"] is True
    assert gate["attempts"] == 4
    assert gate["consecutive_reads"] == 2
    assert nsga["updated_at"] == progress["updated_at"]
    assert gate["generation_identity"]["updated_at"].endswith("15:00:10+09:00")


def test_nsga_switch_gate_stabilizes_on_current7_without_fresh_legacy(
    tmp_path, monkeypatch
):
    index_path, _ = _current7_index_fixture(tmp_path)
    identity = _authenticate_current7_index(index_path)
    nsga_payload = _current7_api_payload(index_path)
    progress_payload = _current7_progress_payload(nsga_payload)
    nsga_payload["tier1_feedback_search"].update(
        {
            "available": False,
            "integrity_verified": False,
            "updated_at": "2020-01-01T00:00:00+00:00",
        }
    )

    def getter(base_url, endpoint, timeout):
        assert base_url == "http://candidate.test"
        return nsga_payload if endpoint == "/api/nsga2" else progress_payload

    monkeypatch.setattr(release_canary_module, "_get_json", getter)
    _, _, nsga, progress, gate = _wait_for_stable_nsga_generation(
        "http://candidate.test",
        expected_current7_index=identity,
        timeout_seconds=1,
        poll_seconds=0,
    )

    assert gate["verified"] is True
    assert gate["attempts"] == 2
    assert nsga["authority_kind"] == progress["authority_kind"] == "current7"
    assert gate["generation_identity"]["snapshot_sha256"] == "9" * 64


def test_nsga_switch_gate_reauthenticates_a_new_current7_snapshot(
    tmp_path, monkeypatch
):
    index_path, status_path = _current7_index_fixture(tmp_path)
    configured = _authenticate_current7_index(index_path)
    _advance_current7_snapshot(index_path, status_path, "7" * 64)
    nsga_payload = _current7_api_payload(index_path)
    progress_payload = _current7_progress_payload(nsga_payload)

    def getter(base_url, endpoint, timeout):
        return nsga_payload if endpoint == "/api/nsga2" else progress_payload

    monkeypatch.setattr(release_canary_module, "_get_json", getter)
    _, _, _, _, gate = _wait_for_stable_nsga_generation(
        "http://candidate.test",
        expected_current7_index=configured,
        timeout_seconds=1,
        poll_seconds=0,
    )

    assert gate["attempts"] == 2
    assert gate["current7_index"]["snapshot_sha256"] == "7" * 64


def test_release_canary_rejects_current7_progress_authority_tamper(tmp_path):
    index_path, _ = _current7_index_fixture(tmp_path)
    identity = _authenticate_current7_index(index_path)
    payload = _current7_api_payload(index_path)
    nsga = _verify_nsga_api(payload, expected_current7_index=identity)
    progress = _current7_progress_payload(payload)
    progress["authority_identity_sha256"] = "0" * 64

    with pytest.raises(CanaryFailure, match="authority identity"):
        _verify_nsga_progress(progress, nsga)


def test_nsga_switch_gate_times_out_fail_closed_on_stale_status(monkeypatch):
    stale_nsga = _nsga_payload()
    stale_nsga["tier1_feedback_search"]["available"] = False
    stale_nsga["tier1_feedback_search"]["integrity_verified"] = False
    stale_progress = _progress_payload(stale_nsga)

    def getter(base_url, endpoint, timeout):
        return stale_nsga if endpoint == "/api/nsga2" else stale_progress

    monkeypatch.setattr(release_canary_module, "_get_json", getter)
    with pytest.raises(CanaryFailure, match="stability gate timed out"):
        _wait_for_stable_nsga_generation(
            "http://candidate.test", timeout_seconds=0, poll_seconds=0
        )


def test_release_canary_rejects_tier1_hard_spec_or_thermal_drift():
    payload = _nsga_payload()
    payload["tier1_feedback_search"]["constraints"]["resonance_min_Hz"] = 20_000.0
    with pytest.raises(CanaryFailure, match="T120/res10k"):
        _verify_nsga_api(payload)

    payload = _nsga_payload()
    payload["tier1_feedback_search"]["temperature_constraint_contract"]["targets"] = (
        list(EXPECTED_TIER1_TEMPERATURE_TARGETS[:-1])
    )
    with pytest.raises(CanaryFailure, match="11-target thermal"):
        _verify_nsga_api(payload)


def test_release_canary_rejects_mojibake_display_text():
    assert _require_utf8_display_text(
        "활성 코호트 1234567890", "active label"
    ).startswith("활성")
    with pytest.raises(CanaryFailure, match="probable mojibake"):
        _require_utf8_display_text("理쒖쟻?ㅺ퀎", "active label")


def test_release_canary_verifies_local_gui_identity_and_capacity():
    payload = {
        "schema_version": 1,
        "available": True,
        "backend": "standalone",
        "result_schema": GUI_RESULT_SCHEMA,
        "solver_identity": {
            "verified": True,
            "revision": DEFAULT_SOLVER_REVISION,
            "runner_sha256": DEFAULT_RUNNER_SHA256,
            "root": "C:/immutable/solver",
        },
        "max_active": 2,
        "active_count": 1,
        "retained_aedt_count": 1,
        "occupied_count": 2,
        "launches": [],
    }
    result = _verify_local_gui_launches(payload)
    assert result["solver_revision"] == DEFAULT_SOLVER_REVISION
    assert result["occupied_count"] == 2


def test_release_canary_verifies_routed_deadline_gui_identity():
    payload = {
        "schema_version": 1,
        "available": True,
        "backend": "standalone",
        "routed": True,
        "result_schema": GUI_RESULT_SCHEMA,
        "solver_identities": [
            {
                "verified": True,
                "revision": DEFAULT_SOLVER_REVISION,
                "runner_sha256": DEFAULT_RUNNER_SHA256,
                "root": "C:/immutable/baseline",
            },
            {
                "verified": True,
                "revision": EXPECTED_LOCAL_GUI_SOLVER["solver_revision"],
                "branch": EXPECTED_LOCAL_GUI_SOLVER["solver_branch"],
                "runner_sha256": DEADLINE_LOCAL_GUI_RUNNER_SHA256,
                "source_runner_sha256": EXPECTED_LOCAL_GUI_SOLVER[
                    "solver_source_runner_sha256"
                ],
                "thermal_module_sha256": EXPECTED_LOCAL_GUI_SOLVER[
                    "solver_thermal_module_sha256"
                ],
                "library_revision": EXPECTED_LOCAL_GUI_SOLVER[
                    "library_revision"
                ],
                "library_dirty": False,
                "root": "C:/immutable/deadline",
            },
        ],
        "max_active": 4,
        "active_count": 0,
        "retained_aedt_count": 0,
        "occupied_count": 0,
        "launches": [],
    }

    result = _verify_local_gui_launches(payload)

    assert result["routed"] is True
    assert result["deadline_solver_revision"] == (
        EXPECTED_LOCAL_GUI_SOLVER["solver_revision"]
    )


def test_release_canary_evidence_is_create_only(tmp_path):
    path = tmp_path / "evidence.json"
    sha = _write_new_json(path, {"passed": False})
    assert len(sha) == 64
    assert json.loads(path.read_text(encoding="utf-8")) == {"passed": False}
    with pytest.raises(CanaryFailure, match="refusing to overwrite"):
        _write_new_json(path, {"passed": True})


def test_runtime_transition_authenticates_seal_and_process_paths(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    venv = tmp_path / "venv"
    executable = venv / "Scripts" / "python.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"python fixture")
    process = {
        "executable": str(executable),
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "cwd": str(source),
        "command": [str(executable), "-m", "uvicorn"],
    }
    _require_process_location(process, source, venv)

    path = tmp_path / "seal.json"
    sha = _write_new_json(path, {"kind": "fixture-seal", "pid": 1})
    _write_sidecar(path, sha)
    value, authenticated_sha = _read_sealed_json(path, "fixture-seal")
    assert value["pid"] == 1
    assert authenticated_sha == sha

    path.write_text('{"kind":"fixture-seal","pid":2}\n', encoding="utf-8")
    with pytest.raises(CanaryFailure, match="sidecar mismatch"):
        _read_sealed_json(path, "fixture-seal")


def test_runtime_transition_binds_optional_artifact_environment(tmp_path):
    current7_path, _ = _current7_index_fixture(tmp_path)
    hpo_path = _hpo_v2_status_fixture(tmp_path)
    current7 = _authenticate_current7_index(current7_path)
    hpo = _authenticate_blocker_hpo_v2_status(hpo_path)
    manifest = {
        "current7_index": current7,
        "blocker_hpo_v2_status": hpo,
    }
    environment = {
        "MFT_TIER1_CURRENT7_INDEX": str(current7_path),
        "MFT_BLOCKER_HPO_V2_STATUS": str(hpo_path),
    }
    # A live heartbeat is mutable.  Preserve its sealed path/contract but prove
    # that post-transition verification records, rather than rejects, a new
    # byte identity.
    hpo_value = json.loads(hpo_path.read_text(encoding="utf-8"))
    hpo_path.write_text(
        json.dumps(hpo_value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    verified = _verify_optional_runtime_artifacts(environment, manifest)

    assert verified["current7_index"]["nested_sha256_verified"] is True
    assert verified["blocker_hpo_v2_status"]["authority_flags_verified_false"] is True
    assert verified["blocker_hpo_v2_status"]["sha256"] != hpo["sha256"]


def test_runtime_transition_binds_deadline_publication_path_and_hash(
    tmp_path, monkeypatch
):
    publication = tmp_path / "deadline-design-publication.json"
    publication.write_text('{"fixture":true}\n', encoding="utf-8")
    publication_sha = hashlib.sha256(publication.read_bytes()).hexdigest()
    authenticated = {
        "publication": {"full_actual_hard_pass": False},
        "generation": {"id": "deadline-generation"},
        "candidate": {
            "id": "deadline-candidate",
            "cooling_variant": "timk3",
            "fan_config": "dual",
            "fan_velocity_m_s": 1.5,
            "solver_revision": EXPECTED_LOCAL_GUI_SOLVER[
                "solver_revision"
            ],
            "thermal_pad_material_policy": (
                deadline_design_module.EXPECTED_TIM["material_policy"]
            ),
            "thermal_pad_native_readback_contract_version": (
                deadline_design_module.EXPECTED_TIM[
                    "native_readback_contract_version"
                ]
            ),
            "thermal_pad_native_readback_attested": 1,
            "thermal_pad_native_thermal_conductivity_W_mK": 3.0,
            "thermal_pad_native_electrical_conductivity_S_m": 0.0,
            "pred_f_res_min_screen_Hz": 16_000.0,
            "resonance_minimum_required_Hz": 15_000.0,
            "resonance_margin_Hz": 1_000.0,
            "gui_build_eligible": True,
            "gui_solve_eligible": False,
        },
    }
    monkeypatch.setattr(
        deadline_design_module,
        "load_publication",
        lambda path, sha256: (
            authenticated
            if Path(path).resolve() == publication.resolve()
            and sha256 == publication_sha
            else pytest.fail("deadline publication identity drifted")
        ),
    )
    declared = {
        "verified": True,
        "publication_path": str(publication.resolve()),
        "publication_sha256": publication_sha,
        "generation_id": "deadline-generation",
        "candidate_id": "deadline-candidate",
        "cooling_variant": "timk3",
        "fan_config": "dual",
        "fan_velocity_m_s": 1.5,
        "solver_revision": EXPECTED_LOCAL_GUI_SOLVER[
            "solver_revision"
        ],
        "thermal_pad_material_policy": (
            deadline_design_module.EXPECTED_TIM["material_policy"]
        ),
        "thermal_pad_native_readback_contract_version": (
            deadline_design_module.EXPECTED_TIM[
                "native_readback_contract_version"
            ]
        ),
        "thermal_pad_native_readback_attested": 1,
        "thermal_pad_native_thermal_conductivity_W_mK": 3.0,
        "thermal_pad_native_electrical_conductivity_S_m": 0.0,
        "half_magnetizing_resonance_Hz": 16_000.0,
        "half_magnetizing_resonance_minimum_Hz": 15_000.0,
        "half_magnetizing_resonance_margin_Hz": 1_000.0,
        "full_actual_hard_pass": False,
        "gui_build_eligible": True,
        "gui_solve_eligible": False,
    }
    environment = {
        "MFT_DEADLINE_DESIGN_PUBLICATION": str(publication),
        "MFT_DEADLINE_DESIGN_PUBLICATION_SHA256": publication_sha,
    }

    verified = _verify_optional_runtime_artifacts(
        environment, {"deadline_design": declared}
    )

    assert verified["deadline_design"] == declared
    with pytest.raises(
        CanaryFailure, match="hash differs from the release manifest"
    ):
        _verify_optional_runtime_artifacts(
            {
                **environment,
                "MFT_DEADLINE_DESIGN_PUBLICATION_SHA256": "0" * 64,
            },
            {"deadline_design": declared},
        )
    with pytest.raises(CanaryFailure, match="was not declared"):
        _verify_optional_runtime_artifacts(environment, {})


def test_runtime_transition_preserves_absent_optional_artifacts(tmp_path):
    assert _verify_optional_runtime_artifacts({}, {}) == {
        "current7_index": None,
        "blocker_hpo_v2_status": None,
        "current7_condition_indexes": [],
        "deadline_design": None,
    }

    current7_path, _ = _current7_index_fixture(tmp_path)
    with pytest.raises(CanaryFailure, match="was not declared"):
        _verify_optional_runtime_artifacts(
            {"MFT_TIER1_CURRENT7_INDEX": str(current7_path)}, {}
        )


def test_runtime_transition_allows_only_sealed_pyvenv_base_pair(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    venv = tmp_path / "venv"
    launcher = venv / "Scripts" / "python.exe"
    launcher.parent.mkdir(parents=True)
    launcher.write_bytes(b"venv launcher")
    base_home = tmp_path / "base"
    base_home.mkdir()
    base = base_home / "python.exe"
    base.write_bytes(b"base interpreter")
    pyvenv = venv / "pyvenv.cfg"
    pyvenv.write_text(
        f"home = {base_home}\nexecutable = {base}\n",
        encoding="utf-8",
    )

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    identity = {
        "pyvenv": {
            "path": str(pyvenv),
            "sha256": digest(pyvenv),
            "home": str(base_home),
            "executable": str(base),
        },
        "venv_launcher": {
            "path": str(launcher),
            "sha256": digest(launcher),
        },
        "base_interpreter": {
            "path": str(base),
            "sha256": digest(base),
        },
        "observed_process_executable": str(base),
    }
    process = {
        "executable": str(base),
        "executable_sha256": digest(base),
        "cwd": str(source),
        "command": [str(base), "-m", "uvicorn"],
    }
    _require_process_location(process, source, venv, identity)

    other_base = tmp_path / "other-base" / "python.exe"
    other_base.parent.mkdir()
    other_base.write_bytes(b"other base")
    wrong_base_process = {
        **process,
        "executable": str(other_base),
        "executable_sha256": digest(other_base),
    }
    with pytest.raises(CanaryFailure, match="canary-observed|does not belong"):
        _require_process_location(wrong_base_process, source, venv, identity)

    base.write_bytes(b"tampered base interpreter")
    with pytest.raises(CanaryFailure, match="base interpreter hash mismatch"):
        _require_process_location(process, source, venv, identity)
    base.write_bytes(b"base interpreter")

    other_venv = tmp_path / "other-venv"
    (other_venv / "Scripts").mkdir(parents=True)
    (other_venv / "Scripts" / "python.exe").write_bytes(b"other launcher")
    (other_venv / "pyvenv.cfg").write_text(
        f"home = {base_home}\nexecutable = {base}\n",
        encoding="utf-8",
    )
    with pytest.raises(CanaryFailure, match="pyvenv.cfg"):
        _require_process_location(process, source, other_venv, identity)

    other_cwd = tmp_path / "other-cwd"
    other_cwd.mkdir()
    with pytest.raises(CanaryFailure, match="source root"):
        _require_process_location(
            {**process, "cwd": str(other_cwd)}, source, venv, identity
        )
