import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from regression_260707.monitoring.readers import (
    ArtifactService,
    DUAL_SUCCESSOR_EXACT_SHA256,
    DUAL_SUCCESSOR_INTERIOR_SHA256,
    DUAL_SUCCESSOR_PLAN_ENV,
    DUAL_SUCCESSOR_PLAN_SHA_ENV,
    DUAL_SUCCESSOR_PLAN_SHA256,
    DUAL_SUCCESSOR_RECEIPT_ENV,
    DUAL_SUCCESSOR_RECEIPT_SHA_ENV,
    DUAL_SUCCESSOR_RELEASE_REVISION,
    DUAL_SUCCESSOR_RELEASE_SOURCE_SHA256,
    DUAL_SUCCESSOR_TEMPERATURE_CONTRACT_SHA256,
    DUAL_SUCCESSOR_UI_ENV,
    DUAL_SUCCESSOR_UI_SHA_ENV,
    DUAL_VALIDATION_ALLOWED_ROOT_ENV,
    DUAL_VALIDATION_CANDIDATE_DIGESTS,
    DUAL_VALIDATION_HANDOFF_SHA256,
    DUAL_VALIDATION_LIBRARY_REVISION,
    DUAL_VALIDATION_MANIFEST_SHA_ENV,
    DUAL_VALIDATION_PLAN_SHA_ENV,
    DUAL_VALIDATION_ROOT_ENV,
    DUAL_VALIDATION_SOLVER_REVISION,
    DUAL_VALIDATION_SOURCE_SHA_ENV,
    DUAL_VALIDATION_TEMPERATURE_CONTRACT,
    SEALED_SUCCESSOR_COHORT_ID,
    SEALED_SUCCESSOR_CONSTRAINT_VERSION,
    SEALED_SUCCESSOR_CONSTRAINTS,
    SEALED_SUCCESSOR_TEMPERATURE_TARGETS,
)


class _NoNsgaScheduler:
    def mft_pipeline_status(self):
        return {"available": False}


class _NoContinuousPipeline:
    def snapshot(self):
        return {"available": False, "warnings": []}


def _canonical_sha(value):
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write(path: Path, value) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _hard_spec():
    return {
        "B_limit_T": 1.2,
        "Llt_target_uH": 27.5,
        "Llt_tol_uH": 0.55,
        "T_limit_C": 120.0,
        "insulation_min_mm": 40.0,
        "magnetizing_inductance_factor": 0.5,
        "n_core_group_max": 4,
        "primary_conductor_thickness_mm": 5.0,
        "q_sigma": 1.0,
        "resonance_min_Hz": 10000.0,
        "size_H_max_mm": 750.0,
        "size_L_max_mm": 1200.0,
        "size_W_max_mm": 1200.0,
        "uncertainty_contract": "q90_conformal_half_width_physical_v1",
    }


def _candidate(artifact_sha: str, order: int) -> dict:
    params = {
        f"p{index:02d}": float(index) + order / 100
        for index in range(70)
    }
    params["N2_side"] = 21
    params_sha = _canonical_sha(params)
    identity = {
        "schema_version": "mft-t120-successor-fea-candidate-dedupe-v1",
        "cohort_id": SEALED_SUCCESSOR_COHORT_ID,
        "constraint_version": SEALED_SUCCESSOR_CONSTRAINT_VERSION,
        "candidate_artifact_sha256": artifact_sha,
        "decoded_fea_params_sha256": params_sha,
    }
    constraints = {name: -1.0 for name in SEALED_SUCCESSOR_CONSTRAINTS}
    constraints["Llt_robust_band"] = -0.04 - order / 1000
    temperatures = {
        name: 100.0 + position / 10
        for position, name in enumerate(SEALED_SUCCESSOR_TEMPERATURE_TARGETS)
    }
    return {
        "display_id": f"candidate-{order}",
        "display_label": "T120 sealed exact" if order == 1 else "T120 interior margin",
        "append_only_order": order,
        "artifact_sha256": artifact_sha,
        "replay_sha256": str(order) * 64,
        "decoded_fea_params": params,
        "decoded_fea_params_sha256": params_sha,
        "dedupe_identity_sha256": _canonical_sha(identity),
        "constraints_G": constraints,
        "robust_temperature_C": temperatures,
        "decoded_geometry": {
            "cw1_mm": 5.0,
            "n_core_group": 4,
            "N1_main": 6,
            "N2_side": 21,
        },
        "exterior_box_mm": [1100.0 + order, 1150.0, 700.0],
        "Llt_robust_G_uH": constraints["Llt_robust_band"],
        "worst_robust_temperature_C": max(temperatures.values()),
        "resonance_min_screen_Hz": 10200.0 + order,
        "total_positive_violation": 0.0,
        "source_kind": "sealed_successor_evidence",
        "production_eligible": False,
        "fea_submission_approved": False,
    }


def _dual_fixture(tmp_path: Path, *, candidate_authority=False):
    hard_spec = _hard_spec()
    candidates = [
        _candidate(DUAL_SUCCESSOR_EXACT_SHA256, 1),
        _candidate(DUAL_SUCCESSOR_INTERIOR_SHA256, 2),
    ]
    if candidate_authority:
        candidates[1]["production_eligible"] = True
    plan = {
        "schema_version": "mft-tier1-t120-dual-successor-validateonly-plan-v1",
        "mode": "ValidateOnly",
        "release": {
            "revision": DUAL_SUCCESSOR_RELEASE_REVISION,
            "git_dirty": False,
            "source": {"sha256": DUAL_SUCCESSOR_RELEASE_SOURCE_SHA256},
        },
        "cohort_id": SEALED_SUCCESSOR_COHORT_ID,
        "constraint_version": SEALED_SUCCESSOR_CONSTRAINT_VERSION,
        "hard_spec": hard_spec,
        "hard_spec_sha256": _canonical_sha(hard_spec),
        "temperature_constraint_contract": {
            "target_count": 11,
            "targets": list(SEALED_SUCCESSOR_TEMPERATURE_TARGETS),
        },
        "temperature_constraint_contract_sha256": (
            DUAL_SUCCESSOR_TEMPERATURE_CONTRACT_SHA256
        ),
        "source_model_manifest_sha256": "1" * 64,
        "deployment_model_manifest_sha256": "2" * 64,
        "evidence_references": {},
        "pointer_cutover": {
            "pointer_switch_verified": True,
            "cohort_id": SEALED_SUCCESSOR_COHORT_ID,
            "constraint_version": SEALED_SUCCESSOR_CONSTRAINT_VERSION,
        },
        "capacity_contract": {
            "global_lane_cap": 28,
            "observe_only_passed": True,
            "fresh_execution_time_recheck_required": True,
        },
        "candidate_count": 2,
        "maximum_submission_count": 2,
        "candidates": candidates,
        "submission_policy": {
            "validate_only": True,
            "pointer_cutover_verified": True,
            "maximum_submission_count": 2,
            "execution_ready": False,
            "operator_authorization_required": True,
            "fresh_cap28_inventory_recheck_required": True,
        },
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "scheduler_mutation_performed": False,
        "live_pointer_mutation_performed": False,
        "monitor_8010_mutation_performed": False,
        "scheduler_8002_mutation_performed": False,
    }
    plan_ref = _write(tmp_path / "dual_successor_validateonly_plan.json", plan)
    ui_candidates = [
        {
            key: candidate[key]
            for key in (
                "display_id",
                "artifact_sha256",
                "decoded_fea_params_sha256",
                "dedupe_identity_sha256",
            )
        }
        for candidate in candidates
    ]
    ui = {
        "schema_version": "mft-tier1-t120-dual-successor-readonly-ui-v1",
        "source_plan": plan_ref,
        "section": "sealed_successor_evidence",
        "read_only": True,
        "display_as_scheduler_terminal": False,
        "display_as_near_or_pareto": False,
        "candidate_count": 2,
        "candidates": ui_candidates,
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
    }
    ui_ref = _write(tmp_path / "dual_successor_readonly_ui.json", ui)
    receipt = {
        "schema_version": "mft-tier1-t120-dual-successor-validation-receipt-v1",
        "validated_at": "2026-07-18T10:12:00+00:00",
        "plan": plan_ref,
        "read_only_ui": ui_ref,
        "validation_passed": True,
        "pointer_cutover_verified": True,
        "candidate_count": 2,
        "maximum_submission_count": 2,
        "all22_all11_and_71_params_verified": True,
        "distinct_dedupe_verified": True,
        "live_mutation_performed": False,
        "fea_submission_performed": False,
    }
    receipt_ref = _write(tmp_path / "validation_receipt.json", receipt)
    return plan_ref, ui_ref, receipt_ref


def _service(tmp_path: Path) -> ArtifactService:
    return ArtifactService(
        tmp_path,
        scheduler=_NoNsgaScheduler(),
        continuous_pipeline=_NoContinuousPipeline(),
        record_runtime=False,
    )


def _configure(monkeypatch, refs):
    plan, ui, receipt = refs
    for path_env, sha_env, reference in (
        (DUAL_SUCCESSOR_PLAN_ENV, DUAL_SUCCESSOR_PLAN_SHA_ENV, plan),
        (DUAL_SUCCESSOR_UI_ENV, DUAL_SUCCESSOR_UI_SHA_ENV, ui),
        (
            DUAL_SUCCESSOR_RECEIPT_ENV,
            DUAL_SUCCESSOR_RECEIPT_SHA_ENV,
            receipt,
        ),
    ):
        monkeypatch.setenv(path_env, reference["path"])
        monkeypatch.setenv(sha_env, reference["sha256"])


def _validation_fixture(tmp_path: Path, refs, *, completed=False):
    root = tmp_path / "runtime" / "dual-validation"
    (root / "candidates").mkdir(parents=True)
    (root / "results").mkdir()
    (root / "sealed-plan-source").mkdir()
    execution_plan_sha = "e" * 64
    source_ref = _write(
        root / "sealed-plan-source" / "experimental_source.json",
        {
            "schema_version": (
                "mft-tier1-t120-active-learning-source-snapshot-v2"
            ),
            "plan_sha256": execution_plan_sha,
            "production_eligible": False,
        },
    )
    manifest = {
        "schema_version": "mft-continuous-fea-validation-v1",
        "backend": "standalone",
        "controller_role": "NSGA candidate standard-FEA validation",
        "task_prefix": "mft-nsgafea",
        "lane_cap": 28,
        "priority": 5,
        "submission_enabled": True,
        "production_gate_bypass": False,
        "scheduler_orphan_adoption_enabled": False,
        "solver_revision": DUAL_VALIDATION_SOLVER_REVISION,
        "library_revision": DUAL_VALIDATION_LIBRARY_REVISION,
        "experimental_source": source_ref["path"],
    }
    manifest_ref = _write(root / "manifest.json", manifest)
    sealed_plan = json.loads(Path(refs[0]["path"]).read_text(encoding="utf-8"))
    records = {}
    active_tasks = []
    for index, candidate in enumerate(sealed_plan["candidates"]):
        artifact_sha = candidate["artifact_sha256"]
        digest = DUAL_VALIDATION_CANDIDATE_DIGESTS[artifact_sha]
        task_id = 9000 + index
        task_name = f"mft-nsgafea-x-{digest[:16]}"
        task_status = "completed" if completed else "running"
        record = {
            "candidate_digest": digest,
            "adapter_rank": index,
            "append_only_order": index + 1,
            "decoded_params_sha256": candidate[
                "decoded_fea_params_sha256"
            ],
            "source_candidate_sha256": artifact_sha,
            "dual_dedupe_identity_sha256": candidate[
                "dedupe_identity_sha256"
            ],
            "sealed_plan_sha256": execution_plan_sha,
            "source_artifact_sha256": DUAL_SUCCESSOR_PLAN_SHA256,
            "source_result": {
                "path": refs[0]["path"],
                "bytes": refs[0]["bytes"],
                "sha256": DUAL_SUCCESSOR_PLAN_SHA256,
            },
            "source_record": {
                "path": str(tmp_path / "handoff.json"),
                "bytes": 1,
                "sha256": DUAL_VALIDATION_HANDOFF_SHA256,
            },
            "source_kind": "sealed_successor_evidence",
            "active_learning_only": True,
            "approval_label": "EXPERIMENTAL / NOT-APPROVED",
            "model_lane": "experimental",
            "production_eligible": False,
            "production_eligible_model": False,
            "pareto_eligible": False,
            "fea_submission_approved": False,
            "solver_revision": DUAL_VALIDATION_SOLVER_REVISION,
            "library_revision": DUAL_VALIDATION_LIBRARY_REVISION,
            "params": candidate["decoded_fea_params"],
            "submission_state": "submitted",
            "task_id": task_id,
            "task_name": task_name,
            "task_status": task_status,
            "collection_state": (
                "collector_succeeded" if completed else "not_terminal"
            ),
            "submitted_at": "2026-07-18T19:46:23+09:00",
        }
        if completed:
            actual_gate = {
                "contract": (
                    "mft-tier1-actual-Llt-and-conditional-all11-T120-v1"
                ),
                "Llt_target_uH": 27.5,
                "Llt_tolerance_uH": 0.55,
                "temperature_limit_C": 120.0,
                "temperature_contract": (
                    DUAL_VALIDATION_TEMPERATURE_CONTRACT
                ),
                "pass": True,
                "reasons": [],
                "actual": {
                    "Llt_full_uH": 27.5,
                    "N2_side": 21,
                    "temperatures": {
                        name: {"applicable": True, "value_C": 100.0}
                        for name in SEALED_SUCCESSOR_TEMPERATURE_TARGETS
                    },
                },
            }
            result = {
                "schema_version": "mft-continuous-fea-validation-v1",
                "candidate_digest": digest,
                "task_id": task_id,
                "approval_label": "EXPERIMENTAL / NOT-APPROVED",
                "candidate_identity_matches": True,
                "result_contract_valid": True,
                "final_design_approved": False,
                "pareto_eligible": False,
                "production_eligible": False,
                "tier1_t120_actual_gate": actual_gate,
                "result": {
                    "Llt": 13.75,
                    "full_model": 0,
                    "N2_side": 21,
                    "P_core_total": 2000.0,
                    "P_winding_total": 3000.0,
                    "B_design_square_material_analytic": 0.8,
                    **{
                        name: 100.0
                        for name in SEALED_SUCCESSOR_TEMPERATURE_TARGETS
                    },
                },
            }
            result_ref = _write(root / "results" / f"task-{task_id}.json", result)
            record.update({
                "result_path": result_ref["path"],
                "result_contract_valid": True,
                "candidate_identity_matches": True,
                "tier1_t120_actual_gate": actual_gate,
                "full_model_validation_candidate_eligible": True,
                "terminal_at": "2026-07-18T22:00:00+09:00",
            })
        else:
            active_tasks.append({
                "candidate_digest": digest,
                "task_id": task_id,
                "task_name": task_name,
                "status": task_status,
            })
        records[digest] = record
        _write(root / "candidates" / f"{digest}.json", record)
    state = {
        "schema_version": "mft-continuous-fea-validation-v1",
        "created_at": "2026-07-18T19:46:19+09:00",
        "updated_at": "2026-07-18T22:00:01+09:00",
        "candidates": records,
    }
    status = {
        "schema_version": "mft-continuous-fea-validation-v1",
        "state": "active",
        "updated_at": "2026-07-18T22:00:02+09:00",
        "last_error": None,
        "production_gate_bypass": False,
        "scheduler_orphan_adoption_enabled": False,
        "active_tasks": active_tasks,
        "counts": {
            "discovered": 2,
            "experimental": 2,
            "production_model": 0,
            "active": 0 if completed else 2,
            "terminal": 2 if completed else 0,
            "valid_result": 2 if completed else 0,
        },
        "paths": {
            "manifest": str((root / "manifest.json").resolve()),
            "state": str((root / "state.json").resolve()),
            "status": str((root / "status.json").resolve()),
            "results": str((root / "results").resolve()),
        },
    }
    state_ref = _write(root / "state.json", state)
    status_ref = _write(root / "status.json", status)
    return {
        "root": root.resolve(),
        "allowed_root": (tmp_path / "runtime").resolve(),
        "manifest": manifest_ref,
        "source": source_ref,
        "execution_plan_sha256": execution_plan_sha,
        "state": state_ref,
        "status": status_ref,
    }


def _configure_validation(monkeypatch, fixture):
    monkeypatch.setenv(DUAL_VALIDATION_ROOT_ENV, str(fixture["root"]))
    monkeypatch.setenv(
        DUAL_VALIDATION_ALLOWED_ROOT_ENV, str(fixture["allowed_root"])
    )
    monkeypatch.setenv(
        DUAL_VALIDATION_MANIFEST_SHA_ENV, fixture["manifest"]["sha256"]
    )
    monkeypatch.setenv(
        DUAL_VALIDATION_PLAN_SHA_ENV, fixture["execution_plan_sha256"]
    )
    monkeypatch.setenv(
        DUAL_VALIDATION_SOURCE_SHA_ENV, fixture["source"]["sha256"]
    )


def test_dual_candidates_are_read_only_and_separate_from_canonical_nsga(
    tmp_path, monkeypatch,
):
    refs = _dual_fixture(tmp_path)
    _configure(monkeypatch, refs)

    payload = _service(tmp_path).nsga2()

    assert payload["candidate_count"] == 0
    assert payload["valid_candidate_count"] == 0
    assert payload["candidates"] == []
    dual = payload["sealed_successors"]
    assert dual["available"] is True
    assert dual["integrity_verified"] is True
    assert dual["candidate_count"] == 2
    assert [
        item["candidate"]["candidate_sha256"] for item in dual["candidates"]
    ] == [DUAL_SUCCESSOR_EXACT_SHA256, DUAL_SUCCESSOR_INTERIOR_SHA256]
    for candidate in dual["candidates"]:
        assert candidate["source_kind"] == "sealed_successor_evidence"
        assert candidate["terminal_result"] is False
        assert candidate["canonical_candidate"] is False
        assert candidate["pareto"] is False
        assert candidate["production"] is False
        assert candidate["fea"] is False
        assert candidate["fea_submission_performed"] is False
        assert candidate["fea_status"] == "not_submitted"
        assert candidate["scheduler_task_id"] is None
        assert candidate["candidate"]["decoded_fea_param_count"] == 71
        assert len(candidate["candidate"]["constraints_G"]) == 22
        assert len(candidate["candidate"]["robust_temperatures_C"]) == 11


def test_dual_plan_tamper_fails_closed_without_stale_candidate(
    tmp_path, monkeypatch,
):
    refs = _dual_fixture(tmp_path)
    _configure(monkeypatch, refs)
    service = _service(tmp_path)
    assert service.nsga2()["sealed_successors"]["candidate_count"] == 2
    plan_path = Path(refs[0]["path"])
    plan_path.write_bytes(plan_path.read_bytes() + b"\n")

    payload = service.nsga2()

    assert payload["candidate_count"] == 0
    assert payload["candidates"] == []
    rejected = payload["sealed_successors"]
    assert rejected["available"] is False
    assert rejected["candidate_count"] == 0
    assert rejected["candidates"] == []
    assert "SHA-256 mismatch" in rejected["warning"]


def test_dual_candidate_authority_attempt_is_rejected(
    tmp_path, monkeypatch,
):
    refs = _dual_fixture(tmp_path, candidate_authority=True)
    _configure(monkeypatch, refs)

    rejected = _service(tmp_path).nsga2()["sealed_successors"]

    assert rejected["available"] is False
    assert rejected["candidates"] == []
    assert "attempted production or FEA authority" in rejected["warning"]


def test_dual_running_standard_fea_is_a_separate_read_only_overlay(
    tmp_path, monkeypatch,
):
    refs = _dual_fixture(tmp_path)
    _configure(monkeypatch, refs)
    runtime = _validation_fixture(tmp_path, refs)
    _configure_validation(monkeypatch, runtime)

    dual = _service(tmp_path).nsga2()["sealed_successors"]

    assert dual["available"] is True
    runtime_view = dual["standard_fea_validation"]
    assert runtime_view["available"] is True
    assert runtime_view["integrity_verified"] is True
    assert runtime_view["candidate_count"] == 2
    for candidate in dual["candidates"]:
        assert candidate["terminal_result"] is False
        assert candidate["pareto"] is False
        assert candidate["production"] is False
        assert candidate["fea"] is False
        assert candidate["scheduler_task_id"] is None
        validation = candidate["standard_fea_validation"]
        assert validation["available"] is True
        assert validation["read_only"] is True
        assert validation["task_status"] == "running"
        assert validation["task"]["id"] in {9000, 9001}
        assert validation["task"]["name"].startswith("mft-nsgafea-x-")
        assert validation["collected"] is False
        assert validation["valid"] is False
        assert validation["actual_gates"] is None


def test_dual_completed_standard_fea_exposes_authenticated_actual_gates(
    tmp_path, monkeypatch,
):
    refs = _dual_fixture(tmp_path)
    _configure(monkeypatch, refs)
    runtime = _validation_fixture(tmp_path, refs, completed=True)
    _configure_validation(monkeypatch, runtime)

    dual = _service(tmp_path).nsga2()["sealed_successors"]

    for candidate in dual["candidates"]:
        validation = candidate["standard_fea_validation"]
        assert validation["task_status"] == "completed"
        assert validation["collected"] is True
        assert validation["valid"] is True
        assert validation["actual_hard_gates_pass"] is True
        assert validation["full_model_validation_candidate_eligible"] is True
        assert validation["result_identity"]["available"] is True
        assert len(validation["result_identity"]["sha256"]) == 64
        assert validation["result_identity"]["candidate_digest"] == (
            validation["candidate_digest"]
        )
        assert validation["actual_gates"]["actual"]["Llt_full_uH"] == 27.5
        assert len(
            validation["actual_gates"]["actual"]["temperatures"]
        ) == 11
        # Validation success never mutates the sealed evidence classification.
        assert candidate["terminal_result"] is False
        assert candidate["pareto"] is False
        assert candidate["production"] is False
        assert candidate["fea"] is False


def test_dual_standard_fea_state_drift_fails_closed_without_hiding_seal(
    tmp_path, monkeypatch,
):
    refs = _dual_fixture(tmp_path)
    _configure(monkeypatch, refs)
    runtime = _validation_fixture(tmp_path, refs)
    _configure_validation(monkeypatch, runtime)
    state_path = Path(runtime["state"]["path"])
    state = json.loads(state_path.read_text(encoding="utf-8"))
    digest = next(iter(state["candidates"]))
    state["candidates"][digest]["source_candidate_sha256"] = "0" * 64
    state_path.write_text(json.dumps(state), encoding="utf-8")

    dual = _service(tmp_path).nsga2()["sealed_successors"]

    assert dual["available"] is True
    assert dual["candidate_count"] == 2
    runtime_view = dual["standard_fea_validation"]
    assert runtime_view["available"] is False
    assert "candidate/state identity drifted" in runtime_view["warning"]
    for candidate in dual["candidates"]:
        validation = candidate["standard_fea_validation"]
        assert validation["available"] is False
        assert validation["integrity_verified"] is False
        assert "rejected" in validation["warning"]
        assert candidate["terminal_result"] is False


def test_dual_standard_fea_runtime_must_stay_under_allowed_root(
    tmp_path, monkeypatch,
):
    refs = _dual_fixture(tmp_path)
    _configure(monkeypatch, refs)
    runtime = _validation_fixture(tmp_path, refs)
    _configure_validation(monkeypatch, runtime)
    monkeypatch.setenv(
        DUAL_VALIDATION_ALLOWED_ROOT_ENV,
        str((tmp_path / "different-root").resolve()),
    )
    (tmp_path / "different-root").mkdir()

    dual = _service(tmp_path).nsga2()["sealed_successors"]

    assert dual["available"] is True
    runtime_view = dual["standard_fea_validation"]
    assert runtime_view["available"] is False
    assert "escapes the allowed root" in runtime_view["warning"]


def test_dual_standard_fea_manifest_hash_drift_is_warning_only_on_overlay(
    tmp_path, monkeypatch,
):
    refs = _dual_fixture(tmp_path)
    _configure(monkeypatch, refs)
    runtime = _validation_fixture(tmp_path, refs)
    _configure_validation(monkeypatch, runtime)
    manifest = Path(runtime["manifest"]["path"])
    manifest.write_bytes(manifest.read_bytes() + b"\n")

    dual = _service(tmp_path).nsga2()["sealed_successors"]

    assert dual["available"] is True
    assert dual["candidate_count"] == 2
    runtime_view = dual["standard_fea_validation"]
    assert runtime_view["available"] is False
    assert "manifest SHA-256 mismatch" in runtime_view["warning"]


def test_dual_actual_gate_is_recomputed_from_canonical_t120_spec(
    tmp_path, monkeypatch,
):
    refs = _dual_fixture(tmp_path)
    _configure(monkeypatch, refs)
    runtime = _validation_fixture(tmp_path, refs, completed=True)
    _configure_validation(monkeypatch, runtime)
    state_path = Path(runtime["state"]["path"])
    state = json.loads(state_path.read_text(encoding="utf-8"))
    digest = next(iter(state["candidates"]))
    record = state["candidates"][digest]
    gate = json.loads(json.dumps(record["tier1_t120_actual_gate"]))
    gate["actual"]["Llt_full_uH"] = 999.0
    for item in gate["actual"]["temperatures"].values():
        item["value_C"] = 999.0
    gate["pass"] = True
    gate["reasons"] = []
    record["tier1_t120_actual_gate"] = gate
    result_path = Path(record["result_path"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["tier1_t120_actual_gate"] = gate
    result["result"]["Llt"] = 499.5
    for name in SEALED_SUCCESSOR_TEMPERATURE_TARGETS:
        result["result"][name] = 999.0
    _write(state_path, state)
    _write(result_path, result)

    overlay = _service(tmp_path).nsga2()["sealed_successors"][
        "standard_fea_validation"
    ]

    assert overlay["available"] is False
    assert "canonical T120 recomputation" in overlay["warning"]


def test_dual_actual_gate_is_cross_checked_against_result_values(
    tmp_path, monkeypatch,
):
    refs = _dual_fixture(tmp_path)
    _configure(monkeypatch, refs)
    runtime = _validation_fixture(tmp_path, refs, completed=True)
    _configure_validation(monkeypatch, runtime)
    state = json.loads(
        Path(runtime["state"]["path"]).read_text(encoding="utf-8")
    )
    record = next(iter(state["candidates"].values()))
    result_path = Path(record["result_path"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["result"]["T_max_Tx"] = 101.0
    _write(result_path, result)

    overlay = _service(tmp_path).nsga2()["sealed_successors"][
        "standard_fea_validation"
    ]

    assert overlay["available"] is False
    assert "temperature snapshot drifted: T_max_Tx" in overlay["warning"]


def test_dual_actual_side_turns_must_match_sealed_parameters(
    tmp_path, monkeypatch,
):
    refs = _dual_fixture(tmp_path)
    _configure(monkeypatch, refs)
    runtime = _validation_fixture(tmp_path, refs, completed=True)
    _configure_validation(monkeypatch, runtime)
    state_path = Path(runtime["state"]["path"])
    state = json.loads(state_path.read_text(encoding="utf-8"))
    record = next(iter(state["candidates"].values()))
    gate = json.loads(json.dumps(record["tier1_t120_actual_gate"]))
    gate["actual"]["N2_side"] = 0
    for name in DUAL_VALIDATION_TEMPERATURE_CONTRACT[
        "side_winding_conditional_targets"
    ]:
        gate["actual"]["temperatures"][name] = {
            "applicable": False,
            "value_C": None,
        }
    record["tier1_t120_actual_gate"] = gate
    result_path = Path(record["result_path"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["tier1_t120_actual_gate"] = gate
    result["result"]["N2_side"] = 0
    _write(state_path, state)
    _write(result_path, result)

    overlay = _service(tmp_path).nsga2()["sealed_successors"][
        "standard_fea_validation"
    ]

    assert overlay["available"] is False
    assert "side-turn identity drifted" in overlay["warning"]


@pytest.mark.parametrize("lifecycle", ["collector_pending", "task_running"])
def test_dual_validity_requires_completed_task_and_collector_success(
    tmp_path, monkeypatch, lifecycle,
):
    refs = _dual_fixture(tmp_path)
    _configure(monkeypatch, refs)
    runtime = _validation_fixture(tmp_path, refs, completed=True)
    _configure_validation(monkeypatch, runtime)
    state_path = Path(runtime["state"]["path"])
    status_path = Path(runtime["status"]["path"])
    state = json.loads(state_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    digest, record = next(iter(state["candidates"].items()))
    if lifecycle == "collector_pending":
        record["collection_state"] = "collector_pending"
    else:
        record["task_status"] = "running"
        status["active_tasks"] = [{
            "candidate_digest": digest,
            "task_id": record["task_id"],
            "task_name": record["task_name"],
            "status": "running",
        }]
        status["counts"]["active"] = 1
        status["counts"]["terminal"] = 1
    status["counts"]["valid_result"] = 1
    _write(state_path, state)
    _write(status_path, status)

    dual = _service(tmp_path).nsga2()["sealed_successors"]

    assert dual["standard_fea_validation"]["available"] is True
    validation = next(
        item["standard_fea_validation"] for item in dual["candidates"]
        if item["standard_fea_validation"]["candidate_digest"] == digest
    )
    assert validation["result_identity"]["available"] is True
    assert validation["valid"] is False
    assert validation["full_model_validation_candidate_eligible"] is False


@pytest.mark.parametrize("subdirectory", ["candidates", "results"])
def test_dual_runtime_rejects_nested_directory_redirects(
    tmp_path, monkeypatch, subdirectory,
):
    refs = _dual_fixture(tmp_path)
    _configure(monkeypatch, refs)
    runtime = _validation_fixture(tmp_path, refs, completed=True)
    _configure_validation(monkeypatch, runtime)
    original = Path(runtime["root"]) / subdirectory
    redirected = tmp_path / f"outside-{subdirectory}"
    original.rename(redirected)
    if os.name == "nt":
        created = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(original), str(redirected)],
            check=False,
            capture_output=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if created.returncode != 0:
            redirected.rename(original)
            pytest.skip("directory junction creation is unavailable")
    else:
        original.symlink_to(redirected, target_is_directory=True)
    try:
        overlay = _service(tmp_path).nsga2()["sealed_successors"][
            "standard_fea_validation"
        ]
    finally:
        if os.name == "nt":
            os.rmdir(original)
        else:
            original.unlink()

    assert overlay["available"] is False
    assert (
        "uses a reparse point" in overlay["warning"]
        or "redirects outside" in overlay["warning"]
    )
