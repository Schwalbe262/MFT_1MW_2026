from __future__ import annotations

import hashlib
import json

import pytest

from regression_260707.monitoring.compute_campaign_status import (
    CURRENT7_CONTROLLER_STATE_ENV,
    CURRENT7_CROSS_BUNDLE_SCHEMA_V1,
    CURRENT7_CROSS_BUNDLE_SCHEMA_V2,
    CURRENT7_CUTOVER_STATUS_ENV,
    CURRENT7_MEMORY_COHORT_POLICY_SCHEMA_V3,
    FINAL_ACCEPTANCE_STATUS_ENV,
    FINAL_MODEL_HARVEST_STATUS_ENV,
    HPO_HARVEST_STATUS_ENV,
    ComputeCampaignStatusReader,
)


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _seal(payload, field):
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {**payload, field: hashlib.sha256(canonical).hexdigest()}


def test_projects_authenticated_hpo_and_current7_resources(tmp_path, monkeypatch):
    hpo = _write(
        tmp_path / "hpo.json",
        _seal(
            {
                "schema_version": "mft-blocker-hpo-v2-slurm-harvest-status-v2",
                "state": "parameter_merge_complete",
                "health": "ok",
                "generated_at": "2026-07-20T15:18:13+00:00",
                "parameter_artifact_eligible": True,
                "readiness": {"parameter_merge_complete": True},
                "counts": {
                    "authenticated_results": 44,
                    "authenticated_trials": 8800,
                },
                "merge_task": {"task_id": 64800, "status": "completed", "exit_code": 0},
                "remote_merge_receipt": {
                    "artifact_receipt": {"generation_id": "b1a75847" * 8}
                },
            },
            "sha256",
        ),
    )
    current7 = _write(
        tmp_path / "current7.json",
        _seal(
            {
                "schema_version": "mft-tier1-current7-slurm-controller-state-v1",
                "bundle_id": "current7-test",
                "revision": 99,
                "stop_requested": False,
                "scale_summary": {
                    "effective_active_total": 112,
                    "scaled_resource_contract": {
                        "cpus_per_seed_task": 8,
                        "memory_mb_per_seed_task": 65536,
                        "max_workers_per_node": 8,
                        "priority": 0,
                    },
                },
            },
            "state_sha256",
        ),
    )
    monkeypatch.setenv(HPO_HARVEST_STATUS_ENV, str(hpo))
    monkeypatch.setenv(CURRENT7_CONTROLLER_STATE_ENV, str(current7))

    snapshot = ComputeCampaignStatusReader().snapshot()

    assert snapshot["read_only"] is True
    hpo_stage = snapshot["stages"]["hpo"]
    assert hpo_stage["complete"] is True
    assert hpo_stage["integrity_verified"] is True
    assert hpo_stage["authenticated_results"] == 44
    assert hpo_stage["authenticated_trials"] == 8800
    assert hpo_stage["task_id"] == 64800
    assert hpo_stage["task_status"] == "completed"
    assert hpo_stage["state"] == "complete"
    assert snapshot["resources"]["active_target"] == 112
    assert snapshot["resources"]["policy_active_target"] == 112
    assert snapshot["resources"]["observed_active_tasks"] is None
    assert snapshot["resources"]["requested_cpus"] == 896
    assert snapshot["resources"]["requested_memory_mb"] == 112 * 65536
    assert snapshot["resources"]["stop_requested"] is False


def test_configured_missing_stage_is_visible_and_warns(tmp_path, monkeypatch):
    missing = tmp_path / "acceptance.json"
    monkeypatch.setenv(FINAL_ACCEPTANCE_STATUS_ENV, str(missing))

    snapshot = ComputeCampaignStatusReader().snapshot()

    stage = snapshot["stages"]["acceptance"]
    assert stage["configured"] is True
    assert stage["available"] is False
    assert stage["state"] == "unavailable"
    assert "does not exist" in stage["error"]
    assert snapshot["warnings"] == [
        "Final 21-target training / quality gate: status file does not exist"
    ]


def test_hpo_completion_fails_closed_when_status_digest_is_tampered(
    tmp_path, monkeypatch
):
    path = _write(
        tmp_path / "hpo.json",
        {
            "schema_version": "mft-blocker-hpo-v2-slurm-harvest-status-v2",
            "state": "parameter_merge_complete",
            "health": "ok",
            "parameter_artifact_eligible": True,
            "readiness": {"parameter_merge_complete": True},
            "merge_task": {"status": "completed", "exit_code": 0},
            "sha256": "0" * 64,
        },
    )
    monkeypatch.setenv(HPO_HARVEST_STATUS_ENV, str(path))

    stage = ComputeCampaignStatusReader().snapshot()["stages"]["hpo"]

    assert stage["integrity_verified"] is False
    assert stage["complete"] is False
    assert stage["state"] == "unavailable"


def test_unconfigured_future_stages_do_not_raise_false_warnings():
    snapshot = ComputeCampaignStatusReader().snapshot()

    assert snapshot["available"] is False
    assert snapshot["warnings"] == []
    assert all(
        stage["state"] == "not_configured" for stage in snapshot["stages"].values()
    )


def _successor_receipt():
    return _seal(
        {
            "schema_version": "mft-tier1-current7-permanent-successor-receipt-v1",
            "created_at": "2026-07-20T19:48:42+00:00",
            "acceptance_terminal_observation": {
                "task_id": 65198,
                "terminal_status": "failed",
                "exit_code": 70,
                "failure_message": "srun: task exited with code 70",
            },
            "state_transition": "resume_authorized_to_successor_committed",
            "preserve_current7_ledger_and_submissions": True,
            "preserved_current7_state": {"scheduler_submit_count": 19},
            "scheduled_task_resume_performed": False,
            "scheduler_mutation_performed": False,
            "stop_marker_retained": True,
        },
        "successor_receipt_sha256",
    )


def _cross_bundle_state_v1():
    return _seal(
        {
            "schema_version": CURRENT7_CROSS_BUNDLE_SCHEMA_V1,
            "revision": 454,
            "scheduler_submit_count": 19,
            "policy": {
                "schema_version": "mft-tier1-current7-cross-bundle-policy-v1",
                "new_bundle_id": "current7-091003f1dbe262ed9aca",
                "global_active_limit": 112,
                "current7_exact_resources": {
                    "cpus": 8,
                    "memory_mb": 65536,
                    "max_workers_per_node": 8,
                    "priority": 0,
                },
            },
            "takeover": {"status": "successor_committed"},
        },
        "state_sha256",
    )


def _memory_cohort_policy_v3():
    contract = {
        "cpus_per_seed_task": 8,
        "memory_mb_per_seed_task": 43008,
        "max_workers_per_node": 16,
        "priority": 0,
    }
    contract_sha = hashlib.sha256(
        json.dumps(
            contract,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    scale_policy = _seal(
        {
            "schema_version": "mft-tier1-current7-scale-policy-v1",
            "scaled_active_total": 500,
        },
        "policy_sha256",
    )
    return _seal(
        {
            "schema_version": CURRENT7_MEMORY_COHORT_POLICY_SCHEMA_V3,
            "bundle_id": "current7-091003f1dbe262ed9aca",
            "activated_at": "2026-07-20T21:46:41+00:00",
            "successor_active_target": 500,
            "successor_cohort": "successor-42g",
            "successor_resource_policy": "scaled-memory-42g-worker16-successor-v2",
            "successor_global_unplaced_limit": 16,
            "successor_target_is_nonbinding": True,
            "successor_unplaced_requires_known_physical_node": True,
            "physical_allocation_admission_remains_authoritative": True,
            "unknown_target_global_fail_closed": True,
            "successor_exact_scheduler_resources": {
                "cpus": 8,
                "memory_mb": 43008,
                "max_workers_per_node": 16,
                "priority": 0,
            },
            "successor_resource_contract": contract,
            "successor_resource_contract_sha256": contract_sha,
            "successor_scale_policy": scale_policy,
            "successor_scale_policy_sha256": scale_policy["policy_sha256"],
        },
        "cohort_policy_sha256",
    )


def _cross_bundle_state_v2():
    return _seal(
        {
            "schema_version": CURRENT7_CROSS_BUNDLE_SCHEMA_V2,
            "revision": 1180,
            "scheduler_submit_count": 226,
            "policy": {
                "schema_version": "mft-tier1-current7-cross-bundle-policy-v1",
                "new_bundle_id": "current7-091003f1dbe262ed9aca",
                "global_active_limit": 112,
                "current7_exact_resources": {
                    "cpus": 8,
                    "memory_mb": 65536,
                    "max_workers_per_node": 8,
                    "priority": 0,
                },
            },
            "memory_cohort_policy": _memory_cohort_policy_v3(),
            "takeover": {"status": "successor_committed"},
        },
        "state_sha256",
    )


def _reseal(payload, field):
    unsigned = {key: value for key, value in payload.items() if key != field}
    return _seal(unsigned, field)


def test_projects_truthful_failed_acceptance_and_independent_safe_cutover(
    tmp_path, monkeypatch
):
    receipt = _write(tmp_path / "successor.json", _successor_receipt())
    monkeypatch.setenv(FINAL_ACCEPTANCE_STATUS_ENV, str(receipt))
    monkeypatch.setenv(CURRENT7_CUTOVER_STATUS_ENV, str(receipt))

    snapshot = ComputeCampaignStatusReader().snapshot()

    acceptance = snapshot["stages"]["acceptance"]
    assert acceptance["available"] is True
    assert acceptance["integrity_verified"] is True
    assert acceptance["schema_version"] == (
        "mft-tier1-current7-permanent-successor-receipt-v1"
    )
    assert acceptance["complete"] is False
    assert acceptance["state"] == "failed"
    assert acceptance["task_id"] == 65198
    assert acceptance["exit_code"] == 70
    assert "code 70" in acceptance["failure_message"]

    cutover = snapshot["stages"]["cutover"]
    assert cutover["available"] is True
    assert cutover["integrity_verified"] is True
    assert cutover["complete"] is True
    assert cutover["state"] == "complete"
    assert cutover["transition"] == "resume_authorized_to_successor_committed"
    assert cutover["model_publication_independent"] is True
    assert cutover["scheduler_submit_count"] == 19


def test_projects_signed_failed_harvest_without_claiming_publication(
    tmp_path, monkeypatch
):
    terminal = _write(
        tmp_path / "failed-terminal.json",
        _seal(
            {
                "schema_version": "mft-blocker-hpo-v2-final-model-terminal-v1",
                "state": "infrastructure_failed_after_claim",
                "task_id": 65198,
                "generation_id": "b1a75847" * 8,
                "terminal_success": False,
                "quality_passed": False,
                "scientific_rejection": False,
                "production_model_eligible": False,
                "promotion_performed": False,
                "error_message": "final model train report contract mismatch",
                "finished_at": "2026-07-20T18:56:20+00:00",
            },
            "sha256",
        ),
    )
    monkeypatch.setenv(FINAL_MODEL_HARVEST_STATUS_ENV, str(terminal))

    stage = ComputeCampaignStatusReader().snapshot()["stages"]["harvest"]

    assert stage["available"] is True
    assert stage["integrity_verified"] is True
    assert stage["complete"] is False
    assert stage["state"] == "infrastructure_failed_after_claim"
    assert stage["task_status"] == "failed"
    assert stage["promotion_performed"] is False
    assert stage["production_model_eligible"] is False
    assert "contract mismatch" in stage["failure_message"]
    assert "unavailable as authenticated" in stage["advisory"]
    assert len(ComputeCampaignStatusReader().snapshot()["advisories"]) == 1


def test_projects_signed_strict_quality_rejection_without_claiming_publication(
    tmp_path, monkeypatch
):
    status = _write(
        tmp_path / "quality-rejected.json",
        _seal(
            {
                "schema_version": "mft-blocker-hpo-final-model-harvest-status-v1",
                "state": "strict_quality_rejected",
                "task_id": 65198,
                "training_run_id": "20260721T034543-868d701a",
                "quality_passed": False,
                "scientific_rejection": True,
                "production_model_eligible": False,
                "promotion_performed": False,
                "failure_message": "strict local quality gate rejected the generation",
                "observed_at": "2026-07-21T05:39:18+09:00",
            },
            "sha256",
        ),
    )
    monkeypatch.setenv(FINAL_MODEL_HARVEST_STATUS_ENV, str(status))

    stage = ComputeCampaignStatusReader().snapshot()["stages"]["harvest"]

    assert stage["available"] is True
    assert stage["complete"] is False
    assert stage["state"] == "strict_quality_rejected"
    assert stage["task_status"] == "failed"
    assert stage["scientific_rejection"] is True
    assert stage["promotion_performed"] is False


def test_projects_signed_promotion_receipt_as_complete(tmp_path, monkeypatch):
    receipt = _write(
        tmp_path / "promotion.json",
        _seal(
            {
                "schema_version": "mft-blocker-hpo-final-model-promotion-receipt-v1",
                "state": "promoted",
                "task_id": 65198,
                "training_run_id": "20260721T034543-868d701a",
                "promotion_performed": True,
                "promoted_at": "2026-07-21T06:00:00+09:00",
            },
            "sha256",
        ),
    )
    monkeypatch.setenv(FINAL_MODEL_HARVEST_STATUS_ENV, str(receipt))

    stage = ComputeCampaignStatusReader().snapshot()["stages"]["harvest"]

    assert stage["available"] is True
    assert stage["complete"] is True
    assert stage["state"] == "complete"
    assert stage["promotion_performed"] is True


def test_current7_cross_bundle_v1_resource_projection_is_unchanged(
    tmp_path, monkeypatch
):
    state = _write(tmp_path / "controller.json", _cross_bundle_state_v1())
    monkeypatch.setenv(CURRENT7_CONTROLLER_STATE_ENV, str(state))

    resources = ComputeCampaignStatusReader().snapshot()["resources"]

    assert resources["available"] is True
    assert resources["integrity_verified"] is True
    assert resources["schema_version"] == CURRENT7_CROSS_BUNDLE_SCHEMA_V1
    assert resources["adapter_version"] == "cross_bundle_v1"
    assert resources["bundle_id"] == "current7-091003f1dbe262ed9aca"
    assert resources["active_target"] == 112
    assert resources["policy_active_target"] == 112
    assert resources["observed_active_tasks"] is None
    assert resources["target_semantics"] == "policy_ceiling_not_observed_running"
    assert resources["requested_cpus"] == 896
    assert resources["requested_memory_mb"] == 112 * 65536
    assert resources["max_workers_per_node"] == 8
    assert resources["global_unplaced_limit"] is None
    assert resources["takeover_status"] == "successor_committed"


def test_literal_v2_uses_only_activated_memory_cohort_successor_contract(
    tmp_path, monkeypatch
):
    state = _write(tmp_path / "controller.json", _cross_bundle_state_v2())
    monkeypatch.setenv(CURRENT7_CONTROLLER_STATE_ENV, str(state))

    resources = ComputeCampaignStatusReader().snapshot()["resources"]

    assert resources["available"] is True
    assert resources["integrity_verified"] is True
    assert resources["policy_integrity_verified"] is True
    assert resources["resource_contract_integrity_verified"] is True
    assert resources["scale_policy_integrity_verified"] is True
    assert resources["schema_version"] == CURRENT7_CROSS_BUNDLE_SCHEMA_V2
    assert resources["adapter_version"] == "cross_bundle_v2_memory_cohort_v3"
    assert resources["memory_cohort_policy_schema"] == (
        CURRENT7_MEMORY_COHORT_POLICY_SCHEMA_V3
    )
    assert resources["bundle_id"] == "current7-091003f1dbe262ed9aca"
    assert resources["active_target"] == 500
    assert resources["policy_active_target"] == 500
    assert resources["observed_active_tasks"] is None
    assert resources["target_semantics"] == (
        "activated_successor_target_not_observed_running"
    )
    assert resources["cpus_per_task"] == 8
    assert resources["memory_mb_per_task"] == 43008
    assert resources["max_workers_per_node"] == 16
    assert resources["global_unplaced_limit"] == 16
    assert resources["requested_cpus"] == 4000
    assert resources["requested_memory_mb"] == 500 * 43008
    assert resources["resource_cohort"] == "successor-42g"
    assert resources["target_nonbinding"] is True


def test_current7_controller_digest_tamper_fails_closed_and_warns(
    tmp_path, monkeypatch
):
    payload = _cross_bundle_state_v2()
    payload["revision"] = 455
    state = _write(tmp_path / "controller.json", payload)
    monkeypatch.setenv(CURRENT7_CONTROLLER_STATE_ENV, str(state))

    snapshot = ComputeCampaignStatusReader().snapshot()

    assert snapshot["resources"]["available"] is False
    assert snapshot["resources"]["integrity_verified"] is False
    assert snapshot["resources"]["error"] == "controller integrity check failed"
    assert snapshot["warnings"] == [
        "Current7 controller resources: controller integrity check failed"
    ]


def test_v2_memory_cohort_policy_digest_tamper_fails_closed(tmp_path, monkeypatch):
    payload = _cross_bundle_state_v2()
    payload["memory_cohort_policy"]["successor_active_target"] = 501
    payload = _reseal(payload, "state_sha256")
    state = _write(tmp_path / "controller.json", payload)
    monkeypatch.setenv(CURRENT7_CONTROLLER_STATE_ENV, str(state))

    snapshot = ComputeCampaignStatusReader().snapshot()
    resources = snapshot["resources"]

    assert resources["available"] is False
    assert resources["integrity_verified"] is False
    assert resources["policy_integrity_verified"] is False
    assert resources["error"] == (
        "Current7 memory cohort policy integrity check failed"
    )
    assert "integrity check failed" in snapshot["warnings"][0]


def test_v2_missing_memory_cohort_policy_does_not_fall_back_to_112(
    tmp_path, monkeypatch
):
    payload = _cross_bundle_state_v2()
    payload.pop("memory_cohort_policy")
    payload = _reseal(payload, "state_sha256")
    state = _write(tmp_path / "controller.json", payload)
    monkeypatch.setenv(CURRENT7_CONTROLLER_STATE_ENV, str(state))

    resources = ComputeCampaignStatusReader().snapshot()["resources"]

    assert resources["available"] is False
    assert resources["integrity_verified"] is False
    assert resources["memory_cohort_policy_schema"] is None
    assert "schema: missing" in resources["error"]
    assert "active_target" not in resources


def test_v2_unsupported_memory_cohort_policy_schema_fails_closed(tmp_path, monkeypatch):
    payload = _cross_bundle_state_v2()
    policy = payload["memory_cohort_policy"]
    policy["schema_version"] = "mft-tier1-current7-memory-cohort-policy-v99"
    payload["memory_cohort_policy"] = _reseal(policy, "cohort_policy_sha256")
    payload = _reseal(payload, "state_sha256")
    state = _write(tmp_path / "controller.json", payload)
    monkeypatch.setenv(CURRENT7_CONTROLLER_STATE_ENV, str(state))

    resources = ComputeCampaignStatusReader().snapshot()["resources"]

    assert resources["available"] is False
    assert resources["integrity_verified"] is False
    assert "policy-v99" in resources["error"]
    assert "active_target" not in resources


def test_v2_signed_but_internally_inconsistent_resources_fail_closed(
    tmp_path, monkeypatch
):
    payload = _cross_bundle_state_v2()
    policy = payload["memory_cohort_policy"]
    policy["successor_exact_scheduler_resources"]["memory_mb"] = 44000
    payload["memory_cohort_policy"] = _reseal(policy, "cohort_policy_sha256")
    payload = _reseal(payload, "state_sha256")
    state = _write(tmp_path / "controller.json", payload)
    monkeypatch.setenv(CURRENT7_CONTROLLER_STATE_ENV, str(state))

    resources = ComputeCampaignStatusReader().snapshot()["resources"]

    assert resources["available"] is False
    assert resources["integrity_verified"] is False
    assert resources["policy_integrity_verified"] is True
    assert resources["resource_contract_integrity_verified"] is True
    assert "contract is incomplete" in resources["error"]


def test_wrong_controller_schema_fails_closed_and_is_visible(tmp_path, monkeypatch):
    state = _write(
        tmp_path / "controller.json",
        _seal(
            {"schema_version": "mft-tier1-current7-cross-bundle-state-v99"},
            "state_sha256",
        ),
    )
    monkeypatch.setenv(CURRENT7_CONTROLLER_STATE_ENV, str(state))

    snapshot = ComputeCampaignStatusReader().snapshot()

    assert snapshot["resources"]["available"] is False
    assert snapshot["resources"]["integrity_verified"] is False
    assert "state-v99" in snapshot["resources"]["error"]
    assert len(snapshot["warnings"]) == 1
    assert "unsupported Current7 controller schema" in snapshot["warnings"][0]


@pytest.mark.parametrize(
    ("env_name", "stage_name", "lane"),
    [
        (FINAL_ACCEPTANCE_STATUS_ENV, "acceptance", "final acceptance"),
        (FINAL_MODEL_HARVEST_STATUS_ENV, "harvest", "final model harvest"),
        (CURRENT7_CUTOVER_STATUS_ENV, "cutover", "Current7 cutover"),
    ],
)
def test_wrong_stage_schema_fails_closed_and_warns(
    tmp_path, monkeypatch, env_name, stage_name, lane
):
    path = _write(
        tmp_path / f"{stage_name}.json",
        _seal(
            {
                "schema_version": f"synthetic-{stage_name}-canary-v1",
                "state": "complete",
            },
            "sha256",
        ),
    )
    monkeypatch.setenv(env_name, str(path))

    snapshot = ComputeCampaignStatusReader().snapshot()
    stage = snapshot["stages"][stage_name]

    assert stage["available"] is False
    assert stage["complete"] is False
    assert stage["state"] == "unavailable"
    assert stage["integrity_verified"] is False
    assert f"unsupported {lane} schema" in stage["error"]
    assert len(snapshot["warnings"]) == 1
