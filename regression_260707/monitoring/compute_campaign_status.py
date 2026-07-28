"""Small, fail-closed projections for the live HPO/retrain/NSGA campaign.

The source status files are controller-owned.  The monitor only reads them and
returns a bounded projection; it never mutates campaign or scheduler state.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


HPO_HARVEST_STATUS_ENV = "MFT_HPO_HARVEST_STATUS"
FINAL_ACCEPTANCE_STATUS_ENV = "MFT_FINAL_ACCEPTANCE_STATUS"
FINAL_MODEL_HARVEST_STATUS_ENV = "MFT_FINAL_MODEL_HARVEST_STATUS"
CURRENT7_CONTROLLER_STATE_ENV = "MFT_CURRENT7_CONTROLLER_STATE"
CURRENT7_CUTOVER_STATUS_ENV = "MFT_CURRENT7_CUTOVER_STATUS"

HPO_HARVEST_SCHEMA = "mft-blocker-hpo-v2-slurm-harvest-status-v2"
CURRENT7_CONTROLLER_SCHEMA = "mft-tier1-current7-slurm-controller-state-v1"
CURRENT7_CROSS_BUNDLE_SCHEMA_V1 = "mft-tier1-current7-cross-bundle-state-v1"
CURRENT7_CROSS_BUNDLE_SCHEMA_V2 = "mft-tier1-current7-cross-bundle-state-v2"
CURRENT7_CROSS_BUNDLE_POLICY_SCHEMA = "mft-tier1-current7-cross-bundle-policy-v1"
CURRENT7_MEMORY_COHORT_POLICY_SCHEMA_V3 = "mft-tier1-current7-memory-cohort-policy-v3"
FINAL_ACCEPTANCE_SUBMISSION_SCHEMA = "mft-blocker-hpo-v2-final-acceptance-submission-v1"
FINAL_ACCEPTANCE_RECONCILE_SCHEMA = "mft-blocker-hpo-v2-final-acceptance-reconcile-v1"
FINAL_MODEL_TERMINAL_SCHEMA = "mft-blocker-hpo-v2-final-model-terminal-v1"
FINAL_MODEL_HARVEST_SCHEMA = "mft-blocker-hpo-final-model-harvest-status-v1"
FINAL_MODEL_PROMOTION_SCHEMA = "mft-blocker-hpo-final-model-promotion-receipt-v1"
PERMANENT_SUCCESSOR_RECEIPT_SCHEMA = "mft-tier1-current7-permanent-successor-receipt-v1"
MAX_STATUS_BYTES = 32 * 1024 * 1024


def _text(value: Any, *, limit: int = 300) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    return value[:limit]


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _canonical_sha256_verified(payload: dict[str, Any], field: str) -> bool:
    expected = payload.get(field)
    if not isinstance(expected, str) or len(expected) != 64:
        return False
    unsigned = {key: value for key, value in payload.items() if key != field}
    canonical = json.dumps(
        unsigned,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest() == expected


def _canonical_sha256_matches(payload: dict[str, Any], expected: Any) -> bool:
    if not isinstance(expected, str) or len(expected) != 64:
        return False
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest() == expected


def _read_status(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        if path.is_symlink():
            return None, "symbolic-link status files are not accepted"
        stat = path.stat()
        if not path.is_file():
            return None, "status path is not a regular file"
        if stat.st_size <= 0 or stat.st_size > MAX_STATUS_BYTES:
            return None, "status file size is outside the accepted bound"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None, "status payload is not a JSON object"
        return payload, None
    except FileNotFoundError:
        return None, "status file does not exist"
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _configured_path(env_name: str) -> Path | None:
    raw = os.environ.get(env_name, "").strip()
    return Path(raw).resolve() if raw else None


def _stage_source(
    env_name: str, label: str
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    path = _configured_path(env_name)
    if path is None:
        return (
            {
                "configured": False,
                "available": False,
                "label": label,
                "state": "not_configured",
            },
            None,
        )
    payload, error = _read_status(path)
    base = {
        "configured": True,
        "available": False,
        "label": label,
        "state": "unavailable",
        "source": str(path),
    }
    if payload is None:
        return ({**base, "error": error}, None)
    return (base, payload)


def _unsupported_stage(
    base: dict[str, Any], payload: dict[str, Any], *, lane: str
) -> dict[str, Any]:
    schema = _text(payload.get("schema_version"), limit=160)
    return {
        **base,
        "schema_version": schema,
        "integrity_verified": False,
        "complete": False,
        "error": f"unsupported {lane} schema: {schema or 'missing'}",
    }


def _unverified_stage(
    base: dict[str, Any], payload: dict[str, Any], *, field: str
) -> dict[str, Any]:
    return {
        **base,
        "schema_version": _text(payload.get("schema_version"), limit=160),
        "integrity_verified": False,
        "complete": False,
        "error": f"status integrity check failed for {field}",
    }


def _hpo_stage() -> dict[str, Any]:
    base, payload = _stage_source(HPO_HARVEST_STATUS_ENV, "HPO parameter merge")
    if payload is None:
        return base
    if payload.get("schema_version") != HPO_HARVEST_SCHEMA:
        return _unsupported_stage(base, payload, lane="HPO harvest")
    if not _canonical_sha256_verified(payload, "sha256"):
        return _unverified_stage(base, payload, field="sha256")
    readiness = _mapping(payload.get("readiness"))
    counts = _mapping(payload.get("counts"))
    merge_task = _mapping(payload.get("merge_task"))
    receipt = _mapping(payload.get("remote_merge_receipt"))
    artifact_receipt = _mapping(receipt.get("artifact_receipt"))
    complete = bool(
        payload.get("health") == "ok"
        and payload.get("state") == "parameter_merge_complete"
        and readiness.get("parameter_merge_complete") is True
        and payload.get("parameter_artifact_eligible") is True
        and merge_task.get("status") == "completed"
        and merge_task.get("exit_code") == 0
    )
    return {
        **base,
        "available": True,
        "schema_version": HPO_HARVEST_SCHEMA,
        "integrity_verified": True,
        "complete": complete,
        "state": "complete" if complete else (_text(payload.get("state")) or "unknown"),
        "health": _text(payload.get("health")),
        "generated_at": _text(payload.get("generated_at")),
        "task_id": _integer(merge_task.get("task_id")),
        "task_status": _text(merge_task.get("status")),
        "generation_id": _text(artifact_receipt.get("generation_id"), limit=128),
        "authenticated_results": _integer(counts.get("authenticated_results")),
        "authenticated_trials": _integer(counts.get("authenticated_trials")),
    }


def _acceptance_stage() -> dict[str, Any]:
    base, payload = _stage_source(
        FINAL_ACCEPTANCE_STATUS_ENV, "Final 21-target training / quality gate"
    )
    if payload is None:
        return base
    schema = payload.get("schema_version")
    if schema == PERMANENT_SUCCESSOR_RECEIPT_SCHEMA:
        digest_field = "successor_receipt_sha256"
    elif schema in {
        FINAL_ACCEPTANCE_SUBMISSION_SCHEMA,
        FINAL_ACCEPTANCE_RECONCILE_SCHEMA,
        FINAL_MODEL_TERMINAL_SCHEMA,
    }:
        digest_field = "sha256"
    else:
        return _unsupported_stage(base, payload, lane="final acceptance")
    if not _canonical_sha256_verified(payload, digest_field):
        return _unverified_stage(base, payload, field=digest_field)

    if schema == PERMANENT_SUCCESSOR_RECEIPT_SCHEMA:
        observation = _mapping(payload.get("acceptance_terminal_observation"))
        task_status = _text(observation.get("terminal_status")) or "unknown"
        return {
            **base,
            "available": True,
            "schema_version": schema,
            "integrity_verified": True,
            "complete": task_status in {"completed", "succeeded"}
            and observation.get("exit_code") == 0,
            "state": task_status,
            "task_id": _integer(observation.get("task_id")),
            "task_status": task_status,
            "exit_code": _integer(observation.get("exit_code")),
            "failure_message": _text(observation.get("failure_message"), limit=500),
            "generated_at": _text(payload.get("created_at")),
        }

    if schema == FINAL_MODEL_TERMINAL_SCHEMA:
        task_status = (
            "succeeded" if payload.get("terminal_success") is True else "failed"
        )
        return {
            **base,
            "available": True,
            "schema_version": schema,
            "integrity_verified": True,
            "complete": payload.get("terminal_success") is True
            and payload.get("quality_passed") is True,
            "state": _text(payload.get("state")) or task_status,
            "task_id": _integer(payload.get("task_id")),
            "task_status": task_status,
            "generation_id": _text(payload.get("generation_id"), limit=128),
            "quality_passed": payload.get("quality_passed") is True,
            "promotion_performed": payload.get("promotion_performed") is True,
            "failure_message": _text(payload.get("error_message"), limit=500),
            "generated_at": _text(payload.get("finished_at")),
        }

    submission = _mapping(payload.get("submission_receipt"))
    task_id = _integer(payload.get("task_id")) or _integer(submission.get("task_id"))
    task_status = (
        _text(payload.get("task_status"))
        or _text(payload.get("task_status_at_receipt"))
        or _text(submission.get("task_status_at_receipt"))
    )
    state = _text(payload.get("state")) or task_status or "submitted"
    return {
        **base,
        "available": True,
        "schema_version": schema,
        "integrity_verified": True,
        "complete": False,
        "state": state,
        "task_id": task_id,
        "task_status": task_status,
        "generation_id": _text(
            payload.get("generation_id") or submission.get("generation_id"), limit=128
        ),
        "generated_at": _text(payload.get("reconciled_at"))
        or _text(payload.get("created_at")),
    }


def _harvest_stage() -> dict[str, Any]:
    base, payload = _stage_source(
        FINAL_MODEL_HARVEST_STATUS_ENV, "Authenticated model publication"
    )
    if payload is None:
        return base
    schema = payload.get("schema_version")
    if schema not in {
        FINAL_MODEL_TERMINAL_SCHEMA,
        FINAL_MODEL_HARVEST_SCHEMA,
        FINAL_MODEL_PROMOTION_SCHEMA,
    }:
        return _unsupported_stage(base, payload, lane="final model harvest")
    if not _canonical_sha256_verified(payload, "sha256"):
        return _unverified_stage(base, payload, field="sha256")

    state = _text(payload.get("state")) or "unknown"
    promotion_performed = payload.get("promotion_performed") is True
    complete = bool(
        schema == FINAL_MODEL_PROMOTION_SCHEMA
        and state == "promoted"
        and promotion_performed
    )
    failed = bool(
        payload.get("terminal_success") is False
        or state
        in {
            "failed",
            "infrastructure_failed_after_claim",
            "quality_rejected",
            "strict_quality_rejected",
        }
    )
    task_status = (
        "failed"
        if failed
        else ("completed" if complete else _text(payload.get("task_status")))
    )
    advisory = (
        "Subsequent local recovery observations are unavailable as authenticated "
        "campaign status."
        if schema == FINAL_MODEL_TERMINAL_SCHEMA and failed
        else None
    )
    return {
        **base,
        "available": True,
        "schema_version": schema,
        "integrity_verified": True,
        "complete": complete,
        "state": "complete" if complete else state,
        "task_id": _integer(payload.get("task_id")),
        "task_status": task_status,
        "generation_id": _text(
            payload.get("generation_id") or payload.get("training_run_id"), limit=128
        ),
        "quality_passed": payload.get("quality_passed") is True,
        "scientific_rejection": payload.get("scientific_rejection") is True,
        "promotion_performed": promotion_performed,
        "production_model_eligible": payload.get("production_model_eligible") is True,
        "failure_message": _text(
            payload.get("error_message") or payload.get("failure_message"), limit=500
        ),
        "advisory": advisory,
        "generated_at": _text(
            payload.get("finished_at")
            or payload.get("promoted_at")
            or payload.get("observed_at")
        ),
    }


def _cutover_stage() -> dict[str, Any]:
    base, payload = _stage_source(
        CURRENT7_CUTOVER_STATUS_ENV, "Current7 controller successor cutover"
    )
    if payload is None:
        return base
    if payload.get("schema_version") != PERMANENT_SUCCESSOR_RECEIPT_SCHEMA:
        return _unsupported_stage(base, payload, lane="Current7 cutover")
    if not _canonical_sha256_verified(payload, "successor_receipt_sha256"):
        return _unverified_stage(base, payload, field="successor_receipt_sha256")
    transition = _text(payload.get("state_transition"), limit=160) or "unknown"
    committed = bool(
        transition == "resume_authorized_to_successor_committed"
        and payload.get("preserve_current7_ledger_and_submissions") is True
        and payload.get("scheduled_task_resume_performed") is False
        and payload.get("scheduler_mutation_performed") is False
        and payload.get("stop_marker_retained") is True
    )
    preserved = _mapping(payload.get("preserved_current7_state"))
    return {
        **base,
        "available": True,
        "schema_version": PERMANENT_SUCCESSOR_RECEIPT_SCHEMA,
        "integrity_verified": True,
        "complete": committed,
        "state": "complete" if committed else transition,
        "transition": transition,
        "generated_at": _text(payload.get("created_at")),
        "scheduler_mutation_performed": payload.get("scheduler_mutation_performed")
        is True,
        "scheduled_task_resume_performed": payload.get(
            "scheduled_task_resume_performed"
        )
        is True,
        "stop_marker_retained": payload.get("stop_marker_retained") is True,
        "scheduler_submit_count": _integer(preserved.get("scheduler_submit_count")),
        "model_publication_independent": True,
    }


def _unavailable_resources(path: Path, error: str) -> dict[str, Any]:
    return {
        "configured": True,
        "available": False,
        "source": str(path),
        "error": error,
    }


def _legacy_current7_resources(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    if not _canonical_sha256_verified(payload, "state_sha256"):
        return {
            **_unavailable_resources(path, "controller integrity check failed"),
            "schema_version": CURRENT7_CONTROLLER_SCHEMA,
            "integrity_verified": False,
        }
    summary = _mapping(payload.get("scale_summary"))
    contract = _mapping(summary.get("scaled_resource_contract"))
    active = _integer(summary.get("effective_active_total"))
    cpus = _integer(contract.get("cpus_per_seed_task"))
    memory_mb = _integer(contract.get("memory_mb_per_seed_task"))
    workers = _integer(contract.get("max_workers_per_node"))
    if not active or not cpus or not memory_mb or not workers:
        return {
            **_unavailable_resources(
                path, "legacy controller resource contract is incomplete"
            ),
            "schema_version": CURRENT7_CONTROLLER_SCHEMA,
            "integrity_verified": True,
        }
    return {
        "configured": True,
        "available": True,
        "schema_version": CURRENT7_CONTROLLER_SCHEMA,
        "adapter_version": "legacy_controller_v1",
        "integrity_verified": True,
        "bundle_id": _text(payload.get("bundle_id"), limit=160),
        "active_target": active,
        "policy_active_target": active,
        "observed_active_tasks": None,
        "observed_task_status_counts": None,
        "target_semantics": "controller_effective_target_not_observed_running",
        "cpus_per_task": cpus,
        "memory_mb_per_task": memory_mb,
        "requested_cpus": active * cpus,
        "requested_memory_mb": active * memory_mb,
        "max_workers_per_node": workers,
        "priority": _integer(contract.get("priority")),
        "stop_requested": payload.get("stop_requested") is True,
        "revision": _integer(payload.get("revision")),
        "scheduler_submit_count": _integer(payload.get("scheduler_submit_count")),
        "source": str(path),
    }


def _cross_bundle_v1_resources(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    policy = _mapping(payload.get("policy"))
    contract = _mapping(policy.get("current7_exact_resources"))
    takeover = _mapping(payload.get("takeover"))
    active = _integer(policy.get("global_active_limit"))
    cpus = _integer(contract.get("cpus"))
    memory_mb = _integer(contract.get("memory_mb"))
    workers = _integer(contract.get("max_workers_per_node"))
    contract_valid = bool(
        policy.get("schema_version") == CURRENT7_CROSS_BUNDLE_POLICY_SCHEMA
        and takeover.get("status") == "successor_committed"
        and _text(policy.get("new_bundle_id"), limit=160)
        and active
        and cpus
        and memory_mb
        and workers
    )
    if not contract_valid:
        return {
            **_unavailable_resources(
                path, "cross-bundle controller contract is incomplete"
            ),
            "schema_version": CURRENT7_CROSS_BUNDLE_SCHEMA_V1,
            "integrity_verified": True,
        }
    return {
        "configured": True,
        "available": True,
        "schema_version": CURRENT7_CROSS_BUNDLE_SCHEMA_V1,
        "adapter_version": "cross_bundle_v1",
        "integrity_verified": True,
        "bundle_id": _text(policy.get("new_bundle_id"), limit=160),
        "active_target": active,
        "policy_active_target": active,
        "observed_active_tasks": None,
        "observed_task_status_counts": None,
        "target_semantics": "policy_ceiling_not_observed_running",
        "cpus_per_task": cpus,
        "memory_mb_per_task": memory_mb,
        "requested_cpus": active * cpus,
        "requested_memory_mb": active * memory_mb,
        "max_workers_per_node": workers,
        "global_unplaced_limit": _integer(policy.get("global_unplaced_limit")),
        "priority": _integer(contract.get("priority")),
        "stop_requested": False,
        "takeover_status": _text(takeover.get("status")),
        "revision": _integer(payload.get("revision")),
        "scheduler_submit_count": _integer(payload.get("scheduler_submit_count")),
        "source": str(path),
    }


def _cross_bundle_v2_resources(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    memory_policy = _mapping(payload.get("memory_cohort_policy"))
    observed_policy_schema = _text(memory_policy.get("schema_version"), limit=160)
    if observed_policy_schema != CURRENT7_MEMORY_COHORT_POLICY_SCHEMA_V3:
        return {
            **_unavailable_resources(
                path,
                "unsupported Current7 memory cohort policy schema: "
                f"{observed_policy_schema or 'missing'}",
            ),
            "schema_version": CURRENT7_CROSS_BUNDLE_SCHEMA_V2,
            "memory_cohort_policy_schema": observed_policy_schema,
            "integrity_verified": False,
            "policy_integrity_verified": False,
        }
    if not _canonical_sha256_verified(memory_policy, "cohort_policy_sha256"):
        return {
            **_unavailable_resources(
                path, "Current7 memory cohort policy integrity check failed"
            ),
            "schema_version": CURRENT7_CROSS_BUNDLE_SCHEMA_V2,
            "memory_cohort_policy_schema": observed_policy_schema,
            "integrity_verified": False,
            "policy_integrity_verified": False,
        }

    contract = _mapping(memory_policy.get("successor_resource_contract"))
    contract_integrity = _canonical_sha256_matches(
        contract, memory_policy.get("successor_resource_contract_sha256")
    )
    scale_policy = _mapping(memory_policy.get("successor_scale_policy"))
    unsigned_scale_policy = {
        key: value for key, value in scale_policy.items() if key != "policy_sha256"
    }
    scale_policy_sha = memory_policy.get("successor_scale_policy_sha256")
    scale_policy_integrity = bool(
        _canonical_sha256_matches(unsigned_scale_policy, scale_policy_sha)
        and scale_policy.get("policy_sha256") == scale_policy_sha
    )
    exact = _mapping(memory_policy.get("successor_exact_scheduler_resources"))
    takeover = _mapping(payload.get("takeover"))
    active = _integer(memory_policy.get("successor_active_target"))
    cpus = _integer(exact.get("cpus"))
    memory_mb = _integer(exact.get("memory_mb"))
    workers = _integer(exact.get("max_workers_per_node"))
    unplaced = _integer(memory_policy.get("successor_global_unplaced_limit"))
    bundle_id = _text(memory_policy.get("bundle_id"), limit=160)
    predecessor_policy = _mapping(payload.get("policy"))
    contract_consistent = bool(
        contract_integrity
        and scale_policy_integrity
        and cpus == _integer(contract.get("cpus_per_seed_task"))
        and memory_mb == _integer(contract.get("memory_mb_per_seed_task"))
        and workers == _integer(contract.get("max_workers_per_node"))
        and _integer(exact.get("priority")) == _integer(contract.get("priority"))
        and active == _integer(scale_policy.get("scaled_active_total"))
    )
    contract_valid = bool(
        contract_consistent
        and takeover.get("status") == "successor_committed"
        and bundle_id
        and bundle_id == _text(predecessor_policy.get("new_bundle_id"), limit=160)
        and active
        and cpus
        and memory_mb
        and workers
        and unplaced
        and _text(memory_policy.get("activated_at"))
        and _text(memory_policy.get("successor_cohort"), limit=160)
        and _text(memory_policy.get("successor_resource_policy"), limit=160)
        and memory_policy.get("physical_allocation_admission_remains_authoritative")
        is True
        and memory_policy.get("successor_unplaced_requires_known_physical_node") is True
        and memory_policy.get("unknown_target_global_fail_closed") is True
    )
    if not contract_valid:
        return {
            **_unavailable_resources(
                path, "Current7 v2 successor resource contract is incomplete"
            ),
            "schema_version": CURRENT7_CROSS_BUNDLE_SCHEMA_V2,
            "memory_cohort_policy_schema": observed_policy_schema,
            "integrity_verified": False,
            "policy_integrity_verified": True,
            "resource_contract_integrity_verified": contract_integrity,
            "scale_policy_integrity_verified": scale_policy_integrity,
        }
    return {
        "configured": True,
        "available": True,
        "schema_version": CURRENT7_CROSS_BUNDLE_SCHEMA_V2,
        "adapter_version": "cross_bundle_v2_memory_cohort_v3",
        "memory_cohort_policy_schema": observed_policy_schema,
        "integrity_verified": True,
        "policy_integrity_verified": True,
        "resource_contract_integrity_verified": True,
        "scale_policy_integrity_verified": True,
        "bundle_id": bundle_id,
        "active_target": active,
        "policy_active_target": active,
        "observed_active_tasks": None,
        "observed_task_status_counts": None,
        "target_semantics": "activated_successor_target_not_observed_running",
        "cpus_per_task": cpus,
        "memory_mb_per_task": memory_mb,
        "requested_cpus": active * cpus,
        "requested_memory_mb": active * memory_mb,
        "max_workers_per_node": workers,
        "global_unplaced_limit": unplaced,
        "priority": _integer(exact.get("priority")),
        "resource_cohort": _text(memory_policy.get("successor_cohort"), limit=160),
        "resource_policy": _text(
            memory_policy.get("successor_resource_policy"), limit=160
        ),
        "activated_at": _text(memory_policy.get("activated_at")),
        "target_nonbinding": memory_policy.get("successor_target_is_nonbinding")
        is True,
        "stop_requested": False,
        "takeover_status": _text(takeover.get("status")),
        "revision": _integer(payload.get("revision")),
        "scheduler_submit_count": _integer(payload.get("scheduler_submit_count")),
        "source": str(path),
    }


def _cross_bundle_current7_resources(
    payload: dict[str, Any], path: Path
) -> dict[str, Any]:
    schema = _text(payload.get("schema_version"), limit=160)
    if not _canonical_sha256_verified(payload, "state_sha256"):
        return {
            **_unavailable_resources(path, "controller integrity check failed"),
            "schema_version": schema,
            "integrity_verified": False,
        }
    if schema == CURRENT7_CROSS_BUNDLE_SCHEMA_V1:
        return _cross_bundle_v1_resources(payload, path)
    if schema == CURRENT7_CROSS_BUNDLE_SCHEMA_V2:
        return _cross_bundle_v2_resources(payload, path)
    return {
        **_unavailable_resources(
            path, f"unsupported Current7 controller schema: {schema or 'missing'}"
        ),
        "schema_version": schema,
        "integrity_verified": False,
    }


def _current7_resources() -> dict[str, Any]:
    path = _configured_path(CURRENT7_CONTROLLER_STATE_ENV)
    if path is None:
        return {"configured": False, "available": False}
    payload, error = _read_status(path)
    if payload is None:
        return _unavailable_resources(path, error or "controller state is unavailable")
    schema = payload.get("schema_version")
    if schema == CURRENT7_CONTROLLER_SCHEMA:
        return _legacy_current7_resources(payload, path)
    if schema in {CURRENT7_CROSS_BUNDLE_SCHEMA_V1, CURRENT7_CROSS_BUNDLE_SCHEMA_V2}:
        return _cross_bundle_current7_resources(payload, path)
    observed = _text(schema, limit=160) or "missing"
    return {
        **_unavailable_resources(
            path, f"unsupported Current7 controller schema: {observed}"
        ),
        "schema_version": None if observed == "missing" else observed,
        "integrity_verified": False,
    }


class ComputeCampaignStatusReader:
    """Build the read-only status shown by the 8010 operator dashboard."""

    def snapshot(self) -> dict[str, Any]:
        stages = {
            "hpo": _hpo_stage(),
            "acceptance": _acceptance_stage(),
            "harvest": _harvest_stage(),
            "cutover": _cutover_stage(),
        }
        resources = _current7_resources()
        warnings = [
            f"{stage['label']}: {stage['error']}"
            for stage in stages.values()
            if stage.get("configured") is True
            and stage.get("available") is not True
            and stage.get("error")
        ]
        if (
            resources.get("configured") is True
            and resources.get("available") is not True
            and resources.get("error")
        ):
            warnings.append(f"Current7 controller resources: {resources['error']}")
        advisories = [
            f"{stage['label']}: {stage['advisory']}"
            for stage in stages.values()
            if stage.get("advisory")
        ]
        return {
            "schema_version": "mft-monitor-compute-campaign-v1",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "available": any(
                stage.get("available") is True for stage in stages.values()
            ),
            "resources": resources,
            "stages": stages,
            "warnings": warnings,
            "advisories": advisories,
            "read_only": True,
        }
