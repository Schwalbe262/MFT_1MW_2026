"""Exact-once n116 successor for one unsubmitted operational recovery plan.

The source and successor remain diagnostic-only.  Geometry, losses, fan speed,
TIM, pads, solver/library revisions, profile parameters, and resource shape
are byte-for-byte inherited from an authenticated timeout-anchor dependency
plan.  Only operational identity and strict placement move to n116.

This module belongs to the MFT project.  It reads Scheduler APIs and consumes
the pinned cutover receipt, but never edits the separate Scheduler repository.
The submit command is the only code path allowed to POST, and it is guarded by
an independent atomic claim, a cross-generation nonterminal-sibling scan, a
fresh dhj02 GPFS audit, and a locked pre-POST revalidation.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Mapping, Sequence

from module.mft_goal_20260726_contract import (
    GOAL_CONTRACT_SCHEMA,
    GOAL_STAGE_SPEC,
    GOAL_STAGE_SPEC_SHA256,
    GOAL_TEMPERATURE_CONTRACT_SHA256,
    canonical_sha256,
)
from regression_260707.verify import scheduler_client
from tools import mft_campaign_atomic_claim as atomic_claim
from tools import mft_goal_dependency_failure_retry as dependency
from tools import mft_goal_diagnostic_standard_probe as probe
from tools import mft_goal_fea_handoff as production


PLAN_SCHEMA = "mft-goal-diagnostic-n116-operational-successor-plan-v1"
SUBMISSION_SCHEMA = (
    "mft-goal-diagnostic-n116-operational-successor-submission-v1"
)
STORAGE_AUDIT_SCHEMA = (
    "mft-goal-diagnostic-n116-operational-storage-audit-v1"
)
INVENTORY_SCHEMA = (
    "mft-goal-diagnostic-n116-operational-sibling-inventory-v1"
)
CLAIM_RECEIPT_SCHEMA = (
    "mft-goal-n116-operational-successor-atomic-claim-receipt-v1"
)
RETRY_GENERATION = "n116-operational-successor-r4"
REQUIRED_NODE = "n116"
REQUIRED_ACCOUNT = "dhj02"
STORAGE_FLOOR_GB = 10.0
PROJECT_RESERVATION_GB = 4.0
MAX_STORAGE_AUDIT_AGE_SECONDS = 300
RESOURCES = {"cpus": 8, "timeout_seconds": 8 * 3600}
CLAIM_ROOT = Path(
    "C:/Users/peets/slurm_scheduler_runtime/mft_goal_20260726/"
    "n116_operational_successor_claims_r4"
)
CLAIM_AUTHORITY_SHA256 = canonical_sha256(
    {
        "campaign_id": "mft-goal-20260726",
        "goal_contract_schema": GOAL_CONTRACT_SCHEMA,
        "hard_spec_sha256": GOAL_STAGE_SPEC_SHA256,
        "temperature_contract_sha256": GOAL_TEMPERATURE_CONTRACT_SHA256,
        "retry_generation": RETRY_GENERATION,
        "strict_node": REQUIRED_NODE,
        "required_account": REQUIRED_ACCOUNT,
        "minimum_storage_floor_gb": STORAGE_FLOOR_GB,
    }
)
TERMINAL_RETRYABLE_STATUSES = frozenset({"failed", "cancelled"})
POST_RECONCILIATION_READS = 8
POST_RECONCILIATION_INTERVAL_SECONDS = 0.25
HandoffContractError = production.HandoffContractError


def _flags() -> dict[str, bool]:
    return {
        "diagnostic_only": True,
        "standard_only": True,
        "production_eligible": False,
        "automatic_promotion": False,
        "full_submission_allowed": False,
        "production_package_allowed": False,
    }


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HandoffContractError(f"{label} is invalid")
    return value


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HandoffContractError(f"{label} is invalid")
    result = float(value)
    if result < 0 or result != result or result in {float("inf"), -float("inf")}:
        raise HandoffContractError(f"{label} is invalid")
    return result


def _task_id(row: Mapping[str, Any]) -> Any:
    return row.get("task_id", row.get("id"))


def _node_name(row: Mapping[str, Any]) -> str:
    return str(
        row.get("actual_node_name")
        or row.get("node_name")
        or row.get("requested_node_name")
        or ""
    ).strip()


def _status(row: Mapping[str, Any]) -> str:
    return str(row.get("status") or row.get("state") or "").strip().lower()


def _normalize_task(row: Mapping[str, Any]) -> dict[str, Any]:
    task_id = _positive_int(_task_id(row), "Scheduler task ID")
    return {
        "task_id": task_id,
        "name": str(row.get("name") or ""),
        "status": _status(row),
        "state": str(row.get("state") or "").strip().lower(),
        "project": str(row.get("project") or ""),
        "dedupe_key": str(row.get("dedupe_key") or ""),
        "cpus": row.get("cpus"),
        "memory_mb": row.get("memory_mb"),
        "timeout_seconds": row.get("timeout_seconds"),
        "aedt_backend": str(row.get("aedt_backend") or ""),
        "account_name": str(
            row.get("account_name")
            or row.get("requested_account_name")
            or ""
        ),
        "node_name": _node_name(row),
        "requested_node_name": str(
            row.get("requested_node_name") or ""
        ),
        "node_name_policy": str(
            row.get("requested_node_name_policy")
            or row.get("node_name_policy")
            or ""
        ),
        "allocation_id": row.get(
            "allocation_id", row.get("assigned_allocation")
        ),
        "slurm_job_id": str(row.get("slurm_job_id") or ""),
        "created_at": row.get("created_at"),
        "started_at": row.get("started_at"),
        "finished_at": row.get("finished_at"),
    }


def _scheduler_candidate_tasks(
    *,
    scheduler_url: str,
    project: str,
    task_name: str,
) -> list[dict[str, Any]]:
    # Request the complete bounded diagnostic namespace; local filtering then
    # catches every prior operational generation for the candidate stem.
    del task_name
    return probe._scheduler_project_tasks(
        scheduler_url=scheduler_url,
        project=project,
        task_name="mft-goal-diag-standard",
    )


def _candidate_inventory(
    rows: Any,
    *,
    candidate_physics_sha256: str,
    successor_task_name: str,
    successor_dedupe_key: str,
    source_task_name: str,
) -> dict[str, Any]:
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes, bytearray))
    ):
        raise HandoffContractError(
            "n116 successor Scheduler task inventory is absent"
        )
    candidate_sha = production._require_sha(
        candidate_physics_sha256, "n116 successor candidate physics SHA"
    )
    stem = candidate_sha[:12]
    matching = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        name = str(raw.get("name") or "")
        dedupe = str(raw.get("dedupe_key") or "")
        if (
            stem not in name
            and name != successor_task_name
            and dedupe != successor_dedupe_key
        ):
            continue
        normalized = _normalize_task(raw)
        if normalized["project"] != scheduler_client.MFT_PROJECT:
            raise HandoffContractError(
                "n116 successor candidate inventory crosses project boundary"
            )
        matching.append(normalized)
    matching.sort(key=lambda row: row["task_id"])
    own = [
        row
        for row in matching
        if row["name"] == successor_task_name
        or row["dedupe_key"] == successor_dedupe_key
    ]
    source = [
        row for row in matching if row["name"] == source_task_name
    ]
    blocking = [
        row
        for row in matching
        if row["status"] not in TERMINAL_RETRYABLE_STATUSES
    ]
    cross_generation_blocking = [
        row for row in blocking if row not in own
    ]
    if len(own) > 1:
        raise HandoffContractError(
            "more than one n116 operational successor sibling exists"
        )
    unsigned = {
        "schema_version": INVENTORY_SCHEMA,
        "candidate_physics_sha256": candidate_sha,
        "candidate_name_stem": stem,
        "successor_task_name": successor_task_name,
        "successor_dedupe_key": successor_dedupe_key,
        "source_task_name": source_task_name,
        "matching_task_count": len(matching),
        "matching_tasks": matching,
        "own_successor_task_count": len(own),
        "own_successor_tasks": own,
        "source_plan_task_count": len(source),
        "source_plan_tasks": source,
        "blocking_task_count": len(blocking),
        "blocking_tasks": blocking,
        "cross_generation_blocking_task_count": len(
            cross_generation_blocking
        ),
        "cross_generation_blocking_tasks": cross_generation_blocking,
    }
    return {**unsigned, "snapshot_sha256": canonical_sha256(unsigned)}


def _validate_inventory(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HandoffContractError(
            "n116 successor candidate inventory is malformed"
        )
    expected = _candidate_inventory(
        value.get("matching_tasks"),
        candidate_physics_sha256=str(
            value.get("candidate_physics_sha256") or ""
        ),
        successor_task_name=str(value.get("successor_task_name") or ""),
        successor_dedupe_key=str(
            value.get("successor_dedupe_key") or ""
        ),
        source_task_name=str(value.get("source_task_name") or ""),
    )
    if expected != dict(value):
        raise HandoffContractError(
            "n116 successor candidate inventory drifted"
        )
    return expected


def _running_n116_inventory(rows: Any) -> list[dict[str, Any]]:
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes, bytearray))
    ):
        raise HandoffContractError(
            "n116 running-task inventory is absent"
        )
    running = []
    for raw in rows:
        if not isinstance(raw, Mapping) or _status(raw) != "running":
            continue
        normalized = _normalize_task(raw)
        if (
            normalized["project"] == scheduler_client.MFT_PROJECT
            and normalized["node_name"] == REQUIRED_NODE
            and normalized["account_name"] == REQUIRED_ACCOUNT
            and normalized["aedt_backend"] == "standalone"
        ):
            if (
                normalized["cpus"] != RESOURCES["cpus"]
                or normalized["memory_mb"] != 32768
            ):
                raise HandoffContractError(
                    "n116 running FEA resource identity drifted"
                )
            running.append(normalized)
    running.sort(key=lambda row: row["task_id"])
    return running


def build_storage_audit(
    *,
    observed_at_kst: str,
    filesystem_type: str,
    fileset_name: str,
    user_quota_scope: str,
    quota_type: str,
    block_used_gb: float,
    block_in_doubt_gb: float,
    block_limit_gb: float,
    rows: Any,
) -> dict[str, Any]:
    try:
        observed = datetime.fromisoformat(observed_at_kst)
    except (TypeError, ValueError) as exc:
        raise HandoffContractError(
            "n116 storage observation timestamp is invalid"
        ) from exc
    if (
        observed.utcoffset() is None
        or observed.utcoffset().total_seconds() != 9 * 3600
        or filesystem_type.lower() != "gpfs"
        or not str(fileset_name or "").strip()
        or user_quota_scope not in {"filesystem", "fileset"}
        or quota_type not in {"USR", "FILESET"}
    ):
        raise HandoffContractError(
            "n116 storage quota identity is invalid"
        )
    used = _finite_nonnegative(block_used_gb, "GPFS used GiB")
    in_doubt = _finite_nonnegative(
        block_in_doubt_gb, "GPFS in-doubt GiB"
    )
    limit = _finite_nonnegative(block_limit_gb, "GPFS limit GiB")
    if limit <= 0:
        raise HandoffContractError("GPFS limit GiB is invalid")
    effective_used = used + max(0.0, in_doubt)
    observed_free = limit - effective_used
    running = _running_n116_inventory(rows)
    running_reservation = len(running) * PROJECT_RESERVATION_GB
    candidate_reservation = PROJECT_RESERVATION_GB
    free_after = (
        observed_free - running_reservation - candidate_reservation
    )
    if free_after < STORAGE_FLOOR_GB:
        raise HandoffContractError(
            "n116 successor storage audit is below the 10 GiB floor"
        )
    unsigned = {
        "schema_version": STORAGE_AUDIT_SCHEMA,
        "observed_at_kst": observed.isoformat(),
        "source": "read_only_gpfs_mmlsquota_y",
        "account_name": REQUIRED_ACCOUNT,
        "node_name": REQUIRED_NODE,
        "filesystem_type": filesystem_type.lower(),
        "fileset_name": str(fileset_name),
        "user_quota_scope": user_quota_scope,
        "quota_type": quota_type,
        "block_used_gb": used,
        "block_in_doubt_gb": in_doubt,
        "block_limit_gb": limit,
        "effective_used_gb": effective_used,
        "observed_free_gb": observed_free,
        "running_n116_task_count": len(running),
        "running_n116_tasks": running,
        "reservation_per_project_gb": PROJECT_RESERVATION_GB,
        "running_prospective_output_reservation_gb": (
            running_reservation
        ),
        "candidate_prospective_output_reservation_gb": (
            candidate_reservation
        ),
        "free_after_running_and_candidate_gb": free_after,
        "minimum_free_floor_gb": STORAGE_FLOOR_GB,
        "arithmetic_passed": True,
        "fresh_pre_submit_reauthentication_required": True,
        "parallel_submit_without_reaudit_allowed": False,
    }
    return production._seal(unsigned)


def _load_storage_audit(
    path: Path,
    *,
    require_fresh: bool,
    now: datetime | None = None,
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    audit = production._validate_seal(
        production._read_json(resolved), STORAGE_AUDIT_SCHEMA
    )
    running = audit.get("running_n116_tasks")
    rebuilt = build_storage_audit(
        observed_at_kst=str(audit.get("observed_at_kst") or ""),
        filesystem_type=str(audit.get("filesystem_type") or ""),
        fileset_name=str(audit.get("fileset_name") or ""),
        user_quota_scope=str(audit.get("user_quota_scope") or ""),
        quota_type=str(audit.get("quota_type") or ""),
        block_used_gb=audit.get("block_used_gb"),
        block_in_doubt_gb=audit.get("block_in_doubt_gb"),
        block_limit_gb=audit.get("block_limit_gb"),
        rows=running,
    )
    if rebuilt != audit:
        raise HandoffContractError("n116 storage audit bytes drifted")
    if require_fresh:
        observed = datetime.fromisoformat(audit["observed_at_kst"])
        current = now or datetime.now(timezone.utc)
        age = (current.astimezone(timezone.utc) - observed.astimezone(
            timezone.utc
        )).total_seconds()
        if age < -30 or age > MAX_STORAGE_AUDIT_AGE_SECONDS:
            raise HandoffContractError(
                "n116 storage audit is stale for submission"
            )
    return audit


def initialize_claim_root(root: Path | None = None) -> dict[str, Any]:
    try:
        return atomic_claim.initialize_claim_root(
            CLAIM_ROOT if root is None else Path(root),
            campaign_id="mft-goal-20260726",
            campaign_authority_sha256=CLAIM_AUTHORITY_SHA256,
        )
    except atomic_claim.ClaimContractError as exc:
        raise HandoffContractError(
            "n116 successor atomic claim root is unavailable"
        ) from exc


def _claim_authority() -> dict[str, Any]:
    try:
        authority = atomic_claim.load_claim_root(CLAIM_ROOT)
    except atomic_claim.ClaimContractError as exc:
        raise HandoffContractError(
            "n116 successor atomic claim root is unavailable"
        ) from exc
    if (
        authority.get("campaign_id") != "mft-goal-20260726"
        or authority.get("campaign_authority_sha256")
        != CLAIM_AUTHORITY_SHA256
    ):
        raise HandoffContractError(
            "n116 successor atomic claim authority drifted"
        )
    return authority


def _claim_reference(
    *,
    logical_authority_task_id: int,
    candidate_physics_sha256: str,
) -> dict[str, Any]:
    try:
        return atomic_claim.build_claim_reference(
            _claim_authority(),
            candidate_physics_sha256=candidate_physics_sha256,
            logical_authority_task_id=logical_authority_task_id,
            retry_generation=RETRY_GENERATION,
        )
    except atomic_claim.ClaimContractError as exc:
        raise HandoffContractError(
            "n116 successor atomic claim reference is invalid"
        ) from exc


def _validate_claim_reference(plan: Mapping[str, Any]) -> dict[str, Any]:
    value = plan.get("operational_successor_atomic_claim_reference")
    record = plan.get("operational_successor")
    if not isinstance(value, Mapping) or not isinstance(record, Mapping):
        raise HandoffContractError(
            "n116 successor atomic claim reference is absent"
        )
    try:
        normalized = atomic_claim.validate_claim_reference(
            value, _claim_authority()
        )
    except atomic_claim.ClaimContractError as exc:
        raise HandoffContractError(
            "n116 successor atomic claim reference drifted"
        ) from exc
    if (
        normalized.get("candidate_physics_sha256")
        != plan.get("candidate_physics_sha256")
        or normalized.get("logical_authority_task_id")
        != record.get("logical_authority_task_id")
        or normalized.get("retry_generation") != RETRY_GENERATION
    ):
        raise HandoffContractError(
            "n116 successor atomic claim binding drifted"
        )
    return normalized


def _task_identity(
    *, logical_authority_task_id: int, candidate_physics_sha256: str
) -> tuple[str, str]:
    logical = _positive_int(
        logical_authority_task_id, "n116 successor logical task ID"
    )
    stem = production._require_sha(
        candidate_physics_sha256, "n116 successor candidate SHA"
    )[:12]
    return (
        f"mft-goal-diag-standard-n116-r4-l{logical}-{stem}",
        f"mft_goal_diag_standard_n116_r4_l{logical}_{stem}",
    )


def create_plan(
    *,
    source_plan_path: Path,
    storage_audit_path: Path,
    scheduler_cutover_receipt_path: Path,
    output: Path,
    task_list_reader: Any = None,
) -> Path:
    (
        source,
        params,
        selected,
        _immediate_submission,
        _expected_anchor,
    ) = dependency._load_plan(source_plan_path)
    if dependency._policy_for_plan(source) is not dependency.R3_POLICY:
        raise HandoffContractError(
            "n116 successor requires an unsubmitted timeout-anchor r3 plan"
        )
    source_stage = source["stage"]
    logical_id = _positive_int(
        source["retry_of_dependency_failure"][
            "logical_authority_task_id"
        ],
        "n116 successor logical authority task ID",
    )
    task_name, workdir = _task_identity(
        logical_authority_task_id=logical_id,
        candidate_physics_sha256=source["candidate_physics_sha256"],
    )
    source_profile = production._read_json(
        source_plan_path.resolve(strict=True).parent
        / source["profile"]["path"]
    )
    dependency._validate_profile(
        source_profile, policy=dependency.R3_POLICY
    )
    strict = probe._strict_node_plan_contract(
        REQUIRED_NODE,
        task_identity_generation=(
            probe.N116_OPERATIONAL_SUCCESSOR_STRICT_TASK_IDENTITY_GENERATION
        ),
    )
    cutover, _unused = probe._validate_scheduler_cutover_receipt(
        scheduler_cutover_receipt_path,
        verify_live_launcher=False,
        require_strict_node=True,
        strict_node_contract=strict,
        require_active_strict=True,
    )
    retained = scheduler_client.retained_aedt_identity(
        task_name,
        params,
        source_profile,
        source["solver_revision"],
        source["library_revision"],
    )
    if (
        retained is None
        or retained["dedupe_key"]
        == source_stage["retained_aedt_bundle"]["dedupe_key"]
        or retained["relative_directory"]
        == source_stage["retained_aedt_bundle"]["relative_directory"]
    ):
        raise HandoffContractError(
            "n116 successor retained identity is not distinct"
        )
    audit = _load_storage_audit(
        storage_audit_path, require_fresh=False
    )
    if audit["running_n116_task_count"] != 3:
        raise HandoffContractError(
            "initial n116 successor plan requires the observed running3 set"
        )
    read_tasks = task_list_reader or _scheduler_candidate_tasks
    rows = read_tasks(
        scheduler_url=probe.DIAGNOSTIC_SCHEDULER_URL,
        project=scheduler_client.MFT_PROJECT,
        task_name=source["candidate_physics_sha256"][:12],
    )
    inventory = _candidate_inventory(
        rows,
        candidate_physics_sha256=source["candidate_physics_sha256"],
        successor_task_name=task_name,
        successor_dedupe_key=retained["dedupe_key"],
        source_task_name=source_stage["task_name"],
    )
    if (
        inventory["blocking_task_count"] != 0
        or inventory["own_successor_task_count"] != 0
        or inventory["source_plan_task_count"] != 0
    ):
        raise HandoffContractError(
            "n116 successor source is submitted or has a nonterminal sibling"
        )
    reference = _claim_reference(
        logical_authority_task_id=logical_id,
        candidate_physics_sha256=source["candidate_physics_sha256"],
    )
    destination = output.resolve()
    if destination.exists():
        raise HandoffContractError(
            f"n116 successor plan output exists: {destination}"
        )
    staging = destination.with_name(
        f".{destination.name}.{os.getpid()}."
        f"{next(tempfile._get_candidate_names())}.tmp"
    )
    staging.mkdir(parents=True)
    try:
        selected_path = production._write_immutable_json(
            staging / "selected_candidate.json", selected
        )
        params_path = production._write_immutable_json(
            staging / "fea_params.json", params
        )
        profile_path = production._write_immutable_json(
            staging
            / "diagnostic_standard_n116_operational_successor_profile.json",
            source_profile,
        )
        unsigned = {
            key: copy.deepcopy(value)
            for key, value in source.items()
            if key
            not in {
                "payload_sha256",
                "schema_version",
                "available_submission_commands",
                "retry_of_dependency_failure",
                "dependency_failure_atomic_claim_reference",
                "scheduler_strict_node_contract",
                "stage",
                "selected_candidate",
                "fea_params",
                "profile",
            }
        }
        unsigned.update(
            {
                "schema_version": PLAN_SCHEMA,
                "selected_candidate": {
                    "path": selected_path.name,
                    "sha256": production._sha256_file(selected_path),
                },
                "fea_params": {
                    "path": params_path.name,
                    "sha256": production._sha256_file(params_path),
                },
                "profile": {
                    "path": profile_path.name,
                    "sha256": production._sha256_file(profile_path),
                    "canonical_sha256": canonical_sha256(source_profile),
                    "source_plan_profile": copy.deepcopy(source["profile"]),
                },
                "stage": {
                    **copy.deepcopy(source_stage),
                    "task_name": task_name,
                    "workdir": workdir,
                    "retained_aedt_bundle": retained,
                    "retention_run_root": (
                        probe._retention_run_root_evidence(retained)
                    ),
                },
                "available_submission_commands": [
                    "submit-n116-operational-successor"
                ],
                "operational_successor": {
                    "schema_version": (
                        "mft-goal-n116-operational-successor-lineage-v1"
                    ),
                    "retry_generation": RETRY_GENERATION,
                    "source_retry_generation": (
                        dependency.R3_RETRY_GENERATION
                    ),
                    "logical_authority_task_id": logical_id,
                    "immediate_failed_task_id": source[
                        "retry_of_dependency_failure"
                    ]["retry_of_task_id"],
                    "dependency_anchor_task_id": source[
                        "retry_of_dependency_failure"
                    ]["dependency_anchor_task_id"],
                    "source_plan": production._file_record(
                        source_plan_path.resolve(strict=True)
                    ),
                    "source_plan_payload_sha256": source["payload_sha256"],
                    "source_task_name": source_stage["task_name"],
                    "source_requested_node_name": (
                        source["scheduler_strict_node_contract"][
                            "requested_node_name"
                        ]
                    ),
                    "target_requested_node_name": REQUIRED_NODE,
                    "operational_changes_only": [
                        "retry_generation",
                        "task_name",
                        "workdir",
                        "retained_aedt_identity",
                        "strict_node_name",
                        "atomic_claim_root",
                    ],
                    "fixed_physics_unchanged": True,
                    "geometry_unchanged": True,
                    "losses_unchanged": True,
                    "fan_1p5_mps_unchanged": True,
                    "tim_and_pads_unchanged": True,
                },
                "source_retry_of_dependency_failure": copy.deepcopy(
                    source["retry_of_dependency_failure"]
                ),
                "operational_successor_atomic_claim_reference": reference,
                "scheduler_strict_node_contract": strict,
                "scheduler_cutover_receipt": production._file_record(
                    scheduler_cutover_receipt_path.resolve(strict=True)
                ),
                "scheduler_cutover_payload_sha256": cutover[
                    "payload_sha256"
                ],
                "initial_storage_audit": {
                    "record": production._file_record(
                        storage_audit_path.resolve(strict=True)
                    ),
                    "payload": audit,
                },
                "initial_candidate_sibling_inventory": inventory,
                "fresh_pre_submit_storage_reauthentication_required": True,
                "cross_generation_nonterminal_sibling_guard_required": True,
                "physics_override_allowed": False,
                "scheduler_repository_modified": False,
                "scheduler_project_mutation_performed": False,
                "scheduler_submission_performed": False,
                **_flags(),
            }
        )
        plan_path = production._write_immutable_json(
            staging / "diagnostic_n116_operational_successor_plan.json",
            production._seal(unsigned),
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / plan_path.name


def _artifact(root: Path, record: Any, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise HandoffContractError(f"{label} record is absent")
    target = production._contained_file(root, record.get("path"), label)
    if production._sha256_file(target) != production._require_sha(
        record.get("sha256"), f"{label} SHA"
    ):
        raise HandoffContractError(f"{label} bytes drifted")
    return target


def load_plan(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    resolved = path.resolve(strict=True)
    plan = production._validate_seal(
        production._read_json(resolved), PLAN_SCHEMA
    )
    if (
        plan.get("campaign_id") != "mft-goal-20260726"
        or plan.get("goal_contract_schema") != GOAL_CONTRACT_SCHEMA
        or plan.get("hard_spec") != GOAL_STAGE_SPEC
        or plan.get("hard_spec_sha256") != GOAL_STAGE_SPEC_SHA256
        or plan.get("temperature_contract_sha256")
        != GOAL_TEMPERATURE_CONTRACT_SHA256
        or plan.get("available_submission_commands")
        != ["submit-n116-operational-successor"]
        or plan.get("physics_override_allowed") is not False
        or plan.get("scheduler_repository_modified") is not False
        or plan.get("scheduler_project_mutation_performed") is not False
        or plan.get("scheduler_submission_performed") is not False
        or plan.get(
            "fresh_pre_submit_storage_reauthentication_required"
        )
        is not True
        or plan.get(
            "cross_generation_nonterminal_sibling_guard_required"
        )
        is not True
        or any(
            plan.get(name) is not expected
            for name, expected in _flags().items()
        )
    ):
        raise HandoffContractError(
            "n116 successor plan contract drifted"
        )
    lineage = plan.get("operational_successor")
    if (
        not isinstance(lineage, Mapping)
        or lineage.get("retry_generation") != RETRY_GENERATION
        or lineage.get("source_retry_generation")
        != dependency.R3_RETRY_GENERATION
        or lineage.get("source_requested_node_name")
        != dependency.REQUIRED_STRICT_NODE_NAME
        or lineage.get("target_requested_node_name") != REQUIRED_NODE
        or any(
            lineage.get(name) is not True
            for name in (
                "fixed_physics_unchanged",
                "geometry_unchanged",
                "losses_unchanged",
                "fan_1p5_mps_unchanged",
                "tim_and_pads_unchanged",
            )
        )
    ):
        raise HandoffContractError(
            "n116 successor operational lineage drifted"
        )
    source_path = probe._recorded_external_file(
        lineage.get("source_plan"), "n116 successor source plan"
    )
    (
        source,
        source_params,
        source_selected,
        _source_submission,
        _source_anchor,
    ) = dependency._load_plan(source_path)
    if (
        dependency._policy_for_plan(source) is not dependency.R3_POLICY
        or lineage.get("source_plan_payload_sha256")
        != source["payload_sha256"]
        or lineage.get("logical_authority_task_id")
        != source["retry_of_dependency_failure"][
            "logical_authority_task_id"
        ]
        or lineage.get("immediate_failed_task_id")
        != source["retry_of_dependency_failure"]["retry_of_task_id"]
        or lineage.get("dependency_anchor_task_id")
        != source["retry_of_dependency_failure"][
            "dependency_anchor_task_id"
        ]
        or lineage.get("source_task_name") != source["stage"]["task_name"]
        or plan.get("source_retry_of_dependency_failure")
        != source["retry_of_dependency_failure"]
        or any(
            plan.get(field) != source.get(field)
            for field in (
                "campaign_id",
                "goal_contract_schema",
                "hard_spec",
                "hard_spec_sha256",
                "temperature_contract_sha256",
                "solver_revision",
                "library_revision",
                "candidate_physics_sha256",
                "search_authority_sha256",
                "fea_params_sha256",
            )
        )
    ):
        raise HandoffContractError(
            "n116 successor source ancestry drifted"
        )
    root = resolved.parent
    params = production._read_json(
        _artifact(root, plan.get("fea_params"), "n116 successor params")
    )
    selected = production._read_json(
        _artifact(
            root,
            plan.get("selected_candidate"),
            "n116 successor selected candidate",
        )
    )
    profile = production._read_json(
        _artifact(root, plan.get("profile"), "n116 successor profile")
    )
    source_profile = production._read_json(
        source_path.parent / source["profile"]["path"]
    )
    if (
        params != source_params
        or selected != source_selected
        or profile != source_profile
        or plan["profile"].get("canonical_sha256")
        != canonical_sha256(profile)
        or plan["stage"].get("effective_params_sha256")
        != source["stage"]["effective_params_sha256"]
    ):
        raise HandoffContractError(
            "n116 successor changes fixed physics"
        )
    logical = int(lineage["logical_authority_task_id"])
    task_name, workdir = _task_identity(
        logical_authority_task_id=logical,
        candidate_physics_sha256=plan["candidate_physics_sha256"],
    )
    retained = scheduler_client.retained_aedt_identity(
        task_name,
        params,
        profile,
        plan["solver_revision"],
        plan["library_revision"],
    )
    stage = plan.get("stage")
    strict = plan.get("scheduler_strict_node_contract")
    probe._strict_node_scheduler_pin(strict)
    if (
        not isinstance(stage, Mapping)
        or strict.get("requested_node_name") != REQUIRED_NODE
        or strict.get("node_name_policy") != "strict"
        or strict.get("task_identity_generation")
        != probe.N116_OPERATIONAL_SUCCESSOR_STRICT_TASK_IDENTITY_GENERATION
        or stage.get("task_name") != task_name
        or stage.get("workdir") != workdir
        or stage.get("resources") != RESOURCES
        or stage.get("retained_aedt_bundle") != retained
        or stage.get("retention_run_root")
        != probe._retention_run_root_evidence(retained)
        or stage.get("scheduler_project") != scheduler_client.MFT_PROJECT
        or stage.get("scheduler_url")
        != probe.DIAGNOSTIC_SCHEDULER_URL
        or stage.get("aedt_backend") != "standalone"
        or stage.get("full_model") != 0
        or stage.get("thermal_symmetry") != "eighth"
    ):
        raise HandoffContractError(
            "n116 successor execution identity drifted"
        )
    cutover_record = plan.get("scheduler_cutover_receipt")
    if not isinstance(cutover_record, Mapping):
        raise HandoffContractError(
            "n116 successor cutover receipt record is absent"
        )
    cutover_path = Path(str(cutover_record.get("path") or ""))
    if production._file_record(cutover_path) != cutover_record:
        raise HandoffContractError(
            "n116 successor cutover receipt bytes drifted"
        )
    cutover, _unused = probe._validate_scheduler_cutover_receipt(
        cutover_path,
        verify_live_launcher=False,
        require_strict_node=True,
        strict_node_contract=strict,
        require_active_strict=True,
    )
    if (
        plan.get("scheduler_cutover_payload_sha256")
        != cutover["payload_sha256"]
    ):
        raise HandoffContractError(
            "n116 successor cutover payload drifted"
        )
    initial = plan.get("initial_storage_audit")
    if not isinstance(initial, Mapping):
        raise HandoffContractError(
            "n116 successor initial storage audit is absent"
        )
    audit_path = Path(str(initial.get("record", {}).get("path") or ""))
    if (
        production._file_record(audit_path) != initial.get("record")
        or _load_storage_audit(audit_path, require_fresh=False)
        != initial.get("payload")
    ):
        raise HandoffContractError(
            "n116 successor initial storage audit drifted"
        )
    initial_inventory = _validate_inventory(
        plan.get("initial_candidate_sibling_inventory")
    )
    if (
        initial_inventory["blocking_task_count"] != 0
        or initial_inventory["own_successor_task_count"] != 0
        or initial_inventory["source_plan_task_count"] != 0
    ):
        raise HandoffContractError(
            "n116 successor initial sibling slot is not empty"
        )
    _validate_claim_reference(plan)
    return plan, params, selected, source


def _live_inventory(
    *,
    plan: Mapping[str, Any],
    rows: Any,
) -> dict[str, Any]:
    return _candidate_inventory(
        rows,
        candidate_physics_sha256=plan["candidate_physics_sha256"],
        successor_task_name=plan["stage"]["task_name"],
        successor_dedupe_key=plan["stage"]["retained_aedt_bundle"][
            "dedupe_key"
        ],
        source_task_name=plan["operational_successor"][
            "source_task_name"
        ],
    )


def _assert_fresh_storage_matches_live(
    audit: Mapping[str, Any], rows: Any
) -> None:
    live = _running_n116_inventory(rows)
    if live != audit.get("running_n116_tasks"):
        raise HandoffContractError(
            "n116 running prospective outputs changed after storage audit"
        )


def _require_ready_capacity(
    *, scheduler_url: str, live_reader: Any
) -> dict[str, Any]:
    value = live_reader(
        scheduler_url=scheduler_url, endpoint="/api/task-capacity"
    )
    if not isinstance(value, Mapping):
        raise HandoffContractError(
            "n116 successor task-capacity response is absent"
        )
    ready = value.get("ready_fit_slots")
    fit = value.get("fit_slots")
    if (
        value.get("memory_pressure_state") != "ok"
        or isinstance(ready, bool)
        or not isinstance(ready, int)
        or ready < 1
        or isinstance(fit, bool)
        or not isinstance(fit, int)
        or fit < 1
    ):
        raise HandoffContractError(
            "n116 successor has no ready Scheduler capacity"
        )
    return copy.deepcopy(dict(value))


def _claim_winner(
    plan_path: Path, plan: Mapping[str, Any]
) -> dict[str, Any]:
    stage = plan["stage"]
    return {
        "plan_payload_sha256": plan["payload_sha256"],
        "plan_file_sha256": production._sha256_file(
            plan_path.resolve(strict=True)
        ),
        "profile_sha256": stage["profile_sha256"],
        "resources": {**RESOURCES, "memory_mb": 32768},
        "task_name": stage["task_name"],
        "dedupe_key": stage["retained_aedt_bundle"]["dedupe_key"],
        "requested_node_name": REQUIRED_NODE,
        "required_account_name": REQUIRED_ACCOUNT,
    }


def _claim_task_evidence(
    task: Mapping[str, Any], pending: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(task, Mapping) or not isinstance(pending, Mapping):
        raise atomic_claim.ClaimContractError(
            "n116 successor claim task evidence is absent"
        )
    row = _normalize_task(task)
    winner = pending.get("winner")
    if (
        not isinstance(winner, Mapping)
        or row["name"] != winner.get("task_name")
        or row["dedupe_key"] != winner.get("dedupe_key")
        or row["project"] != scheduler_client.MFT_PROJECT
        or row["cpus"] != RESOURCES["cpus"]
        or row["memory_mb"] != 32768
        or row["timeout_seconds"] != RESOURCES["timeout_seconds"]
        or row["aedt_backend"] != "standalone"
        or row["requested_node_name"] != REQUIRED_NODE
        or row["node_name_policy"] != "strict"
    ):
        raise atomic_claim.ClaimContractError(
            "n116 successor claim task identity drifted"
        )
    return row


def _read_source_failure(
    *,
    source_plan_path: Path,
    scheduler_url: str,
    task_reader: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    (
        source,
        _params,
        _selected,
        immediate_submission,
        expected_anchor,
    ) = dependency._load_plan(source_plan_path)
    return dependency._read_failure_bundle(
        scheduler_url=scheduler_url,
        immediate_submission=immediate_submission,
        expected_anchor=expected_anchor,
        task_reader=task_reader,
        policy=dependency.R3_POLICY,
    )


def submit(
    *,
    plan_path: Path,
    fresh_storage_audit_path: Path,
    output: Path,
    priority: int = 100,
    scheduler: Any = scheduler_client,
    predictor: Any | None = None,
    task_reader: Any = None,
    task_list_reader: Any = None,
    live_reader: Any = probe._default_scheduler_live_reader,
    reconciliation_waiter: Any = None,
    now: datetime | None = None,
) -> Path:
    target = output.resolve()
    if target.exists():
        raise HandoffContractError(
            f"n116 successor submission output exists: {target}"
        )
    plan, params, selected, source = load_plan(plan_path)
    stage = plan["stage"]
    source_path = Path(
        plan["operational_successor"]["source_plan"]["path"]
    )
    strict = plan["scheduler_strict_node_contract"]
    pin = probe._strict_node_scheduler_pin(strict, require_active=True)
    cutover_path = Path(plan["scheduler_cutover_receipt"]["path"])
    cutover, launcher_before = probe._validate_scheduler_cutover_receipt(
        cutover_path,
        verify_live_launcher=True,
        require_strict_node=True,
        strict_node_contract=strict,
        require_active_strict=True,
    )
    admission = probe._live_scheduler_admission_snapshot(
        scheduler_url=stage["scheduler_url"], reader=live_reader
    )
    capacity = _require_ready_capacity(
        scheduler_url=stage["scheduler_url"], live_reader=live_reader
    )
    launcher_after = probe._live_launcher_identity(
        cutover, expected_sha256=pin["launcher_sha256"]
    )
    if launcher_after != launcher_before:
        raise HandoffContractError(
            "Scheduler launcher changed during n116 successor admission"
        )
    reauthentication = probe._fresh_selection_reauthentication(
        plan=plan, selected=selected, predictor=predictor
    )
    fresh_storage = _load_storage_audit(
        fresh_storage_audit_path, require_fresh=True, now=now
    )
    read_task = task_reader or probe._scheduler_task_snapshot
    read_tasks = task_list_reader or _scheduler_candidate_tasks

    def all_rows() -> list[dict[str, Any]]:
        return read_tasks(
            scheduler_url=stage["scheduler_url"],
            project=scheduler_client.MFT_PROJECT,
            task_name=plan["candidate_physics_sha256"][:12],
        )

    stored_failure = source["retry_of_dependency_failure"]

    def revalidate_source() -> None:
        dependency_evidence, anchor_evidence = _read_source_failure(
            source_plan_path=source_path,
            scheduler_url=stage["scheduler_url"],
            task_reader=read_task,
        )
        if (
            dependency_evidence
            != stored_failure["dependency_failure_evidence"]
            or anchor_evidence
            != stored_failure["anchor_failure_evidence"]
        ):
            raise HandoffContractError(
                "n116 successor source failure evidence changed"
            )

    rows = all_rows()
    _assert_fresh_storage_matches_live(fresh_storage, rows)
    inventory_before = _live_inventory(plan=plan, rows=rows)
    if inventory_before["cross_generation_blocking_task_count"] != 0:
        raise HandoffContractError(
            "n116 successor has an existing queued/submitted sibling"
        )
    revalidate_source()
    reference = _validate_claim_reference(plan)
    winner = _claim_winner(plan_path, plan)
    try:
        acquisition = atomic_claim.acquire_claim(
            CLAIM_ROOT, reference, winner
        )
    except atomic_claim.ClaimContractError as exc:
        raise HandoffContractError(
            "n116 successor atomic claim acquisition failed"
        ) from exc
    claim_status = acquisition["status"]
    own = inventory_before["own_successor_tasks"]
    locked_guard_count = 0

    def locked_pre_submit_guard() -> None:
        nonlocal inventory_before, locked_guard_count
        revalidate_source()
        latest_rows = all_rows()
        _assert_fresh_storage_matches_live(fresh_storage, latest_rows)
        latest = _live_inventory(plan=plan, rows=latest_rows)
        if (
            latest["blocking_task_count"] != 0
            or latest["own_successor_task_count"] != 0
        ):
            raise HandoffContractError(
                "n116 successor locked sibling slot is not empty"
            )
        inventory_before = latest
        # Recheck timestamp under the scheduler client's pre-POST lock.
        _load_storage_audit(
            fresh_storage_audit_path, require_fresh=True, now=now
        )
        _require_ready_capacity(
            scheduler_url=stage["scheduler_url"],
            live_reader=live_reader,
        )
        locked_guard_count += 1

    finalized = None
    if claim_status != "fresh_pending":
        if (
            len(own) != 1
            or inventory_before[
                "cross_generation_blocking_task_count"
            ]
            != 0
        ):
            raise HandoffContractError(
                "n116 successor claim recovery requires exactly one own task"
            )
        recovered = own[0]
        try:
            if claim_status == "existing_pending":
                finalized = atomic_claim.recover_pending_claim(
                    CLAIM_ROOT,
                    reference,
                    acquisition["claim"],
                    matching_tasks=[recovered],
                    sibling_snapshot=inventory_before,
                    evidence_validator=_claim_task_evidence,
                )
            elif claim_status == "existing_finalized":
                finalized = atomic_claim.validate_finalized_claim(
                    CLAIM_ROOT,
                    reference,
                    claim=acquisition["claim"],
                    expected_winner=winner,
                )
                _claim_task_evidence(
                    recovered, finalized["pending_claim"]
                )
            else:
                raise atomic_claim.ClaimContractError(
                    "unsupported n116 successor claim status"
                )
        except atomic_claim.ClaimContractError as exc:
            raise HandoffContractError(
                "n116 successor claim recovery failed closed"
            ) from exc
        submission_result = {
            "task_id": finalized["task_id"],
            "submission_source": "pre_submission_reconciliation",
            "scheduler_mutation_performed": False,
            "api_pre_submission_readback": recovered,
            "api_post_submission_response": None,
        }
        inventory_after = copy.deepcopy(inventory_before)
        _environment, core_policy = production._submission_environment(
            stage="standard",
            solver_revision=plan["solver_revision"],
            license_snapshot_path=None,
        )
    else:
        if inventory_before["blocking_task_count"] != 0:
            raise HandoffContractError(
                "fresh n116 successor claim found an active sibling"
            )
        profile = production._read_json(
            plan_path.resolve(strict=True).parent / plan["profile"]["path"]
        )
        environment, core_policy = production._submission_environment(
            stage="standard",
            solver_revision=plan["solver_revision"],
            license_snapshot_path=None,
        )
        submission_result = scheduler.submit_verification(
            stage["task_name"],
            stage["workdir"],
            params,
            profile,
            mem_mb=int(profile["mem_mb"]),
            cpus=int(profile["cpus"]),
            solver_revision=plan["solver_revision"],
            library_revision=plan["library_revision"],
            priority=priority,
            aedt_backend="standalone",
            submission_env=environment,
            required_project_cap=probe.GOAL_FEA_PROJECT_CAP,
            max_project_active_tasks=probe.GOAL_FEA_PROJECT_CAP,
            scheduler_url=stage["scheduler_url"],
            pre_submit_guard=locked_pre_submit_guard,
            node_name=REQUIRED_NODE,
            node_name_policy="strict",
            return_submission_evidence=True,
        )
        if locked_guard_count != 1:
            raise HandoffContractError(
                "n116 successor submission lacks one locked guard"
            )
        waiter = reconciliation_waiter or (
            lambda: time.sleep(POST_RECONCILIATION_INTERVAL_SECONDS)
        )
        inventory_after = None
        for attempt in range(POST_RECONCILIATION_READS):
            latest = _live_inventory(plan=plan, rows=all_rows())
            if latest["own_successor_task_count"] == 1:
                inventory_after = latest
                break
            if attempt + 1 < POST_RECONCILIATION_READS:
                waiter()
        if inventory_after is None:
            raise HandoffContractError(
                "n116 successor POST did not become durable"
            )
    if not isinstance(submission_result, Mapping):
        raise HandoffContractError(
            "n116 successor Scheduler result is malformed"
        )
    task_id = _positive_int(
        submission_result.get("task_id"),
        "n116 successor submitted task ID",
    )
    if inventory_after["own_successor_task_count"] != 1:
        raise HandoffContractError(
            "n116 successor exactly-one sibling invariant failed"
        )
    if claim_status == "fresh_pending":
        try:
            finalized = atomic_claim.finalize_claim(
                CLAIM_ROOT,
                reference,
                acquisition["claim"],
                task_id=task_id,
                task_readback=inventory_after["own_successor_tasks"][0],
                sibling_snapshot=inventory_after,
                evidence_validator=_claim_task_evidence,
            )
        except atomic_claim.ClaimContractError as exc:
            raise HandoffContractError(
                "n116 successor atomic claim finalization failed"
            ) from exc
    durable_raw = read_task(
        scheduler_url=stage["scheduler_url"], task_id=task_id
    )
    try:
        durable = _claim_task_evidence(
            durable_raw, finalized["pending_claim"]
        )
    except atomic_claim.ClaimContractError as exc:
        raise HandoffContractError(
            "n116 successor durable task identity drifted"
        ) from exc
    receipt = production._seal(
        {
            "schema_version": SUBMISSION_SCHEMA,
            "plan": production._file_record(plan_path.resolve(strict=True)),
            "plan_payload_sha256": plan["payload_sha256"],
            "candidate_physics_sha256": plan[
                "candidate_physics_sha256"
            ],
            "operational_successor": copy.deepcopy(
                plan["operational_successor"]
            ),
            "task_id": task_id,
            "task_name": stage["task_name"],
            "workdir": stage["workdir"],
            "dedupe_key": stage["retained_aedt_bundle"]["dedupe_key"],
            "resources": {**RESOURCES, "memory_mb": 32768},
            "core_policy": core_policy,
            "search_authority_reauthentication": reauthentication,
            "fresh_storage_audit": {
                "record": production._file_record(
                    fresh_storage_audit_path.resolve(strict=True)
                ),
                "payload": fresh_storage,
            },
            "candidate_inventory_before_submission": inventory_before,
            "candidate_inventory_after_submission": inventory_after,
            "atomic_claim": {
                "schema_version": CLAIM_RECEIPT_SCHEMA,
                "acquisition_status": claim_status,
                "fresh_claim_authorized_scheduler_submit_call": (
                    claim_status == "fresh_pending"
                ),
                "finalized_claim": finalized,
            },
            "scheduler_strict_node_contract": {
                "plan_contract": strict,
                "submission_source": submission_result.get(
                    "submission_source"
                ),
                "scheduler_mutation_performed": submission_result.get(
                    "scheduler_mutation_performed"
                ),
                "api_pre_submission_readback": submission_result.get(
                    "api_pre_submission_readback"
                ),
                "api_post_submission_response": submission_result.get(
                    "api_post_submission_response"
                ),
                "api_durable_get_readback": durable,
                "same_node_as_task_id": 0,
                "direct_strict_node_policy_authenticated": True,
            },
            "scheduler_cutover_receipt": production._file_record(
                cutover_path.resolve(strict=True)
            ),
            "scheduler_cutover_payload_sha256": cutover[
                "payload_sha256"
            ],
            "scheduler_live_launcher_identity": launcher_after,
            "scheduler_admission_snapshot": admission,
            "scheduler_capacity_snapshot": capacity,
            "scheduler_url": stage["scheduler_url"],
            "scheduler_project": scheduler_client.MFT_PROJECT,
            "scheduler_project_mutation_performed": False,
            "scheduler_repository_modified": False,
            "scheduler_submission_performed": True,
            "retention_required": True,
            "prune_protection_required": True,
            **_flags(),
        }
    )
    return production._write_immutable_json(target, receipt)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exact-once n116 operational successor"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init-claim-root")
    init.add_argument("--root", type=Path)
    capture = commands.add_parser("capture-storage-audit")
    capture.add_argument("--observed-at-kst", required=True)
    capture.add_argument("--filesystem-type", required=True)
    capture.add_argument("--fileset-name", required=True)
    capture.add_argument("--user-quota-scope", required=True)
    capture.add_argument("--quota-type", required=True)
    capture.add_argument("--block-used-gb", type=float, required=True)
    capture.add_argument("--block-in-doubt-gb", type=float, required=True)
    capture.add_argument("--block-limit-gb", type=float, required=True)
    capture.add_argument("--output", type=Path, required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--source-plan", type=Path, required=True)
    plan.add_argument("--storage-audit", type=Path, required=True)
    plan.add_argument("--scheduler-cutover-receipt", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    submit_parser = commands.add_parser("submit")
    submit_parser.add_argument("--plan", type=Path, required=True)
    submit_parser.add_argument(
        "--fresh-storage-audit", type=Path, required=True
    )
    submit_parser.add_argument("--priority", type=int, default=100)
    submit_parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "init-claim-root":
        result: Any = initialize_claim_root(args.root)
    elif args.command == "capture-storage-audit":
        rows = _scheduler_candidate_tasks(
            scheduler_url=probe.DIAGNOSTIC_SCHEDULER_URL,
            project=scheduler_client.MFT_PROJECT,
            task_name="",
        )
        audit = build_storage_audit(
            observed_at_kst=args.observed_at_kst,
            filesystem_type=args.filesystem_type,
            fileset_name=args.fileset_name,
            user_quota_scope=args.user_quota_scope,
            quota_type=args.quota_type,
            block_used_gb=args.block_used_gb,
            block_in_doubt_gb=args.block_in_doubt_gb,
            block_limit_gb=args.block_limit_gb,
            rows=rows,
        )
        result = production._write_immutable_json(
            args.output.resolve(), audit
        )
    elif args.command == "plan":
        result = create_plan(
            source_plan_path=args.source_plan,
            storage_audit_path=args.storage_audit,
            scheduler_cutover_receipt_path=(
                args.scheduler_cutover_receipt
            ),
            output=args.output,
        )
    else:
        result = submit(
            plan_path=args.plan,
            fresh_storage_audit_path=args.fresh_storage_audit,
            priority=args.priority,
            output=args.output,
        )
    print(json.dumps({"status": "ok", "result": result}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
