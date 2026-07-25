"""Exact-once recovery for MFT Standard tasks killed by a failed dependency.

This is deliberately an MFT-project tool.  It does not import, modify, or
write the separate Scheduler repository.  The one allowed Scheduler mutation
is the guarded task POST performed by ``submit-dependency-failure-retry``.
"""

from __future__ import annotations

import argparse
import copy
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
    attest_fixed_identity,
    canonical_sha256,
)
from regression_260707.verify import scheduler_client
from tools import mft_campaign_atomic_claim as atomic_claim
from tools import mft_goal_diagnostic_standard_probe as probe
from tools import mft_goal_fea_handoff as production


PLAN_SCHEMA = "mft-goal-diagnostic-standard-dependency-failure-plan-v1"
SUBMISSION_SCHEMA = (
    "mft-goal-diagnostic-standard-dependency-failure-submission-v1"
)
RETRY_EVIDENCE_SCHEMA = (
    "mft-goal-diagnostic-standard-dependency-failure-evidence-v1"
)
FAILURE_EVIDENCE_SCHEMA = (
    "mft-goal-diagnostic-standard-dependency-task-failure-v1"
)
ANCHOR_EVIDENCE_SCHEMA = (
    "mft-goal-diagnostic-standard-dependency-anchor-failure-v1"
)
SIBLING_GUARD_SCHEMA = (
    "mft-goal-diagnostic-standard-dependency-sibling-guard-v1"
)
CLAIM_RECEIPT_SCHEMA = (
    "mft-goal-dependency-failure-atomic-claim-receipt-v1"
)
PROFILE_SCHEMA = (
    "mft-goal-diagnostic-standard-dependency-failure-retry-profile-v1"
)
RETRY_GENERATION = "dependency-failure-r1"
FAILURE_CLASS = "scheduler_same_node_dependency_failed"
DEPENDENCY_FAILURE_TEMPLATE = "same_node_as task {anchor_task_id} is failed"
ANCHOR_FAILURE_CLASS = "native_thermal_nonconvergence"
ANCHOR_FAILURE_MESSAGE = (
    "RESULT_JSON: thermal_extraction_failure_reason="
    "solve_not_converged:native_terminal_error"
)
RESOURCES = {"cpus": 8, "timeout_seconds": 8 * 3600}
PROFILE_PATH = (
    probe.REPOSITORY_ROOT
    / "regression_260707"
    / "verify"
    / "profiles"
    / "goal_diagnostic_standard_dependency_failure_retry.json"
)
CLAIM_ROOT = Path(
    "C:/Users/peets/slurm_scheduler_runtime/mft_goal_20260726/"
    "dependency_failure_claims"
)
CLAIM_AUTHORITY_SHA256 = canonical_sha256(
    {
        "campaign_id": "mft-goal-20260726",
        "goal_contract_schema": GOAL_CONTRACT_SCHEMA,
        "hard_spec_sha256": GOAL_STAGE_SPEC_SHA256,
        "temperature_contract_sha256": (
            GOAL_TEMPERATURE_CONTRACT_SHA256
        ),
        "retry_generation": RETRY_GENERATION,
    }
)
MAX_POST_RECONCILIATION_READS = 8
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


def _task_id(snapshot: Mapping[str, Any]) -> Any:
    return snapshot.get("task_id", snapshot.get("id"))


def _profile_content() -> tuple[dict[str, Any], dict[str, Any]]:
    path = PROFILE_PATH.resolve(strict=True)
    profile = production._read_json(path)
    _validate_profile(profile)
    return profile, production._file_record(path)


def _validate_profile(profile: Mapping[str, Any]) -> None:
    timeout_profile = production._read_json(
        probe.TIMEOUT_RETRY_PROFILE_PATH.resolve(strict=True)
    )
    expected = copy.deepcopy(timeout_profile)
    expected["schema_version"] = PROFILE_SCHEMA
    expected["comment"] = (
        "Diagnostic-only eighth-symmetry Standard FEA dependency-failure "
        "retry with retained AEDT project and AEDT results"
    )
    if profile != expected:
        raise HandoffContractError(
            "dependency-failure retry profile contract drifted"
        )
    effective_identity = dict(profile["param_overrides"])
    effective_identity["thermal_pad_conductivity_W_mK"] = 0.2
    attest_fixed_identity(effective_identity)


def initialize_claim_root(root: Path | None = None) -> dict[str, Any]:
    target = CLAIM_ROOT if root is None else Path(root)
    try:
        return atomic_claim.initialize_claim_root(
            target,
            campaign_id="mft-goal-20260726",
            campaign_authority_sha256=CLAIM_AUTHORITY_SHA256,
        )
    except atomic_claim.ClaimContractError as exc:
        raise HandoffContractError(
            "dependency-failure atomic claim root is unavailable"
        ) from exc


def _claim_authority() -> dict[str, Any]:
    try:
        authority = atomic_claim.load_claim_root(CLAIM_ROOT)
    except atomic_claim.ClaimContractError as exc:
        raise HandoffContractError(
            "dependency-failure atomic claim root is unavailable"
        ) from exc
    if (
        authority.get("campaign_id") != "mft-goal-20260726"
        or authority.get("campaign_authority_sha256")
        != CLAIM_AUTHORITY_SHA256
    ):
        raise HandoffContractError(
            "dependency-failure atomic claim authority drifted"
        )
    return authority


def _claim_reference(
    *,
    candidate_physics_sha256: str,
    logical_authority_task_id: int,
) -> dict[str, Any]:
    try:
        return atomic_claim.build_claim_reference(
            _claim_authority(),
            candidate_physics_sha256=production._require_sha(
                candidate_physics_sha256,
                "dependency-failure candidate physics SHA",
            ),
            logical_authority_task_id=logical_authority_task_id,
            retry_generation=RETRY_GENERATION,
        )
    except atomic_claim.ClaimContractError as exc:
        raise HandoffContractError(
            "dependency-failure atomic claim reference is invalid"
        ) from exc


def _validate_claim_reference(plan: Mapping[str, Any]) -> dict[str, Any]:
    reference = plan.get("dependency_failure_atomic_claim_reference")
    record = plan.get("retry_of_dependency_failure")
    if not isinstance(reference, Mapping) or not isinstance(record, Mapping):
        raise HandoffContractError(
            "dependency-failure atomic claim reference is absent"
        )
    try:
        normalized = atomic_claim.validate_claim_reference(
            reference, _claim_authority()
        )
    except atomic_claim.ClaimContractError as exc:
        raise HandoffContractError(
            "dependency-failure atomic claim reference drifted"
        ) from exc
    if (
        normalized.get("candidate_physics_sha256")
        != plan.get("candidate_physics_sha256")
        or normalized.get("logical_authority_task_id")
        != record.get("logical_authority_task_id")
        or normalized.get("retry_generation") != RETRY_GENERATION
    ):
        raise HandoffContractError(
            "dependency-failure atomic claim plan binding drifted"
        )
    return normalized


def _task_identity(
    *,
    logical_authority_task_id: int,
    candidate_physics_sha256: str,
) -> tuple[str, str]:
    logical_id = _positive_int(
        logical_authority_task_id, "logical authority task ID"
    )
    stem = production._require_sha(
        candidate_physics_sha256,
        "dependency-failure candidate physics SHA",
    )[:12]
    return (
        f"mft-goal-diag-standard-dependency-r1-l{logical_id}-{stem}",
        f"mft_goal_diag_standard_dependency_r1_l{logical_id}_{stem}",
    )


def _sibling_name_prefix(plan: Mapping[str, Any]) -> str:
    record = plan["retry_of_dependency_failure"]
    logical_id = _positive_int(
        record.get("logical_authority_task_id"),
        "dependency-failure logical authority task ID",
    )
    return f"mft-goal-diag-standard-dependency-r1-l{logical_id}-"


def _normalized_task_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "name",
        "status",
        "state",
        "exit_code",
        "failure_message",
        "created_at",
        "attached_at",
        "launch_started_at",
        "started_at",
        "finished_at",
        "assigned_allocation",
        "allocation_id",
        "account_name",
        "requested_account_name",
        "slurm_job_id",
        "remote_cwd",
        "remote_dir",
        "required_capability",
        "env_profile",
        "project",
        "scheduling_profile",
        "aedt_backend",
        "priority",
        "timeout_seconds",
        "dedupe_key",
        "same_node_as_task_id",
        "requested_allocation_id",
        "cpus",
        "memory_mb",
        "node_name",
        "requested_node_name",
        "node_name_policy",
        "requested_node_name_policy",
        "strict_node_placement",
        "placement_contract_satisfied",
        "allocation_node_name",
        "actual_node_name",
    )
    return {
        "task_id": _task_id(snapshot),
        **{field: copy.deepcopy(snapshot.get(field)) for field in fields},
    }


def _placement_anchor_identity(
    immediate_submission: Mapping[str, Any],
) -> dict[str, Any]:
    placement = immediate_submission.get("scheduler_placement_contract")
    if not isinstance(placement, Mapping):
        raise HandoffContractError(
            "dependency-failed task lacks same-allocation ancestry"
        )
    anchor_before = placement.get("anchor_before_submission")
    anchor_after = placement.get("anchor_after_submission")
    if (
        not isinstance(anchor_before, Mapping)
        or not isinstance(anchor_after, Mapping)
        or anchor_before != anchor_after
    ):
        raise HandoffContractError(
            "dependency anchor submission identity drifted"
        )
    anchor_task_id = _positive_int(
        placement.get("same_node_as_task_id"),
        "dependency anchor task ID",
    )
    expected = {
        "task_id": anchor_task_id,
        "name": anchor_before.get("name"),
        "dedupe_key": anchor_before.get("dedupe_key"),
        "project": scheduler_client.MFT_PROJECT,
        "scheduling_profile": "fea_bursty",
        "aedt_backend": "standalone",
        "cpus": 8,
        "memory_mb": 32768,
        "timeout_seconds": immediate_submission["resources"][
            "timeout_seconds"
        ],
        "allocation_id": placement.get("expected_allocation_id"),
        "account_name": placement.get("expected_account_name"),
        "slurm_job_id": placement.get("expected_slurm_job_id"),
        "actual_node_name": placement.get("expected_node_name"),
    }
    for key, value in expected.items():
        if anchor_before.get(key) != value:
            raise HandoffContractError(
                "dependency anchor placement ancestry drifted"
            )
    return expected


def _dependency_failure_evidence(
    snapshot: Mapping[str, Any],
    *,
    immediate_submission: Mapping[str, Any],
    anchor_task_id: int,
) -> dict[str, Any]:
    evidence = _normalized_task_snapshot(snapshot)
    expected_message = DEPENDENCY_FAILURE_TEMPLATE.format(
        anchor_task_id=anchor_task_id
    )
    if (
        evidence["task_id"] != immediate_submission["task_id"]
        or evidence["name"] != immediate_submission["task_name"]
        or evidence["dedupe_key"] != immediate_submission["dedupe_key"]
        or evidence["project"] != scheduler_client.MFT_PROJECT
        or evidence["scheduling_profile"] != "fea_bursty"
        or evidence["aedt_backend"] != "standalone"
        or evidence["cpus"] != RESOURCES["cpus"]
        or evidence["memory_mb"] != 32768
        or evidence["timeout_seconds"]
        != immediate_submission["resources"]["timeout_seconds"]
        or evidence["same_node_as_task_id"] != anchor_task_id
        or evidence["requested_node_name"]
        != immediate_submission["scheduler_strict_node_contract"][
            "plan_contract"
        ]["requested_node_name"]
        or evidence["requested_node_name_policy"] != "strict"
        or evidence["strict_node_placement"] is not True
        or evidence["status"] != "failed"
        or evidence["state"] != "failed"
        or evidence["exit_code"] is not None
        or evidence["failure_message"] != expected_message
        or not str(evidence["finished_at"] or "").strip()
    ):
        raise HandoffContractError(
            "dependency-failed task terminal evidence drifted"
        )
    return {
        "schema_version": FAILURE_EVIDENCE_SCHEMA,
        "failure_class": FAILURE_CLASS,
        "failure_message": expected_message,
        "task": evidence,
    }


def _anchor_failure_evidence(
    snapshot: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = _normalized_task_snapshot(snapshot)
    static = {
        key: evidence.get(key)
        for key in expected_identity
    }
    if (
        static != dict(expected_identity)
        or evidence["status"] != "failed"
        or evidence["state"] != "failed"
        or evidence["exit_code"] != 1
        or evidence["failure_message"] != ANCHOR_FAILURE_MESSAGE
        or not str(evidence["started_at"] or "").strip()
        or not str(evidence["finished_at"] or "").strip()
    ):
        raise HandoffContractError(
            "dependency anchor native failure evidence drifted"
        )
    return {
        "schema_version": ANCHOR_EVIDENCE_SCHEMA,
        "failure_class": ANCHOR_FAILURE_CLASS,
        "failure_message": ANCHOR_FAILURE_MESSAGE,
        "task": evidence,
    }


def _failure_bundle(
    *,
    immediate_snapshot: Mapping[str, Any],
    anchor_snapshot: Mapping[str, Any],
    immediate_submission: Mapping[str, Any],
    expected_anchor: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    anchor_id = int(expected_anchor["task_id"])
    dependency = _dependency_failure_evidence(
        immediate_snapshot,
        immediate_submission=immediate_submission,
        anchor_task_id=anchor_id,
    )
    anchor = _anchor_failure_evidence(
        anchor_snapshot, expected_identity=expected_anchor
    )
    dependency_finished = probe._scheduler_timestamp(
        dependency["task"]["finished_at"],
        "dependency-failed task finished_at",
    )
    anchor_finished = probe._scheduler_timestamp(
        anchor["task"]["finished_at"],
        "dependency anchor finished_at",
    )
    if dependency_finished < anchor_finished:
        raise HandoffContractError(
            "dependency task failed before its anchor"
        )
    return dependency, anchor


def _read_failure_bundle(
    *,
    scheduler_url: str,
    immediate_submission: Mapping[str, Any],
    expected_anchor: Mapping[str, Any],
    task_reader: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return _failure_bundle(
        immediate_snapshot=task_reader(
            scheduler_url=scheduler_url,
            task_id=int(immediate_submission["task_id"]),
        ),
        anchor_snapshot=task_reader(
            scheduler_url=scheduler_url,
            task_id=int(expected_anchor["task_id"]),
        ),
        immediate_submission=immediate_submission,
        expected_anchor=expected_anchor,
    )


def _retry_record(
    *,
    immediate_plan_path: Path,
    immediate_submission_path: Path,
    immediate_plan: Mapping[str, Any],
    immediate_submission: Mapping[str, Any],
    logical_submission: Mapping[str, Any],
    dependency_evidence: Mapping[str, Any],
    anchor_evidence: Mapping[str, Any],
    anchor_task_id: int,
    scheduler_url: str,
) -> dict[str, Any]:
    return {
        "schema_version": RETRY_EVIDENCE_SCHEMA,
        "retry_generation": RETRY_GENERATION,
        "retry_of_task_id": immediate_submission["task_id"],
        "logical_authority_task_id": logical_submission["task_id"],
        "immediate_retry_kind": "timeout",
        "immediate_parent_ancestry_sha256": canonical_sha256(
            immediate_plan["retry_of_timeout"]
        ),
        "dependency_anchor_task_id": anchor_task_id,
        "original_plan": production._file_record(
            immediate_plan_path.resolve(strict=True)
        ),
        "original_plan_payload_sha256": immediate_plan["payload_sha256"],
        "original_submission": production._file_record(
            immediate_submission_path.resolve(strict=True)
        ),
        "original_submission_payload_sha256": immediate_submission[
            "payload_sha256"
        ],
        "dependency_failure_evidence": copy.deepcopy(dependency_evidence),
        "dependency_failure_evidence_sha256": canonical_sha256(
            dependency_evidence
        ),
        "anchor_failure_evidence": copy.deepcopy(anchor_evidence),
        "anchor_failure_evidence_sha256": canonical_sha256(
            anchor_evidence
        ),
        "scheduler_url": scheduler_url,
        "failure_class": FAILURE_CLASS,
        "failure_message": DEPENDENCY_FAILURE_TEMPLATE.format(
            anchor_task_id=anchor_task_id
        ),
    }


def create_plan(
    *,
    original_plan_path: Path,
    original_submission_path: Path,
    dependency_anchor_task_id: int,
    strict_node_name: str,
    output: Path,
    scheduler_url: str = probe.DIAGNOSTIC_SCHEDULER_URL,
    task_reader: Any = None,
) -> Path:
    immediate_plan, params, selected = probe._load_plan(original_plan_path)
    if probe._plan_retry_kind(immediate_plan) != "timeout":
        raise HandoffContractError(
            "dependency-failure retry requires exactly one timeout parent"
        )
    immediate_submission = probe._load_submission(
        original_submission_path, plan=immediate_plan
    )
    (
        logical_plan,
        logical_submission,
        _timeout_evidence,
    ) = probe._validate_timeout_retry_record(immediate_plan)
    if probe._plan_retry_kind(logical_plan) is not None:
        raise HandoffContractError(
            "dependency-failure retry ancestry exceeds one timeout parent"
        )
    normalized_url = scheduler_url.rstrip("/")
    if (
        normalized_url != probe.DIAGNOSTIC_SCHEDULER_URL
        or normalized_url != immediate_submission["scheduler_url"]
    ):
        raise HandoffContractError(
            "dependency-failure Scheduler origin drifted"
        )
    expected_anchor = _placement_anchor_identity(immediate_submission)
    anchor_id = _positive_int(
        dependency_anchor_task_id, "dependency anchor task ID"
    )
    if expected_anchor["task_id"] != anchor_id:
        raise HandoffContractError(
            "requested dependency anchor differs from sealed ancestry"
        )
    reader = task_reader or probe._scheduler_task_snapshot
    dependency_evidence, anchor_evidence = _read_failure_bundle(
        scheduler_url=normalized_url,
        immediate_submission=immediate_submission,
        expected_anchor=expected_anchor,
        task_reader=reader,
    )
    original_profile = production._read_json(
        original_plan_path.resolve(strict=True).parent
        / immediate_plan["profile"]["path"]
    )
    profile, profile_source = _profile_content()
    if (
        profile["param_overrides"]
        != original_profile["param_overrides"]
        or profile["fixed_boundary_contract"]
        != original_profile["fixed_boundary_contract"]
        or production._effective_params(params, profile)
        != production._effective_params(params, original_profile)
    ):
        raise HandoffContractError(
            "dependency-failure retry profile changes fixed physics"
        )
    task_name, workdir = _task_identity(
        logical_authority_task_id=int(logical_submission["task_id"]),
        candidate_physics_sha256=str(
            immediate_plan["candidate_physics_sha256"]
        ),
    )
    strict_contract = probe._strict_node_plan_contract(
        strict_node_name,
        task_identity_generation=(
            probe.DEPENDENCY_FAILURE_STRICT_TASK_IDENTITY_GENERATION
        ),
    )
    retained = scheduler_client.retained_aedt_identity(
        task_name,
        params,
        profile,
        immediate_plan["solver_revision"],
        immediate_plan["library_revision"],
    )
    if (
        retained is None
        or retained["dedupe_key"]
        == immediate_plan["stage"]["retained_aedt_bundle"]["dedupe_key"]
        or retained["relative_directory"]
        == immediate_plan["stage"]["retained_aedt_bundle"][
            "relative_directory"
        ]
        or retained["profile_sha256"]
        == immediate_plan["stage"]["retained_aedt_bundle"][
            "profile_sha256"
        ]
    ):
        raise HandoffContractError(
            "dependency-failure retry identity is not distinct"
        )
    record = _retry_record(
        immediate_plan_path=original_plan_path,
        immediate_submission_path=original_submission_path,
        immediate_plan=immediate_plan,
        immediate_submission=immediate_submission,
        logical_submission=logical_submission,
        dependency_evidence=dependency_evidence,
        anchor_evidence=anchor_evidence,
        anchor_task_id=anchor_id,
        scheduler_url=normalized_url,
    )
    reference = _claim_reference(
        candidate_physics_sha256=str(
            immediate_plan["candidate_physics_sha256"]
        ),
        logical_authority_task_id=int(logical_submission["task_id"]),
    )
    destination = output.resolve()
    if destination.exists():
        raise HandoffContractError(
            f"dependency-failure retry plan output exists: {destination}"
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
            staging / "diagnostic_standard_dependency_failure_profile.json",
            profile,
        )
        unsigned = copy.deepcopy(immediate_plan)
        unsigned.pop("payload_sha256", None)
        unsigned.pop("retry_of_timeout", None)
        unsigned.pop("scheduler_strict_node_contract", None)
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
                    "canonical_sha256": canonical_sha256(profile),
                    "source": profile_source,
                },
                "stage": {
                    **copy.deepcopy(immediate_plan["stage"]),
                    "task_name": task_name,
                    "workdir": workdir,
                    "profile_sha256": canonical_sha256(profile),
                    "resources": copy.deepcopy(RESOURCES),
                    "retained_aedt_bundle": retained,
                    "retention_run_root": (
                        probe._retention_run_root_evidence(retained)
                    ),
                },
                "available_submission_commands": [
                    "submit-dependency-failure-retry"
                ],
                "retry_of_dependency_failure": record,
                "dependency_failure_atomic_claim_reference": reference,
                "scheduler_strict_node_contract": strict_contract,
            }
        )
        plan_path = production._write_immutable_json(
            staging / "diagnostic_dependency_failure_retry_plan.json",
            production._seal(unsigned),
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / plan_path.name


def _plan_artifact(root: Path, record: Any, label: str) -> Path:
    if (
        not isinstance(record, Mapping)
        or not {"path", "sha256"}.issubset(record)
    ):
        raise HandoffContractError(f"{label} plan record is malformed")
    target = production._contained_file(root, record["path"], label)
    if production._sha256_file(target) != production._require_sha(
        record["sha256"], f"{label} SHA"
    ):
        raise HandoffContractError(f"{label} bytes drifted")
    return target


def _load_plan(
    path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    resolved = path.resolve(strict=True)
    plan = production._validate_seal(
        production._read_json(resolved), PLAN_SCHEMA
    )
    flags = _flags()
    if (
        plan.get("campaign_id") != "mft-goal-20260726"
        or plan.get("goal_contract_schema") != GOAL_CONTRACT_SCHEMA
        or plan.get("hard_spec") != GOAL_STAGE_SPEC
        or plan.get("hard_spec_sha256") != GOAL_STAGE_SPEC_SHA256
        or plan.get("temperature_contract_sha256")
        != GOAL_TEMPERATURE_CONTRACT_SHA256
        or any(plan.get(name) is not value for name, value in flags.items())
        or plan.get("available_submission_commands")
        != ["submit-dependency-failure-retry"]
        or plan.get("physics_override_allowed") is not False
        or plan.get("scheduler_repository_modified") is not False
        or plan.get("scheduler_project_mutation_performed") is not False
        or plan.get("scheduler_submission_performed") is not False
        or plan.get("retention_required") is not True
        or plan.get("prune_protection_required") is not True
    ):
        raise HandoffContractError(
            "dependency-failure retry plan contract drifted"
        )
    record = plan.get("retry_of_dependency_failure")
    expected_record_fields = {
        "schema_version",
        "retry_generation",
        "retry_of_task_id",
        "logical_authority_task_id",
        "immediate_retry_kind",
        "immediate_parent_ancestry_sha256",
        "dependency_anchor_task_id",
        "original_plan",
        "original_plan_payload_sha256",
        "original_submission",
        "original_submission_payload_sha256",
        "dependency_failure_evidence",
        "dependency_failure_evidence_sha256",
        "anchor_failure_evidence",
        "anchor_failure_evidence_sha256",
        "scheduler_url",
        "failure_class",
        "failure_message",
    }
    if (
        not isinstance(record, Mapping)
        or set(record) != expected_record_fields
        or record.get("schema_version") != RETRY_EVIDENCE_SCHEMA
        or record.get("retry_generation") != RETRY_GENERATION
        or record.get("immediate_retry_kind") != "timeout"
        or record.get("scheduler_url")
        != probe.DIAGNOSTIC_SCHEDULER_URL
        or record.get("failure_class") != FAILURE_CLASS
    ):
        raise HandoffContractError(
            "dependency-failure retry ancestry record drifted"
        )
    immediate_plan_path = probe._recorded_external_file(
        record["original_plan"], "dependency parent plan"
    )
    immediate_plan, parent_params, parent_selected = probe._load_plan(
        immediate_plan_path
    )
    if probe._plan_retry_kind(immediate_plan) != "timeout":
        raise HandoffContractError(
            "dependency-failure parent is not exactly one timeout retry"
        )
    immediate_submission_path = probe._recorded_external_file(
        record["original_submission"], "dependency parent submission"
    )
    immediate_submission = probe._load_submission(
        immediate_submission_path, plan=immediate_plan
    )
    (
        logical_plan,
        logical_submission,
        _timeout_evidence,
    ) = probe._validate_timeout_retry_record(immediate_plan)
    expected_anchor = _placement_anchor_identity(immediate_submission)
    dependency_evidence = record.get("dependency_failure_evidence")
    anchor_evidence = record.get("anchor_failure_evidence")
    if (
        record.get("retry_of_task_id")
        != immediate_submission["task_id"]
        or record.get("logical_authority_task_id")
        != logical_submission["task_id"]
        or record.get("dependency_anchor_task_id")
        != expected_anchor["task_id"]
        or record.get("original_plan_payload_sha256")
        != immediate_plan["payload_sha256"]
        or record.get("original_submission_payload_sha256")
        != immediate_submission["payload_sha256"]
        or record.get("immediate_parent_ancestry_sha256")
        != canonical_sha256(immediate_plan["retry_of_timeout"])
        or not isinstance(dependency_evidence, Mapping)
        or not isinstance(anchor_evidence, Mapping)
        or record.get("dependency_failure_evidence_sha256")
        != canonical_sha256(dependency_evidence)
        or record.get("anchor_failure_evidence_sha256")
        != canonical_sha256(anchor_evidence)
        or record.get("failure_message")
        != DEPENDENCY_FAILURE_TEMPLATE.format(
            anchor_task_id=expected_anchor["task_id"]
        )
        or any(
            plan.get(field) != authority.get(field)
            for authority in (immediate_plan, logical_plan)
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
            "dependency-failure retry ancestry drifted"
        )
    normalized_dependency, normalized_anchor = _failure_bundle(
        immediate_snapshot=dependency_evidence.get("task") or {},
        anchor_snapshot=anchor_evidence.get("task") or {},
        immediate_submission=immediate_submission,
        expected_anchor=expected_anchor,
    )
    if (
        normalized_dependency != dependency_evidence
        or normalized_anchor != anchor_evidence
    ):
        raise HandoffContractError(
            "dependency-failure sealed execution evidence drifted"
        )
    root = resolved.parent
    params = production._read_json(
        _plan_artifact(root, plan.get("fea_params"), "dependency FEA params")
    )
    selected = production._read_json(
        _plan_artifact(
            root,
            plan.get("selected_candidate"),
            "dependency selected candidate",
        )
    )
    if params != parent_params or selected != parent_selected:
        raise HandoffContractError(
            "dependency-failure candidate identity drifted"
        )
    profile_record = plan.get("profile")
    if not isinstance(profile_record, Mapping):
        raise HandoffContractError(
            "dependency-failure profile record is absent"
        )
    profile = production._read_json(
        _plan_artifact(root, profile_record, "dependency retry profile")
    )
    _validate_profile(profile)
    effective = production._effective_params(params, profile)
    stage = plan.get("stage")
    if not isinstance(stage, Mapping):
        raise HandoffContractError(
            "dependency-failure stage is absent"
        )
    expected_task_name, expected_workdir = _task_identity(
        logical_authority_task_id=int(logical_submission["task_id"]),
        candidate_physics_sha256=str(plan["candidate_physics_sha256"]),
    )
    retained = scheduler_client.retained_aedt_identity(
        expected_task_name,
        params,
        profile,
        plan["solver_revision"],
        plan["library_revision"],
    )
    strict_contract = plan.get("scheduler_strict_node_contract")
    probe._strict_node_scheduler_pin(strict_contract)
    if (
        strict_contract.get("task_identity_generation")
        != probe.DEPENDENCY_FAILURE_STRICT_TASK_IDENTITY_GENERATION
        or profile_record.get("canonical_sha256")
        != canonical_sha256(profile)
        or stage.get("name") != "standard"
        or stage.get("task_name") != expected_task_name
        or stage.get("workdir") != expected_workdir
        or stage.get("profile_sha256") != canonical_sha256(profile)
        or stage.get("effective_params_sha256")
        != canonical_sha256(effective)
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
            "dependency-failure execution identity drifted"
        )
    _validate_claim_reference(plan)
    return (
        plan,
        params,
        selected,
        immediate_submission,
        expected_anchor,
    )


def _sibling_snapshot(
    rows: Any, *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes, bytearray))
    ):
        raise HandoffContractError(
            "dependency-failure sibling inventory is absent"
        )
    stage = plan["stage"]
    expected_name = stage["task_name"]
    expected_dedupe = stage["retained_aedt_bundle"]["dedupe_key"]
    prefix = _sibling_name_prefix(plan)
    node_name = plan["scheduler_strict_node_contract"][
        "requested_node_name"
    ]
    matching: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        name = str(raw.get("name") or "")
        dedupe = str(raw.get("dedupe_key") or "")
        if not (
            name == expected_name
            or dedupe == expected_dedupe
            or name.startswith(prefix)
        ):
            continue
        normalized = copy.deepcopy(dict(raw))
        normalized.update(
            {
                "task_id": _task_id(raw),
                "name": name,
                "dedupe_key": dedupe,
                "requested_node_name": raw.get(
                    "requested_node_name", raw.get("node_name")
                ),
                "requested_node_name_policy": raw.get(
                    "requested_node_name_policy",
                    raw.get("node_name_policy"),
                ),
                "same_node_as_task_id": raw.get(
                    "same_node_as_task_id", 0
                ),
            }
        )
        if (
            isinstance(normalized["task_id"], bool)
            or not isinstance(normalized["task_id"], int)
            or normalized["task_id"] <= 0
            or normalized["name"] != expected_name
            or normalized["dedupe_key"] != expected_dedupe
            or normalized.get("project") != scheduler_client.MFT_PROJECT
            or normalized.get("cpus") != RESOURCES["cpus"]
            or normalized.get("memory_mb") != 32768
            or normalized.get("timeout_seconds")
            != RESOURCES["timeout_seconds"]
            or normalized.get("aedt_backend") != "standalone"
            or normalized["requested_node_name"] != node_name
            or normalized["requested_node_name_policy"] != "strict"
            or normalized["same_node_as_task_id"] != 0
        ):
            raise HandoffContractError(
                "dependency-failure sibling identity collision detected"
            )
        matching.append(normalized)
    matching.sort(key=lambda row: row["task_id"])
    if len(matching) > 1:
        raise HandoffContractError(
            "more than one dependency-failure retry sibling exists"
        )
    snapshot = {
        "schema_version": SIBLING_GUARD_SCHEMA,
        "identity": {
            "logical_authority_task_id": plan[
                "retry_of_dependency_failure"
            ]["logical_authority_task_id"],
            "immediate_task_id": plan[
                "retry_of_dependency_failure"
            ]["retry_of_task_id"],
            "dependency_anchor_task_id": plan[
                "retry_of_dependency_failure"
            ]["dependency_anchor_task_id"],
            "retry_generation": RETRY_GENERATION,
            "name_prefix": prefix,
            "candidate_physics_sha256": plan[
                "candidate_physics_sha256"
            ],
            "task_name": expected_name,
            "workdir": stage["workdir"],
            "dedupe_key": expected_dedupe,
            "requested_node_name": node_name,
            "same_node_as_task_id": 0,
            "resources": copy.deepcopy(RESOURCES),
        },
        "matching_task_count": len(matching),
        "matching_tasks": matching,
    }
    snapshot["snapshot_sha256"] = canonical_sha256(snapshot)
    return snapshot


def _sibling_contract(
    *,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    task_id: int,
) -> dict[str, Any]:
    if (
        before.get("identity") != after.get("identity")
        or before.get("matching_task_count") not in {0, 1}
        or after.get("matching_task_count") != 1
        or after.get("matching_tasks", [{}])[0].get("task_id")
        != task_id
        or (
            before.get("matching_task_count") == 1
            and before.get("matching_tasks", [{}])[0].get("task_id")
            != task_id
        )
    ):
        raise HandoffContractError(
            "dependency-failure sibling identity changed during submission"
        )
    return {
        "schema_version": SIBLING_GUARD_SCHEMA,
        "before_submission": copy.deepcopy(before),
        "after_submission": copy.deepcopy(after),
        "exactly_one_sibling_after_submission": True,
    }


def _claim_winner(
    plan_path: Path, plan: Mapping[str, Any]
) -> dict[str, Any]:
    stage = plan["stage"]
    record = plan["retry_of_dependency_failure"]
    return {
        "immediate_task_id": record["retry_of_task_id"],
        "immediate_retry_kind": "timeout",
        "plan_payload_sha256": plan["payload_sha256"],
        "plan_file_sha256": production._sha256_file(
            plan_path.resolve(strict=True)
        ),
        "profile_sha256": stage["profile_sha256"],
        "resources": {
            **copy.deepcopy(RESOURCES),
            "memory_mb": 32768,
        },
        "task_name": stage["task_name"],
        "dedupe_key": stage["retained_aedt_bundle"]["dedupe_key"],
    }


def _claim_task_evidence(
    task: Mapping[str, Any], pending: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(task, Mapping) or not isinstance(pending, Mapping):
        raise atomic_claim.ClaimContractError(
            "dependency-failure claim task evidence is absent"
        )
    row = copy.deepcopy(dict(task))
    winner = pending.get("winner")
    row.update(
        {
            "task_id": _task_id(task),
            "requested_node_name": task.get(
                "requested_node_name", task.get("node_name")
            ),
            "requested_node_name_policy": task.get(
                "requested_node_name_policy",
                task.get("node_name_policy"),
            ),
            "same_node_as_task_id": task.get(
                "same_node_as_task_id", 0
            ),
        }
    )
    if (
        not isinstance(winner, Mapping)
        or isinstance(row["task_id"], bool)
        or not isinstance(row["task_id"], int)
        or row["task_id"] <= 0
        or row.get("name") != winner.get("task_name")
        or row.get("dedupe_key") != winner.get("dedupe_key")
        or row.get("project") != scheduler_client.MFT_PROJECT
        or row.get("cpus") != winner["resources"]["cpus"]
        or row.get("memory_mb") != winner["resources"]["memory_mb"]
        or row.get("timeout_seconds")
        != winner["resources"]["timeout_seconds"]
        or row.get("aedt_backend") != "standalone"
        or not str(row["requested_node_name"] or "").strip()
        or row["requested_node_name_policy"] != "strict"
        or row["same_node_as_task_id"] != 0
        or not str(row.get("status") or "").strip()
        or not str(row.get("state") or "").strip()
    ):
        raise atomic_claim.ClaimContractError(
            "dependency-failure claim task identity drifted"
        )
    return row


def _claim_receipt(
    *,
    acquisition_status: str,
    finalized_claim: Mapping[str, Any],
) -> dict[str, Any]:
    if acquisition_status not in {
        "fresh_pending",
        "existing_pending",
        "existing_finalized",
    }:
        raise HandoffContractError(
            "dependency-failure claim acquisition status drifted"
        )
    return {
        "schema_version": CLAIM_RECEIPT_SCHEMA,
        "acquisition_status": acquisition_status,
        "fresh_claim_authorized_scheduler_submit_call": (
            acquisition_status == "fresh_pending"
        ),
        "recovered_without_scheduler_submit_call": (
            acquisition_status != "fresh_pending"
        ),
        "finalized_claim": copy.deepcopy(dict(finalized_claim)),
    }


def _post_reconcile_sibling(
    *,
    plan: Mapping[str, Any],
    scheduler_url: str,
    task_list_reader: Any,
    wait: Any,
) -> dict[str, Any]:
    latest = None
    for attempt in range(MAX_POST_RECONCILIATION_READS):
        latest = _sibling_snapshot(
            task_list_reader(
                scheduler_url=scheduler_url,
                project=scheduler_client.MFT_PROJECT,
                task_name=_sibling_name_prefix(plan),
            ),
            plan=plan,
        )
        if latest["matching_task_count"] == 1:
            return latest
        if attempt + 1 < MAX_POST_RECONCILIATION_READS:
            wait()
    raise HandoffContractError(
        "dependency-failure post-POST sibling did not become visible"
    )


def submit(
    *,
    plan_path: Path,
    scheduler_cutover_receipt_path: Path,
    output: Path,
    priority: int = 100,
    scheduler: Any = scheduler_client,
    predictor: Any | None = None,
    live_reader: Any = probe._default_scheduler_live_reader,
    task_reader: Any = None,
    task_list_reader: Any = None,
    reconciliation_waiter: Any = None,
) -> Path:
    target = output.resolve()
    if target.exists():
        raise HandoffContractError(
            f"dependency-failure submission receipt exists: {target}"
        )
    (
        plan,
        params,
        selected,
        immediate_submission,
        expected_anchor,
    ) = _load_plan(plan_path)
    strict_contract = plan["scheduler_strict_node_contract"]
    strict_pin = probe._strict_node_scheduler_pin(
        strict_contract, require_active=True
    )
    reauthentication = probe._fresh_selection_reauthentication(
        plan=plan, selected=selected, predictor=predictor
    )
    cutover, launcher_before = probe._validate_scheduler_cutover_receipt(
        scheduler_cutover_receipt_path,
        verify_live_launcher=True,
        require_strict_node=True,
        strict_node_contract=strict_contract,
        require_active_strict=True,
    )
    stage = plan["stage"]
    if cutover["scheduler_url"] != stage["scheduler_url"]:
        raise HandoffContractError(
            "Scheduler cutover endpoint differs from dependency plan"
        )
    admission = probe._live_scheduler_admission_snapshot(
        scheduler_url=stage["scheduler_url"], reader=live_reader
    )
    launcher_after = probe._live_launcher_identity(
        cutover, expected_sha256=strict_pin["launcher_sha256"]
    )
    if launcher_after != launcher_before:
        raise HandoffContractError(
            "Scheduler launcher changed during dependency admission"
        )
    reader = task_reader or probe._scheduler_task_snapshot
    read_tasks = task_list_reader or probe._scheduler_project_tasks
    stored_record = plan["retry_of_dependency_failure"]
    stored_dependency = stored_record["dependency_failure_evidence"]
    stored_anchor = stored_record["anchor_failure_evidence"]

    def revalidate_failures() -> None:
        dependency, anchor = _read_failure_bundle(
            scheduler_url=stage["scheduler_url"],
            immediate_submission=immediate_submission,
            expected_anchor=expected_anchor,
            task_reader=reader,
        )
        if dependency != stored_dependency or anchor != stored_anchor:
            raise HandoffContractError(
                "dependency or anchor evidence changed before submission"
            )

    revalidate_failures()
    sibling_before = _sibling_snapshot(
        read_tasks(
            scheduler_url=stage["scheduler_url"],
            project=scheduler_client.MFT_PROJECT,
            task_name=_sibling_name_prefix(plan),
        ),
        plan=plan,
    )
    reference = _validate_claim_reference(plan)
    winner = _claim_winner(plan_path, plan)
    try:
        acquisition = atomic_claim.acquire_claim(
            CLAIM_ROOT, reference, winner
        )
    except atomic_claim.ClaimContractError as exc:
        raise HandoffContractError(
            "dependency-failure atomic claim acquisition failed"
        ) from exc
    claim_status = acquisition["status"]
    locked_guard_count = 0

    def locked_pre_submit_guard() -> None:
        nonlocal sibling_before, locked_guard_count
        revalidate_failures()
        sibling_before = _sibling_snapshot(
            read_tasks(
                scheduler_url=stage["scheduler_url"],
                project=scheduler_client.MFT_PROJECT,
                task_name=_sibling_name_prefix(plan),
            ),
            plan=plan,
        )
        if sibling_before["matching_task_count"] != 0:
            raise HandoffContractError(
                "fresh dependency-failure claim requires an empty "
                "Scheduler sibling slot before POST"
            )
        locked_guard_count += 1

    finalized_claim = None
    if claim_status != "fresh_pending":
        if sibling_before["matching_task_count"] != 1:
            raise HandoffContractError(
                "dependency-failure claim recovery requires exactly one "
                "matching task and never re-POSTs"
            )
        recovered = sibling_before["matching_tasks"][0]
        try:
            if claim_status == "existing_pending":
                finalized_claim = atomic_claim.recover_pending_claim(
                    CLAIM_ROOT,
                    reference,
                    acquisition["claim"],
                    matching_tasks=[recovered],
                    sibling_snapshot=sibling_before,
                    evidence_validator=_claim_task_evidence,
                )
            elif claim_status == "existing_finalized":
                finalized_claim = atomic_claim.validate_finalized_claim(
                    CLAIM_ROOT,
                    reference,
                    claim=acquisition["claim"],
                    expected_winner=winner,
                )
                _claim_task_evidence(
                    recovered, finalized_claim["pending_claim"]
                )
                if recovered["task_id"] != finalized_claim["task_id"]:
                    raise atomic_claim.ClaimContractError(
                        "finalized dependency claim task drifted"
                    )
            else:
                raise atomic_claim.ClaimContractError(
                    "dependency claim acquisition status is unsupported"
                )
        except atomic_claim.ClaimContractError as exc:
            raise HandoffContractError(
                "dependency-failure atomic claim recovery failed closed"
            ) from exc
        submission_result = {
            "task_id": finalized_claim["task_id"],
            "submission_source": "pre_submission_reconciliation",
            "scheduler_mutation_performed": False,
            "api_pre_submission_readback": recovered,
            "api_post_submission_response": None,
        }
        sibling_after = copy.deepcopy(sibling_before)
    else:
        if sibling_before["matching_task_count"] != 0:
            raise HandoffContractError(
                "fresh dependency-failure claim found an existing sibling"
            )
        profile = production._read_json(
            plan_path.resolve(strict=True).parent / plan["profile"]["path"]
        )
        environment, core_evidence = production._submission_environment(
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
            node_name=strict_contract["requested_node_name"],
            node_name_policy="strict",
            return_submission_evidence=True,
        )
        if locked_guard_count != 1:
            raise HandoffContractError(
                "dependency-failure submission lacks exactly one locked "
                "pre-POST guard execution"
            )
        waiter = reconciliation_waiter or (
            lambda: time.sleep(POST_RECONCILIATION_INTERVAL_SECONDS)
        )
        sibling_after = _post_reconcile_sibling(
            plan=plan,
            scheduler_url=stage["scheduler_url"],
            task_list_reader=read_tasks,
            wait=waiter,
        )
    if (
        not isinstance(submission_result, Mapping)
        or set(submission_result)
        != {
            "task_id",
            "submission_source",
            "scheduler_mutation_performed",
            "api_pre_submission_readback",
            "api_post_submission_response",
        }
    ):
        raise HandoffContractError(
            "dependency-failure strict submission returned no API evidence"
        )
    new_task_id = _positive_int(
        submission_result.get("task_id"),
        "dependency-failure submitted task ID",
    )
    if new_task_id in {
        stored_record["retry_of_task_id"],
        stored_record["logical_authority_task_id"],
        stored_record["dependency_anchor_task_id"],
    }:
        raise HandoffContractError(
            "dependency-failure retry resolved to an ancestry task"
        )
    sibling_guard = _sibling_contract(
        before=sibling_before,
        after=sibling_after,
        task_id=new_task_id,
    )
    if claim_status == "fresh_pending":
        try:
            finalized_claim = atomic_claim.finalize_claim(
                CLAIM_ROOT,
                reference,
                acquisition["claim"],
                task_id=new_task_id,
                task_readback=sibling_after["matching_tasks"][0],
                sibling_snapshot=sibling_after,
                evidence_validator=_claim_task_evidence,
            )
        except atomic_claim.ClaimContractError as exc:
            raise HandoffContractError(
                "dependency-failure atomic claim finalization failed"
            ) from exc
    durable = reader(
        scheduler_url=stage["scheduler_url"], task_id=new_task_id
    )
    try:
        durable_evidence = _claim_task_evidence(
            durable, finalized_claim["pending_claim"]
        )
    except atomic_claim.ClaimContractError as exc:
        raise HandoffContractError(
            "dependency-failure durable direct-node readback drifted"
        ) from exc
    raw_pre = submission_result["api_pre_submission_readback"]
    raw_post = submission_result["api_post_submission_response"]
    trace = {
        "submission_source": submission_result["submission_source"],
        "scheduler_mutation_performed": submission_result[
            "scheduler_mutation_performed"
        ],
        "api_pre_submission_readback": (
            _claim_task_evidence(raw_pre, finalized_claim["pending_claim"])
            if isinstance(raw_pre, Mapping)
            else None
        ),
        "api_post_submission_response": (
            _claim_task_evidence(raw_post, finalized_claim["pending_claim"])
            if isinstance(raw_post, Mapping)
            else None
        ),
        "api_durable_get_readback": durable_evidence,
        "direct_strict_node_policy_authenticated": True,
        "same_node_as_task_id": 0,
    }
    if (
        trace["api_pre_submission_readback"] is None
        and trace["api_post_submission_response"] is None
    ):
        raise HandoffContractError(
            "dependency-failure submission has no API readback"
        )
    if claim_status == "fresh_pending":
        core_policy = core_evidence
    else:
        _environment, core_policy = production._submission_environment(
            stage="standard",
            solver_revision=plan["solver_revision"],
            license_snapshot_path=None,
        )
    receipt = production._seal(
        {
            "schema_version": SUBMISSION_SCHEMA,
            "stage": "standard",
            "plan": production._file_record(plan_path.resolve(strict=True)),
            "plan_payload_sha256": plan["payload_sha256"],
            "candidate_physics_sha256": plan[
                "candidate_physics_sha256"
            ],
            "search_authority_reauthentication": reauthentication,
            "task_id": new_task_id,
            "task_name": stage["task_name"],
            "workdir": stage["workdir"],
            "dedupe_key": stage["retained_aedt_bundle"]["dedupe_key"],
            "solver_revision": plan["solver_revision"],
            "library_revision": plan["library_revision"],
            "profile_sha256": stage["profile_sha256"],
            "effective_params_sha256": stage[
                "effective_params_sha256"
            ],
            "resources": copy.deepcopy(RESOURCES),
            "aedt_backend": "standalone",
            "core_policy": core_policy,
            "retained_aedt_bundle": stage["retained_aedt_bundle"],
            "retention_run_root": stage["retention_run_root"],
            "retry_of_dependency_failure": copy.deepcopy(stored_record),
            "dependency_failure_sibling_guard": sibling_guard,
            "dependency_failure_atomic_claim": _claim_receipt(
                acquisition_status=claim_status,
                finalized_claim=finalized_claim,
            ),
            "scheduler_strict_node_contract": {
                "plan_contract": copy.deepcopy(strict_contract),
                **trace,
            },
            "scheduler_cutover_receipt": production._file_record(
                scheduler_cutover_receipt_path.resolve(strict=True)
            ),
            "scheduler_cutover_payload_sha256": cutover[
                "payload_sha256"
            ],
            "scheduler_live_launcher_identity": launcher_after,
            "scheduler_admission_snapshot": admission,
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
        description="Exact-once MFT dependency-failure retry"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init-dependency-failure-claim-root")
    plan = commands.add_parser("plan-dependency-failure-retry")
    plan.add_argument("--original-plan", type=Path, required=True)
    plan.add_argument("--original-submission", type=Path, required=True)
    plan.add_argument(
        "--dependency-anchor-task-id", type=int, required=True
    )
    plan.add_argument("--strict-node-name", required=True)
    plan.add_argument(
        "--scheduler-url", default=probe.DIAGNOSTIC_SCHEDULER_URL
    )
    plan.add_argument("--output", type=Path, required=True)
    submit_parser = commands.add_parser(
        "submit-dependency-failure-retry"
    )
    submit_parser.add_argument("--plan", type=Path, required=True)
    submit_parser.add_argument(
        "--scheduler-cutover-receipt", type=Path, required=True
    )
    submit_parser.add_argument("--priority", type=int, default=100)
    submit_parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "init-dependency-failure-claim-root":
        authority = initialize_claim_root()
        result = Path(authority["resolved_root"])
    elif args.command == "plan-dependency-failure-retry":
        result = create_plan(
            original_plan_path=args.original_plan,
            original_submission_path=args.original_submission,
            dependency_anchor_task_id=args.dependency_anchor_task_id,
            strict_node_name=args.strict_node_name,
            scheduler_url=args.scheduler_url,
            output=args.output,
        )
    else:
        result = submit(
            plan_path=args.plan,
            scheduler_cutover_receipt_path=(
                args.scheduler_cutover_receipt
            ),
            priority=args.priority,
            output=args.output,
        )
    print(json.dumps({"status": "ok", "path": str(result)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
