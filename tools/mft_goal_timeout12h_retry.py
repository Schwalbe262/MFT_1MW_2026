"""Exact-once 12-hour recovery for authenticated 8-hour MFT timeouts.

This module belongs to the MFT project.  It never edits the separate
Scheduler repository.  Planning and collection are GET-only; the only
Scheduler mutation available here is the atomically guarded task POST in
``submit-timeout12h-retry``.

The original r1 authority remains byte-compatible for tasks 96256/96263.
Late task-96258, task-96289, and task-96265 terminals are isolated under
supplemental r2/r3/r4 claim roots so expanding a reviewed mapping cannot
silently widen an older frozen root.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path, PurePosixPath
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


PLAN_SCHEMA = "mft-goal-diagnostic-standard-timeout12h-plan-v1"
SUBMISSION_SCHEMA = (
    "mft-goal-diagnostic-standard-timeout12h-submission-v1"
)
RETRY_EVIDENCE_SCHEMA = (
    "mft-goal-diagnostic-standard-timeout12h-evidence-v1"
)
FAILURE_EVIDENCE_SCHEMA = (
    "mft-goal-diagnostic-standard-timeout12h-task-failure-v1"
)
STREAM_EVIDENCE_SCHEMA = (
    "mft-goal-diagnostic-standard-timeout12h-stream-evidence-v1"
)
RETENTION_AUDIT_SCHEMA = (
    "mft-goal-diagnostic-standard-timeout12h-retention-audit-v1"
)
STORAGE_AUDIT_SCHEMA = (
    "mft-goal-diagnostic-standard-timeout12h-storage-audit-v1"
)
SIBLING_GUARD_SCHEMA = (
    "mft-goal-diagnostic-standard-timeout12h-sibling-guard-v1"
)
CLAIM_RECEIPT_SCHEMA = "mft-goal-timeout12h-atomic-claim-receipt-v1"
PROFILE_SCHEMA = (
    "mft-goal-diagnostic-standard-timeout12h-retry-profile-v1"
)
RETRY_GENERATION = "timeout12h-r1"
SUPPLEMENTAL_RETRY_GENERATION = "timeout12h-r2"
LATE_ANCHOR_RETRY_GENERATION = "timeout12h-r3"
LATE_BOUNDARY_RETRY_GENERATION = "timeout12h-r4"
RESOURCES = {"cpus": 8, "timeout_seconds": 12 * 3600}
MEMORY_MB = 32768
STRICT_NODE_NAME = "n114"
EXACT_LOGICAL_TO_FAILED_TASK = {96218: 96256, 96226: 96263}
SUPPLEMENTAL_EXACT_LOGICAL_TO_FAILED_TASK = {96224: 96258}
LATE_ANCHOR_EXACT_LOGICAL_TO_FAILED_TASK = {96223: 96289}
LATE_BOUNDARY_EXACT_LOGICAL_TO_FAILED_TASK = {96230: 96265}
PROFILE_PATH = (
    probe.REPOSITORY_ROOT
    / "regression_260707"
    / "verify"
    / "profiles"
    / "goal_diagnostic_standard_timeout12h_retry.json"
)
CLAIM_ROOT = Path(
    "C:/Users/peets/slurm_scheduler_runtime/mft_goal_20260726/"
    "timeout12h_claims"
)
SUPPLEMENTAL_CLAIM_ROOT = Path(
    "C:/Users/peets/slurm_scheduler_runtime/mft_goal_20260726/"
    "timeout12h_claims_r2"
)
LATE_ANCHOR_CLAIM_ROOT = Path(
    "C:/Users/peets/slurm_scheduler_runtime/mft_goal_20260726/"
    "timeout12h_claims_r3"
)
LATE_BOUNDARY_CLAIM_ROOT = Path(
    "C:/Users/peets/slurm_scheduler_runtime/mft_goal_20260726/"
    "timeout12h_claims_r4"
)
CLAIM_AUTHORITY_SHA256 = canonical_sha256(
    {
        "campaign_id": "mft-goal-20260726",
        "goal_contract_schema": GOAL_CONTRACT_SCHEMA,
        "hard_spec_sha256": GOAL_STAGE_SPEC_SHA256,
        "temperature_contract_sha256": GOAL_TEMPERATURE_CONTRACT_SHA256,
        "retry_generation": RETRY_GENERATION,
        "exact_logical_to_failed_task": EXACT_LOGICAL_TO_FAILED_TASK,
    }
)
SUPPLEMENTAL_CLAIM_AUTHORITY_SHA256 = canonical_sha256(
    {
        "campaign_id": "mft-goal-20260726",
        "goal_contract_schema": GOAL_CONTRACT_SCHEMA,
        "hard_spec_sha256": GOAL_STAGE_SPEC_SHA256,
        "temperature_contract_sha256": GOAL_TEMPERATURE_CONTRACT_SHA256,
        "retry_generation": SUPPLEMENTAL_RETRY_GENERATION,
        "exact_logical_to_failed_task": (
            SUPPLEMENTAL_EXACT_LOGICAL_TO_FAILED_TASK
        ),
    }
)
LATE_ANCHOR_CLAIM_AUTHORITY_SHA256 = canonical_sha256(
    {
        "campaign_id": "mft-goal-20260726",
        "goal_contract_schema": GOAL_CONTRACT_SCHEMA,
        "hard_spec_sha256": GOAL_STAGE_SPEC_SHA256,
        "temperature_contract_sha256": GOAL_TEMPERATURE_CONTRACT_SHA256,
        "retry_generation": LATE_ANCHOR_RETRY_GENERATION,
        "exact_logical_to_failed_task": (
            LATE_ANCHOR_EXACT_LOGICAL_TO_FAILED_TASK
        ),
    }
)
LATE_BOUNDARY_CLAIM_AUTHORITY_SHA256 = canonical_sha256(
    {
        "campaign_id": "mft-goal-20260726",
        "goal_contract_schema": GOAL_CONTRACT_SCHEMA,
        "hard_spec_sha256": GOAL_STAGE_SPEC_SHA256,
        "temperature_contract_sha256": GOAL_TEMPERATURE_CONTRACT_SHA256,
        "retry_generation": LATE_BOUNDARY_RETRY_GENERATION,
        "exact_logical_to_failed_task": (
            LATE_BOUNDARY_EXACT_LOGICAL_TO_FAILED_TASK
        ),
    }
)
MAX_STREAM_BYTES = 64 * 1024 * 1024
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
    expected.update(
        {
            "schema_version": PROFILE_SCHEMA,
            "comment": (
                "Diagnostic-only eighth-symmetry Standard FEA bounded "
                "12-hour retry after one authenticated 8-hour timeout, "
                "with retained AEDT project and AEDT results"
            ),
            "timeout_seconds": RESOURCES["timeout_seconds"],
        }
    )
    if profile != expected:
        raise HandoffContractError("timeout12h retry profile drifted")
    fixed = dict(profile["param_overrides"])
    fixed["thermal_pad_conductivity_W_mK"] = 0.2
    attest_fixed_identity(fixed)


def _retry_authority(retry_generation: str) -> dict[str, Any]:
    if retry_generation == RETRY_GENERATION:
        return {
            "retry_generation": RETRY_GENERATION,
            "claim_root": CLAIM_ROOT,
            "claim_authority_sha256": CLAIM_AUTHORITY_SHA256,
            "exact_logical_to_failed_task": EXACT_LOGICAL_TO_FAILED_TASK,
        }
    if retry_generation == SUPPLEMENTAL_RETRY_GENERATION:
        return {
            "retry_generation": SUPPLEMENTAL_RETRY_GENERATION,
            "claim_root": SUPPLEMENTAL_CLAIM_ROOT,
            "claim_authority_sha256": (
                SUPPLEMENTAL_CLAIM_AUTHORITY_SHA256
            ),
            "exact_logical_to_failed_task": (
                SUPPLEMENTAL_EXACT_LOGICAL_TO_FAILED_TASK
            ),
        }
    if retry_generation == LATE_ANCHOR_RETRY_GENERATION:
        return {
            "retry_generation": LATE_ANCHOR_RETRY_GENERATION,
            "claim_root": LATE_ANCHOR_CLAIM_ROOT,
            "claim_authority_sha256": (
                LATE_ANCHOR_CLAIM_AUTHORITY_SHA256
            ),
            "exact_logical_to_failed_task": (
                LATE_ANCHOR_EXACT_LOGICAL_TO_FAILED_TASK
            ),
        }
    if retry_generation == LATE_BOUNDARY_RETRY_GENERATION:
        return {
            "retry_generation": LATE_BOUNDARY_RETRY_GENERATION,
            "claim_root": LATE_BOUNDARY_CLAIM_ROOT,
            "claim_authority_sha256": (
                LATE_BOUNDARY_CLAIM_AUTHORITY_SHA256
            ),
            "exact_logical_to_failed_task": (
                LATE_BOUNDARY_EXACT_LOGICAL_TO_FAILED_TASK
            ),
        }
    raise HandoffContractError("timeout12h retry generation is unsupported")


def _retry_authority_for_pair(
    *, logical_authority_task_id: int, failed_task_id: int
) -> dict[str, Any]:
    matches = [
        _retry_authority(generation)
        for generation in (
            RETRY_GENERATION,
            SUPPLEMENTAL_RETRY_GENERATION,
            LATE_ANCHOR_RETRY_GENERATION,
            LATE_BOUNDARY_RETRY_GENERATION,
        )
        if _retry_authority(generation)[
            "exact_logical_to_failed_task"
        ].get(logical_authority_task_id)
        == failed_task_id
    ]
    if len(matches) != 1:
        raise HandoffContractError(
            "timeout12h exact logical/failed task authority is absent"
        )
    return matches[0]


def initialize_claim_root(
    root: Path | None = None,
    *,
    retry_generation: str = RETRY_GENERATION,
) -> dict[str, Any]:
    retry_authority = _retry_authority(retry_generation)
    target = (
        retry_authority["claim_root"] if root is None else Path(root)
    )
    try:
        return atomic_claim.initialize_claim_root(
            target,
            campaign_id="mft-goal-20260726",
            campaign_authority_sha256=retry_authority[
                "claim_authority_sha256"
            ],
        )
    except atomic_claim.ClaimContractError as exc:
        raise HandoffContractError(
            "timeout12h atomic claim root is unavailable"
        ) from exc


def _claim_authority(retry_generation: str) -> dict[str, Any]:
    retry_authority = _retry_authority(retry_generation)
    try:
        authority = atomic_claim.load_claim_root(
            retry_authority["claim_root"]
        )
    except atomic_claim.ClaimContractError as exc:
        raise HandoffContractError(
            "timeout12h atomic claim root is unavailable"
        ) from exc
    if (
        authority.get("campaign_id") != "mft-goal-20260726"
        or authority.get("campaign_authority_sha256")
        != retry_authority["claim_authority_sha256"]
    ):
        raise HandoffContractError("timeout12h claim authority drifted")
    return authority


def _claim_reference(
    *,
    candidate_physics_sha256: str,
    logical_authority_task_id: int,
    retry_generation: str,
) -> dict[str, Any]:
    try:
        return atomic_claim.build_claim_reference(
            _claim_authority(retry_generation),
            candidate_physics_sha256=production._require_sha(
                candidate_physics_sha256,
                "timeout12h candidate physics SHA",
            ),
            logical_authority_task_id=logical_authority_task_id,
            retry_generation=retry_generation,
        )
    except atomic_claim.ClaimContractError as exc:
        raise HandoffContractError(
            "timeout12h atomic claim reference is invalid"
        ) from exc


def _validate_claim_reference(plan: Mapping[str, Any]) -> dict[str, Any]:
    reference = plan.get("timeout12h_atomic_claim_reference")
    record = plan.get("retry_of_timeout12h")
    if not isinstance(reference, Mapping) or not isinstance(record, Mapping):
        raise HandoffContractError(
            "timeout12h atomic claim reference is absent"
        )
    retry_generation = str(record.get("retry_generation") or "")
    try:
        normalized = atomic_claim.validate_claim_reference(
            reference, _claim_authority(retry_generation)
        )
    except atomic_claim.ClaimContractError as exc:
        raise HandoffContractError(
            "timeout12h atomic claim reference drifted"
        ) from exc
    if (
        normalized.get("candidate_physics_sha256")
        != plan.get("candidate_physics_sha256")
        or normalized.get("logical_authority_task_id")
        != record.get("logical_authority_task_id")
        or normalized.get("retry_generation") != retry_generation
    ):
        raise HandoffContractError(
            "timeout12h atomic claim plan binding drifted"
        )
    return normalized


def _task_identity(
    *,
    logical_authority_task_id: int,
    candidate_physics_sha256: str,
    retry_generation: str = RETRY_GENERATION,
) -> tuple[str, str]:
    _retry_authority(retry_generation)
    logical_id = _positive_int(
        logical_authority_task_id, "timeout12h logical task ID"
    )
    stem = production._require_sha(
        candidate_physics_sha256, "timeout12h candidate physics SHA"
    )[:12]
    generation_slug = retry_generation.removeprefix("timeout12h-")
    return (
        "mft-goal-diag-standard-timeout12h-"
        f"{generation_slug}-l{logical_id}-{stem}",
        "mft_goal_diag_standard_timeout12h_"
        f"{generation_slug}_l{logical_id}_{stem}",
    )


def _scheduler_stream(
    *, scheduler_url: str, task_id: int, stream: str
) -> bytes:
    if stream not in {"stdout", "stderr"}:
        raise HandoffContractError("timeout12h stream name is invalid")
    request = production.urllib.request.Request(
        f"{scheduler_url.rstrip('/')}/api/tasks/{task_id}/{stream}",
        headers={"Accept": "text/plain"},
        method="GET",
    )
    try:
        with production.urllib.request.urlopen(
            request, timeout=120.0
        ) as response:
            raw = response.read(MAX_STREAM_BYTES + 1)
    except (OSError, production.urllib.error.URLError) as exc:
        raise HandoffContractError(
            f"timeout12h Scheduler {stream} fetch failed"
        ) from exc
    if not raw or len(raw) > MAX_STREAM_BYTES:
        raise HandoffContractError(
            f"timeout12h Scheduler {stream} byte bound failed"
        )
    return raw


def _scheduler_remote_files(
    *,
    scheduler_url: str,
    task_id: int,
    glob: str,
) -> dict[str, Any]:
    query = production.urllib.parse.urlencode(
        {"glob": glob, "base": "remote_cwd"}
    )
    request = production.urllib.request.Request(
        f"{scheduler_url.rstrip('/')}/api/tasks/{task_id}/remote-files?"
        f"{query}",
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with production.urllib.request.urlopen(
            request, timeout=120.0
        ) as response:
            raw = response.read(8 * 1024 * 1024 + 1)
    except (OSError, production.urllib.error.URLError) as exc:
        raise HandoffContractError(
            "timeout12h retained-file inventory fetch failed"
        ) from exc
    if len(raw) > 8 * 1024 * 1024:
        raise HandoffContractError(
            "timeout12h retained-file inventory exceeds byte bound"
        )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise HandoffContractError(
            "timeout12h retained-file inventory is invalid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise HandoffContractError(
            "timeout12h retained-file inventory is malformed"
        )
    return value


def _timeout_failure_evidence(
    snapshot: Mapping[str, Any],
    *,
    immediate_submission: Mapping[str, Any],
    logical_authority_task_id: int,
    retry_generation: str,
) -> dict[str, Any]:
    evidence = {
        "schema_version": FAILURE_EVIDENCE_SCHEMA,
        "task_id": _task_id(snapshot),
        "logical_authority_task_id": logical_authority_task_id,
        "name": snapshot.get("name"),
        "status": snapshot.get("status"),
        "state": snapshot.get("state"),
        "exit_code": snapshot.get("exit_code"),
        "failure_message": snapshot.get("failure_message"),
        "timeout_seconds": snapshot.get("timeout_seconds"),
        "slurm_job_id": str(snapshot.get("slurm_job_id") or ""),
        "allocation_id": snapshot.get(
            "allocation_id", snapshot.get("assigned_allocation")
        ),
        "account_name": snapshot.get("account_name"),
        "actual_node_name": snapshot.get("actual_node_name"),
        "cpus": snapshot.get("cpus"),
        "memory_mb": snapshot.get("memory_mb"),
        "aedt_backend": snapshot.get("aedt_backend"),
        "project": snapshot.get("project"),
        "dedupe_key": snapshot.get("dedupe_key"),
        "remote_cwd": snapshot.get("remote_cwd"),
        "remote_dir": snapshot.get("remote_dir"),
        "started_at": snapshot.get("started_at"),
        "finished_at": snapshot.get("finished_at"),
    }
    retry_authority = _retry_authority(retry_generation)
    expected_task = retry_authority[
        "exact_logical_to_failed_task"
    ].get(logical_authority_task_id)
    allocation_id = evidence["allocation_id"]
    if (
        expected_task is None
        or evidence["task_id"] != expected_task
        or evidence["task_id"] != immediate_submission["task_id"]
        or evidence["name"] != immediate_submission["task_name"]
        or evidence["status"] != "failed"
        or evidence["state"] != "failed"
        or evidence["exit_code"] != 124
        or evidence["failure_message"] != "task timed out after 28800s"
        or evidence["timeout_seconds"] != 28800
        or evidence["cpus"] != 8
        or evidence["memory_mb"] != MEMORY_MB
        or evidence["aedt_backend"] != "standalone"
        or evidence["project"] != scheduler_client.MFT_PROJECT
        or evidence["dedupe_key"] != immediate_submission["dedupe_key"]
        or not str(evidence["slurm_job_id"]).isdigit()
        or isinstance(allocation_id, bool)
        or not isinstance(allocation_id, int)
        or allocation_id <= 0
        or not str(evidence["account_name"] or "").strip()
        or not str(evidence["actual_node_name"] or "").strip()
        or not str(evidence["remote_cwd"] or "").strip()
        or not str(evidence["remote_dir"] or "").strip()
        or not str(evidence["started_at"] or "").strip()
        or not str(evidence["finished_at"] or "").strip()
    ):
        raise HandoffContractError(
            "timeout12h recovery requires exact failed/124/28800s task "
            "authorized by its immutable retry generation"
        )
    return evidence


def _stream_evidence(
    stdout: bytes | str,
    stderr: bytes | str,
    *,
    task_id: int,
    retry_generation: str,
) -> dict[str, Any]:
    _retry_authority(retry_generation)
    raw_out = stdout.encode("utf-8") if isinstance(stdout, str) else stdout
    raw_err = stderr.encode("utf-8") if isinstance(stderr, str) else stderr
    if (
        not isinstance(raw_out, bytes)
        or not isinstance(raw_err, bytes)
        or not raw_out
        or not raw_err
        or len(raw_out) > MAX_STREAM_BYTES
        or len(raw_err) > MAX_STREAM_BYTES
    ):
        raise HandoffContractError("timeout12h stream evidence is invalid")
    try:
        out_text = raw_out.decode("utf-8")
        err_text = raw_err.decode("utf-8")
    except UnicodeError as exc:
        raise HandoffContractError(
            "timeout12h stream evidence is not UTF-8"
        ) from exc
    marker = "[thermal] native mesh preflight: "
    lines = [
        line
        for line in err_text.splitlines()
        if marker in line
    ]
    if len(lines) != 1:
        raise HandoffContractError(
            "timeout12h native mesh preflight evidence is ambiguous"
        )
    try:
        preflight = json.loads(lines[0].split(marker, 1)[1])
    except json.JSONDecodeError as exc:
        raise HandoffContractError(
            "timeout12h native mesh preflight JSON is invalid"
        ) from exc
    operations = (
        preflight.get("native_operation_readback", {})
        .get("operation_readbacks", [])
    )
    side_levels = [
        row.get("level")
        for row in operations
        if isinstance(row, Mapping)
        and str(row.get("name") or "").startswith(
            "rx_side_block_mesh_level_"
        )
    ]
    artifacts = preflight.get("fresh_mesh_artifacts")
    if not isinstance(artifacts, list):
        raise HandoffContractError(
            "timeout12h premesh artifact inventory is absent"
        )
    grid_bytes = sum(
        _positive_int(
            row.get("grid_output_size"),
            "timeout12h premesh grid size",
        )
        for row in artifacts
        if isinstance(row, Mapping)
    )
    elapsed = preflight.get("elapsed_s")
    minimum_elapsed_seconds = (
        2 * 3600
        if retry_generation == LATE_ANCHOR_RETRY_GENERATION
        else 4 * 3600
    )
    solve_marker = "Solving design setup ThermalSetup"
    if (
        preflight.get("passed") is not True
        or preflight.get("status") != "passed_standalone_native_premesh"
        or preflight.get("mesh_mapping_coverage_passed") is not True
        or preflight.get("native_operation_readback_passed") is not True
        or preflight.get("standalone_idle_barrier_passed") is not True
        or preflight.get("postflight_error") != ""
        or preflight.get("native_errors") != []
        or side_levels != [5]
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not minimum_elapsed_seconds < float(elapsed) < 8 * 3600
        or preflight.get("fresh_mesh_artifact_count") != len(artifacts)
        or len(artifacts) <= 0
        or grid_bytes <= 0
        or solve_marker not in out_text
        or solve_marker not in err_text
        or "srun: forcing job termination" not in err_text
        or "CANCELLED AT " not in err_text
        or "RESULT_JSON:" in out_text
        or "RESULT_JSON:" in err_text
    ):
        raise HandoffContractError(
            "timeout12h is not justified by successful premesh followed "
            "only by the exact Scheduler timeout"
        )
    return {
        "schema_version": STREAM_EVIDENCE_SCHEMA,
        "task_id": task_id,
        "stdout_sha256": production._sha256_bytes(raw_out),
        "stdout_size_bytes": len(raw_out),
        "stderr_sha256": production._sha256_bytes(raw_err),
        "stderr_size_bytes": len(raw_err),
        "native_premesh_elapsed_seconds": float(elapsed),
        "minimum_native_premesh_elapsed_seconds": (
            minimum_elapsed_seconds
        ),
        "native_premesh_passed": True,
        "thermal_solve_dispatched": True,
        "scheduler_forced_termination_observed": True,
        "result_json_absent": True,
        "rx_side_block_mesh_level": 5,
        "fresh_mesh_artifact_count": len(artifacts),
        "fresh_grid_output_bytes": grid_bytes,
        "mesh_policy": preflight.get("mesh_policy"),
    }


def _retention_audit(
    inventory: Mapping[str, Any],
    *,
    task_id: int,
    retained: Mapping[str, Any],
) -> dict[str, Any]:
    expected_glob = f"{retained['relative_directory']}/**"
    files = inventory.get("files")
    if (
        inventory.get("base") != "remote_cwd"
        or inventory.get("glob") != expected_glob
        or not isinstance(files, list)
    ):
        raise HandoffContractError(
            "timeout12h retained-file audit identity drifted"
        )
    normalized: list[dict[str, Any]] = []
    for row in files:
        if not isinstance(row, Mapping):
            raise HandoffContractError(
                "timeout12h retained-file row is malformed"
            )
        path = str(row.get("path") or "")
        pure = PurePosixPath(path)
        if (
            not path
            or pure.is_absolute()
            or ".." in pure.parts
            or "\\" in path
            or not path.startswith(
                f"{retained['relative_directory']}/"
            )
        ):
            raise HandoffContractError(
                "timeout12h retained-file path is unsafe"
            )
        normalized.append(copy.deepcopy(dict(row)))
    receipt_present = any(
        row.get("path") == retained["receipt_path"] for row in normalized
    )
    manifest_present = any(
        row.get("path") == retained["results_manifest_path"]
        for row in normalized
    )
    if receipt_present or manifest_present:
        raise HandoffContractError(
            "failed timeout task unexpectedly has a complete retained bundle"
        )
    return {
        "schema_version": RETENTION_AUDIT_SCHEMA,
        "task_id": task_id,
        "base": "remote_cwd",
        "glob": expected_glob,
        "files": normalized,
        "file_count": len(normalized),
        "complete_receipt_absent": True,
        "complete_results_manifest_absent": True,
        "fresh_distinct_retention_identity_required": True,
    }


def _storage_audit(
    *,
    observed_at_kst: str,
    account_name: str,
    used_gb: float,
    in_doubt_gb: float,
    limit_gb: float,
    observed_free_gb: float,
    prospective_grid_gb: float,
) -> dict[str, Any]:
    values = (
        used_gb,
        in_doubt_gb,
        limit_gb,
        observed_free_gb,
        prospective_grid_gb,
    )
    if (
        not isinstance(observed_at_kst, str)
        or not observed_at_kst.endswith("+09:00")
        or account_name != "r1jae262"
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value < 0
            for value in values
        )
        or abs((limit_gb - used_gb - in_doubt_gb) - observed_free_gb)
        > 0.02
        or observed_free_gb < prospective_grid_gb + 10.0
    ):
        raise HandoffContractError(
            "timeout12h read-only storage audit is unsafe"
        )
    return {
        "schema_version": STORAGE_AUDIT_SCHEMA,
        "observed_at_kst": observed_at_kst,
        "source": "read_only_gpfs_quota_probe",
        "account_name": account_name,
        "used_gb": float(used_gb),
        "in_doubt_gb": float(in_doubt_gb),
        "limit_gb": float(limit_gb),
        "observed_free_gb": float(observed_free_gb),
        "prospective_grid_gb": float(prospective_grid_gb),
        "safety_floor_gb": 10.0,
        "single_task_arithmetic_passed": True,
        "fresh_pre_submit_storage_reauthentication_required": True,
        "parallel_submit_without_reaudit_allowed": False,
    }


def create_plan(
    *,
    immediate_plan_path: Path,
    immediate_submission_path: Path,
    strict_node_name: str,
    output: Path,
    storage_observed_at_kst: str,
    storage_used_gb: float,
    storage_in_doubt_gb: float,
    storage_limit_gb: float,
    storage_observed_free_gb: float,
    scheduler_url: str = probe.DIAGNOSTIC_SCHEDULER_URL,
    task_reader: Any = None,
    stdout_reader: Any = None,
    stderr_reader: Any = None,
    remote_files_reader: Any = None,
) -> Path:
    immediate_plan, params, selected = probe._load_plan(
        immediate_plan_path
    )
    if (
        probe._plan_retry_kind(immediate_plan) != "timeout"
        or immediate_plan["stage"]["resources"]
        != probe.TIMEOUT_RETRY_RESOURCES
    ):
        raise HandoffContractError(
            "timeout12h recovery requires exactly one 8-hour timeout parent"
        )
    immediate_submission = probe._load_submission(
        immediate_submission_path, plan=immediate_plan
    )
    (
        logical_plan,
        logical_submission,
        _original_timeout_evidence,
    ) = probe._validate_timeout_retry_record(immediate_plan)
    logical_id = _positive_int(
        logical_submission["task_id"], "timeout12h logical authority task"
    )
    immediate_task_id = _positive_int(
        immediate_submission["task_id"],
        "timeout12h immediate failed task",
    )
    retry_authority = _retry_authority_for_pair(
        logical_authority_task_id=logical_id,
        failed_task_id=immediate_task_id,
    )
    retry_generation = str(retry_authority["retry_generation"])
    if (
        scheduler_url.rstrip("/")
        != immediate_submission["scheduler_url"]
        or strict_node_name != STRICT_NODE_NAME
    ):
        raise HandoffContractError(
            "timeout12h exact task mapping, Scheduler, or n114 drifted"
        )
    reader = task_reader or probe._scheduler_task_snapshot
    read_stdout = stdout_reader or (
        lambda **kwargs: _scheduler_stream(stream="stdout", **kwargs)
    )
    read_stderr = stderr_reader or (
        lambda **kwargs: _scheduler_stream(stream="stderr", **kwargs)
    )
    read_remote_files = remote_files_reader or _scheduler_remote_files
    execution = _timeout_failure_evidence(
        reader(
            scheduler_url=scheduler_url.rstrip("/"),
            task_id=int(immediate_submission["task_id"]),
        ),
        immediate_submission=immediate_submission,
        logical_authority_task_id=logical_id,
        retry_generation=retry_generation,
    )
    stream = _stream_evidence(
        read_stdout(
            scheduler_url=scheduler_url.rstrip("/"),
            task_id=int(immediate_submission["task_id"]),
        ),
        read_stderr(
            scheduler_url=scheduler_url.rstrip("/"),
            task_id=int(immediate_submission["task_id"]),
        ),
        task_id=int(immediate_submission["task_id"]),
        retry_generation=retry_generation,
    )
    source_retained = immediate_submission["retained_aedt_bundle"]
    retention = _retention_audit(
        read_remote_files(
            scheduler_url=scheduler_url.rstrip("/"),
            task_id=int(immediate_submission["task_id"]),
            glob=f"{source_retained['relative_directory']}/**",
        ),
        task_id=int(immediate_submission["task_id"]),
        retained=source_retained,
    )
    storage = _storage_audit(
        observed_at_kst=storage_observed_at_kst,
        account_name="r1jae262",
        used_gb=storage_used_gb,
        in_doubt_gb=storage_in_doubt_gb,
        limit_gb=storage_limit_gb,
        observed_free_gb=storage_observed_free_gb,
        prospective_grid_gb=stream["fresh_grid_output_bytes"]
        / (1024**3),
    )
    immediate_profile = production._read_json(
        immediate_plan_path.resolve(strict=True).parent
        / immediate_plan["profile"]["path"]
    )
    profile, profile_source = _profile_content()
    if (
        profile["param_overrides"]
        != immediate_profile["param_overrides"]
        or profile["fixed_boundary_contract"]
        != immediate_profile["fixed_boundary_contract"]
        or production._effective_params(params, profile)
        != production._effective_params(params, immediate_profile)
    ):
        raise HandoffContractError(
            "timeout12h retry changes fixed physics"
        )
    task_name, workdir = _task_identity(
        logical_authority_task_id=logical_id,
        candidate_physics_sha256=immediate_plan[
            "candidate_physics_sha256"
        ],
        retry_generation=retry_generation,
    )
    strict_contract = probe._strict_node_plan_contract(strict_node_name)
    retained = scheduler_client.retained_aedt_identity(
        task_name,
        params,
        profile,
        immediate_plan["solver_revision"],
        immediate_plan["library_revision"],
    )
    if (
        retained is None
        or retained["dedupe_key"] == source_retained["dedupe_key"]
        or retained["relative_directory"]
        == source_retained["relative_directory"]
        or retained["profile_sha256"] == source_retained["profile_sha256"]
    ):
        raise HandoffContractError(
            "timeout12h retry identity is not distinct"
        )
    record = {
        "schema_version": RETRY_EVIDENCE_SCHEMA,
        "retry_generation": retry_generation,
        "logical_authority_task_id": logical_id,
        "retry_of_task_id": immediate_submission["task_id"],
        "immediate_parent_kind": "timeout8h",
        "immediate_plan": production._file_record(
            immediate_plan_path.resolve(strict=True)
        ),
        "immediate_plan_payload_sha256": immediate_plan[
            "payload_sha256"
        ],
        "immediate_submission": production._file_record(
            immediate_submission_path.resolve(strict=True)
        ),
        "immediate_submission_payload_sha256": immediate_submission[
            "payload_sha256"
        ],
        "immediate_parent_ancestry_sha256": canonical_sha256(
            immediate_plan["retry_of_timeout"]
        ),
        "timeout_failure_evidence": execution,
        "timeout_failure_evidence_sha256": canonical_sha256(execution),
        "stream_evidence": stream,
        "stream_evidence_sha256": canonical_sha256(stream),
        "source_retention_audit": retention,
        "source_retention_audit_sha256": canonical_sha256(retention),
        "storage_audit": storage,
        "storage_audit_sha256": canonical_sha256(storage),
        "timeout_change_only": True,
        "source_timeout_seconds": 8 * 3600,
        "target_timeout_seconds": 12 * 3600,
        "fixed_physics_unchanged": True,
        "collector_lineage_parent_plan_payload_sha256": logical_plan[
            "payload_sha256"
        ],
        "collector_lineage_parent_submission_payload_sha256": (
            logical_submission["payload_sha256"]
        ),
    }
    reference = _claim_reference(
        candidate_physics_sha256=immediate_plan[
            "candidate_physics_sha256"
        ],
        logical_authority_task_id=logical_id,
        retry_generation=retry_generation,
    )
    destination = output.resolve()
    if destination.exists():
        raise HandoffContractError(
            f"timeout12h plan output exists: {destination}"
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
            staging / "diagnostic_standard_timeout12h_profile.json",
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
                    "effective_params_sha256": canonical_sha256(
                        production._effective_params(params, profile)
                    ),
                    "resources": copy.deepcopy(RESOURCES),
                    "retained_aedt_bundle": retained,
                    "retention_run_root": (
                        probe._retention_run_root_evidence(retained)
                    ),
                },
                "available_submission_commands": [
                    "submit-timeout12h-retry"
                ],
                "retry_of_timeout12h": record,
                "timeout12h_atomic_claim_reference": reference,
                "scheduler_strict_node_contract": strict_contract,
            }
        )
        plan_path = production._write_immutable_json(
            staging / "diagnostic_timeout12h_retry_plan.json",
            production._seal(unsigned),
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / plan_path.name


def _plan_artifact(
    root: Path, record: Any, label: str
) -> Path:
    if not isinstance(record, Mapping):
        raise HandoffContractError(f"{label} record is absent")
    target = production._contained_file(root, record.get("path"), label)
    if production._sha256_file(target) != production._require_sha(
        record.get("sha256"), f"{label} SHA"
    ):
        raise HandoffContractError(f"{label} bytes drifted")
    return target


def _validate_retry_record(
    plan: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    record = plan.get("retry_of_timeout12h")
    if not isinstance(record, Mapping):
        raise HandoffContractError("timeout12h ancestry record is absent")
    required = {
        "schema_version",
        "retry_generation",
        "logical_authority_task_id",
        "retry_of_task_id",
        "immediate_parent_kind",
        "immediate_plan",
        "immediate_plan_payload_sha256",
        "immediate_submission",
        "immediate_submission_payload_sha256",
        "immediate_parent_ancestry_sha256",
        "timeout_failure_evidence",
        "timeout_failure_evidence_sha256",
        "stream_evidence",
        "stream_evidence_sha256",
        "source_retention_audit",
        "source_retention_audit_sha256",
        "storage_audit",
        "storage_audit_sha256",
        "timeout_change_only",
        "source_timeout_seconds",
        "target_timeout_seconds",
        "fixed_physics_unchanged",
        "collector_lineage_parent_plan_payload_sha256",
        "collector_lineage_parent_submission_payload_sha256",
    }
    retry_generation = str(record.get("retry_generation") or "")
    retry_authority = _retry_authority(retry_generation)
    if (
        set(record) != required
        or record.get("schema_version") != RETRY_EVIDENCE_SCHEMA
        or record.get("immediate_parent_kind") != "timeout8h"
        or record.get("timeout_change_only") is not True
        or record.get("fixed_physics_unchanged") is not True
        or record.get("source_timeout_seconds") != 8 * 3600
        or record.get("target_timeout_seconds") != 12 * 3600
    ):
        raise HandoffContractError("timeout12h ancestry record drifted")
    immediate_plan_path = Path(
        str(record["immediate_plan"].get("path") or "")
    ).resolve(strict=True)
    immediate_submission_path = Path(
        str(record["immediate_submission"].get("path") or "")
    ).resolve(strict=True)
    if (
        production._file_record(immediate_plan_path)
        != record["immediate_plan"]
        or production._file_record(immediate_submission_path)
        != record["immediate_submission"]
    ):
        raise HandoffContractError(
            "timeout12h immediate-parent bytes drifted"
        )
    immediate_plan, _params, _selected = probe._load_plan(
        immediate_plan_path
    )
    if probe._plan_retry_kind(immediate_plan) != "timeout":
        raise HandoffContractError(
            "timeout12h parent is not exactly one timeout retry"
        )
    immediate_submission = probe._load_submission(
        immediate_submission_path, plan=immediate_plan
    )
    (
        logical_plan,
        logical_submission,
        _first_timeout,
    ) = probe._validate_timeout_retry_record(immediate_plan)
    logical_id = _positive_int(
        record.get("logical_authority_task_id"),
        "timeout12h logical authority task",
    )
    execution = record.get("timeout_failure_evidence")
    stream = record.get("stream_evidence")
    retention = record.get("source_retention_audit")
    storage = record.get("storage_audit")
    if (
        record.get("retry_of_task_id")
        != immediate_submission["task_id"]
        or retry_authority["exact_logical_to_failed_task"].get(logical_id)
        != immediate_submission["task_id"]
        or logical_submission["task_id"] != logical_id
        or record.get("immediate_plan_payload_sha256")
        != immediate_plan["payload_sha256"]
        or record.get("immediate_submission_payload_sha256")
        != immediate_submission["payload_sha256"]
        or record.get("immediate_parent_ancestry_sha256")
        != canonical_sha256(immediate_plan["retry_of_timeout"])
        or record.get("collector_lineage_parent_plan_payload_sha256")
        != logical_plan["payload_sha256"]
        or record.get(
            "collector_lineage_parent_submission_payload_sha256"
        )
        != logical_submission["payload_sha256"]
        or not isinstance(execution, Mapping)
        or record.get("timeout_failure_evidence_sha256")
        != canonical_sha256(execution)
        or not isinstance(stream, Mapping)
        or record.get("stream_evidence_sha256")
        != canonical_sha256(stream)
        or not isinstance(retention, Mapping)
        or record.get("source_retention_audit_sha256")
        != canonical_sha256(retention)
        or not isinstance(storage, Mapping)
        or record.get("storage_audit_sha256")
        != canonical_sha256(storage)
        or execution.get("schema_version") != FAILURE_EVIDENCE_SCHEMA
        or execution.get("task_id") != immediate_submission["task_id"]
        or execution.get("logical_authority_task_id") != logical_id
        or execution.get("exit_code") != 124
        or execution.get("timeout_seconds") != 28800
        or stream.get("schema_version") != STREAM_EVIDENCE_SCHEMA
        or stream.get("task_id") != immediate_submission["task_id"]
        or stream.get("native_premesh_passed") is not True
        or stream.get("thermal_solve_dispatched") is not True
        or stream.get("scheduler_forced_termination_observed") is not True
        or retention.get("schema_version") != RETENTION_AUDIT_SCHEMA
        or retention.get("task_id") != immediate_submission["task_id"]
        or retention.get("complete_receipt_absent") is not True
        or storage.get("schema_version") != STORAGE_AUDIT_SCHEMA
        or storage.get("single_task_arithmetic_passed") is not True
        or storage.get("parallel_submit_without_reaudit_allowed") is not False
        or any(
            plan.get(name) != authority.get(name)
            for authority in (immediate_plan, logical_plan)
            for name in (
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
        raise HandoffContractError("timeout12h ancestry drifted")
    return (
        immediate_plan,
        immediate_submission,
        logical_plan,
        logical_submission,
    )


def _load_plan(
    path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    resolved = path.resolve(strict=True)
    plan = production._validate_seal(
        production._read_json(resolved), PLAN_SCHEMA
    )
    startup_successor = isinstance(
        plan.get("startup_successor"), Mapping
    )
    expected_commands = (
        ["submit-startup-retry"]
        if startup_successor
        else ["submit-timeout12h-retry"]
    )
    if (
        plan.get("campaign_id") != "mft-goal-20260726"
        or plan.get("goal_contract_schema") != GOAL_CONTRACT_SCHEMA
        or plan.get("hard_spec") != GOAL_STAGE_SPEC
        or plan.get("hard_spec_sha256") != GOAL_STAGE_SPEC_SHA256
        or plan.get("temperature_contract_sha256")
        != GOAL_TEMPERATURE_CONTRACT_SHA256
        or plan.get("available_submission_commands") != expected_commands
        or plan.get("physics_override_allowed") is not False
        or plan.get("scheduler_repository_modified") is not False
        or plan.get("scheduler_project_mutation_performed") is not False
        or plan.get("scheduler_submission_performed") is not False
        or plan.get("retention_required") is not True
        or plan.get("prune_protection_required") is not True
        or any(
            plan.get(name) is not expected
            for name, expected in _flags().items()
        )
    ):
        raise HandoffContractError("timeout12h plan contract drifted")
    root = resolved.parent
    params = production._read_json(
        _plan_artifact(root, plan.get("fea_params"), "timeout12h params")
    )
    if (
        set(params) != set(probe.ALL_INPUT_KEYS)
        or canonical_sha256(params) != plan.get("fea_params_sha256")
    ):
        raise HandoffContractError(
            "timeout12h parameter identity drifted"
        )
    selected = production._validate_seal(
        production._read_json(
            _plan_artifact(
                root,
                plan.get("selected_candidate"),
                "timeout12h selected candidate",
            )
        ),
        probe.SELECTED_SCHEMA,
    )
    if (
        selected.get("row_contract", {}).get(
            "physical_geometry_sha256"
        )
        != plan.get("candidate_physics_sha256")
        or selected.get("row_contract", {}).get("fea_params_sha256")
        != plan.get("fea_params_sha256")
        or canonical_sha256(probe._authentication_payload(selected))
        != plan.get("search_authority_sha256")
        or any(
            selected.get(name) is not expected
            for name, expected in _flags().items()
        )
    ):
        raise HandoffContractError(
            "timeout12h selected candidate identity drifted"
        )
    profile_record = plan.get("profile")
    profile_path = _plan_artifact(
        root, profile_record, "timeout12h profile"
    )
    profile = production._read_json(profile_path)
    _validate_profile(profile)
    stage = plan.get("stage")
    strict_contract = probe._validate_strict_node_plan_contract(
        plan.get("scheduler_strict_node_contract")
    )
    if startup_successor:
        from tools import mft_goal_startup_retry

        expected_task_name, expected_workdir = (
            mft_goal_startup_retry._task_identity(
                plan["candidate_physics_sha256"]
            )
        )
    else:
        expected_task_name, expected_workdir = _task_identity(
            logical_authority_task_id=plan["retry_of_timeout12h"][
                "logical_authority_task_id"
            ],
            candidate_physics_sha256=plan["candidate_physics_sha256"],
            retry_generation=plan["retry_of_timeout12h"][
                "retry_generation"
            ],
        )
    retained = scheduler_client.retained_aedt_identity(
        expected_task_name,
        params,
        profile,
        plan["solver_revision"],
        plan["library_revision"],
    )
    if (
        not isinstance(stage, Mapping)
        or profile_record.get("canonical_sha256")
        != canonical_sha256(profile)
        or stage.get("name") != "standard"
        or stage.get("task_name") != expected_task_name
        or stage.get("workdir") != expected_workdir
        or stage.get("profile_sha256") != canonical_sha256(profile)
        or stage.get("effective_params_sha256")
        != canonical_sha256(production._effective_params(params, profile))
        or stage.get("resources") != RESOURCES
        or stage.get("retained_aedt_bundle") != retained
        or stage.get("retention_run_root")
        != probe._retention_run_root_evidence(retained)
        or stage.get("scheduler_project") != scheduler_client.MFT_PROJECT
        or stage.get("scheduler_url") != probe.DIAGNOSTIC_SCHEDULER_URL
        or stage.get("aedt_backend") != "standalone"
        or stage.get("full_model") != 0
        or stage.get("thermal_symmetry") != "eighth"
        or strict_contract.get("requested_node_name") != STRICT_NODE_NAME
        or strict_contract.get("node_name_policy") != "strict"
    ):
        raise HandoffContractError(
            "timeout12h execution identity drifted"
        )
    _validate_claim_reference(plan)
    (
        _immediate_plan,
        immediate_submission,
        _logical_plan,
        _logical_submission,
    ) = _validate_retry_record(plan)
    immediate_profile = production._read_json(
        Path(plan["retry_of_timeout12h"]["immediate_plan"]["path"])
        .resolve(strict=True)
        .parent
        / _immediate_plan["profile"]["path"]
    )
    if (
        profile["param_overrides"]
        != immediate_profile["param_overrides"]
        or profile["fixed_boundary_contract"]
        != immediate_profile["fixed_boundary_contract"]
        or production._effective_params(params, profile)
        != production._effective_params(params, immediate_profile)
    ):
        raise HandoffContractError(
            "timeout12h fixed physics changed"
        )
    if startup_successor:
        mft_goal_startup_retry.validate_plan_overlay(
            plan_path=resolved,
            plan=plan,
            params=params,
            selected=selected,
            profile=profile,
        )
    return plan, params, selected, immediate_submission


def load_plan_for_probe(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan, params, selected, _immediate_submission = _load_plan(path)
    return plan, params, selected


def _sibling_prefix(plan: Mapping[str, Any]) -> str:
    record = plan["retry_of_timeout12h"]
    task_name, _workdir = _task_identity(
        logical_authority_task_id=record["logical_authority_task_id"],
        candidate_physics_sha256=plan["candidate_physics_sha256"],
        retry_generation=record["retry_generation"],
    )
    return task_name.rsplit("-", 1)[0] + "-"


def _normalized_task(snapshot: Mapping[str, Any]) -> dict[str, Any]:
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
        "actual_node_name",
        "allocation_node_name",
        "slurm_job_id",
        "cpus",
        "memory_mb",
        "timeout_seconds",
        "aedt_backend",
        "project",
        "dedupe_key",
        "scheduling_profile",
        "required_capability",
        "env_profile",
        "requested_node_name",
        "node_name",
        "requested_node_name_policy",
        "node_name_policy",
        "strict_node_placement",
        "placement_contract_satisfied",
        "same_node_as_task_id",
        "remote_cwd",
        "remote_dir",
    )
    row = {name: snapshot.get(name) for name in fields}
    row["task_id"] = _task_id(snapshot)
    row["requested_node_name"] = snapshot.get(
        "requested_node_name", snapshot.get("node_name")
    )
    row["requested_node_name_policy"] = snapshot.get(
        "requested_node_name_policy",
        snapshot.get("node_name_policy"),
    )
    row["same_node_as_task_id"] = snapshot.get(
        "same_node_as_task_id", 0
    )
    return row


def _sibling_snapshot(
    rows: Any, *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes, bytearray))
    ):
        raise HandoffContractError(
            "timeout12h sibling task inventory is absent"
        )
    stage = plan["stage"]
    expected_name = stage["task_name"]
    expected_dedupe = stage["retained_aedt_bundle"]["dedupe_key"]
    prefix = _sibling_prefix(plan)
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
        normalized = _normalized_task(raw)
        if (
            normalized["name"] != expected_name
            or normalized["dedupe_key"] != expected_dedupe
            or normalized["project"] != scheduler_client.MFT_PROJECT
            or normalized["cpus"] != RESOURCES["cpus"]
            or normalized["memory_mb"] != MEMORY_MB
            or normalized["timeout_seconds"]
            != RESOURCES["timeout_seconds"]
            or normalized["aedt_backend"] != "standalone"
            or normalized["requested_node_name"] != STRICT_NODE_NAME
            or normalized["requested_node_name_policy"] != "strict"
            or normalized["same_node_as_task_id"] != 0
        ):
            raise HandoffContractError(
                "timeout12h sibling identity collision detected"
            )
        matching.append(normalized)
    matching.sort(key=lambda row: row["task_id"])
    if len(matching) > 1:
        raise HandoffContractError(
            "more than one timeout12h retry sibling exists"
        )
    snapshot = {
        "schema_version": SIBLING_GUARD_SCHEMA,
        "identity": {
            "logical_authority_task_id": plan[
                "retry_of_timeout12h"
            ]["logical_authority_task_id"],
            "immediate_task_id": plan["retry_of_timeout12h"][
                "retry_of_task_id"
            ],
            "retry_generation": plan["retry_of_timeout12h"][
                "retry_generation"
            ],
            "name_prefix": prefix,
            "candidate_physics_sha256": plan[
                "candidate_physics_sha256"
            ],
            "task_name": expected_name,
            "workdir": stage["workdir"],
            "dedupe_key": expected_dedupe,
            "requested_node_name": STRICT_NODE_NAME,
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
            "timeout12h sibling identity changed during submission"
        )
    return {
        "schema_version": SIBLING_GUARD_SCHEMA,
        "before_submission": copy.deepcopy(dict(before)),
        "after_submission": copy.deepcopy(dict(after)),
        "exactly_one_sibling_after_submission": True,
    }


def _claim_winner(
    plan_path: Path, plan: Mapping[str, Any]
) -> dict[str, Any]:
    stage = plan["stage"]
    record = plan["retry_of_timeout12h"]
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
            "memory_mb": MEMORY_MB,
        },
        "task_name": stage["task_name"],
        "dedupe_key": stage["retained_aedt_bundle"]["dedupe_key"],
    }


def _claim_task_evidence(
    task: Mapping[str, Any], pending: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(task, Mapping) or not isinstance(pending, Mapping):
        raise atomic_claim.ClaimContractError(
            "timeout12h claim task evidence is absent"
        )
    row = _normalized_task(task)
    winner = pending.get("winner")
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
        or row.get("requested_node_name") != STRICT_NODE_NAME
        or row.get("requested_node_name_policy") != "strict"
        or row.get("same_node_as_task_id") != 0
        or not str(row.get("status") or "").strip()
        or not str(row.get("state") or "").strip()
    ):
        raise atomic_claim.ClaimContractError(
            "timeout12h claim task identity drifted"
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
            "timeout12h claim acquisition status drifted"
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
    for attempt in range(MAX_POST_RECONCILIATION_READS):
        latest = _sibling_snapshot(
            task_list_reader(
                scheduler_url=scheduler_url,
                project=scheduler_client.MFT_PROJECT,
                task_name=_sibling_prefix(plan),
            ),
            plan=plan,
        )
        if latest["matching_task_count"] == 1:
            return latest
        if attempt + 1 < MAX_POST_RECONCILIATION_READS:
            wait()
    raise HandoffContractError(
        "timeout12h post-POST sibling did not become visible"
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
    stdout_reader: Any = None,
    stderr_reader: Any = None,
    remote_files_reader: Any = None,
    task_list_reader: Any = None,
    reconciliation_waiter: Any = None,
    additional_pre_submit_guard: Any = None,
) -> Path:
    target = output.resolve()
    if target.exists():
        raise HandoffContractError(
            f"timeout12h submission receipt exists: {target}"
        )
    if (
        additional_pre_submit_guard is not None
        and not callable(additional_pre_submit_guard)
    ):
        raise HandoffContractError(
            "timeout12h additional pre-submit guard is not callable"
        )
    plan, params, selected, immediate_submission = _load_plan(plan_path)
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
            "Scheduler cutover endpoint differs from timeout12h plan"
        )
    admission = probe._live_scheduler_admission_snapshot(
        scheduler_url=stage["scheduler_url"], reader=live_reader
    )
    launcher_after = probe._live_launcher_identity(
        cutover, expected_sha256=strict_pin["launcher_sha256"]
    )
    if launcher_after != launcher_before:
        raise HandoffContractError(
            "Scheduler launcher changed during timeout12h admission"
        )
    reader = task_reader or probe._scheduler_task_snapshot
    read_stdout = stdout_reader or (
        lambda **kwargs: _scheduler_stream(stream="stdout", **kwargs)
    )
    read_stderr = stderr_reader or (
        lambda **kwargs: _scheduler_stream(stream="stderr", **kwargs)
    )
    read_remote_files = remote_files_reader or _scheduler_remote_files
    read_tasks = task_list_reader or probe._scheduler_project_tasks
    stored = plan["retry_of_timeout12h"]

    def revalidate_parent() -> None:
        latest = _timeout_failure_evidence(
            reader(
                scheduler_url=stage["scheduler_url"],
                task_id=int(immediate_submission["task_id"]),
            ),
            immediate_submission=immediate_submission,
            logical_authority_task_id=int(
                stored["logical_authority_task_id"]
            ),
            retry_generation=str(stored["retry_generation"]),
        )
        latest_stream = _stream_evidence(
            read_stdout(
                scheduler_url=stage["scheduler_url"],
                task_id=int(immediate_submission["task_id"]),
            ),
            read_stderr(
                scheduler_url=stage["scheduler_url"],
                task_id=int(immediate_submission["task_id"]),
            ),
            task_id=int(immediate_submission["task_id"]),
            retry_generation=str(stored["retry_generation"]),
        )
        source_retained = immediate_submission["retained_aedt_bundle"]
        latest_retention = _retention_audit(
            read_remote_files(
                scheduler_url=stage["scheduler_url"],
                task_id=int(immediate_submission["task_id"]),
                glob=f"{source_retained['relative_directory']}/**",
            ),
            task_id=int(immediate_submission["task_id"]),
            retained=source_retained,
        )
        if (
            latest != stored["timeout_failure_evidence"]
            or latest_stream != stored["stream_evidence"]
            or latest_retention != stored["source_retention_audit"]
        ):
            raise HandoffContractError(
                "timeout12h parent evidence changed before submission"
            )

    revalidate_parent()
    sibling_before = _sibling_snapshot(
        read_tasks(
            scheduler_url=stage["scheduler_url"],
            project=scheduler_client.MFT_PROJECT,
            task_name=_sibling_prefix(plan),
        ),
        plan=plan,
    )
    reference = _validate_claim_reference(plan)
    winner = _claim_winner(plan_path, plan)
    claim_root = _retry_authority(
        str(stored["retry_generation"])
    )["claim_root"]
    try:
        acquisition = atomic_claim.acquire_claim(
            claim_root, reference, winner
        )
    except atomic_claim.ClaimContractError as exc:
        raise HandoffContractError(
            "timeout12h atomic claim acquisition failed"
        ) from exc
    claim_status = acquisition["status"]
    locked_guard_count = 0

    def locked_pre_submit_guard() -> None:
        nonlocal sibling_before, locked_guard_count
        revalidate_parent()
        sibling_before = _sibling_snapshot(
            read_tasks(
                scheduler_url=stage["scheduler_url"],
                project=scheduler_client.MFT_PROJECT,
                task_name=_sibling_prefix(plan),
            ),
            plan=plan,
        )
        if sibling_before["matching_task_count"] != 0:
            raise HandoffContractError(
                "fresh timeout12h claim requires an empty sibling slot"
            )
        if additional_pre_submit_guard is not None:
            additional_pre_submit_guard()
        locked_guard_count += 1

    finalized_claim = None
    if claim_status != "fresh_pending":
        if sibling_before["matching_task_count"] != 1:
            raise HandoffContractError(
                "timeout12h claim recovery requires exactly one sibling "
                "and never re-POSTs"
            )
        recovered = sibling_before["matching_tasks"][0]
        try:
            if claim_status == "existing_pending":
                finalized_claim = atomic_claim.recover_pending_claim(
                    claim_root,
                    reference,
                    acquisition["claim"],
                    matching_tasks=[recovered],
                    sibling_snapshot=sibling_before,
                    evidence_validator=_claim_task_evidence,
                )
            elif claim_status == "existing_finalized":
                finalized_claim = atomic_claim.validate_finalized_claim(
                    claim_root,
                    reference,
                    claim=acquisition["claim"],
                    expected_winner=winner,
                )
                _claim_task_evidence(
                    recovered, finalized_claim["pending_claim"]
                )
                if recovered["task_id"] != finalized_claim["task_id"]:
                    raise atomic_claim.ClaimContractError(
                        "finalized timeout12h claim task drifted"
                    )
            else:
                raise atomic_claim.ClaimContractError(
                    "timeout12h claim status is unsupported"
                )
        except atomic_claim.ClaimContractError as exc:
            raise HandoffContractError(
                "timeout12h atomic claim recovery failed closed"
            ) from exc
        submission_result = {
            "task_id": finalized_claim["task_id"],
            "submission_source": "pre_submission_reconciliation",
            "scheduler_mutation_performed": False,
            "api_pre_submission_readback": recovered,
            "api_post_submission_response": None,
        }
        sibling_after = copy.deepcopy(sibling_before)
        _environment, core_policy = production._submission_environment(
            stage="standard",
            solver_revision=plan["solver_revision"],
            license_snapshot_path=None,
        )
    else:
        if sibling_before["matching_task_count"] != 0:
            raise HandoffContractError(
                "fresh timeout12h claim found an existing sibling"
            )
        profile = production._read_json(
            plan_path.resolve(strict=True).parent
            / plan["profile"]["path"]
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
            account_name=str(
                stored["storage_audit"]["account_name"]
            ),
            node_name=STRICT_NODE_NAME,
            node_name_policy="strict",
            return_submission_evidence=True,
        )
        if locked_guard_count != 1:
            raise HandoffContractError(
                "timeout12h submission lacks exactly one locked guard"
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
            "timeout12h strict submission returned no API evidence"
        )
    new_task_id = _positive_int(
        submission_result.get("task_id"),
        "timeout12h submitted task ID",
    )
    if new_task_id in {
        stored["retry_of_task_id"],
        stored["logical_authority_task_id"],
    }:
        raise HandoffContractError(
            "timeout12h retry resolved to an ancestry task"
        )
    sibling_guard = _sibling_contract(
        before=sibling_before,
        after=sibling_after,
        task_id=new_task_id,
    )
    if claim_status == "fresh_pending":
        try:
            finalized_claim = atomic_claim.finalize_claim(
                claim_root,
                reference,
                acquisition["claim"],
                task_id=new_task_id,
                task_readback=sibling_after["matching_tasks"][0],
                sibling_snapshot=sibling_after,
                evidence_validator=_claim_task_evidence,
            )
        except atomic_claim.ClaimContractError as exc:
            raise HandoffContractError(
                "timeout12h atomic claim finalization failed"
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
            "timeout12h durable strict-node readback drifted"
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
            "timeout12h submission has no API readback"
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
            "retry_of_timeout12h": copy.deepcopy(stored),
            "timeout12h_sibling_guard": sibling_guard,
            "timeout12h_atomic_claim": _claim_receipt(
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
            "fresh_pre_submit_storage_reauthentication_required": True,
            **_flags(),
        }
    )
    return production._write_immutable_json(target, receipt)


def _validate_sibling_receipt(
    value: Any, *, plan: Mapping[str, Any], task_id: int
) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {
            "schema_version",
            "before_submission",
            "after_submission",
            "exactly_one_sibling_after_submission",
        }
        or value.get("schema_version") != SIBLING_GUARD_SCHEMA
        or value.get("exactly_one_sibling_after_submission") is not True
    ):
        raise HandoffContractError(
            "timeout12h sibling receipt is malformed"
        )
    before = value["before_submission"]
    after = value["after_submission"]
    if (
        not isinstance(before, Mapping)
        or not isinstance(after, Mapping)
        or before.get("snapshot_sha256")
        != canonical_sha256(
            {
                key: item
                for key, item in before.items()
                if key != "snapshot_sha256"
            }
        )
        or after.get("snapshot_sha256")
        != canonical_sha256(
            {
                key: item
                for key, item in after.items()
                if key != "snapshot_sha256"
            }
        )
        or before.get("identity") != after.get("identity")
        or after.get("matching_task_count") != 1
        or after.get("matching_tasks", [{}])[0].get("task_id")
        != task_id
        or after.get("identity", {}).get("task_name")
        != plan["stage"]["task_name"]
    ):
        raise HandoffContractError(
            "timeout12h sibling receipt drifted"
        )
    return copy.deepcopy(dict(value))


def _validate_claim_receipt(
    value: Any, *, plan: Mapping[str, Any], task_id: int
) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {
            "schema_version",
            "acquisition_status",
            "fresh_claim_authorized_scheduler_submit_call",
            "recovered_without_scheduler_submit_call",
            "finalized_claim",
        }
        or value.get("schema_version") != CLAIM_RECEIPT_SCHEMA
    ):
        raise HandoffContractError(
            "timeout12h atomic claim receipt is malformed"
        )
    status = value.get("acquisition_status")
    if (
        status
        not in {
            "fresh_pending",
            "existing_pending",
            "existing_finalized",
        }
        or value.get("fresh_claim_authorized_scheduler_submit_call")
        is not (status == "fresh_pending")
        or value.get("recovered_without_scheduler_submit_call")
        is not (status != "fresh_pending")
    ):
        raise HandoffContractError(
            "timeout12h atomic claim receipt semantics drifted"
        )
    reference = _validate_claim_reference(plan)
    winner = _claim_winner(
        Path(plan["_resolved_plan_path"]), plan
    )
    claim_root = _retry_authority(
        str(plan["retry_of_timeout12h"]["retry_generation"])
    )["claim_root"]
    try:
        finalized = atomic_claim.validate_finalized_claim(
            claim_root,
            reference,
            claim=value.get("finalized_claim"),
            expected_winner=winner,
        )
    except atomic_claim.ClaimContractError as exc:
        raise HandoffContractError(
            "timeout12h finalized claim drifted"
        ) from exc
    if finalized.get("task_id") != task_id:
        raise HandoffContractError(
            "timeout12h finalized claim task drifted"
        )
    return copy.deepcopy(dict(value))


def _load_submission(
    path: Path, *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    if isinstance(plan.get("startup_successor"), Mapping):
        from tools import mft_goal_startup_retry

        return mft_goal_startup_retry.load_submission_for_probe(
            path, plan=plan
        )
    resolved = path.resolve(strict=True)
    receipt = production._validate_seal(
        production._read_json(resolved), SUBMISSION_SCHEMA
    )
    expected_fields = {
        "schema_version",
        "stage",
        "plan",
        "plan_payload_sha256",
        "candidate_physics_sha256",
        "search_authority_reauthentication",
        "task_id",
        "task_name",
        "workdir",
        "dedupe_key",
        "solver_revision",
        "library_revision",
        "profile_sha256",
        "effective_params_sha256",
        "resources",
        "aedt_backend",
        "core_policy",
        "retained_aedt_bundle",
        "retention_run_root",
        "retry_of_timeout12h",
        "timeout12h_sibling_guard",
        "timeout12h_atomic_claim",
        "scheduler_strict_node_contract",
        "scheduler_cutover_receipt",
        "scheduler_cutover_payload_sha256",
        "scheduler_live_launcher_identity",
        "scheduler_admission_snapshot",
        "scheduler_url",
        "scheduler_project",
        "scheduler_project_mutation_performed",
        "scheduler_repository_modified",
        "scheduler_submission_performed",
        "retention_required",
        "prune_protection_required",
        "fresh_pre_submit_storage_reauthentication_required",
        *_flags(),
        "payload_sha256",
    }
    if set(receipt) != expected_fields:
        raise HandoffContractError(
            "timeout12h submission fields drifted"
        )
    plan_record = receipt.get("plan")
    if not isinstance(plan_record, Mapping):
        raise HandoffContractError(
            "timeout12h submission plan record is absent"
        )
    recorded_plan_path = Path(
        str(plan_record.get("path") or "")
    ).resolve(strict=True)
    if production._file_record(recorded_plan_path) != plan_record:
        raise HandoffContractError(
            "timeout12h submission plan bytes drifted"
        )
    mutable_plan = dict(plan)
    mutable_plan["_resolved_plan_path"] = str(recorded_plan_path)
    stage = plan["stage"]
    task_id = _positive_int(
        receipt.get("task_id"), "timeout12h submission task ID"
    )
    _validate_sibling_receipt(
        receipt.get("timeout12h_sibling_guard"),
        plan=plan,
        task_id=task_id,
    )
    _validate_claim_receipt(
        receipt.get("timeout12h_atomic_claim"),
        plan=mutable_plan,
        task_id=task_id,
    )
    strict = receipt.get("scheduler_strict_node_contract")
    if (
        not isinstance(strict, Mapping)
        or strict.get("plan_contract")
        != plan["scheduler_strict_node_contract"]
        or strict.get("direct_strict_node_policy_authenticated") is not True
        or strict.get("same_node_as_task_id") != 0
        or strict.get("scheduler_mutation_performed")
        not in {True, False}
        or (
            strict.get("api_pre_submission_readback") is None
            and strict.get("api_post_submission_response") is None
        )
        or not isinstance(strict.get("api_durable_get_readback"), Mapping)
        or strict["api_durable_get_readback"].get("task_id") != task_id
        or strict["api_durable_get_readback"].get(
            "requested_node_name"
        )
        != STRICT_NODE_NAME
        or strict["api_durable_get_readback"].get(
            "requested_node_name_policy"
        )
        != "strict"
    ):
        raise HandoffContractError(
            "timeout12h strict-node receipt drifted"
        )
    reauth = receipt.get("search_authority_reauthentication")
    if (
        not isinstance(reauth, Mapping)
        or reauth.get("fresh_candidate_reauthenticated") is not True
        or reauth.get("search_authority_sha256")
        != plan["search_authority_sha256"]
    ):
        raise HandoffContractError(
            "timeout12h submission source reauthentication drifted"
        )
    core = receipt.get("core_policy")
    if (
        not isinstance(core, Mapping)
        or core.get("contract") != "mft-standalone-core-optin-v1"
        or core.get("requested_num_cores") != 8
    ):
        raise HandoffContractError(
            "timeout12h core policy drifted"
        )
    if (
        receipt.get("stage") != "standard"
        or receipt.get("plan_payload_sha256") != plan["payload_sha256"]
        or receipt.get("candidate_physics_sha256")
        != plan["candidate_physics_sha256"]
        or receipt.get("task_name") != stage["task_name"]
        or receipt.get("workdir") != stage["workdir"]
        or receipt.get("dedupe_key")
        != stage["retained_aedt_bundle"]["dedupe_key"]
        or receipt.get("solver_revision") != plan["solver_revision"]
        or receipt.get("library_revision") != plan["library_revision"]
        or receipt.get("profile_sha256") != stage["profile_sha256"]
        or receipt.get("effective_params_sha256")
        != stage["effective_params_sha256"]
        or receipt.get("resources") != RESOURCES
        or receipt.get("aedt_backend") != "standalone"
        or receipt.get("retained_aedt_bundle")
        != stage["retained_aedt_bundle"]
        or receipt.get("retention_run_root")
        != stage["retention_run_root"]
        or receipt.get("retry_of_timeout12h")
        != plan["retry_of_timeout12h"]
        or receipt.get("scheduler_url") != stage["scheduler_url"]
        or receipt.get("scheduler_project")
        != scheduler_client.MFT_PROJECT
        or receipt.get("scheduler_project_mutation_performed") is not False
        or receipt.get("scheduler_repository_modified") is not False
        or receipt.get("scheduler_submission_performed") is not True
        or receipt.get("retention_required") is not True
        or receipt.get("prune_protection_required") is not True
        or receipt.get(
            "fresh_pre_submit_storage_reauthentication_required"
        )
        is not True
        or any(
            receipt.get(name) is not expected
            for name, expected in _flags().items()
        )
    ):
        raise HandoffContractError(
            "timeout12h submission identity drifted"
        )
    return receipt


def load_submission_for_probe(
    path: Path, *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    return _load_submission(path, plan=plan)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exact-once bounded MFT timeout12h recovery"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    claim_init = commands.add_parser("init-timeout12h-claim-root")
    claim_init.add_argument(
        "--retry-generation",
        choices=(
            RETRY_GENERATION,
            SUPPLEMENTAL_RETRY_GENERATION,
            LATE_ANCHOR_RETRY_GENERATION,
            LATE_BOUNDARY_RETRY_GENERATION,
        ),
        default=RETRY_GENERATION,
    )
    plan = commands.add_parser("plan-timeout12h-retry")
    plan.add_argument("--immediate-plan", type=Path, required=True)
    plan.add_argument("--immediate-submission", type=Path, required=True)
    plan.add_argument("--strict-node-name", required=True)
    plan.add_argument(
        "--scheduler-url", default=probe.DIAGNOSTIC_SCHEDULER_URL
    )
    plan.add_argument("--storage-observed-at-kst", required=True)
    plan.add_argument("--storage-used-gb", type=float, required=True)
    plan.add_argument(
        "--storage-in-doubt-gb", type=float, required=True
    )
    plan.add_argument("--storage-limit-gb", type=float, required=True)
    plan.add_argument(
        "--storage-observed-free-gb", type=float, required=True
    )
    plan.add_argument("--output", type=Path, required=True)
    submit_parser = commands.add_parser("submit-timeout12h-retry")
    submit_parser.add_argument("--plan", type=Path, required=True)
    submit_parser.add_argument(
        "--scheduler-cutover-receipt", type=Path, required=True
    )
    submit_parser.add_argument("--priority", type=int, default=100)
    submit_parser.add_argument("--output", type=Path, required=True)
    collect_parser = commands.add_parser("collect-timeout12h-retry")
    collect_parser.add_argument("--plan", type=Path, required=True)
    collect_parser.add_argument("--submission", type=Path, required=True)
    collect_parser.add_argument(
        "--scheduler-url", default=probe.DIAGNOSTIC_SCHEDULER_URL
    )
    collect_parser.add_argument("--output", type=Path, required=True)
    validate_parser = commands.add_parser("validate-timeout12h-plan")
    validate_parser.add_argument("--plan", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "init-timeout12h-claim-root":
        authority = initialize_claim_root(
            retry_generation=args.retry_generation
        )
        result = Path(authority["resolved_root"])
    elif args.command == "plan-timeout12h-retry":
        result = create_plan(
            immediate_plan_path=args.immediate_plan,
            immediate_submission_path=args.immediate_submission,
            strict_node_name=args.strict_node_name,
            output=args.output,
            storage_observed_at_kst=args.storage_observed_at_kst,
            storage_used_gb=args.storage_used_gb,
            storage_in_doubt_gb=args.storage_in_doubt_gb,
            storage_limit_gb=args.storage_limit_gb,
            storage_observed_free_gb=args.storage_observed_free_gb,
            scheduler_url=args.scheduler_url,
        )
    elif args.command == "submit-timeout12h-retry":
        result = submit(
            plan_path=args.plan,
            scheduler_cutover_receipt_path=(
                args.scheduler_cutover_receipt
            ),
            priority=args.priority,
            output=args.output,
        )
    elif args.command == "collect-timeout12h-retry":
        result = probe.collect_standard(
            plan_path=args.plan,
            submission_path=args.submission,
            scheduler_url=args.scheduler_url,
            output=args.output,
        )
    else:
        _load_plan(args.plan)
        result = args.plan.resolve(strict=True)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
