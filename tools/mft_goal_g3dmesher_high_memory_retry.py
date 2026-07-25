"""Fail-closed exact-once recovery plan for task 96302.

The only permitted change from the authenticated task-96302 payload is the
Scheduler memory request, from 32768 MiB to 65536 MiB.  The plan is pinned to
the active 4fac Scheduler cutover but deliberately cannot POST while current
n114 work has no sealed upper bound for its remaining GPFS growth.

This module belongs to the MFT repository.  It never edits the separate
Scheduler repository.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
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
from tools import mft_goal_diagnostic_standard_probe as probe
from tools import mft_goal_fea_handoff as production
from tools import mft_goal_timeout12h_retry as timeout12h


PLAN_SCHEMA = "mft-goal-diagnostic-standard-g3dmesher-high-memory-plan-v1"
PROFILE_SCHEMA = (
    "mft-goal-diagnostic-standard-g3dmesher-high-memory-profile-v1"
)
RETRY_EVIDENCE_SCHEMA = (
    "mft-goal-diagnostic-standard-g3dmesher-high-memory-evidence-v1"
)
FAILURE_EVIDENCE_SCHEMA = (
    "mft-goal-diagnostic-standard-g3dmesher-high-memory-task-failure-v1"
)
STREAM_EVIDENCE_SCHEMA = (
    "mft-goal-diagnostic-standard-g3dmesher-high-memory-stream-v1"
)
STORAGE_AUDIT_SCHEMA = (
    "mft-goal-diagnostic-standard-g3dmesher-high-memory-storage-audit-v1"
)
SUBMIT_REFUSAL_SCHEMA = (
    "mft-goal-diagnostic-standard-g3dmesher-high-memory-submit-refusal-v1"
)
CLAIM_REFERENCE_FIELD = "g3dmesher_high_memory_atomic_claim_reference"
RETRY_RECORD_FIELD = "retry_of_g3dmesher_high_memory"

RETRY_GENERATION = "g3dmesher-high-memory-r1"
SOURCE_LOGICAL_TASK_ID = 96218
SOURCE_FAILED_TASK_ID = 96302
SOURCE_PLAN_PAYLOAD_SHA256 = (
    "a74107e1aba54bd947403d5bfcb0481d49a580df7bef1a588fd2659eec2cd54c"
)
SOURCE_SUBMISSION_PAYLOAD_SHA256 = (
    "7ad71272ad95de52ef93e2ed827d9cba25debb39351131853a3859a7ee81e203"
)
SOURCE_CANDIDATE_PHYSICS_SHA256 = (
    "11e0d8daed351ed784d948e63e68a44386bac06340400259e458aabc7bdd8a9d"
)
SOURCE_MEMORY_MB = 32768
TARGET_MEMORY_MB = 65536
RESOURCES = {
    "cpus": 8,
    "memory_mb": TARGET_MEMORY_MB,
    "timeout_seconds": 12 * 3600,
}
STRICT_NODE_NAME = "n114"
STORAGE_ACCOUNT_NAME = "r1jae262"
STORAGE_SAFETY_FLOOR_GB = 10.0
STORAGE_FRESHNESS_LIMIT_SECONDS = 15 * 60
MAX_STREAM_BYTES = 64 * 1024 * 1024
MAX_TASK_INVENTORY_BYTES = 16 * 1024 * 1024

EXPECTED_FAILURE_MESSAGE = (
    "ansys.aedt.core.internal.errors.GrpcApiError: "
    "Failed to execute gRPC AEDT command: Analyze"
)
G3DMESHER_MARKER = "Process 'G3dMesher' terminated abnormally."
MEMORY_HINT_MARKER = (
    "It may have run out of memory or could have been killed by the user."
)
EXECUTION_ERROR_MARKER = (
    "Simulation completed with execution error on server: n114."
)
AMBIGUOUS_FILE_CLOSE_MARKER = "Error closing file: %1"
RESULT_MARKER = "RESULT_JSON:"
LOSS_TRACE_MARKER = 't_loss = sim.analyze_and_extract("loss"'
DISPATCH_PREFIX = "SOLVER_CORE_DISPATCH_JSON "

CLAIM_ROOT = Path(
    "C:/Users/peets/slurm_scheduler_runtime/mft_goal_20260726/"
    "g3dmesher_high_memory_claims_r1"
)
PROFILE_PATH = (
    probe.REPOSITORY_ROOT
    / "regression_260707"
    / "verify"
    / "profiles"
    / "goal_diagnostic_standard_g3dmesher_high_memory_retry.json"
)
CLAIM_AUTHORITY_SHA256 = canonical_sha256(
    {
        "campaign_id": "mft-goal-20260726",
        "goal_contract_schema": GOAL_CONTRACT_SCHEMA,
        "hard_spec_sha256": GOAL_STAGE_SPEC_SHA256,
        "temperature_contract_sha256": GOAL_TEMPERATURE_CONTRACT_SHA256,
        "retry_generation": RETRY_GENERATION,
        "exact_logical_to_failed_task": {
            SOURCE_LOGICAL_TASK_ID: SOURCE_FAILED_TASK_ID
        },
        "source_plan_payload_sha256": SOURCE_PLAN_PAYLOAD_SHA256,
        "source_submission_payload_sha256": (
            SOURCE_SUBMISSION_PAYLOAD_SHA256
        ),
        "source_memory_mb": SOURCE_MEMORY_MB,
        "target_memory_mb": TARGET_MEMORY_MB,
    }
)

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


def _task_id(snapshot: Mapping[str, Any]) -> Any:
    return snapshot.get("task_id", snapshot.get("id"))


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HandoffContractError(f"{label} is invalid")
    return value


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
            "G3dMesher high-memory claim root is unavailable"
        ) from exc


def _claim_authority() -> dict[str, Any]:
    try:
        authority = atomic_claim.load_claim_root(CLAIM_ROOT)
    except atomic_claim.ClaimContractError as exc:
        raise HandoffContractError(
            "G3dMesher high-memory claim root is unavailable"
        ) from exc
    if (
        authority.get("campaign_id") != "mft-goal-20260726"
        or authority.get("campaign_authority_sha256")
        != CLAIM_AUTHORITY_SHA256
    ):
        raise HandoffContractError(
            "G3dMesher high-memory claim authority drifted"
        )
    return authority


def _claim_reference(candidate_physics_sha256: str) -> dict[str, Any]:
    try:
        return atomic_claim.build_claim_reference(
            _claim_authority(),
            candidate_physics_sha256=production._require_sha(
                candidate_physics_sha256,
                "G3dMesher high-memory candidate physics SHA",
            ),
            logical_authority_task_id=SOURCE_LOGICAL_TASK_ID,
            retry_generation=RETRY_GENERATION,
        )
    except atomic_claim.ClaimContractError as exc:
        raise HandoffContractError(
            "G3dMesher high-memory claim reference is invalid"
        ) from exc


def _validate_claim_reference(plan: Mapping[str, Any]) -> dict[str, Any]:
    reference = plan.get(CLAIM_REFERENCE_FIELD)
    if not isinstance(reference, Mapping):
        raise HandoffContractError(
            "G3dMesher high-memory claim reference is absent"
        )
    try:
        normalized = atomic_claim.validate_claim_reference(
            reference, _claim_authority()
        )
    except atomic_claim.ClaimContractError as exc:
        raise HandoffContractError(
            "G3dMesher high-memory claim reference drifted"
        ) from exc
    if (
        normalized.get("candidate_physics_sha256")
        != plan.get("candidate_physics_sha256")
        or normalized.get("logical_authority_task_id")
        != SOURCE_LOGICAL_TASK_ID
        or normalized.get("retry_generation") != RETRY_GENERATION
    ):
        raise HandoffContractError(
            "G3dMesher high-memory claim binding drifted"
        )
    return normalized


def _source_profile_path(
    source_plan_path: Path, source_plan: Mapping[str, Any]
) -> Path:
    profile = source_plan.get("profile")
    if not isinstance(profile, Mapping):
        raise HandoffContractError("source timeout12h profile is absent")
    return production._contained_file(
        source_plan_path.resolve(strict=True).parent,
        profile.get("path"),
        "source timeout12h profile",
    )


def _load_source(
    source_plan_path: Path, source_submission_path: Path
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    source_plan, params, selected, _parent_submission = (
        timeout12h._load_plan(source_plan_path)
    )
    source_submission = timeout12h._load_submission(
        source_submission_path, plan=source_plan
    )
    source_profile = production._read_json(
        _source_profile_path(source_plan_path, source_plan)
    )
    retry = source_plan.get("retry_of_timeout12h")
    strict_contract = source_plan.get("scheduler_strict_node_contract")
    if (
        not isinstance(retry, Mapping)
        or not isinstance(strict_contract, Mapping)
        or source_plan.get("payload_sha256")
        != SOURCE_PLAN_PAYLOAD_SHA256
        or source_submission.get("payload_sha256")
        != SOURCE_SUBMISSION_PAYLOAD_SHA256
        or source_plan.get("candidate_physics_sha256")
        != SOURCE_CANDIDATE_PHYSICS_SHA256
        or retry.get("logical_authority_task_id")
        != SOURCE_LOGICAL_TASK_ID
        or source_submission.get("task_id") != SOURCE_FAILED_TASK_ID
        or source_submission.get("task_name")
        != source_plan["stage"]["task_name"]
        or source_plan["stage"].get("resources")
        != {"cpus": 8, "timeout_seconds": 12 * 3600}
        or source_submission.get("resources")
        != {"cpus": 8, "timeout_seconds": 12 * 3600}
        or source_profile.get("cpus") != 8
        or source_profile.get("mem_mb") != SOURCE_MEMORY_MB
        or source_profile.get("timeout_seconds") != 12 * 3600
        or source_plan["stage"].get("full_model") != 0
        or source_plan["stage"].get("thermal_symmetry") != "eighth"
        or strict_contract.get("requested_node_name") != STRICT_NODE_NAME
        or strict_contract.get("node_name_policy") != "strict"
    ):
        raise HandoffContractError(
            "exact task-96302 timeout12h source authority drifted"
        )
    _validate_fixed_physics(params, source_profile)
    return source_plan, source_submission, params, selected, source_profile


def _validate_fixed_physics(
    params: Mapping[str, Any], profile: Mapping[str, Any]
) -> None:
    overrides = profile.get("param_overrides")
    fixed = profile.get("fixed_boundary_contract")
    expected_params = {
        "thermal_rx_side_block_mesh_level": 5,
        "fan_velocity": 1.5,
        "fan_config": "dual",
        "core_plate_pad_t": 2.0,
        "wcp_pad_t": 2.0,
        "full_model": 0,
        "thermal_symmetry": "eighth",
    }
    expected_overrides = {
        "fan_velocity": 1.5,
        "fan_config": "dual",
        "core_plate_pad_t": 2.0,
        "wcp_pad_t": 2.0,
        "full_model": 0,
        "thermal_symmetry": "eighth",
        "k_ins": 0.2,
    }
    expected_fixed = {
        "fan_velocity_m_s": 1.5,
        "fan_config": "dual",
        "core_plate_pad_t_mm": 2.0,
        "wcp_pad_t_mm": 2.0,
        "thermal_pad_conductivity_W_mK": 0.2,
    }
    if (
        not isinstance(overrides, Mapping)
        or not isinstance(fixed, Mapping)
        or any(params.get(key) != value for key, value in expected_params.items())
        or any(
            overrides.get(key) != value
            for key, value in expected_overrides.items()
        )
        or any(fixed.get(key) != value for key, value in expected_fixed.items())
        or profile.get("cpus") != 8
        or profile.get("timeout_seconds") != 12 * 3600
    ):
        raise HandoffContractError(
            "G3dMesher high-memory fixed physics/mesh/thermal identity drifted"
        )


def _high_memory_profile(source_profile: Mapping[str, Any]) -> dict[str, Any]:
    profile = copy.deepcopy(dict(source_profile))
    profile.update(
        {
            "schema_version": PROFILE_SCHEMA,
            "comment": (
                "Diagnostic-only exact task-96302 G3dMesher operational "
                "resource recovery; memory-only escalation to 65536 MiB"
            ),
            "mem_mb": TARGET_MEMORY_MB,
        }
    )
    _validate_profile_change(source_profile, profile)
    reviewed = production._read_json(PROFILE_PATH.resolve(strict=True))
    if profile != reviewed:
        raise HandoffContractError(
            "reviewed G3dMesher high-memory profile drifted"
        )
    return profile


def _validate_profile_change(
    source_profile: Mapping[str, Any],
    target_profile: Mapping[str, Any],
) -> None:
    changed = {
        key
        for key in set(source_profile) | set(target_profile)
        if source_profile.get(key) != target_profile.get(key)
    }
    if (
        changed != {"schema_version", "comment", "mem_mb"}
        or target_profile.get("schema_version") != PROFILE_SCHEMA
        or target_profile.get("mem_mb") != TARGET_MEMORY_MB
        or source_profile.get("mem_mb") != SOURCE_MEMORY_MB
    ):
        raise HandoffContractError(
            "G3dMesher recovery is not memory-only"
        )


def _task_failure_evidence(
    snapshot: Mapping[str, Any],
    *,
    source_submission: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = {
        "schema_version": FAILURE_EVIDENCE_SCHEMA,
        "task_id": _task_id(snapshot),
        "logical_authority_task_id": SOURCE_LOGICAL_TASK_ID,
        "name": snapshot.get("name"),
        "status": snapshot.get("status"),
        "state": snapshot.get("state"),
        "exit_code": snapshot.get("exit_code"),
        "failure_message": snapshot.get("failure_message"),
        "timeout_seconds": snapshot.get("timeout_seconds"),
        "cpus": snapshot.get("cpus"),
        "memory_mb": snapshot.get("memory_mb"),
        "aedt_backend": snapshot.get("aedt_backend"),
        "project": snapshot.get("project"),
        "dedupe_key": snapshot.get("dedupe_key"),
        "requested_node_name": snapshot.get(
            "requested_node_name", snapshot.get("node_name")
        ),
        "requested_node_name_policy": snapshot.get(
            "requested_node_name_policy",
            snapshot.get("node_name_policy"),
        ),
        "strict_node_placement": snapshot.get("strict_node_placement"),
        "placement_contract_satisfied": snapshot.get(
            "placement_contract_satisfied"
        ),
        "same_node_as_task_id": snapshot.get("same_node_as_task_id", 0),
        "account_name": snapshot.get("account_name"),
        "actual_node_name": snapshot.get("actual_node_name"),
        "allocation_id": snapshot.get(
            "allocation_id", snapshot.get("assigned_allocation")
        ),
        "slurm_job_id": str(snapshot.get("slurm_job_id") or ""),
        "remote_cwd": snapshot.get("remote_cwd"),
        "remote_dir": snapshot.get("remote_dir"),
        "started_at": snapshot.get("started_at"),
        "finished_at": snapshot.get("finished_at"),
    }
    allocation_id = evidence["allocation_id"]
    if (
        evidence["task_id"] != SOURCE_FAILED_TASK_ID
        or evidence["task_id"] != source_submission.get("task_id")
        or evidence["name"] != source_submission.get("task_name")
        or evidence["status"] != "failed"
        or evidence["state"] != "failed"
        or evidence["exit_code"] != 1
        or evidence["failure_message"] != EXPECTED_FAILURE_MESSAGE
        or evidence["timeout_seconds"] != 12 * 3600
        or evidence["cpus"] != 8
        or evidence["memory_mb"] != SOURCE_MEMORY_MB
        or evidence["aedt_backend"] != "standalone"
        or evidence["project"] != scheduler_client.MFT_PROJECT
        or evidence["dedupe_key"] != source_submission.get("dedupe_key")
        or evidence["requested_node_name"] != STRICT_NODE_NAME
        or evidence["requested_node_name_policy"] != "strict"
        or evidence["strict_node_placement"] is not True
        or evidence["placement_contract_satisfied"] is not True
        or evidence["same_node_as_task_id"] != 0
        or evidence["account_name"] != STORAGE_ACCOUNT_NAME
        or evidence["actual_node_name"] != STRICT_NODE_NAME
        or isinstance(allocation_id, bool)
        or not isinstance(allocation_id, int)
        or allocation_id <= 0
        or not evidence["slurm_job_id"].isdigit()
        or not str(evidence["remote_cwd"] or "").strip()
        or not str(evidence["remote_dir"] or "").strip()
        or not str(evidence["started_at"] or "").strip()
        or not str(evidence["finished_at"] or "").strip()
    ):
        raise HandoffContractError(
            "high-memory recovery requires exact failed task-96302 "
            "exit1/32768MiB/43200s strict-n114 evidence"
        )
    return evidence


def _stream_evidence(
    stdout: bytes | str, stderr: bytes | str
) -> dict[str, Any]:
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
        raise HandoffContractError(
            "G3dMesher high-memory stream evidence is invalid"
        )
    try:
        out_text = raw_out.decode("utf-8")
        err_text = raw_err.decode("utf-8")
    except UnicodeError as exc:
        raise HandoffContractError(
            "G3dMesher high-memory stream evidence is not UTF-8"
        ) from exc

    loss_dispatches: list[dict[str, Any]] = []
    for line in out_text.splitlines():
        if not line.startswith(DISPATCH_PREFIX):
            continue
        try:
            dispatch = json.loads(line.removeprefix(DISPATCH_PREFIX))
        except json.JSONDecodeError as exc:
            raise HandoffContractError(
                "G3dMesher solver dispatch JSON is invalid"
            ) from exc
        if isinstance(dispatch, dict) and dispatch.get("stage") == "loss":
            loss_dispatches.append(dispatch)

    if (
        out_text.count(G3DMESHER_MARKER) != 0
        or err_text.count(G3DMESHER_MARKER) != 1
        or out_text.count(MEMORY_HINT_MARKER) != 0
        or err_text.count(MEMORY_HINT_MARKER) != 1
        or out_text.count(EXECUTION_ERROR_MARKER) != 0
        or err_text.count(EXECUTION_ERROR_MARKER) != 1
        or RESULT_MARKER in out_text
        or RESULT_MARKER in err_text
        or AMBIGUOUS_FILE_CLOSE_MARKER in out_text
        or AMBIGUOUS_FILE_CLOSE_MARKER in err_text
        or LOSS_TRACE_MARKER not in err_text
        or len(loss_dispatches) != 1
        or loss_dispatches[0].get("backend") != "standalone"
        or loss_dispatches[0].get("cores_argument") != 8
        or loss_dispatches[0].get("gpus_argument") != 0
        or loss_dispatches[0].get("tasks_argument") != 1
        or loss_dispatches[0].get("dispatch")
        != "native_analyze_validated_dso"
    ):
        raise HandoffContractError(
            "stderr/stdout are not the exact G3dMesher OOM-like loss "
            "failure with RESULT_JSON absent"
        )
    return {
        "schema_version": STREAM_EVIDENCE_SCHEMA,
        "task_id": SOURCE_FAILED_TASK_ID,
        "stdout_sha256": production._sha256_bytes(raw_out),
        "stdout_size_bytes": len(raw_out),
        "stderr_sha256": production._sha256_bytes(raw_err),
        "stderr_size_bytes": len(raw_err),
        "failure_stage": "loss",
        "g3dmesher_abnormal_marker_count": 1,
        "memory_hint_marker_count": 1,
        "execution_error_marker_count": 1,
        "ambiguous_file_close_marker_absent": True,
        "result_json_absent": True,
        "loss_dispatch": copy.deepcopy(loss_dispatches[0]),
        "classification": "operational_resource_failure_only",
    }


def _scheduler_stream(
    *, scheduler_url: str, task_id: int, stream: str
) -> bytes:
    if stream not in {"stdout", "stderr"}:
        raise HandoffContractError("Scheduler stream name is invalid")
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
            f"G3dMesher Scheduler {stream} fetch failed"
        ) from exc
    if not raw or len(raw) > MAX_STREAM_BYTES:
        raise HandoffContractError(
            f"G3dMesher Scheduler {stream} byte bound failed"
        )
    return raw


def _scheduler_tasks(*, scheduler_url: str) -> list[dict[str, Any]]:
    query = production.urllib.parse.urlencode(
        {
            "project": scheduler_client.MFT_PROJECT,
            "status": "running",
            "limit": 10000,
        }
    )
    request = production.urllib.request.Request(
        f"{scheduler_url.rstrip('/')}/api/tasks?{query}",
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with production.urllib.request.urlopen(
            request, timeout=30.0
        ) as response:
            raw = response.read(MAX_TASK_INVENTORY_BYTES + 1)
    except (OSError, production.urllib.error.URLError) as exc:
        raise HandoffContractError(
            "Scheduler task inventory fetch failed"
        ) from exc
    if len(raw) > MAX_TASK_INVENTORY_BYTES:
        raise HandoffContractError(
            "Scheduler task inventory exceeds byte bound"
        )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise HandoffContractError(
            "Scheduler task inventory is invalid JSON"
        ) from exc
    if not isinstance(value, list) or len(value) >= 10000:
        raise HandoffContractError(
            "Scheduler task inventory is malformed or truncated"
        )
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _parse_kst_timestamp(value: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("+09:00"):
        raise HandoffContractError(
            "storage observation must be an explicit KST timestamp"
        )
    try:
        observed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise HandoffContractError(
            "storage observation timestamp is invalid"
        ) from exc
    if observed.utcoffset() != timedelta(hours=9):
        raise HandoffContractError(
            "storage observation timestamp is not KST"
        )
    return observed


def _storage_audit(
    *,
    observed_at_kst: str,
    used_gb: float,
    in_doubt_gb: float,
    limit_gb: float,
    observed_free_gb: float,
    prospective_grid_bytes: int,
    task_rows: Sequence[Mapping[str, Any]],
    running_remote_dir_bytes: Mapping[int, int],
    now: datetime | None = None,
) -> dict[str, Any]:
    observed = _parse_kst_timestamp(observed_at_kst)
    current = datetime.now(timezone.utc) if now is None else now
    if current.tzinfo is None:
        raise HandoffContractError("storage audit clock is timezone-naive")
    age = (current.astimezone(timezone.utc) - observed.astimezone(timezone.utc))
    age_seconds = age.total_seconds()
    numeric = (used_gb, in_doubt_gb, limit_gb, observed_free_gb)
    prospective_grid_bytes = _positive_int(
        prospective_grid_bytes, "prospective grid bytes"
    )
    prospective_grid_gb = prospective_grid_bytes / (1024**3)
    if (
        age_seconds < -30
        or age_seconds > STORAGE_FRESHNESS_LIMIT_SECONDS
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value < 0
            for value in numeric
        )
        or abs((limit_gb - used_gb - in_doubt_gb) - observed_free_gb)
        > 0.02
        or observed_free_gb
        < prospective_grid_gb + STORAGE_SAFETY_FLOOR_GB
    ):
        raise HandoffContractError(
            "fresh single-task GPFS storage arithmetic is unsafe"
        )

    active: list[dict[str, Any]] = []
    for raw in task_rows:
        if (
            raw.get("project") != scheduler_client.MFT_PROJECT
            or raw.get("status") != "running"
            or raw.get("account_name") != STORAGE_ACCOUNT_NAME
            or raw.get("actual_node_name") != STRICT_NODE_NAME
        ):
            continue
        task_id = _positive_int(_task_id(raw), "running storage task ID")
        remote_dir = str(raw.get("remote_dir") or "").strip()
        byte_count = running_remote_dir_bytes.get(task_id)
        if (
            not remote_dir
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count <= 0
        ):
            raise HandoffContractError(
                "running task remote-dir byte evidence is absent"
            )
        active.append(
            {
                "task_id": task_id,
                "name": str(raw.get("name") or ""),
                "status": "running",
                "account_name": STORAGE_ACCOUNT_NAME,
                "actual_node_name": STRICT_NODE_NAME,
                "remote_dir": remote_dir,
                "current_remote_dir_bytes": byte_count,
                "current_remote_dir_bytes_source": "read_only_ssh_du_sb",
                "remaining_growth_bound_bytes": None,
                "remaining_growth_bounded": False,
            }
        )
    active.sort(key=lambda row: row["task_id"])
    if (
        not active
        or set(running_remote_dir_bytes)
        != {row["task_id"] for row in active}
    ):
        raise HandoffContractError(
            "current n114 unbounded-running set or byte evidence drifted"
        )
    return {
        "schema_version": STORAGE_AUDIT_SCHEMA,
        "observed_at_kst": observed_at_kst,
        "freshness_seconds_at_plan": age_seconds,
        "source": "read_only_gpfs_quota_probe_plus_ssh_du_sb",
        "account_name": STORAGE_ACCOUNT_NAME,
        "used_gb": float(used_gb),
        "in_doubt_gb": float(in_doubt_gb),
        "limit_gb": float(limit_gb),
        "observed_free_gb": float(observed_free_gb),
        "prospective_grid_bytes": prospective_grid_bytes,
        "prospective_grid_gb": prospective_grid_gb,
        "safety_floor_gb": STORAGE_SAFETY_FLOOR_GB,
        "single_task_arithmetic_passed": True,
        "running_unbounded_commitments": active,
        "running_unbounded_commitment_count": len(active),
        "unbounded_running_growth_present": True,
        "aggregate_submission_allowed": False,
        "safe_extra_submit_count": 0,
        "fresh_pre_submit_storage_reauthentication_required": True,
        "parallel_submit_without_reaudit_allowed": False,
    }


def _active_scheduler_contract() -> dict[str, Any]:
    pin = probe._active_strict_node_scheduler_pin()
    if (
        pin.get("pin_generation") != "scheduler-strict-node-4fac-v4"
        or pin.get("revision") != probe.SCHEDULER_STRICT_NODE_REVISION
    ):
        raise HandoffContractError("active Scheduler 4fac pin drifted")
    return {
        "schema_version": probe.STRICT_NODE_PLACEMENT_SCHEMA,
        "pin_generation": pin["pin_generation"],
        "requested_node_name": STRICT_NODE_NAME,
        "node_name_policy": "strict",
        "scheduler_revision": pin["revision"],
        "scheduler_tree": pin["tree"],
        "scheduler_launcher_sha256": pin["launcher_sha256"],
        "scheduler_cutover_receipt_schema": pin[
            "cutover_receipt_schema"
        ],
        "scheduler_cutover_receipt_sha256": pin[
            "cutover_receipt_sha256"
        ],
        "task_identity_generation": RETRY_GENERATION,
        "fallback_allocation_allowed": False,
        "api_submission_readback_required": True,
        "durable_get_readback_required": True,
        "terminal_readback_required": True,
    }


def _task_identity(candidate_physics_sha256: str) -> tuple[str, str]:
    stem = production._require_sha(
        candidate_physics_sha256,
        "G3dMesher high-memory candidate physics SHA",
    )[:12]
    return (
        "mft-goal-diag-standard-g3d-highmem-"
        f"r1-l{SOURCE_LOGICAL_TASK_ID}-{stem}",
        "mft_goal_diag_standard_g3d_highmem_"
        f"r1_l{SOURCE_LOGICAL_TASK_ID}_{stem}",
    )


def create_plan(
    *,
    source_plan_path: Path,
    source_submission_path: Path,
    output: Path,
    storage_observed_at_kst: str,
    storage_used_gb: float,
    storage_in_doubt_gb: float,
    storage_limit_gb: float,
    storage_observed_free_gb: float,
    running_remote_dir_bytes: Mapping[int, int],
    scheduler_url: str = probe.DIAGNOSTIC_SCHEDULER_URL,
    task_reader: Any = None,
    stdout_reader: Any = None,
    stderr_reader: Any = None,
    task_list_reader: Any = None,
    now: datetime | None = None,
) -> Path:
    (
        source_plan,
        source_submission,
        params,
        selected,
        source_profile,
    ) = _load_source(source_plan_path, source_submission_path)
    if (
        scheduler_url.rstrip("/") != probe.DIAGNOSTIC_SCHEDULER_URL
        or scheduler_url.rstrip("/") != source_submission["scheduler_url"]
    ):
        raise HandoffContractError("Scheduler endpoint drifted")

    read_task = task_reader or probe._scheduler_task_snapshot
    read_stdout = stdout_reader or (
        lambda **kwargs: _scheduler_stream(stream="stdout", **kwargs)
    )
    read_stderr = stderr_reader or (
        lambda **kwargs: _scheduler_stream(stream="stderr", **kwargs)
    )
    read_tasks = task_list_reader or _scheduler_tasks
    execution = _task_failure_evidence(
        read_task(
            scheduler_url=scheduler_url.rstrip("/"),
            task_id=SOURCE_FAILED_TASK_ID,
        ),
        source_submission=source_submission,
    )
    stream = _stream_evidence(
        read_stdout(
            scheduler_url=scheduler_url.rstrip("/"),
            task_id=SOURCE_FAILED_TASK_ID,
        ),
        read_stderr(
            scheduler_url=scheduler_url.rstrip("/"),
            task_id=SOURCE_FAILED_TASK_ID,
        ),
    )
    source_grid_bytes = _positive_int(
        source_plan["retry_of_timeout12h"]["stream_evidence"].get(
            "fresh_grid_output_bytes"
        ),
        "source timeout12h prospective grid bytes",
    )
    storage = _storage_audit(
        observed_at_kst=storage_observed_at_kst,
        used_gb=storage_used_gb,
        in_doubt_gb=storage_in_doubt_gb,
        limit_gb=storage_limit_gb,
        observed_free_gb=storage_observed_free_gb,
        prospective_grid_bytes=source_grid_bytes,
        task_rows=read_tasks(scheduler_url=scheduler_url.rstrip("/")),
        running_remote_dir_bytes=running_remote_dir_bytes,
        now=now,
    )

    profile = _high_memory_profile(source_profile)
    effective_source = production._effective_params(params, source_profile)
    effective_target = production._effective_params(params, profile)
    if effective_source != effective_target:
        raise HandoffContractError(
            "G3dMesher high-memory effective physics changed"
        )
    task_name, workdir = _task_identity(
        source_plan["candidate_physics_sha256"]
    )
    retained = scheduler_client.retained_aedt_identity(
        task_name,
        params,
        profile,
        source_plan["solver_revision"],
        source_plan["library_revision"],
    )
    source_retained = source_plan["stage"]["retained_aedt_bundle"]
    if (
        retained is None
        or retained["dedupe_key"] == source_retained["dedupe_key"]
        or retained["relative_directory"]
        == source_retained["relative_directory"]
        or retained["profile_sha256"] == source_retained["profile_sha256"]
    ):
        raise HandoffContractError(
            "G3dMesher high-memory dedupe/retention identity is not distinct"
        )

    retry_record = {
        "schema_version": RETRY_EVIDENCE_SCHEMA,
        "retry_generation": RETRY_GENERATION,
        "logical_authority_task_id": SOURCE_LOGICAL_TASK_ID,
        "retry_of_task_id": SOURCE_FAILED_TASK_ID,
        "source_plan": production._file_record(
            source_plan_path.resolve(strict=True)
        ),
        "source_plan_payload_sha256": source_plan["payload_sha256"],
        "source_submission": production._file_record(
            source_submission_path.resolve(strict=True)
        ),
        "source_submission_payload_sha256": source_submission[
            "payload_sha256"
        ],
        "source_timeout12h_ancestry_sha256": canonical_sha256(
            source_plan["retry_of_timeout12h"]
        ),
        "task_failure_evidence": execution,
        "task_failure_evidence_sha256": canonical_sha256(execution),
        "stream_evidence": stream,
        "stream_evidence_sha256": canonical_sha256(stream),
        "storage_audit": storage,
        "storage_audit_sha256": canonical_sha256(storage),
        "source_memory_mb": SOURCE_MEMORY_MB,
        "target_memory_mb": TARGET_MEMORY_MB,
        "memory_change_only": True,
        "cpu_cores_unchanged": 8,
        "timeout_seconds_unchanged": 12 * 3600,
        "geometry_physics_unchanged": True,
        "mesh_level_unchanged": 5,
        "fan_velocity_m_s_unchanged": 1.5,
        "tim_and_pad_contract_unchanged": True,
        "automatic_submission_allowed": False,
        "scheduler_post_calls": 0,
    }
    scheduler_contract = _active_scheduler_contract()
    reference = _claim_reference(source_plan["candidate_physics_sha256"])

    destination = output.resolve()
    if destination.exists():
        raise HandoffContractError(
            f"G3dMesher high-memory plan output exists: {destination}"
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
            staging / "diagnostic_standard_g3dmesher_high_memory_profile.json",
            profile,
        )
        unsigned = copy.deepcopy(source_plan)
        unsigned.pop("payload_sha256", None)
        unsigned.pop("retry_of_timeout12h", None)
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
                    "source": production._file_record(
                        PROFILE_PATH.resolve(strict=True)
                    ),
                    "parent_source": production._file_record(
                        _source_profile_path(source_plan_path, source_plan)
                    ),
                },
                "stage": {
                    **copy.deepcopy(source_plan["stage"]),
                    "task_name": task_name,
                    "workdir": workdir,
                    "profile_sha256": canonical_sha256(profile),
                    "effective_params_sha256": canonical_sha256(
                        effective_target
                    ),
                    "resources": copy.deepcopy(RESOURCES),
                    "retained_aedt_bundle": retained,
                    "retention_run_root": (
                        probe._retention_run_root_evidence(retained)
                    ),
                },
                "available_submission_commands": [
                    "submit-g3dmesher-high-memory-retry"
                ],
                RETRY_RECORD_FIELD: retry_record,
                CLAIM_REFERENCE_FIELD: reference,
                "scheduler_strict_node_contract": scheduler_contract,
                "submission_gate": {
                    "submit_allowed": False,
                    "scheduler_post_calls": 0,
                    "reason": "current_unbounded_running_storage_growth",
                    "safe_extra_submit_count": 0,
                    "requires_new_authorized_generation_after_reaudit": True,
                },
                "scheduler_submission_performed": False,
            }
        )
        plan_path = production._write_immutable_json(
            staging / "diagnostic_g3dmesher_high_memory_retry_plan.json",
            production._seal(unsigned),
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / plan_path.name


def _plan_artifact(root: Path, record: Any, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise HandoffContractError(f"{label} record is absent")
    target = production._contained_file(root, record.get("path"), label)
    if production._sha256_file(target) != production._require_sha(
        record.get("sha256"), f"{label} SHA"
    ):
        raise HandoffContractError(f"{label} bytes drifted")
    return target


def _validate_storage_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HandoffContractError("high-memory storage audit is absent")
    active = value.get("running_unbounded_commitments")
    if (
        value.get("schema_version") != STORAGE_AUDIT_SCHEMA
        or value.get("account_name") != STORAGE_ACCOUNT_NAME
        or value.get("safety_floor_gb") != STORAGE_SAFETY_FLOOR_GB
        or value.get("single_task_arithmetic_passed") is not True
        or value.get("unbounded_running_growth_present") is not True
        or value.get("aggregate_submission_allowed") is not False
        or value.get("safe_extra_submit_count") != 0
        or value.get("parallel_submit_without_reaudit_allowed") is not False
        or not isinstance(active, list)
        or not active
        or value.get("running_unbounded_commitment_count") != len(active)
        or any(
            not isinstance(row, Mapping)
            or row.get("status") != "running"
            or row.get("account_name") != STORAGE_ACCOUNT_NAME
            or row.get("actual_node_name") != STRICT_NODE_NAME
            or row.get("remaining_growth_bounded") is not False
            or row.get("remaining_growth_bound_bytes") is not None
            for row in active
        )
    ):
        raise HandoffContractError(
            "high-memory fail-closed storage audit drifted"
        )
    return copy.deepcopy(dict(value))


def load_plan(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    plan = production._validate_seal(
        production._read_json(resolved), PLAN_SCHEMA
    )
    retry = plan.get(RETRY_RECORD_FIELD)
    gate = plan.get("submission_gate")
    if (
        plan.get("campaign_id") != "mft-goal-20260726"
        or plan.get("goal_contract_schema") != GOAL_CONTRACT_SCHEMA
        or plan.get("hard_spec") != GOAL_STAGE_SPEC
        or plan.get("hard_spec_sha256") != GOAL_STAGE_SPEC_SHA256
        or plan.get("temperature_contract_sha256")
        != GOAL_TEMPERATURE_CONTRACT_SHA256
        or plan.get("candidate_physics_sha256")
        != SOURCE_CANDIDATE_PHYSICS_SHA256
        or plan.get("available_submission_commands")
        != ["submit-g3dmesher-high-memory-retry"]
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
        or not isinstance(retry, Mapping)
        or not isinstance(gate, Mapping)
        or gate
        != {
            "submit_allowed": False,
            "scheduler_post_calls": 0,
            "reason": "current_unbounded_running_storage_growth",
            "safe_extra_submit_count": 0,
            "requires_new_authorized_generation_after_reaudit": True,
        }
    ):
        raise HandoffContractError(
            "G3dMesher high-memory plan contract drifted"
        )

    root = resolved.parent
    params = production._read_json(
        _plan_artifact(root, plan.get("fea_params"), "high-memory params")
    )
    selected = production._read_json(
        _plan_artifact(
            root, plan.get("selected_candidate"), "high-memory candidate"
        )
    )
    profile_record = plan.get("profile")
    profile = production._read_json(
        _plan_artifact(root, profile_record, "high-memory profile")
    )
    if (
        not isinstance(profile_record, Mapping)
        or profile_record.get("canonical_sha256")
        != canonical_sha256(profile)
        or profile_record.get("source")
        != production._file_record(PROFILE_PATH.resolve(strict=True))
    ):
        raise HandoffContractError(
            "reviewed high-memory profile provenance drifted"
        )
    source_plan_record = retry.get("source_plan")
    source_submission_record = retry.get("source_submission")
    if (
        not isinstance(source_plan_record, Mapping)
        or not isinstance(source_submission_record, Mapping)
    ):
        raise HandoffContractError(
            "G3dMesher high-memory source records are absent"
        )
    source_plan_path = Path(
        str(source_plan_record.get("path") or "")
    ).resolve(strict=True)
    source_submission_path = Path(
        str(source_submission_record.get("path") or "")
    ).resolve(strict=True)
    if (
        production._file_record(source_plan_path) != source_plan_record
        or production._file_record(source_submission_path)
        != source_submission_record
    ):
        raise HandoffContractError(
            "G3dMesher high-memory source bytes drifted"
        )
    (
        source_plan,
        source_submission,
        source_params,
        source_selected,
        source_profile,
    ) = _load_source(source_plan_path, source_submission_path)
    _validate_profile_change(source_profile, profile)
    _validate_fixed_physics(params, profile)
    if profile_record.get("parent_source") != production._file_record(
        _source_profile_path(source_plan_path, source_plan)
    ):
        raise HandoffContractError(
            "source timeout12h profile provenance drifted"
        )
    if params != source_params or selected != source_selected:
        raise HandoffContractError(
            "G3dMesher high-memory candidate/params changed"
        )

    execution = retry.get("task_failure_evidence")
    stream = retry.get("stream_evidence")
    storage = retry.get("storage_audit")
    if (
        set(retry)
        != {
            "schema_version",
            "retry_generation",
            "logical_authority_task_id",
            "retry_of_task_id",
            "source_plan",
            "source_plan_payload_sha256",
            "source_submission",
            "source_submission_payload_sha256",
            "source_timeout12h_ancestry_sha256",
            "task_failure_evidence",
            "task_failure_evidence_sha256",
            "stream_evidence",
            "stream_evidence_sha256",
            "storage_audit",
            "storage_audit_sha256",
            "source_memory_mb",
            "target_memory_mb",
            "memory_change_only",
            "cpu_cores_unchanged",
            "timeout_seconds_unchanged",
            "geometry_physics_unchanged",
            "mesh_level_unchanged",
            "fan_velocity_m_s_unchanged",
            "tim_and_pad_contract_unchanged",
            "automatic_submission_allowed",
            "scheduler_post_calls",
        }
        or retry.get("schema_version") != RETRY_EVIDENCE_SCHEMA
        or retry.get("retry_generation") != RETRY_GENERATION
        or retry.get("logical_authority_task_id")
        != SOURCE_LOGICAL_TASK_ID
        or retry.get("retry_of_task_id") != SOURCE_FAILED_TASK_ID
        or retry.get("source_plan_payload_sha256")
        != source_plan["payload_sha256"]
        or retry.get("source_submission_payload_sha256")
        != source_submission["payload_sha256"]
        or retry.get("source_timeout12h_ancestry_sha256")
        != canonical_sha256(source_plan["retry_of_timeout12h"])
        or not isinstance(execution, Mapping)
        or retry.get("task_failure_evidence_sha256")
        != canonical_sha256(execution)
        or execution.get("schema_version") != FAILURE_EVIDENCE_SCHEMA
        or execution.get("task_id") != SOURCE_FAILED_TASK_ID
        or execution.get("exit_code") != 1
        or execution.get("memory_mb") != SOURCE_MEMORY_MB
        or not isinstance(stream, Mapping)
        or retry.get("stream_evidence_sha256")
        != canonical_sha256(stream)
        or stream.get("schema_version") != STREAM_EVIDENCE_SCHEMA
        or stream.get("classification")
        != "operational_resource_failure_only"
        or stream.get("result_json_absent") is not True
        or not isinstance(storage, Mapping)
        or retry.get("storage_audit_sha256")
        != canonical_sha256(storage)
        or retry.get("source_memory_mb") != SOURCE_MEMORY_MB
        or retry.get("target_memory_mb") != TARGET_MEMORY_MB
        or retry.get("memory_change_only") is not True
        or retry.get("cpu_cores_unchanged") != 8
        or retry.get("timeout_seconds_unchanged") != 12 * 3600
        or retry.get("geometry_physics_unchanged") is not True
        or retry.get("mesh_level_unchanged") != 5
        or retry.get("fan_velocity_m_s_unchanged") != 1.5
        or retry.get("tim_and_pad_contract_unchanged") is not True
        or retry.get("automatic_submission_allowed") is not False
        or retry.get("scheduler_post_calls") != 0
    ):
        raise HandoffContractError(
            "G3dMesher high-memory ancestry drifted"
        )
    _validate_storage_record(storage)

    scheduler_contract = plan.get("scheduler_strict_node_contract")
    if scheduler_contract != _active_scheduler_contract():
        raise HandoffContractError(
            "G3dMesher high-memory active 4fac pin drifted"
        )
    stage = plan.get("stage")
    task_name, workdir = _task_identity(plan["candidate_physics_sha256"])
    retained = scheduler_client.retained_aedt_identity(
        task_name,
        params,
        profile,
        plan["solver_revision"],
        plan["library_revision"],
    )
    if (
        not isinstance(stage, Mapping)
        or stage.get("task_name") != task_name
        or stage.get("workdir") != workdir
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
        or retained is None
        or retained["dedupe_key"]
        == source_plan["stage"]["retained_aedt_bundle"]["dedupe_key"]
        or retained["relative_directory"]
        == source_plan["stage"]["retained_aedt_bundle"][
            "relative_directory"
        ]
    ):
        raise HandoffContractError(
            "G3dMesher high-memory execution identity drifted"
        )
    _validate_claim_reference(plan)
    return plan


def write_submit_refusal(*, plan_path: Path, output: Path) -> Path:
    """Write immutable POST0 evidence; this function has no POST capability."""

    plan = load_plan(plan_path)
    storage = plan[RETRY_RECORD_FIELD]["storage_audit"]
    refusal = production._seal(
        {
            "schema_version": SUBMIT_REFUSAL_SCHEMA,
            "plan": production._file_record(plan_path.resolve(strict=True)),
            "plan_payload_sha256": plan["payload_sha256"],
            "retry_generation": RETRY_GENERATION,
            "submit_allowed": False,
            "reason": "current_unbounded_running_storage_growth",
            "safe_extra_submit_count": 0,
            "running_unbounded_commitment_count": storage[
                "running_unbounded_commitment_count"
            ],
            "running_unbounded_task_ids": [
                row["task_id"]
                for row in storage["running_unbounded_commitments"]
            ],
            "scheduler_post_calls": 0,
            "scheduler_repository_modified": False,
            "scheduler_project_mutation_performed": False,
            "requires_new_authorized_generation_after_reaudit": True,
        }
    )
    return production._write_immutable_json(output.resolve(), refusal)


def _parse_running_bytes(values: Sequence[str]) -> dict[int, int]:
    result: dict[int, int] = {}
    for raw in values:
        left, separator, right = raw.partition("=")
        try:
            task_id = int(left)
            byte_count = int(right)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "running byte evidence must be TASK_ID=BYTES"
            ) from exc
        if (
            separator != "="
            or task_id <= 0
            or byte_count <= 0
            or task_id in result
        ):
            raise argparse.ArgumentTypeError(
                "running byte evidence must be unique positive TASK_ID=BYTES"
            )
        result[task_id] = byte_count
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exact task-96302 G3dMesher high-memory recovery"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init-claim-root")
    plan = commands.add_parser("plan-g3dmesher-high-memory-retry")
    plan.add_argument("--source-plan", type=Path, required=True)
    plan.add_argument("--source-submission", type=Path, required=True)
    plan.add_argument(
        "--scheduler-url", default=probe.DIAGNOSTIC_SCHEDULER_URL
    )
    plan.add_argument("--storage-observed-at-kst", required=True)
    plan.add_argument("--storage-used-gb", type=float, required=True)
    plan.add_argument("--storage-in-doubt-gb", type=float, required=True)
    plan.add_argument("--storage-limit-gb", type=float, required=True)
    plan.add_argument("--storage-observed-free-gb", type=float, required=True)
    plan.add_argument(
        "--running-remote-dir-bytes",
        action="append",
        default=[],
        metavar="TASK_ID=BYTES",
    )
    plan.add_argument("--output", type=Path, required=True)
    refusal = commands.add_parser("submit-g3dmesher-high-memory-retry")
    refusal.add_argument("--plan", type=Path, required=True)
    refusal.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "init-claim-root":
        authority = initialize_claim_root()
        result = Path(authority["resolved_root"])
        status = "ok"
    elif args.command == "plan-g3dmesher-high-memory-retry":
        result = create_plan(
            source_plan_path=args.source_plan,
            source_submission_path=args.source_submission,
            output=args.output,
            storage_observed_at_kst=args.storage_observed_at_kst,
            storage_used_gb=args.storage_used_gb,
            storage_in_doubt_gb=args.storage_in_doubt_gb,
            storage_limit_gb=args.storage_limit_gb,
            storage_observed_free_gb=args.storage_observed_free_gb,
            running_remote_dir_bytes=_parse_running_bytes(
                args.running_remote_dir_bytes
            ),
            scheduler_url=args.scheduler_url,
        )
        status = "ok"
    else:
        result = write_submit_refusal(plan_path=args.plan, output=args.output)
        status = "refused_post0"
    print(json.dumps({"status": status, "path": str(result)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
