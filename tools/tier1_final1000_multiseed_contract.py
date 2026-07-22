"""Fail-closed wire contracts for final1000 finite multi-seed lanes.

This module is deliberately side-effect free.  It converts already validated
single-seed final1000 tasks into a finite batch parent while retaining every
child payload and logical dedupe identity byte-for-byte.  Runtime journaling,
controller submission, and harvesting live in separate modules so importing
the contract can never mutate Scheduler or remote state.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import math
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

try:
    from tier1_corrected_current7_receipt import canonical_sha256
    from tier1_corrected_current7_slurm_bundle import TASK_SCHEMA
    from tier1_final1000_slurm_launch import (
        REQUIRED_SCHEDULER_FIELDS,
        validate_task,
    )
    from tier1_final1000_stage_profiles import BY_ID, stage_profile
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_corrected_current7_receipt import canonical_sha256
    from tools.tier1_corrected_current7_slurm_bundle import TASK_SCHEMA
    from tools.tier1_final1000_slurm_launch import (
        REQUIRED_SCHEDULER_FIELDS,
        validate_task,
    )
    from tools.tier1_final1000_stage_profiles import BY_ID, stage_profile


PROTOCOL_VERSION = "final1000-finite-multiseed-phase-a-v2"
BATCH_PAYLOAD_SCHEMA = "mft-tier1-final1000-multiseed-lane-task-v2"
BATCH_MANIFEST_SCHEMA = "mft-tier1-final1000-multiseed-batch-manifest-v2"
TASK_STATUS_SCHEMA = "mft-tier1-final1000-multiseed-task-status-v2"
CHILD_RECEIPT_SCHEMA = "mft-tier1-final1000-multiseed-child-receipt-v2"
COMPACT_STATUS_SCHEMA = "mft-tier1-final1000-compact-condition-status-v2"
SHARD_MANIFEST_SCHEMA = "mft-tier1-final1000-seed-result-shards-v2"
COMPACT_INDEX_SCHEMA = "mft-tier1-final1000-compact-condition-index-v2"

MIN_BATCH_LENGTH = 1
MAX_BATCH_LENGTH = 8
PHASE_A_OPERATIONAL_BATCH_LENGTHS = frozenset({1, 4})
MAX_PHYSICAL_LANES = 500
TASK_NAME_PREFIX = "mft-t1fg-"
DEDUPE_PREFIX = "mft-tier1-final1000:lane:"
PHASE_A_REMOTE_CODE_FILES = (
    "tools/tier1_corrected_current7_slurm_seed_runner.py",
    "tools/tier1_final1000_multiseed_contract.py",
    "tools/tier1_final1000_multiseed_lane_runner.py",
)

# The sealed single-seed final1000 scientific profile was authored with an
# eight-thread inference field.  Phase A schedules four CPUs, so the finite
# lane derives a new, fully sealed child envelope whose execution-only thread
# cap is exactly the Scheduler CPU request.  Reversing this transform must
# reproduce the original validated single-seed task byte-for-byte.
LEGACY_SINGLE_SEED_INFERENCE_THREADS = 8

TERMINAL_CHILD_STATES = frozenset(
    {"completed", "failed", "stopped", "incomplete", "refused"}
)
TERMINAL_PARENT_STATES = frozenset(
    {"completed", "completed_with_failures", "failed", "stopped", "deadline"}
)

RESOURCE_FIELDS = (
    "remote_cwd",
    "required_capability",
    "env_profile",
    "cpus",
    "memory_mb",
    "scheduling_profile",
    "aedt_backend",
    "gpus",
    "priority",
    "timeout_seconds",
    "max_workers_per_node",
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0)
    ):
        raise RuntimeError(
            f"{label} must be finite" + (" and positive" if positive else "")
        )
    return float(value)


def _child_summary(child: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ordinal": int(child["ordinal"]),
        "seed": int(child["seed"]),
        "payload_sha256": str(child["payload_sha256"]),
        "logical_dedupe_key": str(child["logical_dedupe_key"]),
        "child_task_sha256": str(child["child_task_sha256"]),
    }


def _expected_child_dedupe(
    payload: Mapping[str, Any], resource_policy: Mapping[str, Any]
) -> str:
    resources = {
        key: copy.deepcopy(resource_policy[key])
        for key in (
            "cpus",
            "memory_mb",
            "scheduling_profile",
            "aedt_backend",
            "gpus",
            "priority",
            "timeout_seconds",
            "max_workers_per_node",
        )
    }
    identity = {
        "goal": "final1000",
        "stage_profile_sha256": payload.get("final_goal_stage_profile_sha256"),
        "bundle_id": payload.get("bundle_id"),
        "payload": copy.deepcopy(dict(payload)),
        "resources": resources,
    }
    return f"mft-tier1-final1000:{canonical_sha256(identity)}"


def _single_seed_command(payload_sha256: str) -> str:
    return "\n".join(
        [
            "set -euo pipefail",
            'export PYTHONPATH="$PWD/artifacts/python-site:$PWD/artifacts/code'
            '${PYTHONPATH:+:$PYTHONPATH}"',
            'payload_path="${SLURM_SCHEDULER_PAYLOAD_PATH:?scheduler payload path is missing}"',
            'case "$payload_path" in /*) ;; *) payload_path="$HOME/$payload_path" ;; esac',
            'payload_path=$(realpath -e -- "$payload_path")',
            'payload_root=$(realpath -e -- "$HOME/slurm_scheduler/runs")',
            'case "$payload_path" in "$payload_root"/*/payload.json) ;; '
            '*) echo "unsafe scheduler payload path: $payload_path" >&2; exit 66 ;; esac',
            "exec python -u artifacts/code/tools/"
            "tier1_corrected_current7_slurm_seed_runner.py "
            '--bundle-root "$PWD" --payload "$payload_path" '
            '--payload-root "$payload_root" '
            f"--payload-sha256 {payload_sha256}",
        ]
    )


def _resource_policy_from_task(task: Mapping[str, Any]) -> dict[str, Any]:
    return {field: copy.deepcopy(task[field]) for field in RESOURCE_FIELDS}


def _phase_a_child_task(task: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the exact 4-CPU/4-thread Phase-A execution envelope."""

    legacy = validate_task(task)
    transformed = copy.deepcopy(legacy)
    payload = transformed["payload_json"]
    scheduler_cpus = _integer(transformed.get("cpus"), "child Scheduler CPUs", minimum=1)
    payload["scheduler_cpus"] = scheduler_cpus
    payload["inference_threads"] = scheduler_cpus
    payload_sha = canonical_sha256(payload)
    transformed["command"] = _single_seed_command(payload_sha)
    transformed["dedupe_key"] = _expected_child_dedupe(
        payload, _resource_policy_from_task(transformed)
    )
    return transformed


def _legacy_child_task_from_phase_a(
    task: Mapping[str, Any],
    *,
    expected_stage: Any | None = None,
    expected_wave: str | None = None,
) -> dict[str, Any]:
    """Reverse the execution-only thread transform and authenticate science."""

    actual = copy.deepcopy(dict(task))
    payload = actual.get("payload_json")
    if not isinstance(payload, dict):
        raise RuntimeError("Phase A child payload must be an object")
    scheduler_cpus = payload.get("scheduler_cpus")
    if (
        isinstance(scheduler_cpus, bool)
        or not isinstance(scheduler_cpus, int)
        or scheduler_cpus <= 0
        or scheduler_cpus != actual.get("cpus")
        or payload.get("inference_threads") != scheduler_cpus
    ):
        raise RuntimeError("Phase A child CPU/thread contract drifted")
    payload.pop("scheduler_cpus")
    payload["inference_threads"] = LEGACY_SINGLE_SEED_INFERENCE_THREADS
    payload_sha = canonical_sha256(payload)
    actual["command"] = _single_seed_command(payload_sha)
    actual["dedupe_key"] = _expected_child_dedupe(
        payload, _resource_policy_from_task(actual)
    )
    return validate_task(
        actual,
        expected_stage=expected_stage,
        expected_wave=expected_wave,
    )


def _validate_phase_a_child_task(
    task: Mapping[str, Any],
    *,
    expected_stage: Any | None = None,
    expected_wave: str | None = None,
) -> dict[str, Any]:
    actual = copy.deepcopy(dict(task))
    legacy = _legacy_child_task_from_phase_a(
        actual,
        expected_stage=expected_stage,
        expected_wave=expected_wave,
    )
    expected = _phase_a_child_task(legacy)
    if actual != expected:
        raise RuntimeError("Phase A child transform is not canonical")
    return actual


def _batch_command(payload_sha256: str) -> str:
    return "\n".join(
        [
            "set -euo pipefail",
            'export PYTHONPATH="$PWD/artifacts/python-site:$PWD/artifacts/code${PYTHONPATH:+:$PYTHONPATH}"',
            'payload_path="${SLURM_SCHEDULER_PAYLOAD_PATH:?scheduler payload path is missing}"',
            'case "$payload_path" in /*) ;; *) payload_path="$HOME/$payload_path" ;; esac',
            'payload_path=$(realpath -e -- "$payload_path")',
            'payload_root=$(realpath -e -- "$HOME/slurm_scheduler/runs")',
            'case "$payload_path" in "$payload_root"/*/payload.json) ;; *) echo "unsafe scheduler payload path: $payload_path" >&2; exit 66 ;; esac',
            "exec python -u artifacts/code/tools/tier1_final1000_multiseed_lane_runner.py "
            '--bundle-root "$PWD" --payload "$payload_path" '
            '--payload-root "$payload_root" '
            f"--payload-sha256 {payload_sha256}",
        ]
    )


def batch_manifest_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the immutable on-lane manifest from a validated parent payload."""

    validate_batch_payload(payload)
    unsigned = {
        "schema_version": BATCH_MANIFEST_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "bundle_id": payload["bundle_id"],
        "bundle_manifest_sha256": payload["bundle_manifest_sha256"],
        "stage_id": payload["stage_id"],
        "stage_profile_sha256": payload["stage_profile_sha256"],
        "wave": payload["wave"],
        "batch_length": payload["batch_length"],
        "ordered_children": [_child_summary(child) for child in payload["children"]],
        "resource_policy": copy.deepcopy(payload["resource_policy"]),
        "internal_deadline_seconds": payload["internal_deadline_seconds"],
        "cleanup_reserve_seconds": payload["cleanup_reserve_seconds"],
        "minimum_child_start_budget_seconds": payload[
            "minimum_child_start_budget_seconds"
        ],
        "subprocess_per_seed": True,
        "model_context_reuse": False,
        "seed_reuse_allowed": False,
        "scheduler_mutation_performed": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    return {**unsigned, "manifest_sha256": canonical_sha256(unsigned)}


def _validate_child(
    child: Mapping[str, Any],
    *,
    ordinal: int,
    stage_id: str,
    wave: str,
    bundle_id: str,
    bundle_manifest_sha256: str,
    resource_policy: Mapping[str, Any],
) -> dict[str, Any]:
    if set(child) != {
        "ordinal",
        "seed",
        "task",
        "payload_sha256",
        "logical_dedupe_key",
        "child_task_sha256",
    }:
        raise RuntimeError("batch child fields drifted")
    task = child.get("task")
    if not isinstance(task, dict):
        raise RuntimeError("batch child task must be an object")
    task = _validate_phase_a_child_task(
        task,
        expected_stage=BY_ID[stage_id],
        expected_wave=wave,
    )
    payload = task["payload_json"]
    seed = _integer(child.get("seed"), "child seed")
    lane = payload.get("lane") or {}
    expected_name = f"{BY_ID[stage_id].task_name_stem}-{wave}-{seed}"
    if (
        child.get("ordinal") != ordinal
        or payload.get("schema_version") != TASK_SCHEMA
        or payload.get("seed") != seed
        or lane.get("seed") != seed
        or payload.get("final_goal_stage_id") != stage_id
        or payload.get("bundle_id") != bundle_id
        or payload.get("bundle_manifest_sha256") != bundle_manifest_sha256
        or child.get("payload_sha256") != canonical_sha256(payload)
        or not _is_sha256(child.get("payload_sha256"))
        or not _is_sha256(child.get("child_task_sha256"))
        or child.get("child_task_sha256") != canonical_sha256(task)
        or child.get("logical_dedupe_key")
        != _expected_child_dedupe(payload, resource_policy)
        or child.get("logical_dedupe_key") != task.get("dedupe_key")
        or payload.get("scheduler_cpus") != resource_policy.get("cpus")
        or payload.get("inference_threads") != resource_policy.get("cpus")
        or any(
            task.get(field) != resource_policy.get(field) for field in RESOURCE_FIELDS
        )
        or task.get("name") != expected_name
        or task.get("command") != _single_seed_command(str(child.get("payload_sha256")))
        or PurePosixPath(str(task.get("remote_cwd") or "")).name != bundle_id
        or any(
            payload.get(field) is not False
            for field in (
                "production_eligible",
                "fea_submission_approved",
                "fea_submission_performed",
                "aedt_used",
                "automatic_promotion_allowed",
            )
        )
    ):
        raise RuntimeError("batch child scientific identity mismatch")
    return copy.deepcopy(dict(child))


def validate_batch_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "protocol_version",
        "bundle_id",
        "bundle_manifest_sha256",
        "stage_id",
        "stage_profile_sha256",
        "wave",
        "batch_length",
        "children",
        "resource_policy",
        "internal_deadline_seconds",
        "cleanup_reserve_seconds",
        "minimum_child_start_budget_seconds",
        "subprocess_per_seed",
        "model_context_reuse",
        "seed_reuse_allowed",
        "production_eligible",
        "fea_submission_approved",
        "fea_submission_performed",
        "aedt_used",
        "automatic_promotion_allowed",
    }
    if set(value) != required:
        raise RuntimeError("multi-seed parent payload fields drifted")
    stage_id = str(value.get("stage_id") or "")
    children = value.get("children")
    batch_length = _integer(value.get("batch_length"), "batch length", minimum=1)
    if (
        value.get("schema_version") != BATCH_PAYLOAD_SCHEMA
        or value.get("protocol_version") != PROTOCOL_VERSION
        or stage_id not in BY_ID
        or not _is_sha256(value.get("stage_profile_sha256"))
        or not _is_sha256(value.get("bundle_manifest_sha256"))
        or not isinstance(value.get("bundle_id"), str)
        or not value["bundle_id"]
        or value.get("stage_profile_sha256") != stage_profile(BY_ID[stage_id])["sha256"]
        or value.get("wave") not in {"canary", "ramp", "refill"}
        or not MIN_BATCH_LENGTH <= batch_length <= MAX_BATCH_LENGTH
        or not isinstance(children, list)
        or len(children) != batch_length
        or value.get("subprocess_per_seed") is not True
        or value.get("model_context_reuse") is not False
        or value.get("seed_reuse_allowed") is not False
        or any(
            value.get(field) is not False
            for field in (
                "production_eligible",
                "fea_submission_approved",
                "fea_submission_performed",
                "aedt_used",
                "automatic_promotion_allowed",
            )
        )
    ):
        raise RuntimeError("multi-seed parent payload identity mismatch")
    policy = value.get("resource_policy")
    if (
        not isinstance(policy, dict)
        or set(policy) != set(RESOURCE_FIELDS)
        or not str(policy.get("remote_cwd") or "").startswith("/")
        or policy.get("cpus") != 4
        or policy.get("memory_mb") != 28 * 1024
        or policy.get("max_workers_per_node") != 32
        or policy.get("priority") != 1
        or policy.get("scheduling_profile") != "standard"
        or policy.get("gpus") != 0
    ):
        raise RuntimeError("multi-seed lane resource policy drifted")
    deadline = _finite(
        value.get("internal_deadline_seconds"), "internal deadline", positive=True
    )
    reserve = _finite(
        value.get("cleanup_reserve_seconds"), "cleanup reserve", positive=True
    )
    child_budget = _finite(
        value.get("minimum_child_start_budget_seconds"),
        "minimum child start budget",
        positive=True,
    )
    if (
        deadline + reserve >= int(policy["timeout_seconds"])
        or child_budget + reserve >= int(policy["timeout_seconds"])
        or child_budget >= deadline
    ):
        raise RuntimeError("internal deadline leaves no Scheduler cleanup reserve")
    normalized = [
        _validate_child(
            child,
            ordinal=index,
            stage_id=stage_id,
            wave=str(value["wave"]),
            bundle_id=str(value["bundle_id"]),
            bundle_manifest_sha256=str(value["bundle_manifest_sha256"]),
            resource_policy=policy,
        )
        for index, child in enumerate(children)
    ]
    seeds = [int(child["seed"]) for child in normalized]
    dedupes = [str(child["logical_dedupe_key"]) for child in normalized]
    hashes = [str(child["payload_sha256"]) for child in normalized]
    if (
        len(set(seeds)) != batch_length
        or len(set(dedupes)) != batch_length
        or len(set(hashes)) != batch_length
        or seeds != list(range(seeds[0], seeds[0] + batch_length))
    ):
        raise RuntimeError("batch children are not an exact unique ordered seed block")
    return copy.deepcopy(dict(value))


def build_batch_task(
    child_tasks: Sequence[Mapping[str, Any]],
    *,
    internal_deadline_seconds: int = 82_800,
    cleanup_reserve_seconds: int = 1_800,
    minimum_child_start_budget_seconds: int = 7_200,
) -> dict[str, Any]:
    """Build one finite parent envelope from exact ordered single-seed tasks."""

    if len(child_tasks) not in PHASE_A_OPERATIONAL_BATCH_LENGTHS:
        raise RuntimeError("Phase A operational batch length must be 1 or 4")
    legacy = [validate_task(task) for task in child_tasks]
    legacy_first = legacy[0]
    legacy_first_payload = legacy_first["payload_json"]
    stage_id = str(legacy_first_payload["final_goal_stage_id"])
    legacy_resource_identity = {
        field: copy.deepcopy(legacy_first[field]) for field in RESOURCE_FIELDS
    }
    for task in legacy[1:]:
        payload = task["payload_json"]
        if (
            payload["final_goal_stage_id"] != stage_id
            or payload["bundle_id"] != legacy_first_payload["bundle_id"]
            or payload["bundle_manifest_sha256"]
            != legacy_first_payload["bundle_manifest_sha256"]
            or payload["final_goal_stage_profile_sha256"]
            != legacy_first_payload["final_goal_stage_profile_sha256"]
            or {field: task[field] for field in RESOURCE_FIELDS}
            != legacy_resource_identity
        ):
            raise RuntimeError(
                "multi-seed children cross a bundle/stage/resource boundary"
            )
    validated = [_phase_a_child_task(task) for task in legacy]
    first = validated[0]
    first_payload = first["payload_json"]
    stage_id = str(first_payload["final_goal_stage_id"])
    resource_identity = {
        field: copy.deepcopy(first[field]) for field in RESOURCE_FIELDS
    }
    for task in validated[1:]:
        payload = task["payload_json"]
        if (
            payload["final_goal_stage_id"] != stage_id
            or payload["bundle_id"] != first_payload["bundle_id"]
            or payload["bundle_manifest_sha256"]
            != first_payload["bundle_manifest_sha256"]
            or payload["final_goal_stage_profile_sha256"]
            != first_payload["final_goal_stage_profile_sha256"]
            or {field: task[field] for field in RESOURCE_FIELDS} != resource_identity
        ):
            raise RuntimeError(
                "multi-seed children cross a bundle/stage/resource boundary"
            )
    children = []
    for ordinal, task in enumerate(validated):
        child_task = copy.deepcopy(task)
        payload = child_task["payload_json"]
        children.append(
            {
                "ordinal": ordinal,
                "seed": int(payload["seed"]),
                "task": child_task,
                "payload_sha256": canonical_sha256(payload),
                "logical_dedupe_key": str(child_task["dedupe_key"]),
                "child_task_sha256": canonical_sha256(child_task),
            }
        )
    policy = {field: copy.deepcopy(first[field]) for field in RESOURCE_FIELDS}
    payload = {
        "schema_version": BATCH_PAYLOAD_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "bundle_id": first_payload["bundle_id"],
        "bundle_manifest_sha256": first_payload["bundle_manifest_sha256"],
        "stage_id": stage_id,
        "stage_profile_sha256": first_payload["final_goal_stage_profile_sha256"],
        "wave": first_payload["lane"]["wave"],
        "batch_length": len(children),
        "children": children,
        "resource_policy": policy,
        "internal_deadline_seconds": int(internal_deadline_seconds),
        "cleanup_reserve_seconds": int(cleanup_reserve_seconds),
        "minimum_child_start_budget_seconds": int(minimum_child_start_budget_seconds),
        "subprocess_per_seed": True,
        "model_context_reuse": False,
        "seed_reuse_allowed": False,
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    validate_batch_payload(payload)
    manifest = batch_manifest_from_payload(payload)
    payload_sha = canonical_sha256(payload)
    seed_start = children[0]["seed"]
    seed_end = children[-1]["seed"]
    command = _batch_command(payload_sha)
    dedupe_identity = {
        "protocol_version": PROTOCOL_VERSION,
        "manifest_sha256": manifest["manifest_sha256"],
        "bundle_id": payload["bundle_id"],
        "stage_id": stage_id,
        "ordered_child_identities": [_child_summary(child) for child in children],
        "resource_policy": policy,
    }
    task = {
        **{field: copy.deepcopy(first[field]) for field in REQUIRED_SCHEDULER_FIELDS},
        "name": f"mft-t1fg-lane-{stage_id[:8]}-{seed_start}-{seed_end}",
        "command": command,
        "payload_json": payload,
        "dedupe_key": f"{DEDUPE_PREFIX}{canonical_sha256(dedupe_identity)}",
    }
    return validate_batch_task(task)


def validate_batch_task(task: Mapping[str, Any]) -> dict[str, Any]:
    if set(task) != REQUIRED_SCHEDULER_FIELDS or "requested_allocation_id" in task:
        raise RuntimeError("multi-seed Scheduler envelope fields drifted")
    payload = validate_batch_payload(task.get("payload_json") or {})
    policy = payload["resource_policy"]
    seeds = [int(child["seed"]) for child in payload["children"]]
    expected_name = f"mft-t1fg-lane-{payload['stage_id'][:8]}-{seeds[0]}-{seeds[-1]}"
    if (
        task.get("name") != expected_name
        or not str(task.get("dedupe_key") or "").startswith(DEDUPE_PREFIX)
        or not str(task.get("remote_cwd") or "").startswith("/")
        or any(task.get(field) != policy.get(field) for field in policy)
        or task.get("aedt_backend") != "standalone"
        or task.get("command") != _batch_command(canonical_sha256(payload))
    ):
        raise RuntimeError("multi-seed Scheduler task seal mismatch")
    manifest = batch_manifest_from_payload(payload)
    expected_dedupe = canonical_sha256(
        {
            "protocol_version": PROTOCOL_VERSION,
            "manifest_sha256": manifest["manifest_sha256"],
            "bundle_id": payload["bundle_id"],
            "stage_id": payload["stage_id"],
            "ordered_child_identities": [
                _child_summary(child) for child in payload["children"]
            ],
            "resource_policy": policy,
        }
    )
    if task["dedupe_key"] != f"{DEDUPE_PREFIX}{expected_dedupe}":
        raise RuntimeError("multi-seed parent dedupe identity mismatch")
    return copy.deepcopy(dict(task))


def seal_task_status(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if key != "status_sha256"
    }
    sealed = {**unsigned, "status_sha256": canonical_sha256(unsigned)}
    return validate_task_status(sealed)


def validate_task_status(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "status_sha256"}
    required = {
        "schema_version",
        "protocol_version",
        "task_id",
        "manifest_sha256",
        "state",
        "stop_requested",
        "stop_reason",
        "current_ordinal",
        "current_seed",
        "sealed_child_count",
        "completed_child_count",
        "failed_child_count",
        "started_at",
        "updated_at",
        "finished_at",
        "subprocess_per_seed",
        "model_context_reuse",
        "scheduler_mutation_performed",
        "fea_submission_performed",
        "aedt_used",
        "status_sha256",
    }
    state = str(value.get("state") or "")
    sealed_count = _integer(value.get("sealed_child_count"), "sealed child count")
    completed = _integer(value.get("completed_child_count"), "completed child count")
    failed = _integer(value.get("failed_child_count"), "failed child count")
    current_ordinal = value.get("current_ordinal")
    if current_ordinal is not None:
        _integer(current_ordinal, "current ordinal")
    current_seed = value.get("current_seed")
    if current_seed is not None:
        _integer(current_seed, "current seed")
    if (
        set(value) != required
        or value.get("schema_version") != TASK_STATUS_SCHEMA
        or value.get("protocol_version") != PROTOCOL_VERSION
        or not _is_sha256(value.get("manifest_sha256"))
        or value.get("status_sha256") != canonical_sha256(unsigned)
        or state not in {"starting", "running", "stopping"} | TERMINAL_PARENT_STATES
        or not str(value.get("task_id") or "")
        or completed + failed != sealed_count
        or (current_ordinal is None) != (current_seed is None)
        or value.get("subprocess_per_seed") is not True
        or value.get("model_context_reuse") is not False
        or value.get("scheduler_mutation_performed") is not False
        or value.get("fea_submission_performed") is not False
        or value.get("aedt_used") is not False
        or (state in TERMINAL_PARENT_STATES) != (value.get("finished_at") is not None)
        or (state in TERMINAL_PARENT_STATES and current_ordinal is not None)
        or (state == "starting" and (sealed_count != 0 or current_ordinal is not None))
        or (state == "stopping" and value.get("stop_requested") is not True)
        or (
            state in {"stopped", "deadline"} and value.get("stop_requested") is not True
        )
        or (
            state in {"completed", "completed_with_failures", "failed"}
            and value.get("stop_requested") is not False
        )
        or (state == "completed" and failed != 0)
        or (state == "completed_with_failures" and failed == 0)
    ):
        raise RuntimeError("multi-seed task status seal mismatch")
    return copy.deepcopy(dict(value))


def seal_child_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if key != "receipt_sha256"
    }
    sealed = {**unsigned, "receipt_sha256": canonical_sha256(unsigned)}
    return validate_child_receipt(sealed)


def validate_child_receipt(
    value: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    required = {
        "schema_version",
        "protocol_version",
        "task_id",
        "manifest_sha256",
        "ordinal",
        "seed",
        "payload_sha256",
        "logical_dedupe_key",
        "state",
        "terminal",
        "lane_fatal",
        "exit_code",
        "legacy_status",
        "legacy_status_sha256",
        "result_sha256",
        "started_at",
        "finished_at",
        "wall_time_seconds",
        "failure",
        "production_eligible",
        "fea_submission_performed",
        "aedt_used",
        "receipt_sha256",
    }
    state = str(value.get("state") or "")
    legacy = value.get("legacy_status")
    result_sha = value.get("result_sha256")
    exit_code = value.get("exit_code")
    wall_time = value.get("wall_time_seconds")
    if (
        set(value) != required
        or value.get("schema_version") != CHILD_RECEIPT_SCHEMA
        or value.get("protocol_version") != PROTOCOL_VERSION
        or state not in TERMINAL_CHILD_STATES
        or not str(value.get("task_id") or "")
        or value.get("terminal") is not True
        or not isinstance(value.get("lane_fatal"), bool)
        or isinstance(exit_code, bool)
        or not isinstance(exit_code, int)
        or isinstance(wall_time, bool)
        or not isinstance(wall_time, (int, float))
        or not math.isfinite(float(wall_time))
        or float(wall_time) < 0
        or not str(value.get("logical_dedupe_key") or "").startswith(
            "mft-tier1-final1000:"
        )
        or not _is_sha256(value.get("manifest_sha256"))
        or not _is_sha256(value.get("payload_sha256"))
        or not _is_sha256(value.get("legacy_status_sha256"))
        or not isinstance(legacy, dict)
        or canonical_sha256(legacy) != value.get("legacy_status_sha256")
        or (result_sha is not None and not _is_sha256(result_sha))
        or (state == "completed") != (result_sha is not None)
        or value.get("receipt_sha256") != canonical_sha256(unsigned)
        or value.get("production_eligible") is not False
        or value.get("fea_submission_performed") is not False
        or value.get("aedt_used") is not False
    ):
        raise RuntimeError("multi-seed child terminal receipt seal mismatch")
    ordinal = _integer(value.get("ordinal"), "receipt ordinal")
    seed = _integer(value.get("seed"), "receipt seed")
    if manifest is not None:
        if manifest.get("schema_version") != BATCH_MANIFEST_SCHEMA:
            raise RuntimeError("child receipt was bound to a foreign batch manifest")
        children = manifest.get("ordered_children") or []
        if (
            manifest.get("manifest_sha256") != value.get("manifest_sha256")
            or ordinal >= len(children)
            or children[ordinal].get("seed") != seed
            or children[ordinal].get("payload_sha256") != value.get("payload_sha256")
            or children[ordinal].get("logical_dedupe_key")
            != value.get("logical_dedupe_key")
        ):
            raise RuntimeError("child receipt/manifest binding mismatch")
    return copy.deepcopy(dict(value))


def validate_batch_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "manifest_sha256"}
    children = value.get("ordered_children")
    required = {
        "schema_version",
        "protocol_version",
        "bundle_id",
        "bundle_manifest_sha256",
        "stage_id",
        "stage_profile_sha256",
        "wave",
        "batch_length",
        "ordered_children",
        "resource_policy",
        "internal_deadline_seconds",
        "cleanup_reserve_seconds",
        "minimum_child_start_budget_seconds",
        "subprocess_per_seed",
        "model_context_reuse",
        "seed_reuse_allowed",
        "scheduler_mutation_performed",
        "fea_submission_approved",
        "fea_submission_performed",
        "aedt_used",
        "manifest_sha256",
    }
    if (
        set(value) != required
        or value.get("schema_version") != BATCH_MANIFEST_SCHEMA
        or value.get("protocol_version") != PROTOCOL_VERSION
        or value.get("manifest_sha256") != canonical_sha256(unsigned)
        or not isinstance(children, list)
        or len(children) != value.get("batch_length")
        or not MIN_BATCH_LENGTH <= len(children) <= MAX_BATCH_LENGTH
        or [child.get("ordinal") for child in children] != list(range(len(children)))
        or [child.get("seed") for child in children]
        != list(
            range(
                int(children[0].get("seed", -1)),
                int(children[0].get("seed", -1)) + len(children),
            )
        )
        or len({child.get("payload_sha256") for child in children}) != len(children)
        or len({child.get("logical_dedupe_key") for child in children}) != len(children)
        or value.get("subprocess_per_seed") is not True
        or value.get("model_context_reuse") is not False
        or value.get("seed_reuse_allowed") is not False
        or value.get("scheduler_mutation_performed") is not False
        or value.get("fea_submission_approved") is not False
        or value.get("fea_submission_performed") is not False
        or value.get("aedt_used") is not False
    ):
        raise RuntimeError("multi-seed batch manifest seal mismatch")
    for child in children:
        if (
            not _is_sha256(child.get("payload_sha256"))
            or not _is_sha256(child.get("child_task_sha256"))
            or not str(child.get("logical_dedupe_key") or "").startswith(
                "mft-tier1-final1000:"
            )
        ):
            raise RuntimeError("multi-seed manifest child identity mismatch")
    return copy.deepcopy(dict(value))


def receipt_remote_path(task_id: int, seed: int) -> str:
    return f"runs/task-{int(task_id)}/seed-{int(seed)}/seed_status.json"


def result_remote_path(task_id: int, seed: int, filename: str = "result.json") -> str:
    return f"runs/task-{int(task_id)}/seed-{int(seed)}/{filename}"
