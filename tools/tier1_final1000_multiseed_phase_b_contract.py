"""Fail-closed contracts for concurrent final1000 multi-seed parents.

Phase B changes only the physical Scheduler envelope and execution topology.
Every logical child remains the canonical four-CPU Phase-A child task with its
own immutable payload, dedupe identity, subprocess, seed directory, journal,
and receipt.  A parent requests the exact sum of those child resources and
runs four for the mandatory first canary, eight for normal production, or an
exact 1..7 tail shape after the canary to fill an allocation without wasting
CPU slots.

This module is side-effect free.  In particular, it never contacts Scheduler
and never submits FEA/AEDT work.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

try:
    import tier1_final1000_multiseed_contract as phase_a
    from tier1_final1000_slurm_launch import REQUIRED_SCHEDULER_FIELDS, validate_task
    from tier1_final1000_stage_profiles import BY_ID, stage_profile
except ImportError:  # pragma: no cover - repository import path
    from tools import tier1_final1000_multiseed_contract as phase_a
    from tools.tier1_final1000_slurm_launch import (
        REQUIRED_SCHEDULER_FIELDS,
        validate_task,
    )
    from tools.tier1_final1000_stage_profiles import BY_ID, stage_profile


PROTOCOL_VERSION = "final1000-finite-multiseed-phase-b-v1"
BATCH_PAYLOAD_SCHEMA = "mft-tier1-final1000-concurrent-lane-task-v1"
BATCH_MANIFEST_SCHEMA = "mft-tier1-final1000-concurrent-batch-manifest-v1"
TASK_STATUS_SCHEMA = "mft-tier1-final1000-concurrent-task-status-v1"
CHILD_RECEIPT_SCHEMA = "mft-tier1-final1000-concurrent-child-receipt-v1"
CHILD_RESOURCE_TELEMETRY_SCHEMA = "mft-tier1-final1000-phase-b-child-cpu-telemetry-v1"

CANARY_CONCURRENCY = 4
PRODUCTION_CONCURRENCY = 8
MIN_CONCURRENT_CHILDREN = 1
MAX_CONCURRENT_CHILDREN = 8
# Four is the mandatory first canary and eight is the normal production
# shape.  The other exact shapes are post-canary allocation-tail fillers: for
# example 44 CPUs pack as 8+3 children, 48 as 8+4, and 60 as 8+7.
OPERATIONAL_BATCH_LENGTHS = frozenset(
    range(MIN_CONCURRENT_CHILDREN, MAX_CONCURRENT_CHILDREN + 1)
)
CHILD_CPUS = 4
CHILD_MEMORY_MB = 28 * 1024
CANARY_CPUS = CANARY_CONCURRENCY * CHILD_CPUS
CANARY_MEMORY_MB = CANARY_CONCURRENCY * CHILD_MEMORY_MB
PRODUCTION_CPUS = PRODUCTION_CONCURRENCY * CHILD_CPUS
PRODUCTION_MEMORY_MB = PRODUCTION_CONCURRENCY * CHILD_MEMORY_MB
MIN_MODEL_LOAD_STAGGER_SECONDS = 5.0
MAX_MODEL_LOAD_STAGGER_SECONDS = 10.0
# Use the reviewed lower bound by default.  Eight lanes therefore reach their
# final child launch in 35 seconds instead of 52.5 seconds, while preserving
# the mandatory five-second separation between model-load starts.
DEFAULT_MODEL_LOAD_STAGGER_SECONDS = 5.0
CPU_ISOLATION = {
    "strategy": "child-self-sched-setaffinity-v1",
    "cpus_per_child": CHILD_CPUS,
    "disjoint_child_cpuset_required": True,
    "inherited_scheduler_cpuset_required": True,
    "nested_srun_used": False,
}
RUNTIME_ISOLATION = {
    "root_template": "runs/task-{physical_task_id}/seed-{seed}/runtime",
    "tmp_relative_path": "tmp",
    "joblib_relative_path": "joblib",
    "cache_relative_path": "cache",
    "environment_variables": {
        "TMPDIR": "tmp",
        "TMP": "tmp",
        "TEMP": "tmp",
        "JOBLIB_TEMP_FOLDER": "joblib",
        "XDG_CACHE_HOME": "cache",
    },
    "cleanup_owner": "parent-after-child-reaped",
    "cleanup_scope": "child-runtime-root-only",
    "shared_tmp_cleanup_allowed": False,
}
INFERENCE_SAFETY = {
    "policy": "family_specific_semaphore_free_sklearn_forest_v1",
    "required_release_commit": "6ea0e17e5e028ebb8d89d910c3cec1a0f014dae5",
    "required_smoke_schema": "mft-tier1-semlock-safe-inference-smoke-v1",
    "required_helper_path": "tools/tier1_semlock_safe_inference_smoke.py",
    "required_helper_sha256": (
        "c25afd39752a990c31e1b934c77282d88ac11dab1cad763d4f07c95e4b87dd80"
    ),
    "semaphore_free_sklearn_families": ["extratrees", "randomforest"],
    "sklearn_forest_n_jobs": 1,
    "repeated_predict_stress_required": True,
    "minimum_repeated_predict_calls_per_target": 2,
    "joblib_thread_pool_construction_allowed": False,
    "multiprocessing_semlock_construction_allowed": False,
    "tmp_isolation_claimed_as_enospc_fix": False,
    "direct_enospc_cause": "joblib-threadpool-simplequeue-semlock-churn",
}
CPU_TELEMETRY = {
    "schema_version": "mft-tier1-final1000-phase-b-cpu-telemetry-policy-v1",
    "parent_aggregate_child_cpu_time_required": True,
    "parent_wall_time_required": True,
    "requested_cpu_capacity_seconds_required": True,
    "logical_child_capacity_seconds_required": True,
    "logical_child_utilization_required": True,
    "per_child_process_tree_cpu_time_required": True,
    "per_child_elapsed_utilization_required": True,
    "parent_child_cpu_crosscheck_required": True,
    "dispatch_fill_seconds_required": True,
    "future_comparison_child_cpu_counts": [1, 2, 4],
    "current_child_cpu_contract": CHILD_CPUS,
    "automatic_child_cpu_reduction_allowed": False,
}
DEDUPE_PREFIX = "mft-tier1-final1000:concurrent-lane:"
REMOTE_CODE_FILES = (
    "tools/tier1_corrected_current7_slurm_seed_runner.py",
    "tools/tier1_final1000_multiseed_contract.py",
    "tools/tier1_final1000_multiseed_phase_b_contract.py",
    "tools/tier1_final1000_multiseed_phase_b_runner.py",
    "tools/tier1_semlock_safe_inference_smoke.py",
)


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


def _parent_resource_policy(
    child_policy: Mapping[str, Any], concurrency: int
) -> dict[str, Any]:
    value = copy.deepcopy(dict(child_policy))
    value["cpus"] = int(concurrency) * CHILD_CPUS
    value["memory_mb"] = int(concurrency) * CHILD_MEMORY_MB
    return value


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
            "exec python -u artifacts/code/tools/"
            "tier1_final1000_multiseed_phase_b_runner.py "
            '--bundle-root "$PWD" --payload "$payload_path" '
            '--payload-root "$payload_root" '
            f"--payload-sha256 {payload_sha256}",
        ]
    )


def _child_summary(child: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ordinal": int(child["ordinal"]),
        "seed": int(child["seed"]),
        "payload_sha256": str(child["payload_sha256"]),
        "logical_dedupe_key": str(child["logical_dedupe_key"]),
        "child_task_sha256": str(child["child_task_sha256"]),
    }


def batch_manifest_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
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
        "child_resource_policy": copy.deepcopy(payload["child_resource_policy"]),
        "parent_resource_policy": copy.deepcopy(payload["parent_resource_policy"]),
        "execution_mode": "concurrent",
        "concurrent_children": payload["concurrent_children"],
        "cpu_isolation": copy.deepcopy(payload["cpu_isolation"]),
        "runtime_isolation": copy.deepcopy(payload["runtime_isolation"]),
        "inference_safety": copy.deepcopy(payload["inference_safety"]),
        "cpu_telemetry": copy.deepcopy(payload["cpu_telemetry"]),
        "model_load_stagger_seconds": payload["model_load_stagger_seconds"],
        "internal_deadline_seconds": payload["internal_deadline_seconds"],
        "cleanup_reserve_seconds": payload["cleanup_reserve_seconds"],
        "minimum_child_start_budget_seconds": payload[
            "minimum_child_start_budget_seconds"
        ],
        "subprocess_per_seed": True,
        "model_context_reuse": False,
        "rng_context_reuse": False,
        "seed_reuse_allowed": False,
        "physical_scheduler_task_count": 1,
        "virtual_scheduler_task_ids_created": False,
        "terminal_receipt_commit_rule": "contiguous-ordinal-prefix",
        "scheduler_mutation_performed": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    return {**unsigned, "manifest_sha256": phase_a.canonical_sha256(unsigned)}


def _required_payload_fields() -> set[str]:
    return {
        "schema_version",
        "protocol_version",
        "bundle_id",
        "bundle_manifest_sha256",
        "stage_id",
        "stage_profile_sha256",
        "wave",
        "batch_length",
        "children",
        "child_resource_policy",
        "parent_resource_policy",
        "execution_mode",
        "concurrent_children",
        "cpu_isolation",
        "runtime_isolation",
        "inference_safety",
        "cpu_telemetry",
        "model_load_stagger_seconds",
        "internal_deadline_seconds",
        "cleanup_reserve_seconds",
        "minimum_child_start_budget_seconds",
        "subprocess_per_seed",
        "model_context_reuse",
        "rng_context_reuse",
        "seed_reuse_allowed",
        "production_eligible",
        "fea_submission_approved",
        "fea_submission_performed",
        "aedt_used",
        "automatic_promotion_allowed",
    }


def validate_batch_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != _required_payload_fields():
        raise RuntimeError("Phase B parent payload fields drifted")
    stage_id = str(value.get("stage_id") or "")
    children = value.get("children")
    batch_length = _integer(value.get("batch_length"), "batch length", minimum=1)
    concurrency = _integer(
        value.get("concurrent_children"), "concurrent children", minimum=1
    )
    stagger = _finite(
        value.get("model_load_stagger_seconds"),
        "model-load stagger",
        positive=True,
    )
    if (
        value.get("schema_version") != BATCH_PAYLOAD_SCHEMA
        or value.get("protocol_version") != PROTOCOL_VERSION
        or stage_id not in BY_ID
        or not _is_sha256(value.get("stage_profile_sha256"))
        or value.get("stage_profile_sha256") != stage_profile(BY_ID[stage_id])["sha256"]
        or not _is_sha256(value.get("bundle_manifest_sha256"))
        or not str(value.get("bundle_id") or "")
        or value.get("wave") not in {"canary", "ramp", "refill"}
        or batch_length not in OPERATIONAL_BATCH_LENGTHS
        or concurrency != batch_length
        or not isinstance(children, list)
        or len(children) != batch_length
        or value.get("execution_mode") != "concurrent"
        or value.get("cpu_isolation") != CPU_ISOLATION
        or value.get("runtime_isolation") != RUNTIME_ISOLATION
        or value.get("inference_safety") != INFERENCE_SAFETY
        or value.get("cpu_telemetry") != CPU_TELEMETRY
        or not MIN_MODEL_LOAD_STAGGER_SECONDS
        <= stagger
        <= MAX_MODEL_LOAD_STAGGER_SECONDS
        or value.get("subprocess_per_seed") is not True
        or value.get("model_context_reuse") is not False
        or value.get("rng_context_reuse") is not False
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
        raise RuntimeError("Phase B parent payload identity mismatch")
    child_policy = value.get("child_resource_policy")
    parent_policy = value.get("parent_resource_policy")
    if (
        not isinstance(child_policy, dict)
        or set(child_policy) != set(phase_a.RESOURCE_FIELDS)
        or child_policy.get("cpus") != CHILD_CPUS
        or child_policy.get("memory_mb") != CHILD_MEMORY_MB
        or child_policy.get("max_workers_per_node") != 32
        or child_policy.get("priority") != 1
        or child_policy.get("scheduling_profile") != "standard"
        or child_policy.get("gpus") != 0
        or not str(child_policy.get("remote_cwd") or "").startswith("/")
        or not isinstance(parent_policy, dict)
        or parent_policy != _parent_resource_policy(child_policy, concurrency)
    ):
        raise RuntimeError("Phase B CPU/memory envelope drifted")
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
        deadline + reserve >= int(parent_policy["timeout_seconds"])
        or child_budget + reserve >= int(parent_policy["timeout_seconds"])
        or child_budget >= deadline
    ):
        raise RuntimeError("Phase B deadline leaves no Scheduler cleanup reserve")
    normalized = [
        phase_a._validate_child(
            child,
            ordinal=index,
            stage_id=stage_id,
            wave=str(value["wave"]),
            bundle_id=str(value["bundle_id"]),
            bundle_manifest_sha256=str(value["bundle_manifest_sha256"]),
            resource_policy=child_policy,
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
        raise RuntimeError("Phase B children are not an exact ordered seed block")
    return copy.deepcopy(dict(value))


def build_concurrent_batch_task(
    child_tasks: Sequence[Mapping[str, Any]],
    *,
    internal_deadline_seconds: int = 82_800,
    cleanup_reserve_seconds: int = 1_800,
    minimum_child_start_budget_seconds: int = 7_200,
    model_load_stagger_seconds: float = DEFAULT_MODEL_LOAD_STAGGER_SECONDS,
) -> dict[str, Any]:
    """Build one exact 1..8-child parent (4 canary, 8 production)."""

    if len(child_tasks) not in OPERATIONAL_BATCH_LENGTHS:
        raise RuntimeError("Phase B operational batch length must be between 1 and 8")
    legacy = [validate_task(task) for task in child_tasks]
    first_legacy = legacy[0]
    first_payload = first_legacy["payload_json"]
    stage_id = str(first_payload["final_goal_stage_id"])
    legacy_resources = {
        field: copy.deepcopy(first_legacy[field]) for field in phase_a.RESOURCE_FIELDS
    }
    for task in legacy[1:]:
        payload = task["payload_json"]
        if (
            payload["final_goal_stage_id"] != stage_id
            or payload["bundle_id"] != first_payload["bundle_id"]
            or payload["bundle_manifest_sha256"]
            != first_payload["bundle_manifest_sha256"]
            or payload["final_goal_stage_profile_sha256"]
            != first_payload["final_goal_stage_profile_sha256"]
            or {field: task[field] for field in phase_a.RESOURCE_FIELDS}
            != legacy_resources
        ):
            raise RuntimeError(
                "Phase B children cross a bundle/stage/resource boundary"
            )
    transformed = [phase_a._phase_a_child_task(task) for task in legacy]
    first = transformed[0]
    first_payload = first["payload_json"]
    child_policy = {
        field: copy.deepcopy(first[field]) for field in phase_a.RESOURCE_FIELDS
    }
    children: list[dict[str, Any]] = []
    for ordinal, task in enumerate(transformed):
        payload = task["payload_json"]
        children.append(
            {
                "ordinal": ordinal,
                "seed": int(payload["seed"]),
                "task": copy.deepcopy(task),
                "payload_sha256": phase_a.canonical_sha256(payload),
                "logical_dedupe_key": str(task["dedupe_key"]),
                "child_task_sha256": phase_a.canonical_sha256(task),
            }
        )
    concurrency = len(children)
    parent_policy = _parent_resource_policy(child_policy, concurrency)
    payload = {
        "schema_version": BATCH_PAYLOAD_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "bundle_id": first_payload["bundle_id"],
        "bundle_manifest_sha256": first_payload["bundle_manifest_sha256"],
        "stage_id": stage_id,
        "stage_profile_sha256": first_payload["final_goal_stage_profile_sha256"],
        "wave": first_payload["lane"]["wave"],
        "batch_length": concurrency,
        "children": children,
        "child_resource_policy": child_policy,
        "parent_resource_policy": parent_policy,
        "execution_mode": "concurrent",
        "concurrent_children": concurrency,
        "cpu_isolation": copy.deepcopy(CPU_ISOLATION),
        "runtime_isolation": copy.deepcopy(RUNTIME_ISOLATION),
        "inference_safety": copy.deepcopy(INFERENCE_SAFETY),
        "cpu_telemetry": copy.deepcopy(CPU_TELEMETRY),
        "model_load_stagger_seconds": float(model_load_stagger_seconds),
        "internal_deadline_seconds": int(internal_deadline_seconds),
        "cleanup_reserve_seconds": int(cleanup_reserve_seconds),
        "minimum_child_start_budget_seconds": int(minimum_child_start_budget_seconds),
        "subprocess_per_seed": True,
        "model_context_reuse": False,
        "rng_context_reuse": False,
        "seed_reuse_allowed": False,
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    validate_batch_payload(payload)
    manifest = batch_manifest_from_payload(payload)
    payload_sha = phase_a.canonical_sha256(payload)
    seed_start = int(children[0]["seed"])
    seed_end = int(children[-1]["seed"])
    dedupe_identity = {
        "protocol_version": PROTOCOL_VERSION,
        "manifest_sha256": manifest["manifest_sha256"],
        "parent_payload_sha256": payload_sha,
        "bundle_id": payload["bundle_id"],
        "stage_id": stage_id,
        "ordered_child_identities": [_child_summary(child) for child in children],
        "child_resource_policy": child_policy,
        "parent_resource_policy": parent_policy,
    }
    task = {
        **{field: copy.deepcopy(first[field]) for field in REQUIRED_SCHEDULER_FIELDS},
        **{
            field: copy.deepcopy(parent_policy[field])
            for field in phase_a.RESOURCE_FIELDS
        },
        "name": f"mft-t1fg-clane-{stage_id[:8]}-{seed_start}-{seed_end}",
        "command": _batch_command(payload_sha),
        "payload_json": payload,
        "dedupe_key": f"{DEDUPE_PREFIX}{phase_a.canonical_sha256(dedupe_identity)}",
    }
    return validate_batch_task(task)


def validate_batch_task(task: Mapping[str, Any]) -> dict[str, Any]:
    if set(task) != REQUIRED_SCHEDULER_FIELDS or "requested_allocation_id" in task:
        raise RuntimeError("Phase B Scheduler envelope fields drifted")
    payload = validate_batch_payload(task.get("payload_json") or {})
    parent_policy = payload["parent_resource_policy"]
    seeds = [int(child["seed"]) for child in payload["children"]]
    expected_name = f"mft-t1fg-clane-{payload['stage_id'][:8]}-{seeds[0]}-{seeds[-1]}"
    if (
        task.get("name") != expected_name
        or not str(task.get("dedupe_key") or "").startswith(DEDUPE_PREFIX)
        or any(task.get(field) != parent_policy.get(field) for field in parent_policy)
        or task.get("aedt_backend") != "standalone"
        or task.get("command") != _batch_command(phase_a.canonical_sha256(payload))
    ):
        raise RuntimeError("Phase B Scheduler task seal mismatch")
    manifest = batch_manifest_from_payload(payload)
    expected_dedupe = phase_a.canonical_sha256(
        {
            "protocol_version": PROTOCOL_VERSION,
            "manifest_sha256": manifest["manifest_sha256"],
            "parent_payload_sha256": phase_a.canonical_sha256(payload),
            "bundle_id": payload["bundle_id"],
            "stage_id": payload["stage_id"],
            "ordered_child_identities": [
                _child_summary(child) for child in payload["children"]
            ],
            "child_resource_policy": payload["child_resource_policy"],
            "parent_resource_policy": parent_policy,
        }
    )
    if task["dedupe_key"] != f"{DEDUPE_PREFIX}{expected_dedupe}":
        raise RuntimeError("Phase B parent dedupe identity mismatch")
    return copy.deepcopy(dict(task))


def validate_batch_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "manifest_sha256"}
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
        "child_resource_policy",
        "parent_resource_policy",
        "execution_mode",
        "concurrent_children",
        "cpu_isolation",
        "runtime_isolation",
        "inference_safety",
        "cpu_telemetry",
        "model_load_stagger_seconds",
        "internal_deadline_seconds",
        "cleanup_reserve_seconds",
        "minimum_child_start_budget_seconds",
        "subprocess_per_seed",
        "model_context_reuse",
        "rng_context_reuse",
        "seed_reuse_allowed",
        "physical_scheduler_task_count",
        "virtual_scheduler_task_ids_created",
        "terminal_receipt_commit_rule",
        "scheduler_mutation_performed",
        "fea_submission_approved",
        "fea_submission_performed",
        "aedt_used",
        "manifest_sha256",
    }
    children = value.get("ordered_children")
    if (
        set(value) != required
        or value.get("schema_version") != BATCH_MANIFEST_SCHEMA
        or value.get("protocol_version") != PROTOCOL_VERSION
        or value.get("manifest_sha256") != phase_a.canonical_sha256(unsigned)
        or not isinstance(children, list)
        or len(children) not in OPERATIONAL_BATCH_LENGTHS
        or value.get("batch_length") != len(children)
        or value.get("concurrent_children") != len(children)
        or [child.get("ordinal") for child in children] != list(range(len(children)))
        or [child.get("seed") for child in children]
        != list(
            range(
                int(children[0].get("seed", -1)),
                int(children[0].get("seed", -1)) + len(children),
            )
        )
        or value.get("execution_mode") != "concurrent"
        or value.get("cpu_isolation") != CPU_ISOLATION
        or value.get("runtime_isolation") != RUNTIME_ISOLATION
        or value.get("inference_safety") != INFERENCE_SAFETY
        or value.get("cpu_telemetry") != CPU_TELEMETRY
        or value.get("subprocess_per_seed") is not True
        or value.get("model_context_reuse") is not False
        or value.get("rng_context_reuse") is not False
        or value.get("seed_reuse_allowed") is not False
        or value.get("physical_scheduler_task_count") != 1
        or value.get("virtual_scheduler_task_ids_created") is not False
        or value.get("terminal_receipt_commit_rule") != "contiguous-ordinal-prefix"
        or value.get("scheduler_mutation_performed") is not False
        or value.get("fea_submission_approved") is not False
        or value.get("fea_submission_performed") is not False
        or value.get("aedt_used") is not False
    ):
        raise RuntimeError("Phase B batch manifest seal mismatch")
    return copy.deepcopy(dict(value))


def seal_task_status(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if key != "status_sha256"
    }
    return validate_task_status(
        {**unsigned, "status_sha256": phase_a.canonical_sha256(unsigned)}
    )


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
        "active_children",
        "launched_child_count",
        "finished_child_count",
        "sealed_child_count",
        "completed_child_count",
        "failed_child_count",
        "started_at",
        "updated_at",
        "finished_at",
        "execution_mode",
        "concurrent_children",
        "subprocess_per_seed",
        "model_context_reuse",
        "rng_context_reuse",
        "cpu_isolation",
        "resource_telemetry",
        "physical_scheduler_task_count",
        "virtual_scheduler_task_ids_created",
        "scheduler_mutation_performed",
        "fea_submission_performed",
        "aedt_used",
        "status_sha256",
    }
    state = str(value.get("state") or "")
    active = value.get("active_children")
    launched = _integer(value.get("launched_child_count"), "launched child count")
    finished = _integer(value.get("finished_child_count"), "finished child count")
    sealed = _integer(value.get("sealed_child_count"), "sealed child count")
    completed = _integer(value.get("completed_child_count"), "completed child count")
    failed = _integer(value.get("failed_child_count"), "failed child count")
    concurrency = _integer(
        value.get("concurrent_children"), "concurrent children", minimum=1
    )
    terminal = state in phase_a.TERMINAL_PARENT_STATES
    telemetry = value.get("resource_telemetry")
    if (
        set(value) != required
        or value.get("schema_version") != TASK_STATUS_SCHEMA
        or value.get("protocol_version") != PROTOCOL_VERSION
        or value.get("status_sha256") != phase_a.canonical_sha256(unsigned)
        or not _is_sha256(value.get("manifest_sha256"))
        or not str(value.get("task_id") or "")
        or state
        not in {"starting", "running", "stopping"} | phase_a.TERMINAL_PARENT_STATES
        or not isinstance(active, list)
        or concurrency not in OPERATIONAL_BATCH_LENGTHS
        or len(active) > concurrency
        or completed + failed != sealed
        or not 0 <= sealed <= finished <= launched <= concurrency
        or launched != finished + len(active)
        or value.get("execution_mode") != "concurrent"
        or value.get("subprocess_per_seed") is not True
        or value.get("model_context_reuse") is not False
        or value.get("rng_context_reuse") is not False
        or value.get("cpu_isolation") != CPU_ISOLATION
        or (not terminal and telemetry is not None)
        or value.get("physical_scheduler_task_count") != 1
        or value.get("virtual_scheduler_task_ids_created") is not False
        or value.get("scheduler_mutation_performed") is not False
        or value.get("fea_submission_performed") is not False
        or value.get("aedt_used") is not False
        or terminal != (value.get("finished_at") is not None)
        or (terminal and active)
        or (state == "starting" and (launched or finished or sealed or active))
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
        raise RuntimeError("Phase B task status seal mismatch")
    if terminal:
        required_telemetry = {
            "schema_version",
            "available",
            "measurement",
            "parent_requested_cpus",
            "logical_child_cpus",
            "logical_child_count",
            "parent_wall_time_seconds",
            "aggregate_child_cpu_seconds",
            "requested_cpu_capacity_seconds",
            "aggregate_cpu_utilization_fraction",
            "sum_child_wall_time_seconds",
            "logical_child_cpu_capacity_seconds",
            "logical_child_cpu_utilization_fraction",
            "reported_child_cpu_available_count",
            "reported_child_process_tree_cpu_seconds",
            "reported_child_cpu_capacity_seconds",
            "reported_child_cpu_utilization_fraction",
            "reaped_minus_reported_child_cpu_seconds",
            "scheduler_envelope_idle_capacity_seconds",
            "configured_model_load_stagger_seconds",
            "dispatch_fill_seconds",
            "maximum_planned_dispatch_fill_seconds",
            "scheduler_ready_lane_reserve",
            "scheduler_admission_policy",
            "future_comparison_child_cpu_counts",
            "automatic_child_cpu_reduction_allowed",
        }
        if not isinstance(telemetry, dict) or set(telemetry) != required_telemetry:
            raise RuntimeError("Phase B terminal CPU telemetry is unavailable")
        wall = telemetry.get("parent_wall_time_seconds")
        child_cpu = telemetry.get("aggregate_child_cpu_seconds")
        capacity = telemetry.get("requested_cpu_capacity_seconds")
        utilization = telemetry.get("aggregate_cpu_utilization_fraction")
        child_wall = telemetry.get("sum_child_wall_time_seconds")
        logical_child_capacity = telemetry.get("logical_child_cpu_capacity_seconds")
        logical_child_utilization = telemetry.get(
            "logical_child_cpu_utilization_fraction"
        )
        reported_count = telemetry.get("reported_child_cpu_available_count")
        reported_cpu = telemetry.get("reported_child_process_tree_cpu_seconds")
        reported_capacity = telemetry.get("reported_child_cpu_capacity_seconds")
        reported_utilization = telemetry.get("reported_child_cpu_utilization_fraction")
        reaped_reported_delta = telemetry.get("reaped_minus_reported_child_cpu_seconds")
        idle_capacity = telemetry.get("scheduler_envelope_idle_capacity_seconds")
        stagger = telemetry.get("configured_model_load_stagger_seconds")
        dispatch_fill = telemetry.get("dispatch_fill_seconds")
        planned_dispatch_fill = telemetry.get("maximum_planned_dispatch_fill_seconds")
        available = telemetry.get("available")
        if (
            telemetry.get("schema_version")
            != "mft-tier1-final1000-phase-b-cpu-telemetry-v1"
            or not isinstance(available, bool)
            or telemetry.get("parent_requested_cpus") != concurrency * CHILD_CPUS
            or telemetry.get("logical_child_cpus") != CHILD_CPUS
            or telemetry.get("logical_child_count") != concurrency
            or isinstance(wall, bool)
            or not isinstance(wall, (int, float))
            or not math.isfinite(float(wall))
            or float(wall) < 0
            or isinstance(capacity, bool)
            or not isinstance(capacity, (int, float))
            or not math.isclose(
                float(capacity),
                float(wall) * concurrency * CHILD_CPUS,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or isinstance(child_wall, bool)
            or not isinstance(child_wall, (int, float))
            or not math.isfinite(float(child_wall))
            or float(child_wall) < 0
            or isinstance(logical_child_capacity, bool)
            or not isinstance(logical_child_capacity, (int, float))
            or not math.isclose(
                float(logical_child_capacity),
                float(child_wall) * CHILD_CPUS,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or isinstance(reported_count, bool)
            or not isinstance(reported_count, int)
            or not 0 <= reported_count <= finished
            or isinstance(reported_cpu, bool)
            or not isinstance(reported_cpu, (int, float))
            or not math.isfinite(float(reported_cpu))
            or float(reported_cpu) < 0
            or isinstance(reported_capacity, bool)
            or not isinstance(reported_capacity, (int, float))
            or not math.isfinite(float(reported_capacity))
            or float(reported_capacity) < 0
            or isinstance(reported_utilization, bool)
            or not isinstance(reported_utilization, (int, float))
            or not math.isfinite(float(reported_utilization))
            or float(reported_utilization) < 0
            or (
                float(reported_capacity) > 0
                and not math.isclose(
                    float(reported_utilization),
                    float(reported_cpu) / float(reported_capacity),
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            )
            or (float(reported_capacity) == 0 and float(reported_utilization) != 0)
            or isinstance(idle_capacity, bool)
            or not isinstance(idle_capacity, (int, float))
            or not math.isclose(
                float(idle_capacity),
                max(0.0, float(capacity) - float(logical_child_capacity)),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or isinstance(stagger, bool)
            or not isinstance(stagger, (int, float))
            or not MIN_MODEL_LOAD_STAGGER_SECONDS
            <= float(stagger)
            <= MAX_MODEL_LOAD_STAGGER_SECONDS
            or isinstance(dispatch_fill, bool)
            or not isinstance(dispatch_fill, (int, float))
            or not math.isfinite(float(dispatch_fill))
            or float(dispatch_fill) < 0
            or isinstance(planned_dispatch_fill, bool)
            or not isinstance(planned_dispatch_fill, (int, float))
            or not math.isclose(
                float(planned_dispatch_fill),
                (concurrency - 1) * float(stagger),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or telemetry.get("scheduler_ready_lane_reserve") != 0
            or telemetry.get("scheduler_admission_policy")
            != "natural-terminal-vacancy-only-v1"
            or telemetry.get("future_comparison_child_cpu_counts") != [1, 2, 4]
            or telemetry.get("automatic_child_cpu_reduction_allowed") is not False
            or (
                reaped_reported_delta is not None
                and (
                    reported_count != concurrency
                    or not available
                    or isinstance(reaped_reported_delta, bool)
                    or not isinstance(reaped_reported_delta, (int, float))
                    or not math.isfinite(float(reaped_reported_delta))
                    or not math.isclose(
                        float(reaped_reported_delta),
                        float(child_cpu) - float(reported_cpu),
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    )
                )
            )
            or (
                reaped_reported_delta is None
                and available
                and reported_count == concurrency
            )
            or (
                available
                and (
                    telemetry.get("measurement")
                    != "resource.getrusage(RUSAGE_CHILDREN)-delta"
                    or isinstance(child_cpu, bool)
                    or not isinstance(child_cpu, (int, float))
                    or not math.isfinite(float(child_cpu))
                    or float(child_cpu) < 0
                    or isinstance(utilization, bool)
                    or not isinstance(utilization, (int, float))
                    or not math.isfinite(float(utilization))
                    or float(utilization) < 0
                    or isinstance(logical_child_utilization, bool)
                    or not isinstance(logical_child_utilization, (int, float))
                    or not math.isfinite(float(logical_child_utilization))
                    or float(logical_child_utilization) < 0
                    or (
                        float(logical_child_capacity) > 0
                        and not math.isclose(
                            float(logical_child_utilization),
                            float(child_cpu) / float(logical_child_capacity),
                            rel_tol=1e-9,
                            abs_tol=1e-9,
                        )
                    )
                )
            )
            or (
                not available
                and (
                    telemetry.get("measurement") != "unavailable-on-platform"
                    or child_cpu is not None
                    or utilization is not None
                    or logical_child_utilization is not None
                )
            )
        ):
            raise RuntimeError("Phase B terminal CPU telemetry drifted")
    ordinals: set[int] = set()
    cpus: set[int] = set()
    for child in active:
        if not isinstance(child, dict) or set(child) != {
            "ordinal",
            "seed",
            "pid",
            "cpu_set",
            "launched_at",
        }:
            raise RuntimeError("Phase B active-child journal drifted")
        ordinal = _integer(child.get("ordinal"), "active ordinal")
        _integer(child.get("seed"), "active seed")
        _integer(child.get("pid"), "active pid", minimum=1)
        cpu_set = child.get("cpu_set")
        if (
            ordinal >= launched
            or ordinal in ordinals
            or not isinstance(cpu_set, list)
            or len(cpu_set) != CHILD_CPUS
            or len(set(cpu_set)) != CHILD_CPUS
            or any(_integer(cpu, "active CPU") in cpus for cpu in cpu_set)
            or not str(child.get("launched_at") or "")
        ):
            raise RuntimeError("Phase B active-child isolation drifted")
        ordinals.add(ordinal)
        cpus.update(int(cpu) for cpu in cpu_set)
    return copy.deepcopy(dict(value))


def seal_child_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if key != "receipt_sha256"
    }
    return validate_child_receipt(
        {**unsigned, "receipt_sha256": phase_a.canonical_sha256(unsigned)}
    )


def validate_child_resource_telemetry(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schema_version",
        "available",
        "measurement",
        "child_cpus",
        "wall_time_seconds",
        "process_tree_cpu_seconds",
        "cpu_capacity_seconds",
        "cpu_utilization_fraction",
    }
    available = value.get("available")
    wall = value.get("wall_time_seconds")
    cpu = value.get("process_tree_cpu_seconds")
    capacity = value.get("cpu_capacity_seconds")
    utilization = value.get("cpu_utilization_fraction")
    if (
        set(value) != required
        or value.get("schema_version") != CHILD_RESOURCE_TELEMETRY_SCHEMA
        or not isinstance(available, bool)
        or value.get("child_cpus") != CHILD_CPUS
        or isinstance(wall, bool)
        or not isinstance(wall, (int, float))
        or not math.isfinite(float(wall))
        or float(wall) < 0
        or isinstance(capacity, bool)
        or not isinstance(capacity, (int, float))
        or not math.isclose(
            float(capacity),
            float(wall) * CHILD_CPUS,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        or (
            available
            and (
                value.get("measurement") != "resource.getrusage(self+children)-delta"
                or isinstance(cpu, bool)
                or not isinstance(cpu, (int, float))
                or not math.isfinite(float(cpu))
                or float(cpu) < 0
                or isinstance(utilization, bool)
                or not isinstance(utilization, (int, float))
                or not math.isfinite(float(utilization))
                or float(utilization) < 0
                or (
                    float(capacity) > 0
                    and not math.isclose(
                        float(utilization),
                        float(cpu) / float(capacity),
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    )
                )
            )
        )
        or (
            not available
            and (
                value.get("measurement") != "unavailable-on-platform"
                or cpu is not None
                or utilization is not None
            )
        )
    ):
        raise RuntimeError("Phase B child CPU telemetry drifted")
    return copy.deepcopy(dict(value))


def validate_child_receipt(
    value: Mapping[str, Any], *, manifest: Mapping[str, Any] | None = None
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
        "child_resource_telemetry",
        "failure",
        "cpu_set",
        "child_cpus",
        "child_memory_mb",
        "runtime_scratch_relative_path",
        "runtime_scratch_cleanup_performed",
        "shared_tmp_deleted",
        "production_eligible",
        "fea_submission_performed",
        "aedt_used",
        "receipt_sha256",
    }
    state = str(value.get("state") or "")
    legacy = value.get("legacy_status")
    result_sha = value.get("result_sha256")
    exit_code = value.get("exit_code")
    wall = value.get("wall_time_seconds")
    cpu_set = value.get("cpu_set")
    child_resource_telemetry = value.get("child_resource_telemetry")
    if (
        set(value) != required
        or value.get("schema_version") != CHILD_RECEIPT_SCHEMA
        or value.get("protocol_version") != PROTOCOL_VERSION
        or state not in phase_a.TERMINAL_CHILD_STATES
        or value.get("terminal") is not True
        or not isinstance(value.get("lane_fatal"), bool)
        or isinstance(exit_code, bool)
        or not isinstance(exit_code, int)
        or isinstance(wall, bool)
        or not isinstance(wall, (int, float))
        or not math.isfinite(float(wall))
        or float(wall) < 0
        or not isinstance(child_resource_telemetry, dict)
        or not isinstance(cpu_set, list)
        or len(cpu_set) != CHILD_CPUS
        or len(set(cpu_set)) != CHILD_CPUS
        or value.get("child_cpus") != CHILD_CPUS
        or value.get("child_memory_mb") != CHILD_MEMORY_MB
        or value.get("runtime_scratch_relative_path")
        != f"seed-{value.get('seed')}/runtime"
        or value.get("runtime_scratch_cleanup_performed") is not True
        or value.get("shared_tmp_deleted") is not False
        or not isinstance(legacy, dict)
        or value.get("legacy_status_sha256") != phase_a.canonical_sha256(legacy)
        or not _is_sha256(value.get("manifest_sha256"))
        or not _is_sha256(value.get("payload_sha256"))
        or (result_sha is not None and not _is_sha256(result_sha))
        or (state == "completed") != (result_sha is not None)
        or value.get("receipt_sha256") != phase_a.canonical_sha256(unsigned)
        or value.get("production_eligible") is not False
        or value.get("fea_submission_performed") is not False
        or value.get("aedt_used") is not False
    ):
        raise RuntimeError("Phase B child terminal receipt seal mismatch")
    validate_child_resource_telemetry(child_resource_telemetry)
    ordinal = _integer(value.get("ordinal"), "receipt ordinal")
    seed = _integer(value.get("seed"), "receipt seed")
    if manifest is not None:
        manifest = validate_batch_manifest(manifest)
        children = manifest["ordered_children"]
        if (
            value.get("manifest_sha256") != manifest["manifest_sha256"]
            or ordinal >= len(children)
            or children[ordinal].get("seed") != seed
            or children[ordinal].get("payload_sha256") != value.get("payload_sha256")
            or children[ordinal].get("logical_dedupe_key")
            != value.get("logical_dedupe_key")
        ):
            raise RuntimeError("Phase B child receipt/manifest binding mismatch")
    return copy.deepcopy(dict(value))
