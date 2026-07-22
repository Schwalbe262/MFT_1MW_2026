"""Fail-closed 2-CPU, capacity-aware successor for the live Final1000 lane.

The module is intentionally inert unless ``run --apply`` is selected.  It
imports a *stopped* authenticated 4-CPU controller state, retains every
existing task until natural terminal completion, and renders only new refill
tasks with the sealed 2-CPU/28,672-MiB/thread=2 policy.  Capacity is recomputed
from active, non-exclusive CPU allocations on every cycle.  No cancellation,
preemption, allocation, AEDT, FEA, service, or database mutation is exposed.

One completed promotion-eligible resource2 canary (task 86501) is a mandatory
precondition for state preparation and every subsequent refill cycle.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

try:
    from tier1_corrected_current7_receipt import canonical_sha256
    from tier1_corrected_current7_slurm_controller import TransportReadyProbe
    from tier1_corrected_current7_slurm_publish import scheduler_publication_transport
    from tier1_final1000_resource2_canary import (
        REMOTE_TERMINAL_SCHEMA,
        validate_terminal_evidence,
    )
    from tier1_final1000_multiseed_contract import scheduler_task_identity_matches
    from tier1_final1000_rolling_migration import (
        RESOURCE2_POLICY,
        RESOURCE2_RESOURCE_POLICY_ID,
        SUCCESSOR_POLICY,
        SUCCESSOR_RESOURCE_POLICY_ID,
        _render_from_template_for_policy,
        _validate_task_for_policy,
        chained_harvest_cohort_identity,
        validate_previous_successor_plan,
    )
    from tier1_final1000_slurm_controller import (
        ACTIVE_STATES,
        DEDUPE_PREFIX,
        TASK_NAME_PREFIX,
        TERMINAL_STATES,
        SchedulerApiClient,
        _scheduler_state,
        _task_id,
        _task_templates,
        _validate_previous_successor_state,
        _verify_live_ready,
    )
    from tier1_final1000_stage_profiles import BY_ID, STAGES
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_corrected_current7_receipt import canonical_sha256
    from tools.tier1_corrected_current7_slurm_controller import TransportReadyProbe
    from tools.tier1_corrected_current7_slurm_publish import (
        scheduler_publication_transport,
    )
    from tools.tier1_final1000_resource2_canary import (
        REMOTE_TERMINAL_SCHEMA,
        validate_terminal_evidence,
    )
    from tools.tier1_final1000_multiseed_contract import (
        scheduler_task_identity_matches,
    )
    from tools.tier1_final1000_rolling_migration import (
        RESOURCE2_POLICY,
        RESOURCE2_RESOURCE_POLICY_ID,
        SUCCESSOR_POLICY,
        SUCCESSOR_RESOURCE_POLICY_ID,
        _render_from_template_for_policy,
        _validate_task_for_policy,
        chained_harvest_cohort_identity,
        validate_previous_successor_plan,
    )
    from tools.tier1_final1000_slurm_controller import (
        ACTIVE_STATES,
        DEDUPE_PREFIX,
        TASK_NAME_PREFIX,
        TERMINAL_STATES,
        SchedulerApiClient,
        _scheduler_state,
        _task_id,
        _task_templates,
        _validate_previous_successor_state,
        _verify_live_ready,
    )
    from tools.tier1_final1000_stage_profiles import BY_ID, STAGES


CONFIG_SCHEMA = "mft-tier1-final1000-resource2-capacity-rollout-config-v1"
STATE_SCHEMA = "mft-tier1-final1000-resource2-capacity-rollout-state-v1"
RESULT_SCHEMA = "mft-tier1-final1000-resource2-capacity-rollout-result-v1"
CAPACITY_SCHEMA = "mft-tier1-final1000-resource2-capacity-snapshot-v1"
FALLBACK_SCHEMA = "mft-tier1-final1000-resource2-fallback-v1"

REFILL_MODE_RESOURCE2 = "resource2"
REFILL_MODE_RESOURCE4 = "resource4-fallback"
REFILL_MODES = frozenset({REFILL_MODE_RESOURCE2, REFILL_MODE_RESOURCE4})
ENTRY_STAGE_ID = "entry-1200-t125"
STAGE_WEIGHT_BASIS = {
    "entry-1200-t125": 200,
    "bridge-1150-t115": 160,
    "close-1075-t107p5": 90,
    "final-1000-t100": 50,
}
REQUIRED_RESOURCE2_TERMINAL_GATES = frozenset(
    {
        "scheduler_completed",
        "wrapper_and_seed_completed",
        "exact_two_cpu_affinity",
        "exact_two_thread_binding",
        "per_seed_peak_rss_within_gate",
        "bounded_cgroup_diagnostics_sealed",
        "supported_cgroup_memory_hierarchy",
        "finite_cgroup_ancestor_covers_request",
        "single_seed_throughput_retains_75pct",
        "slot_weighted_throughput_improves_10pct",
        "core_use_retains_80pct",
        "manifest_result_validation_passed",
        "baseline_terminal_was_promotion_eligible",
    }
)

DEFAULT_SCHEDULER_URL = "http://127.0.0.1:8002"
DEFAULT_ACCOUNTS = Path(r"Y:\runtime\slurm_scheduler\config\accounts.yaml")
DEFAULT_SCHEDULER_SOURCE = Path(r"C:\Users\peets\NEC\slurm_scheduler")


class CapacityScheduler(Protocol):
    post_count: int

    def list_namespace_tasks(self) -> Sequence[Mapping[str, Any]]: ...

    def list_allocations(self) -> Sequence[Mapping[str, Any]]: ...

    def get_task(self, task_id: int) -> Mapping[str, Any] | None: ...

    def find_task_by_dedupe(self, dedupe_key: str) -> Mapping[str, Any] | None: ...

    def submit_task(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


class CapacitySchedulerApiClient(SchedulerApiClient):
    """Existing namespace client plus one read-only allocation inventory GET."""

    def list_allocations(self) -> list[dict[str, Any]]:
        value = self._request("/api/allocations")
        if not isinstance(value, list) or any(
            not isinstance(row, dict) for row in value
        ):
            raise RuntimeError("Scheduler allocation inventory is not an object list")
        return [copy.deepcopy(row) for row in value]


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} JSON is unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} JSON must be an object: {path}")
    return value


def _json_bytes(value: Mapping[str, Any]) -> bytes:
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


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.is_dir():
        raise RuntimeError(f"state output is a directory: {path}")
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(_json_bytes(value))
            stream.flush()
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_config(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if key != "config_sha256"
    }
    required = {
        "schema_version",
        "predecessor_launch_plan_sha256",
        "predecessor_controller_revision",
        "resource2_canary_task_id",
        "resource2_canary_package_sha256",
        "resource2_canary_submission_receipt_sha256",
        "minimum_active_target",
        "maximum_active_target",
        "stage_weight_basis",
        "resource2_promoted_stage_ids",
        "cross_stage_resource2_expansion_allowed",
        "promotion_basis",
        "resource2_policy",
        "fallback_resource4_policy",
        "capacity_policy",
        "failure_policy",
        "scheduler_mutation_endpoints",
        "cancellation_performed",
        "preemption_performed",
        "fea_submission_performed",
        "aedt_used",
        "config_sha256",
    }
    stage_weights = value.get("stage_weight_basis")
    capacity = value.get("capacity_policy")
    failure = value.get("failure_policy")
    if (
        set(value) != required
        or value.get("schema_version") != CONFIG_SCHEMA
        or value.get("config_sha256") != canonical_sha256(unsigned)
        or not _is_sha256(value.get("predecessor_launch_plan_sha256"))
        or value.get("predecessor_controller_revision")
        != "4aaf99a02f54d134dfe202b2867a73bab1460577"
        or value.get("resource2_canary_task_id") != 86501
        or not _is_sha256(value.get("resource2_canary_package_sha256"))
        or not _is_sha256(value.get("resource2_canary_submission_receipt_sha256"))
        or value.get("minimum_active_target") != 500
        or isinstance(value.get("maximum_active_target"), bool)
        or not isinstance(value.get("maximum_active_target"), int)
        or not 500 <= int(value["maximum_active_target"]) <= 10_000
        or stage_weights != STAGE_WEIGHT_BASIS
        or value.get("resource2_promoted_stage_ids") != [ENTRY_STAGE_ID]
        or value.get("cross_stage_resource2_expansion_allowed") is not False
        or value.get("promotion_basis")
        != {
            "entry-1200-t125": "terminal-gate-task-86501",
            "bridge-1150-t115": "not-promoted-requires-own-2cpu-terminal-gate",
            "close-1075-t107p5": "not-promoted-requires-own-2cpu-terminal-gate",
            "final-1000-t100": "not-promoted-requires-own-2cpu-terminal-gate",
        }
        or value.get("resource2_policy") != RESOURCE2_POLICY
        or value.get("fallback_resource4_policy") != SUCCESSOR_POLICY
        or not isinstance(capacity, dict)
        or capacity
        != {
            "allocation_states_counted": ["active"],
            "resource_pool": "cpu",
            "exclusive_node": False,
            "scheduling_profile": "standard",
            "slot_formula": "exact-bin-pack(2cpu-promoted-stages,4cpu-unpromoted-stages,28672MiB,32-workers)",
            "draining_counted": False,
            "pending_counted": False,
            "warm_counted": False,
            "recompute_each_cycle": True,
            "target_formula": "max(500,min(exact_mixed_fit,maximum_active_target))",
        }
        or failure
        != {
            "resource2_terminal_failure_threshold": 1,
            "fallback_mode": REFILL_MODE_RESOURCE4,
            "fallback_active_target": 500,
            "cancel_existing_tasks": False,
            "retry_resource2_automatically": False,
        }
        or value.get("scheduler_mutation_endpoints") != ["POST /api/tasks"]
        or any(
            value.get(field) is not False
            for field in (
                "cancellation_performed",
                "preemption_performed",
                "fea_submission_performed",
                "aedt_used",
            )
        )
    ):
        raise RuntimeError("resource2 capacity rollout config seal mismatch")
    return copy.deepcopy(dict(value))


def validate_promotion_gate(
    value: Mapping[str, Any], *, config: Mapping[str, Any]
) -> dict[str, Any]:
    unsigned = {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if key != "remote_terminal_sha256"
    }
    terminal = value.get("terminal_evidence")
    if not isinstance(terminal, dict):
        raise RuntimeError("resource2 canary terminal evidence is missing")
    terminal = validate_terminal_evidence(terminal)
    terminal_gates = terminal.get("gate_results")
    if (
        value.get("schema_version") != REMOTE_TERMINAL_SCHEMA
        or value.get("remote_terminal_sha256") != canonical_sha256(unsigned)
        or value.get("task_id") != config["resource2_canary_task_id"]
        or value.get("package_sha256") != config["resource2_canary_package_sha256"]
        or value.get("submission_receipt_sha256")
        != config["resource2_canary_submission_receipt_sha256"]
        or value.get("scheduler_status") != "completed"
        or value.get("terminal_sha256") != terminal["terminal_sha256"]
        or terminal.get("task_id") != value.get("task_id")
        or terminal.get("package_sha256") != value.get("package_sha256")
        or terminal.get("scheduler_status") != value.get("scheduler_status")
        or not isinstance(terminal_gates, dict)
        or set(terminal_gates) != REQUIRED_RESOURCE2_TERMINAL_GATES
        or any(result is not True for result in terminal_gates.values())
        or value.get("promotion_eligible") is not True
        or terminal.get("promotion_eligible") is not True
        or value.get("scheduler_access") != "GET-only"
        or value.get("remote_access") != "read-only"
        or value.get("scheduler_post_count") != 0
        or value.get("scheduler_cancel_count") != 0
        or value.get("scheduler_preempt_count") != 0
        or value.get("publication_count") != 0
        or value.get("remote_write_count") != 0
        or value.get("automatic_promotion_performed") is not False
        or value.get("fea_submission_performed") is not False
        or value.get("aedt_used") is not False
    ):
        raise RuntimeError("resource2 canary promotion gate did not pass")
    return copy.deepcopy(dict(value))


def _allocation_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    allocation_id = row.get("id")
    if (
        isinstance(allocation_id, bool)
        or not isinstance(allocation_id, int)
        or allocation_id <= 0
    ):
        raise RuntimeError("allocation id is not a positive integer")
    state = str(row.get("state") or "")
    pool = str(row.get("resource_pool") or "")
    exclusive = bool(int(row.get("exclusive_node") or 0))
    cpus = row.get("total_cpus")
    memory = row.get("total_memory_mb")
    if isinstance(cpus, bool) or not isinstance(cpus, int) or cpus < 0:
        raise RuntimeError("allocation total CPU value is invalid")
    if isinstance(memory, bool) or not isinstance(memory, int) or memory < 0:
        raise RuntimeError("allocation total memory value is invalid")
    return {
        "allocation_id": allocation_id,
        "slurm_job_id": str(row.get("slurm_job_id") or ""),
        "account_name": str(row.get("account_name") or ""),
        "partition": str(row.get("partition") or ""),
        "node_name": str(row.get("node_name") or ""),
        "state": state,
        "resource_pool": pool,
        "exclusive_node": exclusive,
        "total_cpus": cpus,
        "total_memory_mb": memory,
    }


def scaled_stage_targets(target: int, weights: Mapping[str, int]) -> dict[str, int]:
    """Hamilton apportionment with deterministic stage-order tie breaking."""

    if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
        raise ValueError("active target must be a positive integer")
    stage_ids = [stage.stage_id for stage in STAGES]
    if set(weights) != set(stage_ids):
        raise RuntimeError("stage weight basis is incomplete")
    total = sum(int(weights[stage_id]) for stage_id in stage_ids)
    if total <= 0 or any(int(weights[stage_id]) <= 0 for stage_id in stage_ids):
        raise RuntimeError("stage weight basis must be positive")
    numerators = {stage_id: target * int(weights[stage_id]) for stage_id in stage_ids}
    result = {stage_id: numerators[stage_id] // total for stage_id in stage_ids}
    remainder = target - sum(result.values())
    rank = sorted(
        stage_ids,
        key=lambda stage_id: (
            -(numerators[stage_id] % total),
            stage_ids.index(stage_id),
        ),
    )
    for stage_id in rank[:remainder]:
        result[stage_id] += 1
    if sum(result.values()) != target:
        raise RuntimeError("scaled stage target does not sum to active target")
    return result


def _mixed_capacity_table(
    rows: Sequence[Mapping[str, Any]],
) -> dict[int, int]:
    """Return maximum 2-CPU slots available for every exact 4-CPU count.

    All jobs consume the same 28,672 MiB and one Standard-profile worker.  A
    small dynamic program is therefore an exact integer bin pack over the
    allocation boundaries; summing aggregate CPU would incorrectly permit a
    task to straddle allocations.
    """

    table = {0: 0}
    memory_mb = int(RESOURCE2_POLICY["memory_mb_per_task"])
    max_workers = int(RESOURCE2_POLICY["max_workers_per_node"])
    for row in rows:
        worker_slots = min(
            int(row["total_memory_mb"]) // memory_mb,
            max_workers,
        )
        cpus = int(row["total_cpus"])
        updated: dict[int, int] = {}
        for existing_fours, existing_twos in table.items():
            for four_count in range(min(worker_slots, cpus // 4) + 1):
                two_count = min(
                    worker_slots - four_count,
                    max(0, cpus - 4 * four_count) // 2,
                )
                total_fours = existing_fours + four_count
                updated[total_fours] = max(
                    updated.get(total_fours, -1),
                    existing_twos + two_count,
                )
        table = updated
    return table


def _maximum_mixed_target(
    rows: Sequence[Mapping[str, Any]], *, config: Mapping[str, Any]
) -> tuple[int, dict[str, int]]:
    promoted = set(config["resource2_promoted_stage_ids"])
    table = _mixed_capacity_table(rows)
    maximum = int(config["maximum_active_target"])
    for target in range(maximum, 0, -1):
        stage_targets = scaled_stage_targets(target, config["stage_weight_basis"])
        two_count = sum(
            count for stage_id, count in stage_targets.items() if stage_id in promoted
        )
        four_count = target - two_count
        if table.get(four_count, -1) >= two_count:
            return target, stage_targets
    return 0, {stage.stage_id: 0 for stage in STAGES}


def capacity_snapshot(
    allocations: Sequence[Mapping[str, Any]], *, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Compute the exact Standard-profile mixed 2/4-CPU target.

    The deployed Scheduler applies worker-count node dynamics only to
    ``fea_bursty``.  For this exact ``standard`` shape, fit is allocation-local;
    active, non-exclusive CPU allocations are therefore packed independently.
    Only stages backed by their own terminal gate may receive 2-CPU refills.
    """

    if config["capacity_policy"]["scheduling_profile"] != "standard":
        raise RuntimeError("capacity calculation is sealed to Standard tasks")
    identities = [_allocation_identity(row) for row in allocations]
    ids = [item["allocation_id"] for item in identities]
    if len(ids) != len(set(ids)):
        raise RuntimeError("allocation inventory repeats an allocation id")
    eligible = sorted(
        (
            item
            for item in identities
            if item["state"] == "active"
            and item["resource_pool"] == "cpu"
            and item["exclusive_node"] is False
        ),
        key=lambda item: item["allocation_id"],
    )
    per_allocation: list[dict[str, Any]] = []
    for item in eligible:
        fit2 = min(
            item["total_cpus"] // int(RESOURCE2_POLICY["cpus_per_task"]),
            item["total_memory_mb"] // int(RESOURCE2_POLICY["memory_mb_per_task"]),
            int(RESOURCE2_POLICY["max_workers_per_node"]),
        )
        fit4 = min(
            item["total_cpus"] // int(SUCCESSOR_POLICY["cpus_per_task"]),
            item["total_memory_mb"] // int(SUCCESSOR_POLICY["memory_mb_per_task"]),
            int(SUCCESSOR_POLICY["max_workers_per_node"]),
        )
        per_allocation.append(
            {
                **item,
                "resource2_fit_slots": max(0, fit2),
                "resource4_fit_slots": max(0, fit4),
            }
        )
    exact_resource2_fit = sum(item["resource2_fit_slots"] for item in per_allocation)
    exact_resource4_fit = sum(item["resource4_fit_slots"] for item in per_allocation)
    exact_mixed_fit, exact_mixed_targets = _maximum_mixed_target(
        per_allocation, config=config
    )
    target = max(
        int(config["minimum_active_target"]),
        min(exact_mixed_fit, int(config["maximum_active_target"])),
    )
    targets = scaled_stage_targets(target, config["stage_weight_basis"])
    promoted = set(config["resource2_promoted_stage_ids"])
    unsigned = {
        "schema_version": CAPACITY_SCHEMA,
        "config_sha256": config["config_sha256"],
        "eligible_allocation_count": len(per_allocation),
        "eligible_allocation_ids": [item["allocation_id"] for item in per_allocation],
        "eligible_total_cpus": sum(item["total_cpus"] for item in per_allocation),
        "eligible_total_memory_mb": sum(
            item["total_memory_mb"] for item in per_allocation
        ),
        "resource2_exact_fit_slots": exact_resource2_fit,
        "resource4_exact_fit_slots": exact_resource4_fit,
        "mixed_exact_fit_slots": exact_mixed_fit,
        "mixed_exact_fit_stage_targets": exact_mixed_targets,
        "bounded_active_target": target,
        "stage_active_targets": targets,
        "stage_refill_policy_ids": {
            stage.stage_id: (
                RESOURCE2_RESOURCE_POLICY_ID
                if stage.stage_id in promoted
                else SUCCESSOR_RESOURCE_POLICY_ID
            )
            for stage in STAGES
        },
        "minimum_target_exceeds_exact_fit": target > exact_mixed_fit,
        "per_allocation": per_allocation,
        "draining_allocation_counted": False,
        "pending_allocation_counted": False,
        "warm_allocation_counted": False,
        "standard_profile_allocation_local_fit": True,
    }
    return {**unsigned, "capacity_sha256": canonical_sha256(unsigned)}


def validate_capacity_snapshot(
    value: Mapping[str, Any], *, config: Mapping[str, Any]
) -> dict[str, Any]:
    rows = value.get("per_allocation")
    if not isinstance(rows, list):
        raise RuntimeError("capacity snapshot has no allocation rows")
    rebuilt = capacity_snapshot(
        [
            {
                "id": row.get("allocation_id"),
                "slurm_job_id": row.get("slurm_job_id"),
                "account_name": row.get("account_name"),
                "partition": row.get("partition"),
                "node_name": row.get("node_name"),
                "state": row.get("state"),
                "resource_pool": row.get("resource_pool"),
                "exclusive_node": int(bool(row.get("exclusive_node"))),
                "total_cpus": row.get("total_cpus"),
                "total_memory_mb": row.get("total_memory_mb"),
            }
            for row in rows
        ],
        config=config,
    )
    if rebuilt != value:
        raise RuntimeError("capacity snapshot does not reproduce")
    return copy.deepcopy(dict(value))


def _seal_nested(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        key: copy.deepcopy(item) for key, item in value.items() if key != "sha256"
    }
    return {**unsigned, "sha256": canonical_sha256(unsigned)}


def _seal_state(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        key: copy.deepcopy(item) for key, item in value.items() if key != "state_sha256"
    }
    migration = unsigned.get("rolling_migration")
    if isinstance(migration, dict):
        unsigned["rolling_migration"] = _seal_nested(migration)
    return {**unsigned, "state_sha256": canonical_sha256(unsigned)}


def _advance_state(state: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(state))
    value["parent_state_sha256"] = value.get("state_sha256")
    value["revision"] = int(value["revision"]) + 1
    return _seal_state(value)


def _advance_state_in_place(state: dict[str, Any]) -> None:
    """Advance a working ledger without invalidating reserved-entry handles."""

    state["parent_state_sha256"] = state.get("state_sha256")
    state["revision"] = int(state["revision"]) + 1
    migration = state.get("rolling_migration")
    if isinstance(migration, Mapping):
        state["rolling_migration"] = _seal_nested(migration)
    unsigned = {key: item for key, item in state.items() if key != "state_sha256"}
    state["state_sha256"] = canonical_sha256(unsigned)


def _successor_cohort_id(state: Mapping[str, Any]) -> str:
    migration = state.get("rolling_migration")
    value = (
        migration.get("successor_harvest_cohort_id")
        if isinstance(migration, Mapping)
        else None
    )
    if not isinstance(value, str) or not value:
        raise RuntimeError("resource2 rollout has no successor harvest cohort")
    return value


def _rollout_metadata(state: Mapping[str, Any]) -> Mapping[str, Any]:
    value = state.get("resource2_rollout")
    if not isinstance(value, Mapping):
        raise RuntimeError("resource2 rollout metadata is absent")
    return value


def _entry_identities(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "stage_id",
        "seed",
        "wave",
        "bundle_id",
        "dedupe_key",
        "task_id",
        "origin",
        "resource_policy_id",
        "harvest_cohort_id",
    )
    return [
        {field: copy.deepcopy(entry.get(field)) for field in fields}
        for entry in entries
    ]


def prepare_state(
    predecessor_state: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
    terminal_gate: Mapping[str, Any],
    allocations: Sequence[Mapping[str, Any]],
    require_predecessor_stopped: bool,
) -> dict[str, Any]:
    plan = validate_previous_successor_plan(plan)
    predecessor = _validate_previous_successor_state(predecessor_state, plan)
    terminal = validate_promotion_gate(terminal_gate, config=config)
    if plan["launch_plan_sha256"] != config["predecessor_launch_plan_sha256"]:
        raise RuntimeError("rollout config/predecessor launch plan mismatch")
    if require_predecessor_stopped and predecessor.get("stop_requested") is not True:
        raise RuntimeError(
            "apply preparation requires a stopped predecessor controller"
        )
    capacity = capacity_snapshot(allocations, config=config)
    migration = copy.deepcopy(predecessor["rolling_migration"])
    cohort_id = str(migration["successor_harvest_cohort_id"])
    cohorts = copy.deepcopy(migration["harvest_cohorts"])
    cohorts[cohort_id] = chained_harvest_cohort_identity(
        cohort_id=cohort_id,
        plan=plan,
        resource_policy_ids=[
            SUCCESSOR_RESOURCE_POLICY_ID,
            RESOURCE2_RESOURCE_POLICY_ID,
        ],
    )
    migration["harvest_cohorts"] = cohorts
    migration = _seal_nested(migration)
    prefix = copy.deepcopy(predecessor["entries"])
    if any(entry.get("state") == "planned" for entry in prefix):
        raise RuntimeError(
            "resource2 handoff requires predecessor reservations to be reconciled"
        )
    rollout = {
        "config_sha256": config["config_sha256"],
        "resource2_terminal_gate_sha256": terminal["remote_terminal_sha256"],
        "predecessor_state_sha256": predecessor["state_sha256"],
        "predecessor_entry_count": len(prefix),
        "predecessor_entry_identities_sha256": canonical_sha256(
            _entry_identities(prefix)
        ),
        "predecessor_next_seed_by_stage": copy.deepcopy(
            predecessor["next_seed_by_stage"]
        ),
        "refill_mode": REFILL_MODE_RESOURCE2,
        "capacity": capacity,
        "fallback": None,
        "resource2_terminal_failure_task_ids": [],
        "resource2_refill_count": 0,
        "resource4_refill_count": 0,
        "cancellation_performed": False,
        "preemption_performed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    value = {
        **copy.deepcopy(predecessor),
        "schema_version": STATE_SCHEMA,
        "stop_requested": False,
        "parent_state_sha256": predecessor["state_sha256"],
        "rolling_migration": migration,
        "resource2_rollout": rollout,
    }
    return validate_state(
        _seal_state(value),
        plan=plan,
        config=config,
        terminal_gate=terminal,
    )


def _policy_for_entry(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    policy_id = str(entry.get("resource_policy_id") or "")
    if policy_id == RESOURCE2_RESOURCE_POLICY_ID:
        return RESOURCE2_POLICY
    if policy_id == SUCCESSOR_RESOURCE_POLICY_ID:
        return SUCCESSOR_POLICY
    raise RuntimeError("rollout refill entry has an unsupported resource policy")


def _task_for_rollout_entry(
    entry: Mapping[str, Any], *, templates: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    policy = _policy_for_entry(entry)
    task = _render_from_template_for_policy(
        templates[str(entry["stage_id"])],
        stage_id=str(entry["stage_id"]),
        seed=int(entry["seed"]),
        wave=str(entry["wave"]),
        policy=policy,
    )
    _validate_task_for_policy(task, policy=policy)
    if task["dedupe_key"] != entry["dedupe_key"]:
        raise RuntimeError("rollout entry cannot reproduce its task dedupe")
    return task


def validate_state(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
    terminal_gate: Mapping[str, Any],
) -> dict[str, Any]:
    plan = validate_previous_successor_plan(plan)
    config = validate_config(config)
    terminal = validate_promotion_gate(terminal_gate, config=config)
    unsigned = {
        key: copy.deepcopy(item) for key, item in value.items() if key != "state_sha256"
    }
    migration = value.get("rolling_migration")
    rollout = value.get("resource2_rollout")
    entries = value.get("entries")
    next_seeds = value.get("next_seed_by_stage")
    if (
        value.get("schema_version") != STATE_SCHEMA
        or value.get("launch_plan_sha256") != plan["launch_plan_sha256"]
        or value.get("state_sha256") != canonical_sha256(unsigned)
        or value.get("scheduler_name_prefix") != TASK_NAME_PREFIX
        or value.get("scheduler_dedupe_prefix") != DEDUPE_PREFIX
        or value.get("scheduler_mutation_endpoints") != ["POST /api/tasks"]
        or value.get("ramp_released") is not True
        or set(value.get("canary_passed_stage_ids") or []) != set(BY_ID)
        or value.get("stop_requested") not in {True, False}
        or isinstance(value.get("revision"), bool)
        or not isinstance(value.get("revision"), int)
        or int(value["revision"]) < 0
        or not isinstance(entries, list)
        or not isinstance(next_seeds, dict)
        or set(next_seeds) != set(BY_ID)
        or not isinstance(migration, dict)
        or migration.get("sha256")
        != canonical_sha256(
            {key: item for key, item in migration.items() if key != "sha256"}
        )
        or not isinstance(rollout, dict)
        or rollout.get("config_sha256") != config["config_sha256"]
        or rollout.get("resource2_terminal_gate_sha256")
        != terminal["remote_terminal_sha256"]
        or rollout.get("refill_mode") not in REFILL_MODES
        or any(
            rollout.get(field) is not False
            for field in (
                "cancellation_performed",
                "preemption_performed",
                "fea_submission_performed",
                "aedt_used",
            )
        )
    ):
        raise RuntimeError("resource2 capacity rollout state seal mismatch")
    prefix_count = rollout.get("predecessor_entry_count")
    if (
        isinstance(prefix_count, bool)
        or not isinstance(prefix_count, int)
        or not 0 <= prefix_count <= len(entries)
        or canonical_sha256(_entry_identities(entries[:prefix_count]))
        != rollout.get("predecessor_entry_identities_sha256")
        or not _is_sha256(rollout.get("predecessor_state_sha256"))
        or not isinstance(rollout.get("predecessor_next_seed_by_stage"), dict)
        or set(rollout["predecessor_next_seed_by_stage"]) != set(BY_ID)
    ):
        raise RuntimeError("resource2 predecessor ledger prefix changed")
    planned_indexes = [
        index for index, entry in enumerate(entries) if entry.get("state") == "planned"
    ]
    if (
        len(planned_indexes) > 1
        or any(index < prefix_count for index in planned_indexes)
        or (planned_indexes and planned_indexes[0] != len(entries) - 1)
    ):
        raise RuntimeError("resource2 rollout has an unsafe pending reservation set")
    validate_capacity_snapshot(rollout.get("capacity") or {}, config=config)
    if rollout["refill_mode"] == REFILL_MODE_RESOURCE2:
        if rollout.get("fallback") is not None:
            raise RuntimeError("resource2 mode unexpectedly contains fallback evidence")
    else:
        fallback = rollout.get("fallback")
        if not isinstance(fallback, dict):
            raise RuntimeError("resource4 fallback mode lacks evidence")
        fallback_unsigned = {
            key: item for key, item in fallback.items() if key != "sha256"
        }
        if (
            fallback.get("schema_version") != FALLBACK_SCHEMA
            or fallback.get("sha256") != canonical_sha256(fallback_unsigned)
            or fallback.get("active_target") != 500
            or fallback.get("cancelled_task_ids") != []
            or not set(fallback.get("trigger_task_ids") or []).issubset(
                set(rollout.get("resource2_terminal_failure_task_ids") or [])
            )
        ):
            raise RuntimeError("resource4 fallback evidence seal mismatch")
    templates = _task_templates(plan)
    successor_cohort = _successor_cohort_id(value)
    dedupes: set[str] = set()
    seeds: set[tuple[str, int]] = set()
    max_seed = {
        stage.stage_id: int(rollout["predecessor_next_seed_by_stage"][stage.stage_id])
        - 1
        for stage in STAGES
    }
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise RuntimeError("rollout state contains a non-object entry")
        stage_id = str(entry.get("stage_id") or "")
        seed = int(entry.get("seed", -1))
        identity = (stage_id, seed)
        dedupe = str(entry.get("dedupe_key") or "")
        if (
            stage_id not in BY_ID
            or entry.get("state") not in ACTIVE_STATES | TERMINAL_STATES
            or dedupe in dedupes
            or identity in seeds
        ):
            raise RuntimeError("rollout state entry identity is invalid or repeated")
        dedupes.add(dedupe)
        seeds.add(identity)
        if index >= prefix_count:
            if (
                str(entry.get("origin") or "") != "successor"
                or str(entry.get("harvest_cohort_id") or "") != successor_cohort
                or str(entry.get("resource_policy_id") or "")
                not in {RESOURCE2_RESOURCE_POLICY_ID, SUCCESSOR_RESOURCE_POLICY_ID}
                or entry.get("wave") != "refill"
            ):
                raise RuntimeError("rollout refill ledger metadata drifted")
            task = _task_for_rollout_entry(entry, templates=templates)
            if entry.get("bundle_id") != task["payload_json"]["bundle_id"]:
                raise RuntimeError("rollout refill bundle identity drifted")
            if entry.get(
                "resource_policy_id"
            ) == RESOURCE2_RESOURCE_POLICY_ID and stage_id not in set(
                config["resource2_promoted_stage_ids"]
            ):
                raise RuntimeError("ungated stage received a resource2 refill policy")
            max_seed[stage_id] = max(max_seed[stage_id], seed)
    for stage in STAGES:
        expected_next = max_seed[stage.stage_id] + 1
        if int(next_seeds[stage.stage_id]) != expected_next:
            raise RuntimeError("rollout next-seed cursor is not contiguous")
    resource2_count = sum(
        entry.get("resource_policy_id") == RESOURCE2_RESOURCE_POLICY_ID
        for entry in entries[prefix_count:]
    )
    resource4_count = sum(
        entry.get("resource_policy_id") == SUCCESSOR_RESOURCE_POLICY_ID
        for entry in entries[prefix_count:]
    )
    if (
        rollout.get("resource2_refill_count") != resource2_count
        or rollout.get("resource4_refill_count") != resource4_count
    ):
        raise RuntimeError("rollout refill resource counters drifted")
    terminal_resource2_failures = _resource2_terminal_failure_ids(value)
    if terminal_resource2_failures != rollout.get(
        "resource2_terminal_failure_task_ids"
    ) or (
        terminal_resource2_failures and rollout["refill_mode"] != REFILL_MODE_RESOURCE4
    ):
        raise RuntimeError("resource2 terminal failure/fallback evidence drifted")
    cohort = migration.get("harvest_cohorts", {}).get(successor_cohort)
    expected_cohort = chained_harvest_cohort_identity(
        cohort_id=successor_cohort,
        plan=plan,
        resource_policy_ids=[
            SUCCESSOR_RESOURCE_POLICY_ID,
            RESOURCE2_RESOURCE_POLICY_ID,
        ],
    )
    if cohort != expected_cohort:
        raise RuntimeError("resource2 harvest cohort identity drifted")
    return copy.deepcopy(dict(value))


def _active_count(state: Mapping[str, Any], stage_id: str | None = None) -> int:
    return sum(
        entry.get("state") in ACTIVE_STATES
        and (stage_id is None or entry.get("stage_id") == stage_id)
        for entry in state["entries"]
    )


def _resource2_terminal_failure_ids(state: Mapping[str, Any]) -> list[int]:
    prefix_count = int(state["resource2_rollout"]["predecessor_entry_count"])
    return _resource2_failure_ids_for_entries(state["entries"][prefix_count:])


def _resource2_failure_ids_for_entries(
    entries: Sequence[Mapping[str, Any]],
) -> list[int]:
    return sorted(
        int(entry["task_id"])
        for entry in entries
        if entry.get("resource_policy_id") == RESOURCE2_RESOURCE_POLICY_ID
        and entry.get("state") in TERMINAL_STATES - {"completed"}
        and entry.get("task_id") is not None
    )


def _weighted_refill_order(
    state: dict[str, Any], *, slots: int, targets: Mapping[str, int]
) -> list[str]:
    if slots < 0:
        raise ValueError("refill slots cannot be negative")
    stage_ids = [stage.stage_id for stage in STAGES]
    if (
        set(targets) != set(stage_ids)
        or sum(int(targets[key]) for key in stage_ids) <= 0
    ):
        raise RuntimeError("capacity stage targets are invalid")
    current = {stage_id: _active_count(state, stage_id) for stage_id in stage_ids}
    credits = {
        stage_id: float(state["refill_deficit_credit_by_stage"][stage_id])
        for stage_id in stage_ids
    }
    cursor = int(state["refill_stage_cursor"])
    selected = []
    while slots > 0:
        eligible = [
            stage_id for stage_id in stage_ids if current[stage_id] < targets[stage_id]
        ]
        if not eligible:
            raise RuntimeError("no capacity-aware stage deficit exists")
        for stage_id in eligible:
            credits[stage_id] += int(targets[stage_id])
        total_weight = sum(int(targets[stage_id]) for stage_id in eligible)
        cyclic = {
            stage_ids[(cursor + offset) % len(stage_ids)]: offset
            for offset in range(len(stage_ids))
        }
        stage_id = min(eligible, key=lambda item: (-credits[item], cyclic[item]))
        credits[stage_id] -= total_weight
        selected.append(stage_id)
        current[stage_id] += 1
        slots -= 1
        cursor = (stage_ids.index(stage_id) + 1) % len(stage_ids)
    state["refill_stage_cursor"] = cursor
    state["refill_deficit_credit_by_stage"] = credits
    return selected


def _append_refill(
    state: dict[str, Any],
    *,
    templates: Mapping[str, Mapping[str, Any]],
    stage_id: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    rollout = state["resource2_rollout"]
    mode = rollout["refill_mode"]
    resource2_stage = mode == REFILL_MODE_RESOURCE2 and stage_id in set(
        config["resource2_promoted_stage_ids"]
    )
    policy_id = (
        RESOURCE2_RESOURCE_POLICY_ID
        if resource2_stage
        else SUCCESSOR_RESOURCE_POLICY_ID
    )
    policy = RESOURCE2_POLICY if resource2_stage else SUCCESSOR_POLICY
    seed = int(state["next_seed_by_stage"][stage_id])
    stage = BY_ID[stage_id]
    if not stage.seed_start <= seed < stage.seed_window_end_exclusive:
        raise RuntimeError(f"{stage_id} seed window exhausted")
    task = _render_from_template_for_policy(
        templates[stage_id],
        stage_id=stage_id,
        seed=seed,
        wave="refill",
        policy=policy,
    )
    entry = {
        "stage_id": stage_id,
        "seed": seed,
        "wave": "refill",
        "bundle_id": task["payload_json"]["bundle_id"],
        "dedupe_key": task["dedupe_key"],
        "task_id": None,
        "state": "planned",
        "origin": "successor",
        "resource_policy_id": policy_id,
        "harvest_cohort_id": _successor_cohort_id(state),
    }
    state["entries"].append(entry)
    state["next_seed_by_stage"][stage_id] = seed + 1
    counter = "resource2_refill_count" if resource2_stage else "resource4_refill_count"
    rollout[counter] = int(rollout[counter]) + 1
    return entry


def _reconcile(
    state: dict[str, Any], scheduler: CapacityScheduler
) -> tuple[int, list[int]]:
    active = [
        entry
        for entry in state["entries"]
        if entry.get("task_id") is not None
        and entry.get("state") not in TERMINAL_STATES
    ]
    inventory = scheduler.list_namespace_tasks()
    by_id: dict[int, Mapping[str, Any]] = {}
    for row in inventory:
        task_id = _task_id(row)
        if (
            not str(row.get("name") or "").startswith(TASK_NAME_PREFIX)
            or not str(row.get("dedupe_key") or "").startswith(DEDUPE_PREFIX)
            or task_id in by_id
        ):
            raise RuntimeError(
                "capacity controller inventory escaped or repeated namespace"
            )
        by_id[task_id] = row
    changed = 0
    failures = []
    for entry in active:
        task_id = int(entry["task_id"])
        observed = by_id.get(task_id) or scheduler.get_task(task_id)
        if observed is None:
            continue
        expected_name = (
            f"{BY_ID[entry['stage_id']].task_name_stem}-{entry['wave']}-{entry['seed']}"
        )
        if (
            _task_id(observed) != task_id
            or observed.get("dedupe_key") != entry["dedupe_key"]
            or observed.get("name") != expected_name
        ):
            raise RuntimeError("capacity controller observed changed task identity")
        mapped = _scheduler_state(observed.get("status"))
        if mapped is not None and mapped != entry["state"]:
            entry["state"] = mapped
            changed += 1
            if entry.get(
                "resource_policy_id"
            ) == RESOURCE2_RESOURCE_POLICY_ID and mapped in TERMINAL_STATES - {
                "completed"
            }:
                failures.append(task_id)
    return changed, sorted(set(failures))


def _activate_fallback(
    state: dict[str, Any], *, failure_task_ids: Sequence[int], reason: str
) -> bool:
    rollout = state["resource2_rollout"]
    ids = sorted(
        {
            *(
                int(task_id)
                for task_id in rollout["resource2_terminal_failure_task_ids"]
            ),
            *(int(task_id) for task_id in failure_task_ids),
        }
    )
    if rollout["refill_mode"] == REFILL_MODE_RESOURCE4:
        changed = ids != rollout["resource2_terminal_failure_task_ids"]
        rollout["resource2_terminal_failure_task_ids"] = ids
        return changed
    unsigned = {
        "schema_version": FALLBACK_SCHEMA,
        "activated_at": _now(),
        "reason": reason,
        "trigger_task_ids": ids,
        "active_target": 500,
        "cancelled_task_ids": [],
        "future_refill_policy_id": SUCCESSOR_RESOURCE_POLICY_ID,
    }
    rollout["refill_mode"] = REFILL_MODE_RESOURCE4
    rollout["fallback"] = {**unsigned, "sha256": canonical_sha256(unsigned)}
    rollout["resource2_terminal_failure_task_ids"] = ids
    return True


def _submit_planned(
    state: dict[str, Any],
    *,
    entries: Sequence[dict[str, Any]],
    templates: Mapping[str, Mapping[str, Any]],
    scheduler: CapacityScheduler,
) -> tuple[int, int]:
    planned = [entry for entry in entries if entry["state"] == "planned"]
    if len(planned) > 1:
        raise RuntimeError("capacity controller may submit only one reservation at a time")
    submitted = 0
    reconciled = 0
    for entry in planned:
        task = _task_for_rollout_entry(entry, templates=templates)
        observed = scheduler.find_task_by_dedupe(entry["dedupe_key"])
        if observed is None:
            observed = scheduler.submit_task(task)
            submitted += 1
        else:
            reconciled += 1
        if not scheduler_task_identity_matches(observed, task):
            raise RuntimeError("Scheduler changed capacity refill task identity")
        entry["task_id"] = _task_id(observed)
        entry["state"] = _scheduler_state(observed.get("status")) or "submitted"
    state["scheduler_submit_count"] = int(state["scheduler_submit_count"]) + submitted
    return submitted, reconciled


def _settle_or_release_planned_without_post(
    state: dict[str, Any],
    *,
    templates: Mapping[str, Mapping[str, Any]],
    scheduler: CapacityScheduler,
) -> tuple[int, int]:
    """Resolve the sole durable reservation after fallback without a POST."""

    planned = [entry for entry in state["entries"] if entry["state"] == "planned"]
    if len(planned) > 1:
        raise RuntimeError("fallback encountered multiple pending reservations")
    if not planned:
        return 0, 0
    entry = planned[0]
    task = _task_for_rollout_entry(entry, templates=templates)
    observed = scheduler.find_task_by_dedupe(entry["dedupe_key"])
    if observed is not None:
        if not scheduler_task_identity_matches(observed, task):
            raise RuntimeError("Scheduler changed pending fallback task identity")
        entry["task_id"] = _task_id(observed)
        entry["state"] = _scheduler_state(observed.get("status")) or "submitted"
        return 1, 0

    prefix_count = int(state["resource2_rollout"]["predecessor_entry_count"])
    index = state["entries"].index(entry)
    stage_id = str(entry["stage_id"])
    seed = int(entry["seed"])
    if (
        index < prefix_count
        or index != len(state["entries"]) - 1
        or int(state["next_seed_by_stage"][stage_id]) != seed + 1
        or entry.get("task_id") is not None
    ):
        raise RuntimeError("pending fallback reservation cannot be released safely")
    state["entries"].pop()
    state["next_seed_by_stage"][stage_id] = seed
    counter = (
        "resource2_refill_count"
        if entry["resource_policy_id"] == RESOURCE2_RESOURCE_POLICY_ID
        else "resource4_refill_count"
    )
    state["resource2_rollout"][counter] = (
        int(state["resource2_rollout"][counter]) - 1
    )
    return 0, 1


def control_once(
    *,
    plan: Mapping[str, Any],
    state: Mapping[str, Any],
    config: Mapping[str, Any],
    terminal_gate: Mapping[str, Any],
    allocations: Sequence[Mapping[str, Any]],
    scheduler: CapacityScheduler | None,
    apply: bool,
    persist: Callable[[Mapping[str, Any]], None] | None = None,
    request_stop: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = validate_previous_successor_plan(plan)
    config = validate_config(config)
    terminal = validate_promotion_gate(terminal_gate, config=config)
    working = validate_state(state, plan=plan, config=config, terminal_gate=terminal)
    if apply and (scheduler is None or persist is None):
        raise RuntimeError("apply control requires Scheduler and persistence")
    actions: list[dict[str, Any]] = []
    writes = 0

    def save() -> None:
        nonlocal writes
        _advance_state_in_place(working)
        if apply:
            assert persist is not None
            persist(working)
            writes += 1

    if working["stop_requested"] or request_stop:
        if not working["stop_requested"]:
            working["stop_requested"] = True
            save()
        actions.append({"action": "stop", "cancelled_task_ids": []})
        return working, _result(working, actions, apply, writes, scheduler)

    snapshot = capacity_snapshot(allocations, config=config)
    if snapshot != working["resource2_rollout"]["capacity"]:
        if apply:
            working["resource2_rollout"]["capacity"] = snapshot
            save()
    if not apply:
        target = (
            500
            if working["resource2_rollout"]["refill_mode"] == REFILL_MODE_RESOURCE4
            else snapshot["bounded_active_target"]
        )
        actions.append(
            {
                "action": "dry_run",
                "active_count": _active_count(working),
                "capacity_target": target,
                "would_reserve": max(0, target - _active_count(working)),
            }
        )
        preview = copy.deepcopy(working)
        preview["resource2_rollout"]["capacity"] = snapshot
        preview = _seal_state(preview)
        return working, _result(preview, actions, False, 0, None)

    assert scheduler is not None
    templates = _task_templates(plan)
    changed, failure_ids = _reconcile(working, scheduler)
    fallback_activated = False
    if failure_ids:
        fallback_activated = _activate_fallback(
            working,
            failure_task_ids=failure_ids,
            reason="resource2_terminal_failure",
        )
        pending_reconciled, pending_released = _settle_or_release_planned_without_post(
            working,
            templates=templates,
            scheduler=scheduler,
        )
        all_failure_ids = _resource2_terminal_failure_ids(working)
        fallback_updated = _activate_fallback(
            working,
            failure_task_ids=all_failure_ids,
            reason="resource2_terminal_failure",
        )
        save()
        actions.append(
            {
                "action": "resource2_failure_halt",
                "refill_mode": working["resource2_rollout"]["refill_mode"],
                "capacity_target": 500,
                "detected_failure_task_ids": all_failure_ids,
                "pending_reconciled_without_post": pending_reconciled,
                "pending_released_without_post": pending_released,
                "fallback_activated": fallback_activated or fallback_updated,
                "cancelled_task_ids": [],
            }
        )
        return working, _result(working, actions, True, writes, scheduler)
    if changed:
        save()
    if (
        working["resource2_rollout"]["refill_mode"] == REFILL_MODE_RESOURCE4
        and any(entry["state"] == "planned" for entry in working["entries"])
    ):
        _pending_reconciled, _pending_released = (
            _settle_or_release_planned_without_post(
                working,
                templates=templates,
                scheduler=scheduler,
            )
        )
        remaining_failure_ids = _resource2_terminal_failure_ids(working)
        if remaining_failure_ids:
            _activate_fallback(
                working,
                failure_task_ids=remaining_failure_ids,
                reason="resource4_fallback_pending_reservation_recovery",
            )
        save()
    previously_planned = [
        entry for entry in working["entries"] if entry["state"] == "planned"
    ]
    recovered_submitted, recovered_reconciled = _submit_planned(
        working,
        entries=previously_planned,
        templates=templates,
        scheduler=scheduler,
    )
    recovered_failure_ids = _resource2_failure_ids_for_entries(previously_planned)
    recovered_fallback = False
    if recovered_failure_ids:
        recovered_fallback = _activate_fallback(
            working,
            failure_task_ids=recovered_failure_ids,
            reason="resource2_terminal_failure_during_planned_recovery",
        )
    if recovered_submitted or recovered_reconciled or recovered_fallback:
        save()
    if recovered_failure_ids:
        actions.append(
            {
                "action": "resource2_failure_halt",
                "refill_mode": working["resource2_rollout"]["refill_mode"],
                "capacity_target": 500,
                "detected_failure_task_ids": recovered_failure_ids,
                "pending_reconciled_without_post": 0,
                "pending_released_without_post": 0,
                "fallback_activated": recovered_fallback,
                "recovered_planned_submitted": recovered_submitted,
                "recovered_planned_reconciled": recovered_reconciled,
                "cancelled_task_ids": [],
            }
        )
        return working, _result(working, actions, True, writes, scheduler)
    mode = working["resource2_rollout"]["refill_mode"]
    target = 500 if mode == REFILL_MODE_RESOURCE4 else snapshot["bounded_active_target"]
    targets = (
        scaled_stage_targets(500, config["stage_weight_basis"])
        if mode == REFILL_MODE_RESOURCE4
        else snapshot["stage_active_targets"]
    )
    reserved_count = 0
    submitted = 0
    reconciled = 0
    order: list[str] = []
    immediate_failure_ids: list[int] = []
    immediate_fallback = False
    terminal_response_halt = False
    while _active_count(working) < target:
        stage_id = _weighted_refill_order(working, slots=1, targets=targets)[0]
        entry = _append_refill(
            working,
            templates=templates,
            stage_id=stage_id,
            config=config,
        )
        order.append(stage_id)
        reserved_count += 1
        # Persist exactly one reservation before its POST.  A restart can then
        # reconcile that dedupe, while no unposted batch remains behind it.
        save()
        one_submitted, one_reconciled = _submit_planned(
            working,
            entries=[entry],
            templates=templates,
            scheduler=scheduler,
        )
        submitted += one_submitted
        reconciled += one_reconciled
        immediate_failure_ids = _resource2_failure_ids_for_entries([entry])
        if immediate_failure_ids:
            immediate_fallback = _activate_fallback(
                working,
                failure_task_ids=immediate_failure_ids,
                reason="resource2_terminal_failure_during_refill_submission",
            )
        save()
        if immediate_failure_ids:
            # The state transition and fallback are durable before returning;
            # no later reservation or Scheduler POST is allowed in this cycle.
            break
        if entry["state"] in TERMINAL_STATES:
            # Avoid an unbounded same-cycle loop for an immediately terminal
            # non-resource2 response.  The next watch tick may refill again.
            terminal_response_halt = True
            break
    mode = working["resource2_rollout"]["refill_mode"]
    target = 500 if mode == REFILL_MODE_RESOURCE4 else snapshot["bounded_active_target"]
    actions.append(
        {
            "action": "capacity_refill",
            "refill_mode": mode,
            "capacity_target": target,
            "reserved": reserved_count,
            "submitted": submitted,
            "reconciled": reconciled,
            "recovered_planned_submitted": recovered_submitted,
            "recovered_planned_reconciled": recovered_reconciled,
            "stage_order": order,
            "fallback_activated": (
                fallback_activated or recovered_fallback or immediate_fallback
            ),
            "detected_failure_task_ids": immediate_failure_ids,
            "terminal_response_halt": terminal_response_halt,
            "cancelled_task_ids": [],
        }
    )
    if (
        _active_count(working) < target
        and not immediate_failure_ids
        and not terminal_response_halt
    ):
        raise RuntimeError("capacity controller failed to restore its active target")
    return working, _result(working, actions, True, writes, scheduler)


def _result(
    state: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
    apply: bool,
    writes: int,
    scheduler: CapacityScheduler | None,
) -> dict[str, Any]:
    rollout = state["resource2_rollout"]
    snapshot = rollout["capacity"]
    mode = rollout["refill_mode"]
    target = 500 if mode == REFILL_MODE_RESOURCE4 else snapshot["bounded_active_target"]
    return {
        "schema_version": RESULT_SCHEMA,
        "apply": bool(apply),
        "state_revision": state["revision"],
        "state_sha256": state["state_sha256"],
        "state_writes": writes,
        "scheduler_post_count": int(getattr(scheduler, "post_count", 0)),
        "refill_mode": mode,
        "logical_active_minimum": 500,
        "logical_active_target": target,
        "active_count": _active_count(state),
        "active_count_by_stage": {
            stage.stage_id: _active_count(state, stage.stage_id) for stage in STAGES
        },
        "eligible_allocation_count": snapshot["eligible_allocation_count"],
        "eligible_total_cpus": snapshot["eligible_total_cpus"],
        "resource2_exact_fit_slots": snapshot["resource2_exact_fit_slots"],
        "resource4_exact_fit_slots": snapshot["resource4_exact_fit_slots"],
        "mixed_exact_fit_slots": snapshot["mixed_exact_fit_slots"],
        "stage_refill_policy_ids": (
            {stage.stage_id: SUCCESSOR_RESOURCE_POLICY_ID for stage in STAGES}
            if mode == REFILL_MODE_RESOURCE4
            else snapshot["stage_refill_policy_ids"]
        ),
        "stage_active_targets": (
            scaled_stage_targets(500, STAGE_WEIGHT_BASIS)
            if mode == REFILL_MODE_RESOURCE4
            else snapshot["stage_active_targets"]
        ),
        "actions": [copy.deepcopy(dict(action)) for action in actions],
        "stop_requested": state["stop_requested"],
        "cancellation_performed": False,
        "preemption_performed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
    }


def activate_manual_fallback(
    state: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
    terminal_gate: Mapping[str, Any],
    expected_state_sha256: str,
) -> dict[str, Any]:
    working = validate_state(
        state, plan=plan, config=config, terminal_gate=terminal_gate
    )
    if working["state_sha256"] != expected_state_sha256:
        raise RuntimeError("manual fallback expected-state SHA mismatch")
    if _activate_fallback(
        working, failure_task_ids=(), reason="operator_requested_resource4_fallback"
    ):
        working = _advance_state(working)
    return validate_state(
        working, plan=plan, config=config, terminal_gate=terminal_gate
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    for name in ("prepare", "run", "fallback"):
        command = sub.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--resource2-terminal", type=Path, required=True)
        command.add_argument("--state", type=Path, required=True)
    prepare = sub.choices["prepare"]
    prepare.add_argument("--predecessor-state", type=Path, required=True)
    prepare.add_argument("--allocations", type=Path)
    prepare.add_argument("--scheduler-url", default=DEFAULT_SCHEDULER_URL)
    prepare.add_argument("--apply", action="store_true")
    run = sub.choices["run"]
    run.add_argument("--scheduler-url", default=DEFAULT_SCHEDULER_URL)
    run.add_argument("--accounts", type=Path, default=DEFAULT_ACCOUNTS)
    run.add_argument("--scheduler-source", type=Path, default=DEFAULT_SCHEDULER_SOURCE)
    run.add_argument("--publication-account", default="harry261")
    run.add_argument("--apply", action="store_true")
    run.add_argument("--watch", action="store_true")
    run.add_argument("--poll-seconds", type=float, default=15.0)
    run.add_argument("--stop-file", type=Path)
    fallback = sub.choices["fallback"]
    fallback.add_argument("--expected-state-sha256", required=True)
    fallback.add_argument("--apply", action="store_true")
    return parser


def _load_common(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan = validate_previous_successor_plan(
        _read_json(args.plan.resolve(strict=True), "launch plan")
    )
    config = validate_config(
        _read_json(args.config.resolve(strict=True), "rollout config")
    )
    terminal = validate_promotion_gate(
        _read_json(args.resource2_terminal.resolve(strict=True), "resource2 terminal"),
        config=config,
    )
    return plan, config, terminal


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    plan, config, terminal = _load_common(args)
    if args.operation == "prepare":
        scheduler = CapacitySchedulerApiClient(args.scheduler_url)
        allocations = (
            json.loads(
                args.allocations.resolve(strict=True).read_text(encoding="utf-8")
            )
            if args.allocations is not None
            else scheduler.list_allocations()
        )
        if not isinstance(allocations, list):
            raise RuntimeError("allocation fixture must be a JSON list")
        if args.apply and args.state.exists():
            raise RuntimeError("immutable successor state already exists")
        state = prepare_state(
            _read_json(
                args.predecessor_state.resolve(strict=True), "predecessor state"
            ),
            plan=plan,
            config=config,
            terminal_gate=terminal,
            allocations=allocations,
            require_predecessor_stopped=bool(args.apply),
        )
        if args.apply:
            _atomic_json(args.state, state)
        value = {
            "operation": "prepare",
            "apply": bool(args.apply),
            "scheduler_post_count": 0,
            "state_sha256": state["state_sha256"],
            "active_target": state["resource2_rollout"]["capacity"][
                "bounded_active_target"
            ],
            "eligible_allocation_count": state["resource2_rollout"]["capacity"][
                "eligible_allocation_count"
            ],
        }
    elif args.operation == "fallback":
        state = activate_manual_fallback(
            _read_json(args.state.resolve(strict=True), "rollout state"),
            plan=plan,
            config=config,
            terminal_gate=terminal,
            expected_state_sha256=args.expected_state_sha256,
        )
        if args.apply:
            _atomic_json(args.state, state)
        value = {
            "operation": "fallback",
            "apply": bool(args.apply),
            "scheduler_post_count": 0,
            "refill_mode": state["resource2_rollout"]["refill_mode"],
            "state_sha256": state["state_sha256"],
        }
    else:
        if args.watch and not args.apply:
            raise RuntimeError("--watch requires --apply")
        if args.poll_seconds <= 0:
            raise ValueError("poll seconds must be positive")
        scheduler = CapacitySchedulerApiClient(args.scheduler_url)

        def persist(value: Mapping[str, Any]) -> None:
            _atomic_json(args.state, value)

        def stop_requested() -> bool:
            return bool(args.stop_file is not None and args.stop_file.is_file())

        if args.apply:
            with scheduler_publication_transport(
                accounts_path=args.accounts,
                scheduler_source=args.scheduler_source,
                account_name=args.publication_account,
            ) as transport:
                probe = TransportReadyProbe(transport)
                _verify_live_ready(plan, apply=True, ready_probe=probe)
                while True:
                    state, value = control_once(
                        plan=plan,
                        state=_read_json(
                            args.state.resolve(strict=True), "rollout state"
                        ),
                        config=config,
                        terminal_gate=terminal,
                        allocations=scheduler.list_allocations(),
                        scheduler=scheduler,
                        apply=True,
                        persist=persist,
                        request_stop=stop_requested(),
                    )
                    print(
                        json.dumps(value, sort_keys=True, ensure_ascii=False),
                        flush=True,
                    )
                    if not args.watch or state["stop_requested"]:
                        break
                    time.sleep(args.poll_seconds)
        else:
            state, value = control_once(
                plan=plan,
                state=_read_json(args.state.resolve(strict=True), "rollout state"),
                config=config,
                terminal_gate=terminal,
                allocations=scheduler.list_allocations(),
                scheduler=None,
                apply=False,
            )
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
