"""Fail-closed Phase-B placement and Phase-A authority-handoff contracts.

This module is deliberately pure.  It converts a sealed launch plan and a
canonical allocation-capacity snapshot into exact concurrent Phase-B parent
envelopes, and it defines the state that a future authorized controller must
carry across the Phase-A -> Phase-B handoff.  It has no HTTP client, no
Scheduler mutation method, and no FEA/AEDT surface.

The current production packing target is 500 logical seeds with stage quotas
300/150/40/10.  The observed empty-pool allocation histogram can start 431 of
those seeds in 62 physical parents; the remaining 69 seeds are represented by
9 queued parents.  Packing never crosses a stage, bundle, or wave boundary.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import copy
import math
from typing import Any, Iterable, Mapping, Protocol, Sequence

try:
    from tier1_final1000_multiseed_contract import canonical_sha256
    from tier1_final1000_multiseed_driver import (
        validate_driver_state,
        validate_supervisor_stop,
    )
    from tier1_final1000_multiseed_phase_b_contract import (
        CHILD_CPUS,
        CHILD_MEMORY_MB,
        DEFAULT_MODEL_LOAD_STAGGER_SECONDS,
        MAX_CONCURRENT_CHILDREN,
        PROTOCOL_VERSION,
        build_concurrent_batch_task,
        validate_batch_task,
    )
    from tier1_final1000_slurm_controller import _refill_task, _task_templates
    from tier1_final1000_slurm_launch import validate_launch_plan
    from tier1_final1000_stage_profiles import STAGES
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_final1000_multiseed_contract import canonical_sha256
    from tools.tier1_final1000_multiseed_driver import (
        validate_driver_state,
        validate_supervisor_stop,
    )
    from tools.tier1_final1000_multiseed_phase_b_contract import (
        CHILD_CPUS,
        CHILD_MEMORY_MB,
        DEFAULT_MODEL_LOAD_STAGGER_SECONDS,
        MAX_CONCURRENT_CHILDREN,
        PROTOCOL_VERSION,
        build_concurrent_batch_task,
        validate_batch_task,
    )
    from tools.tier1_final1000_slurm_controller import _refill_task, _task_templates
    from tools.tier1_final1000_slurm_launch import validate_launch_plan
    from tools.tier1_final1000_stage_profiles import STAGES


ALLOCATION_INVENTORY_SCHEMA = "mft-tier1-final1000-phase-b-allocation-inventory-v1"
PLACEMENT_PLAN_SCHEMA = "mft-tier1-final1000-phase-b-placement-plan-v1"
PLACEMENT_EVIDENCE_SCHEMA = "mft-tier1-final1000-phase-b-placement-evidence-v1"
PHASE_A_ADAPTER_SCHEMA = "mft-tier1-final1000-phase-b-phase-a-adapter-v1"
MIGRATION_STATE_SCHEMA = "mft-tier1-final1000-phase-b-migration-state-v1"
CONSUMER_CAPABILITY_SCHEMA = "mft-tier1-final1000-phase-b-consumer-capability-v1"
PHASE_A_EXIT_RECEIPT_SCHEMA = "mft-tier1-final1000-phase-a-exit-receipt-v1"
FULL_INVENTORY_SHADOW_SCHEMA = "mft-tier1-final1000-phase-b-full-inventory-shadow-v1"
MUTATION_AUTHORITY_PROTOCOL = "single-owner-phase-a-phase-b-handoff-v1"
CPU_BENCHMARK_SCHEMA = "mft-tier1-final1000-phase-b-cpu-benchmark-design-v1"
FULL_INVENTORY_REQUIREMENTS = {
    "protocol": "stable-before-id-paged-full-inventory-v1",
    "required_helper_commit": "dafb4509d52f4abcacb68c7bacfc850468d7dc1f",
    "required_helper_file": "tools/tier1_final1000_slurm_controller.py",
    "required_helper_sha256": (
        "62d4a4dde02add583aad2161e9a61b2f1db36aab11acc1c778cc4a2c977f07eb"
    ),
    "snapshot_schema": "mft-tier1-final1000-inventory-snapshot-v1",
    "snapshot_fence_required": True,
    "cursor": "before_id",
    "sort": "id-desc",
    "page_size": 10_000,
    "full_inventory_required": True,
    "minimum_shadow_row_count": 13_000,
    "minimum_page_count": 2,
    "single_limit_10000_allowed": False,
}
DISPATCH_ADMISSION_POLICY = {
    "schema_version": "mft-tier1-final1000-phase-b-dispatch-admission-v1",
    "logical_child_cpus": CHILD_CPUS,
    "default_model_load_stagger_seconds": DEFAULT_MODEL_LOAD_STAGGER_SECONDS,
    "eight_lane_dispatch_fill_seconds": (
        (MAX_CONCURRENT_CHILDREN - 1) * DEFAULT_MODEL_LOAD_STAGGER_SECONDS
    ),
    "scheduler_ready_lane_reserve": 0,
    "scheduler_ready_lane_policy": "natural-terminal-vacancy-only-v1",
    "speculative_ready_lane_submission_allowed": False,
    "empty_node_or_capacity_prediction_allowed": False,
    "capacity_snapshot_recheck_before_activation_required": True,
    "capacity_snapshot_drift_action": "rerender-or-fail-closed",
}

TOTAL_LOGICAL_TARGET = 500
RUNNING_LOGICAL_TARGET = 431
QUEUED_LOGICAL_TARGET = 69
EXPECTED_PHASE_A_SUPERVISOR_PID = 55_304

# Each value is an allocation's usable count of 4-CPU/28-GiB logical lanes.
CURRENT_EMPTY_POOL_LANE_HISTOGRAM: dict[int, int] = {
    16: 15,
    14: 1,
    12: 1,
    11: 3,
    10: 1,
    9: 2,
    8: 4,
    7: 7,
    6: 1,
    5: 3,
    2: 1,
}

STAGE_LOGICAL_QUOTAS: dict[str, int] = {
    STAGES[0].stage_id: 300,
    STAGES[1].stage_id: 150,
    STAGES[2].stage_id: 40,
    STAGES[3].stage_id: 10,
}

# Shape counts are stated explicitly so review can verify every stage and the
# running/queued boundary without trusting a greedy packing algorithm.
RUNNING_STAGE_SHAPES: dict[str, dict[int, int]] = {
    STAGES[0].stage_id: {8: 18, 7: 7, 6: 1, 5: 3, 4: 1, 3: 3, 2: 1, 1: 2},
    STAGES[1].stage_id: {8: 18, 6: 1},
    STAGES[2].stage_id: {8: 5},
    STAGES[3].stage_id: {8: 1, 2: 1},
}
QUEUED_STAGE_SHAPES: dict[str, dict[int, int]] = {
    STAGES[0].stage_id: {8: 8, 5: 1},
    STAGES[1].stage_id: {},
    STAGES[2].stage_id: {},
    STAGES[3].stage_id: {},
}
EXPECTED_RUNNING_SHAPES = {8: 42, 7: 7, 6: 2, 5: 3, 4: 1, 3: 3, 2: 2, 1: 2}
EXPECTED_QUEUED_SHAPES = {8: 8, 5: 1}
EXPECTED_TOTAL_SHAPES = {8: 50, 7: 7, 6: 2, 5: 4, 4: 1, 3: 3, 2: 2, 1: 2}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _seal(value: Mapping[str, Any], seal_field: str) -> dict[str, Any]:
    unsigned = {
        key: copy.deepcopy(item) for key, item in value.items() if key != seal_field
    }
    return {**unsigned, seal_field: canonical_sha256(unsigned)}


def _phase_a_mutation_authority(supervisor_pid: int) -> dict[str, Any]:
    return {
        "protocol": MUTATION_AUTHORITY_PROTOCOL,
        "active_owner_kind": "phase-a-supervisor",
        "active_owner_identity": f"pid:{supervisor_pid}",
        "active_owner_count": 1,
        "simultaneous_owners_allowed": False,
        "phase_a_authority_relinquished": False,
        "phase_b_authority_granted": False,
        "phase_b_authority_lease_sha256": None,
        "phase_b_scheduler_mutation_enabled": False,
    }


def _validate_mutation_authority(
    value: Mapping[str, Any],
    *,
    supervisor_pid: int,
    released: bool,
) -> None:
    if not isinstance(value, Mapping):
        raise RuntimeError("single mutation authority is missing")
    if set(value) != {
        "protocol",
        "active_owner_kind",
        "active_owner_identity",
        "active_owner_count",
        "simultaneous_owners_allowed",
        "phase_a_authority_relinquished",
        "phase_b_authority_granted",
        "phase_b_authority_lease_sha256",
        "phase_b_scheduler_mutation_enabled",
    }:
        raise RuntimeError("single mutation authority fields drifted")
    common_matches = (
        value.get("protocol") == MUTATION_AUTHORITY_PROTOCOL
        and value.get("active_owner_count") == 1
        and value.get("simultaneous_owners_allowed") is False
        and value.get("phase_b_scheduler_mutation_enabled") is False
    )
    if not common_matches:
        raise RuntimeError("single mutation authority contract drifted")
    if released:
        lease = value.get("phase_b_authority_lease_sha256")
        if (
            value.get("active_owner_kind") != "phase-b-successor-controller"
            or value.get("active_owner_identity") != f"lease:{lease}"
            or value.get("phase_a_authority_relinquished") is not True
            or value.get("phase_b_authority_granted") is not True
            or not _is_sha256(lease)
        ):
            raise RuntimeError("Phase-B mutation authority lease is invalid")
    elif value != _phase_a_mutation_authority(supervisor_pid):
        raise RuntimeError("Phase-A must remain the sole mutation authority")


def _shape_counts(shapes: Mapping[str, Mapping[int, int]]) -> Counter[int]:
    result: Counter[int] = Counter()
    for stage_shapes in shapes.values():
        result.update(
            {
                int(shape): int(count)
                for shape, count in stage_shapes.items()
                if int(count)
            }
        )
    return result


def _logical_count(shapes: Mapping[int, int]) -> int:
    return sum(int(shape) * int(count) for shape, count in shapes.items())


def validate_static_packing_contract() -> None:
    if set(STAGE_LOGICAL_QUOTAS) != {stage.stage_id for stage in STAGES}:
        raise RuntimeError("Phase-B stage quota identity drifted")
    if sum(STAGE_LOGICAL_QUOTAS.values()) != TOTAL_LOGICAL_TARGET:
        raise RuntimeError("Phase-B stage quotas must sum to 500")
    running = _shape_counts(RUNNING_STAGE_SHAPES)
    queued = _shape_counts(QUEUED_STAGE_SHAPES)
    if dict(running) != EXPECTED_RUNNING_SHAPES:
        raise RuntimeError("Phase-B running shape contract drifted")
    if dict(queued) != EXPECTED_QUEUED_SHAPES:
        raise RuntimeError("Phase-B queued shape contract drifted")
    if dict(running + queued) != EXPECTED_TOTAL_SHAPES:
        raise RuntimeError("Phase-B total shape contract drifted")
    if _logical_count(running) != RUNNING_LOGICAL_TARGET:
        raise RuntimeError("Phase-B running logical target drifted")
    if _logical_count(queued) != QUEUED_LOGICAL_TARGET:
        raise RuntimeError("Phase-B queued logical target drifted")
    for stage in STAGES:
        stage_id = stage.stage_id
        actual = _logical_count(
            Counter(RUNNING_STAGE_SHAPES[stage_id])
            + Counter(QUEUED_STAGE_SHAPES[stage_id])
        )
        if actual != STAGE_LOGICAL_QUOTAS[stage_id]:
            raise RuntimeError(f"Phase-B {stage_id} quota/shape identity drifted")


validate_static_packing_contract()


def build_cpu_efficiency_benchmark_design(
    *,
    bundle_manifest_sha256: str,
    model_inventory_sha256: str,
    node_identity: str,
    seed_start: int,
) -> dict[str, Any]:
    """Seal a finite same-node benchmark without changing production CPUs."""

    if (
        not _is_sha256(bundle_manifest_sha256)
        or not _is_sha256(model_inventory_sha256)
        or not str(node_identity or "")
        or isinstance(seed_start, bool)
        or not isinstance(seed_start, int)
        or seed_start < 0
    ):
        raise RuntimeError("CPU benchmark identity is invalid")
    scaling_seeds = list(range(seed_start, seed_start + 3))
    density_seeds = list(range(seed_start + 3, seed_start + 13))
    unsigned = {
        "schema_version": CPU_BENCHMARK_SCHEMA,
        "bundle_manifest_sha256": bundle_manifest_sha256,
        "model_inventory_sha256": model_inventory_sha256,
        "node_identity": node_identity,
        "same_node_required": True,
        "same_bundle_model_payload_seed_required": True,
        "cache_policy": "seed-local-cold-cache-every-run",
        "balanced_order": "replicate-1-ascending-replicate-2-descending",
        "fixed_workload": {
            "population": 320,
            "generations": 20,
            "full_generation_authentication": True,
            "result_science_identity_required": True,
        },
        "single_child_cpu_scaling": {
            "cpu_counts": [4, 6, 8],
            "seeds": scaling_seeds,
            "replicates": 2,
            "logical_run_count": len(scaling_seeds) * 3 * 2,
            "exclusive_cpuset_per_run": True,
        },
        "concurrent_density": {
            "parent_cpus": 32,
            "child_cpus": 4,
            "same_seeds": density_seeds,
            "replicates": 2,
            "configurations": [
                {
                    "id": "exact-8x4",
                    "simultaneous_children": 8,
                    "logical_cpus": 32,
                    "oversubscription_ratio": 1.0,
                    "waves_for_ten_seeds": [8, 2],
                },
                {
                    "id": "experimental-10x4-on-32",
                    "simultaneous_children": 10,
                    "logical_cpus": 40,
                    "oversubscription_ratio": 1.25,
                    "waves_for_ten_seeds": [10],
                },
            ],
            "logical_run_count": len(density_seeds) * 2 * 2,
        },
        "total_logical_run_count": 58,
        "metrics": [
            "wall_time_seconds",
            "seeds_per_hour",
            "cpu_hours_per_seed",
            "child_cpu_utilization_fraction",
            "parent_cpu_utilization_fraction",
            "p95_child_wall_time_seconds",
            "peak_rss_bytes",
            "failure_count",
            "semlock_construction_attempt_count",
        ],
        "eligibility_thresholds": {
            "minimum_throughput_improvement_fraction": 0.10,
            "maximum_cpu_hours_per_seed_regression_fraction": 0.20,
            "maximum_p95_wall_regression_fraction": 0.20,
            "required_failure_count": 0,
            "required_semlock_construction_attempt_count": 0,
            "bit_exact_science_identity_required": True,
        },
        "observational_baseline": {
            "unique_node_mean_busy_fraction": 0.374,
            "physical_cpu_count": 3704,
            "scheduler_allocated_cpu_count": 1838,
            "causal_scaling_claim_allowed": False,
        },
        "production_child_cpus_before_benchmark": CHILD_CPUS,
        "production_child_cpus_after_design": CHILD_CPUS,
        "automatic_resource_policy_change_allowed": False,
        "production_eligible": False,
        "scheduler_write_performed": False,
        "submission_performed": False,
        "remote_write_performed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    return _seal(unsigned, "benchmark_design_sha256")


def build_allocation_inventory(
    allocations: Sequence[Mapping[str, Any]],
    *,
    observed_at: str,
    source: str = "read-only-capacity-snapshot",
) -> dict[str, Any]:
    """Seal a canonical inventory of usable lane capacity.

    ``allocation_id`` is documentary identity only; it is intentionally not
    inserted into Scheduler task envelopes.  A rollout controller must read a
    fresh allocation inventory immediately before activation and require the
    same seal, otherwise it must rerender or stop.
    """

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for allocation in allocations:
        allocation_id = str(allocation.get("allocation_id") or "")
        capacity = allocation.get("usable_lane_capacity")
        if (
            not allocation_id
            or allocation_id in seen
            or isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or capacity < 1
        ):
            raise RuntimeError("allocation inventory row is invalid")
        seen.add(allocation_id)
        normalized.append(
            {
                "allocation_id": allocation_id,
                "usable_lane_capacity": int(capacity),
                "state": "active",
                "capacity_unit": "4cpu-28672mib-child",
            }
        )
    normalized.sort(key=lambda item: item["allocation_id"])
    unsigned = {
        "schema_version": ALLOCATION_INVENTORY_SCHEMA,
        "observed_at": str(observed_at),
        "source": str(source),
        "lane_unit": {"cpus": CHILD_CPUS, "memory_mb": CHILD_MEMORY_MB},
        "allocations": normalized,
        "allocation_count": len(normalized),
        "usable_lane_count": sum(row["usable_lane_capacity"] for row in normalized),
    }
    return _seal(unsigned, "allocation_inventory_sha256")


def validate_allocation_inventory(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        key: item for key, item in value.items() if key != "allocation_inventory_sha256"
    }
    allocations = value.get("allocations")
    if (
        value.get("schema_version") != ALLOCATION_INVENTORY_SCHEMA
        or value.get("allocation_inventory_sha256") != canonical_sha256(unsigned)
        or value.get("lane_unit") != {"cpus": CHILD_CPUS, "memory_mb": CHILD_MEMORY_MB}
        or not isinstance(allocations, list)
        or value.get("allocation_count") != len(allocations)
    ):
        raise RuntimeError("allocation inventory seal mismatch")
    rebuilt = build_allocation_inventory(
        allocations,
        observed_at=str(value.get("observed_at") or ""),
        source=str(value.get("source") or ""),
    )
    if rebuilt != value:
        raise RuntimeError("allocation inventory is not canonical")
    return copy.deepcopy(dict(value))


def current_empty_pool_inventory(
    *, observed_at: str = "2026-07-22T00:00:00+09:00"
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for capacity, count in sorted(CURRENT_EMPTY_POOL_LANE_HISTOGRAM.items()):
        for ordinal in range(count):
            rows.append(
                {
                    "allocation_id": f"documentary-cap{capacity:02d}-{ordinal + 1:02d}",
                    "usable_lane_capacity": capacity,
                }
            )
    return build_allocation_inventory(
        rows,
        observed_at=observed_at,
        source="documentary-current-active39-empty-pool-histogram",
    )


def decompose_allocation_inventory(
    inventory: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Split each allocation into exact <=8-child parent slots."""

    validated = validate_allocation_inventory(inventory)
    slots: list[dict[str, Any]] = []
    for allocation in validated["allocations"]:
        capacity = int(allocation["usable_lane_capacity"])
        shapes = [MAX_CONCURRENT_CHILDREN] * (capacity // MAX_CONCURRENT_CHILDREN)
        remainder = capacity % MAX_CONCURRENT_CHILDREN
        if remainder:
            shapes.append(remainder)
        if sum(shapes) != capacity:
            raise AssertionError("allocation decomposition lost capacity")
        for ordinal, lane_count in enumerate(shapes):
            slots.append(
                {
                    "allocation_id": allocation["allocation_id"],
                    "allocation_slot_ordinal": ordinal,
                    "lane_count": lane_count,
                    "cpus": lane_count * CHILD_CPUS,
                    "memory_mb": lane_count * CHILD_MEMORY_MB,
                }
            )
    if sum(slot["lane_count"] for slot in slots) != validated["usable_lane_count"]:
        raise RuntimeError("allocation decomposition did not consume exact capacity")
    return slots


def _histogram_from_inventory(inventory: Mapping[str, Any]) -> dict[int, int]:
    validated = validate_allocation_inventory(inventory)
    return dict(
        sorted(
            Counter(
                int(row["usable_lane_capacity"]) for row in validated["allocations"]
            ).items()
        )
    )


def require_inventory_identity(
    placement_plan: Mapping[str, Any], current_inventory: Mapping[str, Any]
) -> None:
    current = validate_allocation_inventory(current_inventory)
    if (
        placement_plan.get("rendered_from_allocation_inventory_sha256")
        != current["allocation_inventory_sha256"]
    ):
        raise RuntimeError(
            "allocation inventory drifted; Phase-B placement must be rerendered"
        )


def _stage_shape_sequence(
    shapes_by_stage: Mapping[str, Mapping[int, int]],
) -> dict[str, list[int]]:
    return {
        stage.stage_id: [
            shape
            for shape in range(MAX_CONCURRENT_CHILDREN, 0, -1)
            for _ in range(int(shapes_by_stage[stage.stage_id].get(shape, 0)))
        ]
        for stage in STAGES
    }


def _assign_running_slots(
    slots: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    available_by_shape: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for slot in slots:
        available_by_shape[int(slot["lane_count"])].append(copy.deepcopy(dict(slot)))
    for values in available_by_shape.values():
        values.sort(
            key=lambda item: (
                str(item["allocation_id"]),
                int(item["allocation_slot_ordinal"]),
            )
        )
    expected = Counter(EXPECTED_RUNNING_SHAPES)
    actual = Counter(
        {shape: len(values) for shape, values in available_by_shape.items()}
    )
    if actual != expected:
        raise RuntimeError(
            "allocation inventory does not match the reviewed running shape; "
            "rerender and audit a new placement"
        )
    assigned: list[dict[str, Any]] = []
    remaining = copy.deepcopy(available_by_shape)
    sequences = _stage_shape_sequence(RUNNING_STAGE_SHAPES)
    for stage in STAGES:
        for stage_parent_ordinal, shape in enumerate(sequences[stage.stage_id]):
            if not remaining[shape]:
                raise RuntimeError("running stage shape cannot be assigned")
            slot = remaining[shape].pop(0)
            assigned.append(
                {
                    **slot,
                    "stage_id": stage.stage_id,
                    "stage_parent_ordinal": stage_parent_ordinal,
                    "scheduler_disposition": "running-capacity",
                }
            )
    if any(remaining.values()):
        raise RuntimeError("running placement left an allocation slot unused")
    return assigned


def _queued_slots() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    sequences = _stage_shape_sequence(QUEUED_STAGE_SHAPES)
    global_ordinal = 0
    for stage in STAGES:
        for stage_parent_ordinal, shape in enumerate(sequences[stage.stage_id]):
            result.append(
                {
                    "allocation_id": None,
                    "allocation_slot_ordinal": None,
                    "lane_count": shape,
                    "cpus": shape * CHILD_CPUS,
                    "memory_mb": shape * CHILD_MEMORY_MB,
                    "stage_id": stage.stage_id,
                    "stage_parent_ordinal": stage_parent_ordinal,
                    "queued_parent_ordinal": global_ordinal,
                    "scheduler_disposition": "queued-capacity",
                }
            )
            global_ordinal += 1
    return result


def _logical_child_summary(
    *, stage_id: str, child: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "stage_id": stage_id,
        "seed": int(child["seed"]),
        "logical_dedupe_key": str(child["logical_dedupe_key"]),
        "child_task_sha256": str(child["child_task_sha256"]),
        "payload_sha256": str(child["payload_sha256"]),
    }


def logical_child_inventory_sha256(
    parent_tasks: Iterable[Mapping[str, Any]],
) -> str:
    children: list[dict[str, Any]] = []
    for parent in parent_tasks:
        task = validate_batch_task(parent)
        payload = task["payload_json"]
        children.extend(
            _logical_child_summary(stage_id=payload["stage_id"], child=child)
            for child in payload["children"]
        )
    children.sort(key=lambda item: (item["stage_id"], item["seed"]))
    if len({item["logical_dedupe_key"] for item in children}) != len(children):
        raise RuntimeError("placement duplicates a logical child identity")
    return canonical_sha256({"logical_children": children})


def _build_stage_children(
    plan: Mapping[str, Any], next_seed_by_stage: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    templates = _task_templates(plan)
    result: dict[str, list[dict[str, Any]]] = {}
    for stage in STAGES:
        start = next_seed_by_stage.get(stage.stage_id)
        if isinstance(start, bool) or not isinstance(start, int):
            raise RuntimeError("next seed inventory is incomplete")
        quota = STAGE_LOGICAL_QUOTAS[stage.stage_id]
        if not (
            stage.seed_start <= start
            and start + quota <= stage.seed_window_end_exclusive
        ):
            raise RuntimeError("successor placement escapes a stage seed window")
        result[stage.stage_id] = [
            _refill_task(
                templates[stage.stage_id],
                stage_id=stage.stage_id,
                seed=start + offset,
                wave="refill",
            )
            for offset in range(quota)
        ]
    return result


def _parent_summary(
    task: Mapping[str, Any], placement: Mapping[str, Any]
) -> dict[str, Any]:
    validated = validate_batch_task(task)
    payload = validated["payload_json"]
    return {
        "stage_id": payload["stage_id"],
        "wave": payload["wave"],
        "seed_start": int(payload["children"][0]["seed"]),
        "seed_end": int(payload["children"][-1]["seed"]),
        "lane_count": int(payload["batch_length"]),
        "cpus": int(validated["cpus"]),
        "memory_mb": int(validated["memory_mb"]),
        "parent_dedupe_key": str(validated["dedupe_key"]),
        "parent_task_sha256": canonical_sha256(validated),
        "allocation_id": placement.get("allocation_id"),
        "allocation_slot_ordinal": placement.get("allocation_slot_ordinal"),
        "scheduler_disposition": placement["scheduler_disposition"],
    }


def build_placement_plan(
    plan: Mapping[str, Any],
    *,
    next_seed_by_stage: Mapping[str, Any],
    allocation_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    """Render the reviewed capacity-aware 431-running/69-queued plan."""

    source_plan = validate_launch_plan(plan)
    inventory = validate_allocation_inventory(allocation_inventory)
    if _histogram_from_inventory(inventory) != CURRENT_EMPTY_POOL_LANE_HISTOGRAM:
        raise RuntimeError(
            "current allocation histogram changed; static placement is ineligible"
        )
    running_slots = _assign_running_slots(decompose_allocation_inventory(inventory))
    queued_slots = _queued_slots()
    children = _build_stage_children(source_plan, next_seed_by_stage)
    cursors = {stage.stage_id: 0 for stage in STAGES}
    parent_tasks: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for placement in [*running_slots, *queued_slots]:
        stage_id = str(placement["stage_id"])
        start = cursors[stage_id]
        stop = start + int(placement["lane_count"])
        block = children[stage_id][start:stop]
        if len(block) != int(placement["lane_count"]):
            raise RuntimeError("placement shape exceeded its stage child inventory")
        parent = build_concurrent_batch_task(block)
        payload = parent["payload_json"]
        if (
            payload["stage_id"] != stage_id
            or payload["wave"] != "refill"
            or payload["batch_length"] != placement["lane_count"]
        ):
            raise RuntimeError("placement crossed a stage/wave boundary")
        cursors[stage_id] = stop
        parent_tasks.append(parent)
        summaries.append(_parent_summary(parent, placement))
    if cursors != STAGE_LOGICAL_QUOTAS:
        raise RuntimeError("placement did not consume exact stage quotas")
    shape_counts = Counter(summary["lane_count"] for summary in summaries)
    running_count = sum(
        summary["lane_count"]
        for summary in summaries
        if summary["scheduler_disposition"] == "running-capacity"
    )
    queued_count = sum(
        summary["lane_count"]
        for summary in summaries
        if summary["scheduler_disposition"] == "queued-capacity"
    )
    logical_sha = logical_child_inventory_sha256(parent_tasks)

    # Repack the exact same child tasks one-per-parent.  This is not a launch
    # plan; it proves the scientific child identity does not depend on lanes.
    one_lane_tasks = [
        build_concurrent_batch_task([child])
        for stage in STAGES
        for child in children[stage.stage_id]
    ]
    one_lane_sha = logical_child_inventory_sha256(one_lane_tasks)
    if logical_sha != one_lane_sha:
        raise RuntimeError("logical science identity depends on lane grouping")

    unsigned = {
        "schema_version": PLACEMENT_PLAN_SCHEMA,
        "phase_b_protocol_version": PROTOCOL_VERSION,
        "source_launch_plan_sha256": source_plan["launch_plan_sha256"],
        "rendered_from_allocation_inventory_sha256": inventory[
            "allocation_inventory_sha256"
        ],
        "allocation_histogram": {
            str(key): value
            for key, value in sorted(_histogram_from_inventory(inventory).items())
        },
        "stage_logical_quotas": copy.deepcopy(STAGE_LOGICAL_QUOTAS),
        "next_seed_by_stage": {
            stage.stage_id: int(next_seed_by_stage[stage.stage_id]) for stage in STAGES
        },
        "running_logical_count": running_count,
        "queued_logical_count": queued_count,
        "logical_seed_count": running_count + queued_count,
        "physical_parent_count": len(parent_tasks),
        "running_parent_count": len(running_slots),
        "queued_parent_count": len(queued_slots),
        "shape_parent_counts": {
            str(shape): shape_counts[shape] for shape in sorted(shape_counts)
        },
        "running_stage_shapes": {
            stage_id: {str(shape): count for shape, count in sorted(shapes.items())}
            for stage_id, shapes in RUNNING_STAGE_SHAPES.items()
        },
        "queued_stage_shapes": {
            stage_id: {str(shape): count for shape, count in sorted(shapes.items())}
            for stage_id, shapes in QUEUED_STAGE_SHAPES.items()
        },
        "parent_summaries": summaries,
        "parent_task_inventory_sha256": canonical_sha256(
            {"parent_tasks": parent_tasks}
        ),
        "logical_child_inventory_sha256": logical_sha,
        "one_lane_repack_logical_child_inventory_sha256": one_lane_sha,
        "lane_count_independent_science_identity": True,
        "no_stage_bundle_or_wave_crossing": True,
        "inventory_recheck_required_at_activation": True,
        "inventory_drift_action": "rerender-or-fail-closed",
        "dispatch_admission_policy": copy.deepcopy(DISPATCH_ADMISSION_POLICY),
        "production_eligible": False,
        "scheduler_write_performed": False,
        "submission_performed": False,
        "remote_write_performed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    result = _seal(unsigned, "placement_plan_sha256")
    validate_placement_plan(result, parent_tasks=parent_tasks)
    return {**result, "parent_tasks": parent_tasks}


def validate_placement_plan(
    value: Mapping[str, Any], *, parent_tasks: Sequence[Mapping[str, Any]] | None = None
) -> dict[str, Any]:
    plan_without_tasks = {
        key: item for key, item in value.items() if key != "parent_tasks"
    }
    unsigned = {
        key: item
        for key, item in plan_without_tasks.items()
        if key != "placement_plan_sha256"
    }
    if (
        value.get("schema_version") != PLACEMENT_PLAN_SCHEMA
        or value.get("phase_b_protocol_version") != PROTOCOL_VERSION
        or value.get("placement_plan_sha256") != canonical_sha256(unsigned)
        or not _is_sha256(value.get("source_launch_plan_sha256"))
        or not _is_sha256(value.get("rendered_from_allocation_inventory_sha256"))
        or value.get("stage_logical_quotas") != STAGE_LOGICAL_QUOTAS
        or value.get("running_logical_count") != RUNNING_LOGICAL_TARGET
        or value.get("queued_logical_count") != QUEUED_LOGICAL_TARGET
        or value.get("logical_seed_count") != TOTAL_LOGICAL_TARGET
        or value.get("physical_parent_count") != 71
        or value.get("running_parent_count") != 62
        or value.get("queued_parent_count") != 9
        or value.get("shape_parent_counts")
        != {str(shape): count for shape, count in sorted(EXPECTED_TOTAL_SHAPES.items())}
        or value.get("lane_count_independent_science_identity") is not True
        or value.get("logical_child_inventory_sha256")
        != value.get("one_lane_repack_logical_child_inventory_sha256")
        or value.get("no_stage_bundle_or_wave_crossing") is not True
        or value.get("inventory_recheck_required_at_activation") is not True
        or value.get("inventory_drift_action") != "rerender-or-fail-closed"
        or value.get("dispatch_admission_policy") != DISPATCH_ADMISSION_POLICY
        or any(
            value.get(field) is not False
            for field in (
                "production_eligible",
                "scheduler_write_performed",
                "submission_performed",
                "remote_write_performed",
                "fea_submission_performed",
                "aedt_used",
            )
        )
    ):
        raise RuntimeError("Phase-B placement plan seal mismatch")
    summaries = value.get("parent_summaries")
    if not isinstance(summaries, list) or len(summaries) != 71:
        raise RuntimeError("Phase-B placement parent inventory drifted")
    if parent_tasks is not None:
        validated_tasks = [validate_batch_task(task) for task in parent_tasks]
        if (
            len(validated_tasks) != 71
            or canonical_sha256({"parent_tasks": validated_tasks})
            != value.get("parent_task_inventory_sha256")
            or logical_child_inventory_sha256(validated_tasks)
            != value.get("logical_child_inventory_sha256")
        ):
            raise RuntimeError("Phase-B placement task inventory drifted")
    return copy.deepcopy(plan_without_tasks)


def build_placement_evidence(
    placement_plan: Mapping[str, Any],
    allocation_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    placement = validate_placement_plan(placement_plan)
    inventory = validate_allocation_inventory(allocation_inventory)
    require_inventory_identity(placement, inventory)
    unsigned = {
        "schema_version": PLACEMENT_EVIDENCE_SCHEMA,
        "placement_plan_sha256": placement["placement_plan_sha256"],
        "allocation_inventory_sha256": inventory["allocation_inventory_sha256"],
        "allocation_histogram": placement["allocation_histogram"],
        "stage_logical_quotas": placement["stage_logical_quotas"],
        "running_logical_count": placement["running_logical_count"],
        "queued_logical_count": placement["queued_logical_count"],
        "logical_seed_count": placement["logical_seed_count"],
        "physical_parent_count": placement["physical_parent_count"],
        "running_parent_count": placement["running_parent_count"],
        "queued_parent_count": placement["queued_parent_count"],
        "shape_parent_counts": placement["shape_parent_counts"],
        "running_stage_shapes": placement["running_stage_shapes"],
        "queued_stage_shapes": placement["queued_stage_shapes"],
        "parent_task_inventory_sha256": placement["parent_task_inventory_sha256"],
        "logical_child_inventory_sha256": placement["logical_child_inventory_sha256"],
        "one_lane_repack_logical_child_inventory_sha256": placement[
            "one_lane_repack_logical_child_inventory_sha256"
        ],
        "lane_count_independent_science_identity": True,
        "capacity_snapshot_recheck_required": True,
        "capacity_snapshot_drift_action": "rerender-or-fail-closed",
        "dispatch_admission_policy": copy.deepcopy(DISPATCH_ADMISSION_POLICY),
        "documentary_only": True,
        "production_eligible": False,
        "scheduler_write_performed": False,
        "submission_performed": False,
        "remote_write_performed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    return _seal(unsigned, "evidence_sha256")


def _validate_driver(
    state: Mapping[str, Any],
    plan: Mapping[str, Any],
    monitoring_capability: Mapping[str, Any],
) -> dict[str, Any]:
    return validate_driver_state(state, plan, monitoring_capability)


def adapt_phase_a_controller_state(
    driver_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert Phase-A active parents into observe-only mixed-protocol rows.

    A sequential Phase-A parent claims one currently runnable logical slot;
    remaining children are an internal future backlog, not extra concurrency.
    No imported parent may be cancelled to accelerate quota redistribution.
    """

    if driver_state.get("phase") != "refill":
        raise RuntimeError("Phase-A adapter requires the refill phase")
    controller = driver_state.get("controller_state")
    if not isinstance(controller, dict):
        raise RuntimeError("Phase-A refill has no controller state")
    active_states = {"planned", "submitted", "queued", "running"}
    imported: list[dict[str, Any]] = []
    identities: set[str] = set()
    task_ids: set[int] = set()
    for entry in controller.get("entries") or []:
        if entry.get("state") not in active_states:
            continue
        dedupe = str(entry.get("parent_dedupe_key") or "")
        seeds = entry.get("seeds")
        task_id = entry.get("task_id")
        if (
            not dedupe
            or dedupe in identities
            or entry.get("stage_id") not in STAGE_LOGICAL_QUOTAS
            or not isinstance(seeds, list)
            or not seeds
            or any(
                isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
            )
            or (
                task_id is not None
                and (
                    isinstance(task_id, bool)
                    or not isinstance(task_id, int)
                    or task_id <= 0
                )
            )
            or (task_id is not None and task_id in task_ids)
        ):
            raise RuntimeError("Phase-A imported entry identity is invalid")
        identities.add(dedupe)
        if task_id is not None:
            task_ids.add(task_id)
        record = {
            "source_protocol_version": str(entry.get("protocol_version") or ""),
            "stage_id": str(entry["stage_id"]),
            "bundle_id": str(entry.get("bundle_id") or ""),
            "wave": str(entry.get("wave") or ""),
            "seeds": [int(seed) for seed in seeds],
            "parent_dedupe_key": dedupe,
            "task_id": task_id,
            "state": str(entry["state"]),
            "active_concurrency_claim": 1,
            "internal_future_seed_backlog": max(0, len(seeds) - 1),
            "ownership": "phase-a-observe-until-natural-terminal",
            "cancellation_allowed": False,
            "replacement_before_terminal_allowed": False,
        }
        record["entry_sha256"] = canonical_sha256(record)
        imported.append(record)
    imported.sort(key=lambda item: (item["stage_id"], item["parent_dedupe_key"]))
    next_seeds = controller.get("next_seed_by_stage")
    if not isinstance(next_seeds, dict) or set(next_seeds) != set(STAGE_LOGICAL_QUOTAS):
        raise RuntimeError("Phase-A next-seed inventory is incomplete")
    active_by_stage = {
        stage.stage_id: sum(
            row["active_concurrency_claim"]
            for row in imported
            if row["stage_id"] == stage.stage_id
        )
        for stage in STAGES
    }
    unsigned = {
        "schema_version": PHASE_A_ADAPTER_SCHEMA,
        "source_driver_state_sha256": driver_state["state_sha256"],
        "source_driver_revision": int(driver_state["revision"]),
        "source_controller_state_sha256": controller["state_sha256"],
        "source_phase": "refill",
        "next_seed_by_stage": {
            stage.stage_id: int(next_seeds[stage.stage_id]) for stage in STAGES
        },
        "imported_entries": imported,
        "imported_parent_count": len(imported),
        "imported_active_concurrency_by_stage": active_by_stage,
        "imported_active_concurrency": sum(active_by_stage.values()),
        "phase_a_parent_concurrency_accounting": "one-current-plus-internal-backlog-v1",
        "phase_b_parent_concurrency_accounting": "exact-batch-length-v1",
        "quota_rebalance_policy": "natural-terminal-deficit-only",
        "cancellation_performed": False,
        "scheduler_write_performed": False,
        "virtual_scheduler_task_ids_created": False,
    }
    return _seal(unsigned, "adapter_sha256")


def validate_phase_a_adapter(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "adapter_sha256"}
    entries = value.get("imported_entries")
    if (
        value.get("schema_version") != PHASE_A_ADAPTER_SCHEMA
        or value.get("adapter_sha256") != canonical_sha256(unsigned)
        or value.get("source_phase") != "refill"
        or not _is_sha256(value.get("source_driver_state_sha256"))
        or not _is_sha256(value.get("source_controller_state_sha256"))
        or not isinstance(entries, list)
        or value.get("imported_parent_count") != len(entries)
        or value.get("phase_a_parent_concurrency_accounting")
        != "one-current-plus-internal-backlog-v1"
        or value.get("phase_b_parent_concurrency_accounting") != "exact-batch-length-v1"
        or value.get("quota_rebalance_policy") != "natural-terminal-deficit-only"
        or value.get("cancellation_performed") is not False
        or value.get("scheduler_write_performed") is not False
        or value.get("virtual_scheduler_task_ids_created") is not False
    ):
        raise RuntimeError("Phase-A adapter seal mismatch")
    for entry in entries:
        entry_unsigned = {
            key: item for key, item in entry.items() if key != "entry_sha256"
        }
        if (
            entry.get("entry_sha256") != canonical_sha256(entry_unsigned)
            or entry.get("active_concurrency_claim") != 1
            or entry.get("cancellation_allowed") is not False
            or entry.get("replacement_before_terminal_allowed") is not False
        ):
            raise RuntimeError("Phase-A adapter entry seal mismatch")
    return copy.deepcopy(dict(value))


def validate_full_inventory_snapshot_receipt(
    value: Mapping[str, Any], *, minimum_rows: int = 13_000
) -> dict[str, Any]:
    """Validate the exact receipt emitted by paging helper commit dafb450."""

    required = {
        "schema_version",
        "high_watermark_task_id",
        "before_id",
        "filtered_total",
        "page_size",
        "page_count",
        "task_ids_sha256",
        "server_snapshot_revision",
        "sha256",
    }
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    high_watermark = value.get("high_watermark_task_id")
    before_id = value.get("before_id")
    total = value.get("filtered_total")
    page_size = value.get("page_size")
    page_count = value.get("page_count")
    if (
        set(value) != required
        or value.get("schema_version") != FULL_INVENTORY_REQUIREMENTS["snapshot_schema"]
        or value.get("sha256") != canonical_sha256(unsigned)
        or isinstance(high_watermark, bool)
        or not isinstance(high_watermark, int)
        or high_watermark < 0
        or before_id != high_watermark + 1
        or isinstance(total, bool)
        or not isinstance(total, int)
        or total < minimum_rows
        or page_size != FULL_INVENTORY_REQUIREMENTS["page_size"]
        or page_count != max(1, math.ceil(total / page_size))
        or page_count < FULL_INVENTORY_REQUIREMENTS["minimum_page_count"]
        or not _is_sha256(value.get("task_ids_sha256"))
    ):
        raise RuntimeError("full Scheduler inventory snapshot receipt mismatch")
    return copy.deepcopy(dict(value))


class FullInventoryShadowProvider(Protocol):
    """Minimal reviewed helper surface used by the read-only 13k shadow."""

    post_count: int
    inventory_get_count: int
    inventory_snapshot_receipt: Mapping[str, Any] | None

    def list_namespace_tasks(self) -> list[Mapping[str, Any]]: ...


def attest_full_inventory_shadow(
    provider: FullInventoryShadowProvider,
    *,
    helper_commit: str,
    helper_file_sha256: str,
) -> dict[str, Any]:
    """Exercise the reviewed paged helper and seal a mutation-free 13k shadow.

    The provider call is deliberately singular.  The reviewed helper performs
    its own high-watermark anchor plus every descending ``before_id`` page.
    This adapter checks the returned row order/digest against the helper's
    receipt and proves its Scheduler POST counter stayed at zero.
    """

    if (
        helper_commit != FULL_INVENTORY_REQUIREMENTS["required_helper_commit"]
        or helper_file_sha256 != FULL_INVENTORY_REQUIREMENTS["required_helper_sha256"]
    ):
        raise RuntimeError("full inventory helper release identity mismatch")
    before_posts = getattr(provider, "post_count", None)
    if isinstance(before_posts, bool) or not isinstance(before_posts, int):
        raise RuntimeError("full inventory shadow lacks a Scheduler POST counter")
    if before_posts != 0:
        raise RuntimeError("full inventory shadow did not start mutation-free")
    rows = provider.list_namespace_tasks()
    after_posts = getattr(provider, "post_count", None)
    if after_posts != before_posts:
        raise RuntimeError("full inventory shadow performed a Scheduler mutation")
    receipt = validate_full_inventory_snapshot_receipt(
        getattr(provider, "inventory_snapshot_receipt", None) or {},
        minimum_rows=FULL_INVENTORY_REQUIREMENTS["minimum_shadow_row_count"],
    )
    get_count = getattr(provider, "inventory_get_count", None)
    if (
        not isinstance(rows, list)
        or isinstance(get_count, bool)
        or not isinstance(get_count, int)
        or get_count < int(receipt["page_count"]) + 1
    ):
        raise RuntimeError("full inventory shadow helper execution is incomplete")
    task_ids: list[int] = []
    for row in rows:
        task_id = row.get("id") if isinstance(row, Mapping) else None
        if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
            raise RuntimeError("full inventory shadow row identity is invalid")
        task_ids.append(task_id)
    if (
        len(task_ids) != int(receipt["filtered_total"])
        or len(set(task_ids)) != len(task_ids)
        or any(left <= right for left, right in zip(task_ids, task_ids[1:]))
        or canonical_sha256(task_ids) != receipt["task_ids_sha256"]
        or (task_ids and task_ids[0] != receipt["high_watermark_task_id"])
    ):
        raise RuntimeError("full inventory shadow rows/receipt binding mismatch")
    unsigned = {
        "schema_version": FULL_INVENTORY_SHADOW_SCHEMA,
        "helper_commit": helper_commit,
        "helper_file": FULL_INVENTORY_REQUIREMENTS["required_helper_file"],
        "helper_file_sha256": helper_file_sha256,
        "snapshot_receipt": receipt,
        "snapshot_receipt_sha256": receipt["sha256"],
        "row_count": len(task_ids),
        "page_count": int(receipt["page_count"]),
        "inventory_get_count": get_count,
        "strictly_descending_unique_task_ids": True,
        "complete_inventory_passed": True,
        "minimum_13000_row_shadow_passed": True,
        "scheduler_api_methods_observed": ["GET"],
        "scheduler_post_count_before": before_posts,
        "scheduler_post_count_after": after_posts,
        "scheduler_mutation_performed": False,
    }
    return _seal(unsigned, "attestation_sha256")


def validate_full_inventory_shadow_attestation(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "attestation_sha256"}
    receipt = validate_full_inventory_snapshot_receipt(
        value.get("snapshot_receipt") or {},
        minimum_rows=FULL_INVENTORY_REQUIREMENTS["minimum_shadow_row_count"],
    )
    if (
        value.get("schema_version") != FULL_INVENTORY_SHADOW_SCHEMA
        or value.get("attestation_sha256") != canonical_sha256(unsigned)
        or value.get("helper_commit")
        != FULL_INVENTORY_REQUIREMENTS["required_helper_commit"]
        or value.get("helper_file")
        != FULL_INVENTORY_REQUIREMENTS["required_helper_file"]
        or value.get("helper_file_sha256")
        != FULL_INVENTORY_REQUIREMENTS["required_helper_sha256"]
        or value.get("snapshot_receipt_sha256") != receipt["sha256"]
        or value.get("row_count") != receipt["filtered_total"]
        or value.get("page_count") != receipt["page_count"]
        or isinstance(value.get("inventory_get_count"), bool)
        or not isinstance(value.get("inventory_get_count"), int)
        or value.get("inventory_get_count") < int(receipt["page_count"]) + 1
        or value.get("strictly_descending_unique_task_ids") is not True
        or value.get("complete_inventory_passed") is not True
        or value.get("minimum_13000_row_shadow_passed") is not True
        or value.get("scheduler_api_methods_observed") != ["GET"]
        or value.get("scheduler_post_count_before") != 0
        or value.get("scheduler_post_count_after") != 0
        or value.get("scheduler_mutation_performed") is not False
    ):
        raise RuntimeError("full inventory shadow attestation mismatch")
    return copy.deepcopy(dict(value))


def build_phase_b_consumer_capability(
    *,
    implementation_revision: str,
    test_evidence_sha256: str,
    shadow_validation_sha256: str,
    full_inventory_shadow_attestation: Mapping[str, Any],
    activation_authorized: bool,
) -> dict[str, Any]:
    """Seal the capability a separately reviewed Phase-B consumer must mint."""

    if (
        not implementation_revision
        or not _is_sha256(test_evidence_sha256)
        or not _is_sha256(shadow_validation_sha256)
        or not isinstance(activation_authorized, bool)
    ):
        raise RuntimeError("Phase-B consumer capability input is invalid")
    inventory_shadow = validate_full_inventory_shadow_attestation(
        full_inventory_shadow_attestation
    )
    inventory_receipt = inventory_shadow["snapshot_receipt"]
    if shadow_validation_sha256 != inventory_shadow["attestation_sha256"]:
        raise RuntimeError("Phase-B shadow validation SHA is not authoritative")
    unsigned = {
        "schema_version": CONSUMER_CAPABILITY_SCHEMA,
        "implementation_revision": str(implementation_revision),
        "test_evidence_sha256": test_evidence_sha256,
        "shadow_validation_sha256": shadow_validation_sha256,
        "supported_parent_protocols": [
            "final1000-finite-multiseed-v1",
            PROTOCOL_VERSION,
        ],
        "phase_b_manifest_status_receipt_supported": True,
        "phase_a_observe_only_import_supported": True,
        "mixed_protocol_inventory_supported": True,
        "physical_parent_task_id_preserved": True,
        "virtual_scheduler_task_ids_created": False,
        "scheduler_api_methods": ["GET"],
        "scheduler_mutation_endpoints": [],
        "full_inventory_requirements": copy.deepcopy(FULL_INVENTORY_REQUIREMENTS),
        "full_inventory_snapshot_receipt": inventory_receipt,
        "full_inventory_snapshot_receipt_sha256": inventory_receipt["sha256"],
        "full_inventory_shadow_attestation": inventory_shadow,
        "full_inventory_shadow_attestation_sha256": inventory_shadow[
            "attestation_sha256"
        ],
        "single_limit_10000_inventory_used": False,
        "shadow_validation_passed": True,
        "activation_authorized": activation_authorized,
    }
    return _seal(unsigned, "capability_sha256")


def validate_phase_b_consumer_capability(
    value: Mapping[str, Any], *, require_activation: bool = True
) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "capability_sha256"}
    if (
        value.get("schema_version") != CONSUMER_CAPABILITY_SCHEMA
        or value.get("capability_sha256") != canonical_sha256(unsigned)
        or value.get("supported_parent_protocols")
        != ["final1000-finite-multiseed-v1", PROTOCOL_VERSION]
        or value.get("phase_b_manifest_status_receipt_supported") is not True
        or value.get("phase_a_observe_only_import_supported") is not True
        or value.get("mixed_protocol_inventory_supported") is not True
        or value.get("physical_parent_task_id_preserved") is not True
        or value.get("virtual_scheduler_task_ids_created") is not False
        or value.get("scheduler_api_methods") != ["GET"]
        or value.get("scheduler_mutation_endpoints") != []
        or value.get("full_inventory_requirements") != FULL_INVENTORY_REQUIREMENTS
        or value.get("single_limit_10000_inventory_used") is not False
        or value.get("shadow_validation_passed") is not True
        or not isinstance(value.get("activation_authorized"), bool)
        or (require_activation and value.get("activation_authorized") is not True)
    ):
        raise RuntimeError("Phase-B consumer capability seal mismatch")
    receipt = validate_full_inventory_snapshot_receipt(
        value.get("full_inventory_snapshot_receipt") or {},
        minimum_rows=FULL_INVENTORY_REQUIREMENTS["minimum_shadow_row_count"],
    )
    if value.get("full_inventory_snapshot_receipt_sha256") != receipt["sha256"]:
        raise RuntimeError("Phase-B full inventory receipt binding mismatch")
    shadow = validate_full_inventory_shadow_attestation(
        value.get("full_inventory_shadow_attestation") or {}
    )
    if (
        value.get("full_inventory_shadow_attestation_sha256")
        != shadow["attestation_sha256"]
        or value.get("shadow_validation_sha256") != shadow["attestation_sha256"]
        or shadow["snapshot_receipt_sha256"] != receipt["sha256"]
    ):
        raise RuntimeError("Phase-B full inventory shadow binding mismatch")
    return copy.deepcopy(dict(value))


def build_phase_a_exit_receipt(
    *,
    first_stable_driver_read: Mapping[str, Any],
    second_stable_driver_read: Mapping[str, Any],
    plan: Mapping[str, Any],
    monitoring_capability: Mapping[str, Any],
    supervisor_stop_receipt: Mapping[str, Any],
    supervisor_pid: int,
    process_exited: bool,
) -> dict[str, Any]:
    first = _validate_driver(first_stable_driver_read, plan, monitoring_capability)
    second = _validate_driver(second_stable_driver_read, plan, monitoring_capability)
    stop = validate_supervisor_stop(supervisor_stop_receipt)
    if (
        first != second
        or first["state_sha256"] != second["state_sha256"]
        or first["revision"] != second["revision"]
        or isinstance(supervisor_pid, bool)
        or not isinstance(supervisor_pid, int)
        or supervisor_pid <= 0
        or process_exited is not True
    ):
        raise RuntimeError("Phase-A exit was not observed as a stable terminal handoff")
    unsigned = {
        "schema_version": PHASE_A_EXIT_RECEIPT_SCHEMA,
        "supervisor_pid": supervisor_pid,
        "process_exited": True,
        "supervisor_stop_receipt_sha256": stop["receipt_sha256"],
        "final_driver_state_sha256": first["state_sha256"],
        "final_driver_revision": int(first["revision"]),
        "stable_read_count": 2,
        "stable_reads_identical": True,
        "scheduler_mutation_performed": False,
    }
    return _seal(unsigned, "exit_receipt_sha256")


def validate_phase_a_exit_receipt(
    value: Mapping[str, Any],
    *,
    expected_driver_state_sha256: str,
    supervisor_stop_receipt: Mapping[str, Any],
    expected_supervisor_pid: int,
) -> dict[str, Any]:
    stop = validate_supervisor_stop(supervisor_stop_receipt)
    unsigned = {
        key: item for key, item in value.items() if key != "exit_receipt_sha256"
    }
    if (
        value.get("schema_version") != PHASE_A_EXIT_RECEIPT_SCHEMA
        or value.get("exit_receipt_sha256") != canonical_sha256(unsigned)
        or value.get("supervisor_pid") != expected_supervisor_pid
        or value.get("process_exited") is not True
        or value.get("supervisor_stop_receipt_sha256") != stop["receipt_sha256"]
        or value.get("final_driver_state_sha256") != expected_driver_state_sha256
        or value.get("stable_read_count") != 2
        or value.get("stable_reads_identical") is not True
        or value.get("scheduler_mutation_performed") is not False
    ):
        raise RuntimeError("Phase-A exit receipt seal mismatch")
    return copy.deepcopy(dict(value))


def prepare_migration_state(
    driver_state: Mapping[str, Any],
    plan: Mapping[str, Any],
    monitoring_capability: Mapping[str, Any],
    *,
    allocation_inventory: Mapping[str, Any] | None = None,
    expected_supervisor_pid: int = EXPECTED_PHASE_A_SUPERVISOR_PID,
) -> dict[str, Any]:
    """Build a no-intent shadow handoff from an authenticated Phase-A state."""

    source_plan = validate_launch_plan(plan)
    source = _validate_driver(driver_state, source_plan, monitoring_capability)
    if (
        isinstance(expected_supervisor_pid, bool)
        or not isinstance(expected_supervisor_pid, int)
        or expected_supervisor_pid <= 0
    ):
        raise RuntimeError("Phase-A supervisor PID is invalid")
    phase = str(source["phase"])
    if phase == "failed":
        raise RuntimeError("failed Phase-A state cannot be migrated")
    adapter = None
    placement = None
    migration_phase = "awaiting_phase_a_refill"
    inventory_sha = None
    if phase == "refill":
        if allocation_inventory is None:
            raise RuntimeError(
                "Phase-A refill migration requires an allocation inventory"
            )
        inventory = validate_allocation_inventory(allocation_inventory)
        adapter = adapt_phase_a_controller_state(source)
        rendered = build_placement_plan(
            source_plan,
            next_seed_by_stage=adapter["next_seed_by_stage"],
            allocation_inventory=inventory,
        )
        placement = {
            key: item for key, item in rendered.items() if key != "parent_tasks"
        }
        inventory_sha = inventory["allocation_inventory_sha256"]
        migration_phase = "awaiting_phase_a_supervisor_stop"
    unsigned = {
        "schema_version": MIGRATION_STATE_SCHEMA,
        "revision": 0,
        "parent_state_sha256": None,
        "phase": migration_phase,
        "source_launch_plan_sha256": source_plan["launch_plan_sha256"],
        "source_driver_state_sha256": source["state_sha256"],
        "source_driver_revision": int(source["revision"]),
        "source_driver_phase": phase,
        "expected_phase_a_supervisor_pid": expected_supervisor_pid,
        "phase_a_adapter": adapter,
        "placement_plan": placement,
        "allocation_inventory_sha256": inventory_sha,
        "phase_a_exit_receipt_sha256": None,
        "phase_b_consumer_capability_sha256": None,
        "authority_released": False,
        "mutation_authority": _phase_a_mutation_authority(expected_supervisor_pid),
        "scheduler_submission_intents": [],
        "scheduler_mutation_endpoints": [],
        "scheduler_write_performed": False,
        "cancellation_performed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    result = _seal(unsigned, "state_sha256")
    validate_migration_state(result)
    return result


def validate_migration_state(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "state_sha256"}
    phase = value.get("phase")
    adapter = value.get("phase_a_adapter")
    placement = value.get("placement_plan")
    released = value.get("authority_released")
    if (
        value.get("schema_version") != MIGRATION_STATE_SCHEMA
        or value.get("state_sha256") != canonical_sha256(unsigned)
        or phase
        not in {
            "awaiting_phase_a_refill",
            "awaiting_phase_a_supervisor_stop",
            "ready_for_authorized_activation",
        }
        or not isinstance(value.get("revision"), int)
        or value.get("revision") < 0
        or not _is_sha256(value.get("source_launch_plan_sha256"))
        or not _is_sha256(value.get("source_driver_state_sha256"))
        or not isinstance(value.get("source_driver_revision"), int)
        or not isinstance(value.get("expected_phase_a_supervisor_pid"), int)
        or value.get("expected_phase_a_supervisor_pid") <= 0
        or not isinstance(released, bool)
        or value.get("scheduler_submission_intents") != []
        or value.get("scheduler_mutation_endpoints") != []
        or any(
            value.get(field) is not False
            for field in (
                "scheduler_write_performed",
                "cancellation_performed",
                "fea_submission_performed",
                "aedt_used",
            )
        )
    ):
        raise RuntimeError("Phase-B migration state seal mismatch")
    _validate_mutation_authority(
        value.get("mutation_authority") or {},
        supervisor_pid=int(value["expected_phase_a_supervisor_pid"]),
        released=released,
    )
    if phase == "awaiting_phase_a_refill":
        if (
            adapter is not None
            or placement is not None
            or value.get("allocation_inventory_sha256") is not None
            or released
            or value.get("phase_a_exit_receipt_sha256") is not None
            or value.get("phase_b_consumer_capability_sha256") is not None
        ):
            raise RuntimeError("pre-refill migration state carried authority")
    else:
        if (
            not isinstance(adapter, dict)
            or not isinstance(placement, dict)
            or validate_phase_a_adapter(adapter) != adapter
            or validate_placement_plan(placement) != placement
            or value.get("allocation_inventory_sha256")
            != placement["rendered_from_allocation_inventory_sha256"]
        ):
            raise RuntimeError("Phase-B migration adapter/placement mismatch")
    if phase == "awaiting_phase_a_supervisor_stop":
        if (
            released
            or value.get("phase_a_exit_receipt_sha256") is not None
            or value.get("phase_b_consumer_capability_sha256") is not None
        ):
            raise RuntimeError("Phase-B authority released before stop")
    if phase == "ready_for_authorized_activation":
        if (
            released is not True
            or not _is_sha256(value.get("phase_a_exit_receipt_sha256"))
            or not _is_sha256(value.get("phase_b_consumer_capability_sha256"))
        ):
            raise RuntimeError("released Phase-B authority is incomplete")
    return copy.deepcopy(dict(value))


def release_migration_authority(
    state: Mapping[str, Any],
    *,
    supervisor_stop_receipt: Mapping[str, Any],
    phase_a_exit_receipt: Mapping[str, Any],
    phase_b_consumer_capability: Mapping[str, Any],
    current_allocation_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    """Release pure controller authority after every physical handoff gate.

    This transition still emits no Scheduler intent.  Submission remains a
    separately authorized activation step outside this module.
    """

    current = validate_migration_state(state)
    if current["phase"] != "awaiting_phase_a_supervisor_stop":
        raise RuntimeError("Phase-B migration is not waiting for supervisor stop")
    capability = validate_phase_b_consumer_capability(
        phase_b_consumer_capability, require_activation=True
    )
    exit_receipt = validate_phase_a_exit_receipt(
        phase_a_exit_receipt,
        expected_driver_state_sha256=current["source_driver_state_sha256"],
        supervisor_stop_receipt=supervisor_stop_receipt,
        expected_supervisor_pid=current["expected_phase_a_supervisor_pid"],
    )
    require_inventory_identity(current["placement_plan"], current_allocation_inventory)
    authority_lease_unsigned = {
        "protocol": MUTATION_AUTHORITY_PROTOCOL,
        "source_migration_state_sha256": current["state_sha256"],
        "phase_a_exit_receipt_sha256": exit_receipt["exit_receipt_sha256"],
        "phase_b_consumer_capability_sha256": capability["capability_sha256"],
        "allocation_inventory_sha256": current_allocation_inventory[
            "allocation_inventory_sha256"
        ],
        "scheduler_mutation_enabled": False,
    }
    authority_lease_sha = canonical_sha256(authority_lease_unsigned)
    unsigned = {
        key: copy.deepcopy(item)
        for key, item in current.items()
        if key != "state_sha256"
    }
    unsigned.update(
        {
            "revision": int(current["revision"]) + 1,
            "parent_state_sha256": current["state_sha256"],
            "phase": "ready_for_authorized_activation",
            "phase_a_exit_receipt_sha256": exit_receipt["exit_receipt_sha256"],
            "phase_b_consumer_capability_sha256": capability["capability_sha256"],
            "authority_released": True,
            "mutation_authority": {
                "protocol": MUTATION_AUTHORITY_PROTOCOL,
                "active_owner_kind": "phase-b-successor-controller",
                "active_owner_identity": f"lease:{authority_lease_sha}",
                "active_owner_count": 1,
                "simultaneous_owners_allowed": False,
                "phase_a_authority_relinquished": True,
                "phase_b_authority_granted": True,
                "phase_b_authority_lease_sha256": authority_lease_sha,
                "phase_b_scheduler_mutation_enabled": False,
            },
        }
    )
    result = _seal(unsigned, "state_sha256")
    validate_migration_state(result)
    return result
