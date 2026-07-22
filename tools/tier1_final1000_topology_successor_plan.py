"""Render lane-count-independent Final1000 topology successor plans.

Logical 4-CPU children are sealed independently from parent packing.  A later
Phase-B packer may group one through eight children per parent allocation
without changing any logical seed, dedupe key, payload, or science identity.
This renderer has no scheduler POST/cancel/publish surface.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from tier1_final1000_topology_niche_contract import (
        CANARY_SCHEMA,
        CANARY_STAGE_COUNTS,
        INITIAL_PRODUCTION_STAGE_QUOTAS,
        canonical_sha256,
        contract as topology_contract,
        validate_contract,
    )
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_final1000_topology_niche_contract import (
        CANARY_SCHEMA,
        CANARY_STAGE_COUNTS,
        INITIAL_PRODUCTION_STAGE_QUOTAS,
        canonical_sha256,
        contract as topology_contract,
        validate_contract,
    )


PLAN_SCHEMA = "mft-tier1-final1000-topology-successor-plan-v1"
PACKING_SCHEMA = "mft-tier1-final1000-variable-parent-packing-v1"
STAGE_ORDER = tuple(CANARY_STAGE_COUNTS)
SEED_WINDOWS = {
    # These disjoint high offsets remain inside the existing Final1000 stage
    # windows and their matching N1=6 Current7 island windows.  Consequently
    # the topology-balanced science can roll out through the deployed
    # single-seed 4c controller before variable-parent Phase-B is available.
    "entry-1200-t125": (2_250_000_000, 2_257_500_000),
    "bridge-1150-t115": (2_350_000_000, 2_357_500_000),
    "close-1075-t107p5": (2_450_000_000, 2_457_500_000),
    "final-1000-t100": (2_490_000_000, 2_500_000_000),
}
OPTIMIZER_ISLAND_BY_STAGE = {
    "entry-1200-t125": "n1-6-resanchor-allthermal-2p50c",
    "bridge-1150-t115": "n1-6-resanchor-allthermal-2p75c",
    "close-1075-t107p5": "n1-6-resanchor-allthermal-3p00c",
    "final-1000-t100": "n1-6-resanchor-allthermal-3p00c",
}

# Counts are indexed by lane size 8 .. 1.  This exact partition keeps the
# requested 300/150/40/10 science quotas while allowing Phase-B to use every
# 4-CPU fragment.  It is operational metadata only: changing the parent
# grouping cannot change a logical child's seed, payload hash, or dedupe key.
MIXED_PARENT_COUNTS_BY_STAGE = {
    "entry-1200-t125": {8: 26, 7: 7, 6: 1, 5: 4, 4: 1, 3: 3, 2: 1, 1: 2},
    "bridge-1150-t115": {8: 18, 7: 0, 6: 1, 5: 0, 4: 0, 3: 0, 2: 0, 1: 0},
    "close-1075-t107p5": {8: 5, 7: 0, 6: 0, 5: 0, 4: 0, 3: 0, 2: 0, 1: 0},
    "final-1000-t100": {8: 1, 7: 0, 6: 0, 5: 0, 4: 0, 3: 0, 2: 1, 1: 0},
}
MIXED_PARENT_COUNTS_TOTAL = {
    lane_count: sum(
        stage_counts[lane_count]
        for stage_counts in MIXED_PARENT_COUNTS_BY_STAGE.values()
    )
    for lane_count in range(8, 0, -1)
}

# The historical observed empty-pool histogram is authenticated as evidence,
# not baked into the science identity.  This expected shape demonstrates why
# fixed eight-lane parents expose only 336 logical slots while fragment-aware
# 1..8 packing exposes 431.
EXPECTED_EMPTY_POOL_CAPACITY_HISTOGRAM = {
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
CHILD_RESOURCE = {
    "scheduling_profile": "standard",
    "cpus": 4,
    "memory_mb": 28_672,
    "gpus": 0,
    "priority": 1,
    "max_workers_per_node": 32,
    "aedt_backend": "standalone",
    "aedt_used": False,
    "surrogate_only": True,
}


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _logical_children(
    counts: Mapping[str, int], *, wave: str, ordinal_offset: int = 0
) -> list[dict[str, Any]]:
    children = []
    global_ordinal = int(ordinal_offset)
    for stage in STAGE_ORDER:
        count = int(counts[stage])
        start, end = SEED_WINDOWS[stage]
        for stage_ordinal in range(count):
            seed = start + global_ordinal
            if seed >= end:
                raise RuntimeError("topology successor seed window exhausted")
            stable = {
                "schema_version": "mft-tier1-final1000-logical-seed-v1",
                "wave": str(wave),
                "stage_id": stage,
                "stage_ordinal": stage_ordinal,
                "global_ordinal": global_ordinal,
                "seed": seed,
                "fixed_primary_turns": 6,
                "optimizer_island_id": OPTIMIZER_ISLAND_BY_STAGE[stage],
                "topology_niche_contract_sha256": topology_contract()["sha256"],
                "population": 320,
                "max_generations": 300,
                "resource": dict(CHILD_RESOURCE),
                "fea_submission_approved": False,
                "fea_submission_performed": False,
                "aedt_used": False,
                "automatic_promotion_allowed": False,
            }
            payload_sha = canonical_sha256(stable)
            child = {
                **stable,
                "payload_sha256": payload_sha,
                "dedupe_key": (
                    f"mft-final1000-topology-{wave}-{stage}-{seed}-{payload_sha[:12]}"
                ),
            }
            children.append(child)
            global_ordinal += 1
    if len({item["seed"] for item in children}) != len(children):
        raise RuntimeError("topology successor logical seeds overlap")
    return children


def build_plan(
    bindings: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    niche = validate_contract(topology_contract())
    bindings = dict(bindings or {})
    binding_complete = set(bindings) == set(STAGE_ORDER) and all(
        len(str((bindings[stage] or {}).get("bundle_manifest_sha256") or "")) == 64
        and str((bindings[stage] or {}).get("remote_bundle") or "").startswith("/")
        for stage in STAGE_ORDER
    )
    canary = _logical_children(CANARY_STAGE_COUNTS, wave="canary")
    production = _logical_children(
        INITIAL_PRODUCTION_STAGE_QUOTAS,
        wave="production",
        ordinal_offset=len(canary),
    )
    stable = {
        "schema_version": PLAN_SCHEMA,
        "canary_schema_version": CANARY_SCHEMA,
        "topology_niche_contract_sha256": niche["sha256"],
        "stage_order": list(STAGE_ORDER),
        "seed_windows": {
            stage: {"start": start, "end_exclusive": end}
            for stage, (start, end) in SEED_WINDOWS.items()
        },
        "canary": {
            "logical_seed_count": len(canary),
            "stage_counts": dict(CANARY_STAGE_COUNTS),
            "logical_children": canary,
            "gate": {
                "all_16_terminal_exit_zero": True,
                "all_generation_topology_count_seals_valid": True,
                "every_generation_exact_topology_quota": True,
                "terminal_dynamic_allowances_zero": True,
                "terminal_physical_G_replay_unchanged": True,
                "no_AEDT_or_FEA": True,
            },
        },
        "production": {
            "active_target": 500,
            "initial_stage_quotas": dict(INITIAL_PRODUCTION_STAGE_QUOTAS),
            "logical_children": production,
            "adaptive_gate": {
                "policy": "U8_exploitation_plus_2_guard_exploration_v1",
                "update_batch_logical_seed_count": 10,
                "useful_or_improving_slots": 8,
                "guard_or_uncertainty_slots": 2,
                "terminal_physical_feasibility_and_near_violation_only": True,
                "automatic_hard_constraint_relaxation_allowed": False,
            },
            "parent_lane_target": 8,
            "parent_lane_count_is_not_science_identity": True,
        },
        "logical_child_resource": dict(CHILD_RESOURCE),
        "parent_packing_contract": {
            "lane_count_min": 1,
            "lane_count_max": 8,
            "logical_child_cpus": 4,
            "logical_child_memory_mb": 28_672,
            "packing_may_change_logical_payload": False,
            "packing_may_change_seed_or_dedupe": False,
            "packing_is_excluded_from_science_identity": True,
            "mixed_exact500_parent_count_by_lane": {
                str(key): value
                for key, value in MIXED_PARENT_COUNTS_TOTAL.items()
            },
            "mixed_exact500_stage_partition": {
                stage: {
                    str(key): value for key, value in counts.items()
                }
                for stage, counts in MIXED_PARENT_COUNTS_BY_STAGE.items()
            },
            "mixed_exact500_logical_seed_count": sum(
                lane_count * parent_count
                for lane_count, parent_count in MIXED_PARENT_COUNTS_TOTAL.items()
            ),
            "current_empty_pool_capacity_snapshot_required_for_phase_b_only": True,
            "single_seed_science_rollout_requires_capacity_snapshot": False,
            "phase_b_capacity_snapshot_policy": (
                "recalculate_from_latest_inventory_then_fail_closed_on_"
                "snapshot_drift_before_scheduler_POST"
            ),
        },
        "bundle_bindings": bindings,
        "bundle_bindings_complete": binding_complete,
        "launch_eligible": binding_complete,
        "scheduler_submission_performed": False,
        "scheduler_POST_performed": False,
        "scheduler_cancel_performed": False,
        "publication_performed": False,
        "aedt_used": False,
        "fea_submission_performed": False,
        "existing_jobs_touched": False,
        "automatic_promotion_allowed": False,
    }
    stable["sha256"] = canonical_sha256(stable)
    return validate_plan(stable)


def validate_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on every logical seed and science/binding identity."""

    plan = dict(value or {})
    unsigned = {key: item for key, item in plan.items() if key != "sha256"}
    canary = plan.get("canary") or {}
    production = plan.get("production") or {}
    canary_children = canary.get("logical_children")
    production_children = production.get("logical_children")
    bindings = plan.get("bundle_bindings")
    binding_complete = (
        isinstance(bindings, dict)
        and set(bindings) == set(STAGE_ORDER)
        and all(
            len(str((bindings[stage] or {}).get("bundle_manifest_sha256") or ""))
            == 64
            and str((bindings[stage] or {}).get("remote_bundle") or "").startswith(
                "/"
            )
            for stage in STAGE_ORDER
        )
    )
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("sha256") != canonical_sha256(unsigned)
        or plan.get("topology_niche_contract_sha256")
        != validate_contract(topology_contract())["sha256"]
        or plan.get("stage_order") != list(STAGE_ORDER)
        or not isinstance(canary_children, list)
        or not isinstance(production_children, list)
        or canary.get("logical_seed_count") != 16
        or canary.get("stage_counts") != dict(CANARY_STAGE_COUNTS)
        or len(canary_children) != 16
        or production.get("active_target") != 500
        or production.get("initial_stage_quotas")
        != dict(INITIAL_PRODUCTION_STAGE_QUOTAS)
        or len(production_children) != 500
        or plan.get("logical_child_resource") != CHILD_RESOURCE
        or plan.get("bundle_bindings_complete") is not binding_complete
        or plan.get("launch_eligible") is not binding_complete
        or any(
            plan.get(field) is not False
            for field in (
                "scheduler_submission_performed",
                "scheduler_POST_performed",
                "scheduler_cancel_performed",
                "publication_performed",
                "aedt_used",
                "fea_submission_performed",
                "existing_jobs_touched",
                "automatic_promotion_allowed",
            )
        )
    ):
        raise RuntimeError("topology successor plan top-level seal mismatch")

    all_children = [*canary_children, *production_children]
    identities: set[tuple[str, int]] = set()
    dedupe_keys: set[str] = set()
    for child in all_children:
        if not isinstance(child, dict):
            raise RuntimeError("topology logical child is not an object")
        stable = {
            key: item
            for key, item in child.items()
            if key not in {"payload_sha256", "dedupe_key"}
        }
        stage_id = str(child.get("stage_id") or "")
        wave = str(child.get("wave") or "")
        seed = child.get("seed")
        if (
            stage_id not in SEED_WINDOWS
            or wave not in {"canary", "production"}
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or not SEED_WINDOWS[stage_id][0]
            <= seed
            < SEED_WINDOWS[stage_id][1]
            or child.get("fixed_primary_turns") != 6
            or child.get("optimizer_island_id")
            != OPTIMIZER_ISLAND_BY_STAGE[stage_id]
            or child.get("topology_niche_contract_sha256")
            != plan["topology_niche_contract_sha256"]
            or child.get("population") != 320
            or child.get("max_generations") != 300
            or child.get("resource") != CHILD_RESOURCE
            or child.get("payload_sha256") != canonical_sha256(stable)
            or child.get("dedupe_key")
            != (
                f"mft-final1000-topology-{wave}-{stage_id}-{seed}-"
                f"{child['payload_sha256'][:12]}"
            )
            or any(
                child.get(field) is not False
                for field in (
                    "fea_submission_approved",
                    "fea_submission_performed",
                    "aedt_used",
                    "automatic_promotion_allowed",
                )
            )
        ):
            raise RuntimeError("topology logical child science seal mismatch")
        identity = (stage_id, seed)
        if identity in identities or child["dedupe_key"] in dedupe_keys:
            raise RuntimeError("topology logical child identity is duplicated")
        identities.add(identity)
        dedupe_keys.add(child["dedupe_key"])

    for wave, children, expected_counts in (
        ("canary", canary_children, CANARY_STAGE_COUNTS),
        ("production", production_children, INITIAL_PRODUCTION_STAGE_QUOTAS),
    ):
        if any(item["wave"] != wave for item in children):
            raise RuntimeError("topology logical child wave drifted")
        observed = {
            stage: sum(item["stage_id"] == stage for item in children)
            for stage in STAGE_ORDER
        }
        if observed != dict(expected_counts):
            raise RuntimeError("topology logical child stage quota drifted")
    return plan


def decompose_capacity_histogram(
    histogram: Mapping[int | str, int], *, maximum_lane_count: int = 8
) -> dict[int, int]:
    """Split every pool fragment into deterministic 1..8-lane parents."""

    maximum_lane_count = int(maximum_lane_count)
    if not 1 <= maximum_lane_count <= 8:
        raise ValueError("capacity decomposition lane limit must be 1..8")
    result = {lane_count: 0 for lane_count in range(maximum_lane_count, 0, -1)}
    for raw_capacity, raw_count in histogram.items():
        capacity = int(raw_capacity)
        count = int(raw_count)
        if capacity < 1 or count < 0:
            raise ValueError("capacity histogram contains an invalid bucket")
        complete, remainder = divmod(capacity, maximum_lane_count)
        result[maximum_lane_count] += complete * count
        if remainder:
            result[remainder] += count
    return result


def validate_current_pool_capacity_snapshot(
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate the read-only capacity observation used by a dry run."""

    value = dict(snapshot or {})
    unsigned = dict(value)
    recorded = unsigned.pop("sha256", None)
    histogram = {
        int(key): int(item)
        for key, item in (value.get("logical_capacity_histogram") or {}).items()
    }
    observed_at = str(value.get("observed_at") or "")
    source_sha = str(value.get("source_snapshot_sha256") or "")
    decomposed = decompose_capacity_histogram(histogram)
    logical_capacity = sum(key * item for key, item in histogram.items())
    fixed_eight_capacity = 8 * decomposed[8]
    if (
        recorded != canonical_sha256(unsigned)
        or value.get("schema_version")
        != "mft-tier1-empty-pool-capacity-snapshot-v1"
        or not histogram
        or "T" not in observed_at
        or len(source_sha) != 64
        or any(character not in "0123456789abcdef" for character in source_sha)
        or value.get("logical_capacity") != logical_capacity
        or value.get("fixed_eight_lane_capacity") != fixed_eight_capacity
        or value.get("decomposed_parent_counts")
        != {str(key): item for key, item in decomposed.items()}
        or logical_capacity < 1
        or not 0 <= fixed_eight_capacity <= logical_capacity
        or value.get("scheduler_write_performed") is not False
    ):
        raise RuntimeError("current empty-pool capacity snapshot mismatch")
    return value


def mixed_parent_partition(
    logical_children: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Partition the exact500 production children without identity mutation."""

    children = [dict(item) for item in logical_children]
    if len(children) != 500:
        raise RuntimeError("mixed parent partition requires exact500 children")
    by_stage = {
        stage: [item for item in children if item.get("stage_id") == stage]
        for stage in STAGE_ORDER
    }
    if {stage: len(items) for stage, items in by_stage.items()} != dict(
        INITIAL_PRODUCTION_STAGE_QUOTAS
    ):
        raise RuntimeError("mixed parent stage quotas drifted")
    groups = []
    for stage in STAGE_ORDER:
        cursor = 0
        stage_children = by_stage[stage]
        for lane_count in range(8, 0, -1):
            for _ in range(MIXED_PARENT_COUNTS_BY_STAGE[stage][lane_count]):
                chunk = stage_children[cursor : cursor + lane_count]
                if len(chunk) != lane_count:
                    raise RuntimeError("mixed parent stage partition underfilled")
                groups.append({
                    "parent_ordinal": len(groups),
                    "stage_id": stage,
                    "lane_count": lane_count,
                    "requested_cpus": 4 * lane_count,
                    "logical_child_seeds": [item["seed"] for item in chunk],
                    "logical_child_payload_sha256": [
                        item["payload_sha256"] for item in chunk
                    ],
                    "logical_child_dedupe_keys": [
                        item["dedupe_key"] for item in chunk
                    ],
                })
                cursor += lane_count
        if cursor != len(stage_children):
            raise RuntimeError("mixed parent stage partition did not consume quota")
    identities = [
        (item["seed"], item["payload_sha256"], item["dedupe_key"])
        for item in children
    ]
    packed_identities = [
        identity
        for group in groups
        for identity in zip(
            group["logical_child_seeds"],
            group["logical_child_payload_sha256"],
            group["logical_child_dedupe_keys"],
        )
    ]
    # Groups are stage-major and the logical manifest is also stage-major.
    if packed_identities != identities:
        raise RuntimeError("mixed parent packing changed logical child identity")
    value = {
        "schema_version": "mft-tier1-final1000-mixed-parent-partition-v1",
        "logical_child_count": len(children),
        "parent_count": len(groups),
        "parent_count_by_lane": {
            str(key): item for key, item in MIXED_PARENT_COUNTS_TOTAL.items()
        },
        "stage_parent_count_by_lane": {
            stage: {str(key): item for key, item in counts.items()}
            for stage, counts in MIXED_PARENT_COUNTS_BY_STAGE.items()
        },
        "groups": groups,
        "logical_child_identity_sha256": canonical_sha256(identities),
        "science_identity_mutation": False,
        "scheduler_submission_performed": False,
    }
    value["sha256"] = canonical_sha256(value)
    return value


def pack_logical_children(
    logical_children: Sequence[Mapping[str, Any]],
    *,
    available_cpus: int,
    maximum_lane_count: int = 8,
) -> dict[str, Any]:
    """Create an operational packing view without changing child identity."""

    if (
        isinstance(available_cpus, bool)
        or int(available_cpus) < 4
        or isinstance(maximum_lane_count, bool)
        or not 1 <= int(maximum_lane_count) <= 8
    ):
        raise ValueError("variable parent packing resources are invalid")
    children = [dict(item) for item in logical_children]
    capacity = min(int(maximum_lane_count), int(available_cpus) // 4)
    if capacity < 1:
        raise RuntimeError("parent allocation cannot fit one logical child")
    groups = []
    for offset in range(0, len(children), capacity):
        chunk = children[offset : offset + capacity]
        groups.append({
            "parent_ordinal": len(groups),
            "lane_count": len(chunk),
            "requested_child_cpus": 4 * len(chunk),
            "unused_cpu_capacity": int(available_cpus) - 4 * len(chunk),
            "logical_child_payload_sha256": [
                item["payload_sha256"] for item in chunk
            ],
            "logical_child_seeds": [item["seed"] for item in chunk],
            "logical_child_dedupe_keys": [item["dedupe_key"] for item in chunk],
        })
    value = {
        "schema_version": PACKING_SCHEMA,
        "available_cpus_per_parent": int(available_cpus),
        "maximum_lane_count": int(maximum_lane_count),
        "logical_child_count": len(children),
        "groups": groups,
        "logical_child_identity_sha256": canonical_sha256([
            {
                "seed": item["seed"],
                "payload_sha256": item["payload_sha256"],
                "dedupe_key": item["dedupe_key"],
            }
            for item in children
        ]),
        "logical_payload_mutation": False,
        "scheduler_submission_performed": False,
    }
    value["sha256"] = canonical_sha256(value)
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        staged.write_bytes(
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        os.replace(staged, path)
    finally:
        if staged.exists():
            staged.unlink()


def render(
    output: Path,
    *,
    bindings: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[Path, Path]:
    plan = build_plan(bindings)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan_path = output / "successor-plan.json"
    _atomic_json(plan_path, {"rendered_at": _now(), "plan": plan})
    # The 44-CPU example proves variable 8+3 grouping while the science plan
    # and all eleven child identities remain unchanged.
    example_children = plan["production"]["logical_children"][:11]
    packing = pack_logical_children(
        example_children, available_cpus=44, maximum_lane_count=8
    )
    packing_path = output / "packing-44cpu-8-plus-3.json"
    _atomic_json(packing_path, packing)
    return plan_path, packing_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-render the Final1000 topology successor plan."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bindings", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    bindings = None
    if args.bindings is not None:
        bindings = json.loads(args.bindings.read_text(encoding="utf-8"))
    paths = render(args.output, bindings=bindings)
    print("\n".join(str(path) for path in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
