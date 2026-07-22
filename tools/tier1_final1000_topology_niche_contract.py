"""Sealed optimizer-only topology niches for the Final1000 N1=6 search.

The predecessor protected eight rows for each secondary turn split and then
let ordinary NSGA survival fill the remaining 264 rows.  In production that
made ``N2_main=60`` occupy 272/320 rows.  This contract assigns the complete
population to independent topology niches.  It changes neither the physical
constraint vector nor either objective; terminal candidates are still replayed
with the unchanged physical evaluator.

This module contains no scheduler, AEDT, FEA, publication, or network code.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping


SCHEMA = "mft-tier1-final1000-topology-niche-v1"
WARM_PARTITION_SCHEMA = "mft-tier1-final1000-topology-warm-partition-v1"
CANARY_SCHEMA = "mft-tier1-final1000-topology-canary-plan-v1"
POPULATION = 320
FIXED_PRIMARY_TURNS = 6
TOPOLOGY_COORDINATE_INDEX = 2

# The complete population is reserved.  Consequently no topology can consume
# an unassigned remainder and the sentinel topology can never dominate again.
TOPOLOGY_QUOTA_BY_N2_MAIN = {
    34: 32,
    35: 48,
    36: 80,
    37: 104,
    38: 32,
    39: 16,
    60: 8,
}

# The authenticated half is selected from terminal physical replay.  The last
# 16 rows are deterministic crossover donors and are repaired by the same
# Current7 physics projection before the optimizer evaluates them.
WARM_SOURCE_GROUPS = {
    "llt_thermal_37": {"count": 48, "topology_counts": {37: 48}},
    "resonance_size_34_35": {
        "count": 48,
        "topology_counts": {34: 24, 35: 24},
    },
    "transition_36": {"count": 32, "topology_counts": {36: 32}},
    "boundary_38_39": {
        "count": 16,
        "topology_counts": {38: 8, 39: 8},
    },
    "repaired_crossover": {
        "count": 16,
        "topology_counts": {34: 4, 35: 4, 36: 2, 37: 2, 38: 2, 39: 2},
    },
}

WARM_TOPOLOGY_COUNTS = {
    topology: sum(
        int(group["topology_counts"].get(topology, 0))
        for group in WARM_SOURCE_GROUPS.values()
    )
    for topology in TOPOLOGY_QUOTA_BY_N2_MAIN
}
FRESH_SOBOL_TOPOLOGY_COUNTS = {
    topology: TOPOLOGY_QUOTA_BY_N2_MAIN[topology]
    - WARM_TOPOLOGY_COUNTS[topology]
    for topology in TOPOLOGY_QUOTA_BY_N2_MAIN
}

# Per 160 parent pairs (the production NSGA2 population produces two
# children/pair).  N36 x N37 is deliberately the largest single mating lane;
# exact per-topology survival, not mating isolation, prevents collapse.
PARENT_PAIR_COUNTS_PER_160 = {
    "36x37": 64,
    "34x34": 12,
    "35x35": 16,
    "36x36": 18,
    "37x37": 30,
    "38x38": 10,
    "39x39": 6,
    "60x60": 4,
}

# Piecewise-linear, optimizer-only allowances in physical units.  Every
# schedule is zero before the terminal generation, so the terminal optimizer
# snapshot and the authoritative unscaled physical replay are identical.
ALLOWANCE_SCHEDULES = {
    34: {
        "temperature_C": [[0, 40.0], [80, 25.0], [160, 10.0], [240, 0.0]],
        "Llt_uH": [[0, 1.75], [80, 1.0], [160, 0.4], [240, 0.0]],
        "resonance_Hz": [[0, 0.0], [240, 0.0]],
    },
    35: {
        "temperature_C": [[0, 40.0], [80, 25.0], [160, 10.0], [240, 0.0]],
        "Llt_uH": [[0, 1.75], [80, 1.0], [160, 0.4], [240, 0.0]],
        "resonance_Hz": [[0, 0.0], [240, 0.0]],
    },
    36: {
        "temperature_C": [[0, 20.0], [80, 12.5], [160, 5.0], [240, 0.0]],
        "Llt_uH": [[0, 0.875], [80, 0.5], [160, 0.2], [240, 0.0]],
        "resonance_Hz": [[0, 1500.0], [80, 1000.0], [160, 500.0], [200, 150.0], [240, 0.0]],
    },
    37: {
        "temperature_C": [[0, 0.0], [240, 0.0]],
        "Llt_uH": [[0, 0.0], [240, 0.0]],
        "resonance_Hz": [
            [0, 3000.0],
            [40, 2500.0],
            [80, 2000.0],
            [120, 1000.0],
            [160, 500.0],
            [200, 150.0],
            [240, 0.0],
        ],
    },
    38: {
        "temperature_C": [[0, 20.0], [80, 12.5], [160, 5.0], [240, 0.0]],
        "Llt_uH": [[0, 0.875], [80, 0.5], [160, 0.2], [240, 0.0]],
        "resonance_Hz": [[0, 1500.0], [80, 1000.0], [160, 500.0], [200, 150.0], [240, 0.0]],
    },
    39: {
        "temperature_C": [[0, 0.0], [240, 0.0]],
        "Llt_uH": [[0, 0.0], [240, 0.0]],
        "resonance_Hz": [[0, 0.0], [240, 0.0]],
    },
    60: {
        "temperature_C": [[0, 0.0], [240, 0.0]],
        "Llt_uH": [[0, 0.0], [240, 0.0]],
        "resonance_Hz": [[0, 0.0], [240, 0.0]],
    },
}

CANARY_STAGE_COUNTS = {
    "entry-1200-t125": 6,
    "bridge-1150-t115": 6,
    "close-1075-t107p5": 2,
    "final-1000-t100": 2,
}
INITIAL_PRODUCTION_STAGE_QUOTAS = {
    "entry-1200-t125": 300,
    "bridge-1150-t115": 150,
    "close-1075-t107p5": 40,
    "final-1000-t100": 10,
}


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _interpolate(points: list[list[float]], generation: int) -> float:
    generation = max(0, int(generation))
    if generation <= int(points[0][0]):
        return float(points[0][1])
    for left, right in zip(points, points[1:]):
        x0, y0 = int(left[0]), float(left[1])
        x1, y1 = int(right[0]), float(right[1])
        if generation <= x1:
            fraction = (generation - x0) / (x1 - x0)
            return y0 + fraction * (y1 - y0)
    return float(points[-1][1])


def optimizer_allowances(n2_main: int, generation: int) -> dict[str, float]:
    """Return finite optimizer-only allowances for one topology/generation."""

    schedule = ALLOWANCE_SCHEDULES[int(n2_main)]
    value = {
        name: _interpolate(points, generation)
        for name, points in schedule.items()
    }
    if any(not math.isfinite(item) or item < 0.0 for item in value.values()):
        raise RuntimeError("topology allowance schedule produced an invalid value")
    return value


def contract() -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": SCHEMA,
        "fixed_primary_turns": FIXED_PRIMARY_TURNS,
        "population": POPULATION,
        "coordinate_name": "u_N2_side",
        "coordinate_index": TOPOLOGY_COORDINATE_INDEX,
        "topology_quota_by_N2_main": {
            str(key): value for key, value in TOPOLOGY_QUOTA_BY_N2_MAIN.items()
        },
        "quota_sum": sum(TOPOLOGY_QUOTA_BY_N2_MAIN.values()),
        "survival_policy": (
            "independent_topology_rank_crowding_then_exact_quota_every_generation"
        ),
        "mating_policy": (
            "priority_N36xN37_cross_then_topology_local_pairs_with_exact_"
            "quota_survival_v1"
        ),
        "parent_pair_counts_per_160": dict(PARENT_PAIR_COUNTS_PER_160),
        "N36xN37_cross_pair_fraction": 0.4,
        "cross_offspring_topology_attribution": (
            "alternate_N36_N37_then_current7_physics_repair"
        ),
        "evidence_driven_search_priority": {
            "audited_terminal_completion_count": 11_555,
            "audited_physically_feasible_count": 0,
            "highest_priority_stage": "entry-1200-t125",
            "highest_priority_crossover": "N2_main_36_x_N2_main_37",
            "warm_repaired_crossover_parents": [36, 37],
        },
        "generation_count_audit_required": True,
        "initialization": {
            "authenticated_warm_count": sum(WARM_TOPOLOGY_COUNTS.values()),
            "fresh_sobol_count": sum(FRESH_SOBOL_TOPOLOGY_COUNTS.values()),
            "warm_source_groups": {
                name: {
                    "count": int(group["count"]),
                    "topology_counts": {
                        str(key): int(item)
                        for key, item in group["topology_counts"].items()
                    },
                }
                for name, group in WARM_SOURCE_GROUPS.items()
            },
            "warm_topology_counts": {
                str(key): value for key, value in WARM_TOPOLOGY_COUNTS.items()
            },
            "fresh_sobol_topology_counts": {
                str(key): value
                for key, value in FRESH_SOBOL_TOPOLOGY_COUNTS.items()
            },
            "warm_N2_main_60_allowed": False,
            "fresh_N2_main_60_sentinel_count": 8,
            "same_current7_physics_repair_required": True,
        },
        "optimizer_allowance_schedules": {
            str(topology): schedule
            for topology, schedule in ALLOWANCE_SCHEDULES.items()
        },
        "allowance_interpolation": "piecewise_linear_by_evolution_generation",
        "allowance_units": {
            "resonance_Hz": "Hz",
            "temperature_C": "degC",
            "Llt_uH": "uH",
        },
        "allowances_zero_from_generation": 240,
        "terminal_generation": 300,
        "terminal_physical_G_mutation": False,
        "physical_constraint_G_mutation": False,
        "physical_objective_mutation": False,
        "terminal_physical_replay_required": True,
        "surrogate_only": True,
        "aedt_used": False,
        "fea_submission_performed": False,
        "scheduler_submission_performed": False,
        "automatic_promotion_allowed": False,
    }
    value["sha256"] = canonical_sha256(value)
    return value


def validate_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    unsigned = dict(result)
    recorded = unsigned.pop("sha256", None)
    quotas = {
        int(key): int(item)
        for key, item in (result.get("topology_quota_by_N2_main") or {}).items()
    }
    initialization = result.get("initialization") or {}
    priority = result.get("evidence_driven_search_priority") or {}
    if (
        recorded != canonical_sha256(unsigned)
        or result.get("schema_version") != SCHEMA
        or result.get("fixed_primary_turns") != FIXED_PRIMARY_TURNS
        or result.get("population") != POPULATION
        or quotas != TOPOLOGY_QUOTA_BY_N2_MAIN
        or sum(quotas.values()) != POPULATION
        or result.get("parent_pair_counts_per_160")
        != PARENT_PAIR_COUNTS_PER_160
        or sum(PARENT_PAIR_COUNTS_PER_160.values()) != 160
        or result.get("N36xN37_cross_pair_fraction") != 0.4
        or initialization.get("authenticated_warm_count") != 160
        or initialization.get("fresh_sobol_count") != 160
        or initialization.get("warm_N2_main_60_allowed") is not False
        or initialization.get("fresh_N2_main_60_sentinel_count") != 8
        or priority.get("audited_terminal_completion_count") != 11_555
        or priority.get("audited_physically_feasible_count") != 0
        or priority.get("highest_priority_crossover")
        != "N2_main_36_x_N2_main_37"
        or priority.get("warm_repaired_crossover_parents") != [36, 37]
        or result.get("allowances_zero_from_generation") != 240
        or any(
            any(abs(item) > 0.0 for item in optimizer_allowances(topology, 240).values())
            for topology in quotas
        )
        or result.get("terminal_physical_G_mutation") is not False
        or result.get("physical_constraint_G_mutation") is not False
        or result.get("physical_objective_mutation") is not False
    ):
        raise RuntimeError("Final1000 topology niche contract mismatch")
    return result


validate_contract(contract())
