"""Sealed identities for deep fixed-turn Tier-1 crossover islands.

These islands change optimizer pressure and termination only.  Every terminal
population is replayed through the unchanged physical evaluator before any
result is persisted or considered by a downstream consumer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any


SCHEMA = "mft-tier1-deep-crossover-island-v1"
TERMINATION_STRATEGY = "fixed-n-gen-no-ftol-v1"
POPULATION = 320
FIXED_GENERATIONS = 300
INFERENCE_THREADS = 8
TASK_PRIORITY = -1
LEGACY_TASK_PRIORITY = -2
RESONANCE_SCALE_HZ = 150.0
ANCHOR_PHYSICAL_G_ABS_TOLERANCE = 1e-9
TURN_SPLIT_TOPOLOGIES = (28, 29, 30, 31, 32, 33)
TURN_SPLIT_PARENT_PAIRS = (
    (28, 31), (29, 32), (30, 33),
    (28, 33), (29, 31), (30, 32),
)
BASIN_TURN_SPLIT_TOPOLOGIES_N1_6 = (34, 35, 36, 37, 38, 39, 60)
BASIN_TURN_SPLIT_PARENT_PAIRS_N1_6 = (
    (34, 37),
    (35, 38),
    (36, 39),
    (37, 35),
    (38, 36),
    (39, 37),
    (34, 38),
    (37, 60),
    (39, 60),
    (35, 60),
    (34, 60),
)
BASIN_DONOR_LANES_N1_6 = {
    "llt_target_mean_q90": {
        "topologies_N2_main_N2_side": [[34, 26], [37, 23]],
        "selection_signal": (
            "minimum_Llt_robust_band_with_Llt_mean_target_27p5uH_"
            "and_q90_half_width_at_most_0p55uH"
        ),
        "target_mean_uH": 27.5,
        "maximum_q90_half_width_uH": 0.55,
    },
    "thermal_split": {
        "topologies_N2_main_N2_side": [
            [35, 25], [36, 24], [37, 23], [38, 22], [39, 21],
        ],
        "selection_signal": "minimum_all_target_robust_temperature",
    },
    "exact_size_no_side": {
        "topologies_N2_main_N2_side": [[60, 0]],
        "selection_signal": "exact_size_pass_before_other_constraints",
    },
}
TOPOLOGY_COORDINATE_NAME = "u_N2_side"
TOPOLOGY_COORDINATE_INDEX = 2
TOPOLOGY_INITIAL_COPIES_EACH = 4
TOPOLOGY_MINIMUM_SURVIVORS_EACH = 4
BASIN_TOPOLOGY_INITIAL_COPIES_EACH_N1_6 = 8
BASIN_TOPOLOGY_MINIMUM_SURVIVORS_EACH_N1_6 = 8
TOPOLOGY_MIGRATION_PERIOD_GENERATIONS = 5
TOPOLOGY_MIGRANTS_PER_EVENT = 6
EPSILON_INITIAL_NORMALIZED_POSITIVE_G_SUM = 20.0
EPSILON_DECAY_GENERATIONS = 160


@dataclass(frozen=True)
class DeepCrossoverIsland:
    island_id: str
    variant: str
    runtime_suffix: str
    namespace: str
    task_name_stem: str
    dedupe_namespace: str
    search_focus: str
    active_quota: int
    seed_start: int
    seed_window_end_exclusive: int
    fixed_primary_turns: int
    warm_family: str
    optimizer_llt_scale_uh: float
    optimizer_all_thermal_scale_c: float
    optimizer_resonance_allowance_hz: float | None
    optimizer_llt_allowance_uh: float | None


N1_5_BALANCED = DeepCrossoverIsland(
    island_id="n1-5-size-llt-allthermal-balanced",
    variant="fixed-n1-5-deep-balanced-crossover-v1",
    runtime_suffix="mft_tier1_nsga_n1_5_deep_balanced_t110_res15k_260719",
    namespace=(
        "fixed-n1-5-deep-allthermal2c-llt0p30-res150-"
        "p320-fixedg300-crosswarm64-v1"
    ),
    task_name_stem="mft-t1n5deep",
    dedupe_namespace="mft-tier1-fixed-n1-5-deep-balanced-crossover-nsga",
    search_focus=(
        "cross_size_resonance_near_thermal_with_resonance_llt_core_thermal_"
        "then_balance_all11_temperature_Llt_density_and_size"
    ),
    active_quota=24,
    seed_start=2_107_210_000,
    seed_window_end_exclusive=2_207_210_000,
    fixed_primary_turns=5,
    warm_family="n1-5-size-and-thermal-crossover-32x32-v3",
    optimizer_llt_scale_uh=0.30,
    optimizer_all_thermal_scale_c=2.0,
    optimizer_resonance_allowance_hz=None,
    optimizer_llt_allowance_uh=None,
)


def _n1_6(scale_token: str, scale: float, seed_start: int) -> DeepCrossoverIsland:
    return DeepCrossoverIsland(
        island_id=f"n1-6-resanchor-allthermal-{scale_token}",
        variant=f"fixed-n1-6-deep-allthermal-{scale_token}-v1",
        runtime_suffix=(
            f"mft_tier1_nsga_n1_6_deep_allthermal_{scale_token}_"
            "t110_res15k_260719"
        ),
        namespace=(
            f"fixed-n1-6-deep-allthermal{scale_token}-llt0p05-res150-"
            "eps150-p320-fixedg300-anchorwarm64-v1"
        ),
        task_name_stem=f"mft-t1n6d{scale_token.replace('p', '')}",
        dedupe_namespace=(
            f"mft-tier1-fixed-n1-6-deep-allthermal-{scale_token}-nsga"
        ),
        search_focus=(
            "preserve_resonance_Llt_anchor_by_rebalancing_all11_thermal_"
            f"scale_{scale_token}_and_disable_ftol_early_stop"
        ),
        active_quota=4,
        seed_start=seed_start,
        seed_window_end_exclusive=seed_start + 100_000_000,
        fixed_primary_turns=6,
        warm_family="n1-6-authenticated-anchor-islands-v1",
        optimizer_llt_scale_uh=0.05,
        optimizer_all_thermal_scale_c=scale,
        optimizer_resonance_allowance_hz=150.0,
        optimizer_llt_allowance_uh=0.05,
    )


DEEP_CROSSOVER_ISLANDS = (
    N1_5_BALANCED,
    _n1_6("2p50c", 2.5, 2_207_210_000),
    _n1_6("2p75c", 2.75, 2_307_210_000),
    _n1_6("3p00c", 3.0, 2_407_210_000),
)
BY_VARIANT = {island.variant: island for island in DEEP_CROSSOVER_ISLANDS}
BY_ID = {island.island_id: island for island in DEEP_CROSSOVER_ISLANDS}
TOTAL_ACTIVE_QUOTA = sum(item.active_quota for item in DEEP_CROSSOVER_ISLANDS)


def canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_anchor_current_replay(
    *, current_decoded: dict[str, Any], replayed_physical_g: dict[str, float],
    source_candidate: dict[str, Any], expected_decoded_sha256: str,
    expected_source_physical_g_sha256: str,
) -> dict[str, Any]:
    """Authenticate source provenance and gate an independent current replay."""
    source_decoded = source_candidate.get("decoded_params")
    source_g = source_candidate.get("constraint_G")
    physical_names_match = (
        isinstance(source_g, dict)
        and set(replayed_physical_g) == set(source_g)
    )
    physical_deltas = {
        name: abs(float(replayed_physical_g[name]) - float(source_g[name]))
        for name in replayed_physical_g if physical_names_match
    }
    maximum_physical_delta = max(physical_deltas.values(), default=float("inf"))
    classifications_match = physical_names_match and all(
        (float(replayed_physical_g[name]) <= 1e-9)
        == (float(source_g[name]) <= 1e-9)
        for name in replayed_physical_g
    )
    source_decoded_sha = canonical_sha(source_decoded)
    source_g_sha = canonical_sha(source_g)
    if (
        source_decoded_sha != expected_decoded_sha256
        or source_g_sha != expected_source_physical_g_sha256
        or not physical_names_match
        or not classifications_match
        or not math.isfinite(maximum_physical_delta)
        or any(
            not math.isfinite(float(value))
            for value in replayed_physical_g.values()
        )
    ):
        physical_differences = sorted(
            key for key in set(replayed_physical_g) | set(source_g or {})
            if replayed_physical_g.get(key) != (source_g or {}).get(key)
        )
        physical_delta = {
            key: [replayed_physical_g.get(key), (source_g or {}).get(key)]
            for key in physical_differences
        }
        raise RuntimeError(
            "anchor current physical replay gate differs from source class: "
            f"source_decoded_sha={source_decoded_sha},"
            f"physical_sha={canonical_sha(replayed_physical_g)},"
            f"physical_fields={physical_differences},"
            f"physical_values={physical_delta}"
        )
    return {
        "source_decoded_params_sha256": source_decoded_sha,
        "current_replayed_decoded_params_sha256": canonical_sha(
            current_decoded
        ),
        "source_physical_constraint_G_sha256": source_g_sha,
        "replayed_physical_constraint_G_sha256": canonical_sha(
            replayed_physical_g
        ),
        "source_decoded_params": source_decoded,
        "current_replayed_decoded_params": current_decoded,
        "source_physical_constraint_G": source_g,
        "replayed_physical_constraint_G": replayed_physical_g,
        "maximum_observed_physical_constraint_G_absolute_drift": (
            maximum_physical_delta
        ),
        "physical_constraint_pass_classification_tolerance": 1e-9,
        "physical_constraint_pass_classification_identical": True,
        "source_vs_current_physical_G_numerical_equality_claimed": False,
        "optimizer_deterministic_replay_claimed": False,
        "original_terminal_chromosome_authority": False,
    }


def topology_evolution_contract(fixed_primary_turns: int) -> dict[str, Any]:
    fixed_primary_turns = int(fixed_primary_turns)
    if fixed_primary_turns not in (5, 6):
        raise ValueError("deep crossover topology evolution requires N1=5 or 6")
    if fixed_primary_turns == 6:
        topologies = BASIN_TURN_SPLIT_TOPOLOGIES_N1_6
        parent_pairs = BASIN_TURN_SPLIT_PARENT_PAIRS_N1_6
        initial_copies = BASIN_TOPOLOGY_INITIAL_COPIES_EACH_N1_6
        minimum_survivors = BASIN_TOPOLOGY_MINIMUM_SURVIVORS_EACH_N1_6
        donor_lanes = BASIN_DONOR_LANES_N1_6
        unavailable_topologies: list[int] = []
    else:
        topologies = TURN_SPLIT_TOPOLOGIES
        parent_pairs = TURN_SPLIT_PARENT_PAIRS
        initial_copies = TOPOLOGY_INITIAL_COPIES_EACH
        minimum_survivors = TOPOLOGY_MINIMUM_SURVIVORS_EACH
        donor_lanes = {
            "legacy_turn_split": {
                "topologies_N2_main_N2_side": [
                    [topology, 50 - topology] for topology in topologies
                ],
                "selection_signal": "legacy_deep_crossover",
            }
        }
        unavailable_topologies = []
    protected_slots = minimum_survivors * len(topologies)
    maximum_single_topology = POPULATION - minimum_survivors * (
        len(topologies) - 1
    )
    value = {
        "schema_version": "mft-tier1-basin-aware-turn-split-evolution-v2",
        "fixed_primary_turns": fixed_primary_turns,
        "secondary_total_turns": fixed_primary_turns * 10,
        "requested_global_turn_split_N2_main": list(topologies),
        "turn_split_sub_islands_N2_main": list(topologies),
        "unavailable_requested_topologies_due_pinned_physics_repair": (
            unavailable_topologies
        ),
        "pinned_repair_N2_side_upper_fraction": 0.52,
        "basin_donor_lanes": donor_lanes,
        "basin_donor_lanes_are_coordinate_only": True,
        "source_prediction_or_pass_classification_inherited": False,
        "turn_split_parent_pair_schedule": [
            list(pair) for pair in parent_pairs
        ],
        "cross_lane_parent_pairs": [
            list(pair)
            for pair in parent_pairs
            if fixed_primary_turns == 6
            and (60 in pair or pair in ((34, 38), (37, 35), (39, 37)))
        ],
        "coordinate_name": TOPOLOGY_COORDINATE_NAME,
        "coordinate_index": TOPOLOGY_COORDINATE_INDEX,
        "initial_repaired_copies_per_sub_island": (
            initial_copies
        ),
        "paired_mating": "cross_distinct_turn_split_sub_islands_every_generation",
        "migration": {
            "kind": "copy_elite_genome_then_change_only_u_N2_side",
            "period_generations": TOPOLOGY_MIGRATION_PERIOD_GENERATIONS,
            "first_evolution_generation_included": True,
            "migrants_per_event": len(topologies),
            "target_cycle": list(topologies),
            "offspring_physics_repair_required": True,
        },
        "survival": {
            "kind": (
                "normalized_positive_G_sum_epsilon_then_"
                "positive_count_max_sum_then_rank_crowding"
            ),
            "initial_epsilon": EPSILON_INITIAL_NORMALIZED_POSITIVE_G_SUM,
            "decay_to_zero_generation": EPSILON_DECAY_GENERATIONS,
            "terminal_epsilon": 0.0,
            "epsilon_feasibility": (
                "sum_normalized_positive_G_less_than_or_equal_to_epsilon"
            ),
            "infeasible_order": (
                "positive_constraint_count_then_max_normalized_positive_G_"
                "then_sum_normalized_positive_G"
            ),
            "physical_G_mutation": False,
            "objective_mutation": False,
            "minimum_survivors_per_turn_split_sub_island": (
                minimum_survivors
            ),
        },
        "bounded_diversity_budget": {
            "population": POPULATION,
            "protected_topology_count": len(topologies),
            "minimum_survivors_each": minimum_survivors,
            "protected_slots": protected_slots,
            "protected_population_fraction": protected_slots / POPULATION,
            "maximum_single_protected_topology_count": (
                maximum_single_topology
            ),
            "maximum_single_protected_topology_fraction": (
                maximum_single_topology / POPULATION
            ),
            "additional_model_evaluations": 0,
            "population_change": 0,
            "generation_change": 0,
            "scheduler_task_or_resource_change": False,
        },
        "minimum_evolution_generations": FIXED_GENERATIONS,
        "ftol_early_stop_allowed": False,
        "physical_constraint_G_mutation": False,
        "physical_objective_mutation": False,
        "model_or_training_data_provenance_mutation": False,
        "capacitance_label_precision_remediation_included": False,
        "terminal_physical_replay_required": True,
        "warm_donor_prediction_inheritance_allowed": False,
    }
    value["sha256"] = canonical_sha(value)
    return value


def island_profile(island: DeepCrossoverIsland) -> dict[str, Any]:
    profile = {
        "schema_version": SCHEMA,
        **asdict(island),
        "population": POPULATION,
        "fixed_generations": FIXED_GENERATIONS,
        "inference_threads": INFERENCE_THREADS,
        "task_priority": TASK_PRIORITY,
        "legacy_task_priority": LEGACY_TASK_PRIORITY,
        "optimizer_resonance_scale_Hz": RESONANCE_SCALE_HZ,
        "optimizer_termination_strategy": TERMINATION_STRATEGY,
        "topology_evolution_contract": topology_evolution_contract(
            island.fixed_primary_turns
        ),
        "physical_hard_spec_mutation": False,
        "objective_mutation": False,
        "authoritative_terminal_G": "physical_unscaled_unchanged",
        "automatic_promotion_allowed": False,
    }
    profile["sha256"] = canonical_sha(profile)
    return profile


def validate_inventory() -> None:
    if len(BY_VARIANT) != len(DEEP_CROSSOVER_ISLANDS):
        raise RuntimeError("deep crossover variants are not unique")
    if len(BY_ID) != len(DEEP_CROSSOVER_ISLANDS):
        raise RuntimeError("deep crossover island ids are not unique")
    for field in (
        "runtime_suffix", "namespace", "task_name_stem", "dedupe_namespace",
    ):
        values = [getattr(item, field) for item in DEEP_CROSSOVER_ISLANDS]
        if len(values) != len(set(values)):
            raise RuntimeError(f"deep crossover {field} values overlap")
    windows = sorted(
        (item.seed_start, item.seed_window_end_exclusive, item.variant)
        for item in DEEP_CROSSOVER_ISLANDS
    )
    for start, end, variant in windows:
        if not 0 <= start < end <= 2**32:
            raise RuntimeError(f"deep crossover seed window is invalid: {variant}")
    for left, right in zip(windows, windows[1:]):
        if left[1] > right[0]:
            raise RuntimeError("deep crossover seed windows overlap")
    for island in DEEP_CROSSOVER_ISLANDS:
        pair = (
            island.optimizer_resonance_allowance_hz,
            island.optimizer_llt_allowance_uh,
        )
        if (pair[0] is None) != (pair[1] is None):
            raise RuntimeError("deep crossover allowance pair is incomplete")


validate_inventory()
