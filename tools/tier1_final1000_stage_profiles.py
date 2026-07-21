"""Sealed parallel search profiles for the Tier-1 final 1000 mm goal.

The profiles deliberately form concurrent *search* islands, not an automatic
promotion pipeline.  A relaxed entry island can discover warm donors while
the final island searches the requested envelope from the first wave.  Every
island uses the same 15--20 kHz half-magnetizing self-resonance band and fixes
the primary turn count to six.  Only the box and robust temperature limit are
staircased.

This module is optimizer-only metadata.  It cannot submit Slurm work, start
AEDT, approve FEA, or promote a surrogate candidate to a verified design.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping


SCHEMA = "mft-tier1-final1000-stage-profile-v1"
INVENTORY_SCHEMA = "mft-tier1-final1000-stage-inventory-v1"
WARM_POOL_SCHEMA = "mft-tier1-final1000-warm-pool-input-v1"
GOAL_ID = "mft-tier1-1000x1000x750-res15to20k-t100-n1-6-v1"

RESONANCE_MIN_HZ = 15_000.0
RESONANCE_MAX_HZ = 20_000.0
HEIGHT_MAX_MM = 750.0
FIXED_PRIMARY_TURNS = 6
POPULATION = 320
FIXED_GENERATIONS = 200
INFERENCE_THREADS = 8
TOTAL_ACTIVE_QUOTA = 500
WARM_POOL_SIZE = 64
WARM_REPLAY_QUOTA = 48
WARM_REPAIRED_GEOMETRY_QUOTA = 16

# These keys intentionally match the dynamic stage-spec CLI/result contract.
BASE_HARD_SPEC: dict[str, float | int] = {
    "Llt_target_uH": 27.5,
    "Llt_tol_uH": 0.55,
    "T_limit_C": 100.0,
    "B_limit_T": 1.2,
    "insulation_min_mm": 40.0,
    "q_sigma": 1.0,
    "n_core_group_max": 4.0,
    "primary_conductor_thickness_mm": 5.0,
    "resonance_min_Hz": RESONANCE_MIN_HZ,
    "resonance_max_Hz": RESONANCE_MAX_HZ,
    "magnetizing_inductance_factor": 0.5,
    "size_W_max_mm": 1_000.0,
    "size_L_max_mm": 1_000.0,
    "size_H_max_mm": HEIGHT_MAX_MM,
}

RESONANCE_CONSTRAINT_NAMES = (
    "half_magnetizing_resonance_minimum",
    "half_magnetizing_resonance_maximum",
)


@dataclass(frozen=True)
class FinalGoalStage:
    stage_id: str
    ui_alias: str
    variant: str
    task_name_stem: str
    dedupe_namespace: str
    seed_start: int
    seed_window_end_exclusive: int
    active_quota: int
    size_W_max_mm: float
    size_L_max_mm: float
    size_H_max_mm: float
    robust_temperature_limit_C: float
    optimizer_size_scale_mm: float
    optimizer_all_thermal_scale_C: float
    search_focus: str


STAGES = (
    FinalGoalStage(
        stage_id="entry-1200-t125",
        ui_alias="1200 mm / 125 C entry",
        variant="final1000-n1-6-entry-1200-t125-v1",
        task_name_stem="mft-t1fg-e",
        dedupe_namespace="mft-tier1-final1000-entry-nsga",
        seed_start=2_207_500_000,
        seed_window_end_exclusive=2_257_500_000,
        active_quota=64,
        size_W_max_mm=1_200.0,
        size_L_max_mm=1_200.0,
        size_H_max_mm=HEIGHT_MAX_MM,
        robust_temperature_limit_C=125.0,
        optimizer_size_scale_mm=40.0,
        optimizer_all_thermal_scale_C=2.5,
        search_focus=(
            "recover_band_feasible_N1_6_donors_then_reduce_core_thermal_"
            "and_exterior_size"
        ),
    ),
    FinalGoalStage(
        stage_id="bridge-1150-t115",
        ui_alias="1150 mm / 115 C bridge",
        variant="final1000-n1-6-bridge-1150-t115-v1",
        task_name_stem="mft-t1fg-b",
        dedupe_namespace="mft-tier1-final1000-bridge-nsga",
        seed_start=2_307_500_000,
        seed_window_end_exclusive=2_357_500_000,
        active_quota=96,
        size_W_max_mm=1_150.0,
        size_L_max_mm=1_150.0,
        size_H_max_mm=HEIGHT_MAX_MM,
        robust_temperature_limit_C=115.0,
        optimizer_size_scale_mm=25.0,
        optimizer_all_thermal_scale_C=2.75,
        search_focus=(
            "joint_size_and_all_current7_robust_thermal_pressure_with_"
            "Llt_and_resonance_band_preserved"
        ),
    ),
    FinalGoalStage(
        stage_id="close-1075-t107p5",
        ui_alias="1075 mm / 107.5 C close",
        variant="final1000-n1-6-close-1075-t107p5-v1",
        task_name_stem="mft-t1fg-c",
        dedupe_namespace="mft-tier1-final1000-close-nsga",
        seed_start=2_407_500_000,
        seed_window_end_exclusive=2_457_500_000,
        active_quota=128,
        size_W_max_mm=1_075.0,
        size_L_max_mm=1_075.0,
        size_H_max_mm=HEIGHT_MAX_MM,
        robust_temperature_limit_C=107.5,
        optimizer_size_scale_mm=15.0,
        optimizer_all_thermal_scale_C=3.0,
        search_focus=(
            "near_goal_size_thermal_crossover_with_dense_N2_turn_split_"
            "migration"
        ),
    ),
    FinalGoalStage(
        stage_id="final-1000-t100",
        ui_alias="1000 mm / 100 C final",
        variant="final1000-n1-6-final-1000-t100-v1",
        task_name_stem="mft-t1fg-f",
        dedupe_namespace="mft-tier1-final1000-final-nsga",
        seed_start=2_457_500_000,
        seed_window_end_exclusive=2_507_000_000,
        active_quota=212,
        size_W_max_mm=1_000.0,
        size_L_max_mm=1_000.0,
        size_H_max_mm=HEIGHT_MAX_MM,
        robust_temperature_limit_C=100.0,
        optimizer_size_scale_mm=10.0,
        optimizer_all_thermal_scale_C=3.0,
        search_focus=(
            "direct_final_envelope_search_prioritizing_all_current7_robust_"
            "temperature_and_both_exterior_axes"
        ),
    ),
)

BY_ID = {stage.stage_id: stage for stage in STAGES}
BY_VARIANT = {stage.variant: stage for stage in STAGES}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def hard_spec(stage: FinalGoalStage) -> dict[str, float | int]:
    """Return the exact dynamic evaluator spec for one search stage."""

    value = dict(BASE_HARD_SPEC)
    value.update(
        {
            "T_limit_C": float(stage.robust_temperature_limit_C),
            "size_W_max_mm": float(stage.size_W_max_mm),
            "size_L_max_mm": float(stage.size_L_max_mm),
            "size_H_max_mm": float(stage.size_H_max_mm),
        }
    )
    return value


def warm_pool_input_contract(stage: FinalGoalStage) -> dict[str, Any]:
    """Describe deterministic inputs/selection for a future 48+16 warm pool.

    Terminal X/G alone cannot be re-ranked against a different size,
    temperature, or upper-resonance bound unless the replayed physical metrics
    are retained.  This contract makes that missing handoff explicit and lets
    an offline replay producer generate an authenticated input without the
    launcher guessing from the old 1200/110/minimum-only G vector.
    """

    spec = hard_spec(stage)
    value = {
        "schema_version": WARM_POOL_SCHEMA,
        "stage_id": stage.stage_id,
        "stage_spec_sha256": canonical_sha256(spec),
        "coordinate_dimension": 25,
        "required_terminal_replay_fields": [
            "unit_coordinate",
            "decoded_params_sha256",
            "fixed_primary_turns",
            "exterior_W_mm",
            "exterior_L_mm",
            "exterior_H_mm",
            "robust_max_temperature_C",
            "f_res_tx_half_Lm_Hz",
            "f_res_rx_half_Lm_Hz",
            "physical_constraint_G",
            "retained_constraint_scales",
            "source_result_sha256",
            "source_terminal_row_index",
        ],
        "terminal_filter": {
            "fixed_primary_turns": FIXED_PRIMARY_TURNS,
            "finite_unit_coordinate_in_closed_0_1": True,
            "terminal_physical_replay_attested": True,
            "dedupe": "decoded_params_sha256_then_rounded_unit_coordinate_1e-12",
        },
        "dynamic_G_replay": {
            "exterior_width_limit": (
                f"exterior_W_mm-{spec['size_W_max_mm']:g}"
            ),
            "exterior_length_limit": (
                f"exterior_L_mm-{spec['size_L_max_mm']:g}"
            ),
            "exterior_height_limit": (
                f"exterior_H_mm-{spec['size_H_max_mm']:g}"
            ),
            "temperature_robust_limit_max": (
                f"robust_max_temperature_C-{spec['T_limit_C']:g}"
            ),
            "half_magnetizing_resonance_minimum": (
                "15000-min(f_res_tx_half_Lm_Hz,f_res_rx_half_Lm_Hz)"
            ),
            "half_magnetizing_resonance_maximum": (
                "min(f_res_tx_half_Lm_Hz,f_res_rx_half_Lm_Hz)-"
                "nextafter(20000,-inf)"
            ),
            "old_dynamic_G_must_not_be_reused": True,
        },
        "normalized_positive_G_scales": {
            "exterior_width_limit_mm": stage.optimizer_size_scale_mm,
            "exterior_length_limit_mm": stage.optimizer_size_scale_mm,
            "exterior_height_limit_mm": stage.optimizer_size_scale_mm,
            "temperature_robust_limit_C": stage.optimizer_all_thermal_scale_C,
            "half_magnetizing_resonance_minimum_Hz": 150.0,
            "half_magnetizing_resonance_maximum_Hz": 150.0,
            "retained_constraints": "authenticated_retained_constraint_scales",
        },
        "deterministic_rank": [
            "positive_physical_constraint_count_ascending",
            "maximum_normalized_positive_G_ascending",
            "sum_normalized_positive_G_ascending",
            "dynamic_normalized_positive_G_sum_ascending",
            "decoded_params_sha256_ascending",
            "source_result_sha256_ascending",
            "source_terminal_row_index_ascending",
        ],
        "diversity": {
            "candidate_window_after_rank": 1024,
            "algorithm": "greedy_farthest_unit_euclidean",
            "quality_penalty": 0.05,
            "tie_break": "deterministic_rank_then_coordinate_sha256",
        },
        "output_mix": {
            "shape": [WARM_POOL_SIZE, 25],
            "terminal_replay_N1_6": WARM_REPLAY_QUOTA,
            "independently_repaired_geometry_N1_6": (
                WARM_REPAIRED_GEOMETRY_QUOTA
            ),
            "interleave": "three_terminal_replay_then_one_repaired_geometry",
            "decoded_params_sha256_duplicate_cap": 1,
        },
        "repaired_geometry_input_contract": {
            "artifact": "float64_npy_Nx25_unit_coordinates",
            "fixed_primary_turns": FIXED_PRIMARY_TURNS,
            "same_current7_problem_repair_required": True,
            "post_repair_coordinate_sha256_required": True,
            "repair_contract_sha256_required": True,
            "unrepaired_or_N1_other_than_6_rejected": True,
        },
        "source_mutation_allowed": False,
        "scheduler_write_performed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    value["sha256"] = canonical_sha256(value)
    return value


def stage_profile(stage: FinalGoalStage) -> dict[str, Any]:
    spec = hard_spec(stage)
    profile = {
        "schema_version": SCHEMA,
        **asdict(stage),
        "goal_id": GOAL_ID,
        "hard_spec": spec,
        "stage_spec_sha256": canonical_sha256(spec),
        "warm_pool_input_contract": warm_pool_input_contract(stage),
        "resonance_band": {
            "quantity": "min(f_res_tx_half_Lm_Hz,f_res_rx_half_Lm_Hz)",
            "lower_Hz": RESONANCE_MIN_HZ,
            "lower_inclusive": True,
            "upper_Hz": RESONANCE_MAX_HZ,
            "upper_exclusive": True,
            "lower_constraint_name": RESONANCE_CONSTRAINT_NAMES[0],
            "upper_constraint_name": RESONANCE_CONSTRAINT_NAMES[1],
            "lower_G_formula": "resonance_min_Hz-min(f_tx,f_rx)",
            "upper_G_formula": (
                "min(f_tx,f_rx)-nextafter(resonance_max_Hz,-inf)"
            ),
            "feasible_rule": "both_G_le_zero_and_exact_upper_boundary_rejected",
        },
        "temperature_semantics": (
            "maximum_mu_plus_q90_conformal_half_width_across_current7_targets"
        ),
        "fixed_primary_turns": FIXED_PRIMARY_TURNS,
        "population": POPULATION,
        "fixed_generations": FIXED_GENERATIONS,
        "inference_threads": INFERENCE_THREADS,
        "parallel_launch_required": True,
        "predecessor_completion_required": False,
        "hard_constraint_allowance": 0.0,
        "terminal_physical_replay_required": True,
        "surrogate_only": True,
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    profile["sha256"] = canonical_sha256(profile)
    return profile


def stage_inventory() -> dict[str, Any]:
    profiles = [stage_profile(stage) for stage in STAGES]
    value = {
        "schema_version": INVENTORY_SCHEMA,
        "goal_id": GOAL_ID,
        "launch_mode": "all_stages_parallel_with_independent_refill_quotas",
        "stage_order_is_pressure_only_not_launch_dependency": True,
        "total_active_quota": TOTAL_ACTIVE_QUOTA,
        "fixed_primary_turns": FIXED_PRIMARY_TURNS,
        "profiles": profiles,
        "profile_sha256_by_stage": {
            profile["stage_id"]: profile["sha256"] for profile in profiles
        },
        "surrogate_only": True,
        "fea_submission_approved": False,
        "aedt_used": False,
    }
    value["sha256"] = canonical_sha256(value)
    return value


def validate_stage_spec(
    stage: FinalGoalStage, spec: Mapping[str, Any]
) -> dict[str, float | int]:
    expected = hard_spec(stage)
    if dict(spec) != expected:
        raise RuntimeError(f"{stage.stage_id} dynamic hard spec drifted")
    return expected


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be finite and positive")
    try:
        output = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{label} must be finite and positive") from exc
    if not math.isfinite(output) or output <= 0.0:
        raise RuntimeError(f"{label} must be finite and positive")
    return output


def validate_inventory() -> None:
    if len(STAGES) != 4 or len(BY_ID) != 4 or len(BY_VARIANT) != 4:
        raise RuntimeError("final1000 requires four uniquely identified stages")
    if sum(stage.active_quota for stage in STAGES) != TOTAL_ACTIVE_QUOTA:
        raise RuntimeError("final1000 active quota drifted")
    for field in ("task_name_stem", "dedupe_namespace"):
        values = [getattr(stage, field) for stage in STAGES]
        if len(values) != len(set(values)):
            raise RuntimeError(f"final1000 {field} values overlap")
    windows = sorted(
        (stage.seed_start, stage.seed_window_end_exclusive, stage.stage_id)
        for stage in STAGES
    )
    for start, end, stage_id in windows:
        if not 0 <= start < end <= 2**32:
            raise RuntimeError(f"invalid final1000 seed window: {stage_id}")
    for left, right in zip(windows, windows[1:]):
        if left[1] > right[0]:
            raise RuntimeError("final1000 seed windows overlap")

    widths = []
    lengths = []
    temperatures = []
    for stage in STAGES:
        spec = validate_stage_spec(stage, hard_spec(stage))
        widths.append(_finite_positive(spec["size_W_max_mm"], "width"))
        lengths.append(_finite_positive(spec["size_L_max_mm"], "length"))
        temperatures.append(_finite_positive(spec["T_limit_C"], "temperature"))
        if (
            spec["resonance_min_Hz"] != RESONANCE_MIN_HZ
            or spec["resonance_max_Hz"] != RESONANCE_MAX_HZ
            or spec["size_H_max_mm"] != HEIGHT_MAX_MM
            or stage.active_quota <= 0
        ):
            raise RuntimeError("final1000 immutable band/height/quota drifted")
        profile = stage_profile(stage)
        unsigned = {key: value for key, value in profile.items() if key != "sha256"}
        if profile["sha256"] != canonical_sha256(unsigned):
            raise RuntimeError("final1000 stage profile seal mismatch")
    if any(
        left <= right
        for left, right in zip(widths, widths[1:])
    ) or any(
        left <= right
        for left, right in zip(lengths, lengths[1:])
    ) or any(
        left <= right
        for left, right in zip(temperatures, temperatures[1:])
    ):
        raise RuntimeError("final1000 staircase must narrow monotonically")
    final = hard_spec(STAGES[-1])
    if (
        final["size_W_max_mm"] != 1_000.0
        or final["size_L_max_mm"] != 1_000.0
        or final["size_H_max_mm"] != 750.0
        or final["T_limit_C"] != 100.0
    ):
        raise RuntimeError("final1000 terminal stage no longer matches the goal")

    inventory = stage_inventory()
    unsigned_inventory = {
        key: value for key, value in inventory.items() if key != "sha256"
    }
    if inventory["sha256"] != canonical_sha256(unsigned_inventory):
        raise RuntimeError("final1000 inventory seal mismatch")


validate_inventory()
