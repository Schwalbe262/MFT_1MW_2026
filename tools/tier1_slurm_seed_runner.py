"""Run one authenticated Tier-1 NSGA-II seed in a CPU-only Slurm task."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

try:
    from tier1_n1_6_anchor_island_contract import (
        optimizer_allowance_contract,
    )
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_n1_6_anchor_island_contract import (
        optimizer_allowance_contract,
    )

try:
    from tier1_deep_crossover_contract import (
        BY_VARIANT as DEEP_CROSSOVER_BY_VARIANT,
        island_profile as deep_crossover_island_profile,
        topology_evolution_contract,
    )
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_deep_crossover_contract import (
        BY_VARIANT as DEEP_CROSSOVER_BY_VARIANT,
        island_profile as deep_crossover_island_profile,
        topology_evolution_contract,
    )


SCHEMA = "mft-tier1-slurm-seed-task-v3"
STATUS_SCHEMA = "mft-tier1-slurm-seed-status-v1"
ATTESTATION_SCHEMA = "mft-tier1-slurm-temperature-attestation-v3"
TEMPERATURE_CONTRACT_SCHEMA = "mft-tier1-temperature-constraint-contract-v3"
EXPECTED_TEMPERATURE_LIMIT_C = 110.0
TEMPERATURE_FORMULA = (
    "surrogate_mu_plus_q90_conformal_half_width_le_limit"
)
UNCERTAINTY_CONTRACT = "q90_conformal_half_width_physical_v1"
TEMPERATURE_TARGETS = (
    "T_max_Tx", "T_max_Rx_main", "T_max_Rx_side", "T_max_core",
    "Tprobe_Tx_leeward_max", "Tprobe_Rx_main_leeward_max",
    "Tprobe_Rx_side_leeward_max", "Tprobe_core_center_max",
    "Tprobe_core_center_leg_max", "Tprobe_core_side_leg_max",
    "Tprobe_core_top_yoke_max",
)
SIDE_TEMPERATURE_TARGETS = (
    "T_max_Rx_side", "Tprobe_Rx_side_leeward_max",
)
CORE_THERMAL_PRESSURE_CONSTRAINTS = (
    "temperature_robust_limit:T_max_core",
    "temperature_robust_limit:Tprobe_core_center_max",
    "temperature_robust_limit:Tprobe_core_top_yoke_max",
)
ALL_THERMAL_PRESSURE_CONSTRAINTS = tuple(
    f"temperature_robust_limit:{target}" for target in TEMPERATURE_TARGETS
)
THERMAL_CROSSOVER_ACQUISITION_SCHEMA = (
    "mft-tier1-thermal-crossover-acquisition-ranking-v1"
)
THERMAL_CROSSOVER_SOFT_AXIS_TARGET_MM = 1_000.0
THERMAL_CROSSOVER_SOFT_AXIS_SCALE_MM = 100.0
OPTIMIZER_TERMINATION_SCHEMA = "mft-tier1-optimizer-termination-v1"
DEFAULT_TERMINATION_STRATEGY = "default-ftol-period30-v1"
FIXED_GENERATION_TERMINATION_STRATEGY = "fixed-n-gen-no-ftol-v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def expected_optimizer_termination_contract(
    strategy: str, max_generations: int,
) -> dict:
    if (
        isinstance(max_generations, bool)
        or int(max_generations) != max_generations
        or int(max_generations) < 1
    ):
        raise RuntimeError("optimizer max generations is invalid")
    max_generations = int(max_generations)
    if strategy == DEFAULT_TERMINATION_STRATEGY:
        rule = {
            "kind": "pymoo_DefaultMultiObjectiveTermination",
            "ftol": 0.0025,
            "period": 30,
            "n_max_gen": max_generations,
            "early_stop_allowed": True,
        }
    elif strategy == FIXED_GENERATION_TERMINATION_STRATEGY:
        rule = {
            "kind": "pymoo_minimize_tuple_n_gen",
            "n_max_gen": max_generations,
            "early_stop_allowed": False,
            "completed_evolution_generations": max_generations,
            "expected_algorithm_n_gen_counter": max_generations + 1,
        }
    else:
        raise RuntimeError("unsupported optimizer termination strategy")
    contract = {
        "schema_version": OPTIMIZER_TERMINATION_SCHEMA,
        "strategy": strategy,
        "rule": rule,
        "pinned_initial_population": "optimization.run_nsga2._initial_population",
        "pinned_physics_repair": "optimization.run_nsga2._physics_repair_operator",
        "physical_evaluator_mutation": False,
        "physical_constraint_mutation": False,
        "objective_mutation": False,
        "authoritative_terminal_G": "physical_unscaled_replay",
    }
    contract["sha256"] = canonical_sha(contract)
    return contract


def expected_temperature_contract(hard_spec: dict) -> dict:
    if not isinstance(hard_spec, dict) or "T_limit_C" not in hard_spec:
        raise RuntimeError("hard spec must explicitly contain T_limit_C")
    value = hard_spec["T_limit_C"]
    if isinstance(value, bool):
        raise RuntimeError("T_limit_C must be exactly 110.0 C")
    try:
        limit = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("T_limit_C must be exactly 110.0 C") from exc
    if limit != EXPECTED_TEMPERATURE_LIMIT_C:
        raise RuntimeError("seed task requires T_limit_C exactly 110.0 C")
    unconditional = [
        target for target in TEMPERATURE_TARGETS
        if target not in SIDE_TEMPERATURE_TARGETS
    ]
    return {
        "schema_version": TEMPERATURE_CONTRACT_SCHEMA,
        "semantic_version": "all11-robust-q90-half-width-t110-v4",
        "robust_upper_bound_C": limit,
        "hard_spec_temperature_key": "T_limit_C",
        "formula": TEMPERATURE_FORMULA,
        "uncertainty_contract": UNCERTAINTY_CONTRACT,
        "targets": list(TEMPERATURE_TARGETS),
        "target_count": len(TEMPERATURE_TARGETS),
        "unconditional_targets": unconditional,
        "unconditional_target_count": len(unconditional),
        "side_winding_conditional_targets": list(SIDE_TEMPERATURE_TARGETS),
        "side_winding_conditional_target_count": len(SIDE_TEMPERATURE_TARGETS),
        "side_winding_activation": "finite_N2_side_gt_0",
        "side_winding_absent_behavior": (
            "finite_N2_side_eq_0_disables_with_negative_BIG"
        ),
        "side_winding_missing_behavior": (
            "missing_or_nonfinite_N2_side_fails_with_positive_BIG"
        ),
        "constraint_name_format": "temperature_robust_limit:{target}",
        "source": "model_targets.SURROGATE_TEMPERATURE_TARGETS",
    }


def validate_temperature_contract(
    hard_spec: dict, contract: dict, expected_sha256: str,
) -> str:
    expected = expected_temperature_contract(hard_spec)
    if contract != expected:
        raise RuntimeError("temperature constraint semantic contract mismatch")
    actual_sha = canonical_sha(contract)
    if actual_sha != expected_sha256:
        raise RuntimeError("temperature constraint contract SHA-256 mismatch")
    return actual_sha


def expected_thermal_crossover_acquisition_contract(
    hard_spec: dict, core_thermal_scale_c: float,
) -> dict:
    if isinstance(core_thermal_scale_c, bool):
        raise RuntimeError("acquisition core thermal scale is invalid")
    core_thermal_scale_c = float(core_thermal_scale_c)
    if not math.isfinite(core_thermal_scale_c) or core_thermal_scale_c <= 0.0:
        raise RuntimeError("acquisition core thermal scale is invalid")
    contract = {
        "schema_version": THERMAL_CROSSOVER_ACQUISITION_SCHEMA,
        "active": True,
        "purpose": "candidate_acquisition_ranking_only",
        "authoritative_terminal_G": "physical_unscaled_unchanged",
        "hard_constraint_mutation": False,
        "core_thermal_constraints": list(CORE_THERMAL_PRESSURE_CONSTRAINTS),
        "core_thermal_positive_G_scale_C": core_thermal_scale_c,
        "soft_axis_targets_mm": {
            "exterior_width": THERMAL_CROSSOVER_SOFT_AXIS_TARGET_MM,
            "exterior_length": THERMAL_CROSSOVER_SOFT_AXIS_TARGET_MM,
        },
        "soft_axis_positive_excess_scales_mm": {
            "exterior_width": THERMAL_CROSSOVER_SOFT_AXIS_SCALE_MM,
            "exterior_length": THERMAL_CROSSOVER_SOFT_AXIS_SCALE_MM,
        },
        "soft_axis_mode": (
            "independent_positive_excess_above_1000mm_no_new_hard_G"
        ),
        "physical_axis_recovery": {
            "exterior_width_mm": (
                "G_exterior_width_limit+hard_spec.size_W_max_mm"
            ),
            "exterior_length_mm": (
                "G_exterior_length_limit+hard_spec.size_L_max_mm"
            ),
        },
        "hard_axis_limits_mm": {
            "exterior_width": float(hard_spec["size_W_max_mm"]),
            "exterior_length": float(hard_spec["size_L_max_mm"]),
        },
        "ranking_formula": (
            "base_target_score+sum(max(core_thermal_G_C,0)/core_scale_C)"
            "+max(exterior_width_mm-1000,0)/100"
            "+max(exterior_length_mm-1000,0)/100"
        ),
    }
    contract["sha256"] = canonical_sha(contract)
    return contract


def validate_result_seal(result: dict, payload: dict, manifest: dict) -> None:
    """Authenticate optimizer output before the terminal completion seal."""

    if (
        int(result.get("seed", -1)) != int(payload["seed"])
        or result.get("model_manifest_sha256")
        != manifest["deployment_model_manifest_sha256"]
        or result.get("nsga_code_revision") != manifest["nsga_code_revision"]
        or result.get("constraint_version") != manifest["constraint_version"]
        or result.get("hard_spec") != manifest["hard_spec"]
        or result.get("hard_spec_sha256") != manifest["hard_spec_sha256"]
        or canonical_sha(result.get("hard_spec"))
        != manifest["hard_spec_sha256"]
        or result.get("temperature_constraint_contract")
        != manifest["temperature_constraint_contract"]
        or result.get("temperature_constraint_contract_sha256")
        != manifest["temperature_constraint_contract_sha256"]
        or result.get("production_eligible") is not False
        or result.get("fea_submission_approved") is not False
        or result.get("fea_submission_performed") is not False
        or result.get("aedt_used") is not False
        or result.get("automatic_promotion_allowed") is not False
    ):
        raise RuntimeError("result identity or fail-closed seal mismatch")
    profile = manifest.get("search_profile") or {}
    variant = profile.get("variant")
    if variant in DEEP_CROSSOVER_BY_VARIANT:
        island = DEEP_CROSSOVER_BY_VARIANT[variant]
        if (
            profile.get("deep_crossover_island")
            != deep_crossover_island_profile(island)
            or profile.get("seed_window_end_exclusive")
            != island.seed_window_end_exclusive
            or not island.seed_start <= int(payload["seed"]) < (
                island.seed_window_end_exclusive
            )
        ):
            raise RuntimeError("deep crossover island profile mismatch")
    if "max_generations" in payload or variant in DEEP_CROSSOVER_BY_VARIANT:
        if "max_generations" not in payload:
            raise RuntimeError("deep crossover payload omitted max generations")
        expected_termination_strategy = profile.get(
            "optimizer_termination_strategy", DEFAULT_TERMINATION_STRATEGY,
        )
        expected_termination = expected_optimizer_termination_contract(
            expected_termination_strategy, int(payload["max_generations"]),
        )
        completed_generations = int(result.get("completed_generations", -1))
        if (
            result.get("max_generations") != int(payload["max_generations"])
            or result.get("optimizer_termination_strategy")
            != expected_termination_strategy
            or result.get("optimizer_termination_contract")
            != expected_termination
            or (
                expected_termination_strategy
                == FIXED_GENERATION_TERMINATION_STRATEGY
                and (
                    result.get("fixed_generation_gate_satisfied") is not True
                    or completed_generations
                    != int(payload["max_generations"]) + 1
                    or result.get("evaluated_generations")
                    != int(payload["max_generations"])
                )
            )
        ):
            raise RuntimeError("result optimizer termination seal mismatch")
    snapshot_pairs = (
        ("terminal_population_best_design", "terminal_population_best_constraint_G"),
        (
            "optimizer_terminal_best_design",
            "optimizer_terminal_best_physical_constraint_G",
        ),
    )
    require_terminal_snapshots = (
        variant in DEEP_CROSSOVER_BY_VARIANT
        or "max_generations" in payload
        or any(result.get(snapshot_field) is not None for snapshot_field, _ in snapshot_pairs)
    )
    if require_terminal_snapshots:
        for snapshot_field, constraint_field in snapshot_pairs:
            snapshot = result.get(snapshot_field)
            if (
                not isinstance(snapshot, dict)
                or canonical_sha(snapshot.get("coordinate_unit"))
                != snapshot.get("coordinate_unit_sha256")
                or canonical_sha(snapshot.get("decoded_params"))
                != snapshot.get("decoded_params_sha256")
                or canonical_sha(snapshot.get("physical_constraint_G"))
                != snapshot.get("physical_constraint_G_sha256")
                or snapshot.get("physical_constraint_G")
                != result.get(constraint_field)
            ):
                raise RuntimeError("result terminal design snapshot seal mismatch")
    normalization = result.get("optimizer_constraint_normalization")
    expected_scale = manifest.get("optimizer_resonance_scale_Hz")
    expected_core_scale = manifest.get("optimizer_core_thermal_scale_C")
    expected_llt_scale = manifest.get("optimizer_Llt_scale_uH")
    expected_all_thermal_scale = manifest.get(
        "optimizer_all_active_thermal_scale_C"
    )
    expected_overrides = {}
    if expected_core_scale is not None:
        if isinstance(expected_core_scale, bool):
            raise RuntimeError("manifest core thermal scale is invalid")
        expected_core_scale = float(expected_core_scale)
        if not 0.0 < expected_core_scale < float("inf"):
            raise RuntimeError("manifest core thermal scale is invalid")
        expected_overrides.update({
            name: expected_core_scale
            for name in CORE_THERMAL_PRESSURE_CONSTRAINTS
        })
    if expected_llt_scale is not None:
        if isinstance(expected_llt_scale, bool):
            raise RuntimeError("manifest Llt scale is invalid")
        expected_llt_scale = float(expected_llt_scale)
        if not 0.0 < expected_llt_scale < float("inf"):
            raise RuntimeError("manifest Llt scale is invalid")
        expected_overrides.update({
            "Llt_robust_band": expected_llt_scale,
            "Llt_ensemble_disagreement": 2.0 * expected_llt_scale,
        })
    if expected_all_thermal_scale is not None:
        if expected_core_scale is not None:
            raise RuntimeError(
                "manifest core and all-thermal scales conflict"
            )
        if isinstance(expected_all_thermal_scale, bool):
            raise RuntimeError("manifest all-thermal scale is invalid")
        expected_all_thermal_scale = float(expected_all_thermal_scale)
        if not 0.0 < expected_all_thermal_scale < float("inf"):
            raise RuntimeError("manifest all-thermal scale is invalid")
        expected_overrides.update({
            name: expected_all_thermal_scale
            for name in ALL_THERMAL_PRESSURE_CONSTRAINTS
        })
    expected_overrides = expected_overrides or None
    if (
        not isinstance(normalization, dict)
        or normalization.get("schema_version")
        != "mft-tier1-optimizer-constraint-normalization-v1"
        or normalization.get("purpose") != "optimizer_search_pressure_only"
        or normalization.get("authoritative_terminal_G") != "physical_unscaled"
        or canonical_sha({
            key: value for key, value in normalization.items() if key != "sha256"
        }) != normalization.get("sha256")
        or (normalization.get("scales") or {}).get(
            "half_magnetizing_resonance_minimum"
        ) != expected_scale
        or payload.get("optimizer_resonance_scale_Hz") != expected_scale
        or (
            (
                expected_all_thermal_scale is not None
                or "optimizer_resonance_scale_Hz" in result
            )
            and result.get("optimizer_resonance_scale_Hz") != expected_scale
        )
        or payload.get("optimizer_core_thermal_scale_C")
        != expected_core_scale
        or result.get("optimizer_core_thermal_scale_C")
        != expected_core_scale
        or payload.get("optimizer_Llt_scale_uH") != expected_llt_scale
        or result.get("optimizer_Llt_scale_uH") != expected_llt_scale
        or payload.get("optimizer_all_active_thermal_scale_C")
        != expected_all_thermal_scale
        or result.get("optimizer_all_active_thermal_scale_C")
        != expected_all_thermal_scale
        or normalization.get("explicit_optimizer_only_scale_overrides")
        != expected_overrides
    ):
        raise RuntimeError("result optimizer-only pressure seal mismatch")
    fixed_primary_turns = (manifest.get("search_profile") or {}).get(
        "fixed_primary_turns"
    )
    if fixed_primary_turns is not None:
        fixed_contract = result.get("fixed_primary_turns_contract")
        warm_audit = (
            (result.get("warm_start") or {}).get(
                "fixed_primary_turns_warm_audit"
            ) or {}
        )
        initialization_warm = (
            (result.get("initialization_audit") or {}).get("warm_filter")
            or {}
        )
        minimum_unique = (
            fixed_contract.get(
                "warm_start_post_repair_minimum_unique_count"
            ) if isinstance(fixed_contract, dict) else None
        )
        if (
            isinstance(fixed_primary_turns, bool)
            or int(fixed_primary_turns) != fixed_primary_turns
            or (payload.get("search_profile") or {}).get(
                "fixed_primary_turns"
            ) != fixed_primary_turns
            or result.get("fixed_primary_turns") != fixed_primary_turns
            or not isinstance(fixed_contract, dict)
            or fixed_contract.get("schema_version")
            != "mft-tier1-fixed-primary-turns-repair-v1"
            or fixed_contract.get("fixed_primary_turns")
            != fixed_primary_turns
            or fixed_contract.get("hard_constraint_mutation") is not False
            or fixed_contract.get("objective_mutation") is not False
            or not isinstance(minimum_unique, int)
            or minimum_unique < 1
            or canonical_sha({
                key: value for key, value in fixed_contract.items()
                if key != "sha256"
            }) != fixed_contract.get("sha256")
            or result.get(
                "terminal_population_fixed_primary_turns_verified"
            ) is not True
            or result.get("terminal_population_primary_turn_values")
            != [int(fixed_primary_turns)]
            or warm_audit.get("source_sha_verified_before_repair") is not True
            or warm_audit.get("fixed_primary_turns")
            != fixed_primary_turns
            or warm_audit.get(
                "fixed_primary_turn_coordinate_verified"
            ) is not True
            or warm_audit.get("physical_geometry_dedupe_performed") is not True
            or int(warm_audit.get("post_repair_decoded_unique_count", -1))
            < minimum_unique
            or initialization_warm.get("decoded_unique_count")
            != warm_audit.get("post_repair_decoded_unique_count")
        ):
            raise RuntimeError("result fixed primary-turn seal mismatch")
    if variant in DEEP_CROSSOVER_BY_VARIANT:
        island = DEEP_CROSSOVER_BY_VARIANT[variant]
        expected_topology = topology_evolution_contract(
            island.fixed_primary_turns
        )
        observed_topology = result.get(
            "optimizer_topology_evolution_contract"
        )
        topology_audit = result.get("optimizer_topology_evolution_audit")
        initialization = (
            (result.get("initialization_audit") or {}).get(
                "turn_split_sub_islands"
            ) or {}
        )
        counts = (topology_audit or {}).get("terminal_topology_counts") or {}
        required = expected_topology["turn_split_sub_islands_N2_main"]
        minimum_each = expected_topology["survival"][
            "minimum_survivors_per_turn_split_sub_island"
        ]
        if (
            observed_topology != expected_topology
            or canonical_sha({
                key: value for key, value in (observed_topology or {}).items()
                if key != "sha256"
            }) != (observed_topology or {}).get("sha256")
            or not isinstance(topology_audit, dict)
            or canonical_sha({
                key: value for key, value in topology_audit.items()
                if key != "sha256"
            }) != topology_audit.get("sha256")
            or initialization.get("minimum_copies_each_verified") is not True
            or initialization.get(
                "source_prediction_or_pass_classification_inherited"
            ) is not False
            or topology_audit.get("paired_parent_pairs_emitted", 0) < 1
            or topology_audit.get("migration_events", 0) < 1
            or topology_audit.get("migrants_created", 0) < len(required)
            or topology_audit.get("all_required_topologies_preserved") is not True
            or topology_audit.get("terminal_epsilon_zero") is not True
            or float(topology_audit.get("last_optimizer_epsilon", math.inf))
            != 0.0
            or set(counts) != {str(value) for value in required}
            or any(int(value) < minimum_each for value in counts.values())
            or topology_audit.get("physical_constraint_G_mutation") is not False
            or topology_audit.get("physical_objective_mutation") is not False
            or topology_audit.get("warm_donor_prediction_inheritance") is not False
        ):
            raise RuntimeError("result deep turn-split evolution seal mismatch")
    expected_names = [
        f"temperature_robust_limit:{target}" for target in TEMPERATURE_TARGETS
    ]
    constraint_names = result.get("constraint_names")
    if not isinstance(constraint_names, list):
        raise RuntimeError("result has no constraint name schema")
    expected_resonance_allowance = profile.get(
        "optimizer_resonance_allowance_Hz"
    )
    expected_llt_allowance = profile.get("optimizer_Llt_allowance_uH")
    if ((expected_resonance_allowance is None)
            != (expected_llt_allowance is None)):
        raise RuntimeError("manifest has an incomplete optimizer allowance")
    expected_allowance_contract = None
    if expected_resonance_allowance is not None:
        expected_allowance_contract = optimizer_allowance_contract(
            constraint_names,
            resonance_allowance_hz=expected_resonance_allowance,
            llt_allowance_uh=expected_llt_allowance,
        )
    if (
        result.get("optimizer_resonance_allowance_Hz")
        != expected_resonance_allowance
        or result.get("optimizer_Llt_allowance_uH")
        != expected_llt_allowance
        or result.get("optimizer_constraint_allowance_contract")
        != expected_allowance_contract
        or (
            "search_profile" in manifest
            and payload.get("search_profile") != profile
        )
    ):
        raise RuntimeError("result optimizer-only allowance seal mismatch")
    actual_names = [
        name for name in constraint_names
        if isinstance(name, str) and name.startswith("temperature_robust_limit:")
    ]
    if actual_names != expected_names:
        raise RuntimeError("result thermal constraint schema mismatch")
    if expected_core_scale is not None:
        if normalization.get("constraint_order") != constraint_names:
            raise RuntimeError("thermal pressure constraint order mismatch")
        scales = normalization.get("scales") or {}
        if any(
            scales.get(name) != (
                expected_core_scale
                if name in CORE_THERMAL_PRESSURE_CONSTRAINTS
                else 5.5
            )
            for name in expected_names
        ):
            raise RuntimeError("thermal pressure scale scope mismatch")
        expected_acquisition = (
            expected_thermal_crossover_acquisition_contract(
                manifest["hard_spec"], expected_core_scale,
            )
        )
        if (
            result.get("acquisition_ranking_contract")
            != expected_acquisition
            or manifest.get("acquisition_ranking_contract")
            != expected_acquisition
            or payload.get("acquisition_ranking_contract")
            != expected_acquisition
            or manifest.get("acquisition_ranking_contract_sha256")
            != expected_acquisition["sha256"]
            or payload.get("acquisition_ranking_contract_sha256")
            != expected_acquisition["sha256"]
        ):
            raise RuntimeError("thermal acquisition ranking contract mismatch")
        candidates = (
            result.get("next_target_fea_batch_plan") or {}
        ).get("candidates")
        if not isinstance(candidates, list):
            raise RuntimeError("thermal acquisition candidate inventory missing")
        for candidate in candidates:
            values = candidate.get("constraint_G") or {}
            components = candidate.get(
                "target_acquisition_score_components"
            ) or {}
            try:
                core = sum(
                    max(float(values[name]), 0.0) / expected_core_scale
                    for name in CORE_THERMAL_PRESSURE_CONSTRAINTS
                )
                width_mm = (
                    float(values["exterior_width_limit"])
                    + float(manifest["hard_spec"]["size_W_max_mm"])
                )
                length_mm = (
                    float(values["exterior_length_limit"])
                    + float(manifest["hard_spec"]["size_L_max_mm"])
                )
                width = max(
                    width_mm - THERMAL_CROSSOVER_SOFT_AXIS_TARGET_MM, 0.0
                ) / THERMAL_CROSSOVER_SOFT_AXIS_SCALE_MM
                length = max(
                    length_mm - THERMAL_CROSSOVER_SOFT_AXIS_TARGET_MM, 0.0
                ) / THERMAL_CROSSOVER_SOFT_AXIS_SCALE_MM
                base = float(
                    components["base_target_normalized_positive_violation"]
                )
                total = base + core + width + length
                observed = (
                    float(components["core_thermal_positive_normalized"]),
                    float(components[
                        "soft_width_positive_excess_normalized"
                    ]),
                    float(components[
                        "soft_length_positive_excess_normalized"
                    ]),
                    float(components["total"]),
                    float(candidate["target_acquisition_score"]),
                )
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(
                    "thermal acquisition candidate metadata invalid"
                ) from exc
            expected = (core, width, length, total, total)
            if any(
                not math.isfinite(actual)
                or not math.isclose(actual, wanted, rel_tol=1e-12, abs_tol=1e-12)
                for actual, wanted in zip(observed, expected)
            ):
                raise RuntimeError("thermal acquisition candidate score mismatch")
    if expected_llt_scale is not None:
        scales = normalization.get("scales") or {}
        if (
            scales.get("Llt_robust_band") != expected_llt_scale
            or scales.get("Llt_ensemble_disagreement")
            != 2.0 * expected_llt_scale
        ):
            raise RuntimeError("Llt optimizer pressure scale scope mismatch")
    if expected_all_thermal_scale is not None:
        scales = normalization.get("scales") or {}
        if normalization.get("constraint_order") != constraint_names or any(
            scales.get(name) != expected_all_thermal_scale
            for name in expected_names
        ):
            raise RuntimeError(
                "all-active thermal pressure scale scope mismatch"
            )
        if (
            result.get("acquisition_ranking_contract") is not None
            or manifest.get("acquisition_ranking_contract") is not None
            or payload.get("acquisition_ranking_contract") is not None
            or manifest.get("acquisition_ranking_contract_sha256") is not None
            or payload.get("acquisition_ranking_contract_sha256") is not None
        ):
            raise RuntimeError(
                "orthogonal lane unexpectedly applied soft-axis acquisition"
            )
    for field in (
        "constraint_minimum_G", "terminal_population_best_constraint_G",
    ):
        values = result.get(field)
        if not isinstance(values, dict) or any(
            name not in values for name in expected_names
        ):
            raise RuntimeError(f"result is missing a thermal constraint in {field}")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def contained(root: Path, relative: str) -> Path:
    if not relative or "\\" in relative:
        raise RuntimeError(f"invalid POSIX bundle path: {relative!r}")
    result = (root / relative).resolve()
    root = root.resolve()
    if result != root and root not in result.parents:
        raise RuntimeError(f"bundle path escapes root: {relative}")
    return result


def verify_payload(
    bundle: Path, payload_path: Path, payload_root: Path, expected_sha: str,
) -> tuple[dict, dict]:
    payload_root = payload_root.resolve(strict=True)
    payload_path = payload_path.resolve(strict=True)
    if payload_path.name != "payload.json" or payload_root not in payload_path.parents:
        raise RuntimeError("scheduler payload escaped the scheduler run root")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA:
        raise RuntimeError("unsupported task payload schema")
    if canonical_sha(payload) != expected_sha:
        raise RuntimeError("task payload SHA-256 mismatch")

    manifest_path = bundle / "bundle_manifest.json"
    manifest_sha = sha256(manifest_path)
    if manifest_sha != payload.get("bundle_manifest_sha256"):
        raise RuntimeError("bundle manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("bundle_id") != payload.get("bundle_id"):
        raise RuntimeError("bundle ID mismatch")
    if (
        manifest.get("attestation_schema_version") != ATTESTATION_SCHEMA
        or payload.get("attestation_schema_version") != ATTESTATION_SCHEMA
        or manifest.get("task_schema_version") != SCHEMA
        or payload.get("task_schema_version") != SCHEMA
    ):
        raise RuntimeError("bundle/payload v3 attestation schema mismatch")
    hard_spec = manifest.get("hard_spec")
    if (
        not isinstance(hard_spec, dict)
        or canonical_sha(hard_spec) != manifest.get("hard_spec_sha256")
        or payload.get("hard_spec") != hard_spec
        or canonical_sha(payload.get("hard_spec"))
        != payload.get("hard_spec_sha256")
    ):
        raise RuntimeError("bundle/payload hard-spec semantic mismatch")
    core_scale = manifest.get("optimizer_core_thermal_scale_C")
    expected_acquisition = (
        expected_thermal_crossover_acquisition_contract(
            hard_spec, core_scale
        )
        if core_scale is not None else None
    )
    if (
        manifest.get("acquisition_ranking_contract")
        != expected_acquisition
        or manifest.get("acquisition_ranking_contract_sha256")
        != (
            expected_acquisition["sha256"]
            if expected_acquisition is not None else None
        )
    ):
        raise RuntimeError("bundle thermal acquisition contract mismatch")
    validate_temperature_contract(
        hard_spec,
        manifest.get("temperature_constraint_contract"),
        manifest.get("temperature_constraint_contract_sha256"),
    )
    if (
        payload.get("temperature_constraint_contract")
        != manifest.get("temperature_constraint_contract")
        or payload.get("temperature_constraint_contract_sha256")
        != manifest.get("temperature_constraint_contract_sha256")
    ):
        raise RuntimeError("payload temperature contract is not pinned to bundle")
    if (
        payload.get("production_eligible") is not False
        or payload.get("fea_submission_approved") is not False
        or payload.get("fea_submission_performed") is not False
        or payload.get("aedt_used") is not False
        or payload.get("automatic_promotion_allowed") is not False
        or manifest.get("production_eligible") is not False
        or manifest.get("fea_submission_approved") is not False
        or manifest.get("automatic_promotion_allowed") is not False
    ):
        raise RuntimeError("bundle/payload fail-closed policy mismatch")
    ready = json.loads((bundle / "READY.json").read_text(encoding="utf-8"))
    if (
        ready.get("bundle_id") != manifest["bundle_id"]
        or ready.get("bundle_manifest_sha256") != manifest_sha
        or ready.get("verified") is not True
        or ready.get("attestation_schema_version") != ATTESTATION_SCHEMA
        or ready.get("task_schema_version") != SCHEMA
        or ready.get("hard_spec_sha256") != manifest["hard_spec_sha256"]
        or ready.get("temperature_constraint_contract_sha256")
        != manifest["temperature_constraint_contract_sha256"]
        or ready.get("search_profile") != manifest.get("search_profile")
        or ready.get("optimizer_resonance_scale_Hz")
        != manifest.get("optimizer_resonance_scale_Hz")
        or ready.get("optimizer_core_thermal_scale_C")
        != manifest.get("optimizer_core_thermal_scale_C")
        or ready.get("optimizer_Llt_scale_uH")
        != manifest.get("optimizer_Llt_scale_uH")
        or ready.get("optimizer_all_active_thermal_scale_C")
        != manifest.get("optimizer_all_active_thermal_scale_C")
        or ready.get("acquisition_ranking_contract_sha256")
        != manifest.get("acquisition_ranking_contract_sha256")
        or ready.get("warm_handoff_contract_sha256")
        != manifest.get("warm_handoff_contract_sha256")
        or ready.get("anchor_island_preflight_sha256")
        != manifest.get("anchor_island_preflight_sha256")
        or ready.get("deep_crossover_preflight_sha256")
        != manifest.get("deep_crossover_preflight_sha256")
    ):
        raise RuntimeError("bundle has no matching atomic READY seal")

    expected_pairs = {
        "source_model_manifest_sha256": manifest["source_model_manifest_sha256"],
        "deployment_model_manifest_sha256": manifest[
            "deployment_model_manifest_sha256"
        ],
        "nsga_code_revision": manifest["nsga_code_revision"],
        "hard_spec_sha256": manifest["hard_spec_sha256"],
        "warm_start_sha256": manifest["warm_start_sha256"],
        "constraint_version": manifest["constraint_version"],
        "controller_source_sha256": manifest["controller_source_sha256"],
        "temperature_constraint_contract_sha256": manifest[
            "temperature_constraint_contract_sha256"
        ],
        "search_profile": manifest["search_profile"],
        "optimizer_resonance_scale_Hz": manifest[
            "optimizer_resonance_scale_Hz"
        ],
        "optimizer_core_thermal_scale_C": manifest.get(
            "optimizer_core_thermal_scale_C"
        ),
        "optimizer_Llt_scale_uH": manifest.get("optimizer_Llt_scale_uH"),
        "optimizer_all_active_thermal_scale_C": manifest.get(
            "optimizer_all_active_thermal_scale_C"
        ),
        "acquisition_ranking_contract": manifest.get(
            "acquisition_ranking_contract"
        ),
        "acquisition_ranking_contract_sha256": manifest.get(
            "acquisition_ranking_contract_sha256"
        ),
    }
    for key, value in expected_pairs.items():
        if payload.get(key) != value:
            raise RuntimeError(f"payload {key} is not pinned to bundle")

    required_high_value = {
        "artifacts/code/regression_260707/model_targets.py",
        "artifacts/code/regression_260707/uncertainty_contract.py",
        "artifacts/tier1_n1_6_anchor_island_contract.py",
        "artifacts/tier1_deep_crossover_contract.py",
    }
    if not required_high_value.issubset(manifest["high_value_sha256"]):
        raise RuntimeError("thermal semantic sources are not high-value sealed")
    if isinstance((manifest.get("search_profile") or {}).get("anchor_island"), dict):
        preflight_relative = manifest.get("anchor_island_preflight")
        preflight_sha = manifest.get("anchor_island_preflight_sha256")
        preflight_evidence = manifest.get("anchor_island_preflight_evidence")
        if (
            not isinstance(preflight_relative, str)
            or not isinstance(preflight_sha, str)
            or manifest["high_value_sha256"].get(preflight_relative)
            != preflight_sha
            or not isinstance(preflight_evidence, list)
            or len(preflight_evidence) != 6
            or any(
                relative not in manifest["high_value_sha256"]
                for relative in preflight_evidence
            )
        ):
            raise RuntimeError("anchor-island preflight is not high-value sealed")
    if isinstance(
        (manifest.get("search_profile") or {}).get("deep_crossover_island"),
        dict,
    ):
        preflight_relative = manifest.get("deep_crossover_preflight")
        preflight_sha = manifest.get("deep_crossover_preflight_sha256")
        preflight_evidence = manifest.get("deep_crossover_preflight_evidence")
        if (
            not isinstance(preflight_relative, str)
            or not isinstance(preflight_sha, str)
            or manifest["high_value_sha256"].get(preflight_relative)
            != preflight_sha
            or not isinstance(preflight_evidence, list)
            or len(preflight_evidence) != 5
            or any(
                relative not in manifest["high_value_sha256"]
                for relative in preflight_evidence
            )
        ):
            raise RuntimeError("deep crossover preflight is not high-value sealed")
    for relative, expected in manifest["high_value_sha256"].items():
        path = contained(bundle, relative)
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"high-value bundle artifact mismatch: {relative}")

    base_ready = json.loads(
        Path(manifest["base_bundle_ready_path"]).read_text(encoding="utf-8")
    )
    if (
        base_ready.get("bundle_id") != manifest["base_bundle_id"]
        or base_ready.get("bundle_manifest_sha256")
        != manifest["base_bundle_manifest_sha256"]
        or base_ready.get("runtime_verified") is not True
    ):
        raise RuntimeError("shared strict-full base bundle READY seal mismatch")
    if sha256(Path(manifest["base_bundle_manifest_path"])) != manifest[
        "base_bundle_manifest_sha256"
    ]:
        raise RuntimeError("shared strict-full base bundle manifest mismatch")
    base_report = Path(manifest["base_generation_report_path"])
    if sha256(base_report) != manifest["base_generation_report_sha256"]:
        raise RuntimeError("shared strict-full generation report SHA mismatch")
    return payload, manifest


def run(
    bundle: Path, payload_path: Path, payload_root: Path, expected_sha: str,
    heartbeat_seconds: float,
) -> int:
    bundle = bundle.resolve(strict=True)
    payload, manifest = verify_payload(
        bundle, payload_path, payload_root, expected_sha
    )
    task_id = str(os.environ.get("SLURM_SCHED_TASK_ID") or f"pid-{os.getpid()}")
    seed = int(payload["seed"])
    output = bundle / "runs" / f"task-{task_id}" / f"seed-{seed}"
    status_path = output.parent / "seed_status.json"
    output.mkdir(parents=True, exist_ok=True)
    started_at = now()
    status = {
        "schema_version": STATUS_SCHEMA,
        "state": "running",
        "started_at": started_at,
        "updated_at": started_at,
        "task_id": task_id,
        "seed": seed,
        "cohort_id": payload["cohort_id"],
        "bundle_id": payload["bundle_id"],
        "bundle_manifest_sha256": payload["bundle_manifest_sha256"],
        "payload_sha256": expected_sha,
        "attestation_schema_version": payload[
            "attestation_schema_version"
        ],
        "task_schema_version": payload["task_schema_version"],
        "constraint_version": payload["constraint_version"],
        "hard_spec_sha256": payload["hard_spec_sha256"],
        "temperature_constraint_contract_sha256": payload[
            "temperature_constraint_contract_sha256"
        ],
        "search_profile": payload["search_profile"],
        "optimizer_resonance_scale_Hz": payload[
            "optimizer_resonance_scale_Hz"
        ],
        "optimizer_core_thermal_scale_C": payload.get(
            "optimizer_core_thermal_scale_C"
        ),
        "optimizer_Llt_scale_uH": payload.get("optimizer_Llt_scale_uH"),
        "optimizer_all_active_thermal_scale_C": payload.get(
            "optimizer_all_active_thermal_scale_C"
        ),
        "optimizer_resonance_allowance_Hz": (
            (payload.get("search_profile") or {}).get(
                "optimizer_resonance_allowance_Hz"
            )
        ),
        "optimizer_Llt_allowance_uH": (
            (payload.get("search_profile") or {}).get(
                "optimizer_Llt_allowance_uH"
            )
        ),
        "fixed_primary_turns": (payload.get("search_profile") or {}).get(
            "fixed_primary_turns"
        ),
        "optimizer_termination_strategy": (
            (payload.get("search_profile") or {}).get(
                "optimizer_termination_strategy",
                DEFAULT_TERMINATION_STRATEGY,
            )
        ),
        "acquisition_ranking_contract_sha256": payload.get(
            "acquisition_ranking_contract_sha256"
        ),
        "source_model_manifest_sha256": payload[
            "source_model_manifest_sha256"
        ],
        "deployment_model_manifest_sha256": payload[
            "deployment_model_manifest_sha256"
        ],
        "nsga_code_revision": payload["nsga_code_revision"],
        "population": int(payload["population"]),
        "max_generations": int(payload["max_generations"]),
        "inference_threads": int(payload["inference_threads"]),
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    atomic_json(status_path, status)

    feedback = contained(bundle, manifest["feedback_script"])
    model_dir = contained(bundle, manifest["model_dir"])
    code_root = contained(bundle, manifest["code_root"])
    warm = contained(bundle, manifest["warm_start"])
    command = [
        sys.executable,
        str(feedback),
        "search-seed",
        "--model-dir", str(model_dir),
        "--nsga-code-root", str(code_root),
        "--output", str(output),
        "--seed", str(seed),
        "--population", str(int(payload["population"])),
        "--max-generations", str(int(payload["max_generations"])),
        "--optimizer-termination-strategy",
        str((payload.get("search_profile") or {}).get(
            "optimizer_termination_strategy", DEFAULT_TERMINATION_STRATEGY,
        )),
        "--warm-start", str(warm),
        "--warm-start-sha256", payload["warm_start_sha256"],
        "--inference-threads", str(int(payload["inference_threads"])),
        "--optimizer-resonance-scale-hz",
        str(payload["optimizer_resonance_scale_Hz"]),
    ]
    if payload.get("optimizer_core_thermal_scale_C") is not None:
        command.extend([
            "--optimizer-core-thermal-scale-c",
            str(payload["optimizer_core_thermal_scale_C"]),
        ])
    if payload.get("optimizer_Llt_scale_uH") is not None:
        command.extend([
            "--optimizer-llt-scale-uh",
            str(payload["optimizer_Llt_scale_uH"]),
        ])
    if payload.get("optimizer_all_active_thermal_scale_C") is not None:
        command.extend([
            "--optimizer-all-thermal-scale-c",
            str(payload["optimizer_all_active_thermal_scale_C"]),
        ])
    allowance_profile = payload.get("search_profile") or {}
    resonance_allowance = allowance_profile.get(
        "optimizer_resonance_allowance_Hz"
    )
    llt_allowance = allowance_profile.get("optimizer_Llt_allowance_uH")
    if ((resonance_allowance is None) != (llt_allowance is None)):
        raise RuntimeError("payload has an incomplete optimizer allowance")
    if resonance_allowance is not None:
        command.extend([
            "--optimizer-resonance-allowance-hz",
            str(resonance_allowance),
            "--optimizer-llt-allowance-uh",
            str(llt_allowance),
        ])
    fixed_primary_turns = (payload.get("search_profile") or {}).get(
        "fixed_primary_turns"
    )
    if fixed_primary_turns is not None:
        command.extend([
            "--fixed-primary-turns", str(int(fixed_primary_turns)),
        ])
    environment = os.environ.copy()
    threads = str(int(payload["inference_threads"]))
    for variable in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        environment[variable] = threads
    process = subprocess.Popen(command, cwd=code_root, env=environment)
    stop = threading.Event()

    def terminate(_signum, _frame):
        stop.set()
        if process.poll() is None:
            process.terminate()

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    while process.poll() is None:
        status["updated_at"] = now()
        status["runner_pid"] = os.getpid()
        status["optimizer_pid"] = process.pid
        atomic_json(status_path, status)
        stop.wait(heartbeat_seconds)
    exit_code = int(process.returncode)
    result_path = output / "result.json"
    terminal = dict(status)
    terminal.update({
        "updated_at": now(),
        "finished_at": now(),
        "exit_code": exit_code,
        "state": "failed",
    })
    if exit_code == 0 and result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        try:
            validate_result_seal(result, payload, manifest)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            terminal["failure"] = f"{type(exc).__name__}:{exc}"
            exit_code = 70
        else:
            terminal.update({
                "state": "completed",
                "result_path": str(result_path),
                "result_sha256": sha256(result_path),
                "completed_generations": int(result["completed_generations"]),
                "evaluated_generations": result.get("evaluated_generations"),
                "feasible_pareto_count": int(result["feasible_pareto_count"]),
            })
    elif exit_code == 0:
        terminal["failure"] = "optimizer exited zero without result.json"
        exit_code = 71
    else:
        terminal["failure"] = f"optimizer exit code {exit_code}"
    terminal["exit_code"] = exit_code
    atomic_json(status_path, terminal)
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--payload-sha256", required=True)
    parser.add_argument("--heartbeat-seconds", type=float, default=15.0)
    args = parser.parse_args()
    return run(
        args.bundle, args.payload, args.payload_root, args.payload_sha256,
        args.heartbeat_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
