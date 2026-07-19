from __future__ import annotations

import copy

import pytest

from tools.tier1_slurm_seed_runner import (
    CORE_THERMAL_PRESSURE_CONSTRAINTS,
    TEMPERATURE_TARGETS,
    canonical_sha,
    expected_temperature_contract,
    expected_thermal_crossover_acquisition_contract,
    validate_result_seal,
)


def _sealed_result_triplet() -> tuple[dict, dict, dict]:
    hard_spec = {"T_limit_C": 110.0}
    contract = expected_temperature_contract(hard_spec)
    contract_sha = canonical_sha(contract)
    thermal_names = [
        f"temperature_robust_limit:{target}" for target in TEMPERATURE_TARGETS
    ]
    payload = {
        "seed": 101,
        "optimizer_resonance_scale_Hz": 250.0,
    }
    manifest = {
        "deployment_model_manifest_sha256": "a" * 64,
        "nsga_code_revision": "b" * 40,
        "constraint_version": "constraint-v3",
        "hard_spec": hard_spec,
        "hard_spec_sha256": canonical_sha(hard_spec),
        "temperature_constraint_contract": contract,
        "temperature_constraint_contract_sha256": contract_sha,
        "optimizer_resonance_scale_Hz": 250.0,
    }
    normalization = {
        "schema_version": "mft-tier1-optimizer-constraint-normalization-v1",
        "purpose": "optimizer_search_pressure_only",
        "authoritative_terminal_G": "physical_unscaled",
        "formula": "G_optimizer=G_physical/engineering_scale",
        "constraint_order": [
            "Llt_robust_band", "half_magnetizing_resonance_minimum",
        ],
        "scales": {
            "Llt_robust_band": 0.55,
            "half_magnetizing_resonance_minimum": 250.0,
        },
    }
    normalization["sha256"] = canonical_sha(normalization)
    result = {
        "seed": 101,
        "model_manifest_sha256": manifest[
            "deployment_model_manifest_sha256"
        ],
        "nsga_code_revision": manifest["nsga_code_revision"],
        "constraint_version": manifest["constraint_version"],
        "hard_spec": hard_spec,
        "hard_spec_sha256": manifest["hard_spec_sha256"],
        "temperature_constraint_contract": contract,
        "temperature_constraint_contract_sha256": contract_sha,
        "optimizer_constraint_normalization": normalization,
        "constraint_names": ["Llt_robust_band", *thermal_names],
        "constraint_minimum_G": {name: -1.0 for name in thermal_names},
        "terminal_population_best_constraint_G": {
            name: -1.0 for name in thermal_names
        },
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    return result, payload, manifest


def test_result_completion_seal_accepts_exact_eleven_thermal_constraints():
    result, payload, manifest = _sealed_result_triplet()
    validate_result_seal(result, payload, manifest)


def test_result_completion_seal_rejects_missing_thermal_constraint():
    result, payload, manifest = _sealed_result_triplet()
    missing_name = f"temperature_robust_limit:{TEMPERATURE_TARGETS[-1]}"
    mutated = copy.deepcopy(result)
    mutated["constraint_minimum_G"].pop(missing_name)
    with pytest.raises(RuntimeError, match="missing a thermal constraint"):
        validate_result_seal(mutated, payload, manifest)


def test_result_completion_seal_rejects_thermal_target_order_mutation():
    result, payload, manifest = _sealed_result_triplet()
    mutated = copy.deepcopy(result)
    mutated["constraint_names"][1], mutated["constraint_names"][2] = (
        mutated["constraint_names"][2], mutated["constraint_names"][1]
    )
    with pytest.raises(RuntimeError, match="thermal constraint schema"):
        validate_result_seal(mutated, payload, manifest)


def test_result_completion_seals_exact_three_core_thermal_overrides():
    result, payload, manifest = _sealed_result_triplet()
    hard_spec = {
        "T_limit_C": 110.0,
        "size_W_max_mm": 1_200.0,
        "size_L_max_mm": 1_200.0,
    }
    manifest["hard_spec"] = hard_spec
    manifest["hard_spec_sha256"] = canonical_sha(hard_spec)
    result["hard_spec"] = hard_spec
    result["hard_spec_sha256"] = manifest["hard_spec_sha256"]
    manifest["optimizer_core_thermal_scale_C"] = 1.0
    payload["optimizer_core_thermal_scale_C"] = 1.0
    result["optimizer_core_thermal_scale_C"] = 1.0
    result["constraint_names"].append(
        "half_magnetizing_resonance_minimum"
    )
    result["constraint_names"].extend([
        "exterior_width_limit", "exterior_length_limit",
    ])
    scales = {
        name: (
            1.0 if name in CORE_THERMAL_PRESSURE_CONSTRAINTS else 5.5
        )
        for name in result["constraint_names"]
        if name.startswith("temperature_robust_limit:")
    }
    scales.update({
        "Llt_robust_band": 0.55,
        "half_magnetizing_resonance_minimum": 250.0,
        "exterior_width_limit": 60.0,
        "exterior_length_limit": 60.0,
    })
    normalization = {
        "schema_version": "mft-tier1-optimizer-constraint-normalization-v1",
        "purpose": "optimizer_search_pressure_only",
        "authoritative_terminal_G": "physical_unscaled",
        "formula": "G_optimizer=G_physical/engineering_scale",
        "constraint_order": result["constraint_names"],
        "scales": scales,
        "optimizer_core_thermal_scale_C": 1.0,
        "explicit_optimizer_only_scale_overrides": {
            name: 1.0 for name in CORE_THERMAL_PRESSURE_CONSTRAINTS
        },
        "override_scope": (
            "exactly_three_core_thermal_constraints_physical_G_unchanged"
        ),
    }
    normalization["sha256"] = canonical_sha(normalization)
    result["optimizer_constraint_normalization"] = normalization
    acquisition = expected_thermal_crossover_acquisition_contract(
        hard_spec, 1.0
    )
    result["acquisition_ranking_contract"] = acquisition
    manifest["acquisition_ranking_contract"] = acquisition
    manifest["acquisition_ranking_contract_sha256"] = acquisition["sha256"]
    payload["acquisition_ranking_contract"] = acquisition
    payload["acquisition_ranking_contract_sha256"] = acquisition["sha256"]
    result["next_target_fea_batch_plan"] = {"candidates": [{
        "constraint_G": {
            CORE_THERMAL_PRESSURE_CONSTRAINTS[0]: 1.0,
            CORE_THERMAL_PRESSURE_CONSTRAINTS[1]: 2.0,
            CORE_THERMAL_PRESSURE_CONSTRAINTS[2]: 3.0,
            "exterior_width_limit": -100.0,
            "exterior_length_limit": -250.0,
        },
        "target_acquisition_score_components": {
            "base_target_normalized_positive_violation": 2.0,
            "core_thermal_positive_normalized": 6.0,
            "soft_width_positive_excess_normalized": 1.0,
            "soft_length_positive_excess_normalized": 0.0,
            "total": 9.0,
        },
        "target_acquisition_score": 9.0,
    }]}
    validate_result_seal(result, payload, manifest)

    mutated = copy.deepcopy(result)
    mutated["optimizer_constraint_normalization"][
        "explicit_optimizer_only_scale_overrides"
    ]["temperature_robust_limit:T_max_Tx"] = 1.0
    unsigned = {
        key: value
        for key, value in mutated["optimizer_constraint_normalization"].items()
        if key != "sha256"
    }
    mutated["optimizer_constraint_normalization"]["sha256"] = canonical_sha(
        unsigned
    )
    with pytest.raises(RuntimeError, match="optimizer-only pressure seal"):
        validate_result_seal(mutated, payload, manifest)


def test_result_completion_seals_fixed_primary_turn_terminal_population():
    result, payload, manifest = _sealed_result_triplet()
    profile = {"fixed_primary_turns": 6}
    manifest["search_profile"] = profile
    payload["search_profile"] = profile
    contract = {
        "schema_version": "mft-tier1-fixed-primary-turns-repair-v1",
        "fixed_primary_turns": 6,
        "hard_constraint_mutation": False,
        "objective_mutation": False,
        "warm_start_post_repair_minimum_unique_count": 32,
    }
    contract["sha256"] = canonical_sha(contract)
    result.update({
        "fixed_primary_turns": 6,
        "fixed_primary_turns_contract": contract,
        "terminal_population_primary_turn_values": [6],
        "terminal_population_fixed_primary_turns_verified": True,
        "warm_start": {"fixed_primary_turns_warm_audit": {
            "source_sha_verified_before_repair": True,
            "fixed_primary_turns": 6,
            "fixed_primary_turn_coordinate_verified": True,
            "physical_geometry_dedupe_performed": True,
            "post_repair_decoded_unique_count": 64,
        }},
        "initialization_audit": {
            "warm_filter": {"decoded_unique_count": 64}
        },
    })
    validate_result_seal(result, payload, manifest)

    mutated = copy.deepcopy(result)
    mutated["terminal_population_primary_turn_values"] = [5, 6]
    with pytest.raises(RuntimeError, match="fixed primary-turn seal"):
        validate_result_seal(mutated, payload, manifest)


def test_result_completion_seals_orthogonal_all11_llt_pressure_without_soft_axis():
    result, payload, manifest = _sealed_result_triplet()
    thermal_names = [
        f"temperature_robust_limit:{target}"
        for target in TEMPERATURE_TARGETS
    ]
    constraint_names = [
        "Llt_robust_band",
        *thermal_names,
        "Llt_ensemble_disagreement",
        "half_magnetizing_resonance_minimum",
    ]
    expected_overrides = {
        "Llt_robust_band": 0.25,
        "Llt_ensemble_disagreement": 0.5,
        **{name: 2.0 for name in thermal_names},
    }
    manifest.update({
        "optimizer_resonance_scale_Hz": 150.0,
        "optimizer_core_thermal_scale_C": None,
        "optimizer_Llt_scale_uH": 0.25,
        "optimizer_all_active_thermal_scale_C": 2.0,
        "acquisition_ranking_contract": None,
        "acquisition_ranking_contract_sha256": None,
    })
    payload.update({
        "optimizer_resonance_scale_Hz": 150.0,
        "optimizer_core_thermal_scale_C": None,
        "optimizer_Llt_scale_uH": 0.25,
        "optimizer_all_active_thermal_scale_C": 2.0,
        "acquisition_ranking_contract": None,
        "acquisition_ranking_contract_sha256": None,
    })
    normalization = {
        "schema_version": "mft-tier1-optimizer-constraint-normalization-v1",
        "purpose": "optimizer_search_pressure_only",
        "authoritative_terminal_G": "physical_unscaled",
        "formula": "G_optimizer=G_physical/engineering_scale",
        "constraint_order": constraint_names,
        "scales": {
            "Llt_robust_band": 0.25,
            **{name: 2.0 for name in thermal_names},
            "Llt_ensemble_disagreement": 0.5,
            "half_magnetizing_resonance_minimum": 150.0,
        },
        "optimizer_Llt_scale_uH": 0.25,
        "optimizer_all_active_thermal_scale_C": 2.0,
        "optimizer_all_active_thermal_constraints": thermal_names,
        "all_active_thermal_side_activation": (
            "finite_N2_side_gt_0_else_physical_negative_BIG"
        ),
        "explicit_optimizer_only_scale_overrides": expected_overrides,
        "override_scope": (
            "sealed_Llt_and_all_active_thermal_constraints_"
            "physical_G_unchanged"
        ),
    }
    normalization["sha256"] = canonical_sha(normalization)
    result.update({
        "constraint_names": constraint_names,
        "optimizer_constraint_normalization": normalization,
        "optimizer_resonance_scale_Hz": 150.0,
        "optimizer_core_thermal_scale_C": None,
        "optimizer_Llt_scale_uH": 0.25,
        "optimizer_all_active_thermal_scale_C": 2.0,
        "acquisition_ranking_contract": None,
    })
    validate_result_seal(result, payload, manifest)

    soft_axis_mutation = copy.deepcopy(result)
    soft_axis_mutation["acquisition_ranking_contract"] = {"active": True}
    with pytest.raises(RuntimeError, match="soft-axis acquisition"):
        validate_result_seal(soft_axis_mutation, payload, manifest)

    override_mutation = copy.deepcopy(result)
    override_mutation["optimizer_constraint_normalization"][
        "explicit_optimizer_only_scale_overrides"
    ]["exterior_width_limit"] = 100.0
    unsigned = {
        key: value
        for key, value in override_mutation[
            "optimizer_constraint_normalization"
        ].items()
        if key != "sha256"
    }
    override_mutation["optimizer_constraint_normalization"]["sha256"] = (
        canonical_sha(unsigned)
    )
    with pytest.raises(RuntimeError, match="optimizer-only pressure seal"):
        validate_result_seal(override_mutation, payload, manifest)
