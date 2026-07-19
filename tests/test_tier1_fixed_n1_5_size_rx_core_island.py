from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tools import tier1_fixed_n1_5_size_rx_core_warm_handoff as island


REPO = Path(__file__).resolve().parents[1]


def _constraints(**overrides) -> dict:
    values = {
        name: -1.0 for name in island.sealed.EXPECTED_CONSTRAINT_NAMES
    }
    values.update({
        "decoded_space_shrink": 0.0,
        "minimum_physical_insulation": 0.0,
        "core_group_manufacturability_limit": 0.0,
        "exterior_width_limit": -1.0,
        "exterior_length_limit": -100.0,
        "exterior_height_limit": -100.0,
        island.RESONANCE_CONSTRAINT: -100.0,
        "Llt_robust_band": -0.01,
        **overrides,
    })
    return values


def _params(index: int) -> dict:
    return {
        "N1": 5,
        "N1_main": 5,
        "N1_side": 0,
        "N2": 50,
        "N2_main": 30,
        "N2_side": 20,
        "n_core_group": 4,
        "cw1": 5.0,
        "core_plate_on": 1,
        "wcp_on": 1,
        "cap_on": 1,
        "test_index": index,
    }


def _prepared_row(
    index: int, *, role: str = "resfocus", task_id: int | None = None,
    values: dict | None = None,
) -> dict:
    values = values or _constraints()
    params = _params(index)
    return {
        "role": role,
        "decoded_params": params,
        "decoded_params_sha256": island._json_sha(params),
        "constraint_G": values,
        "constraint_G_sha256": island._json_sha(values),
        "coordinate": np.linspace(0.0, 1.0, 25) + index * 1e-5,
        "topology_pass": island._topology_pass(params),
        "width_mm": 1_200.0 + values["exterior_width_limit"],
        "length_mm": 1_200.0 + values["exterior_length_limit"],
        "height_mm": 750.0 + values["exterior_height_limit"],
        "rx_max_G_C": max(
            values[name] for name in island.RX_THERMAL_CONSTRAINTS
        ),
        "non_core_thermal_max_G_C": max(
            values[name] for name in island.NON_CORE_THERMAL_CONSTRAINTS
        ),
        "core_max_G_C": max(
            values[name] for name in island.CORE_THERMAL_CONSTRAINTS
        ),
        "core_positive_sum_C": sum(
            max(values[name], 0.0)
            for name in island.CORE_THERMAL_CONSTRAINTS
        ),
        "reference": {
            "cohort_id": "cohort",
            "task_id": task_id if task_id is not None else 60_000 + index,
            "seed": 1_000 + index,
            "seed_status_sha256": f"{100_000 + index:064x}",
            "result_sha256": f"{200_000 + index:064x}",
        },
    }


def test_island_eligibility_preserves_width_rx_resonance_and_llt_scope():
    strict = _prepared_row(0)
    assert island._base_preserved(strict)
    assert island._preserved_island_eligible(strict)
    assert island._core_bridge_eligible(strict)

    width_fail = _prepared_row(
        1, values=_constraints(exterior_width_limit=0.01),
    )
    rx_fail = _prepared_row(
        2, values=_constraints(
            **{"temperature_robust_limit:T_max_Rx_main": 0.01}
        ),
    )
    resonance_fail = _prepared_row(
        3, values=_constraints(half_magnetizing_resonance_minimum=0.01),
    )
    for row in (width_fail, rx_fail, resonance_fail):
        assert not island._base_preserved(row)
        assert not island._preserved_island_eligible(row)
        assert not island._core_bridge_eligible(row)

    bridge = _prepared_row(
        4, values=_constraints(
            Llt_robust_band=6.73,
            strict_full_density_support=0.39,
            **{
                "temperature_robust_limit:T_max_core": 1.58,
                "temperature_robust_limit:Tprobe_core_center_max": 1.17,
                "temperature_robust_limit:Tprobe_core_top_yoke_max": 1.59,
            },
        ),
    )
    assert island._base_preserved(bridge)
    assert not island._preserved_island_eligible(bridge)
    assert island._core_bridge_eligible(bridge)
    bridge["constraint_G"]["Llt_robust_band"] = 10.51
    assert not island._core_bridge_eligible(bridge)


def test_diverse_quota_requires_anchor_and_caps_decoded_sha_duplicates():
    rows = [
        _prepared_row(
            index,
            task_id=56699 if index == 0 else None,
        )
        for index in range(40)
    ]
    rows.append(copy.deepcopy(rows[7]))
    selected = island._select_diverse(
        rows, 32, anchor_task=56699, rank_key=island._preserved_rank,
    )
    assert len(selected) == 32
    assert any(row["reference"]["task_id"] == 56699 for row in selected)
    assert len({row["decoded_params_sha256"] for row in selected}) == 32

    with pytest.raises(RuntimeError, match="required anchor"):
        island._select_diverse(
            rows, 32, anchor_task=99999, rank_key=island._preserved_rank,
        )


def test_required_anchor_identity_is_exact_and_fail_closed():
    for role in island.SOURCE_ROLES:
        expected = island.EXPECTED_ANCHOR_IDENTITIES[role]
        row = _prepared_row(
            0, role=role, task_id=expected["task_id"],
        )
        row["reference"].update({
            "seed": expected["seed"],
            "result_sha256": expected["result_sha256"],
        })
        row["decoded_params_sha256"] = expected["decoded_params_sha256"]
        row["constraint_G_sha256"] = expected["constraint_G_sha256"]
        assert island._verify_required_anchor([row], role) is row

        tampered = copy.deepcopy(row)
        tampered["reference"]["result_sha256"] = "f" * 64
        with pytest.raises(RuntimeError, match="SHA/physics identity"):
            island._verify_required_anchor([tampered], role)


def _preflight_fixture() -> tuple[dict, dict]:
    source_model_sha = "4" * 64
    temperature_sha = "9" * 64
    warm_sha = "5" * 64
    contract = {
        "nsga_code_revision": island.EXPECTED_NSGA_REVISION,
        "constraint_version": island.CONSTRAINT_VERSION,
        "hard_spec": island.HARD_SPEC,
        "hard_spec_sha256": island._json_sha(island.HARD_SPEC),
        "constraint_names": list(island.sealed.EXPECTED_CONSTRAINT_NAMES),
        "warm_start": {"sha256": warm_sha},
        "source_evidence": {
            role: {"generation_identity": {
                "source_model_manifest_sha256": source_model_sha,
                "temperature_constraint_contract_sha256": temperature_sha,
            }}
            for role in island.SOURCE_ROLES
        },
    }
    repair = {
        "schema_version": "mft-tier1-fixed-primary-turns-repair-v1",
        "fixed_primary_turns": 5,
        "coordinate_index": 0,
        "hard_constraint_mutation": False,
        "objective_mutation": False,
        "warm_start_post_repair_minimum_unique_count": 32,
    }
    repair["sha256"] = island._json_sha(repair)
    scales = {
        name: 5.5 for name in island.sealed.EXPECTED_CONSTRAINT_NAMES
    }
    scales["Llt_robust_band"] = 0.55
    scales[island.RESONANCE_CONSTRAINT] = 150.0
    optimizer_core = {
        "temperature_robust_limit:T_max_core": 1.0,
        "temperature_robust_limit:Tprobe_core_center_max": 1.0,
        "temperature_robust_limit:Tprobe_core_top_yoke_max": 1.0,
    }
    scales.update(optimizer_core)
    normalization = {
        "schema_version": "mft-tier1-optimizer-constraint-normalization-v1",
        "constraint_order": list(island.sealed.EXPECTED_CONSTRAINT_NAMES),
        "authoritative_terminal_G": "physical_unscaled",
        "scales": scales,
        "explicit_optimizer_only_scale_overrides": optimizer_core,
    }
    normalization["sha256"] = island._json_sha(normalization)
    audit = {
        "schema_version": "mft-tier1-fixed-primary-turns-warm-audit-v1",
        "source_coordinate_count": 64,
        "post_repair_coordinate_shape": [64, 25],
        "post_repair_coordinate_dtype": "float64",
        "post_repair_decoded_unique_count": 64,
        "post_repair_hard_feasible_count": 64,
        "post_repair_rejected_count": 0,
        "minimum_unique_count": 32,
        "fixed_primary_turns": 5,
        "fixed_primary_turn_coordinate_verified": True,
        "physical_geometry_dedupe_performed": True,
        "source_sha_verified_before_repair": True,
        "post_repair_coordinates_sha256": "6" * 64,
    }
    result = {
        "schema_version": island.sealed.SEARCH_SCHEMA,
        "seed": 123,
        "population": 64,
        "max_generations": 1,
        "completed_generations": 2,
        "nsga_code_revision": island.EXPECTED_NSGA_REVISION,
        "constraint_version": island.CONSTRAINT_VERSION,
        "hard_spec": island.HARD_SPEC,
        "hard_spec_sha256": island._json_sha(island.HARD_SPEC),
        "model_manifest_sha256": source_model_sha,
        "temperature_constraint_contract_sha256": temperature_sha,
        "constraint_names": list(island.sealed.EXPECTED_CONSTRAINT_NAMES),
        "fixed_primary_turns": 5,
        "terminal_population_fixed_primary_turns_verified": True,
        "terminal_population_primary_turn_values": [5],
        "fixed_primary_turns_contract": repair,
        "warm_start": {
            "sha256": warm_sha,
            "coordinate_count": 64,
            "dimension_count": 25,
            "post_repair_coordinate_count": 64,
            "repaired_by_current_simultaneous_hard_constraint_problem": True,
            "fixed_primary_turns_repair": repair,
            "fixed_primary_turns_warm_audit": audit,
        },
        "initialization_audit": {"warm_filter": {
            "input_count": 64,
            "hard_feasible_count": 64,
            "decoded_unique_count": 64,
            "rejected_count": 0,
        }},
        "optimizer_constraint_normalization": normalization,
        "acquisition_ranking_contract": {
            "active": True,
            "hard_constraint_mutation": False,
            "authoritative_terminal_G": "physical_unscaled_unchanged",
            "core_thermal_positive_G_scale_C": 1.0,
            "soft_axis_targets_mm": {
                "exterior_width": 1_000.0,
                "exterior_length": 1_000.0,
            },
        },
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    return contract, result


def test_post_repair_preflight_seals_fixed_n1_and_unscaled_hard_contract():
    contract, result = _preflight_fixture()
    summary = island._post_repair_preflight_summary(contract, result)
    assert summary["fixed_primary_turns"] == 5
    assert summary["post_repair_decoded_unique_count"] == 64
    assert summary["optimizer_resonance_scale_Hz"] == 150.0
    assert summary["optimizer_core_thermal_scale_C"] == 1.0

    tampered = copy.deepcopy(result)
    tampered["terminal_population_primary_turn_values"] = [5, 6]
    with pytest.raises(RuntimeError, match="preflight contract mismatch"):
        island._post_repair_preflight_summary(contract, tampered)
    tampered = copy.deepcopy(result)
    tampered["optimizer_constraint_normalization"]["scales"][
        island.RESONANCE_CONSTRAINT
    ] = 151.0
    with pytest.raises(RuntimeError, match="preflight contract mismatch"):
        island._post_repair_preflight_summary(contract, tampered)


def test_size_rx_core_variant_identity_and_fail_closed_profile():
    code = """\
import json
from tools import tier1_fixed_n1_5_size_rx_core_slurm_rolling as entry
r = entry.rolling
print(json.dumps({
    'variant': r.SEARCH_VARIANT,
    'namespace': r.SEARCH_NAMESPACE,
    'turns': r.FIXED_PRIMARY_TURNS,
    'profile': r.search_profile(),
    'acquisition': r.thermal_crossover_acquisition_contract({
        'size_W_max_mm': 1200.0, 'size_L_max_mm': 1200.0,
    }),
    'runtime': str(r.RUNTIME),
}, sort_keys=True))
"""
    environment = os.environ.copy()
    environment.pop("MFT_TIER1_SEARCH_VARIANT", None)
    process = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, env=environment,
        capture_output=True, text=True, check=False,
    )
    assert process.returncode == 0, process.stderr
    value = json.loads(process.stdout)
    profile = value["profile"]
    acquisition = value["acquisition"]
    assert value["variant"] == "fixed-n1-5-size-rx-core-v1"
    assert value["turns"] == 5
    assert profile["optimizer_resonance_scale_Hz"] == 150.0
    assert profile["optimizer_core_thermal_scale_C"] == 1.0
    assert profile["optimizer_Llt_scale_uH"] == 0.55
    assert profile["soft_axis_target_W_mm"] == 1_000.0
    assert profile["soft_axis_target_L_mm"] == 1_000.0
    assert acquisition["hard_constraint_mutation"] is False
    assert acquisition["authoritative_terminal_G"] == (
        "physical_unscaled_unchanged"
    )
    assert value["runtime"].endswith(
        "mft_tier1_nsga_n1_5_size_rx_core_t110_res15k_260719"
    )

    conflict = environment.copy()
    conflict["MFT_TIER1_SEARCH_VARIANT"] = "fixed-n1-6-thermal-bridge-v1"
    process = subprocess.run(
        [
            sys.executable, "-c",
            "import tools.tier1_fixed_n1_5_size_rx_core_slurm_rolling",
        ],
        cwd=REPO, env=conflict, capture_output=True, text=True, check=False,
    )
    assert process.returncode != 0
    assert "conflicts with fixed-N1=5 size/Rx/core" in process.stderr
