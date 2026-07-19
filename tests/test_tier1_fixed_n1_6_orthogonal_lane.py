from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from tools import tier1_fixed_n1_6_bridge_warm_handoff as handoff
from tools import tier1_fixed_n1_6_orthogonal_warm_handoff as entrypoint


def test_orthogonal_builder_fails_closed_before_32_authenticated_bridge_terminals(
    tmp_path: Path, monkeypatch,
):
    code = tmp_path / "code"
    code.mkdir()
    generation = {
        "constraint_version": handoff.CONSTRAINT_VERSION,
        "hard_spec_sha256": handoff._json_sha(handoff.HARD_SPEC),
        "source_model_manifest_sha256": "a" * 64,
        "deployment_model_manifest_sha256": "b" * 64,
        "temperature_constraint_contract_sha256": "c" * 64,
        "hard_spec": handoff.HARD_SPEC,
        "nsga_code_revision": handoff.EXPECTED_NSGA_REVISION,
    }

    monkeypatch.setattr(
        handoff,
        "_verify_nsga_code_root",
        lambda _root: {
            "revision": handoff.EXPECTED_NSGA_REVISION,
            "clean": True,
            "decoder_relative_path": handoff.NSGA_DECODER_RELATIVE_PATH,
            "decoder_sha256": "d" * 64,
        },
    )

    def capture(_root: Path, role: str) -> dict:
        return {"evidence": {
            "generation_identity": generation,
            "terminal_result_count": 31 if role == "bridge" else 64,
        }}

    monkeypatch.setattr(handoff, "_capture_authenticated_source", capture)
    with pytest.raises(RuntimeError, match="at least 32 authenticated bridge"):
        handoff.build_orthogonal_handoff(
            normalized_root=tmp_path / "normalized",
            thermal_root=tmp_path / "thermal",
            bridge_root=tmp_path / "bridge",
            fixed6_root=tmp_path / "fixed6",
            nsga_code_root=code,
            output=tmp_path / "output",
        )
    assert not (tmp_path / "output").exists()


def test_orthogonal_builder_seals_exact_three_way_quota_and_interleave(
    tmp_path: Path, monkeypatch,
):
    code = tmp_path / "code"
    code.mkdir()
    generation = {
        "constraint_version": handoff.CONSTRAINT_VERSION,
        "hard_spec_sha256": handoff._json_sha(handoff.HARD_SPEC),
        "source_model_manifest_sha256": "a" * 64,
        "deployment_model_manifest_sha256": "b" * 64,
        "temperature_constraint_contract_sha256": "c" * 64,
        "hard_spec": handoff.HARD_SPEC,
        "nsga_code_revision": handoff.EXPECTED_NSGA_REVISION,
    }
    role_specs = {
        "normalized": (12, 57059, True, False),
        "thermal": (12, 59279, True, False),
        "bridge": (37, 61000, False, False),
        "fixed6": (3, 59035, False, True),
    }
    row_index = 0
    sources = {}
    for role, (count, anchor, thermal_pass, resonance_pass) in role_specs.items():
        rows = []
        for offset in range(count):
            params = {"N1": 6, "coordinate_index": row_index}
            constraints = {
                name: -1.0
                for name in handoff.sealed.EXPECTED_CONSTRAINT_NAMES
            }
            constraints[handoff.RESONANCE_CONSTRAINT] = (
                -100.0 if resonance_pass else 100.0
            )
            constraints["Llt_robust_band"] = (
                -0.1 if role != "bridge" else 0.4
            )
            constraints["Llt_ensemble_disagreement"] = (
                -0.1 if role != "bridge" else 0.6
            )
            if role == "bridge":
                constraints["Llt_robust_band"] += 0.01 * (row_index % 11)
                constraints["Llt_ensemble_disagreement"] += (
                    0.02 * (row_index % 7)
                )
                constraints[handoff.RESONANCE_CONSTRAINT] += (
                    3.0 * (row_index % 13)
                )
                for thermal_index, name in enumerate(
                    handoff.THERMAL_CONSTRAINTS
                ):
                    constraints[name] = float(
                        ((row_index + 3 * thermal_index) % 9) - 3
                    )
                for constraint_index, name in enumerate(
                    handoff.sealed.EXPECTED_CONSTRAINT_NAMES
                ):
                    if name not in {
                        *handoff.THERMAL_CONSTRAINTS,
                        *handoff.LLT_CONSTRAINTS,
                        handoff.RESONANCE_CONSTRAINT,
                    }:
                        constraints[name] = float(
                            -1.0 - abs(np.sin(
                                (row_index + 1) * (constraint_index + 1)
                            ))
                        )
            rows.append({
                "role": role,
                "decoded_params": params,
                "decoded_params_sha256": handoff._json_sha(params),
                "constraint_G": constraints,
                "reference": {
                    "cohort_id": "cohort",
                    "task_id": anchor if offset == 0 else 70_000 + row_index,
                    "seed": 80_000 + row_index,
                    "seed_status_sha256": f"{90_000 + row_index:064x}",
                    "result_sha256": f"{100_000 + row_index:064x}",
                },
                "original_N1": 6,
                "all_thermal_pass": thermal_pass,
                "Llt_pass": role != "bridge",
                "resonance_pass": resonance_pass,
                "orthogonal_other_pass": True,
                "thermal_positive_sum_C": 0.0 if thermal_pass else 4.0,
                "thermal_positive_max_C": 0.0 if thermal_pass else 2.0,
                "core_positive_sum_C": 0.0,
                "core_positive_max_C": 0.0,
            })
            row_index += 1
        sources[role] = {
            "rows": rows,
            "evidence": {
                "generation_identity": generation,
                "terminal_result_count": 40 if role == "bridge" else count,
                "status": {"sha256": f"{200_000 + len(sources):064x}"},
            },
        }

    monkeypatch.setattr(
        handoff,
        "_verify_nsga_code_root",
        lambda _root: {
            "revision": handoff.EXPECTED_NSGA_REVISION,
            "clean": True,
            "decoder_relative_path": handoff.NSGA_DECODER_RELATIVE_PATH,
            "decoder_sha256": "d" * 64,
        },
    )
    monkeypatch.setattr(
        handoff, "_capture_authenticated_source",
        lambda _root, role: sources[role],
    )
    monkeypatch.setattr(
        handoff, "_load_sobol_schema",
        lambda _root: ([object()] * 25, 5, 8, {}),
    )
    monkeypatch.setattr(
        handoff, "_prepare_rows",
        lambda source: [dict(row) for row in source["rows"]],
    )
    monkeypatch.setattr(
        handoff, "decoded_to_unit",
        lambda params, *_args: np.asarray([
            (
                (params["coordinate_index"] + 1)
                * (2 * dimension + 3)
                + dimension * dimension
            ) % 101 / 100.0
            for dimension in range(25)
        ], dtype=float),
    )
    monkeypatch.setattr(
        handoff, "_seal_source_snapshots", lambda _output, _sources: {},
    )
    monkeypatch.setattr(
        handoff, "_seal_selected_artifacts", lambda _output, _selected: [],
    )

    result = handoff.build_orthogonal_handoff(
        normalized_root=tmp_path / "normalized",
        thermal_root=tmp_path / "thermal",
        bridge_root=tmp_path / "bridge",
        fixed6_root=tmp_path / "fixed6",
        nsga_code_root=code,
        output=tmp_path / "output",
    )
    assert result["category_counts"] == {
        "thermal_llt_pass_branch": 24,
        "bridge_near_branch": 37,
        "resonance_llt_pass_branch": 3,
    }
    assert result["source_role_counts"] == {
        "normalized": 12, "thermal": 12, "bridge": 37, "fixed6": 3,
    }
    assert result["bridge_minimum_terminal_contract"] == {
        "minimum_authenticated_terminal_count": 32,
        "observed_authenticated_terminal_count": 40,
        "captured_status_sha256": sources["bridge"]["evidence"]["status"][
            "sha256"
        ],
        "latest_authenticated_head_required": True,
        "satisfied": True,
    }
    expected_categories = []
    for index in range(37):
        if index < 24:
            expected_categories.append("thermal_llt_pass_branch")
        expected_categories.append("bridge_near_branch")
        if index < 3:
            expected_categories.append("resonance_llt_pass_branch")
    assert [
        item["category"] for item in result["selected_provenance"]
    ] == expected_categories
    assert result["optimizer_Llt_scale_uH"] == 0.25
    assert result["optimizer_Llt_disagreement_scale_uH"] == 0.5
    assert result["optimizer_all_active_thermal_scale_C"] == 2.0
    assert result["soft_axis_pressure_scope"] == (
        "disabled_until_stage_feasible"
    )
    assert result["acquisition_ranking_contract"] is None
    assert result["hard_constraint_mutation"] is False
    assert result["bridge_diversity_evidence"]["passed"] is True
    assert result["bridge_diversity_evidence"][
        "decoded_params_sha256_unique_count"
    ] == 37
    assert result["quota_revision_evidence"]["revised_counts"] == (
        result["category_counts"]
    )
    assert result["fixed6_resonance_llt_preservation"][
        "all_authenticated_unique_points_preserved"
    ] is True
    assert result["fixed6_resonance_llt_preservation"][
        "selected_unique_count"
    ] == 3
    assert result["launch_performed"] is False
    assert result["scheduler_write_performed"] is False


def _orthogonal_preflight_pair() -> tuple[dict, dict]:
    warm_sha = "e" * 64
    source_model_sha = "a" * 64
    temperature_sha = "b" * 64
    generation = {
        "source_model_manifest_sha256": source_model_sha,
        "temperature_constraint_contract_sha256": temperature_sha,
    }
    contract = {
        "schema_version": handoff.ORTHOGONAL_SCHEMA,
        "nsga_code_revision": handoff.EXPECTED_NSGA_REVISION,
        "constraint_version": handoff.CONSTRAINT_VERSION,
        "hard_spec": handoff.HARD_SPEC,
        "hard_spec_sha256": handoff._json_sha(handoff.HARD_SPEC),
        "constraint_names": list(handoff.sealed.EXPECTED_CONSTRAINT_NAMES),
        "warm_start": {"sha256": warm_sha},
        "source_evidence": {
            role: {"generation_identity": generation}
            for role in handoff.ORTHOGONAL_SOURCE_ROLES
        },
    }
    repair = {
        "schema_version": "mft-tier1-fixed-primary-turns-repair-v1",
        "fixed_primary_turns": 6,
        "coordinate_index": 0,
        "hard_constraint_mutation": False,
        "objective_mutation": False,
        "warm_start_post_repair_minimum_unique_count": 32,
    }
    repair["sha256"] = handoff._json_sha(repair)
    scales = {
        name: 1.0 for name in handoff.sealed.EXPECTED_CONSTRAINT_NAMES
    }
    scales.update({
        "Llt_robust_band": 0.25,
        "Llt_ensemble_disagreement": 0.5,
        handoff.RESONANCE_CONSTRAINT: 150.0,
        **{name: 2.0 for name in handoff.THERMAL_CONSTRAINTS},
    })
    overrides = {
        "Llt_robust_band": 0.25,
        "Llt_ensemble_disagreement": 0.5,
        **{name: 2.0 for name in handoff.THERMAL_CONSTRAINTS},
    }
    normalization = {
        "schema_version": "mft-tier1-optimizer-constraint-normalization-v1",
        "purpose": "optimizer_search_pressure_only",
        "authoritative_terminal_G": "physical_unscaled",
        "formula": "G_optimizer=G_physical/engineering_scale",
        "constraint_order": list(handoff.sealed.EXPECTED_CONSTRAINT_NAMES),
        "scales": scales,
        "optimizer_Llt_scale_uH": 0.25,
        "optimizer_all_active_thermal_scale_C": 2.0,
        "optimizer_all_active_thermal_constraints": list(
            handoff.THERMAL_CONSTRAINTS
        ),
        "all_active_thermal_side_activation": (
            "finite_N2_side_gt_0_else_physical_negative_BIG"
        ),
        "explicit_optimizer_only_scale_overrides": overrides,
        "override_scope": (
            "sealed_Llt_and_all_active_thermal_constraints_"
            "physical_G_unchanged"
        ),
    }
    normalization["sha256"] = handoff._json_sha(normalization)
    result = {
        "schema_version": handoff.sealed.SEARCH_SCHEMA,
        "seed": 1_907_198_999,
        "population": 64,
        "max_generations": 1,
        "completed_generations": 1,
        "nsga_code_revision": handoff.EXPECTED_NSGA_REVISION,
        "constraint_version": handoff.CONSTRAINT_VERSION,
        "hard_spec": handoff.HARD_SPEC,
        "hard_spec_sha256": handoff._json_sha(handoff.HARD_SPEC),
        "model_manifest_sha256": source_model_sha,
        "temperature_constraint_contract_sha256": temperature_sha,
        "constraint_names": list(handoff.sealed.EXPECTED_CONSTRAINT_NAMES),
        "fixed_primary_turns": 6,
        "fixed_primary_turns_contract": repair,
        "terminal_population_fixed_primary_turns_verified": True,
        "terminal_population_primary_turn_values": [6],
        "warm_start": {
            "sha256": warm_sha,
            "coordinate_count": 64,
            "dimension_count": 25,
            "post_repair_coordinate_count": 64,
            "repaired_by_current_simultaneous_hard_constraint_problem": True,
            "fixed_primary_turns_repair": repair,
            "fixed_primary_turns_warm_audit": {
                "schema_version": (
                    "mft-tier1-fixed-primary-turns-warm-audit-v1"
                ),
                "source_coordinate_count": 64,
                "post_repair_coordinate_shape": [64, 25],
                "post_repair_coordinate_dtype": "float64",
                "post_repair_coordinates_sha256": "f" * 64,
                "post_repair_decoded_unique_count": 64,
                "post_repair_hard_feasible_count": 64,
                "post_repair_rejected_count": 0,
                "minimum_unique_count": 32,
                "fixed_primary_turns": 6,
                "fixed_primary_turn_coordinate_verified": True,
                "physical_geometry_dedupe_performed": True,
                "source_sha_verified_before_repair": True,
            },
        },
        "initialization_audit": {"warm_filter": {
            "input_count": 64,
            "hard_feasible_count": 64,
            "decoded_unique_count": 64,
            "rejected_count": 0,
        }},
        "optimizer_constraint_normalization": normalization,
        "optimizer_resonance_scale_Hz": 150.0,
        "optimizer_core_thermal_scale_C": None,
        "optimizer_Llt_scale_uH": 0.25,
        "optimizer_all_active_thermal_scale_C": 2.0,
        "acquisition_ranking_contract": None,
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    return contract, result


def test_orthogonal_preflight_seals_exact_pressure_and_no_soft_axis():
    contract, result = _orthogonal_preflight_pair()
    summary = handoff._orthogonal_post_repair_preflight_summary(
        contract, result,
    )
    assert summary["schema_version"] == (
        handoff.ORTHOGONAL_POST_REPAIR_PREFLIGHT_SCHEMA
    )
    assert summary["optimizer_Llt_scale_uH"] == 0.25
    assert summary["optimizer_Llt_disagreement_scale_uH"] == 0.5
    assert summary["optimizer_all_active_thermal_scale_C"] == 2.0
    assert summary["optimizer_all_active_thermal_constraint_count"] == 11
    assert summary["soft_axis_pressure_scope"] == (
        "disabled_until_stage_feasible"
    )

    mutated = copy.deepcopy(result)
    mutated["acquisition_ranking_contract"] = {"active": True}
    with pytest.raises(RuntimeError, match="orthogonal post-repair"):
        handoff._orthogonal_post_repair_preflight_summary(contract, mutated)

    mutated = copy.deepcopy(result)
    mutated["optimizer_constraint_normalization"]["scales"][
        handoff.THERMAL_CONSTRAINTS[-1]
    ] = 5.5
    unsigned = {
        key: value
        for key, value in mutated[
            "optimizer_constraint_normalization"
        ].items()
        if key != "sha256"
    }
    mutated["optimizer_constraint_normalization"]["sha256"] = (
        handoff._json_sha(unsigned)
    )
    with pytest.raises(RuntimeError, match="orthogonal post-repair"):
        handoff._orthogonal_post_repair_preflight_summary(contract, mutated)


def test_orthogonal_warm_entrypoint_defaults_are_non_scheduler_writes():
    args = entrypoint.parser().parse_args([])
    assert args.output == handoff.DEFAULT_ORTHOGONAL_OUTPUT
    assert args.bridge_root == handoff.DEFAULT_BRIDGE
    assert args.seal_post_repair_preflight_result is None
