from __future__ import annotations

import copy
import json
from pathlib import Path
import types

import numpy as np
import pytest

from module import input_parameter_260706 as current_input
from module import mft_goal_20260726_contract as goal
from regression_260707.verify import finalize
from tools import tier1_corrected_generation_adapter as adapter
from tools import tier1_corrected_generation_preflight as preflight
from tools import tier1_final1000_multiseed_consumer as consumer


REPO = Path(__file__).resolve().parents[1]


class _Predictor:
    def __init__(self, value: float, half_width: float = 0.25):
        self.value = float(value)
        self.half_width = float(half_width)
        self.features = ["l1"]

    def predict_mu_sigma(self, frame, conformal=True):
        assert conformal is True
        return (
            np.full(len(frame), self.value),
            np.full(len(frame), self.half_width),
        )

    def disagreement(self, frame):
        return np.zeros(len(frame))


def _models():
    values = {
        target: 1.0 for target in adapter.GOAL_REQUIRED_MODEL_TARGETS
    }
    values.update(
        {
            "Llt_phys": 27.5,
            "k": 0.9,
            "C_tx_tx_F": 1.0e-12,
            "C_rx_rx_F": 1.0e-14,
            "C_tx_rx_F": 5.0e-10,
            **{
                target: 100.0
                for target in goal.WINDING_TEMPERATURE_TARGETS
            },
            **{
                target: 119.0
                for target in goal.CORE_TEMPERATURE_TARGETS
            },
        }
    )
    return {target: _Predictor(value) for target, value in values.items()}


def _goal_problem(primary_turns=5):
    modules = preflight.load_current7_modules(REPO)
    problem_class = preflight.create_current7_problem_class(
        base_problem_class=modules.nsga2_problem.MFTProblem,
        base_constraint_names=modules.nsga2_problem.CONSTRAINT_NAMES,
        base_fixed_stack_mm=modules.nsga2_problem.NSGA_FIXED_THERMAL_STACK_MM,
        sobol_dims=modules.input_parameter._SOBOL_DIMS,
        bounding_box_lit=modules.geometry_metrics.bounding_box_lit,
        input_parameter_module=modules.input_parameter,
        design_analytical_b_field_t=(
            modules.nsga2_problem.design_analytical_b_field_t
        ),
    )
    return problem_class(
        _models(),
        spec=goal.GOAL_STAGE_SPEC,
        density_gate=lambda frame: np.full(len(frame), -1.0),
        fixed_primary_turns=primary_turns,
    )


def _coordinate(problem, *, cw1_unit=0.5, side_unit=0.4):
    result = np.full((1, problem.n_var), 0.5)
    result[0, problem.cw1_coordinate_index] = cw1_unit
    result[0, 2] = side_unit
    return problem.repair_unit_coordinates(result)


def test_goal_contract_is_exact_and_forbids_legacy_scalar_fields():
    assert goal.validate_goal_stage_spec(goal.GOAL_STAGE_SPEC) == (
        goal.GOAL_STAGE_SPEC
    )
    assert goal.GOAL_SIZE_LIMITS_MM == {
        "W": 1200.0,
        "L": 1000.0,
        "H": 750.0,
    }
    assert goal.TEMPERATURE_FAMILY_LIMITS_C == {
        "winding": 100.0,
        "core": 120.0,
    }
    assert goal.GOAL_RESONANCE_MIN_HZ == 15_000.0
    assert "resonance_max_Hz" not in goal.GOAL_STAGE_SPEC
    for legacy in (
        "T_limit_C",
        "n_core_group_max",
        "primary_conductor_thickness_mm",
        "f1_split",
    ):
        mutated = copy.deepcopy(goal.GOAL_STAGE_SPEC)
        mutated[legacy] = 100.0
        with pytest.raises(goal.GoalContractError, match="legacy fields"):
            goal.validate_goal_stage_spec(mutated)


def test_cw1_grid_has_exact_endpoints_and_no_independent_f1_split():
    assert goal.cw1_from_unit_coordinate(0.0) == 1.0
    assert goal.cw1_from_unit_coordinate(0.5) == 5.5
    assert goal.cw1_from_unit_coordinate(1.0) == 10.0
    for value in (1.0, 1.01, 5.5, 9.99, 10.0):
        assert goal.validate_cw1_mm(value) == value
        assert goal.cw1_from_unit_coordinate(
            goal.cw1_unit_coordinate(value)
        ) == value
    for value in (0.99, 1.005, 10.01):
        with pytest.raises(goal.GoalContractError):
            goal.validate_cw1_mm(value)
    assert (
        goal.GOAL_STAGE_SPEC["cw1_search_mm"][
            "f1_split_independent_search_allowed"
        ]
        is False
    )


@pytest.mark.parametrize("primary_turns", [5, 6, 7, 8])
@pytest.mark.parametrize(
    ("unit", "expected_cw1"),
    [(0.0, 1.0), (0.5, 5.5), (1.0, 10.0)],
)
def test_goal_problem_uses_split_temperature_and_generalized_budget(
    primary_turns, unit, expected_cw1
):
    problem = _goal_problem(primary_turns)
    repaired = _coordinate(problem, cw1_unit=unit)
    assert np.array_equal(
        problem.repair_unit_coordinates(repaired), repaired
    )
    output = {}
    problem._evaluate(repaired, output)
    assert output["decoder_valid"].tolist() == [True]
    row = output["frame"].iloc[0]
    assert row["cw1"] == expected_cw1
    assert (
        int(row["N1_main"]) + int(row["N1_side"])
        == primary_turns
    )
    assert 2 <= int(row["n_core_group"]) <= 10
    assert goal.dynamic_core_group_violation(row) <= 0.0
    budget = preflight.winding_budget_identity(
        row, expected_cw1_mm=expected_cw1
    )
    assert budget["passed"] is True
    assert budget["post_decode_field_override_performed"] is False
    winding_index = problem.constraint_names.index(
        "temperature_robust_limit:Tprobe_Tx_leeward_max"
    )
    body_winding_index = problem.constraint_names.index(
        "temperature_robust_limit:T_max_Tx"
    )
    core_index = problem.constraint_names.index(
        "temperature_robust_limit:Tprobe_core_center_max"
    )
    assert output["G"][0, winding_index] == pytest.approx(0.25)
    assert output["G"][0, body_winding_index] == pytest.approx(0.25)
    assert output["G"][0, core_index] == pytest.approx(-0.75)
    assert "core_group_dynamic_validity" in problem.constraint_names
    assert "core_group_manufacturability_limit" not in (
        problem.constraint_names
    )
    assert problem.temperature_contract == goal.GOAL_TEMPERATURE_CONTRACT
    assert problem.spec.get("T_limit_C") is None


def test_goal_problem_rejects_old_turn_strata_and_fixed_identity_override():
    with pytest.raises(ValueError, match="campaign strata"):
        _goal_problem(4)
    with pytest.raises(ValueError, match="campaign strata"):
        _goal_problem(9)

    modules = preflight.load_current7_modules(REPO)
    problem_class = preflight.create_current7_problem_class(
        base_problem_class=modules.nsga2_problem.MFTProblem,
        base_constraint_names=modules.nsga2_problem.CONSTRAINT_NAMES,
        base_fixed_stack_mm=modules.nsga2_problem.NSGA_FIXED_THERMAL_STACK_MM,
        sobol_dims=modules.input_parameter._SOBOL_DIMS,
        bounding_box_lit=modules.geometry_metrics.bounding_box_lit,
        input_parameter_module=modules.input_parameter,
        design_analytical_b_field_t=(
            modules.nsga2_problem.design_analytical_b_field_t
        ),
    )
    with pytest.raises(ValueError, match="fan_velocity"):
        problem_class(
            _models(),
            spec=goal.GOAL_STAGE_SPEC,
            density_gate=lambda frame: np.full(len(frame), -1.0),
            fixed_overrides={"fan_velocity": 2.0},
            fixed_primary_turns=5,
        )


def test_goal_side_winding_masks_both_body_and_probe_constraints():
    problem = _goal_problem(5)
    coordinate = _coordinate(problem, side_unit=0.0)
    output = {}
    problem._evaluate(coordinate, output)
    assert int(output["frame"].iloc[0]["N2_side"]) == 0
    for target in ("T_max_Rx_side", "Tprobe_Rx_side_leeward_max"):
        index = problem.constraint_names.index(
            f"temperature_robust_limit:{target}"
        )
        assert output["G"][0, index] == -preflight.BIG


def test_goal_g0_has_25_targets_and_exact_body_quality_thresholds():
    assert tuple(adapter.GOAL_REQUIRED_MODEL_TARGETS[-11:]) == (
        goal.GOAL_TEMPERATURE_TARGETS
    )
    assert len(goal.GOAL_G0_MODEL_TARGETS) == 25
    assert goal.GOAL_G0_MODEL_TARGETS[-11:] == (
        *goal.PROBE_TEMPERATURE_TARGETS,
        *goal.BODY_WINDING_TEMPERATURE_TARGETS,
        *goal.BODY_CORE_TEMPERATURE_TARGETS,
    )
    assert set(goal.GOAL_G0_MODEL_TARGETS) == (
        set(adapter.GOAL_REQUIRED_MODEL_TARGETS) | {"B_max_core"}
    )
    thresholds = json.loads(
        (
            REPO
            / "regression_260707"
            / "training"
            / "model_quality_thresholds.json"
        ).read_text(encoding="utf-8")
    )
    expected = {
        "min_r2": 0.85,
        "max_rmse": 5.0,
        "max_p90_ape_pct": 10.0,
        "max_interval_p90_width": 10.0,
    }
    for target in ("T_max_Tx", "T_max_Rx_main", "T_max_Rx_side", "T_max_core"):
        assert thresholds["targets"][target] == expected


def _valid_goal_fea_result():
    problem = _goal_problem(5)
    coordinate = _coordinate(problem, cw1_unit=0.5, side_unit=0.0)
    frame, _shrink, valid = problem.decode_batch(coordinate)
    assert valid.tolist() == [True]
    result = frame.iloc[0].to_dict()
    result.update(
        {
            "goal_contract_schema": goal.GOAL_CONTRACT_SCHEMA,
            "hard_spec": copy.deepcopy(goal.GOAL_STAGE_SPEC),
            "Llt": 27.5,
            "full_model": 1,
            "B_max_core": 1.0,
            "T_max_Tx": 100.0,
            "T_max_Rx_main": 100.0,
            "T_max_core": 120.0,
            "P_winding_total": 3.0,
            "P_Tx_main_group": 1.0,
            "P_Rx_main_group": 2.0,
            "P_Rx_side_total": 0.0,
            "P_core_total": 1.0,
            "P_core_plate_total": 1.0,
            "P_wcp_total": 1.0,
            "f_res_min_tx_rx_only_Hz": 15_000.0,
            "thermal_pad_conductivity_W_mK": 0.2,
            **{
                target: 99.0
                for target in goal.PROBE_WINDING_TEMPERATURE_TARGETS
            },
            **{
                target: 119.0
                for target in goal.PROBE_CORE_TEMPERATURE_TARGETS
            },
        }
    )
    return result


def test_final_gate_uses_split_temperature_box_resonance_and_identity():
    valid = _valid_goal_fea_result()
    assert finalize.physical_spec_reasons(valid) == []

    winding_hot = dict(valid, T_max_Tx=100.01)
    assert "temperature_out_of_spec:T_max_Tx" in (
        finalize.physical_spec_reasons(winding_hot)
    )
    core_hot = dict(valid, T_max_core=120.01)
    assert "temperature_out_of_spec:T_max_core" in (
        finalize.physical_spec_reasons(core_hot)
    )
    hot_probe = dict(valid, Tprobe_Tx_leeward_max=100.01)
    assert "temperature_out_of_spec:Tprobe_Tx_leeward_max" in (
        finalize.physical_spec_reasons(hot_probe)
    )
    scalar = dict(valid, T_limit_C=100.0)
    assert "goal_legacy_scalar_temperature_forbidden" in (
        finalize.physical_spec_reasons(scalar)
    )
    wrong_fan = dict(valid, fan_velocity=1.6)
    assert "fixed_identity_mismatch:fan_velocity" in (
        finalize.physical_spec_reasons(wrong_fan)
    )
    low_resonance = dict(valid, f_res_min_tx_rx_only_Hz=14_999.9)
    assert "self_resonance_below_minimum" in (
        finalize.physical_spec_reasons(low_resonance)
    )


def test_consumer_rejects_scalar_or_wrong_split_contract_identity():
    value = {
        "goal_contract_schema": goal.GOAL_CONTRACT_SCHEMA,
        "hard_spec": copy.deepcopy(goal.GOAL_STAGE_SPEC),
        "stage_spec_sha256": goal.GOAL_STAGE_SPEC_SHA256,
        "temperature_contract_sha256": (
            goal.GOAL_TEMPERATURE_CONTRACT_SHA256
        ),
    }
    assert consumer.validate_goal_result_contract(value) is True
    with pytest.raises(RuntimeError, match="scalar T_limit_C"):
        consumer.validate_goal_result_contract({**value, "T_limit_C": 100.0})
    with pytest.raises(RuntimeError, match="identity mismatch"):
        consumer.validate_goal_result_contract(
            {**value, "temperature_contract_sha256": "0" * 64}
        )


def test_terminal_320_table_contains_physical_dedupe_and_provenance():
    problem = _goal_problem(5)
    one = _coordinate(problem, cw1_unit=0.5, side_unit=0.0)
    coordinates = np.repeat(one, preflight.PRODUCTION_POPULATION, axis=0)
    frame, _shrink, valid = problem.decode_batch(coordinates)
    count = preflight.PRODUCTION_POPULATION
    objectives = np.column_stack(
        (np.full(count, 100.0), np.full(count, 200.0))
    )
    physical_g = np.full((count, problem.n_ieq_constr), -1.0)
    normalized_g = np.full((count, problem.n_ieq_constr), -0.5)
    source = {
        "seed": 42,
        "task_id": "87052",
        "bundle_id": "goal-bundle",
        "island_id": "n1-5",
        "dataset_sha256": "c" * 64,
        "model_artifacts_sha256": "a" * 64,
        "model_generation_sha256": "b" * 64,
        "evaluation_spec_sha256": problem.stage_spec_sha256,
        "temperature_contract_sha256": (
            problem.temperature_contract_sha256
        ),
        "hard_constraint_contract_sha256": (
            problem.hard_constraint_contract_sha256
        ),
    }
    table = preflight._terminal_physical_candidate_frame(
        types.SimpleNamespace(problem=problem),
        coordinates=coordinates,
        objectives=objectives,
        optimizer_constraints=normalized_g,
        physical_constraints=physical_g,
        frame=frame,
        decoder_valid=valid,
        source_identity=source,
    )
    assert len(table) == 320
    assert table["terminal_population_index"].tolist() == list(range(320))
    assert table["candidate_physics_sha"].nunique() == 1
    assert (
        table["candidate_physics_sha"]
        == table["physical_geometry_sha256"]
    ).all()
    assert table["source_seed"].unique().tolist() == [42]
    assert table["source_task_id"].unique().tolist() == ["87052"]
    assert table["source_bundle_id"].unique().tolist() == ["goal-bundle"]
    assert table["evaluation_model_artifacts_sha256"].unique().tolist() == [
        "a" * 64
    ]
    assert table["dataset_sha256"].unique().tolist() == ["c" * 64]
    assert table["evaluation_model_sha256"].unique().tolist() == ["a" * 64]
    assert table["constraint_spec_sha256"].unique().tolist() == [
        goal.GOAL_STAGE_SPEC_SHA256
    ]
    assert table["cooling_contract_sha256"].unique().tolist() == [
        goal.FIXED_COOLING_IDENTITY_SHA256
    ]
    assert table["operating_point_sha256"].unique().tolist() == [
        goal.FIXED_OPERATING_IDENTITY_SHA256
    ]
    assert all(
        f"physical_G:{name}" in table.columns
        and f"normalized_G:{name}" in table.columns
        for name in problem.constraint_names
    )
    with pytest.raises(RuntimeError, match="requires 320 rows"):
        preflight._terminal_physical_candidate_frame(
            types.SimpleNamespace(problem=problem),
            coordinates=coordinates[:-1],
            objectives=objectives[:-1],
            optimizer_constraints=normalized_g[:-1],
            physical_constraints=physical_g[:-1],
            frame=frame.iloc[:-1],
            decoder_valid=valid[:-1],
            source_identity=source,
        )
