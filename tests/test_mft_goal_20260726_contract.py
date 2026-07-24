from __future__ import annotations

import copy
import json
from pathlib import Path
import types

import numpy as np
import pandas as pd
import pytest

from module import input_parameter_260706 as current_input
from module import mft_goal_20260726_contract as goal
from regression_260707.verify import finalize
from tools import tier1_corrected_generation_adapter as adapter
from tools import tier1_corrected_generation_preflight as preflight
from tools import tier1_final1000_multiseed_consumer as consumer
from tools import mft_goal_20260726_launch as launch


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

    from regression_260707.training import checkpoint_train

    for target in goal.BODY_WINDING_TEMPERATURE_TARGETS + (
        *goal.BODY_CORE_TEMPERATURE_TARGETS,
    ):
        assert target not in checkpoint_train.TARGETS
        assert checkpoint_train.OPTIONAL_TARGETS[target] == {
            "transform": "t50",
            "metric_focus": "rmse",
        }
        outlier = pd.DataFrame(
            {
                "_strict_valid_full": [True],
                target: [5000.0],
                "physics_data_revision": ["goal-g0"],
            }
        )
        assert checkpoint_train.filter_valid_training_rows(
            outlier, target
        ).empty
    assert len(checkpoint_train.TARGETS) == 21
    assert set(checkpoint_train.TRAINABLE_TARGETS) == set(
        goal.GOAL_G0_MODEL_TARGETS
    )


def test_goal_24_model_inference_binding_is_accepted():
    binding = {
        "threads_per_model": 8,
        "target_count": len(adapter.GOAL_REQUIRED_MODEL_TARGETS),
        "model_count": len(adapter.GOAL_REQUIRED_MODEL_TARGETS),
        "families": ["extratrees"],
        "family_threads": {"extratrees": 1},
        "semaphore_free_families": ["extratrees"],
        "semaphore_free_sklearn_forest": True,
        "policy": preflight.FAMILY_SPECIFIC_INFERENCE_POLICY,
    }
    assert preflight._terminal_inference_binding_contract(binding) == (True, 8)


def test_goal_launcher_builds_isolated_32_seed_canary_rolling_payloads(
    tmp_path,
):
    assignments = launch.seed_assignments(
        mode="rolling32", seed_start=2607261000
    )
    assert len(assignments) == 32
    assert [item["fixed_primary_turns"] for item in assignments[:4]] == [
        5,
        6,
        7,
        8,
    ]
    assert [item["phase"] for item in assignments[:4]] == ["canary"] * 4
    assert {
        turns: sum(
            item["fixed_primary_turns"] == turns for item in assignments
        )
        for turns in goal.GOAL_PRIMARY_TURN_STRATA
    } == {5: 8, 6: 8, 7: 8, 8: 8}
    local_preflight = launch._seal(
        {
            "schema_version": launch.LOCAL_PREFLIGHT_SCHEMA,
            "hard_constraint_contract_sha256": "d" * 64,
            "dataset_sha256": "a" * 64,
            "evaluation_model_sha256": "b" * 64,
            "train_report_sha256": "e" * 64,
            "candidate_sha256": "f" * 64,
            "quality_status_sha256": "1" * 64,
            "code": {"revision": "c" * 40},
            "search_only_proposal": False,
        }
    )
    source = {
        "generation": "G0",
        "candidate": "candidate.json",
        "quality_status": "quality.json",
        "code_root": "repo",
        "expected_code_revision": "c" * 40,
    }
    bundle, tasks, scheduler = launch.build_bundle_values(
        local_preflight=local_preflight,
        assignments=assignments,
        output_root=tmp_path,
        source=source,
    )
    assert bundle["schema_version"] == launch.BUNDLE_SCHEMA
    assert bundle["task_count"] == 32
    assert bundle["legacy_current7_bundle_or_release_identity_reused"] is False
    assert bundle["relocation_contract"]["schema_version"] == (
        launch.RELOCATION_SCHEMA
    )
    assert bundle["relocation_contract"][
        "source_absolute_paths_are_not_worker_authority"
    ] is True
    assert scheduler["maximum_parallel_tasks"] == 32
    assert scheduler["canary_seed_count"] == 4
    assert scheduler["scheduler_submission_performed"] is False
    assert all(launch.validate_task_payload(task) is task for task in tasks)
    assert all(task["population"] == 320 for task in tasks)
    assert all(task["generations"] == 300 for task in tasks)
    assert all(
        task["temperature_contract_sha256"]
        == goal.GOAL_TEMPERATURE_CONTRACT_SHA256
        for task in tasks
    )


def test_goal_launcher_task_rejects_legacy_scalar_temperature():
    assignment = launch.seed_assignments(
        mode="single", seed_start=2607260001, fixed_primary_turns=8
    )
    local_preflight = launch._seal(
        {
            "schema_version": launch.LOCAL_PREFLIGHT_SCHEMA,
            "hard_constraint_contract_sha256": "d" * 64,
            "dataset_sha256": "a" * 64,
            "evaluation_model_sha256": "b" * 64,
            "train_report_sha256": "e" * 64,
            "candidate_sha256": "f" * 64,
            "quality_status_sha256": "1" * 64,
            "code": {"revision": "c" * 40},
            "search_only_proposal": True,
        }
    )
    _bundle, tasks, _scheduler = launch.build_bundle_values(
        local_preflight=local_preflight,
        assignments=assignment,
        output_root=Path("."),
        source={
            "generation": "G0",
            "candidate": "candidate.json",
            "quality_status": "quality.json",
            "code_root": "repo",
            "expected_code_revision": "c" * 40,
        },
    )
    forged = copy.deepcopy(tasks[0])
    forged.pop("payload_sha256")
    forged["stage_spec"]["T_limit_C"] = 100.0
    forged = launch._seal(forged)
    with pytest.raises(goal.GoalContractError, match="legacy fields"):
        launch.validate_task_payload(forged)


def test_goal_launcher_scales_to_authenticated_512_seed_rollout():
    assignments = launch.seed_assignments(
        mode="rolling",
        seed_start=2607262000,
        seed_count=512,
        wave_size=32,
    )
    assert len(assignments) == 512
    assert sum(item["phase"] == "canary" for item in assignments) == 4
    assert len({item["seed"] for item in assignments}) == 512
    assert {
        turns: sum(
            item["fixed_primary_turns"] == turns for item in assignments
        )
        for turns in goal.GOAL_PRIMARY_TURN_STRATA
    } == {5: 128, 6: 128, 7: 128, 8: 128}
    assert max(item["wave"] for item in assignments) == 16


def test_failed_g0_quality_is_sealed_as_search_only_without_lowering_thresholds():
    thresholds_path = (
        REPO
        / "regression_260707"
        / "training"
        / "model_quality_thresholds.json"
    )
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    metrics = {
        "r2": 0.80,
        "rmse": 5.5,
        "p90_ape_pct": 11.0,
        "interval_p90_width": 11.0,
    }
    quality = {
        "passed": False,
        "reasons": ["T_max_Tx:metric_below_minimum:r2"],
        "thresholds_sha256": goal.canonical_sha256(thresholds),
        "targets": {
                target: {
                    "passed": False,
                    "blocking": True,
                    "reasons": [
                        "metric_below_minimum:r2",
                        "metric_above_maximum:rmse",
                        "metric_above_maximum:p90_ape_pct",
                        "metric_above_maximum:interval_p90_width",
                    ],
                "metrics": metrics,
            }
            for target in goal.GOAL_TEMPERATURE_TARGETS
        },
    }
    contract = launch._quality_contract(quality=quality, code_root=REPO)
    assert contract["quality_passed"] is False
    assert contract["search_only_proposal"] is True
    assert contract["thresholds_lowered_or_bypassed"] is False
    assert contract["temperature_target_count"] == 11
    assert set(contract["temperature_status"]) == set(
        goal.GOAL_TEMPERATURE_TARGETS
    )
    assert contract["production_eligible"] is False
    assert contract["automatic_promotion_allowed"] is False


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
    surrogate_valid = np.ones(count, dtype=bool)
    surrogate_valid[7] = False
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
        surrogate_physical_valid=surrogate_valid,
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
    assert table["surrogate_physical_valid"].sum() == 319
    assert table["surrogate_physicality_passed"].sum() == 319
    assert table["physical_constraint_feasible"].all()
    assert table["physical_feasible"].sum() == 319
    assert table.loc[7, "physical_feasible"] == False
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
            surrogate_physical_valid=surrogate_valid[:-1],
            source_identity=source,
        )


def _write_synthetic_goal_seed_result(root: Path, *, seed: int) -> Path:
    root.mkdir()
    constraints = list(preflight.GOAL_CONSTRAINT_NAMES)
    task_sha = goal.canonical_sha256({"synthetic_task_seed": seed})
    rows = []
    for index in range(launch.POPULATION):
        if index == 0:
            volume, loss = (
                (100.0, 200.0) if seed == 101 else (110.0, 190.0)
            )
        elif seed == 101 and index == 1:
            # Physical G alone passes, but a negative objective must remain
            # quarantined by the canonical surrogate-physicality evidence.
            volume, loss = -100.0, -100.0
        else:
            volume, loss = 1000.0 + index, 1000.0 + index
        surrogate_valid = not (seed == 101 and index == 1)
        geometry_sha = goal.canonical_sha256(
            {"seed": seed, "terminal_population_index": index}
        )
        row = {
            "terminal_population_index": index,
            "decoder_valid": True,
            "surrogate_physical_valid": surrogate_valid,
            "surrogate_physicality_passed": surrogate_valid,
            "physical_constraint_feasible": True,
            "physical_feasible": surrogate_valid,
            "physical_geometry_sha256": geometry_sha,
            "canonical_physical_params_sha256": geometry_sha,
            "candidate_physics_sha": geometry_sha,
            "objective_volume_L": volume,
            "objective_total_loss_W": loss,
            "physical_G_json": "{}",
            "normalized_G_json": "{}",
            "coordinate_unit_json": "[]",
            "decoded_physical_params_json": "{}",
            "source_seed": seed,
            "source_task_id": f"task-{seed}",
            "source_bundle_id": task_sha,
            "source_island_id": f"n1-{5 + (seed % 2)}",
            "dataset_sha256": "a" * 64,
            "evaluation_model_sha256": "b" * 64,
            "constraint_spec_sha256": goal.GOAL_STAGE_SPEC_SHA256,
            "cooling_contract_sha256": (
                goal.FIXED_COOLING_IDENTITY_SHA256
            ),
            "operating_point_sha256": (
                goal.FIXED_OPERATING_IDENTITY_SHA256
            ),
            "evaluation_model_artifacts_sha256": "b" * 64,
            "evaluation_model_generation_sha256": "c" * 64,
            "evaluation_spec_sha256": goal.GOAL_STAGE_SPEC_SHA256,
            "evaluation_temperature_contract_sha256": (
                goal.GOAL_TEMPERATURE_CONTRACT_SHA256
            ),
            "evaluation_hard_constraint_contract_sha256": "d" * 64,
        }
        row.update({f"physical_G:{name}": -1.0 for name in constraints})
        row.update({f"normalized_G:{name}": -0.5 for name in constraints})
        rows.append(row)
    table = pd.DataFrame(rows)
    table_path = root / "terminal_physical_candidates.csv"
    table.to_csv(table_path, index=False)
    manifest = launch._seal(
        {
            "schema_version": preflight.GOAL_TERMINAL_TABLE_SCHEMA,
            "goal_contract_required": True,
            "row_count": launch.POPULATION,
            "terminal_population_index_min": 0,
            "terminal_population_index_max": launch.POPULATION - 1,
            "columns": list(table.columns),
            "required_identity_columns": list(table.columns),
            "csv": {
                "path": table_path.name,
                "sha256": adapter.sha256_file(table_path),
                "size_bytes": table_path.stat().st_size,
            },
            "source_identity": {"seed": seed},
            "stage_spec_sha256": goal.GOAL_STAGE_SPEC_SHA256,
            "temperature_contract_sha256": (
                goal.GOAL_TEMPERATURE_CONTRACT_SHA256
            ),
            "hard_constraint_contract_sha256": "d" * 64,
            "one_row_per_terminal_individual": True,
            "physical_deduplication_key": "physical_geometry_sha256",
            "global_pareto_provenance_ready": True,
        }
    )
    manifest_path = root / "terminal_physical_candidates.manifest.json"
    launch._atomic_json(manifest_path, manifest)
    inventory = {
        "terminal_physical_candidates": {
            "path": table_path.name,
            "sha256": adapter.sha256_file(table_path),
            "size_bytes": table_path.stat().st_size,
        },
        "terminal_physical_candidates_manifest": {
            "path": manifest_path.name,
            "sha256": adapter.sha256_file(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
        },
    }
    result = launch._seal(
        {
            "schema_version": launch.SEARCH_RESULT_SCHEMA,
            "campaign_id": "mft-goal-20260726",
            "goal_contract_schema": goal.GOAL_CONTRACT_SCHEMA,
            "task_payload_sha256": task_sha,
            "seed": seed,
            "fixed_primary_turns": 5 + (seed % 2),
            "population": launch.POPULATION,
            "generations": launch.GENERATIONS,
            "evaluated_generations": launch.GENERATIONS,
            "completed_generations": launch.GENERATIONS,
            "stage_spec": copy.deepcopy(goal.GOAL_STAGE_SPEC),
            "hard_spec": copy.deepcopy(goal.GOAL_STAGE_SPEC),
            "hard_spec_sha256": goal.GOAL_STAGE_SPEC_SHA256,
            "stage_spec_sha256": goal.GOAL_STAGE_SPEC_SHA256,
            "constraint_version": "mft-goal-20260726",
            "temperature_contract_sha256": (
                goal.GOAL_TEMPERATURE_CONTRACT_SHA256
            ),
            "hard_constraint_contract_sha256": "d" * 64,
            "dataset_sha256": "a" * 64,
            "evaluation_model_sha256": "b" * 64,
            "operating_point_sha256": (
                goal.FIXED_OPERATING_IDENTITY_SHA256
            ),
            "cooling_contract_sha256": (
                goal.FIXED_COOLING_IDENTITY_SHA256
            ),
            "constraint_names": constraints,
            "temperature_targets": list(goal.GOAL_TEMPERATURE_TARGETS),
            "terminal_population_count": launch.POPULATION,
            "physical_feasible_count": (
                launch.POPULATION - (1 if seed == 101 else 0)
            ),
            "feasible_pareto_count": 1,
            "artifact_inventory": inventory,
            "artifact_inventory_sha256": goal.canonical_sha256(inventory),
            "terminal_physical_candidates_manifest": manifest,
            "legacy_current7_stage_or_release_identity_reused": False,
            "search_only_proposal": False,
            "production_eligible": False,
            "fea_submission_performed": False,
            "automatic_promotion_allowed": False,
        }
    )
    result_path = root / "result.json"
    launch._atomic_json(result_path, result)
    return result_path


def test_global_pareto_recomputes_from_all_terminal_rows_and_physicality(
    tmp_path,
):
    results = [
        _write_synthetic_goal_seed_result(tmp_path / "seed-101", seed=101),
        _write_synthetic_goal_seed_result(tmp_path / "seed-102", seed=102),
    ]
    manifest_path = launch.aggregate_results(
        result_paths=results,
        output=tmp_path / "global",
        minimum_seeds=2,
    )
    manifest = launch._validate_seal(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        schema=launch.GLOBAL_PARETO_SCHEMA,
    )
    assert manifest["input_terminal_row_count"] == 640
    assert manifest["physical_feasible_count"] == 639
    assert manifest["global_pareto_count"] == 2
    assert manifest["seed_local_pareto_merge_used"] is False
    pareto = pd.read_csv(tmp_path / "global" / "global_pareto_front.csv")
    assert sorted(
        zip(
            pareto["objective_volume_L"],
            pareto["objective_total_loss_W"],
        )
    ) == [(100.0, 200.0), (110.0, 190.0)]
    merged = pd.read_csv(
        tmp_path / "global" / "global_terminal_candidates.csv"
    )
    quarantined = merged.loc[
        (merged["source_seed"] == 101)
        & (merged["terminal_population_index"] == 1)
    ].iloc[0]
    assert bool(quarantined["physical_feasible"]) is False
    assert quarantined["global_non_dominated_rank"] == -1
