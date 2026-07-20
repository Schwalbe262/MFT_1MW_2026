from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pandas as pd
import pytest

from module import input_parameter_260706 as current_input
from regression_260707.optimization.design_summary import (
    design_analytical_b_field_t,
)
from regression_260707.monitoring.readers import CANDIDATE_REPORT_FIELDS
from regression_260707.optimization.geometry_metrics import bounding_box_lit
from tools import tier1_corrected_generation_adapter as adapter
from tools import tier1_corrected_generation_preflight as preflight


REPO = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=1, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_generation(tmp_path: Path) -> dict[str, object]:
    runtime = tmp_path / "runtime"
    registry = runtime / "registry"
    run_id = "20260719T151445-b8810300"
    generation = registry / "generations" / run_id
    generation.mkdir(parents=True)
    dataset = runtime / "dataset" / "strict.parquet"
    dataset.parent.mkdir()
    dataset.write_bytes(b"synthetic-strict-dataset")
    dataset_sha = _sha(dataset)
    profile = runtime / "profile" / "standard.json"
    profile_data = {"profile": "synthetic", "version": 1}
    _write_json(profile, profile_data)
    profile_sha = adapter.canonical_sha256(profile_data)
    strict_rows = 7
    recovery = {
        "contract": adapter.RECOVERY_CONTRACT,
        "max_allowed_abs_delta_F": adapter.RECOVERY_MAX_ABS_DELTA_F,
        "row_count": strict_rows,
        "cap_enabled_row_count": strict_rows,
        "eligible_row_count": strict_rows,
        "recovered_row_count": strict_rows,
        "max_observed_abs_delta_F": 5.0e-11,
        "missing_columns": [],
        "status": adapter.RECOVERY_STATUS,
    }
    features = ["feature_a", "feature_b"]
    artifacts = {}
    for target in adapter.CORRECTED_GENERATION_TARGETS:
        target_dir = generation / target
        target_dir.mkdir()
        model = target_dir / "models.pkl"
        model.write_bytes(f"synthetic-model:{target}".encode())
        meta = target_dir / "meta.json"
        _write_json(meta, {
            "target": target,
            "training_run_id": run_id,
            "dataset_sha256": dataset_sha,
            "profile_sha256": profile_sha,
            "strict_full_rows": strict_rows,
            "features": features,
            "feature_schema": features,
            "q90": 1.5,
            "capacitance_recovery": recovery,
        })
        artifacts[f"{target}/models.pkl"] = _sha(model)
        artifacts[f"{target}/meta.json"] = _sha(meta)
    report = {
        "schema_version": 2,
        "training_run_id": run_id,
        "dataset_path": str(dataset.resolve()),
        "dataset_sha256": dataset_sha,
        "profile_path": str(profile.resolve()),
        "profile_sha256": profile_sha,
        "strict_full_rows": strict_rows,
        "features": features,
        "targets": list(adapter.CORRECTED_GENERATION_TARGETS),
        "report": {
            target: {"passed": False}
            for target in adapter.CORRECTED_GENERATION_TARGETS
        },
        "artifacts": artifacts,
        "capacitance_recovery": recovery,
    }
    report_path = generation / "train_report.json"
    _write_json(report_path, report)
    report_sha = _sha(report_path)
    relative = f"generations/{run_id}"
    candidate_path = runtime / "train" / "candidate.json"
    _write_json(candidate_path, {
        "schema_version": 2,
        "training_run_id": run_id,
        "generation": relative,
        "generation_path": str(generation.resolve()),
        "generation_report_sha256": report_sha,
        "dataset_sha256": dataset_sha,
        "strict_full_rows": strict_rows,
        "capacitance_recovery": recovery,
    })
    recovery_targets = {
        target: {"passed": True, "reasons": []}
        for target in adapter.CORRECTED_GENERATION_TARGETS
    }
    quality_path = runtime / "quality" / "quality_status.json"
    _write_json(quality_path, {
        "training_run_id": run_id,
        "generation": relative,
        "generation_report_sha256": report_sha,
        "dataset_sha256": dataset_sha,
        "profile_sha256": profile_sha,
        "strict_full_rows": strict_rows,
        "passed": False,
        "reasons": ["sealed synthetic production blocker"],
        "targets": {
            target: {"passed": False}
            for target in adapter.CORRECTED_GENERATION_TARGETS
        },
        "capacitance_recovery": {
            "schema_version": 1,
            "passed": True,
            "reasons": [],
            "contract": adapter.RECOVERY_CONTRACT,
            "recovered_row_count": strict_rows,
            "max_observed_abs_delta_F": 5.0e-11,
            "dataset_sha256": dataset_sha,
            "profile_sha256": profile_sha,
            "targets": recovery_targets,
        },
    })
    return {
        "generation": generation,
        "candidate": candidate_path,
        "quality": quality_path,
        "report": report,
        "report_sha": report_sha,
        "recovery": recovery,
    }


def _authenticate(paths: dict[str, object]):
    return adapter.authenticate_corrected_generation(
        generation=paths["generation"],
        candidate_path=paths["candidate"],
        quality_path=paths["quality"],
    )


def test_authenticates_failed_quality_only_with_complete_recovery(tmp_path):
    paths = _synthetic_generation(tmp_path)
    authenticated = _authenticate(paths)
    assert authenticated.quality["passed"] is False
    assert authenticated.evidence["generation_target_count"] == 21
    assert authenticated.evidence["artifact_count"] == 42
    assert authenticated.evidence["capacitance_recovery"] == {
        **paths["recovery"],
        "guard_passed": True,
        "guard_target_count": 21,
        "guard_passed_target_count": 21,
    }
    manifest = adapter.adapter_manifest(
        authenticated,
        code_identity={
            "path": str(tmp_path),
            "revision": "a" * 40,
            "clean": True,
        },
    )
    assert adapter.validate_adapter_manifest(manifest) is manifest
    assert manifest["portability"]["remote_relocation_supported"] is True
    assert manifest["scheduler_write_performed"] is False
    documentary = json.loads(json.dumps(manifest))
    documentary["code"]["path"] = r"Y:\source-only\current7"
    assert adapter.validate_adapter_manifest(
        documentary, source_paths_required=False
    ) is documentary
    with pytest.raises(RuntimeError, match="identity is invalid"):
        adapter.validate_adapter_manifest(documentary)


def test_authentication_rejects_incomplete_recovery_and_target_guard(tmp_path):
    paths = _synthetic_generation(tmp_path)
    candidate = json.loads(Path(paths["candidate"]).read_text(encoding="utf-8"))
    candidate["capacitance_recovery"]["eligible_row_count"] -= 1
    _write_json(Path(paths["candidate"]), candidate)
    with pytest.raises(RuntimeError, match="row inventory is incomplete"):
        _authenticate(paths)

    paths = _synthetic_generation(tmp_path / "second")
    quality = json.loads(Path(paths["quality"]).read_text(encoding="utf-8"))
    quality["capacitance_recovery"]["targets"].pop("C_tx_rx_F")
    _write_json(Path(paths["quality"]), quality)
    with pytest.raises(RuntimeError, match="recovery target inventory mismatch"):
        _authenticate(paths)


class _FakePredictor:
    def __init__(self, bundle, value=1.0, half_width=0.1):
        self.bundle = bundle
        self.features = bundle["features"]
        self.value = value
        self.half_width = half_width

    def predict_mu_sigma(self, frame, conformal=True):
        assert conformal is True
        return (
            np.full(len(frame), self.value, dtype=float),
            np.full(len(frame), self.half_width, dtype=float),
        )

    def disagreement(self, frame):
        return np.zeros(len(frame), dtype=float)


def _fake_cache_dependencies(authenticated, *, bad_q90=False):
    calls = {"generation": 0, "records": []}

    class TrainModels:
        @staticmethod
        def load_generation(registry, generation, require_accepted):
            calls["generation"] += 1
            assert require_accepted is False
            return {
                "generation": str(authenticated.generation),
                "generation_relative": authenticated.evidence[
                    "generation_relative"
                ],
                "generation_report_sha256": authenticated.evidence[
                    "train_report"
                ]["sha256"],
                "report": authenticated.report,
            }

    class PredictorClass:
        @classmethod
        def _load_record(cls, target, record):
            calls["records"].append(target)
            report = record["report"]
            bundle = {
                "target": target,
                "training_run_id": report["training_run_id"],
                "dataset_sha256": report["dataset_sha256"],
                "profile_sha256": report["profile_sha256"],
                "strict_full_rows": report["strict_full_rows"],
                "features": report["features"],
                "feature_schema": report["features"],
                "q90": float("nan") if bad_q90 else 1.5,
                "capacitance_recovery": report["capacitance_recovery"],
            }
            return _FakePredictor(bundle)

    return TrainModels, PredictorClass, calls


def test_model_cache_hashes_and_loads_once_then_returns_same_mapping(tmp_path):
    authenticated = _authenticate(_synthetic_generation(tmp_path))
    train_models, predictor, calls = _fake_cache_dependencies(authenticated)
    cache = adapter.CorrectedGenerationModelCache(
        authenticated,
        train_models_module=train_models,
        predictor_class=predictor,
    )
    first = cache.load()
    second = cache.load()
    assert first is second
    assert tuple(first) == adapter.CURRENT_REQUIRED_MODEL_TARGETS
    assert calls == {
        "generation": 1,
        "records": list(adapter.CURRENT_REQUIRED_MODEL_TARGETS),
    }
    assert cache.loaded_once


def test_model_cache_latches_failure_to_prevent_second_full_pass(tmp_path):
    authenticated = _authenticate(_synthetic_generation(tmp_path))
    train_models, predictor, calls = _fake_cache_dependencies(
        authenticated, bad_q90=True
    )
    cache = adapter.CorrectedGenerationModelCache(
        authenticated,
        train_models_module=train_models,
        predictor_class=predictor,
    )
    with pytest.raises(RuntimeError, match="conformal calibration"):
        cache.load()
    with pytest.raises(RuntimeError, match="refusing a second full artifact pass"):
        cache.load()
    assert calls["generation"] == 1
    assert cache.load_calls == 1


def test_resonance_is_tx_rx_only_and_uses_half_magnetizing_inductance():
    measurements = {
        "Llt_phys": 27.5,
        "k": 0.9,
        "C_tx_tx_F": 1.2e-9,
        "C_rx_rx_F": 2.4e-11,
        "C_tx_rx_F": 1e99,
    }
    params = {"N1_main": 6, "N1_side": 0, "N2_main": 18, "N2_side": 42}
    observed = preflight.derive_half_magnetizing_self_resonance(
        measurements, params, magnetizing_inductance_factor=0.5
    )
    leakage_h = 27.5e-6
    tx_magnetizing_h = leakage_h / (1 - 0.9**2) * 0.9**2
    expected_tx = 1 / (2 * math.pi * math.sqrt(0.5 * tx_magnetizing_h * 1.2e-9))
    expected_rx = 1 / (
        2
        * math.pi
        * math.sqrt(0.5 * tx_magnetizing_h * 100 * 2.4e-11)
    )
    assert observed["f_res_tx_half_magnetizing_Hz"] == pytest.approx(expected_tx)
    assert observed["f_res_rx_half_magnetizing_Hz"] == pytest.approx(expected_rx)
    assert observed["f_res_min_tx_rx_only_Hz"] == pytest.approx(
        min(expected_tx, expected_rx)
    )
    assert observed["interwinding_resonance_included"] is False


def _decoded_row(**updates):
    row = {
        "cc_w2c_space_x": 40.0,
        "cc_w2c_space_y": 40.0,
        "w2c_w1c_space_x": 40.0,
        "w2c_w1c_space_y": 40.0,
        "w1c_w2s_gap_x_actual": 40.0,
        "w1s_cs_space_x": 40.0,
        "cs_w1s_space_y": 40.0,
        "h_gap2": 40.0,
        "w2s_w1s_space_x": 40.0,
        "w1s_w2s_space_y": 40.0,
        "N1_main": 6,
        "N1_side": 0,
        "N2_main": 18,
        "N2_side": 42,
        "n_core_group": 4,
        "cw1": 5.0,
        "core_plate_t": 17.0,
        "wcp_t": 23.0,
        "wcp_len_pct": 55.0,
        "core_plate_pad_t": 2.0,
        "wcp_pad_t": 2.0,
        "bbox_x": 1200.0,
        "bbox_y": 1200.0,
        "bbox_z": 750.0,
    }
    row.update(updates)
    return row


def test_full_realized_insulation_and_side_temperature_are_fail_closed():
    frame = pd.DataFrame([
        _decoded_row(w1c_w2s_gap_x_actual=39.0),
        _decoded_row(N1_side=1, w2s_w1s_space_x=38.0),
    ])
    assert preflight.minimum_physical_insulation_violation(frame, 40.0).tolist() == [
        1.0,
        2.0,
    ]
    missing = frame.drop(columns=["h_gap2"])
    assert np.all(
        preflight.minimum_physical_insulation_violation(missing, 40.0)
        == preflight.BIG
    )
    side = pd.DataFrame([{"N2_side": 1}, {"N2_side": 0}, {"N2_side": np.nan}])
    assert preflight.apply_current7_side_temperature_condition(
        [3.0, 4.0, 5.0], side
    ).tolist() == [3.0, -preflight.BIG, preflight.BIG]


_SOBOL_DIMS = tuple(current_input._SOBOL_DIMS)


class _FakeBaseProblem:
    def __init__(self, models, spec=None, density_gate=None, fixed_overrides=None):
        self.models = models
        self.spec = {
            "core_lamination_factor": 0.85,
            "B_area_basis": "gross_geometry_times_lamination_factor",
            **preflight.CURRENT_STAGE_SPEC,
            **(spec or {}),
        }
        self.density_gate = density_gate
        self.constraint_names = preflight.BASE_CONSTRAINT_NAMES
        self.n_ieq_constr = len(self.constraint_names)
        self.n_var = len(_SOBOL_DIMS)
        self.xl = np.zeros(self.n_var)
        self.xu = np.ones(self.n_var)
        self.fixed_overrides = dict(fixed_overrides or {})
        self.fixed_overrides.update(preflight.EXPECTED_SIMPLE_BASE_FIXED_STACK_MM)
        for name in ("core_plate_t", "wcp_t"):
            index = [item[0] for item in _SOBOL_DIMS].index(name)
            self.xl[index] = 0.5
            self.xu[index] = 0.5

    def decode_batch(self, values):
        rows = []
        for coordinate in values:
            row = _decoded_row(
                core_plate_t=10.0 + 20.0 * coordinate[0],
                wcp_t=10.0 + 20.0 * coordinate[1],
                wcp_len_pct=20.0 + 60.0 * coordinate[2],
                N2_side=0,
            )
            row.update(self.fixed_overrides)
            rows.append(row)
        return pd.DataFrame(rows), np.zeros(len(values)), np.ones(len(values), bool)

    def _predict(self, target, frame):
        return self.models[target].predict_mu_sigma(frame, conformal=True)

    def _evaluate(self, values, out, *args, **kwargs):
        frame, shrink, valid = self.decode_batch(values)
        out["F"] = np.tile([1.0, 2.0], (len(values), 1))
        out["G"] = np.full((len(values), self.n_ieq_constr), preflight.BIG)
        out["G"][:, :len(preflight.BASE_CONSTRAINT_NAMES)] = -1.0
        self._predict("Llt_phys", frame)
        out["frame"] = frame


def _problem_models():
    values = {
        "Llt_phys": 27.5,
        "k": 0.9,
        "C_tx_tx_F": 1.2e-9,
        "C_rx_rx_F": 2.4e-11,
    }
    return {
        target: _FakePredictor(
            {"features": ["feature_a"], "target": target},
            value=value,
            half_width=0.1,
        )
        for target, value in values.items()
    }


def _actual_problem(fixed_primary_turns):
    modules = preflight.load_current7_modules(REPO)
    cls = preflight.create_current7_problem_class(
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
    values = {
        target: 1.0 for target in adapter.CURRENT_REQUIRED_MODEL_TARGETS
    }
    values.update({
        "Llt_phys": 27.5,
        "k": 0.9,
        "C_tx_tx_F": 1.2e-9,
        "C_rx_rx_F": 2.4e-11,
        "C_tx_rx_F": 5.0e-10,
        **{target: 50.0 for target in adapter.CURRENT_TEMPERATURE_TARGETS},
    })
    models = {
        target: _FakePredictor(
            {"features": ["l1"], "target": target},
            value=value,
            half_width=0.1,
        )
        for target, value in values.items()
    }
    return cls(
        models,
        density_gate=lambda frame: np.full(len(frame), -1.0),
        fixed_primary_turns=fixed_primary_turns,
    ), models, modules


def _fake_problem_class():
    return preflight.create_current7_problem_class(
        base_problem_class=_FakeBaseProblem,
        base_constraint_names=preflight.BASE_CONSTRAINT_NAMES,
        base_fixed_stack_mm=preflight.EXPECTED_SIMPLE_BASE_FIXED_STACK_MM,
        sobol_dims=_SOBOL_DIMS,
        bounding_box_lit=bounding_box_lit,
        input_parameter_module=current_input,
        design_analytical_b_field_t=design_analytical_b_field_t,
    )


@pytest.mark.parametrize("fixed_primary_turns", [5, 6])
def test_problem_wrapper_repairs_fixed_turns_budget_and_all_hard_constraints(
    fixed_primary_turns,
):
    problem = _fake_problem_class()(
        _problem_models(),
        density_gate=object(),
        fixed_primary_turns=fixed_primary_turns,
    )
    assert problem.spec["q_sigma"] == 1.0
    assert problem.spec["T_limit_C"] == 110.0
    assert tuple(problem.constraint_names) == preflight.CURRENT7_CONSTRAINT_NAMES
    assert problem.n_ieq_constr == 19
    for name in preflight.VARIABLE_COOLING_DIMENSIONS:
        assert name not in problem.fixed_overrides
        index = [item[0] for item in _SOBOL_DIMS].index(name)
        assert problem.xl[index] == 0.0
        assert problem.xu[index] == 1.0
    assert "cw1" not in problem.fixed_overrides
    coordinate = np.full((1, len(_SOBOL_DIMS)), 0.5)
    coordinate[0, 2] = 0.0
    coordinate[0, 8] = 0.65
    coordinate[0, 9] = 0.35
    repaired = problem.repair_unit_coordinates(coordinate)
    assert np.array_equal(problem.repair_unit_coordinates(repaired), repaired)
    out = {}
    problem._evaluate(repaired, out)
    assert out["G"].shape == (1, 19)
    side_index = problem.constraint_names.index(
        "temperature_robust_limit:Tprobe_Rx_side_leeward_max"
    )
    assert out["G"][0, side_index] == -preflight.BIG
    hard = dict(zip(problem.constraint_names, out["G"][0]))
    assert hard["minimum_physical_insulation"] <= 1e-9
    assert hard["core_group_manufacturability_limit"] <= 1e-9
    assert hard["exterior_width_limit"] <= 1e-9
    assert hard["exterior_length_limit"] <= 1e-9
    assert hard["exterior_height_limit"] <= 1e-9
    expected_screen = preflight.derive_half_magnetizing_self_resonance(
        {
            "Llt_phys": 27.5,
            "k": 0.9,
            "C_tx_tx_F": 1.2e-9,
            "C_rx_rx_F": 2.4e-11,
        },
        out["frame"].iloc[0],
    )["f_res_min_tx_rx_only_Hz"]
    assert hard["half_magnetizing_resonance_minimum"] == pytest.approx(
        15_000.0 - expected_screen
    )
    assert out["frame"].iloc[0]["cw1"] == 5.0
    assert out["frame"].iloc[0]["core_plate_t"] == 17.0
    assert out["frame"].iloc[0]["wcp_t"] == 23.0
    assert (
        int(out["frame"].iloc[0]["N1_main"])
        + int(out["frame"].iloc[0]["N1_side"])
        == fixed_primary_turns
    )
    budget = preflight.winding_budget_identity(
        out["frame"].iloc[0], expected_cw1_mm=5.0
    )
    assert budget["passed"] is True
    assert budget["post_decode_field_override_performed"] is False


def test_actual_current_module_contract_can_build_additive_class():
    modules = preflight.load_current7_modules(REPO)
    cls = preflight.create_current7_problem_class(
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
    assert cls.__name__ == "Current7Tier1Problem"
    assert cls.launch_eligible is True
    assert cls.offspring_physics_repair is True


@pytest.mark.parametrize("fixed_primary_turns", [5, 6])
def test_repaired_runner_uses_same_operator_for_warm_offspring_and_terminal(
    tmp_path,
    fixed_primary_turns,
):
    problem, models, modules = _actual_problem(fixed_primary_turns)
    runner = preflight.Current7Tier1Runner(
        authenticated=None,
        code_identity={},
        modules=modules,
        adapter_evidence={},
        model_cache=None,
        models=models,
        inference_binding={},
        density_gate=object(),
        problem=problem,
    )
    raw = np.full((3, len(_SOBOL_DIMS)), 0.5)
    raw[:, 2] = [0.0, 0.3, 0.6]
    warm_path = tmp_path / f"warm_n1_{fixed_primary_turns}.npy"
    np.save(warm_path, raw, allow_pickle=False)
    pre_optimization = []
    result = runner.run_one(
        seed=11,
        population=64,
        max_generations=2,
        warm_start_path=warm_path,
        warm_start_sha256=adapter.sha256_file(warm_path),
        pre_optimization_callback=pre_optimization.append,
    )
    assert len(pre_optimization) == 1
    assert pre_optimization[0]["optimizer_execution_started"] is False
    assert pre_optimization[0]["offspring_repair_operator"][
        "call_count"
    ] == 0
    audit = result.tier1_repair_audit
    assert audit["fixed_primary_turns"] == fixed_primary_turns
    assert audit["stages"] == {
        "initial_population": True,
        "warm_start": True,
        "every_offspring": True,
        "terminal_physical_replay": True,
    }
    assert audit["pymoo_operator"]["call_count"] >= 1
    assert audit["authenticated_warm_start"]["source_authentication"][
        "single_read_hash_and_load"
    ] is True
    assert audit["terminal_physical_replay"][
        "optimizer_physical_G_match"
    ] is True
    terminal = np.asarray(result.pop.get("X"), dtype=float)
    index = problem.fixed_primary_turn_coordinate_index
    assert np.isclose(
        terminal[:, index],
        problem.fixed_primary_turn_unit_coordinate,
        rtol=0.0,
        atol=1e-15,
    ).all()
    output = tmp_path / f"persisted_n1_{fixed_primary_turns}"
    output.mkdir()
    persisted = preflight.persist_search_outputs(runner, result, output)
    assert persisted["terminal_population_primary_turn_values"] == [
        fixed_primary_turns
    ]
    assert persisted["artifact_inventory_sha256"] == adapter.canonical_sha256(
        persisted["artifact_inventory"]
    )
    candidates = json.loads(
        (output / "least_violation_candidates.json").read_text(encoding="utf-8")
    )["candidates"]
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["total_loss_W"] == candidate["predicted_total_loss_W"]
    assert candidate["pred_Llt_phys"] == candidate["pred_Llt_phys_uH"]
    assert candidate["B_design_analytic_T"] == candidate["analytical_B_T"]
    assert set(CANDIDATE_REPORT_FIELDS).issubset(candidate)
    assert candidate["rated_power_W"] == 1_000_000.0
    assert candidate["pred_efficiency_pct"] == pytest.approx(
        1_000_000.0 / (1_000_000.0 + candidate["total_loss_W"]) * 100.0
    )
    assert set(candidate["physical_constraint_G"]) == set(
        preflight.CURRENT7_CONSTRAINT_NAMES
    )
    assert candidate["decoded_params"]["cw1"] == 5.0


def test_candidate_temperature_max_excludes_absent_rx_side_target():
    problem, models, modules = _actual_problem(5)
    models[preflight.SIDE_TEMPERATURE_TARGET].value = 500.0
    runner = preflight.Current7Tier1Runner(
        authenticated=None,
        code_identity={},
        modules=modules,
        adapter_evidence={},
        model_cache=None,
        models=models,
        inference_binding={},
        density_gate=object(),
        problem=problem,
    )
    raw = np.full((1, problem.n_var), 0.5)
    raw[0, 2] = 0.0
    repaired, _ = runner.repair_coordinates(raw, stage="candidate-test")
    evaluation = runner.evaluate_coordinates(repaired)
    assert int(evaluation["frame"].iloc[0]["N2_side"]) == 0
    predictions = preflight._terminal_model_predictions(
        models, evaluation["frame"]
    )
    record = preflight._candidate_records(
        runner,
        coordinates=repaired,
        objectives=evaluation["F"],
        physical_constraints=evaluation["G"],
        frame=evaluation["frame"],
        predictions=predictions,
        indices=[0],
    )[0]
    assert record[f"pred_{preflight.SIDE_TEMPERATURE_TARGET}"] == 500.0
    assert preflight.SIDE_TEMPERATURE_TARGET not in record[
        "active_temperature_targets_for_maximum"
    ]
    assert record["pred_max_temperature_C"] == 50.0
    assert record["pred_max_robust_temperature_C"] == 50.1


def test_authenticate_warm_handoff_accepts_sealed_n1_6_coordinate_contract(
    tmp_path,
):
    problem, _models, _modules = _actual_problem(6)
    values = np.full((2, problem.n_var), 0.5)
    warm_path = tmp_path / "next_warm_start.npy"
    np.save(warm_path, values, allow_pickle=False)
    contract_path = tmp_path / "warm_handoff_contract.json"
    contract = {
        "schema_version": "mft-tier1-fixed-n1-6-anchor-islands-warm-v1",
        "fixed_primary_turns": 6,
        "warm_start": {
            "filename": warm_path.name,
            "sha256": adapter.sha256_file(warm_path),
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "coordinate_contract": (
                "authenticated_n1_6_decoded_to_unit_then_current_repair_v1"
            ),
        },
        "physical_hard_spec_mutation": False,
        "objective_mutation": False,
        "automatic_promotion_allowed": False,
    }
    _write_json(contract_path, contract)
    loaded, evidence = preflight.authenticate_warm_handoff(
        warm_path,
        contract_path,
        fixed_primary_turns=6,
        n_var=problem.n_var,
        expected_contract_file_sha256=adapter.sha256_file(contract_path),
    )
    assert np.array_equal(loaded, values)
    assert evidence["coordinates_only"] is True
    assert evidence["coordinate_contract"] == contract["warm_start"][
        "coordinate_contract"
    ]


def test_smoke_receipt_seals_both_repaired_strata_and_is_launch_eligible(tmp_path):
    authenticated = _authenticate(_synthetic_generation(tmp_path))
    manifest = adapter.adapter_manifest(
        authenticated,
        code_identity={
            "path": str(tmp_path),
            "revision": "b" * 40,
            "clean": True,
        },
    )
    all_models = {
        target: _FakePredictor(
            {"features": ["feature_a"], "target": target},
            value=1.0,
        )
        for target in adapter.CURRENT_REQUIRED_MODEL_TARGETS
    }
    fake_cache = types.SimpleNamespace(
        loaded_once=True,
        load_calls=1,
        full_generation_authentication_passes=1,
    )
    modules = preflight.Current7Modules(
        run_nsga2=None,
        nsga2_problem=None,
        predictor=None,
        train_models=None,
        geometry_metrics=None,
        design_summary=None,
        input_parameter=None,
        evidence={"synthetic": True},
    )
    problem_class = _fake_problem_class()
    first_runner = preflight.Current7Tier1Runner(
        authenticated=authenticated,
        code_identity={"path": str(tmp_path), "revision": "b" * 40, "clean": True},
        modules=modules,
        adapter_evidence=manifest,
        model_cache=fake_cache,
        models=all_models,
        inference_binding={
            "target_count": 20,
            "threads_per_model": 1,
            "model_count": 80,
            "families": ["extratrees"],
        },
        density_gate=object(),
        problem=problem_class(
            all_models,
            density_gate=object(),
            fixed_primary_turns=5,
        ),
    )
    runners = {
        "5": first_runner,
        "6": preflight.runner_for_fixed_primary_turns(first_runner, 6),
    }
    raw = np.full((1, len(_SOBOL_DIMS)), 0.5)
    raw[0, 2] = 0.0
    evaluations = {}
    repair_smoke_by_stratum = {}
    model_smoke_by_stratum = {}
    for turns in preflight.SUPPORTED_FIXED_PRIMARY_TURNS:
        key = str(turns)
        runner = runners[key]
        problem = runner.problem
        initial, initial_evidence = runner.repair_coordinates(
            raw, stage="initial_population"
        )
        warm, warm_evidence = runner.repair_coordinates(
            raw, stage="authenticated_warm_start"
        )
        assert np.array_equal(initial, warm)
        operator = preflight.create_pymoo_physics_repair(problem)
        offspring = operator._do(problem, np.mod(raw + 0.1, 1.0))
        offspring, offspring_evidence = runner.repair_coordinates(
            offspring, stage="every_pymoo_offspring"
        )
        evaluation = runner.evaluate_coordinates(initial)
        _terminal, terminal_evidence = runner.terminal_physical_replay(
            initial,
            expected_f=evaluation["F"],
            expected_g=evaluation["G"],
        )
        repair_smoke = {
            "schema_version": "mft-tier1-current7-repair-smoke-v1",
            "fixed_primary_turns": turns,
            "stages": {
                "initial_population": True,
                "warm_start": True,
                "every_offspring": True,
                "terminal_physical_replay": True,
            },
            "initial_population": initial_evidence,
            "authenticated_warm_start": warm_evidence,
            "every_offspring": {
                "coordinate_repair": offspring_evidence,
                "pymoo_operator": operator.evidence(),
                "output_sha256": adapter.canonical_sha256(offspring.tolist()),
            },
            "terminal_physical_replay": terminal_evidence,
            "same_problem_repair_used_for_all_stages": True,
        }
        repair_smoke["sha256"] = adapter.canonical_sha256(repair_smoke)
        evaluations[key] = evaluation
        repair_smoke_by_stratum[key] = repair_smoke
        model_smoke_by_stratum[key] = preflight._smoke_every_model(
            all_models, evaluation["frame"].iloc[[0]]
        )
    receipt = preflight.build_smoke_receipt(
        runners=runners,
        coordinate_evidence={
            "path": str(tmp_path / "smoke.npy"),
            "sha256": "c" * 64,
            "shape": [1, len(_SOBOL_DIMS)],
            "coordinate_unit_sha256": "d" * 64,
        },
        evaluations=evaluations,
        model_smoke_by_stratum=model_smoke_by_stratum,
        repair_smoke_by_stratum=repair_smoke_by_stratum,
    )
    assert preflight.validate_smoke_receipt(receipt) is receipt
    assert receipt["supported_fixed_primary_turns"] == [5, 6]
    assert set(receipt["strata"]) == {"5", "6"}
    assert all(
        receipt["optimizer_repair"]["strata"][key]["stages"][
            "every_offspring"
        ]
        is True
        for key in ("5", "6")
    )
    assert receipt["runner"]["launch_eligible"] is True
    assert receipt["problem_contract"]["constraint_count"] == 19
    forged = json.loads(json.dumps(receipt))
    forged["strata"].pop("6")
    forged.pop("payload_sha256")
    forged["payload_sha256"] = adapter.canonical_sha256(forged)
    with pytest.raises(RuntimeError, match="contract mismatch"):
        preflight.validate_smoke_receipt(forged)
    receipt_path = tmp_path / "receipt.json"
    _write_json(receipt_path, receipt)
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "tier1_corrected_generation_preflight.py"),
            "validate-receipt",
            str(receipt_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["launch_eligible"] is True


def test_smoke_receipt_rejects_repair_or_launch_bit_flip(tmp_path):
    # A compact mutation test uses the fully formed receipt above indirectly:
    # any caller must recompute payload_sha, and contract validation still
    # rejects a forged launch bit.
    paths = _synthetic_generation(tmp_path)
    authenticated = _authenticate(paths)
    manifest = adapter.adapter_manifest(
        authenticated,
        code_identity={"path": str(tmp_path), "revision": "e" * 40, "clean": True},
    )
    fake = {
        "schema_version": preflight.RECEIPT_SCHEMA,
        "status": (
            "authenticated_model_and_repair_smoke_passed_launch_eligible"
        ),
        "runner": {
            "schema_version": preflight.RUNNER_SCHEMA,
            "problem_schema": preflight.PROBLEM_SCHEMA,
            "full_nsga_executed": False,
            "launch_eligible": False,
        },
        "adapter_manifest": manifest,
    }
    fake["payload_sha256"] = adapter.canonical_sha256(fake)
    with pytest.raises(RuntimeError, match="contract mismatch"):
        preflight.validate_smoke_receipt(fake)


def test_search_seed_cli_runs_synthetic_optimizer_and_seals_artifacts(
    tmp_path, monkeypatch
):
    problem, models, modules = _actual_problem(5)
    runner = preflight.Current7Tier1Runner(
        authenticated=types.SimpleNamespace(),
        code_identity={},
        modules=modules,
        adapter_evidence={},
        model_cache=types.SimpleNamespace(
            loaded_once=True,
            load_calls=1,
            full_generation_authentication_passes=1,
        ),
        models=models,
        inference_binding={
            "target_count": len(adapter.CURRENT_REQUIRED_MODEL_TARGETS),
            "threads_per_model": 8,
        },
        density_gate=object(),
        problem=problem,
    )
    original_topology = preflight.deep_topology_contract

    def synthetic_topology(fixed_primary_turns):
        value = json.loads(json.dumps(original_topology(fixed_primary_turns)))
        value.pop("sha256")
        value["minimum_evolution_generations"] = 2
        value["survival"]["decay_to_zero_generation"] = 1
        value["sha256"] = adapter.canonical_sha256(value)
        return value

    monkeypatch.setattr(preflight, "deep_topology_contract", synthetic_topology)
    monkeypatch.setattr(preflight, "PRODUCTION_POPULATION", 64)
    monkeypatch.setattr(preflight, "PRODUCTION_FIXED_GENERATIONS", 2)
    monkeypatch.setattr(preflight, "observed_peak_rss_bytes", lambda: 123_456)

    bundle = tmp_path / "bundle"
    code_root = bundle / "artifacts" / "code"
    code_root.mkdir(parents=True)
    output = bundle / "runs" / "seed-5"
    output.mkdir(parents=True)
    warm_dir = bundle / "artifacts" / "warm" / "n1-5"
    warm_dir.mkdir(parents=True)
    warm_path = warm_dir / "coordinates.npy"
    raw = np.full((4, problem.n_var), 0.5)
    raw[:, 2] = [0.0, 0.25, 0.5, 0.75]
    np.save(warm_path, raw, allow_pickle=False)
    warm_sha = adapter.sha256_file(warm_path)
    warm_contract_path = warm_dir / "contract.json"
    warm_contract = {
        "schema_version": "synthetic-warm-contract-v1",
        "fixed_primary_turns": 5,
        "warm_start": {"sha256": warm_sha, "shape": list(raw.shape)},
        "warm_rows_are_coordinate_donors_only": True,
        "physical_hard_spec_mutation": False,
        "objective_mutation": False,
        "automatic_promotion_allowed": False,
    }
    warm_contract["sha256"] = adapter.canonical_sha256(warm_contract)
    _write_json(warm_contract_path, warm_contract)
    topology = synthetic_topology(5)
    island_id = "synthetic-n1-5"
    island_profile = {
        "schema_version": "mft-tier1-current7-deep-crossover-island-v1",
        "island_id": island_id,
        "fixed_primary_turns": 5,
        "population": 64,
        "fixed_generations": 2,
        "inference_threads": 8,
        "seed_start": 1,
        "seed_window_end_exclusive": 10,
        "optimizer_termination_strategy": (
            preflight.FIXED_GENERATION_TERMINATION_STRATEGY
        ),
        "optimizer_resonance_scale_Hz": 150.0,
        "optimizer_llt_scale_uH": 0.3,
        "optimizer_all_current7_thermal_scale_C": 2.0,
        "optimizer_resonance_allowance_Hz": None,
        "optimizer_llt_allowance_uH": None,
        "temperature_targets": list(adapter.CURRENT_TEMPERATURE_TARGETS),
        "topology_evolution_contract": topology,
        "offspring_physics_repair_required": True,
        "terminal_physical_replay_required": True,
        "physical_hard_spec_mutation": False,
        "objective_mutation": False,
        "automatic_promotion_allowed": False,
    }
    island_profile["sha256"] = adapter.canonical_sha256(island_profile)
    warm_record = {
        "artifact": {
            "path": warm_path.relative_to(bundle).as_posix(),
            "sha256": warm_sha,
            "size": warm_path.stat().st_size,
        },
        "contract": {
            "path": warm_contract_path.relative_to(bundle).as_posix(),
            "sha256": adapter.sha256_file(warm_contract_path),
            "size": warm_contract_path.stat().st_size,
        },
    }
    code_revision = "a" * 40
    adapter_evidence = {
        "artifact_count": 42,
        "code": {"revision": code_revision},
        "train_report": {"sha256": "b" * 64},
        "dataset": {"sha256": "c" * 64},
        "profile": {"canonical_sha256": "d" * 64},
    }
    receipt = {
        "adapter_manifest": adapter_evidence,
        "adapter_manifest_sha256": adapter.canonical_sha256(adapter_evidence),
        "strata": {
            "5": {
                "optimizer_repair": {
                    "contract_sha256": problem.optimizer_repair_contract[
                        "contract_sha256"
                    ]
                }
            }
        },
    }
    bundle_id = "current7-synthetic"
    manifest = {
        "bundle_id": bundle_id,
        "bundle_code_revision": code_revision,
        "generation_artifact_inventory_sha256": "e" * 64,
        "search_execution": {
            "remote_preflight_schema_version": (
                preflight.REMOTE_PREFLIGHT_SCHEMA
            ),
            "result_schema_version": preflight.SEARCH_RESULT_SCHEMA,
            "optimizer_processes_per_task": 1,
            "model_mapping_instances_per_process": 1,
            "remote_preflight_filename": "remote_preflight.json",
            "result_filename": "result.json",
        },
        "islands": {
            island_id: {
                "current7_profile": island_profile,
                "current7_profile_sha256": island_profile["sha256"],
                "warm": warm_record,
            }
        },
        "fast_ramp": {"maximum_peak_rss_bytes": 10_000_000_000},
    }
    authenticated = types.SimpleNamespace()
    relocation = {"schema_version": "synthetic-relocation-v1"}
    monkeypatch.setattr(
        preflight,
        "authenticate_relocated_bundle_generation",
        lambda **kwargs: (authenticated, receipt, relocation, manifest),
    )
    monkeypatch.setattr(
        preflight,
        "build_relocated_authenticated_runner",
        lambda **kwargs: runner,
    )
    registry = bundle / "registry"
    generation = registry / "generations" / "synthetic"
    generation.mkdir(parents=True)
    dataset = bundle / "dataset.parquet"
    dataset.write_bytes(b"synthetic")
    profile_path = bundle / "profile.json"
    _write_json(profile_path, {"synthetic": True})
    relocation_path = bundle / "relocation.json"
    _write_json(relocation_path, relocation)
    receipt_path = bundle / "receipt.json"
    _write_json(receipt_path, receipt)
    remote_path = output / "remote_preflight.json"
    repair_sha = problem.optimizer_repair_contract["contract_sha256"]
    assert preflight.main([
        "search-seed",
        "--bundle-root", str(bundle),
        "--relocation", str(relocation_path),
        "--adapter-receipt", str(receipt_path),
        "--registry", str(registry),
        "--generation", str(generation),
        "--dataset", str(dataset),
        "--profile", str(profile_path),
        "--warm-start", str(warm_path),
        "--warm-contract", str(warm_contract_path),
        "--output", str(output),
        "--remote-preflight", str(remote_path),
        "--bundle-id", bundle_id,
        "--island-id", island_id,
        "--island-profile-sha256", island_profile["sha256"],
        "--seed", "5",
        "--population", "64",
        "--max-generations", "2",
        "--inference-threads", "8",
        "--fixed-primary-turns", "5",
        "--optimizer-termination-strategy",
        preflight.FIXED_GENERATION_TERMINATION_STRATEGY,
        "--optimizer-resonance-scale-hz", "150",
        "--optimizer-llt-scale-uh", "0.3",
        "--optimizer-all-thermal-scale-c", "2",
        "--optimizer-repair-contract-sha256", repair_sha,
    ]) == 0
    remote = json.loads(remote_path.read_text(encoding="utf-8"))
    assert remote["offspring_repair_operator_installed"] is True
    assert remote["every_offspring_decode_repair_attested"] is False
    assert remote["offspring_repair_execution_status"] == (
        "deferred_until_optimizer_execution"
    )
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    unsigned = dict(result)
    payload_sha = unsigned.pop("payload_sha256")
    assert payload_sha == adapter.canonical_sha256(unsigned)
    assert result["every_offspring_decode_repair_attested"] is True
    assert result["offspring_repair_operator_call_count"] >= 2
    assert result["terminal_physical_replay_attested"] is True
    assert result["hard_spec"] == preflight.CURRENT_STAGE_SPEC
    assert result["stage_spec_sha256"] == preflight.CURRENT_STAGE_SPEC_SHA256
    assert result["artifact_inventory_sha256"] == adapter.canonical_sha256(
        result["artifact_inventory"]
    )
    for artifact in result["artifact_inventory"].values():
        path = output / artifact["path"]
        assert path.stat().st_size == artifact["size_bytes"]
        assert adapter.sha256_file(path) == artifact["sha256"]
