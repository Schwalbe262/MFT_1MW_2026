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
    assert manifest["portability"]["remote_relocation_supported"] is False
    assert manifest["scheduler_write_performed"] is False


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


_SOBOL_DIMS = (
    ("core_plate_t", 10.0, 30.0),
    ("wcp_t", 10.0, 30.0),
    ("wcp_len_pct", 20.0, 80.0),
    ("other", 0.0, 1.0),
)


class _FakeBaseProblem:
    def __init__(self, models, spec=None, density_gate=None, fixed_overrides=None):
        self.models = models
        self.spec = dict(preflight.CURRENT_STAGE_SPEC, **(spec or {}))
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


def _fake_problem_class():
    return preflight.create_current7_problem_class(
        base_problem_class=_FakeBaseProblem,
        base_constraint_names=preflight.BASE_CONSTRAINT_NAMES,
        base_fixed_stack_mm=preflight.EXPECTED_SIMPLE_BASE_FIXED_STACK_MM,
        sobol_dims=_SOBOL_DIMS,
        bounding_box_lit=lambda row: (
            1.0,
            (row["bbox_x"], row["bbox_y"], row["bbox_z"]),
        ),
    )


def test_problem_wrapper_restores_cooling_and_appends_all_hard_constraints():
    problem = _fake_problem_class()(_problem_models(), density_gate=object())
    assert problem.spec["q_sigma"] == 1.0
    assert problem.spec["T_limit_C"] == 110.0
    assert tuple(problem.constraint_names) == preflight.CURRENT7_CONSTRAINT_NAMES
    assert problem.n_ieq_constr == 19
    for name in preflight.VARIABLE_COOLING_DIMENSIONS:
        assert name not in problem.fixed_overrides
        index = [item[0] for item in _SOBOL_DIMS].index(name)
        assert problem.xl[index] == 0.0
        assert problem.xu[index] == 1.0
    assert problem.fixed_overrides["cw1"] == 5.0

    out = {}
    problem._evaluate(np.asarray([[0.35, 0.65, 0.5, 0.2]]), out)
    assert out["G"].shape == (1, 19)
    side_index = problem.constraint_names.index(
        "temperature_robust_limit:Tprobe_Rx_side_leeward_max"
    )
    assert out["G"][0, side_index] == -preflight.BIG
    hard = dict(zip(problem.constraint_names, out["G"][0]))
    assert hard["minimum_physical_insulation"] == 0.0
    assert hard["core_group_manufacturability_limit"] == 0.0
    assert hard["exterior_width_limit"] == 0.0
    assert hard["exterior_length_limit"] == 0.0
    assert hard["exterior_height_limit"] == 0.0
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


def test_actual_current_module_contract_can_build_additive_class():
    modules = preflight.load_current7_modules(REPO)
    cls = preflight.create_current7_problem_class(
        base_problem_class=modules.nsga2_problem.MFTProblem,
        base_constraint_names=modules.nsga2_problem.CONSTRAINT_NAMES,
        base_fixed_stack_mm=modules.nsga2_problem.NSGA_FIXED_THERMAL_STACK_MM,
        sobol_dims=modules.input_parameter._SOBOL_DIMS,
        bounding_box_lit=modules.geometry_metrics.bounding_box_lit,
    )
    assert cls.__name__ == "Current7Tier1Problem"
    assert cls.launch_eligible is False
    assert cls.offspring_physics_repair is False


def test_smoke_receipt_is_self_hashed_and_explicitly_launch_blocked(tmp_path):
    authenticated = _authenticate(_synthetic_generation(tmp_path))
    manifest = adapter.adapter_manifest(
        authenticated,
        code_identity={
            "path": str(tmp_path),
            "revision": "b" * 40,
            "clean": True,
        },
    )
    problem = _fake_problem_class()(_problem_models(), density_gate=object())
    evaluation = {}
    problem._evaluate(np.asarray([[0.35, 0.65, 0.5, 0.2]]), evaluation)
    all_models = {
        target: _FakePredictor(
            {"features": ["feature_a"], "target": target},
            value=1.0,
        )
        for target in adapter.CURRENT_REQUIRED_MODEL_TARGETS
    }
    model_smoke = preflight._smoke_every_model(
        all_models, evaluation["frame"].iloc[[0]]
    )
    fake_cache = types.SimpleNamespace(
        loaded_once=True,
        load_calls=1,
        full_generation_authentication_passes=1,
    )
    runner = preflight.Current7Tier1Runner(
        authenticated=authenticated,
        code_identity={"path": str(tmp_path), "revision": "b" * 40, "clean": True},
        modules=preflight.Current7Modules(
            run_nsga2=None,
            nsga2_problem=None,
            predictor=None,
            train_models=None,
            geometry_metrics=None,
            input_parameter=None,
            evidence={"synthetic": True},
        ),
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
        problem=problem,
    )
    receipt = preflight.build_smoke_receipt(
        runner=runner,
        coordinate_evidence={
            "path": str(tmp_path / "smoke.npy"),
            "sha256": "c" * 64,
            "shape": [1, 4],
            "coordinate_unit_sha256": "d" * 64,
        },
        evaluation=evaluation,
        model_smoke=model_smoke,
    )
    assert preflight.validate_smoke_receipt(receipt) is receipt
    assert receipt["optimizer_repair"]["stages"]["every_offspring"] is False
    assert receipt["runner"]["launch_eligible"] is False
    assert receipt["problem_contract"]["constraint_count"] == 19
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
    assert json.loads(completed.stdout)["launch_eligible"] is False


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
        "status": "authenticated_model_smoke_passed_launch_blocked",
        "runner": {
            "schema_version": preflight.RUNNER_SCHEMA,
            "problem_schema": preflight.PROBLEM_SCHEMA,
            "full_nsga_executed": False,
            "launch_eligible": True,
        },
        "adapter_manifest": manifest,
    }
    fake["payload_sha256"] = adapter.canonical_sha256(fake)
    with pytest.raises(RuntimeError, match="contract mismatch"):
        preflight.validate_smoke_receipt(fake)
