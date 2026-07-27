from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
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
from tools import tier1_corrected_current7_receipt as current7_receipt
from tools import tier1_corrected_current7_slurm_seed_runner as slurm_seed_runner


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


def test_explicit_relocation_uses_content_not_source_absolute_paths(tmp_path):
    paths = _synthetic_generation(tmp_path / "source")
    report = dict(paths["report"])
    documentary_generation = json.loads(
        Path(paths["candidate"]).read_text(encoding="utf-8")
    )["generation_path"]
    relocated = tmp_path / "linux-worker"
    relocated_generation = (
        relocated
        / "registry"
        / "generations"
        / Path(paths["generation"]).name
    )
    shutil.copytree(Path(paths["generation"]), relocated_generation)
    relocated_candidate = relocated / "evidence" / "candidate.json"
    relocated_quality = relocated / "evidence" / "quality.json"
    relocated_candidate.parent.mkdir(parents=True)
    shutil.copy2(Path(paths["candidate"]), relocated_candidate)
    shutil.copy2(Path(paths["quality"]), relocated_quality)
    relocated_dataset = relocated / "dataset" / "strict.parquet"
    relocated_profile = relocated / "profile" / "standard.json"
    relocated_dataset.parent.mkdir()
    relocated_profile.parent.mkdir()
    shutil.copy2(Path(report["dataset_path"]), relocated_dataset)
    shutil.copy2(Path(report["profile_path"]), relocated_profile)
    dataset_bytes = relocated_dataset.read_bytes()
    shutil.rmtree(tmp_path / "source" / "runtime")

    with pytest.raises((FileNotFoundError, RuntimeError)):
        adapter.authenticate_corrected_generation(
            generation=relocated_generation,
            candidate_path=relocated_candidate,
            quality_path=relocated_quality,
        )
    authenticated = adapter.authenticate_corrected_generation(
        generation=relocated_generation,
        candidate_path=relocated_candidate,
        quality_path=relocated_quality,
        dataset_path_override=relocated_dataset,
        profile_path_override=relocated_profile,
        expected_documentary_generation_path=documentary_generation,
    )
    assert authenticated.dataset_path == relocated_dataset.resolve()
    assert authenticated.profile_path == relocated_profile.resolve()
    assert authenticated.evidence["relocation"]["enabled"] is True
    assert authenticated.evidence["relocation"][
        "documentary_generation_path"
    ] == documentary_generation
    assert authenticated.evidence["relocation"][
        "runtime_generation_path"
    ] == str(relocated_generation.resolve())

    with pytest.raises(RuntimeError, match="candidate identity mismatch"):
        adapter.authenticate_corrected_generation(
            generation=relocated_generation,
            candidate_path=relocated_candidate,
            quality_path=relocated_quality,
            dataset_path_override=relocated_dataset,
            profile_path_override=relocated_profile,
            expected_documentary_generation_path=r"Z:\wrong\G0",
        )
    with pytest.raises(RuntimeError, match="requires dataset, profile"):
        adapter.authenticate_corrected_generation(
            generation=relocated_generation,
            candidate_path=relocated_candidate,
            quality_path=relocated_quality,
            dataset_path_override=relocated_dataset,
            expected_documentary_generation_path=documentary_generation,
        )

    relocated_dataset.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="dataset fingerprint mismatch"):
        adapter.authenticate_corrected_generation(
            generation=relocated_generation,
            candidate_path=relocated_candidate,
            quality_path=relocated_quality,
            dataset_path_override=relocated_dataset,
            profile_path_override=relocated_profile,
            expected_documentary_generation_path=documentary_generation,
        )
    relocated_dataset.write_bytes(dataset_bytes)
    _write_json(relocated_profile, {"tampered": True})
    with pytest.raises(RuntimeError, match="profile fingerprint mismatch"):
        adapter.authenticate_corrected_generation(
            generation=relocated_generation,
            candidate_path=relocated_candidate,
            quality_path=relocated_quality,
            dataset_path_override=relocated_dataset,
            profile_path_override=relocated_profile,
            expected_documentary_generation_path=documentary_generation,
        )


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
        self.fitted = types.SimpleNamespace(n_jobs=1)
        bundle = dict(bundle)
        bundle.setdefault("models", [("extratrees", self.fitted)])
        self.bundle = bundle
        self.features = bundle["features"]
        self.value = value
        self.half_width = half_width
        self.inference_threads = 1

    def configure_inference_threads(self, threads=1):
        self.inference_threads = int(threads)
        self.fitted.n_jobs = 1
        return {
            "threads": int(threads),
            "model_count": 1,
            "families": ["extratrees"],
            "family_threads": {"extratrees": 1},
            "semaphore_free_families": ["extratrees"],
            "semaphore_free_sklearn_forest": True,
            "policy": preflight.FAMILY_SPECIFIC_INFERENCE_POLICY,
        }

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
        "h_gap1": 40.0,
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
        "nwh1": 500.0,
        "nwh2": 500.0,
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
        _decoded_row(h_gap1=39.0),
    ])
    assert preflight.minimum_physical_insulation_violation(frame, 40.0).tolist() == [
        1.0,
        2.0,
        1.0,
    ]
    assert preflight.minimum_physical_insulation_violation(
        pd.DataFrame([_decoded_row(h_gap1=20.0)]),
        40.0,
        primary_axial_minimum_mm=20.0,
    ).tolist() == [0.0]
    missing = frame.drop(columns=["h_gap1"])
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


class _ThreadControlledPredictor:
    def __init__(self, target, value, *, multithread_delta=0.0):
        self.fitted = types.SimpleNamespace(n_jobs=8)
        self.bundle = {
            "features": ["l1"],
            "target": target,
            "models": [("extratrees", self.fitted)],
        }
        self.features = self.bundle["features"]
        self.value = float(value)
        self.multithread_delta = float(multithread_delta)
        self.half_width = 0.1
        self.inference_threads = 8

    def configure_inference_threads(self, threads=1):
        self.inference_threads = int(threads)
        self.fitted.n_jobs = 1
        return {
            "threads": int(threads),
            "model_count": 1,
            "families": ["extratrees"],
            "family_threads": {"extratrees": 1},
            "semaphore_free_families": ["extratrees"],
            "semaphore_free_sklearn_forest": True,
            "policy": preflight.FAMILY_SPECIFIC_INFERENCE_POLICY,
        }

    def predict_mu_sigma(self, frame, conformal=True):
        assert conformal is True
        delta = self.multithread_delta if self.inference_threads > 1 else 0.0
        return (
            np.full(len(frame), self.value + delta, dtype=float),
            np.full(len(frame), self.half_width, dtype=float),
        )

    def disagreement(self, frame):
        return np.zeros(len(frame), dtype=float)


def test_production_family_binding_keeps_sklearn_forest_semaphore_free():
    modules = preflight.load_current7_modules(REPO)

    class NJobsModel:
        def __init__(self, value):
            self.n_jobs = -1
            self.value = float(value)

        def predict(self, frame):
            return np.full(len(frame), self.value, dtype=float)

    class CatBoostModel:
        def __init__(self, value):
            self.value = float(value)
            self.thread_counts = []

        def predict(self, frame, thread_count=None):
            self.thread_counts.append(thread_count)
            return np.full(len(frame), self.value, dtype=float)

    fitted_by_target = {}
    models = {}
    for target in adapter.CURRENT_REQUIRED_MODEL_TARGETS:
        fitted = {
            "lightgbm": NJobsModel(1.0),
            "xgboost": NJobsModel(2.0),
            "catboost": CatBoostModel(3.0),
            "extratrees": NJobsModel(4.0),
        }
        fitted_by_target[target] = fitted
        models[target] = modules.predictor.EnsemblePredictor({
            "models": list(fitted.items()),
            "features": ["l1"],
            "transform": "identity",
            "q90": 1.0,
        })

    binding = preflight.bind_surrogate_inference(
        models, modules.run_nsga2, threads=8
    )

    assert binding["threads_per_model"] == 8
    assert binding["family_threads"] == {
        "catboost": 8,
        "extratrees": 1,
        "lightgbm": 8,
        "xgboost": 8,
    }
    assert binding["semaphore_free_families"] == ["extratrees"]
    assert binding["semaphore_free_sklearn_forest"] is True
    assert binding["policy"] == preflight.FAMILY_SPECIFIC_INFERENCE_POLICY
    assert preflight._terminal_inference_binding_contract(binding) == (True, 8)
    for fitted in fitted_by_target.values():
        assert fitted["extratrees"].n_jobs == 1
        assert fitted["lightgbm"].n_jobs == 8
        assert fitted["xgboost"].n_jobs == 8

    first_target = adapter.CURRENT_REQUIRED_MODEL_TARGETS[0]
    models[first_target].predict_mu_sigma(pd.DataFrame({"l1": [0.0]}))
    assert fitted_by_target[first_target]["catboost"].thread_counts == [8]

    unsafe = copy.deepcopy(binding)
    unsafe["family_threads"]["extratrees"] = 8
    with pytest.raises(RuntimeError, match="inference binding contract mismatch"):
        preflight._terminal_inference_binding_contract(unsafe)


def _thread_sensitive_terminal_runner():
    from pymoo.core.population import Population

    problem, base_models, modules = _actual_problem(5)
    deltas = {
        "Llt_phys": 1.0e-6,
        "P_core_total": 1.0e-4,
    }
    models = {
        target: _ThreadControlledPredictor(
            target,
            predictor.value,
            multithread_delta=deltas.get(target, 0.0),
        )
        for target, predictor in base_models.items()
    }
    problem.models = models
    binding = preflight.bind_surrogate_inference(
        models, modules.run_nsga2, threads=8
    )
    runner = preflight.Current7Tier1Runner(
        authenticated=None,
        code_identity={},
        modules=modules,
        adapter_evidence={},
        model_cache=None,
        models=models,
        inference_binding=binding,
        density_gate=object(),
        problem=problem,
    )
    physical_evaluate, scaling = preflight.install_optimizer_scaling(
        problem,
        resonance_scale_hz=150.0,
        llt_scale_uh=0.3,
        all_thermal_scale_c=2.0,
        resonance_allowance_hz=3.0,
        llt_allowance_uh=0.1,
    )
    runner.physical_evaluate = physical_evaluate
    runner.optimizer_scaling = scaling
    raw = np.full((1, problem.n_var), 0.5)
    raw[:, 2] = 0.0
    coordinates, _repair = runner.repair_coordinates(
        raw, stage="thread_sensitive_terminal_fixture"
    )
    optimizer = runner.evaluate_coordinates(coordinates)
    population = Population.new(
        X=coordinates,
        F=optimizer["F"],
        G=optimizer["G"],
        CV=np.full((1, 1), 999.0),
    )
    return runner, coordinates, optimizer, population


def test_terminal_replay_serializes_and_canonicalizes_eight_thread_snapshot(
    tmp_path,
):
    runner, coordinates, optimizer, population = (
        _thread_sensitive_terminal_runner()
    )

    replay, evidence = runner.terminal_physical_replay(
        coordinates,
        expected_f=optimizer["F"],
        expected_g=optimizer["G"],
        terminal_population=population,
    )

    deterministic = evidence["deterministic_terminal_inference"]
    snapshot = evidence["multithread_optimizer_snapshot"]
    assert deterministic["optimizer_inference_threads"] == 8
    assert deterministic["terminal_replay_inference_threads"] == 1
    assert deterministic["objectives_bit_exact"] is True
    assert deterministic["optimizer_G_bit_exact"] is True
    assert deterministic["numeric_tolerance_relaxed"] is False
    assert deterministic["comparison_atol"] == 0.0
    assert deterministic["optimizer_binding_restored"] is True
    assert snapshot["maximum_absolute_objective_delta"] > 0.0
    assert snapshot["maximum_absolute_optimizer_G_delta"] > 0.0
    assert snapshot["numeric_comparison_used_as_terminal_gate"] is False
    assert snapshot["canonicalized_from_serial_authority"] is True
    assert snapshot["canonical_constraint_violation_sha256"]
    assert snapshot["stale_constraint_violation_cache_cleared"] is True
    assert snapshot["trajectory_rank_or_crowding_used_for_persistence"] is False
    assert snapshot["terminal_pareto_recomputed_from_canonical_F_G"] is True
    assert evidence["terminal_population_canonicalized"] is True
    assert np.array_equal(population.get("F"), replay["F"])
    assert np.array_equal(population.get("G"), replay["optimizer_G"])
    assert not np.array_equal(population.get("CV"), np.full((1, 1), 999.0))
    assert "terminal_model_predictions" in replay
    assert deterministic["terminal_model_predictions_sha256"]
    assert all(model.inference_threads == 8 for model in runner.models.values())
    assert all(model.fitted.n_jobs == 1 for model in runner.models.values())

    def unexpected_multithread_prediction(*_args, **_kwargs):
        raise AssertionError("persistence repeated terminal model inference")

    for model in runner.models.values():
        model.predict_mu_sigma = unexpected_multithread_prediction
    output = tmp_path / "serial_terminal_persistence"
    output.mkdir()
    result = types.SimpleNamespace(
        pop=population,
        tier1_terminal_physical_replay=replay,
    )
    preflight.persist_search_outputs(runner, result, output)
    assert np.array_equal(np.load(output / "terminal_F.npy"), replay["F"])
    assert np.array_equal(
        np.load(output / "terminal_G_optimizer.npy"), replay["optimizer_G"]
    )


def test_terminal_replay_rejects_one_ulp_serial_drift_and_restores_threads():
    runner, coordinates, optimizer, population = (
        _thread_sensitive_terminal_runner()
    )
    physical_evaluate = runner.physical_evaluate
    constraint_index = runner.problem.constraint_names.index(
        "core_group_manufacturability_limit"
    )

    def one_ulp_physical_drift(values, out, *args, **kwargs):
        physical_evaluate(values, out, *args, **kwargs)
        constraints = np.asarray(out["G"], dtype=float).copy()
        constraints[:, constraint_index] = np.nextafter(
            constraints[:, constraint_index], np.inf
        )
        out["G"] = constraints

    runner.physical_evaluate = one_ulp_physical_drift
    with pytest.raises(RuntimeError, match="deterministic serial terminal replay"):
        runner.terminal_physical_replay(
            coordinates,
            expected_f=optimizer["F"],
            expected_g=optimizer["G"],
            terminal_population=population,
        )

    assert all(model.inference_threads == 8 for model in runner.models.values())
    assert all(model.fitted.n_jobs == 1 for model in runner.models.values())


def test_terminal_replay_rejects_partial_inference_binding():
    runner, coordinates, optimizer, population = (
        _thread_sensitive_terminal_runner()
    )
    runner.inference_binding = {"threads_per_model": 8}

    with pytest.raises(RuntimeError, match="inference binding is partial"):
        runner.terminal_physical_replay(
            coordinates,
            expected_f=optimizer["F"],
            expected_g=optimizer["G"],
            terminal_population=population,
        )


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


def test_problem_wrapper_evaluates_staged_resonance_band_with_dynamic_schema():
    stage = {
        **preflight.CURRENT_STAGE_SPEC,
        "T_limit_C": 100.0,
        "size_W_max_mm": 1_000.0,
        "size_L_max_mm": 1_000.0,
        "resonance_min_Hz": 15_000.0,
        "resonance_max_Hz": 20_000.0,
    }
    models = _problem_models()
    problem = _fake_problem_class()(
        models,
        spec=stage,
        density_gate=object(),
        fixed_primary_turns=5,
    )
    assert problem.constraint_names == preflight.stage_constraint_names(stage)
    assert problem.n_ieq_constr == 20
    assert problem.temperature_contract["robust_upper_bound_C"] == 100.0

    coordinate = np.full((1, len(_SOBOL_DIMS)), 0.5)
    coordinate[0, 2] = 0.0
    coordinate = problem.repair_unit_coordinates(coordinate)
    out = {}
    problem._evaluate(coordinate, out)
    row = out["frame"].iloc[0]
    screen = preflight.derive_half_magnetizing_self_resonance(
        {
            "Llt_phys": 27.5,
            "k": 0.9,
            "C_tx_tx_F": 1.2e-9,
            "C_rx_rx_F": 2.4e-11,
        },
        row,
        magnetizing_inductance_factor=0.5,
    )
    expected = preflight.half_magnetizing_resonance_band_violations(
        screen["f_res_min_tx_rx_only_Hz"], stage
    )
    for name, violation in expected.items():
        index = problem.constraint_names.index(name)
        assert out["G"][0, index] == pytest.approx(violation)


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


def test_role_partition_keeps_37_and_60_in_actual_initialization_deterministically(
    tmp_path,
):
    default_problem, models, modules = _actual_problem(6)
    final_spec = {
        **preflight.CURRENT_STAGE_SPEC,
        "T_limit_C": 100.0,
        "size_W_max_mm": 1_000.0,
        "size_L_max_mm": 1_000.0,
        "resonance_max_Hz": 20_000.0,
    }
    standard = np.asarray([
        0.3750093752343808,
        0.5,
        0.0020833333333333333,
        0.5166666666666667,
        0.7128571428571429,
        0.5,
        0.34833333333333333,
        0.16667222240741356,
        0.0,
        0.0,
        *([1.0 / 3.0] * 4),
        0.34444444444444444,
        *([1.0 / 3.0] * 4),
        0.5,
        1.0,
        0.0,
        0.5,
        0.4999999999999999,
        0.5,
    ])
    assert standard.shape == (25,)
    topologies = (34, 35, 36, 37, 38, 39, 60)
    donors = []
    for copy_index in range(8):
        for topology in topologies:
            row = standard.copy()
            row[2] = (60 - topology) / 48.0
            row[24] = 0.45 + 0.01 * copy_index
            donors.append(row)
    donors = np.asarray(donors)
    values = np.vstack((standard.reshape(1, -1), donors))
    role = {
        "schema_version": preflight.WARM_ROLE_PARTITION_SCHEMA,
        "fixed_primary_turns": 6,
        "total_count": len(values),
        "standard_hard_feasible_candidates": {
            "start": 0,
            "count": 1,
            "coordinate_unit_sha256": adapter.canonical_sha256(
                values[:1].tolist()
            ),
            "full_hard_geometry_filter_required": True,
            "ordinary_warm_sampling_allowed": True,
            "source_artifact_sha256": "a" * 64,
            "source_contract_file_sha256": "b" * 64,
            "source_hard_geometry_joint_count": 1,
        },
        "basin_structural_coordinate_donors": {
            "start": 1,
            "count": len(donors),
            "coordinate_unit_sha256": adapter.canonical_sha256(
                donors.tolist()
            ),
            "required_gate_names": list(
                preflight.STRUCTURAL_DONOR_REQUIRED_GATES
            ),
            "optimizer_gate_names_not_used_for_donor_admission": list(
                preflight.STRUCTURAL_DONOR_OPTIMIZER_GATES
            ),
            "required_topologies_N2_main": list(topologies),
            "source_selection_contract_sha256": "c" * 64,
            "coordinate_donors_only": True,
            "protected_topology_slots_only": True,
            "ordinary_warm_sampling_allowed": False,
        },
        "maximum_combined_authenticated_fraction": 0.5,
        "minimum_fresh_random_fraction": 0.5,
        "physical_constraint_G_mutation": False,
        "physical_objective_mutation": False,
        "terminal_physical_replay_required": True,
    }
    role["sha256"] = adapter.canonical_sha256(role)
    warm_path = tmp_path / "role-partitioned-warm.npy"
    np.save(warm_path, values, allow_pickle=False)

    class StopBeforeOptimizer(RuntimeError):
        pass

    captured = []
    for _repeat in range(2):
        problem = type(default_problem)(
            models,
            spec=final_spec,
            density_gate=lambda frame: np.full(len(frame), -1.0),
            fixed_primary_turns=6,
        )
        constraint_names = tuple(problem.constraint_names)
        hard_contract_sha = problem.hard_constraint_contract_sha256
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
        observed = []

        def capture(evidence):
            observed.append(evidence)
            raise StopBeforeOptimizer

        with pytest.raises(StopBeforeOptimizer):
            runner.run_one(
                seed=2457500999,
                population=128,
                max_generations=2,
                warm_start_path=warm_path,
                warm_start_sha256=adapter.sha256_file(warm_path),
                warm_start_role_partition=role,
                pre_optimization_callback=capture,
            )
        assert tuple(problem.constraint_names) == constraint_names
        assert problem.hard_constraint_contract_sha256 == hard_contract_sha
        captured.append(observed[0])

    assert captured[0] == captured[1]
    initialization = captured[0]["initial_population"]
    current = initialization["current_run_nsga2"]
    assert current["authenticated_standard_hard_feasible_warm_count"] == 1
    assert current["authenticated_structural_donor_count"] == 56
    assert current["fresh_random_count"] == 71
    topology = initialization["turn_split_sub_islands"]
    assert topology["topology_counts_after_current_repair"]["37"] >= 8
    assert topology["topology_counts_after_current_repair"]["60"] >= 8
    assert all(
        item["source_is_authenticated_warm"]
        for item in topology["donor_sources"]
    )
    preservation = initialization["role_partition_preservation"]
    assert preservation[
        "standard_warm_rows_preserved_after_topology_seeding"
    ] is True
    assert preservation["standard_warm_count"] == 1
    role_audit = captured[0]["authenticated_warm_start"]["role_audit"]
    donor_filter = role_audit["basin_structural_donor_filter"]
    assert donor_filter["decoded_unique_count"] == 56
    assert donor_filter["optimizer_gate_passed_counts"] == {
        "shrink": 56,
        "box": 8,
        "analytical_B": 8,
    }
    assert donor_filter["physical_constraint_G_mutation"] is False
    assert donor_filter["additional_model_evaluations"] == 0
    assert captured[0]["offspring_repair_operator"]["call_count"] == 0

    terminal_problem = type(default_problem)(
        models,
        spec=final_spec,
        density_gate=lambda frame: np.full(len(frame), -1.0),
        fixed_primary_turns=6,
    )
    terminal_runner = preflight.Current7Tier1Runner(
        authenticated=None,
        code_identity={},
        modules=modules,
        adapter_evidence={},
        model_cache=None,
        models=models,
        inference_binding={},
        density_gate=object(),
        problem=terminal_problem,
    )
    result = terminal_runner.run_one(
        seed=2457500999,
        population=128,
        max_generations=2,
        warm_start_path=warm_path,
        warm_start_sha256=adapter.sha256_file(warm_path),
        warm_start_role_partition=role,
    )
    execution = result.tier1_repair_audit
    assert execution["terminal_physical_replay"][
        "optimizer_physical_G_match"
    ] is True
    assert execution["authoritative_terminal_G"] == (
        "physical_unscaled_replay"
    )
    assert result.tier1_topology_evolution_audit[
        "all_required_topologies_preserved"
    ] is True


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


@pytest.mark.parametrize("target", ["P_Rx_side_total", "B_mean_core"])
def test_terminal_surrogate_physicality_gate_rejects_without_clamping(target):
    predictions = {
        target: {
            "mean": np.ones(2, dtype=float),
            "q90_conformal_half_width": np.full(2, 0.1, dtype=float),
        }
        for target in adapter.CURRENT_REQUIRED_MODEL_TARGETS
    }
    predictions["k"]["mean"] = np.full(2, 0.9, dtype=float)
    predictions[target]["mean"] = np.asarray([-0.25, 5.0])

    valid, evidence = preflight._terminal_surrogate_physicality_gate(
        predictions,
        population_size=2,
    )

    assert valid.tolist() == [False, True]
    assert evidence["invalid_population_indices"] == [0]
    assert evidence["violation_count_by_target"] == {target: 1}
    assert evidence["violations"] == [{
        "population_index": 0,
        "target": target,
        "rule": "finite_and_nonnegative",
        "predicted_mean": -0.25,
    }]
    assert evidence["clamping_performed"] is False
    assert evidence["raw_predictions_preserved"] is True
    unsigned = dict(evidence)
    assert unsigned.pop("sha256") == adapter.canonical_sha256(unsigned)


def test_least_violation_prefers_valid_alternative_and_preserves_raw_argmin():
    optimizer_g = np.asarray([
        [0.1, -1.0],
        [0.2, -1.0],
        [0.3, -1.0],
    ])
    selected = preflight._select_terminal_least_violation(
        optimizer_g,
        decoder_valid=[True, True, True],
        surrogate_physical_valid=[False, True, True],
    )
    assert selected["raw_global_index"] == 0
    assert selected["selected_index"] == 1
    assert selected["eligible_count"] == 2
    assert selected["fallback_to_quarantined_raw_global_least"] is False

    fallback = preflight._select_terminal_least_violation(
        optimizer_g,
        decoder_valid=[True, True, True],
        surrogate_physical_valid=[False, False, False],
    )
    assert fallback["raw_global_index"] == 0
    assert fallback["selected_index"] == 0
    assert fallback["eligible_count"] == 0
    assert fallback["fallback_to_quarantined_raw_global_least"] is True


def test_persistence_quarantines_negative_loss_instead_of_aborting_seed(
    tmp_path,
):
    problem, models, modules = _actual_problem(6)
    models["P_Rx_side_total"].value = -0.25
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
    warm_path = tmp_path / "negative-loss-warm.npy"
    np.save(warm_path, raw, allow_pickle=False)
    result = runner.run_one(
        seed=2407500057,
        population=64,
        max_generations=2,
        warm_start_path=warm_path,
        warm_start_sha256=adapter.sha256_file(warm_path),
    )
    output = tmp_path / "negative-loss-persisted"
    output.mkdir()

    persisted = preflight.persist_search_outputs(runner, result, output)

    gate = persisted["infeasibility_report"][
        "terminal_surrogate_physicality_gate"
    ]
    assert gate["population_size"] == 64
    assert gate["valid_count"] == 0
    assert gate["invalid_count"] == 64
    assert gate["violation_count_by_target"] == {"P_Rx_side_total": 64}
    assert gate["clamping_performed"] is False
    assert persisted["physical_feasible_count"] == 0
    assert persisted["feasible_pareto_count"] == 0
    least = json.loads(
        (output / "least_violation_candidates.json").read_text(
            encoding="utf-8"
        )
    )
    assert least["candidate_count"] == 1
    assert least["raw_global_least_population_index"] == (
        least["selected_least_population_index"]
    )
    assert least["fallback_to_quarantined_raw_global_least"] is True
    candidate = least["candidates"][0]
    assert candidate["candidate_record_status"] == (
        "quarantined_nonphysical_surrogate_output"
    )
    assert candidate["predicted_Rx_side_winding_loss_W"] == -0.25
    assert candidate["terminal_surrogate_physicality_gate"]["passed"] is False
    assert candidate["terminal_surrogate_physicality_gate"][
        "clamping_performed"
    ] is False
    assert candidate["physical_feasible"] is False
    assert np.load(output / "terminal_X.npy", allow_pickle=False).shape == (
        64,
        problem.n_var,
    )


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

    def bound_fake_inference(models, threads=1):
        configured = [
            model.configure_inference_threads(threads)
            for model in models.values()
        ]
        return {
            "threads_per_model": int(threads),
            "target_count": len(configured),
            "model_count": sum(item["model_count"] for item in configured),
            "families": sorted({
                family
                for item in configured
                for family in item["families"]
            }),
            "family_threads": {"extratrees": 1},
            "semaphore_free_families": ["extratrees"],
            "semaphore_free_sklearn_forest": True,
            "policy": preflight.FAMILY_SPECIFIC_INFERENCE_POLICY,
        }

    modules = preflight.Current7Modules(
        run_nsga2=types.SimpleNamespace(
            _bound_surrogate_inference=bound_fake_inference
        ),
        nsga2_problem=None,
        predictor=None,
        train_models=None,
        geometry_metrics=None,
        design_summary=None,
        input_parameter=None,
        evidence={"synthetic": True},
    )
    problem_class = _fake_problem_class()
    staged_spec = {
        **preflight.CURRENT_STAGE_SPEC,
        "T_limit_C": 100.0,
        "size_W_max_mm": 1_000.0,
        "size_L_max_mm": 1_000.0,
        "resonance_max_Hz": 20_000.0,
    }
    first_runner = preflight.Current7Tier1Runner(
        authenticated=authenticated,
        code_identity={"path": str(tmp_path), "revision": "b" * 40, "clean": True},
        modules=modules,
        adapter_evidence=manifest,
        model_cache=fake_cache,
        models=all_models,
        inference_binding=bound_fake_inference(all_models, threads=1),
        density_gate=object(),
        problem=problem_class(
            all_models,
            spec=staged_spec,
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
    assert receipt["problem_contract"]["stage_spec"] == staged_spec
    assert receipt["problem_contract"]["constraint_count"] == 20
    staged_identity = current7_receipt.validate_adapter_receipt(receipt)
    assert staged_identity.hard_spec == staged_spec
    assert staged_identity.constraint_names == preflight.stage_constraint_names(
        staged_spec
    )
    assert staged_identity.launch_eligible is True
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
        inference_binding=preflight.bind_surrogate_inference(
            models, modules.run_nsga2, threads=8
        ),
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
        "problem_contract": {
            "stage_spec": problem.stage_spec,
            "stage_spec_sha256": problem.stage_spec_sha256,
        },
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
        "hard_spec": problem.stage_spec,
        "hard_spec_sha256": problem.stage_spec_sha256,
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
        "--stage-spec-json", json.dumps(
            problem.stage_spec, sort_keys=True, separators=(",", ":")
        ),
        "--stage-spec-sha256", problem.stage_spec_sha256,
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

    # Cross the actual producer/consumer seam: the immutable Slurm runner must
    # accept the real search entrypoint's truthful preflight and terminal seal.
    slurm_identity = {
        "hard_spec": preflight.CURRENT_STAGE_SPEC,
        "hard_spec_sha256": preflight.CURRENT_STAGE_SPEC_SHA256,
        "constraint_version": adapter.CURRENT_STAGE_HARD_CONTRACT["stage"],
        "constraint_names": list(preflight.CURRENT7_CONSTRAINT_NAMES),
    }
    slurm_manifest = {
        **manifest,
        "adapter_receipt": {"identity": slurm_identity},
    }
    slurm_payload = {
        "bundle_id": bundle_id,
        "hard_spec": problem.stage_spec,
        "hard_spec_sha256": problem.stage_spec_sha256,
        "constraint_version": slurm_identity["constraint_version"],
        "constraint_names": slurm_identity["constraint_names"],
        "seed": 5,
        "lane": {"island_id": island_id, "fixed_primary_turns": 5},
        "population": 64,
        "max_generations": 2,
        "inference_threads": 8,
        "generation_artifact_inventory_sha256": manifest[
            "generation_artifact_inventory_sha256"
        ],
        "adapter_manifest_sha256": receipt["adapter_manifest_sha256"],
        "train_report_sha256": adapter_evidence["train_report"]["sha256"],
        "dataset_sha256": adapter_evidence["dataset"]["sha256"],
        "profile_canonical_sha256": adapter_evidence["profile"][
            "canonical_sha256"
        ],
        "temperature_contract_sha256": (
            adapter.CURRENT_TEMPERATURE_CONTRACT_SHA256
        ),
        "hard_constraint_contract_sha256": (
            adapter.CURRENT_STAGE_HARD_CONTRACT_SHA256
        ),
        "island_profile_sha256": island_profile["sha256"],
        "warm_artifact_sha256": warm_sha,
        "warm_contract_sha256": warm_record["contract"]["sha256"],
        "optimizer_repair_contract_sha256": repair_sha,
        "maximum_peak_rss_bytes": 10_000_000_000,
    }
    assert slurm_seed_runner.validate_remote_preflight(
        remote,
        payload=slurm_payload,
        manifest=slurm_manifest,
        optimizer_pid=remote["optimizer_pid"],
    ) >= remote["observed_peak_rss_bytes"]
    slurm_seed_runner.validate_result(
        result,
        payload=slurm_payload,
        manifest=slurm_manifest,
    )
