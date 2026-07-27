from __future__ import annotations

import copy
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from module.mft_goal_20260726_contract import canonical_sha256
from tools import mft_goal_corrected_physics_nsga_lane as lane
from tools import mft_goal_diagnostic_compact_scout as scout
from tools import tier1_corrected_generation_preflight as preflight


@pytest.fixture(scope="module")
def model() -> dict[str, Any]:
    return lane._load_bound_model(lane.DEFAULT_MODEL_PATH)


@pytest.fixture
def profile(model: dict[str, Any]) -> dict[str, Any]:
    lane._set_active_model(model)
    return lane._build_search_profile(
        object(),
        geometry_profile=scout._geometry_profile(20.0),
        authorized_seed_start=lane.SEED_START,
        secondary_gap_mode=scout.SECONDARY_GAP_MODE_BOUNDED,
    )


def test_profile_maps_latest_acceptance_and_separate_exact60(
    profile: dict[str, Any],
) -> None:
    validated = lane._validate_search_profile(profile)
    effective = validated["effective_constraint_profile"]
    gate = validated["physics_delta_rx_resonance_gate"]

    assert effective["size_limits_mm"] == {
        "W": 1200.0,
        "L": 900.0,
        "H": 750.0,
    }
    assert effective["temperature_family_limits_C"] == {
        "primary_winding": 110.0,
        "secondary_winding": 130.0,
        "core": 130.0,
    }
    assert validated["authorized_seed_start"] == 2_607_264_400
    assert validated["authorized_seed_count"] == 60
    assert validated["authorized_seed_end_inclusive"] == 2_607_264_459
    assert validated["scheduler_priority"] == 100
    assert validated["campaign_id"] == lane.CAMPAIGN_ID
    assert validated["worker_entrypoint"] == lane.WORKER_ENTRYPOINT
    assert "raw_same_metric_capacitance_gate" not in validated
    assert gate["primary_magnetizing_inductance_H"] == 0.002
    assert gate["secondary_inductance_H"] == 0.2
    assert gate["minimum_frequency_Hz"] == 15_000.0
    assert gate["frequency_lcb_formula"] == "1/(2*pi*sqrt(0.2*C_ucb))"
    assert gate["feature_extrapolation_penalty_active"] is True
    assert gate["raw_two_net_capacitance_predictor_called"] is False
    assert gate["raw_two_net_capacitance_G_present"] is False
    assert gate["single_transfer_ratio_used"] is False
    assert gate["legacy_half_magnetizing_resonance_G_present"] is False
    assert gate["fixed20T_turn_graded_FEA_retraining_required"] is True
    assert validated["fixed_core_plate_thickness_mm"] == 20.0
    assert validated["fixed_winding_cold_plate_thickness_mm"] == 20.0
    assert validated["temperature_contract_sha256"] == effective[
        "temperature_contract_sha256"
    ]
    assert validated["cooling_or_TIM_contract_mutated"] is False


def test_corrected_successor_seed_interval_is_bounded(
    model: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("SEED_START", "SEED_COUNT", "SEED_END"):
        monkeypatch.setattr(lane, name, getattr(lane, name))
    monkeypatch.setattr(lane, "_ACTIVE_MODEL", lane._ACTIVE_MODEL)
    lane._set_seed_interval(2_607_264_460, 40)
    assert lane.SEED_START == 2_607_264_460
    assert lane.SEED_COUNT == 40
    assert lane.SEED_END == 2_607_264_499
    lane._set_active_model(model)
    profile = lane._build_search_profile(
        object(),
        geometry_profile=scout._geometry_profile(20.0),
        authorized_seed_start=lane.SEED_START,
        secondary_gap_mode=scout.SECONDARY_GAP_MODE_BOUNDED,
    )
    validated = lane._validate_search_profile(profile)
    assert validated["authorized_seed_start"] == 2_607_264_460
    assert validated["authorized_seed_count"] == 40
    assert validated["authorized_seed_end_inclusive"] == 2_607_264_499
    with pytest.raises(RuntimeError, match="outside the sealed range"):
        lane._set_seed_interval(2_607_264_999, 2)


def test_bundle_configuration_reads_successor_interval_before_count_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("SEED_START", "SEED_COUNT", "SEED_END"):
        monkeypatch.setattr(lane, name, getattr(lane, name))
    monkeypatch.setattr(lane, "_coordinator_offload", lambda: object())
    monkeypatch.setattr(lane, "configure_runtime", lambda _model: None)
    task_paths = [f"tasks/seed-{index}.json" for index in range(40)]
    for relative in task_paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    (tmp_path / "bundle_manifest.json").write_text(
        '{"task_relative_paths": ['
        + ",".join(f'"{relative}"' for relative in task_paths)
        + "]}\n",
        encoding="utf-8",
    )

    def configure_from_first(_path: Path) -> dict[str, str]:
        lane._set_seed_interval(2_607_264_460, 40)
        return {"model": "fixture"}

    monkeypatch.setattr(lane, "_model_from_payload", configure_from_first)

    assert lane._configure_from_bundle(tmp_path) == {"model": "fixture"}
    assert lane.SEED_START == 2_607_264_460
    assert lane.SEED_COUNT == 40
    assert lane.SEED_END == 2_607_264_499


def test_corrected_runtime_authorization_reaches_diagnostic_gate(
    model: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutated_runtime_attributes = {
        lane.goal_launch: ("GOAL_RUNTIME_TOOL_FILES",),
        scout: (
            "CAMPAIGN_ID",
            "ACTIVATION_SCHEMA",
            "BUNDLE_SCHEMA",
            "TASK_SCHEMA",
            "SCHEDULER_SCHEMA",
            "RESULT_SCHEMA",
            "SEARCH_PROFILE_SCHEMA",
            "SEARCH_PROFILE_INSTALLATION_SCHEMA",
            "_build_search_profile",
            "_validate_search_profile",
            "_secondary_gap_mode_from_profile",
            "_install_search_profile",
        ),
        preflight: (
            "GOAL_FIXED_LM_RESONANCE_SCHEMA",
            "GOAL_DIAGNOSTIC_COMPACT_ACTIVATION_SCHEMA",
            "GOAL_DIAGNOSTIC_COMPACT_CAMPAIGN_ID",
            "goal_fixed_lm2mh_resonance_contract",
            "install_goal_fixed_lm2mh_resonance",
            "derive_fixed_lm2mh_self_resonance",
            "_terminal_surrogate_physicality_gate",
        ),
    }
    for module, names in mutated_runtime_attributes.items():
        for name in names:
            monkeypatch.setattr(module, name, getattr(module, name))
    monkeypatch.setattr(lane, "_ACTIVE_MODEL", lane._ACTIVE_MODEL)
    lane.configure_runtime(model)
    assert (
        preflight.GOAL_DIAGNOSTIC_COMPACT_ACTIVATION_SCHEMA
        == lane.ACTIVATION_SCHEMA
    )
    assert (
        preflight.GOAL_DIAGNOSTIC_COMPACT_CAMPAIGN_ID
        == lane.CAMPAIGN_ID
    )

    artifacts = {"authenticated_model": {"sha256": "1" * 64}}
    dataset_sha = "2" * 64
    quality_sha = "3" * 64
    hard_constraint_sha = "4" * 64
    compact_contract_sha = "5" * 64
    compact_bank_sha = "6" * 64
    source_identity = {
        "dataset_sha256": dataset_sha,
        "evaluation_model_sha256": canonical_sha256(artifacts),
        "quality_status_sha256": quality_sha,
    }
    activation = {
        "schema_version": lane.ACTIVATION_SCHEMA,
        "campaign_id": lane.CAMPAIGN_ID,
        "fixed_primary_turns": 6,
        "source_identity": source_identity,
        "source_quality_passed": True,
        "screening_only": True,
        "production_eligible": False,
        "final_design_claim_allowed": False,
        "fresh512_activation_evidence": False,
        "reserved_fresh512_seed_interval_used": False,
        "fixed_lm2mh_resonance_contract_sha256": (
            lane.corrected_resonance_contract()["sha256"]
        ),
        "effective_hard_constraint_contract_sha256": hard_constraint_sha,
        "compact_search_contract_sha256": compact_contract_sha,
        "compact_coordinate_bank_sha256": compact_bank_sha,
        "scheduler_write_performed": False,
        "scheduler_submission_performed": False,
    }
    activation["payload_sha256"] = canonical_sha256(activation)
    authorization = preflight.seal_goal_compact_run_authorization(
        authorization_mode=preflight.GOAL_COMPACT_AUTH_DIAGNOSTIC,
        source_activation_schema=lane.ACTIVATION_SCHEMA,
        source_activation_payload_sha256=activation["payload_sha256"],
        seed=lane.SEED_START,
        fixed_primary_turns=6,
        dataset_sha256=dataset_sha,
        evaluation_model_sha256=source_identity[
            "evaluation_model_sha256"
        ],
        quality_status_sha256=quality_sha,
        source_quality_passed=True,
        effective_hard_constraint_contract_sha256=hard_constraint_sha,
        compact_search_contract_sha256=compact_contract_sha,
        compact_coordinate_bank_sha256=compact_bank_sha,
    )
    problem = SimpleNamespace(
        fixed_primary_turns=6,
        hard_constraint_contract_sha256=hard_constraint_sha,
        _goal_fixed_lm2mh_resonance_installed=True,
    )
    authenticated = SimpleNamespace(
        quality={"passed": True},
        report={"artifacts": artifacts},
        evidence={
            "dataset": {"sha256": dataset_sha},
            "quality_status": {"sha256": quality_sha},
        },
    )

    observed = preflight.validate_goal_compact_run_authorization(
        authorization,
        source_activation=activation,
        problem=problem,
        seed=lane.SEED_START,
        authenticated=authenticated,
        compact_contract={"sha256": compact_contract_sha},
        compact_bank={"sha256": compact_bank_sha},
    )
    assert observed["sha256"] == authorization["sha256"]


def test_frequency_lcb_is_computed_from_q90_ucb(
    model: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c_ucb = 5.0e-10
    expected = 1.0 / (2.0 * math.pi * math.sqrt(0.2 * c_ucb))
    prediction = {
        "physics_delta_Crx_mean_F": 4.5e-10,
        "physics_delta_Crx_q90_ucb_F": c_ucb,
        "physics_delta_fRx_q90_lcb_Hz": expected,
        "physics_delta_extrapolation_distance": 2.5,
    }
    monkeypatch.setattr(
        lane.physics,
        "_candidate_prediction",
        lambda _row, _model: copy.deepcopy(prediction),
    )
    observed = lane._candidate_prediction({"unused": True}, model)
    assert observed["physics_delta_Crx_q90_ucb_F"] == c_ucb
    assert observed["physics_delta_fRx_q90_lcb_Hz"] == pytest.approx(
        expected, rel=1e-13
    )
    assert 15_000.0 - observed["physics_delta_fRx_q90_lcb_Hz"] == (
        pytest.approx(15_000.0 - expected)
    )

    broken = dict(prediction)
    broken["physics_delta_fRx_q90_lcb_Hz"] = expected + 10.0
    monkeypatch.setattr(
        lane.physics,
        "_candidate_prediction",
        lambda _row, _model: copy.deepcopy(broken),
    )
    with pytest.raises(RuntimeError, match="formula drifted"):
        lane._candidate_prediction({"unused": True}, model)


class _FakeProblem:
    def __init__(self, *, llt_optimum_main_turns: int = 30) -> None:
        self.llt_optimum_main_turns = int(llt_optimum_main_turns)
        self.geometry_constraint_profile_sha256 = scout._geometry_profile(
            20.0
        )["sha256"]
        self.spec = {
            "size_limits_mm": {"W": 1200.0, "L": 1000.0, "H": 750.0},
            "Llt_target_uH": 20.0,
            "Llt_tol_uH": 1.0,
            "q_sigma": 1.0,
        }
        self.temperature_limits_C = {}
        self.temperature_contract = {}
        self.temperature_contract_sha256 = "0" * 64
        self.sobol_dimension_names = (
            "cw1",
            "gap1",
            "u_N2_side",
            "gap2",
            "core_plate_t",
            "wcp_t",
        )
        self.cw1_coordinate_index = 0
        self.n_var = len(self.sobol_dimension_names)
        self.xl = np.zeros(self.n_var, dtype=float)
        self.xu = np.ones(self.n_var, dtype=float)
        self.constraint_names = (
            "Llt_robust_band",
            preflight.RESONANCE_MINIMUM_CONSTRAINT,
            "exterior_width_limit",
        )
        self.constraint_index = {
            name: index for index, name in enumerate(self.constraint_names)
        }
        self.n_ieq_constr = len(self.constraint_names)
        self.hard_constraint_contract = {
            "schema": "fake",
            "constraint_names": list(self.constraint_names),
            "self_resonance": {
                "legacy": "half_magnetizing_C_rx_rx_F",
            },
        }
        self.hard_constraint_contract_sha256 = canonical_sha256(
            self.hard_constraint_contract
        )
        self.underlying_predict_calls: list[str] = []
        self.underlying_decode_calls = 0
        self._evaluate = self._base_evaluate

    def _unit_from_physical(self, name: str, value: float) -> float:
        del name
        return value / 100.0

    def repair_unit_coordinates(self, values: Any) -> Any:
        return np.asarray(values, dtype=float)

    def decode_batch(self, values: Any) -> tuple[Any, Any, Any]:
        self.underlying_decode_calls += 1
        coordinates = np.asarray(values, dtype=float)
        main = preflight._turn_split_main_values(
            coordinates,
            fixed_primary_turns=6,
            coordinate_index=2,
        )
        rows = [
            {
                "cw1": 5.0,
                "gap1": 1.6,
                "gap2": 1.9,
                "cw2": 0.8,
                "core_plate_t": 20.0,
                "wcp_t": 20.0,
                "N1_main": 6,
                "N1_side": 0,
                "N2_main": int(split),
                "N2_side": 60 - int(split),
            }
            for split in main
        ]
        frame = pd.DataFrame(rows)
        shrink = np.zeros(len(frame), dtype=float)
        valid = np.ones(len(frame), dtype=bool)
        self._last_decode = (frame, shrink, valid)
        return frame, shrink, valid

    def _predict(self, target: str, frame: Any) -> tuple[Any, Any]:
        self.underlying_predict_calls.append(target)
        if target == "Llt_phys":
            main = frame["N2_main"].to_numpy(dtype=float)
            return (
                20.0 + np.abs(main - self.llt_optimum_main_turns),
                np.full(len(frame), 0.25, dtype=float),
            )
        return (
            np.full(len(frame), 9.9e9, dtype=float),
            np.zeros(len(frame), dtype=float),
        )

    def _base_evaluate(
        self,
        values: Any,
        out: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        del args, kwargs
        frame, _shrink, valid = self.decode_batch(values)
        # Exercise the exact legacy predictor requests.  The corrected wrapper
        # must intercept both capacitance requests without calling this
        # problem's authenticated raw-capacitance surrogate.
        self._predict("C_rx_rx_F", frame)
        self._predict("C_tx_tx_F", frame)
        self._predict("Llt_phys", frame)
        g = np.zeros((len(values), self.n_ieq_constr), dtype=float)
        g[:, self.constraint_index[preflight.RESONANCE_MINIMUM_CONSTRAINT]] = (
            -123.0
        )
        out["F"] = np.ones((len(values), 2), dtype=float)
        out["G"] = g
        out["frame"] = frame
        out["decoder_valid"] = valid


def test_installed_evaluator_excludes_raw_predictor_and_replaces_G(
    profile: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_f = 16_250.0
    monkeypatch.setattr(
        lane,
        "_candidate_prediction",
        lambda _row, _model: {
            "physics_delta_Crx_mean_F": 4.0e-10,
            "physics_delta_Crx_q90_ucb_F": 4.8e-10,
            "physics_delta_fRx_q90_lcb_Hz": expected_f,
            "physics_delta_extrapolation_distance": 1.25,
        },
    )
    problem = _FakeProblem()
    evidence = lane._install_search_profile(problem, profile)
    out: dict[str, Any] = {}
    original = np.zeros((1, problem.n_var))
    original[0, 2] = preflight._turn_split_unit_coordinate(
        37, fixed_primary_turns=6
    )
    problem._evaluate(original, out)

    assert problem.underlying_predict_calls == ["Llt_phys", "Llt_phys"]
    assert problem.underlying_decode_calls == 1
    assert preflight.RESONANCE_MINIMUM_CONSTRAINT not in problem.constraint_names
    assert lane.PHYSICS_CONSTRAINT_NAME in problem.constraint_names
    physics_index = problem.constraint_index[lane.PHYSICS_CONSTRAINT_NAME]
    assert out["G"][0, physics_index] == pytest.approx(15_000.0 - expected_f)
    assert out["physics_delta_Crx_q90_ucb_F"][0] == 4.8e-10
    assert out["physics_delta_extrapolation_distance"][0] == 1.25
    assert int(out["frame"].iloc[0]["N2_main"]) == 30
    assert (
        out["frame"].iloc[0]["split_repair_original_N2_main"] == 37
    )
    assert (
        out["frame"].iloc[0]["split_repair_selected_N2_main"] == 30
    )
    assert (
        out["frame"].iloc[0]["split_repair_selected_robust_Llt_G"]
        < out["frame"].iloc[0]["split_repair_original_robust_Llt_G"]
    )
    assert (
        out["frame"].iloc[0]["split_repair_neighbor_lower_N2_main"]
        == 29
    )
    assert (
        out["frame"].iloc[0]["split_repair_neighbor_upper_N2_main"]
        == 31
    )
    assert (
        out["raw_two_net_capacitance_predictor_authority_used"].tolist()
        == [False]
    )
    assert out["single_0p759701_transfer_ratio_used"].tolist() == [False]
    assert evidence["raw_C_rx_rx_F_UCB_gate_installed"] is False
    assert evidence["single_0p759701_transfer_ratio_installed"] is False
    assert evidence["legacy_half_magnetizing_resonance_G_installed"] is False
    assert (
        evidence["turn_split_local_repair_installation"][
            "enumerated_split_count_per_geometry"
        ]
        == 48
    )
    assert problem.hard_constraint_contract[
        "raw_two_net_capacitance_G_present"
    ] is False
    assert problem.hard_constraint_contract[
        "legacy_half_magnetizing_resonance_G_present"
    ] is False


def test_split_repair_excludes_unevaluable_zero_side_endpoint(
    profile: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        lane,
        "_candidate_prediction",
        lambda _row, _model: {
            "physics_delta_Crx_mean_F": 4.0e-10,
            "physics_delta_Crx_q90_ucb_F": 4.8e-10,
            "physics_delta_fRx_q90_lcb_Hz": 16_250.0,
            "physics_delta_extrapolation_distance": 1.25,
        },
    )
    problem = _FakeProblem(llt_optimum_main_turns=60)
    lane._install_search_profile(problem, profile)
    coordinates = np.zeros((2, problem.n_var), dtype=float)
    coordinates[:, 2] = preflight._turn_split_unit_coordinate(
        60, fixed_primary_turns=6
    )
    out: dict[str, Any] = {}
    problem._evaluate(coordinates, out)

    assert out["frame"]["N2_main"].astype(int).tolist() == [59, 59]
    assert out["frame"]["N2_side"].astype(int).tolist() == [1, 1]
    assert (
        out["frame"]["split_repair_enumerated_split_count"]
        .astype(int)
        .tolist()
        == [48, 48]
    )
    assert np.isfinite(out["F"]).all()
    assert np.isfinite(out["G"]).all()
    assert all(len(value) == 2 for value in out.values())
    assert (
        out["raw_two_net_capacitance_predictor_authority_used"].tolist()
        == [False, False]
    )
    assert out["single_0p759701_transfer_ratio_used"].tolist() == [
        False,
        False,
    ]


def test_plain_decode_bypasses_expensive_split_enumeration(
    profile: dict[str, Any],
) -> None:
    problem = _FakeProblem()
    lane._install_search_profile(problem, profile)
    coordinates = np.zeros((3, problem.n_var), dtype=float)
    frame, shrink, valid = problem.decode_batch(coordinates)

    assert len(frame) == 3
    assert shrink.shape == (3,)
    assert valid.tolist() == [True, True, True]
    assert problem.underlying_decode_calls == 1
    assert problem.underlying_predict_calls == []


def test_raw_Crx_cannot_quarantine_terminal_candidate() -> None:
    targets = (
        set(preflight.TERMINAL_NONNEGATIVE_SURROGATE_TARGETS)
        | set(preflight.TERMINAL_POSITIVE_SURROGATE_TARGETS)
        | {"k"}
    )
    predictions = {
        target: {
            "mean": np.ones(2, dtype=float),
            "q90_conformal_half_width": np.zeros(2, dtype=float),
        }
        for target in targets
    }
    predictions["k"]["mean"][:] = 0.9
    predictions["C_rx_rx_F"]["mean"][:] = -1.0
    valid, evidence = lane.corrected_terminal_physicality_gate(
        predictions,
        population_size=2,
    )
    assert valid.tolist() == [True, True]
    assert evidence["excluded_non_authoritative_targets"] == ["C_rx_rx_F"]
    assert evidence["C_rx_rx_F_terminal_eligibility_authority"] is False
    assert "C_rx_rx_F" not in evidence["strictly_positive_targets"]
    assert "C_rx_rx_F" not in evidence["nonnegative_targets"]


def test_transport_contract_has_no_legacy_acquisition_authority(
    profile: dict[str, Any],
) -> None:
    execution = lane.offload._physics_delta_execution_contract(profile)
    assert execution is not None
    assert execution["physics_delta_Crx_q90_ucb_gate_active"] is True
    assert execution["physics_delta_fRx_q90_lcb_gate_active"] is True
    assert execution["raw_same_metric_C_rx_rx_F_UCB_gate_active"] is False
    assert (
        execution["raw_two_net_C_optimizer_objective_constraint_authority"]
        is False
    )
    assert execution["raw_two_net_C_terminal_eligibility_authority"] is False
    assert execution["single_0p759701_transfer_ratio_used"] is False
    assert execution["legacy_half_magnetizing_resonance_G_present"] is False
    assert "authenticated_turn_graded_transfer_ratio" not in execution
    assert "provisional_turn_graded_C_acquisition_gate_active" not in execution
    assert execution["fixed20T_turn_graded_FEA_retraining_required"] is True


def test_scheduler_payload_uses_isolated_worker_and_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        lane,
        "_BASE_SCHEDULER_PAYLOAD",
        lambda **_kwargs: {
            "name": lane.TASK_NAME_PREFIX + "abc-s2607264400-n1-6",
            "dedupe_key": lane.DEDUPE_PREFIX + "abc",
            "command": (
                "set -euo pipefail\n"
                "exec python artifacts/code/tools/"
                "mft_goal_diagnostic_compact_scout.py execute "
                "--payload \"$payload_path\""
            ),
        },
    )
    payload = lane._corrected_scheduler_payload(
        plan={},
        task={},
        priority=100,
    )
    assert lane.WORKER_ENTRYPOINT in payload["command"]
    assert "mft_goal_diagnostic_compact_scout.py execute" not in payload[
        "command"
    ]
    assert payload["name"].startswith(lane.TASK_NAME_PREFIX)
    assert payload["dedupe_key"].startswith(lane.DEDUPE_PREFIX)
    lines = payload["command"].splitlines()
    assert lines[:4] == [
        "set -euo pipefail",
        "export OMP_NUM_THREADS=1",
        "export OPENBLAS_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
    ]
    assert payload["command"].count("OMP_NUM_THREADS") == 1
    assert payload["command"].count("OPENBLAS_NUM_THREADS") == 1
    assert payload["command"].count("MKL_NUM_THREADS") == 1


def test_model_file_sha_is_exact() -> None:
    assert lane._sha256_file(lane.DEFAULT_MODEL_PATH) == lane.MODEL_FILE_SHA256
    assert (
        lane._load_bound_model(lane.DEFAULT_MODEL_PATH)["payload_sha256"]
        == lane.MODEL_PAYLOAD_SHA256
    )
