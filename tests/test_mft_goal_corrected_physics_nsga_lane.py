from __future__ import annotations

import copy
import math
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
    assert validated["cooling_or_TIM_contract_mutated"] is False


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
    def __init__(self) -> None:
        self.geometry_constraint_profile_sha256 = scout._geometry_profile(
            20.0
        )["sha256"]
        self.spec = {
            "size_limits_mm": {"W": 1200.0, "L": 1000.0, "H": 750.0}
        }
        self.temperature_limits_C = {}
        self.temperature_contract = {}
        self.temperature_contract_sha256 = "0" * 64
        self.sobol_dimension_names = (
            "cw1",
            "gap1",
            "gap2",
            "core_plate_t",
            "wcp_t",
        )
        self.cw1_coordinate_index = 0
        self.xl = np.zeros(5, dtype=float)
        self.xu = np.ones(5, dtype=float)
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
        self._evaluate = self._base_evaluate

    def _unit_from_physical(self, name: str, value: float) -> float:
        del name
        return value / 100.0

    def _predict(self, target: str, frame: Any) -> tuple[Any, Any]:
        self.underlying_predict_calls.append(target)
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
        row = {
            "cw1": 5.0,
            "gap1": 1.6,
            "gap2": 1.9,
            "cw2": 0.8,
            "core_plate_t": 20.0,
            "wcp_t": 20.0,
            "N1_main": 6,
            "N1_side": 0,
            "N2_main": 37,
            "N2_side": 23,
        }
        frame = pd.DataFrame([row for _ in range(len(values))])
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
        out["decoder_valid"] = np.ones(len(values), dtype=bool)


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
    problem._evaluate(np.zeros((1, 5)), out)

    assert problem.underlying_predict_calls == ["Llt_phys"]
    assert preflight.RESONANCE_MINIMUM_CONSTRAINT not in problem.constraint_names
    assert lane.PHYSICS_CONSTRAINT_NAME in problem.constraint_names
    physics_index = problem.constraint_index[lane.PHYSICS_CONSTRAINT_NAME]
    assert out["G"][0, physics_index] == pytest.approx(15_000.0 - expected_f)
    assert out["physics_delta_Crx_q90_ucb_F"][0] == 4.8e-10
    assert out["physics_delta_extrapolation_distance"][0] == 1.25
    assert out["raw_two_net_capacitance_predictor_authority_used"] is False
    assert out["single_0p759701_transfer_ratio_used"] is False
    assert evidence["raw_C_rx_rx_F_UCB_gate_installed"] is False
    assert evidence["single_0p759701_transfer_ratio_installed"] is False
    assert evidence["legacy_half_magnetizing_resonance_G_installed"] is False
    assert problem.hard_constraint_contract[
        "raw_two_net_capacitance_G_present"
    ] is False
    assert problem.hard_constraint_contract[
        "legacy_half_magnetizing_resonance_G_present"
    ] is False


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


def test_model_file_sha_is_exact() -> None:
    assert lane._sha256_file(lane.DEFAULT_MODEL_PATH) == lane.MODEL_FILE_SHA256
    assert (
        lane._load_bound_model(lane.DEFAULT_MODEL_PATH)["payload_sha256"]
        == lane.MODEL_PAYLOAD_SHA256
    )
