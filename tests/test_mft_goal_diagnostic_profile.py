from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from module.mft_goal_20260726_contract import canonical_sha256
from tools import mft_goal_20260726_launch as launch
from tools import mft_goal_diagnostic_compact_scout as scout
from tools import mft_goal_diagnostic_compact_slurm as offload


def _profile(clearance: int = 20) -> dict:
    geometry = scout._geometry_profile(clearance)
    start = scout.PROFILE_SEED_STARTS[clearance]
    effective = scout._effective_constraint_profile()
    value = {
        "schema_version": scout.SEARCH_PROFILE_SCHEMA,
        "profile_id": (
            f"l900-temp110-130-130-hard-equal-hgap{clearance}-"
            "gap2p35-plates20-exact100"
        ),
        "fixed_primary_turns": 6,
        "fixed_secondary_turns": 60,
        "turns_ratio_N2_over_N1": 10.0,
        "fixed_primary_conductor_thickness_mm": 5.0,
        "fixed_primary_interturn_gap_mm": 1.6,
        "fixed_secondary_interturn_gap_mm": 0.35,
        "fixed_core_plate_thickness_mm": 20.0,
        "fixed_winding_cold_plate_thickness_mm": 20.0,
        "effective_constraint_profile": effective,
        "effective_constraint_profile_payload_sha256": effective[
            "payload_sha256"
        ],
        "geometry_constraint_profile": geometry,
        "geometry_constraint_profile_sha256": geometry["sha256"],
        "winding_height_exact_equality_initialization_and_repair_required": (
            True
        ),
        "maximum_decoded_winding_height_difference_mm": 0.1,
        "authorized_seed_start": start,
        "authorized_seed_count": 100,
        "authorized_seed_end_inclusive": start + 99,
        "raw_same_metric_capacitance_gate": {
            "constraint_name": scout.RAW_CRX_CONSTRAINT_NAME,
            "target": "C_rx_rx_F",
            "measurement_identity": (
                "authenticated_strict_full_raw_two_net_CapMatrix_C_rx_rx_F"
            ),
            "turn_graded_or_corrected_metric_used": False,
            "statistic": "surrogate_mu_plus_q90_conformal_half_width",
            "maximum_F": scout.RAW_CRX_UCB_MAXIMUM_F,
            "optimizer_hard_gate_active": True,
            "constraint_value": "UCB_F/maximum_F-1",
            "predictor_conformal_argument": True,
            "model_artifact_sha256": "1" * 64,
            "model_metadata_sha256": "2" * 64,
            "dataset_sha256": "3" * 64,
            "calibration_recovery_status": "applied",
            "calibration_row_count": 6151,
        },
        "fixed_lm2mh_resonance_contract_sha256": (
            scout.preflight.goal_fixed_lm2mh_resonance_contract()["sha256"]
        ),
        "temperature_contract_sha256": effective[
            "temperature_contract_sha256"
        ],
        "cooling_or_TIM_contract_mutated": False,
        "screening_only": True,
        "production_eligible": False,
        "final_design_claim_allowed": False,
    }
    return launch._seal(value)


def _bounded_profile() -> dict:
    class Authenticated:
        report = {
            "artifacts": {
                "C_rx_rx_F/models.pkl": "1" * 64,
                "C_rx_rx_F/meta.json": "2" * 64,
            },
            "capacitance_recovery": {
                "status": scout.adapter.RECOVERY_STATUS,
                "eligible_row_count": 6151,
                "row_count": 6151,
            },
        }
        evidence = {"dataset": {"sha256": "3" * 64}}

    class Runner:
        authenticated = Authenticated()

    return scout._build_search_profile(
        Runner(),
        geometry_profile=scout._geometry_profile(20),
        authorized_seed_start=2_607_264_300,
        secondary_gap_mode=scout.SECONDARY_GAP_MODE_BOUNDED,
    )


def test_geometry_profile_sha_and_seed_authority_are_exact() -> None:
    for clearance in (20, 30, 40):
        profile = _profile(clearance)
        alignment = profile["geometry_constraint_profile"][
            "winding_height_alignment"
        ]
        assert alignment["mode"] == "hard"
        assert alignment["minimum_overlap_ratio"] == 1.0
        assert scout._validate_search_profile(profile) == profile
        activation = {
            "manufacturing_search_profile": profile,
            "manufacturing_search_profile_payload_sha256": profile[
                "payload_sha256"
            ],
        }
        seeds = offload._authorized_activation_seeds(activation)
        assert seeds == tuple(
            range(
                scout.PROFILE_SEED_STARTS[clearance],
                scout.PROFILE_SEED_STARTS[clearance] + 100,
            )
        )


def test_search_profile_rejects_secondary_gap_drift() -> None:
    profile = _profile(20)
    unsigned = {
        key: copy.deepcopy(value)
        for key, value in profile.items()
        if key != "payload_sha256"
    }
    unsigned["fixed_secondary_interturn_gap_mm"] = 0.351
    with pytest.raises(RuntimeError, match="search profile mismatch"):
        scout._validate_search_profile(launch._seal(unsigned))


def test_bounded_secondary_profile_is_disjoint_and_strict() -> None:
    profile = _bounded_profile()
    assert scout._validate_search_profile(profile) == profile
    assert profile["authorized_seed_start"] == 2_607_264_300
    assert profile["authorized_seed_end_inclusive"] == 2_607_264_399
    assert profile["scheduler_priority"] == 100
    assert profile["secondary_interturn_gap_search_mm"] == {
        "minimum": 0.35,
        "maximum": 2.0,
        "step": 0.001,
        "decoder_native_minimum": 0.3,
        "decoder_native_maximum": 2.0,
        "realized_band_constraint_name": (
            scout.SECONDARY_GAP_BAND_CONSTRAINT_NAME
        ),
    }
    assert profile["secondary_conductor_thickness_search_mm"] == {
        "minimum": 0.3,
        "maximum": 1.0,
        "hard_constraint_name": (
            scout.SECONDARY_CONDUCTOR_BAND_CONSTRAINT_NAME
        ),
    }
    assert "fixed_secondary_interturn_gap_mm" not in profile


def test_search_profile_accepts_only_aligned_hgap20_continuations() -> None:
    profile = _profile(20)
    unsigned = {
        key: copy.deepcopy(value)
        for key, value in profile.items()
        if key != "payload_sha256"
    }
    start = scout.CONTINUATION_SEED_START
    unsigned["profile_id"] = scout._search_profile_id(20, start)
    unsigned["authorized_seed_start"] = start
    unsigned["authorized_seed_end_inclusive"] = start + 99
    continuation = launch._seal(unsigned)
    assert scout._validate_search_profile(continuation) == continuation

    for invalid in (start + 1, start - 1):
        changed = copy.deepcopy(unsigned)
        changed["profile_id"] = scout._search_profile_id(20, invalid)
        changed["authorized_seed_start"] = invalid
        changed["authorized_seed_end_inclusive"] = invalid + 99
        with pytest.raises(RuntimeError, match="search profile mismatch"):
            scout._validate_search_profile(launch._seal(changed))

    with pytest.raises(RuntimeError, match="20 mm aligned exact100"):
        scout._authorized_profile_seed_start(30, start + 200)


def test_l900_compact_contract_is_bound_to_effective_profile() -> None:
    effective = scout._effective_constraint_profile()
    contract = (
        scout.preflight.goal_diagnostic_l900_compact_search_contract(
            6,
            effective_constraint_profile_sha256=effective["payload_sha256"],
        )
    )
    assert contract["hard_size_limits_mm"] == {
        "W": 1200.0,
        "L": 900.0,
        "H": 750.0,
    }
    assert max(
        bounds["L_mm"][1]
        for name, bounds in contract["strata"].items()
        if name != "height_boundary"
    ) == 900.0
    assert (
        scout.preflight.validate_goal_compact_search_contract(
            contract, fixed_primary_turns=6
        )
        == contract
    )


def test_profile_installer_fixes_controls_and_appends_raw_crx_ucb_gate() -> None:
    profile = _profile(20)

    class Problem:
        geometry_constraint_profile_sha256 = profile[
            "geometry_constraint_profile_sha256"
        ]
        sobol_dimension_names = (
            "gap1",
            "gap2",
            "f1_split",
            "core_plate_t",
            "wcp_t",
        )
        cw1_coordinate_index = 2
        xl = np.zeros(5)
        xu = np.ones(5)
        constraint_names = ("base",)
        constraint_index = {"base": 0}
        n_ieq_constr = 1
        hard_constraint_contract = {"schema_version": "fixture-v1"}
        hard_constraint_contract_sha256 = canonical_sha256(
            hard_constraint_contract
        )
        spec = {
            "size_limits_mm": dict(
                scout.GOAL_STAGE_SPEC["size_limits_mm"]
            ),
            "temperature_family_limits_C": dict(
                scout.GOAL_STAGE_SPEC["temperature_family_limits_C"]
            ),
            "temperature_target_limits_C": dict(
                scout.GOAL_STAGE_SPEC["temperature_target_limits_C"]
            ),
        }
        _prediction_cache = None

        @staticmethod
        def _unit_from_physical(name, value):
            expected = {
                "gap1": 1.6,
                "gap2": 0.35,
                "core_plate_t": 20.0,
                "wcp_t": 20.0,
            }
            assert value == expected[name]
            return {
                "gap1": 0.25,
                "gap2": 0.35,
                "core_plate_t": 0.5,
                "wcp_t": 0.75,
            }[name]

        @staticmethod
        def _predict(target, frame):
            assert target == "C_rx_rx_F"
            return (
                np.full(len(frame), 4.0e-10),
                np.full(len(frame), 1.0e-10),
            )

        def _evaluate(self, values, out, *args, **kwargs):
            del args, kwargs
            count = len(values)
            out["G"] = np.zeros((count, 1))
            out["decoder_valid"] = np.ones(count, dtype=bool)
            out["frame"] = pd.DataFrame(
                {
                        "cw1": np.full(count, 5.0),
                        "cw2": np.full(count, 0.8),
                        "gap1": np.full(count, 1.6),
                    "gap2": np.full(count, 0.35),
                    "core_plate_t": np.full(count, 20.0),
                    "wcp_t": np.full(count, 20.0),
                    "N1_main": np.full(count, 6),
                    "N1_side": np.zeros(count),
                    "N2_main": np.full(count, 37),
                    "N2_side": np.full(count, 23),
                }
            )

    problem = Problem()
    evidence = scout._install_search_profile(problem, profile)
    out = {}
    problem._evaluate(np.zeros((2, 5)), out)
    assert evidence["raw_C_rx_rx_F_UCB_gate_installed"] is True
    assert problem.constraint_names[-1] == scout.RAW_CRX_CONSTRAINT_NAME
    assert out["G"].shape == (2, 2)
    assert np.all(out["G"][:, -1] < 0.0)
    assert np.all(problem.xl == problem.xu)
    assert evidence["gap2_coordinate_index"] == 1
    assert evidence["gap2_coordinate"] == 0.35


def test_bounded_aligned_bank_proof_attests_realized_secondary_bands() -> None:
    profile = _bounded_profile()

    class Problem:
        @staticmethod
        def repair_unit_coordinates(values):
            return np.asarray(values, dtype=float)

        @staticmethod
        def decode_batch(values):
            count = len(values)
            return (
                pd.DataFrame(
                    {
                        "nwh1": np.full(count, 600.0),
                        "nwh2": np.full(count, 600.0),
                        "h_gap1": np.full(count, 20.0),
                        "cw1": np.full(count, 5.0),
                        "cw2": np.linspace(0.3, 1.0, count),
                        "gap1": np.full(count, 1.6),
                        "gap2": np.linspace(0.35, 2.0, count),
                        "core_plate_t": np.full(count, 20.0),
                        "wcp_t": np.full(count, 20.0),
                        "N1_main": np.full(count, 6),
                        "N1_side": np.zeros(count),
                        "N2_main": np.full(count, 37),
                        "N2_side": np.full(count, 23),
                    }
                ),
                np.ones(count),
                np.ones(count, dtype=bool),
            )

        @staticmethod
        def _goal_bounding_box_lit(_row):
            return 1.0, (1100.0, 890.0, 700.0)

    coordinates = np.zeros((2, 5))
    proof = scout._aligned_bank_proof(
        Problem(),
        {"coordinates": coordinates.tolist(), "sha256": "4" * 64},
        profile,
    )
    assert proof["secondary_gap_mode"] == scout.SECONDARY_GAP_MODE_BOUNDED
    assert proof["minimum_realized_secondary_interturn_gap_mm"] == 0.35
    assert proof["maximum_realized_secondary_interturn_gap_mm"] == 2.0
    assert proof["minimum_realized_secondary_conductor_thickness_mm"] == 0.3
    assert proof["maximum_realized_secondary_conductor_thickness_mm"] == 1.0
    assert proof["all_rows_gap2_within_allowed_band"] is True
    assert proof["all_rows_cw2_within_allowed_band"] is True


def test_bounded_aligned_bank_proof_rejects_cw2_escape() -> None:
    profile = _bounded_profile()

    class Problem:
        @staticmethod
        def repair_unit_coordinates(values):
            return np.asarray(values, dtype=float)

        @staticmethod
        def decode_batch(values):
            count = len(values)
            return (
                pd.DataFrame(
                    {
                        "nwh1": np.full(count, 600.0),
                        "nwh2": np.full(count, 600.0),
                        "h_gap1": np.full(count, 20.0),
                        "cw1": np.full(count, 5.0),
                        "cw2": np.full(count, 1.001),
                        "gap1": np.full(count, 1.6),
                        "gap2": np.full(count, 0.35),
                        "core_plate_t": np.full(count, 20.0),
                        "wcp_t": np.full(count, 20.0),
                        "N1_main": np.full(count, 6),
                        "N1_side": np.zeros(count),
                        "N2_main": np.full(count, 37),
                        "N2_side": np.full(count, 23),
                    }
                ),
                np.ones(count),
                np.ones(count, dtype=bool),
            )

        @staticmethod
        def _goal_bounding_box_lit(_row):
            return 1.0, (1100.0, 890.0, 700.0)

    with pytest.raises(RuntimeError, match="bank profile proof failed"):
        scout._aligned_bank_proof(
            Problem(),
            {"coordinates": [[0.0] * 5], "sha256": "4" * 64},
            profile,
        )
