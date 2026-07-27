from __future__ import annotations

import numpy as np
import pandas as pd

from module.mft_goal_20260726_contract import canonical_sha256
from tools import mft_goal_20260726_launch as launch
from tools import mft_goal_diagnostic_compact_scout as scout
from tools import mft_goal_diagnostic_compact_slurm as offload


def _profile(clearance: int = 20) -> dict:
    geometry = scout._geometry_profile(clearance)
    start = scout.PROFILE_SEED_STARTS[clearance]
    value = {
        "schema_version": scout.SEARCH_PROFILE_SCHEMA,
        "profile_id": f"hard-equal-hgap{clearance}-plates20-exact32",
        "fixed_primary_turns": 6,
        "fixed_secondary_turns": 60,
        "turns_ratio_N2_over_N1": 10.0,
        "fixed_primary_conductor_thickness_mm": 5.0,
        "fixed_primary_interturn_gap_mm": 1.6,
        "fixed_core_plate_thickness_mm": 20.0,
        "fixed_winding_cold_plate_thickness_mm": 20.0,
        "geometry_constraint_profile": geometry,
        "geometry_constraint_profile_sha256": geometry["sha256"],
        "winding_height_exact_equality_initialization_and_repair_required": (
            True
        ),
        "maximum_decoded_winding_height_difference_mm": 0.1,
        "authorized_seed_start": start,
        "authorized_seed_count": 32,
        "authorized_seed_end_inclusive": start + 31,
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
        "temperature_contract_sha256": (
            scout.GOAL_TEMPERATURE_CONTRACT_SHA256
        ),
        "cooling_or_TIM_contract_mutated": False,
        "screening_only": True,
        "production_eligible": False,
        "final_design_claim_allowed": False,
    }
    return launch._seal(value)


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
                scout.PROFILE_SEED_STARTS[clearance] + 32,
            )
        )


def test_profile_installer_fixes_controls_and_appends_raw_crx_ucb_gate() -> None:
    profile = _profile(20)

    class Problem:
        geometry_constraint_profile_sha256 = profile[
            "geometry_constraint_profile_sha256"
        ]
        sobol_dimension_names = (
            "gap1",
            "f1_split",
            "core_plate_t",
            "wcp_t",
        )
        cw1_coordinate_index = 1
        xl = np.zeros(4)
        xu = np.ones(4)
        constraint_names = ("base",)
        constraint_index = {"base": 0}
        n_ieq_constr = 1
        hard_constraint_contract = {"schema_version": "fixture-v1"}
        hard_constraint_contract_sha256 = canonical_sha256(
            hard_constraint_contract
        )
        _prediction_cache = None

        @staticmethod
        def _unit_from_physical(name, value):
            expected = {
                "gap1": 1.6,
                "core_plate_t": 20.0,
                "wcp_t": 20.0,
            }
            assert value == expected[name]
            return {
                "gap1": 0.25,
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
                    "gap1": np.full(count, 1.6),
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
    problem._evaluate(np.zeros((2, 4)), out)
    assert evidence["raw_C_rx_rx_F_UCB_gate_installed"] is True
    assert problem.constraint_names[-1] == scout.RAW_CRX_CONSTRAINT_NAME
    assert out["G"].shape == (2, 2)
    assert np.all(out["G"][:, -1] < 0.0)
    assert np.all(problem.xl == problem.xu)
