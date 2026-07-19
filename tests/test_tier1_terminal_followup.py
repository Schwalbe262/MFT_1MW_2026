from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from tools.tier1_resonance_feedback import (
    EXPECTED_NSGA_REVISION,
    HARD_SPEC,
    SEARCH_SCHEMA,
)
from tools.tier1_terminal_followup import _measurement_metadata, analyze_terminal


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _params() -> dict:
    return {
        "N1": 8, "N2": 80, "N1_main": 8, "N1_side": 0,
        "N2_main": 80, "N2_side": 0,
        "l1": 70.0, "l2": 360.0, "h1": 610.0, "w1": 315.0,
        "n_core_group": 3, "core_plate_t": 20.0,
        "core_plate_pad_t": 2.0, "core_depth_min": 60.0,
        "core_depth_max": 120.0, "cw1": 5.0, "gap1": 4.0,
        "cw2": 0.8, "gap2": 0.8, "nwh1": 550.0, "nwh2": 410.0,
        "wcp_t": 20.0, "wcp_len_pct": 40.0,
        "cc_w2c_space_x": 40.0, "cc_w2c_space_y": 40.0,
        "w2c_w1c_space_x": 40.0, "w2c_w1c_space_y": 40.0,
        "w1c_w2s_space_x": 40.5, "w1s_cs_space_x": 40.0,
        "cs_w1s_space_y": 40.0, "w2s_w1s_space_x": 40.0,
        "w1s_w2s_space_y": 40.0,
    }


def test_terminal_analysis_requires_matching_nonpositive_constraint_evidence():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        seed = 101
        seed_dir = root / f"seed-{seed}"
        params = _params()
        digest = hashlib.sha256(json.dumps(
            params, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode()).hexdigest()
        constraint = {
            "Llt_robust_band": -0.1,
            "analytical_flux_density_limit": -0.1,
            "decoded_space_shrink": 0.0,
            "minimum_physical_insulation": 0.0,
            "core_group_manufacturability_limit": -1.0,
            "exterior_width_limit": 0.0,
            "exterior_length_limit": -20.0,
            "exterior_height_limit": 0.0,
        }
        result = {
            "schema_version": SEARCH_SCHEMA,
            "seed": seed,
            "model_manifest_sha256": "a" * 64,
            "nsga_code_revision": EXPECTED_NSGA_REVISION,
            "hard_spec": HARD_SPEC,
            "completed_generations": 241,
            "constraint_minimum_G": constraint,
            "feasible_pareto_count": 1,
            "candidates": [{
                "index": 0, "volume_L": 400.0, "total_loss_W": 5000.0,
                "decoded_params": params, "predictions": {},
                "conformal_half_width": {}, "derived_resonance": {},
            }],
            "next_target_fea_batch_plan": {
                "candidates": [{
                    "decoded_params": params,
                    "decoded_params_sha256": digest,
                    "constraint_G": constraint,
                    "target_predictions": {"Llt_phys": 27.5},
                    "target_conformal_half_width": {"Llt_phys": 0.4},
                    "derived_resonance": {"f_res_min_screen_Hz": 20_100.0},
                }],
                "submission_performed": False,
                "current_candidates_eligible_for_submission": False,
                "production_eligible": False,
                "fea_submission_approved": False,
            },
            "production_eligible": False,
            "fea_submission_approved": False,
            "automatic_promotion_allowed": False,
        }
        result_path = seed_dir / "result.json"
        _write(result_path, result)
        launch_path = root / "launch.json"
        launch = {
            "model_manifest_sha256": "a" * 64,
            "nsga_code_revision": EXPECTED_NSGA_REVISION,
            "population": 160,
            "max_generations": 240,
            "jobs": [{"seed": seed, "output": str(seed_dir)}],
        }
        _write(launch_path, launch)
        status_path = root / "status.json"
        _write(status_path, {
            "launch_sha256": _sha(launch_path),
            "model_manifest_sha256": "a" * 64,
            "nsga_code_revision": EXPECTED_NSGA_REVISION,
            "jobs": [{
                "seed": seed, "state": "completed",
                "result_sha256": _sha(result_path),
            }],
        })
        defaults = {
            "core_plate_pad_t": 2.0,
            "core_depth_min": 60.0,
            "core_depth_max": 120.0,
        }
        with patch(
            "tools.tier1_terminal_followup._load_sobol_schema",
            return_value=((('u_N1', 0.0, 1.0),), 5, 8, defaults),
        ):
            analysis = analyze_terminal(
                launch_path=launch_path,
                status_path=status_path,
                nsga_code_root=root,
                output=root / "followup",
            )
        assert analysis["fully_valid_unique_count"] == 1
        assert analysis["first_canary_candidate"]["canary_eligible"] is True
        assert analysis["active_learning_measurement_unique_count"] == 1
        assert analysis[
            "first_active_learning_measurement_candidate"
        ]["measurement_eligible"] is True
        assert analysis[
            "production_validity_and_measurement_eligibility_are_separate"
        ] is True
        assert analysis["canary_submission_performed"] is False
        assert analysis["production_eligible"] is False
        warm = np.load(root / "followup" / "next_warm_start.npy")
        assert warm.shape == (1, 1)


def test_measurement_eligibility_does_not_imply_production_feasibility():
    constraints = {
        "analytical_flux_density_limit": -0.1,
        "decoded_space_shrink": 0.0,
        "minimum_physical_insulation": 0.0,
        "core_group_manufacturability_limit": -1.0,
        "exterior_width_limit": 0.0,
        "exterior_length_limit": -20.0,
        "exterior_height_limit": 0.0,
        "Llt_robust_band": 7.5,
        "temperature_robust_limit:T_max_Tx": 170.0,
        "half_magnetizing_resonance_minimum": -10.0,
        "strict_full_density_support": 0.4,
    }
    metadata = _measurement_metadata({
        "constraint_G": constraints,
        "target_predictions": {"Llt_phys": 32.5},
        "target_conformal_half_width": {"Llt_phys": 3.2},
        "derived_resonance": {"f_res_min_screen_Hz": 20_010.0},
    })
    assert metadata["measurement_eligible"] is True
    assert metadata["measurement_geometry_passed"] is True
    assert constraints["Llt_robust_band"] > 0
    assert constraints["temperature_robust_limit:T_max_Tx"] > 0
