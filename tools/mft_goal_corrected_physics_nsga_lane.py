"""Prepare and operate the isolated physics-delta corrected NSGA lane.

This lane deliberately reuses the authenticated compact NSGA transport and
thermal/geometry evaluator while replacing its legacy resonance acquisition
slot.  The replacement is the calibrated turn-voltage energy-network q90 UCB
and

    f_rx_lcb = 1 / (2*pi*sqrt(0.2*C_rx_ucb)).

The legacy two-net ``C_rx_rx_F`` surrogate, the historical 0.759701 transfer
ratio, and the half-magnetizing resonance screen have no optimizer objective,
constraint, or terminal-eligibility authority in this lane.  Results remain
screening-only until non-rounded turn-graded symmetric FEA and approved
dielectric-stack sensitivity close the physical gate.

Scheduler mutation is possible only through ``stage --apply`` or
``submit --apply``.  ``prepare``, ``plan``, ``preflight`` and ``dry-run`` are
local/read-only with respect to Scheduler state.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from module.mft_goal_20260726_contract import canonical_sha256  # noqa: E402
from tools import mft_goal_20260726_launch as goal_launch  # noqa: E402
from tools import mft_goal_diagnostic_compact_scout as scout  # noqa: E402
from tools import mft_goal_diagnostic_compact_slurm as offload  # noqa: E402
from tools import mft_goal_turn_graded_physics_reranker as physics  # noqa: E402
from tools import tier1_corrected_generation_preflight as preflight  # noqa: E402


CAMPAIGN_ID = "mft-goal-corrected-physics-delta-nsga-n1-6"
ACTIVATION_SCHEMA = "mft-goal-corrected-physics-delta-activation-v1"
BUNDLE_SCHEMA = "mft-goal-corrected-physics-delta-bundle-v1"
TASK_SCHEMA = "mft-goal-corrected-physics-delta-task-v1"
SCHEDULER_SCHEMA = "mft-goal-corrected-physics-delta-scheduler-v1"
RESULT_SCHEMA = "mft-goal-corrected-physics-delta-result-v1"
SEARCH_PROFILE_SCHEMA = "mft-goal-corrected-physics-delta-profile-v1"
INSTALLATION_SCHEMA = "mft-goal-corrected-physics-delta-installation-v1"
RESONANCE_SCHEMA = "mft-goal-physics-delta-lm2mh-resonance-v1"
PREFLIGHT_SCHEMA = "mft-goal-corrected-physics-delta-preflight-v1"
LOCAL_DRY_RUN_SCHEMA = "mft-goal-corrected-physics-delta-local-dry-run-v1"
OFFLOAD_PLAN_SCHEMA = "mft-goal-corrected-physics-delta-offload-plan-v1"
OFFLOAD_DEPLOYMENT_SCHEMA = (
    "mft-goal-corrected-physics-delta-deployment-v1"
)
OFFLOAD_AUTH_SCHEMA = "mft-goal-corrected-physics-delta-offload-auth-v1"
OFFLOAD_DRY_RUN_SCHEMA = (
    "mft-goal-corrected-physics-delta-submission-dry-run-v1"
)
OFFLOAD_RECEIPT_SCHEMA = (
    "mft-goal-corrected-physics-delta-submission-receipt-v1"
)

MODEL_FILE_SHA256 = (
    "0a025986c8e9f8b0c1a5615045c62e79e879d0d3517eb3084c79e40614f7669f"
)
MODEL_PAYLOAD_SHA256 = (
    "c4a2ace66edb8e79d9a7867e82003245c6e35d44a7e6fdebc0c82d16ad06aa36"
)
DEFAULT_MODEL_PATH = (
    Path(r"C:\Users\peets\slurm_scheduler_runtime")
    / "mft_goal_20260726"
    / "physics_delta_reranker_v2"
    / "model.json"
)
DEFAULT_LOCAL_ROOT = (
    Path(r"C:\Users\peets\slurm_scheduler_runtime")
    / "mft_goal_20260726"
    / "corrected_physics_delta_nsga_offload"
)
DEFAULT_REMOTE_ROOT = (
    "/gpfs/tmp_cpu2/mft_goal_20260726/corrected_physics_delta_nsga"
)
DEFAULT_BUNDLE_ROOT = (
    Path(r"C:\Users\peets\slurm_scheduler_runtime")
    / "mft_goal_20260726"
    / "corrected_physics_delta_nsga_4400_4499"
)
DEFAULT_PREFLIGHT_PATH = DEFAULT_LOCAL_ROOT / "sealed_preflight.json"
DEFAULT_DRY_RUN_PATH = DEFAULT_LOCAL_ROOT / "sealed_local_dry_run.json"

SEED_START = 2_607_264_400
SEED_COUNT = 60
SEED_END = SEED_START + SEED_COUNT - 1
SCHEDULER_PRIORITY = 100
TASK_NAME_PREFIX = "mft-goal-physics-delta-nsga-"
DEDUPE_PREFIX = "mft-goal-20260726-physics-delta-nsga:"
WORKER_ENTRYPOINT = (
    "artifacts/code/tools/mft_goal_corrected_physics_nsga_lane.py"
)
PHYSICS_CONSTRAINT_NAME = "physics_delta_fRx_q90_lcb_minimum"
SPLIT_MIN_N2_MAIN = 12
SPLIT_MAX_N2_MAIN = 60
SPLIT_VALUES = tuple(range(SPLIT_MIN_N2_MAIN, SPLIT_MAX_N2_MAIN + 1))
L_PRIMARY_H = 0.002
L_SECONDARY_H = 0.2
RESONANCE_MIN_HZ = 15_000.0
FORBIDDEN_OPTIMIZER_INPUTS = (
    "C_rx_rx_F",
    "pred_C_rx_rx_F_fixed_lm2mh",
    "C_rx_rx_corrected_turn_graded_transfer_estimate_F",
    "0.7597014146941471",
)

_ACTIVE_MODEL: dict[str, Any] | None = None
_BASE_SCHEDULER_PAYLOAD = offload.scheduler_payload


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"JSON input is unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _seal_sha(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "sha256" in result:
        raise RuntimeError("value is already SHA sealed")
    result["sha256"] = canonical_sha256(result)
    return result


def _load_bound_model(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if _sha256_file(resolved) != MODEL_FILE_SHA256:
        raise RuntimeError("physics-delta model file SHA mismatch")
    model = physics._load_model(resolved)
    _validate_embedded_model(model)
    return model


def _validate_embedded_model(value: Mapping[str, Any]) -> dict[str, Any]:
    model = physics._validate_seal(dict(value), physics.MODEL_SCHEMA)
    network = model.get("physics_network") or {}
    metrics = model.get("calibration_metrics") or {}
    if (
        model.get("payload_sha256") != MODEL_PAYLOAD_SHA256
        or model.get("fixed_lm_primary_referred_H") != L_PRIMARY_H
        or model.get("fixed_L2_secondary_H") != L_SECONDARY_H
        or model.get("resonance_minimum_Hz") != RESONANCE_MIN_HZ
        or model.get("raw_two_net_capacitance_role")
        != "not_used_by_this_model"
        or model.get("single_0p759701_ratio_role")
        != "not_used_by_this_model"
        or network.get("raw_two_net_capacitance_used") is not False
        or network.get("single_truth_transfer_ratio_used") is not False
        or network.get("section_physical_multiplicity")
        != {"main": 1, "side": 2}
        or model.get("dielectric_material_basis") != "air_only"
        or model.get("approved_dielectric_stack_sensitivity_required")
        is not True
        or model.get("air_only_result_can_close_final_gate") is not False
        or model.get("physical_feasibility_authority") is not False
        or model.get("automatic_final_promotion_allowed") is not False
        or model.get("final_turn_graded_symmetric_FEA_required") is not True
        or not math.isclose(
            float(metrics.get("loo_q90_abs_log_error", -1.0)),
            0.03905041627147031,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise RuntimeError("physics-delta embedded model authority mismatch")
    return model


def _set_active_model(model: Mapping[str, Any]) -> None:
    global _ACTIVE_MODEL
    _ACTIVE_MODEL = _validate_embedded_model(model)


def _active_model() -> dict[str, Any]:
    if _ACTIVE_MODEL is None:
        raise RuntimeError("physics-delta model was not activated")
    return copy.deepcopy(_ACTIVE_MODEL)


def _calibration_domain(model: Mapping[str, Any]) -> dict[str, Any]:
    delta = model["delta_model"]
    names = list(delta["feature_names"])
    minima = dict(zip(names, delta["feature_minimum"]))
    maxima = dict(zip(names, delta["feature_maximum"]))
    return {
        "gap2_mm": [
            math.exp(float(minima["log_gap2_mm"])),
            math.exp(float(maxima["log_gap2_mm"])),
        ],
        "cw2_mm": [
            math.exp(float(minima["log_cw2_mm"])),
            math.exp(float(maxima["log_cw2_mm"])),
        ],
        "nwh2_mm": [
            math.exp(float(minima["log_nwh2_mm"])),
            math.exp(float(maxima["log_nwh2_mm"])),
        ],
        "calibration_row_count": int(model["calibration_row_count"]),
        "calibration_geometry_sha256": list(
            model["calibration_geometry_sha256"]
        ),
        "old_plate_thickness_domain_not_represented_as_delta_feature": True,
        "fixed_20T_geometry_requires_new_turn_graded_FEA_retraining": True,
    }


def _physics_gate(model: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "constraint_name": PHYSICS_CONSTRAINT_NAME,
        "target": "C_rx_rx_turn_graded_F",
        "model_schema_version": model["schema_version"],
        "model_file_sha256": MODEL_FILE_SHA256,
        "model_payload_sha256": model["payload_sha256"],
        "model": copy.deepcopy(dict(model)),
        "method": model["method"],
        "q90_abs_log_error": model["calibration_metrics"][
            "loo_q90_abs_log_error"
        ],
        "capacitance_ucb_formula": (
            "physics_Ceq_F*exp(delta_prediction+"
            "q90_abs_log_error*(1+feature_extrapolation_distance))"
        ),
        "feature_extrapolation_penalty_active": True,
        "frequency_lcb_formula": "1/(2*pi*sqrt(0.2*C_ucb))",
        "secondary_inductance_H": L_SECONDARY_H,
        "primary_magnetizing_inductance_H": L_PRIMARY_H,
        "turns_ratio_N2_over_N1": 10.0,
        "minimum_frequency_Hz": RESONANCE_MIN_HZ,
        "physical_G_formula": "15000-physics_delta_fRx_q90_lcb_Hz",
        "optimizer_hard_gate_active": True,
        "raw_two_net_capacitance_predictor_called": False,
        "raw_two_net_capacitance_G_present": False,
        "legacy_half_magnetizing_resonance_G_present": False,
        "single_transfer_ratio_used": False,
        "forbidden_optimizer_objective_constraint_eligibility_inputs": list(
            FORBIDDEN_OPTIMIZER_INPUTS
        ),
        "calibration_domain": _calibration_domain(model),
        "prediction_role": "screening_acquisition_only",
        "physical_feasibility_authority": False,
        "final_turn_graded_symmetric_FEA_required": True,
        "approved_dielectric_stack_sensitivity_required": True,
        "fixed20T_turn_graded_FEA_retraining_required": True,
    }


def _turn_split_repair_contract() -> dict[str, Any]:
    return {
        "schema_version": "mft-goal-corrected-N2-split-local-repair-v2",
        "fixed_total_secondary_turns": 60,
        "N2_main_minimum": SPLIT_MIN_N2_MAIN,
        "N2_main_maximum": SPLIT_MAX_N2_MAIN,
        "N2_main_integer_values": list(SPLIT_VALUES),
        "enumerated_split_count_per_geometry": len(SPLIT_VALUES),
        "decoder_coordinate_name": "u_N2_side",
        "decoder_coordinate_index": 2,
        "decoder_mapping": (
            "N2_side=round(60*clip(u_N2_side,0,1)*0.8);"
            "N2_main=60-N2_side"
        ),
        "selection": "minimum_robust_Llt_G",
        "robust_Llt_G_formula": (
            "abs(mu_Llt_phys-Llt_target_uH)+"
            "q_sigma*q90_conformal_half_width-Llt_tol_uH"
        ),
        "original_split_early_hard_rejection_allowed": False,
        "selected_split_Llt_G_remains_hard": True,
        "core_B_temperature_size_physics_C_evaluated_after_split_repair": True,
        "turn_voltage_physics_C_schedule_recomputed_for_selected_split": True,
        "terminal_selected_split_sealed": True,
        "terminal_immediate_neighbor_splits_sealed": True,
        "terminal_before_after_Llt_G_sealed": True,
        "coordinate_genome_is_geometry_donor_not_Llt_rejection_authority": True,
        "enumeration_geometry_policy": (
            "hold_decoded_geometry_conductor_sections_and_gaps_fixed"
        ),
        "split_dependent_derived_features_recomputed": True,
        "decoder_reprojection_per_split_used": False,
        "vectorized_Llt_enumeration": True,
    }


def _build_search_profile(
    runner: Any,
    *,
    geometry_profile: Mapping[str, Any],
    authorized_seed_start: int | None = None,
    secondary_gap_mode: str = scout.SECONDARY_GAP_MODE_BOUNDED,
) -> dict[str, Any]:
    del runner
    geometry = preflight.validate_goal_geometry_constraint_profile(
        geometry_profile
    )
    clearance = scout._profile_clearance_mm(
        geometry["primary_axial_clearance"]["minimum_mm"]
    )
    seed_start = scout._authorized_profile_seed_start(
        clearance,
        SEED_START if authorized_seed_start is None else authorized_seed_start,
    )
    if (
        seed_start != SEED_START
        or secondary_gap_mode != scout.SECONDARY_GAP_MODE_BOUNDED
    ):
        raise RuntimeError(
            "corrected physics NSGA requires bounded exact60 seeds 4400..4459"
        )
    model = _active_model()
    constraints = scout._effective_constraint_profile()
    value = {
        "schema_version": SEARCH_PROFILE_SCHEMA,
        "profile_id": (
            "corrected-physics-delta-l900-temp110-130-130-"
            "hard-equal-hgap20-gap2p35to2p00-cw2p30to1p00-"
            "plates20-exact60-s2607264400"
        ),
        "campaign_id": CAMPAIGN_ID,
        "fixed_primary_turns": scout.FIXED_PRIMARY_TURNS,
        "fixed_secondary_turns": scout.FIXED_SECONDARY_TURNS,
        "turns_ratio_N2_over_N1": 10.0,
        "fixed_primary_conductor_thickness_mm": (
            scout.FIXED_PRIMARY_CONDUCTOR_THICKNESS_MM
        ),
        "fixed_primary_interturn_gap_mm": (
            scout.FIXED_PRIMARY_INTERTURN_GAP_MM
        ),
        "fixed_core_plate_thickness_mm": (
            scout.FIXED_CORE_PLATE_THICKNESS_MM
        ),
        "fixed_winding_cold_plate_thickness_mm": (
            scout.FIXED_WINDING_COLD_PLATE_THICKNESS_MM
        ),
        "effective_constraint_profile": constraints,
        "effective_constraint_profile_payload_sha256": constraints[
            "payload_sha256"
        ],
        "geometry_constraint_profile": copy.deepcopy(geometry),
        "geometry_constraint_profile_sha256": geometry["sha256"],
        "winding_height_exact_equality_initialization_and_repair_required": (
            True
        ),
        "maximum_decoded_winding_height_difference_mm": (
            scout.MAXIMUM_DECODED_WINDING_HEIGHT_DIFFERENCE_MM
        ),
        "authorized_seed_start": seed_start,
        "authorized_seed_count": SEED_COUNT,
        "authorized_seed_end_inclusive": SEED_END,
        "secondary_interturn_gap_search_mm": {
            "minimum": scout.VARIABLE_SECONDARY_INTERTURN_GAP_MINIMUM_MM,
            "maximum": scout.VARIABLE_SECONDARY_INTERTURN_GAP_MAXIMUM_MM,
            "step": scout.VARIABLE_SECONDARY_INTERTURN_GAP_STEP_MM,
            "decoder_native_minimum": 0.3,
            "decoder_native_maximum": 2.0,
            "realized_band_constraint_name": (
                scout.SECONDARY_GAP_BAND_CONSTRAINT_NAME
            ),
        },
        "secondary_conductor_thickness_search_mm": {
            "minimum": scout.SECONDARY_CONDUCTOR_THICKNESS_MINIMUM_MM,
            "maximum": scout.SECONDARY_CONDUCTOR_THICKNESS_MAXIMUM_MM,
            "hard_constraint_name": (
                scout.SECONDARY_CONDUCTOR_BAND_CONSTRAINT_NAME
            ),
        },
        "physics_delta_rx_resonance_gate": _physics_gate(model),
        "turn_split_local_repair": _turn_split_repair_contract(),
        "scheduler_priority": SCHEDULER_PRIORITY,
        "worker_entrypoint": WORKER_ENTRYPOINT,
        "scheduler_task_name_prefix": TASK_NAME_PREFIX,
        "scheduler_dedupe_prefix": DEDUPE_PREFIX,
        "cooling_or_TIM_contract_mutated": False,
        "screening_only": True,
        "production_eligible": False,
        "final_design_claim_allowed": False,
    }
    return goal_launch._seal(value)


def _validate_search_profile(value: Mapping[str, Any]) -> dict[str, Any]:
    profile = goal_launch._validate_seal(
        dict(value), schema=SEARCH_PROFILE_SCHEMA
    )
    geometry = preflight.validate_goal_geometry_constraint_profile(
        profile.get("geometry_constraint_profile") or {}
    )
    constraints = scout._validate_effective_constraint_profile(
        profile.get("effective_constraint_profile") or {}
    )
    gate = profile.get("physics_delta_rx_resonance_gate")
    if not isinstance(gate, Mapping):
        raise RuntimeError("physics-delta resonance gate is unavailable")
    model = _validate_embedded_model(gate.get("model") or {})
    expected_gate = _physics_gate(model)
    expected_keys = {
        "schema_version",
        "profile_id",
        "campaign_id",
        "fixed_primary_turns",
        "fixed_secondary_turns",
        "turns_ratio_N2_over_N1",
        "fixed_primary_conductor_thickness_mm",
        "fixed_primary_interturn_gap_mm",
        "fixed_core_plate_thickness_mm",
        "fixed_winding_cold_plate_thickness_mm",
        "effective_constraint_profile",
        "effective_constraint_profile_payload_sha256",
        "geometry_constraint_profile",
        "geometry_constraint_profile_sha256",
        "winding_height_exact_equality_initialization_and_repair_required",
        "maximum_decoded_winding_height_difference_mm",
        "authorized_seed_start",
        "authorized_seed_count",
        "authorized_seed_end_inclusive",
        "secondary_interturn_gap_search_mm",
        "secondary_conductor_thickness_search_mm",
        "physics_delta_rx_resonance_gate",
        "turn_split_local_repair",
        "scheduler_priority",
        "worker_entrypoint",
        "scheduler_task_name_prefix",
        "scheduler_dedupe_prefix",
        "cooling_or_TIM_contract_mutated",
        "screening_only",
        "production_eligible",
        "final_design_claim_allowed",
        "payload_sha256",
    }
    gap = profile.get("secondary_interturn_gap_search_mm") or {}
    conductor = (
        profile.get("secondary_conductor_thickness_search_mm") or {}
    )
    if (
        set(profile) != expected_keys
        or profile.get("campaign_id") != CAMPAIGN_ID
        or profile.get("authorized_seed_start") != SEED_START
        or profile.get("authorized_seed_count") != SEED_COUNT
        or profile.get("authorized_seed_end_inclusive") != SEED_END
        or profile.get("fixed_primary_turns") != 6
        or profile.get("fixed_secondary_turns") != 60
        or profile.get("turns_ratio_N2_over_N1") != 10.0
        or profile.get("fixed_primary_conductor_thickness_mm") != 5.0
        or profile.get("fixed_primary_interturn_gap_mm") != 1.6
        or profile.get("fixed_core_plate_thickness_mm") != 20.0
        or profile.get("fixed_winding_cold_plate_thickness_mm") != 20.0
        or profile.get("effective_constraint_profile_payload_sha256")
        != constraints["payload_sha256"]
        or profile.get("geometry_constraint_profile_sha256")
        != geometry["sha256"]
        or profile.get(
            "winding_height_exact_equality_initialization_and_repair_required"
        )
        is not True
        or profile.get("maximum_decoded_winding_height_difference_mm")
        != 0.1
        or gate != expected_gate
        or profile.get("turn_split_local_repair")
        != _turn_split_repair_contract()
        or gap
        != {
            "minimum": 0.35,
            "maximum": 2.0,
            "step": 0.001,
            "decoder_native_minimum": 0.3,
            "decoder_native_maximum": 2.0,
            "realized_band_constraint_name": (
                scout.SECONDARY_GAP_BAND_CONSTRAINT_NAME
            ),
        }
        or conductor
        != {
            "minimum": 0.3,
            "maximum": 1.0,
            "hard_constraint_name": (
                scout.SECONDARY_CONDUCTOR_BAND_CONSTRAINT_NAME
            ),
        }
        or profile.get("scheduler_priority") != 100
        or profile.get("worker_entrypoint") != WORKER_ENTRYPOINT
        or profile.get("scheduler_task_name_prefix") != TASK_NAME_PREFIX
        or profile.get("scheduler_dedupe_prefix") != DEDUPE_PREFIX
        or profile.get("cooling_or_TIM_contract_mutated") is not False
        or profile.get("screening_only") is not True
        or profile.get("production_eligible") is not False
        or profile.get("final_design_claim_allowed") is not False
    ):
        raise RuntimeError("corrected physics-delta search profile mismatch")
    return profile


def _secondary_gap_mode_from_profile(
    profile: Mapping[str, Any],
) -> str:
    _validate_search_profile(profile)
    return scout.SECONDARY_GAP_MODE_BOUNDED


def _candidate_prediction(
    row: Mapping[str, Any],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    prediction = physics._candidate_prediction(row, model)
    c_ucb = float(prediction["physics_delta_Crx_q90_ucb_F"])
    expected = 1.0 / (
        2.0 * math.pi * math.sqrt(L_SECONDARY_H * c_ucb)
    )
    observed = float(prediction["physics_delta_fRx_q90_lcb_Hz"])
    if not math.isclose(expected, observed, rel_tol=1e-13, abs_tol=1e-9):
        raise RuntimeError("physics-delta resonance formula drifted")
    return prediction


def _expanded_fixed_geometry_split_frame(
    frame: Any,
    *,
    decoder_valid: Any,
) -> tuple[Any, np.ndarray]:
    """Enumerate the integer secondary split without moving the geometry.

    The base decoder's coordinate projection depends on ``u_N2_side``.  Calling
    that projection 49 times therefore changes l1/l2/spacing in addition to the
    requested turn split.  A local repair must instead hold the decoded
    geometry, conductor section and insulation gaps fixed and update only the
    exact split-dependent derived quantities used by Llt inference.
    """

    count = len(frame)
    repeats = len(SPLIT_VALUES)
    expanded = frame.iloc[
        np.repeat(np.arange(count, dtype=int), repeats)
    ].reset_index(drop=True).copy()
    split_main = np.tile(
        np.asarray(SPLIT_VALUES, dtype=int),
        count,
    )
    split_side = 60 - split_main
    expanded["N2_main"] = split_main
    expanded["N2_side"] = split_side
    if "N2" in expanded.columns:
        expanded["N2"] = 60

    # The unit-test problem intentionally exposes only the turn columns.  The
    # production decoder exposes this complete derived-geometry set.
    required = {
        "l2",
        "cw2",
        "gap2",
        "nwh2",
        "h1",
        "sl2_main_x",
        "sl2_main_y",
        "sl2_side_x",
        "sl2_side_y",
        "cc_w2c_space_x",
        "w2c_w1c_space_x",
        "nwl1_main",
        "w1s_cs_space_x",
        "w1c_w2s_space_x",
    }
    production_geometry = required.issubset(expanded.columns)
    if not production_geometry:
        return expanded, np.ones(len(expanded), dtype=bool)

    def values(name: str) -> np.ndarray:
        return np.asarray(expanded[name], dtype=float)

    cw2 = values("cw2")
    gap2 = values("gap2")
    nwh2 = values("nwh2")
    nwl2_main = (
        split_main * cw2
        + np.maximum(split_main - 1, 0) * gap2
    )
    nwl2_side = np.where(
        split_side > 0,
        split_side * cw2
        + np.maximum(split_side - 1, 0) * gap2,
        0.0,
    )
    expanded["nwl2_main"] = nwl2_main
    expanded["nwl2_side"] = nwl2_side
    expanded["wff2_main"] = np.divide(
        split_main * cw2,
        nwl2_main,
        out=np.zeros_like(cw2),
        where=nwl2_main > 0.0,
    )
    expanded["wff2_side"] = np.divide(
        split_side * cw2,
        nwl2_side,
        out=np.zeros_like(cw2),
        where=nwl2_side > 0.0,
    )

    sl2_main_x = values("sl2_main_x")
    sl2_main_y = values("sl2_main_y")
    sl1_main_x = (
        sl2_main_x
        + 2.0 * nwl2_main
        + 2.0 * values("w2c_w1c_space_x")
    )
    if "w2c_w1c_space_y" in expanded.columns:
        sl1_main_y = (
            sl2_main_y
            + 2.0 * nwl2_main
            + 2.0 * values("w2c_w1c_space_y")
        )
        expanded["sl1_main_y"] = sl1_main_y
    expanded["sl1_main_x"] = sl1_main_x

    center_stack = (
        values("cc_w2c_space_x")
        + nwl2_main
        + values("w2c_w1c_space_x")
        + values("nwl1_main")
    )
    side_stack = values("w1s_cs_space_x") + nwl2_side
    actual_clearance = values("l2") - center_stack - side_stack
    expanded["w1c_w2s_gap_x_actual"] = actual_clearance

    round_corner = (
        values("round_corner") != 0.0
        if "round_corner" in expanded.columns
        else np.zeros(len(expanded), dtype=bool)
    )
    corner_radius = (
        values("corner_radius")
        if "corner_radius" in expanded.columns
        else np.zeros(len(expanded), dtype=float)
    )
    wcp_len_ref_x = sl1_main_x - np.where(
        round_corner,
        2.0 * corner_radius,
        0.0,
    )
    expanded["wcp_len_ref_x"] = wcp_len_ref_x
    if {"wcp_len_pct", "wcp_len_x"}.issubset(expanded.columns):
        # validation_check reports the realized percentage after the
        # authoritative 0.1-mm length quantization.  Rounding it back to one
        # decimal recovers the sampled percentage used by the decoder.
        requested_pct = np.round(values("wcp_len_pct"), 1)
        wcp_len_x = np.round(
            wcp_len_ref_x * requested_pct / 100.0,
            1,
        )
        expanded["wcp_len_x"] = wcp_len_x
        expanded["wcp_len_pct"] = np.divide(
            100.0 * wcp_len_x,
            wcp_len_ref_x,
            out=np.full_like(wcp_len_x, np.nan),
            where=wcp_len_ref_x > 0.0,
        )

    def copper_group(
        turns: np.ndarray,
        inner_x: np.ndarray,
        inner_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        turns_float = np.asarray(turns, dtype=float)
        total_length = 4.0 * (
            turns_float * ((inner_x + inner_y) / 2.0 + cw2)
            + (cw2 + gap2) * turns_float * (turns_float - 1.0)
        )
        mlt = np.divide(
            total_length,
            turns_float,
            out=np.zeros_like(total_length),
            where=turns_float > 0.0,
        )
        mass = total_length * cw2 * nwh2 * 1e-9 * 8940.0
        return mlt, mass

    mlt_main, mass_main = copper_group(
        split_main,
        sl2_main_x,
        sl2_main_y,
    )
    mlt_side, mass_side_single = copper_group(
        split_side,
        values("sl2_side_x"),
        values("sl2_side_y"),
    )
    expanded["MLT_Rx_main_mm"] = mlt_main
    expanded["MLT_Rx_side_mm"] = mlt_side
    expanded["cu_mass_Rx_main_kg"] = mass_main
    expanded["cu_mass_Rx_side_kg"] = 2.0 * mass_side_single
    if "cu_mass_Tx_kg" in expanded.columns:
        expanded["cu_mass_total_kg"] = (
            values("cu_mass_Tx_kg")
            + mass_main
            + 2.0 * mass_side_single
        )
    expanded["window_fill_x"] = np.divide(
        center_stack + side_stack,
        values("l2"),
        out=np.full(len(expanded), np.inf, dtype=float),
        where=values("l2") > 0.0,
    )

    finite_names = tuple(
        name
        for name in (
            "l2",
            "h1",
            "cw1",
            "gap1",
            "cw2",
            "gap2",
            "nwh1",
            "nwh2",
            "core_depth_each",
            "nwl2_main",
            "nwl2_side",
            "w1c_w2s_gap_x_actual",
            "wcp_len_ref_x",
        )
        if name in expanded.columns
    )
    valid = np.ones(len(expanded), dtype=bool)
    if finite_names:
        valid &= np.isfinite(
            expanded.loc[:, finite_names].to_numpy(dtype=float)
        ).all(axis=1)
    valid &= cw2 > 0.0
    valid &= gap2 > 0.0
    valid &= values("l2") > 0.0
    valid &= values("h1") > 0.0
    valid &= nwh2 <= values("h1")
    if "nwh1" in expanded.columns:
        valid &= values("nwh1") <= values("h1")
    if "cw1" in expanded.columns:
        valid &= values("cw1") > 0.0
    if "gap1" in expanded.columns:
        valid &= values("gap1") > 0.0
    if "core_depth_each" in expanded.columns:
        valid &= values("core_depth_each") > 0.0
    valid &= actual_clearance >= values("w1c_w2s_space_x")
    valid &= wcp_len_ref_x > 0.0

    # A decoder exception produces an empty/NaN base row and is already
    # rejected by the finite checks.  A merely split-specific invalid base row
    # is deliberately not copied to all 49 alternatives.
    base_valid = np.asarray(decoder_valid, dtype=bool).reshape(-1)
    if base_valid.shape != (count,):
        raise RuntimeError("N2 split base decoder-valid shape mismatch")
    return expanded, valid


def _install_turn_split_local_repair(
    problem: Any,
    *,
    original_decode: Any,
    original_predict: Any,
) -> dict[str, Any]:
    """Select the minimum robust-Llt integer split before hard evaluation.

    Only the Llt model is evaluated for the 49-way local enumeration.  The
    complete thermal/core/size evaluator and the physics capacitance network
    are then evaluated once at the selected repaired phenotype.
    """

    coordinate_index = 2
    names = tuple(problem.sobol_dimension_names)
    if (
        coordinate_index >= len(names)
        or names[coordinate_index] != "u_N2_side"
    ):
        raise RuntimeError("N2 split decoder coordinate authority drifted")
    unit_values = np.asarray(
        [
            preflight._turn_split_unit_coordinate(
                split,
                fixed_primary_turns=6,
            )
            for split in SPLIT_VALUES
        ],
        dtype=float,
    )
    if (
        len(np.unique(unit_values)) != len(SPLIT_VALUES)
        or not np.all((unit_values >= 0.0) & (unit_values <= 1.0))
    ):
        raise RuntimeError("N2 split enumeration is not one-to-one")

    def split_repaired_decode(values: Any) -> tuple[Any, Any, Any]:
        coordinates = np.asarray(values, dtype=float)
        if (
            coordinates.ndim != 2
            or coordinates.shape[1] != int(problem.n_var)
        ):
            raise RuntimeError("N2 split repair coordinate shape mismatch")
        count = len(coordinates)
        base_frame, base_shrink, base_valid = original_decode(coordinates)
        base_shrink = np.asarray(base_shrink, dtype=float).reshape(-1)
        base_valid = np.asarray(base_valid, dtype=bool).reshape(-1)
        frame, valid = _expanded_fixed_geometry_split_frame(
            base_frame,
            decoder_valid=base_valid,
        )
        expected = count * len(SPLIT_VALUES)
        if (
            len(frame) != expected
            or base_shrink.shape != (count,)
            or valid.shape != (expected,)
        ):
            raise RuntimeError("N2 split expanded decode shape mismatch")
        scores = np.full(expected, preflight.BIG, dtype=float)
        means = np.full(expected, np.nan, dtype=float)
        half_widths = np.full(expected, np.nan, dtype=float)
        indices = np.flatnonzero(valid)
        if len(indices):
            sub = frame.iloc[indices]
            mean, half_width = original_predict("Llt_phys", sub)
            mean = np.asarray(mean, dtype=float).reshape(-1)
            half_width = np.asarray(half_width, dtype=float).reshape(-1)
            if (
                mean.shape != (len(indices),)
                or half_width.shape != (len(indices),)
                or not np.isfinite(mean).all()
                or not np.isfinite(half_width).all()
                or np.any(half_width < 0.0)
            ):
                raise RuntimeError("N2 split Llt inference is invalid")
            target = float(problem.spec["Llt_target_uH"])
            tolerance = float(problem.spec["Llt_tol_uH"])
            q_sigma = float(problem.spec["q_sigma"])
            robust = (
                np.abs(mean - target)
                + q_sigma * half_width
                - tolerance
            )
            scores[indices] = robust
            means[indices] = mean
            half_widths[indices] = half_width

        original_splits = np.asarray(
            preflight._turn_split_main_values(
                coordinates,
                fixed_primary_turns=6,
                coordinate_index=coordinate_index,
            ),
            dtype=int,
        ).reshape(-1)
        selected_indices: list[int] = []
        audit_rows: list[dict[str, Any]] = []
        split_array = np.asarray(SPLIT_VALUES, dtype=int)
        for row_index in range(count):
            start = row_index * len(SPLIT_VALUES)
            stop = start + len(SPLIT_VALUES)
            row_scores = scores[start:stop]
            row_valid = valid[start:stop]
            original_split = int(original_splits[row_index])
            original_offset = int(
                np.argmin(np.abs(split_array - original_split))
            )
            if np.any(row_valid):
                ordering = np.lexsort(
                    (
                        split_array,
                        np.abs(split_array - original_split),
                        row_scores,
                    )
                )
                selected_offset = int(
                    next(index for index in ordering if row_valid[index])
                )
            else:
                selected_offset = original_offset
            selected_index = start + selected_offset
            selected_indices.append(selected_index)

            def split_record(offset: int | None) -> dict[str, Any] | None:
                if offset is None:
                    return None
                absolute = start + offset
                return {
                    "N2_main": int(split_array[offset]),
                    "N2_side": int(60 - split_array[offset]),
                    "decoder_valid": bool(valid[absolute]),
                    "robust_Llt_G": (
                        float(scores[absolute])
                        if math.isfinite(float(scores[absolute]))
                        else None
                    ),
                    "Llt_mean_uH": (
                        float(means[absolute])
                        if math.isfinite(float(means[absolute]))
                        else None
                    ),
                    "Llt_q90_half_width_uH": (
                        float(half_widths[absolute])
                        if math.isfinite(float(half_widths[absolute]))
                        else None
                    ),
                }

            lower = (
                selected_offset - 1 if selected_offset > 0 else None
            )
            upper = (
                selected_offset + 1
                if selected_offset + 1 < len(SPLIT_VALUES)
                else None
            )
            audit_rows.append(
                {
                    "original": split_record(original_offset),
                    "selected": split_record(selected_offset),
                    "neighbor_lower": split_record(lower),
                    "neighbor_upper": split_record(upper),
                    "enumerated_split_count": len(SPLIT_VALUES),
                    "decoder_valid_split_count": int(
                        np.count_nonzero(row_valid)
                    ),
                    "selection": "minimum_robust_Llt_G",
                }
            )

        selected = np.asarray(selected_indices, dtype=int)
        selected_frame = frame.iloc[selected].reset_index(drop=True).copy()
        selected_shrink = base_shrink.copy()
        selected_valid = valid[selected]
        for index, audit in enumerate(audit_rows):
            original = audit["original"] or {}
            chosen = audit["selected"] or {}
            lower = audit["neighbor_lower"] or {}
            upper = audit["neighbor_upper"] or {}
            selected_frame.loc[
                index, "split_repair_original_N2_main"
            ] = original.get("N2_main")
            selected_frame.loc[
                index, "split_repair_selected_N2_main"
            ] = chosen.get("N2_main")
            selected_frame.loc[
                index, "split_repair_original_robust_Llt_G"
            ] = original.get("robust_Llt_G")
            selected_frame.loc[
                index, "split_repair_selected_robust_Llt_G"
            ] = chosen.get("robust_Llt_G")
            selected_frame.loc[
                index, "split_repair_neighbor_lower_N2_main"
            ] = lower.get("N2_main")
            selected_frame.loc[
                index, "split_repair_neighbor_lower_robust_Llt_G"
            ] = lower.get("robust_Llt_G")
            selected_frame.loc[
                index, "split_repair_neighbor_upper_N2_main"
            ] = upper.get("N2_main")
            selected_frame.loc[
                index, "split_repair_neighbor_upper_robust_Llt_G"
            ] = upper.get("robust_Llt_G")
            selected_frame.loc[
                index, "split_repair_enumerated_split_count"
            ] = audit["enumerated_split_count"]
            selected_frame.loc[
                index, "split_repair_decoder_valid_split_count"
            ] = audit["decoder_valid_split_count"]
            selected_frame.loc[
                index, "split_repair_neighbors_json"
            ] = json.dumps(
                {
                    "lower": audit["neighbor_lower"],
                    "upper": audit["neighbor_upper"],
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        if len(selected_frame) != count:
            raise RuntimeError("N2 split selected phenotype count mismatch")
        problem._last_decode = (
            selected_frame,
            selected_shrink,
            selected_valid,
        )
        problem._last_turn_split_repair_audit = copy.deepcopy(audit_rows)
        return selected_frame, selected_shrink, selected_valid

    problem.decode_batch = split_repaired_decode
    problem._turn_split_local_repair_installed = True
    problem._turn_split_local_repair_contract = (
        _turn_split_repair_contract()
    )
    return {
        **_turn_split_repair_contract(),
        "unit_coordinate_minimum": float(np.min(unit_values)),
        "unit_coordinate_maximum": float(np.max(unit_values)),
        "unit_coordinate_sha256": canonical_sha256(unit_values.tolist()),
    }


def _install_search_profile(
    problem: Any,
    search_profile: Mapping[str, Any],
) -> dict[str, Any]:
    profile = _validate_search_profile(search_profile)
    if getattr(problem, "_diagnostic_search_profile_installed", False):
        raise RuntimeError("corrected physics profile was already installed")
    if (
        problem.geometry_constraint_profile_sha256
        != profile["geometry_constraint_profile_sha256"]
    ):
        raise RuntimeError("corrected geometry profile was not installed")
    constraints_profile = scout._validate_effective_constraint_profile(
        profile["effective_constraint_profile"]
    )
    if hasattr(problem, "spec"):
        effective_spec = copy.deepcopy(problem.spec)
        effective_spec["size_limits_mm"] = copy.deepcopy(
            constraints_profile["size_limits_mm"]
        )
        effective_spec["temperature_family_limits_C"] = copy.deepcopy(
            constraints_profile["temperature_family_limits_C"]
        )
        effective_spec["temperature_target_limits_C"] = copy.deepcopy(
            constraints_profile["temperature_target_limits_C"]
        )
        problem.spec = effective_spec
        problem.temperature_limits_C = copy.deepcopy(
            constraints_profile["temperature_target_limits_C"]
        )
        problem.temperature_contract = copy.deepcopy(
            constraints_profile["temperature_contract"]
        )
        problem.temperature_contract_sha256 = constraints_profile[
            "temperature_contract_sha256"
        ]

    names = tuple(problem.sobol_dimension_names)
    required = {"gap1", "gap2", "core_plate_t", "wcp_t"}
    if not required.issubset(names):
        raise RuntimeError("corrected fixed manufacturing coordinate is absent")
    cw1_index = int(problem.cw1_coordinate_index)
    gap1_index = names.index("gap1")
    gap2_index = names.index("gap2")
    core_plate_index = names.index("core_plate_t")
    wcp_index = names.index("wcp_t")
    coordinates = {
        cw1_index: float(preflight.cw1_unit_coordinate(5.0)),
        gap1_index: float(problem._unit_from_physical("gap1", 1.6)),
        core_plate_index: float(
            problem._unit_from_physical("core_plate_t", 20.0)
        ),
        wcp_index: float(problem._unit_from_physical("wcp_t", 20.0)),
    }
    for index, coordinate in coordinates.items():
        problem.xl[index] = coordinate
        problem.xu[index] = coordinate
    gap2_lower = float(problem._unit_from_physical("gap2", 0.35))
    gap2_upper = float(problem._unit_from_physical("gap2", 2.0))
    problem.xl[gap2_index] = gap2_lower
    problem.xu[gap2_index] = gap2_upper

    base_evaluate = problem._evaluate
    original_decode = problem.decode_batch
    original_predict = problem._predict
    split_repair_installation = _install_turn_split_local_repair(
        problem,
        original_decode=original_decode,
        original_predict=original_predict,
    )
    base_names = tuple(problem.constraint_names)
    base_index = dict(problem.constraint_index)
    base_count = int(problem.n_ieq_constr)
    legacy_name = preflight.RESONANCE_MINIMUM_CONSTRAINT
    if legacy_name not in base_index:
        raise RuntimeError("legacy resonance slot is unavailable for replacement")
    resonance_index = int(base_index[legacy_name])
    replaced_names = list(base_names)
    replaced_names[resonance_index] = PHYSICS_CONSTRAINT_NAME
    effective_names = (
        *replaced_names,
        scout.SECONDARY_GAP_BAND_CONSTRAINT_NAME,
        scout.SECONDARY_CONDUCTOR_BAND_CONSTRAINT_NAME,
    )
    if len(set(effective_names)) != len(effective_names):
        raise RuntimeError("corrected physics constraint names are duplicated")
    effective_index = {
        name: index for index, name in enumerate(effective_names)
    }
    model = _validate_embedded_model(
        profile["physics_delta_rx_resonance_gate"]["model"]
    )
    def profiled_evaluate(
        values: Any,
        out: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        prediction_cache: dict[int, list[dict[str, Any]]] = {}

        def predict_rows(frame: Any) -> list[dict[str, Any]]:
            key = id(frame)
            if key not in prediction_cache:
                prediction_cache[key] = [
                    _candidate_prediction(frame.iloc[index], model)
                    for index in range(len(frame))
                ]
            return prediction_cache[key]

        def guarded_predict(target: str, frame: Any) -> tuple[Any, Any]:
            if target == "C_rx_rx_F":
                rows = predict_rows(frame)
                mean = np.asarray(
                    [row["physics_delta_Crx_mean_F"] for row in rows],
                    dtype=float,
                )
                ucb = np.asarray(
                    [row["physics_delta_Crx_q90_ucb_F"] for row in rows],
                    dtype=float,
                )
                return mean, ucb - mean
            if target == "C_tx_tx_F":
                # Current7's superseded compatibility calculation asks for a
                # Tx capacitance.  Supply a non-authoritative finite sentinel;
                # the resulting legacy slot is overwritten before return.
                return (
                    np.full(len(frame), 1e-18, dtype=float),
                    np.zeros(len(frame), dtype=float),
                )
            return original_predict(target, frame)

        problem.constraint_names = base_names
        problem.constraint_index = base_index
        problem.n_ieq_constr = base_count
        problem._predict = guarded_predict
        try:
            base_evaluate(values, out, *args, **kwargs)
        finally:
            problem._predict = original_predict
            problem.constraint_names = effective_names
            problem.constraint_index = effective_index
            problem.n_ieq_constr = len(effective_names)

        base_g = np.asarray(out.get("G"), dtype=float)
        valid = np.asarray(out.get("decoder_valid"), dtype=bool).reshape(-1)
        frame = out.get("frame")
        if (
            base_g.shape != (len(valid), base_count)
            or frame is None
            or len(frame) != len(valid)
        ):
            raise RuntimeError("corrected physics base evaluation shape mismatch")
        physics_g = np.full(len(valid), preflight.BIG, dtype=float)
        c_mean = np.full(len(valid), np.nan, dtype=float)
        c_ucb = np.full(len(valid), np.nan, dtype=float)
        f_lcb = np.full(len(valid), np.nan, dtype=float)
        extrapolation = np.full(len(valid), np.nan, dtype=float)
        gap_g = np.full(len(valid), preflight.BIG, dtype=float)
        cw2_g = np.full(len(valid), preflight.BIG, dtype=float)
        indices = np.flatnonzero(valid)
        if len(indices):
            sub = frame.iloc[indices]
            predictions = predict_rows(sub)
            for local, global_index in enumerate(indices):
                row = sub.iloc[local]
                if (
                    not math.isclose(float(row["cw1"]), 5.0, abs_tol=1e-12)
                    or not math.isclose(
                        float(row["gap1"]), 1.6, abs_tol=1e-12
                    )
                    or not math.isclose(
                        float(row["core_plate_t"]), 20.0, abs_tol=1e-12
                    )
                    or not math.isclose(
                        float(row["wcp_t"]), 20.0, abs_tol=1e-12
                    )
                    or int(row["N1_main"]) + int(row["N1_side"]) != 6
                    or int(row["N2_main"]) + int(row["N2_side"]) != 60
                ):
                    raise RuntimeError(
                        "corrected physics fixed manufacturing controls escaped"
                    )
                gap2 = float(row["gap2"])
                cw2 = float(row["cw2"])
                item = predictions[local]
                c_mean[global_index] = float(
                    item["physics_delta_Crx_mean_F"]
                )
                c_ucb[global_index] = float(
                    item["physics_delta_Crx_q90_ucb_F"]
                )
                f_lcb[global_index] = float(
                    item["physics_delta_fRx_q90_lcb_Hz"]
                )
                extrapolation[global_index] = float(
                    item["physics_delta_extrapolation_distance"]
                )
                physics_g[global_index] = (
                    RESONANCE_MIN_HZ - f_lcb[global_index]
                )
                gap_g[global_index] = max(0.35 / gap2 - 1.0, gap2 / 2.0 - 1.0)
                cw2_g[global_index] = max(0.3 / cw2 - 1.0, cw2 / 1.0 - 1.0)
        base_g[:, resonance_index] = physics_g
        out["G"] = np.column_stack([base_g, gap_g, cw2_g])
        out["physics_delta_Crx_mean_F"] = c_mean
        out["physics_delta_Crx_q90_ucb_F"] = c_ucb
        out["physics_delta_fRx_q90_lcb_Hz"] = f_lcb
        out["physics_delta_extrapolation_distance"] = extrapolation
        out["raw_two_net_capacitance_predictor_authority_used"] = False
        out["single_0p759701_transfer_ratio_used"] = False

    base_contract = copy.deepcopy(problem.hard_constraint_contract)
    if canonical_sha256(base_contract) != problem.hard_constraint_contract_sha256:
        raise RuntimeError("corrected physics base hard contract is unauthenticated")
    effective_contract = copy.deepcopy(base_contract)
    for key in (
        "self_resonance",
        "resonance_authority",
        "resonance_authority_sha256",
        "effective_self_resonance_authority",
        "effective_resonance_authority_precedence",
        "legacy_constraint_name_is_compatibility_alias",
    ):
        effective_contract.pop(key, None)
    gate_contract = copy.deepcopy(
        profile["physics_delta_rx_resonance_gate"]
    )
    gate_contract.pop("model", None)
    effective_contract.update(
        {
            "constraint_names": list(effective_names),
            "corrected_physics_delta_search_profile_sha256": profile[
                "payload_sha256"
            ],
            "physics_delta_rx_resonance_gate": gate_contract,
            "turn_split_local_repair": copy.deepcopy(
                _turn_split_repair_contract()
            ),
            "size_limits_mm": copy.deepcopy(
                constraints_profile["size_limits_mm"]
            ),
            "temperature_contract": copy.deepcopy(
                constraints_profile["temperature_contract"]
            ),
            "temperature_contract_sha256": constraints_profile[
                "temperature_contract_sha256"
            ],
            "raw_two_net_capacitance_G_present": False,
            "legacy_half_magnetizing_resonance_G_present": False,
            "screening_only": True,
            "production_eligible": False,
        }
    )
    problem._evaluate = profiled_evaluate
    problem.constraint_names = effective_names
    problem.constraint_index = effective_index
    problem.n_ieq_constr = len(effective_names)
    problem.hard_constraint_contract = effective_contract
    problem.hard_constraint_contract_sha256 = canonical_sha256(
        effective_contract
    )
    problem._diagnostic_search_profile_installed = True
    problem._diagnostic_search_profile = profile
    problem._physics_delta_model = model
    evidence = goal_launch._seal(
        {
            "schema_version": INSTALLATION_SCHEMA,
            "search_profile_payload_sha256": profile["payload_sha256"],
            "geometry_constraint_profile_sha256": profile[
                "geometry_constraint_profile_sha256"
            ],
            "effective_constraint_profile_payload_sha256": (
                constraints_profile["payload_sha256"]
            ),
            "effective_size_limits_mm": copy.deepcopy(
                constraints_profile["size_limits_mm"]
            ),
            "effective_temperature_family_limits_C": copy.deepcopy(
                constraints_profile["temperature_family_limits_C"]
            ),
            "base_hard_constraint_contract_sha256": canonical_sha256(
                base_contract
            ),
            "effective_hard_constraint_contract_sha256": (
                problem.hard_constraint_contract_sha256
            ),
            "physics_constraint_name": PHYSICS_CONSTRAINT_NAME,
            "replaced_constraint_name": legacy_name,
            "replaced_constraint_index": resonance_index,
            "physics_delta_model_file_sha256": MODEL_FILE_SHA256,
            "physics_delta_model_payload_sha256": MODEL_PAYLOAD_SHA256,
            "raw_C_rx_rx_F_UCB_gate_installed": False,
            "single_0p759701_transfer_ratio_installed": False,
            "legacy_half_magnetizing_resonance_G_installed": False,
            "feature_extrapolation_penalty_active": True,
            "fixed20T_turn_graded_FEA_retraining_required": True,
            "turn_split_local_repair_installation": (
                split_repair_installation
            ),
            "original_split_early_hard_rejection_allowed": False,
            "selected_split_Llt_G_remains_hard": True,
            "physics_C_recomputed_for_selected_split": True,
            "constraint_names": list(effective_names),
            "screening_only": True,
            "production_eligible": False,
        }
    )
    return evidence


def corrected_resonance_contract() -> dict[str, Any]:
    return _seal_sha(
        {
            "schema_version": RESONANCE_SCHEMA,
            "primary_magnetizing_inductance_H": L_PRIMARY_H,
            "turns_ratio_N2_over_N1": 10.0,
            "secondary_inductance_H": L_SECONDARY_H,
            "capacitance_authority": (
                "turn_voltage_energy_network_plus_calibrated_delta_q90_ucb"
            ),
            "capacitance_model_file_sha256": MODEL_FILE_SHA256,
            "capacitance_model_payload_sha256": MODEL_PAYLOAD_SHA256,
            "frequency_equation": "1/(2*pi*sqrt(0.2*C_ucb))",
            "minimum_frequency_Hz": RESONANCE_MIN_HZ,
            "constraint_name": PHYSICS_CONSTRAINT_NAME,
            "effective_self_resonance_authority": (
                "physics_delta_Crx_q90_ucb_fixed_secondary_L2_0p2H"
            ),
            "effective_authority_precedence": (
                "physics_delta_rx_q90_over_all_legacy_resonance_screens"
            ),
            "raw_two_net_capacitance_used": False,
            "single_0p759701_transfer_ratio_used": False,
            "half_magnetizing_resonance_used": False,
            "interwinding_capacitance_included": False,
            "feature_extrapolation_penalty_active": True,
            "turn_split_local_repair": _turn_split_repair_contract(),
            "fixed20T_turn_graded_FEA_retraining_required": True,
            "final_turn_graded_symmetric_FEA_required": True,
            "approved_dielectric_stack_sensitivity_required": True,
            "physical_Lm_2mH_claimed_before_symmetric_FEA": False,
            "screening_only": True,
            "production_eligible": False,
        }
    )


def install_corrected_resonance(
    problem: Any,
) -> tuple[Any, dict[str, Any]]:
    if not getattr(problem, "_diagnostic_search_profile_installed", False):
        raise RuntimeError("physics-delta profile must precede resonance seal")
    if getattr(problem, "_goal_fixed_lm2mh_resonance_installed", False):
        raise RuntimeError("corrected resonance was already installed")
    if PHYSICS_CONSTRAINT_NAME not in problem.constraint_names:
        raise RuntimeError("corrected physics resonance constraint is absent")
    base_contract = copy.deepcopy(problem.hard_constraint_contract)
    base_sha = canonical_sha256(base_contract)
    if base_sha != problem.hard_constraint_contract_sha256:
        raise RuntimeError("corrected resonance base contract is unauthenticated")
    contract = corrected_resonance_contract()
    effective = copy.deepcopy(base_contract)
    effective.update(
        {
            "base_hard_constraint_contract_sha256": base_sha,
            "resonance_authority": copy.deepcopy(contract),
            "resonance_authority_sha256": contract["sha256"],
            "effective_self_resonance_authority": contract[
                "effective_self_resonance_authority"
            ],
            "effective_resonance_authority_precedence": contract[
                "effective_authority_precedence"
            ],
            "base_stage_magnetizing_inductance_factor_ignored": True,
            "core_center_gap_mm_FEA_synthesis_required": True,
            "physical_Lm_2mH_verified": False,
        }
    )
    problem.hard_constraint_contract = effective
    problem.hard_constraint_contract_sha256 = canonical_sha256(effective)
    problem._goal_fixed_lm2mh_resonance_installed = True
    problem._goal_base_hard_constraint_contract_sha256 = base_sha
    problem._goal_fixed_lm2mh_resonance_contract = contract
    evidence = _seal_sha(
        {
            "schema_version": "mft-goal-physics-delta-resonance-installation-v1",
            "base_hard_constraint_contract_sha256": base_sha,
            "effective_hard_constraint_contract_sha256": (
                problem.hard_constraint_contract_sha256
            ),
            "resonance_contract": contract,
            "resonance_contract_sha256": contract["sha256"],
            "resonance_contract_schema": RESONANCE_SCHEMA,
            "effective_self_resonance_authority": contract[
                "effective_self_resonance_authority"
            ],
            "effective_authority_precedence": contract[
                "effective_authority_precedence"
            ],
            "minimum_resonance_constraint_replaced_only": True,
            "other_physical_constraints_mutated": False,
            "raw_two_net_capacitance_G_present": False,
            "legacy_half_magnetizing_resonance_G_present": False,
            "core_center_gap_mm_FEA_synthesis_required": True,
            "physical_Lm_2mH_verified": False,
        }
    )
    return problem._evaluate, evidence


def derive_corrected_terminal_resonance(
    measurements: Mapping[str, Any],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    del measurements
    model = getattr(_ACTIVE_MODEL, "copy", lambda: None)()
    if not isinstance(model, dict):
        raise RuntimeError("terminal physics-delta model is unavailable")
    prediction = _candidate_prediction(params, model)
    f_rx = float(prediction["physics_delta_fRx_q90_lcb_Hz"])
    return {
        "f_res_tx_fixed_lm2mh_Hz": f_rx,
        "f_res_rx_fixed_lm2mh_Hz": f_rx,
        "f_res_min_tx_rx_only_Hz": f_rx,
        "f_res_tx_half_magnetizing_Hz": f_rx,
        "f_res_rx_half_magnetizing_Hz": f_rx,
        "primary_resonant_inductance_H": L_PRIMARY_H,
        "secondary_resonant_inductance_H": L_SECONDARY_H,
        "primary_magnetizing_inductance_H": L_PRIMARY_H,
        "magnetizing_inductance_factor": 1.0,
        "resonance_contract_schema": RESONANCE_SCHEMA,
        "interwinding_resonance_included": False,
        "physics_delta_Crx_mean_F": float(
            prediction["physics_delta_Crx_mean_F"]
        ),
        "physics_delta_Crx_q90_ucb_F": float(
            prediction["physics_delta_Crx_q90_ucb_F"]
        ),
        "physics_delta_fRx_q90_lcb_Hz": f_rx,
        "physics_delta_extrapolation_distance": float(
            prediction["physics_delta_extrapolation_distance"]
        ),
        "tx_resonance_alias_is_non_authoritative": True,
        "raw_two_net_C_rx_rx_F_used": False,
    }


def corrected_terminal_physicality_gate(
    predictions: Mapping[str, Mapping[str, Any]],
    *,
    population_size: int,
) -> tuple[Any, dict[str, Any]]:
    count = int(population_size)
    excluded = {"C_rx_rx_F"}
    nonnegative = tuple(
        target
        for target in preflight.TERMINAL_NONNEGATIVE_SURROGATE_TARGETS
        if target not in excluded
    )
    positive = tuple(
        target
        for target in preflight.TERMINAL_POSITIVE_SURROGATE_TARGETS
        if target not in excluded
    )
    required = set(nonnegative) | set(positive) | {"k"}
    if count < 0 or not required.issubset(predictions):
        raise RuntimeError(
            "corrected terminal physicality inventory is incomplete"
        )
    valid = np.ones(count, dtype=bool)
    violations: list[dict[str, Any]] = []

    def inspect(target: str, rule: str, predicate: Any) -> None:
        values = np.asarray(
            predictions[target].get("mean"), dtype=float
        ).reshape(-1)
        if values.shape != (count,) or not np.isfinite(values).all():
            raise RuntimeError(
                f"corrected terminal physicality values invalid: {target}"
            )
        failed = ~np.asarray(predicate(values), dtype=bool)
        valid[failed] = False
        for raw_index in np.flatnonzero(failed):
            index = int(raw_index)
            violations.append(
                {
                    "population_index": index,
                    "target": target,
                    "rule": rule,
                    "predicted_mean": float(values[index]),
                }
            )

    for target in nonnegative:
        inspect(target, "finite_and_nonnegative", lambda values: values >= 0.0)
    for target in positive:
        inspect(
            target,
            "finite_and_strictly_positive",
            lambda values: values > 0.0,
        )
    inspect(
        "k",
        "finite_and_between_zero_and_one_exclusive",
        lambda values: (values > 0.0) & (values < 1.0),
    )
    invalid = np.flatnonzero(~valid).astype(int).tolist()
    per_target: dict[str, int] = {}
    for violation in violations:
        target = str(violation["target"])
        per_target[target] = per_target.get(target, 0) + 1
    evidence = {
        "schema_version": preflight.TERMINAL_SURROGATE_PHYSICALITY_SCHEMA,
        "population_size": count,
        "valid_count": int(np.count_nonzero(valid)),
        "invalid_count": int(np.count_nonzero(~valid)),
        "invalid_population_indices": invalid,
        "violations": violations,
        "violation_count_by_target": dict(sorted(per_target.items())),
        "nonnegative_targets": list(nonnegative),
        "strictly_positive_targets": list(positive),
        "coupling_rule": "0 < k < 1",
        "raw_predictions_preserved": True,
        "clamping_performed": False,
        "invalid_candidates_marked_infeasible": True,
        "optimizer_constraint_schema_mutated": False,
        "optimizer_objectives_mutated": False,
        "excluded_non_authoritative_targets": ["C_rx_rx_F"],
        "C_rx_rx_F_terminal_eligibility_authority": False,
        "physics_delta_fRx_G_is_terminal_resonance_authority": True,
    }
    evidence["sha256"] = canonical_sha256(evidence)
    return valid, evidence


def _corrected_scheduler_payload(
    *,
    plan: Mapping[str, Any],
    task: Mapping[str, Any],
    priority: int | None = None,
) -> dict[str, Any]:
    payload = _BASE_SCHEDULER_PAYLOAD(
        plan=plan,
        task=task,
        priority=priority,
    )
    old = (
        "artifacts/code/tools/mft_goal_diagnostic_compact_scout.py "
        "execute"
    )
    new = f"{WORKER_ENTRYPOINT} execute"
    command = str(payload["command"])
    if command.count(old) != 1:
        raise RuntimeError("diagnostic worker entrypoint replacement drifted")
    payload["command"] = command.replace(old, new)
    return payload


def configure_runtime(model: Mapping[str, Any] | None = None) -> None:
    if model is not None:
        _set_active_model(model)
    scout.CAMPAIGN_ID = CAMPAIGN_ID
    scout.ACTIVATION_SCHEMA = ACTIVATION_SCHEMA
    scout.BUNDLE_SCHEMA = BUNDLE_SCHEMA
    scout.TASK_SCHEMA = TASK_SCHEMA
    scout.SCHEDULER_SCHEMA = SCHEDULER_SCHEMA
    scout.RESULT_SCHEMA = RESULT_SCHEMA
    scout.SEARCH_PROFILE_SCHEMA = SEARCH_PROFILE_SCHEMA
    scout.SEARCH_PROFILE_INSTALLATION_SCHEMA = INSTALLATION_SCHEMA
    scout._build_search_profile = _build_search_profile
    scout._validate_search_profile = _validate_search_profile
    scout._secondary_gap_mode_from_profile = _secondary_gap_mode_from_profile
    scout._install_search_profile = _install_search_profile
    preflight.GOAL_FIXED_LM_RESONANCE_SCHEMA = RESONANCE_SCHEMA
    preflight.goal_fixed_lm2mh_resonance_contract = corrected_resonance_contract
    preflight.install_goal_fixed_lm2mh_resonance = install_corrected_resonance
    preflight.derive_fixed_lm2mh_self_resonance = (
        derive_corrected_terminal_resonance
    )
    preflight._terminal_surrogate_physicality_gate = (
        corrected_terminal_physicality_gate
    )
    offload.PLAN_SCHEMA = OFFLOAD_PLAN_SCHEMA
    offload.DEPLOYMENT_SCHEMA = OFFLOAD_DEPLOYMENT_SCHEMA
    offload.AUTHENTICATION_SCHEMA = OFFLOAD_AUTH_SCHEMA
    offload.DRY_RUN_SCHEMA = OFFLOAD_DRY_RUN_SCHEMA
    offload.RECEIPT_SCHEMA = OFFLOAD_RECEIPT_SCHEMA
    offload.EXACT_SEED_START = SEED_START
    offload.EXACT_TASK_COUNT = SEED_COUNT
    offload.EXACT_SEEDS = tuple(range(SEED_START, SEED_END + 1))
    offload.TASK_NAME_PREFIX = TASK_NAME_PREFIX
    offload.DEDUPE_PREFIX = DEDUPE_PREFIX
    offload.DEFAULT_LOCAL_ROOT = DEFAULT_LOCAL_ROOT
    offload.DEFAULT_REMOTE_ROOT = DEFAULT_REMOTE_ROOT
    offload.scheduler_payload = _corrected_scheduler_payload


def _model_from_payload(path: Path) -> dict[str, Any]:
    task = _read_json(path)
    profile = (
        (task.get("activation") or {}).get("manufacturing_search_profile")
        or task.get("manufacturing_search_profile")
        or {}
    )
    gate = profile.get("physics_delta_rx_resonance_gate") or {}
    return _validate_embedded_model(gate.get("model") or {})


def _configure_from_bundle(bundle_root: Path) -> dict[str, Any]:
    bundle = _read_json(bundle_root / "bundle_manifest.json")
    relatives = bundle.get("task_relative_paths") or []
    if len(relatives) != SEED_COUNT:
        raise RuntimeError("corrected bundle does not contain exact60 tasks")
    first = (bundle_root / str(relatives[0])).resolve(strict=True)
    model = _model_from_payload(first)
    configure_runtime(model)
    return model


def _configure_from_plan(plan_path: Path) -> dict[str, Any]:
    plan = _read_json(plan_path)
    root = Path(str(plan.get("goal_bundle_root") or "")).resolve(strict=True)
    return _configure_from_bundle(root)


def _rewrite_result(path: Path) -> None:
    value = _read_json(path)
    unsigned = dict(value)
    observed = unsigned.pop("payload_sha256", None)
    if observed != canonical_sha256(unsigned):
        raise RuntimeError("corrected worker result seal mismatch")
    for key in (
        "raw_same_metric_C_rx_rx_F_UCB_gate_active",
        "provisional_turn_graded_C_acquisition_gate_active",
        "authenticated_turn_graded_transfer_ratio",
    ):
        unsigned.pop(key, None)
    unsigned.update(
        {
            "schema_version": RESULT_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "physics_delta_Crx_q90_ucb_gate_active": True,
            "physics_delta_fRx_q90_lcb_gate_active": True,
            "physics_delta_model_file_sha256": MODEL_FILE_SHA256,
            "physics_delta_model_payload_sha256": MODEL_PAYLOAD_SHA256,
            "raw_two_net_C_optimizer_objective_constraint_authority": False,
            "raw_two_net_C_terminal_eligibility_authority": False,
            "single_0p759701_transfer_ratio_used": False,
            "legacy_half_magnetizing_resonance_G_present": False,
            "feature_extrapolation_penalty_active": True,
            "turn_split_local_repair": _turn_split_repair_contract(),
            "original_split_early_hard_rejection_allowed": False,
            "selected_split_Llt_G_remains_hard": True,
            "terminal_selected_and_neighbor_splits_in_decoded_params": True,
            "fixed20T_turn_graded_FEA_retraining_required": True,
            "approved_dielectric_stack_sensitivity_required": True,
            "final_turn_graded_symmetric_FEA_required": True,
        }
    )
    goal_launch._atomic_json(path, goal_launch._seal(unsigned))


def prepare_command(args: argparse.Namespace) -> Path:
    model = _load_bound_model(args.physics_model)
    configure_runtime(model)
    delegated = argparse.Namespace(
        generation=args.generation,
        candidate=args.candidate,
        quality_status=args.quality_status,
        code_root=args.code_root,
        expected_code_revision=args.expected_code_revision,
        runtime_dataset=args.runtime_dataset,
        runtime_profile=args.runtime_profile,
        expected_documentary_generation_path=(
            args.expected_documentary_generation_path
        ),
        seed_start=SEED_START,
        seed_count=SEED_COUNT,
        primary_axial_clearance_mm=20.0,
        secondary_gap_mode=scout.SECONDARY_GAP_MODE_BOUNDED,
        output=args.output,
    )
    return scout.prepare(delegated)


def execute_command(args: argparse.Namespace) -> Path:
    model = _model_from_payload(args.payload)
    configure_runtime(model)
    path = scout.execute(args)
    _rewrite_result(path)
    return path


def plan_command(args: argparse.Namespace) -> Path:
    _configure_from_bundle(args.goal_bundle_root)
    plan, _deployment = offload.build_plan(
        goal_bundle_root=args.goal_bundle_root,
        generation=args.generation,
        candidate=args.candidate,
        quality_status=args.quality_status,
        dataset=args.dataset,
        profile=args.profile,
        local_root=args.local_root,
        remote_root=args.remote_root,
    )
    return Path(plan["local_plan_dir"]) / "offload_plan.json"


class _ReadOnlyAbsentScheduler:
    post_count = 0
    get_count = 0
    cancel_count = 0
    preempt_count = 0

    def list_namespace_tasks(self, name_prefix: str) -> list[dict[str, Any]]:
        if not name_prefix.startswith(TASK_NAME_PREFIX):
            raise RuntimeError("corrected dry-run namespace drifted")
        self.get_count += 1
        return []

    def get_task(self, task_id: int) -> dict[str, Any]:
        raise RuntimeError(f"local dry-run has no task {task_id}")

    def submit_task(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        raise RuntimeError("local dry-run cannot POST Scheduler tasks")


def _preflight_value(plan_path: Path) -> dict[str, Any]:
    _configure_from_plan(plan_path)
    plan, deployment, tasks, authentication = offload.authenticate_plan(
        plan_path
    )
    model = _active_model()
    gate = tasks[0]["manufacturing_search_profile"][
        "physics_delta_rx_resonance_gate"
    ]
    if any(
        task["seed"] != SEED_START + index
        for index, task in enumerate(tasks)
    ):
        raise RuntimeError("corrected preflight seed inventory drifted")
    return goal_launch._seal(
        {
            "schema_version": PREFLIGHT_SCHEMA,
            "created_at": _now(),
            "campaign_id": CAMPAIGN_ID,
            "plan_path": str(plan_path.resolve(strict=True)),
            "plan_file_sha256": _sha256_file(plan_path.resolve(strict=True)),
            "plan_contract_sha256": plan["contract_sha256"],
            "diagnostic_plan_sha256": plan["diagnostic_plan_sha256"],
            "deployment_contract_sha256": deployment["contract_sha256"],
            "authentication_payload_sha256": authentication["sha256"],
            "task_count": len(tasks),
            "seed_start": SEED_START,
            "seed_end_inclusive": SEED_END,
            "scheduler_priority": SCHEDULER_PRIORITY,
            "scheduler_task_name_prefix": TASK_NAME_PREFIX,
            "scheduler_dedupe_prefix": DEDUPE_PREFIX,
            "worker_entrypoint": WORKER_ENTRYPOINT,
            "size_limits_mm": {"W": 1200.0, "L": 900.0, "H": 750.0},
            "temperature_family_limits_C": {
                "primary_winding": 110.0,
                "secondary_winding": 130.0,
                "core": 130.0,
            },
            "primary_magnetizing_inductance_H": L_PRIMARY_H,
            "secondary_inductance_H": L_SECONDARY_H,
            "minimum_fRx_Hz": RESONANCE_MIN_HZ,
            "frequency_lcb_formula": gate["frequency_lcb_formula"],
            "turn_split_local_repair": _turn_split_repair_contract(),
            "original_split_early_hard_rejection_allowed": False,
            "selected_split_Llt_G_remains_hard": True,
            "terminal_selected_and_neighbor_splits_sealed": True,
            "physics_delta_model_file_sha256": MODEL_FILE_SHA256,
            "physics_delta_model_payload_sha256": model["payload_sha256"],
            "raw_two_net_C_optimizer_objective_constraint_authority": False,
            "raw_two_net_C_terminal_eligibility_authority": False,
            "single_0p759701_transfer_ratio_used": False,
            "legacy_half_magnetizing_resonance_G_present": False,
            "feature_extrapolation_penalty_active": True,
            "calibration_domain": gate["calibration_domain"],
            "fixed20T_turn_graded_FEA_retraining_required": True,
            "fixed20T_retraining_completed": False,
            "cooling_or_TIM_contract_mutated": False,
            "scheduler_project_source_included": False,
            "scheduler_project_modified": False,
            "remote_staging_performed": False,
            "scheduler_submission_performed": False,
            "scheduler_post_count": 0,
            "explicit_parent_or_root_authorization_required_before_POST": True,
            "completed_scheduler_slot_confirmation_required_before_POST": True,
            "screening_only": True,
            "production_eligible": False,
            "final_design_claim_allowed": False,
        }
    )


def preflight_command(args: argparse.Namespace) -> Path:
    value = _preflight_value(args.plan)
    if args.output.exists():
        raise RuntimeError("corrected preflight output already exists")
    _write_json(args.output, value)
    return args.output


def dry_run_command(args: argparse.Namespace) -> Path:
    _configure_from_plan(args.plan)
    plan = _read_json(args.plan)
    client = _ReadOnlyAbsentScheduler()
    base = offload.submit(
        plan_path=args.plan,
        scheduler_url="http://127.0.0.1:0",
        priority=SCHEDULER_PRIORITY,
        apply=False,
        client=client,
        remote_ready={
            "bundle_id": plan["bundle_id"],
            "bundle_manifest_sha256": plan["bundle_manifest_sha256"],
            "runtime_verified": True,
            "local_read_only_simulation": True,
        },
    )
    if (
        base.get("scheduler_post_count") != 0
        or base.get("absent_count") != SEED_COUNT
        or len(base.get("tasks") or []) != SEED_COUNT
        or any(
            not row["name"].startswith(TASK_NAME_PREFIX)
            or not row["dedupe_key"].startswith(DEDUPE_PREFIX)
            for row in base["tasks"]
        )
    ):
        raise RuntimeError("corrected local dry-run accounting mismatch")
    value = goal_launch._seal(
        {
            "schema_version": LOCAL_DRY_RUN_SCHEMA,
            "created_at": _now(),
            "campaign_id": CAMPAIGN_ID,
            "plan_path": str(args.plan.resolve(strict=True)),
            "plan_file_sha256": _sha256_file(args.plan.resolve(strict=True)),
            "base_authenticated_dry_run": base,
            "base_authenticated_dry_run_payload_sha256": base["sha256"],
            "task_count": SEED_COUNT,
            "seed_start": SEED_START,
            "seed_end_inclusive": SEED_END,
            "scheduler_priority": SCHEDULER_PRIORITY,
            "scheduler_task_name_prefix": TASK_NAME_PREFIX,
            "scheduler_dedupe_prefix": DEDUPE_PREFIX,
            "worker_entrypoint": WORKER_ENTRYPOINT,
            "local_read_only_scheduler_inventory_simulation": True,
            "remote_READY_claimed": False,
            "remote_staging_performed": False,
            "scheduler_submission_performed": False,
            "scheduler_post_count": 0,
            "explicit_parent_or_root_authorization_required_before_POST": True,
            "completed_scheduler_slot_confirmation_required_before_POST": True,
            "screening_only": True,
            "production_eligible": False,
            "final_design_claim_allowed": False,
        }
    )
    if args.output.exists():
        raise RuntimeError("corrected local dry-run output already exists")
    _write_json(args.output, value)
    return args.output


def stage_command(args: argparse.Namespace) -> str:
    _configure_from_plan(args.plan)
    value = offload.stage(
        plan_path=args.plan,
        accounts_path=args.accounts,
        scheduler_source=args.scheduler_source,
        staging_account=args.staging_account,
        resume_incoming=args.resume_incoming,
        apply=args.apply,
    )
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def submit_command(args: argparse.Namespace) -> str:
    _configure_from_plan(args.plan)
    if args.apply and not args.authorization:
        raise RuntimeError(
            "--apply requires --authorization confirming parent/root approval"
        )
    value = offload.submit(
        plan_path=args.plan,
        scheduler_url=args.scheduler_url,
        accounts_path=args.accounts,
        scheduler_source=args.scheduler_source,
        staging_account=args.staging_account,
        priority=SCHEDULER_PRIORITY,
        apply=args.apply,
        receipt_out=args.receipt_out,
    )
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--quality-status", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--expected-code-revision", required=True)
    parser.add_argument("--runtime-dataset", type=Path)
    parser.add_argument("--runtime-profile", type=Path)
    parser.add_argument("--expected-documentary-generation-path")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    _add_source_arguments(prepare)
    prepare.add_argument(
        "--physics-model", type=Path, default=DEFAULT_MODEL_PATH
    )
    prepare.add_argument("--output", type=Path, default=DEFAULT_BUNDLE_ROOT)
    prepare.set_defaults(handler=prepare_command)
    execute = commands.add_parser("execute")
    execute.add_argument("--payload", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--relocation", type=Path)
    execute.set_defaults(handler=execute_command)
    plan = commands.add_parser("plan")
    plan.add_argument("--goal-bundle-root", type=Path, required=True)
    plan.add_argument("--generation", type=Path, required=True)
    plan.add_argument("--candidate", type=Path, required=True)
    plan.add_argument("--quality-status", type=Path, required=True)
    plan.add_argument("--dataset", type=Path, required=True)
    plan.add_argument("--profile", type=Path, required=True)
    plan.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    plan.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    plan.set_defaults(handler=plan_command)
    preflight_parser = commands.add_parser("preflight")
    preflight_parser.add_argument("--plan", type=Path, required=True)
    preflight_parser.add_argument(
        "--output", type=Path, default=DEFAULT_PREFLIGHT_PATH
    )
    preflight_parser.set_defaults(handler=preflight_command)
    dry = commands.add_parser("dry-run")
    dry.add_argument("--plan", type=Path, required=True)
    dry.add_argument("--output", type=Path, default=DEFAULT_DRY_RUN_PATH)
    dry.set_defaults(handler=dry_run_command)
    for name, handler in (("stage", stage_command), ("submit", submit_command)):
        command = commands.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument(
            "--accounts", type=Path, default=offload.DEFAULT_ACCOUNTS
        )
        command.add_argument(
            "--scheduler-source",
            type=Path,
            default=offload.DEFAULT_SCHEDULER_SOURCE,
        )
        command.add_argument(
            "--staging-account", default=offload.DEFAULT_STAGING_ACCOUNT
        )
        command.add_argument("--apply", action="store_true")
        if name == "stage":
            command.add_argument("--resume-incoming")
        else:
            command.add_argument(
                "--scheduler-url", default=offload.DEFAULT_SCHEDULER_URL
            )
            command.add_argument("--receipt-out", type=Path)
            command.add_argument("--authorization")
        command.set_defaults(handler=handler)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = args.handler(args)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
