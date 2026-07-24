"""Fail-closed current-seven-temperature Tier-1 corrected-model preflight.

The historical all-11 feedback wrapper is intentionally untouched.  This
module binds the direct corrected generation to the current ``run_nsga2``
module, restores the trained cooling controls hidden by the simple current
problem, and adds every simultaneous physical hard gate omitted there.

The wrapper owns one fixed-point physics projection used for initialization,
authenticated warm starts, every pymoo offspring and terminal physical replay.
It fixes one trained primary-turn stratum (N1=5 or N1=6) and consumes cw1=5 mm
inside the winding-budget calculation rather than mutating decoded geometry.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import hashlib
import importlib
import inspect
import io
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import sys
import tempfile
from typing import Any, Mapping

try:
    from tier1_corrected_generation_adapter import (
        AuthenticatedCorrectedGeneration,
        CURRENT_REQUIRED_MODEL_TARGETS,
        CURRENT_REQUIRED_MODEL_TARGETS_SHA256,
        GOAL_REQUIRED_MODEL_TARGETS,
        GOAL_REQUIRED_MODEL_TARGETS_SHA256,
        CURRENT_STAGE_HARD_CONTRACT,
        CURRENT_STAGE_HARD_CONTRACT_SHA256,
        CURRENT_TEMPERATURE_CONTRACT,
        CURRENT_TEMPERATURE_CONTRACT_SHA256,
        CURRENT_TEMPERATURE_TARGETS,
        adapter_manifest,
        authenticate_code_root,
        authenticate_corrected_generation,
        canonical_sha256,
        process_model_cache,
        read_json,
        sha256_file,
        training_profile_sha256,
        validate_adapter_manifest,
    )
except ImportError:  # pragma: no cover - repository module import path
    from tools.tier1_corrected_generation_adapter import (
        AuthenticatedCorrectedGeneration,
        CURRENT_REQUIRED_MODEL_TARGETS,
        CURRENT_REQUIRED_MODEL_TARGETS_SHA256,
        GOAL_REQUIRED_MODEL_TARGETS,
        GOAL_REQUIRED_MODEL_TARGETS_SHA256,
        CURRENT_STAGE_HARD_CONTRACT,
        CURRENT_STAGE_HARD_CONTRACT_SHA256,
        CURRENT_TEMPERATURE_CONTRACT,
        CURRENT_TEMPERATURE_CONTRACT_SHA256,
        CURRENT_TEMPERATURE_TARGETS,
        adapter_manifest,
        authenticate_code_root,
        authenticate_corrected_generation,
        canonical_sha256,
        process_model_cache,
        read_json,
        sha256_file,
        training_profile_sha256,
        validate_adapter_manifest,
    )

from module.mft_goal_20260726_contract import (
    FIXED_COOLING_IDENTITY as GOAL_FIXED_COOLING_IDENTITY,
    FIXED_COOLING_IDENTITY_SHA256 as GOAL_FIXED_COOLING_IDENTITY_SHA256,
    FIXED_OPERATING_IDENTITY as GOAL_FIXED_OPERATING_IDENTITY,
    FIXED_OPERATING_IDENTITY_SHA256 as GOAL_FIXED_OPERATING_IDENTITY_SHA256,
    GOAL_CONTRACT_SCHEMA,
    GOAL_CW1_MAX_MM,
    GOAL_CW1_MIN_MM,
    GOAL_CW1_STEP_MM,
    GOAL_G0_MODEL_TARGETS,
    GOAL_N_CORE_GROUP_MAX,
    GOAL_N_CORE_GROUP_MIN,
    GOAL_PRIMARY_TURN_STRATA,
    GOAL_RESONANCE_MIN_HZ,
    GOAL_SIZE_LIMITS_MM,
    GOAL_STAGE_SPEC,
    GOAL_STAGE_SPEC_SHA256,
    GOAL_TERMINAL_TABLE_SCHEMA,
    GOAL_TEMPERATURE_CONTRACT,
    GOAL_TEMPERATURE_CONTRACT_SHA256,
    GOAL_TEMPERATURE_TARGETS,
    CORE_TEMPERATURE_TARGETS,
    TEMPERATURE_TARGET_LIMITS_C,
    WINDING_TEMPERATURE_TARGETS,
    attest_fixed_identity,
    canonical_sha256 as goal_canonical_sha256,
    cw1_from_unit_coordinate,
    cw1_unit_coordinate,
    dynamic_core_group_bounds,
    dynamic_core_group_violation,
    is_goal_stage_spec,
    temperature_limit_for_target,
    validate_cw1_mm,
    validate_goal_stage_spec,
)


RECEIPT_SCHEMA = "mft-tier1-corrected-generation-smoke-receipt-v2"
GOAL_RECEIPT_SCHEMA = "mft-goal-20260726-g0-smoke-receipt-v1"
RUNNER_SCHEMA = "mft-tier1-current7-corrected-runner-v2"
GOAL_RUNNER_SCHEMA = "mft-goal-20260726-g0-runner-v1"
GOAL_CODE_MANIFEST_SCHEMA = "mft-goal-20260726-code-inventory-v1"
PROBLEM_SCHEMA = "mft-tier1-current7-hard-problem-v2"
OPTIMIZER_REPAIR_SCHEMA = "mft-tier1-current7-physics-repair-v1"
PINNED_PROJECTION_SOURCE_REVISION = "7c832f7f78f92ee2d99b2d37e14c3131f07d9cae"
SUPPORTED_FIXED_PRIMARY_TURNS = (5, 6)
GOAL_SUPPORTED_PRIMARY_TURNS = GOAL_PRIMARY_TURN_STRATA
FIXED_GENERATION_TERMINATION_STRATEGY = "fixed-n-gen-no-ftol-v1"
PRODUCTION_FIXED_GENERATIONS = 300
PRODUCTION_POPULATION = 320
PRODUCTION_INFERENCE_THREADS = 8
DETERMINISTIC_TERMINAL_INFERENCE_THREADS = 1
SKLEARN_FOREST_SEMAPHORE_FREE_FAMILIES = {
    "extratrees", "randomforest",
}
FAMILY_SPECIFIC_INFERENCE_POLICY = (
    "family_specific_semaphore_free_sklearn_forest_v1"
)
REPEATED_PREDICT_STRESS_SCHEMA = "mft-tier1-current7-semlock-free-repeated-predict-v1"
REPEATED_PREDICT_STRESS_REPEATS = 8
PHASE_B_CHILD_PROTOCOL = "final1000-finite-multiseed-phase-b-v1"
PHASE_B_CHILD_PROTOCOL_ENV = "MFT_FINAL1000_PHASE_B_CHILD_PROTOCOL"
SEMLOCK_SAFE_RELEASE_COMMIT = "6ea0e17e5e028ebb8d89d910c3cec1a0f014dae5"
SEMLOCK_SAFE_SMOKE_SCHEMA = "mft-tier1-semlock-safe-inference-smoke-v1"
SEMLOCK_SAFE_HELPER_RELATIVE = "tools/tier1_semlock_safe_inference_smoke.py"
SEMLOCK_SAFE_HELPER_SHA256 = (
    "c25afd39752a990c31e1b934c77282d88ac11dab1cad763d4f07c95e4b87dd80"
)
REMOTE_PREFLIGHT_SCHEMA = "mft-tier1-current7-remote-model-load-v1"
SEARCH_RESULT_SCHEMA = "mft-tier1-current7-search-seed-v1"
WARM_ROLE_PARTITION_SCHEMA = "mft-tier1-authenticated-warm-role-partition-v1"
BIG = 1e6

BASE_CONSTRAINT_NAMES = (
    "Llt_robust_band",
    *(f"temperature_robust_limit:{target}" for target in CURRENT_TEMPERATURE_TARGETS),
    "analytical_flux_density_limit",
    "decoded_space_shrink",
    "secondary_vertical_insulation",
    "strict_full_density_support",
    "Llt_ensemble_disagreement",
)
ADDITIVE_HARD_CONSTRAINT_PREFIX_NAMES = (
    "minimum_physical_insulation",
    "core_group_manufacturability_limit",
)
GOAL_BASE_CONSTRAINT_NAMES = (
    "Llt_robust_band",
    *(f"temperature_robust_limit:{target}" for target in GOAL_TEMPERATURE_TARGETS),
    "analytical_flux_density_limit",
    "decoded_space_shrink",
    "secondary_vertical_insulation",
    "strict_full_density_support",
    "Llt_ensemble_disagreement",
)
GOAL_ADDITIVE_HARD_CONSTRAINT_PREFIX_NAMES = (
    "minimum_physical_insulation",
    "core_group_dynamic_validity",
)
RESONANCE_MINIMUM_CONSTRAINT = "half_magnetizing_resonance_minimum"
RESONANCE_MAXIMUM_CONSTRAINT = "half_magnetizing_resonance_maximum"
SIZE_CONSTRAINT_NAMES = (
    "exterior_width_limit",
    "exterior_length_limit",
    "exterior_height_limit",
)
SIDE_TEMPERATURE_TARGET = "Tprobe_Rx_side_leeward_max"
GOAL_SIDE_TEMPERATURE_TARGETS = (
    "T_max_Rx_side",
    SIDE_TEMPERATURE_TARGET,
)
STRUCTURAL_DONOR_REQUIRED_GATES = (
    "decoder",
    "minimum_physical_insulation",
    "core_group",
    "winding_budget_identity",
    "fixed_primary_turns",
)
STRUCTURAL_DONOR_OPTIMIZER_GATES = (
    "shrink",
    "box",
    "analytical_B",
)

# These surrogate means represent passive physical quantities.  Tree/boosting
# ensembles trained in log1p space can extrapolate to a small negative value
# after expm1, even though every training observation is non-negative.  Such a
# prediction must never be clamped into apparent feasibility.  The terminal
# gate below quarantines the candidate and preserves the raw prediction.
TERMINAL_NONNEGATIVE_SURROGATE_TARGETS = (
    "P_winding_total",
    "P_core_total",
    "P_core_plate_total",
    "P_wcp_total",
    "P_Tx_main_group",
    "P_Rx_main_group",
    "P_Rx_side_total",
    "B_mean_core",
)
TERMINAL_POSITIVE_SURROGATE_TARGETS = (
    "Llt_phys",
    "C_tx_tx_F",
    "C_rx_rx_F",
    "C_tx_rx_F",
)
TERMINAL_SURROGATE_PHYSICALITY_SCHEMA = (
    "mft-tier1-current7-terminal-surrogate-physicality-v1"
)

# These surrogate means represent passive physical quantities.  Tree/boosting
# ensembles trained in log1p space can extrapolate to a small negative value
# after expm1, even though every training observation is non-negative.  Such a
# prediction must never be clamped into apparent feasibility.  The terminal
# gate below quarantines the candidate and preserves the raw prediction.
TERMINAL_NONNEGATIVE_SURROGATE_TARGETS = (
    "P_winding_total",
    "P_core_total",
    "P_core_plate_total",
    "P_wcp_total",
    "P_Tx_main_group",
    "P_Rx_main_group",
    "P_Rx_side_total",
    "B_mean_core",
)
TERMINAL_POSITIVE_SURROGATE_TARGETS = (
    "Llt_phys",
    "C_tx_tx_F",
    "C_rx_rx_F",
    "C_tx_rx_F",
)
TERMINAL_SURROGATE_PHYSICALITY_SCHEMA = (
    "mft-tier1-current7-terminal-surrogate-physicality-v1"
)

CURRENT_STAGE_SPEC = {
    "Llt_target_uH": 27.5,
    "Llt_tol_uH": 0.55,
    "T_limit_C": 110.0,
    "B_limit_T": 1.2,
    "insulation_min_mm": 40.0,
    "q_sigma": 1.0,
    "n_core_group_max": 4.0,
    "primary_conductor_thickness_mm": 5.0,
    "resonance_min_Hz": 15_000.0,
    "resonance_max_Hz": None,
    "magnetizing_inductance_factor": 0.5,
    "size_W_max_mm": 1_200.0,
    "size_L_max_mm": 1_200.0,
    "size_H_max_mm": 750.0,
}
CURRENT_STAGE_SPEC_SHA256 = canonical_sha256(CURRENT_STAGE_SPEC)
STAGED_SPEC_OPTIONAL_KEYS = frozenset()
STAGED_SPEC_MUTABLE_KEYS = frozenset(
    {
        "T_limit_C",
        "resonance_min_Hz",
        "resonance_max_Hz",
        "size_W_max_mm",
        "size_L_max_mm",
        "size_H_max_mm",
    }
)
STAGED_HARD_CONTRACT_SCHEMA = "mft-tier1-staged-hard-constraint-contract-v1"


def _stage_positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be finite and positive")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{label} must be finite and positive") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise RuntimeError(f"{label} must be finite and positive")
    return number


def validate_stage_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical staged hard spec or fail closed.

    Geometry, robust temperature and self-resonance limits are the only stage
    axes.  Manufacturing, leakage, flux and magnetizing-inductance semantics
    remain pinned to the authenticated current-seven problem.
    """

    if is_goal_stage_spec(spec):
        return validate_goal_stage_spec(spec)
    if not isinstance(spec, Mapping):
        raise RuntimeError("Tier-1 stage spec must be an object")
    supplied = dict(spec)
    required = set(CURRENT_STAGE_SPEC)
    allowed = required | set(STAGED_SPEC_OPTIONAL_KEYS)
    if set(supplied) - allowed or required - set(supplied):
        raise RuntimeError("Tier-1 stage spec has missing or unknown keys")
    normalized: dict[str, Any] = {}
    for name in CURRENT_STAGE_SPEC:
        value = supplied[name]
        if name in {"resonance_min_Hz", "resonance_max_Hz"} and value is None:
            normalized[name] = None
        else:
            normalized[name] = _stage_positive_number(value, f"stage spec {name}")
    maximum = supplied.get("resonance_max_Hz")
    normalized["resonance_max_Hz"] = (
        None
        if maximum is None
        else _stage_positive_number(maximum, "stage spec resonance_max_Hz")
    )
    for name, expected in CURRENT_STAGE_SPEC.items():
        if name in STAGED_SPEC_MUTABLE_KEYS:
            continue
        if not math.isclose(
            float(normalized[name]), float(expected), rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(f"Tier-1 staged search forbids overriding {name}")
    minimum = normalized["resonance_min_Hz"]
    maximum = normalized["resonance_max_Hz"]
    if minimum is None and maximum is None:
        raise RuntimeError("Tier-1 stage requires a resonance lower or upper bound")
    if minimum is not None and maximum is not None and minimum >= maximum:
        raise RuntimeError("Tier-1 resonance lower bound must be below upper bound")
    return normalized


def stage_constraint_names(spec: Mapping[str, Any]) -> tuple[str, ...]:
    normalized = validate_stage_spec(spec)
    if is_goal_stage_spec(normalized):
        return (
            GOAL_BASE_CONSTRAINT_NAMES
            + GOAL_ADDITIVE_HARD_CONSTRAINT_PREFIX_NAMES
            + (RESONANCE_MINIMUM_CONSTRAINT,)
            + SIZE_CONSTRAINT_NAMES
        )
    resonance = []
    if normalized["resonance_min_Hz"] is not None:
        resonance.append(RESONANCE_MINIMUM_CONSTRAINT)
    if normalized["resonance_max_Hz"] is not None:
        resonance.append(RESONANCE_MAXIMUM_CONSTRAINT)
    return (
        BASE_CONSTRAINT_NAMES
        + ADDITIVE_HARD_CONSTRAINT_PREFIX_NAMES
        + tuple(resonance)
        + SIZE_CONSTRAINT_NAMES
    )


def stage_spec_from_json_identity(encoded: str, expected_sha256: str) -> dict[str, Any]:
    try:
        value = json.loads(encoded)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("staged hard spec is not valid JSON") from exc
    normalized = validate_stage_spec(value)
    if canonical_sha256(normalized) != str(expected_sha256):
        raise RuntimeError("staged hard-spec SHA-256 mismatch")
    return normalized


def half_magnetizing_resonance_band_violations(
    frequency_hz: Any, spec: Mapping[str, Any]
) -> dict[str, float]:
    """Return authoritative physical ``G <= 0`` values for a staged band."""

    normalized = validate_stage_spec(spec)
    if is_goal_stage_spec(normalized):
        return _resonance_band_violations(
            frequency_hz,
            normalized["resonance_min_Hz"],
            None,
        )
    return _resonance_band_violations(
        frequency_hz,
        normalized["resonance_min_Hz"],
        normalized["resonance_max_Hz"],
    )


def _resonance_band_violations(
    frequency_hz: Any,
    minimum_hz: float | None,
    maximum_hz: float | None,
) -> dict[str, float]:
    frequency = _stage_positive_number(
        frequency_hz, "half-magnetizing self-resonance frequency"
    )
    violations: dict[str, float] = {}
    if minimum_hz is not None:
        violations[RESONANCE_MINIMUM_CONSTRAINT] = float(minimum_hz) - frequency
    if maximum_hz is not None:
        strict_maximum = math.nextafter(float(maximum_hz), -math.inf)
        violations[RESONANCE_MAXIMUM_CONSTRAINT] = frequency - strict_maximum
    return violations


def stage_temperature_contract(spec: Mapping[str, Any]) -> dict[str, Any]:
    normalized = validate_stage_spec(spec)
    if is_goal_stage_spec(normalized):
        return copy.deepcopy(GOAL_TEMPERATURE_CONTRACT)
    if math.isclose(
        normalized["T_limit_C"],
        CURRENT_STAGE_SPEC["T_limit_C"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return copy.deepcopy(CURRENT_TEMPERATURE_CONTRACT)
    contract = copy.deepcopy(CURRENT_TEMPERATURE_CONTRACT)
    limit = float(normalized["T_limit_C"])
    contract.update(
        {
            "schema_version": "mft-tier1-current7-temperature-contract-v2",
            "semantic_version": (
                f"current7-robust-q90-half-width-staged-{limit:g}C-v1"
            ),
            "robust_upper_bound_C": limit,
        }
    )
    return contract


def stage_hard_constraint_contract(spec: Mapping[str, Any]) -> dict[str, Any]:
    normalized = validate_stage_spec(spec)
    if is_goal_stage_spec(normalized):
        return {
            "schema_version": "mft-goal-20260726-hard-constraint-contract-v1",
            "stage": "mft-goal-20260726",
            "goal_contract_schema": GOAL_CONTRACT_SCHEMA,
            "stage_spec_sha256": GOAL_STAGE_SPEC_SHA256,
            "temperature_contract": copy.deepcopy(
                GOAL_TEMPERATURE_CONTRACT
            ),
            "temperature_contract_sha256": (
                GOAL_TEMPERATURE_CONTRACT_SHA256
            ),
            "self_resonance": {
                "operator": ">=",
                "minimum_Hz": GOAL_RESONANCE_MIN_HZ,
                "maximum_Hz": None,
                "aggregation": "min(f_res_tx_self_Hz,f_res_rx_self_Hz)",
                "magnetizing_inductance_factor": normalized[
                    "magnetizing_inductance_factor"
                ],
            },
            "size_limits_mm": copy.deepcopy(GOAL_SIZE_LIMITS_MM),
            "cw1_search_mm": copy.deepcopy(normalized["cw1_search_mm"]),
            "n_core_group_search": copy.deepcopy(
                normalized["n_core_group_search"]
            ),
            "primary_turn_search": copy.deepcopy(
                normalized["primary_turn_search"]
            ),
            "fixed_operating_identity": copy.deepcopy(
                normalized["fixed_operating_identity"]
            ),
            "fixed_cooling_identity": copy.deepcopy(
                normalized["fixed_cooling_identity"]
            ),
            "constraint_names": list(stage_constraint_names(normalized)),
            "portability": {
                "mode": "goal-g0-authenticated-bundle-v1",
                "campaign_id": "mft-goal-20260726",
                "legacy_current7_bundle_identity_reused": False,
                "source_evidence_sha256_required": True,
            },
            "legacy_scalar_temperature_limit_allowed": False,
            "legacy_fixed_cw1_allowed": False,
            "legacy_core_group_ceiling_four_allowed": False,
        }
    if normalized == validate_stage_spec(CURRENT_STAGE_SPEC):
        return copy.deepcopy(CURRENT_STAGE_HARD_CONTRACT)
    minimum = normalized["resonance_min_Hz"]
    maximum = normalized["resonance_max_Hz"]
    resonance: dict[str, Any] = {
        "aggregation": "min(f_res_tx_self_Hz,f_res_rx_self_Hz)",
        "magnetizing_inductance_factor": normalized["magnetizing_inductance_factor"],
        "interwinding_resonance_is_not_part_of_this_band": True,
        "lower_bound_inclusive_Hz": minimum,
        "upper_bound_exclusive_Hz": maximum,
        "minimum_violation_formula": (
            None
            if minimum is None
            else "resonance_min_Hz-min(f_res_tx_half_Lm_Hz,f_res_rx_half_Lm_Hz)"
        ),
        "maximum_violation_formula": (
            None
            if maximum is None
            else (
                "min(f_res_tx_half_Lm_Hz,f_res_rx_half_Lm_Hz)-"
                "nextafter(resonance_max_Hz,-inf)"
            )
        ),
    }
    contract = copy.deepcopy(CURRENT_STAGE_HARD_CONTRACT)
    contract.update(
        {
            "schema_version": STAGED_HARD_CONTRACT_SCHEMA,
            "stage": f"staged-{canonical_sha256(normalized)[:16]}",
            "temperature_limit_C": normalized["T_limit_C"],
            "self_resonance": resonance,
            "size_limits_mm": {
                "W": normalized["size_W_max_mm"],
                "L": normalized["size_L_max_mm"],
                "H": normalized["size_H_max_mm"],
            },
            "constraint_names": list(stage_constraint_names(normalized)),
            "stage_spec_sha256": canonical_sha256(normalized),
        }
    )
    return contract


ADDITIVE_HARD_CONSTRAINT_NAMES = (
    ADDITIVE_HARD_CONSTRAINT_PREFIX_NAMES
    + (RESONANCE_MINIMUM_CONSTRAINT,)
    + SIZE_CONSTRAINT_NAMES
)
CURRENT7_CONSTRAINT_NAMES = stage_constraint_names(CURRENT_STAGE_SPEC)
GOAL_CONSTRAINT_NAMES = stage_constraint_names(GOAL_STAGE_SPEC)

EXPECTED_SIMPLE_BASE_FIXED_STACK_MM = {
    "core_plate_t": 20.0,
    "wcp_t": 20.0,
    "core_plate_pad_t": 2.0,
    "wcp_pad_t": 2.0,
}
VARIABLE_COOLING_DIMENSIONS = (
    "core_plate_t",
    "wcp_t",
    "wcp_len_pct",
)
FIXED_COOLING_PADS_MM = {
    "core_plate_pad_t": 2.0,
    "wcp_pad_t": 2.0,
}

PHYSICAL_INSULATION_COLUMNS = (
    "cc_w2c_space_x",
    "cc_w2c_space_y",
    "w2c_w1c_space_x",
    "w2c_w1c_space_y",
    "w1c_w2s_gap_x_actual",
    "w1s_cs_space_x",
    "cs_w1s_space_y",
    "h_gap2",
)
SIDE_PHYSICAL_INSULATION_COLUMNS = (
    "w2s_w1s_space_x",
    "w1s_w2s_space_y",
)
DECODED_GEOMETRY_IDENTITY_COLUMNS = (
    "N1_main",
    "N1_side",
    "N2_main",
    "N2_side",
    "l1",
    "l2",
    "h1",
    "w1",
    "n_core_group",
    "core_plate_t",
    "core_plate_pad_t",
    "wcp_t",
    "wcp_pad_t",
    "wcp_len_x",
    "cw1",
    "gap1",
    "cw2",
    "gap2",
    "nwh1",
    "nwh2",
    "cc_w2c_space_x",
    "cc_w2c_space_y",
    "w2c_w1c_space_x",
    "w2c_w1c_space_y",
    "w1c_w2s_gap_x_actual",
    "w2s_w1s_space_x",
    "w1s_w2s_space_y",
    "w1s_cs_space_x",
    "cs_w1s_space_y",
)
MAX_REPAIR_FIXED_POINT_ITERATIONS = 12


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise RuntimeError(f"{label} must be finite")
    return number


def _positive_number(value: Any, label: str) -> float:
    number = _finite_number(value, label)
    if number <= 0.0:
        raise RuntimeError(f"{label} must be positive")
    return number


def _row_value(row: Any, name: str) -> Any:
    try:
        return row[name]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"decoded row is missing {name}") from None


def _frame_row(frame: Any, index: int) -> Any:
    if hasattr(frame, "iloc"):
        return frame.iloc[index]
    return frame[index]


def minimum_physical_insulation_violation(
    frame: Any,
    minimum_mm: float,
    *,
    invalid_value: float = BIG,
) -> Any:
    """Evaluate every realized clearance, including conditional side gaps."""

    import numpy as np

    minimum = _positive_number(minimum_mm, "minimum physical insulation")
    invalid = _positive_number(invalid_value, "invalid constraint value")
    result = np.full(len(frame), invalid, dtype=float)
    for index in range(len(frame)):
        row = _frame_row(frame, index)
        try:
            observed = [
                _finite_number(_row_value(row, name), name)
                for name in PHYSICAL_INSULATION_COLUMNS
            ]
            n1_side = _finite_number(_row_value(row, "N1_side"), "N1_side")
            if n1_side < 0.0:
                raise RuntimeError("N1_side must be non-negative")
            if n1_side > 0.0:
                observed.extend(
                    _finite_number(_row_value(row, name), name)
                    for name in SIDE_PHYSICAL_INSULATION_COLUMNS
                )
            result[index] = max(minimum - value for value in observed)
        except RuntimeError:
            result[index] = invalid
    return result


def apply_current7_side_temperature_condition(
    violation: Any,
    frame: Any,
    *,
    invalid_value: float = BIG,
) -> Any:
    """Apply finite-N2-side activation to the one current side target."""

    import numpy as np

    values = np.asarray(violation, dtype=float).reshape(-1)
    if len(values) != len(frame):
        raise RuntimeError("side-temperature violation/frame length mismatch")
    result = np.full(len(values), float(invalid_value), dtype=float)
    for index, value in enumerate(values):
        try:
            side_turns = _finite_number(
                _row_value(_frame_row(frame, index), "N2_side"),
                "N2_side",
            )
            if side_turns < 0.0:
                raise RuntimeError("N2_side must be non-negative")
            result[index] = value if side_turns > 0.0 else -float(invalid_value)
        except RuntimeError:
            result[index] = float(invalid_value)
    return result


def fixed_primary_turn_unit_coordinate(
    fixed_primary_turns: int,
    *,
    minimum_turns: int,
    maximum_turns: int,
) -> float:
    if (
        isinstance(fixed_primary_turns, bool)
        or int(fixed_primary_turns) != fixed_primary_turns
    ):
        raise ValueError("fixed primary turns must be an integer")
    fixed = int(fixed_primary_turns)
    minimum = int(minimum_turns)
    maximum = int(maximum_turns)
    if not minimum <= fixed <= maximum:
        raise ValueError(
            "corrected Tier-1 primary-turn stratum is outside decoder bounds"
        )
    scale = maximum - minimum + 0.9999
    return float(np_clip((fixed - minimum + 0.5) / scale, 0.0, 1.0))


def np_clip(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), float(lower)), float(upper))


def decode_unit_sample_with_cw1(
    input_parameter_module: Any,
    sample: Mapping[str, Any],
    *,
    selected_cw1_mm: float,
    allow_space_shrink: bool,
    space_min: float,
) -> dict[str, Any]:
    """Consume one selected physical cw1 inside the canonical winding budget.

    The current decoder has no physical ``cw1_mm`` argument.  Its unfixed branch
    computes ``nwl1 = budget * f1_split`` and derives cw1 from that pack.  Set
    the latent split to the exact selected-cw1 primary pack before invoking it;
    every downstream secondary pack and realized clearance is then calculated
    by the current decoder from the same physical budget.  No decoded field is
    overwritten afterwards.  The caller may choose a fixed value or the goal
    campaign's 1.00..10.00 mm/0.01 mm decision grid; ``f1_split`` is never an
    independent decision in either mode.
    """

    selected = _positive_number(selected_cw1_mm, "selected cw1")
    maximum = _positive_number(
        input_parameter_module.PRIMARY_CONDUCTOR_MAX_THICKNESS_MM,
        "primary conductor maximum",
    )
    if selected > maximum:
        raise RuntimeError("selected cw1 exceeds the current conductor limit")
    effective = dict(sample)
    space_names = (
        "cc_w2c_space_x",
        "w2c_w1c_space_x",
        "w1c_w2s_space_x",
        "w1s_cs_space_x",
        "cc_w2c_space_y",
        "w2c_w1c_space_y",
        "cs_w1s_space_y",
        "w2s_w1s_space_x",
        "w1s_w2s_space_y",
    )
    for name in space_names:
        if name in effective:
            effective[name] = max(float(effective[name]), float(space_min))
    minimum_turns = int(input_parameter_module.N1_MIN_TURNS)
    maximum_turns = int(input_parameter_module.N1_MAX_TURNS)
    n1 = minimum_turns + int(
        float(effective["u_N1"]) * (maximum_turns - minimum_turns + 0.9999)
    )
    n1_side = 0
    n1_main = n1 - n1_side
    l1 = round(float(effective["l1"]))
    l2 = (round(float(effective["total_length"])) - 4 * l1) / 2.0
    x_spaces = (
        float(effective["cc_w2c_space_x"]),
        float(effective["w2c_w1c_space_x"]),
        float(effective["w1c_w2s_space_x"]),
        float(effective["w1s_cs_space_x"]),
    )
    budget = l2 - sum(x_spaces)
    if not math.isfinite(budget) or abs(budget) <= 1e-15:
        raise RuntimeError("fixed-cw1 decoder has no finite winding budget")
    gap1 = round(float(effective["gap1"]), 1)
    primary_gap_count = max(n1_main - 1, 0) + max(n1_side - 1, 0)
    primary_pack = n1 * selected + primary_gap_count * gap1
    effective["f1_split"] = primary_pack / budget
    decoded = input_parameter_module.decode_unit_sample(
        effective,
        allow_space_shrink=allow_space_shrink,
        space_min=space_min,
    )
    if not math.isclose(
        _finite_number(decoded.get("cw1"), "decoded cw1"),
        selected,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("selected cw1 escaped the winding-budget decoder")
    return decoded


def decode_unit_sample_with_fixed_cw1(
    input_parameter_module: Any,
    sample: Mapping[str, Any],
    *,
    fixed_cw1_mm: float,
    allow_space_shrink: bool,
    space_min: float,
) -> dict[str, Any]:
    """Backward-compatible current7 fixed-cw1 entry point."""

    return decode_unit_sample_with_cw1(
        input_parameter_module,
        sample,
        selected_cw1_mm=fixed_cw1_mm,
        allow_space_shrink=allow_space_shrink,
        space_min=space_min,
    )


def winding_budget_identity(row: Any, *, expected_cw1_mm: float) -> dict[str, Any]:
    """Attest primary, secondary and complete x-space pack identities."""

    try:
        values = {
            name: _finite_number(_row_value(row, name), name)
            for name in (
                "N1_main",
                "N1_side",
                "N2_main",
                "N2_side",
                "cw1",
                "gap1",
                "cw2",
                "gap2",
                "nwl1_main",
                "nwl1_side",
                "nwl2_main",
                "nwl2_side",
                "l2",
                "cc_w2c_space_x",
                "w2c_w1c_space_x",
                "w1c_w2s_gap_x_actual",
                "w1s_cs_space_x",
                "w2s_w1s_space_x",
            )
        }
    except RuntimeError:
        return {"passed": False, "reason": "missing_or_nonfinite_budget_field"}

    def pack(turns: float, conductor: float, gap: float) -> float:
        if turns < 0.0 or not float(turns).is_integer():
            raise RuntimeError("turn count is not a non-negative integer")
        return turns * conductor + max(turns - 1.0, 0.0) * gap

    try:
        expected = {
            "nwl1_main": pack(values["N1_main"], values["cw1"], values["gap1"]),
            "nwl1_side": pack(values["N1_side"], values["cw1"], values["gap1"]),
            "nwl2_main": pack(values["N2_main"], values["cw2"], values["gap2"]),
            "nwl2_side": pack(values["N2_side"], values["cw2"], values["gap2"]),
        }
    except RuntimeError:
        return {"passed": False, "reason": "invalid_turn_count"}
    pack_deltas = {
        name: values[name] - expected_value for name, expected_value in expected.items()
    }
    side_stack = values["w1s_cs_space_x"] + values["nwl2_side"]
    if values["N1_side"] > 0.0:
        side_stack += values["nwl1_side"] + values["w2s_w1s_space_x"]
    reconstructed_l2 = (
        values["cc_w2c_space_x"]
        + values["nwl2_main"]
        + values["w2c_w1c_space_x"]
        + values["nwl1_main"]
        + values["w1c_w2s_gap_x_actual"]
        + side_stack
    )
    space_delta = values["l2"] - reconstructed_l2
    passed = (
        math.isclose(values["cw1"], float(expected_cw1_mm), rel_tol=0.0, abs_tol=1e-12)
        and all(abs(delta) <= 1e-9 for delta in pack_deltas.values())
        and abs(space_delta) <= 1e-9
    )
    return {
        "passed": bool(passed),
        "cw1_mm": values["cw1"],
        "expected_cw1_mm": float(expected_cw1_mm),
        "pack_deltas_mm": pack_deltas,
        "complete_x_space_delta_mm": space_delta,
        "post_decode_field_override_performed": False,
    }


def expected_problem_cw1_mm(problem: Any, row: Any) -> float:
    """Return the exact cw1 authority for one decoded problem row."""

    if getattr(problem, "goal_campaign", False):
        return validate_cw1_mm(_row_value(row, "cw1"))
    return _positive_number(
        problem.spec["primary_conductor_thickness_mm"],
        "fixed primary conductor thickness",
    )


def optimizer_repair_contract(
    *,
    fixed_primary_turns: int,
    coordinate_index: int,
    fixed_unit_coordinate: float,
    goal_campaign: bool = False,
) -> dict[str, Any]:
    value = {
        "schema_version": OPTIMIZER_REPAIR_SCHEMA,
        "projection_source_revision": PINNED_PROJECTION_SOURCE_REVISION,
        "projection_semantics": (
            "old_7c_hard_physics_fixed_point_adapted_to_current7_targets"
        ),
        "maximum_fixed_point_iterations": MAX_REPAIR_FIXED_POINT_ITERATIONS,
        "fixed_primary_turns": int(fixed_primary_turns),
        "coordinate_name": "u_N1",
        "coordinate_index": int(coordinate_index),
        "fixed_unit_coordinate": float(fixed_unit_coordinate),
        **(
            {
                "cw1_search_mm": {
                    "minimum": GOAL_CW1_MIN_MM,
                    "maximum": GOAL_CW1_MAX_MM,
                    "step": GOAL_CW1_STEP_MM,
                    "coordinate_name": "f1_split",
                },
                "f1_split_independent_search_allowed": False,
                "cw1_enforcement": (
                    "selected_grid_value_inside_decoder_winding_budget"
                ),
            }
            if goal_campaign
            else {
                "fixed_cw1_mm": 5.0,
                "cw1_enforcement": "inside_decoder_winding_budget",
            }
        ),
        "winding_budget_identity_required": True,
        "required_stages": [
            "initial_population",
            "authenticated_warm_start",
            "every_pymoo_offspring",
            "terminal_unscaled_physical_replay",
        ],
        "authoritative_terminal_G": "physical_unscaled_replay",
        "variable_cooling_dimensions": list(VARIABLE_COOLING_DIMENSIONS),
        "hard_constraint_mutation": False,
        "objective_mutation": False,
    }
    return {
        "contract": value,
        "contract_sha256": canonical_sha256(value),
    }


def deep_topology_contract(
    fixed_primary_turns: int,
    *,
    enable_final1000_topology_niche: bool = False,
    goal_campaign: bool = False,
) -> dict[str, Any]:
    """Load and self-authenticate the tracked deep-crossover contract."""

    if goal_campaign:
        if enable_final1000_topology_niche:
            raise ValueError("goal campaign does not reuse the Final1000 niche")
        return goal_topology_contract(fixed_primary_turns)
    try:
        from tier1_deep_crossover_contract import topology_evolution_contract
    except ImportError:  # pragma: no cover - repository module import path
        from tools.tier1_deep_crossover_contract import topology_evolution_contract

    value = topology_evolution_contract(
        int(fixed_primary_turns),
        enable_final1000_topology_niche=enable_final1000_topology_niche,
    )
    sealed = dict(value)
    expected = sealed.pop("sha256", None)
    if (
        expected != canonical_sha256(sealed)
        or value.get("fixed_primary_turns") != int(fixed_primary_turns)
        or value.get("ftol_early_stop_allowed") is not False
        or value.get("terminal_physical_replay_required") is not True
    ):
        raise RuntimeError("deep-crossover topology contract authentication failed")
    return value


def goal_topology_contract(fixed_primary_turns: int) -> dict[str, Any]:
    """Return a campaign-local topology contract for every N1=5..8 stratum."""

    turns = int(fixed_primary_turns)
    if turns not in GOAL_SUPPORTED_PRIMARY_TURNS:
        raise ValueError("goal topology requires an N1 stratum from 5 through 8")
    total = 10 * turns
    side_topologies = tuple(
        sorted(
            {
                int(round(total * fraction))
                for fraction in (0.56, 0.58, 0.60, 0.62, 0.64, 0.66)
            }
        )
    )
    topologies = (*side_topologies, total)
    pairs = tuple(
        (topologies[index], topologies[(index + 3) % len(topologies)])
        for index in range(len(topologies))
    )
    minimum_survivors = 4
    protected_slots = minimum_survivors * len(topologies)
    donor_lanes = {
        "goal_turn_split": {
            "topologies_N2_main_N2_side": [
                [topology, total - topology] for topology in topologies
            ],
            "selection_signal": "goal_campaign_geometry_and_thermal_diversity",
        }
    }
    value = {
        "schema_version": "mft-goal-20260726-turn-split-evolution-v1",
        "goal_contract_schema": GOAL_CONTRACT_SCHEMA,
        "goal_stage_spec_sha256": GOAL_STAGE_SPEC_SHA256,
        "fixed_primary_turns": turns,
        "secondary_total_turns": total,
        "requested_global_turn_split_N2_main": list(topologies),
        "turn_split_sub_islands_N2_main": list(topologies),
        "unavailable_requested_topologies_due_pinned_physics_repair": [],
        "pinned_repair_N2_side_upper_fraction": 0.52,
        "basin_donor_lanes": donor_lanes,
        "basin_donor_lanes_are_coordinate_only": True,
        "source_prediction_or_pass_classification_inherited": False,
        "turn_split_parent_pair_schedule": [list(pair) for pair in pairs],
        "cross_lane_parent_pairs": [list(pair) for pair in pairs],
        "coordinate_name": "u_N2_side",
        "coordinate_index": 2,
        "initial_repaired_copies_per_sub_island": 4,
        "paired_mating": "cross_distinct_turn_split_sub_islands_every_generation",
        "migration": {
            "kind": "copy_elite_genome_then_change_only_u_N2_side",
            "period_generations": 5,
            "first_evolution_generation_included": True,
            "migrants_per_event": len(topologies),
            "target_cycle": list(topologies),
            "offspring_physics_repair_required": True,
        },
        "survival": {
            "kind": (
                "normalized_positive_G_sum_epsilon_then_"
                "positive_count_max_sum_then_rank_crowding"
            ),
            "initial_epsilon": 20.0,
            "decay_to_zero_generation": 160,
            "terminal_epsilon": 0.0,
            "epsilon_feasibility": (
                "sum_normalized_positive_G_less_than_or_equal_to_epsilon"
            ),
            "infeasible_order": (
                "positive_constraint_count_then_max_normalized_positive_G_"
                "then_sum_normalized_positive_G"
            ),
            "physical_G_mutation": False,
            "objective_mutation": False,
            "minimum_survivors_per_turn_split_sub_island": minimum_survivors,
        },
        "bounded_diversity_budget": {
            "population": PRODUCTION_POPULATION,
            "protected_topology_count": len(topologies),
            "minimum_survivors_each": minimum_survivors,
            "protected_slots": protected_slots,
            "protected_population_fraction": (
                protected_slots / PRODUCTION_POPULATION
            ),
            "maximum_single_protected_topology_count": (
                PRODUCTION_POPULATION
                - minimum_survivors * (len(topologies) - 1)
            ),
            "maximum_single_protected_topology_fraction": (
                (
                    PRODUCTION_POPULATION
                    - minimum_survivors * (len(topologies) - 1)
                )
                / PRODUCTION_POPULATION
            ),
            "additional_model_evaluations": 0,
            "population_change": 0,
            "generation_change": 0,
            "scheduler_task_or_resource_change": False,
        },
        "minimum_evolution_generations": PRODUCTION_FIXED_GENERATIONS,
        "ftol_early_stop_allowed": False,
        "physical_constraint_G_mutation": False,
        "physical_objective_mutation": False,
        "model_or_training_data_provenance_mutation": False,
        "capacitance_label_precision_remediation_included": False,
        "terminal_physical_replay_required": True,
        "warm_donor_prediction_inheritance_allowed": False,
        "final1000_topology_niche_contract": None,
        "topology_local_mating_required": False,
        "topology_niche_mating_contract": None,
        "authenticated_warm_fresh_mix_required": False,
        "exact_survivor_quota_by_N2_main": None,
        "exact_quota_required_when_population_is_320": False,
        "generation_topology_count_seal_required": False,
        "topology_niche_diversity_budget": None,
    }
    value["sha256"] = canonical_sha256(value)
    return value


def _turn_split_main_values(
    coordinates: Any,
    *,
    fixed_primary_turns: int,
    coordinate_index: int,
) -> Any:
    import numpy as np

    values = np.asarray(coordinates, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    secondary_turns = 10 * int(fixed_primary_turns)
    side_turns = np.rint(
        secondary_turns * np.clip(values[:, int(coordinate_index)], 0.0, 1.0) * 0.8
    ).astype(int)
    return secondary_turns - side_turns


def _turn_split_unit_coordinate(
    n2_main: int,
    *,
    fixed_primary_turns: int,
) -> float:
    secondary_turns = 10 * int(fixed_primary_turns)
    n2_side = secondary_turns - int(n2_main)
    value = n2_side / (0.8 * secondary_turns)
    if not 0.0 <= value <= 1.0:
        raise RuntimeError("turn-split migration coordinate escaped unit interval")
    return float(value)


def _topology_quota_for_population(
    contract: Mapping[str, Any], population: int
) -> dict[int, int]:
    """Return deterministic topology quotas, exact at production size.

    Small optimizer tests retain the historical minimum for every topology and
    apportion only their remaining slots.  The production population consumes
    the complete sealed Final1000 quota map without an unprotected remainder.
    """

    topologies = tuple(
        int(value) for value in contract["turn_split_sub_islands_N2_main"]
    )
    population = int(population)
    minimum_each = int(
        contract["survival"]["minimum_survivors_per_turn_split_sub_island"]
    )
    raw = contract.get("exact_survivor_quota_by_N2_main")
    if not raw:
        if population < minimum_each * len(topologies):
            raise RuntimeError("population cannot preserve every topology minimum")
        result = {topology: minimum_each for topology in topologies}
        result[topologies[0]] += population - sum(result.values())
        return result
    sealed = {int(key): int(value) for key, value in raw.items()}
    if (
        set(sealed) != set(topologies)
        or any(value < minimum_each for value in sealed.values())
        or sum(sealed.values()) <= 0
    ):
        raise RuntimeError("sealed topology quota map is invalid")
    contract_population = int(
        (contract.get("bounded_diversity_budget") or {}).get("population", 0)
    )
    if population == contract_population:
        if sum(sealed.values()) != population:
            raise RuntimeError("production topology quotas do not fill population")
        return sealed
    minimum_total = minimum_each * len(topologies)
    if population < minimum_total:
        raise RuntimeError("test population cannot preserve topology minima")
    remaining = population - minimum_total
    weights = {
        topology: max(0, sealed[topology] - minimum_each)
        for topology in topologies
    }
    weight_total = sum(weights.values())
    result = {topology: minimum_each for topology in topologies}
    if remaining and weight_total:
        fractional = {
            topology: remaining * weights[topology] / weight_total
            for topology in topologies
        }
        for topology in topologies:
            result[topology] += int(math.floor(fractional[topology]))
        left = population - sum(result.values())
        order = sorted(
            topologies,
            key=lambda topology: (
                -(fractional[topology] - math.floor(fractional[topology])),
                -sealed[topology],
                topology,
            ),
        )
        for topology in order[:left]:
            result[topology] += 1
    elif remaining:
        result[topologies[0]] += remaining
    if sum(result.values()) != population:
        raise RuntimeError("apportioned topology quota does not fill population")
    return result


def validate_topology_niche_warm_partition(
    partition: Mapping[str, Any], coordinates: Any
) -> dict[str, Any]:
    """Authenticate the exact 160-row Final1000 warm semantic partition."""

    import numpy as np

    try:
        from tier1_final1000_topology_niche_contract import (
            WARM_PARTITION_SCHEMA,
            WARM_SOURCE_GROUPS,
            WARM_TOPOLOGY_COUNTS,
        )
    except ImportError:  # pragma: no cover - repository module import path
        from tools.tier1_final1000_topology_niche_contract import (
            WARM_PARTITION_SCHEMA,
            WARM_SOURCE_GROUPS,
            WARM_TOPOLOGY_COUNTS,
        )

    values = np.asarray(coordinates, dtype=float)
    value = dict(partition or {})
    unsigned = dict(value)
    recorded_sha = unsigned.pop("sha256", None)
    groups = value.get("groups") or {}
    if (
        values.ndim != 2
        or len(values) != 160
        or value.get("schema_version") != WARM_PARTITION_SCHEMA
        or recorded_sha != canonical_sha256(unsigned)
        or value.get("fixed_primary_turns") != 6
        or value.get("total_count") != len(values)
        or set(groups) != set(WARM_SOURCE_GROUPS)
        or value.get("N2_main_60_warm_count") != 0
        or value.get("same_current7_physics_repair_required") is not True
        or value.get("physical_constraint_G_mutation") is not False
        or value.get("physical_objective_mutation") is not False
    ):
        raise RuntimeError("topology niche warm partition header mismatch")
    cursor = 0
    aggregate = {topology: 0 for topology in WARM_TOPOLOGY_COUNTS}
    for name, expected in WARM_SOURCE_GROUPS.items():
        group = groups.get(name) or {}
        count = int(expected["count"])
        if group.get("start") != cursor or group.get("count") != count:
            raise RuntimeError("topology niche warm group range mismatch")
        rows = values[cursor : cursor + count]
        observed = _turn_split_main_values(
            rows, fixed_primary_turns=6, coordinate_index=2
        )
        counts = {
            topology: int(np.count_nonzero(observed == topology))
            for topology in WARM_TOPOLOGY_COUNTS
        }
        expected_counts = {
            int(key): int(item)
            for key, item in expected["topology_counts"].items()
        }
        if (
            {key: item for key, item in counts.items() if item}
            != expected_counts
            or group.get("topology_counts")
            != {str(key): item for key, item in expected_counts.items()}
            or group.get("coordinate_sha256")
            != canonical_sha256(rows.tolist())
        ):
            raise RuntimeError("topology niche warm group coordinate mismatch")
        for topology, item in counts.items():
            aggregate[topology] += item
        cursor += count
    if aggregate != WARM_TOPOLOGY_COUNTS or cursor != len(values):
        raise RuntimeError("topology niche warm aggregate mismatch")
    return value


def _basin_lane_topologies(
    contract: Mapping[str, Any],
) -> dict[str, tuple[int, ...]]:
    lanes = contract.get("basin_donor_lanes") or {}
    if not isinstance(lanes, Mapping) or not lanes:
        raise RuntimeError("turn-split contract has no basin donor lanes")
    result: dict[str, tuple[int, ...]] = {}
    for name, lane in lanes.items():
        pairs = (lane or {}).get("topologies_N2_main_N2_side") or []
        topologies = tuple(int(pair[0]) for pair in pairs)
        if not topologies or len(topologies) != len(set(topologies)):
            raise RuntimeError("basin donor lane topology inventory is invalid")
        result[str(name)] = topologies
    return result


def select_basin_warm_start_indices(
    coordinates: Any,
    *,
    count: int,
    contract: Mapping[str, Any],
    random_state: Any,
) -> tuple[Any, dict[str, Any]]:
    """Select warm donors without dropping a rare authenticated topology."""

    import numpy as np

    values = np.asarray(coordinates, dtype=float)
    count = min(int(count), len(values))
    if values.ndim != 2 or count < 0:
        raise RuntimeError("basin warm selection shape is invalid")
    topologies = tuple(
        int(value) for value in contract["turn_split_sub_islands_N2_main"]
    )
    observed = _turn_split_main_values(
        values,
        fixed_primary_turns=contract["fixed_primary_turns"],
        coordinate_index=contract["coordinate_index"],
    )
    selected: list[int] = []
    copies = int(contract["initial_repaired_copies_per_sub_island"])
    for topology in topologies:
        candidates = np.flatnonzero(observed == topology)
        if len(candidates):
            candidates = random_state.permutation(candidates)
            selected.extend(
                int(index)
                for index in candidates[: min(copies, len(candidates))]
                if int(index) not in selected
            )
    if len(selected) > count:
        selected = selected[:count]
    remaining = np.asarray(
        [index for index in range(len(values)) if index not in selected],
        dtype=int,
    )
    if len(selected) < count:
        selected.extend(
            int(index)
            for index in random_state.permutation(remaining)[: count - len(selected)]
        )
    selected_values = observed[np.asarray(selected, dtype=int)]
    selected_counts = {
        str(topology): int(np.count_nonzero(selected_values == topology))
        for topology in topologies
    }
    available_counts = {
        str(topology): int(np.count_nonzero(observed == topology))
        for topology in topologies
    }
    audit = {
        "schema_version": "mft-tier1-basin-warm-selection-v1",
        "requested_count": count,
        "selected_count": len(selected),
        "available_topology_counts": available_counts,
        "selected_topology_counts": selected_counts,
        "rare_available_topology_dropped": any(
            available_counts[str(topology)] > 0 and selected_counts[str(topology)] == 0
            for topology in topologies
        ),
        "prior_prediction_or_pass_classification_inherited": False,
        "additional_model_evaluations": 0,
    }
    if audit["rare_available_topology_dropped"]:
        raise RuntimeError("basin warm selection dropped an available topology")
    audit["sha256"] = canonical_sha256(audit)
    return np.asarray(selected, dtype=int), audit


def seed_turn_split_sub_islands(
    problem: Any,
    initial: Any,
    contract: Mapping[str, Any],
    *,
    warm_donor_count: int = 0,
    protected_warm_donors_only: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Install repaired copies while retaining distinct basin donor genomes."""

    import numpy as np

    values = np.asarray(initial, dtype=float).copy()
    topologies = tuple(
        int(value) for value in contract["turn_split_sub_islands_N2_main"]
    )
    copies = int(contract["initial_repaired_copies_per_sub_island"])
    coordinate_index = int(contract["coordinate_index"])
    required = len(topologies) * copies
    if values.ndim != 2 or len(values) < required:
        raise RuntimeError(
            f"population requires at least {required} rows for topology seeding"
        )
    if not 0 <= int(warm_donor_count) <= len(values):
        raise RuntimeError("warm donor count escaped the initial population")
    if not isinstance(protected_warm_donors_only, bool):
        raise RuntimeError("protected warm-donor policy must be boolean")
    if protected_warm_donors_only and int(warm_donor_count) < 1:
        raise RuntimeError("protected warm-donor policy has no donor rows")
    lanes = _basin_lane_topologies(contract)
    source_values = values.copy()
    source_observed = _turn_split_main_values(
        source_values,
        fixed_primary_turns=contract["fixed_primary_turns"],
        coordinate_index=coordinate_index,
    )
    protected = set(topologies)
    donor_sources = []
    for copy_index in range(copies):
        for topology_index, n2_main in enumerate(topologies):
            row = copy_index * len(topologies) + topology_index
            lane_topologies = {
                topology
                for values_in_lane in lanes.values()
                if n2_main in values_in_lane
                for topology in values_in_lane
            }
            candidate_tiers = (
                ("same_topology", source_observed == n2_main),
                (
                    "same_basin_lane",
                    np.isin(source_observed, sorted(lane_topologies)),
                ),
                (
                    "protected_basin_topology",
                    np.isin(source_observed, sorted(protected)),
                ),
                ("any_initial_coordinate", np.ones(len(values), dtype=bool)),
            )
            ordered_candidates: list[int] = []
            candidate_source_tier: dict[int, str] = {}
            for tier, mask in candidate_tiers:
                candidates = np.flatnonzero(mask)
                if protected_warm_donors_only:
                    candidates = candidates[candidates < int(warm_donor_count)]
                elif int(warm_donor_count):
                    candidates = np.concatenate(
                        (
                            candidates[candidates < int(warm_donor_count)],
                            candidates[candidates >= int(warm_donor_count)],
                        )
                    )
                for candidate in candidates:
                    candidate = int(candidate)
                    if candidate not in candidate_source_tier:
                        ordered_candidates.append(candidate)
                        candidate_source_tier[candidate] = tier
            if not ordered_candidates:  # pragma: no cover
                raise RuntimeError("turn-split basin has no coordinate donor")
            source_index = ordered_candidates[copy_index % len(ordered_candidates)]
            source_tier = candidate_source_tier[source_index]
            values[row] = source_values[source_index]
            values[row, coordinate_index] = _turn_split_unit_coordinate(
                n2_main,
                fixed_primary_turns=contract["fixed_primary_turns"],
            )
            donor_sources.append(
                {
                    "target_N2_main": n2_main,
                    "copy_index": copy_index,
                    "source_row": source_index,
                    "source_N2_main": int(source_observed[source_index]),
                    "source_is_authenticated_warm": bool(
                        source_index < int(warm_donor_count)
                    ),
                    "source_tier": source_tier,
                }
            )
    values = np.asarray(problem.repair_unit_coordinates(values), dtype=float)
    observed = _turn_split_main_values(
        values,
        fixed_primary_turns=contract["fixed_primary_turns"],
        coordinate_index=coordinate_index,
    )
    counts = {
        str(topology): int(np.count_nonzero(observed == topology))
        for topology in topologies
    }
    if any(counts[str(topology)] < copies for topology in topologies):
        raise RuntimeError("turn-split initialization repair lost a sub-island")
    audit = {
        "schema_version": "mft-tier1-basin-aware-initialization-v2",
        "topology_counts_after_current_repair": counts,
        "minimum_copies_each_verified": True,
        "basin_lane_topologies": {
            name: list(lane_topologies) for name, lane_topologies in lanes.items()
        },
        "basin_seeded_counts": {
            name: sum(counts[str(topology)] for topology in lane_topologies)
            for name, lane_topologies in lanes.items()
        },
        "donor_sources": donor_sources,
        "same_or_same_lane_donor_count": sum(
            item["source_tier"] in {"same_topology", "same_basin_lane"}
            for item in donor_sources
        ),
        "warm_donor_count": int(warm_donor_count),
        "protected_warm_donors_only": protected_warm_donors_only,
        "warm_coordinates_are_donors_only": True,
        "source_prediction_or_pass_classification_inherited": False,
        "additional_model_evaluations": 0,
    }
    audit["sha256"] = canonical_sha256(audit)
    return values, audit


def build_topology_niche_initial_population(
    problem: Any,
    authenticated_warm: Any,
    contract: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    """Build the sealed 160 warm + 160 topology-stratified Sobol population."""

    import numpy as np
    from scipy.stats import qmc

    try:
        from tier1_final1000_topology_niche_contract import (
            FRESH_SOBOL_TOPOLOGY_COUNTS,
            WARM_TOPOLOGY_COUNTS,
            validate_contract as validate_niche_contract,
        )
    except ImportError:  # pragma: no cover - repository module import path
        from tools.tier1_final1000_topology_niche_contract import (
            FRESH_SOBOL_TOPOLOGY_COUNTS,
            WARM_TOPOLOGY_COUNTS,
            validate_contract as validate_niche_contract,
        )

    niche = validate_niche_contract(
        contract.get("final1000_topology_niche_contract") or {}
    )
    warm = np.asarray(authenticated_warm, dtype=float).copy()
    if warm.shape != (160, int(problem.n_var)) or not np.isfinite(warm).all():
        raise RuntimeError("topology niche warm population must be exactly 160xN")
    warm_labels = np.asarray([
        topology
        for topology, count in WARM_TOPOLOGY_COUNTS.items()
        for _ in range(count)
    ], dtype=int)
    # The artifact is grouped semantically rather than by topology.  Preserve
    # its rows and read their authenticated topology instead of reordering it.
    warm_observed = _turn_split_main_values(
        warm,
        fixed_primary_turns=6,
        coordinate_index=int(contract["coordinate_index"]),
    )
    warm_counts = {
        topology: int(np.count_nonzero(warm_observed == topology))
        for topology in WARM_TOPOLOGY_COUNTS
    }
    if warm_counts != WARM_TOPOLOGY_COUNTS or np.any(warm_observed == 60):
        raise RuntimeError("authenticated topology niche warm mix drifted")
    del warm_labels

    sobol = qmc.Sobol(d=int(problem.n_var), scramble=True, seed=int(seed))
    # 256 is the smallest power-of-two draw containing the required 160 rows;
    # using random_base2 avoids Sobol balance warnings and is reproducible.
    fresh = np.asarray(sobol.random_base2(m=8)[:160], dtype=float)
    fresh_labels = np.asarray([
        topology
        for topology, count in FRESH_SOBOL_TOPOLOGY_COUNTS.items()
        for _ in range(count)
    ], dtype=int)
    if len(fresh_labels) != len(fresh):
        raise RuntimeError("fresh Sobol topology quota does not contain 160 rows")
    coordinate_index = int(contract["coordinate_index"])
    for row, topology in enumerate(fresh_labels):
        fresh[row, coordinate_index] = _turn_split_unit_coordinate(
            int(topology), fixed_primary_turns=6
        )
    raw = np.vstack([warm, fresh])
    repaired = np.asarray(problem.repair_unit_coordinates(raw), dtype=float)
    if repaired.shape != raw.shape or not np.isfinite(repaired).all():
        raise RuntimeError("topology niche initialization repair failed")
    observed = _turn_split_main_values(
        repaired,
        fixed_primary_turns=6,
        coordinate_index=coordinate_index,
    )
    expected = {
        int(key): int(item)
        for key, item in niche["topology_quota_by_N2_main"].items()
    }
    counts = {
        topology: int(np.count_nonzero(observed == topology))
        for topology in expected
    }
    if counts != expected:
        raise RuntimeError("physics repair changed a sealed topology niche quota")
    fresh_observed = observed[160:]
    fresh_counts = {
        topology: int(np.count_nonzero(fresh_observed == topology))
        for topology in expected
    }
    if fresh_counts != FRESH_SOBOL_TOPOLOGY_COUNTS:
        raise RuntimeError("fresh Sobol sentinel/topology mix drifted")
    audit = {
        "schema_version": "mft-tier1-final1000-topology-initialization-v1",
        "seed": int(seed),
        "population": len(repaired),
        "authenticated_warm_count": 160,
        "fresh_sobol_count": 160,
        "warm_topology_counts": {
            str(key): item for key, item in warm_counts.items()
        },
        "fresh_sobol_topology_counts": {
            str(key): item for key, item in fresh_counts.items()
        },
        "complete_topology_counts": {
            str(key): item for key, item in counts.items()
        },
        "fresh_N2_main_60_sentinel_count": fresh_counts[60],
        "warm_N2_main_60_count": warm_counts[60],
        "sobol": {
            "implementation": "scipy.stats.qmc.Sobol",
            "scramble": True,
            "power_of_two_draw": 256,
            "used_prefix_count": 160,
            "raw_prefix_sha256": canonical_sha256(fresh.tolist()),
        },
        "same_current7_physics_repair_applied": True,
        "additional_model_evaluations": 0,
        "physical_constraint_G_mutation": False,
        "physical_objective_mutation": False,
    }
    audit["sha256"] = canonical_sha256(audit)
    return repaired, audit


def create_deep_topology_components(
    problem: Any,
    contract: Mapping[str, Any],
    repair: Any,
) -> tuple[Any, Any, Any]:
    """Create paired mating, periodic migration and epsilon survival."""

    import numpy as np
    from pymoo.core.duplicate import DefaultDuplicateElimination
    from pymoo.core.mating import Mating
    from pymoo.core.selection import Selection
    from pymoo.core.survival import Survival
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.survival.rank_and_crowding import RankAndCrowding

    topologies = tuple(
        int(value) for value in contract["turn_split_sub_islands_N2_main"]
    )
    parent_pairs = tuple(
        tuple(int(value) for value in pair)
        for pair in contract["turn_split_parent_pair_schedule"]
    )
    fixed_turns = int(contract["fixed_primary_turns"])
    coordinate_index = int(contract["coordinate_index"])
    minimum_each = int(
        contract["survival"]["minimum_survivors_per_turn_split_sub_island"]
    )
    initial_epsilon = float(contract["survival"]["initial_epsilon"])
    epsilon_decay_generation = int(contract["survival"]["decay_to_zero_generation"])
    migration_period = int(contract["migration"]["period_generations"])
    migrants_per_event = int(contract["migration"]["migrants_per_event"])
    topology_local_mating = bool(contract.get("topology_local_mating_required"))
    niche_mating = contract.get("topology_niche_mating_contract") or {}
    niche_pair_counts = {
        str(key): int(item)
        for key, item in (
            niche_mating.get("parent_pair_counts_per_160") or {}
        ).items()
    }
    niche_pair_cycle: tuple[tuple[int, int], ...] = tuple()
    if niche_pair_counts:
        if sum(niche_pair_counts.values()) != 160:
            raise RuntimeError("topology niche parent-pair schedule must sum to 160")
        parsed_pairs = []
        for label, count in niche_pair_counts.items():
            parts = label.split("x")
            if len(parts) != 2:
                raise RuntimeError("topology niche parent-pair label is invalid")
            pair = (int(parts[0]), int(parts[1]))
            if not set(pair) <= set(topologies) or count < 0:
                raise RuntimeError("topology niche parent-pair schedule is invalid")
            parsed_pairs.extend([pair] * count)
        niche_pair_cycle = tuple(parsed_pairs)
        if (
            niche_mating.get("priority_cross_pair") != "36x37"
            or niche_pair_counts.get("36x37") != 64
            or not math.isclose(
                float(niche_mating.get("priority_cross_pair_fraction", -1.0)),
                0.4,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise RuntimeError("N36/N37 priority mating schedule drifted")
    production_population = int(
        (contract.get("bounded_diversity_budget") or {}).get("population", 0)
    )
    production_quotas = _topology_quota_for_population(
        contract, production_population
    )
    topology_parent_cycle = tuple(
        topology
        for topology in topologies
        for _ in range(max(1, production_quotas[topology] // 2))
    )

    def topology_values(population: Any) -> Any:
        return _turn_split_main_values(
            population.get("X"),
            fixed_primary_turns=fixed_turns,
            coordinate_index=coordinate_index,
        )

    def individual_score(
        individual: Any,
    ) -> tuple[float, int, float, float, float]:
        rank = individual.get("rank")
        crowding = individual.get("crowding")
        constraints = np.asarray(individual.get("G"), dtype=float)
        positive_g = np.maximum(constraints, 0.0)
        positive_count = int(np.count_nonzero(positive_g > 0.0))
        positive_max = float(positive_g.max())
        positive_sum = float(positive_g.sum())
        rank_value = float(rank) if rank is not None else math.inf
        crowding_value = float(crowding) if crowding is not None else -math.inf
        return (
            rank_value,
            positive_count,
            positive_max,
            positive_sum,
            -crowding_value,
        )

    class TurnSplitPairedSelection(Selection):
        def __init__(self) -> None:
            super().__init__()
            self.selection_calls = 0
            self.parent_pairs_emitted = 0
            self.topology_local_pairs_emitted = 0
            self.parent_pair_topology_counts = {
                str(topology): 0 for topology in topologies
            }
            self.cross_36x37_pairs_emitted = 0
            self.last_parent_pair_topologies: list[tuple[int, int]] = []
            self.generation_parent_pair_counts: list[dict[str, Any]] = []

        @staticmethod
        def _pick(pop: Any, candidates: Any, random_state: Any) -> int:
            candidates = np.asarray(candidates, dtype=int)
            if len(candidates) == 0:
                candidates = np.arange(len(pop), dtype=int)
            draw = random_state.choice(
                candidates, size=min(2, len(candidates)), replace=False
            )
            return int(min(draw, key=lambda index: individual_score(pop[index])))

        def _do(
            self,
            active_problem: Any,
            pop: Any,
            n_select: int,
            n_parents: int,
            *args: Any,
            random_state: Any = None,
            **kwargs: Any,
        ) -> Any:
            if active_problem is not problem or int(n_parents) != 2:
                raise RuntimeError("turn-split paired selection contract mismatch")
            observed = topology_values(pop)
            selected = np.empty((int(n_select), 2), dtype=int)
            offset = self.parent_pairs_emitted
            call_pairs: list[tuple[int, int]] = []
            for index in range(int(n_select)):
                if niche_pair_cycle:
                    left, right = niche_pair_cycle[
                        (offset + index) % len(niche_pair_cycle)
                    ]
                elif topology_local_mating:
                    left = right = topology_parent_cycle[
                        (offset + index) % len(topology_parent_cycle)
                    ]
                else:
                    left, right = parent_pairs[(offset + index) % len(parent_pairs)]
                selected[index, 0] = self._pick(
                    pop, np.flatnonzero(observed == left), random_state
                )
                selected[index, 1] = self._pick(
                    pop, np.flatnonzero(observed == right), random_state
                )
                call_pairs.append((left, right))
                if left == right:
                    self.topology_local_pairs_emitted += 1
                    self.parent_pair_topology_counts[str(left)] += 1
                elif {left, right} == {36, 37}:
                    self.cross_36x37_pairs_emitted += 1
            pair_counts = {
                label: sum(
                    pair == tuple(int(value) for value in label.split("x"))
                    for pair in call_pairs
                )
                for label in niche_pair_counts
            }
            algorithm = kwargs.get("algorithm")
            generation = int(getattr(algorithm, "n_gen", 0) or 0)
            pair_record = {
                "selection_call": self.selection_calls + 1,
                "algorithm_generation": generation,
                "pair_count": len(call_pairs),
                "pair_counts": pair_counts,
                "cross_36x37_pair_count": sum(
                    set(pair) == {36, 37} for pair in call_pairs
                ),
            }
            pair_record["sha256"] = canonical_sha256(pair_record)
            self.generation_parent_pair_counts.append(pair_record)
            self.last_parent_pair_topologies = call_pairs
            self.selection_calls += 1
            self.parent_pairs_emitted += int(n_select)
            return selected

    class TurnSplitMigrationMating(Mating):
        def __init__(self, selection: Any) -> None:
            super().__init__(
                selection,
                SBX(eta=15, prob=0.9),
                PM(eta=20),
                repair=repair,
                eliminate_duplicates=DefaultDuplicateElimination(),
                n_max_iterations=100,
            )
            self.migration_events = 0
            self.migrants_created = 0
            self.last_migration_generation = None
            self.cross_36x37_offspring_attributed = 0
            self.generation_cross_offspring_counts: list[dict[str, Any]] = []

        def _do(
            self,
            active_problem: Any,
            pop: Any,
            n_offsprings: int,
            parents: Any = None,
            random_state: Any = None,
            **kwargs: Any,
        ) -> Any:
            if active_problem is not problem:
                raise RuntimeError("turn-split mating received a different problem")
            algorithm = kwargs.get("algorithm")
            generation = int(getattr(algorithm, "n_gen", 0) or 0)
            # The optimizer-only allowance wrapper reads this value during the
            # immediately following offspring evaluation.  Physical replay
            # never consumes it.
            problem._tier1_optimizer_generation = generation
            offspring = super()._do(
                active_problem,
                pop,
                n_offsprings,
                parents=parents,
                random_state=random_state,
                **kwargs,
            )
            migrate = (
                self.last_migration_generation is None
                or generation - self.last_migration_generation >= migration_period
            )
            if migrate and len(offspring):
                observed = topology_values(pop)
                migrants = []
                for target_index, target in enumerate(topologies):
                    source = topologies[target_index - 1]
                    candidates = np.flatnonzero(observed == source)
                    if len(candidates) == 0:
                        continue
                    elite_index = min(
                        candidates, key=lambda index: individual_score(pop[index])
                    )
                    coordinate = np.asarray(
                        pop[int(elite_index)].X, dtype=float
                    ).copy()
                    coordinate[coordinate_index] = _turn_split_unit_coordinate(
                        target, fixed_primary_turns=fixed_turns
                    )
                    migrants.append(coordinate)
                if migrants:
                    while len(migrants) < migrants_per_event:
                        migrants.extend(list(migrants))
                    migrants = migrants[: min(migrants_per_event, len(offspring))]
                    repaired_migrants = np.asarray(
                        repair._do(
                            active_problem,
                            np.asarray(migrants, dtype=float),
                            algorithm=algorithm,
                        ),
                        dtype=float,
                    )
                    if repaired_migrants.shape != (
                        len(migrants), int(active_problem.n_var)
                    ):
                        raise RuntimeError("turn-split migrant repair shape mismatch")
                    for index, coordinate in enumerate(repaired_migrants):
                        offspring[index].set("X", coordinate)
                    self.migration_events += 1
                    self.migrants_created += len(migrants)
                    self.last_migration_generation = generation

            cross_pair_count = sum(
                set(pair) == {36, 37}
                for pair in self.selection.last_parent_pair_topologies
            )
            cross_offspring_count = min(
                len(offspring), 2 * int(cross_pair_count)
            )
            attributed_counts = {36: 0, 37: 0}
            if niche_pair_cycle and cross_offspring_count:
                coordinates = np.asarray(
                    [offspring[index].X for index in range(cross_offspring_count)],
                    dtype=float,
                )
                for index in range(cross_offspring_count):
                    topology = 36 if index % 2 == 0 else 37
                    coordinates[index, coordinate_index] = (
                        _turn_split_unit_coordinate(
                            topology, fixed_primary_turns=fixed_turns
                        )
                    )
                    attributed_counts[topology] += 1
                repaired_cross = np.asarray(
                    repair._do(
                        active_problem,
                        coordinates,
                        algorithm=algorithm,
                    ),
                    dtype=float,
                )
                observed_cross = _turn_split_main_values(
                    repaired_cross,
                    fixed_primary_turns=fixed_turns,
                    coordinate_index=coordinate_index,
                )
                if (
                    repaired_cross.shape != coordinates.shape
                    or int(np.count_nonzero(observed_cross == 36))
                    != attributed_counts[36]
                    or int(np.count_nonzero(observed_cross == 37))
                    != attributed_counts[37]
                ):
                    raise RuntimeError("N36/N37 cross offspring attribution failed")
                for index, coordinate in enumerate(repaired_cross):
                    offspring[index].set("X", coordinate)
                self.cross_36x37_offspring_attributed += cross_offspring_count
            cross_record = {
                "algorithm_generation": generation,
                "cross_36x37_parent_pair_count": int(cross_pair_count),
                "cross_offspring_attributed_count": int(cross_offspring_count),
                "attributed_topology_counts": {
                    str(key): item for key, item in attributed_counts.items()
                },
            }
            cross_record["sha256"] = canonical_sha256(cross_record)
            self.generation_cross_offspring_counts.append(cross_record)
            return offspring

    class TurnSplitEpsilonSurvival(Survival):
        def __init__(self) -> None:
            super().__init__(filter_infeasible=False)
            self.ranking = RankAndCrowding()
            self.survival_calls = 0
            self.last_epsilon = None
            self.last_topology_counts: dict[str, int] = {}
            self.minimum_topology_count_observed = math.inf
            self.maximum_single_topology_count_observed = 0
            self.generation_topology_counts: list[dict[str, Any]] = []

        def _do(
            self,
            active_problem: Any,
            pop: Any,
            *args: Any,
            n_survive: int | None = None,
            random_state: Any = None,
            **kwargs: Any,
        ) -> Any:
            if active_problem is not problem or n_survive is None:
                raise RuntimeError("turn-split survival contract mismatch")
            n_survive = min(int(n_survive), len(pop))
            algorithm = kwargs.get("algorithm")
            generation = int(getattr(algorithm, "n_gen", 1) or 1)
            evolution_generation = max(0, generation - 1)
            fraction = max(0.0, 1.0 - evolution_generation / epsilon_decay_generation)
            epsilon = initial_epsilon * fraction * fraction
            constraints = np.asarray(pop.get("G"), dtype=float)
            positive_g = np.maximum(constraints, 0.0)
            positive_count = np.count_nonzero(positive_g > 0.0, axis=1)
            positive_max = positive_g.max(axis=1)
            positive_sum = positive_g.sum(axis=1)
            # Preserve the sealed aggregate epsilon admission exactly.  The
            # minimax tuple only orders rows that fail this sum gate; it must
            # never turn several sub-epsilon misses into an objective-feasible
            # row (for example G=[15, 15] at epsilon=20).
            epsilon_feasible = np.flatnonzero(positive_sum <= epsilon)
            epsilon_infeasible = np.flatnonzero(positive_sum > epsilon)
            global_order: list[int] = []
            if len(epsilon_feasible):
                ranked = self.ranking._do(
                    active_problem,
                    pop[epsilon_feasible],
                    n_survive=len(epsilon_feasible),
                    random_state=random_state,
                )
                identity = {
                    id(individual): index for index, individual in enumerate(pop)
                }
                global_order.extend(identity[id(individual)] for individual in ranked)
            infeasible_order = np.lexsort(
                (
                    positive_sum[epsilon_infeasible],
                    positive_max[epsilon_infeasible],
                    positive_count[epsilon_infeasible],
                )
            )
            for rank_offset, index in enumerate(epsilon_infeasible[infeasible_order]):
                pop[int(index)].set("rank", len(global_order) + rank_offset)
                pop[int(index)].set("crowding", -float(positive_max[int(index)]))
                global_order.append(int(index))
            observed = topology_values(pop)
            selected: list[int] = []
            exact_quota_mode = bool(
                contract.get("exact_survivor_quota_by_N2_main")
            )
            quota = _topology_quota_for_population(contract, n_survive)
            for topology in topologies:
                topology_rows = [
                    index
                    for index in global_order
                    if observed[index] == topology and index not in selected
                ]
                required = quota[topology] if exact_quota_mode else minimum_each
                if len(topology_rows) < required:
                    raise RuntimeError(
                        "topology-local survival candidate pool underfilled"
                    )
                selected.extend(topology_rows[:required])
            if exact_quota_mode:
                if len(selected) != n_survive:
                    raise RuntimeError(
                        "topology-local survival quotas do not fill population"
                    )
            else:
                selected.extend(
                    index for index in global_order if index not in selected
                )
            survivors = pop[np.asarray(selected[:n_survive], dtype=int)]
            survivor_topologies = topology_values(survivors)
            counts = {
                str(topology): int(np.count_nonzero(survivor_topologies == topology))
                for topology in topologies
            }
            expected_counts = (
                {
                    str(topology): int(quota[topology])
                    for topology in topologies
                }
                if exact_quota_mode
                else counts
            )
            if exact_quota_mode and counts != expected_counts:
                raise RuntimeError(
                    "epsilon survival violated an exact topology niche quota"
                )
            if not exact_quota_mode and any(
                counts[str(topology)] < minimum_each for topology in topologies
            ):
                raise RuntimeError(
                    "epsilon survival lost a required turn-split sub-island"
                )
            self.survival_calls += 1
            self.last_epsilon = float(epsilon)
            self.last_topology_counts = counts
            self.minimum_topology_count_observed = min(
                self.minimum_topology_count_observed, min(counts.values())
            )
            self.maximum_single_topology_count_observed = max(
                self.maximum_single_topology_count_observed,
                max(counts.values()),
            )
            count_record = {
                "survival_call": self.survival_calls,
                "algorithm_generation": generation,
                "counts": counts,
                "expected_counts": expected_counts,
                "exact_quota_verified": exact_quota_mode,
            }
            count_record["sha256"] = canonical_sha256(count_record)
            self.generation_topology_counts.append(count_record)
            return survivors

    selection = TurnSplitPairedSelection()
    mating = TurnSplitMigrationMating(selection)
    survival = TurnSplitEpsilonSurvival()
    return selection, mating, survival


def derive_half_magnetizing_self_resonance(
    measurements: Mapping[str, Any],
    params: Mapping[str, Any],
    *,
    magnetizing_inductance_factor: float = 0.5,
) -> dict[str, float]:
    """Derive the authoritative Tx/Rx-only half-Lm resonance screen."""

    if not isinstance(measurements, Mapping) or not callable(
        getattr(params, "get", None)
    ):
        raise RuntimeError("resonance inputs must be mappings")
    leakage_h = _positive_number(measurements.get("Llt_phys"), "Llt_phys") * 1e-6
    coupling = _finite_number(measurements.get("k"), "k")
    if not 0.0 < coupling < 1.0:
        raise RuntimeError("k must satisfy 0 < k < 1")
    c_tx = _positive_number(measurements.get("C_tx_tx_F"), "C_tx_tx_F")
    c_rx = _positive_number(measurements.get("C_rx_rx_F"), "C_rx_rx_F")
    factor = _positive_number(
        magnetizing_inductance_factor,
        "magnetizing inductance factor",
    )
    if factor > 1.0:
        raise RuntimeError("magnetizing inductance factor must be <= 1")
    n1 = _finite_number(params.get("N1_main", 0.0), "N1_main") + _finite_number(
        params.get("N1_side", 0.0), "N1_side"
    )
    n2 = _finite_number(params.get("N2_main", 0.0), "N2_main") + _finite_number(
        params.get("N2_side", 0.0), "N2_side"
    )
    if n1 <= 0.0 or n2 <= 0.0:
        raise RuntimeError("positive primary and secondary turns are required")

    tx_self_h = leakage_h / (1.0 - coupling * coupling)
    tx_magnetizing_h = tx_self_h * coupling * coupling
    rx_magnetizing_h = tx_magnetizing_h * (n2 / n1) ** 2

    def lc_hz(inductance_h: float, capacitance_f: float) -> float:
        frequency = 1.0 / (2.0 * math.pi * math.sqrt(inductance_h * capacitance_f))
        return _positive_number(frequency, "self resonance frequency")

    tx_frequency = lc_hz(factor * tx_magnetizing_h, c_tx)
    rx_frequency = lc_hz(factor * rx_magnetizing_h, c_rx)
    return {
        "f_res_tx_half_magnetizing_Hz": tx_frequency,
        "f_res_rx_half_magnetizing_Hz": rx_frequency,
        "f_res_min_tx_rx_only_Hz": min(tx_frequency, rx_frequency),
        "magnetizing_inductance_factor": factor,
        "interwinding_resonance_included": False,
    }


def create_current7_problem_class(
    *,
    base_problem_class: Any,
    base_constraint_names: Any,
    base_fixed_stack_mm: Mapping[str, Any],
    sobol_dims: Any,
    bounding_box_lit: Any,
    input_parameter_module: Any,
    design_analytical_b_field_t: Any,
) -> type:
    """Create an additive wrapper without importing or modifying legacy code."""

    import numpy as np

    if tuple(base_constraint_names) != BASE_CONSTRAINT_NAMES:
        raise RuntimeError("current7 simple-base constraint schema mismatch")
    if dict(base_fixed_stack_mm) != EXPECTED_SIMPLE_BASE_FIXED_STACK_MM:
        raise RuntimeError("current7 simple-base fixed-stack schema mismatch")
    normalized_dims = tuple(
        (str(name), float(lower), float(upper)) for name, lower, upper in sobol_dims
    )
    names = [item[0] for item in normalized_dims]
    if len(names) != len(set(names)):
        raise RuntimeError("current7 Sobol schema contains duplicate dimensions")
    for name in VARIABLE_COOLING_DIMENSIONS:
        if names.count(name) != 1:
            raise RuntimeError(f"current7 Sobol schema has no unique {name}")
        _, lower, upper = normalized_dims[names.index(name)]
        if not lower < upper:
            raise RuntimeError(f"current7 cooling dimension is not variable: {name}")
    for required in (
        "KEYS",
        "N1_MIN_TURNS",
        "N1_MAX_TURNS",
        "PRIMARY_CONDUCTOR_MAX_THICKNESS_MM",
        "unit_to_dims",
        "decode_unit_sample",
        "create_input_parameter",
        "validation_check",
        "get_drawing_default_params",
    ):
        if not hasattr(input_parameter_module, required):
            raise RuntimeError(f"current7 input module is missing {required}")
    sobol_index = {name: index for index, name in enumerate(names)}

    class Current7Tier1Problem(base_problem_class):
        problem_schema = PROBLEM_SCHEMA
        temperature_contract = CURRENT_TEMPERATURE_CONTRACT
        temperature_contract_sha256 = CURRENT_TEMPERATURE_CONTRACT_SHA256
        hard_constraint_contract = CURRENT_STAGE_HARD_CONTRACT
        hard_constraint_contract_sha256 = CURRENT_STAGE_HARD_CONTRACT_SHA256
        stage_spec_sha256 = CURRENT_STAGE_SPEC_SHA256
        offspring_physics_repair = True
        launch_eligible = True

        def __init__(
            self,
            models: Mapping[str, Any],
            spec: Mapping[str, Any] | None = None,
            density_gate: Any = None,
            fixed_overrides: Mapping[str, Any] | None = None,
            fixed_primary_turns: int | None = None,
        ) -> None:
            effective_spec = validate_stage_spec(spec or CURRENT_STAGE_SPEC)
            goal_campaign = is_goal_stage_spec(effective_spec)
            constraint_names = stage_constraint_names(effective_spec)
            temperature_contract = stage_temperature_contract(effective_spec)
            hard_constraint_contract = stage_hard_constraint_contract(effective_spec)

            overrides = dict(fixed_overrides or {})
            for name in VARIABLE_COOLING_DIMENSIONS:
                if name in overrides:
                    raise ValueError(
                        f"Tier-1 corrected search requires variable cooling: {name}"
                    )
            required_fixed = (
                {
                    **GOAL_FIXED_OPERATING_IDENTITY,
                    **{
                        name: value
                        for name, value in GOAL_FIXED_COOLING_IDENTITY.items()
                        if name != "thermal_pad_conductivity_W_mK"
                    },
                }
                if goal_campaign
                else {**FIXED_COOLING_PADS_MM}
            )

            def fixed_value_matches(observed: Any, expected: Any) -> bool:
                if isinstance(expected, str):
                    return observed == expected
                try:
                    return bool(
                        np.isclose(
                            _finite_number(observed, "fixed control"),
                            float(expected),
                            rtol=0.0,
                            atol=1e-12,
                        )
                    )
                except RuntimeError:
                    return False

            for name, expected in required_fixed.items():
                if name in overrides and not fixed_value_matches(
                    overrides[name], expected
                ):
                    raise ValueError(
                        f"Tier-1 corrected search fixes {name}={expected!r}"
                    )
            overrides.update(required_fixed)
            super().__init__(
                models,
                spec=effective_spec,
                density_gate=density_gate,
                fixed_overrides=overrides,
            )
            self.goal_campaign = goal_campaign
            expected_base_constraint_names = (
                GOAL_BASE_CONSTRAINT_NAMES
                if goal_campaign
                else BASE_CONSTRAINT_NAMES
            )
            if tuple(self.constraint_names) != expected_base_constraint_names:
                raise RuntimeError("current7 base constraint schema drifted")
            if int(self.n_ieq_constr) != len(expected_base_constraint_names):
                raise RuntimeError("current7 base constraint width drifted")
            for name, expected in effective_spec.items():
                if goal_campaign:
                    if self.spec.get(name) != expected:
                        raise RuntimeError(f"goal stage spec escaped: {name}")
                elif expected is None:
                    if self.spec.get(name) is not None:
                        raise RuntimeError(f"Tier-1 stage spec escaped: {name}")
                elif not math.isclose(
                    _finite_number(self.spec.get(name), name),
                    float(expected),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise RuntimeError(f"Tier-1 stage spec escaped: {name}")

            # The simple current problem overwrites both trained plate controls
            # and clamps their unit coordinates.  Remove only those known exact
            # mutations; pads and cw1 remain fixed and are attested after decode.
            for name in ("core_plate_t", "wcp_t"):
                if not np.isclose(
                    _finite_number(self.fixed_overrides.get(name), name),
                    EXPECTED_SIMPLE_BASE_FIXED_STACK_MM[name],
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise RuntimeError(f"unexpected base cooling override: {name}")
                self.fixed_overrides.pop(name)
                coordinate = names.index(name)
                self.xl[coordinate] = 0.0
                self.xu[coordinate] = 1.0
            for name in VARIABLE_COOLING_DIMENSIONS:
                if name in self.fixed_overrides:
                    raise RuntimeError(f"cooling control remained fixed: {name}")
                coordinate = names.index(name)
                if not (
                    np.isclose(self.xl[coordinate], 0.0)
                    and np.isclose(self.xu[coordinate], 1.0)
                ):
                    raise RuntimeError(f"cooling unit bounds remained clamped: {name}")
            for name, expected in required_fixed.items():
                if not fixed_value_matches(
                    self.fixed_overrides.get(name), expected
                ):
                    raise RuntimeError(f"fixed manufacturing control escaped: {name}")

            allowed_turns = (
                GOAL_SUPPORTED_PRIMARY_TURNS
                if goal_campaign
                else SUPPORTED_FIXED_PRIMARY_TURNS
            )
            if fixed_primary_turns not in allowed_turns:
                raise ValueError(
                    "fixed_primary_turns is outside the campaign strata"
                )
            self.fixed_primary_turns = int(fixed_primary_turns)
            self.fixed_primary_turn_coordinate_index = sobol_index["u_N1"]
            self.fixed_primary_turn_unit_coordinate = (
                fixed_primary_turn_unit_coordinate(
                    self.fixed_primary_turns,
                    minimum_turns=input_parameter_module.N1_MIN_TURNS,
                    maximum_turns=input_parameter_module.N1_MAX_TURNS,
                )
            )
            self.xl[self.fixed_primary_turn_coordinate_index] = (
                self.fixed_primary_turn_unit_coordinate
            )
            self.xu[self.fixed_primary_turn_coordinate_index] = (
                self.fixed_primary_turn_unit_coordinate
            )
            self.optimizer_repair_contract = optimizer_repair_contract(
                fixed_primary_turns=self.fixed_primary_turns,
                coordinate_index=self.fixed_primary_turn_coordinate_index,
                fixed_unit_coordinate=self.fixed_primary_turn_unit_coordinate,
                goal_campaign=goal_campaign,
            )
            self.cw1_coordinate_index = sobol_index["f1_split"]

            self.base_constraint_names = expected_base_constraint_names
            self.stage_spec = effective_spec
            self.stage_spec_sha256 = canonical_sha256(effective_spec)
            self.temperature_contract = temperature_contract
            self.temperature_contract_sha256 = canonical_sha256(temperature_contract)
            self.hard_constraint_contract = hard_constraint_contract
            self.hard_constraint_contract_sha256 = canonical_sha256(
                hard_constraint_contract
            )
            self.constraint_names = constraint_names
            self.constraint_index = {
                name: index for index, name in enumerate(constraint_names)
            }
            self.n_ieq_constr = len(constraint_names)
            self.variable_cooling_dimensions = VARIABLE_COOLING_DIMENSIONS
            self.sobol_dimension_names = tuple(names)
            self.fixed_cooling_pads_mm = dict(FIXED_COOLING_PADS_MM)
            self._last_decode = None
            self._prediction_cache = None

        @staticmethod
        def _unit_from_physical(name: str, value: float) -> float:
            index = sobol_index[name]
            _dimension, lower, upper = normalized_dims[index]
            return float(np.clip((float(value) - lower) / (upper - lower), 0.0, 1.0))

        @staticmethod
        def _physical_from_unit(name: str, value: float) -> float:
            index = sobol_index[name]
            _dimension, lower, upper = normalized_dims[index]
            return lower + (upper - lower) * float(value)

        @staticmethod
        def _integer_unit(target: int, minimum: int, maximum: int) -> float:
            scale = maximum - minimum + 0.9999
            return float(np.clip((target - minimum + 0.5) / scale, 0.0, 1.0))

        def _size_limit_mm(self, axis: str) -> float:
            if self.goal_campaign:
                return float(self.spec["size_limits_mm"][axis])
            return float(self.spec[f"size_{axis}_max_mm"])

        def _selected_cw1_mm(self, coordinate: Any) -> float:
            if not self.goal_campaign:
                return float(self.spec["primary_conductor_thickness_mm"])
            return cw1_from_unit_coordinate(
                coordinate[self.cw1_coordinate_index]
            )

        def _project_row(self, raw: Any) -> Any:
            """Port the pinned 7c hard-physics coordinate projection."""

            row = np.minimum(
                np.maximum(np.asarray(raw, dtype=float), self.xl), self.xu
            ).copy()

            def get(name: str) -> float:
                return self._physical_from_unit(name, row[sobol_index[name]])

            def put(name: str, value: float) -> None:
                row[sobol_index[name]] = self._unit_from_physical(name, value)

            n1 = self.fixed_primary_turns
            row[self.fixed_primary_turn_coordinate_index] = (
                self.fixed_primary_turn_unit_coordinate
            )
            selected_cw1 = self._selected_cw1_mm(row)
            if self.goal_campaign:
                row[self.cw1_coordinate_index] = cw1_unit_coordinate(
                    selected_cw1
                )
            n2 = n1 * 10
            side_selector = float(row[sobol_index["u_N2_side"]])
            if side_selector < 0.10:
                n2_side = 0
            else:
                decoded_native = round(n2 * side_selector * 0.8)
                n2_side = int(
                    np.clip(
                        decoded_native,
                        int(np.ceil(0.20 * n2)),
                        int(np.floor(0.52 * n2)),
                    )
                )
            row[sobol_index["u_N2_side"]] = min(1.0, (n2_side + 0.1) / (0.8 * n2))

            length_limit = min(
                self._size_limit_mm(
                    "W" if self.goal_campaign else "L"
                ),
                float(normalized_dims[sobol_index["total_length"]][2]),
            )
            height_limit = min(
                self._size_limit_mm("H"),
                float(normalized_dims[sobol_index["total_height"]][2]),
            )
            put("total_height", np.clip(get("total_height"), 500.0, height_limit))

            for name in (
                "cc_w2c_space_x",
                "w2c_w1c_space_x",
                "w1c_w2s_space_x",
                "w1s_cs_space_x",
                "cc_w2c_space_y",
                "w2c_w1c_space_y",
                "cs_w1s_space_y",
                "w2s_w1s_space_x",
                "w1s_w2s_space_y",
            ):
                minimum = self.spec["insulation_min_mm"]
                if name == "w1c_w2s_space_x":
                    minimum += 0.5
                value = np.clip(get(name), minimum, 50.0)
                if n2_side and name in {
                    "cc_w2c_space_x",
                    "w2c_w1c_space_x",
                    "w1c_w2s_space_x",
                    "w1s_cs_space_x",
                }:
                    value = minimum
                put(name, value)
            put("gap1", np.clip(get("gap1"), 0.3, 5.0))
            put("gap2", np.clip(get("gap2"), 0.3, 2.0))

            defaults = input_parameter_module.get_drawing_default_params()
            plate_t = round(float(get("core_plate_t")), 1)
            core_stack = plate_t + 2.0 * FIXED_COOLING_PADS_MM["core_plate_pad_t"]
            d_min = float(defaults["core_depth_min"])
            d_max = float(defaults["core_depth_max"])
            minimum_groups = (
                GOAL_N_CORE_GROUP_MIN if self.goal_campaign else 1
            )
            maximum_groups = (
                GOAL_N_CORE_GROUP_MAX
                if self.goal_campaign
                else int(self.spec["n_core_group_max"])
            )
            floors = np.asarray([40.0, 40.0, 40.5, 40.0])
            l1_box_ceiling = (length_limit - 2.0 * float(floors.sum()) / 0.45) / 4.0
            l1_ceiling = min(100.0, l1_box_ceiling)
            if n2_side:
                projected_x_spaces = np.asarray(
                    [
                        get("cc_w2c_space_x"),
                        get("w2c_w1c_space_x"),
                        get("w1c_w2s_space_x"),
                        get("w1s_cs_space_x"),
                    ]
                )
                minimum_side_pack = n2_side * 0.3 + max(n2_side - 1, 0) * 0.3
                side_exterior_l1_ceiling = (
                    length_limit
                    - 2.0 * float(projected_x_spaces.sum()) / 0.45
                    - 2.0 * (float(projected_x_spaces[3]) + minimum_side_pack)
                ) / 4.0
                l1_ceiling = min(l1_ceiling, max(40.0, side_exterior_l1_ceiling))
            l1 = float(np.clip(round(get("l1")), 40.0, l1_ceiling))

            required_area_mm2 = (
                float(defaults["V1_rms"])
                * 1e6
                / (
                    4.0
                    * float(defaults["freq"])
                    * n1
                    * float(self.spec["core_lamination_factor"])
                    * float(self.spec["B_limit_T"])
                )
            )
            selector = float(row[sobol_index["u_ngroup"]])
            current_rounded_w1 = round(get("w1"))
            current_n_min = max(
                minimum_groups,
                int(np.ceil((current_rounded_w1 - core_stack) / (d_max + core_stack))),
            )
            current_n_max = max(
                current_n_min,
                int(np.floor((current_rounded_w1 - core_stack) / (d_min + core_stack))),
            )
            desired_group = current_n_min + int(
                selector * (current_n_max - current_n_min + 0.9999)
            )
            desired_group = int(
                np.clip(desired_group, minimum_groups, maximum_groups)
            )
            minimum_group_for_b = int(
                np.ceil(required_area_mm2 / (2.0 * max(l1, 1.0) * d_max))
            )
            target_group = int(
                np.clip(
                    max(desired_group, minimum_group_for_b),
                    minimum_groups,
                    maximum_groups,
                )
            )
            required_l1 = required_area_mm2 / (2.0 * target_group * d_max) * 1.005
            l1 = float(np.clip(max(l1, required_l1), 40.0, l1_ceiling))
            put("l1", l1)

            required_iron_depth = required_area_mm2 / (2.0 * l1) * 1.005
            minimum_iron_depth = max(target_group * d_min, required_iron_depth)
            maximum_iron_depth = target_group * d_max * (1.0 - 1e-6)
            raw_iron_depth = get("w1") - (target_group + 1.0) * core_stack
            iron_depth = float(
                np.clip(raw_iron_depth, minimum_iron_depth, maximum_iron_depth)
            )
            put("w1", iron_depth + (target_group + 1.0) * core_stack)

            def project_budget() -> None:
                projected_l1 = round(get("l1"))
                x_names = (
                    "cc_w2c_space_x",
                    "w2c_w1c_space_x",
                    "w1c_w2s_space_x",
                    "w1s_cs_space_x",
                )
                spaces = np.asarray([max(get(name), 40.0) for name in x_names])
                spaces[2] = max(spaces[2], 40.5)
                total_length = min(round(get("total_length")), length_limit)
                minimum_length = 4.0 * projected_l1 + 2.0 * float(spaces.sum()) / 0.45
                if minimum_length > length_limit:
                    available = max(
                        float(floors.sum()),
                        0.45 * (length_limit - 4.0 * projected_l1) / 2.0,
                    )
                    excess = max(0.0, float(spaces.sum()) - available)
                    reducible = spaces - floors
                    if reducible.sum() > 0.0:
                        spaces -= reducible * min(1.0, excess / reducible.sum())
                    minimum_length = (
                        4.0 * projected_l1 + 2.0 * float(spaces.sum()) / 0.45
                    )
                for name, value in zip(x_names, spaces):
                    put(name, value)

                gap1 = np.clip(round(get("gap1"), 1), 0.3, 5.0)
                gap2 = np.clip(round(get("gap2"), 3), 0.3, 2.0)
                primary_pack = (
                    n1 * selected_cw1
                    + max(n1 - 1, 0) * gap1
                )
                secondary_gaps = max(n2 - n2_side - 1, 0) + max(n2_side - 1, 0)
                secondary_pack = n2 * 0.3 + secondary_gaps * gap2
                required_l2 = float(spaces.sum()) + primary_pack + secondary_pack
                winding_minimum_length = 4.0 * projected_l1 + 2.0 * required_l2
                if winding_minimum_length > length_limit and secondary_gaps:
                    available_gap_budget = (
                        (length_limit - 4.0 * projected_l1) / 2.0
                        - float(spaces.sum())
                        - primary_pack
                        - n2 * 0.3
                    )
                    gap2 = np.clip(available_gap_budget / secondary_gaps, 0.3, gap2)
                    secondary_pack = n2 * 0.3 + secondary_gaps * gap2
                    winding_minimum_length = 4.0 * projected_l1 + 2.0 * (
                        float(spaces.sum()) + primary_pack + secondary_pack
                    )
                total_length = np.clip(
                    np.ceil(
                        max(
                            total_length,
                            minimum_length + 1.0,
                            winding_minimum_length,
                        )
                    ),
                    500.0,
                    length_limit,
                )

                def secondary_packs(length: float) -> tuple[float, float]:
                    decoded_l2 = (round(length) - 4.0 * projected_l1) / 2.0
                    winding_budget = decoded_l2 - float(spaces.sum())
                    available_secondary = winding_budget - primary_pack
                    cw2 = (available_secondary - secondary_gaps * gap2) / n2
                    cw2 = round(cw2, 3)
                    main_turns = n2 - n2_side
                    main_pack = main_turns * cw2 + max(main_turns - 1, 0) * gap2
                    side_pack = (
                        n2_side * cw2 + max(n2_side - 1, 0) * gap2 if n2_side else 0.0
                    )
                    return main_pack, side_pack

                main_pack, side_pack = secondary_packs(total_length)
                if n2_side:
                    exterior_x = total_length + 2.0 * (float(spaces[3]) + side_pack)
                    if exterior_x > length_limit and gap2 > 0.3:
                        gap2 = 0.3
                        secondary_pack = n2 * 0.3 + secondary_gaps * gap2
                        winding_minimum_length = 4.0 * projected_l1 + 2.0 * (
                            float(spaces.sum()) + primary_pack + secondary_pack
                        )
                        main_pack, side_pack = secondary_packs(total_length)
                        exterior_x = total_length + 2.0 * (float(spaces[3]) + side_pack)
                    excess = max(0.0, exterior_x - length_limit)
                    if excess:
                        side_fraction = n2_side / n2
                        reduced = np.floor(
                            total_length - excess / (1.0 + side_fraction) - 1.0
                        )
                        total_length = max(
                            np.ceil(
                                max(
                                    minimum_length + 1.0,
                                    winding_minimum_length,
                                )
                            ),
                            reduced,
                        )
                        main_pack, side_pack = secondary_packs(total_length)

                slot = (
                    round(get("wcp_t"), 1) + 2.0 * (FIXED_COOLING_PADS_MM["wcp_pad_t"])
                )
                tx_y_gap_sum = 2.0 * slot + max(n1 - 3, 0) * gap1
                primary_y_pack = (
                    n1 * selected_cw1 + tx_y_gap_sum
                )
                center_y_allowance = 2.0 * (
                    get("cc_w2c_space_y")
                    + main_pack
                    + get("w2c_w1c_space_y")
                    + primary_y_pack
                )
                side_y_allowance = (
                    2.0 * (get("cs_w1s_space_y") + side_pack) if n2_side else 0.0
                )
                maximum_core_width = self._size_limit_mm(
                    "L" if self.goal_campaign else "W"
                ) - max(
                    center_y_allowance, side_y_allowance
                )
                put("w1", min(get("w1"), np.floor(maximum_core_width)))
                put("total_length", total_length)
                put("gap1", gap1)
                put("gap2", gap2)

                h1 = round(get("total_height")) - 2.0 * projected_l1
                if h1 > 0.0:
                    wh2_limit = 1.0 - 2.0 * self.spec["insulation_min_mm"] / h1
                    put("wh2", min(get("wh2"), wh2_limit))

            project_budget()
            rounded_w1 = round(get("w1"))
            n_min = max(
                minimum_groups,
                int(np.ceil((rounded_w1 - core_stack) / (d_max + core_stack))),
            )
            n_max = max(
                n_min,
                int(np.floor((rounded_w1 - core_stack) / (d_min + core_stack))),
            )
            target_group = int(
                np.clip(
                    target_group,
                    n_min,
                    min(n_max, maximum_groups),
                )
            )
            row[sobol_index["u_ngroup"]] = np.clip(
                (target_group - n_min + 0.5) / (n_max - n_min + 0.9999),
                0.0,
                1.0,
            )
            row[self.fixed_primary_turn_coordinate_index] = (
                self.fixed_primary_turn_unit_coordinate
            )
            return np.minimum(np.maximum(row, self.xl), self.xu)

        def repair_unit_coordinates(self, coordinates: Any) -> Any:
            values = np.asarray(coordinates, dtype=float)
            one_dimensional = values.ndim == 1
            if one_dimensional:
                values = values.reshape(1, -1)
            if values.ndim != 2 or values.shape[1] != self.n_var:
                raise ValueError("NSGA coordinates have the wrong shape")
            if not np.isfinite(values).all():
                raise ValueError("NSGA coordinates must be finite")
            repaired = values.copy()
            repaired[:, self.fixed_primary_turn_coordinate_index] = (
                self.fixed_primary_turn_unit_coordinate
            )
            for _iteration in range(MAX_REPAIR_FIXED_POINT_ITERATIONS):
                projected = np.vstack([self._project_row(row) for row in repaired])
                projected[:, self.fixed_primary_turn_coordinate_index] = (
                    self.fixed_primary_turn_unit_coordinate
                )
                if np.array_equal(projected, repaired):
                    break
                repaired = projected
            else:
                raise RuntimeError(
                    "physics repair did not reach an exact coordinate fixed point"
                )
            return repaired[0] if one_dimensional else repaired

        def decode_batch(self, values: Any) -> tuple[Any, Any, Any]:
            import pandas as pd

            coordinates = np.asarray(values, dtype=float)
            if coordinates.ndim != 2 or coordinates.shape[1] != self.n_var:
                raise RuntimeError("current7 decoder coordinate schema mismatch")
            repaired = np.asarray(
                self.repair_unit_coordinates(coordinates), dtype=float
            )
            if not np.array_equal(repaired, coordinates):
                raise RuntimeError(
                    "optimizer evaluation bypassed current7 physics repair"
                )
            rows = []
            shrink = np.zeros(len(coordinates), dtype=float)
            valid = np.ones(len(coordinates), dtype=bool)
            selected_cw1_values = np.asarray(
                [self._selected_cw1_mm(row) for row in coordinates],
                dtype=float,
            )
            for index, coordinate in enumerate(coordinates):
                try:
                    sample = input_parameter_module.unit_to_dims(coordinate)
                    decoded = decode_unit_sample_with_cw1(
                        input_parameter_module,
                        sample,
                        selected_cw1_mm=selected_cw1_values[index],
                        allow_space_shrink=False,
                        space_min=self.spec["insulation_min_mm"],
                    )
                    shrink[index] = decoded.pop("_space_shrink_needed", 0.0)
                    decoded.update(self.fixed_overrides)
                    frame_input = input_parameter_module.create_input_parameter(
                        {
                            key: decoded[key]
                            for key in input_parameter_module.KEYS
                            if key in decoded
                        }
                    )
                    ok, derived = input_parameter_module.validation_check(
                        frame_input, strict=False
                    )
                    valid[index] = bool(ok)
                    rows.append(derived.iloc[0])
                except Exception:
                    valid[index] = False
                    rows.append(pd.Series(dtype=float))
            frame = pd.DataFrame(rows).reset_index(drop=True)
            valid = np.asarray(valid, dtype=bool).reshape(-1)
            if len(valid) != len(values) or len(frame) != len(values):
                raise RuntimeError("current7 decoder returned an invalid row count")
            for index in np.flatnonzero(valid):
                row = _frame_row(frame, int(index))
                for name, expected in {
                    "cw1": selected_cw1_values[index],
                    **FIXED_COOLING_PADS_MM,
                }.items():
                    if not np.isclose(
                        _finite_number(_row_value(row, name), name),
                        expected,
                        rtol=0.0,
                        atol=1e-12,
                    ):
                        raise RuntimeError(f"post-decode fixed control escaped: {name}")
                for name in VARIABLE_COOLING_DIMENSIONS:
                    _finite_number(_row_value(row, name), name)
                budget = winding_budget_identity(
                    row,
                    expected_cw1_mm=selected_cw1_values[index],
                )
                if budget.get("passed") is not True:
                    raise RuntimeError(
                        "decoded winding budget identity failed: "
                        f"{budget.get('reason', 'numerical_mismatch')}"
                    )
                observed_n1 = int(
                    _finite_number(_row_value(row, "N1_main"), "N1_main")
                ) + int(_finite_number(_row_value(row, "N1_side"), "N1_side"))
                if observed_n1 != self.fixed_primary_turns:
                    raise RuntimeError("decoded primary turns escaped fixed stratum")
                if self.goal_campaign:
                    validate_cw1_mm(_row_value(row, "cw1"))
                    identity_values = _jsonable_decoded_parameters(row)
                    identity_values["thermal_pad_conductivity_W_mK"] = (
                        GOAL_FIXED_COOLING_IDENTITY[
                            "thermal_pad_conductivity_W_mK"
                        ]
                    )
                    attest_fixed_identity(identity_values)
            self._last_decode = (frame, np.asarray(shrink, dtype=float), valid)
            return frame, shrink, valid

        def hard_geometry_audit(self, coordinates: Any) -> dict[str, Any]:
            values = np.asarray(coordinates, dtype=float)
            if values.ndim == 1:
                values = values.reshape(1, -1)
            repaired = np.asarray(self.repair_unit_coordinates(values), dtype=float)
            frame, shrink, decoder_valid = self.decode_batch(repaired)
            insulation = minimum_physical_insulation_violation(
                frame,
                self.spec["insulation_min_mm"],
                invalid_value=BIG,
            )
            count = len(repaired)
            groups = np.zeros(count, dtype=bool)
            boxes = np.zeros(count, dtype=bool)
            flux = np.zeros(count, dtype=bool)
            budgets = np.zeros(count, dtype=bool)
            turns = np.zeros(count, dtype=bool)
            identities = [None] * count
            budget_evidence = [None] * count
            for index in range(count):
                if not decoder_valid[index]:
                    continue
                row = frame.iloc[index]
                try:
                    if self.goal_campaign:
                        groups[index] = (
                            dynamic_core_group_violation(row) <= 0.0
                        )
                    else:
                        groups[index] = _finite_number(
                            row["n_core_group"], "n_core_group"
                        ) <= float(self.spec["n_core_group_max"])
                    _volume, dimensions = bounding_box_lit(row)
                    boxes[index] = all(
                        _finite_number(observed, "box dimension") <= limit
                        for observed, limit in zip(
                            dimensions,
                            (
                                self._size_limit_mm("W"),
                                self._size_limit_mm("L"),
                                self._size_limit_mm("H"),
                            ),
                        )
                    )
                    flux[index] = _finite_number(
                        design_analytical_b_field_t(
                            row,
                            core_lamination_factor=self.spec["core_lamination_factor"],
                            area_basis=self.spec["B_area_basis"],
                        ),
                        "analytical B",
                    ) <= float(self.spec["B_limit_T"])
                    budget = winding_budget_identity(
                        row,
                        expected_cw1_mm=_finite_number(row["cw1"], "cw1"),
                    )
                    budget_evidence[index] = budget
                    budgets[index] = budget.get("passed") is True
                    observed_n1 = int(row["N1_main"]) + int(row["N1_side"])
                    turns[index] = observed_n1 == self.fixed_primary_turns
                    identities[index] = tuple(
                        row.get(column, None)
                        for column in DECODED_GEOMETRY_IDENTITY_COLUMNS
                    )
                except (KeyError, RuntimeError, TypeError, ValueError, OverflowError):
                    continue
            masks = {
                "decoder": np.asarray(decoder_valid, dtype=bool),
                "shrink": np.isfinite(shrink) & (shrink <= 1e-9),
                "minimum_physical_insulation": (
                    np.isfinite(insulation) & (insulation <= 1e-9)
                ),
                "core_group": groups,
                "box": boxes,
                "analytical_B": flux,
                "winding_budget_identity": budgets,
                "fixed_primary_turns": turns,
            }
            joint = np.logical_and.reduce(tuple(masks.values()))
            unique = {
                identities[index]
                for index in np.flatnonzero(joint)
                if identities[index] is not None
            }
            return {
                "count": count,
                "counts": {
                    name: int(np.count_nonzero(mask)) for name, mask in masks.items()
                },
                "joint_count": int(np.count_nonzero(joint)),
                "unique_geometry_count": len(unique),
                "gate_masks": masks,
                "joint_mask": joint,
                "geometry_identities": identities,
                "budget_evidence": budget_evidence,
                "frame": frame,
                "repaired_coordinates": repaired,
                "shrink": shrink,
                "insulation_violation": insulation,
            }

        def filter_warm_start_coordinates(
            self, coordinates: Any
        ) -> tuple[Any, dict[str, Any]]:
            values = np.asarray(coordinates, dtype=float)
            audit = self.hard_geometry_audit(values)
            kept = []
            seen = set()
            repaired = audit["repaired_coordinates"]
            for index in np.flatnonzero(audit["joint_mask"]):
                identity = audit["geometry_identities"][int(index)]
                if identity in seen:
                    continue
                seen.add(identity)
                kept.append(repaired[int(index)])
            result = (
                np.asarray(kept, dtype=float).reshape(-1, self.n_var)
                if kept
                else np.empty((0, self.n_var), dtype=float)
            )
            return result, {
                "input_count": int(len(values)),
                "hard_feasible_count": audit["joint_count"],
                "decoded_unique_count": int(len(result)),
                "rejected_count": int(len(values) - len(result)),
                "fixed_primary_turns": self.fixed_primary_turns,
                "winding_budget_identity_passed_count": audit["counts"][
                    "winding_budget_identity"
                ],
            }

        def filter_structural_donor_coordinates(
            self, coordinates: Any
        ) -> tuple[Any, dict[str, Any]]:
            """Keep coordinate-only donors without claiming hard feasibility.

            Basin donors are allowed to violate optimizer-facing shrink, box,
            and analytical-flux gates.  Those physical ``G`` terms remain in
            the unchanged problem evaluation and terminal replay.  Decoder,
            manufacturing, insulation, winding-budget, and fixed-turn
            identities are mandatory before a donor may seed a protected
            topology slot.
            """

            values = np.asarray(coordinates, dtype=float)
            audit = self.hard_geometry_audit(values)
            gate_masks = audit["gate_masks"]
            if not set(STRUCTURAL_DONOR_REQUIRED_GATES) <= set(gate_masks):
                raise RuntimeError("structural donor gate inventory drifted")
            structural_mask = np.logical_and.reduce(
                tuple(
                    np.asarray(gate_masks[name], dtype=bool)
                    for name in STRUCTURAL_DONOR_REQUIRED_GATES
                )
            )
            repaired = audit["repaired_coordinates"]
            kept = []
            seen = set()
            for index in np.flatnonzero(structural_mask):
                identity = audit["geometry_identities"][int(index)]
                if identity is None or identity in seen:
                    continue
                seen.add(identity)
                kept.append(repaired[int(index)])
            result = (
                np.asarray(kept, dtype=float).reshape(-1, self.n_var)
                if kept
                else np.empty((0, self.n_var), dtype=float)
            )
            evidence = {
                "schema_version": (
                    "mft-tier1-coordinate-only-structural-donor-filter-v1"
                ),
                "input_count": int(len(values)),
                "structurally_accepted_count": int(np.count_nonzero(structural_mask)),
                "decoded_unique_count": int(len(result)),
                "structurally_rejected_count": int(
                    len(values) - np.count_nonzero(structural_mask)
                ),
                "duplicate_geometry_count": int(
                    np.count_nonzero(structural_mask) - len(result)
                ),
                "required_gate_names": list(STRUCTURAL_DONOR_REQUIRED_GATES),
                "required_gate_passed_counts": {
                    name: int(np.count_nonzero(gate_masks[name]))
                    for name in STRUCTURAL_DONOR_REQUIRED_GATES
                },
                "optimizer_gate_names_not_used_for_donor_admission": list(
                    STRUCTURAL_DONOR_OPTIMIZER_GATES
                ),
                "optimizer_gate_passed_counts": {
                    name: int(np.count_nonzero(gate_masks[name]))
                    for name in STRUCTURAL_DONOR_OPTIMIZER_GATES
                },
                "coordinate_sha256": canonical_sha256(result.tolist()),
                "coordinate_donors_only": True,
                "ordinary_warm_sampling_allowed": False,
                "prior_objectives_or_constraints_inherited": False,
                "additional_model_evaluations": 0,
                "physical_constraint_G_mutation": False,
                "physical_objective_mutation": False,
                "terminal_physical_replay_required": True,
            }
            evidence["sha256"] = canonical_sha256(evidence)
            return result, evidence

        def _predict(self, target: str, frame: Any) -> tuple[Any, Any]:
            cache_key = (str(target), tuple(getattr(frame, "index", range(len(frame)))))
            if (
                self._prediction_cache is not None
                and cache_key in self._prediction_cache
            ):
                return self._prediction_cache[cache_key]
            try:
                mean, half_width = self.models[target].predict_mu_sigma(
                    frame, conformal=True
                )
            except TypeError as exc:
                raise RuntimeError(
                    f"surrogate does not accept explicit conformal=True: {target}"
                ) from exc
            mean = np.asarray(mean, dtype=float).reshape(-1)
            half_width = np.asarray(half_width, dtype=float).reshape(-1)
            if (
                len(mean) != len(frame)
                or len(half_width) != len(frame)
                or not np.isfinite(mean).all()
                or not np.isfinite(half_width).all()
                or np.any(half_width < 0.0)
            ):
                raise RuntimeError(
                    f"surrogate returned invalid q90 half-width output: {target}"
                )
            result = (mean, half_width)
            if self._prediction_cache is not None:
                self._prediction_cache[cache_key] = result
            return result

        def _evaluate(
            self, values: Any, out: dict[str, Any], *args: Any, **kwargs: Any
        ) -> None:
            self._last_decode = None
            self._prediction_cache = {}
            try:
                super()._evaluate(values, out, *args, **kwargs)
                if self._last_decode is None:
                    raise RuntimeError("current7 base evaluation bypassed the decoder")
                frame, _shrink, valid = self._last_decode
                objectives = np.asarray(out.get("F"), dtype=float)
                constraints = np.asarray(out.get("G"), dtype=float)
                expected_shape = (len(values), len(self.constraint_names))
                if objectives.shape != (len(values), 2):
                    raise RuntimeError("current7 objective shape mismatch")
                if constraints.shape != expected_shape:
                    raise RuntimeError("current7 constraint shape mismatch")
                additive = constraints[:, len(self.base_constraint_names) :]
                if not np.all(additive == BIG):
                    raise RuntimeError(
                        "current7 base wrote into additive hard constraints"
                    )

                indices = np.flatnonzero(valid)
                if len(indices):
                    sub = (
                        frame.iloc[indices]
                        if hasattr(frame, "iloc")
                        else [frame[int(index)] for index in indices]
                    )
                    side_targets = (
                        GOAL_SIDE_TEMPERATURE_TARGETS
                        if self.goal_campaign
                        else (SIDE_TEMPERATURE_TARGET,)
                    )
                    for side_target in side_targets:
                        side_index = self.constraint_index[
                            f"temperature_robust_limit:{side_target}"
                        ]
                        constraints[indices, side_index] = (
                            apply_current7_side_temperature_condition(
                                constraints[indices, side_index],
                                sub,
                                invalid_value=BIG,
                            )
                        )
                    insulation_index = self.constraint_index[
                        "minimum_physical_insulation"
                    ]
                    group_index = self.constraint_index[
                        (
                            "core_group_dynamic_validity"
                            if self.goal_campaign
                            else "core_group_manufacturability_limit"
                        )
                    ]
                    constraints[indices, insulation_index] = (
                        minimum_physical_insulation_violation(
                            sub,
                            self.spec["insulation_min_mm"],
                            invalid_value=BIG,
                        )
                    )

                    for local_index, global_index in enumerate(indices):
                        row = _frame_row(sub, local_index)
                        try:
                            constraints[global_index, group_index] = (
                                dynamic_core_group_violation(row)
                                if self.goal_campaign
                                else (
                                    _finite_number(
                                        _row_value(row, "n_core_group"),
                                        "n_core_group",
                                    )
                                    - float(self.spec["n_core_group_max"])
                                )
                            )
                        except (RuntimeError, ValueError):
                            constraints[global_index, group_index] = BIG

                    mean_llt, _ = self._predict("Llt_phys", sub)
                    mean_k, _ = self._predict("k", sub)
                    mean_c_tx, _ = self._predict("C_tx_tx_F", sub)
                    mean_c_rx, _ = self._predict("C_rx_rx_F", sub)
                    for local_index, global_index in enumerate(indices):
                        row = _frame_row(sub, local_index)
                        try:
                            screen = derive_half_magnetizing_self_resonance(
                                {
                                    "Llt_phys": mean_llt[local_index],
                                    "k": mean_k[local_index],
                                    "C_tx_tx_F": mean_c_tx[local_index],
                                    "C_rx_rx_F": mean_c_rx[local_index],
                                },
                                row,
                                magnetizing_inductance_factor=self.spec[
                                    "magnetizing_inductance_factor"
                                ],
                            )
                            minimum = screen["f_res_min_tx_rx_only_Hz"]
                            resonance_g = _resonance_band_violations(
                                minimum,
                                self.stage_spec["resonance_min_Hz"],
                                (
                                    None
                                    if self.goal_campaign
                                    else self.stage_spec["resonance_max_Hz"]
                                ),
                            )
                            for resonance_name, violation in resonance_g.items():
                                constraints[
                                    global_index,
                                    self.constraint_index[resonance_name],
                                ] = violation
                        except (
                            RuntimeError,
                            ValueError,
                            OverflowError,
                            ZeroDivisionError,
                        ):
                            for resonance_name in (
                                RESONANCE_MINIMUM_CONSTRAINT,
                                RESONANCE_MAXIMUM_CONSTRAINT,
                            ):
                                if resonance_name in self.constraint_names:
                                    constraints[
                                        global_index,
                                        self.constraint_index[resonance_name],
                                    ] = BIG

                        try:
                            _volume, dimensions = bounding_box_lit(row)
                            dimensions = tuple(
                                _finite_number(value, "exterior dimension")
                                for value in dimensions
                            )
                            if len(dimensions) != 3:
                                raise RuntimeError("exterior dimension count mismatch")
                            limits = (
                                self._size_limit_mm("W"),
                                self._size_limit_mm("L"),
                                self._size_limit_mm("H"),
                            )
                            for name, observed, limit in zip(
                                SIZE_CONSTRAINT_NAMES, dimensions, limits
                            ):
                                constraints[
                                    global_index, self.constraint_index[name]
                                ] = observed - float(limit)
                        except (
                            RuntimeError,
                            KeyError,
                            TypeError,
                            ValueError,
                            OverflowError,
                        ):
                            for name in SIZE_CONSTRAINT_NAMES:
                                constraints[
                                    global_index, self.constraint_index[name]
                                ] = BIG

                objectives[~np.isfinite(objectives)] = BIG
                constraints[~np.isfinite(constraints)] = BIG
                out["F"] = objectives
                out["G"] = constraints
                out["frame"] = frame
                out["decoder_valid"] = valid
            finally:
                self._prediction_cache = None

    Current7Tier1Problem.__name__ = "Current7Tier1Problem"
    Current7Tier1Problem.__qualname__ = "Current7Tier1Problem"
    return Current7Tier1Problem


@dataclass(frozen=True)
class Current7Modules:
    run_nsga2: Any
    nsga2_problem: Any
    predictor: Any
    train_models: Any
    geometry_metrics: Any
    design_summary: Any
    input_parameter: Any
    evidence: dict[str, Any]


def _path_is_below(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath([str(path.resolve()), str(root.resolve())]) == str(
            root.resolve()
        )
    except (OSError, ValueError):
        return False


def _module_path(module: Any, code_root: Path, label: str) -> Path:
    path = Path(str(getattr(module, "__file__", ""))).resolve(strict=True)
    if not path.is_file() or not _path_is_below(path, code_root):
        raise RuntimeError(f"current7 module escaped authenticated code root: {label}")
    return path


def load_current7_modules(code_root: Path) -> Current7Modules:
    """Import and attest the exact current optimizer implementation."""

    root = code_root.resolve(strict=True)
    regression_root = root / "regression_260707"
    training_root = regression_root / "training"
    if not regression_root.is_dir() or not training_root.is_dir():
        raise RuntimeError("authenticated code root has no current7 optimizer tree")
    for path in (root, regression_root, training_root):
        token = str(path)
        if token not in sys.path:
            sys.path.insert(0, token)

    modules = {
        "run_nsga2": importlib.import_module("optimization.run_nsga2"),
        "nsga2_problem": importlib.import_module("optimization.nsga2_problem"),
        "predictor": importlib.import_module("predictor"),
        "train_models": importlib.import_module("train_models"),
        "geometry_metrics": importlib.import_module("optimization.geometry_metrics"),
        "design_summary": importlib.import_module("optimization.design_summary"),
        "input_parameter": importlib.import_module("module.input_parameter_260706"),
    }
    paths = {name: _module_path(module, root, name) for name, module in modules.items()}
    run_nsga2 = modules["run_nsga2"]
    nsga2_problem = modules["nsga2_problem"]
    predictor = modules["predictor"]
    input_parameter = modules["input_parameter"]
    if tuple(run_nsga2.REQUIRED_MODEL_TARGETS) != CURRENT_REQUIRED_MODEL_TARGETS:
        raise RuntimeError("current run_nsga2 required-model contract mismatch")
    if tuple(nsga2_problem.T_TARGETS) != CURRENT_TEMPERATURE_TARGETS:
        raise RuntimeError("current run_nsga2 temperature target contract mismatch")
    if tuple(nsga2_problem.CONSTRAINT_NAMES) != BASE_CONSTRAINT_NAMES:
        raise RuntimeError("current run_nsga2 base constraint contract mismatch")
    if (
        dict(nsga2_problem.NSGA_FIXED_THERMAL_STACK_MM)
        != EXPECTED_SIMPLE_BASE_FIXED_STACK_MM
    ):
        raise RuntimeError("current run_nsga2 simple cooling clamp contract mismatch")
    if not callable(getattr(run_nsga2, "run_one", None)):
        raise RuntimeError("current run_nsga2 has no run_one interface")
    signature = inspect.signature(predictor.EnsemblePredictor.predict_mu_sigma)
    conformal = signature.parameters.get("conformal")
    if conformal is None or conformal.default is not True:
        raise RuntimeError(
            "corrected predictor no longer defaults to q90 conformal output"
        )
    evidence = {
        "module_files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sorted(paths.items())
        },
        "required_model_targets_sha256": CURRENT_REQUIRED_MODEL_TARGETS_SHA256,
        "temperature_contract_sha256": CURRENT_TEMPERATURE_CONTRACT_SHA256,
        "base_constraint_names": list(BASE_CONSTRAINT_NAMES),
        "base_constraint_count": len(BASE_CONSTRAINT_NAMES),
        "simple_base_fixed_stack_mm": EXPECTED_SIMPLE_BASE_FIXED_STACK_MM,
        "predict_mu_sigma_conformal_default": True,
        "run_interface": "optimization.run_nsga2.run_one",
    }
    return Current7Modules(
        run_nsga2=run_nsga2,
        nsga2_problem=nsga2_problem,
        predictor=predictor,
        train_models=modules["train_models"],
        geometry_metrics=modules["geometry_metrics"],
        design_summary=modules["design_summary"],
        input_parameter=input_parameter,
        evidence=evidence,
    )


def install_optimizer_scaling(
    problem: Any,
    *,
    resonance_scale_hz: float,
    llt_scale_uh: float,
    all_thermal_scale_c: float,
    resonance_allowance_hz: float | None = None,
    llt_allowance_uh: float | None = None,
    topology_allowance_contract: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Normalize optimizer G while retaining the physical evaluator for replay."""

    import numpy as np

    if getattr(problem, "_tier1_optimizer_scaling_installed", False):
        raise RuntimeError("optimizer scaling was already installed")
    resonance_scale = _positive_number(resonance_scale_hz, "resonance scale")
    llt_scale = _positive_number(llt_scale_uh, "Llt scale")
    thermal_scale = _positive_number(all_thermal_scale_c, "thermal scale")
    if (resonance_allowance_hz is None) != (llt_allowance_uh is None):
        raise RuntimeError("optimizer resonance/Llt allowances must be paired")
    resonance_allowance = (
        0.0
        if resonance_allowance_hz is None
        else _finite_number(resonance_allowance_hz, "resonance allowance")
    )
    llt_allowance = (
        0.0
        if llt_allowance_uh is None
        else _finite_number(llt_allowance_uh, "Llt allowance")
    )
    if resonance_allowance < 0.0 or llt_allowance < 0.0:
        raise ValueError("optimizer allowances must be non-negative")
    names = tuple(problem.constraint_names)
    dynamic_niche = None
    if topology_allowance_contract is not None:
        try:
            from tier1_final1000_topology_niche_contract import (
                optimizer_allowances as niche_optimizer_allowances,
                validate_contract as validate_niche_contract,
            )
        except ImportError:  # pragma: no cover - repository module import path
            from tools.tier1_final1000_topology_niche_contract import (
                optimizer_allowances as niche_optimizer_allowances,
                validate_contract as validate_niche_contract,
            )
        dynamic_niche = validate_niche_contract(topology_allowance_contract)
    else:
        niche_optimizer_allowances = None
    scales = {name: 1.0 for name in names}
    allowances = {name: 0.0 for name in names}
    scales["Llt_robust_band"] = llt_scale
    for resonance_name in (
        RESONANCE_MINIMUM_CONSTRAINT,
        RESONANCE_MAXIMUM_CONSTRAINT,
    ):
        if resonance_name in names:
            scales[resonance_name] = resonance_scale
    allowances["Llt_robust_band"] = llt_allowance
    for resonance_name in (
        RESONANCE_MINIMUM_CONSTRAINT,
        RESONANCE_MAXIMUM_CONSTRAINT,
    ):
        if resonance_name in names:
            allowances[resonance_name] = resonance_allowance
    for name in names:
        if name.startswith("temperature_robust_limit:"):
            scales[name] = thermal_scale
    scale_vector = np.asarray([scales[name] for name in names], dtype=float)
    allowance_vector = np.asarray([allowances[name] for name in names], dtype=float)
    if (
        not np.isfinite(scale_vector).all()
        or np.any(scale_vector <= 0.0)
        or not np.isfinite(allowance_vector).all()
        or np.any(allowance_vector < 0.0)
    ):
        raise RuntimeError("optimizer scaling vector is invalid")
    physical_evaluate = problem._evaluate
    problem._tier1_optimizer_generation = 0

    def optimizer_evaluate(
        values: Any,
        out: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        physical_evaluate(values, out, *args, **kwargs)
        physical_g = np.asarray(out.get("G"), dtype=float)
        if physical_g.ndim != 2 or physical_g.shape[1] != len(names):
            raise RuntimeError("physical G shape escaped optimizer normalization")
        dynamic_allowance = np.zeros_like(physical_g)
        if dynamic_niche is not None:
            topologies = _turn_split_main_values(
                values,
                fixed_primary_turns=6,
                coordinate_index=int(dynamic_niche["coordinate_index"]),
            )
            generation = int(
                getattr(problem, "_tier1_optimizer_generation", 0) or 0
            )
            for topology in np.unique(topologies):
                allowance = niche_optimizer_allowances(
                    int(topology), generation
                )
                rows = np.flatnonzero(topologies == topology)
                for index, name in enumerate(names):
                    if name in {
                        "Llt_robust_band",
                        "Llt_ensemble_disagreement",
                    }:
                        dynamic_allowance[rows, index] = allowance["Llt_uH"]
                    elif name.startswith("temperature_robust_limit:"):
                        dynamic_allowance[rows, index] = allowance[
                            "temperature_C"
                        ]
                    elif name in {
                        RESONANCE_MINIMUM_CONSTRAINT,
                        RESONANCE_MAXIMUM_CONSTRAINT,
                    }:
                        dynamic_allowance[rows, index] = allowance[
                            "resonance_Hz"
                        ]
        out["G"] = (
            physical_g - allowance_vector - dynamic_allowance
        ) / scale_vector

    contract = {
        "schema_version": "mft-tier1-current7-optimizer-scaling-v1",
        "constraint_names": list(names),
        "scales": scales,
        "allowances": allowances,
        "scale_vector": scale_vector.tolist(),
        "allowance_vector": allowance_vector.tolist(),
        "physical_constraint_sign_preserved_without_allowance": (
            resonance_allowance == 0.0
            and llt_allowance == 0.0
            and dynamic_niche is None
        ),
        "dynamic_topology_allowance_contract": dynamic_niche,
        "dynamic_topology_allowance_contract_sha256": (
            None if dynamic_niche is None else dynamic_niche["sha256"]
        ),
        "dynamic_allowances_zero_from_generation": (
            None
            if dynamic_niche is None
            else dynamic_niche["allowances_zero_from_generation"]
        ),
        "physical_hard_constraint_mutation": False,
        "physical_objective_mutation": False,
        "terminal_unscaled_physical_replay_required": True,
    }
    contract["sha256"] = canonical_sha256(contract)
    problem._evaluate = optimizer_evaluate
    problem._tier1_optimizer_scaling_installed = True
    return physical_evaluate, contract


def validate_warm_role_partition(
    partition: Mapping[str, Any],
    coordinates: Any,
    *,
    fixed_primary_turns: int,
) -> dict[str, Any]:
    """Authenticate disjoint standard-warm and structural-donor row roles."""

    import numpy as np

    values = np.asarray(coordinates, dtype=float)
    if values.ndim != 2:
        raise RuntimeError("warm role partition coordinate schema mismatch")
    value = dict(partition or {})
    unsigned = dict(value)
    recorded_sha = unsigned.pop("sha256", None)
    standard = value.get("standard_hard_feasible_candidates") or {}
    basin = value.get("basin_structural_coordinate_donors") or {}
    standard_start = standard.get("start")
    standard_count = standard.get("count")
    basin_start = basin.get("start")
    basin_count = basin.get("count")
    integer_fields = (
        standard_start,
        standard_count,
        basin_start,
        basin_count,
        value.get("total_count"),
    )
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in integer_fields
    ):
        raise RuntimeError("warm role partition ranges are invalid")
    if (
        value.get("schema_version") != WARM_ROLE_PARTITION_SCHEMA
        or recorded_sha != canonical_sha256(unsigned)
        or int(fixed_primary_turns) != 6
        or value.get("fixed_primary_turns") != int(fixed_primary_turns)
        or standard_start != 0
        or standard_count < 1
        or basin_start != standard_count
        or basin_count < 1
        or value.get("total_count") != len(values)
        or standard_count + basin_count != len(values)
        or standard.get("full_hard_geometry_filter_required") is not True
        or standard.get("ordinary_warm_sampling_allowed") is not True
        or len(str(standard.get("source_artifact_sha256") or "")) != 64
        or len(str(standard.get("source_contract_file_sha256") or "")) != 64
        or not isinstance(standard.get("source_hard_geometry_joint_count"), int)
        or standard.get("source_hard_geometry_joint_count") < 1
        or basin.get("coordinate_donors_only") is not True
        or basin.get("protected_topology_slots_only") is not True
        or basin.get("ordinary_warm_sampling_allowed") is not False
        or basin.get("required_gate_names") != list(STRUCTURAL_DONOR_REQUIRED_GATES)
        or basin.get("optimizer_gate_names_not_used_for_donor_admission")
        != list(STRUCTURAL_DONOR_OPTIMIZER_GATES)
        or basin.get("required_topologies_N2_main")
        != list(deep_topology_contract(6)["turn_split_sub_islands_N2_main"])
        or len(str(basin.get("source_selection_contract_sha256") or "")) != 64
        or value.get("maximum_combined_authenticated_fraction") != 0.5
        or value.get("minimum_fresh_random_fraction") != 0.5
        or value.get("physical_constraint_G_mutation") is not False
        or value.get("physical_objective_mutation") is not False
        or value.get("terminal_physical_replay_required") is not True
    ):
        raise RuntimeError("warm role partition contract mismatch")
    standard_values = values[standard_start : standard_start + standard_count]
    basin_values = values[basin_start : basin_start + basin_count]
    if standard.get("coordinate_unit_sha256") != canonical_sha256(
        standard_values.tolist()
    ) or basin.get("coordinate_unit_sha256") != canonical_sha256(basin_values.tolist()):
        raise RuntimeError("warm role partition coordinate identity mismatch")
    return value


def _terminal_inference_binding_contract(
    binding: Mapping[str, Any],
) -> tuple[bool, int]:
    """Reject partial thread bindings while allowing empty unit-test runners."""

    if not isinstance(binding, Mapping):
        raise RuntimeError("terminal replay inference binding must be a mapping")
    if not binding:
        return False, DETERMINISTIC_TERMINAL_INFERENCE_THREADS
    required = {
        "threads_per_model",
        "target_count",
        "model_count",
        "families",
        "family_threads",
        "semaphore_free_families",
        "semaphore_free_sklearn_forest",
        "policy",
    }
    if not required.issubset(binding):
        raise RuntimeError("terminal replay inference binding is partial")
    threads = binding.get("threads_per_model")
    target_count = binding.get("target_count")
    model_count = binding.get("model_count")
    families = binding.get("families")
    family_threads = binding.get("family_threads")
    semaphore_free_families = binding.get("semaphore_free_families")
    policy = binding.get("policy")
    supported_families = {
        "lightgbm",
        "xgboost",
        "catboost",
        "extratrees",
        "randomforest",
    }
    if (
        isinstance(threads, bool)
        or not isinstance(threads, int)
        or not 1 <= threads <= PRODUCTION_INFERENCE_THREADS
        or isinstance(target_count, bool)
        or target_count not in {
            len(CURRENT_REQUIRED_MODEL_TARGETS),
            len(GOAL_REQUIRED_MODEL_TARGETS),
        }
        or isinstance(model_count, bool)
        or not isinstance(model_count, int)
        or model_count < target_count
        or not isinstance(families, list)
        or not families
        or families != sorted(set(families))
        or not set(families).issubset(supported_families)
        or not isinstance(family_threads, Mapping)
        or set(family_threads) != set(families)
        or any(
            family_threads[family]
            != (1 if family in SKLEARN_FOREST_SEMAPHORE_FREE_FAMILIES else threads)
            for family in families
        )
        or semaphore_free_families
        != sorted(set(families) & SKLEARN_FOREST_SEMAPHORE_FREE_FAMILIES)
        or binding.get("semaphore_free_sklearn_forest") is not True
        or policy != FAMILY_SPECIFIC_INFERENCE_POLICY
    ):
        raise RuntimeError("terminal replay inference binding contract mismatch")
    return True, int(threads)


@dataclass
class Current7Tier1Runner:
    authenticated: Any
    code_identity: dict[str, Any]
    modules: Current7Modules
    adapter_evidence: dict[str, Any]
    model_cache: Any
    models: Mapping[str, Any]
    inference_binding: dict[str, Any]
    density_gate: Any
    problem: Any
    physical_evaluate: Any = None
    optimizer_scaling: Mapping[str, Any] | None = None
    prepared_repair_operator: Any = None
    prepared_topology_contract: Mapping[str, Any] | None = None

    @property
    def launch_eligible(self) -> bool:
        return True

    def install_offspring_repair_operator(
        self,
        *,
        topology_contract: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.prepared_repair_operator is not None:
            raise RuntimeError("offspring repair operator was already installed")
        default_topology = (
            goal_topology_contract(self.problem.fixed_primary_turns)
            if getattr(self.problem, "goal_campaign", False)
            else deep_topology_contract(self.problem.fixed_primary_turns)
        )
        topology = dict(
            topology_contract or default_topology
        )
        sealed_topology = {
            key: item for key, item in topology.items() if key != "sha256"
        }
        if (
            topology.get("sha256") != canonical_sha256(sealed_topology)
            or topology.get("fixed_primary_turns")
            != self.problem.fixed_primary_turns
        ):
            raise RuntimeError("offspring topology contract is unauthenticated")
        operator = create_pymoo_physics_repair(self.problem)
        self.prepared_repair_operator = operator
        self.prepared_topology_contract = topology
        evidence = {
            "schema_version": "mft-tier1-current7-repair-installation-v1",
            "fixed_primary_turns": self.problem.fixed_primary_turns,
            "optimizer_repair_contract_sha256": (
                self.problem.optimizer_repair_contract["contract_sha256"]
            ),
            "topology_evolution_contract_sha256": topology["sha256"],
            "offspring_repair_operator_installed": True,
            "operator_executed_before_optimization": False,
            "operator_execution_attestation_deferred_to_terminal_result": True,
            "operator_initial_evidence": operator.evidence(),
        }
        evidence["sha256"] = canonical_sha256(evidence)
        return evidence

    def repair_coordinates(
        self,
        coordinates: Any,
        *,
        stage: str,
    ) -> tuple[Any, dict[str, Any]]:
        import numpy as np

        values = np.asarray(coordinates, dtype=float)
        if values.ndim != 2 or values.shape[1] != int(self.problem.n_var):
            raise RuntimeError(f"{stage} repair coordinate schema mismatch")
        repaired = np.asarray(self.problem.repair_unit_coordinates(values), dtype=float)
        repeated = np.asarray(
            self.problem.repair_unit_coordinates(repaired), dtype=float
        )
        fixed_index = self.problem.fixed_primary_turn_coordinate_index
        fixed_unit = self.problem.fixed_primary_turn_unit_coordinate
        if (
            repaired.shape != values.shape
            or not np.array_equal(repeated, repaired)
            or not np.isclose(
                repaired[:, fixed_index], fixed_unit, rtol=0.0, atol=1e-15
            ).all()
        ):
            raise RuntimeError(f"{stage} physics repair attestation failed")
        evidence = {
            "stage": stage,
            "input_count": int(len(values)),
            "output_count": int(len(repaired)),
            "fixed_point_idempotent": True,
            "fixed_primary_turns": self.problem.fixed_primary_turns,
            "fixed_primary_coordinate_verified": True,
            "repaired_coordinate_sha256": canonical_sha256(repaired.tolist()),
        }
        evidence["sha256"] = canonical_sha256(evidence)
        return repaired, evidence

    def prepare_authenticated_warm_start(
        self,
        coordinates: Any,
        *,
        role_partition: Mapping[str, Any] | None,
        stage: str,
        topology_niche_partition: Mapping[str, Any] | None = None,
    ) -> tuple[Any, Any, dict[str, Any]]:
        """Repair and separate ordinary warm rows from basin-only donors."""

        import numpy as np

        values = np.asarray(coordinates, dtype=float)
        partition = None
        niche_partition = None
        if role_partition is not None and topology_niche_partition is not None:
            raise RuntimeError("warm start supplied two incompatible role contracts")
        if role_partition is not None:
            partition = validate_warm_role_partition(
                role_partition,
                values,
                fixed_primary_turns=self.problem.fixed_primary_turns,
            )
        if topology_niche_partition is not None:
            niche_partition = validate_topology_niche_warm_partition(
                topology_niche_partition, values
            )
        repaired, repair_evidence = self.repair_coordinates(
            values, stage=stage
        )
        if niche_partition is not None:
            unique_niche_warm, niche_filter = (
                self.problem.filter_structural_donor_coordinates(repaired)
            )
            if (
                len(repaired) != 160
                or niche_filter.get("structurally_accepted_count") != 160
                or niche_filter.get("structurally_rejected_count") != 0
                or len(unique_niche_warm)
                != niche_filter.get("decoded_unique_count")
            ):
                raise RuntimeError(
                    "topology niche warm repair/filter did not structurally accept all 160 rows"
                )
            # The ordinary basin path deduplicates decoded geometry.  The
            # topology-niche artifact instead seals an exact semantic 160-row
            # partition, so retain every structurally accepted repaired row
            # (including repeated decoded geometries) to preserve its quotas.
            # Fresh Sobol rows still provide 160 independent coordinates.
            niche_warm = repaired
            topology_contract = deep_topology_contract(
                6, enable_final1000_topology_niche=True
            )
            observed = _turn_split_main_values(
                niche_warm,
                fixed_primary_turns=6,
                coordinate_index=topology_contract["coordinate_index"],
            )
            expected = {
                int(key): int(item)
                for key, item in topology_contract[
                    "final1000_topology_niche_contract"
                ]["initialization"]["warm_topology_counts"].items()
            }
            counts = {
                topology: int(np.count_nonzero(observed == topology))
                for topology in expected
            }
            if counts != expected or counts[60] != 0:
                raise RuntimeError("topology niche warm repair changed its mix")
            empty = np.empty((0, self.problem.n_var), dtype=float)
            evidence = {
                "schema_version": "mft-tier1-authenticated-topology-niche-warm-audit-v1",
                "role_partitioned": True,
                "topology_niche_partition": niche_partition,
                "repair": repair_evidence,
                "structural_geometry_filter": niche_filter,
                "retained_count": int(len(niche_warm)),
                "decoded_unique_count": int(len(unique_niche_warm)),
                "duplicate_geometry_count": int(
                    len(niche_warm) - len(unique_niche_warm)
                ),
                "exact_semantic_partition_rows_preserved_before_dedupe": True,
                "topology_counts": {
                    str(key): item for key, item in counts.items()
                },
                "N2_main_60_warm_excluded": True,
                "prior_objectives_or_constraints_inherited": False,
                "additional_model_evaluations": 0,
                "physical_constraint_G_mutation": False,
                "physical_objective_mutation": False,
                "terminal_physical_replay_required": True,
            }
            evidence["sha256"] = canonical_sha256(evidence)
            return niche_warm, empty, evidence
        if partition is None:
            standard_values = repaired
            donor_values = np.empty((0, self.problem.n_var), dtype=float)
        else:
            standard = partition["standard_hard_feasible_candidates"]
            basin = partition["basin_structural_coordinate_donors"]
            standard_start = int(standard["start"])
            basin_start = int(basin["start"])
            standard_values = repaired[
                standard_start : standard_start + int(standard["count"])
            ]
            donor_values = repaired[basin_start : basin_start + int(basin["count"])]
        standard_warm, standard_filter = self.problem.filter_warm_start_coordinates(
            standard_values
        )
        if len(standard_warm) < 1:
            raise RuntimeError(
                "authenticated standard warm role has no hard-feasible design"
            )
        donor_filter = None
        structural_donors = np.empty((0, self.problem.n_var), dtype=float)
        topology_counts: dict[str, int] = {}
        if partition is not None:
            structural_donors, donor_filter = (
                self.problem.filter_structural_donor_coordinates(donor_values)
            )
            if len(structural_donors) < 1:
                raise RuntimeError(
                    "authenticated basin role has no structural coordinate donor"
                )
            topology_contract = (
                goal_topology_contract(self.problem.fixed_primary_turns)
                if getattr(self.problem, "goal_campaign", False)
                else deep_topology_contract(self.problem.fixed_primary_turns)
            )
            observed = _turn_split_main_values(
                structural_donors,
                fixed_primary_turns=self.problem.fixed_primary_turns,
                coordinate_index=topology_contract["coordinate_index"],
            )
            required = tuple(
                int(item)
                for item in topology_contract["turn_split_sub_islands_N2_main"]
            )
            topology_counts = {
                str(topology): int(np.count_nonzero(observed == topology))
                for topology in required
            }
            if any(topology_counts[str(topology)] < 1 for topology in required):
                raise RuntimeError(
                    "authenticated basin role omitted a required topology"
                )
        evidence = {
            "schema_version": "mft-tier1-authenticated-warm-role-audit-v1",
            "role_partitioned": partition is not None,
            "repair": repair_evidence,
            "standard_hard_geometry_filter": standard_filter,
            "standard_hard_feasible_count": int(len(standard_warm)),
            "basin_structural_donor_filter": donor_filter,
            "basin_structural_donor_count": int(len(structural_donors)),
            "basin_structural_topology_counts": topology_counts,
            "structural_donors_excluded_from_ordinary_warm_sampling": True,
            "prior_objectives_or_constraints_inherited": False,
            "additional_model_evaluations": 0,
            "physical_constraint_G_mutation": False,
            "physical_objective_mutation": False,
            "terminal_physical_replay_required": True,
        }
        evidence["sha256"] = canonical_sha256(evidence)
        return standard_warm, structural_donors, evidence

    def evaluate_coordinates(
        self,
        coordinates: Any,
        *,
        physical: bool = False,
    ) -> dict[str, Any]:
        import numpy as np

        values = np.asarray(coordinates, dtype=float)
        if values.ndim != 2 or values.shape[1] != int(self.problem.n_var):
            raise RuntimeError("smoke coordinate schema mismatch")
        if not np.isfinite(values).all():
            raise RuntimeError("smoke coordinates are not finite")
        if np.any(values < self.problem.xl) or np.any(values > self.problem.xu):
            raise RuntimeError("smoke coordinates are outside current problem bounds")
        out: dict[str, Any] = {}
        if physical and self.physical_evaluate is not None:
            self.physical_evaluate(values, out)
        else:
            self.problem._evaluate(values, out)
        objectives = np.asarray(out.get("F"), dtype=float)
        constraints = np.asarray(out.get("G"), dtype=float)
        valid = np.asarray(out.get("decoder_valid"), dtype=bool)
        if (
            objectives.shape != (len(values), 2)
            or constraints.shape != (len(values), len(self.problem.constraint_names))
            or valid.shape != (len(values),)
            or not np.isfinite(objectives).all()
            or not np.isfinite(constraints).all()
        ):
            raise RuntimeError("current7 smoke evaluation did not seal finite F/G")
        return {
            "F": objectives,
            "G": constraints,
            "frame": out["frame"],
            "decoder_valid": valid,
        }

    def terminal_physical_replay(
        self,
        coordinates: Any,
        *,
        expected_f: Any | None = None,
        expected_g: Any | None = None,
        terminal_population: Any | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        import numpy as np

        repaired, repair_evidence = self.repair_coordinates(
            coordinates, stage="terminal_physical_replay"
        )
        if not np.array_equal(repaired, np.asarray(coordinates, dtype=float)):
            raise RuntimeError("terminal population was not already repaired")

        managed_binding, optimizer_threads = _terminal_inference_binding_contract(
            self.inference_binding
        )
        optimizer_binding = None
        serial_binding = {
            "threads_per_model": DETERMINISTIC_TERMINAL_INFERENCE_THREADS,
            "policy": "implicit_serial_unmanaged_test_runner_v1",
        }
        restored_binding = None
        if managed_binding:
            optimizer_binding = bind_surrogate_inference(
                self.models,
                self.modules.run_nsga2,
                threads=optimizer_threads,
            )
            if optimizer_binding != self.inference_binding:
                raise RuntimeError("optimizer inference binding attestation drifted")
        try:
            if managed_binding:
                serial_binding = bind_surrogate_inference(
                    self.models,
                    self.modules.run_nsga2,
                    threads=DETERMINISTIC_TERMINAL_INFERENCE_THREADS,
                )
            deterministic_optimizer = self.evaluate_coordinates(
                repaired, physical=False
            )
            replay = self.evaluate_coordinates(repaired, physical=True)
            _required_target_contract(self.models)
            terminal_predictions = _terminal_model_predictions(
                self.models, replay["frame"]
            )
        finally:
            if managed_binding:
                restored_binding = bind_surrogate_inference(
                    self.models,
                    self.modules.run_nsga2,
                    threads=optimizer_threads,
                )
                if restored_binding != self.inference_binding:
                    raise RuntimeError(
                        "optimizer inference binding was not restored after terminal replay"
                    )

        objectives = np.asarray(replay["F"], dtype=float)
        constraints = np.asarray(replay["G"], dtype=float)
        deterministic_objectives = np.asarray(deterministic_optimizer["F"], dtype=float)
        deterministic_constraints = np.asarray(
            deterministic_optimizer["G"], dtype=float
        )
        optimizer_constraints = constraints
        if self.optimizer_scaling is not None:
            scales = np.asarray(self.optimizer_scaling["scale_vector"], dtype=float)
            allowances = np.asarray(
                self.optimizer_scaling["allowance_vector"], dtype=float
            )
            optimizer_constraints = (constraints - allowances) / scales

        f_match = np.array_equal(objectives, deterministic_objectives)
        g_match = np.array_equal(optimizer_constraints, deterministic_constraints)
        if not f_match or not g_match:
            raise RuntimeError(
                "deterministic serial terminal replay differs from optimizer-side replay"
            )

        expected_objectives = None
        expected_constraints = None
        if expected_f is not None:
            expected_objectives = np.asarray(expected_f, dtype=float)
            if (
                expected_objectives.shape != objectives.shape
                or not np.isfinite(expected_objectives).all()
            ):
                raise RuntimeError("terminal optimizer objective snapshot is invalid")
        if expected_g is not None:
            expected_constraints = np.asarray(expected_g, dtype=float)
            if (
                expected_constraints.shape != optimizer_constraints.shape
                or not np.isfinite(expected_constraints).all()
            ):
                raise RuntimeError("terminal optimizer G snapshot is invalid")
        snapshot_f_exact = expected_objectives is None or np.array_equal(
            expected_objectives, objectives
        )
        snapshot_g_exact = expected_constraints is None or np.array_equal(
            expected_constraints, optimizer_constraints
        )
        if optimizer_threads == DETERMINISTIC_TERMINAL_INFERENCE_THREADS and (
            not snapshot_f_exact or not snapshot_g_exact
        ):
            raise RuntimeError(
                "serial optimizer snapshot differs from deterministic terminal replay"
            )

        snapshot_f_delta = (
            None
            if expected_objectives is None
            else float(np.max(np.abs(expected_objectives - objectives), initial=0.0))
        )
        snapshot_g_delta = (
            None
            if expected_constraints is None
            else float(
                np.max(
                    np.abs(expected_constraints - optimizer_constraints),
                    initial=0.0,
                )
            )
        )
        snapshot_feasibility_match = expected_constraints is None or np.array_equal(
            expected_constraints <= 0.0,
            optimizer_constraints <= 0.0,
        )

        population_canonicalized = False
        canonical_cv_sha256 = None
        if terminal_population is not None:
            if not callable(getattr(terminal_population, "get", None)) or not callable(
                getattr(terminal_population, "set", None)
            ):
                raise RuntimeError("terminal population cannot be canonicalized")
            population_x = np.asarray(terminal_population.get("X"), dtype=float)
            if not np.array_equal(population_x, repaired):
                raise RuntimeError(
                    "terminal population coordinates differ from deterministic replay"
                )
            terminal_population.set("F", objectives)
            terminal_population.set("G", optimizer_constraints)
            terminal_population.set("CV", None)
            canonical_cv = np.asarray(terminal_population.get("CV"), dtype=float)
            if (
                not np.array_equal(
                    np.asarray(terminal_population.get("F"), dtype=float), objectives
                )
                or not np.array_equal(
                    np.asarray(terminal_population.get("G"), dtype=float),
                    optimizer_constraints,
                )
                or (
                    canonical_cv.shape != (len(repaired), 1)
                    or not np.isfinite(canonical_cv).all()
                    or not np.array_equal(
                        np.asarray(terminal_population.get("CV"), dtype=float),
                        canonical_cv,
                    )
                )
            ):
                raise RuntimeError(
                    "terminal population canonicalization attestation failed"
                )
            canonical_cv_sha256 = canonical_sha256(canonical_cv.tolist())
            population_canonicalized = True

        replay["optimizer_G"] = optimizer_constraints
        terminal_prediction_sha256 = None
        if terminal_predictions is not None:
            replay["terminal_model_predictions"] = terminal_predictions
            terminal_prediction_sha256 = canonical_sha256(
                _canonical_terminal_prediction_payload(
                    terminal_predictions,
                    population_size=len(repaired),
                )
            )
        replay["terminal_model_predictions_serially_replayed"] = (
            terminal_predictions is not None
        )
        replay["terminal_model_predictions_sha256"] = terminal_prediction_sha256
        budget_passed = []
        for index in range(len(replay["frame"])):
            if not replay["decoder_valid"][index]:
                budget_passed.append(False)
                continue
            budget_passed.append(
                winding_budget_identity(
                    replay["frame"].iloc[index],
                    expected_cw1_mm=expected_problem_cw1_mm(
                        self.problem, replay["frame"].iloc[index]
                    ),
                ).get("passed")
                is True
            )
        if not all(budget_passed):
            raise RuntimeError("terminal winding-budget identity failed")
        deterministic_replay = {
            "schema_version": "mft-tier1-deterministic-terminal-inference-v1",
            "optimizer_inference_threads": optimizer_threads,
            "terminal_replay_inference_threads": (
                DETERMINISTIC_TERMINAL_INFERENCE_THREADS
            ),
            "managed_inference_binding": managed_binding,
            "optimizer_binding": optimizer_binding,
            "serial_binding": serial_binding,
            "restored_binding": restored_binding,
            "optimizer_binding_restored": (
                not managed_binding or restored_binding == self.inference_binding
            ),
            "optimizer_side_objective_sha256": canonical_sha256(
                deterministic_objectives.tolist()
            ),
            "optimizer_side_G_sha256": canonical_sha256(
                deterministic_constraints.tolist()
            ),
            "physical_replay_objective_sha256": canonical_sha256(objectives.tolist()),
            "physical_replay_optimizer_G_sha256": canonical_sha256(
                optimizer_constraints.tolist()
            ),
            "terminal_model_predictions_sha256": terminal_prediction_sha256,
            "terminal_model_predictions_serially_replayed": (
                terminal_predictions is not None
            ),
            "objectives_bit_exact": bool(f_match),
            "optimizer_G_bit_exact": bool(g_match),
            "numeric_tolerance_relaxed": False,
            "comparison_rtol": 0.0,
            "comparison_atol": 0.0,
        }
        deterministic_replay["sha256"] = canonical_sha256(deterministic_replay)
        optimizer_snapshot = {
            "schema_version": "mft-tier1-multithread-optimizer-snapshot-v1",
            "provided_objectives": expected_objectives is not None,
            "provided_G": expected_constraints is not None,
            "objective_sha256": (
                None
                if expected_objectives is None
                else canonical_sha256(expected_objectives.tolist())
            ),
            "optimizer_G_sha256": (
                None
                if expected_constraints is None
                else canonical_sha256(expected_constraints.tolist())
            ),
            "objectives_bit_exact_to_serial_authority": bool(snapshot_f_exact),
            "optimizer_G_bit_exact_to_serial_authority": bool(snapshot_g_exact),
            "maximum_absolute_objective_delta": snapshot_f_delta,
            "maximum_absolute_optimizer_G_delta": snapshot_g_delta,
            "constraint_feasibility_classification_match": bool(
                snapshot_feasibility_match
            ),
            "numeric_comparison_used_as_terminal_gate": False,
            "canonicalized_from_serial_authority": population_canonicalized,
            "canonical_constraint_violation_sha256": canonical_cv_sha256,
            "stale_constraint_violation_cache_cleared": population_canonicalized,
            "trajectory_rank_or_crowding_used_for_persistence": False,
            "terminal_pareto_recomputed_from_canonical_F_G": (population_canonicalized),
        }
        optimizer_snapshot["sha256"] = canonical_sha256(optimizer_snapshot)
        evidence = {
            "schema_version": "mft-tier1-current7-terminal-replay-v1",
            "coordinate_count": int(len(repaired)),
            "repair": repair_evidence,
            "objective_sha256": canonical_sha256(objectives.tolist()),
            "physical_unscaled_G_sha256": canonical_sha256(constraints.tolist()),
            "optimizer_G_sha256": canonical_sha256(optimizer_constraints.tolist()),
            "optimizer_objectives_match": bool(f_match),
            "optimizer_physical_G_match": bool(g_match),
            "comparison_basis": (
                "deterministic_serial_optimizer_side_vs_serial_physical_replay"
            ),
            "deterministic_terminal_inference": deterministic_replay,
            "multithread_optimizer_snapshot": optimizer_snapshot,
            "terminal_population_canonicalized": population_canonicalized,
            "authoritative_terminal_state": "deterministic_serial_physical_replay",
            "all_winding_budget_identities_passed": True,
            "fixed_primary_turns": self.problem.fixed_primary_turns,
        }
        evidence["sha256"] = canonical_sha256(evidence)
        return replay, evidence

    def run_one(
        self,
        *,
        seed: int,
        population: int,
        max_generations: int = 600,
        warm_start_path: Path | None = None,
        warm_start_sha256: str | None = None,
        warm_start_role_partition: Mapping[str, Any] | None = None,
        warm_start_niche_partition: Mapping[str, Any] | None = None,
        optimizer_termination_strategy: str = (
            FIXED_GENERATION_TERMINATION_STRATEGY
        ),
        pre_optimization_callback: Any | None = None,
    ) -> Any:
        """Run current NSGA semantics with one repair on every path."""

        import numpy as np
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize

        if isinstance(population, bool) or int(population) < 4:
            raise ValueError("population must be an integer of at least four")
        if isinstance(max_generations, bool) or int(max_generations) < 1:
            raise ValueError("max_generations must be a positive integer")
        population = int(population)
        max_generations = int(max_generations)
        if optimizer_termination_strategy != FIXED_GENERATION_TERMINATION_STRATEGY:
            raise ValueError("corrected Tier-1 requires fixed-n-gen-no-ftol-v1")
        if pre_optimization_callback is not None and not callable(
            pre_optimization_callback
        ):
            raise TypeError("pre_optimization_callback must be callable")
        niche_active = warm_start_niche_partition is not None
        if niche_active and (
            self.problem.fixed_primary_turns != 6 or population != 320
        ):
            raise RuntimeError(
                "Final1000 topology niche requires N1=6 population320"
            )
        topology_contract = (
            deep_topology_contract(
                self.problem.fixed_primary_turns,
                enable_final1000_topology_niche=True,
            )
            if niche_active
            else (
                goal_topology_contract(self.problem.fixed_primary_turns)
                if getattr(self.problem, "goal_campaign", False)
                else deep_topology_contract(self.problem.fixed_primary_turns)
            )
        )
        if self.prepared_topology_contract is not None:
            prepared_niche = self.prepared_topology_contract.get(
                "final1000_topology_niche_contract"
            )
            if niche_active and prepared_niche is None:
                raise RuntimeError(
                    "Final1000 topology niche was not activated before optimization"
                )
            topology_contract = dict(self.prepared_topology_contract)
        coordinate_index = int(topology_contract["coordinate_index"])
        if (
            self.problem.sobol_dimension_names[coordinate_index]
            != topology_contract["coordinate_name"]
        ):
            raise RuntimeError("turn-split topology coordinate schema drifted")
        rng = np.random.default_rng(int(seed))
        if bool(warm_start_path) != bool(warm_start_sha256):
            raise RuntimeError(
                "warm-start path and expected SHA-256 must be supplied together"
            )
        if warm_start_role_partition is not None and warm_start_path is None:
            raise RuntimeError("warm role partition requires a warm-start artifact")
        if warm_start_niche_partition is not None and warm_start_path is None:
            raise RuntimeError("warm niche partition requires a warm-start artifact")
        if (
            warm_start_role_partition is not None
            and warm_start_niche_partition is not None
        ):
            raise RuntimeError("warm start cannot have two role partitions")
        warm_audit = None
        warm = None
        structural_donors = None
        if warm_start_path is not None:
            warm_values, source_authentication = load_authenticated_warm_start(
                warm_start_path,
                str(warm_start_sha256),
                n_var=self.problem.n_var,
            )
            warm, structural_donors, role_audit = (
                self.prepare_authenticated_warm_start(
                    warm_values,
                    role_partition=warm_start_role_partition,
                    stage="authenticated_warm_start",
                    topology_niche_partition=warm_start_niche_partition,
                )
            )
            warm_audit = {
                "source_authentication": source_authentication,
                "role_audit": role_audit,
                "source_sha_verified": True,
            }
            warm_audit["sha256"] = canonical_sha256(warm_audit)

        warm_selection_audit = None
        standard_selection_audit = None
        donor_selection_audit = None
        donor_count = 0
        standard_count = 0
        niche_initialization_audit = None
        if warm_start_niche_partition is not None:
            if warm is None or len(warm) != 160 or (
                structural_donors is not None and len(structural_donors)
            ):
                raise RuntimeError("topology niche warm execution shape mismatch")
            raw_initial, niche_initialization_audit = (
                build_topology_niche_initial_population(
                    self.problem,
                    warm,
                    topology_contract,
                    seed=int(seed),
                )
            )
            standard_count = 160
        elif structural_donors is not None and len(structural_donors):
            protected_count = int(
                topology_contract["initial_repaired_copies_per_sub_island"]
            ) * len(topology_contract["turn_split_sub_islands_N2_main"])
            maximum_authenticated = population // 2
            if protected_count >= maximum_authenticated:
                raise RuntimeError(
                    "population cannot mix protected donors, standard warm, "
                    "and at least one-half fresh random"
                )
            if len(structural_donors) < protected_count:
                raise RuntimeError(
                    "structural donor reservoir is smaller than its protected "
                    "topology budget"
                )
            donor_count = protected_count
            standard_count = min(len(warm), maximum_authenticated - donor_count)
            donor_selected, donor_selection_audit = select_basin_warm_start_indices(
                structural_donors,
                count=donor_count,
                contract=topology_contract,
                random_state=rng,
            )
            standard_selected, standard_selection_audit = (
                select_basin_warm_start_indices(
                    warm,
                    count=standard_count,
                    contract=topology_contract,
                    random_state=rng,
                )
            )
            if (
                len(donor_selected) != donor_count
                or len(standard_selected) != standard_count
            ):
                raise RuntimeError("authenticated warm role selection underfilled")
            raw_initial = np.vstack(
                [
                    structural_donors[donor_selected],
                    warm[standard_selected],
                    rng.random(
                        (
                            population - donor_count - standard_count,
                            self.problem.n_var,
                        )
                    ),
                ]
            )
            warm_selection_audit = donor_selection_audit
        elif warm is not None and len(warm):
            standard_count = min(len(warm), population // 2)
            selected, standard_selection_audit = select_basin_warm_start_indices(
                warm,
                count=standard_count,
                contract=topology_contract,
                random_state=rng,
            )
            raw_initial = np.vstack(
                [
                    warm[selected],
                    rng.random((population - standard_count, self.problem.n_var)),
                ]
            )
            warm_selection_audit = standard_selection_audit
        else:
            raw_initial = rng.random((population, self.problem.n_var))
        authenticated_count = donor_count + standard_count
        current_initialization_audit = {
            "schema_version": "mft-current-run-nsga2-initialization-v1",
            "policy_source": "optimization.run_nsga2.run_one",
            "population": population,
            "authenticated_warm_injected_count": int(authenticated_count),
            "authenticated_standard_hard_feasible_warm_count": int(standard_count),
            "authenticated_structural_donor_count": int(donor_count),
            "fresh_random_count": int(population - authenticated_count),
            "maximum_warm_fraction": 0.5,
            "basin_stratified_warm_selection": warm_selection_audit,
            "standard_warm_selection": standard_selection_audit,
            "structural_donor_selection": donor_selection_audit,
            "structural_donors_excluded_from_ordinary_warm_sampling": True,
            "topology_niche_initialization": niche_initialization_audit,
        }
        current_initialization_audit["sha256"] = canonical_sha256(
            current_initialization_audit
        )
        initial, base_initial_repair = self.repair_coordinates(
            raw_initial, stage="initial_population"
        )
        standard_before_topology_sha = None
        if donor_count:
            standard_before_topology_sha = canonical_sha256(
                initial[donor_count : donor_count + standard_count].tolist()
            )
        repair_operator = self.prepared_repair_operator or create_pymoo_physics_repair(
            self.problem
        )
        if niche_initialization_audit is not None:
            topology_initialization = niche_initialization_audit
        else:
            initial, topology_initialization = seed_turn_split_sub_islands(
                self.problem,
                initial,
                topology_contract,
                warm_donor_count=donor_count or standard_count,
                protected_warm_donors_only=bool(donor_count),
            )
        if donor_count and not all(
            item["source_is_authenticated_warm"]
            for item in topology_initialization["donor_sources"]
        ):
            raise RuntimeError(
                "protected topology initialization escaped structural donors"
            )
        initial, initial_audit = self.repair_coordinates(
            initial, stage="initial_population_after_topology_seeding"
        )
        role_partition_preservation = None
        if donor_count:
            standard_after_topology_sha = canonical_sha256(
                initial[donor_count : donor_count + standard_count].tolist()
            )
            if standard_after_topology_sha != standard_before_topology_sha:
                raise RuntimeError(
                    "protected topology seeding overwrote standard warm rows"
                )
            role_partition_preservation = {
                "standard_warm_rows_preserved_after_topology_seeding": True,
                "standard_warm_coordinate_sha256": (standard_after_topology_sha),
                "standard_warm_count": int(standard_count),
                "structural_donor_protected_slot_count": int(donor_count),
            }
            role_partition_preservation["sha256"] = canonical_sha256(
                role_partition_preservation
            )
        pre_optimization_evidence = {
            "schema_version": "mft-tier1-current7-pre-optimization-v1",
            "fixed_primary_turns": self.problem.fixed_primary_turns,
            "initial_population": {
                "current_run_nsga2": current_initialization_audit,
                "base_repair": base_initial_repair,
                "turn_split_sub_islands": topology_initialization,
                "repair": initial_audit,
                "role_partition_preservation": role_partition_preservation,
            },
            "authenticated_warm_start": warm_audit,
            "offspring_repair_operator": repair_operator.evidence(),
            "offspring_repair_operator_installed": True,
            "offspring_repair_operator_executed": False,
            "optimizer_execution_started": False,
        }
        pre_optimization_evidence["sha256"] = canonical_sha256(
            pre_optimization_evidence
        )
        if pre_optimization_callback is not None:
            pre_optimization_callback(pre_optimization_evidence)
        selection, mating, survival = create_deep_topology_components(
            self.problem, topology_contract, repair_operator
        )
        algorithm = NSGA2(
            pop_size=population,
            sampling=initial,
            repair=repair_operator,
            selection=selection,
            mating=mating,
            survival=survival,
            eliminate_duplicates=True,
        )
        result = minimize(
            self.problem,
            algorithm,
            ("n_gen", max_generations),
            seed=int(seed),
            verbose=False,
            save_history=False,
        )
        terminal = getattr(result, "pop", None)
        if terminal is None:
            terminal = getattr(getattr(result, "algorithm", None), "pop", None)
        if terminal is None or not callable(getattr(terminal, "get", None)):
            raise RuntimeError("optimizer returned no terminal population")
        terminal_x = np.asarray(terminal.get("X"), dtype=float)
        terminal_f = np.asarray(terminal.get("F"), dtype=float)
        terminal_g = np.asarray(terminal.get("G"), dtype=float)
        self.problem._tier1_optimizer_generation = max_generations
        replay, terminal_audit = self.terminal_physical_replay(
            terminal_x,
            expected_f=terminal_f,
            expected_g=terminal_g,
            terminal_population=terminal,
        )
        if terminal_audit.get("terminal_population_canonicalized") is not True:
            raise RuntimeError("terminal population was not canonicalized")
        executed_repair = getattr(result.algorithm, "repair", None)
        if not callable(getattr(executed_repair, "evidence", None)):
            raise RuntimeError("executed optimizer lost the physics repair operator")
        if executed_repair.call_count < 1 or executed_repair.row_count < 1:
            raise RuntimeError("pymoo never invoked the offspring repair operator")
        executed = result.algorithm
        terminal_frame = replay["frame"]
        topologies = tuple(
            int(value) for value in topology_contract["turn_split_sub_islands_N2_main"]
        )
        terminal_topology_counts = {
            str(topology): sum(
                int(_finite_number(value, "N2_main")) == topology
                for value in terminal_frame["N2_main"]
            )
            for topology in topologies
        }
        minimum_each = int(
            topology_contract["survival"]["minimum_survivors_per_turn_split_sub_island"]
        )
        expected_terminal_quota = _topology_quota_for_population(
            topology_contract, population
        )
        exact_quota_mode = bool(
            topology_contract.get("exact_survivor_quota_by_N2_main")
        )
        diversity_budget = topology_contract["bounded_diversity_budget"]
        niche_diversity_budget = topology_contract.get(
            "topology_niche_diversity_budget"
        )
        maximum_single_topology = int(
            niche_diversity_budget["maximum_single_topology_count"]
            if niche_diversity_budget
            else diversity_budget["maximum_single_protected_topology_count"]
        )
        lane_topologies = _basin_lane_topologies(topology_contract)
        terminal_lane_counts = {
            name: sum(terminal_topology_counts[str(topology)] for topology in values)
            for name, values in lane_topologies.items()
        }
        operator_audit = {
            "paired_selection_calls": int(executed.mating.selection.selection_calls),
            "paired_parent_pairs_emitted": int(
                executed.mating.selection.parent_pairs_emitted
            ),
            "topology_local_pairs_emitted": int(
                executed.mating.selection.topology_local_pairs_emitted
            ),
            "cross_36x37_pairs_emitted": int(
                executed.mating.selection.cross_36x37_pairs_emitted
            ),
            "cross_36x37_pair_fraction": float(
                executed.mating.selection.cross_36x37_pairs_emitted
                / max(1, executed.mating.selection.parent_pairs_emitted)
            ),
            "parent_pair_topology_counts": dict(
                executed.mating.selection.parent_pair_topology_counts
            ),
            "generation_parent_pair_counts": list(
                executed.mating.selection.generation_parent_pair_counts
            ),
            "cross_36x37_offspring_attributed": int(
                executed.mating.cross_36x37_offspring_attributed
            ),
            "generation_cross_offspring_counts": list(
                executed.mating.generation_cross_offspring_counts
            ),
            "migration_events": int(executed.mating.migration_events),
            "migrants_created": int(executed.mating.migrants_created),
            "survival_calls": int(executed.survival.survival_calls),
            "last_optimizer_epsilon": float(executed.survival.last_epsilon),
            "minimum_topology_count_observed": int(
                executed.survival.minimum_topology_count_observed
            ),
            "maximum_single_topology_count_observed": int(
                executed.survival.maximum_single_topology_count_observed
            ),
            "last_topology_counts": dict(
                executed.survival.last_topology_counts
            ),
            "generation_topology_counts": list(
                executed.survival.generation_topology_counts
            ),
            "generation_topology_count_seals_sha256": canonical_sha256([
                record["sha256"]
                for record in executed.survival.generation_topology_counts
            ]),
        }
        expected_last_epsilon = (
            float(topology_contract["survival"]["initial_epsilon"])
            * max(
                0.0,
                1.0
                - max(0, max_generations - 1)
                / int(topology_contract["survival"]["decay_to_zero_generation"]),
            )
            ** 2
        )
        if (
            operator_audit["paired_selection_calls"] < 1
            or operator_audit["paired_parent_pairs_emitted"] < 1
            or (
                topology_contract.get("topology_local_mating_required") is True
                and operator_audit["topology_local_pairs_emitted"]
                != operator_audit["paired_parent_pairs_emitted"]
            )
            or (
                topology_contract.get("topology_niche_mating_contract")
                and (
                    operator_audit["cross_36x37_pairs_emitted"] < 1
                    or operator_audit["cross_36x37_pair_fraction"] < 0.39
                    or operator_audit["cross_36x37_offspring_attributed"] < 1
                    or not operator_audit["generation_parent_pair_counts"]
                    or not operator_audit["generation_cross_offspring_counts"]
                    or any(
                        record.get("sha256")
                        != canonical_sha256({
                            key: value
                            for key, value in record.items()
                            if key != "sha256"
                        })
                        for record in (
                            operator_audit["generation_parent_pair_counts"]
                            + operator_audit["generation_cross_offspring_counts"]
                        )
                    )
                )
            )
            or operator_audit["migration_events"] < 1
            or operator_audit["migrants_created"] < len(topologies)
            or operator_audit["survival_calls"] < 2
            or operator_audit["minimum_topology_count_observed"] < minimum_each
            or operator_audit["maximum_single_topology_count_observed"]
            > maximum_single_topology
            or operator_audit["last_topology_counts"] != terminal_topology_counts
            or (
                exact_quota_mode
                and terminal_topology_counts
                != {
                    str(topology): expected_terminal_quota[topology]
                    for topology in topologies
                }
            )
            or (
                not exact_quota_mode
                and any(
                    terminal_topology_counts[str(topology)] < minimum_each
                    for topology in topologies
                )
            )
            or len(operator_audit["generation_topology_counts"])
            != operator_audit["survival_calls"]
            or any(
                (
                    exact_quota_mode
                    and record.get("exact_quota_verified") is not True
                )
                or record.get("sha256")
                != canonical_sha256({
                    key: value
                    for key, value in record.items()
                    if key != "sha256"
                })
                for record in operator_audit["generation_topology_counts"]
            )
            or not math.isclose(
                operator_audit["last_optimizer_epsilon"],
                expected_last_epsilon,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise RuntimeError("turn-split topology evolution audit failed")
        observed_generation_counter = int(result.algorithm.n_gen)
        if observed_generation_counter != max_generations + 1:
            raise RuntimeError(
                "fixed-generation optimizer counter differs from its sealed gate"
            )
        topology_audit = {
            "schema_version": "mft-tier1-turn-split-evolution-audit-v1",
            **operator_audit,
            "terminal_topology_counts": terminal_topology_counts,
            "terminal_basin_lane_counts": terminal_lane_counts,
            "bounded_diversity_budget": diversity_budget,
            "topology_niche_diversity_budget": niche_diversity_budget,
            "all_required_topologies_preserved": True,
            "single_topology_collapse_prevented": True,
            "exact_topology_quota_every_generation_verified": exact_quota_mode,
            "terminal_epsilon_zero": bool(expected_last_epsilon == 0.0),
            "physical_constraint_G_mutation": False,
            "physical_objective_mutation": False,
            "warm_donor_prediction_inheritance": False,
            "offspring_repair_operator": executed_repair.evidence(),
        }
        topology_audit["sha256"] = canonical_sha256(topology_audit)
        execution_audit = {
            "schema_version": "mft-tier1-current7-repair-execution-v1",
            "fixed_primary_turns": self.problem.fixed_primary_turns,
            "repair_contract": self.problem.optimizer_repair_contract,
            "initial_population": {
                "current_run_nsga2": current_initialization_audit,
                "base_repair": base_initial_repair,
                "turn_split_sub_islands": topology_initialization,
                "repair": initial_audit,
                "role_partition_preservation": role_partition_preservation,
            },
            "authenticated_warm_start": warm_audit,
            "pymoo_operator": executed_repair.evidence(),
            "terminal_physical_replay": terminal_audit,
            "stages": {
                "initial_population": True,
                "warm_start": warm_audit is not None,
                "every_offspring": True,
                "terminal_physical_replay": True,
            },
            "authoritative_terminal_G": "physical_unscaled_replay",
            "fixed_generation_counter": observed_generation_counter,
            "evaluated_generations": max_generations,
            "topology_evolution": topology_audit,
        }
        execution_audit["sha256"] = canonical_sha256(execution_audit)
        result.tier1_repair_audit = execution_audit
        result.tier1_topology_evolution_contract = topology_contract
        result.tier1_topology_evolution_audit = topology_audit
        result.tier1_evaluated_generations = max_generations
        result.tier1_completed_generations = observed_generation_counter
        result.tier1_terminal_physical_replay = replay
        return result


def create_pymoo_physics_repair(problem: Any) -> Any:
    import numpy as np
    from pymoo.core.repair import Repair

    class Current7PhysicsRepair(Repair):
        def __init__(self) -> None:
            super().__init__()
            self.call_count = 0
            self.row_count = 0
            self.last_output_sha256 = None

        def _do(self, active_problem: Any, values: Any, **kwargs: Any) -> Any:
            if active_problem is not problem:
                raise RuntimeError("pymoo repair received a different problem")
            repaired = problem.repair_unit_coordinates(values)
            array = np.asarray(repaired, dtype=float)
            self.call_count += 1
            self.row_count += int(len(array))
            self.last_output_sha256 = canonical_sha256(array.tolist())
            return repaired

        def evidence(self) -> dict[str, Any]:
            value = {
                "schema_version": "mft-tier1-current7-pymoo-repair-audit-v1",
                "call_count": self.call_count,
                "row_count": self.row_count,
                "last_output_sha256": self.last_output_sha256,
                "every_call_uses_problem_repair_unit_coordinates": True,
            }
            value["sha256"] = canonical_sha256(value)
            return value

    return Current7PhysicsRepair()


def bind_surrogate_inference(
    models: Mapping[str, Any],
    run_nsga2_module: Any,
    *,
    threads: int,
) -> dict[str, Any]:
    """Bind safe per-family threads up to the eight-CPU lane limit."""

    if (
        isinstance(threads, bool)
        or int(threads) != threads
        or not 1 <= int(threads) <= 8
    ):
        raise ValueError("inference threads must be an integer from 1 through 8")
    threads = int(threads)
    binding = run_nsga2_module._bound_surrogate_inference(models, threads=threads)
    managed, bound_threads = _terminal_inference_binding_contract(binding)
    if not managed or bound_threads != threads:
        raise RuntimeError("family-specific inference binding attestation failed")
    return binding


def attest_semlock_free_repeated_predict(
    models: Mapping[str, Any],
    frame: Any,
    *,
    repeats: int = REPEATED_PREDICT_STRESS_REPEATS,
    parallel_backends_module: Any | None = None,
) -> dict[str, Any]:
    """Fail if repeated sklearn prediction constructs ThreadPool/SemLock.

    The live ENOSPC trace was not a filesystem-TMP exhaustion: joblib's
    ``ThreadingBackend`` created a new ``multiprocessing.pool.ThreadPool`` and
    ``SimpleQueue``/``SemLock`` for repeated forest predictions.  Binding
    sklearn forests to ``n_jobs=1`` avoids that path.  This runtime attestation
    exercises every loaded target repeatedly while both primitive constructors
    are fail-fast, so Phase-B cannot amplify an unverified inference path.
    """

    import _multiprocessing
    import importlib
    from unittest import mock

    import numpy as np

    if parallel_backends_module is None:
        try:
            parallel_backends_module = importlib.import_module(
                "joblib._parallel_backends"
            )
        except ImportError:  # compact unit/smoke environments
            parallel_backends_module = importlib.import_module("multiprocessing.pool")

    required_targets = _required_target_contract(models)
    if (
        isinstance(repeats, bool)
        or not isinstance(repeats, int)
        or repeats < 2
        or len(frame) < 1
    ):
        raise RuntimeError("repeated-predict SemLock stress input is invalid")
    forest_count = 0
    for target, predictor in models.items():
        bundle = getattr(predictor, "bundle", None)
        members = bundle.get("models") if isinstance(bundle, Mapping) else None
        if not isinstance(members, list) or not members:
            raise RuntimeError(f"surrogate model inventory is unavailable: {target}")
        for family, fitted in members:
            if str(family).lower() not in SKLEARN_FOREST_SEMAPHORE_FREE_FAMILIES:
                continue
            forest_count += 1
            if (
                not hasattr(fitted, "n_jobs")
                or isinstance(fitted.n_jobs, bool)
                or int(fitted.n_jobs) != 1
            ):
                raise RuntimeError(
                    f"sklearn forest is not semaphore-free before stress: {target}"
                )
    if forest_count < len(required_targets):
        raise RuntimeError("repeated-predict stress did not cover every target forest")

    attempts = {"joblib_thread_pool": 0, "multiprocessing_semlock": 0}

    def forbidden_thread_pool(*_args: Any, **_kwargs: Any) -> Any:
        attempts["joblib_thread_pool"] += 1
        raise RuntimeError("joblib ThreadPool construction is forbidden")

    def forbidden_semlock(*_args: Any, **_kwargs: Any) -> Any:
        attempts["multiprocessing_semlock"] += 1
        raise RuntimeError("multiprocessing SemLock construction is forbidden")

    output_hashes: list[str] = []
    sample = frame.iloc[:1] if hasattr(frame, "iloc") else frame[:1]
    with (
        mock.patch.object(
            parallel_backends_module,
            "ThreadPool",
            side_effect=forbidden_thread_pool,
        ),
        mock.patch.object(_multiprocessing, "SemLock", side_effect=forbidden_semlock),
    ):
        for _repeat in range(repeats):
            for target in required_targets:
                mean, half_width = models[target].predict_mu_sigma(
                    sample, conformal=True
                )
                mean = np.asarray(mean, dtype=float).reshape(-1)
                half_width = np.asarray(half_width, dtype=float).reshape(-1)
                if (
                    len(mean) != 1
                    or len(half_width) != 1
                    or not np.isfinite(mean).all()
                    or not np.isfinite(half_width).all()
                ):
                    raise RuntimeError(
                        f"repeated-predict stress output is invalid: {target}"
                    )
                output_hashes.append(
                    canonical_sha256(
                        {
                            "target": target,
                            "mean": mean.tolist(),
                            "half_width": half_width.tolist(),
                        }
                    )
                )
    if attempts != {"joblib_thread_pool": 0, "multiprocessing_semlock": 0}:
        raise RuntimeError("parallel primitive was constructed during stress")
    per_repeat = len(required_targets)
    first = output_hashes[:per_repeat]
    if any(
        output_hashes[offset : offset + per_repeat] != first
        for offset in range(0, len(output_hashes), per_repeat)
    ):
        raise RuntimeError("repeated-predict stress was not deterministic")
    unsigned = {
        "schema_version": REPEATED_PREDICT_STRESS_SCHEMA,
        "status": "passed",
        "repeats": repeats,
        "sample_rows": 1,
        "target_count": len(required_targets),
        "predict_call_count": repeats * len(required_targets),
        "covered_sklearn_forest_count": forest_count,
        "joblib_thread_pool_construction_attempt_count": 0,
        "multiprocessing_semlock_construction_attempt_count": 0,
        "sklearn_forest_n_jobs": 1,
        "output_identity_sha256": canonical_sha256({"per_target_output_sha256": first}),
        "tmp_isolation_claimed_as_fix": False,
        "direct_enospc_cause": "joblib-threadpool-simplequeue-semlock-churn",
    }
    return {**unsigned, "sha256": canonical_sha256(unsigned)}


def phase_b_semlock_safe_release_identity(
    *, code_root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Bind Phase-B preflight to the reviewed 6ea SemLock-safe helper bytes."""

    marker = os.environ.get(PHASE_B_CHILD_PROTOCOL_ENV)
    if marker is None:
        return None
    if marker != PHASE_B_CHILD_PROTOCOL:
        raise RuntimeError("Phase-B child protocol marker is invalid")
    helper = _contained_input(
        code_root,
        code_root / SEMLOCK_SAFE_HELPER_RELATIVE,
        "SemLock-safe inference helper",
    )
    inventory_key = f"artifacts/code/{SEMLOCK_SAFE_HELPER_RELATIVE}"
    inventory_record = (manifest.get("code_inventory") or {}).get(inventory_key)
    if (
        not helper.is_file()
        or sha256_file(helper) != SEMLOCK_SAFE_HELPER_SHA256
        or not isinstance(inventory_record, Mapping)
        or inventory_record.get("sha256") != SEMLOCK_SAFE_HELPER_SHA256
        or inventory_record.get("size") != helper.stat().st_size
    ):
        raise RuntimeError("Phase-B SemLock-safe 6ea helper identity mismatch")
    value = {
        "required_release_commit": SEMLOCK_SAFE_RELEASE_COMMIT,
        "smoke_schema": SEMLOCK_SAFE_SMOKE_SCHEMA,
        "helper_relative_path": SEMLOCK_SAFE_HELPER_RELATIVE,
        "helper_sha256": SEMLOCK_SAFE_HELPER_SHA256,
        "semaphore_free_sklearn_families": ["extratrees", "randomforest"],
        "helper_code_inventory_authenticated": True,
        "tmp_isolation_claimed_as_enospc_fix": False,
    }
    return {**value, "sha256": canonical_sha256(value)}


def authenticate_goal_code_inventory(
    *,
    code_root: Path,
    code_manifest_path: Path,
    expected_manifest_payload_sha256: str,
    expected_code_inventory_sha256: str,
    expected_code_revision: str,
) -> dict[str, Any]:
    """Authenticate a checkout-free goal runtime before importing from it."""

    root = code_root.resolve(strict=True)
    manifest_path = code_manifest_path.resolve(strict=True)
    manifest = read_json(manifest_path)
    unsigned = dict(manifest)
    payload_sha256 = unsigned.pop("payload_sha256", None)
    revision = str(expected_code_revision).lower()
    expected_manifest_sha = str(expected_manifest_payload_sha256).lower()
    expected_inventory_sha = str(expected_code_inventory_sha256).lower()
    for value, length, label in (
        (revision, 40, "goal code revision"),
        (expected_manifest_sha, 64, "goal code manifest SHA-256"),
        (expected_inventory_sha, 64, "goal code inventory SHA-256"),
    ):
        if (
            len(value) != length
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RuntimeError(f"{label} is invalid")
    inventory = manifest.get("code_inventory")
    files = manifest.get("files")
    if (
        manifest.get("schema_version") != GOAL_CODE_MANIFEST_SCHEMA
        or payload_sha256 != canonical_sha256(unsigned)
        or payload_sha256 != expected_manifest_sha
        or manifest.get("code_revision") != revision
        or manifest.get("code_root_relative") != "artifacts/code"
        or not isinstance(inventory, Mapping)
        or not inventory
        or files != inventory
        or manifest.get("code_inventory_sha256")
        != canonical_sha256(inventory)
        or manifest.get("code_inventory_sha256") != expected_inventory_sha
        or manifest.get("revision_marker")
        != "artifacts/code/.source-revision"
        or manifest.get("staged_path_rule")
        != "bundle_root/<code_inventory_key>"
        or manifest.get("source_checkout_mutated") is not False
        or manifest.get("remote_git_checkout_required") is not False
        or manifest.get("scheduler_project_code_included") is not False
    ):
        raise RuntimeError("goal code manifest identity mismatch")

    verified: dict[str, Any] = {}
    expected_runtime_files: set[str] = set()
    for relative, record in sorted(inventory.items()):
        pure = PurePosixPath(str(relative))
        if (
            pure.is_absolute()
            or len(pure.parts) < 3
            or pure.parts[:2] != ("artifacts", "code")
            or ".." in pure.parts
        ):
            raise RuntimeError(f"goal code inventory path mismatch: {relative}")
        runtime_relative = PurePosixPath(*pure.parts[2:]).as_posix()
        path = root.joinpath(*PurePosixPath(runtime_relative).parts).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                f"goal code inventory file escaped root: {relative}"
            ) from exc
        if not path.is_file():
            raise RuntimeError(f"goal code inventory file is missing: {relative}")
        size = path.stat().st_size
        digest = sha256_file(path)
        if (
            not isinstance(record, Mapping)
            or set(record) != {"sha256", "size"}
            or record.get("size") != size
            or record.get("sha256") != digest
        ):
            raise RuntimeError(f"goal code file authentication failed: {relative}")
        expected_runtime_files.add(runtime_relative)
        verified[str(relative)] = {"sha256": digest, "size": size}

    actual_runtime_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
    }
    if actual_runtime_files != expected_runtime_files:
        raise RuntimeError("goal staged code file inventory is not exact")
    marker = root / ".source-revision"
    if marker.read_bytes() != f"{revision}\n".encode("ascii"):
        raise RuntimeError("goal staged code revision marker mismatch")

    evidence = {
        "path": str(root),
        "revision": revision,
        "clean": True,
        "authentication_mode": "sealed_goal_code_inventory",
        "code_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "payload_sha256": payload_sha256,
        },
        "code_inventory_sha256": canonical_sha256(inventory),
        "verified_inventory_sha256": canonical_sha256(verified),
        "verified_file_count": len(verified),
        "revision_marker_authenticated": True,
        "git_checkout_required": False,
    }
    evidence["sha256"] = canonical_sha256(evidence)
    return evidence


def build_authenticated_runner(
    *,
    generation: Path,
    candidate_path: Path,
    quality_path: Path,
    code_root: Path,
    expected_code_revision: str,
    fixed_primary_turns: int,
    stage_spec: Mapping[str, Any] | None = None,
    inference_threads: int = 1,
    dataset_path_override: Path | None = None,
    profile_path_override: Path | None = None,
    expected_documentary_generation_path: str | None = None,
    code_manifest_path: Path | None = None,
    expected_code_manifest_payload_sha256: str | None = None,
    expected_code_inventory_sha256: str | None = None,
) -> Current7Tier1Runner:
    """Authenticate, load exactly once, and construct the smoke-only runner."""

    if (
        isinstance(inference_threads, bool)
        or int(inference_threads) != inference_threads
    ):
        raise ValueError("inference_threads must be an integer")
    inference_threads = int(inference_threads)
    if not 1 <= inference_threads <= 8:
        raise ValueError("inference_threads must be from 1 through 8")

    normalized_stage_spec = validate_stage_spec(stage_spec or CURRENT_STAGE_SPEC)
    goal_campaign = is_goal_stage_spec(normalized_stage_spec)
    expected_required_targets = (
        GOAL_REQUIRED_MODEL_TARGETS
        if goal_campaign
        else CURRENT_REQUIRED_MODEL_TARGETS
    )
    authenticated = authenticate_corrected_generation(
        generation=generation,
        candidate_path=candidate_path,
        quality_path=quality_path,
        goal_campaign=goal_campaign,
        dataset_path_override=dataset_path_override,
        profile_path_override=profile_path_override,
        expected_documentary_generation_path=(
            expected_documentary_generation_path
        ),
    )
    code_inventory_requested = any(
        value is not None
        for value in (
            code_manifest_path,
            expected_code_manifest_payload_sha256,
            expected_code_inventory_sha256,
        )
    )
    if code_inventory_requested:
        if (
            code_manifest_path is None
            or expected_code_manifest_payload_sha256 is None
            or expected_code_inventory_sha256 is None
        ):
            raise RuntimeError(
                "goal staged code authentication requires manifest and identities"
            )
        if not goal_campaign:
            raise RuntimeError(
                "checkout-free code inventory is restricted to the goal campaign"
            )
        code_identity = authenticate_goal_code_inventory(
            code_root=code_root,
            code_manifest_path=code_manifest_path,
            expected_manifest_payload_sha256=(
                expected_code_manifest_payload_sha256
            ),
            expected_code_inventory_sha256=expected_code_inventory_sha256,
            expected_code_revision=expected_code_revision,
        )
    else:
        code_identity = authenticate_code_root(code_root, expected_code_revision)
    modules = load_current7_modules(Path(code_identity["path"]))
    if goal_campaign:
        manifest = {
            **authenticated.evidence,
            "code": dict(code_identity),
            "temperature_contract": copy.deepcopy(GOAL_TEMPERATURE_CONTRACT),
            "temperature_contract_sha256": GOAL_TEMPERATURE_CONTRACT_SHA256,
            "hard_constraint_contract": stage_hard_constraint_contract(
                normalized_stage_spec
            ),
            "hard_constraint_contract_sha256": canonical_sha256(
                stage_hard_constraint_contract(normalized_stage_spec)
            ),
            "stage_spec": copy.deepcopy(normalized_stage_spec),
            "stage_spec_sha256": canonical_sha256(normalized_stage_spec),
            "model_loading": {
                "process_scope": "single_local_process",
                "cache_policy": "one_generation_authentication_and_unpickle_pass",
                "generation_copy_performed": False,
                "legacy_feedback_wrapper_used": False,
            },
            "scheduler_write_performed": False,
            "slurm_submission_performed": False,
            "canonical_pointer_write_performed": False,
        }
    else:
        manifest = adapter_manifest(authenticated, code_identity=code_identity)
        validate_adapter_manifest(manifest)

    cache = process_model_cache(
        authenticated,
        train_models_module=modules.train_models,
        predictor_class=modules.predictor.EnsemblePredictor,
    )
    models = cache.load()
    if tuple(models) != expected_required_targets or not cache.loaded_once:
        raise RuntimeError("corrected current7 models were not cached exactly once")
    inference_binding = bind_surrogate_inference(
        models, modules.run_nsga2, threads=inference_threads
    )
    if (
        inference_binding.get("target_count") != len(expected_required_targets)
        or inference_binding.get("threads_per_model") != inference_threads
    ):
        raise RuntimeError("corrected current7 inference binding mismatch")
    density_gate = modules.run_nsga2.build_density_gate(
        str(authenticated.dataset_path),
        list(models["Llt_phys"].features),
    )
    problem_class = create_current7_problem_class(
        base_problem_class=modules.nsga2_problem.MFTProblem,
        base_constraint_names=modules.nsga2_problem.CONSTRAINT_NAMES,
        base_fixed_stack_mm=modules.nsga2_problem.NSGA_FIXED_THERMAL_STACK_MM,
        sobol_dims=modules.input_parameter._SOBOL_DIMS,
        bounding_box_lit=modules.geometry_metrics.bounding_box_lit,
        input_parameter_module=modules.input_parameter,
        design_analytical_b_field_t=(modules.nsga2_problem.design_analytical_b_field_t),
    )
    problem = problem_class(
        models,
        spec=normalized_stage_spec,
        density_gate=density_gate,
        fixed_primary_turns=fixed_primary_turns,
    )
    if (
        tuple(problem.constraint_names) != stage_constraint_names(problem.stage_spec)
        or int(problem.n_ieq_constr) != len(problem.constraint_names)
        or problem.offspring_physics_repair is not True
        or problem.launch_eligible is not True
    ):
        raise RuntimeError("corrected current7 problem contract mismatch")
    return Current7Tier1Runner(
        authenticated=authenticated,
        code_identity=code_identity,
        modules=modules,
        adapter_evidence=manifest,
        model_cache=cache,
        models=models,
        inference_binding=inference_binding,
        density_gate=density_gate,
        problem=problem,
    )


def runner_for_fixed_primary_turns(
    runner: Current7Tier1Runner,
    fixed_primary_turns: int,
) -> Current7Tier1Runner:
    """Reuse one authenticated model cache for the other fixed-N1 stratum."""

    allowed_turns = (
        GOAL_SUPPORTED_PRIMARY_TURNS
        if getattr(runner.problem, "goal_campaign", False)
        else SUPPORTED_FIXED_PRIMARY_TURNS
    )
    if fixed_primary_turns not in allowed_turns:
        raise ValueError("fixed_primary_turns is outside the campaign strata")
    problem = type(runner.problem)(
        runner.models,
        spec=runner.problem.stage_spec,
        density_gate=runner.density_gate,
        fixed_primary_turns=int(fixed_primary_turns),
    )
    if (
        problem.fixed_primary_turns != int(fixed_primary_turns)
        or problem.launch_eligible is not True
        or problem.offspring_physics_repair is not True
    ):
        raise RuntimeError("dual-stratum corrected problem construction failed")
    return Current7Tier1Runner(
        authenticated=runner.authenticated,
        code_identity=runner.code_identity,
        modules=runner.modules,
        adapter_evidence=runner.adapter_evidence,
        model_cache=runner.model_cache,
        models=runner.models,
        inference_binding=runner.inference_binding,
        density_gate=runner.density_gate,
        problem=problem,
    )


def _contained_input(root: Path, path: Path, label: str) -> Path:
    root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"{label} escaped the immutable bundle") from exc
    return resolved


def _authenticate_relocated_code_inventory(
    root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate the small code inventory without rehashing model bundles."""

    inventory = manifest.get("code_inventory") or {}
    if (
        not isinstance(inventory, Mapping)
        or not inventory
        or manifest.get("code_inventory_sha256") != canonical_sha256(inventory)
    ):
        raise RuntimeError("bundle code inventory identity mismatch")
    files = manifest.get("files") or {}
    verified: dict[str, Any] = {}
    for relative, record in sorted(inventory.items()):
        relative_path = Path(str(relative))
        if (
            relative_path.is_absolute()
            or not relative_path.parts
            or relative_path.parts[:2] != ("artifacts", "code")
            or ".." in relative_path.parts
            or files.get(relative) != record
        ):
            raise RuntimeError(f"bundle code inventory path mismatch: {relative}")
        path = _contained_input(root, root / relative_path, "bundle code file")
        if not path.is_file():
            raise RuntimeError(f"bundle code inventory file is missing: {relative}")
        size = path.stat().st_size
        digest = sha256_file(path)
        if (
            not isinstance(record, Mapping)
            or record.get("size") != size
            or record.get("sha256") != digest
        ):
            raise RuntimeError(f"bundle code file authentication failed: {relative}")
        verified[str(relative)] = {"size": size, "sha256": digest}
    evidence = {
        "file_count": len(verified),
        "code_inventory_sha256": canonical_sha256(inventory),
        "verified_inventory_sha256": canonical_sha256(verified),
        "full_model_artifact_rehash_performed": False,
    }
    evidence["sha256"] = canonical_sha256(evidence)
    return evidence


def authenticate_relocated_bundle_generation(
    *,
    bundle_root: Path,
    relocation_path: Path,
    adapter_receipt_path: Path,
    registry: Path,
    generation: Path,
    dataset: Path,
    profile: Path,
) -> tuple[
    AuthenticatedCorrectedGeneration,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Authenticate immutable bundle-relative bytes without rewriting evidence."""

    root = bundle_root.resolve(strict=True)
    manifest_path = _contained_input(
        root, root / "bundle_manifest.json", "bundle manifest"
    )
    manifest = read_json(manifest_path)
    stable = dict(manifest)
    bundle_id = stable.pop("bundle_id", None)
    contract_sha = stable.pop("contract_sha256", None)
    if (
        not isinstance(bundle_id, str)
        or contract_sha != canonical_sha256(stable)
        or bundle_id != f"current7-{contract_sha[:20]}"
    ):
        raise RuntimeError("bundle manifest canonical identity mismatch")
    code_authentication = _authenticate_relocated_code_inventory(root, manifest)
    relocation_path = _contained_input(root, relocation_path, "relocation map")
    receipt_path = _contained_input(root, adapter_receipt_path, "adapter receipt")
    registry = _contained_input(root, registry, "relocated registry")
    generation = _contained_input(root, generation, "relocated generation")
    dataset = _contained_input(root, dataset, "relocated dataset")
    profile = _contained_input(root, profile, "relocated profile")
    relocation = read_json(relocation_path)
    receipt = read_json(receipt_path)
    validate_smoke_receipt(receipt, relocated_source_evidence=True)
    paths = relocation.get("bundle_paths") or {}

    def expected_path(name: str) -> Path:
        relative = Path(str(paths.get(name) or ""))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise RuntimeError(f"relocation path is unsafe: {name}")
        return _contained_input(root, root / relative, f"relocated {name}")

    if (
        relocation.get("schema_version") != "mft-tier1-current7-relocation-v1"
        or relocation.get("source_absolute_paths_are_documentary_only") is not True
        or relocation.get("local_adapter_authentication_replayed_remotely") is not False
        or relocation.get("generation_report_bytes_mutated") is not False
        or relocation.get("remote_git_checkout_required") is not False
    ):
        raise RuntimeError("corrected-generation relocation contract mismatch")
    expected_arguments = {
        "adapter_receipt": receipt_path,
        "registry": registry,
        "generation": generation,
        "dataset": dataset,
        "profile": profile,
    }
    if any(expected_path(name) != path for name, path in expected_arguments.items()):
        raise RuntimeError("corrected-generation relocation argument mismatch")
    if (
        manifest.get("relocation", {}).get("contract_sha256")
        != canonical_sha256(relocation)
        or manifest.get("relocation", {}).get("sha256") != sha256_file(relocation_path)
        or manifest.get("adapter_receipt", {}).get("file_sha256")
        != sha256_file(receipt_path)
    ):
        raise RuntimeError("bundle relocation/receipt fingerprints mismatch")
    adapter_evidence = receipt["adapter_manifest"]
    relocated = relocation.get("relocated_identity") or {}
    report_path = expected_path("train_report")
    candidate_path = expected_path("candidate")
    quality_path = expected_path("quality_status")
    report = read_json(report_path)
    candidate = read_json(candidate_path)
    quality = read_json(quality_path)
    generation_inventory = relocated.get("generation_artifacts") or {}
    if (
        generation != report_path.parent
        or generation.parent.parent != registry
        or report.get("training_run_id") != generation.name
        or report.get("training_run_id") != adapter_evidence.get("training_run_id")
        or sha256_file(report_path)
        != (adapter_evidence.get("train_report") or {}).get("sha256")
        or sha256_file(candidate_path)
        != (adapter_evidence.get("candidate") or {}).get("sha256")
        or sha256_file(quality_path)
        != (adapter_evidence.get("quality_status") or {}).get("sha256")
        or sha256_file(dataset) != (adapter_evidence.get("dataset") or {}).get("sha256")
        or training_profile_sha256(read_json(profile))
        != (adapter_evidence.get("profile") or {}).get("canonical_sha256")
        or set(generation_inventory) != set(report.get("artifacts") or {})
        or canonical_sha256(generation_inventory)
        != relocated.get("generation_artifact_inventory_sha256")
        or relocated.get("generation_artifact_inventory_sha256")
        != manifest.get("generation_artifact_inventory_sha256")
    ):
        raise RuntimeError("relocated corrected-generation identity mismatch")
    for relative, expected_sha in report["artifacts"].items():
        record = generation_inventory.get(relative) or {}
        if record.get("sha256") != expected_sha:
            raise RuntimeError(f"relocated generation inventory mismatch: {relative}")
    evidence = dict(adapter_evidence)
    evidence.update(
        {
            "generation": str(generation),
            "registry": str(registry),
            "generation_relative": generation.relative_to(registry).as_posix(),
            "train_report": {
                "path": str(report_path),
                "sha256": sha256_file(report_path),
            },
            "candidate": {
                "path": str(candidate_path),
                "sha256": sha256_file(candidate_path),
            },
            "quality_status": {
                **dict(adapter_evidence["quality_status"]),
                "path": str(quality_path),
            },
            "dataset": {
                **dict(adapter_evidence["dataset"]),
                "path": str(dataset),
            },
            "profile": {
                **dict(adapter_evidence["profile"]),
                "path": str(profile),
            },
            "relocation_contract_sha256": canonical_sha256(relocation),
            "generation_artifact_inventory_sha256": canonical_sha256(
                generation_inventory
            ),
            "relocated_code_authentication": code_authentication,
        }
    )
    authenticated = AuthenticatedCorrectedGeneration(
        generation=generation,
        registry=registry,
        report_path=report_path,
        report=report,
        candidate_path=candidate_path,
        candidate=candidate,
        quality_path=quality_path,
        quality=quality,
        dataset_path=dataset,
        profile_path=profile,
        evidence=evidence,
    )
    return authenticated, receipt, relocation, manifest


def build_relocated_authenticated_runner(
    *,
    authenticated: AuthenticatedCorrectedGeneration,
    receipt: Mapping[str, Any],
    code_root: Path,
    fixed_primary_turns: int,
    inference_threads: int,
    stage_spec: Mapping[str, Any],
) -> Current7Tier1Runner:
    """Load one relocated generation once and construct one fixed-N1 runner."""

    modules = load_current7_modules(code_root)
    normalized_stage_spec = validate_stage_spec(stage_spec)
    goal_campaign = is_goal_stage_spec(normalized_stage_spec)
    expected_required_targets = (
        GOAL_REQUIRED_MODEL_TARGETS
        if goal_campaign
        else CURRENT_REQUIRED_MODEL_TARGETS
    )
    adapter_evidence = dict(receipt.get("adapter_manifest") or {})
    if goal_campaign:
        if (
            adapter_evidence.get("schema_version")
            != "mft-goal-20260726-g0-generation-adapter-v1"
            or adapter_evidence.get("generation_targets")
            != list(GOAL_G0_MODEL_TARGETS)
            or adapter_evidence.get("required_model_targets")
            != list(expected_required_targets)
        ):
            raise RuntimeError("relocated goal G0 adapter manifest mismatch")
    else:
        adapter_evidence = validate_adapter_manifest(
            adapter_evidence, source_paths_required=False
        )
    cache = process_model_cache(
        authenticated,
        train_models_module=modules.train_models,
        predictor_class=modules.predictor.EnsemblePredictor,
    )
    models = cache.load()
    if tuple(models) != expected_required_targets or not cache.loaded_once:
        raise RuntimeError("relocated models were not authenticated exactly once")
    inference_binding = bind_surrogate_inference(
        models, modules.run_nsga2, threads=inference_threads
    )
    density_gate = modules.run_nsga2.build_density_gate(
        str(authenticated.dataset_path), list(models["Llt_phys"].features)
    )
    problem_class = create_current7_problem_class(
        base_problem_class=modules.nsga2_problem.MFTProblem,
        base_constraint_names=modules.nsga2_problem.CONSTRAINT_NAMES,
        base_fixed_stack_mm=modules.nsga2_problem.NSGA_FIXED_THERMAL_STACK_MM,
        sobol_dims=modules.input_parameter._SOBOL_DIMS,
        bounding_box_lit=modules.geometry_metrics.bounding_box_lit,
        input_parameter_module=modules.input_parameter,
        design_analytical_b_field_t=(modules.nsga2_problem.design_analytical_b_field_t),
    )
    problem = problem_class(
        models,
        spec=normalized_stage_spec,
        density_gate=density_gate,
        fixed_primary_turns=fixed_primary_turns,
    )
    code_identity = dict(adapter_evidence["code"])
    code_identity["runtime_relocated_path"] = str(code_root.resolve(strict=True))
    return Current7Tier1Runner(
        authenticated=authenticated,
        code_identity=code_identity,
        modules=modules,
        adapter_evidence=dict(adapter_evidence),
        model_cache=cache,
        models=models,
        inference_binding=inference_binding,
        density_gate=density_gate,
        problem=problem,
    )


def _atomic_json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(
            value,
            indent=1,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    descriptor, staged_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged_name, path)
    finally:
        if os.path.exists(staged_name):
            os.remove(staged_name)


def _atomic_npy(path: Path, values: Any) -> None:
    import numpy as np

    descriptor, staged_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.save(stream, np.asarray(values), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged_name, path)
    finally:
        if os.path.exists(staged_name):
            os.remove(staged_name)


def _atomic_csv(path: Path, frame: Any) -> None:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    descriptor, staged_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged_name, path)
    finally:
        if os.path.exists(staged_name):
            os.remove(staged_name)


def load_authenticated_warm_start(
    path: Path,
    expected_sha256: str,
    *,
    n_var: int,
) -> tuple[Any, dict[str, Any]]:
    """Load coordinate-only warm bytes after exact caller-pinned authentication."""

    import numpy as np

    warm_path = path.resolve(strict=True)
    if not warm_path.is_file():
        raise RuntimeError("warm-start path is not a regular file")
    expected = str(expected_sha256 or "").strip().lower()
    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise RuntimeError("warm-start expected SHA-256 is invalid")
    try:
        payload = warm_path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"warm-start bytes cannot be read: {exc}") from exc
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise RuntimeError("warm-start SHA-256 mismatch")
    try:
        values = np.asarray(
            np.load(io.BytesIO(payload), allow_pickle=False),
            dtype=float,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError(f"warm-start array cannot be loaded: {exc}") from exc
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != int(n_var):
        raise RuntimeError(
            f"warm-start array has shape {values.shape}; expected (N, {int(n_var)})"
        )
    if not np.isfinite(values).all():
        raise RuntimeError("warm-start coordinates are not finite")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise RuntimeError("warm-start coordinates escaped the unit hypercube")
    evidence = {
        "schema_version": "mft-tier1-authenticated-warm-start-v1",
        "path": str(warm_path),
        "sha256": actual,
        "byte_count": len(payload),
        "shape": [int(values.shape[0]), int(values.shape[1])],
        "coordinate_unit_sha256": canonical_sha256(values.tolist()),
        "allow_pickle": False,
        "single_read_hash_and_load": True,
        "coordinates_only": True,
        "prior_objectives_reused": False,
        "prior_constraints_reused": False,
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    return values, evidence


def authenticate_warm_handoff(
    warm_start: Path,
    warm_contract: Path,
    *,
    fixed_primary_turns: int,
    n_var: int,
    expected_contract_file_sha256: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Authenticate a relocated, self-sealed fixed-turn coordinate handoff."""

    contract_path = warm_contract.resolve(strict=True)
    contract = read_json(contract_path)
    unsigned = dict(contract)
    recorded_sha = unsigned.pop("sha256", None)
    contract_file_sha = sha256_file(contract_path)
    warm_record = contract.get("warm_start") or {}
    expected_warm_sha = str(warm_record.get("sha256") or "").lower()
    values, artifact = load_authenticated_warm_start(
        warm_start,
        expected_warm_sha,
        n_var=n_var,
    )
    role_partition = contract.get("warm_role_partition")
    validated_role_partition = (
        None
        if role_partition is None
        else validate_warm_role_partition(
            role_partition,
            values,
            fixed_primary_turns=int(fixed_primary_turns),
        )
    )
    niche_partition = contract.get("topology_niche_partition")
    validated_niche_partition = (
        None
        if niche_partition is None
        else validate_topology_niche_warm_partition(niche_partition, values)
    )
    if validated_niche_partition is not None and validated_role_partition is not None:
        raise RuntimeError("warm handoff cannot mix legacy and topology niche roles")
    if validated_role_partition is not None:
        standard_role = validated_role_partition["standard_hard_feasible_candidates"]
        basin_role = validated_role_partition["basin_structural_coordinate_donors"]
        basin_selection = contract.get("basin_aware_selection") or {}
        standard_source = contract.get("standard_warm_source") or {}
        unsigned_standard_source = dict(standard_source)
        standard_source_sha = unsigned_standard_source.pop("sha256", None)
        if (
            standard_source_sha != canonical_sha256(unsigned_standard_source)
            or standard_role.get("source_artifact_sha256")
            != (standard_source.get("artifact") or {}).get("sha256")
            or standard_role.get("source_contract_file_sha256")
            != (standard_source.get("contract") or {}).get("file_sha256")
            or standard_role.get("source_hard_geometry_joint_count")
            != standard_source.get("hard_geometry_joint_count")
            or standard_source.get("remote_current_problem_hard_filter_required")
            is not True
            or basin_role.get("source_selection_contract_sha256")
            != basin_selection.get("sha256")
            or basin_selection.get("coordinate_donors_only") is not True
        ):
            raise RuntimeError("warm basin role/source selection identity mismatch")
    if (
        (
            expected_contract_file_sha256 is not None
            and contract_file_sha != str(expected_contract_file_sha256).lower()
        )
        or (recorded_sha is not None and recorded_sha != canonical_sha256(unsigned))
        or contract.get("fixed_primary_turns") != int(fixed_primary_turns)
        or warm_record.get("shape") != list(values.shape)
        or not (
            contract.get("warm_rows_are_coordinate_donors_only") is True
            or contract.get("fixed_primary_turns_scope")
            == "runner_repair_after_authenticated_inverse_coordinate_handoff"
            or (
                int(fixed_primary_turns) == 6
                and warm_record.get("coordinate_contract")
                == ("authenticated_n1_6_decoded_to_unit_then_current_repair_v1")
            )
        )
        or contract.get(
            "physical_hard_spec_mutation",
            contract.get("hard_constraint_mutation", False),
        )
        is not False
        or contract.get("objective_mutation", False) is not False
        or contract.get("automatic_promotion_allowed") is not False
    ):
        raise RuntimeError("fixed-turn warm handoff contract mismatch")
    evidence = {
        "schema_version": "mft-tier1-current7-warm-handoff-authentication-v1",
        "fixed_primary_turns": int(fixed_primary_turns),
        "artifact": artifact,
        "contract_path": str(contract_path),
        "contract_file_sha256": contract_file_sha,
        "contract_canonical_sha256": recorded_sha,
        "contract_schema_version": contract.get("schema_version"),
        "source_sha_verified_before_repair": True,
        "coordinates_only": True,
        "coordinate_contract": warm_record.get("coordinate_contract"),
        "warm_role_partition": validated_role_partition,
        "topology_niche_partition": validated_niche_partition,
        "prior_prediction_or_pass_classification_reused": False,
    }
    evidence["sha256"] = canonical_sha256(evidence)
    return values, evidence


def load_smoke_coordinate(
    path: Path,
    expected_sha256: str,
    *,
    n_var: int,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    coordinate_path = path.resolve(strict=True)
    expected = str(expected_sha256 or "").strip().lower()
    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise RuntimeError("smoke coordinate expected SHA-256 is invalid")
    actual = sha256_file(coordinate_path)
    if actual != expected:
        raise RuntimeError("smoke coordinate SHA-256 mismatch")
    values = np.asarray(
        np.load(coordinate_path, allow_pickle=False),
        dtype=float,
    )
    if values.shape != (1, int(n_var)) or not np.isfinite(values).all():
        raise RuntimeError("smoke coordinate must contain exactly one finite row")
    return values, {
        "path": str(coordinate_path),
        "sha256": actual,
        "shape": [1, int(n_var)],
        "coordinate_unit_sha256": canonical_sha256(values[0].tolist()),
    }


def _required_target_contract(models_or_predictions: Mapping[str, Any]) -> tuple[str, ...]:
    observed = tuple(models_or_predictions)
    for allowed in (
        CURRENT_REQUIRED_MODEL_TARGETS,
        GOAL_REQUIRED_MODEL_TARGETS,
    ):
        if observed == allowed:
            return allowed
    raise RuntimeError("model target order/set mismatch")


def _smoke_every_model(
    models: Mapping[str, Any],
    frame: Any,
) -> dict[str, Any]:
    import numpy as np

    required_targets = _required_target_contract(models)
    evidence = {}
    for target in required_targets:
        try:
            mean, half_width = models[target].predict_mu_sigma(
                frame,
                conformal=True,
            )
        except TypeError as exc:
            raise RuntimeError(
                f"smoke model does not accept explicit conformal=True: {target}"
            ) from exc
        mean = np.asarray(mean, dtype=float).reshape(-1)
        half_width = np.asarray(half_width, dtype=float).reshape(-1)
        if (
            mean.shape != (len(frame),)
            or half_width.shape != (len(frame),)
            or not np.isfinite(mean).all()
            or not np.isfinite(half_width).all()
            or np.any(half_width < 0.0)
        ):
            raise RuntimeError(f"smoke model inference failed: {target}")
        evidence[target] = {
            "mean": float(mean[0]),
            "q90_conformal_half_width": float(half_width[0]),
            "finite": True,
        }
    return {
        "target_count": len(evidence),
        "targets": evidence,
        "all_required_targets_exercised": True,
        "explicit_conformal_argument": True,
        "additional_half_width_multiplier": 1.0,
    }


def _dev_shm_semaphore_snapshot() -> dict[str, Any]:
    """Observe Linux POSIX semaphore files without mutating shared memory."""

    root = Path("/dev/shm")
    if os.name != "posix" or not root.is_dir():
        return {
            "available": False,
            "semaphore_names": [],
            "semaphore_count": 0,
            "free_bytes": None,
            "free_inodes": None,
        }
    names = sorted(
        item.name for item in root.iterdir() if item.name.startswith("sem.")
    )
    stat = os.statvfs(root)
    return {
        "available": True,
        "semaphore_names": names,
        "semaphore_count": len(names),
        "free_bytes": int(stat.f_bavail * stat.f_frsize),
        "free_inodes": int(stat.f_favail),
    }


def semlock_safe_prediction_stress(
    models: Mapping[str, Any],
    frame: Any,
    inference_binding: Mapping[str, Any],
    *,
    rounds: int = 8,
) -> dict[str, Any]:
    """Repeat every prediction and fail closed on SemLock/ENOSPC evidence."""

    import numpy as np

    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 2:
        raise ValueError("SemLock stress rounds must be an integer >= 2")
    managed, _threads = _terminal_inference_binding_contract(inference_binding)
    family_threads = dict(inference_binding.get("family_threads") or {})
    if (
        not managed
        or inference_binding.get("semaphore_free_sklearn_forest") is not True
        or "extratrees" not in family_threads
        or family_threads.get("extratrees") != 1
    ):
        raise RuntimeError("SemLock stress requires serial sklearn forests")
    required_targets = _required_target_contract(models)
    before = _dev_shm_semaphore_snapshot()
    prediction_digests = []
    calls = 0
    enospc_observed = False
    try:
        for _round in range(rounds):
            for target in required_targets:
                mean, half_width = models[target].predict_mu_sigma(
                    frame, conformal=True
                )
                mean = np.asarray(mean, dtype=float).reshape(-1)
                half_width = np.asarray(half_width, dtype=float).reshape(-1)
                if (
                    mean.shape != (len(frame),)
                    or half_width.shape != (len(frame),)
                    or not np.isfinite(mean).all()
                    or not np.isfinite(half_width).all()
                ):
                    raise RuntimeError("SemLock stress prediction became non-finite")
                prediction_digests.append(canonical_sha256({
                    "target": target,
                    "mean": mean.tolist(),
                    "half_width": half_width.tolist(),
                }))
                calls += 1
    except OSError as exc:
        enospc_observed = exc.errno == errno.ENOSPC
        if enospc_observed:
            raise RuntimeError(
                "SemLock-safe prediction stress hit ENOSPC"
            ) from exc
        raise
    after = _dev_shm_semaphore_snapshot()
    new_names = sorted(
        set(after["semaphore_names"]) - set(before["semaphore_names"])
    )
    if before["available"] and new_names:
        raise RuntimeError(
            "prediction stress left new /dev/shm semaphore entries"
        )
    value = {
        "schema_version": "mft-tier1-semlock-safe-prediction-stress-v1",
        "rounds": rounds,
        "target_count": len(required_targets),
        "prediction_call_count": calls,
        "prediction_digest_sha256": canonical_sha256(prediction_digests),
        "inference_policy": inference_binding.get("policy"),
        "family_threads": family_threads,
        "sklearn_extratrees_n_jobs": family_threads["extratrees"],
        "before": before,
        "after": after,
        "new_semaphore_names": new_names,
        "semaphore_entry_growth_count": len(new_names),
        "enospc_observed": enospc_observed,
        "stress_passed": True,
    }
    value["sha256"] = canonical_sha256(value)
    return value


def _jsonable_decoded_parameters(row: Any) -> dict[str, Any]:
    """Preserve the complete decoded row without emitting JSON NaN values."""

    import numpy as np

    source = row.to_dict() if callable(getattr(row, "to_dict", None)) else dict(row)
    decoded: dict[str, Any] = {}
    for name, value in source.items():
        key = str(name)
        if value is None:
            decoded[key] = None
        elif isinstance(value, (bool, np.bool_)):
            decoded[key] = bool(value)
        elif isinstance(value, (int, np.integer)):
            decoded[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            number = float(value)
            decoded[key] = number if math.isfinite(number) else None
        elif isinstance(value, str):
            decoded[key] = value
        else:
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError):
                decoded[key] = str(value)
            else:
                decoded[key] = number if math.isfinite(number) else None
    return decoded


def _terminal_model_predictions(
    models: Mapping[str, Any],
    frame: Any,
) -> dict[str, dict[str, Any]]:
    """Evaluate all current models once more for explicit harvest columns."""

    import numpy as np

    required_targets = _required_target_contract(models)
    predictions: dict[str, dict[str, Any]] = {}
    for target in required_targets:
        try:
            mean, half_width = models[target].predict_mu_sigma(frame, conformal=True)
        except TypeError as exc:
            raise RuntimeError(
                f"terminal model does not accept conformal=True: {target}"
            ) from exc
        mean = np.asarray(mean, dtype=float).reshape(-1)
        half_width = np.asarray(half_width, dtype=float).reshape(-1)
        if (
            mean.shape != (len(frame),)
            or half_width.shape != (len(frame),)
            or not np.isfinite(mean).all()
            or not np.isfinite(half_width).all()
            or np.any(half_width < 0.0)
        ):
            raise RuntimeError(f"terminal model inference failed: {target}")
        predictions[target] = {
            "mean": mean,
            "q90_conformal_half_width": half_width,
        }
    return predictions


def _canonical_terminal_prediction_payload(
    predictions: Mapping[str, Mapping[str, Any]],
    *,
    population_size: int,
) -> dict[str, dict[str, list[float]]]:
    """Validate and serialize the in-memory deterministic prediction snapshot."""

    import numpy as np

    count = int(population_size)
    if count < 1 or not isinstance(predictions, Mapping):
        raise RuntimeError("terminal prediction snapshot is invalid")
    required_targets = _required_target_contract(predictions)
    payload: dict[str, dict[str, list[float]]] = {}
    for target in required_targets:
        prediction = predictions[target]
        if not isinstance(prediction, Mapping) or set(prediction) != {
            "mean",
            "q90_conformal_half_width",
        }:
            raise RuntimeError(
                f"terminal prediction snapshot schema mismatch: {target}"
            )
        mean = np.asarray(prediction["mean"], dtype=float).reshape(-1)
        half_width = np.asarray(
            prediction["q90_conformal_half_width"], dtype=float
        ).reshape(-1)
        if (
            mean.shape != (count,)
            or half_width.shape != (count,)
            or not np.isfinite(mean).all()
            or not np.isfinite(half_width).all()
            or np.any(half_width < 0.0)
        ):
            raise RuntimeError(
                f"terminal prediction snapshot values are invalid: {target}"
            )
        payload[target] = {
            "mean": mean.tolist(),
            "q90_conformal_half_width": half_width.tolist(),
        }
    return payload


def _terminal_surrogate_physicality_gate(
    predictions: Mapping[str, Mapping[str, Any]],
    *,
    population_size: int,
) -> tuple[Any, dict[str, Any]]:
    """Reject non-physical terminal predictions without changing their value.

    This is deliberately a candidate-validity gate rather than an additional
    optimizer constraint.  It leaves the authenticated constraint contract
    and terminal replay arrays untouched while ensuring that a non-physical
    report-only component cannot be published as feasible or abort a completed
    seed during serialization.
    """

    import numpy as np

    count = int(population_size)
    if count < 0:
        raise RuntimeError("terminal surrogate gate population is invalid")
    required = (
        set(TERMINAL_NONNEGATIVE_SURROGATE_TARGETS)
        | set(TERMINAL_POSITIVE_SURROGATE_TARGETS)
        | {"k"}
    )
    if not required.issubset(predictions):
        raise RuntimeError("terminal surrogate gate target inventory is incomplete")

    valid = np.ones(count, dtype=bool)
    violations: list[dict[str, Any]] = []

    def inspect(target: str, rule: str, predicate: Any) -> None:
        values = np.asarray(predictions[target].get("mean"), dtype=float).reshape(-1)
        if values.shape != (count,) or not np.isfinite(values).all():
            raise RuntimeError(
                f"terminal surrogate gate received invalid means: {target}"
            )
        failed = ~np.asarray(predicate(values), dtype=bool)
        if failed.shape != (count,):
            raise RuntimeError("terminal surrogate gate rule shape mismatch")
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

    for target in TERMINAL_NONNEGATIVE_SURROGATE_TARGETS:
        inspect(target, "finite_and_nonnegative", lambda values: values >= 0.0)
    for target in TERMINAL_POSITIVE_SURROGATE_TARGETS:
        inspect(target, "finite_and_strictly_positive", lambda values: values > 0.0)
    inspect(
        "k",
        "finite_and_between_zero_and_one_exclusive",
        lambda values: (values > 0.0) & (values < 1.0),
    )

    invalid_indices = np.flatnonzero(~valid).astype(int).tolist()
    per_target: dict[str, int] = {}
    for violation in violations:
        target = str(violation["target"])
        per_target[target] = per_target.get(target, 0) + 1
    evidence = {
        "schema_version": TERMINAL_SURROGATE_PHYSICALITY_SCHEMA,
        "population_size": count,
        "valid_count": int(np.count_nonzero(valid)),
        "invalid_count": int(np.count_nonzero(~valid)),
        "invalid_population_indices": invalid_indices,
        "violations": violations,
        "violation_count_by_target": dict(sorted(per_target.items())),
        "nonnegative_targets": list(TERMINAL_NONNEGATIVE_SURROGATE_TARGETS),
        "strictly_positive_targets": list(TERMINAL_POSITIVE_SURROGATE_TARGETS),
        "coupling_rule": "0 < k < 1",
        "raw_predictions_preserved": True,
        "clamping_performed": False,
        "invalid_candidates_marked_infeasible": True,
        "optimizer_constraint_schema_mutated": False,
        "optimizer_objectives_mutated": False,
    }
    evidence["sha256"] = canonical_sha256(evidence)
    return valid, evidence


def _minimum_realized_insulation_mm(row: Any) -> float:
    values = [
        _finite_number(_row_value(row, name), name)
        for name in PHYSICAL_INSULATION_COLUMNS
    ]
    n1_side = _finite_number(_row_value(row, "N1_side"), "N1_side")
    if n1_side < 0.0:
        raise RuntimeError("N1_side must be non-negative")
    if n1_side > 0.0:
        values.extend(
            _finite_number(_row_value(row, name), name)
            for name in SIDE_PHYSICAL_INSULATION_COLUMNS
        )
    return float(min(values))


def _candidate_records(
    runner: Current7Tier1Runner,
    *,
    coordinates: Any,
    objectives: Any,
    physical_constraints: Any,
    frame: Any,
    predictions: Mapping[str, Mapping[str, Any]],
    surrogate_physicality: Mapping[str, Any] | None = None,
    indices: Any,
) -> list[dict[str, Any]]:
    """Build explicit UI/harvest records from authoritative physical replay."""

    import numpy as np

    x = np.asarray(coordinates, dtype=float)
    f = np.asarray(objectives, dtype=float)
    g = np.asarray(physical_constraints, dtype=float)
    if surrogate_physicality is None:
        _, surrogate_physicality = _terminal_surrogate_physicality_gate(
            predictions,
            population_size=len(x),
        )
    physicality = dict(surrogate_physicality)
    physicality_sha = physicality.pop("sha256", None)
    invalid_indices = physicality.get("invalid_population_indices")
    violations = physicality.get("violations")
    if (
        physicality_sha != canonical_sha256(physicality)
        or physicality.get("schema_version") != TERMINAL_SURROGATE_PHYSICALITY_SCHEMA
        or physicality.get("population_size") != len(x)
        or not isinstance(invalid_indices, list)
        or not isinstance(violations, list)
        or physicality.get("clamping_performed") is not False
        or physicality.get("raw_predictions_preserved") is not True
    ):
        raise RuntimeError("terminal surrogate physicality evidence is invalid")
    invalid_index_set = {int(value) for value in invalid_indices}
    violations_by_index: dict[int, list[dict[str, Any]]] = {}
    for raw_violation in violations:
        if not isinstance(raw_violation, dict):
            raise RuntimeError("terminal surrogate physicality violation is invalid")
        violation = dict(raw_violation)
        population_index = int(violation.get("population_index", -1))
        if population_index < 0 or population_index >= len(x):
            raise RuntimeError("terminal surrogate physicality index is invalid")
        violations_by_index.setdefault(population_index, []).append(violation)
    if invalid_index_set != set(violations_by_index):
        raise RuntimeError("terminal surrogate physicality indices are inconsistent")
    records: list[dict[str, Any]] = []
    required_model_targets = _required_target_contract(predictions)
    temperature_targets = tuple(
        getattr(runner.problem, "temperature_targets", CURRENT_TEMPERATURE_TARGETS)
    )
    if getattr(runner.problem, "goal_campaign", False):
        if temperature_targets != GOAL_TEMPERATURE_TARGETS:
            raise RuntimeError("goal terminal temperature target schema mismatch")
    elif temperature_targets != CURRENT_TEMPERATURE_TARGETS:
        raise RuntimeError("current7 terminal temperature target schema mismatch")
    for raw_index in np.asarray(indices, dtype=int).reshape(-1):
        index = int(raw_index)
        row = _frame_row(frame, index)
        decoded = _jsonable_decoded_parameters(row)
        physical_geometry = {
            name: decoded.get(name)
            for name in DECODED_GEOMETRY_IDENTITY_COLUMNS
        }
        physical_geometry_sha256 = canonical_sha256(physical_geometry)
        canonical_physical_params_sha256 = canonical_sha256(decoded)
        candidate_physics_sha = physical_geometry_sha256
        goal_identity: dict[str, Any] = {}
        if getattr(runner.problem, "goal_campaign", False):
            identity_values = dict(decoded)
            identity_values["thermal_pad_conductivity_W_mK"] = (
                GOAL_FIXED_COOLING_IDENTITY[
                    "thermal_pad_conductivity_W_mK"
                ]
            )
            fixed_identity = attest_fixed_identity(identity_values)
            goal_identity = {
                "goal_contract_schema": GOAL_CONTRACT_SCHEMA,
                "hard_spec": copy.deepcopy(runner.problem.stage_spec),
                "hard_spec_sha256": runner.problem.stage_spec_sha256,
                "stage_spec_sha256": runner.problem.stage_spec_sha256,
                "temperature_contract_sha256": (
                    runner.problem.temperature_contract_sha256
                ),
                "hard_constraint_contract_sha256": (
                    runner.problem.hard_constraint_contract_sha256
                ),
                "constraint_version": (
                    runner.problem.hard_constraint_contract["stage"]
                ),
                "fixed_operating_cooling_identity": (
                    fixed_identity["observed"]
                ),
                "fixed_identity_attestation": fixed_identity,
            }
        provenance_identity = {
            "physical_geometry_sha256": physical_geometry_sha256,
            "canonical_physical_params_sha256": (
                canonical_physical_params_sha256
            ),
            "candidate_physics_sha": candidate_physics_sha,
            **goal_identity,
        }
        means = {
            target: float(predictions[target]["mean"][index])
            for target in required_model_targets
        }
        half_widths = {
            target: float(predictions[target]["q90_conformal_half_width"][index])
            for target in required_model_targets
        }
        aggregate_loss = sum(
            means[target]
            for target in (
                "P_winding_total",
                "P_core_total",
                "P_core_plate_total",
                "P_wcp_total",
            )
        )
        if not math.isclose(
            aggregate_loss, float(f[index, 1]), rel_tol=1e-9, abs_tol=1e-6
        ):
            raise RuntimeError("terminal loss objective differs from model harvest")
        constraint_g = {
            name: float(g[index, position])
            for position, name in enumerate(runner.problem.constraint_names)
        }
        physical_constraint_feasible = all(
            value <= 0.0 for value in constraint_g.values()
        )
        candidate_violations = violations_by_index.get(index, [])
        physicality_gate = {
            "schema_version": TERMINAL_SURROGATE_PHYSICALITY_SCHEMA,
            "passed": not candidate_violations,
            "violations": candidate_violations,
            "raw_predictions_preserved": True,
            "clamping_performed": False,
        }
        physicality_gate["sha256"] = canonical_sha256(physicality_gate)
        if candidate_violations:
            volume_l, dimensions = runner.modules.geometry_metrics.bounding_box_lit(row)
            width_mm, length_mm, height_mm = (
                _finite_number(value, "exterior dimension") for value in dimensions
            )
            record = {
                **provenance_identity,
                "candidate_id": f"terminal-{index:04d}",
                "candidate_record_status": ("quarantined_nonphysical_surrogate_output"),
                "terminal_population_index": index,
                "coordinate_unit": x[index].tolist(),
                "size_W_mm": width_mm,
                "size_L_mm": length_mm,
                "size_H_mm": height_mm,
                "volume_L": _finite_number(volume_l, "volume"),
                "predicted_total_loss_W": float(aggregate_loss),
                "total_loss_W": float(aggregate_loss),
                "predicted_winding_loss_W": means["P_winding_total"],
                "predicted_core_loss_W": means["P_core_total"],
                "predicted_core_plate_loss_W": means["P_core_plate_total"],
                "predicted_winding_cold_plate_loss_W": means["P_wcp_total"],
                "predicted_Tx_main_winding_loss_W": means["P_Tx_main_group"],
                "predicted_Rx_main_winding_loss_W": means["P_Rx_main_group"],
                "predicted_Rx_side_winding_loss_W": means["P_Rx_side_total"],
                "pred_Llt_phys_uH": means["Llt_phys"],
                "pred_Llt_phys": means["Llt_phys"],
                "surrogate_mean_predictions": means,
                "surrogate_q90_conformal_half_widths": half_widths,
                "terminal_surrogate_physicality_gate": physicality_gate,
                "decoded_params": decoded,
                "physical_constraint_G": constraint_g,
                "physical_constraint_margin": {
                    name: -value for name, value in constraint_g.items()
                },
                "physical_constraint_feasible": physical_constraint_feasible,
                "physical_feasible": False,
                "production_eligible": False,
                "fea_submission_approved": False,
                "fea_submission_performed": False,
                "aedt_used": False,
                "automatic_promotion_allowed": False,
            }
            records.append(record)
            continue
        design_report = runner.modules.design_summary.pareto_design_summary(
            row,
            means,
            aggregate_loss,
            leakage_target_uH=runner.problem.spec["Llt_target_uH"],
            core_lamination_factor=runner.problem.spec["core_lamination_factor"],
            B_area_basis=runner.problem.spec["B_area_basis"],
        )
        width_mm = _finite_number(design_report["size_W_mm"], "exterior width")
        length_mm = _finite_number(design_report["size_L_mm"], "exterior length")
        height_mm = _finite_number(design_report["size_H_mm"], "exterior height")
        volume_l = _finite_number(design_report["volume_L"], "volume")
        analytical_b = _finite_number(
            design_report["B_design_analytic_T"], "analytical B"
        )
        resonance = derive_half_magnetizing_self_resonance(
            {
                "Llt_phys": means["Llt_phys"],
                "k": means["k"],
                "C_tx_tx_F": means["C_tx_tx_F"],
                "C_rx_rx_F": means["C_rx_rx_F"],
            },
            row,
            magnetizing_inductance_factor=runner.problem.spec[
                "magnetizing_inductance_factor"
            ],
        )
        cross_frequency = 1.0 / (
            2.0
            * math.pi
            * math.sqrt(
                _positive_number(means["Llt_phys"], "Llt_phys")
                * 1e-6
                * _positive_number(means["C_tx_rx_F"], "C_tx_rx_F")
            )
        )
        cross_frequency = _positive_number(
            cross_frequency, "interwinding resonance frequency"
        )
        temperature_limits = getattr(
            runner.problem, "temperature_limits_C", None
        )
        if temperature_limits is None:
            temperature_limits = {
                target: float(runner.problem.spec["T_limit_C"])
                for target in temperature_targets
            }
        temperature_predictions = {
            target: {
                "mean_C": means[target],
                "q90_conformal_half_width_C": half_widths[target],
                "robust_upper_C": means[target] + half_widths[target],
                "limit_C": float(temperature_limits[target]),
                "robust_margin_C": (
                    float(temperature_limits[target])
                    - means[target]
                    - half_widths[target]
                ),
            }
            for target in temperature_targets
        }
        n2_side = int(_finite_number(_row_value(row, "N2_side"), "N2_side"))
        conditional_side_targets = (
            set(GOAL_SIDE_TEMPERATURE_TARGETS)
            if getattr(runner.problem, "goal_campaign", False)
            else {SIDE_TEMPERATURE_TARGET}
        )
        active_temperature_targets = [
            target
            for target in temperature_targets
            if target not in conditional_side_targets or n2_side > 0
        ]
        maximum_temperature = max(
            means[target] for target in active_temperature_targets
        )
        maximum_robust_temperature = max(
            means[target] + half_widths[target] for target in active_temperature_targets
        )
        winding_targets = [
            target
            for target in active_temperature_targets
            if target in WINDING_TEMPERATURE_TARGETS
        ]
        core_targets = [
            target
            for target in active_temperature_targets
            if target in CORE_TEMPERATURE_TARGETS
        ]
        maximum_robust_winding_temperature = max(
            means[target] + half_widths[target] for target in winding_targets
        )
        maximum_robust_core_temperature = max(
            means[target] + half_widths[target] for target in core_targets
        )
        budget = winding_budget_identity(
            row,
            expected_cw1_mm=expected_problem_cw1_mm(runner.problem, row),
        )
        if budget.get("passed") is not True:
            raise RuntimeError("harvest winding-budget identity failed")
        record = {
            **design_report,
            **provenance_identity,
            "candidate_id": f"terminal-{index:04d}",
            "terminal_population_index": index,
            "coordinate_unit": x[index].tolist(),
            "size_W_mm": width_mm,
            "size_L_mm": length_mm,
            "size_H_mm": height_mm,
            "volume_L": float(volume_l),
            "predicted_total_loss_W": float(aggregate_loss),
            "total_loss_W": float(aggregate_loss),
            "predicted_winding_loss_W": means["P_winding_total"],
            "predicted_core_loss_W": means["P_core_total"],
            "predicted_core_plate_loss_W": means["P_core_plate_total"],
            "predicted_winding_cold_plate_loss_W": means["P_wcp_total"],
            "predicted_Tx_main_winding_loss_W": means["P_Tx_main_group"],
            "predicted_Rx_main_winding_loss_W": means["P_Rx_main_group"],
            "predicted_Rx_side_winding_loss_W": means["P_Rx_side_total"],
            "pred_Llt_phys_uH": means["Llt_phys"],
            "pred_Llt_phys": means["Llt_phys"],
            "q90_Llt_phys_half_width_uH": half_widths["Llt_phys"],
            "robust_Llt_low_uH": means["Llt_phys"] - half_widths["Llt_phys"],
            "robust_Llt_high_uH": means["Llt_phys"] + half_widths["Llt_phys"],
            "pred_k": means["k"],
            "pred_B_mean_core_T": means["B_mean_core"],
            "pred_B_mean_core": means["B_mean_core"],
            "q90_B_mean_core_half_width_T": half_widths["B_mean_core"],
            "analytical_B_T": analytical_b,
            "B_design_analytic_T": analytical_b,
            "minimum_realized_insulation_mm": (_minimum_realized_insulation_mm(row)),
            "pred_C_tx_tx_F": means["C_tx_tx_F"],
            "pred_C_rx_rx_F": means["C_rx_rx_F"],
            "pred_C_tx_rx_F": means["C_tx_rx_F"],
            **resonance,
            "pred_f_res_tx_screen_Hz": resonance["f_res_tx_half_magnetizing_Hz"],
            "pred_f_res_rx_screen_Hz": resonance["f_res_rx_half_magnetizing_Hz"],
            "pred_f_res_min_screen_Hz": resonance["f_res_min_tx_rx_only_Hz"],
            "pred_f_res_interwinding_screen_Hz": cross_frequency,
            "pred_max_temperature_C": maximum_temperature,
            "pred_max_robust_temperature_C": maximum_robust_temperature,
            "robust_max_temperature_C": maximum_robust_temperature,
            "pred_max_robust_winding_temperature_C": (
                maximum_robust_winding_temperature
            ),
            "pred_max_robust_core_temperature_C": (
                maximum_robust_core_temperature
            ),
            **{
                f"pred_{target}": means[target]
                for target in temperature_targets
            },
            **{
                f"q90_{target}_half_width_C": half_widths[target]
                for target in temperature_targets
            },
            **{
                f"robust_{target}_upper_C": (means[target] + half_widths[target])
                for target in temperature_targets
            },
            "N1_main": int(_finite_number(_row_value(row, "N1_main"), "N1_main")),
            "N1_side": int(_finite_number(_row_value(row, "N1_side"), "N1_side")),
            "N2_main": int(_finite_number(_row_value(row, "N2_main"), "N2_main")),
            "N2_side": n2_side,
            "n_core_group": int(
                _finite_number(_row_value(row, "n_core_group"), "n_core_group")
            ),
            "cw1_mm": _finite_number(_row_value(row, "cw1"), "cw1"),
            "temperature_predictions": temperature_predictions,
            "active_temperature_targets_for_maximum": (active_temperature_targets),
            "surrogate_mean_predictions": means,
            "surrogate_q90_conformal_half_widths": half_widths,
            "decoded_params": decoded,
            "winding_budget_identity": budget,
            "terminal_surrogate_physicality_gate": physicality_gate,
            "physical_constraint_G": constraint_g,
            "physical_constraint_margin": {
                name: -value for name, value in constraint_g.items()
            },
            "physical_constraint_feasible": physical_constraint_feasible,
            "physical_feasible": physical_constraint_feasible,
            "production_eligible": False,
            "fea_submission_approved": False,
            "fea_submission_performed": False,
            "aedt_used": False,
            "automatic_promotion_allowed": False,
        }
        records.append(record)
    return records


def _candidate_csv_frame(
    records: list[dict[str, Any]], *, template: dict[str, Any]
) -> Any:
    import pandas as pd

    rows = []
    source = records or [template]
    for record in source:
        flat = {}
        for name, value in record.items():
            if isinstance(value, (dict, list)):
                flat[f"{name}_json"] = json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
            else:
                flat[name] = value
        rows.append(flat)
    frame = pd.DataFrame(rows)
    return frame if records else frame.iloc[0:0]


def _terminal_physical_candidate_frame(
    runner: Current7Tier1Runner,
    *,
    coordinates: Any,
    objectives: Any,
    optimizer_constraints: Any,
    physical_constraints: Any,
    frame: Any,
    decoder_valid: Any,
    surrogate_physical_valid: Any,
    source_identity: Mapping[str, Any] | None,
) -> Any:
    """Build one provenance-complete row for every terminal individual."""

    import numpy as np
    import pandas as pd

    x = np.asarray(coordinates, dtype=float)
    f = np.asarray(objectives, dtype=float)
    optimizer_g = np.asarray(optimizer_constraints, dtype=float)
    physical_g = np.asarray(physical_constraints, dtype=float)
    valid = np.asarray(decoder_valid, dtype=bool).reshape(-1)
    surrogate_valid = np.asarray(
        surrogate_physical_valid, dtype=bool
    ).reshape(-1)
    count = len(x)
    expected_g = (count, len(runner.problem.constraint_names))
    if (
        x.ndim != 2
        or f.shape != (count, 2)
        or optimizer_g.shape != expected_g
        or physical_g.shape != expected_g
        or valid.shape != (count,)
        or surrogate_valid.shape != (count,)
        or len(frame) != count
        or not np.isfinite(x).all()
        or not np.isfinite(f).all()
        or not np.isfinite(optimizer_g).all()
        or not np.isfinite(physical_g).all()
    ):
        raise RuntimeError("terminal physical candidate table inputs are invalid")

    source = dict(source_identity or {})
    goal_campaign = bool(getattr(runner.problem, "goal_campaign", False))
    required_source = {
        "seed",
        "task_id",
        "bundle_id",
        "island_id",
        "dataset_sha256",
        "model_artifacts_sha256",
        "model_generation_sha256",
        "evaluation_spec_sha256",
        "temperature_contract_sha256",
        "hard_constraint_contract_sha256",
    }
    if goal_campaign:
        if count != PRODUCTION_POPULATION:
            raise RuntimeError(
                "goal terminal physical candidate table requires 320 rows"
            )
        if not valid.all():
            raise RuntimeError(
                "goal terminal physical candidate table requires every row decoded"
            )
        if required_source - set(source):
            raise RuntimeError(
                "goal terminal physical candidate table source identity is incomplete"
            )
        if any(
            source.get(name) in (None, "")
            for name in required_source
        ):
            raise RuntimeError(
                "goal terminal physical candidate table source identity is null"
            )
        if (
            source["evaluation_spec_sha256"]
            != runner.problem.stage_spec_sha256
            or source["temperature_contract_sha256"]
            != runner.problem.temperature_contract_sha256
            or source["hard_constraint_contract_sha256"]
            != runner.problem.hard_constraint_contract_sha256
        ):
            raise RuntimeError(
                "goal terminal physical candidate table contract identity drifted"
            )

    rows: list[dict[str, Any]] = []
    for index in range(count):
        decoded = _jsonable_decoded_parameters(_frame_row(frame, index))
        geometry = {
            name: decoded.get(name)
            for name in DECODED_GEOMETRY_IDENTITY_COLUMNS
        }
        if goal_campaign and any(value is None for value in geometry.values()):
            raise RuntimeError(
                "goal terminal physical geometry identity is incomplete"
            )
        geometry_sha = canonical_sha256(geometry)
        physical_params_sha = canonical_sha256(decoded)
        candidate_physics_sha = geometry_sha
        physical_values = {
            name: float(physical_g[index, position])
            for position, name in enumerate(runner.problem.constraint_names)
        }
        normalized_values = {
            name: float(optimizer_g[index, position])
            for position, name in enumerate(runner.problem.constraint_names)
        }
        physical_constraint_feasible = bool(
            valid[index]
            and all(value <= 0.0 for value in physical_values.values())
        )
        physical_feasible = bool(
            physical_constraint_feasible and surrogate_valid[index]
        )
        row = {
            "terminal_population_index": index,
            "decoder_valid": bool(valid[index]),
            "surrogate_physical_valid": bool(surrogate_valid[index]),
            "surrogate_physicality_passed": bool(
                surrogate_valid[index]
            ),
            "physical_geometry_sha256": geometry_sha,
            "canonical_physical_params_sha256": physical_params_sha,
            "candidate_physics_sha": candidate_physics_sha,
            "objective_volume_L": float(f[index, 0]),
            "objective_total_loss_W": float(f[index, 1]),
            "physical_constraint_feasible": physical_constraint_feasible,
            "physical_feasible": physical_feasible,
            "physical_G_json": json.dumps(
                physical_values,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            "normalized_G_json": json.dumps(
                normalized_values,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            "coordinate_unit_json": json.dumps(
                x[index].tolist(),
                separators=(",", ":"),
                allow_nan=False,
            ),
            "decoded_physical_params_json": json.dumps(
                decoded,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ),
            "source_seed": source.get("seed"),
            "source_task_id": source.get("task_id"),
            "source_bundle_id": source.get("bundle_id"),
            "source_island_id": source.get("island_id"),
            "dataset_sha256": source.get("dataset_sha256"),
            "evaluation_model_sha256": source.get(
                "model_artifacts_sha256"
            ),
            "constraint_spec_sha256": runner.problem.stage_spec_sha256,
            "cooling_contract_sha256": (
                GOAL_FIXED_COOLING_IDENTITY_SHA256
                if goal_campaign
                else None
            ),
            "operating_point_sha256": (
                GOAL_FIXED_OPERATING_IDENTITY_SHA256
                if goal_campaign
                else None
            ),
            "evaluation_model_artifacts_sha256": source.get(
                "model_artifacts_sha256"
            ),
            "evaluation_model_generation_sha256": source.get(
                "model_generation_sha256"
            ),
            "evaluation_spec_sha256": runner.problem.stage_spec_sha256,
            "evaluation_temperature_contract_sha256": (
                runner.problem.temperature_contract_sha256
            ),
            "evaluation_hard_constraint_contract_sha256": (
                runner.problem.hard_constraint_contract_sha256
            ),
        }
        for name, value in physical_values.items():
            row[f"physical_G:{name}"] = value
        for name, value in normalized_values.items():
            row[f"normalized_G:{name}"] = value
        rows.append(row)
    result = pd.DataFrame(rows)
    if (
        len(result) != count
        or result["terminal_population_index"].tolist() != list(range(count))
        or result["candidate_physics_sha"].isna().any()
    ):
        raise RuntimeError("terminal physical candidate table assembly failed")
    return result


def _select_terminal_least_violation(
    optimizer_constraints: Any,
    *,
    decoder_valid: Any,
    surrogate_physical_valid: Any,
) -> dict[str, Any]:
    """Select a reportable near-candidate without hiding the raw argmin."""

    import numpy as np

    constraints = np.asarray(optimizer_constraints, dtype=float)
    decoded = np.asarray(decoder_valid, dtype=bool).reshape(-1)
    physical = np.asarray(surrogate_physical_valid, dtype=bool).reshape(-1)
    if (
        constraints.ndim != 2
        or len(constraints) < 1
        or decoded.shape != (len(constraints),)
        or physical.shape != (len(constraints),)
        or not np.isfinite(constraints).all()
    ):
        raise RuntimeError("terminal least-violation selection inputs are invalid")
    positive_sum = np.maximum(constraints, 0.0).sum(axis=1)
    raw_global_index = int(np.argmin(positive_sum))
    eligible_indices = np.flatnonzero(decoded & physical)
    if len(eligible_indices):
        selected_index = int(
            eligible_indices[np.argmin(positive_sum[eligible_indices])]
        )
        fallback = False
    else:
        selected_index = raw_global_index
        fallback = True
    return {
        "positive_G_sum": positive_sum,
        "raw_global_index": raw_global_index,
        "selected_index": selected_index,
        "eligible_count": int(len(eligible_indices)),
        "fallback_to_quarantined_raw_global_least": fallback,
    }


def persist_search_outputs(
    runner: Current7Tier1Runner,
    result: Any,
    output: Path,
    *,
    source_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist physical terminal, Pareto and closest-candidate evidence."""

    import numpy as np
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    output = output.resolve(strict=True)
    terminal = getattr(result, "pop", None)
    if terminal is None:
        terminal = getattr(getattr(result, "algorithm", None), "pop", None)
    if terminal is None or not callable(getattr(terminal, "get", None)):
        raise RuntimeError("terminal population is unavailable for persistence")
    terminal_x = np.asarray(terminal.get("X"), dtype=float)
    terminal_f = np.asarray(terminal.get("F"), dtype=float)
    optimizer_g = np.asarray(terminal.get("G"), dtype=float)
    replay = result.tier1_terminal_physical_replay
    physical_g = np.asarray(replay["G"], dtype=float)
    decoder_valid = np.asarray(replay["decoder_valid"], dtype=bool)
    frame = replay["frame"]
    expected = (len(terminal_x), len(runner.problem.constraint_names))
    if (
        terminal_x.ndim != 2
        or terminal_x.shape[1] != int(runner.problem.n_var)
        or terminal_f.shape != (len(terminal_x), 2)
        or optimizer_g.shape != expected
        or physical_g.shape != expected
        or decoder_valid.shape != (len(terminal_x),)
        or len(frame) != len(terminal_x)
        or not np.isfinite(terminal_x).all()
        or not np.isfinite(terminal_f).all()
        or not np.isfinite(optimizer_g).all()
        or not np.isfinite(physical_g).all()
    ):
        raise RuntimeError("terminal persistence arrays are invalid")
    predictions = replay.get("terminal_model_predictions")
    serial_predictions = replay.get("terminal_model_predictions_serially_replayed")
    if serial_predictions is True:
        replay_objectives = np.asarray(replay.get("F"), dtype=float)
        replay_optimizer_g = np.asarray(replay.get("optimizer_G"), dtype=float)
        if not np.array_equal(terminal_f, replay_objectives) or not np.array_equal(
            optimizer_g, replay_optimizer_g
        ):
            raise RuntimeError(
                "terminal population differs from deterministic replay authority"
            )
        if predictions is None:
            raise RuntimeError("deterministic terminal prediction snapshot is missing")
        prediction_payload = _canonical_terminal_prediction_payload(
            predictions,
            population_size=len(terminal_x),
        )
        if replay.get("terminal_model_predictions_sha256") != canonical_sha256(
            prediction_payload
        ):
            raise RuntimeError(
                "deterministic terminal prediction snapshot SHA mismatch"
            )
    elif serial_predictions is False:
        raise RuntimeError("canonical terminal replay omitted serial model predictions")
    elif predictions is None:
        predictions = _terminal_model_predictions(runner.models, frame)
    surrogate_physical_valid, surrogate_physicality = (
        _terminal_surrogate_physicality_gate(
            predictions,
            population_size=len(terminal_x),
        )
    )
    physical_constraint_feasible = decoder_valid & np.all(physical_g <= 0.0, axis=1)
    physical_feasible = physical_constraint_feasible & surrogate_physical_valid
    feasible_indices = np.flatnonzero(physical_feasible)
    if len(feasible_indices):
        local = NonDominatedSorting().do(
            terminal_f[feasible_indices], only_non_dominated_front=True
        )
        pareto_indices = feasible_indices[np.asarray(local, dtype=int)]
    else:
        pareto_indices = np.empty(0, dtype=int)
    least_selection = _select_terminal_least_violation(
        optimizer_g,
        decoder_valid=decoder_valid,
        surrogate_physical_valid=surrogate_physical_valid,
    )
    optimizer_positive = least_selection["positive_G_sum"]
    raw_global_least_index = int(least_selection["raw_global_index"])
    least_index = int(least_selection["selected_index"])
    least_indices = np.asarray([least_index], dtype=int)
    pareto_records = _candidate_records(
        runner,
        coordinates=terminal_x,
        objectives=terminal_f,
        physical_constraints=physical_g,
        frame=frame,
        predictions=predictions,
        surrogate_physicality=surrogate_physicality,
        indices=pareto_indices,
    )
    least_records = _candidate_records(
        runner,
        coordinates=terminal_x,
        objectives=terminal_f,
        physical_constraints=physical_g,
        frame=frame,
        predictions=predictions,
        surrogate_physicality=surrogate_physicality,
        indices=least_indices,
    )
    terminal_physical_candidates = _terminal_physical_candidate_frame(
        runner,
        coordinates=terminal_x,
        objectives=terminal_f,
        optimizer_constraints=optimizer_g,
        physical_constraints=physical_g,
        frame=frame,
        decoder_valid=decoder_valid,
        surrogate_physical_valid=surrogate_physical_valid,
        source_identity=source_identity,
    )
    paths = {
        "pareto_X": output / "pareto_X.npy",
        "pareto_F": output / "pareto_F.npy",
        "pareto_G_physical": output / "pareto_G_physical.npy",
        "pareto_front": output / "pareto_front.csv",
        "pareto_candidates": output / "pareto_candidates.json",
        "terminal_X": output / "terminal_X.npy",
        "terminal_F": output / "terminal_F.npy",
        "terminal_G_optimizer": output / "terminal_G_optimizer.npy",
        "terminal_G_physical": output / "terminal_G_physical.npy",
        "terminal_physical_candidates": (
            output / "terminal_physical_candidates.csv"
        ),
        "terminal_physical_candidates_manifest": (
            output / "terminal_physical_candidates.manifest.json"
        ),
        "least_violation_X": output / "least_violation_X.npy",
        "least_violation_F": output / "least_violation_F.npy",
        "least_violation_G_physical": output / "least_violation_G_physical.npy",
        "least_violation_front": output / "least_violation.csv",
        "least_violation_candidates": output / "least_violation_candidates.json",
        "infeasibility_report": output / "infeasibility_report.json",
    }
    if any(path.exists() for path in paths.values()):
        raise RuntimeError("search output artifact already exists")
    arrays = {
        "pareto_X": terminal_x[pareto_indices],
        "pareto_F": terminal_f[pareto_indices],
        "pareto_G_physical": physical_g[pareto_indices],
        "terminal_X": terminal_x,
        "terminal_F": terminal_f,
        "terminal_G_optimizer": optimizer_g,
        "terminal_G_physical": physical_g,
        "least_violation_X": terminal_x[least_indices],
        "least_violation_F": terminal_f[least_indices],
        "least_violation_G_physical": physical_g[least_indices],
    }
    for name, values in arrays.items():
        _atomic_npy(paths[name], values)
    _atomic_csv(
        paths["terminal_physical_candidates"],
        terminal_physical_candidates,
    )
    terminal_table_manifest = {
        "schema_version": GOAL_TERMINAL_TABLE_SCHEMA,
        "goal_contract_required": bool(
            getattr(runner.problem, "goal_campaign", False)
        ),
        "row_count": int(len(terminal_physical_candidates)),
        "terminal_population_index_min": int(
            terminal_physical_candidates["terminal_population_index"].min()
        ),
        "terminal_population_index_max": int(
            terminal_physical_candidates["terminal_population_index"].max()
        ),
        "columns": list(terminal_physical_candidates.columns),
        "required_identity_columns": [
            "terminal_population_index",
            "decoder_valid",
            "surrogate_physical_valid",
            "surrogate_physicality_passed",
            "physical_constraint_feasible",
            "physical_feasible",
            "physical_geometry_sha256",
            "canonical_physical_params_sha256",
            "candidate_physics_sha",
            "objective_volume_L",
            "objective_total_loss_W",
            "physical_G_json",
            "normalized_G_json",
            "source_seed",
            "source_task_id",
            "source_bundle_id",
            "source_island_id",
            "dataset_sha256",
            "evaluation_model_sha256",
            "constraint_spec_sha256",
            "cooling_contract_sha256",
            "operating_point_sha256",
            "evaluation_model_artifacts_sha256",
            "evaluation_model_generation_sha256",
            "evaluation_spec_sha256",
            "evaluation_temperature_contract_sha256",
            "evaluation_hard_constraint_contract_sha256",
        ],
        "csv": {
            "path": paths["terminal_physical_candidates"].name,
            "sha256": sha256_file(paths["terminal_physical_candidates"]),
            "size_bytes": paths["terminal_physical_candidates"].stat().st_size,
        },
        "source_identity": copy.deepcopy(dict(source_identity or {})),
        "stage_spec_sha256": runner.problem.stage_spec_sha256,
        "temperature_contract_sha256": (
            runner.problem.temperature_contract_sha256
        ),
        "hard_constraint_contract_sha256": (
            runner.problem.hard_constraint_contract_sha256
        ),
        "one_row_per_terminal_individual": True,
        "physical_deduplication_key": "physical_geometry_sha256",
        "global_pareto_provenance_ready": True,
    }
    terminal_table_manifest["payload_sha256"] = canonical_sha256(
        terminal_table_manifest
    )
    _atomic_json(
        paths["terminal_physical_candidates_manifest"],
        terminal_table_manifest,
    )
    _atomic_csv(
        paths["pareto_front"],
        _candidate_csv_frame(pareto_records, template=least_records[0]),
    )
    _atomic_csv(
        paths["least_violation_front"],
        _candidate_csv_frame(least_records, template=least_records[0]),
    )
    _atomic_json(
        paths["pareto_candidates"],
        {
            "schema_version": "mft-tier1-current7-pareto-candidates-v1",
            "authoritative_constraints": "terminal_unscaled_physical_replay",
            "candidate_count": len(pareto_records),
            "candidates": pareto_records,
            "production_eligible": False,
            "fea_submission_performed": False,
            "automatic_promotion_allowed": False,
        },
    )
    _atomic_json(
        paths["least_violation_candidates"],
        {
            "schema_version": "mft-tier1-current7-least-violation-candidates-v1",
            "ranking": "minimum_sum_positive_optimizer_normalized_G",
            "ranking_eligibility": ("decoded_and_terminal_surrogate_physicality_valid"),
            "eligible_population_count": least_selection["eligible_count"],
            "raw_global_least_population_index": raw_global_least_index,
            "selected_least_population_index": least_index,
            "fallback_to_quarantined_raw_global_least": least_selection[
                "fallback_to_quarantined_raw_global_least"
            ],
            "candidate_count": len(least_records),
            "candidates": least_records,
            "production_eligible": False,
            "fea_submission_performed": False,
            "automatic_promotion_allowed": False,
        },
    )
    per_constraint = []
    for position, name in enumerate(runner.problem.constraint_names):
        values = physical_g[:, position]
        per_constraint.append(
            {
                "index": position,
                "name": name,
                "finite_count": int(np.count_nonzero(np.isfinite(values))),
                "passing_count": int(np.count_nonzero(values <= 0.0)),
                "minimum_physical_G": float(np.min(values)),
                "median_physical_G": float(np.median(values)),
                "minimum_positive_violation": float(np.min(np.maximum(values, 0.0))),
            }
        )
    infeasibility = {
        "schema_version": "mft-tier1-current7-infeasibility-report-v1",
        "authoritative_constraints": "terminal_unscaled_physical_replay",
        "population_size": int(len(terminal_x)),
        "physical_constraint_feasible_count": int(
            np.count_nonzero(physical_constraint_feasible)
        ),
        "physical_feasible_count": int(np.count_nonzero(physical_feasible)),
        "optimizer_feasible_count": int(
            np.count_nonzero(np.all(optimizer_g <= 0.0, axis=1))
        ),
        "raw_global_least_population_index": raw_global_least_index,
        "raw_global_least_optimizer_positive_G_sum": float(
            optimizer_positive[raw_global_least_index]
        ),
        "raw_global_least_physical_positive_G_sum": float(
            np.maximum(physical_g[raw_global_least_index], 0.0).sum()
        ),
        "raw_global_least_surrogate_physicality_passed": bool(
            surrogate_physical_valid[raw_global_least_index]
        ),
        "raw_global_least_physical_constraint_G": {
            name: float(physical_g[raw_global_least_index, position])
            for position, name in enumerate(runner.problem.constraint_names)
        },
        "least_violation_population_index": least_index,
        "least_violation_selection_eligibility": (
            "decoded_and_terminal_surrogate_physicality_valid"
        ),
        "least_violation_eligible_population_count": least_selection["eligible_count"],
        "fallback_to_quarantined_raw_global_least": least_selection[
            "fallback_to_quarantined_raw_global_least"
        ],
        "least_optimizer_positive_G_sum": float(optimizer_positive[least_index]),
        "least_physical_positive_G_sum": float(
            np.maximum(physical_g[least_index], 0.0).sum()
        ),
        "least_physical_constraint_G": {
            name: float(physical_g[least_index, position])
            for position, name in enumerate(runner.problem.constraint_names)
        },
        "least_candidate_surrogate_physicality_passed": bool(
            surrogate_physical_valid[least_index]
        ),
        "terminal_surrogate_physicality_gate": surrogate_physicality,
        "constraints": per_constraint,
        "production_eligible": False,
        "fea_submission_performed": False,
        "automatic_promotion_allowed": False,
    }
    _atomic_json(paths["infeasibility_report"], infeasibility)
    inventory = {}
    for name, path in paths.items():
        record = {
            "path": path.name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        if name in arrays:
            value = np.asarray(arrays[name])
            record.update({"shape": list(value.shape), "dtype": str(value.dtype)})
        inventory[name] = record
    return {
        "terminal_population_count": int(len(terminal_x)),
        "physical_feasible_count": int(np.count_nonzero(physical_feasible)),
        "feasible_pareto_count": int(len(pareto_indices)),
        "least_violation_count": 1,
        "terminal_population_primary_turn_values": sorted(
            {
                int(
                    _finite_number(
                        _row_value(_frame_row(frame, index), "N1_main"), "N1_main"
                    )
                )
                + int(
                    _finite_number(
                        _row_value(_frame_row(frame, index), "N1_side"), "N1_side"
                    )
                )
                for index in range(len(frame))
            }
        ),
        "artifact_inventory": inventory,
        "artifact_inventory_sha256": canonical_sha256(inventory),
        "infeasibility_report": infeasibility,
        "terminal_physical_candidates_manifest": terminal_table_manifest,
    }


def build_smoke_receipt(
    *,
    runners: Mapping[str, Current7Tier1Runner],
    coordinate_evidence: Mapping[str, Any],
    evaluations: Mapping[str, Mapping[str, Any]],
    model_smoke_by_stratum: Mapping[str, Mapping[str, Any]],
    repair_smoke_by_stratum: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Seal one launch gate only after both fixed-N1 strata pass."""

    import numpy as np

    if not runners:
        raise RuntimeError("smoke runner inventory is empty")
    probe_runner = next(iter(runners.values()))
    goal_campaign = bool(
        getattr(probe_runner.problem, "goal_campaign", False)
    )
    supported_turns = (
        GOAL_SUPPORTED_PRIMARY_TURNS
        if goal_campaign
        else SUPPORTED_FIXED_PRIMARY_TURNS
    )
    required_targets = (
        GOAL_REQUIRED_MODEL_TARGETS
        if goal_campaign
        else CURRENT_REQUIRED_MODEL_TARGETS
    )
    required_targets_sha256 = (
        GOAL_REQUIRED_MODEL_TARGETS_SHA256
        if goal_campaign
        else CURRENT_REQUIRED_MODEL_TARGETS_SHA256
    )
    base_constraint_names = (
        GOAL_BASE_CONSTRAINT_NAMES if goal_campaign else BASE_CONSTRAINT_NAMES
    )
    expected_keys = tuple(str(value) for value in supported_turns)
    supplied = (
        tuple(sorted(runners))
        == tuple(sorted(evaluations))
        == tuple(sorted(model_smoke_by_stratum))
        == tuple(sorted(repair_smoke_by_stratum))
    )
    if not supplied or set(runners) != set(expected_keys):
        raise RuntimeError("smoke inventory does not cover every campaign N1 stratum")
    first = runners[expected_keys[0]]
    stage_spec = validate_stage_spec(first.problem.stage_spec)
    constraint_names = tuple(first.problem.constraint_names)
    temperature_contract = first.problem.temperature_contract
    hard_constraint_contract = first.problem.hard_constraint_contract
    if (
        not first.model_cache.loaded_once
        or first.model_cache.load_calls != 1
        or first.model_cache.full_generation_authentication_passes != 1
    ):
        raise RuntimeError("dual-stratum receipt requires one shared model load")
    manifest = dict(first.adapter_evidence)
    if not goal_campaign:
        manifest = validate_adapter_manifest(manifest)
    elif (
        manifest.get("schema_version")
        != "mft-goal-20260726-g0-generation-adapter-v1"
        or manifest.get("generation_targets") != list(GOAL_G0_MODEL_TARGETS)
        or manifest.get("required_model_targets") != list(required_targets)
    ):
        raise RuntimeError("goal G0 adapter manifest mismatch")
    expected_repair_stages = {
        "initial_population": True,
        "warm_start": True,
        "every_offspring": True,
        "terminal_physical_replay": True,
    }
    strata: dict[str, Any] = {}
    for turns in supported_turns:
        key = str(turns)
        runner = runners[key]
        if (
            runner.model_cache is not first.model_cache
            or runner.models is not first.models
            or runner.modules is not first.modules
            or runner.authenticated is not first.authenticated
            or runner.problem.fixed_primary_turns != turns
            or runner.problem.stage_spec != stage_spec
            or tuple(runner.problem.constraint_names) != constraint_names
            or not runner.launch_eligible
            or not runner.problem.launch_eligible
        ):
            raise RuntimeError(
                "dual-stratum runners do not share one authenticated load"
            )
        evaluation = evaluations[key]
        objectives = np.asarray(evaluation["F"], dtype=float)
        constraints = np.asarray(evaluation["G"], dtype=float)
        valid = np.asarray(evaluation["decoder_valid"], dtype=bool)
        frame = evaluation["frame"]
        if (
            objectives.shape != (1, 2)
            or constraints.shape != (1, len(constraint_names))
            or valid.tolist() != [True]
            or len(frame) != 1
            or not np.isfinite(objectives).all()
            or not np.isfinite(constraints).all()
        ):
            raise RuntimeError(f"N1={turns} smoke row is not finite and decoded")
        row = _frame_row(frame, 0)
        decoded_controls = {
            name: _finite_number(_row_value(row, name), name)
            for name in (
                "cw1",
                "core_plate_t",
                "wcp_t",
                "wcp_len_pct",
                "core_plate_pad_t",
                "wcp_pad_t",
                "n_core_group",
                "N1_main",
                "N1_side",
                "N2_main",
                "N2_side",
            )
        }
        observed_turns = int(decoded_controls["N1_main"]) + int(
            decoded_controls["N1_side"]
        )
        expected_cw1 = (
            validate_cw1_mm(decoded_controls["cw1"])
            if goal_campaign
            else stage_spec["primary_conductor_thickness_mm"]
        )
        if observed_turns != turns or not math.isclose(
            decoded_controls["cw1"], expected_cw1,
            rel_tol=0.0, abs_tol=1e-12,
        ):
            raise RuntimeError(f"N1={turns} decoded fixed controls escaped")
        budget_identity = winding_budget_identity(
            row,
            expected_cw1_mm=expected_cw1,
        )
        if budget_identity.get("passed") is not True:
            raise RuntimeError(f"N1={turns} winding-budget identity failed")
        model_smoke = dict(model_smoke_by_stratum[key])
        semlock_stress = model_smoke.get("semlock_safe_prediction_stress") or {}
        if not semlock_stress:
            # Direct authenticated callers predating the SemLock gate may
            # supply the ordinary model smoke only.  Execute the same stress
            # here instead of weakening the launch receipt.
            semlock_stress = semlock_safe_prediction_stress(
                runner.models,
                frame.iloc[[0]],
                runner.inference_binding,
                rounds=8,
            )
            model_smoke["semlock_safe_prediction_stress"] = semlock_stress
        semlock_unsigned = {
            name: item
            for name, item in semlock_stress.items()
            if name != "sha256"
        }
        if (
            model_smoke.get("target_count") != len(required_targets)
            or model_smoke.get("all_required_targets_exercised") is not True
            or semlock_stress.get("sha256")
            != canonical_sha256(semlock_unsigned)
            or semlock_stress.get("prediction_call_count")
            != 8 * len(required_targets)
            or semlock_stress.get("sklearn_extratrees_n_jobs") != 1
            or semlock_stress.get("semaphore_entry_growth_count") != 0
            or semlock_stress.get("enospc_observed") is not False
            or semlock_stress.get("stress_passed") is not True
        ):
            raise RuntimeError(f"N1={turns} model inference inventory mismatch")
        model_smoke["sha256"] = canonical_sha256(model_smoke)
        repair_smoke = dict(repair_smoke_by_stratum[key])
        repair_stage_sha = repair_smoke.pop("sha256", None)
        if (
            repair_stage_sha != canonical_sha256(repair_smoke)
            or repair_smoke.get("stages") != expected_repair_stages
            or repair_smoke.get("fixed_primary_turns") != turns
            or repair_smoke.get("same_problem_repair_used_for_all_stages") is not True
        ):
            raise RuntimeError(f"N1={turns} optimizer repair smoke mismatch")
        repair_smoke["sha256"] = repair_stage_sha
        optimizer_repair = {
            "schema_version": OPTIMIZER_REPAIR_SCHEMA,
            **runner.problem.optimizer_repair_contract,
            "stages": expected_repair_stages,
            "stage_evidence": repair_smoke,
            "fixed_primary_turns_repair": True,
            "fixed_primary_turns": turns,
            "launch_eligible": True,
        }
        optimizer_repair["sha256"] = canonical_sha256(optimizer_repair)
        problem = {
            "fixed_primary_turns": turns,
            "primary_winding_budget_identity_attested": True,
            "primary_winding_budget_identity": budget_identity,
            "optimizer_repair_contract_sha256": (
                runner.problem.optimizer_repair_contract["contract_sha256"]
            ),
            "launch_eligible": True,
        }
        problem["sha256"] = canonical_sha256(problem)
        smoke = {
            "decoder_valid": True,
            "decoded_controls": decoded_controls,
            "objectives": {
                "bounding_box_volume_L": float(objectives[0, 0]),
                "predicted_total_loss_W": float(objectives[0, 1]),
            },
            "physical_constraint_G": {
                name: float(constraints[0, index])
                for index, name in enumerate(constraint_names)
            },
            "finite_objectives": True,
            "finite_constraints": True,
            "design_feasibility_required_for_smoke": False,
        }
        smoke["sha256"] = canonical_sha256(smoke)
        stratum = {
            "fixed_primary_turns": turns,
            "problem": problem,
            "optimizer_repair": optimizer_repair,
            "model_smoke": model_smoke,
            "smoke": smoke,
            "launch_eligible": True,
        }
        stratum["sha256"] = canonical_sha256(stratum)
        strata[key] = stratum

    model_load = {
        "required_targets": list(required_targets),
        "required_targets_sha256": required_targets_sha256,
        "loaded_target_count": len(first.models),
        "cache_load_calls": first.model_cache.load_calls,
        "full_generation_authentication_passes": (
            first.model_cache.full_generation_authentication_passes
        ),
        "models_loaded_once_per_process": first.model_cache.loaded_once,
        "generation_copy_performed": False,
        "inference_binding": first.inference_binding,
        "model_smoke_completed": True,
        "supported_fixed_primary_turns": list(supported_turns),
        "strata": {key: strata[key]["model_smoke"] for key in expected_keys},
        "all_supported_strata_exercised": True,
        "semlock_safe_prediction_stress_required": True,
        "sklearn_extratrees_n_jobs": 1,
        "repeated_prediction_stress_rounds_per_stratum": 8,
        "dev_shm_semaphore_growth_allowed": False,
        "enospc_allowed": False,
    }
    problem_contract = {
        "stage_spec": stage_spec,
        "stage_spec_sha256": canonical_sha256(stage_spec),
        "temperature_contract": temperature_contract,
        "temperature_contract_sha256": canonical_sha256(temperature_contract),
        "hard_constraint_contract": hard_constraint_contract,
        "hard_constraint_contract_sha256": canonical_sha256(hard_constraint_contract),
        "constraint_names": list(constraint_names),
        "constraint_count": len(constraint_names),
        "base_constraint_count": len(base_constraint_names),
        "additive_hard_constraint_count": (
            len(constraint_names) - len(base_constraint_names)
        ),
        "base_secondary_vertical_insulation_retained": True,
        "minimum_physical_insulation_is_authoritative_superset": True,
        "variable_cooling_dimensions": list(VARIABLE_COOLING_DIMENSIONS),
        "fixed_cooling_pads_mm": FIXED_COOLING_PADS_MM,
        "simple_base_20mm_plate_clamp_superseded": True,
        "primary_conductor_enforcement": (
            "fixed_cw1_consumed_inside_decoder_winding_budget"
        ),
        "side_temperature_condition_applied": True,
        "q90_additional_multiplier": 1.0,
        "supported_fixed_primary_turns": list(supported_turns),
        "strata": {key: strata[key]["problem"] for key in expected_keys},
    }
    optimizer_repair = {
        "schema_version": OPTIMIZER_REPAIR_SCHEMA,
        "supported_fixed_primary_turns": list(supported_turns),
        "strata": {key: strata[key]["optimizer_repair"] for key in expected_keys},
        "all_supported_strata_passed": True,
        "launch_eligible": True,
    }
    smoke = {
        "coordinate": dict(coordinate_evidence),
        "supported_fixed_primary_turns": list(supported_turns),
        "strata": {key: strata[key]["smoke"] for key in expected_keys},
        "all_supported_strata_smoked": True,
    }
    receipt = {
        "schema_version": (
            GOAL_RECEIPT_SCHEMA if goal_campaign else RECEIPT_SCHEMA
        ),
        "status": (
            (
                "authenticated_goal_four_stratum_g0_smoke_passed_launch_eligible"
                if goal_campaign
                else "authenticated_dual_stratum_model_and_repair_smoke_passed_launch_eligible"
            )
        ),
        "created_at": datetime.now(timezone.utc)
        .astimezone()
        .isoformat(timespec="seconds"),
        "supported_fixed_primary_turns": list(supported_turns),
        "strata": strata,
        "runner": {
            "schema_version": (
                GOAL_RUNNER_SCHEMA if goal_campaign else RUNNER_SCHEMA
            ),
            "problem_schema": PROBLEM_SCHEMA,
            "run_interface": "Current7Tier1Runner.run_one",
            "current_run_nsga2_semantics_source": ("optimization.run_nsga2.run_one"),
            "current_initialization_semantics_source": (
                "optimization.run_nsga2.run_one"
            ),
            "supported_fixed_primary_turns": list(supported_turns),
            "full_nsga_executed": False,
            "launch_eligible": True,
        },
        "adapter_manifest": manifest,
        "adapter_manifest_sha256": canonical_sha256(manifest),
        "code_modules": first.modules.evidence,
        "model_load": model_load,
        "model_load_sha256": canonical_sha256(model_load),
        "problem_contract": problem_contract,
        "problem_contract_sha256": canonical_sha256(problem_contract),
        "optimizer_repair": optimizer_repair,
        "optimizer_repair_sha256": canonical_sha256(optimizer_repair),
        "smoke": smoke,
        "smoke_sha256": canonical_sha256(smoke),
        "portability": hard_constraint_contract["portability"],
        "scheduler_write_performed": False,
        "slurm_submission_performed": False,
        "canonical_pointer_write_performed": False,
        "production_eligible": False,
        "automatic_promotion_allowed": False,
    }
    receipt["payload_sha256"] = canonical_sha256(receipt)
    return receipt


def validate_smoke_receipt(
    value: Any,
    *,
    relocated_source_evidence: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("corrected-generation smoke receipt must be an object")
    expected_payload_sha = value.get("payload_sha256")
    payload = dict(value)
    payload.pop("payload_sha256", None)
    if expected_payload_sha != canonical_sha256(payload):
        raise RuntimeError("corrected-generation smoke receipt payload SHA mismatch")
    runner = value.get("runner") or {}
    loading = value.get("model_load") or {}
    problem = value.get("problem_contract") or {}
    repair = value.get("optimizer_repair") or {}
    smoke = value.get("smoke") or {}
    strata = value.get("strata") or {}
    try:
        stage_spec = validate_stage_spec(problem.get("stage_spec") or {})
    except RuntimeError as exc:
        raise RuntimeError(
            "corrected-generation smoke receipt contract mismatch"
        ) from exc
    expected_constraint_names = stage_constraint_names(stage_spec)
    expected_temperature_contract = stage_temperature_contract(stage_spec)
    expected_hard_contract = stage_hard_constraint_contract(stage_spec)
    goal_campaign = is_goal_stage_spec(stage_spec)
    supported_turns = (
        GOAL_SUPPORTED_PRIMARY_TURNS
        if goal_campaign
        else SUPPORTED_FIXED_PRIMARY_TURNS
    )
    required_targets = (
        GOAL_REQUIRED_MODEL_TARGETS
        if goal_campaign
        else CURRENT_REQUIRED_MODEL_TARGETS
    )
    required_targets_sha256 = (
        GOAL_REQUIRED_MODEL_TARGETS_SHA256
        if goal_campaign
        else CURRENT_REQUIRED_MODEL_TARGETS_SHA256
    )
    expected_turns = list(supported_turns)
    expected_keys = {str(turns) for turns in supported_turns}
    expected_receipt_schema = (
        GOAL_RECEIPT_SCHEMA if goal_campaign else RECEIPT_SCHEMA
    )
    expected_runner_schema = (
        GOAL_RUNNER_SCHEMA if goal_campaign else RUNNER_SCHEMA
    )
    expected_status = (
        "authenticated_goal_four_stratum_g0_smoke_passed_launch_eligible"
        if goal_campaign
        else "authenticated_dual_stratum_model_and_repair_smoke_passed_launch_eligible"
    )
    if (
        value.get("schema_version") != expected_receipt_schema
        or value.get("status") != expected_status
        or value.get("supported_fixed_primary_turns") != expected_turns
        or set(strata) != expected_keys
        or runner.get("schema_version") != expected_runner_schema
        or runner.get("problem_schema") != PROBLEM_SCHEMA
        or runner.get("run_interface") != "Current7Tier1Runner.run_one"
        or runner.get("current_run_nsga2_semantics_source")
        != "optimization.run_nsga2.run_one"
        or runner.get("current_initialization_semantics_source")
        != "optimization.run_nsga2.run_one"
        or runner.get("supported_fixed_primary_turns") != expected_turns
        or runner.get("full_nsga_executed") is not False
        or runner.get("launch_eligible") is not True
        or loading.get("required_targets") != list(required_targets)
        or loading.get("required_targets_sha256")
        != required_targets_sha256
        or loading.get("loaded_target_count") != len(required_targets)
        or loading.get("cache_load_calls") != 1
        or loading.get("full_generation_authentication_passes") != 1
        or loading.get("models_loaded_once_per_process") is not True
        or loading.get("generation_copy_performed") is not False
        or loading.get("model_smoke_completed") is not True
        or loading.get("supported_fixed_primary_turns") != expected_turns
        or set((loading.get("strata") or {})) != expected_keys
        or loading.get("all_supported_strata_exercised") is not True
        or loading.get("semlock_safe_prediction_stress_required") is not True
        or loading.get("sklearn_extratrees_n_jobs") != 1
        or loading.get("repeated_prediction_stress_rounds_per_stratum") != 8
        or loading.get("dev_shm_semaphore_growth_allowed") is not False
        or loading.get("enospc_allowed") is not False
        or problem.get("stage_spec") != stage_spec
        or problem.get("stage_spec_sha256") != canonical_sha256(stage_spec)
        or problem.get("temperature_contract") != expected_temperature_contract
        or problem.get("temperature_contract_sha256")
        != canonical_sha256(expected_temperature_contract)
        or problem.get("hard_constraint_contract") != expected_hard_contract
        or problem.get("hard_constraint_contract_sha256")
        != canonical_sha256(expected_hard_contract)
        or problem.get("constraint_names") != list(expected_constraint_names)
        or problem.get("constraint_count") != len(expected_constraint_names)
        or problem.get("minimum_physical_insulation_is_authoritative_superset")
        is not True
        or problem.get("simple_base_20mm_plate_clamp_superseded") is not True
        or problem.get("primary_conductor_enforcement")
        != "fixed_cw1_consumed_inside_decoder_winding_budget"
        or problem.get("q90_additional_multiplier") != 1.0
        or problem.get("supported_fixed_primary_turns") != expected_turns
        or set((problem.get("strata") or {})) != expected_keys
        or repair.get("schema_version") != OPTIMIZER_REPAIR_SCHEMA
        or repair.get("supported_fixed_primary_turns") != expected_turns
        or set((repair.get("strata") or {})) != expected_keys
        or repair.get("all_supported_strata_passed") is not True
        or repair.get("launch_eligible") is not True
        or smoke.get("supported_fixed_primary_turns") != expected_turns
        or set((smoke.get("strata") or {})) != expected_keys
        or smoke.get("all_supported_strata_smoked") is not True
        or value.get("portability") != expected_hard_contract["portability"]
        or any(
            value.get(field) is not False
            for field in (
                "scheduler_write_performed",
                "slurm_submission_performed",
                "canonical_pointer_write_performed",
                "production_eligible",
                "automatic_promotion_allowed",
            )
        )
    ):
        raise RuntimeError("corrected-generation smoke receipt contract mismatch")
    if goal_campaign:
        adapter_value = value.get("adapter_manifest") or {}
        if (
            adapter_value.get("schema_version")
            != "mft-goal-20260726-g0-generation-adapter-v1"
            or adapter_value.get("generation_targets")
            != list(GOAL_G0_MODEL_TARGETS)
            or adapter_value.get("required_model_targets")
            != list(required_targets)
        ):
            raise RuntimeError("goal G0 adapter manifest contract mismatch")
    else:
        validate_adapter_manifest(
            value.get("adapter_manifest"),
            source_paths_required=not relocated_source_evidence,
        )
    if value.get("adapter_manifest_sha256") != canonical_sha256(
        value["adapter_manifest"]
    ):
        raise RuntimeError("corrected-generation adapter manifest SHA mismatch")
    if value.get("model_load_sha256") != canonical_sha256(loading):
        raise RuntimeError("corrected-generation model-load SHA mismatch")
    if value.get("problem_contract_sha256") != canonical_sha256(problem):
        raise RuntimeError("corrected-generation problem-contract SHA mismatch")
    if value.get("optimizer_repair_sha256") != canonical_sha256(repair):
        raise RuntimeError("corrected-generation optimizer-repair SHA mismatch")
    if value.get("smoke_sha256") != canonical_sha256(smoke):
        raise RuntimeError("corrected-generation smoke SHA mismatch")
    expected_stages = {
        "initial_population": True,
        "warm_start": True,
        "every_offspring": True,
        "terminal_physical_replay": True,
    }
    for turns in supported_turns:
        key = str(turns)
        stratum = strata[key]
        if not isinstance(stratum, dict):
            raise RuntimeError(f"N1={turns} receipt stratum must be an object")
        sealed_stratum = dict(stratum)
        stratum_sha = sealed_stratum.pop("sha256", None)
        stratum_problem = stratum.get("problem") or {}
        stratum_repair = stratum.get("optimizer_repair") or {}
        stratum_model = stratum.get("model_smoke") or {}
        stratum_smoke = stratum.get("smoke") or {}
        contract = stratum_repair.get("contract") or {}
        stage_evidence = stratum_repair.get("stage_evidence") or {}
        semlock_stress = (
            stratum_model.get("semlock_safe_prediction_stress") or {}
        )
        semlock_unsigned = {
            name: item
            for name, item in semlock_stress.items()
            if name != "sha256"
        }
        cw1_contract_valid = (
            (
                contract.get("cw1_search_mm")
                == {
                    "minimum": GOAL_CW1_MIN_MM,
                    "maximum": GOAL_CW1_MAX_MM,
                    "step": GOAL_CW1_STEP_MM,
                    "coordinate_name": "f1_split",
                }
                and contract.get("f1_split_independent_search_allowed")
                is False
                and contract.get("cw1_enforcement")
                == "selected_grid_value_inside_decoder_winding_budget"
                and "fixed_cw1_mm" not in contract
            )
            if goal_campaign
            else (
                contract.get("fixed_cw1_mm") == 5.0
                and contract.get("cw1_enforcement")
                == "inside_decoder_winding_budget"
            )
        )
        if (
            stratum_sha != canonical_sha256(sealed_stratum)
            or stratum.get("fixed_primary_turns") != turns
            or stratum.get("launch_eligible") is not True
            or (problem.get("strata") or {}).get(key) != stratum_problem
            or (repair.get("strata") or {}).get(key) != stratum_repair
            or (loading.get("strata") or {}).get(key) != stratum_model
            or (smoke.get("strata") or {}).get(key) != stratum_smoke
            or stratum_problem.get("fixed_primary_turns") != turns
            or stratum_problem.get("primary_winding_budget_identity_attested")
            is not True
            or (stratum_problem.get("primary_winding_budget_identity") or {}).get(
                "passed"
            )
            is not True
            or stratum_problem.get("launch_eligible") is not True
            or stratum_repair.get("schema_version") != OPTIMIZER_REPAIR_SCHEMA
            or stratum_repair.get("stages") != expected_stages
            or stratum_repair.get("fixed_primary_turns_repair") is not True
            or stratum_repair.get("fixed_primary_turns") != turns
            or stratum_repair.get("launch_eligible") is not True
            or contract.get("schema_version") != OPTIMIZER_REPAIR_SCHEMA
            or contract.get("projection_source_revision")
            != PINNED_PROJECTION_SOURCE_REVISION
            or not cw1_contract_valid
            or contract.get("fixed_primary_turns") != turns
            or contract.get("required_stages")
            != [
                "initial_population",
                "authenticated_warm_start",
                "every_pymoo_offspring",
                "terminal_unscaled_physical_replay",
            ]
            or stratum_repair.get("contract_sha256") != canonical_sha256(contract)
            or stage_evidence.get("stages") != expected_stages
            or stage_evidence.get("fixed_primary_turns") != turns
            or stage_evidence.get("same_problem_repair_used_for_all_stages") is not True
            or stratum_model.get("target_count") != len(required_targets)
            or stratum_model.get("all_required_targets_exercised") is not True
            or set((stratum_model.get("targets") or {}))
            != set(required_targets)
            or stratum_model.get("additional_half_width_multiplier") != 1.0
            or semlock_stress.get("sha256")
            != canonical_sha256(semlock_unsigned)
            or semlock_stress.get("prediction_call_count")
            != 8 * len(required_targets)
            or semlock_stress.get("sklearn_extratrees_n_jobs") != 1
            or semlock_stress.get("semaphore_entry_growth_count") != 0
            or semlock_stress.get("enospc_observed") is not False
            or semlock_stress.get("stress_passed") is not True
            or stratum_smoke.get("decoder_valid") is not True
            or stratum_smoke.get("finite_objectives") is not True
            or stratum_smoke.get("finite_constraints") is not True
            or set((stratum_smoke.get("physical_constraint_G") or {}))
            != set(expected_constraint_names)
        ):
            raise RuntimeError(f"N1={turns} corrected-generation stratum mismatch")
        for label, nested in (
            ("problem", stratum_problem),
            ("optimizer repair", stratum_repair),
            ("model smoke", stratum_model),
            ("physical smoke", stratum_smoke),
        ):
            sealed_nested = dict(nested)
            nested_sha = sealed_nested.pop("sha256", None)
            if nested_sha != canonical_sha256(sealed_nested):
                raise RuntimeError(f"N1={turns} {label} SHA mismatch")
        sealed_stage = dict(stage_evidence)
        stage_sha = sealed_stage.pop("sha256", None)
        if stage_sha != canonical_sha256(sealed_stage):
            raise RuntimeError(f"N1={turns} repair stage SHA mismatch")
    return value


def observed_peak_rss_bytes() -> int:
    """Return a positive process RSS/high-water observation on Linux or Windows."""

    candidates: list[int] = []
    status = Path(f"/proc/{os.getpid()}/status")
    if status.is_file():
        try:
            for line in status.read_text(encoding="ascii").splitlines():
                if line.startswith(("VmHWM:", "VmRSS:")):
                    candidates.append(int(line.split()[1]) * 1024)
        except (OSError, UnicodeError, ValueError, IndexError):
            pass
    try:
        import resource

        usage = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        candidates.append(usage if sys.platform == "darwin" else usage * 1024)
    except (ImportError, OSError, TypeError, ValueError):
        pass
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            ):
                candidates.extend(
                    [
                        int(counters.PeakWorkingSetSize),
                        int(counters.WorkingSetSize),
                    ]
                )
        except (AttributeError, OSError, TypeError, ValueError):
            pass
    observed = max(candidates, default=0)
    if observed <= 0:
        raise RuntimeError("process RSS observation is unavailable")
    return observed


def _profile_number_matches(value: Any, expected: Any, label: str) -> bool:
    if value is None or expected is None:
        return value is None and expected is None
    return math.isclose(
        _finite_number(value, label),
        _finite_number(expected, label),
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def validate_search_thread_contract(
    *,
    profile_inference_threads: Any,
    inference_threads: Any,
    scheduler_cpus: Any,
) -> int:
    """Authenticate an execution cap beneath the sealed eight-thread profile."""

    values = (profile_inference_threads, inference_threads, scheduler_cpus)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise RuntimeError("search CPU/thread contract must contain integers")
    profile_threads = int(profile_inference_threads)
    runtime_threads = int(inference_threads)
    allocated_cpus = int(scheduler_cpus)
    if (
        profile_threads != PRODUCTION_INFERENCE_THREADS
        or not 1 <= runtime_threads <= profile_threads
        or allocated_cpus != runtime_threads
    ):
        raise RuntimeError("search CPU/thread allocation exceeds its sealed profile")
    return runtime_threads


def select_remote_warm_preflight_filter(
    warm_role_preflight: Mapping[str, Any],
    *,
    topology_niche_enabled: bool,
) -> Mapping[str, Any]:
    """Select the authenticated filter evidence for the active warm role.

    Topology-niche warm pools are structural coordinate donors by design and
    therefore do not carry the ordinary hard-feasible filter field.  Keep the
    selection explicit and fail closed on a role/schema mismatch so the remote
    preflight cannot silently attest the wrong warm-pool semantics.
    """

    if topology_niche_enabled:
        expected_schema = "mft-tier1-authenticated-topology-niche-warm-audit-v1"
        filter_key = "structural_geometry_filter"
    else:
        expected_schema = "mft-tier1-authenticated-warm-role-audit-v1"
        filter_key = "standard_hard_geometry_filter"
    if warm_role_preflight.get("schema_version") != expected_schema:
        raise RuntimeError("remote warm preflight role/schema mismatch")
    evidence = warm_role_preflight.get(filter_key)
    if not isinstance(evidence, Mapping) or not evidence:
        raise RuntimeError("remote warm preflight filter evidence is missing")
    return evidence


def run_search_seed(
    *,
    bundle_root: Path,
    relocation_path: Path,
    adapter_receipt_path: Path,
    registry: Path,
    generation: Path,
    dataset: Path,
    profile: Path,
    warm_start: Path,
    warm_contract: Path,
    output: Path,
    remote_preflight: Path,
    bundle_id: str,
    island_id: str,
    island_profile_sha256: str,
    seed: int,
    population: int,
    max_generations: int,
    inference_threads: int,
    scheduler_cpus: int | None = None,
    fixed_primary_turns: int,
    optimizer_termination_strategy: str,
    optimizer_resonance_scale_hz: float,
    optimizer_llt_scale_uh: float,
    optimizer_all_thermal_scale_c: float,
    optimizer_repair_contract_sha256: str,
    stage_spec: Mapping[str, Any],
    stage_spec_sha256: str,
    optimizer_resonance_allowance_hz: float | None = None,
    optimizer_llt_allowance_uh: float | None = None,
) -> Path:
    """Run one authenticated fixed-generation current7 seed inside a bundle."""

    root = bundle_root.resolve(strict=True)
    output_root = output.resolve(strict=True)
    if not output_root.is_dir():
        raise RuntimeError("search output must be an existing directory")
    try:
        output_root.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("search output escaped the immutable bundle") from exc
    authenticated, receipt, relocation, manifest = (
        authenticate_relocated_bundle_generation(
            bundle_root=root,
            relocation_path=relocation_path,
            adapter_receipt_path=adapter_receipt_path,
            registry=registry,
            generation=generation,
            dataset=dataset,
            profile=profile,
        )
    )
    normalized_stage_spec = validate_stage_spec(stage_spec)
    goal_campaign = is_goal_stage_spec(normalized_stage_spec)
    expected_turns = (
        GOAL_SUPPORTED_PRIMARY_TURNS
        if goal_campaign
        else SUPPORTED_FIXED_PRIMARY_TURNS
    )
    expected_required_targets = (
        GOAL_REQUIRED_MODEL_TARGETS
        if goal_campaign
        else CURRENT_REQUIRED_MODEL_TARGETS
    )
    expected_required_targets_sha256 = (
        GOAL_REQUIRED_MODEL_TARGETS_SHA256
        if goal_campaign
        else CURRENT_REQUIRED_MODEL_TARGETS_SHA256
    )
    expected_temperature_targets = (
        GOAL_TEMPERATURE_TARGETS
        if goal_campaign
        else CURRENT_TEMPERATURE_TARGETS
    )
    if (
        canonical_sha256(normalized_stage_spec) != str(stage_spec_sha256)
        or (receipt.get("problem_contract") or {}).get("stage_spec")
        != normalized_stage_spec
        or (receipt.get("problem_contract") or {}).get("stage_spec_sha256")
        != str(stage_spec_sha256)
        or manifest.get("hard_spec") != normalized_stage_spec
        or manifest.get("hard_spec_sha256") != str(stage_spec_sha256)
    ):
        raise RuntimeError("staged hard-spec manifest/receipt identity mismatch")
    execution = manifest.get("search_execution") or {}
    remote_path = remote_preflight.resolve()
    result_path = output_root / str(execution.get("result_filename") or "")
    if (
        manifest.get("bundle_id") != str(bundle_id)
        or execution.get("remote_preflight_schema_version") != REMOTE_PREFLIGHT_SCHEMA
        or execution.get("result_schema_version") != SEARCH_RESULT_SCHEMA
        or execution.get("optimizer_processes_per_task") != 1
        or execution.get("model_mapping_instances_per_process") != 1
        or remote_path
        != output_root / str(execution.get("remote_preflight_filename") or "")
        or result_path.parent != output_root
        or not result_path.name
        or remote_path.exists()
        or result_path.exists()
    ):
        raise RuntimeError("bundle search interface identity mismatch")
    islands = manifest.get("islands") or {}
    island = islands.get(str(island_id)) or {}
    island_profile = island.get("current7_profile") or {}
    unsigned_profile = dict(island_profile)
    recorded_profile_sha = unsigned_profile.pop("sha256", None)
    profile_topology = island_profile.get("topology_evolution_contract") or {}
    topology_niche_enabled = (
        profile_topology.get("final1000_topology_niche_contract") is not None
    )
    topology_contract = (
        goal_topology_contract(int(fixed_primary_turns))
        if goal_campaign
        else (
            deep_topology_contract(
                int(fixed_primary_turns),
                enable_final1000_topology_niche=True,
            )
            if topology_niche_enabled
            else deep_topology_contract(int(fixed_primary_turns))
        )
    )
    expected_allowance_resonance = island_profile.get(
        "optimizer_resonance_allowance_Hz"
    )
    expected_allowance_llt = island_profile.get("optimizer_llt_allowance_uH")
    runtime_threads = validate_search_thread_contract(
        profile_inference_threads=island_profile.get("inference_threads"),
        inference_threads=inference_threads,
        scheduler_cpus=(
            inference_threads if scheduler_cpus is None else scheduler_cpus
        ),
    )
    if (
        int(fixed_primary_turns) not in expected_turns
        or recorded_profile_sha != canonical_sha256(unsigned_profile)
        or recorded_profile_sha != str(island_profile_sha256)
        or island.get("current7_profile_sha256") != recorded_profile_sha
        or island_profile.get("island_id") != str(island_id)
        or island_profile.get("fixed_primary_turns") != int(fixed_primary_turns)
        or island_profile.get("population") != PRODUCTION_POPULATION
        or int(population) != PRODUCTION_POPULATION
        or island_profile.get("fixed_generations") != PRODUCTION_FIXED_GENERATIONS
        or int(max_generations) != PRODUCTION_FIXED_GENERATIONS
        or island_profile.get("inference_threads") != PRODUCTION_INFERENCE_THREADS
        or int(inference_threads) != runtime_threads
        or island_profile.get("optimizer_termination_strategy")
        != FIXED_GENERATION_TERMINATION_STRATEGY
        or optimizer_termination_strategy != FIXED_GENERATION_TERMINATION_STRATEGY
        or island_profile.get("topology_evolution_contract") != topology_contract
        or island_profile.get("temperature_targets")
        != list(expected_temperature_targets)
        or island_profile.get("offspring_physics_repair_required") is not True
        or island_profile.get("terminal_physical_replay_required") is not True
        or island_profile.get("physical_hard_spec_mutation") is not False
        or island_profile.get("objective_mutation") is not False
        or island_profile.get("automatic_promotion_allowed") is not False
        or not _profile_number_matches(
            optimizer_resonance_scale_hz,
            island_profile.get("optimizer_resonance_scale_Hz"),
            "optimizer resonance scale",
        )
        or not _profile_number_matches(
            optimizer_llt_scale_uh,
            island_profile.get("optimizer_llt_scale_uH"),
            "optimizer Llt scale",
        )
        or not _profile_number_matches(
            optimizer_all_thermal_scale_c,
            island_profile.get("optimizer_all_current7_thermal_scale_C"),
            "optimizer thermal scale",
        )
        or not _profile_number_matches(
            optimizer_resonance_allowance_hz,
            expected_allowance_resonance,
            "optimizer resonance allowance",
        )
        or not _profile_number_matches(
            optimizer_llt_allowance_uh,
            expected_allowance_llt,
            "optimizer Llt allowance",
        )
        or not (
            int(island_profile.get("seed_start"))
            <= int(seed)
            < int(island_profile.get("seed_window_end_exclusive"))
        )
    ):
        raise RuntimeError("deep-crossover island profile mismatch")
    adapter_evidence = receipt["adapter_manifest"]
    if manifest.get("bundle_code_revision") != (adapter_evidence.get("code") or {}).get(
        "revision"
    ) or receipt.get("adapter_manifest_sha256") != canonical_sha256(adapter_evidence):
        raise RuntimeError("bundle/adapter code identity mismatch")
    warm = island.get("warm") or {}
    warm_artifact_record = warm.get("artifact") or {}
    warm_contract_record = warm.get("contract") or {}
    expected_warm_path = _contained_input(
        root,
        root / Path(str(warm_artifact_record.get("path") or "")),
        "island warm artifact",
    )
    expected_warm_contract_path = _contained_input(
        root,
        root / Path(str(warm_contract_record.get("path") or "")),
        "island warm contract",
    )
    supplied_warm_path = _contained_input(root, warm_start, "warm artifact")
    supplied_warm_contract_path = _contained_input(root, warm_contract, "warm contract")
    if (
        expected_warm_path != supplied_warm_path
        or expected_warm_contract_path != supplied_warm_contract_path
        or warm_artifact_record.get("sha256") != sha256_file(supplied_warm_path)
        or warm_contract_record.get("sha256")
        != sha256_file(supplied_warm_contract_path)
        or warm_artifact_record.get("size") != supplied_warm_path.stat().st_size
        or warm_contract_record.get("size")
        != supplied_warm_contract_path.stat().st_size
    ):
        raise RuntimeError("island warm artifact identity mismatch")
    code_root = _contained_input(
        root, root / "artifacts" / "code", "relocated code root"
    )
    semlock_safe_release = phase_b_semlock_safe_release_identity(
        code_root=code_root,
        manifest=manifest,
    )
    runner = build_relocated_authenticated_runner(
        authenticated=authenticated,
        receipt=receipt,
        code_root=code_root,
        fixed_primary_turns=int(fixed_primary_turns),
        inference_threads=int(inference_threads),
        stage_spec=normalized_stage_spec,
    )
    stratum_repair = receipt["strata"][str(int(fixed_primary_turns))][
        "optimizer_repair"
    ]
    problem_repair_sha = runner.problem.optimizer_repair_contract["contract_sha256"]
    if (
        stratum_repair.get("contract_sha256") != problem_repair_sha
        or str(optimizer_repair_contract_sha256) != problem_repair_sha
        or not runner.model_cache.loaded_once
        or runner.model_cache.load_calls != 1
        or runner.model_cache.full_generation_authentication_passes != 1
        or tuple(runner.models) != expected_required_targets
        or runner.inference_binding.get("target_count")
        != len(expected_required_targets)
        or runner.inference_binding.get("threads_per_model") != int(inference_threads)
    ):
        raise RuntimeError("remote model-load/repair identity mismatch")
    warm_values, warm_handoff = authenticate_warm_handoff(
        supplied_warm_path,
        supplied_warm_contract_path,
        fixed_primary_turns=int(fixed_primary_turns),
        n_var=runner.problem.n_var,
        expected_contract_file_sha256=warm_contract_record["sha256"],
    )
    filtered_warm, structural_donors, warm_role_preflight = (
        runner.prepare_authenticated_warm_start(
            warm_values,
            role_partition=warm_handoff.get("warm_role_partition"),
            stage="remote_preflight_authenticated_warm_start",
            topology_niche_partition=warm_handoff.get(
                "topology_niche_partition"
            ),
        )
    )
    if len(filtered_warm) < 1:
        raise RuntimeError("remote warm preflight has no repaired hard-feasible row")
    if (
        warm_handoff.get("warm_role_partition") is not None
        and len(structural_donors) < 1
    ):
        raise RuntimeError("remote warm preflight has no structural basin donor")
    warm_preflight_filter = select_remote_warm_preflight_filter(
        warm_role_preflight,
        topology_niche_enabled=topology_niche_enabled,
    )
    stress_evaluation = runner.evaluate_coordinates(filtered_warm[:1])
    if not bool(stress_evaluation["decoder_valid"][0]):
        raise RuntimeError("remote SemLock stress coordinate did not decode")
    remote_semlock_stress = semlock_safe_prediction_stress(
        runner.models,
        stress_evaluation["frame"].iloc[[0]],
        runner.inference_binding,
        rounds=8,
    )
    if (
        (warm_handoff.get("topology_niche_partition") is not None)
        is not topology_niche_enabled
    ):
        raise RuntimeError("bundle profile/warm topology niche activation mismatch")
    semaphore_stress = None
    if semlock_safe_release is not None:
        stress_frame, _stress_shrink, stress_valid = runner.problem.decode_batch(
            filtered_warm[:1]
        )
        if len(stress_frame) != 1 or not bool(stress_valid[0]):
            raise RuntimeError("remote repeated-predict stress has no valid sample")
        semaphore_stress = attest_semlock_free_repeated_predict(
            runner.models,
            stress_frame,
        )
    physical_evaluate, optimizer_scaling = install_optimizer_scaling(
        runner.problem,
        resonance_scale_hz=optimizer_resonance_scale_hz,
        llt_scale_uh=optimizer_llt_scale_uh,
        all_thermal_scale_c=optimizer_all_thermal_scale_c,
        resonance_allowance_hz=optimizer_resonance_allowance_hz,
        llt_allowance_uh=optimizer_llt_allowance_uh,
        topology_allowance_contract=topology_contract.get(
            "final1000_topology_niche_contract"
        ),
    )
    runner.physical_evaluate = physical_evaluate
    runner.optimizer_scaling = optimizer_scaling
    repair_installation = runner.install_offspring_repair_operator(
        topology_contract=topology_contract
    )
    maximum_rss = int((manifest.get("fast_ramp") or {}).get(
        "maximum_peak_rss_bytes", 0
    ))
    if maximum_rss <= 0:
        raise RuntimeError("bundle maximum RSS contract is invalid")
    callback_state: dict[str, Any] = {"count": 0, "preflight": None}

    def seal_remote_preflight(pre_optimization: Mapping[str, Any]) -> None:
        if callback_state["count"] != 0 or remote_path.exists():
            raise RuntimeError("remote preflight callback executed more than once")
        operator = pre_optimization.get("offspring_repair_operator") or {}
        initial = pre_optimization.get("initial_population") or {}
        warm_execution = pre_optimization.get("authenticated_warm_start") or {}
        observed_rss = observed_peak_rss_bytes()
        if (
            operator.get("call_count") != 0
            or operator.get("row_count") != 0
            or pre_optimization.get("offspring_repair_operator_installed") is not True
            or pre_optimization.get("offspring_repair_operator_executed") is not False
            or not initial
            or not warm_execution
            or observed_rss > maximum_rss
        ):
            raise RuntimeError("remote pre-optimization execution gate failed")
        value = {
            "schema_version": REMOTE_PREFLIGHT_SCHEMA,
            "status": "passed",
            "bundle_id": str(bundle_id),
            "seed": int(seed),
            "island_id": str(island_id),
            "fixed_primary_turns": int(fixed_primary_turns),
            "optimizer_pid": os.getpid(),
            "optimizer_processes": 1,
            "model_mapping_instances": 1,
            "full_generation_authentication_passes": 1,
            "authenticated_artifact_count": int(adapter_evidence["artifact_count"]),
            "generation_artifact_inventory_sha256": manifest[
                "generation_artifact_inventory_sha256"
            ],
            "adapter_manifest_sha256": receipt["adapter_manifest_sha256"],
            "train_report_sha256": adapter_evidence["train_report"]["sha256"],
            "dataset_sha256": adapter_evidence["dataset"]["sha256"],
            "profile_canonical_sha256": adapter_evidence["profile"]["canonical_sha256"],
            "temperature_contract_sha256": (runner.problem.temperature_contract_sha256),
            "hard_constraint_contract_sha256": (
                runner.problem.hard_constraint_contract_sha256
            ),
            "stage_spec_sha256": runner.problem.stage_spec_sha256,
            "island_profile_sha256": recorded_profile_sha,
            "warm_artifact_sha256": warm_artifact_record["sha256"],
            "warm_contract_sha256": warm_contract_record["sha256"],
            "loaded_model_count": len(runner.models),
            "loaded_model_targets_sha256": expected_required_targets_sha256,
            "temperature_targets": list(expected_temperature_targets),
            "inference_threads": int(inference_threads),
            **(
                {
                    "inference_binding": dict(runner.inference_binding),
                    "semlock_free_repeated_predict": semaphore_stress,
                    "semlock_safe_inference_release": semlock_safe_release,
                }
                if semlock_safe_release is not None
                else {}
            ),
            "scheduler_cpus": int(
                inference_threads if scheduler_cpus is None else scheduler_cpus
            ),
            "observed_peak_rss_bytes": observed_rss,
            "maximum_peak_rss_bytes": maximum_rss,
            "optimizer_repair_contract_sha256": problem_repair_sha,
            "optimizer_scaling_contract_sha256": optimizer_scaling["sha256"],
            "semlock_safe_prediction_stress": remote_semlock_stress,
            "topology_evolution_contract_sha256": topology_contract["sha256"],
            "topology_niche_contract_sha256": (
                None
                if topology_contract.get("final1000_topology_niche_contract")
                is None
                else topology_contract["final1000_topology_niche_contract"][
                    "sha256"
                ]
            ),
            "offspring_physics_repair": True,
            "initial_repair_attested": True,
            "warm_repair_attested": True,
            "every_offspring_decode_repair_attested": False,
            "offspring_repair_operator_installed": True,
            "offspring_repair_operator_contract_sha256": problem_repair_sha,
            "offspring_repair_execution_status": ("deferred_until_optimizer_execution"),
            "terminal_physical_replay_required": True,
            "initial_population_repair_evidence": initial,
            "warm_handoff_authentication": warm_handoff,
            "warm_preflight_repair": warm_role_preflight["repair"],
            "warm_preflight_filter": warm_preflight_filter,
            "warm_role_preflight": warm_role_preflight,
            "pre_optimization_evidence": dict(pre_optimization),
            "repair_installation": repair_installation,
            "legacy_feedback_wrapper_used": False,
            "local_adapter_authentication_replayed": False,
            "production_eligible": False,
            "fea_submission_approved": False,
            "fea_submission_performed": False,
            "aedt_used": False,
            "automatic_promotion_allowed": False,
        }
        value["payload_sha256"] = canonical_sha256(value)
        _atomic_json(remote_path, value)
        callback_state.update(count=1, preflight=value)

    result = runner.run_one(
        seed=int(seed),
        population=int(population),
        max_generations=int(max_generations),
        warm_start_path=supplied_warm_path,
        warm_start_sha256=warm_artifact_record["sha256"],
        warm_start_role_partition=warm_handoff.get("warm_role_partition"),
        warm_start_niche_partition=warm_handoff.get(
            "topology_niche_partition"
        ),
        optimizer_termination_strategy=optimizer_termination_strategy,
        pre_optimization_callback=seal_remote_preflight,
    )
    if callback_state["count"] != 1 or not remote_path.is_file():
        raise RuntimeError("optimizer did not publish its remote preflight")
    repair_audit = result.tier1_repair_audit
    topology_audit = result.tier1_topology_evolution_audit
    operator_audit = repair_audit.get("pymoo_operator") or {}
    terminal_replay_audit = repair_audit.get("terminal_physical_replay") or {}
    deterministic_terminal = (
        terminal_replay_audit.get("deterministic_terminal_inference") or {}
    )
    optimizer_snapshot = (
        terminal_replay_audit.get("multithread_optimizer_snapshot") or {}
    )
    if (
        repair_audit.get("stages")
        != {
            "initial_population": True,
            "warm_start": True,
            "every_offspring": True,
            "terminal_physical_replay": True,
        }
        or not isinstance(operator_audit.get("call_count"), int)
        or operator_audit["call_count"] < int(max_generations)
        or not isinstance(operator_audit.get("row_count"), int)
        or operator_audit["row_count"] < 1
        or terminal_replay_audit.get("optimizer_physical_G_match") is not True
        or terminal_replay_audit.get("terminal_population_canonicalized") is not True
        or deterministic_terminal.get("optimizer_inference_threads") != runtime_threads
        or deterministic_terminal.get("terminal_replay_inference_threads")
        != DETERMINISTIC_TERMINAL_INFERENCE_THREADS
        or deterministic_terminal.get("optimizer_binding_restored") is not True
        or deterministic_terminal.get("objectives_bit_exact") is not True
        or deterministic_terminal.get("optimizer_G_bit_exact") is not True
        or deterministic_terminal.get("numeric_tolerance_relaxed") is not False
        or deterministic_terminal.get("terminal_model_predictions_serially_replayed")
        is not True
        or optimizer_snapshot.get("canonicalized_from_serial_authority") is not True
        or optimizer_snapshot.get("stale_constraint_violation_cache_cleared")
        is not True
        or not isinstance(
            optimizer_snapshot.get("canonical_constraint_violation_sha256"), str
        )
        or len(optimizer_snapshot["canonical_constraint_violation_sha256"]) != 64
        or topology_audit.get("terminal_epsilon_zero") is not True
        or topology_audit.get("all_required_topologies_preserved") is not True
        or int(result.tier1_evaluated_generations) != int(max_generations)
        or int(result.tier1_completed_generations) != int(max_generations) + 1
    ):
        raise RuntimeError("terminal optimizer repair/topology audit failed")
    authenticated_report = getattr(authenticated, "report", {})
    authenticated_artifacts = (
        authenticated_report.get("artifacts")
        if isinstance(authenticated_report, Mapping)
        else None
    )
    model_artifacts_sha256 = (
        canonical_sha256(authenticated_artifacts)
        if isinstance(authenticated_artifacts, Mapping)
        and authenticated_artifacts
        else str(manifest["generation_artifact_inventory_sha256"])
    )
    persisted = persist_search_outputs(
        runner,
        result,
        output_root,
        source_identity={
            "seed": int(seed),
            "task_id": str(
                os.environ.get("SLURM_SCHED_TASK_ID")
                or f"pid-{os.getpid()}"
            ),
            "bundle_id": str(bundle_id),
            "island_id": str(island_id),
            "dataset_sha256": adapter_evidence["dataset"]["sha256"],
            "model_artifacts_sha256": model_artifacts_sha256,
            "model_generation_sha256": adapter_evidence["train_report"][
                "sha256"
            ],
            "evaluation_spec_sha256": runner.problem.stage_spec_sha256,
            "temperature_contract_sha256": (
                runner.problem.temperature_contract_sha256
            ),
            "hard_constraint_contract_sha256": (
                runner.problem.hard_constraint_contract_sha256
            ),
        },
    )
    terminal_turn_values = persisted["terminal_population_primary_turn_values"]
    if terminal_turn_values != [int(fixed_primary_turns)]:
        raise RuntimeError("terminal population escaped fixed primary turns")
    result_value = {
        "schema_version": SEARCH_RESULT_SCHEMA,
        "bundle_id": str(bundle_id),
        "seed": int(seed),
        "island_id": str(island_id),
        "population": int(population),
        "max_generations": int(max_generations),
        "evaluated_generations": int(result.tier1_evaluated_generations),
        "completed_generations": int(result.tier1_completed_generations),
        "inference_threads": int(inference_threads),
        "scheduler_cpus": int(
            inference_threads if scheduler_cpus is None else scheduler_cpus
        ),
        "optimizer_pid": os.getpid(),
        "optimizer_processes": 1,
        "generation_artifact_inventory_sha256": manifest[
            "generation_artifact_inventory_sha256"
        ],
        "adapter_manifest_sha256": receipt["adapter_manifest_sha256"],
        "train_report_sha256": adapter_evidence["train_report"]["sha256"],
        "dataset_sha256": adapter_evidence["dataset"]["sha256"],
        "profile_canonical_sha256": adapter_evidence["profile"]["canonical_sha256"],
        "temperature_contract_sha256": (runner.problem.temperature_contract_sha256),
        "hard_constraint_contract_sha256": (
            runner.problem.hard_constraint_contract_sha256
        ),
        "hard_spec": runner.problem.stage_spec,
        "stage_spec_sha256": runner.problem.stage_spec_sha256,
        "constraint_version": runner.problem.hard_constraint_contract["stage"],
        "island_profile_sha256": recorded_profile_sha,
        "warm_artifact_sha256": warm_artifact_record["sha256"],
        "warm_contract_sha256": warm_contract_record["sha256"],
        "relocation_contract_sha256": canonical_sha256(relocation),
        "loaded_model_count": len(runner.models),
        "loaded_model_targets_sha256": expected_required_targets_sha256,
        "temperature_targets": list(expected_temperature_targets),
        "constraint_names": list(runner.problem.constraint_names),
        "fixed_primary_turns": int(fixed_primary_turns),
        "terminal_population_primary_turn_values": terminal_turn_values,
        "terminal_population_fixed_primary_turns_verified": True,
        "offspring_physics_repair": True,
        "initial_population_repair_attested": True,
        "warm_start_repair_attested": True,
        "every_offspring_decode_repair_attested": True,
        "terminal_physical_replay_attested": True,
        "optimizer_repair_contract_sha256": problem_repair_sha,
        "optimizer_scaling_contract": optimizer_scaling,
        "optimizer_repair_audit": repair_audit,
        "offspring_repair_operator_call_count": operator_audit["call_count"],
        "offspring_repair_operator_row_count": operator_audit["row_count"],
        "terminal_physical_replay_evidence": terminal_replay_audit,
        "optimizer_topology_evolution_contract": (
            result.tier1_topology_evolution_contract
        ),
        "topology_evolution_contract_sha256": topology_contract["sha256"],
        "topology_niche_contract_sha256": (
            None
            if topology_contract.get("final1000_topology_niche_contract") is None
            else topology_contract["final1000_topology_niche_contract"]["sha256"]
        ),
        "semlock_safe_prediction_stress": remote_semlock_stress,
        "semlock_safe_prediction_stress_sha256": remote_semlock_stress[
            "sha256"
        ],
        "semlock_safe_prediction_stress_attested": True,
        "optimizer_topology_evolution_audit": topology_audit,
        "terminal_population_count": persisted["terminal_population_count"],
        "physical_feasible_count": persisted["physical_feasible_count"],
        "feasible_pareto_count": persisted["feasible_pareto_count"],
        "least_violation_count": persisted["least_violation_count"],
        "artifact_inventory": persisted["artifact_inventory"],
        "artifact_inventory_sha256": persisted["artifact_inventory_sha256"],
        "infeasibility_report": persisted["infeasibility_report"],
        "terminal_physical_candidates_manifest": persisted[
            "terminal_physical_candidates_manifest"
        ],
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    result_value["payload_sha256"] = canonical_sha256(result_value)
    _atomic_json(result_path, result_value)
    written = read_json(result_path)
    sealed = dict(written)
    expected_sha = sealed.pop("payload_sha256", None)
    if (
        expected_sha != canonical_sha256(sealed)
        or written.get("schema_version") != SEARCH_RESULT_SCHEMA
        or written.get("terminal_population_fixed_primary_turns_verified") is not True
        or written.get("every_offspring_decode_repair_attested") is not True
        or written.get("terminal_physical_replay_attested") is not True
    ):
        raise RuntimeError("written search result seal failed")
    return result_path


def run_smoke_preflight(
    *,
    generation: Path,
    candidate_path: Path,
    quality_path: Path,
    code_root: Path,
    expected_code_revision: str,
    coordinate_path: Path,
    coordinate_sha256: str,
    warm_start_paths: Mapping[int, Path],
    warm_start_sha256: Mapping[int, str],
    output: Path,
    warm_start_contract_paths: Mapping[int, Path] | None = None,
    warm_start_contract_sha256: Mapping[int, str] | None = None,
    stage_spec: Mapping[str, Any] | None = None,
    inference_threads: int = 1,
) -> Path:
    import numpy as np

    output_root = output.resolve()
    if output_root.exists():
        raise RuntimeError("corrected-generation preflight output must not exist")
    if not output_root.parent.is_dir():
        raise RuntimeError("corrected-generation preflight output parent is missing")

    normalized_stage_spec = validate_stage_spec(stage_spec or CURRENT_STAGE_SPEC)
    supported_turns = (
        GOAL_SUPPORTED_PRIMARY_TURNS
        if is_goal_stage_spec(normalized_stage_spec)
        else SUPPORTED_FIXED_PRIMARY_TURNS
    )
    first_runner = build_authenticated_runner(
        generation=generation,
        candidate_path=candidate_path,
        quality_path=quality_path,
        code_root=code_root,
        expected_code_revision=expected_code_revision,
        fixed_primary_turns=supported_turns[0],
        stage_spec=normalized_stage_spec,
        inference_threads=inference_threads,
    )
    runners = {
        str(turns): (
            first_runner
            if turns == supported_turns[0]
            else runner_for_fixed_primary_turns(first_runner, turns)
        )
        for turns in supported_turns
    }
    if set(warm_start_paths) != set(supported_turns) or set(
        warm_start_sha256
    ) != set(supported_turns):
        raise RuntimeError(
            "warm-start inventory must cover every campaign N1 stratum"
        )
    contract_paths = dict(warm_start_contract_paths or {})
    contract_sha256 = dict(warm_start_contract_sha256 or {})
    if (
        set(contract_paths) != set(contract_sha256)
        or not set(contract_paths) <= set(supported_turns)
    ):
        raise RuntimeError("smoke warm-contract path/SHA inventory mismatch")
    for protected in (
        first_runner.authenticated.generation,
        first_runner.authenticated.registry,
        Path(first_runner.code_identity["path"]),
        coordinate_path.resolve(strict=True),
        *(warm_start_paths[turns].resolve(strict=True) for turns in supported_turns),
        *(contract_paths[turns].resolve(strict=True) for turns in contract_paths),
    ):
        if _path_is_below(output_root, protected) or _path_is_below(
            protected, output_root
        ):
            raise RuntimeError("preflight output overlaps authenticated input")

    raw_coordinate, coordinate_evidence = load_smoke_coordinate(
        coordinate_path,
        coordinate_sha256,
        n_var=first_runner.problem.n_var,
    )
    evaluations: dict[str, Any] = {}
    model_smoke_by_stratum: dict[str, Any] = {}
    repair_smoke_by_stratum: dict[str, Any] = {}
    for turns in supported_turns:
        key = str(turns)
        runner = runners[key]
        initial, initial_repair = runner.repair_coordinates(
            raw_coordinate, stage="initial_population"
        )
        if turns in contract_paths:
            warm_raw, warm_authentication = authenticate_warm_handoff(
                warm_start_paths[turns],
                contract_paths[turns],
                fixed_primary_turns=turns,
                n_var=runner.problem.n_var,
                expected_contract_file_sha256=contract_sha256[turns],
            )
            if (
                (warm_authentication.get("artifact") or {}).get("sha256")
                != str(warm_start_sha256[turns]).strip().lower()
            ):
                raise RuntimeError(
                    f"N1={turns} smoke warm artifact SHA disagrees with contract"
                )
            filtered_warm, structural_donors, warm_role_audit = (
                runner.prepare_authenticated_warm_start(
                    warm_raw,
                    role_partition=warm_authentication.get(
                        "warm_role_partition"
                    ),
                    topology_niche_partition=warm_authentication.get(
                        "topology_niche_partition"
                    ),
                    stage="authenticated_warm_start",
                )
            )
            niche_partition = warm_authentication.get(
                "topology_niche_partition"
            )
            warm_repair = warm_role_audit["repair"]
            if niche_partition is not None:
                # A topology-niche pool is deliberately a 160-row structural
                # coordinate-donor handoff.  Its source may contain zero rows
                # satisfying the final box/thermal search gates.  Authenticate
                # its exact semantic partition and require all rows to survive
                # the same decoder/manufacturing/insulation repair as the
                # production runner; physical G is still evaluated unchanged.
                if len(filtered_warm) != 160 or len(structural_donors) != 0:
                    raise RuntimeError(
                        f"N1={turns} topology niche smoke did not retain exact 160 donors"
                    )
                warm_filter = warm_role_audit["structural_geometry_filter"]
                warm_filter_semantics = "structural_coordinate_donors"
            else:
                if len(filtered_warm) < 1:
                    raise RuntimeError(
                        f"N1={turns} warm pool has no repaired hard-feasible row"
                    )
                warm_filter = warm_role_audit[
                    "standard_hard_geometry_filter"
                ]
                warm_filter_semantics = "hard_geometry_feasible"
        else:
            warm_raw, warm_authentication = load_authenticated_warm_start(
                warm_start_paths[turns],
                warm_start_sha256[turns],
                n_var=runner.problem.n_var,
            )
            warm, warm_repair = runner.repair_coordinates(
                warm_raw, stage="authenticated_warm_start"
            )
            filtered_warm, warm_filter = (
                runner.problem.filter_warm_start_coordinates(warm)
            )
            if len(filtered_warm) < 1:
                raise RuntimeError(
                    f"N1={turns} warm pool has no repaired hard-feasible row"
                )
            warm_role_audit = None
            warm_filter_semantics = "hard_geometry_feasible"
        mutation = np.mod(
            raw_coordinate
            + np.linspace(0.013, 0.211, runner.problem.n_var).reshape(1, -1),
            1.0,
        )
        pymoo_repair = create_pymoo_physics_repair(runner.problem)
        repaired_offspring = pymoo_repair._do(runner.problem, mutation)
        repaired_offspring, offspring_repair = runner.repair_coordinates(
            repaired_offspring, stage="every_pymoo_offspring"
        )
        if pymoo_repair.call_count != 1 or pymoo_repair.row_count != 1:
            raise RuntimeError(
                f"N1={turns} pymoo offspring repair smoke did not execute once"
            )
        evaluation = runner.evaluate_coordinates(initial)
        if not bool(evaluation["decoder_valid"][0]):
            raise RuntimeError(f"N1={turns} smoke coordinate did not decode")
        _terminal, terminal_replay = runner.terminal_physical_replay(
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
            "initial_population": initial_repair,
            "authenticated_warm_start": {
                "source_authentication": warm_authentication,
                "coordinate_repair": warm_repair,
                "hard_geometry_filter": warm_filter,
                "filter_semantics": warm_filter_semantics,
                "role_audit": warm_role_audit,
                "filtered_coordinate_sha256": canonical_sha256(
                    np.asarray(filtered_warm, dtype=float).tolist()
                ),
            },
            "every_offspring": {
                "coordinate_repair": offspring_repair,
                "pymoo_operator": pymoo_repair.evidence(),
                "output_sha256": canonical_sha256(
                    np.asarray(repaired_offspring, dtype=float).tolist()
                ),
            },
            "terminal_physical_replay": terminal_replay,
            "same_problem_repair_used_for_all_stages": True,
        }
        repair_smoke["sha256"] = canonical_sha256(repair_smoke)
        evaluations[key] = evaluation
        smoke_frame = evaluation["frame"].iloc[[0]]
        model_smoke = _smoke_every_model(runner.models, smoke_frame)
        model_smoke["semlock_safe_prediction_stress"] = (
            semlock_safe_prediction_stress(
                runner.models,
                smoke_frame,
                runner.inference_binding,
                rounds=8,
            )
        )
        model_smoke_by_stratum[key] = model_smoke
        repair_smoke_by_stratum[key] = repair_smoke
    receipt = build_smoke_receipt(
        runners=runners,
        coordinate_evidence=coordinate_evidence,
        evaluations=evaluations,
        model_smoke_by_stratum=model_smoke_by_stratum,
        repair_smoke_by_stratum=repair_smoke_by_stratum,
    )
    validate_smoke_receipt(receipt)

    output_root.mkdir()
    receipt_path = output_root / "authentication_receipt.json"
    _atomic_json(receipt_path, receipt)
    validate_smoke_receipt(json.loads(receipt_path.read_text(encoding="utf-8")))
    return receipt_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Authenticate and smoke one corrected current7 generation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--generation", type=Path, required=True)
    smoke.add_argument("--candidate", type=Path, required=True)
    smoke.add_argument("--quality-status", type=Path, required=True)
    smoke.add_argument("--code-root", type=Path, required=True)
    smoke.add_argument("--expected-code-revision", required=True)
    smoke.add_argument("--coordinate", type=Path, required=True)
    smoke.add_argument("--coordinate-sha256", required=True)
    smoke.add_argument("--warm-start-n1-5", type=Path, required=True)
    smoke.add_argument("--warm-start-n1-5-sha256", required=True)
    smoke.add_argument("--warm-start-n1-6", type=Path, required=True)
    smoke.add_argument("--warm-start-n1-6-sha256", required=True)
    smoke.add_argument("--warm-start-n1-6-contract", type=Path)
    smoke.add_argument("--warm-start-n1-6-contract-sha256")
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--inference-threads", type=int, default=1)
    smoke.add_argument("--stage-spec-json")
    smoke.add_argument("--stage-spec-sha256")
    search = subparsers.add_parser("search-seed")
    search.add_argument("--bundle-root", type=Path, required=True)
    search.add_argument("--relocation", type=Path, required=True)
    search.add_argument("--adapter-receipt", type=Path, required=True)
    search.add_argument("--registry", type=Path, required=True)
    search.add_argument("--generation", type=Path, required=True)
    search.add_argument("--dataset", type=Path, required=True)
    search.add_argument("--profile", type=Path, required=True)
    search.add_argument("--warm-start", type=Path, required=True)
    search.add_argument("--warm-contract", type=Path, required=True)
    search.add_argument("--output", type=Path, required=True)
    search.add_argument("--remote-preflight", type=Path, required=True)
    search.add_argument("--bundle-id", required=True)
    search.add_argument("--island-id", required=True)
    search.add_argument("--island-profile-sha256", required=True)
    search.add_argument("--seed", type=int, required=True)
    search.add_argument("--population", type=int, required=True)
    search.add_argument("--max-generations", type=int, required=True)
    search.add_argument("--inference-threads", type=int, required=True)
    # Backward-compatible for authenticated direct callers; the production
    # Slurm runner supplies this explicitly and run_search_seed fail-closes to
    # the same value as --inference-threads when it is omitted.
    search.add_argument("--scheduler-cpus", type=int)
    search.add_argument("--fixed-primary-turns", type=int, required=True)
    search.add_argument("--optimizer-termination-strategy", required=True)
    search.add_argument("--optimizer-resonance-scale-hz", type=float, required=True)
    search.add_argument("--optimizer-llt-scale-uh", type=float, required=True)
    search.add_argument("--optimizer-all-thermal-scale-c", type=float, required=True)
    search.add_argument("--optimizer-repair-contract-sha256", required=True)
    search.add_argument("--stage-spec-json", required=True)
    search.add_argument("--stage-spec-sha256", required=True)
    search.add_argument("--optimizer-resonance-allowance-hz", type=float)
    search.add_argument("--optimizer-llt-allowance-uh", type=float)
    validate = subparsers.add_parser("validate-receipt")
    validate.add_argument("receipt", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate-receipt":
        receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
        validate_smoke_receipt(receipt)
        print(
            json.dumps(
                {
                    "valid": True,
                    "schema_version": receipt["schema_version"],
                    "payload_sha256": receipt["payload_sha256"],
                    "launch_eligible": True,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "search-seed":
        stage_spec = stage_spec_from_json_identity(
            args.stage_spec_json, args.stage_spec_sha256
        )
        result_path = run_search_seed(
            bundle_root=args.bundle_root,
            relocation_path=args.relocation,
            adapter_receipt_path=args.adapter_receipt,
            registry=args.registry,
            generation=args.generation,
            dataset=args.dataset,
            profile=args.profile,
            warm_start=args.warm_start,
            warm_contract=args.warm_contract,
            output=args.output,
            remote_preflight=args.remote_preflight,
            bundle_id=args.bundle_id,
            island_id=args.island_id,
            island_profile_sha256=args.island_profile_sha256,
            seed=args.seed,
            population=args.population,
            max_generations=args.max_generations,
            inference_threads=args.inference_threads,
            scheduler_cpus=args.scheduler_cpus,
            fixed_primary_turns=args.fixed_primary_turns,
            optimizer_termination_strategy=(args.optimizer_termination_strategy),
            optimizer_resonance_scale_hz=(args.optimizer_resonance_scale_hz),
            optimizer_llt_scale_uh=args.optimizer_llt_scale_uh,
            optimizer_all_thermal_scale_c=(args.optimizer_all_thermal_scale_c),
            optimizer_repair_contract_sha256=(args.optimizer_repair_contract_sha256),
            stage_spec=stage_spec,
            stage_spec_sha256=args.stage_spec_sha256,
            optimizer_resonance_allowance_hz=(args.optimizer_resonance_allowance_hz),
            optimizer_llt_allowance_uh=args.optimizer_llt_allowance_uh,
        )
        print(result_path)
        return 0
    if (args.stage_spec_json is None) != (args.stage_spec_sha256 is None):
        raise RuntimeError("smoke stage-spec JSON/SHA must be supplied together")
    if (args.warm_start_n1_6_contract is None) != (
        args.warm_start_n1_6_contract_sha256 is None
    ):
        raise RuntimeError("N1=6 smoke warm-contract path/SHA must be supplied together")
    smoke_stage_spec = (
        validate_stage_spec(CURRENT_STAGE_SPEC)
        if args.stage_spec_json is None
        else stage_spec_from_json_identity(args.stage_spec_json, args.stage_spec_sha256)
    )
    receipt_path = run_smoke_preflight(
        generation=args.generation,
        candidate_path=args.candidate,
        quality_path=args.quality_status,
        code_root=args.code_root,
        expected_code_revision=args.expected_code_revision,
        coordinate_path=args.coordinate,
        coordinate_sha256=args.coordinate_sha256,
        warm_start_paths={
            5: args.warm_start_n1_5,
            6: args.warm_start_n1_6,
        },
        warm_start_sha256={
            5: args.warm_start_n1_5_sha256,
            6: args.warm_start_n1_6_sha256,
        },
        warm_start_contract_paths=(
            {}
            if args.warm_start_n1_6_contract is None
            else {6: args.warm_start_n1_6_contract}
        ),
        warm_start_contract_sha256=(
            {}
            if args.warm_start_n1_6_contract_sha256 is None
            else {6: args.warm_start_n1_6_contract_sha256}
        ),
        output=args.output,
        stage_spec=smoke_stage_spec,
        inference_threads=args.inference_threads,
    )
    print(receipt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
