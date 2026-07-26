"""Authoritative contract for the 2026-07-26 1 MW MFT goal campaign.

The legacy current7 campaign used one scalar temperature limit, a fixed
5 mm primary conductor and a four-core-group ceiling.  Those fields are
deliberately not aliases for this campaign: every producer and consumer must
bind this exact payload and must fail closed if a legacy field is mixed into
it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Mapping


GOAL_CAMPAIGN_ID = "mft-goal-20260726"
GOAL_CONTRACT_SCHEMA = "mft-goal-20260726-hard-contract-v2"
GOAL_TEMPERATURE_CONTRACT_SCHEMA = (
    "mft-goal-20260726-split-temperature-contract-v2"
)
GOAL_TERMINAL_TABLE_SCHEMA = (
    "mft-goal-20260726-terminal-physical-candidates-v1"
)

PROBE_PRIMARY_WINDING_TEMPERATURE_TARGETS = (
    "Tprobe_Tx_leeward_max",
)
PROBE_SECONDARY_WINDING_TEMPERATURE_TARGETS = (
    "Tprobe_Rx_main_leeward_max",
    "Tprobe_Rx_side_leeward_max",
)
PROBE_WINDING_TEMPERATURE_TARGETS = (
    *PROBE_PRIMARY_WINDING_TEMPERATURE_TARGETS,
    *PROBE_SECONDARY_WINDING_TEMPERATURE_TARGETS,
)
PROBE_CORE_TEMPERATURE_TARGETS = (
    "Tprobe_core_center_max",
    "Tprobe_core_center_leg_max",
    "Tprobe_core_side_leg_max",
    "Tprobe_core_top_yoke_max",
)
PROBE_TEMPERATURE_TARGETS = (
    *PROBE_WINDING_TEMPERATURE_TARGETS,
    *PROBE_CORE_TEMPERATURE_TARGETS,
)
BODY_PRIMARY_WINDING_TEMPERATURE_TARGETS = ("T_max_Tx",)
BODY_SECONDARY_WINDING_TEMPERATURE_TARGETS = (
    "T_max_Rx_main",
    "T_max_Rx_side",
)
BODY_WINDING_TEMPERATURE_TARGETS = (
    *BODY_PRIMARY_WINDING_TEMPERATURE_TARGETS,
    *BODY_SECONDARY_WINDING_TEMPERATURE_TARGETS,
)
BODY_CORE_TEMPERATURE_TARGETS = ("T_max_core",)
WINDING_TEMPERATURE_TARGETS = (
    *BODY_WINDING_TEMPERATURE_TARGETS,
    *PROBE_WINDING_TEMPERATURE_TARGETS,
)
PRIMARY_WINDING_TEMPERATURE_TARGETS = (
    *BODY_PRIMARY_WINDING_TEMPERATURE_TARGETS,
    *PROBE_PRIMARY_WINDING_TEMPERATURE_TARGETS,
)
SECONDARY_WINDING_TEMPERATURE_TARGETS = (
    *BODY_SECONDARY_WINDING_TEMPERATURE_TARGETS,
    *PROBE_SECONDARY_WINDING_TEMPERATURE_TARGETS,
)
CORE_TEMPERATURE_TARGETS = (
    *BODY_CORE_TEMPERATURE_TARGETS,
    *PROBE_CORE_TEMPERATURE_TARGETS,
)
GOAL_TEMPERATURE_TARGETS = (
    *WINDING_TEMPERATURE_TARGETS,
    *CORE_TEMPERATURE_TARGETS,
)
GOAL_G0_MODEL_TARGETS = (
    "Llt_phys",
    "k",
    "C_tx_tx_F",
    "C_rx_rx_F",
    "C_tx_rx_F",
    "P_winding_total",
    "P_Tx_main_group",
    "P_Rx_main_group",
    "P_Rx_side_total",
    "P_core_total",
    "P_core_plate_total",
    "P_wcp_total",
    "B_max_core",
    "B_mean_core",
    *PROBE_TEMPERATURE_TARGETS,
    *BODY_WINDING_TEMPERATURE_TARGETS,
    *BODY_CORE_TEMPERATURE_TARGETS,
)
if len(GOAL_G0_MODEL_TARGETS) != 25:
    raise RuntimeError("goal G0 must contain exactly 25 model targets")
TEMPERATURE_FAMILY_LIMITS_C = {
    "primary_winding": 100.0,
    "secondary_winding": 120.0,
    "core": 120.0,
}
TEMPERATURE_TARGET_FAMILIES = {
    **{
        target: "primary_winding"
        for target in PRIMARY_WINDING_TEMPERATURE_TARGETS
    },
    **{
        target: "secondary_winding"
        for target in SECONDARY_WINDING_TEMPERATURE_TARGETS
    },
    **{target: "core" for target in CORE_TEMPERATURE_TARGETS},
}
TEMPERATURE_TARGET_LIMITS_C = {
    target: TEMPERATURE_FAMILY_LIMITS_C[family]
    for target, family in TEMPERATURE_TARGET_FAMILIES.items()
}
ACTUAL_TEMPERATURE_FIELD_LIMITS_C = {
    target: TEMPERATURE_TARGET_LIMITS_C[target]
    for target in (
        *BODY_WINDING_TEMPERATURE_TARGETS,
        *BODY_CORE_TEMPERATURE_TARGETS,
    )
}

GOAL_SIZE_LIMITS_MM = {"W": 1_200.0, "L": 1_000.0, "H": 750.0}
GOAL_RESONANCE_MIN_HZ = 15_000.0
GOAL_CW1_MIN_MM = 1.0
GOAL_CW1_MAX_MM = 10.0
GOAL_CW1_STEP_MM = 0.01
GOAL_CW1_VALUE_COUNT = (
    int(round((GOAL_CW1_MAX_MM - GOAL_CW1_MIN_MM) / GOAL_CW1_STEP_MM))
    + 1
)
GOAL_N_CORE_GROUP_MIN = 2
GOAL_N_CORE_GROUP_MAX = 10
GOAL_CORE_DEPTH_MIN_MM = 60.0
GOAL_CORE_DEPTH_MAX_MM = 120.0
GOAL_N1_MIN_TURNS = 5
GOAL_N1_MAX_TURNS = 8
GOAL_PRIMARY_TURN_STRATA = tuple(
    range(GOAL_N1_MIN_TURNS, GOAL_N1_MAX_TURNS + 1)
)

FIXED_OPERATING_IDENTITY = {
    "freq": 1_000.0,
    "V1_rms": 1_000.0,
    "V2_rms": 10_000.0,
    "I1_rated": 1_000.0,
    "I2_rated": 100.0,
    "I2_phase_deg": 0.0,
    "P_target": 1_000_000.0,
    "plate_temp": 50.0,
    "air_temp": 50.0,
    "conductor_temp_C": 80.0,
}
FIXED_COOLING_IDENTITY = {
    "fan_velocity": 1.5,
    "fan_config": "dual",
    "core_plate_on": 1,
    "wcp_on": 1,
    "core_plate_pad_t": 2.0,
    "wcp_pad_t": 2.0,
    "k_ins": 0.2,
    "core_k_thermal": 2.0,
    "thermal_pad_conductivity_W_mK": 0.2,
}

LEGACY_GOAL_FORBIDDEN_KEYS = frozenset(
    {
        "T_limit_C",
        "temperature_limit_C",
        "resonance_max_Hz",
        "n_core_group_max",
        "primary_conductor_thickness_mm",
        "fixed_primary_turns",
        "f1_split",
    }
)

GOAL_TEMPERATURE_CONTRACT = {
    "schema_version": GOAL_TEMPERATURE_CONTRACT_SCHEMA,
    "semantic_version": (
        "split-primary100-secondary120-core120-q90-half-width-v2"
    ),
    "families": {
        "primary_winding": {
            "robust_upper_bound_C": 100.0,
            "targets": list(PRIMARY_WINDING_TEMPERATURE_TARGETS),
        },
        "secondary_winding": {
            "robust_upper_bound_C": 120.0,
            "targets": list(SECONDARY_WINDING_TEMPERATURE_TARGETS),
        },
        "core": {
            "robust_upper_bound_C": 120.0,
            "targets": list(CORE_TEMPERATURE_TARGETS),
        },
    },
    "target_family": dict(TEMPERATURE_TARGET_FAMILIES),
    "target_limits_C": dict(TEMPERATURE_TARGET_LIMITS_C),
    "side_winding_conditional_targets": [
        "T_max_Rx_side",
        "Tprobe_Rx_side_leeward_max",
    ],
    "side_winding_activation": "finite_N2_side_gt_0",
    "side_winding_absent_behavior": (
        "finite_N2_side_eq_0_disables_with_negative_BIG"
    ),
    "side_winding_missing_behavior": (
        "missing_or_nonfinite_N2_side_fails_with_positive_BIG"
    ),
    "formula": "surrogate_mu_plus_q90_conformal_half_width_le_target_limit",
    "legacy_scalar_temperature_limit_allowed": False,
}

GOAL_STAGE_SPEC = {
    "contract_schema": GOAL_CONTRACT_SCHEMA,
    "campaign_id": GOAL_CAMPAIGN_ID,
    "Llt_target_uH": 27.5,
    "Llt_tol_uH": 0.55,
    "B_limit_T": 1.2,
    "insulation_min_mm": 40.0,
    "q_sigma": 1.0,
    "temperature_family_limits_C": dict(TEMPERATURE_FAMILY_LIMITS_C),
    "temperature_target_limits_C": dict(TEMPERATURE_TARGET_LIMITS_C),
    "resonance_min_Hz": GOAL_RESONANCE_MIN_HZ,
    "magnetizing_inductance_factor": 0.5,
    "size_limits_mm": dict(GOAL_SIZE_LIMITS_MM),
    "cw1_search_mm": {
        "minimum": GOAL_CW1_MIN_MM,
        "maximum": GOAL_CW1_MAX_MM,
        "step": GOAL_CW1_STEP_MM,
        "value_count": GOAL_CW1_VALUE_COUNT,
        "chromosome_coordinate": "f1_split",
        "f1_split_independent_search_allowed": False,
        "decoder_enforcement": (
            "selected_cw1_consumed_inside_winding_budget_then_identity_attested"
        ),
    },
    "n_core_group_search": {
        "minimum": GOAL_N_CORE_GROUP_MIN,
        "maximum": GOAL_N_CORE_GROUP_MAX,
        "selection": "canonical_decoder_dynamic_core_depth_validity",
        "fixed_ceiling_four_allowed": False,
    },
    "core_depth_mm": {
        "minimum": GOAL_CORE_DEPTH_MIN_MM,
        "maximum": GOAL_CORE_DEPTH_MAX_MM,
    },
    "primary_turn_search": {
        "minimum": GOAL_N1_MIN_TURNS,
        "maximum": GOAL_N1_MAX_TURNS,
        "strata": list(GOAL_PRIMARY_TURN_STRATA),
    },
    "fixed_operating_identity": dict(FIXED_OPERATING_IDENTITY),
    "fixed_cooling_identity": dict(FIXED_COOLING_IDENTITY),
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


GOAL_STAGE_SPEC_SHA256 = canonical_sha256(GOAL_STAGE_SPEC)
GOAL_TEMPERATURE_CONTRACT_SHA256 = canonical_sha256(
    GOAL_TEMPERATURE_CONTRACT
)
FIXED_OPERATING_IDENTITY_SHA256 = canonical_sha256(
    FIXED_OPERATING_IDENTITY
)
FIXED_COOLING_IDENTITY_SHA256 = canonical_sha256(FIXED_COOLING_IDENTITY)


class GoalContractError(RuntimeError):
    """Raised when a producer or consumer mixes incompatible goal semantics."""


def is_goal_stage_spec(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("contract_schema") == GOAL_CONTRACT_SCHEMA
    )


def validate_goal_stage_spec(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GoalContractError("goal stage spec must be an object")
    supplied = dict(value)
    forbidden = sorted(LEGACY_GOAL_FORBIDDEN_KEYS.intersection(supplied))
    if forbidden:
        raise GoalContractError(
            "goal stage spec contains forbidden legacy fields: "
            + ",".join(forbidden)
        )
    if supplied.get("contract_schema") != GOAL_CONTRACT_SCHEMA:
        raise GoalContractError("goal stage spec schema is missing or invalid")
    if set(supplied) != set(GOAL_STAGE_SPEC):
        raise GoalContractError("goal stage spec has missing or unknown keys")
    try:
        digest = canonical_sha256(supplied)
    except (TypeError, ValueError) as exc:
        raise GoalContractError("goal stage spec is not canonical JSON") from exc
    if digest != GOAL_STAGE_SPEC_SHA256:
        raise GoalContractError("goal stage spec differs from the fixed goal")
    return copy.deepcopy(GOAL_STAGE_SPEC)


def temperature_limit_for_target(target: str) -> float:
    try:
        return float(TEMPERATURE_TARGET_LIMITS_C[str(target)])
    except KeyError as exc:
        raise GoalContractError(
            f"temperature target has no goal family: {target}"
        ) from exc


def cw1_from_unit_coordinate(value: Any) -> float:
    unit = _finite_number(value, "cw1 unit coordinate")
    if unit < 0.0 or unit > 1.0:
        raise GoalContractError("cw1 unit coordinate must be within [0,1]")
    index = min(
        GOAL_CW1_VALUE_COUNT - 1,
        int(math.floor(unit * GOAL_CW1_VALUE_COUNT)),
    )
    return round(GOAL_CW1_MIN_MM + index * GOAL_CW1_STEP_MM, 2)


def cw1_unit_coordinate(value: Any) -> float:
    cw1 = validate_cw1_mm(value)
    index = int(round((cw1 - GOAL_CW1_MIN_MM) / GOAL_CW1_STEP_MM))
    return (index + 0.5) / GOAL_CW1_VALUE_COUNT


def validate_cw1_mm(value: Any) -> float:
    cw1 = _finite_number(value, "cw1")
    if not GOAL_CW1_MIN_MM <= cw1 <= GOAL_CW1_MAX_MM:
        raise GoalContractError("cw1 must be within [1.00,10.00] mm")
    index = round((cw1 - GOAL_CW1_MIN_MM) / GOAL_CW1_STEP_MM)
    snapped = GOAL_CW1_MIN_MM + index * GOAL_CW1_STEP_MM
    if not math.isclose(cw1, snapped, rel_tol=0.0, abs_tol=1e-9):
        raise GoalContractError("cw1 must lie on the 0.01 mm grid")
    return round(snapped, 2)


def dynamic_core_group_bounds(row: Mapping[str, Any]) -> tuple[int, int]:
    plate = _finite_number(row.get("core_plate_t"), "core_plate_t")
    pad = _finite_number(row.get("core_plate_pad_t"), "core_plate_pad_t")
    width = round(_finite_number(row.get("w1"), "w1"))
    depth_min = _finite_number(row.get("core_depth_min"), "core_depth_min")
    depth_max = _finite_number(row.get("core_depth_max"), "core_depth_max")
    if (
        depth_min != GOAL_CORE_DEPTH_MIN_MM
        or depth_max != GOAL_CORE_DEPTH_MAX_MM
    ):
        raise GoalContractError("core depth bounds differ from 60..120 mm")
    stack = plate + 2.0 * pad
    native_minimum = max(
        1,
        int(math.ceil((width - stack) / (depth_max + stack))),
    )
    native_maximum = max(
        native_minimum,
        int(math.floor((width - stack) / (depth_min + stack))),
    )
    minimum = max(GOAL_N_CORE_GROUP_MIN, native_minimum)
    maximum = min(GOAL_N_CORE_GROUP_MAX, native_maximum)
    if minimum > maximum:
        raise GoalContractError("geometry has no valid core-group count")
    return minimum, maximum


def dynamic_core_group_violation(row: Mapping[str, Any]) -> float:
    group = _finite_number(row.get("n_core_group"), "n_core_group")
    if not float(group).is_integer():
        raise GoalContractError("n_core_group must be an integer")
    minimum, maximum = dynamic_core_group_bounds(row)
    return max(float(minimum) - group, group - float(maximum))


def fixed_identity_expectations() -> dict[str, Any]:
    return {
        **copy.deepcopy(FIXED_OPERATING_IDENTITY),
        **copy.deepcopy(FIXED_COOLING_IDENTITY),
    }


def fixed_identity_mismatches(
    values: Mapping[str, Any],
    *,
    require_thermal_pad_metadata: bool = True,
) -> list[dict[str, Any]]:
    if not isinstance(values, Mapping):
        return [
            {
                "field": "*",
                "reason": "identity_is_not_an_object",
                "expected": fixed_identity_expectations(),
                "observed": None,
            }
        ]
    mismatches: list[dict[str, Any]] = []
    for field, expected in fixed_identity_expectations().items():
        if (
            field == "thermal_pad_conductivity_W_mK"
            and not require_thermal_pad_metadata
        ):
            continue
        if field not in values:
            mismatches.append(
                {
                    "field": field,
                    "reason": "missing",
                    "expected": expected,
                    "observed": None,
                }
            )
            continue
        observed = values[field]
        if isinstance(expected, str):
            matched = observed == expected
        else:
            try:
                observed = _finite_number(observed, field)
            except GoalContractError:
                matched = False
            else:
                matched = observed == float(expected)
        if not matched:
            mismatches.append(
                {
                    "field": field,
                    "reason": "not_exact",
                    "expected": expected,
                    "observed": observed,
                }
            )
    return mismatches


def attest_fixed_identity(
    values: Mapping[str, Any],
    *,
    require_thermal_pad_metadata: bool = True,
) -> dict[str, Any]:
    mismatches = fixed_identity_mismatches(
        values,
        require_thermal_pad_metadata=require_thermal_pad_metadata,
    )
    if mismatches:
        detail = "; ".join(
            f"{item['field']} expected={item['expected']!r} "
            f"observed={item['observed']!r}"
            for item in mismatches
        )
        raise GoalContractError("fixed operating/cooling identity failed: " + detail)
    evidence = {
        "contract_schema": GOAL_CONTRACT_SCHEMA,
        "goal_spec_sha256": GOAL_STAGE_SPEC_SHA256,
        "expected": fixed_identity_expectations(),
        "observed": {
            key: values[key]
            for key in fixed_identity_expectations()
            if key in values
        },
        "mismatches": [],
        "attested": True,
    }
    evidence["sha256"] = canonical_sha256(evidence)
    return evidence


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise GoalContractError(f"{label} must be numeric, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GoalContractError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise GoalContractError(f"{label} must be finite")
    return number
