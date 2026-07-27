"""Manufacturable physical air-gap topology contract.

The historical fixed-run model split only the center leg.  That representation
is retained so old AEDT/result artifacts remain readable, but it is not a
manufacturable final-design authority.  New tuning/final flows must use one
identical physical gap in the center leg and both side legs.
"""

import math


AIR_GAP_CONTRACT_SCHEMA = "mft-core-air-gap-topology-contract-v1"

LEGACY_CENTER_ONLY_TOPOLOGY = "legacy_center_leg_only"
EQUAL_THREE_LEG_TOPOLOGY = "equal_center_and_both_side_legs"

LEGACY_CENTER_ONLY_RESULT_TOPOLOGY = (
    "center_leg_bottom_top_physical_air_interval"
)
EQUAL_THREE_LEG_RESULT_TOPOLOGY = (
    "center_and_both_side_legs_bottom_top_physical_air_intervals"
)


def topology_from_flag(equal_three_leg):
    """Return the canonical topology name for an input flag."""
    return (
        EQUAL_THREE_LEG_TOPOLOGY
        if int(equal_three_leg) == 1
        else LEGACY_CENTER_ONLY_TOPOLOGY
    )


def build_air_gap_contract(
    *,
    gap_mm,
    equal_three_leg,
    symmetric_fea_verified=False,
    physical_lm_h=None,
    target_lm_h=0.002,
    tolerance_h=0.00002,
    final_promotion_allowed=False,
):
    """Build and validate a fail-closed air-gap promotion contract."""
    gap = float(gap_mm)
    flag = int(equal_three_leg)
    if not math.isfinite(gap) or gap <= 0.0:
        raise ValueError("physical air-gap contract requires gap_mm > 0")
    if flag not in (0, 1) or float(equal_three_leg) != float(flag):
        raise ValueError("equal_three_leg must be exactly 0 or 1")

    verified = bool(symmetric_fea_verified)
    lm_h = None if physical_lm_h is None else float(physical_lm_h)
    if lm_h is not None and not math.isfinite(lm_h):
        raise ValueError("physical_lm_h must be finite when supplied")
    target = float(target_lm_h)
    tolerance = float(tolerance_h)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("target_lm_h must be finite and > 0")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance_h must be finite and >= 0")
    lm_within_tolerance = (
        lm_h is not None and abs(lm_h - target) <= tolerance
    )
    physical_verified = flag == 1 and verified and lm_within_tolerance
    promotion = bool(final_promotion_allowed)
    if promotion and not physical_verified:
        raise ValueError(
            "final promotion requires equal-three-leg symmetric FEA "
            "verification of physical Lm"
        )

    return {
        "schema_version": AIR_GAP_CONTRACT_SCHEMA,
        "topology": topology_from_flag(flag),
        "core_equal_three_leg_air_gap": flag,
        "physical_gap_mm": gap,
        "gapped_leg_count": 3 if flag else 1,
        "identical_physical_gap_all_three_legs": flag == 1,
        "manufacturable_final_topology": flag == 1,
        "legacy_artifact_only": flag == 0,
        # This requirement remains true even when describing a legacy artifact:
        # the legacy artifact is diagnostic and still needs replacement.
        "equal_three_leg_air_gap_FEA_required": True,
        "target_physical_Lm_H": target,
        "physical_Lm_tolerance_H": tolerance,
        "physical_Lm_H": lm_h,
        "physical_Lm_within_tolerance": lm_within_tolerance,
        "symmetric_FEA_geometry_verified": verified,
        "physical_Lm_2mH_symmetric_FEA_verified": physical_verified,
        "final_promotion_blocked_by_air_gap": not physical_verified,
        "final_promotion_allowed": promotion,
    }


def validate_air_gap_contract(
    contract, *, require_equal_three_leg=False, require_verified=False
):
    """Validate a serialized contract and optionally enforce promotion gates."""
    if not isinstance(contract, dict):
        raise ValueError("air-gap contract must be a mapping")
    rebuilt = build_air_gap_contract(
        gap_mm=contract.get("physical_gap_mm"),
        equal_three_leg=contract.get("core_equal_three_leg_air_gap"),
        symmetric_fea_verified=contract.get(
            "symmetric_FEA_geometry_verified", False
        ),
        physical_lm_h=contract.get("physical_Lm_H"),
        target_lm_h=contract.get("target_physical_Lm_H", 0.002),
        tolerance_h=contract.get("physical_Lm_tolerance_H", 0.00002),
        final_promotion_allowed=contract.get(
            "final_promotion_allowed", False
        ),
    )
    for key, expected in rebuilt.items():
        if contract.get(key) != expected:
            raise ValueError(
                f"air-gap contract field mismatch: {key}="
                f"{contract.get(key)!r} != {expected!r}"
            )
    if contract.get("schema_version") != AIR_GAP_CONTRACT_SCHEMA:
        raise ValueError("unsupported air-gap contract schema")
    if require_equal_three_leg and rebuilt["core_equal_three_leg_air_gap"] != 1:
        raise ValueError(
            "legacy center-only air gap is diagnostic-only; "
            "equal-three-leg topology is required"
        )
    if require_verified and not rebuilt[
        "physical_Lm_2mH_symmetric_FEA_verified"
    ]:
        raise ValueError(
            "equal-three-leg physical Lm=2mH symmetric FEA is not verified"
        )
    return rebuilt
