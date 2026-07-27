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


def positive_y_retained_core_group_bounds(
    *,
    n_group,
    w1_mm,
    core_plate_t_mm,
    core_plate_pad_t_mm,
    tolerance_mm=1e-9,
):
    """Return core groups with non-zero volume after the ``y >= 0`` cut.

    ``w1`` is the complete core/cold-plate stack depth.  Core group ``i`` has
    the same independent placement expression as ``create_core``:

    ``y0 = -w1/2 + i*stack + (i-1)*depth``.

    A group whose upper face is on the symmetry plane has no positive-side
    volume and is therefore excluded.  An odd-group center block crosses the
    plane and is returned with its retained lower bound clamped to zero.
    """
    count = int(n_group)
    if count <= 0 or float(n_group) != float(count):
        raise ValueError("n_group must be a positive integer")
    w1 = float(w1_mm)
    plate = float(core_plate_t_mm)
    pad = float(core_plate_pad_t_mm)
    tolerance = float(tolerance_mm)
    values = (w1, plate, pad, tolerance)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("core-stack dimensions and tolerance must be finite")
    if w1 <= 0.0 or plate < 0.0 or pad < 0.0 or tolerance < 0.0:
        raise ValueError("invalid core-stack dimensions or tolerance")
    stack = plate + 2.0 * pad
    depth = (w1 - (count + 1) * stack) / count
    if depth <= 0.0:
        raise ValueError("core group depth must be positive")

    retained = []
    for index in range(1, count + 1):
        y_min = -w1 / 2.0 + index * stack + (index - 1) * depth
        y_max = y_min + depth
        if y_max > tolerance:
            retained.append(
                {
                    "index": index,
                    "source_y_min_mm": y_min,
                    "retained_y_min_mm": max(0.0, y_min),
                    "retained_y_max_mm": y_max,
                }
            )
    if not retained:
        raise ValueError("positive-y symmetry cut retained no core groups")
    return tuple(retained)


def attest_equal_three_leg_symmetry_piece_bounds(
    piece_bounding_boxes,
    *,
    n_group,
    w1_mm,
    core_plate_t_mm,
    core_plate_pad_t_mm,
    l1_mm,
    l2_mm,
    h1_mm,
    gap_mm,
    tolerance_mm=1e-6,
):
    """Attest the physical gap pieces surviving XY+/XZ+/YZ- symmetry cuts.

    The expected object names come from the independently calculated core
    group y extents, not from the post-cut object list.  Bounding boxes then
    prove that the retained objects occupy z>=gap/2, y>=0 and x<=0, including
    the half-depth center group in odd ``n_group`` layouts.
    """
    if not isinstance(piece_bounding_boxes, dict):
        raise ValueError("piece_bounding_boxes must be a mapping")
    l1 = float(l1_mm)
    l2 = float(l2_mm)
    h1 = float(h1_mm)
    gap = float(gap_mm)
    tolerance = float(tolerance_mm)
    if not all(
        math.isfinite(value) for value in (l1, l2, h1, gap, tolerance)
    ):
        raise ValueError("symmetry dimensions and tolerance must be finite")
    if (
        l1 <= 0.0
        or l2 <= 0.0
        or h1 <= 0.0
        or gap <= 0.0
        or gap >= h1
        or tolerance < 0.0
    ):
        raise ValueError("invalid equal-three-leg symmetry dimensions")

    groups = positive_y_retained_core_group_bounds(
        n_group=n_group,
        w1_mm=w1_mm,
        core_plate_t_mm=core_plate_t_mm,
        core_plate_pad_t_mm=core_plate_pad_t_mm,
        tolerance_mm=min(tolerance, 1e-9),
    )
    expected = {}
    z_min = gap / 2.0
    z_max = h1 / 2.0
    x_bounds = {
        "left": (-(2.0 * l1 + l2), -(l1 + l2)),
        "center": (-l1, 0.0),
    }
    for group in groups:
        for leg, (x_min, x_max) in x_bounds.items():
            expected[
                f"core_{group['index']}_leg_{leg}_top"
            ] = (
                x_min,
                group["retained_y_min_mm"],
                z_min,
                x_max,
                group["retained_y_max_mm"],
                z_max,
            )

    actual_names = set(piece_bounding_boxes)
    expected_names = set(expected)
    if actual_names != expected_names:
        raise ValueError(
            "equal-three-leg air-gap symmetry retention mismatch: "
            f"actual={sorted(actual_names)!r}, "
            f"expected={sorted(expected_names)!r}"
        )
    maximum_error = 0.0
    for name, expected_box in expected.items():
        actual_box = tuple(float(value) for value in piece_bounding_boxes[name])
        if len(actual_box) != 6 or not all(
            math.isfinite(value) for value in actual_box
        ):
            raise ValueError(
                f"symmetry bounding-box readback unavailable for {name}"
            )
        errors = [
            abs(actual - target)
            for actual, target in zip(actual_box, expected_box)
        ]
        local_error = max(errors)
        maximum_error = max(maximum_error, local_error)
        if local_error > tolerance:
            raise ValueError(
                "equal-three-leg air-gap symmetry bounding-box mismatch: "
                f"name={name}, actual={actual_box!r}, "
                f"expected={expected_box!r}, max_error={local_error:.12g}mm"
            )
    return {
        "retained_group_indices": tuple(
            group["index"] for group in groups
        ),
        "retained_group_count": len(groups),
        "retained_piece_count": len(expected),
        "maximum_bounding_box_error_mm": maximum_error,
        "half_gap_readback_mm": 2.0 * z_min,
    }


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
