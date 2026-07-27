import pytest

from module.core_air_gap_contract import (
    EQUAL_THREE_LEG_TOPOLOGY,
    LEGACY_CENTER_ONLY_TOPOLOGY,
    attest_equal_three_leg_symmetry_piece_bounds,
    build_air_gap_contract,
    positive_y_retained_core_group_bounds,
    validate_air_gap_contract,
)


def test_legacy_center_only_is_readable_but_diagnostic_only():
    contract = build_air_gap_contract(
        gap_mm=0.7,
        equal_three_leg=0,
        symmetric_fea_verified=True,
        physical_lm_h=0.002,
    )
    assert contract["topology"] == LEGACY_CENTER_ONLY_TOPOLOGY
    assert contract["legacy_artifact_only"] is True
    assert contract["equal_three_leg_air_gap_FEA_required"] is True
    assert contract["physical_Lm_2mH_symmetric_FEA_verified"] is False
    with pytest.raises(ValueError, match="diagnostic-only"):
        validate_air_gap_contract(
            contract, require_equal_three_leg=True
        )


def test_equal_three_leg_stays_blocked_until_symmetric_lm_is_verified():
    unverified = build_air_gap_contract(
        gap_mm=0.7,
        equal_three_leg=1,
    )
    assert unverified["topology"] == EQUAL_THREE_LEG_TOPOLOGY
    assert unverified["gapped_leg_count"] == 3
    assert unverified["final_promotion_blocked_by_air_gap"] is True
    with pytest.raises(ValueError, match="not verified"):
        validate_air_gap_contract(unverified, require_verified=True)

    verified = build_air_gap_contract(
        gap_mm=0.7,
        equal_three_leg=1,
        symmetric_fea_verified=True,
        physical_lm_h=0.002,
        final_promotion_allowed=True,
    )
    assert validate_air_gap_contract(
        verified,
        require_equal_three_leg=True,
        require_verified=True,
    )["final_promotion_allowed"] is True


def test_promotion_cannot_be_asserted_outside_lm_tolerance():
    with pytest.raises(ValueError, match="final promotion requires"):
        build_air_gap_contract(
            gap_mm=0.7,
            equal_three_leg=1,
            symmetric_fea_verified=True,
            physical_lm_h=0.0019,
            final_promotion_allowed=True,
        )


def test_n4_symmetry_retains_only_positive_y_groups_3_and_4():
    groups = positive_y_retained_core_group_bounds(
        n_group=4,
        w1_mm=480.0,
        core_plate_t_mm=20.0,
        core_plate_pad_t_mm=2.0,
    )
    assert tuple(group["index"] for group in groups) == (3, 4)
    assert [
        (group["retained_y_min_mm"], group["retained_y_max_mm"])
        for group in groups
    ] == [(12.0, 102.0), (126.0, 216.0)]

    boxes = {
        "core_3_leg_left_top": (-470.0, 12.0, 0.175, -385.0, 102.0, 273.5),
        "core_3_leg_center_top": (-85.0, 12.0, 0.175, 0.0, 102.0, 273.5),
        "core_4_leg_left_top": (-470.0, 126.0, 0.175, -385.0, 216.0, 273.5),
        "core_4_leg_center_top": (-85.0, 126.0, 0.175, 0.0, 216.0, 273.5),
    }
    audit = attest_equal_three_leg_symmetry_piece_bounds(
        boxes,
        n_group=4,
        w1_mm=480.0,
        core_plate_t_mm=20.0,
        core_plate_pad_t_mm=2.0,
        l1_mm=85.0,
        l2_mm=300.0,
        h1_mm=547.0,
        gap_mm=0.35,
    )
    assert audit["retained_group_indices"] == (3, 4)
    assert audit["retained_piece_count"] == 4
    assert audit["maximum_bounding_box_error_mm"] == 0.0


def test_n5_symmetry_clamps_center_group_to_positive_half():
    groups = positive_y_retained_core_group_bounds(
        n_group=5,
        w1_mm=484.0,
        core_plate_t_mm=20.0,
        core_plate_pad_t_mm=2.0,
    )
    assert tuple(group["index"] for group in groups) == (3, 4, 5)
    assert [
        (group["retained_y_min_mm"], group["retained_y_max_mm"])
        for group in groups
    ] == [(0.0, 34.0), (58.0, 126.0), (150.0, 218.0)]

    boxes = {}
    for index, y_min, y_max in (
        (3, 0.0, 34.0),
        (4, 58.0, 126.0),
        (5, 150.0, 218.0),
    ):
        boxes[f"core_{index}_leg_left_top"] = (
            -460.0, y_min, 0.25, -370.0, y_max, 285.0
        )
        boxes[f"core_{index}_leg_center_top"] = (
            -90.0, y_min, 0.25, 0.0, y_max, 285.0
        )
    audit = attest_equal_three_leg_symmetry_piece_bounds(
        boxes,
        n_group=5,
        w1_mm=484.0,
        core_plate_t_mm=20.0,
        core_plate_pad_t_mm=2.0,
        l1_mm=90.0,
        l2_mm=280.0,
        h1_mm=570.0,
        gap_mm=0.5,
    )
    assert audit["retained_group_indices"] == (3, 4, 5)
    assert audit["retained_piece_count"] == 6

    boxes["core_3_leg_center_top"] = (
        -90.0, -34.0, 0.25, 0.0, 34.0, 285.0
    )
    with pytest.raises(ValueError, match="bounding-box mismatch"):
        attest_equal_three_leg_symmetry_piece_bounds(
            boxes,
            n_group=5,
            w1_mm=484.0,
            core_plate_t_mm=20.0,
            core_plate_pad_t_mm=2.0,
            l1_mm=90.0,
            l2_mm=280.0,
            h1_mm=570.0,
            gap_mm=0.5,
        )
