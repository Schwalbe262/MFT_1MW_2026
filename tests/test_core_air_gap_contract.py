import pytest

from module.core_air_gap_contract import (
    EQUAL_THREE_LEG_TOPOLOGY,
    LEGACY_CENTER_ONLY_TOPOLOGY,
    build_air_gap_contract,
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
