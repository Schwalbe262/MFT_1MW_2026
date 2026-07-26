import numpy as np

from tools import mft_goal_fixed_primary_5t_campaign as campaign


def test_targeted_campaign_searches_all_four_turn_strata_with_fresh_seeds():
    assert campaign.TARGETED_SEED_COUNT == 256
    assert campaign.TARGETED_SEED_COUNT % 4 == 0
    assert "splittemp-v2" in campaign.TARGETED_CAMPAIGN_ID


def test_targeted_hard_spec_has_independent_temperature_limits():
    assert "winding_temperature_max_C" not in campaign.TARGETED_HARD_SPEC
    assert campaign.TARGETED_HARD_SPEC["temperature_family_limits_C"] == {
        "primary_winding": 100.0,
        "secondary_winding": 120.0,
        "core": 120.0,
    }
    assert (
        campaign.TARGETED_HARD_SPEC[
            "legacy_scalar_winding_temperature_limit_allowed"
        ]
        is False
    )


def test_split_temperature_transform_changes_only_secondary_constraints():
    names = (
        "temperature_robust_limit:T_max_Tx",
        *campaign.SECONDARY_TEMPERATURE_CONSTRAINTS,
        "temperature_robust_limit:T_max_core",
    )
    indices = {name: index for index, name in enumerate(names)}
    physical_g = np.array(
        [
            [10.0, 15.0, -25.0, 14.0, -24.0, -2.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ]
    )

    observed = campaign._apply_targeted_split_temperature_constraints(
        physical_g.copy(),
        constraint_index=indices,
        valid_indices=np.array([0]),
    )

    np.testing.assert_allclose(
        observed[0],
        [10.0, -5.0, -45.0, -6.0, -44.0, -2.0],
    )
    np.testing.assert_allclose(observed[1], physical_g[1])


def test_axis_mapping_smoke_ignores_invalid_decoder_rows():
    assert campaign._physically_decoded_smoke_indices(
        [False, True, False, True]
    ) == (1, 3)

    with np.testing.assert_raises_regex(
        RuntimeError, "no physically decoded rows"
    ):
        campaign._physically_decoded_smoke_indices([False, False])
