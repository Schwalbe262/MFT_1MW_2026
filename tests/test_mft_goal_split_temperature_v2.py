from module import mft_goal_20260726_contract as goal
from tools import mft_goal_fixed_lm2mh_targeted_collect as collector


def test_split_temperature_contract_has_independent_primary_secondary_limits():
    assert goal.GOAL_CONTRACT_SCHEMA.endswith("-v2")
    assert goal.GOAL_TEMPERATURE_CONTRACT_SCHEMA.endswith("-v2")
    assert goal.temperature_limit_for_target("T_max_Tx") == 100.0
    assert goal.temperature_limit_for_target("Tprobe_Tx_leeward_max") == 100.0
    assert goal.temperature_limit_for_target("T_max_Rx_main") == 120.0
    assert goal.temperature_limit_for_target("T_max_Rx_side") == 120.0
    assert (
        goal.temperature_limit_for_target("Tprobe_Rx_main_leeward_max")
        == 120.0
    )
    assert goal.temperature_limit_for_target("T_max_core") == 120.0


def test_completed_v1_nsga_rows_are_reclassified_without_changing_predictions():
    physical_v1 = {
        "temperature_robust_limit:T_max_Tx": 5.0,
        "temperature_robust_limit:Tprobe_Tx_leeward_max": 4.0,
        "temperature_robust_limit:T_max_Rx_main": 15.0,
        "temperature_robust_limit:T_max_Rx_side": -25.0,
        "temperature_robust_limit:Tprobe_Rx_main_leeward_max": 14.0,
        "temperature_robust_limit:Tprobe_Rx_side_leeward_max": -24.0,
        "temperature_robust_limit:T_max_core": -2.0,
    }
    normalized_v1 = {name: value / 10.0 for name, value in physical_v1.items()}

    physical_v2, normalized_v2 = (
        collector._reclassify_secondary_temperature_constraints(
            physical_v1,
            normalized_v1,
        )
    )

    assert physical_v2["temperature_robust_limit:T_max_Tx"] == 5.0
    assert physical_v2["temperature_robust_limit:T_max_Rx_main"] == -5.0
    assert physical_v2["temperature_robust_limit:T_max_Rx_side"] == -45.0
    assert normalized_v2["temperature_robust_limit:T_max_Rx_main"] == -0.5
    assert normalized_v2["temperature_robust_limit:T_max_Rx_side"] == -4.5
    assert collector._temperatures(physical_v2) == (105.0, 115.0, 118.0)
    assert physical_v1["temperature_robust_limit:T_max_Rx_main"] == 15.0
