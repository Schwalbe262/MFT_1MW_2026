from tools import mft_goal_targeted_symmetric_fea_batch as targeted


def _valid_contract() -> dict:
    return {
        "thermal_rx_block_interface_contract_version": "test-v1",
        "thermal_rx_main_interface_coverage_passed": True,
        "thermal_rx_main_unpaired_interfaces": [],
        "thermal_temperature_limiter_triggered": False,
        "thermal_temperature_limiter_max_K": 400.0,
        "thermal_result_scientific_valid": True,
    }


def _gate(
    result: dict,
    *,
    task_log_text: str = "",
    temperature_C: float = 100.0,
) -> dict:
    return targeted._thermal_scientific_truth_contract(
        result,
        observed_temperatures_C={"T_max_Tx": temperature_C},
        task_log_text=task_log_text,
    )


def test_truth_gate_accepts_complete_explicit_contract() -> None:
    assert _gate(_valid_contract())["valid"] is True


def test_truth_gate_rejects_legacy_result_with_missing_fields() -> None:
    gated = _gate({})
    assert gated["valid"] is False
    assert len(gated["reasons"]) == len(
        targeted.THERMAL_RX_INTERFACE_CONTRACT_FIELDS
    )


def test_truth_gate_rejects_unpaired_and_wall_reset_marker() -> None:
    result = _valid_contract()
    result["thermal_rx_main_unpaired_interfaces"] = ["153"]
    gated = _gate(
        result,
        task_log_text="set unpaired interface zone 153 to type wall",
    )
    assert gated["valid"] is False
    assert any("unpaired_interfaces_present" in row for row in gated["reasons"])
    assert any("task_log_marker" in row for row in gated["reasons"])


def test_truth_gate_rejects_temperature_limiter_fields_and_log() -> None:
    result = _valid_contract()
    result["thermal_temperature_limiter_triggered"] = True
    result["thermal_temperature_limiter_max_K"] = 5000.0
    result["thermal_result_scientific_valid"] = False
    gated = _gate(
        result,
        task_log_text="temperature limited to 5.000000e+03",
        temperature_C=4726.85,
    )
    assert gated["valid"] is False
    assert any("near_5000K" in row for row in gated["reasons"])
    assert any("task_log_marker" in row for row in gated["reasons"])
