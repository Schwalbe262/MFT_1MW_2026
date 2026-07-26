import copy
from unittest import mock

from tools import mft_goal_rx_interface_preflight_guard as guard


def _marker() -> dict:
    return {
        "schema": "thermal-rx-interface-predispatch-v1",
        "passed": True,
        "thermal_rx_block_interface_contract_version": (
            "thermal-rx-block-interface-coverage-v1"
        ),
        "mesh_policy": (
            "b6-rx-block-shared-region-wcp-pad-symmetry-contact-clipped-v1"
        ),
        "mesh_plan_contract_version": "thermal-mesh-plan-v7",
        "rx_main_objects": ["Rx_main_block_xn", "Rx_main_block_yp"],
        "rx_block_shared_pack_count": 1,
        "rx_main_shared_operations": [
            {
                "plan_name": "rx_main_block_mesh_level",
                "objects": ["Rx_main_block_xn", "Rx_main_block_yp"],
                "shared_region": True,
                "separate_objects": False,
                "native_separate_objects": False,
            }
        ],
        "shared_intent_and_native_readback_passed": True,
        "native_operation_readback_passed": True,
        "premesh_status": "explicit_opt_in_symmetry_direct_analyze",
        "generate_mesh_returned": False,
        "direct_analyze_gate_passed": True,
        "fixed_cooling_identity": {
            "schema": "mft-fixed-thermal-boundary-v1",
            "contract_sha256": "a" * 64,
            "fan_config": "dual",
            "fan_velocity_m_s": 1.5,
            "core_plate_pad_t_mm": 2.0,
            "wcp_pad_t_mm": 2.0,
            "thermal_pad_conductivity_W_mK": 0.2,
            "cooling_boundary_modified": False,
        },
        "terminal_interface_coverage_pending": True,
        "thermal_rx_main_interface_coverage_passed": False,
        "thermal_result_scientific_valid": False,
    }


def test_accepts_exact_corrected_predispatch_marker() -> None:
    assert guard._validate_marker(_marker()) == []


def test_accepts_exact_canary_scheduler_readback() -> None:
    task = {
        "project": "MFT_1MW_2026v1",
        "cpus": 8,
        "memory_mb": 65_536,
        "aedt_backend": "standalone",
        "priority": 100,
        "timeout_seconds": 43_200,
    }
    assert guard._validate_task_readback(task) == []
    task["timeout_seconds"] = 14_400
    assert guard._validate_task_readback(task) == [
        "task_readback_timeout_seconds_mismatch"
    ]


def test_rejects_old_separate_objects_topology() -> None:
    value = copy.deepcopy(_marker())
    value["rx_main_shared_operations"][0]["separate_objects"] = True
    value["rx_main_shared_operations"][0]["native_separate_objects"] = True
    value["shared_intent_and_native_readback_passed"] = False
    reasons = guard._validate_marker(value)
    assert "rx_main_shared_operation_separate_objects_mismatch" in reasons
    assert (
        "rx_main_shared_operation_native_separate_objects_mismatch" in reasons
    )
    assert "shared_intent_and_native_readback_passed_mismatch" in reasons


def test_rejects_fixed_cooling_identity_drift() -> None:
    value = copy.deepcopy(_marker())
    cooling = value["fixed_cooling_identity"]
    cooling["fan_config"] = "single"
    cooling["core_plate_pad_t_mm"] = 1.5
    cooling["wcp_pad_t_mm"] = 3.0
    reasons = guard._validate_marker(value)
    assert "fixed_cooling_identity_fan_config_mismatch" in reasons
    assert (
        "fixed_cooling_identity_core_plate_pad_t_mismatch" in reasons
    )
    assert "fixed_cooling_identity_wcp_pad_t_mismatch" in reasons


def test_extracts_last_exact_marker() -> None:
    encoded = guard.json.dumps(_marker(), separators=(",", ":"))
    parsed = guard._extract_marker(
        "noise\nTHERMAL_RX_INTERFACE_PREFLIGHT_JSON=" + encoded + "\n"
    )
    assert parsed == _marker()


def test_scheduler_log_reader_accepts_empty_and_plain_text() -> None:
    class Response:
        def __init__(self, value: bytes):
            self.value = value

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return self.value

    with mock.patch.object(
        guard.urllib.request, "urlopen", return_value=Response(b"")
    ):
        assert guard._scheduler_json("http://scheduler/stdout") == ""
    with mock.patch.object(
        guard.urllib.request,
        "urlopen",
        return_value=Response(b"plain solver log"),
    ):
        assert (
            guard._scheduler_json("http://scheduler/stdout")
            == "plain solver log"
        )
