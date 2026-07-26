import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from module import thermal_260706 as thermal


def _standard_df(**overrides):
    values = {
        "thermal_symmetry": "eighth",
        "full_model": 0,
        "fan_config": "dual",
        "fan_velocity": 1.5,
        "core_plate_on": 1,
        "core_plate_pad_t": 2.0,
        "wcp_on": 1,
        "wcp_pad_t": 2.0,
        "k_ins": 0.2,
    }
    values.update(overrides)
    return pd.DataFrame([values])


def _mesh_plan():
    return {
        "schema": thermal.THERMAL_MESH_PLAN_CONTRACT_VERSION,
        "policy": thermal.THERMAL_MESH_POLICY,
        "plan_sha256": "b" * 64,
        "operations": [
            {
                "name": "thin_mesh",
                "category": "test",
                "operation_type": "object_level",
                "level": 5,
                "objects": ["pad_a", "pad_b"],
                "shared_region": False,
                "separate_objects": True,
                "actual_operation_names": ["thin_mesh_L_5"],
            }
        ],
        "operation_count": 1,
        "assigned_object_count": 2,
        "required_thin_object_count": 2,
        "required_thin_objects": ["pad_a", "pad_b"],
        "required_objects_missing": [],
        "core_plate_assembly_count": 0,
        "wcp_assembly_count": 1,
        "wcp_pad_mesh_region_count": 0,
        "rx_retained_pack_count": 0,
        "shared_operation_count": 0,
        "object_level_operation_count": 1,
        "mesh_region_operation_count": 0,
        "separate_object_operation_count": 1,
    }


def _request(monkeypatch, **overrides):
    from module import aedt_pool_adapter

    monkeypatch.setenv(
        thermal.SYMMETRY_THERMAL_DIRECT_ANALYZE_ENV,
        thermal.SYMMETRY_THERMAL_DIRECT_ANALYZE_TOKEN,
    )
    with patch.object(
        aedt_pool_adapter, "pooled_backend_enabled", return_value=False
    ):
        return thermal._symmetry_thermal_direct_analyze_request(
            _standard_df(**overrides)
        )


def test_default_remains_native_premesh_when_env_is_absent(monkeypatch):
    monkeypatch.delenv(
        thermal.SYMMETRY_THERMAL_DIRECT_ANALYZE_ENV, raising=False
    )

    assert (
        thermal._symmetry_thermal_direct_analyze_request(_standard_df())
        is None
    )


def test_invalid_opt_in_token_fails_closed(monkeypatch):
    monkeypatch.setenv(
        thermal.SYMMETRY_THERMAL_DIRECT_ANALYZE_ENV, "true"
    )

    with pytest.raises(RuntimeError, match="invalid versioned opt-in token"):
        thermal._symmetry_thermal_direct_analyze_request(_standard_df())


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"thermal_symmetry": "full"}, "thermal_symmetry"),
        ({"full_model": 1}, "full_model"),
        ({"fan_config": "single"}, "fan_config"),
        ({"core_plate_on": 0}, "core_plate_on"),
        ({"wcp_on": 0}, "wcp_on"),
        ({"k_ins": 0.25}, "k_ins"),
        ({"fan_velocity": 2.0}, "fixed-boundary"),
        ({"core_plate_pad_t": 1.0}, "fixed-boundary"),
        ({"wcp_pad_t": 1.0}, "fixed-boundary"),
    ],
)
def test_opt_in_rejects_nonstandard_or_cooling_drift(
    monkeypatch, override, message
):
    with pytest.raises(RuntimeError, match=message):
        _request(monkeypatch, **override)


def test_direct_gate_skips_generate_mesh_and_seals_result_receipt(
    monkeypatch,
):
    from module import aedt_pool_adapter

    request = _request(monkeypatch)
    generate_mesh = Mock(
        side_effect=AssertionError("GenerateMesh must not be called")
    )
    native_ipk = SimpleNamespace(
        mesh=SimpleNamespace(generate_mesh=generate_mesh),
        modeler=SimpleNamespace(oeditor=SimpleNamespace()),
    )
    setup_readback = {
        "schema": (
            thermal.THERMAL_SETUP_CONTROL_READBACK_CONTRACT_VERSION
        ),
        "wrapper_passed": True,
        "native_complete": True,
    }
    identity = {
        "project": "thermal_test",
        "design": "icepak_thermal",
        "setups": ["ThermalSetup"],
        "setup_control_readback": setup_readback,
        "native_ipk": native_ipk,
        "native_design": SimpleNamespace(),
    }
    native_operation_readback = {
        "missing_operation_names": [],
        "required_thin_objects_missing": [],
    }
    with patch.object(
        aedt_pool_adapter, "pooled_backend_enabled", return_value=False
    ), patch.object(
        thermal, "_prepare_thermal_dispatch", return_value=identity
    ) as prepare, patch.object(
        thermal,
        "_native_thermal_mesh_operation_readback",
        return_value=native_operation_readback,
    ) as operation_readback:
        receipt = thermal._symmetry_thermal_direct_analyze_preflight(
            SimpleNamespace(),
            native_ipk,
            SimpleNamespace(),
            _mesh_plan(),
            request,
            {
                "thermal_conductivity_W_mK": 0.2,
                "electrical_conductivity_S_m": 0.0,
            },
        )

    prepare.assert_called_once()
    operation_readback.assert_called_once()
    generate_mesh.assert_not_called()
    assert receipt["direct_analyze_gate_passed"] is True
    assert receipt["generate_mesh_called"] is False
    assert receipt["analysis_dispatched_after_premesh"] is False
    assert (
        receipt["direct_analyze_contract"][
            "scientific_truth_gates_unchanged"
        ]
        is True
    )
    assert (
        receipt["direct_analyze_contract"][
            "thermal_iteration_settings_modified_by_fast_path"
        ]
        is False
    )

    receipt["analysis_dispatched_after_mesh_policy_gate"] = True
    metadata = thermal._thermal_mesh_result_metadata(
        _mesh_plan(), receipt
    )
    assert metadata["thermal_mesh_native_generation_passed"] == [0]
    assert metadata["thermal_mesh_direct_analyze_opt_in"] == [1]
    assert metadata["thermal_mesh_preflight_status"] == [
        thermal.SYMMETRY_THERMAL_DIRECT_ANALYZE_STATUS
    ]


def test_direct_result_receipt_tampering_fails_closed(monkeypatch):
    from module import aedt_pool_adapter

    request = _request(monkeypatch)
    native_ipk = SimpleNamespace(
        modeler=SimpleNamespace(oeditor=SimpleNamespace())
    )
    identity = {
        "setup_control_readback": {
            "schema": (
                thermal.THERMAL_SETUP_CONTROL_READBACK_CONTRACT_VERSION
            ),
            "wrapper_passed": True,
            "native_complete": True,
        },
        "native_ipk": native_ipk,
        "native_design": SimpleNamespace(),
    }
    with patch.object(
        aedt_pool_adapter, "pooled_backend_enabled", return_value=False
    ), patch.object(
        thermal, "_prepare_thermal_dispatch", return_value=identity
    ), patch.object(
        thermal,
        "_native_thermal_mesh_operation_readback",
        return_value={
            "missing_operation_names": [],
            "required_thin_objects_missing": [],
        },
    ):
        receipt = thermal._symmetry_thermal_direct_analyze_preflight(
            SimpleNamespace(),
            native_ipk,
            SimpleNamespace(),
            _mesh_plan(),
            request,
            {
                "thermal_conductivity_W_mK": 0.2,
                "electrical_conductivity_S_m": 0.0,
            },
        )

    receipt["analysis_dispatched_after_mesh_policy_gate"] = True
    receipt["direct_analyze_contract"][
        "scientific_truth_gates_unchanged"
    ] = False
    with pytest.raises(RuntimeError, match="pre-solve attestation"):
        thermal._thermal_mesh_result_metadata(_mesh_plan(), receipt)


def test_cli_exposes_explicit_direct_analyze_switch(monkeypatch):
    import run_simulation_260706 as runner

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_simulation_260706.py", "--symmetry-thermal-direct-analyze"],
    )
    args = runner.parse_args()

    assert args.symmetry_thermal_direct_analyze is True
