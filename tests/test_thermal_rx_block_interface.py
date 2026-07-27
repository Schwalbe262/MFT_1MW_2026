from types import SimpleNamespace
from unittest.mock import Mock
import json

from module import thermal_260706 as thermal


def _case_text(*faces):
    return (
        "(cfd-post-mesh-info ((0 0 "
        "(rx_main_block_xn_solid rx_main_block_yp_solid region_fluid) "
        + " ".join(faces)
        + ")))\n"
    )


def test_unpaired_retained_rx_interfaces_are_rejected(tmp_path):
    path = tmp_path / "current.nc_cas"
    path.write_text(
        _case_text(
            "(rx_main_block_xn_side wall "
            "rx_main_block_xn_solid region_fluid)",
            "(rx_main_block_yp_side wall "
            "rx_main_block_yp_solid region_fluid)",
            "(interf153 wall rx_main_block_xn_solid)",
            "(interf150 wall rx_main_block_yp_solid)",
        ),
        encoding="utf-8",
    )

    coverage = thermal._parse_thermal_rx_block_interface_case(
        path, ["Rx_main_block_xn", "Rx_main_block_yp"]
    )

    assert coverage["passed"] is False
    assert coverage["rx_main_adjacency_passed"] is False
    assert coverage["unpaired_interfaces"] == ["interf150", "interf153"]


def test_exact_6x60_unpaired_rx_main_forensic_signature_is_rejected(
    tmp_path,
):
    path = tmp_path / "ThermalSetup.nc_cas"
    path.write_text(
        _case_text(
            "(rx_main_block_yp_side_1_shadow wall "
            "rx_main_block_xn_solid rx_main_block_yp_solid)",
            "(rx_main_block_yp_side_1 wall "
            "rx_main_block_yp_solid rx_main_block_xn_solid)",
            "(rx_main_block_yp_side wall rx_main_block_yp_solid)",
            "(rx_main_block_xn_side wall rx_main_block_xn_solid)",
            "(interf160 wall rx_main_block_xn_solid)",
            "(interf158 wall rx_main_block_yp_solid)",
        ),
        encoding="utf-8",
    )

    coverage = thermal._parse_thermal_rx_block_interface_case(
        path, ["Rx_main_block_xn", "Rx_main_block_yp"]
    )

    assert coverage == {
        "schema": thermal.THERMAL_RX_BLOCK_INTERFACE_CONTRACT_VERSION,
        "passed": False,
        "expected_rx_main_object_count": 2,
        "expected_rx_main_objects": [
            "Rx_main_block_xn",
            "Rx_main_block_yp",
        ],
        "missing_rx_main_solids": [],
        "missing_fluid_coupling": [
            "Rx_main_block_xn",
            "Rx_main_block_yp",
        ],
        "rx_main_adjacency_component_count": 1,
        "rx_main_adjacency_components": [[
            "rx_main_block_xn_solid",
            "rx_main_block_yp_solid",
        ]],
        "rx_main_adjacency_passed": True,
        "unpaired_interfaces": ["interf158", "interf160"],
        "native_wall_record_count": 6,
    }


def test_paired_rx_fluid_and_block_coverage_is_accepted(tmp_path):
    path = tmp_path / "current.nc_cas"
    path.write_text(
        _case_text(
            "(rx_main_block_xn_side wall "
            "rx_main_block_xn_solid region_fluid)",
            "(rx_main_block_xn_side_shadow wall "
            "region_fluid rx_main_block_xn_solid)",
            "(rx_main_block_yp_side wall "
            "rx_main_block_yp_solid region_fluid)",
            "(rx_main_block_yp_side_shadow wall "
            "region_fluid rx_main_block_yp_solid)",
            "(rx_main_block_yp_side_1 wall "
            "rx_main_block_yp_solid rx_main_block_xn_solid)",
            "(rx_main_block_yp_side_1_shadow wall "
            "rx_main_block_xn_solid rx_main_block_yp_solid)",
        ),
        encoding="utf-8",
    )

    coverage = thermal._parse_thermal_rx_block_interface_case(
        path, ["Rx_main_block_xn", "Rx_main_block_yp"]
    )

    assert coverage["passed"] is True
    assert coverage["missing_fluid_coupling"] == []
    assert coverage["rx_main_adjacency_component_count"] == 1
    assert coverage["unpaired_interfaces"] == []


def test_large_native_case_is_parsed_from_bounded_prefix(tmp_path):
    path = tmp_path / "ThermalSetup.nc_cas"
    payload = _case_text(
        "(rx_main_block_xn_side wall "
        "rx_main_block_xn_solid region_fluid)",
        "(rx_main_block_yp_side wall "
        "rx_main_block_yp_solid region_fluid)",
        "(rx_main_bridge wall "
        "rx_main_block_xn_solid rx_main_block_yp_solid)",
    ).encode("utf-8")
    with path.open("wb") as stream:
        stream.write(payload)
        stream.seek(80 * 1024 * 1024)
        stream.write(b"\0")

    coverage = thermal._parse_thermal_rx_block_interface_case(
        path, ["Rx_main_block_xn", "Rx_main_block_yp"]
    )

    assert path.stat().st_size > 64 * 1024 * 1024
    assert coverage["passed"] is True


def test_predispatch_receipt_seals_shared_readback_and_stays_pending(
    capsys,
):
    plan = {
        "schema": thermal.THERMAL_MESH_PLAN_CONTRACT_VERSION,
        "policy": thermal.THERMAL_MESH_POLICY,
        "plan_sha256": "a" * 64,
        "rx_main_block_objects": [
            "Rx_main_block_xn",
            "Rx_main_block_yp",
        ],
        "rx_block_shared_pack_count": 1,
        "operations": [{
            "name": "rx_main_block_mesh_level",
            "category": "Rx_main_blocks",
            "objects": [
                "Rx_main_block_xn",
                "Rx_main_block_yp",
            ],
            "shared_region": True,
            "separate_objects": False,
            "actual_operation_names": ["rx_main_block_mesh_level"],
        }],
    }
    preflight = {
        "status": thermal.SYMMETRY_THERMAL_DIRECT_ANALYZE_STATUS,
        "passed": False,
        "generate_mesh_returned": False,
        "direct_analyze_gate_passed": True,
        "native_operation_readback_passed": True,
        "native_operation_readback": {
            "missing_operation_names": [],
            "required_thin_objects_missing": [],
            "operation_readbacks": [{
                "name": "rx_main_block_mesh_level",
                "separate_objects": False,
            }],
        },
    }
    df = thermal.pd.DataFrame([{
        "fan_config": "dual",
        "fan_velocity": 1.5,
        "core_plate_pad_t": 2.0,
        "wcp_pad_t": 2.0,
    }])

    receipt = thermal._thermal_rx_interface_predispatch_receipt(
        plan,
        preflight,
        df,
        {"thermal_conductivity_W_mK": 0.2},
    )
    thermal._emit_thermal_rx_interface_predispatch_receipt(receipt)

    line = capsys.readouterr().out.strip()
    prefix = "THERMAL_RX_INTERFACE_PREFLIGHT_JSON="
    assert line.startswith(prefix)
    emitted = json.loads(line[len(prefix):])
    assert emitted["passed"] is True
    assert emitted["mesh_policy"] == thermal.THERMAL_MESH_POLICY
    assert emitted["rx_main_shared_operations"][0][
        "native_separate_objects"
    ] is False
    assert emitted["terminal_interface_coverage_pending"] is True
    assert emitted["thermal_rx_main_interface_coverage_passed"] is False
    assert emitted["thermal_result_scientific_valid"] is False


def test_exact_eighth_rx_main_topology_uses_one_shared_mesh_operation():
    operations = []

    def assign_mesh_level(levels, name):
        level = next(iter(set(levels.values())))
        operation = SimpleNamespace(
            name=name,
            props={"Objects": list(levels), "Level": str(level)},
            auto_update=True,
            update=Mock(return_value=True),
        )
        operations.append(operation)
        mesh.meshoperations.append(operation)
        return [name]

    mesh = SimpleNamespace(
        assign_mesh_level=Mock(side_effect=assign_mesh_level),
        meshoperations=[],
    )
    objs = {
        "core_plates": [],
        "core_pads": [],
        "wcp_plates": [],
        "wcp_pads": [],
        "Tx": [],
        "Rx_main_blocks": [
            SimpleNamespace(name="Rx_main_block_xn"),
            SimpleNamespace(name="Rx_main_block_yp"),
        ],
        "Rx_side_blocks": [],
        "Rx_side2_blocks": [],
        "Rx_main_explicit": [],
        "Rx_side_explicit": [],
        "Rx_side2_explicit": [],
        "Rx_main_insulation": [],
        "Rx_side_insulation": [],
        "Rx_side2_insulation": [],
    }

    plan = thermal._assign_thermal_mesh(
        SimpleNamespace(mesh=mesh), objs, mode="eighth"
    )
    operation = plan["operations"][0]

    assert plan["rx_main_block_objects"] == [
        "Rx_main_block_xn",
        "Rx_main_block_yp",
    ]
    assert plan["rx_block_shared_pack_count"] == 1
    assert plan["shared_operation_count"] == 1
    assert operation["name"] == "rx_main_block_mesh_level"
    assert operation["objects"] == plan["rx_main_block_objects"]
    assert operation["level"] == 5
    assert operation["shared_region"] is True
    assert operation["separate_objects"] is False
    assert (
        operations[0].props["Mesh Object(s) Separately Enabled"]
        is False
    )
