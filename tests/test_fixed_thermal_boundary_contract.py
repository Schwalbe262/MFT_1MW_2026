import pytest

from module.fixed_boundary_contract import (
    FixedBoundaryContractError,
    FIXED_CORE_PLATE_PAD_THICKNESS_MM,
    FIXED_FAN_VELOCITY_M_S,
    FIXED_THERMAL_PAD_CONDUCTIVITY_W_MK,
    FIXED_WCP_PAD_THICKNESS_MM,
    attest_fixed_boundary,
)
from module.input_parameter_260706 import get_drawing_default_params
from module import thermal_260706 as thermal
from tools import recover_full_postsolve_project as recovery


def test_input_and_fresh_icepak_share_authoritative_fixed_boundary():
    defaults = get_drawing_default_params()
    assert defaults["fan_velocity"] == FIXED_FAN_VELOCITY_M_S == 1.5
    assert defaults["wcp_pad_t"] == FIXED_WCP_PAD_THICKNESS_MM == 2.0
    assert (
        defaults["core_plate_pad_t"]
        == FIXED_CORE_PLATE_PAD_THICKNESS_MM
        == 2.0
    )
    assert (
        thermal.THERMAL_PAD_CONDUCTIVITY_W_MK
        == FIXED_THERMAL_PAD_CONDUCTIVITY_W_MK
        == 0.2
    )
    assert thermal.THERMAL_PAD_MATERIAL_POLICY == (
        "fixed_boundary_tim_k0p2_native_attested_0p2WmK_"
        "electrically_insulating_v1"
    )


def test_fixed_parameter_attestation_rejects_provisional_overrides():
    evidence = attest_fixed_boundary(
        {
            "fan_velocity": 1.5,
            "wcp_pad_t": 2.0,
            "core_plate_pad_t": 2.0,
        },
        thermal_pad_conductivity_w_mk=0.2,
    )
    assert evidence["authoritative_fixed_boundary_attested"] is True
    assert evidence["mismatches"] == []
    with pytest.raises(FixedBoundaryContractError, match="fan_velocity"):
        attest_fixed_boundary(
            {
                "fan_velocity": 6.0,
                "wcp_pad_t": 1.0,
                "core_plate_pad_t": 1.0,
            },
            thermal_pad_conductivity_w_mk=3.0,
        )


def test_historical_k3_source_is_separate_from_fresh_k0p2_rebuild():
    assert recovery.SOURCE_THERMAL_PAD_CONDUCTIVITY_W_MK == 3.0
    assert recovery.SOURCE_THERMAL_PAD_MATERIAL_POLICY.startswith(
        "deadline_tim_k3_"
    )
    assert recovery.THERMAL_PAD_CONDUCTIVITY_W_MK == 0.2
    assert recovery.THERMAL_PAD_MATERIAL_POLICY == (
        "fixed_boundary_tim_k0p2_native_attested_0p2WmK_"
        "electrically_insulating_v1"
    )
