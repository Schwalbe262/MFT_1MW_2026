from types import SimpleNamespace

import pytest

from module.fixed_boundary_contract import (
    FIXED_BOUNDARY_CONTRACT_SHA256,
    FIXED_BOUNDARY_CONTRACT_SCHEMA,
    FIXED_CORE_PLATE_PAD_THICKNESS_MM,
    FIXED_FAN_VELOCITY_M_S,
    FIXED_THERMAL_PAD_CONDUCTIVITY_W_MK,
    FIXED_WCP_PAD_THICKNESS_MM,
    FixedBoundaryContractError,
    attest_fixed_boundary,
    evaluate_fixed_boundary,
    fixed_boundary_classification_metadata,
    fixed_boundary_result_metadata,
)
from module.input_parameter_260706 import get_drawing_default_params
from run_simulation_260706 import Simulation


def _fixed_parameters():
    return {
        "fan_velocity": 1.5,
        "core_plate_pad_t": 2.0,
        "wcp_pad_t": 2.0,
    }


def test_exact_fixed_boundary_is_authoritative_and_durable():
    evidence = attest_fixed_boundary(
        _fixed_parameters(),
        thermal_pad_conductivity_w_mk=0.2,
    )
    metadata = fixed_boundary_result_metadata(evidence)

    assert evidence["schema"] == FIXED_BOUNDARY_CONTRACT_SCHEMA
    assert evidence["contract_sha256"] == (
        FIXED_BOUNDARY_CONTRACT_SHA256
    )
    assert evidence["mismatches"] == []
    assert evidence["authority_class"] == (
        "authoritative_fixed_boundary"
    )
    assert evidence["diagnostic_only"] is False
    assert metadata["fixed_boundary_authoritative_attested"] == 1
    assert metadata["fixed_boundary_diagnostic_only"] == 0
    assert metadata["fixed_boundary_promotion_forbidden"] == 0
    assert metadata["fixed_boundary_fan_velocity_m_s"] == 1.5
    assert metadata["fixed_boundary_core_plate_pad_t_mm"] == 2.0
    assert metadata["fixed_boundary_wcp_pad_t_mm"] == 2.0
    assert (
        metadata[
            "fixed_boundary_thermal_pad_conductivity_W_mK"
        ]
        == 0.2
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fan_velocity", 6.0),
        ("core_plate_pad_t", 1.0),
        ("wcp_pad_t", 1.0),
    ],
)
def test_deadline_overrides_are_diagnostic_only_and_rejected(
    field, value
):
    parameters = _fixed_parameters()
    parameters[field] = value

    evidence = evaluate_fixed_boundary(
        parameters,
        thermal_pad_conductivity_w_mk=0.2,
    )

    assert evidence["authority_class"] == "diagnostic_override_only"
    assert evidence["diagnostic_only"] is True
    assert evidence["promotion_forbidden_by_fixed_boundary"] is True
    assert (
        evidence[
            "canonical_dataset_mutation_forbidden_by_fixed_boundary"
        ]
        is True
    )
    assert [item["field"] for item in evidence["mismatches"]] == [
        field
    ]
    with pytest.raises(
        FixedBoundaryContractError,
        match="non-authoritative fixed-boundary candidate rejected",
    ):
        attest_fixed_boundary(
            parameters,
            thermal_pad_conductivity_w_mk=0.2,
        )


def test_deadline_tim_k3_is_diagnostic_only_and_cannot_emit_authority():
    evidence = evaluate_fixed_boundary(
        _fixed_parameters(),
        thermal_pad_conductivity_w_mk=3.0,
    )

    assert evidence["diagnostic_only"] is True
    assert evidence["mismatches"] == [{
        "field": "thermal_pad_conductivity_W_mK",
        "reason": "not_exact",
        "expected": 0.2,
        "observed": 3.0,
    }]
    metadata = fixed_boundary_classification_metadata(evidence)
    assert metadata["fixed_boundary_authoritative_attested"] == 0
    assert metadata["fixed_boundary_diagnostic_only"] == 1
    assert metadata["fixed_boundary_promotion_forbidden"] == 1
    assert (
        metadata[
            "fixed_boundary_canonical_dataset_mutation_forbidden"
        ]
        == 1
    )
    with pytest.raises(
        FixedBoundaryContractError,
        match="result metadata requires exact fixed-boundary attestation",
    ):
        fixed_boundary_result_metadata(evidence)


def test_missing_or_nonfinite_values_fail_closed():
    missing = evaluate_fixed_boundary(
        {"fan_velocity": 1.5},
        thermal_pad_conductivity_w_mk=float("nan"),
    )

    assert missing["diagnostic_only"] is True
    assert {
        item["field"] for item in missing["mismatches"]
    } == {
        "core_plate_pad_t",
        "wcp_pad_t",
        "thermal_pad_conductivity_W_mK",
    }


def test_defaults_are_wired_to_the_authoritative_contract():
    defaults = get_drawing_default_params()

    assert defaults["fan_velocity"] == FIXED_FAN_VELOCITY_M_S
    assert defaults["core_plate_pad_t"] == (
        FIXED_CORE_PLATE_PAD_THICKNESS_MM
    )
    assert defaults["wcp_pad_t"] == FIXED_WCP_PAD_THICKNESS_MM
    assert FIXED_THERMAL_PAD_CONDUCTIVITY_W_MK == 0.2


def test_maxwell_geometry_material_overwrites_deadline_tim_override():
    class Materials:
        def __init__(self):
            self.material_keys = {}

        def add_material(self, name):
            material = SimpleNamespace()
            self.material_keys[name] = material
            return material

        def __getitem__(self, name):
            return self.material_keys[name]

    materials = Materials()
    stale = materials.add_material("thermal_pad")
    stale.conductivity = 123.0
    stale.thermal_conductivity = 3.0
    sim = SimpleNamespace(
        design1=SimpleNamespace(materials=materials)
    )

    Simulation.create_thermal_pad_material(sim)

    assert stale.conductivity == 0
    assert stale.permittivity == 4
    assert stale.permeability == 1
    assert stale.thermal_conductivity == 0.2
