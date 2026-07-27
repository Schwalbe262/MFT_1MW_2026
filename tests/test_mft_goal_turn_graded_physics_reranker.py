from __future__ import annotations

import math

import numpy as np
import pytest

from tools import mft_goal_turn_graded_physics_reranker as reranker


def _geometry(gap2=1.8):
    return {
        "N1_main": 6,
        "N1_side": 0,
        "N2_main": 37,
        "N2_side": 23,
        "cw2": 0.9,
        "gap2": gap2,
        "nwh2": 500.0,
        "h1": 600.0,
        "l2": 350.0,
        "sl2_main_x": 220.0,
        "sl2_main_y": 620.0,
        "sl2_side_x": 180.0,
        "sl2_side_y": 580.0,
        "cc_w2c_space_x": 40.0,
        "cc_w2c_space_y": 40.0,
        "w2c_w1c_space_x": 40.0,
        "w2c_w1c_space_y": 40.0,
        "w2s_w1s_space_x": 40.0,
        "w1s_cs_space_x": 40.0,
        "w1c_w2s_gap_actual": 50.0,
        "h_gap2": 50.0,
    }


def _zero_delta_model():
    feature_count = len(reranker.DELTA_FEATURE_NAMES)
    return {
        "payload_sha256": "unit-test-model",
        "delta_model": {
            "coefficients_intercept_then_standardized_features": (
                [0.0] * (feature_count + 1)
            ),
            "feature_mean": [0.0] * feature_count,
            "feature_scale": [1.0] * feature_count,
            "feature_minimum": [-1.0e9] * feature_count,
            "feature_maximum": [1.0e9] * feature_count,
        },
        "calibration_metrics": {
            "loo_q90_abs_log_error": 0.0,
        },
    }


def _approved_stack(thickness_mm=0.3, relative_permittivity=4.0):
    return reranker._seal(  # noqa: SLF001
        {
            "schema_version": (
                reranker.APPROVED_DIELECTRIC_STACK_SCHEMA
            ),
            "case_id": "unit-test-approved-case",
            "gap_fill_policy": "approved_layers_plus_residual_air",
            "approval": {
                "status": "approved",
                "authority_kind": "project_approved_dielectric_stack",
                "scope": "secondary_interturn_gap_sensitivity",
                "approved_by": "unit-test-approver",
                "approved_at": "2026-07-27T00:00:00+00:00",
                "authority_reference": "unit-test://stack-approval",
            },
            "dielectric_layers": [
                {
                    "material_id": "unit-test-material",
                    "thickness_mm": thickness_mm,
                    "relative_permittivity": relative_permittivity,
                    "relative_permittivity_authority": {
                        "authority_kind": (
                            "project_approved_material_property"
                        ),
                        "reference": "unit-test://material-property",
                    },
                }
            ],
        }
    )


def test_physics_network_uses_60_turn_midpoint_schedule_without_raw_c():
    result = reranker.physics_energy_network(_geometry())
    assert result["turn_voltage_schedule"]["turn_count"] == 60
    assert result["turn_voltage_schedule"]["section_order"] == ["main", "side"]
    assert result["turn_voltage_schedule"]["section_polarities"] == {
        "main": 1,
        "side": 1,
    }
    assert result["physics_Ceq_F"] > 0.0
    assert result["raw_two_net_capacitance_used"] is False
    assert result["single_truth_transfer_ratio_used"] is False
    assert result["physical_feasibility_authority"] is False
    assert result["parallel_plate_fringe_coefficient"] is None
    assert (
        result["fringing_role"]
        == "calibrated_residual_not_fixed_truth"
    )


def test_physics_network_accepts_minimum_one_side_turn_endpoint():
    geometry = _geometry()
    geometry["N2_main"] = 59
    geometry["N2_side"] = 1
    result = reranker.physics_energy_network(geometry)

    assert result["turn_voltage_schedule"]["turn_count"] == 60
    assert result["delta_features"]["side_turn_fraction"] == pytest.approx(
        1.0 / 60.0
    )
    assert math.isfinite(result["physics_Ceq_F"])
    assert result["physics_Ceq_F"] > 0.0


def test_smaller_interturn_gap_increases_adjacent_energy_component():
    tight = reranker.physics_energy_network(_geometry(0.5))
    loose = reranker.physics_energy_network(_geometry(2.0))
    key = "adjacent_turn_energy_Ceq_F"
    assert tight["components"][key] > loose["components"][key]


def test_fringe_geometry_kernel_is_explicit_and_has_no_fixed_coefficient():
    tight = reranker.physics_energy_network(_geometry(0.5))
    loose = reranker.physics_energy_network(_geometry(2.0))
    key = "adjacent_edge_to_area_gap_aspect"
    tight_kernel = tight["component_breakdown"][key]
    loose_kernel = loose["component_breakdown"][key]
    assert tight_kernel > 0.0
    assert loose_kernel > tight_kernel
    feature = "log1p_adjacent_edge_to_area_gap_aspect"
    assert tight["delta_features"][feature] == pytest.approx(
        math.log1p(tight_kernel)
    )


def test_symmetric_energy_restores_two_physical_side_sections():
    result = reranker.physics_energy_network(_geometry())
    breakdown = result["component_breakdown"]
    assert breakdown["section_physical_multiplicity"] == {
        "main": 1.0,
        "side": 2.0,
    }
    for family in (
        "adjacent_turn",
        "axial_ground",
        "inner_ground",
        "outer_ground",
    ):
        retained = breakdown[f"{family}_retained_by_section_F"]
        physical = breakdown[f"{family}_physical_by_section_F"]
        assert physical["main"] == retained["main"]
        assert physical["side"] == 2.0 * retained["side"]
    assert (
        breakdown["main_side_bridge_physical_F"]
        == 2.0 * breakdown["main_side_bridge_retained_F"]
    )
    assert result["dielectric_material_basis"] == "air_only"
    assert result["dielectric_stack_sensitivity_required"] is True


def test_dielectric_sensitivity_is_fail_closed_without_approved_stack():
    with pytest.raises(reranker.RerankError, match="fail-closed"):
        reranker.dielectric_stack_sensitivity(
            _geometry(),
            _zero_delta_model(),
            None,
        )


def test_dielectric_sensitivity_uses_explicit_series_stack_formula():
    geometry = _geometry()
    model = _zero_delta_model()
    stack = _approved_stack()
    network = reranker.physics_energy_network(geometry)
    result = reranker.dielectric_stack_sensitivity(
        geometry,
        model,
        stack,
    )
    expected_equivalent_air = 1.5 + 0.3 / 4.0
    expected_kappa = 1.8 / expected_equivalent_air
    adjacent = network["components"]["adjacent_turn_energy_Ceq_F"]
    expected_stack_ucb = (
        network["physics_Ceq_F"]
        + (expected_kappa - 1.0) * adjacent
    )
    assert result["residual_air_thickness_mm"] == pytest.approx(1.5)
    assert result["equivalent_air_thickness_mm"] == pytest.approx(
        expected_equivalent_air
    )
    assert result["kappa"] == pytest.approx(expected_kappa)
    assert result["conservative_fringe_factor"] == pytest.approx(1.0)
    assert result["C_stack_q90_ucb_F"] == pytest.approx(
        expected_stack_ucb
    )
    assert result["example_or_default_permittivity_used"] is False
    assert result["fixed_fringe_coefficient_used"] is False
    assert result["physical_feasibility_authority"] is False


def test_dielectric_sensitivity_rejects_unapproved_stack():
    unsigned = _approved_stack()
    unsigned.pop("payload_sha256")
    unsigned["approval"]["status"] = "draft"
    draft = reranker._seal(unsigned)  # noqa: SLF001
    with pytest.raises(reranker.RerankError, match="not approved"):
        reranker.dielectric_stack_sensitivity(
            _geometry(),
            _zero_delta_model(),
            draft,
        )


def test_ridge_log_delta_recovers_smooth_positive_predictions():
    x = np.asarray(
        [[math.log(value), value / 10.0] for value in range(1, 13)],
        dtype=float,
    )
    true_beta = np.asarray([0.2, -0.1, 0.3])
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    target = np.column_stack(
        [np.ones(len(x)), (x - mean) / scale]
    ) @ true_beta
    coefficients, fitted_mean, fitted_scale = reranker._ridge_fit(  # noqa: SLF001
        x,
        target,
        alpha=1e-9,
    )
    prediction = reranker._predict_delta(  # noqa: SLF001
        x,
        coefficients,
        fitted_mean,
        fitted_scale,
    )
    assert np.max(np.abs(prediction - target)) < 1e-8
