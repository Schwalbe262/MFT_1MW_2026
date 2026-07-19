from __future__ import annotations

import numpy as np
import pytest

from tools import tier1_resonance_feedback as feedback


CONSTRAINTS = (
    "Llt_robust_band",
    "temperature_robust_limit:T_max_core",
    "analytical_flux_density_limit",
    "decoded_space_shrink",
    "minimum_physical_insulation",
    "strict_full_density_support",
    "Llt_ensemble_disagreement",
    "core_group_manufacturability_limit",
    "half_magnetizing_resonance_minimum",
    "exterior_width_limit",
    "exterior_length_limit",
    "exterior_height_limit",
)


def test_optimizer_constraint_normalization_is_complete_and_dimensionless():
    contract = feedback.optimizer_constraint_normalization(
        CONSTRAINTS, feedback.HARD_SPEC,
    )

    assert contract["schema_version"] == (
        "mft-tier1-optimizer-constraint-normalization-v1"
    )
    assert contract["constraint_order"] == list(CONSTRAINTS)
    assert set(contract["scales"]) == set(CONSTRAINTS)
    assert all(value > 0.0 for value in contract["scales"].values())
    assert contract["scales"]["Llt_robust_band"] == 0.55
    assert contract["scales"]["half_magnetizing_resonance_minimum"] == 1_000.0
    assert contract["scales"]["exterior_width_limit"] == 60.0
    unsigned = dict(contract)
    digest = unsigned.pop("sha256")
    assert digest == feedback._json_sha(unsigned)


def test_install_optimizer_normalization_retains_physical_evaluator():
    class DummyProblem:
        constraint_names = CONSTRAINTS

        def _evaluate(self, X, out, *args, **kwargs):
            del args, kwargs
            rows = len(X)
            out["F"] = np.ones((rows, 2), dtype=float)
            out["G"] = np.tile(
                np.asarray([
                    0.55, 5.5, 0.12, 1.0, 4.0, 0.1,
                    1.1, 1.0, 1_000.0, 60.0, 60.0, 37.5,
                ]),
                (rows, 1),
            )
            out["frame"] = "physical-frame"

    problem = DummyProblem()
    physical_evaluate, contract = (
        feedback.install_optimizer_constraint_normalization(
            problem, feedback.HARD_SPEC,
        )
    )
    normalized = {}
    problem._evaluate(np.zeros((2, 1)), normalized)

    assert normalized["frame"] == "physical-frame"
    np.testing.assert_allclose(normalized["G"], np.ones((2, len(CONSTRAINTS))))

    physical = {}
    physical_evaluate(np.zeros((1, 1)), physical)
    expected = np.asarray([
        contract["scales"][name] for name in CONSTRAINTS
    ])[None, :]
    np.testing.assert_allclose(physical["G"], expected)


def test_optimizer_constraint_normalization_fails_closed_for_unknown_name():
    with pytest.raises(RuntimeError, match="undefined"):
        feedback.optimizer_constraint_normalization(
            ("unsealed_constraint",), feedback.HARD_SPEC,
        )


def test_resonance_focus_scale_is_optimizer_only_and_exactly_250_hz():
    class DummyProblem:
        constraint_names = CONSTRAINTS

        def _evaluate(self, X, out, *args, **kwargs):
            del args, kwargs
            physical = np.zeros((len(X), len(CONSTRAINTS)), dtype=float)
            physical[:, CONSTRAINTS.index(
                "half_magnetizing_resonance_minimum"
            )] = 500.0
            out["G"] = physical
            out["F"] = np.zeros((len(X), 2), dtype=float)

    problem = DummyProblem()
    physical_evaluate, contract = (
        feedback.install_optimizer_constraint_normalization(
            problem, feedback.HARD_SPEC, resonance_scale_hz=250.0,
        )
    )
    optimizer = {}
    problem._evaluate(np.zeros((1, 1)), optimizer)
    physical = {}
    physical_evaluate(np.zeros((1, 1)), physical)

    resonance_index = CONSTRAINTS.index(
        "half_magnetizing_resonance_minimum"
    )
    assert contract["scales"][
        "half_magnetizing_resonance_minimum"
    ] == 250.0
    assert optimizer["G"][0, resonance_index] == 2.0
    assert physical["G"][0, resonance_index] == 500.0
    default = feedback.optimizer_constraint_normalization(
        CONSTRAINTS, feedback.HARD_SPEC,
    )
    assert default["scales"][
        "half_magnetizing_resonance_minimum"
    ] == 1_000.0


def test_core_thermal_pressure_changes_only_three_optimizer_columns():
    core_constraints = feedback.CORE_THERMAL_PRESSURE_CONSTRAINTS
    names = (
        "Llt_robust_band",
        "temperature_robust_limit:T_max_Tx",
        *core_constraints,
        "temperature_robust_limit:Tprobe_core_center_leg_max",
        "half_magnetizing_resonance_minimum",
    )

    class DummyProblem:
        constraint_names = names

        def _evaluate(self, X, out, *args, **kwargs):
            del args, kwargs
            physical = np.tile(
                np.asarray([0.55, 5.5, 3.0, 4.0, 5.0, 5.5, 500.0]),
                (len(X), 1),
            )
            out["G"] = physical
            out["F"] = np.zeros((len(X), 2), dtype=float)
            out["physical_marker"] = "unchanged"

    problem = DummyProblem()
    physical_evaluate, contract = (
        feedback.install_optimizer_constraint_normalization(
            problem,
            feedback.HARD_SPEC,
            resonance_scale_hz=250.0,
            core_thermal_scale_c=1.0,
        )
    )
    optimizer = {}
    physical = {}
    X = np.zeros((1, 1))
    problem._evaluate(X, optimizer)
    physical_evaluate(X, physical)

    expected_physical = np.asarray(
        [[0.55, 5.5, 3.0, 4.0, 5.0, 5.5, 500.0]]
    )
    np.testing.assert_allclose(physical["G"], expected_physical)
    np.testing.assert_allclose(
        optimizer["G"],
        [[1.0, 1.0, 3.0, 4.0, 5.0, 1.0, 2.0]],
    )
    assert physical["physical_marker"] == optimizer["physical_marker"]
    assert contract["scales"]["temperature_robust_limit:T_max_Tx"] == 5.5
    assert contract["scales"][
        "temperature_robust_limit:Tprobe_core_center_leg_max"
    ] == 5.5
    assert contract["explicit_optimizer_only_scale_overrides"] == {
        name: 1.0 for name in core_constraints
    }
    assert contract["authoritative_terminal_G"] == "physical_unscaled"


def test_balanced_four_celsius_scale_preserves_physical_core_constraints():
    core_constraints = feedback.CORE_THERMAL_PRESSURE_CONSTRAINTS
    names = (*core_constraints, "half_magnetizing_resonance_minimum")

    class DummyProblem:
        constraint_names = names

        def _evaluate(self, X, out, *args, **kwargs):
            del args, kwargs
            out["G"] = np.tile(
                np.asarray([8.0, 4.0, -2.0, 500.0]),
                (len(X), 1),
            )

    problem = DummyProblem()
    physical_evaluate, contract = (
        feedback.install_optimizer_constraint_normalization(
            problem,
            feedback.HARD_SPEC,
            resonance_scale_hz=250.0,
            core_thermal_scale_c=4.0,
        )
    )
    optimizer = {}
    physical = {}
    X = np.zeros((1, 1))
    problem._evaluate(X, optimizer)
    physical_evaluate(X, physical)

    np.testing.assert_allclose(optimizer["G"], [[2.0, 1.0, -0.5, 2.0]])
    np.testing.assert_allclose(physical["G"], [[8.0, 4.0, -2.0, 500.0]])
    assert contract["explicit_optimizer_only_scale_overrides"] == {
        name: 4.0 for name in core_constraints
    }
    assert contract["authoritative_terminal_G"] == "physical_unscaled"


def test_core_thermal_pressure_requires_complete_sealed_three_constraint_set():
    with pytest.raises(RuntimeError, match="sealed three constraints"):
        feedback.optimizer_constraint_normalization(
            ("temperature_robust_limit:T_max_core",),
            feedback.HARD_SPEC,
            core_thermal_scale_c=1.0,
        )


def test_thermal_acquisition_adds_core_and_anisotropic_soft_axis_pressure_only():
    names = (
        *feedback.CORE_THERMAL_PRESSURE_CONSTRAINTS,
        "exterior_width_limit",
        "exterior_length_limit",
    )
    physical_g = np.asarray([
        [10.0, 11.0, 12.0, -100.0, -250.0],
        [-1.0, -2.0, -3.0, -250.0, -200.0],
    ])
    before = physical_g.copy()
    total, components, contract = (
        feedback.thermal_crossover_acquisition_pressure(
            physical_g, names, feedback.HARD_SPEC, 1.0,
        )
    )

    # W = -100 + 1200 = 1100 mm, so the independent W soft excess is 1.0.
    # L = -250 + 1200 = 950 mm, so the L pressure remains zero.
    np.testing.assert_allclose(total, [34.0, 0.0])
    np.testing.assert_allclose(
        components["core_thermal_positive_normalized"], [33.0, 0.0]
    )
    np.testing.assert_allclose(
        components["soft_width_positive_excess_normalized"], [1.0, 0.0]
    )
    np.testing.assert_allclose(
        components["soft_length_positive_excess_normalized"], [0.0, 0.0]
    )
    np.testing.assert_array_equal(physical_g, before)
    assert contract["authoritative_terminal_G"] == (
        "physical_unscaled_unchanged"
    )
    assert contract["hard_constraint_mutation"] is False
    assert contract["soft_axis_mode"] == (
        "independent_positive_excess_above_1000mm_no_new_hard_G"
    )
    unsigned = dict(contract)
    digest = unsigned.pop("sha256")
    assert digest == feedback._json_sha(unsigned)


def test_inactive_thermal_acquisition_is_zero_and_does_not_require_new_columns():
    physical_g = np.asarray([[2.0]])
    total, components, contract = (
        feedback.thermal_crossover_acquisition_pressure(
            physical_g,
            ("Llt_robust_band",),
            feedback.HARD_SPEC,
            None,
        )
    )
    np.testing.assert_allclose(total, [0.0])
    assert all(float(values[0]) == 0.0 for values in components.values())
    assert contract["active"] is False
    np.testing.assert_array_equal(physical_g, [[2.0]])


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), True])
def test_explicit_optimizer_resonance_scale_fails_closed(value):
    with pytest.raises(RuntimeError, match="finite and positive"):
        feedback.optimizer_constraint_normalization(
            CONSTRAINTS, feedback.HARD_SPEC, resonance_scale_hz=value,
        )


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), True])
def test_explicit_optimizer_core_thermal_scale_fails_closed(value):
    names = (*feedback.CORE_THERMAL_PRESSURE_CONSTRAINTS,)
    with pytest.raises(RuntimeError, match="finite and positive"):
        feedback.optimizer_constraint_normalization(
            names, feedback.HARD_SPEC, core_thermal_scale_c=value,
        )


def test_orthogonal_all_thermal_and_llt_pressure_is_optimizer_only():
    names = (
        "Llt_robust_band",
        *feedback.ALL_THERMAL_PRESSURE_CONSTRAINTS,
        "Llt_ensemble_disagreement",
        "half_magnetizing_resonance_minimum",
    )
    physical_row = np.asarray([
        0.5,
        *([4.0] * len(feedback.ALL_THERMAL_PRESSURE_CONSTRAINTS)),
        1.0,
        300.0,
    ])

    class DummyProblem:
        constraint_names = names

        def _evaluate(self, X, out, *args, **kwargs):
            del args, kwargs
            out["G"] = np.tile(physical_row, (len(X), 1))

    problem = DummyProblem()
    physical_evaluate, contract = (
        feedback.install_optimizer_constraint_normalization(
            problem,
            feedback.HARD_SPEC,
            resonance_scale_hz=150.0,
            llt_scale_uh=0.25,
            all_thermal_scale_c=2.0,
        )
    )
    optimizer = {}
    physical = {}
    X = np.zeros((1, 1))
    problem._evaluate(X, optimizer)
    physical_evaluate(X, physical)

    np.testing.assert_allclose(physical["G"], physical_row[None, :])
    np.testing.assert_allclose(
        optimizer["G"],
        [[2.0, *([2.0] * 11), 2.0, 2.0]],
    )
    expected_overrides = {
        "Llt_robust_band": 0.25,
        "Llt_ensemble_disagreement": 0.5,
        **{
            name: 2.0
            for name in feedback.ALL_THERMAL_PRESSURE_CONSTRAINTS
        },
    }
    assert contract["explicit_optimizer_only_scale_overrides"] == (
        expected_overrides
    )
    assert contract["optimizer_all_active_thermal_constraints"] == list(
        feedback.ALL_THERMAL_PRESSURE_CONSTRAINTS
    )
    assert contract["all_active_thermal_side_activation"] == (
        "finite_N2_side_gt_0_else_physical_negative_BIG"
    )
    assert contract["authoritative_terminal_G"] == "physical_unscaled"


def test_orthogonal_pressure_requires_all_eleven_thermal_and_both_llt_columns():
    missing_thermal = (
        "Llt_robust_band",
        *feedback.ALL_THERMAL_PRESSURE_CONSTRAINTS[:-1],
        "Llt_ensemble_disagreement",
    )
    with pytest.raises(RuntimeError, match="all-11 constraint set"):
        feedback.optimizer_constraint_normalization(
            missing_thermal,
            feedback.HARD_SPEC,
            all_thermal_scale_c=2.0,
        )
    with pytest.raises(RuntimeError, match="both sealed Llt constraints"):
        feedback.optimizer_constraint_normalization(
            (
                "Llt_robust_band",
                *feedback.ALL_THERMAL_PRESSURE_CONSTRAINTS,
            ),
            feedback.HARD_SPEC,
            llt_scale_uh=0.25,
        )
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        feedback.optimizer_constraint_normalization(
            (
                "Llt_robust_band",
                *feedback.ALL_THERMAL_PRESSURE_CONSTRAINTS,
                "Llt_ensemble_disagreement",
            ),
            feedback.HARD_SPEC,
            core_thermal_scale_c=2.0,
            all_thermal_scale_c=2.0,
        )


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), True])
def test_explicit_optimizer_orthogonal_scales_fail_closed(value):
    names = (
        "Llt_robust_band",
        *feedback.ALL_THERMAL_PRESSURE_CONSTRAINTS,
        "Llt_ensemble_disagreement",
    )
    with pytest.raises(RuntimeError, match="finite and positive"):
        feedback.optimizer_constraint_normalization(
            names, feedback.HARD_SPEC, llt_scale_uh=value,
        )
    with pytest.raises(RuntimeError, match="finite and positive"):
        feedback.optimizer_constraint_normalization(
            names, feedback.HARD_SPEC, all_thermal_scale_c=value,
        )
