import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np


REGRESSION_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = REGRESSION_ROOT.parent
for path in (str(REGRESSION_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from model_targets import SURROGATE_WINDING_COMPONENT_LOSS_TARGETS
from optimization.design_summary import (
    B_AREA_BASIS_DATASHEET_NET,
    B_AREA_BASIS_GROSS_WITH_LAMINATION,
    design_analytical_b_field_t,
    legacy_0p7_b_field_t,
    pareto_design_summary,
    rated_power_w,
)
from optimization.nsga2_problem import (
    MFTProblem,
    NSGA_FIXED_THERMAL_STACK_MM,
)
from optimization.run_nsga2 import REQUIRED_MODEL_TARGETS
from module.input_parameter_260706 import _SOBOL_DIMS
from training.checkpoint_train import TARGETS
from verify.finalize import REQUIRED_OPTIMIZATION_MODELS


def decoded_row(**updates):
    row = {
        "N1_main": 6,
        "N1_side": 0,
        "N2_main": 18,
        "N2_side": 42,
        "l1": 89.0,
        "l2": 236.5,
        "h1": 347.0,
        "w1": 530.0,
        "sl1_main_x": 600.0,
        "nwl1_main": 30.0,
        "sl1_main_y": 600.0,
        "nwb1_main_y": 30.0,
        "sl2_side_x": 120.0,
        "sl2_side_y": 600.0,
        "nwl1_side": 0.0,
        "nwl2_main": 20.0,
        "nwl2_side": 30.0,
        "nwh1": 280.0,
        "nwh2": 270.0,
        "cw1": 5.0,
        "cw2": 0.665,
        "gap1": 1.6,
        "gap2": 0.339,
        "n_core_group": 3,
        "core_depth_each": 150.0,
        "Ae_m2": 0.0801,
        "core_plate_t": 20.0,
        "core_plate_pad_t": 2.0,
        "wcp_t": 20.0,
        "wcp_pad_t": 2.0,
        "wcp_len_pct": 50.0,
        "wcp_len_x": 300.0,
        "P_target": 1_000_000.0,
        "V1_rms": 1000.0,
        "I1_rated": 1000.0,
        "V2_rms": 10_000.0,
        "I2_rated": 100.0,
        "freq": 1000.0,
    }
    row.update(updates)
    return row


def physical_predictions(**updates):
    values = {
        "Llt_phys": 27.9,
        "P_winding_total": 3000.0,
        "P_Tx_main_group": 1700.0,
        "P_Rx_main_group": 900.0,
        "P_Rx_side_total": 400.0,
        "P_core_total": 2000.0,
        "P_core_plate_total": 200.0,
        "P_wcp_total": 100.0,
    }
    values.update(updates)
    return values


class DesignSummaryTests(unittest.TestCase):
    def test_summary_uses_physical_component_models_and_active_spec(self):
        row = decoded_row()
        predictions = physical_predictions()
        summary = pareto_design_summary(
            row,
            predictions,
            5300.0,
            leakage_target_uH=31.25,
            core_lamination_factor=0.82,
            B_area_basis=B_AREA_BASIS_GROSS_WITH_LAMINATION,
        )

        self.assertEqual(summary["leakage_target_uH"], 31.25)
        self.assertEqual(summary["pred_primary_winding_loss_W"], 1700.0)
        self.assertEqual(
            summary["pred_secondary_winding_loss_W"], 1300.0
        )
        self.assertEqual(summary["pred_total_winding_loss_W"], 3000.0)
        self.assertEqual(summary["pred_total_loss_W"], 5300.0)
        self.assertEqual(summary["surrogate_output_basis"], "full_physical")
        self.assertNotIn("pred_flux_density_T", summary)
        self.assertTrue(math.isclose(
            summary["B_design_analytic_T"],
            design_analytical_b_field_t(
                row,
                core_lamination_factor=0.82,
                area_basis=B_AREA_BASIS_GROSS_WITH_LAMINATION,
            ),
            rel_tol=0.0,
            abs_tol=1e-15,
        ))
        self.assertTrue(math.isclose(
            summary["B_design_analytic_T"],
            1000.0 / (4.0 * 1000.0 * 6.0 * 0.0801 * 0.82),
            rel_tol=0.0,
            abs_tol=1e-15,
        ))
        self.assertTrue(math.isclose(
            summary["B_legacy_0p7_T"],
            legacy_0p7_b_field_t(row),
            rel_tol=0.0,
            abs_tol=1e-15,
        ))
        self.assertTrue(math.isclose(
            summary["B_legacy_0p7_T"],
            1000.0 / (
                4.0 * 1000.0 * 6.0
                * (2.0 * 0.530 * 0.089 * 0.7)
            ),
            rel_tol=0.0,
            abs_tol=1e-15,
        ))
        self.assertEqual(summary["B_design_waveform"], "bipolar_square")
        self.assertEqual(summary["B_denominator_coefficient"], 4.0)
        self.assertEqual(summary["Ae_m2"], 0.0801)
        self.assertEqual(summary["Ae_gross_m2"], 0.0801)
        self.assertTrue(math.isclose(
            summary["Ae_effective_m2"], 0.0801 * 0.82
        ))
        self.assertEqual(summary["core_lamination_factor"], 0.82)
        self.assertEqual(
            summary["B_area_basis"],
            B_AREA_BASIS_GROSS_WITH_LAMINATION,
        )
        self.assertEqual(summary["turns_primary"], 6)
        self.assertEqual(summary["turns_secondary_center"], 18)
        self.assertEqual(summary["turns_secondary_side"], 42)
        self.assertEqual(summary["core_cold_plate_thickness_mm"], 20.0)
        self.assertEqual(summary["core_thermal_pad_thickness_mm"], 2.0)
        self.assertEqual(summary["winding_cold_plate_thickness_mm"], 20.0)
        self.assertEqual(summary["winding_thermal_pad_thickness_mm"], 2.0)
        self.assertEqual(summary["wcp_len_pct"], 50.0)
        self.assertEqual(summary["wcp_len_x_mm"], 300.0)
        self.assertEqual(summary["cw1_conductor_thickness_mm"], 5.0)
        self.assertEqual(summary["cw2_conductor_thickness_mm"], 0.665)
        self.assertEqual(summary["nwl2_side_pack_width_mm"], 30.0)
        self.assertEqual(summary["nwh1_winding_height_mm"], 280.0)
        self.assertEqual(summary["core_depth_each_mm"], 150.0)
        self.assertEqual(summary["n_core_group"], 3)
        self.assertTrue(math.isclose(
            summary["pred_efficiency_pct"],
            1_000_000.0 / 1_005_300.0 * 100.0,
        ))

    def test_summary_does_not_require_bmax_surrogate_prediction(self):
        summary = pareto_design_summary(
            decoded_row(),
            physical_predictions(),
            5300.0,
            leakage_target_uH=27.5,
            core_lamination_factor=0.82,
            B_area_basis=B_AREA_BASIS_GROSS_WITH_LAMINATION,
        )
        self.assertGreater(summary["B_design_analytic_T"], 0.0)
        self.assertGreater(summary["B_legacy_0p7_T"], 0.0)

    def test_summary_fails_instead_of_fabricating_missing_physics(self):
        missing_component = physical_predictions()
        missing_component.pop("P_Rx_main_group")
        with self.assertRaises(KeyError):
            pareto_design_summary(
                decoded_row(),
                missing_component,
                5300.0,
                leakage_target_uH=27.5,
                core_lamination_factor=0.82,
                B_area_basis=B_AREA_BASIS_GROSS_WITH_LAMINATION,
            )
        with self.assertRaises(ValueError):
            pareto_design_summary(
                decoded_row(),
                physical_predictions(),
                9999.0,
                leakage_target_uH=27.5,
                core_lamination_factor=0.82,
                B_area_basis=B_AREA_BASIS_GROSS_WITH_LAMINATION,
            )
        with self.assertRaises(ValueError):
            pareto_design_summary(
                decoded_row(N1_side=1),
                physical_predictions(),
                5300.0,
                leakage_target_uH=27.5,
                core_lamination_factor=0.82,
                B_area_basis=B_AREA_BASIS_GROSS_WITH_LAMINATION,
            )

    def test_datasheet_net_area_basis_does_not_apply_factor_twice(self):
        row = decoded_row()
        value = design_analytical_b_field_t(
            row,
            core_lamination_factor=0.82,
            area_basis=B_AREA_BASIS_DATASHEET_NET,
        )
        expected = 1000.0 / (4.0 * 1000.0 * 6.0 * row["Ae_m2"])
        self.assertTrue(math.isclose(value, expected))

    def test_rated_power_uses_port_rating_only_when_target_is_zero(self):
        self.assertEqual(
            rated_power_w(decoded_row(
                P_target=0.0,
                V1_rms=900.0,
                I1_rated=1000.0,
                V2_rms=10_000.0,
                I2_rated=100.0,
            )),
            900_000.0,
        )


class OptimizationContractTests(unittest.TestCase):
    def test_component_targets_are_required_end_to_end(self):
        thresholds = json.loads(
            (REGRESSION_ROOT / "training" / "model_quality_thresholds.json")
            .read_text(encoding="utf-8")
        )["targets"]
        for target in SURROGATE_WINDING_COMPONENT_LOSS_TARGETS:
            with self.subTest(target=target):
                self.assertIn(target, TARGETS)
                self.assertIn(target, thresholds)
                self.assertIn(target, REQUIRED_MODEL_TARGETS)
                self.assertIn(target, REQUIRED_OPTIMIZATION_MODELS)

    def test_optimizer_fixes_plate_and_pad_layers_separately(self):
        models = {target: object() for target in REQUIRED_MODEL_TARGETS}
        problem = MFTProblem(models, density_gate=lambda frame: np.zeros(len(frame)))
        self.assertEqual(problem.fixed_overrides, NSGA_FIXED_THERMAL_STACK_MM)
        for name in ("core_plate_t", "wcp_t"):
            index = next(
                i for i, (dimension, _, _) in enumerate(
                    _SOBOL_DIMS
                ) if dimension == name
            )
            self.assertEqual(problem.xl[index], 0.5)
            self.assertEqual(problem.xu[index], 0.5)
        self.assertEqual(problem.fixed_overrides["core_plate_pad_t"], 2.0)
        self.assertEqual(problem.fixed_overrides["wcp_pad_t"], 2.0)
        decoded, _, _ = problem.decode_batch(
            np.random.default_rng(0).random((1, problem.n_var))
        )
        self.assertEqual(float(decoded.iloc[0]["core_plate_t"]), 20.0)
        self.assertEqual(float(decoded.iloc[0]["wcp_t"]), 20.0)
        self.assertEqual(float(decoded.iloc[0]["core_plate_pad_t"]), 2.0)
        self.assertEqual(float(decoded.iloc[0]["wcp_pad_t"]), 2.0)

    def test_optimizer_rejects_conflicting_plate_override(self):
        models = {target: object() for target in REQUIRED_MODEL_TARGETS}
        with self.assertRaises(ValueError):
            MFTProblem(
                models,
                density_gate=lambda frame: np.zeros(len(frame)),
                fixed_overrides={"core_plate_t": 25.0},
            )


if __name__ == "__main__":
    unittest.main()
