import math
from pathlib import Path
import sys
import unittest

import pandas as pd


REGRESSION_ROOT = Path(__file__).resolve().parent.parent
if str(REGRESSION_ROOT) not in sys.path:
    sys.path.insert(0, str(REGRESSION_ROOT))

from optimization.resonance import derive_resonances, validate_resonances


def _predictions():
    return {
        "Llt_phys": 30.0,
        "k": 0.9,
        "C_tx_tx_F": 12.0e-9,
        "C_rx_rx_F": 1.2e-9,
        "C_tx_rx_F": 0.5e-9,
    }


def _params():
    return {
        "N1_main": 6,
        "N1_side": 0,
        "N2_main": 18,
        "N2_side": 42,
    }


class DeriveResonanceTests(unittest.TestCase):
    def test_uses_leakage_coupling_and_physical_turns_ratio(self):
        predictions = _predictions()
        actual = derive_resonances(predictions, _params())

        leakage_h = predictions["Llt_phys"] * 1e-6
        tx_self_h = leakage_h / (1.0 - predictions["k"] ** 2)
        rx_self_h = tx_self_h * (60.0 / 6.0) ** 2
        expected = {
            "f_res_tx_self_Hz": 1.0
            / (2.0 * math.pi * math.sqrt(
                tx_self_h * predictions["C_tx_tx_F"]
            )),
            "f_res_rx_self_Hz": 1.0
            / (2.0 * math.pi * math.sqrt(
                rx_self_h * predictions["C_rx_rx_F"]
            )),
            "f_res_interwinding_Hz": 1.0
            / (2.0 * math.pi * math.sqrt(
                leakage_h * predictions["C_tx_rx_F"]
            )),
        }
        self.assertEqual(set(actual), set(expected))
        for name in expected:
            self.assertAlmostEqual(actual[name], expected[name], places=12)

    def test_accepts_derived_total_turn_fields(self):
        split = derive_resonances(_predictions(), _params())
        totals = derive_resonances(_predictions(), {"N1": 6, "N2": 60})
        self.assertEqual(split, totals)

    def test_rejects_nonphysical_inputs(self):
        for updates in (
            {"Llt_phys": 0.0},
            {"k": 1.0},
            {"k": -0.01},
            {"C_tx_rx_F": 0.0},
        ):
            with self.subTest(updates=updates):
                predictions = {**_predictions(), **updates}
                with self.assertRaises(ValueError):
                    derive_resonances(predictions, _params())


class ValidateResonanceTests(unittest.TestCase):
    def _row(self, solver_scale=1.0):
        predictions = _predictions()
        solver = derive_resonances(predictions, _params())
        return {
            **_params(),
            "cap_on": 1,
            "full_model": 0,
            # Exercise the same eighth-to-full conversion as checkpoint training.
            "Llt": predictions["Llt_phys"] / 2.0,
            "k": predictions["k"],
            "C_tx_tx_F": predictions["C_tx_tx_F"],
            "C_rx_rx_F": predictions["C_rx_rx_F"],
            "C_tx_rx_F": predictions["C_tx_rx_F"],
            **{name: value / solver_scale for name, value in solver.items()},
        }

    def test_exact_rows_report_zero_error_and_skip_cap_off(self):
        frame = pd.DataFrame([
            self._row(),
            {"cap_on": 0},
        ])
        report = validate_resonances(frame, assert_threshold=True)

        self.assertTrue(report["passed"])
        self.assertEqual(report["rows_total"], 2)
        self.assertEqual(report["rows_eligible"], 1)
        self.assertEqual(report["rows_valid"], 1)
        self.assertEqual(report["combined"]["median_relative_error"], 0.0)
        for output in report["outputs"].values():
            self.assertEqual(output["median_relative_error"], 0.0)

    def test_configurable_threshold_reports_and_asserts_failure(self):
        # Dividing the solver value by 1.1 gives exactly 10% relative error.
        frame = pd.DataFrame([self._row(solver_scale=1.1)])

        failed = validate_resonances(
            frame,
            median_relative_error_threshold=0.05,
        )
        self.assertFalse(failed["passed"])
        self.assertEqual(set(failed["failed_outputs"]), set(failed["outputs"]))
        with self.assertRaisesRegex(AssertionError, "required < 5%"):
            validate_resonances(
                frame,
                median_relative_error_threshold=0.05,
                assert_threshold=True,
            )

        passed = validate_resonances(
            frame,
            median_relative_error_threshold=0.11,
            assert_threshold=True,
        )
        self.assertTrue(passed["passed"])


if __name__ == "__main__":
    unittest.main()
