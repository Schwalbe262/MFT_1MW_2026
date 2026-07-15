import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import checkpoint_train
from predictor import EnsemblePredictor, RELATIVE_SIGMA_FLOOR_POLICY
import train_models


class ConstantTransformedModel:
    def __init__(self, value):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value, dtype=float)


class RelativeMetricContractTests(unittest.TestCase):
    def test_capacitance_bundle_uses_scale_aware_uncertainty_floor(self):
        self.assertEqual(
            train_models.SIGMA_FLOOR_POLICY,
            RELATIVE_SIGMA_FLOOR_POLICY,
        )
        transformed = math.log(1e-10)
        base_bundle = {
            "models": [("constant", ConstantTransformedModel(transformed))],
            "features": ["x"],
            "transform": "log",
            "q90": 1.0,
        }
        frame = pd.DataFrame({"x": [0.0]})

        _, legacy_sigma = EnsemblePredictor(base_bundle).predict_mu_sigma(frame)
        _, cap_sigma = EnsemblePredictor({
            **base_bundle,
            "sigma_floor_policy": RELATIVE_SIGMA_FLOOR_POLICY,
        }).predict_mu_sigma(frame)

        self.assertEqual(float(legacy_sigma[0]), 1e-9)
        self.assertGreater(float(cap_sigma[0]), 0.0)
        self.assertLess(float(cap_sigma[0]), 1e-20)

    def test_log_transform_preserves_subnanofarad_scale(self):
        values = np.array([1e-10, 3e-10, 1e-9, 1e-7])

        restored = checkpoint_train.inverse_y(
            checkpoint_train.transform_y(values, "log"), "log"
        )

        np.testing.assert_allclose(restored, values, rtol=1e-12, atol=0.0)
        self.assertEqual(len(np.unique(checkpoint_train.transform_y(values, "log"))), 4)

    def test_capacitance_relative_metrics_include_all_positive_values(self):
        actual = np.array([1e-10, 3e-10, 1e-9])
        error = actual * 0.1

        metrics = checkpoint_train.relative_error_summary(
            actual,
            error,
            tolerance=checkpoint_train.CAPACITANCE_RELATIVE_METRIC_TOLERANCE,
        )

        self.assertEqual(metrics["mape_n"], 3)
        self.assertEqual(metrics["mape_excluded_zero_count"], 0)
        self.assertEqual(metrics["mape_zero_abs_tolerance"], 0.0)
        self.assertAlmostEqual(metrics["mape_pct"], 10.0)

    def test_mape_excludes_zero_and_near_zero_but_keeps_other_rows(self):
        actual = np.array([0.0, 0.5e-9, 2.0e-9, -2.0, 4.0])
        error = np.array([1000.0, 1000.0, 2.0e-9, 1.0, 4.0])

        metrics = checkpoint_train.relative_error_summary(actual, error)

        self.assertEqual(metrics["mape_n"], 3)
        self.assertEqual(metrics["mape_excluded_zero_count"], 2)
        self.assertEqual(metrics["mape_zero_abs_tolerance"], 1e-9)
        self.assertAlmostEqual(metrics["mape_pct"], 100 * (1.0 + 0.5 + 1.0) / 3)
        self.assertAlmostEqual(metrics["p90_ape_pct"], 100.0)

    def test_all_zero_targets_report_undefined_relative_metrics_with_counts(self):
        metrics = checkpoint_train.relative_error_summary(
            [0.0, 1e-9, -1e-9], [1.0, 2.0, 3.0]
        )

        self.assertEqual(metrics["mape_n"], 0)
        self.assertEqual(metrics["mape_excluded_zero_count"], 3)
        self.assertTrue(math.isnan(metrics["mape_pct"]))
        self.assertTrue(math.isnan(metrics["p90_ape_pct"]))

    def test_train_evaluation_uses_same_mask_for_ape_and_relative_interval(self):
        metrics = train_models._evaluation_relative_metrics(
            y_true=np.array([0.0, 2.0, 4.0]),
            error=np.array([1000.0, 1.0, 4.0]),
            half_width=np.array([1000.0, 0.2, 0.8]),
        )

        self.assertEqual(metrics["mape_n"], 2)
        self.assertEqual(metrics["mape_excluded_zero_count"], 1)
        self.assertAlmostEqual(metrics["mape_pct"], 75.0)
        self.assertAlmostEqual(metrics["p90_ape_pct"], 95.0)
        self.assertAlmostEqual(metrics["interval_p90_half_width_pct"], 19.0)

    def test_relative_metric_length_mismatches_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "lengths differ"):
            checkpoint_train.relative_error_summary([1.0], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
