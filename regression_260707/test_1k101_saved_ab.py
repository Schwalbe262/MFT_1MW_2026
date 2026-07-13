import copy
import json
import math
from pathlib import Path
import unittest

from regression_260707.verify.run_1k101_saved_ab import (
    evaluate_numerical_gates,
    parse_args,
    relative_error,
)


GATE_PATH = (
    Path(__file__).resolve().parent / "verify" / "1k101_native_ab_gate.json"
)


def gate_spec():
    return json.loads(GATE_PATH.read_text(encoding="utf-8"))


def passing_case(kf=0.85):
    source_peak = 100.0
    linkage = source_peak / (2.0 * math.pi * 1000.0)
    reference = {
        "B_material_T": 1.0,
        "native_raw_loss_W": 100.0,
        "margin_adjusted_loss_W": 115.0,
        "frequency_hz": 1000.0,
        "source_voltage_peak_V": source_peak,
    }
    observed = {
        "material_readback_exact": True,
        "geometry_relative_errors": {"area": 0.0, "volume": 0.0, "mass": 0.0},
        "native_B_pack_mean_T": kf,
        "native_B_material_mean_T": 1.0,
        "native_core_loss_raw_W": 100.0,
        "induced_voltage_peak_V": source_peak,
        "flux_linkage_peak_Wb_turn": linkage,
        "thermal_required": False,
    }
    return observed, reference


class RelativeErrorTests(unittest.TestCase):
    def test_explicit_denominator_matches_faraday_definition(self):
        self.assertAlmostEqual(
            relative_error(100.0, 99.0, denominator=100.0), 0.01
        )

    def test_nonfinite_or_missing_inputs_fail_closed(self):
        self.assertIsNone(relative_error(None, 1.0))
        self.assertIsNone(relative_error(float("nan"), 1.0))
        self.assertIsNone(relative_error(1.0, float("inf")))


class SavedAbGateEvaluationTests(unittest.TestCase):
    def test_nominal_kf085_case_passes_every_applicable_metric(self):
        observed, reference = passing_case(0.85)
        result = evaluate_numerical_gates(
            observed, reference, gate_spec(), 0.85
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["applicable_metric_count"], 7)
        ratio = result["metrics"]["B_material_to_B_pack_ratio"]
        self.assertAlmostEqual(ratio["value"], 1.0 / 0.85)
        self.assertEqual(
            ratio["expected_spec_key"],
            "B_material_kf0p85_ratio_expected",
        )
        self.assertTrue(ratio["passed"])
        thermal = result["metrics"]["thermal_native_power_balance_relative_error"]
        self.assertFalse(thermal["applicable"])
        self.assertTrue(thermal["passed"])

    def test_nominal_kf070_uses_its_explicit_expected_ratio(self):
        observed, reference = passing_case(0.70)
        result = evaluate_numerical_gates(
            observed, reference, gate_spec(), 0.70
        )

        ratio = result["metrics"]["B_material_to_B_pack_ratio"]
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(ratio["expected"], 1.0 / 0.70)
        self.assertEqual(
            ratio["expected_spec_key"],
            "B_material_kf0p70_ratio_expected",
        )

    def test_all_relative_limits_are_inclusive(self):
        observed, reference = passing_case(0.85)
        gates = gate_spec()["numerical_gates"]
        observed.update({
            "geometry_relative_errors": {
                "area": gates["geometry_relative_error_max"],
                "volume": 0.0,
                "mass": 0.0,
            },
            "native_B_material_mean_T": 1.0 + gates[
                "standard_Bavg_vs_Faraday_Bmaterial_relative_error_max"
            ],
            "native_core_loss_raw_W": 100.0 * (
                1.0 + gates[
                    "native_loss_vs_Faraday_POWERLITE_mass_relative_error_max"
                ]
            ),
            "induced_voltage_peak_V": 100.0,
            "flux_linkage_peak_Wb_turn": 99.0 / (2.0 * math.pi * 1000.0),
        })
        # Preserve the exact material/pack ratio while moving the B error to its limit.
        observed["native_B_pack_mean_T"] = (
            observed["native_B_material_mean_T"] * 0.85
        )
        result = evaluate_numerical_gates(
            observed, reference, gate_spec(), 0.85
        )

        self.assertTrue(result["passed"], result)
        self.assertAlmostEqual(
            result["metrics"]["faraday_relative_error"]["value"], 0.01
        )

    def test_just_over_a_limit_fails_the_aggregate(self):
        observed, reference = passing_case(0.85)
        observed["native_B_material_mean_T"] = 1.1500001
        observed["native_B_pack_mean_T"] = observed["native_B_material_mean_T"] * 0.85
        result = evaluate_numerical_gates(
            observed, reference, gate_spec(), 0.85
        )

        metric = result["metrics"][
            "standard_Bavg_vs_Faraday_Bmaterial_relative_error"
        ]
        self.assertFalse(metric["passed"])
        self.assertFalse(result["passed"])

    def test_native_loss_gate_compares_raw_not_margin_adjusted_reference(self):
        observed, reference = passing_case(0.85)
        observed["native_core_loss_raw_W"] = 140.0
        result = evaluate_numerical_gates(
            observed, reference, gate_spec(), 0.85
        )

        metric = result["metrics"][
            "native_loss_vs_Faraday_POWERLITE_mass_relative_error"
        ]
        self.assertAlmostEqual(metric["value"], 0.40)
        self.assertFalse(metric["passed"])

    def test_missing_evidence_is_structured_and_fails_closed(self):
        observed, reference = passing_case(0.85)
        observed.pop("flux_linkage_peak_Wb_turn")
        observed["geometry_relative_errors"]["mass"] = float("nan")
        result = evaluate_numerical_gates(
            observed, reference, gate_spec(), 0.85
        )

        for name in ("faraday_relative_error", "geometry_relative_error"):
            metric = result["metrics"][name]
            self.assertFalse(metric["available"])
            self.assertFalse(metric["passed"])
            self.assertIsNone(metric["value"])
            self.assertIn("reason", metric)
        self.assertFalse(result["passed"])

    def test_required_thermal_evidence_is_not_silently_skipped(self):
        observed, reference = passing_case(0.85)
        observed["thermal_required"] = True
        result = evaluate_numerical_gates(
            observed, reference, gate_spec(), 0.85
        )

        thermal = result["metrics"]["thermal_native_power_balance_relative_error"]
        self.assertTrue(thermal["applicable"])
        self.assertFalse(thermal["available"])
        self.assertFalse(thermal["passed"])
        self.assertFalse(result["passed"])

    def test_every_metric_has_a_boolean_and_gate_limit_provenance(self):
        observed, reference = passing_case(0.85)
        result = evaluate_numerical_gates(
            observed, reference, gate_spec(), 0.85
        )

        for name, metric in result["metrics"].items():
            with self.subTest(metric=name):
                self.assertIs(type(metric["passed"]), bool)
                self.assertIn("applicable", metric)
                if name == "B_material_to_B_pack_ratio":
                    self.assertIn("relative_tolerance", metric)
                    self.assertIn("tolerance_spec_key", metric)
                elif name == "material_readback_exact":
                    self.assertIn("expected", metric)
                    self.assertIn("spec_key", metric)
                else:
                    self.assertIn("limit", metric)
                    self.assertIn("spec_key", metric)

    def test_missing_gate_configuration_is_rejected(self):
        observed, reference = passing_case(0.85)
        specification = copy.deepcopy(gate_spec())
        del specification["numerical_gates"]["faraday_relative_error_max"]

        with self.assertRaisesRegex(ValueError, "missing keys"):
            evaluate_numerical_gates(observed, reference, specification, 0.85)

    def test_negative_error_evidence_cannot_pass(self):
        observed, reference = passing_case(0.85)
        observed["geometry_relative_errors"] = {
            "area": -1.0, "volume": -2.0, "mass": -3.0,
        }
        result = evaluate_numerical_gates(
            observed, reference, gate_spec(), 0.85
        )

        self.assertFalse(result["metrics"]["geometry_relative_error"]["passed"])
        self.assertFalse(result["passed"])

    def test_only_declared_ab_factors_are_accepted(self):
        observed, reference = passing_case(0.80)
        with self.assertRaisesRegex(ValueError, "not an ab_cases"):
            evaluate_numerical_gates(observed, reference, gate_spec(), 0.80)

    def test_reference_kf1_uses_derived_identity_ratio(self):
        observed, reference = passing_case(1.0)
        result = evaluate_numerical_gates(
            observed, reference, gate_spec(), 1.0
        )
        ratio = result["metrics"]["B_material_to_B_pack_ratio"]

        self.assertTrue(result["passed"])
        self.assertEqual(ratio["expected"], 1.0)
        self.assertEqual(ratio["expected_spec_key"], "derived_1_over_kf")
        self.assertIn("conversion_contract", ratio["attestation_scope"])

    def test_invalid_gate_limits_are_rejected(self):
        observed, reference = passing_case(0.85)
        for value in (-0.1, float("nan"), "not-a-number"):
            specification = copy.deepcopy(gate_spec())
            specification["numerical_gates"]["faraday_relative_error_max"] = value
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "finite and nonnegative"
            ):
                evaluate_numerical_gates(
                    observed, reference, specification, 0.85
                )

    def test_evaluation_is_strict_json_serializable(self):
        observed, reference = passing_case(0.85)
        observed["native_core_loss_raw_W"] = float("nan")
        result = evaluate_numerical_gates(
            observed, reference, gate_spec(), 0.85
        )

        json.dumps(result, allow_nan=False)
        self.assertIsNone(result["metrics"][
            "native_loss_vs_Faraday_POWERLITE_mass_relative_error"
        ]["value"])


class SavedAbCliTests(unittest.TestCase):
    def test_nonfinite_cli_numbers_are_rejected_before_aedt(self):
        base = ["--project-path", "missing.aedt", "--out", "out.json"]
        for extra in (("--kf", "nan"), ("--kf", "0.85", "--op-timeout-seconds", "inf")):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                parse_args([*base, *extra])


if __name__ == "__main__":
    unittest.main()
