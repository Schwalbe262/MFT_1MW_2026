import json
import math
import unittest

from module.thermal_probe_contract import (
    ProbeSheetCollection,
    aggregate_rx_side_faces,
    parse_temperature_celsius,
    serialize_probe_failures,
    validate_probe_rectangle,
)


class RxSideThermalProbeContractTest(unittest.TestCase):
    def test_full_model_selects_hottest_of_all_four_faces_and_pairs_mean(self):
        values = {
            "left_outer_max": 91.0, "left_outer_mean": 80.0,
            "right_outer_max": 94.0, "right_outer_mean": 81.0,
            "left_inner_max": 103.0, "left_inner_mean": 87.0,
            "right_inner_max": 98.0, "right_inner_mean": 85.0,
        }

        aggregate, selected = aggregate_rx_side_faces(
            values,
            ("left_outer", "right_outer"),
            ("left_inner", "right_inner"),
        )

        self.assertEqual(selected, "left_inner")
        self.assertEqual(aggregate["Tprobe_Rx_side_outer_max"], 94.0)
        self.assertEqual(aggregate["Tprobe_Rx_side_outer_mean"], 81.0)
        self.assertEqual(aggregate["Tprobe_Rx_side_inner_max"], 103.0)
        self.assertEqual(aggregate["Tprobe_Rx_side_inner_mean"], 87.0)
        self.assertEqual(aggregate["Tprobe_Rx_side_leeward_max"], 103.0)
        # It is the paired mean of the selected face, not (80+81+87+85)/4.
        self.assertEqual(aggregate["Tprobe_Rx_side_leeward_mean"], 87.0)

    def test_missing_inner_face_fails_closed(self):
        values = {
            "outer_max": 91.0, "outer_mean": 80.0,
            "inner_max": 103.0,
        }

        aggregate, selected = aggregate_rx_side_faces(
            values, ("outer",), ("inner",)
        )

        self.assertEqual(aggregate, {})
        self.assertEqual(selected, "")


class ProbeExtractionContractTest(unittest.TestCase):
    def test_rectangle_validation_rejects_nonfinite_and_zero_spans(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_probe_rectangle(
                "Tprobe_bad", "XZ", [0.0, math.nan, 0.0], [2.0, 3.0]
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_probe_rectangle(
                "Tprobe_bad", "XZ", [0.0, 0.0, 0.0], [2.0, 0.0]
            )

    def test_rectangle_validation_normalizes_numeric_geometry(self):
        orientation, origin, sizes = validate_probe_rectangle(
            "Tprobe_ok", "xz", ["1", 2, 3.5], [4, "5.25"]
        )
        self.assertEqual(orientation, "XZ")
        self.assertEqual(origin, (1.0, 2.0, 3.5))
        self.assertEqual(sizes, (4.0, 5.25))

    def test_temperature_parser_handles_units_and_rejects_nonfinite(self):
        self.assertAlmostEqual(parse_temperature_celsius("84.25cel"), 84.25)
        self.assertAlmostEqual(parse_temperature_celsius(357.15, "K"), 84.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            parse_temperature_celsius(float("nan"))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            parse_temperature_celsius("84.0 F")

    def test_failure_serialization_is_structured_and_deterministic(self):
        sheets = ProbeSheetCollection()
        sheets.expect("Tprobe_core")
        sheets.record_failure(
            "Tprobe_core", "geometry", "invalid_rectangle", "zero span"
        )
        payload = serialize_probe_failures(sheets.failures)
        self.assertEqual(json.loads(payload), [{
            "probe": "Tprobe_core",
            "stage": "geometry",
            "reason": "invalid_rectangle",
            "detail": "zero span",
        }])


if __name__ == "__main__":
    unittest.main()
