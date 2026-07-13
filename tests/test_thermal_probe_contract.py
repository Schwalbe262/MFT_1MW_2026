import unittest

from module.thermal_probe_contract import aggregate_rx_side_faces


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


if __name__ == "__main__":
    unittest.main()
