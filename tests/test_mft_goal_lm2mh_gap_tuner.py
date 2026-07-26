import math
import unittest

from tools import mft_goal_lm2mh_gap_tuner as tuner


def _observation(gap_mm, lm_h):
    return {
        "contract_valid": True,
        "core_center_gap_mm": gap_mm,
        "target_abs_error_H": abs(lm_h - tuner.TARGET_LM_H),
        "result_sha256": f"{int(gap_mm * 1000):064x}",
        "full_physical_matrix_readback": {
            "Lm_primary_referred_H": lm_h,
        },
    }


class GapDecisionTests(unittest.TestCase):
    def test_selects_tuned_observation_inside_explicit_tolerance(self):
        decision = tuner._next_gap_decision(
            [_observation(0.75, 0.00201)], refinement_count=0
        )
        self.assertEqual(decision["status"], "tuned")
        self.assertEqual(decision["selected"]["core_center_gap_mm"], 0.75)

    def test_bracket_uses_guarded_secant_inside_interval(self):
        decision = tuner._next_gap_decision(
            [
                _observation(0.5, 0.0024),
                _observation(1.0, 0.0016),
            ],
            refinement_count=1,
        )
        self.assertEqual(decision["status"], "refine_bracket")
        self.assertEqual(
            decision["algorithm"],
            "guarded_secant_with_bisection_fallback",
        )
        self.assertGreater(decision["next_gaps_mm"][0], 0.5)
        self.assertLess(decision["next_gaps_mm"][0], 1.0)

    def test_gap_only_tuning_fails_if_zero_gap_is_already_below_target(self):
        decision = tuner._next_gap_decision(
            [
                _observation(0.0, 0.0018),
                _observation(0.5, 0.0014),
            ],
            refinement_count=0,
        )
        self.assertEqual(
            decision["status"], "infeasible_gap_only_cannot_raise_Lm"
        )


class MatrixReadbackTests(unittest.TestCase):
    def test_attests_native_and_full_physical_matrix_readback(self):
        l11 = 1010.0
        native_lm = 1000.0
        coupling = math.sqrt(native_lm / l11)
        l22 = 101000.0
        mutual = coupling * math.sqrt(l11 * l22)
        result = {
            "full_model": 0,
            "round_corner": 0,
            "matrix_on": 1,
            "cap_on": 0,
            "loss_on": 0,
            "thermal_on": 0,
            "core_center_gap_mm": 0.75,
            "core_center_gap_geometry_attested": 1,
            "core_center_gap_readback_mm": 0.75,
            "core_center_gap_topology":
                "center_leg_bottom_top_physical_air_interval",
            "core_center_gap_symmetry_geometry_attested": 1,
            "core_center_gap_removed_volume_readback_mm3": 123.0,
            "core_center_gap_removed_volume_rel_error": 0.0,
            "Ltx": l11,
            "Lrx": l22,
            "M": mutual,
            "k": coupling,
            "Lmt": native_lm,
            "Llt": l11 - native_lm,
            "matrix_percent_error": 0.5,
            "matrix_min_converged": 1,
            "conv_passes_matrix": 6,
            "conv_consecutive_matrix": 1,
            "conv_error_pct_matrix": 0.4,
            "conv_delta_pct_matrix": 0.2,
        }
        observed = tuner._matrix_observation(result, 0.75)

        self.assertTrue(observed["contract_valid"])
        self.assertTrue(observed["within_tolerance"])
        self.assertAlmostEqual(
            observed["native_eighth_matrix_readback"]["Lm_primary_uH"],
            1000.0,
        )
        self.assertAlmostEqual(
            observed["full_physical_matrix_readback"][
                "Lm_primary_referred_H"
            ],
            0.002,
        )
        self.assertAlmostEqual(
            observed["full_physical_matrix_readback"]["L11_uH"],
            2.0 * l11,
        )


if __name__ == "__main__":
    unittest.main()
