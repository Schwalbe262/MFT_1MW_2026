import sys
import unittest
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
TRAINING_DIR = HERE / "training"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TRAINING_DIR))

import thermal_quality  # noqa: E402
import checkpoint_train  # noqa: E402
import train_models  # noqa: E402
import tune_optuna  # noqa: E402


def strict_row(**updates):
    row = {
        "git_hash": "mixed-profile-revision-is-allowed",
        "result_valid_em": 1,
        "result_valid_thermal": 1,
        "thermal_solved": 1,
        "thermal_extraction_complete": 1,
        "thermal_required_missing_count": 0,
        "thermal_required_group_mask": 11,
        "thermal_convergence_available": 1,
        "thermal_converged": 1,
        "thermal_iterations": 132,
        "thermal_residual_flow_limit": 1e-3,
        "thermal_residual_energy_limit": 1e-7,
        "thermal_residual_continuity": 8e-4,
        "thermal_residual_x_velocity": 4e-4,
        "thermal_residual_y_velocity": 9e-4,
        "thermal_residual_z_velocity": 4e-4,
        "thermal_residual_energy": 5e-8,
        "thermal_rx_model": "homogenized_blocks",
        "thermal_rx_power_balance_ok": 1,
        "thermal_rx_power_balance_group_count": 1,
        "thermal_rx_power_balance_max_abs_w": 0.0,
        "thermal_rx_expected_power_w": 120.0,
        "thermal_rx_assigned_power_w": 120.0,
        "N2_side": 0,
        "n_explicit_turns": 0,
        "T_max_Tx": 80.0,
        "T_max_Rx_main": 81.0,
        "T_max_Rx_side": float("nan"),
        "T_max_core": 82.0,
        "Tprobe_Tx_leeward_max": 79.0,
        "Tprobe_Rx_main_leeward_max": 80.0,
        "Tprobe_Rx_side_leeward_max": float("nan"),
        "Tprobe_core_center_max": 81.0,
    }
    row.update(updates)
    return row


class ThermalTrainingQualityTests(unittest.TestCase):
    def test_mixed_revision_rows_pass_without_exact_profile_match(self):
        rows = pd.DataFrame([
            strict_row(git_hash="revision-a", mesh_level_thermal=3),
            strict_row(
                git_hash="revision-b",
                mesh_level_thermal=4,
                n_explicit_turns=2,
                thermal_rx_model="hybrid_explicit",
            ),
        ])

        self.assertEqual(thermal_quality.strict_thermal_mask(rows).tolist(), [True, True])

        without_explicit_count = rows.iloc[[0]].drop(columns=["n_explicit_turns"])
        self.assertTrue(thermal_quality.strict_thermal_mask(without_explicit_count).iloc[0])

    def test_each_strict_evidence_family_fails_closed(self):
        invalid_updates = {
            "thermal-flag": {"result_valid_thermal": 0},
            "em-flag": {"result_valid_em": 0},
            "extraction": {"thermal_extraction_complete": 0},
            "required-group": {"thermal_required_missing_count": 1},
            "required-temperature": {"Tprobe_core_center_max": float("nan")},
            "flow-residual": {"thermal_residual_continuity": 1.1e-3},
            "energy-residual": {"thermal_residual_energy": 1.1e-7},
            "power-balance-flag": {"thermal_rx_power_balance_ok": 0},
            "power-balance-value": {"thermal_rx_assigned_power_w": 119.0},
            "model": {"thermal_rx_model": "unknown"},
            "zero-explicit-model": {
                "n_explicit_turns": 0,
                "thermal_rx_model": "hybrid_explicit",
            },
        }
        for name, updates in invalid_updates.items():
            with self.subTest(name=name):
                frame = pd.DataFrame([strict_row(**updates)])
                self.assertFalse(thermal_quality.strict_thermal_mask(frame).iloc[0])

    def test_side_temperatures_and_mask_are_required_only_when_side_turns_exist(self):
        valid_side = strict_row(
            N2_side=2,
            thermal_required_group_mask=15,
            T_max_Rx_side=83.0,
            Tprobe_Rx_side_leeward_max=82.0,
        )
        missing_side = dict(valid_side, Tprobe_Rx_side_leeward_max=float("nan"))
        wrong_mask = dict(valid_side, thermal_required_group_mask=11)

        mask = thermal_quality.strict_thermal_mask(
            pd.DataFrame([valid_side, missing_side, wrong_mask])
        )

        self.assertEqual(mask.tolist(), [True, False, False])

    def test_legacy_thermal_solved_eighteen_rows_are_em_only(self):
        legacy = pd.DataFrame([
            {
                "thermal_solved": 1,
                "N2_side": 2,
                "Tprobe_Tx_leeward_max": 75.0 + index,
                "Llt": 13.0 + index,
            }
            for index in range(18)
        ])

        thermal_mask = thermal_quality.target_training_mask(
            legacy, "Tprobe_Tx_leeward_max"
        )
        em_mask = thermal_quality.target_training_mask(legacy, "Llt_phys")
        annotated = thermal_quality.annotate_thermal_tier(legacy)

        self.assertEqual(int(thermal_mask.sum()), 0)
        self.assertEqual(int(em_mask.sum()), 18)
        self.assertEqual(len(annotated), 18)
        self.assertEqual(set(annotated["thermal_strict_valid"]), {0})
        self.assertEqual(set(annotated["thermal_training_tier"]), {"em_only"})

    def test_all_trainers_share_one_mask_and_quality_flags_are_not_features(self):
        self.assertIs(
            checkpoint_train.target_training_mask,
            thermal_quality.target_training_mask,
        )
        self.assertIs(
            train_models.target_training_mask,
            thermal_quality.target_training_mask,
        )
        self.assertIs(
            tune_optuna.target_training_mask,
            thermal_quality.target_training_mask,
        )

        frame = pd.DataFrame({
            "design_parameter": [1.0, 2.0],
            "thermal_strict_valid": [0, 1],
            "thermal_iterations": [100, 120],
            "result_valid_thermal": [0, 1],
            "Tprobe_Tx_leeward_max": [70.0, 71.0],
        })
        self.assertEqual(checkpoint_train.feature_columns(frame), ["design_parameter"])


if __name__ == "__main__":
    unittest.main()
