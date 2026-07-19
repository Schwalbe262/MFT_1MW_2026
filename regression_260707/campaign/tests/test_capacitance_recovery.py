import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAMPAIGN_DIR))

import train_io  # noqa: E402


SPECS = train_io.CAPACITANCE_RECOVERY_SPECS


def _resonance(inductance_h, capacitance_f):
    return 1.0 / (2.0 * math.pi * math.sqrt(inductance_h * capacitance_f))


def _recoverable_row(cap_on=1):
    exact = {
        "C_tx_tx_F": 1.856789012345678e-10,
        "C_rx_rx_F": 8.721234567890123e-10,
        "C_tx_rx_F": 3.456789012345678e-10,
    }
    inductance = {
        "cap_L_tx_self_H": 180.123456789e-6,
        "cap_L_rx_self_H": 780.987654321e-6,
        "cap_L_leakage_H": 27.5123456789e-6,
    }
    row = {"cap_on": cap_on}
    for target, frequency, inductance_column in SPECS:
        row[target] = round(exact[target] / 1e-10) * 1e-10
        row[inductance_column] = inductance[inductance_column]
        row[frequency] = _resonance(
            inductance[inductance_column], exact[target]
        )
    return row, exact


class CapacitanceRecoveryTests(unittest.TestCase):
    def test_recovers_all_three_targets_atomically_without_mutating_source(self):
        enabled, exact = _recoverable_row()
        disabled, _ = _recoverable_row(cap_on=0)
        raw = pd.DataFrame([enabled, disabled])
        before = raw.copy(deep=True)

        recovered = train_io.recover_quantized_capacitance_targets(raw)

        pd.testing.assert_frame_equal(raw, before)
        for target, _, _ in SPECS:
            self.assertAlmostEqual(recovered.loc[0, target], exact[target], 24)
            self.assertEqual(recovered.loc[1, target], raw.loc[1, target])
        self.assertEqual(
            recovered["capacitance_recovered_from_resonance"].tolist(), [1, 0]
        )
        self.assertEqual(
            recovered["capacitance_recovery_required"].tolist(), [1, 0]
        )
        self.assertEqual(
            recovered.loc[0, "capacitance_recovery_contract"],
            train_io.CAPACITANCE_RECOVERY_CONTRACT,
        )
        audit = recovered.attrs[train_io.CAPACITANCE_RECOVERY_ATTR]
        self.assertEqual(audit["recovered_row_count"], 1)
        self.assertLessEqual(
            audit["max_observed_abs_delta_F"],
            train_io.CAPACITANCE_QUANTIZATION_MAX_ABS_DELTA_F,
        )

    def test_semantic_mismatch_fails_closed(self):
        row, _ = _recoverable_row()
        row["C_rx_rx_F"] += 1e-9

        with self.assertRaisesRegex(ValueError, "semantic mismatch"):
            train_io.recover_quantized_capacitance_targets(
                pd.DataFrame([row])
            )

    def test_missing_lc_columns_preserves_legacy_values(self):
        raw = pd.DataFrame({
            "cap_on": [1],
            "C_tx_tx_F": [1.5e-8],
            "C_rx_rx_F": [1.0e-9],
            "C_tx_rx_F": [4.0e-10],
        })

        recovered = train_io.recover_quantized_capacitance_targets(raw)

        for target, _, _ in SPECS:
            self.assertEqual(recovered.loc[0, target], raw.loc[0, target])
        self.assertEqual(
            recovered.attrs[train_io.CAPACITANCE_RECOVERY_ATTR]["status"],
            "columns_unavailable_passthrough",
        )
        self.assertEqual(
            recovered.loc[0, "capacitance_recovered_from_resonance"], 0
        )
        self.assertEqual(recovered.loc[0, "capacitance_recovery_required"], 0)

    def test_nonfinite_lc_evidence_is_not_synthesized(self):
        row, _ = _recoverable_row()
        row["cap_L_leakage_H"] = np.nan
        raw = pd.DataFrame([row])

        recovered = train_io.recover_quantized_capacitance_targets(raw)

        for target, _, _ in SPECS:
            self.assertEqual(recovered.loc[0, target], raw.loc[0, target])
        self.assertEqual(
            recovered.loc[0, "capacitance_recovered_from_resonance"], 0
        )
        self.assertEqual(recovered.loc[0, "capacitance_recovery_required"], 1)

    def test_build_train_io_projects_recovery_audit_and_schema_v9(self):
        row, exact = _recoverable_row()

        view = train_io.build_train_io(pd.DataFrame([row]))

        self.assertEqual(view.loc[0, "train_io_schema_version"], 9)
        self.assertEqual(
            view.loc[0, "capacitance_recovered_from_resonance"], 1
        )
        for target, _, _ in SPECS:
            self.assertAlmostEqual(view.loc[0, target], exact[target], 24)


if __name__ == "__main__":
    unittest.main()
