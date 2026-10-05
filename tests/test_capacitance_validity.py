"""Capacitance acceptance is separate from dielectric qualification."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import tempfile
import unittest

import pandas as pd

from module.electrostatic_cap import (
    build_capacitance_payload,
    parse_maxwell_capacitance_export,
)
from run_simulation_260706 import (
    Simulation,
    _cap_result_validation,
    _em_result_validation,
)


CAP_EXPORT = """Capacitance Unit: pF
Capacitance
        CapTx CapRx
CapTx   10 -2
CapRx   -2 4
"""


def solved_row():
    row = build_capacitance_payload(
        parse_maxwell_capacitance_export(CAP_EXPORT), 100.0, 400.0, 10.0,
        full_model=False,
    )
    row.update({
        "cap_percent_error": 1.0,
        "conv_passes_cap": 4,
        "conv_consecutive_cap": 1,
        "conv_error_pct_cap": 0.5,
        "conv_delta_pct_cap": 0.25,
        "mesh_tets_cap": 1000,
        "matrix_percent_error": 1.0,
        "matrix_min_converged": 1,
        "conv_passes_matrix": 3,
        "conv_consecutive_matrix": 1,
        "conv_error_pct_matrix": 0.5,
        "conv_delta_pct_matrix": 0.25,
        "Ltx": 100.0, "Lrx": 400.0, "M": 100.0, "k": 0.5,
        "Lmt": 50.0, "Lmr": 200.0, "Llt": 10.0, "Llr": 40.0,
    })
    return row


class CapacitanceValidityTests(unittest.TestCase):
    def test_converged_cap_is_numerically_valid_without_insulation_verdict(self):
        self.assertEqual(
            _cap_result_validation(pd.DataFrame([solved_row()])),
            (True, "valid_capacitance_only"),
        )

    def test_failed_cap_rejects_otherwise_valid_magnetic_result(self):
        row = solved_row()
        row["conv_error_pct_cap"] = 2.0
        frame = pd.DataFrame([row])
        self.assertTrue(_em_result_validation(frame, loss_on=False)[0])
        valid, reason = _em_result_validation(frame, loss_on=False, cap_on=True)
        self.assertFalse(valid)
        self.assertIn("cap: error energy 2% exceeds 1%", reason)

    def test_missing_or_unconverged_cap_telemetry_cannot_pass(self):
        for column, value in (
            ("conv_passes_cap", float("nan")),
            ("conv_consecutive_cap", 0),
            ("conv_consecutive_cap", 5),
            ("conv_delta_pct_cap", 1.1),
            ("conv_error_pct_cap", float("inf")),
            ("mesh_tets_cap", 0),
        ):
            with self.subTest(column=column, value=value):
                row = solved_row()
                row[column] = value
                self.assertFalse(_cap_result_validation(pd.DataFrame([row]))[0])

    def test_missing_capacitance_and_nonpassive_matrix_are_rejected(self):
        for changes in (
            {"C_tx_tx_F": float("nan")},
            {"C_rx_rx_raw_F": 0},
            {"C_tx_rx_signed_F": 1e-12},
            {"C_tx_rx_F": 1e-9, "C_tx_rx_signed_F": -1e-9},
        ):
            with self.subTest(changes=changes):
                row = solved_row()
                row.update(changes)
                self.assertFalse(_cap_result_validation(pd.DataFrame([row]))[0])

    def test_disabled_cap_has_no_verdict_and_preserves_em_contract(self):
        self.assertEqual(
            _cap_result_validation(None, cap_on=False), (None, "not_requested")
        )
        frame = pd.DataFrame([solved_row()]).drop(columns=["C_tx_tx_F"])
        self.assertTrue(_em_result_validation(frame, loss_on=False, cap_on=False)[0])

    def test_opt_in_graded_cap_requires_its_own_convergence_and_energy(self):
        row = {
            "cap_percent_error": 1.0,
            "conv_passes_cap_turn_graded_rx": 4,
            "conv_consecutive_cap_turn_graded_rx": 1,
            "conv_error_pct_cap_turn_graded_rx": 0.5,
            "conv_delta_pct_cap_turn_graded_rx": 0.25,
            "mesh_tets_cap_turn_graded_rx": 1000,
            "electrostatic_energy_raw_J": 1e-12,
            "C_eq_raw_F": 2e-12,
            "C_eq_full_F": 16e-12,
            "C_rx_rx_turn_graded_F": 16e-12,
        }
        self.assertTrue(_cap_result_validation(
            pd.DataFrame([row]), cap_on=False, graded_active_winding="Rx",
        )[0])
        row["electrostatic_energy_raw_J"] = 0.0
        self.assertFalse(_cap_result_validation(
            pd.DataFrame([row]), cap_on=False, graded_active_winding="Rx",
        )[0])
        row["electrostatic_energy_raw_J"] = 1e-12
        row["conv_consecutive_cap_turn_graded_rx"] = 0
        self.assertFalse(_cap_result_validation(
            pd.DataFrame([row]), cap_on=False, graded_active_winding="Rx",
        )[0])

    def test_native_cap_convergence_export_failure_is_not_nan_success(self):
        with tempfile.TemporaryDirectory() as directory:
            simulation = Simulation.__new__(Simulation)
            simulation.project_path = str(Path(directory))
            simulation.df_plus = pd.DataFrame([{"cap_percent_error": 1.0}])
            simulation.design1 = SimpleNamespace(
                available_variations=SimpleNamespace(nominal_w_values=[]),
                odesign=SimpleNamespace(ExportConvergence=Mock(return_value=False)),
            )
            with self.assertRaisesRegex(RuntimeError, "capacitance convergence extraction failed"):
                simulation._get_convergence_info_locked("cap")

    def test_cap_extraction_exception_propagates_once_for_both_backends(self):
        for backend in ("standalone", "pooled"):
            with self.subTest(backend=backend):
                simulation = Simulation.__new__(Simulation)
                simulation.aedt_backend = backend
                simulation._backend_mode = Mock(return_value=backend)
                simulation.solve_attempts = {}
                simulation.NUM_CORE = 1
                simulation.design1 = SimpleNamespace(
                    setup=SimpleNamespace(analyze=Mock(return_value=None)),
                )
                simulation._record_solver_core_dispatch = Mock()
                simulation.save_project = Mock()
                simulation._analyze_exact_pooled_design = Mock(return_value=0.5)
                simulation._verified_pooled_native_setup = Mock()
                simulation.aedt_automation_transaction = Mock(side_effect=nullcontext)
                extractor = Mock(side_effect=RuntimeError("native C export missing"))
                with self.assertRaisesRegex(RuntimeError, "native C export missing"):
                    simulation.analyze_and_extract("cap", extractor)
                extractor.assert_called_once_with()
                if backend == "pooled":
                    simulation._analyze_exact_pooled_design.assert_called_once_with("cap")
                else:
                    simulation.design1.setup.analyze.assert_called_once_with(cores=1)
                    self.assertEqual(simulation.solve_attempts["cap"], 1)


if __name__ == "__main__":
    unittest.main()
