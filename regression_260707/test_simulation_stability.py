import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from run_simulation_260706 import (
    Simulation,
    SolutionDataUnavailableError,
    _completion_exit_code,
    _thermal_failure_frame,
    _thermal_result_is_valid,
    log_failed_sample,
)


class _FakeSolution:
    def __init__(self, values, units=None):
        self._values = values
        self.units_data = units or {}

    def data_real(self, expression):
        value = self._values[expression]
        if isinstance(value, Exception):
            raise value
        return value


class _FakePost:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.field_calls = []

    def get_solution_data(self, **kwargs):
        self.calls.append(kwargs)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response

    def get_solution_data_per_variation(self, **kwargs):
        self.field_calls.append(kwargs)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def _simulation_with_post(post):
    simulation = Simulation.__new__(Simulation)
    simulation.design1 = SimpleNamespace(post=post)
    simulation.extraction_attempts = {}
    simulation.extraction_backends = {}
    return simulation


class SolutionDataTests(unittest.TestCase):
    def test_reads_finite_data_and_converts_units(self):
        post = _FakePost([_FakeSolution({"L": [2.0]}, {"L": "H"})])
        simulation = _simulation_with_post(post)

        frame = simulation._solution_data_frame(
            ["L"], aliases=["L_uH"], target_units={"L": "uH"},
            extraction_key="matrix", retry_delay=0,
        )

        self.assertEqual(frame["L_uH"].iloc[0], 2_000_000.0)
        self.assertEqual(simulation.extraction_backends["matrix"], "get_solution_data")
        self.assertEqual(post.calls[0]["setup_sweep_name"], "Setup1 : LastAdaptive")

    def test_none_response_is_not_verified_no_data(self):
        simulation = _simulation_with_post(_FakePost([None, None, None]))

        with self.assertRaises(RuntimeError) as raised:
            simulation._solution_data_frame(["L"], retry_delay=0)

        self.assertNotIsInstance(raised.exception, SolutionDataUnavailableError)

    def test_magnetic_mutual_inductance_preserves_historical_absolute_convention(self):
        params = [
            "Matrix.L(Tx_winding,Tx_winding)",
            "Matrix.L(Rx_winding,Rx_winding)",
            "Matrix.L(Tx_winding,Rx_winding)",
            "abs(Matrix.CplCoef(Tx_winding,Rx_winding))",
            "Matrix.L(Tx_winding,Tx_winding)*(abs(Matrix.CplCoef(Tx_winding,Rx_winding))^2)",
            "Matrix.L(Rx_winding,Rx_winding)*(abs(Matrix.CplCoef(Tx_winding,Rx_winding))^2)",
            "Matrix.L(Tx_winding,Tx_winding)*(1-abs(Matrix.CplCoef(Tx_winding,Rx_winding))^2)",
            "Matrix.L(Rx_winding,Rx_winding)*(1-abs(Matrix.CplCoef(Tx_winding,Rx_winding))^2)",
            "PerWindingSolidLoss(Tx_winding)",
            "PerWindingSolidLoss(Rx_winding)",
        ]
        values = {expression: [1.0] for expression in params}
        values[params[2]] = [-28.5]
        simulation = _simulation_with_post(_FakePost([_FakeSolution(values)]))

        frame = simulation.get_magnetic_parameter()

        self.assertEqual(frame["M"].iloc[0], 28.5)

    def test_solution_object_with_empty_expression_is_query_failure(self):
        empty = _FakeSolution({"L": []})
        simulation = _simulation_with_post(_FakePost([empty, empty, empty]))

        with self.assertRaises(RuntimeError) as raised:
            simulation._solution_data_frame(["L"], retry_delay=0)

        self.assertNotIsInstance(raised.exception, SolutionDataUnavailableError)

    def test_fields_use_native_frequency_only_query(self):
        post = _FakePost([_FakeSolution({"P_loss": [12.5]}, {"P_loss": "W"})])
        simulation = _simulation_with_post(post)

        frame = simulation._solution_data_frame(
            ["P_loss"], report_category="Fields", extraction_key="loss", retry_delay=0
        )

        self.assertEqual(frame["P_loss"].iloc[0], 12.5)
        self.assertEqual(post.calls, [])
        self.assertEqual(post.field_calls[0]["solution_type"], "Fields")
        self.assertEqual(post.field_calls[0]["context"], [])
        self.assertEqual(
            post.field_calls[0]["sweeps"],
            {"Freq": ["All"], "Phase": ["0deg"]},
        )
        self.assertEqual(
            simulation.extraction_backends["loss"],
            "get_solution_data_per_variation",
        )

    def test_partial_expression_data_is_not_verified_no_data(self):
        partial = _FakeSolution({"L": [1.0], "M": []})
        simulation = _simulation_with_post(_FakePost([partial, partial, partial]))

        with self.assertRaises(RuntimeError) as raised:
            simulation._solution_data_frame(["L", "M"], retry_delay=0)

        self.assertNotIsInstance(raised.exception, SolutionDataUnavailableError)

    def test_false_response_is_not_verified_no_data(self):
        simulation = _simulation_with_post(_FakePost([False, False, False]))

        with self.assertRaises(RuntimeError) as raised:
            simulation._solution_data_frame(["L"], retry_delay=0)

        self.assertNotIsInstance(raised.exception, SolutionDataUnavailableError)

    def test_transport_failure_does_not_claim_no_data(self):
        failures = [RuntimeError("grpc") for _ in range(3)]
        simulation = _simulation_with_post(_FakePost(failures))

        with self.assertRaises(RuntimeError) as raised:
            simulation._solution_data_frame(["L"], retry_delay=0)

        self.assertNotIsInstance(raised.exception, SolutionDataUnavailableError)

    def test_mixed_empty_and_transport_failures_do_not_claim_no_data(self):
        simulation = _simulation_with_post(
            _FakePost([None, RuntimeError("grpc"), RuntimeError("grpc")])
        )

        with self.assertRaises(RuntimeError) as raised:
            simulation._solution_data_frame(["L"], retry_delay=0)

        self.assertNotIsInstance(raised.exception, SolutionDataUnavailableError)


class AnalyzePolicyTests(unittest.TestCase):
    @staticmethod
    def _simulation(analyze_results):
        results = iter(analyze_results)
        setup = SimpleNamespace(analyze=lambda **_: next(results))
        simulation = Simulation.__new__(Simulation)
        simulation.design1 = SimpleNamespace(setup=setup)
        simulation.solve_attempts = {}
        simulation.NUM_CORE = 4
        simulation.save_project = lambda: None
        simulation._log_recent_aedt_messages = lambda _label: None
        return simulation

    def test_none_analyze_result_is_success_and_solves_once(self):
        simulation = self._simulation([None])
        extracted = []

        simulation.analyze_and_extract("matrix", lambda: extracted.append(True))

        self.assertEqual(simulation.solve_attempts["matrix"], 1)
        self.assertEqual(extracted, [True])

    def test_empty_data_error_never_resolves(self):
        simulation = self._simulation([None])

        with self.assertRaisesRegex(SolutionDataUnavailableError, "empty"):
            simulation.analyze_and_extract(
                "loss", lambda: (_ for _ in ()).throw(SolutionDataUnavailableError("empty"))
            )

        self.assertEqual(simulation.solve_attempts["loss"], 1)

    def test_transport_failure_never_resolves(self):
        simulation = self._simulation([None])

        with self.assertRaisesRegex(RuntimeError, "grpc"):
            simulation.analyze_and_extract("loss", lambda: (_ for _ in ()).throw(RuntimeError("grpc")))

        self.assertEqual(simulation.solve_attempts["loss"], 1)


class NativeProjectHandleTests(unittest.TestCase):
    @staticmethod
    def _native_project(active_design=None):
        calls = []

        class NativeProject:
            def SetActiveDesign(self, name):
                calls.append(("active", name))
                return active_design

            def Save(self):
                calls.append(("save", None))

        return NativeProject(), calls

    def test_prefers_explicit_pyproject_native_handle(self):
        native, _ = self._native_project()
        simulation = Simulation.__new__(Simulation)
        simulation.project = SimpleNamespace(
            project=native,
            proj=native,
            oproject=lambda: (_ for _ in ()).throw(AssertionError("dynamic fallback used")),
        )

        self.assertIs(simulation._native_project_handle(), native)

    def test_falls_back_to_raw_solver_project_handle(self):
        native, _ = self._native_project()
        simulation = Simulation.__new__(Simulation)
        simulation.project = SimpleNamespace(oproject="misleading dynamic attribute")
        simulation.design1 = SimpleNamespace(
            solver_instance=SimpleNamespace(oproject=native)
        )

        self.assertIs(simulation._native_project_handle(), native)

    def test_create_design_recovery_updates_raw_solver_design(self):
        class NativeDesign:
            def __init__(self):
                self.settings_calls = []

            def SetDesignSettings(self, *args):
                self.settings_calls.append(args)

        native_design = NativeDesign()
        native_project, calls = self._native_project(native_design)
        design_solutions = SimpleNamespace(_odesign=None)
        solver_instance = SimpleNamespace(
            _odesign=None,
            design_solutions=design_solutions,
        )
        wrapped_design = SimpleNamespace(
            odesign=None,
            solver_instance=solver_instance,
        )
        project_wrapper = SimpleNamespace(
            project=native_project,
            proj=native_project,
            oproject="misleading dynamic attribute",
            create_design=lambda **_: wrapped_design,
        )
        simulation = Simulation.__new__(Simulation)
        simulation.project = project_wrapper

        with patch("run_simulation_260706.time.sleep"):
            simulation.create_design("maxwell_loss")

        self.assertIs(solver_instance._odesign, native_design)
        self.assertIs(design_solutions._odesign, native_design)
        self.assertFalse(hasattr(wrapped_design, "_odesign"))
        self.assertEqual(calls, [("active", "maxwell_loss")])
        self.assertEqual(len(native_design.settings_calls), 1)

    def test_save_project_uses_native_fallback(self):
        native, calls = self._native_project()
        simulation = Simulation.__new__(Simulation)
        simulation.project = SimpleNamespace(project=native, proj=native)

        def fail_save():
            raise RuntimeError("wrapper save failed")

        simulation.design1 = SimpleNamespace(save_project=fail_save)

        simulation.save_project()

        self.assertEqual(calls, [("save", None)])


class FieldsReporterTests(unittest.TestCase):
    def test_reacquires_reporter_after_stale_calcstack(self):
        class Reporter:
            def __init__(self, stale=False):
                self.stale = stale
                self.added = []

            def DoesNamedExpressionExists(self, _name):
                return False

            def CalcStack(self, _operation):
                if self.stale:
                    raise RuntimeError("stale reporter")

            def EnterQty(self, _quantity):
                pass

            def AddNamedExpression(self, name, _category):
                self.added.append(name)

        reporters = [Reporter(stale=True), Reporter()]
        native_designs = [
            SimpleNamespace(GetModule=lambda _name, reporter=reporter: reporter)
            for reporter in reporters
        ]
        active_calls = []

        def set_active(name):
            active_calls.append(name)
            return native_designs.pop(0)

        simulation = Simulation.__new__(Simulation)
        simulation.design1 = SimpleNamespace(design_name="maxwell_loss", odesign=None)
        native_project = SimpleNamespace(SetActiveDesign=set_active)
        simulation.project = SimpleNamespace(
            project=native_project,
            proj=native_project,
            # Mirrors the misleading dynamic attribute seen on pyProject.
            oproject=lambda: None,
        )

        result = simulation._add_field_expression(
            "P_test", lambda reporter: reporter.EnterQty("EMLoss"), retry_delay=0
        )

        self.assertEqual(result, "P_test")
        self.assertEqual(active_calls, ["maxwell_loss", "maxwell_loss"])
        self.assertEqual(reporters[1].added, ["P_test"])

    def test_save_calculation_registers_expressions_on_simulation(self):
        class Reporter:
            def __init__(self):
                self.assignment = None

            def EnterQty(self, quantity):
                self.quantity = quantity

            def EnterVol(self, assignment):
                self.assignment = assignment

            def CalcOp(self, operation):
                self.operation = operation

        obj = lambda name: SimpleNamespace(name=name)
        simulation = Simulation.__new__(Simulation)
        simulation.df_plus = pd.DataFrame({"N1_side": [0], "N2_side": [0]})
        simulation.design1 = SimpleNamespace(
            Tx_windings_main=[obj("tx_inner"), obj("tx_outer")],
            Tx_windings_side=[],
            Rx_windings_main=[obj("rx_inner"), obj("rx_outer")],
            Rx_windings_side=[],
            core_plates=[obj("core_plate_0")],
            wcp_plates=[obj("wcp_0")],
        )
        registered = {}

        def add_expression(name, builder):
            reporter = Reporter()
            builder(reporter)
            registered[name] = (
                reporter.quantity, reporter.assignment, reporter.operation
            )
            return name

        simulation._add_field_expression = add_expression
        simulation._export_field_report = lambda _name, expressions: pd.DataFrame(
            [{expression: float(index + 1) for index, expression in enumerate(expressions)}]
        )

        simulation.save_calculation()

        self.assertEqual(
            registered["P_Tx_main_winding_inner"],
            ("EMLoss", "tx_inner", "Integrate"),
        )
        self.assertEqual(
            registered["P_core_plate_0"],
            ("EMLoss", "core_plate_0", "Integrate"),
        )
        self.assertEqual(simulation.df_calculator3["P_core_plate"].iloc[0], 5.0)
        self.assertEqual(simulation.df_calculator3["P_winding_plate"].iloc[0], 6.0)

    def test_rx_end_turn_selection_removes_overlaps(self):
        turns = [SimpleNamespace(name=f"rx_{index}") for index in range(3)]

        overlapping = Simulation._select_explicit_turns(turns, 2)
        all_turns = Simulation._select_explicit_turns(turns, -1)

        self.assertEqual([turn.name for turn in overlapping], ["rx_0", "rx_1", "rx_2"])
        self.assertEqual([turn.name for turn in all_turns], ["rx_0", "rx_1", "rx_2"])
        self.assertEqual(Simulation._select_explicit_turns(turns, 0), [])


class FailureLogTests(unittest.TestCase):
    def test_jsonl_accepts_schema_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failures.jsonl"
            log_failed_sample(pd.DataFrame([{"a": 1}]), "validation: bad", str(path))
            log_failed_sample(pd.DataFrame([{"a": 2, "new": 3}]), "runtime: failed", str(path))

            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["parameters"], {"a": 1})
        self.assertEqual(records[1]["parameters"], {"a": 2, "new": 3})
        self.assertEqual(records[1]["failure_stage"], "runtime")


class ThermalCompletionPolicyTests(unittest.TestCase):
    def test_only_complete_finite_thermal_rows_advance_thermal_count(self):
        valid = pd.DataFrame({
            "thermal_solved": [1],
            "thermal_extraction_complete": [1],
            "thermal_required_group_mask": [15],
            "T_max_Tx": [80.0],
            "T_max_Rx_main": [81.0],
            "T_max_Rx_side": [82.0],
            "T_max_core": [83.0],
        })
        self.assertTrue(_thermal_result_is_valid(valid))
        invalid = valid.copy()
        invalid.loc[0, "T_max_core"] = float("nan")
        self.assertFalse(_thermal_result_is_valid(invalid))
        side_optional = valid.copy()
        side_optional.loc[0, "thermal_required_group_mask"] = 11
        side_optional.loc[0, "T_max_Rx_side"] = float("nan")
        self.assertTrue(_thermal_result_is_valid(side_optional))
        self.assertFalse(_thermal_result_is_valid(pd.DataFrame({"thermal_solved": [0]})))
        self.assertFalse(_thermal_result_is_valid(pd.DataFrame({"other": [1]})))
        self.assertFalse(_thermal_result_is_valid(None))

    def test_thermal_exception_row_preserves_failure_provenance(self):
        frame = _thermal_failure_frame(RuntimeError("source assignment failed"))

        self.assertEqual(frame["thermal_solved"].iloc[0], 0)
        self.assertEqual(frame["thermal_extraction_complete"].iloc[0], 0)
        self.assertEqual(frame["thermal_required_group_mask"].iloc[0], 15)
        self.assertEqual(frame["thermal_required_missing_count"].iloc[0], 4)
        self.assertEqual(frame["thermal_error_type"].iloc[0], "RuntimeError")
        self.assertIn("source assignment failed", frame["thermal_error_message"].iloc[0])

    def test_short_batch_is_not_reported_as_complete(self):
        self.assertEqual(_completion_exit_code(8, 8), 0)
        self.assertEqual(_completion_exit_code(7, 8), 1)
        self.assertEqual(_completion_exit_code(1, 8), 1)
        self.assertEqual(_completion_exit_code(0, 8), 1)


if __name__ == "__main__":
    unittest.main()
