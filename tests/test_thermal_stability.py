import math
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pandas as pd

from module import thermal_260706 as thermal


class _Object:
    def __init__(self, name, volume=1.0):
        self.name = name
        self.is3d = True
        self.volume = volume


class _Boundary:
    def __init__(self, props=None, update_result=True, post_update_props=None):
        self.props = props or {}
        self.auto_update = True
        self.update_result = update_result
        self.post_update_props = post_update_props or {}
        self.update_calls = 0

    def update(self):
        self.update_calls += 1
        self.props.update(self.post_update_props)
        return self.update_result


class _DesignHandle:
    def __init__(self, name):
        self._name = name

    def GetName(self):
        return self._name

    def GetModule(self, _name):
        return SimpleNamespace()


class _ProjectHandle:
    def __init__(self):
        self.active_calls = 0
        self.last_design = None

    def SetActiveDesign(self, name):
        self.active_calls += 1
        self.last_design = _DesignHandle(name)
        return self.last_design


class _Setup:
    def __init__(self, update_result=True):
        self.name = "ThermalSetup"
        self.props = {}
        self.update_result = update_result
        self.update_calls = 0

    def update(self):
        self.update_calls += 1
        return self.update_result


class _Face:
    def __init__(self, face_id):
        self.id = face_id


class _Region:
    def __init__(self):
        self.top_face_x = _Face(1)
        self.bottom_face_x = _Face(2)
        self.top_face_y = _Face(3)
        self.bottom_face_y = _Face(4)
        self.top_face_z = _Face(5)
        self.bottom_face_z = _Face(6)


class _BoundaryIcepak:
    def __init__(
        self,
        fail_boundary=None,
        fixed_temperature=None,
        fixed_update_result=True,
        fixed_post_update_props=None,
    ):
        self.fail_boundary = fail_boundary
        self.fixed_temperature = fixed_temperature
        self.fixed_update_result = fixed_update_result
        self.fixed_post_update_props = fixed_post_update_props
        self.source_calls = []
        self.source_boundaries = []
        self.ambient_calls = []
        self.region = _Region()
        self.modeler = SimpleNamespace(
            object_names=[],
            delete=Mock(return_value=True),
            create_air_region=Mock(return_value=self.region),
        )

    def _result(self, name):
        return False if name == self.fail_boundary else _Boundary()

    def assign_source(self, **kwargs):
        self.source_calls.append(kwargs)
        if kwargs["boundary_name"] == self.fail_boundary:
            return False
        temperature = self.fixed_temperature or kwargs["assignment_value"]
        boundary = _Boundary({
            "Objects": list(kwargs["assignment"]),
            "Thermal Condition": kwargs["thermal_condition"],
            "Temperature": temperature,
        }, self.fixed_update_result, self.fixed_post_update_props)
        self.source_boundaries.append(boundary)
        return boundary

    def set_ambient_temp(self, value):
        self.ambient_calls.append(value)

    def assign_symmetry_wall(self, **kwargs):
        return self._result(kwargs["boundary_name"])

    def assign_velocity_free_opening(self, **kwargs):
        return self._result(kwargs["boundary_name"])

    def assign_pressure_free_opening(self, **kwargs):
        return self._result(kwargs["boundary_name"])


class _FieldSummary:
    def __init__(self, icepak):
        self._icepak = icepak
        self._names = []

    def add_calculation(self, _entity, _geometry, name, _quantity):
        self._names.append(name)

    def get_field_summary_data(self, **_kwargs):
        index = self._icepak.field_summary_calls
        self._icepak.field_summary_calls += 1
        response = self._icepak.field_summary_responses[min(index, len(self._icepak.field_summary_responses) - 1)]
        if response is False:
            return False
        rows = []
        for name in self._names:
            if name not in response:
                continue
            maximum, mean = response[name]
            rows.append({
                "Entity": "Object",
                "Quantity": "Temperature",
                "Geometry Name": name,
                "Max": maximum,
                "Mean": mean,
            })
        return pd.DataFrame(rows)


class _Icepak:
    def __init__(self, analyze_result, field_summary_responses, setup_result=True, setup_update_result=True):
        self.analyze_result = analyze_result
        self.analyze_calls = 0
        self.field_summary_calls = 0
        self.field_summary_responses = field_summary_responses
        self.setup_result = setup_result
        self.setup_update_result = setup_update_result
        self.design_name = "icepak_thermal"
        self.existing_analysis_sweeps = ["ThermalSetup : SteadyState"]
        self.mesh = SimpleNamespace(assign_mesh_level=Mock())
        self.post = SimpleNamespace(create_field_summary=lambda: _FieldSummary(self))
        self.oproject = _ProjectHandle()
        self._odesign = _DesignHandle("stale_design")
        self.design_solutions = SimpleNamespace(_odesign=self._odesign)

    def create_setup(self, name):
        if not self.setup_result:
            return False
        self.setup = _Setup(self.setup_update_result)
        self.setup.name = name
        return self.setup

    def analyze(self, **_kwargs):
        self.analyze_calls += 1
        if isinstance(self.analyze_result, Exception):
            raise self.analyze_result
        return self.analyze_result


class _DesignWrapper:
    def __init__(self, solver):
        self.solver_instance = solver

    def __getattr__(self, name):
        return getattr(self.solver_instance, name)


class ThermalStabilityTest(unittest.TestCase):
    @staticmethod
    def _convergence(converged=True):
        return {
            "thermal_convergence_available": 1,
            "thermal_converged": 1 if converged else 0,
            "thermal_iterations": 151 if converged else 142,
            "thermal_residual_continuity": 7.9912e-4 if converged else 1.0657e18,
            "thermal_residual_x_velocity": 3.6686e-4 if converged else 1.1056e-1,
            "thermal_residual_y_velocity": 9.8308e-4 if converged else 5.6684e-2,
            "thermal_residual_z_velocity": 3.9535e-4 if converged else 1.4627e-1,
            "thermal_residual_energy": 4.3936e-9 if converged else 1.2163e-1,
            "thermal_residual_flow_limit": 1e-3,
            "thermal_residual_energy_limit": 1e-7,
            "thermal_convergence_reason": "converged" if converged else "residual_threshold",
            "thermal_monitor_file": "monitor.sd",
        }

    def _run(
        self,
        analyze_result,
        responses,
        include_side=False,
        setup_result=True,
        setup_update_result=True,
        convergence=None,
    ):
        ipk = _Icepak(
            analyze_result,
            responses,
            setup_result=setup_result,
            setup_update_result=setup_update_result,
        )
        wrapper = _DesignWrapper(ipk)
        project = SimpleNamespace(create_design=lambda **_kwargs: wrapper)
        sim = SimpleNamespace(
            project=project,
            df_plus=pd.DataFrame({
                "thermal_symmetry": ["eighth"],
                "thermal_max_iterations": [100],
                "N2_side": [1 if include_side else 0],
            }),
            input_df=pd.DataFrame([{}]),
            NUM_CORE=4,
            PROJECT_NAME="thermal_test",
            save_project=Mock(),
        )
        side = [_Object("Rx_side_0")] if include_side else []
        objects = {
            "Tx": [_Object("Tx_main_0")],
            "Rx_main_explicit": [_Object("Rx_main_0")],
            "Rx_main_blocks": [],
            "Rx_side_explicit": side,
            "Rx_side_blocks": [],
            "Rx_side2_explicit": [],
            "Rx_side2_blocks": [],
            "core": [_Object("core_1")],
            "wcp_pads": [],
            "core_pads": [],
        }

        with ExitStack() as stack:
            stack.enter_context(patch.object(thermal, "set_design_variables"))
            stack.enter_context(patch.object(thermal, "_create_thermal_materials"))
            stack.enter_context(patch.object(thermal, "_build_geometry", return_value=objects))
            stack.enter_context(patch.object(thermal, "_create_probe_sheets", return_value=[]))
            stack.enter_context(patch.object(thermal, "_assign_losses"))
            stack.enter_context(patch.object(thermal, "_assign_boundaries"))
            stack.enter_context(patch.object(thermal, "_assign_thermal_mesh"))
            stack.enter_context(patch.object(
                thermal,
                "_thermal_convergence_telemetry",
                return_value=convergence or self._convergence(),
            ))
            stack.enter_context(patch("time.sleep"))
            result = thermal.run_thermal_analysis(sim)
        return ipk, result.iloc[0]

    @staticmethod
    def _loss_objects():
        return {
            "Tx": [_Object("Tx_main_0")],
            "Rx_main_explicit": [],
            "Rx_main_blocks": [],
            "Rx_side_explicit": [],
            "Rx_side_blocks": [],
            "Rx_side2_explicit": [],
            "Rx_side2_blocks": [],
            "core": [],
        }

    @staticmethod
    def _boundary_sim():
        return SimpleNamespace(df_plus=pd.DataFrame({
            "plate_temp": [45.0],
            "air_temp": [50.0],
            "fan_velocity": [1.5],
            "fan_config": ["dual"],
        }))

    def test_none_and_false_analyze_returns_are_validated_from_data(self):
        complete = {
            "Tx_main_0": (81.0, 70.0),
            "Rx_main_0": (88.0, 72.0),
            "core_1": (91.0, 75.0),
        }
        for analyze_result in (None, False):
            with self.subTest(analyze_result=analyze_result):
                ipk, row = self._run(analyze_result, [complete])
                self.assertEqual(ipk.analyze_calls, 1)
                self.assertEqual(row["thermal_solved"], 1)
                self.assertEqual(row["thermal_extraction_complete"], 1)
                self.assertEqual(row["thermal_required_group_mask"], 11)
                self.assertEqual(row["thermal_required_group_count"], 3)
                self.assertTrue(math.isnan(row["T_max_Rx_side"]))
                self.assertEqual(ipk.oproject.active_calls, 2)
                self.assertIs(ipk._odesign, ipk.oproject.last_design)
                self.assertIs(ipk.design_solutions._odesign, ipk.oproject.last_design)
                self.assertFalse(
                    ipk.setup.props["Solution Initialization - Use Model Based Flow Initialization"]
                )
                self.assertEqual(ipk.setup.props["Under-relaxation - Pressure"], "0.7")
                self.assertFalse(
                    ipk.setup.props["Sequential Solve of Flow and Energy Equations"]
                )
                self.assertEqual(ipk.setup.props["Convergence Criteria - Flow"], "0.001")
                self.assertEqual(ipk.setup.props["Convergence Criteria - Energy"], "1e-07")

    def test_unconverged_residuals_skip_field_summary_and_fail_gate(self):
        complete = {
            "Tx_main_0": (81.0, 70.0),
            "Rx_main_0": (88.0, 72.0),
            "core_1": (91.0, 75.0),
        }
        ipk, row = self._run(None, [complete], convergence=self._convergence(False))
        self.assertEqual(ipk.field_summary_calls, 0)
        self.assertEqual(row["thermal_solved"], 0)
        self.assertEqual(row["thermal_converged"], 0)
        self.assertEqual(row["thermal_extraction_complete"], 0)
        self.assertEqual(row["thermal_convergence_reason"], "residual_threshold")

    def test_partial_field_summary_preserves_values_but_fails_gate(self):
        partial = {
            "Tx_main_0": (81.0, 70.0),
            "core_1": (91.0, 75.0),
        }
        ipk, row = self._run(None, [partial, partial, partial])
        self.assertEqual(ipk.analyze_calls, 1)
        self.assertEqual(ipk.field_summary_calls, 3)
        self.assertEqual(row["thermal_solved"], 0)
        self.assertEqual(row["thermal_extraction_complete"], 0)
        self.assertEqual(row["thermal_missing_count"], 2)
        self.assertEqual(row["thermal_required_missing_count"], 1)
        self.assertEqual(row["T_max_Tx"], 81.0)
        self.assertEqual(row["T_max_core"], 91.0)
        self.assertTrue(math.isnan(row["T_max_Rx_main"]))
        self.assertEqual(row["thermal_calculator_attempts"], 0)
        self.assertEqual(ipk.oproject.active_calls, 4)

    def test_report_failure_does_not_launch_another_solve(self):
        ipk, row = self._run(None, [False, False, False])
        self.assertEqual(ipk.analyze_calls, 1)
        self.assertEqual(ipk.field_summary_calls, 3)
        self.assertEqual(row["thermal_solved"], 0)
        self.assertEqual(row["thermal_solution_data_available"], 0)
        self.assertEqual(row["thermal_missing_count"], 6)

    def test_side_requirement_comes_from_input_not_surviving_objects(self):
        missing_side = {
            "Tx_main_0": (81.0, 70.0),
            "Rx_main_0": (88.0, 72.0),
            "core_1": (91.0, 75.0),
        }
        _ipk, row = self._run(None, [missing_side] * 3, include_side=True)
        self.assertEqual(row["thermal_required_group_mask"], 15)
        self.assertEqual(row["thermal_required_group_count"], 4)
        self.assertEqual(row["thermal_required_missing_count"], 1)
        self.assertEqual(row["thermal_solved"], 0)

        complete = dict(missing_side, Rx_side_0=(89.0, 73.0))
        _ipk, row = self._run(None, [complete], include_side=True)
        self.assertEqual(row["thermal_required_group_mask"], 15)
        self.assertEqual(row["thermal_required_missing_count"], 0)
        self.assertEqual(row["thermal_solved"], 1)

    def test_setup_creation_and_update_fail_hard(self):
        with self.assertRaisesRegex(RuntimeError, "create_setup returned no ThermalSetup"):
            self._run(None, [{}], setup_result=False)
        with self.assertRaisesRegex(RuntimeError, "ThermalSetup update returned False"):
            self._run(None, [{}], setup_update_result=False)

    def test_required_loss_key_missing_fails_hard(self):
        sim = SimpleNamespace(
            df_plus=pd.DataFrame({"n_core_group": [0]}),
            loss_map_phys={},
        )
        with self.assertRaisesRegex(KeyError, "required thermal loss key missing: P_turn_Tx_main_0"):
            thermal._assign_losses(SimpleNamespace(), sim, self._loss_objects(), mode="full")

    def test_solid_block_false_return_is_not_recorded(self):
        sim = SimpleNamespace(
            df_plus=pd.DataFrame({"n_core_group": [0]}),
            loss_map_phys={"P_turn_Tx_main_0": 12.5, "P_Rx_main_group": 0.0},
        )
        ipk = SimpleNamespace(assign_solid_block=Mock(return_value=False))
        with self.assertRaisesRegex(RuntimeError, "solid block source for Tx_main_0 returned no boundary"):
            thermal._assign_losses(ipk, sim, self._loss_objects(), mode="full")
        self.assertFalse(hasattr(sim, "thermal_injected"))

    def test_solid_block_props_are_validated_before_recording(self):
        sim = SimpleNamespace(
            df_plus=pd.DataFrame({"n_core_group": [0]}),
            loss_map_phys={"P_turn_Tx_main_0": 12.5, "P_Rx_main_group": 0.0},
        )
        wrong = _Boundary({
            "Block Type": "Solid",
            "Objects": ["Tx_main_0"],
            "Total Power": "0W",
        })
        ipk = SimpleNamespace(assign_solid_block=Mock(return_value=wrong))
        with self.assertRaisesRegex(RuntimeError, "property 'Total Power' mismatch"):
            thermal._assign_losses(ipk, sim, self._loss_objects(), mode="full")
        self.assertFalse(hasattr(sim, "thermal_injected"))

    def test_solid_block_success_records_injected_loss(self):
        sim = SimpleNamespace(
            df_plus=pd.DataFrame({"n_core_group": [0]}),
            loss_map_phys={"P_turn_Tx_main_0": 12.5, "P_Rx_main_group": 0.0},
        )

        def assign_solid_block(name, power):
            return _Boundary({
                "Block Type": "Solid",
                "Objects": [name],
                "Total Power": power,
            })

        injected = thermal._assign_losses(
            SimpleNamespace(assign_solid_block=assign_solid_block),
            sim,
            self._loss_objects(),
            mode="full",
        )
        self.assertEqual(injected, {"Tx_main_0": 12.5})
        self.assertEqual(sim.thermal_injected, injected)

    def test_blocks_only_rx_side_still_receives_group_loss(self):
        side_block = _Object("Rx_side_block", volume=4.0)
        objects = self._loss_objects()
        objects["Tx"] = []
        objects["Rx_side_blocks"] = [side_block]
        sim = SimpleNamespace(
            df_plus=pd.DataFrame({"n_core_group": [0]}),
            loss_map_phys={"P_Rx_main_group": 0.0, "P_Rx_side_group": 25.0},
        )

        def assign_solid_block(name, power):
            return _Boundary({
                "Block Type": "Solid",
                "Objects": [name],
                "Total Power": power,
            })

        injected = thermal._assign_losses(
            SimpleNamespace(assign_solid_block=assign_solid_block),
            sim,
            objects,
            mode="full",
        )
        self.assertEqual(injected, {"Rx_side_block": 25.0})

    def test_fixed_temperature_uses_supported_condition_and_props(self):
        ipk = _BoundaryIcepak()
        objs = {
            "core_plates": [_Object("core_plate")],
            "wcp_plates": [_Object("wcp_plate")],
        }
        thermal._assign_boundaries(ipk, self._boundary_sim(), objs, eighth=True, mode="eighth")
        self.assertEqual(len(ipk.source_calls), 1)
        call = ipk.source_calls[0]
        self.assertEqual(call["thermal_condition"], "Temperature")
        self.assertEqual(call["assignment_value"], "45.0cel")
        boundary = ipk.source_boundaries[0]
        self.assertEqual(boundary.update_calls, 1)
        self.assertEqual(boundary.props["Thermal Condition"], "Fixed Temperature")
        self.assertEqual(boundary.props["Temperature"], "45.0cel")
        self.assertTrue(boundary.auto_update)
        self.assertEqual(ipk.ambient_calls, [50.0])

    def test_fixed_temperature_props_mismatch_fails_hard(self):
        ipk = _BoundaryIcepak(fixed_post_update_props={"Temperature": "AmbientTemp"})
        objs = {"core_plates": [_Object("core_plate")], "wcp_plates": []}
        with self.assertRaisesRegex(RuntimeError, "property 'Temperature' mismatch"):
            thermal._assign_boundaries(ipk, self._boundary_sim(), objs, eighth=True, mode="eighth")

    def test_fixed_temperature_update_false_fails_hard(self):
        ipk = _BoundaryIcepak(fixed_update_result=False)
        objs = {"core_plates": [_Object("core_plate")], "wcp_plates": []}
        with self.assertRaisesRegex(RuntimeError, "fixed temperature source.*update failed"):
            thermal._assign_boundaries(ipk, self._boundary_sim(), objs, eighth=True, mode="eighth")

    def test_fixed_temperature_false_return_fails_hard(self):
        ipk = _BoundaryIcepak(fail_boundary="cold_plates_fixed_T")
        objs = {"core_plates": [_Object("core_plate")], "wcp_plates": []}
        with self.assertRaisesRegex(RuntimeError, "fixed temperature source.*returned no boundary"):
            thermal._assign_boundaries(ipk, self._boundary_sim(), objs, eighth=True, mode="eighth")

    def test_any_opening_or_symmetry_false_return_fails_hard(self):
        objs = {"core_plates": [], "wcp_plates": []}
        for boundary_name in ("sym_x0", "outlet_z"):
            with self.subTest(boundary_name=boundary_name):
                ipk = _BoundaryIcepak(fail_boundary=boundary_name)
                with self.assertRaisesRegex(
                    RuntimeError, f"thermal boundary {boundary_name} returned no boundary"
                ):
                    thermal._assign_boundaries(ipk, self._boundary_sim(), objs, eighth=True, mode="eighth")

    def test_required_geometry_validates_full_side_groups_separately(self):
        base = {
            "core": [_Object("core")],
            "Tx": [_Object("Tx")],
            "Rx_main_explicit": [_Object("Rx_main")],
            "Rx_main_blocks": [],
            "Rx_side_explicit": [_Object("Rx_side")],
            "Rx_side_blocks": [],
            "Rx_side2_explicit": [_Object("Rx_side2")],
            "Rx_side2_blocks": [],
        }
        thermal._require_thermal_geometry(base, "full", 1)

        missing_side2 = dict(base, Rx_side2_explicit=[])
        with self.assertRaisesRegex(RuntimeError, "Rx_side2"):
            thermal._require_thermal_geometry(missing_side2, "full", 1)

        missing_side = dict(base, Rx_side_explicit=[])
        with self.assertRaisesRegex(RuntimeError, "Rx_side"):
            thermal._require_thermal_geometry(missing_side, "full", 1)

        thermal._require_thermal_geometry(missing_side2, "eighth", 1)

    def test_required_geometry_includes_enabled_cooling_hardware(self):
        base = {
            "core": [_Object("core")],
            "Tx": [_Object("Tx")],
            "Rx_main_explicit": [_Object("Rx_main")],
            "Rx_main_blocks": [],
            "Rx_side_explicit": [],
            "Rx_side_blocks": [],
            "Rx_side2_explicit": [],
            "Rx_side2_blocks": [],
            "core_plates": [],
            "core_pads": [],
            "wcp_plates": [],
            "wcp_pads": [],
        }
        for keyword, expected in (
            ("require_core_plates", "core_plates"),
            ("require_core_pads", "core_pads"),
            ("require_wcp_plates", "wcp_plates"),
            ("require_wcp_pads", "wcp_pads"),
        ):
            with self.subTest(keyword=keyword), self.assertRaisesRegex(RuntimeError, expected):
                thermal._require_thermal_geometry(base, "eighth", 0, **{keyword: True})

    def test_explicit_rx_gets_one_object_mesh_operation(self):
        pad_mesh_operation = SimpleNamespace(name="pad_mesh", update=Mock(return_value=True))
        rx_mesh_operation = SimpleNamespace(name="rx_mesh", update=Mock(return_value=True))
        mesh = SimpleNamespace(
            assign_mesh_level=Mock(side_effect=[["pad_mesh"], ["rx_mesh"]]),
            meshoperations=[pad_mesh_operation, rx_mesh_operation],
        )
        rx0 = _Object("Rx_main_0")
        rx1 = _Object("Rx_side_0")
        objs = {
            "wcp_pads": [_Object("wcp_pad")],
            "core_pads": [_Object("core_pad")],
            "Rx_main_explicit": [rx0],
            "Rx_side_explicit": [rx1, rx0],
            "Rx_side2_explicit": [],
            "Rx_main_blocks": [_Object("Rx_main_block")],
        }
        thermal._assign_thermal_mesh(SimpleNamespace(mesh=mesh), objs)

        self.assertEqual(mesh.assign_mesh_level.call_args_list, [
            call({"wcp_pad": 2, "core_pad": 2}, name="pad_mesh_level"),
            call({"Rx_main_0": 3, "Rx_side_0": 3}, name="rx_mesh_level"),
        ])
        pad_mesh_operation.update.assert_called_once_with()
        rx_mesh_operation.update.assert_called_once_with()

    def test_thermal_mesh_failures_are_not_silenced(self):
        empty = {
            "wcp_pads": [],
            "core_pads": [],
            "Rx_main_explicit": [_Object("Rx_main_0")],
            "Rx_side_explicit": [],
            "Rx_side2_explicit": [],
        }
        mesh = SimpleNamespace(
            assign_mesh_level=Mock(return_value=[]),
            meshoperations=[],
        )
        with self.assertRaisesRegex(RuntimeError, "rx_mesh_level assignment"):
            thermal._assign_thermal_mesh(SimpleNamespace(mesh=mesh), empty)

        failed_rx_op = SimpleNamespace(name="rx_mesh", update=Mock(return_value=False))
        mesh.assign_mesh_level.return_value = ["rx_mesh"]
        mesh.meshoperations = [failed_rx_op]
        with self.assertRaisesRegex(RuntimeError, "rx_mesh_level mesh operation update failed"):
            thermal._assign_thermal_mesh(SimpleNamespace(mesh=mesh), empty)

        pad_only = dict(empty, Rx_main_explicit=[], wcp_pads=[_Object("pad")])
        mesh.assign_mesh_level.return_value = []
        with self.assertRaisesRegex(RuntimeError, "pad_mesh_level assignment"):
            thermal._assign_thermal_mesh(SimpleNamespace(mesh=mesh), pad_only)

        failed_pad_op = SimpleNamespace(name="pad_mesh", update=Mock(return_value=False))
        mesh.assign_mesh_level.return_value = ["pad_mesh"]
        mesh.meshoperations = [failed_pad_op]
        with self.assertRaisesRegex(RuntimeError, "pad_mesh_level mesh operation update failed"):
            thermal._assign_thermal_mesh(SimpleNamespace(mesh=mesh), pad_only)

    def test_native_residual_parser_accepts_428_and_rejects_429(self):
        stable = (
            "1.5100000000000000e+02 Continuity(7.9911999999999995e-04)"
            "XVelocity(3.6685999999999999e-04)YVelocity(9.8308000000000011e-04)"
            "ZVelocity(3.9534999999999999e-04)Energy(4.3936000000000002e-09)\n"
        )
        divergent = (
            "1.4200000000000000e+02 Continuity(1.0657000000000000e+18)"
            "XVelocity(1.1056000000000001e-01)YVelocity(5.6683999999999998e-02)"
            "ZVelocity(1.4627000000000001e-01)Energy(1.2163000000000000e-01)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            stable_path = Path(tmp, "stable.sd")
            failed_path = Path(tmp, "failed.sd")
            stable_path.write_text(stable, encoding="utf-8")
            failed_path.write_text(divergent, encoding="utf-8")
            self.assertTrue(thermal._parse_thermal_residual_monitor(stable_path)["converged"])
            parsed = thermal._parse_thermal_residual_monitor(failed_path)
            self.assertFalse(parsed["converged"])
            self.assertEqual(parsed["iteration"], 142)
            self.assertEqual(parsed["values"]["Continuity"], 1.0657e18)

            invalid_tail = Path(tmp, "invalid_tail.sd")
            invalid_tail.write_text(
                stable
                + "152 Continuity(nan)XVelocity(1e-4)YVelocity(1e-4)"
                  "ZVelocity(1e-4)Energy(1e-9)\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-finite"):
                thermal._parse_thermal_residual_monitor(invalid_tail)

            truncated_tail = Path(tmp, "truncated_tail.sd")
            truncated_tail.write_text(
                stable + "152 Continuity(8e-4)XVelocity(4e-4)\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "incomplete"):
                thermal._parse_thermal_residual_monitor(truncated_tail)

            duplicate_tail = Path(tmp, "duplicate_tail.sd")
            duplicate_tail.write_text(
                stable
                + "152 Continuity(8e-4)Continuity(7e-4)XVelocity(4e-4)"
                  "YVelocity(9e-4)ZVelocity(4e-4)Energy(4e-9)\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "incomplete"):
                thermal._parse_thermal_residual_monitor(duplicate_tail)

    def test_convergence_reader_uses_latest_history_and_ignores_solution_monitor(self):
        stable = (
            "151 Continuity(7e-4)XVelocity(3e-4)YVelocity(9e-4)"
            "ZVelocity(3e-4)Energy(4e-9)\n"
        )
        failed = (
            "142 Continuity(1e18)XVelocity(1e-1)YVelocity(1e-1)"
            "ZVelocity(1e-1)Energy(1e-1)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            design = root / "icepak_thermal.results"
            design.mkdir()
            old = design / "DV1_S1_MON0_V0.sd"
            newest = design / "DV1_S2_MON0_V0.sd"
            ignored = design / "DV1_SOL3_MON0_V0.sd"
            old.write_text(stable, encoding="utf-8")
            newest.write_text(failed, encoding="utf-8")
            ignored.write_text(stable, encoding="utf-8")
            os.utime(old, (1, 1))
            os.utime(newest, (2, 2))
            os.utime(ignored, (3, 3))
            ipk = SimpleNamespace(
                design_name="icepak_thermal",
                results_directory=str(root),
            )
            setup = SimpleNamespace(props={
                "Convergence Criteria - Flow": "0.001",
                "Convergence Criteria - Energy": "1e-07",
            })
            result = thermal._thermal_convergence_telemetry(
                SimpleNamespace(project_path=None), ipk, setup, attempts=1
            )
            self.assertEqual(result["thermal_convergence_available"], 1)
            self.assertEqual(result["thermal_converged"], 0)
            self.assertEqual(result["thermal_monitor_file"], newest.name)

    def test_missing_or_malformed_residual_monitor_fails_closed(self):
        setup = SimpleNamespace(props={})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            design = root / "icepak_thermal.results"
            design.mkdir()
            ipk = SimpleNamespace(design_name="icepak_thermal", results_directory=str(root))
            missing = thermal._thermal_convergence_telemetry(
                SimpleNamespace(project_path=None), ipk, setup, attempts=1
            )
            self.assertEqual(missing["thermal_converged"], 0)
            self.assertEqual(missing["thermal_convergence_reason"], "monitor_missing")
            (design / "DV1_S1_MON0_V0.sd").write_text("not residual data", encoding="utf-8")
            malformed = thermal._thermal_convergence_telemetry(
                SimpleNamespace(project_path=None), ipk, setup, attempts=1
            )
            self.assertEqual(malformed["thermal_converged"], 0)
            self.assertEqual(malformed["thermal_convergence_reason"], "monitor_malformed")

    def test_split_requires_truthy_result_and_live_retained_object(self):
        obj = _Object("core_1")
        modeler = SimpleNamespace(split=Mock(return_value=False), object_names=[obj.name])
        with self.assertRaisesRegex(RuntimeError, "thermal geometry split failed"):
            thermal._split_retained(SimpleNamespace(modeler=modeler), [obj], "XY", "PositiveOnly")

        modeler = SimpleNamespace(split=Mock(return_value=[obj.name]), object_names=[])
        with self.assertRaisesRegex(RuntimeError, "retained no live objects"):
            thermal._split_retained(SimpleNamespace(modeler=modeler), [obj], "XY", "PositiveOnly")

        modeler = SimpleNamespace(split=Mock(return_value=[obj.name]), object_names=[obj.name])
        alive = thermal._split_retained(SimpleNamespace(modeler=modeler), [obj], "XY", "PositiveOnly")
        self.assertEqual(alive, [obj])


if __name__ == "__main__":
    unittest.main()
