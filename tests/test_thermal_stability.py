import math
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from module import thermal_260706 as thermal


class _Object:
    def __init__(self, name, volume=1.0):
        self.name = name
        self.is3d = True
        self.volume = volume


class _Boundary:
    def __init__(self, props=None):
        self.props = props or {}


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
    def __init__(self, fail_boundary=None, fixed_temperature=None):
        self.fail_boundary = fail_boundary
        self.fixed_temperature = fixed_temperature
        self.source_calls = []
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
        return _Boundary({
            "Objects": list(kwargs["assignment"]),
            "Thermal Condition": kwargs["thermal_condition"],
            "Temperature": temperature,
        })

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
    def _run(
        self,
        analyze_result,
        responses,
        include_side=False,
        setup_result=True,
        setup_update_result=True,
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
            stack.enter_context(patch("time.sleep"))
            stack.enter_context(patch.object(thermal.os, "name", "posix"))
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
        self.assertEqual(ipk.ambient_calls, [50.0])

    def test_fixed_temperature_props_mismatch_fails_hard(self):
        ipk = _BoundaryIcepak(fixed_temperature="AmbientTemp")
        objs = {"core_plates": [_Object("core_plate")], "wcp_plates": []}
        with self.assertRaisesRegex(RuntimeError, "property 'Temperature' mismatch"):
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
