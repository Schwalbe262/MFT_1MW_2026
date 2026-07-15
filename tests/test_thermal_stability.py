import json
import math
import os
import tempfile
import unittest
from contextlib import ExitStack, nullcontext
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


class _NoLiveDimensionObject:
    """Object identity whose editor-backed dimension must never be queried."""

    def __init__(self, name, volume=1.0):
        self.name = name
        self.volume = volume

    @property
    def is3d(self):
        raise AssertionError("post-solve Object3d.is3d query is forbidden")


class _Material:
    pass


class _Materials:
    def __init__(self):
        self.material_keys = {}

    def add_material(self, name):
        material = _Material()
        self.material_keys[name] = material
        return material


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
        self.analyze_calls = []

    def GetName(self):
        return self._name

    def GetDesignType(self):
        return "Icepak"

    def GetModule(self, name):
        if name == "AnalysisSetup":
            return SimpleNamespace(GetSetups=lambda: ["ThermalSetup"])
        return SimpleNamespace()

    def Analyze(self, setup_name, blocking):
        self.analyze_calls.append((setup_name, blocking))
        return 0


class _ProjectHandle:
    def __init__(self, name="thermal_test"):
        self._name = name
        self.active_calls = 0
        self.last_design = None

    def GetName(self):
        return self._name

    def SetActiveDesign(self, name):
        self.active_calls += 1
        self.last_design = _DesignHandle(name)
        return self.last_design


class _SingleActivationProjectHandle(_ProjectHandle):
    """Models a native project proxy that becomes stale while Analyze runs."""

    def SetActiveDesign(self, name):
        if self.active_calls:
            raise RuntimeError("stale q7 project proxy reused after native solve")
        return super().SetActiveDesign(name)


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
    def __init__(
        self, analyze_result, field_summary_responses, setup_result=True,
        setup_update_result=True, scalar_responses=None,
    ):
        self.analyze_result = analyze_result
        self.analyze_calls = 0
        self.field_summary_calls = 0
        self.field_summary_responses = field_summary_responses
        self.scalar_responses = scalar_responses or {}
        self.scalar_calls = []
        self.setup_result = setup_result
        self.setup_update_result = setup_update_result
        self.design_name = "icepak_thermal"
        self.setup_names = ["ThermalSetup"]
        self.existing_analysis_sweeps = ["ThermalSetup : SteadyState"]
        self.mesh = SimpleNamespace(assign_mesh_level=Mock())
        self.post = SimpleNamespace(
            create_field_summary=lambda: _FieldSummary(self),
            get_scalar_field_value=self._get_scalar_field_value,
        )
        self._oproject = _ProjectHandle()
        self.odesktop = SimpleNamespace(
            AreThereSimulationsRunning=lambda: False,
            GetMessages=lambda *_args: [],
        )
        self._odesign = _DesignHandle("stale_design")
        self.design_solutions = SimpleNamespace(_odesign=self._odesign)

    @property
    def oproject(self):
        # PyAEDT exposes the native project through its refreshed private
        # handle; model that indirection so a postflight rebind is observable.
        return self._oproject

    def _get_scalar_field_value(self, _quantity, **kwargs):
        key = (kwargs["object_name"], kwargs["scalar_function"])
        self.scalar_calls.append((key, kwargs["object_type"], kwargs["solution"]))
        value = self.scalar_responses.get(key)
        if isinstance(value, Exception):
            raise value
        return value

    def create_setup(self, name):
        if not self.setup_result:
            return False
        self.setup = _Setup(self.setup_update_result)
        self.setup.name = name
        return self.setup

    def analyze(self, **_kwargs):
        index = self.analyze_calls
        self.analyze_calls += 1
        result = self.analyze_result
        if isinstance(result, (list, tuple)):
            result = result[min(index, len(result) - 1)]
        if isinstance(result, Exception):
            raise result
        return result


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

    @staticmethod
    def _monitor_failure(reason):
        return {
            "thermal_convergence_available": 0,
            "thermal_converged": 0,
            "thermal_iterations": 0,
            "thermal_residual_continuity": float("nan"),
            "thermal_residual_x_velocity": float("nan"),
            "thermal_residual_y_velocity": float("nan"),
            "thermal_residual_z_velocity": float("nan"),
            "thermal_residual_energy": float("nan"),
            "thermal_residual_flow_limit": float("nan"),
            "thermal_residual_energy_limit": float("nan"),
            "thermal_convergence_reason": reason,
            "thermal_monitor_file": "",
        }

    def _run(
        self,
        analyze_result,
        responses,
        include_side=False,
        n1_side=0,
        probe_names=(),
        setup_result=True,
        setup_update_result=True,
        convergence=None,
        tx_count=1,
        scalar_responses=None,
        core_k_anisotropic=1,
        pooled=False,
        rebind_sequence=None,
        object_factory=None,
        probe_factory=None,
    ):
        ipk = _Icepak(
            analyze_result,
            responses,
            setup_result=setup_result,
            setup_update_result=setup_update_result,
            scalar_responses=scalar_responses,
        )
        wrapper = _DesignWrapper(ipk)
        project = SimpleNamespace(create_design=lambda **_kwargs: wrapper)
        if rebind_sequence is None:
            rebind_project = Mock(return_value=_ProjectHandle("thermal_test"))
        else:
            rebind_project = Mock(side_effect=list(rebind_sequence))
        sim = SimpleNamespace(
            project=project,
            _rebind_native_project_for_design_creation=rebind_project,
            df_plus=pd.DataFrame({
                "thermal_symmetry": ["eighth"],
                "thermal_max_iterations": [100],
                "N1_side": [n1_side],
                "N2_side": [1 if include_side else 0],
                "n_explicit_turns": [1],
                "core_k_anisotropic": [core_k_anisotropic],
                "core_k_thermal": [2.0],
                "core_lamination_factor": [0.85],
                "core_k_alloy": [9.0],
                "core_k_interlayer": [0.2],
            }),
            input_df=pd.DataFrame([{}]),
            NUM_CORE=4,
            PROJECT_NAME="thermal_test",
            save_project=Mock(),
            _ensure_pooled_shared_results_directory=Mock(),
            aedt_native_solve_window=Mock(return_value=nullcontext()),
            solver_may_be_running=False,
        )
        object_factory = object_factory or _Object
        probe_factory = probe_factory or (
            lambda name: SimpleNamespace(name=name, is3d=False)
        )
        side = [object_factory("Rx_side_0")] if include_side else []
        objects = {
            "Tx": [object_factory(f"Tx_main_{index}") for index in range(tx_count)],
            "Rx_main_explicit": [object_factory("Rx_main_0")],
            "Rx_main_blocks": [],
            "Rx_side_explicit": side,
            "Rx_side_blocks": [],
            "Rx_side2_explicit": [],
            "Rx_side2_blocks": [],
            "core": [object_factory("core_1")],
            "wcp_pads": [],
            "core_pads": [],
        }
        probe_sheets = [probe_factory(name) for name in probe_names]

        def record_mock_power_balance(_ipk, target_sim, _objects, **_kwargs):
            target_sim.thermal_rx_model = "hybrid_explicit"
            target_sim.thermal_rx_power_balance = [{
                "group": "P_Rx_main_group",
                "name_hint": "Rx_main_0",
                "expected_w": 0.0,
                "assigned_w": 0.0,
            }]
            # The real loss allocator always emits the native-core transport
            # contract. This helper replaces that allocator, so provide the
            # same neutral legacy telemetry explicitly.
            target_sim.thermal_core_loss_contract_version = "legacy_test"
            target_sim.thermal_core_loss_source = "legacy_test"
            target_sim.thermal_core_loss_correction_factor = 1.0
            target_sim.thermal_core_expected_injected_w = 0.0
            target_sim.thermal_core_requested_wrapper_echo_w = 0.0
            target_sim.thermal_core_native_readback_w = 0.0
            target_sim.thermal_core_restore_factor = 1.0
            target_sim.thermal_core_native_restored_full_w = 0.0
            target_sim.thermal_core_full_expected_margin_adjusted_w = 0.0
            target_sim.thermal_core_native_restored_rel_error = 0.0
            target_sim.thermal_core_native_readback_count = 0
            target_sim.thermal_core_power_balance_abs_error_w = 0.0
            target_sim.thermal_core_power_balance_rel_error = 0.0
            return {}

        with ExitStack() as stack:
            from module import aedt_pool_adapter

            stack.enter_context(patch.object(
                aedt_pool_adapter, "pooled_backend_enabled", return_value=pooled
            ))
            stack.enter_context(patch.object(thermal, "set_design_variables"))
            stack.enter_context(patch.object(thermal, "_create_thermal_materials"))
            stack.enter_context(patch.object(thermal, "_build_geometry", return_value=objects))
            stack.enter_context(patch.object(
                thermal, "_create_probe_sheets", return_value=probe_sheets))
            stack.enter_context(patch.object(
                thermal, "_assign_losses", side_effect=record_mock_power_balance))
            stack.enter_context(patch.object(thermal, "_assign_boundaries"))
            stack.enter_context(patch.object(thermal, "_assign_thermal_mesh"))
            if isinstance(convergence, (list, tuple)):
                convergence_values = list(convergence)
                convergence_index = {"value": 0}

                def next_convergence(*_args, **_kwargs):
                    index = min(
                        convergence_index["value"],
                        len(convergence_values) - 1,
                    )
                    convergence_index["value"] += 1
                    return convergence_values[index]

                telemetry = stack.enter_context(patch.object(
                    thermal,
                    "_thermal_convergence_telemetry",
                    side_effect=next_convergence,
                ))
            else:
                telemetry = stack.enter_context(patch.object(
                    thermal,
                    "_thermal_convergence_telemetry",
                    return_value=convergence or self._convergence(),
                ))

            def poll_once(target_sim, target_ipk, target_setup, snapshot, **_kwargs):
                value = thermal._thermal_convergence_telemetry(
                    target_sim,
                    target_ipk,
                    target_setup,
                    attempts=1,
                    monitor_snapshot=snapshot,
                )
                return value, False, ""

            stack.enter_context(patch.object(
                thermal,
                "_poll_thermal_dispatch_evidence",
                side_effect=poll_once,
            ))
            sleeper = stack.enter_context(patch("time.sleep"))
            result = thermal.run_thermal_analysis(sim)
        ipk.telemetry_mock = telemetry
        ipk.sleep_mock = sleeper
        ipk.rebind_project_mock = rebind_project
        self.assertGreaterEqual(rebind_project.call_count, 2)
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

    @staticmethod
    def _core_k_frame(anisotropic=1, legacy=2.0):
        return pd.DataFrame({
            "core_k_anisotropic": [anisotropic],
            "core_k_thermal": [legacy],
            "core_lamination_factor": [0.85],
            "core_k_alloy": [9.0],
            "core_k_interlayer": [0.2],
            "k_ins": [0.2],
            "cw2": [0.665],
            "gap2": [0.339],
        })

    def test_wound_core_rule_of_mixtures_and_piece_orientation(self):
        k_inplane, k_throughstack = (
            thermal._derive_wound_core_conductivity(0.85, 9.0, 0.2)
        )
        self.assertAlmostEqual(k_inplane, 7.68)
        self.assertAlmostEqual(k_throughstack, 1.1842105263157894)

        for name in (
                "core_1_leg_left", "core_2_leg_center",
                "core_3_leg_right"):
            self.assertEqual(
                thermal._core_thermal_material_for_piece(name),
                "core_amorphous_thermal_leg",
            )
        for name in ("core_1_yoke_top", "core_3_yoke_bottom"):
            self.assertEqual(
                thermal._core_thermal_material_for_piece(name),
                "core_amorphous_thermal_yoke",
            )
        with self.assertRaisesRegex(ValueError, "unrecognized segmented"):
            thermal._core_thermal_material_for_piece("core_1")

        materials = _Materials()
        thermal._create_thermal_materials(
            SimpleNamespace(materials=materials), self._core_k_frame()
        )
        self.assertEqual(
            materials.material_keys[
                "core_amorphous_thermal_leg"
            ].thermal_conductivity,
            [k_throughstack, k_inplane, k_inplane],
        )
        self.assertEqual(
            materials.material_keys[
                "core_amorphous_thermal_yoke"
            ].thermal_conductivity,
            [k_inplane, k_inplane, k_throughstack],
        )
        self.assertNotIn("core_amorphous_thermal", materials.material_keys)

    def test_legacy_core_material_path_remains_scalar(self):
        frame = self._core_k_frame(anisotropic=0, legacy=2.75)
        # These anchors are irrelevant when the explicit legacy opt-out is set.
        frame["core_k_alloy"] = -1.0
        frame["core_k_interlayer"] = -1.0
        contract = thermal._core_thermal_conductivity_contract(frame)
        self.assertEqual(contract["thermal_core_conductivity_model"], "isotropic_legacy")
        self.assertEqual(contract["thermal_core_k_inplane"], 2.75)
        self.assertEqual(contract["thermal_core_k_throughstack"], 2.75)

        materials = _Materials()
        thermal._create_thermal_materials(
            SimpleNamespace(materials=materials), frame
        )
        self.assertEqual(
            materials.material_keys[
                "core_amorphous_thermal"
            ].thermal_conductivity,
            2.75,
        )
        self.assertNotIn(
            "core_amorphous_thermal_leg", materials.material_keys
        )
        self.assertNotIn(
            "core_amorphous_thermal_yoke", materials.material_keys
        )

    def test_thermal_geometry_segments_only_the_anisotropic_core(self):
        def core_call(anisotropic):
            frame = pd.DataFrame({
                "core_k_anisotropic": [anisotropic],
                "core_k_thermal": [2.0],
                "core_lamination_factor": [0.85],
                "core_k_alloy": [9.0],
                "core_k_interlayer": [0.2],
                "n_explicit_turns": [1],
                "l1": [89.0],
                "l2": [236.5],
                "nwh1": [284.5],
                "nwh2": [284.5],
                "n_core_group": [1],
                "core_plate_on": [0],
                "core_plate_pad_t": [0.0],
                "wcp_on": [0],
                "wcp_pad_t": [0.0],
                "N1_main": [1],
                "N2_side": [0],
                "nwl1_main": [100.0],
                "wff1_main": [0.5],
                "sl1_main_x": [100.0],
                "sl1_main_y": [100.0],
            })
            core = _Object("core_1")
            tx = _Object("Tx_main_0")
            rx = _Object("Rx_main_0")
            ipk = SimpleNamespace(modeler=SimpleNamespace(
                object_names=[core.name, tx.name, rx.name]
            ))
            sim = SimpleNamespace(df_plus=frame)
            with patch.object(
                    thermal, "create_core",
                    return_value=([core], [], [])) as create_core_mock, \
                    patch.object(
                        thermal, "get_tx_y_gaps", return_value=([], [])
                    ), \
                    patch.object(
                        thermal, "create_coil",
                        return_value=([tx], None, 1.0, None, None, None),
                    ), \
                    patch.object(
                        thermal, "_build_rx_group",
                        return_value=([rx], []),
                    ):
                thermal._build_geometry(ipk, sim, mode="full")
            return create_core_mock.call_args.kwargs

        anisotropic = core_call(1)
        self.assertIs(anisotropic["segmented_lamination"], True)
        self.assertEqual(
            anisotropic["core_material_leg"],
            "core_amorphous_thermal_leg",
        )
        self.assertEqual(
            anisotropic["core_material_yoke"],
            "core_amorphous_thermal_yoke",
        )

        legacy = core_call(0)
        self.assertEqual(legacy["core_material"], "core_amorphous_thermal")
        self.assertNotIn("segmented_lamination", legacy)
        self.assertNotIn("core_material_leg", legacy)
        self.assertNotIn("core_material_yoke", legacy)

    def test_thermal_payload_tags_anisotropic_and_legacy_core_models(self):
        response = {
            "Tx_main_0": (81.0, 70.0),
            "Rx_main_0": (88.0, 72.0),
            "core_1": (91.0, 75.0),
        }
        _ipk, anisotropic = self._run(None, [response])
        self.assertEqual(
            anisotropic["thermal_core_conductivity_model"],
            "anisotropic_wound_rule_of_mixtures_v1",
        )
        self.assertAlmostEqual(anisotropic["thermal_core_k_inplane"], 7.68)
        self.assertAlmostEqual(
            anisotropic["thermal_core_k_throughstack"],
            1.1842105263157894,
        )

        _ipk, legacy = self._run(
            None, [response], core_k_anisotropic=0
        )
        self.assertEqual(
            legacy["thermal_core_conductivity_model"], "isotropic_legacy"
        )
        self.assertEqual(legacy["thermal_core_k_inplane"], 2.0)
        self.assertEqual(legacy["thermal_core_k_throughstack"], 2.0)

    @staticmethod
    def _probe_frame(n_group):
        # A compact, fully derived geometry row exercises the real probe
        # placement formulas without opening AEDT.
        from module.input_parameter_260706 import (
            create_input_parameter,
            get_drawing_default_params,
            validation_check,
        )

        params = get_drawing_default_params()
        params["n_core_group"] = n_group
        ok, frame = validation_check(
            create_input_parameter(params), strict=True
        )
        if not ok:
            raise AssertionError("probe test fixture did not validate")
        return frame

    def test_core_probe_depth_uses_core_not_central_plate(self):
        odd = self._probe_frame(3)
        self.assertEqual(thermal._core_probe_y_positions(odd, "eighth"), [0.0])

        even = self._probe_frame(4)
        stack_t = (
            float(even["core_plate_t"].iloc[0])
            + 2.0 * float(even["core_plate_pad_t"].iloc[0])
        )
        expected = 0.5 * (
            float(even["core_depth_each"].iloc[0]) + stack_t
        )
        self.assertEqual(
            thermal._core_probe_y_positions(even, "eighth"), [expected]
        )
        self.assertEqual(
            thermal._core_probe_y_positions(even, "quarter"), [expected]
        )
        self.assertEqual(
            thermal._core_probe_y_positions(even, "full"),
            [-expected, expected],
        )

    def test_probe_sheets_cover_center_and_outer_side_legs(self):
        frame = self._probe_frame(4)
        calls = []

        def create_rectangle(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(name=kwargs["name"], is3d=False, model=True)

        ipk = SimpleNamespace(
            modeler=SimpleNamespace(create_rectangle=create_rectangle)
        )
        sheets = thermal._create_probe_sheets(
            ipk, frame, {}, eighth=True, mode="eighth"
        )
        names = {sheet.name for sheet in sheets}
        self.assertIn("Tprobe_core_center_leg", names)
        self.assertIn("Tprobe_core_side_leg", names)
        self.assertIn("Tprobe_core_top_yoke", names)
        self.assertIn("Tprobe_Rx_side_side", names)
        self.assertIn("Tprobe_Rx_side1_inner", names)
        self.assertNotIn("Tprobe_Rx_side2_side", names)
        by_name = {call["name"]: call for call in calls}
        center = by_name["Tprobe_core_center_leg"]
        side = by_name["Tprobe_core_side_leg"]
        top_yoke = by_name["Tprobe_core_top_yoke"]
        expected_y = thermal._core_probe_y_positions(frame, "eighth")[0]
        self.assertAlmostEqual(float(center["origin"][1][:-2]), expected_y)
        self.assertAlmostEqual(float(side["origin"][1][:-2]), expected_y)
        self.assertAlmostEqual(float(top_yoke["origin"][1][:-2]), expected_y)
        l1 = float(frame["l1"].iloc[0])
        l2 = float(frame["l2"].iloc[0])
        self.assertAlmostEqual(
            float(side["origin"][0][:-2]),
            -(2.0 * l1 + l2) + 0.05 * l1,
        )
        self.assertAlmostEqual(
            float(top_yoke["origin"][2][:-2]),
            float(frame["h1"].iloc[0]) / 2.0 + 0.05 * l1,
        )
        # Negative-x side winding: outward is the more-negative radial pack;
        # inward is its +x mirror toward the transformer centre.
        outward = by_name["Tprobe_Rx_side_side"]
        inward = by_name["Tprobe_Rx_side1_inner"]
        self.assertLess(
            float(outward["origin"][0][:-2]),
            float(inward["origin"][0][:-2]),
        )
        self.assertEqual(outward["sizes"], inward["sizes"])

    def test_full_probe_sheets_mirror_inner_and_outer_side_faces(self):
        frame = self._probe_frame(4)
        calls = []

        def create_rectangle(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(name=kwargs["name"], is3d=False, model=True)

        ipk = SimpleNamespace(
            modeler=SimpleNamespace(create_rectangle=create_rectangle)
        )
        sheets = thermal._create_probe_sheets(
            ipk, frame, {}, mode="full"
        )
        names = {sheet.name for sheet in sheets}
        self.assertTrue({
            "Tprobe_Rx_side_side", "Tprobe_Rx_side1_inner",
            "Tprobe_Rx_side2_side", "Tprobe_Rx_side2_inner",
        }.issubset(names))
        by_name = {call["name"]: call for call in calls}
        left_outer = float(
            by_name["Tprobe_Rx_side_side"]["origin"][0][:-2]
        )
        left_inner = float(
            by_name["Tprobe_Rx_side1_inner"]["origin"][0][:-2]
        )
        right_outer = float(
            by_name["Tprobe_Rx_side2_side"]["origin"][0][:-2]
        )
        right_inner = float(
            by_name["Tprobe_Rx_side2_inner"]["origin"][0][:-2]
        )
        self.assertLess(left_outer, left_inner)
        self.assertGreater(right_outer, right_inner)
        left_ranges = thermal._rx_side_face_x_ranges(
            frame, -(float(frame["l1"].iloc[0]) * 1.5
                     + float(frame["l2"].iloc[0]))
        )
        right_ranges = thermal._rx_side_face_x_ranges(
            frame, float(frame["l1"].iloc[0]) * 1.5
            + float(frame["l2"].iloc[0])
        )
        for relation in ("outward", "inward"):
            self.assertAlmostEqual(
                left_ranges[relation][0], -right_ranges[relation][1]
            )
            self.assertAlmostEqual(
                left_ranges[relation][1], -right_ranges[relation][0]
            )

    def test_invalid_probe_span_is_recorded_before_aedt_creation(self):
        frame = self._probe_frame(4)
        frame.loc[:, "l2"] = 5.0
        calls = []

        def create_rectangle(**kwargs):
            calls.append(kwargs["name"])
            return SimpleNamespace(name=kwargs["name"], is3d=False, model=True)

        ipk = SimpleNamespace(
            modeler=SimpleNamespace(create_rectangle=create_rectangle)
        )
        sheets = thermal._create_probe_sheets(
            ipk, frame, {}, eighth=True, mode="eighth"
        )

        self.assertNotIn("Tprobe_core_top_yoke", calls)
        self.assertIn("Tprobe_core_top_yoke", sheets.expected_names)
        failure = next(
            item for item in sheets.failures
            if item["probe"] == "Tprobe_core_top_yoke"
        )
        self.assertEqual(failure["stage"], "geometry")
        self.assertEqual(failure["reason"], "invalid_rectangle")

    def test_core_probe_aggregates_center_and_side_legs(self):
        complete = {
            "Tx_main_0": (81.0, 70.0),
            "Rx_main_0": (88.0, 72.0),
            "core_1": (91.0, 75.0),
            "Tprobe_core_center_leg": (84.0, 78.0),
            "Tprobe_core_side_leg": (89.0, 81.0),
            "Tprobe_core_top_yoke": (92.0, 85.0),
        }
        _ipk, row = self._run(
            None,
            [complete],
            probe_names=[
                "Tprobe_core_center_leg",
                "Tprobe_core_side_leg",
                "Tprobe_core_top_yoke",
            ],
        )
        self.assertEqual(row["Tprobe_core_center_leg_max"], 84.0)
        self.assertEqual(row["Tprobe_core_side_leg_max"], 89.0)
        self.assertEqual(row["Tprobe_core_top_yoke_max"], 92.0)
        self.assertEqual(row["Tprobe_core_center_max"], 92.0)
        self.assertEqual(row["Tprobe_core_center_mean"], 85.0)
        self.assertEqual(row["thermal_extraction_complete"], 1)

    def test_even_full_core_probe_selects_hottest_depth_and_leg(self):
        complete = {
            "Tx_main_0": (81.0, 70.0),
            "Rx_main_0": (88.0, 72.0),
            "core_1": (91.0, 75.0),
            "Tprobe_core_center_leg_neg": (83.0, 77.0),
            "Tprobe_core_center_leg_pos": (86.0, 79.0),
            "Tprobe_core_side_leg_neg": (90.0, 82.0),
            "Tprobe_core_side_leg_pos": (88.0, 80.0),
            "Tprobe_core_top_yoke_neg": (92.0, 84.0),
            "Tprobe_core_top_yoke_pos": (91.0, 83.0),
        }
        _ipk, row = self._run(
            None,
            [complete],
            probe_names=[
                "Tprobe_core_center_leg_neg",
                "Tprobe_core_center_leg_pos",
                "Tprobe_core_side_leg_neg",
                "Tprobe_core_side_leg_pos",
                "Tprobe_core_top_yoke_neg",
                "Tprobe_core_top_yoke_pos",
            ],
        )
        self.assertEqual(row["Tprobe_core_center_leg_max"], 86.0)
        self.assertEqual(row["Tprobe_core_side_leg_max"], 90.0)
        self.assertEqual(row["Tprobe_core_top_yoke_max"], 92.0)
        self.assertEqual(row["Tprobe_core_center_max"], 92.0)
        self.assertEqual(row["Tprobe_core_center_mean"], 84.0)

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
                self.assertEqual(ipk.oproject.active_calls, 3)
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

    def test_pooled_q7_stale_solve_project_is_rebound_before_extraction(self):
        complete = {
            "Tx_main_0": (81.0, 70.0),
            "Rx_main_0": (88.0, 72.0),
            "core_1": (91.0, 75.0),
        }
        initial_project = _ProjectHandle("thermal_test")
        solve_project = _SingleActivationProjectHandle("thermal_test")
        postflight_project = _ProjectHandle("thermal_test")

        ipk, row = self._run(
            None,
            [complete],
            pooled=True,
            rebind_sequence=[
                initial_project,
                solve_project,
                postflight_project,
            ],
        )

        self.assertEqual(row["thermal_solved"], 1)
        self.assertEqual(row["thermal_extraction_complete"], 1)
        self.assertEqual(ipk.rebind_project_mock.call_count, 3)
        self.assertEqual(solve_project.active_calls, 1)
        self.assertEqual(
            solve_project.last_design.analyze_calls,
            [("ThermalSetup", True)],
        )
        self.assertIs(ipk.oproject, postflight_project)
        self.assertEqual(postflight_project.active_calls, 2)

    def test_post_solve_extraction_never_queries_object_dimension_proxy(self):
        complete = {
            "Tx_main_0": (81.0, 70.0),
            "Rx_main_0": (88.0, 72.0),
            "core_1": (91.0, 75.0),
            "Tprobe_Tx_side": (86.0, 74.0),
        }

        _ipk, row = self._run(
            None,
            [complete],
            n1_side=1,
            probe_names=["Tprobe_Tx_side"],
            object_factory=_NoLiveDimensionObject,
            probe_factory=_NoLiveDimensionObject,
        )

        self.assertEqual(row["thermal_solved"], 1)
        self.assertEqual(row["thermal_extraction_complete"], 1)
        self.assertEqual(row["Tprobe_Tx_side_max"], 86.0)
        self.assertEqual(row["T_max_core"], 91.0)

    def test_post_solve_rebind_failure_is_explicit_and_fail_closed(self):
        complete = {
            "Tx_main_0": (81.0, 70.0),
            "Rx_main_0": (88.0, 72.0),
            "core_1": (91.0, 75.0),
        }

        with self.assertRaisesRegex(
            RuntimeError,
            "thermal post-solve exact project/design rebind failed",
        ):
            self._run(
                None,
                [complete],
                rebind_sequence=[
                    _ProjectHandle("thermal_test"),
                    _ProjectHandle("thermal_test"),
                    RuntimeError("q7 exact project is no longer available"),
                ],
            )

    def test_false_with_missing_monitor_retries_once_and_accepts_fresh_convergence(self):
        complete = {
            "Tx_main_0": (81.0, 70.0),
            "Rx_main_0": (88.0, 72.0),
            "core_1": (91.0, 75.0),
        }
        ipk, row = self._run(
            [False, None],
            [complete],
            convergence=[
                self._monitor_failure("monitor_missing"),
                self._monitor_failure("monitor_missing"),
                self._convergence(),
            ],
        )
        self.assertEqual(ipk.analyze_calls, 2)
        self.assertEqual(row["thermal_solve_attempts"], 2)
        self.assertEqual(row["thermal_analyze_call_ok"], 1)
        self.assertEqual(row["thermal_analyze_return_false"], 1)
        self.assertEqual(row["thermal_solved"], 1)
        ipk.sleep_mock.assert_not_called()
        self.assertEqual(ipk.telemetry_mock.call_count, 3)

    def test_exception_with_malformed_monitor_does_not_double_dispatch(self):
        complete = {
            "Tx_main_0": (81.0, 70.0),
            "Rx_main_0": (88.0, 72.0),
            "core_1": (91.0, 75.0),
        }
        ipk, row = self._run(
            [RuntimeError("AnalyzeAll failed"), None],
            [complete],
            convergence=[
                self._monitor_failure("monitor_malformed"),
            ],
        )
        self.assertEqual(ipk.analyze_calls, 1)
        self.assertEqual(row["thermal_solve_attempts"], 1)
        self.assertEqual(row["thermal_analyze_call_ok"], 0)
        self.assertEqual(row["thermal_solved"], 0)
        self.assertEqual(row["thermal_convergence_reason"], "monitor_malformed")
        ipk.sleep_mock.assert_not_called()

    def test_retry_is_capped_at_two_attempts(self):
        ipk, row = self._run(
            [False, False],
            [{}],
            convergence=[
                self._monitor_failure("monitor_missing"),
                self._monitor_failure("monitor_missing"),
                self._monitor_failure("monitor_malformed"),
            ],
        )
        self.assertEqual(ipk.analyze_calls, 2)
        self.assertEqual(ipk.telemetry_mock.call_count, 3)
        self.assertEqual(row["thermal_solve_attempts"], 2)
        self.assertEqual(row["thermal_solved"], 0)
        ipk.sleep_mock.assert_not_called()

    def test_successful_invocation_with_missing_monitor_does_not_retry(self):
        ipk, row = self._run(
            None,
            [{}],
            convergence=self._monitor_failure("monitor_missing"),
        )
        self.assertEqual(ipk.analyze_calls, 1)
        self.assertEqual(row["thermal_solve_attempts"], 1)
        self.assertEqual(row["thermal_convergence_reason"], "monitor_missing")
        self.assertEqual(row["thermal_solved"], 0)
        ipk.sleep_mock.assert_not_called()

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
        self.assertEqual(row["thermal_solve_attempts"], 1)
        ipk.sleep_mock.assert_not_called()

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
        self.assertEqual(ipk.oproject.active_calls, 5)

    def test_one_missing_modeled_tx_turn_invalidates_tx_group(self):
        missing_second_tx = {
            "Tx_main_0": (81.0, 70.0),
            "Rx_main_0": (88.0, 72.0),
            "core_1": (91.0, 75.0),
        }
        ipk, row = self._run(
            None,
            [missing_second_tx] * 3,
            tx_count=2,
        )

        self.assertEqual(ipk.field_summary_calls, 3)
        self.assertEqual(row["thermal_solved"], 0)
        self.assertEqual(row["thermal_extraction_complete"], 0)
        self.assertEqual(row["thermal_required_missing_count"], 1)
        self.assertTrue(math.isnan(row["T_max_Tx"]))
        self.assertEqual(row["T_max_Tx_main_0"], 81.0)
        self.assertTrue(math.isnan(row["T_max_Tx_main_1"]))

    def test_missing_tx_side_probe_is_optional_only_without_tx_side_turns(self):
        complete_groups = {
            "Tx_main_0": (81.0, 70.0),
            "Rx_main_0": (88.0, 72.0),
            "core_1": (91.0, 75.0),
        }
        ipk, row = self._run(
            None,
            [complete_groups],
            n1_side=0,
            probe_names=["Tprobe_Tx_side"],
        )
        self.assertEqual(ipk.field_summary_calls, 1)
        self.assertEqual(row["thermal_solved"], 1)
        self.assertEqual(row["thermal_extraction_complete"], 1)
        self.assertEqual(row["thermal_missing_count"], 2)
        self.assertTrue(math.isnan(row["Tprobe_Tx_side_max"]))
        self.assertTrue(math.isnan(row["Tprobe_Tx_side_mean"]))

        _ipk, row = self._run(
            None,
            [complete_groups],
            n1_side=1,
            probe_names=["Tprobe_Tx_side"],
        )
        self.assertEqual(row["thermal_solved"], 1)
        self.assertEqual(row["thermal_extraction_complete"], 0)
        self.assertEqual(row["thermal_missing_count"], 2)

    def test_missing_probe_uses_bounded_saved_field_scalar_fallback(self):
        complete_groups = {
            "Tx_main_0": (81.0, 70.0),
            "Rx_main_0": (88.0, 72.0),
            "core_1": (91.0, 75.0),
        }
        ipk, row = self._run(
            None,
            [complete_groups] * 3,
            n1_side=1,
            probe_names=["Tprobe_Tx_side"],
            scalar_responses={
                ("Tprobe_Tx_side", "Maximum"): 86.0,
                ("Tprobe_Tx_side", "Mean"): 74.0,
            },
        )

        self.assertEqual(ipk.field_summary_calls, 3)
        self.assertEqual(row["thermal_calculator_attempts"], 2)
        self.assertEqual(len(ipk.scalar_calls), 2)
        self.assertTrue(all(call[1] == "surface" for call in ipk.scalar_calls))
        self.assertEqual(row["Tprobe_Tx_side_max"], 86.0)
        self.assertEqual(row["Tprobe_Tx_side_mean"], 74.0)
        self.assertEqual(row["thermal_extraction_complete"], 1)
        self.assertEqual(row["thermal_probe_failure_count"], 0)
        self.assertEqual(json.loads(row["thermal_probe_failures_json"]), [])

    def test_failed_probe_fallback_is_structured_and_stays_quarantinable(self):
        complete_groups = {
            "Tx_main_0": (81.0, 70.0),
            "Rx_main_0": (88.0, 72.0),
            "core_1": (91.0, 75.0),
        }
        ipk, row = self._run(
            None,
            [complete_groups] * 3,
            n1_side=1,
            probe_names=["Tprobe_Tx_side"],
        )

        self.assertEqual(row["thermal_extraction_complete"], 0)
        self.assertEqual(row["thermal_extraction_failure_reason"],
                         "required_probe_temperature_missing")
        self.assertEqual(row["thermal_probe_failure_count"], 1)
        [failure] = json.loads(row["thermal_probe_failures_json"])
        self.assertEqual(failure["probe"], "Tprobe_Tx_side")
        self.assertEqual(failure["stage"], "extraction")
        self.assertEqual(failure["reason"], "saved_field_fallback_exhausted")
        self.assertEqual(
            failure["columns"],
            ["Tprobe_Tx_side_max", "Tprobe_Tx_side_mean"],
        )
        self.assertEqual(len(ipk.scalar_calls), 2)

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

    def test_single_rx_turn_receives_exact_group_loss_without_turn_report(self):
        side_turn = _Object("Rx_side_0_0", volume=4.0)
        objects = self._loss_objects()
        objects["Tx"] = []
        objects["Rx_side_explicit"] = [side_turn]
        sim = SimpleNamespace(
            df_plus=pd.DataFrame({"n_core_group": [0], "n_explicit_turns": [0]}),
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

        self.assertEqual(injected, {"Rx_side_0_0": 25.0})
        self.assertEqual(sim.thermal_rx_power_balance[-1]["assigned_w"], 25.0)
        self.assertEqual(sim.thermal_rx_model, "homogenized_blocks")

    def test_segmented_core_loss_is_volume_weighted_once_per_group(self):
        def assign_solid_block(name, power):
            return _Boundary({
                "Block Type": "Solid",
                "Objects": [name],
                "Total Power": power,
            })

        frame = pd.DataFrame({
            "n_core_group": [1],
            "w1": [8.0],
            "core_plate_t": [0.0],
            "core_plate_pad_t": [0.0],
            "n_explicit_turns": [0],
        })
        pieces = [
            _Object("core_1_leg_left", volume=1.0),
            _Object("core_1_leg_center", volume=1.0),
            _Object("core_1_yoke_top", volume=2.0),
        ]
        objects = self._loss_objects()
        objects["Tx"] = []
        objects["core"] = pieces
        sim = SimpleNamespace(
            df_plus=frame,
            loss_map_phys={"P_Rx_main_group": 0.0, "P_core_1": 80.0},
        )
        injected = thermal._assign_losses(
            SimpleNamespace(assign_solid_block=assign_solid_block),
            sim,
            objects,
            mode="eighth",
        )
        self.assertEqual(
            injected,
            {
                "core_1_leg_left": 2.5,
                "core_1_leg_center": 2.5,
                "core_1_yoke_top": 5.0,
            },
        )
        self.assertEqual(sum(injected.values()), 10.0)
        self.assertEqual(sim.thermal_core_expected_injected_w, 10.0)

        # The unsegmented branch must not require a volume and must retain the
        # exact pre-extension source assignment.
        legacy_objects = self._loss_objects()
        legacy_objects["Tx"] = []
        legacy_objects["core"] = [SimpleNamespace(name="core_1")]
        legacy_sim = SimpleNamespace(
            df_plus=frame,
            loss_map_phys={"P_Rx_main_group": 0.0, "P_core_1": 80.0},
        )
        legacy_injected = thermal._assign_losses(
            SimpleNamespace(assign_solid_block=assign_solid_block),
            legacy_sim,
            legacy_objects,
            mode="eighth",
        )
        self.assertEqual(legacy_injected, {"core_1": 10.0})

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
        pad_mesh_operation = SimpleNamespace(
            name="pad_mesh", props={}, auto_update=True, update=Mock(return_value=True))
        rx_main_block_mesh_operation = SimpleNamespace(
            name="rx_main_block_mesh", props={}, auto_update=True, update=Mock(return_value=True))
        rx_mesh_operation = SimpleNamespace(
            name="rx_mesh", props={}, auto_update=True, update=Mock(return_value=True))
        tx_mesh_operation = SimpleNamespace(
            name="tx_mesh", props={}, auto_update=True, update=Mock(return_value=True))
        mesh = SimpleNamespace(
            assign_mesh_level=Mock(
                side_effect=[["pad_mesh"], ["tx_mesh"], ["rx_main_block_mesh"], ["rx_mesh"]]),
            meshoperations=[
                pad_mesh_operation, tx_mesh_operation,
                rx_main_block_mesh_operation, rx_mesh_operation,
            ],
        )
        tx0 = _Object("Tx_main_0")
        tx1 = _Object("Tx_main_1")
        rx0 = _Object("Rx_main_0")
        rx1 = _Object("Rx_side_0")
        objs = {
            "Tx": [tx0, tx1],
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
            call({"Tx_main_0": 4, "Tx_main_1": 4}, name="tx_mesh_level"),
            call({"Rx_main_block": 4}, name="rx_main_block_mesh_level"),
            call({"Rx_main_0": 3, "Rx_side_0": 3}, name="rx_mesh_level"),
        ])
        pad_mesh_operation.update.assert_called_once_with()
        tx_mesh_operation.update.assert_called_once_with()
        rx_main_block_mesh_operation.update.assert_called_once_with()
        rx_mesh_operation.update.assert_called_once_with()
        for operation in (
            pad_mesh_operation, tx_mesh_operation,
            rx_main_block_mesh_operation, rx_mesh_operation,
        ):
            self.assertIs(operation.props["Mesh Object(s) Separately Enabled"], False)

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
        with self.assertRaisesRegex(RuntimeError, "rx_main_single_turn_mesh_level assignment"):
            thermal._assign_thermal_mesh(SimpleNamespace(mesh=mesh), empty)

        failed_rx_op = SimpleNamespace(
            name="rx_main_single", props={}, auto_update=True, update=Mock(return_value=False))
        mesh.assign_mesh_level.return_value = ["rx_main_single"]
        mesh.meshoperations = [failed_rx_op]
        with self.assertRaisesRegex(
                RuntimeError, "rx_main_single_turn_mesh_level mesh operation update failed"):
            thermal._assign_thermal_mesh(SimpleNamespace(mesh=mesh), empty)

        pad_only = dict(empty, Rx_main_explicit=[], wcp_pads=[_Object("pad")])
        mesh.assign_mesh_level.return_value = []
        with self.assertRaisesRegex(RuntimeError, "pad_mesh_level assignment"):
            thermal._assign_thermal_mesh(SimpleNamespace(mesh=mesh), pad_only)

        failed_pad_op = SimpleNamespace(
            name="pad_mesh", props={}, auto_update=True, update=Mock(return_value=False))
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

    def test_convergence_reader_rejects_monitors_older_than_solve_start(self):
        stable = (
            "151 Continuity(7e-4)XVelocity(3e-4)YVelocity(9e-4)"
            "ZVelocity(3e-4)Energy(4e-9)\n"
        )
        setup = SimpleNamespace(props={
            "Convergence Criteria - Flow": "0.001",
            "Convergence Criteria - Energy": "1e-07",
        })
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            design = root / "icepak_thermal.results"
            design.mkdir()
            monitor = design / "DV1_S1_MON0_V0.sd"
            monitor.write_text(stable, encoding="utf-8")
            old_ns = 1_000_000_000
            solve_start_ns = 2_000_000_000
            os.utime(monitor, ns=(old_ns, old_ns))
            ipk = SimpleNamespace(
                design_name="icepak_thermal", results_directory=str(root),
            )

            stale = thermal._thermal_convergence_telemetry(
                SimpleNamespace(project_path=None), ipk, setup,
                attempts=1, not_before_ns=solve_start_ns,
            )
            self.assertEqual(stale["thermal_convergence_reason"], "monitor_missing")
            self.assertEqual(stale["thermal_monitor_file"], "")

            fresh_ns = 3_000_000_000
            os.utime(monitor, ns=(fresh_ns, fresh_ns))
            fresh = thermal._thermal_convergence_telemetry(
                SimpleNamespace(project_path=None), ipk, setup,
                attempts=1, not_before_ns=solve_start_ns,
            )
            self.assertEqual(fresh["thermal_convergence_reason"], "converged")
            self.assertEqual(fresh["thermal_monitor_file"], monitor.name)

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
