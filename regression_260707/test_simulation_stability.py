import json
import logging
import math
import os
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

import pandas as pd

from run_simulation_260706 import (
    Simulation,
    SolutionDataUnavailableError,
    _assign_native_copied_core_loss,
    _assert_native_core_loss_assignment,
    _completion_exit_code,
    _configure_copied_loss_excitations,
    _create_simulation_session,
    _configure_copied_loss_setup,
    _configure_em_conductor_mesh,
    _configure_loss_copy_skin_mesh,
    _delete_copied_solution_or_raise,
    _em_result_is_valid,
    _em_result_validation,
    _edit_native_copied_loss_winding,
    _matrix_source_signature,
    _native_winding_child,
    _parse_rl_matrix_export,
    _remap_copied_design_objects,
    _retry_copied_loss_preparation,
    _set_copied_loss_winding_excitation,
    _thermal_failure_frame,
    _thermal_result_is_valid,
    _wait_for_ready_copied_loss_design,
    _validate_saved_copied_loss_preparation,
    log_failed_sample,
)
from module.input_parameter_260706 import (
    ALL_INPUT_KEYS,
    COLD_PLATE_MAX_T_MM,
    COLD_PLATE_MIN_T_MM,
    ELECTROSTATIC_STAGE_INPUT_KEYS,
    KEYS,
    PRE_ELECTROSTATIC_INPUT_KEYS,
    N1_MAX_TURNS,
    PRE_ANISOTROPIC_CORE_K_INPUT_KEYS,
    PRIMARY_CONDUCTOR_MAX_THICKNESS_MM,
    THERMAL_CORE_CONDUCTIVITY_INPUT_KEYS,
    WCP_LENGTH_MAX_PCT,
    WCP_LENGTH_MIN_PCT,
    _SOBOL_DIMS,
    create_input_parameter,
    decode_unit_sample,
    get_drawing_default_params,
    get_design_var_columns,
    get_tx_y_gaps,
    unit_to_dims,
    validation_check,
)
from module.modeling_260706 import (
    create_coil as create_coil_geometry,
    create_core as create_core_geometry,
    create_winding_cooling_plates,
)
from module.thermal_260706 import (
    _assign_thermal_mesh,
    _build_homog_blocks,
    _build_rx_group,
    _partition_rx_turns,
    _prepare_thermal_dispatch,
    _rx_layout,
    _snapshot_thermal_monitors,
    _solve_exact_thermal_setup,
    _thermal_convergence_telemetry,
    _volume_weighted_powers,
    run_thermal_analysis,
)
from module.core_material_contract import PHYSICS_DATA_REVISION
from module.thermal_probe_contract import (
    RX_SIDE_FACE_MAX_RULE,
    RX_SIDE_FACE_MEAN_RULE,
    RX_SIDE_FACE_PROBE_CONTRACT_VERSION,
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


class DesktopSessionRetryTests(unittest.TestCase):
    @patch("run_simulation_260706.time.sleep")
    @patch("run_simulation_260706._terminate_spawned_descendants")
    @patch(
        "run_simulation_260706._snapshot_descendants",
        side_effect=[{10: (1, 1.0)}, {11: (1, 2.0)}],
    )
    @patch("run_simulation_260706.Simulation")
    @patch("run_simulation_260706.pyDesktop")
    def test_half_created_desktop_is_cleaned_and_retried(
            self, desktop_factory, simulation_factory, snapshot,
            terminate, sleep):
        first_desktop = Mock()
        second_desktop = Mock()
        expected_simulation = Mock()
        desktop_factory.side_effect = [first_desktop, second_desktop]
        simulation_factory.side_effect = [
            AttributeError("'NoneType' object has no attribute 'EnableAutoSave'"),
            expected_simulation,
        ]

        desktop, simulation = _create_simulation_session(
            max_attempts=2,
            retry_delay_s=0.25,
        )

        self.assertIs(desktop, second_desktop)
        self.assertIs(simulation, expected_simulation)
        first_desktop.release_desktop.assert_called_once_with(
            close_projects=True,
            close_on_exit=True,
        )
        terminate.assert_called_once_with(
            {10: (1, 1.0)},
            {11: (1, 2.0)},
            wait_s=5,
        )
        sleep.assert_called_once_with(0.25)
        self.assertEqual(snapshot.call_count, 2)

    @patch("run_simulation_260706.time.sleep")
    @patch("run_simulation_260706._terminate_spawned_descendants")
    @patch(
        "run_simulation_260706._snapshot_descendants",
        side_effect=[{}, {}, {}],
    )
    @patch("run_simulation_260706.Simulation")
    @patch("run_simulation_260706.pyDesktop")
    def test_retry_exhaustion_is_explicit(
            self, desktop_factory, simulation_factory, _snapshot,
            _terminate, sleep):
        desktops = [Mock(), Mock()]
        desktop_factory.side_effect = desktops
        simulation_factory.side_effect = [RuntimeError("grpc-1"), RuntimeError("grpc-2")]

        with self.assertRaisesRegex(
                RuntimeError, "AEDT desktop startup failed after 2 attempts"):
            _create_simulation_session(max_attempts=2, retry_delay_s=1)

        for desktop in desktops:
            desktop.release_desktop.assert_called_once_with(
                close_projects=True,
                close_on_exit=True,
            )
        sleep.assert_called_once_with(1.0)

    @patch("run_simulation_260706.time.sleep")
    @patch("run_simulation_260706._snapshot_descendants", return_value={})
    @patch("run_simulation_260706.acquire_pooled_desktop")
    @patch("run_simulation_260706.aedt_backend", return_value="pooled")
    def test_pooled_attach_failure_is_one_durable_intent_not_local_retries(
            self, _backend, acquire, _snapshot, sleep):
        acquire.side_effect = RuntimeError("attach failed")

        with self.assertRaisesRegex(
                RuntimeError, "AEDT desktop startup failed after 1 attempts"):
            _create_simulation_session(max_attempts=3, retry_delay_s=30)

        acquire.assert_called_once_with(
            desktop_factory=ANY,
            non_graphical=ANY,
        )
        sleep.assert_not_called()


class ThermalCoreConductivityInputContractTests(unittest.TestCase):
    def test_controls_are_accepted_but_not_sealed_or_design_variables(self):
        defaults = get_drawing_default_params()
        self.assertEqual(defaults["core_k_anisotropic"], 1)
        self.assertEqual(defaults["core_k_alloy"], 9.0)
        self.assertEqual(defaults["core_k_interlayer"], 0.2)
        self.assertTrue(
            set(THERMAL_CORE_CONDUCTIVITY_INPUT_KEYS).issubset(ALL_INPUT_KEYS)
        )
        self.assertTrue(
            set(THERMAL_CORE_CONDUCTIVITY_INPUT_KEYS).isdisjoint(KEYS)
        )

        frame = create_input_parameter({
            "core_k_anisotropic": 0,
            "core_k_alloy": 8.5,
            "core_k_interlayer": 0.25,
        })
        self.assertEqual(int(frame["core_k_anisotropic"].iloc[0]), 0)
        self.assertEqual(float(frame["core_k_alloy"].iloc[0]), 8.5)
        self.assertEqual(float(frame["core_k_interlayer"].iloc[0]), 0.25)
        self.assertTrue(
            set(THERMAL_CORE_CONDUCTIVITY_INPUT_KEYS).isdisjoint(
                get_design_var_columns(frame)
            )
        )

        pre_extension = create_input_parameter(
            frame[PRE_ANISOTROPIC_CORE_K_INPUT_KEYS]
        )
        self.assertEqual(int(pre_extension["core_k_anisotropic"].iloc[0]), 1)
        self.assertEqual(float(pre_extension["core_k_alloy"].iloc[0]), 9.0)
        self.assertEqual(
            float(pre_extension["core_k_interlayer"].iloc[0]), 0.2
        )

    def test_controls_require_binary_flag_and_positive_finite_anchors(self):
        cases = (
            ("core_k_anisotropic", 2, "must be 0 or 1"),
            ("core_k_anisotropic", 0.5, "must be 0 or 1"),
            ("core_k_alloy", 0.0, "finite and > 0"),
            ("core_k_alloy", float("nan"), "finite and > 0"),
            ("core_k_interlayer", -0.1, "finite and > 0"),
        )
        for key, value, message in cases:
            with self.subTest(key=key, value=value):
                params = get_drawing_default_params()
                params[key] = value
                with self.assertRaisesRegex(ValueError, message):
                    validation_check(create_input_parameter(params), strict=True)


class PrimaryTurnDomainTests(unittest.TestCase):
    def test_sobol_domain_caps_total_primary_turns_at_eight(self):
        for u_n1 in (0.0, 0.2, 0.5, 0.8, 1.0):
            unit = [0.5] * len(_SOBOL_DIMS)
            unit[0] = u_n1
            decoded = decode_unit_sample(unit_to_dims(unit))
            self.assertLessEqual(
                int(decoded["N1_main"]) + int(decoded["N1_side"]),
                N1_MAX_TURNS,
            )
        self.assertEqual(
            int(decode_unit_sample(unit_to_dims([1.0] + [0.5] * (
                len(_SOBOL_DIMS) - 1)))["N1_main"]),
            N1_MAX_TURNS,
        )

    def test_fixed_input_above_eight_turns_is_rejected(self):
        params = get_drawing_default_params()
        params["N1_main"] = N1_MAX_TURNS + 1
        with self.assertRaisesRegex(
                ValueError, f"N1 {N1_MAX_TURNS + 1} > {N1_MAX_TURNS}"):
            validation_check(create_input_parameter(params), strict=True)

    def test_primary_conductor_thickness_is_capped_at_ten_mm(self):
        params = get_drawing_default_params()
        # Keep the surrounding fixed geometry roomy enough that this test
        # isolates the conductor-thickness boundary rather than x-clearance.
        params["l2"] = 300.0
        params["cw1"] = PRIMARY_CONDUCTOR_MAX_THICKNESS_MM
        ok, frame = validation_check(create_input_parameter(params), strict=True)
        self.assertTrue(ok)
        self.assertEqual(
            float(frame["cw1"].iloc[0]),
            PRIMARY_CONDUCTOR_MAX_THICKNESS_MM,
        )

        params["cw1"] = PRIMARY_CONDUCTOR_MAX_THICKNESS_MM + 0.01
        with self.assertRaisesRegex(
                ValueError, r"cw1 10\.01 > 10\.0mm"):
            validation_check(create_input_parameter(params), strict=True)

        params["cw1"] = float("nan")
        with self.assertRaisesRegex(ValueError, r"cw1 must be finite"):
            validation_check(create_input_parameter(params), strict=True)

    def test_winding_and_core_plate_thicknesses_vary_independently(self):
        low = {key: 0.5 for key, _lo, _hi in _SOBOL_DIMS}
        high = dict(low)
        low.update({"wcp_t": COLD_PLATE_MIN_T_MM,
                    "core_plate_t": COLD_PLATE_MAX_T_MM})
        high.update({"wcp_t": COLD_PLATE_MAX_T_MM,
                     "core_plate_t": COLD_PLATE_MIN_T_MM})
        decoded_low = decode_unit_sample(low)
        decoded_high = decode_unit_sample(high)
        self.assertEqual(decoded_low["wcp_t"], COLD_PLATE_MIN_T_MM)
        self.assertEqual(decoded_low["core_plate_t"], COLD_PLATE_MAX_T_MM)
        self.assertEqual(decoded_high["wcp_t"], COLD_PLATE_MAX_T_MM)
        self.assertEqual(decoded_high["core_plate_t"], COLD_PLATE_MIN_T_MM)

    def test_winding_plate_length_varies_as_percent_and_records_mm(self):
        midpoint = {
            key: lo + 0.5 * (hi - lo) for key, lo, hi in _SOBOL_DIMS
        }
        for requested_pct in (WCP_LENGTH_MIN_PCT, 50.0, WCP_LENGTH_MAX_PCT):
            sample = dict(midpoint, wcp_len_pct=requested_pct)
            decoded = decode_unit_sample(sample)
            ok, frame = validation_check(
                create_input_parameter({key: decoded[key] for key in KEYS}),
                strict=True,
            )
            self.assertTrue(ok)
            reference_mm = float(frame["wcp_len_ref_x"].iloc[0])
            length_mm = float(frame["wcp_len_x"].iloc[0])
            actual_pct = float(frame["wcp_len_pct"].iloc[0])
            self.assertAlmostEqual(
                length_mm, reference_mm * requested_pct / 100.0, delta=0.11
            )
            self.assertAlmostEqual(actual_pct, requested_pct, delta=0.05)

    def test_fixed_winding_plate_mm_is_back_converted_and_range_checked(self):
        params = get_drawing_default_params()
        ok, valid = validation_check(create_input_parameter(params), strict=True)
        self.assertTrue(ok)
        reference_mm = float(valid["wcp_len_ref_x"].iloc[0])
        self.assertAlmostEqual(
            float(valid["wcp_len_pct"].iloc[0]),
            100.0 * float(valid["wcp_len_x"].iloc[0]) / reference_mm,
        )

        params["wcp_len_x"] = 0.1 * reference_mm
        with self.assertRaisesRegex(ValueError, "wcp_len_pct.*outside"):
            validation_check(create_input_parameter(params), strict=True)

    def test_winding_plate_slot_adds_two_independent_pad_layers(self):
        params = get_drawing_default_params()
        params.update({"N1_main": 6, "N1_side": 0,
                       "wcp_t": 20.0, "wcp_pad_t": 2.0})
        gaps, slots = get_tx_y_gaps(create_input_parameter(params))
        self.assertEqual(gaps[0], 24.0)
        self.assertEqual(gaps[-1], 24.0)
        self.assertEqual(slots, [0, 4])


class ElectrostaticInputContractTests(unittest.TestCase):
    def test_optional_cap_controls_are_accepted_but_not_sealed_or_design_vars(self):
        defaults = get_drawing_default_params()
        self.assertEqual(defaults["cap_on"], 1)
        self.assertEqual(defaults["cap_max_passes"], 10)
        self.assertEqual(defaults["cap_percent_error"], 1.0)
        self.assertTrue(set(ELECTROSTATIC_STAGE_INPUT_KEYS).issubset(ALL_INPUT_KEYS))
        self.assertTrue(set(ELECTROSTATIC_STAGE_INPUT_KEYS).isdisjoint(KEYS))

        frame = create_input_parameter({
            "cap_on": 1,
            "cap_max_passes": 12,
            "cap_percent_error": 0.75,
        })
        self.assertEqual(int(frame["cap_on"].iloc[0]), 1)
        self.assertEqual(int(frame["cap_max_passes"].iloc[0]), 12)
        self.assertEqual(float(frame["cap_percent_error"].iloc[0]), 0.75)
        self.assertTrue(
            set(ELECTROSTATIC_STAGE_INPUT_KEYS).isdisjoint(
                get_design_var_columns(frame)
            )
        )

        legacy_frame = create_input_parameter(
            frame[PRE_ELECTROSTATIC_INPUT_KEYS]
        )
        self.assertEqual(int(legacy_frame["cap_on"].iloc[0]), 1)
        self.assertEqual(int(legacy_frame["cap_max_passes"].iloc[0]), 10)
        self.assertEqual(float(legacy_frame["cap_percent_error"].iloc[0]), 1.0)

    def test_cap_stage_requires_matrix_inductance(self):
        params = get_drawing_default_params()
        params.update({"cap_on": 1, "matrix_on": 0})
        with self.assertRaisesRegex(ValueError, "cap_on=1 requires matrix_on=1"):
            validation_check(create_input_parameter(params), strict=True)

    def test_cap_adaptive_controls_are_positive_and_finite(self):
        cases = (
            ("cap_max_passes", 0, "positive integer"),
            ("cap_max_passes", 1.5, "positive integer"),
            ("cap_percent_error", 0.0, "finite and > 0"),
            ("cap_percent_error", float("nan"), "finite and > 0"),
        )
        for key, value, message in cases:
            with self.subTest(key=key, value=value):
                params = get_drawing_default_params()
                params[key] = value
                with self.assertRaisesRegex(ValueError, message):
                    validation_check(create_input_parameter(params), strict=True)


class _GeometryObject:
    def __init__(self, name):
        self.name = name
        self.color = None


class _GeometryModeler:
    def __init__(self):
        self.boxes = []
        self.subtract_calls = []

    def create_box(self, origin, sizes, name, material):
        obj = _GeometryObject(name)
        self.boxes.append({"origin": origin, "sizes": sizes, "name": name,
                           "material": material, "object": obj})
        return obj

    def subtract(self, assignments, tools, keep_originals=False):
        self.subtract_calls.append((list(assignments), list(tools), keep_originals))


class ColdPlateGeometryTests(unittest.TestCase):
    def test_core_cooling_uses_two_i_plates_not_window_subtracted_u_frame(self):
        design = SimpleNamespace(modeler=_GeometryModeler())
        cores, plates, pads = create_core_geometry(
            design, n_group=1, plate_on=True, pad_on=True)
        self.assertEqual(len(cores), 1)
        # Two y stacks x (left side, center, mirrored right side). The retained
        # x<=0 solver domain contains the requested side+center pair.
        self.assertEqual(len(plates), 6)
        self.assertEqual(len(pads), 12)
        self.assertEqual(
            {plate.name.rsplit("_", 2)[-1] for plate in plates},
            {"left", "center", "right"},
        )
        plate_boxes = [box for box in design.modeler.boxes
                       if box["object"] in plates]
        pad_boxes = [box for box in design.modeler.boxes
                     if box["object"] in pads]
        center_boxes = [box for box in plate_boxes
                        if box["name"].endswith("_center")]
        side_boxes = [box for box in plate_boxes
                      if "_side_" in box["name"]]
        self.assertTrue(all(box["sizes"][0] == "2*l1"
                            for box in center_boxes))
        self.assertTrue(all(box["sizes"][0] == "l1" for box in side_boxes))
        self.assertTrue(all(box["origin"][0] == "-l1"
                            for box in center_boxes))
        self.assertEqual(
            {box["origin"][0] for box in side_boxes},
            {"-(2*l1+l2)", "(l1+l2)"},
        )
        self.assertTrue(all(box["sizes"][1] == "core_plate_t"
                            for box in plate_boxes))
        self.assertTrue(all(box["sizes"][1] == "core_plate_pad_t"
                            for box in pad_boxes))
        assignments, _tools, keep = design.modeler.subtract_calls[-1]
        self.assertEqual(assignments, cores)
        self.assertFalse(keep)

    def test_i_plate_symmetry_reconstructs_left_right_and_y_twins(self):
        simulation = Simulation.__new__(Simulation)
        simulation.loss_is_sym = True
        simulation.df_plus = pd.DataFrame([{
            "w1": 232.0,
            "n_core_group": 2,
            "core_plate_t": 20.0,
            "core_plate_pad_t": 2.0,
        }])
        self.assertEqual(simulation._sym_cut_count("core_plate_2_side_left"), 2)
        self.assertEqual(simulation._mirror_mult("core_plate_2_side_left"), 2.0)
        self.assertEqual(simulation._sym_cut_count("core_plate_1_side_left"), 1)
        self.assertEqual(simulation._mirror_mult("core_plate_1_side_left"), 4.0)
        self.assertEqual(simulation._sym_cut_count("core_plate_2_center"), 3)
        self.assertEqual(simulation._mirror_mult("core_plate_2_center"), 1.0)
        self.assertEqual(simulation._sym_cut_count("core_plate_1_center"), 2)
        self.assertEqual(simulation._mirror_mult("core_plate_1_center"), 2.0)
        self.assertEqual(
            simulation._phys_factor("P_core_plate_2_side_left", False)
            * simulation._mirror_mult("core_plate_2_side_left"),
            2.0,
        )
        self.assertEqual(
            simulation._phys_factor("P_core_plate_1_center", False)
            * simulation._mirror_mult("core_plate_1_center"),
            2.0,
        )

    def test_winding_plate_aluminum_and_pads_keep_separate_thicknesses(self):
        design = SimpleNamespace(modeler=_GeometryModeler())
        plates, pads = create_winding_cooling_plates(
            design, "Tx_main_wcp", space_width=100, coil_width=10,
            y_gaps=[24], slot_indices=[0], wcp_len_x=50,
            wcp_t=20, pad_t=2, height=40)
        self.assertEqual(len(plates), 2)
        self.assertEqual(len(pads), 4)
        plate_boxes = [box for box in design.modeler.boxes
                       if box["object"] in plates]
        pad_boxes = [box for box in design.modeler.boxes
                     if box["object"] in pads]
        self.assertTrue(all(box["sizes"][1] == "20mm" for box in plate_boxes))
        self.assertTrue(all(box["sizes"][1] == "2mm" for box in pad_boxes))


class SolutionDataTests(unittest.TestCase):
    RL_EXPORT = """Inductance Unit: nH

1000Hz
\tR,L
\t\t\tTx_winding\tRx_winding
\t\tTx_winding  1.0E-03, 9.664177569404639E+05  2.0E-03, -9.557699432250109E+06
\t\tRx_winding  2.0E-03, -9.557699432250109E+06  8.0E-02, 9.634317587777689E+07
"""

    def test_parses_and_validates_official_rl_matrix_export(self):
        row = _parse_rl_matrix_export(self.RL_EXPORT, 1000.0)

        self.assertAlmostEqual(row["Ltx"], 966.4177569404639)
        self.assertAlmostEqual(row["M"], 9557.699432250109)
        self.assertAlmostEqual(row["Llt"], 18.24869768999621)

    def test_rl_matrix_export_rejects_asymmetry_and_wrong_frequency(self):
        asymmetric = self.RL_EXPORT.replace(
            "2.0E-03, -9.557699432250109E+06  8.0E-02",
            "2.0E-03, -8.557699432250109E+06  8.0E-02",
        )
        with self.assertRaisesRegex(RuntimeError, "not symmetric"):
            _parse_rl_matrix_export(asymmetric, 1000.0)
        with self.assertRaisesRegex(RuntimeError, "no 900Hz"):
            _parse_rl_matrix_export(self.RL_EXPORT, 900.0)

    def test_rl_matrix_export_rejects_unknown_unit_and_cross_frequency_section(self):
        with self.assertRaisesRegex(RuntimeError, "unsupported inductance unit"):
            _parse_rl_matrix_export(
                self.RL_EXPORT.replace("Inductance Unit: nH", "Inductance Unit: furlong"),
                1000.0,
            )

        missing_target_section = self.RL_EXPORT.replace("\tR,L", "") + "\n2000Hz\n\tR,L\n"
        with self.assertRaisesRegex(RuntimeError, "no R,L section"):
            _parse_rl_matrix_export(missing_target_section, 1000.0)

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
        post = _FakePost([_FakeSolution(values)])
        simulation = _simulation_with_post(post)

        frame = simulation.get_magnetic_parameter()

        self.assertEqual(frame["M"].iloc[0], 28.5)
        self.assertEqual(post.calls[0]["report_category"], "AC Magnetic")
        self.assertEqual(post.calls[0]["context"], "Matrix")

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


class CopiedLossMeshPolicyTests(unittest.TestCase):
    class _Winding:
        def __init__(self, name, update_result=True):
            self.name = name
            self.props = {"IsSolid": False}
            self.update_result = update_result
            self.update_calls = 0

        def update(self):
            self.update_calls += 1
            return self.update_result

    @staticmethod
    def _simulation(matrix_skin_mesh):
        calls = []

        def assign_skin_depth():
            calls.append(("skin", {}))
            return 2

        def assign_plate_settings(**kwargs):
            calls.append(("plates", kwargs))
            return 5

        return SimpleNamespace(
            df_plus=pd.DataFrame({"matrix_skin_mesh": [matrix_skin_mesh]}),
            tx_winding=CopiedLossMeshPolicyTests._Winding("Tx"),
            rx_winding=CopiedLossMeshPolicyTests._Winding("Rx"),
            assign_skin_depth=assign_skin_depth,
            assign_plate_settings=assign_plate_settings,
        ), calls

    def test_reuses_winding_mesh_inherited_from_matrix(self):
        simulation, calls = self._simulation(1)

        assigned = _configure_loss_copy_skin_mesh(simulation)

        self.assertFalse(assigned)
        self.assertEqual(calls, [])
        self.assertFalse(simulation.tx_winding.props["IsSolid"])

    def test_restores_solid_windings_and_all_skin_mesh_for_loss(self):
        simulation, calls = self._simulation(0)

        assigned = _configure_loss_copy_skin_mesh(simulation)

        self.assertTrue(assigned)
        self.assertTrue(simulation.tx_winding.props["IsSolid"])
        self.assertTrue(simulation.rx_winding.props["IsSolid"])
        self.assertEqual(simulation.tx_winding.props["Resistance"], "0ohm")
        self.assertEqual(simulation.rx_winding.props["ParallelBranchesNum"], "1")
        self.assertEqual(simulation.tx_winding.update_calls, 1)
        self.assertEqual(simulation.rx_winding.update_calls, 1)
        self.assertEqual(simulation.loss_winding_solid_update_count, 2)
        self.assertEqual(simulation.loss_winding_mesh_operation_count, 2)
        self.assertEqual(simulation.loss_conductor_mesh_operation_count, 3)
        self.assertEqual(simulation.loss_plate_eddy_on_readback_count, 5)
        self.assertEqual(calls, [
            ("skin", {}),
            ("plates", {"enable_eddy_effects": True, "assign_skin_mesh": True}),
        ])

    def test_native_solid_copy_skips_stale_wrapper_edits_but_keeps_mesh(self):
        simulation, calls = self._simulation(0)
        simulation.tx_winding.update_result = False
        simulation.rx_winding.update_result = False

        assigned = _configure_loss_copy_skin_mesh(
            simulation, native_windings_solid=True
        )

        self.assertTrue(assigned)
        self.assertEqual(simulation.tx_winding.update_calls, 0)
        self.assertEqual(simulation.rx_winding.update_calls, 0)
        self.assertEqual(calls, [
            ("skin", {}),
            ("plates", {"enable_eddy_effects": True, "assign_skin_mesh": True}),
        ])

    def test_winding_update_failure_stops_before_loss_mesh(self):
        simulation, calls = self._simulation(0)
        simulation.tx_winding.update_result = False

        with self.assertRaisesRegex(RuntimeError, "failed to set Tx winding"):
            _configure_loss_copy_skin_mesh(simulation)

        self.assertEqual(calls, [])

    def test_lightweight_matrix_has_no_skin_operations(self):
        simulation, calls = self._simulation(0)

        assigned = _configure_em_conductor_mesh(simulation, "matrix")

        self.assertFalse(assigned)
        self.assertEqual(simulation.matrix_conductor_policy, "stranded_no_eddy_no_skin")
        self.assertEqual(simulation.matrix_winding_stranded_count, 2)
        self.assertEqual(simulation.matrix_conductor_mesh_operation_count, 0)
        self.assertEqual(simulation.matrix_plate_eddy_off_readback_count, 5)
        self.assertEqual(calls, [
            ("plates", {"enable_eddy_effects": False, "assign_skin_mesh": False}),
        ])

    def test_matrix_windings_are_stranded_when_skin_is_disabled(self):
        calls = []

        class _Design:
            def assign_winding(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace()

        simulation = Simulation.__new__(Simulation)
        simulation.df_plus = pd.DataFrame({
            "matrix_skin_mesh": [0],
            "I1_rated": [1000.0],
            "I2_rated": [100.0],
        })
        simulation.design1 = _Design()

        simulation.assign_winding(mode="matrix")

        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call["is_solid"] is False for call in calls))

    def test_lightweight_matrix_disables_plate_eddy_without_mesh(self):
        design = SimpleNamespace(
            core_plates=[SimpleNamespace(name="core_plate")],
            wcp_plates=[],
        )
        simulation = Simulation.__new__(Simulation)
        simulation.design1 = design
        simulation._set_plate_eddy_effects_native = Mock(return_value=1)

        simulation.assign_plate_settings(
            enable_eddy_effects=False, assign_skin_mesh=False
        )

        simulation._set_plate_eddy_effects_native.assert_called_once_with(False)

    def test_plate_eddy_readback_mismatch_fails_closed(self):
        design = SimpleNamespace(
            core_plates=[SimpleNamespace(name="core_plate")],
            wcp_plates=[],
        )
        simulation = Simulation.__new__(Simulation)
        simulation.design1 = design
        simulation._set_plate_eddy_effects_native = Mock(
            side_effect=RuntimeError("native eddy-effect readback mismatch")
        )

        with self.assertRaisesRegex(RuntimeError, "readback mismatch"):
            simulation.assign_plate_settings(
                enable_eddy_effects=True, assign_skin_mesh=False
            )

    def test_plate_mesh_uses_fresh_name_selections_not_object_handles(self):
        assign_skin_depth = Mock(return_value=SimpleNamespace(name="plate_skin_depth"))
        design = SimpleNamespace(
            core_plates=[SimpleNamespace(name="core_plate")],
            wcp_plates=[SimpleNamespace(name="wcp_plate")],
            mesh=SimpleNamespace(assign_skin_depth=assign_skin_depth),
        )
        simulation = Simulation.__new__(Simulation)
        simulation.design1 = design
        simulation.df_plus = pd.DataFrame({"freq": [1000.0]})
        simulation._set_plate_eddy_effects_native = Mock(return_value=2)

        count = simulation.assign_plate_settings(
            enable_eddy_effects=True, assign_skin_mesh=True
        )

        self.assertEqual(count, 2)
        self.assertEqual(
            assign_skin_depth.call_args.kwargs["assignment"],
            ["core_plate", "wcp_plate"],
        )
        self.assertTrue(all(
            isinstance(name, str)
            for name in assign_skin_depth.call_args.kwargs["assignment"]
        ))


class NativeEddyTransactionTests(unittest.TestCase):
    class _Editor:
        def __init__(self, copper, aluminum, failure=None):
            self.copper = list(copper)
            self.aluminum = list(aluminum)
            self.failure = failure
            self.calls = []

        def GetObjectsByMaterial(self, material):
            self.calls.append(material)
            if self.failure is not None:
                failure, self.failure = self.failure, None
                raise failure
            if material == "copper_80C":
                return list(self.copper)
            if material == "aluminum":
                return list(self.aluminum)
            raise AssertionError(material)

    class _Boundary:
        def __init__(self, state, set_failure=None, read_failure=None, overrides=None):
            self.state = state
            self.set_failure = set_failure
            self.read_failure = read_failure
            self.overrides = overrides or {}
            self.set_calls = []

        def SetEddyEffect(self, payload):
            self.set_calls.append(payload)
            if self.set_failure is not None:
                failure, self.set_failure = self.set_failure, None
                raise failure
            self.assert_vector_shape(payload)
            for record in payload[1][1:]:
                name = record[2]
                self.state[name]["eddy"] = bool(record[4])
                self.state[name]["displacement"] = bool(record[6])

        @staticmethod
        def assert_vector_shape(payload):
            if payload[0] != "NAME:Eddy Effect Setting":
                raise AssertionError(payload)
            if payload[1][0] != "NAME:EddyEffectVector":
                raise AssertionError(payload)

        def _read(self, kind, name):
            if self.read_failure is not None:
                failure, self.read_failure = self.read_failure, None
                raise failure
            return self.overrides.get((kind, name), self.state[name][kind])

        def GetEddyEffect(self, name):
            return self._read("eddy", name)

        def GetDisplacementCurrent(self, name):
            return self._read("displacement", name)

    class _RawDesign:
        def __init__(self, name, editor, boundary, design_type="Maxwell 3D",
                     solution="AC Magnetic"):
            self.name = name
            self.editor = editor
            self.boundary = boundary
            self.design_type = design_type
            self.solution = solution
            self.editor_calls = []

        def GetName(self):
            return self.name

        def GetDesignType(self):
            return self.design_type

        def GetSolutionType(self):
            return self.solution

        def SetActiveEditor(self, name):
            self.editor_calls.append(name)
            if name != "3D Modeler":
                raise AssertionError(name)
            return self.editor

        def GetModule(self, name):
            if name != "BoundarySetup":
                raise AssertionError(name)
            return self.boundary

    class _Project:
        def __init__(self, design, name="simulation_native_eddy"):
            self.design = design
            self.name = name
            self.active_calls = []

        def GetName(self):
            return self.name

        def SetActiveDesign(self, name):
            self.active_calls.append(name)
            return self.design

    @staticmethod
    def _objects(prefix, count):
        return [SimpleNamespace(name=f"{prefix}_{index}") for index in range(count)]

    def _simulation(self, winding_count=4, plate_count=2):
        tx_count = winding_count // 2
        tx = self._objects("Tx", tx_count)
        rx = self._objects("Rx", winding_count - tx_count)
        core_count = plate_count // 2
        core = self._objects("core_plate", core_count)
        wcp = self._objects("wcp_plate", plate_count - core_count)
        analyze = Mock(side_effect=AssertionError("eddy recovery must not solve"))
        design = SimpleNamespace(
            design_name="maxwell_matrix1",
            Tx_windings=tx,
            Rx_windings=rx,
            core_plates=core,
            wcp_plates=wcp,
            setup=SimpleNamespace(analyze=analyze),
            eddy_effects_on=Mock(
                side_effect=AssertionError("high-level conductor discovery is forbidden")
            ),
            get_all_conductors_names=Mock(
                side_effect=AssertionError("high-level conductor discovery is forbidden")
            ),
            modeler=SimpleNamespace(
                refresh_all_ids=Mock(
                    side_effect=AssertionError("modeler refresh is forbidden")
                )
            ),
        )
        simulation = Simulation.__new__(Simulation)
        simulation.PROJECT_NAME = "simulation_native_eddy"
        simulation.winding_conductor_material = "copper_80C"
        simulation.design1 = design
        simulation.solve_attempts = {"matrix": 1, "loss": 0}
        names = [item.name for item in tx + rx + core + wcp]
        state = {
            name: {"eddy": False, "displacement": False} for name in names
        }
        return simulation, [item.name for item in tx + rx], [item.name for item in core + wcp], state

    def _stack(self, copper, aluminum, state, **boundary_kwargs):
        editor = self._Editor(copper, aluminum)
        boundary = self._Boundary(state, **boundary_kwargs)
        raw = self._RawDesign("maxwell_matrix1", editor, boundary)
        project = self._Project(raw)
        return project, raw, editor, boundary

    def test_writes_and_reads_exact_full_vectors_without_hardcoded_count(self):
        for enable_eddy_effects in (False, True):
            for winding_count in (99, 87):
                with self.subTest(
                        enable_eddy_effects=enable_eddy_effects,
                        winding_count=winding_count):
                    simulation, windings, plates, state = self._simulation(
                        winding_count=winding_count, plate_count=5
                    )
                    project, raw, editor, boundary = self._stack(
                        windings, plates, state
                    )
                    simulation._refresh_native_project_handle = Mock(return_value=project)

                    count = simulation._set_plate_eddy_effects_native(
                        enable_eddy_effects,
                        max_attempts=1,
                        sleeper=lambda _seconds: None,
                    )

                    self.assertEqual(count, 5)
                    self.assertEqual(editor.calls, ["copper_80C", "aluminum"])
                    self.assertEqual(raw.editor_calls, ["3D Modeler"])
                    self.assertEqual(project.active_calls, ["maxwell_matrix1"])
                    vector = boundary.set_calls[0][1]
                    self.assertEqual(len(vector) - 1, winding_count + 5)
                    records = {record[2]: record for record in vector[1:]}
                    self.assertEqual(set(records), set(windings + plates))
                    self.assertTrue(all(records[name][4] is False for name in windings))
                    self.assertTrue(all(
                        records[name][4] is enable_eddy_effects for name in plates
                    ))
                    self.assertTrue(all(
                        record[6] is False for record in records.values()
                    ))
                    self.assertTrue(all(
                        state[name]["eddy"] is False for name in windings
                    ))
                    self.assertTrue(all(
                        state[name]["eddy"] is enable_eddy_effects for name in plates
                    ))
                    self.assertEqual(
                        simulation.solve_attempts, {"matrix": 1, "loss": 0}
                    )
                    simulation.design1.setup.analyze.assert_not_called()
                    simulation.design1.eddy_effects_on.assert_not_called()
                    simulation.design1.get_all_conductors_names.assert_not_called()
                    simulation.design1.modeler.refresh_all_ids.assert_not_called()

    def test_transient_query_write_and_read_failures_use_fresh_handles(self):
        for stage in ("query", "write", "read"):
            with self.subTest(stage=stage):
                simulation, windings, plates, state = self._simulation()
                first_project, _raw1, editor1, boundary1 = self._stack(
                    windings, plates, state,
                    set_failure=(RuntimeError("transient SetEddyEffect") if stage == "write" else None),
                    read_failure=(RuntimeError("transient readback") if stage == "read" else None),
                )
                if stage == "query":
                    editor1.failure = RuntimeError("transient GetObjectsByMaterial")
                second_project, _raw2, _editor2, boundary2 = self._stack(
                    windings, plates, state
                )
                simulation._refresh_native_project_handle = Mock(
                    side_effect=[first_project, second_project]
                )
                sleeps = []

                count = simulation._set_plate_eddy_effects_native(
                    True, max_attempts=2, sleeper=sleeps.append
                )

                self.assertEqual(count, len(plates))
                self.assertEqual(simulation._refresh_native_project_handle.call_count, 2)
                self.assertEqual(sleeps, [0.5])
                self.assertEqual(len(boundary2.set_calls), 1)
                self.assertEqual(
                    len(boundary1.set_calls), 0 if stage == "query" else 1
                )
                simulation.design1.setup.analyze.assert_not_called()

    def test_material_universe_mismatch_never_writes(self):
        cases = {
            "missing": lambda windings, plates: (windings[:-1], plates),
            "extra": lambda windings, plates: (windings + ["unexpected"], plates),
            "duplicate": lambda windings, plates: (windings + [windings[0]], plates),
            "mis-material": lambda windings, plates: (
                windings[1:], plates + [windings[0]]
            ),
        }
        for label, mutate in cases.items():
            with self.subTest(case=label):
                simulation, windings, plates, state = self._simulation()
                copper, aluminum = mutate(list(windings), list(plates))
                project, _raw, _editor, boundary = self._stack(
                    copper, aluminum, state
                )
                simulation._refresh_native_project_handle = Mock(return_value=project)

                with self.assertRaisesRegex(RuntimeError, "failed closed"):
                    simulation._set_plate_eddy_effects_native(
                        True, max_attempts=1, sleeper=lambda _seconds: None
                    )

                self.assertEqual(boundary.set_calls, [])
                simulation.design1.setup.analyze.assert_not_called()

    def test_wrong_design_identity_is_immediate_and_never_writes(self):
        simulation, windings, plates, state = self._simulation()
        project, _raw, _editor, boundary = self._stack(windings, plates, state)
        project.design.name = "maxwell_matrix"
        simulation._refresh_native_project_handle = Mock(return_value=project)

        with self.assertRaisesRegex(RuntimeError, "design identity mismatch"):
            simulation._set_plate_eddy_effects_native(
                True, max_attempts=5, sleeper=lambda _seconds: None
            )

        simulation._refresh_native_project_handle.assert_called_once_with()
        self.assertEqual(boundary.set_calls, [])
        simulation.design1.setup.analyze.assert_not_called()

    def test_permanent_transport_failure_is_bounded(self):
        simulation, _windings, _plates, _state = self._simulation()
        simulation._refresh_native_project_handle = Mock(
            side_effect=RuntimeError("permanent transport failure")
        )
        sleeps = []

        with self.assertRaisesRegex(RuntimeError, "failed closed"):
            simulation._set_plate_eddy_effects_native(
                True, max_attempts=5, sleeper=sleeps.append
            )

        self.assertEqual(simulation._refresh_native_project_handle.call_count, 5)
        self.assertEqual(sleeps, [0.5, 1.0, 2.0, 4.0])
        simulation.design1.setup.analyze.assert_not_called()

    def test_deadline_prevents_another_native_transaction(self):
        class Clock:
            def __init__(self):
                self.value = 0.0

            def __call__(self):
                return self.value

            def sleep(self, seconds):
                self.value += seconds

        simulation, _windings, _plates, _state = self._simulation()
        simulation._refresh_native_project_handle = Mock(
            side_effect=RuntimeError("transport remains unavailable")
        )
        clock = Clock()

        with self.assertRaisesRegex(RuntimeError, "failed closed"):
            simulation._set_plate_eddy_effects_native(
                True, max_attempts=5, timeout_s=1.0,
                initial_retry_delay=1.0, clock=clock, sleeper=clock.sleep,
            )

        simulation._refresh_native_project_handle.assert_called_once_with()
        self.assertEqual(clock.value, 1.0)
        simulation.design1.setup.analyze.assert_not_called()

    def test_any_full_vector_readback_mismatch_fails_closed(self):
        for kind, target in (
                ("eddy", "winding"),
                ("eddy", "plate"),
                ("displacement", "winding")):
            with self.subTest(kind=kind, target=target):
                simulation, windings, plates, state = self._simulation()
                name = windings[0] if target == "winding" else plates[0]
                expected = False if target == "winding" else True
                override = not expected if kind == "eddy" else True
                project, _raw, _editor, boundary = self._stack(
                    windings, plates, state,
                    overrides={(kind, name): override},
                )
                simulation._refresh_native_project_handle = Mock(return_value=project)

                with self.assertRaisesRegex(RuntimeError, "readback mismatch"):
                    simulation._set_plate_eddy_effects_native(
                        True, max_attempts=1, sleeper=lambda _seconds: None
                    )

                self.assertEqual(len(boundary.set_calls), 1)
                simulation.design1.setup.analyze.assert_not_called()


class CopiedLossObjectIntegrityTests(unittest.TestCase):
    @staticmethod
    def _object(name):
        return SimpleNamespace(name=name)

    def test_remap_preserves_every_object_name_and_order(self):
        old = SimpleNamespace(
            Tx_windings=[self._object("Tx_0"), self._object("Tx_1")],
            empty=[],
        )

        def find_object(source):
            return [self._object(item.name) for item in source]

        new = SimpleNamespace(model3d=SimpleNamespace(find_object=find_object))

        _remap_copied_design_objects(old, new, ("Tx_windings", "empty"))

        self.assertEqual([item.name for item in new.Tx_windings], ["Tx_0", "Tx_1"])
        self.assertEqual(new.empty, [])

    def test_remap_rejects_one_missing_turn(self):
        old = SimpleNamespace(
            Rx_windings=[self._object("Rx_0"), self._object("Rx_1")]
        )
        new = SimpleNamespace(model3d=SimpleNamespace(
            find_object=lambda _source: [self._object("Rx_0")]
        ))

        with self.assertRaisesRegex(RuntimeError, "remap mismatch for Rx_windings"):
            _remap_copied_design_objects(old, new, ("Rx_windings",))

    @staticmethod
    def _native_design(name, delete_effect=None):
        analysis = SimpleNamespace(GetSetups=lambda: ["Setup1"])
        design = SimpleNamespace(
            GetName=lambda: name,
            GetDesignType=lambda: "Maxwell 3D",
            GetSolutionType=lambda: "AC Magnetic",
            GetModule=lambda module: analysis if module == "AnalysisSetup" else None,
            DeleteFullVariation=Mock(
                side_effect=delete_effect) if delete_effect else Mock(return_value=None),
        )
        return design

    def test_solution_delete_uses_fallback_after_primary_failure(self):
        primary = self._native_design("copy", RuntimeError("grpc"))
        fallback = self._native_design("copy")

        backend = _delete_copied_solution_or_raise(primary, fallback, "copy")

        self.assertEqual(backend, "native")
        fallback.DeleteFullVariation.assert_called_once_with("All", False)

    def test_solution_delete_fails_when_both_paths_fail(self):
        primary = self._native_design("copy", RuntimeError("grpc"))
        fallback = self._native_design("copy", RuntimeError("com"))

        with self.assertRaisesRegex(RuntimeError, "copied solution deletion failed"):
            _delete_copied_solution_or_raise(primary, fallback, "copy")

    def test_solution_delete_never_touches_stale_source_binding(self):
        source = self._native_design("maxwell_matrix")
        copied = self._native_design("maxwell_matrix1")

        backend = _delete_copied_solution_or_raise(
            source, copied, "maxwell_matrix1"
        )

        self.assertEqual(backend, "native")
        source.DeleteFullVariation.assert_not_called()
        copied.DeleteFullVariation.assert_called_once_with("All", False)


class _NativeTree:
    def __init__(self, children=None, properties=None):
        self.children = children or {}
        self.properties = properties or {}

    def GetChildNames(self):
        return list(self.children)

    def GetChildObject(self, name):
        return self.children[name]

    def GetPropNames(self):
        return list(self.properties)

    def GetPropValue(self, name):
        return self.properties[name]


class _NativeLossDesign:
    def __init__(self, name, matrix=False, solved=True):
        self.name = name
        self.matrix = matrix
        self.solved = solved
        self.deleted_solution = Mock()
        self.windings = {
            "Tx_winding": _NativeTree(properties={
                "Type": "Winding Group", "Winding Type": "Current",
                "Current": "1414A", "Phase": "0deg", "IsSolid": "Stranded",
            }),
            "Rx_winding": _NativeTree(properties={
                "Type": "Winding Group", "Winding Type": "Current",
                "Current": "141.4A", "Phase": "0deg", "IsSolid": "Stranded",
            }),
        }
        self.core_loss = {}
        self.native_winding_edit_noop = False
        self.native_core_loss_noop = False
        self.edit_winding_calls = []
        self.set_core_loss_calls = []

    def _edit_winding_group(self, name, arguments):
        self.edit_winding_calls.append((name, list(arguments)))
        if self.native_winding_edit_noop:
            return
        if arguments[0] != f"NAME:{name}" or len(arguments[1:]) % 2:
            raise AssertionError(arguments)
        updates = {
            str(arguments[index]).removesuffix(":="): arguments[index + 1]
            for index in range(1, len(arguments), 2)
        }
        if "Type" in updates:
            updates["Winding Type"] = updates.pop("Type")
        self.windings[name].properties.update(updates)

    def _set_core_loss(self, names, on_field):
        self.set_core_loss_calls.append((list(names), on_field))
        if self.native_core_loss_noop:
            return
        for name in names:
            self.core_loss[name] = True

    def GetName(self):
        return self.name

    def GetDesignType(self):
        return "Maxwell 3D"

    def GetSolutionType(self):
        return "AC Magnetic"

    def GetChildObject(self, name):
        if name == "Parameters":
            return _NativeTree(children={"Matrix": _NativeTree()} if self.matrix else {})
        if name == "Boundaries":
            return _NativeTree(children=self.windings)
        if name == "Excitations":
            return _NativeTree(children={})
        raise KeyError(name)

    def GetModule(self, name):
        if name == "AnalysisSetup":
            return SimpleNamespace(GetSetups=lambda: ["Setup1"])
        if name == "Solutions":
            return SimpleNamespace(
                GetAvailableVariations=lambda _setup: ["Freq='1000Hz'"] if self.solved else []
            )
        if name == "BoundarySetup":
            return SimpleNamespace(
                GetCoreLossEffect=lambda object_name: self.core_loss.get(object_name, False),
                EditWindingGroup=self._edit_winding_group,
                SetCoreLoss=self._set_core_loss,
            )
        raise KeyError(name)

    def DeleteFullVariation(self, *_args):
        self.deleted_solution(*_args)
        self.solved = False


class _NativeLossProject:
    def __init__(self, source):
        self.designs = [source]
        self.deleted = []
        self.active = source.name

    def GetDesigns(self):
        return list(self.designs)

    def SetActiveDesign(self, name):
        matches = [item for item in self.designs if item.name == name]
        if len(matches) != 1:
            raise RuntimeError(name)
        self.active = name
        return matches[0]

    def DeleteDesign(self, name):
        matches = [item for item in self.designs if item.name == name]
        if len(matches) != 1:
            raise RuntimeError(name)
        self.designs.remove(matches[0])
        self.deleted.append(name)


class _WindingProps(dict):
    def _setitem_without_update(self, key, value):
        dict.__setitem__(self, key, value)


class _CopiedWinding:
    def __init__(self, name, native_child, update_result=True, write_native=True):
        self.name = name
        self.native_child = native_child
        self.update_result = update_result
        self.write_native = write_native
        self.auto_update = True
        self.props = _WindingProps(native_child.properties.copy())
        self.update_calls = 0

    def update(self):
        self.update_calls += 1
        if self.update_result and self.write_native:
            self.native_child.properties.update(self.props)
        return self.update_result


def _prepared_wrapper(raw, windings=None):
    setup = SimpleNamespace(
        _child_object=object(),
        properties={
            "Max. Number of Passes": 10,
            "Min. Converged Passes": 2,
            "Percent Error": 1.5,
        },
    )
    return SimpleNamespace(
        design_name=raw.name,
        solution_type="AC Magnetic",
        solver_instance=SimpleNamespace(odesign=raw),
        get_setup=lambda name: setup if name == "Setup1" else False,
        get_excitation=lambda excitation_name: windings,
    )


class CopiedLossPreparationRetryTests(unittest.TestCase):
    def test_matrix_solution_probe_stops_after_canonical_name_succeeds(self):
        source = _NativeLossDesign("maxwell_matrix", matrix=True, solved=True)
        project = _NativeLossProject(source)
        original_get_module = source.GetModule
        calls = []

        def get_module(name):
            if name != "Solutions":
                return original_get_module(name)

            def get_available_variations(sweep_name):
                calls.append(sweep_name)
                if sweep_name != "Setup1 : LastAdaptive":
                    raise AssertionError("legacy fallback must not be probed")
                return ["Freq='1000Hz'"]

            return SimpleNamespace(
                GetAvailableVariations=get_available_variations
            )

        source.GetModule = get_module

        signature = _matrix_source_signature(project, "maxwell_matrix")

        self.assertEqual(calls, ["Setup1 : LastAdaptive"])
        self.assertEqual(signature["solution_marker"], ("Freq='1000Hz'",))

    def test_matrix_solution_probe_uses_legacy_name_only_after_exception(self):
        source = _NativeLossDesign("maxwell_matrix", matrix=True, solved=True)
        project = _NativeLossProject(source)
        original_get_module = source.GetModule
        calls = []

        def get_module(name):
            if name != "Solutions":
                return original_get_module(name)

            def get_available_variations(sweep_name):
                calls.append(sweep_name)
                if sweep_name == "Setup1 : LastAdaptive":
                    raise RuntimeError("canonical spelling unavailable")
                return ["Freq='1000Hz'"]

            return SimpleNamespace(
                GetAvailableVariations=get_available_variations
            )

        source.GetModule = get_module

        signature = _matrix_source_signature(project, "maxwell_matrix")

        self.assertEqual(
            calls,
            ["Setup1 : LastAdaptive", "Setup1 : Last Adaptive"],
        )
        self.assertEqual(signature["solution_marker"], ("Freq='1000Hz'",))

    def test_native_winding_prefers_boundaries_when_alias_is_in_both_roots(self):
        boundary_child = _NativeTree(properties={"Current": "1A"})
        excitation_alias = _NativeTree(properties={"Current": "stale"})

        class _AliasedDesign:
            def GetChildObject(self, name):
                if name == "Boundaries":
                    return _NativeTree(children={"Tx_winding": boundary_child})
                if name == "Excitations":
                    return _NativeTree(children={"Tx_winding": excitation_alias})
                raise KeyError(name)

        selected = _native_winding_child(_AliasedDesign(), "Tx_winding")

        self.assertIs(selected, boundary_child)

    def test_none_winding_name_is_rejected_before_update(self):
        winding = _CopiedWinding(None, _NativeTree(properties={
            "Current": "1A", "Phase": "0deg",
        }))

        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            _set_copied_loss_winding_excitation(
                winding, "Tx_winding", "2A", "1deg", "Tx"
            )

        self.assertEqual(winding.update_calls, 0)

    def test_false_winding_update_fails_closed(self):
        winding = _CopiedWinding(
            "Tx_winding",
            _NativeTree(properties={"Current": "1A", "Phase": "0deg"}),
            update_result=False,
        )

        with self.assertRaisesRegex(RuntimeError, "returned False"):
            _set_copied_loss_winding_excitation(
                winding, "Tx_winding", "2A", "1deg", "Tx"
            )

    def test_native_readback_rejects_native_edit_without_effect(self):
        raw = _NativeLossDesign("copy")
        raw.native_winding_edit_noop = True
        wrapper = _prepared_wrapper(raw, [])

        with self.assertRaisesRegex(RuntimeError, "native value mismatch"):
            _configure_copied_loss_excitations(
                wrapper, raw, "copy", "1420A", "-8deg", "141.4A", "-4deg"
            )

    def test_native_readback_accepts_all_four_updated_values(self):
        raw = _NativeLossDesign("copy")
        wrapper = _prepared_wrapper(raw, [])

        result = _configure_copied_loss_excitations(
            wrapper, raw, "copy", "1420A", "-8deg", "141.4A", "-4deg"
        )

        self.assertEqual(
            result,
            (raw.windings["Tx_winding"], raw.windings["Rx_winding"]),
        )
        self.assertEqual(len(raw.edit_winding_calls), 2)
        self.assertEqual(
            raw.windings["Tx_winding"].properties["Type"], "Winding Group"
        )
        self.assertEqual(
            raw.windings["Tx_winding"].properties["Winding Type"], "Current"
        )
        for name, arguments in raw.edit_winding_calls:
            self.assertEqual(arguments[0], f"NAME:{name}")
            values = dict(zip(arguments[1::2], arguments[2::2]))
            self.assertEqual(values["Type:="], "Current")
            self.assertIs(values["IsSolid:="], True)
            self.assertEqual(values["Resistance:="], "0ohm")
            self.assertEqual(values["Inductance:="], "0H")
            self.assertEqual(values["Voltage:="], "0V")
            self.assertEqual(values["ParallelBranchesNum:="], "1")

    def test_exact_native_winding_identity_is_checked_before_edit(self):
        raw = _NativeLossDesign("source")

        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            _edit_native_copied_loss_winding(
                raw, "copy", "Tx_winding", "1420A", "-8deg"
            )

        self.assertEqual(raw.edit_winding_calls, [])

    def test_native_winding_type_is_distinct_from_object_kind(self):
        raw = _NativeLossDesign("copy")
        raw.windings["Tx_winding"].properties["Winding Type"] = "Voltage"

        with self.assertRaisesRegex(RuntimeError, "non-current native winding"):
            _edit_native_copied_loss_winding(
                raw, "copy", "Tx_winding", "1420A", "-8deg"
            )

        self.assertEqual(raw.edit_winding_calls, [])

    def test_native_core_assignment_uses_complete_exact_name_list(self):
        raw = _NativeLossDesign("copy")

        count = _assign_native_copied_core_loss(
            raw, "copy", ["core_1", "core_2"]
        )

        self.assertEqual(count, 2)
        self.assertEqual(
            raw.set_core_loss_calls,
            [(["core_1", "core_2"], False)],
        )
        self.assertTrue(raw.core_loss["core_1"])
        self.assertTrue(raw.core_loss["core_2"])

    def test_native_core_assignment_rejects_false_readback(self):
        raw = _NativeLossDesign("copy")
        raw.native_core_loss_noop = True

        with self.assertRaisesRegex(RuntimeError, "core loss disabled"):
            _assign_native_copied_core_loss(raw, "copy", ["core_1"])

    def test_fresh_copy_retry_deletes_only_failed_copy(self):
        source = _NativeLossDesign("maxwell_matrix", matrix=True, solved=True)
        project = _NativeLossProject(source)
        calls = []

        def prepare(_before, attempt):
            raw = _NativeLossDesign(f"copy{attempt}")
            project.designs.append(raw)
            calls.append(attempt)
            if attempt == 1:
                raise RuntimeError("DataModel not ready")
            return _prepared_wrapper(raw)

        prepared, attempts = _retry_copied_loss_preparation(
            project, "maxwell_matrix", prepare,
            max_attempts=3, retry_delay_s=0, sleeper=lambda _seconds: None,
        )

        self.assertEqual(attempts, 2)
        self.assertEqual(prepared.design_name, "copy2")
        self.assertEqual(project.deleted, ["copy1"])
        self.assertEqual([item.name for item in project.designs], ["maxwell_matrix", "copy2"])
        self.assertEqual(calls, [1, 2])

    def test_model_only_accepts_unsolved_source_without_dispatching_or_deleting(self):
        source = _NativeLossDesign("maxwell_matrix", matrix=True, solved=False)
        project = _NativeLossProject(source)
        calls = []

        def prepare(_before, attempt):
            raw = _NativeLossDesign(f"copy{attempt}", solved=False)
            project.designs.append(raw)
            calls.append(attempt)
            return _prepared_wrapper(raw)

        prepared, attempts = _retry_copied_loss_preparation(
            project, "maxwell_matrix", prepare,
            max_attempts=3, retry_delay_s=0, sleeper=lambda _seconds: None,
            require_source_solved=False,
        )

        self.assertEqual(prepared.design_name, "copy1")
        self.assertEqual(attempts, 1)
        self.assertEqual(calls, [1])
        self.assertFalse(source.solved)
        self.assertEqual(project.deleted, [])

    def test_source_winding_mutation_aborts_before_a_second_copy(self):
        source = _NativeLossDesign("maxwell_matrix", matrix=True, solved=True)
        project = _NativeLossProject(source)
        calls = []

        def prepare(_before, attempt):
            project.designs.append(_NativeLossDesign(f"copy{attempt}"))
            calls.append(attempt)
            source.windings["Tx_winding"].properties["Current"] = "999A"
            raise RuntimeError("stale wrapper mutated source")

        with self.assertRaisesRegex(RuntimeError, "matrix source changed"):
            _retry_copied_loss_preparation(
                project, "maxwell_matrix", prepare,
                max_attempts=3, retry_delay_s=0, sleeper=lambda _seconds: None,
            )

        self.assertEqual(calls, [1])
        self.assertEqual(project.deleted, ["copy1"])
        self.assertEqual(
            [item.name for item in project.designs], ["maxwell_matrix"]
        )

    def test_source_winding_type_mutation_aborts_before_a_second_copy(self):
        source = _NativeLossDesign("maxwell_matrix", matrix=True, solved=True)
        project = _NativeLossProject(source)
        calls = []

        def prepare(_before, attempt):
            project.designs.append(_NativeLossDesign(f"copy{attempt}"))
            calls.append(attempt)
            source.windings["Tx_winding"].properties["Winding Type"] = "Voltage"
            raise RuntimeError("source physical winding mode changed")

        with self.assertRaisesRegex(RuntimeError, "matrix source changed"):
            _retry_copied_loss_preparation(
                project, "maxwell_matrix", prepare,
                max_attempts=3, retry_delay_s=0, sleeper=lambda _seconds: None,
            )

        self.assertEqual(calls, [1])
        self.assertEqual(project.deleted, ["copy1"])
        self.assertEqual(
            [item.name for item in project.designs], ["maxwell_matrix"]
        )

    def test_deterministic_prepare_error_is_bounded_and_cleans_every_copy(self):
        source = _NativeLossDesign("maxwell_matrix", matrix=True, solved=True)
        project = _NativeLossProject(source)

        def prepare(_before, attempt):
            project.designs.append(_NativeLossDesign(f"copy{attempt}"))
            raise RuntimeError("deterministic")

        with self.assertRaisesRegex(RuntimeError, "failed after 3 fresh copies"):
            _retry_copied_loss_preparation(
                project, "maxwell_matrix", prepare,
                max_attempts=3, retry_delay_s=0, sleeper=lambda _seconds: None,
            )

        self.assertEqual(project.deleted, ["copy1", "copy2", "copy3"])
        self.assertEqual([item.name for item in project.designs], ["maxwell_matrix"])

    def test_cleanup_rejects_delayed_second_copy_outside_baseline(self):
        source = _NativeLossDesign("maxwell_matrix", matrix=True, solved=True)
        project = _NativeLossProject(source)
        original_delete = project.DeleteDesign

        def delete_with_delayed_copy(name):
            original_delete(name)
            project.designs.append(_NativeLossDesign("late_copy"))

        project.DeleteDesign = delete_with_delayed_copy

        def prepare(_before, _attempt):
            project.designs.append(_NativeLossDesign("copy1"))
            raise RuntimeError("DataModel not ready")

        with self.assertRaisesRegex(RuntimeError, "exact baseline"):
            _retry_copied_loss_preparation(
                project, "maxwell_matrix", prepare,
                max_attempts=3, retry_delay_s=0, sleeper=lambda _seconds: None,
            )

        self.assertEqual(project.deleted, ["copy1"])
        self.assertEqual(
            [item.name for item in project.designs],
            ["maxwell_matrix", "late_copy"],
        )

    def test_core_loss_false_readback_fails_closed(self):
        raw = _NativeLossDesign("copy")
        raw.core_loss["core_1"] = False

        with self.assertRaisesRegex(RuntimeError, "core loss disabled"):
            _assert_native_core_loss_assignment(raw, ["core_1"])

    def test_core_loss_readback_exception_fails_closed(self):
        def fail_readback(_object_name):
            raise RuntimeError("grpc unavailable")

        raw = SimpleNamespace(
            GetModule=lambda _name: SimpleNamespace(
                GetCoreLossEffect=fail_readback
            )
        )

        with self.assertRaisesRegex(RuntimeError, "readback failed"):
            _assert_native_core_loss_assignment(raw, ["core_1"])


class CopiedLossSavedSnapshotTests(unittest.TestCase):
    @staticmethod
    def _snapshot(copied_solved=False):
        winding = lambda current, phase: {
            "Type": "Current", "IsSolid": True,
            "Current": current, "Phase": phase,
        }
        return {
            "AnsoftProject": {
                "Maxwell3DModel": [
                    {
                        "Name": "maxwell_matrix",
                        "MaxwellParameterSetup": {
                            "MaxwellParameters": {"Matrix": {"ID": 0}}
                        },
                    },
                    {
                        "Name": "copy",
                        "MaxwellParameterSetup": {"MaxwellParameters": {}},
                        "BoundarySetup": {
                            "Boundaries": {
                                "Tx_winding": winding("1420A", "-8deg"),
                                "Rx_winding": winding("141.4A", "-4deg"),
                            },
                            "GlobalBoundData": {"CoreLossObjectIDs": [10, 11]},
                        },
                        "AnalysisSetup": {"SolveSetups": {"Setup1": {
                            "SetupType": "AC Magnetic", "Enabled": True,
                            "MaximumPasses": 10,
                            "MinimumConvergedPasses": 2,
                            "PercentError": 1.5,
                            "Frequency": "1000Hz",
                        }}},
                    },
                ],
            },
            "ProjectPreview": {"DesignInfo": [
                {"DesignName": "maxwell_matrix", "IsSolved": True},
                {"DesignName": "copy", "IsSolved": copied_solved},
            ]},
        }

    def _validate(self, snapshot, require_source_solved=True):
        with tempfile.NamedTemporaryFile(suffix=".aedt") as stream:
            with patch(
                    "ansys.aedt.core.internal.load_aedt_file.load_entire_aedt_file",
                    return_value=snapshot):
                return _validate_saved_copied_loss_preparation(
                    stream.name, "maxwell_matrix", "copy",
                    "1420A", "-8deg", "141.4A", "-4deg", 2,
                    10, 2, 1.5, 1000,
                    require_source_solved=require_source_solved,
                )

    def test_accepts_authoritative_unsolved_ready_copy(self):
        evidence = self._validate(self._snapshot())

        self.assertTrue(evidence["source_solved"])
        self.assertFalse(evidence["copied_solved"])
        self.assertEqual(evidence["core_loss_ids"], 2)

    def test_rejects_copy_that_still_owns_inherited_solution(self):
        with self.assertRaisesRegex(RuntimeError, "still owns inherited solution"):
            self._validate(self._snapshot(copied_solved=True))

    def test_model_only_accepts_unsolved_source_and_unsolved_copy(self):
        snapshot = self._snapshot()
        snapshot["ProjectPreview"]["DesignInfo"][0]["IsSolved"] = False

        evidence = self._validate(snapshot, require_source_solved=False)

        self.assertFalse(evidence["source_solved"])
        self.assertFalse(evidence["copied_solved"])

    def test_rejects_saved_stranded_winding(self):
        snapshot = self._snapshot()
        snapshot["AnsoftProject"]["Maxwell3DModel"][1][
            "BoundarySetup"
        ]["Boundaries"]["Tx_winding"]["IsSolid"] = False

        with self.assertRaisesRegex(RuntimeError, "Tx_winding is not solid"):
            self._validate(snapshot)

    def test_rejects_saved_core_loss_count_mismatch(self):
        snapshot = self._snapshot()
        snapshot["AnsoftProject"]["Maxwell3DModel"][1][
            "BoundarySetup"
        ]["GlobalBoundData"]["CoreLossObjectIDs"] = [10]

        with self.assertRaisesRegex(RuntimeError, "core-loss ID count mismatch"):
            self._validate(snapshot)

    def test_rejects_saved_source_without_matrix_parameter(self):
        snapshot = self._snapshot()
        del snapshot["AnsoftProject"]["Maxwell3DModel"][0][
            "MaxwellParameterSetup"
        ]["MaxwellParameters"]["Matrix"]

        with self.assertRaisesRegex(RuntimeError, "source lost its Matrix"):
            self._validate(snapshot)

    def test_rejects_saved_setup_mismatch(self):
        snapshot = self._snapshot()
        snapshot["AnsoftProject"]["Maxwell3DModel"][1][
            "AnalysisSetup"
        ]["SolveSetups"]["Setup1"]["MaximumPasses"] = 11

        with self.assertRaisesRegex(RuntimeError, "MaximumPasses mismatch"):
            self._validate(snapshot)


class _FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class _FakeRawDesign:
    def __init__(self, name, design_type="Maxwell 3D", solution="AC Magnetic", setups=("Setup1",)):
        self.name = name
        self.design_type = design_type
        self.solution = solution
        self.setups = setups

    def GetName(self):
        return self.name

    def GetDesignType(self):
        return self.design_type

    def GetSolutionType(self):
        return self.solution

    def GetModule(self, name):
        if name != "AnalysisSetup":
            raise AssertionError(name)
        return SimpleNamespace(GetSetups=lambda: self.setups)


class _FakeProject:
    def __init__(self, design_sequences):
        self.design_sequences = list(design_sequences)
        self.index = 0
        self.active_calls = []
        self.last_designs = []

    def GetDesigns(self):
        if self.index < len(self.design_sequences):
            self.last_designs = self.design_sequences[self.index]
            self.index += 1
        return self.last_designs

    def SetActiveDesign(self, name):
        self.active_calls.append(name)
        return next(design for design in self.last_designs if design.GetName() == name)


def _fake_wrapper(solution="AC Magnetic", setup=True):
    setup_object = SimpleNamespace(
        _child_object=object(),
        properties={
            "Max. Number of Passes": 10,
            "Min. Converged Passes": 1,
            "Percent Error": 1.5,
        },
    ) if setup is True else setup
    return SimpleNamespace(
        design_name="copy",
        solution_type=solution,
        get_setup=lambda name: setup_object if name == "Setup1" else False,
    )


class CopiedLossReadinessTests(unittest.TestCase):
    def test_waits_through_wrong_solution_and_missing_setup(self):
        source = _FakeRawDesign("source")
        project = _FakeProject([
            [source],
            [source, _FakeRawDesign("copy", solution="Magnetostatic")],
            [source, _FakeRawDesign("copy", setups=())],
            [source, _FakeRawDesign("copy")],
            [source, _FakeRawDesign("copy")],
        ])
        clock = _FakeClock()
        factory_calls = []

        wrapper, setup = _wait_for_ready_copied_loss_design(
            project, {"source"},
            lambda name, solution: factory_calls.append((name, solution)) or _fake_wrapper(),
            timeout_s=2, poll_s=0.25, clock=clock, sleeper=clock.sleep,
        )

        self.assertEqual(wrapper.design_name, "copy")
        self.assertIs(setup, wrapper.get_setup("Setup1"))
        self.assertEqual(factory_calls, [("copy", "AC Magnetic")])
        self.assertEqual(project.active_calls, ["copy"])

    def test_retries_transient_wrapper_state(self):
        source = _FakeRawDesign("source")
        ready = _FakeRawDesign("copy")
        project = _FakeProject([[source, ready]] * 5)
        wrappers = iter([
            _fake_wrapper(solution="Magnetostatic"),
            _fake_wrapper(setup=False),
            _fake_wrapper(setup=SimpleNamespace(_child_object=None, properties={})),
            _fake_wrapper(),
        ])
        clock = _FakeClock()

        wrapper, setup = _wait_for_ready_copied_loss_design(
            project, {"source"}, lambda _name, _solution: next(wrappers),
            timeout_s=2, poll_s=0.25, clock=clock, sleeper=clock.sleep,
        )

        self.assertEqual(wrapper.solution_type, "AC Magnetic")
        self.assertTrue(hasattr(setup, "properties"))
        self.assertEqual(project.active_calls, ["copy", "copy", "copy", "copy"])

    def test_accepts_eddy_current_solution_alias(self):
        source = _FakeRawDesign("source")
        ready = _FakeRawDesign("copy", solution="EddyCurrent")
        project = _FakeProject([[source, ready]] * 2)
        clock = _FakeClock()

        wrapper, setup = _wait_for_ready_copied_loss_design(
            project, {"source"},
            lambda _name, _solution: _fake_wrapper(solution="EddyCurrent"),
            timeout_s=1, poll_s=0.25, clock=clock, sleeper=clock.sleep,
        )

        self.assertEqual(wrapper.solution_type, "EddyCurrent")
        self.assertTrue(setup.properties)

    def test_timeout_is_fail_closed_and_never_binds_source(self):
        source = _FakeRawDesign("source")
        project = _FakeProject([[source]] * 6)
        clock = _FakeClock()
        factory_calls = []

        with self.assertRaisesRegex(RuntimeError, "new_names.*\[\]"):
            _wait_for_ready_copied_loss_design(
                project, {"source"},
                lambda *_args: factory_calls.append(True),
                timeout_s=1, poll_s=0.25, clock=clock, sleeper=clock.sleep,
            )

        self.assertEqual(factory_calls, [])
        self.assertEqual(project.active_calls, [])


class CoreLossAssignmentTests(unittest.TestCase):
    @staticmethod
    def _simulation(result):
        simulation = Simulation.__new__(Simulation)
        simulation.design1 = SimpleNamespace(
            core_objs=[SimpleNamespace(name="core_1")],
            set_core_losses=lambda **_kwargs: result,
        )
        return simulation

    def test_false_core_loss_assignment_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "set_core_losses returned False"):
            self._simulation(False).assign_core_loss()

    def test_true_core_loss_assignment_continues(self):
        self.assertIsNone(self._simulation(True).assign_core_loss())


class CopiedLossSetupConfigurationTests(unittest.TestCase):
    def test_updates_and_reads_back_required_properties(self):
        setup = _fake_wrapper().get_setup("Setup1")

        configured = _configure_copied_loss_setup(setup, 12, 3, 0.75)

        self.assertIs(configured, setup)
        self.assertEqual(setup.properties["Max. Number of Passes"], 12)
        self.assertEqual(setup.properties["Min. Converged Passes"], 3)
        self.assertEqual(setup.properties["Percent Error"], 0.75)

    def test_rejects_property_updates_that_do_not_reach_com(self):
        class DiscardingSetup:
            _child_object = object()

            @property
            def properties(self):
                return {
                    "Max. Number of Passes": 10,
                    "Min. Converged Passes": 1,
                    "Percent Error": 1.5,
                }

        with self.assertRaisesRegex(RuntimeError, "read-back failed"):
            _configure_copied_loss_setup(DiscardingSetup(), 12, 3, 0.75)


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
        self.assertGreaterEqual(
            simulation.stage_timings["stage_time_matrix_solve_s"], 0
        )
        self.assertGreaterEqual(
            simulation.stage_timings["stage_time_matrix_extract_s"], 0
        )
        self.assertGreaterEqual(
            simulation.stage_timings["stage_time_matrix_analyze_total_s"],
            simulation.stage_timings["stage_time_matrix_solve_s"],
        )

    def test_pooled_analyze_uses_exact_native_project_scoped_dispatch(self):
        simulation = self._simulation([None])
        simulation.aedt_backend = "pooled"
        simulation.solver_may_be_running = False
        simulation.design1.setup.analyze = Mock(return_value=None)
        simulation._analyze_exact_pooled_design = Mock(return_value=0.0)
        simulation.aedt_automation_transaction = lambda: nullcontext()

        simulation.analyze_and_extract("matrix", lambda: None)

        simulation._analyze_exact_pooled_design.assert_called_once_with("matrix")
        simulation.design1.setup.analyze.assert_not_called()
        self.assertFalse(simulation.solver_may_be_running)

    def test_standalone_analyze_keeps_explicit_core_contract(self):
        simulation = self._simulation([None])
        simulation.design1.setup.analyze = Mock(return_value=None)

        simulation.analyze_and_extract("matrix", lambda: None)

        simulation.design1.setup.analyze.assert_called_once_with(cores=4)

    def test_cap_stage_records_solve_extraction_and_total_time(self):
        simulation = self._simulation([None])

        simulation.analyze_and_extract("cap", lambda: None)

        self.assertEqual(simulation.solve_attempts["cap"], 1)
        self.assertGreaterEqual(
            simulation.stage_timings["stage_time_cap_solve_s"], 0
        )
        self.assertGreaterEqual(
            simulation.stage_timings["stage_time_cap_extract_s"], 0
        )
        self.assertGreaterEqual(
            simulation.stage_timings["stage_time_cap_analyze_total_s"],
            simulation.stage_timings["stage_time_cap_solve_s"],
        )

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


class CopiedLossNativeAnalyzeTests(unittest.TestCase):
    class _AnalysisModule:
        def __init__(self, setups=("Setup1",)):
            self.setups = setups

        def GetSetups(self):
            return self.setups

    class _Design:
        def __init__(self, name="maxwell_matrix1", analyze_result=0,
                     design_type="Maxwell 3D", solution="AC Magnetic",
                     setups=("Setup1",)):
            self.name = name
            self.design_type = design_type
            self.solution = solution
            self.analysis = CopiedLossNativeAnalyzeTests._AnalysisModule(setups)
            self.Analyze = Mock(return_value=analyze_result)

        def GetName(self):
            return self.name

        def GetDesignType(self):
            return self.design_type

        def GetSolutionType(self):
            return self.solution

        def GetModule(self, name):
            if name != "AnalysisSetup":
                raise AssertionError(name)
            return self.analysis

    class _Project:
        def __init__(self, design, name="simulation_native_analyze"):
            self.design = design
            self.name = name
            self.active_calls = []
            self.Save = Mock(return_value=None)

        def GetName(self):
            return self.name

        def SetActiveDesign(self, name):
            self.active_calls.append(name)
            return self.design

    class _Desktop:
        def __init__(self, project, registry_failures=0):
            self.project = project
            self.registry_failures = registry_failures
            self.active_config = "Local"
            self.project_calls = []
            self.registry_loads = []
            self.registry_sets = []
            self.running = False

        def SetActiveProject(self, name):
            self.project_calls.append(name)
            return self.project

        def AreThereSimulationsRunning(self):
            return self.running

        def GetRegistryString(self, _key):
            if self.registry_failures:
                self.registry_failures -= 1
                raise RuntimeError("transient GetRegistryString")
            return self.active_config

        def SetRegistryFromFile(self, path):
            self.registry_loads.append(path)

        def SetRegistryString(self, key, value):
            self.registry_sets.append((key, value))
            self.active_config = value

    @staticmethod
    def _acf_text(cores=4, tasks=1):
        return f"""$begin 'DSOConfig'
ConfigName='pyaedt_config'
DesignType='Maxwell 3D'
MachineName='localhost'
NumEngines={tasks}
NumCores={cores}
NumGPUs=0
UseAutoSettings=True
$end 'DSOConfig'
"""

    def _simulation(self, root, registry_failures=0, analyze_result=0,
                    design_name="maxwell_matrix1"):
        matrix_working = Path(root) / "matrix_working"
        matrix_working.mkdir()
        (matrix_working / "pyaedt_config.acf").write_text(
            self._acf_text(), encoding="utf-8"
        )
        raw_design = self._Design(
            name=design_name, analyze_result=analyze_result
        )
        project = self._Project(raw_design)
        desktop = self._Desktop(project, registry_failures=registry_failures)
        high_level_analyze = Mock(
            side_effect=AssertionError("copied loss must not use Setup.analyze")
        )
        simulation = Simulation.__new__(Simulation)
        simulation.PROJECT_NAME = "simulation_native_analyze"
        simulation.project_path = str(root)
        simulation.desktop = SimpleNamespace(odesktop=desktop)
        simulation.project = SimpleNamespace(
            project=project,
            proj=project,
            desktop=simulation.desktop,
        )
        simulation.design1 = SimpleNamespace(
            design_name="maxwell_matrix1",
            setup=SimpleNamespace(analyze=high_level_analyze),
            save_project=Mock(return_value=None),
        )
        simulation.design_matrix = SimpleNamespace(
            solver_instance=SimpleNamespace(
                working_directory=str(matrix_working)
            )
        )
        simulation._matrix_hpc_acf_path = str(
            matrix_working / "pyaedt_config.acf"
        )
        simulation.NUM_CORE = 4
        simulation.NUM_TASK = 1
        simulation.loss_native_analyze_required = True
        simulation.solve_attempts = {}
        simulation._log_recent_aedt_messages = Mock()
        return simulation, desktop, project, raw_design, high_level_analyze

    @staticmethod
    def _no_wait_preflight(simulation, max_attempts=5):
        original = simulation._prepare_copied_loss_native_analysis
        simulation._prepare_copied_loss_native_analysis = lambda: original(
            max_attempts=max_attempts,
            initial_retry_delay=0,
            sleeper=lambda _seconds: None,
        )

    def test_transient_registry_preflight_then_exactly_one_native_solve(self):
        with tempfile.TemporaryDirectory() as root:
            simulation, desktop, project, design, high_level = self._simulation(
                root, registry_failures=2
            )
            self._no_wait_preflight(simulation)
            extracted = []

            simulation.analyze_and_extract(
                "loss", lambda: extracted.append(True)
            )

            design.Analyze.assert_called_once_with("Setup1", True)
            high_level.assert_not_called()
            self.assertEqual(extracted, [True])
            self.assertEqual(simulation.solve_attempts, {"loss": 1})
            self.assertEqual(simulation.design1.save_project.call_count, 2)
            simulation.project.project.Save.assert_not_called()
            self.assertEqual(desktop.active_config, "Local")
            self.assertEqual(len(desktop.registry_loads), 1)
            self.assertEqual(
                project.active_calls,
                ["maxwell_matrix1"] * 5,
            )

    def test_transient_getpath_capture_then_cached_loss_solve_exactly_once(self):
        with tempfile.TemporaryDirectory() as root:
            simulation, _desktop, _project, design, high_level = self._simulation(
                root
            )
            matrix_working = Path(root) / "matrix_working"

            class FlakySolver:
                def __init__(self):
                    self.calls = 0

                @property
                def working_directory(self):
                    self.calls += 1
                    if self.calls < 3:
                        raise RuntimeError("Failed to execute gRPC AEDT command: GetPath")
                    return str(matrix_working)

            solver = FlakySolver()
            simulation.design_matrix = SimpleNamespace(solver_instance=solver)
            del simulation._matrix_hpc_acf_path
            simulation.solve_attempts = {"matrix": 1}
            sleeper = Mock()

            captured = simulation._capture_matrix_hpc_acf(
                max_attempts=3, retry_delay=0.25, sleeper=sleeper
            )
            self._no_wait_preflight(simulation)
            simulation.analyze_and_extract("loss", lambda: None)

            self.assertEqual(captured, str(matrix_working / "pyaedt_config.acf"))
            self.assertEqual(solver.calls, 3)
            self.assertEqual(
                [call.args[0] for call in sleeper.call_args_list],
                [0.25, 0.5],
            )
            design.Analyze.assert_called_once_with("Setup1", True)
            high_level.assert_not_called()
            self.assertEqual(simulation.solve_attempts, {"matrix": 1, "loss": 1})

    def test_getpath_capture_exhaustion_never_dispatches_loss_solve(self):
        with tempfile.TemporaryDirectory() as root:
            simulation, desktop, _project, design, high_level = self._simulation(
                root
            )

            class FailedSolver:
                def __init__(self):
                    self.calls = 0

                @property
                def working_directory(self):
                    self.calls += 1
                    raise RuntimeError(
                        "Failed to execute gRPC AEDT command: GetPath"
                    )

            solver = FailedSolver()
            simulation.design_matrix = SimpleNamespace(solver_instance=solver)
            del simulation._matrix_hpc_acf_path
            simulation.solve_attempts = {"matrix": 1}
            sleeper = Mock()

            with self.assertRaisesRegex(
                    RuntimeError, "capture failed before copied-loss design creation"):
                simulation._capture_matrix_hpc_acf(
                    max_attempts=3, retry_delay=0.25, sleeper=sleeper
                )

            self.assertEqual(solver.calls, 3)
            self.assertEqual(
                [call.args[0] for call in sleeper.call_args_list],
                [0.25, 0.5],
            )
            design.Analyze.assert_not_called()
            high_level.assert_not_called()
            self.assertEqual(simulation.solve_attempts, {"matrix": 1})
            self.assertEqual(desktop.registry_loads, [])

    def test_save_preflight_and_analyze_are_adjacent_and_ordered(self):
        with tempfile.TemporaryDirectory() as root:
            simulation, desktop, project, design, _high_level = self._simulation(root)
            events = []
            phase = ["before"]
            original_prepare = simulation._prepare_copied_loss_native_analysis
            original_set_active_design = project.SetActiveDesign
            original_get_registry = desktop.GetRegistryString
            simulation.design1.save_project = Mock(
                side_effect=lambda: events.append("save")
            )

            def prepare():
                events.append("preflight_start")
                context = original_prepare(
                    initial_retry_delay=0,
                    sleeper=lambda _seconds: None,
                )
                events.append("preflight_done")
                phase[0] = "armed"
                return context

            def set_active_design(name):
                if phase[0] == "armed":
                    events.append("unexpected_set_active_design")
                return original_set_active_design(name)

            def get_registry(key):
                if phase[0] == "armed":
                    events.append("unexpected_get_registry")
                return original_get_registry(key)

            def analyze(*_args):
                events.append("analyze")
                phase[0] = "dispatched"
                return 0

            simulation._prepare_copied_loss_native_analysis = prepare
            project.SetActiveDesign = set_active_design
            desktop.GetRegistryString = get_registry
            design.Analyze.side_effect = analyze

            simulation.analyze_and_extract("loss", lambda: None)

            self.assertLess(events.index("save"), events.index("preflight_start"))
            self.assertLess(
                events.index("preflight_start"), events.index("preflight_done")
            )
            self.assertEqual(
                events[events.index("preflight_done") + 1],
                "analyze",
            )
            self.assertNotIn("unexpected_set_active_design", events)
            self.assertNotIn("unexpected_get_registry", events)
            self.assertEqual(simulation.solve_attempts, {"loss": 1})
            self.assertEqual(desktop.active_config, "Local")
            self.assertEqual(simulation.design1.save_project.call_count, 2)

    def test_logging_failure_before_preflight_never_mutates_dso(self):
        with tempfile.TemporaryDirectory() as root:
            simulation, desktop, _project, design, _high_level = self._simulation(root)

            with patch.object(
                    logging, "info", side_effect=RuntimeError("logging failed")):
                with self.assertRaisesRegex(RuntimeError, "logging failed"):
                    simulation.analyze_and_extract("loss", lambda: None)

            design.Analyze.assert_not_called()
            self.assertEqual(simulation.solve_attempts, {})
            self.assertEqual(desktop.active_config, "Local")
            self.assertEqual(desktop.registry_loads, [])
            simulation.design1.save_project.assert_called_once_with()

    def test_exception_after_preflight_restores_without_analyze(self):
        class ExplodingTelemetry(dict):
            def get(self, _key, _default=None):
                raise RuntimeError("telemetry update failed")

        with tempfile.TemporaryDirectory() as root:
            simulation, desktop, _project, design, _high_level = self._simulation(root)
            simulation.solve_attempts = ExplodingTelemetry()
            self._no_wait_preflight(simulation)

            with self.assertRaisesRegex(RuntimeError, "telemetry update failed"):
                simulation.analyze_and_extract("loss", lambda: None)

            design.Analyze.assert_not_called()
            self.assertEqual(desktop.active_config, "Local")
            self.assertEqual(len(desktop.registry_loads), 1)

    def test_preflight_exhaustion_never_dispatches_or_extracts(self):
        with tempfile.TemporaryDirectory() as root:
            simulation, _desktop, _project, design, high_level = self._simulation(
                root, registry_failures=10
            )
            self._no_wait_preflight(simulation, max_attempts=3)
            extracted = []

            with self.assertRaisesRegex(RuntimeError, "preflight failed closed"):
                simulation.analyze_and_extract(
                    "loss", lambda: extracted.append(True)
                )

            design.Analyze.assert_not_called()
            high_level.assert_not_called()
            self.assertEqual(extracted, [])
            self.assertEqual(simulation.solve_attempts, {})
            simulation.design1.save_project.assert_called_once_with()

    def test_wrong_design_identity_fails_immediately_without_mutation(self):
        with tempfile.TemporaryDirectory() as root:
            simulation, desktop, _project, design, high_level = self._simulation(
                root, design_name="maxwell_matrix"
            )
            self._no_wait_preflight(simulation)

            with self.assertRaisesRegex(RuntimeError, "design identity mismatch"):
                simulation.analyze_and_extract("loss", lambda: None)

            self.assertEqual(len(desktop.project_calls), 1)
            self.assertEqual(desktop.registry_loads, [])
            design.Analyze.assert_not_called()
            high_level.assert_not_called()
            self.assertEqual(simulation.solve_attempts, {})

    def test_ambiguous_or_invalid_native_return_never_retries(self):
        outcomes = (
            RuntimeError("response lost after possible dispatch"),
            False,
            True,
            1,
        )
        for outcome in outcomes:
            with self.subTest(outcome=repr(outcome)), tempfile.TemporaryDirectory() as root:
                simulation, desktop, _project, design, high_level = self._simulation(root)
                if isinstance(outcome, Exception):
                    design.Analyze.side_effect = outcome
                else:
                    design.Analyze.return_value = outcome
                self._no_wait_preflight(simulation)
                extracted = []

                with self.assertRaises((RuntimeError, type(outcome)) if isinstance(
                        outcome, Exception) else RuntimeError):
                    simulation.analyze_and_extract(
                        "loss", lambda: extracted.append(True)
                    )

                design.Analyze.assert_called_once_with("Setup1", True)
                high_level.assert_not_called()
                self.assertEqual(extracted, [])
                self.assertEqual(simulation.solve_attempts, {"loss": 1})
                self.assertEqual(desktop.active_config, "Local")

    def test_void_native_return_uses_postcheck_and_extraction_as_proof(self):
        with tempfile.TemporaryDirectory() as root:
            simulation, desktop, _project, design, high_level = self._simulation(
                root, analyze_result=None
            )
            self._no_wait_preflight(simulation)
            extracted = []

            simulation.analyze_and_extract(
                "loss", lambda: extracted.append(True)
            )

            design.Analyze.assert_called_once_with("Setup1", True)
            high_level.assert_not_called()
            self.assertEqual(extracted, [True])
            self.assertEqual(simulation.solve_attempts, {"loss": 1})
            self.assertEqual(desktop.active_config, "Local")

    def test_predispatch_save_failure_restores_dso_without_solve(self):
        with tempfile.TemporaryDirectory() as root:
            simulation, desktop, _project, design, _high_level = self._simulation(root)
            simulation.design1.save_project.side_effect = RuntimeError(
                "wrapper save failed"
            )
            simulation.project.project.Save.side_effect = RuntimeError(
                "native save failed"
            )
            self._no_wait_preflight(simulation)

            with self.assertRaisesRegex(RuntimeError, "Failed to save project"):
                simulation.analyze_and_extract("loss", lambda: None)

            design.Analyze.assert_not_called()
            self.assertEqual(simulation.solve_attempts, {})
            self.assertEqual(desktop.active_config, "Local")
            self.assertEqual(desktop.registry_loads, [])

    def test_predispatch_wrapper_save_failure_uses_verified_native_save(self):
        with tempfile.TemporaryDirectory() as root:
            simulation, _desktop, _project, design, _high_level = self._simulation(root)
            simulation.design1.save_project.side_effect = RuntimeError(
                "wrapper save failed"
            )
            self._no_wait_preflight(simulation)

            simulation.analyze_and_extract("loss", lambda: None)

            design.Analyze.assert_called_once_with("Setup1", True)
            self.assertEqual(simulation.project.project.Save.call_count, 2)
            self.assertEqual(simulation.solve_attempts, {"loss": 1})

    def test_extraction_or_postcheck_failure_never_resolves_again(self):
        for stage in ("postcheck", "extract"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as root:
                simulation, _desktop, _project, design, _high_level = self._simulation(root)
                self._no_wait_preflight(simulation)
                extracted = []
                if stage == "postcheck":
                    simulation._postcheck_copied_loss_native_analysis = Mock(
                        side_effect=RuntimeError("postcheck failed")
                    )
                    extractor = lambda: extracted.append(True)
                else:
                    extractor = lambda: (_ for _ in ()).throw(
                        RuntimeError("extract failed")
                    )

                with self.assertRaisesRegex(RuntimeError, f"{stage} failed"):
                    simulation.analyze_and_extract("loss", extractor)

                design.Analyze.assert_called_once_with("Setup1", True)
                self.assertEqual(simulation.solve_attempts, {"loss": 1})
                self.assertEqual(extracted, [])

    def test_rejects_matrix_acf_with_wrong_core_contract(self):
        with tempfile.TemporaryDirectory() as root:
            simulation, _desktop, _project, design, _high_level = self._simulation(root)
            acf = Path(
                simulation.design_matrix.solver_instance.working_directory
            ) / "pyaedt_config.acf"
            acf.write_text(self._acf_text(cores=8), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "ACF contract mismatch"):
                simulation.analyze_and_extract("loss", lambda: None)

            design.Analyze.assert_not_called()
            self.assertEqual(simulation.solve_attempts, {})

    def test_rejects_truncated_matrix_acf_without_dso_end(self):
        with tempfile.TemporaryDirectory() as root:
            simulation, _desktop, _project, design, _high_level = self._simulation(root)
            acf = Path(
                simulation.design_matrix.solver_instance.working_directory
            ) / "pyaedt_config.acf"
            acf.write_text(
                self._acf_text().replace("$end 'DSOConfig'\n", ""),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "ACF contract mismatch"):
                simulation.analyze_and_extract("loss", lambda: None)

            design.Analyze.assert_not_called()
            self.assertEqual(simulation.solve_attempts, {})

    def test_accepts_nested_crlf_acf_from_path_with_spaces(self):
        with tempfile.TemporaryDirectory() as root:
            simulation, _desktop, _project, _design, _high_level = self._simulation(root)
            old = Path(simulation.design_matrix.solver_instance.working_directory)
            spaced = Path(root) / "matrix working directory"
            old.rename(spaced)
            simulation.design_matrix.solver_instance.working_directory = str(spaced)
            inner = self._acf_text().replace("\n", "\r\n")
            (spaced / "pyaedt_config.acf").write_text(
                "$begin 'Configs'\r\n" + inner + "$end 'Configs'\r\n",
                encoding="utf-8",
                newline="",
            )

            path = simulation._validated_matrix_hpc_acf()

            self.assertEqual(path, str(spaced / "pyaedt_config.acf"))

    def test_identity_failure_after_dso_activation_restores_without_solve(self):
        with tempfile.TemporaryDirectory() as root:
            simulation, desktop, project, design, _high_level = self._simulation(root)
            wrong = self._Design(name="maxwell_matrix")
            returns = iter([design, wrong])
            project.SetActiveDesign = Mock(side_effect=lambda _name: next(returns))
            self._no_wait_preflight(simulation)

            with self.assertRaisesRegex(RuntimeError, "design identity mismatch"):
                simulation.analyze_and_extract("loss", lambda: None)

            design.Analyze.assert_not_called()
            self.assertEqual(simulation.solve_attempts, {})
            self.assertEqual(desktop.active_config, "Local")
            self.assertEqual(len(desktop.registry_loads), 1)

    def test_restore_readback_failure_never_retries_dispatched_solve(self):
        with tempfile.TemporaryDirectory() as root:
            simulation, desktop, _project, design, _high_level = self._simulation(root)
            original_set = desktop.SetRegistryString

            def refuse_restore(key, value):
                if value == "Local":
                    return None
                return original_set(key, value)

            desktop.SetRegistryString = Mock(side_effect=refuse_restore)
            self._no_wait_preflight(simulation)
            original_restore = simulation._restore_native_maxwell_dso
            simulation._restore_native_maxwell_dso = lambda key, value: original_restore(
                key, value, max_attempts=2, retry_delay=0,
                sleeper=lambda _seconds: None,
            )
            extracted = []

            with self.assertRaisesRegex(RuntimeError, "DSO restore failed"):
                simulation.analyze_and_extract(
                    "loss", lambda: extracted.append(True)
                )

            design.Analyze.assert_called_once_with("Setup1", True)
            self.assertEqual(simulation.solve_attempts, {"loss": 1})
            self.assertEqual(extracted, [])

    def test_preexisting_pyaedt_config_needs_no_restore(self):
        with tempfile.TemporaryDirectory() as root:
            simulation, desktop, _project, design, _high_level = self._simulation(root)
            desktop.active_config = "pyaedt_config"
            self._no_wait_preflight(simulation)

            simulation.analyze_and_extract("loss", lambda: None)

            design.Analyze.assert_called_once_with("Setup1", True)
            self.assertEqual(desktop.active_config, "pyaedt_config")
            self.assertTrue(any(
                value == "pyaedt_config" for _key, value in desktop.registry_sets
            ))
            self.assertFalse(any(
                value == "Local" for _key, value in desktop.registry_sets
            ))


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

    @staticmethod
    def _project_wrapper(native_project, desktop):
        class ProjectWrapper:
            def __init__(self):
                self.project = native_project
                self.proj = native_project
                self.desktop = SimpleNamespace(odesktop=desktop)

            @property
            def name(self):
                return self.project.GetName()

        return ProjectWrapper()

    def test_pooled_rebind_enumerates_fresh_exact_project(self):
        stale = SimpleNamespace(GetName=Mock(return_value="simulation_test"))
        fresh = SimpleNamespace(GetName=Mock(return_value="simulation_test"))
        simulation = Simulation.__new__(Simulation)
        simulation.aedt_backend = "pooled"
        simulation.PROJECT_NAME = "simulation_test"
        simulation.project = self._project_wrapper(stale, SimpleNamespace())
        simulation._refresh_native_project_handle = Mock(return_value=fresh)

        result = simulation._rebind_native_project_for_design_creation()

        self.assertIs(result, fresh)
        simulation._refresh_native_project_handle.assert_called_once_with()
        stale.GetName.assert_not_called()

    def test_rebinds_stale_project_before_design_creation(self):
        stale = SimpleNamespace(
            GetName=Mock(side_effect=RuntimeError("stale GetName"))
        )
        fresh = SimpleNamespace(GetName=Mock(return_value="simulation_test"))
        set_active_project = Mock(return_value=fresh)
        desktop = SimpleNamespace(SetActiveProject=set_active_project)
        simulation = Simulation.__new__(Simulation)
        simulation.PROJECT_NAME = "simulation_test"
        simulation.desktop = SimpleNamespace(odesktop=desktop)
        simulation.project = self._project_wrapper(stale, desktop)

        result = simulation._rebind_native_project_for_design_creation()

        self.assertIs(result, fresh)
        self.assertIs(simulation.project.project, fresh)
        self.assertIs(simulation.project.proj, fresh)
        self.assertEqual(fresh.GetName.call_count, 2)
        stale.GetName.assert_not_called()
        set_active_project.assert_called_once_with("simulation_test")

    def test_project_rebind_retries_only_before_design_creation(self):
        fresh = SimpleNamespace(GetName=Mock(return_value="simulation_test"))
        set_active_project = Mock(
            side_effect=[RuntimeError("transient transport"), fresh]
        )
        desktop = SimpleNamespace(SetActiveProject=set_active_project)
        stale = SimpleNamespace(GetName=Mock(side_effect=RuntimeError("stale")))
        sleeper = Mock()
        simulation = Simulation.__new__(Simulation)
        simulation.PROJECT_NAME = "simulation_test"
        simulation.desktop = SimpleNamespace(odesktop=desktop)
        simulation.project = self._project_wrapper(stale, desktop)

        result = simulation._rebind_native_project_for_design_creation(
            max_attempts=3, retry_delay=0.5, sleeper=sleeper
        )

        self.assertIs(result, fresh)
        self.assertEqual(set_active_project.call_count, 2)
        sleeper.assert_called_once_with(0.5)

    def test_project_rebind_rolls_back_partial_binding_before_retry(self):
        stale = SimpleNamespace(GetName=Mock(side_effect=RuntimeError("stale")))
        first = SimpleNamespace(GetName=Mock(side_effect=[
            "simulation_test", RuntimeError("transient rebound readback")
        ]))
        second = SimpleNamespace(GetName=Mock(return_value="simulation_test"))
        responses = iter([first, second])
        observed_bindings = []
        desktop = SimpleNamespace()
        simulation = Simulation.__new__(Simulation)
        simulation.PROJECT_NAME = "simulation_test"
        simulation.desktop = SimpleNamespace(odesktop=desktop)
        simulation.project = self._project_wrapper(stale, desktop)

        def set_active_project(_name):
            observed_bindings.append(simulation.project.project)
            return next(responses)

        desktop.SetActiveProject = Mock(side_effect=set_active_project)

        result = simulation._rebind_native_project_for_design_creation(
            max_attempts=2, retry_delay=0, sleeper=lambda _seconds: None
        )

        self.assertIs(result, second)
        self.assertEqual(observed_bindings, [stale, stale])
        self.assertIs(simulation.project.project, second)
        self.assertIs(simulation.project.proj, second)

    def test_project_rebind_exhaustion_preserves_original_handles(self):
        stale = SimpleNamespace(GetName=Mock(side_effect=RuntimeError("stale")))
        set_active_project = Mock(side_effect=RuntimeError("permanent transport"))
        desktop = SimpleNamespace(SetActiveProject=set_active_project)
        sleeper = Mock()
        simulation = Simulation.__new__(Simulation)
        simulation.PROJECT_NAME = "simulation_test"
        simulation.desktop = SimpleNamespace(odesktop=desktop)
        simulation.project = self._project_wrapper(stale, desktop)

        with self.assertRaisesRegex(RuntimeError, "rebind failed before design creation"):
            simulation._rebind_native_project_for_design_creation(
                max_attempts=3, retry_delay=0.5, sleeper=sleeper
            )

        self.assertEqual(set_active_project.call_count, 3)
        self.assertEqual(
            [call.args[0] for call in sleeper.call_args_list],
            [0.5, 1.0],
        )
        self.assertIs(simulation.project.project, stale)
        self.assertIs(simulation.project.proj, stale)

    def test_project_rebind_requires_positive_attempt_count(self):
        simulation = Simulation.__new__(Simulation)

        with self.assertRaisesRegex(ValueError, "max_attempts must be positive"):
            simulation._rebind_native_project_for_design_creation(max_attempts=0)

    def test_project_rebind_identity_mismatch_fails_without_retry(self):
        stale = SimpleNamespace(GetName=Mock(side_effect=RuntimeError("stale")))
        wrong = SimpleNamespace(GetName=Mock(return_value="another_project"))
        set_active_project = Mock(return_value=wrong)
        desktop = SimpleNamespace(SetActiveProject=set_active_project)
        simulation = Simulation.__new__(Simulation)
        simulation.PROJECT_NAME = "simulation_test"
        simulation.desktop = SimpleNamespace(odesktop=desktop)
        simulation.project = self._project_wrapper(stale, desktop)

        with self.assertRaisesRegex(RuntimeError, "thermal project identity mismatch"):
            simulation._rebind_native_project_for_design_creation(
                max_attempts=3, retry_delay=0, sleeper=lambda _seconds: None
            )

        set_active_project.assert_called_once_with("simulation_test")
        self.assertIs(simulation.project.project, stale)
        self.assertIs(simulation.project.proj, stale)

    def test_thermal_entry_rebinds_before_icepak_creation(self):
        class StopAfterCreate(Exception):
            pass

        events = []
        analyze = Mock(side_effect=AssertionError("thermal recovery must not re-solve EM"))
        solve_attempts = {"matrix": 1, "loss": 1}

        def stop_after_create(**_kwargs):
            events.append("create")
            raise StopAfterCreate()

        create_design = Mock(side_effect=stop_after_create)
        simulation = SimpleNamespace(
            df_plus=pd.DataFrame({"thermal_symmetry": ["eighth"]}),
            design1=SimpleNamespace(setup=SimpleNamespace(analyze=analyze)),
            solve_attempts=solve_attempts,
            _rebind_native_project_for_design_creation=lambda: events.append("rebind"),
            project=SimpleNamespace(create_design=create_design),
        )

        with self.assertRaises(StopAfterCreate):
            run_thermal_analysis(simulation)

        self.assertEqual(events, ["rebind", "create"])
        create_design.assert_called_once_with(
            name="icepak_thermal", solver="icepak",
            solution="SteadyState TemperatureAndFlow",
        )
        self.assertEqual(simulation.solve_attempts, {"matrix": 1, "loss": 1})
        analyze.assert_not_called()


class ThermalDispatchPolicyTests(unittest.TestCase):
    class _Clock:
        def __init__(self):
            self.value = 0.0

        def __call__(self):
            return self.value

        def sleep(self, seconds):
            self.value += float(seconds)

    @staticmethod
    def _telemetry(reason, converged=0, monitor_file=""):
        return {
            "thermal_convergence_available": 1 if monitor_file else 0,
            "thermal_converged": int(converged),
            "thermal_iterations": 10 if monitor_file else 0,
            "thermal_residual_continuity": 5e-4 if monitor_file else float("nan"),
            "thermal_residual_x_velocity": 4e-4 if monitor_file else float("nan"),
            "thermal_residual_y_velocity": 4e-4 if monitor_file else float("nan"),
            "thermal_residual_z_velocity": 4e-4 if monitor_file else float("nan"),
            "thermal_residual_energy": 1e-8 if monitor_file else float("nan"),
            "thermal_residual_flow_limit": 1e-3 if monitor_file else float("nan"),
            "thermal_residual_energy_limit": 1e-7 if monitor_file else float("nan"),
            "thermal_convergence_reason": reason,
            "thermal_monitor_file": monitor_file,
        }

    @staticmethod
    def _harness(analyze_side_effect=None, setups=("ThermalSetup",), enabled=True):
        class NativeSolver(SimpleNamespace):
            @property
            def setup_names(self):
                module = getattr(self, "_oanalysis", None)
                return list(module.GetSetups()) if module else []

        analysis = SimpleNamespace(GetSetups=Mock(return_value=list(setups)))
        native_design = SimpleNamespace(
            GetName=Mock(return_value="icepak_thermal"),
            GetDesignType=Mock(return_value="Icepak"),
            GetModule=Mock(return_value=analysis),
        )
        native_project = SimpleNamespace(
            GetName=Mock(return_value="simulation_test"),
            SetActiveDesign=Mock(return_value=native_design),
        )
        if isinstance(analyze_side_effect, list):
            analyze = Mock(side_effect=analyze_side_effect)
        else:
            analyze = Mock(return_value=analyze_side_effect)
        native_design.Analyze = analyze
        native_solver = NativeSolver(
            design_name="icepak_thermal",
            oproject=native_project,
            _oanalysis=SimpleNamespace(
                GetSetups=Mock(return_value=["StaleSetup"])
            ),
            analyze=analyze,
        )
        mesh_operation = SimpleNamespace(
            name="tx_mesh_level_L", props={"Level": "4", "Objects": ["tx_0", "tx_1"]}
        )
        ipk = SimpleNamespace(
            design_name="icepak_thermal",
            solver_instance=native_solver,
            modeler=SimpleNamespace(object_names=["tx_0", "tx_1"], obounding_box=[0] * 6),
            mesh=SimpleNamespace(meshoperations=[mesh_operation]),
        )
        desktop = SimpleNamespace(
            AreThereSimulationsRunning=Mock(return_value=False),
            GetMessages=Mock(return_value=["Icepak startup diagnostic"]),
        )
        rebind = Mock(return_value=native_project)
        simulation = SimpleNamespace(
            PROJECT_NAME="simulation_test",
            NUM_CORE=4,
            _rebind_native_project_for_design_creation=rebind,
            _native_desktop_handle=Mock(return_value=desktop),
            save_project=Mock(),
            aedt_native_solve_window=lambda: nullcontext(),
        )
        setup = SimpleNamespace(name="ThermalSetup", props={
            "Enabled": enabled,
            "Convergence Criteria - Flow": "0.001",
            "Convergence Criteria - Energy": "1e-07",
        })
        return simulation, ipk, setup, analyze, rebind

    def test_stale_pyaedt_analysis_cache_is_rebound_before_native_dispatch(self):
        simulation, ipk, setup, _analyze, _rebind = self._harness(None)
        native_solver = ipk.solver_instance
        stale_analysis = native_solver._oanalysis
        native_design = native_solver.oproject.SetActiveDesign.return_value
        native_design.Analyze = Mock(return_value=None)

        def emulated_pyaedt_analyze(*, setup, cores, blocking):
            # PyAEDT analyze_setup silently returns its initial True value when
            # the requested name is absent from its cached setup_names.
            if setup in native_solver.setup_names:
                return native_solver._odesign.Analyze(setup, blocking)
            return True

        native_solver.analyze = emulated_pyaedt_analyze
        converged = self._telemetry(
            "converged", converged=1, monitor_file="fresh.sd"
        )

        with patch(
            "module.thermal_260706._thermal_convergence_telemetry",
            return_value=converged,
        ):
            result = _solve_exact_thermal_setup(
                simulation, ipk, setup, monitor_grace_s=0,
            )

        self.assertIsNot(native_solver._oanalysis, stale_analysis)
        self.assertIs(
            native_solver._oanalysis, native_design.GetModule.return_value
        )
        native_design.Analyze.assert_called_once_with("ThermalSetup", True)
        self.assertEqual(result["solve_attempts"], 1)
        self.assertEqual(result["convergence"]["thermal_converged"], 1)
        forensic = json.loads(result["forensic_json"])
        self.assertEqual(
            forensic["attempts"][0]["identity"]["wrapper_setups"],
            ["ThermalSetup"],
        )

    def test_false_with_delayed_converged_monitor_dispatches_exact_setup_once(self):
        simulation, ipk, setup, analyze, rebind = self._harness(False)
        clock = self._Clock()
        missing = self._telemetry("monitor_missing")
        converged = self._telemetry("converged", converged=1, monitor_file="fresh.sd")

        with patch(
            "module.thermal_260706._thermal_convergence_telemetry",
            side_effect=[missing, converged],
        ):
            result = _solve_exact_thermal_setup(
                simulation, ipk, setup, monitor_grace_s=2, poll_s=1,
                clock=clock, sleeper=clock.sleep,
            )

        analyze.assert_called_once_with(
            setup="ThermalSetup", cores=4, blocking=True
        )
        rebind.assert_called_once_with()
        self.assertEqual(result["solve_attempts"], 1)
        self.assertFalse(result["analyze_call_ok"])
        self.assertTrue(result["analyze_return_false"])
        self.assertEqual(result["convergence"]["thermal_converged"], 1)

    def test_pooled_thermal_dispatch_reuses_host_dso_without_cores_override(self):
        simulation, ipk, setup, analyze, _rebind = self._harness(None)
        simulation.solver_may_be_running = False
        converged = self._telemetry(
            "converged", converged=1, monitor_file="fresh.sd"
        )

        def assert_dispatch_window(*_args, **_kwargs):
            self.assertTrue(simulation.solver_may_be_running)
            return None

        analyze.side_effect = assert_dispatch_window

        with patch(
            "module.aedt_pool_adapter.pooled_backend_enabled",
            return_value=True,
        ), patch(
            "module.thermal_260706._thermal_convergence_telemetry",
            return_value=converged,
        ):
            result = _solve_exact_thermal_setup(
                simulation, ipk, setup, monitor_grace_s=0,
            )

        analyze.assert_called_once_with("ThermalSetup", True)
        self.assertEqual(result["solve_attempts"], 1)
        self.assertEqual(result["convergence"]["thermal_converged"], 1)
        self.assertFalse(simulation.solver_may_be_running)

    def test_pooled_thermal_preflight_failure_is_project_local(self):
        simulation, ipk, setup, analyze, _rebind = self._harness(
            None, setups=("OtherSetup",)
        )
        simulation.solver_may_be_running = False

        with patch(
            "module.aedt_pool_adapter.pooled_backend_enabled",
            return_value=True,
        ), self.assertRaisesRegex(RuntimeError, "native thermal setup mismatch"):
            _solve_exact_thermal_setup(
                simulation, ipk, setup, monitor_grace_s=0,
            )

        analyze.assert_not_called()
        self.assertFalse(simulation.solver_may_be_running)

    def test_pooled_thermal_false_without_native_evidence_stays_uncertain(self):
        simulation, ipk, setup, _analyze, _rebind = self._harness(False)
        simulation.solver_may_be_running = False
        missing = self._telemetry("monitor_missing")
        desktop = simulation._native_desktop_handle.return_value
        desktop.AreThereSimulationsRunning.side_effect = [
            False,
            RuntimeError("transport state unavailable"),
        ]

        with patch(
            "module.aedt_pool_adapter.pooled_backend_enabled",
            return_value=True,
        ), patch(
            "module.thermal_260706._thermal_convergence_telemetry",
            return_value=missing,
        ), self.assertRaisesRegex(
            RuntimeError, "pooled thermal native solve state is uncertain"
        ):
            _solve_exact_thermal_setup(
                simulation, ipk, setup, monitor_grace_s=0,
            )

        self.assertTrue(simulation.solver_may_be_running)

    def test_pooled_thermal_exception_with_monitor_stays_uncertain_until_idle(self):
        simulation, ipk, setup, analyze, _rebind = self._harness(None)
        analyze.side_effect = RuntimeError("native analyze transport failed")
        simulation.solver_may_be_running = False
        malformed = self._telemetry("monitor_malformed", monitor_file="fresh.sd")
        desktop = simulation._native_desktop_handle.return_value
        desktop.AreThereSimulationsRunning.side_effect = [
            False,
            RuntimeError("transport state unavailable"),
        ]

        with patch(
            "module.aedt_pool_adapter.pooled_backend_enabled",
            return_value=True,
        ), patch(
            "module.thermal_260706._thermal_convergence_telemetry",
            return_value=malformed,
        ), self.assertRaisesRegex(
            RuntimeError, "pooled thermal native solve state is uncertain"
        ):
            _solve_exact_thermal_setup(
                simulation, ipk, setup, monitor_grace_s=0,
            )

        self.assertTrue(simulation.solver_may_be_running)

    def test_false_and_missing_monitor_rebinds_and_retries_only_once(self):
        simulation, ipk, setup, analyze, rebind = self._harness([False, False])
        missing = self._telemetry("monitor_missing")

        with patch(
            "module.thermal_260706._thermal_convergence_telemetry",
            side_effect=[missing, missing, missing],
        ):
            result = _solve_exact_thermal_setup(
                simulation, ipk, setup, monitor_grace_s=0,
            )

        self.assertEqual(analyze.call_count, 2)
        for call in analyze.call_args_list:
            self.assertEqual(call.kwargs, {
                "setup": "ThermalSetup", "cores": 4, "blocking": True,
            })
        self.assertEqual(rebind.call_count, 2)
        self.assertEqual(result["solve_attempts"], 2)
        self.assertEqual(result["dispatch_status"], "false")
        self.assertEqual(
            result["convergence"]["thermal_convergence_reason"], "monitor_missing"
        )
        forensic = json.loads(result["forensic_json"])
        self.assertEqual(forensic["schema"], "thermal-dispatch-forensic-v1")
        self.assertEqual(len(forensic["attempts"]), 2)
        self.assertEqual(
            forensic["attempts"][0]["aedt_messages"],
            ["Icepak startup diagnostic"],
        )
        self.assertEqual(
            forensic["attempts"][0]["identity"]["model_context"]
            ["mesh_operations"][0],
            {"level": "4", "name": "tx_mesh_level_L", "object_count": 2},
        )

    def test_fresh_residual_threshold_is_terminal_without_retry(self):
        simulation, ipk, setup, analyze, rebind = self._harness(False)
        residual = self._telemetry(
            "residual_threshold", converged=0, monitor_file="fresh.sd"
        )

        with patch(
            "module.thermal_260706._thermal_convergence_telemetry",
            return_value=residual,
        ):
            result = _solve_exact_thermal_setup(
                simulation, ipk, setup, monitor_grace_s=0,
            )

        analyze.assert_called_once_with(
            setup="ThermalSetup", cores=4, blocking=True
        )
        rebind.assert_called_once_with()
        self.assertEqual(result["solve_attempts"], 1)
        self.assertEqual(
            result["convergence"]["thermal_convergence_reason"],
            "residual_threshold",
        )

    def test_exception_and_missing_monitor_dispatches_at_most_twice(self):
        simulation, ipk, setup, analyze, rebind = self._harness([
            RuntimeError("startup one"), RuntimeError("startup two"),
        ])
        missing = self._telemetry("monitor_missing")

        with patch(
            "module.thermal_260706._thermal_convergence_telemetry",
            side_effect=[missing, missing, missing],
        ):
            result = _solve_exact_thermal_setup(
                simulation, ipk, setup, monitor_grace_s=0,
            )

        self.assertEqual(analyze.call_count, 2)
        self.assertEqual(rebind.call_count, 2)
        self.assertEqual(result["solve_attempts"], 2)
        self.assertEqual(result["dispatch_status"], "exception")
        self.assertEqual(result["dispatch_exception_type"], "RuntimeError")

    def test_numeric_running_state_zero_allows_and_one_blocks_dispatch(self):
        simulation, ipk, setup, analyze, _rebind = self._harness(None)
        desktop = simulation._native_desktop_handle.return_value
        desktop.AreThereSimulationsRunning.return_value = 0
        contract = _prepare_thermal_dispatch(simulation, ipk, setup)
        self.assertEqual(contract["design"], "icepak_thermal")

        desktop.AreThereSimulationsRunning.return_value = 1
        with self.assertRaisesRegex(RuntimeError, "overlapping simulation"):
            _solve_exact_thermal_setup(
                simulation, ipk, setup, monitor_grace_s=0,
            )
        analyze.assert_not_called()

    def test_none_with_missing_monitor_never_blindly_retries(self):
        simulation, ipk, setup, analyze, rebind = self._harness(None)
        missing = self._telemetry("monitor_missing")

        with patch(
            "module.thermal_260706._thermal_convergence_telemetry",
            return_value=missing,
        ):
            result = _solve_exact_thermal_setup(
                simulation, ipk, setup, monitor_grace_s=0,
            )

        analyze.assert_called_once_with(
            setup="ThermalSetup", cores=4, blocking=True
        )
        rebind.assert_called_once_with()
        self.assertEqual(result["solve_attempts"], 1)
        self.assertEqual(result["dispatch_status"], "success")
        self.assertEqual(
            result["convergence"]["thermal_convergence_reason"], "monitor_missing"
        )

    def test_setup_identity_and_enabled_state_fail_before_dispatch(self):
        for setups, enabled, expected in [
            (("OtherSetup",), True, "native thermal setup mismatch"),
            (("ThermalSetup",), False, "ThermalSetup is disabled"),
        ]:
            with self.subTest(setups=setups, enabled=enabled):
                simulation, ipk, setup, analyze, _rebind = self._harness(
                    None, setups=setups, enabled=enabled
                )
                with self.assertRaisesRegex(RuntimeError, expected):
                    _prepare_thermal_dispatch(simulation, ipk, setup)
                analyze.assert_not_called()

        simulation, ipk, setup, analyze, _rebind = self._harness(None)
        native_design = ipk.solver_instance.oproject.SetActiveDesign.return_value
        native_setup = SimpleNamespace(GetPropValue=Mock(return_value=False))
        native_analysis = SimpleNamespace(
            GetChildObject=Mock(return_value=native_setup)
        )
        native_design.GetChildObject = Mock(return_value=native_analysis)
        with self.assertRaisesRegex(RuntimeError, "native ThermalSetup Enabled"):
            _prepare_thermal_dispatch(simulation, ipk, setup)
        analyze.assert_not_called()

    def test_monitor_snapshot_rejects_stale_but_accepts_same_mtime_content_change(self):
        stale = (
            "1 Continuity(0.0020) XVelocity(0.0001) YVelocity(0.0001) "
            "ZVelocity(0.0001) Energy(0.00000001)\n"
        )
        converged = stale.replace("0.0020", "0.0002")
        self.assertEqual(len(stale), len(converged))
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "icepak_thermal.results"
            results.mkdir()
            monitor = results / "case_S1_MON1_V1.sd"
            monitor.write_text(stale, encoding="utf-8")
            native = SimpleNamespace(results_directory=directory)
            ipk = SimpleNamespace(
                design_name="icepak_thermal", solver_instance=native
            )
            simulation = SimpleNamespace()
            setup = SimpleNamespace(props={
                "Convergence Criteria - Flow": "0.001",
                "Convergence Criteria - Energy": "1e-07",
            })
            snapshot = _snapshot_thermal_monitors(simulation, ipk)
            unchanged = _thermal_convergence_telemetry(
                simulation, ipk, setup, attempts=1, monitor_snapshot=snapshot
            )
            original_stat = monitor.stat()
            os.utime(
                monitor,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1_000_000),
            )
            touched_only = _thermal_convergence_telemetry(
                simulation, ipk, setup, attempts=1, monitor_snapshot=snapshot
            )
            monitor.write_text(converged, encoding="utf-8")
            os.utime(
                monitor,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            changed = _thermal_convergence_telemetry(
                simulation, ipk, setup, attempts=1, monitor_snapshot=snapshot
            )

        self.assertEqual(
            unchanged["thermal_convergence_reason"], "monitor_missing"
        )
        self.assertEqual(
            touched_only["thermal_convergence_reason"], "monitor_missing"
        )
        self.assertEqual(changed["thermal_convergence_reason"], "converged")
        self.assertEqual(changed["thermal_monitor_file"], "case_S1_MON1_V1.sd")


class FieldsReporterTests(unittest.TestCase):
    def test_calcop_failure_clears_shared_stack_without_masking_error(self):
        class Reporter:
            def __init__(self):
                self.stack_operations = []

            def DoesNamedExpressionExists(self, _name):
                return False

            def CalcStack(self, operation):
                self.stack_operations.append(operation)
                if len(self.stack_operations) > 1:
                    raise RuntimeError("cleanup transport failed")

            def CalcOp(self, _operation):
                raise RuntimeError("Failed to execute gRPC AEDT command: CalcOp")

        reporter = Reporter()
        simulation = Simulation.__new__(Simulation)
        simulation._fresh_fields_reporter = Mock(return_value=reporter)

        with self.assertRaisesRegex(
            RuntimeError, "failed to register field expression 'Phi_test'"
        ):
            simulation._add_field_expression(
                "Phi_test",
                lambda fields: fields.CalcOp("Integrate"),
                max_attempts=1,
                retry_delay=0,
            )

        self.assertEqual(reporter.stack_operations, ["clear", "clear"])

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
            SimpleNamespace(
                GetName=lambda: "maxwell_loss",
                GetModule=lambda _name, reporter=reporter: reporter,
            )
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

    def test_recovers_reporter_from_fresh_project_without_another_solve(self):
        class Reporter:
            def __init__(self):
                self.added = []

            def DoesNamedExpressionExists(self, _name):
                return False

            def CalcStack(self, _operation):
                pass

            def EnterQty(self, _quantity):
                pass

            def AddNamedExpression(self, name, _category):
                self.added.append(name)

        class BrokenProject:
            def __init__(self):
                self.get_calls = 0
                self.set_calls = 0

            def GetActiveDesign(self):
                self.get_calls += 1
                raise RuntimeError("transient GetActiveDesign gRPC failure")

            def SetActiveDesign(self, _name):
                self.set_calls += 1
                raise RuntimeError("transient SetActiveDesign gRPC failure")

        reporter = Reporter()
        native_design = SimpleNamespace(
            GetName=lambda: "project;maxwell_loss",
            GetModule=lambda _name: reporter,
        )
        fresh_project = SimpleNamespace(
            GetActiveDesign=lambda: native_design,
            SetActiveDesign=Mock(side_effect=AssertionError("switch is unnecessary")),
        )
        broken_project = BrokenProject()
        desktop = SimpleNamespace(
            odesktop=SimpleNamespace(SetActiveProject=Mock(return_value=fresh_project))
        )
        analyze = Mock(side_effect=AssertionError("recovery must not re-solve"))
        simulation = Simulation.__new__(Simulation)
        simulation.design1 = SimpleNamespace(
            design_name="maxwell_loss",
            setup=SimpleNamespace(analyze=analyze),
        )
        simulation.project = SimpleNamespace(
            project=broken_project,
            proj=broken_project,
            desktop=desktop,
            name="project",
        )

        result = simulation._add_field_expression(
            "P_core_5", lambda value: value.EnterQty("CoreLoss"), retry_delay=0
        )

        self.assertEqual(result, "P_core_5")
        self.assertEqual(reporter.added, ["P_core_5"])
        self.assertEqual(broken_project.get_calls, 1)
        self.assertEqual(broken_project.set_calls, 1)
        desktop.odesktop.SetActiveProject.assert_called_once_with("project")
        fresh_project.SetActiveDesign.assert_not_called()
        analyze.assert_not_called()

    def test_permanent_reporter_recovery_failure_remains_fail_closed(self):
        class BrokenProject:
            def __init__(self):
                self.get_calls = 0
                self.set_calls = 0

            def GetActiveDesign(self):
                self.get_calls += 1
                raise RuntimeError("permanent GetActiveDesign failure")

            def SetActiveDesign(self, _name):
                self.set_calls += 1
                raise RuntimeError("permanent SetActiveDesign failure")

        broken_project = BrokenProject()
        set_active_project = Mock(side_effect=RuntimeError("permanent project failure"))
        analyze = Mock(side_effect=AssertionError("recovery must not re-solve"))
        simulation = Simulation.__new__(Simulation)
        simulation.design1 = SimpleNamespace(
            design_name="maxwell_loss",
            setup=SimpleNamespace(analyze=analyze),
        )
        simulation.project = SimpleNamespace(
            project=broken_project,
            proj=broken_project,
            desktop=SimpleNamespace(
                odesktop=SimpleNamespace(SetActiveProject=set_active_project)
            ),
            name="project",
        )

        with self.assertRaisesRegex(
            RuntimeError, "failed to register field expression 'P_core_5'"
        ):
            simulation._add_field_expression(
                "P_core_5", lambda _reporter: None,
                max_attempts=3, retry_delay=0,
            )

        self.assertEqual(broken_project.get_calls, 3)
        self.assertEqual(broken_project.set_calls, 3)
        self.assertEqual(set_active_project.call_count, 3)
        analyze.assert_not_called()

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


class ThermalHomogenizationTests(unittest.TestCase):
    class _Modeler:
        def __init__(self):
            self.calls = []

        def create_box(self, origin, sizes, name, material):
            numeric_sizes = [float(str(value).removesuffix("mm")) for value in sizes]
            obj = SimpleNamespace(
                name=name,
                volume=math.prod(numeric_sizes),
                origin=origin,
                sizes=sizes,
                material=material,
            )
            self.calls.append(obj)
            return obj

    def test_zero_explicit_turns_partitions_every_turn_into_blocks(self):
        turns = [SimpleNamespace(name=f"rx_{index}") for index in range(5)]

        explicit, blocked = _partition_rx_turns(turns, 0)

        self.assertEqual(explicit, [])
        self.assertEqual([turn.name for turn in blocked], [turn.name for turn in turns])
        explicit, blocked = _partition_rx_turns(turns, 2)
        self.assertEqual([turn.name for turn in explicit], ["rx_0", "rx_1", "rx_3", "rx_4"])
        self.assertEqual([turn.name for turn in blocked], ["rx_2"])

    def test_create_coil_false_return_preserves_the_polyline_root_cause(self):
        modeler = SimpleNamespace(create_polyline=Mock(return_value=False))
        design = SimpleNamespace(modeler=modeler)

        with self.assertRaisesRegex(
            RuntimeError,
            r"create_polyline returned no object for Rx_main_0_0.*"
            r"xsection_width=1\.0.*xsection_height=80\.0",
        ):
            create_coil_geometry(
                design=design,
                name="Rx_main",
                window_height=80.0,
                window_length=2.0,
                window_layer=2,
                N_input=1,
                width_fill_factor=1.0,
                space_length=100.0,
                space_width=120.0,
                shape="rectangle",
                color=[10, 10, 255],
            )

    def test_single_turn_stays_exact_copper_with_zero_explicit_setting(self):
        turn = SimpleNamespace(name="Rx_side_0_0")

        explicit, blocked = _partition_rx_turns([turn], 0)

        self.assertEqual(explicit, [turn])
        self.assertEqual(blocked, [])

    def test_zero_explicit_blocks_span_the_complete_rx_pack(self):
        frame = pd.DataFrame([{
            "n_explicit_turns": 0,
            "cw2": 0.5,
            "gap2": 0.25,
            "N2_main": 5,
            "sl2_main_x": 100.0,
            "sl2_main_y": 120.0,
        }])
        modeler = self._Modeler()
        ipk = SimpleNamespace(modeler=modeler)

        blocks = _build_homog_blocks(ipk, frame, "main", "Rx_main", 0.0, 80.0)

        _, width, x_pos, y_pos = _rx_layout(frame, "main")
        x_inner, x_outer = x_pos[0] - width / 2, x_pos[-1] + width / 2
        y_inner, y_outer = y_pos[0] - width / 2, y_pos[-1] + width / 2
        by_name = {block.name: block for block in blocks}
        self.assertEqual(set(by_name), {
            "Rx_main_block_xp", "Rx_main_block_xn",
            "Rx_main_block_yp", "Rx_main_block_yn",
        })
        self.assertEqual(by_name["Rx_main_block_xp"].origin[0], f"{x_inner}mm")
        self.assertEqual(by_name["Rx_main_block_xp"].sizes[0], f"{x_outer - x_inner}mm")
        self.assertEqual(by_name["Rx_main_block_yp"].origin[1], f"{y_inner}mm")
        self.assertEqual(by_name["Rx_main_block_yp"].sizes[1], f"{y_outer - y_inner}mm")

    def test_zero_explicit_group_skips_disposable_rx_polylines(self):
        # Regression for the observed Rx_main_13_0 CreatePolyline transport failure.
        frame = pd.DataFrame([{
            "N2_main": 22,
            "nwh2": 654.6,
        }])
        blocks = [SimpleNamespace(name=f"Rx_main_block_{tag}") for tag in (
            "xp", "xn", "yp", "yn",
        )]
        ipk = SimpleNamespace(modeler=SimpleNamespace())

        with patch(
            "module.thermal_260706.create_coil",
            side_effect=AssertionError("disposable Rx polylines must not be created"),
        ) as create, patch(
            "module.thermal_260706._build_homog_blocks", return_value=blocks,
        ) as build:
            explicit, actual_blocks = _build_rx_group(
                ipk, frame, "main", "Rx_main", 0.0, 0, 654.6,
            )

        create.assert_not_called()
        build.assert_called_once_with(
            ipk, frame, "main", "Rx_main", 0.0, 654.6,
        )
        self.assertEqual(explicit, [])
        self.assertIs(actual_blocks, blocks)

    def test_zero_explicit_single_turn_remains_exact_copper(self):
        frame = pd.DataFrame([{
            "N2_side": 1,
            "nwh2": 80.0,
            "nwl2_side": 0.5,
            "wff2_side": 1.0,
            "sl2_side_x": 100.0,
            "sl2_side_y": 120.0,
        }])
        turn = SimpleNamespace(name="Rx_side_0_0")
        ipk = SimpleNamespace(modeler=SimpleNamespace())

        with patch(
            "module.thermal_260706.create_coil",
            return_value=([turn], 1, 0.5, 80.0, 0.0, 0.0),
        ) as create, patch(
            "module.thermal_260706._build_homog_blocks",
            side_effect=AssertionError("one turn must not be homogenized"),
        ) as build:
            explicit, blocks = _build_rx_group(
                ipk, frame, "side", "Rx_side", -10.0, 0, 80.0,
            )

        create.assert_called_once()
        build.assert_not_called()
        self.assertEqual(explicit, [turn])
        self.assertEqual(blocks, [])

    def test_volume_weighted_distribution_preserves_group_power(self):
        blocks = [SimpleNamespace(volume=1.0), SimpleNamespace(volume=3.0)]

        powers = _volume_weighted_powers(blocks, 80.0)

        self.assertEqual(powers, [20.0, 60.0])
        self.assertAlmostEqual(sum(powers), 80.0)
        with self.assertRaisesRegex(RuntimeError, "invalid thermal block volumes"):
            _volume_weighted_powers([SimpleNamespace(volume=0.0)], 1.0)

    def test_production_default_uses_homogenized_rx_blocks(self):
        defaults = get_drawing_default_params()
        self.assertEqual(defaults["n_explicit_turns"], 0)
        self.assertEqual(defaults["matrix_percent_error"], 1.5)
        self.assertEqual(defaults["matrix_max_passes"], 20)
        self.assertEqual(defaults["percent_error"], 1.5)
        self.assertEqual(defaults["thermal_rx_side_block_mesh_level"], 5)
        candidate = create_input_parameter({
            "thermal_rx_side_block_mesh_level": 4
        })
        self.assertEqual(
            candidate["thermal_rx_side_block_mesh_level"].iloc[0], 4
        )


class EmCompletionPolicyTests(unittest.TestCase):
    @staticmethod
    def _valid_result():
        return pd.DataFrame([{
            "matrix_percent_error": 1.5,
            "matrix_min_converged": 1,
            "conv_passes_matrix": 6,
            "conv_consecutive_matrix": 1,
            "conv_error_pct_matrix": 1.1,
            "conv_delta_pct_matrix": 0.2,
            "Ltx": 900.0,
            "Lrx": 90_000.0,
            "M": 8_900.0,
            "k": 0.99,
            "Lmt": 882.0,
            "Lmr": 88_200.0,
            "Llt": 18.0,
            "Llr": 1_800.0,
            "percent_error": 1.5,
            "min_converged": 2,
            "conv_passes_loss": 4,
            "conv_consecutive_loss": 2,
            "conv_error_pct_loss": 0.7,
            "conv_delta_pct_loss": 0.3,
            "P_core_total": 1000.0,
            "P_core_plate_total": 10.0,
            "P_wcp_total": 20.0,
            "P_winding_total": 2000.0,
            "B_mean_core": 0.8,
            "B_max_core": 1.0,
        }])

    def test_accepts_finite_converged_matrix_and_loss(self):
        self.assertTrue(_em_result_is_valid(self._valid_result()))

    def test_rejects_observed_skin_free_matrix_false_positive(self):
        result = self._valid_result()
        result.loc[0, "conv_error_pct_matrix"] = 13.254
        result.loc[0, "conv_delta_pct_matrix"] = 0.1659

        valid, reason = _em_result_validation(result)

        self.assertFalse(valid)
        self.assertIn("matrix: energy error 13.254% exceeds 1.5%", reason)

    def test_rejects_non_finite_output_or_missing_convergence(self):
        result = self._valid_result()
        result.loc[0, "Llt"] = float("nan")
        self.assertFalse(_em_result_is_valid(result))

        missing_delta = self._valid_result().drop(columns=["conv_delta_pct_loss"])
        self.assertFalse(_em_result_is_valid(missing_delta))

        too_few_consecutive = self._valid_result()
        too_few_consecutive.loc[0, "conv_consecutive_loss"] = 1
        self.assertFalse(_em_result_is_valid(too_few_consecutive))

        missing_history_count = self._valid_result().drop(
            columns=["conv_consecutive_loss"]
        )
        self.assertFalse(_em_result_is_valid(missing_history_count))

    def test_validates_only_enabled_stages(self):
        matrix_only = self._valid_result().drop(columns=[
            "percent_error", "conv_passes_loss", "conv_consecutive_loss",
            "conv_error_pct_loss", "conv_delta_pct_loss", *list((
                "P_core_total", "P_core_plate_total", "P_wcp_total",
                "P_winding_total", "B_mean_core", "B_max_core",
            )),
        ])
        self.assertTrue(
            _em_result_is_valid(matrix_only, matrix_on=True, loss_on=False)
        )
        self.assertFalse(
            _em_result_is_valid(matrix_only, matrix_on=False, loss_on=False)
        )


class ConvergenceHistoryTests(unittest.TestCase):
    @staticmethod
    def _history(rows, completed=None):
        completed = len(rows) if completed is None else completed
        return "\n".join([
            "Number of Passes",
            f"Completed : {completed}",
            "Maximum   : 10",
            "Minimum   : 2",
            "Criterion : Energy Error/Delta Energy (%)",
            "Target    : (1.5, 1.5)",
            "Pass|# Tetrahedra|Total Energy (J)|Energy Error (%)|Delta Energy (%)|",
            *rows,
            "",
        ])

    @staticmethod
    def _extract(history):
        with tempfile.TemporaryDirectory() as tmp:
            def export_convergence(_setup, _variation, path):
                if history is not None:
                    Path(path).write_text(history, encoding="utf-8")

            simulation = Simulation.__new__(Simulation)
            simulation.project_path = tmp
            simulation.df_plus = pd.DataFrame([{
                "matrix_percent_error": 1.5,
                "percent_error": 1.5,
            }])
            simulation.design1 = SimpleNamespace(
                available_variations=SimpleNamespace(nominal_w_values=[]),
                odesign=SimpleNamespace(ExportConvergence=export_convergence),
            )
            return simulation.get_convergence_info("loss")

    @staticmethod
    def _result_with_loss_metrics(metrics):
        result = EmCompletionPolicyTests._valid_result()
        for column, value in metrics.iloc[0].items():
            result.loc[0, column] = value
        return result

    def test_last_row_only_passing_is_not_enough(self):
        metrics = self._extract(self._history([
            "1|100|1.0|5.0|N/A|",
            "2|120|1.0|1.6|0.5|",
            "3|140|1.0|1.0|0.4|",
        ]))

        self.assertEqual(metrics.loc[0, "conv_passes_loss"], 3)
        self.assertEqual(metrics.loc[0, "conv_consecutive_loss"], 1)
        valid, reason = _em_result_validation(
            self._result_with_loss_metrics(metrics)
        )
        self.assertFalse(valid)
        self.assertIn(
            "loss: consecutive converged pass count 1 is below 2", reason
        )

    def test_last_configured_n_rows_passing_is_valid(self):
        metrics = self._extract(self._history([
            "1|100|1.0|5.0|N/A|",
            "2|120|1.0|1.4|1.0|",
            "3|140|1.0|0.8|0.4|",
        ]))

        self.assertEqual(metrics.loc[0, "conv_passes_loss"], 3)
        self.assertEqual(metrics.loc[0, "conv_consecutive_loss"], 2)
        self.assertTrue(_em_result_is_valid(
            self._result_with_loss_metrics(metrics)
        ))

    def test_malformed_or_missing_history_is_invalid(self):
        histories = [
            self._history([
                "1|100|1.0|5.0|N/A|",
                "2|120|1.0|broken|0.5|",
            ]),
            None,
        ]
        for history in histories:
            with self.subTest(history=history):
                metrics = self._extract(history)
                self.assertTrue(math.isnan(
                    metrics.loc[0, "conv_consecutive_loss"]
                ))
                self.assertFalse(_em_result_is_valid(
                    self._result_with_loss_metrics(metrics)
                ))


class ThermalMeshPolicyTests(unittest.TestCase):
    class _Operation:
        def __init__(self, name, update_ok=True):
            self.name = name
            self.props = {}
            self.auto_update = True
            self.update_ok = update_ok
            self.update_calls = 0

        def update(self):
            self.update_calls += 1
            return self.update_ok

    class _Mesh:
        def __init__(self, failing_name=None):
            self.meshoperations = []
            self.calls = []
            self.failing_name = failing_name

        def assign_mesh_level(self, levels, name):
            self.calls.append((dict(levels), name))
            operation_name = f"{name}_L"
            operation = ThermalMeshPolicyTests._Operation(
                operation_name, update_ok=name != self.failing_name
            )
            operation.props.update({"Objects": list(levels), "Level": str(next(iter(levels.values())))})
            self.meshoperations.append(operation)
            return [operation_name]

    @staticmethod
    def _objects():
        obj = lambda name: SimpleNamespace(name=name)
        return {
            "Tx": [obj("tx_0"), obj("tx_1")],
            "wcp_pads": [obj("wcp_pad")],
            "core_pads": [obj("core_pad")],
            "Rx_main_explicit": [obj("rx_main")],
            "Rx_main_blocks": [obj("rx_main_block")],
            "Rx_side_explicit": [obj("rx_side")],
            "Rx_side_blocks": [obj("rx_side_block")],
            "Rx_side2_explicit": [],
            "Rx_side2_blocks": [],
        }

    def test_thin_solids_and_windings_share_the_fluid_mesh_region(self):
        mesh = self._Mesh()

        _assign_thermal_mesh(SimpleNamespace(mesh=mesh), self._objects())

        self.assertEqual(mesh.calls, [
            ({"wcp_pad": 2, "core_pad": 2}, "pad_mesh_level"),
            ({"tx_0": 4, "tx_1": 4}, "tx_mesh_level"),
            ({"rx_main_block": 4}, "rx_main_block_mesh_level"),
            ({"rx_side_block": 5}, "rx_side_block_mesh_level"),
            ({"rx_main": 3, "rx_side": 3}, "rx_mesh_level"),
        ])
        self.assertEqual(len(mesh.meshoperations), 5)
        for operation in mesh.meshoperations:
            self.assertFalse(operation.auto_update)
            self.assertIs(operation.props["Mesh Object(s) Separately Enabled"], False)
            self.assertEqual(operation.update_calls, 1)

    def test_efficiency_ab_can_relax_only_multi_turn_side_blocks(self):
        mesh = self._Mesh()

        _assign_thermal_mesh(
            SimpleNamespace(mesh=mesh), self._objects(), side_block_level=4
        )

        self.assertIn(
            ({"rx_side_block": 4}, "rx_side_block_mesh_level"),
            mesh.calls,
        )

    def test_separate_object_mesh_update_failure_is_fatal(self):
        mesh = self._Mesh(failing_name="tx_mesh_level")

        with self.assertRaisesRegex(RuntimeError, "tx_mesh_level mesh operation update failed"):
            _assign_thermal_mesh(SimpleNamespace(mesh=mesh), self._objects())

    def test_single_rx_turn_gets_pack_local_finest_mesh_level(self):
        mesh = self._Mesh()
        obj = lambda name: SimpleNamespace(name=name)
        objects = {
            "Tx": [], "wcp_pads": [], "core_pads": [],
            "Rx_main_explicit": [], "Rx_main_blocks": [obj("rx_main_block")],
            "Rx_side_explicit": [obj("Rx_side_0_0")], "Rx_side_blocks": [],
            "Rx_side2_explicit": [], "Rx_side2_blocks": [],
        }

        _assign_thermal_mesh(SimpleNamespace(mesh=mesh), objects)

        self.assertEqual(mesh.calls, [
            ({"rx_main_block": 4}, "rx_main_block_mesh_level"),
            ({"Rx_side_0_0": 5}, "rx_side_single_turn_mesh_level"),
        ])
        for operation in mesh.meshoperations:
            self.assertIs(operation.props["Mesh Object(s) Separately Enabled"], False)


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
            "thermal_convergence_available": [1],
            "thermal_converged": [1],
            "thermal_iterations": [151],
            "thermal_extraction_complete": [1],
            "thermal_residual_flow_limit": [1e-3],
            "thermal_residual_energy_limit": [1e-7],
            "thermal_residual_continuity": [8e-4],
            "thermal_residual_x_velocity": [4e-4],
            "thermal_residual_y_velocity": [9e-4],
            "thermal_residual_z_velocity": [4e-4],
            "thermal_residual_energy": [4e-9],
            "thermal_rx_model": ["homogenized_blocks"],
            "thermal_rx_power_balance_ok": [1],
            "thermal_rx_power_balance_group_count": [2],
            "thermal_rx_power_balance_max_abs_w": [0.0],
            "thermal_rx_expected_power_w": [120.0],
            "thermal_rx_assigned_power_w": [120.0],
            "thermal_required_group_mask": [15],
            "T_max_Tx": [80.0],
            "T_max_Rx_main": [81.0],
            "T_max_Rx_side": [82.0],
            "T_max_core": [83.0],
            "Tprobe_Tx_leeward_max": [79.0],
            "Tprobe_Rx_main_leeward_max": [80.0],
            "Tprobe_Rx_side_leeward_max": [81.0],
            "Tprobe_core_center_max": [82.0],
        })
        self.assertTrue(_thermal_result_is_valid(valid))
        false_return_with_native_convergence = valid.copy()
        false_return_with_native_convergence["thermal_analyze_call_ok"] = 0
        false_return_with_native_convergence["thermal_analyze_return_false"] = 1
        false_return_with_native_convergence["thermal_dispatch_status"] = "false"
        self.assertTrue(_thermal_result_is_valid(false_return_with_native_convergence))
        invalid = valid.copy()
        invalid.loc[0, "T_max_core"] = float("nan")
        self.assertFalse(_thermal_result_is_valid(invalid))
        side_optional = valid.copy()
        side_optional.loc[0, "thermal_required_group_mask"] = 11
        side_optional.loc[0, "T_max_Rx_side"] = float("nan")
        self.assertTrue(_thermal_result_is_valid(side_optional))
        no_convergence = valid.drop(columns=["thermal_converged"])
        self.assertFalse(_thermal_result_is_valid(no_convergence))
        divergent = valid.copy()
        divergent.loc[0, "thermal_residual_continuity"] = 2e-3
        self.assertFalse(_thermal_result_is_valid(divergent))
        loose_criteria = valid.copy()
        loose_criteria.loc[0, "thermal_residual_flow_limit"] = 1e-2
        self.assertFalse(_thermal_result_is_valid(loose_criteria))
        zero_iterations = valid.copy()
        zero_iterations["thermal_iterations"] = 0
        self.assertFalse(_thermal_result_is_valid(zero_iterations))
        unbalanced = valid.copy()
        unbalanced["thermal_rx_assigned_power_w"] = 119.0
        self.assertFalse(_thermal_result_is_valid(unbalanced))
        missing_balance = valid.drop(columns=["thermal_rx_power_balance_ok"])
        self.assertFalse(_thermal_result_is_valid(missing_balance))
        saturated = valid.copy()
        saturated.loc[0, "T_max_Rx_main"] = 4726.85
        self.assertFalse(_thermal_result_is_valid(saturated))
        saturated_probe = valid.copy()
        saturated_probe.loc[0, "Tprobe_core_center_max"] = 4726.85
        self.assertFalse(_thermal_result_is_valid(saturated_probe))
        self.assertFalse(_thermal_result_is_valid(pd.DataFrame({"thermal_solved": [0]})))
        self.assertFalse(_thermal_result_is_valid(pd.DataFrame({"other": [1]})))
        self.assertFalse(_thermal_result_is_valid(None))

    def test_new_physics_revision_requires_complete_rx_side_face_contract(self):
        base = pd.DataFrame({
            "physics_data_revision": [PHYSICS_DATA_REVISION],
            "thermal_solved": [1],
            "thermal_convergence_available": [1],
            "thermal_converged": [1],
            "thermal_iterations": [151],
            "thermal_extraction_complete": [1],
            "thermal_residual_flow_limit": [1e-3],
            "thermal_residual_energy_limit": [1e-7],
            "thermal_residual_continuity": [8e-4],
            "thermal_residual_x_velocity": [4e-4],
            "thermal_residual_y_velocity": [9e-4],
            "thermal_residual_z_velocity": [4e-4],
            "thermal_residual_energy": [4e-9],
            "thermal_rx_model": ["homogenized_blocks"],
            "thermal_rx_power_balance_ok": [1],
            "thermal_rx_power_balance_group_count": [2],
            "thermal_rx_power_balance_max_abs_w": [0.0],
            "thermal_rx_expected_power_w": [120.0],
            "thermal_rx_assigned_power_w": [120.0],
            "thermal_required_group_mask": [15],
            "thermal_symmetry": ["eighth"],
            "T_max_Tx": [80.0], "T_max_Rx_main": [81.0],
            "T_max_Rx_side": [82.0], "T_max_core": [83.0],
            "Tprobe_Tx_leeward_max": [79.0],
            "Tprobe_Rx_main_leeward_max": [80.0],
            "Tprobe_Rx_side_leeward_max": [91.0],
            "Tprobe_Rx_side_leeward_mean": [84.0],
            "Tprobe_Rx_side_outer_max": [88.0],
            "Tprobe_Rx_side_outer_mean": [82.0],
            "Tprobe_Rx_side_inner_max": [91.0],
            "Tprobe_Rx_side_inner_mean": [84.0],
            "Tprobe_core_center_max": [82.0],
            "thermal_rx_side_probe_contract_version": [
                RX_SIDE_FACE_PROBE_CONTRACT_VERSION
            ],
            "thermal_rx_side_probe_max_rule": [RX_SIDE_FACE_MAX_RULE],
            "thermal_rx_side_probe_mean_rule": [RX_SIDE_FACE_MEAN_RULE],
            "thermal_rx_side_probe_selected_face": [
                "Tprobe_Rx_side1_inner"
            ],
            "thermal_rx_side_probe_face_count": [2],
        })
        self.assertTrue(_thermal_result_is_valid(base))
        missing_inner = base.drop(columns=["Tprobe_Rx_side_inner_max"])
        self.assertFalse(_thermal_result_is_valid(missing_inner))
        legacy = base.drop(columns=[
            "physics_data_revision", "Tprobe_Rx_side_inner_max",
            "Tprobe_Rx_side_inner_mean", "Tprobe_Rx_side_outer_max",
            "Tprobe_Rx_side_outer_mean", "Tprobe_Rx_side_leeward_mean",
            "thermal_rx_side_probe_contract_version",
            "thermal_rx_side_probe_max_rule",
            "thermal_rx_side_probe_mean_rule",
            "thermal_rx_side_probe_selected_face",
            "thermal_rx_side_probe_face_count",
        ])
        self.assertTrue(_thermal_result_is_valid(legacy))

    def test_thermal_exception_row_preserves_failure_provenance(self):
        frame = _thermal_failure_frame(
            RuntimeError("source assignment failed"),
            core_conductivity={
                "thermal_core_conductivity_model": (
                    "anisotropic_wound_rule_of_mixtures_v1"
                ),
                "thermal_core_k_inplane": 7.68,
                "thermal_core_k_throughstack": 1.1842105263157894,
            },
        )

        self.assertEqual(frame["thermal_solved"].iloc[0], 0)
        self.assertEqual(frame["thermal_extraction_complete"].iloc[0], 0)
        self.assertEqual(frame["thermal_required_group_mask"].iloc[0], 15)
        self.assertEqual(frame["thermal_required_missing_count"].iloc[0], 4)
        self.assertEqual(frame["thermal_error_type"].iloc[0], "RuntimeError")
        self.assertIn("source assignment failed", frame["thermal_error_message"].iloc[0])
        self.assertEqual(
            frame["thermal_core_conductivity_model"].iloc[0],
            "anisotropic_wound_rule_of_mixtures_v1",
        )
        self.assertEqual(frame["thermal_core_k_inplane"].iloc[0], 7.68)

    def test_short_batch_is_not_reported_as_complete(self):
        self.assertEqual(_completion_exit_code(8, 8), 0)
        self.assertEqual(_completion_exit_code(7, 8), 1)
        self.assertEqual(_completion_exit_code(1, 8), 1)
        self.assertEqual(_completion_exit_code(0, 8), 1)


if __name__ == "__main__":
    unittest.main()
