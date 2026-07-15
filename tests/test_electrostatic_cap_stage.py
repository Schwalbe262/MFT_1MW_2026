import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pandas as pd

from module.electrostatic_cap import build_capacitance_timing_payload
from run_simulation_260706 import Simulation


CAP_MATRIX_EXPORT = """\
DesignVariation :
Solution : Setup1 : LastAdaptive
Parameter : CapMatrix
Capacitance Unit: pF

Capacitance
        CapTx CapRx
CapTx   10 -2
CapRx   -2 4

Capacitive Coupling Coefficient
        CapTx CapRx
CapTx   1 -0.316227766
CapRx   -0.316227766 1
"""


def _objects(*names):
    return [SimpleNamespace(name=name) for name in names]


class ElectrostaticStageMockTests(unittest.TestCase):
    def test_eighth_design_copies_solids_and_assigns_nets_ground_and_setup(self):
        source_solver = object()
        source = SimpleNamespace(
            solver_instance=source_solver,
            Tx_windings=_objects("Tx_0", "Tx_1"),
            Rx_windings=_objects("Rx_0"),
            core_objs=_objects("Core_0"),
            core_plates=_objects("CorePlate_0"),
            wcp_plates=_objects("WcpPlate_0"),
            core_pads=_objects("CorePad_0"),
            wcp_pads=_objects("WcpPad_0"),
        )
        geometry_names = [
            "Tx_0", "Tx_1", "Rx_0", "Core_0", "CorePlate_0",
            "WcpPlate_0", "CorePad_0", "WcpPad_0",
        ]
        region = SimpleNamespace(
            name="Region",
            top_face_x="cut_x",
            bottom_face_x="remote_x",
            top_face_y="remote_y",
            bottom_face_y="cut_y",
            top_face_z="remote_z",
            bottom_face_z="cut_z",
        )
        modeler = SimpleNamespace(
            model_units=None,
            object_names=list(geometry_names),
            create_air_region=Mock(return_value=region),
        )
        tx_voltage = SimpleNamespace(name="CapTx")
        rx_voltage = SimpleNamespace(name="CapRx")
        ground_solids = SimpleNamespace(name="CapGroundSolids")
        ground_region = SimpleNamespace(name="CapGroundRegion")
        even_symmetry = SimpleNamespace(name="CapEvenSymmetry")
        matrix = SimpleNamespace(name="CapMatrix")
        setup = SimpleNamespace(name="Setup1")
        cap_design = SimpleNamespace(
            modeler=modeler,
            copy_solid_bodies_from=Mock(return_value=True),
            assign_symmetry=Mock(return_value=even_symmetry),
            assign_voltage=Mock(side_effect=[
                tx_voltage, rx_voltage, ground_solids, ground_region,
            ]),
            assign_matrix=Mock(return_value=matrix),
            create_setup=Mock(return_value=setup),
        )

        simulation = Simulation.__new__(Simulation)
        simulation.design_matrix = source
        simulation.full_model = False
        simulation.input_df = object()
        simulation.df_plus = pd.DataFrame([{
            "cap_max_passes": 7,
            "cap_percent_error": 0.5,
        }])

        def create_design(**_kwargs):
            simulation.design1 = cap_design

        simulation.create_design = Mock(side_effect=create_design)
        with patch("run_simulation_260706.set_design_variables") as set_variables:
            result = simulation.create_capacitance_design()

        self.assertIs(result, cap_design)
        simulation.create_design.assert_called_once_with(
            name="maxwell_cap", solution="Electrostatic"
        )
        set_variables.assert_called_once_with(cap_design, simulation.input_df)
        self.assertEqual(modeler.model_units, "mm")
        cap_design.copy_solid_bodies_from.assert_called_once_with(
            source_solver,
            assignment=geometry_names,
            no_vacuum=False,
            no_pec=False,
            include_sheets=False,
        )
        modeler.create_air_region.assert_called_once_with(
            x_pos=0.0,
            y_pos=200.0,
            z_pos=200.0,
            x_neg=200.0,
            y_neg=0.0,
            z_neg=0.0,
            is_percentage=True,
        )
        cap_design.assign_symmetry.assert_called_once_with(
            assignment=["cut_x", "cut_y", "cut_z"],
            symmetry_name="CapEvenSymmetry",
            is_odd=False,
        )
        self.assertEqual(cap_design.assign_voltage.call_args_list, [
            call(assignment=["Tx_0", "Tx_1"], amplitude="1V", name="CapTx"),
            call(assignment=["Rx_0"], amplitude="0V", name="CapRx"),
            call(
                assignment=["Core_0", "CorePlate_0", "WcpPlate_0"],
                amplitude="0V",
                name="CapGroundSolids",
            ),
            call(
                assignment=["remote_x", "remote_y", "remote_z"],
                amplitude="0V",
                name="CapGroundRegion",
            ),
        ])
        cap_design.assign_matrix.assert_called_once_with(
            assignment=["CapTx", "CapRx"], matrix_name="CapMatrix"
        )
        cap_design.create_setup.assert_called_once_with(
            name="Setup1",
            setup_type="Electrostatic",
            MaximumPasses=7,
            MinimumPasses=1,
            MinimumConvergedPasses=1,
            PercentError=0.5,
            SolveFieldOnly=False,
            SolveMatrixAtLast=True,
        )
        self.assertIs(cap_design.setup, setup)
        self.assertEqual(cap_design.cap_dielectric_names, ["CorePad_0", "WcpPad_0"])
        self.assertEqual(simulation.cap_geometry_copy_count, len(geometry_names))
        self.assertEqual(simulation.cap_region_created_count, 1)
        self.assertEqual(simulation.cap_region_remote_padding_percent, 200.0)
        self.assertEqual(simulation.cap_grounded_solid_count, 3)
        self.assertEqual(simulation.cap_grounded_region_face_count, 3)
        self.assertEqual(simulation.cap_symmetry_face_count, 3)

    def test_native_export_retry_does_not_repeat_solve_and_cleans_temp_files(self):
        exported_paths = []

        def export_c_matrix(**kwargs):
            output_path = Path(kwargs["output_file"])
            exported_paths.append(output_path)
            if len(exported_paths) == 1:
                return False
            output_path.write_text(CAP_MATRIX_EXPORT, encoding="utf-8")
            return True

        analyze = Mock(return_value=None)
        design = SimpleNamespace(
            setup=SimpleNamespace(analyze=analyze),
            export_c_matrix=Mock(side_effect=export_c_matrix),
        )
        simulation = Simulation.__new__(Simulation)
        simulation.design1 = design
        simulation.df1 = pd.DataFrame([{
            "Ltx": 100.0,
            "Lrx": 400.0,
            "Llt": 10.0,
        }])
        simulation.full_model = False
        simulation.NUM_CORE = 4
        simulation.solve_attempts = {}
        simulation.extraction_attempts = {}
        simulation.extraction_backends = {}
        simulation.extraction_units = {}
        simulation.save_project = Mock()
        simulation._log_recent_aedt_messages = Mock()

        with patch("run_simulation_260706.time.sleep") as sleep:
            simulation.analyze_and_extract(
                "cap",
                lambda: simulation.get_capacitance_parameter(
                    max_attempts=2, retry_delay=0.0
                ),
            )

        analyze.assert_called_once_with(cores=4)
        simulation.save_project.assert_called_once_with()
        self.assertEqual(simulation.solve_attempts["cap"], 1)
        self.assertEqual(simulation.extraction_attempts["cap"], 2)
        self.assertEqual(simulation.extraction_backends["cap"], "export_c_matrix")
        self.assertEqual(simulation.extraction_units["cap"], "F")
        sleep.assert_called_once_with(0.0)
        self.assertEqual(len(design.export_c_matrix.call_args_list), 2)
        for invocation in design.export_c_matrix.call_args_list:
            kwargs = invocation.kwargs
            self.assertEqual(kwargs["matrix_name"], "CapMatrix")
            self.assertEqual(kwargs["setup"], "Setup1")
            self.assertEqual(kwargs["default_adaptive"], "LastAdaptive")
            self.assertFalse(kwargs["is_post_processed"])
            self.assertEqual(Path(kwargs["output_file"]).suffix, ".txt")
        self.assertEqual(len(exported_paths), 2)
        self.assertTrue(all(not os.path.exists(path) for path in exported_paths))
        self.assertAlmostEqual(simulation.df_cap["C_tx_tx_F"].iloc[0], 80e-12)
        self.assertAlmostEqual(simulation.df_cap["C_tx_rx_F"].iloc[0], 16e-12)
        self.assertEqual(
            simulation.df_cap["cap_capacitance_restoration_factor"].iloc[0],
            8.0,
        )
        self.assertGreaterEqual(simulation.stage_timings["stage_time_cap_extract_s"], 0.0)
        timing = pd.DataFrame([build_capacitance_timing_payload(
            simulation.stage_timings["stage_time_cap_solve_s"],
            simulation.stage_timings["stage_time_cap_extract_s"],
            simulation.stage_timings["stage_time_cap_analyze_total_s"],
        )])
        combined = pd.concat(
            [simulation.df_cap, timing, simulation.get_execution_telemetry()],
            axis=1,
        )
        self.assertTrue(combined.columns.is_unique)

    def test_pooled_export_retries_in_exact_shared_results_root(self):
        exported_paths = []

        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory) / "lease-workspace"
            project_name = "mft-pooled-cap"
            project_path = workspace / project_name
            project_path.mkdir(parents=True)

            def export_c_matrix(**kwargs):
                output_path = Path(kwargs["output_file"])
                exported_paths.append(output_path)
                if len(exported_paths) == 1:
                    return False
                output_path.write_text(CAP_MATRIX_EXPORT, encoding="utf-8")
                return True

            simulation = Simulation.__new__(Simulation)
            simulation.aedt_backend = "pooled"
            simulation.aedt_lease = SimpleNamespace(
                workspace_path=str(workspace)
            )
            simulation.PROJECT_NAME = project_name
            simulation.project_path = str(project_path)
            simulation.design1 = SimpleNamespace(
                export_c_matrix=Mock(side_effect=export_c_matrix)
            )
            simulation.df1 = pd.DataFrame([{
                "Ltx": 100.0,
                "Lrx": 400.0,
                "Llt": 10.0,
            }])
            simulation.full_model = False
            simulation.extraction_attempts = {}
            simulation.extraction_backends = {}
            simulation.extraction_units = {}
            simulation._prepare_pooled_solution_data_app = Mock()

            with patch("run_simulation_260706.time.sleep") as sleep:
                result = simulation.get_capacitance_parameter(
                    max_attempts=2, retry_delay=0.0
                )

            results_root = (
                project_path / f"{project_name}.aedtresults"
            ).resolve()
            self.assertEqual(len(exported_paths), 2)
            self.assertEqual(
                {path.resolve().parent for path in exported_paths},
                {results_root},
            )
            self.assertTrue(all(
                path.name.startswith("mft_cap_export_")
                and path.suffix == ".txt"
                for path in exported_paths
            ))
            self.assertEqual(len(set(exported_paths)), 2)
            self.assertTrue(all(not path.exists() for path in exported_paths))
            self.assertAlmostEqual(result["C_tx_tx_F"].iloc[0], 80e-12)
            self.assertEqual(
                simulation.extraction_backends["cap"], "export_c_matrix"
            )
            sleep.assert_called_once_with(0.0)

    def test_pooled_export_rejects_non_regular_output_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory) / "lease-workspace"
            project_name = "mft-pooled-cap"
            project_path = workspace / project_name
            project_path.mkdir(parents=True)
            exported_paths = []

            def export_c_matrix(**kwargs):
                output_path = Path(kwargs["output_file"])
                exported_paths.append(output_path)
                output_path.mkdir()
                return True

            simulation = Simulation.__new__(Simulation)
            simulation.aedt_backend = "pooled"
            simulation.aedt_lease = SimpleNamespace(
                workspace_path=str(workspace)
            )
            simulation.PROJECT_NAME = project_name
            simulation.project_path = str(project_path)
            simulation.design1 = SimpleNamespace(
                export_c_matrix=Mock(side_effect=export_c_matrix)
            )
            simulation.df1 = pd.DataFrame([{
                "Ltx": 100.0,
                "Lrx": 400.0,
                "Llt": 10.0,
            }])
            simulation.full_model = False
            simulation.extraction_attempts = {}
            simulation.extraction_backends = {}
            simulation.extraction_units = {}
            simulation._prepare_pooled_solution_data_app = Mock()

            with self.assertRaisesRegex(
                    RuntimeError, "not a plain file"):
                simulation.get_capacitance_parameter(
                    max_attempts=1, retry_delay=0.0
                )

            self.assertEqual(len(exported_paths), 1)
            self.assertTrue(exported_paths[0].is_dir())
            self.assertNotIn("cap", simulation.extraction_backends)


if __name__ == "__main__":
    unittest.main()
