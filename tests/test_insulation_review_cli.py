"""Exercise the copied-project CLI without launching AEDT."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import json
import math
import tempfile
import unittest

from tools.mft_insulation_review import (
    SOLUTION, boundary_contract, copy_project_snapshot, extract, main,
    maximum_location_mm, voltage_peak_v,
)


class FakeBoundary:
    def __init__(self, name, objects=None, voltage="0V", faces=None):
        self.name = name
        self.type = "Voltage"
        self.props = {"Voltage": voltage}
        if objects is not None:
            self.props["Objects"] = objects
        if faces is not None:
            self.props["Faces"] = faces
        self.update = Mock(return_value=True)


class FakeReporter:
    def __init__(self):
        self.commands = []
        self.axis = None

    def CalcStack(self, action):
        self.commands.append(("CalcStack", action))

    def CopyNamedExprToStack(self, name):
        self.commands.append(("CopyNamedExprToStack", name))

    def EnterVol(self, name):
        self.commands.append(("EnterVol", name))

    def CalcOp(self, name):
        self.commands.append(("CalcOp", name))
        if name.startswith("Scalar"):
            self.axis = name[-1]

    def CalculatorWrite(self, filename, solution, variation):
        self.commands.append(("CalculatorWrite", solution, variation))
        Path(filename).write_text("Native calculator scalar\n" + {"X": "0.003", "Y": "-0.004", "Z": "0.005"}[self.axis] + "\n")


class FakeApp:
    def __init__(self, project, **kwargs):
        self.project_file = project
        self.design_name = kwargs["design"]
        self.solution_type = "Electrostatic"
        self.aedt_version_id = "2025.2"
        self.boundaries = [
            FakeBoundary("CapTx", ["Tx_main_0_0", "Tx_main_1_0"]),
            FakeBoundary("CapRx", ["Rx_main_0_0"], "28*sqrt(2) kV"),
            FakeBoundary("CapGroundSolids", ["Core", "Plate"]),
            FakeBoundary("CapGroundRegion", faces=[1, 2, 3]),
        ]
        solids = {
            name: SimpleNamespace(name=name, material_name=material)
            for name, material in (
                ("Tx_main_0_0", "copper"), ("Tx_main_1_0", "copper"),
                ("Rx_main_0_0", "copper"), ("Core", "core"),
                ("Plate", "aluminum"), ("Region", "air"), ("Pad", "thermal_pad"),
            )
        }

        class Modeler(dict):
            @property
            def object_names(self):
                return list(self)

        self.modeler = Modeler(solids)
        self.materials = {
            name: SimpleNamespace(conductivity=SimpleNamespace(value=value))
            for name, value in (("copper", "58000000"), ("aluminum", "38000000"),
                                ("core", "0"), ("air", "0"), ("thermal_pad", "0"))
        }
        self.reporter = FakeReporter()
        self.post = SimpleNamespace(ofieldsreporter=self.reporter,
                                    get_scalar_field_value=Mock(side_effect=self.scalar))
        self.available_variations = SimpleNamespace(get_independent_nominal_values=lambda: {"span": "20mm", "memo": "1"})
        self.variable_manager = SimpleNamespace(variables={"span": SimpleNamespace(sweep=True),
                                                           "memo": SimpleNamespace(sweep=False)})
        self.setup = SimpleNamespace(props={"MaximumPasses": 5, "SolveFieldOnly": False},
                                     update=Mock(return_value=True), analyze=Mock(return_value=None))
        self.release_desktop = Mock()
        self.save_project = Mock(side_effect=lambda: Path(project).write_text("modified copied project"))

    def get_setup(self, name):
        assert name == "Setup1"
        return self.setup

    def scalar(self, quantity, **kwargs):
        name = kwargs["object_name"]
        if quantity == "Voltage":
            for boundary in self.boundaries:
                if name in boundary.props.get("Objects", []):
                    return voltage_peak_v(boundary.props["Voltage"])
        if quantity == "Mag_E":
            amplitude = max(abs(voltage_peak_v(boundary.props["Voltage"])) for boundary in self.boundaries)
            return amplitude * (2.0 if name == "Region" else 0.5)
        raise RuntimeError("unknown native quantity")


class InsulationReviewCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.aedt"
        self.source.write_bytes(b"original native project")
        self.results = self.source.with_suffix(".aedtresults")
        self.results.mkdir()
        (self.results / "fields.bin").write_bytes(b"original solved fields")

    def arguments(self, *, case="secondary", rerun=False):
        return SimpleNamespace(project=self.source, design="maxwell_cap", case=case,
                               output_dir=self.root / "review", rerun=rerun)

    def factory(self, **kwargs):
        self.assertNotEqual(Path(kwargs["project"]), self.source)
        self.assertEqual(Path(kwargs["project"]).read_bytes(), self.source.read_bytes())
        self.assertEqual(Path(kwargs["project"]).with_suffix(".aedtresults").joinpath("fields.bin").read_bytes(), b"original solved fields")
        self.assertTrue(kwargs["non_graphical"])
        self.assertTrue(kwargs["new_desktop"])
        self.app = FakeApp(**kwargs)
        return self.app

    def test_numeric_voltage_parser_handles_actual_saved_kv_expression(self):
        self.assertAlmostEqual(voltage_peak_v("28*sqrt(2) kV"), 28000 * math.sqrt(2))
        self.assertEqual(voltage_peak_v("-(3+1)/2 V"), -2)
        self.assertEqual(voltage_peak_v("0V"), 0)
        for expression in ("Vtest V", "__import__('os') V", "math.sqrt(2)V", "2**3V",
                           "1/0V", "sqrt(-1)V", "1e309V", "1+V", "28kA"):
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    voltage_peak_v(expression)

    @patch("tools.mft_insulation_review.importlib.metadata.version", return_value="0.22.0")
    def test_actual_saved_voltage_normalizes_e_and_remains_unqualified(self, _version):
        payload = extract(self.arguments(), app_factory=self.factory)
        test = payload["tests"]["secondaryHV_primaryHWground"]
        air = next(item for item in test["hotspots"] if item["object_name"] == "Region")
        self.assertEqual(air["E_1V_V_per_m"], 2)
        self.assertAlmostEqual(air["E_actual_V_per_m"], 56000 * math.sqrt(2))
        self.assertEqual(air["location_mm"], [3.0, -4.0, 5.0])
        assessment = json.loads((self.root / "review" / "insulation_assessment.json").read_text())
        self.assertEqual(assessment["status"], "insufficient_data")
        self.assertEqual(self.source.read_bytes(), b"original native project")
        self.assertEqual((self.results / "fields.bin").read_bytes(), b"original solved fields")
        self.app.setup.analyze.assert_not_called()
        self.app.release_desktop.assert_called_once_with(close_projects=True, close_desktop=True)

    @patch("tools.mft_insulation_review.importlib.metadata.version", return_value="0.22.0")
    def test_rerun_modifies_only_copy_and_sets_exact_primary_test(self, _version):
        payload = extract(self.arguments(case="primary", rerun=True), app_factory=self.factory)
        self.assertEqual(self.app.boundaries[0].props["Voltage"], "1V")
        self.assertEqual(self.app.boundaries[1].props["Voltage"], "0V")
        self.app.setup.analyze.assert_called_once_with(cores=4)
        self.assertEqual(self.app.setup.props["MaximumPasses"], 10)
        self.assertEqual(self.app.setup.props["MinimumPasses"], 2)
        self.assertTrue(self.app.setup.props["SolveFieldOnly"])
        self.assertFalse(self.app.setup.props["SolveMatrixAtLast"])
        self.assertEqual(payload["provenance"]["original_boundaries"]["CapRx"]["Voltage"], "28*sqrt(2) kV")
        self.assertFalse(payload["convergence"]["field_local_max_converged"])
        self.assertEqual(self.source.read_bytes(), b"original native project")
        self.assertEqual((self.results / "fields.bin").read_bytes(), b"original solved fields")

    @patch("tools.mft_insulation_review.importlib.metadata.version", return_value="0.22.0")
    def test_mismatched_case_requires_rerun_and_still_closes_owned_desktop(self, _version):
        with self.assertRaisesRegex(ValueError, "selected case"):
            extract(self.arguments(case="primary"), app_factory=self.factory)
        self.assertTrue((self.root / "review" / "extraction_failure.json").is_file())
        self.app.release_desktop.assert_called_once_with(close_projects=True, close_desktop=True)
        self.assertEqual(self.source.read_bytes(), b"original native project")

    @patch("tools.mft_insulation_review.importlib.metadata.version", return_value="0.22.0")
    def test_stale_solved_voltage_rejected_before_field_normalization(self, _version):
        def stale_factory(**kwargs):
            app = self.factory(**kwargs)
            app.post.get_scalar_field_value = Mock(return_value=1.0)
            return app
        with self.assertRaisesRegex(ValueError, "solved potential"):
            extract(self.arguments(), app_factory=stale_factory)
        self.app.release_desktop.assert_called_once()

    @patch("tools.mft_insulation_review.importlib.metadata.version", return_value="0.22.0")
    def test_unavailable_voltage_quantity_requires_fresh_rerun(self, _version):
        def unsupported_factory(**kwargs):
            app = self.factory(**kwargs)
            original = app.scalar
            app.post.get_scalar_field_value = Mock(
                side_effect=lambda quantity, **options: False if quantity == "Voltage" else original(quantity, **options),
            )
            return app

        with self.assertRaisesRegex(ValueError, "use --rerun"):
            extract(self.arguments(), app_factory=unsupported_factory)
        fresh = self.arguments(rerun=True)
        fresh.output_dir = self.root / "fresh_review"
        payload = extract(fresh, app_factory=unsupported_factory)
        self.assertEqual(payload["provenance"]["potential_readback"]["status"], "unavailable")
        self.app.setup.analyze.assert_called_once_with(cores=4)
        self.assertFalse(payload["convergence"]["field_local_max_converged"])

    @patch("tools.mft_insulation_review.importlib.metadata.version", return_value="0.22.0")
    def test_nonfinite_native_field_is_rejected_and_source_is_preserved(self, _version):
        def invalid_factory(**kwargs):
            app = self.factory(**kwargs)
            original = app.scalar
            app.post.get_scalar_field_value = Mock(
                side_effect=lambda quantity, **options: float("inf") if quantity == "Mag_E" else original(quantity, **options),
            )
            return app

        with self.assertRaisesRegex(ValueError, "invalid native E maximum"):
            extract(self.arguments(), app_factory=invalid_factory)
        self.assertEqual(self.source.read_bytes(), b"original native project")
        self.app.release_desktop.assert_called_once()

    def test_shorting_and_hardware_coverage_are_required(self):
        app = FakeApp(str(self.source), design="maxwell_cap")
        app.boundaries[1].props["Objects"].append("Tx_main_0_0")
        with self.assertRaisesRegex(ValueError, "disjoint"):
            boundary_contract(app)
        app.boundaries[1].props["Objects"] = ["Rx_main_0_0"]
        app.modeler["Bolt"] = SimpleNamespace(name="Bolt", material_name="aluminum")
        with self.assertRaisesRegex(ValueError, "not grounded: Bolt"):
            boundary_contract(app)

    def test_native_integer_ids_and_winding_prefixed_pads_and_plates(self):
        app = FakeApp(str(self.source), design="maxwell_cap")
        app.modeler["Tx_main_wcp_pad_1_in_p"] = SimpleNamespace(
            name="Tx_main_wcp_pad_1_in_p", material_name="thermal_pad",
        )
        app.modeler["Tx_main_wcp_1_p"] = SimpleNamespace(
            name="Tx_main_wcp_1_p", material_name="aluminum",
        )
        app.boundaries[2].props["Objects"].append("Tx_main_wcp_1_p")
        app.modeler.objects = {index: obj for index, obj in enumerate(app.modeler.values(), 100)}
        object_ids = {obj.name: index for index, obj in app.modeler.objects.items()}
        for boundary in app.boundaries[:3]:
            boundary.props["Objects"] = [object_ids[name] for name in boundary.props["Objects"]]
        _, groups, dielectric = boundary_contract(app)
        self.assertEqual(groups["CapTx"], {"Tx_main_0_0", "Tx_main_1_0"})
        self.assertIn("Tx_main_wcp_1_p", groups["CapGroundSolids"])
        self.assertIn("Tx_main_wcp_pad_1_in_p", {obj.name for obj in dielectric})

    def test_snapshot_rejects_overwrite_and_recursive_results_destination(self):
        copy_project_snapshot(self.source, self.root / "review")
        with self.assertRaises(FileExistsError):
            copy_project_snapshot(self.source, self.root / "review")
        with self.assertRaisesRegex(ValueError, "outside source"):
            copy_project_snapshot(self.source, self.results / "nested")
        self.assertEqual(self.source.read_bytes(), b"original native project")

    def test_manual_maxpos_exports_mks_coordinates_and_sweep_variables(self):
        app = FakeApp(str(self.source), design="maxwell_cap")
        self.assertEqual(maximum_location_mm(app, "Region", self.root), [3, -4, 5])
        self.assertIn(("CalcOp", "VolumeValue"), app.reporter.commands)
        self.assertIn(("CalcOp", "MaxPos"), app.reporter.commands)
        self.assertIn(("CalculatorWrite", ["Solution:=", SOLUTION], ["span:=", "20mm"]), app.reporter.commands)

    def test_offline_assessment_requires_no_solver_and_reports_missing_evidence(self):
        evidence = self.root / "evidence.json"
        evidence.write_text("{}")
        output = self.root / "assessment.json"
        self.assertEqual(main(["assess", "--input", str(evidence), "--output", str(output)]), 0)
        self.assertEqual(json.loads(output.read_text())["status"], "insufficient_data")


if __name__ == "__main__":
    unittest.main()
