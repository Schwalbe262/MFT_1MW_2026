import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from module.insulation_assessment import TEST_NAMES, assess_insulation


def complete_payload():
    payload = {
        "model": {
            "source_geometry_kind": "actual_cad", "cad_identity": "vendor-r2-sha256",
            "target_cad_identity": "vendor-r2-sha256", "verified_final_cad": True,
            "dielectric_complete": True, "conductive_hardware_complete": True,
            "material_assignments_verified": True, "model_basis": "full",
            "geometry_factor": 1,
        },
        "provenance": {"project": "native.aedt", "design": "Insulation"},
        "capacitance_solve": {"success": True, "source": "matrix.txt"},
        "materials": [{
            "name": "paper", "allowable_peak_field_V_per_m": 1e6,
            "limit_evidence": {"source": "material-qualified-ac-limit.json",
                               "applicable_to_ac_withstand": True},
        }, {"name": "copper", "is_conductor": True}],
        "convergence": {"field_local_max_converged": True, "source": "mesh-history.json",
                        "field_max_relative_change": 0.01, "field_max_relative_tolerance": 0.02},
        "test_conditions": {"hold_time_s": 60, "frequency_Hz": 60},
        "pd_test_conditions": {"voltage_rms_V": 3000, "hold_time_s": 10, "frequency_Hz": 60},
        "tests": {}, "pd_measurements": [], "withstand_measurements": [],
    }
    for name, winding in zip(TEST_NAMES, ("primary", "secondary")):
        boundary = {"terminals_within_each_winding_shorted": True,
                    "primary_secondary_connected": False, "other_winding_grounded": True,
                    "all_core_cooling_hardware_grounded": True, "excited_winding": winding}
        payload["tests"][name] = {
            "native_field_evidence": {"source": "native-E-export.csv", "solution": "Setup1",
                                      "normalization_voltage_V": 1, "unit": "V/m"},
            "boundary_conditions": boundary, "dielectric_regions_covered": True,
            "hotspots": [{"object_name": "Paper1", "material": "paper",
                          "E_1V_V_per_m": 2, "location_mm": [-1, 2, 3]},
                         {"object_name": "Copper1", "material": "copper",
                          "E_1V_V_per_m": 0, "location_mm": [0, 0, 0]}],
            "electrostatic_energy_1V_J": 1e-10,
        }
        common = {"test_name": name, "source": "lab-report.pdf",
                  "specimen_cad_identity": "vendor-r2-sha256", "hold_time_s": 60,
                  "frequency_Hz": 60, "boundary_conditions": copy.deepcopy(boundary),
                  "calibration": {"source": "calibration.pdf", "date": "2026-08-01", "charge_pC": 10}}
        payload["withstand_measurements"].append({**copy.deepcopy(common),
                                                 "kind": "measured_ac_withstand",
                                                 "applied_voltage_rms_V": 20000, "breakdown": False})
        payload["pd_measurements"].append({**copy.deepcopy(common),
                                          "kind": "measured_apparent_charge",
                                          "applied_voltage_rms_V": 3000, "apparent_charge_pC": 15})
    return payload


class InsulationAssessmentTest(unittest.TestCase):
    def test_eighth_restores_energy_but_never_multiplies_local_field_by_eight(self):
        full = assess_insulation(complete_payload())
        payload = complete_payload()
        payload["model"].update(model_basis="eighth", geometry_factor=8,
                                electric_potential_symmetry_verified=True)
        eighth = assess_insulation(payload)
        for name in TEST_NAMES:
            f = full["tests"][name]["electric_field_assessment"]
            e = eighth["tests"][name]["electric_field_assessment"]
            self.assertAlmostEqual(f["maximum_E_peak_20kV_V_per_m"], 2 * 20000 * math.sqrt(2))
            self.assertEqual(f["maximum_E_peak_20kV_V_per_m"], e["maximum_E_peak_20kV_V_per_m"])
            self.assertEqual(e["local_E_geometry_factor"], 1)
            self.assertAlmostEqual(e["electrostatic_energy_full_at_voltage_peak_J"],
                                   8 * f["electrostatic_energy_full_at_voltage_peak_J"])
            self.assertEqual(len(e["hotspots"]), 1)  # Conductors need no dielectric strength.

    def test_native_higher_voltage_can_be_explicitly_normalized_before_assessment(self):
        payload = complete_payload()
        native = payload["tests"][TEST_NAMES[1]]["native_field_evidence"]
        native.update(solver_excitation_voltage_peak_V=28000 * math.sqrt(2),
                      normalization_method="divide_native_E_by_solver_excitation_peak")
        result = assess_insulation(payload)
        self.assertAlmostEqual(result["tests"][TEST_NAMES[1]]["electric_field_assessment"]
                               ["maximum_E_peak_20kV_V_per_m"], 40000 * math.sqrt(2))

    def test_cap_success_and_within_limit_fields_do_not_create_actual_test_pass(self):
        payload = complete_payload()
        payload.pop("withstand_measurements")
        payload.pop("pd_measurements")
        result = assess_insulation(payload)
        self.assertEqual(result["capacitance_solve"]["status"], "success")
        self.assertEqual(result["status"], "insufficient_data")
        for test in result["tests"].values():
            self.assertEqual(test["electric_field_assessment"]["status"], "within_limit")
            self.assertEqual(test["actual_20kV_withstand_qualification"]["status"], "insufficient_data")
            self.assertEqual(test["pd_apparent_charge_assessment"]["status"], "insufficient_data")

    def test_actual_cad_mismatch_and_omitted_paper_keep_diagnostic_E_but_block_assessment(self):
        payload = complete_payload()
        payload["model"].update(source_geometry_kind="parametric", cad_identity="pre-vendor-CAD",
                                verified_final_cad=False, dielectric_complete=False)
        result = assess_insulation(payload)
        field = result["tests"][TEST_NAMES[0]]["electric_field_assessment"]
        self.assertEqual(field["status"], "insufficient_data")
        self.assertIn("source_cad_identity_does_not_match_target", field["reasons"])
        self.assertIn("dielectric_complete_not_verified", field["reasons"])
        self.assertGreater(field["maximum_E_peak_20kV_V_per_m"], 0)

    def test_material_strength_is_not_invented_from_permittivity_or_capacitance(self):
        payload = complete_payload()
        payload["materials"][0] = {"name": "paper", "permittivity": 3.5}
        field = assess_insulation(payload)["tests"][TEST_NAMES[0]]["electric_field_assessment"]
        self.assertEqual(field["status"], "insufficient_data")
        self.assertIsNone(field["hotspots"][0]["allowable_peak_field_V_per_m"])
        self.assertIsNone(field["hotspots"][0]["field_margin"])

    def test_capacitance_or_energy_convergence_cannot_certify_local_E_convergence(self):
        payload = complete_payload()
        payload["convergence"] = {"capacitance_converged": True, "energy_relative_change": 0.001,
                                  "source": "cap-log.txt", "field_local_max_converged": False}
        field = assess_insulation(payload)["tests"][TEST_NAMES[0]]["electric_field_assessment"]
        self.assertEqual(field["status"], "insufficient_data")
        self.assertIn("local_field_maximum_convergence_not_verified", field["reasons"])

    def test_wrong_boundary_and_unverified_eighth_symmetry_block_complete_assessment(self):
        payload = complete_payload()
        payload["model"].update(model_basis="eighth", geometry_factor=8)
        payload["tests"][TEST_NAMES[0]]["boundary_conditions"]["primary_secondary_connected"] = True
        field = assess_insulation(payload)["tests"][TEST_NAMES[0]]["electric_field_assessment"]
        self.assertIn("electric_potential_symmetry_not_verified", field["reasons"])
        self.assertIn("primary_secondary_separation_not_verified", field["reasons"])

    def test_complete_field_evidence_reports_excess_without_claiming_withstand_failure(self):
        payload = complete_payload()
        payload["tests"][TEST_NAMES[0]]["hotspots"][0]["E_1V_V_per_m"] = 100
        result = assess_insulation(payload)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["tests"][TEST_NAMES[0]]["electric_field_assessment"]["status"], "exceeds_limit")
        self.assertEqual(result["tests"][TEST_NAMES[0]]["actual_20kV_withstand_qualification"]["status"], "pass")

    def test_complete_measurements_qualify_only_their_recorded_requested_conditions(self):
        result = assess_insulation(complete_payload())
        self.assertEqual(result["status"], "qualified_by_measurement_and_supported_by_field_assessment")
        self.assertEqual(result["test_conditions"]["voltage_rms_V"], 20000)
        for test in result["tests"].values():
            self.assertEqual(test["actual_20kV_withstand_qualification"]["status"], "pass")
            self.assertEqual(test["pd_apparent_charge_assessment"]["status"], "pass")
            self.assertEqual(test["pd_apparent_charge_assessment"]["required_conditions"]["voltage_rms_V"], 3000)

    def test_PD_above_15pC_and_observed_breakdown_are_independent_failures(self):
        payload = complete_payload()
        payload["pd_measurements"][0]["apparent_charge_pC"] = 15.01
        payload["withstand_measurements"][1]["breakdown"] = True
        result = assess_insulation(payload)
        self.assertEqual(result["tests"][TEST_NAMES[0]]["pd_apparent_charge_assessment"]["status"], "fail")
        self.assertEqual(result["tests"][TEST_NAMES[1]]["actual_20kV_withstand_qualification"]["status"], "fail")

    def test_unknown_PD_voltage_is_not_assumed_to_be_20kV(self):
        payload = complete_payload()
        payload.pop("pd_test_conditions")
        payload["pd_measurements"][0]["applied_voltage_rms_V"] = 20000
        pd = assess_insulation(payload)["tests"][TEST_NAMES[0]]["pd_apparent_charge_assessment"]
        self.assertEqual(pd["status"], "insufficient_data")
        self.assertIn("pd_required_voltage_not_specified", pd["reasons"])
        self.assertNotIn("voltage_rms_V", pd["required_conditions"])

    def test_calculated_C_times_V_and_missing_PD_calibration_never_pass(self):
        for update in ({"kind": "calculated_C_times_V"}, {"calibration": {}},
                       {"applied_voltage_rms_V": None}, {"hold_time_s": None}):
            with self.subTest(update=update):
                payload = complete_payload()
                payload["pd_measurements"][0].update(update)
                pd = assess_insulation(payload)["tests"][TEST_NAMES[0]]["pd_apparent_charge_assessment"]
                self.assertEqual(pd["status"], "insufficient_data")

    def test_missing_locations_and_wrong_normalization_are_explicit(self):
        payload = complete_payload()
        payload["tests"][TEST_NAMES[0]]["hotspots"][0]["location_mm"] = None
        payload["tests"][TEST_NAMES[1]]["native_field_evidence"]["normalization_voltage_V"] = 20000
        result = assess_insulation(payload)
        first = result["tests"][TEST_NAMES[0]]["electric_field_assessment"]
        second = result["tests"][TEST_NAMES[1]]["electric_field_assessment"]
        self.assertGreater(first["maximum_E_peak_20kV_V_per_m"], 0)
        self.assertIn("hotspot_location_missing_or_invalid", first["reasons"])
        self.assertIsNone(second["maximum_E_peak_20kV_V_per_m"])
        self.assertIn("native_field_not_normalized_to_1V", second["reasons"])

    def test_sparse_input_always_emits_both_tests_and_strict_JSON(self):
        result = assess_insulation({})
        self.assertEqual(set(result["tests"]), set(TEST_NAMES))
        self.assertEqual(result["status"], "insufficient_data")
        json.dumps(result, allow_nan=False)

    def test_input_mapping_json_and_path_agree_without_mutating_input(self):
        payload = complete_payload()
        original = copy.deepcopy(payload)
        expected = assess_insulation(payload)
        self.assertEqual(payload, original)
        self.assertEqual(assess_insulation(json.dumps(payload)), expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(assess_insulation(path), expected)

    def test_nonfinite_input_and_non20kV_withstand_request_are_rejected(self):
        with self.assertRaises(ValueError):
            assess_insulation({"model": {"geometry_factor": float("nan")}})
        with self.assertRaises(ValueError):
            assess_insulation({"test_conditions": {"voltage_rms_V": 28000}})


if __name__ == "__main__":
    unittest.main()
