"""Independent sphere electrostatics and exact AC crossing regressions."""

from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest

from module.pd_void_model import (
    EPSILON_0_F_PER_M, EventLimitExceeded,
    air_streamer_inception_field_V_per_m, estimate_pd_void,
    estimate_pd_void_batch, first_cycle_events, sphere_apparent_charge_C,
)
from tools.mft_pd_void_estimate import main


def inputs():
    return {
        "field_evidence": {
            "source": "synthetic uniform host field for analytic regression",
            "solution": "1 V uniform-field solution", "object_name": "HostBox",
            "material_name": "epsr4", "location_mm": [0, 0, 0],
            "point_in_dielectric": True, "model_basis": "full", "unit": "V/m",
            "normalization_voltage_V": 1, "local_field_at_1V_V_per_m": 1000,
            "local_field_converged": False,
        },
        "void": {
            "radius_m": 0.0001, "minimum_host_boundary_distance_m": 0.001,
            "containment_evidence": "synthetic box center 1 mm from boundary",
            "relative_permittivity_host": 4, "relative_permittivity_gas": 1,
            "pressure_Pa": 101325, "extinction_ratio": 0.5,
        },
        "voltage_scenarios_rms_V": [5000, 10000, 15000, 20000],
        "frequency_Hz": 60,
    }


class PdVoidModelTests(unittest.TestCase):
    def test_surface_charge_weighting_integral_matches_induced_charge(self):
        # Integrate -sigma*phi_weight over the sphere. Its net free charge is
        # zero; phi_weight's additive constant therefore contributes nothing.
        radius, epsh, epsg, drop, weighting = 1e-4, 4, 1, 2e6, 700
        k = 3 * epsh / (epsg + 2 * epsh)
        sigma = EPSILON_0_F_PER_M * (epsg + 2 * epsh) * drop
        steps, integral = 20000, 0.0
        for index in range(steps):
            u = -1 + (index + 0.5) * 2 / steps
            free_surface_charge = sigma * u
            phi_weight = -k * weighting * radius * u
            area_element = 2 * math.pi * radius ** 2 * 2 / steps
            integral -= free_surface_charge * phi_weight * area_element
        actual = sphere_apparent_charge_C(radius, epsh, drop, weighting)
        self.assertAlmostEqual(actual / integral, 1, places=8)

    def test_independently_reviewed_air_sphere_numeric_example(self):
        result = estimate_pd_void(inputs())
        row = result["radius_cases"][0]
        self.assertAlmostEqual(row["inception_field_V_per_m"], 7136499.012279391, places=7)
        self.assertAlmostEqual(row["conditional_virgin_PDIV_rms_V"], 3784.700134135392, places=8)
        self.assertAlmostEqual(row["threshold_pulse_q_app_pC_if_event"], 1.5880852043432678, places=12)
        self.assertEqual(result["physical_qualification"], "insufficient_data")

    def test_radius_cubed_drop_and_weighting_scaling(self):
        baseline = sphere_apparent_charge_C(1e-4, 4, 1e6, 1000)
        self.assertAlmostEqual(sphere_apparent_charge_C(2e-4, 4, 1e6, 1000) / baseline, 8)
        self.assertAlmostEqual(sphere_apparent_charge_C(1e-4, 4, 2e6, 1000) / baseline, 2)
        self.assertAlmostEqual(sphere_apparent_charge_C(1e-4, 4, 1e6, 3000) / baseline, 3)

    def test_terminal_coupling_times_local_void_voltage_not_global_cv(self):
        payload = inputs()
        payload["void"]["radius_m"] = 0.0003
        payload["field_evidence"]["local_field_at_1V_V_per_m"] = 500
        row = estimate_pd_void(payload)["radius_cases"][0]
        self.assertAlmostEqual(
            row["equivalent_terminal_coupling_capacitance_F"] * row["cavity_voltage_drop_V"] * 1e12,
            row["threshold_pulse_q_app_pC_if_event"], places=11,
        )
        self.assertAlmostEqual(row["equivalent_terminal_coupling_capacitance_pF"], 0.0100138504976, places=9)
        before = deepcopy(row["scenarios"])
        payload["global_transformer_capacitance_F"] = 1000.0
        self.assertEqual(estimate_pd_void(payload)["radius_cases"][0]["scenarios"], before)

    def test_no_event_is_conditional_not_pd_free(self):
        payload = inputs()
        payload["field_evidence"]["local_field_at_1V_V_per_m"] = 6.123568585
        result = estimate_pd_void(payload)
        for scenario in result["radius_cases"][0]["scenarios"]:
            self.assertEqual(scenario["q_peak_pC"], 0)
            self.assertEqual(scenario["event_count"], 0)
            self.assertEqual(scenario["event_status"], "no_events_under_assumptions")
            self.assertTrue(scenario["does_not_establish_pd_free"])
            self.assertEqual(scenario["physical_qualification"], "insufficient_data")

    def test_exact_threshold_phase_rms_conversion_and_charge_memory(self):
        payload = inputs()
        payload["void"]["inception_field_V_per_m"] = 1e6
        payload["voltage_scenarios_rms_V"] = [1200]
        result = estimate_pd_void(payload)
        scenario = result["radius_cases"][0]["scenarios"][0]
        amplitude = (4 / 3) * 1000 * 1200 * math.sqrt(2)
        first = scenario["events"][0]
        self.assertAlmostEqual(scenario["voltage_peak_V"], 1200 * math.sqrt(2))
        self.assertAlmostEqual(first["phase_rad"], math.asin(1e6 / amplitude))
        self.assertAlmostEqual(first["time_s"], first["phase_rad"] / (2 * math.pi * 60))
        for event in scenario["events"]:
            polarity = event["polarity"]
            self.assertAlmostEqual(event["total_void_field_before_V_per_m"] / 1e6, polarity)
            self.assertAlmostEqual(event["total_void_field_after_V_per_m"] / 5e5, polarity)
            self.assertAlmostEqual((event["wall_memory_after_V_per_m"] - event["wall_memory_before_V_per_m"]) / 5e5, -polarity)
            self.assertAlmostEqual(abs(event["q_app_pC"]), scenario["q_peak_pC"])
            self.assertEqual(event["q_app_pC"] > 0, polarity > 0)
            increment = event["hemisphere_free_charge_increment_north_C"]
            self.assertEqual(increment + event["hemisphere_free_charge_increment_south_C"], 0)
            # A surface charge increment changes the counter-field memory;
            # it is not the total charge left on the wall after this event.
            expected_increment = (-math.pi * payload["void"]["radius_m"] ** 2
                                  * EPSILON_0_F_PER_M * (1 + 2 * 4)
                                  * (event["wall_memory_after_V_per_m"] - event["wall_memory_before_V_per_m"]))
            self.assertAlmostEqual(increment / expected_increment, 1)
            self.assertNotIn("hemisphere_free_charge_north_C", event)
        self.assertGreater(scenario["positive_event_count"], 0)
        self.assertGreater(scenario["negative_event_count"], 0)
        # No sampled-peak overshoot: per-event charge remains the threshold
        # drop when the assumed RMS voltage is increased.
        payload["voltage_scenarios_rms_V"] = [5000]
        higher = estimate_pd_void(payload)["radius_cases"][0]["scenarios"][0]
        self.assertAlmostEqual(higher["q_peak_pC"], scenario["q_peak_pC"])
        self.assertGreater(higher["event_count"], scenario["event_count"])

    def test_below_inception_and_exact_peak_onset(self):
        options = dict(inception_field_V_per_m=100, extinction_field_V_per_m=50, frequency_Hz=50)
        self.assertEqual(first_cycle_events(void_peak_field_V_per_m=99, **options), ([], 0))
        events, _ = first_cycle_events(void_peak_field_V_per_m=100, **options)
        self.assertAlmostEqual(events[0]["phase_deg"], 90)

    def test_event_limit_raises_without_silent_truncation(self):
        with self.assertRaises(EventLimitExceeded):
            first_cycle_events(void_peak_field_V_per_m=1000, inception_field_V_per_m=100,
                               extinction_field_V_per_m=50, frequency_Hz=60, max_events=1)
        with self.assertRaises(EventLimitExceeded):
            first_cycle_events(void_peak_field_V_per_m=110, inception_field_V_per_m=100,
                               extinction_field_V_per_m=99.999999999999, frequency_Hz=60, max_events=10000)

    def test_invalid_and_missing_geometry_fields_units_and_air_inputs(self):
        mutations = (
            ("void", "radius_m", 0), ("void", "radius_m", 0.001),
            ("void", "relative_permittivity_host", -1), ("void", "extinction_ratio", 1),
            ("void", "pressure_Pa", 0), ("void", "gas_species", "nitrogen"),
            ("field_evidence", "normalization_voltage_V", True),
            ("field_evidence", "normalization_voltage_V", 20000),
            ("field_evidence", "local_field_geometry_multiplier", 8),
            ("field_evidence", "model_basis", "eighth"),
            ("field_evidence", "point_in_dielectric", False),
            ("field_evidence", "local_field_at_1V_V_per_m", float("inf")),
        )
        for group, key, value in mutations:
            with self.subTest(group=group, key=key):
                payload = inputs()
                payload[group][key] = value
                with self.assertRaises(ValueError):
                    estimate_pd_void(payload)
        payload = inputs()
        del payload["field_evidence"]["source"]
        with self.assertRaises(ValueError):
            estimate_pd_void(payload)

    def test_offline_cli_batch_and_duplicate_ids(self):
        models = []
        for case in ("primary", "secondary"):
            for extinction in (0.3, 0.5, 0.8):
                model = inputs()
                model.update(case=case, scenario_id=f"{case}_ext_{extinction}")
                model["void"]["extinction_ratio"] = extinction
                models.append(model)
        batch = {"models": models, "provenance": {"native_input_sha256": "synthetic-regression"}}
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "input.json", Path(directory) / "result.json"
            source.write_text(json.dumps(batch))
            original = source.read_bytes()
            self.assertEqual(main(["--input", str(source), "--output", str(target)]), 0)
            saved = json.loads(target.read_text())
            self.assertEqual(len(saved["model_results"]), 6)
            self.assertEqual(saved["provenance"], batch["provenance"])
            self.assertEqual(saved["physical_qualification"], "insufficient_data")
            self.assertEqual(source.read_bytes(), original)
        duplicate = deepcopy(batch)
        duplicate["models"][1]["scenario_id"] = duplicate["models"][0]["scenario_id"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            estimate_pd_void_batch(duplicate)
        with self.assertRaises(ValueError):
            estimate_pd_void_batch({"models": []})


if __name__ == "__main__":
    unittest.main()
