import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from regression_260707.verify.compare_efficiency_ab import (
    compare_records,
    compare_time,
    load_result,
    render_markdown,
)
from regression_260707.verify.run_efficiency_ab import (
    merge_params,
    parse_result_json,
    run_arm,
)


def _complete_record(**updates):
    record = {
        "matrix_on": 1,
        "loss_on": 1,
        "thermal_on": 1,
        "result_valid_em": 1,
        "result_valid_thermal": 1,
        "thermal_converged": 1,
        "thermal_extraction_complete": 1,
        "thermal_rx_power_balance_ok": 1,
        "ab_return_code": 0,
        "ab_experiment_sha256": "pair-a",
        "Llt": 100.0,
        "k": 0.98,
        "P_core_total": 1000.0,
        "P_winding_total": 2000.0,
        "B_mean_core": 1.0,
        "Tprobe_Tx_leeward_max": 90.0,
        "Tprobe_core_center_max": 95.0,
        "ab_process_wall_s": 600.0,
    }
    record.update(updates)
    return record


class ComparatorLogicTests(unittest.TestCase):
    def test_complete_variant_within_thresholds_passes_and_reports_saving(self):
        baseline = _complete_record()
        variant = _complete_record(
            Llt=100.4,
            k=0.9761,
            P_core_total=1019.0,
            P_winding_total=1970.0,
            B_mean_core=1.019,
            Tprobe_Tx_leeward_max=91.9,
            Tprobe_core_center_max=93.1,
            ab_process_wall_s=450.0,
        )

        comparison = compare_records(baseline, variant)

        self.assertTrue(comparison["passed"])
        self.assertAlmostEqual(comparison["time"]["saved_seconds"], 150.0)
        self.assertAlmostEqual(comparison["time"]["saved_pct"], 25.0)
        self.assertIn("**PASS**", render_markdown(comparison))

    def test_threshold_boundaries_fail_by_family(self):
        baseline = _complete_record()
        variant = _complete_record(
            Llt=100.6,
            P_core_total=1021.0,
            B_mean_core=1.03,
            Tprobe_Tx_leeward_max=92.1,
        )

        comparison = compare_records(baseline, variant)

        self.assertFalse(comparison["passed"])
        failed = {
            (item["target"], item["family"])
            for item in comparison["targets"]
            if not item["passed"]
        }
        self.assertIn(("Llt", "Llt/k"), failed)
        self.assertIn(("P_core_total", "loss"), failed)
        self.assertIn(("B_mean_core", "B"), failed)
        self.assertIn(("Tprobe_Tx_leeward_max", "temperature"), failed)

    def test_missing_required_family_fails_closed(self):
        baseline = {
            "loss_on": 1, "thermal_on": 0, "Llt": 10, "k": 0.9,
            "result_valid_em": 1,
        }
        variant = dict(baseline)

        comparison = compare_records(baseline, variant)

        self.assertFalse(comparison["passed"])
        self.assertFalse(comparison["family_counts"]["loss"]["passed"])
        self.assertFalse(comparison["family_counts"]["B"]["passed"])

    def test_invalid_solver_or_wrapper_status_fails_closed(self):
        baseline = _complete_record()
        variant = _complete_record(
            result_valid_em=0,
            result_valid_thermal=0,
            thermal_converged=0,
            ab_return_code=1,
        )

        comparison = compare_records(baseline, variant)

        self.assertFalse(comparison["passed"])
        failed = {
            (item["arm"], item["field"])
            for item in comparison["quality_checks"]
            if not item["passed"]
        }
        self.assertIn(("variant", "result_valid_em"), failed)
        self.assertIn(("variant", "result_valid_thermal"), failed)
        self.assertIn(("variant", "thermal_converged"), failed)
        self.assertIn(("variant", "ab_return_code"), failed)

    def test_mismatched_experiment_identity_fails_closed(self):
        baseline = _complete_record()
        variant = _complete_record(ab_experiment_sha256="pair-b")

        comparison = compare_records(baseline, variant)

        self.assertFalse(comparison["passed"])
        identity = next(
            item for item in comparison["quality_checks"]
            if item["field"] == "ab_experiment_sha256"
        )
        self.assertFalse(identity["passed"])

    def test_swapped_harness_arms_fail_closed(self):
        baseline = _complete_record(ab_arm="variant")
        variant = _complete_record(ab_arm="baseline")

        comparison = compare_records(baseline, variant)

        self.assertFalse(comparison["passed"])
        arm_checks = [
            item for item in comparison["quality_checks"]
            if item["field"] == "ab_arm"
        ]
        self.assertEqual(len(arm_checks), 2)
        self.assertTrue(all(not item["passed"] for item in arm_checks))

    def test_optional_both_nan_target_is_skipped(self):
        baseline = _complete_record(Tprobe_Rx_side_leeward_max=float("nan"))
        variant = _complete_record(Tprobe_Rx_side_leeward_max=float("nan"))

        comparison = compare_records(baseline, variant)

        side = next(
            item for item in comparison["targets"]
            if item["target"] == "Tprobe_Rx_side_leeward_max"
        )
        self.assertEqual(side["reason"], "both_non_finite_skipped")
        self.assertTrue(comparison["passed"])

    def test_thresholds_are_configurable(self):
        baseline = _complete_record()
        variant = _complete_record(Llt=100.75)
        self.assertFalse(compare_records(baseline, variant)["passed"])
        comparison = compare_records(
            baseline, variant, {"electromagnetic_relative_pct": 1.0}
        )
        self.assertTrue(comparison["passed"])

    def test_missing_timing_component_is_not_reported_as_saving(self):
        baseline = {
            "matrix_on": 1, "loss_on": 1, "thermal_on": 1,
            "time_matrix": 10, "time_loss": 20, "time_thermal": 30,
        }
        variant = {
            "matrix_on": 1, "loss_on": 1, "thermal_on": 1,
            "time_matrix": 8, "time_loss": 18,
        }

        timing = compare_time(baseline, variant)

        self.assertFalse(timing["comparable"])
        self.assertIsNone(timing["saved_seconds"])


class ComparatorIoTests(unittest.TestCase):
    def test_load_result_selects_last_csv_row_by_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["Llt", "k"])
                writer.writeheader()
                writer.writerow({"Llt": "10", "k": "0.8"})
                writer.writerow({"Llt": "11", "k": "0.9"})
            self.assertEqual(load_result(path), {"Llt": "11", "k": "0.9"})

    def test_load_result_accepts_json_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.json"
            path.write_text(json.dumps([{"Llt": 1}, {"Llt": 2}]), encoding="utf-8")
            self.assertEqual(load_result(path)["Llt"], 2)


class AbRunnerUnitTests(unittest.TestCase):
    def test_plain_and_structured_overlays(self):
        baseline, variant, normalized = merge_params(
            {"matrix_max_passes": 20}, {"matrix_max_passes": 16}
        )
        self.assertEqual(baseline["matrix_max_passes"], 20)
        self.assertEqual(variant["matrix_max_passes"], 16)
        self.assertEqual(normalized["baseline"], {})

        baseline, variant, _ = merge_params(
            {"matrix_max_passes": 20},
            {"baseline": {"keep_project": 0},
             "variant": {"matrix_max_passes": 16}},
        )
        self.assertEqual(baseline["keep_project"], 0)
        self.assertEqual(variant["matrix_max_passes"], 16)

    def test_result_parser_uses_last_streamed_result(self):
        result = parse_result_json([
            "noise\n",
            'RESULT_JSON {"Llt": 1}\n',
            'RESULT_JSON {"Llt": 2}\n',
        ])
        self.assertEqual(result, {"Llt": 2})

    def test_run_arm_emits_harvestable_result_and_replaces_stale_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = root / "fake_runner.py"
            runner.write_text(
                "import json\n"
                "print('RESULT_JSON ' + json.dumps({'Llt': 1.0}))\n",
                encoding="utf-8",
            )
            params = root / "params.json"
            params.write_text('{"matrix_on": 1}\n', encoding="utf-8")
            result = root / "result.json"
            result.write_text('{"stale": true}\n', encoding="utf-8")
            output = io.StringIO()

            with redirect_stdout(output):
                run_arm(
                    "baseline", params, result, root / "run.log", runner,
                    sys.executable, [], root, "pair-sha",
                )

            saved = json.loads(result.read_text(encoding="utf-8"))
            self.assertNotIn("stale", saved)
            self.assertEqual(saved["ab_experiment_sha256"], "pair-sha")
            self.assertTrue(any(
                line.startswith("RESULT_JSON ")
                for line in output.getvalue().splitlines()
            ))

    def test_failed_rerun_removes_stale_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = root / "no_result.py"
            runner.write_text("print('no result')\n", encoding="utf-8")
            params = root / "params.json"
            params.write_text('{"matrix_on": 1}\n', encoding="utf-8")
            result = root / "result.json"
            result.write_text('{"stale": true}\n', encoding="utf-8")

            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(
                RuntimeError, "emitted no RESULT_JSON"
            ):
                run_arm(
                    "variant", params, result, root / "run.log", runner,
                    sys.executable, [], root, "pair-sha",
                )

            self.assertFalse(result.exists())


if __name__ == "__main__":
    unittest.main()
