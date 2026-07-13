import json
import sys
import unittest
from pathlib import Path
from unittest import mock


CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAMPAIGN_DIR))

import plan_future_thermal_recovery4 as planner  # noqa: E402


FUTURE_SOLVER = "f" * 40


class FutureThermalRecoveryPlanTests(unittest.TestCase):
    def test_plan_has_four_exact_sources_unique_identities_and_no_mutation_path(self):
        plan = planner.build_plan(FUTURE_SOLVER)

        self.assertEqual(plan["mode"], "plan_only")
        self.assertIs(plan["submission_enabled"], False)
        self.assertEqual(plan["scheduler_mutation_count"], 0)
        self.assertEqual(plan["task_count"], 4)
        self.assertEqual(plan["concurrency"], 4)
        self.assertEqual(plan["pilot_gate"]["strict_valid_required"], 4)
        self.assertIs(plan["pilot_gate"]["partial_pass_allowed"], False)
        self.assertEqual(
            [task["source_task_id"] for task in plan["tasks"]],
            [27794, 27928, 27880, 27758],
        )
        self.assertEqual(
            [task["source_manifest_index"] for task in plan["tasks"]],
            [39, 173, 125, 3],
        )
        self.assertEqual(
            [task["source_candidate_raw_index"] for task in plan["tasks"]],
            [1978, 2481, 2313, 1855],
        )
        self.assertEqual(len({task["name"] for task in plan["tasks"]}), 4)
        self.assertEqual(len({task["dedupe_key"] for task in plan["tasks"]}), 4)
        self.assertTrue(all(
            task["resources"] == planner.RESOURCES for task in plan["tasks"]
        ))
        self.assertTrue(all(
            task["effective_params"]["matrix_on"] == 1
            and task["effective_params"]["loss_on"] == 1
            and task["effective_params"]["thermal_on"] == 1
            and task["effective_params"]["P_target"] == 1_000_000.0
            for task in plan["tasks"]
        ))
        for task in plan["tasks"][:3]:
            acceptance = task["acceptance"]
            self.assertIs(acceptance["strict_valid_required"], True)
            self.assertEqual(
                acceptance["thermal_entrypoint_exact"], "ThermalSetup")
            self.assertIs(acceptance["analyze_all_forbidden"], True)
            self.assertIs(acceptance["fresh_monitor_required"], True)
            self.assertLessEqual(acceptance["startup_retry_max"], 1)
        self.assertIs(
            plan["tasks"][3]["acceptance"]["known_good_nonregression"], True)
        unsigned = dict(plan)
        seal = unsigned.pop("plan_sha256")
        self.assertEqual(seal, planner._sha(unsigned))

    def test_source_drift_and_current_solver_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError, "future solver"):
            planner.build_plan(planner.SOURCE_SOLVER)

        manifest = planner._load_source_manifest()
        drifted = json.loads(json.dumps(manifest))
        drifted["tasks"][39]["params"]["core_plate_t"] += 0.1
        with self.assertRaisesRegex(RuntimeError, "drifted"):
            planner._source_record(drifted, planner.SOURCE_CASES[0])

    def test_build_plan_never_calls_scheduler_network_or_submission(self):
        with mock.patch.object(
                planner.scheduler_client.requests, "get",
                side_effect=AssertionError("network forbidden")), mock.patch.object(
                planner.scheduler_client.requests, "post",
                side_effect=AssertionError("mutation forbidden")), mock.patch.object(
                planner.scheduler_client, "submit_verification",
                side_effect=AssertionError("submission forbidden")):
            plan = planner.build_plan(FUTURE_SOLVER)

        self.assertEqual(plan["scheduler_mutation_count"], 0)


if __name__ == "__main__":
    unittest.main()
