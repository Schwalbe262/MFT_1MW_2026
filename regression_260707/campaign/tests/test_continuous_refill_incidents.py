import copy
import sys
import unittest
from pathlib import Path
from unittest import mock


CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAMPAIGN_DIR))

import _continuous_refill_b171c7c as controller  # noqa: E402


def operator_cancelled_tasks():
    common = {
        "project": "MFT_1MW_2026v1",
        "status": "cancelled",
        "account_name": "r1jae262",
        "requested_account_name": "",
        "allocation_id": 8_019,
        "slurm_job_id": "731354",
        "allocation_node_name": "n045",
        "started_at": None,
        "exit_code": None,
        "failure_message": "",
    }
    rows = [
        {
            **common,
            "id": 28_746,
            "name": "mft-camp-sb171c7c-le6b9b9d-18221",
            "dedupe_key": (
                "mft-al:mft-camp-sb171c7c-le6b9b9d-18221:"
                f"{controller.SOLVER}:{controller.LIBRARY}:a1a34b62c7f252d7"),
            "created_at": "2026-07-12 10:59:54",
            "attached_at": "2026-07-12 11:00:46",
            "launch_started_at": "2026-07-12 11:00:47",
            "finished_at": "2026-07-12 14:24:22",
        },
        {
            **common,
            "id": 28_747,
            "name": "mft-camp-sb171c7c-le6b9b9d-18222",
            "dedupe_key": (
                "mft-al:mft-camp-sb171c7c-le6b9b9d-18222:"
                f"{controller.SOLVER}:{controller.LIBRARY}:23c0b44a6cf1e097"),
            "created_at": "2026-07-12 10:59:58",
            "attached_at": "2026-07-12 11:00:47",
            "launch_started_at": "2026-07-12 11:00:48",
            "finished_at": "2026-07-12 14:24:21",
        },
        {
            **common,
            "id": 28_748,
            "name": "mft-camp-sb171c7c-le6b9b9d-18223",
            "dedupe_key": (
                "mft-al:mft-camp-sb171c7c-le6b9b9d-18223:"
                f"{controller.SOLVER}:{controller.LIBRARY}:ec8039750141c58e"),
            "created_at": "2026-07-12 11:00:04",
            "attached_at": "2026-07-12 11:00:48",
            "launch_started_at": "2026-07-12 11:00:49",
            "finished_at": "2026-07-12 14:24:21",
        },
    ]
    return rows


def resolved_scheduler_parent_cancel_tasks():
    common = {
        "project": "MFT_1MW_2026v1",
        "status": "failed",
        "account_name": "dw16",
        "requested_account_name": "",
        "exit_code": 143,
        "failure_message": "parent Slurm allocation cancelled",
        "scheduling_profile": "fea_bursty",
        "timeout_seconds": 14_400,
    }
    return [
        {"id": task_id, **common, **copy.deepcopy(expected)}
        for task_id, expected in (
            controller.rapid_campaign
            ._RESOLVED_SCHEDULER_PARENT_CANCEL_TASKS.items())
    ]



class OperatorCancelledStalePrepolicyTests(unittest.TestCase):
    def test_exact_three_task_incident_matches_both_seals(self):
        evidence = controller._operator_cancelled_stale_prepolicy_evidence(
            operator_cancelled_tasks())

        self.assertEqual(evidence["task_ids"], [28_746, 28_747, 28_748])
        self.assertEqual(
            evidence["identity_sha256"],
            controller.OPERATOR_CANCELLED_STALE_PREPOLICY_IDENTITY_SHA256,
        )
        self.assertEqual(
            evidence["audit_sha256"],
            controller.OPERATOR_CANCELLED_STALE_PREPOLICY_AUDIT_SHA256,
        )
        self.assertTrue(evidence["retained_in_lifetime_invalid_accounting"])
        self.assertTrue(
            evidence["excluded_from_current_simulation_valid_rate"])

    def test_any_identity_drift_fails_closed(self):
        tasks = operator_cancelled_tasks()
        tasks[1]["launch_started_at"] = "2026-07-12 11:00:55"

        with self.assertRaisesRegex(RuntimeError, "task 28747 drifted"):
            controller._operator_cancelled_stale_prepolicy_evidence(tasks)

    def test_sealed_null_started_cancellations_are_excluded_from_health_rate(self):
        tasks = operator_cancelled_tasks()
        tasks.append({
            "id": 30_001,
            "started_at": "2026-07-12 12:00:00",
        })
        outcomes = [
            {
                "task_id": row["id"],
                "state": "invalid",
                "expected_failure_reason": (
                    "operator_cancelled_stale_prepolicy_launch"),
            }
            for row in tasks[:3]
        ]
        outcomes.append({"task_id": 30_001, "state": "valid"})

        health = controller._production_health_cohort({
            "tasks": tasks,
            "outcomes": outcomes,
        })

        self.assertEqual(health["terminal"], 1)
        self.assertEqual(health["valid"], 1)
        self.assertEqual(health["valid_rate"], 1.0)
        self.assertEqual(health["outcomes"], [outcomes[-1]])

    def test_unsealed_null_started_terminal_still_fails_closed(self):
        task = {
            "id": 30_002,
            "status": "cancelled",
            "launch_started_at": "2026-07-12 11:00:48",
            "started_at": None,
        }
        outcome = {"task_id": 30_002, "state": "invalid"}

        with self.assertRaisesRegex(RuntimeError, "has no started_at"):
            controller._production_health_cohort({
                "tasks": [task], "outcomes": [outcome],
            })


class ResolvedSchedulerParentCancelTests(unittest.TestCase):
    def test_exact_26_task_incident_matches_identity_audit_and_runtime_seals(self):
        evidence = controller._resolved_scheduler_parent_cancel_evidence(
            resolved_scheduler_parent_cancel_tasks())

        self.assertEqual(
            evidence["task_ids"],
            list(controller.RESOLVED_SCHEDULER_PARENT_CANCEL_IDS),
        )
        self.assertEqual(
            evidence["identity_sha256"],
            controller.RESOLVED_SCHEDULER_PARENT_CANCEL_IDENTITY_SHA256,
        )
        self.assertEqual(
            evidence["audit_sha256"],
            controller.RESOLVED_SCHEDULER_PARENT_CANCEL_AUDIT_SHA256,
        )
        self.assertEqual(len(evidence["parents"]), 4)
        self.assertTrue(evidence["excluded_from_current_runtime_health"])
        self.assertTrue(evidence["retained_in_lifetime_invalid_accounting"])

    def test_exact_incident_is_reclassified_from_cached_failure(self):
        task = resolved_scheduler_parent_cancel_tasks()[0]
        outcome = {
            "task_id": task["id"],
            "name": task["name"],
            "status": "failed",
            "state": "invalid",
            "error_message": "old cached partial solver stdout",
            "error_fingerprint": "stale",
        }

        refreshed = controller.rapid_campaign._refresh_failure_outcome(
            outcome, task)

        self.assertEqual(
            refreshed["expected_failure_reason"],
            "resolved_scheduler_parent_cancel_incident",
        )
        self.assertIsNone(refreshed["error_fingerprint"])

    def test_exact_incident_stays_lifetime_invalid_but_is_excluded_from_health(self):
        tasks = resolved_scheduler_parent_cancel_tasks()
        current = {
            "id": 30_100,
            "started_at": "2026-07-12 15:30:00",
        }
        outcomes = [
            {
                "task_id": task["id"],
                "state": "invalid",
                "expected_failure_reason": (
                    "resolved_scheduler_parent_cancel_incident"),
            }
            for task in tasks
        ]
        valid = {"task_id": current["id"], "state": "valid"}
        health = controller._production_health_cohort({
            "tasks": [*tasks, current],
            "outcomes": [*outcomes, valid],
        })

        self.assertEqual(len(outcomes), 26)
        self.assertEqual(health["terminal"], 1)
        self.assertEqual(health["valid"], 1)
        self.assertEqual(health["valid_rate"], 1.0)
        self.assertEqual(health["outcomes"], [valid])

    def test_identity_drift_and_future_exit143_remain_fail_closed(self):
        tasks = resolved_scheduler_parent_cancel_tasks()
        tasks[0]["slurm_job_id"] = "999999"
        with self.assertRaisesRegex(
                RuntimeError, "task 29026 drifted"):
            controller._resolved_scheduler_parent_cancel_evidence(tasks)

        future = {
            **resolved_scheduler_parent_cancel_tasks()[0],
            "id": 39_999,
            "name": "mft-camp-sb171c7c-le6b9b9d-99999",
            "dedupe_key": (
                "mft-al:mft-camp-sb171c7c-le6b9b9d-99999:"
                f"{controller.SOLVER}:{controller.LIBRARY}:future"),
            "started_at": "2026-07-12 16:00:00",
            "finished_at": "2026-07-12 17:00:00",
        }
        outcome = {"status": "failed", "state": "invalid"}
        refreshed = controller.rapid_campaign._refresh_failure_outcome(
            outcome, future)
        self.assertIsNone(refreshed["expected_failure_reason"])
        self.assertIsNotNone(refreshed["error_fingerprint"])



def sealed_old_timeout_contract_tasks():
    common = {
        "project": "MFT_1MW_2026v1",
        "status": "failed",
        "requested_account_name": "",
        "exit_code": 124,
        "failure_message": "task timed out after 14400s",
        "scheduling_profile": "fea_bursty",
        "timeout_seconds": 14_400,
    }
    return [
        {"id": task_id, **common, **copy.deepcopy(expected)}
        for task_id, expected in (
            controller.rapid_campaign
            ._SEALED_OLD_TIMEOUT_CONTRACT_TASKS.items())
    ]


class SealedOldTimeoutContractTests(unittest.TestCase):
    def test_exact_six_match_both_seals_and_leave_future_timeout_fail_closed(self):
        tasks = sealed_old_timeout_contract_tasks()
        evidence = controller._sealed_old_timeout_contract_evidence(tasks)
        self.assertEqual(
            evidence["identity_sha256"],
            controller.SEALED_OLD_TIMEOUT_CONTRACT_IDENTITY_SHA256)
        self.assertEqual(
            evidence["audit_sha256"],
            controller.SEALED_OLD_TIMEOUT_CONTRACT_AUDIT_SHA256)

        refreshed = controller.rapid_campaign._refresh_failure_outcome(
            {"status": "failed", "state": "invalid"}, tasks[0])
        self.assertEqual(
            refreshed["expected_failure_reason"],
            "sealed_old_timeout_contract_incident")
        self.assertIsNone(refreshed["error_fingerprint"])

        future = {**tasks[0], "id": 40_000}
        future_outcome = controller.rapid_campaign._refresh_failure_outcome(
            {"status": "failed", "state": "invalid"}, future)
        self.assertIsNone(future_outcome["expected_failure_reason"])
        self.assertIsNotNone(future_outcome["error_fingerprint"])

    def test_exact_six_are_lifetime_invalid_but_not_current_health(self):
        tasks = sealed_old_timeout_contract_tasks()
        outcomes = [
            {
                "task_id": task["id"],
                "state": "invalid",
                "expected_failure_reason":
                    "sealed_old_timeout_contract_incident",
                "error_message": "failure_message=task timed out after 14400s",
            }
            for task in tasks
        ]
        current = {"id": 40_001, "started_at": "2026-07-12 15:40:00"}
        valid = {"task_id": current["id"], "state": "valid"}
        health = controller._production_health_cohort({
            "tasks": [*tasks, current],
            "outcomes": [*outcomes, valid],
        })
        self.assertEqual(health["terminal"], 1)
        self.assertEqual(health["valid_rate"], 1.0)


@unittest.skip(
    "archived one-off _continuous_refill_b171c7c.py target-rollback incident "
    "tests require runtime evidence that is absent from clean worktrees")
class Target300RollbackTests(unittest.TestCase):
    @staticmethod
    def _production_state():
        artifact = controller._target300_rollback_artifact()
        tasks = []
        outcomes = []
        cache = {}
        for prepared in artifact["eligible_snapshot"]:
            task = {
                **copy.deepcopy(prepared),
                "status": "cancelled",
                "finished_at": "2026-07-12 16:18:14",
            }
            outcome = {
                "task_id": task["id"],
                "name": task["name"],
                "status": "cancelled",
                "state": "invalid",
                "reason": "task_cancelled",
                "error_fingerprint": None,
                "error_message": "",
            }
            tasks.append(task)
            outcomes.append(outcome)
            cache[str(task["id"])] = dict(outcome)
        return {"tasks": tasks, "outcomes": outcomes, "cache": cache}

    def test_exact_artifact_and_77_task_scope_are_immutable(self):
        artifact = controller._target300_rollback_artifact()
        self.assertEqual(
            artifact["cancelled_ids"],
            list(controller.TARGET300_ROLLBACK_CANCELLED_IDS))
        self.assertEqual(artifact["final_active"], 300)
        self.assertEqual(artifact["final_statuses"], {"running": 300})

    def test_exact_cancelled_outcomes_remain_lifetime_invalid_but_leave_health(self):
        production_state = self._production_state()
        evidence = {
            "task_ids": list(controller.TARGET300_ROLLBACK_CANCELLED_IDS),
        }
        controller._classify_target300_rollback_outcomes(
            production_state, evidence)
        health = controller._production_health_cohort(
            production_state,
            set(controller.TARGET300_ROLLBACK_CANCELLED_IDS))

        self.assertEqual(len(production_state["outcomes"]), 77)
        self.assertTrue(all(
            row["state"] == "invalid"
            and row["expected_failure_reason"]
                == "user_target_rollback_cancelled"
            for row in production_state["outcomes"]))
        self.assertEqual(health["terminal"], 0)

    def test_missing_authorization_or_started_near_miss_fails_closed(self):
        production_state = self._production_state()
        evidence = {
            "task_ids": list(controller.TARGET300_ROLLBACK_CANCELLED_IDS),
        }
        controller._classify_target300_rollback_outcomes(
            production_state, evidence)
        with self.assertRaisesRegex(RuntimeError, "authorization is incomplete"):
            controller._production_health_cohort(
                production_state,
                set(controller.TARGET300_ROLLBACK_CANCELLED_IDS[:-1]))

        production_state["tasks"][0]["started_at"] = (
            "2026-07-12 16:18:00")
        with self.assertRaisesRegex(RuntimeError, "classification drifted"):
            controller._production_health_cohort(
                production_state,
                set(controller.TARGET300_ROLLBACK_CANCELLED_IDS))



class Pool300ContractTests(unittest.TestCase):
    @staticmethod
    def _legacy250_state():
        state = controller._new_state()
        state.update({
            "target_active": controller.LEGACY_TARGET_ACTIVE,
            "cycle_serial": controller.TARGET_400_TRANSITION_CYCLE,
            "terminal_cycle_highwater": controller.TARGET_400_TRANSITION_CYCLE,
        })
        return state

    @staticmethod
    def _prior400_state():
        state = controller._new_state()
        state.update({
            "target_active": controller.PREVIOUS_TARGET_ACTIVE,
            "cycle_serial": controller.TARGET_300_TRANSITION_CYCLE,
            "terminal_cycle_highwater": controller.TARGET_300_TRANSITION_CYCLE,
        })
        return state

    def test_target_and_authorization_contract_are_exactly_300(self):
        self.assertEqual(controller.TARGET_ACTIVE, 300)
        self.assertEqual(controller.REFILL_ACTION, "refill_300")
        self.assertEqual(controller.EVIDENCE_MODE, "dynamic_project_cap_v1")
        self.assertEqual(controller._new_state()["target_active"], 300)

    def test_historical_250_to_400_then_current_400_to_300_migrations(self):
        legacy = self._legacy250_state()
        controller._validate_state(legacy)
        self.assertTrue(controller._promote_target400_state(legacy))
        self.assertEqual(legacy["target_active"], 400)
        controller._validate_state(legacy)

        current = self._prior400_state()
        controller._validate_state(current)
        self.assertTrue(controller._migrate_target300_state(current))
        self.assertEqual(current["target_active"], 300)
        controller._validate_state(current)
        self.assertFalse(controller._migrate_target300_state(current))

    def test_target300_migration_at_any_other_boundary_fails_closed(self):
        state = self._prior400_state()
        state["cycle_serial"] -= 1
        state["terminal_cycle_highwater"] -= 1
        with self.assertRaisesRegex(RuntimeError, "terminal cycle336"):
            controller._migrate_target300_state(state)

        invalid = controller._new_state()
        invalid["target_active"] = 399
        with self.assertRaisesRegex(RuntimeError, "legacy250, prior400, or current300"):
            controller._validate_state(invalid)

    def test_cycle_targets_are_250_then_400_then_300(self):
        def payload(serial, target):
            return {
                "schema_version": 2,
                "state_revision": 0,
                "cycle_serial": serial,
                "plan_sha256": controller.PLAN_SHA256,
                "target_active": target,
                "status": "completed",
                "formal_journal": {"events": []},
            }

        controller._validate_cycle(
            payload(334, 250), Path("cycle-000334.json"))
        controller._validate_cycle(
            payload(335, 400), Path("cycle-000335.json"))
        controller._validate_cycle(
            payload(336, 400), Path("cycle-000336.json"))
        controller._validate_cycle(
            payload(337, 300), Path("cycle-000337.json"))
        with self.assertRaisesRegex(RuntimeError, "campaign identity drifted"):
            controller._validate_cycle(
                payload(336, 300), Path("cycle-000336.json"))

    def test_cli_requires_dynamic_cap_flag_and_rejects_legacy_pool400_flag(self):
        with mock.patch.object(controller, "run_once", return_value={}) as run:
            result = controller.main([
                "--execute",
                "--reviewed-plan-sha", controller.PLAN_SHA256,
                "--authorize-dynamic-project-cap",
            ])
        self.assertEqual(result, 0)
        self.assertTrue(
            run.call_args.kwargs["authorize_dynamic_project_cap"])

        with self.assertRaises(SystemExit):
            controller.main([
                "--execute",
                "--reviewed-plan-sha", controller.PLAN_SHA256,
                "--authorize-concurrent-pool400",
            ])

    def test_batch_wrapper_persists_plan_and_ledger_once_each(self):
        generation = f"{controller.SOLVER}:{controller.LIBRARY}:seed{controller.SEED}"
        snapshot = {
            "state_revision": 0,
            "serial": controller.INITIAL_SERIAL + 300,
            "submitted_samples": 300,
            "outstanding": [],
            "candidate_generation": generation,
            "candidate_cursor": 10,
            "candidate_cursors": {generation: 10},
            "candidate_raw_index": 9,
            "task_ids_by_generation": {generation: []},
            "task_expected_rows": {},
            "adoption_sha256": controller.PLAN_SHA256,
            "adoption_manifest": "plan.json",
            "used_names": [],
            "used_params_sha256": [],
            "used_dedupe_keys": [],
        }
        journal = {
            "events": [],
            "batch_commit": True,
            "submitted_count": 0,
        }
        cycle = {"formal_journal": journal}
        next_values = [
            (11, 101, {"candidate": 1}),
            (12, 102, {"candidate": 2}),
            (13, 103, {"candidate": 3}),
        ]

        with mock.patch.object(
                controller, "_load_feeder_state",
                side_effect=lambda *_args, **_kwargs: copy.deepcopy(snapshot)
        ), mock.patch.object(
            controller, "_save_feeder_state"
        ) as save_feeder, mock.patch.object(
            controller, "_save_cycle"
        ) as save_cycle, mock.patch.object(
            controller, "_candidate_contract"
        ), mock.patch.object(
            controller.pinned_pilot, "candidate_digest",
            side_effect=lambda params: f"sha-{params['candidate']}"
        ), mock.patch.object(
            controller.feeder, "next_valid_candidate",
            side_effect=next_values,
        ), mock.patch.object(
            controller.feeder, "submit", side_effect=[701, 702, 703]
        ), mock.patch.object(
            controller.production, "_task_metadata",
            side_effect=lambda task_id, _expected: {"id": task_id},
        ):
            with controller._feeder_io(
                    {"plan": {"tasks": []}}, Path("cycle-000335.json"),
                    cycle, journal):
                cursor = 10
                params_rows = []
                for index in range(3):
                    cursor, raw_index, params = (
                        controller.feeder.next_valid_candidate(cursor))
                    name = f"batch-{index}"
                    journal["events"].append({
                        "name": name,
                        "candidate_raw_index": raw_index,
                        "dedupe_key": f"dedupe-{index}",
                        "task_id": None,
                        "accepted_or_reconciled": False,
                        "ledger_committed": False,
                        "uncertain": False,
                    })
                    params_rows.append((name, params))
                for index, (name, params) in enumerate(params_rows):
                    task_id = controller.feeder.submit(
                        name, "work", params,
                        controller.SOLVER, controller.LIBRARY)
                    journal["events"][index]["task_id"] = task_id
                    journal["events"][index]["accepted_or_reconciled"] = True
                committed = copy.deepcopy(snapshot)
                controller.feeder.save_state(committed)

        save_feeder.assert_called_once()
        self.assertEqual(save_cycle.call_count, 2)
        self.assertEqual(
            [call.args[2] for call in save_cycle.call_args_list],
            ["mutation_about_to_submit", "ledger_committed"],
        )
        self.assertEqual(journal["submitted_count"], 3)
        self.assertTrue(all(row["ledger_committed"] for row in journal["events"]))


if __name__ == "__main__":
    unittest.main()
