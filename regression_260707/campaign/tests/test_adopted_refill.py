import unittest
from pathlib import Path
from unittest import mock

import sys


CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAMPAIGN_DIR))

import feeder  # noqa: E402


SOLVER = "a" * 40
LIBRARY = "b" * 40
ADOPTION = "c" * 64
RESOURCES = {
    "cpus": 4,
    "memory_mb": 65_536,
    "timeout_seconds": 14_400,
}


def decision(terminal=20, valid=18, paused=False):
    return {
        "paused": paused,
        "target_active": 300,
        "action": "refill_300",
        "production": {
            "terminal": terminal,
            "valid": valid,
            "valid_rate": valid / terminal,
        },
    }


def authorize(**overrides):
    kwargs = {
        "max_samples": 12_000,
        "solver_revision": SOLVER,
        "library_revision": LIBRARY,
        "candidate_seed": 260710,
        "local_passed": True,
        "adoption_sha256": ADOPTION,
        "initial_count": 250,
        "evidence_mode": "preloaded250_v1",
        "strict_rows": 25,
        "target_strict_rows": 3_000,
        **RESOURCES,
    }
    kwargs.update(overrides)
    with mock.patch.object(
            feeder.scheduler_client,
            "campaign_mutation_lock_is_held",
            return_value=True,
    ):
        return feeder._authorize_adopted_refill(decision(), **kwargs)


class AdoptedAuthorizationTests(unittest.TestCase):
    def test_exact_preloaded_contract_issues_distinct_sealed_authorization(self):
        auth = authorize()

        self.assertIsInstance(auth, feeder._AdoptedRefillAuthorization)
        self.assertEqual(auth.target, 300)
        self.assertEqual(auth.max_samples, 12_000)
        self.assertEqual(auth.adoption_sha256, ADOPTION)
        self.assertEqual(auth.initial_count, 250)
        self.assertEqual(auth.strict_rows, 25)
        self.assertEqual(auth.target_strict_rows, 3_000)
        self.assertEqual(
            (auth.cpus, auth.memory_mb, auth.timeout_seconds),
            (4, 65_536, 14_400),
        )

    def test_fleet_evidence_below_twenty_or_ninety_percent_is_rejected(self):
        with mock.patch.object(
                feeder.scheduler_client,
                "campaign_mutation_lock_is_held",
                return_value=True,
        ):
            with self.assertRaisesRegex(feeder.SchedulerError, "fleet20/90"):
                feeder._authorize_adopted_refill(
                    decision(terminal=19, valid=18),
                    max_samples=12_000,
                    solver_revision=SOLVER,
                    library_revision=LIBRARY,
                    candidate_seed=260710,
                    local_passed=True,
                    adoption_sha256=ADOPTION,
                    initial_count=250,
                    evidence_mode="preloaded250_v1",
                    strict_rows=25,
                    target_strict_rows=3_000,
                    **RESOURCES,
                )
            with self.assertRaisesRegex(feeder.SchedulerError, "fleet20/90"):
                feeder._authorize_adopted_refill(
                    decision(terminal=20, valid=17),
                    max_samples=12_000,
                    solver_revision=SOLVER,
                    library_revision=LIBRARY,
                    candidate_seed=260710,
                    local_passed=True,
                    adoption_sha256=ADOPTION,
                    initial_count=250,
                    evidence_mode="preloaded250_v1",
                    strict_rows=25,
                    target_strict_rows=3_000,
                    **RESOURCES,
                )

    def test_truthy_local_string_and_coerced_initial_count_are_rejected(self):
        with self.assertRaisesRegex(feeder.SchedulerError, "local3"):
            authorize(local_passed="yes")
        with self.assertRaisesRegex(feeder.SchedulerError, "preloaded-250"):
            authorize(initial_count="250")

    def test_target_reached_and_wrong_resources_are_rejected(self):
        with self.assertRaisesRegex(feeder.SchedulerError, "strict-row"):
            authorize(strict_rows=3_000)
        with self.assertRaisesRegex(feeder.SchedulerError, "4 CPU/64 GiB/4 hours"):
            authorize(memory_mb=32_768)

    def test_step_wrapper_binds_every_authorized_field_and_journal(self):
        auth = authorize()
        journal = {"events": []}
        with mock.patch.object(
                feeder.scheduler_client,
                "campaign_mutation_lock_is_held",
                return_value=True,
        ), mock.patch.object(feeder, "_step_locked", return_value=True) as step:
            result = feeder._step_from_adopted_controller(
                12_000,
                authorization=auth,
                target=300,
                buffer=0,
                solver_revision=SOLVER,
                library_revision=LIBRARY,
                candidate_seed=260710,
                adoption_sha256=ADOPTION,
                initial_count=250,
                evidence_mode="preloaded250_v1",
                strict_rows=25,
                target_strict_rows=3_000,
                journal=journal,
                **RESOURCES,
            )

        self.assertTrue(result)
        step.assert_called_once_with(
            12_000,
            target=300,
            buffer=0,
            solver_revision=SOLVER,
            library_revision=LIBRARY,
            candidate_seed=260710,
            _adopted_authorization=auth,
            _submit_resources=RESOURCES,
            _refill_journal=journal,
        )


class AdoptedRefillJournalTests(unittest.TestCase):
    def test_partial_submission_records_committed_and_failed_events(self):
        auth = authorize()
        generation = f"{SOLVER}:{LIBRARY}:seed260710"
        state = {
            "serial": 17111,
            "submitted_samples": 250,
            "candidate_generation": generation,
            "candidate_cursor": 939,
            "candidate_cursors": {generation: 939},
            "outstanding": [],
            "task_expected_rows": {},
        }
        campaign_counts = {"queued": 297, "attaching": 0, "running": 0}
        capacity = {
            "ready_fit_slots": 0,
            "queue_state": "open",
            "queue_reason": "",
            "project_active": 297,
            "project_submission_slots": 3,
            "submission_allowed": True,
        }
        journal = {"events": []}

        with mock.patch.object(
                feeder.scheduler_client,
                "campaign_mutation_lock_is_held",
                return_value=True,
        ), mock.patch.object(feeder, "load_state", return_value=state), mock.patch.object(
            feeder,
            "scheduler_snapshot",
            return_value=(campaign_counts, campaign_counts, [], capacity),
        ), mock.patch.object(
            feeder,
            "dataset_collection_snapshot",
            return_value=(10, set()),
        ), mock.patch.object(feeder, "campaign_inventory", return_value=[]), mock.patch.object(
            feeder,
            "reserved_unjudged_rows",
            return_value=0,
        ), mock.patch.object(
            feeder,
            "cpu_submission_headroom",
            return_value=(0, 0, 0, 297),
        ), mock.patch.object(
            feeder,
            "submit",
            side_effect=[501, 502, RuntimeError("third submit failed")],
        ), mock.patch.object(feeder, "save_state") as save_state, mock.patch.object(
            feeder.time,
            "sleep",
        ):
            with self.assertRaisesRegex(RuntimeError, "third submit failed"):
                feeder._step_from_adopted_controller(
                    12_000,
                    authorization=auth,
                    target=300,
                    buffer=0,
                    solver_revision=SOLVER,
                    library_revision=LIBRARY,
                    candidate_seed=260710,
                    adoption_sha256=ADOPTION,
                    initial_count=250,
                    evidence_mode="preloaded250_v1",
                    strict_rows=25,
                    target_strict_rows=3_000,
                    journal=journal,
                    **RESOURCES,
                )

        self.assertTrue(journal["entered"])
        self.assertFalse(journal["completed"])
        self.assertEqual(journal["planned_count"], 3)
        self.assertEqual(journal["submitted_count"], 2)
        self.assertEqual(len(journal["events"]), 3)
        self.assertEqual(
            [event["task_id"] for event in journal["events"]],
            [501, 502, None],
        )
        self.assertEqual(
            [event["ledger_committed"] for event in journal["events"]],
            [True, True, False],
        )
        self.assertEqual(journal["events"][2]["exception_type"], "RuntimeError")
        self.assertFalse(journal["events"][2]["uncertain"])
        self.assertEqual(save_state.call_count, 2)


if __name__ == "__main__":
    unittest.main()
