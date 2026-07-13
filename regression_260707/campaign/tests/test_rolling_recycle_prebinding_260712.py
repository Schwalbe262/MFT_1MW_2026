import copy
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock


import sys


CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAMPAIGN_DIR))

import _rolling_recycle_prebinding_260712 as rolling  # noqa: E402


class RollingRecycleLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.paths = mock.patch.multiple(
            rolling,
            ROOT=root,
            PLAN_PATH=root / "plan.json",
            LEDGER_PATH=root / "ledger.json",
            LEDGER_LOCK_PATH=root / "ledger.lock",
        )
        self.paths.start()
        candidate = {
            "task_id": 101,
            "name": "mft-camp-sb171c7c-le6b9b9d-101",
            "dedupe_key": (
                "mft-al:mft-camp-sb171c7c-le6b9b9d-101:"
                f"{rolling.SOLVER}:{rolling.LIBRARY}:test"),
            "started_at": "2026-07-12T06:00:00+00:00",
            "attached_at": "2026-07-12T05:59:00+00:00",
            "allocation_id": 8001,
            "slurm_job_id": "1",
            "account_name": "mft",
            "node_name": "n001",
        }
        unsigned = {
            "schema": "mft-prebinding-rolling-plan-v1",
            "created_at": "2026-07-12T07:00:00+00:00",
            "authorization": rolling.AUTHORIZATION,
            "authorization_sha256": rolling.AUTHORIZATION_SHA256,
            "candidate_count": 1,
            "candidates": [candidate],
        }
        self.plan = {**unsigned, "plan_sha256": rolling._sha(unsigned)}
        self.reviewed = mock.patch.object(
            rolling, "REVIEWED_ROLLING_PLAN_SHA256", self.plan["plan_sha256"])
        self.reviewed.start()
        rolling._write_once_verified(rolling.PLAN_PATH, self.plan)
        self.ledger = rolling._load_ledger(self.plan, create=True)
        self.entry = {
            **candidate,
            "sequence": 1,
            "phase": "cancel_preauthorized",
            "cancel_preauthorized_at": "2026-07-12T07:01:00+00:00",
            "active_ids_before": list(range(1, rolling.TARGET_ACTIVE + 1)),
        }

    def tearDown(self):
        self.reviewed.stop()
        self.paths.stop()
        self.temp.cleanup()

    def _save_entry(self):
        ledger = copy.deepcopy(self.ledger)
        ledger["entries"] = [copy.deepcopy(self.entry)]
        rolling._save_ledger(ledger)
        return rolling._load_ledger(self.plan, create=False)

    def test_exclusion_requires_both_authorization_and_live_cancelled_status(self):
        self._save_entry()
        base = {
            "id": 101,
            "project": rolling.scheduler_client.MFT_PROJECT,
            "name": self.entry["name"],
            "dedupe_key": self.entry["dedupe_key"],
            "started_at": self.entry["started_at"],
        }
        self.assertEqual(
            rolling.authorized_cancelled_task_ids([{**base, "status": "running"}]),
            set())
        self.assertEqual(
            rolling.authorized_cancelled_task_ids([{**base, "status": "failed"}]),
            set())
        self.assertEqual(
            rolling.authorized_cancelled_task_ids([{**base, "status": "cancelled"}]),
            {101})

    def test_duplicate_task_id_fails_closed_even_with_distinct_sequences(self):
        duplicate = copy.deepcopy(self.ledger)
        duplicate["entries"] = [
            copy.deepcopy(self.entry),
            {**copy.deepcopy(self.entry), "sequence": 2, "phase": "status_race"},
        ]
        duplicate["ledger_sha256"] = rolling._sha(
            rolling._ledger_seal_input(duplicate))
        with self.assertRaisesRegex(RuntimeError, "task ID is invalid/duplicate"):
            rolling.validate_ledger(duplicate, self.plan)

    def test_resealed_alternate_plan_is_not_the_reviewed_cohort(self):
        alternate = copy.deepcopy(self.plan)
        alternate["candidates"][0]["task_id"] = 102
        alternate["plan_sha256"] = rolling._sha({
            key: value for key, value in alternate.items()
            if key != "plan_sha256"
        })
        with self.assertRaisesRegex(RuntimeError, "reviewed sealed cohort"):
            rolling.validate_plan(alternate)

    def test_immutable_generation_commits_when_canonical_replace_is_denied(self):
        ledger = copy.deepcopy(self.ledger)
        ledger["entries"] = [copy.deepcopy(self.entry)]
        with mock.patch.object(
                rolling.durable.os, "replace",
                side_effect=PermissionError("simulated RaiDrive WinError 5")):
            rolling._save_ledger(ledger)
        loaded = rolling._load_ledger(self.plan, create=False)
        self.assertEqual(loaded["entries"], [self.entry])
        self.assertGreater(loaded["state_revision"], self.ledger["state_revision"])

    def test_ready_wait_tolerates_refill_deficit_but_requires_exact250(self):
        snapshots = [
            {"project_active": 244},
            {"project_active": rolling.TARGET_ACTIVE},
        ]
        sleeps = []
        with mock.patch.object(
                rolling, "_controller_health", return_value={"healthy": True}), \
                mock.patch.object(
                    rolling.scheduler_client, "campaign_mutation_lock",
                    side_effect=lambda: nullcontext()), \
                mock.patch.object(
                    rolling.scheduler_client, "live_project_submission_snapshot",
                    side_effect=snapshots):
            ready = rolling._wait_until_ready(60, sleeper=sleeps.append)
        self.assertEqual(ready["snapshot"]["project_active"], 250)
        self.assertEqual(sleeps, [5])

    def test_ready_wait_fails_closed_above_exact250(self):
        with mock.patch.object(
                rolling, "_controller_health", return_value={"healthy": True}), \
                mock.patch.object(
                    rolling.scheduler_client, "campaign_mutation_lock",
                    side_effect=lambda: nullcontext()), \
                mock.patch.object(
                    rolling.scheduler_client, "live_project_submission_snapshot",
                    return_value={"project_active": 251}):
            with self.assertRaisesRegex(RuntimeError, "exceeds exact250"):
                rolling._wait_until_ready(60, sleeper=lambda _: None)


if __name__ == "__main__":
    unittest.main()
