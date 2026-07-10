import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAMPAIGN_DIR))

import provisional_wave  # noqa: E402


SOLVER = "a" * 40
LIBRARY = "b" * 40
PROFILE = {
    "cli_flags": "--thermal --headless",
    "param_overrides": dict(
        provisional_wave.scheduler_client.STANDARD_PROFILE_CONTRACT),
    "timeout_seconds": provisional_wave.scheduler_client.DEFAULT_TASK_TIMEOUT_SECONDS,
}


def fake_candidates(count):
    return [
        {"candidate": index, "N1_main": 7, "N1_side": 0}
        for index in range(count)
    ]


class ProvisionalWaveTests(unittest.TestCase):
    def build_plan(self, count=3):
        candidates = iter(fake_candidates(count))
        with mock.patch.object(
                provisional_wave.pinned_pilot, "cursor_after_valid_candidates",
                return_value=10), mock.patch.object(
                    provisional_wave.pinned_pilot, "next_valid_candidate",
                    side_effect=lambda cursor, seed: (
                        cursor + 1, cursor, next(candidates))):
            return provisional_wave.build_plan(
                SOLVER, LIBRARY, PROFILE, count=count)

    def test_plan_is_generation_specific_and_uses_300_unique_candidates(self):
        candidates = iter(fake_candidates(provisional_wave.TASK_COUNT))
        with mock.patch.object(
                provisional_wave.pinned_pilot, "cursor_after_valid_candidates",
                return_value=10) as cursor, mock.patch.object(
                    provisional_wave.pinned_pilot, "next_valid_candidate",
                    side_effect=lambda current, seed: (
                        current + 1, current, next(candidates))):
            records = provisional_wave.build_plan(SOLVER, LIBRARY, PROFILE)

        self.assertEqual(len(records), 300)
        self.assertEqual(len({record["name"] for record in records}), 300)
        self.assertEqual(len({record["dedupe_key"] for record in records}), 300)
        self.assertTrue(records[0]["name"].endswith("prov-000"))
        self.assertTrue(records[-1]["name"].endswith("prov-299"))
        cursor.assert_called_once_with(
            provisional_wave.pinned_pilot.PILOT_RESERVED_VALID_CANDIDATES,
            seed=provisional_wave.SEED)

    def test_default_cli_is_plan_only_and_never_calls_scheduler(self):
        records = self.build_plan(provisional_wave.TASK_COUNT)
        with mock.patch.object(
                provisional_wave, "build_plan", return_value=records), mock.patch.object(
                    provisional_wave.scheduler_client, "submit_verification") as submit, \
                mock.patch.object(provisional_wave.requests, "get") as get, \
                mock.patch.object(provisional_wave.requests, "post") as post:
            provisional_wave.main([
                "--solver-revision", SOLVER,
                "--library-revision", LIBRARY,
            ])
        submit.assert_not_called()
        get.assert_not_called()
        post.assert_not_called()

    def test_execute_requires_explicit_provisional_acknowledgement(self):
        with mock.patch.object(
                provisional_wave.scheduler_client, "submit_verification") as submit:
            with self.assertRaises(SystemExit):
                provisional_wave.main([
                    "--solver-revision", SOLVER,
                    "--library-revision", LIBRARY,
                    "--execute",
                ])
        submit.assert_not_called()

    @unittest.skipUnless(sys.platform == "win32", "Windows state path contract")
    def test_windows_default_manifest_uses_local_app_data(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(
                provisional_wave.os.environ,
                {"LOCALAPPDATA": td}, clear=False):
            provisional_wave.os.environ.pop(
                provisional_wave.STATE_DIR_ENV, None)
            path = provisional_wave.manifest_path(SOLVER, LIBRARY)

        self.assertEqual(
            path.parent,
            Path(td) / "MFT_1MW_2026" / "provisional_manifests")

    def test_manifest_state_env_override_has_priority(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(
                provisional_wave.os.environ,
                {provisional_wave.STATE_DIR_ENV: td}, clear=False):
            path = provisional_wave.manifest_path(SOLVER, LIBRARY)

        self.assertEqual(path.parent, Path(td).resolve())

    def test_resume_uses_dedupe_and_persists_each_exact_task_id(self):
        records = self.build_plan()
        expected = provisional_wave.new_manifest(
            SOLVER, LIBRARY, PROFILE, records)
        inventory = []
        submitted_ids = [101, 102, 103]

        def task_detail(path, params=None):
            if path == "/api/health":
                return {"status": "ok"}
            if path == "/api/tasks":
                return list(inventory)
            task_id = int(path.rsplit("/", 1)[-1])
            record = records[submitted_ids.index(task_id)]
            task = {
                "id": task_id,
                "name": record["name"],
                "dedupe_key": record["dedupe_key"],
                "remote_cwd": provisional_wave.scheduler_client.GPFS_RUNS_REMOTE_CWD,
                "status": "queued",
            }
            if not any(existing["id"] == task_id for existing in inventory):
                inventory.append(task)
            return task

        with tempfile.TemporaryDirectory() as td, mock.patch.object(
                provisional_wave, "build_plan", return_value=records), mock.patch.object(
                    provisional_wave, "_validate_local_revisions"), mock.patch.object(
                    provisional_wave, "_scheduler_json", side_effect=task_detail), \
                mock.patch.object(
                    provisional_wave.scheduler_client, "submit_verification",
                    side_effect=submitted_ids) as submit:
            path = Path(td) / "ledger.json"
            ledger = provisional_wave.submit_generation(
                SOLVER, LIBRARY, PROFILE, path, count=3)
            persisted = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(submit.call_count, 3)
        self.assertEqual([record["task_id"] for record in ledger["tasks"]], submitted_ids)
        self.assertEqual([record["task_id"] for record in persisted["tasks"]], submitted_ids)
        provisional_wave.validate_manifest(
            persisted, expected, require_all_task_ids=True)

    def test_inventory_rejects_same_prefix_with_wrong_dedupe(self):
        records = self.build_plan(1)
        task = {
            "id": 41,
            "name": records[0]["name"],
            "dedupe_key": "wrong",
            "remote_cwd": provisional_wave.scheduler_client.GPFS_RUNS_REMOTE_CWD,
        }
        with self.assertRaisesRegex(RuntimeError, "dedupe_key"):
            provisional_wave._validate_inventory([task], records)

    def test_cancel_posts_only_explicit_validated_ledger_ids(self):
        records = self.build_plan(3)
        manifest = provisional_wave.new_manifest(SOLVER, LIBRARY, PROFILE, records)
        for task_id, record in zip((501, 502, 503), manifest["tasks"]):
            record["task_id"] = task_id
        tasks = [
            {
                "id": record["task_id"],
                "name": record["name"],
                "dedupe_key": record["dedupe_key"],
                "remote_cwd": provisional_wave.scheduler_client.GPFS_RUNS_REMOTE_CWD,
                "status": status,
            }
            for record, status in zip(manifest["tasks"], ("running", "queued", "completed"))
        ]
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"cancelled": [501, 502], "count": 2}
        with mock.patch.object(
                provisional_wave, "_task_inventory", return_value=tasks), \
                mock.patch.object(
                    provisional_wave.requests, "post", return_value=response) as post:
            result = provisional_wave.cancel_generation(manifest, manifest)

        self.assertEqual(result, {"active": [501, 502], "cancelled": [501, 502]})
        _, kwargs = post.call_args
        self.assertEqual(kwargs["params"]["task_ids"], "501,502")
        self.assertEqual(
            kwargs["params"]["statuses"], "queued,attaching,running")
        self.assertNotIn("name_contains", kwargs["params"])

    def test_cancel_fails_closed_for_unledgered_scheduler_task(self):
        records = self.build_plan(1)
        manifest = provisional_wave.new_manifest(SOLVER, LIBRARY, PROFILE, records)
        task = {
            "id": 999,
            "name": records[0]["name"],
            "dedupe_key": records[0]["dedupe_key"],
            "remote_cwd": provisional_wave.scheduler_client.GPFS_RUNS_REMOTE_CWD,
            "status": "running",
        }
        with mock.patch.object(
                provisional_wave, "_task_inventory", return_value=[task]), \
                mock.patch.object(provisional_wave.requests, "post") as post:
            with self.assertRaisesRegex(RuntimeError, "absent from the exact ledger"):
                provisional_wave.cancel_generation(manifest, manifest)
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
