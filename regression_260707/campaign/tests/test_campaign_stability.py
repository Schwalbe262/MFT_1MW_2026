import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAMPAIGN_DIR))

import collect_wave  # noqa: E402
import feeder  # noqa: E402
import pinned_pilot  # noqa: E402


class DirectEntrypointTests(unittest.TestCase):
    def test_collect_wave_help_runs_from_regression_root(self):
        regression_root = CAMPAIGN_DIR.parent
        result = subprocess.run(
            [sys.executable, str(CAMPAIGN_DIR / "collect_wave.py"), "--help"],
            cwd=regression_root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--prefix", result.stdout)


class FakeResponse:
    def __init__(self, status_code=200, text="", payload=None, headers=None):
        self.status_code = status_code
        self.text = text
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        return self._payload


def project_capacity_gate(
        counts, hard_cap, queue_state="ready", ready_fit_slots=100,
        server_cap=300, queue_reason=None):
    active = sum(int(counts.get(status, 0) or 0) for status in feeder.ACTIVE_TASK_STATUSES)
    project_slots = max(0, min(server_cap - active, hard_cap - active))
    queue_allowed = queue_state != "blocked"
    return {
        "ready_fit_slots": ready_fit_slots,
        "queue_state": queue_state,
        "queue_reason": queue_reason or queue_state,
        "queue_submission_allowed": queue_allowed,
        "submission_allowed": queue_allowed and project_slots > 0,
        "project": pinned_pilot.MFT_PROJECT,
        "project_max_active_tasks": server_cap,
        "project_required_hard_cap": hard_cap,
        "project_counts": dict(counts),
        "project_active": active,
        "project_server_open_slots": max(0, server_cap - active),
        "project_stage_open_slots": max(0, hard_cap - active),
        "project_submission_slots": project_slots,
    }


class FeederTests(unittest.TestCase):
    def test_ready_marker_is_atomic_and_revision_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feeder.ready"
            feeder.publish_ready_marker(path, "a" * 40, "b" * 40)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["solver_revision"], "a" * 40)
            self.assertEqual(payload["library_revision"], "b" * 40)
            self.assertGreater(payload["pid"], 0)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_direct_and_cli_feeder_cannot_enter_above_fifty(self):
        self.assertLessEqual(
            feeder.TARGET_ACTIVE + feeder.BUFFER,
            feeder.MAX_STANDALONE_ACTIVE,
        )
        with mock.patch.object(feeder, "scheduler_snapshot") as snapshot:
            with self.assertRaisesRegex(
                    feeder.SchedulerError, "only rapid_campaign"):
                feeder.step(
                    1000, target=300, buffer=0,
                    solver_revision="a" * 40, library_revision="b" * 40)
        snapshot.assert_not_called()

        argv = [
            "feeder.py", "--once", "--target", "51", "--buffer", "0",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
                feeder.al_driver, "_current_solver_revision") as current:
            with self.assertRaisesRegex(
                    feeder.SchedulerError, "standalone feeder hard cap"):
                feeder.main()
        current.assert_not_called()

        with self.assertRaises(TypeError):
            feeder.step(
                1000, target=300, buffer=0,
                solver_revision="a" * 40, library_revision="b" * 40,
                _rapid_authorized=True,
            )

    def test_unjudged_terminal_and_legacy_active_outputs_remain_reserved(self):
        state = {
            "outstanding": [3, 4],
            "task_expected_rows": {"4": 1},
        }
        tasks = [
            {"id": 1, "name": "mft-camp-c-legacy", "status": "running"},
            {"id": 2, "name": "mft-camp-sabc-ldef-2", "status": "completed"},
            {"id": 3, "name": "mft-camp-sabc-ldef-3", "status": "completed"},
        ]

        reserved = feeder.reserved_unjudged_rows(state, tasks, judged_ids={3})

        self.assertEqual(reserved, 10)  # legacy 8 + completed 1 + ledger-only 1

    def test_dataset_snapshot_recovers_valid_collector_staging_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            train = Path(directory) / "train.parquet"
            cache = Path(directory) / "collect_cache.json"
            pd.DataFrame([{"value": 1}]).to_parquet(train, index=False)
            Path(str(cache) + ".tmp.tmp").write_text(json.dumps({
                "harvested": [7], "nodata": [8], "local_parts": [],
            }), encoding="utf-8")

            with mock.patch.object(feeder, "TRAIN_PARQUET", str(train)), \
                    mock.patch.object(feeder, "COLLECT_CACHE", str(cache)):
                rows, judged = feeder.dataset_collection_snapshot()

        self.assertEqual(rows, 1)
        self.assertEqual(judged, {7, 8})

    def test_dataset_snapshot_fails_closed_when_existing_cache_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            train = Path(directory) / "train.parquet"
            cache = Path(directory) / "collect_cache.json"
            pd.DataFrame([{"value": 1}]).to_parquet(train, index=False)

            with mock.patch.object(feeder, "TRAIN_PARQUET", str(train)), \
                    mock.patch.object(feeder, "COLLECT_CACHE", str(cache)):
                with self.assertRaisesRegex(
                        feeder.SchedulerError, "canonical cache is missing"):
                    feeder.dataset_collection_snapshot()

    def test_pinned_pilot_capacity_counts_global_tasks_and_warm_cpus(self):
        statuses = {"queued": 8, "attaching": 1, "running": 142}
        allocations = [
            {"state": "active", "resource_pool": "cpu", "total_cpus": 400, "free_cpus": 400},
            {"state": "warm", "resource_pool": "cpu", "total_cpus": 400, "free_cpus": 400},
        ]

        snapshot = pinned_pilot.calculate_submission_headroom(
            statuses, allocations, ready_fit_slots=24)

        self.assertEqual(snapshot["global_active"], 151)
        self.assertEqual(snapshot["total_slots"], 170)
        self.assertEqual(snapshot["headroom"], 19)
        self.assertTrue(snapshot["submission_allowed"])

    def test_pinned_pilot_opening_queue_allows_demand_without_ready_slots(self):
        snapshot = pinned_pilot.calculate_submission_headroom(
            {"queued": 0, "attaching": 0, "running": 100},
            [],
            ready_fit_slots=0,
            queue_state="opening",
            queue_reason="opening demand pools",
        )

        self.assertEqual(snapshot["headroom"], 0)
        self.assertEqual(snapshot["queue_state"], "opening")
        self.assertTrue(snapshot["submission_allowed"])

    def test_pinned_pilot_blocked_queue_refuses_demand(self):
        snapshot = pinned_pilot.calculate_submission_headroom(
            {}, [], ready_fit_slots=0, queue_state="blocked",
            queue_reason="allocation backoff active for cpu",
        )

        self.assertFalse(snapshot["submission_allowed"])

    def test_queue_contract_allows_ready_pending_and_opening_only(self):
        for queue_state in ("ready", "pending", "opening"):
            with self.subTest(queue_state=queue_state):
                self.assertTrue(
                    pinned_pilot.queue_allows_demand_submission(queue_state))
        self.assertFalse(
            pinned_pilot.queue_allows_demand_submission("blocked"))
        with self.assertRaisesRegex(RuntimeError, "invalid queue_state"):
            pinned_pilot.queue_allows_demand_submission("unknown")

    def test_mft_project_inventory_enforces_server_and_stage_caps(self):
        projects = [{
            "name": pinned_pilot.MFT_PROJECT,
            "max_active_tasks": 300,
            "auto_pull": False,
        }]
        tasks = [
            {"id": index + 1, "name": f"candidate-{index}",
             "project": pinned_pilot.MFT_PROJECT, "status": "queued"}
            for index in range(40)
        ] + [
            {"id": index + 41, "name": f"candidate-{index + 40}",
             "project": pinned_pilot.MFT_PROJECT, "status": "running"}
            for index in range(10)
        ]

        snapshot = pinned_pilot.project_submission_snapshot(
            projects, tasks, required_hard_cap=50)

        self.assertEqual(snapshot["project_active"], 50)
        self.assertEqual(snapshot["project_stage_open_slots"], 0)
        self.assertEqual(snapshot["project_submission_slots"], 0)
        reduced = pinned_pilot.project_submission_snapshot(
                projects=[{
                    "name": pinned_pilot.MFT_PROJECT,
                    "max_active_tasks": 100,
                    "auto_pull": False,
                }],
                project_tasks=[],
                required_hard_cap=300,
            )
        self.assertEqual(reduced["project_max_active_tasks"], 100)
        self.assertEqual(reduced["project_submission_slots"], 100)
        with self.assertRaisesRegex(RuntimeError, "missing or ambiguous"):
            pinned_pilot.project_submission_snapshot(
                projects=[], project_tasks=[], required_hard_cap=10)

    def test_project_capacity_unions_legacy_active_without_double_count(self):
        projects = [{
            "name": pinned_pilot.MFT_PROJECT,
            "max_active_tasks": 300,
            "auto_pull": False,
        }]
        tagged = [
            {"id": 1, "name": "mft-camp-new-1", "project": pinned_pilot.MFT_PROJECT,
             "status": "queued"},
            {"id": 2, "name": "candidate-al", "project": pinned_pilot.MFT_PROJECT,
             "status": "running"},
        ]
        legacy_scan = [
            dict(tagged[0]),
            {"id": 3, "name": "mft-camp-old-3", "project": "",
             "status": "attaching"},
        ]

        snapshot = pinned_pilot.project_submission_snapshot(
            projects, tagged, required_hard_cap=10,
            legacy_tasks=legacy_scan)

        self.assertEqual(snapshot["project_active"], 3)
        self.assertEqual(snapshot["project_tagged_active"], 2)
        self.assertEqual(snapshot["legacy_active"], 1)
        self.assertEqual(snapshot["project_submission_slots"], 7)
        self.assertEqual(snapshot["project_counts"], {
            "queued": 1, "attaching": 1, "running": 1})

    def test_project_capacity_rejects_duplicate_and_foreign_legacy_rows(self):
        projects = [{
            "name": pinned_pilot.MFT_PROJECT,
            "max_active_tasks": 300,
            "auto_pull": False,
        }]
        row = {"id": 1, "name": "mft-camp-old", "project": "",
               "status": "queued"}
        with self.assertRaisesRegex(RuntimeError, "duplicate legacy MFT task ID"):
            pinned_pilot.project_submission_snapshot(
                projects, [], required_hard_cap=10,
                legacy_tasks=[row, dict(row)])
        with self.assertRaisesRegex(RuntimeError, "unexpected project"):
            pinned_pilot.project_submission_snapshot(
                projects, [], required_hard_cap=10,
                legacy_tasks=[{
                    **row, "id": 2, "project": "IPMSM"}])
        with self.assertRaisesRegex(RuntimeError, "exceeds absolute cap 300"):
            pinned_pilot.project_submission_snapshot(
                projects, [], required_hard_cap=301)

    def test_campaign_inventory_preserves_project_and_projectless_legacy_union(self):
        rows = [
            {"id": 1, "name": "mft-camp-new", "project": pinned_pilot.MFT_PROJECT,
             "status": "running"},
            {"id": 2, "name": "mft-camp-legacy", "project": "",
             "status": "completed"},
        ]
        with mock.patch.object(feeder, "_scheduler_json", return_value=rows) as get:
            self.assertEqual(feeder.campaign_inventory(), rows)
        self.assertEqual(get.call_args.kwargs["params"], {
            "compact": True,
            "limit": feeder.CAMPAIGN_INVENTORY_PAGE_SIZE,
            "name_prefix": feeder.CAMPAIGN_PREFIX,
        })

        with mock.patch.object(feeder, "_scheduler_json", return_value=[{
                "id": 3, "name": "mft-camp-foreign", "project": "IPMSM",
                "status": "running"}]):
            with self.assertRaisesRegex(feeder.SchedulerError, "unexpected project"):
                feeder.campaign_inventory()

    def test_campaign_inventory_pages_all_rows_for_exact_reservations(self):
        pages = [
            [
                {"id": 5, "name": "mft-camp-sx-ly-5", "project": "MFT_1MW_2026v1",
                 "status": "completed"},
                {"id": 4, "name": "mft-camp-sx-ly-4", "project": "",
                 "status": "running"},
            ],
            [
                {"id": 3, "name": "mft-camp-sx-ly-3", "project": "MFT_1MW_2026v1",
                 "status": "failed"},
                {"id": 2, "name": "mft-camp-sx-ly-2", "project": "MFT_1MW_2026v1",
                 "status": "completed"},
            ],
            [
                {"id": 1, "name": "mft-camp-c-legacy", "project": "",
                 "status": "completed"},
            ],
        ]
        with mock.patch.object(feeder, "CAMPAIGN_INVENTORY_PAGE_SIZE", 2), \
                mock.patch.object(feeder, "_scheduler_json", side_effect=pages) as get:
            tasks = feeder.campaign_inventory()

        self.assertEqual([task["id"] for task in tasks], [5, 4, 3, 2, 1])
        self.assertEqual(get.call_args_list[0].kwargs["params"], {
            "compact": True, "limit": 2, "name_prefix": feeder.CAMPAIGN_PREFIX,
        })
        self.assertEqual(get.call_args_list[1].kwargs["params"]["before_id"], 4)
        self.assertEqual(get.call_args_list[2].kwargs["params"]["before_id"], 2)
        self.assertEqual(
            feeder.reserved_unjudged_rows(
                {"outstanding": [], "task_expected_rows": {}},
                tasks,
                judged_ids={2, 3},
            ),
            10,
        )  # IDs 4 and 5 reserve one row each; legacy ID 1 reserves eight.

    def test_campaign_inventory_fails_closed_if_scheduler_ignores_cursor(self):
        repeated = [
            {"id": 2, "name": "mft-camp-sx-ly-2", "project": "MFT_1MW_2026v1",
             "status": "running"},
            {"id": 1, "name": "mft-camp-sx-ly-1", "project": "MFT_1MW_2026v1",
             "status": "running"},
        ]
        with mock.patch.object(feeder, "CAMPAIGN_INVENTORY_PAGE_SIZE", 2), \
                mock.patch.object(feeder, "_scheduler_json", side_effect=[repeated, repeated]):
            with self.assertRaisesRegex(
                    feeder.SchedulerError, "invalid/duplicate campaign task ID"):
                feeder.campaign_inventory()

    def test_scheduler_snapshot_isolates_mft_from_ipmsm_and_allows_opening(self):
        calls = []

        def scheduler_json(path, params=None):
            calls.append((path, params))
            if path == "/api/tasks/summary":
                return {"statuses": {"running": 100}}
            if path == "/api/allocations":
                return []
            if path == "/api/projects":
                return [{
                    "name": pinned_pilot.MFT_PROJECT,
                    "max_active_tasks": 300,
                    "auto_pull": False,
                }]
            if path == "/api/tasks":
                return []
            if path == "/api/task-capacity":
                return {
                    "ready_fit_slots": 0,
                    "queue_state": "opening",
                    "queue_reason": "opening demand pools",
                }
            raise AssertionError(path)

        with mock.patch.object(
                feeder, "_scheduler_json", side_effect=scheduler_json):
            project_counts, global_counts, _allocations, capacity = (
                feeder.scheduler_snapshot(required_hard_cap=300))

        self.assertEqual(sum(project_counts.values()), 0)
        self.assertEqual(global_counts["running"], 100)
        self.assertEqual(capacity["project_submission_slots"], 300)
        self.assertTrue(capacity["submission_allowed"])
        self.assertIn(("/api/projects", None), calls)
        self.assertIn((
            "/api/tasks",
            {
                "limit": 10000,
                "project": pinned_pilot.MFT_PROJECT,
                "status": "queued,attaching,running",
            },
        ), calls)
        self.assertIn((
            "/api/tasks",
            {
                "limit": 10000,
                "name_prefix": pinned_pilot.LEGACY_MFT_NAME_PREFIX,
                "status": "queued,attaching,running",
            },
        ), calls)

    def test_pinned_capacity_snapshot_requires_the_mft_project(self):
        calls = []

        def scheduler_json(path, params=None):
            calls.append((path, params))
            if path == "/api/tasks/summary":
                return {"statuses": {"running": 100}}
            if path == "/api/allocations":
                return []
            if path == "/api/projects":
                return [{
                    "name": pinned_pilot.MFT_PROJECT,
                    "max_active_tasks": 300,
                    "auto_pull": False,
                }]
            if path == "/api/tasks":
                return []
            if path == "/api/task-capacity":
                return {
                    "ready_fit_slots": 0,
                    "queue_state": "opening",
                    "queue_reason": "opening demand pools",
                }
            raise AssertionError(path)

        with mock.patch.object(
                pinned_pilot, "_get_json", side_effect=scheduler_json):
            snapshot = pinned_pilot.capacity_snapshot()

        self.assertEqual(
            snapshot["project_required_hard_cap"],
            pinned_pilot.PILOT_PROJECT_HARD_CAP,
        )
        self.assertEqual(snapshot["project_submission_slots"], 10)
        self.assertTrue(snapshot["submission_allowed"])
        self.assertIn(("/api/projects", None), calls)
        self.assertIn((
            "/api/tasks",
            {
                "limit": 10000,
                "name_prefix": pinned_pilot.LEGACY_MFT_NAME_PREFIX,
                "status": "queued,attaching,running",
            },
        ), calls)

    def test_refill_uses_campaign_demand_not_global_ready_capacity(self):
        state = {
            "serial": 100,
            "submitted_samples": 0,
            "outstanding": list(range(500)),
        }
        counts = {"queued": 1, "attaching": 1, "running": 2}
        allocations = [
            {
                "state": "active",
                "resource_pool": "cpu",
                "total_cpus": 64,
                "free_cpus": 32,
            },
            {
                "state": "pending",
                "resource_pool": "cpu",
                "total_cpus": 256,
                "free_cpus": 256,
            },
        ]

        with mock.patch.object(feeder, "load_state", return_value=state), mock.patch.object(
            feeder, "scheduler_snapshot", return_value=(
                counts, counts, allocations,
                project_capacity_gate(counts, hard_cap=10),
            )
        ), mock.patch.object(
            feeder, "dataset_row_count", return_value=0
        ), mock.patch.object(
            feeder, "dataset_collection_snapshot", return_value=(0, set(range(500)))
        ), mock.patch.object(
            feeder, "campaign_inventory", return_value=[]
        ), mock.patch.object(
            feeder, "cursor_after_valid_candidates", return_value=0
        ), mock.patch.object(
            feeder, "next_valid_candidate", return_value=(1, 0, {"candidate": 1})
        ), mock.patch.object(
            feeder, "submit", side_effect=[901, 902, 903, 904, 905, 906]
        ) as submit_mock, mock.patch.object(
            feeder, "save_state"
        ) as save_mock, mock.patch.object(
            feeder.time, "sleep"
        ):
            self.assertTrue(feeder.step(
                1000, target=10, buffer=0,
                solver_revision="a" * 40, library_revision="b" * 40))

        # The MFT campaign owns its target independently of unrelated tasks.
        self.assertEqual(submit_mock.call_count, 6)
        self.assertEqual(state["serial"], 106)
        self.assertEqual(state["submitted_samples"], 6)
        self.assertEqual(save_mock.call_count, 6)

    def test_target_plus_buffer_is_an_absolute_hard_cap(self):
        counts = {"queued": 2, "attaching": 0, "running": 0}
        allocations = [
            {
                "state": "active",
                "resource_pool": "cpu",
                "total_cpus": 128,
                "free_cpus": 128,
            }
        ]
        with mock.patch.object(
            feeder,
            "load_state",
            return_value={"serial": 0, "submitted_samples": 0},
        ), mock.patch.object(
            feeder, "scheduler_snapshot", return_value=(
                counts, counts, allocations,
                project_capacity_gate(counts, hard_cap=3),
            )
        ) as scheduler_snapshot, mock.patch.object(
            feeder, "dataset_row_count", return_value=0
        ), mock.patch.object(
            feeder, "dataset_collection_snapshot", return_value=(0, set())
        ), mock.patch.object(
            feeder, "campaign_inventory", return_value=[]
        ), mock.patch.object(
            feeder, "cursor_after_valid_candidates", return_value=0
        ), mock.patch.object(
            feeder, "next_valid_candidate", return_value=(1, 0, {"candidate": 1})
        ), mock.patch.object(
            feeder, "submit", return_value=42
        ) as submit_mock, mock.patch.object(
            feeder, "save_state"
        ), mock.patch.object(
            feeder.time, "sleep"
        ):
            feeder.step(
                1000, target=2, buffer=1,
                solver_revision="a" * 40, library_revision="b" * 40)
        self.assertEqual(submit_mock.call_count, 1)
        scheduler_snapshot.assert_called_once_with(
            3,
            require_exact_project_cap=False,
            require_full_project=False,
        )

    def test_opening_queue_submits_mft_deficit_to_trigger_scale_out(self):
        state = {"serial": 0, "submitted_samples": 0}
        campaign_counts = {"queued": 0, "attaching": 0, "running": 0}
        global_counts = {"queued": 0, "attaching": 0, "running": 100}
        capacity_gate = project_capacity_gate(
            campaign_counts,
            hard_cap=3,
            queue_state="opening",
            ready_fit_slots=0,
            queue_reason="no single ready pool has 4 free CPUs; opening demand pools",
        )
        with mock.patch.object(feeder, "load_state", return_value=state), mock.patch.object(
            feeder, "scheduler_snapshot",
            return_value=(campaign_counts, global_counts, [], capacity_gate),
        ), mock.patch.object(
            feeder, "dataset_collection_snapshot", return_value=(0, set())
        ), mock.patch.object(
            feeder, "campaign_inventory", return_value=[]
        ), mock.patch.object(
            feeder, "cursor_after_valid_candidates", return_value=0
        ), mock.patch.object(
            feeder, "next_valid_candidate", return_value=(1, 0, {"candidate": 1})
        ), mock.patch.object(
            feeder, "submit", side_effect=[901, 902, 903]
        ) as submit_mock, mock.patch.object(
            feeder, "save_state"
        ), mock.patch.object(feeder.time, "sleep"):
            self.assertTrue(feeder.step(
                1000, target=3, buffer=0,
                solver_revision="a" * 40, library_revision="b" * 40))

        self.assertEqual(submit_mock.call_count, 3)
        self.assertEqual(state["submitted_samples"], 3)

    def test_blocked_queue_is_the_only_capacity_state_that_stops_refill(self):
        campaign_counts = {"queued": 0, "attaching": 0, "running": 0}
        global_counts = {"queued": 0, "attaching": 0, "running": 100}
        capacity_gate = project_capacity_gate(
            campaign_counts,
            hard_cap=3,
            queue_state="blocked",
            ready_fit_slots=0,
            queue_reason="allocation backoff active for cpu",
        )
        with mock.patch.object(
            feeder, "load_state", return_value={"serial": 0, "submitted_samples": 0}
        ), mock.patch.object(
            feeder, "scheduler_snapshot",
            return_value=(campaign_counts, global_counts, [], capacity_gate),
        ), mock.patch.object(
            feeder, "dataset_collection_snapshot", return_value=(0, set())
        ), mock.patch.object(
            feeder, "campaign_inventory", return_value=[]
        ), mock.patch.object(feeder, "submit") as submit_mock:
            self.assertTrue(feeder.step(
                1000, target=3, buffer=0,
                solver_revision="a" * 40, library_revision="b" * 40))

        submit_mock.assert_not_called()

    def test_none_submission_does_not_advance_and_retry_uses_same_candidate(self):
        state = {
            "serial": 7,
            "submitted_samples": 0,
            "candidate_generation": f"{'a' * 40}:{'b' * 40}:seed260710",
            "candidate_cursor": 10,
        }
        counts = {"queued": 0, "attaching": 0, "running": 0}
        with mock.patch.object(feeder, "load_state", return_value=state), mock.patch.object(
            feeder, "scheduler_snapshot", return_value=(
                counts, counts, [], project_capacity_gate(counts, hard_cap=1))
        ), mock.patch.object(
            feeder, "dataset_collection_snapshot", return_value=(0, set())
        ), mock.patch.object(
            feeder, "campaign_inventory", return_value=[]
        ), mock.patch.object(
            feeder, "next_valid_candidate",
            return_value=(11, 10, {"candidate": "same"})
        ) as candidate, mock.patch.object(
            feeder, "submit", side_effect=[None, 901]
        ) as submit, mock.patch.object(
            feeder, "save_state"
        ) as save, mock.patch.object(feeder.time, "sleep"):
            with self.assertRaisesRegex(
                    feeder.SchedulerError, "candidate state was not advanced"):
                feeder.step(
                    1000, target=1, buffer=0,
                    solver_revision="a" * 40, library_revision="b" * 40)
            self.assertEqual(state, {
                "serial": 7,
                "submitted_samples": 0,
                "candidate_generation": f"{'a' * 40}:{'b' * 40}:seed260710",
                "candidate_cursor": 10,
            })
            save.assert_not_called()

            self.assertTrue(feeder.step(
                1000, target=1, buffer=0,
                solver_revision="a" * 40, library_revision="b" * 40))

        self.assertEqual(candidate.call_count, 2)
        self.assertEqual(
            [call.args[0] for call in candidate.call_args_list], [10, 10])
        self.assertEqual(
            [call.args[0] for call in submit.call_args_list],
            ["mft-camp-saaaaaaa-lbbbbbbb-00008"] * 2,
        )
        self.assertEqual(state["serial"], 8)
        self.assertEqual(state["candidate_cursor"], 11)
        self.assertEqual(state["outstanding"], [901])
        save.assert_called_once()

    def test_returning_to_an_earlier_seed_resumes_its_own_cursor(self):
        solver = "a" * 40
        library = "b" * 40
        generation_one = f"{solver}:{library}:seed1"
        generation_two = f"{solver}:{library}:seed2"
        state = {
            "serial": 20,
            "submitted_samples": 20,
            "candidate_generation": generation_two,
            "candidate_cursor": 21,
            "candidate_cursors": {
                generation_one: 11,
                generation_two: 21,
            },
        }
        counts = {"queued": 0, "attaching": 0, "running": 0}
        with mock.patch.object(feeder, "load_state", return_value=state), \
                mock.patch.object(feeder, "scheduler_snapshot", return_value=(
                    counts, counts, [], project_capacity_gate(counts, hard_cap=1))), \
                mock.patch.object(
                    feeder, "dataset_collection_snapshot", return_value=(0, set())), \
                mock.patch.object(feeder, "campaign_inventory", return_value=[]), \
                mock.patch.object(feeder, "cursor_after_valid_candidates") as initial, \
                mock.patch.object(
                    feeder, "next_valid_candidate",
                    return_value=(12, 11, {"candidate": "seed-one-next"})) as candidate, \
                mock.patch.object(feeder, "submit", return_value=902), \
                mock.patch.object(feeder, "save_state"), \
                mock.patch.object(feeder.time, "sleep"):
            self.assertTrue(feeder.step(
                1000, target=1, buffer=0,
                solver_revision=solver, library_revision=library,
                candidate_seed=1,
            ))

        initial.assert_not_called()
        candidate.assert_called_once_with(11, seed=1)
        self.assertEqual(state["candidate_cursors"], {
            generation_one: 12,
            generation_two: 21,
        })

    def test_submission_exception_rolls_back_unsaved_candidate_state(self):
        state = {"serial": 2, "submitted_samples": 0}
        original = dict(state)
        counts = {"queued": 0, "attaching": 0, "running": 0}
        with mock.patch.object(feeder, "load_state", return_value=state), mock.patch.object(
            feeder, "scheduler_snapshot", return_value=(
                counts, counts, [], project_capacity_gate(counts, hard_cap=1))
        ), mock.patch.object(
            feeder, "dataset_collection_snapshot", return_value=(0, set())
        ), mock.patch.object(
            feeder, "campaign_inventory", return_value=[]
        ), mock.patch.object(
            feeder, "cursor_after_valid_candidates", return_value=10
        ), mock.patch.object(
            feeder, "next_valid_candidate", return_value=(11, 10, {"candidate": 1})
        ), mock.patch.object(
            feeder, "submit", side_effect=RuntimeError("uncertain POST")
        ), mock.patch.object(feeder, "save_state") as save:
            with self.assertRaisesRegex(RuntimeError, "uncertain POST"):
                feeder.step(
                    1000, target=1, buffer=0,
                    solver_revision="a" * 40, library_revision="b" * 40)

        self.assertEqual(state, original)
        save.assert_not_called()

    def test_dataset_rows_not_submission_ledger_bound_total(self):
        state = {"serial": 10, "submitted_samples": 12000}
        counts = {"queued": 0, "attaching": 0, "running": 0}
        allocations = [{
            "state": "active", "resource_pool": "cpu",
            "total_cpus": 64, "free_cpus": 64,
        }]
        with mock.patch.object(feeder, "load_state", return_value=state), mock.patch.object(
            feeder, "scheduler_snapshot", return_value=(
                counts, counts, allocations,
                project_capacity_gate(
                    counts, hard_cap=10, ready_fit_slots=20),
            )
        ), mock.patch.object(
            feeder, "dataset_row_count", return_value=998
        ), mock.patch.object(
            feeder, "dataset_collection_snapshot", return_value=(998, set())
        ), mock.patch.object(
            feeder, "campaign_inventory", return_value=[]
        ), mock.patch.object(
            feeder, "cursor_after_valid_candidates", return_value=0
        ), mock.patch.object(
            feeder, "next_valid_candidate", return_value=(1, 0, {"candidate": 1})
        ), mock.patch.object(
            feeder, "submit", side_effect=[901, 902]
        ) as submit_mock, mock.patch.object(
            feeder, "save_state"
        ), mock.patch.object(feeder.time, "sleep"):
            self.assertTrue(feeder.step(
                1000, target=10, buffer=0,
                solver_revision="a" * 40, library_revision="b" * 40))

        self.assertEqual(submit_mock.call_count, 2)

    def test_active_tasks_reserve_projected_dataset_rows(self):
        counts = {"queued": 1, "attaching": 0, "running": 1}
        allocations = [{
            "state": "active", "resource_pool": "cpu",
            "total_cpus": 64, "free_cpus": 64,
        }]
        with mock.patch.object(
            feeder, "load_state", return_value={"serial": 0, "submitted_samples": 0}
        ), mock.patch.object(
            feeder, "scheduler_snapshot", return_value=(
                counts, counts, allocations,
                project_capacity_gate(
                    counts, hard_cap=10, ready_fit_slots=20),
            )
        ), mock.patch.object(
            feeder, "dataset_row_count", return_value=998
        ), mock.patch.object(
            feeder, "dataset_collection_snapshot", return_value=(998, set())
        ), mock.patch.object(
            feeder, "campaign_inventory", return_value=[
                {"id": 1, "name": "mft-camp-srev-lrev-1", "status": "queued"},
                {"id": 2, "name": "mft-camp-srev-lrev-2", "status": "running"},
            ]
        ), mock.patch.object(feeder, "submit") as submit_mock:
            self.assertTrue(feeder.step(
                1000, target=10, buffer=0,
                solver_revision="a" * 40, library_revision="b" * 40))

        submit_mock.assert_not_called()


class CollectorFetchTests(unittest.TestCase):
    def test_next_collector_process_uses_scheduler_environment(self):
        with mock.patch.dict(
                "os.environ", {"MFT_SCHEDULER_URL": "http://127.0.0.1:8001/"}):
            self.assertEqual(
                collect_wave._configured_scheduler_url(),
                "http://127.0.0.1:8001",
            )

    def test_local_transport_failure_activates_8001_for_following_requests(self):
        responses = [
            collect_wave.requests.ConnectionError("legacy listener unavailable"),
            FakeResponse(200, text="recovered"),
            FakeResponse(200, text="next"),
        ]
        with mock.patch.object(
                collect_wave, "SCHEDULER", collect_wave.DEFAULT_SCHEDULER), \
                mock.patch.object(
                    collect_wave.requests, "get", side_effect=responses
                ) as get_mock, mock.patch.object(
                    collect_wave.time, "sleep"
                ) as sleep_mock:
            self.assertEqual(
                collect_wave._get_response("/api/health").text,
                "recovered",
            )
            self.assertEqual(
                collect_wave._get_response("/api/tasks").text,
                "next",
            )

        self.assertEqual([call.args[0] for call in get_mock.call_args_list], [
            "http://127.0.0.1:8000/api/health",
            "http://127.0.0.1:8001/api/health",
            "http://127.0.0.1:8001/api/tasks",
        ])
        sleep_mock.assert_not_called()

    def test_explicit_remote_scheduler_never_falls_back_to_loopback(self):
        with mock.patch.object(
                collect_wave, "SCHEDULER", "https://scheduler.example.test"), \
                mock.patch.object(
                    collect_wave.requests,
                    "get",
                    side_effect=collect_wave.requests.ConnectionError("unavailable"),
                ) as get_mock, mock.patch.object(
                    collect_wave.time, "sleep"
                ):
            with self.assertRaises(collect_wave.FetchError):
                collect_wave._get_response("/api/health", attempts=2)

        self.assertEqual(get_mock.call_count, 2)
        self.assertTrue(all(
            call.args[0] == "https://scheduler.example.test/api/health"
            for call in get_mock.call_args_list
        ))

    def test_429_and_5xx_are_retried_with_bounded_backoff(self):
        responses = [
            FakeResponse(429, headers={"Retry-After": "0.1"}),
            FakeResponse(503),
            FakeResponse(200, text="ready"),
        ]
        with mock.patch.object(
            collect_wave.requests, "get", side_effect=responses
        ) as get_mock, mock.patch.object(collect_wave.time, "sleep") as sleep_mock:
            response = collect_wave._get_response("/api/test")

        self.assertEqual(response.text, "ready")
        self.assertEqual(get_mock.call_count, 3)
        self.assertEqual(sleep_mock.call_count, 2)
        self.assertEqual(sleep_mock.call_args_list[0].args[0], 0.1)

    def test_non_retryable_4xx_fails_immediately(self):
        with mock.patch.object(
            collect_wave.requests, "get", return_value=FakeResponse(404)
        ) as get_mock, mock.patch.object(collect_wave.time, "sleep") as sleep_mock:
            with self.assertRaises(collect_wave.FetchError):
                collect_wave._get_response("/api/missing")
        self.assertEqual(get_mock.call_count, 1)
        sleep_mock.assert_not_called()

    def test_compact_cursor_pagination_never_probes_task_details(self):
        first_page = [
            {
                "id": task_id,
                "name": f"mft-camp-{task_id}",
                "status": "completed",
                "project": "MFT_1MW_2026v1",
                "started_at": "2026-07-16 00:00:00",
            }
            for task_id in range(4000, 4000 - collect_wave.TASK_LIST_LIMIT, -1)
        ]
        last_task = {
            "id": 1999,
            "name": "mft-camp-1999",
            "status": "running",
            "project": "MFT_1MW_2026v1",
            "started_at": "2026-07-16 00:00:00",
        }
        calls = []

        def get_json(path, **kwargs):
            calls.append((path, kwargs))
            self.assertEqual(path, "/api/tasks")
            if len(calls) == 1:
                return first_page
            if len(calls) == 2:
                return [last_task]
            raise AssertionError("unexpected scheduler page")

        with mock.patch.object(
                collect_wave, "_get_json", side_effect=get_json):
            tasks = collect_wave.list_tasks("mft-camp")

        self.assertEqual(len(tasks), collect_wave.TASK_LIST_LIMIT + 1)
        self.assertEqual(tasks[-1], last_task)
        self.assertEqual(calls[0], (
            "/api/tasks",
            {
                "params": {
                    "compact": "true",
                    "limit": collect_wave.TASK_LIST_LIMIT,
                    "name_prefix": "mft-camp",
                },
                "timeout": 30,
            },
        ))
        self.assertEqual(
            calls[1][1]["params"]["before_id"],
            min(task["id"] for task in first_page),
        )

    def test_compact_cursor_fails_closed_when_page_does_not_advance(self):
        page = [
            {"id": task_id, "name": f"mft-camp-{task_id}"}
            for task_id in range(3000, 3000 - collect_wave.TASK_LIST_LIMIT, -1)
        ]
        with mock.patch.object(
                collect_wave, "_get_json", return_value=page) as get_json:
            with self.assertRaisesRegex(
                    collect_wave.FetchError, "cursor did not advance"):
                collect_wave.list_tasks("mft-camp")

        self.assertEqual(get_json.call_count, 2)

    def test_empty_prefix_is_rejected_without_scheduler_request(self):
        with mock.patch.object(collect_wave, "_get_json") as get_json:
            with self.assertRaisesRegex(ValueError, "prefix must be non-empty"):
                collect_wave.list_tasks("")
        get_json.assert_not_called()


class CollectorDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.dataset_dir = self.tempdir.name
        self.local_results_csv = Path(self.dataset_dir) / "local-results.csv"
        self.local_parts_dir = Path(self.dataset_dir) / "local-parts"
        self.source_rank_path = Path(self.dataset_dir) / "source_ranks.parquet"
        self.patches = [
            mock.patch.object(collect_wave, "DATASET_DIR", self.dataset_dir),
            mock.patch.object(
                collect_wave, "CACHE_PATH", str(Path(self.dataset_dir) / "collect_cache.json")
            ),
            mock.patch.object(
                collect_wave,
                "LOCAL_RESULTS_CSV",
                str(self.local_results_csv),
            ),
            mock.patch.object(
                collect_wave,
                "LOCAL_RESULTS_PARTS_DIR",
                str(self.local_parts_dir),
            ),
        ]
        for patcher in self.patches:
            patcher.start()
        # An existing dataset and its collector cache are one durable unit in
        # production.  Tests that intentionally exercise first-run/missing-
        # cache recovery remove this fixture explicitly below.
        Path(collect_wave.CACHE_PATH).write_text(
            json.dumps(collect_wave._empty_cache()), encoding="utf-8")

    def tearDown(self):
        for patcher in reversed(self.patches):
            patcher.stop()
        self.tempdir.cleanup()

    def write_source_ranks(self, rows):
        pd.DataFrame(rows).to_parquet(self.source_rank_path, index=False)

    def read_source_ranks(self):
        return pd.read_parquet(self.source_rank_path)

    def test_cache_save_uses_verified_direct_fallback_when_replace_is_denied(self):
        cache = {"harvested": [7], "nodata": [8], "local_parts": ["part.parquet"]}
        denied = PermissionError(5, "RaiDrive rename denied")
        with mock.patch.object(
                collect_wave.os, "replace", side_effect=denied) as replace, \
                mock.patch.object(collect_wave.time, "sleep"):
            collect_wave._save_cache(cache)

        self.assertEqual(replace.call_count, collect_wave.CACHE_WRITE_ATTEMPTS)
        self.assertEqual(collect_wave._load_cache(), cache)
        self.assertEqual(
            list(Path(self.dataset_dir).glob(".collect_cache.json.*.tmp")), [])

    def test_load_cache_recovers_valid_legacy_staging_file(self):
        Path(collect_wave.CACHE_PATH).unlink()
        recovery = Path(collect_wave.CACHE_PATH + ".tmp.tmp")
        expected = {"harvested": [7], "nodata": [8], "local_parts": []}
        recovery.write_text(json.dumps(expected), encoding="utf-8")
        (Path(self.dataset_dir) / "train.parquet").write_bytes(b"existing")

        self.assertEqual(collect_wave._load_cache(), expected)

    def test_load_cache_fails_closed_when_existing_dataset_has_no_cache(self):
        Path(collect_wave.CACHE_PATH).unlink()
        (Path(self.dataset_dir) / "train.parquet").write_bytes(b"existing")

        with self.assertRaisesRegex(RuntimeError, "canonical missing"):
            collect_wave._load_cache()

    def test_cancelled_task_partial_result_is_harvested(self):
        tasks = [{
            "id": 77, "name": "mft-camp-c-77", "status": "cancelled",
            "started_at": "2026-07-10T01:00:00Z",
        }]
        stdout = 'RESULT_JSON {"project_name":"partial","saved_at":"t1","value":1.0}'

        with mock.patch.object(
                collect_wave, "list_tasks", return_value=tasks), mock.patch.object(
                    collect_wave, "fetch_stdout", return_value=stdout):
            result = collect_wave.main([
                "--prefix", "mft-camp", "--cancelled-fetch-limit", "1"])

        self.assertEqual(result["new_unique_rows"], 1)
        cache = json.loads(Path(collect_wave.CACHE_PATH).read_text(encoding="utf-8"))
        self.assertEqual(cache["harvested"], [77])

    def test_never_started_cancelled_task_is_nodata_without_stdout_fetch(self):
        tasks = [{
            "id": 78, "name": "mft-camp-c-78", "status": "cancelled",
            "started_at": None,
        }]
        with mock.patch.object(
                collect_wave, "list_tasks", return_value=tasks), mock.patch.object(
                    collect_wave, "fetch_stdout") as fetch_stdout:
            result = collect_wave.main([
                "--prefix", "mft-camp", "--cancelled-fetch-limit", "1"])

        self.assertEqual(result["new_unique_rows"], 0)
        fetch_stdout.assert_not_called()
        cache = json.loads(Path(collect_wave.CACHE_PATH).read_text(encoding="utf-8"))
        self.assertEqual(cache["nodata"], [78])

    def test_running_fetch_limit_zero_leaves_remote_stdout_for_terminal_pass(self):
        tasks = [{
            "id": 79, "name": "mft-camp-srev-lrev-79", "status": "running",
            "started_at": "2026-07-12T01:00:00Z",
        }]
        with mock.patch.object(
                collect_wave, "list_tasks", return_value=tasks), mock.patch.object(
                    collect_wave, "fetch_stdout") as fetch_stdout:
            result = collect_wave.main([
                "--prefix", "mft-camp", "--running-fetch-limit", "0"])

        self.assertEqual(result["new_unique_rows"], 0)
        fetch_stdout.assert_not_called()

    def test_source_rank_sidecar_is_strict_and_replay_recoverable(self):
        master = pd.DataFrame([
            {"project_name": "old", "saved_at": "t0", "value": 1.0},
            {"project_name": "new", "saved_at": "t1", "value": 2.0},
        ])
        sidecar = pd.DataFrame([{
            "project_name": "old", "saved_at": "t0",
            collect_wave.SOURCE_RANK_COLUMN: 40,
        }])
        replay = pd.DataFrame([{
            "project_name": "new", "saved_at": "t1",
            "value": 2.0,
            collect_wave.SOURCE_RANK_COLUMN: 20,
        }])

        repaired = collect_wave._validated_source_rank_sidecar(
            master, sidecar, replay, ["project_name", "saved_at"])

        self.assertEqual(
            repaired.set_index("project_name").loc[
                "new", collect_wave.SOURCE_RANK_COLUMN], 20)
        with self.assertRaisesRegex(RuntimeError, "does not cover"):
            collect_wave._validated_source_rank_sidecar(
                master, sidecar, replay.iloc[0:0], ["project_name", "saved_at"])
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            collect_wave._validated_source_rank_sidecar(
                master.iloc[:1], pd.concat([sidecar, sidecar], ignore_index=True),
                replay, ["project_name", "saved_at"])
        with self.assertRaisesRegex(RuntimeError, "payload differs"):
            collect_wave._validated_source_rank_sidecar(
                master, sidecar, replay.assign(value=3.0),
                ["project_name", "saved_at"])
        with self.assertRaisesRegex(RuntimeError, "payload differs"):
            collect_wave._validated_source_rank_sidecar(
                master, sidecar, replay.drop(columns="value"),
                ["project_name", "saved_at"])

    def test_dedup_preserves_distinct_null_key_legacy_rows(self):
        rows = pd.DataFrame(
            [
                {"project_name": None, "saved_at": None, "value": 1.0, "task_id": 1},
                {"project_name": None, "saved_at": None, "value": 2.0, "task_id": 2},
                {"project_name": None, "saved_at": None, "value": 1.0, "task_id": 99},
                {"project_name": "p1", "saved_at": "t1", "value": 3.0},
                {"project_name": "p1", "saved_at": "t1", "value": 4.0},
            ]
        )

        result = collect_wave.deduplicate_rows(
            rows, ["project_name", "saved_at"])

        self.assertEqual(len(result), 3)
        self.assertEqual(set(result["value"]), {1.0, 2.0, 4.0})
        legacy = result[result["project_name"].isna()]
        self.assertEqual(set(legacy["value"]), {1.0, 2.0})

        missing_column = pd.DataFrame(
            [
                {"project_name": "legacy", "value": 5.0},
                {"project_name": "legacy", "value": 6.0},
            ]
        )
        missing_result = collect_wave.deduplicate_rows(
            missing_column, ["project_name", "saved_at"])
        self.assertEqual(len(missing_result), 2)

    def test_mixed_keyed_and_unkeyed_rows_use_separate_identities(self):
        old = pd.DataFrame(
            [
                {"project_name": "p1", "saved_at": "t1", "value": 1.0, "task_id": 1},
                {"project_name": None, "saved_at": None, "value": 10.0, "task_id": 2},
                {"project_name": "p3", "saved_at": None, "value": 30.0, "task_id": 3},
            ]
        )
        incoming = pd.DataFrame(
            [
                {"project_name": "p1", "saved_at": "t1", "value": 999.0, "task_id": 10},
                {"project_name": "p2", "saved_at": "t2", "value": 2.0, "task_id": 11},
                {"project_name": None, "saved_at": None, "value": 10.0, "task_id": 12},
                {"project_name": None, "saved_at": None, "value": 11.0, "task_id": 13},
                {"project_name": "p3", "saved_at": None, "value": 30.0, "task_id": 14},
                {"project_name": "p3", "saved_at": None, "value": 31.0, "task_id": 15},
            ]
        )

        result = collect_wave.select_new_unique_rows(
            incoming, old, ["project_name", "saved_at"])

        self.assertEqual(len(result), 3)
        self.assertEqual(set(result["value"]), {2.0, 11.0, 31.0})

    def test_fetch_error_never_enters_nodata_cache(self):
        cache = {"nodata": [], "harvested": []}
        tasks = [{"id": 7, "name": "mft-camp-c-7", "status": "completed"}]
        with mock.patch.object(collect_wave, "list_tasks", return_value=tasks), mock.patch.object(
            collect_wave, "_load_cache", return_value=cache
        ), mock.patch.object(
            collect_wave, "fetch_stdout", side_effect=collect_wave.FetchError("timeout")
        ), mock.patch.object(
            collect_wave, "_save_cache"
        ) as save_cache:
            result = collect_wave.main(["--prefix", "mft-camp"])

        self.assertEqual(result, {"new_unique_rows": 0, "fetch_errors": 1})
        self.assertEqual(cache["nodata"], [])
        save_cache.assert_not_called()

    def test_successful_empty_stdout_enters_nodata_cache(self):
        cache = {"nodata": [], "harvested": []}
        saved = []
        tasks = [{"id": 10, "name": "mft-camp-c-10", "status": "completed"}]
        with mock.patch.object(collect_wave, "list_tasks", return_value=tasks), mock.patch.object(
            collect_wave, "_load_cache", return_value=cache
        ), mock.patch.object(
            collect_wave, "fetch_stdout", return_value="completed without result rows"
        ), mock.patch.object(
            collect_wave, "_save_cache", side_effect=lambda value: saved.append(value.copy())
        ):
            result = collect_wave.main(["--prefix", "mft-camp"])

        self.assertEqual(result, {"new_unique_rows": 0, "fetch_errors": 0})
        self.assertEqual(cache["nodata"], [10])
        self.assertEqual(saved[0]["nodata"], [10])

    def test_terminal_result_json_precedes_legacy_csv_block(self):
        tasks = [{"id": 13, "name": "mft-camp-c-13", "status": "completed"}]
        stdout = "\n".join(
            [
                "===RESULT_CSV===",
                "project_name,saved_at,value",
                "p1,t1,1.0",
                "===FAILED_CSV===",
                'RESULT_JSON {"project_name":"p1","saved_at":"t1","value":2.0}',
            ]
        )

        with mock.patch.object(collect_wave, "list_tasks", return_value=tasks), mock.patch.object(
            collect_wave, "fetch_stdout", return_value=stdout
        ):
            result = collect_wave.main(["--prefix", "mft-camp"])

        self.assertEqual(result["new_unique_rows"], 1)
        master = pd.read_parquet(Path(self.dataset_dir) / "train.parquet")
        self.assertEqual(master.loc[0, "value"], 2.0)
        self.assertNotIn(collect_wave.SOURCE_RANK_COLUMN, master.columns)
        ranks = self.read_source_ranks()
        self.assertEqual(
            ranks.loc[0, collect_wave.SOURCE_RANK_COLUMN],
            collect_wave.SOURCE_RANK_JSON,
        )

    def test_terminal_csv_recovers_rows_missing_from_partial_json(self):
        tasks = [{"id": 14, "name": "mft-camp-c-14", "status": "completed"}]
        stdout = "\n".join(
            [
                "===RESULT_CSV===",
                "project_name,saved_at,value",
                "p1,t1,1.0",
                "p2,t2,3.0",
                "===FAILED_CSV===",
                'RESULT_JSON {"project_name":"p1","saved_at":"t1","value":2.0}',
                "RESULT_JSON truncated",
            ]
        )

        with mock.patch.object(collect_wave, "list_tasks", return_value=tasks), mock.patch.object(
            collect_wave, "fetch_stdout", return_value=stdout
        ):
            result = collect_wave.main(["--prefix", "mft-camp"])

        self.assertEqual(result["new_unique_rows"], 2)
        master = pd.read_parquet(Path(self.dataset_dir) / "train.parquet").set_index(
            "project_name"
        )
        self.assertEqual(master.loc["p1", "value"], 2.0)
        self.assertEqual(master.loc["p2", "value"], 3.0)
        self.assertNotIn(collect_wave.SOURCE_RANK_COLUMN, master.columns)
        ranks = self.read_source_ranks().set_index("project_name")
        self.assertEqual(
            ranks.loc["p1", collect_wave.SOURCE_RANK_COLUMN],
            collect_wave.SOURCE_RANK_JSON,
        )
        self.assertEqual(
            ranks.loc["p2", collect_wave.SOURCE_RANK_COLUMN],
            collect_wave.SOURCE_RANK_TERMINAL_CSV,
        )

    def test_local_csv_uses_producer_lock_and_skips_incomplete_rows(self):
        pd.DataFrame(
            [
                {"project_name": "complete", "saved_at": "t1", "value": 1.0},
                {"project_name": None, "saved_at": None, "value": 2.0},
            ]
        ).to_csv(self.local_results_csv, index=False)

        with mock.patch.object(collect_wave, "FileLock") as file_lock:
            frames, pending = collect_wave.load_local_result_frames(
                {"local_parts": []}
            )

        file_lock.assert_called_once_with(str(self.local_results_csv) + ".lock")
        self.assertEqual(pending, [])
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0]["project_name"].tolist(), ["complete"])

    def test_local_parts_override_csv_and_are_cached_after_merge(self):
        pd.DataFrame(
            [{"project_name": "p1", "saved_at": "t1", "value": 1.0, "csv_only": 1}]
        ).to_csv(self.local_results_csv, index=False)
        self.local_parts_dir.mkdir()
        part = self.local_parts_dir / "part_001.parquet"
        pd.DataFrame(
            [{"project_name": "p1", "saved_at": "t1", "value": 2.0, "part_only": 1}]
        ).to_parquet(part, index=False)

        with mock.patch.object(collect_wave, "list_tasks", return_value=[]):
            first = collect_wave.main(["--prefix", "mft-camp"])

        self.assertEqual(first["new_unique_rows"], 1)
        master_path = Path(self.dataset_dir) / "train.parquet"
        master = pd.read_parquet(master_path)
        self.assertEqual(master.loc[0, "value"], 2.0)
        self.assertEqual(master.loc[0, "part_only"], 1)
        self.assertNotIn(collect_wave.SOURCE_RANK_COLUMN, master.columns)
        ranks = self.read_source_ranks()
        self.assertEqual(
            ranks.loc[0, collect_wave.SOURCE_RANK_COLUMN],
            collect_wave.SOURCE_RANK_LOCAL_PART,
        )
        cache = json.loads(Path(collect_wave.CACHE_PATH).read_text(encoding="utf-8"))
        self.assertEqual(cache["local_parts"], [part.name])

        original_read_parquet = pd.read_parquet
        with mock.patch.object(collect_wave, "list_tasks", return_value=[]), mock.patch.object(
            collect_wave.pd, "read_parquet", wraps=original_read_parquet
        ) as read_parquet:
            second = collect_wave.main(["--prefix", "mft-camp"])

        self.assertEqual(second["new_unique_rows"], 0)
        read_paths = [Path(call.args[0]) for call in read_parquet.call_args_list]
        self.assertNotIn(part, read_paths)

    def test_higher_rank_local_part_replaces_existing_master_row(self):
        master_path = Path(self.dataset_dir) / "train.parquet"
        pd.DataFrame(
            [{"project_name": "p1", "saved_at": "t1", "value": 1.0}]
        ).to_parquet(master_path, index=False)
        self.write_source_ranks(
            [{
                "project_name": "p1", "saved_at": "t1",
                collect_wave.SOURCE_RANK_COLUMN: collect_wave.SOURCE_RANK_LOCAL_CSV,
            }]
        )
        self.local_parts_dir.mkdir()
        part = self.local_parts_dir / "part_upgrade.parquet"
        pd.DataFrame(
            [{"project_name": "p1", "saved_at": "t1", "value": 2.0}]
        ).to_parquet(part, index=False)

        with mock.patch.object(collect_wave, "list_tasks", return_value=[]):
            result = collect_wave.main(["--prefix", "mft-camp"])

        self.assertEqual(result["new_unique_rows"], 1)
        master = pd.read_parquet(master_path)
        self.assertEqual(len(master), 1)
        self.assertEqual(master.loc[0, "value"], 2.0)
        self.assertNotIn(collect_wave.SOURCE_RANK_COLUMN, master.columns)
        ranks = self.read_source_ranks()
        self.assertEqual(
            ranks.loc[0, collect_wave.SOURCE_RANK_COLUMN],
            collect_wave.SOURCE_RANK_LOCAL_PART,
        )
        cache = json.loads(Path(collect_wave.CACHE_PATH).read_text(encoding="utf-8"))
        self.assertEqual(cache["local_parts"], [part.name])

    def test_sidecar_replace_failure_leaves_part_uncached_for_safe_retry(self):
        master_path = Path(self.dataset_dir) / "train.parquet"
        pd.DataFrame(
            [{"project_name": "p1", "saved_at": "t1", "value": 1.0}]
        ).to_parquet(master_path, index=False)
        self.write_source_ranks(
            [{
                "project_name": "p1", "saved_at": "t1",
                collect_wave.SOURCE_RANK_COLUMN: collect_wave.SOURCE_RANK_LOCAL_CSV,
            }]
        )
        self.local_parts_dir.mkdir()
        part = self.local_parts_dir / "part_retry.parquet"
        pd.DataFrame(
            [{"project_name": "p1", "saved_at": "t1", "value": 2.0}]
        ).to_parquet(part, index=False)
        Path(collect_wave.CACHE_PATH).write_text(json.dumps({
            "harvested": [], "nodata": [], "local_parts": [],
        }), encoding="utf-8")
        original_replace = collect_wave.os.replace

        def fail_rank_replace(source, target):
            if Path(target).name == self.source_rank_path.name:
                raise OSError("rank sidecar unavailable")
            return original_replace(source, target)

        with mock.patch.object(collect_wave, "list_tasks", return_value=[]), mock.patch.object(
            collect_wave.os, "replace", side_effect=fail_rank_replace
        ):
            with self.assertRaisesRegex(OSError, "rank sidecar unavailable"):
                collect_wave.main(["--prefix", "mft-camp"])

        self.assertEqual(pd.read_parquet(master_path).loc[0, "value"], 2.0)
        self.assertEqual(
            self.read_source_ranks().loc[0, collect_wave.SOURCE_RANK_COLUMN],
            collect_wave.SOURCE_RANK_LOCAL_CSV,
        )
        interrupted_cache = json.loads(
            Path(collect_wave.CACHE_PATH).read_text(encoding="utf-8"))
        self.assertEqual(interrupted_cache["local_parts"], [])

        with mock.patch.object(collect_wave, "list_tasks", return_value=[]):
            result = collect_wave.main(["--prefix", "mft-camp"])

        self.assertEqual(result["new_unique_rows"], 1)
        cache = json.loads(Path(collect_wave.CACHE_PATH).read_text(encoding="utf-8"))
        self.assertEqual(cache["local_parts"], [part.name])
        self.assertEqual(
            self.read_source_ranks().loc[0, collect_wave.SOURCE_RANK_COLUMN],
            collect_wave.SOURCE_RANK_LOCAL_PART,
        )

    def test_corrupt_local_part_is_not_cached_and_is_retried(self):
        self.local_parts_dir.mkdir()
        good = self.local_parts_dir / "part_good.parquet"
        bad = self.local_parts_dir / "part_bad.parquet"
        pd.DataFrame(
            [{"project_name": "good", "saved_at": "t1", "value": 1.0}]
        ).to_parquet(good, index=False)
        bad.write_bytes(b"incomplete parquet")

        with mock.patch.object(collect_wave, "list_tasks", return_value=[]):
            first = collect_wave.main(["--prefix", "mft-camp"])

        self.assertEqual(first["new_unique_rows"], 1)
        cache = json.loads(Path(collect_wave.CACHE_PATH).read_text(encoding="utf-8"))
        self.assertEqual(cache["local_parts"], [good.name])

        pd.DataFrame(
            [{"project_name": "recovered", "saved_at": "t2", "value": 2.0}]
        ).to_parquet(bad, index=False)
        with mock.patch.object(collect_wave, "list_tasks", return_value=[]):
            second = collect_wave.main(["--prefix", "mft-camp"])

        self.assertEqual(second["new_unique_rows"], 1)
        cache = json.loads(Path(collect_wave.CACHE_PATH).read_text(encoding="utf-8"))
        self.assertEqual(set(cache["local_parts"]), {good.name, bad.name})
        master = pd.read_parquet(Path(self.dataset_dir) / "train.parquet")
        self.assertEqual(set(master["project_name"]), {"good", "recovered"})

    def test_local_part_cache_waits_for_successful_dataset_transaction(self):
        master = Path(self.dataset_dir) / "train.parquet"
        pd.DataFrame(
            [{"project_name": "p1", "saved_at": "t1", "value": 1.0}]
        ).to_parquet(master, index=False)
        self.write_source_ranks([{
            "project_name": "p1", "saved_at": "t1",
            collect_wave.SOURCE_RANK_COLUMN: collect_wave.SOURCE_RANK_LOCAL_PART,
        }])
        before = master.read_bytes()
        self.local_parts_dir.mkdir()
        part = self.local_parts_dir / "part_pending.parquet"
        pd.DataFrame(
            [{"project_name": "p2", "saved_at": "t2", "value": 2.0}]
        ).to_parquet(part, index=False)
        cache = {"nodata": [], "harvested": [], "local_parts": []}
        original_to_parquet = pd.DataFrame.to_parquet

        def fail_master_parquet(frame, path, *args, **kwargs):
            if Path(path).name.startswith(".train.parquet."):
                raise RuntimeError("disk full")
            return original_to_parquet(frame, path, *args, **kwargs)

        with mock.patch.object(collect_wave, "list_tasks", return_value=[]), mock.patch.object(
            collect_wave, "_load_cache", return_value=cache
        ), mock.patch.object(
            collect_wave, "_save_cache"
        ) as save_cache, mock.patch.object(
            pd.DataFrame, "to_parquet", autospec=True, side_effect=fail_master_parquet
        ):
            with self.assertRaisesRegex(RuntimeError, "disk full"):
                collect_wave.main(["--prefix", "mft-camp"])

        self.assertEqual(master.read_bytes(), before)
        self.assertEqual(cache["local_parts"], [])
        save_cache.assert_not_called()

    def test_duplicate_rows_do_not_rewrite_dataset_outputs(self):
        master = Path(self.dataset_dir) / "train.parquet"
        manifest = Path(self.dataset_dir) / "manifest.json"
        pd.DataFrame(
            [{"project_name": "p1", "saved_at": "t1", "value": 1.0}]
        ).to_parquet(master, index=False)
        self.write_source_ranks(
            [{
                "project_name": "p1", "saved_at": "t1",
                collect_wave.SOURCE_RANK_COLUMN: collect_wave.SOURCE_RANK_JSON,
            }]
        )
        manifest.write_text("sentinel", encoding="utf-8")
        before = master.read_bytes()
        tasks = [{"id": 8, "name": "mft-camp-c-8", "status": "running"}]
        stdout = 'RESULT_JSON {"project_name":"p1","saved_at":"t1","value":2.0}'

        with mock.patch.object(collect_wave, "list_tasks", return_value=tasks), mock.patch.object(
            collect_wave, "fetch_stdout", return_value=stdout
        ):
            result = collect_wave.main(["--prefix", "mft-camp"])

        self.assertEqual(result["new_unique_rows"], 0)
        self.assertEqual(master.read_bytes(), before)
        self.assertEqual(manifest.read_text(encoding="utf-8"), "sentinel")
        self.assertEqual(list(Path(self.dataset_dir).glob("collected_*.parquet")), [])

    def test_terminal_duplicate_commits_harvested_after_locked_merge(self):
        master = Path(self.dataset_dir) / "train.parquet"
        pd.DataFrame(
            [{"project_name": "p1", "saved_at": "t1", "value": 1.0}]
        ).to_parquet(master, index=False)
        self.write_source_ranks(
            [{
                "project_name": "p1", "saved_at": "t1",
                collect_wave.SOURCE_RANK_COLUMN: collect_wave.SOURCE_RANK_JSON,
            }]
        )
        cache = {"nodata": [], "harvested": []}
        tasks = [{"id": 11, "name": "mft-camp-c-11", "status": "completed"}]
        stdout = 'RESULT_JSON {"project_name":"p1","saved_at":"t1","value":2.0}'

        with mock.patch.object(collect_wave, "list_tasks", return_value=tasks), mock.patch.object(
            collect_wave, "fetch_stdout", return_value=stdout
        ), mock.patch.object(
            collect_wave, "_load_cache", return_value=cache
        ):
            result = collect_wave.main(["--prefix", "mft-camp"])

        self.assertEqual(result["new_unique_rows"], 0)
        persisted = json.loads(Path(collect_wave.CACHE_PATH).read_text(encoding="utf-8"))
        self.assertEqual(persisted["harvested"], [11])

    def test_policy_recollect_replaces_demoted_row_once_without_raw_duplicate(self):
        task_id = 41761
        revision = "1519dd3"
        project_name = "simulation35400"
        saved_at = "2026-07-16T08:16:08+09:00"
        master = Path(self.dataset_dir) / "train.parquet"
        pd.DataFrame([{
            "project_name": project_name,
            "saved_at": saved_at,
            "git_hash": revision,
            "Tprobe_core_center_max": float("nan"),
            "task_id": task_id,
            "task_name": "mft-1to3-q18",
        }]).to_parquet(master, index=False)
        self.write_source_ranks([{
            "project_name": project_name,
            "saved_at": saved_at,
            collect_wave.SOURCE_RANK_COLUMN: collect_wave.SOURCE_RANK_JSON,
        }])
        Path(collect_wave.CACHE_PATH).write_text(json.dumps({
            "nodata": [], "harvested": [task_id], "local_parts": [],
        }), encoding="utf-8")
        task = {
            "id": task_id,
            "name": "mft-1to3-q18-s522-r35400",
            "status": "completed",
        }
        stdout = "RESULT_JSON " + json.dumps({
            "project_name": project_name,
            "saved_at": saved_at,
            "git_hash": revision,
            "Tprobe_core_center_max": 87.0,
        })

        previous_ancestry = collect_wave.PROBE_FIX_HASHES_OK
        collect_wave.PROBE_FIX_HASHES_OK = {revision: True}
        try:
            with mock.patch.object(
                    collect_wave, "list_tasks", return_value=[]), mock.patch.object(
                    collect_wave, "_get_json", return_value=task) as get_task, mock.patch.object(
                    collect_wave, "fetch_stdout", return_value=stdout):
                first = collect_wave.main([
                    "--prefix", "mft-1to3", "--recollect-task", str(task_id)
                ])
                second = collect_wave.main([
                    "--prefix", "mft-1to3", "--recollect-task", str(task_id)
                ])
        finally:
            collect_wave.PROBE_FIX_HASHES_OK = previous_ancestry

        self.assertEqual(get_task.call_count, 2)
        self.assertEqual(first["new_unique_rows"], 1)
        self.assertEqual(second["new_unique_rows"], 0)
        repaired = pd.read_parquet(master)
        self.assertEqual(len(repaired), 1)
        self.assertEqual(repaired["Tprobe_core_center_max"].iloc[0], 87.0)
        ranks = self.read_source_ranks()
        self.assertEqual(len(ranks), 1)
        self.assertEqual(
            ranks[collect_wave.SOURCE_RANK_COLUMN].iloc[0],
            collect_wave.SOURCE_RANK_JSON + collect_wave.SOURCE_RANK_RECOLLECT_OFFSET,
        )
        cache = json.loads(Path(collect_wave.CACHE_PATH).read_text(encoding="utf-8"))
        self.assertEqual(cache["harvested"], [task_id])
        self.assertEqual(len(list(Path(self.dataset_dir).glob("collected_*.parquet"))), 1)

    def test_parquet_failure_keeps_master_and_terminal_cache_uncommitted(self):
        master = Path(self.dataset_dir) / "train.parquet"
        pd.DataFrame(
            [{"project_name": "p1", "saved_at": "t1", "value": 1.0}]
        ).to_parquet(master, index=False)
        self.write_source_ranks([{
            "project_name": "p1", "saved_at": "t1",
            collect_wave.SOURCE_RANK_COLUMN: collect_wave.SOURCE_RANK_JSON,
        }])
        before = master.read_bytes()
        cache = {"nodata": [], "harvested": []}
        tasks = [{"id": 12, "name": "mft-camp-c-12", "status": "completed"}]
        stdout = 'RESULT_JSON {"project_name":"p2","saved_at":"t2","value":2.0}'
        original_to_parquet = pd.DataFrame.to_parquet

        def fail_master_parquet(frame, path, *args, **kwargs):
            if Path(path).name.startswith(".train.parquet."):
                raise RuntimeError("disk full")
            return original_to_parquet(frame, path, *args, **kwargs)

        with mock.patch.object(collect_wave, "list_tasks", return_value=tasks), mock.patch.object(
            collect_wave, "fetch_stdout", return_value=stdout
        ), mock.patch.object(
            collect_wave, "_load_cache", return_value=cache
        ), mock.patch.object(
            collect_wave, "_save_cache"
        ) as save_cache, mock.patch.object(
            pd.DataFrame, "to_parquet", autospec=True, side_effect=fail_master_parquet
        ):
            with self.assertRaisesRegex(RuntimeError, "disk full"):
                collect_wave.main(["--prefix", "mft-camp"])

        self.assertEqual(master.read_bytes(), before)
        self.assertEqual(cache["harvested"], [])
        save_cache.assert_not_called()
        self.assertEqual(list(Path(self.dataset_dir).glob("collected_*.parquet")), [])
        self.assertEqual(list(Path(self.dataset_dir).glob(".*.tmp")), [])

    def test_new_unique_row_updates_all_outputs_once(self):
        master = Path(self.dataset_dir) / "train.parquet"
        pd.DataFrame(
            [{"project_name": "p1", "saved_at": "t1", "value": 1.0}]
        ).to_parquet(master, index=False)
        self.write_source_ranks([{
            "project_name": "p1", "saved_at": "t1",
            collect_wave.SOURCE_RANK_COLUMN: collect_wave.SOURCE_RANK_JSON,
        }])
        tasks = [{"id": 9, "name": "mft-camp-c-9", "status": "running"}]
        stdout = 'RESULT_JSON {"project_name":"p2","saved_at":"t2","value":2.0}'

        with mock.patch.object(collect_wave, "list_tasks", return_value=tasks), mock.patch.object(
            collect_wave, "fetch_stdout", return_value=stdout
        ):
            result = collect_wave.main(["--prefix", "mft-camp"])

        self.assertEqual(result["new_unique_rows"], 1)
        self.assertEqual(len(pd.read_parquet(master)), 2)
        snapshots = list(Path(self.dataset_dir).glob("collected_*.parquet"))
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(len(pd.read_parquet(snapshots[0])), 1)
        manifest = json.loads(
            (Path(self.dataset_dir) / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["new_unique_rows"], 1)
        self.assertEqual(manifest["total_rows"], 2)

    def test_concurrent_merges_re_read_master_under_file_lock(self):
        first = pd.DataFrame(
            [{"project_name": "p1", "saved_at": "t1", "value": 1.0}]
        )
        second = pd.DataFrame(
            [{"project_name": "p2", "saved_at": "t2", "value": 2.0}]
        )
        barrier = threading.Barrier(2)
        results = []
        errors = []
        original_stage = collect_wave._stage_parquet

        def slow_stage(frame, target):
            if target.endswith("train.parquet"):
                time.sleep(0.05)
            return original_stage(frame, target)

        def worker(frame, task_id):
            try:
                barrier.wait()
                results.append(
                    collect_wave.merge_dataset(
                        frame, ["project_name", "saved_at"], "mft-camp",
                        pending_harvested=[task_id])
                )
            except Exception as exc:
                errors.append(exc)

        with mock.patch.object(collect_wave, "_stage_parquet", side_effect=slow_stage):
            threads = [
                threading.Thread(target=worker, args=(first, 21)),
                threading.Thread(target=worker, args=(second, 22)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)

        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)
        master = pd.read_parquet(Path(self.dataset_dir) / "train.parquet")
        self.assertEqual(set(master["project_name"]), {"p1", "p2"})
        manifest = json.loads(
            (Path(self.dataset_dir) / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["total_rows"], 2)
        self.assertEqual(len(list(Path(self.dataset_dir).glob("collected_*.parquet"))), 2)
        cache = json.loads(Path(collect_wave.CACHE_PATH).read_text(encoding="utf-8"))
        self.assertEqual(set(cache["harvested"]), {21, 22})


class ProbeSanitizerTests(unittest.TestCase):
    BASELINE = collect_wave.PROBE_FIX_COMMIT
    SIDE_ABBREV = "a1b2c3d"
    SIDE_FULL = SIDE_ABBREV + "0" * 33
    OLD_ABBREV = "b1c2d3e"
    OLD_FULL = OLD_ABBREV + "0" * 33
    UNKNOWN_ABBREV = "c1d2e3f"

    def setUp(self):
        self.previous_cache = collect_wave.PROBE_FIX_HASHES_OK
        collect_wave.PROBE_FIX_HASHES_OK = None

    def tearDown(self):
        collect_wave.PROBE_FIX_HASHES_OK = self.previous_cache

    @staticmethod
    def _completed(command, returncode=0, stdout="", stderr=""):
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)

    def _git_side_effect(self, *, side_result=0, old_result=1):
        def run(command, **kwargs):
            if command[:4] == ["git", "rev-parse", "--verify", "--quiet"]:
                revision = command[-1].split("^", 1)[0]
                resolved = {
                    self.BASELINE: self.BASELINE,
                    self.SIDE_ABBREV: self.SIDE_FULL,
                    self.OLD_ABBREV: self.OLD_FULL,
                }.get(revision)
                if resolved is None:
                    return self._completed(command, 128, stderr="unknown revision")
                return self._completed(command, stdout=resolved + "\n")
            if command[:3] == ["git", "merge-base", "--is-ancestor"]:
                if command[-1] == self.SIDE_FULL:
                    return self._completed(command, side_result)
                if command[-1] == self.OLD_FULL:
                    return self._completed(command, old_result)
            raise AssertionError(f"unexpected git query: {command}")

        return run

    @staticmethod
    def _probe_frame(revision, temperature=87.0):
        return pd.DataFrame([{
            "git_hash": revision,
            "Tprobe_core_center_max": temperature,
            "Tprobe_Rx_main_side_max": temperature + 1,
            "Tprobe_Rx_side_leeward_max": temperature + 2,
        }])

    def test_side_branch_descendant_is_preserved_by_true_merge_base_query(self):
        frame = self._probe_frame(self.SIDE_ABBREV)
        with mock.patch(
                "subprocess.run", side_effect=self._git_side_effect()) as run:
            sanitized, count = collect_wave.sanitize_bad_probes(frame.copy())

        self.assertEqual(count, 0)
        self.assertEqual(sanitized["Tprobe_core_center_max"].iloc[0], 87.0)
        merge_calls = [
            call.args[0] for call in run.call_args_list
            if call.args[0][:3] == ["git", "merge-base", "--is-ancestor"]
        ]
        self.assertEqual(merge_calls, [[
            "git", "merge-base", "--is-ancestor", self.BASELINE, self.SIDE_FULL
        ]])
        self.assertNotIn("HEAD", merge_calls[0])
        for call in run.call_args_list:
            self.assertEqual(call.kwargs["encoding"], "utf-8")
            self.assertEqual(call.kwargs["errors"], "replace")
            self.assertFalse(call.kwargs["check"])

    def test_non_descendant_is_demoted_but_unrelated_leeward_probe_remains(self):
        frame = self._probe_frame(self.OLD_ABBREV, temperature=77.0)
        with mock.patch("subprocess.run", side_effect=self._git_side_effect()):
            sanitized, count = collect_wave.sanitize_bad_probes(frame.copy())

        self.assertEqual(count, 1)
        self.assertTrue(pd.isna(sanitized["Tprobe_core_center_max"].iloc[0]))
        self.assertTrue(pd.isna(sanitized["Tprobe_Rx_main_side_max"].iloc[0]))
        self.assertEqual(sanitized["Tprobe_Rx_side_leeward_max"].iloc[0], 79.0)

    def test_unknown_commit_is_untrusted_without_running_merge_base(self):
        frame = self._probe_frame(self.UNKNOWN_ABBREV)
        with mock.patch(
                "subprocess.run", side_effect=self._git_side_effect()) as run:
            sanitized, count = collect_wave.sanitize_bad_probes(frame.copy())

        self.assertEqual(count, 1)
        self.assertTrue(pd.isna(sanitized["Tprobe_core_center_max"].iloc[0]))
        self.assertFalse(any(
            call.args[0][:3] == ["git", "merge-base", "--is-ancestor"]
            for call in run.call_args_list
        ))

    def test_git_classification_failure_aborts_without_mutating_data(self):
        frame = pd.DataFrame({
            "git_hash": ["newhash"],
            "Tprobe_core_center_max": [87.0],
        })
        with mock.patch("subprocess.run", side_effect=OSError("git unavailable")):
            with self.assertRaisesRegex(RuntimeError, "ancestry classification failed"):
                collect_wave.sanitize_bad_probes(frame)
        self.assertEqual(frame["Tprobe_core_center_max"].iloc[0], 87.0)


class ThermalValidityTests(unittest.TestCase):
    def test_legacy_false_success_is_demoted_but_em_row_is_preserved(self):
        frame = pd.DataFrame([
            {
                "project_name": "missing-side",
                "thermal_solved": 1,
                "N2_side": 2,
                "T_max_Tx": 80.0,
                "T_max_Rx_main": 81.0,
                "T_max_Rx_side": float("nan"),
                "T_max_core": 82.0,
                "Tprobe_Tx_leeward_max": 79.0,
                "Tprobe_Rx_main_leeward_max": 80.0,
                "Tprobe_core_center_max": 81.0,
            },
            {
                "project_name": "no-side-required",
                "thermal_solved": 1,
                "N2_side": 0,
                "T_max_Tx": 80.0,
                "T_max_Rx_main": 81.0,
                "T_max_Rx_side": float("nan"),
                "T_max_core": 82.0,
            },
            {
                "project_name": "new-incomplete-contract",
                "thermal_solved": 1,
                "N2_side": 2,
                "T_max_Tx": 80.0,
                "T_max_Rx_main": 81.0,
                "T_max_Rx_side": 82.0,
                "T_max_core": 83.0,
                "thermal_required_group_mask": 15,
                "thermal_required_missing_count": 0,
                "thermal_extraction_complete": 0,
            },
            {
                "project_name": "new-complete-contract",
                "thermal_solved": 1,
                "N2_side": 0,
                "T_max_Tx": 80.0,
                "T_max_Rx_main": 81.0,
                "T_max_Rx_side": float("nan"),
                "T_max_core": 82.0,
                "Tprobe_Tx_leeward_max": 79.0,
                "Tprobe_Rx_main_leeward_max": 80.0,
                "Tprobe_core_center_max": 81.0,
                "thermal_required_group_mask": 11,
                "thermal_required_missing_count": 0,
                "thermal_extraction_complete": 1,
                "thermal_convergence_available": 1,
                "thermal_converged": 1,
                "thermal_iterations": 151,
                "thermal_residual_flow_limit": 1e-3,
                "thermal_residual_energy_limit": 1e-7,
                "thermal_residual_continuity": 8e-4,
                "thermal_residual_x_velocity": 4e-4,
                "thermal_residual_y_velocity": 9e-4,
                "thermal_residual_z_velocity": 4e-4,
                "thermal_residual_energy": 4e-9,
            },
            {
                "project_name": "loose-residual-contract",
                "thermal_solved": 1,
                "N2_side": 0,
                "T_max_Tx": 80.0,
                "T_max_Rx_main": 81.0,
                "T_max_Rx_side": float("nan"),
                "T_max_core": 82.0,
                "thermal_required_group_mask": 11,
                "thermal_required_missing_count": 0,
                "thermal_extraction_complete": 1,
                "thermal_convergence_available": 1,
                "thermal_converged": 1,
                "thermal_iterations": 151,
                "thermal_residual_flow_limit": 1e-2,
                "thermal_residual_energy_limit": 1e-7,
                "thermal_residual_continuity": 8e-3,
                "thermal_residual_x_velocity": 4e-3,
                "thermal_residual_y_velocity": 9e-3,
                "thermal_residual_z_velocity": 4e-3,
                "thermal_residual_energy": 4e-9,
            },
        ])
        saturated = frame.iloc[[3]].copy()
        saturated.loc[:, "project_name"] = "solver-temperature-ceiling"
        saturated.loc[:, "T_max_Rx_main"] = 4726.85
        saturated.loc[:, "result_valid_em"] = 1
        saturated.loc[:, "result_valid_thermal"] = 1
        frame = pd.concat([frame, saturated], ignore_index=True)

        normalized, count = collect_wave.normalize_thermal_validity(frame)

        # Even the superficially complete legacy row is demoted because it has
        # no native Rx power-balance proof; stored thermal flags are insufficient.
        self.assertEqual(count, 6)
        self.assertEqual(normalized["project_name"].tolist(), frame["project_name"].tolist())
        self.assertEqual(normalized["thermal_solved"].tolist(), [0, 0, 0, 0, 0, 0])
        self.assertEqual(normalized["result_valid_thermal"].iloc[0], 0)
        self.assertEqual(normalized["result_valid_thermal"].iloc[1], 0)
        self.assertEqual(normalized["result_valid_thermal"].iloc[2], 0)
        self.assertEqual(normalized["result_valid_thermal"].iloc[3], 0)
        self.assertEqual(normalized["result_valid_thermal"].iloc[4], 0)
        self.assertEqual(normalized["result_valid_thermal"].iloc[5], 0)
        self.assertEqual(normalized["result_valid_em"].iloc[5], 1)

    def test_existing_master_saturation_is_atomically_demoted(self):
        row = {
            "project_name": "simulation438",
            "thermal_solved": 1,
            "result_valid_em": 1,
            "result_valid_thermal": 1,
            "N2_side": 0,
            "T_max_Tx": 85.0,
            "T_max_Rx_main": 4726.85,
            "T_max_Rx_side": float("nan"),
            "T_max_core": 76.0,
            "Tprobe_Tx_leeward_max": 84.0,
            "Tprobe_Rx_main_leeward_max": 4726.85,
            "Tprobe_core_center_max": 75.0,
            "thermal_required_group_mask": 11,
            "thermal_required_missing_count": 0,
            "thermal_extraction_complete": 1,
            "thermal_convergence_available": 1,
            "thermal_converged": 1,
            "thermal_iterations": 176,
            "thermal_residual_flow_limit": 1e-3,
            "thermal_residual_energy_limit": 1e-7,
            "thermal_residual_continuity": 8e-4,
            "thermal_residual_x_velocity": 4e-4,
            "thermal_residual_y_velocity": 9e-4,
            "thermal_residual_z_velocity": 4e-4,
            "thermal_residual_energy": 4e-9,
        }
        previous = collect_wave.DATASET_DIR
        try:
            with tempfile.TemporaryDirectory() as directory:
                collect_wave.DATASET_DIR = directory
                master = Path(directory) / "train.parquet"
                pd.DataFrame([row]).to_parquet(master, index=False)

                self.assertEqual(collect_wave.repair_master_thermal_validity(), 1)
                repaired = pd.read_parquet(master)
                self.assertEqual(repaired["thermal_solved"].iloc[0], 0)
                self.assertEqual(repaired["result_valid_thermal"].iloc[0], 0)
                self.assertEqual(repaired["result_valid_em"].iloc[0], 1)
                self.assertEqual(repaired["T_max_Rx_main"].iloc[0], 4726.85)
                self.assertEqual(collect_wave.repair_master_thermal_validity(), 0)
        finally:
            collect_wave.DATASET_DIR = previous


class ProvenanceFilterTests(unittest.TestCase):
    def test_explicit_dirty_rows_are_rejected_but_legacy_missing_flags_remain(self):
        frame = pd.DataFrame([
            {"sample": "clean", "git_dirty": 0, "pyaedt_library_git_dirty": 0},
            {"sample": "solver-dirty", "git_dirty": 1, "pyaedt_library_git_dirty": 0},
            {"sample": "library-dirty", "git_dirty": 0, "pyaedt_library_git_dirty": 1},
            {"sample": "legacy", "git_dirty": None, "pyaedt_library_git_dirty": None},
            {"sample": "fallback", "git_dirty": 0, "pyaedt_library_git_dirty": 0,
             "matrix_extraction_backend": "get_solution_data"},
        ])

        filtered, rejected = collect_wave.reject_explicit_dirty_provenance(frame)

        self.assertEqual(rejected, 3)
        self.assertEqual(filtered["sample"].tolist(), ["clean", "legacy"])


if __name__ == "__main__":
    unittest.main()
