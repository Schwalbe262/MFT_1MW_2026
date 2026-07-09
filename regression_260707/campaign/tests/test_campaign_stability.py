import json
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


class FakeResponse:
    def __init__(self, status_code=200, text="", payload=None, headers=None):
        self.status_code = status_code
        self.text = text
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        return self._payload


class FeederTests(unittest.TestCase):
    def test_refill_uses_scheduler_truth_and_cpu_cap_not_ledger(self):
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
            feeder, "scheduler_snapshot", return_value=(counts, allocations)
        ), mock.patch.object(
            feeder, "submit", side_effect=[901, 902, 903, 904, 905]
        ) as submit_mock, mock.patch.object(
            feeder, "save_state"
        ) as save_mock, mock.patch.object(
            feeder.time, "sleep"
        ):
            self.assertTrue(feeder.step(1000, target=130, buffer=40))

        # total cap=13, availability cap=(2 running + 1 attaching)+6=9;
        # scheduler active=4, so exactly five tasks may be admitted.
        self.assertEqual(submit_mock.call_count, 5)
        self.assertEqual(state["serial"], 105)
        self.assertEqual(state["submitted_samples"], 40)
        self.assertEqual(save_mock.call_count, 5)

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
            feeder, "scheduler_snapshot", return_value=(counts, allocations)
        ), mock.patch.object(
            feeder, "submit", return_value=42
        ) as submit_mock, mock.patch.object(
            feeder, "save_state"
        ), mock.patch.object(
            feeder.time, "sleep"
        ):
            feeder.step(1000, target=2, buffer=1)
        self.assertEqual(submit_mock.call_count, 1)


class CollectorFetchTests(unittest.TestCase):
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


class CollectorDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.dataset_dir = self.tempdir.name
        self.patches = [
            mock.patch.object(collect_wave, "DATASET_DIR", self.dataset_dir),
            mock.patch.object(
                collect_wave, "CACHE_PATH", str(Path(self.dataset_dir) / "collect_cache.json")
            ),
            mock.patch.object(
                collect_wave,
                "LOCAL_RESULTS_CSV",
                str(Path(self.dataset_dir) / "does-not-exist.csv"),
            ),
        ]
        for patcher in self.patches:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.patches):
            patcher.stop()
        self.tempdir.cleanup()

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

    def test_duplicate_rows_do_not_rewrite_dataset_outputs(self):
        master = Path(self.dataset_dir) / "train.parquet"
        manifest = Path(self.dataset_dir) / "manifest.json"
        pd.DataFrame(
            [{"project_name": "p1", "saved_at": "t1", "value": 1.0}]
        ).to_parquet(master, index=False)
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

    def test_parquet_failure_keeps_master_and_terminal_cache_uncommitted(self):
        master = Path(self.dataset_dir) / "train.parquet"
        pd.DataFrame(
            [{"project_name": "p1", "saved_at": "t1", "value": 1.0}]
        ).to_parquet(master, index=False)
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


if __name__ == "__main__":
    unittest.main()
