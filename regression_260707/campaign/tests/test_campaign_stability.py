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

    def tearDown(self):
        for patcher in reversed(self.patches):
            patcher.stop()
        self.tempdir.cleanup()

    def write_source_ranks(self, rows):
        pd.DataFrame(rows).to_parquet(self.source_rank_path, index=False)

    def read_source_ranks(self):
        return pd.read_parquet(self.source_rank_path)

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
        self.assertFalse(Path(collect_wave.CACHE_PATH).exists())

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


class ProbeSanitizerTests(unittest.TestCase):
    def test_git_ancestry_uses_utf8_and_preserves_rx_side_leeward(self):
        responses = [
            mock.Mock(stdout=""),
            mock.Mock(stdout="newhash000000000000000000000000000000000\n"),
        ]
        frame = pd.DataFrame([
            {
                "git_hash": "newhash",
                "Tprobe_core_center_max": 87.0,
                "Tprobe_Rx_main_side_max": 88.0,
                "Tprobe_Rx_side_leeward_max": 89.0,
            },
            {
                "git_hash": "oldhash",
                "Tprobe_core_center_max": 77.0,
                "Tprobe_Rx_main_side_max": 78.0,
                "Tprobe_Rx_side_leeward_max": 79.0,
            },
        ])
        previous = collect_wave.PROBE_FIX_HASHES_OK
        collect_wave.PROBE_FIX_HASHES_OK = None
        try:
            with mock.patch("subprocess.run", side_effect=responses) as run:
                sanitized, count = collect_wave.sanitize_bad_probes(frame.copy())
        finally:
            collect_wave.PROBE_FIX_HASHES_OK = previous

        self.assertEqual(count, 1)
        self.assertEqual(sanitized["Tprobe_core_center_max"].iloc[0], 87.0)
        self.assertTrue(pd.isna(sanitized["Tprobe_core_center_max"].iloc[1]))
        self.assertTrue(pd.isna(sanitized["Tprobe_Rx_main_side_max"].iloc[1]))
        self.assertEqual(sanitized["Tprobe_Rx_side_leeward_max"].tolist(), [89.0, 79.0])
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            self.assertEqual(call.kwargs["encoding"], "utf-8")
            self.assertEqual(call.kwargs["errors"], "replace")
            self.assertTrue(call.kwargs["check"])
        self.assertEqual(run.call_args_list[0].args[0][:3], ["git", "merge-base", "--is-ancestor"])
        self.assertEqual(run.call_args_list[1].args[0][:3], ["git", "rev-list", "--ancestry-path"])

    def test_git_classification_failure_aborts_without_mutating_data(self):
        frame = pd.DataFrame({
            "git_hash": ["newhash"],
            "Tprobe_core_center_max": [87.0],
        })
        previous = collect_wave.PROBE_FIX_HASHES_OK
        collect_wave.PROBE_FIX_HASHES_OK = None
        try:
            with mock.patch("subprocess.run", side_effect=OSError("git unavailable")):
                with self.assertRaisesRegex(RuntimeError, "ancestry classification failed"):
                    collect_wave.sanitize_bad_probes(frame)
        finally:
            collect_wave.PROBE_FIX_HASHES_OK = previous
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
        ])

        normalized, count = collect_wave.normalize_thermal_validity(frame)

        self.assertEqual(count, 2)
        self.assertEqual(normalized["project_name"].tolist(), frame["project_name"].tolist())
        self.assertEqual(normalized["thermal_solved"].tolist(), [0, 1, 0])
        self.assertEqual(normalized["result_valid_thermal"].iloc[0], 0)
        self.assertTrue(pd.isna(normalized["result_valid_thermal"].iloc[1]))
        self.assertEqual(normalized["result_valid_thermal"].iloc[2], 0)


if __name__ == "__main__":
    unittest.main()
