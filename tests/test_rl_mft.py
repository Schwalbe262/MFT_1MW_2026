from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from unittest import mock

from rl_mft.parameters import propose_batch
from rl_mft.orchestrator import build_parser, run_loop
from rl_mft.reward import attach_rewards, compute_reward, load_reward_config
from rl_mft.scheduler_client import SlurmSchedulerConfig, fetch_remote_result_csv
from rl_mft.scheduler_client import submit_dynamic_batch


class RlMftTests(unittest.TestCase):
    def test_propose_batch_creates_valid_candidates(self) -> None:
        candidates = propose_batch(loop=1, batch_size=3)
        self.assertEqual(len(candidates), 3)
        self.assertEqual(len({candidate.candidate_id for candidate in candidates}), 3)
        self.assertIn("w1", candidates[0].parameters)

    def test_reward_penalizes_loss(self) -> None:
        good = {"Lmt": "10", "k": "0.8", "Tx_loss": "100", "Rx_loss": "100", "Llt": "1", "Llr": "1"}
        bad = {"Lmt": "10", "k": "0.8", "Tx_loss": "1000", "Rx_loss": "1000", "Llt": "1", "Llr": "1"}
        self.assertGreater(compute_reward(good), compute_reward(bad))

    def test_failed_rows_get_large_penalty(self) -> None:
        rows = attach_rewards([{"status": "failed", "candidate_id": "x"}])
        self.assertLess(float(rows[0]["reward"]), -1e8)

    def test_reward_config_changes_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reward.json"
            path.write_text(json.dumps({"maximize": {"k": 1.0}, "minimize": {}, "targets": {}, "failed_reward": -5}), encoding="utf-8")
            config = load_reward_config(path)
            self.assertEqual(compute_reward({"k": "2", "Lmt": "100"}, config), 2.0)
            self.assertEqual(attach_rewards([{"status": "failed"}], config)[0]["reward"], "-5")

    def test_fetch_remote_result_csv_writes_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "results.csv"
            with mock.patch("rl_mft.scheduler_client._request_text", return_value="candidate_id,reward\nx,1\n"):
                ok = fetch_remote_result_csv(SlurmSchedulerConfig(base_url="http://scheduler"), "10", output)
            self.assertTrue(ok)
            self.assertEqual(output.read_text(encoding="utf-8"), "candidate_id,reward\nx,1\n")

    def test_submit_dynamic_batch_clears_shared_result_file(self) -> None:
        captured = {}

        class FakeResponse:
            def read(self):
                return b""

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

        def fake_urlopen(request, timeout=30):
            captured["data"] = request.data.decode("utf-8")
            return FakeResponse()

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen), \
            mock.patch("rl_mft.scheduler_client._request_json", return_value=[{"id": 1, "job_name": "mft-rl-loop-0001-1-1"}]):
            job_ids = submit_dynamic_batch(SlurmSchedulerConfig(), loop=1, candidate_count=1, candidates_jsonl_content="{}")
        self.assertEqual(job_ids, ["1"])
        self.assertIn("rm+-f+simulation_results.csv+simulation_results.csv.lock", captured["data"])

    def test_slurm_loop_scores_fetched_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("rl_mft.records.RUNS_DIR", Path(tmp) / "rl_runs"), \
                mock.patch("rl_mft.records.STATE_PATH", Path(tmp) / "rl_runs" / "state.json"), \
                mock.patch("rl_mft.records.NOTE_PATH", Path(tmp) / "note.md"), \
                mock.patch("rl_mft.records.INSIGHT_PATH", Path(tmp) / "insight.md"), \
                mock.patch("rl_mft.orchestrator.submit_dynamic_batch", return_value=["101"]), \
                mock.patch("rl_mft.orchestrator.wait_for_jobs", return_value=[{"id": 101, "status": "completed"}]), \
                mock.patch("rl_mft.orchestrator.fetch_remote_result_csv") as fetch:
                Path(tmp, "note.md").write_text("# notes\n", encoding="utf-8")
                Path(tmp, "insight.md").write_text("# insights\n", encoding="utf-8")

                def write_result(config, job_id, output_path, remote_file="simulation_results.csv"):
                    output_path.write_text("candidate_id,Lmt,k,Tx_loss,Rx_loss,Llt,Llr\nL0001-C0001,10,0.8,100,100,1,1\n", encoding="utf-8")
                    return True

                fetch.side_effect = write_result
                args = build_parser().parse_args(["--backend", "slurm", "--batch-size", "1", "--wait"])
                summary = run_loop(args)
                self.assertEqual(summary.status, "completed")
                self.assertEqual(summary.completed, 1)
                self.assertEqual(summary.best_candidate_id, "L0001-C0001")
                self.assertGreater(summary.best_reward, 0)
                state = json.loads((Path(tmp) / "rl_runs" / "state.json").read_text(encoding="utf-8"))
                self.assertEqual(state["live_best_candidate_id"], "L0001-C0001")
                self.assertGreater(state["live_best_reward"], 0)
                self.assertEqual(state["live_best_outputs"]["Llt"], 1.0)
                self.assertEqual(summary.best_outputs["Tx_loss"], 100.0)

    def test_slurm_loop_discards_stale_result_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("rl_mft.records.RUNS_DIR", Path(tmp) / "rl_runs"), \
                mock.patch("rl_mft.records.STATE_PATH", Path(tmp) / "rl_runs" / "state.json"), \
                mock.patch("rl_mft.records.NOTE_PATH", Path(tmp) / "note.md"), \
                mock.patch("rl_mft.records.INSIGHT_PATH", Path(tmp) / "insight.md"), \
                mock.patch("rl_mft.orchestrator.submit_dynamic_batch", return_value=["101"]), \
                mock.patch("rl_mft.orchestrator.wait_for_jobs", return_value=[{"id": 101, "status": "completed"}]), \
                mock.patch("rl_mft.orchestrator.fetch_remote_result_csv") as fetch:
                Path(tmp, "note.md").write_text("# notes\n", encoding="utf-8")
                Path(tmp, "insight.md").write_text("# insights\n", encoding="utf-8")

                def write_stale_result(config, job_id, output_path, remote_file="simulation_results.csv"):
                    output_path.write_text("candidate_id,Lmt,k,Tx_loss,Rx_loss,Llt,Llr\nL9999-C0001,10,0.8,100,100,1,1\n", encoding="utf-8")
                    return True

                fetch.side_effect = write_stale_result
                args = build_parser().parse_args(["--backend", "slurm", "--batch-size", "1", "--wait"])
                summary = run_loop(args)
                self.assertEqual(summary.completed, 0)
                self.assertEqual(summary.failed, 1)
                self.assertIsNone(summary.best_candidate_id)
                self.assertIn("Discarded 1 stale result row", summary.message or "")


if __name__ == "__main__":
    unittest.main()
