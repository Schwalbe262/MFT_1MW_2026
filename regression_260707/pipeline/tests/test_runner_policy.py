from concurrent.futures import ThreadPoolExecutor
import inspect
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np

from regression_260707.pipeline.artifacts import GenerationStore
from regression_260707.pipeline.policy import (
    next_training_checkpoint,
    tuning_decision,
)
from regression_260707.pipeline.queue import DurableJobQueue
from regression_260707.pipeline.runner import JobRunner


class PolicyTests(unittest.TestCase):
    def test_checkpoint_and_tuning_gates(self):
        self.assertEqual(next_training_checkpoint(2188, 2000), None)
        self.assertEqual(next_training_checkpoint(3000, 2000), 3000)
        self.assertFalse(tuning_decision(3999).due)
        self.assertTrue(tuning_decision(4000).due)
        self.assertFalse(tuning_decision(5999, last_tuned_rows=4000).due)
        self.assertTrue(tuning_decision(6000, last_tuned_rows=4000).due)
        drift = tuning_decision(4100, last_tuned_rows=4000, drift_detected=True)
        self.assertTrue(drift.due)
        self.assertEqual(drift.reason, "dataset_drift")

    def test_tuned_params_are_content_pinned_and_forwarded_to_training(self):
        from regression_260707.training.checkpoint_contract import (
            checkpoint_contract_identity,
        )
        from regression_260707.training.checkpoint_orchestrator import (
            training_commands,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profile.json"
            thresholds = root / "thresholds.json"
            quality = root / "quality.py"
            targets = root / "targets.py"
            params = root / "params.json"
            profile.write_text('{"profile":1}', encoding="utf-8")
            thresholds.write_text('{"threshold":1}', encoding="utf-8")
            quality.write_text("QUALITY=1\n", encoding="utf-8")
            targets.write_text("TARGETS=1\n", encoding="utf-8")
            params.write_text('{"lightgbm":{"x":{"params":{"n_estimators":4}}}}', encoding="utf-8")
            identity = checkpoint_contract_identity(
                profile, thresholds, quality, targets, params=params
            )
            command = training_commands(
                "snapshot", "curve", "registry", 200,
                str(profile.resolve()), 4000, "metrics", "candidate",
                str(params.resolve()),
            )[1]

            self.assertEqual(identity["schema_version"], 3)
            self.assertRegex(identity["params_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                command[command.index("--params") + 1], str(params.resolve())
            )

    def test_pipeline_consumers_pin_model_generation_against_pruning(self):
        from regression_260707.training.checkpoint_orchestrator import (
            _pinned_training_runs,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline_root = root / "custom-pipeline"
            queue = DurableJobQueue(pipeline_root / "jobs.sqlite3")
            queue.enqueue(
                "optimize", "model-run-a", {"command": ["optimize"]},
                input_generation="model:run-a",
            )
            with mock.patch.dict(
                "os.environ", {"MFT_PIPELINE_ROOT": str(pipeline_root)}
            ):
                pinned = _pinned_training_runs(root / "runtime")
            self.assertIn("run-a", pinned)


class RunnerTests(unittest.TestCase):
    def test_runner_publishes_output_and_unblocks_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = DurableJobQueue(root / "jobs.sqlite3")
            store = GenerationStore(root / "artifacts")
            parent = queue.enqueue(
                "optimize",
                "model-a",
                {
                    "command": [
                        sys.executable,
                        "-c",
                        (
                            "from pathlib import Path; "
                            "p=Path(r'{work_dir}')/'result'; p.mkdir(); "
                            "(p/'front.csv').write_text('x,y\\n1,2\\n')"
                        ),
                    ],
                    "publish": {
                        "kind": "optimization",
                        "source": "{work_dir}/result",
                    },
                },
                now=1,
            )
            child = queue.enqueue(
                "verify_standard",
                "model-a",
                {
                    "command": [
                        sys.executable, "-c",
                        "import pathlib; assert pathlib.Path(r'{dependency_optimize_output}').is_dir()",
                    ],
                    "dependency_kinds": {"optimize": "optimization"},
                },
                dependencies=[parent.id],
                now=1,
            )
            runner = JobRunner(queue, store, root / "work", owner="worker")
            first = runner.run_once(["optimize"])
            self.assertEqual(first.state, "succeeded")
            self.assertTrue(Path(first.output_generation).is_dir())
            second = runner.run_once(["verify_standard"])
            self.assertEqual(second.id, child.id)
            self.assertEqual(second.state, "succeeded")

    def test_lease_is_renewed_while_publishing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = DurableJobQueue(root / "jobs.sqlite3")
            store = GenerationStore(root / "artifacts")
            job = queue.enqueue(
                "optimize",
                "slow-publish",
                {
                    "command": [
                        sys.executable,
                        "-c",
                        (
                            "from pathlib import Path; "
                            "p=Path(r'{work_dir}')/'result'; p.mkdir(); "
                            "(p/'value.txt').write_text('ok')"
                        ),
                    ],
                    "publish": {
                        "kind": "optimization",
                        "source": "{work_dir}/result",
                    },
                },
            )
            runner = JobRunner(queue, store, root / "work", owner="worker")
            runner.lease_seconds = 2.0
            original = store.publish_tree

            def slow_publish(*args, **kwargs):
                time.sleep(3)
                return original(*args, **kwargs)

            with mock.patch.object(store, "publish_tree", side_effect=slow_publish):
                completed = runner.run_once(["optimize"])
            self.assertEqual(completed.id, job.id)
            self.assertEqual(completed.state, "succeeded")

    def test_result_json_must_name_an_authenticated_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = DurableJobQueue(root / "jobs.sqlite3")
            store = GenerationStore(root / "artifacts")
            outside = root / "outside"
            outside.mkdir()
            job = queue.enqueue(
                "tune",
                "forged-result",
                {
                    "command": [
                        sys.executable,
                        "-c",
                        (
                            "import json; from pathlib import Path; "
                            "Path(r'{work_dir}/result.json').write_text("
                            "json.dumps({'generation_path':r'"
                            + str(outside)
                            + "','generation_id':'"
                            + "0" * 64
                            + "'}))"
                        ),
                    ],
                    "result_json": "{work_dir}/result.json",
                    "result_output_key": "generation_path",
                    "result_generation_kind": "tuning",
                    "result_generation_id_key": "generation_id",
                    "retry": False,
                },
            )
            result = JobRunner(
                queue, store, root / "work", owner="worker"
            ).run_once(["tune"])
            self.assertEqual(result.id, job.id)
            self.assertEqual(result.state, "failed")
            self.assertIn("generation escapes store root", result.terminal_reason)

    def test_stop_event_terminates_command_tree_and_requeues_owned_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = DurableJobQueue(root / "jobs.sqlite3")
            store = GenerationStore(root / "artifacts")
            job = queue.enqueue(
                "optimize",
                "shutdown",
                {
                    "command": [
                        sys.executable, "-c", "import time; time.sleep(60)"
                    ],
                    "retry_backoff_seconds": 0,
                },
            )
            stop = threading.Event()
            runner = JobRunner(queue, store, root / "work", owner="worker")
            output = []
            thread = threading.Thread(
                target=lambda: output.append(
                    runner.run_once(["optimize"], stop_event=stop)
                )
            )
            thread.start()
            deadline = time.time() + 10
            while time.time() < deadline:
                if queue.get(job.id).state == "running":
                    break
                time.sleep(0.05)
            stop.set()
            thread.join(15)
            self.assertFalse(thread.is_alive())
            self.assertEqual(output[0].state, "retry_wait")
            self.assertEqual(output[0].terminal_reason, "worker_shutdown")

    def test_declared_deterministic_exit_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = DurableJobQueue(root / "jobs.sqlite3")
            store = GenerationStore(root / "artifacts")
            job = queue.enqueue(
                "optimize",
                "deterministic-infeasible",
                {
                    "command": [sys.executable, "-c", "raise SystemExit(42)"],
                    "retry": True,
                    "retry_backoff_seconds": 0,
                    "non_retryable_exit_codes": [42],
                },
                max_attempts=3,
            )

            result = JobRunner(
                queue, store, root / "work", owner="worker"
            ).run_once(["optimize"])

            self.assertEqual(result.id, job.id)
            self.assertEqual(result.state, "failed")
            self.assertEqual(result.attempt, 1)
            self.assertIn("command_exit:42;non_retryable=true", result.terminal_reason)

    def test_unlisted_exit_remains_retryable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = DurableJobQueue(root / "jobs.sqlite3")
            store = GenerationStore(root / "artifacts")
            job = queue.enqueue(
                "optimize",
                "transient-failure",
                {
                    "command": [sys.executable, "-c", "raise SystemExit(41)"],
                    "retry": True,
                    "retry_backoff_seconds": 0,
                    "non_retryable_exit_codes": [42],
                },
                max_attempts=3,
            )

            result = JobRunner(
                queue, store, root / "work", owner="worker"
            ).run_once(["optimize"])

            self.assertEqual(result.id, job.id)
            self.assertEqual(result.state, "retry_wait")
            self.assertEqual(result.attempt, 1)
            self.assertIn("command_exit:41;non_retryable=false", result.terminal_reason)

    def test_non_retryable_exit_codes_must_be_an_integer_list(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = DurableJobQueue(root / "jobs.sqlite3")
            store = GenerationStore(root / "artifacts")
            job = queue.enqueue(
                "optimize",
                "invalid-non-retryable-codes",
                {
                    "command": [sys.executable, "-c", "raise SystemExit(99)"],
                    "non_retryable_exit_codes": "42",
                },
                max_attempts=3,
            )

            result = JobRunner(
                queue, store, root / "work", owner="worker"
            ).run_once(["optimize"])

            self.assertEqual(result.id, job.id)
            self.assertEqual(result.state, "failed")
            self.assertEqual(result.attempt, 1)
            self.assertEqual(
                result.terminal_reason, "invalid non_retryable_exit_codes payload"
            )


class NsgaParallelTests(unittest.TestCase):
    def test_restarts_are_parallel_bounded_and_returned_in_seed_order(self):
        from regression_260707.optimization import run_nsga2

        class Result:
            def __init__(self, seed):
                self.X = np.array([[seed]])
                self.F = np.array([[seed, seed]])
                self.algorithm = type("Algorithm", (), {"n_gen": seed})()

        with mock.patch.object(
            run_nsga2, "run_one", side_effect=lambda problem, seed, **kw: Result(seed)
        ):
            results = run_nsga2.run_restarts(
                object(), 4, 10, workers=4, executor_factory=ThreadPoolExecutor
            )
        self.assertEqual([int(item[0][0, 0]) for item in results], [1000, 1001, 1002, 1003])
        with self.assertRaisesRegex(ValueError, "between 1 and 4"):
            run_nsga2.run_restarts(object(), 16, 200, workers=5)

    def test_al_driver_no_longer_promotes_models(self):
        from regression_260707 import al_driver

        source = inspect.getsource(al_driver.stage_train)
        self.assertIn("load_active_generation", source)
        self.assertNotIn("promote_generation", source)


class ActiveLearningFreshnessTests(unittest.TestCase):
    def test_checkpoint_authenticates_named_source_generation_before_training(self):
        from regression_260707.training.checkpoint_orchestrator import (
            _authenticate_source_dataset_generation,
        )
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.parquet"
            source.write_bytes(b"immutable")
            generation = GenerationStore(root / "artifacts").publish_files(
                "dataset", {"train.parquet": source}
            )
            authenticated = generation.path / "train.parquet"
            identity = f"dataset:{generation.generation_id}"
            self.assertEqual(
                _authenticate_source_dataset_generation(authenticated, identity),
                hashlib.sha256(authenticated.read_bytes()).hexdigest(),
            )
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                _authenticate_source_dataset_generation(
                    authenticated, "dataset:" + "0" * 64
                )

    def test_source_dataset_generation_is_content_authenticated(self):
        from regression_260707 import al_driver

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.parquet"
            source.write_bytes(b"immutable-source")
            generation = GenerationStore(root / "artifacts").publish_files(
                "dataset", {"train.parquet": source}
            )
            source_path = generation.path / "train.parquet"
            import hashlib

            digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            identity = f"dataset:{generation.generation_id}"
            report = {
                "source_dataset_path": str(source_path),
                "source_dataset_sha256": digest,
                "source_dataset_generation": identity,
            }
            quality = {
                "source_dataset_sha256": digest,
                "source_dataset_generation": identity,
            }
            self.assertEqual(
                al_driver._authenticate_training_source_dataset(report, quality),
                (str(source_path), identity),
            )
            quality["source_dataset_generation"] = "dataset:" + "0" * 64
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                al_driver._authenticate_training_source_dataset(report, quality)

    def test_train_waits_for_new_promotion_and_only_then_marks_retrain_done(self):
        from regression_260707 import al_driver
        import training.train_models as train_models

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dataset = root / "strict.parquet"
            model_dataset.write_bytes(b"strict")
            import hashlib

            dataset_sha = hashlib.sha256(model_dataset.read_bytes()).hexdigest()
            quality_sha = hashlib.sha256(
                Path(al_driver.QUALITY_THRESHOLDS_PATH).read_bytes()
            ).hexdigest()
            report = {
                "training_run_id": "old-run",
                "strict_full_rows": 3000,
                "dataset_path": str(model_dataset),
                "dataset_sha256": dataset_sha,
                "source_dataset_sha256": "c" * 64,
            }
            quality = {
                "passed": True,
                "training_run_id": "old-run",
                "dataset_sha256": dataset_sha,
                "solver_revision": "a" * 40,
                "library_revision": "b" * 40,
                "quality_thresholds_sha256": quality_sha,
            }
            active = {
                "report": report,
                "quality": quality,
                "generation": str(root / "generation"),
            }
            state = {
                "round": 2,
                "training_run_id": "old-run",
                "post_convergence_retrain_pending": True,
            }
            with mock.patch.object(
                train_models, "load_active_generation", return_value=active
            ), mock.patch.object(al_driver, "_bind_runtime_identity"), mock.patch.multiple(
                al_driver,
                PINNED_SOLVER_REVISION="a" * 40,
                PINNED_LIBRARY_REVISION="b" * 40,
                AL_ROOT=str(root / "al_rounds"),
                RUNTIME_ROOT=str(root),
                OUTPUT_ROOT=str(root),
            ):
                with self.assertRaisesRegex(RuntimeError, "checkpoint_not_ready"):
                    al_driver.stage_train(state)
                self.assertNotIn("post_convergence_retrain_done", state)

                report["training_run_id"] = "new-run"
                quality["training_run_id"] = "new-run"
                with mock.patch.object(
                    al_driver,
                    "_authenticate_training_source_dataset",
                    return_value=(str(root / "source.parquet"), "dataset:" + "d" * 64),
                ):
                    al_driver.stage_train(state)
            self.assertEqual(state["training_run_id"], "new-run")
            self.assertTrue(state["post_convergence_retrain_done"])
            self.assertNotIn("post_convergence_retrain_pending", state)


if __name__ == "__main__":
    unittest.main()
