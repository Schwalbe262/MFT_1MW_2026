from pathlib import Path
import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from regression_260707.pipeline.artifacts import GenerationStore
from regression_260707.pipeline.controller import ContinuousController
from regression_260707.pipeline.orchestrator import (
    PipelineOrchestrator,
    descriptor_from_active_registry,
)
from regression_260707.pipeline.queue import DurableJobQueue


class OrchestratorTests(unittest.TestCase):
    def test_checkpoint_output_root_is_separate_from_identity_run_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            (runtime / "training").mkdir(parents=True)
            dataset = root / "train.parquet"
            dataset.write_bytes(b"strict checkpoint dataset")
            local_output = root / "local-checkpoint-output"
            queue = DurableJobQueue(root / "pipeline" / "jobs.sqlite3")
            orchestrator = PipelineOrchestrator(
                queue,
                GenerationStore(root / "pipeline" / "artifacts"),
                runtime,
                python=str(root / "python"),
                checkpoint_output_root=local_output,
            )

            result = orchestrator.plan_cycle(
                dataset_path=dataset,
                strict_full_rows=3000,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                now=1200,
            )

            command = queue.get(result.jobs["train"]).payload["command"]
            output_index = command.index("--output-root")
            run_index = command.index("--run-root")
            self.assertEqual(
                Path(command[output_index + 1]), local_output.resolve()
            )
            self.assertEqual(
                Path(command[run_index + 1]).parent,
                (runtime / "training" / "checkpoint_runs").resolve(),
            )
            self.assertNotEqual(
                Path(command[output_index + 1]),
                Path(command[run_index + 1]),
            )
            self.assertRegex(
                queue.get(result.jobs["train"]).idempotency_key,
                r"-e[0-9a-f]{12}$",
            )
            threads_index = command.index("--model-threads")
            self.assertEqual(command[threads_index + 1], "24")

    def test_checkpoint_route_change_creates_a_distinct_idempotency_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            (runtime / "training").mkdir(parents=True)
            dataset = root / "train.parquet"
            dataset.write_bytes(b"same immutable checkpoint dataset")
            queue = DurableJobQueue(root / "pipeline" / "jobs.sqlite3")
            store = GenerationStore(root / "pipeline" / "artifacts")

            first = PipelineOrchestrator(
                queue,
                store,
                runtime,
                checkpoint_output_root=root / "first-output",
            ).plan_cycle(
                dataset_path=dataset,
                strict_full_rows=3000,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                now=1200,
            )
            second = PipelineOrchestrator(
                queue,
                store,
                runtime,
                checkpoint_output_root=root / "second-output",
            ).plan_cycle(
                dataset_path=dataset,
                strict_full_rows=3000,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                now=1200,
            )

            self.assertNotEqual(first.jobs["train"], second.jobs["train"])
            self.assertNotEqual(
                queue.get(first.jobs["train"]).idempotency_key,
                queue.get(second.jobs["train"]).idempotency_key,
            )

    def test_checkpoint_thread_budget_changes_execution_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            (runtime / "training").mkdir(parents=True)
            dataset = root / "train.parquet"
            dataset.write_bytes(b"thread-budget checkpoint dataset")
            queue = DurableJobQueue(root / "pipeline" / "jobs.sqlite3")
            store = GenerationStore(root / "pipeline" / "artifacts")

            first = PipelineOrchestrator(
                queue, store, runtime
            ).plan_cycle(
                dataset_path=dataset,
                strict_full_rows=3000,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                model_threads=4,
                now=1200,
            )
            second = PipelineOrchestrator(
                queue, store, runtime
            ).plan_cycle(
                dataset_path=dataset,
                strict_full_rows=3000,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                model_threads=8,
                now=1200,
            )

            first_job = queue.get(first.jobs["train"])
            second_job = queue.get(second.jobs["train"])
            self.assertNotEqual(
                first_job.idempotency_key, second_job.idempotency_key
            )
            second_threads = second_job.payload["command"].index(
                "--model-threads"
            )
            self.assertEqual(
                second_job.payload["command"][second_threads + 1], "8"
            )

    def test_checkpoint_output_root_defaults_to_runtime_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            (runtime / "training").mkdir(parents=True)
            dataset = root / "train.parquet"
            dataset.write_bytes(b"default checkpoint dataset")
            queue = DurableJobQueue(root / "pipeline" / "jobs.sqlite3")
            result = PipelineOrchestrator(
                queue,
                GenerationStore(root / "pipeline" / "artifacts"),
                runtime,
            ).plan_cycle(
                dataset_path=dataset,
                strict_full_rows=3000,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                now=1200,
            )

            command = queue.get(result.jobs["train"]).payload["command"]
            output_index = command.index("--output-root")
            self.assertEqual(
                Path(command[output_index + 1]),
                (runtime / "training").resolve(),
            )

    def test_4k_cycle_orders_tuning_before_training_and_overlaps_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            (runtime / "training").mkdir(parents=True)
            (runtime / "campaign").mkdir()
            dataset = root / "train.parquet"
            dataset.write_bytes(b"immutable test dataset")
            queue = DurableJobQueue(root / "pipeline" / "jobs.sqlite3")
            store = GenerationStore(root / "pipeline" / "artifacts")
            orchestrator = PipelineOrchestrator(
                queue, store, runtime, python=str(root / "python")
            )
            result = orchestrator.plan_cycle(
                dataset_path=dataset,
                strict_full_rows=4000,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                now=1200,
            )

            self.assertEqual(set(result.jobs), {"collect", "tune", "train"})
            collect = queue.get(result.jobs["collect"])
            self.assertTrue(
                collect.idempotency_key.startswith("collector-v4-e")
            )
            command = collect.payload["command"]
            self.assertEqual(
                [
                    command[index + 1]
                    for index, value in enumerate(command)
                    if value == "--extra-prefix"
                ],
                ["mft-1to3", "mft-1x3", "mft-mixed", "mft-9way"],
            )
            train_dependencies = queue.dependencies(result.jobs["train"])
            self.assertEqual([job.id for job in train_dependencies], [result.jobs["tune"]])
            tune_command = queue.get(result.jobs["tune"]).payload["command"]
            thread_option = tune_command.index("--model-threads")
            worker_option = tune_command.index("--job-workers")
            budget_option = tune_command.index("--max-model-thread-budget")
            self.assertEqual(tune_command[thread_option + 1], "1")
            self.assertEqual(tune_command[worker_option + 1], "24")
            self.assertEqual(tune_command[budget_option + 1], "24")
            self.assertIsNotNone(queue.claim("collector", job_types=["collect"], now=1201))
            self.assertIsNotNone(queue.claim("tuner", job_types=["tune"], now=1201))
            self.assertIsNone(queue.claim("trainer", job_types=["train"], now=1201))

    def test_collector_payload_pins_canonical_solver_git_repo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "immutable-deployment"
            runtime.mkdir()
            canonical_repo = root / "canonical-solver-repo"
            canonical_repo.mkdir()
            dataset = root / "train.parquet"
            dataset.write_bytes(b"collector dataset")
            live_dataset = root / "live" / "train.parquet"
            live_dataset.parent.mkdir()
            queue = DurableJobQueue(root / "pipeline" / "jobs.sqlite3")

            with mock.patch.dict(
                os.environ,
                {"MFT_SOLVER_GIT_REPO": str(canonical_repo)},
                clear=False,
            ):
                result = PipelineOrchestrator(
                    queue,
                    GenerationStore(root / "pipeline" / "artifacts"),
                    runtime,
                ).plan_cycle(
                    dataset_path=dataset,
                    dataset_series_path=live_dataset,
                    strict_full_rows=0,
                    solver_revision="a" * 40,
                    library_revision="b" * 40,
                    now=1200,
                )

            collector = queue.get(result.jobs["collect"])
            self.assertEqual(
                collector.payload["env"]["MFT_SOLVER_GIT_REPO"],
                str(canonical_repo.resolve()),
            )
            self.assertEqual(
                collector.payload["env"]["MFT_COLLECTOR_DATASET_DIR"],
                str(live_dataset.resolve().parent),
            )

    def test_running_collector_is_reused_across_deployment_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical_repo = root / "canonical-solver-repo"
            canonical_repo.mkdir()
            runtimes = [root / "deploy-a", root / "deploy-b"]
            for runtime in runtimes:
                (runtime / "campaign").mkdir(parents=True)
            dataset = root / "train.parquet"
            dataset.write_bytes(b"same collection demand")
            queue = DurableJobQueue(root / "pipeline" / "jobs.sqlite3")
            store = GenerationStore(root / "pipeline" / "artifacts")

            first = PipelineOrchestrator(
                queue,
                store,
                runtimes[0],
                solver_git_repo=canonical_repo,
            ).plan_cycle(
                dataset_path=dataset,
                strict_full_rows=0,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                now=1200,
            )
            running = queue.claim(
                "collector",
                job_types=["collect"],
                lease_seconds=1000,
                now=1201,
            )
            second = PipelineOrchestrator(
                queue,
                store,
                runtimes[1],
                solver_git_repo=canonical_repo,
            ).plan_cycle(
                dataset_path=dataset,
                strict_full_rows=0,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                now=1202,
            )

            self.assertEqual(first.jobs["collect"], running.id)
            self.assertEqual(second.jobs["collect"], running.id)
            self.assertEqual(queue.get(running.id).state, "running")
            active = queue.list(
                states=["running", "queued", "retry_wait"],
                job_types=["collect"],
            )
            self.assertEqual([job.id for job in active], [running.id])

    def test_queued_and_retry_collector_are_reused_across_deployments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical_repo = root / "canonical-solver-repo"
            canonical_repo.mkdir()
            runtimes = [root / "deploy-a", root / "deploy-b"]
            for runtime in runtimes:
                (runtime / "campaign").mkdir(parents=True)
            dataset = root / "train.parquet"
            dataset.write_bytes(b"same queued collection demand")
            queue = DurableJobQueue(root / "pipeline" / "jobs.sqlite3")
            store = GenerationStore(root / "pipeline" / "artifacts")
            first_orchestrator = PipelineOrchestrator(
                queue, store, runtimes[0], solver_git_repo=canonical_repo
            )
            second_orchestrator = PipelineOrchestrator(
                queue, store, runtimes[1], solver_git_repo=canonical_repo
            )
            first = first_orchestrator.plan_cycle(
                dataset_path=dataset,
                strict_full_rows=0,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                now=1200,
            )
            queued = second_orchestrator.plan_cycle(
                dataset_path=dataset,
                strict_full_rows=0,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                now=1201,
            )
            self.assertEqual(queued.jobs["collect"], first.jobs["collect"])

            running = queue.claim(
                "collector",
                job_types=["collect"],
                lease_seconds=1000,
                now=1202,
            )
            retry = queue.fail(
                running.id,
                running.owner_lease,
                "transient",
                retry=True,
                base_backoff_seconds=100,
                now=1203,
            )
            self.assertEqual(retry.state, "retry_wait")
            reused_retry = second_orchestrator.plan_cycle(
                dataset_path=dataset,
                strict_full_rows=0,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                now=1204,
            )
            self.assertEqual(reused_retry.jobs["collect"], retry.id)
            self.assertEqual(queue.get(retry.id).state, "retry_wait")

    def test_running_legacy_v4_collector_is_rollout_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical_repo = root / "canonical-solver-repo"
            canonical_repo.mkdir()
            runtimes = [root / "deploy-v4", root / "deploy-v5"]
            for runtime in runtimes:
                (runtime / "campaign").mkdir(parents=True)
            dataset = root / "train.parquet"
            dataset.write_bytes(b"legacy rollout demand")
            queue = DurableJobQueue(root / "pipeline" / "jobs.sqlite3")
            store = GenerationStore(root / "pipeline" / "artifacts")
            first = PipelineOrchestrator(
                queue, store, runtimes[0], solver_git_repo=canonical_repo
            ).plan_cycle(
                dataset_path=dataset,
                strict_full_rows=0,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                now=1200,
            )
            legacy = queue.get(first.jobs["collect"])
            legacy_payload = json.loads(json.dumps(legacy.payload))
            legacy_payload.pop("collector_demand", None)
            legacy_payload.pop("execution_contract", None)
            legacy_payload["command"][1] = str(
                runtimes[0] / "campaign" / "collect_wave.py"
            )
            connection = queue._connect()
            try:
                connection.execute(
                    "UPDATE jobs SET idempotency_key=?, input_generation=NULL, "
                    "coalesce_key=NULL, payload_json=? WHERE id=?",
                    (
                        "collector-v3-elegacy-window-2",
                        json.dumps(
                            legacy_payload,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        legacy.id,
                    ),
                )
            finally:
                connection.close()
            running = queue.claim(
                "legacy-collector",
                job_types=["collect"],
                lease_seconds=1000,
                now=1201,
            )
            rollout = PipelineOrchestrator(
                queue, store, runtimes[1], solver_git_repo=canonical_repo
            ).plan_cycle(
                dataset_path=dataset,
                strict_full_rows=0,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                now=1202,
            )
            self.assertEqual(rollout.jobs["collect"], running.id)
            self.assertEqual(queue.get(running.id).state, "running")

    def test_running_collector_keeps_only_latest_window_followup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            (runtime / "campaign").mkdir(parents=True)
            canonical_repo = root / "canonical-solver-repo"
            canonical_repo.mkdir()
            dataset = root / "train.parquet"
            dataset.write_bytes(b"stable collection input")
            queue = DurableJobQueue(root / "pipeline" / "jobs.sqlite3")
            orchestrator = PipelineOrchestrator(
                queue,
                GenerationStore(root / "pipeline" / "artifacts"),
                runtime,
                solver_git_repo=canonical_repo,
            )
            first = orchestrator.plan_cycle(
                dataset_path=dataset,
                strict_full_rows=0,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                now=1200,
            )
            running = queue.claim(
                "collector",
                job_types=["collect"],
                lease_seconds=5000,
                now=1201,
            )
            second = orchestrator.plan_cycle(
                dataset_path=dataset,
                strict_full_rows=0,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                now=1800,
            )
            third = orchestrator.plan_cycle(
                dataset_path=dataset,
                strict_full_rows=0,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                now=2400,
            )

            self.assertEqual(first.jobs["collect"], running.id)
            self.assertNotEqual(second.jobs["collect"], running.id)
            self.assertEqual(
                queue.get(second.jobs["collect"]).state,
                "cancelled",
            )
            self.assertEqual(
                queue.get(second.jobs["collect"]).terminal_reason,
                f"superseded_by:{third.jobs['collect']}",
            )
            active = queue.list(
                states=["running", "queued", "retry_wait"],
                job_types=["collect"],
            )
            self.assertEqual(
                [(job.id, job.state) for job in active],
                [(running.id, "running"), (third.jobs["collect"], "queued")],
            )

    def test_dataset_change_coalesces_one_same_window_followup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            (runtime / "campaign").mkdir(parents=True)
            canonical_repo = root / "canonical-solver-repo"
            canonical_repo.mkdir()
            dataset = root / "train.parquet"
            dataset.write_bytes(b"generation one")
            queue = DurableJobQueue(root / "pipeline" / "jobs.sqlite3")
            orchestrator = PipelineOrchestrator(
                queue,
                GenerationStore(root / "pipeline" / "artifacts"),
                runtime,
                solver_git_repo=canonical_repo,
            )
            first = orchestrator.plan_cycle(
                dataset_path=dataset,
                strict_full_rows=0,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                now=1200,
            )
            running = queue.claim(
                "collector",
                job_types=["collect"],
                lease_seconds=5000,
                now=1201,
            )
            dataset.write_bytes(b"generation two")
            second = orchestrator.plan_cycle(
                dataset_path=dataset,
                strict_full_rows=0,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                now=1202,
            )
            dataset.write_bytes(b"generation three")
            third = orchestrator.plan_cycle(
                dataset_path=dataset,
                strict_full_rows=0,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                now=1203,
            )

            self.assertEqual(first.jobs["collect"], running.id)
            self.assertEqual(queue.get(second.jobs["collect"]).state, "cancelled")
            latest = queue.get(third.jobs["collect"])
            self.assertEqual(latest.state, "queued")
            self.assertEqual(latest.input_generation, third.dataset_generation)
            active = queue.list(
                states=["running", "queued", "retry_wait"],
                job_types=["collect"],
            )
            self.assertEqual(len(active), 2)
            self.assertEqual(
                {job.state for job in active}, {"running", "queued"}
            )

    def test_model_cycle_enqueues_optimizer_standard_and_fine_dag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            runtime.mkdir()
            dataset = root / "train.parquet"
            model_dataset = root / "strict.parquet"
            quality = root / "quality.json"
            generation = root / "generation"
            library_root = root / "library"
            for path in (dataset, model_dataset, quality):
                path.write_bytes(b"x")
            generation.mkdir()
            library_root.mkdir()
            queue = DurableJobQueue(root / "pipeline" / "jobs.sqlite3")
            orchestrator = PipelineOrchestrator(
                queue,
                GenerationStore(root / "pipeline" / "artifacts"),
                runtime,
                python=str(root / "python"),
            )
            result = orchestrator.plan_cycle(
                dataset_path=dataset,
                strict_full_rows=3000,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                active_model={
                    "training_run_id": "model-g1",
                    "strict_full_rows": 3000,
                    "generation": str(generation),
                    "dataset": str(model_dataset),
                    "quality_status": str(quality),
                },
                verification_commands={
                    "standard": {
                        "adapter": "mft_scheduler_v1",
                        "execute": True,
                        "library_root": str(library_root),
                    },
                    "fine": {
                        "adapter": "mft_scheduler_v1",
                        "execute": True,
                        "library_root": str(library_root),
                    },
                },
                now=1200,
            )
            self.assertTrue(
                {"optimize", "verify_standard", "verify_fine"}.issubset(result.jobs)
            )
            self.assertEqual(
                [job.id for job in queue.dependencies(result.jobs["verify_standard"])],
                [result.jobs["optimize"]],
            )
            self.assertEqual(
                [job.id for job in queue.dependencies(result.jobs["verify_fine"])],
                [result.jobs["verify_standard"]],
            )

    def test_new_model_enqueues_new_pinned_optimizer_without_stopping_old_lane(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            runtime.mkdir()
            dataset = root / "train.parquet"
            dataset.write_bytes(b"live collection")
            queue = DurableJobQueue(root / "pipeline" / "jobs.sqlite3")
            orchestrator = PipelineOrchestrator(
                queue,
                GenerationStore(root / "pipeline" / "artifacts"),
                runtime,
                python=str(root / "python"),
            )

            def active_model(name, rows):
                generation = root / name
                generation.mkdir()
                model_dataset = generation / "strict.parquet"
                quality = generation / "quality_gate.json"
                model_dataset.write_bytes(name.encode("ascii"))
                quality.write_bytes(b"{}")
                return {
                    "training_run_id": name,
                    "strict_full_rows": rows,
                    "generation": str(generation),
                    "dataset": str(model_dataset),
                    "quality_status": str(quality),
                }

            first_model = active_model("model-g1", 3000)
            first = orchestrator.plan_cycle(
                dataset_path=dataset,
                strict_full_rows=3000,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                active_model=first_model,
                now=1200,
            )
            running = queue.claim(
                "optimizer-g1", job_types=["optimize"], now=1201
            )
            self.assertEqual(running.id, first.jobs["optimize"])

            second_model = active_model("model-g2", 4500)
            second = orchestrator.plan_cycle(
                dataset_path=dataset,
                strict_full_rows=4500,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                active_model=second_model,
                now=1202,
            )

            first_job = queue.get(first.jobs["optimize"])
            second_job = queue.get(second.jobs["optimize"])
            self.assertEqual(first_job.state, "running")
            self.assertEqual(second_job.state, "queued")
            self.assertNotEqual(first_job.id, second_job.id)
            self.assertEqual(first_job.input_generation, "model:model-g1")
            self.assertEqual(second_job.input_generation, "model:model-g2")
            first_command = first_job.payload["command"]
            second_command = second_job.payload["command"]
            self.assertEqual(
                first_command[first_command.index("--registry-generation") + 1],
                first_model["generation"],
            )
            self.assertEqual(
                second_command[second_command.index("--registry-generation") + 1],
                second_model["generation"],
            )
            self.assertEqual(first_job.payload["publish"]["metadata"]["strict_full_rows"], 3000)
            self.assertEqual(second_job.payload["publish"]["metadata"]["strict_full_rows"], 4500)

    def test_missing_verification_config_is_reported_as_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            runtime.mkdir()
            dataset = root / "train.parquet"
            model_dataset = root / "strict.parquet"
            quality = root / "quality.json"
            generation = root / "generation"
            for path in (dataset, model_dataset, quality):
                path.write_bytes(b"x")
            generation.mkdir()
            queue = DurableJobQueue(root / "pipeline" / "jobs.sqlite3")
            result = PipelineOrchestrator(
                queue,
                GenerationStore(root / "pipeline" / "artifacts"),
                runtime,
            ).plan_cycle(
                dataset_path=dataset,
                strict_full_rows=3000,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                active_model={
                    "training_run_id": "model-g1",
                    "strict_full_rows": 3000,
                    "generation": str(generation),
                    "dataset": str(model_dataset),
                    "quality_status": str(quality),
                },
                now=1200,
            )
            self.assertEqual(
                result.blocked,
                {
                    "verification_standard": "standard_command_not_configured",
                    "verification_fine": "standard_verification_blocked",
                },
            )
            self.assertNotIn("verify_standard", result.jobs)

    def test_growing_dataset_reuses_pending_tune_and_refreshes_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            (runtime / "training").mkdir(parents=True)
            dataset = root / "train.parquet"
            dataset.write_bytes(b"first")
            queue = DurableJobQueue(root / "pipeline" / "jobs.sqlite3")
            orchestrator = PipelineOrchestrator(
                queue,
                GenerationStore(root / "pipeline" / "artifacts"),
                runtime,
            )
            first = orchestrator.plan_cycle(
                dataset_path=dataset,
                strict_full_rows=4000,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                now=1200,
            )
            dataset.write_bytes(b"second")
            second = orchestrator.plan_cycle(
                dataset_path=dataset,
                strict_full_rows=4100,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                now=1800,
            )
            self.assertNotEqual(
                first.dataset_generation, second.dataset_generation
            )
            self.assertEqual(second.jobs["tune"], first.jobs["tune"])
            self.assertNotEqual(second.jobs["train"], first.jobs["train"])
            self.assertEqual(queue.get(first.jobs["tune"]).state, "queued")
            self.assertEqual(queue.get(first.jobs["train"]).state, "cancelled")
            self.assertEqual(queue.get(second.jobs["train"]).state, "queued")
            self.assertEqual(
                queue.get(second.jobs["tune"]).input_generation,
                first.dataset_generation,
            )
            self.assertEqual(
                queue.get(second.jobs["train"]).input_generation,
                second.dataset_generation,
            )
            self.assertEqual(
                [
                    dependency.id
                    for dependency in queue.dependencies(second.jobs["train"])
                ],
                [first.jobs["tune"]],
            )

    def test_new_runtime_deployment_gets_distinct_execution_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "train.parquet"
            dataset.write_bytes(b"same immutable dataset")
            queue = DurableJobQueue(root / "pipeline" / "jobs.sqlite3")
            store = GenerationStore(root / "pipeline" / "artifacts")

            runtimes = [root / "deploy-a", root / "deploy-b"]
            for index, runtime in enumerate(runtimes):
                (runtime / "training").mkdir(parents=True)
                (runtime / "campaign").mkdir()
                (runtime / "training" / "tune_optuna.py").write_text(
                    f"# deployment {index}\n", encoding="utf-8"
                )
                (
                    runtime / "training" / "checkpoint_orchestrator.py"
                ).write_text(f"# deployment {index}\n", encoding="utf-8")

            first = PipelineOrchestrator(
                queue, store, runtimes[0], python=str(root / "python")
            ).plan_cycle(
                dataset_path=dataset,
                strict_full_rows=4000,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                now=1200,
            )
            second = PipelineOrchestrator(
                queue, store, runtimes[1], python=str(root / "python")
            ).plan_cycle(
                dataset_path=dataset,
                strict_full_rows=4000,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                now=1200,
            )

            for job_type in ("collect", "tune", "train"):
                self.assertNotEqual(
                    first.jobs[job_type], second.jobs[job_type]
                )
                self.assertNotEqual(
                    queue.get(first.jobs[job_type]).idempotency_key,
                    queue.get(second.jobs[job_type]).idempotency_key,
                )
            self.assertEqual(queue.get(first.jobs["tune"]).state, "cancelled")
            self.assertEqual(queue.get(first.jobs["train"]).state, "cancelled")
            second_tune = queue.get(second.jobs["tune"])
            self.assertTrue(
                second_tune.payload["command"][1].startswith(
                    str(runtimes[1].resolve())
                )
            )

    def test_running_same_content_tune_survives_runtime_deployment_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "train.parquet"
            dataset.write_bytes(b"same immutable dataset")
            queue = DurableJobQueue(root / "pipeline" / "jobs.sqlite3")
            store = GenerationStore(root / "pipeline" / "artifacts")
            runtimes = [root / "deploy-a", root / "deploy-b"]
            for runtime in runtimes:
                (runtime / "training").mkdir(parents=True)
                (runtime / "campaign").mkdir()
                (runtime / "training" / "tune_optuna.py").write_text(
                    "# identical reviewed tuner\n", encoding="utf-8"
                )
                (
                    runtime / "training" / "checkpoint_orchestrator.py"
                ).write_text("# identical checkpoint\n", encoding="utf-8")

            first = PipelineOrchestrator(
                queue, store, runtimes[0], python=str(root / "python")
            ).plan_cycle(
                dataset_path=dataset,
                strict_full_rows=4000,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                now=1200,
            )

            # HPO 454 was created by the root-sensitive v3 planner.  Give this
            # fixture a deliberately different legacy suffix so reuse depends
            # on its immutable command/script contract, not on the new key.
            first_tune = queue.get(first.jobs["tune"])
            legacy_key = first_tune.idempotency_key.rsplit("-e", 1)[0] + (
                "-e" + "f" * 12
            )
            connection = queue._connect()
            try:
                connection.execute(
                    "UPDATE jobs SET idempotency_key=? WHERE id=?",
                    (legacy_key, first_tune.id),
                )
            finally:
                connection.close()
            running = queue.claim(
                "existing-hpo",
                job_types=["tune"],
                lease_seconds=1000,
                now=1201,
            )
            self.assertEqual(running.id, first.jobs["tune"])

            # Also reproduce the queued tune/train pair created by the
            # root-sensitive deployment before the fixed controller arrives.
            stale_tune_payload = json.loads(json.dumps(running.payload))
            stale_tune_payload.pop("execution_contract", None)
            stale_tune_payload["command"][1] = str(
                runtimes[1] / "training" / "tune_optuna.py"
            )
            stale_tune_payload["cwd"] = str(runtimes[1].resolve())
            stale_tune = queue.enqueue(
                "tune",
                "tune-stale-root-e" + "e" * 12,
                stale_tune_payload,
                input_generation=first.dataset_generation,
                coalesce_key=running.coalesce_key,
                coalesce_pending=True,
                priority=70,
                max_attempts=3,
                now=1201.25,
            )
            runtime_execution_key = hashlib.sha256(json.dumps(
                {
                    "python": os.path.normcase(str((root / "python").resolve())),
                    "runtime_root": os.path.normcase(str(runtimes[1].resolve())),
                },
                sort_keys=True,
            ).encode("utf-8")).hexdigest()[:12]
            checkpoint_execution_key = hashlib.sha256(json.dumps(
                {
                    "model_threads": 24,
                    "output_root": os.path.normcase(
                        str((runtimes[1] / "training").resolve())
                    ),
                    "runtime_execution_key": runtime_execution_key,
                    "script_sha256": hashlib.sha256(
                        (
                            runtimes[1]
                            / "training"
                            / "checkpoint_orchestrator.py"
                        ).read_bytes()
                    ).hexdigest(),
                },
                sort_keys=True,
            ).encode("utf-8")).hexdigest()[:12]
            dataset_generation = first.dataset_generation.split(":", 1)[1]
            stale_train = queue.enqueue(
                "train",
                (
                    f"checkpoint-3000-{dataset_generation}"
                    f"-e{checkpoint_execution_key}"
                ),
                {"command": ["stale-root-train"]},
                input_generation=first.dataset_generation,
                coalesce_key=running.coalesce_key,
                coalesce_pending=True,
                dependencies=[stale_tune.id],
                priority=80,
                max_attempts=3,
                now=1201.5,
            )

            second = PipelineOrchestrator(
                queue, store, runtimes[1], python=str(root / "python")
            ).plan_cycle(
                dataset_path=dataset,
                strict_full_rows=4000,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                now=1202,
            )

            reused = queue.get(second.jobs["tune"])
            self.assertEqual(reused.id, running.id)
            self.assertEqual(reused.state, "running")
            self.assertEqual(reused.owner_lease, running.owner_lease)
            self.assertEqual(reused.lease_until, running.lease_until)
            self.assertEqual(queue.get(stale_tune.id).state, "cancelled")
            self.assertEqual(queue.get(stale_train.id).state, "cancelled")
            self.assertNotEqual(first.jobs["collect"], second.jobs["collect"])
            self.assertEqual(
                [
                    dependency.id
                    for dependency in queue.dependencies(second.jobs["train"])
                ],
                [running.id],
            )
            self.assertIn(
                f"-t{running.id}-e",
                queue.get(second.jobs["train"]).idempotency_key,
            )
            active_tunes = queue.list(
                states=["running", "queued", "retry_wait"],
                job_types=["tune"],
            )
            self.assertEqual([job.id for job in active_tunes], [running.id])

    def test_running_tune_recovers_from_legacy_pending_successors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            (runtime / "training").mkdir(parents=True)
            dataset = root / "train.parquet"
            dataset.write_bytes(b"first")
            queue = DurableJobQueue(root / "pipeline" / "jobs.sqlite3")
            orchestrator = PipelineOrchestrator(
                queue,
                GenerationStore(root / "pipeline" / "artifacts"),
                runtime,
            )
            first = orchestrator.plan_cycle(
                dataset_path=dataset,
                strict_full_rows=4000,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                now=1200,
            )
            running_tune = queue.claim(
                "tuner",
                job_types=["tune"],
                lease_seconds=1000,
                now=1201,
            )
            self.assertEqual(running_tune.id, first.jobs["tune"])

            # Reproduce the pre-fix shape from the live queue: the active tune
            # is left running while a newer pending tune/train pair supersedes
            # its original dependent checkpoint.
            cohort_key = running_tune.coalesce_key
            legacy_tune = queue.enqueue(
                "tune",
                "legacy-newer-tune",
                {"command": ["legacy-tune"]},
                input_generation="dataset:legacy",
                coalesce_key=cohort_key,
                coalesce_pending=True,
                priority=70,
                max_attempts=3,
                now=1250,
            )
            first_train_key = queue.get(
                first.jobs["train"]
            ).idempotency_key
            checkpoint_prefix = "-".join(first_train_key.split("-", 2)[:2])
            execution_suffix = first_train_key[first_train_key.rfind("-e"):]
            legacy_train = queue.enqueue(
                "train",
                f"{checkpoint_prefix}-legacy{execution_suffix}",
                {"command": ["legacy-train"]},
                input_generation="dataset:legacy",
                coalesce_key=cohort_key,
                coalesce_pending=True,
                dependencies=[legacy_tune.id],
                priority=80,
                max_attempts=3,
                now=1250,
            )
            self.assertEqual(queue.get(first.jobs["train"]).state, "cancelled")

            dataset.write_bytes(b"second")
            recovered = orchestrator.plan_cycle(
                dataset_path=dataset,
                strict_full_rows=4100,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                now=1300,
            )
            self.assertEqual(recovered.jobs["tune"], running_tune.id)
            self.assertEqual(queue.get(legacy_tune.id).state, "cancelled")
            self.assertEqual(
                queue.get(legacy_tune.id).terminal_reason,
                f"superseded_by:{running_tune.id}",
            )
            self.assertEqual(queue.get(legacy_train.id).state, "cancelled")
            self.assertEqual(
                [
                    job.id
                    for job in queue.dependencies(recovered.jobs["train"])
                ],
                [running_tune.id],
            )

            dataset.write_bytes(b"third")
            stable = orchestrator.plan_cycle(
                dataset_path=dataset,
                strict_full_rows=4200,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                now=1400,
            )
            self.assertEqual(stable.jobs["tune"], recovered.jobs["tune"])
            self.assertNotEqual(stable.jobs["train"], recovered.jobs["train"])
            self.assertEqual(
                queue.get(recovered.jobs["train"]).state, "cancelled"
            )
            latest_train = queue.get(stable.jobs["train"])
            self.assertEqual(
                latest_train.input_generation, stable.dataset_generation
            )
            self.assertEqual(
                [job.id for job in queue.dependencies(latest_train.id)],
                [running_tune.id],
            )

    def test_completed_tuning_wave_allows_the_next_growth_wave(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            (runtime / "training").mkdir(parents=True)
            dataset = root / "train.parquet"
            dataset.write_bytes(b"four thousand")
            queue = DurableJobQueue(root / "pipeline" / "jobs.sqlite3")
            store = GenerationStore(root / "pipeline" / "artifacts")
            orchestrator = PipelineOrchestrator(queue, store, runtime)
            first = orchestrator.plan_cycle(
                dataset_path=dataset,
                strict_full_rows=4000,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                now=1200,
            )
            claimed = queue.claim(
                "tuner", job_types=["tune"], lease_seconds=1000, now=1201
            )
            command = claimed.payload["command"]
            contract_sha256 = command[
                command.index("--data-contract-sha256") + 1
            ]
            params = root / "params.json"
            params.write_text("{}", encoding="utf-8")
            tuning = store.publish_files(
                "tuning",
                {"params.json": params},
                metadata={
                    "strict_full_rows": 4000,
                    "solver_revision": "a" * 40,
                    "library_revision": "b" * 40,
                    "data_contract_sha256": contract_sha256,
                },
            )
            queue.succeed(
                claimed.id,
                "tuner",
                output_generation=f"tuning:{tuning.generation_id}",
                now=1202,
            )

            # The child retains its original dependency edge after that tune
            # succeeds.  Replanning identical bytes must reuse it instead of
            # colliding with the same idempotency key and a rewritten payload.
            same_snapshot = orchestrator.plan_cycle(
                dataset_path=dataset,
                strict_full_rows=4000,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                now=1300,
            )
            self.assertNotIn("tune", same_snapshot.jobs)
            self.assertEqual(same_snapshot.jobs["train"], first.jobs["train"])

            dataset.write_bytes(b"five thousand")
            below_growth_gate = orchestrator.plan_cycle(
                dataset_path=dataset,
                strict_full_rows=5000,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                now=1800,
            )
            self.assertNotIn("tune", below_growth_gate.jobs)
            self.assertNotEqual(
                below_growth_gate.jobs["train"], first.jobs["train"]
            )
            self.assertEqual(queue.get(first.jobs["train"]).state, "cancelled")
            self.assertEqual(
                queue.get(below_growth_gate.jobs["train"]).input_generation,
                below_growth_gate.dataset_generation,
            )
            ready_train = queue.claim(
                "trainer",
                job_types=["train"],
                lease_seconds=1000,
                now=1801,
            )
            self.assertEqual(
                ready_train.id, below_growth_gate.jobs["train"]
            )

            dataset.write_bytes(b"six thousand")
            next_wave = orchestrator.plan_cycle(
                dataset_path=dataset,
                strict_full_rows=6000,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                now=2400,
            )
            self.assertIn("tune", next_wave.jobs)
            self.assertNotEqual(next_wave.jobs["tune"], first.jobs["tune"])
            self.assertEqual(next_wave.jobs["train"], ready_train.id)
            self.assertEqual(
                queue.get(next_wave.jobs["tune"]).input_generation,
                next_wave.dataset_generation,
            )

    def test_durable_checkpoint_state_advances_without_dataset_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            (runtime / "training").mkdir(parents=True)
            (runtime / "campaign").mkdir()
            dataset = root / "train.parquet"
            dataset.write_bytes(b"stable dataset generation")
            queue = DurableJobQueue(root / "pipeline" / "jobs.sqlite3")
            orchestrator = PipelineOrchestrator(
                queue,
                GenerationStore(root / "pipeline" / "artifacts"),
                runtime,
                python=str(root / "python"),
            )

            training_dir = str(Path(__file__).resolve().parents[2] / "training")
            import sys

            if training_dir not in sys.path:
                sys.path.insert(0, training_dir)
            from checkpoint_contract import checkpoint_contract_identity

            contract = checkpoint_contract_identity(solver_revision="a" * 40)
            run_root = (
                runtime
                / "training"
                / "checkpoint_runs"
                / ("b" * 40 + "-c" + contract["checkpoint_contract_key"])
            )
            run_root.mkdir(parents=True)
            (run_root / "checkpoint_state.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "identity": {
                            **contract,
                            "library_revision": "b" * 40,
                        },
                        "completed": [
                            {"threshold": 500, "kind": "metrics_only"}
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = orchestrator.plan_cycle(
                dataset_path=dataset,
                strict_full_rows=2188,
                solver_revision="a" * 40,
                library_revision="b" * 40,
                now=1200,
            )

            train = queue.get(result.jobs["train"])
            self.assertIn("checkpoint-1000-", train.idempotency_key)

    def test_tuning_lookup_is_scoped_to_the_full_data_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            params = root / "params.json"
            params.write_text("{}", encoding="utf-8")
            store = GenerationStore(root / "artifacts")
            store.publish_files(
                "tuning", {"params.json": params},
                metadata={
                    "strict_full_rows": 9000,
                    "solver_revision": "c" * 40,
                    "library_revision": "b" * 40,
                    "data_contract_sha256": "d" * 64,
                },
            )
            accepted = store.publish_files(
                "tuning", {"params.json": params},
                metadata={
                    "strict_full_rows": 4000,
                    "solver_revision": "a" * 40,
                    "library_revision": "b" * 40,
                    "data_contract_sha256": "d" * 64,
                },
            )
            orchestrator = PipelineOrchestrator(
                DurableJobQueue(root / "jobs.sqlite3"), store, root
            )
            latest = orchestrator.latest_tuning(
                "a" * 40, "b" * 40, "d" * 64
            )
            self.assertEqual(latest[0], 4000)
            self.assertEqual(Path(latest[1]), accepted.path)

    def test_active_descriptor_rejects_a_different_revision_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "strict.parquet"
            dataset.write_bytes(b"immutable")
            generation = root / "generation"
            generation.mkdir()
            fake = types.ModuleType("train_models")
            fake.load_active_generation = lambda _registry: {
                "generation": str(generation),
                "report": {
                    "training_run_id": "model-g1",
                    "strict_full_rows": 4000,
                    "dataset_path": str(dataset),
                },
                "quality": {
                    "solver_revision": "c" * 40,
                    "library_revision": "b" * 40,
                },
            }

            with (
                mock.patch.dict(sys.modules, {"train_models": fake}),
                self.assertRaisesRegex(RuntimeError, "solver revision"),
            ):
                descriptor_from_active_registry(
                    root / "registry",
                    solver_revision="a" * 40,
                    library_revision="b" * 40,
                )

            fake.load_active_generation = lambda _registry: {
                "generation": str(generation),
                "report": {
                    "training_run_id": "model-g1",
                    "strict_full_rows": 4000,
                    "dataset_path": str(dataset),
                },
                "quality": {
                    "solver_revision": "a" * 40,
                    "library_revision": "b" * 40,
                },
            }
            with mock.patch.dict(sys.modules, {"train_models": fake}):
                descriptor = descriptor_from_active_registry(
                    root / "registry",
                    solver_revision="a" * 40,
                    library_revision="b" * 40,
                )
            self.assertEqual(descriptor["training_run_id"], "model-g1")


class ControllerSnapshotTests(unittest.TestCase):
    def test_count_and_plan_consume_the_same_immutable_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            live = root / "live.parquet"
            live.write_bytes(b"first-generation")

            class FakeOrchestrator:
                runtime_root = root
                store = GenerationStore(root / "artifacts")

                def plan_cycle(self, **kwargs):
                    self.planned_bytes = Path(kwargs["dataset_path"]).read_bytes()
                    self.series = kwargs["dataset_series_path"]
                    self.model_threads = kwargs["model_threads"]
                    return "planned"

            orchestrator = FakeOrchestrator()
            controller = ContinuousController(
                orchestrator,
                dataset=live,
                registry=root / "registry",
                solver_revision="a" * 40,
                library_revision="b" * 40,
            )

            def inspect(snapshot):
                self.assertEqual(Path(snapshot).read_bytes(), b"first-generation")
                live.write_bytes(b"second-generation")
                controller._last_raw_rows = 5432
                return 4000

            controller.inspect_strict_rows = inspect
            self.assertEqual(controller.plan_once(), "planned")
            self.assertEqual(orchestrator.planned_bytes, b"first-generation")
            self.assertEqual(orchestrator.series, str(live.resolve()))
            self.assertEqual(orchestrator.model_threads, 24)
            status = json.loads(
                (root / "surrogate_status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["state"], "waiting_for_next_dataset_check")
            self.assertEqual(status["raw_rows"], 5432)
            self.assertEqual(status["strict_full_rows"], 4000)
            self.assertEqual(status["last_error"], None)
            manifest = json.loads(
                (root / "active_surrogate.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["state"], "awaiting_activation")
            self.assertEqual(manifest["rows_until_activation"], 0)
            self.assertIn("fail closed", manifest["consumer_contract"])


if __name__ == "__main__":
    unittest.main()
