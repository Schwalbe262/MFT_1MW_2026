from pathlib import Path
import json
import tempfile
import unittest

from regression_260707.pipeline.artifacts import GenerationStore
from regression_260707.pipeline.controller import ContinuousController
from regression_260707.pipeline.orchestrator import PipelineOrchestrator
from regression_260707.pipeline.queue import DurableJobQueue


class OrchestratorTests(unittest.TestCase):
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
            train_dependencies = queue.dependencies(result.jobs["train"])
            self.assertEqual([job.id for job in train_dependencies], [result.jobs["tune"]])
            self.assertIsNotNone(queue.claim("collector", job_types=["collect"], now=1201))
            self.assertIsNotNone(queue.claim("tuner", job_types=["tune"], now=1201))
            self.assertIsNone(queue.claim("trainer", job_types=["train"], now=1201))

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

    def test_newer_dataset_coalesces_pending_tune_and_checkpoint(self):
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
            self.assertEqual(queue.get(first.jobs["tune"]).state, "cancelled")
            self.assertEqual(queue.get(first.jobs["train"]).state, "cancelled")
            self.assertEqual(queue.get(second.jobs["tune"]).state, "queued")
            self.assertEqual(queue.get(second.jobs["train"]).state, "queued")

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
                return 4000

            controller.inspect_strict_rows = inspect
            self.assertEqual(controller.plan_once(), "planned")
            self.assertEqual(orchestrator.planned_bytes, b"first-generation")
            self.assertEqual(orchestrator.series, str(live.resolve()))


if __name__ == "__main__":
    unittest.main()
