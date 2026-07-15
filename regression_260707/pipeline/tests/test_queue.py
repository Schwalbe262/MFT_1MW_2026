from pathlib import Path
import tempfile
import unittest

from regression_260707.pipeline.queue import DurableJobQueue


class DurableJobQueueTests(unittest.TestCase):
    def test_idempotency_key_replays_only_identical_input(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = DurableJobQueue(Path(directory) / "jobs.sqlite3")
            first = queue.enqueue(
                "train", "dataset-a", {"command": ["python", "train.py"]},
                input_generation="dataset:a", priority=5, now=1,
            )
            replay = queue.enqueue(
                "train", "dataset-a", {"command": ["python", "train.py"]},
                input_generation="dataset:a", priority=5, now=2,
            )
            self.assertEqual(first.id, replay.id)
            with self.assertRaisesRegex(RuntimeError, "different job inputs"):
                queue.enqueue(
                    "train", "dataset-a", {"command": ["python", "other.py"]},
                    input_generation="dataset:a", priority=5, now=3,
                )

    def test_dependency_dag_and_terminal_propagation(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = DurableJobQueue(Path(directory) / "jobs.sqlite3")
            parent = queue.enqueue("tune", "a", {"command": ["tune"]}, now=1)
            child = queue.enqueue(
                "train", "b", {"command": ["train"]},
                dependencies=[parent.id], now=1,
            )
            claimed = queue.claim("tuner", now=2)
            self.assertEqual(claimed.id, parent.id)
            self.assertIsNone(queue.claim("trainer", job_types=["train"], now=2))
            queue.fail(parent.id, "tuner", "bad input", retry=False, now=3)
            queue.reconcile(now=4)
            self.assertEqual(queue.get(child.id).state, "failed")
            self.assertEqual(
                queue.get(child.id).terminal_reason,
                f"dependency_failed:{parent.id}",
            )

    def test_expired_lease_retries_then_becomes_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = DurableJobQueue(Path(directory) / "jobs.sqlite3")
            job = queue.enqueue(
                "optimize", "model-a", {"command": ["opt"]},
                max_attempts=2, now=0,
            )
            first = queue.claim("worker-1", lease_seconds=10, now=1)
            self.assertEqual(first.attempt, 1)
            queue.reconcile(now=12)
            self.assertEqual(queue.get(job.id).state, "queued")
            second = queue.claim("worker-2", lease_seconds=10, now=13)
            self.assertEqual(second.attempt, 2)
            queue.reconcile(now=24)
            terminal = queue.get(job.id)
            self.assertEqual(terminal.state, "failed")
            self.assertEqual(terminal.terminal_reason, "worker_lease_expired")

    def test_only_owner_can_heartbeat_or_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = DurableJobQueue(Path(directory) / "jobs.sqlite3")
            job = queue.enqueue("collect", "window", {"command": ["collect"]}, now=1)
            queue.claim("owner", lease_seconds=20, now=2)
            self.assertFalse(queue.heartbeat(job.id, "other", now=3))
            self.assertTrue(queue.heartbeat(job.id, "owner", lease_seconds=20, now=3))
            with self.assertRaisesRegex(RuntimeError, "no longer owns"):
                queue.succeed(job.id, "other", now=4)
            complete = queue.succeed(
                job.id, "owner", output_generation="generation:x", now=4
            )
            self.assertEqual(complete.state, "succeeded")
            self.assertEqual(complete.output_generation, "generation:x")

    def test_expired_owner_cannot_fail_or_finish_before_reconcile(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = DurableJobQueue(Path(directory) / "jobs.sqlite3")
            first = queue.enqueue("collect", "first", {"command": ["collect"]}, now=0)
            queue.claim("owner", lease_seconds=10, now=1)
            with self.assertRaisesRegex(RuntimeError, "no longer owns"):
                queue.succeed(first.id, "owner", now=12)

            queue.reconcile(now=12)
            queue.claim("owner-2", lease_seconds=10, now=13)
            with self.assertRaisesRegex(RuntimeError, "no longer owns"):
                queue.fail(first.id, "owner-2", "late", now=24)

    def test_pending_jobs_coalesce_but_running_job_is_never_cancelled(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = DurableJobQueue(Path(directory) / "jobs.sqlite3")
            first = queue.enqueue(
                "tune", "dataset-1", {"command": ["tune", "1"]},
                coalesce_key="cohort", coalesce_pending=True, now=1,
            )
            queue.claim("worker", job_types=["tune"], lease_seconds=100, now=2)
            second = queue.enqueue(
                "tune", "dataset-2", {"command": ["tune", "2"]},
                coalesce_key="cohort", coalesce_pending=True, now=3,
            )
            third = queue.enqueue(
                "tune", "dataset-3", {"command": ["tune", "3"]},
                coalesce_key="cohort", coalesce_pending=True, now=4,
            )

            self.assertEqual(queue.get(first.id).state, "running")
            self.assertEqual(queue.get(second.id).state, "cancelled")
            self.assertEqual(
                queue.get(second.id).terminal_reason, f"superseded_by:{third.id}"
            )
            self.assertEqual(queue.get(third.id).state, "queued")

            cancelled = queue.cancel_coalesced_pending(
                "tune", "cohort", reason="cohort_no_longer_due", now=5
            )
            self.assertEqual(cancelled, 1)
            self.assertEqual(queue.get(first.id).state, "running")
            self.assertEqual(queue.get(third.id).state, "cancelled")
            self.assertEqual(
                queue.get(third.id).terminal_reason, "cohort_no_longer_due"
            )


if __name__ == "__main__":
    unittest.main()
