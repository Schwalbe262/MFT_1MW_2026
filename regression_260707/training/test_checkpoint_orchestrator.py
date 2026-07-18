import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from regression_260707.training import checkpoint_orchestrator as checkpoint


class AtomicStrictStatusTests(unittest.TestCase):
    def test_permission_denied_replace_keeps_generation_and_repairs_canonical(self):
        payload = {
            "time": "2026-07-12T22:00:00+09:00",
            "strict_full_rows": 456,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "strict_data_status.json"
            with mock.patch.object(
                    checkpoint.os, "replace", side_effect=PermissionError("denied")):
                checkpoint._atomic_json(payload, path)

            generations = list(path.parent.glob(".strict-status-*.json"))
            canonical = json.loads(path.read_text(encoding="utf-8"))
            immutable = json.loads(generations[0].read_text(encoding="utf-8"))

        self.assertEqual(len(generations), 1)
        self.assertEqual(canonical, payload)
        self.assertEqual(immutable, payload)

    def test_checkpoint_state_has_verified_direct_repair_and_generation(self):
        payload = {
            "schema_version": 2,
            "completed": [{"threshold": 500}],
            "attempts": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint_state.json"
            with mock.patch.object(
                    checkpoint.os, "replace", side_effect=PermissionError("denied")):
                checkpoint._atomic_json(payload, path)

            generations = list(path.parent.glob(".checkpoint-state-*.json"))
            canonical = json.loads(path.read_text(encoding="utf-8"))
            immutable = json.loads(generations[0].read_text(encoding="utf-8"))

        self.assertEqual(len(generations), 1)
        self.assertEqual(canonical, payload)
        self.assertEqual(immutable, payload)

    def test_checkpoint_state_repair_failure_is_not_silently_accepted(self):
        payload = {"schema_version": 2, "completed": [], "attempts": []}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint_state.json"
            real_open = open

            def deny_canonical(target, *args, **kwargs):
                if os.path.abspath(target) == os.path.abspath(path):
                    raise PermissionError("canonical denied")
                return real_open(target, *args, **kwargs)

            with mock.patch.object(
                    checkpoint.os, "replace", side_effect=PermissionError("denied")
            ), mock.patch("builtins.open", side_effect=deny_canonical):
                with self.assertRaisesRegex(
                        RuntimeError, "recovery_generation="):
                    checkpoint._atomic_json(payload, path)

            generations = list(path.parent.glob(".checkpoint-state-*.json"))

        self.assertEqual(len(generations), 1)


class TrainingCommandTests(unittest.TestCase):
    def test_cli_rejects_parallelism_above_declared_thread_budget(self):
        argv = [
            "checkpoint_orchestrator.py",
            "--model-threads", "3",
            "--target-workers", "4",
            "--max-model-thread-budget", "8",
        ]
        with mock.patch.object(checkpoint.sys, "argv", argv):
            with self.assertRaises(SystemExit):
                checkpoint.main()

    def test_candidate_command_forwards_model_thread_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commands = checkpoint.training_commands(
                str(root / "snapshot.parquet"),
                str(root / "learning_curve.csv"),
                str(root / "registry"),
                100,
                str(root / "profile.json"),
                3000,
                str(root / "metrics.json"),
                str(root / "candidate.json"),
                model_threads=8,
                target_workers=4,
                max_model_thread_budget=32,
            )

        candidate = commands[1]
        self.assertEqual(
            candidate[candidate.index("--model-threads") + 1], "8"
        )
        self.assertEqual(
            candidate[candidate.index("--target-workers") + 1], "4"
        )
        self.assertEqual(
            candidate[candidate.index("--max-model-thread-budget") + 1], "32"
        )

    def test_checkpoint_command_writes_non_authoritative_parity_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = root / "threshold_001000_attempt_000002.json"
            commands = checkpoint.training_commands(
                str(root / "snapshot.parquet"),
                str(root / "learning_curve.csv"),
                str(root / "registry"),
                100,
                str(root / "profile.json"),
                1000,
                str(metrics),
            )

        self.assertEqual(len(commands), 1)
        command = commands[0]
        self.assertEqual(
            command[command.index("--result-json") + 1], str(metrics)
        )
        self.assertEqual(
            command[command.index("--parity-json") + 1],
            str(metrics.with_suffix(".parity.json")),
        )
        self.assertNotIn("--skip-curve-append", command)


class CheckpointIdentityRelocationTests(unittest.TestCase):
    @staticmethod
    def _identity(root, *, profile_sha="profile-sha", thresholds_sha="thresholds-sha"):
        return {
            "dataset": "Y:/canonical/train.parquet",
            "profile_path": str(Path(root) / "profiles" / "standard.json"),
            "profile_sha256": profile_sha,
            "thresholds_path": str(Path(root) / "training" / "thresholds.json"),
            "thresholds_sha256": thresholds_sha,
            "activation_minimum_strict_full_rows": 3000,
            "quality_contract_sha256": "quality-sha",
            "model_targets_sha256": "targets-sha",
            "checkpoint_contract_schema_version": 2,
            "checkpoint_contract_sha256": "contract-sha",
            "checkpoint_contract_key": "contract-key",
            "registry_protocol_version": 2,
            "physics_data_revision": "physics-v1",
            "solver_revision_cohort": "solver-cohort",
            "library_revision": "library-revision",
        }

    def test_content_identical_immutable_path_move_preserves_completions(self):
        old = self._identity("C:/deployments/old")
        new = self._identity("C:/deployments/new")
        completion = {"threshold": 3000, "training_run_id": "accepted-run"}
        state = {"identity": old, "completed": [completion]}

        changed = checkpoint._ensure_identity(state, new)

        self.assertTrue(changed)
        self.assertEqual(state["identity"], new)
        self.assertEqual(state["completed"], [completion])
        self.assertEqual(
            state["recovery"][-1]["reason"],
            "checkpoint_identity_paths_relocated",
        )
        self.assertEqual(
            state["recovery"][-1]["fields"],
            ["profile_path", "thresholds_path"],
        )

    def test_path_move_with_changed_content_still_fails_closed(self):
        old = self._identity("C:/deployments/old")
        new = self._identity("C:/deployments/new", profile_sha="changed-profile")
        state = {"identity": old, "completed": [{"threshold": 3000}]}

        with self.assertRaisesRegex(
                RuntimeError, "checkpoint runtime identity changed"):
            checkpoint._ensure_identity(state, new)

        self.assertEqual(state["identity"], old)
        self.assertEqual(state["completed"], [{"threshold": 3000}])

    def test_metrics_completion_accepts_same_hash_profile_relocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot.parquet"
            snapshot.write_bytes(b"immutable snapshot")
            snapshot_sha = checkpoint._sha256(snapshot)
            old_profile = root / "old" / "profiles" / "standard.json"
            new_profile = root / "new" / "profiles" / "standard.json"
            metrics = root / "metrics.json"
            metrics.write_text(json.dumps({
                "checkpoint": 500,
                "dataset": str(snapshot.resolve()),
                "dataset_sha256": snapshot_sha,
                "profile": str(old_profile.resolve()),
                "profile_sha256": "profile-sha",
                "strict_full_rows": 500,
                "metrics": [{"target": "Llt_phys"}],
            }), encoding="utf-8")
            item = {
                "kind": "metrics_only",
                "threshold": 500,
                "actual_strict_full_rows": 500,
                "snapshot": str(snapshot),
                "snapshot_sha256": snapshot_sha,
                "metrics_result": str(metrics),
                "metrics_result_sha256": checkpoint._sha256(metrics),
                "profile_path": str(old_profile),
                "profile_sha256": "profile-sha",
                "thresholds_sha256": "thresholds-sha",
                "activation_minimum_strict_full_rows": 3000,
            }
            expected = {
                "profile_path": str(new_profile),
                "profile_sha256": "profile-sha",
                "thresholds_sha256": "thresholds-sha",
                "activation_minimum_strict_full_rows": 3000,
            }

            error = checkpoint._completion_error(
                item, str(root / "registry"), expected
            )

        self.assertIsNone(error)


if __name__ == "__main__":
    unittest.main()
