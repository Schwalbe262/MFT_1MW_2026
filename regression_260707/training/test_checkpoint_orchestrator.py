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

            generations = list(path.parent.glob(path.name + ".gen-*.json"))
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

            generations = list(path.parent.glob(path.name + ".gen-*.json"))
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

            generations = list(path.parent.glob(path.name + ".gen-*.json"))

        self.assertEqual(len(generations), 1)


class TrainingCommandTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
