import copy
import hashlib
import json
import shlex
import sys
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from unittest.mock import Mock, patch


CAMPAIGN_DIR = Path(__file__).resolve().parent
if str(CAMPAIGN_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_DIR))

import feeder


SOLVER_REVISION = "a" * 40
LIBRARY_REVISION = "b" * 40
# Compact-JSON digest from the pre-pooled payload builder for this fixed fixture.
LEGACY_PAYLOAD_SHA256 = (
    "611abcc5adac9e6f4870b9e2dae7453cbb01f69018b4fa70fefc56ffeefab973"
)


class FeederPooledSubmissionTests(unittest.TestCase):
    def _capture_cli_payload(
            self, cli_args, *, fail_local_revision_checks=False):
        accepted = Mock(status_code=201)
        accepted.json.return_value = {"id": 123}
        argv = [
            "feeder.py",
            "--once",
            "--target", "1",
            "--max-samples", "1",
            "--solver-revision", SOLVER_REVISION,
            "--library-revision", LIBRARY_REVISION,
            *cli_args,
        ]
        campaign_counts = {"queued": 0, "attaching": 0, "running": 0}
        capacity = {
            "ready_fit_slots": 1,
            "project_submission_slots": 1,
            "submission_allowed": True,
            "queue_state": "ready",
            "queue_reason": "",
            "project_active": 0,
        }
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", argv))
            solver_revision_check = stack.enter_context(patch.object(
                feeder.al_driver,
                "_current_solver_revision",
                return_value=SOLVER_REVISION,
            ))
            library_revision_check = stack.enter_context(patch.object(
                feeder.al_driver,
                "_current_library_revision",
                return_value=LIBRARY_REVISION,
            ))
            if fail_local_revision_checks:
                error = AssertionError(
                    "local revision vetting must be bypassed")
                solver_revision_check.side_effect = error
                library_revision_check.side_effect = error
            stack.enter_context(patch.object(feeder, "validate_p08_completion"))
            stack.enter_context(patch.object(feeder, "_require_deployed_revisions"))
            stack.enter_context(patch.object(
                feeder,
                "campaign_mutation_lock",
                side_effect=lambda: nullcontext(),
            ))
            stack.enter_context(patch.object(
                feeder.scheduler_client,
                "campaign_mutation_lock_is_held",
                return_value=True,
            ))
            stack.enter_context(patch.object(
                feeder,
                "load_state",
                return_value={"serial": 0, "submitted_samples": 0},
            ))
            stack.enter_context(patch.object(
                feeder,
                "scheduler_snapshot",
                return_value=(campaign_counts, campaign_counts, [], capacity),
            ))
            stack.enter_context(patch.object(
                feeder, "dataset_collection_snapshot", return_value=(0, set())))
            stack.enter_context(patch.object(
                feeder, "campaign_inventory", return_value=[]))
            stack.enter_context(patch.object(
                feeder, "reserved_unjudged_rows", return_value=0))
            stack.enter_context(patch.object(
                feeder, "cursor_after_valid_candidates", return_value=0))
            stack.enter_context(patch.object(
                feeder,
                "next_valid_candidate",
                return_value=(1, 0, {"candidate": 1}),
            ))
            stack.enter_context(patch.object(feeder, "save_state"))
            stack.enter_context(patch.object(feeder.time, "sleep"))
            stack.enter_context(patch.object(
                feeder.scheduler_client,
                "reconcile_task_id",
                return_value=None,
            ))
            stack.enter_context(patch.object(
                feeder.scheduler_client,
                "live_project_submission_snapshot",
                return_value={"project_submission_slots": 300},
            ))
            post = stack.enter_context(patch.object(
                feeder.scheduler_client.requests,
                "post",
                return_value=accepted,
            ))
            feeder.main()

        if fail_local_revision_checks:
            solver_revision_check.assert_not_called()
            library_revision_check.assert_not_called()
        post.assert_called_once()
        return copy.deepcopy(post.call_args.kwargs["json"])

    def test_trust_pinned_revisions_bypasses_local_revision_vetting(self):
        with patch("builtins.print") as output:
            self._capture_cli_payload(
                ["--trust-pinned-revisions"],
                fail_local_revision_checks=True,
            )

        warning_lines = [
            call.args[0]
            for call in output.call_args_list
            if call.args and isinstance(call.args[0], str)
            and "WARNING" in call.args[0]
        ]
        self.assertEqual(warning_lines, [
            "[feeder] WARNING: local revision vetting was bypassed; "
            f"using pinned solver SHA {SOLVER_REVISION} and "
            f"library SHA {LIBRARY_REVISION}"
        ])

    def test_aedt_pooled_injects_backend_environment_and_resources(self):
        expected_env = {
            "MFT_AEDT_BACKEND": "pooled",
            "MFT_AEDT_SHARED_CANARY": "1",
            "MFT_AEDT_SCHEDULER_URL": "https://pool.example.test:8443",
            "MFT_SLURM_SCHEDULER_ROOT": "/opt/pool package",
            "SLURM_AEDT_POOL_BOOTSTRAP_TOKEN_FILE": "/run/pool token",
        }

        defaults = feeder._argument_parser().parse_args([
            "--aedt-pooled",
            "--aedt-pool-url", "https://pool.example.test:8443",
        ])
        self.assertEqual(
            defaults.aedt_pool_pkg_root, feeder.DEFAULT_AEDT_POOL_PKG_ROOT)
        self.assertEqual(
            defaults.aedt_pool_token_file, feeder.DEFAULT_AEDT_POOL_TOKEN_FILE)
        self.assertEqual(defaults.pooled_cpus, 1)
        self.assertEqual(defaults.pooled_memory_mb, 6144)

        pooled_payload = self._capture_cli_payload([
            "--aedt-pooled",
            "--aedt-pool-url", "https://pool.example.test:8443",
            "--aedt-pool-pkg-root", "/opt/pool package",
            "--aedt-pool-token-file", "/run/pool token",
            "--pooled-cpus", "2",
            "--pooled-memory-mb", "8192",
        ])
        self.assertEqual(pooled_payload["aedt_backend"], "pooled")
        self.assertEqual(pooled_payload["cpus"], 2)
        self.assertEqual(pooled_payload["memory_mb"], 8192)
        self.assertNotIn("submission_env", pooled_payload)

        env_exports = "".join(
            f"export {key}={shlex.quote(value)}; "
            for key, value in sorted(expected_env.items())
        )
        self.assertEqual(pooled_payload["command"].count(env_exports), 1)
        for key, value in expected_env.items():
            self.assertIn(
                f"export {key}={shlex.quote(value)};",
                pooled_payload["command"],
            )

        legacy_payload = self._capture_cli_payload([])
        normalized = copy.deepcopy(pooled_payload)
        normalized.pop("aedt_backend")
        normalized["cpus"] = legacy_payload["cpus"]
        normalized["memory_mb"] = legacy_payload["memory_mb"]
        normalized["command"] = normalized["command"].replace(env_exports, "", 1)
        self.assertEqual(normalized, legacy_payload)

    def test_without_aedt_pooled_keeps_exact_legacy_payload_shape(self):
        payload = self._capture_cli_payload([
            "--aedt-pool-url", "https://ignored.example.test",
            "--aedt-pool-pkg-root", "/ignored/pkg",
            "--aedt-pool-token-file", "/ignored/token",
            "--pooled-cpus", "2",
            "--pooled-memory-mb", "8192",
        ])
        self.assertEqual(set(payload), {
            "name", "project", "remote_cwd", "command",
            "required_capability", "env_profile", "scheduling_profile",
            "cpus", "memory_mb", "gpus", "account_name", "node_name",
            "max_workers_per_node", "priority", "timeout_seconds",
            "dedupe_key", "cleanup_globs",
        })
        self.assertEqual(payload["cpus"], 4)
        self.assertEqual(payload["memory_mb"], 32768)
        self.assertNotIn("aedt_backend", payload)
        self.assertNotIn("submission_env", payload)
        legacy_bytes = json.dumps(
            payload, ensure_ascii=True, separators=(",", ":")
        ).encode("utf-8")
        self.assertEqual(
            hashlib.sha256(legacy_bytes).hexdigest(), LEGACY_PAYLOAD_SHA256)
        for key in (
                "MFT_AEDT_BACKEND",
                "MFT_AEDT_SHARED_CANARY",
                "MFT_AEDT_SCHEDULER_URL",
                "MFT_SLURM_SCHEDULER_ROOT",
                "SLURM_AEDT_POOL_BOOTSTRAP_TOKEN_FILE"):
            self.assertNotIn(key, payload["command"])


if __name__ == "__main__":
    unittest.main()
