import copy
import hashlib
import importlib
import json
import logging
import os
import sys
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from unittest.mock import Mock, call, patch

from filelock import FileLock


CAMPAIGN_DIR = Path(__file__).resolve().parent
if str(CAMPAIGN_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_DIR))

import feeder


SOLVER_REVISION = "a" * 40
LIBRARY_REVISION = "b" * 40
# Compact-JSON digest from the pre-pooled payload builder for this fixed fixture.
LEGACY_PAYLOAD_SHA256 = (
    "c9729f8f224a26bbf0161381c06ffe54709d678f48163bb032cbc19ec9d4dd1e"
)


class FeederSchedulerEndpointTests(unittest.TestCase):
    def test_next_feeder_process_uses_scheduler_environment(self):
        with patch.dict(
                os.environ, {"MFT_SCHEDULER_URL": "http://127.0.0.1:8001/"}):
            self.assertEqual(
                feeder._configured_scheduler_url(),
                "http://127.0.0.1:8001",
            )

    def test_local_transport_failure_syncs_reads_and_submissions_to_8001(self):
        recovered = Mock(status_code=200)
        recovered.json.return_value = {"status": "ok"}
        with patch.object(
                feeder, "SCHEDULER", feeder.DEFAULT_SCHEDULER), patch.object(
                feeder.scheduler_client,
                "SCHEDULER",
                feeder.DEFAULT_SCHEDULER,
        ), patch.object(
                feeder.requests,
                "get",
                side_effect=[
                    feeder.requests.ConnectionError(
                        "legacy listener unavailable"
                    ),
                    recovered,
                ],
        ) as get_mock, patch.object(feeder.time, "sleep") as sleep_mock:
            self.assertEqual(
                feeder._scheduler_json("/api/health"),
                {"status": "ok"},
            )
            self.assertEqual(feeder.SCHEDULER, feeder.LOCAL_SCHEDULER_FALLBACK)
            self.assertEqual(
                feeder.scheduler_client.SCHEDULER,
                feeder.LOCAL_SCHEDULER_FALLBACK,
            )

        self.assertEqual([call.args[0] for call in get_mock.call_args_list], [
            "http://127.0.0.1:8000/api/health",
            "http://127.0.0.1:8001/api/health",
        ])
        sleep_mock.assert_not_called()

    def test_explicit_remote_scheduler_never_falls_back_to_loopback(self):
        with patch.object(
                feeder, "SCHEDULER", "https://scheduler.example.test"), \
                patch.object(
                    feeder.scheduler_client,
                    "SCHEDULER",
                    "https://scheduler.example.test",
                ), patch.object(
                    feeder, "SCHEDULER_ATTEMPTS", 2
                ), patch.object(
                    feeder.requests,
                    "get",
                    side_effect=feeder.requests.ConnectionError("unavailable"),
                ) as get_mock, patch.object(feeder.time, "sleep"):
            with self.assertRaises(feeder.SchedulerError):
                feeder._scheduler_json("/api/health")

        self.assertEqual(get_mock.call_count, 2)
        self.assertTrue(all(
            call.args[0] == "https://scheduler.example.test/api/health"
            for call in get_mock.call_args_list
        ))


class FeederPooledSubmissionTests(unittest.TestCase):
    def _capture_cli_payload(
            self, cli_args, *, fail_local_revision_checks=False, target=1,
            project_max_active_tasks=300):
        accepted = Mock(status_code=201)
        accepted.json.return_value = {"id": 123}
        argv = [
            "feeder.py",
            "--once",
            "--target", str(target),
            "--max-samples", "1",
            "--solver-revision", SOLVER_REVISION,
            "--library-revision", LIBRARY_REVISION,
            *cli_args,
        ]
        def scheduler_json(path, params=None):
            if path == "/api/tasks/summary":
                return {"statuses": {}}
            if path == "/api/allocations":
                return []
            if path == "/api/projects":
                return [{
                    "name": feeder.MFT_PROJECT,
                    "max_active_tasks": project_max_active_tasks,
                    "auto_pull": False,
                }]
            if path == "/api/tasks":
                return []
            if path == "/api/task-capacity":
                return {
                    "ready_fit_slots": 1,
                    "queue_state": "ready",
                    "queue_reason": "",
                }
            raise AssertionError((path, params))

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
            p08_completion_check = stack.enter_context(patch.object(
                feeder, "validate_p08_completion"))
            if fail_local_revision_checks:
                error = AssertionError(
                    "local revision vetting and p08 completion must be bypassed")
                solver_revision_check.side_effect = error
                library_revision_check.side_effect = error
                p08_completion_check.side_effect = error
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
                "_scheduler_json",
                side_effect=scheduler_json,
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
            live_snapshot = stack.enter_context(patch.object(
                feeder.scheduler_client,
                "live_project_submission_snapshot",
                return_value={
                    "project_submission_slots": project_max_active_tasks,
                },
            ))
            post = stack.enter_context(patch.object(
                feeder.scheduler_client.requests,
                "post",
                return_value=accepted,
            ))
            feeder.main()

        if target > feeder.MFT_PROJECT_MAX_ACTIVE_TASKS:
            live_snapshot.assert_called_once_with(
                target,
                max_project_active_tasks=(
                    feeder.MAX_POOLED_PROJECT_ACTIVE_TASKS),
            )
        if fail_local_revision_checks:
            solver_revision_check.assert_not_called()
            library_revision_check.assert_not_called()
            p08_completion_check.assert_not_called()
        post.assert_called_once()
        return copy.deepcopy(post.call_args.kwargs["json"])

    def test_aedt_pooled_accepts_target_500_with_project_cap_510(self):
        payload = self._capture_cli_payload(
            [
                "--aedt-pooled",
                "--aedt-pool-url", "https://pool.example.test:8443",
            ],
            target=500,
            project_max_active_tasks=510,
        )

        self.assertEqual(payload["aedt_backend"], "pooled")

    def test_non_pooled_retains_target_and_project_cap_errors(self):
        with self.assertRaises(feeder.SchedulerError) as target_error:
            self._capture_cli_payload([], target=51)
        self.assertEqual(
            str(target_error.exception),
            "standalone feeder hard cap is 50; use rapid_campaign.py for "
            "300-task production promotion",
        )

        with self.assertRaises(feeder.SchedulerError) as project_cap_error:
            self._capture_cli_payload([], project_max_active_tasks=301)
        self.assertEqual(
            str(project_cap_error.exception),
            "scheduler MFT project max_active_tasks must be an integer "
            "between 1 and 300, got 301",
        )

    def test_high_target_requires_the_pooled_backend_marker(self):
        with self.assertRaises(feeder.SchedulerError) as error:
            feeder.step(1, target=500, pooled_submission={})

        self.assertEqual(
            str(error.exception),
            "direct feeder hard cap is 50; only rapid_campaign may authorize "
            "production promotion",
        )

    def test_prevalidated_pooled_cycle_skips_initial_reconcile_and_capacity(self):
        accepted = Mock(status_code=201)
        accepted.json.return_value = {"id": 124}
        events = []

        def reconcile(*_args, **_kwargs):
            events.append("reconcile")
            return None

        def post(*_args, **_kwargs):
            events.append("post")
            return accepted

        with patch.object(
                feeder.scheduler_client,
                "campaign_mutation_lock_is_held",
                return_value=True,
        ), patch.object(
                feeder.scheduler_client,
                "reconcile_task_id",
                side_effect=reconcile,
        ) as dedupe, patch.object(
                feeder.scheduler_client,
                "live_project_submission_snapshot",
        ) as capacity, patch.object(
                feeder.scheduler_client.requests,
                "post",
                side_effect=post,
        ) as submit:
            task_id = feeder.scheduler_client.submit_verification(
                "q23-prevalidated", "wd", {"candidate": 1}, {},
                solver_revision=SOLVER_REVISION,
                library_revision=LIBRARY_REVISION,
                aedt_backend="pooled",
                required_hard_cap=500,
                max_project_active_tasks=600,
                prevalidated_cycle=True,
            )

        self.assertEqual(task_id, 124)
        self.assertEqual(events, ["post"])
        dedupe.assert_not_called()
        capacity.assert_not_called()
        submit.assert_called_once()

    def test_normal_cycle_still_reconciles_before_post(self):
        accepted = Mock(status_code=201)
        accepted.json.return_value = {"id": 125}
        events = []

        with patch.object(
                feeder.scheduler_client,
                "campaign_mutation_lock_is_held",
                return_value=True,
        ), patch.object(
                feeder.scheduler_client,
                "reconcile_task_id",
                side_effect=lambda *_args, **_kwargs: events.append("reconcile"),
        ) as dedupe, patch.object(
                feeder.scheduler_client,
                "live_project_submission_snapshot",
                return_value={"project_submission_slots": 1},
        ), patch.object(
                feeder.scheduler_client.requests,
                "post",
                side_effect=lambda *_args, **_kwargs: (
                    events.append("post") or accepted
                ),
        ):
            task_id = feeder.scheduler_client.submit_verification(
                "normal-cycle", "wd", {"candidate": 1}, {},
                solver_revision=SOLVER_REVISION,
                library_revision=LIBRARY_REVISION,
            )

        self.assertEqual(task_id, 125)
        self.assertEqual(events, ["reconcile", "post"])
        dedupe.assert_called_once()

    def test_prevalidated_uncertain_post_still_reconciles(self):
        events = []

        def reconcile(*_args, **_kwargs):
            events.append("reconcile")
            return 126

        def post(*_args, **_kwargs):
            events.append("post")
            raise feeder.scheduler_client.requests.Timeout("response lost")

        with patch.object(
                feeder.scheduler_client,
                "campaign_mutation_lock_is_held",
                return_value=True,
        ), patch.object(
                feeder.scheduler_client,
                "reconcile_task_id",
                side_effect=reconcile,
        ) as dedupe, patch.object(
                feeder.scheduler_client,
                "live_project_submission_snapshot",
        ) as capacity, patch.object(
                feeder.scheduler_client.requests,
                "post",
                side_effect=post,
        ) as submit:
            task_id = feeder.scheduler_client.submit_verification(
                "prevalidated-response-loss", "wd", {"candidate": 1}, {},
                solver_revision=SOLVER_REVISION,
                library_revision=LIBRARY_REVISION,
                aedt_backend="pooled",
                required_hard_cap=500,
                max_project_active_tasks=600,
                prevalidated_cycle=True,
            )

        self.assertEqual(task_id, 126)
        self.assertEqual(events, ["post", "reconcile"])
        dedupe.assert_called_once()
        capacity.assert_not_called()
        submit.assert_called_once()

    def test_prevalidated_cycle_cannot_bypass_non_pooled_capacity(self):
        with patch.object(
                feeder.scheduler_client,
                "campaign_mutation_lock_is_held",
                return_value=True,
        ), patch.object(
                feeder.scheduler_client.requests,
                "post",
        ) as post:
            with self.assertRaisesRegex(
                    feeder.scheduler_client.ProjectContractError,
                    "requires pooled submission"):
                feeder.scheduler_client.submit_verification(
                    "invalid-prevalidated", "wd", {}, {},
                    solver_revision=SOLVER_REVISION,
                    library_revision=LIBRARY_REVISION,
                    required_hard_cap=500,
                    prevalidated_cycle=True,
                )
        post.assert_not_called()

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
            "[feeder] WARNING: local revision vetting and the p08 completion "
            "gate were bypassed; "
            f"using pinned solver SHA {SOLVER_REVISION} and "
            f"library SHA {LIBRARY_REVISION}"
        ])

    def test_aedt_pooled_injects_backend_environment_and_resources(self):
        expected_env = {
            "AEDT_POOL_AUTOMATION_LOCK_TIMEOUT_SECONDS": "7200",
            "MFT_AEDT_BACKEND": "pooled",
            "MFT_AEDT_SHARED_CANARY": "1",
            "MFT_AEDT_SCHEDULER_URL": "https://pool.example.test:8443",
            "MFT_AEDT_SESSION_VERSION": "2025.2",
            "MFT_AEDT_SESSION_PROFILE": feeder.AEDT_SESSION_PROFILE,
            "MFT_AEDT_ISOLATION_POLICY": "family",
            "MFT_AEDT_POOL_FILL_TIMEOUT_SECONDS": "900",
            "MFT_AEDT_POOL_WORKSPACE": (
                "/gpfs/tmp_cpu2/mft_pool/mft-${SLURM_SCHED_TASK_ID}"
            ),
            "MFT_AEDT_WORKSPACE_PATH": (
                "/gpfs/tmp_cpu2/mft_pool/mft-${SLURM_SCHED_TASK_ID}"
            ),
            "MFT_SLURM_SCHEDULER_ROOT": "/opt/pool package",
            "SLURM_AEDT_POOL_CLIENT_TOKEN_FILE": "/run/pool token",
        }

        defaults = feeder._argument_parser().parse_args([
            "--aedt-pooled",
            "--aedt-pool-url", "https://pool.example.test:8443",
        ])
        self.assertEqual(
            defaults.aedt_pool_pkg_root,
            "$HOME/slurm_scheduler/aedt_pool_pkg",
        )
        self.assertEqual(
            defaults.aedt_pool_client_token_file,
            "$HOME/slurm_scheduler/aedt_pool_client",
        )
        self.assertEqual(defaults.aedt_session_version, "2025.2")
        self.assertEqual(defaults.aedt_isolation_policy, "family")
        self.assertEqual(defaults.pooled_cpus, 1)
        self.assertEqual(defaults.pooled_memory_mb, 6144)

        default_payload = self._capture_cli_payload([
            "--aedt-pooled",
            "--aedt-pool-url", "https://pool.example.test:8443",
        ])
        self.assertEqual(default_payload["cpus"], 1)
        self.assertIn(
            'export MFT_SLURM_SCHEDULER_ROOT='
            '"$HOME/slurm_scheduler/aedt_pool_pkg";',
            default_payload["command"],
        )
        self.assertIn(
            'export SLURM_AEDT_POOL_CLIENT_TOKEN_FILE='
            '"$HOME/slurm_scheduler/aedt_pool_client";',
            default_payload["command"],
        )

        pooled_payload = self._capture_cli_payload([
            "--aedt-pooled",
            "--aedt-pool-url", "https://pool.example.test:8443",
            "--aedt-pool-pkg-root", "/opt/pool package",
            "--aedt-pool-client-token-file", "/run/pool token",
            "--pooled-cpus", "2",
            "--pooled-memory-mb", "8192",
        ])
        self.assertEqual(pooled_payload["aedt_backend"], "pooled")
        self.assertEqual(pooled_payload["cpus"], 2)
        self.assertEqual(pooled_payload["memory_mb"], 8192)
        self.assertNotIn("submission_env", pooled_payload)
        expected_env_setup = "\n".join(
            f"export {key}="
            f"{feeder.scheduler_client._shell_double_quote_expandable(value)}"
            for key, value in sorted(expected_env.items())
        )
        self.assertEqual(pooled_payload["env_setup"], expected_env_setup)
        pooled_command = pooled_payload["command"]
        self.assertIn(
            'MFT_WORKDIR="$MFT_GPFS_WORKDIR";',
            pooled_command,
        )
        self.assertNotIn("MFT_NVME_WORKDIR", pooled_command)
        self.assertNotIn("MFT_ENROOT_FREE_KB", pooled_command)
        self.assertNotIn(
            "findmnt -n -o FSTYPE -T /enroot",
            pooled_command,
        )
        self.assertIn(
            'cleanup() { rm -rf -- "${MFT_GPFS_WORKDIR}" 2>/dev/null; };',
            pooled_command,
        )

        env_exports = "".join(
            f"export {key}="
            f"{feeder.scheduler_client._shell_double_quote_expandable(value)}; "
            for key, value in sorted(expected_env.items())
        )
        self.assertEqual(pooled_command.count(env_exports), 1)
        for key, value in expected_env.items():
            self.assertIn(
                f"export {key}="
                f"{feeder.scheduler_client._shell_double_quote_expandable(value)};",
                pooled_command,
            )
        self.assertNotIn(
            "SLURM_AEDT_POOL_BOOTSTRAP_TOKEN_FILE", pooled_command
        )
        self.assertIn(
            'export MFT_AEDT_POOL_WORKSPACE='
            '"/gpfs/tmp_cpu2/mft_pool/mft-${SLURM_SCHED_TASK_ID}";',
            pooled_command,
        )
        self.assertNotIn("MFT_AEDT_POOL_WORKSPACE_ROOT", pooled_command)

        shared_payload = self._capture_cli_payload([
            "--aedt-pooled",
            "--aedt-pool-url", "https://pool.example.test:8443",
            "--aedt-isolation-policy", "shared_if_compatible",
        ])
        self.assertIn(
            'export MFT_AEDT_SESSION_VERSION="2025.2";',
            shared_payload["command"],
        )
        self.assertIn(
            'export MFT_AEDT_ISOLATION_POLICY="shared_if_compatible";',
            shared_payload["command"],
        )
        self.assertIn(
            'export MFT_AEDT_POOL_WORKSPACE='
            '"/gpfs/tmp_cpu2/mft_pool/mft-${SLURM_SCHED_TASK_ID}";',
            shared_payload["command"],
        )

        legacy_payload = self._capture_cli_payload([])
        legacy_command = legacy_payload["command"]
        self.assertIn("MFT_NVME_WORKDIR=/enroot/", legacy_command)
        self.assertIn("MFT_ENROOT_FREE_KB=", legacy_command)
        self.assertIn(
            "findmnt -n -o FSTYPE -T /enroot",
            legacy_command,
        )
        self.assertIn("MFT_WORKDIR=$MFT_NVME_WORKDIR;", legacy_command)
        normalized = copy.deepcopy(pooled_payload)
        normalized.pop("aedt_backend")
        normalized["cpus"] = legacy_payload["cpus"]
        normalized["memory_mb"] = legacy_payload["memory_mb"]
        normalized.pop("command")
        normalized.pop("env_setup")
        legacy_without_command = copy.deepcopy(legacy_payload)
        legacy_without_command.pop("command")
        self.assertEqual(normalized, legacy_without_command)

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
                "AEDT_POOL_AUTOMATION_LOCK_TIMEOUT_SECONDS",
                "MFT_SLURM_SCHEDULER_ROOT",
                "SLURM_AEDT_POOL_CLIENT_TOKEN_FILE",
                "SLURM_AEDT_POOL_BOOTSTRAP_TOKEN_FILE"):
            self.assertNotIn(key, payload["command"])


class FeederSimulationPolicyTests(unittest.TestCase):
    def test_policy_snapshot_uses_desired_not_project_safety_cap(self):
        project = {
            "name": feeder.MFT_PROJECT,
            "max_active_tasks": 510,
            "simulation_policy": {
                "desired_simulations": 500,
                "effective_simulations": 472,
                "validated_concurrency_limit": 500,
                "policy_revision": 19,
                "scale_down_mode": "drain",
                "resource_constraint": {"code": "license_headroom"},
            },
        }
        with patch.object(feeder, "_scheduler_json", return_value=project) as get:
            policy = feeder.simulation_policy_snapshot()

        get.assert_called_once_with(
            f"/api/projects/{feeder.MFT_PROJECT}")
        self.assertEqual(policy["desired_simulations"], 500)
        self.assertEqual(policy["effective_simulations"], 472)
        self.assertEqual(policy["policy_revision"], 19)

    def test_policy_snapshot_fails_closed_above_validated_limit(self):
        project = {
            "name": feeder.MFT_PROJECT,
            "desired_simulations": 500,
            "validated_concurrency_limit": 250,
            "policy_revision": 3,
            "scale_down_mode": "drain",
        }
        with patch.object(feeder, "_scheduler_json", return_value=project):
            with self.assertRaisesRegex(
                    feeder.SchedulerError, "exceeds the validated"):
                feeder.simulation_policy_snapshot()

    def test_pooled_loop_prefers_durable_policy_over_explicit_fallback(self):
        args = feeder._argument_parser().parse_args([
            "--loop", "600",
            "--target", "40",
            "--aedt-pooled",
            "--aedt-pool-url", "https://pool.example.test",
        ])
        expected = {
            "desired_simulations": 500,
            "effective_simulations": 480,
            "validated_concurrency_limit": 500,
            "policy_revision": 20,
            "scale_down_mode": "drain",
            "resource_constraint": None,
        }
        with patch.object(
                feeder, "simulation_policy_snapshot", return_value=expected):
            target, policy = feeder._cycle_target(args)

        self.assertEqual(target, 500)
        self.assertEqual(policy, expected)

    def test_new_pooled_controller_has_no_fixed_cli_target(self):
        args = feeder._argument_parser().parse_args([
            "--loop", "600",
            "--aedt-pooled",
            "--aedt-pool-url", "https://pool.example.test",
        ])
        self.assertIsNone(args.target)
        with patch.object(
                feeder,
                "simulation_policy_snapshot",
                side_effect=feeder.SchedulerError("policy unavailable"),
        ):
            with self.assertRaisesRegex(
                    feeder.SchedulerError, "policy unavailable"):
                feeder._cycle_target(args)

    def test_explicit_target_is_only_an_old_scheduler_fallback(self):
        args = feeder._argument_parser().parse_args([
            "--loop", "600",
            "--target", "40",
            "--aedt-pooled",
            "--aedt-pool-url", "https://pool.example.test",
        ])
        with patch.object(
                feeder,
                "simulation_policy_snapshot",
                side_effect=feeder.SimulationPolicyUnavailable(
                    "old scheduler"),
        ):
            target, policy = feeder._cycle_target(args)
        self.assertEqual(target, 40)
        self.assertIsNone(policy)

        with patch.object(
                feeder,
                "simulation_policy_snapshot",
                side_effect=feeder.SchedulerError("invalid live policy"),
        ):
            with self.assertRaisesRegex(
                    feeder.SchedulerError, "invalid live policy"):
                feeder._cycle_target(args)

    def test_loop_controller_lock_rejects_duplicate_process(self):
        with self.subTest("lifetime lock"):
            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as directory:
                lock_path = str(Path(directory) / "feeder-controller.lock")
                argv = [
                    "feeder.py", "--loop", "600", "--target", "0",
                    "--solver-revision", SOLVER_REVISION,
                    "--library-revision", LIBRARY_REVISION,
                    "--trust-pinned-revisions",
                ]
                with FileLock(lock_path).acquire(timeout=0):
                    with patch.object(sys, "argv", argv), patch.object(
                            feeder, "CONTROLLER_LOCK", lock_path):
                        with self.assertRaisesRegex(
                                feeder.SchedulerError,
                                "another feeder controller owns"):
                            feeder.main()


def test_state_round_trip_with_configured_directory(tmp_path):
    state_dir = tmp_path / "mft_feeder"
    state = {"serial": 17, "submitted_samples": 85}
    state_path = state_dir / "feeder_state.json"

    assert not state_dir.exists()
    try:
        with patch.dict(
                os.environ,
                {"MFT_FEEDER_STATE_DIR": str(state_dir)},
        ):
            importlib.reload(feeder)
            feeder.save_state(state)

            assert Path(feeder.STATE) == state_path
            assert state_path.is_file()
            assert feeder.load_state() == state
            assert not Path(f"{state_path}.tmp").exists()
    finally:
        importlib.reload(feeder)


def test_load_state_tolerates_empty_and_corrupt_files(tmp_path, caplog):
    expected = {"serial": 0, "submitted_samples": 0}
    for name, contents in (("empty", ""), ("corrupt", "{not-json")):
        state_path = tmp_path / name / "feeder_state.json"
        state_path.parent.mkdir()
        state_path.write_text(contents, encoding="utf-8")
        caplog.clear()

        with patch.object(feeder, "STATE", str(state_path)), caplog.at_level(
                logging.WARNING, logger=feeder.__name__):
            assert feeder.load_state() == expected

        assert "is empty or corrupt" in caplog.text
        assert "starting fresh" in caplog.text


def test_save_state_falls_back_to_direct_write(tmp_path):
    state_path = tmp_path / "feeder_state.json"
    tmp_state_path = Path(f"{state_path}.tmp")
    state = {"serial": 23, "submitted_samples": 115}

    with patch.object(feeder, "STATE", str(state_path)), patch.object(
            feeder.os,
            "replace",
            side_effect=PermissionError(5, "atomic rename unsupported"),
    ) as replace, patch.object(feeder.time, "sleep") as sleep:
        feeder.save_state(state)

    assert replace.call_args_list == [
        call(str(tmp_state_path), str(state_path)),
        call(str(tmp_state_path), str(state_path)),
        call(str(tmp_state_path), str(state_path)),
    ]
    assert sleep.call_args_list == [call(0.5), call(0.5)]
    assert json.loads(state_path.read_text(encoding="utf-8")) == state
    assert not tmp_state_path.exists()


def test_q23_state_permission_fallback_skips_known_useless_retries(tmp_path):
    state_path = tmp_path / "feeder_state.json"
    tmp_state_path = Path(f"{state_path}.tmp")
    state = {"serial": 24, "submitted_samples": 120}

    with patch.object(feeder, "STATE", str(state_path)), patch.object(
            feeder.os,
            "replace",
            side_effect=PermissionError(5, "atomic rename unsupported"),
    ) as replace, patch.object(feeder.time, "sleep") as sleep:
        feeder.save_state(state, immediate_permission_fallback=True)

    replace.assert_called_once_with(str(tmp_state_path), str(state_path))
    sleep.assert_not_called()
    assert json.loads(state_path.read_text(encoding="utf-8")) == state
    assert not tmp_state_path.exists()


def test_q23_state_fallback_still_retries_other_os_errors(tmp_path):
    state_path = tmp_path / "feeder_state.json"
    state = {"serial": 25, "submitted_samples": 125}
    real_replace = feeder.os.replace
    attempts = 0

    def transient_replace(source, target):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError(16, "temporarily busy")
        return real_replace(source, target)

    with patch.object(feeder, "STATE", str(state_path)), patch.object(
            feeder.os,
            "replace",
            side_effect=transient_replace,
    ) as replace, patch.object(feeder.time, "sleep") as sleep:
        feeder.save_state(state, immediate_permission_fallback=True)

    assert replace.call_count == 2
    sleep.assert_called_once_with(0.5)
    assert json.loads(state_path.read_text(encoding="utf-8")) == state


def test_q23_sealed_cycle_forwards_prevalidation_and_has_no_submit_delay():
    counts = {"queued": 0, "attaching": 0, "running": 0}
    capacity = {
        "ready_fit_slots": 500,
        "project_submission_slots": 500,
        "submission_allowed": True,
        "queue_state": "ready",
        "queue_reason": "",
        "project_active": 0,
    }
    state = {"serial": 0, "submitted_samples": 0}
    q23_submission = {
        "aedt_backend": "pooled",
        "prevalidated_cycle": True,
        "submission_delay_seconds": 0.0,
        "immediate_state_permission_fallback": True,
    }

    with patch.object(
            feeder.scheduler_client,
            "campaign_mutation_lock_is_held",
            return_value=True,
    ), patch.object(
            feeder,
            "scheduler_snapshot",
            return_value=(counts, counts, [], capacity),
    ) as snapshot, patch.object(
            feeder,
            "dataset_collection_snapshot",
            return_value=(0, set()),
    ), patch.object(
            feeder,
            "campaign_inventory",
            return_value=[],
    ), patch.object(
            feeder,
            "reserved_unjudged_rows",
            return_value=0,
    ), patch.object(
            feeder,
            "load_state",
            return_value=state,
    ), patch.object(
            feeder,
            "cursor_after_valid_candidates",
            return_value=0,
    ), patch.object(
            feeder,
            "next_valid_candidate",
            return_value=(1, 0, {"candidate": 1}),
    ), patch.object(
            feeder,
            "submit",
            return_value=701,
    ) as submit, patch.object(
            feeder,
            "save_state",
    ) as save, patch.object(feeder.time, "sleep") as sleep:
        assert feeder._step_locked(
            None,
            target=1,
            solver_revision=SOLVER_REVISION,
            library_revision=LIBRARY_REVISION,
            _pooled_submission=q23_submission,
        ) is True

    snapshot.assert_called_once()
    submit.assert_called_once()
    assert submit.call_args.kwargs["prevalidated_cycle"] is True
    assert "submission_delay_seconds" not in submit.call_args.kwargs
    assert "immediate_state_permission_fallback" not in submit.call_args.kwargs
    assert submit.call_args.kwargs["required_hard_cap"] == 1
    save.assert_called_once()
    assert save.call_args.kwargs == {"immediate_permission_fallback": True}
    sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
