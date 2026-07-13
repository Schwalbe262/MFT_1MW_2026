import copy
import json
import os
import sys
import tempfile
import unittest
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock


CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAMPAIGN_DIR))

import _adopted_refill_sha688c6f9 as controller  # noqa: E402


def evidence(action="refill_300", *, terminal=20, valid=18, paused=False):
    return {
        "time": "2026-07-12T00:00:00+00:00",
        "action": action,
        "paused": paused,
        "pause_reasons": ["health_gate"] if paused else [],
        "local3_passed": True,
        "production_active": 230,
        "production_terminal": terminal,
        "production_valid": valid,
        "production_invalid": terminal - valid,
        "production_valid_rate": valid / terminal if terminal else None,
        "strict_full_rows": 10,
        "strict_progress": {
            "previous_rows": 0,
            "observed_rows": 10,
            "growth_at": "2026-07-12T00:00:00+00:00",
            "reasons": [],
        },
        "target_strict_rows": controller.TARGET_STRICT_ROWS,
        "task_outcomes": {},
        "initial_statuses": {"running": 230, "completed": 20},
    }


def adoption():
    return {
        "manifest": {"tasks": []},
        "submission_journal": {},
        "inventory": [],
        "initial_tasks": [],
        "generation_task_count": controller.INITIAL_COUNT,
        "statuses": {"running": controller.INITIAL_COUNT},
        "adoption_sha256": controller.MANIFEST_SHA256,
    }


class AtomicAndStateTests(unittest.TestCase):
    def test_atomic_json_retries_permission_errors_with_a_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            real_replace = os.replace
            attempts = []

            def flaky_replace(source, destination):
                attempts.append((source, destination))
                if len(attempts) < 3:
                    raise PermissionError("transient mounted-drive denial")
                return real_replace(source, destination)

            with mock.patch.object(
                    controller.os, "replace", side_effect=flaky_replace), mock.patch.object(
                controller.time, "sleep",
            ) as sleep:
                controller._atomic_json(path, {"ok": True})

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"ok": True})
            self.assertEqual(len(attempts), 3)
            self.assertEqual(sleep.call_count, 2)

    def test_best_effort_canonical_does_not_retry_after_sealed_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            canonical = controller._new_controller_state()
            committed = copy.deepcopy(canonical)
            committed["state_revision"] = 1
            controller._write_immutable_generation(path, committed)
            path.write_text(json.dumps(canonical), encoding="utf-8")
            attempts = []

            def denied_replace(source, destination):
                attempts.append((source, destination))
                raise PermissionError("mounted drive cannot overwrite canonical")

            with mock.patch.object(
                    controller.os, "replace", side_effect=denied_replace), mock.patch.object(
                    controller.time, "sleep") as sleep, mock.patch("sys.stderr"):
                repaired = controller._best_effort_canonical(
                    path, committed)

            self.assertFalse(repaired)
            self.assertEqual(len(attempts), 1)
            sleep.assert_not_called()
            self.assertEqual(
                controller._authoritative_state(
                    path, controller._validate_controller_state, repair=False),
                committed,
            )

    def test_best_effort_canonical_staging_failure_is_one_shot(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            path.write_text(json.dumps({"state_revision": 0}), encoding="utf-8")
            real_open = Path.open

            def deny_staged_write(candidate, mode="r", *args, **kwargs):
                if mode == "wb" and candidate.name.endswith(".tmp"):
                    raise PermissionError("mounted drive denied staged write")
                return real_open(candidate, mode, *args, **kwargs)

            with mock.patch.object(
                    Path, "open", new=deny_staged_write), mock.patch.object(
                    controller.time, "sleep") as sleep, mock.patch("sys.stderr"):
                repaired = controller._best_effort_canonical(
                    path, {"state_revision": 1})

            self.assertFalse(repaired)
            sleep.assert_not_called()

    def test_initial_feeder_state_starts_after_sealed_250_and_rejects_rewind(self):
        state = controller._initial_feeder_state(adoption())

        self.assertEqual(state["serial"], 17_361)
        self.assertEqual(state["candidate_cursor"], 1_843)
        self.assertEqual(state["candidate_raw_index"], 1_842)
        self.assertEqual(state["outstanding"], list(range(27_471, 27_721)))
        controller._validate_feeder_state(state)

        for key, value in (("serial", 17_360), ("candidate_cursor", 1_842)):
            rewound = copy.deepcopy(state)
            rewound[key] = value
            with self.assertRaisesRegex(RuntimeError, "replay the old cursor/serial"):
                controller._validate_feeder_state(rewound)

    def test_primary_turn_cap_is_total_main_plus_side(self):
        params = {
            "cw1": 1.0,
            "N1_main": 8,
            "N1_side": 1,
            "wcp_t": 20.0,
            "core_plate_t": 20.0,
            "wcp_pad_t": 2.0,
            "core_plate_pad_t": 2.0,
        }
        with self.assertRaisesRegex(RuntimeError, "primary cap mismatch"):
            controller._validate_candidate_contract(params, "test")


class DurableGenerationStateTests(unittest.TestCase):
    def test_destructive_replace_failure_recovers_controller_from_generations(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "controller.json"

            def destructive_denial(source, destination):
                Path(destination).unlink(missing_ok=True)
                raise PermissionError("mounted drive removed target before denial")

            with mock.patch.object(
                    controller, "STATE_PATH", path), mock.patch.object(
                    controller, "ATOMIC_ATTEMPTS", 2), mock.patch.object(
                    controller.time, "sleep"), mock.patch.object(
                    controller.os, "replace", side_effect=destructive_denial), mock.patch(
                    "sys.stderr"):
                created = controller._load_controller_state(create=True)

            self.assertEqual(created["state_revision"], 1)
            self.assertFalse(path.exists())
            self.assertEqual(
                len(list(path.parent.glob(f"{path.name}.gen-*.json"))), 2)

            with mock.patch.object(
                    controller, "STATE_PATH", path), mock.patch.object(
                    controller, "_initialize_durable_state",
                    side_effect=AssertionError("must not initialize over generations")):
                recovered = controller._load_controller_state(create=False)

            self.assertEqual(recovered, created)

    def test_same_revision_generation_conflict_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "controller.json"
            first = controller._new_controller_state()
            second = copy.deepcopy(first)
            second["last_action"] = "different-valid-payload"
            controller._write_immutable_generation(path, first)
            controller._write_immutable_generation(path, second)

            with self.assertRaisesRegex(RuntimeError, "same revision"):
                controller._load_durable_state(
                    path,
                    controller._validate_controller_state,
                    controller._new_controller_state,
                    create=False,
                )

    def test_canonical_generation_same_revision_conflict_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "controller.json"
            generation_state = controller._new_controller_state()
            canonical_state = copy.deepcopy(generation_state)
            canonical_state["last_action"] = "conflicting-canonical"
            controller._write_immutable_generation(path, generation_state)
            controller._atomic_json(path, canonical_state)

            with self.assertRaisesRegex(RuntimeError, "same revision"):
                controller._load_durable_state(
                    path,
                    controller._validate_controller_state,
                    controller._new_controller_state,
                    create=True,
                )

    def test_explicit_legacy_migration_preserves_payload_and_creates_two_generations(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "controller.json"
            legacy = controller._new_controller_state()
            legacy.pop("state_revision")
            legacy["last_action"] = "preserve-me"
            legacy["task_outcomes"] = {"27928": {"state": "invalid"}}
            path.write_text(json.dumps(legacy), encoding="utf-8")

            migrated = controller._migrate_legacy_state(
                path, controller._validate_controller_state)
            loaded = controller._load_durable_state(
                path,
                controller._validate_controller_state,
                controller._new_controller_state,
                create=False,
            )

            self.assertEqual(migrated["state_revision"], 1)
            self.assertEqual(loaded, migrated)
            self.assertEqual(loaded["last_action"], "preserve-me")
            self.assertEqual(
                loaded["task_outcomes"], {"27928": {"state": "invalid"}})
            self.assertEqual(
                len(list(path.parent.glob(f"{path.name}.gen-*.json"))), 2)

    def test_corrupt_generation_fails_closed_even_with_valid_canonical(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "controller.json"
            state = controller._new_controller_state()
            controller._atomic_json(path, state)
            corrupt = path.with_name(
                f"{path.name}.gen-{0:020d}-{'0' * 64}.json")
            corrupt.write_text("{broken", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "unreadable JSON state candidate"):
                controller._load_durable_state(
                    path,
                    controller._validate_controller_state,
                    controller._new_controller_state,
                    create=False,
                )

    def test_feeder_missing_canonical_never_rewinds_cursor_or_serial(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "feeder.json"
            with mock.patch.object(controller, "FEEDER_STATE_PATH", path):
                state = controller._load_feeder_state(adoption(), create=True)
                state["serial"] = 17_611
                state["candidate_cursor"] = 2_795
                state["candidate_raw_index"] = 2_794
                state["candidate_cursors"][state["candidate_generation"]] = 2_795
                controller._save_durable_state(
                    path,
                    state,
                    controller._validate_feeder_state,
                    transition_validator=controller._validate_feeder_transition,
                )
                committed_revision = state["state_revision"]
                path.unlink()

                recovered = controller._load_feeder_state(adoption(), create=False)

            self.assertEqual(recovered["state_revision"], committed_revision)
            self.assertEqual(recovered["serial"], 17_611)
            self.assertEqual(recovered["candidate_cursor"], 2_795)
            self.assertEqual(recovered["candidate_raw_index"], 2_794)

    def test_corrupt_recovery_artifact_forbids_fresh_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "controller.json"
            path.with_name(f"{path.name}.tmp").write_text(
                "{broken", encoding="utf-8")
            with mock.patch.object(controller, "STATE_PATH", path), mock.patch.object(
                    controller, "_initialize_durable_state",
                    side_effect=AssertionError("must fail instead of initialize")):
                with self.assertRaisesRegex(RuntimeError, "history exists"):
                    controller._load_controller_state(create=True)

    def test_read_only_fresh_load_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "controller.json"
            feeder_path = Path(tmp) / "feeder.json"
            with mock.patch.object(
                    controller, "STATE_PATH", state_path), mock.patch.object(
                    controller, "FEEDER_STATE_PATH", feeder_path):
                state = controller._load_controller_state(create=False)
                feeder_state = controller._load_feeder_state(adoption(), create=False)

            self.assertEqual(state["state_revision"], 0)
            self.assertEqual(feeder_state["state_revision"], 0)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_feeder_revisions_are_monotonic_and_stale_save_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "feeder.json"
            state = controller._load_durable_state(
                path,
                controller._validate_feeder_state,
                lambda: controller._initial_feeder_state(adoption()),
                create=True,
            )
            stale = copy.deepcopy(state)
            for expected_revision in (2, 3):
                state["serial"] += 1
                state["candidate_cursor"] += 1
                state["candidate_raw_index"] += 1
                state["candidate_cursors"][state["candidate_generation"]] = state[
                    "candidate_cursor"]
                controller._save_durable_state(
                    path,
                    state,
                    controller._validate_feeder_state,
                    transition_validator=controller._validate_feeder_transition,
                )
                self.assertEqual(state["state_revision"], expected_revision)

            current_revision_rewind = copy.deepcopy(state)
            current_revision_rewind["candidate_cursor"] -= 1
            current_revision_rewind["candidate_raw_index"] -= 1
            current_revision_rewind["candidate_cursors"][
                current_revision_rewind["candidate_generation"]] -= 1
            with self.assertRaisesRegex(RuntimeError, "replay progress"):
                controller._save_durable_state(
                    path,
                    current_revision_rewind,
                    controller._validate_feeder_state,
                    transition_validator=controller._validate_feeder_transition,
                )

            with self.assertRaisesRegex(RuntimeError, "stale state save refused"):
                controller._save_durable_state(
                    path,
                    stale,
                    controller._validate_feeder_state,
                    transition_validator=controller._validate_feeder_transition,
                )


class ReadOnlyAndReviewSealTests(unittest.TestCase):
    def test_wrong_review_sha_fails_before_deployment_or_scheduler_access(self):
        with mock.patch.object(
                controller.feeder, "_require_deployed_revisions") as deployed, mock.patch.object(
                controller, "authenticate_adoption") as authenticate:
            with self.assertRaisesRegex(RuntimeError, "root-reviewed"):
                controller.run_once(execute=True, reviewed_manifest_sha="0" * 64)

        deployed.assert_not_called()
        authenticate.assert_not_called()

    def test_read_only_creates_no_state_and_invokes_no_mutation_api(self):
        state = controller._new_controller_state()
        adopted = adoption()
        with mock.patch.object(
                controller.feeder, "_require_deployed_revisions"), mock.patch.object(
                controller, "authenticate_adoption", return_value=adopted), mock.patch.object(
                controller, "_load_controller_state", return_value=state), mock.patch.object(
                controller, "_load_feeder_state", return_value={}), mock.patch.object(
                controller, "_evidence", return_value=evidence("wait_local3")), mock.patch.object(
                controller, "FileLock") as file_lock, mock.patch.object(
                controller, "_atomic_json") as atomic, mock.patch.object(
                controller.scheduler_client, "campaign_mutation_lock") as mutation_lock, mock.patch.object(
                controller.feeder, "_authorize_adopted_refill") as authorize, mock.patch.object(
                controller.feeder, "_step_from_adopted_controller") as step:
            result = controller.run_once(execute=False)

        self.assertEqual(result["mode"], "read_only")
        self.assertEqual(result["action"], "wait_local3")
        file_lock.assert_not_called()
        atomic.assert_not_called()
        mutation_lock.assert_not_called()
        authorize.assert_not_called()
        step.assert_not_called()


class AdoptionInventoryTests(unittest.TestCase):
    @staticmethod
    def _initial_inventory():
        records = []
        submissions = {}
        tasks = []
        for index in range(controller.INITIAL_COUNT):
            name = f"{controller.PREFIX}{controller.INITIAL_FIRST_SERIAL + index:05d}"
            dedupe = f"dedupe-{index}"
            task_id = controller.INITIAL_FIRST_ID + index
            records.append({"index": index, "name": name, "dedupe_key": dedupe})
            submissions[str(index)] = {"task_id": task_id}
            tasks.append({
                "id": task_id,
                "name": name,
                "status": "running",
                "project": controller.scheduler_client.MFT_PROJECT,
                "dedupe_key": dedupe,
                "cpus": controller.CPUS,
                "memory_mb": controller.MEMORY_MB,
                "gpus": 0,
                "timeout_seconds": controller.TIMEOUT_SECONDS,
                "required_capability": "conda:pyaedt2026v1",
                "env_profile": "pyaedt2026v1",
                "scheduling_profile": "fea_bursty",
                "remote_cwd": controller.scheduler_client.GPFS_RUNS_REMOTE_CWD,
            })
        return {"tasks": records}, {"submissions": submissions}, tasks

    def test_exact_initial_ids_are_authenticated_while_later_refills_are_allowed(self):
        manifest, journal, tasks = self._initial_inventory()
        tasks.append({
            "id": controller.INITIAL_LAST_ID + 1,
            "name": f"{controller.PREFIX}{controller.INITIAL_LAST_SERIAL + 1:05d}",
            "status": "queued",
        })
        with mock.patch.object(
                controller.feeder, "campaign_inventory", return_value=tasks):
            result = controller._authenticate_scheduler_cohort(manifest, journal)

        self.assertEqual(len(result["initial_tasks"]), controller.INITIAL_COUNT)
        self.assertEqual(result["generation_task_count"], controller.INITIAL_COUNT + 1)
        self.assertEqual(result["statuses"], {"running": controller.INITIAL_COUNT})


class PromotionGateTests(unittest.TestCase):
    def test_health_gate_reasons_pause_and_strict_stall_is_fail_closed(self):
        state = controller._new_controller_state()
        state["promoted_at"] = (
            datetime.now(timezone.utc) - timedelta(minutes=91)).isoformat()
        state["last_strict_rows"] = 10
        state["last_strict_growth_at"] = state["promoted_at"]
        production = {
            "active": 200,
            "outcomes": [],
            "cache": {},
        }
        with mock.patch.object(
                controller.rapid_campaign,
                "inspect_production_tasks",
                return_value=production,
        ), mock.patch.object(
            controller.rapid_campaign,
            "_production_gate_reasons",
            return_value=["repeated_runtime_error:abc:3"],
        ) as health, mock.patch.object(
            controller, "_strict_rows", return_value=10,
        ), mock.patch.object(controller, "_local3_passed", return_value=True):
            result = controller._evidence(adoption(), state)

        health.assert_called_once_with(production)
        self.assertTrue(result["paused"])
        self.assertEqual(result["action"], "manual_intervention")
        self.assertIn("repeated_runtime_error:abc:3", result["pause_reasons"])
        self.assertIn("strict_dataset_growth_stalled_90m", result["pause_reasons"])

    def test_stale_prelock_success_is_revoked_by_locked_evidence(self):
        state = controller._new_controller_state()
        locked = evidence("manual_intervention", paused=True)
        with mock.patch.object(
                controller.feeder, "_require_deployed_revisions"), mock.patch.object(
                controller, "authenticate_adoption", side_effect=[adoption(), adoption()]), mock.patch.object(
                controller, "_load_controller_state", return_value=state), mock.patch.object(
                controller, "_load_feeder_state", return_value={}), mock.patch.object(
                controller, "_recover_incomplete_cycles", return_value=[]), mock.patch.object(
                controller, "_evidence", side_effect=[evidence(), locked]), mock.patch.object(
                controller, "_save_controller_state"), mock.patch.object(
                controller, "FileLock", return_value=nullcontext()), mock.patch.object(
                controller.scheduler_client,
                "campaign_mutation_lock",
                return_value=nullcontext(),
        ), mock.patch.object(
            controller.feeder, "_authorize_adopted_refill",
        ) as authorize, mock.patch.object(
            controller.feeder, "_step_from_adopted_controller",
        ) as step:
            result = controller.run_once(
                execute=True,
                reviewed_manifest_sha=controller.MANIFEST_SHA256,
            )

        self.assertEqual(result["action"], "manual_intervention")
        authorize.assert_not_called()
        step.assert_not_called()


class FormalApiAndRecoveryTests(unittest.TestCase):
    def test_ready_cycle_uses_only_formal_authorize_and_step_apis(self):
        state = controller._new_controller_state()
        auth = object()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
                controller, "CYCLE_ROOT", Path(tmp) / "cycles"), mock.patch.object(
                controller.feeder, "_require_deployed_revisions"), mock.patch.object(
                controller, "authenticate_adoption", side_effect=[adoption(), adoption()]), mock.patch.object(
                controller, "_load_controller_state", return_value=state), mock.patch.object(
                controller, "_load_feeder_state", return_value={}), mock.patch.object(
                controller, "_recover_incomplete_cycles", return_value=[]), mock.patch.object(
                controller, "_evidence", side_effect=[evidence(), evidence()]), mock.patch.object(
                controller, "_save_controller_state"), mock.patch.object(
                controller, "FileLock", return_value=nullcontext()), mock.patch.object(
                controller.scheduler_client,
                "campaign_mutation_lock",
                return_value=nullcontext(),
        ), mock.patch.object(
            controller.feeder, "_authorize_adopted_refill", return_value=auth,
        ) as authorize, mock.patch.object(
            controller.feeder, "_step_from_adopted_controller", return_value=True,
        ) as step, mock.patch.object(controller.feeder, "submit") as raw_submit:
            result = controller.run_once(
                execute=True,
                reviewed_manifest_sha=controller.MANIFEST_SHA256,
            )

        authorize.assert_called_once()
        step.assert_called_once()
        raw_submit.assert_not_called()
        self.assertEqual(result["mutation"]["cycle"], 1)
        self.assertEqual(state["cycle_serial"], 1)
        self.assertIsNotNone(state["promoted_at"])

    def test_incomplete_cycle_journal_is_marked_recoverable(self):
        state = controller._new_controller_state()
        state["cycle_serial"] = 1
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
                controller, "CYCLE_ROOT", Path(tmp)):
            path = controller._cycle_path(1)
            path.write_text(json.dumps({
                "cycle_serial": 1,
                "adoption_sha256": controller.MANIFEST_SHA256,
                "status": "authorized_running",
            }), encoding="utf-8")

            recovered = controller._recover_incomplete_cycles(state)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(recovered, [str(path.resolve())])
        self.assertEqual(payload["status"], "interrupted_recoverable")
        self.assertEqual(payload["recovery_feeder_state"], str(controller.FEEDER_STATE_PATH.resolve()))


if __name__ == "__main__":
    unittest.main()
