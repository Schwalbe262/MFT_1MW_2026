import copy
from datetime import datetime, timedelta, timezone
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAMPAIGN_DIR))

import _continuous_refill_b171c7c as controller  # noqa: E402
import feeder  # noqa: E402


class ContinuousRefillB171Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = controller._static_bundle()

    def setUp(self):
        patches = [
            mock.patch.object(
                controller, "_operator_cancelled_stale_prepolicy_evidence",
                return_value={"task_ids": []}),
            mock.patch.object(
                controller, "_resolved_scheduler_parent_cancel_evidence",
                return_value={"task_ids": []}),
            mock.patch.object(
                controller, "_sealed_old_timeout_contract_evidence",
                return_value={"task_ids": []}),
            mock.patch.object(
                controller, "_target300_rollback_cancelled_evidence",
                return_value={"task_ids": []}),
            mock.patch.object(
                controller, "_classify_target300_rollback_outcomes"),
            mock.patch.object(
                controller, "_dynamic_target_cancelled_evidence",
                return_value={"task_ids": []}),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_strict_snapshot_uses_newest_valid_immutable_generation(self):
        now = datetime.now(timezone.utc)
        identity = {
            "solver_revision": controller.SOLVER,
            "library_revision": controller.LIBRARY,
        }
        with tempfile.TemporaryDirectory() as directory:
            canonical = Path(directory) / "strict_data_status.json"
            canonical.write_text("{partial", encoding="utf-8")
            old = canonical.with_name(canonical.name + ".gen-old.json")
            old.write_text(json.dumps({
                "time": (now - timedelta(minutes=5)).isoformat(),
                "strict_full_rows": 111,
                "state_identity": identity,
            }), encoding="utf-8")
            newest = canonical.with_name(canonical.name + ".gen-new.json")
            newest.write_text(json.dumps({
                "time": now.isoformat(),
                "strict_full_rows": 456,
                "state_identity": identity,
            }), encoding="utf-8")
            with mock.patch.object(controller, "STRICT_STATUS_PATH", canonical):
                snapshot = controller._strict_snapshot()

        self.assertTrue(snapshot["pinned"])
        self.assertEqual(snapshot["rows"], 456)
        self.assertEqual(snapshot["source"], str(newest))

    @staticmethod
    def _recovery_row(expected, status="running"):
        return {
            "id": expected["task_id"],
            "name": expected["name"],
            "dedupe_key": expected["dedupe_key"],
            "project": controller.scheduler_client.MFT_PROJECT,
            "status": status,
            "cpus": 4,
            "memory_mb": 65_536,
            "gpus": 0,
            "timeout_seconds": 14_400,
            "env_profile": "pyaedt2026v1",
            "required_capability": "conda:pyaedt2026v1",
            "scheduling_profile": "fea_bursty",
            "remote_cwd": "__SLURM_SCHEDULER_ACCOUNT_WORKSPACE__/runs",
        }

    @staticmethod
    def _recovery_gate(submitted, gate_sha="a" * 64):
        return {
            "gate_sha256": gate_sha,
            "tasks": [
                {
                    "task_id": expected["task_id"],
                    "name": expected["name"],
                    "status": "completed",
                    "result_state": controller.scheduler_client.RESULT_VALID,
                    "strict_valid": True,
                    "result_sha256": f"{index:x}" * 64,
                    "effective_params_match": True,
                    "saturation_columns": [],
                }
                for index, expected in enumerate(submitted, start=1)
            ],
        }

    @staticmethod
    def _cycle_payload(serial=1):
        target = (
            controller.LEGACY_TARGET_ACTIVE
            if serial <= controller.TARGET_400_TRANSITION_CYCLE
            else controller.PREVIOUS_TARGET_ACTIVE
            if serial <= controller.TARGET_300_TRANSITION_CYCLE
            else controller.TARGET_ACTIVE
        )
        return {
            "schema_version": 2,
            "cycle_serial": serial,
            "created_at": "2026-07-12T00:00:00+00:00",
            "updated_at": "2026-07-12T00:00:00+00:00",
            "status": "authorized_pending",
            "plan_sha256": controller.PLAN_SHA256,
            "target_active": target,
            "evidence": {},
            "formal_journal": {"events": []},
            "error": None,
        }

    @staticmethod
    def _production_outcome(task_id, *, state="valid", status=None,
                            fingerprint=None, message=None, reason=None,
                            saturation=()):
        return {
            "task_id": task_id,
            "name": f"mft-camp-sb171c7c-le6b9b9d-{task_id}",
            "status": status or ("completed" if state == "valid" else "failed"),
            "state": state,
            "reason": reason,
            "error_fingerprint": fingerprint,
            "error_message": message,
            "terminal_at": "2026-07-12T05:00:00+00:00",
            "saturation_columns": list(saturation),
        }

    def _production_evidence(self, outcomes, started_at_by_id, *, recovery=None,
                             strict=None):
        production = {
            "tasks": [
                {
                    "id": row["task_id"], "name": row["name"],
                    "status": row["status"],
                    "started_at": started_at_by_id[row["task_id"]],
                }
                for row in outcomes
            ],
            "active": 0,
            "outcomes": outcomes,
            "cache": {str(row["task_id"]): row for row in outcomes},
        }
        recovery = recovery or {
            "tasks": [], "active": 0, "completed_valid": 0,
            "reasons": [], "wait_reasons": [],
        }
        strict = strict or {"pinned": True, "rows": 0}
        with mock.patch.object(
                controller, "_rejected_submission_evidence",
                return_value={"task_id": controller.REJECTED_TASK_ID}), \
                mock.patch.object(feeder, "campaign_inventory", return_value=[]), \
                mock.patch.object(
                    controller, "_inspect_production_with_retry",
                    return_value=production), \
                mock.patch.object(
                    controller, "_recovery_live_evidence", return_value=recovery), \
                mock.patch.object(
                    controller, "_strict_snapshot", return_value=strict), \
                mock.patch.object(
                    controller, "_operator_cancelled_stale_prepolicy_evidence",
                    return_value={"task_ids": []}), \
                mock.patch.object(
                    controller, "_resolved_scheduler_parent_cancel_evidence",
                    return_value={"task_ids": []}), \
                mock.patch.object(
                    controller, "_sealed_old_timeout_contract_evidence",
                    return_value={"task_ids": []}), \
                mock.patch.object(
                    controller, "_target300_rollback_cancelled_evidence",
                    return_value={"task_ids": []}), \
                mock.patch.object(
                    controller, "_classify_target300_rollback_outcomes"), \
                mock.patch.object(
                    controller, "_dynamic_target_cancelled_evidence",
                    return_value={"task_ids": []}):
            return controller._evidence({"task_outcomes": {}}, self.bundle)

    @staticmethod
    def _no_mutation_evidence(serial, feeder_serial=17_797):
        feeder = {
            "state_revision": 187,
            "serial": feeder_serial,
            "candidate_cursor": 3446,
            "submitted_samples": 186,
        }
        evidence = {
            "schema": controller.RECONCILIATION_EVIDENCE_SCHEMA,
            "cycle_serial": serial,
            "action": "reconciled_no_mutation",
            "observed_at": "2026-07-12T04:00:00+00:00",
            "controller_stopped": True,
            "feeder_before": feeder,
            "feeder_after": copy.deepcopy(feeder),
            "scheduler": {
                "matching_task_ids": [],
                "production_names_above_feeder_serial": [],
                "max_production_serial": feeder_serial,
            },
            "interrupted_artifacts": [],
        }
        evidence["evidence_sha256"] = controller._sha(evidence)
        return evidence

    def test_static_seals_local_recovery_and_rejected_never_started_task(self):
        with mock.patch.object(
                controller.scheduler_client.requests, "get",
                side_effect=AssertionError("static audit queried scheduler")), \
                mock.patch.object(
                    controller.scheduler_client.requests, "post",
                    side_effect=AssertionError("static audit mutated scheduler")):
            audited = controller.static_audit()
        self.assertEqual(audited["scheduler_query_count"], 0)
        self.assertEqual(audited["scheduler_mutation_count"], 0)
        self.assertEqual(
            audited["local_recovery"]["log_sha256"],
            controller.LOCAL_RECOVERY_LOG_SHA256)
        self.assertTrue(audited["local_recovery"]["b171_descendant"])
        rejected = audited["rejected_identity_cancellation"]
        self.assertEqual(rejected["task_id"], controller.REJECTED_TASK_ID)
        self.assertTrue(rejected["never_started"])
        self.assertTrue(rejected["excluded_from_production_health"])

    def test_all_first_300_feeder_payloads_preserve_sealed_dedupe_order(self):
        initial = controller._initial_feeder_state(self.bundle)
        state = copy.deepcopy(initial)
        journal = {"events": []}
        profile = json.loads(
            Path(feeder.PROFILE_PATH).read_text(encoding="utf-8"))
        profile["timeout_seconds"] = controller.TIMEOUT_SECONDS
        generation = state["candidate_generation"]
        cursor = controller.INITIAL_CURSOR
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(
                    controller, "_load_feeder_state", return_value=initial), \
                mock.patch.object(controller, "_save_feeder_state"), \
                mock.patch.object(controller, "_save_cycle"):
            cycle_path = Path(tmp) / "cycle.json"
            cycle = self._cycle_payload()
            cycle["cycle_serial"] = 1
            # The helper validates serial from the six-digit production path.
            cycle_path = Path(tmp) / "cycle-000001.json"
            controller._initialize_cycle(cycle_path, cycle)
            with controller._feeder_io(
                    self.bundle, cycle_path, cycle, journal):
                for offset, expected in enumerate(self.bundle["plan"]["tasks"]):
                    next_cursor, raw_index, params = feeder.next_valid_candidate(
                        cursor, seed=controller.SEED)
                    self.assertEqual(next_cursor, expected["candidate_cursor_after"])
                    self.assertEqual(raw_index, expected["candidate_raw_index"])
                    self.assertEqual(params, expected["params"])
                    self.assertEqual(list(params), list(expected["params"]))
                    identity = controller.scheduler_client.verification_submission_identity(
                        expected["name"], params, profile,
                        controller.SOLVER, controller.LIBRARY)
                    self.assertEqual(identity["dedupe_key"], expected["dedupe_key"])
                    journal["events"].append({
                        "name": expected["name"],
                        "dedupe_key": expected["dedupe_key"],
                        "task_id": 90_000 + offset,
                        "accepted_or_reconciled": True,
                        "ledger_committed": False,
                    })
                    state["serial"] = controller.INITIAL_SERIAL + offset + 1
                    state["candidate_cursor"] = next_cursor
                    state["candidate_cursors"][generation] = next_cursor
                    state["candidate_raw_index"] = raw_index
                    feeder.save_state(copy.deepcopy(state))
                    cursor = next_cursor
                _, _, future = feeder.next_valid_candidate(
                    cursor, seed=controller.SEED)
                self.assertEqual(list(future), sorted(future))

    def test_external_stale_pin_cancellation_requires_complete_sealed_identity(self):
        ids = (101, 102)

        def row(task_id):
            name = f"{controller.PREFIX}{task_id}"
            return {
                "id": task_id,
                "name": name,
                "dedupe_key": (
                    f"mft-al:{name}:{controller.SOLVER}:"
                    f"{controller.LIBRARY}:{task_id:016x}"
                ),
                "project": controller.scheduler_client.MFT_PROJECT,
                "status": "cancelled",
                "account_name": "dw16",
                "created_at": "2026-07-12 02:48:00",
                "attached_at": None,
                "launch_started_at": None,
                "started_at": None,
                "finished_at": "2026-07-12 10:39:55",
                "allocation_id": None,
            }

        rows = [row(task_id) for task_id in ids]
        identities = [{
            field: item.get(field)
            for field in controller._EXTERNAL_CANCELLATION_IDENTITY_FIELDS
        } for item in rows]
        digest = controller._sha(identities)
        with mock.patch.object(
                controller, "EXTERNAL_STALE_PIN_CANCELLED_IDS", ids), \
                mock.patch.object(
                    controller,
                    "EXTERNAL_STALE_PIN_CANCELLATION_IDENTITY_SHA256",
                    digest):
            evidence = controller._external_stale_pin_cancellation_evidence(rows)
            self.assertEqual(evidence["task_ids"], list(ids))
            self.assertEqual(evidence["identity_sha256"], digest)
            self.assertTrue(evidence["current_attempt_never_started"])
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                controller._external_stale_pin_cancellation_evidence(rows[:1])
            drifted = copy.deepcopy(rows)
            drifted[0]["created_at"] = "2026-07-12 02:48:01"
            with self.assertRaisesRegex(RuntimeError, "seal drifted"):
                controller._external_stale_pin_cancellation_evidence(drifted)
            launched = copy.deepcopy(rows)
            launched[0]["launch_started_at"] = "2026-07-12 10:38:00"
            with self.assertRaisesRegex(
                    RuntimeError, "current_attempt_unlaunched"):
                controller._external_stale_pin_cancellation_evidence(launched)

    def test_remote_step_cancellation_requires_exact_interval_and_identity(self):
        ids = tuple(range(201, 213))

        def row(index, task_id):
            name = f"{controller.PREFIX}{task_id}"
            return {
                "id": task_id,
                "name": name,
                "dedupe_key": (
                    f"mft-al:{name}:{controller.SOLVER}:"
                    f"{controller.LIBRARY}:{task_id:016x}"
                ),
                "project": controller.scheduler_client.MFT_PROJECT,
                "status": "cancelled",
                "account_name": f"account-{index % 4}",
                "requested_account_name": "",
                "allocation_id": 9001 + (index % 8),
                "slurm_job_id": str(730001 + (index % 8)),
                "allocation_node_name": f"n{41 + index % 8:03d}",
                "created_at": "2026-07-12 06:49:00",
                "attached_at": "2026-07-12 08:31:00",
                "launch_started_at": "2026-07-12 08:31:01",
                "started_at": "2026-07-12 08:31:02",
                "finished_at": f"2026-07-12 11:12:{12 + index % 6:02d}",
                "exit_code": None,
                "failure_message": "",
            }

        rows = [row(index, task_id) for index, task_id in enumerate(ids)]
        identities = [{
            field: item.get(field)
            for field in controller._REMOTE_STEP_CANCELLATION_IDENTITY_FIELDS
        } for item in rows]
        digest = controller._sha(identities)
        with mock.patch.object(
                controller, "REMOTE_STEP_CANCELLED_IDS", ids), \
                mock.patch.object(
                    controller,
                    "REMOTE_STEP_CANCELLATION_IDENTITY_SHA256", digest):
            evidence = controller._remote_step_cancellation_evidence(rows)
            self.assertEqual(evidence["task_ids"], list(ids))
            self.assertEqual(evidence["identity_sha256"], digest)
            self.assertEqual(evidence["allocation_count"], 8)
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                controller._remote_step_cancellation_evidence(rows[:-1])
            failed = copy.deepcopy(rows)
            failed[0]["exit_code"] = 1
            with self.assertRaisesRegex(RuntimeError, "no_solver_exit"):
                controller._remote_step_cancellation_evidence(failed)
            messaged = copy.deepcopy(rows)
            messaged[0]["failure_message"] = "unexpected"
            with self.assertRaisesRegex(RuntimeError, "no_failure_message"):
                controller._remote_step_cancellation_evidence(messaged)
            outside = copy.deepcopy(rows)
            outside[0]["finished_at"] = "2026-07-12 11:12:18"
            with self.assertRaisesRegex(RuntimeError, "interval drifted"):
                controller._remote_step_cancellation_evidence(outside)

    def test_external_cancellation_excludes_only_exact_ids_from_health(self):
        external_ids = (101, 102)

        def external_row(task_id):
            name = f"{controller.PREFIX}{task_id}"
            return {
                "id": task_id,
                "name": name,
                "dedupe_key": (
                    f"mft-al:{name}:{controller.SOLVER}:"
                    f"{controller.LIBRARY}:{task_id:016x}"
                ),
                "project": controller.scheduler_client.MFT_PROJECT,
                "status": "cancelled",
                "account_name": "dw16",
                "created_at": "2026-07-12 02:48:00",
                "attached_at": None,
                "launch_started_at": None,
                "started_at": None,
                "finished_at": "2026-07-12 10:39:55",
                "allocation_id": None,
            }

        external = [external_row(task_id) for task_id in external_ids]
        identities = [{
            field: item.get(field)
            for field in controller._EXTERNAL_CANCELLATION_IDENTITY_FIELDS
        } for item in external]
        identity_sha256 = controller._sha(identities)

        remote_ids = tuple(range(201, 213))

        def remote_row(index, task_id):
            name = f"{controller.PREFIX}{task_id}"
            return {
                "id": task_id,
                "name": name,
                "dedupe_key": (
                    f"mft-al:{name}:{controller.SOLVER}:"
                    f"{controller.LIBRARY}:{task_id:016x}"
                ),
                "project": controller.scheduler_client.MFT_PROJECT,
                "status": "cancelled",
                "account_name": f"account-{index % 4}",
                "requested_account_name": "",
                "allocation_id": 9001 + (index % 8),
                "slurm_job_id": str(730001 + (index % 8)),
                "allocation_node_name": f"n{41 + index % 8:03d}",
                "created_at": "2026-07-12 06:49:00",
                "attached_at": "2026-07-12 08:31:00",
                "launch_started_at": "2026-07-12 08:31:01",
                "started_at": "2026-07-12 08:31:02",
                "finished_at": f"2026-07-12 11:12:{12 + index % 6:02d}",
                "exit_code": None,
                "failure_message": "",
            }

        remote = [
            remote_row(index, task_id)
            for index, task_id in enumerate(remote_ids)
        ]
        remote_identities = [{
            field: item.get(field)
            for field in controller._REMOTE_STEP_CANCELLATION_IDENTITY_FIELDS
        } for item in remote]
        remote_identity_sha256 = controller._sha(remote_identities)

        unrelated_id = 103
        unrelated_name = f"{controller.PREFIX}99999"
        unrelated = {
            "id": unrelated_id,
            "name": unrelated_name,
            "dedupe_key": (
                f"mft-al:{unrelated_name}:{controller.SOLVER}:"
                f"{controller.LIBRARY}:unrelated"
            ),
            "project": controller.scheduler_client.MFT_PROJECT,
            "status": "cancelled",
            "account_name": "other-account",
            "created_at": "2026-07-12 06:52:30",
            "attached_at": "2026-07-12 06:53:00",
            "launch_started_at": "2026-07-12 06:52:59",
            "started_at": "2026-07-12 06:53:00",
            "finished_at": "2026-07-12 07:00:00",
            "allocation_id": 9001,
        }
        unrelated_outcome = self._production_outcome(
            unrelated_id, state="invalid", status="cancelled",
            reason="task_cancelled")
        unrelated_outcome["name"] = unrelated_name
        unrelated_outcome["terminal_at"] = "2026-07-12T07:00:00+00:00"
        state = {
            "task_outcomes": {
                "101": {"poison": "external-101"},
                "102": {"poison": "external-102"},
                **{
                    str(task_id): {"poison": f"remote-{task_id}"}
                    for task_id in remote_ids
                },
                str(unrelated_id): copy.deepcopy(unrelated_outcome),
            },
        }
        recovery = {
            "tasks": [], "active": 0, "completed_valid": 0,
            "reasons": [], "wait_reasons": [],
        }
        real_inspect = controller._inspect_production_with_retry

        with mock.patch.object(
                controller, "EXTERNAL_STALE_PIN_CANCELLED_IDS", external_ids), \
                mock.patch.object(
                    controller,
                    "EXTERNAL_STALE_PIN_CANCELLATION_IDENTITY_SHA256",
                    identity_sha256), \
                mock.patch.object(
                    controller, "REMOTE_STEP_CANCELLED_IDS", remote_ids), \
                mock.patch.object(
                    controller,
                    "REMOTE_STEP_CANCELLATION_IDENTITY_SHA256",
                    remote_identity_sha256), \
                mock.patch.object(
                    controller, "_rejected_submission_evidence",
                    return_value={"task_id": controller.REJECTED_TASK_ID}), \
                mock.patch.object(
                    feeder, "campaign_inventory",
                    return_value=[*external, *remote, unrelated]), \
                mock.patch.object(
                    controller.rolling_recycle,
                    "authorized_cancelled_task_ids", return_value=set()), \
                mock.patch.object(
                    controller, "_inspect_production_with_retry",
                    wraps=real_inspect) as inspect_production, \
                mock.patch.object(
                    controller, "_recovery_live_evidence",
                    return_value=recovery), \
                mock.patch.object(
                    controller, "_strict_snapshot",
                    return_value={"pinned": True, "rows": 0}):
            evidence = controller._evidence(state, self.bundle)

        self.assertEqual(inspect_production.call_args.args[0], [unrelated])
        self.assertEqual(
            inspect_production.call_args.kwargs["cached_outcomes"],
            {str(unrelated_id): unrelated_outcome})
        self.assertEqual(
            evidence["external_stale_pin_cancellation"]["task_ids"],
            list(external_ids))
        self.assertEqual(
            evidence["remote_step_cancellation"]["task_ids"],
            list(remote_ids))
        self.assertEqual(evidence["production_lifetime"], {
            "terminal": 1, "valid": 0, "valid_rate": 0.0,
        })
        self.assertEqual(evidence["production_health_cohort"], {
            "cutoff_started_at": controller.PRODUCTION_HEALTH_COHORT_CUTOFF,
            "terminal": 1, "valid": 0, "valid_rate": 0.0,
        })
        self.assertEqual(set(evidence["task_outcomes"]), {str(unrelated_id)})
        surviving_outcome = evidence["task_outcomes"][str(unrelated_id)]
        self.assertEqual(surviving_outcome["state"], "invalid")
        self.assertEqual(surviving_outcome["reason"], "task_cancelled")
        self.assertEqual(
            surviving_outcome["error_message"], "status=cancelled")
        self.assertEqual(evidence["action"], "refill_300")
        self.assertFalse(evidence["paused"])

    def test_production_inspection_retries_transient_429_then_succeeds(self):
        expected = {"active": 1, "outcomes": []}
        unavailable = RuntimeError(
            "terminal result is unavailable: HTTP 429 Too Many Requests")
        sleeper = mock.Mock()
        with mock.patch.object(
                controller.rapid_campaign, "inspect_production_tasks",
                side_effect=[unavailable, unavailable, expected]) as inspect_production:
            actual = controller._inspect_production_with_retry(
                [{"id": 1}], controller.SOLVER, controller.LIBRARY,
                cached_outcomes={"1": {"state": "valid"}}, sleeper=sleeper)
        self.assertIs(actual, expected)
        self.assertEqual(inspect_production.call_count, 3)
        self.assertEqual(sleeper.call_args_list, [mock.call(5.0), mock.call(5.0)])

    def test_production_inspection_raises_non_429_without_retry(self):
        unavailable = RuntimeError(
            "terminal result is unavailable: HTTP 503 Service Unavailable")
        sleeper = mock.Mock()
        with mock.patch.object(
                controller.rapid_campaign, "inspect_production_tasks",
                side_effect=unavailable) as inspect_production, \
                self.assertRaisesRegex(RuntimeError, "HTTP 503"):
            controller._inspect_production_with_retry(
                [{"id": 1}], controller.SOLVER, controller.LIBRARY,
                cached_outcomes={}, sleeper=sleeper)
        inspect_production.assert_called_once()
        sleeper.assert_not_called()

    def test_rejected_same_name_is_excluded_from_inventory_and_cached_health(self):
        rejected = {
            "id": controller.REJECTED_TASK_ID,
            "name": controller.REJECTED_TASK_NAME,
            "status": "cancelled",
        }
        accepted = {
            "id": controller.REJECTED_TASK_ID + 1,
            "name": controller.REJECTED_TASK_NAME,
            "status": "running",
        }
        seen = {}

        def inspect_production(tasks, _solver, _library, cached_outcomes=None):
            seen["tasks"] = tasks
            seen["cache"] = cached_outcomes
            return {"tasks": tasks, "active": 1, "outcomes": [], "cache": cached_outcomes}

        recovery = {
            "tasks": [], "active": 4, "completed_valid": 0,
            "reasons": [], "wait_reasons": [],
        }
        poisoned = {
            "task_outcomes": {str(controller.REJECTED_TASK_ID): {"poison": True}},
            "pause_reasons": ["old_transient_reason"],
        }
        with mock.patch.object(
                controller, "_rejected_submission_evidence",
                return_value={"task_id": controller.REJECTED_TASK_ID}), \
                mock.patch.object(
                    feeder, "campaign_inventory", return_value=[rejected, accepted]), \
                mock.patch.object(
                    controller.rapid_campaign, "inspect_production_tasks",
                    side_effect=inspect_production), \
                mock.patch.object(
                    controller, "_recovery_live_evidence", return_value=recovery), \
                mock.patch.object(
                    controller, "_strict_snapshot",
                    return_value={"pinned": False, "rows": 0}):
            evidence = controller._evidence(poisoned, self.bundle)
        self.assertEqual(seen["tasks"], [accepted])
        self.assertEqual(seen["cache"], {})
        self.assertEqual(evidence["action"], "refill_300")
        self.assertFalse(evidence["paused"])
        self.assertEqual(evidence["pause_reasons"], [])

    def test_only_live_authorized_rolling_cancellation_is_excluded_from_health(self):
        cancelled = {"id": 42, "status": "cancelled"}
        accepted = {"id": 43, "status": "running"}
        seen = {}

        def inspect_production(tasks, _solver, _library, cached_outcomes=None):
            seen["tasks"] = tasks
            seen["cache"] = cached_outcomes
            return {
                "tasks": tasks, "active": 1, "outcomes": [],
                "cache": cached_outcomes,
            }

        recovery = {
            "tasks": [], "active": 4, "completed_valid": 0,
            "reasons": [], "wait_reasons": [],
        }
        with mock.patch.object(
                controller, "_rejected_submission_evidence",
                return_value={"task_id": controller.REJECTED_TASK_ID}), \
                mock.patch.object(
                    feeder, "campaign_inventory",
                    return_value=[cancelled, accepted]), \
                mock.patch.object(
                    controller.rolling_recycle,
                    "authorized_cancelled_task_ids", return_value={42}), \
                mock.patch.object(
                    controller.rapid_campaign, "inspect_production_tasks",
                    side_effect=inspect_production), \
                mock.patch.object(
                    controller, "_recovery_live_evidence", return_value=recovery), \
                mock.patch.object(
                    controller, "_strict_snapshot",
                    return_value={"pinned": False, "rows": 0}):
            evidence = controller._evidence(
                {"task_outcomes": {"42": {"poison": True}, "43": {"ok": True}}},
                self.bundle)
        self.assertEqual(seen["tasks"], [accepted])
        self.assertEqual(seen["cache"], {"43": {"ok": True}})
        self.assertEqual(evidence["rolling_recycle_cancelled_exclusions"], [42])

    def test_collector_pin_lag_alerts_without_stopping_mature_refill(self):
        recovery = {
            "tasks": [], "active": 4, "completed_valid": 0,
            "reasons": [], "wait_reasons": [],
        }

        def outcome(task_id):
            return {
                "task_id": task_id,
                "name": f"mft-camp-sb171c7c-le6b9b9d-{task_id}",
                "status": "completed",
                "state": "valid",
                "reason": None,
                "error_fingerprint": None,
                "error_message": None,
                "terminal_at": None,
                "saturation_columns": [],
            }

        def evidence_for(count):
            outcomes = [outcome(index + 1) for index in range(count)]
            production = {
                "tasks": [
                    {"id": row["task_id"], "started_at": "2026-07-12T06:53:00Z"}
                    for row in outcomes
                ],
                "active": 0, "outcomes": outcomes,
                "cache": {str(row["task_id"]): row for row in outcomes},
            }
            with mock.patch.object(
                    controller, "_rejected_submission_evidence",
                    return_value={"task_id": controller.REJECTED_TASK_ID}), \
                    mock.patch.object(feeder, "campaign_inventory", return_value=[]), \
                    mock.patch.object(
                        controller.rapid_campaign, "inspect_production_tasks",
                        return_value=production), \
                    mock.patch.object(
                        controller, "_recovery_live_evidence", return_value=recovery), \
                    mock.patch.object(
                        controller, "_strict_snapshot",
                        return_value={"pinned": False, "rows": 0}):
                return controller._evidence({"task_outcomes": {}}, self.bundle)

        self.assertEqual(evidence_for(19)["action"], "refill_300")
        checkpoint = evidence_for(20)
        self.assertEqual(checkpoint["action"], "refill_300")
        self.assertFalse(checkpoint["paused"])
        self.assertEqual(checkpoint["wait_reasons"], [])
        self.assertIn(
            "strict_collector_not_pinned_to_b171",
            checkpoint["production_nonblocking_alerts"])

    def test_pre_cutoff_three_identical_errors_are_lifetime_nonblocking(self):
        message = (
            "stderr_pyaedt=[thermal] solve rejected before extraction: "
            "analyze-call-ok=True, converged=0, reason=monitor_missing")
        outcomes = [
            self._production_outcome(
                task_id, state="invalid", fingerprint="same-monitor-error",
                message=message)
            for task_id in range(1, 4)
        ]
        evidence = self._production_evidence(
            outcomes, {task_id: "2026-07-12T06:52:06Z" for task_id in range(1, 4)})

        self.assertEqual(evidence["action"], "refill_300")
        self.assertEqual(evidence["pause_reasons"], [])
        self.assertEqual(evidence["production_lifetime"]["terminal"], 3)
        self.assertEqual(evidence["production_health_cohort"]["terminal"], 0)

    def test_post_cutoff_three_identical_errors_alert_without_stopping_refill(self):
        message = (
            "stderr_pyaedt=[thermal] solve rejected before extraction: "
            "analyze-call-ok=True, converged=0, reason=monitor_missing")
        outcomes = [
            self._production_outcome(
                task_id, state="invalid", fingerprint="same-monitor-error",
                message=message)
            for task_id in range(1, 4)
        ]
        evidence = self._production_evidence(
            outcomes, {task_id: "2026-07-12T06:52:07Z" for task_id in range(1, 4)})

        self.assertEqual(evidence["action"], "refill_300")
        self.assertFalse(evidence["paused"])
        self.assertIn(
            "repeated_runtime_error:same-monitor-error:3",
            evidence["production_nonblocking_alerts"])

    def test_four_sparse_field_summary_failures_in_795_do_not_pause_refill(self):
        message = (
            "stderr_pyaedt=[thermal] validation failed: "
            "field-summary-data=True, required-missing=1, missing-total=2, "
            "analyze-call-ok=True")
        fingerprint = "23825321daac37fe"
        outcomes = [
            self._production_outcome(task_id)
            for task_id in range(1, 796)
        ]
        for task_id in (100, 300, 500, 781):
            outcomes[task_id - 1] = self._production_outcome(
                task_id, state="invalid", fingerprint=fingerprint,
                message=message)
        evidence = self._production_evidence(
            outcomes,
            {row["task_id"]: "2026-07-12T06:53:00Z" for row in outcomes},
        )

        self.assertEqual(evidence["action"], "refill_300")
        self.assertFalse(evidence["paused"])
        self.assertFalse(any(
            reason.startswith(f"repeated_runtime_error:{fingerprint}:")
            for reason in evidence["pause_reasons"]
        ))
        self.assertEqual(
            evidence["production_health_cohort"]["terminal"], 795)
        self.assertEqual(
            evidence["production_health_cohort"]["valid"], 791)

    def test_fleet_rate_uses_twenty_post_cutoff_terminals(self):
        pre = [self._production_outcome(task_id) for task_id in range(1, 81)]
        post = [self._production_outcome(task_id) for task_id in range(81, 98)]
        post.extend(
            self._production_outcome(task_id, state="invalid")
            for task_id in range(98, 101)
        )
        outcomes = pre + post
        started = {
            row["task_id"]: (
                "2026-07-12T06:52:06Z" if row["task_id"] <= 80
                else "2026-07-12T06:52:08Z")
            for row in outcomes
        }
        evidence = self._production_evidence(outcomes, started)

        self.assertEqual(evidence["production_lifetime"], {
            "terminal": 100, "valid": 97, "valid_rate": 0.97,
        })
        self.assertEqual(evidence["production_health_cohort"], {
            "cutoff_started_at": controller.PRODUCTION_HEALTH_COHORT_CUTOFF,
            "terminal": 20, "valid": 17, "valid_rate": 0.85,
        })
        self.assertEqual(evidence["action"], "refill_300")
        self.assertFalse(evidence["paused"])
        self.assertIn(
            "fleet_valid_rate_below_90pct:0.850",
            evidence["production_nonblocking_alerts"])

    def test_lifetime_statistics_remain_append_only_across_health_cutoff(self):
        outcomes = [
            self._production_outcome(1),
            self._production_outcome(2, state="invalid"),
            self._production_outcome(3),
        ]
        evidence = self._production_evidence(outcomes, {
            1: "2026-07-12T06:51:00Z",
            2: "2026-07-12T06:51:30Z",
            3: "2026-07-12T06:53:00Z",
        })

        self.assertEqual(evidence["production_terminal_b171"], 3)
        self.assertEqual(evidence["production_valid_b171"], 2)
        self.assertEqual(evidence["production_lifetime"]["terminal"], 3)
        self.assertEqual(evidence["production_health_cohort"]["terminal"], 1)

    def test_terminal_started_at_missing_or_malformed_fails_closed(self):
        outcome = self._production_outcome(1)
        for value in (None, "not-a-time", "2026-07-12"):
            with self.subTest(value=value), self.assertRaisesRegex(
                    RuntimeError, "started_at"):
                self._production_evidence([outcome], {1: value})

    def test_post_cutoff_saturation_and_revision_gates_alert_without_stopping(self):
        outcomes = [
            self._production_outcome(1, state="invalid", status="completed",
                                     saturation=("T_max_core",)),
            self._production_outcome(2, state="invalid", status="completed",
                                     reason="solver_revision_mismatch"),
        ]
        evidence = self._production_evidence(
            outcomes, {1: "2026-07-12T06:53:00Z", 2: "2026-07-12T06:53:01Z"})

        self.assertEqual(evidence["action"], "refill_300")
        self.assertFalse(evidence["paused"])
        self.assertTrue(any(
            reason.startswith("thermal_saturation_detected:")
            for reason in evidence["production_nonblocking_alerts"]))
        self.assertTrue(any(
            reason.startswith("revision_mismatch_detected:")
            for reason in evidence["production_nonblocking_alerts"]))

    def test_recovery_failures_and_unreviewed_gate_are_nonblocking_alerts(self):
        recovery = {
            "tasks": [{"task_id": 28077, "state": "invalid"}],
            "active": 0, "completed_valid": 0,
            "reasons": ["recovery4_terminal_timeout:28077:exit124"],
            "wait_reasons": ["recovery4_terminal_gate_not_root_reviewed"],
        }
        evidence = self._production_evidence([], {}, recovery=recovery)

        self.assertEqual(evidence["action"], "refill_300")
        self.assertEqual(evidence["pause_reasons"], [])
        self.assertEqual(evidence["recovery4_nonblocking_alerts"], [
            "recovery4_terminal_gate_not_root_reviewed",
            "recovery4_terminal_timeout:28077:exit124",
        ])

    def test_recovery_active_failure_and_terminal_gate_states(self):
        submitted = self.bundle["recovery_submission"]["tasks"]
        by_id = {row["task_id"]: row for row in submitted}

        def active_detail(task_id):
            return self._recovery_row(by_id[task_id])

        with mock.patch.object(
                controller.production, "_task_detail", side_effect=active_detail), \
                mock.patch.object(
                    controller.scheduler_client, "fetch_result",
                    side_effect=AssertionError("active recovery fetched stdout")):
            active = controller._recovery_live_evidence(self.bundle)
        self.assertEqual(active["active"], 4)
        self.assertEqual(active["completed_valid"], 0)
        self.assertEqual(active["reasons"], [])
        self.assertEqual(active["wait_reasons"], [])

        completed_id = submitted[-1]["task_id"]

        def mixed_detail(task_id):
            status = "completed" if task_id == completed_id else "running"
            return self._recovery_row(by_id[task_id], status=status)

        with mock.patch.object(
                controller.production, "_task_detail", side_effect=mixed_detail), \
                mock.patch.object(
                    controller.scheduler_client, "fetch_result",
                    side_effect=AssertionError("partial recovery fetched stdout")):
            mixed = controller._recovery_live_evidence(self.bundle)
        self.assertEqual(mixed["active"], 3)
        self.assertEqual(mixed["completed_valid"], 0)
        self.assertEqual(mixed["tasks"][-1]["state"], "terminal_unreviewed")
        self.assertEqual(mixed["wait_reasons"], [])

        def failed_detail(task_id):
            status = "failed" if task_id == submitted[0]["task_id"] else "running"
            row = self._recovery_row(by_id[task_id], status=status)
            if status == "failed":
                row["exit_code"] = 124
            return row

        with mock.patch.object(
                controller.production, "_task_detail", side_effect=failed_detail):
            failed = controller._recovery_live_evidence(self.bundle)
        self.assertEqual(
            failed["reasons"],
            [f"recovery4_terminal_timeout:{submitted[0]['task_id']}:exit124"])
        self.assertEqual(failed["tasks"][0]["failure_class"], "timeout")

        def complete_detail(task_id):
            return self._recovery_row(by_id[task_id], status="completed")

        with mock.patch.object(
                controller.production, "_task_detail", side_effect=complete_detail), \
                mock.patch.object(
                    controller.scheduler_client, "fetch_result",
                    side_effect=AssertionError("unreviewed recovery fetched stdout")), \
                mock.patch.object(controller.production, "_load_gate") as load_gate:
            waiting = controller._recovery_live_evidence(self.bundle)
        load_gate.assert_not_called()
        self.assertEqual(waiting["completed_valid"], 0)
        self.assertTrue(all(
            row["state"] == "terminal_unreviewed" for row in waiting["tasks"]))
        self.assertEqual(
            waiting["wait_reasons"],
            ["recovery4_terminal_gate_not_root_reviewed"])

        gate = self._recovery_gate(submitted)
        with mock.patch.object(
                controller.production, "_task_detail", side_effect=complete_detail), \
                mock.patch.object(
                    controller.scheduler_client, "fetch_result",
                    side_effect=AssertionError("reviewed recovery fetched stdout")), \
                mock.patch.object(
                    controller.production, "_load_gate",
                    return_value=gate) as load_gate:
            passed = controller._recovery_live_evidence(
                self.bundle, reviewed_recovery_gate_sha="a" * 64)
        self.assertEqual(passed["wait_reasons"], [])
        self.assertEqual(passed["completed_valid"], 4)
        self.assertEqual(passed["terminal_gate_sha256"], "a" * 64)
        self.assertTrue(all(row["strict_valid"] for row in passed["tasks"]))
        self.assertEqual(
            [row["result_sha256"] for row in passed["tasks"]],
            [f"{index:x}" * 64 for index in range(1, 5)])
        load_gate.assert_called_once_with(
            controller.production.GATE_PATH, "a" * 64,
            self.bundle["recovery_submission"], required=True)

    def test_recovery_gate_is_restart_safe_cache_and_drift_fails_closed(self):
        submitted = self.bundle["recovery_submission"]["tasks"]
        by_id = {row["task_id"]: row for row in submitted}

        def complete_detail(task_id):
            return self._recovery_row(by_id[task_id], status="completed")

        gate = self._recovery_gate(submitted, gate_sha="b" * 64)
        with mock.patch.object(
                controller.production, "_task_detail", side_effect=complete_detail), \
                mock.patch.object(
                    controller.scheduler_client, "fetch_result",
                    side_effect=AssertionError("sealed cache fetched stdout")), \
                mock.patch.object(
                    controller.production, "_load_gate",
                    side_effect=lambda *_args, **_kwargs: copy.deepcopy(gate)) as load_gate:
            first = controller._recovery_live_evidence(
                self.bundle, reviewed_recovery_gate_sha="b" * 64)
            # A fresh call models a controller restart: no mutable recovery
            # verdict is supplied, and the sealed artifact remains sufficient.
            restarted = controller._recovery_live_evidence(
                self.bundle, reviewed_recovery_gate_sha="b" * 64)
        self.assertEqual(first, restarted)
        self.assertEqual(load_gate.call_count, 2)

        drifted = self._recovery_gate(submitted, gate_sha="c" * 64)
        del drifted["tasks"][0]["result_sha256"]
        with mock.patch.object(
                controller.production, "_task_detail", side_effect=complete_detail), \
                mock.patch.object(
                    controller.scheduler_client, "fetch_result",
                    side_effect=AssertionError("drifted gate fetched stdout")), \
                mock.patch.object(
                    controller.production, "_load_gate", return_value=drifted), \
                self.assertRaisesRegex(RuntimeError, "outcome cache row 1 drifted"):
            controller._recovery_live_evidence(
                self.bundle, reviewed_recovery_gate_sha="c" * 64)

        with self.assertRaisesRegex(RuntimeError, "terminal-gate SHA is invalid"):
            controller._recovery_live_evidence(
                self.bundle, reviewed_recovery_gate_sha="not-a-sha")

    def test_legacy_28080_valid_verdict_is_distrusted_without_terminal_gate(self):
        submitted = self.bundle["recovery_submission"]["tasks"]
        by_id = {row["task_id"]: row for row in submitted}
        completed_id = submitted[-1]["task_id"]
        legacy_state = controller._new_state()
        legacy_state["last_evidence"] = {
            "recovery4_completed_valid": 1,
            "recovery4_tasks": [{
                "task_id": completed_id,
                "status": "completed",
                "state": "valid",
            }],
        }

        def mixed_detail(task_id):
            status = "completed" if task_id == completed_id else "running"
            return self._recovery_row(by_id[task_id], status=status)

        production = {"active": 0, "outcomes": [], "cache": {}}
        with mock.patch.object(
                controller, "_rejected_submission_evidence",
                return_value={"task_id": controller.REJECTED_TASK_ID}), \
                mock.patch.object(feeder, "campaign_inventory", return_value=[]), \
                mock.patch.object(
                    controller, "_inspect_production_with_retry",
                    return_value=production), \
                mock.patch.object(
                    controller.production, "_task_detail", side_effect=mixed_detail), \
                mock.patch.object(
                    controller.scheduler_client, "fetch_result",
                    side_effect=AssertionError("legacy verdict fetched stdout")), \
                mock.patch.object(
                    controller, "_strict_snapshot",
                    return_value={"pinned": True, "rows": 0}):
            evidence = controller._evidence(legacy_state, self.bundle)

        self.assertEqual(evidence["recovery4_completed_valid"], 0)
        by_live_id = {
            row["task_id"]: row for row in evidence["recovery4_tasks"]
        }
        self.assertEqual(
            by_live_id[completed_id]["state"], "terminal_unreviewed")

    def test_recovery_strict_forensic_allows_one_exact_startup_retry(self):
        identity = {
            "design": "icepak_thermal",
            "design_type": "Icepak",
            "setups": ["ThermalSetup"],
            "wrapper_setups": ["ThermalSetup"],
        }
        result = {
            "thermal_monitor_file": "fresh.sd",
            "thermal_solve_attempts": 2,
            "thermal_dispatch_forensic_json": json.dumps({
                "schema": "thermal-dispatch-forensic-v1",
                "attempts": [
                    {
                        "attempt": 1,
                        "dispatch_status": "exception",
                        "monitor_reason": "monitor_missing",
                        "monitor_file": "",
                        "native_running": False,
                        "identity": identity,
                    },
                    {
                        "attempt": 2,
                        "dispatch_status": "success",
                        "monitor_reason": "converged",
                        "monitor_file": "fresh.sd",
                        "native_running": False,
                        "identity": identity,
                    },
                ],
                "final_convergence": {
                    "converged": 1,
                    "reason": "converged",
                    "monitor_file": "fresh.sd",
                },
            }),
        }
        with mock.patch.object(
                controller.scheduler_client, "is_valid_result", return_value=True), \
                mock.patch.object(
                    controller.scheduler_client, "result_matches_params",
                    return_value=True), \
                mock.patch.object(
                    controller.rapid_campaign, "thermal_saturation_columns",
                    return_value=[]):
            self.assertTrue(controller._strict_recovery_result(
                result, {"effective_params": {}}))
            malformed = copy.deepcopy(result)
            forensic = json.loads(malformed["thermal_dispatch_forensic_json"])
            forensic["attempts"][1]["identity"]["setups"] = ["AnalyzeAll"]
            malformed["thermal_dispatch_forensic_json"] = json.dumps(forensic)
            self.assertFalse(controller._strict_recovery_result(
                malformed, {"effective_params": {}}))
            wrong_status = copy.deepcopy(result)
            forensic = json.loads(wrong_status["thermal_dispatch_forensic_json"])
            forensic["attempts"][-1]["dispatch_status"] = "exception"
            wrong_status["thermal_dispatch_forensic_json"] = json.dumps(forensic)
            self.assertFalse(controller._strict_recovery_result(
                wrong_status, {"effective_params": {}}))
            wrong_count = copy.deepcopy(result)
            wrong_count["thermal_solve_attempts"] = 1
            self.assertFalse(controller._strict_recovery_result(
                wrong_count, {"effective_params": {}}))

    def test_dynamic_authorization_is_bound_to_exact_runtime_target(self):
        decision = {
            "paused": False,
            "target_active": 250,
            "action": "refill_250",
            "production": {"terminal": 20, "valid": 18, "valid_rate": 0.9},
        }
        kwargs = {
            "max_samples": 12_000,
            "solver_revision": controller.SOLVER,
            "library_revision": controller.LIBRARY,
            "candidate_seed": controller.SEED,
            "local_passed": True,
            "adoption_sha256": controller.PLAN_SHA256,
            "initial_count": 0,
            "cpus": 4,
            "memory_mb": 65_536,
            "timeout_seconds": 14_400,
            "evidence_mode": "dynamic_project_cap_v1",
            "strict_rows": 0,
            "target_strict_rows": 3_000,
        }
        journal = {"events": []}
        with mock.patch.object(
                feeder.scheduler_client, "campaign_mutation_lock_is_held",
                return_value=True):
            authorization = feeder._authorize_adopted_refill(decision, **kwargs)
            with mock.patch.object(feeder, "_step_locked", return_value=True) as step:
                result = feeder._step_from_adopted_controller(
                    12_000, authorization=authorization, target=250, buffer=0,
                    journal=journal, **{
                        key: value for key, value in kwargs.items()
                        if key not in ("max_samples", "local_passed")
                    })
        self.assertTrue(result)
        self.assertEqual(authorization.target, 250)
        self.assertEqual(step.call_args.kwargs["target"], 250)
        self.assertEqual(step.call_args.kwargs["buffer"], 0)
        source = inspect.getsource(controller._execute_new_target_transition)
        self.assertIn("cancel_queued_tasks_cas", source)

    def test_pre_post_and_accepted_commit_journal_boundaries_are_durable(self):
        initial = controller._initial_feeder_state(self.bundle)
        first = self.bundle["plan"]["tasks"][0]
        event = {
            "name": first["name"],
            "dedupe_key": first["dedupe_key"],
            "ledger_committed": False,
        }
        state = copy.deepcopy(initial)
        state["serial"] += 1
        state["candidate_cursor"] = first["candidate_cursor_after"]
        state["candidate_cursors"][state["candidate_generation"]] = first[
            "candidate_cursor_after"]
        state["candidate_raw_index"] = first["candidate_raw_index"]
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(
                    controller, "_load_feeder_state", return_value=initial), \
                mock.patch.object(controller, "_save_feeder_state") as save_state, \
                mock.patch.object(feeder, "submit", return_value=42), \
                mock.patch.object(
                    controller.production, "_task_metadata",
                    return_value={"id": 42}):
            cycle_path = Path(tmp) / "cycle.json"
            cycle_path = Path(tmp) / "cycle-000001.json"
            cycle = self._cycle_payload()
            controller._initialize_cycle(cycle_path, cycle)
            journal = {"events": [event]}
            with controller._feeder_io(
                    self.bundle, cycle_path, cycle, journal):
                _, _, params = feeder.next_valid_candidate(
                    controller.INITIAL_CURSOR, seed=controller.SEED)
                task_id = feeder.submit(
                    first["name"], "wd", params,
                    controller.SOLVER, controller.LIBRARY)
                self.assertEqual(task_id, 42)
                event["task_id"] = 42
                event["accepted_or_reconciled"] = True
                precommit = json.loads(cycle_path.read_text(encoding="utf-8"))
                self.assertEqual(precommit["status"], "accepted_readback_pending_commit")
                feeder.save_state(state)
            committed = json.loads(cycle_path.read_text(encoding="utf-8"))
        self.assertTrue(save_state.called)
        self.assertEqual(committed["status"], "ledger_committed")
        self.assertTrue(committed["formal_journal"]["events"][0]["ledger_committed"])

    def test_immutable_cycle_generations_survive_replace_winerror5(self):
        initial = controller._initial_feeder_state(self.bundle)
        first = self.bundle["plan"]["tasks"][0]
        state = copy.deepcopy(initial)
        state["serial"] += 1
        state["candidate_cursor"] = first["candidate_cursor_after"]
        state["candidate_cursors"][state["candidate_generation"]] = first[
            "candidate_cursor_after"]
        state["candidate_raw_index"] = first["candidate_raw_index"]
        observed_submit_statuses = []

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(controller, "_load_feeder_state", return_value=initial), \
                mock.patch.object(controller, "_save_feeder_state"), \
                mock.patch.object(
                    controller.production, "_task_metadata", return_value={"id": 42}), \
                mock.patch.object(
                    controller.durable.os, "replace",
                    side_effect=PermissionError(5, "RaiDrive denied canonical replace")):
            cycle_path = Path(tmp) / "cycle-000001.json"
            cycle = self._cycle_payload()
            controller._initialize_cycle(cycle_path, cycle)
            self.assertFalse(cycle_path.exists())
            self.assertEqual(
                controller._load_cycle(cycle_path)["status"], "authorized_pending")

            def submit_after_precommit(*_args, **_kwargs):
                observed_submit_statuses.append(
                    controller._load_cycle(cycle_path)["status"])
                return 42

            journal = {"events": [{
                "name": first["name"],
                "dedupe_key": first["dedupe_key"],
                "ledger_committed": False,
            }]}
            with mock.patch.object(feeder, "submit", side_effect=submit_after_precommit):
                with controller._feeder_io(
                        self.bundle, cycle_path, cycle, journal):
                    _, _, params = feeder.next_valid_candidate(
                        controller.INITIAL_CURSOR, seed=controller.SEED)
                    self.assertEqual(feeder.submit(
                        first["name"], "wd", params,
                        controller.SOLVER, controller.LIBRARY), 42)
                    journal["events"][0]["task_id"] = 42
                    journal["events"][0]["accepted_or_reconciled"] = True
                    self.assertEqual(
                        controller._load_cycle(cycle_path)["status"],
                        "accepted_readback_pending_commit")
                    feeder.save_state(state)

            authoritative = controller._load_cycle(cycle_path)
            self.assertEqual(observed_submit_statuses, ["mutation_about_to_submit"])
            self.assertEqual(authoritative["status"], "ledger_committed")
            self.assertTrue(
                authoritative["formal_journal"]["events"][0]["ledger_committed"])
            generations = list(cycle_path.parent.glob(
                f"{cycle_path.name}.gen-*.json"))
            self.assertEqual(len(generations), 4)

    def test_startup_fails_closed_on_interrupted_and_missing_cycles(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(controller, "CYCLE_ROOT", Path(tmp)):
            first_path = controller._cycle_path(1)
            first = self._cycle_payload(1)
            controller._initialize_cycle(first_path, first)
            controller._save_cycle(first_path, first, "completed")
            second_path = controller._cycle_path(2)
            second = self._cycle_payload(2)
            controller._initialize_cycle(second_path, second)
            second["formal_journal"]["events"].append({
                "name": "candidate",
                "dedupe_key": "dedupe",
                "ledger_committed": False,
            })
            controller._save_cycle(second_path, second, "mutation_about_to_submit")
            with self.assertRaisesRegex(
                    RuntimeError, "cycle-000002:mutation_about_to_submit"):
                controller._assert_no_unreconciled_cycles(2)
            second_evidence = self._no_mutation_evidence(2)
            state_path = Path(tmp) / "controller-state.json"
            with mock.patch.object(controller, "STATE_PATH", state_path):
                controller._publish_reconciled_no_mutation(
                    2, second_evidence, second_evidence["evidence_sha256"])
            controller._assert_no_unreconciled_cycles(2)
            with self.assertRaisesRegex(RuntimeError, "cycle-000003:missing"):
                controller._assert_no_unreconciled_cycles(3)

    def test_terminal_cycle_highwater_checks_only_new_suffix_and_ahead_artifacts(self):
        loaded = []

        def load_cycle(path):
            serial = int(path.stem.rsplit("-", 1)[1])
            loaded.append(serial)
            return {"status": "completed"}

        with mock.patch.object(
                controller, "_cycle_serials_on_disk", return_value={1, 2}), \
                mock.patch.object(controller, "_load_cycle", side_effect=load_cycle):
            highwater = controller._assert_no_unreconciled_cycles(
                2, terminal_highwater=1)

        self.assertEqual(highwater, 2)
        self.assertEqual(loaded, [2])

        with mock.patch.object(
                controller, "_cycle_serials_on_disk", return_value={1, 2, 3}), \
                mock.patch.object(
                    controller, "_load_cycle", return_value={"status": "completed"}):
            with self.assertRaisesRegex(
                    RuntimeError, "cycle-000003:ahead_of_controller_state"):
                controller._assert_no_unreconciled_cycles(
                    2, terminal_highwater=2)

        with self.assertRaisesRegex(RuntimeError, "outside controller state"):
            controller._assert_no_unreconciled_cycles(2, terminal_highwater=3)

    def test_legacy_state_implicitly_seals_only_prefix_before_latest_cycle(self):
        legacy = controller._new_state()
        legacy.pop("terminal_cycle_highwater")
        legacy["cycle_serial"] = 317
        legacy["target_active"] = controller.LEGACY_TARGET_ACTIVE
        controller._validate_state(legacy)

        with mock.patch.object(
                controller, "_cycle_serials_on_disk", return_value={1, 316, 317}), \
                mock.patch.object(
                    controller, "_load_cycle", return_value={"status": "completed"}) as load:
            legacy_highwater = max(0, legacy["cycle_serial"] - 1)
            audited = controller._assert_no_unreconciled_cycles(
                legacy["cycle_serial"], legacy_highwater)

        self.assertEqual(audited, 317)
        self.assertEqual(load.call_count, 1)
        self.assertEqual(load.call_args.args[0].name, "cycle-000317.json")

    def test_reviewed_no_mutation_reconciliation_is_sealed_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(controller, "CYCLE_ROOT", Path(tmp)), \
                mock.patch.object(
                    controller, "STATE_PATH", Path(tmp) / "controller-state.json"):
            path = controller._cycle_path(35)
            cycle = self._cycle_payload(35)
            controller._initialize_cycle(path, cycle)
            evidence = self._no_mutation_evidence(35)
            sealed = controller._publish_reconciled_no_mutation(
                35, evidence, evidence["evidence_sha256"])
            self.assertEqual(sealed["status"], "reconciled_no_mutation")
            self.assertEqual(sealed["formal_journal"]["submitted_count"], 0)
            revision = sealed["state_revision"]
            again = controller._publish_reconciled_no_mutation(
                35, evidence, evidence["evidence_sha256"])
            self.assertEqual(again["state_revision"], revision)
            changed = copy.deepcopy(evidence)
            changed["observed_at"] = "2026-07-12T04:01:00+00:00"
            changed.pop("evidence_sha256")
            changed["evidence_sha256"] = controller._sha(changed)
            with self.assertRaisesRegex(RuntimeError, "already terminal"):
                controller._publish_reconciled_no_mutation(
                    35, changed, changed["evidence_sha256"])

    def test_loop_exits_after_first_exception_without_retry(self):
        argv = [
            "--execute",
            "--reviewed-plan-sha", controller.PLAN_SHA256,
            "--authorize-dynamic-project-cap",
            "--loop", "60",
        ]
        with mock.patch.object(
                controller, "run_once", side_effect=RuntimeError("journal failed")) as run, \
                mock.patch.object(controller.time, "sleep") as sleep:
            self.assertEqual(controller.main(argv), 2)
        run.assert_called_once()
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
