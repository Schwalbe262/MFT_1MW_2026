import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock


import sys


CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAMPAIGN_DIR))

import rapid_campaign  # noqa: E402


SOLVER_REVISION = "a" * 40
LIBRARY_REVISION = "b" * 40
NOW = datetime(2026, 7, 11, 3, 0, tzinfo=timezone.utc)


def outcome(task_id, state="valid", **overrides):
    payload = {
        "task_id": task_id,
        "name": f"task-{task_id}",
        "status": "completed" if state in ("valid", "invalid") else state,
        "state": state if state in ("valid", "invalid") else "invalid",
        "reason": None if state == "valid" else f"task_{state}",
        "error_fingerprint": None,
        "error_message": None,
        "terminal_at": (NOW + timedelta(seconds=task_id)).isoformat(),
        "saturation_columns": [],
    }
    payload.update(overrides)
    return payload


def pilot_outcome(task_id, state="valid", result=None):
    return {
        "task_id": task_id,
        "status": "completed" if state != "pending" else "running",
        "state": state,
        "reason": None if state == "valid" else "result_invalid",
        "result": result,
    }


def pilots(valid=0, invalid=0, manifests=True):
    states = ["valid"] * valid + ["invalid"] * invalid
    states += ["pending"] * max(0, 10 - len(states))
    return {
        "p02": {
            "exists": manifests,
            "outcomes": [
                pilot_outcome(index + 1, state)
                for index, state in enumerate(states[:2])
            ] if manifests else [],
        },
        "p08": {
            "exists": manifests,
            "outcomes": [
                pilot_outcome(index + 3, state)
                for index, state in enumerate(states[2:])
            ] if manifests else [],
        },
    }


def production(items=(), active=0):
    return {
        "tasks": [], "active": active, "outcomes": list(items), "cache": {}}


class PromotionDecisionTests(unittest.TestCase):
    def test_stage_hard_caps_remain_ten_fifty_and_three_hundred(self):
        self.assertEqual(
            rapid_campaign.STAGE_TARGETS,
            {
                rapid_campaign.STAGE_LOCAL3: 3,
                rapid_campaign.STAGE_PILOT10: 10,
                rapid_campaign.STAGE_FLEET50: 50,
                rapid_campaign.STAGE_PRODUCTION300: 300,
            },
        )
        self.assertEqual(max(rapid_campaign.STAGE_TARGETS.values()), 300)

    def test_five_strict_valid_pilots_promote_early_to_fifty(self):
        state = rapid_campaign.new_state(SOLVER_REVISION, LIBRARY_REVISION)

        decision = rapid_campaign.decide_campaign(
            state, True, pilots(valid=5), production(active=5), 274, now=NOW)

        self.assertFalse(decision["paused"])
        self.assertEqual(decision["stage"], rapid_campaign.STAGE_FLEET50)
        self.assertEqual(decision["target_active"], 50)
        self.assertEqual(decision["action"], "refill_50")
        self.assertEqual(decision["pilot"], {"valid": 5, "invalid": 0, "pending": 5})

    def test_twenty_terminal_at_ninety_percent_promotes_to_three_hundred(self):
        state = rapid_campaign.new_state(SOLVER_REVISION, LIBRARY_REVISION)
        rows = [outcome(index, "valid") for index in range(18)]
        rows += [outcome(100 + index, "invalid") for index in range(2)]

        decision = rapid_campaign.decide_campaign(
            state, True, pilots(valid=10), production(rows), 300, now=NOW)

        self.assertFalse(decision["paused"])
        self.assertEqual(decision["stage"], rapid_campaign.STAGE_PRODUCTION300)
        self.assertEqual(decision["target_active"], 300)
        self.assertEqual(decision["production"]["valid_rate"], 0.9)

    def test_sub_ninety_percent_fleet_gate_latches_pause(self):
        state = rapid_campaign.new_state(SOLVER_REVISION, LIBRARY_REVISION)
        rows = [outcome(index, "valid") for index in range(17)]
        rows += [outcome(100 + index, "invalid") for index in range(3)]

        decision = rapid_campaign.decide_campaign(
            state, True, pilots(valid=10), production(rows), 300, now=NOW)

        self.assertTrue(decision["paused"])
        self.assertEqual(decision["action"], "manual_intervention")
        self.assertTrue(any(
            reason.startswith("fleet20_valid_rate_below_90pct")
            for reason in decision["pause_reasons"]))

    def test_three_matching_runtime_errors_stop_refill(self):
        state = rapid_campaign.new_state(SOLVER_REVISION, LIBRARY_REVISION)
        failed = [
            outcome(
                index, "failed", error_fingerprint="same-fingerprint",
                error_message="AEDT process exited", reason="task_failed")
            for index in range(3)
        ]

        decision = rapid_campaign.decide_campaign(
            state, True, pilots(valid=10), production(failed), 300, now=NOW)

        self.assertTrue(decision["paused"])
        self.assertIn(
            "repeated_runtime_error:same-fingerprint:3",
            decision["pause_reasons"])

    def test_sparse_lifetime_fingerprint_does_not_permanently_stop_refill(self):
        rows = [outcome(index, "valid") for index in range(795)]
        for index in (100, 300, 500, 780):
            rows[index] = outcome(
                index, "failed", error_fingerprint="sparse-fingerprint",
                error_message="AEDT process exited", reason="task_failed")

        reasons = rapid_campaign._production_gate_reasons(production(rows))

        self.assertFalse(any(
            reason.startswith("repeated_runtime_error:sparse-fingerprint:")
            for reason in reasons
        ))

    def test_recent_fingerprint_ratio_stops_refill_without_a_trailing_streak(self):
        rows = []
        for group in range(6):
            base = group * 5
            rows.extend(outcome(base + offset, "valid") for offset in range(4))
            rows.append(outcome(
                base + 4, "failed", error_fingerprint="ratio-fingerprint",
                error_message="AEDT process exited", reason="task_failed"))
        # The latest 30 retain all six failures but end in a success, so this
        # exercises the ratio signal independently of the trailing-streak one.
        rows.append(outcome(999, "valid"))

        reasons = rapid_campaign._production_gate_reasons(production(rows))

        self.assertIn(
            "repeated_runtime_error:ratio-fingerprint:6", reasons)

    def test_three_trailing_matching_errors_stop_below_recent_ratio(self):
        rows = [outcome(index, "valid") for index in range(27)]
        rows.extend(
            outcome(
                100 + index, "failed", error_fingerprint="trailing-fingerprint",
                error_message="AEDT process exited", reason="task_failed")
            for index in range(3)
        )

        reasons = rapid_campaign._production_gate_reasons(production(rows))

        self.assertIn(
            "repeated_runtime_error:trailing-fingerprint:3", reasons)

    def test_repeated_residual_nonconvergence_does_not_stop_healthy_fleet(self):
        message = (
            "stderr_pyaedt=[thermal] solve rejected before extraction: "
            "analyze-call-ok=True, converged=0, reason=residual_threshold"
        )
        failed = [
            outcome(
                index,
                "failed",
                error_fingerprint="same-residual-fingerprint",
                error_message=message,
                reason="task_failed",
            )
            for index in range(4)
        ]

        reasons = rapid_campaign._production_gate_reasons(production(failed))

        self.assertTrue(
            rapid_campaign._is_expected_sample_nonconvergence(message)
        )
        self.assertFalse(any(
            reason.startswith("repeated_runtime_error:") for reason in reasons
        ))

    def test_recent_thirty_below_seventy_percent_stops_refill(self):
        state = rapid_campaign.new_state(SOLVER_REVISION, LIBRARY_REVISION)
        state["target_active"] = 300
        rows = [outcome(index, "valid") for index in range(20)]
        rows += [outcome(100 + index, "invalid") for index in range(10)]

        decision = rapid_campaign.decide_campaign(
            state, True, pilots(valid=10), production(rows), 300, now=NOW)

        self.assertTrue(any(
            reason.startswith("recent_valid_rate_below_70pct")
            for reason in decision["pause_reasons"]))

    def test_prepolicy_timeouts_do_not_displace_current_runtime_health_window(self):
        state = rapid_campaign.new_state(SOLVER_REVISION, LIBRARY_REVISION)
        state["target_active"] = 300
        current = [outcome(index, "valid") for index in range(24)]
        current += [outcome(100 + index, "invalid") for index in range(6)]
        late_prepolicy_timeouts = [
            outcome(
                200 + index,
                "failed",
                status="failed",
                reason="task_failed",
                expected_failure_reason="scheduler_prepolicy_timeout",
            )
            for index in range(7)
        ]

        decision = rapid_campaign.decide_campaign(
            state,
            True,
            pilots(valid=10),
            production(current + late_prepolicy_timeouts),
            300,
            now=NOW,
        )

        self.assertFalse(any(
            reason.startswith("recent_valid_rate_below_70pct")
            for reason in decision["pause_reasons"]))

    def test_postpolicy_timeouts_still_reduce_current_runtime_health_rate(self):
        state = rapid_campaign.new_state(SOLVER_REVISION, LIBRARY_REVISION)
        state["target_active"] = 300
        rows = [outcome(index, "valid") for index in range(20)]
        rows += [
            outcome(
                100 + index,
                "failed",
                status="failed",
                reason="task_failed",
                expected_failure_reason=None,
                error_fingerprint="postpolicy-timeout",
                error_message="task timed out after 14400s",
            )
            for index in range(10)
        ]

        decision = rapid_campaign.decide_campaign(
            state, True, pilots(valid=10), production(rows), 300, now=NOW)

        self.assertTrue(any(
            reason.startswith("recent_valid_rate_below_70pct")
            for reason in decision["pause_reasons"]))

    def test_sealed_operator_cancellations_do_not_displace_runtime_window(self):
        state = rapid_campaign.new_state(SOLVER_REVISION, LIBRARY_REVISION)
        state["target_active"] = 300
        current = [outcome(index, "valid") for index in range(24)]
        current += [outcome(100 + index, "invalid") for index in range(6)]
        operator_cancelled = [
            outcome(
                28_746 + index,
                "cancelled",
                status="cancelled",
                reason="task_cancelled",
                expected_failure_reason=(
                    "operator_cancelled_stale_prepolicy_launch"),
            )
            for index in range(3)
        ]

        decision = rapid_campaign.decide_campaign(
            state, True, pilots(valid=10),
            production(current + operator_cancelled), 300, now=NOW)

        self.assertFalse(any(
            reason.startswith("recent_valid_rate_below_70pct")
            for reason in decision["pause_reasons"]))

    def test_after_300_promotion_uses_recent_health_not_old_fleet_gate(self):
        state = rapid_campaign.new_state(SOLVER_REVISION, LIBRARY_REVISION)
        state["stage"] = rapid_campaign.STAGE_PRODUCTION300
        state["target_active"] = 300
        rows = [outcome(index, "valid") for index in range(24)]
        rows += [outcome(100 + index, "invalid") for index in range(6)]

        decision = rapid_campaign.decide_campaign(
            state, True, pilots(valid=10), production(rows), 300, now=NOW)

        self.assertFalse(decision["paused"])
        self.assertEqual(decision["target_active"], 300)

    def test_pause_is_latched_until_explicit_clear(self):
        state = rapid_campaign.new_state(SOLVER_REVISION, LIBRARY_REVISION)
        state["paused"] = True
        state["pause_reasons"] = ["previous_failure"]

        decision = rapid_campaign.decide_campaign(
            state, True, pilots(valid=10), production(), 300, now=NOW)

        self.assertTrue(decision["paused"])
        self.assertEqual(decision["action"], "manual_intervention")
        self.assertIn("previous_failure", decision["pause_reasons"])

    def test_no_dataset_growth_for_ninety_minutes_after_success_stops_refill(self):
        state = rapid_campaign.new_state(SOLVER_REVISION, LIBRARY_REVISION)
        state["target_active"] = 50
        state["last_dataset_rows"] = 274
        state["last_dataset_growth_at"] = (
            NOW - timedelta(minutes=91)).isoformat()
        state["first_production_valid_at"] = (
            NOW - timedelta(minutes=100)).isoformat()

        decision = rapid_campaign.decide_campaign(
            state, True, pilots(valid=10), production(), 274, now=NOW)

        self.assertTrue(decision["paused"])
        self.assertIn(
            "valid_dataset_growth_stalled_90m", decision["pause_reasons"])


class PhysicalValidityTests(unittest.TestCase):
    def test_icepak_saturation_is_detected_even_when_success_flags_exist(self):
        result = {
            "thermal_solved": 1,
            "result_valid_thermal": 1,
            "T_max_Rx_main": 4726.85,
            "Tprobe_Rx_main_leeward_max": 4726.85,
        }

        saturated = rapid_campaign.thermal_saturation_columns(result)
        reason = rapid_campaign.invalid_result_reason(
            result, SOLVER_REVISION, LIBRARY_REVISION, "invalid")

        self.assertEqual(
            saturated,
            ["T_max_Rx_main", "Tprobe_Rx_main_leeward_max"])
        self.assertTrue(reason.startswith("thermal_saturation:"))

    def test_saturated_completed_production_task_is_never_valid(self):
        task = {
            "id": 41,
            "name": rapid_campaign.production_prefix(
                SOLVER_REVISION, LIBRARY_REVISION) + "00041",
            "status": "completed",
        }
        fetched = rapid_campaign.scheduler_client.ResultFetch(
            rapid_campaign.scheduler_client.RESULT_INVALID,
            {"T_max_Rx_main": 4726.85},
        )
        with mock.patch.object(
                rapid_campaign.scheduler_client, "fetch_result",
                return_value=fetched), mock.patch.object(
                    rapid_campaign.scheduler_client, "is_valid_result") as validate:
            inspected = rapid_campaign.inspect_production_tasks(
                [task], SOLVER_REVISION, LIBRARY_REVISION)

        validate.assert_not_called()
        self.assertEqual(inspected["outcomes"][0]["state"], "invalid")
        self.assertEqual(
            inspected["outcomes"][0]["saturation_columns"],
            ["T_max_Rx_main"])

    def test_provisional_generation_names_do_not_enter_feeder_gate(self):
        task = {
            "name": rapid_campaign.production_prefix(
                SOLVER_REVISION, LIBRARY_REVISION) + "prov-001"
        }
        self.assertFalse(rapid_campaign.is_feeder_task(
            task, SOLVER_REVISION, LIBRARY_REVISION))

    def test_cached_terminal_outcome_avoids_repeated_stdout_fetch(self):
        name = rapid_campaign.production_prefix(
            SOLVER_REVISION, LIBRARY_REVISION) + "00041"
        task = {"id": 41, "name": name, "status": "completed"}
        cached = outcome(41, "valid", name=name)
        with mock.patch.object(
                rapid_campaign.scheduler_client, "fetch_result") as fetch:
            inspected = rapid_campaign.inspect_production_tasks(
                [task], SOLVER_REVISION, LIBRARY_REVISION,
                cached_outcomes={"41": cached})

        fetch.assert_not_called()
        self.assertEqual(inspected["outcomes"], [cached])


class FailureClassificationTests(unittest.TestCase):
    @staticmethod
    def _failed_task(task_id, **overrides):
        task = {
            "id": task_id,
            "name": rapid_campaign.production_prefix(
                SOLVER_REVISION, LIBRARY_REVISION) + f"{task_id:05d}",
            "status": "failed",
            "exit_code": 1,
        }
        task.update(overrides)
        return task

    def test_structured_runtime_error_does_not_fetch_stderr(self):
        fetch = mock.Mock(side_effect=AssertionError("stderr fetch is unnecessary"))

        message = rapid_campaign._failure_message(
            self._failed_task(
                41, error_message="Failed to execute gRPC AEDT command: Analyze"),
            stderr_fetcher=fetch,
        )

        self.assertEqual(
            message,
            "error_message=Failed to execute gRPC AEDT command: Analyze",
        )
        fetch.assert_not_called()
        self.assertIsNotNone(
            rapid_campaign._runtime_error_fingerprint(message))

    def test_stale_info_failure_message_uses_stderr_root_cause(self):
        stderr = """INFO:Global:Boundary Symmetry Symmetry3 has been created.
ERROR:root:run_one_loop failed: [matrix] result extraction failed after 3 attempts
RuntimeError: [matrix] result extraction failed after 3 attempts
"""

        message = rapid_campaign._failure_message(
            self._failed_task(
                42,
                failure_message=(
                    "INFO:Global:Boundary Symmetry Symmetry1 has been created.\n"
                    "INFO:Global:Boundary Symmetry Symmetry2 has been created."
                ),
            ),
            stderr_fetcher=mock.Mock(return_value=stderr),
        )

        self.assertEqual(
            message,
            "stderr_run_one_loop=[matrix] result extraction failed after 3 attempts",
        )

    def test_thermal_rejection_outranks_later_aedt_session_cleanup_error(self):
        stderr = """WARNING:Global:No mesh operation found.
ERROR:root:[thermal] solve rejected before extraction: analyze-call-ok=True, converged=0, reason=monitor_missing
INFO:Global:Project simulation closed correctly
ERROR:Global:A(n) <class 'TypeError'> error occurred while retrieving information for the active AEDT sessions: argument of type 'NoneType' is not iterable
INFO:Global:Desktop has been released and closed.
"""

        message = rapid_campaign._stderr_failure_message(stderr)

        self.assertEqual(
            message,
            "stderr_pyaedt=[thermal] solve rejected before extraction: "
            "analyze-call-ok=True, converged=0, reason=monitor_missing",
        )
        self.assertEqual(
            rapid_campaign._runtime_error_fingerprint(message),
            "d44e42386932a56a",
        )

    def test_structured_cleanup_error_selectively_checks_stderr_for_thermal_root(self):
        cleanup = (
            "A(n) <class 'TypeError'> error occurred while retrieving information "
            "for the active AEDT sessions: argument of type 'NoneType' is not iterable"
        )
        stderr = (
            "ERROR:root:[thermal] solve rejected before extraction: "
            "analyze-call-ok=True, converged=0, reason=monitor_missing\n"
            f"ERROR:Global:{cleanup}\n"
        )
        fetch = mock.Mock(return_value=stderr)

        message = rapid_campaign._failure_message(
            self._failed_task(49, error_message=cleanup), stderr_fetcher=fetch
        )

        fetch.assert_called_once_with(49)
        self.assertEqual(
            message,
            "stderr_pyaedt=[thermal] solve rejected before extraction: "
            "analyze-call-ok=True, converged=0, reason=monitor_missing",
        )
        self.assertEqual(
            rapid_campaign._runtime_error_fingerprint(message),
            "d44e42386932a56a",
        )

    def test_structured_cleanup_error_is_not_a_runtime_fingerprint(self):
        cleanup = (
            "A(n) <class 'TypeError'> error occurred while retrieving information "
            "for the active AEDT sessions: argument of type 'NoneType' is not iterable"
        )

        message = rapid_campaign._failure_message(
            self._failed_task(50, error_message=cleanup),
            stderr_fetcher=mock.Mock(return_value=f"ERROR:Global:{cleanup}\n"),
        )

        self.assertEqual(message, "exit_code=1")
        self.assertFalse(rapid_campaign._is_informative_runtime_message(cleanup))
        self.assertIsNone(rapid_campaign._runtime_error_fingerprint(cleanup))

    def test_cached_cleanup_only_failures_remain_invalid_without_false_pause(self):
        cleanup = (
            "A(n) <class 'TypeError'> error occurred while retrieving information "
            "for the active AEDT sessions: argument of type 'NoneType' is not iterable"
        )
        tasks = [self._failed_task(task_id) for task_id in range(50, 53)]
        cached = {
            str(task["id"]): outcome(
                task["id"],
                "failed",
                name=task["name"],
                status="failed",
                reason="task_failed",
                error_message=f"stderr_pyaedt={cleanup}",
                error_fingerprint=rapid_campaign.error_fingerprint(cleanup),
            )
            for task in tasks
        }
        stderr = (
            "INFO:Global:Project simulation closed correctly\n"
            f"ERROR:Global:{cleanup}\n"
            "INFO:Global:Desktop has been released and closed.\n"
        )

        with mock.patch.object(
                rapid_campaign, "_fetch_task_stderr", return_value=stderr) as fetch:
            inspected = rapid_campaign.inspect_production_tasks(
                tasks,
                SOLVER_REVISION,
                LIBRARY_REVISION,
                cached_outcomes=cached,
            )

        self.assertEqual(fetch.call_count, 3)
        self.assertTrue(all(
            item["state"] == "invalid" and item["reason"] == "task_failed"
            for item in inspected["outcomes"]
        ))
        self.assertTrue(all(
            item["error_message"] == "exit_code=1"
            and item["error_fingerprint"] is None
            for item in inspected["outcomes"]
        ))
        self.assertFalse(any(
            reason.startswith("repeated_runtime_error:")
            for reason in rapid_campaign._production_gate_reasons(inspected)
        ))

    def test_exit_code_only_failure_has_no_repeated_error_fingerprint(self):
        task = self._failed_task(43)
        with mock.patch.object(
                rapid_campaign, "_fetch_task_stderr", return_value=""):
            inspected = rapid_campaign.inspect_production_tasks(
                [task], SOLVER_REVISION, LIBRARY_REVISION)

        result = inspected["outcomes"][0]
        self.assertEqual(result["error_message"], "exit_code=1")
        self.assertIsNone(result["error_fingerprint"])

    def test_prepolicy_four_hour_timeout_is_invalid_but_not_current_runtime_outage(self):
        task = self._failed_task(
            28_460,
            name="mft-camp-sb171c7c-le6b9b9d-17941",
            exit_code=124,
            failure_message="task timed out after 14400s",
            started_at="2026-07-12 09:05:25",
        )

        result = rapid_campaign._refresh_failure_outcome(
            {"status": "failed"}, task,
        )

        self.assertEqual(
            result["expected_failure_reason"], "scheduler_prepolicy_timeout",
        )
        self.assertIsNone(result["error_fingerprint"])

    def test_exact_operator_cancelled_stale_launch_is_classified(self):
        task = self._failed_task(
            28_747,
            name="mft-camp-sb171c7c-le6b9b9d-18222",
            status="cancelled",
            exit_code=None,
            failure_message="",
            project="MFT_1MW_2026v1",
            account_name="r1jae262",
            requested_account_name="",
            allocation_id=8_019,
            slurm_job_id="731354",
            allocation_node_name="n045",
            created_at="2026-07-12 10:59:58",
            attached_at="2026-07-12 11:00:47",
            launch_started_at="2026-07-12 11:00:48",
            started_at=None,
            finished_at="2026-07-12 14:24:21",
        )

        result = rapid_campaign._refresh_failure_outcome(
            {"status": "cancelled"}, task,
        )

        self.assertEqual(
            result["expected_failure_reason"],
            "operator_cancelled_stale_prepolicy_launch",
        )
        self.assertIsNone(result["error_fingerprint"])

    def test_unsealed_cancelled_launch_remains_generic_invalid(self):
        task = self._failed_task(
            29_999,
            status="cancelled",
            exit_code=None,
            failure_message="",
            project="MFT_1MW_2026v1",
            account_name="r1jae262",
            requested_account_name="",
            allocation_id=8_019,
            slurm_job_id="731354",
            allocation_node_name="n045",
            created_at="2026-07-12 10:59:58",
            attached_at="2026-07-12 11:00:47",
            launch_started_at="2026-07-12 11:00:48",
            started_at=None,
            finished_at="2026-07-12 14:24:21",
        )

        result = rapid_campaign._refresh_failure_outcome(
            {"status": "cancelled"}, task,
        )

        self.assertIsNone(result["expected_failure_reason"])
        self.assertIsNone(result["error_fingerprint"])

    def test_postpolicy_four_hour_timeout_remains_stopping_runtime_error(self):
        task = self._failed_task(
            29_999,
            exit_code=124,
            failure_message="task timed out after 14400s",
            started_at="2026-07-12 11:00:55",
        )

        inspected = rapid_campaign.inspect_production_tasks(
            [task], SOLVER_REVISION, LIBRARY_REVISION,
        )

        result = inspected["outcomes"][0]
        self.assertIsNone(result["expected_failure_reason"])
        self.assertEqual(result["error_fingerprint"], "bca1d81ac1b45aca")

    def test_cancelled_task_never_fetches_stderr_or_gets_runtime_fingerprint(self):
        task = self._failed_task(48, status="cancelled", exit_code=None)
        with mock.patch.object(
                rapid_campaign, "_fetch_task_stderr") as fetch:
            inspected = rapid_campaign.inspect_production_tasks(
                [task], SOLVER_REVISION, LIBRARY_REVISION)

        fetch.assert_not_called()
        result = inspected["outcomes"][0]
        self.assertEqual(result["error_message"], "status=cancelled")
        self.assertIsNone(result["error_fingerprint"])

    def test_legacy_cached_exit_only_fingerprint_is_reclassified(self):
        task = self._failed_task(44)
        name = task["name"]
        cached = outcome(
            44,
            "failed",
            name=name,
            status="failed",
            reason="task_failed",
            error_message="exit_code=1",
            error_fingerprint=rapid_campaign.error_fingerprint("exit_code=1"),
        )
        stderr = (
            "ERROR:root:run_one_loop failed: Failed to execute gRPC AEDT "
            "command: Analyze\n"
        )
        with mock.patch.object(
                rapid_campaign, "_fetch_task_stderr", return_value=stderr) as fetch:
            inspected = rapid_campaign.inspect_production_tasks(
                [task], SOLVER_REVISION, LIBRARY_REVISION,
                cached_outcomes={"44": cached},
            )

        refreshed = inspected["outcomes"][0]
        fetch.assert_called_once_with(44)
        self.assertEqual(
            refreshed["error_message"],
            "stderr_run_one_loop=Failed to execute gRPC AEDT command: Analyze",
        )
        self.assertIsNotNone(refreshed["error_fingerprint"])
        self.assertNotEqual(
            refreshed["error_fingerprint"], cached["error_fingerprint"])
        self.assertEqual(inspected["cache"]["44"], refreshed)

    def test_legacy_cached_exit_only_is_safe_when_stderr_unavailable(self):
        rows = [
            outcome(
                task_id,
                "failed",
                status="failed",
                reason="task_failed",
                error_message="exit_code=1",
                error_fingerprint=rapid_campaign.error_fingerprint("exit_code=1"),
            )
            for task_id in range(3)
        ]

        reasons = rapid_campaign._production_gate_reasons(production(rows))

        self.assertFalse(any(
            reason.startswith("repeated_runtime_error:") for reason in reasons
        ))

    def test_three_legacy_cached_prefixed_info_rows_do_not_pause(self):
        message = (
            "failure_message=PyAEDT INFO: Parsing design objects.\n"
            "PyAEDT INFO: Boundary Symmetry Symmetry3 has been created."
        )
        rows = [
            outcome(
                task_id,
                "failed",
                status="failed",
                reason="task_failed",
                error_message=message,
                error_fingerprint=rapid_campaign.error_fingerprint(message),
            )
            for task_id in range(3)
        ]

        reasons = rapid_campaign._production_gate_reasons(production(rows))

        self.assertFalse(
            rapid_campaign._is_informative_runtime_message(message)
        )
        self.assertFalse(any(
            reason.startswith("repeated_runtime_error:") for reason in reasons
        ))

    def test_legacy_cached_prefixed_info_is_refetched_and_reclassified(self):
        task = self._failed_task(49)
        name = task["name"]
        old_message = (
            "failure_message=PyAEDT INFO: Parsing design objects.\n"
            "PyAEDT INFO: Boundary Symmetry Symmetry3 has been created."
        )
        cached = outcome(
            49,
            "failed",
            name=name,
            status="failed",
            reason="task_failed",
            error_message=old_message,
            error_fingerprint=rapid_campaign.error_fingerprint(old_message),
        )
        stderr = (
            "ERROR:root:run_one_loop failed: copied loss preparation failed\n"
        )
        with mock.patch.object(
                rapid_campaign, "_fetch_task_stderr", return_value=stderr) as fetch:
            inspected = rapid_campaign.inspect_production_tasks(
                [task], SOLVER_REVISION, LIBRARY_REVISION,
                cached_outcomes={"49": cached},
            )

        refreshed = inspected["outcomes"][0]
        fetch.assert_called_once_with(49)
        self.assertEqual(
            refreshed["error_message"],
            "stderr_run_one_loop=copied loss preparation failed",
        )
        self.assertIsNotNone(refreshed["error_fingerprint"])
        self.assertNotEqual(
            refreshed["error_fingerprint"], cached["error_fingerprint"]
        )
        self.assertEqual(inspected["cache"]["49"], refreshed)

    def test_legacy_prefixed_preamble_with_exception_remains_informative(self):
        message = (
            "failure_message=PyAEDT INFO: Parsing design objects.\n"
            "RuntimeError: copied loss preparation failed"
        )

        self.assertTrue(
            rapid_campaign._is_informative_runtime_message(message)
        )
        self.assertIsNotNone(
            rapid_campaign._runtime_error_fingerprint(message)
        )

    def test_three_identical_stderr_runtime_errors_still_pause(self):
        tasks = [self._failed_task(task_id) for task_id in range(45, 48)]
        stderr = (
            "ERROR:root:run_one_loop failed: Failed to execute gRPC AEDT "
            "command: Analyze\n"
        )
        with mock.patch.object(
                rapid_campaign, "_fetch_task_stderr", return_value=stderr):
            inspected = rapid_campaign.inspect_production_tasks(
                tasks, SOLVER_REVISION, LIBRARY_REVISION)

        fingerprints = {
            item["error_fingerprint"] for item in inspected["outcomes"]
        }
        self.assertEqual(len(fingerprints), 1)
        fingerprint = fingerprints.pop()
        self.assertIsNotNone(fingerprint)
        self.assertIn(
            f"repeated_runtime_error:{fingerprint}:3",
            rapid_campaign._production_gate_reasons(inspected),
        )


class CandidateAuditTests(unittest.TestCase):
    def test_six_hour_profile_fails_before_candidate_or_submission_work(self):
        with mock.patch.object(
                rapid_campaign.provisional_wave, "_load_profile",
                return_value={"timeout_seconds": 21600}), mock.patch.object(
                    rapid_campaign.provisional_wave, "build_plan") as build:
            with self.assertRaisesRegex(RuntimeError, "at most 7200"):
                rapid_campaign.candidate_supply_audit(
                    SOLVER_REVISION, LIBRARY_REVISION, count=3)
        build.assert_not_called()

    def test_candidate_audit_uses_provisional_plan_and_checks_uniqueness(self):
        records = [
            {
                "index": index,
                "name": f"candidate-{index}",
                "candidate_raw_index": 10 + index,
                "params_sha256": f"digest-{index}",
                "dedupe_key": f"key-{index}",
                "params": {"candidate": index},
                "task_id": None,
            }
            for index in range(3)
        ]
        with mock.patch.object(
                rapid_campaign.provisional_wave, "_load_profile",
                return_value={"timeout_seconds": 7200}), mock.patch.object(
                    rapid_campaign.provisional_wave, "build_plan",
                    return_value=records) as build, mock.patch.object(
                        rapid_campaign.provisional_wave, "new_manifest",
                        return_value={"tasks": records}) as manifest, mock.patch.object(
                            rapid_campaign.provisional_wave, "validate_manifest") as validate:
            audit = rapid_campaign.candidate_supply_audit(
                SOLVER_REVISION, LIBRARY_REVISION, seed=7, count=3)

        build.assert_called_once()
        manifest.assert_called_once()
        validate.assert_called_once()
        self.assertEqual(audit["count"], 3)
        self.assertEqual(audit["first_raw_index"], 10)
        self.assertEqual(audit["last_raw_index"], 12)
        self.assertEqual(len(audit["plan_sha256"]), 64)


class ControllerMutationTests(unittest.TestCase):
    def test_different_seed_controllers_share_one_project_mutation_lock(self):
        active = 0
        max_active = 0
        entered = []
        guard = threading.Lock()
        first_entered = threading.Event()
        release_first = threading.Event()

        def locked_body(*_args, seed, **_kwargs):
            nonlocal active, max_active
            with guard:
                active += 1
                max_active = max(max_active, active)
                entered.append(seed)
            first_entered.set()
            if seed == 1:
                release_first.wait(timeout=5)
            with guard:
                active -= 1
            return {"seed": seed}

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
                rapid_campaign.pinned_pilot,
                "CAMPAIGN_MUTATION_LOCK_PATH",
                Path(directory) / "campaign.lock"), mock.patch.object(
                    rapid_campaign, "_run_once_locked",
                    side_effect=locked_body):
            results = []
            threads = [
                threading.Thread(
                    target=lambda seed=seed: results.append(
                        rapid_campaign.run_once(
                            SOLVER_REVISION, LIBRARY_REVISION,
                            seed=seed, execute=True)))
                for seed in (1, 2)
            ]
            threads[0].start()
            self.assertTrue(first_entered.wait(timeout=2))
            threads[1].start()
            time.sleep(0.1)
            self.assertEqual(entered, [1])
            release_first.set()
            for thread in threads:
                thread.join(timeout=5)
                self.assertFalse(thread.is_alive())

        self.assertEqual(max_active, 1)
        self.assertEqual(entered, [1, 2])
        self.assertEqual({result["seed"] for result in results}, {1, 2})

    def test_legacy_serial_above_12000_still_refills_under_default_ceiling(self):
        state = {
            "serial": 16361,
            "submitted_samples": 16361,
            "outstanding": [],
        }
        counts = {"queued": 0, "attaching": 0, "running": 0}
        allocations = [{
            "state": "active",
            "resource_pool": "cpu",
            "total_cpus": 64,
            "free_cpus": 64,
        }]
        feeder = rapid_campaign.feeder
        with mock.patch.object(feeder, "load_state", return_value=state), \
                mock.patch.object(
                    feeder, "scheduler_snapshot",
                    return_value=(
                        counts, counts, allocations,
                        {"ready_fit_slots": 20, "queue_state": "ready",
                         "queue_reason": "ready",
                         "queue_submission_allowed": True,
                         "submission_allowed": True,
                         "project": rapid_campaign.pinned_pilot.MFT_PROJECT,
                         "project_max_active_tasks": 400,
                         "project_required_hard_cap": 1,
                         "project_counts": counts,
                         "project_active": 0,
                         "project_server_open_slots": 300,
                         "project_stage_open_slots": 1,
                         "project_submission_slots": 1},
                    )), \
                mock.patch.object(
                    feeder, "dataset_collection_snapshot",
                    return_value=(12000, set())), \
                mock.patch.object(feeder, "campaign_inventory", return_value=[]), \
                mock.patch.object(
                    feeder, "cursor_after_valid_candidates", return_value=10), \
                mock.patch.object(
                    feeder, "next_valid_candidate",
                    return_value=(11, 10, {"candidate": 10})), \
                mock.patch.object(feeder, "submit", return_value=901) as submit, \
                mock.patch.object(feeder, "save_state"), \
                mock.patch.object(feeder.time, "sleep"):
            feeder.step(
                rapid_campaign.DEFAULT_MAX_SAMPLES,
                target=1,
                buffer=0,
                solver_revision=SOLVER_REVISION,
                library_revision=LIBRARY_REVISION,
            )

        submit.assert_called_once()
        self.assertEqual(state["serial"], 16362)

    def test_execute_delegates_continuous_refill_to_existing_feeder(self):
        self.assertGreater(rapid_campaign.DEFAULT_MAX_SAMPLES, 12000)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "rapid.json"
            local_path = root / (
                rapid_campaign.pinned_pilot.local_gate_tag(
                    SOLVER_REVISION, LIBRARY_REVISION) + ".json")
            local_path.write_text("{}", encoding="utf-8")
            with mock.patch.object(
                    rapid_campaign, "_validate_pinned_local_revisions",
                    return_value=(SOLVER_REVISION, LIBRARY_REVISION)), mock.patch.object(
                        rapid_campaign.deployment_gate, "validate_deployment"), mock.patch.object(
                        rapid_campaign, "candidate_supply_audit",
                        return_value={"count": 300, "plan_sha256": "x"}), mock.patch.object(
                            rapid_campaign.pinned_pilot, "validate_local_gate"), mock.patch.object(
                                rapid_campaign, "inspect_pilots",
                                return_value=pilots(valid=5)), mock.patch.object(
                                    rapid_campaign.feeder, "campaign_inventory",
                                    return_value=[]), mock.patch.object(
                                        rapid_campaign, "inspect_production_tasks",
                                        return_value=production()), mock.patch.object(
                                            rapid_campaign.feeder,
                                            "dataset_collection_snapshot",
                                            return_value=(274, set())), mock.patch.object(
                                                rapid_campaign.feeder,
                                                "_authorize_rapid_refill",
                                                return_value="sealed") as authorize, mock.patch.object(
                                                    rapid_campaign.feeder,
                                                    "_step_from_rapid_controller") as refill:
                result = rapid_campaign.run_once(
                    SOLVER_REVISION,
                    LIBRARY_REVISION,
                    execute=True,
                    library_root=root,
                    state_path=state_path,
                    manifest_dir=root,
                    now=NOW,
                )

        authorize.assert_called_once()
        refill.assert_called_once_with(
            rapid_campaign.DEFAULT_MAX_SAMPLES,
            authorization="sealed",
            target=50,
            buffer=0,
            solver_revision=SOLVER_REVISION,
            library_revision=LIBRARY_REVISION,
            candidate_seed=rapid_campaign.DEFAULT_SEED,
        )
        self.assertEqual(result["mutation"]["refill_target"], 50)


if __name__ == "__main__":
    unittest.main()
