import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAMPAIGN_DIR))

import _continuous_refill_b171c7c as controller  # noqa: E402


def project_contract(cap):
    return {
        "name": controller.scheduler_client.MFT_PROJECT,
        "max_active_tasks": cap,
        "auto_pull": False,
        "updated_at": f"2026-07-13 00:{cap % 60:02d}:00",
    }


def project_snapshot(cap, *, queued=0, attaching=0, running=0):
    counts = {
        "queued": queued,
        "attaching": attaching,
        "running": running,
    }
    active = sum(counts.values())
    return {
        "project": controller.scheduler_client.MFT_PROJECT,
        "project_max_active_tasks": cap,
        "project_counts": counts,
        "project_active": active,
        "project_tagged_active": active,
        "legacy_active": 0,
    }


def dynamic_state(target=300, transition_serial=1, cycle=337):
    state = controller._new_state()
    state.update({
        "target_active": target,
        "target_policy": controller.TARGET_POLICY_DYNAMIC,
        "target_transition_serial": transition_serial,
        "target_transition_highwater": transition_serial,
        "target_cancelled_tasks": {},
        "dynamic_policy_adopted_cycle": cycle,
        "cycle_serial": cycle,
        "terminal_cycle_highwater": cycle,
    })
    controller._validate_state(state)
    return state


def queued_task(task_id, serial, *, status="queued"):
    name = f"mft-camp-sb171c7c-le6b9b9d-{serial:05d}"
    return {
        "id": task_id,
        "name": name,
        "dedupe_key": (
            f"mft-al:{name}:{controller.SOLVER}:{controller.LIBRARY}:"
            f"{task_id:016x}"),
        "project": controller.scheduler_client.MFT_PROJECT,
        "status": status,
        "created_at": "2026-07-13 00:00:00",
        "attached_at": None,
        "launch_started_at": None,
        "started_at": None,
        "finished_at": None,
        "allocation_id": None,
        "assigned_allocation": None,
        "slurm_job_id": "",
        "allocation_node_name": "",
        "account_name": "",
        "requested_account_name": "",
        "exit_code": None,
        "failure_message": "",
        "cpus": controller.CPUS,
        "memory_mb": controller.MEMORY_MB,
        "timeout_seconds": controller.TIMEOUT_SECONDS,
        "scheduling_profile": "fea_bursty",
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "gpus": 0,
    }


def cancelled(row):
    result = copy.deepcopy(row)
    result.update({
        "status": "cancelled",
        "finished_at": "2026-07-13 00:01:00",
    })
    return result


def attaching(row):
    result = copy.deepcopy(row)
    result.update({
        "status": "attaching",
        "attached_at": "2026-07-13 00:00:30",
        "launch_started_at": "2026-07-13 00:00:30",
        "allocation_id": 9001,
        "assigned_allocation": 9001,
        "slurm_job_id": "733000",
        "allocation_node_name": "n001",
        "account_name": "account",
    })
    return result


def prepared_identity(row, cycle=337):
    return controller._safe_controller_queued_identity(row, cycle)


class DynamicTargetControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cycle_root = self.root / "cycles"
        self.paths = mock.patch.multiple(
            controller,
            CYCLE_ROOT=self.cycle_root,
            STATE_PATH=self.root / "controller-state.json",
            FEEDER_STATE_PATH=self.root / "feeder-state.json",
        )
        self.paths.start()
        self.addCleanup(self.paths.stop)

    def _save_state_patch(self):
        saved = []

        def save(state):
            controller._validate_state(state)
            saved.append(copy.deepcopy(state))

        return saved, mock.patch.object(controller, "_save_state", side_effect=save)

    def _seed_adoption_journal(self):
        fixed = controller._new_state()
        fixed.update({"cycle_serial": 337, "terminal_cycle_highwater": 337})
        transition = controller._new_target_transition(
            fixed, project_contract(300), project_snapshot(300, running=300),
            action="adopt_dynamic_policy", eligible=[], selected=[])
        transition.update({
            "status": "completed",
            "readback": [],
            "readback_identity_sha256": controller._sha([]),
            "settled_at": "2026-07-13T00:00:01+00:00",
        })
        controller._initialize_target_transition(
            controller._target_transition_path(1), transition)

    def test_external_or_running_active_above_target_waits_without_refill(self):
        self.assertEqual(
            controller._maintained_pool_action(
                target_active=250,
                logical_active=271,
                target_reached=False,
                reasons=[],
                wait_reasons=[],
            ),
            "wait_natural_drain_to_250",
        )
        self.assertEqual(
            controller._maintained_pool_action(
                target_active=275,
                logical_active=273,
                target_reached=False,
                reasons=[],
                wait_reasons=[],
            ),
            "refill_275",
        )

    def test_policy_adoption_is_same_target_and_journaled_without_task_mutation(self):
        state = controller._new_state()
        state.update({"cycle_serial": 337, "terminal_cycle_highwater": 337})
        saved, save_patch = self._save_state_patch()
        with save_patch, mock.patch.object(
                controller, "_strict_live_project_contract",
                return_value=project_contract(300)), mock.patch.object(
                controller.scheduler_client, "cancel_queued_tasks_cas") as cancel:
            transition = controller._execute_new_target_transition(
                state, {}, project_contract(300),
                project_snapshot(300, running=300))

        self.assertEqual(transition["status"], "completed")
        self.assertEqual(transition["action"], "adopt_dynamic_policy")
        self.assertEqual(transition["selected_tasks"], [])
        self.assertEqual(state["target_active"], 300)
        self.assertEqual(state["target_policy"], controller.TARGET_POLICY_DYNAMIC)
        self.assertEqual(state["target_transition_serial"], 1)
        self.assertTrue(saved)
        cancel.assert_not_called()

    def test_decrease_cancels_only_proven_controller_queued_exact_excess(self):
        state = dynamic_state()
        owned = [queued_task(40_000 + index, 19_000 + index)
                 for index in range(5)]
        external = queued_task(50_000, 29_999)
        eligible = [prepared_identity(row) for row in owned]
        inventory_after = [cancelled(row) for row in owned] + [external]
        saved, save_patch = self._save_state_patch()
        with save_patch, mock.patch.object(
                controller, "_load_feeder_state", return_value={}), mock.patch.object(
                controller, "_controller_owned_queued_candidates",
                return_value=eligible), mock.patch.object(
                controller.feeder, "campaign_inventory",
                side_effect=[[*owned, external], inventory_after]), mock.patch.object(
                controller.scheduler_client, "cancel_queued_tasks_cas",
                return_value={
                    "cancelled": sorted(row["id"] for row in owned),
                    "count": 5,
                }) as cancel, mock.patch.object(
                controller, "_strict_live_project_contract",
                return_value=project_contract(295)):
            transition = controller._execute_new_target_transition(
                state, {}, project_contract(295),
                project_snapshot(295, queued=5, running=295))

        self.assertEqual(transition["status"], "completed")
        self.assertEqual(transition["cancelled_ids"],
                         sorted(row["id"] for row in owned))
        self.assertNotIn(external["id"], transition["cancelled_ids"])
        cancel.assert_called_once_with([row["id"] for row in eligible])
        self.assertEqual(state["target_active"], 295)
        self.assertEqual(len(state["target_cancelled_tasks"]), 5)
        self.assertTrue(saved)

        evidence = controller._dynamic_target_cancelled_evidence(
            state, inventory_after)
        outcomes = [{
            "task_id": row["id"],
            "name": row["name"],
            "status": "cancelled",
            "state": "invalid",
            "error_fingerprint": None,
        } for row in inventory_after if row["id"] != external["id"]]
        production_state = {
            "tasks": inventory_after,
            "outcomes": outcomes,
            "cache": {str(row["task_id"]): dict(row) for row in outcomes},
        }
        controller._classify_dynamic_target_cancelled_outcomes(
            production_state, evidence)
        health = controller._production_health_cohort(
            production_state,
            authenticated_dynamic_target_cancelled_ids=set(
                evidence["task_ids"]),
        )
        self.assertEqual(health["terminal"], 0)
        self.assertTrue(all(
            row["expected_failure_reason"]
                == "operator_target_reduction_cancelled"
            for row in production_state["outcomes"]))

    def test_attach_race_is_skipped_and_never_cancelled_as_running(self):
        state = dynamic_state()
        first = queued_task(41_001, 19_101)
        raced = queued_task(41_002, 19_102)
        eligible = [prepared_identity(first), prepared_identity(raced)]
        inventory_after = [cancelled(first), attaching(raced)]
        saved, save_patch = self._save_state_patch()
        with save_patch, mock.patch.object(
                controller, "_load_feeder_state", return_value={}), mock.patch.object(
                controller, "_controller_owned_queued_candidates",
                return_value=eligible), mock.patch.object(
                controller.feeder, "campaign_inventory",
                side_effect=[[first, raced], inventory_after]), mock.patch.object(
                controller.scheduler_client, "cancel_queued_tasks_cas",
                return_value={"cancelled": [first["id"]], "count": 1}), mock.patch.object(
                controller, "_strict_live_project_contract",
                return_value=project_contract(298)):
            transition = controller._execute_new_target_transition(
                state, {}, project_contract(298),
                project_snapshot(298, queued=2, running=298))

        self.assertEqual(transition["cancelled_ids"], [first["id"]])
        self.assertEqual(transition["skipped_ids"], [raced["id"]])
        self.assertEqual(set(state["target_cancelled_tasks"]), {str(first["id"])})

    def test_uncertain_cancel_restarts_from_presealed_selection(self):
        self._seed_adoption_journal()
        state = dynamic_state()
        task = queued_task(42_001, 19_201)
        identity = prepared_identity(task)
        saved, save_patch = self._save_state_patch()
        with save_patch, mock.patch.object(
                controller, "_load_feeder_state", return_value={}), mock.patch.object(
                controller, "_controller_owned_queued_candidates",
                return_value=[identity]), mock.patch.object(
                controller.feeder, "campaign_inventory", return_value=[task]), mock.patch.object(
                controller.scheduler_client, "cancel_queued_tasks_cas",
                side_effect=controller.scheduler_client.TaskSubmissionUncertain(
                    "lost response")), self.assertRaises(
                        controller.scheduler_client.TaskSubmissionUncertain):
            controller._execute_new_target_transition(
                state, {}, project_contract(299),
                project_snapshot(299, queued=1, running=299))

        interrupted = controller._load_target_transition(
            controller._target_transition_path(2))
        self.assertEqual(interrupted["status"], "cancelling")
        self.assertEqual(state["target_active"], 300)

        with save_patch, mock.patch.object(
                controller.feeder, "campaign_inventory",
                side_effect=[[task], [cancelled(task)]]), mock.patch.object(
                controller.scheduler_client, "cancel_queued_tasks_cas",
                return_value={"cancelled": [task["id"]], "count": 1}) as retry:
            controller._reconcile_target_transition_suffix(state, 299)

        retry.assert_called_once_with([task["id"]])
        self.assertEqual(state["target_active"], 299)
        self.assertIn(str(task["id"]), state["target_cancelled_tasks"])
        terminal = controller._load_target_transition(
            controller._target_transition_path(2))
        self.assertEqual(terminal["status"], "completed")

    def test_superseded_uncertain_decrease_does_not_retry_queued_cancel(self):
        self._seed_adoption_journal()
        state = dynamic_state()
        task = queued_task(43_001, 19_301)
        identity = prepared_identity(task)
        saved, save_patch = self._save_state_patch()
        with save_patch, mock.patch.object(
                controller, "_load_feeder_state", return_value={}), mock.patch.object(
                controller, "_controller_owned_queued_candidates",
                return_value=[identity]), mock.patch.object(
                controller.feeder, "campaign_inventory", return_value=[task]), mock.patch.object(
                controller.scheduler_client, "cancel_queued_tasks_cas",
                side_effect=controller.scheduler_client.TaskSubmissionUncertain(
                    "lost response")), self.assertRaises(
                        controller.scheduler_client.TaskSubmissionUncertain):
            controller._execute_new_target_transition(
                state, {}, project_contract(299),
                project_snapshot(299, queued=1, running=299))

        with save_patch, mock.patch.object(
                controller.feeder, "campaign_inventory", return_value=[task]), mock.patch.object(
                controller.scheduler_client, "cancel_queued_tasks_cas") as cancel:
            controller._reconcile_target_transition_suffix(state, 300)

        cancel.assert_not_called()
        self.assertEqual(state["target_active"], 300)
        terminal = controller._load_target_transition(
            controller._target_transition_path(2))
        self.assertEqual(terminal["status"], "superseded")
        self.assertEqual(terminal["skipped_ids"], [task["id"]])

    def test_ownership_requires_feeder_and_terminal_cycle_and_ignores_external(self):
        owned = queued_task(44_001, 19_401)
        external = queued_task(44_002, 19_402)
        cycle = {
            "schema_version": 1,
            "cycle_serial": 337,
            "created_at": "2026-07-13T00:00:00+00:00",
            "updated_at": "2026-07-13T00:00:00+00:00",
            "status": "completed",
            "plan_sha256": controller.PLAN_SHA256,
            "target_active": 300,
            "target_policy": controller.TARGET_POLICY_DYNAMIC,
            "project_cap_observed": 300,
            "formal_journal": {"events": [{
                "task_id": owned["id"],
                "name": owned["name"],
                "dedupe_key": owned["dedupe_key"],
                "accepted_or_reconciled": True,
                "ledger_committed": True,
                "scheduler_metadata": {
                    "id": owned["id"],
                    "name": owned["name"],
                    "dedupe_key": owned["dedupe_key"],
                },
            }]},
            "error": None,
        }
        controller._initialize_cycle(controller._cycle_path(337), cycle)
        feeder_state = {
            "outstanding": [owned["id"]],
            "task_expected_rows": {str(owned["id"]): 1},
        }
        candidates = controller._controller_owned_queued_candidates(
            [owned, external], feeder_state, 337)

        self.assertEqual([row["id"] for row in candidates], [owned["id"]])


if __name__ == "__main__":
    unittest.main()
