import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAMPAIGN_DIR))

import pinned_pilot  # noqa: E402


SOLVER_REVISION = "a" * 40
LIBRARY_REVISION = "b" * 40
SEED = 260710
P02_CANDIDATES = [{"candidate": 0}, {"candidate": 1}]
P08_CANDIDATES = [{"candidate": index} for index in range(2, 10)]


def pilot_capacity(queue_state="ready", project_active=0, queue_reason=None):
    queue_allowed = queue_state != "blocked"
    project_slots = max(
        0, pinned_pilot.PILOT_PROJECT_HARD_CAP - project_active)
    return {
        "headroom": 0 if queue_state != "ready" else project_slots,
        "queue_state": queue_state,
        "queue_reason": queue_reason or queue_state,
        "queue_submission_allowed": queue_allowed,
        "submission_allowed": queue_allowed and project_slots > 0,
        "project": pinned_pilot.MFT_PROJECT,
        "project_max_active_tasks": 300,
        "project_required_hard_cap": pinned_pilot.PILOT_PROJECT_HARD_CAP,
        "project_counts": {
            "queued": project_active, "attaching": 0, "running": 0},
        "project_active": project_active,
        "project_server_open_slots": 300 - project_active,
        "project_stage_open_slots": project_slots,
        "project_submission_slots": project_slots,
    }


def p02_manifest(task_ids=(101, 102), **overrides):
    tag = pinned_pilot.pilot_tag(
        SOLVER_REVISION, LIBRARY_REVISION, "p02", SEED, 0)
    payload = {
        "tag": tag,
        "stage": "p02",
        "solver_revision": SOLVER_REVISION,
        "library_revision": LIBRARY_REVISION,
        "seed": SEED,
        "offset": 0,
        "task_count": 2,
        "executed": True,
        "tasks": [
            {
                "index": index,
                "name": f"mft-pilot-{tag}-{index:02d}",
                "params_sha256": pinned_pilot.candidate_digest(
                    pinned_pilot.effective_candidate(P02_CANDIDATES[index])),
                "task_id": task_id,
            }
            for index, task_id in enumerate(task_ids)
        ],
    }
    payload.update(overrides)
    return payload


def p08_manifest(**overrides):
    tag = pinned_pilot.pilot_tag(
        SOLVER_REVISION, LIBRARY_REVISION, "p08", SEED, 2)
    payload = {
        "tag": tag,
        "stage": "p08",
        "solver_revision": SOLVER_REVISION,
        "library_revision": LIBRARY_REVISION,
        "seed": SEED,
        "offset": 2,
        "task_count": 8,
        "executed": True,
        "tasks": [
            {
                "index": index,
                "name": f"mft-pilot-{tag}-{index:02d}",
                "params_sha256": pinned_pilot.candidate_digest(
                    pinned_pilot.effective_candidate(candidate)),
                "task_id": 200 + index,
            }
            for index, candidate in enumerate(P08_CANDIDATES)
        ],
    }
    payload.update(overrides)
    return payload


class PilotStageContractTests(unittest.TestCase):
    def test_first_production_candidate_starts_after_all_ten_pilots(self):
        pilots = pinned_pilot.deterministic_candidates(10, offset=0, seed=SEED)
        cursor = pinned_pilot.cursor_after_valid_candidates(10, seed=SEED)
        _, _, production = pinned_pilot.next_valid_candidate(cursor, seed=SEED)

        pilot_digests = {pinned_pilot.candidate_digest(candidate) for candidate in pilots}
        self.assertNotIn(pinned_pilot.candidate_digest(production), pilot_digests)

    def test_offsets_are_derived_and_wrong_offsets_are_rejected(self):
        self.assertEqual(pinned_pilot.resolve_stage_contract("p02", 2), 0)
        self.assertEqual(pinned_pilot.resolve_stage_contract("p08", 8), 2)
        with self.assertRaisesRegex(ValueError, "requires offset 0"):
            pinned_pilot.resolve_stage_contract("p02", 2, 2)
        with self.assertRaisesRegex(ValueError, "requires offset 2"):
            pinned_pilot.resolve_stage_contract("p08", 8, 0)
        with self.assertRaisesRegex(ValueError, "requires exactly 8"):
            pinned_pilot.resolve_stage_contract("p08", 2, 2)

    def test_seed_and_offset_are_part_of_pilot_identity(self):
        first = pinned_pilot.pilot_tag(
            SOLVER_REVISION, LIBRARY_REVISION, "p08", SEED, 2)
        second = pinned_pilot.pilot_tag(
            SOLVER_REVISION, LIBRARY_REVISION, "p08", SEED + 1, 2)
        third = pinned_pilot.pilot_tag(
            SOLVER_REVISION, LIBRARY_REVISION, "p08", SEED, 3)

        self.assertIn(f"seed{SEED}", first)
        self.assertIn("-o2", first)
        self.assertEqual(len({first, second, third}), 3)


class PilotPredecessorGateTests(unittest.TestCase):
    def _write_manifest(self, root, payload):
        tag = pinned_pilot.pilot_tag(
            SOLVER_REVISION, LIBRARY_REVISION, "p02", SEED, 0)
        path = Path(root) / f"{tag}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_two_completed_strict_valid_results_open_p08_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_manifest(directory, p02_manifest())
            fetched = [
                pinned_pilot.scheduler_client.ResultFetch(
                    pinned_pilot.scheduler_client.RESULT_VALID,
                    pinned_pilot.effective_candidate(candidate))
                for candidate in P02_CANDIDATES
            ]
            with mock.patch.object(
                    pinned_pilot, "deterministic_candidates",
                    return_value=P02_CANDIDATES), mock.patch.object(
                    pinned_pilot.scheduler_client, "get_status",
                    side_effect=["completed", "completed"]) as status, \
                    mock.patch.object(
                        pinned_pilot.scheduler_client, "fetch_result",
                        side_effect=fetched) as fetch, \
                    mock.patch.object(
                        pinned_pilot.scheduler_client, "is_valid_result",
                        return_value=True) as validate:
                result = pinned_pilot.validate_p02_predecessor(
                    SOLVER_REVISION, LIBRARY_REVISION, SEED,
                    manifest_dir=directory)

        self.assertEqual(result["manifest"], str(path))
        self.assertEqual(result["task_ids"], [101, 102])
        self.assertEqual(status.call_count, 2)
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(validate.call_count, 2)
        for call in fetch.call_args_list:
            self.assertEqual(call.kwargs["expected_revision"], SOLVER_REVISION)
            self.assertEqual(
                call.kwargs["expected_library_revision"], LIBRARY_REVISION)

    def test_noncompleted_or_invalid_result_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_manifest(directory, p02_manifest())
            with mock.patch.object(
                    pinned_pilot, "deterministic_candidates",
                    return_value=P02_CANDIDATES), mock.patch.object(
                    pinned_pilot.scheduler_client, "get_status",
                    return_value="running"), mock.patch.object(
                        pinned_pilot.scheduler_client, "fetch_result") as fetch:
                with self.assertRaisesRegex(RuntimeError, "is not completed"):
                    pinned_pilot.validate_p02_predecessor(
                        SOLVER_REVISION, LIBRARY_REVISION, SEED,
                        manifest_dir=directory)
            fetch.assert_not_called()

            invalid = pinned_pilot.scheduler_client.ResultFetch(
                pinned_pilot.scheduler_client.RESULT_INVALID, {"valid": False})
            with mock.patch.object(
                    pinned_pilot, "deterministic_candidates",
                    return_value=P02_CANDIDATES), mock.patch.object(
                    pinned_pilot.scheduler_client, "get_status",
                    return_value="completed"), mock.patch.object(
                        pinned_pilot.scheduler_client, "fetch_result",
                        return_value=invalid), mock.patch.object(
                        pinned_pilot.scheduler_client, "is_valid_result") as validate:
                with self.assertRaisesRegex(RuntimeError, "no strict-valid result"):
                    pinned_pilot.validate_p02_predecessor(
                        SOLVER_REVISION, LIBRARY_REVISION, SEED,
                        manifest_dir=directory)
            validate.assert_not_called()

    def test_manifest_identity_and_task_ids_fail_closed(self):
        cases = [
            p02_manifest(seed=SEED + 1),
            p02_manifest(executed=False),
            p02_manifest(task_count=8),
            p02_manifest(task_ids=(101, 101)),
        ]
        for payload in cases:
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as directory:
                self._write_manifest(directory, payload)
                with mock.patch.object(
                        pinned_pilot.scheduler_client, "get_status") as status:
                    with self.assertRaises(RuntimeError):
                        pinned_pilot.validate_p02_predecessor(
                            SOLVER_REVISION, LIBRARY_REVISION, SEED,
                            manifest_dir=directory)
                status.assert_not_called()

    def test_missing_or_unreadable_manifest_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "readable p02 manifest"):
                pinned_pilot.validate_p02_predecessor(
                    SOLVER_REVISION, LIBRARY_REVISION, SEED,
                    manifest_dir=directory)


class PilotCompletionGateTests(unittest.TestCase):
    def test_p08_requires_all_eight_matching_candidate_results(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = p08_manifest()
            path = Path(directory) / f"{payload['tag']}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            fetched = [
                pinned_pilot.scheduler_client.ResultFetch(
                    pinned_pilot.scheduler_client.RESULT_VALID,
                    pinned_pilot.effective_candidate(candidate))
                for candidate in P08_CANDIDATES
            ]
            with mock.patch.object(
                    pinned_pilot, "deterministic_candidates",
                    return_value=P08_CANDIDATES), mock.patch.object(
                        pinned_pilot.scheduler_client, "get_status",
                        return_value="completed"), mock.patch.object(
                            pinned_pilot.scheduler_client, "fetch_result",
                            side_effect=fetched), mock.patch.object(
                                pinned_pilot.scheduler_client, "is_valid_result",
                                return_value=True):
                result = pinned_pilot.validate_p08_completion(
                    SOLVER_REVISION, LIBRARY_REVISION, SEED,
                    manifest_dir=directory)

        self.assertEqual(result["task_ids"], list(range(200, 208)))

    def test_p08_rejects_parameter_digest_before_scheduler_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = p08_manifest()
            payload["tasks"][3]["params_sha256"] = "bad"
            path = Path(directory) / f"{payload['tag']}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch.object(
                    pinned_pilot, "deterministic_candidates",
                    return_value=P08_CANDIDATES), mock.patch.object(
                        pinned_pilot.scheduler_client, "get_status") as status:
                with self.assertRaisesRegex(RuntimeError, "parameter digest"):
                    pinned_pilot.validate_p08_completion(
                        SOLVER_REVISION, LIBRARY_REVISION, SEED,
                        manifest_dir=directory)
            status.assert_not_called()


class PilotMainTests(unittest.TestCase):
    def _run_main(self, stage, tasks, seed, execute=False, capacity=None):
        argv = [
            "pinned_pilot.py",
            "--tasks", str(tasks),
            "--stage", stage,
            "--seed", str(seed),
            "--solver-revision", SOLVER_REVISION,
            "--library-revision", LIBRARY_REVISION,
            "--library-root", str(CAMPAIGN_DIR),
        ]
        if execute:
            argv.append("--execute")
        candidates = [{"candidate": index} for index in range(tasks)]
        predecessor = {
            "manifest": "p02.json", "tag": "p02-tag", "task_ids": [101, 102]}
        submit_ids = list(range(1000, 1000 + tasks))
        capacity = capacity or pilot_capacity()
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
                pinned_pilot.al_driver, "_current_solver_revision",
                return_value=SOLVER_REVISION), mock.patch.object(
                    pinned_pilot.al_driver, "_current_library_revision",
                    return_value=LIBRARY_REVISION), mock.patch.object(
                    pinned_pilot, "deterministic_candidates",
                    return_value=candidates) as select, mock.patch.object(
                    pinned_pilot, "capacity_snapshot",
                    return_value=capacity), mock.patch.object(
                     pinned_pilot, "validate_p02_predecessor",
                     return_value=predecessor) as validate, mock.patch.object(
                    pinned_pilot.deployment_gate, "validate_deployment"), mock.patch.object(
                    pinned_pilot.scheduler_client, "reconcile_task_id",
                    return_value=None), mock.patch.object(
                     pinned_pilot.scheduler_client, "submit_verification",
                    side_effect=submit_ids) as submit, mock.patch.object(
                    pinned_pilot, "_atomic_manifest") as install, mock.patch(
                    "builtins.print"):
            pinned_pilot.main()
        return {
            "select": select,
            "validate": validate,
            "submit": submit,
            "install": install,
            "predecessor": predecessor,
        }

    def test_p02_main_uses_offset_zero_and_seed_in_manifest_identity(self):
        calls = self._run_main("p02", 2, SEED, execute=False)

        calls["select"].assert_called_once_with(2, offset=0, seed=SEED)
        calls["validate"].assert_not_called()
        calls["submit"].assert_not_called()
        manifest, path = calls["install"].call_args.args
        self.assertEqual(manifest["stage"], "p02")
        self.assertEqual(manifest["seed"], SEED)
        self.assertEqual(manifest["offset"], 0)
        self.assertEqual(manifest["task_count"], 2)
        self.assertIn(f"p02-seed{SEED}-o0", manifest["tag"])
        self.assertEqual(path.name, f"{manifest['tag']}.preview.json")
        self.assertTrue(all(manifest["tag"] in task["name"] for task in manifest["tasks"]))

    def test_p08_main_gates_then_uses_offset_two_in_names_and_manifest(self):
        calls = self._run_main("p08", 8, SEED + 1, execute=True)

        calls["validate"].assert_called_once_with(
            SOLVER_REVISION, LIBRARY_REVISION, SEED + 1)
        calls["select"].assert_called_once_with(8, offset=2, seed=SEED + 1)
        self.assertEqual(calls["submit"].call_count, 8)
        manifest, path = calls["install"].call_args.args
        self.assertEqual(manifest["predecessor"], calls["predecessor"])
        self.assertEqual(manifest["offset"], 2)
        self.assertEqual(manifest["seed"], SEED + 1)
        self.assertIn(f"p08-seed{SEED + 1}-o2", manifest["tag"])
        self.assertEqual(path.name, f"{manifest['tag']}.json")
        for call in calls["submit"].call_args_list:
            self.assertEqual(
                call.kwargs["priority"], pinned_pilot.TEST_TASK_PRIORITY)
        for index, task in enumerate(manifest["tasks"]):
            self.assertEqual(task["name"], f"mft-pilot-{manifest['tag']}-{index:02d}")
            self.assertEqual(task["task_id"], 1000 + index)

    def test_p08_submits_queued_demand_while_scheduler_is_opening_pools(self):
        calls = self._run_main(
            "p08", 8, SEED + 2, execute=True,
            capacity=pilot_capacity(
                queue_state="opening", queue_reason="opening demand pools"),
        )

        self.assertEqual(calls["submit"].call_count, 8)

    def test_partial_p08_retry_reconciles_existing_ids_and_gates_only_missing(self):
        candidates = [{"candidate": index} for index in range(8)]
        existing = [701, 702, 703, 704, None, None, None, None]
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
                pinned_pilot.al_driver, "_current_solver_revision",
                return_value=SOLVER_REVISION), mock.patch.object(
                    pinned_pilot.al_driver, "_current_library_revision",
                    return_value=LIBRARY_REVISION), mock.patch.object(
                        pinned_pilot, "deterministic_candidates",
                        return_value=candidates), mock.patch.object(
                            pinned_pilot, "validate_p02_predecessor",
                            return_value={"task_ids": [101, 102]}), mock.patch.object(
                                pinned_pilot.deployment_gate,
                                "validate_deployment"), mock.patch.object(
                                    pinned_pilot, "capacity_snapshot",
                                    return_value=pilot_capacity(project_active=4)) as capacity, mock.patch.object(
                                        pinned_pilot.scheduler_client,
                                        "reconcile_task_id",
                                        side_effect=existing) as reconcile, mock.patch.object(
                                            pinned_pilot.scheduler_client,
                                            "submit_verification",
                                            side_effect=[705, 706, 707, 708]) as submit:
            result = pinned_pilot.submit_pilot_stage(
                SOLVER_REVISION,
                LIBRARY_REVISION,
                "p08",
                seed=SEED,
                execute=True,
                manifest_dir=directory,
                library_root=CAMPAIGN_DIR,
            )
            canonical_exists = result["manifest_path"].is_file()
            partial_exists = result["manifest_path"].with_suffix(
                ".partial.json").is_file()

        self.assertEqual(reconcile.call_count, 8)
        self.assertEqual(submit.call_count, 4)
        capacity.assert_called_once_with(
            required_hard_cap=pinned_pilot.PILOT_PROJECT_HARD_CAP)
        manifest = result["manifest"]
        self.assertEqual(
            [record["task_id"] for record in manifest["tasks"]],
            [701, 702, 703, 704, 705, 706, 707, 708],
        )
        self.assertEqual(manifest["missing_task_count"], 4)
        self.assertTrue(canonical_exists)
        self.assertFalse(partial_exists)

    def test_partial_p08_crash_keeps_only_partial_ledger(self):
        candidates = [{"candidate": index} for index in range(8)]
        existing = [701, 702, 703, 704, None, None, None, None]
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
                pinned_pilot.al_driver, "_current_solver_revision",
                return_value=SOLVER_REVISION), mock.patch.object(
                    pinned_pilot.al_driver, "_current_library_revision",
                    return_value=LIBRARY_REVISION), mock.patch.object(
                        pinned_pilot, "deterministic_candidates",
                        return_value=candidates), mock.patch.object(
                            pinned_pilot, "validate_p02_predecessor",
                            return_value={"task_ids": [101, 102]}), mock.patch.object(
                                pinned_pilot.deployment_gate,
                                "validate_deployment"), mock.patch.object(
                                    pinned_pilot, "capacity_snapshot",
                                    return_value=pilot_capacity(project_active=4)), mock.patch.object(
                                        pinned_pilot.scheduler_client,
                                        "reconcile_task_id",
                                        side_effect=existing), mock.patch.object(
                                            pinned_pilot.scheduler_client,
                                            "submit_verification",
                                            side_effect=[705, RuntimeError("crash")]):
            with self.assertRaisesRegex(RuntimeError, "crash"):
                pinned_pilot.submit_pilot_stage(
                    SOLVER_REVISION,
                    LIBRARY_REVISION,
                    "p08",
                    seed=SEED,
                    execute=True,
                    manifest_dir=directory,
                    library_root=CAMPAIGN_DIR,
                )
            tag = pinned_pilot.pilot_tag(
                SOLVER_REVISION, LIBRARY_REVISION, "p08", SEED, 2)
            canonical = Path(directory) / f"{tag}.json"
            partial = Path(directory) / f"{tag}.partial.json"
            self.assertFalse(canonical.exists())
            self.assertTrue(partial.is_file())
            ledger = json.loads(partial.read_text(encoding="utf-8"))
            self.assertEqual(
                [record["task_id"] for record in ledger["tasks"]],
                [701, 702, 703, 704, 705, None, None, None],
            )

    def test_blocked_scheduler_capacity_prevents_pilot_submission(self):
        candidates = [{"candidate": index} for index in range(2)]
        blocked = pilot_capacity(
            queue_state="blocked",
            queue_reason="allocation backoff active for cpu")
        with mock.patch.object(
                pinned_pilot.al_driver, "_current_solver_revision",
                return_value=SOLVER_REVISION), mock.patch.object(
                    pinned_pilot.al_driver, "_current_library_revision",
                    return_value=LIBRARY_REVISION), mock.patch.object(
                        pinned_pilot, "deterministic_candidates",
                        return_value=candidates), mock.patch.object(
                        pinned_pilot, "capacity_snapshot",
                            return_value=blocked), mock.patch.object(
                                pinned_pilot.scheduler_client,
                                "reconcile_task_id", return_value=None), mock.patch.object(
                                pinned_pilot, "validate_local_gate",
                                return_value={}), mock.patch.object(
                                    pinned_pilot.deployment_gate,
                                    "validate_deployment"), mock.patch.object(
                                        pinned_pilot.scheduler_client,
                                        "submit_verification") as submit, mock.patch.object(
                                            pinned_pilot, "_atomic_manifest") as install:
            with self.assertRaisesRegex(
                    RuntimeError, "pilot submission is blocked"):
                pinned_pilot.submit_pilot_stage(
                    SOLVER_REVISION,
                    LIBRARY_REVISION,
                    "p02",
                    seed=SEED,
                    execute=True,
                    library_root=CAMPAIGN_DIR,
                )

        submit.assert_not_called()
        install.assert_not_called()


if __name__ == "__main__":
    unittest.main()
