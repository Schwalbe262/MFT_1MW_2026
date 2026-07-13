import copy
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAMPAIGN_DIR))

import _submit_production300_b171c7c as submitter  # noqa: E402


@unittest.skip(
    "archived one-off _submit_production300_b171c7c.py is retained as "
    "historical evidence and is no longer runnable operationally")
class Production300SubmitterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = submitter.static_audit()

    def _gate(self):
        rows = []
        for index, expected in enumerate(self.bundle["recovery_submission"]["tasks"]):
            row = {
                "ordinal": expected["ordinal"],
                "task_id": expected["task_id"],
                "source_task_id": expected["source_task_id"],
                "name": expected["name"],
                "dedupe_key": expected["dedupe_key"],
                "status": "completed",
                "result_state": "valid",
                "strict_valid": True,
                "result_sha256": f"{index + 1:x}" * 64,
            }
            if index < 3:
                row["thermal_dispatch"] = {
                    "entrypoint": "ThermalSetup",
                    "analyze_all_call_count": 0,
                    "fresh_monitor": True,
                    "startup_retry_count": index % 2,
                }
            else:
                row["known_good_nonregression"] = True
            rows.append(row)
        gate = {
            "schema": submitter.GATE_SCHEMA,
            "created_at": "2026-07-12T00:00:00+00:00",
            "gate_decision": "pass",
            "solver_revision": submitter.SOLVER,
            "library_revision": submitter.LIBRARY,
            "recovery_plan_sha256": submitter.RECOVERY_PLAN_SHA256,
            "recovery_submission_sha256": submitter.RECOVERY_SUBMISSION_SHA256,
            "task_count": 4,
            "strict_valid_count": 4,
            "all_strict_valid": True,
            "partial_pass_allowed": False,
            "tasks": rows,
        }
        gate["gate_sha256"] = submitter._sha(gate)
        return gate

    def test_static_audit_rechecks_exact_real_artifacts(self):
        bundle = self.bundle
        self.assertEqual(bundle["plan"]["plan_sha256"], submitter.PLAN_SHA256)
        self.assertEqual(len(bundle["plan_records"]), 300)
        self.assertEqual(len(bundle["old_records"]), 250)
        self.assertEqual(
            [row["task_id"] for row in bundle["old_records"]],
            list(range(27755, 28005)),
        )
        self.assertEqual(
            [row["task_id"] for row in bundle["recovery_submission"]["tasks"]],
            [28077, 28078, 28079, 28080],
        )
        self.assertEqual(len({row["name"] for row in bundle["plan_records"]}), 300)
        self.assertEqual(len({row["dedupe_key"] for row in bundle["plan_records"]}), 300)

    def test_clean_solver_deployment_root_selects_exact_clean_worktree(self):
        with tempfile.TemporaryDirectory() as tmp:
            dirty = Path(tmp) / "dirty"
            clean = Path(tmp) / "clean"
            records = (
                f"worktree {dirty}\nHEAD {submitter.SOLVER}\nbranch refs/heads/main\n\n"
                f"worktree {clean}\nHEAD {submitter.SOLVER}\nbranch refs/heads/fix\n"
            )

            def fake_git(repo, *args):
                repo = Path(repo).resolve()
                if args == ("worktree", "list", "--porcelain"):
                    return records
                if args == ("rev-parse", "HEAD"):
                    return submitter.SOLVER
                if args == ("status", "--porcelain", "--untracked-files=no"):
                    return " M user.ipynb" if repo == dirty.resolve() else ""
                raise AssertionError((repo, args))

            with mock.patch.dict(submitter.os.environ, {}, clear=True), \
                    mock.patch.object(submitter, "_git", side_effect=fake_git):
                selected = submitter._clean_solver_deployment_root()

        self.assertEqual(selected, clean.resolve())

    def test_clean_solver_deployment_root_fails_without_exact_clean_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            dirty = Path(tmp) / "dirty"
            records = (
                f"worktree {dirty}\nHEAD {submitter.SOLVER}\nbranch refs/heads/main\n"
            )

            def fake_git(repo, *args):
                if args == ("worktree", "list", "--porcelain"):
                    return records
                if args == ("rev-parse", "HEAD"):
                    return submitter.SOLVER
                if args == ("status", "--porcelain", "--untracked-files=no"):
                    return " M user.ipynb"
                raise AssertionError((repo, args))

            with mock.patch.dict(submitter.os.environ, {}, clear=True), \
                    mock.patch.object(submitter, "_git", side_effect=fake_git):
                with self.assertRaisesRegex(RuntimeError, "no clean exact-SHA"):
                    submitter._clean_solver_deployment_root()

    def test_default_audit_missing_gate_has_zero_scheduler_activity(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
                submitter, "deployment_audit", return_value={"solver": {}, "library": {}}), \
                mock.patch.object(submitter.requests, "get") as get, \
                mock.patch.object(submitter.requests, "post") as post, \
                mock.patch.object(submitter.scheduler_client, "submit_verification") as submit, \
                mock.patch.object(submitter.scheduler_client, "campaign_mutation_lock") as lock:
            result = submitter.audit(Path(tmp) / "missing-gate.json")

        self.assertEqual(result["mode"], "audit-only")
        self.assertEqual(result["scheduler_query_count"], 0)
        self.assertEqual(result["scheduler_mutation_count"], 0)
        self.assertFalse(result["execution_ready"])
        self.assertTrue(any("terminal_gate_missing" in item for item in result["blockers"]))
        get.assert_not_called()
        post.assert_not_called()
        submit.assert_not_called()
        lock.assert_not_called()

    def test_execute_missing_gate_fails_before_lock_or_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing-terminal-gate.json"
            with mock.patch.object(submitter, "GATE_PATH", missing), \
                    mock.patch.object(submitter, "static_audit", return_value=self.bundle), \
                    mock.patch.object(submitter, "deployment_audit") as deployment, \
                    mock.patch.object(submitter.scheduler_client, "campaign_mutation_lock") as lock, \
                    mock.patch.object(submitter.requests, "get") as get, \
                    mock.patch.object(submitter.requests, "post") as post:
                with self.assertRaisesRegex(RuntimeError, "gate evidence is missing"):
                    submitter.execute(
                        submitter.PLAN_PATH, submitter.PLAN_SHA256,
                        missing, "a" * 64,
                    )
        deployment.assert_not_called()
        lock.assert_not_called()
        get.assert_not_called()
        post.assert_not_called()

    def test_cli_execute_requires_explicit_plan_and_reviewed_seals(self):
        with mock.patch.object(submitter, "execute") as execute, \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                submitter.main(["--execute"])
        execute.assert_not_called()

    def test_terminal_gate_enforces_all_exact_dispatch_facts(self):
        gate = self._gate()
        accepted = submitter._validate_gate(
            gate, gate["gate_sha256"], self.bundle["recovery_submission"],
        )
        self.assertIs(accepted, gate)

        mutations = (
            ("wrong setup", lambda value: value["tasks"][0]["thermal_dispatch"].update(
                entrypoint="AnalyzeAll")),
            ("AnalyzeAll call", lambda value: value["tasks"][1]["thermal_dispatch"].update(
                analyze_all_call_count=1)),
            ("stale monitor", lambda value: value["tasks"][2]["thermal_dispatch"].update(
                fresh_monitor=False)),
            ("too many retries", lambda value: value["tasks"][0]["thermal_dispatch"].update(
                startup_retry_count=2)),
            ("known-good regression", lambda value: value["tasks"][3].update(
                known_good_nonregression=False)),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                drifted = copy.deepcopy(gate)
                drifted.pop("gate_sha256")
                mutate(drifted)
                drifted["gate_sha256"] = submitter._sha(drifted)
                with self.assertRaises(RuntimeError):
                    submitter._validate_gate(
                        drifted, drifted["gate_sha256"],
                        self.bundle["recovery_submission"],
                    )

    def test_live_old_audit_selects_only_exact_active_rows(self):
        expected = [
            {"index": 0, "task_id": 10, "name": "old-0", "dedupe_key": "d0"},
            {"index": 1, "task_id": 11, "name": "old-1", "dedupe_key": "d1"},
            {"index": 2, "task_id": 12, "name": "old-2", "dedupe_key": "d2"},
        ]
        inventory = [
            {"id": 10, "name": "old-0", "dedupe_key": "d0",
             "project": submitter.scheduler_client.MFT_PROJECT, "status": "queued"},
            {"id": 11, "name": "old-1", "dedupe_key": "d1",
             "project": submitter.scheduler_client.MFT_PROJECT, "status": "completed"},
            {"id": 12, "name": "old-2", "dedupe_key": "d2",
             "project": submitter.scheduler_client.MFT_PROJECT, "status": "running"},
            {"id": 999, "name": "other-mft", "dedupe_key": "other",
             "project": submitter.scheduler_client.MFT_PROJECT, "status": "running"},
        ]
        audited = submitter._validate_live_old(inventory, expected)
        self.assertEqual([row["task_id"] for row in audited if row["active"]], [10, 12])
        inventory[0]["dedupe_key"] = "wrong"
        with self.assertRaisesRegex(RuntimeError, "drifted"):
            submitter._validate_live_old(inventory, expected)

    def test_cancel_request_contains_only_prevalidated_active_ids_and_settles(self):
        active = [
            {"task_id": 10, "name": "old-0", "dedupe_key": "d0"},
            {"task_id": 12, "name": "old-2", "dedupe_key": "d2"},
        ]
        response = mock.Mock()
        response.json.return_value = {"cancelled": [10, 12], "count": 2}
        response.raise_for_status.return_value = None

        def detail(task_id):
            source = active[0] if task_id == 10 else active[1]
            return {
                "id": task_id, "name": source["name"],
                "dedupe_key": source["dedupe_key"],
                "project": submitter.scheduler_client.MFT_PROJECT,
                "status": "cancelled",
            }

        with mock.patch.object(submitter.requests, "post", return_value=response) as post, \
                mock.patch.object(submitter, "_task_detail", side_effect=detail):
            result = submitter._cancel_exact_active(active)
        self.assertEqual(result["requested_ids"], [10, 12])
        params = post.call_args.kwargs["params"]
        self.assertEqual(params["task_ids"], "10,12")
        self.assertEqual(params["statuses"], "attaching,queued,running")

    def test_locked_execution_orders_cancel_then_sequential_submit_and_seals(self):
        plan_tasks = [
            {"index": 0, "name": "new-0", "dedupe_key": "new-d0",
             "params_sha256": "p0", "workdir": "w0", "params": {"x": 0}},
            {"index": 1, "name": "new-1", "dedupe_key": "new-d1",
             "params_sha256": "p1", "workdir": "w1", "params": {"x": 1}},
        ]
        bundle = {
            "plan": {"tasks": plan_tasks},
            "profile": {"timeout_seconds": 14400},
            "old_records": [],
            "recovery_submission": {"tasks": []},
        }
        active = [{
            "index": 0, "task_id": 10, "name": "old-0",
            "dedupe_key": "old-d0", "status": "queued", "active": True,
        }]
        gate = {"gate_sha256": "f" * 64}
        events = []

        def cancel(rows):
            events.append(("cancel", [row["task_id"] for row in rows]))
            return {
                "requested_ids": [10], "acknowledgement": {"cancelled": [10]},
                "after": [{"task_id": 10, "status": "cancelled"}],
            }

        def submit(**kwargs):
            events.append(("submit", kwargs["name"]))
            return 100 + len([event for event in events if event[0] == "submit"])

        def metadata(task_id, task):
            events.append(("readback", task["name"]))
            return {"id": task_id, "name": task["name"], "dedupe_key": task["dedupe_key"]}

        with tempfile.TemporaryDirectory() as tmp:
            partial = Path(tmp) / "partial.json"
            final = Path(tmp) / "final.json"
            with mock.patch.object(submitter, "PARTIAL_PATH", partial), \
                    mock.patch.object(submitter, "FINAL_PATH", final), \
                    mock.patch.object(submitter.scheduler_client,
                                      "campaign_mutation_lock_is_held", return_value=True), \
                    mock.patch.object(submitter, "_assert_no_new_duplicates",
                                      return_value={"existing_name_count": 0,
                                                    "existing_dedupe_count": 0}), \
                    mock.patch.object(submitter, "_inventory", return_value=[]), \
                    mock.patch.object(submitter, "_validate_live_old", return_value=active), \
                    mock.patch.object(submitter, "_validate_live_recovery", return_value=[]), \
                    mock.patch.object(submitter, "_capacity",
                                      side_effect=[{"project_submission_slots": 1},
                                                   {"project_submission_slots": 2},
                                                   {"project_submission_slots": 0}]), \
                    mock.patch.object(submitter, "_cancel_exact_active", side_effect=cancel), \
                    mock.patch.object(submitter.scheduler_client,
                                      "submit_verification", side_effect=submit), \
                    mock.patch.object(submitter, "_task_metadata", side_effect=metadata):
                result = submitter._execute_locked(bundle, gate, {"clean": True})
            self.assertFalse(partial.exists())
            saved = json.loads(final.read_text(encoding="utf-8"))

        self.assertEqual(
            events,
            [("cancel", [10]), ("submit", "new-0"), ("readback", "new-0"),
             ("submit", "new-1"), ("readback", "new-1")],
        )
        self.assertEqual(result["execution_state"], "complete")
        self.assertEqual(result["scheduler_mutation_count"], 3)
        unsigned = dict(saved)
        seal = unsigned.pop("submission_sha256")
        self.assertEqual(seal, submitter._sha(unsigned))

    def test_duplicate_namespace_aborts_before_any_mutation(self):
        with mock.patch.object(submitter.scheduler_client,
                               "campaign_mutation_lock_is_held", return_value=True), \
                mock.patch.object(submitter, "_assert_no_existing_output"), \
                mock.patch.object(submitter, "_assert_no_new_duplicates",
                                  side_effect=RuntimeError("namespace is not empty")), \
                mock.patch.object(submitter, "_cancel_exact_active") as cancel, \
                mock.patch.object(submitter.scheduler_client, "submit_verification") as submit:
            with self.assertRaisesRegex(RuntimeError, "namespace is not empty"):
                submitter._execute_locked(
                    {"plan": {"tasks": []}}, {"gate_sha256": "f" * 64}, {},
                )
        cancel.assert_not_called()
        submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
