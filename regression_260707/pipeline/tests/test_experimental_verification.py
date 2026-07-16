import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from regression_260707.pipeline.artifacts import GenerationStore, sha256_file
from regression_260707.pipeline import experimental_verification
from regression_260707.pipeline.scheduler_verification import (
    EXACT_COHORT_RESULT_FLAGS,
    EXACT_COHORT_SCHEMA_VERSION,
    EXPERIMENTAL_SUBMISSION_ACCOUNTS,
    EXPERIMENTAL_SUBMISSION_ENV,
    _canonical_sha256,
    _experimental_execution_contract,
    _guard_pool_evidence,
    _wait_for_exact_cohort_guard,
)


TRAINING_SOLVER = "a" * 40
TRAINING_LIBRARY = "b" * 40
FEA_SOLVER = "c" * 40
FEA_LIBRARY = "d" * 40
QUALITY_SHA = "e" * 64


def _cohort():
    session_id = 700
    session_generation = 1
    solve_generation = 9
    members = []
    for index in range(3):
        task_id = 5001 + index
        lease_id = 8001 + index
        members.append({
            "task_id": task_id,
            "lease_id": lease_id,
            "session_id": session_id,
            "session_generation": session_generation,
            "solve_generation": solve_generation,
            "slot_index": index,
            "name": f"mft-q21b-exact-{index}",
            "dedupe_key": f"q21b:{task_id}:{FEA_SOLVER}:{FEA_LIBRARY}",
            "account_name": ("r1jae262", "dhj02", "r1jae262")[index],
            "project": "MFT_1MW_2026v1",
            "priority": 10,
            "aedt_backend": "pooled",
            "solver_revision": FEA_SOLVER,
            "library_revision": FEA_LIBRARY,
            "lease_request_key": f"q21b:{task_id}",
            "lease_project_name": f"mft-{task_id}-{lease_id}",
        })
    return {
        "schema_version": EXACT_COHORT_SCHEMA_VERSION,
        "cohort_id": "q21b-exact-test",
        "session_id": session_id,
        "session_generation": session_generation,
        "solve_batch_generation": solve_generation,
        "solver_revision": FEA_SOLVER,
        "library_revision": FEA_LIBRARY,
        "members": members,
    }


def _request():
    return {
        "stage": "fine",
        "expected_count": 3,
        "selection_policy": "experimental_pareto_span_3_v1",
        "experimental_active_learning": True,
        "production_eligible": False,
        "quality_gate_passed": False,
        "quality_blockers": {"Llt_phys": ["mape"]},
        "quality_status_sha256": QUALITY_SHA,
        "solver_revision": FEA_SOLVER,
        "library_revision": FEA_LIBRARY,
        "training_solver_revision": TRAINING_SOLVER,
        "training_library_revision": TRAINING_LIBRARY,
        "fea_solver_revision": FEA_SOLVER,
        "fea_library_revision": FEA_LIBRARY,
        "candidates": [
            {"candidate_id": f"candidate-{index}"} for index in range(3)
        ],
    }


def _config(solver_root):
    cohort = _cohort()
    return {
        "adapter": "mft_scheduler_v1",
        "execute": True,
        "library_root": str(solver_root),
        "solver_root": str(solver_root),
        "priority": -10,
        "required_hard_cap": 3,
        "max_project_active_tasks": 600,
        "aedt_backend": "pooled",
        "submission_env": dict(EXPERIMENTAL_SUBMISSION_ENV),
        "accounts": list(EXPERIMENTAL_SUBMISSION_ACCOUNTS),
        "guard_cohort": cohort,
        "guard_cohort_sha256": _canonical_sha256(cohort),
        "guard_timeout_seconds": 3600,
        "experimental_quality_status_sha256": QUALITY_SHA,
    }


class ExperimentalSchedulerPolicyTests(unittest.TestCase):
    def test_execution_contract_is_explicit_and_sealed(self):
        with tempfile.TemporaryDirectory() as directory:
            options = _experimental_execution_contract(
                _request(), _config(Path(directory))
            )
            self.assertEqual(options["required_hard_cap"], 3)
            self.assertEqual(options["priority"], -10)
            weakened = _config(Path(directory))
            weakened["required_hard_cap"] = 4
            with self.assertRaisesRegex(RuntimeError, "safety contract"):
                _experimental_execution_contract(_request(), weakened)

            tampered = _config(Path(directory))
            tampered["guard_cohort"]["members"][0]["task_id"] += 1
            with self.assertRaisesRegex(RuntimeError, "cohort seal"):
                _experimental_execution_contract(_request(), tampered)

            inconsistent = _config(Path(directory))
            inconsistent["guard_cohort"]["members"][0]["solve_generation"] += 1
            inconsistent["guard_cohort_sha256"] = _canonical_sha256(
                inconsistent["guard_cohort"]
            )
            with self.assertRaisesRegex(RuntimeError, "member contract"):
                _experimental_execution_contract(_request(), inconsistent)

    def test_exact_guard_authenticates_terminal_results_leases_and_session(self):
        cohort = _cohort()
        tasks = {}
        stdout = {}
        leases = []
        for member in cohort["members"]:
            tasks[member["task_id"]] = {
                "id": member["task_id"],
                "name": member["name"],
                "status": "completed",
                "exit_code": 0,
                "priority": member["priority"],
                "project": member["project"],
                "aedt_backend": member["aedt_backend"],
                "account_name": member["account_name"],
                "dedupe_key": member["dedupe_key"],
            }
            result = {key: 1 for key in EXACT_COHORT_RESULT_FLAGS}
            result.update({
                "thermal_required_missing_count": 0,
                "thermal_probe_failure_count": 0,
                "thermal_dispatch_status": "success",
                "matrix_solve_attempts": 1,
                "loss_solve_attempts": 1,
                "thermal_solve_attempts": 1,
                "git_dirty": 0,
                "pyaedt_library_git_dirty": 0,
                "git_hash": FEA_SOLVER,
                "pyaedt_library_git_hash": FEA_LIBRARY,
                "project_name": member["lease_project_name"],
                "saved_at": "2026-07-16 12:00:00",
            })
            stdout[member["task_id"]] = (
                f"MFT_LIBRARY_GIT_HASH {FEA_LIBRARY}\n"
                f"RESULT_JSON {json.dumps(result)}\n"
            )
            leases.append({
                "id": member["lease_id"],
                "task_id": member["task_id"],
                "session_id": cohort["session_id"],
                "slot_index": member["slot_index"],
                "request_key": member["lease_request_key"],
                "project_name": member["lease_project_name"],
                "state": "released",
                "failure_message": "",
            })
        pool = {
            "sessions": [{
                "id": cohort["session_id"],
                "generation": cohort["session_generation"],
                "solve_batch_generation": cohort["solve_batch_generation"],
                "slots_total": 3,
                "state": "ready",
                "process_id": "12345",
                "failure_message": "",
                "quarantine_reason": "",
                "last_fault_at": None,
                "last_fault_evidence_json": "{}",
                "active_lease_count": 0,
                "free_slot_count": 3,
            }],
            "session_history": [],
            "leases": leases,
        }

        def get(url, **_kwargs):
            response = mock.Mock()
            if url.endswith("/api/aedt-pool"):
                response.json.return_value = pool
            elif url.endswith("/stdout"):
                task_id = int(url.split("/")[-2])
                response.text = stdout[task_id]
            else:
                task_id = int(url.rsplit("/", 1)[1])
                response.json.return_value = tasks[task_id]
            return response

        scheduler = mock.Mock()
        scheduler.SCHEDULER = "http://scheduler"
        scheduler.MFT_PROJECT = "MFT_1MW_2026v1"
        scheduler.requests.get.side_effect = get
        state = {}
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            _wait_for_exact_cohort_guard(
                scheduler,
                state,
                state_path,
                {
                    "guard_cohort": cohort,
                    "guard_cohort_sha256": _canonical_sha256(cohort),
                    "guard_timeout_seconds": 60,
                },
                5,
            )
            persisted = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(persisted["pre_submission_guard"]["outcome"], "passed")
        self.assertEqual(scheduler.requests.get.call_count, 7)

        # A later verification job may observe the session only after normal
        # idle scale-down.  The durable ready->idle->drain->closed sequence is
        # still reusable evidence; an unhealthy/failed close is not.
        closed_session = pool["sessions"].pop()
        closed_session.update({
            "state": "closed",
            "idle_since": "2026-07-16 12:10:00",
            "drain_requested_at": "2026-07-16 12:15:00",
            "closed_at": "2026-07-16 12:15:10",
        })
        pool["session_history"] = [closed_session]
        passed, observation = _guard_pool_evidence(scheduler, cohort)
        self.assertTrue(passed)
        self.assertEqual(
            observation["session"]["reusable_evidence"],
            "clean_idle_close_after_release",
        )


class ExperimentalVerificationAdapterTests(unittest.TestCase):
    def test_three_pareto_candidates_create_nonproduction_truth_manifest(self):
        from module.input_parameter_260706 import create_input_parameter

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            base = create_input_parameter({}).iloc[0].to_dict()
            rows = []
            for index in range(3):
                row = dict(base)
                row.update(
                    volume_L=100.0 + 50.0 * index,
                    total_loss_W=2000.0 - 500.0 * index,
                )
                rows.append(row)
            front = source / "pareto_front.csv"
            vectors = source / "pareto_X.npy"
            pd.DataFrame(rows).to_csv(front, index=False)
            np.save(vectors, np.arange(6, dtype=float).reshape(3, 2))
            manifest = {
                "experimental_active_learning": True,
                "production_eligible": False,
                "quality_gate_passed": False,
                "quality_blockers": {"Llt_phys": ["mape"]},
                "strict_full_rows": 2188,
                "experimental_minimum_strict_full_rows": 2000,
                "quality_status_sha256": QUALITY_SHA,
                "solver_revision": TRAINING_SOLVER,
                "library_revision": TRAINING_LIBRARY,
                "training_solver_revision": TRAINING_SOLVER,
                "training_library_revision": TRAINING_LIBRARY,
                "fea_solver_revision": FEA_SOLVER,
                "fea_library_revision": FEA_LIBRARY,
                "dataset_sha256": "d" * 64,
                "training_run_id": "training-run",
                "pareto_front_sha256": sha256_file(front),
                "pareto_X_sha256": sha256_file(vectors),
            }
            (source / "optimization_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            store = GenerationStore(root / "artifacts")
            generation = store.publish_tree(
                "experimental_optimization",
                source,
                metadata={
                    "production_eligible": False,
                    "quality_gate_passed": False,
                },
            )

            def fake_scheduler(request_path, result_path, _config):
                request = json.loads(Path(request_path).read_text(encoding="utf-8"))
                results = [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "completed": True,
                        "valid": True,
                        "task_id": 5000 + rank,
                        "attempt": 0,
                        "actual_volume_L": 110.0 + rank,
                        "actual_total_loss_W": 1200.0 + rank,
                    }
                    for rank, candidate in enumerate(request["candidates"])
                ]
                Path(result_path).write_text(
                    json.dumps(
                        {"schema_version": 1, "stage": "fine", "results": results}
                    ),
                    encoding="utf-8",
                )

            output = root / "output"
            with mock.patch.object(
                experimental_verification,
                "run_scheduler_verification",
                side_effect=fake_scheduler,
            ):
                status = experimental_verification.run(
                    generation.path, output, {}
                )
            selection = json.loads(
                (output / "selection.json").read_text(encoding="utf-8")
            )
            truth = json.loads(
                (output / "truth_ingest_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(selection["candidates"]), 3)
            self.assertFalse(selection["production_eligible"])
            self.assertEqual(selection["solver_revision"], FEA_SOLVER)
            self.assertEqual(
                selection["training_solver_revision"], TRAINING_SOLVER
            )
            self.assertEqual(len(truth["results"]), 3)
            self.assertEqual(truth["fea_solver_revision"], FEA_SOLVER)
            self.assertEqual(truth["training_solver_revision"], TRAINING_SOLVER)
            self.assertFalse(status["production_eligible"])
            self.assertTrue((output / "COMPLETED").is_file())


if __name__ == "__main__":
    unittest.main()
