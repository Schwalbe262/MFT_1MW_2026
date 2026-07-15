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
    EXPERIMENTAL_LIBRARY_REVISION,
    EXPERIMENTAL_SOLVER_REVISION,
    Q7_GUARD_ACCOUNTS,
    Q7_GUARD_TASK_IDS,
    Q7_SUBMISSION_ENV,
    _experimental_execution_contract,
    _wait_for_q7_guard,
)


def _request():
    return {
        "stage": "fine",
        "expected_count": 3,
        "selection_policy": "experimental_pareto_span_3_v1",
        "experimental_active_learning": True,
        "production_eligible": False,
        "quality_gate_passed": False,
        "quality_blockers": {"Llt_phys": ["mape"]},
        "quality_status_sha256": "c" * 64,
        "solver_revision": EXPERIMENTAL_SOLVER_REVISION,
        "library_revision": EXPERIMENTAL_LIBRARY_REVISION,
        "candidates": [
            {"candidate_id": f"candidate-{index}"} for index in range(3)
        ],
    }


def _config(solver_root):
    return {
        "adapter": "mft_scheduler_v1",
        "execute": True,
        "library_root": str(solver_root),
        "solver_root": str(solver_root),
        "priority": -10,
        "required_hard_cap": 3,
        "max_project_active_tasks": 600,
        "aedt_backend": "pooled",
        "submission_env": dict(Q7_SUBMISSION_ENV),
        "accounts": list(Q7_GUARD_ACCOUNTS),
        "q7_guard_task_ids": list(Q7_GUARD_TASK_IDS),
        "q7_guard_timeout_seconds": 3600,
        "experimental_quality_status_sha256": "c" * 64,
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

    def test_q7_guard_authenticates_all_three_successes(self):
        responses = []
        for task_id, account in zip(Q7_GUARD_TASK_IDS, Q7_GUARD_ACCOUNTS):
            response = mock.Mock()
            response.json.return_value = {
                "id": task_id,
                "name": f"mft-1to3-q7-full-267860a-r{task_id}",
                "status": "completed",
                "exit_code": 0,
                "priority": 10,
                "project": "MFT_1MW_2026v1",
                "aedt_backend": "pooled",
                "account_name": account,
            }
            responses.append(response)
        scheduler = mock.Mock()
        scheduler.SCHEDULER = "http://scheduler"
        scheduler.MFT_PROJECT = "MFT_1MW_2026v1"
        scheduler.requests.get.side_effect = responses
        state = {}
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            _wait_for_q7_guard(
                scheduler,
                state,
                state_path,
                {
                    "guard_ids": list(Q7_GUARD_TASK_IDS),
                    "guard_timeout_seconds": 60,
                },
                5,
            )
            persisted = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(persisted["pre_submission_guard"]["outcome"], "passed")
        self.assertEqual(scheduler.requests.get.call_count, 3)


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
                "quality_status_sha256": "c" * 64,
                "solver_revision": EXPERIMENTAL_SOLVER_REVISION,
                "library_revision": EXPERIMENTAL_LIBRARY_REVISION,
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
            self.assertEqual(len(truth["results"]), 3)
            self.assertFalse(status["production_eligible"])
            self.assertTrue((output / "COMPLETED").is_file())


if __name__ == "__main__":
    unittest.main()
