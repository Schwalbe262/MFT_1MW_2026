import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from regression_260707.pipeline.artifacts import GenerationStore
from regression_260707.pipeline import verification_adapter
from regression_260707.pipeline.scheduler_verification import (
    run_scheduler_verification,
)


class VerificationAdapterTests(unittest.TestCase):
    @staticmethod
    def _optimization(store, root, count=40, include_fea_revisions=True):
        source = root / "optimization"
        source.mkdir()
        pd.DataFrame(
            {
                "volume_L": np.linspace(100.0, 200.0, count),
                "total_loss_W": np.linspace(2000.0, 1000.0, count),
                "design_value": np.arange(count),
            }
        ).to_csv(source / "pareto_front.csv", index=False)
        np.save(source / "pareto_X.npy", np.arange(count * 2).reshape(count, 2))
        manifest = {
            "solver_revision": "a" * 40,
            "library_revision": "b" * 40,
            "training_solver_revision": "a" * 40,
            "training_library_revision": "b" * 40,
        }
        if include_fea_revisions:
            manifest.update({
                "fea_solver_revision": "c" * 40,
                "fea_library_revision": "d" * 40,
            })
        (source / "optimization_manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        return store.publish_tree("optimization", source)

    @staticmethod
    def _fake_scheduler(request_path, result_path, _config):
        request = json.loads(Path(request_path).read_text(encoding="utf-8"))
        results = []
        for rank, candidate in enumerate(request["candidates"]):
            results.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "completed": True,
                    "valid": True,
                    "task_id": 1000 + rank,
                    "attempt": 0,
                    "actual_volume_L": 1000.0 - rank,
                    "actual_total_loss_W": 500.0 + rank,
                }
            )
        Path(result_path).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "stage": request["stage"],
                    "results": results,
                }
            ),
            encoding="utf-8",
        )

    def test_exact_33_then_smallest_valid_exact_3_are_authenticated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = GenerationStore(root / "artifacts")
            optimization = self._optimization(store, root)
            standard_output = root / "standard-output"
            with mock.patch(
                "regression_260707.pipeline.scheduler_verification."
                "run_scheduler_verification",
                side_effect=self._fake_scheduler,
            ):
                status = verification_adapter.run(
                    "standard",
                    optimization.path,
                    standard_output,
                    33,
                    {"adapter": "mft_scheduler_v1"},
                )
            self.assertTrue(status["completed"])
            self.assertEqual(status["terminal_count"], 33)
            standard_request = json.loads(
                (standard_output / "selection.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(standard_request["candidates"]), 33)
            self.assertEqual(standard_request["solver_revision"], "c" * 40)
            self.assertEqual(
                standard_request["training_solver_revision"], "a" * 40
            )

            standard = store.publish_tree(
                "verification_standard", standard_output
            )
            fine_output = root / "fine-output"
            with mock.patch(
                "regression_260707.pipeline.scheduler_verification."
                "run_scheduler_verification",
                side_effect=self._fake_scheduler,
            ):
                fine_status = verification_adapter.run(
                    "fine",
                    standard.path,
                    fine_output,
                    3,
                    {"adapter": "mft_scheduler_v1"},
                )
            self.assertEqual(fine_status["terminal_count"], 3)
            fine_request = json.loads(
                (fine_output / "selection.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(fine_request["candidates"]), 3)
            self.assertEqual(fine_request["solver_revision"], "c" * 40)
            self.assertEqual(
                fine_request["training_solver_revision"], "a" * 40
            )
            expected = [
                item["candidate_id"]
                for item in standard_request["candidates"][-3:][::-1]
            ]
            self.assertEqual(
                [item["candidate_id"] for item in fine_request["candidates"]],
                expected,
            )

    def test_standard_fails_closed_when_front_has_fewer_than_33(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = GenerationStore(root / "artifacts")
            optimization = self._optimization(store, root, count=32)
            with self.assertRaisesRegex(RuntimeError, "requires 33"):
                verification_adapter.run(
                    "standard",
                    optimization.path,
                    root / "output",
                    33,
                    {"adapter": "mft_scheduler_v1"},
                )

    def test_sealed_counts_cannot_be_weakened(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = GenerationStore(root / "artifacts")
            optimization = self._optimization(store, root)
            with self.assertRaisesRegex(RuntimeError, "sealed at 33"):
                verification_adapter.run(
                    "standard",
                    optimization.path,
                    root / "output",
                    3,
                    {"adapter": "mft_scheduler_v1"},
                )

    def test_training_revision_cannot_be_reused_as_implicit_fea_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = GenerationStore(root / "artifacts")
            optimization = self._optimization(
                store, root, include_fea_revisions=False
            )
            with self.assertRaisesRegex(RuntimeError, "separate exact training/FEA"):
                verification_adapter.run(
                    "standard",
                    optimization.path,
                    root / "output",
                    33,
                    {"adapter": "mft_scheduler_v1"},
                )


class SchedulerVerificationTests(unittest.TestCase):
    def test_reviewed_client_retries_once_and_records_terminal_identity(self):
        from module.input_parameter_260706 import KEYS, create_input_parameter
        from verify import scheduler_client

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "library"
            library.mkdir()
            complete = create_input_parameter({}).iloc[0].to_dict()
            request = {
                "schema_version": 1,
                "stage": "standard",
                "expected_count": 1,
                "input_generation_id": "f" * 64,
                "solver_revision": "a" * 40,
                "library_revision": "b" * 40,
                "candidates": [
                    {
                        "candidate_id": "candidate-1",
                        "parameters": {key: complete[key] for key in KEYS},
                    }
                ],
            }
            request_path = root / "selection.json"
            result_path = root / "verification_results.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            invalid = scheduler_client.ResultFetch(
                scheduler_client.RESULT_INVALID, None
            )
            submitted_ids = iter((123, 124))
            submission_markers = []

            def submit_with_sealed_state(*args, **kwargs):
                state = json.loads(
                    (root / "scheduler_state.json").read_text(encoding="utf-8")
                )
                marker = state["records"]["candidate-1"]["outcome"]
                submission_markers.append(marker)
                return next(submitted_ids)

            with mock.patch(
                "campaign.deployment_gate.validate_deployment"
            ) as deployment, mock.patch.object(
                scheduler_client,
                "effective_verification_params",
                side_effect=lambda params, profile: dict(params),
            ), mock.patch.object(
                scheduler_client,
                "submit_verification",
                side_effect=submit_with_sealed_state,
            ) as submit, mock.patch.object(
                scheduler_client,
                "wait_all",
                side_effect=lambda ids, **kw: {value: "completed" for value in ids},
            ), mock.patch.object(
                scheduler_client, "fetch_result", return_value=invalid
            ):
                result = run_scheduler_verification(
                    request_path,
                    result_path,
                    {
                        "adapter": "mft_scheduler_v1",
                        "execute": True,
                        "library_root": str(library),
                        "poll_seconds": 5,
                        "timeout_seconds": 60,
                    },
                )
            deployment.assert_called_once()
            self.assertEqual(submit.call_count, 2)
            self.assertEqual(
                submission_markers, ["submitting", "retry_submitting"]
            )
            self.assertTrue(result["results"][0]["completed"])
            self.assertFalse(result["results"][0]["valid"])
            self.assertEqual(result["results"][0]["attempt"], 1)


if __name__ == "__main__":
    unittest.main()
