import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


TRAINING_ROOT = Path(__file__).resolve().parent
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import tune_optuna  # noqa: E402


class _Trial:
    def suggest_int(self, _name, low, _high):
        return low

    def suggest_float(self, _name, low, _high, **_kwargs):
        return low

    def report(self, _value, _step):
        return None

    def should_prune(self):
        return False


class _Study:
    def __init__(self):
        self.best_params = {}
        self.best_value = None

    def optimize(self, objective, n_trials, show_progress_bar):
        assert n_trials == 1
        assert show_progress_bar is False
        self.best_value = objective(_Trial())


class _Model:
    def fit(self, _features, _target):
        return self

    def predict(self, features):
        return np.zeros(len(features), dtype=float)


def _fake_optuna():
    module = types.ModuleType("optuna")
    module.TrialPruned = RuntimeError
    module.samplers = types.SimpleNamespace(TPESampler=lambda **_kwargs: object())
    module.pruners = types.SimpleNamespace(MedianPruner=lambda **_kwargs: object())
    module.create_study = lambda **_kwargs: _Study()
    return module


class ModelThreadBudgetTests(unittest.TestCase):
    @staticmethod
    def _applied_recovery_audit():
        return {
            "contract": tune_optuna.CAPACITANCE_RECOVERY_CONTRACT,
            "max_allowed_abs_delta_F": 5.1e-11,
            "row_count": 7212,
            "cap_enabled_row_count": 7212,
            "eligible_row_count": 7212,
            "recovered_row_count": 7212,
            "max_observed_abs_delta_F": 4.999e-11,
            "missing_columns": [],
            "status": "applied",
        }

    def test_strict_loader_preserves_corrected_capacitance_audit(self):
        exact = {
            "C_tx_tx_F": 1.856789012345678e-10,
            "C_rx_rx_F": 8.721234567890123e-10,
            "C_tx_rx_F": 3.456789012345678e-10,
        }
        evidence = {
            "C_tx_tx_F": ("f_res_tx_self_Hz", "cap_L_tx_self_H", 180e-6),
            "C_rx_rx_F": ("f_res_rx_self_Hz", "cap_L_rx_self_H", 780e-6),
            "C_tx_rx_F": (
                "f_res_interwinding_Hz", "cap_L_leakage_H", 27.5e-6
            ),
        }
        row = {"cap_on": 1, "full_model": 1}
        for target, value in exact.items():
            frequency, inductance, inductance_value = evidence[target]
            row[target] = round(value / 1e-10) * 1e-10
            row[inductance] = inductance_value
            row[frequency] = 1.0 / (
                2.0 * np.pi * np.sqrt(inductance_value * value)
            )
        raw = pd.DataFrame([row])

        def annotate(frame, **_kwargs):
            return frame.assign(_strict_valid_full=True)

        with mock.patch.object(
            tune_optuna.pd, "read_parquet", return_value=raw
        ), mock.patch("quality_contract.annotate_validity", side_effect=annotate):
            frame, strict_count = tune_optuna._load_strict_dataset("unused")

        audit = tune_optuna.capacitance_recovery_audit(frame)
        self.assertEqual(strict_count, 1)
        self.assertEqual(audit["contract"], tune_optuna.CAPACITANCE_RECOVERY_CONTRACT)
        self.assertEqual(audit["status"], "applied")
        self.assertEqual(audit["recovered_row_count"], 1)
        self.assertIsNotNone(audit["max_observed_abs_delta_F"])

    def test_production_capacitance_gate_is_fail_closed(self):
        jobs = [("C_tx_tx_F", "lightgbm")]
        legacy = {
            **self._applied_recovery_audit(),
            "recovered_row_count": 0,
            "max_observed_abs_delta_F": None,
            "status": "evidence_unavailable_legacy_passthrough",
        }
        with self.assertRaisesRegex(RuntimeError, "requires applied corrected"):
            tune_optuna._require_production_capacitance_recovery(
                jobs, legacy, experimental=False
            )

        # Experimental HPO remains usable for legacy fixtures, and production
        # jobs without a capacitance target are outside this recovery gate.
        tune_optuna._require_production_capacitance_recovery(
            jobs, legacy, experimental=True
        )
        tune_optuna._require_production_capacitance_recovery(
            [("Llt_phys", "lightgbm")], legacy, experimental=False
        )

    def test_main_seals_recovery_audit_into_generation_metadata(self):
        audit = self._applied_recovery_audit()
        frame = pd.DataFrame({
            "feature": [1.0],
            "C_tx_tx_F": [1.8e-10],
        })
        frame.attrs[tune_optuna.CAPACITANCE_RECOVERY_ATTR] = audit
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "strict.parquet"
            dataset.write_bytes(b"strict dataset")
            result_json = root / "result.json"
            argv = [
                "tune_optuna.py",
                "--target", "C_tx_tx_F",
                "--family", "lightgbm",
                "--trials", "1",
                "--dataset", str(dataset),
                "--artifact-root", str(root / "artifacts"),
                "--result-json", str(result_json),
            ]
            published = {
                "schema_version": 1,
                "generation_id": "generation",
                "generation_path": str(root / "generation"),
                "params_path": str(root / "generation" / "params.json"),
                "manifest_path": str(root / "generation" / "manifest.json"),
            }
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                tune_optuna,
                "_load_strict_dataset",
                return_value=(frame, 4000),
            ), mock.patch.object(
                tune_optuna, "feature_columns", return_value=["feature"]
            ), mock.patch.object(
                tune_optuna,
                "run_tuning_jobs",
                return_value=(
                    {"lightgbm": {"C_tx_tx_F": {"params": {}}}},
                    [{
                        "target": "C_tx_tx_F",
                        "family": "lightgbm",
                        "eligible_rows": 4000,
                        "cv_mse_transformed": 0.0,
                    }],
                    1,
                ),
            ), mock.patch.object(
                tune_optuna, "_publish_generation", return_value=published
            ) as publish:
                tune_optuna.main()

        metadata = publish.call_args.args[2]
        self.assertEqual(metadata["capacitance_recovery"], audit)

    def test_main_rejects_uncorrected_production_capacitance_hpo(self):
        frame = pd.DataFrame({
            "feature": [1.0],
            "C_tx_tx_F": [2e-10],
        })
        frame.attrs[tune_optuna.CAPACITANCE_RECOVERY_ATTR] = {
            **self._applied_recovery_audit(),
            "recovered_row_count": 0,
            "max_observed_abs_delta_F": None,
            "status": "evidence_unavailable_legacy_passthrough",
        }
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "strict.parquet"
            dataset.write_bytes(b"legacy dataset")
            argv = [
                "tune_optuna.py",
                "--target", "C_tx_tx_F",
                "--family", "lightgbm",
                "--trials", "1",
                "--dataset", str(dataset),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                tune_optuna,
                "_load_strict_dataset",
                return_value=(frame, 4000),
            ), mock.patch.object(
                tune_optuna, "feature_columns", return_value=["feature"]
            ), mock.patch.object(tune_optuna, "run_tuning_jobs") as tuning:
                with self.assertRaisesRegex(
                    RuntimeError, "requires applied corrected"
                ):
                    tune_optuna.main()
        tuning.assert_not_called()

    def test_publish_result_and_manifest_seal_recovery_audit(self):
        audit = self._applied_recovery_audit()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_json = root / "result.json"
            result = tune_optuna._publish_generation(
                root / "artifacts",
                {"lightgbm": {}},
                {
                    "dataset_sha256": "a" * 64,
                    "capacitance_recovery": audit,
                },
                result_json=result_json,
            )
            manifest = json.loads(
                Path(result["manifest_path"]).read_text(encoding="utf-8")
            )
            persisted_result = json.loads(result_json.read_text(encoding="utf-8"))

        self.assertEqual(manifest["metadata"]["capacitance_recovery"], audit)
        self.assertEqual(result["capacitance_recovery"], audit)
        self.assertEqual(persisted_result["capacitance_recovery"], audit)

    def test_every_model_family_receives_the_explicit_runtime_budget(self):
        expected_parameters = {
            "lightgbm": "n_jobs",
            "xgboost": "n_jobs",
            "catboost": "thread_count",
            "extratrees": "n_jobs",
        }
        for family, parameter in expected_parameters.items():
            with self.subTest(family=family):
                runtime = tune_optuna.model_params_with_thread_budget(
                    family,
                    {parameter: -1, "search_value": 7},
                    24,
                )
                self.assertEqual(runtime[parameter], 24)
                self.assertEqual(runtime["search_value"], 7)

    def test_tuning_objective_passes_budget_to_every_fold_and_family(self):
        frame = pd.DataFrame({
            "feature": np.arange(8, dtype=float),
            "target": np.arange(8, dtype=float),
        })
        expected_parameters = {
            "lightgbm": "n_jobs",
            "xgboost": "n_jobs",
            "catboost": "thread_count",
            "extratrees": "n_jobs",
        }
        for family, parameter in expected_parameters.items():
            captured = []

            def make_model(actual_family, params, seed):
                captured.append((actual_family, dict(params), seed))
                return _Model()

            with (
                self.subTest(family=family),
                mock.patch.dict(sys.modules, {"optuna": _fake_optuna()}),
                mock.patch.object(
                    tune_optuna,
                    "TARGETS",
                    {"target": {"transform": "none"}},
                ),
                mock.patch.object(
                    tune_optuna,
                    "filter_valid_training_rows",
                    side_effect=lambda value, _target: value,
                ),
                mock.patch.object(
                    tune_optuna,
                    "transform_y",
                    side_effect=lambda value, _transform: value,
                ),
                mock.patch.object(tune_optuna, "make_model", side_effect=make_model),
            ):
                tune_optuna.tune(
                    "target",
                    family,
                    1,
                    frame,
                    ["feature"],
                    model_threads=7,
                )

            self.assertEqual(len(captured), 4)
            self.assertEqual({item[0] for item in captured}, {family})
            self.assertEqual({item[1][parameter] for item in captured}, {7})
            self.assertEqual({item[2] for item in captured}, {0, 1, 2, 3})

    def test_non_positive_or_boolean_budget_is_rejected(self):
        for invalid in (0, -1, False):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                tune_optuna.model_params_with_thread_budget(
                    "lightgbm", {}, invalid
                )

    def test_parallel_studies_share_one_budget_and_keep_declared_order(self):
        calls = []

        def fake_tune(
            target, family, trials, _frame, _features, *, model_threads
        ):
            calls.append((target, family, trials, model_threads))
            if target == "slow":
                time.sleep(0.02)
            value = 1.0 if target == "slow" else 2.0
            return {"target": target}, value, 100

        with mock.patch.object(tune_optuna, "tune", side_effect=fake_tune):
            params, results, workers = tune_optuna.run_tuning_jobs(
                [("slow", "lightgbm"), ("fast", "xgboost")],
                3,
                object(),
                ["feature"],
                model_threads=1,
                job_workers=2,
                max_model_thread_budget=2,
            )

        self.assertEqual(workers, 2)
        self.assertEqual([item["target"] for item in results], ["slow", "fast"])
        self.assertEqual(
            params["lightgbm"]["slow"]["params"], {"target": "slow"}
        )
        self.assertEqual(
            params["xgboost"]["fast"]["params"], {"target": "fast"}
        )
        self.assertEqual(
            {
                (target, family, trials, threads)
                for target, family, trials, threads in calls
            },
            {
                ("slow", "lightgbm", 3, 1),
                ("fast", "xgboost", 3, 1),
            },
        )

    def test_parallel_studies_reject_an_oversubscribed_budget(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            tune_optuna.run_tuning_jobs(
                [("a", "lightgbm"), ("b", "xgboost")],
                1,
                object(),
                ["feature"],
                model_threads=2,
                job_workers=2,
                max_model_thread_budget=3,
            )


if __name__ == "__main__":
    unittest.main()
