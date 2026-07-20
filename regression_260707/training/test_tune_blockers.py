import hashlib
import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock


TRAINING_ROOT = Path(__file__).resolve().parent
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import tune_blockers  # noqa: E402


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _FakeTrial:
    def __init__(self, state, value=2.0, params=None, evidence=None):
        self.state = state
        self.value = value
        self.params = params or {}
        self.user_attrs = {}
        if evidence is not None:
            self.user_attrs["quality_gate_evidence"] = evidence

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value

    def report(self, _value, _step):
        return None


class _FakeStudy:
    def __init__(self, completed_trials):
        evidence = {
            "score": 2.0,
            "failed_constraint_count": 0,
            "failed_constraints": [],
        }
        self.trials = [
            _FakeTrial(
                "COMPLETE",
                value=2.0,
                params={"existing": index},
                evidence=evidence,
            )
            for index in range(completed_trials)
        ]
        self.user_attrs = {}

    def get_trials(self, deepcopy=False):
        assert deepcopy is False
        return self.trials

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value

    def optimize(self, objective, n_trials, show_progress_bar):
        assert show_progress_bar is False
        for index in range(n_trials):
            trial = _FakeTrial("RUNNING", params={"new": index})
            trial.value = objective(trial)
            trial.state = "COMPLETE"
            self.trials.append(trial)

    @property
    def best_trial(self):
        return min(self.trials, key=lambda trial: trial.value)

    @property
    def best_params(self):
        return self.best_trial.params

    @property
    def best_value(self):
        return self.best_trial.value


def _fake_optuna(study, captured):
    module = types.ModuleType("optuna")

    class Storage:
        def __init__(self, url):
            self.url = url
            self.engine = types.SimpleNamespace(dispose=lambda: None)

        def remove_session(self):
            return None

    module.trial = types.SimpleNamespace(
        TrialState=types.SimpleNamespace(COMPLETE="COMPLETE")
    )
    module.storages = types.SimpleNamespace(RDBStorage=Storage)
    module.samplers = types.SimpleNamespace(
        TPESampler=lambda **kwargs: ("sampler", kwargs)
    )
    module.pruners = types.SimpleNamespace(NopPruner=lambda: "pruner")

    def create_study(**kwargs):
        captured.update(kwargs)
        return study

    module.create_study = create_study
    return module


class ProductionBlockerHpoTests(unittest.TestCase):
    def test_repository_config_is_exact_blocker_only_staged_manifest(self):
        config = tune_blockers.load_blocker_config(
            TRAINING_ROOT / "blocker_hpo_cap_recovery_6151.json"
        )
        self.assertEqual(config["trial_stages"], [8, 20, 50, 200])
        self.assertEqual(config["trials_per_job"], 200)
        self.assertEqual(len(config["blockers"]), 11)
        self.assertEqual(len(config["families"]), 4)
        self.assertNotIn("C_tx_tx_F", config["blockers"])

    def test_quality_objective_matches_gate_boundaries_and_reasons(self):
        limits = {
            "min_r2": 0.98,
            "max_mape_pct": 2.0,
            "max_p90_ape_pct": 4.0,
        }
        passing = tune_blockers.quality_gate_objective(
            {
                "interval_coverage": 0.85,
                "r2": 0.98,
                "mape_pct": 2.0,
                "p90_ape_pct": 4.0,
            },
            limits,
            0.85,
        )
        self.assertEqual(passing["failed_constraint_count"], 0)
        self.assertEqual(passing["failed_constraints"], [])
        for ratio in passing["normalized_gate_ratios"].values():
            self.assertAlmostEqual(ratio, 1.0)

        failing = tune_blockers.quality_gate_objective(
            {
                "interval_coverage": 0.80,
                "r2": 0.90,
                "mape_pct": 3.0,
                "p90_ape_pct": 4.0,
            },
            limits,
            0.85,
        )
        self.assertEqual(
            set(failing["failed_constraints"]),
            {
                "metric_below_minimum:interval_coverage",
                "metric_below_minimum:r2",
                "metric_above_maximum:mape_pct",
            },
        )
        self.assertGreater(failing["score"], 3_000_000.0)

    def test_corrected_recovery_is_required_for_non_capacitance_jobs(self):
        config = {
            "source": {"strict_full_rows": 6151},
            "capacitance_recovery": {
                "recovered_row_count": 6151,
                "max_allowed_abs_delta_F": 5.1e-11,
            },
        }
        quality = {
            "capacitance_recovery": {
                "max_observed_abs_delta_F": 4.9999999972927653e-11,
            }
        }
        audit = {
            "contract": tune_blockers.CAPACITANCE_RECOVERY_CONTRACT,
            "status": "applied",
            "recovered_row_count": 6151,
            "max_observed_abs_delta_F": 4.9999999972927653e-11,
        }
        tune_blockers._require_corrected_recovery(
            config, quality, audit, 6151
        )
        with self.assertRaisesRegex(RuntimeError, "requires authenticated"):
            tune_blockers._require_corrected_recovery(
                config, quality, {**audit, "status": "legacy_passthrough"}, 6151
            )

    def test_source_documents_are_content_pinned_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "strict.parquet"
            thresholds_path = root / "thresholds.json"
            quality_path = root / "quality.json"
            config_path = root / "config.json"
            dataset.write_bytes(b"immutable strict dataset")
            thresholds = {
                "minimum_interval_coverage": 0.85,
                "targets": {"Llt_phys": {"min_r2": 0.98}},
            }
            thresholds_path.write_text(
                json.dumps(thresholds, indent=1), encoding="utf-8"
            )
            canonical_thresholds = tune_blockers._canonical_sha256(thresholds)
            profile_sha = "b" * 64
            quality = {
                "passed": False,
                "training_run_id": "run-1",
                "generation": "generations/run-1",
                "dataset_sha256": _sha256(dataset),
                "profile_sha256": profile_sha,
                "strict_full_rows": 1,
                "thresholds_sha256": canonical_thresholds,
                "quality_thresholds_sha256": _sha256(thresholds_path),
                "targets": {
                    "Llt_phys": {
                        "blocking": True,
                        "passed": False,
                        "reasons": ["metric_below_minimum:r2"],
                    }
                },
                "capacitance_recovery": {
                    "passed": True,
                    "reasons": [],
                    "contract": tune_blockers.CAPACITANCE_RECOVERY_CONTRACT,
                    "recovered_row_count": 1,
                    "dataset_sha256": _sha256(dataset),
                    "profile_sha256": profile_sha,
                    "max_observed_abs_delta_F": 1e-12,
                },
            }
            quality_path.write_text(json.dumps(quality), encoding="utf-8")
            config_path.write_text("{}", encoding="utf-8")
            config = {
                "source": {
                    "training_run_id": "run-1",
                    "generation": "generations/run-1",
                    "dataset_sha256": _sha256(dataset),
                    "profile_sha256": profile_sha,
                    "strict_full_rows": 1,
                    "quality_status_sha256": _sha256(quality_path),
                    "quality_thresholds_sha256": canonical_thresholds,
                    "quality_thresholds_file_sha256": _sha256(thresholds_path),
                },
                "blockers": ["Llt_phys"],
                "expected_blocking_reasons": {
                    "Llt_phys": ["metric_below_minimum:r2"]
                },
                "capacitance_recovery": {"recovered_row_count": 1},
            }
            contract = tune_blockers._validate_source_documents(
                config,
                config_path,
                dataset,
                quality_path,
                thresholds_path,
            )
            self.assertEqual(contract["dataset_sha256"], _sha256(dataset))

            dataset.write_bytes(b"drifted strict dataset")
            with self.assertRaisesRegex(RuntimeError, "dataset fingerprint"):
                tune_blockers._validate_source_documents(
                    config,
                    config_path,
                    dataset,
                    quality_path,
                    thresholds_path,
                )

    def test_tune_one_appends_only_to_requested_cumulative_stage(self):
        study = _FakeStudy(completed_trials=3)
        captured = {}
        fake_optuna = _fake_optuna(study, captured)
        metrics = {"interval_coverage": 0.90, "r2": 0.99}
        source_contract = {
            "quality_thresholds_sha256": "c" * 64,
            "thresholds": {"minimum_interval_coverage": 0.85},
        }
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            sys.modules, {"optuna": fake_optuna}
        ), mock.patch.object(
            tune_blockers.tune_optuna,
            "sample_params",
            return_value={"n_estimators": 400},
        ), mock.patch.object(
            tune_blockers.tune_optuna,
            "model_params_with_thread_budget",
            return_value={"n_estimators": 400, "n_jobs": 1},
        ), mock.patch.object(
            tune_blockers,
            "train_target",
            return_value=({"models": []}, metrics),
        ) as train:
            _, _, _, _, resume = tune_blockers.tune_one(
                "Llt_phys",
                "lightgbm",
                5,
                object(),
                ["feature"],
                {"min_r2": 0.98},
                source_contract,
                model_threads=1,
                sample_weight_column="sample_weight",
                minimum_eligible_rows=4000,
                study_root=directory,
                config_sha256="d" * 64,
            )

        self.assertEqual(train.call_count, 2)
        self.assertEqual(resume["completed_trials_before"], 3)
        self.assertEqual(resume["completed_trials_after"], 5)
        self.assertEqual(resume["completed_trials_added"], 2)
        self.assertTrue(captured["load_if_exists"])
        self.assertTrue(captured["storage"].url.startswith("sqlite:///"))

    def test_real_optuna_sqlite_study_resumes_across_stages(self):
        metrics = {"interval_coverage": 0.90, "r2": 0.99}
        source_contract = {
            "quality_thresholds_sha256": "c" * 64,
            "thresholds": {"minimum_interval_coverage": 0.85},
        }
        common = dict(
            model_threads=1,
            sample_weight_column="sample_weight",
            minimum_eligible_rows=4000,
            config_sha256="d" * 64,
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            tune_blockers.tune_optuna,
            "sample_params",
            side_effect=lambda trial, _family: {
                "n_estimators": trial.suggest_int("n_estimators", 10, 20)
            },
        ), mock.patch.object(
            tune_blockers,
            "train_target",
            return_value=({"models": []}, metrics),
        ) as train:
            first = tune_blockers.tune_one(
                "Llt_phys", "lightgbm", 1, object(), ["feature"],
                {"min_r2": 0.98}, source_contract,
                study_root=directory, **common,
            )
            second = tune_blockers.tune_one(
                "Llt_phys", "lightgbm", 2, object(), ["feature"],
                {"min_r2": 0.98}, source_contract,
                study_root=directory, **common,
            )

        self.assertEqual(train.call_count, 2)
        self.assertEqual(first[4]["completed_trials_before"], 0)
        self.assertEqual(first[4]["completed_trials_after"], 1)
        self.assertEqual(second[4]["completed_trials_before"], 1)
        self.assertEqual(second[4]["completed_trials_after"], 2)

    def test_parallel_jobs_keep_order_and_share_explicit_budget(self):
        calls = []

        def fake_tune(target, family, trials, *_args, **kwargs):
            calls.append((target, family, trials, kwargs["model_threads"]))
            if target == "slow":
                time.sleep(0.02)
            contract = {"sha256": target * 4}
            resume = {
                "completed_trials_after": trials,
                "completed_trials_before": 0,
                "completed_trials_added": trials,
            }
            return {"target": target}, 1.0, {"score": 1.0}, contract, resume

        source = {
            "thresholds": {
                "minimum_interval_coverage": 0.85,
                "targets": {"slow": {}, "fast": {}},
            }
        }
        with mock.patch.object(
            tune_blockers, "tune_one", side_effect=fake_tune
        ):
            params, results, workers = tune_blockers.run_blocker_jobs(
                [("slow", "lightgbm"), ("fast", "xgboost")],
                8,
                object(),
                ["feature"],
                source,
                model_threads=1,
                job_workers=2,
                max_model_thread_budget=2,
                sample_weight_column="sample_weight",
                minimum_eligible_rows=4000,
                study_root="studies",
                config_sha256="e" * 64,
            )

        self.assertEqual(workers, 2)
        self.assertEqual([item["target"] for item in results], ["slow", "fast"])
        self.assertEqual(params["lightgbm"]["slow"]["completed_trials"], 8)
        self.assertEqual(
            set(calls),
            {
                ("slow", "lightgbm", 8, 1),
                ("fast", "xgboost", 8, 1),
            },
        )

    def test_published_metadata_is_parameter_only_and_fail_closed(self):
        audit = {
            "contract": tune_blockers.CAPACITANCE_RECOVERY_CONTRACT,
            "status": "applied",
            "recovered_row_count": 1,
            "max_observed_abs_delta_F": 1e-12,
        }
        config = {
            "trial_stages": [8, 20],
            "trials_per_job": 20,
            "blockers": ["Llt_phys"],
            "families": ["lightgbm"],
            "cohort": {
                "solver_revision": "1" * 40,
                "library_revision": "2" * 40,
                "data_contract_sha256": "3" * 64,
            },
            "source": {
                "training_run_id": "run",
                "generation": "generations/run",
                "profile_sha256": "4" * 64,
                "strict_full_rows": 1,
            },
            "capacitance_recovery": {
                "recovered_row_count": 1,
                "max_allowed_abs_delta_F": 5.1e-11,
            },
            "minimum_eligible_rows_per_target": 1,
            "sample_weight_column": "sample_weight",
        }
        source = {
            "config_sha256": "5" * 64,
            "dataset_path": "strict.parquet",
            "dataset_sha256": "6" * 64,
            "quality_status": {
                "capacitance_recovery": {
                    "max_observed_abs_delta_F": 1e-12,
                }
            },
            "quality_status_sha256": "7" * 64,
            "quality_thresholds_sha256": "8" * 64,
            "quality_thresholds_file_sha256": "9" * 64,
            "thresholds": {
                "minimum_interval_coverage": 0.85,
                "targets": {"Llt_phys": {"min_r2": 0.98}},
            },
        }
        published = {
            "generation_id": "generation",
            "generation_path": "generation",
        }
        argv = [
            "tune_blockers.py",
            "--config", "config.json",
            "--dataset", "strict.parquet",
            "--quality-status", "quality.json",
            "--thresholds", "thresholds.json",
            "--artifact-root", "artifacts",
            "--result-json", "result.json",
            "--study-root", "studies",
            "--stage-trials", "20",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            tune_blockers, "load_blocker_config", return_value=config
        ), mock.patch.object(
            tune_blockers, "_validate_source_documents", return_value=source
        ), mock.patch.object(
            tune_blockers.tune_optuna,
            "_load_strict_dataset",
            return_value=(object(), 1),
        ), mock.patch.object(
            tune_blockers, "capacitance_recovery_audit", return_value=audit
        ), mock.patch.object(
            tune_blockers, "feature_columns", return_value=["feature"]
        ), mock.patch.object(
            tune_blockers,
            "_eligible_target_counts",
            return_value={"Llt_phys": 1},
        ), mock.patch.object(
            tune_blockers,
            "run_blocker_jobs",
            return_value=({"lightgbm": {}}, [], 1),
        ) as run, mock.patch.object(
            tune_blockers.tune_optuna,
            "_publish_generation",
            return_value=published,
        ) as publish:
            tune_blockers.main()

        self.assertEqual(run.call_args.args[1], 20)
        metadata = publish.call_args.args[2]
        self.assertEqual(metadata["lane"], "production_gate_blocker_hpo")
        self.assertTrue(metadata["parameter_artifact_eligible"])
        self.assertFalse(metadata["production_eligible"])
        self.assertFalse(metadata["production_model_eligible"])
        self.assertFalse(metadata["promotion_approved"])
        self.assertFalse(metadata["fea_submission_approved"])


if __name__ == "__main__":
    unittest.main()
