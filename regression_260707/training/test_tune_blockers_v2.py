import json
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd


TRAINING_ROOT = Path(__file__).resolve().parent
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import train_models  # noqa: E402
import outer_acceptance_ledger as acceptance_ledger  # noqa: E402
import tune_blockers_v2 as v2  # noqa: E402


class _Lease:
    def ensure_owned(self):
        return None


class _Rss:
    def check(self, _label):
        return 1

    def start(self, _stop_event):
        return None

    def stop(self):
        return None


def _implementation(target="Llt_phys"):
    feature = {"features": ["feature"], "sha256": "f" * 64}
    search = {"sha256": "s" * 64}
    return {
        "sha256": "i" * 64,
        "files": {
            "tune_blockers_v2.py": "1" * 64,
            "train_models.py": "2" * 64,
            "tune_optuna.py": "3" * 64,
            "checkpoint_train.py": "4" * 64,
        },
        "feature_schema": feature,
        "target_configs": {target: {"sha256": "t" * 64}},
        "search_space": search,
        "runtime": {"python_version": "test", "packages": {}},
    }


def _source():
    return {
        "dataset_sha256": "d" * 64,
        "quality_thresholds_sha256": "q" * 64,
        "thresholds": {"minimum_interval_coverage": 0.85},
    }


def _holdout():
    return {
        "sha256": "h" * 64,
        "evaluation_row_count": 10,
    }


def _complete_params():
    return {
        family: {
            target: {"params": {"n_estimators": 10}}
            for target in train_models.V2_BLOCKER_TARGETS
        }
        for family in train_models.V2_HPO_FAMILIES
    }


def _receipt_metadata(stage=200):
    return {
        "dataset_sha256": "d" * 64,
        "source_quality_status_sha256": "q" * 64,
        "parameter_artifact_eligible": True,
        "production_eligible": False,
        "production_model_eligible": False,
        "fea_submission_approved": False,
        "promotion_approved": False,
        "blockers": list(train_models.V2_BLOCKER_TARGETS),
        "families": list(train_models.V2_HPO_FAMILIES),
        "configured_final_trials_per_job": 200,
        "selected_cumulative_trials_per_job": stage,
        "outer_acceptance_holdouts_sha256": "h" * 64,
        "outer_acceptance_policy": {
            "eligible_after_cumulative_trials": 200,
            "eligible": stage == 200,
            "single_use_required": True,
        },
    }


class BlockerHpoV2Tests(unittest.TestCase):
    def test_repository_config_is_new_namespace_and_cannot_claim_data_contract(self):
        path = TRAINING_ROOT / "blocker_hpo_v2_cap_recovery_6151.json"
        config = v2.load_blocker_config(path)
        self.assertEqual(config["trial_stages"], [20, 50, 200])
        self.assertEqual(config["maximum_job_workers"], 8)
        resources = v2.validate_execution_resources(
            config,
            model_threads=1,
            job_workers=8,
            max_model_thread_budget=8,
            max_rss_gb=64,
            job_count=44,
            require_rss=True,
        )
        self.assertEqual(resources["job_workers_effective"], 8)
        self.assertEqual(
            len(config["blockers"]) * len(config["families"])
            * config["trial_stages"][0],
            880,
        )
        with self.assertRaisesRegex(ValueError, "model_threads"):
            v2.validate_execution_resources(
                config,
                model_threads=2,
                job_workers=4,
                max_model_thread_budget=8,
                max_rss_gb=64,
                job_count=44,
                require_rss=True,
            )
        self.assertEqual(config["objective"], v2.OBJECTIVE_NAME)
        self.assertNotIn("data_contract_sha256", config["cohort"])
        self.assertNotIn(8, config["trial_stages"])

        drifted = json.loads(path.read_text(encoding="utf-8"))
        drifted["cohort"]["data_contract_sha256"] = "a" * 64
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "bad.json"
            bad.write_text(json.dumps(drifted), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "unverified"):
                v2.load_blocker_config(bad)

    def test_exact_seed42_evaluation_rows_are_sealed_and_removed(self):
        frame = pd.DataFrame({
            "_strict_valid_full": True,
            "Llt_phys": np.linspace(1.0, 2.0, 100),
            "physics_data_revision": "revision",
            "feature": np.arange(100),
        })
        holdout = v2.seal_final_evaluation_holdout(
            frame, "Llt_phys", "d" * 64
        )
        self.assertEqual(holdout["evaluation_row_count"], 10)
        self.assertEqual(len(set(holdout["absolute_row_positions"])), 10)
        hpo = v2.frame_without_holdout(frame, holdout)
        self.assertEqual(len(hpo), 90)
        self.assertFalse(
            set(holdout["absolute_row_positions"]).intersection(
                set(hpo["feature"].tolist())
            )
        )
        self.assertEqual(
            holdout,
            v2.seal_final_evaluation_holdout(frame, "Llt_phys", "d" * 64),
        )
        changed = v2.seal_final_evaluation_holdout(
            frame, "Llt_phys", "e" * 64
        )
        self.assertNotEqual(holdout["sha256"], changed["sha256"])

    def test_deterministic_proposals_are_unique_and_absolute_ordinal_stable(self):
        first = v2.deterministic_proposal_sequence("identity", "xgboost", 20)
        second = v2.deterministic_proposal_sequence("identity", "xgboost", 50)
        self.assertEqual(first, second[:20])
        self.assertEqual(len({item["sha256"] for item in second}), 50)
        self.assertNotEqual(
            first,
            v2.deterministic_proposal_sequence("other", "xgboost", 20),
        )

    def test_existing_trials_require_every_exact_study_attribute(self):
        proposal = v2.deterministic_proposal_sequence(
            "identity", "extratrees", 1
        )[0]
        proposals = [proposal]
        contract = v2.objective_contract(
            "Llt_phys",
            "extratrees",
            {"min_r2": 0.98},
            _source(),
            _implementation(),
            _holdout(),
            1,
        )
        expected = v2._static_study_attrs(
            "c" * 64, contract, _implementation(), _holdout()
        )
        metrics = {"interval_coverage": 0.9, "r2": 0.99}
        runtime = v2._runtime_params_for_proposal(
            "extratrees", proposal, 1
        )
        evidence = v2.seal_trial_evidence(
            target="Llt_phys",
            family="extratrees",
            proposal=proposal,
            search_params=proposal["params"],
            runtime_params=runtime,
            contract=contract,
            metrics=metrics,
        )
        trial = types.SimpleNamespace(
            number=0,
            state=types.SimpleNamespace(name="COMPLETE"),
            user_attrs={
                "proposal_ordinal": 0,
                "proposal_salt": proposal["salt"],
                "proposal_contract": v2.PROPOSAL_SCHEMA,
                "proposal_sha256": proposal["sha256"],
                "trial_contract_sha256": contract["sha256"],
                "quality_gate_evidence": evidence,
                "quality_gate_evidence_sha256": evidence["sha256"],
                "invocation_id": "test",
            },
            params=proposal["params"],
            system_attrs={},
            value=evidence["quality_gate"]["score"],
        )
        missing = types.SimpleNamespace(
            user_attrs={},
            get_trials=lambda deepcopy=False: [trial],
        )
        with self.assertRaisesRegex(RuntimeError, "provenance mismatch"):
            v2.validate_study_inventory(
                missing,
                expected,
                proposals,
                target="Llt_phys",
                family="extratrees",
                contract=contract,
            )
        exact = types.SimpleNamespace(
            user_attrs=dict(expected),
            get_trials=lambda deepcopy=False: [trial],
        )
        inventory = v2.validate_study_inventory(
            exact,
            expected,
            proposals,
            target="Llt_phys",
            family="extratrees",
            contract=contract,
        )
        self.assertEqual(inventory["completed"], 1)
        trial.value += 1.0
        with self.assertRaisesRegex(RuntimeError, "value differs"):
            v2.validate_study_inventory(
                exact,
                expected,
                proposals,
                target="Llt_phys",
                family="extratrees",
                contract=contract,
            )
        trial.value = evidence["quality_gate"]["score"]
        evidence["runtime_params"]["n_jobs"] = 2
        with self.assertRaisesRegex(RuntimeError, "fingerprint"):
            v2.validate_study_inventory(
                exact,
                expected,
                proposals,
                target="Llt_phys",
                family="extratrees",
                contract=contract,
            )
        evidence["runtime_params"]["n_jobs"] = 1
        tampered = json.loads(json.dumps(evidence))
        tampered["quality_gate"]["score"] += 1.0
        unsigned = {
            key: value for key, value in tampered.items() if key != "sha256"
        }
        tampered["sha256"] = v2._canonical_sha256(unsigned)
        trial.user_attrs["quality_gate_evidence"] = tampered
        trial.user_attrs["quality_gate_evidence_sha256"] = tampered["sha256"]
        trial.value = tampered["quality_gate"]["score"]
        with self.assertRaisesRegex(RuntimeError, "cannot be reproduced"):
            v2.validate_study_inventory(
                exact,
                expected,
                proposals,
                target="Llt_phys",
                family="extratrees",
                contract=contract,
            )
        trial.user_attrs["quality_gate_evidence"] = evidence
        trial.user_attrs["quality_gate_evidence_sha256"] = evidence["sha256"]
        trial.value = evidence["quality_gate"]["score"]
        trial.state = types.SimpleNamespace(name="RUNNING")
        with self.assertRaisesRegex(RuntimeError, "non-resumable"):
            v2.validate_study_inventory(
                exact,
                expected,
                proposals,
                target="Llt_phys",
                family="extratrees",
                contract=contract,
            )

    def test_real_optuna_stages_append_deterministic_unique_proposals(self):
        metrics = {"interval_coverage": 0.9, "r2": 0.99}
        common = dict(
            target="Llt_phys",
            family="extratrees",
            frame=object(),
            features=["feature"],
            limits={"min_r2": 0.98},
            source=_source(),
            implementation=_implementation(),
            holdout=_holdout(),
            model_threads=1,
            sample_weight_column="sample_weight",
            minimum_eligible_rows=4,
            config_sha256="c" * 64,
            stop_event=threading.Event(),
            lease=_Lease(),
            rss_guard=_Rss(),
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            v2.train_models,
            "train_target",
            return_value=({"models": []}, metrics),
        ) as train:
            first = v2.tune_one(
                requested_total_trials=2,
                study_root=directory,
                invocation_id="first",
                **common,
            )
            second = v2.tune_one(
                requested_total_trials=4,
                study_root=directory,
                invocation_id="second",
                **common,
            )
            specification = v2.build_study_spec(
                "Llt_phys",
                "extratrees",
                4,
                {"min_r2": 0.98},
                _source(),
                _implementation(),
                _holdout(),
                model_threads=1,
                study_root=directory,
                config_sha256="c" * 64,
            )
            preaudit = v2.preaudit_existing_studies(
                directory, [specification]
            )
            self.assertEqual(preaudit["existing_study_count"], 1)
            foreign = Path(directory) / "v2__foreign.sqlite3"
            foreign.write_bytes(b"foreign")
            with self.assertRaisesRegex(RuntimeError, "foreign or stale"):
                v2.preaudit_existing_studies(directory, [specification])
        self.assertEqual(train.call_count, 4)
        self.assertEqual(first["study"]["completed_trials_added"], 2)
        self.assertEqual(second["study"]["completed_trials_before"], 2)
        self.assertEqual(second["study"]["completed_trials_after"], 4)
        self.assertEqual(len(second["study"]["stage_history"]), 2)
        self.assertIn("blocker-v2", second["study"]["study_name"])

    def test_cross_process_run_lease_rejects_a_second_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            with v2.RunLease(
                directory, lease_seconds=5, identity={"run": 1}
            ):
                with self.assertRaisesRegex(RuntimeError, "already leased"):
                    with v2.RunLease(
                        directory, lease_seconds=5, identity={"run": 2}
                    ):
                        pass
            with v2.RunLease(
                directory, lease_seconds=5, identity={"run": 3}
            ) as lease:
                lease.ensure_owned()

    def test_v2_study_root_rejects_physical_v1_database_colocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v2.require_local_sqlite_root(root)
            (root / "Llt_phys__lightgbm__legacy.sqlite3").write_bytes(b"v1")
            with self.assertRaisesRegex(RuntimeError, "exploratory/unknown"):
                v2.require_v2_only_study_root(root)
            (root / "Llt_phys__lightgbm__legacy.sqlite3").unlink()
            (root / "v2__Llt_phys__lightgbm__new.sqlite3").write_bytes(b"v2")
            v2.require_v2_only_study_root(root)

    def test_failure_cancels_pending_jobs_instead_of_draining_the_queue(self):
        calls = []

        def fail_first(target, *_args, **_kwargs):
            calls.append(target)
            if target == "fail":
                raise RuntimeError("boom")
            return {
                "params": {},
                "quality_gate_objective_score": 1.0,
                "quality_gate_evidence": {},
                "objective_contract": {"sha256": "x"},
                "study": {"completed_trials_after": 1},
            }

        jobs = [("fail", "extratrees")] + [
            (f"pending-{index}", "extratrees") for index in range(20)
        ]
        source = {"thresholds": {"targets": {
            target: {} for target, _ in jobs
        }}}
        holdouts = {
            target: {"absolute_row_positions": [], "evaluation_row_count": 0}
            for target, _ in jobs
        }
        with mock.patch.object(
            v2, "tune_one", side_effect=fail_first
        ), mock.patch.object(
            v2, "frame_without_holdout", return_value=object()
        ), mock.patch.object(
            v2, "build_study_spec", return_value={}
        ), mock.patch.object(
            v2,
            "preaudit_existing_studies",
            return_value={"audited": []},
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                v2.run_blocker_jobs(
                    jobs, 1, object(), ["feature"], source, {}, holdouts,
                    model_threads=1,
                    job_workers=1,
                    max_model_thread_budget=1,
                    sample_weight_column=None,
                    minimum_eligible_rows=1,
                    study_root="unused",
                    config_sha256="c" * 64,
                    lease=_Lease(),
                    rss_guard=_Rss(),
                    invocation_id="run",
                    invocation_resources={"job_workers_effective": 1},
                )
        self.assertLess(len(calls), len(jobs))

    def test_parameter_shape_is_consumed_by_full_training_with_thread_override(self):
        tuned = {
            "xgboost": {
                "Llt_phys": {
                    "params": {"max_depth": 5, "n_jobs": 99},
                    "quality_gate_objective_score": 0.5,
                }
            }
        }
        params = train_models.family_params_for_target(
            "Llt_phys", tuned, model_threads=1
        )
        self.assertEqual(params["xgboost"]["max_depth"], 5)
        self.assertEqual(params["xgboost"]["n_jobs"], 1)
        self.assertIn("n_estimators", params["lightgbm"])
        with self.assertRaisesRegex(ValueError, "invalid tuning target"):
            train_models.family_params_for_target(
                "Llt_phys",
                {"xgboost": {"Llt_phys": {"params": "not-a-mapping"}}},
                model_threads=1,
            )

    def test_v2_params_consumer_authenticates_receipt_and_complete_inventory(self):
        params = _complete_params()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = v2._publish_generation(
                root / "artifacts",
                params,
                {"schema_version": v2.HOLDOUT_SCHEMA, "targets": {}},
                _receipt_metadata(),
                root / "result.json",
            )
            loaded, provenance = train_models.load_tuning_params(
                result["params_path"], result["receipt_path"]
            )
            self.assertEqual(loaded, params)
            self.assertEqual(provenance["mode"], "authenticated_blocker_hpo_v2")

            rejected_receipt = json.loads(
                Path(result["receipt_path"]).read_text(encoding="utf-8")
            )
            rejected_receipt["metadata"]["parameter_artifact_eligible"] = False
            rejected_unsigned = {
                key: value for key, value in rejected_receipt.items()
                if key != "sha256"
            }
            rejected_receipt["sha256"] = train_models._canonical_sha256(
                rejected_unsigned
            )
            rejected_path = root / "rejected.receipt.json"
            rejected_path.write_text(json.dumps(rejected_receipt), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "flag mismatch"):
                train_models.load_tuning_params(
                    result["params_path"], rejected_path
                )

            broken_payload = json.loads(
                Path(result["params_path"]).read_text(encoding="utf-8")
            )
            broken_payload["xgboost"].pop("Llt_phys")
            broken = root / "broken.json"
            broken.write_text(json.dumps(broken_payload), encoding="utf-8")
            receipt = json.loads(Path(result["receipt_path"]).read_text(encoding="utf-8"))
            receipt["params_sha256"] = train_models._sha256(broken)
            receipt["params_canonical_sha256"] = train_models._canonical_sha256(
                broken_payload
            )
            unsigned = {key: value for key, value in receipt.items() if key != "sha256"}
            receipt["sha256"] = train_models._canonical_sha256(unsigned)
            bad_receipt = root / "broken.receipt.json"
            bad_receipt.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "target inventory"):
                train_models.load_tuning_params(broken, bad_receipt)
            with self.assertRaisesRegex(RuntimeError, "explicit params receipt"):
                train_models.load_tuning_params(result["params_path"])

    def test_legacy_params_are_not_misidentified_and_nonmapping_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "legacy.json"
            legacy.write_text(json.dumps({"xgboost": {"max_depth": 5}}))
            loaded, provenance = train_models.load_tuning_params(legacy)
            self.assertEqual(loaded["xgboost"]["max_depth"], 5)
            self.assertEqual(provenance["mode"], "legacy_explicit_no_v2_receipt")
            invalid = root / "invalid.json"
            invalid.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "top-level mapping"):
                train_models.load_tuning_params(invalid)

    def test_outer_acceptance_claim_is_final_stage_and_single_use(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final = v2._publish_generation(
                root / "final-artifacts",
                _complete_params(),
                {"schema_version": v2.HOLDOUT_SCHEMA, "targets": {}},
                _receipt_metadata(stage=200),
                root / "final-result.json",
            )
            ledger = root / "acceptance.claim.json"
            claim = acceptance_ledger.claim_outer_acceptance_once(
                final["receipt_path"], ledger, "candidate-retrain-1"
            )
            self.assertEqual(claim["status"], "claimed_no_result")
            with self.assertRaisesRegex(RuntimeError, "already claimed"):
                acceptance_ledger.claim_outer_acceptance_once(
                    final["receipt_path"], ledger, "candidate-retrain-2"
                )

            early = v2._publish_generation(
                root / "early-artifacts",
                _complete_params(),
                {"schema_version": v2.HOLDOUT_SCHEMA, "targets": {}},
                _receipt_metadata(stage=20),
                root / "early-result.json",
            )
            with self.assertRaisesRegex(RuntimeError, "not eligible"):
                acceptance_ledger.claim_outer_acceptance_once(
                    early["receipt_path"], root / "early.claim.json", "early"
                )

    def test_live_status_is_atomic_and_reports_trials_jobs_and_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            status = v2.LiveStatus(
                path,
                config_sha256="c" * 64,
                dataset_sha256="d" * 64,
                stage=20,
                jobs=[("Llt_phys", "extratrees")],
            )
            status.preflight_complete({
                "ready": True,
                "strict_full_rows": 6151,
                "job_count": 1,
                "planned_model_fits_from_clean_studies": 100,
                "implementation": {"sha256": "i" * 64},
            })
            status.apply_preaudit({
                "audited": [{
                    "target": "Llt_phys",
                    "family": "extratrees",
                    "completed": 2,
                    "waiting": 0,
                }],
            })
            status.trial_started("Llt_phys", "extratrees", 2)
            status.trial_finished("Llt_phys", "extratrees", 2, "COMPLETE")
            status.job_completed("Llt_phys", "extratrees")
            status.terminal_success({
                "generation_id": "generation",
                "generation_path": "path",
                "params_path": "params",
                "receipt_path": "receipt",
                "receipt_sha256": "r" * 64,
                "receipt_canonical_sha256": "s" * 64,
            })
            written = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(written["phase"], "succeeded")
            self.assertEqual(written["trials"]["total"], 20)
            self.assertEqual(written["trials"]["complete"], 3)
            self.assertEqual(written["trials"]["running"], 0)
            self.assertEqual(written["result"]["receipt_sha256"], "r" * 64)
            self.assertFalse(written["production_eligible"])
            failed_path = Path(directory) / "failed-status.json"
            failed = v2.LiveStatus(
                failed_path,
                config_sha256="c" * 64,
                dataset_sha256="d" * 64,
                stage=20,
                jobs=[("Llt_phys", "extratrees")],
            )
            failed.terminal_failure(RuntimeError("boom"))
            failed_document = json.loads(
                failed_path.read_text(encoding="utf-8")
            )
            self.assertEqual(failed_document["phase"], "failed")
            self.assertEqual(failed_document["error"]["message"], "boom")

    def test_v2_publisher_has_consistent_schema_and_holdout_artifact(self):
        metadata = {
            "dataset_sha256": "d" * 64,
            "source_quality_status_sha256": "q" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = v2._publish_generation(
                root / "artifacts",
                {"extratrees": {}},
                {"schema_version": v2.HOLDOUT_SCHEMA, "targets": {}},
                metadata,
                root / "result.json",
            )
            written = json.loads((root / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["schema_version"], v2.RESULT_SCHEMA)
            self.assertEqual(written["schema_version"], v2.RESULT_SCHEMA)
            self.assertTrue(Path(result["params_path"]).is_file())
            self.assertTrue(Path(result["outer_acceptance_holdouts_path"]).is_file())
            self.assertTrue(Path(result["receipt_path"]).is_file())
            receipt = json.loads(Path(result["receipt_path"]).read_text(encoding="utf-8"))
            unsigned = {key: value for key, value in receipt.items() if key != "sha256"}
            self.assertEqual(receipt["sha256"], v2._canonical_sha256(unsigned))
            self.assertEqual(
                receipt["params_sha256"], v2._sha256(result["params_path"])
            )

    def test_rss_guard_fails_closed(self):
        guard = v2.RssGuard(1e-9)
        with self.assertRaisesRegex(RuntimeError, "RSS guard exceeded"):
            guard.check("unit-test")

    def test_rss_watchdog_requests_cooperative_mid_fit_cancellation(self):
        guard = v2.RssGuard(1.0)
        cancellation = threading.Event()
        with mock.patch.object(
            guard,
            "_observed_bytes",
            return_value=guard.maximum_bytes + 1,
        ):
            guard.start(cancellation, interval_seconds=0.01)
            self.assertTrue(cancellation.wait(1.0))
            with self.assertRaisesRegex(RuntimeError, "RSS guard exceeded"):
                guard.check("mid-fit")
            guard.stop()


if __name__ == "__main__":
    unittest.main()
