import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

from filelock import FileLock, Timeout
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "training"))

from training.checkpoint_orchestrator import (  # noqa: E402
    _ensure_identity,
    _load_state,
    reconcile_completed,
    training_commands,
)
import training.checkpoint_orchestrator as checkpoint_orchestrator  # noqa: E402
from training.model_quality_gate import evaluate_generation  # noqa: E402
import training.train_models as train_models  # noqa: E402
from training.predictor import EnsemblePredictor  # noqa: E402
from monitoring.readers import ArtifactService  # noqa: E402
import al_driver  # noqa: E402


class ConstantModel:
    def predict(self, frame):
        return np.ones(len(frame), dtype=float)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _candidate(root, run_id="candidate", registry=None):
    root = Path(root)
    registry = Path(registry or root / "registry")
    dataset = root / f"snapshot-{run_id}.parquet"
    dataset.write_bytes(f"stable-{run_id}".encode("utf-8"))
    dataset_sha256 = _sha256(dataset)
    generation = registry / "generations" / run_id
    target_dir = generation / "Llt_phys"
    target_dir.mkdir(parents=True)
    metrics = {
        "r2": 0.99,
        "mape_pct": 1.0,
        "interval_coverage": 0.90,
    }
    bundle = {
        "models": [("constant", ConstantModel())],
        "features": ["N1_main"],
        "transform": None,
        "q90": 1.0,
        "target": "Llt_phys",
        "metrics": metrics,
        "training_run_id": run_id,
        "dataset_sha256": dataset_sha256,
    }
    with open(target_dir / "models.pkl", "wb") as handle:
        pickle.dump(bundle, handle)
    meta = {key: value for key, value in bundle.items() if key != "models"}
    (target_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    report = {
        "schema_version": 2,
        "training_run_id": run_id,
        "dataset_sha256": dataset_sha256,
        "profile_sha256": "profile-sha",
        "strict_full_rows": 3000,
        "features": ["N1_main"],
        "targets": ["Llt_phys"],
        "report": {"Llt_phys": metrics},
        "artifacts": {
            "Llt_phys/models.pkl": _sha256(target_dir / "models.pkl"),
            "Llt_phys/meta.json": _sha256(target_dir / "meta.json"),
        },
    }
    report_path = generation / "train_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    thresholds = {
        "minimum_strict_full_rows": 3000,
        "minimum_interval_coverage": 0.85,
        "targets": {
            "Llt_phys": {"min_r2": 0.98, "max_mape_pct": 2.0}
        },
    }
    quality = evaluate_generation(
        str(registry), str(generation), str(dataset), thresholds
    )
    if not quality["passed"]:
        raise AssertionError(quality["reasons"])
    return {
        "registry": str(registry),
        "dataset": str(dataset),
        "generation": str(generation),
        "relative": f"generations/{run_id}",
        "quality": quality,
        "thresholds": thresholds,
    }


def _promote(candidate, token=None):
    token = token or train_models.registry_pointer_token(candidate["registry"])
    candidate["quality"].setdefault("checkpoint", 3000)
    return train_models.promote_generation(
        candidate["registry"], candidate["generation"], candidate["quality"],
        dataset=candidate["dataset"], profile_sha256="profile-sha",
        thresholds_sha256=candidate["quality"]["thresholds_sha256"],
        expected_pointer=token,
    )


class RegistryPublicationTests(unittest.TestCase):
    def test_candidate_is_invisible_until_passing_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _candidate(tmp)
            registry = Path(candidate["registry"])
            self.assertFalse((registry / "current.json").exists())
            self.assertFalse((registry / "Llt_phys" / "models.pkl").exists())
            with self.assertRaises(FileNotFoundError):
                EnsemblePredictor.load("Llt_phys", registry=str(registry))

            failed = dict(candidate["quality"], passed=False)
            with self.assertRaisesRegex(RuntimeError, "failed quality"):
                train_models.promote_generation(
                    str(registry), candidate["generation"], failed,
                    dataset=candidate["dataset"],
                    profile_sha256="profile-sha",
                    thresholds_sha256=failed["thresholds_sha256"],
                    expected_pointer=train_models.registry_pointer_token(registry),
                )
            self.assertFalse((registry / "current.json").exists())

            _promote(candidate)
            predictor = EnsemblePredictor.load(
                "Llt_phys", registry=str(registry)
            )
            mu, _ = predictor.predict_mu_sigma(pd.DataFrame({"N1_main": [1]}))
            self.assertEqual(mu.tolist(), [1.0])

    def test_no_pointer_never_falls_back_to_rejected_flat_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp, "registry")
            flat = registry / "Llt_phys"
            flat.mkdir(parents=True)
            with open(flat / "models.pkl", "wb") as handle:
                pickle.dump({"training_run_id": "rejected"}, handle)
            with self.assertRaises(FileNotFoundError):
                EnsemblePredictor.load("Llt_phys", registry=str(registry))
            train_models.restore_active_generation(str(registry), None)
            self.assertFalse(flat.exists())

    def test_pointer_replace_is_last_commit_and_failure_preserves_old_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = _candidate(tmp, "old")
            _promote(old)
            old_pointer = Path(old["registry"], "current.json").read_bytes()
            new = _candidate(tmp, "new", registry=old["registry"])
            token = train_models.registry_pointer_token(old["registry"])
            original_atomic_json = train_models._atomic_json

            def fail_current(value, path):
                if os.path.basename(path) == "current.json":
                    raise OSError("simulated process death before pointer replace")
                return original_atomic_json(value, path)

            with patch.object(train_models, "_atomic_json", side_effect=fail_current):
                with self.assertRaisesRegex(OSError, "simulated process death"):
                    train_models.promote_generation(
                        old["registry"], new["generation"], new["quality"],
                        dataset=new["dataset"], profile_sha256="profile-sha",
                        thresholds_sha256=new["quality"]["thresholds_sha256"],
                        expected_pointer=token,
                    )
            self.assertEqual(
                Path(old["registry"], "current.json").read_bytes(), old_pointer
            )
            self.assertEqual(
                train_models.load_active_generation(old["registry"])["report"][
                    "training_run_id"
                ],
                "old",
            )

    def test_stale_evaluated_candidate_cannot_overwrite_newer_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = _candidate(tmp, "first")
            second = _candidate(tmp, "second", registry=first["registry"])
            initial = train_models.registry_pointer_token(first["registry"])
            _promote(first, initial)
            with self.assertRaisesRegex(RuntimeError, "pointer changed"):
                _promote(second, initial)
            active = train_models.load_active_generation(first["registry"])
            self.assertEqual(active["report"]["training_run_id"], "first")

    def test_promotion_binds_profile_and_threshold_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _candidate(tmp)
            for key in ("profile_sha256", "thresholds_sha256"):
                bad = dict(candidate["quality"])
                bad[key] = "wrong"
                with self.assertRaisesRegex(RuntimeError, key):
                    train_models.promote_generation(
                        candidate["registry"], candidate["generation"], bad,
                        dataset=candidate["dataset"],
                        profile_sha256="profile-sha",
                        thresholds_sha256=candidate["quality"]["thresholds_sha256"],
                        expected_pointer=train_models.registry_pointer_token(
                            candidate["registry"]
                        ),
                    )
                self.assertFalse(
                    Path(candidate["registry"], "current.json").exists()
                )


class RegistryWriterTests(unittest.TestCase):
    def test_direct_candidate_builder_owns_writer_lock_and_does_not_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "snapshot.parquet"
            dataset.write_bytes(b"snapshot")
            profile = root / "profile.json"
            profile.write_text(
                json.dumps({"param_overrides": {"full_model": 0}}),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                registry=str(root / "registry"), dataset=str(dataset),
                profile=str(profile), weight_col="sample_weight", min_rows=1,
            )
            frame = pd.DataFrame({"N1_main": [1.0], "Llt_phys": [2.0]})
            metrics = {
                "r2": 0.99, "mape_pct": 1.0, "p90_ape_pct": 1.5,
                "interval_coverage": 0.9,
            }
            bundle = {
                "models": [("constant", ConstantModel())],
                "features": ["N1_main"], "transform": None, "q90": 1.0,
                "target": "Llt_phys", "metrics": metrics,
            }
            with patch.object(
                train_models, "train_target", return_value=(bundle, metrics)
            ):
                result = train_models.build_candidate(
                    args, frame, ["N1_main"], 1, ["Llt_phys"],
                    lambda target: {}, lock_timeout=0,
                )
            self.assertTrue(Path(result["generation_path"]).is_dir())
            self.assertFalse(Path(args.registry, "current.json").exists())
            self.assertFalse(Path(args.registry, "Llt_phys").exists())

            with FileLock(args.registry + ".training.lock", timeout=0):
                with patch.object(train_models, "_build_candidate") as unlocked:
                    with self.assertRaises(Timeout):
                        train_models.build_candidate(
                            args, frame, ["N1_main"], 1, ["Llt_phys"],
                            lambda target: {}, lock_timeout=0,
                        )
                    unlocked.assert_not_called()


class CheckpointRecoveryTests(unittest.TestCase):
    def _completion(self, candidate, pointer):
        metrics_result = Path(candidate["dataset"] + ".metrics.json")
        profile_path = os.path.abspath(candidate["dataset"] + ".profile.json")
        metrics_result.write_text(json.dumps({
            "checkpoint": 3000,
            "dataset": os.path.abspath(candidate["dataset"]),
            "dataset_sha256": _sha256(candidate["dataset"]),
            "profile": profile_path,
            "profile_sha256": pointer["profile_sha256"],
            "strict_full_rows": 3000,
            "metrics": [{"target": "Llt_phys", "r2": 0.99}],
        }), encoding="utf-8")
        return {
            "kind": "accepted_generation",
            "threshold": 3000,
            "actual_strict_full_rows": 3000,
            "snapshot": candidate["dataset"],
            "snapshot_sha256": _sha256(candidate["dataset"]),
            "metrics_result": str(metrics_result),
            "metrics_result_sha256": _sha256(metrics_result),
            "profile_path": profile_path,
            "training_run_id": pointer["training_run_id"],
            "generation": pointer["generation"],
            "generation_report_sha256": pointer["generation_report_sha256"],
            "quality_gate_sha256": pointer["quality_gate_sha256"],
            "profile_sha256": pointer["profile_sha256"],
            "thresholds_sha256": pointer["thresholds_sha256"],
            "activation_minimum_strict_full_rows": 3000,
            "quality_passed": True,
        }

    def test_completed_threshold_requires_snapshot_and_valid_active_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _candidate(tmp)
            pointer = _promote(candidate)
            state = {"schema_version": 2, "completed": [
                self._completion(candidate, pointer)
            ], "attempts": []}
            valid, issues = reconcile_completed(state, candidate["registry"])
            self.assertEqual([item["threshold"] for item in valid], [3000])
            self.assertEqual(issues, [])

            Path(candidate["registry"], "current.json").write_text(
                "{corrupt", encoding="utf-8"
            )
            valid, issues = reconcile_completed(state, candidate["registry"])
            self.assertEqual(valid, [])
            self.assertTrue(any(
                "active_generation_invalid" in issue["reason"]
                for issue in issues
            ))

    def test_snapshot_fingerprint_mismatch_makes_threshold_due(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _candidate(tmp)
            pointer = _promote(candidate)
            state = {"schema_version": 2, "completed": [
                self._completion(candidate, pointer)
            ], "attempts": []}
            Path(candidate["dataset"]).write_bytes(b"tampered")
            valid, issues = reconcile_completed(state, candidate["registry"])
            self.assertEqual(valid, [])
            self.assertEqual(issues[0]["reason"], "snapshot_fingerprint_mismatch")

    def test_completion_threshold_is_bound_to_accepted_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _candidate(tmp)
            pointer = _promote(candidate)
            item = self._completion(candidate, pointer)
            item["threshold"] = 4000
            state = {"schema_version": 2, "completed": [item], "attempts": []}
            valid, issues = reconcile_completed(state, candidate["registry"])
            self.assertEqual(valid, [])
            self.assertIn("threshold", issues[0]["reason"])

    def test_newer_accepted_al_generation_supersedes_without_erasing_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = _candidate(tmp, "checkpoint")
            pointer = _promote(checkpoint)
            item = self._completion(checkpoint, pointer)
            al_candidate = _candidate(
                tmp, "al-round", registry=checkpoint["registry"]
            )
            train_models.promote_generation(
                al_candidate["registry"], al_candidate["generation"],
                al_candidate["quality"], dataset=al_candidate["dataset"],
                profile_sha256="profile-sha",
                thresholds_sha256=al_candidate["quality"]["thresholds_sha256"],
                expected_pointer=train_models.registry_pointer_token(
                    al_candidate["registry"]
                ),
            )
            identity = {
                "profile_sha256": "profile-sha",
                "thresholds_sha256": pointer["thresholds_sha256"],
                "activation_minimum_strict_full_rows": 3000,
            }
            valid, issues = reconcile_completed(
                {"completed": [item]}, checkpoint["registry"], identity
            )
            self.assertEqual(valid, [item])
            self.assertEqual(issues, [])

    def test_schema_one_completion_migrates_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp, "checkpoint_state.json")
            state_path.write_text(json.dumps({
                "schema_version": 1,
                "completed": [{"threshold": 3000}],
                "attempts": [],
            }), encoding="utf-8")
            state = _load_state(str(state_path))
            self.assertEqual(state["schema_version"], 2)
            self.assertEqual(state["completed"], [])
            self.assertEqual(
                state["invalidated_completed"][0]["reason"],
                "schema_v1_completion_has_no_cryptographic_evidence",
            )

    def test_partial_identity_upgrade_invalidates_prior_completions(self):
        state = {
            "schema_version": 2,
            "identity": {"profile_sha256": "profile"},
            "completed": [{"threshold": 3000}],
            "attempts": [],
        }
        identity = {
            "profile_sha256": "profile",
            "solver_revision": "a" * 40,
            "library_revision": "b" * 40,
            "quality_contract_sha256": "contract",
        }
        _ensure_identity(state, identity)
        self.assertEqual(state["completed"], [])
        self.assertIn(
            "identity_schema_changed",
            state["invalidated_completed"][0]["reason"],
        )

    def test_metrics_only_checkpoint_does_not_require_or_publish_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp, "snapshot.parquet")
            snapshot.write_bytes(b"metrics-snapshot")
            metrics_result = Path(tmp, "metrics.json")
            metrics_result.write_text(json.dumps({
                "checkpoint": 500,
                "dataset": os.path.abspath(snapshot),
                "dataset_sha256": _sha256(snapshot),
                "profile": os.path.abspath(Path(tmp, "profile.json")),
                "profile_sha256": "profile",
                "strict_full_rows": 2000,
                "metrics": [{"target": "Llt_phys", "r2": 0.5}],
            }), encoding="utf-8")
            item = {
                "kind": "metrics_only", "threshold": 500,
                "actual_strict_full_rows": 2000,
                "snapshot": str(snapshot), "snapshot_sha256": _sha256(snapshot),
                "metrics_result": str(metrics_result),
                "metrics_result_sha256": _sha256(metrics_result),
                "profile_path": os.path.abspath(Path(tmp, "profile.json")),
                "profile_sha256": "profile", "thresholds_sha256": "thresholds",
                "activation_minimum_strict_full_rows": 3000,
            }
            valid, issues = reconcile_completed(
                {"completed": [item]}, str(Path(tmp, "registry"))
            )
            self.assertEqual(valid, [item])
            self.assertEqual(issues, [])
            self.assertFalse(Path(tmp, "registry", "current.json").exists())
            mutated = dict(item, threshold=1000)
            valid, issues = reconcile_completed(
                {"completed": [mutated]}, str(Path(tmp, "registry"))
            )
            self.assertEqual(valid, [])
            self.assertEqual(
                issues[0]["reason"], "checkpoint_metrics_checkpoint_mismatch"
            )

    def test_absolute_profile_is_identical_in_every_child_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = os.path.abspath(os.path.join(tmp, "profile.json"))
            commands = training_commands(
                "snapshot", "curve", "registry", 200, profile,
                3000, "metrics", "candidate",
            )
            supplied = [
                command[command.index("--profile") + 1] for command in commands
            ]
            self.assertEqual(supplied, [profile, profile])
            self.assertEqual(len(training_commands(
                "snapshot", "curve", "registry", 200, profile, 500, "metrics"
            )), 1)
            with self.assertRaisesRegex(ValueError, "absolute"):
                training_commands(
                    "snapshot", "curve", "registry", 200,
                    "relative/profile.json", 3000, "metrics", "candidate",
                )

    def test_early_checkpoint_executes_metrics_only_without_model_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root / "profile.json"
            profile_data = {"param_overrides": {"full_model": 0}}
            profile.write_text(json.dumps(profile_data), encoding="utf-8")
            thresholds_path = root / "thresholds.json"
            thresholds_path.write_text(json.dumps({
                "minimum_strict_full_rows": 3000,
                "minimum_interval_coverage": 0.85,
                "targets": {},
            }), encoding="utf-8")
            dataset = root / "train.parquet"
            output = root / "training"
            strict = pd.DataFrame({
                "project_name": [f"simulation-{index}" for index in range(500)],
                "_strict_valid_em": [True] * 500,
                "_strict_valid_thermal": [True] * 500,
                "_strict_valid_full": [True] * 500,
                "_strict_invalid_reasons": [""] * 500,
            })
            profile_sha256 = hashlib.sha256(json.dumps(
                profile_data, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")).hexdigest()
            commands = []

            def fake_run(command):
                commands.append(command)
                self.assertIn("checkpoint_train.py", command[1])
                result_path = command[command.index("--result-json") + 1]
                snapshot_path = command[command.index("--dataset") + 1]
                checkpoint = int(command[command.index("--checkpoint") + 1])
                Path(result_path).parent.mkdir(parents=True, exist_ok=True)
                Path(result_path).write_text(json.dumps({
                    "checkpoint": checkpoint,
                    "dataset": os.path.abspath(snapshot_path),
                    "dataset_sha256": _sha256(snapshot_path),
                    "profile": os.path.abspath(profile),
                    "profile_sha256": profile_sha256,
                    "strict_full_rows": 500,
                    "metrics": [{"target": "Llt_phys", "r2": 0.5}],
                }), encoding="utf-8")

            argv = [
                "checkpoint_orchestrator.py",
                "--runtime-root", str(root),
                "--dataset", str(dataset),
                "--output-root", str(output),
                "--profile", str(profile),
                "--thresholds", str(thresholds_path),
                "--execute",
            ]
            with patch.object(sys, "argv", argv), patch.object(
                checkpoint_orchestrator, "inspect_dataset",
                return_value=(strict, strict, strict, {}),
            ), patch.object(checkpoint_orchestrator, "_run", side_effect=fake_run):
                checkpoint_orchestrator.main()
            state = json.loads(
                (output / "checkpoint_state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(commands), 1)
            self.assertEqual(state["completed"][0]["kind"], "metrics_only")
            self.assertFalse((output / "registry" / "current.json").exists())


class ActiveLearningPinTests(unittest.TestCase):
    def test_every_downstream_stage_rechecks_pinned_training_evidence(self):
        downstream = (
            "optimize", "select", "submit", "wait", "ingest", "check",
            "final_select", "fine_submit", "fine_wait", "final_report",
        )
        for name in downstream:
            source = inspect.getsource(getattr(al_driver, f"stage_{name}"))
            self.assertIn("_assert_training_invariants(st)", source, name)

    def test_deleted_or_replaced_snapshot_and_generation_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _candidate(tmp)
            pointer = _promote(candidate)
            quality_path = Path(tmp, "quality.json")
            quality_path.write_text(
                json.dumps(candidate["quality"]), encoding="utf-8"
            )
            state = {
                "training_dataset": candidate["dataset"],
                "training_run_id": pointer["training_run_id"],
                "training_generation": candidate["generation"],
                "model_quality_snapshot_path": str(quality_path),
                "training_strict_full_rows": 3000,
            }
            original_snapshot = Path(candidate["dataset"]).read_bytes()
            model_path = Path(candidate["generation"], "Llt_phys", "models.pkl")
            original_model = model_path.read_bytes()
            with patch.object(
                al_driver, "REGISTRY", candidate["registry"]
            ), patch.object(al_driver, "_assert_runtime_training_invariants"):
                al_driver._assert_training_invariants(state)
                Path(candidate["dataset"]).write_bytes(b"replaced")
                with self.assertRaisesRegex(RuntimeError, "dataset_sha256"):
                    al_driver._assert_training_invariants(state)
                Path(candidate["dataset"]).write_bytes(original_snapshot)
                model_path.write_bytes(b"replaced-generation")
                with self.assertRaisesRegex(RuntimeError, "fingerprint"):
                    al_driver._assert_training_invariants(state)
                model_path.write_bytes(original_model)


class MonitoringRegistryTests(unittest.TestCase):
    def test_corrupt_pointer_never_reuses_cached_last_good_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp, "training", "registry")
            candidate = _candidate(tmp, registry=registry)
            _promote(candidate)
            service = ArtifactService(
                Path(tmp), record_runtime=False
            )
            first = service.models(current_data_count=3000)
            self.assertGreater(first["trained_count"], 0)
            (registry / "current.json").write_text("{broken", encoding="utf-8")
            second = service.models(current_data_count=3000)
            self.assertEqual(second["trained_count"], 0)
            self.assertTrue(any("pointer" in item for item in second["warnings"]))


class LoopSeparationTests(unittest.TestCase):
    def test_collector_and_trainer_are_independent_managed_roots(self):
        collector = (HERE / "campaign" / "auto_collect_loop.sh").read_text(
            encoding="utf-8"
        )
        trainer = (HERE / "campaign" / "auto_checkpoint_loop.sh").read_text(
            encoding="utf-8"
        )
        relaunch = (HERE / "campaign" / "relaunch.sh").read_text(
            encoding="utf-8"
        )
        manager = (HERE / "campaign" / "manage_campaign_loops.ps1").read_text(
            encoding="utf-8"
        )
        checkpoint_train = (HERE / "training" / "checkpoint_train.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("checkpoint_orchestrator.py", collector)
        self.assertIn("checkpoint_orchestrator.py", trainer)
        self.assertIn("--solver-revision", trainer)
        self.assertIn("--library-revision", trainer)
        self.assertIn('MFT_SOLVER_REVISION="$SOLVER_REVISION"', relaunch)
        self.assertIn('MFT_LIBRARY_REVISION="$LIBRARY_REVISION"', relaunch)
        self.assertIn("auto_collect_loop.sh", relaunch)
        self.assertIn("auto_checkpoint_loop.sh", relaunch)
        self.assertIn("trainer_roots=", manager)
        self.assertIn("ActiveTrainers", manager)
        self.assertIn("ActiveTrainerControllers", manager)
        self.assertNotIn("$snapshot.ActiveTrainers.Count -gt 1", manager)
        self.assertIn("checkpoint_train|train_models", manager)
        self.assertIn('FileLock(args.curve_csv + ".lock"', checkpoint_train)


if __name__ == "__main__":
    unittest.main()
