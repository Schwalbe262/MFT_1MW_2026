import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "training"))
sys.path.insert(0, str(HERE / "verify"))

from quality_contract import annotate_validity, validate_record
from training.checkpoint_orchestrator import checkpoint_sequence, due_with_backoff
from training.checkpoint_train import feature_columns, filter_valid_training_rows
from training.model_quality_gate import evaluate_registry
from training.train_models import capture_active_generation, restore_active_generation
from verify import scheduler_client
import scheduler_client as runtime_scheduler_client
from verify.finalize import (
    INSULATION_COLUMNS, _candidate_digest, write_final_artifacts,
)
from test_al_integrity import valid_result
import al_driver
from optimization.nsga2_problem import MFTProblem, T_TARGETS
from optimization import run_nsga2


class StrictRowContractTests(unittest.TestCase):
    def test_legacy_false_positive_is_quarantined(self):
        row = valid_result(conv_error_pct_matrix=13.254)
        result = validate_record(row)
        self.assertFalse(result.em_valid)
        self.assertIn("matrix:error_exceeds_tolerance", result.reasons)
        self.assertFalse(validate_record(
            valid_result(), expected_solver_revision="c" * 40
        ).full_valid)
        self.assertFalse(validate_record(
            valid_result(V1_rms=900.0)
        ).full_valid)
        self.assertFalse(validate_record(
            valid_result(fan_velocity=2.0)
        ).full_valid)

    def test_delta_and_thermal_evidence_are_fail_closed(self):
        missing_delta = valid_result()
        missing_delta.pop("conv_delta_pct_loss")
        self.assertFalse(validate_record(missing_delta).full_valid)
        self.assertFalse(validate_record(
            valid_result(thermal_rx_power_balance_ok=0)
        ).full_valid)
        self.assertFalse(validate_record(
            valid_result(T_max_Rx_main=4700.0)
        ).full_valid)
        self.assertFalse(validate_record(
            valid_result(conv_consecutive_loss=1)
        ).full_valid)
        # A late one-pass dip below tolerance is not two consecutive converged
        # adaptive passes, regardless of the total adaptive pass index.
        self.assertFalse(validate_record(
            valid_result(conv_passes_loss=9, conv_consecutive_loss=1)
        ).full_valid)
        self.assertFalse(validate_record(
            valid_result(matrix_solve_attempts=2)
        ).full_valid)
        self.assertFalse(validate_record(
            valid_result(thermal_rx_model="hybrid_explicit")
        ).full_valid)
        self.assertFalse(validate_record(
            valid_result(P_wcp_total=-1.0)
        ).full_valid)

    def test_training_filter_uses_strict_full_for_every_target(self):
        good = valid_result(N1_main=3, l1=100.0)
        bad = valid_result(
            N1_main=4, l1=110.0, conv_error_pct_matrix=13.254
        )
        frame = pd.DataFrame([good, bad])
        audited = annotate_validity(frame)
        self.assertEqual(int(audited["_strict_valid_full"].sum()), 1)
        self.assertEqual(len(filter_valid_training_rows(frame, "Llt")), 1)

    def test_feature_whitelist_excludes_postsolve_leakage(self):
        frame = pd.DataFrame([
            valid_result(N1_main=3, l1=100.0, sample_weight=1.0),
            valid_result(N1_main=4, l1=110.0, sample_weight=3.0),
        ])
        features = feature_columns(frame)
        self.assertIn("N1_main", features)
        self.assertIn("l1", features)
        self.assertNotIn("sample_weight", features)
        self.assertFalse(any(name.startswith("thermal_") for name in features))
        self.assertFalse(any(name.startswith("result_") for name in features))


class CheckpointScheduleTests(unittest.TestCase):
    def test_schedule_has_3000_and_every_1000_afterward(self):
        self.assertEqual(checkpoint_sequence(499), [])
        self.assertEqual(checkpoint_sequence(3000), [500, 1000, 2000, 3000])
        self.assertEqual(
            checkpoint_sequence(6500),
            [500, 1000, 2000, 3000, 4000, 5000, 6000],
        )

    def test_failed_quality_gate_waits_for_growth_or_backoff(self):
        now = datetime.fromisoformat("2026-07-11T04:00:00")
        state = {"attempts": [{
            "threshold": 3000,
            "strict_full_rows": 3000,
            "status": "failed",
            "finished_at": "2026-07-11T03:50:00",
        }]}
        ready, deferred = due_with_backoff(
            [3000], state, 3100, minimum_new_rows=250,
            backoff_seconds=3600, now=now,
        )
        self.assertEqual(ready, [])
        self.assertEqual(deferred[0]["threshold"], 3000)
        ready, _ = due_with_backoff(
            [3000], state, 3250, minimum_new_rows=250,
            backoff_seconds=3600, now=now,
        )
        self.assertEqual(ready, [3000])


class ActiveLearningInvariantTests(unittest.TestCase):
    def _pinned_runtime(self):
        return (
            patch.object(al_driver, "PINNED_SOLVER_REVISION", "a" * 40),
            patch.object(al_driver, "PINNED_LIBRARY_REVISION", "b" * 40),
        )

    def test_legacy_or_revision_mismatched_state_cannot_resume(self):
        solver_patch, library_patch = self._pinned_runtime()
        with solver_patch, library_patch:
            fresh = {
                "stage": "TRAIN", "training_run_id": None,
                "task_map": {}, "final_verification": None,
            }
            identity = al_driver._bind_runtime_identity(fresh)
            self.assertEqual(identity["minimum_strict_full_rows"], 3000)
            self.assertEqual(fresh["runtime_identity"], identity)

            with self.assertRaisesRegex(RuntimeError, "legacy/unpinned"):
                al_driver._bind_runtime_identity({"stage": "SELECT"})

            mismatched = dict(fresh)
            mismatched["runtime_identity"] = dict(
                fresh["runtime_identity"], solver_revision="c" * 40
            )
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                al_driver._bind_runtime_identity(mismatched)

    def test_every_downstream_stage_reasserts_hard_3000_row_gate(self):
        solver_patch, library_patch = self._pinned_runtime()
        with solver_patch, library_patch:
            state = {
                "stage": "TRAIN", "training_run_id": None,
                "task_map": {},
                "final_verification": None,
            }
            al_driver._bind_runtime_identity(state)
            state.update({
                "stage": "OPTIMIZE",
                "training_run_id": "run-underfilled",
                "training_strict_full_rows": 2999,
                "model_quality_snapshot": {
                    "passed": True,
                    "strict_full_rows": 2999,
                },
            })
            with self.assertRaisesRegex(
                RuntimeError, "insufficient_pinned_strict_full_rows"
            ):
                al_driver._assert_training_invariants(state)


class NSGAHardGateTests(unittest.TestCase):
    def test_custom_thresholds_cannot_weaken_production_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            custom = Path(tmp, "weak.json")
            custom.write_text(
                json.dumps({"minimum_strict_full_rows": 1}), encoding="utf-8"
            )
            with patch.object(sys, "argv", [
                "run_nsga2.py", "--quality-thresholds", str(custom),
                "--output-root", tmp,
            ]), self.assertRaisesRegex(
                SystemExit, "differ from the vetted production contract"
            ):
                run_nsga2.main()

    def test_pinned_generation_below_3000_cannot_optimize(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp, "snapshot.parquet")
            dataset.write_bytes(b"underfilled")
            dataset_sha = hashlib.sha256(b"underfilled").hexdigest()
            generation = Path(tmp, "generation")
            generation.mkdir()
            quality = Path(tmp, "quality.json")
            threshold_sha = hashlib.sha256(
                Path(run_nsga2.VETTED_QUALITY_THRESHOLDS).read_bytes()
            ).hexdigest()
            run_id = "underfilled-run"
            Path(generation, "train_report.json").write_text(json.dumps({
                "training_run_id": run_id,
                "dataset_sha256": dataset_sha,
                "strict_full_rows": 2999,
            }), encoding="utf-8")
            quality.write_text(json.dumps({
                "passed": True,
                "training_run_id": run_id,
                "dataset_sha256": dataset_sha,
                "strict_full_rows": 2999,
                "quality_thresholds_sha256": threshold_sha,
                "solver_revision": "a" * 40,
                "library_revision": "b" * 40,
            }), encoding="utf-8")
            with patch.object(sys, "argv", [
                "run_nsga2.py",
                "--dataset", str(dataset),
                "--registry-generation", str(generation),
                "--quality-status", str(quality),
                "--quality-thresholds", run_nsga2.VETTED_QUALITY_THRESHOLDS,
                "--output-root", tmp,
            ]), self.assertRaisesRegex(
                SystemExit, "generation/quality/dataset identity mismatch"
            ):
                run_nsga2.main()


class ModelQualityGateTests(unittest.TestCase):
    def test_generation_fingerprint_and_uq_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = os.path.join(tmp, "snapshot.parquet")
            Path(dataset).write_bytes(b"stable-snapshot")
            digest = hashlib.sha256(b"stable-snapshot").hexdigest()
            registry = os.path.join(tmp, "registry")
            generation = os.path.join(registry, "generations", "run1")
            target_dir = os.path.join(generation, "Llt_phys")
            os.makedirs(target_dir)
            Path(os.path.join(target_dir, "models.pkl")).write_bytes(b"model")
            metrics = {
                "r2": 0.99,
                "mape_pct": 1.0,
                "interval_coverage": 0.90,
            }
            meta = {
                "training_run_id": "run1",
                "dataset_sha256": digest,
                "features": ["N1_main"],
                "metrics": metrics,
            }
            Path(os.path.join(target_dir, "meta.json")).write_text(
                json.dumps(meta), encoding="utf-8"
            )
            report = {
                "training_run_id": "run1",
                "dataset_sha256": digest,
                "strict_full_rows": 3000,
                "features": ["N1_main"],
            }
            Path(os.path.join(generation, "train_report.json")).write_text(
                json.dumps(report), encoding="utf-8"
            )
            Path(os.path.join(registry, "current.json")).write_text(
                json.dumps({
                    "generation": "generations/run1",
                    "dataset_sha256": digest,
                }),
                encoding="utf-8",
            )
            thresholds = {
                "minimum_strict_full_rows": 3000,
                "minimum_interval_coverage": 0.85,
                "targets": {"Llt_phys": {"min_r2": 0.98, "max_mape_pct": 2.0}},
            }
            self.assertTrue(
                evaluate_registry(registry, dataset, thresholds)["passed"]
            )
            meta["metrics"]["interval_coverage"] = 0.5
            Path(os.path.join(target_dir, "meta.json")).write_text(
                json.dumps(meta), encoding="utf-8"
            )
            self.assertFalse(
                evaluate_registry(registry, dataset, thresholds)["passed"]
            )

    def test_failed_generation_can_restore_previous_atomic_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = os.path.join(tmp, "registry")
            old = os.path.join(registry, "generations", "old")
            os.makedirs(old)
            report = {"training_run_id": "old", "targets": [], "report": {}}
            Path(old, "train_report.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
            Path(registry, "current.json").write_text(
                json.dumps({"generation": "generations/old"}), encoding="utf-8"
            )
            captured = capture_active_generation(registry)
            Path(registry, "current.json").write_text(
                json.dumps({"generation": "generations/rejected"}), encoding="utf-8"
            )
            restore_active_generation(registry, captured)
            pointer = json.loads(
                Path(registry, "current.json").read_text(encoding="utf-8")
            )
            self.assertEqual(pointer["generation"], "generations/old")


class OptimizationPhysicsTests(unittest.TestCase):
    def test_total_loss_objective_includes_winding_cooling_plates(self):
        class Predictor:
            def __init__(self, value):
                self.value = value

            def predict_mu_sigma(self, frame):
                return (
                    np.full(len(frame), self.value, dtype=float),
                    np.zeros(len(frame), dtype=float),
                )

            def disagreement(self, frame):
                return np.zeros(len(frame), dtype=float)

        values = {
            "Llt_phys": 27.5,
            "P_winding_total": 100.0,
            "P_core_total": 200.0,
            "P_core_plate_total": 300.0,
            "P_wcp_total": 400.0,
            "B_max_core": 1.0,
            **{target: 80.0 for target in T_TARGETS},
        }
        problem = MFTProblem(
            {target: Predictor(value) for target, value in values.items()},
            density_gate=lambda frame: np.zeros(len(frame)),
        )
        problem.decode_batch = lambda X: (
            pd.DataFrame([{"h_gap2": 50.0}] * len(X)),
            np.zeros(len(X)),
            np.ones(len(X), dtype=bool),
        )
        out = {}
        with patch(
            "optimization.nsga2_problem.bounding_box_lit",
            return_value=(123.0, (1.0, 1.0, 1.0)),
        ):
            problem._evaluate(np.zeros((1, problem.n_var)), out)
        self.assertEqual(out["F"][0, 1], 1000.0)


class FineValidationTests(unittest.TestCase):
    def fine_result(self, **updates):
        row = valid_result(
            full_model=1,
            Llt=27.5,
            loss_sym_on=0,
            thermal_symmetry="full",
            n_explicit_turns=2,
            thermal_rx_model="hybrid_explicit",
            matrix_skin_mesh=1,
            matrix_percent_error=0.5,
            matrix_max_passes=24,
            matrix_min_converged=2,
            percent_error=0.5,
            max_passes=18,
            min_converged=2,
            keep_project=1,
            matrix_conductor_policy="solid_skin",
            matrix_winding_stranded_count=0,
            matrix_conductor_mesh_operation_count=3,
            matrix_plate_eddy_off_readback_count=0,
            loss_winding_solid_update_count=-1,
            loss_winding_mesh_operation_count=-1,
            loss_conductor_mesh_operation_count=-1,
            loss_plate_eddy_on_readback_count=-1,
            conv_consecutive_matrix=2,
            conv_consecutive_loss=2,
            conv_error_pct_matrix=0.3,
            conv_delta_pct_matrix=0.2,
            conv_error_pct_loss=0.3,
            conv_delta_pct_loss=0.2,
        )
        row.update(updates)
        return row

    def test_full_model_fine_profile_is_supported(self):
        profile = json.loads(
            (HERE / "verify" / "profiles" / "fine.json").read_text(encoding="utf-8")
        )
        self.assertTrue(scheduler_client.is_valid_result(
            self.fine_result(), expected_profile=profile["param_overrides"]
        ))
        self.assertFalse(scheduler_client.is_valid_result(
            self.fine_result(conv_delta_pct_matrix=0.6),
            expected_profile=profile["param_overrides"],
        ))
        self.assertFalse(scheduler_client.is_valid_result(
            self.fine_result(conv_consecutive_matrix=1),
            expected_profile=profile["param_overrides"],
        ))
        self.assertFalse(scheduler_client.is_valid_result(
            self.fine_result(P_wcp_total=-1.0),
            expected_profile=profile["param_overrides"],
        ))

    def test_fine_failure_falls_back_to_next_smallest_candidate(self):
        geometry = {column: 40.0 for column in INSULATION_COLUMNS}
        small_params = {"l1": 50.0}
        next_params = {"l1": 60.0}
        selected = [
            {
                "candidate_digest": _candidate_digest(small_params),
                "volume_L": 100.0,
                "round": 1,
                "task_id": 10,
                "params": small_params,
                "geometry_evidence": geometry,
            },
            {
                "candidate_digest": _candidate_digest(next_params),
                "volume_L": 110.0,
                "round": 1,
                "task_id": 11,
                "params": next_params,
                "geometry_evidence": geometry,
            },
        ]
        failed = self.fine_result(T_max_Tx=101.0, **small_params)
        passed = self.fine_result(**next_params)
        with tempfile.TemporaryDirectory() as tmp:
            artifact = write_final_artifacts(
                tmp, selected, [failed, passed], {"passed": True},
                "a" * 40, "b" * 40,
            )
            self.assertTrue(artifact["passed"])
            self.assertEqual(
                artifact["candidate"]["candidate_digest"],
                _candidate_digest(next_params),
            )
            report = Path(tmp, "final_report.md").read_text(encoding="utf-8")
            self.assertNotIn("Monte Carlo", report)
            self.assertIn("정확히 동일", report)

    def test_fine_uses_actual_geometry_and_exact_candidate_identity(self):
        params = {"l1": 50.0}
        candidate = {
            "candidate_digest": _candidate_digest(params),
            "volume_L": 100.0,
            "params": params,
            # Stale/spoofed Pareto evidence must not override actual FEA.
            "geometry_evidence": {
                column: 40.0 for column in INSULATION_COLUMNS
            },
        }
        bad_gap = self.fine_result(l1=50.0, cc_w2c_space_x=10.0)
        wrong_candidate = self.fine_result(l1=51.0)
        with tempfile.TemporaryDirectory() as tmp:
            gap_artifact = write_final_artifacts(
                tmp, [candidate], [bad_gap], {"passed": True},
                "a" * 40, "b" * 40,
            )
            self.assertFalse(gap_artifact["passed"])
            self.assertIn(
                "insulation_below_minimum:cc_w2c_space_x",
                gap_artifact["fine_attempts"][-1]["reasons"],
            )
            identity_artifact = write_final_artifacts(
                tmp, [candidate], [wrong_candidate], {"passed": True},
                "a" * 40, "b" * 40,
            )
            self.assertFalse(identity_artifact["passed"])
            self.assertIn(
                "candidate_result_identity_mismatch",
                identity_artifact["fine_attempts"][-1]["reasons"],
            )

    def test_invalid_fine_result_after_retry_blocks_larger_candidate(self):
        state = {
            "round": 2,
            "stage": "FINE_WAIT",
            "fine_solver_git_revision": "a" * 40,
            "fine_pyaedt_library_git_revision": "b" * 40,
            "final_candidates": [{"params": {}, "candidate_digest": "small"}],
            "fine_task_records": {
                "0": {
                    "active_id": 77,
                    "attempt": 1,
                    "outcome": "pending",
                }
            },
        }
        with patch.object(
            runtime_scheduler_client, "wait_all", return_value={77: "completed"}
        ), patch.object(
            runtime_scheduler_client, "fetch_result",
            return_value=runtime_scheduler_client.ResultFetch(
                runtime_scheduler_client.RESULT_INVALID
            ),
        ), patch.object(al_driver, "save_state"):
            al_driver.stage_fine_wait(state)
        self.assertEqual(state["stage"], "FINE_BLOCKED")
        self.assertIn("smaller candidate", state["fine_block_reason"])

    def test_larger_unverified_candidate_does_not_block_smaller_fine_pass(self):
        candidates = []
        records = {}
        for rank, (l1, volume) in enumerate(((50.0, 100.0), (60.0, 110.0), (70.0, 120.0))):
            params = {"l1": l1}
            candidates.append({
                "params": params,
                "candidate_digest": _candidate_digest(params),
                "volume_L": volume,
            })
            records[str(rank)] = {
                "active_id": 77 + rank,
                "attempt": 0,
                "outcome": "valid",
                "result": self.fine_result(l1=l1),
            }
        records["2"].update(attempt=1, outcome="pending", result=None)
        state = {
            "round": 2,
            "stage": "FINE_WAIT",
            "fine_solver_git_revision": "a" * 40,
            "fine_pyaedt_library_git_revision": "b" * 40,
            "final_candidates": candidates,
            "fine_task_records": records,
        }
        with patch.object(
            runtime_scheduler_client, "wait_all", return_value={79: "completed"}
        ), patch.object(
            runtime_scheduler_client, "fetch_result",
            return_value=runtime_scheduler_client.ResultFetch(
                runtime_scheduler_client.RESULT_INVALID
            ),
        ), patch.object(al_driver, "save_state"):
            al_driver.stage_fine_wait(state)
        self.assertEqual(state["stage"], "FINAL_REPORT")
        self.assertEqual(records["2"]["outcome"], "unverified")


if __name__ == "__main__":
    unittest.main()
