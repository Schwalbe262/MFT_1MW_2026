import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

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
from verify.finalize import INSULATION_COLUMNS, write_final_artifacts
from test_al_integrity import valid_result
import al_driver


class StrictRowContractTests(unittest.TestCase):
    def test_legacy_false_positive_is_quarantined(self):
        row = valid_result(conv_error_pct_matrix=13.254)
        result = validate_record(row)
        self.assertFalse(result.em_valid)
        self.assertIn("matrix:error_exceeds_tolerance", result.reasons)
        self.assertFalse(validate_record(
            valid_result(), expected_solver_revision="c" * 40
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
            valid_result(conv_passes_loss=1)
        ).full_valid)
        self.assertFalse(validate_record(
            valid_result(matrix_solve_attempts=2)
        ).full_valid)
        self.assertFalse(validate_record(
            valid_result(thermal_rx_model="hybrid_explicit")
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
            self.fine_result(conv_passes_matrix=1),
            expected_profile=profile["param_overrides"],
        ))

    def test_fine_failure_falls_back_to_next_smallest_candidate(self):
        geometry = {column: 40.0 for column in INSULATION_COLUMNS}
        selected = [
            {
                "candidate_digest": "small",
                "volume_L": 100.0,
                "round": 1,
                "task_id": 10,
                "params": {},
                "geometry_evidence": geometry,
            },
            {
                "candidate_digest": "next",
                "volume_L": 110.0,
                "round": 1,
                "task_id": 11,
                "params": {},
                "geometry_evidence": geometry,
            },
        ]
        failed = self.fine_result(T_max_Tx=101.0)
        passed = self.fine_result()
        with tempfile.TemporaryDirectory() as tmp:
            artifact = write_final_artifacts(
                tmp, selected, [failed, passed], {"passed": True},
                "a" * 40, "b" * 40,
            )
            self.assertTrue(artifact["passed"])
            self.assertEqual(artifact["candidate"]["candidate_digest"], "next")
            report = Path(tmp, "final_report.md").read_text(encoding="utf-8")
            self.assertNotIn("Monte Carlo", report)
            self.assertIn("정확히 동일", report)

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


if __name__ == "__main__":
    unittest.main()
