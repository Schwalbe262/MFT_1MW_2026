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
from model_targets import (
    CORE_REGION_TEMPERATURE_TARGETS,
    SURROGATE_TEMPERATURE_TARGETS,
    SURROGATE_WINDING_COMPONENT_LOSS_TARGETS,
)
from training.checkpoint_orchestrator import checkpoint_sequence, due_with_backoff
from training.checkpoint_train import (
    TARGETS as CHECKPOINT_TARGETS,
    feature_columns,
    filter_valid_training_rows,
)
from training.model_quality_gate import evaluate_generation, evaluate_registry
from training.train_models import (
    capture_active_generation, promote_generation, registry_pointer_token,
    restore_active_generation,
)
from verify import scheduler_client
import scheduler_client as runtime_scheduler_client
from verify.finalize import (
    ALLOWED_CANDIDATE_INPUT_SCHEMAS, INSULATION_COLUMNS,
    QUALITY_THRESHOLDS_PATH, REQUIRED_OPTIMIZATION_MODELS, _candidate_digest,
    candidate_identity_reasons, write_final_artifacts,
)
from module.input_parameter_260706 import (
    ALL_INPUT_KEYS,
    KEYS,
    PRE_ANISOTROPIC_CORE_K_INPUT_KEYS,
    create_input_parameter,
)
from module.core_material_contract import PHYSICS_DATA_REVISION
from module.thermal_probe_contract import (
    RX_SIDE_FACE_MAX_RULE,
    RX_SIDE_FACE_MEAN_RULE,
    RX_SIDE_FACE_PROBE_CONTRACT_VERSION,
)
from optimization.geometry_metrics import bounding_box_lit
from test_al_integrity import valid_result
import al_driver
from optimization.nsga2_problem import MFTProblem, T_TARGETS
from optimization import run_nsga2
from monitoring.readers import TEMPERATURE_TARGETS as MONITORING_TEMPERATURE_TARGETS
from campaign import collect_wave


CALCOP_UNAVAILABLE_REASON = (
    "grpc_calcop_unavailable:Failed to execute gRPC AEDT command: CalcOp"
)


def _valid_native_result(**updates):
    row = valid_result(
        physics_data_revision=PHYSICS_DATA_REVISION,
        Tprobe_Rx_side_leeward_mean=84.0,
        Tprobe_Rx_side_outer_max=88.0,
        Tprobe_Rx_side_outer_mean=82.0,
        Tprobe_Rx_side_inner_max=91.0,
        Tprobe_Rx_side_inner_mean=84.0,
        thermal_rx_side_probe_contract_version=(
            RX_SIDE_FACE_PROBE_CONTRACT_VERSION
        ),
        thermal_rx_side_probe_max_rule=RX_SIDE_FACE_MAX_RULE,
        thermal_rx_side_probe_mean_rule=RX_SIDE_FACE_MEAN_RULE,
        thermal_rx_side_probe_selected_face="Tprobe_Rx_side1_inner",
        thermal_rx_side_probe_face_count=2,
        core_native_material_readback_attested=1,
        core_loss_native_attested=1,
        flux_linkage_attested=1,
        B_mean_faraday_attested=1,
        core_loss_native_rel_error=0.01,
        core_loss_native_tolerance_rel=0.30,
        B_mean_material_vs_sine_analytic_rel_error=0.01,
        B_mean_faraday_tolerance_rel=0.15,
        core_loss_reference_basis=(
            "sinusoidal_faraday_Bpack_then_Bmaterial_div_kf_then_"
            "POWERLITE_Wkg_times_effective_mass"
        ),
        center_leg_surface_flux_integral_applicable=1,
        center_leg_surface_flux_integral_available=1,
        center_leg_surface_flux_integral_passed=1,
        center_leg_surface_flux_integral_status="available",
        center_leg_surface_flux_integral_reason="",
        winding_flux_linkage_readback_applicable=1,
        winding_flux_linkage_readback_available=1,
        winding_flux_linkage_readback_passed=1,
        winding_flux_linkage_readback_status="available",
        winding_flux_linkage_readback_reason="",
        Tx_flux_linkage_faraday_rel_error=0.01,
        Tx_induced_vs_source_peak_rel_error=0.05,
        core_surface_flux_vs_linkage_rel_error=0.01,
        core_surface_flux_vs_induced_voltage_rel_error=0.01,
        core_native_model_approval_status=(
            "approved_by_isolated_solved_kf_ab"
        ),
        thermal_core_loss_source=(
            "aedt_native_lamination_loss_attested_then_margin_adjusted"
        ),
        thermal_core_native_readback_count=3,
        thermal_core_native_restored_rel_error=0.0,
        thermal_core_loss_correction_factor=1.15,
    )
    row["thermal_core_full_expected_margin_adjusted_w"] = row[
        "P_core_total"
    ]
    row.update(updates)
    return row


def _native_result_with_unavailable_winding_readback(**updates):
    row = _valid_native_result(
        flux_linkage_attested=0,
        winding_flux_linkage_readback_applicable=0,
        winding_flux_linkage_readback_available=0,
        winding_flux_linkage_readback_passed=1,
        winding_flux_linkage_readback_status="unavailable",
        winding_flux_linkage_readback_reason=CALCOP_UNAVAILABLE_REASON,
        Tx_flux_linkage_faraday_rel_error=float("nan"),
        Tx_induced_vs_source_peak_rel_error=float("nan"),
        core_surface_flux_vs_linkage_rel_error=float("nan"),
        core_surface_flux_vs_induced_voltage_rel_error=float("nan"),
        center_leg_surface_flux_integral_passed=0,
    )
    row.update(updates)
    return row


class CoreTemperatureTargetContractTests(unittest.TestCase):
    def test_three_core_regions_are_independent_end_to_end_targets(self):
        expected = tuple(SURROGATE_TEMPERATURE_TARGETS)
        self.assertEqual(tuple(T_TARGETS), expected)
        self.assertEqual(tuple(al_driver.T_TARGETS), expected)
        self.assertEqual(tuple(MONITORING_TEMPERATURE_TARGETS), expected)
        self.assertTrue(
            set(CORE_REGION_TEMPERATURE_TARGETS).issubset(CHECKPOINT_TARGETS)
        )
        self.assertTrue(set(expected).issubset(REQUIRED_OPTIMIZATION_MODELS))
        self.assertTrue(
            set(SURROGATE_WINDING_COMPONENT_LOSS_TARGETS).issubset(
                CHECKPOINT_TARGETS
            )
        )
        self.assertTrue(
            set(SURROGATE_WINDING_COMPONENT_LOSS_TARGETS).issubset(
                REQUIRED_OPTIMIZATION_MODELS
            )
        )


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

    def test_native_winding_readback_present_and_passing_is_full_valid(self):
        row = _valid_native_result()

        self.assertTrue(validate_record(row).full_valid)
        self.assertTrue(scheduler_client.is_valid_result(row))

    def test_native_winding_calcop_unavailability_remains_full_valid(self):
        row = _native_result_with_unavailable_winding_readback()

        result = validate_record(row)
        self.assertTrue(result.full_valid, result.reasons)
        self.assertNotIn(
            "native_lamination:flux_linkage_attested", result.reasons
        )
        self.assertTrue(scheduler_client.is_valid_result(row))

    def test_native_available_winding_readback_failure_is_quarantined(self):
        cases = (
            (
                {
                    "flux_linkage_attested": 0,
                    "winding_flux_linkage_readback_passed": 0,
                    "Tx_flux_linkage_faraday_rel_error": 0.010001,
                },
                "native_lamination:flux_linkage_attested",
            ),
            (
                {"winding_flux_linkage_readback_passed": 0},
                (
                    "native_lamination:"
                    "winding_flux_linkage_readback_evidence_invalid"
                ),
            ),
            (
                {"core_surface_flux_vs_linkage_rel_error": 0.050001},
                "native_lamination:core_surface_flux_vs_linkage_rel_error",
            ),
            (
                {
                    "core_surface_flux_vs_induced_voltage_rel_error": 0.050001
                },
                (
                    "native_lamination:"
                    "core_surface_flux_vs_induced_voltage_rel_error"
                ),
            ),
        )
        for updates, expected_reason in cases:
            with self.subTest(reason=expected_reason):
                row = _valid_native_result(**updates)

                result = validate_record(row)
                self.assertFalse(result.full_valid)
                self.assertIn(expected_reason, result.reasons)
                self.assertFalse(scheduler_client.is_valid_result(row))

    def test_native_winding_unavailability_does_not_waive_b_mean(self):
        row = _native_result_with_unavailable_winding_readback()
        row.pop("B_mean_faraday_attested")

        result = validate_record(row)
        self.assertFalse(result.full_valid)
        self.assertIn(
            "native_lamination:B_mean_faraday_attested", result.reasons
        )
        self.assertFalse(scheduler_client.is_valid_result(row))

    def test_new_revision_rx_side_inner_face_is_fail_closed_but_legacy_is_valid(self):
        legacy = valid_result()
        self.assertTrue(validate_record(legacy).full_valid)
        revised = _valid_native_result()
        self.assertTrue(validate_record(revised).full_valid)
        normalized, demoted = collect_wave.normalize_thermal_validity(
            pd.DataFrame([revised])
        )
        self.assertEqual(demoted, 0)
        self.assertTrue(scheduler_client.is_valid_result(
            normalized.iloc[0].to_dict()
        ))
        revised.pop("Tprobe_Rx_side_inner_max")
        result = validate_record(revised)
        self.assertFalse(result.thermal_valid)
        self.assertIn(
            "thermal_temperature:nonfinite:Tprobe_Rx_side_inner_max",
            result.reasons,
        )
        normalized, demoted = collect_wave.normalize_thermal_validity(
            pd.DataFrame([revised])
        )
        self.assertEqual(demoted, 1)
        self.assertFalse(scheduler_client.is_valid_result(revised))

    def test_b171_legacy_row_survives_collector_quality_and_scheduler_paths(self):
        # b171 predates physics_data_revision and the new inner-face columns.
        # It must remain valid under its pinned controller while the new
        # revision uses the stricter face-probe contract.
        legacy = valid_result()
        self.assertNotIn("physics_data_revision", legacy)
        self.assertNotIn("Tprobe_Rx_side_inner_max", legacy)

        normalized, demoted = collect_wave.normalize_thermal_validity(
            pd.DataFrame([legacy])
        )
        record = normalized.iloc[0].to_dict()

        self.assertEqual(demoted, 0)
        self.assertEqual(record["thermal_solved"], 1)
        self.assertTrue(validate_record(record).full_valid)
        self.assertTrue(scheduler_client.is_valid_result(
            record,
            expected_revision=legacy["git_hash"],
            expected_library_revision=legacy["pyaedt_library_git_hash"],
        ))
        self.assertFalse(validate_record(
            valid_result(thermal_rx_model="hybrid_explicit")
        ).full_valid)
        self.assertFalse(validate_record(
            valid_result(P_wcp_total=-1.0)
        ).full_valid)

    def test_explicit_turn_profile_identity_matches_thermal_model(self):
        current = validate_record(valid_result())
        historical = validate_record(valid_result(
            n_explicit_turns=2,
            thermal_rx_model="hybrid_explicit",
        ))

        self.assertNotIn("profile_mismatch:n_explicit_turns", current.reasons)
        self.assertIn("profile_mismatch:n_explicit_turns", historical.reasons)
        self.assertFalse(historical.full_valid)

    def test_training_filter_uses_strict_full_for_every_target(self):
        good = valid_result(N1_main=3, l1=100.0)
        bad = valid_result(
            N1_main=4, l1=110.0, conv_error_pct_matrix=13.254
        )
        frame = pd.DataFrame([good, bad])
        audited = annotate_validity(frame)
        self.assertEqual(int(audited["_strict_valid_full"].sum()), 1)
        self.assertEqual(len(filter_valid_training_rows(frame, "Llt")), 1)

    def test_winding_component_outputs_are_complete_and_sum_to_total(self):
        missing = valid_result()
        missing.pop("P_Rx_main_group")
        self.assertFalse(validate_record(missing).full_valid)
        self.assertFalse(validate_record(valid_result(
            P_Tx_main_group=2401.0,
        )).full_valid)
        self.assertTrue(validate_record(valid_result()).full_valid)

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

    def test_reconciliation_recovery_bypasses_failure_backoff(self):
        now = datetime.fromisoformat("2026-07-11T04:00:00")
        state = {"attempts": [{
            "threshold": 3000, "strict_full_rows": 3000,
            "status": "failed", "finished_at": "2026-07-11T03:59:00",
        }]}
        ready, deferred = due_with_backoff(
            [3000], state, 3000, minimum_new_rows=250,
            backoff_seconds=3600, now=now, force_ready={3000},
        )
        self.assertEqual(ready, [3000])
        self.assertEqual(deferred, [])

    def test_malformed_reconciliation_threshold_does_not_crash_schedule(self):
        ready, deferred = due_with_backoff(
            [500], {"attempts": []}, 500, force_ready={"corrupt", None}
        )
        self.assertEqual(ready, [500])
        self.assertEqual(deferred, [])


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
                "round": 1, "stage": "TRAIN", "q_mult": 1.0,
                "task_map": {}, "history": [],
            }
            identity = al_driver._bind_runtime_identity(fresh)
            self.assertEqual(identity["minimum_strict_full_rows"], 3000)
            self.assertEqual(fresh["runtime_identity"], identity)

            with self.assertRaisesRegex(RuntimeError, "legacy/unpinned"):
                al_driver._bind_runtime_identity({"stage": "SELECT"})
            with self.assertRaisesRegex(RuntimeError, "legacy/unpinned"):
                al_driver._bind_runtime_identity({
                    "round": 7, "stage": "TRAIN", "q_mult": 1.0,
                    "task_map": {}, "history": [{"round": 6}],
                })

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
                "round": 1, "stage": "TRAIN", "q_mult": 1.0,
                "task_map": {}, "history": [],
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

    def test_claimed_3000_rows_are_recomputed_from_pinned_parquet(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp, "snapshot.parquet")
            pd.DataFrame().to_parquet(dataset, index=False)
            dataset_sha = hashlib.sha256(dataset.read_bytes()).hexdigest()
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
                "strict_full_rows": 3000,
            }), encoding="utf-8")
            quality.write_text(json.dumps({
                "passed": True,
                "training_run_id": run_id,
                "dataset_sha256": dataset_sha,
                "strict_full_rows": 3000,
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
                SystemExit, "strict-full cohort does not match quality metadata"
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
                "profile_sha256": "profile-sha",
                "features": ["N1_main"],
                "artifacts": {
                    "Llt_phys/models.pkl": hashlib.sha256(b"model").hexdigest(),
                    "Llt_phys/meta.json": hashlib.sha256(
                        json.dumps(meta).encode("utf-8")
                    ).hexdigest(),
                },
            }
            report_path = Path(os.path.join(generation, "train_report.json"))
            report_path.write_text(json.dumps(report), encoding="utf-8")
            thresholds = {
                "minimum_strict_full_rows": 3000,
                "minimum_interval_coverage": 0.85,
                "targets": {"Llt_phys": {"min_r2": 0.98, "max_mape_pct": 2.0}},
            }
            quality = evaluate_generation(
                registry, generation, dataset, thresholds
            )
            self.assertTrue(quality["passed"], quality["reasons"])
            promote_generation(
                registry, generation, quality,
                dataset=dataset, profile_sha256="profile-sha",
                thresholds_sha256=quality["thresholds_sha256"],
                expected_pointer=registry_pointer_token(registry),
            )
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
            target_dir = Path(old, "Llt_phys")
            target_dir.mkdir(parents=True)
            Path(target_dir, "models.pkl").write_bytes(b"old-model")
            Path(target_dir, "meta.json").write_text("{}", encoding="utf-8")
            old_dataset = Path(tmp, "old-dataset.bin")
            old_dataset.write_bytes(b"dataset")
            old_dataset_sha256 = hashlib.sha256(b"dataset").hexdigest()
            report = {
                "training_run_id": "old", "targets": ["Llt_phys"],
                "dataset_sha256": old_dataset_sha256,
                "profile_sha256": "profile",
                "strict_full_rows": 3000, "report": {},
                "artifacts": {
                    "Llt_phys/models.pkl": hashlib.sha256(b"old-model").hexdigest(),
                    "Llt_phys/meta.json": hashlib.sha256(b"{}").hexdigest(),
                },
            }
            report_path = Path(old, "train_report.json")
            report_path.write_text(json.dumps(report), encoding="utf-8")
            quality = {
                "passed": True, "training_run_id": "old",
                "dataset_sha256": old_dataset_sha256,
                "profile_sha256": "profile",
                "thresholds_sha256": "thresholds",
                "generation": "generations/old",
                "generation_report_sha256": hashlib.sha256(
                    report_path.read_bytes()
                ).hexdigest(),
            }
            promote_generation(
                registry, old, quality,
                dataset=old_dataset,
                profile_sha256="profile",
                thresholds_sha256="thresholds",
                expected_pointer=registry_pointer_token(registry),
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
            "P_Tx_main_group": 60.0,
            "P_Rx_main_group": 30.0,
            "P_Rx_side_total": 10.0,
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
            pd.DataFrame([{
                "h_gap2": 50.0,
                "N1_main": 6,
                "N1_side": 0,
                "V1_rms": 1000.0,
                "freq": 1000.0,
                "Ae_m2": 0.08,
            }] * len(X)),
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
        self.assertEqual(
            out["G"].shape,
            (1, 1 + len(T_TARGETS) + 5),
        )
        self.assertTrue(np.isfinite(out["G"]).all())
        self.assertTrue((np.abs(out["G"]) < 1.0e5).all())


class FineValidationTests(unittest.TestCase):
    def candidate(self, l1, volume_L=100.0):
        safe_inputs = {
            "l1": l1,
            "cc_w2c_space_x": 40.0,
            "cc_w2c_space_y": 40.0,
            "w2c_w1c_space_x": 40.0,
            "w2c_w1c_space_y": 40.0,
            "w1c_w2s_space_x": 40.0,
            "w2s_w1s_space_x": 40.0,
            "w1s_w2s_space_y": 40.0,
            "w1s_cs_space_x": 40.0,
            "cs_w1s_space_y": 40.0,
        }
        complete = create_input_parameter(safe_inputs).iloc[0].to_dict()
        # The Sobol/candidate identity remains the sealed KEYS schema. Fixed
        # material policy columns are echoed by the solver but are not design
        # coordinates and must not alter candidate hashes.
        params = {key: complete[key] for key in KEYS}
        standard_profile = json.loads(
            (HERE / "verify" / "profiles" / "standard.json").read_text(
                encoding="utf-8"
            )
        )
        fine_profile = json.loads(
            (HERE / "verify" / "profiles" / "fine.json").read_text(
                encoding="utf-8"
            )
        )
        standard_params = runtime_scheduler_client.effective_verification_params(
            params, standard_profile
        )
        fine_params = runtime_scheduler_client.effective_verification_params(
            params, fine_profile
        )
        return {
            "candidate_digest": _candidate_digest(params),
            "volume_L": volume_L,
            "params": params,
            "standard_params": standard_params,
            "fine_params": fine_params,
            "geometry_evidence": {
                column: 40.0 for column in INSULATION_COLUMNS
            },
        }

    def quality_snapshot(self):
        return {
            "passed": True,
            "strict_full_rows": 3000,
            "solver_revision": "a" * 40,
            "library_revision": "b" * 40,
            "quality_thresholds_sha256": hashlib.sha256(
                Path(QUALITY_THRESHOLDS_PATH).read_bytes()
            ).hexdigest(),
            "targets": {target: {} for target in REQUIRED_OPTIMIZATION_MODELS},
            "training_run_id": "quality-run",
            "dataset_sha256": "d" * 64,
        }

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
            sl1_main_x=100.0,
            nwl1_main=10.0,
            sl1_main_y=100.0,
            nwb1_main_y=10.0,
            sl2_side_x=100.0,
            sl2_side_y=100.0,
            nwl2_side=10.0,
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
        selected = [self.candidate(50.0), self.candidate(60.0)]
        selected[0].update(round=1, task_id=10)
        selected[1].update(round=1, task_id=11)
        failed = self.fine_result(
            **dict(selected[0]["fine_params"], T_max_Tx=101.0)
        )
        passed = self.fine_result(**selected[1]["fine_params"])
        with tempfile.TemporaryDirectory() as tmp, patch(
            "verify.finalize._final_candidate_provenance_reasons", return_value=[]
        ):
            artifact = write_final_artifacts(
                tmp, selected, [failed, passed], self.quality_snapshot(),
                "a" * 40, "b" * 40,
            )
            self.assertTrue(artifact["passed"])
            self.assertEqual(
                artifact["candidate"]["candidate_digest"],
                selected[1]["candidate_digest"],
            )
            report = Path(tmp, "final_report.md").read_text(encoding="utf-8")
            self.assertNotIn("Monte Carlo", report)
            self.assertIn("정확히 동일", report)

    def test_fine_uses_actual_geometry_and_exact_candidate_identity(self):
        candidate = self.candidate(50.0)
        # Stale/spoofed Pareto evidence must not override actual FEA.
        bad_gap_params = dict(candidate["fine_params"], cc_w2c_space_x=10.0)
        bad_gap = self.fine_result(**bad_gap_params)
        wrong_params = dict(candidate["fine_params"], l1=51.0)
        wrong_candidate = self.fine_result(**wrong_params)
        with tempfile.TemporaryDirectory() as tmp, patch(
            "verify.finalize._final_candidate_provenance_reasons", return_value=[]
        ):
            gap_artifact = write_final_artifacts(
                tmp, [candidate], [bad_gap], self.quality_snapshot(),
                "a" * 40, "b" * 40,
            )
            self.assertFalse(gap_artifact["passed"])
            self.assertIn(
                "insulation_below_minimum:cc_w2c_space_x",
                gap_artifact["fine_attempts"][-1]["reasons"],
            )
            identity_artifact = write_final_artifacts(
                tmp, [candidate], [wrong_candidate], self.quality_snapshot(),
                "a" * 40, "b" * 40,
            )
            self.assertFalse(identity_artifact["passed"])
            self.assertIn(
                "candidate_result_identity_mismatch",
                identity_artifact["fine_attempts"][-1]["reasons"],
            )

    def test_final_pass_requires_standard_manifest_and_3000_row_quality(self):
        candidate = self.candidate(50.0)
        result = self.fine_result(**candidate["fine_params"])
        with tempfile.TemporaryDirectory() as tmp:
            artifact = write_final_artifacts(
                tmp, [candidate], [result], {"passed": True},
                "a" * 40, "b" * 40,
            )
        self.assertFalse(artifact["passed"])
        reasons = artifact["fine_attempts"][-1]["reasons"]
        self.assertIn("model_quality_below_3000_strict_rows", reasons)
        self.assertIn("candidate_optimization_manifest_unavailable", reasons)
        self.assertIn("strict_standard_contract_failed", reasons)

    def test_candidate_identity_requires_all_71_submitted_inputs(self):
        candidate = self.candidate(50.0)
        result = self.fine_result(**candidate["fine_params"])
        candidate["params"].pop("fan_config")
        reasons = candidate_identity_reasons(
            result, candidate, expected_params_key="fine_params"
        )
        self.assertIn("candidate_params_missing", reasons)

    def test_candidate_schemas_preserve_sealed_digest_across_revision_metadata(self):
        complete = create_input_parameter({}).iloc[0].to_dict()
        sealed = {key: complete[key] for key in KEYS}
        pre_anisotropic = {
            key: complete[key]
            for key in PRE_ANISOTROPIC_CORE_K_INPUT_KEYS
        }
        legacy_thermal = dict(
            complete,
            core_k_anisotropic=0,
            core_k_alloy=8.5,
            core_k_interlayer=0.25,
        )

        self.assertEqual(len(KEYS), 71)
        self.assertEqual(len(PRE_ANISOTROPIC_CORE_K_INPUT_KEYS), 75)
        self.assertEqual(len(ALL_INPUT_KEYS), 78)
        self.assertEqual(
            {len(schema) for schema in ALLOWED_CANDIDATE_INPUT_SCHEMAS},
            {71, 75, 78},
        )
        for params in (sealed, pre_anisotropic, complete, legacy_thermal):
            with self.subTest(schema_size=len(params)):
                self.assertIn(
                    frozenset(params), ALLOWED_CANDIDATE_INPUT_SCHEMAS
                )
                self.assertEqual(
                    _candidate_digest(params), _candidate_digest(sealed)
                )

        unknown = dict(complete, unknown_candidate_input=1)
        self.assertNotIn(
            frozenset(unknown), ALLOWED_CANDIDATE_INPUT_SCHEMAS
        )

    def test_final_ranking_uses_fine_result_volume_not_stored_candidate_volume(self):
        first = self.candidate(50.0, volume_L=1.0)
        second = self.candidate(60.0, volume_L=2.0)
        first_result = self.fine_result(**first["fine_params"])
        second_result = self.fine_result(**second["fine_params"])
        actual = {
            first["candidate_digest"]: float(bounding_box_lit(first_result)[0]),
            second["candidate_digest"]: float(bounding_box_lit(second_result)[0]),
        }
        # Reverse/spoof the stored ordering. The fine geometry remains the only
        # authoritative source for the winner and reported volume.
        first["volume_L"], second["volume_L"] = 0.1, 0.01
        with tempfile.TemporaryDirectory() as tmp, patch(
            "verify.finalize._final_candidate_provenance_reasons", return_value=[]
        ):
            artifact = write_final_artifacts(
                tmp, [first, second], [first_result, second_result],
                self.quality_snapshot(), "a" * 40, "b" * 40,
            )
        expected_digest = min(actual, key=actual.get)
        self.assertTrue(artifact["passed"])
        self.assertEqual(artifact["candidate_id"], expected_digest)
        self.assertAlmostEqual(
            artifact["candidate"]["volume_L"], actual[expected_digest]
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
        ), patch.object(al_driver, "save_state"), patch.object(
            al_driver, "_assert_training_invariants"
        ):
            al_driver.stage_fine_wait(state)
        self.assertEqual(state["stage"], "FINE_BLOCKED")
        self.assertIn("smaller candidate", state["fine_block_reason"])

    def test_larger_unverified_candidate_does_not_block_smaller_fine_pass(self):
        candidates = []
        records = {}
        for rank, l1 in enumerate((50.0, 60.0, 70.0)):
            candidate = self.candidate(l1)
            result = self.fine_result(**candidate["fine_params"])
            candidate["volume_L"] = float(bounding_box_lit(result)[0])
            candidates.append(candidate)
            records[str(rank)] = {
                "active_id": 77 + rank,
                "attempt": 0,
                "outcome": "valid",
                "result": result,
            }
        candidates[2]["volume_L"] = max(
            candidates[0]["volume_L"], candidates[1]["volume_L"]
        ) + 1.0
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
        ), patch.object(
            al_driver, "_assert_training_invariants"
        ), patch.object(al_driver, "save_state"):
            al_driver.stage_fine_wait(state)
        self.assertEqual(state["stage"], "FINAL_REPORT")
        self.assertEqual(records["2"]["outcome"], "unverified")


if __name__ == "__main__":
    unittest.main()
