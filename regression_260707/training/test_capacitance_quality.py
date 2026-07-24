import hashlib
import json
import pickle
import tempfile
import unittest
from pathlib import Path

from regression_260707.training.model_quality_gate import evaluate_generation
from regression_260707.training.checkpoint_train import TRAINABLE_TARGETS
from regression_260707.training.capacitance_recovery_guard import (
    CAPACITANCE_RECOVERY_CONTRACT,
)
from regression_260707.training.train_models import (
    promote_generation,
    registry_pointer_token,
)


CAPACITANCE_TARGETS = ("C_tx_tx_F", "C_rx_rx_F", "C_tx_rx_F")
RECOVERY_EVIDENCE = {
    "contract": CAPACITANCE_RECOVERY_CONTRACT,
    "recovered_row_count": 3000,
    "max_observed_abs_delta_F": 4.9e-11,
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_candidate(root, metrics_by_target):
    root = Path(root)
    registry = root / "registry"
    generation = registry / "generations" / "candidate"
    generation.mkdir(parents=True)
    dataset = root / "snapshot.parquet"
    dataset.write_bytes(b"stable snapshot")
    dataset_sha256 = _sha256(dataset)
    artifacts = {}

    for target, metrics in metrics_by_target.items():
        target_dir = generation / target
        target_dir.mkdir()
        model_path = target_dir / "models.pkl"
        bundle = {
            "training_run_id": "candidate",
            "dataset_sha256": dataset_sha256,
            "profile_sha256": "profile-sha",
            "capacitance_recovery": RECOVERY_EVIDENCE,
        }
        with model_path.open("wb") as handle:
            pickle.dump(bundle, handle)
        meta_path = target_dir / "meta.json"
        meta_path.write_text(
            json.dumps({
                "training_run_id": "candidate",
                "dataset_sha256": dataset_sha256,
                "profile_sha256": "profile-sha",
                "capacitance_recovery": RECOVERY_EVIDENCE,
                "features": ["N1_main"],
                "metrics": metrics,
            }),
            encoding="utf-8",
        )
        artifacts[f"{target}/models.pkl"] = _sha256(model_path)
        artifacts[f"{target}/meta.json"] = _sha256(meta_path)

    (generation / "train_report.json").write_text(
        json.dumps({
            "training_run_id": "candidate",
            "dataset_sha256": dataset_sha256,
            "profile_sha256": "profile-sha",
            "strict_full_rows": 3000,
            "capacitance_recovery": RECOVERY_EVIDENCE,
            "features": ["N1_main"],
            "targets": list(metrics_by_target),
            "report": metrics_by_target,
            "artifacts": artifacts,
        }),
        encoding="utf-8",
    )
    return registry, generation, dataset


def _rewrite_target_artifacts(generation, target, update):
    generation = Path(generation)
    report_path = generation / "train_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    meta_path = generation / target / "meta.json"
    model_path = generation / target / "models.pkl"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with model_path.open("rb") as handle:
        bundle = pickle.load(handle)
    update(meta)
    update(bundle)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with model_path.open("wb") as handle:
        pickle.dump(bundle, handle)
    report["artifacts"][f"{target}/meta.json"] = _sha256(meta_path)
    report["artifacts"][f"{target}/models.pkl"] = _sha256(model_path)
    report_path.write_text(json.dumps(report), encoding="utf-8")


def _remove_all_recovery_evidence(generation):
    generation = Path(generation)
    report_path = generation / "train_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.pop("capacitance_recovery", None)
    for target in report["targets"]:
        meta_path = generation / target / "meta.json"
        model_path = generation / target / "models.pkl"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        with model_path.open("rb") as handle:
            bundle = pickle.load(handle)
        meta.pop("capacitance_recovery", None)
        bundle.pop("capacitance_recovery", None)
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with model_path.open("wb") as handle:
            pickle.dump(bundle, handle)
        report["artifacts"][f"{target}/meta.json"] = _sha256(meta_path)
        report["artifacts"][f"{target}/models.pkl"] = _sha256(model_path)
    report_path.write_text(json.dumps(report), encoding="utf-8")


class CapacitanceQualityThresholdTests(unittest.TestCase):
    def test_capacitance_thresholds_are_loose_and_advisory(self):
        threshold_path = Path(__file__).with_name("model_quality_thresholds.json")
        targets = json.loads(threshold_path.read_text(encoding="utf-8"))["targets"]
        self.assertEqual(set(targets), set(TRAINABLE_TARGETS))

        expected = {
            "blocking": False,
            "min_r2": 0.90,
            "max_normalized_rmse_pct": 20.0,
        }
        for target in CAPACITANCE_TARGETS:
            with self.subTest(target=target):
                self.assertEqual(targets[target], expected)

    def test_failed_advisory_target_does_not_fail_generation(self):
        passing = {
            "r2": 0.99,
            "normalized_rmse_pct": 1.0,
            "interval_coverage": 0.90,
        }
        failing = {
            "r2": 0.10,
            "normalized_rmse_pct": 80.0,
            "interval_coverage": 0.50,
        }
        thresholds = {
            "minimum_strict_full_rows": 3000,
            "minimum_interval_coverage": 0.85,
            "targets": {
                "Llt_phys": {
                    "min_r2": 0.90,
                    "max_normalized_rmse_pct": 20.0,
                },
                "C_tx_tx_F": {
                    "blocking": False,
                    "min_r2": 0.90,
                    "max_normalized_rmse_pct": 20.0,
                },
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            registry, generation, dataset = _write_candidate(
                directory,
                {"Llt_phys": passing, "C_tx_tx_F": failing},
            )
            result = evaluate_generation(
                str(registry), str(generation), str(dataset), thresholds
            )

        self.assertTrue(result["passed"], result["reasons"])
        self.assertEqual(result["reasons"], [])
        self.assertTrue(result["targets"]["Llt_phys"]["blocking"])
        self.assertTrue(result["targets"]["Llt_phys"]["passed"])
        self.assertFalse(result["targets"]["C_tx_tx_F"]["blocking"])
        self.assertFalse(result["targets"]["C_tx_tx_F"]["passed"])
        self.assertEqual(
            result["advisories"],
            [
                "C_tx_tx_F:interval_coverage_below_minimum",
                "C_tx_tx_F:metric_below_minimum:r2",
                "C_tx_tx_F:metric_above_maximum:normalized_rmse_pct",
            ],
        )

    def test_targets_remain_blocking_by_default(self):
        failing = {
            "r2": 0.10,
            "normalized_rmse_pct": 80.0,
            "interval_coverage": 0.90,
        }
        thresholds = {
            "minimum_strict_full_rows": 3000,
            "minimum_interval_coverage": 0.85,
            "targets": {
                "Llt_phys": {
                    "min_r2": 0.90,
                    "max_normalized_rmse_pct": 20.0,
                },
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            registry, generation, dataset = _write_candidate(
                directory, {"Llt_phys": failing}
            )
            result = evaluate_generation(
                str(registry), str(generation), str(dataset), thresholds
            )

        self.assertFalse(result["passed"])
        self.assertTrue(result["targets"]["Llt_phys"]["blocking"])
        self.assertIn("Llt_phys:metric_below_minimum:r2", result["reasons"])
        self.assertEqual(result["advisories"], [])


class CapacitanceRecoveryPromotionGuardTests(unittest.TestCase):
    def _thresholds(self, targets=("Llt_phys",)):
        return {
            "minimum_strict_full_rows": 3000,
            "minimum_interval_coverage": 0.85,
            "targets": {
                target: {"min_r2": 0.90, "max_normalized_rmse_pct": 20.0}
                for target in targets
            },
        }

    def _metrics(self):
        return {
            "r2": 0.99,
            "normalized_rmse_pct": 1.0,
            "interval_coverage": 0.90,
        }

    def test_legacy_candidate_fails_quality_and_forged_pass_cannot_promote(self):
        with tempfile.TemporaryDirectory() as directory:
            registry, generation, dataset = _write_candidate(
                directory, {"Llt_phys": self._metrics()}
            )
            _remove_all_recovery_evidence(generation)
            thresholds = self._thresholds()
            quality = evaluate_generation(
                str(registry), str(generation), str(dataset), thresholds
            )
            self.assertFalse(quality["passed"])
            self.assertIn(
                "capacitance_recovery:train_report:evidence_missing",
                quality["reasons"],
            )

            forged = dict(quality, passed=True)
            with self.assertRaisesRegex(
                RuntimeError, "capacitance recovery provenance rejected"
            ):
                promote_generation(
                    str(registry),
                    str(generation),
                    forged,
                    dataset=str(dataset),
                    profile_sha256="profile-sha",
                    thresholds_sha256=quality["thresholds_sha256"],
                    expected_pointer=registry_pointer_token(str(registry)),
                )
            self.assertFalse((Path(registry) / "current.json").exists())

    def test_unthresholded_target_recovery_mismatch_is_blocking(self):
        with tempfile.TemporaryDirectory() as directory:
            registry, generation, dataset = _write_candidate(
                directory,
                {
                    "Llt_phys": self._metrics(),
                    "C_tx_tx_F": self._metrics(),
                },
            )

            def mismatch(payload):
                payload["capacitance_recovery"] = {
                    **payload["capacitance_recovery"],
                    "recovered_row_count": 2999,
                }

            _rewrite_target_artifacts(generation, "C_tx_tx_F", mismatch)
            quality = evaluate_generation(
                str(registry),
                str(generation),
                str(dataset),
                self._thresholds(("Llt_phys",)),
            )
            self.assertFalse(quality["passed"])
            self.assertTrue(any(
                reason.startswith(
                    "capacitance_recovery:C_tx_tx_F:"
                ) and reason.endswith("recovery_identity_mismatch")
                for reason in quality["reasons"]
            ))

    def test_model_bundle_dataset_and_profile_identity_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            registry, generation, dataset = _write_candidate(
                directory, {"Llt_phys": self._metrics()}
            )

            def mismatch(payload):
                payload["dataset_sha256"] = "wrong-dataset"
                payload["profile_sha256"] = "wrong-profile"

            _rewrite_target_artifacts(generation, "Llt_phys", mismatch)
            quality = evaluate_generation(
                str(registry), str(generation), str(dataset), self._thresholds()
            )
            self.assertFalse(quality["passed"])
            self.assertIn(
                "capacitance_recovery:Llt_phys:model_bundle:dataset_identity_mismatch",
                quality["reasons"],
            )
            self.assertIn(
                "capacitance_recovery:Llt_phys:model_bundle:profile_identity_mismatch",
                quality["reasons"],
            )

    def test_delta_above_quantization_bound_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            registry, generation, dataset = _write_candidate(
                directory, {"Llt_phys": self._metrics()}
            )
            generation = Path(generation)
            report_path = generation / "train_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["capacitance_recovery"][
                "max_observed_abs_delta_F"
            ] = 5.1000001e-11
            report_path.write_text(json.dumps(report), encoding="utf-8")
            quality = evaluate_generation(
                str(registry), str(generation), str(dataset), self._thresholds()
            )
            self.assertFalse(quality["passed"])
            self.assertIn(
                "capacitance_recovery:train_report:"
                "max_observed_abs_delta_F_out_of_bounds",
                quality["reasons"],
            )


if __name__ == "__main__":
    unittest.main()
