import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from regression_260707.training.model_quality_gate import evaluate_generation
from regression_260707.training.checkpoint_train import TARGETS


CAPACITANCE_TARGETS = ("C_tx_tx_F", "C_rx_rx_F", "C_tx_rx_F")


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
        model_path.write_bytes(b"model")
        meta_path = target_dir / "meta.json"
        meta_path.write_text(
            json.dumps({
                "training_run_id": "candidate",
                "dataset_sha256": dataset_sha256,
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
            "features": ["N1_main"],
            "artifacts": artifacts,
        }),
        encoding="utf-8",
    )
    return registry, generation, dataset


class CapacitanceQualityThresholdTests(unittest.TestCase):
    def test_capacitance_thresholds_are_loose_and_advisory(self):
        threshold_path = Path(__file__).with_name("model_quality_thresholds.json")
        targets = json.loads(threshold_path.read_text(encoding="utf-8"))["targets"]
        self.assertEqual(set(targets), set(TARGETS))

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


if __name__ == "__main__":
    unittest.main()
