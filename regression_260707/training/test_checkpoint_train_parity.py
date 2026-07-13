import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from regression_260707.training import checkpoint_train as checkpoint


class CheckpointParityTests(unittest.TestCase):
    def test_target_cohort_rejects_physics_revision_mixing(self):
        frame = pd.DataFrame({
            "target": [1.0, 2.0],
            "_strict_valid_full": [True, True],
            "physics_data_revision": ["legacy-a", "native-b"],
        })
        with self.assertRaisesRegex(
            RuntimeError, "mixes physics_data_revision cohorts"
        ):
            checkpoint.filter_valid_training_rows(frame, "target")

        legacy = frame.drop(columns=["physics_data_revision"])
        filtered = checkpoint.filter_valid_training_rows(legacy, "target")
        self.assertEqual(
            filtered.attrs["physics_data_revision_cohort"],
            checkpoint.LEGACY_PHYSICS_DATA_REVISION,
        )

    def test_cv_metrics_default_contract_and_optional_yhat(self):
        lightgbm = types.ModuleType("lightgbm")
        sklearn = types.ModuleType("sklearn")
        model_selection = types.ModuleType("sklearn.model_selection")
        sklearn.__path__ = []

        class FakeRegressor:
            def __init__(self, **_kwargs):
                self.value = None

            def fit(self, _x, y):
                self.value = float(np.mean(y))

            def predict(self, x):
                return np.full(len(x), self.value)

        class FakeKFold:
            def __init__(self, n_splits, shuffle, random_state):
                self.n_splits = n_splits
                self.shuffle = shuffle
                self.random_state = random_state

            def split(self, x):
                indexes = np.arange(len(x))
                for fold in range(self.n_splits):
                    test = indexes[indexes % self.n_splits == fold]
                    train = indexes[indexes % self.n_splits != fold]
                    yield train, test

        lightgbm.LGBMRegressor = FakeRegressor
        model_selection.KFold = FakeKFold
        sklearn.model_selection = model_selection
        modules = {
            "lightgbm": lightgbm,
            "sklearn": sklearn,
            "sklearn.model_selection": model_selection,
        }
        x = pd.DataFrame({"x": np.arange(10, dtype=float)})
        y = np.arange(1, 11, dtype=float)
        with mock.patch.dict(sys.modules, modules):
            default = checkpoint.cv_metrics(x, y, None)
            metrics, yhat = checkpoint.cv_metrics(
                x, y, None, return_yhat=True
            )

        self.assertEqual(default, metrics)
        # The zero-aware relative-error union adds MAPE evidence keys on top
        # of the original four-metric contract.
        self.assertLessEqual(
            {"r2", "rmse", "mape_pct", "p90_ape_pct"}, set(default)
        )
        self.assertLessEqual(
            {"mape_n", "mape_excluded_zero_count", "mape_zero_abs_tolerance"},
            set(default),
        )
        self.assertEqual(len(yhat), len(y))
        self.assertTrue(np.isfinite(yhat).all())

    def test_parity_sampling_is_deterministic_and_bounded(self):
        y = np.arange(2001, dtype=float)
        first = checkpoint._parity_target(y, y + 0.25, range(len(y)))
        second = checkpoint._parity_target(y, y + 0.25, range(len(y)))

        self.assertEqual(first, second)
        self.assertEqual(first["sample_count"], 2000)
        self.assertEqual(first["sampling"]["method"], "evenly_spaced_position")
        positions = [item["row_position"] for item in first["pairs"]]
        self.assertEqual(positions[0], 0)
        self.assertEqual(positions[-1], 2000)
        self.assertEqual(len(positions), len(set(positions)))

        all_rows = checkpoint._parity_target(
            y[:2000], y[:2000] + 0.25, range(2000)
        )
        self.assertEqual(all_rows["sample_count"], 2000)
        self.assertEqual(all_rows["sampling"]["method"], "all")

    def test_parity_only_cli_requires_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            sys, "argv", ["checkpoint_train.py", "--parity-json", str(Path(directory) / "parity.json")]
        ), self.assertRaises(SystemExit) as raised:
            checkpoint.main()
        self.assertEqual(raised.exception.code, 2)

    def test_parity_only_cli_writes_atomic_sidecar_without_curve_append(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "snapshot.parquet"
            dataset.write_bytes(b"snapshot")
            profile = root / "profile.json"
            profile.write_text("{}", encoding="utf-8")
            parity = root / "metrics.parity.json"
            curve = root / "learning_curve.csv"
            frame = pd.DataFrame({
                "feature": np.arange(100, dtype=float),
                "Llt_phys": np.linspace(20.0, 40.0, 100),
                "_strict_valid_full": [True] * 100,
            })

            def fake_cv(_x, y, _kind, return_yhat=False, **_kwargs):
                metrics = {
                    "r2": 0.9,
                    "rmse": 0.1,
                    "mape_pct": 1.0,
                    "p90_ape_pct": 2.0,
                }
                yhat = np.asarray(y, dtype=float) + 0.5
                return (metrics, yhat) if return_yhat else metrics

            argv = [
                "checkpoint_train.py",
                "--dataset", str(dataset),
                "--curve-csv", str(curve),
                "--profile", str(profile),
                "--checkpoint", "500",
                "--parity-json", str(parity),
                "--skip-curve-append",
            ]
            import quality_contract

            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                checkpoint.pd, "read_parquet", return_value=frame
            ), mock.patch.object(
                quality_contract, "annotate_validity", return_value=frame
            ), mock.patch.object(
                quality_contract, "load_profile", return_value={}
            ), mock.patch.object(
                checkpoint, "to_physical", return_value=frame
            ), mock.patch.object(
                checkpoint, "feature_columns", return_value=["feature"]
            ), mock.patch.object(
                checkpoint, "TARGETS", {"Llt_phys": {"transform": None}}
            ), mock.patch.object(
                checkpoint, "cv_metrics", side_effect=fake_cv
            ):
                checkpoint.main()

            payload = json.loads(parity.read_text(encoding="utf-8"))
            self.assertFalse(curve.exists())
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["artifact_type"], "checkpoint_cv_oof_parity")
            self.assertEqual(payload["checkpoint"], 500)
            self.assertEqual(payload["strict_full_rows"], 100)
            self.assertEqual(payload["features"], ["feature"])
            self.assertEqual(set(payload["targets"]), {"Llt_phys"})
            self.assertEqual(
                payload["target_physics_data_revision_cohorts"],
                {"Llt_phys": checkpoint.LEGACY_PHYSICS_DATA_REVISION},
            )
            self.assertEqual(
                payload["targets"]["Llt_phys"][
                    "physics_data_revision_cohort"
                ],
                checkpoint.LEGACY_PHYSICS_DATA_REVISION,
            )
            self.assertNotIn("slice", payload["targets"]["Llt_phys"])
            self.assertEqual(payload["targets"]["Llt_phys"]["sample_count"], 100)
            self.assertEqual(
                payload["targets"]["Llt_phys"]["pairs"][0]["predicted"],
                payload["targets"]["Llt_phys"]["pairs"][0]["actual"] + 0.5,
            )


if __name__ == "__main__":
    unittest.main()
