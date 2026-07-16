import sys
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


if __name__ == "__main__":
    unittest.main()
