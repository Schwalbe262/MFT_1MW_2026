import sys
import threading
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from regression_260707.training import checkpoint_train as checkpoint


class CheckpointModelThreadBudgetTests(unittest.TestCase):
    def test_cv_metrics_applies_explicit_lightgbm_thread_limit_to_every_fold(self):
        lightgbm = types.ModuleType("lightgbm")
        sklearn = types.ModuleType("sklearn")
        model_selection = types.ModuleType("sklearn.model_selection")
        sklearn.__path__ = []
        constructor_arguments = []

        class FakeRegressor:
            def __init__(self, **kwargs):
                constructor_arguments.append(kwargs)
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
        frame = pd.DataFrame({"x": np.arange(20, dtype=float)})
        target = np.arange(1, 21, dtype=float)

        with mock.patch.dict(sys.modules, modules):
            checkpoint.cv_metrics(
                frame,
                target,
                None,
                model_threads=2,
            )

        self.assertEqual(len(constructor_arguments), 5)
        self.assertTrue(
            all(arguments["n_jobs"] == 2 for arguments in constructor_arguments)
        )

    def test_parallelism_rejects_requested_oversubscription(self):
        with self.assertRaisesRegex(
            ValueError,
            r"target_workers \* model_threads exceeds",
        ):
            checkpoint._checkpoint_parallelism(
                model_threads=3,
                target_workers=4,
                max_model_thread_budget=8,
            )

    def test_parallel_target_smoke_is_bounded_and_returns_declaration_order(self):
        targets = {
            f"target_{index}": {"transform": None}
            for index in range(6)
        }
        barrier = threading.Barrier(3)
        lock = threading.Lock()
        active = 0
        maximum_active = 0

        def fake_evaluate(
            _df,
            _feats,
            _profile,
            target,
            _cfg,
            **_kwargs,
        ):
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            try:
                barrier.wait(timeout=5)
                return {
                    "target": target,
                    "rows": [{"target": target}],
                    "parity": None,
                    "revision_cohort": "test",
                    "messages": [target],
                }
            finally:
                with lock:
                    active -= 1

        with mock.patch.object(
            checkpoint,
            "_evaluate_target",
            side_effect=fake_evaluate,
        ):
            results, parallelism = checkpoint._evaluate_targets(
                pd.DataFrame(),
                [],
                "profile.json",
                targets,
                include_parity=False,
                model_threads=2,
                target_workers=3,
                max_model_thread_budget=6,
                stamp="2026-07-18 00:00:00",
            )

        self.assertEqual(maximum_active, 3)
        self.assertEqual(
            [result["target"] for result in results],
            list(targets),
        )
        self.assertEqual(parallelism["effective_target_workers"], 3)
        self.assertEqual(parallelism["effective_total_model_threads"], 6)
        self.assertLessEqual(
            parallelism["effective_total_model_threads"],
            parallelism["max_model_thread_budget"],
        )

    def test_effective_worker_count_is_bounded_by_available_targets(self):
        targets = {
            "first": {"transform": None},
            "second": {"transform": None},
        }

        def fake_evaluate(
            _df,
            _feats,
            _profile,
            target,
            _cfg,
            **_kwargs,
        ):
            return {
                "target": target,
                "rows": [],
                "parity": None,
                "revision_cohort": None,
                "messages": [],
            }

        with mock.patch.object(
            checkpoint,
            "_evaluate_target",
            side_effect=fake_evaluate,
        ):
            _results, parallelism = checkpoint._evaluate_targets(
                pd.DataFrame(),
                [],
                "profile.json",
                targets,
                include_parity=False,
                model_threads=2,
                target_workers=4,
                max_model_thread_budget=8,
                stamp="2026-07-18 00:00:00",
            )

        self.assertEqual(parallelism["effective_target_workers"], 2)
        self.assertEqual(parallelism["effective_total_model_threads"], 4)


if __name__ == "__main__":
    unittest.main()
