from __future__ import annotations

import gc
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor


REPO = Path(__file__).resolve().parents[1]
TRAINING = REPO / "regression_260707" / "training"
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

import predictor  # noqa: E402
from tools import tier1_semlock_safe_inference_smoke as smoke  # noqa: E402


class _NativeJobsModel:
    def __init__(self, value: float):
        self.value = float(value)
        self.n_jobs = 99

    def predict(self, frame):
        return np.full(len(frame), self.value, dtype=float)


class _CatBoostModel:
    def __init__(self, value: float):
        self.value = float(value)
        self.thread_counts: list[int | None] = []

    def predict(self, frame, thread_count=None):
        self.thread_counts.append(thread_count)
        return np.full(len(frame), self.value, dtype=float)


def _fitted_forests():
    rng = np.random.default_rng(20260722)
    columns = [f"x{index}" for index in range(5)]
    train = pd.DataFrame(rng.normal(size=(128, 5)), columns=columns)
    target = (
        0.45 * train["x0"].to_numpy()
        - 0.2 * train["x1"].to_numpy()
        + train["x2"].to_numpy() ** 2
    )
    frame = pd.DataFrame(rng.normal(size=(48, 5)), columns=columns)
    extra = ExtraTreesRegressor(
        n_estimators=32,
        max_depth=8,
        random_state=20260722,
        n_jobs=4,
    ).fit(train, target)
    random = RandomForestRegressor(
        n_estimators=32,
        max_depth=8,
        random_state=20260722,
        n_jobs=4,
    ).fit(train, target)
    return frame, extra, random


def test_family_safe_binding_stress_has_parity_and_no_resource_growth():
    frame, extra, random = _fitted_forests()
    extra_parallel = extra.predict(frame)
    random_parallel = random.predict(frame)
    lightgbm = _NativeJobsModel(1.0)
    xgboost = _NativeJobsModel(2.0)
    catboost = _CatBoostModel(3.0)
    ensemble = predictor.EnsemblePredictor({
        "models": [
            ("lightgbm", lightgbm),
            ("xgboost", xgboost),
            ("catboost", catboost),
            ("extratrees", extra),
            ("random_forest", random),
        ],
        "features": list(frame.columns),
        "transform": "identity",
        "q90": 1.0,
    })

    receipt = smoke.attest_predictor(
        ensemble, frame, threads=4, repeats=64
    )

    assert receipt["status"] == "passed"
    assert receipt["family_threads"] == {
        "catboost": 4,
        "extratrees": 1,
        "lightgbm": 4,
        "randomforest": 1,
        "xgboost": 4,
    }
    assert receipt["semaphore_free_families"] == [
        "extratrees", "randomforest",
    ]
    assert receipt["forbidden_constructor_calls"] == {
        "joblib_threadpool": 0,
        "multiprocessing_semlock": 0,
    }
    assert set(receipt["positive_resource_growth"].values()) <= {None, 0}
    assert receipt["repeated_prediction_bitwise_equal"] is True
    assert extra.n_jobs == 1
    assert random.n_jobs == 1
    assert lightgbm.n_jobs == 4
    assert xgboost.n_jobs == 4
    assert set(catboost.thread_counts) == {4}

    extra_serial = extra.predict(frame)
    random_serial = random.predict(frame)
    np.testing.assert_allclose(
        extra_serial, extra_parallel, rtol=1e-15, atol=1e-15
    )
    np.testing.assert_allclose(
        random_serial, random_parallel, rtol=1e-15, atol=1e-15
    )

    gc.collect()
    final_resources = smoke.process_resource_snapshot()
    assert final_resources["process_threads"] <= (
        receipt["resource_before"]["process_threads"]
    )


def test_mislabeled_sklearn_forest_fails_closed_before_prediction():
    frame, extra, _random = _fitted_forests()
    ensemble = predictor.EnsemblePredictor({
        "models": [("lightgbm", extra)],
        "features": list(frame.columns),
        "transform": "identity",
        "q90": 1.0,
    })

    with pytest.raises(RuntimeError, match="family label"):
        ensemble.configure_inference_threads(4)


def test_remote_smoke_rejects_positive_process_resource_growth():
    frame, extra, _random = _fitted_forests()
    ensemble = predictor.EnsemblePredictor({
        "models": [("extratrees", extra)],
        "features": list(frame.columns),
        "transform": "identity",
        "q90": 1.0,
    })
    snapshots = iter([
        {"process_threads": 2, "process_file_descriptors": 4},
        {"process_threads": 3, "process_file_descriptors": 4},
    ])

    with pytest.raises(RuntimeError, match="resource inventory grew"):
        smoke.attest_predictor(
            ensemble,
            frame,
            threads=4,
            repeats=2,
            resource_probe=lambda: next(snapshots),
        )
