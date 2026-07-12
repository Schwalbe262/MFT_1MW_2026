import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from regression_260707.training import checkpoint_train
from regression_260707.training import train_models


class _MeanRegressor:
    def __init__(self, **_kwargs):
        self.value = 0.0

    def fit(self, _features, target):
        self.value = float(np.mean(target))
        return self

    def predict(self, features):
        return np.full(len(features), self.value, dtype=float)


def test_cv_metrics_has_exact_zero_aware_metric_contract(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "lightgbm",
        SimpleNamespace(LGBMRegressor=_MeanRegressor),
    )
    features = pd.DataFrame({"x": np.arange(10, dtype=float)})
    actual = np.asarray([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    metrics, predicted = checkpoint_train.cv_metrics(
        features,
        actual,
        kind=None,
        n_splits=5,
        return_yhat=True,
    )

    assert set(metrics) == {
        "r2",
        "rmse",
        "mape_pct",
        "p90_ape_pct",
        "mape_n",
        "mape_excluded_zero_count",
        "mape_zero_abs_tolerance",
    }
    assert len(predicted) == len(actual)
    assert metrics["mape_n"] == 8
    assert metrics["mape_excluded_zero_count"] == 2
    assert metrics["mape_zero_abs_tolerance"] == 1e-9
    assert np.isfinite(metrics["r2"])
    assert np.isfinite(metrics["rmse"])


def test_structural_zero_is_excluded_without_dropping_rmse_inputs():
    metrics = checkpoint_train._zero_aware_percentage_metrics(
        [0.0, 10.0],
        [5.0, 11.0],
    )

    assert metrics == {
        "mape_pct": 10.0,
        "p90_ape_pct": 10.0,
        "mape_n": 1,
        "mape_excluded_zero_count": 1,
        "mape_zero_abs_tolerance": 1e-9,
    }
    # The zero row is still present in the ordinary error vector used by
    # cv_metrics for R2/RMSE; only percentage denominators are masked.
    error = np.asarray([5.0, 1.0])
    assert float(np.sqrt(np.mean(error ** 2))) > 3.0


def test_parity_sampling_is_deterministic_and_keeps_endpoints():
    actual = np.arange(10, dtype=float)
    predicted = actual + 0.25
    parity = checkpoint_train._parity_target(
        actual,
        predicted,
        row_index=np.arange(100, 110),
        limit=4,
    )

    assert parity["n"] == 10
    assert parity["sample_count"] == 4
    assert parity["sampling"] == {
        "method": "evenly_spaced_position",
        "limit": 4,
    }
    assert [pair["row_position"] for pair in parity["pairs"]] == [0, 3, 6, 9]
    assert [pair["row_index"] for pair in parity["pairs"]] == [100, 103, 106, 109]


def test_train_models_keeps_zero_rows_for_training_r2_and_rmse(monkeypatch):
    from sklearn.model_selection import train_test_split

    monkeypatch.setattr(
        train_models,
        "make_model",
        lambda _family, _params, seed: _MeanRegressor(random_state=seed),
    )
    count = 90
    target = np.where(
        np.arange(count) % 3,
        np.arange(count, dtype=float) + 1.0,
        0.0,
    )
    frame = pd.DataFrame({
        "x": np.arange(count, dtype=float),
        "P_Rx_side_total": target,
        "_strict_valid_full": True,
    })

    bundle, metrics = train_models.train_target(
        frame,
        ["x"],
        "P_Rx_side_total",
        {"transform": None},
        {"fake": {}},
        min_rows=20,
    )

    indices = np.arange(count)
    _, holdout = train_test_split(
        indices,
        test_size=train_models.CALIBRATION_FRAC + train_models.EVALUATION_FRAC,
        random_state=train_models.SEED,
    )
    _, evaluation = train_test_split(
        holdout,
        test_size=(
            train_models.EVALUATION_FRAC
            / (train_models.CALIBRATION_FRAC + train_models.EVALUATION_FRAC)
        ),
        random_state=train_models.SEED + 1,
    )
    predicted, _ = train_models._ensemble_prediction(
        bundle["models"],
        frame[["x"]].iloc[evaluation],
        transform=None,
    )
    expected_rmse = float(
        np.sqrt(np.mean((predicted - target[evaluation]) ** 2))
    )

    assert metrics["rmse"] == expected_rmse
    assert metrics["mape_excluded_zero_count"] == int(
        np.count_nonzero(np.abs(target[evaluation]) <= 1e-9)
    )
    assert metrics["mape_excluded_zero_count"] > 0
    assert metrics["mape_n"] + metrics["mape_excluded_zero_count"] == len(
        evaluation
    )
