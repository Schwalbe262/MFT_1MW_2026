"""
Optuna 하이퍼파라미터 튜닝 (타겟 x 패밀리, TPE + MedianPruner).

데이터가 충분히 쌓인 뒤(>=4k) 1회 크게 돌리고, 결과 best_params.json은
train_models.py --params로 주입. AL 라운드 중에는 고정 (재학습만).

사용:
  python tune_optuna.py --target Llt_phys --family lightgbm --trials 300
  python tune_optuna.py --all --trials 200          # 전 타겟/패밀리 순차
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "..", "data", "dataset", "train.parquet")
OUT = os.path.join(HERE, "best_params.json")

from checkpoint_train import (  # noqa: E402
    TARGETS,
    feature_columns,
    filter_valid_training_rows,
    to_physical,
    transform_y,
)
from train_models import make_model  # noqa: E402


def sample_params(trial, family):
    if family == "lightgbm":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 400, 3000),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            num_leaves=trial.suggest_int("num_leaves", 31, 255),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 60),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            verbose=-1)
    if family == "xgboost":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 400, 3000),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            max_depth=trial.suggest_int("max_depth", 4, 12),
            min_child_weight=trial.suggest_float("min_child_weight", 1, 20),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            verbosity=0)
    if family == "catboost":
        return dict(
            iterations=trial.suggest_int("iterations", 400, 3000),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            depth=trial.suggest_int("depth", 4, 10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.5, 30, log=True),
            verbose=0)
    if family == "extratrees":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1500),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_float("max_features", 0.4, 1.0),
            n_jobs=-1)
    raise ValueError(family)


def tune(target, family, trials, df, feats):
    import optuna
    from sklearn.model_selection import KFold

    df = filter_valid_training_rows(df, target)
    sub = df.dropna(subset=[target])
    sub = sub[np.isfinite(sub[target])]
    X = sub[feats].fillna(0.0).reset_index(drop=True)
    y = transform_y(sub[target].to_numpy(dtype=float), TARGETS[target]["transform"])

    kf = KFold(n_splits=4, shuffle=True, random_state=0)

    def objective(trial):
        params = sample_params(trial, family)
        errs = []
        for f_i, (tr, te) in enumerate(kf.split(X)):
            m = make_model(family, params, seed=f_i)
            m.fit(X.iloc[tr], y[tr])
            p = m.predict(X.iloc[te])
            errs.append(float(np.mean((p - y[te]) ** 2)))
            trial.report(float(np.mean(errs)), f_i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(errs))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=7),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return study.best_params, study.best_value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=None)
    ap.add_argument("--family", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--dataset", default=DATASET)
    args = ap.parse_args()

    df = to_physical(pd.read_parquet(args.dataset))
    feats = feature_columns(df)

    best = json.load(open(OUT)) if os.path.isfile(OUT) else {}
    jobs = ([(args.target, args.family)] if args.target and args.family else
            [(t, f) for t in TARGETS if t in df.columns
             for f in ("lightgbm", "xgboost", "catboost", "extratrees")] if args.all else [])
    if not jobs:
        raise SystemExit("--target+--family 또는 --all 지정")

    for t, f in jobs:
        print(f"\n=== tune {t} / {f} ({args.trials} trials) ===")
        params, val = tune(t, f, args.trials, df, feats)
        best.setdefault(f, {})
        # 타겟별 최적을 패밀리 공통으로 쓰지 않고 타겟별 저장
        best[f][t] = {"params": params, "cv_mse_transformed": val}
        json.dump(best, open(OUT, "w"), indent=1)
        print(f"best: {params}")

    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
