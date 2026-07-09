"""
본 학습 파이프라인: 타겟별 이종 4패밀리 x 5-fold cross-fit 앙상블 + 컨포멀 불확실성.

- 패밀리: LightGBM / XGBoost / CatBoost / ExtraTrees (타겟당 20 예측기)
- 예측 = 전 예측기 median, sigma = 예측기 spread(표준편차)
- 컨포멀 보정: 홀드아웃(10%)에서 s=|y-mu|/sigma 의 q90 -> 보정 반폭 w(x)=q90*sigma(x)
- 산출: registry/<target>/ {models.pkl, meta.json} (predictor.py가 로드)

사용:
  python train_models.py                       # 전 타겟
  python train_models.py --targets Llt_phys    # 일부만
  python train_models.py --params best_params.json   # Optuna 튜닝 결과 반영
"""
import argparse
import json
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "..", "data", "dataset", "train.parquet")
REGISTRY = os.path.join(HERE, "registry")

from checkpoint_train import (  # noqa: E402 (동일 디렉토리)
    TARGETS, to_physical, feature_columns, transform_y, inverse_y,
)

N_FOLDS = 5
HOLDOUT_FRAC = 0.10
SEED = 42


def default_family_params():
    return {
        "lightgbm": dict(n_estimators=1200, learning_rate=0.04, num_leaves=63,
                         subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, verbose=-1),
        "xgboost": dict(n_estimators=1200, learning_rate=0.04, max_depth=8,
                        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, verbosity=0),
        "catboost": dict(iterations=1200, learning_rate=0.04, depth=8, verbose=0),
        "extratrees": dict(n_estimators=600, min_samples_leaf=2, n_jobs=-1),
    }


def make_model(family, params, seed):
    if family == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(random_state=seed, **params)
    if family == "xgboost":
        import xgboost as xgb
        return xgb.XGBRegressor(random_state=seed, **params)
    if family == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(random_seed=seed, allow_writing_files=False, **params)
    if family == "extratrees":
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(random_state=seed, **params)
    raise ValueError(family)


def train_target(df, feats, target, cfg, family_params, sample_weight_col=None, min_rows=200):
    from sklearn.model_selection import KFold, train_test_split

    # 온도 타겟: thermal 솔브가 성공한 행만 (thermal_solved 플래그, 2026-07-09)

    if target.startswith('Tprobe') and 'thermal_solved' in df.columns:

        df = df[df['thermal_solved'].fillna(0) == 1]

    sub = df.dropna(subset=[target])
    sub = sub[np.isfinite(sub[target])]
    if len(sub) < min_rows:
        return None, f"insufficient rows ({len(sub)})"

    X = sub[feats].fillna(0.0).reset_index(drop=True)
    y_raw = sub[target].to_numpy(dtype=float)
    w = (sub[sample_weight_col].fillna(1.0).to_numpy(dtype=float)
         if sample_weight_col and sample_weight_col in sub.columns else np.ones(len(sub)))
    kind = cfg["transform"]
    y = transform_y(y_raw, kind)

    idx_tr, idx_ho = train_test_split(np.arange(len(X)), test_size=HOLDOUT_FRAC, random_state=SEED)
    Xtr, ytr, wtr = X.iloc[idx_tr], y[idx_tr], w[idx_tr]
    Xho, yho_raw = X.iloc[idx_ho], y_raw[idx_ho]

    models = []
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for family, params in family_params.items():
        for f_i, (tr, _) in enumerate(kf.split(Xtr)):
            m = make_model(family, params, seed=SEED + f_i)
            try:
                m.fit(Xtr.iloc[tr], ytr[tr], sample_weight=wtr[tr])
            except TypeError:
                m.fit(Xtr.iloc[tr], ytr[tr])
            models.append((family, m))

    # 홀드아웃 예측 -> 지표 + 컨포멀 q90
    preds_t = np.stack([m.predict(Xho) for _, m in models])   # (n_models, n_ho) 변환공간
    mu_t = np.median(preds_t, axis=0)
    sg_t = preds_t.std(axis=0)
    mu = inverse_y(mu_t, kind)
    # sigma를 원공간으로 근사 전파 (1차): |d inverse/d t| * sg_t
    eps = 1e-9
    deriv = np.abs(inverse_y(mu_t + 1e-4, kind) - inverse_y(mu_t - 1e-4, kind)) / 2e-4
    sg = np.maximum(deriv * sg_t, eps)

    err = mu - yho_raw
    rel = np.abs(err) / np.clip(np.abs(yho_raw), 1e-9, None)
    scores = np.abs(err) / sg
    q90 = float(np.quantile(scores, 0.90))

    metrics = {
        "n_train": int(len(idx_tr)), "n_holdout": int(len(idx_ho)),
        "r2": float(1 - np.sum(err**2) / (np.sum((yho_raw - yho_raw.mean())**2) or 1e-12)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape_pct": float(np.mean(rel) * 100),
        "p90_ape_pct": float(np.quantile(rel, 0.9) * 100),
        "q90_conformal": q90,
    }
    bundle = {"models": models, "features": feats, "transform": kind,
              "q90": q90, "target": target, "metrics": metrics,
              "trained_at": datetime.now().isoformat(timespec="seconds")}
    return bundle, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--params", default=None, help="패밀리별 하이퍼파라미터 JSON (Optuna 결과)")
    ap.add_argument("--weight-col", default="sample_weight")
    ap.add_argument("--min-rows", type=int, default=200,
                    help="타겟당 최소 행 수 (리허설용으로 낮출 수 있음)")
    args = ap.parse_args()

    df = to_physical(pd.read_parquet(args.dataset))
    feats = feature_columns(df)
    base_params = default_family_params()
    tuned = {}
    if args.params and os.path.isfile(args.params):
        tuned = json.load(open(args.params))

    def fam_params_for(target):
        """tune_optuna 구조({family: {target: {params: ...}}}) 또는 flat({family: params}) 지원"""
        out = {}
        for fam, base in base_params.items():
            p = dict(base)
            spec = tuned.get(fam, {})
            if target in spec and isinstance(spec[target], dict) and "params" in spec[target]:
                p.update(spec[target]["params"])
            elif spec and all(not isinstance(v, dict) for v in spec.values()):
                p.update(spec)  # flat 구조
            out[fam] = p
        return out

    os.makedirs(REGISTRY, exist_ok=True)
    targets = args.targets or list(TARGETS.keys())
    report = {}
    for t in targets:
        if t not in df.columns:
            print(f"[skip] {t}: 컬럼 없음")
            continue
        bundle, metrics = train_target(df, feats, t, TARGETS[t], fam_params_for(t), args.weight_col,
                                       min_rows=args.min_rows)
        if bundle is None:
            print(f"[skip] {t}: {metrics}")
            continue
        tdir = os.path.join(REGISTRY, t)
        os.makedirs(tdir, exist_ok=True)
        with open(os.path.join(tdir, "models.pkl"), "wb") as f:
            pickle.dump(bundle, f)
        with open(os.path.join(tdir, "meta.json"), "w") as f:
            json.dump({k: v for k, v in bundle.items() if k not in ("models",)}, f, indent=1, default=str)
        report[t] = metrics
        print(f"{t:32s} R2={metrics['r2']:.4f} MAPE={metrics['mape_pct']:.2f}% "
              f"P90={metrics['p90_ape_pct']:.2f}% q90={metrics['q90_conformal']:.2f}")

    with open(os.path.join(REGISTRY, "train_report.json"), "w") as f:
        json.dump({"time": datetime.now().isoformat(), "report": report}, f, indent=1)


if __name__ == "__main__":
    main()
