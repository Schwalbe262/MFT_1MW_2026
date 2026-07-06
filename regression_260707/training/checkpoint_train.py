"""
체크포인트 회귀 학습 + 학습곡선 추적 (데이터 수집 중 상시 모니터링용).

- dataset/train.parquet 로드 -> 실물(_phys) 변환 -> 타겟별 LightGBM 5-fold CV
- 지표(R2/MAPE/RMSE)를 learning_curve.csv에 데이터 개수와 함께 축적
- 관심 슬라이스(Llt_phys 20~40uH) 지표 별도 기록
- 사용: python checkpoint_train.py [--full]  (--full 이면 4패밀리 앙상블까지)
"""
import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "..", "data", "dataset", "train.parquet")
CURVE_CSV = os.path.join(HERE, "learning_curve.csv")

# 회귀 타겟 (실물 기준 컬럼명, 변환 후)
TARGETS = {
    "Llt_phys": {"transform": "log", "metric_focus": "mape"},
    "k": {"transform": None, "metric_focus": "rmse"},
    "P_winding_total": {"transform": "log1p", "metric_focus": "mape"},
    "P_core_total": {"transform": "log1p", "metric_focus": "mape"},
    "P_core_plate_total": {"transform": "log1p", "metric_focus": "mape"},
    "Tprobe_Tx_leeward_max": {"transform": "t50", "metric_focus": "rmse"},
    "Tprobe_Rx_main_leeward_max": {"transform": "t50", "metric_focus": "rmse"},
    "Tprobe_Rx_side_leeward_max": {"transform": "t50", "metric_focus": "rmse"},
    "Tprobe_core_center_max": {"transform": "t50", "metric_focus": "rmse"},
}

# 특징량: 입력 파라미터 + 파생 물리량 (결과/메타 컬럼 제외)
EXCLUDE_PREFIXES = ("Ltx", "Lrx", "M", "k", "Lm", "Ll", "Tx_loss", "Rx_loss", "P_", "B_",
                    "I1_", "phi", "I2_phase_used", "T_", "Tprobe_", "time", "conv_", "mesh_",
                    "git_hash", "project_name", "saved_at", "task_", "fail_")


def to_physical(df):
    """대칭 매트릭스 L 컬럼 -> 실물 (x2). 손실/B는 이미 _phys로 기록됨."""
    out = df.copy()
    sym = out.get("full_model", 0).fillna(0).astype(float) == 0
    for c in ["Ltx", "Lrx", "M", "Lmt", "Lmr", "Llt", "Llr"]:
        if c in out.columns:
            out[f"{c}_phys"] = out[c] * np.where(sym, 2.0, 1.0)
    return out


def feature_columns(df):
    cols = []
    for c in df.columns:
        if any(c.startswith(p) for p in EXCLUDE_PREFIXES):
            continue
        if c.endswith("_phys") or c.endswith("_raw"):
            continue
        if df[c].dtype == object:
            continue
        if df[c].nunique(dropna=True) <= 1:
            continue
        cols.append(c)
    return cols


def transform_y(y, kind):
    if kind == "log":
        return np.log(np.clip(y, 1e-9, None))
    if kind == "log1p":
        return np.log1p(np.clip(y, 0, None))
    if kind == "t50":
        return np.log(np.clip(y - 50.0, 1e-3, None))
    return y


def inverse_y(t, kind):
    if kind == "log":
        return np.exp(t)
    if kind == "log1p":
        return np.expm1(t)
    if kind == "t50":
        return np.exp(t) + 50.0
    return t


def cv_metrics(X, y, kind, n_splits=5, seed=42):
    import lightgbm as lgb
    from sklearn.model_selection import KFold

    yt = transform_y(y, kind)
    preds = np.full(len(y), np.nan)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        model = lgb.LGBMRegressor(
            n_estimators=800, learning_rate=0.05, num_leaves=63,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=seed, verbose=-1)
        model.fit(X.iloc[tr], yt[tr])
        preds[te] = model.predict(X.iloc[te])
    yhat = inverse_y(preds, kind)
    err = yhat - y
    rel = np.abs(err) / np.clip(np.abs(y), 1e-9, None)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1e-12
    return {
        "r2": 1 - ss_res / ss_tot,
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape_pct": float(np.mean(rel) * 100),
        "p90_ape_pct": float(np.quantile(rel, 0.9) * 100),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DATASET)
    args = ap.parse_args()

    df = pd.read_parquet(args.dataset)
    df = to_physical(df)
    feats = feature_columns(df)
    n_total = len(df)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"dataset: {n_total} rows, {len(feats)} features")

    rows = []
    for target, cfg in TARGETS.items():
        if target not in df.columns:
            print(f"  [skip] {target} (컬럼 없음)")
            continue
        sub = df.dropna(subset=[target])
        sub = sub[np.isfinite(sub[target])]
        if len(sub) < 100:
            print(f"  [skip] {target} (n={len(sub)} < 100)")
            continue
        X = sub[feats].fillna(0.0)
        y = sub[target].to_numpy(dtype=float)

        m = cv_metrics(X, y, cfg["transform"])
        row = {"time": stamp, "target": target, "n": len(sub), **m, "slice": "global"}
        rows.append(row)
        print(f"  {target:32s} n={len(sub):6d}  R2={m['r2']:.4f}  MAPE={m['mape_pct']:.2f}%  "
              f"P90APE={m['p90_ape_pct']:.2f}%  RMSE={m['rmse']:.4g}")

        # 관심 슬라이스: Llt_phys 20~40uH 영역
        if "Llt_phys" in sub.columns:
            sl = sub[(sub["Llt_phys"] >= 20) & (sub["Llt_phys"] <= 40)]
            if len(sl) >= 100:
                ms = cv_metrics(sl[feats].fillna(0.0), sl[target].to_numpy(dtype=float), cfg["transform"])
                rows.append({"time": stamp, "target": target, "n": len(sl), **ms, "slice": "Llt20-40"})
                print(f"    └ slice Llt 20-40uH: n={len(sl)}  MAPE={ms['mape_pct']:.2f}%  P90={ms['p90_ape_pct']:.2f}%")

    if rows:
        curve = pd.DataFrame(rows)
        header = not os.path.isfile(CURVE_CSV)
        curve.to_csv(CURVE_CSV, mode="a", header=header, index=False)
        print(f"learning curve appended -> {CURVE_CSV}")


if __name__ == "__main__":
    main()
