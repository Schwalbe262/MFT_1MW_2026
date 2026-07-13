"""
체크포인트 회귀 학습 + 학습곡선 추적 (데이터 수집 중 상시 모니터링용).

- dataset/train.parquet 로드 -> 실물(_phys) 변환 -> 타겟별 LightGBM 5-fold CV
- 지표(R2/MAPE/RMSE)를 learning_curve.csv에 데이터 개수와 함께 축적
- 관심 슬라이스(Llt_phys 20~40uH) 지표 별도 기록
- 사용: python checkpoint_train.py [--full]  (--full 이면 4패밀리 앙상블까지)
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
from filelock import FileLock

HERE = os.path.dirname(os.path.abspath(__file__))
REGRESSION_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if REGRESSION_ROOT not in sys.path:
    sys.path.insert(0, REGRESSION_ROOT)
from model_targets import (
    SURROGATE_TEMPERATURE_TARGETS,
    SURROGATE_WINDING_COMPONENT_LOSS_TARGETS,
)

DATASET = os.path.join(HERE, "..", "data", "dataset", "train.parquet")
CURVE_CSV = os.path.join(HERE, "learning_curve.csv")
MAX_TRUSTED_TEMPERATURE_C = 4700.0
MIN_TRUSTED_TEMPERATURE_C = -273.15
PARITY_SCHEMA_VERSION = 1
PARITY_MAX_PAIRS_PER_TARGET = 2_000
LEGACY_PHYSICS_DATA_REVISION = "legacy_unspecified"
MAPE_ZERO_ABS_TOLERANCE = 1e-9


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(value, path):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, staged = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp",
        dir=os.path.dirname(path),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=1, default=str)
        os.replace(staged, path)
    finally:
        if os.path.exists(staged):
            os.remove(staged)

# 회귀 타겟 (실물 기준 컬럼명, 변환 후)
TARGETS = {
    "Llt_phys": {"transform": "log", "metric_focus": "mape"},
    "k": {"transform": None, "metric_focus": "rmse"},
    "P_winding_total": {"transform": "log1p", "metric_focus": "mape"},
    **{
        target: {"transform": "log1p", "metric_focus": "mape"}
        for target in SURROGATE_WINDING_COMPONENT_LOSS_TARGETS
    },
    "P_core_total": {"transform": "log1p", "metric_focus": "mape"},
    "P_core_plate_total": {"transform": "log1p", "metric_focus": "mape"},
    "P_wcp_total": {"transform": "log1p", "metric_focus": "mape"},
    "B_max_core": {"transform": None, "metric_focus": "rmse"},
    "B_mean_core": {"transform": None, "metric_focus": "rmse"},
    **{
        target: {"transform": "t50", "metric_focus": "rmse"}
        for target in SURROGATE_TEMPERATURE_TARGETS
    },
}

# 특징량: 입력 파라미터 + 파생 물리량 (결과/메타 컬럼 제외)
def filter_valid_training_rows(df, target, profile=None):
    """Return the shared strict-full cohort used by every surrogate target.

    Using one cohort prevents a model from silently learning from legacy EM
    false positives while the temperature models see a different population.
    Validity is recomputed from error, delta, residual, extraction, power
    balance, temperature saturation, profile, and provenance evidence.
    """
    if "_strict_valid_full" not in df.columns:
        from quality_contract import annotate_validity

        df = annotate_validity(df, profile)
    keep = df["_strict_valid_full"].fillna(False).astype(bool)
    if target not in df.columns:
        keep &= False
    else:
        values = pd.to_numeric(df[target], errors="coerce")
        keep &= values.map(np.isfinite)
        if target.startswith("Tprobe"):
            keep &= values.gt(MIN_TRUSTED_TEMPERATURE_C) & values.lt(
                MAX_TRUSTED_TEMPERATURE_C
            )
    filtered = df.loc[keep].copy()
    if filtered.empty:
        filtered.attrs["physics_data_revision_cohort"] = ""
        return filtered
    if "physics_data_revision" in filtered.columns:
        revisions = (
            filtered["physics_data_revision"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", LEGACY_PHYSICS_DATA_REVISION)
        )
    else:
        revisions = pd.Series(
            LEGACY_PHYSICS_DATA_REVISION, index=filtered.index, dtype=object
        )
    cohorts = tuple(sorted(set(revisions.tolist())))
    if len(cohorts) != 1:
        raise RuntimeError(
            f"target {target} mixes physics_data_revision cohorts: {cohorts}"
        )
    filtered.attrs["physics_data_revision_cohort"] = cohorts[0]
    return filtered


def to_physical(df):
    """대칭 매트릭스 L 컬럼 -> 실물 (x2). 손실/B는 이미 _phys로 기록됨."""
    from campaign.train_io import add_wcp_length_features

    out = add_wcp_length_features(df)
    sym = out.get("full_model", 0).fillna(0).astype(float) == 0
    for c in ["Ltx", "Lrx", "M", "Lmt", "Lmr", "Llt", "Llr"]:
        if c in out.columns:
            out[f"{c}_phys"] = out[c] * np.where(sym, 2.0, 1.0)
    return out


def feature_columns(df):
    """Select design-time inputs only; never post-solve quality/output data."""
    from campaign.train_io import (
        DESIGN_INPUT_COLUMNS,
        GEOMETRY_DERIVED_COLUMNS,
        PHYSICAL_CONTEXT_COLUMNS,
    )

    allowed = (
        *DESIGN_INPUT_COLUMNS,
        *PHYSICAL_CONTEXT_COLUMNS,
        *GEOMETRY_DERIVED_COLUMNS,
    )
    return [
        column
        for column in allowed
        if column in df.columns
        and pd.api.types.is_numeric_dtype(df[column])
        and df[column].nunique(dropna=True) > 1
    ]


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


def relative_metric_mask(y_true, tolerance=MAPE_ZERO_ABS_TOLERANCE):
    """Select finite targets whose absolute value makes APE meaningful."""
    values = np.asarray(y_true, dtype=float).reshape(-1)
    return np.isfinite(values) & (np.abs(values) > float(tolerance))


def relative_error_summary(
        y_true, error, tolerance=MAPE_ZERO_ABS_TOLERANCE):
    """Return MAPE evidence without letting zero targets dominate APE."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    residual = np.asarray(error, dtype=float).reshape(-1)
    if len(actual) != len(residual):
        raise ValueError("relative metric target and error lengths differ")
    mask = relative_metric_mask(actual, tolerance=tolerance)
    relative = np.abs(residual[mask]) / np.abs(actual[mask])
    return {
        "mape_pct": (
            float(np.mean(relative) * 100) if len(relative) else float("nan")
        ),
        "p90_ape_pct": (
            float(np.quantile(relative, 0.9) * 100)
            if len(relative) else float("nan")
        ),
        "mape_n": int(mask.sum()),
        "mape_excluded_zero_count": int(len(actual) - mask.sum()),
        "mape_zero_abs_tolerance": float(tolerance),
    }


def cv_metrics(X, y, kind, n_splits=5, seed=42, return_yhat=False):
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
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1e-12
    metrics = {
        "r2": 1 - ss_res / ss_tot,
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        **relative_error_summary(y, err),
    }
    if return_yhat:
        return metrics, yhat
    return metrics


def _evenly_spaced_positions(length, limit=PARITY_MAX_PAIRS_PER_TARGET):
    """Return deterministic positions, retaining both endpoints when sampled."""
    length = int(length)
    limit = int(limit)
    if length <= 0 or limit <= 0:
        return []
    if length <= limit:
        return list(range(length))
    if limit == 1:
        return [0]
    return [
        index * (length - 1) // (limit - 1)
        for index in range(limit)
    ]


def _json_index(value):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _parity_target(y, yhat, row_index, limit=PARITY_MAX_PAIRS_PER_TARGET):
    actual = np.asarray(y, dtype=float).reshape(-1)
    predicted = np.asarray(yhat, dtype=float).reshape(-1)
    indexes = list(row_index)
    if len(actual) != len(predicted) or len(actual) != len(indexes):
        raise ValueError("parity actual, predicted, and row-index lengths differ")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("parity values must all be finite")
    positions = _evenly_spaced_positions(len(actual), limit=limit)
    method = "all" if len(actual) <= int(limit) else "evenly_spaced_position"
    return {
        "n": len(actual),
        "sample_count": len(positions),
        "sampling": {
            "method": method,
            "limit": int(limit),
        },
        "pairs": [
            {
                "row_position": int(position),
                "row_index": _json_index(indexes[position]),
                "actual": float(actual[position]),
                "predicted": float(predicted[position]),
            }
            for position in positions
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--curve-csv", default=CURVE_CSV)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--result-json", default=None)
    ap.add_argument("--parity-json", default=None)
    ap.add_argument("--skip-curve-append", action="store_true")
    ap.add_argument("--checkpoint", type=int, default=None)
    args = ap.parse_args()

    from quality_contract import DEFAULT_PROFILE_PATH, annotate_validity, load_profile

    args.dataset = os.path.abspath(args.dataset)
    args.curve_csv = os.path.abspath(args.curve_csv)
    args.profile = os.path.abspath(args.profile or DEFAULT_PROFILE_PATH)
    args.result_json = (
        os.path.abspath(args.result_json) if args.result_json else None
    )
    args.parity_json = (
        os.path.abspath(args.parity_json) if args.parity_json else None
    )
    if (args.result_json or args.parity_json) and args.checkpoint is None:
        ap.error("--checkpoint is required with --result-json or --parity-json")
    profile_data = load_profile(args.profile)
    profile_sha256 = hashlib.sha256(
        json.dumps(
            profile_data, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()

    raw = pd.read_parquet(args.dataset)
    df = annotate_validity(raw, args.profile)
    df = to_physical(df)
    feats = feature_columns(df)
    n_total = int(df["_strict_valid_full"].sum())
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"dataset: raw={len(df)} strict_full={n_total}, {len(feats)} features")
    if not feats:
        raise SystemExit("no design-time training features remain after strict filtering")

    rows = []
    parity_targets = {}
    target_revision_cohorts = {}
    for target, cfg in TARGETS.items():
        if target not in df.columns:
            print(f"  [skip] {target} (컬럼 없음)")
            continue
        # 온도 타겟: thermal 솔브가 성공한 행만 (thermal_solved 플래그, 2026-07-09)
        target_df = filter_valid_training_rows(df, target, args.profile)
        revision_cohort = target_df.attrs.get(
            "physics_data_revision_cohort", ""
        )
        sub = target_df.dropna(subset=[target])
        sub = sub[np.isfinite(sub[target])]
        if len(sub) < 100:
            print(f"  [skip] {target} (n={len(sub)} < 100)")
            continue
        X = sub[feats].fillna(0.0)
        y = sub[target].to_numpy(dtype=float)

        if args.parity_json:
            m, yhat = cv_metrics(
                X, y, cfg["transform"], return_yhat=True
            )
            parity_targets[target] = _parity_target(y, yhat, sub.index)
            parity_targets[target][
                "physics_data_revision_cohort"
            ] = revision_cohort
        else:
            m = cv_metrics(X, y, cfg["transform"])
        target_revision_cohorts[target] = revision_cohort
        row = {
            "time": stamp, "target": target, "n": len(sub), **m,
            "slice": "global",
            "physics_data_revision_cohort": revision_cohort,
        }
        rows.append(row)
        print(f"  {target:32s} n={len(sub):6d}  R2={m['r2']:.4f}  MAPE={m['mape_pct']:.2f}%  "
              f"P90APE={m['p90_ape_pct']:.2f}%  RMSE={m['rmse']:.4g}")

        # 관심 슬라이스: Llt_phys 20~40uH 영역
        if "Llt_phys" in sub.columns:
            sl = sub[(sub["Llt_phys"] >= 20) & (sub["Llt_phys"] <= 40)]
            if len(sl) >= 100:
                ms = cv_metrics(sl[feats].fillna(0.0), sl[target].to_numpy(dtype=float), cfg["transform"])
                rows.append({
                    "time": stamp, "target": target, "n": len(sl), **ms,
                    "slice": "Llt20-40",
                    "physics_data_revision_cohort": revision_cohort,
                })
                print(f"    └ slice Llt 20-40uH: n={len(sl)}  MAPE={ms['mape_pct']:.2f}%  P90={ms['p90_ape_pct']:.2f}%")

    if rows:
        if not args.skip_curve_append:
            curve = pd.DataFrame(rows)
            os.makedirs(os.path.dirname(os.path.abspath(args.curve_csv)), exist_ok=True)
            with FileLock(args.curve_csv + ".lock", timeout=1):
                header = not os.path.isfile(args.curve_csv)
                curve.to_csv(args.curve_csv, mode="a", header=header, index=False)
            print(f"learning curve appended -> {args.curve_csv}")
        else:
            print("learning curve append skipped")
        completed_at = datetime.now().isoformat(timespec="seconds")
        dataset_sha256 = (
            _sha256(args.dataset)
            if args.result_json or args.parity_json else None
        )
        if args.result_json:
            _atomic_json({
                "schema_version": 1,
                "completed_at": completed_at,
                "checkpoint": args.checkpoint,
                "dataset": args.dataset,
                "dataset_sha256": dataset_sha256,
                "profile": args.profile,
                "profile_sha256": profile_sha256,
                "strict_full_rows": n_total,
                "features": list(feats),
                "target_physics_data_revision_cohorts": target_revision_cohorts,
                "metrics": rows,
            }, args.result_json)
        if args.parity_json:
            _atomic_json({
                "schema_version": PARITY_SCHEMA_VERSION,
                "artifact_type": "checkpoint_cv_oof_parity",
                "completed_at": completed_at,
                "checkpoint": args.checkpoint,
                "dataset": args.dataset,
                "dataset_sha256": dataset_sha256,
                "profile": args.profile,
                "profile_sha256": profile_sha256,
                "strict_full_rows": n_total,
                "features": list(feats),
                "prediction_kind": "out_of_fold",
                "cv": {"n_splits": 5, "shuffle": True, "seed": 42},
                "max_pairs_per_target": PARITY_MAX_PAIRS_PER_TARGET,
                "target_physics_data_revision_cohorts": target_revision_cohorts,
                "targets": parity_targets,
            }, args.parity_json)
    else:
        raise SystemExit("no target has enough strict-full rows for checkpoint metrics")


if __name__ == "__main__":
    main()
