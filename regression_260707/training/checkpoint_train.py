"""
체크포인트 회귀 학습 + 학습곡선 추적 (데이터 수집 중 상시 모니터링용).

- dataset/train.parquet 로드 -> 실물(_phys) 변환 -> 타겟별 LightGBM 5-fold CV
- 지표(R2/MAPE/RMSE)를 learning_curve.csv에 데이터 개수와 함께 축적
- 관심 슬라이스(Llt_phys 20~40uH) 지표 별도 기록
- 사용: python checkpoint_train.py [--full]  (--full 이면 4패밀리 앙상블까지)
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from model_targets import (  # noqa: E402 - direct-script path is installed above
    SURROGATE_CAPACITANCE_TARGETS,
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
CAPACITANCE_RELATIVE_METRIC_TOLERANCE = 0.0
DEFAULT_MODEL_THREADS = 1
DEFAULT_TARGET_WORKERS = 1


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
        # The checkpoint run root is deliberately identity-scoped and deep.
        # Repeating a parity sidecar's full basename in the temporary name can
        # push an otherwise valid final path past the legacy Windows MAX_PATH
        # boundary.  mkstemp already supplies collision-safe randomness; a
        # constant short prefix preserves same-directory atomic replacement.
        prefix=".tmp-", suffix=".tmp",
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
    **{
        target: {
            "transform": "log",
            "metric_focus": "mape",
            # Valid capacitances are strictly positive, including values below
            # the legacy near-zero tolerance used by loss-like targets.
            "relative_metric_tolerance": CAPACITANCE_RELATIVE_METRIC_TOLERANCE,
        }
        for target in SURROGATE_CAPACITANCE_TARGETS
    },
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
    """Return strict-full rows satisfying the target-specific contract.

    Strict-full validity is the common first tier.  The second tier removes
    rows whose requested target is unavailable or invalid.  This lets newer
    outputs, such as electrostatic capacitance, train only on eligible cohorts
    without admitting legacy EM false positives into any target.
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
        if target in SURROGATE_CAPACITANCE_TARGETS:
            keep &= values.gt(0)
            if "cap_on" in df.columns:
                cap_enabled = pd.to_numeric(
                    df["cap_on"], errors="coerce"
                ).eq(1)
                keep &= cap_enabled
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
        return np.log(np.clip(y, np.finfo(float).tiny, None))
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


def cv_metrics(
    X,
    y,
    kind,
    n_splits=5,
    seed=42,
    return_yhat=False,
    relative_tolerance=MAPE_ZERO_ABS_TOLERANCE,
    model_threads=DEFAULT_MODEL_THREADS,
):
    import lightgbm as lgb
    from sklearn.model_selection import KFold

    if isinstance(model_threads, bool):
        raise ValueError("model_threads must be a positive integer")
    try:
        model_threads = int(model_threads)
    except (TypeError, ValueError) as exc:
        raise ValueError("model_threads must be a positive integer") from exc
    if model_threads < 1:
        raise ValueError("model_threads must be a positive integer")

    yt = transform_y(y, kind)
    preds = np.full(len(y), np.nan)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        model = lgb.LGBMRegressor(
            n_estimators=800, learning_rate=0.05, num_leaves=63,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=seed, verbose=-1, n_jobs=model_threads)
        model.fit(X.iloc[tr], yt[tr])
        preds[te] = model.predict(X.iloc[te])
    yhat = inverse_y(preds, kind)
    err = yhat - y
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1e-12
    metrics = {
        "r2": 1 - ss_res / ss_tot,
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        **relative_error_summary(y, err, tolerance=relative_tolerance),
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


def _checkpoint_parallelism(
    model_threads,
    target_workers,
    max_model_thread_budget=None,
):
    """Return one fail-closed target/model thread budget.

    Each concurrent target can request ``model_threads`` LightGBM workers.
    Production supplies an independently declared total ceiling. Standalone
    callers that omit it still get a finite ceiling equal to the requested
    product, never the model library's unbounded default.
    """
    values = {
        "model_threads": model_threads,
        "target_workers": target_workers,
    }
    normalized = {}
    for label, value in values.items():
        if isinstance(value, bool):
            raise ValueError(f"{label} must be a positive integer")
        try:
            normalized[label] = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be a positive integer") from exc
        if normalized[label] < 1:
            raise ValueError(f"{label} must be a positive integer")

    requested = normalized["model_threads"] * normalized["target_workers"]
    if max_model_thread_budget is None:
        budget = requested
        externally_declared = False
    else:
        if isinstance(max_model_thread_budget, bool):
            raise ValueError(
                "max_model_thread_budget must be a positive integer"
            )
        try:
            budget = int(max_model_thread_budget)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "max_model_thread_budget must be a positive integer"
            ) from exc
        if budget < 1:
            raise ValueError(
                "max_model_thread_budget must be a positive integer"
            )
        externally_declared = True
    if requested > budget:
        raise ValueError(
            "target_workers * model_threads exceeds max_model_thread_budget"
        )
    return {
        **normalized,
        "max_model_thread_budget": budget,
        "maximum_total_model_threads": requested,
        "budget_externally_declared": externally_declared,
    }


def _evaluate_target(
    df,
    feats,
    profile,
    target,
    cfg,
    *,
    include_parity,
    model_threads,
    stamp,
):
    """Evaluate one target without mutating shared output state."""
    if target not in df.columns:
        return {
            "target": target,
            "rows": [],
            "parity": None,
            "revision_cohort": None,
            "messages": [f"  [skip] {target} (column missing)"],
        }

    target_df = filter_valid_training_rows(df, target, profile)
    revision_cohort = target_df.attrs.get(
        "physics_data_revision_cohort", ""
    )
    sub = target_df.dropna(subset=[target])
    sub = sub[np.isfinite(sub[target])]
    if len(sub) < 100:
        return {
            "target": target,
            "rows": [],
            "parity": None,
            "revision_cohort": None,
            "messages": [f"  [skip] {target} (n={len(sub)} < 100)"],
        }

    X = sub[feats].fillna(0.0)
    y = sub[target].to_numpy(dtype=float)
    relative_tolerance = cfg.get(
        "relative_metric_tolerance", MAPE_ZERO_ABS_TOLERANCE
    )
    parity = None
    if include_parity:
        metrics, yhat = cv_metrics(
            X,
            y,
            cfg["transform"],
            return_yhat=True,
            relative_tolerance=relative_tolerance,
            model_threads=model_threads,
        )
        parity = _parity_target(y, yhat, sub.index)
        parity["physics_data_revision_cohort"] = revision_cohort
    else:
        metrics = cv_metrics(
            X,
            y,
            cfg["transform"],
            relative_tolerance=relative_tolerance,
            model_threads=model_threads,
        )

    rows = [{
        "time": stamp,
        "target": target,
        "n": len(sub),
        **metrics,
        "slice": "global",
        "physics_data_revision_cohort": revision_cohort,
    }]
    messages = [
        f"  {target:32s} n={len(sub):6d}  R2={metrics['r2']:.4f}  "
        f"MAPE={metrics['mape_pct']:.2f}%  "
        f"P90APE={metrics['p90_ape_pct']:.2f}%  "
        f"RMSE={metrics['rmse']:.4g}"
    ]

    if "Llt_phys" in sub.columns:
        sliced = sub[(sub["Llt_phys"] >= 20) & (sub["Llt_phys"] <= 40)]
        if len(sliced) >= 100:
            slice_metrics = cv_metrics(
                sliced[feats].fillna(0.0),
                sliced[target].to_numpy(dtype=float),
                cfg["transform"],
                relative_tolerance=relative_tolerance,
                model_threads=model_threads,
            )
            rows.append({
                "time": stamp,
                "target": target,
                "n": len(sliced),
                **slice_metrics,
                "slice": "Llt20-40",
                "physics_data_revision_cohort": revision_cohort,
            })
            messages.append(
                f"    slice Llt 20-40uH: n={len(sliced)}  "
                f"MAPE={slice_metrics['mape_pct']:.2f}%  "
                f"P90={slice_metrics['p90_ape_pct']:.2f}%"
            )
    return {
        "target": target,
        "rows": rows,
        "parity": parity,
        "revision_cohort": revision_cohort,
        "messages": messages,
    }


def _evaluate_targets(
    df,
    feats,
    profile,
    targets,
    *,
    include_parity,
    model_threads,
    target_workers,
    max_model_thread_budget,
    stamp,
):
    """Run targets concurrently while returning declaration order."""
    parallelism = _checkpoint_parallelism(
        model_threads,
        target_workers,
        max_model_thread_budget,
    )
    items = list(targets.items())
    if not items:
        parallelism["effective_target_workers"] = 0
        parallelism["effective_total_model_threads"] = 0
        return [], parallelism
    effective_workers = min(parallelism["target_workers"], len(items))
    parallelism["effective_target_workers"] = effective_workers
    parallelism["effective_total_model_threads"] = (
        effective_workers * parallelism["model_threads"]
    )

    def evaluate(item):
        target, cfg = item
        return _evaluate_target(
            df,
            feats,
            profile,
            target,
            cfg,
            include_parity=include_parity,
            model_threads=parallelism["model_threads"],
            stamp=stamp,
        )

    if effective_workers == 1:
        return [evaluate(item) for item in items], parallelism

    ordered = [None] * len(items)
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {
            executor.submit(evaluate, item): index
            for index, item in enumerate(items)
        }
        for future in as_completed(futures):
            ordered[futures[future]] = future.result()
    if any(result is None for result in ordered):
        raise RuntimeError("parallel checkpoint target inventory is incomplete")
    return ordered, parallelism


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--curve-csv", default=CURVE_CSV)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--result-json", default=None)
    ap.add_argument("--parity-json", default=None)
    ap.add_argument("--skip-curve-append", action="store_true")
    ap.add_argument("--checkpoint", type=int, default=None)
    ap.add_argument(
        "--model-threads",
        type=int,
        default=DEFAULT_MODEL_THREADS,
        help="threads used by each LightGBM fit (default: 1)",
    )
    ap.add_argument(
        "--target-workers",
        type=int,
        default=DEFAULT_TARGET_WORKERS,
        help="independent checkpoint targets evaluated concurrently (default: 1)",
    )
    ap.add_argument(
        "--max-model-thread-budget",
        type=int,
        default=None,
        help="fail-closed ceiling for target-workers times model-threads",
    )
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
    try:
        requested_parallelism = _checkpoint_parallelism(
            args.model_threads,
            args.target_workers,
            args.max_model_thread_budget,
        )
    except ValueError as exc:
        ap.error(str(exc))
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
    evaluations, training_parallelism = _evaluate_targets(
        df,
        feats,
        args.profile,
        TARGETS,
        include_parity=bool(args.parity_json),
        model_threads=requested_parallelism["model_threads"],
        target_workers=requested_parallelism["target_workers"],
        max_model_thread_budget=args.max_model_thread_budget,
        stamp=stamp,
    )
    for evaluation in evaluations:
        for message in evaluation["messages"]:
            print(message)
        rows.extend(evaluation["rows"])
        if evaluation["revision_cohort"] is None:
            continue
        target = evaluation["target"]
        target_revision_cohorts[target] = evaluation["revision_cohort"]
        if evaluation["parity"] is not None:
            parity_targets[target] = evaluation["parity"]

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
                "training_parallelism": training_parallelism,
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
                "training_parallelism": training_parallelism,
                "max_pairs_per_target": PARITY_MAX_PAIRS_PER_TARGET,
                "target_physics_data_revision_cohorts": target_revision_cohorts,
                "targets": parity_targets,
            }, args.parity_json)
    else:
        raise SystemExit("no target has enough strict-full rows for checkpoint metrics")


if __name__ == "__main__":
    main()
