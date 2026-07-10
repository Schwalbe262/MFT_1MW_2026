"""Train one coherent surrogate generation with independently tested UQ.

Every required target is trained from the same recomputed strict-full cohort.
Models are first written to a generation directory and become visible through
one atomic ``current.json`` pointer, so a failed/partial run cannot mix stale
and new targets.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
import pickle
import shutil
import tempfile
import uuid

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "..", "data", "dataset", "train.parquet")
REGISTRY = os.path.join(HERE, "registry")

from checkpoint_train import (  # noqa: E402
    TARGETS,
    feature_columns,
    filter_valid_training_rows,
    inverse_y,
    to_physical,
    transform_y,
)

N_FOLDS = 5
CALIBRATION_FRAC = 0.10
EVALUATION_FRAC = 0.10
SEED = 42


def default_family_params():
    return {
        "lightgbm": dict(
            n_estimators=1200, learning_rate=0.04, num_leaves=63,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, verbose=-1,
        ),
        "xgboost": dict(
            n_estimators=1200, learning_rate=0.04, max_depth=8,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, verbosity=0,
        ),
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
        return CatBoostRegressor(
            random_seed=seed, allow_writing_files=False, **params
        )
    if family == "extratrees":
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(random_state=seed, **params)
    raise ValueError(family)


def _ensemble_prediction(models, Xpart, transform):
    predictions = np.stack([model.predict(Xpart) for _, model in models])
    mu_t = np.median(predictions, axis=0)
    sigma_t = predictions.std(axis=0)
    mu = inverse_y(mu_t, transform)
    derivative = np.abs(
        inverse_y(mu_t + 1e-4, transform) - inverse_y(mu_t - 1e-4, transform)
    ) / 2e-4
    return mu, np.maximum(derivative * sigma_t, 1e-9)


def train_target(
    df,
    feats,
    target,
    cfg,
    family_params,
    sample_weight_col=None,
    min_rows=200,
):
    """Train a target and evaluate it on data not used for calibration."""
    from sklearn.model_selection import KFold, train_test_split

    sub = filter_valid_training_rows(df, target)
    sub = sub.dropna(subset=[target])
    sub = sub[np.isfinite(pd.to_numeric(sub[target], errors="coerce"))]
    if len(sub) < min_rows:
        return None, f"insufficient strict-full rows ({len(sub)})"

    X = sub[feats].fillna(0.0).reset_index(drop=True)
    y_raw = sub[target].to_numpy(dtype=float)
    weights = (
        sub[sample_weight_col].fillna(1.0).to_numpy(dtype=float)
        if sample_weight_col and sample_weight_col in sub.columns
        else np.ones(len(sub))
    )
    transform = cfg["transform"]
    y = transform_y(y_raw, transform)

    indices = np.arange(len(X))
    idx_train, idx_holdout = train_test_split(
        indices,
        test_size=CALIBRATION_FRAC + EVALUATION_FRAC,
        random_state=SEED,
    )
    idx_calibration, idx_evaluation = train_test_split(
        idx_holdout,
        test_size=EVALUATION_FRAC / (CALIBRATION_FRAC + EVALUATION_FRAC),
        random_state=SEED + 1,
    )
    X_train, y_train = X.iloc[idx_train], y[idx_train]
    weight_train = weights[idx_train]

    models = []
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for family, params in family_params.items():
        for fold_index, (fit_indices, _) in enumerate(folds.split(X_train)):
            model = make_model(family, params, seed=SEED + fold_index)
            try:
                model.fit(
                    X_train.iloc[fit_indices],
                    y_train[fit_indices],
                    sample_weight=weight_train[fit_indices],
                )
            except TypeError:
                model.fit(X_train.iloc[fit_indices], y_train[fit_indices])
            models.append((family, model))

    mu_cal, sigma_cal = _ensemble_prediction(
        models, X.iloc[idx_calibration], transform
    )
    calibration_scores = (
        np.abs(mu_cal - y_raw[idx_calibration]) / sigma_cal
    )
    quantile_level = min(
        1.0,
        np.ceil((len(calibration_scores) + 1) * 0.90) / len(calibration_scores),
    )
    try:
        q90 = float(
            np.quantile(calibration_scores, quantile_level, method="higher")
        )
    except TypeError:  # NumPy < 1.22
        q90 = float(
            np.quantile(calibration_scores, quantile_level, interpolation="higher")
        )

    y_evaluation = y_raw[idx_evaluation]
    mu, sigma = _ensemble_prediction(models, X.iloc[idx_evaluation], transform)
    half_width = q90 * sigma
    error = mu - y_evaluation
    relative_error = np.abs(error) / np.clip(np.abs(y_evaluation), 1e-9, None)
    relative_half_width = half_width / np.clip(
        np.abs(y_evaluation), 1e-9, None
    )
    target_scale = max(
        float(np.quantile(y_evaluation, 0.9) - np.quantile(y_evaluation, 0.1)),
        float(np.median(np.abs(y_evaluation))),
        1e-9,
    )
    metrics = {
        "n_train": int(len(idx_train)),
        "n_calibration": int(len(idx_calibration)),
        "n_evaluation": int(len(idx_evaluation)),
        # Compatibility name used by the existing monitor.
        "n_holdout": int(len(idx_evaluation)),
        "r2": float(
            1
            - np.sum(error ** 2)
            / (np.sum((y_evaluation - y_evaluation.mean()) ** 2) or 1e-12)
        ),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "normalized_rmse_pct": float(
            np.sqrt(np.mean(error ** 2)) / target_scale * 100
        ),
        "mape_pct": float(np.mean(relative_error) * 100),
        "p90_ape_pct": float(np.quantile(relative_error, 0.9) * 100),
        "q90_conformal": q90,
        "interval_coverage": float(np.mean(np.abs(error) <= half_width)),
        "interval_mean_width": float(np.mean(2.0 * half_width)),
        "interval_p90_width": float(np.quantile(2.0 * half_width, 0.9)),
        "interval_p90_half_width_pct": float(
            np.quantile(relative_half_width, 0.9) * 100
        ),
    }
    bundle = {
        "models": models,
        "features": list(feats),
        "transform": transform,
        "q90": q90,
        "target": target,
        "metrics": metrics,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    return bundle, metrics


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


def _install_compatibility_view(registry, generation_dir, targets, report):
    """Refresh old registry paths for the already-running monitoring UI."""
    for target in targets:
        target_dir = os.path.join(registry, target)
        os.makedirs(target_dir, exist_ok=True)
        for filename in ("models.pkl", "meta.json"):
            source = os.path.join(generation_dir, target, filename)
            destination = os.path.join(target_dir, filename)
            fd, staged = tempfile.mkstemp(
                prefix=f".{filename}.", suffix=".tmp", dir=target_dir
            )
            os.close(fd)
            try:
                shutil.copy2(source, staged)
                os.replace(staged, destination)
            finally:
                if os.path.exists(staged):
                    os.remove(staged)
    _atomic_json(report, os.path.join(registry, "train_report.json"))


def capture_active_generation(registry):
    """Capture enough metadata to roll back an activation after gate failure."""
    pointer_path = os.path.join(registry, "current.json")
    if not os.path.isfile(pointer_path):
        return None
    with open(pointer_path, encoding="utf-8") as handle:
        pointer = json.load(handle)
    generation = os.path.abspath(os.path.join(registry, pointer["generation"]))
    with open(
        os.path.join(generation, "train_report.json"), encoding="utf-8"
    ) as handle:
        report = json.load(handle)
    return {"pointer": pointer, "generation": generation, "report": report}


def restore_active_generation(registry, captured):
    """Restore the prior atomic pointer and compatibility view."""
    pointer_path = os.path.join(registry, "current.json")
    if captured is None:
        if os.path.isfile(pointer_path):
            os.remove(pointer_path)
        return
    _atomic_json(captured["pointer"], pointer_path)
    configured_targets = captured["report"].get("targets")
    targets = configured_targets if isinstance(configured_targets, list) else list(TARGETS)
    _install_compatibility_view(
        registry, captured["generation"], targets, captured["report"]
    )


def discard_inactive_generation(registry, generation):
    """Delete only a rejected, inactive generation inside registry/generations."""
    if not generation:
        return
    registry = os.path.abspath(registry)
    generations_root = os.path.abspath(os.path.join(registry, "generations"))
    target = os.path.abspath(generation)
    if os.path.commonpath([target, generations_root]) != generations_root:
        raise RuntimeError("refusing to discard a generation outside registry")
    pointer_path = os.path.join(registry, "current.json")
    if os.path.isfile(pointer_path):
        with open(pointer_path, encoding="utf-8") as handle:
            active = os.path.abspath(
                os.path.join(registry, json.load(handle)["generation"])
            )
        if active == target:
            raise RuntimeError("refusing to discard the active generation")
    if os.path.isdir(target):
        shutil.rmtree(target)


def prune_inactive_generations(registry, keep=3, protected_run_ids=()):
    """Bound successful registry history without deleting active/AL-pinned runs."""
    registry = os.path.abspath(registry)
    generations_root = os.path.join(registry, "generations")
    if not os.path.isdir(generations_root):
        return []
    active_run_id = None
    pointer_path = os.path.join(registry, "current.json")
    if os.path.isfile(pointer_path):
        with open(pointer_path, encoding="utf-8") as handle:
            active_run_id = json.load(handle).get("training_run_id")
    protected = set(protected_run_ids)
    if active_run_id:
        protected.add(active_run_id)
    inventory = []
    for name in os.listdir(generations_root):
        path = os.path.join(generations_root, name)
        report_path = os.path.join(path, "train_report.json")
        if name.startswith(".") or not os.path.isdir(path) or not os.path.isfile(report_path):
            continue
        try:
            with open(report_path, encoding="utf-8") as handle:
                run_id = json.load(handle).get("training_run_id")
        except Exception:
            continue
        inventory.append((os.path.getmtime(path), path, run_id))
    inventory.sort(reverse=True)
    keep_paths = {path for _, path, _ in inventory[: max(0, keep)]}
    removed = []
    for _, path, run_id in inventory:
        if path in keep_paths or run_id in protected:
            continue
        discard_inactive_generation(registry, path)
        removed.append(path)
    return removed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--params", default=None)
    parser.add_argument("--weight-col", default="sample_weight")
    parser.add_argument("--min-rows", type=int, default=200)
    parser.add_argument("--registry", default=REGISTRY)
    parser.add_argument("--profile", default=None)
    args = parser.parse_args()

    from quality_contract import annotate_validity, load_profile

    raw = pd.read_parquet(args.dataset)
    frame = to_physical(annotate_validity(raw, args.profile))
    features = feature_columns(frame)
    if not features:
        raise SystemExit("no design-time features remain after strict filtering")
    strict_count = int(frame["_strict_valid_full"].sum())

    base_params = default_family_params()
    tuned = {}
    if args.params and os.path.isfile(args.params):
        with open(args.params, encoding="utf-8") as handle:
            tuned = json.load(handle)

    def family_params_for(target):
        output = {}
        for family, base in base_params.items():
            params = dict(base)
            specification = tuned.get(family, {})
            if (
                target in specification
                and isinstance(specification[target], dict)
                and "params" in specification[target]
            ):
                params.update(specification[target]["params"])
            elif specification and all(
                not isinstance(value, dict) for value in specification.values()
            ):
                params.update(specification)
            output[family] = params
        return output

    targets = args.targets or list(TARGETS)
    omitted = [target for target in TARGETS if target not in targets]
    if omitted:
        raise SystemExit(
            f"partial registry generations are forbidden; omitted targets: {omitted}"
        )

    registry = os.path.abspath(args.registry)
    generations = os.path.join(registry, "generations")
    os.makedirs(generations, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    staging = os.path.join(generations, f".{run_id}.tmp")
    generation_dir = os.path.join(generations, run_id)
    os.makedirs(staging, exist_ok=False)

    dataset_sha256 = _sha256(args.dataset)
    profile_data = load_profile(args.profile)
    profile_sha256 = hashlib.sha256(
        json.dumps(
            profile_data, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    target_reports = {}
    try:
        for target in targets:
            if target not in frame.columns:
                raise RuntimeError(f"required target column is missing: {target}")
            bundle, metrics = train_target(
                frame,
                features,
                target,
                TARGETS[target],
                family_params_for(target),
                args.weight_col,
                min_rows=args.min_rows,
            )
            if bundle is None:
                raise RuntimeError(
                    f"required target {target} cannot be trained: {metrics}"
                )
            bundle.update(
                {
                    "training_run_id": run_id,
                    "dataset_sha256": dataset_sha256,
                    "strict_full_rows": strict_count,
                    "profile_sha256": profile_sha256,
                    "feature_schema": list(features),
                }
            )
            target_dir = os.path.join(staging, target)
            os.makedirs(target_dir, exist_ok=True)
            with open(os.path.join(target_dir, "models.pkl"), "wb") as handle:
                pickle.dump(bundle, handle)
            with open(
                os.path.join(target_dir, "meta.json"), "w", encoding="utf-8"
            ) as handle:
                json.dump(
                    {key: value for key, value in bundle.items() if key != "models"},
                    handle,
                    indent=1,
                    default=str,
                )
            target_reports[target] = metrics
            print(
                f"{target:32s} R2={metrics['r2']:.4f} "
                f"MAPE={metrics['mape_pct']:.2f}% "
                f"P90={metrics['p90_ape_pct']:.2f}% "
                f"coverage={metrics['interval_coverage']:.3f}"
            )

        report = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "training_run_id": run_id,
            "dataset_path": os.path.abspath(args.dataset),
            "dataset_sha256": dataset_sha256,
            "raw_rows": int(len(frame)),
            "strict_full_rows": strict_count,
            "profile_sha256": profile_sha256,
            "features": list(features),
            "targets": list(targets),
            "report": target_reports,
        }
        _atomic_json(report, os.path.join(staging, "train_report.json"))
        os.replace(staging, generation_dir)
        pointer = {
            "schema_version": 1,
            "training_run_id": run_id,
            "generation": os.path.relpath(
                generation_dir, registry
            ).replace("\\", "/"),
            "dataset_sha256": dataset_sha256,
            "strict_full_rows": strict_count,
            "activated_at": datetime.now().isoformat(timespec="seconds"),
        }
        _atomic_json(pointer, os.path.join(registry, "current.json"))
        _install_compatibility_view(
            registry, generation_dir, targets, report
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
