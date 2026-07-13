"""Build coherent surrogate candidates without exposing ungated models.

Every required target is trained from the same recomputed strict-full cohort.
This command only builds an immutable candidate generation.  The caller must
evaluate that generation and use :func:`promote_generation` to publish it.
Registry mutation APIs own the writer lock themselves; callers must never wrap
them in the same lock.
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

from filelock import FileLock
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "..", "data", "dataset", "train.parquet")
REGISTRY = os.path.join(HERE, "registry")

from checkpoint_train import (  # noqa: E402
    MAPE_ZERO_ABS_TOLERANCE,
    TARGETS,
    feature_columns,
    filter_valid_training_rows,
    inverse_y,
    relative_error_summary,
    relative_metric_mask,
    to_physical,
    transform_y,
)

N_FOLDS = 5
CALIBRATION_FRAC = 0.10
EVALUATION_FRAC = 0.10
SEED = 42
REGISTRY_SCHEMA_VERSION = 2
QUALITY_GATE_FILENAME = "quality_gate.json"


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


def _evaluation_relative_metrics(y_true, error, half_width):
    """Compute evaluation relative metrics on one auditable nonzero mask."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    residual = np.asarray(error, dtype=float).reshape(-1)
    widths = np.asarray(half_width, dtype=float).reshape(-1)
    if len(actual) != len(residual) or len(actual) != len(widths):
        raise ValueError("evaluation relative metric lengths differ")
    mask = relative_metric_mask(actual)
    summary = relative_error_summary(actual, residual)
    relative_half_width = np.abs(widths[mask]) / np.abs(actual[mask])
    summary["interval_p90_half_width_pct"] = (
        float(np.quantile(relative_half_width, 0.9) * 100)
        if len(relative_half_width) else float("nan")
    )
    if summary["mape_zero_abs_tolerance"] != MAPE_ZERO_ABS_TOLERANCE:
        raise RuntimeError("relative metric tolerance contract drifted")
    return summary


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
    if "physics_data_revision_cohort" not in sub.attrs:
        raise RuntimeError(
            f"target {target} training rows are missing "
            "physics_data_revision_cohort metadata"
        )
    revision_cohort = sub.attrs["physics_data_revision_cohort"]
    sub = sub.dropna(subset=[target])
    sub = sub[np.isfinite(pd.to_numeric(sub[target], errors="coerce"))]
    if len(sub) < min_rows:
        return None, f"insufficient strict-full rows ({len(sub)})"
    if not isinstance(revision_cohort, str) or not revision_cohort.strip():
        raise RuntimeError(
            f"target {target} training rows are missing "
            "physics_data_revision_cohort metadata"
        )

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
    relative_metrics = _evaluation_relative_metrics(
        y_evaluation, error, half_width
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
        **relative_metrics,
        "q90_conformal": q90,
        "interval_coverage": float(np.mean(np.abs(error) <= half_width)),
        "interval_mean_width": float(np.mean(2.0 * half_width)),
        "interval_p90_width": float(np.quantile(2.0 * half_width, 0.9)),
        "interval_p90_half_width_pct": float(
            np.quantile(relative_half_width, 0.9) * 100
        ),
        "physics_data_revision_cohort": revision_cohort,
    }
    bundle = {
        "models": models,
        "features": list(feats),
        "transform": transform,
        "q90": q90,
        "target": target,
        "physics_data_revision_cohort": revision_cohort,
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


def _registry_lock(registry, timeout=1):
    """Return the single registry writer lock used by every mutation API."""
    return FileLock(os.path.abspath(registry) + ".training.lock", timeout=timeout)


def registry_pointer_token(registry):
    """Capture the exact pointer state for compare-before-promote semantics."""
    pointer_path = os.path.join(os.path.abspath(registry), "current.json")
    if not os.path.isfile(pointer_path):
        return {"exists": False, "sha256": None}
    return {"exists": True, "sha256": _sha256(pointer_path)}


def _resolve_generation(registry, generation):
    registry = os.path.abspath(registry)
    generations_root = os.path.abspath(os.path.join(registry, "generations"))
    value = os.fspath(generation)
    target = os.path.abspath(
        value if os.path.isabs(value) else os.path.join(registry, value)
    )
    if (
        target == generations_root
        or os.path.commonpath([target, generations_root]) != generations_root
    ):
        raise RuntimeError("registry generation escapes generations root")
    return target


def load_generation(registry, generation, require_accepted=False):
    """Load and cross-check one immutable generation and its gate evidence."""
    registry = os.path.abspath(registry)
    generation_dir = _resolve_generation(registry, generation)
    report_path = os.path.join(generation_dir, "train_report.json")
    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)
    run_id = report.get("training_run_id")
    if not isinstance(run_id, str) or not run_id:
        raise RuntimeError("generation report has no training_run_id")
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise RuntimeError("generation report has no artifact manifest")
    for relative_path, expected_sha256 in artifacts.items():
        artifact = os.path.abspath(os.path.join(generation_dir, relative_path))
        if os.path.commonpath([artifact, generation_dir]) != generation_dir:
            raise RuntimeError(f"generation artifact escapes root: {relative_path}")
        if not os.path.isfile(artifact):
            raise RuntimeError(f"generation artifact is missing: {relative_path}")
        if _sha256(artifact) != expected_sha256:
            raise RuntimeError(
                f"generation artifact fingerprint mismatch: {relative_path}"
            )
    report_sha256 = _sha256(report_path)
    relative = os.path.relpath(generation_dir, registry).replace("\\", "/")
    record = {
        "generation": generation_dir,
        "generation_relative": relative,
        "report": report,
        "generation_report_sha256": report_sha256,
    }
    if require_accepted:
        gate_path = os.path.join(generation_dir, QUALITY_GATE_FILENAME)
        with open(gate_path, encoding="utf-8") as handle:
            quality = json.load(handle)
        if quality.get("passed") is not True:
            raise RuntimeError("generation has no passing quality gate")
        if quality.get("training_run_id") != run_id:
            raise RuntimeError("quality gate training run mismatch")
        if quality.get("dataset_sha256") != report.get("dataset_sha256"):
            raise RuntimeError("quality gate dataset mismatch")
        if quality.get("profile_sha256") != report.get("profile_sha256"):
            raise RuntimeError("quality gate profile mismatch")
        if quality.get("generation_report_sha256") != report_sha256:
            raise RuntimeError("quality gate generation report mismatch")
        if quality.get("generation") != relative:
            raise RuntimeError("quality gate generation path mismatch")
        record.update(
            quality=quality,
            quality_gate_sha256=_sha256(gate_path),
        )
    return record


def load_active_generation(registry):
    """Load the accepted active generation; legacy/ungated pointers fail closed."""
    registry = os.path.abspath(registry)
    pointer_path = os.path.join(registry, "current.json")
    with open(pointer_path, encoding="utf-8") as handle:
        pointer = json.load(handle)
    if pointer.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise RuntimeError("unsupported or ungated registry pointer schema")
    relative = pointer.get("generation")
    if not isinstance(relative, str) or not relative.strip():
        raise RuntimeError("registry pointer has no generation")
    record = load_generation(registry, relative, require_accepted=True)
    report = record["report"]
    checks = {
        "training_run_id": report.get("training_run_id"),
        "dataset_sha256": report.get("dataset_sha256"),
        "profile_sha256": report.get("profile_sha256"),
        "strict_full_rows": report.get("strict_full_rows"),
        "generation_report_sha256": record["generation_report_sha256"],
        "quality_gate_sha256": record["quality_gate_sha256"],
        "thresholds_sha256": record["quality"].get("thresholds_sha256"),
    }
    for key, expected in checks.items():
        if pointer.get(key) != expected:
            raise RuntimeError(f"registry pointer {key} mismatch")
    record["pointer"] = pointer
    return record


def _remove_compatibility_view(registry):
    """Remove unsafe flat artifacts now that all readers follow current.json."""
    registry = os.path.abspath(registry)
    warnings = []
    for target in TARGETS:
        target_dir = os.path.join(registry, target)
        if os.path.isdir(target_dir):
            try:
                shutil.rmtree(target_dir)
            except OSError as exc:
                warnings.append(f"{target_dir}: {exc}")
    for filename in ("train_report.json", "compatibility.json"):
        path = os.path.join(registry, filename)
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError as exc:
                warnings.append(f"{path}: {exc}")
    return warnings


def capture_active_generation(registry):
    """Capture a validated accepted generation for explicit administrative rollback."""
    pointer_path = os.path.join(registry, "current.json")
    if not os.path.isfile(pointer_path):
        return None
    return load_active_generation(registry)


def restore_active_generation(registry, captured, lock_timeout=1):
    """Restore a validated pointer, removing every unsafe legacy flat artifact."""
    registry = os.path.abspath(registry)
    with _registry_lock(registry, timeout=lock_timeout):
        pointer_path = os.path.join(registry, "current.json")
        if captured is None:
            if os.path.isfile(pointer_path):
                os.remove(pointer_path)
            _remove_compatibility_view(registry)
            return
        record = load_generation(
            registry, captured["generation"], require_accepted=True
        )
        if captured.get("pointer", {}).get("training_run_id") != record[
            "report"
        ].get("training_run_id"):
            raise RuntimeError("captured pointer and generation do not match")
        _atomic_json(captured["pointer"], pointer_path)
        _remove_compatibility_view(registry)


def promote_generation(
    registry, generation, quality, dataset, profile_sha256, thresholds_sha256,
    expected_pointer=None, lock_timeout=1,
):
    """Atomically publish a candidate only after a passing, matching gate.

    This function owns the writer lock.  ``expected_pointer`` should be a token
    returned by :func:`registry_pointer_token`; a concurrent promotion then
    fails instead of silently overwriting a newer accepted generation.
    """
    registry = os.path.abspath(registry)
    with _registry_lock(registry, timeout=lock_timeout):
        if expected_pointer is not None:
            actual = registry_pointer_token(registry)
            if actual != expected_pointer:
                raise RuntimeError("registry pointer changed while candidate was evaluated")
        record = load_generation(registry, generation, require_accepted=False)
        report = record["report"]
        if quality.get("passed") is not True:
            raise RuntimeError("refusing to promote a failed quality gate")
        dataset_sha256 = _sha256(dataset)
        if (
            quality.get("dataset_sha256") != dataset_sha256
            or report.get("dataset_sha256") != dataset_sha256
        ):
            raise RuntimeError("quality gate dataset_sha256 mismatch")
        if (
            not isinstance(profile_sha256, str)
            or not profile_sha256
            or quality.get("profile_sha256") != profile_sha256
            or report.get("profile_sha256") != profile_sha256
        ):
            raise RuntimeError("quality gate profile_sha256 mismatch")
        if (
            not isinstance(thresholds_sha256, str)
            or not thresholds_sha256
            or quality.get("thresholds_sha256") != thresholds_sha256
        ):
            raise RuntimeError("quality gate thresholds_sha256 mismatch")
        checks = {
            "training_run_id": report.get("training_run_id"),
            "dataset_sha256": report.get("dataset_sha256"),
            "profile_sha256": report.get("profile_sha256"),
            "generation": record["generation_relative"],
            "generation_report_sha256": record["generation_report_sha256"],
        }
        for key, expected in checks.items():
            if quality.get(key) != expected:
                raise RuntimeError(f"quality gate {key} mismatch")
        accepted_quality = dict(quality)
        accepted_quality.update(
            schema_version=REGISTRY_SCHEMA_VERSION,
            accepted_at=datetime.now().isoformat(timespec="seconds"),
        )
        gate_path = os.path.join(record["generation"], QUALITY_GATE_FILENAME)
        _atomic_json(accepted_quality, gate_path)
        gate_sha256 = _sha256(gate_path)
        pointer = {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "training_run_id": report["training_run_id"],
            "generation": record["generation_relative"],
            "dataset_sha256": report["dataset_sha256"],
            "profile_sha256": report["profile_sha256"],
            "strict_full_rows": report["strict_full_rows"],
            "generation_report_sha256": record["generation_report_sha256"],
            "quality_gate_sha256": gate_sha256,
            "thresholds_sha256": accepted_quality["thresholds_sha256"],
            "activated_at": datetime.now().isoformat(timespec="seconds"),
        }
        # Readers are pointer-only, so compatibility cleanup is best-effort and
        # happens before the atomic pointer commit.  current.json is the final
        # fallible step: if this function raises, the candidate is not active.
        _remove_compatibility_view(registry)
        _atomic_json(pointer, os.path.join(registry, "current.json"))
        return pointer


def _discard_inactive_generation_unlocked(registry, generation):
    """Delete only a rejected, inactive generation inside registry/generations."""
    if not generation:
        return
    registry = os.path.abspath(registry)
    target = _resolve_generation(registry, generation)
    pointer_path = os.path.join(registry, "current.json")
    if os.path.isfile(pointer_path):
        try:
            with open(pointer_path, encoding="utf-8") as handle:
                active = _resolve_generation(
                    registry, json.load(handle)["generation"]
                )
        except Exception as exc:
            raise RuntimeError(
                "refusing to discard while active pointer is unreadable"
            ) from exc
        if active == target:
            raise RuntimeError("refusing to discard the active generation")
    if os.path.isdir(target):
        shutil.rmtree(target)


def discard_inactive_generation(registry, generation, lock_timeout=1):
    """Delete an inactive generation while owning the registry writer lock."""
    if not generation:
        return
    with _registry_lock(registry, timeout=lock_timeout):
        _discard_inactive_generation_unlocked(registry, generation)


def prune_inactive_generations(
    registry, keep=3, protected_run_ids=(), lock_timeout=1
):
    """Bound successful registry history without deleting active/AL-pinned runs."""
    registry = os.path.abspath(registry)
    with _registry_lock(registry, timeout=lock_timeout):
        return _prune_inactive_generations_unlocked(
            registry, keep=keep, protected_run_ids=protected_run_ids
        )


def _prune_inactive_generations_unlocked(registry, keep, protected_run_ids):
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
        _discard_inactive_generation_unlocked(registry, path)
        removed.append(path)
    return removed


def _build_candidate(args, frame, features, strict_count, targets, family_params_for):
    """Write one complete generation.  Caller must hold the registry lock."""
    from quality_contract import load_profile

    registry = args.registry
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
    target_revision_cohorts = {}
    artifact_sha256 = {}
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
            revision_cohort = bundle.get("physics_data_revision_cohort")
            if not isinstance(revision_cohort, str) or not revision_cohort.strip():
                raise RuntimeError(
                    f"required target {target} did not report "
                    "physics_data_revision_cohort metadata"
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
            model_path = os.path.join(target_dir, "models.pkl")
            meta_path = os.path.join(target_dir, "meta.json")
            with open(model_path, "wb") as handle:
                pickle.dump(bundle, handle)
            with open(meta_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {key: value for key, value in bundle.items() if key != "models"},
                    handle,
                    indent=1,
                    default=str,
                )
            artifact_sha256[f"{target}/models.pkl"] = _sha256(model_path)
            artifact_sha256[f"{target}/meta.json"] = _sha256(meta_path)
            target_reports[target] = metrics
            target_revision_cohorts[target] = revision_cohort
            print(
                f"{target:32s} R2={metrics['r2']:.4f} "
                f"MAPE={metrics['mape_pct']:.2f}% "
                f"P90={metrics['p90_ape_pct']:.2f}% "
                f"coverage={metrics['interval_coverage']:.3f}"
            )

        report = {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "time": datetime.now().isoformat(timespec="seconds"),
            "training_run_id": run_id,
            "dataset_path": args.dataset,
            "dataset_sha256": dataset_sha256,
            "raw_rows": int(len(frame)),
            "strict_full_rows": strict_count,
            "profile_path": args.profile,
            "profile_sha256": profile_sha256,
            "features": list(features),
            "targets": list(targets),
            "target_physics_data_revision_cohorts": target_revision_cohorts,
            "artifacts": artifact_sha256,
            "report": target_reports,
        }
        report_path = os.path.join(staging, "train_report.json")
        _atomic_json(report, report_path)
        os.replace(staging, generation_dir)
        relative = os.path.relpath(generation_dir, registry).replace("\\", "/")
        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "training_run_id": run_id,
            "generation": relative,
            "generation_path": generation_dir,
            "generation_report_sha256": _sha256(
                os.path.join(generation_dir, "train_report.json")
            ),
            "dataset_sha256": dataset_sha256,
            "strict_full_rows": strict_count,
        }
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_candidate(
    args, frame, features, strict_count, targets, family_params_for,
    lock_timeout=1,
):
    """Build a candidate while owning the registry's sole writer lock."""
    with _registry_lock(args.registry, timeout=lock_timeout):
        return _build_candidate(
            args, frame, features, strict_count, targets, family_params_for
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--params", default=None)
    parser.add_argument("--weight-col", default="sample_weight")
    parser.add_argument("--min-rows", type=int, default=200)
    parser.add_argument("--registry", default=REGISTRY)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--result-json", default=None)
    parser.add_argument("--lock-timeout", type=float, default=1.0)
    args = parser.parse_args()

    from quality_contract import annotate_validity

    args.dataset = os.path.abspath(args.dataset)
    args.registry = os.path.abspath(args.registry)
    args.profile = os.path.abspath(args.profile) if args.profile else None
    args.params = os.path.abspath(args.params) if args.params else None
    args.result_json = (
        os.path.abspath(args.result_json) if args.result_json else None
    )
    if args.lock_timeout < 0:
        parser.error("lock timeout must be non-negative")

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

    candidate = build_candidate(
        args, frame, features, strict_count, targets, family_params_for,
        lock_timeout=args.lock_timeout,
    )
    if args.result_json:
        _atomic_json(candidate, args.result_json)
    print(json.dumps({"candidate_generation": candidate}, ensure_ascii=False))


if __name__ == "__main__":
    main()
