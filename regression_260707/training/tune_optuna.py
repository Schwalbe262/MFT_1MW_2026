"""Create an immutable Optuna parameter generation.

Production tuning starts at 4,000 recomputed strict-full rows.  The durable
pipeline decides the subsequent 2,000-row/20-percent cadence; this command
authenticates its exact dataset, search implementation, revision pins, and
output params so checkpoint training can consume them without a mutable
``best_params.json`` race.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REGRESSION_ROOT = HERE.parent
REPO_ROOT = REGRESSION_ROOT.parent
if str(REGRESSION_ROOT) not in sys.path:
    sys.path.insert(0, str(REGRESSION_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASET = REGRESSION_ROOT / "data" / "dataset" / "train.parquet"
LEGACY_OUT = HERE / "best_params.json"
MIN_STRICT_FULL_ROWS = 4000
FAMILIES = ("lightgbm", "xgboost", "catboost", "extratrees")
TUNING_SCHEMA_VERSION = 1

from checkpoint_train import (  # noqa: E402
    TARGETS,
    feature_columns,
    filter_valid_training_rows,
    to_physical,
    transform_y,
)
from train_models import make_model  # noqa: E402


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(value, path):
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, staged = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp",
        dir=os.path.dirname(path),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=1, ensure_ascii=False, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, path)
    finally:
        if os.path.exists(staged):
            os.remove(staged)


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
            verbose=-1,
        )
    if family == "xgboost":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 400, 3000),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            max_depth=trial.suggest_int("max_depth", 4, 12),
            min_child_weight=trial.suggest_float("min_child_weight", 1, 20),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            verbosity=0,
        )
    if family == "catboost":
        return dict(
            iterations=trial.suggest_int("iterations", 400, 3000),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            depth=trial.suggest_int("depth", 4, 10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.5, 30, log=True),
            verbose=0,
        )
    if family == "extratrees":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1500),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_float("max_features", 0.4, 1.0),
            n_jobs=-1,
        )
    raise ValueError(family)


def tune(target, family, trials, df, feats):
    import optuna
    from sklearn.model_selection import KFold

    eligible = filter_valid_training_rows(df, target)
    sub = eligible.dropna(subset=[target])
    sub = sub[np.isfinite(sub[target])]
    if len(sub) < 4:
        raise RuntimeError(f"target {target} has fewer than four eligible rows")
    X = sub[feats].fillna(0.0).reset_index(drop=True)
    y = transform_y(
        sub[target].to_numpy(dtype=float), TARGETS[target]["transform"]
    )
    folds = KFold(n_splits=4, shuffle=True, random_state=0)

    def objective(trial):
        params = sample_params(trial, family)
        errors = []
        for fold_index, (train_index, test_index) in enumerate(folds.split(X)):
            model = make_model(family, params, seed=fold_index)
            model.fit(X.iloc[train_index], y[train_index])
            prediction = model.predict(X.iloc[test_index])
            errors.append(float(np.mean((prediction - y[test_index]) ** 2)))
            trial.report(float(np.mean(errors)), fold_index)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(errors))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=7),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return study.best_params, study.best_value, len(sub)


def _load_strict_dataset(path, solver_revision=None, library_revision=None):
    from quality_contract import annotate_validity

    raw = pd.read_parquet(path)
    audited = annotate_validity(
        raw,
        expected_solver_revision=solver_revision,
        expected_library_revision=library_revision,
    )
    strict_count = int(audited["_strict_valid_full"].sum())
    return to_physical(audited), strict_count


def _publish_generation(
    artifact_root, params, metadata, result_json=None, legacy_output=None
):
    from pipeline.artifacts import GenerationStore

    with tempfile.TemporaryDirectory(prefix="mft-optuna-") as directory:
        params_path = Path(directory) / "params.json"
        _atomic_json(params, params_path)
        generation = GenerationStore(artifact_root).publish_files(
            "tuning",
            {"params.json": params_path},
            metadata=metadata,
            parents=[f"dataset:{metadata['dataset_sha256']}"],
        )
    result = {
        "schema_version": TUNING_SCHEMA_VERSION,
        "generation_id": generation.generation_id,
        "generation_path": str(generation.path),
        "params_path": str(generation.path / "params.json"),
        "manifest_path": str(generation.path / "manifest.json"),
    }
    if result_json:
        _atomic_json(result, result_json)
    if legacy_output:
        # Compatibility is explicit.  Training workers consume the immutable
        # generation path returned above, never this convenience copy.
        _atomic_json(params, legacy_output)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=None)
    parser.add_argument("--family", choices=FAMILIES, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--result-json", default=None)
    parser.add_argument("--base-params", default=None)
    parser.add_argument("--legacy-output", default=None)
    parser.add_argument("--min-strict-full-rows", type=int, default=MIN_STRICT_FULL_ROWS)
    parser.add_argument("--solver-revision", default=None)
    parser.add_argument("--library-revision", default=None)
    parser.add_argument("--data-contract-sha256", default=None)
    args = parser.parse_args()

    cohort_pins = (
        bool(args.solver_revision),
        bool(args.library_revision),
        bool(args.data_contract_sha256),
    )
    if any(cohort_pins) and not all(cohort_pins):
        parser.error(
            "solver, library, and data-contract revisions must be pinned together"
        )
    for label, value in (
        ("solver", args.solver_revision), ("library", args.library_revision)
    ):
        if value and not re.fullmatch(r"[0-9a-fA-F]{40}", value):
            parser.error(f"{label} revision must be a full SHA")
    if args.data_contract_sha256 and not re.fullmatch(
        r"[0-9a-fA-F]{64}", args.data_contract_sha256
    ):
        parser.error("data contract must be a full SHA-256")
    if args.trials < 1 or args.min_strict_full_rows < MIN_STRICT_FULL_ROWS:
        parser.error("trials must be positive and the production row gate is >=4000")
    if not ((args.target and args.family) or args.all):
        parser.error("supply --target with --family, or --all")
    if args.all and (args.target or args.family):
        parser.error("--all cannot be combined with a single target/family")
    if args.target and args.target not in TARGETS:
        parser.error(f"unknown target: {args.target}")

    dataset = os.path.abspath(args.dataset)
    frame, strict_count = _load_strict_dataset(
        dataset,
        args.solver_revision.lower() if args.solver_revision else None,
        args.library_revision.lower() if args.library_revision else None,
    )
    if strict_count < args.min_strict_full_rows:
        raise SystemExit(
            f"Optuna requires >= {args.min_strict_full_rows} strict-full rows; "
            f"found {strict_count}"
        )
    features = feature_columns(frame)
    if not features:
        raise SystemExit("no design-time tuning features remain")
    params = {}
    if args.base_params:
        with open(args.base_params, encoding="utf-8") as handle:
            params = json.load(handle)
    jobs = (
        [(args.target, args.family)]
        if args.target and args.family
        else [
            (target, family)
            for target in TARGETS if target in frame.columns
            for family in FAMILIES
        ]
    )
    results = []
    for target, family in jobs:
        print(f"\n=== tune {target} / {family} ({args.trials} trials) ===")
        best, value, eligible_rows = tune(
            target, family, args.trials, frame, features
        )
        params.setdefault(family, {})[target] = {
            "params": best,
            "cv_mse_transformed": value,
        }
        results.append({
            "target": target,
            "family": family,
            "eligible_rows": eligible_rows,
            "cv_mse_transformed": value,
        })

    artifact_root = os.path.abspath(
        args.artifact_root or REGRESSION_ROOT / "pipeline_runtime" / "artifacts"
    )
    metadata = {
        "tuning_schema_version": TUNING_SCHEMA_VERSION,
        "dataset_sha256": _sha256(dataset),
        "strict_full_rows": strict_count,
        "solver_revision": (
            args.solver_revision.lower() if args.solver_revision else None
        ),
        "library_revision": (
            args.library_revision.lower() if args.library_revision else None
        ),
        "data_contract_sha256": (
            args.data_contract_sha256.lower()
            if args.data_contract_sha256 else None
        ),
        "search_implementation_sha256": _sha256(__file__),
        "feature_schema_sha256": hashlib.sha256(
            json.dumps(features, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "trials_per_job": args.trials,
        "jobs": results,
        "sampler": "TPESampler(seed=7)",
        "pruner": "MedianPruner(n_warmup_steps=1)",
    }
    result = _publish_generation(
        artifact_root,
        params,
        metadata,
        result_json=args.result_json,
        legacy_output=args.legacy_output,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
