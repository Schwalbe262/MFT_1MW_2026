"""Tune only authenticated production-quality blockers.

This entry point is deliberately separate from :mod:`tune_optuna`.  It binds
one immutable failed quality report, its threshold document, its strict
dataset, and the corrected-capacitance recovery evidence before Optuna can
create a study.  Each trial is scored with the same physical metrics,
train/calibration/evaluation split, and conformal interval implementation used
by ``train_models.train_target``.  The resulting parameter generation is only
an input to a complete 21-target retrain; it is never a model promotion.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
REGRESSION_ROOT = HERE.parent
REPO_ROOT = REGRESSION_ROOT.parent
for path in (HERE, REGRESSION_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import tune_optuna  # noqa: E402
from campaign.train_io import (  # noqa: E402
    CAPACITANCE_RECOVERY_CONTRACT,
    capacitance_recovery_audit,
)
from checkpoint_train import TARGETS, feature_columns  # noqa: E402
from train_models import train_target  # noqa: E402


CONFIG_SCHEMA = "mft-production-blocker-hpo-config-v1"
OBJECTIVE_SCHEMA = "mft-production-quality-gate-objective-v1"
OBJECTIVE_NAME = "production_gate_metric_ratio_with_conformal_interval_v1"
TUNING_SCHEMA_VERSION = 2
NONFINITE_RATIO = 1_000_000.0
FOLDS_PER_TRIAL = 5


def _sha256(path):
    return tune_optuna._sha256(path)


def _canonical_sha256(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def _load_json(path, label):
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except Exception as exc:
        raise RuntimeError(f"{label} is unavailable: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _full_sha(value, length, label):
    if not isinstance(value, str) or not re.fullmatch(
        rf"[0-9a-fA-F]{{{length}}}", value
    ):
        raise RuntimeError(f"{label} must be a full SHA-{length * 4}")
    return value.lower()


def _positive_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RuntimeError(f"{label} must be a positive integer")
    return int(value)


def _finite_number(value):
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def load_blocker_config(path):
    """Load and structurally validate one immutable blocker selection."""

    config = _load_json(path, "blocker HPO config")
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise RuntimeError("unsupported blocker HPO config schema")
    if config.get("mode") != "production_blocker_only":
        raise RuntimeError("blocker HPO config is not production-only")
    if config.get("objective") != OBJECTIVE_NAME:
        raise RuntimeError("blocker HPO objective contract mismatch")

    targets = config.get("blockers")
    if (
        not isinstance(targets, list)
        or not targets
        or any(not isinstance(target, str) or not target for target in targets)
        or len(set(targets)) != len(targets)
    ):
        raise RuntimeError("blocker target manifest is invalid")
    unknown_targets = sorted(set(targets) - set(TARGETS))
    if unknown_targets:
        raise RuntimeError(f"unknown blocker targets: {unknown_targets}")

    families = config.get("families")
    if (
        not isinstance(families, list)
        or not families
        or len(set(families)) != len(families)
        or any(family not in tune_optuna.FAMILIES for family in families)
    ):
        raise RuntimeError("blocker model-family manifest is invalid")

    expected_reasons = config.get("expected_blocking_reasons")
    if not isinstance(expected_reasons, dict) or set(expected_reasons) != set(
        targets
    ):
        raise RuntimeError("blocking reason manifest does not match targets")
    for target, reasons in expected_reasons.items():
        if (
            not isinstance(reasons, list)
            or not reasons
            or any(not isinstance(reason, str) or not reason for reason in reasons)
            or len(set(reasons)) != len(reasons)
        ):
            raise RuntimeError(f"invalid blocking reasons for {target}")

    source = config.get("source")
    cohort = config.get("cohort")
    recovery = config.get("capacitance_recovery")
    if not all(isinstance(value, dict) for value in (source, cohort, recovery)):
        raise RuntimeError("source/cohort/recovery config is missing")
    for key in (
        "dataset_sha256", "profile_sha256", "quality_status_sha256",
        "quality_thresholds_sha256", "quality_thresholds_file_sha256",
    ):
        _full_sha(source.get(key), 64, f"source.{key}")
    _full_sha(cohort.get("solver_revision"), 40, "cohort.solver_revision")
    _full_sha(cohort.get("library_revision"), 40, "cohort.library_revision")
    _full_sha(
        cohort.get("data_contract_sha256"), 64,
        "cohort.data_contract_sha256",
    )
    _positive_integer(source.get("strict_full_rows"), "source.strict_full_rows")
    final_trials = _positive_integer(
        config.get("trials_per_job"), "trials_per_job"
    )
    trial_stages = config.get("trial_stages")
    if (
        not isinstance(trial_stages, list)
        or not trial_stages
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in trial_stages
        )
        or trial_stages != sorted(set(trial_stages))
        or trial_stages[-1] != final_trials
    ):
        raise RuntimeError(
            "trial_stages must be increasing cumulative totals ending at "
            "trials_per_job"
        )
    _positive_integer(
        config.get("minimum_eligible_rows_per_target"),
        "minimum_eligible_rows_per_target",
    )
    if recovery.get("contract") != CAPACITANCE_RECOVERY_CONTRACT:
        raise RuntimeError("capacitance recovery contract mismatch")
    _positive_integer(
        recovery.get("recovered_row_count"),
        "capacitance_recovery.recovered_row_count",
    )
    maximum_delta = _finite_number(recovery.get("max_allowed_abs_delta_F"))
    if maximum_delta is None or maximum_delta < 0.0:
        raise RuntimeError("capacitance recovery delta bound is invalid")
    return config


def _blocking_targets(quality_status):
    statuses = quality_status.get("targets")
    if not isinstance(statuses, dict):
        raise RuntimeError("quality status target inventory is unavailable")
    return sorted(
        target for target, status in statuses.items()
        if isinstance(status, dict)
        and status.get("blocking") is True
        and status.get("passed") is not True
    )


def _validate_source_documents(
    config, config_path, dataset_path, quality_status_path, thresholds_path,
):
    """Fail closed before loading models or creating an Optuna study."""

    source = config["source"]
    dataset_path = os.path.abspath(dataset_path)
    quality_status_path = os.path.abspath(quality_status_path)
    thresholds_path = os.path.abspath(thresholds_path)
    if _sha256(dataset_path) != source["dataset_sha256"].lower():
        raise RuntimeError("strict dataset fingerprint mismatch")
    if _sha256(quality_status_path) != source[
        "quality_status_sha256"
    ].lower():
        raise RuntimeError("source quality status fingerprint mismatch")

    quality = _load_json(quality_status_path, "source quality status")
    thresholds = _load_json(thresholds_path, "quality thresholds")
    thresholds_fingerprint = _canonical_sha256(thresholds)
    if thresholds_fingerprint != source["quality_thresholds_sha256"].lower():
        raise RuntimeError("quality threshold canonical fingerprint mismatch")
    if _sha256(thresholds_path) != source[
        "quality_thresholds_file_sha256"
    ].lower():
        raise RuntimeError("quality threshold file fingerprint mismatch")
    if quality.get("thresholds_sha256") != thresholds_fingerprint:
        raise RuntimeError("quality status threshold fingerprint mismatch")
    if quality.get("quality_thresholds_sha256") != _sha256(thresholds_path):
        raise RuntimeError("quality status threshold file fingerprint mismatch")

    checks = {
        "training_run_id": source.get("training_run_id"),
        "generation": source.get("generation"),
        "dataset_sha256": source.get("dataset_sha256"),
        "profile_sha256": source.get("profile_sha256"),
        "strict_full_rows": source.get("strict_full_rows"),
    }
    for key, expected in checks.items():
        if quality.get(key) != expected:
            raise RuntimeError(f"quality status {key} mismatch")
    if quality.get("passed") is not False:
        raise RuntimeError("source quality status is not an explicit failure")

    blockers = _blocking_targets(quality)
    if blockers != sorted(config["blockers"]):
        raise RuntimeError(
            "configured blockers differ from authenticated quality blockers"
        )
    for target in config["blockers"]:
        target_status = quality["targets"].get(target) or {}
        actual_reasons = sorted(target_status.get("reasons") or [])
        expected_reasons = sorted(
            config["expected_blocking_reasons"][target]
        )
        if actual_reasons != expected_reasons:
            raise RuntimeError(f"blocking reason drift for {target}")
        limits = (thresholds.get("targets") or {}).get(target)
        if not isinstance(limits, dict) or limits.get("blocking", True) is not True:
            raise RuntimeError(f"{target} is not blocking in quality thresholds")
    minimum_coverage = _finite_number(
        thresholds.get("minimum_interval_coverage")
    )
    if (
        minimum_coverage is None
        or minimum_coverage <= 0.0
        or minimum_coverage >= 1.0
    ):
        raise RuntimeError("minimum interval coverage is invalid")

    recovery = quality.get("capacitance_recovery")
    if (
        not isinstance(recovery, dict)
        or recovery.get("passed") is not True
        or recovery.get("reasons") not in ([], ())
        or recovery.get("contract") != CAPACITANCE_RECOVERY_CONTRACT
        or recovery.get("recovered_row_count")
        != config["capacitance_recovery"]["recovered_row_count"]
        or recovery.get("dataset_sha256") != source["dataset_sha256"]
        or recovery.get("profile_sha256") != source["profile_sha256"]
    ):
        raise RuntimeError("source quality capacitance recovery was rejected")

    return {
        "config": config,
        "config_sha256": _sha256(config_path),
        "quality_status": quality,
        "quality_status_sha256": _sha256(quality_status_path),
        "thresholds": thresholds,
        "quality_thresholds_sha256": thresholds_fingerprint,
        "quality_thresholds_file_sha256": _sha256(thresholds_path),
        "dataset_path": dataset_path,
        "dataset_sha256": _sha256(dataset_path),
    }


def _require_corrected_recovery(config, quality, audit, strict_count):
    """Require corrected recovery even though blocker targets are non-capacitance."""

    expected = config["capacitance_recovery"]
    source = config["source"]
    maximum_delta = float(expected["max_allowed_abs_delta_F"])
    observed = _finite_number(audit.get("max_observed_abs_delta_F"))
    source_observed = _finite_number(
        (quality.get("capacitance_recovery") or {}).get(
            "max_observed_abs_delta_F"
        )
    )
    if (
        audit.get("contract") != CAPACITANCE_RECOVERY_CONTRACT
        or audit.get("status") != "applied"
        or audit.get("recovered_row_count") != expected["recovered_row_count"]
        or int(strict_count) != int(source["strict_full_rows"])
        or observed is None
        or observed < 0.0
        or observed > maximum_delta
        or source_observed is None
        or not math.isclose(observed, source_observed, rel_tol=0.0, abs_tol=1e-25)
    ):
        raise RuntimeError(
            "production blocker HPO requires authenticated corrected "
            f"capacitance recovery: audit={audit}"
        )


def _eligible_target_counts(frame, targets):
    counts = {}
    for target in targets:
        eligible = tune_optuna.filter_valid_training_rows(frame, target)
        values = tune_optuna.pd.to_numeric(
            eligible[target], errors="coerce"
        ).to_numpy(dtype=float)
        counts[target] = int(np.isfinite(values).sum())
    return counts


def quality_gate_objective(metrics, limits, minimum_interval_coverage):
    """Score exact gate metrics; every ratio is <=1 precisely when it passes."""

    if not isinstance(metrics, dict) or not isinstance(limits, dict):
        raise ValueError("quality metrics and limits must be mappings")
    ratios = {}
    failures = []

    def add_minimum(metric, limit):
        value = _finite_number(metrics.get(metric))
        bound = _finite_number(limit)
        if value is None or bound is None or bound >= 1.0:
            ratio = NONFINITE_RATIO
        else:
            ratio = max(0.0, 1.0 - value) / (1.0 - bound)
        ratios[metric] = float(ratio)
        if ratio > 1.0:
            failures.append(f"metric_below_minimum:{metric}")

    def add_maximum(metric, limit):
        value = _finite_number(metrics.get(metric))
        bound = _finite_number(limit)
        if value is None or bound is None or bound <= 0.0 or value < 0.0:
            ratio = NONFINITE_RATIO
        else:
            ratio = value / bound
        ratios[metric] = float(ratio)
        if ratio > 1.0:
            failures.append(f"metric_above_maximum:{metric}")

    add_minimum("interval_coverage", minimum_interval_coverage)
    for key, limit in limits.items():
        if key == "blocking":
            continue
        if key.startswith("min_"):
            add_minimum(key.removeprefix("min_"), limit)
        elif key.startswith("max_"):
            add_maximum(key.removeprefix("max_"), limit)
        else:
            raise ValueError(f"unsupported quality threshold: {key}")

    excess = [max(0.0, ratio - 1.0) for ratio in ratios.values()]
    score = (
        len(failures) * 1_000_000.0
        + sum(excess) * 1_000.0
        + float(np.mean(list(ratios.values())))
    )
    return {
        "score": float(score),
        "failed_constraint_count": len(failures),
        "failed_constraints": failures,
        "normalized_gate_ratios": ratios,
        "metrics": dict(metrics),
    }


def objective_contract(target, limits, source_contract):
    document = {
        "schema_version": OBJECTIVE_SCHEMA,
        "name": OBJECTIVE_NAME,
        "target": target,
        "quality_thresholds_sha256": source_contract[
            "quality_thresholds_sha256"
        ],
        "minimum_interval_coverage": source_contract["thresholds"][
            "minimum_interval_coverage"
        ],
        "target_limits": dict(limits),
        "metric_implementation": "train_models.train_target",
        "training_split": {
            "seed": 42,
            "train_fraction": 0.80,
            "calibration_fraction": 0.10,
            "evaluation_fraction": 0.10,
            "folds_per_family": 5,
        },
        "interval": "q90_conformal_on_disjoint_calibration_partition",
        "family_scope": "single_family_proxy",
        "final_acceptance": (
            "complete_21_target_four_family_retrain_then_model_quality_gate"
        ),
        "score": (
            "1e6*failed_constraint_count+1e3*sum(excess_ratio)+mean(ratio)"
        ),
    }
    return {**document, "sha256": _canonical_sha256(document)}


def tune_one(
    target, family, requested_total_trials, frame, features, limits,
    source_contract, *, model_threads, sample_weight_column,
    minimum_eligible_rows, study_root, config_sha256,
):
    import optuna

    contract = objective_contract(target, limits, source_contract)
    minimum_coverage = float(
        source_contract["thresholds"]["minimum_interval_coverage"]
    )

    def objective(trial):
        params = tune_optuna.sample_params(trial, family)
        runtime_params = tune_optuna.model_params_with_thread_budget(
            family, params, model_threads
        )
        bundle, metrics = train_target(
            frame,
            features,
            target,
            TARGETS[target],
            {family: runtime_params},
            sample_weight_col=sample_weight_column,
            min_rows=minimum_eligible_rows,
        )
        if bundle is None:
            raise RuntimeError(f"{target} cannot be tuned: {metrics}")
        evidence = quality_gate_objective(metrics, limits, minimum_coverage)
        trial.set_user_attr("quality_gate_evidence", evidence)
        trial.report(evidence["score"], 0)
        return evidence["score"]

    identity = _canonical_sha256({
        "config_sha256": config_sha256,
        "objective_contract_sha256": contract["sha256"],
        "target": target,
        "family": family,
    })
    study_name = f"mft-blocker-{identity[:24]}"
    study_path = Path(study_root).resolve() / (
        f"{target}__{family}__{identity[:16]}.sqlite3"
    )
    study_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{study_path.as_posix()}"
    sampler_seed = 7 + int(requested_total_trials)
    storage_backend = optuna.storages.RDBStorage(url=storage)
    try:
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=sampler_seed),
            pruner=optuna.pruners.NopPruner(),
            storage=storage_backend,
            study_name=study_name,
            load_if_exists=True,
        )
        existing_contract = study.user_attrs.get("objective_contract_sha256")
        if existing_contract not in (None, contract["sha256"]):
            raise RuntimeError("resumed Optuna study objective contract mismatch")
        study.set_user_attr("objective_contract_sha256", contract["sha256"])
        study.set_user_attr("config_sha256", config_sha256)
        complete_state = optuna.trial.TrialState.COMPLETE
        completed_before = sum(
            trial.state == complete_state
            for trial in study.get_trials(deepcopy=False)
        )
        if completed_before > requested_total_trials:
            raise RuntimeError(
                f"{target}/{family} already has {completed_before} completed "
                f"trials, beyond requested stage {requested_total_trials}"
            )
        remaining = requested_total_trials - completed_before
        if remaining:
            study.optimize(objective, n_trials=remaining, show_progress_bar=False)
        completed_after = sum(
            trial.state == complete_state
            for trial in study.get_trials(deepcopy=False)
        )
        if completed_after != requested_total_trials:
            raise RuntimeError(
                f"{target}/{family} stage is incomplete: "
                f"{completed_after}/{requested_total_trials} completed trials"
            )
        evidence = study.best_trial.user_attrs.get("quality_gate_evidence")
        if not isinstance(evidence, dict):
            raise RuntimeError("best blocker HPO trial has no gate evidence")
        resume = {
            "study_name": study_name,
            "study_storage_path": str(study_path),
            "sampler_seed": sampler_seed,
            "requested_total_trials": int(requested_total_trials),
            "completed_trials_before": int(completed_before),
            "completed_trials_after": int(completed_after),
            "completed_trials_added": int(completed_after - completed_before),
        }
        return (
            study.best_params,
            float(study.best_value),
            evidence,
            contract,
            resume,
        )
    finally:
        storage_backend.remove_session()
        storage_backend.engine.dispose()


def run_blocker_jobs(
    jobs, requested_total_trials, frame, features, source_contract, *, model_threads,
    job_workers, max_model_thread_budget, sample_weight_column,
    minimum_eligible_rows, study_root, config_sha256,
):
    if (
        isinstance(job_workers, bool)
        or isinstance(max_model_thread_budget, bool)
        or int(job_workers) < 1
        or int(max_model_thread_budget) < 1
    ):
        raise ValueError("tuning worker and thread budgets must be positive")
    workers = min(int(job_workers), max(1, len(jobs)))
    if workers * int(model_threads) > int(max_model_thread_budget):
        raise ValueError(
            "job_workers * model_threads exceeds max_model_thread_budget"
        )

    def execute(index, target, family):
        limits = source_contract["thresholds"]["targets"][target]
        print(
            f"\n=== blocker tune {target} / {family} "
            f"(cumulative stage {requested_total_trials} trials) ==="
        )
        result = tune_one(
            target, family, requested_total_trials, frame, features, limits,
            source_contract,
            model_threads=model_threads,
            sample_weight_column=sample_weight_column,
            minimum_eligible_rows=minimum_eligible_rows,
            study_root=study_root,
            config_sha256=config_sha256,
        )
        return index, target, family, result

    completed = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(execute, index, target, family)
            for index, (target, family) in enumerate(jobs)
        ]
        for future in as_completed(futures):
            item = future.result()
            completed[item[0]] = item

    params = {}
    results = []
    for item in completed:
        if item is None:
            raise RuntimeError("blocker tuning result inventory is incomplete")
        _, target, family, result = item
        best, score, evidence, contract, resume = result
        params.setdefault(family, {})[target] = {
            "params": best,
            "quality_gate_objective_score": score,
            "quality_gate_objective_contract_sha256": contract["sha256"],
            "completed_trials": resume["completed_trials_after"],
        }
        results.append({
            "target": target,
            "family": family,
            "quality_gate_objective_score": score,
            "quality_gate_evidence": evidence,
            "objective_contract": contract,
            "resume": resume,
        })
    return params, results, workers


def _merge_base_params(base_params_path, tuned_params):
    if not base_params_path:
        return dict(tuned_params), None
    base = _load_json(base_params_path, "base tuning parameters")
    merged = json.loads(json.dumps(base))
    for family, targets in tuned_params.items():
        merged.setdefault(family, {}).update(targets)
    return merged, _sha256(base_params_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--quality-status", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--result-json", default=None)
    parser.add_argument("--base-params", default=None)
    parser.add_argument(
        "--study-root",
        default=None,
        help="persistent per-target Optuna SQLite studies used for append/resume",
    )
    parser.add_argument(
        "--stage-trials",
        type=int,
        default=None,
        help="cumulative trials per target/family; must be a configured stage",
    )
    parser.add_argument("--model-threads", type=int, default=1)
    parser.add_argument("--job-workers", type=int, default=1)
    parser.add_argument("--max-model-thread-budget", type=int, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    config = load_blocker_config(config_path)
    selected_stage = (
        args.stage_trials
        if args.stage_trials is not None
        else config["trial_stages"][0]
    )
    if selected_stage not in config["trial_stages"]:
        parser.error(
            "stage-trials must be one of "
            + ", ".join(str(value) for value in config["trial_stages"])
        )
    source_contract = _validate_source_documents(
        config,
        config_path,
        args.dataset,
        args.quality_status,
        args.thresholds,
    )
    if args.model_threads < 1 or args.job_workers < 1:
        parser.error("model threads and job workers must be positive")
    maximum_budget = (
        args.max_model_thread_budget
        if args.max_model_thread_budget is not None
        else args.model_threads
    )
    if maximum_budget < 1:
        parser.error("maximum model thread budget must be positive")
    actual_workers = min(
        args.job_workers,
        len(config["blockers"]) * len(config["families"]),
    )
    if actual_workers * args.model_threads > maximum_budget:
        parser.error(
            "effective job workers times model threads exceeds maximum budget"
        )
    if not args.preflight_only and (
        not args.artifact_root or not args.result_json or not args.study_root
    ):
        parser.error(
            "production blocker HPO requires explicit artifact-root, "
            "result-json, and persistent study-root"
        )

    cohort = config["cohort"]
    frame, strict_count = tune_optuna._load_strict_dataset(
        source_contract["dataset_path"],
        cohort["solver_revision"].lower(),
        cohort["library_revision"].lower(),
    )
    audit = capacitance_recovery_audit(frame)
    _require_corrected_recovery(
        config, source_contract["quality_status"], audit, strict_count
    )
    features = feature_columns(frame)
    if not features:
        raise RuntimeError("no design-time blocker HPO features remain")
    eligible_counts = _eligible_target_counts(frame, config["blockers"])
    insufficient = {
        target: count for target, count in eligible_counts.items()
        if count < config["minimum_eligible_rows_per_target"]
    }
    if insufficient:
        raise RuntimeError(
            "blocker targets have insufficient eligible strict-full rows: "
            f"{insufficient}"
        )
    jobs = [
        (target, family)
        for target in config["blockers"]
        for family in config["families"]
    ]
    preflight = {
        "schema_version": CONFIG_SCHEMA,
        "ready": True,
        "config_sha256": source_contract["config_sha256"],
        "dataset_sha256": source_contract["dataset_sha256"],
        "strict_full_rows": int(strict_count),
        "capacitance_recovery": audit,
        "quality_status_sha256": source_contract["quality_status_sha256"],
        "quality_thresholds_sha256": source_contract[
            "quality_thresholds_sha256"
        ],
        "blockers": list(config["blockers"]),
        "families": list(config["families"]),
        "job_count": len(jobs),
        "trial_stages": list(config["trial_stages"]),
        "selected_cumulative_trials_per_job": selected_stage,
        "planned_trials_from_clean_studies": len(jobs) * selected_stage,
        "planned_model_fits_from_clean_studies": (
            len(jobs) * selected_stage * FOLDS_PER_TRIAL
        ),
        "folds_per_trial": FOLDS_PER_TRIAL,
        "minimum_eligible_rows_per_target": config[
            "minimum_eligible_rows_per_target"
        ],
        "eligible_rows_by_target": eligible_counts,
        "resume_semantics": (
            "append_completed_trials_to_selected_cumulative_stage"
        ),
        "feature_count": len(features),
        "model_threads": args.model_threads,
        "job_workers": actual_workers,
        "maximum_model_thread_budget": maximum_budget,
    }
    if args.preflight_only:
        print(json.dumps({"preflight": preflight}, ensure_ascii=False))
        return

    tuned, results, workers = run_blocker_jobs(
        jobs,
        selected_stage,
        frame,
        features,
        source_contract,
        model_threads=args.model_threads,
        job_workers=args.job_workers,
        max_model_thread_budget=maximum_budget,
        sample_weight_column=config.get("sample_weight_column"),
        minimum_eligible_rows=config["minimum_eligible_rows_per_target"],
        study_root=os.path.abspath(args.study_root),
        config_sha256=source_contract["config_sha256"],
    )
    params, base_params_sha256 = _merge_base_params(args.base_params, tuned)
    metadata = {
        "tuning_schema_version": TUNING_SCHEMA_VERSION,
        "lane": "production_gate_blocker_hpo",
        "scope": "authenticated_blockers_only",
        "parameter_artifact_eligible": True,
        "production_eligible": False,
        "production_model_eligible": False,
        "fea_submission_approved": False,
        "promotion_approved": False,
        "final_acceptance_required": (
            "complete_21_target_retrain_and_model_quality_gate"
        ),
        "config_sha256": source_contract["config_sha256"],
        "source_quality_status_sha256": source_contract[
            "quality_status_sha256"
        ],
        "source_training_run_id": config["source"]["training_run_id"],
        "source_generation": config["source"]["generation"],
        "dataset_sha256": source_contract["dataset_sha256"],
        "strict_full_rows": int(strict_count),
        "profile_sha256": config["source"]["profile_sha256"],
        "capacitance_recovery": audit,
        "quality_thresholds_sha256": source_contract[
            "quality_thresholds_sha256"
        ],
        "quality_thresholds_file_sha256": source_contract[
            "quality_thresholds_file_sha256"
        ],
        "solver_revision": cohort["solver_revision"].lower(),
        "library_revision": cohort["library_revision"].lower(),
        "data_contract_sha256": cohort["data_contract_sha256"].lower(),
        "search_implementation_sha256": _sha256(__file__),
        "base_params_sha256": base_params_sha256,
        "blockers": list(config["blockers"]),
        "families": list(config["families"]),
        "trial_stages": list(config["trial_stages"]),
        "configured_final_trials_per_job": config["trials_per_job"],
        "selected_cumulative_trials_per_job": selected_stage,
        "model_threads": args.model_threads,
        "job_workers": workers,
        "maximum_total_model_threads": workers * args.model_threads,
        "max_model_thread_budget": maximum_budget,
        "sampler": "TPESampler(seed=7+selected_cumulative_trials_per_job)",
        "sampler_seed": 7 + selected_stage,
        "pruner": "NopPruner",
        "jobs": results,
    }
    result = tune_optuna._publish_generation(
        os.path.abspath(args.artifact_root),
        params,
        metadata,
        result_json=os.path.abspath(args.result_json),
    )
    print(json.dumps({"preflight": preflight, "result": result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
