"""Nested-holdout, deterministic blocker-only HPO.

Version 2 intentionally cannot read or resume the exploratory version-1
studies.  The exact rows used by the final ``SEED=42`` model-quality
evaluation are sealed first and removed from every HPO fit.  Optuna provides
durable trial storage, while proposals are an explicit deterministic sequence
rather than sampler state that cannot be reproduced across process restarts.

The output remains a parameter-only input to a complete 21-target retrain.  It
does not approve a model, FEA submission, or registry promotion.
"""

from __future__ import annotations

import argparse
import atexit
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
import tempfile
import threading
import time
import uuid

# These variables are set before NumPy or any estimator runtime is imported.
# They are also authenticated in every objective contract.  Model-library
# thread parameters remain pinned independently below.
FORCED_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
for _thread_variable, _thread_value in FORCED_THREAD_ENVIRONMENT.items():
    os.environ[_thread_variable] = _thread_value

import numpy as np  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402


HERE = Path(__file__).resolve().parent
REGRESSION_ROOT = HERE.parent
REPO_ROOT = REGRESSION_ROOT.parent
for path in (HERE, REGRESSION_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_models  # noqa: E402
import tune_blockers as blocker_v1  # noqa: E402
import tune_optuna  # noqa: E402
from campaign.train_io import (  # noqa: E402
    CAPACITANCE_RECOVERY_CONTRACT,
    capacitance_recovery_audit,
)
from checkpoint_train import TARGETS, feature_columns, filter_valid_training_rows  # noqa: E402


CONFIG_SCHEMA = "mft-production-blocker-hpo-config-v2"
OBJECTIVE_SCHEMA = "mft-production-quality-nested-holdout-objective-v2"
OBJECTIVE_NAME = "nested_gate_metric_ratio_deterministic_proposals_v2"
STUDY_SCHEMA = "mft-production-blocker-optuna-study-v2"
RESULT_SCHEMA = "mft-production-blocker-hpo-result-v2"
HOLDOUT_SCHEMA = "mft-final-seed42-evaluation-holdouts-v2"
PROPOSAL_SCHEMA = "mft-deterministic-hash-random-proposals-v1"
STAGE_HISTORY_SCHEMA = "mft-blocker-hpo-stage-history-v2"
TRIAL_EVIDENCE_SCHEMA = "mft-blocker-hpo-trial-evidence-v2"
LIVE_STATUS_SCHEMA = "mft-blocker-hpo-live-status-v2"
TUNING_SCHEMA_VERSION = 2
FOLDS_PER_TRIAL = 5
_ACTIVE_LIVE_STATUS = None

IMPLEMENTATION_FILES = {
    "tune_blockers_v2.py": HERE / "tune_blockers_v2.py",
    "tune_blockers.py": HERE / "tune_blockers.py",
    "tune_optuna.py": HERE / "tune_optuna.py",
    "train_models.py": HERE / "train_models.py",
    "checkpoint_train.py": HERE / "checkpoint_train.py",
    "model_targets.py": REGRESSION_ROOT / "model_targets.py",
    "campaign/train_io.py": REGRESSION_ROOT / "campaign" / "train_io.py",
    "quality_contract.py": REGRESSION_ROOT / "quality_contract.py",
    "pipeline/artifacts.py": REGRESSION_ROOT / "pipeline" / "artifacts.py",
    "training/outer_acceptance_ledger.py": HERE / "outer_acceptance_ledger.py",
}
RUNTIME_DISTRIBUTIONS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "scipy": "scipy",
    "scikit_learn": "scikit-learn",
    "joblib": "joblib",
    "threadpoolctl": "threadpoolctl",
    "optuna": "optuna",
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "catboost": "catboost",
    "psutil": "psutil",
}

# This is the executable search contract.  Constants are applied to the model
# but are not stored as Optuna suggestions, matching train_models consumption.
SEARCH_SPACES = {
    "lightgbm": {
        "suggestions": [
            ["n_estimators", "int", 400, 3000],
            ["learning_rate", "log_float", 0.01, 0.15],
            ["num_leaves", "int", 31, 255],
            ["min_child_samples", "int", 5, 60],
            ["subsample", "float", 0.6, 1.0],
            ["colsample_bytree", "float", 0.6, 1.0],
            ["reg_lambda", "log_float", 0.001, 10.0],
        ],
        "constants": {"verbose": -1},
    },
    "xgboost": {
        "suggestions": [
            ["n_estimators", "int", 400, 3000],
            ["learning_rate", "log_float", 0.01, 0.15],
            ["max_depth", "int", 4, 12],
            ["min_child_weight", "float", 1.0, 20.0],
            ["subsample", "float", 0.6, 1.0],
            ["colsample_bytree", "float", 0.6, 1.0],
            ["reg_lambda", "log_float", 0.001, 10.0],
        ],
        "constants": {"verbosity": 0},
    },
    "catboost": {
        "suggestions": [
            ["iterations", "int", 400, 3000],
            ["learning_rate", "log_float", 0.01, 0.15],
            ["depth", "int", 4, 10],
            ["l2_leaf_reg", "log_float", 0.5, 30.0],
        ],
        "constants": {"verbose": 0},
    },
    "extratrees": {
        "suggestions": [
            ["n_estimators", "int", 200, 1500],
            ["min_samples_leaf", "int", 1, 10],
            ["max_features", "float", 0.4, 1.0],
        ],
        "constants": {},
    },
}


def _sha256(path):
    return tune_optuna._sha256(path)


def _canonical_sha256(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def _load_json(path, label):
    return blocker_v1._load_json(path, label)


def _positive_integer(value, label):
    return blocker_v1._positive_integer(value, label)


def _full_sha(value, length, label):
    return blocker_v1._full_sha(value, length, label)


def load_blocker_config(path):
    """Validate the v2 manifest without accepting a v1 study contract."""

    config = _load_json(path, "blocker HPO v2 config")
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise RuntimeError("unsupported blocker HPO v2 config schema")
    if config.get("mode") != "production_blocker_only_nested_holdout":
        raise RuntimeError("blocker HPO v2 mode mismatch")
    if config.get("objective") != OBJECTIVE_NAME:
        raise RuntimeError("blocker HPO v2 objective mismatch")

    targets = config.get("blockers")
    if (
        not isinstance(targets, list) or not targets
        or len(set(targets)) != len(targets)
        or any(not isinstance(target, str) or target not in TARGETS for target in targets)
    ):
        raise RuntimeError("blocker HPO v2 target manifest is invalid")
    if tuple(targets) != train_models.V2_BLOCKER_TARGETS:
        raise RuntimeError("blocker HPO v2 requires the exact production blocker set")
    families = config.get("families")
    if (
        not isinstance(families, list) or not families
        or len(set(families)) != len(families)
        or any(family not in SEARCH_SPACES for family in families)
    ):
        raise RuntimeError("blocker HPO v2 family manifest is invalid")
    if tuple(families) != train_models.V2_HPO_FAMILIES:
        raise RuntimeError("blocker HPO v2 requires the exact model-family set")
    reasons = config.get("expected_blocking_reasons")
    if not isinstance(reasons, dict) or set(reasons) != set(targets):
        raise RuntimeError("blocker HPO v2 reason manifest is invalid")
    for target, values in reasons.items():
        if (
            not isinstance(values, list) or not values
            or len(set(values)) != len(values)
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise RuntimeError(f"invalid blocker reasons for {target}")

    source = config.get("source")
    cohort = config.get("cohort")
    recovery = config.get("capacitance_recovery")
    if not all(isinstance(value, dict) for value in (source, cohort, recovery)):
        raise RuntimeError("blocker HPO v2 source/cohort/recovery is missing")
    for key in (
        "dataset_sha256", "profile_sha256", "quality_status_sha256",
        "quality_thresholds_sha256", "quality_thresholds_file_sha256",
    ):
        _full_sha(source.get(key), 64, f"source.{key}")
    _positive_integer(source.get("strict_full_rows"), "source.strict_full_rows")
    _full_sha(cohort.get("solver_revision"), 40, "cohort.solver_revision")
    _full_sha(cohort.get("library_revision"), 40, "cohort.library_revision")
    if "data_contract_sha256" in cohort or "data_contract_sha256" in config:
        raise RuntimeError(
            "v2 refuses an unverified data_contract_sha256 metadata claim"
        )
    if recovery.get("contract") != CAPACITANCE_RECOVERY_CONTRACT:
        raise RuntimeError("capacitance recovery contract mismatch")
    _positive_integer(
        recovery.get("recovered_row_count"),
        "capacitance_recovery.recovered_row_count",
    )
    maximum_delta = blocker_v1._finite_number(
        recovery.get("max_allowed_abs_delta_F")
    )
    if maximum_delta is None or maximum_delta < 0.0:
        raise RuntimeError("capacitance recovery delta bound is invalid")

    final_trials = _positive_integer(config.get("trials_per_job"), "trials_per_job")
    stages = config.get("trial_stages")
    if (
        not isinstance(stages, list) or not stages
        or stages != sorted(set(stages)) or stages[-1] != final_trials
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in stages)
    ):
        raise RuntimeError("v2 trial stages are invalid cumulative totals")
    if stages[0] <= 8:
        raise RuntimeError("v2 cannot reuse the exploratory eight-trial stage")
    maximum_workers = _positive_integer(
        config.get("maximum_job_workers"), "maximum_job_workers"
    )
    maximum_thread_budget = _positive_integer(
        config.get("maximum_model_thread_budget"),
        "maximum_model_thread_budget",
    )
    maximum_rss_gb = blocker_v1._finite_number(
        config.get("maximum_process_rss_gb")
    )
    if maximum_rss_gb is None or maximum_rss_gb <= 0.0:
        raise RuntimeError("maximum_process_rss_gb must be positive")
    execution = config.get("objective_execution_contract")
    if not isinstance(execution, dict):
        raise RuntimeError("objective execution contract is missing")
    if execution.get("model_threads") != 1:
        raise RuntimeError("v2 objective model_threads must be exactly one")
    if execution.get("thread_environment") != FORCED_THREAD_ENVIRONMENT:
        raise RuntimeError("v2 objective thread environment contract mismatch")
    if maximum_workers != 8 or maximum_thread_budget != 8:
        raise RuntimeError("v2 authenticated concurrency ceiling must be 8x1")
    _positive_integer(
        config.get("minimum_eligible_rows_per_target"),
        "minimum_eligible_rows_per_target",
    )
    return config


def _runtime_evidence(features):
    thread_environment = {
        key: os.environ.get(key) for key in FORCED_THREAD_ENVIRONMENT
    }
    if thread_environment != FORCED_THREAD_ENVIRONMENT:
        raise RuntimeError("v2 forced thread environment drifted")
    files = {}
    for label, path in IMPLEMENTATION_FILES.items():
        if not path.is_file():
            raise RuntimeError(f"implementation file is unavailable: {path}")
        files[label] = _sha256(path)
    packages = {}
    for label, distribution in RUNTIME_DISTRIBUTIONS.items():
        try:
            packages[label] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"required HPO runtime distribution is missing: {distribution}"
            ) from exc
    feature_document = {"features": list(features)}
    feature_document["sha256"] = _canonical_sha256(feature_document)
    target_configs = {
        target: {
            "config": TARGETS[target],
            "sha256": _canonical_sha256(TARGETS[target]),
        }
        for target in TARGETS
    }
    search_document = {
        "schema_version": PROPOSAL_SCHEMA,
        "spaces": SEARCH_SPACES,
    }
    search_document["sha256"] = _canonical_sha256(search_document)
    document = {
        "files": files,
        "feature_schema": feature_document,
        "target_configs": target_configs,
        "search_space": search_document,
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "packages": packages,
            "forced_thread_environment": thread_environment,
        },
    }
    return {**document, "sha256": _canonical_sha256(document)}


def _eligible_target_frame(frame, target):
    eligible = filter_valid_training_rows(frame, target)
    eligible = eligible.dropna(subset=[target])
    values = tune_optuna.pd.to_numeric(
        eligible[target], errors="coerce"
    ).to_numpy(dtype=float)
    return eligible.loc[np.isfinite(values)].copy()


def seal_final_evaluation_holdout(frame, target, dataset_sha256):
    """Seal the exact rows selected by train_models' final SEED=42 split."""

    from sklearn.model_selection import train_test_split

    if (
        train_models.SEED != 42
        or train_models.CALIBRATION_FRAC != 0.10
        or train_models.EVALUATION_FRAC != 0.10
        or train_models.N_FOLDS != FOLDS_PER_TRIAL
    ):
        raise RuntimeError("final model split implementation drifted from v2")
    if not frame.index.is_unique:
        raise RuntimeError("v2 holdout sealing requires a unique frame index")
    eligible = _eligible_target_frame(frame, target)
    relative = np.arange(len(eligible), dtype=int)
    _, holdout = train_test_split(
        relative,
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
    eligible_labels = eligible.index.to_numpy()
    evaluation_labels = eligible_labels[np.asarray(evaluation, dtype=int)]
    absolute_positions = frame.index.get_indexer(evaluation_labels)
    if len(absolute_positions) != len(evaluation) or np.any(absolute_positions < 0):
        raise RuntimeError("final evaluation rows cannot be mapped to dataset positions")
    absolute_positions = sorted(int(value) for value in absolute_positions.tolist())
    document = {
        "schema_version": HOLDOUT_SCHEMA,
        "target": target,
        "dataset_sha256": dataset_sha256,
        "eligible_row_count": int(len(eligible)),
        "evaluation_row_count": len(absolute_positions),
        "absolute_row_positions": absolute_positions,
        "split": {
            "seed": 42,
            "holdout_fraction": 0.20,
            "evaluation_fraction_of_holdout": 0.50,
            "evaluation_seed": 43,
        },
    }
    return {**document, "sha256": _canonical_sha256(document)}


def frame_without_holdout(frame, holdout):
    positions = np.asarray(holdout["absolute_row_positions"], dtype=int)
    if len(positions) != int(holdout["evaluation_row_count"]):
        raise RuntimeError("holdout row count evidence mismatch")
    if len(set(positions.tolist())) != len(positions):
        raise RuntimeError("holdout contains duplicate row positions")
    if len(positions) and (positions.min() < 0 or positions.max() >= len(frame)):
        raise RuntimeError("holdout row position is outside the strict dataset")
    keep = np.ones(len(frame), dtype=bool)
    keep[positions] = False
    output = frame.iloc[keep].copy()
    output.attrs.update(frame.attrs)
    return output


def _unit_interval(identity, ordinal, parameter, salt):
    digest = hashlib.sha256(
        f"{identity}:{ordinal}:{parameter}:{salt}".encode("utf-8")
    ).digest()
    integer = int.from_bytes(digest, "big")
    return (integer + 0.5) / float(1 << (8 * len(digest)))


def deterministic_proposal(identity, family, ordinal, salt=0):
    if family not in SEARCH_SPACES:
        raise ValueError(family)
    params = {}
    for name, kind, low, high in SEARCH_SPACES[family]["suggestions"]:
        unit = _unit_interval(identity, ordinal, name, salt)
        if kind == "int":
            span = int(high) - int(low) + 1
            raw = int.from_bytes(hashlib.sha256(
                f"{identity}:{ordinal}:{name}:{salt}:int".encode("utf-8")
            ).digest(), "big")
            value = int(low) + raw % span
        elif kind == "float":
            value = float(low) + unit * (float(high) - float(low))
        elif kind == "log_float":
            value = math.exp(
                math.log(float(low))
                + unit * (math.log(float(high)) - math.log(float(low)))
            )
        else:
            raise RuntimeError(f"unsupported v2 search parameter kind: {kind}")
        params[name] = value
    return params


def deterministic_proposal_sequence(identity, family, total):
    proposals = []
    used = set()
    for ordinal in range(int(total)):
        salt = 0
        while True:
            params = deterministic_proposal(identity, family, ordinal, salt=salt)
            fingerprint = _canonical_sha256(params)
            if fingerprint not in used:
                break
            salt += 1
            if salt > 1000:
                raise RuntimeError("cannot construct a unique deterministic proposal")
        used.add(fingerprint)
        proposals.append({
            "ordinal": ordinal,
            "salt": salt,
            "params": params,
            "sha256": fingerprint,
        })
    return proposals


def _suggest_fixed_proposal(trial, family):
    params = {}
    for name, kind, low, high in SEARCH_SPACES[family]["suggestions"]:
        if kind == "int":
            params[name] = trial.suggest_int(name, int(low), int(high))
        elif kind == "float":
            params[name] = trial.suggest_float(name, float(low), float(high))
        elif kind == "log_float":
            params[name] = trial.suggest_float(
                name, float(low), float(high), log=True
            )
        else:
            raise RuntimeError(f"unsupported v2 search parameter kind: {kind}")
    return params


class RssGuard:
    def __init__(self, maximum_gb):
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError("v2 RSS guard requires psutil") from exc
        self.process = psutil.Process(os.getpid())
        self.maximum_bytes = int(float(maximum_gb) * (1024 ** 3))
        if self.maximum_bytes < 1:
            raise ValueError("maximum RSS must be positive")
        self.maximum_observed_bytes = 0
        self.exceeded_event = threading.Event()
        self.monitor_stop_event = threading.Event()
        self.monitor_thread = None
        self.cancellation_event = None

    def _observed_bytes(self):
        processes = [self.process]
        try:
            processes.extend(self.process.children(recursive=True))
        except Exception:
            pass
        observed = 0
        for process in processes:
            try:
                observed += int(process.memory_info().rss)
            except Exception:
                continue
        self.maximum_observed_bytes = max(self.maximum_observed_bytes, observed)
        return observed

    def check(self, label):
        observed = self._observed_bytes()
        if observed > self.maximum_bytes:
            self.exceeded_event.set()
            if self.cancellation_event is not None:
                self.cancellation_event.set()
            raise RuntimeError(
                f"RSS guard exceeded at {label}: {observed}>{self.maximum_bytes}"
            )
        if self.exceeded_event.is_set():
            raise RuntimeError(
                "RSS guard was exceeded during an in-flight model fit: "
                f"peak={self.maximum_observed_bytes}>{self.maximum_bytes}"
            )
        return observed

    def start(self, cancellation_event, interval_seconds=0.5):
        if self.monitor_thread is not None:
            raise RuntimeError("RSS watchdog was already started")
        self.cancellation_event = cancellation_event
        self.monitor_stop_event.clear()

        def monitor():
            while not self.monitor_stop_event.wait(interval_seconds):
                try:
                    observed = self._observed_bytes()
                except Exception:
                    continue
                if observed > self.maximum_bytes:
                    self.exceeded_event.set()
                    cancellation_event.set()
                    return

        self.monitor_thread = threading.Thread(
            target=monitor,
            name="mft-blocker-hpo-v2-rss-watchdog",
            daemon=True,
        )
        self.monitor_thread.start()

    def stop(self):
        self.monitor_stop_event.set()
        if self.monitor_thread is not None and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        self.monitor_thread = None


class RunLease:
    """One fail-closed cross-process writer lease for a v2 study root."""

    def __init__(self, root, *, lease_seconds, identity):
        self.root = Path(root).resolve()
        self.path = self.root / ".mft-blocker-hpo-v2.lock"
        self.lease_seconds = float(lease_seconds)
        if self.lease_seconds < 5.0:
            raise ValueError("v2 lease must be at least five seconds")
        self.identity = dict(identity)
        self.token = uuid.uuid4().hex
        self.stop_event = threading.Event()
        self.lost_event = threading.Event()
        self.thread = threading.Thread(
            target=self._heartbeat,
            name="mft-blocker-hpo-v2-lease",
            daemon=True,
        )

    def _payload(self):
        now = time.time()
        return {
            "schema_version": STUDY_SCHEMA,
            "token": self.token,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "heartbeat_epoch": now,
            "expires_epoch": now + self.lease_seconds,
            **self.identity,
        }

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(
                self.path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError as exc:
            try:
                owner = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                owner = {"status": "unreadable"}
            raise RuntimeError(
                "blocker HPO v2 study root is already leased; never auto-steal "
                f"it (see docs/HPO_V2_STALE_RUN_RECOVERY.md): {owner}"
            ) from exc
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self._payload(), handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            self.path.unlink(missing_ok=True)
            raise
        self.thread.start()
        return self

    def _heartbeat(self):
        interval = max(1.0, self.lease_seconds / 3.0)
        while not self.stop_event.wait(interval):
            try:
                current = json.loads(self.path.read_text(encoding="utf-8"))
                if current.get("token") != self.token:
                    raise RuntimeError("v2 lease ownership token changed")
                tune_optuna._atomic_json(self._payload(), self.path)
            except Exception:
                self.lost_event.set()
                return

    def ensure_owned(self):
        if self.lost_event.is_set():
            raise RuntimeError("blocker HPO v2 lease was lost")
        try:
            current = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError("blocker HPO v2 lease is unreadable") from exc
        if current.get("token") != self.token:
            self.lost_event.set()
            raise RuntimeError("blocker HPO v2 lease ownership changed")

    def __exit__(self, *_):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=max(2.0, self.lease_seconds / 3.0 + 1.0))
        try:
            current = json.loads(self.path.read_text(encoding="utf-8"))
            if current.get("token") == self.token:
                self.path.unlink(missing_ok=True)
        except Exception:
            pass


class LiveStatus:
    """Thread-safe atomic status for the UI and external watchdogs."""

    def __init__(self, path, *, config_sha256, dataset_sha256, stage, jobs):
        self.path = Path(path).resolve()
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.terminal = False
        self.running_ordinals = {f"{target}/{family}": set() for target, family in jobs}
        now = self._now()
        self.document = {
            "schema_version": LIVE_STATUS_SCHEMA,
            "phase": "starting",
            "config_sha256": config_sha256,
            "dataset_sha256": dataset_sha256,
            "selected_cumulative_trials_per_job": int(stage),
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at": now,
            "heartbeat_at": now,
            "activity_at": now,
            "trials": {
                "total": len(jobs) * int(stage),
                "complete": 0,
                "running": 0,
                "failed": 0,
            },
            "jobs": {
                f"{target}/{family}": {
                    "target": target,
                    "family": family,
                    "total": int(stage),
                    "complete": 0,
                    "running": 0,
                    "failed": 0,
                    "status": "pending",
                    "last_ordinal": None,
                }
                for target, family in jobs
            },
            "result": None,
            "error": None,
            "production_eligible": False,
            "production_model_eligible": False,
            "fea_submission_approved": False,
            "promotion_approved": False,
        }
        self._write(activity=True)
        atexit.register(self._atexit)

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat()

    def _write(self, *, activity=False):
        with self.lock:
            now = self._now()
            self.document["heartbeat_at"] = now
            if activity:
                self.document["activity_at"] = now
            tune_optuna._atomic_json(self.document, self.path)

    def start(self):
        if self.thread is not None:
            return

        def heartbeat():
            while not self.stop_event.wait(10.0):
                self._write(activity=False)

        self.thread = threading.Thread(
            target=heartbeat,
            name="mft-blocker-hpo-v2-status-heartbeat",
            daemon=True,
        )
        self.thread.start()

    def _recount(self):
        jobs = self.document["jobs"].values()
        self.document["trials"].update({
            "complete": sum(item["complete"] for item in jobs),
            "running": sum(item["running"] for item in jobs),
            "failed": sum(item["failed"] for item in jobs),
        })

    def preflight_complete(self, preflight):
        with self.lock:
            self.document["phase"] = "preflight_complete"
            self.document["preflight"] = {
                "ready": preflight.get("ready"),
                "strict_full_rows": preflight.get("strict_full_rows"),
                "job_count": preflight.get("job_count"),
                "planned_model_fits_from_clean_studies": preflight.get(
                    "planned_model_fits_from_clean_studies"
                ),
                "implementation_sha256": (
                    preflight.get("implementation") or {}
                ).get("sha256"),
            }
            self._write(activity=True)

    def apply_preaudit(self, preaudit):
        with self.lock:
            for item in preaudit.get("audited", []):
                key = f"{item['target']}/{item['family']}"
                job = self.document["jobs"][key]
                job["complete"] = int(item["completed"])
                job["status"] = "resuming" if item["completed"] else "pending"
            self.document["phase"] = "optimizing"
            self.document["study_preaudit"] = preaudit
            self._recount()
            self._write(activity=True)

    def trial_started(self, target, family, ordinal):
        key = f"{target}/{family}"
        with self.lock:
            if ordinal in self.running_ordinals[key]:
                return
            self.running_ordinals[key].add(int(ordinal))
            job = self.document["jobs"][key]
            job["running"] = len(self.running_ordinals[key])
            job["status"] = "running"
            job["last_ordinal"] = int(ordinal)
            self._recount()
            self._write(activity=True)

    def trial_finished(self, target, family, ordinal, state):
        key = f"{target}/{family}"
        with self.lock:
            self.running_ordinals[key].discard(int(ordinal))
            job = self.document["jobs"][key]
            job["running"] = len(self.running_ordinals[key])
            if state == "COMPLETE":
                job["complete"] += 1
            else:
                job["failed"] += 1
                job["status"] = "failed"
            job["last_ordinal"] = int(ordinal)
            self._recount()
            self._write(activity=True)

    def job_completed(self, target, family):
        key = f"{target}/{family}"
        with self.lock:
            self.document["jobs"][key]["status"] = "complete"
            self._write(activity=True)

    def terminal_success(self, result):
        with self.lock:
            self.document["phase"] = "succeeded"
            self.document["result"] = {
                "generation_id": result.get("generation_id"),
                "generation_path": result.get("generation_path"),
                "params_path": result.get("params_path"),
                "receipt_path": result.get("receipt_path"),
                "receipt_sha256": result.get("receipt_sha256"),
                "receipt_canonical_sha256": result.get(
                    "receipt_canonical_sha256"
                ),
            }
            self._write(activity=True)
            self.terminal = True
        self.stop()

    def terminal_failure(self, error):
        with self.lock:
            if self.terminal:
                return
            for key, ordinals in self.running_ordinals.items():
                self.document["jobs"][key]["running"] = 0
                ordinals.clear()
            self._recount()
            self.document["phase"] = "failed"
            self.document["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            self._write(activity=True)
            self.terminal = True
        self.stop()

    def close_preflight(self):
        with self.lock:
            self._write(activity=True)
            self.terminal = True
        self.stop()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None
        try:
            atexit.unregister(self._atexit)
        except Exception:
            pass

    def _atexit(self):
        if not self.terminal:
            try:
                self.terminal_failure(
                    RuntimeError("process exited without a terminal v2 status")
                )
            except Exception:
                pass


def require_v2_only_study_root(root):
    """Reject physical co-location with the exploratory v1 SQLite studies."""

    root = Path(root).resolve()
    if not root.exists():
        return
    legacy = sorted(
        path.name
        for path in root.glob("*.sqlite3")
        if not path.name.startswith("v2__")
    )
    if legacy:
        raise RuntimeError(
            "v2 study root contains exploratory/unknown SQLite studies: "
            f"{legacy}"
        )


def require_local_sqlite_root(root):
    """Keep SQLite journals off UNC/mapped-network filesystems on Windows."""

    resolved = Path(root).resolve()
    if os.name != "nt":
        return
    if str(resolved).startswith("\\\\"):
        raise RuntimeError("v2 SQLite study root must be on a local filesystem")
    import ctypes

    anchor = resolved.anchor
    drive_type = ctypes.windll.kernel32.GetDriveTypeW(str(anchor))
    # DRIVE_FIXED=3.  Removable media is also rejected because an interrupted
    # detach would leave un-auditable RUNNING trials.
    if drive_type != 3:
        raise RuntimeError(
            "v2 SQLite study root must be a fixed local drive: "
            f"path={resolved}, drive_type={drive_type}"
        )


def objective_contract(
    target, family, limits, source, implementation, holdout, model_threads,
):
    if int(model_threads) != 1:
        raise RuntimeError("v2 objective model_threads contract must be one")
    document = {
        "schema_version": OBJECTIVE_SCHEMA,
        "name": OBJECTIVE_NAME,
        "target": target,
        "family": family,
        "dataset_sha256": source["dataset_sha256"],
        "quality_thresholds_sha256": source["quality_thresholds_sha256"],
        "minimum_interval_coverage": source["thresholds"][
            "minimum_interval_coverage"
        ],
        "target_limits": dict(limits),
        "target_config_sha256": implementation["target_configs"][target]["sha256"],
        "feature_schema_sha256": implementation["feature_schema"]["sha256"],
        "implementation_sha256": implementation["sha256"],
        "search_space_sha256": implementation["search_space"]["sha256"],
        "runtime": implementation["runtime"],
        "model_threads": int(model_threads),
        "forced_thread_environment": dict(FORCED_THREAD_ENVIRONMENT),
        "threadpoolctl_runtime_limit": 1,
        "outer_acceptance_holdout_sha256": holdout["sha256"],
        "outer_acceptance_holdout_rows": holdout["evaluation_row_count"],
        "outer_acceptance_policy": (
            "exact final SEED42 evaluation rows excluded from every HPO fit"
        ),
        "inner_metric_implementation": "train_models.train_target",
        "proposal_contract": PROPOSAL_SCHEMA,
        "sampler_state_continuity": (
            "not_applicable_explicit_per_ordinal_deterministic_proposals"
        ),
        "final_acceptance": (
            "complete_21_target_four_family_retrain_on_full_dataset_then_"
            "evaluate_once_on_sealed_outer_holdout"
        ),
    }
    return {**document, "sha256": _canonical_sha256(document)}


def _static_study_attrs(config_sha256, contract, implementation, holdout):
    return {
        "study_schema": STUDY_SCHEMA,
        "config_sha256": config_sha256,
        "objective_contract_sha256": contract["sha256"],
        "implementation_sha256": implementation["sha256"],
        "tune_blockers_v2_sha256": implementation["files"][
            "tune_blockers_v2.py"
        ],
        "train_models_sha256": implementation["files"]["train_models.py"],
        "tune_optuna_sha256": implementation["files"]["tune_optuna.py"],
        "checkpoint_train_sha256": implementation["files"][
            "checkpoint_train.py"
        ],
        "feature_schema_sha256": implementation["feature_schema"]["sha256"],
        "target_config_sha256": contract["target_config_sha256"],
        "search_space_sha256": implementation["search_space"]["sha256"],
        "runtime_sha256": _canonical_sha256(implementation["runtime"]),
        "model_threads": contract["model_threads"],
        "forced_thread_environment_sha256": _canonical_sha256(
            contract["forced_thread_environment"]
        ),
        "holdout_sha256": holdout["sha256"],
        "holdout_row_count": int(holdout["evaluation_row_count"]),
        "proposal_contract": PROPOSAL_SCHEMA,
    }


def _trial_state_name(trial):
    return getattr(trial.state, "name", str(trial.state)).upper()


def _runtime_params_for_proposal(family, proposal, model_threads):
    params = {
        **proposal["params"],
        **SEARCH_SPACES[family]["constants"],
    }
    return tune_optuna.model_params_with_thread_budget(
        family, params, model_threads
    )


def seal_trial_evidence(
    *, target, family, proposal, search_params, runtime_params, contract,
    metrics,
):
    gate = blocker_v1.quality_gate_objective(
        metrics,
        contract["target_limits"],
        contract["minimum_interval_coverage"],
    )
    document = {
        "schema_version": TRIAL_EVIDENCE_SCHEMA,
        "target": target,
        "family": family,
        "proposal_contract": PROPOSAL_SCHEMA,
        "proposal_ordinal": proposal["ordinal"],
        "proposal_salt": proposal["salt"],
        "proposal_sha256": proposal["sha256"],
        "search_params": dict(search_params),
        "search_params_sha256": _canonical_sha256(search_params),
        "runtime_params": dict(runtime_params),
        "runtime_params_sha256": _canonical_sha256(runtime_params),
        "objective_contract_sha256": contract["sha256"],
        "model_threads": contract["model_threads"],
        "forced_thread_environment": dict(FORCED_THREAD_ENVIRONMENT),
        "quality_gate": gate,
    }
    return {**document, "sha256": _canonical_sha256(document)}


def validate_completed_trial_evidence(
    trial, proposal, *, target, family, contract,
):
    evidence = trial.user_attrs.get("quality_gate_evidence")
    if not isinstance(evidence, dict):
        raise RuntimeError("completed v2 trial has no gate evidence")
    fingerprint = evidence.get("sha256")
    unsigned = {key: value for key, value in evidence.items() if key != "sha256"}
    if (
        not isinstance(fingerprint, str)
        or fingerprint != _canonical_sha256(unsigned)
        or trial.user_attrs.get("quality_gate_evidence_sha256") != fingerprint
    ):
        raise RuntimeError("completed v2 trial evidence fingerprint mismatch")
    expected_runtime = _runtime_params_for_proposal(
        family, proposal, contract["model_threads"]
    )
    exact = {
        "schema_version": TRIAL_EVIDENCE_SCHEMA,
        "target": target,
        "family": family,
        "proposal_contract": PROPOSAL_SCHEMA,
        "proposal_ordinal": proposal["ordinal"],
        "proposal_salt": proposal["salt"],
        "proposal_sha256": proposal["sha256"],
        "search_params": proposal["params"],
        "search_params_sha256": proposal["sha256"],
        "runtime_params": expected_runtime,
        "runtime_params_sha256": _canonical_sha256(expected_runtime),
        "objective_contract_sha256": contract["sha256"],
        "model_threads": contract["model_threads"],
        "forced_thread_environment": FORCED_THREAD_ENVIRONMENT,
    }
    for key, expected in exact.items():
        if evidence.get(key) != expected:
            raise RuntimeError(f"completed v2 trial evidence mismatch: {key}")
    gate = evidence.get("quality_gate")
    if not isinstance(gate, dict) or not isinstance(gate.get("metrics"), dict):
        raise RuntimeError("completed v2 trial gate evidence is incomplete")
    recomputed = blocker_v1.quality_gate_objective(
        gate["metrics"],
        contract["target_limits"],
        contract["minimum_interval_coverage"],
    )
    if _canonical_sha256(gate) != _canonical_sha256(recomputed):
        raise RuntimeError("completed v2 trial gate score cannot be reproduced")
    value = getattr(trial, "value", None)
    score = recomputed.get("score")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) != float(score)
    ):
        raise RuntimeError("completed v2 trial value differs from gate score")
    return evidence


def validate_study_inventory(
    study, expected_attrs, proposals, *, target, family, contract,
):
    trials = list(study.get_trials(deepcopy=False))
    attrs = dict(study.user_attrs)
    if trials:
        for key, expected in expected_attrs.items():
            if key not in attrs or attrs[key] != expected:
                raise RuntimeError(
                    f"existing v2 study provenance mismatch: {key}"
                )
    elif attrs:
        for key, expected in expected_attrs.items():
            if key in attrs and attrs[key] != expected:
                raise RuntimeError(f"empty v2 study provenance mismatch: {key}")
    history = attrs.get("stage_history", [])
    if not isinstance(history, list):
        raise RuntimeError("v2 study stage history is invalid")

    seen_ordinals = set()
    seen_hashes = set()
    completed_ordinals = []
    waiting_ordinals = []
    for trial in trials:
        state = _trial_state_name(trial)
        if state not in {"COMPLETE", "WAITING"}:
            raise RuntimeError(
                "v2 study contains non-resumable trial state; quarantine the "
                "root per docs/HPO_V2_STALE_RUN_RECOVERY.md: "
                f"{state}"
            )
        ordinal = trial.user_attrs.get("proposal_ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int):
            raise RuntimeError("v2 trial proposal ordinal is missing")
        if ordinal < 0 or ordinal >= len(proposals) or ordinal in seen_ordinals:
            raise RuntimeError("v2 trial proposal ordinal is invalid or duplicate")
        if getattr(trial, "number", None) != ordinal:
            raise RuntimeError("v2 trial number differs from proposal ordinal")
        expected = proposals[ordinal]
        if trial.user_attrs.get("trial_contract_sha256") != expected_attrs[
            "objective_contract_sha256"
        ]:
            raise RuntimeError("v2 trial objective contract mismatch")
        if trial.user_attrs.get("proposal_sha256") != expected["sha256"]:
            raise RuntimeError("v2 trial proposal fingerprint mismatch")
        if trial.user_attrs.get("proposal_contract") != PROPOSAL_SCHEMA:
            raise RuntimeError("v2 trial proposal contract mismatch")
        if trial.user_attrs.get("proposal_salt") != expected["salt"]:
            raise RuntimeError("v2 trial proposal salt mismatch")
        invocation_id = trial.user_attrs.get("invocation_id")
        if not isinstance(invocation_id, str) or not invocation_id:
            raise RuntimeError("v2 trial invocation identity is missing")
        if expected["sha256"] in seen_hashes:
            raise RuntimeError("v2 study contains duplicate proposal parameters")
        actual_params = (
            dict(trial.params)
            if state == "COMPLETE"
            else dict((trial.system_attrs or {}).get("fixed_params") or {})
        )
        if (
            not actual_params
            or _canonical_sha256(actual_params) != expected["sha256"]
        ):
            raise RuntimeError("v2 stored proposal parameters drifted")
        if state == "COMPLETE":
            validate_completed_trial_evidence(
                trial,
                expected,
                target=target,
                family=family,
                contract=contract,
            )
        seen_ordinals.add(ordinal)
        seen_hashes.add(expected["sha256"])
        (completed_ordinals if state == "COMPLETE" else waiting_ordinals).append(
            ordinal
        )
    completed_ordinals.sort()
    waiting_ordinals.sort()
    if completed_ordinals != list(range(len(completed_ordinals))):
        raise RuntimeError("v2 completed proposal ordinals are not a prefix")
    if waiting_ordinals and waiting_ordinals != list(
        range(len(completed_ordinals), len(completed_ordinals) + len(waiting_ordinals))
    ):
        raise RuntimeError("v2 waiting proposal ordinals are not contiguous")
    return {
        "trials": trials,
        "completed": len(completed_ordinals),
        "waiting": len(waiting_ordinals),
        "history": history,
    }


def build_study_spec(
    target, family, requested_total_trials, limits, source, implementation,
    holdout, *, model_threads, study_root, config_sha256,
):
    contract = objective_contract(
        target,
        family,
        limits,
        source,
        implementation,
        holdout,
        model_threads,
    )
    identity = _canonical_sha256({
        "study_schema": STUDY_SCHEMA,
        "config_sha256": config_sha256,
        "objective_contract_sha256": contract["sha256"],
        "target": target,
        "family": family,
    })
    proposals = deterministic_proposal_sequence(
        identity, family, requested_total_trials
    )
    return {
        "target": target,
        "family": family,
        "contract": contract,
        "identity": identity,
        "proposals": proposals,
        "study_name": f"mft-blocker-v2-{identity[:24]}",
        "study_path": Path(study_root).resolve() / (
            f"v2__{target}__{family}__{identity[:16]}.sqlite3"
        ),
        "sampler_seed": int(identity[:8], 16),
        "expected_attrs": _static_study_attrs(
            config_sha256, contract, implementation, holdout
        ),
    }


def preaudit_existing_studies(study_root, specifications):
    """Read and validate every existing expected DB before any optimization."""

    import optuna

    root = Path(study_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    expected_paths = {specification["study_path"] for specification in specifications}
    if len(expected_paths) != len(specifications):
        raise RuntimeError("v2 expected study path inventory contains duplicates")
    existing_paths = set(root.glob("*.sqlite3"))
    unexpected = sorted(str(path) for path in existing_paths - expected_paths)
    if unexpected:
        raise RuntimeError(
            "v2 study root contains a foreign or stale SQLite database: "
            f"{unexpected}"
        )

    audited = []
    for specification in specifications:
        study_path = specification["study_path"]
        if study_path not in existing_paths:
            continue
        storage = optuna.storages.RDBStorage(
            url=(
                f"sqlite:///file:{study_path.as_posix()}?mode=ro&uri=true"
            ),
            engine_kwargs={"connect_args": {"uri": True}},
        )
        try:
            study = optuna.load_study(
                study_name=specification["study_name"],
                storage=storage,
                sampler=optuna.samplers.RandomSampler(
                    seed=specification["sampler_seed"]
                ),
            )
            inventory = validate_study_inventory(
                study,
                specification["expected_attrs"],
                specification["proposals"],
                target=specification["target"],
                family=specification["family"],
                contract=specification["contract"],
            )
            audited.append({
                "study_path": str(study_path),
                "target": specification["target"],
                "family": specification["family"],
                "completed": inventory["completed"],
                "waiting": inventory["waiting"],
            })
        finally:
            storage.remove_session()
            storage.engine.dispose()
    return {
        "expected_study_count": len(specifications),
        "existing_study_count": len(existing_paths),
        "audited": audited,
    }


def tune_one(
    target, family, requested_total_trials, frame, features, limits, source,
    implementation, holdout, *, model_threads, sample_weight_column,
    minimum_eligible_rows, study_root, config_sha256, stop_event, lease,
    rss_guard, invocation_id, invocation_resources=None, study_spec=None,
    live_status=None,
):
    import optuna

    specification = study_spec or build_study_spec(
        target,
        family,
        requested_total_trials,
        limits,
        source,
        implementation,
        holdout,
        model_threads=model_threads,
        study_root=study_root,
        config_sha256=config_sha256,
    )
    if specification["target"] != target or specification["family"] != family:
        raise RuntimeError("v2 study specification target/family mismatch")
    contract = specification["contract"]
    identity = specification["identity"]
    proposals = specification["proposals"]
    study_name = specification["study_name"]
    study_path = specification["study_path"]
    study_path.parent.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{study_path.as_posix()}"
    )
    sampler_seed = specification["sampler_seed"]
    expected_attrs = specification["expected_attrs"]

    def objective(trial):
        if stop_event.is_set():
            raise RuntimeError("v2 run cancellation requested")
        lease.ensure_owned()
        rss_guard.check(f"before:{target}:{family}:trial-{trial.number}")
        ordinal = trial.user_attrs.get("proposal_ordinal")
        if not isinstance(ordinal, int) or ordinal >= len(proposals):
            raise RuntimeError("v2 objective received an unqueued proposal")
        expected = proposals[ordinal]
        if live_status is not None:
            live_status.trial_started(target, family, ordinal)
        if trial.user_attrs.get("proposal_sha256") != expected["sha256"]:
            raise RuntimeError("v2 queued proposal fingerprint mismatch")
        search_params = _suggest_fixed_proposal(trial, family)
        if _canonical_sha256(search_params) != expected["sha256"]:
            raise RuntimeError("v2 sampler escaped the deterministic proposal")
        runtime_params = _runtime_params_for_proposal(
            family, expected, model_threads
        )
        with threadpool_limits(limits=1):
            bundle, metrics = train_models.train_target(
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
        lease.ensure_owned()
        rss_guard.check(f"after:{target}:{family}:trial-{trial.number}")
        if stop_event.is_set():
            raise RuntimeError("v2 run cancellation requested")
        evidence = seal_trial_evidence(
            target=target,
            family=family,
            proposal=expected,
            search_params=search_params,
            runtime_params=runtime_params,
            contract=contract,
            metrics=metrics,
        )
        trial.set_user_attr("quality_gate_evidence", evidence)
        trial.set_user_attr("quality_gate_evidence_sha256", evidence["sha256"])
        score = evidence["quality_gate"]["score"]
        trial.report(score, 0)
        return score

    try:
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.RandomSampler(seed=sampler_seed),
            pruner=optuna.pruners.NopPruner(),
            storage=storage,
            study_name=study_name,
            load_if_exists=True,
        )
        inventory = validate_study_inventory(
            study,
            expected_attrs,
            proposals,
            target=target,
            family=family,
            contract=contract,
        )
        if not inventory["trials"]:
            for key, value in expected_attrs.items():
                study.set_user_attr(key, value)
        # Revalidate after initialization; any existing trial now requires every
        # static attribute to be exact.
        inventory = validate_study_inventory(
            study,
            expected_attrs,
            proposals,
            target=target,
            family=family,
            contract=contract,
        )
        completed_before = inventory["completed"]
        if completed_before > requested_total_trials:
            raise RuntimeError("v2 study is beyond the requested cumulative stage")
        already_enqueued = completed_before + inventory["waiting"]
        if already_enqueued > requested_total_trials:
            raise RuntimeError("v2 study has proposals beyond the requested stage")
        for proposal in proposals[already_enqueued:requested_total_trials]:
            study.enqueue_trial(
                proposal["params"],
                user_attrs={
                    "trial_contract_sha256": contract["sha256"],
                    "proposal_contract": PROPOSAL_SCHEMA,
                    "proposal_ordinal": proposal["ordinal"],
                    "proposal_salt": proposal["salt"],
                    "proposal_sha256": proposal["sha256"],
                    "invocation_id": invocation_id,
                },
                skip_if_exists=False,
            )
        remaining = requested_total_trials - completed_before
        if remaining:
            callbacks = None
            if live_status is not None:
                def status_callback(_study, frozen_trial):
                    ordinal = frozen_trial.user_attrs.get("proposal_ordinal")
                    if isinstance(ordinal, int):
                        live_status.trial_finished(
                            target,
                            family,
                            ordinal,
                            _trial_state_name(frozen_trial),
                        )

                callbacks = [status_callback]
            study.optimize(
                objective,
                n_trials=remaining,
                show_progress_bar=False,
                catch=(),
                callbacks=callbacks,
            )
        inventory_after = validate_study_inventory(
            study,
            expected_attrs,
            proposals,
            target=target,
            family=family,
            contract=contract,
        )
        if (
            inventory_after["completed"] != requested_total_trials
            or inventory_after["waiting"] != 0
        ):
            raise RuntimeError(
                f"v2 stage incomplete: {inventory_after['completed']}/"
                f"{requested_total_trials}, waiting={inventory_after['waiting']}"
            )
        stage_record = {
            "schema_version": STAGE_HISTORY_SCHEMA,
            "invocation_id": invocation_id,
            "requested_cumulative_trials": requested_total_trials,
            "completed_before": completed_before,
            "completed_after": inventory_after["completed"],
            "proposal_ordinals_enqueued_this_invocation": list(
                range(already_enqueued, requested_total_trials)
            ),
            "proposal_ordinals_completed_this_invocation": list(
                range(completed_before, requested_total_trials)
            ),
            "proposal_contract": PROPOSAL_SCHEMA,
            "sampler": "RandomSampler fallback; all trials explicitly enqueued",
            "sampler_seed": sampler_seed,
            "sampler_state_continuity": (
                "not_used; proposals are deterministic by absolute ordinal"
            ),
            "invocation_resources": dict(invocation_resources or {}),
        }
        history = list(inventory_after["history"])
        if completed_before < requested_total_trials:
            history.append(stage_record)
            study.set_user_attr("stage_history", history)
        evidence = study.best_trial.user_attrs.get("quality_gate_evidence")
        if not isinstance(evidence, dict):
            raise RuntimeError("best v2 trial has no quality evidence")
        if not math.isfinite(float(study.best_value)):
            raise RuntimeError("best v2 trial objective is nonfinite")
        return {
            "params": study.best_params,
            "quality_gate_objective_score": float(study.best_value),
            "quality_gate_evidence": evidence["quality_gate"],
            "trial_evidence": evidence,
            "objective_contract": contract,
            "study": {
                "study_name": study_name,
                "study_storage_path": str(study_path),
                "study_identity_sha256": identity,
                "requested_total_trials": requested_total_trials,
                "completed_trials_before": completed_before,
                "completed_trials_after": requested_total_trials,
                "completed_trials_added": requested_total_trials - completed_before,
                "stage_history": history,
            },
        }
    finally:
        storage.remove_session()
        storage.engine.dispose()


def run_blocker_jobs(
    jobs, requested_total_trials, frame, features, source, implementation,
    holdouts, *, model_threads, job_workers, max_model_thread_budget,
    sample_weight_column, minimum_eligible_rows, study_root, config_sha256,
    lease, rss_guard, invocation_id, invocation_resources, live_status=None,
):
    workers = min(int(job_workers), max(1, len(jobs)))
    if workers < 1 or int(model_threads) < 1:
        raise ValueError("v2 worker and model thread budgets must be positive")
    if workers * int(model_threads) > int(max_model_thread_budget):
        raise ValueError("v2 model thread budget is exceeded")
    stop_event = threading.Event()
    specifications = []
    for target, family in jobs:
        specifications.append(build_study_spec(
            target,
            family,
            requested_total_trials,
            source["thresholds"]["targets"][target],
            source,
            implementation,
            holdouts[target],
            model_threads=model_threads,
            study_root=study_root,
            config_sha256=config_sha256,
        ))
    preaudit = preaudit_existing_studies(study_root, specifications)
    if live_status is not None:
        live_status.apply_preaudit(preaudit)

    def execute(index, target, family):
        if stop_event.is_set():
            raise RuntimeError("v2 run cancellation requested")
        hpo_frame = frame_without_holdout(frame, holdouts[target])
        limits = source["thresholds"]["targets"][target]
        result = tune_one(
            target, family, requested_total_trials, hpo_frame, features, limits,
            source, implementation, holdouts[target],
            model_threads=model_threads,
            sample_weight_column=sample_weight_column,
            minimum_eligible_rows=minimum_eligible_rows,
            study_root=study_root,
            config_sha256=config_sha256,
            stop_event=stop_event,
            lease=lease,
            rss_guard=rss_guard,
            invocation_id=invocation_id,
            invocation_resources=invocation_resources,
            study_spec=specifications[index],
            live_status=live_status,
        )
        if live_status is not None:
            live_status.job_completed(target, family)
        return index, target, family, result

    completed = [None] * len(jobs)
    rss_guard.start(stop_event)
    executor = None
    futures = []
    try:
        executor = ThreadPoolExecutor(max_workers=workers)
        futures = [
            executor.submit(execute, index, target, family)
            for index, (target, family) in enumerate(jobs)
        ]
        for future in as_completed(futures):
            item = future.result()
            completed[item[0]] = item
    except BaseException:
        stop_event.set()
        for future in futures:
            future.cancel()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    finally:
        rss_guard.stop()

    params = {}
    results = []
    for item in completed:
        if item is None:
            raise RuntimeError("v2 blocker tuning inventory is incomplete")
        _, target, family, result = item
        params.setdefault(family, {})[target] = {
            "params": result["params"],
            "quality_gate_objective_score": result[
                "quality_gate_objective_score"
            ],
            "quality_gate_objective_contract_sha256": result[
                "objective_contract"
            ]["sha256"],
            "completed_trials": result["study"]["completed_trials_after"],
        }
        results.append({
            "target": target,
            "family": family,
            **{key: value for key, value in result.items() if key != "params"},
        })
    return params, results, workers, preaudit


def _publish_generation(artifact_root, params, holdouts, metadata, result_json):
    from pipeline.artifacts import GenerationStore

    params_payload = {
        train_models.V2_HPO_PARAMS_MARKER: {
            "schema_version": train_models.V2_HPO_PARAMS_SCHEMA,
            "blockers": list(train_models.V2_BLOCKER_TARGETS),
            "families": list(train_models.V2_HPO_FAMILIES),
        },
        **params,
    }
    with tempfile.TemporaryDirectory(prefix="mft-blocker-hpo-v2-") as directory:
        directory = Path(directory)
        params_path = directory / "params.json"
        holdouts_path = directory / "outer_acceptance_holdouts.json"
        receipt_path = directory / "hpo_receipt.json"
        tune_optuna._atomic_json(params_payload, params_path)
        tune_optuna._atomic_json(holdouts, holdouts_path)
        receipt_document = {
            "schema_version": RESULT_SCHEMA,
            "metadata": metadata,
            "params_sha256": _sha256(params_path),
            "params_canonical_sha256": _canonical_sha256(params_payload),
            "holdouts_sha256": _canonical_sha256(holdouts),
            "holdouts_file_sha256": _sha256(holdouts_path),
        }
        receipt = {
            **receipt_document,
            "sha256": _canonical_sha256(receipt_document),
        }
        tune_optuna._atomic_json(receipt, receipt_path)
        generation = GenerationStore(artifact_root).publish_files(
            "tuning",
            {
                "params.json": params_path,
                "outer_acceptance_holdouts.json": holdouts_path,
                "hpo_receipt.json": receipt_path,
            },
            metadata=metadata,
            parents=[
                f"dataset:{metadata['dataset_sha256']}",
                f"quality:{metadata['source_quality_status_sha256']}",
            ],
        )
    result = {
        "schema_version": RESULT_SCHEMA,
        "generation_id": generation.generation_id,
        "generation_path": str(generation.path),
        "params_path": str(generation.path / "params.json"),
        "outer_acceptance_holdouts_path": str(
            generation.path / "outer_acceptance_holdouts.json"
        ),
        "receipt_path": str(generation.path / "hpo_receipt.json"),
        "receipt_sha256": _sha256(generation.path / "hpo_receipt.json"),
        "receipt_canonical_sha256": receipt["sha256"],
    }
    tune_optuna._atomic_json(result, result_json)
    return result


def validate_execution_resources(
    config, *, model_threads, job_workers, max_model_thread_budget,
    max_rss_gb, job_count, require_rss,
):
    expected_threads = config["objective_execution_contract"]["model_threads"]
    if int(model_threads) != expected_threads:
        raise ValueError(
            f"v2 model_threads must match authenticated value {expected_threads}"
        )
    if int(job_workers) < 1 or int(job_workers) > config["maximum_job_workers"]:
        raise ValueError("v2 job_workers exceeds the authenticated ceiling")
    workers = min(int(job_workers), max(1, int(job_count)))
    budget = int(max_model_thread_budget)
    if (
        budget < workers * expected_threads
        or budget > config["maximum_model_thread_budget"]
    ):
        raise ValueError("v2 effective model thread budget is invalid")
    if require_rss and max_rss_gb is None:
        raise ValueError("v2 requires an explicit max_rss_gb")
    if max_rss_gb is not None and (
        not math.isfinite(float(max_rss_gb))
        or float(max_rss_gb) <= 0.0
        or float(max_rss_gb) > float(config["maximum_process_rss_gb"])
    ):
        raise ValueError("v2 max_rss_gb exceeds the authenticated ceiling")
    if {
        key: os.environ.get(key) for key in FORCED_THREAD_ENVIRONMENT
    } != FORCED_THREAD_ENVIRONMENT:
        raise ValueError("v2 forced thread environment mismatch")
    return {
        "model_threads": expected_threads,
        "job_workers_requested": int(job_workers),
        "job_workers_effective": workers,
        "max_model_thread_budget": budget,
        "max_rss_gb": float(max_rss_gb) if max_rss_gb is not None else None,
        "forced_thread_environment": dict(FORCED_THREAD_ENVIRONMENT),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--quality-status", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--result-json", default=None)
    parser.add_argument("--study-root", default=None)
    parser.add_argument("--stage-trials", type=int, default=None)
    parser.add_argument("--model-threads", type=int, default=1)
    parser.add_argument("--job-workers", type=int, default=1)
    parser.add_argument("--max-model-thread-budget", type=int, default=None)
    parser.add_argument("--max-rss-gb", type=float, default=None)
    parser.add_argument("--status-json", default=None)
    parser.add_argument("--lock-lease-seconds", type=float, default=120.0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    config = load_blocker_config(config_path)
    selected_stage = (
        args.stage_trials
        if args.stage_trials is not None else config["trial_stages"][0]
    )
    if selected_stage not in config["trial_stages"]:
        parser.error("stage-trials must be one configured v2 cumulative stage")
    maximum_budget = (
        args.max_model_thread_budget
        if args.max_model_thread_budget is not None else args.model_threads
    )
    jobs = [
        (target, family)
        for target in config["blockers"]
        for family in config["families"]
    ]
    if not args.preflight_only:
        if not all((args.artifact_root, args.result_json, args.study_root)):
            parser.error(
                "v2 requires explicit artifact-root, result-json, and study-root"
            )
        if not args.status_json:
            parser.error("v2 execution requires an explicit status-json")
    try:
        invocation_resources = validate_execution_resources(
            config,
            model_threads=args.model_threads,
            job_workers=args.job_workers,
            max_model_thread_budget=maximum_budget,
            max_rss_gb=args.max_rss_gb,
            job_count=len(jobs),
            require_rss=not args.preflight_only,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    actual_workers = invocation_resources["job_workers_effective"]
    global _ACTIVE_LIVE_STATUS
    live_status = None
    if args.status_json:
        live_status = LiveStatus(
            os.path.abspath(args.status_json),
            config_sha256=_sha256(config_path),
            dataset_sha256=config["source"]["dataset_sha256"],
            stage=selected_stage,
            jobs=jobs,
        )
        live_status.start()
        _ACTIVE_LIVE_STATUS = live_status

    source = blocker_v1._validate_source_documents(
        config,
        config_path,
        args.dataset,
        args.quality_status,
        args.thresholds,
    )
    cohort = config["cohort"]
    frame, strict_count = tune_optuna._load_strict_dataset(
        source["dataset_path"],
        cohort["solver_revision"].lower(),
        cohort["library_revision"].lower(),
    )
    audit = capacitance_recovery_audit(frame)
    blocker_v1._require_corrected_recovery(
        config, source["quality_status"], audit, strict_count
    )
    features = feature_columns(frame)
    if not features:
        raise RuntimeError("v2 has no design-time features")
    implementation = _runtime_evidence(features)
    holdouts = {
        "schema_version": HOLDOUT_SCHEMA,
        "dataset_sha256": source["dataset_sha256"],
        "targets": {
            target: seal_final_evaluation_holdout(
                frame, target, source["dataset_sha256"]
            )
            for target in config["blockers"]
        },
    }
    holdouts["sha256"] = _canonical_sha256(holdouts)
    eligible_counts = {}
    hpo_counts = {}
    for target in config["blockers"]:
        eligible_counts[target] = len(_eligible_target_frame(frame, target))
        hpo_frame = frame_without_holdout(frame, holdouts["targets"][target])
        hpo_counts[target] = len(_eligible_target_frame(hpo_frame, target))
        removed = eligible_counts[target] - hpo_counts[target]
        expected_removed = holdouts["targets"][target]["evaluation_row_count"]
        if removed != expected_removed:
            raise RuntimeError(
                f"v2 outer holdout exclusion mismatch for {target}: "
                f"{removed}!={expected_removed}"
            )
    insufficient = {
        target: count for target, count in hpo_counts.items()
        if count < config["minimum_eligible_rows_per_target"]
    }
    if insufficient:
        raise RuntimeError(f"v2 nested HPO rows are insufficient: {insufficient}")

    preflight = {
        "schema_version": CONFIG_SCHEMA,
        "ready": True,
        "legacy_stage8_policy": (
            "exploratory_only; never warm-started, enqueued, or resumed"
        ),
        "config_sha256": source["config_sha256"],
        "dataset_sha256": source["dataset_sha256"],
        "strict_full_rows": int(strict_count),
        "quality_status_sha256": source["quality_status_sha256"],
        "quality_thresholds_sha256": source["quality_thresholds_sha256"],
        "capacitance_recovery": audit,
        "implementation": implementation,
        "holdouts": holdouts,
        "eligible_rows_by_target": eligible_counts,
        "hpo_rows_after_outer_holdout_by_target": hpo_counts,
        "selected_cumulative_trials_per_job": selected_stage,
        "job_count": len(jobs),
        "planned_model_fits_from_clean_studies": (
            len(jobs) * selected_stage * FOLDS_PER_TRIAL
        ),
        "model_threads": args.model_threads,
        "job_workers": actual_workers,
        "maximum_configured_job_workers": config["maximum_job_workers"],
        "maximum_model_thread_budget": maximum_budget,
        "max_rss_gb": args.max_rss_gb,
        "invocation_resources": invocation_resources,
        "data_contract_sha256_policy": (
            "omitted; no runtime-verifiable contract artifact was supplied"
        ),
    }
    if live_status is not None:
        live_status.preflight_complete(preflight)
    if args.preflight_only:
        if live_status is not None:
            live_status.close_preflight()
            _ACTIVE_LIVE_STATUS = None
        print(json.dumps({"preflight": preflight}, ensure_ascii=False))
        return

    invocation_id = uuid.uuid4().hex
    study_root = os.path.abspath(args.study_root)
    require_local_sqlite_root(study_root)
    require_v2_only_study_root(study_root)
    rss_guard = RssGuard(args.max_rss_gb)
    lease_identity = {
        "invocation_id": invocation_id,
        "config_sha256": source["config_sha256"],
        "requested_cumulative_trials": selected_stage,
    }
    with RunLease(
        study_root,
        lease_seconds=args.lock_lease_seconds,
        identity=lease_identity,
    ) as lease:
        lease.ensure_owned()
        tuned, results, workers, study_preaudit = run_blocker_jobs(
            jobs,
            selected_stage,
            frame,
            features,
            source,
            implementation,
            holdouts["targets"],
            model_threads=args.model_threads,
            job_workers=args.job_workers,
            max_model_thread_budget=maximum_budget,
            sample_weight_column=config.get("sample_weight_column"),
            minimum_eligible_rows=config["minimum_eligible_rows_per_target"],
            study_root=study_root,
            config_sha256=source["config_sha256"],
            lease=lease,
            rss_guard=rss_guard,
            invocation_id=invocation_id,
            invocation_resources=invocation_resources,
            live_status=live_status,
        )
        lease.ensure_owned()
        current_implementation = _runtime_evidence(features)
        if current_implementation != implementation:
            raise RuntimeError(
                "v2 implementation or runtime changed during the HPO invocation"
            )
        metadata = {
            "tuning_schema_version": TUNING_SCHEMA_VERSION,
            "lane": "production_gate_blocker_hpo_v2_nested_holdout",
            "scope": "authenticated_blockers_only",
            "legacy_stage8_accepted": False,
            "parameter_artifact_eligible": True,
            "production_eligible": False,
            "production_model_eligible": False,
            "fea_submission_approved": False,
            "promotion_approved": False,
            "final_acceptance_required": (
                "complete_21_target_retrain_then_once_only_sealed_outer_gate"
            ),
            "config_sha256": source["config_sha256"],
            "source_quality_status_sha256": source["quality_status_sha256"],
            "source_training_run_id": config["source"]["training_run_id"],
            "source_generation": config["source"]["generation"],
            "dataset_sha256": source["dataset_sha256"],
            "strict_full_rows": int(strict_count),
            "profile_sha256": config["source"]["profile_sha256"],
            "capacitance_recovery": audit,
            "quality_thresholds_sha256": source[
                "quality_thresholds_sha256"
            ],
            "quality_thresholds_file_sha256": source[
                "quality_thresholds_file_sha256"
            ],
            "solver_revision": cohort["solver_revision"].lower(),
            "library_revision": cohort["library_revision"].lower(),
            "data_contract_sha256_policy": (
                "omitted_unverifiable_claim"
            ),
            "implementation": implementation,
            "outer_acceptance_holdouts_sha256": holdouts["sha256"],
            "outer_acceptance_policy": {
                "eligible_after_cumulative_trials": config["trials_per_job"],
                "selected_cumulative_trials": selected_stage,
                "eligible": selected_stage == config["trials_per_job"],
                "single_use_required": True,
                "claim_ledger_schema": (
                    "mft-outer-acceptance-single-use-claim-v1"
                ),
                "pre_final_stage_policy": (
                    "never evaluate sealed outer holdout"
                ),
            },
            "eligible_rows_by_target": eligible_counts,
            "hpo_rows_after_outer_holdout_by_target": hpo_counts,
            "blockers": list(config["blockers"]),
            "families": list(config["families"]),
            "trial_stages": list(config["trial_stages"]),
            "configured_final_trials_per_job": config["trials_per_job"],
            "selected_cumulative_trials_per_job": selected_stage,
            "proposal_contract": PROPOSAL_SCHEMA,
            "sampler_state_continuity": (
                "not_used_explicit_deterministic_per_ordinal_proposals"
            ),
            "model_threads": args.model_threads,
            "job_workers": workers,
            "maximum_configured_job_workers": config["maximum_job_workers"],
            "maximum_total_model_threads": workers * args.model_threads,
            "max_model_thread_budget": maximum_budget,
            "max_rss_gb": args.max_rss_gb,
            "invocation_id": invocation_id,
            "invocation_resources": invocation_resources,
            "study_preaudit": study_preaudit,
            "jobs": results,
        }
        result = _publish_generation(
            os.path.abspath(args.artifact_root),
            tuned,
            holdouts,
            metadata,
            os.path.abspath(args.result_json),
        )
    live_status.terminal_success(result)
    _ACTIVE_LIVE_STATUS = None
    print(json.dumps({"preflight": preflight, "result": result}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if _ACTIVE_LIVE_STATUS is not None:
            _ACTIVE_LIVE_STATUS.terminal_failure(error)
            _ACTIVE_LIVE_STATUS = None
        raise
