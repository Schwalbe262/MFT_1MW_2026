"""Persistent strict-valid checkpoint retraining orchestrator.

Schedule: 500, 1,000, 2,000, 3,000, then every additional 1,000 rows.
The counter is recomputed from the fail-closed quality contract, never raw
parquet length or stored validity flags.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from filelock import FileLock
import pandas as pd


HERE = Path(__file__).resolve().parent
REGRESSION_ROOT = HERE.parent
if str(REGRESSION_ROOT) not in sys.path:
    sys.path.insert(0, str(REGRESSION_ROOT))
DEFAULT_THRESHOLDS = HERE / "model_quality_thresholds.json"
STATE_SCHEMA_VERSION = 2
REGISTRY_PROTOCOL_VERSION = 2


def checkpoint_sequence(valid_count):
    fixed = [500, 1000, 2000, 3000]
    if valid_count >= 4000:
        fixed.extend(range(4000, valid_count + 1, 1000))
    return [threshold for threshold in fixed if threshold <= valid_count]


def due_with_backoff(
    due, state, strict_count, minimum_new_rows=250, backoff_seconds=3600,
    now=None, force_ready=(),
):
    """Defer failed checkpoints until data grows or the retry timer expires."""
    now = now or datetime.now()
    ready, deferred = [], []
    attempts = state.get("attempts", [])
    forced = set()
    for value in force_ready:
        try:
            forced.add(int(value))
        except (TypeError, ValueError, OverflowError):
            continue
    for threshold in due:
        if threshold in forced:
            ready.append(threshold)
            continue
        failures = [
            item for item in attempts
            if isinstance(item, dict)
            and item.get("threshold") == threshold
            and item.get("status") == "failed"
        ]
        if not failures:
            ready.append(threshold)
            continue
        last = failures[-1]
        new_rows = strict_count - int(last.get("strict_full_rows") or 0)
        timestamp = last.get("finished_at") or last.get("started_at")
        try:
            elapsed = (now - datetime.fromisoformat(timestamp)).total_seconds()
        except (TypeError, ValueError):
            elapsed = backoff_seconds
        if new_rows >= minimum_new_rows or elapsed >= backoff_seconds:
            ready.append(threshold)
        else:
            deferred.append({
                "threshold": threshold,
                "new_strict_rows": new_rows,
                "retry_after_seconds": max(0, int(backoff_seconds - elapsed)),
            })
    return ready, deferred


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


def _atomic_parquet(frame, path):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, staged = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp",
        dir=os.path.dirname(path),
    )
    os.close(fd)
    try:
        frame.to_parquet(staged, index=False)
        os.replace(staged, path)
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_state(path):
    if not os.path.isfile(path):
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "completed": [],
            "attempts": [],
        }
    try:
        with open(path, encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "completed": [],
            "attempts": [],
            "recovery": [{
                "time": datetime.now().isoformat(timespec="seconds"),
                "reason": f"corrupt_checkpoint_state:{exc}",
                "source_sha256": _sha256(path) if os.path.isfile(path) else None,
            }],
            "_corrupt_source": path,
        }
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint state must be a JSON object")
    schema = state.get("schema_version", 1)
    if schema == 1:
        legacy = state.get("completed", [])
        state["schema_version"] = STATE_SCHEMA_VERSION
        state["completed"] = []
        if legacy:
            state.setdefault("invalidated_completed", []).extend({
                "threshold": item.get("threshold") if isinstance(item, dict) else None,
                "reason": "schema_v1_completion_has_no_cryptographic_evidence",
                "invalidated_at": datetime.now().isoformat(timespec="seconds"),
                "legacy": item,
            } for item in legacy)
        state.setdefault("attempts", [])
        state["_migrated_from_schema"] = 1
        return state
    if schema != STATE_SCHEMA_VERSION:
        raise RuntimeError("unsupported checkpoint state schema")
    if not isinstance(state.get("completed", []), list):
        raise RuntimeError("checkpoint completed state must be a list")
    if not isinstance(state.get("attempts", []), list):
        raise RuntimeError("checkpoint attempts state must be a list")
    return state


def _save_state(state, path):
    """Persist state atomically, retaining corrupt input for forensic recovery."""
    corrupt_source = state.get("_corrupt_source")
    if corrupt_source and os.path.isfile(corrupt_source):
        digest = _sha256(corrupt_source)[:12]
        backup = os.path.join(
            os.path.dirname(path), f"checkpoint_state.corrupt.{digest}.json"
        )
        if not os.path.exists(backup):
            shutil.copy2(corrupt_source, backup)
    persistent = {
        key: value for key, value in state.items() if not key.startswith("_")
    }
    _atomic_json(persistent, path)
    state.pop("_corrupt_source", None)
    state.pop("_migrated_from_schema", None)


def _ensure_identity(state, identity):
    stored = state.get("identity")
    if stored is None:
        if state.get("completed"):
            state.setdefault("invalidated_completed", []).extend({
                "threshold": item.get("threshold") if isinstance(item, dict) else None,
                "training_run_id": (
                    item.get("training_run_id") if isinstance(item, dict) else None
                ),
                "reason": "completion_has_no_checkpoint_identity",
                "invalidated_at": datetime.now().isoformat(timespec="seconds"),
            } for item in state["completed"])
            state["completed"] = []
        state["identity"] = identity
        return True
    if not isinstance(stored, dict):
        raise RuntimeError("checkpoint runtime identity is invalid")
    common = set(stored) & set(identity)
    mismatched = [key for key in common if stored.get(key) != identity.get(key)]
    if mismatched:
        raise RuntimeError(
            "checkpoint runtime identity changed ("
            + ", ".join(sorted(mismatched))
            + "); use a new output root or archive state"
        )
    if stored != identity:
        changed_keys = sorted(set(stored) ^ set(identity))
        if state.get("completed"):
            state.setdefault("invalidated_completed", []).extend({
                "threshold": item.get("threshold") if isinstance(item, dict) else None,
                "training_run_id": (
                    item.get("training_run_id") if isinstance(item, dict) else None
                ),
                "reason": (
                    "completion_identity_schema_changed:"
                    + ",".join(changed_keys)
                ),
                "invalidated_at": datetime.now().isoformat(timespec="seconds"),
            } for item in state["completed"])
            state["completed"] = []
        state["identity"] = identity
        state.setdefault("recovery", []).append({
            "time": datetime.now().isoformat(timespec="seconds"),
            "reason": "checkpoint_identity_upgraded_fail_closed",
        })
        return True
    return False


def _completion_error(item, registry, expected_identity=None):
    """Return None only when every completion artifact is still coherent."""
    if not isinstance(item, dict):
        return "completion_not_object"
    required = (
        "kind", "threshold", "actual_strict_full_rows", "snapshot",
        "snapshot_sha256", "metrics_result", "metrics_result_sha256",
        "profile_path", "profile_sha256", "thresholds_sha256",
        "activation_minimum_strict_full_rows",
    )
    missing = [key for key in required if item.get(key) in (None, "")]
    if missing:
        return "completion_evidence_missing:" + ",".join(missing)
    try:
        threshold = int(item["threshold"])
    except (TypeError, ValueError, OverflowError):
        return "completion_threshold_invalid"
    if expected_identity:
        for key in (
            "profile_path", "profile_sha256", "thresholds_sha256",
            "activation_minimum_strict_full_rows",
        ):
            if key in expected_identity and item.get(key) != expected_identity.get(key):
                return f"completion_{key}_does_not_match_runtime_identity"
    snapshot = os.path.abspath(item["snapshot"])
    if not os.path.isfile(snapshot):
        return "snapshot_missing"
    if _sha256(snapshot) != item["snapshot_sha256"]:
        return "snapshot_fingerprint_mismatch"
    try:
        actual_rows = int(item["actual_strict_full_rows"])
        activation_minimum = int(item["activation_minimum_strict_full_rows"])
    except (TypeError, ValueError, OverflowError):
        return "completion_row_count_invalid"
    if threshold > actual_rows:
        return "completion_threshold_exceeds_snapshot_rows"
    metrics_path = os.path.abspath(item["metrics_result"])
    if not os.path.isfile(metrics_path):
        return "checkpoint_metrics_missing"
    if _sha256(metrics_path) != item["metrics_result_sha256"]:
        return "checkpoint_metrics_fingerprint_mismatch"
    try:
        with open(metrics_path, encoding="utf-8") as handle:
            metrics = json.load(handle)
    except Exception as exc:
        return f"checkpoint_metrics_invalid:{exc}"
    metric_checks = {
        "checkpoint": threshold,
        "dataset": snapshot,
        "dataset_sha256": item["snapshot_sha256"],
        "profile": os.path.abspath(item["profile_path"]),
        "profile_sha256": item["profile_sha256"],
        "strict_full_rows": actual_rows,
    }
    for key, expected in metric_checks.items():
        if metrics.get(key) != expected:
            return f"checkpoint_metrics_{key}_mismatch"
    if not isinstance(metrics.get("metrics"), list) or not metrics["metrics"]:
        return "checkpoint_metrics_empty"

    parity_path_value = item.get("parity_result")
    parity_hash_value = item.get("parity_result_sha256")
    if bool(parity_path_value) != bool(parity_hash_value):
        return "checkpoint_parity_evidence_incomplete"
    if parity_path_value:
        parity_path = os.path.abspath(parity_path_value)
        expected_parity_path = os.path.splitext(metrics_path)[0] + ".parity.json"
        if parity_path != expected_parity_path:
            return "checkpoint_parity_path_mismatch"
        if not os.path.isfile(parity_path):
            return "checkpoint_parity_missing"
        if _sha256(parity_path) != parity_hash_value:
            return "checkpoint_parity_fingerprint_mismatch"
        try:
            with open(parity_path, encoding="utf-8") as handle:
                parity = json.load(handle)
        except Exception as exc:
            return f"checkpoint_parity_invalid:{exc}"
        parity_checks = {
            "schema_version": 1,
            "artifact_type": "checkpoint_cv_oof_parity",
            "checkpoint": threshold,
            "dataset": snapshot,
            "dataset_sha256": item["snapshot_sha256"],
            "profile": os.path.abspath(item["profile_path"]),
            "profile_sha256": item["profile_sha256"],
            "strict_full_rows": actual_rows,
            "prediction_kind": "out_of_fold",
        }
        for key, expected in parity_checks.items():
            if parity.get(key) != expected:
                return f"checkpoint_parity_{key}_mismatch"
        if not isinstance(parity.get("targets"), dict) or not parity["targets"]:
            return "checkpoint_parity_targets_empty"
    kind = item.get("kind")
    if kind == "metrics_only":
        if threshold >= activation_minimum:
            return "metrics_only_completion_at_activation_threshold"
        return None
    if kind != "accepted_generation":
        return "completion_kind_invalid"
    if threshold < activation_minimum:
        return "accepted_generation_below_activation_threshold"
    generation_required = (
        "training_run_id", "generation", "generation_report_sha256",
        "quality_gate_sha256",
    )
    missing = [
        key for key in generation_required if item.get(key) in (None, "")
    ]
    if missing:
        return "completion_evidence_missing:" + ",".join(missing)
    from train_models import load_generation

    try:
        record = load_generation(
            registry, item["generation"], require_accepted=True
        )
    except Exception as exc:
        return f"generation_invalid:{exc}"
    report = record["report"]
    quality = record["quality"]
    if actual_rows != int(report.get("strict_full_rows", -1)):
        return "completion_generation_row_count_mismatch"
    if threshold > actual_rows:
        return "completion_threshold_exceeds_generation_rows"
    if quality.get("checkpoint") != threshold:
        return "completion_threshold_gate_mismatch"
    checks = {
        "training_run_id": report.get("training_run_id"),
        "generation": record["generation_relative"],
        "generation_report_sha256": record["generation_report_sha256"],
        "quality_gate_sha256": record["quality_gate_sha256"],
        "snapshot_sha256": report.get("dataset_sha256"),
        "profile_sha256": report.get("profile_sha256"),
        "thresholds_sha256": quality.get("thresholds_sha256"),
    }
    for key, expected in checks.items():
        if item.get(key) != expected:
            return f"completion_{key}_mismatch"
    if quality.get("passed") is not True:
        return "completion_quality_not_accepted"
    return None


def reconcile_completed(state, registry, expected_identity=None):
    """Validate state against snapshots and the one accepted active generation."""
    valid = []
    invalid = []
    for item in state.get("completed", []):
        error = _completion_error(item, registry, expected_identity)
        if error:
            invalid.append((item, error))
        else:
            valid.append(item)
    accepted_entries = [
        item for item in valid if item.get("kind") == "accepted_generation"
    ]
    if accepted_entries:
        try:
            from train_models import load_active_generation

            active = load_active_generation(registry)
        except Exception as exc:
            invalid.extend(
                (item, f"active_generation_invalid:{exc}")
                for item in accepted_entries
            )
            accepted_ids = {id(item) for item in accepted_entries}
            valid = [item for item in valid if id(item) not in accepted_ids]
        else:
            active_run = active["report"].get("training_run_id")
            matching = [
                index for index, item in enumerate(valid)
                if item.get("kind") == "accepted_generation"
                and item.get("training_run_id") == active_run
                and item.get("generation") == active["generation_relative"]
            ]
            if not matching:
                report = active["report"]
                quality = active["quality"]
                supersedes = bool(
                    expected_identity
                    and report.get("profile_sha256")
                    == expected_identity.get("profile_sha256")
                    and quality.get("thresholds_sha256")
                    == expected_identity.get("thresholds_sha256")
                    and int(report.get("strict_full_rows", -1))
                    >= max(int(item["threshold"]) for item in accepted_entries)
                )
                if not supersedes:
                    invalid.extend(
                        (item, "active_generation_not_in_checkpoint_state")
                        for item in accepted_entries
                    )
                    accepted_ids = {id(item) for item in accepted_entries}
                    valid = [
                        item for item in valid if id(item) not in accepted_ids
                    ]
            else:
                last_active = matching[-1]
                trailing = [
                    item for item in valid[last_active + 1:]
                    if item.get("kind") == "accepted_generation"
                ]
                invalid.extend(
                    (item, "completion_newer_than_active_generation")
                    for item in trailing
                )
                trailing_ids = {id(item) for item in trailing}
                valid = [item for item in valid if id(item) not in trailing_ids]
    issues = [{
        "threshold": item.get("threshold") if isinstance(item, dict) else None,
        "training_run_id": (
            item.get("training_run_id") if isinstance(item, dict) else None
        ),
        "reason": reason,
    } for item, reason in invalid]
    return valid, issues


def _run(command):
    print("[checkpoint] $ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=REGRESSION_ROOT)
    if result.returncode:
        raise RuntimeError(
            f"checkpoint command failed ({result.returncode}): {command}"
        )


def training_commands(
    snapshot, curve, registry, min_rows, profile, threshold, metrics_result,
    candidate_result=None,
):
    """Build child commands with one already-normalized absolute profile path."""
    if not profile or not os.path.isabs(profile):
        raise ValueError("checkpoint profile must be an absolute path")
    parity_result = os.path.splitext(metrics_result)[0] + ".parity.json"
    commands = [[
            sys.executable,
            str(HERE / "checkpoint_train.py"),
            "--dataset", snapshot,
            "--curve-csv", curve,
            "--profile", profile,
            "--checkpoint", str(threshold),
            "--result-json", metrics_result,
            "--parity-json", parity_result,
        ]]
    if candidate_result:
        commands.append([
            sys.executable,
            str(HERE / "train_models.py"),
            "--dataset", snapshot,
            "--registry", registry,
            "--min-rows", str(min_rows),
            "--profile", profile,
            "--result-json", candidate_result,
        ])
    return commands


def _snapshot_name(output_root, threshold, frame):
    identities = []
    for column in ("project_name", "saved_at", "git_hash"):
        if column in frame.columns:
            identities.extend(frame[column].fillna("").astype(str).tolist())
    digest = hashlib.sha256("\n".join(identities).encode("utf-8")).hexdigest()[:12]
    return os.path.join(
        output_root, "snapshots", f"strict_{threshold:06d}_{digest}.parquet"
    )


def _pinned_training_runs(runtime_root):
    run_ids = set()
    for path in Path(runtime_root, "al_rounds").glob(
        "round_*/model_quality_snapshot.json"
    ):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("training_run_id"):
                run_ids.add(value["training_run_id"])
        except Exception:
            continue
    return run_ids


def inspect_dataset(
    dataset, profile=None, expected_solver_revision=None,
    expected_library_revision=None,
):
    from quality_contract import annotate_validity

    raw = pd.read_parquet(dataset)
    audited = annotate_validity(
        raw, profile,
        expected_solver_revision=expected_solver_revision,
        expected_library_revision=expected_library_revision,
    )
    strict = audited.loc[audited["_strict_valid_full"]].copy()
    reason_counts = Counter()
    for reasons in audited.loc[
        ~audited["_strict_valid_full"], "_strict_invalid_reasons"
    ]:
        reason_counts.update(reason for reason in str(reasons).split(";") if reason)
    return raw, audited, strict, dict(reason_counts.most_common())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", default=str(REGRESSION_ROOT))
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--run-root", default=None,
        help=(
            "identity-scoped checkpoint state and evidence root; shared model "
            "and monitoring artifacts remain under --output-root"
        ),
    )
    parser.add_argument("--profile", default=None)
    parser.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS))
    parser.add_argument("--min-rows", type=int, default=200)
    parser.add_argument("--solver-revision", default=None)
    parser.add_argument("--library-revision", default=None)
    parser.add_argument("--retry-min-new-rows", type=int, default=250)
    parser.add_argument("--retry-backoff-seconds", type=int, default=3600)
    parser.add_argument("--max-checkpoints-per-run", type=int, default=1)
    parser.add_argument(
        "--execute", action="store_true",
        help="write snapshots and train; without this flag the command is read-only",
    )
    args = parser.parse_args()

    if bool(args.solver_revision) != bool(args.library_revision):
        parser.error("solver and library revisions must be pinned together")
    for label, revision in (
        ("solver", args.solver_revision), ("library", args.library_revision),
    ):
        if revision and not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            parser.error(f"{label} revision must be a full 40-character SHA")
    if args.solver_revision:
        args.solver_revision = args.solver_revision.lower()
        args.library_revision = args.library_revision.lower()
    if args.retry_min_new_rows < 0 or args.retry_backoff_seconds < 0:
        parser.error("checkpoint retry limits must be non-negative")

    runtime_root = os.path.abspath(args.runtime_root)
    dataset = os.path.abspath(
        args.dataset
        or os.path.join(runtime_root, "data", "dataset", "train.parquet")
    )
    output_root = os.path.abspath(
        args.output_root or os.path.join(runtime_root, "training")
    )
    run_root = os.path.abspath(args.run_root or output_root)
    state_path = os.path.join(run_root, "checkpoint_state.json")
    from quality_contract import DEFAULT_PROFILE_PATH, load_profile

    args.profile = os.path.abspath(args.profile or DEFAULT_PROFILE_PATH)
    args.thresholds = os.path.abspath(args.thresholds)
    profile_data = load_profile(args.profile)
    profile_sha256 = hashlib.sha256(
        json.dumps(profile_data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    with open(args.thresholds, encoding="utf-8") as handle:
        thresholds = json.load(handle)
    thresholds_sha256 = hashlib.sha256(
        json.dumps(thresholds, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    identity = {
        "dataset": dataset,
        "profile_path": args.profile,
        "profile_sha256": profile_sha256,
        "thresholds_path": args.thresholds,
        "thresholds_sha256": thresholds_sha256,
        "activation_minimum_strict_full_rows": int(
            thresholds["minimum_strict_full_rows"]
        ),
        "quality_contract_sha256": _sha256(
            os.path.join(REGRESSION_ROOT, "quality_contract.py")
        ),
        "registry_protocol_version": REGISTRY_PROTOCOL_VERSION,
        "solver_revision": args.solver_revision,
        "library_revision": args.library_revision,
    }
    raw, audited, strict, quarantine = inspect_dataset(
        dataset, args.profile, args.solver_revision, args.library_revision
    )
    state = _load_state(state_path)
    _ensure_identity(state, identity)
    registry = os.path.join(output_root, "registry")
    reconciled, reconciliation_issues = reconcile_completed(
        state, registry, expected_identity=identity
    )
    completed = {
        int(item["threshold"])
        for item in reconciled
        if isinstance(item, dict) and "threshold" in item
    }
    due = [
        threshold for threshold in checkpoint_sequence(len(strict))
        if threshold not in completed
    ]
    due, deferred = due_with_backoff(
        due, state, len(strict), args.retry_min_new_rows,
        args.retry_backoff_seconds,
        force_ready={item.get("threshold") for item in reconciliation_issues},
    )
    summary = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset,
        "raw_rows": int(len(raw)),
        "strict_em_rows": int(audited["_strict_valid_em"].sum()),
        "strict_full_rows": int(len(strict)),
        "quarantined_rows": int(len(raw) - len(strict)),
        "quarantine_reasons": quarantine,
        "completed_thresholds": sorted(completed),
        "due_thresholds": due,
        "deferred_thresholds": deferred,
        "checkpoint_state_schema": state.get("schema_version"),
        "checkpoint_run_root": run_root,
        "reconciliation_issues": reconciliation_issues,
        "state_identity": identity,
        "execute": bool(args.execute),
        "manufacturing_tolerance_policy": (
            "excluded; exact-as-FEA geometry is assumed"
        ),
    }
    print(json.dumps(summary, ensure_ascii=False))
    if not args.execute:
        return

    os.makedirs(output_root, exist_ok=True)
    with FileLock(os.path.join(output_root, "checkpoint.lock"), timeout=1):
        # Re-read all mutable inputs under the lock.  Two loop processes can
        # observe the same due threshold before locking, but only the first may
        # claim it; the second recomputes state after the first commits.
        locked_profile = load_profile(args.profile)
        locked_profile_sha256 = hashlib.sha256(
            json.dumps(
                locked_profile, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        with open(args.thresholds, encoding="utf-8") as handle:
            locked_thresholds = json.load(handle)
        locked_thresholds_sha256 = hashlib.sha256(
            json.dumps(
                locked_thresholds, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        if locked_profile_sha256 != profile_sha256:
            raise RuntimeError("quality profile changed during checkpoint inspection")
        if locked_thresholds_sha256 != thresholds_sha256:
            raise RuntimeError("quality thresholds changed during checkpoint inspection")
        raw, audited, strict, quarantine = inspect_dataset(
            dataset, args.profile, args.solver_revision, args.library_revision
        )
        state = _load_state(state_path)
        _ensure_identity(state, identity)
        reconciled, reconciliation_issues = reconcile_completed(
            state, registry, expected_identity=identity
        )
        if reconciliation_issues:
            state.setdefault("invalidated_completed", []).extend({
                **issue,
                "invalidated_at": datetime.now().isoformat(timespec="seconds"),
            } for issue in reconciliation_issues)
        state["completed"] = reconciled
        _save_state(state, state_path)
        completed = {
            int(item["threshold"])
            for item in reconciled
            if isinstance(item, dict) and "threshold" in item
        }
        due = [
            threshold for threshold in checkpoint_sequence(len(strict))
            if threshold not in completed
        ]
        due, deferred = due_with_backoff(
            due, state, len(strict), args.retry_min_new_rows,
            args.retry_backoff_seconds,
            force_ready={item.get("threshold") for item in reconciliation_issues},
        )
        summary.update({
            "time": datetime.now().isoformat(timespec="seconds"),
            "raw_rows": int(len(raw)),
            "strict_em_rows": int(audited["_strict_valid_em"].sum()),
            "strict_full_rows": int(len(strict)),
            "quarantined_rows": int(len(raw) - len(strict)),
            "quarantine_reasons": quarantine,
            "completed_thresholds": sorted(completed),
            "due_thresholds": due,
            "deferred_thresholds": deferred,
            "reconciliation_issues": reconciliation_issues,
        })
        _atomic_json(summary, os.path.join(output_root, "strict_data_status.json"))
        for threshold in due[: max(0, args.max_checkpoints_per_run)]:
            attempt = {
                "threshold": threshold,
                "strict_full_rows": int(len(strict)),
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "status": "running",
            }
            state.setdefault("attempts", []).append(attempt)
            _save_state(state, state_path)
            snapshot = _snapshot_name(run_root, threshold, strict)
            clean_snapshot = strict.drop(
                columns=[
                    "_strict_valid_em", "_strict_valid_thermal",
                    "_strict_valid_full", "_strict_invalid_reasons",
                ],
                errors="ignore",
            )
            _atomic_parquet(clean_snapshot, snapshot)
            registry = os.path.join(output_root, "registry")
            curve = os.path.join(output_root, "learning_curve.csv")
            quality_status = os.path.join(output_root, "model_quality_status.json")
            from train_models import (
                discard_inactive_generation, load_generation,
                promote_generation, prune_inactive_generations,
                registry_pointer_token,
            )
            candidate_generation = None
            attempt_number = len(state["attempts"])
            metrics_result = os.path.join(
                run_root,
                "checkpoint_metrics",
                f"threshold_{threshold:06d}_attempt_{attempt_number:06d}.json",
            )
            parity_result = os.path.splitext(metrics_result)[0] + ".parity.json"
            candidate_result = os.path.join(
                run_root,
                "candidate_results",
                f"threshold_{threshold:06d}_attempt_{attempt_number:06d}.json",
            )
            activation_minimum = int(
                locked_thresholds["minimum_strict_full_rows"]
            )
            activation_required = threshold >= activation_minimum
            expected_pointer = registry_pointer_token(registry)
            promoted = False
            try:
                for command in training_commands(
                    snapshot, curve, registry, args.min_rows,
                    args.profile, threshold, metrics_result,
                    candidate_result if activation_required else None,
                ):
                    _run(command)
                with open(metrics_result, encoding="utf-8") as handle:
                    metric_evidence = json.load(handle)
                with open(parity_result, encoding="utf-8") as handle:
                    parity_evidence = json.load(handle)
                snapshot_sha256 = _sha256(snapshot)
                metric_checks = {
                    "dataset": os.path.abspath(snapshot),
                    "dataset_sha256": snapshot_sha256,
                    "profile": args.profile,
                    "profile_sha256": locked_profile_sha256,
                    "strict_full_rows": int(len(strict)),
                }
                for key, expected in metric_checks.items():
                    if metric_evidence.get(key) != expected:
                        raise RuntimeError(
                            f"checkpoint metrics {key} mismatch"
                        )
                parity_checks = {
                    "schema_version": 1,
                    "artifact_type": "checkpoint_cv_oof_parity",
                    "checkpoint": threshold,
                    **metric_checks,
                    "prediction_kind": "out_of_fold",
                }
                for key, expected in parity_checks.items():
                    if parity_evidence.get(key) != expected:
                        raise RuntimeError(
                            f"checkpoint parity {key} mismatch"
                        )
                if not isinstance(parity_evidence.get("targets"), dict) or not parity_evidence["targets"]:
                    raise RuntimeError("checkpoint parity targets are empty")
                common_completion = {
                    "threshold": threshold,
                    "actual_strict_full_rows": int(len(strict)),
                    "snapshot": snapshot,
                    "snapshot_sha256": snapshot_sha256,
                    "metrics_result": metrics_result,
                    "metrics_result_sha256": _sha256(metrics_result),
                    "parity_result": parity_result,
                    "parity_result_sha256": _sha256(parity_result),
                    "profile_path": args.profile,
                    "profile_sha256": locked_profile_sha256,
                    "thresholds_sha256": locked_thresholds_sha256,
                    "activation_minimum_strict_full_rows": activation_minimum,
                    "completed_at": datetime.now().isoformat(timespec="seconds"),
                }
                if not activation_required:
                    completed_item = {
                        **common_completion,
                        "kind": "metrics_only",
                        "quality_passed": None,
                    }
                    state.setdefault("completed", []).append(completed_item)
                    attempt.update(status="completed", **completed_item)
                    _save_state(state, state_path)
                    continue
                with open(candidate_result, encoding="utf-8") as handle:
                    candidate = json.load(handle)
                candidate_generation = candidate["generation"]
                from model_quality_gate import evaluate_generation

                quality = evaluate_generation(
                    registry, candidate_generation, snapshot, locked_thresholds
                )
                quality.update(
                    {
                        "quality_thresholds_sha256": thresholds_sha256,
                        "solver_revision": args.solver_revision,
                        "library_revision": args.library_revision,
                        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
                        "checkpoint": threshold,
                    }
                )
                _atomic_json(quality, quality_status)
                if not quality["passed"]:
                    raise RuntimeError(
                        "surrogate quality gate failed: "
                        + "; ".join(quality["reasons"][:10])
                    )
                pointer = promote_generation(
                    registry,
                    candidate_generation,
                    quality,
                    dataset=snapshot,
                    profile_sha256=locked_profile_sha256,
                    thresholds_sha256=locked_thresholds_sha256,
                    expected_pointer=expected_pointer,
                )
                promoted = True
                accepted = load_generation(
                    registry, candidate_generation, require_accepted=True
                )
                completed_item = {
                    **common_completion,
                    "kind": "accepted_generation",
                    "training_run_id": pointer["training_run_id"],
                    "generation": pointer["generation"],
                    "generation_report_sha256": pointer[
                        "generation_report_sha256"
                    ],
                    "quality_gate_sha256": pointer["quality_gate_sha256"],
                    "quality_passed": True,
                }
                if accepted["quality_gate_sha256"] != completed_item[
                    "quality_gate_sha256"
                ]:
                    raise RuntimeError("accepted gate changed before state commit")
                state.setdefault("completed", []).append(completed_item)
                attempt.update(status="completed", **completed_item)
                _save_state(state, state_path)
                try:
                    protected_runs = _pinned_training_runs(runtime_root)
                    protected_runs.update(
                        item.get("training_run_id")
                        for item in state.get("completed", [])
                        if isinstance(item, dict) and item.get("training_run_id")
                    )
                    prune_inactive_generations(
                        registry, keep=3,
                        protected_run_ids=protected_runs,
                    )
                except Exception as prune_error:
                    print(
                        f"[checkpoint] generation retention warning: {prune_error}",
                        file=sys.stderr,
                    )
            except Exception as exc:
                # Candidate generation never changes current.json.  There is no
                # routine rollback path: either promotion committed a fully
                # gated pointer or the previous pointer remains byte-identical.
                if candidate_generation and not promoted:
                    try:
                        discard_inactive_generation(registry, candidate_generation)
                    except Exception as discard_error:
                        print(
                            f"[checkpoint] inactive candidate cleanup warning: {discard_error}",
                            file=sys.stderr,
                        )
                attempt.update(
                    status=(
                        "promotion_committed_state_recovery_required"
                        if promoted else "failed"
                    ),
                    error=str(exc),
                    finished_at=datetime.now().isoformat(timespec="seconds"),
                )
                _save_state(state, state_path)
                if os.path.isfile(snapshot) and not promoted:
                    os.remove(snapshot)
                raise


if __name__ == "__main__":
    main()
