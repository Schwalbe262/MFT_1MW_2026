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


def checkpoint_sequence(valid_count):
    fixed = [500, 1000, 2000, 3000]
    if valid_count >= 4000:
        fixed.extend(range(4000, valid_count + 1, 1000))
    return [threshold for threshold in fixed if threshold <= valid_count]


def due_with_backoff(
    due, state, strict_count, minimum_new_rows=250, backoff_seconds=3600,
    now=None,
):
    """Defer failed checkpoints until data grows or the retry timer expires."""
    now = now or datetime.now()
    ready, deferred = [], []
    attempts = state.get("attempts", [])
    for threshold in due:
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


def _load_state(path):
    if not os.path.isfile(path):
        return {"schema_version": 1, "completed": [], "attempts": []}
    with open(path, encoding="utf-8") as handle:
        state = json.load(handle)
    if state.get("schema_version") != 1:
        raise RuntimeError("unsupported checkpoint state schema")
    return state


def _run(command):
    print("[checkpoint] $ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=REGRESSION_ROOT)
    if result.returncode:
        raise RuntimeError(
            f"checkpoint command failed ({result.returncode}): {command}"
        )


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
    state_path = os.path.join(output_root, "checkpoint_state.json")
    from quality_contract import load_profile

    profile_data = load_profile(args.profile)
    profile_sha256 = hashlib.sha256(
        json.dumps(profile_data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    with open(args.thresholds, "rb") as handle:
        thresholds_sha256 = hashlib.sha256(handle.read()).hexdigest()
    identity = {
        "dataset": dataset,
        "profile_sha256": profile_sha256,
        "thresholds_sha256": thresholds_sha256,
        "solver_revision": args.solver_revision,
        "library_revision": args.library_revision,
    }
    raw, audited, strict, quarantine = inspect_dataset(
        dataset, args.profile, args.solver_revision, args.library_revision
    )
    state = _load_state(state_path)
    completed = {
        int(item["threshold"])
        for item in state.get("completed", [])
        if isinstance(item, dict) and "threshold" in item
    }
    due = [
        threshold for threshold in checkpoint_sequence(len(strict))
        if threshold not in completed
    ]
    due, deferred = due_with_backoff(
        due, state, len(strict), args.retry_min_new_rows,
        args.retry_backoff_seconds,
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
        raw, audited, strict, quarantine = inspect_dataset(
            dataset, args.profile, args.solver_revision, args.library_revision
        )
        state = _load_state(state_path)
        stored_identity = state.get("identity")
        if stored_identity is None:
            if state.get("completed"):
                raise RuntimeError(
                    "checkpoint state has completed work but no runtime identity"
                )
            state["identity"] = identity
            _atomic_json(state, state_path)
        elif stored_identity != identity:
            raise RuntimeError(
                "checkpoint runtime identity changed; use a new output root or archive state"
            )
        completed = {
            int(item["threshold"])
            for item in state.get("completed", [])
            if isinstance(item, dict) and "threshold" in item
        }
        due = [
            threshold for threshold in checkpoint_sequence(len(strict))
            if threshold not in completed
        ]
        due, deferred = due_with_backoff(
            due, state, len(strict), args.retry_min_new_rows,
            args.retry_backoff_seconds,
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
            _atomic_json(state, state_path)
            snapshot = _snapshot_name(output_root, threshold, strict)
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
                capture_active_generation, discard_inactive_generation,
                prune_inactive_generations, restore_active_generation,
            )
            registry_lock = FileLock(registry + ".training.lock", timeout=1)
            try:
                registry_lock.acquire()
            except Exception as lock_error:
                attempt.update(
                    status="lock_deferred",
                    error=str(lock_error),
                    finished_at=datetime.now().isoformat(timespec="seconds"),
                )
                _atomic_json(state, state_path)
                if os.path.isfile(snapshot):
                    os.remove(snapshot)
                print("[checkpoint] registry writer is busy; retrying next cycle")
                continue
            previous_generation = None
            generation_captured = False
            try:
                previous_generation = capture_active_generation(registry)
                generation_captured = True
                _run([
                    sys.executable,
                    str(HERE / "checkpoint_train.py"),
                    "--dataset", snapshot,
                    "--curve-csv", curve,
                    *( ["--profile", args.profile] if args.profile else [] ),
                ])
                _run([
                    sys.executable,
                    str(HERE / "train_models.py"),
                    "--dataset", snapshot,
                    "--registry", registry,
                    "--min-rows", str(args.min_rows),
                    *( ["--profile", args.profile] if args.profile else [] ),
                ])
                from model_quality_gate import evaluate_registry
                with open(args.thresholds, encoding="utf-8") as handle:
                    thresholds = json.load(handle)
                quality = evaluate_registry(registry, snapshot, thresholds)
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
                required_gate = threshold >= int(
                    thresholds["minimum_strict_full_rows"]
                )
                if required_gate and not quality["passed"]:
                    raise RuntimeError(
                        "surrogate quality gate failed: "
                        + "; ".join(quality["reasons"][:10])
                    )
                with open(
                    os.path.join(registry, "current.json"), encoding="utf-8"
                ) as handle:
                    pointer = json.load(handle)
                completed_item = {
                    "threshold": threshold,
                    "actual_strict_full_rows": int(len(strict)),
                    "snapshot": snapshot,
                    "training_run_id": pointer["training_run_id"],
                    "quality_passed": bool(quality["passed"]),
                    "completed_at": datetime.now().isoformat(timespec="seconds"),
                }
                state.setdefault("completed", []).append(completed_item)
                attempt.update(status="completed", **completed_item)
                _atomic_json(state, state_path)
                try:
                    prune_inactive_generations(
                        registry, keep=3,
                        protected_run_ids=_pinned_training_runs(runtime_root),
                    )
                except Exception as prune_error:
                    print(
                        f"[checkpoint] generation retention warning: {prune_error}",
                        file=sys.stderr,
                    )
            except Exception as exc:
                # Model files may have been fully trained but a downstream
                # quality/uncertainty gate failed.  Keep that generation for
                # audit while restoring the last accepted pointer atomically.
                rejected_generation = None
                pointer_path = os.path.join(registry, "current.json")
                if os.path.isfile(pointer_path):
                    try:
                        with open(pointer_path, encoding="utf-8") as handle:
                            current_pointer = json.load(handle)
                        candidate_generation = os.path.abspath(
                            os.path.join(registry, current_pointer["generation"])
                        )
                        prior_path = (
                            previous_generation.get("generation")
                            if previous_generation else None
                        )
                        if candidate_generation != prior_path:
                            rejected_generation = candidate_generation
                    except Exception:
                        rejected_generation = None
                if generation_captured:
                    restore_active_generation(registry, previous_generation)
                discard_inactive_generation(registry, rejected_generation)
                attempt.update(
                    status="failed",
                    error=str(exc),
                    finished_at=datetime.now().isoformat(timespec="seconds"),
                )
                _atomic_json(state, state_path)
                if os.path.isfile(snapshot):
                    os.remove(snapshot)
                raise
            finally:
                registry_lock.release()


if __name__ == "__main__":
    main()
