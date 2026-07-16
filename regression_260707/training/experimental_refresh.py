"""Adopt a bounded experimental HPO wave and publish only a better surrogate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import psutil

try:
    from .experimental_surrogate import (
        atomic_json, build_experimental_quality, publish_if_better, sha256_file,
    )
except ImportError:
    from experimental_surrogate import (
        atomic_json, build_experimental_quality, publish_if_better, sha256_file,
    )


def _now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _parse_job(value):
    parts = value.split("|", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("job must be target|family|pid|result_json")
    target, family, pid, result = parts
    return {
        "target": target,
        "family": family,
        "pid": int(pid),
        "result_json": os.path.abspath(result),
    }


def _process_state(pid):
    try:
        process = psutil.Process(int(pid))
        return {
            "alive": process.is_running() and process.status() != psutil.STATUS_ZOMBIE,
            "status": process.status(),
            "cpu_seconds": sum(process.cpu_times()[:2]),
            "rss_bytes": process.memory_info().rss,
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return {"alive": False, "status": "exited"}


def _status(path, phase, jobs, **extra):
    inventory = []
    for job in jobs:
        item = dict(job)
        item.update(_process_state(job["pid"]))
        item["result_ready"] = os.path.isfile(job["result_json"])
        inventory.append(item)
    value = {
        "schema_version": 1,
        "updated_at": _now(),
        "phase": phase,
        "lane": "experimental",
        "eligibility": "FEA-NOT-APPROVED",
        "jobs": inventory,
        **extra,
    }
    atomic_json(path, value)
    return value


def _merge_params(jobs, output):
    merged = {}
    evidence = []
    for job in jobs:
        result = _read_json(job["result_json"])
        params_path = os.path.abspath(result["params_path"])
        params = _read_json(params_path)
        for family, targets in params.items():
            destination = merged.setdefault(family, {})
            overlap = set(destination) & set(targets)
            if overlap:
                raise RuntimeError("duplicate HPO parameter target: " + ",".join(overlap))
            destination.update(targets)
        evidence.append({
            "target": job["target"],
            "family": job["family"],
            "generation_id": result["generation_id"],
            "params_path": params_path,
            "params_sha256": sha256_file(params_path),
        })
    atomic_json(output, merged)
    return evidence


def _strict_snapshot(
    source, output_root, profile, solver_revision, library_revision,
    inspector=None,
):
    """Build a local immutable candidate cohort with exact revision pins."""
    if inspector is None:
        from checkpoint_orchestrator import inspect_dataset
        inspector = inspect_dataset

    source = os.path.abspath(source)
    source_sha256 = sha256_file(source)
    snapshot_root = Path(output_root).resolve() / "snapshots"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_root / (
        f"strict_{source_sha256[:16]}_{solver_revision[:12]}.parquet"
    )
    evidence_path = snapshot.with_suffix(".json")
    if snapshot.is_file() and evidence_path.is_file():
        evidence = _read_json(evidence_path)
        expected = {
            "source_dataset_sha256": source_sha256,
            "solver_revision": solver_revision,
            "library_revision": library_revision,
            "snapshot_sha256": sha256_file(snapshot),
        }
        if all(evidence.get(key) == value for key, value in expected.items()):
            return str(snapshot), evidence

    raw, audited, strict, quarantine = inspector(
        source, profile, solver_revision, library_revision
    )
    clean = strict.drop(
        columns=[
            "_strict_valid_em", "_strict_valid_thermal",
            "_strict_valid_full", "_strict_invalid_reasons",
        ],
        errors="ignore",
    )
    fd, staged = tempfile.mkstemp(
        prefix=f".{snapshot.name}.", suffix=".tmp", dir=snapshot_root
    )
    os.close(fd)
    try:
        clean.to_parquet(staged, index=False)
        os.replace(staged, snapshot)
    finally:
        try:
            os.remove(staged)
        except FileNotFoundError:
            pass
    evidence = {
        "schema_version": 1,
        "created_at": _now(),
        "source_dataset": source,
        "source_dataset_sha256": source_sha256,
        "snapshot": str(snapshot),
        "snapshot_sha256": sha256_file(snapshot),
        "raw_rows": int(len(raw)),
        "strict_em_rows": int(audited["_strict_valid_em"].sum()),
        "strict_full_rows": int(len(strict)),
        "quarantined_rows": int(len(raw) - len(strict)),
        "quarantine_reasons": quarantine,
        "solver_revision": solver_revision,
        "library_revision": library_revision,
        "profile": os.path.abspath(profile),
    }
    atomic_json(evidence_path, evidence)
    return str(snapshot), evidence


def _wait_process(process, status_path, jobs, phase, poll, **extra):
    while process.poll() is None:
        _status(
            status_path, phase, jobs,
            worker_pid=process.pid, worker_command=process.args, **extra,
        )
        time.sleep(poll)
    if process.returncode:
        _status(
            status_path, f"{phase}_failed_retryable", jobs,
            worker_pid=process.pid, worker_command=process.args,
            error=f"{phase} process exited {process.returncode}", **extra,
        )
        raise RuntimeError(f"{phase} process exited {process.returncode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", action="append", type=_parse_job, required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-dataset-generation", default=None)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--pointer", required=True)
    parser.add_argument("--consumer-source", required=True)
    parser.add_argument("--incumbent-source", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--solver-revision", required=True)
    parser.add_argument("--library-revision", required=True)
    parser.add_argument("--model-threads", type=int, default=12)
    parser.add_argument("--trials-per-job", type=int, default=50)
    parser.add_argument("--hpo-threads-per-job", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=10)
    args = parser.parse_args()
    if (
        args.model_threads < 1 or args.poll_seconds < 1
        or args.trials_per_job < 1 or args.hpo_threads_per_job < 1
    ):
        parser.error("thread and poll budgets must be positive")

    root = Path(args.runtime_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "status.json"
    work = root / "refresh_work"
    work.mkdir(parents=True, exist_ok=True)
    jobs = [
        {
            **job,
            "trials": args.trials_per_job,
            "model_threads": args.hpo_threads_per_job,
        }
        for job in args.job
    ]
    _status(
        status_path, "experimental_hpo", jobs,
        supervisor_pid=os.getpid(), dataset=os.path.abspath(args.dataset),
        dataset_sha256=sha256_file(args.dataset),
        trials_per_job=args.trials_per_job,
        total_trials=args.trials_per_job * len(jobs),
        total_model_thread_budget=args.hpo_threads_per_job * len(jobs),
    )
    while True:
        states = [_process_state(job["pid"]) for job in jobs]
        ready = [os.path.isfile(job["result_json"]) for job in jobs]
        if all(ready):
            break
        failed = [
            job["target"] for job, state, present in zip(jobs, states, ready)
            if not state["alive"] and not present
        ]
        if failed:
            raise RuntimeError("HPO child exited without result: " + ",".join(failed))
        _status(
            status_path, "experimental_hpo", jobs,
            supervisor_pid=os.getpid(), dataset=os.path.abspath(args.dataset),
            dataset_sha256=sha256_file(args.dataset),
            trials_per_job=args.trials_per_job,
            total_trials=args.trials_per_job * len(jobs),
            total_model_thread_budget=args.hpo_threads_per_job * len(jobs),
        )
        time.sleep(args.poll_seconds)

    merged_params = work / "merged_params.json"
    tuning_evidence = _merge_params(jobs, merged_params)
    training_dataset, strict_snapshot_evidence = _strict_snapshot(
        args.dataset, work, args.profile,
        args.solver_revision.lower(), args.library_revision.lower(),
    )
    candidate_result = work / "candidate_result.json"
    training_log = work / "candidate_training.log"
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "train_models.py"),
        "--dataset", training_dataset,
        "--registry", os.path.abspath(args.registry),
        "--profile", os.path.abspath(args.profile),
        "--params", str(merged_params),
        "--model-threads", str(args.model_threads),
        "--result-json", str(candidate_result),
        "--source-dataset-path", os.path.abspath(args.dataset),
    ]
    if args.source_dataset_generation:
        command.extend([
            "--source-dataset-generation", args.source_dataset_generation
        ])
    with training_log.open("ab") as log:
        process = subprocess.Popen(
            command, cwd=str(Path(__file__).resolve().parents[1]),
            stdout=log, stderr=subprocess.STDOUT,
            creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
        )
        _wait_process(
            process, status_path, jobs, "candidate_training",
            args.poll_seconds, supervisor_pid=os.getpid(),
            tuning_evidence=tuning_evidence,
            merged_params=str(merged_params),
            merged_params_sha256=sha256_file(merged_params),
            strict_snapshot=strict_snapshot_evidence,
        )
    candidate = _read_json(candidate_result)
    quality = build_experimental_quality(
        args.registry, candidate["generation"], training_dataset,
        args.thresholds, args.profile,
        args.solver_revision, args.library_revision,
    )
    result = publish_if_better(
        registry=args.registry,
        generation=candidate["generation"],
        dataset=training_dataset,
        quality=quality,
        evidence_root=root / "evidence",
        pointer_path=args.pointer,
        consumer_path=args.consumer_source,
        incumbent_source=args.incumbent_source,
        solver_revision=args.solver_revision,
        library_revision=args.library_revision,
    )
    _status(
        status_path,
        "candidate_promoted" if result["promoted"] else "candidate_rejected",
        jobs, supervisor_pid=os.getpid(), result=result,
        tuning_evidence=tuning_evidence,
        strict_snapshot=strict_snapshot_evidence,
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    # The supervisor launcher captures the traceback. A failure never rewrites
    # the active pointer because publication is the final guarded operation.
    main()
