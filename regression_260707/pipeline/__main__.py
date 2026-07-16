"""Command-line administration for the durable pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from .artifacts import GenerationStore
from .controller import ContinuousController
from .orchestrator import (
    DEFAULT_MODEL_THREADS,
    PipelineOrchestrator,
    descriptor_from_active_registry,
)
from .queue import DurableJobQueue
from .runner import JobRunner
from .runtime_lock import AlreadyRunningError, RoleInstanceLock
from .supervisor import PipelineSupervisor


SINGLETON_EXIT_CODE = 73


def _paths(args):
    runtime = Path(args.runtime_root).resolve()
    pipeline_root = Path(args.pipeline_root or runtime / "pipeline_runtime").resolve()
    return (
        runtime,
        pipeline_root,
        DurableJobQueue(pipeline_root / "jobs.sqlite3"),
        GenerationStore(pipeline_root / "artifacts"),
    )


def _verification_config_identity(path: str | None) -> str | None:
    if not path:
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run_role_locked(pipeline_root, role, metadata, callback):
    try:
        with RoleInstanceLock(pipeline_root, role, metadata):
            return callback()
    except AlreadyRunningError as exc:
        print(
            json.dumps({
                "pipeline_singleton_error": str(exc),
                "role": role,
                "lock_path": str(exc.path),
                "owner": exc.owner,
                "exit_code": SINGLETON_EXIT_CODE,
            }, ensure_ascii=False),
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(SINGLETON_EXIT_CODE) from None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--pipeline-root", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status")
    status.add_argument("--limit", type=int, default=30)

    worker = sub.add_parser("worker")
    worker.add_argument("--types", required=True)
    worker.add_argument("--once", action="store_true")

    sub.add_parser("supervise")

    control = sub.add_parser("control")
    control.add_argument("--dataset", default=None)
    control.add_argument("--registry", default=None)
    control.add_argument("--solver-revision", required=True)
    control.add_argument("--library-revision", required=True)
    control.add_argument("--interval-seconds", type=int, default=600)
    control.add_argument("--optuna-trials", type=int, default=200)
    control.add_argument(
        "--model-threads", type=int, default=DEFAULT_MODEL_THREADS
    )
    control.add_argument("--verification-commands", default=None)
    control.add_argument("--once", action="store_true")

    plan = sub.add_parser("plan")
    plan.add_argument("--dataset", default=None)
    plan.add_argument("--strict-full-rows", type=int, required=True)
    plan.add_argument("--solver-revision", required=True)
    plan.add_argument("--library-revision", required=True)
    plan.add_argument("--registry", default=None)
    plan.add_argument("--no-active-model", action="store_true")
    plan.add_argument("--drift-detected", action="store_true")
    plan.add_argument("--quality-regression", action="store_true")
    plan.add_argument("--optuna-trials", type=int, default=200)
    plan.add_argument(
        "--model-threads", type=int, default=DEFAULT_MODEL_THREADS
    )
    plan.add_argument("--verification-commands", default=None)
    args = parser.parse_args()

    runtime, pipeline_root, queue, store = _paths(args)
    if args.command == "status":
        print(json.dumps({
            "stats": queue.stats(),
            "jobs": [job.__dict__ for job in queue.list(limit=args.limit)],
        }, indent=1, ensure_ascii=False))
        return
    if args.command == "worker":
        runner = JobRunner(queue, store, pipeline_root / "work")
        types = [value.strip() for value in args.types.split(",") if value.strip()]
        if args.once:
            job = runner.run_once(types)
            print(json.dumps(job.__dict__ if job else None, default=str))
        else:
            runner.run_forever(types)
        return
    if args.command == "supervise":
        _run_role_locked(
            pipeline_root,
            "supervisor",
            {"command": "supervise"},
            lambda: PipelineSupervisor(
                queue.path, store.root, pipeline_root / "work"
            ).run(),
        )
        return
    if args.command == "control":
        verification_commands = None
        if args.verification_commands:
            verification_commands = json.loads(
                Path(args.verification_commands).read_text(encoding="utf-8")
            )
        controller = ContinuousController(
            PipelineOrchestrator(queue, store, runtime, python=sys.executable),
            dataset=(
                args.dataset or runtime / "data" / "dataset" / "train.parquet"
            ),
            registry=(args.registry or runtime / "training" / "registry"),
            solver_revision=args.solver_revision.lower(),
            library_revision=args.library_revision.lower(),
            optuna_trials=args.optuna_trials,
            model_threads=args.model_threads,
            verification_commands=verification_commands,
        )
        def run_controller():
            if args.once:
                print(json.dumps(controller.plan_once().__dict__, indent=1))
            else:
                controller.run_forever(args.interval_seconds)

        _run_role_locked(
            pipeline_root,
            "controller",
            {
                "command": "control",
                "solver_revision": args.solver_revision.lower(),
                "library_revision": args.library_revision.lower(),
                "model_threads": args.model_threads,
                "verification_config_sha256": _verification_config_identity(
                    args.verification_commands
                ),
            },
            run_controller,
        )
        return

    dataset = os.path.abspath(
        args.dataset or runtime / "data" / "dataset" / "train.parquet"
    )
    registry = os.path.abspath(
        args.registry or runtime / "training" / "registry"
    )
    active = None
    if not args.no_active_model and os.path.isfile(
        os.path.join(registry, "current.json")
    ):
        active = descriptor_from_active_registry(registry)
    commands = None
    if args.verification_commands:
        commands = json.loads(
            Path(args.verification_commands).read_text(encoding="utf-8")
        )
    result = PipelineOrchestrator(
        queue, store, runtime, python=sys.executable
    ).plan_cycle(
        dataset_path=dataset,
        strict_full_rows=args.strict_full_rows,
        solver_revision=args.solver_revision,
        library_revision=args.library_revision,
        active_model=active,
        drift_detected=args.drift_detected,
        quality_regression=args.quality_regression,
        optuna_trials=args.optuna_trials,
        model_threads=args.model_threads,
        verification_commands=commands,
    )
    print(json.dumps(result.__dict__, indent=1))


if __name__ == "__main__":
    main()
