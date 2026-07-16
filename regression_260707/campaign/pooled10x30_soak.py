"""Keep exactly 30 MFT pooled projects backed by at most 10 AEDT sessions.

This is deliberately independent from the standalone 400-project production
controller.  It counts only its own name prefix and only ``aedt_backend=pooled``
tasks, while the scheduler's MFT project cap remains the shared 500-task
stop-loss.  Completed, failed, or cancelled soak tasks are refilled until this
controller is stopped; it never cancels any task.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import requests
from filelock import FileLock, Timeout as FileLockTimeout


HERE = Path(__file__).resolve().parent
REGRESSION_ROOT = HERE.parent
REPO_ROOT = REGRESSION_ROOT.parent
VERIFY_ROOT = REGRESSION_ROOT / "verify"
for path in (HERE, REGRESSION_ROOT, VERIFY_ROOT, REPO_ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

import deployment_gate
import feeder
import q22_bounded_soak as q22
import scheduler_client
from module.core_material_contract import PHYSICS_DATA_REVISION


SCHEMA = "mft-pooled10x30-soak-controller-v1"
CAMPAIGN_ID = "mft-pooled10x30-soak-260717"
PROJECT = "MFT_1MW_2026v1"
SOLVER_REVISION = "7768510433858c9056f04320e66819d5fcc90f1a"
LIBRARY_REVISION = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
IMMUTABLE_SCHEDULER_PACKAGE_REVISION = (
    "4df497a49a7eccfe441b0d198540990ebbffcd4a"
)
SCHEDULER_PACKAGE_REVISION = "04eb5156bcb1219ae0ee43bb24ab1aacd08f36af"
CANDIDATE_SEED = 260717
TARGET_PROJECTS = 30
POOL_SESSIONS = 10
PROJECTS_PER_AEDT = 3
PROJECT_CPUS = 4
SESSION_BASE_CPUS = 1
SESSION_RESERVED_CPUS = 13
PROJECT_CAP = 500
TASK_CPUS = 1
TASK_MEMORY_MB = 6144
TASK_TIMEOUT_SECONDS = 86_400
NAME_PREFIX = (
    f"mft-camp-s{SOLVER_REVISION[:7]}-l{LIBRARY_REVISION[:7]}-p10x30-"
)
ACTIVE_STATES = ("queued", "attaching", "running")
ELIGIBLE_ACCOUNTS = (
    "dhj02",
    "harry261",
    "jji0930",
    "dw16",
    "r1jae262",
)

DEFAULT_SCHEDULER_URL = "http://127.0.0.1:8002"
DEFAULT_POOL_URL = "http://172.16.10.37:18790"
DEFAULT_RUNTIME_DIR = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_pooled10x30"
)
DEFAULT_DATASET_DIR = Path(
    r"Y:\git\MFT_solver_pooled_260714\regression_260707\data\dataset"
)
DEFAULT_LIBRARY_ROOT = Path(r"Y:\git\pyaedt_library_release_e6b9_260715")
DEFAULT_SOLVER_ROOT = Path(r"Y:\git\MFT_1MW_2026")
DEFAULT_ACCOUNTS_CONFIG = Path(r"Y:\runtime\slurm_scheduler\config\accounts.yaml")
DEFAULT_AUDIT_PYTHON = Path(r"C:\Python314\python.exe")
PROFILE_PATH = VERIFY_ROOT / "profiles" / "q23_same_node_full.json"


class GateError(RuntimeError):
    """A live contract is absent or has drifted."""


def _http_json(url: str, path: str, *, params: Mapping[str, Any] | None = None) -> Any:
    try:
        response = requests.get(
            f"{url.rstrip('/')}{path}", params=params, timeout=30
        )
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        raise GateError(f"scheduler GET {path} failed: {exc}") from exc


def verify_pool(summary: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(summary, Mapping):
        raise GateError("AEDT pool summary is not an object")
    config = summary.get("config")
    if not isinstance(config, Mapping):
        raise GateError("AEDT pool summary has no config")
    required = {
        "enabled": True,
        "adapter_ready": True,
        "validation_passed": True,
        "operational": True,
        "max_aedt_sessions": POOL_SESSIONS,
        "target_project_concurrency": TARGET_PROJECTS,
        "projects_per_aedt": PROJECTS_PER_AEDT,
        "project_cpus": PROJECT_CPUS,
        "session_reserved_cpus": SESSION_RESERVED_CPUS,
        "native_solve_mode": "validated_parallel",
        "parallel_safe_native_solve_families": ["mft_validated_async"],
    }
    drift = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in required.items()
        if config.get(key) != expected
    }
    if drift:
        raise GateError(f"pooled10x30 live pool contract drifted: {drift}")
    latest = summary.get("latest_validation")
    if not isinstance(latest, Mapping) or latest.get("status") != "passed":
        raise GateError("latest AEDT pool validation is not passed")
    plan = summary.get("plan") if isinstance(summary.get("plan"), Mapping) else {}
    hard_count = int(plan.get("hard_session_count") or 0)
    if hard_count > POOL_SESSIONS:
        raise GateError(
            f"active/starting AEDT hard count {hard_count} exceeds {POOL_SESSIONS}"
        )
    # Historical unhealthy records are intentionally not a hard-counted state.
    return {
        "hard_session_count": hard_count,
        "active_session_count": int(plan.get("active_session_count") or 0),
        "starting_session_count": int(plan.get("starting_session_count") or 0),
        "live_projects": int(plan.get("live_projects") or 0),
        "queued_pooled_task_backlog": int(
            plan.get("queued_pooled_task_backlog") or 0
        ),
        "archival_unhealthy_count": int(plan.get("unhealthy_session_count") or 0),
    }


def pool_snapshot(scheduler_url: str) -> dict[str, Any]:
    return verify_pool(_http_json(scheduler_url, "/api/aedt-pool"))


def _task_rows(payload: Any) -> list[dict[str, Any]]:
    rows = payload if isinstance(payload, list) else (
        payload.get("tasks") if isinstance(payload, Mapping) else None
    )
    if not isinstance(rows, list):
        raise GateError("scheduler returned an invalid soak task inventory")
    return rows


def active_inventory(scheduler_url: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = _task_rows(_http_json(
        scheduler_url,
        "/api/tasks",
        params={
            "limit": 1000,
            "name_prefix": NAME_PREFIX,
            "status": ",".join(ACTIVE_STATES),
        },
    ))
    if len(rows) > TARGET_PROJECTS:
        raise GateError(
            f"pooled10x30 active inventory exceeds target: {len(rows)}"
        )
    seen: set[int] = set()
    counts = Counter({state: 0 for state in ACTIVE_STATES})
    for row in rows:
        if not isinstance(row, Mapping):
            raise GateError("soak inventory contains a non-object")
        task_id = row.get("id", row.get("task_id"))
        status = str(row.get("status") or row.get("state") or "").lower()
        checks = {
            "id": type(task_id) is int and task_id > 0 and task_id not in seen,
            "name": str(row.get("name") or "").startswith(NAME_PREFIX),
            "project": str(row.get("project") or "") == PROJECT,
            "backend": str(row.get("aedt_backend") or "").lower() == "pooled",
            "status": status in ACTIVE_STATES,
        }
        if not all(checks.values()):
            raise GateError(
                f"soak task identity drift for task {task_id}: {checks}"
            )
        seen.add(task_id)
        counts[status] += 1
    return [dict(row) for row in rows], dict(counts)


def submission_environment(pool_url: str) -> dict[str, str]:
    return {
        "MFT_AEDT_BACKEND": "pooled",
        "MFT_AEDT_SHARED_CANARY": "1",
        "MFT_AEDT_SCHEDULER_URL": pool_url,
        "MFT_SLURM_SCHEDULER_ROOT": "$HOME/slurm_scheduler/aedt_pool_pkg",
        "SLURM_AEDT_POOL_CLIENT_TOKEN_FILE": (
            "$HOME/slurm_scheduler/aedt_pool_client"
        ),
        "MFT_AEDT_POOL_WORKSPACE": (
            "/gpfs/tmp_cpu2/mft_pool/mft-${SLURM_SCHED_TASK_ID}"
        ),
        "MFT_AEDT_WORKSPACE_PATH": (
            "/gpfs/tmp_cpu2/mft_pool/mft-${SLURM_SCHED_TASK_ID}"
        ),
        "MFT_AEDT_SESSION_VERSION": "2025.2",
        "MFT_AEDT_SESSION_PROFILE": feeder.AEDT_SESSION_PROFILE,
        "MFT_AEDT_ISOLATION_POLICY": "family",
        "AEDT_POOL_AUTOMATION_LOCK_TIMEOUT_SECONDS": "7200",
        "AEDT_POOL_NATIVE_PIPELINE_BARRIER_TIMEOUT_SECONDS": "7200",
        "MFT_AEDT_RELEASE_WAIT_SECONDS": "7200",
        "MFT_AEDT_POOLED_SOLVE_TIMEOUT_SECONDS": "7200",
        "MFT_AEDT_POOL_FILL_TIMEOUT_SECONDS": "900",
        "MFT_AEDT_WORKLOAD_FAMILY": "mft_validated_async",
        "MFT_AEDT_ASYNC_DISPATCH_SETTLE_SECONDS": "2",
        "MFT_CAMPAIGN_ID": CAMPAIGN_ID,
        "MFT_CAMPAIGN_PARENT": "q24-validated-async-aedt500-260716",
        "MFT_CAMPAIGN_PHYSICS_DATA_REVISION": PHYSICS_DATA_REVISION,
        "MFT_CAMPAIGN_MIGRATION_REPLACEMENT_SOLVER": SOLVER_REVISION,
        "MFT_CAMPAIGN_IMMUTABLE_SCHEDULER_PACKAGE_REVISION": (
            IMMUTABLE_SCHEDULER_PACKAGE_REVISION
        ),
        "MFT_CAMPAIGN_SCHEDULER_PACKAGE_REVISION": (
            SCHEDULER_PACKAGE_REVISION
        ),
    }


def _initial_state() -> dict[str, Any]:
    cursor = feeder.cursor_after_valid_candidates(
        feeder.PILOT_RESERVED_VALID_CANDIDATES, seed=CANDIDATE_SEED
    )
    return {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "solver_revision": SOLVER_REVISION,
        "library_revision": LIBRARY_REVISION,
        "candidate_seed": CANDIDATE_SEED,
        "serial": 0,
        "candidate_cursor": int(cursor),
        "accepted_tasks": 0,
        "recent_task_ids": [],
    }


def load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return _initial_state()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise GateError(f"pooled10x30 state is unreadable: {exc}") from exc
    fixed = {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "solver_revision": SOLVER_REVISION,
        "library_revision": LIBRARY_REVISION,
        "candidate_seed": CANDIDATE_SEED,
    }
    drift = {key: value.get(key) for key, expected in fixed.items()
             if value.get(key) != expected}
    if drift:
        raise GateError(f"pooled10x30 state identity drifted: {drift}")
    for key in ("serial", "candidate_cursor", "accepted_tasks"):
        if type(value.get(key)) is not int or int(value[key]) < 0:
            raise GateError(f"pooled10x30 state {key} is invalid")
    return dict(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    staged.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(staged, path)


def verify_deployment_and_packages(args: argparse.Namespace) -> dict[str, Any]:
    deployment = deployment_gate.validate_deployment(
        args.solver_root,
        SOLVER_REVISION,
        args.library_root,
        LIBRARY_REVISION,
    )
    original = q22.SCHEDULER_PACKAGE_REVISION
    q22.SCHEDULER_PACKAGE_REVISION = SCHEDULER_PACKAGE_REVISION
    try:
        packages = q22.audit_remote_packages(
            args.accounts_config,
            ELIGIBLE_ACCOUNTS,
            args.audit_python,
        )
    finally:
        q22.SCHEDULER_PACKAGE_REVISION = original
    return {"deployment": deployment, "packages": packages}


def execute_cycle(args: argparse.Namespace, static_gates: Mapping[str, Any]) -> dict[str, Any]:
    state_path = args.runtime_dir / "state.json"
    status: dict[str, Any]
    scheduler_client.SCHEDULER = args.scheduler_url.rstrip("/")
    feeder.SCHEDULER = scheduler_client.SCHEDULER
    with scheduler_client.campaign_mutation_lock(timeout=300):
        pool_before = pool_snapshot(args.scheduler_url)
        inventory_before, counts_before = active_inventory(args.scheduler_url)
        project_capacity = scheduler_client.live_project_submission_snapshot(
            PROJECT_CAP,
            max_project_active_tasks=PROJECT_CAP,
        )
        active_before = len(inventory_before)
        deficit = TARGET_PROJECTS - active_before
        project_slots = int(project_capacity["project_submission_slots"])
        submit_count = min(deficit, project_slots)
        state = load_state(state_path)
        submitted: list[int] = []
        environment = submission_environment(args.pool_url)

        for _ in range(submit_count):
            serial = int(state["serial"]) + 1
            next_cursor, raw_index, params = feeder.next_valid_candidate(
                int(state["candidate_cursor"]), seed=CANDIDATE_SEED
            )
            name = f"{NAME_PREFIX}{serial:07d}"
            task_id = feeder.submit(
                name,
                f"mft_p10x30_t{serial % 500:03d}",
                params,
                SOLVER_REVISION,
                LIBRARY_REVISION,
                cpus=TASK_CPUS,
                memory_mb=TASK_MEMORY_MB,
                timeout_seconds=TASK_TIMEOUT_SECONDS,
                aedt_backend="pooled",
                submission_env=environment,
                required_hard_cap=PROJECT_CAP,
                max_project_active_tasks=PROJECT_CAP,
                # Leave flexible soak work unpinned.  The scheduler owns the
                # live account decision because it can see per-account GPFS
                # headroom and must reserve a whole three-project AEDT cohort.
                # A durable round-robin pin can strand queued work on an
                # account that cannot safely start another AEDT session.
                account_name="",
                profile_path=str(PROFILE_PATH),
                prevalidated_cycle=True,
            )
            if type(task_id) is not int or task_id <= 0:
                raise GateError(f"scheduler rejected pooled soak task {name}")
            state["serial"] = serial
            state["candidate_cursor"] = int(next_cursor)
            state["candidate_raw_index"] = int(raw_index)
            state["accepted_tasks"] = int(state["accepted_tasks"]) + 1
            recent = list(state.get("recent_task_ids") or [])
            recent.append(task_id)
            state["recent_task_ids"] = recent[-100:]
            write_json(state_path, state)
            submitted.append(task_id)

        inventory_after, counts_after = active_inventory(args.scheduler_url)
        active_after = len(inventory_after)
        if active_after > TARGET_PROJECTS:
            raise GateError("pooled10x30 post-submit count exceeded target")
        phase = (
            "steady"
            if active_after == TARGET_PROJECTS
            else "waiting-project-capacity"
        )
        status = {
            "schema": SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "phase": phase,
            "updated_at_epoch": time.time(),
            "target_projects": TARGET_PROJECTS,
            "target_aedt_sessions": POOL_SESSIONS,
            "projects_per_aedt": PROJECTS_PER_AEDT,
            "name_prefix": NAME_PREFIX,
            "active_before": active_before,
            "active_after": active_after,
            "counts_before": counts_before,
            "counts_after": counts_after,
            "submitted_task_ids": submitted,
            "project_active": int(project_capacity["project_active"]),
            "project_slots_before": project_slots,
            "pool_before": pool_before,
            "static_gates": dict(static_gates),
            "no_cancellation_performed": True,
            "standalone_tasks_counted": False,
        }
    write_json(args.runtime_dir / "status.json", status)
    return status


def blocked_status(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    status = {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase": "blocked-fail-closed",
        "updated_at_epoch": time.time(),
        "blocker": f"{type(exc).__name__}: {exc}",
        "no_submission_attempted_or_unreconciled": True,
        "no_cancellation_performed": True,
    }
    write_json(args.runtime_dir / "status.json", status)
    return status


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--authorize-pooled10-aedt-30-projects",
        action="store_true",
        dest="authorize",
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=15)
    parser.add_argument("--static-gate-interval-seconds", type=int, default=1800)
    parser.add_argument("--scheduler-url", default=DEFAULT_SCHEDULER_URL)
    parser.add_argument("--pool-url", default=DEFAULT_POOL_URL)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--solver-root", type=Path, default=DEFAULT_SOLVER_ROOT)
    parser.add_argument("--library-root", type=Path, default=DEFAULT_LIBRARY_ROOT)
    parser.add_argument(
        "--accounts-config", type=Path, default=DEFAULT_ACCOUNTS_CONFIG
    )
    parser.add_argument("--audit-python", type=Path, default=DEFAULT_AUDIT_PYTHON)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.interval_seconds < 5:
        raise GateError("interval-seconds must be at least 5")
    if args.static_gate_interval_seconds < 300:
        raise GateError("static-gate-interval-seconds must be at least 300")
    if not args.dataset_dir.is_dir():
        raise GateError(f"canonical dataset directory is missing: {args.dataset_dir}")
    args.runtime_dir.mkdir(parents=True, exist_ok=True)
    scheduler_client.SCHEDULER = args.scheduler_url.rstrip("/")
    feeder.SCHEDULER = scheduler_client.SCHEDULER

    plan = {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "mode": "write-free-plan",
        "target": {"aedt_sessions": 10, "projects": 30, "per_aedt": 3},
        "solver_revision": SOLVER_REVISION,
        "library_revision": LIBRARY_REVISION,
        "scheduler_package_revision": SCHEDULER_PACKAGE_REVISION,
        "candidate_seed": CANDIDATE_SEED,
        "name_prefix": NAME_PREFIX,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "standalone_controller_independent": True,
        "cancellation": "never",
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if not args.authorize:
        raise GateError(
            "--authorize-pooled10-aedt-30-projects is required with --execute"
        )

    lock_path = args.runtime_dir / "controller.lock"
    try:
        with FileLock(str(lock_path), timeout=0):
            static_gates: dict[str, Any] = {}
            last_static_gate = 0.0
            while True:
                try:
                    now = time.monotonic()
                    if now - last_static_gate >= args.static_gate_interval_seconds:
                        static_gates = verify_deployment_and_packages(args)
                        last_static_gate = now
                    status = execute_cycle(args, static_gates)
                except Exception as exc:
                    status = blocked_status(args, exc)
                print(json.dumps(status, sort_keys=True), flush=True)
                if args.once:
                    return 0 if status.get("phase") in {
                        "steady", "waiting-project-capacity"
                    } else 2
                time.sleep(args.interval_seconds)
    except FileLockTimeout as exc:
        raise GateError("another pooled10x30 controller owns the runtime lock") from exc


if __name__ == "__main__":
    raise SystemExit(main())
