"""Audit or safely deploy the exact q23 client package to all five accounts.

Dry-run is read-only.  Execute mode preflights the complete account set before
the first write, validates the target in a temporary Git worktree, and rolls
back already switched accounts if any later account fails.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import yaml


FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
DEFAULT_ACCOUNTS = ("dhj02", "harry261", "jji0930", "dw16", "r1jae262")
DEFAULT_SCHEDULER_DB = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\data\slurm_scheduler.db"
)
DEFAULT_CONTROLLER_LOCK = Path(
    r"Y:\git\MFT_1MW_2026\regression_260707\campaign\feeder-controller.lock"
)
DEFAULT_LOCK_CHECK_PYTHON = Path(
    r"C:\Users\peets\anaconda3\envs\pyaedt2026v1\python.exe"
)
Q22_CAMPAIGN_ID = "q22-bounded-soak500-260716"
LIVE_TASK_STATES = ("queued", "attaching", "running")
LIVE_LEASE_STATES = ("offered", "leased", "attaching", "active", "releasing")


def _load_accounts(path: Path) -> dict[str, dict[str, Any]]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows = value.get("accounts") if isinstance(value, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError("accounts config has no accounts list")
    return {
        str(row.get("name") or ""): row
        for row in rows
        if isinstance(row, dict)
    }


def _connect(row: dict[str, Any]) -> Any:
    import paramiko

    key = Path(str(row.get("private_key_path") or ""))
    if not key.is_file():
        raise RuntimeError(f"SSH key is missing: {key}")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=str(row.get("host") or ""),
        port=int(row.get("port") or 22),
        username=str(row.get("username") or ""),
        key_filename=str(key),
        look_for_keys=False,
        allow_agent=False,
        timeout=15,
        banner_timeout=15,
        auth_timeout=15,
    )
    return client


def _run(client: Any, command: str, timeout: int = 120) -> str:
    _stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    output = stdout.read().decode("utf-8", errors="replace").strip()
    error = stderr.read().decode("utf-8", errors="replace").strip()
    return_code = stdout.channel.recv_exit_status()
    if return_code:
        raise RuntimeError(
            f"remote command failed rc={return_code}: {error or output or '<no output>'}"
        )
    return output


def _clean_guard(expected: str) -> str:
    return (
        "set -eu; root=\"$HOME/slurm_scheduler/aedt_pool_pkg\"; "
        "test -d \"$root/.git\"; "
        f"test \"$(git -C \"$root\" rev-parse HEAD)\" = \"{expected}\"; "
        "git -C \"$root\" diff --quiet HEAD --; "
        "status=$(git -C \"$root\" status --porcelain --untracked-files=all "
        "| grep -Ev '^\\?\\? batch\\.log$' || true); test -z \"$status\"; "
    )


def _old_import_check(root: str = '"$root"') -> str:
    return (
        f"PYTHONPATH={root} python -c \"from "
        "slurm_scheduler.aedt_attach_client import AedtProjectLease; "
        "assert hasattr(AedtProjectLease, 'wait_for_native_pipeline_barrier')\""
    )


def _target_import_check(root: str) -> str:
    return (
        f"PYTHONPATH={root} python -c \"from "
        "slurm_scheduler.aedt_attach_client import "
        "AedtProjectLease, MAX_POOL_FILL_TIMEOUT_SECONDS; "
        "assert hasattr(AedtProjectLease, 'wait_for_native_pipeline_barrier'); "
        "assert MAX_POOL_FILL_TIMEOUT_SECONDS == 7200.0\""
    )


def _preflight_command(current: str, env_setup: str) -> str:
    return (
        _clean_guard(current)
        + env_setup
        + f"; {_old_import_check()}; printf '%s\\n' {current}"
    )


def _exact_audit_command(expected: str, env_setup: str) -> str:
    return (
        _clean_guard(expected)
        + env_setup
        + f"; {_target_import_check(chr(34) + '$root' + chr(34))}; "
        + f"printf '%s\\n' {expected}"
    )


def _deploy_command(current: str, target: str, env_setup: str) -> str:
    stage = f"$HOME/slurm_scheduler/.q23_pkg_verify_{target[:12]}_$$"
    return (
        _clean_guard(current)
        + f"target={target}; stage=\"{stage}\"; switched=0; "
        "cleanup() { git -C \"$root\" worktree remove --force \"$stage\" "
        ">/dev/null 2>&1 || true; }; trap cleanup EXIT; "
        "finish() { rc=$?; trap - EXIT; cleanup; "
        f"if [ \"$rc\" -ne 0 ] && [ \"$switched\" -eq 1 ]; then "
        f"git -C \"$root\" checkout --detach {current} >/dev/null 2>&1 || true; "
        "fi; exit \"$rc\"; }; trap finish EXIT; "
        "git -C \"$root\" fetch --no-tags origin \"$target\"; "
        "git -C \"$root\" cat-file -e \"$target^{commit}\"; "
        "git -C \"$root\" worktree add --detach \"$stage\" \"$target\"; "
        + env_setup
        + f"; {_target_import_check(chr(34) + '$stage' + chr(34))}; "
        "cleanup; git -C \"$root\" checkout --detach \"$target\"; switched=1; "
        f"{_target_import_check(chr(34) + '$root' + chr(34))}; "
        "git -C \"$root\" diff --quiet HEAD --; "
        "status=$(git -C \"$root\" status --porcelain --untracked-files=all "
        "| grep -Ev '^\\?\\? batch\\.log$' || true); test -z \"$status\"; "
        "switched=0; trap - EXIT; printf '%s\\n' \"$target\""
    )


def _rollback_command(current: str) -> str:
    return (
        "set -eu; root=\"$HOME/slurm_scheduler/aedt_pool_pkg\"; "
        "git -C \"$root\" diff --quiet HEAD --; "
        f"git -C \"$root\" checkout --detach {current}; "
        f"test \"$(git -C \"$root\" rev-parse HEAD)\" = \"{current}\""
    )


def _verify_q22_controller_stopped(
    scheduler_db: Path,
    controller_lock: Path,
    lock_check_python: Path,
    scheduled_task_name: str,
) -> None:
    """Verify the q22 supervisor, tasks, leases, and controller lock are idle."""

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", scheduled_task_name):
        raise RuntimeError("invalid q22 scheduled task name")
    state_script = (
        "$task=Get-ScheduledTask "
        f"-TaskName '{scheduled_task_name}' -ErrorAction Stop; "
        "[Console]::Out.Write([string]$task.State)"
    )
    state_result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            state_script,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=False,
    )
    state = state_result.stdout.strip()
    if state_result.returncode or state not in {"Ready", "Disabled"}:
        raise RuntimeError(
            "q22 scheduled controller is not verifiably stopped: "
            f"{state or 'query failed'}"
        )
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            scheduler_db.resolve().as_uri() + "?mode=ro", uri=True
        )
        task_placeholders = ",".join("?" for _ in LIVE_TASK_STATES)
        lease_placeholders = ",".join("?" for _ in LIVE_LEASE_STATES)
        live_tasks = int(connection.execute(
            f"""
            SELECT COUNT(*) FROM tasks
            WHERE status IN ({task_placeholders}) AND command LIKE ?
            """,
            (*LIVE_TASK_STATES, f"%{Q22_CAMPAIGN_ID}%"),
        ).fetchone()[0])
        live_leases = int(connection.execute(
            f"""
            SELECT COUNT(*)
            FROM aedt_project_leases AS lease
            JOIN tasks AS task ON task.id = lease.task_id
            WHERE lease.state IN ({lease_placeholders}) AND task.command LIKE ?
            """,
            (*LIVE_LEASE_STATES, f"%{Q22_CAMPAIGN_ID}%"),
        ).fetchone()[0])
    except sqlite3.Error as exc:
        raise RuntimeError(f"q22 scheduler boundary query failed: {exc}") from exc
    finally:
        if connection is not None:
            connection.close()
    if live_tasks or live_leases:
        raise RuntimeError(
            "q22 controller is not settled: "
            f"live_tasks={live_tasks} live_leases={live_leases}"
        )
    if not lock_check_python.is_file():
        raise RuntimeError(
            f"controller lock-check Python is missing: {lock_check_python}"
        )
    lock_result = subprocess.run(
        [
            str(lock_check_python),
            "-c",
            (
                "import sys; from filelock import FileLock; "
                "lock=FileLock(sys.argv[1]); lock.acquire(timeout=0); lock.release()"
            ),
            str(controller_lock),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=False,
    )
    if lock_result.returncode:
        raise RuntimeError("q22 canonical controller lock is still held")


def deploy(
    config_path: Path,
    account_names: list[str],
    current: str,
    target: str,
    *,
    execute: bool,
    exact_audit: bool = False,
    q22_controller_stopped: bool = False,
    scheduler_db: Path = DEFAULT_SCHEDULER_DB,
    controller_lock: Path = DEFAULT_CONTROLLER_LOCK,
    lock_check_python: Path = DEFAULT_LOCK_CHECK_PYTHON,
    scheduled_task_name: str = "MFT_Q22_OpenEnded500",
) -> list[dict[str, Any]]:
    if not FULL_SHA.fullmatch(current) or not FULL_SHA.fullmatch(target):
        raise RuntimeError("current and target must be full lowercase commit SHAs")
    if execute and tuple(account_names) != DEFAULT_ACCOUNTS:
        raise RuntimeError(
            "q23 execute deployment requires the exact ordered five-account set"
        )
    if execute and exact_audit:
        raise RuntimeError("exact audit mode is read-only")
    if execute and not q22_controller_stopped:
        raise RuntimeError("q23 execute deployment requires q22 stop acknowledgement")
    if execute:
        _verify_q22_controller_stopped(
            scheduler_db,
            controller_lock,
            lock_check_python,
            scheduled_task_name,
        )
    configured = _load_accounts(config_path)
    selected: list[tuple[str, dict[str, Any], str]] = []
    for name in account_names:
        row = configured.get(name)
        if not row:
            raise RuntimeError(f"unknown account: {name}")
        if "conda:pyaedt2026v1" not in (row.get("capabilities") or []):
            raise RuntimeError(f"account lacks pyaedt2026v1: {name}")
        env_setup = str(
            (row.get("env_profiles") or {}).get("pyaedt2026v1") or ""
        ).strip()
        if not env_setup:
            raise RuntimeError(f"account has no pyaedt2026v1 setup: {name}")
        selected.append((name, row, env_setup))

    results = []
    for name, row, env_setup in selected:
        client = _connect(row)
        try:
            command = (
                _exact_audit_command(current, env_setup)
                if exact_audit
                else _preflight_command(current, env_setup)
            )
            output = _run(client, command, timeout=45)
        finally:
            client.close()
        if output.splitlines()[-1:] != [current]:
            raise RuntimeError(f"preflight returned the wrong revision for {name}")
        results.append(
            {"account": name, "current": current, "target": target, "ready": True}
        )
    if not execute:
        return results

    attempted: list[tuple[str, dict[str, Any]]] = []
    try:
        for name, row, env_setup in selected:
            client = _connect(row)
            attempted.append((name, row))
            try:
                output = _run(
                    client,
                    _deploy_command(current, target, env_setup),
                    timeout=300,
                )
            finally:
                client.close()
            if output.splitlines()[-1:] != [target]:
                raise RuntimeError(f"deployment returned wrong revision for {name}")
    except Exception as deployment_error:
        rollback_errors = []
        for name, row in reversed(attempted):
            try:
                client = _connect(row)
                try:
                    _run(client, _rollback_command(current), timeout=60)
                finally:
                    client.close()
            except Exception as rollback_error:
                rollback_errors.append(f"{name}: {rollback_error}")
        if rollback_errors:
            raise RuntimeError(
                f"q23 deployment failed ({deployment_error}); rollback also "
                f"failed for {'; '.join(rollback_errors)}"
            ) from deployment_error
        raise
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accounts-config", required=True, type=Path)
    parser.add_argument("--account", action="append")
    parser.add_argument("--expected-current", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--exact-audit", action="store_true")
    parser.add_argument("--q22-controller-stopped", action="store_true")
    parser.add_argument("--scheduler-db", type=Path, default=DEFAULT_SCHEDULER_DB)
    parser.add_argument("--controller-lock", type=Path, default=DEFAULT_CONTROLLER_LOCK)
    parser.add_argument(
        "--lock-check-python", type=Path, default=DEFAULT_LOCK_CHECK_PYTHON
    )
    parser.add_argument("--q22-scheduled-task", default="MFT_Q22_OpenEnded500")
    args = parser.parse_args()
    if args.execute and not args.q22_controller_stopped:
        parser.error("--execute requires --q22-controller-stopped")
    result = deploy(
        args.accounts_config,
        args.account or list(DEFAULT_ACCOUNTS),
        args.expected_current,
        args.target,
        execute=args.execute,
        exact_audit=args.exact_audit,
        q22_controller_stopped=args.q22_controller_stopped,
        scheduler_db=args.scheduler_db,
        controller_lock=args.controller_lock,
        lock_check_python=args.lock_check_python,
        scheduled_task_name=args.q22_scheduled_task,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
