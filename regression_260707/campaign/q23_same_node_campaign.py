"""Clean q23 same-node controller built on the proven q22 refill engine.

Q23 deliberately uses a new campaign/manifest identity.  It refuses to submit
while any q22 task or lease is live, requires the exact scheduler client commit
at runtime, and pins the current feeder serial as its clean provenance boundary.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import q22_bounded_soak as engine


CAMPAIGN_ID = "q23-same-node500-260716"
SCHEMA = "q23-same-node-open-ended-controller-v1"
PREDECESSOR_CAMPAIGN_ID = "q22-bounded-soak500-260716"
Q22_RUNTIME_EVIDENCE_PACKAGE = "9150e7fa7f72fdf00fb8113e157398b410833c40"
EXPECTED_POOL_SESSIONS = 170
EXPECTED_MIN_IDLE_AEDT_SESSIONS = 3
EXPECTED_PROJECTS_PER_AEDT = 3
EXPECTED_POOL_TARGET = 500
PROJECT_CPUS = 4
POOL_FILL_TIMEOUT_SECONDS = 7_200
DEFAULT_ELIGIBLE_ACCOUNTS = (
    "dhj02",
    "harry261",
    "jji0930",
    "dw16",
    "r1jae262",
)
PROFILE_PATH = (
    Path(__file__).resolve().parents[1]
    / "verify"
    / "profiles"
    / "q23_same_node_full.json"
)

_ORIGINAL_VERIFY_COMPATIBILITY = engine.verify_compatibility
_ORIGINAL_VERIFY_PROFILE = engine.verify_profile
_ORIGINAL_VERIFY_POOL_AND_POLICY = engine.verify_pool_and_policy
_ORIGINAL_RUN_LIVE_GATES = engine.run_live_gates
_ORIGINAL_STATIC_PLAN = engine.static_plan
_ORIGINAL_MANIFEST_IDENTITY = engine.manifest_identity
_ORIGINAL_LOAD_OR_CREATE_MANIFEST = engine.load_or_create_manifest
_RUNTIME_PREFLIGHT_ARGS: argparse.Namespace | None = None


def verify_q22_retired(db_path: Path) -> dict[str, Any]:
    """Fail closed until every predecessor task is terminal."""

    try:
        with engine._connect_readonly(db_path) as connection:
            task_rows = connection.execute(
                """
                SELECT id, status FROM tasks
                WHERE project = ?
                  AND status IN ('queued','attaching','running')
                  AND command LIKE ?
                ORDER BY id
                """,
                (engine.PROJECT, f"%{PREDECESSOR_CAMPAIGN_ID}%"),
            ).fetchall()
            placeholders = ",".join("?" for _ in engine.LIVE_LEASE_STATES)
            lease_rows = connection.execute(
                f"""
                SELECT lease.id, lease.state, lease.task_id, lease.session_id
                FROM aedt_project_leases AS lease
                JOIN tasks AS task ON task.id = lease.task_id
                WHERE task.project = ?
                  AND task.command LIKE ?
                  AND lease.state IN ({placeholders})
                ORDER BY lease.id
                """,
                (
                    engine.PROJECT,
                    f"%{PREDECESSOR_CAMPAIGN_ID}%",
                    *engine.LIVE_LEASE_STATES,
                ),
            ).fetchall()
    except sqlite3.Error as exc:
        raise engine.GateError(f"q23 predecessor query failed: {exc}") from exc
    if task_rows or lease_rows:
        task_sample = [int(row["id"]) for row in task_rows[:20]]
        lease_sample = [int(row["id"]) for row in lease_rows[:20]]
        raise engine.GateError(
            "q23 clean boundary requires every q22 task and lease to be "
            "terminal; "
            f"live_tasks={len(task_rows)} task_sample={task_sample} "
            f"live_leases={len(lease_rows)} lease_sample={lease_sample}"
        )
    return {
        "predecessor_campaign": PREDECESSOR_CAMPAIGN_ID,
        "live_tasks": 0,
        "live_leases": 0,
        "boundary": "clean-no-q22-live-task-or-lease",
    }


def _audit_q23_remote_packages(
    config_path: Path,
    eligible_accounts: Sequence[str],
    audit_python: Path = engine.DEFAULT_SSH_AUDIT_PYTHON,
) -> list[dict[str, Any]]:
    """Audit the exact SHA and the 7200-second client ceiling on every account."""

    if not audit_python.is_file():
        raise engine.GateError(f"q23 audit Python is missing: {audit_python}")
    command = [
        str(audit_python),
        str(Path(__file__).with_name("q23_remote_package_deploy.py")),
        "--accounts-config",
        str(config_path),
        "--expected-current",
        engine.SCHEDULER_PACKAGE_REVISION,
        "--target",
        engine.SCHEDULER_PACKAGE_REVISION,
        "--exact-audit",
    ]
    for account in eligible_accounts:
        command.extend(["--account", account])
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=max(90, 60 * len(eligible_accounts)),
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise engine.GateError(
            f"q23 exact remote package audit could not run: {exc}"
        ) from exc
    if result.returncode:
        raise engine.GateError(
            f"q23 exact remote package audit failed: {result.stdout.strip()}"
        )
    try:
        payload = json.loads(result.stdout)
    except ValueError as exc:
        raise engine.GateError(
            "q23 exact remote package audit returned invalid JSON"
        ) from exc
    expected_accounts = list(eligible_accounts)
    if not isinstance(payload, list) or any(
        not isinstance(row, dict) for row in payload
    ):
        raise engine.GateError(
            "q23 exact remote package audit returned incomplete evidence"
        )
    observed_accounts = [row.get("account") for row in payload]
    if observed_accounts != expected_accounts or any(
        row.get("current") != engine.SCHEDULER_PACKAGE_REVISION
        or row.get("ready") is not True
        for row in payload
    ):
        raise engine.GateError(
            "q23 exact remote package audit returned incomplete evidence"
        )
    return [
        {
            "account": row["account"],
            "package": row["current"],
            "max_pool_fill_timeout_seconds": POOL_FILL_TIMEOUT_SECONDS,
        }
        for row in payload
    ]


def _verify_q23_compatibility(
    repo_root: Path = engine.REPO_ROOT,
    manifest_path: Path = engine.COMPATIBILITY_PATH,
) -> dict[str, Any]:
    """Reuse the physics proof while keeping the q23 package pin independent."""

    target_package = engine.SCHEDULER_PACKAGE_REVISION
    engine.SCHEDULER_PACKAGE_REVISION = Q22_RUNTIME_EVIDENCE_PACKAGE
    try:
        evidence = _ORIGINAL_VERIFY_COMPATIBILITY(repo_root, manifest_path)
    finally:
        engine.SCHEDULER_PACKAGE_REVISION = target_package
    return {
        **evidence,
        "q23_scheduler_package": {
            "runtime_evidence_predecessor": Q22_RUNTIME_EVIDENCE_PACKAGE,
            "exact_same_node_package": target_package,
            "physics_effect": "none",
        },
    }


def _verify_q23_profile(path: Path = PROFILE_PATH) -> dict[str, Any]:
    return _ORIGINAL_VERIFY_PROFILE(path)


def _verify_q23_pool_and_policy(base_url: str) -> tuple[dict[str, Any], int]:
    summary, logical_target = _ORIGINAL_VERIFY_POOL_AND_POLICY(base_url)
    config = summary.get("config") or {}
    if config.get("min_idle_aedt_sessions") != EXPECTED_MIN_IDLE_AEDT_SESSIONS:
        raise engine.GateError(
            "q23 AEDT pool min_idle_aedt_sessions must remain "
            f"{EXPECTED_MIN_IDLE_AEDT_SESSIONS}"
        )
    return summary, logical_target


def _run_q23_live_gates(args: argparse.Namespace) -> dict[str, Any]:
    gates = _ORIGINAL_RUN_LIVE_GATES(args)
    return {
        **gates,
        "clean_boundary": verify_q22_retired(args.scheduler_db),
    }


def _q23_manifest_identity(
    baseline_serial: int,
    state_path: Path,
    eligible_accounts: Sequence[str],
) -> dict[str, Any]:
    identity = _ORIGINAL_MANIFEST_IDENTITY(
        baseline_serial, state_path, eligible_accounts
    )
    identity["pool_topology"]["min_idle_aedt_sessions"] = (
        EXPECTED_MIN_IDLE_AEDT_SESSIONS
    )
    identity["timeouts"]["pool_fill"] = POOL_FILL_TIMEOUT_SECONDS
    identity["adoption"]["semantics"] = (
        "clean-q23-boundary-no-q22-live-task-no-replay-open-ended-refill"
    )
    return identity


def _load_or_create_q23_manifest(
    path: Path,
    state_path: Path,
    eligible_accounts: Sequence[str],
    *,
    execute: bool,
    baseline_serial: int | None = None,
) -> dict[str, Any]:
    """Require an exact current baseline when q23 identity is first created."""

    if not path.exists():
        current_serial = engine._state_serial(state_path)
        if baseline_serial is None or int(baseline_serial) != current_serial:
            raise engine.GateError(
                "q23 first-launch baseline serial must equal the current feeder "
                f"serial {current_serial}"
            )
        try:
            current_rows = int(engine.feeder.dataset_row_count())
        except (
            engine.feeder.SchedulerError,
            engine.FileLockTimeout,
            OSError,
            TypeError,
            ValueError,
        ) as exc:
            raise engine.GateError(
                f"q23 baseline dataset is unreadable: {exc}"
            ) from exc
        if current_rows != engine.ADOPTED_BASELINE_DATASET_ROWS:
            raise engine.GateError(
                "q23 first-launch baseline dataset rows must equal the current "
                f"row count {current_rows}"
            )
        if execute:
            if _RUNTIME_PREFLIGHT_ARGS is None:
                raise engine.GateError(
                    "q23 locked first-launch preflight context is missing"
                )
            engine.audit_remote_packages(
                _RUNTIME_PREFLIGHT_ARGS.accounts_config,
                _RUNTIME_PREFLIGHT_ARGS.eligible_accounts,
                _RUNTIME_PREFLIGHT_ARGS.ssh_audit_python,
            )
            verify_q22_retired(_RUNTIME_PREFLIGHT_ARGS.scheduler_db)
    return _ORIGINAL_LOAD_OR_CREATE_MANIFEST(
        path,
        state_path,
        eligible_accounts,
        execute=execute,
        baseline_serial=baseline_serial,
    )


def _q23_static_plan(
    args: argparse.Namespace, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    plan = _ORIGINAL_STATIC_PLAN(args, manifest)
    plan["active_control"]["pool"] = (
        "170 AEDT x 3 projects = 510 capacity; 3 idle sessions; target <= 500"
    )
    plan["execution_requires"] = [
        "all q22 predecessor tasks are terminal and its controller remains stopped",
        "pool remains 170x3 with min_idle_aedt_sessions=3 and target 500",
        "each same-node project requests and accounts 4 CPUs",
        "solve-permit cohort fill timeout remains 7200 seconds",
        "all five eligible accounts have the exact runtime scheduler package",
        "solver/library revisions remain exact and physics-compatible",
    ]
    return plan


def configure_engine(
    scheduler_package_revision: str,
    baseline_serial: int,
    baseline_dataset_rows: int,
) -> None:
    if not engine.FULL_SHA.fullmatch(scheduler_package_revision):
        raise engine.GateError(
            "q23 scheduler package revision must be a full lowercase commit SHA"
        )
    if baseline_serial < 0 or baseline_dataset_rows < 0:
        raise engine.GateError("q23 clean baseline values must be non-negative")

    engine.CAMPAIGN_ID = CAMPAIGN_ID
    engine.SCHEMA = SCHEMA
    engine.ACCOUNT_EXPANSION_SCHEMA = f"{SCHEMA}-account-expansion-v1"
    engine.LEGACY_SCHEMA = f"{SCHEMA}-unsupported-legacy"
    engine.LEGACY_ACCOUNT_EXPANSION_SCHEMA = f"{SCHEMA}-unsupported-legacy-v2"
    engine.THIN_CLIENT_CPUS = PROJECT_CPUS
    engine.HOST_CPUS_PER_ATTACHED_LEASE = PROJECT_CPUS
    engine.EXPECTED_SESSION_RESERVED_CPUS = (
        PROJECT_CPUS * EXPECTED_PROJECTS_PER_AEDT
    )
    engine.EXPECTED_POOL_SESSIONS = EXPECTED_POOL_SESSIONS
    engine.EXPECTED_PROJECTS_PER_AEDT = EXPECTED_PROJECTS_PER_AEDT
    engine.EXPECTED_POOL_TARGET = EXPECTED_POOL_TARGET
    engine.EXPECTED_POOL_CAPACITY = (
        EXPECTED_POOL_SESSIONS * EXPECTED_PROJECTS_PER_AEDT
    )
    engine.POOL_FILL_TIMEOUT_SECONDS = POOL_FILL_TIMEOUT_SECONDS
    engine.PROFILE_PATH = PROFILE_PATH
    engine.DEFAULT_ELIGIBLE_ACCOUNTS = DEFAULT_ELIGIBLE_ACCOUNTS
    engine.ADOPTED_BASELINE_SERIAL = int(baseline_serial)
    engine.ADOPTED_BASELINE_DATASET_ROWS = int(baseline_dataset_rows)
    engine.SCHEDULER_PACKAGE_REVISION = scheduler_package_revision
    engine.verify_compatibility = _verify_q23_compatibility
    engine.verify_profile = _verify_q23_profile
    engine.verify_pool_and_policy = _verify_q23_pool_and_policy
    engine.run_live_gates = _run_q23_live_gates
    engine.static_plan = _q23_static_plan
    engine.manifest_identity = _q23_manifest_identity
    engine.load_or_create_manifest = _load_or_create_q23_manifest
    engine.audit_remote_packages = _audit_q23_remote_packages


def _bootstrap_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--scheduler-package-revision", required=True)
    parser.add_argument("--adopt-baseline-serial", required=True, type=int)
    parser.add_argument("--adopt-baseline-dataset-rows", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    global _RUNTIME_PREFLIGHT_ARGS
    bootstrap, forwarded = _bootstrap_parser().parse_known_args(argv)
    if any(
        item == "--manifest-version" or item.startswith("--manifest-version=")
        for item in forwarded
    ):
        raise engine.GateError("q23 uses only its clean version-1 manifest")
    configure_engine(
        bootstrap.scheduler_package_revision,
        bootstrap.adopt_baseline_serial,
        bootstrap.adopt_baseline_dataset_rows,
    )
    engine_argv = [
        *forwarded,
        "--manifest-version",
        "1",
        "--adopt-baseline-serial",
        str(bootstrap.adopt_baseline_serial),
    ]
    contract_args = engine._parser().parse_args(engine_argv)
    contract_args.eligible_accounts = tuple(
        contract_args.eligible_accounts or engine.DEFAULT_ELIGIBLE_ACCOUNTS
    )
    if contract_args.eligible_accounts != DEFAULT_ELIGIBLE_ACCOUNTS:
        raise engine.GateError(
            "q23 requires the exact audited five-account placement set in "
            "its pinned order"
        )
    _RUNTIME_PREFLIGHT_ARGS = contract_args
    # The q22 engine creates its immutable manifest before its first controller
    # cycle.  Q23 deliberately performs the complete read-only gate set first,
    # so all five package audits and the predecessor boundary precede any write.
    if any(
        item in ("--execute-mft-family-production", "--execute-approved-after-mixed")
        for item in forwarded
    ):
        engine.run_live_gates(contract_args)
    return engine.main(engine_argv)


if __name__ == "__main__":
    raise SystemExit(main())
