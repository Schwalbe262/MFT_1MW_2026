"""Rolling q23 -> q24 controller for validated async pooled AEDT pipelines.

The scheduler's MFT project count remains the one 500-way source of truth.
Every live q23 task therefore continues to occupy its existing logical slot;
this controller never cancels, resubmits, or rewrites it. Only deficits that
appear after the transition are emitted with the reviewed 8fab610 solver pin
and the scheduler-whitelisted async workload family.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
from typing import Any, Mapping, Sequence

import q22_bounded_soak as engine
import q23_same_node_campaign as q23
from module.core_material_contract import PHYSICS_DATA_REVISION
from regression_260707 import quality_contract


CAMPAIGN_ID = "q24-validated-async-aedt500-260716"
SCHEMA = "q24-validated-async-aedt-rolling-controller-v1"
PREDECESSOR_CAMPAIGN_ID = q23.CAMPAIGN_ID
PREDECESSOR_SOLVER = "092a35bb6e9552fa9c0ef7388c6059606844f2cd"
Q22_PROVEN_RUNTIME_SOLVER = "c7a0c792e2babc74ad1596a6b95b45379a6f903d"
REFILL_SOLVER = "8fab610dfca7180732bd0b38923aa6c71e2129bb"
REFILL_PARENT = "8b1a65ca46509b0fe3fe64420709dea2d15de1a4"
PREDECESSOR_SCHEDULER_PACKAGE = (
    "3febcfa0b803ce4313cc5b8d38f4aa3695af9506"
)
LIBRARY_REVISION = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
COMPATIBILITY_PATH = Path(__file__).with_name(
    "q24_validated_async_aedt_compatibility.json"
)
DEFAULT_ELIGIBLE_ACCOUNTS = q23.DEFAULT_ELIGIBLE_ACCOUNTS
ACTIVE_STATES = ("queued", "attaching", "running")
PREDECESSOR_GENERATION = (
    f"{PREDECESSOR_SOLVER}:{LIBRARY_REVISION}:seed{engine.CANDIDATE_SEED}"
)
REFILL_GENERATION = (
    f"{REFILL_SOLVER}:{LIBRARY_REVISION}:seed{engine.CANDIDATE_SEED}"
)

_ORIGINAL_VERIFY_COMPATIBILITY = q23._ORIGINAL_VERIFY_COMPATIBILITY
_ORIGINAL_RUN_LIVE_GATES = q23._ORIGINAL_RUN_LIVE_GATES
_ORIGINAL_MANIFEST_IDENTITY = q23._ORIGINAL_MANIFEST_IDENTITY
_ORIGINAL_LOAD_OR_CREATE_MANIFEST = q23._ORIGINAL_LOAD_OR_CREATE_MANIFEST
_ORIGINAL_STATIC_PLAN = q23._ORIGINAL_STATIC_PLAN
_RUNTIME_PREFLIGHT_ARGS: argparse.Namespace | None = None


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_q24_pool_and_policy(base_url: str) -> tuple[dict[str, Any], int]:
    """Require the server-side stop-loss to authorize only q24 async MFT."""

    summary, logical_target = q23._verify_q23_pool_and_policy(base_url)
    config = summary.get("config") or {}
    expected_families = ["mft_validated_async"]
    if (
        config.get("native_solve_mode") != "validated_parallel"
        or config.get("parallel_safe_native_solve_families")
        != expected_families
    ):
        raise engine.GateError(
            "q24 scheduler must run validated_parallel with only "
            "mft_validated_async authorized for parallel native solves"
        )
    return summary, logical_target


def _predecessor_manifest_path(state_path: Path) -> Path:
    return state_path.resolve().parent / f"{PREDECESSOR_CAMPAIGN_ID}.manifest.json"


def _validate_predecessor_manifest(
    path: Path,
    state_path: Path,
    eligible_accounts: Sequence[str],
) -> dict[str, Any]:
    if not path.is_file():
        raise engine.GateError(f"q24 predecessor manifest is missing: {path}")
    value = engine._read_json(path)
    immutable = {
        key: item for key, item in value.items()
        if key not in {"identity_sha256", "created_at_epoch", "runtime_control"}
    }
    if value.get("identity_sha256") != engine._digest(immutable):
        raise engine.GateError("q24 predecessor manifest identity drifted")
    expected = {
        "schema": q23.SCHEMA,
        "campaign_id": PREDECESSOR_CAMPAIGN_ID,
        "candidate_seed": engine.CANDIDATE_SEED,
        "solver_revision": PREDECESSOR_SOLVER,
        "proven_runtime_solver_revision": Q22_PROVEN_RUNTIME_SOLVER,
        "library_revision": LIBRARY_REVISION,
        "scheduler_package_revision": PREDECESSOR_SCHEDULER_PACKAGE,
        "physics_data_revision": PHYSICS_DATA_REVISION,
        "state_path": str(state_path.resolve()),
        "eligible_accounts": list(eligible_accounts),
        "max_logical_active": q23.EXPECTED_POOL_TARGET,
    }
    drift = {
        key: (expected_value, value.get(key))
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    if drift:
        raise engine.GateError(
            f"q24 predecessor manifest pins drifted: {drift}"
        )
    expected_topology = {
        "sessions": q23.EXPECTED_POOL_SESSIONS,
        "projects_per_aedt": q23.EXPECTED_PROJECTS_PER_AEDT,
        "capacity": (
            q23.EXPECTED_POOL_SESSIONS * q23.EXPECTED_PROJECTS_PER_AEDT
        ),
        "target": q23.EXPECTED_POOL_TARGET,
        "min_idle_aedt_sessions": q23.EXPECTED_MIN_IDLE_AEDT_SESSIONS,
        "session_base_cpus": q23.AEDT_SESSION_BASE_CPUS,
        "session_reserved_cpus": (
            q23.AEDT_SESSION_BASE_CPUS
            + q23.PROJECT_CPUS * q23.EXPECTED_PROJECTS_PER_AEDT
        ),
    }
    if value.get("pool_topology") != expected_topology:
        raise engine.GateError("q24 predecessor pool topology drifted")
    baseline = value.get("baseline_serial")
    if isinstance(baseline, bool) or not isinstance(baseline, int) or baseline < 0:
        raise engine.GateError("q24 predecessor baseline serial is invalid")
    return value


def _candidate_cursor_transition(
    state_path: Path,
    transition_serial: int,
    *,
    execute: bool,
) -> dict[str, Any]:
    """Seed q24 from q23's next candidate without replaying old designs."""

    state = engine._read_json(state_path)
    serial = state.get("serial")
    if isinstance(serial, bool) or not isinstance(serial, int) or serial < 0:
        raise engine.GateError("q24 feeder state serial is invalid")
    cursors = state.get("candidate_cursors")
    if not isinstance(cursors, dict):
        raise engine.GateError("q24 feeder candidate cursor ledger is missing")
    migrations = state.get("candidate_cursor_migrations") or {}
    if not isinstance(migrations, dict):
        raise engine.GateError("q24 candidate cursor migration ledger is invalid")
    existing = migrations.get(REFILL_GENERATION)
    if existing is not None:
        if not isinstance(existing, dict):
            raise engine.GateError("q24 candidate cursor migration record is invalid")
        expected_fixed = {
            "schema": "solver-successor-cursor-v1",
            "campaign_id": CAMPAIGN_ID,
            "source_generation": PREDECESSOR_GENERATION,
            "replacement_generation": REFILL_GENERATION,
            "transition_serial": int(transition_serial),
            "semantics": "continue-next-candidate-no-replay",
        }
        drift = {
            key: (expected, existing.get(key))
            for key, expected in expected_fixed.items()
            if existing.get(key) != expected
        }
        source_cursor = existing.get("source_cursor")
        replacement_initial_cursor = existing.get("replacement_initial_cursor")
        if (
            drift
            or isinstance(source_cursor, bool)
            or not isinstance(source_cursor, int)
            or source_cursor < 0
            or replacement_initial_cursor != source_cursor
            or cursors.get(PREDECESSOR_GENERATION) != source_cursor
            or not isinstance(cursors.get(REFILL_GENERATION), int)
            or int(cursors[REFILL_GENERATION]) < replacement_initial_cursor
        ):
            raise engine.GateError(
                f"q24 candidate cursor migration drifted: {drift}"
            )
        return dict(existing)

    if serial != int(transition_serial):
        raise engine.GateError(
            "q24 cursor migration requires the stopped transition serial"
        )
    if state.get("candidate_generation") != PREDECESSOR_GENERATION:
        raise engine.GateError(
            "q24 cursor migration source generation is not the q23 solver"
        )
    source_cursor = cursors.get(PREDECESSOR_GENERATION)
    if (
        isinstance(source_cursor, bool)
        or not isinstance(source_cursor, int)
        or source_cursor < 0
        or state.get("candidate_cursor") != source_cursor
    ):
        raise engine.GateError("q24 predecessor candidate cursor is inconsistent")
    if REFILL_GENERATION in cursors:
        raise engine.GateError(
            "q24 replacement cursor exists without an audited migration record"
        )
    transition = {
        "schema": "solver-successor-cursor-v1",
        "campaign_id": CAMPAIGN_ID,
        "source_generation": PREDECESSOR_GENERATION,
        "replacement_generation": REFILL_GENERATION,
        "transition_serial": int(transition_serial),
        "source_cursor": source_cursor,
        "replacement_initial_cursor": source_cursor,
        "semantics": "continue-next-candidate-no-replay",
    }
    if execute:
        if Path(engine.feeder.STATE).resolve() != state_path.resolve():
            raise engine.GateError("q24 configured feeder state path drifted")
        updated = dict(state)
        updated_cursors = dict(cursors)
        updated_cursors[REFILL_GENERATION] = source_cursor
        updated_migrations = dict(migrations)
        updated_migrations[REFILL_GENERATION] = transition
        updated["candidate_cursors"] = updated_cursors
        updated["candidate_cursor_migrations"] = updated_migrations
        engine.feeder.save_state(updated, immediate_permission_fallback=True)
    return transition


def verify_rolling_inventory(db_path: Path) -> dict[str, Any]:
    """Accept only the exact predecessor/replacement pins in live MFT slots."""

    placeholders = ",".join("?" for _ in ACTIVE_STATES)
    try:
        with engine._connect_readonly(db_path) as connection:
            rows = connection.execute(
                f"""
                SELECT id, name, status, dedupe_key
                FROM tasks
                WHERE project = ?
                  AND status IN ({placeholders})
                ORDER BY id
                """,
                (engine.PROJECT, *ACTIVE_STATES),
            ).fetchall()
    except sqlite3.Error as exc:
        raise engine.GateError(f"q24 rolling inventory query failed: {exc}") from exc

    counts: Counter[str] = Counter()
    ids: dict[str, list[int]] = {"predecessor": [], "replacement": []}
    for row in rows:
        dedupe = str(row["dedupe_key"] or "")
        name = str(row["name"] or "")
        status = str(row["status"] or "")
        if f":{PREDECESSOR_SOLVER}:{LIBRARY_REVISION}:" in dedupe \
                and name.startswith(f"mft-camp-s{PREDECESSOR_SOLVER[:7]}-"):
            cohort = "predecessor"
        elif f":{REFILL_SOLVER}:{LIBRARY_REVISION}:" in dedupe \
                and name.startswith(f"mft-camp-s{REFILL_SOLVER[:7]}-"):
            cohort = "replacement"
        else:
            raise engine.GateError(
                "q24 found an unapproved live MFT campaign task: "
                f"id={row['id']} name={name!r}"
            )
        counts[f"{cohort}_{status}"] += 1
        ids[cohort].append(int(row["id"]))
    return {
        "logical_active": len(rows),
        "counts": dict(sorted(counts.items())),
        "predecessor_live": len(ids["predecessor"]),
        "replacement_live": len(ids["replacement"]),
        "predecessor_sample": ids["predecessor"][:20],
        "replacement_sample": ids["replacement"][:20],
        "counting_semantics": (
            "both exact cohorts count toward one MFT project target; "
            "new submissions use replacement only"
        ),
        "cancellations": 0,
    }


def _verify_q24_compatibility(
    repo_root: Path = engine.REPO_ROOT,
    _legacy_manifest_path: Path = engine.COMPATIBILITY_PATH,
) -> dict[str, Any]:
    """Reuse q22 physics proof and verify the exact reviewed async fix."""

    target_solver = engine.CAMPAIGN_SOLVER
    target_proven_runtime = engine.PROVEN_RUNTIME_SOLVER
    target_package = engine.SCHEDULER_PACKAGE_REVISION
    engine.CAMPAIGN_SOLVER = PREDECESSOR_SOLVER
    engine.PROVEN_RUNTIME_SOLVER = Q22_PROVEN_RUNTIME_SOLVER
    engine.SCHEDULER_PACKAGE_REVISION = q23.Q22_RUNTIME_EVIDENCE_PACKAGE
    try:
        predecessor = _ORIGINAL_VERIFY_COMPATIBILITY(
            repo_root, engine.COMPATIBILITY_PATH
        )
    finally:
        engine.CAMPAIGN_SOLVER = target_solver
        engine.PROVEN_RUNTIME_SOLVER = target_proven_runtime
        engine.SCHEDULER_PACKAGE_REVISION = target_package

    evidence = engine._read_json(COMPATIBILITY_PATH)
    expected_fields = {
        "schema": "q24-validated-async-aedt-rolling-compatibility-v1",
        "predecessor_campaign": PREDECESSOR_CAMPAIGN_ID,
        "predecessor_solver_revision": PREDECESSOR_SOLVER,
        "replacement_solver_revision": REFILL_SOLVER,
        "replacement_parent_revision": REFILL_PARENT,
        "library_revision": LIBRARY_REVISION,
        "physics_data_revision": PHYSICS_DATA_REVISION,
    }
    drift = {
        key: (expected, evidence.get(key))
        for key, expected in expected_fields.items()
        if evidence.get(key) != expected
    }
    if drift:
        raise engine.GateError(f"q24 compatibility evidence drifted: {drift}")
    expected_runtime = {
        "workload_family": "mft_validated_async",
        "parallel_native_solve_permits": 3,
        "async_dispatch_settle_seconds": 2,
        "predecessor_family_remains_serialized": True,
    }
    if evidence.get("runtime_contract") != expected_runtime:
        raise engine.GateError("q24 async runtime contract drifted")
    if target_solver != REFILL_SOLVER:
        raise engine.GateError("q24 refill solver pin drifted")
    if PREDECESSOR_SOLVER not in quality_contract.PHYSICS_EQUIVALENT_SOLVER_REVISIONS.get(
        REFILL_SOLVER, frozenset()
    ):
        raise engine.GateError("q24 directional physics approval is absent")

    try:
        parent = engine._git(repo_root, "rev-parse", f"{REFILL_SOLVER}^")
        changed = engine._git(
            repo_root, "diff", "--name-only", REFILL_PARENT, REFILL_SOLVER
        ).splitlines()
        ancestry = subprocess.run(
            [
                "git", "-c", f"safe.directory={repo_root.as_posix()}",
                "-C", str(repo_root), "merge-base", "--is-ancestor",
                PREDECESSOR_SOLVER, REFILL_SOLVER,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=90,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise engine.GateError(f"q24 Git compatibility check failed: {exc}") from exc
    if parent != REFILL_PARENT or ancestry.returncode:
        raise engine.GateError("q24 reviewed replacement ancestry drifted")
    if changed != evidence.get("reviewed_fix_paths"):
        raise engine.GateError("q24 reviewed replacement path set drifted")
    return {
        **predecessor,
        "q24_validated_async_aedt_migration": {
            **evidence,
            "evidence_sha256": _sha256_file(COMPATIBILITY_PATH),
            "scheduler_package_revision": target_package,
        },
    }


def _q24_manifest_identity(
    baseline_serial: int,
    state_path: Path,
    eligible_accounts: Sequence[str],
) -> dict[str, Any]:
    identity = _ORIGINAL_MANIFEST_IDENTITY(
        baseline_serial, state_path, eligible_accounts
    )
    predecessor_path = _predecessor_manifest_path(state_path)
    predecessor = _validate_predecessor_manifest(
        predecessor_path, state_path, eligible_accounts
    )
    if baseline_serial < int(predecessor["baseline_serial"]):
        raise engine.GateError("q24 transition serial precedes q23 baseline")
    identity["migration"] = {
        "kind": "append-only-solver-rolling-replacement",
        "predecessor_campaign": PREDECESSOR_CAMPAIGN_ID,
        "predecessor_manifest_identity_sha256": predecessor["identity_sha256"],
        "predecessor_solver_revision": PREDECESSOR_SOLVER,
        "replacement_solver_revision": REFILL_SOLVER,
        "predecessor_scheduler_package_revision": (
            PREDECESSOR_SCHEDULER_PACKAGE
        ),
        "replacement_scheduler_package_revision": (
            engine.SCHEDULER_PACKAGE_REVISION
        ),
        "transition_serial": int(baseline_serial),
        "compatibility_evidence_sha256": _sha256_file(COMPATIBILITY_PATH),
        "live_counting": "predecessor-plus-replacement-one-project-target",
        "new_refill": "replacement-only",
        "predecessor_task_mutation": "none",
        "cancellation": "none",
        "candidate_cursor_transition": _candidate_cursor_transition(
            state_path,
            baseline_serial,
            execute=False,
        ),
    }
    identity["adoption"]["semantics"] = (
        "adopt-all-q23-live-and-accepted-serials-before-transition; "
        "refill-only-with-validated-async-q24-solver"
    )
    return identity


def _load_or_create_q24_manifest(
    path: Path,
    state_path: Path,
    eligible_accounts: Sequence[str],
    *,
    execute: bool,
    baseline_serial: int | None = None,
) -> dict[str, Any]:
    if not path.exists():
        current_serial = engine._state_serial(state_path)
        if baseline_serial is None or int(baseline_serial) != current_serial:
            raise engine.GateError(
                "q24 transition baseline must equal stopped q23 feeder serial "
                f"{current_serial}"
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
                f"q24 transition dataset is unreadable: {exc}"
            ) from exc
        if current_rows != int(engine.ADOPTED_BASELINE_DATASET_ROWS):
            raise engine.GateError(
                "q24 transition dataset rows must equal stopped q23 row count "
                f"{current_rows}"
            )
        _validate_predecessor_manifest(
            _predecessor_manifest_path(state_path),
            state_path,
            eligible_accounts,
        )
        if execute:
            if _RUNTIME_PREFLIGHT_ARGS is None:
                raise engine.GateError("q24 locked transition preflight is missing")
            engine.audit_remote_packages(
                _RUNTIME_PREFLIGHT_ARGS.accounts_config,
                _RUNTIME_PREFLIGHT_ARGS.eligible_accounts,
                _RUNTIME_PREFLIGHT_ARGS.ssh_audit_python,
            )
            verify_rolling_inventory(_RUNTIME_PREFLIGHT_ARGS.scheduler_db)
        _candidate_cursor_transition(
            state_path,
            int(baseline_serial),
            execute=execute,
        )
    return _ORIGINAL_LOAD_OR_CREATE_MANIFEST(
        path,
        state_path,
        eligible_accounts,
        execute=execute,
        baseline_serial=baseline_serial,
    )


def _q24_pooled_submission(args: argparse.Namespace) -> dict[str, Any]:
    submission = q23._q23_pooled_submission(args)
    environment = dict(submission.get("submission_env") or {})
    environment.update({
        "MFT_CAMPAIGN_MIGRATION_PREDECESSOR": PREDECESSOR_CAMPAIGN_ID,
        "MFT_CAMPAIGN_MIGRATION_PREDECESSOR_SOLVER": PREDECESSOR_SOLVER,
        "MFT_CAMPAIGN_MIGRATION_REPLACEMENT_SOLVER": REFILL_SOLVER,
        "MFT_AEDT_WORKLOAD_FAMILY": "mft_validated_async",
        "MFT_AEDT_ASYNC_DISPATCH_SETTLE_SECONDS": "2",
    })
    return {**submission, "submission_env": environment}


def _run_q24_current_live_gates(args: argparse.Namespace) -> dict[str, Any]:
    """Validate current production state without replaying retired canaries."""

    engine.verify_compatibility(engine.REPO_ROOT, engine.COMPATIBILITY_PATH)
    engine.verify_profile(engine.PROFILE_PATH)
    engine.verify_local_library(args.library_root)
    try:
        deployment = engine.deployment_gate.validate_deployment(
            args.deployment_solver_root,
            engine.CAMPAIGN_SOLVER,
            args.library_root,
            engine.LIBRARY_REVISION,
        )
    except Exception as exc:
        raise engine.GateError(
            f"remote deployment revision gate failed: {exc}"
        ) from exc
    pool, logical_target = engine.verify_pool_and_policy(args.scheduler_url)
    packages = engine.audit_remote_packages(
        args.accounts_config,
        args.eligible_accounts,
        args.ssh_audit_python,
    )
    config = pool.get("config") or {}
    return {
        "deployment": deployment,
        "logical_target": logical_target,
        "pool_validation_passed": bool(config.get("validation_passed")),
        "scheduler_policy": {
            "source": "current-live-config",
            "native_solve_mode": config.get("native_solve_mode"),
            "parallel_safe_native_solve_families": config.get(
                "parallel_safe_native_solve_families"
            ),
        },
        "packages": packages,
    }


def _run_q24_live_gates(args: argparse.Namespace) -> dict[str, Any]:
    gates = _run_q24_current_live_gates(args)
    state_path = args.state_dir / "feeder_state.json"
    predecessor = _validate_predecessor_manifest(
        _predecessor_manifest_path(state_path),
        state_path,
        args.eligible_accounts,
    )
    return {
        **gates,
        "rolling_inventory": verify_rolling_inventory(args.scheduler_db),
        "predecessor_manifest": {
            "campaign": predecessor["campaign_id"],
            "identity_sha256": predecessor["identity_sha256"],
            "solver_revision": predecessor["solver_revision"],
        },
        "no_cancellation_performed": True,
    }


def _q24_static_plan(
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    plan = _ORIGINAL_STATIC_PLAN(args, manifest)
    plan["migration"] = dict(manifest["migration"])
    plan["active_control"]["pool"] = (
        "173 AEDT x 3 projects = 519 capacity; one 500-project target counts "
        "both rolling cohorts"
    )
    plan["execution_requires"] = [
        "q23 controller is stopped but every q23 scheduler task remains untouched",
        "transition baseline equals the stopped feeder serial and dataset row count",
        "all five accounts retain the exact audited scheduler package",
        "replacement solver is an advertised branch head",
        "only replacement solver is emitted after the transition serial",
    ]
    return plan


def configure_engine(
    scheduler_package_revision: str,
    baseline_serial: int,
    baseline_dataset_rows: int,
) -> None:
    q23.configure_engine(
        scheduler_package_revision, baseline_serial, baseline_dataset_rows
    )
    engine.CAMPAIGN_ID = CAMPAIGN_ID
    engine.SCHEMA = SCHEMA
    engine.ACCOUNT_EXPANSION_SCHEMA = f"{SCHEMA}-unsupported-v2"
    engine.LEGACY_SCHEMA = f"{SCHEMA}-unsupported-legacy"
    engine.LEGACY_ACCOUNT_EXPANSION_SCHEMA = f"{SCHEMA}-unsupported-legacy-v2"
    engine.CAMPAIGN_SOLVER = REFILL_SOLVER
    engine.PROVEN_RUNTIME_SOLVER = PREDECESSOR_SOLVER
    engine.LIBRARY_REVISION = LIBRARY_REVISION
    engine.verify_compatibility = _verify_q24_compatibility
    engine.verify_pool_and_policy = _verify_q24_pool_and_policy
    engine.run_live_gates = _run_q24_live_gates
    engine.static_plan = _q24_static_plan
    engine.manifest_identity = _q24_manifest_identity
    engine.load_or_create_manifest = _load_or_create_q24_manifest
    engine.pooled_submission = _q24_pooled_submission


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
        raise engine.GateError("q24 uses only its append-only version-1 manifest")
    configure_engine(
        bootstrap.scheduler_package_revision,
        bootstrap.adopt_baseline_serial,
        bootstrap.adopt_baseline_dataset_rows,
    )
    engine_argv = [
        *forwarded,
        "--manifest-version", "1",
        "--adopt-baseline-serial", str(bootstrap.adopt_baseline_serial),
    ]
    contract_args = engine._parser().parse_args(engine_argv)
    contract_args.eligible_accounts = tuple(
        contract_args.eligible_accounts or engine.DEFAULT_ELIGIBLE_ACCOUNTS
    )
    if contract_args.eligible_accounts != DEFAULT_ELIGIBLE_ACCOUNTS:
        raise engine.GateError("q24 requires the exact audited five-account set")
    _RUNTIME_PREFLIGHT_ARGS = contract_args
    if any(
        item in ("--execute-mft-family-production", "--execute-approved-after-mixed")
        for item in forwarded
    ):
        engine.run_live_gates(contract_args)
    return engine.main(engine_argv)


if __name__ == "__main__":
    raise SystemExit(main())
