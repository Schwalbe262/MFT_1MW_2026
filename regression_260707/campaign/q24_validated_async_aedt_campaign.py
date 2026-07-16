"""Rolling q23 -> q24 controller for validated async pooled AEDT pipelines.

The scheduler's MFT project count remains the one 500-way source of truth.
Every live predecessor task therefore continues to occupy its existing logical
slot; this controller never cancels, resubmits, or rewrites it. Only deficits
that appear after an append-only solver or scheduler-package transition are
emitted with the selected refill pins and whitelisted async workload family.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time
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
SUCCESSOR_COMPATIBILITY_PATH = Path(__file__).with_name(
    "q24_validated_async_aedt_successors.json"
)
DEFAULT_ELIGIBLE_ACCOUNTS = q23.DEFAULT_ELIGIBLE_ACCOUNTS
ACTIVE_STATES = ("queued", "attaching", "running")
PREDECESSOR_GENERATION = (
    f"{PREDECESSOR_SOLVER}:{LIBRARY_REVISION}:seed{engine.CANDIDATE_SEED}"
)
REFILL_GENERATION = (
    f"{REFILL_SOLVER}:{LIBRARY_REVISION}:seed{engine.CANDIDATE_SEED}"
)
ACTIVE_REFILL_SOLVER = REFILL_SOLVER
IMMUTABLE_Q24_COMPATIBILITY_EVIDENCE_SHA256 = (
    "2021f45c414f697c7516fc75b5fe3423c229f9806c20c6858fb71645d26aa6cb"
)
SCHEDULER_PACKAGE_SUCCESSOR_LEDGER_KEY = (
    "scheduler_package_successor_ledger"
)
SCHEDULER_PACKAGE_SUCCESSOR_SCHEMA = (
    "q24-scheduler-package-successor-v1"
)
MANIFEST_SCHEDULER_PACKAGE_REVISION = engine.SCHEDULER_PACKAGE_REVISION
ACTIVE_SCHEDULER_PACKAGE_REVISION = engine.SCHEDULER_PACKAGE_REVISION

_ORIGINAL_VERIFY_COMPATIBILITY = q23._ORIGINAL_VERIFY_COMPATIBILITY
_ORIGINAL_RUN_LIVE_GATES = q23._ORIGINAL_RUN_LIVE_GATES
_ORIGINAL_MANIFEST_IDENTITY = q23._ORIGINAL_MANIFEST_IDENTITY
_ORIGINAL_LOAD_OR_CREATE_MANIFEST = q23._ORIGINAL_LOAD_OR_CREATE_MANIFEST
_ORIGINAL_STATIC_PLAN = q23._ORIGINAL_STATIC_PLAN
_ORIGINAL_WRITE_STATUS = engine._write_status
_ORIGINAL_VERIFY_OWNED_SERIALS = engine.verify_owned_serials
_ORIGINAL_EXECUTE_CYCLE = engine.execute_cycle
_RUNTIME_PREFLIGHT_ARGS: argparse.Namespace | None = None
STATUS_WRITE_ATTEMPTS = 3


def _candidate_generation(solver_revision: str) -> str:
    return (
        f"{solver_revision}:{LIBRARY_REVISION}:seed{engine.CANDIDATE_SEED}"
    )


def _successor_compatibility() -> dict[str, Any]:
    evidence = engine._read_json(SUCCESSOR_COMPATIBILITY_PATH)
    expected = {
        "schema": "q24-validated-async-aedt-successors-v1",
        "campaign_id": CAMPAIGN_ID,
        "initial_refill_solver_revision": REFILL_SOLVER,
        "library_revision": LIBRARY_REVISION,
        "physics_data_revision": PHYSICS_DATA_REVISION,
    }
    drift = {
        key: (value, evidence.get(key))
        for key, value in expected.items()
        if evidence.get(key) != value
    }
    if drift:
        raise engine.GateError(
            f"q24 successor compatibility header drifted: {drift}"
        )
    successors = evidence.get("approved_successors")
    if not isinstance(successors, list):
        raise engine.GateError("q24 approved successor ledger is invalid")
    seen = {REFILL_SOLVER}
    available_cursor_sources = {REFILL_SOLVER}
    normalized = []
    for raw in successors:
        if not isinstance(raw, dict):
            raise engine.GateError("q24 approved successor record is invalid")
        required_strings = (
            "solver_revision",
            "parent_revision",
            "cursor_predecessor_solver_revision",
            "physics_effect",
        )
        if any(not isinstance(raw.get(key), str) for key in required_strings):
            raise engine.GateError("q24 successor pin fields are invalid")
        solver = raw["solver_revision"]
        parent = raw["parent_revision"]
        cursor_source = raw["cursor_predecessor_solver_revision"]
        if (
            len(solver) != 40
            or len(parent) != 40
            or len(cursor_source) != 40
            or solver in seen
        ):
            raise engine.GateError("q24 successor revision ledger is invalid")
        if cursor_source not in available_cursor_sources:
            raise engine.GateError(
                "q24 successor cursor predecessor is not an earlier approved pin"
            )
        paths = raw.get("reviewed_fix_paths")
        if (
            not isinstance(paths, list)
            or not paths
            or not all(isinstance(path, str) and path for path in paths)
            or len(set(paths)) != len(paths)
            or paths != sorted(paths)
        ):
            raise engine.GateError("q24 successor reviewed path set is invalid")
        if raw["physics_effect"] != "none":
            raise engine.GateError("q24 successor must be physics-neutral")
        if raw.get("runtime_contract") != {
            "workload_family": "mft_validated_async",
            "parallel_native_solve_permits": 3,
            "predecessor_family_remains_serialized": True,
        }:
            raise engine.GateError("q24 successor runtime contract drifted")
        seen.add(solver)
        available_cursor_sources.add(solver)
        normalized.append(dict(raw))
    return {**evidence, "approved_successors": normalized}


def _successor_entries() -> dict[str, dict[str, Any]]:
    return {
        item["solver_revision"]: item
        for item in _successor_compatibility()["approved_successors"]
    }


def _allowed_refill_solvers() -> tuple[str, ...]:
    return (REFILL_SOLVER, *_successor_entries().keys())


def _select_refill_solver(solver_revision: str) -> str:
    if (
        not isinstance(solver_revision, str)
        or len(solver_revision) != 40
        or any(character not in "0123456789abcdef" for character in solver_revision)
    ):
        raise engine.GateError("q24 --refill-solver requires a full lowercase SHA")
    if solver_revision not in _allowed_refill_solvers():
        raise engine.GateError(
            "q24 --refill-solver is absent from the approved successor ledger"
        )
    return solver_revision


def _select_scheduler_package_revision(
    manifest_revision: str,
    successor_revision: str | None,
) -> str:
    if not engine.FULL_SHA.fullmatch(str(manifest_revision or "")):
        raise engine.GateError(
            "q24 immutable scheduler package requires a full lowercase SHA"
        )
    if successor_revision is None:
        return manifest_revision
    if not engine.FULL_SHA.fullmatch(str(successor_revision or "")):
        raise engine.GateError(
            "q24 --scheduler-package-successor requires a full lowercase SHA"
        )
    if successor_revision == manifest_revision:
        raise engine.GateError(
            "q24 scheduler package successor must differ from the immutable "
            "manifest package"
        )
    return successor_revision


def _audit_q24_remote_packages(
    config_path: Path,
    eligible_accounts: Sequence[str],
    audit_python: Path = engine.DEFAULT_SSH_AUDIT_PYTHON,
) -> list[dict[str, Any]]:
    """Audit the selected refill package without changing manifest identity."""

    manifest_revision = engine.SCHEDULER_PACKAGE_REVISION
    if manifest_revision != MANIFEST_SCHEDULER_PACKAGE_REVISION:
        raise engine.GateError("q24 immutable scheduler package pin drifted")
    engine.SCHEDULER_PACKAGE_REVISION = ACTIVE_SCHEDULER_PACKAGE_REVISION
    try:
        rows = q23._audit_q23_remote_packages(
            config_path,
            eligible_accounts,
            audit_python,
        )
    finally:
        engine.SCHEDULER_PACKAGE_REVISION = manifest_revision
    return [
        {
            **row,
            "immutable_manifest_package": MANIFEST_SCHEDULER_PACKAGE_REVISION,
            "selected_refill_package": ACTIVE_SCHEDULER_PACKAGE_REVISION,
        }
        for row in rows
    ]


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


def _candidate_successor_transition(
    state_path: Path,
    target_solver: str,
    *,
    execute: bool,
) -> dict[str, Any]:
    """Move a successor to the current high-water cursor exactly once."""

    target_solver = _select_refill_solver(target_solver)
    state = engine._read_json(state_path)
    cursors = state.get("candidate_cursors")
    migrations = state.get("candidate_cursor_migrations") or {}
    if not isinstance(cursors, dict) or not isinstance(migrations, dict):
        raise engine.GateError("q24 successor cursor ledger is invalid")

    entries = _successor_entries()
    successor_generations = {
        _candidate_generation(solver) for solver in entries
    }
    if target_solver == REFILL_SOLVER:
        if (
            state.get("candidate_generation") in successor_generations
            or any(generation in migrations for generation in successor_generations)
        ):
            raise engine.GateError(
                "q24 refill solver rollback is forbidden after successor migration"
            )
        return {
            "schema": "solver-successor-cursor-v1",
            "campaign_id": CAMPAIGN_ID,
            "replacement_generation": REFILL_GENERATION,
            "selected_solver_revision": REFILL_SOLVER,
            "semantics": "initial-q24-refill-no-successor-rollout",
        }

    entry = entries[target_solver]
    source_solver = entry["cursor_predecessor_solver_revision"]
    source_generation = _candidate_generation(source_solver)
    target_generation = _candidate_generation(target_solver)
    entry_digest = engine._digest(entry)
    existing = migrations.get(target_generation)
    if existing is not None:
        if not isinstance(existing, dict):
            raise engine.GateError(
                "q24 successor cursor migration record is invalid"
            )
        expected_fixed = {
            "schema": "solver-successor-cursor-v1",
            "campaign_id": CAMPAIGN_ID,
            "source_generation": source_generation,
            "replacement_generation": target_generation,
            "source_solver_revision": source_solver,
            "replacement_solver_revision": target_solver,
            "compatibility_entry_sha256": entry_digest,
            "semantics": "continue-next-candidate-no-replay",
        }
        drift = {
            key: (expected, existing.get(key))
            for key, expected in expected_fixed.items()
            if existing.get(key) != expected
        }
        source_cursor = existing.get("source_cursor")
        replacement_initial_cursor = existing.get(
            "replacement_initial_cursor"
        )
        current_generation = state.get("candidate_generation")
        current_cursor = state.get("candidate_cursor")
        if (
            drift
            or isinstance(source_cursor, bool)
            or not isinstance(source_cursor, int)
            or source_cursor < 0
            or replacement_initial_cursor != source_cursor
            or cursors.get(source_generation) != source_cursor
            or not isinstance(cursors.get(target_generation), int)
            or int(cursors[target_generation]) < source_cursor
            or current_generation not in {source_generation, target_generation}
            or current_cursor != cursors.get(current_generation)
        ):
            raise engine.GateError(
                f"q24 successor cursor migration drifted: {drift}"
            )
        return dict(existing)

    current_generation = state.get("candidate_generation")
    source_cursor = cursors.get(source_generation)
    serial = state.get("serial")
    if current_generation != source_generation:
        raise engine.GateError(
            "q24 successor migration source is not the active refill solver"
        )
    if (
        isinstance(serial, bool)
        or not isinstance(serial, int)
        or serial < 0
        or isinstance(source_cursor, bool)
        or not isinstance(source_cursor, int)
        or source_cursor < 0
        or state.get("candidate_cursor") != source_cursor
    ):
        raise engine.GateError("q24 successor source cursor is inconsistent")
    if target_generation in cursors:
        raise engine.GateError(
            "q24 successor cursor exists without an audited migration record"
        )
    transition = {
        "schema": "solver-successor-cursor-v1",
        "campaign_id": CAMPAIGN_ID,
        "source_generation": source_generation,
        "replacement_generation": target_generation,
        "source_solver_revision": source_solver,
        "replacement_solver_revision": target_solver,
        "transition_serial": serial,
        "source_cursor": source_cursor,
        "replacement_initial_cursor": source_cursor,
        "compatibility_entry_sha256": entry_digest,
        "semantics": "continue-next-candidate-no-replay",
    }
    if execute:
        if Path(engine.feeder.STATE).resolve() != state_path.resolve():
            raise engine.GateError("q24 configured feeder state path drifted")
        updated = dict(state)
        updated_cursors = dict(cursors)
        updated_cursors[target_generation] = source_cursor
        updated_migrations = dict(migrations)
        updated_migrations[target_generation] = transition
        updated["candidate_cursors"] = updated_cursors
        updated["candidate_cursor_migrations"] = updated_migrations
        engine.feeder.save_state(updated, immediate_permission_fallback=True)
    return transition


def _scheduler_package_successor_transition(
    state_path: Path,
    manifest: Mapping[str, Any],
    target_revision: str,
    *,
    execute: bool,
) -> dict[str, Any]:
    """Append or adopt one package-only refill transition.

    Package rollout intentionally has no candidate-generation effect. Existing
    tasks retain their persisted submission environment, while only serials
    accepted after this record use the selected package.
    """

    manifest_revision = str(
        manifest.get("scheduler_package_revision")
        or MANIFEST_SCHEDULER_PACKAGE_REVISION
    )
    if manifest_revision != MANIFEST_SCHEDULER_PACKAGE_REVISION:
        raise engine.GateError("q24 immutable scheduler package pin drifted")
    migration = manifest.get("migration")
    if (
        isinstance(migration, Mapping)
        and migration.get("replacement_scheduler_package_revision")
        != manifest_revision
    ):
        raise engine.GateError("q24 manifest package migration pin drifted")
    manifest_identity = manifest.get("identity_sha256")
    if not isinstance(manifest_identity, str) or not manifest_identity:
        raise engine.GateError("q24 manifest identity is invalid")
    baseline_serial = manifest.get("baseline_serial")
    if (
        isinstance(baseline_serial, bool)
        or not isinstance(baseline_serial, int)
        or baseline_serial < 0
    ):
        raise engine.GateError("q24 manifest baseline serial is invalid")
    if not engine.FULL_SHA.fullmatch(str(target_revision or "")):
        raise engine.GateError(
            "q24 selected scheduler package requires a full lowercase SHA"
        )

    state = engine._read_json(state_path)
    state_serial = state.get("serial")
    if (
        isinstance(state_serial, bool)
        or not isinstance(state_serial, int)
        or state_serial < baseline_serial
    ):
        raise engine.GateError("q24 scheduler package ledger serial is invalid")
    raw_ledger = state.get(SCHEDULER_PACKAGE_SUCCESSOR_LEDGER_KEY, [])
    if not isinstance(raw_ledger, list):
        raise engine.GateError("q24 scheduler package successor ledger is invalid")

    ledger: list[dict[str, Any]] = []
    expected_source = manifest_revision
    previous_digest: str | None = None
    previous_serial = baseline_serial
    seen_revisions = {manifest_revision}
    for index, raw_record in enumerate(raw_ledger, start=1):
        if not isinstance(raw_record, dict):
            raise engine.GateError(
                "q24 scheduler package successor record is invalid"
            )
        record = dict(raw_record)
        expected_fixed = {
            "schema": SCHEDULER_PACKAGE_SUCCESSOR_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "manifest_identity_sha256": manifest_identity,
            "sequence": index,
            "predecessor_scheduler_package_revision": expected_source,
            "previous_transition_sha256": previous_digest,
            "semantics": (
                "existing-tasks-retain-package-new-refills-use-replacement"
            ),
            "live_counting": "all-package-cohorts-one-logical-project-target",
            "task_mutation": "none",
            "cancellation": "none",
        }
        drift = {
            key: (expected, record.get(key))
            for key, expected in expected_fixed.items()
            if record.get(key) != expected
        }
        replacement = record.get("replacement_scheduler_package_revision")
        transition_serial = record.get("transition_serial")
        recorded_digest = record.get("transition_sha256")
        digest_payload = {
            key: value
            for key, value in record.items()
            if key != "transition_sha256"
        }
        if (
            drift
            or not engine.FULL_SHA.fullmatch(str(replacement or ""))
            or replacement in seen_revisions
            or isinstance(transition_serial, bool)
            or not isinstance(transition_serial, int)
            or transition_serial < previous_serial
            or transition_serial > state_serial
            or recorded_digest != engine._digest(digest_payload)
        ):
            raise engine.GateError(
                f"q24 scheduler package successor ledger drifted: {drift}"
            )
        ledger.append(record)
        seen_revisions.add(replacement)
        expected_source = replacement
        previous_serial = transition_serial
        previous_digest = recorded_digest

    if target_revision == expected_source:
        if ledger:
            return {
                **ledger[-1],
                "ledger_action": "adopt",
                "ledger_length": len(ledger),
            }
        return {
            "schema": SCHEDULER_PACKAGE_SUCCESSOR_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "manifest_identity_sha256": manifest_identity,
            "selected_scheduler_package_revision": manifest_revision,
            "semantics": "immutable-manifest-package-no-successor-rollout",
            "ledger_action": "none",
            "ledger_length": 0,
        }
    if target_revision in seen_revisions:
        raise engine.GateError(
            "q24 scheduler package rollback is forbidden after successor rollout"
        )

    record_payload = {
        "schema": SCHEDULER_PACKAGE_SUCCESSOR_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "manifest_identity_sha256": manifest_identity,
        "sequence": len(ledger) + 1,
        "predecessor_scheduler_package_revision": expected_source,
        "replacement_scheduler_package_revision": target_revision,
        "transition_serial": state_serial,
        "previous_transition_sha256": previous_digest,
        "semantics": "existing-tasks-retain-package-new-refills-use-replacement",
        "live_counting": "all-package-cohorts-one-logical-project-target",
        "task_mutation": "none",
        "cancellation": "none",
    }
    record = {
        **record_payload,
        "transition_sha256": engine._digest(record_payload),
    }
    action = "preview-append"
    if execute:
        if Path(engine.feeder.STATE).resolve() != state_path.resolve():
            raise engine.GateError("q24 configured feeder state path drifted")
        updated = dict(state)
        updated[SCHEDULER_PACKAGE_SUCCESSOR_LEDGER_KEY] = [*ledger, record]
        engine.feeder.save_state(updated, immediate_permission_fallback=True)
        action = "append"
    return {
        **record,
        "ledger_action": action,
        "ledger_length": len(ledger) + 1,
    }


def verify_rolling_inventory(db_path: Path) -> dict[str, Any]:
    """Accept only the predecessor and append-only approved refill pins."""

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
    cohorts = {
        PREDECESSOR_SOLVER: "predecessor",
        REFILL_SOLVER: "initial_replacement",
        **{
            solver: f"successor_{solver[:7]}"
            for solver in _successor_entries()
        },
    }
    ids: dict[str, list[int]] = {cohort: [] for cohort in cohorts.values()}
    for row in rows:
        dedupe = str(row["dedupe_key"] or "")
        name = str(row["name"] or "")
        status = str(row["status"] or "")
        matched = [
            cohort
            for solver, cohort in cohorts.items()
            if f":{solver}:{LIBRARY_REVISION}:" in dedupe
            and name.startswith(f"mft-camp-s{solver[:7]}-")
        ]
        if len(matched) != 1:
            raise engine.GateError(
                "q24 found an unapproved live MFT campaign task: "
                f"id={row['id']} name={name!r}"
            )
        cohort = matched[0]
        counts[f"{cohort}_{status}"] += 1
        ids[cohort].append(int(row["id"]))
    refill_cohorts = [
        cohort for cohort in ids if cohort != "predecessor"
    ]
    return {
        "logical_active": len(rows),
        "counts": dict(sorted(counts.items())),
        "predecessor_live": len(ids["predecessor"]),
        "replacement_live": sum(len(ids[cohort]) for cohort in refill_cohorts),
        "selected_refill_solver": ACTIVE_REFILL_SOLVER,
        "immutable_manifest_scheduler_package": (
            MANIFEST_SCHEDULER_PACKAGE_REVISION
        ),
        "selected_refill_scheduler_package": (
            ACTIVE_SCHEDULER_PACKAGE_REVISION
        ),
        "selected_refill_live": len(
            ids[cohorts[ACTIVE_REFILL_SOLVER]]
        ),
        "predecessor_sample": ids["predecessor"][:20],
        "replacement_sample": [
            task_id
            for cohort in refill_cohorts
            for task_id in ids[cohort]
        ][:20],
        "counting_semantics": (
            "all exact solver and scheduler-package cohorts count toward one "
            "MFT project target; new submissions use only the explicitly "
            "selected refill solver and scheduler package"
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
    selected_solver = _select_refill_solver(target_solver)
    if PREDECESSOR_SOLVER not in quality_contract.PHYSICS_EQUIVALENT_SOLVER_REVISIONS.get(
        REFILL_SOLVER, frozenset()
    ):
        raise engine.GateError("q24 directional physics approval is absent")

    try:
        initial_parent = engine._git(
            repo_root, "rev-parse", f"{REFILL_SOLVER}^"
        )
        initial_changed = engine._git(
            repo_root, "diff", "--name-only", REFILL_PARENT, REFILL_SOLVER
        ).splitlines()
        initial_ancestry = subprocess.run(
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
    if initial_parent != REFILL_PARENT or initial_ancestry.returncode:
        raise engine.GateError("q24 reviewed replacement ancestry drifted")
    if initial_changed != evidence.get("reviewed_fix_paths"):
        raise engine.GateError("q24 reviewed replacement path set drifted")

    successor_rollout: dict[str, Any] | None = None
    if selected_solver != REFILL_SOLVER:
        entry = _successor_entries()[selected_solver]
        try:
            selected_parent = engine._git(
                repo_root, "rev-parse", f"{selected_solver}^"
            )
            selected_changed = engine._git(
                repo_root,
                "diff",
                "--name-only",
                entry["parent_revision"],
                selected_solver,
            ).splitlines()
            selected_ancestry = subprocess.run(
                [
                    "git", "-c", f"safe.directory={repo_root.as_posix()}",
                    "-C", str(repo_root), "merge-base", "--is-ancestor",
                    REFILL_SOLVER, selected_solver,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=90,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise engine.GateError(
                f"q24 successor Git compatibility check failed: {exc}"
            ) from exc
        if (
            selected_parent != entry["parent_revision"]
            or selected_ancestry.returncode
        ):
            raise engine.GateError("q24 reviewed successor ancestry drifted")
        if selected_changed != entry["reviewed_fix_paths"]:
            raise engine.GateError("q24 reviewed successor path set drifted")
        successor_rollout = {
            **entry,
            "compatibility_entry_sha256": engine._digest(entry),
        }
    return {
        **predecessor,
        "q24_validated_async_aedt_migration": {
            **evidence,
            "evidence_sha256": _sha256_file(COMPATIBILITY_PATH),
            "scheduler_package_revision": target_package,
        },
        "q24_refill_selection": {
            "initial_refill_solver_revision": REFILL_SOLVER,
            "selected_refill_solver_revision": selected_solver,
            "successor_rollout": successor_rollout,
        },
        "q24_scheduler_package_selection": {
            "immutable_manifest_scheduler_package_revision": (
                MANIFEST_SCHEDULER_PACKAGE_REVISION
            ),
            "selected_refill_scheduler_package_revision": (
                ACTIVE_SCHEDULER_PACKAGE_REVISION
            ),
            "manifest_mutation": "none",
            "existing_task_mutation": "none",
            "cancellation": "none",
        },
    }


def _q24_manifest_identity(
    baseline_serial: int,
    state_path: Path,
    eligible_accounts: Sequence[str],
) -> dict[str, Any]:
    # The already-persisted q24 v1 manifest remains pinned to the initial
    # 8fab refill forever. Runtime successor selection is recorded in the
    # cursor migration ledger, not by rewriting this immutable identity.
    active_solver = engine.CAMPAIGN_SOLVER
    engine.CAMPAIGN_SOLVER = REFILL_SOLVER
    try:
        identity = _ORIGINAL_MANIFEST_IDENTITY(
            baseline_serial, state_path, eligible_accounts
        )
    finally:
        engine.CAMPAIGN_SOLVER = active_solver
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
        # This digest is part of the already-persisted q24 manifest identity.
        # Keep it frozen even if the separately audited evidence file gains
        # operational metadata after the campaign starts.
        "compatibility_evidence_sha256": (
            IMMUTABLE_Q24_COMPATIBILITY_EVIDENCE_SHA256
        ),
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
        "MFT_CAMPAIGN_MIGRATION_INITIAL_REPLACEMENT_SOLVER": REFILL_SOLVER,
        "MFT_CAMPAIGN_MIGRATION_REPLACEMENT_SOLVER": ACTIVE_REFILL_SOLVER,
        "MFT_AEDT_WORKLOAD_FAMILY": "mft_validated_async",
        "MFT_AEDT_ASYNC_DISPATCH_SETTLE_SECONDS": "2",
        "MFT_CAMPAIGN_IMMUTABLE_SCHEDULER_PACKAGE_REVISION": (
            MANIFEST_SCHEDULER_PACKAGE_REVISION
        ),
        "MFT_CAMPAIGN_SCHEDULER_PACKAGE_REVISION": (
            ACTIVE_SCHEDULER_PACKAGE_REVISION
        ),
    })
    return {
        **submission,
        "submission_env": environment,
        # Maintain the user-owned logical target even while immediate CPU
        # placement is backlogged. Scheduler/AEDT admission still decides
        # when each accepted task can attach and run.
        "scheduler_admission_owns_queueing": True,
        # Plan deterministic names/cursors first, rely on scheduler dedupe for
        # every POST, then persist the feeder ledger once per refill batch.
        "batch_state_commit": True,
    }


def _verify_q24_owned_serials(
    db_path: Path,
    manifest: Mapping[str, Any],
    current_serial: int,
) -> None:
    """Require one exact approved refill task for every accepted q24 serial."""

    baseline = int(manifest["baseline_serial"])
    if current_serial == baseline:
        return
    by_serial: dict[int, tuple[str, str]] = {}
    try:
        with engine._connect_readonly(db_path) as connection:
            for solver in _allowed_refill_solvers():
                prefix = (
                    f"mft-camp-s{solver[:7]}-l{LIBRARY_REVISION[:7]}-"
                )
                rows = connection.execute(
                    "SELECT name, dedupe_key FROM tasks WHERE name LIKE ?",
                    (prefix + "%",),
                ).fetchall()
                for row in rows:
                    name = str(row["name"] or "")
                    suffix = name[len(prefix):]
                    if not suffix.isdecimal():
                        continue
                    serial = int(suffix)
                    if serial <= baseline or serial > current_serial:
                        continue
                    dedupe = str(row["dedupe_key"] or "")
                    if (
                        f":{solver}:{LIBRARY_REVISION}:" not in dedupe
                        or not dedupe.startswith(f"mft-al:{name}:")
                    ):
                        raise engine.GateError(
                            f"q24 serial {serial} task identity drifted"
                        )
                    if serial in by_serial:
                        raise engine.GateError(
                            f"q24 serial {serial} exists in multiple refill cohorts"
                        )
                    by_serial[serial] = (solver, name)
    except sqlite3.Error as exc:
        raise engine.GateError(f"q24 ownership query failed: {exc}") from exc

    for serial in range(baseline + 1, current_serial + 1):
        if serial not in by_serial:
            raise engine.GateError(
                f"q24 accepted serial {serial} has no exact approved task"
            )


def _execute_q24_cycle(
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Refill 500 with one selected pin and an atomic no-replay handoff."""

    gates = engine.run_live_gates(args)
    state_path = args.state_dir / "feeder_state.json"
    successor_transition: dict[str, Any] | None = None
    package_transition: dict[str, Any] | None = None
    with engine.feeder.campaign_mutation_lock():
        _pool, logical_target = engine.verify_pool_and_policy(args.scheduler_url)
        if logical_target:
            successor_transition = _candidate_successor_transition(
                state_path,
                ACTIVE_REFILL_SOLVER,
                execute=True,
            )
            package_preview = _scheduler_package_successor_transition(
                state_path,
                manifest,
                ACTIVE_SCHEDULER_PACKAGE_REVISION,
                execute=False,
            )
            if package_preview["ledger_action"] == "preview-append":
                engine.audit_remote_packages(
                    args.accounts_config,
                    args.eligible_accounts,
                    args.ssh_audit_python,
                )
            package_transition = _scheduler_package_successor_transition(
                state_path,
                manifest,
                ACTIVE_SCHEDULER_PACKAGE_REVISION,
                execute=True,
            )
        current_serial = engine._state_serial(state_path)
        engine.verify_owned_serials(
            args.scheduler_db, manifest, current_serial
        )
        progress = engine.campaign_progress(manifest, current_serial)
        if logical_target:
            engine.feeder.step(
                None,
                target=logical_target,
                buffer=0,
                solver_revision=ACTIVE_REFILL_SOLVER,
                library_revision=LIBRARY_REVISION,
                candidate_seed=engine.CANDIDATE_SEED,
                pooled_submission=engine.pooled_submission(args),
            )
            current_serial = engine._state_serial(state_path)
            engine.verify_owned_serials(
                args.scheduler_db, manifest, current_serial
            )
            progress = engine.campaign_progress(manifest, current_serial)
        return {
            "schema": SCHEMA,
            "campaign": CAMPAIGN_ID,
            "manifest": {
                "version": int(manifest.get("manifest_version") or 1),
                "identity_sha256": manifest["identity_sha256"],
                "eligible_accounts": list(manifest["eligible_accounts"]),
                "initial_refill_solver_revision": REFILL_SOLVER,
                "immutable_scheduler_package_revision": (
                    MANIFEST_SCHEDULER_PACKAGE_REVISION
                ),
            },
            "phase": (
                "stop-requested-draining"
                if not logical_target
                else "open-ended-refill"
            ),
            "updated_at_epoch": time.time(),
            "progress": progress,
            "logical_target": logical_target,
            "open_ended": True,
            "submission_ceiling": None,
            "completion_and_failure_refill": True,
            "selected_refill_solver_revision": ACTIVE_REFILL_SOLVER,
            "selected_refill_scheduler_package_revision": (
                ACTIVE_SCHEDULER_PACKAGE_REVISION
            ),
            "candidate_cursor_successor_transition": successor_transition,
            "scheduler_package_successor_transition": package_transition,
            "gates": gates,
            "no_cancellation_performed": True,
        }


def _write_q24_status(path: Path, payload: Mapping[str, Any]) -> None:
    """Keep control alive when RaiDrive temporarily denies atomic replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(payload), indent=2, sort_keys=True)
    last_error: OSError | None = None
    for attempt in range(STATUS_WRITE_ATTEMPTS):
        staged = path.with_name(
            f".{path.name}.{os.getpid()}.{time.time_ns()}.{attempt}.tmp"
        )
        try:
            with staged.open("w", encoding="utf-8") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(staged, path)
            return
        except OSError as exc:
            last_error = exc
            try:
                staged.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt + 1 < STATUS_WRITE_ATTEMPTS:
                time.sleep(0.1 * (attempt + 1))

    try:
        with path.open("w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        print(
            "[q24] status write deferred; retaining the previous status: "
            f"atomic={last_error!r}, direct={exc!r}",
            flush=True,
        )


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
    plan["refill_rollout"] = {
        "immutable_manifest_solver": REFILL_SOLVER,
        "selected_refill_solver": ACTIVE_REFILL_SOLVER,
        "immutable_manifest_scheduler_package": (
            MANIFEST_SCHEDULER_PACKAGE_REVISION
        ),
        "selected_refill_scheduler_package": (
            ACTIVE_SCHEDULER_PACKAGE_REVISION
        ),
        "cursor_transition": "atomic-high-water-no-replay",
        "scheduler_package_transition": "append-only-state-ledger",
        "live_task_mutation": "none",
        "cancellation": "none",
    }
    plan["active_control"]["pool"] = (
        "173 AEDT x 3 projects = 519 capacity; one 500-project target counts "
        "both rolling cohorts"
    )
    plan["execution_requires"] = [
        "q23 controller is stopped but every q23 scheduler task remains untouched",
        "transition baseline equals the stopped feeder serial and dataset row count",
        "all five accounts expose the exact selected refill scheduler package",
        "replacement solver is an advertised branch head",
        "only selected solver/package pins are emitted after their transitions",
    ]
    return plan


def configure_engine(
    scheduler_package_revision: str,
    baseline_serial: int,
    baseline_dataset_rows: int,
    refill_solver: str = REFILL_SOLVER,
    scheduler_package_successor: str | None = None,
) -> None:
    global ACTIVE_REFILL_SOLVER
    global ACTIVE_SCHEDULER_PACKAGE_REVISION
    global MANIFEST_SCHEDULER_PACKAGE_REVISION
    ACTIVE_REFILL_SOLVER = _select_refill_solver(refill_solver)
    selected_scheduler_package = _select_scheduler_package_revision(
        scheduler_package_revision,
        scheduler_package_successor,
    )
    q23.configure_engine(
        scheduler_package_revision, baseline_serial, baseline_dataset_rows
    )
    MANIFEST_SCHEDULER_PACKAGE_REVISION = scheduler_package_revision
    ACTIVE_SCHEDULER_PACKAGE_REVISION = selected_scheduler_package
    engine.CAMPAIGN_ID = CAMPAIGN_ID
    engine.SCHEMA = SCHEMA
    engine.ACCOUNT_EXPANSION_SCHEMA = f"{SCHEMA}-unsupported-v2"
    engine.LEGACY_SCHEMA = f"{SCHEMA}-unsupported-legacy"
    engine.LEGACY_ACCOUNT_EXPANSION_SCHEMA = f"{SCHEMA}-unsupported-legacy-v2"
    engine.CAMPAIGN_SOLVER = ACTIVE_REFILL_SOLVER
    engine.PROVEN_RUNTIME_SOLVER = PREDECESSOR_SOLVER
    engine.LIBRARY_REVISION = LIBRARY_REVISION
    engine.verify_compatibility = _verify_q24_compatibility
    engine.verify_pool_and_policy = _verify_q24_pool_and_policy
    engine.run_live_gates = _run_q24_live_gates
    engine.static_plan = _q24_static_plan
    engine.manifest_identity = _q24_manifest_identity
    engine.load_or_create_manifest = _load_or_create_q24_manifest
    engine.pooled_submission = _q24_pooled_submission
    engine.verify_owned_serials = _verify_q24_owned_serials
    engine.execute_cycle = _execute_q24_cycle
    engine.audit_remote_packages = _audit_q24_remote_packages
    engine._write_status = _write_q24_status


def _bootstrap_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--scheduler-package-revision", required=True)
    parser.add_argument("--adopt-baseline-serial", required=True, type=int)
    parser.add_argument("--adopt-baseline-dataset-rows", required=True, type=int)
    parser.add_argument("--refill-solver", default=REFILL_SOLVER)
    parser.add_argument("--scheduler-package-successor")
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
        bootstrap.refill_solver,
        bootstrap.scheduler_package_successor,
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
