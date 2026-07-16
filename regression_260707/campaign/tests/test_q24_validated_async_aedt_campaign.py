from contextlib import ExitStack, nullcontext
import json
import os
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

import pytest


CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
if str(CAMPAIGN_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_DIR))

import q22_bounded_soak as engine
import q23_same_node_campaign as q23
import q24_validated_async_aedt_campaign as q24
from regression_260707 import quality_contract


PACKAGE_SHA = "ffffffffffffffffffffffffffffffffffffffff"
BASELINE_SERIAL = 18_000
BASELINE_ROWS = 5_233
CONFIGURED_NAMES = (
    "CAMPAIGN_ID",
    "SCHEMA",
    "ACCOUNT_EXPANSION_SCHEMA",
    "LEGACY_SCHEMA",
    "LEGACY_ACCOUNT_EXPANSION_SCHEMA",
    "CAMPAIGN_SOLVER",
    "PROVEN_RUNTIME_SOLVER",
    "LIBRARY_REVISION",
    "THIN_CLIENT_CPUS",
    "HOST_CPUS_PER_ATTACHED_LEASE",
    "EXPECTED_SESSION_RESERVED_CPUS",
    "EXPECTED_POOL_SESSIONS",
    "EXPECTED_PROJECTS_PER_AEDT",
    "EXPECTED_POOL_TARGET",
    "EXPECTED_POOL_CAPACITY",
    "POOL_FILL_TIMEOUT_SECONDS",
    "PROFILE_PATH",
    "DEFAULT_ELIGIBLE_ACCOUNTS",
    "ADOPTED_BASELINE_SERIAL",
    "ADOPTED_BASELINE_DATASET_ROWS",
    "SCHEDULER_PACKAGE_REVISION",
    "verify_compatibility",
    "verify_profile",
    "verify_pool_and_policy",
    "run_live_gates",
    "static_plan",
    "manifest_identity",
    "load_or_create_manifest",
    "pooled_submission",
    "verify_owned_serials",
    "execute_cycle",
    "audit_remote_packages",
    "_write_status",
)


@pytest.fixture()
def configured_engine():
    previous = {name: getattr(engine, name) for name in CONFIGURED_NAMES}
    previous_state = engine.feeder.STATE
    previous_active_refill = q24.ACTIVE_REFILL_SOLVER
    q24.configure_engine(PACKAGE_SHA, BASELINE_SERIAL, BASELINE_ROWS)
    try:
        yield engine
    finally:
        q24.ACTIVE_REFILL_SOLVER = previous_active_refill
        engine.feeder.STATE = previous_state
        for name, value in previous.items():
            setattr(engine, name, value)


def _approve_successor(monkeypatch, tmp_path, solver="a" * 40):
    evidence = {
        "schema": "q24-validated-async-aedt-successors-v1",
        "campaign_id": q24.CAMPAIGN_ID,
        "initial_refill_solver_revision": q24.REFILL_SOLVER,
        "library_revision": q24.LIBRARY_REVISION,
        "physics_data_revision": engine.PHYSICS_REVISION,
        "approved_successors": [
            {
                "solver_revision": solver,
                "parent_revision": "b" * 40,
                "cursor_predecessor_solver_revision": q24.REFILL_SOLVER,
                "reviewed_fix_paths": [
                    "regression_260707/test_simulation_stability.py",
                    "run_simulation_260706.py",
                ],
                "physics_effect": "none",
                "runtime_contract": {
                    "workload_family": "mft_validated_async",
                    "parallel_native_solve_permits": 3,
                    "predecessor_family_remains_serialized": True,
                },
            }
        ],
    }
    path = tmp_path / "successors.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    monkeypatch.setattr(q24, "SUCCESSOR_COMPATIBILITY_PATH", path)
    return solver, evidence["approved_successors"][0]


def _state_payload(serial=BASELINE_SERIAL, cursor=5_594):
    return {
        "serial": serial,
        "submitted_samples": 5_058,
        "candidate_generation": q24.PREDECESSOR_GENERATION,
        "candidate_cursor": cursor,
        "candidate_cursors": {q24.PREDECESSOR_GENERATION: cursor},
    }


def _write_state(path, *, serial=BASELINE_SERIAL, cursor=5_594):
    path.write_text(
        json.dumps(_state_payload(serial=serial, cursor=cursor)),
        encoding="utf-8",
    )


def _write_predecessor_manifest(state_path, accounts=q24.DEFAULT_ELIGIBLE_ACCOUNTS):
    identity = {
        "schema": q23.SCHEMA,
        "campaign_id": q23.CAMPAIGN_ID,
        "baseline_serial": 17_785,
        "state_path": str(state_path.resolve()),
        "candidate_seed": engine.CANDIDATE_SEED,
        "solver_revision": q24.PREDECESSOR_SOLVER,
        "proven_runtime_solver_revision": q24.Q22_PROVEN_RUNTIME_SOLVER,
        "library_revision": q24.LIBRARY_REVISION,
        "scheduler_package_revision": q24.PREDECESSOR_SCHEDULER_PACKAGE,
        "physics_data_revision": engine.PHYSICS_REVISION,
        "eligible_accounts": list(accounts),
        "max_logical_active": q23.EXPECTED_POOL_TARGET,
        "pool_topology": {
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
        },
    }
    manifest = {
        **identity,
        "identity_sha256": engine._digest(identity),
        "created_at_epoch": 1.0,
        "runtime_control": {"open_ended": True},
    }
    path = state_path.parent / f"{q23.CAMPAIGN_ID}.manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _task_name(solver, serial):
    return (
        f"mft-camp-s{solver[:7]}-l{q24.LIBRARY_REVISION[:7]}-{serial:05d}"
    )


def _dedupe(solver, serial):
    return (
        f"mft-al:{_task_name(solver, serial)}:{solver}:"
        f"{q24.LIBRARY_REVISION}:digest"
    )


def test_q24_submission_emits_only_validated_async_replacement(configured_engine):
    args = configured_engine._parser().parse_args([])
    args.eligible_accounts = q24.DEFAULT_ELIGIBLE_ACCOUNTS
    submission = configured_engine.pooled_submission(args)
    environment = submission["submission_env"]

    assert configured_engine.CAMPAIGN_SOLVER == q24.REFILL_SOLVER
    assert configured_engine.PROVEN_RUNTIME_SOLVER == q24.PREDECESSOR_SOLVER
    assert submission["cpus"] == 4
    assert submission["account_names"] == q24.DEFAULT_ELIGIBLE_ACCOUNTS
    assert submission["prevalidated_cycle"] is True
    assert submission["scheduler_admission_owns_queueing"] is True
    assert submission["batch_state_commit"] is True
    assert environment["MFT_CAMPAIGN_ID"] == q24.CAMPAIGN_ID
    assert environment["MFT_CAMPAIGN_MIGRATION_PREDECESSOR"] == q23.CAMPAIGN_ID
    assert environment["MFT_CAMPAIGN_MIGRATION_PREDECESSOR_SOLVER"] == (
        q24.PREDECESSOR_SOLVER
    )
    assert environment["MFT_CAMPAIGN_MIGRATION_REPLACEMENT_SOLVER"] == (
        q24.REFILL_SOLVER
    )
    assert environment["MFT_AEDT_WORKLOAD_FAMILY"] == "mft_validated_async"
    assert environment["MFT_AEDT_ASYNC_DISPATCH_SETTLE_SECONDS"] == "2"


def test_q24_refills_exact_deficit_while_resource_queue_is_backed_off(
    configured_engine, tmp_path
):
    args = configured_engine._parser().parse_args([])
    args.eligible_accounts = q24.DEFAULT_ELIGIBLE_ACCOUNTS
    submission = configured_engine.pooled_submission(args)
    state = {"serial": BASELINE_SERIAL, "submitted_samples": 0}
    counts = {"queued": 0, "attaching": 0, "running": 0}
    capacity = {
        "ready_fit_slots": 0,
        "queue_state": "blocked",
        "queue_reason": "allocation backoff active for cpu",
        "queue_submission_allowed": False,
        "project_counts": counts,
        "project_active": 0,
        "project_submission_slots": 3,
        "submission_allowed": False,
    }
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                engine.feeder.scheduler_client,
                "campaign_mutation_lock_is_held",
                return_value=True,
            )
        )
        stack.enter_context(patch.object(engine.feeder, "load_state", return_value=state))
        stack.enter_context(
            patch.object(
                engine.feeder,
                "scheduler_snapshot",
                return_value=(counts, counts, [], capacity),
            )
        )
        stack.enter_context(
            patch.object(engine.feeder, "dataset_collection_snapshot", return_value=(0, set()))
        )
        stack.enter_context(patch.object(engine.feeder, "campaign_inventory", return_value=[]))
        stack.enter_context(
            patch.object(engine.feeder, "cursor_after_valid_candidates", return_value=0)
        )
        stack.enter_context(
            patch.object(
                engine.feeder,
                "next_valid_candidate",
                side_effect=[
                    (1, 0, {"candidate": 1}),
                    (2, 1, {"candidate": 2}),
                    (3, 2, {"candidate": 3}),
                ],
            )
        )
        submit = stack.enter_context(
            patch.object(engine.feeder, "submit", side_effect=[901, 902, 903])
        )
        stack.enter_context(patch.object(engine.feeder, "save_state"))
        stack.enter_context(patch.object(engine.feeder.time, "sleep"))
        assert engine.feeder.step(
            None,
            target=3,
            solver_revision=q24.REFILL_SOLVER,
            library_revision=q24.LIBRARY_REVISION,
            pooled_submission=submission,
        )

    assert submit.call_count == 3
    assert state["serial"] == BASELINE_SERIAL + 3
    assert all(
        call.kwargs["submission_env"]["MFT_AEDT_WORKLOAD_FAMILY"]
        == "mft_validated_async"
        for call in submit.call_args_list
    )


def test_q24_batch_commits_state_once_for_486_tasks(configured_engine):
    args = configured_engine._parser().parse_args([])
    args.eligible_accounts = q24.DEFAULT_ELIGIBLE_ACCOUNTS
    submission = configured_engine.pooled_submission(args)
    state = {"serial": BASELINE_SERIAL, "submitted_samples": 0}
    counts = {"queued": 0, "attaching": 0, "running": 0}
    capacity = {
        "ready_fit_slots": 0,
        "queue_state": "blocked",
        "queue_reason": "scheduler admission owns queueing",
        "project_counts": counts,
        "project_active": 0,
        "project_submission_slots": 486,
        "submission_allowed": False,
    }

    def candidate(cursor, *, seed):
        return cursor + 1, cursor, {"candidate": cursor}

    with ExitStack() as stack:
        stack.enter_context(patch.object(
            engine.feeder.scheduler_client,
            "campaign_mutation_lock_is_held",
            return_value=True,
        ))
        stack.enter_context(patch.object(engine.feeder, "load_state", return_value=state))
        stack.enter_context(patch.object(
            engine.feeder,
            "scheduler_snapshot",
            return_value=(counts, counts, [], capacity),
        ))
        stack.enter_context(patch.object(
            engine.feeder, "dataset_collection_snapshot", return_value=(0, set())
        ))
        stack.enter_context(patch.object(engine.feeder, "campaign_inventory", return_value=[]))
        stack.enter_context(patch.object(
            engine.feeder, "cursor_after_valid_candidates", return_value=0
        ))
        stack.enter_context(patch.object(
            engine.feeder, "next_valid_candidate", side_effect=candidate
        ))
        submit = stack.enter_context(patch.object(
            engine.feeder,
            "submit",
            side_effect=lambda *_args, **_kwargs: 100_000 + int(
                _args[0].rsplit("-", 1)[1]
            ),
        ))
        save = stack.enter_context(patch.object(engine.feeder, "save_state"))
        stack.enter_context(patch.object(engine.feeder.time, "sleep"))

        assert engine.feeder.step(
            None,
            target=486,
            solver_revision=q24.REFILL_SOLVER,
            library_revision=q24.LIBRARY_REVISION,
            pooled_submission=submission,
        )

    assert submit.call_count == 486
    assert save.call_count == 1
    assert state["serial"] == BASELINE_SERIAL + 486


def test_q24_batch_restart_replays_same_names_after_middle_post_failure(
    configured_engine,
):
    args = configured_engine._parser().parse_args([])
    args.eligible_accounts = q24.DEFAULT_ELIGIBLE_ACCOUNTS
    submission = configured_engine.pooled_submission(args)
    state = {"serial": BASELINE_SERIAL, "submitted_samples": 0}
    counts = {"queued": 0, "attaching": 0, "running": 0}
    capacity = {
        "ready_fit_slots": 0,
        "queue_state": "blocked",
        "queue_reason": "scheduler admission owns queueing",
        "project_counts": counts,
        "project_active": 0,
        "project_submission_slots": 3,
        "submission_allowed": False,
    }
    phase = {"value": 1}
    names = {1: [], 2: []}

    def candidate(cursor, *, seed):
        return cursor + 1, cursor, {"candidate": cursor}

    def submit(name, *_args, **_kwargs):
        names[phase["value"]].append(name)
        if phase["value"] == 1 and len(names[1]) == 2:
            raise RuntimeError("middle POST failed")
        return 200_000 + int(name.rsplit("-", 1)[1])

    with ExitStack() as stack:
        stack.enter_context(patch.object(
            engine.feeder.scheduler_client,
            "campaign_mutation_lock_is_held",
            return_value=True,
        ))
        stack.enter_context(patch.object(engine.feeder, "load_state", return_value=state))
        stack.enter_context(patch.object(
            engine.feeder,
            "scheduler_snapshot",
            return_value=(counts, counts, [], capacity),
        ))
        stack.enter_context(patch.object(
            engine.feeder, "dataset_collection_snapshot", return_value=(0, set())
        ))
        stack.enter_context(patch.object(engine.feeder, "campaign_inventory", return_value=[]))
        stack.enter_context(patch.object(
            engine.feeder, "cursor_after_valid_candidates", return_value=0
        ))
        stack.enter_context(patch.object(
            engine.feeder, "next_valid_candidate", side_effect=candidate
        ))
        stack.enter_context(patch.object(engine.feeder, "submit", side_effect=submit))
        save = stack.enter_context(patch.object(engine.feeder, "save_state"))
        stack.enter_context(patch.object(engine.feeder.time, "sleep"))

        with pytest.raises(RuntimeError, match="middle POST failed"):
            engine.feeder.step(
                None,
                target=3,
                solver_revision=q24.REFILL_SOLVER,
                library_revision=q24.LIBRARY_REVISION,
                pooled_submission=submission,
            )
        assert save.call_count == 0
        assert state["serial"] == BASELINE_SERIAL

        phase["value"] = 2
        assert engine.feeder.step(
            None,
            target=3,
            solver_revision=q24.REFILL_SOLVER,
            library_revision=q24.LIBRARY_REVISION,
            pooled_submission=submission,
        )

    assert names[1] == names[2][:2]
    assert len(names[2]) == 3
    assert save.call_count == 1
    assert state["serial"] == BASELINE_SERIAL + 3


def test_non_q24_pooled_refill_retains_per_task_state_commits(configured_engine):
    args = configured_engine._parser().parse_args([])
    args.eligible_accounts = q24.DEFAULT_ELIGIBLE_ACCOUNTS
    submission = q23._q23_pooled_submission(args)
    state = {"serial": BASELINE_SERIAL, "submitted_samples": 0}
    counts = {"queued": 0, "attaching": 0, "running": 0}
    capacity = {
        "ready_fit_slots": 3,
        "queue_state": "ready",
        "queue_reason": "",
        "project_counts": counts,
        "project_active": 0,
        "project_submission_slots": 3,
        "submission_allowed": True,
    }

    with ExitStack() as stack:
        stack.enter_context(patch.object(
            engine.feeder.scheduler_client,
            "campaign_mutation_lock_is_held",
            return_value=True,
        ))
        stack.enter_context(patch.object(engine.feeder, "load_state", return_value=state))
        stack.enter_context(patch.object(
            engine.feeder,
            "scheduler_snapshot",
            return_value=(counts, counts, [], capacity),
        ))
        stack.enter_context(patch.object(
            engine.feeder, "dataset_collection_snapshot", return_value=(0, set())
        ))
        stack.enter_context(patch.object(engine.feeder, "campaign_inventory", return_value=[]))
        stack.enter_context(patch.object(
            engine.feeder, "cursor_after_valid_candidates", return_value=0
        ))
        stack.enter_context(patch.object(
            engine.feeder,
            "next_valid_candidate",
            side_effect=lambda cursor, *, seed: (
                cursor + 1, cursor, {"candidate": cursor}
            ),
        ))
        stack.enter_context(patch.object(
            engine.feeder,
            "submit",
            side_effect=[301, 302, 303],
        ))
        save = stack.enter_context(patch.object(engine.feeder, "save_state"))
        stack.enter_context(patch.object(engine.feeder.time, "sleep"))

        assert engine.feeder.step(
            None,
            target=3,
            solver_revision=q24.PREDECESSOR_SOLVER,
            library_revision=q24.LIBRARY_REVISION,
            pooled_submission=submission,
        )

    assert save.call_count == 3


def test_queueing_override_rejects_non_q24_campaign(configured_engine):
    args = configured_engine._parser().parse_args([])
    args.eligible_accounts = q24.DEFAULT_ELIGIBLE_ACCOUNTS
    submission = configured_engine.pooled_submission(args)
    submission["submission_env"] = {
        **submission["submission_env"],
        "MFT_CAMPAIGN_ID": q23.CAMPAIGN_ID,
    }
    with patch.object(
        engine.feeder.scheduler_client,
        "campaign_mutation_lock_is_held",
        return_value=True,
    ), pytest.raises(engine.feeder.SchedulerError, match="reserved for the exact q24"):
        engine.feeder.step(
            None,
            target=3,
            solver_revision=q24.REFILL_SOLVER,
            library_revision=q24.LIBRARY_REVISION,
            pooled_submission=submission,
        )


def test_q24_uses_last_validated_dataset_snapshot_while_collector_is_locked(
    configured_engine
):
    args = configured_engine._parser().parse_args([])
    args.eligible_accounts = q24.DEFAULT_ELIGIBLE_ACCOUNTS
    submission = configured_engine.pooled_submission(args)
    state = {"serial": BASELINE_SERIAL, "submitted_samples": 0}
    counts = {"queued": 0, "attaching": 0, "running": 0}
    capacity = {
        "ready_fit_slots": 0,
        "queue_state": "blocked",
        "queue_reason": "allocation backoff active for cpu",
        "queue_submission_allowed": False,
        "project_active": 0,
        "project_submission_slots": 1,
        "submission_allowed": False,
    }
    key = (
        os.path.abspath(engine.feeder.TRAIN_PARQUET),
        os.path.abspath(engine.feeder.COLLECT_CACHE),
    )
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                engine.feeder.scheduler_client,
                "campaign_mutation_lock_is_held",
                return_value=True,
            )
        )
        stack.enter_context(patch.object(engine.feeder, "load_state", return_value=state))
        stack.enter_context(
            patch.object(
                engine.feeder,
                "scheduler_snapshot",
                return_value=(counts, counts, [], capacity),
            )
        )
        stack.enter_context(
            patch.object(
                engine.feeder,
                "dataset_collection_snapshot",
                side_effect=engine.feeder.FileLockTimeout("train.parquet.lock"),
            )
        )
        stack.enter_context(
            patch.object(
                engine.feeder,
                "_LAST_DATASET_COLLECTION_SNAPSHOT",
                (key, 5_237, frozenset()),
            )
        )
        stack.enter_context(patch.object(engine.feeder, "campaign_inventory", return_value=[]))
        stack.enter_context(
            patch.object(engine.feeder, "cursor_after_valid_candidates", return_value=0)
        )
        stack.enter_context(
            patch.object(
                engine.feeder,
                "next_valid_candidate",
                return_value=(1, 0, {"candidate": 1}),
            )
        )
        submit = stack.enter_context(patch.object(engine.feeder, "submit", return_value=901))
        stack.enter_context(patch.object(engine.feeder, "save_state"))
        stack.enter_context(patch.object(engine.feeder.time, "sleep"))
        assert engine.feeder.step(
            None,
            target=1,
            solver_revision=q24.REFILL_SOLVER,
            library_revision=q24.LIBRARY_REVISION,
            pooled_submission=submission,
        )

    submit.assert_called_once()


def test_q24_defers_cycle_when_first_dataset_snapshot_is_locked(configured_engine):
    args = configured_engine._parser().parse_args([])
    args.eligible_accounts = q24.DEFAULT_ELIGIBLE_ACCOUNTS
    submission = configured_engine.pooled_submission(args)
    counts = {"queued": 0, "attaching": 0, "running": 0}
    capacity = {
        "ready_fit_slots": 0,
        "queue_state": "blocked",
        "queue_reason": "allocation backoff active for cpu",
        "queue_submission_allowed": False,
        "project_active": 0,
        "project_submission_slots": 1,
        "submission_allowed": False,
    }
    with patch.object(
        engine.feeder.scheduler_client,
        "campaign_mutation_lock_is_held",
        return_value=True,
    ), patch.object(
        engine.feeder,
        "load_state",
        return_value={"serial": BASELINE_SERIAL, "submitted_samples": 0},
    ), patch.object(
        engine.feeder,
        "scheduler_snapshot",
        return_value=(counts, counts, [], capacity),
    ), patch.object(
        engine.feeder,
        "dataset_collection_snapshot",
        side_effect=engine.feeder.FileLockTimeout("train.parquet.lock"),
    ), patch.object(
        engine.feeder,
        "_LAST_DATASET_COLLECTION_SNAPSHOT",
        None,
    ), patch.object(engine.feeder, "submit") as submit:
        assert engine.feeder.step(
            None,
            target=1,
            solver_revision=q24.REFILL_SOLVER,
            library_revision=q24.LIBRARY_REVISION,
            pooled_submission=submission,
        )

    submit.assert_not_called()


def test_q24_status_falls_back_when_atomic_replace_is_denied(
    configured_engine, tmp_path
):
    path = tmp_path / "q24.status.json"
    with patch.object(q24.os, "replace", side_effect=PermissionError("denied")), patch.object(
        q24.time, "sleep"
    ):
        configured_engine._write_status(path, {"phase": "open-ended-refill"})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "phase": "open-ended-refill"
    }
    assert list(tmp_path.glob("*.tmp")) == []


def test_q24_pool_gate_requires_server_side_validated_parallel(monkeypatch):
    summary = {
        "config": {
            "native_solve_mode": "validated_parallel",
            "parallel_safe_native_solve_families": ["mft_validated_async"],
        }
    }
    monkeypatch.setattr(
        q23,
        "_verify_q23_pool_and_policy",
        lambda _url: (summary, 500),
    )

    assert q24._verify_q24_pool_and_policy("http://pool") == (summary, 500)

    summary["config"]["native_solve_mode"] = "serial"
    with pytest.raises(engine.GateError, match="validated_parallel"):
        q24._verify_q24_pool_and_policy("http://pool")


def test_q24_live_gates_use_current_policy_not_retired_q21b_evidence(
    configured_engine, tmp_path
):
    args = configured_engine._parser().parse_args([])
    args.state_dir = tmp_path
    args.eligible_accounts = q24.DEFAULT_ELIGIBLE_ACCOUNTS
    predecessor = _write_predecessor_manifest(tmp_path / "feeder_state.json")
    pool = {
        "config": {
            "validation_passed": True,
            "native_solve_mode": "validated_parallel",
            "parallel_safe_native_solve_families": ["mft_validated_async"],
        }
    }
    with ExitStack() as stack:
        stack.enter_context(patch.object(engine, "verify_compatibility"))
        stack.enter_context(patch.object(engine, "verify_profile"))
        stack.enter_context(patch.object(engine, "verify_local_library"))
        stack.enter_context(
            patch.object(
                engine.deployment_gate,
                "validate_deployment",
                return_value={"solver": q24.REFILL_SOLVER},
            )
        )
        stack.enter_context(
            patch.object(engine, "verify_pool_and_policy", return_value=(pool, 500))
        )
        stack.enter_context(
            patch.object(engine, "audit_remote_packages", return_value=["audited"])
        )
        retired = stack.enter_context(
            patch.object(
                engine,
                "verify_scheduler_evidence",
                side_effect=engine.GateError("stale q21b task 41796"),
            )
        )
        stack.enter_context(
            patch.object(
                q24,
                "verify_rolling_inventory",
                return_value={"logical_active": 500},
            )
        )
        gates = q24._run_q24_live_gates(args)

    retired.assert_not_called()
    assert gates["logical_target"] == 500
    assert gates["pool_validation_passed"] is True
    assert gates["scheduler_policy"] == {
        "source": "current-live-config",
        "native_solve_mode": "validated_parallel",
        "parallel_safe_native_solve_families": ["mft_validated_async"],
    }
    assert gates["packages"] == ["audited"]
    assert gates["predecessor_manifest"]["identity_sha256"] == (
        predecessor["identity_sha256"]
    )


def test_rolling_inventory_counts_both_exact_cohorts_and_rejects_others(tmp_path):
    db_path = tmp_path / "scheduler.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE tasks (id INTEGER, name TEXT, status TEXT, "
            "dedupe_key TEXT, project TEXT)"
        )
        connection.executemany(
            "INSERT INTO tasks VALUES (?, ?, ?, ?, ?)",
            [
                (
                    1,
                    _task_name(q24.PREDECESSOR_SOLVER, 17_999),
                    "queued",
                    _dedupe(q24.PREDECESSOR_SOLVER, 17_999),
                    engine.PROJECT,
                ),
                (
                    2,
                    _task_name(q24.REFILL_SOLVER, 18_001),
                    "running",
                    _dedupe(q24.REFILL_SOLVER, 18_001),
                    engine.PROJECT,
                ),
                (
                    3,
                    _task_name("7768510433858c9056f04320e66819d5fcc90f1a", 18_002),
                    "attaching",
                    _dedupe("7768510433858c9056f04320e66819d5fcc90f1a", 18_002),
                    engine.PROJECT,
                ),
                (
                    4,
                    _task_name(q24.PREDECESSOR_SOLVER, 17_998),
                    "completed",
                    _dedupe(q24.PREDECESSOR_SOLVER, 17_998),
                    engine.PROJECT,
                ),
            ],
        )

    inventory = q24.verify_rolling_inventory(db_path)
    assert inventory["logical_active"] == 3
    assert inventory["predecessor_live"] == 1
    assert inventory["replacement_live"] == 2
    assert inventory["cancellations"] == 0

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "INSERT INTO tasks VALUES (5, 'manual-mft', 'attaching', "
            "'unapproved', ?)",
            (engine.PROJECT,),
        )
    with pytest.raises(engine.GateError, match="unapproved live MFT"):
        q24.verify_rolling_inventory(db_path)


def test_cursor_transition_is_write_free_in_preview_and_idempotent_in_execute(
    configured_engine, tmp_path
):
    state_path = tmp_path / "feeder_state.json"
    _write_state(state_path)
    configured_engine.feeder.STATE = str(state_path)

    preview = q24._candidate_cursor_transition(
        state_path, BASELINE_SERIAL, execute=False
    )
    assert preview["source_cursor"] == 5_594
    assert q24.REFILL_GENERATION not in json.loads(
        state_path.read_text(encoding="utf-8")
    )["candidate_cursors"]

    migrated = q24._candidate_cursor_transition(
        state_path, BASELINE_SERIAL, execute=True
    )
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert migrated == preview
    assert persisted["candidate_generation"] == q24.PREDECESSOR_GENERATION
    assert persisted["candidate_cursor"] == 5_594
    assert persisted["candidate_cursors"][q24.REFILL_GENERATION] == 5_594
    assert q24._candidate_cursor_transition(
        state_path, BASELINE_SERIAL, execute=True
    ) == migrated


def test_cursor_transition_rejects_unproven_replacement_cursor(
    configured_engine, tmp_path
):
    state_path = tmp_path / "feeder_state.json"
    state = _state_payload()
    state["candidate_cursors"][q24.REFILL_GENERATION] = 0
    state_path.write_text(json.dumps(state), encoding="utf-8")
    configured_engine.feeder.STATE = str(state_path)

    with pytest.raises(engine.GateError, match="without an audited migration"):
        q24._candidate_cursor_transition(
            state_path, BASELINE_SERIAL, execute=True
        )


def test_successor_cursor_transition_preserves_high_water_and_forbids_rollback(
    configured_engine, monkeypatch, tmp_path
):
    successor, entry = _approve_successor(monkeypatch, tmp_path)
    q24.configure_engine(
        PACKAGE_SHA, BASELINE_SERIAL, BASELINE_ROWS, successor
    )
    state_path = tmp_path / "feeder_state.json"
    state = {
        "serial": BASELINE_SERIAL + 17,
        "submitted_samples": 5_075,
        "candidate_generation": q24.REFILL_GENERATION,
        "candidate_cursor": 6_123,
        "candidate_cursors": {
            q24.PREDECESSOR_GENERATION: 5_594,
            q24.REFILL_GENERATION: 6_123,
        },
        "candidate_cursor_migrations": {},
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")
    configured_engine.feeder.STATE = str(state_path)

    preview = q24._candidate_successor_transition(
        state_path, successor, execute=False
    )
    target_generation = q24._candidate_generation(successor)
    assert preview["transition_serial"] == BASELINE_SERIAL + 17
    assert preview["source_cursor"] == 6_123
    assert target_generation not in json.loads(
        state_path.read_text(encoding="utf-8")
    )["candidate_cursors"]

    migrated = q24._candidate_successor_transition(
        state_path, successor, execute=True
    )
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert migrated == preview
    assert persisted["serial"] == BASELINE_SERIAL + 17
    assert persisted["candidate_generation"] == q24.REFILL_GENERATION
    assert persisted["candidate_cursor"] == 6_123
    assert persisted["candidate_cursors"][target_generation] == 6_123
    assert migrated["compatibility_entry_sha256"] == engine._digest(entry)

    persisted["candidate_generation"] = target_generation
    persisted["candidate_cursor"] = 6_130
    persisted["candidate_cursors"][target_generation] = 6_130
    state_path.write_text(json.dumps(persisted), encoding="utf-8")
    assert q24._candidate_successor_transition(
        state_path, successor, execute=True
    ) == migrated
    with pytest.raises(engine.GateError, match="rollback is forbidden"):
        q24._candidate_successor_transition(
            state_path, q24.REFILL_SOLVER, execute=True
        )


def test_successor_selection_keeps_existing_q24_manifest_pinned_to_initial_refill(
    configured_engine, monkeypatch, tmp_path
):
    successor, _entry = _approve_successor(monkeypatch, tmp_path)
    q24.configure_engine(
        PACKAGE_SHA, BASELINE_SERIAL, BASELINE_ROWS, successor
    )
    state_path = tmp_path / "feeder_state.json"
    _write_state(state_path)
    _write_predecessor_manifest(state_path)
    configured_engine.feeder.STATE = str(state_path)
    monkeypatch.setattr(configured_engine.feeder, "dataset_row_count", lambda: BASELINE_ROWS)

    manifest = q24._load_or_create_q24_manifest(
        tmp_path / "q24.manifest.json",
        state_path,
        q24.DEFAULT_ELIGIBLE_ACCOUNTS,
        execute=False,
        baseline_serial=BASELINE_SERIAL,
    )

    assert configured_engine.CAMPAIGN_SOLVER == successor
    assert manifest["solver_revision"] == q24.REFILL_SOLVER
    assert manifest["migration"]["replacement_solver_revision"] == q24.REFILL_SOLVER


def test_q24_manifest_adopts_q23_boundary_and_records_no_replay(
    configured_engine, monkeypatch, tmp_path
):
    state_path = tmp_path / "feeder_state.json"
    _write_state(state_path)
    predecessor = _write_predecessor_manifest(state_path)
    configured_engine.feeder.STATE = str(state_path)
    monkeypatch.setattr(configured_engine.feeder, "dataset_row_count", lambda: 5_233)

    manifest = q24._load_or_create_q24_manifest(
        tmp_path / "q24.manifest.json",
        state_path,
        q24.DEFAULT_ELIGIBLE_ACCOUNTS,
        execute=False,
        baseline_serial=BASELINE_SERIAL,
    )

    migration = manifest["migration"]
    assert manifest["baseline_serial"] == BASELINE_SERIAL
    assert manifest["solver_revision"] == q24.REFILL_SOLVER
    assert migration["predecessor_manifest_identity_sha256"] == (
        predecessor["identity_sha256"]
    )
    assert migration["transition_serial"] == BASELINE_SERIAL
    assert migration["predecessor_scheduler_package_revision"] == (
        q24.PREDECESSOR_SCHEDULER_PACKAGE
    )
    assert migration["replacement_scheduler_package_revision"] == PACKAGE_SHA
    assert migration["new_refill"] == "replacement-only"
    assert migration["predecessor_task_mutation"] == "none"
    assert migration["cancellation"] == "none"
    assert migration["candidate_cursor_transition"]["source_cursor"] == 5_594
    assert "adopt-all-q23-live-and-accepted" in manifest["adoption"]["semantics"]


def test_locked_first_manifest_reaudits_before_cursor_write(
    configured_engine, monkeypatch, tmp_path
):
    state_path = tmp_path / "feeder_state.json"
    _write_state(state_path)
    _write_predecessor_manifest(state_path)
    configured_engine.feeder.STATE = str(state_path)
    events = []
    runtime_args = type(
        "Args",
        (),
        {
            "accounts_config": tmp_path / "accounts.yaml",
            "eligible_accounts": q24.DEFAULT_ELIGIBLE_ACCOUNTS,
            "ssh_audit_python": tmp_path / "python.exe",
            "scheduler_db": tmp_path / "scheduler.db",
        },
    )()
    monkeypatch.setattr(q24, "_RUNTIME_PREFLIGHT_ARGS", runtime_args)
    monkeypatch.setattr(configured_engine.feeder, "dataset_row_count", lambda: 5_233)
    monkeypatch.setattr(
        configured_engine,
        "audit_remote_packages",
        lambda *_args: events.append("packages"),
    )
    monkeypatch.setattr(
        q24,
        "verify_rolling_inventory",
        lambda _path: events.append("inventory"),
    )
    monkeypatch.setattr(
        q24,
        "_candidate_cursor_transition",
        lambda *_args, **_kwargs: events.append("cursor") or {},
    )
    monkeypatch.setattr(
        q24,
        "_ORIGINAL_LOAD_OR_CREATE_MANIFEST",
        lambda *_args, **_kwargs: events.append("manifest") or {"ok": True},
    )

    assert q24._load_or_create_q24_manifest(
        tmp_path / "q24.manifest.json",
        state_path,
        q24.DEFAULT_ELIGIBLE_ACCOUNTS,
        execute=True,
        baseline_serial=BASELINE_SERIAL,
    ) == {"ok": True}
    assert events == ["packages", "inventory", "cursor", "manifest"]


def test_q24_compatibility_is_exact_directional_and_restores_q22_pins(
    configured_engine,
):
    evidence = q24._verify_q24_compatibility()
    assert evidence["q24_validated_async_aedt_migration"][
        "replacement_solver_revision"
    ] == q24.REFILL_SOLVER
    assert configured_engine.CAMPAIGN_SOLVER == q24.REFILL_SOLVER
    assert configured_engine.PROVEN_RUNTIME_SOLVER == q24.PREDECESSOR_SOLVER
    assert q24.PREDECESSOR_SOLVER in (
        quality_contract.PHYSICS_EQUIVALENT_SOLVER_REVISIONS[q24.REFILL_SOLVER]
    )
    assert q24.REFILL_SOLVER not in (
        quality_contract.PHYSICS_EQUIVALENT_SOLVER_REVISIONS[
            "26afff8de2936f605783395fbff19d5f1d26b354"
        ]
    )


def test_q24_compatibility_approves_exact_handle_loss_successor(
    configured_engine,
):
    successor = "7768510433858c9056f04320e66819d5fcc90f1a"
    q24.configure_engine(
        PACKAGE_SHA, BASELINE_SERIAL, BASELINE_ROWS, successor
    )

    evidence = q24._verify_q24_compatibility()

    selection = evidence["q24_refill_selection"]
    assert selection["selected_refill_solver_revision"] == successor
    assert selection["successor_rollout"]["parent_revision"] == (
        "d424b75a0830693f9d16a4cbf2485cd9d1733a3c"
    )
    assert selection["successor_rollout"]["reviewed_fix_paths"] == [
        "run_simulation_260706.py",
        "tests/test_simulation_stability.py",
    ]
    assert configured_engine.CAMPAIGN_SOLVER == successor


def test_execute_cycle_adopts_old_slots_but_passes_only_new_pin_to_refill(
    configured_engine, tmp_path
):
    args = configured_engine._parser().parse_args([])
    args.state_dir = tmp_path
    args.eligible_accounts = q24.DEFAULT_ELIGIBLE_ACCOUNTS
    manifest = {
        "baseline_serial": BASELINE_SERIAL,
        "identity_sha256": "identity",
        "eligible_accounts": list(args.eligible_accounts),
    }
    with ExitStack() as stack:
        stack.enter_context(patch.object(engine, "run_live_gates", return_value={}))
        stack.enter_context(
            patch.object(engine.feeder, "campaign_mutation_lock", return_value=nullcontext())
        )
        stack.enter_context(
            patch.object(engine, "verify_pool_and_policy", return_value=({}, 500))
        )
        stack.enter_context(
            patch.object(q24, "_candidate_successor_transition", return_value={})
        )
        stack.enter_context(
            patch.object(engine, "_state_serial", side_effect=[18_000, 18_001])
        )
        stack.enter_context(patch.object(engine, "verify_owned_serials"))
        step = stack.enter_context(patch.object(engine.feeder, "step"))
        status = engine.execute_cycle(args, manifest)

    assert step.call_args.args[0] is None
    assert step.call_args.kwargs["target"] == 500
    assert step.call_args.kwargs["solver_revision"] == q24.REFILL_SOLVER
    assert "max_new_tasks" not in step.call_args.kwargs
    assert status["progress"]["accepted_simulations"] == 1
    assert status["no_cancellation_performed"] is True


def test_execute_cycle_rolls_refill_to_successor_without_changing_target(
    configured_engine, monkeypatch, tmp_path
):
    successor, _entry = _approve_successor(monkeypatch, tmp_path)
    q24.configure_engine(
        PACKAGE_SHA, BASELINE_SERIAL, BASELINE_ROWS, successor
    )
    args = configured_engine._parser().parse_args([])
    args.state_dir = tmp_path
    args.eligible_accounts = q24.DEFAULT_ELIGIBLE_ACCOUNTS
    manifest = {
        "baseline_serial": BASELINE_SERIAL,
        "identity_sha256": "identity",
        "eligible_accounts": list(args.eligible_accounts),
    }
    transition = {"replacement_solver_revision": successor}
    with ExitStack() as stack:
        stack.enter_context(patch.object(engine, "run_live_gates", return_value={}))
        stack.enter_context(
            patch.object(engine.feeder, "campaign_mutation_lock", return_value=nullcontext())
        )
        stack.enter_context(
            patch.object(engine, "verify_pool_and_policy", return_value=({}, 500))
        )
        migrate = stack.enter_context(
            patch.object(
                q24,
                "_candidate_successor_transition",
                return_value=transition,
            )
        )
        stack.enter_context(
            patch.object(engine, "_state_serial", side_effect=[18_010, 18_011])
        )
        stack.enter_context(patch.object(engine, "verify_owned_serials"))
        step = stack.enter_context(patch.object(engine.feeder, "step"))
        status = engine.execute_cycle(args, manifest)

    migrate.assert_called_once_with(
        tmp_path / "feeder_state.json", successor, execute=True
    )
    assert step.call_args.kwargs["target"] == 500
    assert step.call_args.kwargs["solver_revision"] == successor
    assert status["logical_target"] == 500
    assert status["selected_refill_solver_revision"] == successor
    assert status["candidate_cursor_successor_transition"] == transition
    assert status["no_cancellation_performed"] is True


def test_post_transition_ownership_ignores_adopted_old_serial_and_requires_new(
    configured_engine, tmp_path
):
    db_path = tmp_path / "scheduler.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE tasks (name TEXT, dedupe_key TEXT)")
        connection.executemany(
            "INSERT INTO tasks VALUES (?, ?)",
            [
                (
                    _task_name(q24.PREDECESSOR_SOLVER, BASELINE_SERIAL),
                    _dedupe(q24.PREDECESSOR_SOLVER, BASELINE_SERIAL),
                ),
                (
                    _task_name(q24.REFILL_SOLVER, BASELINE_SERIAL + 1),
                    _dedupe(q24.REFILL_SOLVER, BASELINE_SERIAL + 1),
                ),
                (
                    _task_name(
                        "7768510433858c9056f04320e66819d5fcc90f1a",
                        BASELINE_SERIAL + 2,
                    ),
                    _dedupe(
                        "7768510433858c9056f04320e66819d5fcc90f1a",
                        BASELINE_SERIAL + 2,
                    ),
                ),
            ],
        )
    configured_engine.verify_owned_serials(
        db_path, {"baseline_serial": BASELINE_SERIAL}, BASELINE_SERIAL + 2
    )
    with pytest.raises(engine.GateError, match="serial 18003"):
        configured_engine.verify_owned_serials(
            db_path, {"baseline_serial": BASELINE_SERIAL}, BASELINE_SERIAL + 3
        )


def test_q24_main_rejects_manifest_override_and_partial_accounts(monkeypatch):
    common = [
        "--scheduler-package-revision",
        PACKAGE_SHA,
        "--adopt-baseline-serial",
        str(BASELINE_SERIAL),
        "--adopt-baseline-dataset-rows",
        str(BASELINE_ROWS),
    ]
    with pytest.raises(engine.GateError, match="append-only version-1"):
        q24.main([*common, "--manifest-version=2"])

    monkeypatch.setattr(q24, "configure_engine", lambda *_args: None)
    with pytest.raises(engine.GateError, match="exact audited five-account"):
        q24.main([*common, "--eligible-account", "dhj02"])


def test_q24_supervisor_is_open_ended_and_pins_transition_arguments():
    source = Path(q24.__file__).with_name(
        "start_q24_validated_async_aedt_supervisor.ps1"
    ).read_text(encoding="utf-8")
    assert "while ($true)" in source
    assert "q24_validated_async_aedt_campaign.py" in source
    assert "--execute-mft-family-production" in source
    assert "--scheduler-package-revision" in source
    assert "--adopt-baseline-serial" in source
    assert "--adopt-baseline-dataset-rows" in source
