from contextlib import ExitStack, nullcontext
import json
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
    "audit_remote_packages",
)


@pytest.fixture()
def configured_engine():
    previous = {name: getattr(engine, name) for name in CONFIGURED_NAMES}
    previous_state = engine.feeder.STATE
    q24.configure_engine(PACKAGE_SHA, BASELINE_SERIAL, BASELINE_ROWS)
    try:
        yield engine
    finally:
        engine.feeder.STATE = previous_state
        for name, value in previous.items():
            setattr(engine, name, value)


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
                    _task_name(q24.PREDECESSOR_SOLVER, 17_998),
                    "completed",
                    _dedupe(q24.PREDECESSOR_SOLVER, 17_998),
                    engine.PROJECT,
                ),
            ],
        )

    inventory = q24.verify_rolling_inventory(db_path)
    assert inventory["logical_active"] == 2
    assert inventory["predecessor_live"] == 1
    assert inventory["replacement_live"] == 1
    assert inventory["cancellations"] == 0

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "INSERT INTO tasks VALUES (4, 'manual-mft', 'attaching', "
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
            ],
        )
    configured_engine.verify_owned_serials(
        db_path, {"baseline_serial": BASELINE_SERIAL}, BASELINE_SERIAL + 1
    )
    with pytest.raises(engine.GateError, match="serial 18002"):
        configured_engine.verify_owned_serials(
            db_path, {"baseline_serial": BASELINE_SERIAL}, BASELINE_SERIAL + 2
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
