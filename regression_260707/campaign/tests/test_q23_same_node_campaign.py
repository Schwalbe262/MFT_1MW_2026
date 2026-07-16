import json
from pathlib import Path
import sqlite3
import sys

import pytest


CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
if str(CAMPAIGN_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_DIR))

import q22_bounded_soak as engine
import q23_remote_package_deploy as package_deploy
import q23_same_node_campaign as q23


PACKAGE_SHA = "a" * 40
CONFIGURED_NAMES = (
    "CAMPAIGN_ID",
    "SCHEMA",
    "ACCOUNT_EXPANSION_SCHEMA",
    "LEGACY_SCHEMA",
    "LEGACY_ACCOUNT_EXPANSION_SCHEMA",
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
    "audit_remote_packages",
)


@pytest.fixture()
def configured_engine():
    previous = {name: getattr(engine, name) for name in CONFIGURED_NAMES}
    q23.configure_engine(PACKAGE_SHA, 18_000, 5_233)
    try:
        yield engine
    finally:
        for name, value in previous.items():
            setattr(engine, name, value)


def test_q23_submission_is_clean_four_cpu_same_node_contract(configured_engine):
    args = configured_engine._parser().parse_args([])
    args.eligible_accounts = q23.DEFAULT_ELIGIBLE_ACCOUNTS
    submission = configured_engine.pooled_submission(args)
    environment = submission["submission_env"]

    assert configured_engine.CAMPAIGN_ID == q23.CAMPAIGN_ID
    assert configured_engine.audit_remote_packages is q23._audit_q23_remote_packages
    assert configured_engine.EXPECTED_POOL_SESSIONS == 173
    assert configured_engine.EXPECTED_POOL_CAPACITY == 519
    assert configured_engine.EXPECTED_SESSION_RESERVED_CPUS == 13
    projects_per_account = (
        q23.EXPECTED_POOL_TARGET + len(q23.DEFAULT_ELIGIBLE_ACCOUNTS) - 1
    ) // len(q23.DEFAULT_ELIGIBLE_ACCOUNTS)
    packed_demand = len(q23.DEFAULT_ELIGIBLE_ACCOUNTS) * (
        projects_per_account + q23.EXPECTED_PROJECTS_PER_AEDT - 1
    ) // q23.EXPECTED_PROJECTS_PER_AEDT
    assert configured_engine.EXPECTED_POOL_SESSIONS - packed_demand == 3
    assert submission["cpus"] == 4
    assert submission["account_names"] == q23.DEFAULT_ELIGIBLE_ACCOUNTS
    assert environment["MFT_AEDT_POOL_FILL_TIMEOUT_SECONDS"] == "900"
    assert environment["MFT_CAMPAIGN_ID"] == q23.CAMPAIGN_ID
    assert environment["MFT_CAMPAIGN_SCHEDULER_PACKAGE_REVISION"] == PACKAGE_SHA


def test_q23_profile_and_manifest_identity_pin_four_cpus(configured_engine):
    profile = configured_engine.verify_profile(q23.PROFILE_PATH)
    identity = configured_engine.manifest_identity(
        18_000,
        Path("feeder_state.json"),
        q23.DEFAULT_ELIGIBLE_ACCOUNTS,
    )
    manifest = configured_engine.build_manifest(identity)
    assert profile["cpus"] == 4
    assert profile["timeout_seconds"] == 86_400
    assert identity["campaign_id"] == q23.CAMPAIGN_ID
    assert identity["scheduler_package_revision"] == PACKAGE_SHA
    assert identity["eligible_accounts"] == list(q23.DEFAULT_ELIGIBLE_ACCOUNTS)
    assert identity["pool_topology"]["min_idle_aedt_sessions"] == 3
    assert identity["pool_topology"]["session_base_cpus"] == 1
    assert identity["pool_topology"]["session_reserved_cpus"] == 13
    assert manifest["runtime_control"]["cpu_accounting"] == {
        "thin_client_cpus": 4,
        "host_cpus_per_attached_lease": 4,
        "full_session_host_cpus": 13,
        "owner": "scheduler-live-aedt-project-leases",
    }
    assert identity["timeouts"]["pool_fill"] == 7_200
    assert identity["adoption"]["semantics"].startswith("clean-q23-boundary")


def test_q23_pool_gate_requires_three_idle_sessions(monkeypatch):
    summary = {"config": {"min_idle_aedt_sessions": 3}}
    monkeypatch.setattr(
        q23,
        "_ORIGINAL_VERIFY_POOL_AND_POLICY",
        lambda _url: (summary, 500),
    )
    assert q23._verify_q23_pool_and_policy("http://scheduler") == (summary, 500)

    summary["config"]["min_idle_aedt_sessions"] = 2
    with pytest.raises(engine.GateError, match="must remain 3"):
        q23._verify_q23_pool_and_policy("http://scheduler")


def test_q23_clean_boundary_rejects_any_live_q22_task(tmp_path):
    db_path = tmp_path / "scheduler.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE tasks (id INTEGER, status TEXT, project TEXT, command TEXT)"
        )
        connection.execute(
            "CREATE TABLE aedt_project_leases "
            "(id INTEGER, state TEXT, task_id INTEGER, session_id INTEGER)"
        )
        connection.execute(
            "INSERT INTO tasks VALUES (1, 'cancelled', ?, ?)",
            (engine.PROJECT, q23.PREDECESSOR_CAMPAIGN_ID),
        )
    assert q23.verify_q22_retired(db_path)["live_tasks"] == 0

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "INSERT INTO tasks VALUES (2, 'running', ?, ?)",
            (engine.PROJECT, q23.PREDECESSOR_CAMPAIGN_ID),
        )
    with pytest.raises(engine.GateError, match="live_tasks=1"):
        q23.verify_q22_retired(db_path)

    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE tasks SET status = 'cancelled' WHERE id = 2")
        connection.execute(
            "INSERT INTO aedt_project_leases VALUES (10, 'releasing', 2, 500)"
        )
    with pytest.raises(engine.GateError, match="live_leases=1"):
        q23.verify_q22_retired(db_path)


def test_q23_requires_full_runtime_package_sha():
    with pytest.raises(engine.GateError, match="full lowercase commit SHA"):
        q23.configure_engine("deadbeef", 18_000, 5_233)


def test_q23_first_manifest_requires_exact_current_baselines(
    configured_engine, monkeypatch, tmp_path
):
    state_path = tmp_path / "feeder_state.json"
    state_path.write_text('{"serial": 18000}', encoding="utf-8")
    manifest_path = tmp_path / "q23.manifest.json"
    monkeypatch.setattr(configured_engine.feeder, "dataset_row_count", lambda: 5_233)
    monkeypatch.setattr(
        q23,
        "_ORIGINAL_LOAD_OR_CREATE_MANIFEST",
        lambda *_args, **_kwargs: {"ok": True},
    )

    assert q23._load_or_create_q23_manifest(
        manifest_path,
        state_path,
        q23.DEFAULT_ELIGIBLE_ACCOUNTS,
        execute=False,
        baseline_serial=18_000,
    ) == {"ok": True}
    with pytest.raises(engine.GateError, match="current feeder serial 18000"):
        q23._load_or_create_q23_manifest(
            manifest_path,
            state_path,
            q23.DEFAULT_ELIGIBLE_ACCOUNTS,
            execute=False,
            baseline_serial=17_999,
        )
    monkeypatch.setattr(configured_engine.feeder, "dataset_row_count", lambda: 5_234)
    with pytest.raises(engine.GateError, match="current row count 5234"):
        q23._load_or_create_q23_manifest(
            manifest_path,
            state_path,
            q23.DEFAULT_ELIGIBLE_ACCOUNTS,
            execute=False,
            baseline_serial=18_000,
        )


def test_q23_first_manifest_converts_dataset_lock_timeout(
    configured_engine, monkeypatch, tmp_path
):
    state_path = tmp_path / "feeder_state.json"
    state_path.write_text('{"serial": 18000}', encoding="utf-8")

    def locked_dataset():
        raise engine.FileLockTimeout("train.parquet.lock")

    monkeypatch.setattr(configured_engine.feeder, "dataset_row_count", locked_dataset)
    with pytest.raises(engine.GateError, match="baseline dataset is unreadable"):
        q23._load_or_create_q23_manifest(
            tmp_path / "q23.manifest.json",
            state_path,
            q23.DEFAULT_ELIGIBLE_ACCOUNTS,
            execute=False,
            baseline_serial=18_000,
        )


def test_q23_reaudits_packages_and_boundary_inside_first_manifest_lock(
    configured_engine, monkeypatch, tmp_path
):
    state_path = tmp_path / "feeder_state.json"
    state_path.write_text('{"serial": 18000}', encoding="utf-8")
    events = []
    runtime_args = type("Args", (), {
        "accounts_config": tmp_path / "accounts.yaml",
        "eligible_accounts": q23.DEFAULT_ELIGIBLE_ACCOUNTS,
        "ssh_audit_python": tmp_path / "python.exe",
        "scheduler_db": tmp_path / "scheduler.db",
    })()
    monkeypatch.setattr(q23, "_RUNTIME_PREFLIGHT_ARGS", runtime_args)
    monkeypatch.setattr(configured_engine.feeder, "dataset_row_count", lambda: 5_233)
    monkeypatch.setattr(
        configured_engine,
        "audit_remote_packages",
        lambda *_args: events.append("packages"),
    )
    monkeypatch.setattr(
        q23,
        "verify_q22_retired",
        lambda _path: events.append("boundary"),
    )
    monkeypatch.setattr(
        q23,
        "_ORIGINAL_LOAD_OR_CREATE_MANIFEST",
        lambda *_args, **_kwargs: events.append("manifest") or {"ok": True},
    )

    assert q23._load_or_create_q23_manifest(
        tmp_path / "q23.manifest.json",
        state_path,
        q23.DEFAULT_ELIGIBLE_ACCOUNTS,
        execute=True,
        baseline_serial=18_000,
    ) == {"ok": True}
    assert events == ["packages", "boundary", "manifest"]


def test_q23_execute_runs_full_preflight_before_engine_main(monkeypatch):
    events = []
    monkeypatch.setattr(q23, "configure_engine", lambda *_args: None)
    monkeypatch.setattr(q23.engine, "run_live_gates", lambda _args: events.append("gates"))
    monkeypatch.setattr(q23.engine, "main", lambda _argv: events.append("main") or 0)

    arguments = [
        "--scheduler-package-revision", PACKAGE_SHA,
        "--adopt-baseline-serial", "18000",
        "--adopt-baseline-dataset-rows", "5233",
        "--execute-mft-family-production",
    ]
    for account in q23.DEFAULT_ELIGIBLE_ACCOUNTS:
        arguments.extend(["--eligible-account", account])
    assert q23.main(arguments) == 0
    assert events == ["gates", "main"]


def test_q23_rejects_equals_form_manifest_override():
    with pytest.raises(engine.GateError, match="clean version-1 manifest"):
        q23.main([
            "--scheduler-package-revision", PACKAGE_SHA,
            "--adopt-baseline-serial", "18000",
            "--adopt-baseline-dataset-rows", "5233",
            "--manifest-version=2",
        ])


def test_q23_rejects_partial_account_set(monkeypatch):
    monkeypatch.setattr(q23, "configure_engine", lambda *_args: None)
    with pytest.raises(engine.GateError, match="exact audited five-account"):
        q23.main([
            "--scheduler-package-revision", PACKAGE_SHA,
            "--adopt-baseline-serial", "18000",
            "--adopt-baseline-dataset-rows", "5233",
            "--eligible-account", "dhj02",
        ])


def test_five_account_deployer_is_dry_run_by_default_and_validates_target():
    assert package_deploy.DEFAULT_ACCOUNTS == q23.DEFAULT_ELIGIBLE_ACCOUNTS
    command = package_deploy._deploy_command("1" * 40, "2" * 40, "true")
    assert "worktree add --detach" in command
    assert "MAX_POOL_FILL_TIMEOUT_SECONDS == 7200.0" in command
    assert "checkout --detach 1111111111111111111111111111111111111111" in command
    assert "switched=1" in command
    assert "trap finish EXIT" in command
    exact_audit = package_deploy._exact_audit_command("2" * 40, "true")
    assert "MAX_POOL_FILL_TIMEOUT_SECONDS == 7200.0" in exact_audit


def test_q23_remote_audit_requires_exact_ceiling(
    configured_engine, monkeypatch, tmp_path
):
    payload = [
        {
            "account": name,
            "current": PACKAGE_SHA,
            "target": PACKAGE_SHA,
            "ready": True,
        }
        for name in q23.DEFAULT_ELIGIBLE_ACCOUNTS
    ]

    class Result:
        returncode = 0
        stdout = json.dumps(payload)

    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return Result()

    monkeypatch.setattr(q23.subprocess, "run", fake_run)
    audit_python = tmp_path / "python.exe"
    audit_python.write_bytes(b"")
    evidence = q23._audit_q23_remote_packages(
        tmp_path / "accounts.yaml",
        q23.DEFAULT_ELIGIBLE_ACCOUNTS,
        audit_python,
    )
    assert len(evidence) == 5
    assert all(row["max_pool_fill_timeout_seconds"] == 7_200 for row in evidence)
    assert "--exact-audit" in commands[0]


def test_execute_deployer_rejects_partial_account_set_before_io(tmp_path):
    with pytest.raises(RuntimeError, match="exact ordered five-account"):
        package_deploy.deploy(
            tmp_path / "accounts.yaml",
            ["dhj02"],
            "1" * 40,
            "2" * 40,
            execute=True,
        )


def test_execute_deployer_requires_q22_stop_ack_before_config_io(tmp_path):
    with pytest.raises(RuntimeError, match="q22 stop acknowledgement"):
        package_deploy.deploy(
            tmp_path / "accounts.yaml",
            list(package_deploy.DEFAULT_ACCOUNTS),
            "1" * 40,
            "2" * 40,
            execute=True,
        )


def test_deployer_rolls_back_the_uncertain_current_account(monkeypatch, tmp_path):
    current = "1" * 40
    target = "2" * 40
    rows = {
        name: {
            "name": name,
            "capabilities": ["conda:pyaedt2026v1"],
            "env_profiles": {"pyaedt2026v1": "true"},
        }
        for name in package_deploy.DEFAULT_ACCOUNTS
    }

    class Client:
        def close(self):
            pass

    deploy_calls = 0
    rollback_calls = 0

    def fake_run(_client, command, timeout=120):
        nonlocal deploy_calls, rollback_calls
        if "worktree add --detach" in command:
            deploy_calls += 1
            if deploy_calls == 2:
                raise RuntimeError("uncertain remote result")
            return target
        if "worktree add --detach" not in command and "checkout --detach" in command:
            rollback_calls += 1
            return ""
        return current

    monkeypatch.setattr(package_deploy, "_load_accounts", lambda _path: rows)
    monkeypatch.setattr(package_deploy, "_connect", lambda _row: Client())
    monkeypatch.setattr(package_deploy, "_run", fake_run)
    monkeypatch.setattr(
        package_deploy, "_verify_q22_controller_stopped", lambda *_args: None
    )

    with pytest.raises(RuntimeError, match="uncertain remote result"):
        package_deploy.deploy(
            tmp_path / "accounts.yaml",
            list(package_deploy.DEFAULT_ACCOUNTS),
            current,
            target,
            execute=True,
            q22_controller_stopped=True,
        )
    assert rollback_calls == 2
