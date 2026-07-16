from contextlib import ExitStack
import json
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

import pytest


CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
if str(CAMPAIGN_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_DIR))

import feeder
import q22_bounded_soak as q22


def test_profile_is_explicit_full_extraction_with_required_timeout():
    profile = q22.verify_profile()
    overrides = profile["param_overrides"]
    assert {
        key: overrides[key]
        for key in ("full_model", "matrix_on", "cap_on", "loss_on", "thermal_on")
    } == {
        "full_model": 0,
        "matrix_on": 1,
        "cap_on": 1,
        "loss_on": 1,
        "thermal_on": 1,
    }
    assert profile["timeout_seconds"] == 86400


def test_mutable_demand_progress_decrease_never_requests_cancellation():
    manifest = {"baseline_serial": 100}
    extended = q22.campaign_progress(manifest, 107, 12)
    reduced = q22.campaign_progress(manifest, 107, 3)
    assert extended == {
        "baseline_serial": 100,
        "current_serial": 107,
        "accepted_simulations": 7,
        "total_simulations": 12,
        "remaining_simulations": 5,
        "oversupplied_by": 0,
    }
    assert reduced["remaining_simulations"] == 0
    assert reduced["oversupplied_by"] == 4


def test_manifest_dry_run_is_write_free_and_identity_is_fail_closed(tmp_path):
    state = tmp_path / "feeder_state.json"
    state.write_text(json.dumps({"serial": 123, "submitted_samples": 10}), encoding="utf-8")
    target = tmp_path / "manifest.json"
    accounts = ("account-a", "account-b")

    manifest = q22.load_or_create_manifest(
        target, state, accounts, execute=False
    )
    assert manifest["baseline_serial"] == 123
    assert manifest["demand_control"]["default_total_simulations"] == 500
    assert manifest["demand_control"]["mutable_with_cas"] is True
    assert not target.exists()

    persisted = q22.load_or_create_manifest(
        target, state, accounts, execute=True
    )
    assert target.exists()
    assert persisted["identity_sha256"] == manifest["identity_sha256"]
    tampered = json.loads(target.read_text(encoding="utf-8"))
    tampered["solver_revision"] = "f" * 40
    target.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(q22.GateError, match="immutable identity drifted"):
        q22.load_or_create_manifest(target, state, accounts, execute=False)


def test_v2_manifest_is_atomic_append_only_account_superset(tmp_path):
    state = tmp_path / "feeder_state.json"
    state.write_text(json.dumps({"serial": 123}), encoding="utf-8")
    predecessor = tmp_path / "manifest.json"
    expanded_path = tmp_path / "manifest.v2.json"
    original = ("account-a", "account-b")
    expanded = (*original, "account-c", "account-d")
    v1 = q22.load_or_create_manifest(
        predecessor,
        state,
        original,
        execute=True,
        baseline_serial=120,
    )
    state.write_text(json.dumps({"serial": 127}), encoding="utf-8")

    v2 = q22.load_or_create_account_expansion_manifest(
        expanded_path,
        predecessor,
        state,
        expanded,
        execute=True,
        baseline_serial=120,
    )

    assert v2["schema"] == q22.ACCOUNT_EXPANSION_SCHEMA
    assert v2["baseline_serial"] == v1["baseline_serial"] == 120
    assert v2["transition"] == {
        "kind": "append-only-account-superset",
        "predecessor_schema": q22.SCHEMA,
        "predecessor_identity_sha256": v1["identity_sha256"],
        "predecessor_eligible_accounts": list(original),
        "transition_serial": 127,
        "baseline_and_demand_semantics": (
            "same-baseline-and-campaign-demand-no-resubmission"
        ),
    }
    assert v2["eligible_accounts"] == list(expanded)
    assert expanded_path.is_file()

    with pytest.raises(q22.GateError, match="append a strict superset"):
        q22.load_or_create_account_expansion_manifest(
            tmp_path / "bad.v2.json",
            predecessor,
            state,
            ("account-a", "account-c", "account-b"),
            execute=False,
            baseline_serial=120,
        )


def test_submission_contract_pins_accounts_resources_and_all_timeouts():
    args = q22._parser().parse_args([])
    args.eligible_accounts = q22.DEFAULT_ELIGIBLE_ACCOUNTS
    submission = q22.pooled_submission(args)
    environment = submission["submission_env"]
    assert submission["account_names"] == q22.DEFAULT_ELIGIBLE_ACCOUNTS
    assert submission["cpus"] == 1
    assert submission["memory_mb"] == 6144
    assert submission["timeout_seconds"] == 86400
    assert environment["MFT_AEDT_RELEASE_WAIT_SECONDS"] == "7200"
    assert environment["AEDT_POOL_AUTOMATION_LOCK_TIMEOUT_SECONDS"] == "7200"
    assert environment[
        "AEDT_POOL_NATIVE_PIPELINE_BARRIER_TIMEOUT_SECONDS"
    ] == "7200"
    assert environment["MFT_AEDT_ISOLATION_POLICY"] == "family"
    assert environment["MFT_CAMPAIGN_SCHEDULER_PACKAGE_REVISION"] == (
        q22.SCHEDULER_PACKAGE_REVISION
    )


def test_pool_gate_keeps_total_demand_separate_from_active_30():
    pool = {
        "config": {
            "max_aedt_sessions": 10,
            "projects_per_aedt": 3,
            "target_project_concurrency": 30,
            "enabled": True,
            "adapter_ready": True,
            "validation_passed": True,
            "operational": True,
        },
        "latest_validation": {
            "id": 9,
            "status": "passed",
            "mixed_mft_ipmsm_isolation_passed": True,
        },
    }
    project = {
        "name": q22.PROJECT,
        "max_active_tasks": 500,
        "simulation_policy": {
            "desired_simulations": 30,
            "effective_simulations": 27,
            "validated_concurrency_limit": 30,
            "scale_down_mode": "drain",
        },
    }

    def response(_base, path):
        if path == "/api/aedt-pool":
            return pool
        if path == f"/api/projects/{q22.PROJECT}":
            return project
        raise AssertionError(path)

    with patch.object(q22, "_http_json", side_effect=response):
        _, target = q22.verify_pool_and_policy("http://scheduler")
    assert target == 27

    pool["config"]["target_project_concurrency"] = 500
    with patch.object(q22, "_http_json", side_effect=response):
        with pytest.raises(q22.GateError, match="10x3/30"):
            q22.verify_pool_and_policy("http://scheduler")


def test_owned_serials_reject_unrelated_feeder_mutation(tmp_path):
    db = tmp_path / "scheduler.db"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE tasks (name TEXT, dedupe_key TEXT)")
        prefix = (
            f"mft-camp-s{q22.CAMPAIGN_SOLVER[:7]}-"
            f"l{q22.LIBRARY_REVISION[:7]}-"
        )
        connection.execute(
            "INSERT INTO tasks VALUES (?, ?)",
            (f"{prefix}00101", f"mft-al:x:{q22.CAMPAIGN_SOLVER}:{q22.LIBRARY_REVISION}:d"),
        )
    manifest = {"baseline_serial": 100}
    q22.verify_owned_serials(db, manifest, 101)
    with pytest.raises(q22.GateError, match="serial 102"):
        q22.verify_owned_serials(db, manifest, 102)


def test_feeder_exact_remaining_cap_and_deterministic_account_pins():
    initial_state = {"serial": 100, "submitted_samples": 0}
    submitted = []

    def next_candidate(cursor, seed):
        return cursor + 1, cursor, {"candidate": cursor}

    def submit(name, workdir, params, solver, library, **kwargs):
        submitted.append((name, kwargs["account_name"]))
        return 9000 + len(submitted)

    campaign_counts = {"queued": 0, "attaching": 0, "running": 0}
    capacity = {
        "ready_fit_slots": 30,
        "project_submission_slots": 500,
        "submission_allowed": True,
        "queue_state": "ready",
        "queue_reason": "",
        "project_active": 0,
    }
    with ExitStack() as stack:
        stack.enter_context(patch.object(
            feeder.scheduler_client, "campaign_mutation_lock_is_held", return_value=True
        ))
        stack.enter_context(patch.object(feeder, "load_state", return_value=initial_state))
        stack.enter_context(patch.object(
            feeder, "scheduler_snapshot",
            return_value=(campaign_counts, {}, [], capacity),
        ))
        stack.enter_context(patch.object(
            feeder, "cpu_submission_headroom", return_value=(30, 100, 100, 0)
        ))
        stack.enter_context(patch.object(
            feeder, "dataset_collection_snapshot", return_value=(0, set())
        ))
        stack.enter_context(patch.object(feeder, "campaign_inventory", return_value=[]))
        stack.enter_context(patch.object(feeder, "reserved_unjudged_rows", return_value=0))
        stack.enter_context(patch.object(
            feeder, "cursor_after_valid_candidates", return_value=0
        ))
        stack.enter_context(patch.object(
            feeder, "next_valid_candidate", side_effect=next_candidate
        ))
        stack.enter_context(patch.object(feeder, "submit", side_effect=submit))
        stack.enter_context(patch.object(feeder, "save_state"))
        stack.enter_context(patch.object(feeder.time, "sleep"))
        feeder._step_locked(
            10_000,
            target=30,
            solver_revision="a" * 40,
            library_revision="b" * 40,
            _pooled_submission={
                "aedt_backend": "pooled",
                "account_names": ("account-a", "account-b", "account-c"),
            },
            _max_new_tasks=2,
        )

    assert submitted == [
        ("mft-camp-saaaaaaa-lbbbbbbb-00101", "account-b"),
        ("mft-camp-saaaaaaa-lbbbbbbb-00102", "account-c"),
    ]
    assert initial_state["serial"] == 102


@pytest.mark.parametrize("value", [-1, True, 1.5, "1"])
def test_feeder_rejects_invalid_exact_remaining_cap(value):
    with pytest.raises(feeder.SchedulerError, match="max_new_tasks"):
        feeder.step(100, target=0, max_new_tasks=value)
