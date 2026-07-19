from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import tarfile

import pytest

from tools import tier1_slurm_rolling as rolling
from tools.tier1_slurm_rolling import (
    ATTESTATION_SCHEMA,
    TASK_SCHEMA,
    assert_existing_task_attestation,
    build_task_contract,
    canonical_bytes,
    canonical_sha,
    publish_pointer,
    task_payload,
    temperature_constraint_contract,
    validate_temperature_constraint_contract,
)


def test_copy_tree_and_inventory_support_extended_length_destination(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    source_path = source / "payload.json"
    source_path.write_text('{"ready": true}\n', encoding="utf-8")
    runtime = tmp_path / ("thermal-crossover-runtime-" + "r" * 64)
    bundle = (
        runtime
        / "deployments"
        / ("res15k-model-hard-identity-" + "c" * 48)
        / "bundle"
    )
    destination = (
        bundle
        / "artifacts"
        / "code"
        / "regression_260707"
        / "verify"
        / ("experiment-aedt-isolation-" + "x" * 48)
    )
    target = destination / "payload.json"
    assert len(os.path.abspath(target)) > 260

    try:
        rolling._copy_tree(source, destination)
        inventory = rolling._deployment_files(bundle)

        relative = target.relative_to(bundle).as_posix()
        assert rolling._local_io_path(target).read_text(encoding="utf-8") == (
            '{"ready": true}\n'
        )
        assert inventory[relative] == {
            "sha256": rolling.sha256(source_path),
            "size": source_path.stat().st_size,
        }
        archive = tmp_path / "bundle.tar"
        with tarfile.open(archive, "w") as stream:
            stream.add(
                rolling._local_io_path(bundle / "artifacts"),
                arcname="artifacts",
            )
        with tarfile.open(archive) as stream:
            assert relative in stream.getnames()
    finally:
        if runtime.exists() or os.name == "nt":
            shutil.rmtree(rolling._local_io_path(runtime), ignore_errors=True)


def _pointer() -> dict:
    hard_spec = {"T_limit_C": 110.0}
    temperature_contract = temperature_constraint_contract(hard_spec)
    return {
        "schema_version": rolling.POINTER_SCHEMA,
        "current": {
            "cohort_id": "res15k-model-hard-0123456789",
            "bundle_id": "tier1-res15k-model-hard-0123456789",
            "bundle_manifest_sha256": "a" * 64,
            "constraint_version": (
                "mft-tier1-envelope-1200x1200x750-res15k-"
                "t110all11-core4-5t-lmhalf-v4"
            ),
            "hard_spec": hard_spec,
            "hard_spec_sha256": canonical_sha(hard_spec),
            "attestation_schema_version": ATTESTATION_SCHEMA,
            "task_schema_version": TASK_SCHEMA,
            "controller_source_sha256": "9" * 64,
            "temperature_constraint_contract": temperature_contract,
            "temperature_constraint_contract_sha256": canonical_sha(
                temperature_contract
            ),
            "source_model_manifest_sha256": "c" * 64,
            "deployment_model_manifest_sha256": "d" * 64,
            "nsga_code_revision": "e" * 40,
            "warm_start_sha256": "f" * 64,
            "search_profile": rolling.search_profile(),
            "optimizer_resonance_scale_Hz": (
                rolling.OPTIMIZER_RESONANCE_SCALE_HZ
            ),
            "base_python_site": "/gpfs/base/python-site",
            "remote_bundle": "/gpfs/tier1/bundle",
            "rolling_target": 32,
            "refill_allowed": True,
        }
    }


def test_seed_payload_is_cpu_only_fail_closed_and_exactly_deduped():
    first = task_payload(_pointer(), 101)
    second = task_payload(_pointer(), 102)
    contract = first["payload_json"]

    assert contract["schema_version"] == TASK_SCHEMA
    assert contract["hard_spec"] == {"T_limit_C": 110.0}
    assert contract["temperature_constraint_contract"][
        "robust_upper_bound_C"
    ] == 110.0
    assert contract["seed"] == 101
    assert contract["population"] == 320
    assert contract["max_generations"] == 600
    assert contract["inference_threads"] == 8
    assert contract["optimizer_resonance_scale_Hz"] == 1_000.0
    assert contract["production_eligible"] is False
    assert contract["fea_submission_approved"] is False
    assert contract["fea_submission_performed"] is False
    assert contract["aedt_used"] is False
    assert first["cpus"] == 8
    assert first["memory_mb"] == 32768
    assert first["scheduling_profile"] == "standard"
    assert first["aedt_backend"] == "standalone"
    assert first["priority"] == -8
    assert first["gpus"] == 0
    assert first["dedupe_key"].startswith("mft-tier1-normalized-nsga:")
    assert first["dedupe_key"] != second["dedupe_key"]
    assert first["name"] != second["name"]


def test_constraint_identity_is_present_in_name_and_payload():
    payload = task_payload(_pointer(), 1807180000)
    assert payload["name"].startswith("mft-t1norm-0123456789-s")
    assert payload["payload_json"]["constraint_version"].endswith(
        "t110all11-core4-5t-lmhalf-v4"
    )
    assert payload["payload_json"]["hard_spec_sha256"] == canonical_sha(
        {"T_limit_C": 110.0}
    )


def test_non_110_temperature_limits_fail_closed():
    for hard_spec in (
        {}, {"T_limit_C": 97.0}, {"T_limit_C": 100.0},
        {"T_limit_C": 120.0},
    ):
        with pytest.raises(RuntimeError):
            temperature_constraint_contract(hard_spec)


def test_temperature_target_missing_extra_or_reordered_is_rejected():
    hard_spec = {"T_limit_C": 110.0}
    valid = temperature_constraint_contract(hard_spec)
    mutations = []
    missing = copy.deepcopy(valid)
    missing["targets"].pop()
    mutations.append(missing)
    extra = copy.deepcopy(valid)
    extra["targets"].append("T_unattested")
    mutations.append(extra)
    reordered = copy.deepcopy(valid)
    reordered["targets"][0], reordered["targets"][1] = (
        reordered["targets"][1], reordered["targets"][0]
    )
    mutations.append(reordered)
    for mutation in mutations:
        with pytest.raises(RuntimeError, match="semantic contract"):
            validate_temperature_constraint_contract(
                hard_spec, mutation, canonical_sha(mutation)
            )


def test_initial_and_refill_use_byte_equivalent_contract_except_seed():
    pointer = _pointer()
    initial = build_task_contract(pointer, 101)
    refill = build_task_contract(pointer, 102)
    assert task_payload(pointer, 101)["payload_json"] == initial
    assert task_payload(pointer, 102)["payload_json"] == refill
    initial_without_seed = dict(initial)
    refill_without_seed = dict(refill)
    initial_without_seed.pop("seed")
    refill_without_seed.pop("seed")
    assert canonical_bytes(initial_without_seed) == canonical_bytes(
        refill_without_seed
    )


def test_missing_existing_task_attestation_halts_refill():
    pointer = _pointer()
    submitted = task_payload(pointer, 101)
    task = {
        "id": 7,
        "name": submitted["name"],
        "remote_cwd": submitted["remote_cwd"],
        "dedupe_key": submitted["dedupe_key"],
    }
    assert_existing_task_attestation(pointer, [task])
    task["dedupe_key"] = "mft-tier1-nsga:legacy-or-missing-contract"
    with pytest.raises(RuntimeError, match="missing canonical payload"):
        assert_existing_task_attestation(pointer, [task])


def test_priority_migration_attests_updated_and_claimed_legacy_tasks(
    monkeypatch,
):
    monkeypatch.setattr(
        rolling, "PRIORITY_MIGRATION_SEED_CUTOFF", 1907191032,
    )
    pointer = _pointer()
    legacy = task_payload(pointer, 1907191000)
    task = {
        "id": 7,
        "name": legacy["name"],
        "remote_cwd": legacy["remote_cwd"],
        "dedupe_key": legacy["dedupe_key"],
        "status": "queued",
        "priority": -8,
    }
    assert_existing_task_attestation(pointer, [task])
    task.update({"status": "running", "priority": -9})
    assert_existing_task_attestation(pointer, [task])
    task.update({"status": "queued", "priority": -9})
    with pytest.raises(RuntimeError, match="missing canonical payload"):
        assert_existing_task_attestation(pointer, [task])


def test_priority_dedupe_cutoff_is_deterministic(monkeypatch):
    monkeypatch.setattr(
        rolling, "PRIORITY_MIGRATION_SEED_CUTOFF", 1907191032,
    )
    pointer = _pointer()
    legacy = task_payload(pointer, 1907191031)
    successor = task_payload(pointer, 1907191032)
    assert legacy["priority"] == -9
    assert successor["priority"] == -8
    assert legacy["dedupe_key"] != successor["dedupe_key"]
    post_cutoff = task_payload(pointer, 1907191032, priority=-9)
    task = {
        "id": 8,
        "name": post_cutoff["name"],
        "remote_cwd": post_cutoff["remote_cwd"],
        "dedupe_key": post_cutoff["dedupe_key"],
        "status": "queued",
        "priority": -8,
    }
    with pytest.raises(RuntimeError, match="missing canonical payload"):
        assert_existing_task_attestation(pointer, [task])


def test_priority_migration_updates_only_queued_legacy_tasks(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        rolling, "PRIORITY_MIGRATION_SEED_CUTOFF", 1907191032,
    )
    pointer = _pointer()
    pointer_path = tmp_path / "canonical" / "model_pointer.json"
    rolling.atomic_json(pointer_path, pointer)
    legacy = task_payload(pointer, 1907191000)
    before = [{
        "id": 7,
        "name": legacy["name"],
        "remote_cwd": legacy["remote_cwd"],
        "dedupe_key": legacy["dedupe_key"],
        "status": "queued",
        "priority": -9,
    }]
    after = [{**before[0], "priority": -8}]
    inventories = iter((before, after))
    monkeypatch.setattr(
        rolling, "list_cohort_tasks", lambda *_args, **_kwargs: next(inventories)
    )
    calls = []

    def api_json(url, method="GET", payload=None, timeout=30):
        calls.append((url, method, payload, timeout))
        return {"id": 7, "priority": -8}

    monkeypatch.setattr(rolling, "api_json", api_json)
    result = rolling.migrate_priority(runtime=tmp_path, apply=True)
    assert result["applied"] is True
    assert result["queued_update_ids"] == [7]
    assert calls[0][1:3] == ("POST", {"priority": -8})
    assert (
        tmp_path / "controller" / "priority_migration.json"
    ).is_file()


def test_pointer_cutover_retains_old_cohort_without_refill(tmp_path):
    runtime = tmp_path / "runtime"
    canonical = runtime / "canonical"
    canonical.mkdir(parents=True)
    old = _pointer()["current"]
    old.update({
        "refill_allowed": True,
        "rolling_target": 256,
        "production_eligible": False,
        "fea_submission_approved": False,
    })
    (canonical / "model_pointer.json").write_text(json.dumps({
        "schema_version": "mft-tier1-slurm-model-pointer-v1",
        "current": old,
        "legacy_cohorts": [{"cohort_id": "older-cohort"}],
        "production_eligible": False,
        "fea_submission_approved": False,
        "automatic_promotion_allowed": False,
    }), encoding="utf-8")

    successor = copy.deepcopy(old)
    successor.update({
        "cohort_id": "res15k-model-hard-successor",
        "bundle_id": "tier1-res15k-model-hard-successor",
        "bundle_manifest_sha256": "1" * 64,
        "nsga_code_revision": "2" * 40,
        "warm_start_sha256": "3" * 64,
        "remote_bundle": "/gpfs/tier1/successor",
    })
    plan = tmp_path / "deployment_plan.json"
    plan.write_text(json.dumps(successor), encoding="utf-8")

    published = publish_pointer(plan, runtime)

    assert published["current"]["cohort_id"] == successor["cohort_id"]
    assert published["current"]["refill_allowed"] is True
    assert published["current"]["rolling_target"] == 32
    assert published["current"]["production_eligible"] is False
    assert published["current"]["fea_submission_approved"] is False
    assert len(published["legacy_cohorts"]) == 2
    retained = published["legacy_cohorts"][-1]
    assert retained["cohort_id"] == old["cohort_id"]
    assert retained["refill_allowed"] is False
    assert retained["legacy_reason"] == (
        "atomic_constraint_or_model_pointer_switch"
    )


def test_comparison_ui_gate_requires_completed_authenticated_deep_candidate(
    monkeypatch, tmp_path
):
    identity = {
        "constraint_version": "constraint-v1",
        "hard_spec_sha256": "1" * 64,
        "source_model_manifest_sha256": "2" * 64,
        "deployment_model_manifest_sha256": "3" * 64,
        "temperature_constraint_contract_sha256": "4" * 64,
    }
    live = {
        **identity,
        "updated_at": "live",
        "terminal_results": [],
        "aggregate_pareto": {
            "authenticated_terminal_count": 1,
            "candidate_preview_count": 0,
            "candidates": [],
        },
    }
    deep = {
        **identity,
        "updated_at": "deep",
        "terminal_results": [{
            "authenticated": True,
            "terminal_state": "failed",
        }],
        "aggregate_pareto": {
            "authenticated_terminal_count": 1,
            "candidate_preview_count": 0,
            "candidates": [],
        },
    }

    def snapshot(path):
        if Path(path) == rolling.LIVE_RUNTIME:
            return {"active_cohort_id": "live"}, live, "a" * 64
        return {"active_cohort_id": "deep"}, deep, "b" * 64

    monkeypatch.setattr(rolling, "_sealed_status_snapshot", snapshot)
    rolling._publish_read_only_comparison(tmp_path)
    output = rolling.read_json(
        tmp_path / "comparison" / "live_plus_deep.json"
    )
    assert output["deep_terminal_ready"] is False
    assert output["ui_promotion_eligible"] is False

    deep["terminal_results"][0]["terminal_state"] = "completed"
    deep["aggregate_pareto"].update({
        "candidate_preview_count": 1,
        "candidates": [{
            "total_loss_W": 5500.0,
            "volume_L": 900.0,
            "decoded_params": {"n_core_group": 4},
        }],
    })
    rolling._publish_read_only_comparison(tmp_path)
    output = rolling.read_json(
        tmp_path / "comparison" / "live_plus_deep.json"
    )
    assert output["deep_terminal_ready"] is True
    assert output["ui_promotion_eligible"] is True


def _assert_canonical_identity(runtime, status):
    index = rolling.read_json(runtime / "canonical" / "index.json")
    status_path = Path(index["status"]["path"])
    sealed_status = rolling.read_json(status_path)
    assert rolling.sha256(status_path) == index["status"]["sha256"]
    assert sealed_status == status
    assert index["active_cohort_id"] == status["cohort_id"]
    assert index["constraint_version"] == status["constraint_version"]
    assert index["hard_spec_sha256"] == status["hard_spec_sha256"]
    assert index["source_model_manifest_sha256"] == status[
        "source_model_manifest_sha256"
    ]
    assert index["deployment_model_manifest_sha256"] == status[
        "deployment_model_manifest_sha256"
    ]
    assert index["updated_at"] == status["updated_at"]


def test_32_deficit_refill_keeps_identity_healthy_and_at_target_truthful(
    monkeypatch, tmp_path
):
    pointer = _pointer()
    pointer["current"]["hard_spec_sha256"] = rolling.canonical_sha(
        pointer["current"]["hard_spec"]
    )
    pointer_path = tmp_path / "canonical" / "model_pointer.json"
    rolling.atomic_json(pointer_path, pointer)
    current = pointer["current"]
    monkeypatch.setattr(
        rolling,
        "_feedback_contract",
        lambda _: (
            current["constraint_version"],
            current["hard_spec"],
            current["nsga_code_revision"],
        ),
    )
    monkeypatch.setattr(rolling, "list_cohort_tasks", lambda *_: [])
    monkeypatch.setattr(rolling, "harvest_terminal_results", lambda *_, **__: [])
    next_task_id = iter(range(1, 33))
    monkeypatch.setattr(
        rolling,
        "api_json",
        lambda *_, **__: {"task_id": next(next_task_id), "deduped": False},
    )

    snapshots = []
    original_publish = rolling._publish_status

    def publish_and_read(*args, **kwargs):
        status = original_publish(*args, **kwargs)
        _assert_canonical_identity(tmp_path, status)
        snapshots.append(
            {
                "healthy": status["healthy"],
                "at_target": status["rolling"]["at_target"],
                "deficit": status["rolling"]["deficit"],
                "freshness": status["freshness_deadline_seconds"],
                "priorities": {
                    task["priority"]
                    for task in status["latest_tasks"]
                    if "priority" in task
                },
            }
        )
        return status

    monkeypatch.setattr(rolling, "_publish_status", publish_and_read)
    final = rolling.reconcile_once("http://scheduler", tmp_path, apply=True)

    assert [item["deficit"] for item in snapshots] == [
        32, 28, 24, 20, 16, 12, 8, 4, 0
    ]
    assert all(item["healthy"] is True for item in snapshots)
    assert all(item["at_target"] is False for item in snapshots[:-1])
    assert snapshots[-1]["at_target"] is True
    assert all(
        not item["priorities"]
        or item["priorities"] == {rolling.TASK_PRIORITY}
        for item in snapshots
    )
    assert snapshots[-1]["priorities"] == {rolling.TASK_PRIORITY}
    assert all(
        item["freshness"] == rolling.FRESHNESS_DEADLINE_SECONDS == 60
        for item in snapshots
    )
    assert final["rolling"]["refill_in_progress"] is False
    assert final["rolling"]["over_target"] is False
    assert final["health_contract_version"] == rolling.HEALTH_CONTRACT_VERSION


def test_over_target_or_explicit_error_remains_unhealthy(tmp_path):
    pointer = _pointer()
    rolling.atomic_json(tmp_path / "canonical" / "model_pointer.json", pointer)
    prefix = rolling._task_prefix(pointer["current"]["cohort_id"])
    tasks = [
        {"id": index, "name": f"{prefix}{index}", "status": "queued"}
        for index in range(1, 34)
    ]
    status = rolling._publish_status(
        pointer,
        tasks,
        runtime=tmp_path,
        controller={"pid": 1},
    )
    assert status["healthy"] is False
    assert status["rolling"]["over_target"] is True
    assert status["rolling"]["at_target"] is False
    assert "exceeds rolling target" in status["error"]

    status = rolling._publish_status(
        pointer,
        tasks[:32],
        runtime=tmp_path,
        controller={"pid": 1},
        error="scheduler inventory failed authentication",
    )
    assert status["healthy"] is False
    assert status["rolling"]["at_target"] is True
    assert status["error"] == "scheduler inventory failed authentication"


def test_controller_error_never_refreshes_canonical_index(monkeypatch, tmp_path):
    index_path = tmp_path / "canonical" / "index.json"
    original_index = {
        "schema_version": rolling.INDEX_SCHEMA,
        "updated_at": "2026-07-18T00:00:00+00:00",
        "freshness_deadline_seconds": rolling.FRESHNESS_DEADLINE_SECONDS,
    }
    rolling.atomic_json(index_path, original_index)

    def fail_reconcile(*_, **__):
        raise RuntimeError("scheduler inventory unavailable")

    monkeypatch.setattr(rolling, "reconcile_once", fail_reconcile)
    monkeypatch.setattr(
        rolling.time,
        "sleep",
        lambda _: (_ for _ in ()).throw(SystemExit("one test iteration")),
    )
    with pytest.raises(SystemExit, match="one test iteration"):
        rolling.controller_worker("http://scheduler", tmp_path, 10.0)

    assert rolling.read_json(index_path) == original_index
    heartbeat = rolling.read_json(
        tmp_path / "canonical" / "controller_status.json"
    )
    assert heartbeat["error"] == "RuntimeError:scheduler inventory unavailable"
    assert heartbeat["updated_at"] != original_index["updated_at"]


def test_1001_task_inventory_is_not_truncated_or_identity_inconsistent(tmp_path):
    pointer = _pointer()
    rolling.atomic_json(tmp_path / "canonical" / "model_pointer.json", pointer)
    prefix = rolling._task_prefix(pointer["current"]["cohort_id"])
    tasks = [
        {
            "id": index,
            "name": f"{prefix}{index}",
            "status": "queued" if index <= 32 else "completed",
        }
        for index in range(1, 1002)
    ]

    status = rolling._publish_status(
        pointer,
        tasks,
        runtime=tmp_path,
        controller={"pid": 1},
    )
    _assert_canonical_identity(tmp_path, status)

    assert rolling.CANONICAL_TASK_WINDOW_LIMIT == 10_000
    assert status["scheduler_task_count"] == 1001
    assert len(status["latest_tasks"]) == status["scheduler_task_count"]
    assert status["state_counts"] == {"completed": 969, "queued": 32}
    assert status["rolling"]["seed_identity_count"] == 1001
    assert status["rolling"]["duplicate_seed_count"] == 0
    assert status["rolling"]["active_plus_queued"] == 32
    assert status["rolling"]["at_target"] is True
    assert status["healthy"] is True
    assert {item["seed"] for item in status["latest_tasks"]} == set(
        range(1, 1002)
    )


def test_fresh_terminal_bypasses_failed_low_id_retry(monkeypatch, tmp_path):
    pointer = _pointer()
    current = pointer["current"]
    prefix = rolling._task_prefix(current["cohort_id"])
    failed_record = rolling._terminal_record_dir(
        tmp_path, current["cohort_id"], 1
    ) / "record.json"
    rolling.atomic_json(failed_record, {
        "schema_version": "mft-tier1-slurm-terminal-result-v1",
        "task_id": 1,
        "authenticated": False,
        "harvest_error": "TimeoutError:old remote timeout",
    })
    status = {
        "schema_version": "mft-tier1-slurm-seed-status-v1",
        "task_id": "2",
        "seed": 2,
        "cohort_id": current["cohort_id"],
        "state": "failed",
        "failure": "terminal test failure",
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "attestation_schema_version": rolling.ATTESTATION_SCHEMA,
        "task_schema_version": rolling.TASK_SCHEMA,
        "constraint_version": current["constraint_version"],
        "hard_spec_sha256": current["hard_spec_sha256"],
        "temperature_constraint_contract_sha256": current[
            "temperature_constraint_contract_sha256"
        ],
        "search_profile": current["search_profile"],
        "optimizer_resonance_scale_Hz": current[
            "optimizer_resonance_scale_Hz"
        ],
        "payload_sha256": rolling.canonical_sha(
            rolling.build_task_contract(pointer, 2)
        ),
    }

    def remote_bytes(url, timeout=30):
        assert "/api/tasks/2/" in url
        return json.dumps(status).encode("utf-8")

    monkeypatch.setattr(rolling, "api_bytes", remote_bytes)
    records = rolling.harvest_terminal_results(
        "http://scheduler",
        pointer,
        [
            {"id": 1, "name": f"{prefix}1", "status": "failed"},
            {"id": 2, "name": f"{prefix}2", "status": "failed"},
        ],
        tmp_path,
        max_new=1,
    )

    by_id = {record["task_id"]: record for record in records}
    assert by_id[1]["authenticated"] is False
    assert by_id[2]["authenticated"] is True
    assert by_id[2]["terminal_state"] == "failed"
    assert rolling.HARVEST_MAX_NEW_PER_TICK == 16


def test_publish_status_excludes_unauthenticated_harvest_retry(tmp_path):
    pointer = _pointer()
    current = pointer["current"]
    rolling.atomic_json(
        tmp_path / "canonical" / "model_pointer.json", pointer
    )
    pending_path = rolling._terminal_record_dir(
        tmp_path, current["cohort_id"], 1
    ) / "record.json"
    pending = {
        "schema_version": "mft-tier1-slurm-terminal-result-v1",
        "task_id": 1,
        "seed": 1,
        "authenticated": False,
        "terminal_state": "completed",
        "harvest_error": "TimeoutError:remote result is not visible yet",
    }
    authenticated = {
        "schema_version": "mft-tier1-slurm-terminal-result-v1",
        "task_id": 2,
        "seed": 2,
        "authenticated": True,
        "terminal_state": "failed",
    }
    rolling.atomic_json(pending_path, pending)
    rolling.atomic_json(
        rolling._terminal_record_dir(
            tmp_path, current["cohort_id"], 2
        ) / "record.json",
        authenticated,
    )

    status = rolling._publish_status(
        pointer,
        [
            {"id": 1, "name": "pending", "status": "completed"},
            {"id": 2, "name": "authenticated", "status": "failed"},
        ],
        runtime=tmp_path,
        controller={"pid": 1},
    )

    assert [row["task_id"] for row in status["terminal_results"]] == [2]
    assert status["aggregate_pareto"]["authenticated_terminal_count"] == 1
    assert pending_path.is_file()
    assert rolling.read_json(pending_path) == pending


def test_bounded_terminal_snapshot_pins_global_preview_sources(monkeypatch, tmp_path):
    records = [
        {"task_id": task_id, "seed": task_id, "authenticated": True}
        for task_id in range(1100, 0, -1)
    ]
    aggregate_calls = []

    def fake_aggregate(_runtime, selected):
        aggregate_calls.append([item["task_id"] for item in selected])
        if len(selected) == 1100:
            return {
                "candidates": [{"source_task_id": 5}],
                "near_feasible": [
                    {"source_task_id": 4},
                    {"source_task_id": 3},
                ],
                "simultaneous_best": {"task_id": 2},
            }
        return {
            "authenticated_terminal_count": len(selected),
            "source_task_ids": [item["task_id"] for item in selected],
        }

    monkeypatch.setattr(rolling, "_aggregate_pareto", fake_aggregate)
    bounded, aggregate = rolling._bounded_terminal_snapshot(tmp_path, records)

    assert rolling.TERMINAL_RESULT_WINDOW_LIMIT == 1_000
    assert len(bounded) == 1_000
    assert [item["task_id"] for item in bounded] == sorted(
        (item["task_id"] for item in bounded), reverse=True
    )
    assert {2, 3, 4, 5}.issubset({item["task_id"] for item in bounded})
    assert 101 not in {item["task_id"] for item in bounded}
    assert aggregate["authenticated_terminal_count"] == 1_000
    assert len(aggregate_calls) == 2
    assert len(aggregate_calls[0]) == 1100
    assert len(aggregate_calls[1]) == 1_000


def test_aggregate_pareto_declares_bounded_candidate_preview(tmp_path):
    result_path = tmp_path / "results" / "task-7" / "result.json"
    result_path.parent.mkdir(parents=True)
    candidates = [
        {
            "decoded_params": {"design": index},
            "volume_L": float(index + 1),
            "total_loss_W": float(130 - index),
        }
        for index in range(129)
    ]
    result_path.write_text(json.dumps({
        "candidates": candidates,
        "next_target_fea_batch_plan": {"candidates": []},
    }), encoding="utf-8")
    records = [{
        "authenticated": True,
        "terminal_state": "completed",
        "task_id": 7,
        "seed": 700,
        "result": {
            "local_path": str(result_path),
            "sha256": rolling.sha256(result_path),
        },
        "constraint_minimum_G": {},
        "terminal_minimum_total_positive_violation": 0.0,
    }]

    aggregate = rolling._aggregate_pareto(tmp_path, records)

    assert aggregate["pareto_count"] == 129
    assert aggregate["candidate_preview_count"] == 128
    assert aggregate["candidate_preview_limit"] == 128
    assert aggregate["candidate_preview_truncated"] is True
    assert len(aggregate["candidates"]) == 128
    assert aggregate["candidates"][0]["decoded_params"]["design"] == 0
    assert aggregate["candidates"][-1]["decoded_params"]["design"] == 128
    selection = aggregate["candidate_preview_selection"]
    assert selection["method"] == (
        "evenly_spaced_volume_order_including_objective_extremes"
    )
    assert selection["selected_front_indexes"][0] == 0
    assert selection["selected_front_indexes"][-1] == 128
    assert len(set(selection["selected_front_indexes"])) == 128


def test_aggregate_pareto_small_preview_keeps_all_candidates(tmp_path):
    result_path = tmp_path / "results" / "task-8" / "result.json"
    result_path.parent.mkdir(parents=True)
    candidates = [
        {
            "decoded_params": {"design": index},
            "volume_L": float(index + 1),
            "total_loss_W": float(4 - index),
        }
        for index in range(3)
    ]
    result_path.write_text(json.dumps({
        "candidates": candidates,
        "next_target_fea_batch_plan": {"candidates": []},
    }), encoding="utf-8")
    records = [{
        "authenticated": True,
        "terminal_state": "completed",
        "task_id": 8,
        "seed": 800,
        "result": {
            "local_path": str(result_path),
            "sha256": rolling.sha256(result_path),
        },
        "constraint_minimum_G": {},
        "terminal_minimum_total_positive_violation": 0.0,
    }]

    aggregate = rolling._aggregate_pareto(tmp_path, records)

    assert [
        item["decoded_params"]["design"]
        for item in aggregate["candidates"]
    ] == [0, 1, 2]
    assert aggregate["candidate_preview_selection"] == {
        "method": "all",
        "ordered_by": ["volume_L", "total_loss_W"],
        "includes_minimum_volume": True,
        "includes_minimum_loss": True,
        "selected_front_indexes": [0, 1, 2],
    }


def test_aggregate_pareto_near_candidates_keep_zero_score_first(tmp_path):
    result_path = tmp_path / "results" / "task-9" / "result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps({
        "candidates": [],
        "next_target_fea_batch_plan": {
            "candidates": [
                {"design": "positive", "target_acquisition_score": 0.25},
                {"design": "zero", "target_acquisition_score": 0.0},
                {"design": "missing"},
            ],
        },
    }), encoding="utf-8")
    records = [{
        "authenticated": True,
        "terminal_state": "completed",
        "task_id": 9,
        "seed": 900,
        "result": {
            "local_path": str(result_path),
            "sha256": rolling.sha256(result_path),
        },
        "constraint_minimum_G": {},
        "terminal_minimum_total_positive_violation": 0.0,
    }]

    aggregate = rolling._aggregate_pareto(tmp_path, records)

    assert [item["design"] for item in aggregate["near_feasible"]] == [
        "zero",
        "positive",
        "missing",
    ]


def test_fast_two_objective_front_matches_reference_with_equal_objectives():
    values = [
        {"design": "a", "volume_L": 5.0, "total_loss_W": 9.0},
        {"design": "b", "volume_L": 5.0, "total_loss_W": 8.0},
        {"design": "c", "volume_L": 5.0, "total_loss_W": 8.0},
        {"design": "d", "volume_L": 6.0, "total_loss_W": 7.0},
        {"design": "e", "volume_L": 7.0, "total_loss_W": 7.0},
        {"design": "f", "volume_L": 8.0, "total_loss_W": 6.0},
        {"design": "g", "volume_L": 9.0, "total_loss_W": 10.0},
    ]

    actual = rolling._two_objective_nondominated(values)

    assert [item["design"] for item in actual] == ["b", "c", "d", "f"]


def test_fast_two_objective_front_matches_quadratic_reference():
    values = [
        {
            "design": index,
            "volume_L": float((index * 37) % 113),
            "total_loss_W": float((index * 53) % 127),
        }
        for index in range(1_000)
    ]

    expected = [
        item
        for index, item in enumerate(values)
        if not any(
            other_index != index
            and other["volume_L"] <= item["volume_L"]
            and other["total_loss_W"] <= item["total_loss_W"]
            and (
                other["volume_L"] < item["volume_L"]
                or other["total_loss_W"] < item["total_loss_W"]
            )
            for other_index, other in enumerate(values)
        )
    ]

    actual = rolling._two_objective_nondominated(values)

    assert {item["design"] for item in actual} == {
        item["design"] for item in expected
    }


def test_publish_over_1000_terminals_keeps_index_and_sources_coherent(
    monkeypatch, tmp_path
):
    pointer = _pointer()
    rolling.atomic_json(tmp_path / "canonical" / "model_pointer.json", pointer)
    prefix = rolling._task_prefix(pointer["current"]["cohort_id"])
    tasks = [
        {
            "id": task_id,
            "name": f"{prefix}{task_id}",
            "status": "queued" if task_id <= 32 else "completed",
        }
        for task_id in range(1, 1133)
    ]
    records = [
        {
            "task_id": task_id,
            "seed": task_id,
            "authenticated": True,
            "terminal_state": "completed",
        }
        for task_id in range(1132, 32, -1)
    ]

    def aggregate(_runtime, selected):
        selected_ids = {int(item["task_id"]) for item in selected}
        return {
            "authenticated_terminal_count": len(selected),
            "candidates": (
                [{"source_task_id": 33}] if 33 in selected_ids else []
            ),
            "near_feasible": [
                {"source_task_id": task_id}
                for task_id in (34, 35)
                if task_id in selected_ids
            ],
            "simultaneous_best": (
                {"task_id": 36} if 36 in selected_ids else None
            ),
        }

    monkeypatch.setattr(
        rolling, "_load_terminal_records", lambda *_: records
    )
    monkeypatch.setattr(rolling, "_aggregate_pareto", aggregate)

    status = rolling._publish_status(
        pointer,
        tasks,
        runtime=tmp_path,
        controller={"pid": 1},
    )
    _assert_canonical_identity(tmp_path, status)

    terminal_ids = {
        int(item["task_id"]) for item in status["terminal_results"]
    }
    aggregate_sources = {
        int(item["source_task_id"])
        for key in ("candidates", "near_feasible")
        for item in status["aggregate_pareto"][key]
    }
    aggregate_sources.add(
        int(status["aggregate_pareto"]["simultaneous_best"]["task_id"])
    )
    assert status["scheduler_task_count"] == 1_132
    assert len(status["latest_tasks"]) == 1_132
    assert len(status["terminal_results"]) == 1_000
    assert status["aggregate_pareto"]["authenticated_terminal_count"] == 1_000
    assert aggregate_sources == {33, 34, 35, 36}
    assert aggregate_sources <= terminal_ids
    assert status["rolling"]["at_target"] is True
    assert status["healthy"] is True
