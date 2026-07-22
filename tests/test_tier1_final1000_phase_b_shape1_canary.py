from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from tests.test_tier1_final1000_multiseed_phase_a import _base_task
from tools import tier1_final1000_multiseed_phase_b_contract as phase_b
from tools import tier1_final1000_phase_b_shape1_canary as canary
from tools import tier1_final1000_slurm_launch as launch
from tools import tier1_final1000_stage_profiles as profiles


def _config(
    parent: dict[str, Any],
    *,
    seed: int | None = None,
    seed_selection_policy: str | None = None,
) -> dict[str, Any]:
    payload = parent["payload_json"]
    stage = profiles.BY_ID[canary.ENTRY_STAGE_ID]
    selected_seed = (
        int(payload["children"][0]["seed"]) if seed is None else int(seed)
    )
    selected_policy = seed_selection_policy or (
        canary.LEGACY_SEED_SELECTION_POLICY
        if selected_seed == stage.seed_window_end_exclusive - 1
        else canary.EXPLICIT_SEED_SELECTION_POLICY
    )
    unsigned = {
        "schema_version": canary.CONFIG_SCHEMA,
        "diagnostic_id": "test-entry-delta-shape1-v1",
        "stage_id": canary.ENTRY_STAGE_ID,
        "seed": selected_seed,
        "seed_selection_policy": selected_policy,
        "expected_offload_plan_file_sha256": "1" * 64,
        "expected_bundle_id": payload["bundle_id"],
        "expected_bundle_manifest_sha256": payload["bundle_manifest_sha256"],
        "expected_bundle_contract_sha256": "2" * 64,
        "lane_shape": 1,
        "logical_seed_count": 1,
        "priority": 1,
        "cpus": 4,
        "memory_mb": 28_672,
        "scheduling_profile": "standard",
        "gpus": 0,
        "max_workers_per_node": 32,
        "timeout_seconds": 86_400,
        "scheduler_dedupe_policy": (
            "parent-and-logical-dedupes-absent-from-complete-namespace-before-first-post"
        ),
        "remote_ready_reread_required": True,
        "phase_b_runner_required": True,
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    return {**unsigned, "config_sha256": canary.canonical_sha256(unsigned)}


def _package(*, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    stage = profiles.BY_ID[canary.ENTRY_STAGE_ID]
    selected_seed = stage.seed_window_end_exclusive - 1 if seed is None else seed
    child = launch.build_stage_task(
        _base_task(stage, selected_seed), stage=stage, wave="refill"
    )
    parent = phase_b.build_concurrent_batch_task([child])
    config = _config(parent)
    ready = {"schema_version": "test-ready", "bundle_id": config["expected_bundle_id"]}
    publication = {
        "receipt_sha256": "3" * 64,
        "ready": ready,
        "ready_sha256": canary.canonical_sha256(ready),
    }
    package = canary._assemble_package(
        config=config,
        plan={
            "bundle_id": config["expected_bundle_id"],
            "bundle_manifest_sha256": config["expected_bundle_manifest_sha256"],
            "contract_sha256": config["expected_bundle_contract_sha256"],
            "remote_bundle": parent["remote_cwd"],
        },
        publication=publication,
        parent_task=parent,
        offload_plan_file_sha256=config["expected_offload_plan_file_sha256"],
        publication_receipt_file_sha256="4" * 64,
        required_runtime_code_sha256="5" * 64,
    )
    return package, publication


def _scheduler_row(task: dict[str, Any], task_id: int = 99001) -> dict[str, Any]:
    return {
        "id": task_id,
        "task_id": task_id,
        "status": "queued",
        "state": "queued",
        "name": task["name"],
        "dedupe_key": task["dedupe_key"],
        "task_json": copy.deepcopy(task),
    }


class FakeTransport:
    def __init__(self, ready: dict[str, Any], events: list[str]):
        self.ready = ready
        self.events = events

    def read_bytes(self, path: str) -> bytes:
        self.events.append("ready")
        assert path.endswith("/READY.json")
        return json.dumps(self.ready).encode("utf-8")


class FakeScheduler:
    def __init__(
        self,
        task: dict[str, Any],
        events: list[str],
        rows: list[dict[str, Any]] | None = None,
    ):
        self.task = task
        self.events = events
        self.rows = rows or []
        self.post_count = 0
        self.row = _scheduler_row(task)

    def list_complete_namespace_tasks(self) -> list[dict[str, Any]]:
        self.events.append("inventory")
        return copy.deepcopy(self.rows)

    def submit_task(self, task: dict[str, Any]) -> dict[str, Any]:
        self.events.append("post")
        assert task == self.task
        self.post_count += 1
        return copy.deepcopy(self.row)

    def get_task(self, task_id: int) -> dict[str, Any]:
        self.events.append("detail")
        assert task_id == self.row["id"]
        return copy.deepcopy(self.row)


def test_committed_entry_delta_config_is_exactly_sealed():
    path = (
        Path(__file__).parents[1]
        / "docs"
        / "evidence"
        / "tier1_final1000_phase_b_entry_shape1_canary_20260723.json"
    )
    value = canary.validate_config(json.loads(path.read_text(encoding="utf-8")))
    assert value["expected_bundle_id"] == "current7-d9a2dd76acad776f3015"
    assert value["seed"] == 2_257_499_999
    assert (value["cpus"], value["memory_mb"], value["priority"]) == (
        4,
        28_672,
        1,
    )


def test_shape1_retry_accepts_explicit_unused_seed_inside_entry_window():
    stage = profiles.BY_ID[canary.ENTRY_STAGE_ID]
    retry_seed = stage.seed_window_end_exclusive - 2
    package, _publication = _package(seed=retry_seed)
    config = _config(package["parent_task"])

    assert retry_seed == 2_257_499_998
    assert canary.validate_config(config) == config
    assert package["seed"] == retry_seed
    assert package["parent_task"]["payload_json"]["children"][0]["seed"] == retry_seed


@pytest.mark.parametrize("offset", [-1, 0])
def test_shape1_retry_rejects_seed_outside_authenticated_entry_window(offset: int):
    stage = profiles.BY_ID[canary.ENTRY_STAGE_ID]
    seed = (
        stage.seed_start - 1
        if offset < 0
        else stage.seed_window_end_exclusive
    )
    package, _publication = _package()
    config = _config(
        package["parent_task"],
        seed=seed,
        seed_selection_policy=canary.EXPLICIT_SEED_SELECTION_POLICY,
    )
    with pytest.raises(RuntimeError, match="config seal mismatch"):
        canary.validate_config(config)


def test_package_uses_phase_b_parent_and_exposes_all_terminal_evidence():
    package, _publication = _package()
    sealed = canary.validate_package(package)
    task = sealed["parent_task"]
    assert task["payload_json"]["batch_length"] == 1
    assert "tier1_final1000_multiseed_phase_b_runner.py" in task["command"]
    assert "tier1_final1000_multiseed_lane_runner.py" not in task["command"]
    assert (task["cpus"], task["memory_mb"], task["priority"]) == (4, 28_672, 1)
    assert task["payload_json"]["fea_submission_performed"] is False
    assert task["payload_json"]["aedt_used"] is False
    evidence = sealed["terminal_evidence"]
    assert evidence["child_receipt_path"].endswith("/seed_status.json")
    assert evidence["child_stdout_path"].endswith("/child_stdout.log")
    assert evidence["child_stderr_path"].endswith("/child_stderr.log")
    assert "semlock_stress_sha256" in evidence["semlock_evidence_json_pointer"]
    assert evidence["affinity_evidence_json_pointer"] == "/cpu_set"
    assert "observed_peak_rss_bytes" in evidence["rss_evidence_json_pointer"]
    assert "dispatch_fill_seconds" in evidence["dispatch_evidence_json_pointer"]
    assert "parent_wall_time_seconds" in evidence[
        "parent_elapsed_seconds_json_pointer"
    ]
    assert "aggregate_child_cpu_seconds" in evidence[
        "total_child_cpu_seconds_json_pointer"
    ]
    assert evidence["derived_cpu_metrics"]["parent_average_cores_used"] == (
        "aggregate_child_cpu_seconds / parent_wall_time_seconds"
    )
    assert sealed["terminal_gates"]["maximum_dispatch_fill_seconds"] == 35.0
    assert sealed["terminal_gates"][
        "minimum_weighted_average_cores_used_per_logical_seed"
    ] == 0.75
    assert sealed["terminal_gates"][
        "low_cpu_utilization_blocks_shape_promotion"
    ] is True


def test_dry_run_rereads_ready_before_complete_inventory_and_never_posts():
    package, publication = _package()
    events: list[str] = []
    scheduler = FakeScheduler(package["parent_task"], events)
    result = canary._submission_outcome(
        package=package,
        publication=publication,
        transport=FakeTransport(publication["ready"], events),
        scheduler=scheduler,
        apply=False,
        receipt_out=None,
    )
    assert events == ["ready", "inventory"]
    assert result["both_dedupes_absent_before_post"] is True
    assert result["submitted_count"] == 0
    assert result["scheduler_post_count"] == 0
    assert result["task_status"] == "absent"


@pytest.mark.parametrize("collision", ["parent", "logical"])
def test_any_prior_parent_or_logical_dedupe_fails_closed(collision: str):
    package, publication = _package()
    task = package["parent_task"]
    row = _scheduler_row(task)
    if collision == "logical":
        row.pop("task_json")
        row["dedupe_key"] = package["logical_child_dedupe_key"]
    events: list[str] = []
    scheduler = FakeScheduler(task, events, rows=[row])
    with pytest.raises(RuntimeError, match="not distinct from all prior tasks"):
        canary._submission_outcome(
            package=package,
            publication=publication,
            transport=FakeTransport(publication["ready"], events),
            scheduler=scheduler,
            apply=False,
            receipt_out=None,
        )
    assert scheduler.post_count == 0


def test_explicit_retry_seed_reuse_is_blocked_by_complete_namespace():
    stage = profiles.BY_ID[canary.ENTRY_STAGE_ID]
    package, publication = _package(seed=stage.seed_window_end_exclusive - 2)
    row = _scheduler_row(package["parent_task"])
    row.pop("task_json")
    row["dedupe_key"] = package["logical_child_dedupe_key"]
    scheduler = FakeScheduler(package["parent_task"], [], rows=[row])

    with pytest.raises(RuntimeError, match="not distinct from all prior tasks"):
        canary._submission_outcome(
            package=package,
            publication=publication,
            transport=FakeTransport(publication["ready"], []),
            scheduler=scheduler,
            apply=False,
            receipt_out=None,
        )
    assert scheduler.post_count == 0


def test_apply_posts_exactly_one_parent_and_writes_immutable_receipt(tmp_path: Path):
    package, publication = _package()
    events: list[str] = []
    scheduler = FakeScheduler(package["parent_task"], events)
    receipt = tmp_path / "shape1_submission.json"
    result = canary._submission_outcome(
        package=package,
        publication=publication,
        transport=FakeTransport(publication["ready"], events),
        scheduler=scheduler,
        apply=True,
        receipt_out=receipt,
    )
    assert events == ["ready", "inventory", "post", "detail"]
    assert result["submitted_count"] == 1
    assert result["scheduler_post_count"] == 1
    assert result["scheduler_cancel_count"] == 0
    assert result["scheduler_preempt_count"] == 0
    assert receipt.is_file()
    assert canary._validate_submission_receipt(
        json.loads(receipt.read_text(encoding="utf-8")), package=package
    ) == result


def test_resource_or_runner_tamper_is_rejected():
    package, _publication = _package()
    config = _config(package["parent_task"])
    tampered = copy.deepcopy(package["parent_task"])
    tampered["memory_mb"] -= 1
    with pytest.raises(RuntimeError, match="Scheduler task seal mismatch"):
        canary._assert_exact_parent(
            tampered, config=config, remote_bundle=tampered["remote_cwd"]
        )


def _cpu_evidence_inputs(
    shape: int, *, average_cores_per_seed: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    child_wall = 100.0
    cpu_per_child = child_wall * average_cores_per_seed
    parent_wall = child_wall + (shape - 1) * 5.0
    total_cpu = cpu_per_child * shape
    receipts = [
        {
            "ordinal": ordinal,
            "seed": 1000 + ordinal,
            "state": "completed",
            "exit_code": 0,
            "failure": None,
            "result_sha256": "a" * 64,
            "cpu_set": list(range(ordinal * 4, ordinal * 4 + 4)),
            "legacy_status": {
                "phase_b_semlock_stress_sha256": "b" * 64,
                "observed_peak_rss_bytes": 1024,
            },
            "child_resource_telemetry": {
                "available": True,
                "wall_time_seconds": child_wall,
                "process_tree_cpu_seconds": cpu_per_child,
            },
        }
        for ordinal in range(shape)
    ]
    status = {
        "state": "completed",
        "completed_child_count": shape,
        "failed_child_count": 0,
        "resource_telemetry": {
            "available": True,
            "parent_requested_cpus": shape * 4,
            "logical_child_count": shape,
            "parent_wall_time_seconds": parent_wall,
            "aggregate_child_cpu_seconds": total_cpu,
            "aggregate_cpu_utilization_fraction": total_cpu / (parent_wall * shape * 4),
            "sum_child_wall_time_seconds": child_wall * shape,
            "reported_child_process_tree_cpu_seconds": total_cpu,
            "reported_child_cpu_available_count": shape,
            "reported_child_cpu_utilization_fraction": average_cores_per_seed / 4,
            "dispatch_fill_seconds": (shape - 1) * 5.0,
        },
    }
    return status, receipts


def test_terminal_cpu_evaluator_derives_core_use_and_blocks_low_utilization(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(canary, "validate_task_status", lambda value: dict(value))
    monkeypatch.setattr(canary, "validate_child_receipt", lambda value: dict(value))
    status1, receipts1 = _cpu_evidence_inputs(1, average_cores_per_seed=1.0)
    baseline = canary.evaluate_terminal_cpu_evidence(
        status1, receipts1, expected_shape=1
    )
    assert baseline["total_child_cpu_seconds"] == 100.0
    assert baseline["parent_elapsed_seconds"] == 100.0
    assert baseline["parent_average_cores_used"] == 1.0
    assert baseline["weighted_average_cores_used_per_logical_seed"] == 1.0
    assert baseline["promotion_eligible"] is True

    status8, receipts8 = _cpu_evidence_inputs(8, average_cores_per_seed=0.5)
    low = canary.evaluate_terminal_cpu_evidence(
        status8,
        receipts8,
        expected_shape=8,
        baseline_metrics=baseline,
    )
    assert low["parent_requested_cpus"] == 32
    assert low["weighted_average_cores_used_per_logical_seed"] == 0.5
    assert low["per_seed_core_use_ratio_vs_baseline"] == 0.5
    assert low["low_cpu_utilization_detected"] is True
    assert "affinity-thread-environment" in low["low_cpu_diagnosis"]
    assert low["promotion_eligible"] is False
    assert low["automatic_promotion_performed"] is False


def _submission_receipt_for_package(
    package: dict[str, Any], *, task_id: int = 99001
) -> dict[str, Any]:
    unsigned = {
        "schema_version": canary.SUBMISSION_SCHEMA,
        "observed_at": "2026-07-23T02:42:57+09:00",
        "apply": True,
        "package_sha256": package["package_sha256"],
        "bundle_id": package["bundle_id"],
        "ready_sha256": package["ready_sha256"],
        "remote_ready_reread_count": 1,
        "complete_namespace_row_count": 1,
        "complete_namespace_sha256": "c" * 64,
        "parent_dedupe_key": package["parent_dedupe_key"],
        "logical_child_dedupe_key": package["logical_child_dedupe_key"],
        "both_dedupes_absent_before_post": True,
        "task_id": task_id,
        "task_status": "queued",
        "submitted_count": 1,
        "scheduler_endpoint": "POST /api/tasks",
        "scheduler_post_count": 1,
        "scheduler_cancel_count": 0,
        "scheduler_preempt_count": 0,
        "remote_write_count": 0,
        "terminal_evidence": package["terminal_evidence"],
        "terminal_gates": package["terminal_gates"],
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    return {**unsigned, "receipt_sha256": canary.canonical_sha256(unsigned)}


def test_remote_terminal_evaluator_refuses_nonterminal_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    package, _publication = _package()
    package_path = tmp_path / "package.json"
    submission_path = tmp_path / "submission.json"
    package_path.write_text(json.dumps(package), encoding="utf-8")
    submission_path.write_text(
        json.dumps(_submission_receipt_for_package(package)), encoding="utf-8"
    )
    events: list[str] = []
    scheduler = FakeScheduler(package["parent_task"], events)
    monkeypatch.setattr(canary, "SchedulerApiClient", lambda _url: scheduler)
    with pytest.raises(RuntimeError, match="not terminal: queued"):
        canary.evaluate_remote_terminal(
            package_path=package_path,
            submission_receipt_path=submission_path,
            scheduler_url="http://scheduler.test",
            task_id=99001,
            output_path=tmp_path / "terminal.json",
        )
    assert not (tmp_path / "terminal.json").exists()


def test_remote_terminal_evaluator_binds_files_result_and_cpu_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    package, _publication = _package()
    package_path = tmp_path / "package.json"
    submission_path = tmp_path / "submission.json"
    package_path.write_text(json.dumps(package), encoding="utf-8")
    submission_path.write_text(
        json.dumps(_submission_receipt_for_package(package)), encoding="utf-8"
    )
    row = _scheduler_row(package["parent_task"])
    row["status"] = "completed"
    row["state"] = "succeeded"

    class CompletedScheduler:
        post_count = 0

        def get_task(self, task_id: int) -> dict[str, Any]:
            assert task_id == 99001
            return copy.deepcopy(row)

    monkeypatch.setattr(canary, "SchedulerApiClient", lambda _url: CompletedScheduler())
    result_bytes = b'{"result":"ok"}\n'
    result_sha = canary.hashlib.sha256(result_bytes).hexdigest()
    stdout_bytes = b"optimizer stdout\n"
    stderr_bytes = b""
    manifest = {"manifest_sha256": "d" * 64}
    status = {"status_sha256": "e" * 64}
    child = {
        "receipt_sha256": "f" * 64,
        "result_sha256": result_sha,
        "stdout_relative_path": f"seed-{package['seed']}/child_stdout.log",
        "stdout_sha256": canary.hashlib.sha256(stdout_bytes).hexdigest(),
        "stdout_size_bytes": len(stdout_bytes),
        "stderr_relative_path": f"seed-{package['seed']}/child_stderr.log",
        "stderr_sha256": canary.hashlib.sha256(stderr_bytes).hexdigest(),
        "stderr_size_bytes": len(stderr_bytes),
    }
    raw_by_suffix = {
        "batch_manifest.json": json.dumps(manifest).encode(),
        "task_status.json": json.dumps(status).encode(),
        "seed_status.json": json.dumps(child).encode(),
        "child_stdout.log": stdout_bytes,
        "child_stderr.log": stderr_bytes,
        "result.json": result_bytes,
    }

    def remote_bytes(**kwargs: Any) -> bytes:
        path = kwargs["relative_path"]
        return next(value for suffix, value in raw_by_suffix.items() if path.endswith(suffix))

    metrics = {
        "terminal_cpu_evidence_sha256": "9" * 64,
        "promotion_eligible": True,
    }
    monkeypatch.setattr(canary, "_remote_file_bytes", remote_bytes)
    monkeypatch.setattr(canary, "validate_batch_manifest", lambda value: dict(value))
    monkeypatch.setattr(
        canary, "batch_manifest_from_payload", lambda _payload: copy.deepcopy(manifest)
    )
    monkeypatch.setattr(canary, "validate_task_status", lambda value: dict(value))
    monkeypatch.setattr(
        canary,
        "validate_child_receipt",
        lambda value, **_kwargs: dict(value),
    )
    monkeypatch.setattr(
        canary,
        "evaluate_terminal_cpu_evidence",
        lambda *_args, **_kwargs: copy.deepcopy(metrics),
    )
    output = tmp_path / "terminal.json"
    sealed = canary.evaluate_remote_terminal(
        package_path=package_path,
        submission_receipt_path=submission_path,
        scheduler_url="http://scheduler.test",
        task_id=99001,
        output_path=output,
    )
    assert sealed["scheduler_status"] == "completed"
    assert sealed["remote_files"]["result"]["sha256"] == result_sha
    assert sealed["remote_files"]["child_stdout"]["sha256"] == child[
        "stdout_sha256"
    ]
    assert sealed["remote_files"]["child_stderr"]["size"] == 0
    assert sealed["scheduler_remote_file_get_count"] == 6
    assert sealed["terminal_cpu_evidence_sha256"] == "9" * 64
    assert sealed["shape4_submission_allowed"] is True
    assert sealed["scheduler_post_count"] == 0
    assert output.is_file()
