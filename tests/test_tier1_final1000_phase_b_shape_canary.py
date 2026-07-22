from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tests.test_tier1_final1000_multiseed_phase_a import _base_task
from tools import tier1_final1000_multiseed_phase_b_contract as phase_b
from tools import tier1_final1000_phase_b_shape1_canary as shape1
from tools import tier1_final1000_phase_b_shape_canary as canary
from tools import tier1_final1000_slurm_launch as launch
from tools import tier1_final1000_stage_profiles as profiles


def _seeds(shape: int) -> list[int]:
    end = profiles.BY_ID[canary.ENTRY_STAGE_ID].seed_window_end_exclusive
    return list(range(end - shape - 20, end - 20))


def _parent(shape: int, *, seeds: list[int] | None = None) -> dict[str, Any]:
    stage = profiles.BY_ID[canary.ENTRY_STAGE_ID]
    selected_seeds = _seeds(shape) if seeds is None else list(seeds)
    children = [
        launch.build_stage_task(_base_task(stage, seed), stage=stage, wave="refill")
        for seed in selected_seeds
    ]
    return phase_b.build_concurrent_batch_task(children)


def _config(
    parent: dict[str, Any], *, shape: int, baseline_sha: str | None = None
) -> dict[str, Any]:
    payload = parent["payload_json"]
    seeds = [int(child["seed"]) for child in payload["children"]]
    unsigned = {
        "schema_version": canary.CONFIG_SCHEMA,
        "diagnostic_id": f"test-entry-shape{shape}-canary-v1",
        "stage_id": canary.ENTRY_STAGE_ID,
        "seeds": seeds,
        "seed_selection_policy": canary.SEED_SELECTION_POLICY,
        "expected_offload_plan_file_sha256": "1" * 64,
        "expected_bundle_id": payload["bundle_id"],
        "expected_bundle_manifest_sha256": payload["bundle_manifest_sha256"],
        "expected_bundle_contract_sha256": "2" * 64,
        "lane_shape": shape,
        "logical_seed_count": shape,
        "required_shape4_remote_terminal_sha256": baseline_sha,
        "priority": 1,
        "cpus": shape * phase_b.CHILD_CPUS,
        "memory_mb": shape * phase_b.CHILD_MEMORY_MB,
        "scheduling_profile": "standard",
        "gpus": 0,
        "max_workers_per_node": 32,
        "timeout_seconds": 86_400,
        "scheduler_dedupe_policy": canary.SCHEDULER_DEDUPE_POLICY,
        "remote_ready_reread_required": True,
        "phase_b_runner_required": True,
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    return {**unsigned, "config_sha256": canary.canonical_sha256(unsigned)}


def _package(
    shape: int,
    *,
    baseline_sha: str | None = None,
    seeds: list[int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parent = _parent(shape, seeds=seeds)
    config = _config(parent, shape=shape, baseline_sha=baseline_sha)
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
    return canary.validate_package(package), publication


def _scheduler_row(task: dict[str, Any], task_id: int = 99101) -> dict[str, Any]:
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
        return json.dumps(self.ready).encode()


class FakeScheduler:
    def __init__(
        self,
        task: dict[str, Any],
        events: list[str],
        *,
        rows: list[dict[str, Any]] | None = None,
        concurrent_rows_after_post: list[dict[str, Any]] | None = None,
        publish_submitted_row: bool = True,
    ):
        self.task = task
        self.events = events
        self.rows = rows or []
        self.concurrent_rows_after_post = concurrent_rows_after_post or []
        self.publish_submitted_row = publish_submitted_row
        self.post_count = 0
        self.row = _scheduler_row(task)

    def list_complete_namespace_tasks(self) -> list[dict[str, Any]]:
        self.events.append("inventory")
        return copy.deepcopy(self.rows)

    def submit_task(self, task: dict[str, Any]) -> dict[str, Any]:
        self.events.append("post")
        assert task == self.task
        self.post_count += 1
        if self.publish_submitted_row:
            self.rows.append(copy.deepcopy(self.row))
        self.rows.extend(copy.deepcopy(self.concurrent_rows_after_post))
        return copy.deepcopy(self.row)

    def get_task(self, task_id: int) -> dict[str, Any]:
        self.events.append("detail")
        assert task_id == self.row["id"]
        return copy.deepcopy(self.row)


@pytest.mark.parametrize("shape", [1, 4, 8])
def test_package_seals_exact_concurrent_parent_and_every_logical_identity(shape: int):
    baseline = "9" * 64 if shape == 8 else None
    package, _publication = _package(shape, baseline_sha=baseline)
    parent = package["parent_task"]
    assert parent["payload_json"]["batch_length"] == shape
    assert parent["payload_json"]["concurrent_children"] == shape
    assert (parent["cpus"], parent["memory_mb"]) == (
        shape * 4,
        shape * 28_672,
    )
    assert len(package["logical_child_dedupe_keys"]) == shape
    assert len(set(package["logical_child_dedupe_keys"])) == shape
    assert package["parent_dedupe_key"] not in package["logical_child_dedupe_keys"]
    assert len(package["terminal_evidence"]["children"]) == shape
    assert package["terminal_gates"]["shape4_baseline_required"] is (shape == 8)
    assert package["terminal_gates"]["minimum_throughput_ratio_vs_shape4"] == (
        1.7 if shape == 8 else None
    )
    assert "tier1_final1000_multiseed_phase_b_runner.py" in parent["command"]
    assert parent["payload_json"]["fea_submission_performed"] is False
    assert parent["payload_json"]["aedt_used"] is False


def test_shape8_config_fails_closed_without_shape4_terminal_lineage():
    parent = _parent(8)
    with pytest.raises(RuntimeError, match="config seal mismatch"):
        canary.validate_config(_config(parent, shape=8, baseline_sha=None))


def test_generic_shape1_supports_explicit_2257499998_retry_identity():
    retry_seed = 2_257_499_998
    package, _publication = _package(1, seeds=[retry_seed])

    assert package["lane_shape"] == 1
    assert package["seeds"] == [retry_seed]
    assert package["parent_task"]["cpus"] == 4
    assert package["parent_task"]["memory_mb"] == 28_672
    assert package["logical_child_dedupe_keys"] == [
        package["parent_task"]["payload_json"]["children"][0][
            "logical_dedupe_key"
        ]
    ]


def test_generic_shape1_retry_reuse_is_blocked_by_complete_namespace():
    package, publication = _package(1, seeds=[2_257_499_998])
    row = _scheduler_row(package["parent_task"])
    row.pop("task_json")
    row["dedupe_key"] = package["logical_child_dedupe_keys"][0]
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


@pytest.mark.parametrize("shape", [1, 4, 8])
def test_complete_namespace_dry_run_never_posts(shape: int):
    package, publication = _package(
        shape, baseline_sha="9" * 64 if shape == 8 else None
    )
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
    assert result["all_dedupes_absent_before_post"] is True
    assert result["submitted_parent_count"] == 0
    assert result["scheduler_post_count"] == 0


def test_any_logical_dedupe_collision_in_full_namespace_blocks_post():
    package, publication = _package(4)
    row = _scheduler_row(package["parent_task"])
    row.pop("task_json")
    row["dedupe_key"] = package["logical_child_dedupe_keys"][2]
    events: list[str] = []
    scheduler = FakeScheduler(package["parent_task"], events, rows=[row])
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


@pytest.mark.parametrize("shape", [1, 4, 8])
def test_apply_performs_one_additive_parent_post_only(tmp_path: Path, shape: int):
    package, publication = _package(
        shape, baseline_sha="9" * 64 if shape == 8 else None
    )
    events: list[str] = []
    scheduler = FakeScheduler(package["parent_task"], events)
    receipt_path = tmp_path / f"shape{shape}-submission.json"
    result = canary._submission_outcome(
        package=package,
        publication=publication,
        transport=FakeTransport(publication["ready"], events),
        scheduler=scheduler,
        apply=True,
        receipt_out=receipt_path,
    )
    assert events == ["ready", "inventory", "post", "detail", "inventory"]
    assert result["submitted_parent_count"] == 1
    assert result["scheduler_post_count"] == 1
    assert result["scheduler_cancel_count"] == 0
    assert result["scheduler_preempt_count"] == 0
    assert result["post_submit_parent_match_count"] == 1
    assert result["post_submit_logical_match_count"] == 0
    assert result["all_dedupes_revalidated_after_post"] is True
    assert canary._validate_submission_receipt(
        json.loads(receipt_path.read_text()), package=package
    ) == result


def test_post_submit_namespace_race_fails_closed_after_exactly_one_post(
    tmp_path: Path,
):
    package, publication = _package(4)
    raced = _scheduler_row(package["parent_task"], task_id=99102)
    raced.pop("task_json")
    raced["dedupe_key"] = package["logical_child_dedupe_keys"][2]
    events: list[str] = []
    scheduler = FakeScheduler(
        package["parent_task"],
        events,
        concurrent_rows_after_post=[raced],
    )
    receipt_path = tmp_path / "raced-submission.json"

    with pytest.raises(RuntimeError, match="identity changed during Scheduler POST"):
        canary._submission_outcome(
            package=package,
            publication=publication,
            transport=FakeTransport(publication["ready"], events),
            scheduler=scheduler,
            apply=True,
            receipt_out=receipt_path,
        )

    assert scheduler.post_count == 1
    assert events == ["ready", "inventory", "post", "detail", "inventory"]
    assert not receipt_path.exists()


def test_post_submit_namespace_must_contain_authenticated_parent(tmp_path: Path):
    package, publication = _package(1)
    scheduler = FakeScheduler(
        package["parent_task"], [], publish_submitted_row=False
    )

    with pytest.raises(RuntimeError, match="identity changed during Scheduler POST"):
        canary._submission_outcome(
            package=package,
            publication=publication,
            transport=FakeTransport(publication["ready"], []),
            scheduler=scheduler,
            apply=True,
            receipt_out=tmp_path / "missing-parent.json",
        )

    assert scheduler.post_count == 1


def _cpu_inputs(
    shape: int,
    *,
    parent_wall: float,
    average_cores_per_seed: float = 1.0,
    dispatch_fill_seconds: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    child_wall = 100.0
    cpu_per_child = child_wall * average_cores_per_seed
    total_cpu = cpu_per_child * shape
    receipts = []
    for ordinal in range(shape):
        cpu_set = list(range(ordinal * 4, ordinal * 4 + 4))
        legacy = {
            "phase_b_semlock_stress_sha256": "b" * 64,
            "observed_peak_rss_bytes": 1024,
            "phase_b_cpu_set": cpu_set,
        }
        receipts.append({
            "receipt_sha256": f"{ordinal + 1:x}" * 64,
            "task_id": "99101",
            "manifest_sha256": "d" * 64,
            "ordinal": ordinal,
            "seed": 1000 + ordinal,
            "payload_sha256": f"{ordinal + 5:x}" * 64,
            "logical_dedupe_key": f"logical-{ordinal}",
            "state": "completed",
            "exit_code": 0,
            "failure": None,
            "result_sha256": "a" * 64,
            "stdout_relative_path": f"seed-{1000 + ordinal}/child_stdout.log",
            "stdout_sha256": "1" * 64,
            "stdout_size_bytes": 10,
            "stderr_relative_path": f"seed-{1000 + ordinal}/child_stderr.log",
            "stderr_sha256": "2" * 64,
            "stderr_size_bytes": 0,
            "cpu_set": cpu_set,
            "legacy_status": legacy,
            "legacy_status_sha256": canary.canonical_sha256(legacy),
            "child_resource_telemetry": {
                "available": True,
                "wall_time_seconds": child_wall,
                "process_tree_cpu_seconds": cpu_per_child,
            },
        })
    status = {
        "status_sha256": "e" * 64,
        "task_id": "99101",
        "manifest_sha256": "d" * 64,
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
            "dispatch_fill_seconds": (
                (shape - 1) * 5.0
                if dispatch_fill_seconds is None
                else dispatch_fill_seconds
            ),
        },
    }
    return status, receipts


def _memory(shape: int, *, safe: bool = True) -> dict[str, Any]:
    requested = shape * phase_b.CHILD_MEMORY_MB * 1024**2
    observations = [
        {
            "ordinal": ordinal,
            "seed": 1000 + ordinal,
            "legacy_status_sha256": canary.canonical_sha256(
                {
                    "phase_b_semlock_stress_sha256": "b" * 64,
                    "observed_peak_rss_bytes": 1024,
                    "phase_b_cpu_set": list(
                        range(ordinal * 4, ordinal * 4 + 4)
                    ),
                }
            ),
            "observed_peak_rss_bytes": 1024,
        }
        for ordinal in range(shape)
    ]
    return {
        "evidence_sha256": "f" * 64,
        "task_id": "99101",
        "manifest_sha256": "d" * 64,
        "task_status_sha256": "e" * 64,
        "logical_child_count": shape,
        "child_requested_memory_bytes": phase_b.CHILD_MEMORY_MB * 1024**2,
        "parent_requested_memory_bytes": requested,
        "child_peak_rss_available_count": shape,
        "child_peak_rss_sum_bytes": shape * 1024,
        "child_peak_rss_max_bytes": 1024,
        "child_peak_rss_observations_sha256": canary.canonical_sha256(
            observations
        ),
        "cgroup_available": safe,
        "cgroup_peak_bytes": shape * 1024,
        "cgroup_limit_bytes": requested if safe else None,
        "cgroup_limit_unbounded": not safe,
        "total_child_peak_rss_within_parent_request": True,
        "cgroup_peak_within_limit": safe,
        "cgroup_limit_covers_parent_request": safe,
        "safety_passed": safe,
    }


def _patch_terminal_validators(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(canary, "validate_task_status", lambda value: dict(value))
    monkeypatch.setattr(
        canary, "validate_child_receipt", lambda value, **_kwargs: dict(value)
    )
    monkeypatch.setattr(
        canary, "validate_parent_memory_evidence", lambda value: dict(value)
    )
    monkeypatch.setattr(shape1, "validate_task_status", lambda value: dict(value))
    monkeypatch.setattr(shape1, "validate_child_receipt", lambda value: dict(value))


def test_shape8_promotion_requires_1p7x_shape4_and_memory_cpu_safety(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_terminal_validators(monkeypatch)
    status4, receipts4 = _cpu_inputs(4, parent_wall=100.0)
    baseline = canary.evaluate_terminal_evidence(
        status4,
        receipts4,
        _memory(4),
        expected_shape=4,
        semlock_or_enospc_log_occurrences=0,
    )
    assert baseline["promotion_eligible"] is True
    assert baseline["next_shape_allowed"] == 8

    status8, receipts8 = _cpu_inputs(8, parent_wall=110.0)
    promoted = canary.evaluate_terminal_evidence(
        status8,
        receipts8,
        _memory(8),
        expected_shape=8,
        semlock_or_enospc_log_occurrences=0,
        shape4_baseline=baseline,
    )
    assert promoted["throughput_ratio_vs_shape4"] == pytest.approx(20.0 / 11.0)
    assert promoted["weighted_average_cores_used_per_logical_seed"] == 1.0
    assert promoted["production_shape8_promotion_eligible"] is True
    assert promoted["promotion_eligible"] is True

    slow_status, slow_receipts = _cpu_inputs(8, parent_wall=130.0)
    slow = canary.evaluate_terminal_evidence(
        slow_status,
        slow_receipts,
        _memory(8),
        expected_shape=8,
        semlock_or_enospc_log_occurrences=0,
        shape4_baseline=baseline,
    )
    assert slow["throughput_ratio_vs_shape4"] < 1.7
    assert slow["gate_results"]["throughput_at_least_1p7x_shape4"] is False
    assert slow["promotion_eligible"] is False


@pytest.mark.parametrize("failure", ["low-cores", "unsafe-memory", "semlock-log"])
def test_shape8_safety_failures_block_promotion(
    monkeypatch: pytest.MonkeyPatch, failure: str
):
    _patch_terminal_validators(monkeypatch)
    status4, receipts4 = _cpu_inputs(4, parent_wall=100.0)
    baseline = canary.evaluate_terminal_evidence(
        status4,
        receipts4,
        _memory(4),
        expected_shape=4,
        semlock_or_enospc_log_occurrences=0,
    )
    average = 0.5 if failure == "low-cores" else 1.0
    status8, receipts8 = _cpu_inputs(
        8, parent_wall=110.0, average_cores_per_seed=average
    )
    result = canary.evaluate_terminal_evidence(
        status8,
        receipts8,
        _memory(8, safe=failure != "unsafe-memory"),
        expected_shape=8,
        semlock_or_enospc_log_occurrences=1 if failure == "semlock-log" else 0,
        shape4_baseline=baseline,
    )
    assert result["promotion_eligible"] is False
    assert result["production_shape8_promotion_eligible"] is False


def test_terminal_identity_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_terminal_validators(monkeypatch)
    status, receipts = _cpu_inputs(4, parent_wall=100.0)
    memory = _memory(4)
    memory["task_status_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="identity drifted"):
        canary.evaluate_terminal_evidence(
            status,
            receipts,
            memory,
            expected_shape=4,
            semlock_or_enospc_log_occurrences=0,
        )


@pytest.mark.parametrize("failure", ["negative", "overlap", "legacy-mismatch"])
def test_receipt_cpu_set_must_match_nonnegative_applied_legacy_affinity(
    monkeypatch: pytest.MonkeyPatch, failure: str
):
    _patch_terminal_validators(monkeypatch)
    status, receipts = _cpu_inputs(4, parent_wall=100.0)
    if failure == "negative":
        receipts[0]["cpu_set"][0] = -1
        receipts[0]["legacy_status"]["phase_b_cpu_set"][0] = -1
    elif failure == "overlap":
        receipts[1]["cpu_set"] = list(receipts[0]["cpu_set"])
        receipts[1]["legacy_status"]["phase_b_cpu_set"] = list(
            receipts[0]["cpu_set"]
        )
    else:
        receipts[2]["legacy_status"]["phase_b_cpu_set"] = [40, 41, 42, 43]

    with pytest.raises(RuntimeError, match="receipt/applied CPU-set identity drifted"):
        canary.evaluate_terminal_evidence(
            status,
            receipts,
            _memory(4),
            expected_shape=4,
            semlock_or_enospc_log_occurrences=0,
        )


def test_shape4_dispatch_uses_15_second_shape_gate(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_terminal_validators(monkeypatch)
    status, receipts = _cpu_inputs(
        4, parent_wall=100.0, dispatch_fill_seconds=30.0
    )
    terminal = canary.evaluate_terminal_evidence(
        status,
        receipts,
        _memory(4),
        expected_shape=4,
        semlock_or_enospc_log_occurrences=0,
    )
    assert terminal["maximum_dispatch_fill_seconds"] == 15.0
    assert terminal["gate_results"]["dispatch_fill_within_shape_gate"] is False
    assert terminal["promotion_eligible"] is False


def test_parent_cpu_summary_cannot_hide_idle_child_receipts(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_terminal_validators(monkeypatch)
    status, receipts = _cpu_inputs(4, parent_wall=100.0)
    for receipt in receipts:
        receipt["child_resource_telemetry"]["process_tree_cpu_seconds"] = 0.0
    terminal = canary.evaluate_terminal_evidence(
        status,
        receipts,
        _memory(4),
        expected_shape=4,
        semlock_or_enospc_log_occurrences=0,
    )
    assert terminal["weighted_average_cores_used_per_logical_seed"] == 0.0
    assert terminal["gate_results"][
        "parent_and_child_cpu_telemetry_reconciled"
    ] is False
    assert terminal["promotion_eligible"] is False


def test_fault_scan_only_counts_error_signatures():
    assert canary._fault_occurrences([b"semlock safe helper passed\n"]) == 0
    assert canary._fault_occurrences([b"ENOSPC not observed\n"]) == 0
    assert canary._fault_occurrences(
        [b"OSError: [Errno 28] No space left on device\n"]
    ) >= 1
    assert canary._fault_occurrences(
        [
            b"Traceback (most recent call last):\n"
            b"  File 'multiprocessing/synchronize.py', line 57, in __init__\n"
            b"    self._semlock = _multiprocessing.SemLock(kind, value, maxvalue)\n"
            b"FileNotFoundError: [Errno 2] No such file or directory\n"
        ]
    ) >= 1
    assert canary._fault_occurrences(
        [
            b"Traceback (most recent call last):\n"
            b"  File 'multiprocessing/synchronize.py', line 57, in __init__\n"
            b"    self._semlock = _multiprocessing.SemLock(kind, value, maxvalue)\n"
            b"PermissionError: [Errno 13] Permission denied\n"
        ]
    ) >= 1
    assert canary._fault_occurrences(
        [b"RuntimeError: unrelated stdout error\n", b"SemLock safe helper passed\n"]
    ) == 0


def _submission_receipt(package: dict[str, Any], *, task_id: int = 99101) -> dict[str, Any]:
    unsigned = {
        "schema_version": canary.SUBMISSION_SCHEMA,
        "observed_at": "2026-07-23T04:00:00+09:00",
        "apply": True,
        "package_sha256": package["package_sha256"],
        "bundle_id": package["bundle_id"],
        "lane_shape": package["lane_shape"],
        "ready_sha256": package["ready_sha256"],
        "remote_ready_reread_count": 1,
        "complete_namespace_row_count": 0,
        "complete_namespace_max_task_id": 0,
        "complete_namespace_sha256": "c" * 64,
        "post_submit_namespace_row_count": 1,
        "post_submit_namespace_max_task_id": task_id,
        "post_submit_namespace_sha256": "d" * 64,
        "post_submit_parent_match_count": 1,
        "post_submit_logical_match_count": 0,
        "all_dedupes_revalidated_after_post": True,
        "parent_dedupe_key": package["parent_dedupe_key"],
        "logical_child_dedupe_keys": package["logical_child_dedupe_keys"],
        "all_dedupes_absent_before_post": True,
        "task_id": task_id,
        "task_status": "queued",
        "submitted_parent_count": 1,
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


def test_remote_terminal_validator_binds_every_runtime_file_record(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_terminal_validators(monkeypatch)
    monkeypatch.setattr(canary, "validate_batch_manifest", lambda value: dict(value))
    status, receipts = _cpu_inputs(4, parent_wall=100.0)
    terminal = canary.evaluate_terminal_evidence(
        status,
        receipts,
        _memory(4),
        expected_shape=4,
        semlock_or_enospc_log_occurrences=0,
    )
    task_root = "runs/task-99101"
    files: dict[str, dict[str, Any]] = {
        "batch_manifest": {
            "relative_path": f"{task_root}/batch_manifest.json",
            "size": 100,
            "sha256": "3" * 64,
        },
        "task_status": {
            "relative_path": f"{task_root}/task_status.json",
            "size": 100,
            "sha256": "4" * 64,
        },
        "parent_memory_evidence": {
            "relative_path": f"{task_root}/{phase_b.PARENT_MEMORY_EVIDENCE_FILENAME}",
            "size": 100,
            "sha256": "5" * 64,
        },
    }
    for ordinal, receipt in enumerate(receipts):
        seed = receipt["seed"]
        files.update(
            {
                f"child_{ordinal}_receipt": {
                    "relative_path": f"{task_root}/seed-{seed}/seed_status.json",
                    "size": 100,
                    "sha256": "6" * 64,
                },
                f"child_{ordinal}_stdout": {
                    "relative_path": f"{task_root}/{receipt['stdout_relative_path']}",
                    "size": receipt["stdout_size_bytes"],
                    "sha256": receipt["stdout_sha256"],
                },
                f"child_{ordinal}_stderr": {
                    "relative_path": f"{task_root}/{receipt['stderr_relative_path']}",
                    "size": receipt["stderr_size_bytes"],
                    "sha256": receipt["stderr_sha256"],
                },
                f"child_{ordinal}_result": {
                    "relative_path": f"{task_root}/seed-{seed}/result.json",
                    "size": 100,
                    "sha256": receipt["result_sha256"],
                },
            }
        )
    manifest = {
        "manifest_sha256": "d" * 64,
        "ordered_children": [
            {
                "seed": receipt["seed"],
                "payload_sha256": receipt["payload_sha256"],
                "logical_dedupe_key": receipt["logical_dedupe_key"],
            }
            for receipt in receipts
        ],
    }
    manifest_bind_calls: list[int] = []

    def bind_receipt_to_manifest(
        value: dict[str, Any], *, manifest: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if manifest is not None:
            ordinal = int(value["ordinal"])
            manifest_bind_calls.append(ordinal)
            child = manifest["ordered_children"][ordinal]
            if (
                child["seed"] != value["seed"]
                or child["payload_sha256"] != value["payload_sha256"]
                or child["logical_dedupe_key"] != value["logical_dedupe_key"]
            ):
                raise RuntimeError(
                    "Phase B child receipt/manifest binding mismatch"
                )
        return dict(value)

    monkeypatch.setattr(canary, "validate_child_receipt", bind_receipt_to_manifest)
    unsigned = {
        "schema_version": canary.REMOTE_TERMINAL_SCHEMA,
        "observed_at": "2026-07-23T04:00:00+09:00",
        "package_sha256": "7" * 64,
        "submission_receipt_sha256": "8" * 64,
        "bundle_id": "current7-test",
        "lane_shape": 4,
        "logical_seed_count": 4,
        "task_id": 99101,
        "scheduler_status": "completed",
        "scheduler_detail_get_count": 2,
        "scheduler_remote_file_get_count": 19,
        "remote_files": files,
        "batch_manifest": manifest,
        "batch_manifest_sha256": manifest["manifest_sha256"],
        "task_status_sha256": terminal["task_status_sha256"],
        "parent_memory_evidence_sha256": terminal[
            "parent_memory_evidence_sha256"
        ],
        "child_receipt_sha256s": terminal["child_receipt_sha256s"],
        "shape4_baseline_remote_terminal_sha256": None,
        "terminal_evidence": terminal,
        "terminal_evidence_sha256": terminal["terminal_evidence_sha256"],
        "promotion_eligible": True,
        "next_shape_allowed": 8,
        "production_shape8_promotion_eligible": False,
        "scheduler_access": "GET-only",
        "scheduler_post_count": 0,
        "scheduler_cancel_count": 0,
        "scheduler_preempt_count": 0,
        "remote_access": "read-only",
        "remote_write_count": 0,
        "automatic_promotion_performed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    remote = {
        **unsigned,
        "remote_terminal_sha256": canary.canonical_sha256(unsigned),
    }
    assert canary.validate_remote_terminal(remote) == remote
    assert manifest_bind_calls == [0, 1, 2, 3]

    tampered = copy.deepcopy(remote)
    tampered["remote_files"]["child_0_stdout"]["sha256"] = "0" * 64
    tampered_unsigned = {
        key: value
        for key, value in tampered.items()
        if key != "remote_terminal_sha256"
    }
    tampered["remote_terminal_sha256"] = canary.canonical_sha256(
        tampered_unsigned
    )
    with pytest.raises(RuntimeError, match="file SHA/size drifted"):
        canary.validate_remote_terminal(tampered)

    # A fully re-sealed terminal artifact must still be rejected when a child
    # payload identity no longer matches the separately fetched manifest.
    rebound = copy.deepcopy(remote)
    rebound["terminal_evidence"]["child_receipts"][0][
        "payload_sha256"
    ] = "0" * 64
    terminal_unsigned = {
        key: value
        for key, value in rebound["terminal_evidence"].items()
        if key != "terminal_evidence_sha256"
    }
    rebound["terminal_evidence"]["terminal_evidence_sha256"] = (
        canary.canonical_sha256(terminal_unsigned)
    )
    rebound["terminal_evidence_sha256"] = rebound["terminal_evidence"][
        "terminal_evidence_sha256"
    ]
    rebound_unsigned = {
        key: value
        for key, value in rebound.items()
        if key != "remote_terminal_sha256"
    }
    rebound["remote_terminal_sha256"] = canary.canonical_sha256(
        rebound_unsigned
    )
    with pytest.raises(RuntimeError, match="receipt/manifest binding mismatch"):
        canary.validate_remote_terminal(rebound)


def test_remote_shape4_evaluator_is_get_only_and_hashes_every_child_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    package, _publication = _package(4)
    package_path = tmp_path / "package.json"
    submission_path = tmp_path / "submission.json"
    package_path.write_text(json.dumps(package), encoding="utf-8")
    submission_path.write_text(
        json.dumps(_submission_receipt(package)), encoding="utf-8"
    )
    row = _scheduler_row(package["parent_task"])
    row["status"] = "completed"
    row["state"] = "succeeded"

    class CompletedScheduler:
        post_count = 0

        def __init__(self) -> None:
            self.get_count = 0

        def get_task(self, task_id: int) -> dict[str, Any]:
            assert task_id == 99101
            self.get_count += 1
            return copy.deepcopy(row)

    scheduler = CompletedScheduler()
    monkeypatch.setattr(canary, "SchedulerApiClient", lambda _url: scheduler)
    manifest = {"manifest_sha256": "d" * 64}
    status = {"status_sha256": "e" * 64}
    memory = {"evidence_sha256": "f" * 64}
    raw_by_path: dict[str, bytes] = {
        "batch_manifest.json": json.dumps(manifest).encode(),
        "task_status.json": json.dumps(status).encode(),
        phase_b.PARENT_MEMORY_EVIDENCE_FILENAME: json.dumps(memory).encode(),
    }
    receipt_rows: dict[int, dict[str, Any]] = {}
    for ordinal, seed in enumerate(package["seeds"]):
        stdout = f"seed {seed} stdout\n".encode()
        stderr = b""
        result = f'{{"seed":{seed}}}\n'.encode()
        receipt = {
            "receipt_sha256": f"{ordinal + 1:x}" * 64,
            "ordinal": ordinal,
            "seed": seed,
            "stdout_relative_path": f"seed-{seed}/child_stdout.log",
            "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
            "stdout_size_bytes": len(stdout),
            "stderr_relative_path": f"seed-{seed}/child_stderr.log",
            "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
            "stderr_size_bytes": len(stderr),
            "result_sha256": hashlib.sha256(result).hexdigest(),
        }
        receipt_rows[seed] = receipt
        raw_by_path[f"seed-{seed}/seed_status.json"] = json.dumps(receipt).encode()
        raw_by_path[f"seed-{seed}/child_stdout.log"] = stdout
        raw_by_path[f"seed-{seed}/child_stderr.log"] = stderr
        raw_by_path[f"seed-{seed}/result.json"] = result

    def remote_bytes(**kwargs: Any) -> bytes:
        path = str(kwargs["relative_path"])
        return next(value for suffix, value in raw_by_path.items() if path.endswith(suffix))

    terminal = {
        "lane_shape": 4,
        "terminal_evidence_sha256": "9" * 64,
        "promotion_eligible": True,
        "next_shape_allowed": 8,
        "production_shape8_promotion_eligible": False,
    }
    monkeypatch.setattr(canary.shape1, "_remote_file_bytes", remote_bytes)
    monkeypatch.setattr(canary, "validate_batch_manifest", lambda value: dict(value))
    monkeypatch.setattr(
        canary, "batch_manifest_from_payload", lambda _payload: copy.deepcopy(manifest)
    )
    monkeypatch.setattr(canary, "validate_task_status", lambda value: dict(value))
    monkeypatch.setattr(
        canary, "validate_parent_memory_evidence", lambda value: dict(value)
    )
    monkeypatch.setattr(
        canary, "validate_child_receipt", lambda value, **_kwargs: dict(value)
    )
    monkeypatch.setattr(
        canary, "evaluate_terminal_evidence", lambda *_args, **_kwargs: copy.deepcopy(terminal)
    )
    monkeypatch.setattr(canary, "validate_remote_terminal", lambda value: dict(value))
    output = tmp_path / "terminal.json"
    sealed = canary.evaluate_remote_terminal(
        package_path=package_path,
        submission_receipt_path=submission_path,
        scheduler_url="http://scheduler.test",
        task_id=99101,
        output_path=output,
    )
    assert scheduler.get_count == 2
    assert sealed["scheduler_access"] == "GET-only"
    assert sealed["scheduler_post_count"] == 0
    assert sealed["scheduler_cancel_count"] == 0
    assert sealed["scheduler_remote_file_get_count"] == 19
    assert sealed["next_shape_allowed"] == 8
    for ordinal, seed in enumerate(package["seeds"]):
        assert sealed["remote_files"][f"child_{ordinal}_stdout"]["sha256"] == (
            receipt_rows[seed]["stdout_sha256"]
        )
        assert sealed["remote_files"][f"child_{ordinal}_stderr"]["size"] == 0
    assert output.is_file()
