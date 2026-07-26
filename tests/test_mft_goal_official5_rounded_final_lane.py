from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from tools import mft_goal_official5_rounded_final_collector as collector
from tools import mft_goal_official5_rounded_final_prepare as rounded
from tools import mft_goal_official5_rounded_final_submit as submitter


REVISION = "d" * 40
OBSERVED_AT = datetime(2026, 7, 26, 15, 0, tzinfo=timezone.utc)
TASK_ID = 98_765


class Lock:
    def __enter__(self) -> Lock:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


class GateReader:
    def __init__(self) -> None:
        self.health = {
            "ok": True,
            "scheduler_ok": True,
            "scheduler_thread_alive": True,
            "scheduler_stalled": False,
        }
        self.anchor = {
            "task_id": rounded.SOURCE_LINEAGE_TASK_ID,
            "name": submitter.SOURCE_ANCHOR_TASK_NAME,
            "dedupe_key": submitter.SOURCE_ANCHOR_DEDUPE_KEY,
            "project": rounded.PROJECT,
            "status": "failed",
            "requested_account_name": rounded.ACCOUNT_NAME,
            "requested_node_name": rounded.NODE_NAME,
            "account_name": rounded.ACCOUNT_NAME,
            "node_name": rounded.NODE_NAME,
            "actual_node_name": rounded.NODE_NAME,
            "allocation_node_name": rounded.NODE_NAME,
            "allocation_id": rounded.SOURCE_ALLOCATION_ID,
            "assigned_allocation": rounded.SOURCE_ALLOCATION_ID,
            "slurm_job_id": rounded.SOURCE_SLURM_JOB_ID,
            "cpus": rounded.CPUS,
            "memory_mb": rounded.MEMORY_MB,
            "aedt_backend": "standalone",
            "scheduling_profile": "fea_bursty",
        }
        self.capacity = {
            "queue_state": "ready",
            "queue_reason": (
                "ready to attach to allocation "
                f"{rounded.SOURCE_ALLOCATION_ID}"
            ),
            "ready_fit_slots": 8,
            "pending_fit_slots": 0,
            "inflight_fit_slots": 8,
            "memory_pressure_state": "ok",
            "preferred_node_relaxed": False,
            "standalone_aedt_available": 1,
            "allocations": [
                {
                    "allocation_id": rounded.SOURCE_ALLOCATION_ID,
                    "account_name": rounded.ACCOUNT_NAME,
                    "node_name": rounded.NODE_NAME,
                    "fit_slots": 8,
                    "free_cpus": 64,
                    "free_memory_mb": 866_173,
                    "memory_pressure_state": "ok",
                }
            ],
        }
        self.allocations = [
            {
                "id": rounded.SOURCE_ALLOCATION_ID,
                "account_name": rounded.ACCOUNT_NAME,
                "node_name": rounded.NODE_NAME,
                "slurm_job_id": rounded.SOURCE_SLURM_JOB_ID,
                "state": "active",
                "node_pestat_state": "mix",
                "node_cpu_total": 256,
                "node_cpu_used": 136,
                "node_memory_free_mb": 654_971,
            }
        ]
        self.licenses = {
            "server_up": True,
            "admission": {
                "enabled": True,
                "snapshot_valid": True,
                "snapshot_age_seconds": 1,
                "snapshot_max_age_seconds": 60,
                "blocked_reason": None,
                "features": {
                    submitter.direct.LICENSE_FEATURE: {
                        "admit_headroom": 1
                    }
                },
                "reserve_by_feature": {
                    submitter.direct.LICENSE_FEATURE: (
                        submitter.direct.LICENSE_RESERVE_FLOOR
                    )
                },
            },
        }
        self.accounts = [
            {
                "account_name": rounded.ACCOUNT_NAME,
                "running": 10,
                "pending": 1,
                "max_running": 10,
                "max_pending": 10,
                "max_total": 20,
                "storage_path": "slurm_scheduler",
                "storage_used_gb": None,
                "storage_quota_gb": None,
            }
        ]
        self.capabilities = [
            {
                "capability": "conda:pyaedt2026v1",
                "accounts": [rounded.ACCOUNT_NAME],
            }
        ]
        self.active: list[dict[str, Any]] = []
        self.inventory: list[dict[str, Any]] = []

    def reader(self, path: str, query: Any) -> Any:
        if path == "/api/health":
            return copy.deepcopy(self.health)
        if path == f"/api/tasks/{rounded.SOURCE_LINEAGE_TASK_ID}":
            return copy.deepcopy(self.anchor)
        if path == "/api/task-capacity":
            assert dict(query or []) == dict(submitter.capacity_query())
            return copy.deepcopy(self.capacity)
        if path == "/api/allocations":
            return copy.deepcopy(self.allocations)
        if path == "/api/licenses":
            return copy.deepcopy(self.licenses)
        if path == "/api/accounts/status/live":
            return copy.deepcopy(self.accounts)
        if path == "/api/capabilities":
            return copy.deepcopy(self.capabilities)
        if path == "/api/tasks":
            if "name_prefix" in dict(query or []):
                return copy.deepcopy(self.inventory)
            return copy.deepcopy(self.active)
        raise AssertionError(f"unexpected GET {path} {query}")


def _payload() -> dict[str, Any]:
    return {
        "name": rounded.TASK_NAME,
        "project": rounded.PROJECT,
        "account_name": rounded.ACCOUNT_NAME,
        "node_name": rounded.NODE_NAME,
        "node_name_policy": "strict",
        "cpus": rounded.CPUS,
        "memory_mb": rounded.MEMORY_MB,
        "max_workers_per_node": rounded.MAX_WORKERS_PER_NODE,
        "timeout_seconds": rounded.SCHEDULER_SECONDS,
        "aedt_backend": "standalone",
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "scheduling_profile": "fea_bursty",
        "dedupe_key": "rounded-test-dedupe",
        "command": (
            f"git checkout {REVISION}; "
            "python run_simulation_260706.py --fixed --thermal "
            "--headless --symmetry-thermal-direct-analyze "
            "--params cand.json;"
        ),
    }


def _readback(payload: dict[str, Any], status: str = "queued") -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "name": rounded.TASK_NAME,
        "dedupe_key": payload["dedupe_key"],
        "project": rounded.PROJECT,
        "requested_account_name": rounded.ACCOUNT_NAME,
        "requested_node_name": rounded.NODE_NAME,
        "requested_node_name_policy": "strict",
        "preferred_node_relaxed": False,
        "same_node_as_task_id": rounded.SAME_NODE_AS_TASK_ID,
        "requested_allocation_id": 0,
        "cpus": rounded.CPUS,
        "memory_mb": rounded.MEMORY_MB,
        "timeout_seconds": rounded.SCHEDULER_SECONDS,
        "max_workers_per_node": rounded.MAX_WORKERS_PER_NODE,
        "aedt_backend": "standalone",
        "status": status,
        "account_name": "",
        "node_name": "",
    }


def test_live_gate_accepts_terminal_lineage_and_saturated_running_cap() -> None:
    scheduler = GateReader()
    gate = submitter.live_gate(
        expected_dedupe_key="new-rounded-dedupe",
        reader=scheduler.reader,
        observed_at=OBSERVED_AT,
    )
    submitter.validate_seal(gate, submitter.LIVE_GATE_SCHEMA)
    assert gate["source_allocation_id"] == 14_620
    assert gate["same_node_as_task_id"] == 0
    assert gate["source_lineage_task_id"] == 96_328
    assert gate["active_target_node_fea_task_ids"] == []
    assert gate["account_gate"]["running_cap_saturated"] is True
    assert gate["account_gate"]["new_slurm_job_requested"] is False
    assert gate["allocation_remaining_seconds"] > (
        rounded.SCHEDULER_SECONDS
        + submitter.MIN_FORCE_END_BUFFER_SECONDS
    )


@pytest.mark.parametrize("drift", ("allocation", "runtime", "active"))
def test_live_gate_fails_closed_on_exact_lane_drift(drift: str) -> None:
    scheduler = GateReader()
    observed = OBSERVED_AT
    if drift == "allocation":
        scheduler.capacity["allocations"][0]["allocation_id"] = 1
    elif drift == "runtime":
        observed = submitter.ALLOCATION_FORCE_END_UTC
    else:
        scheduler.active.append(
            {
                "task_id": 1,
                "status": "running",
                "node_name": rounded.NODE_NAME,
                "aedt_backend": "standalone",
            }
        )
    with pytest.raises(rounded.PostdeadlineContractError):
        submitter.live_gate(
            expected_dedupe_key="new-rounded-dedupe",
            reader=scheduler.reader,
            observed_at=observed,
        )


def test_one_shot_submit_posts_once_and_second_call_only_gets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "rounded"
    root.mkdir()
    monkeypatch.setattr(submitter, "OUTPUT_ROOT", root)
    payload = _payload()
    plan = submitter.sealed(
        {
            "schema_version": "test-rounded-plan-v1",
            "solver_revision": REVISION,
            "scheduler_payload": payload,
            "scheduler_payload_sha256": submitter.payload_sha256(payload),
        }
    )
    plan_path = submitter.write_immutable_json(
        root / rounded.PLAN_NAME, plan
    )
    submitted = False
    post_calls = 0

    def gate(**_kwargs: Any) -> dict[str, Any]:
        return submitter.sealed(
            {
                "schema_version": submitter.LIVE_GATE_SCHEMA,
                "scheduler_get_only": True,
                "scheduler_post_calls": 0,
            }
        )

    def poster(
        _url: str, posted: dict[str, Any]
    ) -> tuple[int, dict[str, Any], None]:
        nonlocal submitted, post_calls
        assert posted == payload
        submitted = True
        post_calls += 1
        return 201, {"task_id": TASK_ID}, None

    def reader(path: str, query: Any) -> Any:
        if path == "/api/tasks":
            assert submitted
            return [
                {
                    "task_id": TASK_ID,
                    "name": rounded.TASK_NAME,
                    "dedupe_key": payload["dedupe_key"],
                }
            ]
        if path == f"/api/tasks/{TASK_ID}":
            return _readback(payload)
        raise AssertionError(f"unexpected GET {path} {query}")

    monkeypatch.setattr(submitter, "live_gate", gate)
    kwargs = {
        "authorize_post": submitter.POST_AUTHORIZATION,
        "plan_path": plan_path,
        "reader": reader,
        "poster": poster,
        "lock_factory": Lock,
        "plan_loader": lambda _path: plan,
    }
    final_path = submitter.submit(**kwargs)
    final = submitter.validate_seal(
        submitter.read_json(final_path), submitter.FINAL_SCHEMA
    )
    assert final["task_id"] == TASK_ID
    assert post_calls == 1
    assert submitter.submit(**kwargs) == final_path
    assert post_calls == 1


def test_collector_active_poll_is_get_only() -> None:
    payload = _payload()
    task = _readback(payload, status="running")
    task.update(
        {
            "account_name": rounded.ACCOUNT_NAME,
            "node_name": rounded.NODE_NAME,
            "actual_node_name": rounded.NODE_NAME,
            "allocation_node_name": rounded.NODE_NAME,
            "allocation_id": rounded.SOURCE_ALLOCATION_ID,
            "assigned_allocation": rounded.SOURCE_ALLOCATION_ID,
            "slurm_job_id": rounded.SOURCE_SLURM_JOB_ID,
        }
    )
    contract = {
        "task_id": TASK_ID,
        "dedupe_key": payload["dedupe_key"],
    }
    result = collector.collect(
        contract=contract,
        output=Path("unused"),
        reader=lambda _path, _query: copy.deepcopy(task),
    )
    assert result == {
        "event": "rounded_task_active",
        "task_id": TASK_ID,
        "status": "running",
        "scheduler_get_only": True,
        "scheduler_post_calls": 0,
    }
