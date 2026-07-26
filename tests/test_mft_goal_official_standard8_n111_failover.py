from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any

import pytest

from tools import mft_goal_official_standard8_n111_failover as failover


OBSERVED = datetime(2026, 7, 26, 12, 40, tzinfo=timezone.utc)


def _capacity() -> dict[str, Any]:
    return {
        "fit_slots": 0,
        "ready_fit_slots": 0,
        "pending_fit_slots": 0,
        "inflight_fit_slots": 0,
        "memory_pressure_state": "ok",
        "preferred_node_relaxed": False,
        "allocations": [],
        "queue_state": "opening",
        "queue_reason": failover.OPENING_QUEUE_REASON,
        "standalone_aedt_available": 393,
    }


class Scheduler:
    def __init__(self) -> None:
        self.original_status = "queued"
        self.freed_allocation_state = "closed"
        self.n114_state = "drain"
        self.calls: list[tuple[str, tuple[tuple[str, Any], ...]]] = []

    def reader(
        self, path: str, query: list[tuple[str, Any]] | None
    ) -> Any:
        normalized = tuple(query or ())
        self.calls.append((path, normalized))
        if path == "/api/health":
            return {
                "ok": True,
                "scheduler_ok": True,
                "scheduler_thread_alive": True,
                "scheduler_stalled": False,
            }
        if path == f"/api/tasks/{failover.SOURCE_TASK_ID}":
            return {
                "task_id": failover.SOURCE_TASK_ID,
                "name": failover.SOURCE_TASK_NAME,
                "status": "completed",
                "exit_code": 0,
                "required_capability": "conda:pyaedt2026v1",
                "env_profile": "pyaedt2026v1",
                "dedupe_key": failover.SOURCE_TASK_DEDUPE_KEY,
            }
        if path == f"/api/tasks/{failover.ORIGINAL_OFFICIAL8_TASK_ID}":
            return {
                "task_id": failover.ORIGINAL_OFFICIAL8_TASK_ID,
                "name": failover.ORIGINAL_OFFICIAL8_TASK_NAME,
                "status": self.original_status,
                "state": self.original_status,
                "exit_code": None,
                "dedupe_key": failover.ORIGINAL_OFFICIAL8_DEDUPE_KEY,
                "project": failover.PROJECT,
                "requested_account_name": (
                    failover.ORIGINAL_OFFICIAL8_ACCOUNT
                ),
                "requested_node_name": failover.ORIGINAL_OFFICIAL8_NODE,
                "requested_node_name_policy": "strict",
                "preferred_node_relaxed": False,
                "allocation_id": None,
                "assigned_allocation": None,
                "required_capability": "conda:pyaedt2026v1",
                "env_profile": "pyaedt2026v1",
                "aedt_backend": "standalone",
            }
        if path == f"/api/tasks/{failover.FREED_TASK_ID}":
            return {
                "task_id": failover.FREED_TASK_ID,
                "status": "failed",
                "state": "failed",
                "exit_code": 1,
                "allocation_id": failover.FREED_ALLOCATION_ID,
                "assigned_allocation": failover.FREED_ALLOCATION_ID,
                "slurm_job_id": failover.FREED_SLURM_JOB_ID,
                "requested_account_name": failover.ACCOUNT_NAME,
                "requested_node_name": failover.NODE_NAME,
                "requested_node_name_policy": "strict",
                "actual_node_name": failover.NODE_NAME,
                "allocation_node_name": failover.NODE_NAME,
            }
        if path == "/api/task-capacity":
            return _capacity()
        if path == "/api/tasks":
            return []
        if path == "/api/allocations":
            return [
                {
                    "id": failover.FREED_ALLOCATION_ID,
                    "account_name": failover.ACCOUNT_NAME,
                    "node_name": failover.NODE_NAME,
                    "slurm_job_id": failover.FREED_SLURM_JOB_ID,
                    "state": self.freed_allocation_state,
                    "closed_at": "2026-07-26 12:31:33",
                    "node_pestat_state": "mix",
                    "node_metrics_observed_at": "2026-07-26 12:39:00",
                },
                {
                    "id": 14_634,
                    "account_name": "harry261",
                    "node_name": failover.ORIGINAL_OFFICIAL8_NODE,
                    "slurm_job_id": "832694",
                    "state": "closed",
                    "closed_at": "2026-07-25 12:33:22",
                    "node_pestat_state": self.n114_state,
                    "node_metrics_observed_at": "2026-07-26 12:39:00",
                },
            ]
        raise AssertionError(f"unexpected GET {path} {normalized}")


def test_live_preflight_proves_failover_and_preserves_original() -> None:
    scheduler = Scheduler()
    result = failover.live_preflight(
        reader=scheduler.reader,
        expected_dedupe_key="fresh-dedupe",
        observed_at=OBSERVED,
    )
    assert result["original_task_96329_preserved_queued"] is True
    assert result["allocation_14644_closed"] is True
    assert result["n114_drain_observed"] is True
    assert result["active_n111_fea_count"] == 0
    assert result["scheduler_get_only"] is True
    capacity_calls = [
        query
        for path, query in scheduler.calls
        if path == "/api/task-capacity"
    ]
    assert len(capacity_calls) == 2
    assert ("node_name", "n111") in capacity_calls[0]
    assert ("node_name", "n114") in capacity_calls[1]


@pytest.mark.parametrize(
    ("attribute", "value", "match"),
    [
        (
            "original_status",
            "running",
            "no longer exact queued n114",
        ),
        (
            "freed_allocation_state",
            "active",
            "exact closed n111 allocation",
        ),
        (
            "n114_state",
            "mix",
            "does not prove current n114 drain",
        ),
    ],
)
def test_live_preflight_fails_closed_on_identity_drift(
    attribute: str, value: str, match: str
) -> None:
    scheduler = Scheduler()
    setattr(scheduler, attribute, value)
    with pytest.raises(failover.PostdeadlineContractError, match=match):
        failover.live_preflight(
            reader=scheduler.reader,
            expected_dedupe_key="fresh-dedupe",
            observed_at=OBSERVED,
        )


def test_failover_patch_is_exact_and_restored() -> None:
    before = {
        name: copy.deepcopy(getattr(failover.official8, name))
        for name in failover._PATCH
        if name != "live_preflight"
    }
    with failover._failover_contract():
        assert failover.official8.ACCOUNT_NAME == "r1jae262"
        assert failover.official8.NODE_NAME == "n111"
        assert failover.official8.OUTPUT_ROOT == failover.OUTPUT_ROOT
        assert failover.official8.POST_AUTHORIZATION == (
            failover.POST_AUTHORIZATION
        )
        assert failover.official8.live_preflight is failover.live_preflight
        assert ("node_name", "n111") in failover.capacity_query()
    for name, value in before.items():
        assert getattr(failover.official8, name) == value


def test_authorization_rejected_before_post() -> None:
    posts = 0

    def poster(_url: str, _payload: Any) -> tuple[int, dict, None]:
        nonlocal posts
        posts += 1
        return 201, {"task_id": 1}, None

    with pytest.raises(
        failover.PostdeadlineContractError,
        match="authorization is absent",
    ):
        failover.submit(authorize_post="wrong", poster=poster)
    assert posts == 0
