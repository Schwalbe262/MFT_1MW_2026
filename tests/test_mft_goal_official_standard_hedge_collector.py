from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools import mft_goal_official_standard_hedge_collector as collector


def contract() -> dict[str, Any]:
    return {
        "scheduler_url": "http://scheduler.test",
        "task_id": 96330,
        "task_name": "hedge-task",
        "dedupe_key": "hedge-dedupe",
        "candidate_physics_sha256": "a" * 64,
        "node_name": "node10",
        "strict_account_name": "mft",
        "strict_node_name": "node10",
        "submission_file_sha256": "b" * 64,
    }


def task(*, terminal: bool = False) -> dict[str, Any]:
    return {
        "task_id": 96330,
        "name": "hedge-task",
        "dedupe_key": "hedge-dedupe",
        "project": collector.submitter.PROJECT,
        "requested_account_name": "mft",
        "requested_node_name": "node10",
        "requested_node_name_policy": "strict",
        "same_node_as_task_id": 0,
        "cpus": collector.submitter.CPUS,
        "memory_mb": collector.submitter.MEMORY_MB,
        "timeout_seconds": collector.submitter.SCHEDULER_SECONDS,
        "max_workers_per_node": collector.submitter.MAX_WORKERS_PER_NODE,
        "aedt_backend": "standalone",
        "scheduling_profile": "fea_bursty",
        "status": "completed" if terminal else "running",
        "state": "succeeded" if terminal else "running",
        "exit_code": 0 if terminal else None,
        "preferred_node_relaxed": False,
        "node_name_policy": "strict",
        "placement_contract_satisfied": True,
        "account_name": "mft",
        "node_name": "node10",
        "actual_node_name": "node10",
        "allocation_node_name": "node10",
        "allocation_id": 14648,
        "slurm_job_id": "840585",
    }


def getter_for(value: dict[str, Any], calls: list[str]):
    def getter(url: str, **_kwargs: Any) -> bytes:
        calls.append(url)
        return json.dumps(value).encode("utf-8")

    return getter


@pytest.mark.skipif(
    not collector.AGGREGATE_SUBMIT_MANIFEST.is_file(),
    reason="sealed hedge submission package is not present",
)
def test_sealed_package_authenticates_and_builds_adapters(
    tmp_path: Path,
) -> None:
    package = collector.authenticate_submission_package()
    contracts = collector.build_collection_contracts(
        package,
        output_root=tmp_path,
    )

    assert [item["task_id"] for item in contracts] == [
        96330,
        96331,
        96332,
    ]
    assert [item["parameter_digest"] for item in contracts] == [
        "81a33633f21c74a3",
        "2f125ba0bfae7f8c",
        "d71e655337a09462",
    ]


def test_active_poll_is_get_only(tmp_path: Path) -> None:
    calls: list[str] = []
    finished, event = collector.poll_lane_once(
        contract=contract(),
        output=tmp_path / "lane",
        sequence=3,
        getter=getter_for(task(), calls),
    )

    assert finished is False
    assert event["event"] == "active"
    assert event["slurm_job_id"] == "840585"
    assert calls == ["http://scheduler.test/api/tasks/96330"]


def test_terminal_success_delegates_standard_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    delegated: list[tuple[int, Path]] = []

    def collect_success(
        *,
        contract: dict[str, Any],
        task: dict[str, Any],
        output: Path,
        getter: Any,
    ) -> dict[str, Any]:
        del getter
        delegated.append((task["task_id"], output))
        assert contract["task_id"] == task["task_id"]
        return {"event": "collected", "task_id": task["task_id"]}

    monkeypatch.setattr(collector.standard, "collect_success", collect_success)
    output = tmp_path / "lane"
    finished, event = collector.poll_lane_once(
        contract=contract(),
        output=output,
        sequence=4,
        getter=getter_for(task(terminal=True), calls),
    )

    assert finished is True
    assert event["event"] == "collected"
    assert delegated == [(96330, output)]
    assert calls == ["http://scheduler.test/api/tasks/96330"]
