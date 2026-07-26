from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from typing import Any

import pytest

from tools import mft_goal_postdeadline_ui_updater as updater


OBSERVED = "2026-07-26T18:20:00+09:00"


def _item(item_id: str, state: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "title": f"title {item_id}",
        "detail": f"detail {item_id}",
        "state": state,
        "updated_at": "2026-07-26T18:00:00+09:00",
        "progress_pct": 1,
        "evidence": ["preserved"],
    }


def _status() -> dict[str, Any]:
    current = [_item(spec.card_id, "in_progress") for spec in updater.TASK_SPECS]
    current.extend(
        [
            {
                **_item("fea-handoff", "in_progress"),
                "title": (
                    "SLURM · ALLOCATION JOBS 4 · SUBMITTED 120 · "
                    "RUNNING 4 · QUEUED 0 · COLLECTIONS 0"
                ),
            },
            _item("parallel-workstreams", "in_progress"),
        ]
    )
    return {
        "schema_version": updater.STATUS_SCHEMA,
        "goal_id": "mft-goal-20260726",
        "goal": "goal",
        "owner": "Codex",
        "generated_at": "2026-07-26T18:00:00+09:00",
        "deadline_at": "2026-07-26T18:00:00+09:00",
        "original_deadline_missed": True,
        "summary": "stale lifecycle summary that must be refreshed",
        "current": current,
        "completed": [_item("completed-static", "completed")],
        "attention": [
            _item("pareto-truth-boundary", "attention"),
            _item("attention-static", "blocked"),
        ],
        "unknown_top_level": {"preserve": True},
    }


def _task(
    spec: updater.TaskSpec,
    *,
    state: str = "running",
    allocation_id: int | None = 100,
    slurm_job_id: str = "200",
    failure_message: str = "",
) -> dict[str, Any]:
    actual_node = spec.requested_node if allocation_id else ""
    return {
        "id": spec.task_id,
        "task_id": spec.task_id,
        "name": spec.task_name,
        "state": state,
        "status": state,
        "queue_state": state,
        "cpus": spec.cpus,
        "memory_mb": spec.memory_mb,
        "timeout_seconds": spec.timeout_seconds,
        "same_node_as_task_id": 0,
        "requested_node_name": spec.requested_node,
        "node_name": spec.requested_node,
        "node_name_policy": "strict",
        "actual_node_name": actual_node,
        "allocation_node_name": actual_node,
        "placement_contract_satisfied": state != "queued",
        "preferred_node_relaxed": False,
        "allocation_id": allocation_id,
        "slurm_job_id": slurm_job_id,
        "account_name": spec.requested_account or "",
        "requested_account_name": spec.requested_account or "",
        "created_at": "2026-07-26 09:00:00",
        "started_at": "2026-07-26 09:01:00" if allocation_id else None,
        "finished_at": None,
        "exit_code": None,
        "failure_message": failure_message,
    }


def _mixed_tasks() -> dict[int, dict[str, Any]]:
    specs = updater.TASK_SPECS
    return {
        specs[0].task_id: updater._validate_task(
            specs[0], _task(specs[0], allocation_id=101)
        ),
        specs[1].task_id: updater._validate_task(
            specs[1],
            _task(
                specs[1],
                state="queued",
                allocation_id=None,
                slurm_job_id="",
            ),
        ),
        specs[2].task_id: updater._validate_task(
            specs[2],
            _task(specs[2], state="succeeded", allocation_id=103),
        ),
        specs[3].task_id: updater._validate_task(
            specs[3],
            _task(
                specs[3],
                state="failed",
                allocation_id=104,
                failure_message="task timed out",
            ),
        ),
    }


def _reader_from(tasks: dict[int, dict[str, Any]]):
    def reader(_scheduler_url: str, task_id: int) -> dict[str, Any]:
        return copy.deepcopy(tasks[task_id])

    return reader


def test_merge_preserves_protected_truth_and_seals_lifecycle_only() -> None:
    source = _status()
    completed = copy.deepcopy(source["completed"])
    attention = copy.deepcopy(source["attention"])
    parallel = copy.deepcopy(source["current"][-1])

    merged = updater.merge_status(
        source,
        _mixed_tasks(),
        observed_at=OBSERVED,
    )
    sync = updater.validate_status_sync(merged)

    assert merged["generated_at"] == OBSERVED
    assert merged["completed"] == completed
    assert merged["attention"] == attention
    assert "18:20 KST" in merged["summary"]
    assert "running1 · queued1 · terminal2" in merged["summary"]
    assert "physical feasible0" in merged["summary"]
    assert "scientific/production PASS가 아닙니다" in merged["summary"]
    assert merged["current"][-1] == parallel
    assert merged["unknown_top_level"] == {"preserve": True}
    assert sync["allocation_jobs_active"] == 1
    assert sync["running"] == 1
    assert sync["queued"] == 1
    assert sync["submitted_total"] == 120
    assert sync["collections_preserved"] == 0
    assert sync["scheduler_methods_used"] == ["GET"]
    assert sync["scientific_pass_generated"] is False

    success = next(
        item
        for item in merged["current"]
        if item["id"] == updater.TASK_SPECS[2].card_id
    )
    failed = next(
        item
        for item in merged["current"]
        if item["id"] == updater.TASK_SPECS[3].card_id
    )
    assert "COLLECTION/PASS PENDING" in success["title"]
    assert any(
        "collection_authenticated=false" in value for value in success["evidence"]
    )
    assert "task timed out" in failed["detail"]
    assert "과학적 infeasibility" in failed["detail"]

    handoff = next(item for item in merged["current"] if item["id"] == "fea-handoff")
    assert (
        handoff["title"] == "SLURM · ALLOCATION JOBS 1 · SUBMITTED 120 · "
        "RUNNING 1 · QUEUED 1 · COLLECTIONS 0"
    )


def test_synchronize_once_is_atomic_and_identity_failure_keeps_source(
    tmp_path: Path,
) -> None:
    path = tmp_path / "codex-work-status.json"
    path.write_text(
        json.dumps(_status(), ensure_ascii=False),
        encoding="utf-8",
    )
    raw_tasks = {
        spec.task_id: _task(spec, allocation_id=200 + index)
        for index, spec in enumerate(updater.TASK_SPECS)
    }

    sync = updater.synchronize_once(
        status_file=path,
        task_reader=_reader_from(raw_tasks),
        observed_at=OBSERVED,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert updater.validate_status_sync(payload) == sync
    assert not list(tmp_path.glob("*.tmp"))

    before = path.read_bytes()
    drifted = copy.deepcopy(raw_tasks)
    drifted[96325]["name"] = "wrong-task-name"
    with pytest.raises(updater.UpdaterError, match="name drifted"):
        updater.synchronize_once(
            status_file=path,
            task_reader=_reader_from(drifted),
            observed_at=OBSERVED,
        )
    assert path.read_bytes() == before


class _Response(io.BytesIO):
    def __init__(self, payload: bytes):
        super().__init__(payload)
        self.headers: dict[str, str] = {}

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def test_scheduler_reader_uses_bounded_get(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = updater.TASK_SPECS[0]
    observed: dict[str, Any] = {}

    def urlopen(request: Any, *, timeout: float) -> _Response:
        observed["method"] = request.get_method()
        observed["url"] = request.full_url
        observed["timeout"] = timeout
        return _Response(json.dumps(_task(spec)).encode("utf-8"))

    monkeypatch.setattr(updater.urllib.request, "urlopen", urlopen)
    result = updater._get_scheduler_task(
        "http://127.0.0.1:8002",
        spec.task_id,
        timeout_seconds=3.0,
    )

    assert result["task_id"] == spec.task_id
    assert observed == {
        "method": "GET",
        "url": f"http://127.0.0.1:8002/api/tasks/{spec.task_id}",
        "timeout": 3.0,
    }


def test_run_once_writes_sealed_pid_and_log(tmp_path: Path) -> None:
    status_file = tmp_path / "codex-work-status.json"
    status_file.write_text(
        json.dumps(_status(), ensure_ascii=False),
        encoding="utf-8",
    )
    pid_file = tmp_path / "updater.pid.json"
    log_file = tmp_path / "updater.jsonl"
    lock_file = tmp_path / "updater.lock"
    tasks = {
        spec.task_id: _task(spec, allocation_id=300 + index)
        for index, spec in enumerate(updater.TASK_SPECS)
    }

    result = updater.run_updater(
        status_file=status_file,
        scheduler_url=updater.DEFAULT_SCHEDULER_URL,
        once=True,
        interval_seconds=60,
        pid_file=pid_file,
        log_file=log_file,
        lock_file=lock_file,
        task_reader=_reader_from(tasks),
    )

    assert result is not None
    pid = json.loads(pid_file.read_text(encoding="utf-8"))
    updater.validate_seal(pid, updater.PID_SCHEMA)
    assert pid["scheduler_methods_allowed"] == ["GET"]
    assert pid["scheduler_mutation_performed"] is False
    events = [
        json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == [
        "updater_started",
        "cycle_completed",
    ]
    for event in events:
        updater.validate_seal(event, updater.LOG_SCHEMA)


def test_cli_modes_and_default_interval() -> None:
    parser = updater._parser()
    assert parser.parse_args(["--once"]).interval_seconds == 60
    assert parser.parse_args(["--watch"]).watch is True
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--once", "--watch"])
