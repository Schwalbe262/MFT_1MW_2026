from __future__ import annotations

import copy

import pytest

from tools import mft_goal_diagnostic_compact_retry as retry


def _base_payload() -> dict:
    return {
        "name": "mft-goal-diag-compact-0123456789abcdef01234567-s1-n1-6",
        "remote_cwd": "/gpfs/compact",
        "command": "set -euo pipefail\nexec python worker.py",
        "payload_json": {"seed": 2607264153},
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "cpus": 8,
        "memory_mb": 65536,
        "scheduling_profile": "standard",
        "aedt_backend": "standalone",
        "gpus": 0,
        "priority": 10,
        "timeout_seconds": 14400,
        "max_workers_per_node": 8,
        "dedupe_key": retry.offload.DEDUPE_PREFIX + "a" * 64,
    }


def test_retry_payload_is_pinned_warning_capped_and_distinct(monkeypatch):
    monkeypatch.setattr(
        retry.offload,
        "scheduler_payload",
        lambda **_kwargs: copy.deepcopy(_base_payload()),
    )
    monkeypatch.setattr(
        retry.offload,
        "_task_name_prefix",
        lambda _plan: (
            "mft-goal-diag-compact-0123456789abcdef01234567-"
        ),
    )
    task = {"seed": 2607264153}
    payload = retry._retry_payload(
        base_plan={},
        task=task,
        original_task_id=97198,
        requested_account="dhj02",
    )
    assert payload["account_name"] == "dhj02"
    assert payload["name"].startswith(
        "mft-goal-diag-compact-0123456789abcdef01234567-retry1-"
    )
    assert "exec python -W ignore worker.py" in payload["command"]
    assert payload["dedupe_key"].startswith(retry.offload.DEDUPE_PREFIX)
    assert payload["dedupe_key"] != _base_payload()["dedupe_key"]


def test_retry_payload_rejects_quota_full_account(monkeypatch):
    monkeypatch.setattr(
        retry.offload,
        "scheduler_payload",
        lambda **_kwargs: copy.deepcopy(_base_payload()),
    )
    with pytest.raises(RuntimeError, match="not authorized"):
        retry._retry_payload(
            base_plan={},
            task={"seed": 2607264153},
            original_task_id=97198,
            requested_account="harry261",
        )


def test_only_exact_prestart_failure_is_authorized():
    detail = {
        "id": 97198,
        "task_id": 97198,
        "status": "failed",
        "account_name": "harry261",
        "started_at": None,
        "remote_dir": "",
        "stdout_path": "",
        "stderr_path": "",
        "exit_code_path": "",
        "exit_code": None,
        "failure_message": "Failure",
        "allocation_id": 14724,
        "slurm_job_id": "845825",
        "attached_at": "2026-07-27 07:51:43",
        "launch_started_at": "2026-07-27 07:51:43",
        "finished_at": "2026-07-27 07:51:45",
    }
    evidence = retry._prestart_quota_failure(
        detail, expected_task_id=97198
    )
    assert evidence["remote_paths_empty"] is True
    changed = dict(detail, started_at="2026-07-27 07:51:44")
    with pytest.raises(RuntimeError, match="not the authorized"):
        retry._prestart_quota_failure(changed, expected_task_id=97198)


def test_scheduler_client_exposes_no_cancel_or_preempt():
    client = retry.SchedulerClient("http://127.0.0.1:1")
    assert not hasattr(client, "cancel_task")
    assert not hasattr(client, "preempt_task")
