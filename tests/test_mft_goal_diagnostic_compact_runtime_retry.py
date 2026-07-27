from __future__ import annotations

import copy

import pytest

from tools import mft_goal_diagnostic_compact_runtime_retry as retry2


def _base_payload() -> dict:
    return {
        "name": "mft-goal-diag-compact-0123456789abcdef01234567-s1-n1-6",
        "remote_cwd": "/gpfs/compact",
        "command": "set -euo pipefail\nexec python worker.py",
        "payload_json": {"seed": 2607264113},
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
        "dedupe_key": retry2.offload.DEDUPE_PREFIX + "a" * 64,
    }


def _runtime_failure() -> dict:
    return {
        "id": 97158,
        "task_id": 97158,
        "status": "failed",
        "account_name": "harry261",
        "allocation_id": 123,
        "slurm_job_id": "845800",
        "started_at": "2026-07-27 08:00:00",
        "finished_at": "2026-07-27 08:02:00",
        "exit_code": 1,
        "remote_dir": "/gpfs/run/97158",
        "stdout_path": "/gpfs/run/97158/stdout.log",
        "stderr_path": "/gpfs/run/97158/stderr.log",
        "exit_code_path": "/gpfs/run/97158/exit_code",
        "failure_message": "Exited with exit code 1",
    }


def test_only_started_runtime_exit1_failure_is_authorized():
    proof = retry2._runtime_exit1_failure(
        _runtime_failure(), expected_task_id=97158
    )
    assert proof["runtime_paths_nonempty"] is True
    assert proof["exit_code"] == 1
    with pytest.raises(RuntimeError, match="not an authorized"):
        retry2._runtime_exit1_failure(
            dict(_runtime_failure(), started_at=None),
            expected_task_id=97158,
        )
    with pytest.raises(RuntimeError, match="not an authorized"):
        retry2._runtime_exit1_failure(
            dict(_runtime_failure(), exit_code=2),
            expected_task_id=97158,
        )


def test_retry2_payload_is_pinned_warning_capped_and_distinct(monkeypatch):
    monkeypatch.setattr(
        retry2.offload,
        "scheduler_payload",
        lambda **_kwargs: copy.deepcopy(_base_payload()),
    )
    monkeypatch.setattr(
        retry2.offload,
        "_task_name_prefix",
        lambda _plan: (
            "mft-goal-diag-compact-0123456789abcdef01234567-"
        ),
    )
    payload = retry2._retry2_payload(
        base_plan={},
        task={"seed": 2607264113},
        original_task_id=97158,
        failure_proof_sha256="b" * 64,
        requested_account="dhj02",
    )
    assert payload["account_name"] == "dhj02"
    assert payload["name"].startswith(
        "mft-goal-diag-compact-0123456789abcdef01234567-retry2-"
    )
    assert "exec python -W ignore worker.py" in payload["command"]
    assert payload["dedupe_key"].startswith(retry2.offload.DEDUPE_PREFIX)
    assert payload["dedupe_key"] != _base_payload()["dedupe_key"]
    with pytest.raises(RuntimeError, match="unauthorized"):
        retry2._retry2_payload(
            base_plan={},
            task={"seed": 2607264113},
            original_task_id=97158,
            failure_proof_sha256="b" * 64,
            requested_account="harry261",
        )


def test_inventory_queries_the_exact_retry2_prefix():
    prefix = (
        "mft-goal-diag-compact-0123456789abcdef01234567-retry2-"
    )
    payload = {
        "name": f"{prefix}s2607264113-n1-6-adhj02",
        "dedupe_key": retry2.offload.DEDUPE_PREFIX + "c" * 64,
    }
    plan = {
        "retry_name_prefix": prefix,
        "tasks": [
            {
                "requested_account": "dhj02",
                "scheduler_payload": payload,
            }
        ],
    }

    class _Client:
        def list_tasks(self, observed_prefix):
            assert observed_prefix == prefix
            return [
                {
                    "id": 98100,
                    "task_id": 98100,
                    "status": "queued",
                    "requested_account_name": "dhj02",
                    **payload,
                }
            ]

    result = retry2._inventory(_Client(), plan)
    assert list(result) == [payload["dedupe_key"]]


def test_runtime_retry_client_exposes_no_cancel_or_preempt():
    client = retry2.retry1.SchedulerClient("http://127.0.0.1:1")
    assert not hasattr(client, "cancel_task")
    assert not hasattr(client, "preempt_task")
