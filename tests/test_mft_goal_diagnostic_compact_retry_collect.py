from __future__ import annotations

from tools import mft_goal_diagnostic_compact_collect as collect
from tools import mft_goal_diagnostic_compact_retry_collect as overlay


class _Scheduler:
    def __init__(self, detail):
        self.detail = detail
        self.get_count = 0

    def get_task(self, task_id):
        self.get_count += 1
        assert task_id == 97245
        return dict(self.detail)


def _context():
    return {
        "entries": [
            {
                "seed": 2607264153,
                "task_id": 97245,
                "dedupe_key": "retry-dedupe",
                "task": {"payload_sha256": "a" * 64},
                "scheduler_payload": {
                    "name": "retry-task",
                    "dedupe_key": "retry-dedupe",
                    "account_name": "dhj02",
                },
            }
        ]
    }


def test_retry_observation_authenticates_requested_account(monkeypatch):
    observed_expected = {}

    def authenticate(row, expected, *, label):
        observed_expected.update(expected)
        assert label == "diagnostic collector GET"
        return 97245, "running"

    monkeypatch.setattr(collect.offload, "_task_authentication", authenticate)
    detail = {
        "status": "running",
        "requested_account_name": "dhj02",
        "account_name": "dhj02",
        "exit_code": None,
    }
    rows = collect.observe_tasks(_context(), _Scheduler(detail))
    assert rows[0]["observation_error"] is None
    assert rows[0]["scheduler_status"] == "running"
    assert "account_name" not in observed_expected


def test_retry_observation_fails_closed_on_account_drift(monkeypatch):
    monkeypatch.setattr(
        collect.offload,
        "_task_authentication",
        lambda *_args, **_kwargs: (97245, "running"),
    )
    detail = {
        "status": "running",
        "requested_account_name": "harry261",
        "account_name": "harry261",
        "exit_code": None,
    }
    rows = collect.observe_tasks(_context(), _Scheduler(detail))
    assert rows[0]["scheduler_status"] == "query_error"
    assert "account placement identity changed" in rows[0]["observation_error"]


def test_overlay_module_has_no_scheduler_mutation_client():
    assert not hasattr(overlay, "SchedulerClient")
    assert "POST /api/tasks" not in (overlay.__doc__ or "")
