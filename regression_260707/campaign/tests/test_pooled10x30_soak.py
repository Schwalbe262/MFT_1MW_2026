from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest


CAMPAIGN = Path(__file__).resolve().parents[1]
if str(CAMPAIGN) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN))

import pooled10x30_soak as soak


def valid_pool(*, unhealthy=32):
    return {
        "config": {
            "enabled": True,
            "adapter_ready": True,
            "validation_passed": True,
            "operational": True,
            "max_aedt_sessions": 10,
            "target_project_concurrency": 30,
            "projects_per_aedt": 3,
            "project_cpus": 4,
            "session_reserved_cpus": 13,
            "native_solve_mode": "validated_parallel",
            "parallel_safe_native_solve_families": ["mft_validated_async"],
        },
        "latest_validation": {"status": "passed"},
        "plan": {
            "hard_session_count": 0,
            "active_session_count": 0,
            "starting_session_count": 0,
            "live_projects": 0,
            "queued_pooled_task_backlog": 0,
            "unhealthy_session_count": unhealthy,
        },
    }


def task(task_id, status="queued"):
    return {
        "id": task_id,
        "name": f"{soak.NAME_PREFIX}{task_id:07d}",
        "status": status,
        "project": soak.PROJECT,
        "aedt_backend": "pooled",
    }


def args(tmp_path):
    return SimpleNamespace(
        scheduler_url="http://scheduler.test",
        pool_url="http://pool.test",
        runtime_dir=tmp_path,
    )


def test_pool_gate_accepts_archival_unhealthy_rows():
    result = soak.verify_pool(valid_pool(unhealthy=32))
    assert result["archival_unhealthy_count"] == 32
    assert result["hard_session_count"] == 0


def test_pool_gate_rejects_target_drift():
    payload = valid_pool()
    payload["config"]["target_project_concurrency"] = 500
    with pytest.raises(soak.GateError, match="contract drifted"):
        soak.verify_pool(payload)


def test_active_inventory_is_backend_and_prefix_scoped():
    with patch.object(soak, "_http_json", return_value=[task(7, "running")]):
        rows, counts = soak.active_inventory("http://scheduler.test")
    assert [row["id"] for row in rows] == [7]
    assert counts == {"queued": 0, "attaching": 0, "running": 1}

    bad = task(8)
    bad["aedt_backend"] = "standalone"
    with patch.object(soak, "_http_json", return_value=[bad]):
        with pytest.raises(soak.GateError, match="identity drift"):
            soak.active_inventory("http://scheduler.test")


def test_environment_preserves_validated_async_contract():
    env = soak.submission_environment("http://pool.test")
    assert env["MFT_CAMPAIGN_ID"] == soak.CAMPAIGN_ID
    assert env["MFT_AEDT_WORKLOAD_FAMILY"] == "mft_validated_async"
    assert env["MFT_AEDT_ASYNC_DISPATCH_SETTLE_SECONDS"] == "2"
    assert env["MFT_AEDT_ISOLATION_POLICY"] == "family"
    assert env["MFT_CAMPAIGN_MIGRATION_REPLACEMENT_SOLVER"] == (
        soak.SOLVER_REVISION
    )


def test_cycle_submits_only_own_deficit_and_respects_shared_project_slots(tmp_path):
    before = [task(i) for i in range(1, 29)]
    after = before + [task(101), task(102)]
    inventories = iter([
        (before, {"queued": 28, "attaching": 0, "running": 0}),
        (after, {"queued": 30, "attaching": 0, "running": 0}),
    ])
    candidates = iter([
        (101, 1001, {"x": 1}),
        (102, 1002, {"x": 2}),
    ])
    ids = iter([101, 102])
    capacity = {
        "project_submission_slots": 70,
        "project_active": 430,
    }
    with patch.object(
        soak.scheduler_client, "campaign_mutation_lock", return_value=nullcontext()
    ), patch.object(
        soak, "pool_snapshot", return_value={"hard_session_count": 0}
    ), patch.object(
        soak, "active_inventory", side_effect=lambda *_: next(inventories)
    ), patch.object(
        soak.scheduler_client,
        "live_project_submission_snapshot",
        return_value=capacity,
    ), patch.object(
        soak, "load_state", return_value={
            **soak._initial_state(), "candidate_cursor": 100,
        }
    ), patch.object(
        soak.feeder, "next_valid_candidate", side_effect=lambda *_args, **_kw: next(candidates)
    ), patch.object(
        soak.feeder, "submit", side_effect=lambda *_args, **_kw: next(ids)
    ) as submit, patch.object(soak, "write_json"):
        status = soak.execute_cycle(args(tmp_path), {"gate": "passed"})

    assert status["active_before"] == 28
    assert status["active_after"] == 30
    assert status["submitted_task_ids"] == [101, 102]
    assert submit.call_count == 2
    for call in submit.call_args_list:
        assert call.kwargs["aedt_backend"] == "pooled"
        assert call.kwargs["required_hard_cap"] == 500
        assert call.kwargs["prevalidated_cycle"] is True
        assert call.kwargs["account_name"] == ""


def test_cycle_waits_when_shared_project_has_no_slots(tmp_path):
    inventory = [task(i) for i in range(1, 21)]
    counts = {"queued": 20, "attaching": 0, "running": 0}
    with patch.object(
        soak.scheduler_client, "campaign_mutation_lock", return_value=nullcontext()
    ), patch.object(
        soak, "pool_snapshot", return_value={"hard_session_count": 0}
    ), patch.object(
        soak, "active_inventory", side_effect=[(inventory, counts), (inventory, counts)]
    ), patch.object(
        soak.scheduler_client,
        "live_project_submission_snapshot",
        return_value={"project_submission_slots": 0, "project_active": 500},
    ), patch.object(soak, "load_state") as load, patch.object(
        soak.feeder, "submit"
    ) as submit, patch.object(soak, "write_json"):
        status = soak.execute_cycle(args(tmp_path), {})

    assert status["phase"] == "waiting-project-capacity"
    load.assert_called_once()
    submit.assert_not_called()


def test_controller_has_no_static_account_selector():
    assert not hasattr(soak, "_account_for_serial")
