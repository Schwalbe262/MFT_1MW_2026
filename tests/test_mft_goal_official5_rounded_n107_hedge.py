from __future__ import annotations

from typing import Any

import pytest

from tools import mft_goal_official5_rounded_n107_hedge as hedge


def test_lane_is_strict_fresh_n107_single_worker() -> None:
    query = dict(hedge.capacity_query())
    assert hedge.ACCOUNT_NAME == "harry261"
    assert hedge.NODE_NAME == "n107"
    assert hedge.MAX_WORKERS_PER_NODE == 1
    assert query["account_name"] == "harry261"
    assert query["node_name"] == "n107"
    assert query["node_name_policy"] == "strict"
    assert query["max_workers_per_node"] == 1
    assert hedge.SCHEDULER_SECONDS == 12_900


def test_reviewed_payload_patch_is_scoped_and_restored() -> None:
    original = {
        name: getattr(hedge.rounded, name)
        for name in (
            "ACCOUNT_NAME",
            "NODE_NAME",
            "MAX_WORKERS_PER_NODE",
            "SAME_NODE_AS_TASK_ID",
            "SOURCE_ALLOCATION_ID",
        )
    }
    with hedge._rounded_payload_patch():
        assert hedge.rounded.ACCOUNT_NAME == "harry261"
        assert hedge.rounded.NODE_NAME == "n107"
        assert hedge.rounded.MAX_WORKERS_PER_NODE == 1
        assert hedge.rounded.SAME_NODE_AS_TASK_ID == 0
        assert hedge.rounded.SOURCE_ALLOCATION_ID == 0
    assert {
        name: getattr(hedge.rounded, name) for name in original
    } == original


def test_task_script_attestation_matches_scheduler_plain_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = (
        "git checkout "
        f"{hedge.SOLVER_REVISION}; "
        f"printf '%s' '{{\"candidate\":\"{hedge.CANDIDATE_SHA256[:12]}\","
        "\"round_corner\":1,\"corner_radius\":10.0,"
        "\"corner_segments\":4,\"full_model\":0,"
        "\"thermal_symmetry\":\"eighth\",\"fan_velocity\":1.5,"
        "\"k_ins\":0.2,\"core_plate_pad_t\":2.0,\"wcp_pad_t\":2.0}' "
        "> cand.json; python run_simulation_260706.py "
        "--symmetry-thermal-direct-analyze --params cand.json"
    )
    monkeypatch.setattr(
        hedge,
        "_get_text",
        lambda *_args, **_kwargs: f"#!/bin/bash\n{command}\n",
    )
    result = hedge._task_script_attestation(
        96_342,
        {"status": "running"},
        {"scheduler_payload": {"command": command}},
    )
    assert result["available"] is True
    assert result["exact_scheduler_command_present"] is True
    assert result["all_physics_markers_present"] is True
    assert all(result["physics_marker_presence"].values())


def test_wrong_authorization_fails_before_any_post() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def poster(
        path: str, payload: dict[str, Any]
    ) -> tuple[int | None, dict[str, Any] | None, str | None]:
        calls.append((path, payload))
        raise AssertionError("POST must not be reached")

    with pytest.raises(hedge.PostdeadlineContractError):
        hedge.submit(authorize_post="wrong", poster=poster)
    assert calls == []


def test_watch_interval_guard_is_fail_closed() -> None:
    with pytest.raises(hedge.PostdeadlineContractError):
        hedge.collect_watch(interval_seconds=9)
