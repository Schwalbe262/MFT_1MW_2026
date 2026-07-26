from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tools import mft_goal_local_symmetric_selection_watch as watch
from tools import mft_goal_local_trust_acquisition as acquisition


TASK_IDS = watch.EXPECTED_TASK_IDS
LIFECYCLE_TASK_IDS = watch.EXPECTED_LIFECYCLE_TASK_IDS
LOCAL_SHA_BY_TASK = {
    96328: "6" * 64,
    96330: "1" * 64,
    96331: "c" * 64,
    96338: "5" * 64,
}
NONLOCAL_SHA_BY_TASK = {
    96325: "a" * 64,
    96327: "b" * 64,
    96333: "d" * 64,
}


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(watch._canonical_bytes(value) + b"\n")
    return path


def _hard_evidence(normalized_margin: float) -> dict[str, Any]:
    scales = {
        "width_mm": 1200.0,
        "length_mm": 1000.0,
        "height_mm": 750.0,
        "resonance_Hz": 15_000.0,
        "winding_max_C": 100.0,
        "core_max_C": 120.0,
    }
    return {
        name: {
            "margin": normalized_margin * scale,
            "passed": normalized_margin >= 0.0,
        }
        for name, scale in scales.items()
    }


def _observation(
    task_id: int,
    *,
    candidate_sha: str,
    passed: bool = False,
    normalized_margin: float = -0.01,
    volume: float = 800.0,
    loss: float = 5300.0,
) -> dict[str, Any]:
    value = {
        "schema_version": acquisition.OBSERVATION_SCHEMA,
        "task_id": task_id,
        "candidate_physics_sha256": candidate_sha,
        "actual_volume_L": volume,
        "actual_total_loss_W": loss,
        "actual_dimensions_mm": {
            "W": 1190.0,
            "L": 990.0,
            "H": 740.0,
        },
        "actual_resonance_Hz": 15_100.0,
        "actual_winding_max_C": 99.0,
        "actual_core_max_C": 119.0,
        "hard_constraint_evidence": _hard_evidence(normalized_margin),
        "measured_hard_constraints_passed": passed,
    }
    return acquisition.seal(value)


def test_selection_keeps_authoritative_resonance_in_hz() -> None:
    residual_hz = 78.45761081833734
    observation = _observation(
        96338,
        candidate_sha=LOCAL_SHA_BY_TASK[96338],
        passed=True,
        normalized_margin=0.1,
    )
    observation["actual_resonance_Hz"] = 15_000.0 + residual_hz
    observation["hard_constraint_evidence"]["resonance_Hz"][
        "margin"
    ] = residual_hz

    selected = watch._select_passing(  # noqa: SLF001
        [(Path("authenticated-observation.json"), observation)]
    )

    assert selected is not None
    assert selected["actual_resonance_Hz"] == pytest.approx(
        15_078.457610818337
    )
    assert selected["minimum_normalized_actual_margin"] == pytest.approx(
        residual_hz / 15_000.0
    )


def _source_state(
    tmp_path: Path,
    statuses: dict[int, str],
    observations: dict[int, dict[str, Any]],
) -> Path:
    lanes = []
    for task_id in LIFECYCLE_TASK_IDS:
        status = statuses.get(task_id, "pending")
        selection_effective = task_id in TASK_IDS
        lane: dict[str, Any] = {
            "task_id": task_id,
            "selection_effective": selection_effective,
            "status": status,
        }
        if status == "collection_ready":
            observation_path = _write_json(
                tmp_path / "observations" / f"task{task_id}.json",
                observations[task_id],
            )
            lane["observation"] = watch._file_record(observation_path)
            lane["measured_hard_constraints_passed"] = observations[
                task_id
            ]["measured_hard_constraints_passed"]
        elif status == "terminal_failure":
            failure = _write_json(
                tmp_path / "failures" / f"task{task_id}.json",
                {
                    "schema_version": "test-operational-failure-v1",
                    "task_id": task_id,
                    "scheduler_mutation_performed": False,
                },
            )
            lane["failure_ledger"] = watch._file_record(failure)
        else:
            lane["collection_root"] = str(
                tmp_path / "pending" / str(task_id)
            )
        lanes.append(lane)
    effective_lanes = [
        item for item in lanes if item["selection_effective"]
    ]
    counts = {
        status: sum(item["status"] == status for item in effective_lanes)
        for status in watch.ALLOWED_LANE_STATUSES
    }
    lifecycle_counts = {
        status: sum(item["status"] == status for item in lanes)
        for status in watch.ALLOWED_LANE_STATUSES
    }
    state = watch._seal(
        {
            "schema_version": watch.SOURCE_STATE_SCHEMA,
            "expected_lane_count": len(TASK_IDS),
            "lifecycle_lane_count": len(LIFECYCLE_TASK_IDS),
            "effective_task_ids": list(TASK_IDS),
            "lifecycle_task_ids": list(LIFECYCLE_TASK_IDS),
            "selection_superseded_task_ids": list(
                watch.SELECTION_SUPERSEDED_TASK_IDS
            ),
            "collection_count": counts["collection_ready"],
            "pending_count": counts["pending"],
            "terminal_failure_count": counts["terminal_failure"],
            "lifecycle_collection_count": lifecycle_counts[
                "collection_ready"
            ],
            "lifecycle_pending_count": lifecycle_counts["pending"],
            "lifecycle_terminal_failure_count": lifecycle_counts[
                "terminal_failure"
            ],
            "lanes": lanes,
            "scheduler_mutation_performed": False,
            "orchestrator_scheduler_methods_used": [],
        }
    )
    return _write_json(tmp_path / "source-state.json", state)


def _patch_observation_auth(
    monkeypatch: pytest.MonkeyPatch,
    observations: dict[int, dict[str, Any]],
) -> None:
    def fake_auth(
        record: dict[str, Any],
        *,
        expected_task_id: int,
    ) -> tuple[Path, dict[str, Any]]:
        path = watch._verify_file_record(record, "test observation")
        return path, observations[expected_task_id]

    monkeypatch.setattr(watch, "_authenticate_observation", fake_auth)


def _patch_local_acquisition(
    monkeypatch: pytest.MonkeyPatch,
    *,
    calls: list[dict[str, Any]],
    complete_candidate_count: int,
) -> None:
    anchors = [
        SimpleNamespace(candidate_physics_sha256=value)
        for value in LOCAL_SHA_BY_TASK.values()
    ]
    monkeypatch.setattr(
        watch.acquisition,
        "load_authenticated_anchor_set",
        lambda _path: ({}, anchors, Path("standard.csv"), Path("front.csv")),
    )

    def fake_prepare(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        count = (
            complete_candidate_count
            if kwargs["current_results_complete"]
            else 0
        )
        output = Path(kwargs["output_directory"])
        candidates = [
            {
                "decoded_params": {
                    "full_model": 0,
                    "thermal_symmetry": "eighth",
                }
            }
            for _index in range(count)
        ]
        candidate_set = acquisition.seal(
            {
                "schema_version": acquisition.CANDIDATE_SET_SCHEMA,
                "prepare_only": True,
                "stage": "symmetric_standard",
                "candidate_count": count,
                "candidates": candidates,
                "scheduler_post_calls": 0,
                "automatic_full_trigger": False,
            }
        )
        candidate_path = _write_json(
            output / "local_symmetric_candidates.json",
            candidate_set,
        )
        manifest = acquisition.seal(
            {
                "schema_version": acquisition.MANIFEST_SCHEMA,
                "candidate_count": count,
                "finite_stop_criteria": {
                    "current_round": kwargs["current_round"],
                    "max_correction_rounds": kwargs["max_rounds"],
                    "max_parallel_fea_batch": kwargs["max_batch"],
                },
                "scheduler_submission_performed": False,
                "submission_capability_present": False,
                "scheduler_methods_used": [],
                "automatic_full_trigger": False,
                "automatic_candidate_continuation": False,
            }
        )
        manifest_path = _write_json(
            output / "acquisition_manifest.json", manifest
        )
        return {
            "manifest_path": str(manifest_path),
            "candidate_set_path": str(candidate_path),
            "candidate_count": count,
            "stop_reason": (
                None if count else "awaiting_current_symmetric_results"
            ),
            "measured_selection": {
                "status": "test",
                "new_local_neighbors_required": bool(count),
            },
        }

    monkeypatch.setattr(
        watch.acquisition,
        "prepare_local_acquisition",
        fake_prepare,
    )


def test_pending_measurement_emits_nds_but_no_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations = {
        96331: _observation(
            96331,
            candidate_sha=LOCAL_SHA_BY_TASK[96331],
        )
    }
    source = _source_state(
        tmp_path,
        {96331: "collection_ready"},
        observations,
    )
    _patch_observation_auth(monkeypatch, observations)
    calls: list[dict[str, Any]] = []
    _patch_local_acquisition(
        monkeypatch,
        calls=calls,
        complete_candidate_count=3,
    )

    state = watch.process_cycle(
        source_state_path=source,
        aggregate_manifest=tmp_path / "aggregate.json",
        output_root=tmp_path / "output",
    )

    assert state["status"] == "partial_measured_waiting"
    assert state["watch_complete"] is False
    assert state["exact_measured_nds"]["row_count"] == 1
    assert state["finite_local_budget"]["candidate_count_this_round"] == 0
    assert len(calls) == 1
    assert calls[0]["current_results_complete"] is False
    assert calls[0]["pending_official_task_ids"] == [
        task_id for task_id in TASK_IDS if task_id != 96331
    ]
    assert calls[0]["max_batch"] == 3
    assert calls[0]["max_rounds"] == 2
    assert state["scheduler_methods_used"] == []
    assert state["automatic_full_trigger"] is False


def test_any_actual_pass_stops_immediately_and_uses_required_ranking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations = {
        96330: _observation(
            96330,
            candidate_sha=LOCAL_SHA_BY_TASK[96330],
            passed=True,
            normalized_margin=0.01,
            loss=5200.0,
        ),
        96331: _observation(
            96331,
            candidate_sha=LOCAL_SHA_BY_TASK[96331],
            passed=True,
            normalized_margin=0.02,
            loss=5400.0,
        ),
    }
    source = _source_state(
        tmp_path,
        {
            96330: "collection_ready",
            96331: "collection_ready",
        },
        observations,
    )
    _patch_observation_auth(monkeypatch, observations)
    monkeypatch.setattr(
        watch,
        "_run_acquisition",
        lambda **_kwargs: pytest.fail(
            "a passing symmetric result must not open local acquisition"
        ),
    )

    state = watch.process_cycle(
        source_state_path=source,
        aggregate_manifest=tmp_path / "aggregate.json",
        output_root=tmp_path / "output",
    )

    assert state["status"] == "selected_symmetric_hard_pass"
    assert state["watch_complete"] is True
    assert state["pending_count"] == 5
    assert state["selected_symmetric_result"]["task_id"] == 96331
    assert state["local_acquisition"] is None
    assert state["finite_local_budget"]["candidate_count_this_round"] == 0
    assert state["full_model_started_by_watcher"] is False


def test_replacement_96338_is_selected_while_96332_and_96337_are_lifecycle_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations = {
        96332: _observation(
            96332,
            candidate_sha="2" * 64,
            passed=True,
            normalized_margin=0.50,
            loss=100.0,
        ),
        96337: _observation(
            96337,
            candidate_sha="7" * 64,
            passed=True,
            normalized_margin=0.40,
            loss=200.0,
        ),
        96338: _observation(
            96338,
            candidate_sha=LOCAL_SHA_BY_TASK[96338],
            passed=True,
            normalized_margin=0.01,
            loss=5200.0,
        ),
    }
    source = _source_state(
        tmp_path,
        {
            96332: "collection_ready",
            96337: "collection_ready",
            96338: "collection_ready",
        },
        observations,
    )
    authenticated: list[int] = []

    def authenticate_effective(
        record: dict[str, Any],
        *,
        expected_task_id: int,
    ) -> tuple[Path, dict[str, Any]]:
        assert expected_task_id not in {96332, 96337}
        authenticated.append(expected_task_id)
        return (
            watch._verify_file_record(record, "test observation"),
            observations[expected_task_id],
        )

    monkeypatch.setattr(
        watch, "_authenticate_observation", authenticate_effective
    )
    monkeypatch.setattr(
        watch,
        "_run_acquisition",
        lambda **_kwargs: pytest.fail(
            "replacement hard pass must stop without acquisition"
        ),
    )

    state = watch.process_cycle(
        source_state_path=source,
        aggregate_manifest=tmp_path / "aggregate.json",
        output_root=tmp_path / "output",
    )

    assert authenticated == [96338]
    assert state["selected_symmetric_result"]["task_id"] == 96338
    assert state["authenticated_observation_count"] == 1
    assert state["exact_measured_nds"]["row_count"] == 1
    assert state["effective_lane_count"] == 7
    assert state["lifecycle_lane_count"] == 9
    assert state["selection_superseded_task_ids"] == [96332, 96337]
    lifecycle = {
        lane["task_id"]: lane
        for lane in state["lanes"]
        if not lane["selection_effective"]
    }
    assert set(lifecycle) == {96332, 96337}
    assert all(
        lane["selection_eligibility"] == "superseded_lifecycle_only"
        and lane["authenticated_for_selection"] is False
        and lane["included_in_exact_measured_nds"] is False
        for lane in lifecycle.values()
    )
    assert state["full_model_started_by_watcher"] is False
    assert state["automatic_full_trigger"] is False


def test_all_terminal_excludes_failure_and_caps_prepare_only_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations = {
        task_id: _observation(
            task_id,
            candidate_sha=(
                LOCAL_SHA_BY_TASK[task_id]
                if task_id in LOCAL_SHA_BY_TASK
                else NONLOCAL_SHA_BY_TASK[task_id]
            ),
        )
        for task_id in TASK_IDS
        if task_id != 96333
    }
    statuses = {
        task_id: (
            "terminal_failure"
            if task_id == 96333
            else "collection_ready"
        )
        for task_id in TASK_IDS
    }
    source = _source_state(tmp_path, statuses, observations)
    _patch_observation_auth(monkeypatch, observations)
    calls: list[dict[str, Any]] = []
    _patch_local_acquisition(
        monkeypatch,
        calls=calls,
        complete_candidate_count=3,
    )

    state = watch.process_cycle(
        source_state_path=source,
        aggregate_manifest=tmp_path / "aggregate.json",
        output_root=tmp_path / "output",
        current_round=1,
    )

    assert state["status"] == "local_prepare_only_batch_ready"
    assert state["watch_complete"] is True
    assert state["current_results_complete"] is True
    assert state["terminal_failure_count"] == 1
    assert state["exact_measured_nds"]["row_count"] == 6
    assert len(calls) == 1
    assert calls[0]["excluded_invalid_task_ids"] == [96333]
    assert calls[0]["pending_official_task_ids"] == []
    assert len(calls[0]["observation_paths"]) == 4
    assert state["local_acquisition"]["candidate_count"] == 3
    assert state["finite_local_budget"] == {
        "current_round": 1,
        "max_rounds": 2,
        "max_candidates_per_round": 3,
        "max_candidates_total": 6,
        "candidate_count_this_round": 3,
    }
    failed_lane = next(
        item for item in state["lanes"] if item["task_id"] == 96333
    )
    assert (
        failed_lane["selection_eligibility"]
        == "excluded_operational_failure"
    )
    assert failed_lane["local_neighbor_generation_from_failure_allowed"] is False


def test_tampered_source_state_publishes_sealed_fail_closed_heartbeat(
    tmp_path: Path,
) -> None:
    source = _source_state(tmp_path, {}, {})
    raw = json.loads(source.read_text("utf-8"))
    raw["pending_count"] = 0
    _write_json(source, raw)
    output = tmp_path / "output"

    state = watch.process_cycle(
        source_state_path=source,
        aggregate_manifest=tmp_path / "aggregate.json",
        output_root=output,
    )

    assert state["status"] == "blocked_fail_closed"
    assert state["watch_complete"] is False
    assert state["local_acquisition"] is None
    persisted = watch._validate_seal(
        json.loads((output / "state.json").read_text("utf-8")),
        expected_schema=watch.STATE_SCHEMA,
        label="persisted state",
    )
    heartbeat = watch._validate_seal(
        json.loads((output / "heartbeat.json").read_text("utf-8")),
        expected_schema=watch.HEARTBEAT_SCHEMA,
        label="heartbeat",
    )
    assert persisted["payload_sha256"] == state["payload_sha256"]
    assert heartbeat["state"]["payload_sha256"] == state["payload_sha256"]
    assert heartbeat["scheduler_methods_used"] == []
    assert heartbeat["automatic_full_trigger"] is False


def test_effective_replacement_mapping_drift_fails_closed(
    tmp_path: Path,
) -> None:
    source = _source_state(tmp_path, {}, {})
    raw = json.loads(source.read_text("utf-8"))
    raw.pop("payload_sha256")
    for lane in raw["lanes"]:
        if lane["task_id"] == 96337:
            lane["selection_effective"] = True
        elif lane["task_id"] == 96338:
            lane["selection_effective"] = False
    raw["effective_task_ids"] = [
        96337 if task_id == 96338 else task_id
        for task_id in raw["effective_task_ids"]
    ]
    raw["selection_superseded_task_ids"] = [96332, 96338]
    _write_json(source, watch._seal(raw))

    state = watch.process_cycle(
        source_state_path=source,
        aggregate_manifest=tmp_path / "aggregate.json",
        output_root=tmp_path / "output",
    )

    assert state["status"] == "blocked_fail_closed"
    assert "authority drifted" in state["fail_closed_error"]["message"]
    assert state["automatic_full_trigger"] is False


def test_exact_measured_nds_is_over_all_authenticated_observations(
    tmp_path: Path,
) -> None:
    observations = [
        (
            tmp_path / "a.json",
            _observation(
                1,
                candidate_sha="a" * 64,
                volume=1.0,
                loss=2.0,
            ),
        ),
        (
            tmp_path / "b.json",
            _observation(
                2,
                candidate_sha="b" * 64,
                volume=2.0,
                loss=1.0,
            ),
        ),
        (
            tmp_path / "c.json",
            _observation(
                3,
                candidate_sha="c" * 64,
                volume=3.0,
                loss=3.0,
            ),
        ),
    ]
    output = tmp_path / "nds.json"

    result = watch._write_exact_measured_nds(
        observations, output_path=output
    )

    assert [row["exact_measured_non_dominated_rank"] for row in result["rows"]] == [
        0,
        0,
        1,
    ]
    assert result["row_count"] == 3
    assert result["rank0_count"] == 2
    assert result["scheduler_methods_used"] == []
