from __future__ import annotations

import copy
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pytest

from tools import mft_goal_official_standard_hedge_prepare as hedge


OBSERVED_AT = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)


def test_reviewed_candidate_mapping_is_exact_and_excludes_storage_guard():
    assert [
        (
            spec.selection_order,
            spec.candidate_sha256[:12],
            spec.account_name,
            spec.node_name,
        )
        for spec in hedge.CANDIDATES
    ] == [
        (1, "896084a59793", "dhj02", "n110"),
        (12, "828cb282cf4f", "r1jae262", "n112"),
        (5, "909d249ebe45", "jji0930", "n115"),
    ]
    assert len({spec.candidate_sha256 for spec in hedge.CANDIDATES}) == 3
    assert len({spec.node_name for spec in hedge.CANDIDATES}) == 3
    assert len({spec.account_name for spec in hedge.CANDIDATES}) == 3
    assert "harry261" in hedge.EXCLUDED_ACCOUNT_GUARDS
    assert all(
        spec.account_name not in hedge.EXCLUDED_ACCOUNT_GUARDS
        for spec in hedge.CANDIDATES
    )


def test_prepare_only_parser_has_no_submit_command():
    parser = hedge._parser()
    parsed = parser.parse_args(["prepare"])
    assert parsed.command == "prepare"
    with pytest.raises(SystemExit):
        parser.parse_args(["submit"])
    assert not hasattr(hedge, "submit")


def _fake_specs_and_profile():
    profile = {
        "schema_version": "test-profile-v1",
        "param_overrides": {
            "thermal_on": 1,
            "loss_on": 1,
            "keep_project": 1,
        },
    }
    specs = []
    params_by_order = {}
    for spec in hedge.CANDIDATES:
        params = {
            "selection_order": spec.selection_order,
            "thermal_on": 0,
            "loss_on": 0,
            "keep_project": 0,
        }
        profiled = copy.deepcopy(params)
        profiled.update(profile["param_overrides"])
        fake_spec = replace(
            spec,
            raw_fea_params_sha256=hedge.payload_sha256(params),
            profiled_fea_params_sha256=hedge.payload_sha256(profiled),
        )
        specs.append(fake_spec)
        params_by_order[spec.selection_order] = params
    return tuple(specs), profile, params_by_order


def _fake_selected(spec: hedge.CandidateSpec) -> dict[str, Any]:
    return hedge.sealed(
        {
            "schema_version": hedge.CANDIDATE_SCHEMA,
            **hedge.SAFETY_FLAGS,
            "campaign_id": hedge.CAMPAIGN_ID,
            "prepare_only": True,
            "submission_capability_present": False,
            "candidate_physics_sha256": spec.candidate_sha256,
        }
    )


def _fake_preflight(
    spec: hedge.CandidateSpec,
    _dedupe: str,
    _observed_at: datetime | None,
) -> dict[str, Any]:
    return {
        "schema_version": hedge.PREFLIGHT_SCHEMA,
        "account_name": spec.account_name,
        "node_name": spec.node_name,
        "node_name_policy": "strict",
        "queue_state": "opening",
        "queue_reason": hedge.OPENING_QUEUE_REASON,
        "ready_fit_slots": 0,
        "pending_fit_slots": 0,
        "inflight_fit_slots": 0,
        "active_target_fea_count": 0,
        "preferred_node_relaxed": False,
        "scheduler_get_only": True,
        "scheduler_post_calls": 0,
    }


def test_prepare_writes_three_atomic_sealed_get_only_plans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    specs, profile, params_by_order = _fake_specs_and_profile()
    monkeypatch.setattr(hedge, "CANDIDATES", specs)
    authentications = []
    payloads = []
    preflights = []

    def authenticate(spec):
        authentications.append(spec.selection_order)
        return (
            _fake_selected(spec),
            copy.deepcopy(params_by_order[spec.selection_order]),
        )

    def build(spec, _params, _profile):
        payloads.append(spec.selection_order)
        dedupe = f"test-dedupe-{spec.selection_order}"
        return (
            {
                "name": spec.task_name,
                "project": hedge.PROJECT,
                "account_name": spec.account_name,
                "node_name": spec.node_name,
                "node_name_policy": "strict",
                "cpus": hedge.CPUS,
                "memory_mb": hedge.MEMORY_MB,
                "max_workers_per_node": hedge.MAX_WORKERS_PER_NODE,
                "aedt_backend": "standalone",
                "required_capability": "conda:pyaedt2026v1",
                "timeout_seconds": hedge.SCHEDULER_SECONDS,
                "dedupe_key": dedupe,
                "command": "prepared but not submitted",
            },
            {"MFT_STANDALONE_CORE_COUNT": "8"},
            {
                "dedupe_key": dedupe,
                "artifact_path": f"retained/{spec.short_sha}/symmetric.aedt",
            },
        )

    def preflight(spec, dedupe, observed_at):
        preflights.append((spec.selection_order, dedupe, observed_at))
        return _fake_preflight(spec, dedupe, observed_at)

    output = tmp_path / "prepared"
    manifest_path = hedge.prepare(
        output=output,
        specs=specs,
        authenticator=authenticate,
        profile_reader=lambda: copy.deepcopy(profile),
        payload_builder=build,
        preflight=preflight,
        observed_at=OBSERVED_AT,
    )

    assert authentications == [1, 12, 5]
    assert payloads == [1, 12, 5]
    assert [item[0] for item in preflights] == [1, 12, 5]
    manifest = hedge.validate_seal(
        hedge.read_json(manifest_path), hedge.MANIFEST_SCHEMA
    )
    assert manifest["official_standard_candidate_count"] == 3
    assert manifest["official_standard_selection_orders"] == [1, 12, 5]
    assert manifest["scheduler_get_preflights"] == 3
    assert manifest["scheduler_post_calls"] == 0
    assert manifest["scheduler_submission_performed"] is False
    assert manifest["submission_capability_present"] is False

    for lane in manifest["lanes"]:
        lane_root = output / Path(lane["plan"]["path"]).parent
        plan = hedge.validate_seal(
            hedge.read_json(output / lane["plan"]["path"]),
            hedge.PLAN_SCHEMA,
        )
        receipt = hedge.validate_seal(
            hedge.read_json(lane_root / hedge.RECEIPT_NAME),
            hedge.PREPARE_SCHEMA,
        )
        assert plan["prepare_only"] is True
        assert plan["scheduler_post_calls"] == 0
        assert plan["scheduler_submission_performed"] is False
        assert plan["submission_capability_present"] is False
        assert plan["future_submission_implementation_present"] is False
        assert plan["placement"]["node_name_policy"] == "strict"
        assert plan["placement"]["allocation_state_at_prepare"] == "opening"
        assert plan["placement"]["preferred_node_relaxed_allowed"] is False
        assert receipt["submit_command_argv"] is None
        assert receipt["ready_for_submission"] is False
        assert receipt["scheduler_post_calls"] == 0


def test_prepare_rolls_back_if_any_preflight_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    specs, profile, params_by_order = _fake_specs_and_profile()
    monkeypatch.setattr(hedge, "CANDIDATES", specs)

    def authenticate(spec):
        return _fake_selected(spec), params_by_order[spec.selection_order]

    def build(spec, _params, _profile):
        dedupe = f"test-dedupe-{spec.selection_order}"
        return (
            {
                "name": spec.task_name,
                "project": hedge.PROJECT,
                "account_name": spec.account_name,
                "node_name": spec.node_name,
                "node_name_policy": "strict",
                "cpus": hedge.CPUS,
                "memory_mb": hedge.MEMORY_MB,
                "max_workers_per_node": hedge.MAX_WORKERS_PER_NODE,
                "aedt_backend": "standalone",
                "required_capability": "conda:pyaedt2026v1",
                "timeout_seconds": hedge.SCHEDULER_SECONDS,
                "dedupe_key": dedupe,
            },
            {},
            {"dedupe_key": dedupe},
        )

    def fail_third(spec, dedupe, observed_at):
        if spec.selection_order == 5:
            raise hedge.PostdeadlineContractError("third GET gate failed")
        return _fake_preflight(spec, dedupe, observed_at)

    output = tmp_path / "prepared"
    with pytest.raises(
        hedge.PostdeadlineContractError, match="third GET gate failed"
    ):
        hedge.prepare(
            output=output,
            specs=specs,
            authenticator=authenticate,
            profile_reader=lambda: profile,
            payload_builder=build,
            preflight=fail_third,
            observed_at=OBSERVED_AT,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".prepared.*.tmp"))


def _live_reader(spec: hedge.CandidateSpec):
    calls = []

    def reader(path, query):
        calls.append((path, query))
        if path == "/api/health":
            return {
                "ok": True,
                "scheduler_ok": True,
                "scheduler_thread_alive": True,
                "scheduler_stalled": False,
            }
        if path == f"/api/tasks/{spec.source_task_id}":
            return {
                "task_id": spec.source_task_id,
                "name": spec.source_task_name,
                "status": "completed",
                "exit_code": 0,
                "required_capability": "conda:pyaedt2026v1",
                "env_profile": "pyaedt2026v1",
                "dedupe_key": spec.source_task_dedupe_key,
            }
        if path == "/api/task-capacity":
            query_map = dict(query or [])
            assert query_map["account_name"] == spec.account_name
            assert query_map["node_name"] == spec.node_name
            assert query_map["node_name_policy"] == "strict"
            return {
                "queue_state": "opening",
                "queue_reason": hedge.OPENING_QUEUE_REASON,
                "fit_slots": 0,
                "ready_fit_slots": 0,
                "pending_fit_slots": 0,
                "inflight_fit_slots": 0,
                "memory_pressure_state": "ok",
                "preferred_node_relaxed": False,
                "standalone_aedt_available": 395,
                "allocations": [],
            }
        if path == "/api/tasks":
            return []
        if path == "/api/accounts/status/live":
            return [
                {
                    "account_name": spec.account_name,
                    "running": 2,
                    "pending": 0,
                    "max_running": 10,
                    "max_pending": 10,
                    "max_total": 20,
                }
            ]
        if path == "/api/capabilities":
            return [
                {
                    "capability": "conda:pyaedt2026v1",
                    "accounts": [spec.account_name],
                }
            ]
        raise AssertionError(f"unexpected GET {path} {query}")

    return reader, calls


def test_live_preflight_reuses_candidate8_get_gate_and_restores_globals():
    spec = hedge.CANDIDATES[0]
    reader, calls = _live_reader(spec)
    old_values = {
        "ACCOUNT_NAME": hedge.candidate8.ACCOUNT_NAME,
        "NODE_NAME": hedge.candidate8.NODE_NAME,
        "SOURCE_TASK_ID": hedge.candidate8.SOURCE_TASK_ID,
        "TASK_NAME": hedge.candidate8.TASK_NAME,
    }
    result = hedge.live_preflight(
        spec,
        expected_dedupe_key="new-dedupe",
        reader=reader,
        observed_at=OBSERVED_AT,
    )
    assert result["schema_version"] == hedge.PREFLIGHT_SCHEMA
    assert result["account_name"] == "dhj02"
    assert result["node_name"] == "n110"
    assert result["active_target_fea_count"] == 0
    assert result["scheduler_get_only"] is True
    assert result["scheduler_post_calls"] == 0
    assert {path for path, _query in calls} >= {
        "/api/health",
        f"/api/tasks/{spec.source_task_id}",
        "/api/task-capacity",
        "/api/tasks",
        "/api/accounts/status/live",
        "/api/capabilities",
    }
    assert {
        "ACCOUNT_NAME": hedge.candidate8.ACCOUNT_NAME,
        "NODE_NAME": hedge.candidate8.NODE_NAME,
        "SOURCE_TASK_ID": hedge.candidate8.SOURCE_TASK_ID,
        "TASK_NAME": hedge.candidate8.TASK_NAME,
    } == old_values


def test_derive_payload_uses_exact_lane_and_restores_reviewed_globals(
    monkeypatch: pytest.MonkeyPatch,
):
    spec = hedge.CANDIDATES[1]
    observed = {}
    old = {
        "ACCOUNT_NAME": hedge.reviewed.ACCOUNT_NAME,
        "NODE_NAME": hedge.reviewed.NODE_NAME,
        "TASK_NAME": hedge.reviewed.TASK_NAME,
        "WORKDIR": hedge.reviewed.WORKDIR,
    }

    def capture(params, profile):
        observed.update(
            {
                "account": hedge.reviewed.ACCOUNT_NAME,
                "node": hedge.reviewed.NODE_NAME,
                "task": hedge.reviewed.TASK_NAME,
                "workdir": hedge.reviewed.WORKDIR,
                "params": params,
                "profile": profile,
            }
        )
        return (
            {"dedupe_key": "lane-dedupe"},
            {"ENV": "1"},
            {"dedupe_key": "lane-dedupe"},
        )

    monkeypatch.setattr(hedge.reviewed, "_capture_scheduler_payload", capture)
    monkeypatch.setattr(
        hedge.reviewed,
        "validate_scheduler_payload",
        lambda _payload, _retained: None,
    )
    result = hedge.derive_scheduler_payload(
        spec, {"x": 1}, {"param_overrides": {}}
    )
    assert result[0]["dedupe_key"] == "lane-dedupe"
    assert observed["account"] == spec.account_name
    assert observed["node"] == spec.node_name
    assert observed["task"] == spec.task_name
    assert observed["workdir"] == spec.workdir
    assert {
        "ACCOUNT_NAME": hedge.reviewed.ACCOUNT_NAME,
        "NODE_NAME": hedge.reviewed.NODE_NAME,
        "TASK_NAME": hedge.reviewed.TASK_NAME,
        "WORKDIR": hedge.reviewed.WORKDIR,
    } == old


def test_manifest_json_has_no_submission_authority(tmp_path: Path):
    payload = hedge.sealed(
        {
            "schema_version": hedge.MANIFEST_SCHEMA,
            "prepare_only": True,
            "submission_capability_present": False,
            "scheduler_post_calls": 0,
        }
    )
    path = hedge.write_immutable_json(tmp_path / "manifest.json", payload)
    decoded = json.loads(path.read_text(encoding="utf-8"))
    assert decoded["prepare_only"] is True
    assert decoded["submission_capability_present"] is False
    assert decoded["scheduler_post_calls"] == 0
