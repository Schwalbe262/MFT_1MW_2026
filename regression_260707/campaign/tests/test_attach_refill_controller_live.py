from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from unittest import mock

import pytest


CAMPAIGN_ROOT = Path(__file__).resolve().parents[1]
REGRESSION_ROOT = CAMPAIGN_ROOT.parent
VERIFY_ROOT = REGRESSION_ROOT / "verify"
for item in (CAMPAIGN_ROOT, REGRESSION_ROOT, VERIFY_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import attach_aware_refill_controller as controller


SCHEDULER_URL = "http://scheduler.test:8000"


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _scheduler_get(active: int):
    if active < 2:
        raise ValueError("test scheduler fixture requires at least two active tasks")
    project_tasks = [
        {
            "id": task_id,
            "name": f"project-task-{task_id}",
            "project": controller.scheduler_client.MFT_PROJECT,
            "status": "queued" if task_id < active - 1 else "attaching",
        }
        for task_id in range(1, active)
    ]
    legacy_tasks = [
        {
            "id": active,
            "name": (
                f"{controller.scheduler_client.LEGACY_MFT_NAME_PREFIX}{active}"
            ),
            "project": "",
            "status": "running",
        }
    ]
    project = {
        "name": controller.scheduler_client.MFT_PROJECT,
        "max_active_tasks": 500,
        "auto_pull": False,
    }

    def get(url, params=None, timeout=None):
        assert timeout == 30
        if url == (
            f"{SCHEDULER_URL}/api/projects/"
            f"{controller.scheduler_client.MFT_PROJECT}"
        ):
            assert params is None
            return _Response(project)
        if url == f"{SCHEDULER_URL}/api/tasks" and params:
            if params.get("project") == controller.scheduler_client.MFT_PROJECT:
                return _Response(project_tasks)
            if params.get("name_prefix") == (
                controller.scheduler_client.LEGACY_MFT_NAME_PREFIX
            ):
                return _Response(legacy_tasks)
        raise AssertionError(f"unexpected scheduler GET: {url} {params}")

    return get


def _candidate(cursor=0, seed=controller.DEFAULT_CANDIDATE_SEED):
    return cursor + 1, cursor, {"candidate": cursor, "seed": seed}


def _forbid_mutations(monkeypatch):
    sentinels = {
        "post": mock.Mock(side_effect=AssertionError("scheduler POST forbidden")),
        "submit": mock.Mock(side_effect=AssertionError("task submit forbidden")),
        "cancel": mock.Mock(side_effect=AssertionError("task cancel forbidden")),
        "cancel_cas": mock.Mock(
            side_effect=AssertionError("queued task cancellation forbidden")
        ),
    }
    monkeypatch.setattr(controller.scheduler_client.requests, "post", sentinels["post"])
    monkeypatch.setattr(
        controller.scheduler_client, "submit_verification", sentinels["submit"]
    )
    monkeypatch.setattr(controller.scheduler_client, "cancel", sentinels["cancel"])
    monkeypatch.setattr(
        controller.scheduler_client,
        "cancel_queued_tasks_cas",
        sentinels["cancel_cas"],
    )
    return sentinels


def _plan_args(state_path: Path) -> list[str]:
    return [
        "plan",
        "--policy",
        str(controller.DEFAULT_POLICY_PATH),
        "--scheduler-url",
        SCHEDULER_URL,
        "--state-path",
        str(state_path),
    ]


def _run_args(state_path: Path, generation_id: str) -> list[str]:
    return [
        "run",
        "--policy",
        str(controller.DEFAULT_POLICY_PATH),
        "--scheduler-url",
        SCHEDULER_URL,
        "--state-path",
        str(state_path),
        "--authorize-generation",
        generation_id,
    ]


def test_live_plan_is_get_only_standalone_and_repeatable(
    tmp_path, monkeypatch, capsys
):
    state_path = tmp_path / "restart_v3_controller_state.json"
    get = mock.Mock(side_effect=_scheduler_get(active=498))
    monkeypatch.setattr(controller.scheduler_client.requests, "get", get)
    monkeypatch.setattr(controller.pinned_pilot, "next_valid_candidate", _candidate)
    submission_identity = (
        controller.scheduler_client.verification_submission_identity
    )
    captured_identities = []

    def capture_identity(*args, **kwargs):
        identity = submission_identity(*args, **kwargs)
        captured_identities.append(identity)
        return identity

    monkeypatch.setattr(
        controller.scheduler_client,
        "verification_submission_identity",
        capture_identity,
    )
    sentinels = _forbid_mutations(monkeypatch)

    assert controller.main(_plan_args(state_path)) == 0
    first = json.loads(capsys.readouterr().out)
    first_state_bytes = state_path.read_bytes()

    assert first["mode"] == "plan"
    assert first["scheduler_query_count"] == 3
    assert first["scheduler_mutation_count"] == 0
    assert first["active_counts"] == {
        "queued": 496,
        "attaching": 1,
        "running": 1,
    }
    assert first["active_project_tasks"] == 498
    assert first["logical_project_deficit"] == 2
    assert first["would_submit"] == first["would_replace"] == 2
    assert first["state_initialized"] is True
    assert first["selected_backend"] == "standalone"
    assert first["backend_reason"] == "pooled_fraction_zero"
    assert first["pooled_fraction"] == 0.0
    assert first["pooled_project_count"] == 0
    assert first["standalone_project_count"] == 2
    assert first["projects_per_aedt"] == 2
    assert first["desired_aedt_sessions"] == 0
    assert first["cancel_task_ids"] == []
    assert first["mass_cancel_authorized"] is False
    assert not any(call.args[0].endswith("/api/aedt-pool") for call in get.call_args_list)

    policy = controller._load_policy(controller.DEFAULT_POLICY_PATH)
    profile = controller._load_profile(controller.DEFAULT_TIMEOUT_SECONDS)
    physics_revision = policy.provenance.physics_data_revision
    lamination_factor = policy.provenance.core_lamination_factor
    assert len(captured_identities) == 2
    for action, identity in zip(first["planned_actions"], captured_identities):
        assert action["name"].startswith("mft-camp-rv3-")
        assert action["backend"] == "standalone"
        assert action["params"]["physics_data_revision"] == physics_revision
        assert action["params"]["core_lamination_factor"] == lamination_factor
        assert identity["merged"]["physics_data_revision"] == physics_revision
        assert identity["merged"]["core_lamination_factor"] == lamination_factor
        assert identity["dedupe_key"] == action["dedupe_key"]

        changed_revision = dict(action["params"])
        changed_revision["physics_data_revision"] = physics_revision + "-changed"
        changed_factor = dict(action["params"])
        changed_factor["core_lamination_factor"] = 0.70
        for changed in (changed_revision, changed_factor):
            changed_identity = submission_identity(
                action["name"],
                changed,
                profile,
                policy.provenance.solver_revision,
                policy.provenance.library_revision,
                dedupe_scope=first["generation"]["digest"],
            )
            assert changed_identity["dedupe_key"] != action["dedupe_key"]

    state = json.loads(first_state_bytes)
    assert state["schema"] == controller.STATE_SCHEMA
    assert state["generation"] == first["generation"]
    start_cursor = first["generation"]["identity"]["candidate_start_cursor"]
    assert first["generation"]["identity"]["candidate_valid_offset"] == 35
    assert first["generation"]["identity"]["project_concurrency_target"] == 500
    assert state["candidate_cursor"] == start_cursor
    assert state["next_serial"] == 1
    assert state["state_revision"] == 0
    assert state["submissions"] == []

    assert controller.main(_plan_args(state_path)) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["state_initialized"] is False
    assert second["generation"] == first["generation"]
    assert second["candidate_cursor"] == first["candidate_cursor"] == start_cursor
    assert second["state_revision"] == first["state_revision"] == 0
    assert second["planned_actions"] == first["planned_actions"]
    assert state_path.read_bytes() == first_state_bytes

    assert get.call_count == 6
    for sentinel in sentinels.values():
        sentinel.assert_not_called()


def test_generation_identity_changes_with_concurrency_and_physics_pins():
    policy = controller._load_policy(controller.DEFAULT_POLICY_PATH)
    profile = controller._load_profile(controller.DEFAULT_TIMEOUT_SECONDS)

    def generation(candidate_policy):
        return controller._generation_contract(
            candidate_policy,
            candidate_seed=controller.DEFAULT_CANDIDATE_SEED,
            profile=profile,
            cpus=controller.DEFAULT_CPUS,
            memory_mb=controller.DEFAULT_MEMORY_MB,
        )

    base = generation(policy)
    changed_revision = generation(
        replace(
            policy,
            provenance=replace(
                policy.provenance,
                physics_data_revision=(
                    policy.provenance.physics_data_revision + "-changed"
                ),
            ),
        )
    )
    changed_factor = generation(
        replace(
            policy,
            provenance=replace(
                policy.provenance, core_lamination_factor=0.70
            ),
        )
    )
    changed_target = generation(
        replace(policy, project_concurrency_target=499)
    )

    assert len(
        {
            base["id"],
            changed_revision["id"],
            changed_factor["id"],
            changed_target["id"],
        }
    ) == 4
    assert len(
        {
            base["digest"],
            changed_revision["digest"],
            changed_factor["digest"],
            changed_target["digest"],
        }
    ) == 4


def test_live_policy_rejects_nonzero_pooled_fraction(tmp_path):
    payload = json.loads(
        controller.DEFAULT_POLICY_PATH.read_text(encoding="utf-8")
    )
    payload["pooled_fraction"] = 0.01
    path = tmp_path / "unreviewed-pooled-policy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="pooled_fraction.*False"):
        controller._load_policy(path)


@pytest.mark.parametrize(
    "name",
    [
        "adopted_refill_688c6f9_state.json",
        "adopted_refill_688c6f9_feeder_state.json",
        "continuous_refill_b171c7c_state.json",
        "continuous_refill_b171c7c_feeder_state.json",
    ],
)
def test_restart_v3_rejects_old_generation_state_paths(tmp_path, name):
    with pytest.raises(ValueError, match="prior-generation state"):
        controller._validate_state_path(tmp_path / name)


def test_run_with_wrong_generation_fails_before_scheduler_access(
    tmp_path, monkeypatch, capsys
):
    state_path = tmp_path / "restart_v3_controller_state.json"
    get = mock.Mock(side_effect=AssertionError("scheduler GET forbidden"))
    monkeypatch.setattr(controller.scheduler_client.requests, "get", get)
    sentinels = _forbid_mutations(monkeypatch)

    assert controller.main(_run_args(state_path, "restart-v3-wrong")) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["mode"] == "run"
    assert error["action"] == "failed_closed"
    assert "run requires exact --authorize-generation restart-v3-" in error["error"]
    assert not state_path.exists()
    get.assert_not_called()
    for sentinel in sentinels.values():
        sentinel.assert_not_called()


def test_authorized_mocked_run_advances_fresh_state_once(
    tmp_path, monkeypatch, capsys
):
    state_path = tmp_path / "restart_v3_controller_state.json"
    get = mock.Mock(side_effect=_scheduler_get(active=499))
    monkeypatch.setattr(controller.scheduler_client.requests, "get", get)
    monkeypatch.setattr(controller.pinned_pilot, "next_valid_candidate", _candidate)
    sentinels = _forbid_mutations(monkeypatch)

    assert controller.main(_plan_args(state_path)) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["would_submit"] == 1

    submit = mock.Mock(return_value=901)
    monkeypatch.setattr(controller.scheduler_client, "submit_verification", submit)
    monkeypatch.setattr(
        controller.scheduler_client,
        "campaign_mutation_lock",
        lambda: nullcontext(),
    )
    assert controller.main(_run_args(state_path, plan["generation"]["id"])) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["mode"] == "run"
    assert result["action"] == "rolling_refill_complete"
    assert result["accepted_or_reconciled_count"] == 1
    assert result["state_revision"] == 1
    assert result["cancel_task_ids"] == []
    assert result["mass_cancel_authorized"] is False
    submit.assert_called_once()
    submitted = submit.call_args.kwargs
    assert submitted["aedt_backend"] == "standalone"
    assert submitted["params"]["physics_data_revision"] == (
        controller.PRODUCTION_PHYSICS_DATA_REVISION
    )
    assert submitted["params"]["core_lamination_factor"] == (
        controller.PRODUCTION_CORE_LAMINATION_FACTOR
    )
    assert submitted["dedupe_scope"] == plan["generation"]["digest"]

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["state_revision"] == 1
    start_cursor = plan["generation"]["identity"]["candidate_start_cursor"]
    assert state["candidate_cursor"] == start_cursor + 1
    assert state["next_serial"] == 2
    assert len(state["submissions"]) == 1
    assert state["submissions"][0]["task_id"] == 901
    assert state["submissions"][0]["name"] == plan["planned_actions"][0]["name"]
    sentinels["post"].assert_not_called()
    sentinels["cancel"].assert_not_called()
    sentinels["cancel_cas"].assert_not_called()
