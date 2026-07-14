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
    def __init__(self, payload, *, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")
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

    rollout_flip = generation(
        replace(
            policy,
            pooled_fraction=0.75,
            node_local_pooled_enabled=not policy.node_local_pooled_enabled,
        )
    )
    assert rollout_flip["id"] == base["id"]
    assert rollout_flip["digest"] == base["digest"]
    assert len(
        {
            base["digest"],
            changed_revision["digest"],
            changed_factor["digest"],
            changed_target["digest"],
        }
    ) == 4


def test_live_policy_accepts_reviewed_nonzero_pooled_fraction(tmp_path):
    payload = json.loads(
        controller.DEFAULT_POLICY_PATH.read_text(encoding="utf-8")
    )
    payload["pooled_fraction"] = 0.01
    path = tmp_path / "unreviewed-pooled-policy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    policy = controller._load_policy(path)
    assert policy.pooled_fraction == 0.01
    assert controller.pool_gate(policy)["eligible"] is True


def test_live_policy_allows_node_local_readiness_kill_switch(tmp_path):
    payload = json.loads(
        controller.DEFAULT_POLICY_PATH.read_text(encoding="utf-8")
    )
    payload["pooled_fraction"] = 1.0
    payload["node_local_pooled_enabled"] = False
    path = tmp_path / "pooled-disabled-policy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    policy = controller._load_policy(path)

    assert policy.node_local_pooled_enabled is False
    assert controller.pool_gate(policy)["eligible"] is False


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


class _TextResponse(_Response):
    def __init__(self, text):
        super().__init__(None)
        self.text = text


def _active_allocation(
    allocation_id=41,
    account="account-a",
    *,
    free_cpus=64,
    free_memory_mb=262_144,
    state="active",
):
    return {
        "id": allocation_id,
        "account_name": account,
        "state": state,
        "resource_pool": "cpu",
        "node_name": f"n{allocation_id}",
        "slurm_job_id": f"job-{allocation_id}",
        "free_cpus": free_cpus,
        "free_memory_mb": free_memory_mb,
    }


def _pooled_runtime(tmp_path, monkeypatch, *, discovery_text=""):
    policy = replace(
        controller._load_policy(controller.DEFAULT_POLICY_PATH),
        pooled_fraction=1.0,
    )
    profile = controller._load_profile(controller.DEFAULT_TIMEOUT_SECONDS)
    generation = controller._generation_contract(
        policy,
        candidate_seed=controller.DEFAULT_CANDIDATE_SEED,
        profile=profile,
        cpus=controller.DEFAULT_CPUS,
        memory_mb=controller.DEFAULT_MEMORY_MB,
    )
    state_path = tmp_path / "restart_v3_controller_state.json"
    state = controller._new_state(generation, SCHEDULER_URL)
    controller._atomic_save_state(state_path, state)
    monkeypatch.setattr(controller.pinned_pilot, "next_valid_candidate", _candidate)
    monkeypatch.setattr(
        controller.scheduler_client,
        "campaign_mutation_lock",
        lambda: nullcontext(),
    )
    scheduler = {
        "discovery_text": discovery_text,
        "host_posts": [],
        "host_task_id": 701,
        "host_status": "running",
        "host_name": None,
        "host_project": controller.NODE_CANARY_HOST_PROJECT,
        "host_post_status": 201,
        "host_cancel_posts": [],
        "host_cancel_status": 200,
        "host_cancel_exception": None,
        "active_host_tasks": [],
        "host_inventory_gets": [],
        "active_project_tasks": 498,
        "client_statuses": {},
    }

    def get(url, params=None, timeout=None):
        if url == f"{SCHEDULER_URL}/api/allocations":
            assert params is None
            assert timeout == 30
            return _Response([_active_allocation()])
        if url == f"{SCHEDULER_URL}/api/tasks/{scheduler['host_task_id']}":
            assert params is None
            assert timeout == 30
            submitted_name = (
                scheduler["host_posts"][-1]["name"]
                if scheduler["host_posts"]
                else None
            )
            return _Response(
                {
                    "id": scheduler["host_task_id"],
                    "status": scheduler["host_status"],
                    "name": scheduler["host_name"] or submitted_name,
                    "project": scheduler["host_project"],
                }
            )
        if url == (
            f"{SCHEDULER_URL}/api/tasks/{scheduler['host_task_id']}/stdout"
        ):
            assert params == {"max_bytes": controller.NODE_CANARY_STDOUT_MAX_BYTES}
            assert timeout == 30
            return _TextResponse(scheduler["discovery_text"])
        for task_id, status in scheduler["client_statuses"].items():
            if url == f"{SCHEDULER_URL}/api/tasks/{task_id}":
                assert params is None
                assert timeout == 30
                return _Response({"id": task_id, "status": status})
        if url == f"{SCHEDULER_URL}/api/tasks" and params == {
            "limit": 10_000,
            "project": controller.NODE_CANARY_HOST_PROJECT,
            "name_prefix": "mft-aedt-pooled-",
            "status": "queued,attaching,running",
        }:
            assert timeout == 30
            scheduler["host_inventory_gets"].append(dict(params))
            return _Response(list(scheduler["active_host_tasks"]))
        if (
            url == f"{SCHEDULER_URL}/api/tasks"
            and params
            and str(params.get("name_prefix") or "").endswith("-host")
        ):
            assert timeout == 30
            return _Response([])
        return _scheduler_get(active=scheduler["active_project_tasks"])(
            url, params=params, timeout=timeout
        )

    def post(url, json=None, timeout=None):
        if url == (
            f"{SCHEDULER_URL}/api/tasks/"
            f"{scheduler['host_task_id']}/cancel"
        ):
            assert json is None
            assert timeout == 60
            scheduler["host_cancel_posts"].append(scheduler["host_task_id"])
            if scheduler["host_cancel_exception"] is not None:
                raise scheduler["host_cancel_exception"]
            return _Response({}, status_code=scheduler["host_cancel_status"])
        assert url == f"{SCHEDULER_URL}/api/tasks"
        assert timeout == 20
        scheduler["host_posts"].append(dict(json))
        return _Response(
            {"id": scheduler["host_task_id"]},
            status_code=scheduler["host_post_status"],
            text="allocation is no longer attachable",
        )

    monkeypatch.setattr(controller.scheduler_client.requests, "get", get)
    monkeypatch.setattr(controller.scheduler_client.requests, "post", post)
    return policy, profile, generation, state_path, scheduler


def test_node_canary_discovery_line_parsing_is_loopback_and_n2():
    line = "NODE_CANARY_DISCOVERY " + json.dumps(
        {
            "schema_version": 1,
            "mode": "scheduler_managed_node_local_canary",
            "scheduler_url": "http://127.0.0.1:8123",
            "expected_projects": 2,
            "node": "n116",
            "rollback_file": "/tmp/bundle.rollback",
        }
    )

    discovery = controller.parse_node_canary_discovery(
        f"booting\n{line}\n", expected_projects=2
    )

    assert discovery["scheduler_url"] == "http://127.0.0.1:8123"
    assert discovery["node"] == "n116"
    with pytest.raises(ValueError, match="loopback"):
        controller.parse_node_canary_discovery(
            line.replace("127.0.0.1", "scheduler.example"),
            expected_projects=2,
        )
    with pytest.raises(ValueError, match="contract failed"):
        controller.parse_node_canary_discovery(line, expected_projects=1)


def test_allocation_selection_round_robins_accounts_and_respects_capacity():
    two_bundle_capacity = {
        "free_cpus": 18,
        "free_memory_mb": 270_336,
    }
    allocations = [
        _active_allocation(11, "account-a", **two_bundle_capacity),
        _active_allocation(21, "account-b", **two_bundle_capacity),
    ]

    selected, last = controller.select_host_allocations(
        allocations,
        [2, 2, 2, 2],
        client_cpus=4,
        client_memory_mb=65_536,
    )

    assert [item["account_name"] for item in selected] == [
        "account-a",
        "account-b",
        "account-a",
        "account-b",
    ]
    assert last == "account-b"
    resumed, resumed_last = controller.select_host_allocations(
        allocations,
        [2],
        client_cpus=4,
        client_memory_mb=65_536,
        last_account="account-a",
    )
    assert resumed[0]["account_name"] == "account-b"
    assert resumed_last == "account-b"


def test_allocation_selection_never_overbooks_host_bundle_footprint():
    one_slot = _active_allocation(
        31,
        "account-a",
        free_cpus=9,
        free_memory_mb=135_168,
    )
    selected, _ = controller.select_host_allocations(
        [one_slot],
        [2, 2],
        client_cpus=4,
        client_memory_mb=65_536,
    )
    assert [item["id"] if item else None for item in selected] == [31, None]

    selected, _ = controller.select_host_allocations(
        [
            one_slot,
            _active_allocation(32, "account-b", state="warm"),
            _active_allocation(33, "account-b", free_cpus=8),
        ],
        [2],
        client_cpus=4,
        client_memory_mb=65_536,
        reserved_by_allocation={
            31: {"cpus": 9, "memory_mb": 135_168, "hosts": 1}
        },
    )
    assert selected == [None]


def test_pooled_bundle_resume_uses_persisted_host_and_builds_exact_clients(
    tmp_path, monkeypatch
):
    policy, profile, generation, state_path, scheduler = _pooled_runtime(
        tmp_path, monkeypatch
    )
    submit = mock.Mock(side_effect=[801, 802])
    monkeypatch.setattr(controller.scheduler_client, "submit_verification", submit)

    first = controller._run_cycle(
        policy, generation, profile, SCHEDULER_URL, state_path
    )
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    bundle = persisted["pooled_bundles"][0]

    assert first["action"] == "pooled_bundle_pending"
    assert bundle["phase"] == "discovery_wait"
    assert bundle["host_task_id"] == 701
    assert len(scheduler["host_posts"]) == 1
    host = scheduler["host_posts"][0]
    assert host["entrypoint"] == "aedt_node_canary_host"
    assert host["requested_allocation_id"] == 41
    assert host["project"] == "_aedt_pool_hosts"
    assert (
        host["timeout_seconds"]
        == controller.NODE_CANARY_HOST_TASK_TIMEOUT_SECONDS
        == 6 * 3600
    )
    assert controller.NODE_CANARY_HOST_TASK_TIMEOUT_SECONDS != (
        controller.NODE_CANARY_HOST_TIMEOUT_SECONDS
    )
    assert (
        f"--timeout-seconds {controller.NODE_CANARY_HOST_TIMEOUT_SECONDS}"
        in host["command"]
    )
    assert host["payload_json"] == {
        "aedt_canary_bundle_id": bundle["bundle_id"],
        "aedt_canary_expected_projects": 2,
        "aedt_canary_discovery_file": bundle["coordination_files"]["discovery"],
        "aedt_canary_evidence_file": bundle["coordination_files"]["evidence"],
        "aedt_canary_rollback_file": bundle["coordination_files"]["rollback"],
        "aedt_canary_scheduler_revision": controller.NODE_CANARY_SCHEDULER_REVISION,
    }
    assert controller.NODE_CANARY_SCHEDULER_REVISION in host["command"]
    submit.assert_not_called()

    scheduler["discovery_text"] = "NODE_CANARY_DISCOVERY " + json.dumps(
        {
            "schema_version": 1,
            "mode": "scheduler_managed_node_local_canary",
            "scheduler_url": "http://127.0.0.1:8123",
            "expected_projects": 2,
            "node": "n41",
            "rollback_file": bundle["coordination_files"]["rollback"],
        }
    )
    second = controller._run_cycle(
        policy, generation, profile, SCHEDULER_URL, state_path
    )

    assert second["accepted_or_reconciled_count"] == 2
    assert len(scheduler["host_posts"]) == 1
    assert submit.call_count == 2
    for call in submit.call_args_list:
        kwargs = call.kwargs
        assert kwargs["aedt_backend"] == "pooled"
        assert kwargs["entrypoint"] == "aedt_node_canary_client"
        assert kwargs["same_node_as_task_id"] == 701
        assert kwargs["payload_json"] == {
            "aedt_canary_bundle_id": bundle["bundle_id"],
            "aedt_canary_expected_projects": 2,
        }
        assert kwargs["submission_env"]["MFT_AEDT_BACKEND"] == "pooled"
        assert kwargs["submission_env"]["MFT_AEDT_SHARED_CANARY"] == "1"
        assert kwargs["submission_env"]["MFT_AEDT_SCHEDULER_URL"] == (
            "http://127.0.0.1:8123"
        )
        assert kwargs["submission_env"]["MFT_SLURM_SCHEDULER_ROOT"] == (
            f"~/slurm_scheduler/runs/{bundle['bundle_id']}-host"
        )
    resumed = json.loads(state_path.read_text(encoding="utf-8"))
    assert resumed["reservations"] == []
    assert resumed["state_revision"] == 2
    assert resumed["pooled_bundles"][0]["phase"] == "clients_tracked"
    assert resumed["pooled_bundles"][0]["client_task_ids"] == [801, 802]


def test_readiness_kill_switch_falls_back_before_client_admission(
    tmp_path, monkeypatch
):
    policy, profile, generation, state_path, scheduler = _pooled_runtime(
        tmp_path, monkeypatch
    )
    submit = mock.Mock(side_effect=[851, 852])
    monkeypatch.setattr(controller.scheduler_client, "submit_verification", submit)

    controller._run_cycle(policy, generation, profile, SCHEDULER_URL, state_path)
    disabled = replace(policy, node_local_pooled_enabled=False)

    result = controller._run_cycle(
        disabled, generation, profile, SCHEDULER_URL, state_path
    )

    assert result["accepted_or_reconciled_count"] == 2
    assert len(scheduler["host_posts"]) == 1
    assert all(
        call.kwargs["aedt_backend"] == "standalone"
        for call in submit.call_args_list
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    bundle = state["pooled_bundles"][0]
    assert bundle["phase"] == "complete"
    assert "pool gate closed" in bundle["failure_reason"]
    assert bundle["client_task_ids"] == [None, None]


def test_all_terminal_pooled_clients_cancel_host_on_completion(
    tmp_path, monkeypatch
):
    discovery = "NODE_CANARY_DISCOVERY " + json.dumps(
        {
            "schema_version": 1,
            "mode": "scheduler_managed_node_local_canary",
            "scheduler_url": "http://127.0.0.1:8123",
            "expected_projects": 2,
            "node": "n41",
            "rollback_file": "/tmp/bundle.rollback",
        }
    )
    policy, profile, generation, state_path, scheduler = _pooled_runtime(
        tmp_path, monkeypatch, discovery_text=discovery
    )
    submit = mock.Mock(side_effect=[801, 802])
    monkeypatch.setattr(controller.scheduler_client, "submit_verification", submit)

    controller._run_cycle(policy, generation, profile, SCHEDULER_URL, state_path)
    scheduler["client_statuses"] = {801: "completed", 802: "completed"}
    scheduler["active_project_tasks"] = 500
    fetch = mock.Mock(
        return_value=controller.scheduler_client.ResultFetch(
            controller.scheduler_client.RESULT_VALID, {"accepted": True}
        )
    )
    monkeypatch.setattr(controller.scheduler_client, "fetch_result", fetch)

    result = controller._run_cycle(
        policy, generation, profile, SCHEDULER_URL, state_path
    )

    assert scheduler["host_cancel_posts"] == [701]
    assert fetch.call_count == 2
    state = json.loads(state_path.read_text(encoding="utf-8"))
    completed_bundle = state["pooled_bundles"][0]
    assert completed_bundle["phase"] == "complete"
    assert completed_bundle["host_cancel_status"] == "requested"
    assert completed_bundle["host_cancel_error"] is None
    assert completed_bundle["host_cancel_task_id"] == 701
    assert completed_bundle["host_cancel_at"]
    visible_bundle = result["pooled_bundles"][0]
    assert visible_bundle["host_cancel_status"] == "requested"
    assert visible_bundle["host_cancel_error"] is None
    assert visible_bundle["host_cancel_task_id"] == 701
    assert visible_bundle["host_cancel_at"]


def test_discovery_timeout_falls_back_to_new_standalone_identities(
    tmp_path, monkeypatch
):
    policy, profile, generation, state_path, scheduler = _pooled_runtime(
        tmp_path, monkeypatch
    )
    submit = mock.Mock(side_effect=[901, 902])
    monkeypatch.setattr(controller.scheduler_client, "submit_verification", submit)

    controller._run_cycle(policy, generation, profile, SCHEDULER_URL, state_path)
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    persisted["pooled_bundles"][0]["discovery_deadline_at"] = (
        "2000-01-01T00:00:00Z"
    )
    controller._atomic_save_state(state_path, persisted)
    scheduler["discovery_text"] = "NODE_CANARY_DISCOVERY " + json.dumps(
        {
            "schema_version": 1,
            "mode": "scheduler_managed_node_local_canary",
            "scheduler_url": "http://127.0.0.1:8123",
            "expected_projects": 2,
            "node": "late-node",
            "rollback_file": "/tmp/late.rollback",
        }
    )

    result = controller._run_cycle(
        policy, generation, profile, SCHEDULER_URL, state_path
    )

    assert result["accepted_or_reconciled_count"] == 2
    assert len(scheduler["host_posts"]) == 1
    assert scheduler["host_cancel_posts"] == [701]
    assert submit.call_count == 2
    for call in submit.call_args_list:
        assert call.kwargs["aedt_backend"] == "standalone"
        assert "-sa-retry-" in call.kwargs["name"]
        assert "entrypoint" not in call.kwargs
        assert "same_node_as_task_id" not in call.kwargs
    state = json.loads(state_path.read_text(encoding="utf-8"))
    completed_bundle = state["pooled_bundles"][0]
    assert completed_bundle["phase"] == "complete"
    assert completed_bundle["fallback_task_ids"] == [901, 902]
    assert completed_bundle["host_cancel_status"] == "requested"
    assert completed_bundle["host_cancel_error"] is None
    assert completed_bundle["host_cancel_task_id"] == 701
    assert completed_bundle["host_cancel_at"]
    visible_bundle = result["pooled_bundles"][0]
    assert visible_bundle["host_cancel_status"] == "requested"
    assert visible_bundle["host_cancel_error"] is None
    assert visible_bundle["host_cancel_task_id"] == 701
    assert visible_bundle["host_cancel_at"]
    assert all(item["backend"] == "standalone" for item in state["submissions"])
    assert all(
        "-sa-retry-" in item["name"] for item in state["submissions"]
    )
    assert controller._reserved_allocation_footprints(
        state, generation
    ) == {
        41: {"cpus": 1, "memory_mb": 4_096, "hosts": 1}
    }

    scheduler["host_status"] = "completed"
    events = []
    controller._refresh_bundle_host_terminals(
        state, generation, SCHEDULER_URL, state_path, events
    )

    assert state["pooled_bundles"][0]["host_terminal_status"] == "completed"
    assert events[0]["transition"] == "host_terminal_observed"
    assert controller._reserved_allocation_footprints(state, generation) == {}


def test_discovery_fallback_records_cancel_failure_without_raising(
    tmp_path, monkeypatch
):
    policy, profile, generation, state_path, scheduler = _pooled_runtime(
        tmp_path, monkeypatch
    )
    submit = mock.Mock(side_effect=[905, 906])
    monkeypatch.setattr(controller.scheduler_client, "submit_verification", submit)

    controller._run_cycle(policy, generation, profile, SCHEDULER_URL, state_path)
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    persisted["pooled_bundles"][0]["discovery_deadline_at"] = (
        "2000-01-01T00:00:00Z"
    )
    controller._atomic_save_state(state_path, persisted)
    scheduler["host_cancel_status"] = 503

    result = controller._run_cycle(
        policy, generation, profile, SCHEDULER_URL, state_path
    )

    assert result["accepted_or_reconciled_count"] == 2
    assert scheduler["host_cancel_posts"] == [701]
    assert submit.call_count == 2
    state = json.loads(state_path.read_text(encoding="utf-8"))
    completed_bundle = state["pooled_bundles"][0]
    assert completed_bundle["phase"] == "complete"
    assert completed_bundle["fallback_task_ids"] == [905, 906]
    assert completed_bundle["host_cancel_status"] == "failed"
    assert "HTTP 503" in completed_bundle["host_cancel_error"]
    assert completed_bundle["host_cancel_task_id"] == 701
    assert completed_bundle["host_cancel_at"]
    visible_bundle = result["pooled_bundles"][0]
    assert visible_bundle["host_cancel_status"] == "failed"
    assert "HTTP 503" in visible_bundle["host_cancel_error"]
    assert visible_bundle["host_cancel_task_id"] == 701
    assert visible_bundle["host_cancel_at"]


def test_discovery_fallback_refuses_mismatched_host_identity(
    tmp_path, monkeypatch
):
    policy, profile, generation, state_path, scheduler = _pooled_runtime(
        tmp_path, monkeypatch
    )
    submit = mock.Mock(side_effect=[907, 908])
    monkeypatch.setattr(controller.scheduler_client, "submit_verification", submit)

    controller._run_cycle(policy, generation, profile, SCHEDULER_URL, state_path)
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    persisted["pooled_bundles"][0]["discovery_deadline_at"] = (
        "2000-01-01T00:00:00Z"
    )
    controller._atomic_save_state(state_path, persisted)
    mismatched_name = "mft-aedt-pooled-not-this-controller-host"
    scheduler["host_name"] = mismatched_name

    result = controller._run_cycle(
        policy, generation, profile, SCHEDULER_URL, state_path
    )

    assert result["accepted_or_reconciled_count"] == 2
    assert scheduler["host_cancel_posts"] == []
    assert submit.call_count == 2
    state = json.loads(state_path.read_text(encoding="utf-8"))
    completed_bundle = state["pooled_bundles"][0]
    assert completed_bundle["phase"] == "complete"
    assert completed_bundle["fallback_task_ids"] == [907, 908]
    assert completed_bundle["host_cancel_status"] == "identity_mismatch"
    assert mismatched_name in completed_bundle["host_cancel_error"]
    assert completed_bundle["host_cancel_task_id"] == 701
    assert completed_bundle["host_cancel_at"]
    visible_bundle = result["pooled_bundles"][0]
    assert visible_bundle["host_cancel_status"] == "identity_mismatch"
    assert mismatched_name in visible_bundle["host_cancel_error"]
    assert visible_bundle["host_cancel_task_id"] == 701
    assert visible_bundle["host_cancel_at"]


def test_startup_reconciliation_cancels_only_state_owned_stale_host(
    tmp_path, monkeypatch
):
    policy, profile, generation, state_path, scheduler = _pooled_runtime(
        tmp_path, monkeypatch
    )
    submit = mock.Mock(side_effect=AssertionError("client submission forbidden"))
    monkeypatch.setattr(controller.scheduler_client, "submit_verification", submit)

    controller._run_cycle(policy, generation, profile, SCHEDULER_URL, state_path)
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    owned_bundle = persisted["pooled_bundles"][0]
    owned_bundle["phase"] = "complete"
    owned_bundle["failure_reason"] = "simulated stale pre-fix bundle"
    controller._atomic_save_state(state_path, persisted)
    scheduler["active_host_tasks"] = [
        {
            "id": 701,
            "name": owned_bundle["host_name"],
            "project": controller.NODE_CANARY_HOST_PROJECT,
            "status": "running",
        },
        {
            "id": 702,
            "name": "mft-aedt-pooled-state-absent-host",
            "project": controller.NODE_CANARY_HOST_PROJECT,
            "status": "running",
        },
        {
            "id": 703,
            "name": "mft-aedt-n3canary-manual-host",
            "project": controller.NODE_CANARY_HOST_PROJECT,
            "status": "running",
        },
    ]

    result = controller._run_cycle(
        policy, generation, profile, SCHEDULER_URL, state_path
    )

    assert len(scheduler["host_inventory_gets"]) == 1
    assert scheduler["host_cancel_posts"] == [701]
    submit.assert_not_called()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    reconciled_bundle = state["pooled_bundles"][0]
    assert reconciled_bundle["host_cancel_status"] == "requested"
    assert reconciled_bundle["host_cancel_error"] is None
    assert reconciled_bundle["host_cancel_task_id"] == 701
    assert reconciled_bundle["host_cancel_at"]
    visible_bundle = result["pooled_bundles"][0]
    assert visible_bundle["host_cancel_status"] == "requested"
    assert visible_bundle["host_cancel_error"] is None
    assert visible_bundle["host_cancel_task_id"] == 701
    assert visible_bundle["host_cancel_at"]


def test_terminal_host_before_discovery_falls_back_without_cancellation(
    tmp_path, monkeypatch
):
    policy, profile, generation, state_path, scheduler = _pooled_runtime(
        tmp_path, monkeypatch
    )
    submit = mock.Mock(side_effect=[911, 912])
    monkeypatch.setattr(controller.scheduler_client, "submit_verification", submit)

    controller._run_cycle(policy, generation, profile, SCHEDULER_URL, state_path)
    scheduler["host_status"] = "failed"
    result = controller._run_cycle(
        policy, generation, profile, SCHEDULER_URL, state_path
    )

    assert result["accepted_or_reconciled_count"] == 2
    assert result["cancel_task_ids"] == []
    assert len(scheduler["host_posts"]) == 1
    assert scheduler["host_cancel_posts"] == []
    assert all(
        call.kwargs["aedt_backend"] == "standalone"
        for call in submit.call_args_list
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    bundle = state["pooled_bundles"][0]
    assert bundle["phase"] == "complete"
    assert "became terminal before discovery: failed" in bundle["failure_reason"]
    assert bundle["host_terminal_status"] == "failed"
    assert bundle["host_cancel_status"] == "already_terminal"
    assert bundle["host_cancel_error"] is None
    assert bundle["host_cancel_task_id"] == 701
    assert bundle["host_cancel_at"]


@pytest.mark.parametrize("interruption", ["host_failure", "client_rejection"])
def test_partial_client_admission_retries_missing_rows_standalone(
    tmp_path, monkeypatch, interruption
):
    policy, profile, generation, state_path, scheduler = _pooled_runtime(
        tmp_path, monkeypatch
    )
    controller._run_cycle(policy, generation, profile, SCHEDULER_URL, state_path)
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    bundle = persisted["pooled_bundles"][0]
    scheduler["discovery_text"] = "NODE_CANARY_DISCOVERY " + json.dumps(
        {
            "schema_version": 1,
            "mode": "scheduler_managed_node_local_canary",
            "scheduler_url": "http://127.0.0.1:8123",
            "expected_projects": 2,
            "node": "n41",
            "rollback_file": bundle["coordination_files"]["rollback"],
        }
    )

    admitted = []

    def submit_first_client(**kwargs):
        admitted.append(kwargs)
        if len(admitted) == 1:
            if interruption == "host_failure":
                scheduler["host_status"] = "failed"
            return 801
        return None

    monkeypatch.setattr(
        controller.scheduler_client,
        "submit_verification",
        mock.Mock(side_effect=submit_first_client),
    )
    first = controller._run_cycle(
        policy, generation, profile, SCHEDULER_URL, state_path
    )

    assert first["accepted_or_reconciled_count"] == 1
    assert len(admitted) == (1 if interruption == "host_failure" else 2)
    partial = json.loads(state_path.read_text(encoding="utf-8"))
    partial_bundle = partial["pooled_bundles"][0]
    assert partial_bundle["phase"] == "clients_partial_tracked"
    assert partial_bundle["client_task_ids"] == [801, None]
    assert partial_bundle["host_terminal_status"] == (
        "failed" if interruption == "host_failure" else None
    )
    assert (
        "terminal during client admission"
        if interruption == "host_failure"
        else "definitively rejected"
    ) in partial_bundle["failure_reason"]

    scheduler["client_statuses"][801] = "failed"
    fallback_submit = mock.Mock(side_effect=[901, 902])
    monkeypatch.setattr(
        controller.scheduler_client,
        "submit_verification",
        fallback_submit,
    )
    second = controller._run_cycle(
        policy, generation, profile, SCHEDULER_URL, state_path
    )

    assert second["accepted_or_reconciled_count"] == 2
    assert fallback_submit.call_count == 2
    assert all(
        call.kwargs["aedt_backend"] == "standalone"
        and "-sa-retry-" in call.kwargs["name"]
        for call in fallback_submit.call_args_list
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    completed = state["pooled_bundles"][0]
    assert completed["phase"] == "complete"
    assert completed["client_fallback_task_ids"] == [901, 902]
    assert state["reservations"] == []
    assert [(item["serial"], item["backend"]) for item in state["submissions"]] == [
        (1, "pooled"),
        (2, "standalone"),
    ]


def test_rejected_exact_host_placement_falls_back_instead_of_wedging(
    tmp_path, monkeypatch
):
    policy, profile, generation, state_path, scheduler = _pooled_runtime(
        tmp_path, monkeypatch
    )
    scheduler["host_post_status"] = 409
    submit = mock.Mock(side_effect=[921, 922])
    monkeypatch.setattr(controller.scheduler_client, "submit_verification", submit)

    result = controller._run_cycle(
        policy, generation, profile, SCHEDULER_URL, state_path
    )

    assert result["accepted_or_reconciled_count"] == 2
    assert len(scheduler["host_posts"]) == 1
    state = json.loads(state_path.read_text(encoding="utf-8"))
    bundle = state["pooled_bundles"][0]
    assert bundle["phase"] == "complete"
    assert "HTTP 409" in bundle["failure_reason"]
    assert all(
        call.kwargs["aedt_backend"] == "standalone"
        for call in submit.call_args_list
    )


def test_expired_host_submit_intent_falls_back_without_creating_late_host(
    tmp_path, monkeypatch
):
    policy, profile, generation, state_path, scheduler = _pooled_runtime(
        tmp_path, monkeypatch
    )

    def uncertain_post(url, json=None, timeout=None):
        scheduler["host_posts"].append(dict(json))
        raise TimeoutError("controller lost the host POST response")

    monkeypatch.setattr(
        controller.scheduler_client.requests, "post", uncertain_post
    )
    with pytest.raises(TimeoutError, match="lost the host POST"):
        controller._run_cycle(
            policy, generation, profile, SCHEDULER_URL, state_path
        )

    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    bundle = persisted["pooled_bundles"][0]
    assert bundle["phase"] == "host_submit"
    assert bundle["host_task_id"] is None
    assert bundle["discovery_deadline_at"] is not None
    bundle["discovery_deadline_at"] = "2000-01-01T00:00:00Z"
    controller._atomic_save_state(state_path, persisted)

    late_post = mock.Mock(side_effect=AssertionError("late host POST"))
    submit = mock.Mock(side_effect=[931, 932])
    monkeypatch.setattr(controller.scheduler_client.requests, "post", late_post)
    monkeypatch.setattr(controller.scheduler_client, "submit_verification", submit)

    result = controller._run_cycle(
        policy, generation, profile, SCHEDULER_URL, state_path
    )

    late_post.assert_not_called()
    assert result["accepted_or_reconciled_count"] == 2
    assert all(
        call.kwargs["aedt_backend"] == "standalone"
        for call in submit.call_args_list
    )
    resumed = json.loads(state_path.read_text(encoding="utf-8"))
    resumed_bundle = resumed["pooled_bundles"][0]
    assert resumed_bundle["phase"] == "complete"
    assert resumed_bundle["host_task_id"] is None
    assert "deadline elapsed" in resumed_bundle["failure_reason"]


def test_terminal_missing_client_row_retries_as_new_standalone_identity(
    tmp_path, monkeypatch
):
    discovery = "NODE_CANARY_DISCOVERY " + json.dumps(
        {
            "schema_version": 1,
            "mode": "scheduler_managed_node_local_canary",
            "scheduler_url": "http://127.0.0.1:8123",
            "expected_projects": 2,
            "node": "n41",
            "rollback_file": "/tmp/bundle.rollback",
        }
    )
    policy, profile, generation, state_path, scheduler = _pooled_runtime(
        tmp_path, monkeypatch, discovery_text=discovery
    )
    submit = mock.Mock(side_effect=[801, 802, 903])
    monkeypatch.setattr(controller.scheduler_client, "submit_verification", submit)

    controller._run_cycle(policy, generation, profile, SCHEDULER_URL, state_path)
    scheduler["client_statuses"] = {801: "completed", 802: "failed"}
    fetch = mock.Mock(
        return_value=controller.scheduler_client.ResultFetch(
            controller.scheduler_client.RESULT_VALID, {"accepted": True}
        )
    )
    monkeypatch.setattr(controller.scheduler_client, "fetch_result", fetch)

    result = controller._run_cycle(
        policy, generation, profile, SCHEDULER_URL, state_path
    )

    assert result["accepted_or_reconciled_count"] == 1
    assert submit.call_count == 3
    retry = submit.call_args_list[-1].kwargs
    assert retry["aedt_backend"] == "standalone"
    assert retry["name"].endswith("-1")
    assert "-sa-retry-" in retry["name"]
    assert retry["name"] != submit.call_args_list[1].kwargs["name"]
    fetch.assert_called_once()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    bundle = state["pooled_bundles"][0]
    assert bundle["phase"] == "complete"
    assert bundle["client_fallback_task_ids"] == [None, 903]
    assert bundle["missing_candidate_indices"] == [1]
