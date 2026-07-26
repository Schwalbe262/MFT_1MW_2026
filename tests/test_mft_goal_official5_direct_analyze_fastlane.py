from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from tools import mft_goal_official5_direct_analyze_fastlane as fast


OBSERVED_AT = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)
SOLVER_REVISION = "d" * 40
TASK_ID = 98_765


class Lock:
    def __enter__(self) -> Lock:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


def _revision_attestation(
    revision: str = SOLVER_REVISION,
) -> dict[str, Any]:
    return fast.sealed(
        {
            "schema_version": fast.REVISION_SCHEMA,
            "solver_revision": revision,
            "committed_source_readback": True,
            "direct_path_present": True,
            "thermal_source": {
                "path": "module/thermal_260706.py",
                "sha256": "1" * 64,
                "size_bytes": 1,
            },
            "runner_source": {
                "path": "run_simulation_260706.py",
                "sha256": "2" * 64,
                "size_bytes": 1,
            },
            "required_env_name": fast.DIRECT_ENV_NAME,
            "required_env_token_sha256": fast.DIRECT_ENV_TOKEN_SHA256,
            "full_model_allowed": False,
        }
    )


def _source_package() -> dict[str, Any]:
    authority = fast.sealed(
        {
            "schema_version": fast.SOURCE_AUTHORITY_SCHEMA,
            **fast.SAFETY_FLAGS,
            "candidate_physics_sha256": fast.CANDIDATE_SHA256,
            "official_standard_selection_order": 5,
            "source_task_id": fast.SOURCE_TASK_ID,
            "source_task_name": fast.SOURCE_TASK_NAME,
            "source_task_dedupe_key": fast.SOURCE_TASK_DEDUPE_KEY,
            "source_reauthenticated": True,
            "source_task_scheduler_post_calls": 1,
            "new_lane_scheduler_post_calls": 0,
        }
    )
    selected = fast.sealed(
        {
            "schema_version": "test-selected-candidate-v1",
            "candidate_physics_sha256": fast.CANDIDATE_SHA256,
        }
    )
    params = {
        "fan_velocity": 1.5,
        "k_ins": 0.2,
        "core_plate_pad_t": 2.0,
        "wcp_pad_t": 2.0,
        "thermal_symmetry": "eighth",
        "full_model": 0,
    }
    profile = {"schema_version": "test-execution-profile-v1"}
    return {
        "authority": authority,
        "selected": selected,
        "params": params,
        "profile": profile,
        "source_plan": {},
    }


def _command(*, direct: bool, revision: str = SOLVER_REVISION) -> str:
    core_auth = fast.standalone_core_auth_sha256(revision)
    prefix = (
        f"git checkout {revision}; "
        f'export MFT_STANDALONE_CORE_AUTH_SHA256="{core_auth}"; '
    )
    if direct:
        return (
            prefix
            + f'export {fast.DIRECT_ENV_NAME}="{fast.DIRECT_ENV_TOKEN}"; '
            + "timeout --signal=TERM --kill-after=300s 43200s "
            + "python run_simulation_260706.py --fixed --thermal --headless "
            + "--symmetry-thermal-direct-analyze --params cand.json; "
            + "simulation_rc=$?;"
        )
    return (
        prefix
        + "timeout --signal=TERM --kill-after=300s 43200s "
        + "python run_simulation_260706.py --fixed --thermal --headless "
        + "--params cand.json; simulation_rc=$?;"
    )


def _direct_bundle(
    _params: dict[str, Any],
    _profile: dict[str, Any],
    revision: str = SOLVER_REVISION,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    dedupe = f"direct-test:{revision}:official5:n115-samenode"
    payload = {
        "name": fast.TASK_NAME,
        "project": fast.PROJECT,
        "account_name": fast.ACCOUNT_NAME,
        "node_name": fast.NODE_NAME,
        "node_name_policy": "strict",
        "same_node_as_task_id": fast.SAME_NODE_AS_TASK_ID,
        "cpus": fast.CPUS,
        "memory_mb": fast.MEMORY_MB,
        "max_workers_per_node": fast.MAX_WORKERS_PER_NODE,
        "timeout_seconds": fast.SCHEDULER_SECONDS,
        "aedt_backend": "standalone",
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "scheduling_profile": "fea_bursty",
        "dedupe_key": dedupe,
        "command": _command(direct=True, revision=revision),
    }
    environment = {
        fast.DIRECT_ENV_NAME: fast.DIRECT_ENV_TOKEN,
        "MFT_STANDALONE_CORE_CONTRACT": (
            fast.STANDALONE_CORE_CONTRACT
        ),
        "MFT_STANDALONE_CORE_COUNT": str(fast.CPUS),
        "MFT_STANDALONE_CORE_AUTH_SHA256": (
            fast.standalone_core_auth_sha256(revision)
        ),
    }
    retained = {
        "dedupe_key": dedupe,
        "solver_revision": revision,
        "library_revision": fast.LIBRARY_REVISION,
        "stage": "standard",
        "artifact_path": f"retained/{revision}/symmetric.aedt",
        "results_path": f"retained/{revision}/symmetric.aedtresults",
        "transport": {
            "chunk_directory": (
                f"retained/{revision}/symmetric.aedt.chunks"
            )
        },
        "retention_required": True,
        "prune_protection_required": True,
    }
    return payload, environment, retained


class GateReader:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.health = {
            "ok": True,
            "scheduler_ok": True,
            "scheduler_thread_alive": True,
            "scheduler_stalled": False,
        }
        self.source_task = {
            "task_id": fast.SOURCE_TASK_ID,
            "name": fast.SOURCE_TASK_NAME,
            "dedupe_key": fast.SOURCE_TASK_DEDUPE_KEY,
            "project": fast.PROJECT,
            "requested_account_name": fast.SOURCE_SPEC.account_name,
            "requested_node_name": fast.SOURCE_SPEC.node_name,
            "requested_node_name_policy": "strict",
            "preferred_node_relaxed": False,
            "status": "running",
            "allocation_id": fast.SOURCE_ALLOCATION_ID,
            "assigned_allocation": fast.SOURCE_ALLOCATION_ID,
            "slurm_job_id": fast.SOURCE_SLURM_JOB_ID,
            "account_name": fast.ACCOUNT_NAME,
            "node_name": fast.NODE_NAME,
            "actual_node_name": fast.NODE_NAME,
            "allocation_node_name": fast.NODE_NAME,
            "same_node_as_task_id": 0,
            "cpus": fast.CPUS,
            "memory_mb": fast.MEMORY_MB,
            "scheduling_profile": "fea_bursty",
            "aedt_backend": "standalone",
        }
        self.storage_audit = {
            "task_id": fast.STORAGE_AUDIT_TASK_ID,
            "name": fast.STORAGE_AUDIT_TASK_NAME,
            "dedupe_key": fast.STORAGE_AUDIT_DEDUPE_KEY,
            "status": "completed",
            "state": "succeeded",
            "exit_code": 0,
            "project": fast.PROJECT,
            "account_name": fast.ACCOUNT_NAME,
            "requested_account_name": fast.ACCOUNT_NAME,
            "allocation_id": fast.SOURCE_ALLOCATION_ID,
            "assigned_allocation": fast.SOURCE_ALLOCATION_ID,
            "slurm_job_id": fast.SOURCE_SLURM_JOB_ID,
            "node_name": fast.NODE_NAME,
            "requested_node_name": fast.NODE_NAME,
            "actual_node_name": fast.NODE_NAME,
            "allocation_node_name": fast.NODE_NAME,
            "same_node_as_task_id": fast.SAME_NODE_AS_TASK_ID,
            "same_node_as_node_name": fast.NODE_NAME,
            "same_node_as_allocation_id": fast.SOURCE_ALLOCATION_ID,
            "requested_node_name_policy": "strict",
            "preferred_node_relaxed": False,
            "finished_at": OBSERVED_AT.isoformat(),
            "stdout": (
                "MFT_LIVE_INVENTORY_V1 "
                "source=/enroot/mft_campaign-test "
                "node=n115 job=840582\n"
                "/dev/nvme0n1 937234648 198246084 738988564 "
                "22% /enroot\n"
            ),
        }
        self.capacity = {
            "queue_state": "ready",
            "queue_reason": (
                "ready to attach to allocation "
                f"{fast.SOURCE_ALLOCATION_ID}"
            ),
            "ready_fit_slots": 7,
            "pending_fit_slots": 0,
            "inflight_fit_slots": 7,
            "memory_pressure_state": "ok",
            "preferred_node_relaxed": False,
            "standalone_aedt_available": 1,
            "allocations": [
                {
                    "allocation_id": fast.SOURCE_ALLOCATION_ID,
                    "account_name": fast.ACCOUNT_NAME,
                    "node_name": fast.NODE_NAME,
                    "fit_slots": 7,
                    "free_cpus": 64,
                    "free_memory_mb": 861_787,
                    "memory_pressure_state": "ok",
                }
            ],
        }
        self.allocations = [
            {
                "id": fast.SOURCE_ALLOCATION_ID,
                "account_name": fast.ACCOUNT_NAME,
                "node_name": fast.NODE_NAME,
                "slurm_job_id": fast.SOURCE_SLURM_JOB_ID,
                "state": "active",
                "resource_pool": "cpu",
                "node_pestat_state": "mix",
                "free_cpus": 64,
                "free_memory_mb": 861_787,
                "node_cpu_total": max(fast.CPUS, 32),
                "node_cpu_used": 0,
                "node_memory_total_mb": max(fast.MEMORY_MB, 200_000),
                "node_memory_free_mb": max(fast.MEMORY_MB, 180_000),
                "node_metrics_observed_at": OBSERVED_AT.isoformat(),
            }
        ]
        self.licenses = {
            "server_up": True,
            "admission": {
                "enabled": True,
                "snapshot_valid": True,
                "snapshot_age_seconds": 1,
                "snapshot_max_age_seconds": 60,
                "blocked_reason": None,
                "features": {
                    fast.LICENSE_FEATURE: {"admit_headroom": 1}
                },
                "reserve_by_feature": {
                    fast.LICENSE_FEATURE: fast.LICENSE_RESERVE_FLOOR
                },
            },
        }
        self.accounts = [
            {
                "account_name": fast.ACCOUNT_NAME,
                "running": 9,
                "pending": 0,
                "max_running": 10,
                "max_pending": 10,
                "max_total": 20,
                "storage_used_gb": 20,
                "storage_quota_gb": 100,
            }
        ]
        self.capabilities = [
            {
                "capability": "conda:pyaedt2026v1",
                "accounts": [fast.ACCOUNT_NAME],
            }
        ]
        self.active: list[dict[str, Any]] = [
            {
                "task_id": fast.SOURCE_TASK_ID,
                "status": "running",
                "node_name": fast.NODE_NAME,
                "aedt_backend": "standalone",
                "scheduling_profile": "fea_bursty",
            }
        ]
        self.inventory: list[dict[str, Any]] = []

    def reader(self, path: str, query: Any) -> Any:
        self.calls.append((path, copy.deepcopy(query)))
        if path == "/api/health":
            return copy.deepcopy(self.health)
        if path == f"/api/tasks/{fast.SOURCE_TASK_ID}":
            return copy.deepcopy(self.source_task)
        if path == f"/api/tasks/{fast.STORAGE_AUDIT_TASK_ID}":
            assert dict(query or []) == {
                "include_output": "true",
                "output_limit": 65536,
            }
            return copy.deepcopy(self.storage_audit)
        if path == "/api/task-capacity":
            assert dict(query or []) == dict(fast.capacity_query())
            return copy.deepcopy(self.capacity)
        if path == "/api/allocations":
            return copy.deepcopy(self.allocations)
        if path == "/api/licenses":
            return copy.deepcopy(self.licenses)
        if path == "/api/accounts/status/live":
            return copy.deepcopy(self.accounts)
        if path == "/api/capabilities":
            return copy.deepcopy(self.capabilities)
        if path == "/api/tasks":
            if "name_prefix" in dict(query or []):
                return copy.deepcopy(self.inventory)
            return copy.deepcopy(self.active)
        raise AssertionError(f"unexpected GET {path} {query}")


def _patch_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    output = tmp_path / "direct-fastlane"
    monkeypatch.setattr(fast, "OUTPUT_ROOT", output)
    return output


def _prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, Any], GateReader]:
    output = _patch_output(tmp_path, monkeypatch)
    scheduler = GateReader()
    plan_path = fast.prepare(
        solver_revision=SOLVER_REVISION,
        output=output,
        source_authenticator=_source_package,
        revision_attester=lambda revision: _revision_attestation(revision),
        payload_builder=_direct_bundle,
        reader=scheduler.reader,
        observed_at=OBSERVED_AT,
    )
    plan = fast.validate_seal(
        fast.read_json(plan_path), fast.PLAN_SCHEMA
    )
    return plan_path, plan, scheduler


def _task_readback(
    payload: dict[str, Any],
    *,
    task_id: int = TASK_ID,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "name": fast.TASK_NAME,
        "dedupe_key": payload["dedupe_key"],
        "project": fast.PROJECT,
        "requested_account_name": fast.ACCOUNT_NAME,
        "requested_node_name": fast.NODE_NAME,
        "requested_node_name_policy": "strict",
        "preferred_node_relaxed": False,
        "same_node_as_task_id": fast.SAME_NODE_AS_TASK_ID,
        "cpus": fast.CPUS,
        "memory_mb": fast.MEMORY_MB,
        "timeout_seconds": fast.SCHEDULER_SECONDS,
        "max_workers_per_node": fast.MAX_WORKERS_PER_NODE,
        "aedt_backend": "standalone",
        "status": "queued",
        "account_name": "",
        "node_name": "",
    }


def test_revision_attestation_requires_exact_committed_direct_path() -> None:
    thermal = "\n".join(
        (
            fast.DIRECT_ENV_NAME,
            fast.DIRECT_ENV_TOKEN,
            "_symmetry_thermal_direct_analyze_request",
            "_symmetry_thermal_direct_analyze_preflight",
            "generate_mesh_call_policy",
            "forbidden_before_analyze",
        )
    ).encode()
    runner = "\n".join(
        (
            "--symmetry-thermal-direct-analyze",
            "SYMMETRY_THERMAL_DIRECT_ANALYZE_ENV",
            "SYMMETRY_THERMAL_DIRECT_ANALYZE_TOKEN",
        )
    ).encode()

    def reader(path: str, revision: str) -> bytes:
        assert revision == SOLVER_REVISION
        return thermal if path.startswith("module/") else runner

    attestation = fast.attest_solver_revision(
        SOLVER_REVISION, source_reader=reader
    )
    fast.validate_seal(attestation, fast.REVISION_SCHEMA)
    assert attestation["direct_path_present"] is True
    assert attestation["full_model_allowed"] is False
    assert attestation["thermal_source"]["sha256"] == fast.hashlib.sha256(
        thermal
    ).hexdigest()

    with pytest.raises(
        fast.PostdeadlineContractError, match="lowercase 40-hex"
    ):
        fast.attest_solver_revision("D" * 40, source_reader=reader)
    with pytest.raises(
        fast.PostdeadlineContractError, match="lacks the committed direct"
    ):
        fast.attest_solver_revision(
            SOLVER_REVISION,
            source_reader=lambda path, _revision: (
                thermal if path.startswith("module/") else b"old runner"
            ),
        )


def test_payload_derivation_adds_exact_opt_in_and_keeps_standard_symmetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = _source_package()["params"]

    def base_builder(
        _params: dict[str, Any], _profile: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        payload, environment, retained = _direct_bundle(
            _params, _profile
        )
        payload["command"] = _command(direct=False)
        environment.pop(fast.DIRECT_ENV_NAME)
        return payload, environment, retained

    monkeypatch.setattr(
        fast.reviewed,
        "validate_scheduler_payload",
        lambda _payload, _retained: None,
    )
    payload, environment, retained = fast.derive_direct_payload(
        params, {}, SOLVER_REVISION, base_builder=base_builder
    )
    fast.validate_direct_payload(
        payload,
        environment,
        retained,
        params=params,
        solver_revision=SOLVER_REVISION,
    )
    assert payload["command"].count(
        "--symmetry-thermal-direct-analyze"
    ) == 1
    assert "--full" not in payload["command"]
    assert retained["stage"] == "standard"
    assert retained["artifact_path"].endswith("/symmetric.aedt")


def test_live_gate_is_get_only_and_requires_exact_same_node_state() -> None:
    scheduler = GateReader()
    gate = fast.live_gate(
        expected_dedupe_key="new-unique-dedupe",
        reader=scheduler.reader,
        observed_at=OBSERVED_AT,
    )
    fast.validate_seal(gate, fast.LIVE_GATE_SCHEMA)
    assert gate["scheduler_get_only"] is True
    assert gate["scheduler_post_calls"] == 0
    assert gate["same_node_ready_only"] is True
    assert gate["same_node_as_task_id"] == fast.SOURCE_TASK_ID
    assert gate["source_allocation_id"] == fast.SOURCE_ALLOCATION_ID
    assert gate["active_target_node_fea_count"] == 1
    assert gate["active_target_node_fea_task_ids"] == [
        fast.SOURCE_TASK_ID
    ]
    assert gate["account_cap_and_storage_gate"]["storage"][
        "mode"
    ] == "declared_quota"
    assert [path for path, _query in scheduler.calls].count(
        "/api/tasks"
    ) == 2


@pytest.mark.parametrize(
    "failure",
    (
        "account_cap",
        "storage_audit",
        "license",
        "node_memory",
        "extra_active_n115",
        "queue_reason",
    ),
)
def test_live_gate_fails_closed_when_any_admission_evidence_drifts(
    failure: str,
) -> None:
    scheduler = GateReader()
    if failure == "account_cap":
        scheduler.accounts[0]["running"] = scheduler.accounts[0][
            "max_running"
        ]
    elif failure == "storage_audit":
        scheduler.accounts[0].pop("storage_used_gb")
        scheduler.accounts[0].pop("storage_quota_gb")
        scheduler.accounts[0]["storage_path"] = "slurm_scheduler"
        scheduler.storage_audit["stdout"] = "invalid"
    elif failure == "license":
        scheduler.licenses["admission"]["features"][
            fast.LICENSE_FEATURE
        ]["admit_headroom"] = 0
    elif failure == "node_memory":
        scheduler.allocations[0]["node_memory_free_mb"] = (
            fast.MEMORY_MB - 1
        )
    elif failure == "extra_active_n115":
        scheduler.active.append(
            {
                "task_id": 1,
                "status": "running",
                "node_name": fast.NODE_NAME,
                "aedt_backend": "standalone",
            }
        )
    elif failure == "queue_reason":
        scheduler.capacity["queue_reason"] = "different"
    with pytest.raises(fast.PostdeadlineContractError):
        fast.live_gate(
            expected_dedupe_key="new-unique-dedupe",
            reader=scheduler.reader,
            observed_at=OBSERVED_AT,
        )


def test_prepare_seals_diagnostic_standard_plan_without_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, plan, scheduler = _prepare(tmp_path, monkeypatch)
    assert plan_path.is_file()
    assert plan["diagnostic_only"] is True
    assert plan["noncanonical"] is True
    assert plan["original_deadline_missed"] is True
    assert plan["scheduler_post_calls"] == 0
    assert plan["scheduler_submission_performed"] is False
    assert plan["standard_only"] is True
    assert plan["symmetric_model"] is True
    assert plan["same_node_fast_lane"] is True
    assert (
        plan["placement"]["same_node_as_task_id"]
        == fast.SAME_NODE_AS_TASK_ID
    )
    assert (
        plan["scheduler_payload"]["same_node_as_task_id"]
        == fast.SAME_NODE_AS_TASK_ID
    )
    assert plan["full_model"] is False
    assert plan["thermal_symmetry"] == "eighth"
    assert plan["fixed_boundary"] == fast.FIXED_BOUNDARY
    assert plan["direct_analyze_opt_in"] == {
        "env_name": fast.DIRECT_ENV_NAME,
        "env_value": fast.DIRECT_ENV_TOKEN,
        "env_value_sha256": fast.DIRECT_ENV_TOKEN_SHA256,
        "cli_flag": "--symmetry-thermal-direct-analyze",
        "exact_opt_in_required": True,
        "standalone_core_contract": fast.STANDALONE_CORE_CONTRACT,
        "standalone_core_count": fast.CPUS,
        "standalone_core_auth_sha256": (
            fast.standalone_core_auth_sha256(SOLVER_REVISION)
        ),
    }
    assert plan["single_attempt_contract"]["post_call_budget"] == 1
    assert plan["single_attempt_contract"][
        "attempt_consumed_before_network"
    ] is True
    assert plan["retained_aedt_bundle"]["artifact_path"].endswith(
        "/symmetric.aedt"
    )
    assert scheduler.calls
    assert not (plan_path.parent / fast.ATTEMPT_NAME).exists()


def test_wrong_authorization_rejects_before_path_lookup_or_post(
    tmp_path: Path,
) -> None:
    posts = 0

    def poster(*_args: Any) -> Any:
        nonlocal posts
        posts += 1
        raise AssertionError("POST must not run")

    with pytest.raises(
        fast.PostdeadlineContractError, match="authorization is absent"
    ):
        fast.submit(
            authorize_post="wrong",
            plan_path=tmp_path / "missing-plan.json",
            poster=poster,
        )
    assert posts == 0


def test_submit_writes_intent_and_attempt_before_only_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, plan, _scheduler = _prepare(tmp_path, monkeypatch)
    monkeypatch.setattr(
        fast,
        "live_gate",
        lambda **_kwargs: fast.sealed(
            {
                "schema_version": fast.LIVE_GATE_SCHEMA,
                "scheduler_get_only": True,
                "scheduler_post_calls": 0,
            }
        ),
    )
    posts = 0
    task = _task_readback(plan["scheduler_payload"])

    def poster(_url: str, payload: dict[str, Any]) -> Any:
        nonlocal posts
        posts += 1
        assert payload == plan["scheduler_payload"]
        assert (
            fast.OUTPUT_ROOT
            / fast.SUBMISSION_DIRECTORY_NAME
            / fast.INTENT_NAME
        ).is_file()
        assert (fast.OUTPUT_ROOT / fast.ATTEMPT_NAME).is_file()
        return 201, {"task_id": TASK_ID}, None

    def reader(path: str, query: Any) -> Any:
        if path == "/api/tasks":
            assert dict(query or [])["name_prefix"] == fast.TASK_NAME
            return [task]
        if path == f"/api/tasks/{TASK_ID}":
            return task
        raise AssertionError(f"unexpected GET {path}")

    final_path = fast.submit(
        authorize_post=fast.POST_AUTHORIZATION,
        plan_path=plan_path,
        reader=reader,
        poster=poster,
        observed_at=OBSERVED_AT,
        lock_factory=Lock,
        plan_loader=lambda _path: plan,
    )
    assert posts == 1
    final = fast.validate_seal(
        fast.read_json(final_path), fast.FINAL_SCHEMA
    )
    receipt = fast.validate_seal(
        fast.read_json(final_path.parent / fast.RECEIPT_NAME),
        fast.RECEIPT_SCHEMA,
    )
    attempt = fast.validate_seal(
        fast.read_json(fast.OUTPUT_ROOT / fast.ATTEMPT_NAME),
        fast.ATTEMPT_SCHEMA,
    )
    assert final["full_model_started"] is False
    assert receipt["scheduler_post_calls"] == 1
    assert receipt["exact_get_reconciled"] is False
    assert attempt["post_call_consumed_before_network"] is True

    assert (
        fast.submit(
            authorize_post=fast.POST_AUTHORIZATION,
            plan_path=plan_path,
            reader=reader,
            poster=poster,
            observed_at=OBSERVED_AT,
            lock_factory=Lock,
            plan_loader=lambda _path: plan,
        )
        == final_path
    )
    assert posts == 1


def test_ambiguous_post_consumes_budget_and_second_call_never_posts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, plan, _scheduler = _prepare(tmp_path, monkeypatch)
    monkeypatch.setattr(
        fast,
        "live_gate",
        lambda **_kwargs: fast.sealed(
            {
                "schema_version": fast.LIVE_GATE_SCHEMA,
                "scheduler_get_only": True,
                "scheduler_post_calls": 0,
            }
        ),
    )
    posts = 0

    def ambiguous(_url: str, _payload: dict[str, Any]) -> Any:
        nonlocal posts
        posts += 1
        return None, None, "connection reset"

    def reader(path: str, _query: Any) -> Any:
        if path == "/api/tasks":
            return []
        raise AssertionError(f"unexpected GET {path}")

    kwargs = {
        "authorize_post": fast.POST_AUTHORIZATION,
        "plan_path": plan_path,
        "reader": reader,
        "poster": ambiguous,
        "observed_at": OBSERVED_AT,
        "lock_factory": Lock,
        "plan_loader": lambda _path: plan,
    }
    with pytest.raises(
        fast.PostdeadlineContractError, match="no retry"
    ):
        fast.submit(**kwargs)
    assert posts == 1
    assert (fast.OUTPUT_ROOT / fast.ATTEMPT_NAME).is_file()
    assert (
        fast.OUTPUT_ROOT
        / fast.SUBMISSION_DIRECTORY_NAME
        / fast.AMBIGUOUS_NAME
    ).is_file()

    with pytest.raises(
        fast.PostdeadlineContractError, match="already consumed"
    ):
        fast.submit(**kwargs)
    assert posts == 1


def test_exact_get_reconciliation_accepts_only_strict_readback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, plan, _scheduler = _prepare(tmp_path, monkeypatch)
    monkeypatch.setattr(
        fast,
        "live_gate",
        lambda **_kwargs: fast.sealed(
            {
                "schema_version": fast.LIVE_GATE_SCHEMA,
                "scheduler_get_only": True,
                "scheduler_post_calls": 0,
            }
        ),
    )
    task = _task_readback(plan["scheduler_payload"])

    def reader(path: str, _query: Any) -> Any:
        if path == "/api/tasks":
            return [task]
        if path == f"/api/tasks/{TASK_ID}":
            return task
        raise AssertionError(f"unexpected GET {path}")

    final_path = fast.submit(
        authorize_post=fast.POST_AUTHORIZATION,
        plan_path=plan_path,
        reader=reader,
        poster=lambda _url, _payload: (
            None,
            None,
            "response lost after create",
        ),
        observed_at=OBSERVED_AT,
        lock_factory=Lock,
        plan_loader=lambda _path: plan,
    )
    receipt = fast.validate_seal(
        fast.read_json(final_path.parent / fast.RECEIPT_NAME),
        fast.RECEIPT_SCHEMA,
    )
    assert receipt["exact_get_reconciled"] is True

    drift = copy.deepcopy(task)
    drift["preferred_node_relaxed"] = True
    with pytest.raises(
        fast.PostdeadlineContractError, match="readback drifted"
    ):
        fast._authenticated_readback(
            lambda _path, _query: drift,
            task_id=TASK_ID,
            dedupe_key=plan["dedupe_key"],
        )
