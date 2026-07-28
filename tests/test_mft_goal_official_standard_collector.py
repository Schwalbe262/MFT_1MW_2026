from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import pytest

from tools import mft_goal_official_standard_collector as collector
from tools import mft_goal_official_standard_postdeadline as official
from tools import mft_goal_postdeadline_standard_collector as base


def _task(*, state: str = "running") -> dict[str, object]:
    return {
        "task_id": collector.TASK_ID,
        "id": collector.TASK_ID,
        "name": collector.TASK_NAME,
        "dedupe_key": collector.DEDUPE_KEY,
        "project": base.SCHEDULER_PROJECT,
        "requested_account_name": collector.EXPECTED_ACCOUNT,
        "requested_node_name": collector.EXPECTED_NODE,
        "requested_node_name_policy": "strict",
        "same_node_as_task_id": 0,
        "cpus": 8,
        "memory_mb": 98_304,
        "timeout_seconds": 45_300,
        "max_workers_per_node": 1,
        "aedt_backend": "standalone",
        "account_name": collector.EXPECTED_ACCOUNT,
        "actual_node_name": collector.EXPECTED_NODE,
        "allocation_id": collector.EXPECTED_ALLOCATION_ID,
        "assigned_allocation": collector.EXPECTED_ALLOCATION_ID,
        "slurm_job_id": collector.EXPECTED_SLURM_JOB_ID,
        "status": state,
        "state": state,
    }


def test_live_official_contract_reauthenticates_exact_files() -> None:
    contract = collector.load_contract(
        plan_path=collector.DEFAULT_PLAN,
        submission_path=collector.DEFAULT_SUBMISSION,
    )
    assert contract["task_id"] == collector.TASK_ID
    assert contract["candidate_physics_sha256"] == (
        collector.CANDIDATE_PHYSICS_SHA256
    )
    assert contract["parameter_digest"] == "b713a9235ad804af"
    assert contract["node_name"] == collector.EXPECTED_NODE
    assert contract["retained"]["root"] == "goal-fea-retained/a05448710be08cc2"


def test_historical_scheduler_payload_compatibility_is_exact() -> None:
    plan = json.loads(collector.DEFAULT_PLAN.read_text("utf-8"))
    _, params = official.authenticate_official_candidate()
    current, _environment, _retained = official._capture_scheduler_payload(
        params, official.reviewed_profile()
    )

    assert official._scheduler_payload_matches_reviewed_plan(plan, current)

    drifted = copy.deepcopy(current)
    drifted["command"] += " true;"
    assert not official._scheduler_payload_matches_reviewed_plan(
        plan, drifted
    )

    unsealed = copy.deepcopy(plan)
    unsealed["scheduler_payload_sha256"] = "0" * 64
    assert not official._scheduler_payload_matches_reviewed_plan(
        unsealed, current
    )


def test_live_task_get_requires_exact_bound_slurm_identity() -> None:
    contract = collector.load_contract(
        plan_path=collector.DEFAULT_PLAN,
        submission_path=collector.DEFAULT_SUBMISSION,
    )
    task = _task()

    def getter(*_args: object, **_kwargs: object) -> bytes:
        return base.canonical_bytes(task)

    assert collector.get_task(contract, getter=getter)["task_id"] == 96_328
    drifted = copy.deepcopy(task)
    drifted["slurm_job_id"] = "wrong"

    def bad_getter(*_args: object, **_kwargs: object) -> bytes:
        return base.canonical_bytes(drifted)

    with pytest.raises(
        collector.OfficialCollectionError, match="allocation/Slurm"
    ):
        collector.get_task(contract, getter=bad_getter)


def test_active_poll_is_get_only_and_has_no_scientific_claim(
    tmp_path: Path,
) -> None:
    contract = collector.load_contract(
        plan_path=collector.DEFAULT_PLAN,
        submission_path=collector.DEFAULT_SUBMISSION,
    )
    task = _task()

    def getter(*_args: object, **_kwargs: object) -> bytes:
        return base.canonical_bytes(task)

    terminal, event = collector.poll_once(
        contract=contract,
        output=tmp_path / "collection",
        getter=getter,
    )
    assert terminal is False
    assert event["event"] == "active"
    assert event["scheduler_get_only"] is True
    assert event["scheduler_mutation_performed"] is False
    assert event["scientific_pass_claimed"] is False
    assert (
        tmp_path / ".collection.watch" / "latest_poll.json"
    ).is_file()


def test_terminal_failure_is_scheduler_only(
    tmp_path: Path,
) -> None:
    contract = collector.load_contract(
        plan_path=collector.DEFAULT_PLAN,
        submission_path=collector.DEFAULT_SUBMISSION,
    )
    task = _task(state="failed")

    def getter(*_args: object, **_kwargs: object) -> bytes:
        return base.canonical_bytes(task)

    terminal, event = collector.poll_once(
        contract=contract,
        output=tmp_path / "collection",
        getter=getter,
    )
    assert terminal is True
    assert event["event"] == "failure_ledger"
    ledger = json.loads(
        (tmp_path / "collection.failure_ledger.json").read_text("utf-8")
    )
    assert ledger["scientific_pass_claimed"] is False
    assert ledger["scientific_infeasible_claimed"] is False
    assert ledger["scheduler_mutation_performed"] is False


def test_fixed_task_identity_cannot_be_overridden() -> None:
    with pytest.raises(
        collector.OfficialCollectionError, match="fixed official"
    ):
        collector.load_contract(
            plan_path=collector.DEFAULT_PLAN,
            submission_path=collector.DEFAULT_SUBMISSION,
            expected_task_id=collector.TASK_ID + 1,
        )


def test_submission_seal_drift_is_rejected(tmp_path: Path) -> None:
    submission = json.loads(
        collector.DEFAULT_SUBMISSION.read_text("utf-8")
    )
    submission["node_name"] = "n999"
    path = tmp_path / "submission.json"
    path.write_bytes(base.canonical_bytes(submission) + b"\n")
    with pytest.raises(
        collector.OfficialCollectionError, match="file SHA"
    ):
        collector.load_contract(
            plan_path=collector.DEFAULT_PLAN,
            submission_path=path,
        )


def test_plan_safety_and_fixed_boundary_validators_fail_closed() -> None:
    plan = json.loads(collector.DEFAULT_PLAN.read_text("utf-8"))
    params = json.loads(
        (collector.DEFAULT_PLAN.parent / "fea_params.json").read_text("utf-8")
    )
    profile = json.loads(
        (collector.DEFAULT_PLAN.parent / "execution_profile.json").read_text(
            "utf-8"
        )
    )
    plan["fixed_boundary"]["fan_velocity_m_s"] = 2.0
    unsigned = dict(plan)
    unsigned.pop("payload_sha256")
    plan["payload_sha256"] = base.payload_sha256(unsigned)
    with pytest.raises(
        collector.OfficialCollectionError, match="fixed identity"
    ):
        collector._validate_plan(  # noqa: SLF001
            plan,
            params,
            profile,
            expected_candidate_sha256=collector.CANDIDATE_PHYSICS_SHA256,
            expected_task_name=collector.TASK_NAME,
            expected_dedupe_key=collector.DEDUPE_KEY,
            scheduler_url=collector.SCHEDULER_URL,
        )


def test_official_schema_constants_are_bound() -> None:
    assert official.PLAN_SCHEMA == (
        "mft-goal-official-standard-postdeadline-plan-v1"
    )
    assert official.SUBMISSION_SCHEMA == (
        "mft-goal-official-standard-postdeadline-submission-v1"
    )
    assert collector.PLAN_PAYLOAD_SHA256 == json.loads(
        collector.DEFAULT_PLAN.read_text("utf-8")
    )["payload_sha256"]
    assert collector.SUBMISSION_PAYLOAD_SHA256 == json.loads(
        collector.DEFAULT_SUBMISSION.read_text("utf-8")
    )["payload_sha256"]


def test_source_exposes_no_scheduler_mutation_transport() -> None:
    source = inspect.getsource(collector)
    assert 'method="POST"' not in source
    assert "requests.post" not in source
    assert "/api/tasks/{contract['task_id']}" in source
    assert "remote-file" not in source  # delegated only to reviewed GET collector
