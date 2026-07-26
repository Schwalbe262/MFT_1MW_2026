from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools import mft_goal_official_standard8_collector as collector
from tools import mft_goal_official_standard8_postdeadline as official
from tools import mft_goal_postdeadline_standard_collector as base


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _queued_snapshot() -> dict:
    submission = _read(collector.DEFAULT_SUBMISSION)
    return copy.deepcopy(submission["task_readback"])


def _attached_snapshot() -> dict:
    task = _queued_snapshot()
    task.update(
        {
            "status": "running",
            "state": "running",
            "attached_at": "2026-07-26 11:30:00",
            "started_at": "2026-07-26 11:30:03",
            "assigned_allocation": 14648,
            "allocation_id": 14648,
            "slurm_job_id": "839600",
            "allocation_node_name": "n114",
            "actual_node_name": "n114",
            "placement_contract_satisfied": True,
        }
    )
    return task


def test_fixed_candidate_task_and_output_identity() -> None:
    assert collector.TASK_ID == 96329
    assert collector.EXPECTED_ACCOUNT == "jji0930"
    assert collector.EXPECTED_NODE == "n114"
    assert (
        collector.CANDIDATE_PHYSICS_SHA256
        == "622097dde126e12540a9aad52cb992d880ddb47a7d881f722858f624fd8878f4"
    )
    assert collector.DEFAULT_OUTPUT == (
        collector.RUNTIME_ROOT
        / "authenticated_get_collection_task96329_v1"
    )


def test_real_immutable_contract_reauthenticates() -> None:
    contract = collector.load_contract()
    assert contract["task_id"] == 96329
    assert contract["task_name"] == collector.TASK_NAME
    assert contract["dedupe_key"] == collector.DEDUPE_KEY
    assert contract["candidate_physics_sha256"] == (
        collector.CANDIDATE_PHYSICS_SHA256
    )
    assert contract["node_name"] == "n114"
    assert contract["submission"]["scheduler_post_calls"] == 1
    assert contract["retained"]["artifact_path"].endswith("/symmetric.aedt")


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (collector.DEFAULT_PLAN, collector.PLAN_FILE_SHA256),
        (
            collector.RUNTIME_ROOT / "submission" / "submission_intent.json",
            collector.INTENT_FILE_SHA256,
        ),
        (
            collector.RUNTIME_ROOT / "scheduler_post_attempt.json",
            collector.ATTEMPT_FILE_SHA256,
        ),
        (collector.DEFAULT_SUBMISSION, collector.SUBMISSION_FILE_SHA256),
        (
            collector.RUNTIME_ROOT / "submission" / "final_seal.json",
            collector.FINAL_FILE_SHA256,
        ),
    ],
)
def test_every_durable_source_file_has_fixed_sha(
    path: Path,
    expected: str,
) -> None:
    assert base.sha256_file(path) == expected


def test_queued_strict_opening_is_accepted() -> None:
    task = collector._validate_task_snapshot(  # noqa: SLF001
        _queued_snapshot(),
        label="queued",
    )
    assert task["state"] == "queued"
    assert task["preferred_node_relaxed"] is False
    assert task["actual_node_name"] == ""
    assert task["allocation_id"] is None


def test_attached_task_requires_exact_n114_and_no_relaxation() -> None:
    task = collector._validate_task_snapshot(  # noqa: SLF001
        _attached_snapshot(),
        label="attached",
    )
    assert task["actual_node_name"] == "n114"
    assert task["placement_contract_satisfied"] is True

    wrong_node = _attached_snapshot()
    wrong_node["actual_node_name"] = "n109"
    with pytest.raises(
        collector.Official8CollectionError,
        match="attached strict n114",
    ):
        collector._validate_task_snapshot(wrong_node, label="wrong")  # noqa: SLF001

    relaxed = _attached_snapshot()
    relaxed["preferred_node_relaxed"] = True
    with pytest.raises(
        collector.Official8CollectionError,
        match="strict requested placement",
    ):
        collector._validate_task_snapshot(relaxed, label="relaxed")  # noqa: SLF001


def test_queued_task_cannot_claim_an_unbound_actual_node() -> None:
    task = _queued_snapshot()
    task["actual_node_name"] = "n114"
    with pytest.raises(
        collector.Official8CollectionError,
        match="queued placement unexpectedly bound",
    ):
        collector._validate_task_snapshot(task, label="queued")  # noqa: SLF001


def test_opening_preflight_rejects_relaxation_and_nonzero_slots() -> None:
    plan = _read(collector.DEFAULT_PLAN)
    good = plan["preflight"]
    collector._validate_opening_preflight(good, "good")  # noqa: SLF001

    relaxed = copy.deepcopy(good)
    relaxed["preferred_node_relaxed"] = True
    with pytest.raises(
        collector.Official8CollectionError,
        match="strict opening",
    ):
        collector._validate_opening_preflight(relaxed, "relaxed")  # noqa: SLF001

    available = copy.deepcopy(good)
    available["capacity"]["ready_fit_slots"] = 1
    with pytest.raises(
        collector.Official8CollectionError,
        match="strict opening",
    ):
        collector._validate_opening_preflight(available, "available")  # noqa: SLF001


def test_get_task_uses_only_exact_scheduler_get() -> None:
    calls: list[tuple[str, dict]] = []
    raw = json.dumps(_queued_snapshot()).encode()

    def getter(url: str, **kwargs: object) -> bytes:
        calls.append((url, dict(kwargs)))
        return raw

    contract = {"scheduler_url": collector.SCHEDULER_URL, "task_id": 96329}
    task = collector.get_task(contract, getter=getter)
    assert task["state"] == "queued"
    assert calls == [
        (
            "http://127.0.0.1:8002/api/tasks/96329",
            {"max_bytes": 1024 * 1024, "timeout": 30.0},
        )
    ]


def test_queued_poll_is_observation_only(tmp_path: Path) -> None:
    contract = collector.load_contract()
    raw = json.dumps(_queued_snapshot()).encode()

    def getter(_url: str, **_kwargs: object) -> bytes:
        return raw

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
    assert event["scientific_infeasible_claimed"] is False
    assert Path(event["poll_path"]).is_file()


def test_final_seal_proves_task96329_and_one_post() -> None:
    final = _read(
        collector.RUNTIME_ROOT / "submission" / "final_seal.json"
    )
    assert final["schema_version"] == official.FINAL_SEAL_SCHEMA
    assert final["payload_sha256"] == collector.FINAL_PAYLOAD_SHA256
    assert final["task_id"] == 96329
    assert final["scheduler_post_calls"] == 1
    assert final["immutable_evidence_complete"] is True


def test_collector_source_exposes_no_scheduler_mutation_path() -> None:
    source = Path(collector.__file__).read_text(encoding="utf-8")
    assert "http_post" not in source
    assert "_post_json" not in source
    assert "requests.post" not in source
    assert "scheduler_mutation_performed\": False" in source
    assert "scientific_pass_claimed\": False" in source
