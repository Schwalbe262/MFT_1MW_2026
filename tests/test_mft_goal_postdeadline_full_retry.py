from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import mft_goal_postdeadline_full_retry as retry


def _synthetic_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> retry.SourceBundle:
    candidate_object = {
        "I1_rated": 1000.0,
        **retry.REQUIRED_PHYSICS_FIELDS,
    }
    candidate = json.dumps(
        candidate_object,
        sort_keys=False,
        separators=(",", ":"),
    ).encode("ascii")
    command = b"".join(
        (
            retry.SOURCE_COMMAND_START,
            b"; WD1=",
            retry.SOURCE_WORKDIR_SLUG.encode("ascii"),
            b"; WD2=",
            retry.SOURCE_WORKDIR_SLUG.encode("ascii"),
            b"; printf '%s' '",
            candidate,
            b"' > cand.json && ",
            retry.SOURCE_SOLVER_INVOCATION,
            b"; RETAIN=",
            b"|".join([retry.SOURCE_RETAIN_TOKEN.encode("ascii")] * 7),
            b"; MARKER=",
            retry.SOURCE_MARKER_JSON,
            b"; CONTEXT_DEDUPE=",
            retry.SOURCE_DEDUPE_KEY.encode("ascii"),
            b"; MARKER_SHA=",
            retry.SOURCE_MARKER_CONTRACT_SHA256.encode("ascii"),
            b"; END",
        )
    )
    monkeypatch.setattr(retry, "EXPECTED_SOURCE_COMMAND_SIZE", len(command))
    monkeypatch.setattr(
        retry,
        "EXPECTED_SOURCE_COMMAND_SHA256",
        hashlib.sha256(command).hexdigest(),
    )
    monkeypatch.setattr(retry, "EXPECTED_CANDIDATE_JSON_SIZE", len(candidate))
    monkeypatch.setattr(
        retry,
        "EXPECTED_CANDIDATE_JSON_SHA256",
        hashlib.sha256(candidate).hexdigest(),
    )
    task = dict(retry.REQUIRED_SOURCE_TASK_FIELDS)
    blobs = {
        "task": b"synthetic-task",
        "task.sh": command + b"\n",
        "stdout": b"",
        "stderr": b"",
    }
    return retry.SourceBundle(
        task=task,
        blobs=blobs,
        command=command,
        command_offset_in_task_sh=0,
    )


def test_byte_allowlist_preserves_candidate_and_all_other_bytes(
    monkeypatch: pytest.MonkeyPatch,
):
    bundle = _synthetic_bundle(monkeypatch)
    transformed, audit, candidate = retry.transform_source_command(bundle.command)

    assert [item["name"] for item in audit] == [
        "workdir_slug",
        "retention_dedupe_key",
        "retained_path_token",
        "retention_marker_contract_sha256",
        "solver_walltime_guard_prefix",
    ]
    assert [item["occurrences"] for item in audit] == [2, 2, 7, 1, 1]
    assert candidate in transformed
    assert transformed.count(retry.NEW_SOLVER_INVOCATION) == 1
    assert transformed.count(retry.SOURCE_SOLVER_INVOCATION) == 1
    assert retry.NEW_DEDUPE_KEY.encode("ascii") in transformed
    assert retry.NEW_RETAIN_TOKEN.encode("ascii") in transformed
    assert retry.SOURCE_WORKDIR_SLUG.encode("ascii") not in transformed


def test_plan_seals_missed_deadline_and_noncanonical_classification(
    monkeypatch: pytest.MonkeyPatch,
):
    bundle = _synthetic_bundle(monkeypatch)
    monkeypatch.setattr(
        retry,
        "fetch_and_validate_source",
        lambda _client: bundle,
    )

    plan = retry.build_plan(object(), created_at_utc="2026-07-26T10:00:00Z")
    payload_metadata = plan["payload"]["payload_json"]

    assert plan["classification"]["original_deadline_missed"] is True
    assert plan["classification"]["production"] is False
    assert plan["classification"]["canonical"] is False
    assert payload_metadata["original_deadline_missed"] is True
    assert payload_metadata["production"] is False
    assert payload_metadata["canonical"] is False
    assert plan["payload"]["timeout_seconds"] == 86_400
    assert (
        plan["execution_budget"]["planned_minimum_success_retention_window_seconds"]
        == 3_600
    )
    assert plan["command_reuse"]["physics_candidate"]["byte_exact_unchanged"] is True
    assert set(plan["source"]["blobs"]) == {
        "task",
        "task.sh",
        "stdout",
        "stderr",
    }
    assert retry.validate_plan_for_submission(plan, bundle) == plan["payload"]


def test_source_blob_hash_drift_is_rejected():
    with pytest.raises(retry.ContractError, match="source task.sh drifted"):
        retry._verify_blob("task.sh", b"one changed byte")


def test_plan_tampering_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
):
    bundle = _synthetic_bundle(monkeypatch)
    monkeypatch.setattr(
        retry,
        "fetch_and_validate_source",
        lambda _client: bundle,
    )
    plan = retry.build_plan(object(), created_at_utc="2026-07-26T10:00:00Z")
    plan["classification"]["canonical"] = True

    with pytest.raises(retry.ContractError, match="seal is invalid"):
        retry.validate_plan_for_submission(plan, bundle)


class _FakeSubmissionClient:
    def __init__(self):
        self.post_count = 0

    def get_json(self, endpoint: str):
        assert endpoint.startswith("/api/tasks?")
        return []

    def post_json_once(self, endpoint: str, payload: dict):
        assert endpoint == "/api/tasks"
        if self.post_count:
            raise AssertionError("test client received more than one POST")
        self.post_count += 1
        return 201, {
            "id": 100_001,
            "name": payload["name"],
            "dedupe_key": payload["dedupe_key"],
            "timeout_seconds": payload["timeout_seconds"],
            "deduped": False,
        }


def test_apply_gate_and_immutable_ledger_allow_only_one_post(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    bundle = _synthetic_bundle(monkeypatch)
    monkeypatch.setattr(
        retry,
        "fetch_and_validate_source",
        lambda _client: bundle,
    )
    plan = retry.build_plan(object(), created_at_utc="2026-07-26T10:00:00Z")
    client = _FakeSubmissionClient()
    ledger = tmp_path / "single-post-ledger.json"

    dry_run = retry.submit_plan(plan, client, apply=False)
    assert dry_run["submitted"] is False
    assert client.post_count == 0
    assert not ledger.exists()

    receipt = retry.submit_plan(
        plan,
        client,
        apply=True,
        attempt_ledger=ledger,
    )
    assert receipt["submitted"] is True
    assert receipt["post_attempts_consumed"] == 1
    assert client.post_count == 1
    assert (
        json.loads(ledger.read_text(encoding="utf-8"))["post_attempt_consumed"] is True
    )

    with pytest.raises(retry.ContractError, match="refusing to overwrite"):
        retry.submit_plan(
            plan,
            client,
            apply=True,
            attempt_ledger=ledger,
        )
    assert client.post_count == 1
