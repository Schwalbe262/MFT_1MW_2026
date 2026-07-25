from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import copy
import os
from pathlib import Path
import shutil

import pytest

from tools import mft_campaign_atomic_claim as claims


CAMPAIGN_ID = "mft-goal-20260726"
CAMPAIGN_AUTHORITY_SHA = "a" * 64
CANDIDATE_SHA = "b" * 64
LOGICAL_TASK_ID = 96214
RETRY_GENERATION = "operational-pressure-r1"
PENDING_TIME = "2026-07-25T05:30:00Z"
FINAL_TIME = "2026-07-25T05:31:00Z"


def _winner(
    *,
    immediate_task_id: int = LOGICAL_TASK_ID,
    immediate_retry_kind: str = "none",
) -> dict:
    return {
        "immediate_task_id": immediate_task_id,
        "immediate_retry_kind": immediate_retry_kind,
        "plan_payload_sha256": "c" * 64,
        "plan_file_sha256": "d" * 64,
        "profile_sha256": "e" * 64,
        "resources": {
            "cpus": 8,
            "memory_mb": 32768,
            "timeout_seconds": (28800 if immediate_retry_kind == "timeout" else 14400),
        },
        "task_name": (
            "mft-goal-diag-standard-pressure-r1-"
            f"t{LOGICAL_TASK_ID}-{CANDIDATE_SHA[:12]}"
        ),
        "dedupe_key": (
            "mft-goal-operational-pressure:" f"{LOGICAL_TASK_ID}:{CANDIDATE_SHA}"
        ),
    }


def _task(winner: dict, task_id: int = 99001) -> dict:
    return {
        "task_id": task_id,
        "id": task_id,
        "name": winner["task_name"],
        "dedupe_key": winner["dedupe_key"],
        "project": "MFT_1MW_2026v1",
        "status": "queued",
        "state": "queued",
        "cpus": winner["resources"]["cpus"],
        "memory_mb": winner["resources"]["memory_mb"],
        "timeout_seconds": winner["resources"]["timeout_seconds"],
    }


def _sibling_snapshot(task: dict) -> dict:
    return {
        "schema_version": "test-sibling-snapshot-v1",
        "matching_task_count": 1,
        "matching_tasks": [
            {
                "task_id": task["task_id"],
                "name": task["name"],
                "dedupe_key": task["dedupe_key"],
            }
        ],
    }


def _strict_task_validator(task: dict, pending: dict) -> dict:
    if (
        task.get("project") != "MFT_1MW_2026v1"
        or task.get("cpus") != pending["winner"]["resources"]["cpus"]
        or task.get("memory_mb") != pending["winner"]["resources"]["memory_mb"]
        or task.get("timeout_seconds")
        != pending["winner"]["resources"]["timeout_seconds"]
    ):
        raise claims.ClaimContractError("test task evidence drifted")
    return task


def _process_acquire(
    root: str,
    reference: dict,
    winner: dict,
    nonce: str,
) -> tuple[str, str]:
    try:
        result = claims.acquire_claim(
            Path(root),
            reference,
            winner,
            nonce=nonce,
            now=PENDING_TIME,
            wait_seconds=3.0,
        )
    except Exception as exc:  # pragma: no cover - returned to parent process
        return ("error", f"{type(exc).__name__}: {exc}")
    return (result["status"], result["claim"]["winner_sha256"])


def _process_acquire_args(
    arguments: tuple[str, dict, dict, str],
) -> tuple[str, str]:
    return _process_acquire(*arguments)


def _authority_and_reference(tmp_path: Path) -> tuple[Path, dict, dict]:
    root = tmp_path / "campaign-claims"
    authority = claims.initialize_claim_root(
        root,
        campaign_id=CAMPAIGN_ID,
        campaign_authority_sha256=CAMPAIGN_AUTHORITY_SHA,
        root_id="1" * 32,
        now="2026-07-25T05:00:00Z",
    )
    reference = claims.build_claim_reference(
        authority,
        candidate_physics_sha256=CANDIDATE_SHA,
        logical_authority_task_id=LOGICAL_TASK_ID,
        retry_generation=RETRY_GENERATION,
    )
    return root, authority, reference


def test_root_authority_is_frozen_and_rejects_alternate_paths(tmp_path, monkeypatch):
    root, authority, reference = _authority_and_reference(tmp_path)

    assert (
        claims.initialize_claim_root(
            root,
            campaign_id=CAMPAIGN_ID,
            campaign_authority_sha256=CAMPAIGN_AUTHORITY_SHA,
        )
        == authority
    )
    assert claims.load_claim_root(root, expected_authority=authority) == authority
    assert claims.validate_claim_reference(reference, authority) == reference
    assert reference["claim_key"].startswith(f"{RETRY_GENERATION}-t{LOGICAL_TASK_ID}-")

    with pytest.raises(claims.ClaimContractError, match="absolute canonical"):
        claims.load_claim_root(Path(root.name))
    with pytest.raises(claims.ClaimContractError, match="different authority"):
        claims.initialize_claim_root(
            root,
            campaign_id=CAMPAIGN_ID,
            campaign_authority_sha256=CAMPAIGN_AUTHORITY_SHA,
            root_id="2" * 32,
        )

    copied = tmp_path / "copied-campaign-claims"
    shutil.copytree(root, copied)
    with pytest.raises(claims.ClaimContractError, match="physical identity drifted"):
        claims.load_claim_root(copied)
    with pytest.raises(claims.ClaimContractError):
        claims.acquire_claim(
            copied,
            reference,
            _winner(),
            nonce="3" * 32,
            now=PENDING_TIME,
        )

    original_is_reparse = claims._is_reparse

    def forced_reparse(path: Path) -> bool:
        return os.path.normcase(str(path)) == os.path.normcase(
            str(root.resolve())
        ) or original_is_reparse(path)

    monkeypatch.setattr(claims, "_is_reparse", forced_reparse)
    with pytest.raises(claims.ClaimContractError, match="reparse point"):
        claims.load_claim_root(root)


def test_atomic_process_claim_has_one_winner_and_one_logical_identity(
    tmp_path,
):
    root, _authority, reference = _authority_and_reference(tmp_path)
    winner = _winner()
    arguments = [
        (str(root), reference, winner, f"{index:032x}") for index in range(1, 7)
    ]
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_process_acquire_args, arguments))

    statuses = [status for status, _digest in results]
    assert "error" not in statuses, results
    assert statuses.count("fresh_pending") == 1
    assert statuses.count("existing_pending") == len(arguments) - 1
    assert len({digest for _status, digest in results}) == 1

    pending = claims.validate_pending_claim(root, reference, expected_winner=winner)
    assert pending["winner"]["immediate_task_id"] == LOGICAL_TASK_ID

    timeout_sibling = _winner(
        immediate_task_id=96275,
        immediate_retry_kind="timeout",
    )
    with pytest.raises(claims.ClaimContractError, match="different ancestry"):
        claims.acquire_claim(
            root,
            reference,
            timeout_sibling,
            nonce="f" * 32,
            now=PENDING_TIME,
        )

    # Placement is intentionally absent from the claim winner.  Strict node
    # variants therefore cannot create another logical retry identity.
    assert "node_name" not in pending["winner"]
    assert "requested_node_name" not in pending["winner"]


def test_partial_claim_directory_fails_closed_without_repost(tmp_path):
    root, _authority, reference = _authority_and_reference(tmp_path)
    claim_directory = root / Path(reference["relative_claim_directory"])
    os.mkdir(claim_directory)

    with pytest.raises(claims.ClaimContractError, match="manual recovery"):
        claims.acquire_claim(
            root,
            reference,
            _winner(),
            nonce="4" * 32,
            now=PENDING_TIME,
            wait_seconds=0,
        )
    assert not (claim_directory / claims.PENDING_CLAIM_NAME).exists()
    assert not (claim_directory / claims.FINALIZED_CLAIM_NAME).exists()


def test_crash_before_post_stays_pending_and_zero_or_many_fail_closed(
    tmp_path,
):
    root, _authority, reference = _authority_and_reference(tmp_path)
    winner = _winner()
    acquired = claims.acquire_claim(
        root,
        reference,
        winner,
        nonce="5" * 32,
        now=PENDING_TIME,
    )
    pending = acquired["claim"]
    task = _task(winner)

    for rows in ([], [task, {**task, "task_id": 99002, "id": 99002}]):
        with pytest.raises(claims.ClaimContractError, match="exactly one"):
            claims.recover_pending_claim(
                root,
                reference,
                pending,
                matching_tasks=rows,
                sibling_snapshot=_sibling_snapshot(task),
                evidence_validator=_strict_task_validator,
                now=FINAL_TIME,
            )

    claim_directory = root / Path(reference["relative_claim_directory"])
    assert not (claim_directory / claims.FINALIZED_CLAIM_NAME).exists()
    assert (
        claims.acquire_claim(
            root,
            reference,
            winner,
            nonce="6" * 32,
            now=PENDING_TIME,
        )["status"]
        == "existing_pending"
    )


def test_crash_after_post_recovers_exact_task_and_finalizes_immutably(
    tmp_path,
):
    root, _authority, reference = _authority_and_reference(tmp_path)
    winner = _winner(
        immediate_task_id=96275,
        immediate_retry_kind="timeout",
    )
    pending = claims.acquire_claim(
        root,
        reference,
        winner,
        nonce="7" * 32,
        now=PENDING_TIME,
    )["claim"]
    task = _task(winner)
    siblings = _sibling_snapshot(task)

    finalized = claims.recover_pending_claim(
        root,
        reference,
        pending,
        matching_tasks=[task],
        sibling_snapshot=siblings,
        evidence_validator=_strict_task_validator,
        now=FINAL_TIME,
    )
    assert finalized["state"] == "finalized"
    assert finalized["task_id"] == task["task_id"]
    assert finalized["pending_claim"] == pending
    assert (
        claims.validate_finalized_claim(
            root,
            reference,
            claim=finalized,
            expected_winner=winner,
        )
        == finalized
    )
    replay = claims.acquire_claim(
        root,
        reference,
        winner,
        nonce="8" * 32,
        now=PENDING_TIME,
    )
    assert replay == {
        "status": "existing_finalized",
        "claim": finalized,
    }

    with pytest.raises(claims.ClaimContractError, match="different ancestry"):
        claims.acquire_claim(
            root,
            reference,
            _winner(),
            nonce="9" * 32,
            now=PENDING_TIME,
        )
    forged = copy.deepcopy(finalized)
    forged["task_id"] += 1
    with pytest.raises(claims.ClaimContractError, match="seal mismatch"):
        claims.validate_finalized_claim(
            root,
            reference,
            claim=forged,
            expected_winner=winner,
        )


def test_finalization_rejects_wrong_task_sibling_and_validator_evidence(
    tmp_path,
):
    root, _authority, reference = _authority_and_reference(tmp_path)
    winner = _winner()
    pending = claims.acquire_claim(
        root,
        reference,
        winner,
        nonce="a" * 32,
        now=PENDING_TIME,
    )["claim"]
    task = _task(winner)

    with pytest.raises(
        claims.ClaimContractError, match="canonical submission identity"
    ):
        claims.finalize_claim(
            root,
            reference,
            pending,
            task_id=task["task_id"],
            task_readback={**task, "name": "wrong-task"},
            sibling_snapshot=_sibling_snapshot(task),
            now=FINAL_TIME,
        )
    with pytest.raises(claims.ClaimContractError, match="exactly one matching sibling"):
        claims.finalize_claim(
            root,
            reference,
            pending,
            task_id=task["task_id"],
            task_readback=task,
            sibling_snapshot={
                "matching_task_count": 0,
                "matching_tasks": [],
            },
            now=FINAL_TIME,
        )
    with pytest.raises(claims.ClaimContractError, match="validator rejected"):
        claims.finalize_claim(
            root,
            reference,
            pending,
            task_id=task["task_id"],
            task_readback=task,
            sibling_snapshot=_sibling_snapshot(task),
            now=FINAL_TIME,
            evidence_validator=lambda _task, _pending: (_ for _ in ()).throw(
                ValueError("rejected")
            ),
        )
    with pytest.raises(claims.ClaimContractError, match="predates"):
        claims.finalize_claim(
            root,
            reference,
            pending,
            task_id=task["task_id"],
            task_readback=task,
            sibling_snapshot=_sibling_snapshot(task),
            now="2026-07-25T05:29:59Z",
        )

    claim_directory = root / Path(reference["relative_claim_directory"])
    assert not (claim_directory / claims.FINALIZED_CLAIM_NAME).exists()
