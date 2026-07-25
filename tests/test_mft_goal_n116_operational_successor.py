from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from tools import mft_goal_n116_operational_successor as successor


CANDIDATE_SHA = "1" * 64
TASK_NAME = "mft-goal-diag-standard-n116-r4-l96212-111111111111"
DEDUPE = "mft-al:n116-successor"
SOURCE_NAME = (
    "mft-goal-diag-standard-dependency-r3-l96212-111111111111"
)


def _task(
    task_id: int,
    *,
    name: str,
    status: str,
    node: str = "n116",
    account: str = "dhj02",
    dedupe: str = "",
) -> dict:
    return {
        "id": task_id,
        "name": name,
        "status": status,
        "state": status,
        "project": "MFT_1MW_2026v1",
        "dedupe_key": dedupe,
        "cpus": 8,
        "memory_mb": 32768,
        "timeout_seconds": 28800,
        "aedt_backend": "standalone",
        "account_name": account,
        "node_name": node,
        "requested_node_name": node,
        "requested_node_name_policy": "strict",
        "allocation_id": 14621,
        "slurm_job_id": "829952",
        "created_at": "2026-07-25 13:00:00",
        "started_at": "2026-07-25 13:01:00",
        "finished_at": None if status == "running" else "2026-07-25 13:02:00",
    }


def _running_three() -> list[dict]:
    return [
        _task(
            task_id,
            name=(
                "mft-goal-diag-standard-pressure-r1-"
                f"t{task_id}-{task_id:012d}"
            ),
            status="running",
        )
        for task_id in (96294, 96295, 96296)
    ]


def test_storage_audit_accounts_for_running_three_and_candidate() -> None:
    audit = successor.build_storage_audit(
        observed_at_kst="2026-07-25T22:46:00+09:00",
        filesystem_type="gpfs",
        fileset_name="root",
        user_quota_scope="filesystem",
        quota_type="USR",
        block_used_gb=42.56877136230469,
        block_in_doubt_gb=0.015655517578125,
        block_limit_gb=110.0,
        rows=_running_three(),
    )
    assert audit["running_n116_task_count"] == 3
    assert audit["running_prospective_output_reservation_gb"] == 12.0
    assert audit["candidate_prospective_output_reservation_gb"] == 4.0
    assert audit["free_after_running_and_candidate_gb"] == pytest.approx(
        51.41557312011719
    )
    assert audit["minimum_free_floor_gb"] == 10.0
    assert audit["arithmetic_passed"] is True


def test_storage_audit_fails_closed_below_ten_gib() -> None:
    with pytest.raises(
        successor.HandoffContractError, match="below the 10 GiB floor"
    ):
        successor.build_storage_audit(
            observed_at_kst="2026-07-25T22:46:00+09:00",
            filesystem_type="gpfs",
            fileset_name="root",
            user_quota_scope="filesystem",
            quota_type="USR",
            block_used_gb=85.0,
            block_in_doubt_gb=0.0,
            block_limit_gb=110.0,
            rows=_running_three(),
        )


def test_storage_audit_freshness_is_bounded(tmp_path: Path) -> None:
    audit = successor.build_storage_audit(
        observed_at_kst="2026-07-25T22:46:00+09:00",
        filesystem_type="gpfs",
        fileset_name="root",
        user_quota_scope="filesystem",
        quota_type="USR",
        block_used_gb=42.5,
        block_in_doubt_gb=0.0,
        block_limit_gb=110.0,
        rows=_running_three(),
    )
    path = tmp_path / "audit.json"
    successor.production._write_immutable_json(path, audit)
    fresh_now = datetime.fromisoformat("2026-07-25T22:49:00+09:00")
    assert (
        successor._load_storage_audit(
            path, require_fresh=True, now=fresh_now
        )
        == audit
    )
    stale_now = fresh_now + timedelta(minutes=3)
    with pytest.raises(
        successor.HandoffContractError, match="stale for submission"
    ):
        successor._load_storage_audit(
            path, require_fresh=True, now=stale_now
        )


def test_candidate_inventory_allows_only_retryable_terminals() -> None:
    rows = [
        _task(
            96212,
            name="mft-goal-diag-standard-111111111111",
            status="failed",
        ),
        _task(
            96293,
            name=(
                "mft-goal-diag-standard-timeout-strict-r2-n116-"
                "111111111111"
            ),
            status="failed",
        ),
    ]
    inventory = successor._candidate_inventory(
        rows,
        candidate_physics_sha256=CANDIDATE_SHA,
        successor_task_name=TASK_NAME,
        successor_dedupe_key=DEDUPE,
        source_task_name=SOURCE_NAME,
    )
    assert inventory["matching_task_count"] == 2
    assert inventory["blocking_task_count"] == 0
    assert inventory["own_successor_task_count"] == 0
    assert inventory["source_plan_task_count"] == 0
    assert successor._validate_inventory(inventory) == inventory


@pytest.mark.parametrize("status", ["queued", "submitted", "running", "succeeded"])
def test_candidate_inventory_blocks_nonretryable_sibling(status: str) -> None:
    inventory = successor._candidate_inventory(
        [
            _task(
                96310,
                name=(
                    "mft-goal-diag-standard-dependency-r3-l96212-"
                    "111111111111"
                ),
                status=status,
            )
        ],
        candidate_physics_sha256=CANDIDATE_SHA,
        successor_task_name=TASK_NAME,
        successor_dedupe_key=DEDUPE,
        source_task_name=SOURCE_NAME,
    )
    assert inventory["blocking_task_count"] == 1
    assert inventory["cross_generation_blocking_task_count"] == 1


def test_capacity_zero_fails_closed() -> None:
    def reader(**_kwargs):
        return {
            "fit_slots": 0,
            "ready_fit_slots": 0,
            "memory_pressure_state": "ok",
        }

    with pytest.raises(
        successor.HandoffContractError, match="no ready Scheduler capacity"
    ):
        successor._require_ready_capacity(
            scheduler_url="http://127.0.0.1:8002",
            live_reader=reader,
        )


def test_capacity_positive_is_sealed_for_receipt() -> None:
    snapshot = {
        "fit_slots": 5,
        "ready_fit_slots": 5,
        "memory_pressure_state": "ok",
    }
    assert successor._require_ready_capacity(
        scheduler_url="http://127.0.0.1:8002",
        live_reader=lambda **_kwargs: snapshot,
    ) == snapshot


def test_claim_root_is_independent_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "n116-r4-claims"
    monkeypatch.setattr(successor, "CLAIM_ROOT", root)
    first = successor.initialize_claim_root()
    second = successor.initialize_claim_root()
    assert first == second
    assert first["campaign_authority_sha256"] == (
        successor.CLAIM_AUTHORITY_SHA256
    )
    assert "n116-r4-claims" in first["resolved_root"]


def test_task_identity_is_generation_and_node_distinct() -> None:
    name, workdir = successor._task_identity(
        logical_authority_task_id=96212,
        candidate_physics_sha256=CANDIDATE_SHA,
    )
    assert name == TASK_NAME
    assert workdir == "mft_goal_diag_standard_n116_r4_l96212_111111111111"
