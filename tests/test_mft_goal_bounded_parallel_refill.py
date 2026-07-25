from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from tools import mft_goal_bounded_parallel_refill as parallel
from tools import mft_goal_fea_handoff as production
from tools import mft_goal_safe_refill as refill


BOUNDS = {
    96223: 20_564_214_482,
    96224: 29_092_264_298,
    96230: 26_314_693_564,
}


def _candidate(logical: int) -> dict[str, Any]:
    bound = BOUNDS[logical]
    stem = {96223: "2a1bb6f2be79", 96224: "436565e3f360", 96230: "b7c30cb70b95"}[
        logical
    ]
    return {
        "logical_authority_task_id": logical,
        "retry_generation": f"timeout12h-r{logical}",
        "candidate_physics_sha256": stem + "a" * 52,
        "candidate_stem": stem,
        "plan": {
            "path": f"source-{logical}.json",
            "sha256": "b" * 64,
            "size_bytes": 1,
        },
        "plan_payload_sha256": "c" * 64,
        "task_name": f"mft-timeout12h-l{logical}-{stem}",
        "dedupe_key": f"sealed-dedupe-{logical}",
        "account_name": refill.TARGET_ACCOUNT,
        "prospective_grid_gb": bound / (1024**3),
        "fresh_grid_output_bytes": bound,
        "claim_key": f"sealed-claim-{logical}",
        "claim_directory": f"absent-claim-{logical}",
        "fixed_identity_attestation_sha256": refill.FIXED_IDENTITY_SHA256,
    }


def _plan(tmp_path: Path) -> dict[str, Any]:
    return {
        "scheduler_url": refill.SCHEDULER_URL,
        "watcher_plan": {"path": str(tmp_path / "watch.json")},
        "watcher_extension_directory": str(tmp_path / "extensions"),
        "cutover_receipt": {"path": str(tmp_path / "cutover.json")},
        "candidates": [_candidate(96223), _candidate(96230), _candidate(96224)],
    }


def _reclaim(tmp_path: Path, plan: dict[str, Any]) -> Path:
    full_minimum = (
        BOUNDS[96223] * parallel.FULL_SYMMETRY_EXPANSION_FACTOR / (1024**3)
        + refill.SAFETY_FLOOR_GB
    )
    release = 64.42045545578003
    observed = 103.7890625
    value = {
        "schema_version": parallel.RECLAIM_SCHEMA,
        "archive_target_filesystem": "local C:",
        "archive_growth_charge_to_gpfs_bytes": 0,
        "guaranteed_regular_allocated_release_bytes": int(release * 1024**3),
        "guaranteed_regular_allocated_release_gib": release,
        "capacity_floor": {
            "active_output_bound_gib": 0.0,
            "conditions": ["manifest", "delete", "no new writes"],
            "full_minimum_gib": full_minimum,
            "observed_free_gib": observed,
            "projected_postdelete_free_gib": observed + release,
            "projected_margin_gib": observed + release - full_minimum,
        },
    }
    return production._write_immutable_json(tmp_path / "reclaim.json", value)


def _slot(logical: int, execution: int) -> dict[str, Any]:
    candidate = _candidate(logical)
    return {
        "logical_authority_task_id": logical,
        "execution_task_id": execution,
        "task_name": candidate["task_name"],
        "dedupe_key": candidate["dedupe_key"],
        "candidate_physics_sha256": candidate["candidate_physics_sha256"],
        "fixed_identity_attestation_sha256": refill.FIXED_IDENTITY_SHA256,
    }


def _task(logical: int, execution: int) -> dict[str, Any]:
    slot = _slot(logical, execution)
    return {
        "task_id": execution,
        "id": execution,
        "name": slot["task_name"],
        "dedupe_key": slot["dedupe_key"],
        "project": refill.PROJECT,
        "account_name": refill.TARGET_ACCOUNT,
        "requested_account_name": refill.TARGET_ACCOUNT,
        "aedt_backend": "standalone",
        "required_capability": "conda:pyaedt2026v1",
        "cpus": 8,
        "memory_mb": 32768,
        "timeout_seconds": 43200,
        "requested_node_name": refill.TARGET_NODE,
        "requested_node_name_policy": "strict",
        "status": "running",
    }


def _capacity() -> dict[str, Any]:
    return {
        "ready_fit_slots": 7,
        "memory_pressure_state": "ok",
        "standalone_aedt_available": 399,
        "allocations": [
            {
                "account_name": refill.TARGET_ACCOUNT,
                "node_name": refill.TARGET_NODE,
                "state": "active",
                "fit_slots": 7,
            }
        ],
    }


def _storage() -> dict[str, Any]:
    return {
        "limiting_quota": {
            "raw_free_gb": 110.31979370117188,
            "in_doubt_gb": 5.516357421875,
        }
    }


def _evaluate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    tasks: list[dict[str, Any]],
    extensions: dict[int, dict[str, Any]],
    deadline: bool = True,
) -> dict[str, Any]:
    plan = _plan(tmp_path)
    reclaim_path = _reclaim(tmp_path, plan)
    monkeypatch.setattr(
        parallel, "EXPECTED_RECLAIM_EVIDENCE_PATH", reclaim_path.resolve()
    )
    monkeypatch.setattr(
        parallel,
        "EXPECTED_RECLAIM_EVIDENCE_SHA256",
        production._sha256_file(reclaim_path),
    )
    monkeypatch.setattr(refill, "_claim_state", lambda _candidate: "unsubmitted")
    return parallel.evaluate(
        plan,
        extension_authority_path=tmp_path / "authority.json",
        reclaim_evidence_path=reclaim_path,
        task_reader=lambda **_kwargs: tasks,
        capacity_reader=lambda **_kwargs: _capacity(),
        storage_reader=lambda _plan: _storage(),
        validate_cutover=False,
        extension_reader=lambda *_args, **_kwargs: extensions,
        deadline_reader=lambda: deadline,
    )


def test_first_parallel_candidate_uses_active_plus_prospective_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _evaluate(
        tmp_path,
        monkeypatch,
        tasks=[_task(96223, 96303)],
        extensions={96223: _slot(96223, 96303)},
    )
    assert result["selected_logical_authority_task_id"] == 96230
    row = result["candidates"][0]
    assert row["logical_authority_task_id"] == 96230
    assert row["eligible"] is True
    assert row["active_authenticated_storage_bound_gib"] == pytest.approx(
        BOUNDS[96223] / (1024**3)
    )
    assert (
        row["fresh_gpfs_remaining_after_active_prospective_floor_gib"]
        > 0
    )
    assert result["phase_transition_readiness"][
        "standard_and_full_reservations_temporally_separated"
    ] is True


def test_second_candidate_reaudits_both_active_standard_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _evaluate(
        tmp_path,
        monkeypatch,
        tasks=[_task(96223, 96303), _task(96230, 96304)],
        extensions={
            96223: _slot(96223, 96303),
            96230: _slot(96230, 96304),
        },
    )
    assert result["selected_logical_authority_task_id"] == 96224
    selected = next(
        row
        for row in result["candidates"]
        if row["logical_authority_task_id"] == 96224
    )
    assert selected[
        "cumulative_bound_with_this_candidate_gib"
    ] == pytest.approx(sum(BOUNDS.values()) / (1024**3))
    assert selected[
        "fresh_gpfs_remaining_after_active_prospective_floor_gib"
    ] == pytest.approx(24.049768455326557)


def test_queued_requested_r1_task_cannot_escape_cumulative_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queued = _task(96230, 96304)
    queued["status"] = "queued"
    queued["account_name"] = ""
    result = _evaluate(
        tmp_path,
        monkeypatch,
        tasks=[_task(96223, 96303), queued],
        extensions={
            96223: _slot(96223, 96303),
            96230: _slot(96230, 96304),
        },
    )
    assert result["selected_logical_authority_task_id"] == 96224
    assert result["active_authenticated_storage_bound_gib"] == pytest.approx(
        (BOUNDS[96223] + BOUNDS[96230]) / (1024**3)
    )


def test_unknown_r1_fea_or_closed_deadline_blocks_all_posts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unknown = {
        **_task(96223, 99999),
        "name": "unknown-r1-fea",
        "dedupe_key": "unknown",
    }
    result = _evaluate(
        tmp_path,
        monkeypatch,
        tasks=[_task(96223, 96303), unknown],
        extensions={96223: _slot(96223, 96303)},
        deadline=False,
    )
    assert result["selected_logical_authority_task_id"] is None
    assert result["unapproved_r1_active_fea_task_ids"] == [99999]
    assert all(not row["eligible"] for row in result["candidates"])


def test_lifetime_two_additional_submission_ceiling_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extensions = {
        96223: _slot(96223, 96303),
        96230: _slot(96230, 96304),
        96224: _slot(96224, 96305),
    }
    result = _evaluate(
        tmp_path,
        monkeypatch,
        tasks=[
            _task(96223, 96303),
            _task(96230, 96304),
            _task(96224, 96305),
        ],
        extensions=extensions,
    )
    assert result["remaining_additional_post_authority"] == 0
    assert result["selected_logical_authority_task_id"] is None


def test_cutoff_is_0530_kst_and_boundary_is_closed() -> None:
    assert parallel.SUBMISSION_CUTOFF_KST.isoformat() == (
        "2026-07-26T05:30:00+09:00"
    )
    assert parallel.SUBMISSION_CUTOFF_UTC.isoformat() == (
        "2026-07-25T20:30:00+00:00"
    )
    assert parallel._deadline_open_at(
        parallel.SUBMISSION_CUTOFF_UTC - parallel.timedelta(microseconds=1)
    )
    assert not parallel._deadline_open_at(parallel.SUBMISSION_CUTOFF_UTC)


def test_reclaim_file_byte_drift_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(tmp_path)
    reclaim_path = _reclaim(tmp_path, plan)
    monkeypatch.setattr(
        parallel, "EXPECTED_RECLAIM_EVIDENCE_PATH", reclaim_path.resolve()
    )
    monkeypatch.setattr(
        parallel, "EXPECTED_RECLAIM_EVIDENCE_SHA256", "0" * 64
    )
    with pytest.raises(
        parallel.BoundedParallelRefillError, match="immutable byte identity"
    ):
        parallel._phase_transition_evidence(
            plan, reclaim_evidence_path=reclaim_path
        )


def test_direct_script_entrypoint_bootstraps_repository_root(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [sys.executable, str(Path(parallel.__file__).resolve()), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Bounded parallel exact MFT SAFE REFILL extension" in completed.stdout
