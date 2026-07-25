from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from tools import mft_campaign_atomic_claim as atomic_claim
from tools import mft_goal_full_fastlane_watcher as fastlane
from tools import mft_goal_terminal_success_watcher as terminal


def _gpfs(account: str = "dhj02") -> dict[str, Any]:
    return {
        "account_name": account,
        "filesystem_type": "gpfs",
        "fileset_name": "root",
        "user_quota_scope": "filesystem",
        "quota_type": "USR",
        "block_used_gb": 40.0,
        "block_in_doubt_gb": 0.0,
        "block_limit_gb": 110.0,
    }


def _task(
    task_id: int,
    *,
    account: str = "dhj02",
    timeout: int = 28800,
    status: str = "running",
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "id": task_id,
        "name": f"mft-goal-diag-standard-{task_id}",
        "status": status,
        "state": status,
        "project": "MFT_1MW_2026v1",
        "dedupe_key": f"mft-test:{task_id}",
        "account_name": account,
        "cpus": 8,
        "memory_mb": 32768,
        "timeout_seconds": timeout,
        "aedt_backend": "standalone",
        "created_at": "2026-07-25 14:00:00",
        "started_at": "2026-07-25 14:01:00",
        "finished_at": None,
    }


def _capacity(account: str = "dhj02") -> dict[str, Any]:
    return {
        "fit_slots": 1,
        "ready_fit_slots": 1,
        "pending_fit_slots": 0,
        "inflight_fit_slots": 1,
        "memory_pressure_state": "ok",
        "allocations": [
            {
                "allocation_id": 1,
                "account_name": account,
                "slurm_job_id": "123",
                "state": "warm",
                "partition": "cpu",
                "node_name": "n116",
                "free_cpus": 16,
                "free_memory_mb": 98304,
                "free_gpus": 0,
                "gpu_model": "",
                "fit_slots": 1,
                "memory_pressure_state": "ok",
                "node_memory_free_percent": 60.0,
            }
        ],
    }


def test_full_profile_keeps_reviewed_physics_and_resources() -> None:
    profile, _record = fastlane.promotion._full_profile()
    assert profile["mem_mb"] == 98304
    assert profile["cpus"] == 16
    assert profile["timeout_seconds"] == 43200
    assert profile["param_overrides"]["full_model"] == 1
    assert profile["param_overrides"]["thermal_symmetry"] == "full"
    assert profile["param_overrides"]["fan_velocity"] == 1.5
    assert profile["param_overrides"]["fan_config"] == "dual"
    assert profile["fixed_boundary_contract"] == {
        "thermal_pad_conductivity_W_mK": 0.2,
        "core_plate_pad_t_mm": 2.0,
        "wcp_pad_t_mm": 2.0,
        "fan_velocity_m_s": 1.5,
        "fan_config": "dual",
        "core_plate_on": 1,
        "wcp_on": 1,
    }


def test_terminal_authority_requires_immutable_nds_receipt_and_rank0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "truth_pareto_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    receipt = tmp_path / "truth.receipt.json"
    receipt_value = terminal._sealed(
        {
            "schema_version": terminal.NDS_RECEIPT_SCHEMA,
            "campaign_id": "mft-goal-20260726",
            "authority_key": "a" * 16,
            "logical_authority_task_ids": [96212],
            "input_collections": [],
            "truth_manifest": fastlane.production._file_record(manifest),
            "global_nondominated_sort_performed": True,
            "scheduler_mutation_performed": False,
            "created_at_utc": "2026-07-25T14:00:00+00:00",
        }
    )
    fastlane.production._write_immutable_json(receipt, receipt_value)
    rank0 = {
        "candidate_physics_sha256": "1" * 64,
        "truth_non_dominated_rank": 0,
    }
    monkeypatch.setattr(
        fastlane.promotion,
        "_load_truth_manifest",
        lambda _path: (
            {"payload_sha256": "2" * 64, "rank0_count": 1},
            [rank0],
            {},
        ),
    )
    authenticated = fastlane._authenticate_terminal_authority(
        manifest_path=manifest, receipt_path=receipt
    )
    assert authenticated["rank0_rows"] == [rank0]


def test_terminal_authority_rejects_mutating_promotion_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "truth_pareto_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    receipt = tmp_path / "truth.receipt.json"
    fastlane.production._write_immutable_json(
        receipt,
        terminal._sealed(
            {
                "schema_version": terminal.NDS_RECEIPT_SCHEMA,
                "campaign_id": "mft-goal-20260726",
                "truth_manifest": fastlane.production._file_record(manifest),
                "global_nondominated_sort_performed": True,
                "scheduler_mutation_performed": True,
            }
        ),
    )
    monkeypatch.setattr(
        fastlane.promotion,
        "_load_truth_manifest",
        lambda _path: (
            {"payload_sha256": "2" * 64, "rank0_count": 1},
            [
                {
                    "candidate_physics_sha256": "1" * 64,
                    "truth_non_dominated_rank": 0,
                }
            ],
            {},
        ),
    )
    with pytest.raises(
        fastlane.HandoffContractError,
        match="rank-0 promotion authority drifted",
    ):
        fastlane._authenticate_terminal_authority(
            manifest_path=manifest, receipt_path=receipt
        )


def test_gpfs_audit_reserves_active_tasks_and_candidate() -> None:
    audit = fastlane.build_gpfs_audit(
        observed_at_utc="2026-07-25T23:00:00+09:00",
        account_observations=[_gpfs()],
        rows=[_task(1), _task(2)],
    )
    account = audit["account_observations"][0]
    assert account["active_bounded_task_count"] == 2
    assert account["active_task_reservation_gb"] == 8.0
    assert account["candidate_reservation_gb"] == 4.0
    assert account["free_after_active_and_candidate_gb"] == 58.0
    assert account["minimum_free_floor_gb"] == 10.0
    assert audit["active_unbounded_storage_conflict_count"] == 0
    assert audit["arithmetic_passed"] is True


@pytest.mark.parametrize(
    ("account", "timeout"),
    [("", 28800), ("unknown", 28800), ("dhj02", 0)],
)
def test_gpfs_audit_blocks_unbounded_or_unaudited_active_work(
    account: str, timeout: int
) -> None:
    audit = fastlane.build_gpfs_audit(
        observed_at_utc="2026-07-25T23:00:00+09:00",
        account_observations=[_gpfs()],
        rows=[_task(1, account=account, timeout=timeout)],
    )
    assert audit["active_unbounded_storage_conflict_count"] == 1
    assert audit["arithmetic_passed"] is False


def test_gpfs_audit_is_fresh_and_fails_closed_when_stale(
    tmp_path: Path,
) -> None:
    observed = datetime(2026, 7, 25, 14, 0, tzinfo=timezone.utc)
    audit = fastlane.build_gpfs_audit(
        observed_at_utc=observed.isoformat(),
        account_observations=[_gpfs()],
        rows=[],
    )
    path = tmp_path / "audit.json"
    fastlane.production._write_immutable_json(path, audit)
    assert fastlane._load_gpfs_audit(
        path, require_fresh=True, now=observed + timedelta(seconds=90)
    ) == audit
    with pytest.raises(fastlane.HandoffContractError, match="GPFS audit is stale"):
        fastlane._load_gpfs_audit(
            path, require_fresh=True, now=observed + timedelta(seconds=121)
        )


def test_capacity_must_be_ready_on_exact_audited_account() -> None:
    snapshot = _capacity()
    assert (
        fastlane._require_account_capacity(
            scheduler_url="http://127.0.0.1:8002",
            account_name="dhj02",
            live_reader=lambda **_kwargs: snapshot,
        )
        == snapshot
    )
    wrong = _capacity(account="other")
    with pytest.raises(
        fastlane.HandoffContractError,
        match="no ready 16-core/98304MB capacity",
    ):
        fastlane._require_account_capacity(
            scheduler_url="http://127.0.0.1:8002",
            account_name="dhj02",
            live_reader=lambda **_kwargs: wrong,
        )


def test_capacity_zero_fails_closed() -> None:
    with pytest.raises(
        fastlane.HandoffContractError,
        match="no ready 16-core/98304MB capacity",
    ):
        fastlane._require_account_capacity(
            scheduler_url="http://127.0.0.1:8002",
            account_name="dhj02",
            live_reader=lambda **_kwargs: {
                "fit_slots": 0,
                "ready_fit_slots": 0,
                "memory_pressure_state": "ok",
                "allocations": [],
            },
        )


def test_priority_fence_binds_manifest_rank0_and_selection(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    promotion_receipt = tmp_path / "promotion.json"
    manifest.write_text("{}\n", encoding="utf-8")
    promotion_receipt.write_text("{}\n", encoding="utf-8")
    selection = fastlane._sealed(
        {
            "schema_version": fastlane.SELECTION_SCHEMA,
            "truth_manifest": fastlane.production._file_record(manifest),
            "truth_manifest_payload_sha256": "1" * 64,
            "promotion_receipt": fastlane.production._file_record(
                promotion_receipt
            ),
            "promotion_receipt_payload_sha256": "2" * 64,
            "candidate_physics_sha256": "3" * 64,
            "standard_task_id": 96212,
        }
    )
    selection_path = tmp_path / "selection.json"
    fastlane.production._write_immutable_json(selection_path, selection)
    plan = {
        "output_root": str(tmp_path),
        "priority_fence_path": str(tmp_path / "shared" / "fence.json"),
    }
    fence = fastlane._ensure_priority_fence(
        plan=plan, selection=selection
    )
    assert fence["selection"] == fastlane.production._file_record(
        selection_path
    )
    assert fence["selection_payload_sha256"] == selection["payload_sha256"]
    assert fence["truth_non_dominated_rank"] == 0
    assert fence["safe_refill_next_post_blocked"] is True
    assert fence["safe_refill_logical_authority_task_ids"] == [
        96223,
        96224,
        96230,
    ]


def test_active_safe_refill_yields_to_priority_full() -> None:
    active = [
        _task(1),
        {
            **_task(2),
            "name": "mft-goal-diag-timeout-r4-l96230-b7c30cb70b95",
        },
    ]
    assert [
        row["task_id"]
        for row in fastlane._active_safe_refill_tasks(active)
    ] == [2]


def test_latest_license_snapshot_requires_fresh_full_headroom(
    tmp_path: Path,
) -> None:
    checked = datetime(2026, 7, 25, 14, 0, tzinfo=timezone.utc)
    directory = tmp_path / "capture"
    directory.mkdir()
    snapshot = {
        "schema": "mft-aedt-license-headroom-snapshot-v1",
        "server_up": True,
        "server": "1055@172.16.10.81",
        "checked_at": checked.isoformat(),
        "features": {
            "anshpc": {"total": 32, "used": 0},
            "elec_solve_maxwell": {"total": 2, "used": 0},
            "electronics_desktop": {"total": 2, "used": 0},
            "electronics3d_gui": {"total": 2, "used": 0},
        },
    }
    path = directory / "snapshot.json"
    path.write_text(
        fastlane.json.dumps(snapshot, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    assert fastlane._latest_license_snapshot(
        tmp_path, now=checked + timedelta(seconds=300)
    ) == path.resolve()
    with pytest.raises(
        fastlane.HandoffContractError,
        match="no fresh passing 16-core license snapshot",
    ):
        fastlane._latest_license_snapshot(
            tmp_path, now=checked + timedelta(seconds=601)
        )


def test_without_activation_cycle_cannot_reach_submit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = {
        "payload_sha256": "1" * 64,
        "output_root": str(tmp_path),
    }
    selection = {
        "payload_sha256": "2" * 64,
        "candidate_physics_sha256": "3" * 64,
    }
    gates = {"selected_account_name": "dhj02"}
    monkeypatch.setattr(fastlane, "_load_watch_plan", lambda _path: plan)
    monkeypatch.setattr(fastlane, "_prepare_selection", lambda _plan: selection)
    monkeypatch.setattr(fastlane, "_fresh_gates", lambda *_args, **_kwargs: gates)

    def forbidden(**_kwargs):
        raise AssertionError("submission path was reached without activation")

    monkeypatch.setattr(fastlane, "_execute_submission", forbidden)
    state = fastlane.process_cycle(watch_plan_path=tmp_path / "watch_plan.json")
    assert state["status"] == "ready_live_post_disabled_pending_root_review"
    assert state["scheduler_submission_performed"] is False
    assert state["live_post_default_enabled"] is False


def test_submission_adapter_posts_once_and_injects_locked_guard(
    tmp_path: Path,
) -> None:
    calls = []
    guards = []

    class Delegate:
        @staticmethod
        def submit_verification(*args, **kwargs):
            calls.append((args, kwargs))
            kwargs["pre_submit_guard"]()
            return 96310

    adapter = fastlane._SubmissionAdapter(
        delegate=Delegate,
        account_name="dhj02",
        guard=lambda: guards.append("guarded"),
        post_result_path=tmp_path / "post_result.json",
    )
    assert adapter.submit_verification("name", "work", {}) == 96310
    assert calls[0][1]["account_name"] == "dhj02"
    assert guards == ["guarded"]
    assert adapter.scheduler_post_count == 1
    with pytest.raises(fastlane.HandoffContractError, match="re-POST"):
        adapter.submit_verification("name", "work", {})


def test_recovery_adapter_never_calls_scheduler_delegate(
    tmp_path: Path,
) -> None:
    class Forbidden:
        @staticmethod
        def submit_verification(*_args, **_kwargs):
            raise AssertionError("recovery attempted a Scheduler POST")

    adapter = fastlane._SubmissionAdapter(
        delegate=Forbidden,
        account_name="dhj02",
        guard=lambda: None,
        post_result_path=tmp_path / "post_result.json",
        recovered_task_id=96310,
    )
    assert adapter.submit_verification("name", "work", {}) == 96310
    assert adapter.scheduler_post_count == 0
    assert not (tmp_path / "post_result.json").exists()


def test_singleton_claim_rejects_a_second_full_winner(tmp_path: Path) -> None:
    authority = fastlane._initialize_claim_root(tmp_path / "claims")
    reference = fastlane._claim_reference(authority)
    winner = {
        "immediate_task_id": 1,
        "immediate_retry_kind": "none",
        "plan_payload_sha256": "1" * 64,
        "plan_file_sha256": "2" * 64,
        "profile_sha256": "3" * 64,
        "resources": dict(fastlane.FULL_RESOURCES),
        "task_name": "mft-goal-truth-full-first",
        "dedupe_key": "mft:test:first",
    }
    first = atomic_claim.acquire_claim(
        tmp_path / "claims", reference, winner
    )
    assert first["status"] == "fresh_pending"
    second_winner = dict(winner)
    second_winner["task_name"] = "mft-goal-truth-full-second"
    second_winner["dedupe_key"] = "mft:test:second"
    with pytest.raises(
        atomic_claim.ClaimContractError,
        match="different ancestry",
    ):
        atomic_claim.acquire_claim(
            tmp_path / "claims", reference, second_winner
        )


def test_active_4fac_is_mandatory_at_watch_plan_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_parent = tmp_path / "terminal"
    state_parent.mkdir()
    cutover = tmp_path / "cutover.json"
    cutover.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        fastlane,
        "_initialize_claim_root",
        lambda _root: {"resolved_root": str(tmp_path / "claims")},
    )
    monkeypatch.setattr(
        fastlane,
        "_claim_reference",
        lambda _authority: {"payload_sha256": "1" * 64},
    )
    monkeypatch.setattr(
        fastlane.diagnostic,
        "_validate_scheduler_cutover_receipt",
        lambda *_args, **_kwargs: (
            {
                "candidate_revision": "0" * 40,
                "pin_generation": "scheduler-strict-node-legacy",
                "payload_sha256": "2" * 64,
            },
            None,
        ),
    )
    with pytest.raises(
        fastlane.HandoffContractError,
        match="requires active Scheduler 4fac",
    ):
        fastlane.initialize_watch_plan(
            terminal_state_path=state_parent / "state.json",
            scheduler_cutover_receipt_path=cutover,
            license_snapshot_directory=tmp_path / "licenses",
            gpfs_audit_directory=tmp_path / "gpfs",
            output_root=tmp_path / "fastlane",
            claim_root=tmp_path / "claims",
        )
