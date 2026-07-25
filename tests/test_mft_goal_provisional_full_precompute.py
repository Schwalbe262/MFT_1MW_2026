from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from tools import mft_campaign_atomic_claim as atomic_claim
from tools import mft_goal_fea_handoff as production
from tools import mft_goal_provisional_full_precompute as provisional


def _selected(
    *,
    candidate: str,
    rank: int,
    volume: float,
    loss: float,
    first: float,
    second: float,
    dimensions: dict[str, float],
) -> dict[str, Any]:
    return {
        "selected_row": {
            "diagnostic_selection_rank": rank,
            "objective_volume_L": volume,
            "objective_total_loss_W": loss,
        },
        "row_contract": {
            "normalized_G": {
                "Llt_ensemble_disagreement": first,
                "Llt_robust_band": second,
                "half_magnetizing_resonance_minimum": -0.1,
                "temperature_robust_limit:T_max_Tx": -0.02,
                "temperature_robust_limit:T_max_core": -0.03,
            },
            "exterior_dimensions_mm": dimensions,
            "surrogate_mean_Llt_in_band": True,
            "goal_non_Llt_hard_spec_passed": True,
            "surrogate_robust_feasible": False,
            "physical_geometry_sha256": candidate,
        },
    }


def test_selection_rationale_freezes_b7_over_2a() -> None:
    target_plan = {
        "candidate_physics_sha256": provisional.TARGET_CANDIDATE_SHA256,
        "retry_of_timeout12h": {"logical_authority_task_id": 96230},
    }
    comparison_plan = {
        "candidate_physics_sha256": provisional.COMPARISON_CANDIDATE_SHA256,
        "retry_of_timeout12h": {"logical_authority_task_id": 96223},
    }
    target = _selected(
        candidate=provisional.TARGET_CANDIDATE_SHA256,
        rank=1,
        volume=803.9135169656241,
        loss=5291.746327152702,
        first=0.6965652185724749,
        second=1.5486579169672403,
        dimensions={"W": 1186.354, "L": 970.8220000000001, "H": 698.0},
    )
    comparison = _selected(
        candidate=provisional.COMPARISON_CANDIDATE_SHA256,
        rank=9,
        volume=860.394184155,
        loss=5496.370392297698,
        first=1.8502247647684107,
        second=1.9055607822619043,
        dimensions={"W": 1188.99, "L": 964.846, "H": 750.0},
    )
    result = provisional._selection_rationale(
        target_plan=target_plan,
        target_selected=target,
        comparison_plan=comparison_plan,
        comparison_selected=comparison,
    )
    assert result["target"]["normalized_positive_violation_sum"] == pytest.approx(
        2.245223135539715
    )
    assert result["comparison"]["normalized_positive_violation_sum"] == pytest.approx(
        3.7557855470303148
    )
    assert result["robust_Llt_constraint_passed"] is False
    assert result["provisional_diagnostic_only_required"] is True
    assert result["source_actual_standard"]["task_id"] == 96304


def test_project_only_retention_bound_covers_aedt_and_base64_not_results() -> None:
    expected = 16 * 1024**3
    result = provisional._retention_storage_bound(expected)
    assert result["expected_full_aedt_bytes"] == expected
    assert result["base64_encoded_bytes_upper_bound"] == 4 * ((expected + 2) // 3)
    assert result["gpfs_retention_bound_bytes"] == (
        expected
        + result["base64_encoded_bytes_upper_bound"]
        + provisional.RETENTION_METADATA_RESERVE_BYTES
    )
    assert result["results_directory_included"] is False


def test_exact_capacity_fails_closed_without_r1_n114(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = {
        "allocations": [
            {
                "account_name": provisional.ACCOUNT_NAME,
                "node_name": "n116",
                "free_cpus": 16,
                "free_memory_mb": 98304,
                "slurm_job_id": "10",
            }
        ]
    }
    monkeypatch.setattr(
        provisional.fastlane,
        "_require_account_capacity",
        lambda **_kwargs: response,
    )
    with pytest.raises(provisional.HandoffContractError, match="r1/n114"):
        provisional._exact_capacity(
            live_reader=lambda **_kwargs: {}, scheduler_url="scheduler"
        )


def test_parallel_active_standard_bounds_are_reserved_and_unknown_fea_fails() -> None:
    plan = {
        "active_standard_storage_authorities": [
            {
                "logical_authority_task_id": 96230,
                "task_name": "standard-b7",
                "dedupe_key": "dedupe-b7",
                "fresh_grid_output_bytes": 20 * 1024**3,
                "fresh_grid_output_gib": 20.0,
                "bound_source": "authenticated_timeout_native_premesh_grid_bytes",
            }
        ]
    }
    task = {
        "task_id": 96304,
        "id": 96304,
        "name": "standard-b7",
        "dedupe_key": "dedupe-b7",
        "account_name": provisional.ACCOUNT_NAME,
        "aedt_backend": "standalone",
    }
    result = provisional._active_storage_reservation(plan, [task])
    assert result["active_bounded_task_count"] == 1
    assert result["active_storage_bound_gib"] == 20.0
    with pytest.raises(provisional.HandoffContractError, match="lacks authenticated"):
        provisional._active_storage_reservation(
            plan,
            [
                {
                    **task,
                    "task_id": 96399,
                    "id": 96399,
                    "name": "unknown-full",
                }
            ],
        )


def test_cutoff_blocks_before_live_scheduler_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        provisional,
        "load_plan",
        lambda _path: (
            {
                "target_source_plan": {"path": "unused"},
                "scheduler_url": provisional.SCHEDULER_URL,
            },
            {},
            {},
        ),
    )

    def forbidden(**_kwargs: Any) -> Any:
        raise AssertionError("live reader must not run after cutoff")

    with pytest.raises(provisional.HandoffContractError, match="cutoff"):
        provisional.fresh_gates(
            plan_path=Path("plan.json"),
            license_snapshot_path=Path("snapshot.json"),
            now=datetime.fromisoformat("2026-07-26T05:30:01+09:00"),
            live_reader=forbidden,
            active_task_reader=forbidden,
            task_reader=forbidden,
            validate_cutover=False,
        )


def _submission_fixture(
    tmp_path: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    claim_root = tmp_path / "claims"
    authority = atomic_claim.initialize_claim_root(
        claim_root,
        campaign_id=provisional.CAMPAIGN_ID,
        campaign_authority_sha256="a" * 64,
    )
    reference = atomic_claim.build_claim_reference(
        authority,
        candidate_physics_sha256=provisional.TARGET_CANDIDATE_SHA256,
        logical_authority_task_id=provisional.LOGICAL_AUTHORITY_TASK_ID,
        retry_generation=provisional.CLAIM_GENERATION,
    )
    retained = {
        "dedupe_key": "mft-al:provisional-full:test",
        "schema_version": "mft-goal-fea-retained-aedt-v1",
        "artifact_path": "goal-fea-retained/test/full.aedt",
    }
    plan = production._seal(
        {
            "schema_version": provisional.PLAN_SCHEMA,
            "output_root": str(tmp_path.resolve()),
            "payload_marker": "fixture",
            "profile": {
                "path": str(tmp_path / "goal_full.json"),
                "sha256": "c" * 64,
                "size_bytes": 1,
            },
            "profile_canonical_sha256": "b" * 64,
            "stage": {
                "task_name": "mft-goal-provisional-full-test",
                "workdir": "mft_goal_provisional_full_test",
                "retained_aedt": retained,
            },
            "retention_storage_bound": provisional._retention_storage_bound(1024**3),
            "scheduler_url": provisional.SCHEDULER_URL,
            "scheduler_project": provisional.SCHEDULER_PROJECT,
            "claim_root_authority": authority,
            "claim_reference": reference,
            **provisional._flags(),
        }
    )
    path = production._write_immutable_json(
        tmp_path / "provisional_full_precompute_plan.json", plan
    )
    task = {
        "task_id": 97000,
        "id": 97000,
        "name": plan["stage"]["task_name"],
        "status": "queued",
        "state": "queued",
        "project": provisional.SCHEDULER_PROJECT,
        "dedupe_key": retained["dedupe_key"],
        "cpus": 16,
        "memory_mb": 98304,
        "timeout_seconds": 43200,
        "aedt_backend": "standalone",
        "account_name": provisional.ACCOUNT_NAME,
        "requested_account_name": provisional.ACCOUNT_NAME,
        "actual_node_name": None,
        "allocation_node_name": None,
        "requested_node_name": provisional.NODE_NAME,
        "node_name_policy": "strict",
        "requested_node_name_policy": "strict",
        "same_node_as_task_id": 0,
    }
    return path, plan, task


def test_submit_defers_all_scratch_env_and_posts_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, plan, task = _submission_fixture(tmp_path)
    monkeypatch.setattr(
        provisional, "load_plan", lambda _path: (plan, {"w1": 1}, {"cpus": 16})
    )
    monkeypatch.setattr(
        provisional.production,
        "_submission_environment",
        lambda **_kwargs: (
            {
                "MFT_STANDALONE_CORE_CONTRACT": "full",
                "MFT_RUNTIME_LICENSE_REFRESH": "1",
            },
            {},
        ),
    )
    gates = production._seal(
        {
            "schema_version": provisional.GATE_SCHEMA,
            "plan": production._file_record(plan_path),
            "plan_payload_sha256": plan["payload_sha256"],
            **provisional._flags(),
        }
    )
    gate_calls = {"count": 0}

    def gate_reader(**_kwargs: Any) -> dict[str, Any]:
        gate_calls["count"] += 1
        return gates

    sibling_calls = {"count": 0}

    def sibling_reader(**_kwargs: Any) -> list[dict[str, Any]]:
        sibling_calls["count"] += 1
        return [] if sibling_calls["count"] <= 2 else [task]

    class Scheduler:
        calls: list[dict[str, Any]] = []

        @classmethod
        def submit_verification(cls, *_args: Any, **kwargs: Any) -> dict[str, Any]:
            cls.calls.append(kwargs)
            kwargs["pre_submit_guard"]()
            return {
                "task_id": task["task_id"],
                "submission_source": "api_post",
                "scheduler_mutation_performed": True,
                "api_pre_submission_readback": None,
                "api_post_submission_response": task,
            }

    license_path = tmp_path / "snapshot.json"
    license_path.write_text("{}\n", encoding="utf-8")
    receipt_path = provisional.submit(
        plan_path=plan_path,
        license_snapshot_path=license_path,
        output=tmp_path / "submission.json",
        scheduler=Scheduler,
        gate_reader=gate_reader,
        sibling_reader=sibling_reader,
    )
    receipt = production._read_json(receipt_path)
    assert gate_calls["count"] == 2
    assert len(Scheduler.calls) == 1
    options = Scheduler.calls[0]
    assert options["submission_env_after_workdir"] is True
    assert options["required_workdir_prefix"] == "/enroot/"
    for key in ("ANS_TEMP_PATH", "TMPDIR", "TMP", "TEMP"):
        assert options["submission_env"][key] == "$MFT_WORKDIR"
    assert "ANS_MW_INHERIT_TMP" not in options["submission_env"]
    assert options["account_name"] == "r1jae262"
    assert options["node_name"] == "n114"
    assert receipt["scheduler_post_count_this_run"] == 1
    assert receipt["provisional"] is True
    assert receipt["production_eligible"] is False
    assert (tmp_path / "post_intent.json").is_file()
    assert (tmp_path / "post_result.json").is_file()


def test_pending_intent_without_sibling_forbids_repost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, plan, _task = _submission_fixture(tmp_path)
    monkeypatch.setattr(
        provisional, "load_plan", lambda _path: (plan, {}, {"cpus": 16})
    )
    monkeypatch.setattr(
        provisional.production,
        "_submission_environment",
        lambda **_kwargs: ({"MFT_RUNTIME_LICENSE_REFRESH": "1"}, {}),
    )
    gates = production._seal(
        {
            "schema_version": provisional.GATE_SCHEMA,
            "plan": production._file_record(plan_path),
            "plan_payload_sha256": plan["payload_sha256"],
            **provisional._flags(),
        }
    )

    class CrashingScheduler:
        calls = 0

        @classmethod
        def submit_verification(cls, *_args: Any, **kwargs: Any) -> Any:
            cls.calls += 1
            kwargs["pre_submit_guard"]()
            raise RuntimeError("ambiguous transport failure")

    license_path = tmp_path / "snapshot.json"
    license_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="ambiguous"):
        provisional.submit(
            plan_path=plan_path,
            license_snapshot_path=license_path,
            output=tmp_path / "submission-1.json",
            scheduler=CrashingScheduler,
            gate_reader=lambda **_kwargs: gates,
            sibling_reader=lambda **_kwargs: [],
        )
    assert CrashingScheduler.calls == 1
    assert (tmp_path / "post_intent.json").is_file()

    class ForbiddenScheduler:
        @staticmethod
        def submit_verification(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("second lifetime POST attempted")

    with pytest.raises(provisional.HandoffContractError, match="re-POST is forbidden"):
        provisional.submit(
            plan_path=plan_path,
            license_snapshot_path=license_path,
            output=tmp_path / "submission-2.json",
            scheduler=ForbiddenScheduler,
            gate_reader=lambda **_kwargs: gates,
            sibling_reader=lambda **_kwargs: [],
        )


def test_pending_claim_with_one_sibling_recovers_without_fresh_gate_or_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, plan, task = _submission_fixture(tmp_path)
    monkeypatch.setattr(
        provisional, "load_plan", lambda _path: (plan, {}, {"cpus": 16})
    )
    monkeypatch.setattr(
        provisional.production,
        "_submission_environment",
        lambda **_kwargs: ({"MFT_RUNTIME_LICENSE_REFRESH": "1"}, {}),
    )
    gates = production._seal(
        {
            "schema_version": provisional.GATE_SCHEMA,
            "plan": production._file_record(plan_path),
            "plan_payload_sha256": plan["payload_sha256"],
            **provisional._flags(),
        }
    )

    class AmbiguousScheduler:
        @staticmethod
        def submit_verification(*_args: Any, **kwargs: Any) -> Any:
            kwargs["pre_submit_guard"]()
            raise RuntimeError("response lost after POST")

    license_path = tmp_path / "snapshot.json"
    license_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="response lost"):
        provisional.submit(
            plan_path=plan_path,
            license_snapshot_path=license_path,
            output=tmp_path / "submission-1.json",
            scheduler=AmbiguousScheduler,
            gate_reader=lambda **_kwargs: gates,
            sibling_reader=lambda **_kwargs: [],
        )

    def forbidden_gate(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("recovery attempted a fresh submit-time gate")

    class ForbiddenScheduler:
        @staticmethod
        def submit_verification(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("recovery attempted a second POST")

    path = provisional.submit(
        plan_path=plan_path,
        license_snapshot_path=license_path,
        output=tmp_path / "submission-2.json",
        scheduler=ForbiddenScheduler,
        gate_reader=forbidden_gate,
        sibling_reader=lambda **_kwargs: [task],
    )
    receipt = production._read_json(path)
    assert receipt["atomic_claim_acquisition_status"] == "existing_pending"
    assert receipt["scheduler_post_count_this_run"] == 0
    assert receipt["task_id"] == task["task_id"]
