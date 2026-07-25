import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from regression_260707.verify import scheduler_client
from tools import mft_goal_g3dmesher_high_memory_retry as recovery
from tools import mft_goal_fea_handoff as production


def _source_bundle(tmp_path: Path):
    params = {
        "thermal_rx_side_block_mesh_level": 5,
        "fan_velocity": 1.5,
        "fan_config": "dual",
        "core_plate_pad_t": 2.0,
        "wcp_pad_t": 2.0,
        "full_model": 0,
        "thermal_symmetry": "eighth",
        "k_ins": 0.2,
    }
    profile = production._read_json(recovery.timeout12h.PROFILE_PATH)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    profile_path = source_dir / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    plan_path = source_dir / "plan.json"
    plan_path.write_text("{}\n", encoding="utf-8")
    submission_path = source_dir / "submission.json"
    submission_path.write_text("{}\n", encoding="utf-8")
    solver = "a" * 40
    library = "b" * 40
    source_name = "source-timeout12h"
    retained = scheduler_client.retained_aedt_identity(
        source_name, params, profile, solver, library
    )
    plan = {
        "schema_version": "source",
        "payload_sha256": recovery.SOURCE_PLAN_PAYLOAD_SHA256,
        "campaign_id": "mft-goal-20260726",
        "goal_contract_schema": recovery.GOAL_CONTRACT_SCHEMA,
        "hard_spec": recovery.GOAL_STAGE_SPEC,
        "hard_spec_sha256": recovery.GOAL_STAGE_SPEC_SHA256,
        "temperature_contract_sha256": (
            recovery.GOAL_TEMPERATURE_CONTRACT_SHA256
        ),
        "candidate_physics_sha256": (
            recovery.SOURCE_CANDIDATE_PHYSICS_SHA256
        ),
        "solver_revision": solver,
        "library_revision": library,
        "profile": {"path": profile_path.name},
        "stage": {
            "name": "standard",
            "task_name": source_name,
            "workdir": "source_timeout12h",
            "profile_sha256": production.canonical_sha256(profile),
            "effective_params_sha256": "c" * 64,
            "resources": {"cpus": 8, "timeout_seconds": 12 * 3600},
            "retained_aedt_bundle": retained,
            "retention_run_root": {"source": "source"},
            "scheduler_project": scheduler_client.MFT_PROJECT,
            "scheduler_url": recovery.probe.DIAGNOSTIC_SCHEDULER_URL,
            "aedt_backend": "standalone",
            "full_model": 0,
            "thermal_symmetry": "eighth",
        },
        "retry_of_timeout12h": {
            "logical_authority_task_id": recovery.SOURCE_LOGICAL_TASK_ID,
            "stream_evidence": {"fresh_grid_output_bytes": 28_982_733_864},
        },
        "scheduler_strict_node_contract": {
            "requested_node_name": recovery.STRICT_NODE_NAME,
            "node_name_policy": "strict",
        },
        "available_submission_commands": ["source"],
        "physics_override_allowed": False,
        "scheduler_repository_modified": False,
        "scheduler_project_mutation_performed": False,
        "scheduler_submission_performed": False,
        "retention_required": True,
        "prune_protection_required": True,
        **recovery._flags(),
    }
    submission = {
        "payload_sha256": recovery.SOURCE_SUBMISSION_PAYLOAD_SHA256,
        "task_id": recovery.SOURCE_FAILED_TASK_ID,
        "task_name": source_name,
        "dedupe_key": retained["dedupe_key"],
        "resources": {"cpus": 8, "timeout_seconds": 12 * 3600},
        "scheduler_url": recovery.probe.DIAGNOSTIC_SCHEDULER_URL,
    }
    return plan_path, submission_path, plan, submission, params, profile


def _failed_snapshot(source_submission):
    return {
        "task_id": recovery.SOURCE_FAILED_TASK_ID,
        "name": source_submission["task_name"],
        "status": "failed",
        "state": "failed",
        "exit_code": 1,
        "failure_message": recovery.EXPECTED_FAILURE_MESSAGE,
        "timeout_seconds": 12 * 3600,
        "cpus": 8,
        "memory_mb": recovery.SOURCE_MEMORY_MB,
        "aedt_backend": "standalone",
        "project": scheduler_client.MFT_PROJECT,
        "dedupe_key": source_submission["dedupe_key"],
        "requested_node_name": recovery.STRICT_NODE_NAME,
        "requested_node_name_policy": "strict",
        "strict_node_placement": True,
        "placement_contract_satisfied": True,
        "same_node_as_task_id": 0,
        "account_name": recovery.STORAGE_ACCOUNT_NAME,
        "actual_node_name": recovery.STRICT_NODE_NAME,
        "allocation_id": 14492,
        "slurm_job_id": "824575",
        "remote_cwd": "__SLURM_SCHEDULER_ACCOUNT_WORKSPACE__/runs",
        "remote_dir": "slurm_scheduler/runs/task-96302",
        "started_at": "2026-07-25 13:42:04",
        "finished_at": "2026-07-25 13:49:08",
    }


def _streams():
    dispatch = {
        "schema": "mft-solver-core-dispatch-v1",
        "stage": "loss",
        "backend": "standalone",
        "dispatch": "native_analyze_validated_dso",
        "cores_argument": 8,
        "gpus_argument": 0,
        "tasks_argument": 1,
    }
    stdout = recovery.DISPATCH_PREFIX + json.dumps(dispatch)
    stderr = "\n".join(
        [
            recovery.G3DMESHER_MARKER,
            recovery.MEMORY_HINT_MARKER,
            recovery.EXECUTION_ERROR_MARKER,
            recovery.LOSS_TRACE_MARKER,
        ]
    )
    return stdout, stderr


def _running_task(task_id=96298):
    return {
        "task_id": task_id,
        "name": f"active-{task_id}",
        "project": scheduler_client.MFT_PROJECT,
        "status": "running",
        "account_name": recovery.STORAGE_ACCOUNT_NAME,
        "actual_node_name": recovery.STRICT_NODE_NAME,
        "remote_dir": f"slurm_scheduler/runs/task-{task_id}",
    }


def test_plan_is_memory_only_4fac_pinned_and_post0(tmp_path, monkeypatch):
    (
        source_plan_path,
        source_submission_path,
        source_plan,
        source_submission,
        params,
        source_profile,
    ) = _source_bundle(tmp_path)
    selected = {"candidate": "exact"}
    monkeypatch.setattr(
        recovery,
        "_load_source",
        lambda *_args: (
            source_plan,
            source_submission,
            params,
            selected,
            source_profile,
        ),
    )
    monkeypatch.setattr(
        production,
        "_effective_params",
        lambda raw_params, profile: {
            **raw_params,
            **profile["param_overrides"],
        },
    )
    monkeypatch.setattr(
        recovery,
        "_claim_reference",
        lambda sha: {
            "candidate_physics_sha256": sha,
            "logical_authority_task_id": recovery.SOURCE_LOGICAL_TASK_ID,
            "retry_generation": recovery.RETRY_GENERATION,
        },
    )
    stdout, stderr = _streams()
    observed = datetime(2026, 7, 25, 22, 55, tzinfo=timezone(timedelta(hours=9)))
    plan_path = recovery.create_plan(
        source_plan_path=source_plan_path,
        source_submission_path=source_submission_path,
        output=tmp_path / "plan",
        storage_observed_at_kst=observed.isoformat(),
        storage_used_gb=100.0,
        storage_in_doubt_gb=6.0,
        storage_limit_gb=200.0,
        storage_observed_free_gb=94.0,
        running_remote_dir_bytes={96298: 43_822},
        task_reader=lambda **_kwargs: _failed_snapshot(source_submission),
        stdout_reader=lambda **_kwargs: stdout,
        stderr_reader=lambda **_kwargs: stderr,
        task_list_reader=lambda **_kwargs: [_running_task()],
        now=observed.astimezone(timezone.utc),
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    profile = json.loads(
        (plan_path.parent / plan["profile"]["path"]).read_text(encoding="utf-8")
    )
    retry = plan[recovery.RETRY_RECORD_FIELD]
    assert profile["mem_mb"] == recovery.TARGET_MEMORY_MB
    assert {
        key
        for key in set(source_profile) | set(profile)
        if source_profile.get(key) != profile.get(key)
    } == {"schema_version", "comment", "mem_mb"}
    assert plan["stage"]["resources"] == recovery.RESOURCES
    assert plan["stage"]["full_model"] == 0
    assert plan["stage"]["thermal_symmetry"] == "eighth"
    assert retry["memory_change_only"] is True
    assert retry["mesh_level_unchanged"] == 5
    assert retry["fan_velocity_m_s_unchanged"] == 1.5
    assert retry["storage_audit"]["safe_extra_submit_count"] == 0
    assert (
        plan["scheduler_strict_node_contract"]["pin_generation"]
        == "scheduler-strict-node-4fac-v4"
    )
    assert (
        plan["scheduler_strict_node_contract"]["scheduler_revision"]
        == recovery.probe.SCHEDULER_STRICT_NODE_REVISION
    )
    assert (
        plan["stage"]["retained_aedt_bundle"]["dedupe_key"]
        != source_plan["stage"]["retained_aedt_bundle"]["dedupe_key"]
    )
    assert (
        plan["stage"]["retained_aedt_bundle"]["relative_directory"]
        != source_plan["stage"]["retained_aedt_bundle"][
            "relative_directory"
        ]
    )

    monkeypatch.setattr(recovery, "load_plan", lambda _path: plan)
    refusal_path = recovery.write_submit_refusal(
        plan_path=plan_path, output=tmp_path / "submit-refusal.json"
    )
    refusal = production._validate_seal(
        json.loads(refusal_path.read_text(encoding="utf-8")),
        recovery.SUBMIT_REFUSAL_SCHEMA,
    )
    assert refusal["submit_allowed"] is False
    assert refusal["scheduler_post_calls"] == 0
    assert refusal["running_unbounded_task_ids"] == [96298]


def test_exact_failure_rejects_96300_class_and_any_result():
    source_submission = {
        "task_id": recovery.SOURCE_FAILED_TASK_ID,
        "task_name": "source",
        "dedupe_key": "dedupe",
    }
    snapshot = _failed_snapshot(source_submission)
    assert (
        recovery._task_failure_evidence(
            snapshot, source_submission=source_submission
        )["exit_code"]
        == 1
    )
    stdout, stderr = _streams()
    assert recovery._stream_evidence(stdout, stderr)["failure_stage"] == "loss"

    with pytest.raises(
        production.HandoffContractError, match="exact G3dMesher"
    ):
        recovery._stream_evidence(
            stdout,
            stderr.replace(
                recovery.MEMORY_HINT_MARKER,
                recovery.AMBIGUOUS_FILE_CLOSE_MARKER,
            ),
        )
    with pytest.raises(
        production.HandoffContractError, match="exact G3dMesher"
    ):
        recovery._stream_evidence(stdout + "\nRESULT_JSON: {}", stderr)
    wrong_memory = {**snapshot, "memory_mb": recovery.TARGET_MEMORY_MB}
    with pytest.raises(
        production.HandoffContractError, match="exact failed task-96302"
    ):
        recovery._task_failure_evidence(
            wrong_memory, source_submission=source_submission
        )


def test_claim_authority_is_separate_and_generation_bound(
    tmp_path, monkeypatch
):
    root = tmp_path / "g3dmesher-high-memory-claims"
    monkeypatch.setattr(recovery, "CLAIM_ROOT", root)
    authority = recovery.initialize_claim_root()
    reference = recovery._claim_reference(
        recovery.SOURCE_CANDIDATE_PHYSICS_SHA256
    )
    assert Path(authority["resolved_root"]) == root.resolve()
    assert "timeout12h_claims" not in str(root)
    assert (
        authority["campaign_authority_sha256"]
        == recovery.CLAIM_AUTHORITY_SHA256
    )
    assert reference["retry_generation"] == recovery.RETRY_GENERATION
    assert (
        reference["logical_authority_task_id"]
        == recovery.SOURCE_LOGICAL_TASK_ID
    )
