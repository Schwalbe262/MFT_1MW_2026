import importlib.util
import json
from pathlib import Path

import pytest

from tools import mft_campaign_atomic_claim as atomic_claim
from tools import mft_goal_diagnostic_standard_probe as probe
from tools import mft_goal_fea_handoff as production
from tools import mft_goal_timeout12h_retry as timeout12h


def _helpers():
    path = Path(__file__).with_name(
        "test_mft_goal_diagnostic_standard_probe.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_timeout12h_probe_test_helpers", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stream_bundle(*, elapsed_s=18_500.0):
    preflight = {
        "passed": True,
        "status": "passed_standalone_native_premesh",
        "mesh_mapping_coverage_passed": True,
        "native_operation_readback_passed": True,
        "standalone_idle_barrier_passed": True,
        "postflight_error": "",
        "native_errors": [],
        "elapsed_s": elapsed_s,
        "fresh_mesh_artifact_count": 1,
        "fresh_mesh_artifacts": [
            {"name": "mesh.sd", "grid_output_size": 28 * 1024**3}
        ],
        "mesh_policy": "b3-rxmain-l5-wcp-pad-padded-regions-v1",
        "native_operation_readback": {
            "operation_readbacks": [
                {
                    "name": "rx_side_block_mesh_level_TEST_L_5",
                    "level": 5,
                }
            ]
        },
    }
    solve = "Solving design setup ThermalSetup"
    stdout = f"setup complete\n{solve}\n"
    stderr = (
        "[thermal] native mesh preflight: "
        + json.dumps(preflight, sort_keys=True, separators=(",", ":"))
        + f"\n{solve}\n"
        + "srun: forcing job termination\n"
        + "slurmstepd-n114: error: STEP CANCELLED AT 2026-07-25T15:00:00\n"
    )
    return stdout.encode(), stderr.encode()


def _timeout8h_snapshot(submission):
    return {
        "task_id": 96256,
        "name": submission["task_name"],
        "status": "failed",
        "state": "failed",
        "exit_code": 124,
        "failure_message": "task timed out after 28800s",
        "timeout_seconds": 28800,
        "slurm_job_id": "824575",
        "allocation_id": 14492,
        "assigned_allocation": 14492,
        "account_name": "r1jae262",
        "actual_node_name": "n114",
        "cpus": 8,
        "memory_mb": 32768,
        "aedt_backend": "standalone",
        "project": "MFT_1MW_2026v1",
        "dedupe_key": submission["dedupe_key"],
        "remote_cwd": "__SLURM_SCHEDULER_ACCOUNT_WORKSPACE__/runs",
        "remote_dir": "slurm_scheduler/runs/task-96256",
        "started_at": "2026-07-24 22:27:32",
        "finished_at": "2026-07-25 06:28:22",
    }


def _strict_task(plan, task_id=96301):
    return {
        "task_id": task_id,
        "id": task_id,
        "name": plan["stage"]["task_name"],
        "status": "running",
        "state": "running",
        "exit_code": None,
        "failure_message": "",
        "created_at": "2026-07-25 08:10:00",
        "attached_at": "2026-07-25 08:10:01",
        "launch_started_at": "2026-07-25 08:10:02",
        "started_at": "2026-07-25 08:10:03",
        "finished_at": None,
        "assigned_allocation": 14492,
        "allocation_id": 14492,
        "account_name": "r1jae262",
        "requested_account_name": "",
        "actual_node_name": "n114",
        "allocation_node_name": "n114",
        "slurm_job_id": "824575",
        "cpus": 8,
        "memory_mb": 32768,
        "timeout_seconds": 43200,
        "aedt_backend": "standalone",
        "project": "MFT_1MW_2026v1",
        "dedupe_key": plan["stage"]["retained_aedt_bundle"][
            "dedupe_key"
        ],
        "scheduling_profile": "fea_bursty",
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "requested_node_name": "n114",
        "node_name": "n114",
        "requested_node_name_policy": "strict",
        "node_name_policy": "strict",
        "strict_node_placement": True,
        "placement_contract_satisfied": True,
        "same_node_as_task_id": 0,
        "remote_cwd": "__SLURM_SCHEDULER_ACCOUNT_WORKSPACE__/runs",
        "remote_dir": "slurm_scheduler/runs/task-96301",
    }


def _setup_immediate_timeout(tmp_path, monkeypatch, helpers):
    fixture = helpers._fixture(tmp_path, monkeypatch)
    original_plan_path = helpers._make_plan(tmp_path, fixture)
    cutover_path = helpers._scheduler_cutover(tmp_path, monkeypatch)

    class OriginalScheduler(helpers._FakeScheduler):
        def submit_verification(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return 96218

    original_submission_path = probe.submit_standard(
        plan_path=original_plan_path,
        scheduler_cutover_receipt_path=cutover_path,
        output=tmp_path / "original-submission.json",
        scheduler=OriginalScheduler(),
        predictor=helpers._Predictor(),
        live_reader=helpers._live_scheduler_reader,
    )
    original_plan = probe._load_plan(original_plan_path)[0]
    original_submission = probe._load_submission(
        original_submission_path, plan=original_plan
    )
    immediate_plan_path = probe.create_timeout_retry_plan(
        original_plan_path=original_plan_path,
        original_submission_path=original_submission_path,
        output=tmp_path / "timeout8h-plan",
        task_reader=lambda **_kwargs: helpers._timeout_task_snapshot(
            original_submission
        ),
    )

    class TimeoutScheduler(helpers._FakeScheduler):
        def submit_verification(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return 96256

    immediate_submission_path = probe.submit_timeout_retry(
        plan_path=immediate_plan_path,
        scheduler_cutover_receipt_path=cutover_path,
        output=tmp_path / "timeout8h-submission.json",
        scheduler=TimeoutScheduler(),
        predictor=helpers._Predictor(),
        live_reader=helpers._live_scheduler_reader,
        task_reader=lambda **_kwargs: helpers._timeout_task_snapshot(
            original_submission
        ),
    )
    immediate_plan = probe._load_plan(immediate_plan_path)[0]
    immediate_submission = probe._load_submission(
        immediate_submission_path, plan=immediate_plan
    )
    return immediate_plan_path, immediate_submission_path, immediate_submission


def _create_timeout12h_plan(
    tmp_path, monkeypatch, helpers, immediate_plan, immediate_submission
):
    claim_root = (tmp_path / "timeout12h-claims").resolve()
    monkeypatch.setattr(timeout12h, "CLAIM_ROOT", claim_root)
    atomic_claim.initialize_claim_root(
        claim_root,
        campaign_id="mft-goal-20260726",
        campaign_authority_sha256=timeout12h.CLAIM_AUTHORITY_SHA256,
        root_id="7" * 32,
        now="2026-07-25T08:00:00Z",
    )
    stdout, stderr = _stream_bundle()
    return timeout12h.create_plan(
        immediate_plan_path=immediate_plan,
        immediate_submission_path=immediate_submission,
        strict_node_name="n114",
        storage_observed_at_kst="2026-07-25T17:08:00+09:00",
        storage_used_gb=100.5357,
        storage_in_doubt_gb=5.2039,
        storage_limit_gb=200.0,
        storage_observed_free_gb=94.2604,
        output=tmp_path / "timeout12h-plan",
        task_reader=lambda **_kwargs: _timeout8h_snapshot(
            probe._load_submission(
                immediate_submission,
                plan=probe._load_plan(immediate_plan)[0],
            )
        ),
        stdout_reader=lambda **_kwargs: stdout,
        stderr_reader=lambda **_kwargs: stderr,
        remote_files_reader=lambda **kwargs: {
            "base": "remote_cwd",
            "glob": kwargs["glob"],
            "files": [],
        },
    )


def test_timeout12h_plan_is_exact_timeout_only_and_probe_loadable(
    tmp_path, monkeypatch
):
    helpers = _helpers()
    immediate_plan, immediate_submission, _receipt = (
        _setup_immediate_timeout(tmp_path, monkeypatch, helpers)
    )
    plan_path = _create_timeout12h_plan(
        tmp_path,
        monkeypatch,
        helpers,
        immediate_plan,
        immediate_submission,
    )
    plan, params, _selected = probe._load_plan(plan_path)
    immediate, immediate_params, _ = probe._load_plan(immediate_plan)
    profile = production._read_json(
        plan_path.parent / plan["profile"]["path"]
    )
    immediate_profile = production._read_json(
        immediate_plan.parent / immediate["profile"]["path"]
    )
    assert params == immediate_params
    assert plan["stage"]["resources"] == {
        "cpus": 8,
        "timeout_seconds": 43200,
    }
    assert plan["retry_of_timeout12h"]["logical_authority_task_id"] == 96218
    assert plan["retry_of_timeout12h"]["retry_of_task_id"] == 96256
    assert plan["scheduler_strict_node_contract"][
        "requested_node_name"
    ] == "n114"
    assert profile["param_overrides"] == immediate_profile["param_overrides"]
    assert (
        profile["fixed_boundary_contract"]
        == immediate_profile["fixed_boundary_contract"]
    )
    assert plan["retry_of_timeout12h"]["storage_audit"][
        "parallel_submit_without_reaudit_allowed"
    ] is False


def test_timeout12h_supplemental_authority_isolated_from_frozen_r1(
    tmp_path, monkeypatch
):
    helpers = _helpers()
    immediate_plan, immediate_submission, _receipt = (
        _setup_immediate_timeout(tmp_path, monkeypatch, helpers)
    )
    supplemental_root = (tmp_path / "timeout12h-claims-r2").resolve()
    monkeypatch.setattr(timeout12h, "EXACT_LOGICAL_TO_FAILED_TASK", {})
    monkeypatch.setattr(
        timeout12h,
        "SUPPLEMENTAL_EXACT_LOGICAL_TO_FAILED_TASK",
        {96218: 96256},
    )
    monkeypatch.setattr(
        timeout12h, "SUPPLEMENTAL_CLAIM_ROOT", supplemental_root
    )
    atomic_claim.initialize_claim_root(
        supplemental_root,
        campaign_id="mft-goal-20260726",
        campaign_authority_sha256=(
            timeout12h.SUPPLEMENTAL_CLAIM_AUTHORITY_SHA256
        ),
        root_id="9" * 32,
        now="2026-07-25T13:00:00Z",
    )
    stdout, stderr = _stream_bundle()
    plan_path = timeout12h.create_plan(
        immediate_plan_path=immediate_plan,
        immediate_submission_path=immediate_submission,
        strict_node_name="n114",
        storage_observed_at_kst="2026-07-25T22:00:00+09:00",
        storage_used_gb=100.2698,
        storage_in_doubt_gb=5.2039,
        storage_limit_gb=200.0,
        storage_observed_free_gb=94.5263,
        output=tmp_path / "timeout12h-r2-plan",
        task_reader=lambda **_kwargs: _timeout8h_snapshot(
            probe._load_submission(
                immediate_submission,
                plan=probe._load_plan(immediate_plan)[0],
            )
        ),
        stdout_reader=lambda **_kwargs: stdout,
        stderr_reader=lambda **_kwargs: stderr,
        remote_files_reader=lambda **kwargs: {
            "base": "remote_cwd",
            "glob": kwargs["glob"],
            "files": [],
        },
    )
    plan = probe._load_plan(plan_path)[0]
    record = plan["retry_of_timeout12h"]
    assert (
        record["retry_generation"]
        == timeout12h.SUPPLEMENTAL_RETRY_GENERATION
    )
    assert "-timeout12h-r2-l96218-" in plan["stage"]["task_name"]
    assert plan["timeout12h_atomic_claim_reference"][
        "retry_generation"
    ] == timeout12h.SUPPLEMENTAL_RETRY_GENERATION
    assert supplemental_root.exists()


def test_timeout12h_late_anchor_r3_isolated_and_accepts_two_hour_premesh():
    authority = timeout12h._retry_authority(
        timeout12h.LATE_ANCHOR_RETRY_GENERATION
    )
    assert authority["claim_root"] == timeout12h.LATE_ANCHOR_CLAIM_ROOT
    assert authority["exact_logical_to_failed_task"] == {96223: 96289}
    stdout, stderr = _stream_bundle(elapsed_s=10_171.31)
    with pytest.raises(
        production.HandoffContractError,
        match="successful premesh",
    ):
        timeout12h._stream_evidence(
            stdout,
            stderr,
            task_id=96289,
            retry_generation=timeout12h.SUPPLEMENTAL_RETRY_GENERATION,
        )
    evidence = timeout12h._stream_evidence(
        stdout,
        stderr,
        task_id=96289,
        retry_generation=timeout12h.LATE_ANCHOR_RETRY_GENERATION,
    )
    assert evidence["native_premesh_elapsed_seconds"] == 10_171.31
    assert evidence["minimum_native_premesh_elapsed_seconds"] == 7200
    assert evidence["thermal_solve_dispatched"] is True


def test_timeout12h_late_boundary_r4_has_single_exact_authority():
    authority = timeout12h._retry_authority(
        timeout12h.LATE_BOUNDARY_RETRY_GENERATION
    )
    assert authority["claim_root"] == (
        timeout12h.LATE_BOUNDARY_CLAIM_ROOT
    )
    assert authority["exact_logical_to_failed_task"] == {96230: 96265}
    selected = timeout12h._retry_authority_for_pair(
        logical_authority_task_id=96230,
        failed_task_id=96265,
    )
    assert selected["retry_generation"] == (
        timeout12h.LATE_BOUNDARY_RETRY_GENERATION
    )


def test_timeout12h_atomic_submit_and_submission_dispatch(
    tmp_path, monkeypatch
):
    helpers = _helpers()
    immediate_plan, immediate_submission, immediate_receipt = (
        _setup_immediate_timeout(tmp_path, monkeypatch, helpers)
    )
    strict_cutover = helpers._strict_scheduler_cutover(
        tmp_path, monkeypatch
    )
    plan_path = _create_timeout12h_plan(
        tmp_path,
        monkeypatch,
        helpers,
        immediate_plan,
        immediate_submission,
    )
    plan = probe._load_plan(plan_path)[0]
    new_task = _strict_task(plan)
    submitted = {"value": False}

    class StrictScheduler:
        def submit_verification(self, *args, **kwargs):
            assert kwargs["node_name"] == "n114"
            assert kwargs["node_name_policy"] == "strict"
            assert kwargs.get("same_node_as_task_id", 0) == 0
            kwargs["pre_submit_guard"]()
            submitted["value"] = True
            return {
                "task_id": new_task["task_id"],
                "submission_source": "api_post_submission",
                "scheduler_mutation_performed": True,
                "api_pre_submission_readback": None,
                "api_post_submission_response": new_task,
            }

    stdout, stderr = _stream_bundle()

    def task_reader(**kwargs):
        if kwargs["task_id"] == 96256:
            return _timeout8h_snapshot(immediate_receipt)
        if kwargs["task_id"] == new_task["task_id"]:
            return new_task
        raise AssertionError(kwargs)

    def task_list_reader(**_kwargs):
        return [new_task] if submitted["value"] else []

    submission_path = timeout12h.submit(
        plan_path=plan_path,
        scheduler_cutover_receipt_path=strict_cutover,
        output=tmp_path / "timeout12h-submission.json",
        scheduler=StrictScheduler(),
        predictor=helpers._Predictor(),
        live_reader=helpers._live_scheduler_reader,
        task_reader=task_reader,
        stdout_reader=lambda **_kwargs: stdout,
        stderr_reader=lambda **_kwargs: stderr,
        remote_files_reader=lambda **kwargs: {
            "base": "remote_cwd",
            "glob": kwargs["glob"],
            "files": [],
        },
        task_list_reader=task_list_reader,
        reconciliation_waiter=lambda: None,
    )
    loaded = probe._load_submission(submission_path, plan=plan)
    assert loaded["task_id"] == 96301
    assert loaded["scheduler_strict_node_contract"][
        "same_node_as_task_id"
    ] == 0
    assert loaded["timeout12h_sibling_guard"][
        "exactly_one_sibling_after_submission"
    ] is True
    assert loaded["timeout12h_atomic_claim"][
        "fresh_claim_authorized_scheduler_submit_call"
    ] is True
    collector_scheduler = helpers._FakeScheduler()
    collector_scheduler.result = helpers._result(plan_path, loaded)
    metadata_reader, manifest_reader = helpers._remote_evidence(
        loaded, collector_scheduler.result
    )
    terminal = {
        **new_task,
        "status": "completed",
        "state": "succeeded",
        "exit_code": 0,
        "failure_message": "",
        "finished_at": "2026-07-25 17:00:00",
    }
    collection_path = probe.collect_standard(
        plan_path=plan_path,
        submission_path=submission_path,
        output=tmp_path / "collection.json",
        scheduler=collector_scheduler,
        remote_reader=metadata_reader,
        manifest_reader=manifest_reader,
        task_reader=lambda **_kwargs: terminal,
    )
    collection = production._read_json(collection_path)
    assert collection["task_id"] == 96301
    assert collection["scheduler_get_only_collection"] is True
    assert probe._load_collection(
        collection_path, predictor=helpers._Predictor()
    )[1][
        "schema_version"
    ] == timeout12h.PLAN_SCHEMA


def test_timeout12h_rejects_physics_or_wrong_exact_task(
    tmp_path, monkeypatch
):
    helpers = _helpers()
    immediate_plan, immediate_submission, immediate_receipt = (
        _setup_immediate_timeout(tmp_path, monkeypatch, helpers)
    )
    claim_root = (tmp_path / "timeout12h-claims").resolve()
    monkeypatch.setattr(timeout12h, "CLAIM_ROOT", claim_root)
    atomic_claim.initialize_claim_root(
        claim_root,
        campaign_id="mft-goal-20260726",
        campaign_authority_sha256=timeout12h.CLAIM_AUTHORITY_SHA256,
        root_id="8" * 32,
        now="2026-07-25T08:00:00Z",
    )
    stdout, stderr = _stream_bundle()
    wrong = _timeout8h_snapshot(immediate_receipt)
    wrong["task_id"] = 96263
    with pytest.raises(
        production.HandoffContractError,
        match="exact failed/124/28800s",
    ):
        timeout12h.create_plan(
            immediate_plan_path=immediate_plan,
            immediate_submission_path=immediate_submission,
            strict_node_name="n114",
            storage_observed_at_kst="2026-07-25T17:08:00+09:00",
            storage_used_gb=100.5357,
            storage_in_doubt_gb=5.2039,
            storage_limit_gb=200.0,
            storage_observed_free_gb=94.2604,
            output=tmp_path / "forbidden-plan",
            task_reader=lambda **_kwargs: wrong,
            stdout_reader=lambda **_kwargs: stdout,
            stderr_reader=lambda **_kwargs: stderr,
            remote_files_reader=lambda **kwargs: {
                "base": "remote_cwd",
                "glob": kwargs["glob"],
                "files": [],
            },
        )
