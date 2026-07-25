import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from regression_260707.verify import scheduler_client
from tools import mft_goal_file_close_failure_audit as audit
from tools import mft_goal_fea_handoff as production


def _streams(spec):
    dispatch = {
        "schema": "mft-solver-core-dispatch-v1",
        "stage": "loss",
        "backend": "standalone",
        "dispatch": "native_analyze_validated_dso",
        "cores_argument": 8,
        "gpus_argument": 0,
        "tasks_argument": 1,
    }
    workdir = (
        f"/enroot/mft_campaign-{spec.task_name.replace('-', '_')}-"
        "stest-ltest-ptest-ttest"
    )
    stdout = (
        f"MFT_WORKDIR {workdir}\n"
        f"{audit.DISPATCH_PREFIX}{json.dumps(dispatch, separators=(',', ':'))}\n"
    ).encode()
    stderr = "\n".join(
        [
            audit.FILE_CLOSE_MARKER,
            audit.EXECUTION_ERROR_MARKER,
            audit.SCRIPT_MACRO_MARKER,
            audit.LOSS_TRACE_MARKER,
            audit.EXPECTED_FAILURE_MESSAGE,
            "",
        ]
    ).encode()
    return stdout, stderr, workdir


def _script(spec, workdir):
    return (
        "#!/usr/bin/env bash\n"
        f"MFT_NVME_WORKDIR={workdir}; "
        'MFT_ENROOT_FREE_KB=$(df -Pk /enroot 2>/dev/null | '
        "awk 'NR==2 {print $4}'); "
        'if [ "$(findmnt -n -o FSTYPE -T /enroot 2>/dev/null)" = xfs ] '
        f'&& [ "${{MFT_ENROOT_FREE_KB:-0}}" -ge '
        f"{audit.ENROOT_ROUTE_THRESHOLD_KB} ]; then "
        "MFT_WORKDIR=$MFT_NVME_WORKDIR; fi; "
        "printf 'MFT_WORKDIR %s\\n' \"$MFT_WORKDIR\"; "
        'cleanup() { rm -rf -- "${MFT_NVME_WORKDIR}" '
        '"${MFT_GPFS_WORKDIR}" 2>/dev/null; }; '
        "python run_simulation_260706.py; simulation_rc=$?; "
        f'python retain.py "$MFT_TASK_ROOT/'
        f'{spec.retained_relative_directory}/symmetric.aedt"\n'
    ).encode()


def _bound_specs():
    result = []
    streams = {}
    scripts = {}
    for source in audit.SOURCE_SPECS:
        stdout, stderr, workdir = _streams(source)
        script = _script(source, workdir)
        spec = replace(
            source,
            expected_stdout_sha256=production._sha256_bytes(stdout),
            expected_stdout_size_bytes=len(stdout),
            expected_stderr_sha256=production._sha256_bytes(stderr),
            expected_stderr_size_bytes=len(stderr),
            expected_task_script_sha256=production._sha256_bytes(script),
            expected_task_script_size_bytes=len(script),
        )
        result.append(spec)
        streams[spec.failed_task_id] = {
            "stdout": stdout,
            "stderr": stderr,
        }
        scripts[spec.failed_task_id] = script
    return tuple(result), streams, scripts


def _task(spec):
    return {
        "task_id": spec.failed_task_id,
        "name": spec.task_name,
        "status": "failed",
        "state": "failed",
        "exit_code": 1,
        "failure_message": audit.EXPECTED_FAILURE_MESSAGE,
        "project": scheduler_client.MFT_PROJECT,
        "dedupe_key": spec.dedupe_key,
        "aedt_backend": "standalone",
        "cpus": audit.EXPECTED_CPUS,
        "memory_mb": audit.EXPECTED_MEMORY_MB,
        "timeout_seconds": audit.EXPECTED_TIMEOUT_SECONDS,
        "account_name": audit.EXPECTED_ACCOUNT,
        "requested_account_name": audit.EXPECTED_ACCOUNT,
        "requested_node_name": audit.EXPECTED_NODE,
        "requested_node_name_policy": "strict",
        "strict_node_placement": True,
        "placement_contract_satisfied": True,
        "actual_node_name": audit.EXPECTED_NODE,
        "allocation_id": audit.EXPECTED_ALLOCATION_ID,
        "slurm_job_id": audit.EXPECTED_SLURM_JOB_ID,
        "remote_cwd": "__SLURM_SCHEDULER_ACCOUNT_WORKSPACE__/runs",
        "remote_dir": (
            f"slurm_scheduler/runs/2026-07-25/task-"
            f"{spec.failed_task_id}-123456"
        ),
        "started_at": spec.expected_started_at,
        "finished_at": spec.expected_finished_at,
    }


def _events(specs):
    rows = []
    event_id = 2000
    for spec in specs:
        for kind, message in (
            (
                "task_requeued",
                f"task {spec.task_name} requeued after memory-pressure kill "
                "(attempt 1/3)",
            ),
            ("task_failed", f"task {spec.task_name} failed (exit 1)"),
            ("task_cleanup", f"cleaned task {spec.task_name}"),
        ):
            rows.append(
                {
                    "id": event_id,
                    "created_at": "2026-07-25 14:00:00",
                    "kind": kind,
                    "entity_type": "task",
                    "entity_id": str(spec.failed_task_id),
                    "account_name": audit.EXPECTED_ACCOUNT,
                    "message": message,
                }
            )
            event_id -= 1
    while len(rows) < audit.EVENT_LIMIT:
        rows.append(
            {
                "id": event_id,
                "created_at": "2026-07-25 13:00:00",
                "kind": "unrelated",
                "entity_type": "task",
                "entity_id": str(80000 + len(rows)),
                "account_name": "other",
                "message": "unrelated",
            }
        )
        event_id -= 1
    return rows


def _allocation():
    return {
        "id": audit.EXPECTED_ALLOCATION_ID,
        "account_name": audit.EXPECTED_ACCOUNT,
        "partition": "cpu2",
        "node_name": audit.EXPECTED_NODE,
        "slurm_job_id": audit.EXPECTED_SLURM_JOB_ID,
        "state": "active",
        "total_cpus": 64,
        "free_cpus": 64,
        "total_memory_mb": 870568,
        "free_memory_mb": 870568,
        "remote_dir": "runs/allocation",
        "failure_message": "",
        "drain_reason": "high CPU utilization",
        "created_at": "2026-07-24 14:15:06",
        "started_at": "2026-07-24 14:15:41",
        "last_active_at": "2026-07-25 14:30:00",
        "updated_at": "2026-07-25 14:30:00",
        "node_pestat_state": "mix",
        "node_cpu_used": 64,
        "node_cpu_total": 256,
        "node_memory_used_mb": 500000,
        "node_memory_free_mb": 500000,
        "node_memory_total_mb": 1000000,
        "node_metrics_observed_at": "2026-07-25 14:30:00",
    }


def test_stream_class_is_ambiguous_not_oom_or_physical():
    specs, streams, _scripts = _bound_specs()
    evidence = audit._stream_evidence(
        streams[specs[0].failed_task_id]["stdout"],
        streams[specs[0].failed_task_id]["stderr"],
        spec=specs[0],
    )
    assert evidence["failure_stage"] == "loss"
    assert evidence["result_json_absent"] is True
    assert evidence["g3dmesher_marker_absent"] is True
    assert evidence["oom_memory_hint_absent"] is True
    assert all(
        count == 0
        for count in evidence["explicit_io_errno_marker_counts"].values()
    )
    classification = audit._classification()
    assert classification["operational_io_or_storage_failure_suspected"]
    assert classification["operational_root_cause_authenticated"] is False
    assert classification["physical_failure_authenticated"] is False
    assert classification["retry_authorized"] is False

    with pytest.raises(
        production.HandoffContractError, match="exact ambiguous"
    ):
        audit._stream_evidence(
            streams[specs[0].failed_task_id]["stdout"],
            streams[specs[0].failed_task_id]["stderr"]
            + audit.G3DMESHER_MARKER.encode(),
            spec=specs[0],
        )
    with pytest.raises(
        production.HandoffContractError, match="exact ambiguous"
    ):
        audit._stream_evidence(
            streams[specs[0].failed_task_id]["stdout"]
            + b"RESULT_JSON {}\n",
            streams[specs[0].failed_task_id]["stderr"],
            spec=specs[0],
        )


def test_task_script_records_enroot_routing_but_no_failure_snapshot():
    specs, streams, scripts = _bound_specs()
    spec = specs[0]
    evidence = audit._task_script_evidence(
        scripts[spec.failed_task_id],
        spec=spec,
        stdout=streams[spec.failed_task_id]["stdout"],
    )
    assert evidence["solver_storage_scope"] == "compute-local-/enroot"
    assert evidence["routing_free_space_threshold_gib"] == 200
    assert evidence["failure_time_free_space_logged"] is False
    assert evidence["failure_time_inode_state_logged"] is False
    assert evidence["storage_device_identity_logged"] is False


def test_end_to_end_refusal_is_get_only_post0_cancel0(
    tmp_path, monkeypatch
):
    specs, streams, scripts = _bound_specs()
    monkeypatch.setattr(audit, "SOURCE_SPECS", specs)
    by_task = {spec.failed_task_id: spec for spec in specs}

    def fake_load_source(*, spec, plan_path, submission_path):
        del plan_path, submission_path
        submission = {
            "dedupe_key": spec.dedupe_key,
            "task_id": spec.failed_task_id,
        }
        normalized = {
            "logical_authority_task_id": spec.logical_task_id,
            "failed_task_id": spec.failed_task_id,
            "candidate_physics_sha256": spec.candidate_physics_sha256,
            "solver_revision": spec.solver_revision,
            "library_revision": audit.COMMON_LIBRARY_REVISION,
            "profile_sha256": audit.COMMON_PROFILE_SHA256,
            "effective_params_sha256": spec.effective_params_sha256,
            "plan": {"path": "plan"},
            "submission": {"path": "submission"},
            "task_name": spec.task_name,
            "dedupe_key": spec.dedupe_key,
        }
        return {}, submission, normalized

    monkeypatch.setattr(audit, "_load_source", fake_load_source)

    def remote_files_reader(*, task_id, glob, **_kwargs):
        spec = by_task[task_id]
        root = f"2026-07-25/task-{task_id}-123456"
        if glob == f"{root}/**":
            files = [
                f"{root}/{name}"
                for name in (
                    "exit_code",
                    "stderr.log",
                    "stdout.log",
                    "task.sh",
                    "wrapper.log",
                )
            ]
        elif glob == f"{root}/{spec.retained_relative_directory}/**":
            files = []
        else:
            raise AssertionError(glob)
        return {"base": "remote_cwd", "glob": glob, "files": files}

    def remote_file_reader(*, task_id, path, **_kwargs):
        if path.endswith("/task.sh"):
            return scripts[task_id]
        if path.endswith("/wrapper.log"):
            return b""
        if path.endswith("/exit_code"):
            return b"1\n"
        raise AssertionError(path)

    result = audit.create_no_retry_refusal(
        plan_paths=[tmp_path / "p0", tmp_path / "p1"],
        submission_paths=[tmp_path / "s0", tmp_path / "s1"],
        output=tmp_path / "no-retry.json",
        task_reader=lambda *, task_id, **_kwargs: _task(by_task[task_id]),
        stream_reader=lambda *, task_id, stream, **_kwargs: streams[task_id][
            stream
        ],
        event_reader=lambda **_kwargs: _events(specs),
        allocation_reader=lambda **_kwargs: [_allocation()],
        remote_files_reader=remote_files_reader,
        remote_file_reader=remote_file_reader,
        now=datetime(2026, 7, 25, 14, 40, tzinfo=timezone.utc),
    )
    refusal = production._validate_seal(
        json.loads(result.read_text(encoding="utf-8")),
        audit.AUDIT_SCHEMA,
    )
    assert refusal["classification"]["retry_authorized"] is False
    assert refusal["classification"]["same_payload_resubmission_allowed"] is False
    assert refusal["scope"]["excluded_task_ids"] == [96302]
    assert refusal["scope"]["task_96302_g3dmesher_oom_authority_mixed"] is False
    assert refusal["scheduler_http_methods_used"] == ["GET"]
    assert refusal["scheduler_get_count"] == 18
    assert refusal["scheduler_post_calls"] == 0
    assert refusal["scheduler_cancel_calls"] == 0
    assert refusal["successor_plan_written"] is False
    assert all(
        evidence["retained_bundle_present"] is False
        for evidence in refusal["retention_evidence"]
    )
