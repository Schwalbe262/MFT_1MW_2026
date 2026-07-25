"""Read-only audit for the task-96300/task-96301 file-close failures.

The two exact diagnostic Standard tasks reached the Maxwell loss solve and
then emitted the same ``Error closing file: %1`` terminal signature.  That
signature is correlated with a shared n114 allocation and compute-local
``/enroot`` routing, but it does not contain an errno, a target path, or a
failure-time filesystem snapshot.  This module therefore creates immutable
POST0/no-retry evidence.  It has no Scheduler mutation or submission path.

Task 96302 is intentionally outside this audit because its G3dMesher
OOM-like signature has a separate recovery authority.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Callable, Mapping, Sequence

from regression_260707.verify import scheduler_client
from tools import mft_goal_diagnostic_standard_probe as probe
from tools import mft_goal_fea_handoff as production


AUDIT_SCHEMA = "mft-goal-file-close-no-retry-refusal-v1"
SOURCE_PLAN_SCHEMA = "mft-goal-diagnostic-standard-plan-v1"
SOURCE_SUBMISSION_SCHEMA = "mft-goal-diagnostic-standard-submission-v1"
STREAM_SCHEMA = "mft-goal-file-close-terminal-stream-v1"
TASK_SCHEMA = "mft-goal-file-close-terminal-task-v1"
TASK_SCRIPT_SCHEMA = "mft-goal-file-close-task-script-storage-routing-v1"
RETENTION_SCHEMA = "mft-goal-file-close-retention-absence-v1"
EVENT_SCHEMA = "mft-goal-file-close-event-window-v1"
ALLOCATION_SCHEMA = "mft-goal-file-close-allocation-snapshot-v1"

SCHEDULER_URL = probe.DIAGNOSTIC_SCHEDULER_URL
SCHEDULER_PROJECT = scheduler_client.MFT_PROJECT
EXPECTED_ACCOUNT = "r1jae262"
EXPECTED_NODE = "n114"
EXPECTED_ALLOCATION_ID = 14492
EXPECTED_SLURM_JOB_ID = "824575"
EXPECTED_CPUS = 8
EXPECTED_MEMORY_MB = 32768
EXPECTED_TIMEOUT_SECONDS = 8 * 3600
EXCLUDED_TASK_IDS = (96302,)

MAX_STREAM_BYTES = 64 * 1024 * 1024
MAX_TASK_SCRIPT_BYTES = 1024 * 1024
MAX_REMOTE_FILE_BYTES = 64 * 1024 * 1024
MAX_EVENT_BYTES = 16 * 1024 * 1024
MAX_ALLOCATION_BYTES = 16 * 1024 * 1024
EVENT_LIMIT = 1000
ENROOT_ROUTE_THRESHOLD_KB = 200 * 1024 * 1024

EXPECTED_FAILURE_MESSAGE = (
    "ansys.aedt.core.internal.errors.GrpcApiError: "
    "Failed to execute gRPC AEDT command: Analyze"
)
FILE_CLOSE_MARKER = "Error closing file: %1"
EXECUTION_ERROR_MARKER = (
    "Simulation completed with execution error on server: n114."
)
SCRIPT_MACRO_MARKER = "Setup1 has failed with execution error."
G3DMESHER_MARKER = "Process 'G3dMesher' terminated abnormally."
MEMORY_HINT_MARKER = (
    "It may have run out of memory or could have been killed by the user."
)
RESULT_MARKER = "RESULT_JSON"
LOSS_TRACE_MARKER = 't_loss = sim.analyze_and_extract("loss"'
DISPATCH_PREFIX = "SOLVER_CORE_DISPATCH_JSON "
EXPLICIT_IO_ERRNO_MARKERS = (
    "No space left on device",
    "Disk quota exceeded",
    "Input/output error",
    "Read-only file system",
    "ENOSPC",
    "EDQUOT",
    "EIO",
)


@dataclass(frozen=True)
class SourceSpec:
    logical_task_id: int
    failed_task_id: int
    candidate_physics_sha256: str
    solver_revision: str
    plan_payload_sha256: str
    submission_payload_sha256: str
    plan_file_sha256: str
    submission_file_sha256: str
    task_name: str
    dedupe_key: str
    effective_params_sha256: str
    retained_relative_directory: str
    retained_marker_path: str
    retained_marker_contract_sha256: str
    expected_stdout_sha256: str
    expected_stdout_size_bytes: int
    expected_stderr_sha256: str
    expected_stderr_size_bytes: int
    expected_task_script_sha256: str
    expected_task_script_size_bytes: int
    expected_started_at: str
    expected_finished_at: str


SOURCE_SPECS = (
    SourceSpec(
        logical_task_id=96225,
        failed_task_id=96300,
        candidate_physics_sha256=(
            "7a6ccac265d3b0d4ecc585837fe04173dab226d93d15d61ade677e3dd6828f3a"
        ),
        solver_revision="f02356e718e3b094c0bad4370462860e788f46bc",
        plan_payload_sha256=(
            "ccdca40658d1282ff05fd3f04924e9dbd82e99f0bf1804a6c02397863c6c785d"
        ),
        submission_payload_sha256=(
            "825fba34f72c6a425ccb30ee55641b34b0563f1f97c0e7b7ab7457ef2e01249f"
        ),
        plan_file_sha256=(
            "84384c8d0eec4cf0e43152c5c7fa92de6adb43091467e106b7e2b31842650654"
        ),
        submission_file_sha256=(
            "95dfca8fa6fb0baaaa65bec75ffaa59905239c49b3f08d662acb666f08621f99"
        ),
        task_name=(
            "mft-goal-diag-standard-mesh-canary-r1-"
            "l96225-7a6ccac265d3"
        ),
        dedupe_key=(
            "mft-al:mft-goal-diag-standard-mesh-canary-r1-"
            "l96225-7a6ccac265d3:"
            "f02356e718e3b094c0bad4370462860e788f46bc:"
            "e6b9b9d20a832ff5c3f7ca97218737a0b8650781:"
            "8b0b9947f7e93c8a"
        ),
        effective_params_sha256=(
            "8b0b9947f7e93c8a45e70adbedc7f3e26552b1748ec248eb07e232c716856bf5"
        ),
        retained_relative_directory="goal-fea-retained/90c90506f7237c66",
        retained_marker_path=(
            "goal-fea-retained/90c90506f7237c66/"
            ".slurm-scheduler-preserve.json"
        ),
        retained_marker_contract_sha256=(
            "1beb61f40b4af67df4c9be6c1e8f596424cc3dc1d37b960965381022f9e85bdc"
        ),
        expected_stdout_sha256=(
            "85e93f0d803897f05ac827ec6f8886b6db15a3852edd11d7cb5e8c4f098621e1"
        ),
        expected_stdout_size_bytes=17610,
        expected_stderr_sha256=(
            "1704f70d620c06a259314af37c56b0f8f98d68b650772d5b939df3bf1fe68d6d"
        ),
        expected_stderr_size_bytes=16503,
        expected_task_script_sha256=(
            "a8ef99f14432dbca718341c0b5fffd86af0310bb6a1b0ddc9267756d482ab125"
        ),
        expected_task_script_size_bytes=14090,
        expected_started_at="2026-07-25 13:42:05",
        expected_finished_at="2026-07-25 13:52:57",
    ),
    SourceSpec(
        logical_task_id=96220,
        failed_task_id=96301,
        candidate_physics_sha256=(
            "05580bda40b279fdbd7856639419732903323fd3964e5fe110fa0acf2aba3b39"
        ),
        solver_revision="010fb4644bf48da69c3a221b12d083428dd15dcb",
        plan_payload_sha256=(
            "7ad5a5d60c32b11aef4ad89dd52edf8f01110b60972d4998ecff322848d7e3ff"
        ),
        submission_payload_sha256=(
            "e47180423996d530085feaa5f475ea95ec9a194f1f6f1b87e4afd68476bb7304"
        ),
        plan_file_sha256=(
            "65c5417f33ac006f2ab46bbb926f9d98c212d8a7fd508dabc6b862891a782866"
        ),
        submission_file_sha256=(
            "c814c0b0fd2f072bde9054641dec7fd05a807b3733eeff794d65c96be70f5771"
        ),
        task_name=(
            "mft-goal-diag-standard-mesh-canary-r1-"
            "l96220-05580bda40b2"
        ),
        dedupe_key=(
            "mft-al:mft-goal-diag-standard-mesh-canary-r1-"
            "l96220-05580bda40b2:"
            "010fb4644bf48da69c3a221b12d083428dd15dcb:"
            "e6b9b9d20a832ff5c3f7ca97218737a0b8650781:"
            "c36b4ee4a75f4162"
        ),
        effective_params_sha256=(
            "c36b4ee4a75f416279ff07d8cc952c295ae34572b63a86c44c5e0f98974dc775"
        ),
        retained_relative_directory="goal-fea-retained/8ebe9a5a4d477588",
        retained_marker_path=(
            "goal-fea-retained/8ebe9a5a4d477588/"
            ".slurm-scheduler-preserve.json"
        ),
        retained_marker_contract_sha256=(
            "3759b6fb93dd16d4d20267df24ea3a2b2a3e2093c75411075b3c1a7298260175"
        ),
        expected_stdout_sha256=(
            "443cbcb146921da94d8167c128b2efe52605eb01daa76805cf40ab102fb98f06"
        ),
        expected_stdout_size_bytes=17615,
        expected_stderr_sha256=(
            "8388168e418392817e97c2d9edffbca9560153b51f289eb653259f1477b4c7f5"
        ),
        expected_stderr_size_bytes=16503,
        expected_task_script_sha256=(
            "117a3da8b91a3adba5eaf2e5130893c4e54f1cc49881fac167824975de61f9c9"
        ),
        expected_task_script_size_bytes=14090,
        expected_started_at="2026-07-25 13:42:04",
        expected_finished_at="2026-07-25 14:15:43",
    ),
)

COMMON_LIBRARY_REVISION = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
COMMON_PROFILE_SHA256 = (
    "a60f0d4aef4a77bc2751628e9c064b229bd46bc237ce687b3d10d7488df160f4"
)
COMMON_HARD_SPEC_SHA256 = (
    "ce1302bd19f3f0dcfda2b86c2906530da95841ac6fa64f7d7e2c59a50468bbbd"
)
COMMON_TEMPERATURE_SHA256 = (
    "484258f61430d59474c15c72d0804012315e4700312f878a32bbaf492ab0cb01"
)

HandoffContractError = production.HandoffContractError


def _task_id(value: Mapping[str, Any]) -> Any:
    return value.get("task_id", value.get("id"))


def _http_bytes(url: str, *, max_bytes: int, accept: str) -> bytes:
    request = production.urllib.request.Request(
        url,
        headers={"Accept": accept},
        method="GET",
    )
    try:
        with production.urllib.request.urlopen(
            request, timeout=120.0
        ) as response:
            raw = response.read(max_bytes + 1)
    except (OSError, production.urllib.error.URLError) as exc:
        raise HandoffContractError(
            f"file-close audit Scheduler GET failed: {url}"
        ) from exc
    if len(raw) > max_bytes:
        raise HandoffContractError(
            f"file-close audit Scheduler response exceeds bound: {url}"
        )
    return raw


def _http_json(url: str, *, max_bytes: int) -> Any:
    raw = _http_bytes(url, max_bytes=max_bytes, accept="application/json")
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise HandoffContractError(
            f"file-close audit Scheduler JSON is invalid: {url}"
        ) from exc


def _scheduler_task(*, scheduler_url: str, task_id: int) -> dict[str, Any]:
    value = _http_json(
        f"{scheduler_url.rstrip('/')}/api/tasks/{task_id}",
        max_bytes=1024 * 1024,
    )
    if not isinstance(value, dict):
        raise HandoffContractError("Scheduler task response is malformed")
    return value


def _scheduler_stream(
    *, scheduler_url: str, task_id: int, stream: str
) -> bytes:
    if stream not in {"stdout", "stderr"}:
        raise HandoffContractError("file-close stream name is invalid")
    return _http_bytes(
        f"{scheduler_url.rstrip('/')}/api/tasks/{task_id}/{stream}",
        max_bytes=MAX_STREAM_BYTES,
        accept="text/plain",
    )


def _scheduler_events(*, scheduler_url: str) -> list[dict[str, Any]]:
    value = _http_json(
        f"{scheduler_url.rstrip('/')}/api/events?limit={EVENT_LIMIT}",
        max_bytes=MAX_EVENT_BYTES,
    )
    if not isinstance(value, list) or not all(
        isinstance(item, dict) for item in value
    ):
        raise HandoffContractError("Scheduler event response is malformed")
    return value


def _scheduler_allocations(*, scheduler_url: str) -> list[dict[str, Any]]:
    value = _http_json(
        f"{scheduler_url.rstrip('/')}/api/allocations",
        max_bytes=MAX_ALLOCATION_BYTES,
    )
    if not isinstance(value, list) or not all(
        isinstance(item, dict) for item in value
    ):
        raise HandoffContractError(
            "Scheduler allocation response is malformed"
        )
    return value


def _scheduler_remote_files(
    *, scheduler_url: str, task_id: int, glob: str
) -> dict[str, Any]:
    query = production.urllib.parse.urlencode(
        {"glob": glob, "base": "remote_cwd"}
    )
    value = _http_json(
        f"{scheduler_url.rstrip('/')}/api/tasks/{task_id}/remote-files?"
        f"{query}",
        max_bytes=MAX_ALLOCATION_BYTES,
    )
    if not isinstance(value, dict):
        raise HandoffContractError(
            "Scheduler remote-file inventory is malformed"
        )
    return value


def _scheduler_remote_file(
    *,
    scheduler_url: str,
    task_id: int,
    path: str,
    max_bytes: int,
) -> bytes:
    pure = PurePosixPath(path)
    if (
        not path
        or pure.is_absolute()
        or ".." in pure.parts
        or max_bytes <= 0
        or max_bytes > MAX_REMOTE_FILE_BYTES
    ):
        raise HandoffContractError("unsafe file-close remote-file request")
    query = production.urllib.parse.urlencode(
        {"path": path, "base": "remote_cwd", "max_bytes": max_bytes}
    )
    return _http_bytes(
        f"{scheduler_url.rstrip('/')}/api/tasks/{task_id}/remote-file?"
        f"{query}",
        max_bytes=max_bytes,
        accept="application/octet-stream",
    )


def _verified_file(
    base: Path, record: Mapping[str, Any], label: str
) -> Path:
    if not isinstance(record, Mapping):
        raise HandoffContractError(f"{label} file record is absent")
    path = production._contained_file(base, record.get("path"), label)
    if production._sha256_file(path) != record.get("sha256"):
        raise HandoffContractError(f"{label} SHA drifted")
    return path


def _validate_fixed_physics(
    plan: Mapping[str, Any], params: Mapping[str, Any], profile: Mapping[str, Any]
) -> None:
    hard_spec = plan.get("hard_spec")
    cooling = (
        hard_spec.get("fixed_cooling_identity")
        if isinstance(hard_spec, Mapping)
        else None
    )
    overrides = profile.get("param_overrides")
    fixed = profile.get("fixed_boundary_contract")
    effective = production._effective_params(dict(params), dict(profile))
    expected_effective = {
        "fan_velocity": 1.5,
        "fan_config": "dual",
        "k_ins": 0.2,
        "core_plate_pad_t": 2.0,
        "wcp_pad_t": 2.0,
        "full_model": 0,
        "thermal_symmetry": "eighth",
        "thermal_rx_side_block_mesh_level": 4,
    }
    expected_fixed = {
        "fan_velocity_m_s": 1.5,
        "fan_config": "dual",
        "thermal_pad_conductivity_W_mK": 0.2,
        "core_plate_pad_t_mm": 2.0,
        "wcp_pad_t_mm": 2.0,
    }
    expected_cooling = {
        "fan_velocity": 1.5,
        "fan_config": "dual",
        "k_ins": 0.2,
        "thermal_pad_conductivity_W_mK": 0.2,
        "core_plate_pad_t": 2.0,
        "wcp_pad_t": 2.0,
    }
    if (
        not isinstance(overrides, Mapping)
        or not isinstance(fixed, Mapping)
        or not isinstance(cooling, Mapping)
        or any(
            effective.get(name) != value
            for name, value in expected_effective.items()
        )
        or any(fixed.get(name) != value for name, value in expected_fixed.items())
        or any(
            cooling.get(name) != value
            for name, value in expected_cooling.items()
        )
        or profile.get("mem_mb") != EXPECTED_MEMORY_MB
        or profile.get("cpus") != EXPECTED_CPUS
        or profile.get("timeout_seconds") != EXPECTED_TIMEOUT_SECONDS
        or plan.get("hard_spec_sha256") != COMMON_HARD_SPEC_SHA256
        or plan.get("temperature_contract_sha256")
        != COMMON_TEMPERATURE_SHA256
    ):
        raise HandoffContractError(
            "file-close source fixed fan/TIM/pad/physics identity drifted"
        )


def _load_source(
    *,
    spec: SourceSpec,
    plan_path: Path,
    submission_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    resolved_plan = plan_path.resolve(strict=True)
    resolved_submission = submission_path.resolve(strict=True)
    if (
        production._sha256_file(resolved_plan) != spec.plan_file_sha256
        or production._sha256_file(resolved_submission)
        != spec.submission_file_sha256
    ):
        raise HandoffContractError(
            f"task-{spec.failed_task_id} source file identity drifted"
        )
    plan = production._validate_seal(
        production._read_json(resolved_plan), SOURCE_PLAN_SCHEMA
    )
    submission = production._validate_seal(
        production._read_json(resolved_submission),
        SOURCE_SUBMISSION_SCHEMA,
    )
    plan_root = resolved_plan.parent
    params_path = _verified_file(
        plan_root, plan.get("fea_params"), "file-close FEA params"
    )
    profile_path = _verified_file(
        plan_root, plan.get("profile"), "file-close profile"
    )
    params = production._read_json(params_path)
    profile = production._read_json(profile_path)
    stage = plan.get("stage")
    retained = (
        stage.get("retained_aedt_bundle")
        if isinstance(stage, Mapping)
        else None
    )
    strict_plan = plan.get("scheduler_strict_node_contract")
    strict_runtime = submission.get(
        "mesh_quality_canary_strict_runtime_contract"
    )
    submission_plan = submission.get("plan")
    if (
        plan.get("payload_sha256") != spec.plan_payload_sha256
        or submission.get("payload_sha256")
        != spec.submission_payload_sha256
        or plan.get("candidate_physics_sha256")
        != spec.candidate_physics_sha256
        or submission.get("candidate_physics_sha256")
        != spec.candidate_physics_sha256
        or plan.get("solver_revision") != spec.solver_revision
        or submission.get("solver_revision") != spec.solver_revision
        or plan.get("library_revision") != COMMON_LIBRARY_REVISION
        or submission.get("library_revision") != COMMON_LIBRARY_REVISION
        or not isinstance(stage, Mapping)
        or stage.get("task_name") != spec.task_name
        or stage.get("effective_params_sha256")
        != spec.effective_params_sha256
        or stage.get("profile_sha256") != COMMON_PROFILE_SHA256
        or stage.get("resources")
        != {"cpus": EXPECTED_CPUS, "timeout_seconds": EXPECTED_TIMEOUT_SECONDS}
        or stage.get("full_model") != 0
        or stage.get("thermal_symmetry") != "eighth"
        or not isinstance(retained, Mapping)
        or retained.get("relative_directory")
        != spec.retained_relative_directory
        or retained.get("marker_path") != spec.retained_marker_path
        or retained.get("marker_contract_sha256")
        != spec.retained_marker_contract_sha256
        or retained.get("dedupe_key") != spec.dedupe_key
        or submission.get("task_id") != spec.failed_task_id
        or submission.get("task_name") != spec.task_name
        or submission.get("dedupe_key") != spec.dedupe_key
        or submission.get("resources")
        != {"cpus": EXPECTED_CPUS, "timeout_seconds": EXPECTED_TIMEOUT_SECONDS}
        or submission.get("scheduler_submission_performed") is not True
        or submission.get("scheduler_url") != SCHEDULER_URL
        or not isinstance(submission_plan, Mapping)
        or submission_plan.get("path") != str(resolved_plan)
        or submission_plan.get("sha256") != spec.plan_file_sha256
        or not isinstance(strict_plan, Mapping)
        or strict_plan.get("requested_node_name") != EXPECTED_NODE
        or strict_plan.get("node_name_policy") != "strict"
        or not isinstance(strict_runtime, Mapping)
        or strict_runtime.get("expected_node_name") != EXPECTED_NODE
        or strict_runtime.get("expected_allocation_id")
        != EXPECTED_ALLOCATION_ID
        or str(strict_runtime.get("expected_slurm_job_id"))
        != EXPECTED_SLURM_JOB_ID
        or strict_runtime.get("expected_account_name") != EXPECTED_ACCOUNT
    ):
        raise HandoffContractError(
            f"task-{spec.failed_task_id} exact plan/submission authority drifted"
        )
    _validate_fixed_physics(plan, params, profile)
    normalized = {
        "logical_authority_task_id": spec.logical_task_id,
        "failed_task_id": spec.failed_task_id,
        "candidate_physics_sha256": spec.candidate_physics_sha256,
        "solver_revision": spec.solver_revision,
        "library_revision": COMMON_LIBRARY_REVISION,
        "profile_sha256": COMMON_PROFILE_SHA256,
        "effective_params_sha256": spec.effective_params_sha256,
        "plan": production._file_record(resolved_plan),
        "plan_payload_sha256": spec.plan_payload_sha256,
        "submission": production._file_record(resolved_submission),
        "submission_payload_sha256": spec.submission_payload_sha256,
        "task_name": spec.task_name,
        "dedupe_key": spec.dedupe_key,
        "resources": {
            "cpus": EXPECTED_CPUS,
            "memory_mb": EXPECTED_MEMORY_MB,
            "timeout_seconds": EXPECTED_TIMEOUT_SECONDS,
        },
        "fixed_physics": {
            "fan_velocity_m_s": 1.5,
            "fan_config": "dual",
            "tim_or_insulation_conductivity_W_mK": 0.2,
            "core_plate_pad_t_mm": 2.0,
            "wcp_pad_t_mm": 2.0,
            "full_model": 0,
            "thermal_symmetry": "eighth",
            "effective_rx_side_block_mesh_level": 4,
        },
        "retention_identity": {
            "relative_directory": spec.retained_relative_directory,
            "marker_path": spec.retained_marker_path,
            "marker_contract_sha256": (
                spec.retained_marker_contract_sha256
            ),
        },
    }
    return plan, submission, normalized


def _task_failure_evidence(
    snapshot: Mapping[str, Any],
    *,
    spec: SourceSpec,
    submission: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = {
        "schema_version": TASK_SCHEMA,
        "task_id": _task_id(snapshot),
        "logical_authority_task_id": spec.logical_task_id,
        "name": snapshot.get("name"),
        "status": snapshot.get("status"),
        "state": snapshot.get("state"),
        "exit_code": snapshot.get("exit_code"),
        "failure_message": snapshot.get("failure_message"),
        "project": snapshot.get("project"),
        "dedupe_key": snapshot.get("dedupe_key"),
        "aedt_backend": snapshot.get("aedt_backend"),
        "cpus": snapshot.get("cpus"),
        "memory_mb": snapshot.get("memory_mb"),
        "timeout_seconds": snapshot.get("timeout_seconds"),
        "account_name": snapshot.get("account_name"),
        "requested_account_name": snapshot.get("requested_account_name"),
        "requested_node_name": snapshot.get(
            "requested_node_name", snapshot.get("node_name")
        ),
        "requested_node_name_policy": snapshot.get(
            "requested_node_name_policy", snapshot.get("node_name_policy")
        ),
        "strict_node_placement": snapshot.get("strict_node_placement"),
        "placement_contract_satisfied": snapshot.get(
            "placement_contract_satisfied"
        ),
        "actual_node_name": snapshot.get("actual_node_name"),
        "allocation_id": snapshot.get(
            "allocation_id", snapshot.get("assigned_allocation")
        ),
        "slurm_job_id": str(snapshot.get("slurm_job_id") or ""),
        "remote_cwd": snapshot.get("remote_cwd"),
        "remote_dir": snapshot.get("remote_dir"),
        "started_at": snapshot.get("started_at"),
        "finished_at": snapshot.get("finished_at"),
    }
    if (
        evidence["task_id"] != spec.failed_task_id
        or evidence["name"] != spec.task_name
        or evidence["status"] != "failed"
        or evidence["state"] != "failed"
        or evidence["exit_code"] != 1
        or evidence["failure_message"] != EXPECTED_FAILURE_MESSAGE
        or evidence["project"] != SCHEDULER_PROJECT
        or evidence["dedupe_key"] != submission.get("dedupe_key")
        or evidence["aedt_backend"] != "standalone"
        or evidence["cpus"] != EXPECTED_CPUS
        or evidence["memory_mb"] != EXPECTED_MEMORY_MB
        or evidence["timeout_seconds"] != EXPECTED_TIMEOUT_SECONDS
        or evidence["account_name"] != EXPECTED_ACCOUNT
        or evidence["requested_account_name"] != EXPECTED_ACCOUNT
        or evidence["requested_node_name"] != EXPECTED_NODE
        or evidence["requested_node_name_policy"] != "strict"
        or evidence["strict_node_placement"] is not True
        or evidence["placement_contract_satisfied"] is not True
        or evidence["actual_node_name"] != EXPECTED_NODE
        or evidence["allocation_id"] != EXPECTED_ALLOCATION_ID
        or evidence["slurm_job_id"] != EXPECTED_SLURM_JOB_ID
        or evidence["remote_cwd"]
        != "__SLURM_SCHEDULER_ACCOUNT_WORKSPACE__/runs"
        or not re.fullmatch(
            rf"slurm_scheduler/runs/2026-07-25/task-"
            rf"{spec.failed_task_id}-[0-9]+",
            str(evidence["remote_dir"] or ""),
        )
        or evidence["started_at"] != spec.expected_started_at
        or evidence["finished_at"] != spec.expected_finished_at
    ):
        raise HandoffContractError(
            f"task-{spec.failed_task_id} terminal Scheduler identity drifted"
        )
    return evidence


def _loss_dispatch(text: str) -> dict[str, Any]:
    dispatches = []
    for line in text.splitlines():
        if not line.startswith(DISPATCH_PREFIX):
            continue
        try:
            value = json.loads(line.removeprefix(DISPATCH_PREFIX))
        except json.JSONDecodeError as exc:
            raise HandoffContractError(
                "file-close loss dispatch marker is invalid JSON"
            ) from exc
        if isinstance(value, dict) and value.get("stage") == "loss":
            dispatches.append(value)
    if len(dispatches) != 1:
        raise HandoffContractError(
            "file-close evidence requires exactly one loss dispatch"
        )
    dispatch = dispatches[0]
    if (
        dispatch.get("backend") != "standalone"
        or dispatch.get("dispatch") != "native_analyze_validated_dso"
        or dispatch.get("cores_argument") != EXPECTED_CPUS
        or dispatch.get("gpus_argument") != 0
        or dispatch.get("tasks_argument") != 1
    ):
        raise HandoffContractError("file-close loss dispatch identity drifted")
    return dispatch


def _stream_evidence(
    stdout: bytes | str,
    stderr: bytes | str,
    *,
    spec: SourceSpec,
) -> dict[str, Any]:
    raw_out = stdout.encode("utf-8") if isinstance(stdout, str) else stdout
    raw_err = stderr.encode("utf-8") if isinstance(stderr, str) else stderr
    if (
        not isinstance(raw_out, bytes)
        or not isinstance(raw_err, bytes)
        or not raw_out
        or not raw_err
        or len(raw_out) > MAX_STREAM_BYTES
        or len(raw_err) > MAX_STREAM_BYTES
    ):
        raise HandoffContractError("file-close stream bytes are invalid")
    try:
        out_text = raw_out.decode("utf-8")
        err_text = raw_err.decode("utf-8")
    except UnicodeError as exc:
        raise HandoffContractError(
            "file-close streams are not valid UTF-8"
        ) from exc
    combined = f"{out_text}\n{err_text}"
    out_sha = production._sha256_bytes(raw_out)
    err_sha = production._sha256_bytes(raw_err)
    explicit_errno_counts = {
        marker: combined.count(marker) for marker in EXPLICIT_IO_ERRNO_MARKERS
    }
    if (
        out_sha != spec.expected_stdout_sha256
        or len(raw_out) != spec.expected_stdout_size_bytes
        or err_sha != spec.expected_stderr_sha256
        or len(raw_err) != spec.expected_stderr_size_bytes
        or out_text.count(FILE_CLOSE_MARKER) != 0
        or err_text.count(FILE_CLOSE_MARKER) != 1
        or out_text.count(EXECUTION_ERROR_MARKER) != 0
        or err_text.count(EXECUTION_ERROR_MARKER) != 1
        or err_text.count(SCRIPT_MACRO_MARKER) != 1
        or RESULT_MARKER in combined
        or G3DMESHER_MARKER in combined
        or MEMORY_HINT_MARKER in combined
        or LOSS_TRACE_MARKER not in err_text
        or any(explicit_errno_counts.values())
    ):
        raise HandoffContractError(
            f"task-{spec.failed_task_id} is not the exact ambiguous "
            "file-close loss failure"
        )
    dispatch = _loss_dispatch(out_text)
    return {
        "schema_version": STREAM_SCHEMA,
        "task_id": spec.failed_task_id,
        "stdout_sha256": out_sha,
        "stdout_size_bytes": len(raw_out),
        "stderr_sha256": err_sha,
        "stderr_size_bytes": len(raw_err),
        "failure_stage": "loss",
        "loss_dispatch": dispatch,
        "file_close_placeholder_marker_count": 1,
        "execution_error_marker_count": 1,
        "script_macro_error_marker_count": 1,
        "result_json_absent": True,
        "g3dmesher_marker_absent": True,
        "oom_memory_hint_absent": True,
        "explicit_io_errno_marker_counts": explicit_errno_counts,
        "target_file_path_identified": False,
    }


def _task_root_relative(remote_dir: str) -> str:
    prefix = "slurm_scheduler/runs/"
    if not remote_dir.startswith(prefix):
        raise HandoffContractError("file-close remote directory drifted")
    relative = remote_dir.removeprefix(prefix)
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or len(pure.parts) != 2:
        raise HandoffContractError("file-close remote directory is unsafe")
    return relative


def _task_script_evidence(
    raw: bytes | str,
    *,
    spec: SourceSpec,
    stdout: bytes | str,
) -> dict[str, Any]:
    script = raw.encode("utf-8") if isinstance(raw, str) else raw
    stdout_raw = stdout.encode("utf-8") if isinstance(stdout, str) else stdout
    if (
        not isinstance(script, bytes)
        or not script
        or len(script) > MAX_TASK_SCRIPT_BYTES
        or production._sha256_bytes(script)
        != spec.expected_task_script_sha256
        or len(script) != spec.expected_task_script_size_bytes
    ):
        raise HandoffContractError(
            f"task-{spec.failed_task_id} task.sh identity drifted"
        )
    try:
        text = script.decode("utf-8")
        out_text = stdout_raw.decode("utf-8")
    except UnicodeError as exc:
        raise HandoffContractError(
            "file-close task script/stdout is not UTF-8"
        ) from exc
    nvme_match = re.search(r"MFT_NVME_WORKDIR=(/enroot/[^;]+)", text)
    threshold_match = re.search(
        r'MFT_ENROOT_FREE_KB=.*?; if .*? -ge ([0-9]+) \]; then',
        text,
    )
    workdir_match = re.search(r"(?m)^MFT_WORKDIR (/.+)$", out_text)
    expected_slug = spec.task_name.replace("-", "_")
    if (
        nvme_match is None
        or threshold_match is None
        or workdir_match is None
        or not nvme_match.group(1).startswith(
            f"/enroot/mft_campaign-{expected_slug}-"
        )
        or workdir_match.group(1) != nvme_match.group(1)
        or int(threshold_match.group(1)) != ENROOT_ROUTE_THRESHOLD_KB
        or 'findmnt -n -o FSTYPE -T /enroot' not in text
        or '= xfs ]' not in text
        or 'df -Pk /enroot' not in text
        or "printf 'MFT_WORKDIR %s\\n'" not in text
        or "MFT_ENROOT_FREE_KB %s" in text
        or "simulation_rc=$?" not in text
        or f"$MFT_TASK_ROOT/{spec.retained_relative_directory}" not in text
        or (
            'cleanup() { rm -rf -- "${MFT_NVME_WORKDIR}" '
            '"${MFT_GPFS_WORKDIR}"'
        )
        not in text
    ):
        raise HandoffContractError(
            f"task-{spec.failed_task_id} storage routing contract drifted"
        )
    return {
        "schema_version": TASK_SCRIPT_SCHEMA,
        "task_id": spec.failed_task_id,
        "task_script_sha256": production._sha256_bytes(script),
        "task_script_size_bytes": len(script),
        "solver_workdir": workdir_match.group(1),
        "solver_storage_scope": "compute-local-/enroot",
        "required_filesystem_type": "xfs",
        "routing_free_space_threshold_kb": ENROOT_ROUTE_THRESHOLD_KB,
        "routing_free_space_threshold_gib": (
            ENROOT_ROUTE_THRESHOLD_KB / 1024 / 1024
        ),
        "failure_time_free_space_logged": False,
        "failure_time_inode_state_logged": False,
        "storage_device_identity_logged": False,
        "cleanup_removes_compute_local_workdir": True,
        "retention_export_target_scope": "account-GPFS-task-root",
        "retention_export_target": spec.retained_relative_directory,
    }


def _retention_evidence(
    *,
    spec: SourceSpec,
    task: Mapping[str, Any],
    root_inventory: Mapping[str, Any],
    retained_inventory: Mapping[str, Any],
    task_script: bytes,
    wrapper: bytes,
    exit_code: bytes,
    stdout: bytes,
    stderr: bytes,
) -> dict[str, Any]:
    root = _task_root_relative(str(task["remote_dir"]))
    expected_files = sorted(
        f"{root}/{name}"
        for name in (
            "exit_code",
            "stderr.log",
            "stdout.log",
            "task.sh",
            "wrapper.log",
        )
    )
    root_files = root_inventory.get("files")
    retained_files = retained_inventory.get("files")
    if (
        root_inventory.get("base") != "remote_cwd"
        or root_inventory.get("glob") != f"{root}/**"
        or not isinstance(root_files, list)
        or sorted(root_files) != expected_files
        or retained_inventory.get("base") != "remote_cwd"
        or retained_inventory.get("glob")
        != f"{root}/{spec.retained_relative_directory}/**"
        or retained_files != []
        or wrapper != b""
        or exit_code != b"1\n"
    ):
        raise HandoffContractError(
            f"task-{spec.failed_task_id} wrapper/retention inventory drifted"
        )
    sizes = {
        "exit_code": len(exit_code),
        "stderr.log": len(stderr),
        "stdout.log": len(stdout),
        "task.sh": len(task_script),
        "wrapper.log": len(wrapper),
    }
    return {
        "schema_version": RETENTION_SCHEMA,
        "task_id": spec.failed_task_id,
        "remote_task_root": root,
        "remote_wrapper_files": expected_files,
        "remote_wrapper_file_sizes": sizes,
        "remote_wrapper_total_bytes": sum(sizes.values()),
        "planned_retention_relative_directory": (
            spec.retained_relative_directory
        ),
        "planned_retention_marker_path": spec.retained_marker_path,
        "planned_retention_marker_contract_sha256": (
            spec.retained_marker_contract_sha256
        ),
        "retained_file_count": 0,
        "retained_bundle_present": False,
        "retention_marker_present": False,
        "failed_compute_local_workdir_preserved": False,
        "retention_absence_is_failure_cause_evidence": False,
        "retention_absence_limits_forensic_recovery": True,
    }


def _event_evidence(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(events) != EVENT_LIMIT:
        raise HandoffContractError(
            "file-close event window must be the exact 1000-event GET"
        )
    selected = [
        dict(event)
        for event in events
        if str(event.get("entity_id"))
        in {str(spec.failed_task_id) for spec in SOURCE_SPECS}
    ]
    expected_kinds = {
        spec.failed_task_id: {
            "task_requeued": 1,
            "task_failed": 1,
            "task_cleanup": 1,
        }
        for spec in SOURCE_SPECS
    }
    counts: dict[int, dict[str, int]] = {
        task_id: {kind: 0 for kind in kinds}
        for task_id, kinds in expected_kinds.items()
    }
    for event in selected:
        task_id = int(str(event.get("entity_id")))
        kind = str(event.get("kind") or "")
        if kind in counts[task_id]:
            counts[task_id][kind] += 1
        if (
            event.get("account_name") != EXPECTED_ACCOUNT
            or event.get("entity_type") != "task"
        ):
            raise HandoffContractError(
                "file-close task event account/entity drifted"
            )
    requeues = [
        event
        for event in selected
        if event.get("kind") == "task_requeued"
    ]
    if (
        counts != expected_kinds
        or any(
            "requeued after memory-pressure kill (attempt 1/3)"
            not in str(event.get("message") or "")
            for event in requeues
        )
    ):
        raise HandoffContractError(
            "file-close exact task event history drifted"
        )
    ordered = sorted(selected, key=lambda item: int(item["id"]))
    return {
        "schema_version": EVENT_SCHEMA,
        "scheduler_event_limit": EVENT_LIMIT,
        "full_window_count": len(events),
        "full_window_newest_event_id": events[0].get("id"),
        "full_window_oldest_event_id": events[-1].get("id"),
        "full_window_sha256": production.canonical_sha256(list(events)),
        "selected_events": ordered,
        "selected_event_counts": {
            str(task_id): kinds for task_id, kinds in sorted(counts.items())
        },
        "both_tasks_had_one_prior_memory_pressure_requeue": True,
        "prior_requeue_is_not_final_failure_cause_authentication": True,
    }


def _allocation_evidence(
    allocations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    matches = [
        dict(row)
        for row in allocations
        if row.get("id") == EXPECTED_ALLOCATION_ID
    ]
    if len(matches) != 1:
        raise HandoffContractError(
            "file-close allocation identity is absent or ambiguous"
        )
    row = matches[0]
    if (
        row.get("account_name") != EXPECTED_ACCOUNT
        or row.get("node_name") != EXPECTED_NODE
        or str(row.get("slurm_job_id") or "") != EXPECTED_SLURM_JOB_ID
        or row.get("partition") != "cpu2"
        or row.get("total_cpus") != 64
        or row.get("total_memory_mb") != 870568
    ):
        raise HandoffContractError(
            "file-close shared allocation identity drifted"
        )
    storage_keys = sorted(
        key
        for key in row
        if any(
            token in key.lower()
            for token in ("disk", "storage", "filesystem", "inode")
        )
    )
    if storage_keys:
        raise HandoffContractError(
            "allocation unexpectedly gained unaudited storage fields"
        )
    selected_fields = (
        "id",
        "account_name",
        "partition",
        "node_name",
        "slurm_job_id",
        "state",
        "total_cpus",
        "free_cpus",
        "total_memory_mb",
        "free_memory_mb",
        "remote_dir",
        "failure_message",
        "drain_reason",
        "created_at",
        "started_at",
        "last_active_at",
        "updated_at",
        "node_pestat_state",
        "node_cpu_used",
        "node_cpu_total",
        "node_memory_used_mb",
        "node_memory_free_mb",
        "node_memory_total_mb",
        "node_metrics_observed_at",
    )
    return {
        "schema_version": ALLOCATION_SCHEMA,
        "allocation": {name: row.get(name) for name in selected_fields},
        "allocation_storage_fields_present": storage_keys,
        "failure_time_enroot_storage_snapshot_present": False,
        "node_metrics_are_post_failure_and_non_causal": True,
    }


def _classification() -> dict[str, Any]:
    return {
        "common_terminal_signature_authenticated": True,
        "failure_stage": "Maxwell-loss",
        "physical_failure_authenticated": False,
        "operational_io_or_storage_failure_suspected": True,
        "operational_root_cause_authenticated": False,
        "compute_local_enroot_exhaustion_authenticated": False,
        "account_gpfs_quota_failure_authenticated": False,
        "memory_or_g3dmesher_failure_authenticated_for_these_tasks": False,
        "classification": (
            "ambiguous-operational-io-or-storage-like-not-authenticated"
        ),
        "causal_gaps": [
            "Error closing file uses unresolved %1 and identifies no path",
            "no ENOSPC, EDQUOT, EIO, or read-only-filesystem errno is logged",
            "no failure-time /enroot free-space or inode snapshot is logged",
            "allocation GET exposes no compute-local storage telemetry",
            "the retained bundle and failed /enroot workdir are absent",
            "two correlated failures do not independently prove root cause",
        ],
        "correlation_evidence": [
            "different candidate physics hashes",
            "different solver revisions",
            "same profile and fixed cooling identity",
            "same n114 allocation 14492 / Slurm job 824575",
            "same compute-local /enroot routing",
            "same Maxwell loss-stage file-close signature",
            "both tasks previously requeued once after Scheduler memory pressure",
        ],
        "retry_authorized": False,
        "same_payload_resubmission_allowed": False,
        "operational_successor_plan_created": False,
        "automatic_retry_allowed": False,
        "required_new_authority": (
            "authenticate an exact operational cause with failure-time "
            "compute-local storage/errno/path evidence, then create a new "
            "independently reviewed immutable generation"
        ),
    }


def create_no_retry_refusal(
    *,
    plan_paths: Sequence[Path],
    submission_paths: Sequence[Path],
    output: Path,
    scheduler_url: str = SCHEDULER_URL,
    task_reader: Callable[..., Mapping[str, Any]] | None = None,
    stream_reader: Callable[..., bytes | str] | None = None,
    event_reader: Callable[..., Sequence[Mapping[str, Any]]] | None = None,
    allocation_reader: Callable[..., Sequence[Mapping[str, Any]]] | None = None,
    remote_files_reader: Callable[..., Mapping[str, Any]] | None = None,
    remote_file_reader: Callable[..., bytes | str] | None = None,
    now: datetime | None = None,
) -> Path:
    """Create immutable GET-only refusal evidence for the exact two tasks."""

    normalized_scheduler = scheduler_url.rstrip("/")
    if (
        normalized_scheduler != SCHEDULER_URL
        or len(plan_paths) != len(SOURCE_SPECS)
        or len(submission_paths) != len(SOURCE_SPECS)
    ):
        raise HandoffContractError(
            "file-close audit source count or Scheduler origin drifted"
        )
    read_task = task_reader or _scheduler_task
    read_stream = stream_reader or _scheduler_stream
    read_events = event_reader or _scheduler_events
    read_allocations = allocation_reader or _scheduler_allocations
    read_remote_files = remote_files_reader or _scheduler_remote_files
    read_remote_file = remote_file_reader or _scheduler_remote_file

    sources = []
    task_evidence = []
    stream_evidence = []
    script_evidence = []
    retention_evidence = []
    for spec, plan_path, submission_path in zip(
        SOURCE_SPECS, plan_paths, submission_paths, strict=True
    ):
        _plan, submission, normalized = _load_source(
            spec=spec,
            plan_path=Path(plan_path),
            submission_path=Path(submission_path),
        )
        task = dict(
            read_task(
                scheduler_url=normalized_scheduler,
                task_id=spec.failed_task_id,
            )
        )
        terminal = _task_failure_evidence(
            task, spec=spec, submission=submission
        )
        stdout = read_stream(
            scheduler_url=normalized_scheduler,
            task_id=spec.failed_task_id,
            stream="stdout",
        )
        stderr = read_stream(
            scheduler_url=normalized_scheduler,
            task_id=spec.failed_task_id,
            stream="stderr",
        )
        streams = _stream_evidence(stdout, stderr, spec=spec)
        root = _task_root_relative(str(task["remote_dir"]))
        task_script = read_remote_file(
            scheduler_url=normalized_scheduler,
            task_id=spec.failed_task_id,
            path=f"{root}/task.sh",
            max_bytes=MAX_TASK_SCRIPT_BYTES,
        )
        wrapper = read_remote_file(
            scheduler_url=normalized_scheduler,
            task_id=spec.failed_task_id,
            path=f"{root}/wrapper.log",
            max_bytes=MAX_TASK_SCRIPT_BYTES,
        )
        exit_code = read_remote_file(
            scheduler_url=normalized_scheduler,
            task_id=spec.failed_task_id,
            path=f"{root}/exit_code",
            max_bytes=1024,
        )
        routing = _task_script_evidence(
            task_script, spec=spec, stdout=stdout
        )
        root_inventory = read_remote_files(
            scheduler_url=normalized_scheduler,
            task_id=spec.failed_task_id,
            glob=f"{root}/**",
        )
        retained_inventory = read_remote_files(
            scheduler_url=normalized_scheduler,
            task_id=spec.failed_task_id,
            glob=f"{root}/{spec.retained_relative_directory}/**",
        )
        retention = _retention_evidence(
            spec=spec,
            task=task,
            root_inventory=root_inventory,
            retained_inventory=retained_inventory,
            task_script=(
                task_script.encode("utf-8")
                if isinstance(task_script, str)
                else task_script
            ),
            wrapper=(
                wrapper.encode("utf-8")
                if isinstance(wrapper, str)
                else wrapper
            ),
            exit_code=(
                exit_code.encode("utf-8")
                if isinstance(exit_code, str)
                else exit_code
            ),
            stdout=stdout.encode("utf-8") if isinstance(stdout, str) else stdout,
            stderr=stderr.encode("utf-8") if isinstance(stderr, str) else stderr,
        )
        sources.append(normalized)
        task_evidence.append(terminal)
        stream_evidence.append(streams)
        script_evidence.append(routing)
        retention_evidence.append(retention)

    if (
        len({item["candidate_physics_sha256"] for item in sources}) != 2
        or len({item["solver_revision"] for item in sources}) != 2
        or len({item["dedupe_key"] for item in sources}) != 2
        or len({item["profile_sha256"] for item in sources}) != 1
        or len({item["allocation_id"] for item in task_evidence}) != 1
        or len({item["slurm_job_id"] for item in task_evidence}) != 1
        or len({item["actual_node_name"] for item in task_evidence}) != 1
    ):
        raise HandoffContractError(
            "file-close cross-task comparison identity drifted"
        )
    events = _event_evidence(
        list(read_events(scheduler_url=normalized_scheduler))
    )
    allocation = _allocation_evidence(
        list(read_allocations(scheduler_url=normalized_scheduler))
    )
    observed = now or datetime.now(timezone.utc)
    if observed.tzinfo is None:
        raise HandoffContractError(
            "file-close audit observation time must be timezone-aware"
        )
    refusal = production._seal(
        {
            "schema_version": AUDIT_SCHEMA,
            "campaign_id": "mft-goal-20260726",
            "observed_at_utc": observed.astimezone(timezone.utc).isoformat(),
            "scheduler_url": normalized_scheduler,
            "scheduler_project": SCHEDULER_PROJECT,
            "scope": {
                "logical_authority_task_ids": sorted(
                    spec.logical_task_id for spec in SOURCE_SPECS
                ),
                "failed_task_ids": sorted(
                    spec.failed_task_id for spec in SOURCE_SPECS
                ),
                "excluded_task_ids": list(EXCLUDED_TASK_IDS),
                "excluded_task_signatures_used": False,
                "task_96302_g3dmesher_oom_authority_mixed": False,
            },
            "source_authorities": sources,
            "terminal_task_evidence": task_evidence,
            "terminal_stream_evidence": stream_evidence,
            "task_script_storage_routing": script_evidence,
            "retention_evidence": retention_evidence,
            "event_evidence": events,
            "allocation_evidence": allocation,
            "classification": _classification(),
            "available_submission_commands": [],
            "scheduler_http_methods_used": ["GET"],
            "scheduler_get_count": 18,
            "scheduler_post_calls": 0,
            "scheduler_cancel_calls": 0,
            "scheduler_priority_mutations": 0,
            "scheduler_repository_modified": False,
            "scheduler_project_mutation_performed": False,
            "successor_plan_written": False,
            "same_payload_retry_written": False,
        }
    )
    return production._write_immutable_json(Path(output).resolve(), refusal)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "GET-only exact task-96300/task-96301 file-close failure audit"
        )
    )
    parser.add_argument("--plan-96300", type=Path, required=True)
    parser.add_argument("--submission-96300", type=Path, required=True)
    parser.add_argument("--plan-96301", type=Path, required=True)
    parser.add_argument("--submission-96301", type=Path, required=True)
    parser.add_argument("--scheduler-url", default=SCHEDULER_URL)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = create_no_retry_refusal(
        plan_paths=[args.plan_96300, args.plan_96301],
        submission_paths=[
            args.submission_96300,
            args.submission_96301,
        ],
        scheduler_url=args.scheduler_url,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "status": "refused_post0",
                "path": str(result),
                "scheduler_post_calls": 0,
                "retry_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
