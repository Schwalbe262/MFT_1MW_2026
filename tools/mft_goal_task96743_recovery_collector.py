"""GET-only recovery collector for the exact tuned symmetric retry task 96743.

Task 96743 was submitted as a failover retry without its own local sealed
plan/receipt.  The older task-96483 plan and receipt bind a different task
name, solver revision, dedupe key, and retained token, so they are explicitly
forbidden as retry authority.

This narrow collector creates a new recovery plan from:

* the exact live task GET and immutable remote-dir ``task.sh``;
* the sealed tuned-gap campaign/final manifest and source parameter bytes;
* the reviewed Standard execution profile;
* the exact retry solver/library/parameter identity.

On terminal success it authenticates stdout ``RESULT_JSON``, the remote
retention receipt and prune marker, downloads every declared base64 chunk,
reconstructs and hashes ``symmetric.aedt``, and writes an atomic sealed local
collection.  A final-artifact winner authority is emitted only when the
strict solver result, full physical goal gates, split temperature limits,
2 mH tolerance, and retained artifact authentication all pass.

There is no POST, cancel, priority, or Scheduler project mutation surface.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from module.fixed_boundary_contract import (  # noqa: E402
    FIXED_BOUNDARY_CONTRACT_SHA256,
)
from module.input_parameter_260706 import (  # noqa: E402
    create_input_parameter,
    validation_check,
)
from module.mft_goal_20260726_contract import (  # noqa: E402
    FIXED_COOLING_IDENTITY_SHA256,
    FIXED_OPERATING_IDENTITY_SHA256,
    GOAL_CONTRACT_SCHEMA,
    GOAL_SIZE_LIMITS_MM,
    GOAL_STAGE_SPEC_SHA256,
    GOAL_TEMPERATURE_CONTRACT_SHA256,
    TEMPERATURE_FAMILY_LIMITS_C,
    TEMPERATURE_TARGET_FAMILIES,
    attest_fixed_identity,
)
from regression_260707.optimization import geometry_metrics  # noqa: E402
from regression_260707.verify import scheduler_client  # noqa: E402
from tools import mft_goal_fea_handoff as handoff  # noqa: E402
from tools import mft_goal_final_artifact_pipeline as final_pipeline  # noqa: E402
from tools import mft_goal_lm2mh_gap_tuner as tuner  # noqa: E402
from tools import mft_goal_terminal_collector as terminal  # noqa: E402
from tools import tier1_corrected_generation_preflight as preflight  # noqa: E402


RECOVERY_PLAN_SCHEMA = "mft-goal-task96743-recovery-plan-v1"
POLL_SCHEMA = "mft-goal-task96743-recovery-poll-v1"
FAILURE_SCHEMA = "mft-goal-task96743-recovery-failure-v1"
COLLECTION_SCHEMA = "mft-goal-task96743-recovery-collection-v1"
COLLECTION_SEAL_SCHEMA = "mft-goal-task96743-recovery-collection-seal-v1"
HEARTBEAT_SCHEMA = "mft-goal-task96743-recovery-heartbeat-v1"
PID_SCHEMA = "mft-goal-task96743-recovery-pid-v1"

SCHEDULER_URL = "http://127.0.0.1:8002"
SCHEDULER_PROJECT = "MFT_1MW_2026v1"
TASK_ID = 96_743
TASK_NAME = "mft-final-sym-gap-2b2138a99445-g00860423-r1"
SOLVER_REVISION = "6af3e7e7cfba187b9711891c9457ce64835131b5"
LIBRARY_REVISION = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
PARAMETER_DIGEST = "0700f09bad77ebcc"
DEDUPE_KEY = (
    f"mft-al:{TASK_NAME}:{SOLVER_REVISION}:"
    f"{LIBRARY_REVISION}:{PARAMETER_DIGEST}"
)
CANDIDATE_PHYSICS_SHA256 = (
    "2b2138a99445c4ed7d50db5b07f7617789a6ff7af735a85fd7038fd1ab266d60"
)

ACCOUNT_NAME = "dhj02"
NODE_NAME = "n110"
ALLOCATION_ID = 14_648
SLURM_JOB_ID = "840585"
REMOTE_DIR = "slurm_scheduler/runs/2026-07-27/task-96743-1785098640"
REMOTE_CWD = "__SLURM_SCHEDULER_ACCOUNT_WORKSPACE__/runs"
CPUS = 8
MEMORY_MB = 65_536
TIMEOUT_SECONDS = 14_400
PRIORITY = 100
MAX_WORKERS_PER_NODE = 1
CORE_CONTRACT = handoff.STANDARD_CORE_CONTRACT
CORE_AUTH_SHA256 = (
    "499cb9538d1de66fa1006c4581298a4d6d58badbcd0424a677ae1b280c77de48"
)

TASK_SH_SIZE_BYTES = 10_352
TASK_SH_SHA256 = (
    "05b4c71d9a085b8be582bd2c1b8f3386d92c7d23b41809f2b85af75f272a6cd1"
)
PROFILE_CANONICAL_SHA256 = handoff.PROFILE_CANONICAL_SHA256["standard"]
PROFILE_FILE_SHA256 = (
    "9fb1f9da9117506b3588a4201835d27bb0dec635cdccddc58a3ee0d53f1d350b"
)

TUNING_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\rank1_neighborhood_2b213_physical_gap_tuning_v1"
)
CAMPAIGN_PATH = TUNING_ROOT / "campaign_manifest.json"
TUNED_MANIFEST_PATH = TUNING_ROOT / "tuned_gap_manifest.json"
PARAMS_PATH = TUNING_ROOT / "downstream_params" / "symmetric_loss_thermal.json"
PROFILE_PATH = (
    REPOSITORY_ROOT
    / "regression_260707"
    / "verify"
    / "profiles"
    / "goal_standard.json"
)
CAMPAIGN_FILE_SHA256 = (
    "824dadc3a55bda844b7d23c7dc54f851f573aff59661b19b17484a613482cb16"
)
CAMPAIGN_PAYLOAD_SHA256 = (
    "f5ee4b838c530170f4b8b5fe951dd10a8ef662cedfe1aa36b3f2be0b03ad6daf"
)
TUNED_FILE_SHA256 = (
    "2447663f1fb9d296bf5d041ad35a65a2e956a7d2c02f3e1a6df2f7b16a9935ee"
)
TUNED_PAYLOAD_SHA256 = (
    "e9e570ebb1c219921c303170933a88c2be02352e9c7af8314e0e938d7d904ab5"
)
PARAMS_FILE_SHA256 = (
    "934e70fc4d3a1a8d59742ef0a017c3213174497fba1d4dae5496d5fc219a6622"
)

RETAINED_TOKEN = "3c262e829b18d3a2"
RETAINED_ROOT = f"goal-fea-retained/{RETAINED_TOKEN}"
ARTIFACT_PATH = f"{RETAINED_ROOT}/symmetric.aedt"
RECEIPT_PATH = f"{ARTIFACT_PATH}.receipt.json"
MARKER_PATH = f"{RETAINED_ROOT}/.slurm-scheduler-preserve.json"
CHUNK_DIRECTORY = f"{ARTIFACT_PATH}.chunks"
MARKER_CONTRACT = {
    "owner": SCHEDULER_PROJECT,
    "preserve": True,
    "reason": (
        "Retain goal FEA AEDT until authenticated package collection; "
        f"stage=standard; dedupe_key={DEDUPE_KEY}"
    ),
    "schema": "slurm-scheduler-prune-protection-v1",
}
MARKER_CONTRACT_SHA256 = (
    "0aa32db17cefe2493d2f7f91d346129cf3039f7cb27d1bb3fea55cb9b6df060e"
)

RAW_CHUNK_BYTES = 768_000
MAX_ENCODED_CHUNK_BYTES = 1_024_000
MAX_AEDT_BYTES = 64 * 1024 * 1024 * 1024
MAX_TASK_OUTPUT_BYTES = terminal.MAX_TASK_OUTPUT_BYTES
MAX_METADATA_BYTES = handoff.MAX_REMOTE_METADATA_BYTES
TARGET_LM_H = 0.002
TARGET_LM_ABS_TOLERANCE_H = 0.00002
CAPACITANCE_TRUTH_SCHEMA = (
    "mft-goal-task96743-capacitance-truth-boundary-v1"
)
DEFAULT_OUTPUT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\task96743_recovery_collection_v1"
)
ACTIVE_STATES = {"queued", "attaching", "running"}
TERMINAL_SUCCESS = {("completed", "succeeded")}
TERMINAL_FAILURE = {
    "failed",
    "cancelled",
    "canceled",
    "timed_out",
    "timeout",
}

_CANDIDATE_JSON_PATTERN = re.compile(
    rb"printf '%s' '(\{[^']+\})' > cand\.json"
)


class RecoveryError(RuntimeError):
    """The exact task/source/result/retained artifact contract drifted."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RecoveryError(f"{label} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise RecoveryError(f"{label} must be a JSON object")
    return value


def _immutable_bytes(path: Path, value: bytes) -> Path:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != value:
            raise RecoveryError(f"immutable output bytes differ: {target}")
        return target
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    staging = Path(staging_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, target)
    finally:
        try:
            staging.unlink()
        except FileNotFoundError:
            pass
    return target


def _immutable_json(path: Path, value: Mapping[str, Any]) -> Path:
    return _immutable_bytes(path, final_pipeline.canonical_bytes(value) + b"\n")


def _replace_json(path: Path, value: Mapping[str, Any]) -> Path:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    staging = Path(staging_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(final_pipeline.canonical_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, target)
    finally:
        try:
            staging.unlink()
        except FileNotFoundError:
            pass
    return target


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    return final_pipeline.file_record(path, relative_to=relative_to)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise RecoveryError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RecoveryError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise RecoveryError(f"{label} must be finite")
    return result


def _task_state(task: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(task.get("status") or "").strip().lower(),
        str(task.get("state") or "").strip().lower(),
    )


def validate_task(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RecoveryError("Scheduler task GET is not an object")
    task = copy.deepcopy(dict(value))
    expected = {
        "task_id": TASK_ID,
        "id": TASK_ID,
        "name": TASK_NAME,
        "dedupe_key": DEDUPE_KEY,
        "project": SCHEDULER_PROJECT,
        "account_name": ACCOUNT_NAME,
        "requested_node_name": NODE_NAME,
        "requested_node_name_policy": "strict",
        "node_name": NODE_NAME,
        "node_name_policy": "strict",
        "actual_node_name": NODE_NAME,
        "allocation_node_name": NODE_NAME,
        "assigned_allocation": ALLOCATION_ID,
        "allocation_id": ALLOCATION_ID,
        "slurm_job_id": SLURM_JOB_ID,
        "remote_cwd": REMOTE_CWD,
        "remote_dir": REMOTE_DIR,
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "scheduling_profile": "fea_bursty",
        "aedt_backend": "standalone",
        "priority": PRIORITY,
        "timeout_seconds": TIMEOUT_SECONDS,
        "max_workers_per_node": MAX_WORKERS_PER_NODE,
        "same_node_as_task_id": 0,
        "cpus": CPUS,
        "memory_mb": MEMORY_MB,
        "gpus": 0,
        "placement_contract_satisfied": True,
        "strict_node_placement": True,
        "preferred_node_relaxed": False,
    }
    for key, expected_value in expected.items():
        if task.get(key) != expected_value:
            raise RecoveryError(f"task96743 GET identity drifted: {key}")
    status, state = _task_state(task)
    if (
        (status, state) not in TERMINAL_SUCCESS
        and status not in ACTIVE_STATES | TERMINAL_FAILURE
        and state not in ACTIVE_STATES | TERMINAL_FAILURE
    ):
        raise RecoveryError(
            f"unsupported task96743 lifecycle: status={status}, state={state}"
        )
    return task


def _decoded_params(effective_params: Mapping[str, Any]) -> dict[str, Any]:
    valid, frame, errors = validation_check(
        create_input_parameter(dict(effective_params)),
        strict=True,
        return_errors=True,
    )
    if not valid:
        raise RecoveryError(
            f"effective task96743 params failed validation: {errors}"
        )
    return preflight._jsonable_decoded_parameters(frame.iloc[0])  # noqa: SLF001


def load_source_contract() -> dict[str, Any]:
    try:
        campaign, root = tuner._load_campaign(CAMPAIGN_PATH)  # noqa: SLF001
        tuned = tuner._validate_seal(  # noqa: SLF001
            tuner._read_json(TUNED_MANIFEST_PATH),  # noqa: SLF001
            tuner.FINAL_SCHEMA,
        )
    except Exception as exc:
        raise RecoveryError("tuned-gap source authentication failed") from exc
    if root != TUNING_ROOT.resolve():
        raise RecoveryError("tuned-gap source root drifted")
    records = {
        "campaign": _file_record(CAMPAIGN_PATH),
        "tuned_manifest": _file_record(TUNED_MANIFEST_PATH),
        "params": _file_record(PARAMS_PATH),
        "profile": _file_record(PROFILE_PATH),
    }
    expected_file_hashes = {
        "campaign": CAMPAIGN_FILE_SHA256,
        "tuned_manifest": TUNED_FILE_SHA256,
        "params": PARAMS_FILE_SHA256,
        "profile": PROFILE_FILE_SHA256,
    }
    for name, expected in expected_file_hashes.items():
        if records[name]["sha256"] != expected:
            raise RecoveryError(f"source {name} bytes drifted")
    if (
        campaign.get("payload_sha256") != CAMPAIGN_PAYLOAD_SHA256
        or tuned.get("payload_sha256") != TUNED_PAYLOAD_SHA256
        or tuned.get("campaign_payload_sha256") != CAMPAIGN_PAYLOAD_SHA256
        or tuned.get("physical_gap_geometry_attested") is not True
        or tuned.get("symmetric_matrix_convergence_attested") is not True
        or tuned.get("native_L11_L22_M_k_Lm_readback_attested") is not True
        or tuned.get("source_candidate", {}).get(
            "physical_geometry_sha256"
        )
        != CANDIDATE_PHYSICS_SHA256
        or tuned.get("tuned_core_center_gap_mm") != 0.860423
    ):
        raise RecoveryError("tuned-gap source identity drifted")
    params_record = tuned.get("downstream_params", {}).get(
        "symmetric_loss_thermal"
    )
    if (
        not isinstance(params_record, Mapping)
        or params_record.get("sha256") != PARAMS_FILE_SHA256
        or params_record.get("size_bytes") != PARAMS_PATH.stat().st_size
    ):
        raise RecoveryError("tuned symmetric parameter binding drifted")
    params = _json_object(PARAMS_PATH, "tuned symmetric params")
    profile = _json_object(PROFILE_PATH, "Standard profile")
    if (
        tuner._sha(profile) != PROFILE_CANONICAL_SHA256  # noqa: SLF001
        or profile.get("stage") != "standard"
        or profile.get("cli_flags") != "--thermal --headless"
    ):
        raise RecoveryError("Standard profile contract drifted")
    identity = scheduler_client.verification_submission_identity(
        TASK_NAME,
        params,
        profile,
        SOLVER_REVISION,
        LIBRARY_REVISION,
    )
    if (
        identity.get("dedupe_key") != DEDUPE_KEY
        or identity.get("parameter_digest") != PARAMETER_DIGEST
    ):
        raise RecoveryError("task96743 effective identity cannot be reconstructed")
    effective_params = copy.deepcopy(identity["merged"])
    decoded = _decoded_params(effective_params)
    n1 = int(decoded["N1_main"]) + int(decoded["N1_side"])
    n2 = int(decoded["N2_main"]) + int(decoded["N2_side"])
    if (
        n1 != 8
        or n2 != 80
        or float(decoded["cw1"]) != 5.0
        or float(decoded["gap1"]) != 1.6
        or int(decoded["full_model"]) != 0
        or int(decoded["round_corner"]) != 0
    ):
        raise RecoveryError("task96743 tuned geometry/turn contract drifted")
    return {
        "campaign": campaign,
        "tuned": tuned,
        "params": params,
        "profile": profile,
        "effective_params": effective_params,
        "decoded_params": decoded,
        "records": records,
    }


def validate_task_script(
    task_sh: bytes, *, effective_params: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        len(task_sh) != TASK_SH_SIZE_BYTES
        or _sha256_bytes(task_sh) != TASK_SH_SHA256
    ):
        raise RecoveryError("task96743 task.sh bytes drifted")
    match = _CANDIDATE_JSON_PATTERN.search(task_sh)
    if match is None or len(_CANDIDATE_JSON_PATTERN.findall(task_sh)) != 1:
        raise RecoveryError("task96743 task.sh candidate JSON is ambiguous")
    try:
        embedded = json.loads(match.group(1).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RecoveryError("task96743 embedded candidate JSON is invalid") from exc
    if embedded != dict(effective_params):
        raise RecoveryError("task96743 task.sh effective params drifted")
    text = task_sh.decode("utf-8")
    required_once = (
        f"export SLURM_SCHED_TASK_ID={TASK_ID}",
        f'git fetch -q origin {SOLVER_REVISION}',
        (
            'git -C "${MFT_WORKDIR}/pyaedt_library" fetch -q origin '
            f"{LIBRARY_REVISION}"
        ),
        "python run_simulation_260706.py --fixed --thermal --headless "
        "--params cand.json",
        ARTIFACT_PATH,
        RECEIPT_PATH,
        MARKER_PATH,
        CHUNK_DIRECTORY,
        DEDUPE_KEY,
        CORE_AUTH_SHA256,
    )
    if any(text.count(item) < 1 for item in required_once):
        raise RecoveryError("task96743 task.sh required contract is absent")
    if "96483" in text or "dd4d9287dd679b2a" in text:
        raise RecoveryError("cancelled task96483 authority leaked into retry task.sh")
    return {
        "embedded_effective_params": embedded,
        "task_sh_sha256": TASK_SH_SHA256,
        "task_sh_size_bytes": TASK_SH_SIZE_BYTES,
        "cancelled_task96483_authority_reused": False,
    }


def _client_task(client: Any) -> dict[str, Any]:
    return validate_task(client.get_json(f"/api/tasks/{TASK_ID}"))


def _client_task_sh(client: Any, effective_params: Mapping[str, Any]) -> bytes:
    task_sh = client.remote_file(
        TASK_ID,
        "task.sh",
        base="remote_dir",
        max_bytes=terminal.MAX_REMOTE_FILE_BYTES,
    )
    validate_task_script(task_sh, effective_params=effective_params)
    return task_sh


def prepare_recovery_plan(*, output: Path, client: Any) -> Path:
    root = output.resolve()
    plan_path = root / "recovery_plan.json"
    if plan_path.exists():
        load_recovery_plan(plan_path)
        return plan_path
    source = load_source_contract()
    task = _client_task(client)
    task_sh = _client_task_sh(client, source["effective_params"])
    source_root = root / "recovery_source"
    task_sh_path = _immutable_bytes(source_root / "task.sh", task_sh)
    task_path = _immutable_json(source_root / "initial_task_get.json", task)
    retained_submission = {
        "schema_version": "mft-goal-task96743-retained-identity-v1",
        "stage": "standard",
        "task_id": TASK_ID,
        "task_name": TASK_NAME,
        "dedupe_key": DEDUPE_KEY,
        "solver_revision": SOLVER_REVISION,
        "library_revision": LIBRARY_REVISION,
        "profile_sha256": PROFILE_CANONICAL_SHA256,
        "parameter_digest": PARAMETER_DIGEST,
        "retained_aedt": {
            "schema_version": scheduler_client.RETAINED_AEDT_SCHEMA,
            "stage": "standard",
            "dedupe_key": DEDUPE_KEY,
            "parameter_digest": PARAMETER_DIGEST,
            "solver_revision": SOLVER_REVISION,
            "library_revision": LIBRARY_REVISION,
            "profile_sha256": PROFILE_CANONICAL_SHA256,
            "artifact_path": ARTIFACT_PATH,
            "receipt_path": RECEIPT_PATH,
            "marker_path": MARKER_PATH,
            "marker_contract": copy.deepcopy(MARKER_CONTRACT),
            "marker_contract_sha256": MARKER_CONTRACT_SHA256,
            "retention_required": True,
            "prune_protection_required": True,
            "transport": {
                "schema_version": (
                    scheduler_client.RETAINED_AEDT_TEXT_CHUNK_SCHEMA
                ),
                "encoding": "base64",
                "chunk_directory": CHUNK_DIRECTORY,
                "raw_chunk_bytes": RAW_CHUNK_BYTES,
                "max_encoded_chunk_bytes": MAX_ENCODED_CHUNK_BYTES,
            },
        },
        "core_policy": {
            "contract": CORE_CONTRACT,
            "requested_num_cores": CPUS,
            "auth_sha256": CORE_AUTH_SHA256,
        },
    }
    plan = final_pipeline.seal(
        {
            "schema_version": RECOVERY_PLAN_SCHEMA,
            "created_at_utc": _now(),
            "task_binding": {
                "task_id": TASK_ID,
                "task_name": TASK_NAME,
                "dedupe_key": DEDUPE_KEY,
                "scheduler_url": SCHEDULER_URL,
                "scheduler_project": SCHEDULER_PROJECT,
                "solver_revision": SOLVER_REVISION,
                "library_revision": LIBRARY_REVISION,
                "parameter_digest": PARAMETER_DIGEST,
                "candidate_physics_sha256": CANDIDATE_PHYSICS_SHA256,
                "node_name": NODE_NAME,
                "allocation_id": ALLOCATION_ID,
                "slurm_job_id": SLURM_JOB_ID,
            },
            "source_authority": {
                **copy.deepcopy(source["records"]),
                "campaign_payload_sha256": CAMPAIGN_PAYLOAD_SHA256,
                "tuned_manifest_payload_sha256": TUNED_PAYLOAD_SHA256,
                "cancelled_task96483_plan_used_as_authority": False,
                "cancelled_task96483_receipt_used_as_authority": False,
                "authority_basis": (
                    "sealed tuned-gap source plus exact task96743 GET/task.sh"
                ),
            },
            "effective_params": copy.deepcopy(source["effective_params"]),
            "effective_params_sha256": tuner._sha(  # noqa: SLF001
                source["effective_params"]
            ),
            "decoded_params_sha256": tuner._sha(  # noqa: SLF001
                source["decoded_params"]
            ),
            "initial_task_get": _file_record(task_path),
            "initial_task_get_sha256": tuner._sha(task),  # noqa: SLF001
            "task_sh": _file_record(task_sh_path),
            "retained_submission": retained_submission,
            "scheduler_get_only": True,
            "scheduler_mutation_performed": False,
            "candidate_selection_performed_by_collector": False,
            "scientific_pass_claimed": False,
        }
    )
    return _immutable_json(plan_path, plan)


def load_recovery_plan(path: Path) -> dict[str, Any]:
    value = final_pipeline.validate_seal(
        _json_object(path, "recovery plan"),
        schema=RECOVERY_PLAN_SCHEMA,
        label="recovery plan",
    )
    source = load_source_contract()
    task_sh_record = value.get("task_sh")
    initial_record = value.get("initial_task_get")
    if not isinstance(task_sh_record, Mapping) or not isinstance(
        initial_record, Mapping
    ):
        raise RecoveryError("recovery plan local evidence is absent")
    task_sh_path = Path(str(task_sh_record.get("path") or ""))
    initial_path = Path(str(initial_record.get("path") or ""))
    if (
        _file_record(task_sh_path) != task_sh_record
        or _file_record(initial_path) != initial_record
    ):
        raise RecoveryError("recovery plan local evidence drifted")
    validate_task_script(
        task_sh_path.read_bytes(),
        effective_params=source["effective_params"],
    )
    validate_task(_json_object(initial_path, "initial task GET"))
    binding = value.get("task_binding")
    if (
        not isinstance(binding, Mapping)
        or binding.get("task_id") != TASK_ID
        or binding.get("task_name") != TASK_NAME
        or binding.get("dedupe_key") != DEDUPE_KEY
        or binding.get("solver_revision") != SOLVER_REVISION
        or binding.get("library_revision") != LIBRARY_REVISION
        or binding.get("parameter_digest") != PARAMETER_DIGEST
        or binding.get("candidate_physics_sha256")
        != CANDIDATE_PHYSICS_SHA256
        or value.get("effective_params") != source["effective_params"]
        or value.get("scheduler_get_only") is not True
        or value.get("scheduler_mutation_performed") is not False
    ):
        raise RecoveryError("recovery plan identity drifted")
    return value


def _extract_result(stdout: bytes) -> dict[str, Any]:
    try:
        text = stdout.decode("utf-8")
    except UnicodeError as exc:
        raise RecoveryError("task96743 stdout is not UTF-8") from exc
    result = None
    library_markers = []
    for line in text.splitlines():
        if line.startswith("MFT_LIBRARY_GIT_HASH "):
            marker = line.removeprefix("MFT_LIBRARY_GIT_HASH ").strip()
            if re.fullmatch(r"[0-9a-f]{40}", marker):
                library_markers.append(marker)
        if line.startswith("RESULT_JSON "):
            try:
                candidate = json.loads(line.removeprefix("RESULT_JSON "))
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                result = candidate
    if result is None:
        raise terminal.TransientGetError(
            "task96743 terminal stdout has no RESULT_JSON"
        )
    if not library_markers or library_markers[-1] != LIBRARY_REVISION:
        raise RecoveryError("task96743 stdout library marker drifted")
    return result


def _capacitance_truth_boundary(
    result: Mapping[str, Any], *, source: Mapping[str, Any]
) -> dict[str, Any]:
    """Classify task96743 capacitance without upgrading legacy truth.

    This retry submitted the ordinary CapTx/CapRx two-net electrostatic
    screen.  It did not submit the opt-in per-turn voltage schedule for
    either winding, much less authenticated Tx and Rx graded-cap results in
    the same full-chain task.  Separate diagnostic sweep tasks can later be
    cited by a distinct upstream authority, but this collector never merges
    them into task96743 or calls its legacy frequency the final resonance.
    """

    effective = source["effective_params"]
    submitted_mode = str(
        effective.get("cap_turn_graded_active_winding", "off")
    )
    graded_keys = sorted(
        key
        for key in result
        if "turn_graded" in str(key).lower()
    )
    legacy_frequency = result.get("f_res_min_tx_rx_only_Hz")
    return {
        "schema_version": CAPACITANCE_TRUTH_SCHEMA,
        "task_id": TASK_ID,
        "submitted_cap_turn_graded_active_winding": submitted_mode,
        "submitted_turn_graded_capacitance": submitted_mode in {"Tx", "Rx"},
        "legacy_two_equipotential_capacitance_present": (
            legacy_frequency is not None
        ),
        "legacy_two_equipotential_resonance_Hz": legacy_frequency,
        "legacy_two_equipotential_resonance_is_final_truth": False,
        "result_turn_graded_field_names": graded_keys,
        "actual_turn_graded_Tx_authenticated": False,
        "actual_turn_graded_Rx_authenticated": False,
        "same_full_chain_graded_cap_task_authenticated": False,
        "separate_sweep_automatically_merged": False,
        "actual_final_resonance_available": False,
        "winner_authority_allowed": False,
        "blocker": (
            "task96743 used legacy CapTx/CapRx equipotential capacitance; "
            "authenticated actual-connection turn-graded Tx and Rx "
            "capacitance provenance is required for final resonance"
        ),
    }


def authenticate_result(
    result: Mapping[str, Any], *, source: Mapping[str, Any]
) -> dict[str, Any]:
    effective = dict(source["effective_params"])
    profile = source["profile"]
    if (
        str(result.get("git_hash") or "").lower() != SOLVER_REVISION
        or str(result.get("pyaedt_library_git_hash") or "").lower()
        != LIBRARY_REVISION
        or not scheduler_client.result_matches_params(result, effective)
    ):
        raise RecoveryError("task96743 RESULT_JSON execution identity drifted")
    try:
        handoff._validate_result_core_policy(  # noqa: SLF001
            result,
            {
                "stage": "standard",
                "task_id": TASK_ID,
                "solver_revision": SOLVER_REVISION,
                "core_policy": {
                    "contract": CORE_CONTRACT,
                    "requested_num_cores": CPUS,
                    "auth_sha256": CORE_AUTH_SHA256,
                },
            },
        )
        fixed = attest_fixed_identity(
            result, require_thermal_pad_metadata=True
        )
    except Exception as exc:
        raise RecoveryError(
            "task96743 solver-core/fixed identity authentication failed"
        ) from exc
    if str(result.get("solver_core_slurm_job_id_readback") or "") != SLURM_JOB_ID:
        raise RecoveryError("task96743 RESULT_JSON Slurm job identity drifted")
    strict_solver_valid = scheduler_client.is_valid_result(
        dict(result),
        expected_revision=SOLVER_REVISION,
        expected_library_revision=LIBRARY_REVISION,
        expected_profile=dict(profile.get("param_overrides") or {}),
    )
    non_capacitance_reasons: list[str] = []
    if not strict_solver_valid:
        non_capacitance_reasons.append(
            "strict_solver_result_contract_invalid"
        )
    try:
        selected = {
            "row_contract": {
                "decoded_params": copy.deepcopy(source["decoded_params"])
            }
        }
        goal_reasons = handoff._goal_result_reasons(  # noqa: SLF001
            result, selected
        )
    except Exception as exc:
        goal_reasons = [
            f"goal_physical_spec_evaluation_failed:{type(exc).__name__}"
        ]
    non_capacitance_reasons.extend(goal_reasons)
    capacitance_truth = _capacitance_truth_boundary(
        result, source=source
    )

    actuals: dict[str, Any] = {}
    temperature_targets: dict[str, Any] = {}
    family_maxima: dict[str, float] = {}
    try:
        volume_l, dimensions = geometry_metrics.bounding_box_lit(result)
        width, length, height = (
            _finite(item, "actual exterior dimension") for item in dimensions
        )
        active, temperature_targets, target_gate = (
            handoff._temperature_gate_evidence(result)  # noqa: SLF001
        )
        family_values = {
            family: [] for family in TEMPERATURE_FAMILY_LIMITS_C
        }
        for target in active:
            family = TEMPERATURE_TARGET_FAMILIES[target]
            family_values[family].append(
                _finite(
                    temperature_targets[target]["actual_C"],
                    f"actual {target}",
                )
            )
        if any(not values for values in family_values.values()):
            raise RecoveryError("temperature family evidence is incomplete")
        family_maxima = {
            family: max(values) for family, values in family_values.items()
        }
        resonance = _finite(
            result.get("f_res_min_tx_rx_only_Hz"), "actual resonance"
        )
        lm_h = 2.0 * _finite(result.get("Lmt"), "native symmetric Lmt") * 1e-6
        actuals = {
            "volume_L": float(volume_l),
            "dimensions_mm": {"W": width, "L": length, "H": height},
            "actual_graded_resonance_Hz": None,
            "legacy_two_equipotential_resonance_Hz": resonance,
            "legacy_two_equipotential_resonance_screen_pass_15kHz": (
                resonance >= 15_000.0
            ),
            "Lm_primary_referred_H": lm_h,
            "temperature_family_max_C": family_maxima,
        }
        if not target_gate:
            non_capacitance_reasons.append(
                "split_temperature_target_gate_failed"
            )
        for axis, limit in GOAL_SIZE_LIMITS_MM.items():
            if actuals["dimensions_mm"][axis] > float(limit):
                non_capacitance_reasons.append(
                    f"actual_{axis}_dimension_exceeds_limit"
                )
        for family, limit in TEMPERATURE_FAMILY_LIMITS_C.items():
            if family_maxima[family] > float(limit):
                non_capacitance_reasons.append(
                    f"actual_{family}_temperature_exceeds_limit"
                )
        if abs(lm_h - TARGET_LM_H) > TARGET_LM_ABS_TOLERANCE_H:
            non_capacitance_reasons.append(
                "actual_Lm_outside_2mH_tolerance"
            )
    except Exception as exc:
        non_capacitance_reasons.append(
            f"measured_hard_constraint_evaluation_failed:{type(exc).__name__}"
        )

    unique_non_capacitance_reasons = list(
        dict.fromkeys(non_capacitance_reasons)
    )
    winner_reasons = [
        *unique_non_capacitance_reasons,
        "actual_turn_graded_Tx_capacitance_provenance_absent",
        "actual_turn_graded_Rx_capacitance_provenance_absent",
        "task96743_legacy_two_net_resonance_not_final_truth",
    ]
    return {
        "strict_solver_result_valid": strict_solver_valid,
        "exact_effective_params_echo_valid": True,
        "solver_core_authentication_passed": True,
        "fixed_identity_attestation": fixed,
        "goal_physical_spec_reasons": list(goal_reasons),
        "legacy_two_net_goal_physical_spec_passed": not goal_reasons,
        "goal_physical_spec_passed": False,
        "measured_actuals": actuals,
        "active_temperature_targets": list(temperature_targets),
        "actual_temperature_targets": copy.deepcopy(temperature_targets),
        "capacitance_truth_boundary": capacitance_truth,
        "non_capacitance_blocking_reasons": (
            unique_non_capacitance_reasons
        ),
        "measured_non_capacitance_constraints_passed": (
            not unique_non_capacitance_reasons
        ),
        "graded_capacitance_provenance_authenticated": False,
        "winner_blocking_reasons": winner_reasons,
        "measured_hard_constraints_passed": False,
    }


def _retained_submission(plan: Mapping[str, Any]) -> dict[str, Any]:
    value = plan.get("retained_submission")
    if not isinstance(value, Mapping):
        raise RecoveryError("retained submission identity is absent")
    return copy.deepcopy(dict(value))


def _validate_remote_metadata(
    *,
    receipt_raw: bytes,
    marker_raw: bytes,
    result: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        receipt = json.loads(receipt_raw.decode("utf-8"))
        marker = json.loads(marker_raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RecoveryError("remote receipt/marker is invalid JSON") from exc
    submission = _retained_submission(plan)
    try:
        receipt = handoff._validate_remote_receipt_payload(  # noqa: SLF001
            receipt,
            submission=submission,
            result=result,
        )
        marker = handoff._validate_marker_payload(  # noqa: SLF001
            marker,
            expected=submission["retained_aedt"],
        )
    except Exception as exc:
        raise RecoveryError("remote retained AEDT metadata drifted") from exc
    if _sha256_bytes(marker_raw) != receipt["marker_sha256"]:
        raise RecoveryError("remote marker bytes differ from retained receipt")
    return receipt, marker


def _inventory_contract(
    inventory: Sequence[str], receipt: Mapping[str, Any]
) -> list[str]:
    paths = sorted(str(path) for path in inventory)
    expected_chunks = [
        f"{CHUNK_DIRECTORY}/{index:08d}.b64"
        for index in range(int(receipt["transport_chunk_count"]))
    ]
    expected = sorted(
        [ARTIFACT_PATH, RECEIPT_PATH, MARKER_PATH, *expected_chunks]
    )
    if paths != expected:
        raise terminal.TransientGetError(
            "task96743 retained remote inventory is incomplete"
        )
    return paths


def _reconstruct_chunks(
    *,
    client: Any,
    receipt: Mapping[str, Any],
    staging: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    count = int(receipt["transport_chunk_count"])
    size_expected = int(receipt["artifact_size_bytes"])
    if (
        count <= 0
        or not 0 < size_expected <= MAX_AEDT_BYTES
        or count != math.ceil(size_expected / RAW_CHUNK_BYTES)
    ):
        raise RecoveryError("remote retained AEDT size/chunk count drifted")
    required_free = size_expected + math.ceil(size_expected / 3) * 4 + 1024**3
    if shutil.disk_usage(staging).free < required_free:
        raise terminal.TransientGetError(
            "insufficient local free space for authenticated AEDT reconstruction"
        )
    chunks_root = staging / "symmetric.aedt.chunks"
    chunks_root.mkdir()
    partial = staging / ".symmetric.aedt.partial"
    digest = hashlib.sha256()
    written = 0
    inventory: list[dict[str, Any]] = []
    try:
        with partial.open("xb") as assembled:
            for index in range(count):
                remote_path = f"{CHUNK_DIRECTORY}/{index:08d}.b64"
                encoded = client.remote_file(
                    TASK_ID,
                    remote_path,
                    max_bytes=MAX_ENCODED_CHUNK_BYTES,
                )
                expected_raw = min(RAW_CHUNK_BYTES, size_expected - written)
                if len(encoded) != 4 * math.ceil(expected_raw / 3):
                    raise RecoveryError(f"encoded chunk size drifted: {index}")
                try:
                    raw = base64.b64decode(encoded, validate=True)
                except (ValueError, binascii.Error) as exc:
                    raise RecoveryError(
                        f"retained chunk base64 drifted: {index}"
                    ) from exc
                if len(raw) != expected_raw:
                    raise RecoveryError(f"raw chunk size drifted: {index}")
                local_chunk = chunks_root / f"{index:08d}.b64"
                _immutable_bytes(local_chunk, encoded)
                inventory.append(_file_record(local_chunk, relative_to=staging))
                assembled.write(raw)
                digest.update(raw)
                written += len(raw)
            assembled.flush()
            os.fsync(assembled.fileno())
        if (
            written != size_expected
            or digest.hexdigest() != receipt["artifact_sha256"]
        ):
            raise RecoveryError("reconstructed symmetric AEDT identity drifted")
        artifact = staging / "symmetric.aedt"
        os.replace(partial, artifact)
    finally:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass
    return artifact, inventory


def _copy_source_evidence(
    source: Mapping[str, Any], staging: Path
) -> dict[str, dict[str, Any]]:
    destination = staging / "source"
    destination.mkdir()
    source_paths = {
        "campaign_manifest.json": CAMPAIGN_PATH,
        "tuned_gap_manifest.json": TUNED_MANIFEST_PATH,
        "symmetric_loss_thermal.json": PARAMS_PATH,
        "goal_standard.json": PROFILE_PATH,
    }
    for name, path in source_paths.items():
        shutil.copy2(path, destination / name)
    records = {
        name: _file_record(destination / name, relative_to=staging)
        for name in source_paths
    }
    expected = {
        "campaign_manifest.json": source["records"]["campaign"]["sha256"],
        "tuned_gap_manifest.json": source["records"]["tuned_manifest"]["sha256"],
        "symmetric_loss_thermal.json": source["records"]["params"]["sha256"],
        "goal_standard.json": source["records"]["profile"]["sha256"],
    }
    if any(records[name]["sha256"] != sha for name, sha in expected.items()):
        raise RecoveryError("copied source evidence drifted")
    return records


def _existing_collection(root: Path) -> dict[str, Any] | None:
    collection = root / "collection"
    if not collection.is_dir():
        return None
    seal_path = collection / "collection_seal.json"
    receipt_path = collection / "collection_receipt.json"
    seal_value = final_pipeline.validate_seal(
        _json_object(seal_path, "collection seal"),
        schema=COLLECTION_SEAL_SCHEMA,
        label="collection seal",
    )
    receipt = final_pipeline.validate_seal(
        _json_object(receipt_path, "collection receipt"),
        schema=COLLECTION_SCHEMA,
        label="collection receipt",
    )
    if (
        seal_value.get("collection_receipt") != _file_record(
            receipt_path, relative_to=collection
        )
        or seal_value.get("collection_receipt_payload_sha256")
        != receipt["payload_sha256"]
        or receipt.get("task_id") != TASK_ID
    ):
        raise RecoveryError("existing task96743 collection drifted")
    return receipt


def _winner_authority(
    *,
    root: Path,
    collection: Mapping[str, Any],
    source: Mapping[str, Any],
) -> Path | None:
    attestation = collection["result_attestation"]
    if (
        attestation.get("measured_hard_constraints_passed") is not True
        or attestation.get(
            "graded_capacitance_provenance_authenticated"
        )
        is not True
    ):
        return None
    actuals = attestation["measured_actuals"]
    collection_root = root / "collection"
    receipt_path = collection_root / "collection_receipt.json"
    artifact_path = collection_root / "symmetric.aedt"
    authority = final_pipeline.seal(
        {
            "schema_version": final_pipeline.WINNER_AUTHORITY_SCHEMA,
            "source": {
                "task_id": TASK_ID,
                "candidate_physics_sha256": CANDIDATE_PHYSICS_SHA256,
                "selection_performed_upstream": True,
                "selection_authority_kind": (
                    "user_directed_exact_task96743_terminal_hard_pass"
                ),
                "candidate_promotion_performed_by_pipeline": False,
                "selected_symmetric_hard_pass": True,
                "upstream_selection_receipt": _file_record(receipt_path),
                "retained_symmetric_aedt": _file_record(artifact_path),
            },
            "contracts": {
                "goal_contract_schema": GOAL_CONTRACT_SCHEMA,
                "goal_stage_spec_sha256": GOAL_STAGE_SPEC_SHA256,
                "goal_temperature_contract_sha256": (
                    GOAL_TEMPERATURE_CONTRACT_SHA256
                ),
                "fixed_operating_identity_sha256": (
                    FIXED_OPERATING_IDENTITY_SHA256
                ),
                "fixed_cooling_identity_sha256": (
                    FIXED_COOLING_IDENTITY_SHA256
                ),
                "fixed_boundary_contract_sha256": (
                    FIXED_BOUNDARY_CONTRACT_SHA256
                ),
            },
            "params": copy.deepcopy(source["effective_params"]),
            "symmetric_verification": {
                "full_model": 0,
                "round_corner": 0,
                "matrix_solved": True,
                "capacitance_solved": True,
                "loss_solved": True,
                "thermal_solved": True,
                "measured_hard_constraints_passed": True,
                "rounded_fea_used": False,
                "actual_dimensions_mm": copy.deepcopy(
                    actuals["dimensions_mm"]
                ),
                "actual_resonance_Hz": actuals["resonance_Hz"],
                "actual_Lm_primary_referred_H": actuals[
                    "Lm_primary_referred_H"
                ],
                "actual_temperature_family_max_C": copy.deepcopy(
                    actuals["temperature_family_max_C"]
                ),
                "fixed_boundary": {
                    "fan_velocity_m_s": 1.5,
                    "fan_config": "dual",
                    "core_plate_pad_t_mm": 2.0,
                    "wcp_pad_t_mm": 2.0,
                    "thermal_pad_conductivity_W_mK": 0.2,
                    "TIM_mutated": False,
                },
            },
        }
    )
    path = _immutable_json(root / "winner_authority.json", authority)
    final_pipeline.validate_winner_authority(
        authority, authority_directory=path.parent
    )
    return path


def collect_success(
    *,
    root: Path,
    plan: Mapping[str, Any],
    task: Mapping[str, Any],
    client: Any,
) -> dict[str, Any]:
    existing = _existing_collection(root)
    source = load_source_contract()
    if existing is not None:
        winner = _winner_authority(
            root=root, collection=existing, source=source
        )
        return {
            "event": "already_collected",
            "collection": str(root / "collection"),
            "winner_authority": str(winner) if winner else None,
            "measured_hard_constraints_passed": existing[
                "result_attestation"
            ]["measured_hard_constraints_passed"],
        }
    stdout = client.task_output(
        TASK_ID, "stdout", max_bytes=MAX_TASK_OUTPUT_BYTES
    )
    stderr = client.task_output(
        TASK_ID, "stderr", max_bytes=MAX_TASK_OUTPUT_BYTES
    )
    result = _extract_result(stdout)
    result_attestation = authenticate_result(result, source=source)
    receipt_raw = client.remote_file(
        TASK_ID, RECEIPT_PATH, max_bytes=MAX_METADATA_BYTES
    )
    marker_raw = client.remote_file(
        TASK_ID, MARKER_PATH, max_bytes=MAX_METADATA_BYTES
    )
    if not receipt_raw or not marker_raw:
        raise terminal.TransientGetError(
            "task96743 retention metadata is not published yet"
        )
    receipt, marker = _validate_remote_metadata(
        receipt_raw=receipt_raw,
        marker_raw=marker_raw,
        result=result,
        plan=plan,
    )
    remote_inventory = _inventory_contract(
        client.remote_files(TASK_ID, f"{RETAINED_ROOT}/**"),
        receipt,
    )

    destination = root / "collection"
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".collection.",
            suffix=".tmp",
            dir=root,
        )
    )
    try:
        artifact, chunk_inventory = _reconstruct_chunks(
            client=client, receipt=receipt, staging=staging
        )
        (staging / "task.sh").write_bytes(
            (root / "recovery_source" / "task.sh").read_bytes()
        )
        (staging / "scheduler_stdout.log").write_bytes(stdout)
        (staging / "scheduler_stderr.log").write_bytes(stderr)
        (staging / "terminal_task.json").write_bytes(
            final_pipeline.canonical_bytes(task) + b"\n"
        )
        (staging / "result.json").write_bytes(
            final_pipeline.canonical_bytes(result) + b"\n"
        )
        (staging / "remote_retained_receipt.json").write_bytes(receipt_raw)
        (staging / "prune_protection_marker.json").write_bytes(marker_raw)
        (staging / "remote_inventory.json").write_bytes(
            final_pipeline.canonical_bytes(remote_inventory) + b"\n"
        )
        source_records = _copy_source_evidence(source, staging)
        evidence_records = {
            name: _file_record(staging / name, relative_to=staging)
            for name in (
                "task.sh",
                "scheduler_stdout.log",
                "scheduler_stderr.log",
                "terminal_task.json",
                "result.json",
                "remote_retained_receipt.json",
                "prune_protection_marker.json",
                "remote_inventory.json",
            )
        }
        collection = final_pipeline.seal(
            {
                "schema_version": COLLECTION_SCHEMA,
                "created_at_utc": _now(),
                "task_id": TASK_ID,
                "task_name": TASK_NAME,
                "dedupe_key": DEDUPE_KEY,
                "candidate_physics_sha256": CANDIDATE_PHYSICS_SHA256,
                "solver_revision": SOLVER_REVISION,
                "library_revision": LIBRARY_REVISION,
                "parameter_digest": PARAMETER_DIGEST,
                "recovery_plan": _file_record(
                    root / "recovery_plan.json"
                ),
                "recovery_plan_payload_sha256": plan["payload_sha256"],
                "terminal_task_sha256": tuner._sha(task),  # noqa: SLF001
                "result": copy.deepcopy(result),
                "result_sha256": tuner._sha(result),  # noqa: SLF001
                "result_attestation": result_attestation,
                "remote_retained_receipt": copy.deepcopy(receipt),
                "prune_protection_marker": copy.deepcopy(marker),
                "remote_inventory": remote_inventory,
                "retained_symmetric_aedt": _file_record(
                    artifact, relative_to=staging
                ),
                "retained_chunks": chunk_inventory,
                "retained_chunk_inventory_sha256": tuner._sha(  # noqa: SLF001
                    chunk_inventory
                ),
                "evidence_files": evidence_records,
                "source_files": source_records,
                "cancelled_task96483_plan_used_as_authority": False,
                "cancelled_task96483_receipt_used_as_authority": False,
                "candidate_selection_performed_by_collector": False,
                "scheduler_get_only_collection": True,
                "scheduler_mutation_performed": False,
                "rounded_FEA_performed": False,
                "scientific_pass_claimed": result_attestation[
                    "measured_hard_constraints_passed"
                ],
                "production_claimed": False,
            }
        )
        receipt_path = _immutable_json(
            staging / "collection_receipt.json", collection
        )
        collection_seal = final_pipeline.seal(
            {
                "schema_version": COLLECTION_SEAL_SCHEMA,
                "created_at_utc": _now(),
                "task_id": TASK_ID,
                "candidate_physics_sha256": CANDIDATE_PHYSICS_SHA256,
                "collection_receipt": _file_record(
                    receipt_path, relative_to=staging
                ),
                "collection_receipt_payload_sha256": collection[
                    "payload_sha256"
                ],
                "artifact_sha256": receipt["artifact_sha256"],
                "artifact_size_bytes": receipt["artifact_size_bytes"],
                "all_declared_chunks_reconstructed": True,
                "scheduler_get_only_collection": True,
                "scheduler_mutation_performed": False,
                "cancelled_task96483_authority_reused": False,
            }
        )
        _immutable_json(staging / "collection_seal.json", collection_seal)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    winner = _winner_authority(
        root=root, collection=collection, source=source
    )
    return {
        "event": "collected",
        "collection": str(destination),
        "artifact_sha256": receipt["artifact_sha256"],
        "artifact_size_bytes": receipt["artifact_size_bytes"],
        "measured_hard_constraints_passed": result_attestation[
            "measured_hard_constraints_passed"
        ],
        "winner_blocking_reasons": result_attestation[
            "winner_blocking_reasons"
        ],
        "winner_authority": str(winner) if winner else None,
        "scheduler_get_only_collection": True,
        "scheduler_mutation_performed": False,
    }


def _failure_ledger(
    *, root: Path, plan: Mapping[str, Any], task: Mapping[str, Any], client: Any
) -> Path:
    stdout = client.task_output(
        TASK_ID, "stdout", max_bytes=MAX_TASK_OUTPUT_BYTES
    )
    stderr = client.task_output(
        TASK_ID, "stderr", max_bytes=MAX_TASK_OUTPUT_BYTES
    )
    ledger = final_pipeline.seal(
        {
            "schema_version": FAILURE_SCHEMA,
            "observed_at_utc": _now(),
            "task_id": TASK_ID,
            "task_name": TASK_NAME,
            "dedupe_key": DEDUPE_KEY,
            "recovery_plan": _file_record(root / "recovery_plan.json"),
            "recovery_plan_payload_sha256": plan["payload_sha256"],
            "terminal_task": copy.deepcopy(task),
            "terminal_task_sha256": tuner._sha(task),  # noqa: SLF001
            "stdout_sha256": _sha256_bytes(stdout),
            "stderr_sha256": _sha256_bytes(stderr),
            "failure_message": str(task.get("failure_message") or ""),
            "scientific_infeasibility_claimed": False,
            "candidate_selection_performed_by_collector": False,
            "scheduler_get_only_collection": True,
            "scheduler_mutation_performed": False,
            "cancelled_task96483_authority_reused": False,
        }
    )
    _immutable_bytes(root / "failure_stdout.log", stdout)
    _immutable_bytes(root / "failure_stderr.log", stderr)
    return _immutable_json(root / "failure_ledger.json", ledger)


def poll_once(*, output: Path, client: Any) -> tuple[bool, dict[str, Any]]:
    root = output.resolve()
    plan_path = prepare_recovery_plan(output=root, client=client)
    plan = load_recovery_plan(plan_path)
    source = load_source_contract()
    task = _client_task(client)
    _client_task_sh(client, source["effective_params"])
    status, state = _task_state(task)
    poll = final_pipeline.seal(
        {
            "schema_version": POLL_SCHEMA,
            "observed_at_utc": _now(),
            "task_id": TASK_ID,
            "task_name": TASK_NAME,
            "dedupe_key": DEDUPE_KEY,
            "status": status,
            "state": state,
            "task_snapshot_sha256": tuner._sha(task),  # noqa: SLF001
            "task_sh_sha256": TASK_SH_SHA256,
            "recovery_plan_payload_sha256": plan["payload_sha256"],
            "scheduler_get_only": True,
            "scheduler_mutation_performed": False,
        }
    )
    _immutable_json(root / "polls" / f"{_timestamp()}.json", poll)
    _replace_json(root / "latest_poll.json", poll)
    if (status, state) in TERMINAL_SUCCESS:
        try:
            result = collect_success(
                root=root, plan=plan, task=task, client=client
            )
        except terminal.TransientGetError as exc:
            return False, {
                "event": "terminal_success_pending_retention",
                "task_id": TASK_ID,
                "status": status,
                "state": state,
                "reason": str(exc),
            }
        return True, result
    if status in TERMINAL_FAILURE or state in TERMINAL_FAILURE:
        ledger = _failure_ledger(
            root=root, plan=plan, task=task, client=client
        )
        return True, {
            "event": "terminal_failure",
            "task_id": TASK_ID,
            "failure_ledger": str(ledger),
            "scientific_infeasibility_claimed": False,
        }
    return False, {
        "event": "active",
        "task_id": TASK_ID,
        "status": status,
        "state": state,
        "actual_node_name": task["actual_node_name"],
        "allocation_id": task["allocation_id"],
        "slurm_job_id": task["slurm_job_id"],
        "scheduler_get_only": True,
        "scheduler_mutation_performed": False,
    }


def _heartbeat(
    *,
    output: Path,
    state: str,
    detail: Mapping[str, Any],
    get_count: int,
) -> Path:
    value = final_pipeline.seal(
        {
            "schema_version": HEARTBEAT_SCHEMA,
            "observed_at_utc": _now(),
            "pid": os.getpid(),
            "task_id": TASK_ID,
            "watch_state": state,
            "detail": copy.deepcopy(dict(detail)),
            "scheduler_GET_count": get_count,
            "scheduler_mutation_performed": False,
        }
    )
    return _replace_json(output.resolve() / "heartbeat.json", value)


def watch(*, output: Path, client: Any, interval: float) -> dict[str, Any]:
    root = output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    pid = final_pipeline.seal(
        {
            "schema_version": PID_SCHEMA,
            "created_at_utc": _now(),
            "pid": os.getpid(),
            "task_id": TASK_ID,
            "scheduler_get_only": True,
            "scheduler_mutation_performed": False,
        }
    )
    _replace_json(root / "watcher.pid.json", pid)
    while True:
        try:
            done, detail = poll_once(output=root, client=client)
            _heartbeat(
                output=root,
                state="terminal" if done else "watching",
                detail=detail,
                get_count=int(getattr(client, "get_count", 0)),
            )
            print(
                json.dumps(detail, sort_keys=True, separators=(",", ":")),
                flush=True,
            )
            if done:
                return detail
        except terminal.TransientGetError as exc:
            detail = {
                "event": "transient_get_error",
                "error": str(exc),
                "task_id": TASK_ID,
            }
            _heartbeat(
                output=root,
                state="transient_get_error",
                detail=detail,
                get_count=int(getattr(client, "get_count", 0)),
            )
        time.sleep(interval)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scheduler-url", default=SCHEDULER_URL)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=20.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 10.0 <= args.interval <= 300.0:
        raise RecoveryError("watch interval must be within 10..300 seconds")
    if args.scheduler_url.rstrip("/") != SCHEDULER_URL:
        raise RecoveryError("task96743 Scheduler origin drifted")
    client = terminal.GetOnlyClient(SCHEDULER_URL, timeout_seconds=120.0)
    if args.prepare_only:
        plan = prepare_recovery_plan(output=args.output, client=client)
        result = {
            "event": "recovery_plan_prepared",
            "plan": str(plan),
            "scheduler_GET_count": client.get_count,
            "scheduler_mutation_performed": False,
        }
    elif args.once:
        done, result = poll_once(output=args.output, client=client)
        result["watch_complete"] = done
        result["scheduler_GET_count"] = client.get_count
    else:
        result = watch(
            output=args.output,
            client=client,
            interval=args.interval,
        )
    print(
        json.dumps(result, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RecoveryError, terminal.CollectorError) as exc:
        print(
            json.dumps(
                {
                    "event": "task96743_recovery_error",
                    "error": str(exc),
                    "scheduler_mutation_performed": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        raise SystemExit(2) from exc
