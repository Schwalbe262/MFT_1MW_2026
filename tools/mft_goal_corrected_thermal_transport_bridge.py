#!/usr/bin/env python3
"""Bridge one terminal corrected-thermal result through bounded GET files.

This is deliberately different from ``mft_goal_snapshot_live_aedt.py``:

* the source task must be terminal and successful;
* the source is the atomically retained GPFS package, never a live AEDT file;
* ``symmetric.aedt`` and ``corrected_result.json`` are authenticated together;
* source task/node/Slurm/plan identity is carried into every transport receipt.

The publisher is an MFT task payload.  It does not import, modify, or vendor
Slurm Scheduler source.  The optional ``prepare`` command only performs
Scheduler GET requests and emits a sealed, not-yet-submitted task plan.
"""

from __future__ import annotations

import argparse
import base64
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import stat
import sys
import time
from typing import Any, Callable, Mapping
from urllib import error, parse, request


AUTHORITY_SCHEMA = "mft-corrected-thermal-terminal-success-authority-v1"
PLAN_SCHEMA = "mft-corrected-thermal-terminal-transport-plan-v1"
TRANSPORT_SCHEMA = "mft-corrected-thermal-terminal-transport-v1"
TRANSPORT_RESULT_SCHEMA = "mft-corrected-thermal-terminal-transport-result-v1"
PENDING_SCHEMA = "mft-corrected-thermal-terminal-transport-pending-v1"
ATTEMPT_SCHEMA = "mft-corrected-thermal-terminal-transport-post-attempt-v1"
SUBMISSION_RECEIPT_SCHEMA = (
    "mft-corrected-thermal-terminal-transport-submission-receipt-v1"
)
SUBMISSION_SEAL_SCHEMA = "mft-corrected-thermal-terminal-transport-submission-seal-v1"
COLLECTION_SEAL_SCHEMA = "mft-corrected-thermal-terminal-transport-collection-seal-v1"
SOURCE_FAILURE_SCHEMA = "mft-corrected-thermal-terminal-transport-source-failure-v1"

POST_AUTHORIZATION_TOKEN = "AUTHORIZE_MFT_THERMAL_TRANSPORT_T96324_J839461_SINGLE_POST"
DEFAULT_ORCHESTRATION_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\postdeadline_thermal_transport_bridge_v1"
)
WATCH_INTERVAL_SECONDS = 60

SOURCE_TASK_ID = 96324
SOURCE_TASK_NAME = "mft-goal-corrected-thermal-l96230-b7c30cb70b95-postdeadline-r6-n111"
SOURCE_DEDUPE_KEY = (
    "mft-al:mft-goal-corrected-thermal-l96230-b7c30cb70b95-"
    "postdeadline-r6-n111:"
    "a0208331949f70c21cde948e853a331d7f7f9824:"
    "e6b9b9d20a832ff5c3f7ca97218737a0b8650781:"
    "086c88ea0cf2a41a"
)
SOURCE_ACCOUNT = "r1jae262"
SOURCE_NODE = "n111"
SOURCE_ALLOCATION_ID = 14644
SOURCE_SLURM_JOB_ID = "839461"
SOURCE_CPUS = 8
SOURCE_MEMORY_MB = 294_912
SOURCE_TIMEOUT_SECONDS = 45_000
SOURCE_REMOTE_DIR = "slurm_scheduler/runs/2026-07-26/task-96324-1785056187"
SOURCE_TASK_SH_SIZE = 10_115
SOURCE_TASK_SH_SHA256 = (
    "193ec5ed5e6a8eb983a950c29a04f158d3045e5e2fdb76aa6d1bfc28b02accf6"
)

SOURCE_PLAN_FILE_SHA256 = (
    "efe8ad5cd5991e28a774e23dd3c17a4889f9578d675ebd0e2a860d44b63b4d79"
)
SOURCE_PLAN_PAYLOAD_SHA256 = (
    "8e3e4de4ea23f993d1787199af5bc48f4895dbdbe4b46b983254ad030b651352"
)
SOURCE_SUBMISSION_FILE_SHA256 = (
    "81db11dd38900e2fc480b4615fce80908447e26cfb611a109c37a76097badf53"
)
SOURCE_SUBMISSION_PAYLOAD_SHA256 = (
    "c05740e6737b50a2fbeb80b1518b3d12a090ef1514346dcbcd33988409c2b2eb"
)
SOURCE_COMMAND_SHA256 = (
    "41dfe4f93c47a7dedba9e86ed5fb4cbb7481845d19e2a5e881d504e29ef63a7b"
)
SOURCE_CHECKPOINT_MANIFEST_SHA256 = (
    "ce0bf14dbdb83ceb0a4e5d00f387eb2273c53dba8c67edcccae73fa5a4aad43b"
)
SOURCE_CANDIDATE_SHA256 = (
    "b7c30cb70b95611821d233306688586985f75f026c246e4fcf87b4bb2ea63412"
)
SOURCE_EXECUTOR_REVISION = "a0208331949f70c21cde948e853a331d7f7f9824"
SOURCE_EXECUTION_PLAN_PAYLOAD_SHA256 = (
    "0d94990d26b96780b1bcf2c94af7d81b0d0fc132a7ed01068cf93fb7272fe981"
)
SOURCE_EXECUTION_PLAN_FILE_SHA256 = (
    "aeaee4f7862845dc852e7892a2e3ac711fc2595e6d53945226b62466ccb87b71"
)
SOURCE_SUBMISSION_CONTRACT_SHA256 = (
    "086c88ea0cf2a41a1d32089cd0ea796a09f5c0ba34003e632fccd478dc265df2"
)

SOURCE_RETAINED_ROOT = Path(
    "/gpfs/home1/r1jae262/slurm_scheduler/mft_goal_20260726/"
    "corrected_thermal_minimum_native_r5_n111/"
    "b7c-ce0bf14dbdb8-a0208331949f"
)
SOURCE_RETAINED_REMOTE = (
    "mft_goal_20260726/corrected_thermal_minimum_native_r5_n111/"
    "b7c-ce0bf14dbdb8-a0208331949f"
)
OUTPUT_RELATIVE_ROOT = "mft_goal_20260726/corrected_thermal_terminal_transport_v1"
TRANSPORT_DIRECTORY_NAME = "symmetric-from-t96324-j839461-terminal-transport-v1"

FIXED_PHYSICS = {
    "fan_velocity_m_per_s": 1.5,
    "thermal_pad_thickness_mm": 2.0,
    "tim_conductivity_w_per_mk": 0.2,
}
CLASSIFICATION = {
    "diagnostic_only": True,
    "canonical": False,
    "production_truth_eligible": False,
}

RAW_CHUNK_BYTES = 384_000
MAX_ENCODED_CHUNK_BYTES = 512_000
MAX_AEDT_BYTES = 512 * 1024 * 1024
MAX_JSON_BYTES = 1024 * 1024
MAX_TASK_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_CHUNK_COUNT = math.ceil(MAX_AEDT_BYTES / RAW_CHUNK_BYTES)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
NODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class BridgeError(RuntimeError):
    """Raised when the terminal transport cannot be proven safely."""


class PendingSource(BridgeError):
    """Raised by prepare when source task 96324 is not terminal yet."""


class SubmissionNotReady(BridgeError):
    """Raised when live capacity/health is not safe for the sole POST."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def thermal_canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value) + b"\n").hexdigest()


def sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "payload_sha256" in result:
        raise BridgeError("payload_sha256 already exists")
    result["payload_sha256"] = canonical_sha256(result)
    return result


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: Any, label: str) -> str:
    text = str(value or "").strip().casefold()
    if SHA256_RE.fullmatch(text) is None:
        raise BridgeError(f"{label} is not lowercase SHA-256")
    return text


def _git_revision(value: Any, label: str) -> str:
    text = str(value or "").strip().casefold()
    if GIT_SHA1_RE.fullmatch(text) is None:
        raise BridgeError(f"{label} is not lowercase Git SHA-1")
    return text


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool):
        raise BridgeError(f"{label} is not a valid integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BridgeError(f"{label} is not a valid integer") from exc
    if parsed < minimum:
        raise BridgeError(f"{label} is not a valid integer")
    return parsed


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BridgeError(f"{label} is not an object")
    return copy.deepcopy(dict(value))


def _safe_relative(value: str, label: str) -> PurePosixPath:
    raw = str(value or "")
    pure = PurePosixPath(raw)
    if (
        not raw
        or "\\" in raw
        or "\x00" in raw
        or pure.is_absolute()
        or ".." in pure.parts
        or any(
            part in {"", "."} or SAFE_COMPONENT_RE.fullmatch(part) is None
            for part in pure.parts
        )
    ):
        raise BridgeError(f"{label} is not a safe relative path")
    return pure


def _regular_file(
    path: Path,
    label: str,
    *,
    maximum_bytes: int | None = None,
    expected_uid: int | None = None,
) -> os.stat_result:
    metadata = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or (expected_uid is not None and metadata.st_uid != expected_uid)
        or (maximum_bytes is not None and not 0 <= metadata.st_size <= maximum_bytes)
    ):
        raise BridgeError(f"{label} is not one bounded regular file")
    return metadata


def _plain_directory(
    path: Path,
    label: str,
    *,
    expected_uid: int | None = None,
) -> os.stat_result:
    metadata = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or (expected_uid is not None and metadata.st_uid != expected_uid)
    ):
        raise BridgeError(f"{label} is not one plain directory")
    return metadata


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _regular_file(path, label, maximum_bytes=MAX_JSON_BYTES)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BridgeError(f"{label} is invalid JSON") from exc
    return _mapping(value, label)


def _parse_json_bytes(value: bytes, label: str) -> dict[str, Any]:
    if len(value) > MAX_JSON_BYTES:
        raise BridgeError(f"{label} exceeds its JSON bound")
    try:
        parsed = json.loads(value)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise BridgeError(f"{label} is invalid JSON") from exc
    return _mapping(parsed, label)


def _verify_seal(
    value: Mapping[str, Any],
    *,
    schema: str,
    thermal_newline: bool = False,
) -> dict[str, Any]:
    result = _mapping(value, schema)
    claimed = _sha(result.get("payload_sha256"), f"{schema} payload")
    unsigned = dict(result)
    unsigned.pop("payload_sha256", None)
    actual = (
        thermal_canonical_sha256(unsigned)
        if thermal_newline
        else canonical_sha256(unsigned)
    )
    if result.get("schema") != schema or claimed != actual:
        raise BridgeError(f"{schema} seal is invalid")
    return result


def _atomic_json(path: Path, value: Mapping[str, Any]) -> Path:
    payload = canonical_bytes(value) + b"\n"
    if len(payload) > MAX_JSON_BYTES:
        raise BridgeError(f"{path.name} exceeds the 1 MiB GET bound")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> Path:
    """Durably consume one evidence path without replacement."""
    payload = canonical_bytes(value) + b"\n"
    if len(payload) > MAX_JSON_BYTES:
        raise BridgeError(f"{path.name} exceeds the 1 MiB evidence bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.chmod(path, 0o400)
    except OSError:
        pass
    return path


def _task_exact_fields() -> dict[str, Any]:
    return {
        "task_id": SOURCE_TASK_ID,
        "name": SOURCE_TASK_NAME,
        "dedupe_key": SOURCE_DEDUPE_KEY,
        "account_name": SOURCE_ACCOUNT,
        "allocation_id": SOURCE_ALLOCATION_ID,
        "slurm_job_id": SOURCE_SLURM_JOB_ID,
        "node_name": SOURCE_NODE,
        "actual_node_name": SOURCE_NODE,
        "cpus": SOURCE_CPUS,
        "memory_mb": SOURCE_MEMORY_MB,
        "timeout_seconds": SOURCE_TIMEOUT_SECONDS,
        "remote_dir": SOURCE_REMOTE_DIR,
    }


def _plan_exact_fields() -> dict[str, Any]:
    return {
        "file_sha256": SOURCE_PLAN_FILE_SHA256,
        "payload_sha256": SOURCE_PLAN_PAYLOAD_SHA256,
        "submission_file_sha256": SOURCE_SUBMISSION_FILE_SHA256,
        "submission_payload_sha256": SOURCE_SUBMISSION_PAYLOAD_SHA256,
        "canonical_command_sha256": SOURCE_COMMAND_SHA256,
        "submission_contract_sha256": SOURCE_SUBMISSION_CONTRACT_SHA256,
        "execution_plan_file_sha256": SOURCE_EXECUTION_PLAN_FILE_SHA256,
        "execution_plan_payload_sha256": (SOURCE_EXECUTION_PLAN_PAYLOAD_SHA256),
    }


def validate_source_plan_files(
    plan_path: Path,
    submission_path: Path,
) -> dict[str, Any]:
    plan_bytes = plan_path.read_bytes()
    submission_bytes = submission_path.read_bytes()
    if sha256_bytes(plan_bytes) != SOURCE_PLAN_FILE_SHA256:
        raise BridgeError("task96324 plan file SHA drifted")
    if sha256_bytes(submission_bytes) != SOURCE_SUBMISSION_FILE_SHA256:
        raise BridgeError("task96324 submission file SHA drifted")
    plan = _parse_json_bytes(plan_bytes, "task96324 plan")
    submission = _parse_json_bytes(submission_bytes, "task96324 submission receipt")
    task_identity = _mapping(plan.get("task_identity"), "plan task identity")
    readback = _mapping(submission.get("task_readback"), "submission task readback")
    if (
        plan.get("plan_payload_sha256") != SOURCE_PLAN_PAYLOAD_SHA256
        or plan.get("canonical_command_sha256") != SOURCE_COMMAND_SHA256
        or plan.get("canonical") is not False
        or plan.get("diagnostic_only") is not True
        or plan.get("production_truth_eligible") is not False
        or task_identity.get("name") != SOURCE_TASK_NAME
        or task_identity.get("dedupe_key") != SOURCE_DEDUPE_KEY
        or task_identity.get("candidate_sha256") != SOURCE_CANDIDATE_SHA256
        or plan.get("contract", {}).get("fixed_physics") != FIXED_PHYSICS
        or plan.get("execution_contract", {}).get("checkpoint_manifest_sha256")
        != SOURCE_CHECKPOINT_MANIFEST_SHA256
        or plan.get("execution_contract", {}).get("executor_revision")
        != SOURCE_EXECUTOR_REVISION
        or plan.get("execution_contract", {}).get("plan_payload_sha256")
        != SOURCE_EXECUTION_PLAN_PAYLOAD_SHA256
        or plan.get("execution_contract", {}).get("submission_contract_sha256")
        != SOURCE_SUBMISSION_CONTRACT_SHA256
        or submission.get("payload_sha256") != SOURCE_SUBMISSION_PAYLOAD_SHA256
        or submission.get("command_sha256") != SOURCE_COMMAND_SHA256
        or submission.get("scheduler_post_calls") != 1
        or submission.get("scheduler_submission_performed") is not True
        or readback.get("task_id") != SOURCE_TASK_ID
        or readback.get("name") != SOURCE_TASK_NAME
        or readback.get("dedupe_key") != SOURCE_DEDUPE_KEY
    ):
        raise BridgeError("task96324 plan/submission identity drifted")
    return {
        "plan": plan,
        "submission": submission,
        "identity": _plan_exact_fields(),
    }


def _last_prefixed_json(data: bytes, prefix: str) -> dict[str, Any] | None:
    result = None
    for line in data.decode("utf-8", errors="replace").splitlines():
        if not line.startswith(prefix):
            continue
        try:
            parsed = json.loads(line[len(prefix) :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            result = parsed
    return result


def _validate_source_task_identity(task: Mapping[str, Any]) -> dict[str, Any]:
    value = _mapping(task, "source task readback")
    for key, expected in _task_exact_fields().items():
        if value.get(key) != expected:
            raise BridgeError(f"source task field drifted: {key}={value.get(key)!r}")
    return value


def _validate_success_task(task: Mapping[str, Any]) -> dict[str, Any]:
    value = _validate_source_task_identity(task)
    if (
        value.get("status") != "completed"
        or value.get("state") != "succeeded"
        or value.get("exit_code") != 0
        or not value.get("started_at")
        or not value.get("finished_at")
    ):
        raise BridgeError("source task is not terminal-success")
    return value


def _validate_marker(
    marker: Mapping[str, Any],
    *,
    expected_destination: str | None = None,
) -> dict[str, Any]:
    value = _verify_seal(
        marker,
        schema="mft-corrected-thermal-retention-marker-v1",
        thermal_newline=True,
    )
    if (
        value.get("diagnostic_only") is not True
        or value.get("canonical") is not False
        or value.get("production_truth_eligible") is not False
        or value.get("task_name") != SOURCE_TASK_NAME
        or value.get("dedupe_key") != SOURCE_DEDUPE_KEY
        or value.get("submission_contract_sha256") != SOURCE_SUBMISSION_CONTRACT_SHA256
        or value.get("execution_plan_payload_sha256")
        != SOURCE_EXECUTION_PLAN_PAYLOAD_SHA256
        or value.get("execution_exit_code") != 0
        or (
            expected_destination is not None
            and value.get("retained_destination") != expected_destination
        )
    ):
        raise BridgeError("corrected-thermal success marker drifted")
    return value


def build_terminal_success_authority(
    *,
    task: Mapping[str, Any],
    stdout: bytes,
    stderr: bytes,
    plan_path: Path,
    submission_path: Path,
    observed_at_utc: str | None = None,
) -> dict[str, Any]:
    """Create the GET-derived authority only after exact terminal success."""
    source_task = _validate_success_task(task)
    plan_evidence = validate_source_plan_files(plan_path, submission_path)
    if len(stdout) > MAX_TASK_OUTPUT_BYTES or len(stderr) > MAX_TASK_OUTPUT_BYTES:
        raise BridgeError("source task output exceeds the bounded GET window")
    marker_raw = _last_prefixed_json(stdout, "CORRECTED_THERMAL_JSON ")
    if marker_raw is None:
        raise BridgeError("terminal source stdout has no success marker")
    marker = _validate_marker(
        marker_raw,
        expected_destination=str(SOURCE_RETAINED_ROOT),
    )
    return sealed(
        {
            "schema": AUTHORITY_SCHEMA,
            **CLASSIFICATION,
            "transport_authorized": True,
            "source_task": {
                key: copy.deepcopy(source_task.get(key))
                for key in (
                    *tuple(_task_exact_fields()),
                    "status",
                    "state",
                    "exit_code",
                    "started_at",
                    "finished_at",
                )
            },
            "source_task_files": {
                "task_sh": {
                    "relative_path": ("2026-07-26/task-96324-1785056187/task.sh"),
                    "size_bytes": SOURCE_TASK_SH_SIZE,
                    "sha256": SOURCE_TASK_SH_SHA256,
                },
                "stdout": {
                    "relative_path": ("2026-07-26/task-96324-1785056187/stdout.log"),
                    "size_bytes": len(stdout),
                    "sha256": sha256_bytes(stdout),
                },
                "stderr": {
                    "relative_path": ("2026-07-26/task-96324-1785056187/stderr.log"),
                    "size_bytes": len(stderr),
                    "sha256": sha256_bytes(stderr),
                },
                "exit_code": {
                    "relative_path": ("2026-07-26/task-96324-1785056187/exit_code"),
                    "parsed_value": 0,
                },
            },
            "source_plan": plan_evidence["identity"],
            "source_success_marker": marker,
            "source_retained_root": str(SOURCE_RETAINED_ROOT),
            "source_retained_remote": SOURCE_RETAINED_REMOTE,
            "observed_at_utc": observed_at_utc
            or datetime.now(timezone.utc).isoformat(),
            "authority_source": "scheduler_get_only_plus_sealed_local_plan",
            "scheduler_mutations": 0,
        }
    )


def _verify_authority(value: Mapping[str, Any]) -> dict[str, Any]:
    authority = _verify_seal(value, schema=AUTHORITY_SCHEMA)
    if (
        authority.get("diagnostic_only") is not True
        or authority.get("canonical") is not False
        or authority.get("production_truth_eligible") is not False
        or authority.get("transport_authorized") is not True
        or authority.get("source_retained_root") != str(SOURCE_RETAINED_ROOT)
        or authority.get("source_retained_remote") != SOURCE_RETAINED_REMOTE
        or authority.get("source_plan") != _plan_exact_fields()
        or authority.get("scheduler_mutations") != 0
    ):
        raise BridgeError("terminal-success authority contract drifted")
    _validate_success_task(
        _mapping(authority.get("source_task"), "authority source task")
    )
    _validate_marker(
        _mapping(
            authority.get("source_success_marker"),
            "authority success marker",
        ),
        expected_destination=str(SOURCE_RETAINED_ROOT),
    )
    return authority


def _verify_task_files(
    task_directory: Path,
    authority: Mapping[str, Any],
    *,
    expected_uid: int | None,
) -> tuple[bytes, bytes]:
    expected_suffix = PurePosixPath("2026-07-26/task-96324-1785056187")
    if task_directory.as_posix().split("/")[-2:] != list(expected_suffix.parts):
        raise BridgeError("source task directory identity drifted")
    _plain_directory(task_directory, "source task directory", expected_uid=expected_uid)
    rows = _mapping(authority.get("source_task_files"), "authority source task files")
    for label in ("task_sh", "stdout", "stderr", "exit_code"):
        row = _mapping(rows.get(label), f"authority {label}")
        relative = _safe_relative(str(row.get("relative_path")), label)
        if PurePosixPath(*relative.parts[-2:]).as_posix() != (
            "task-96324-1785056187/"
            + {
                "task_sh": "task.sh",
                "stdout": "stdout.log",
                "stderr": "stderr.log",
                "exit_code": "exit_code",
            }[label]
        ):
            raise BridgeError(f"authority {label} relative path drifted")

    task_sh = task_directory / "task.sh"
    task_metadata = _regular_file(
        task_sh,
        "source task.sh",
        maximum_bytes=MAX_JSON_BYTES,
        expected_uid=expected_uid,
    )
    if (
        task_metadata.st_size != SOURCE_TASK_SH_SIZE
        or sha256_file(task_sh) != SOURCE_TASK_SH_SHA256
    ):
        raise BridgeError("source task.sh identity drifted")
    for sentinel in (
        f"export SLURM_SCHED_TASK_ID={SOURCE_TASK_ID}".encode(),
        b"strict node placement mismatch: expected n111",
        b"strict allocation mismatch: expected 839461",
    ):
        if sentinel not in task_sh.read_bytes():
            raise BridgeError("source task.sh placement sentinel is absent")

    exit_code = task_directory / "exit_code"
    _regular_file(
        exit_code,
        "source exit_code",
        maximum_bytes=64,
        expected_uid=expected_uid,
    )
    try:
        parsed_exit = int(exit_code.read_text(encoding="ascii").strip())
    except (UnicodeError, ValueError) as exc:
        raise BridgeError("source exit_code is invalid") from exc
    if parsed_exit != 0:
        raise BridgeError("source exit_code is not zero")

    output: dict[str, bytes] = {}
    for label, filename in (("stdout", "stdout.log"), ("stderr", "stderr.log")):
        path = task_directory / filename
        row = _mapping(rows.get(label), f"authority {label}")
        metadata = _regular_file(
            path,
            f"source {label}",
            maximum_bytes=MAX_TASK_OUTPUT_BYTES,
            expected_uid=expected_uid,
        )
        data = path.read_bytes()
        if metadata.st_size != _positive_int(
            row.get("size_bytes"),
            f"authority {label} size",
            allow_zero=True,
        ) or sha256_bytes(data) != _sha(row.get("sha256"), f"authority {label} SHA"):
            raise BridgeError(f"source {label} identity drifted")
        output[label] = data
    marker = _last_prefixed_json(output["stdout"], "CORRECTED_THERMAL_JSON ")
    if marker != authority.get("source_success_marker"):
        raise BridgeError("source stdout marker differs from authority")
    return output["stdout"], output["stderr"]


def _authenticate_retained_package(
    root: Path,
    marker: Mapping[str, Any],
    *,
    expected_uid: int | None,
) -> dict[str, Any]:
    _plain_directory(root, "retained package root", expected_uid=expected_uid)
    if root.name != "b7c-ce0bf14dbdb8-a0208331949f":
        raise BridgeError("retained package label drifted")
    manifest_path = root / "manifest.json"
    manifest = _verify_seal(
        _read_json(manifest_path, "retained package manifest"),
        schema="mft-corrected-thermal-minimum-retained-package-v1",
        thermal_newline=True,
    )
    if (
        manifest.get("diagnostic_only") is not True
        or manifest.get("canonical") is not False
        or manifest.get("candidate_sha256") != SOURCE_CANDIDATE_SHA256
        or manifest.get("checkpoint_manifest_sha256")
        != SOURCE_CHECKPOINT_MANIFEST_SHA256
        or manifest.get("executor_solver_revision") != SOURCE_EXECUTOR_REVISION
        or marker.get("retention_manifest_payload_sha256")
        != manifest.get("payload_sha256")
        or marker.get("retention_manifest_file_sha256") != sha256_file(manifest_path)
    ):
        raise BridgeError("retained package manifest identity drifted")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or len(raw_files) != 5:
        raise BridgeError("retained package file inventory drifted")
    rows: dict[str, dict[str, Any]] = {}
    total = 0
    for raw in raw_files:
        row = _mapping(raw, "retained package file row")
        if set(row) != {"path", "size_bytes", "sha256"}:
            raise BridgeError("retained package file row fields drifted")
        relative = _safe_relative(str(row["path"]), "retained file")
        text = relative.as_posix()
        if text in rows:
            raise BridgeError("retained package has duplicate file rows")
        path = root.joinpath(*relative.parts)
        size = _positive_int(
            row.get("size_bytes"),
            f"retained size {text}",
            allow_zero=True,
        )
        metadata = _regular_file(
            path,
            f"retained file {text}",
            maximum_bytes=(
                MAX_AEDT_BYTES if text == "symmetric.aedt" else MAX_JSON_BYTES
            ),
            expected_uid=expected_uid,
        )
        digest = _sha(row.get("sha256"), f"retained SHA {text}")
        if metadata.st_size != size or sha256_file(path) != digest:
            raise BridgeError(f"retained file authentication failed: {text}")
        rows[text] = {**row, "path_object": path}
        total += size
    if (
        set(rows).issuperset(
            {
                "symmetric.aedt",
                "corrected_result.json",
                "execution_receipt.json",
            }
        )
        is False
        or len([name for name in rows if name.startswith("convergence/")]) != 1
        or len([name for name in rows if name.startswith("profile/")]) != 1
        or total != manifest.get("minimum_bundle_bytes")
    ):
        raise BridgeError("retained package member contract drifted")

    verified_marker_rows = marker.get("verified_files")
    normalized_rows = [
        {
            "path": name,
            "size_bytes": int(rows[name]["size_bytes"]),
            "sha256": str(rows[name]["sha256"]),
        }
        for name in rows
    ]
    if (
        not isinstance(verified_marker_rows, list)
        or sorted(
            verified_marker_rows,
            key=lambda item: str(item.get("path", ""))
            if isinstance(item, Mapping)
            else "",
        )
        != sorted(normalized_rows, key=lambda item: item["path"])
        or marker.get("minimum_bundle_bytes") != total
    ):
        raise BridgeError("success marker/file manifest binding drifted")

    execution_path = rows["execution_receipt.json"]["path_object"]
    corrected_path = rows["corrected_result.json"]["path_object"]
    aedt_path = rows["symmetric.aedt"]["path_object"]
    execution = _read_json(execution_path, "retained execution receipt")
    corrected = _read_json(corrected_path, "retained corrected result")
    if (
        execution.get("schema") != "mft-corrected-thermal-checkpoint-execution-v1"
        or execution.get("status") != "diagnostic_complete"
        or execution.get("diagnostic_only") is not True
        or execution.get("canonical") is not False
        or execution.get("production_truth_eligible") is not False
        or execution.get("source_checkpoint_manifest_sha256")
        != SOURCE_CHECKPOINT_MANIFEST_SHA256
        or execution.get("fixed_physics") != FIXED_PHYSICS
        or marker.get("corrected_receipt_sha256") != sha256_file(execution_path)
        or corrected.get("schema") != "mft-corrected-thermal-diagnostic-result-v1"
        or corrected.get("diagnostic_only") is not True
        or corrected.get("canonical") is not False
        or corrected.get("source_provenance") != execution.get("source_provenance")
        or corrected.get("executor_provenance") != execution.get("executor_provenance")
        or corrected.get("constraint_observation")
        != execution.get("constraint_observation")
        or corrected.get("temperatures") != execution.get("temperatures")
    ):
        raise BridgeError("retained execution/result binding drifted")
    executor = _mapping(
        execution.get("executor_provenance"),
        "retained executor provenance",
    )
    core = _mapping(
        execution.get("solver_core_contract"),
        "retained solver core contract",
    )
    if (
        executor.get("executor_solver_revision") != SOURCE_EXECUTOR_REVISION
        or executor.get("execution_plan_sha256") != SOURCE_EXECUTION_PLAN_FILE_SHA256
        or executor.get("execution_plan_payload_sha256")
        != SOURCE_EXECUTION_PLAN_PAYLOAD_SHA256
        or core.get("scheduler_task_id") != SOURCE_TASK_ID
        or core.get("slurm_job_id") != int(SOURCE_SLURM_JOB_ID)
        or core.get("slurm_cpus_per_task") != SOURCE_CPUS
        or core.get("passed") is not True
    ):
        raise BridgeError("retained solver task/plan identity drifted")
    observation = _mapping(
        corrected.get("constraint_observation"),
        "corrected constraint observation",
    )
    temperatures = _mapping(corrected.get("temperatures"), "corrected temperatures")
    if (
        observation.get("winding_limit_c") != 100.0
        or observation.get("core_limit_c") != 120.0
        or not isinstance(observation.get("all_temperature_constraints_pass"), bool)
        or not temperatures
    ):
        raise BridgeError("corrected thermal constraint observation drifted")
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "rows": rows,
        "execution": execution,
        "execution_path": execution_path,
        "corrected": corrected,
        "corrected_path": corrected_path,
        "aedt_path": aedt_path,
        "constraint_observation": observation,
        "temperatures": temperatures,
    }


def _write_chunk(path: Path, raw: bytes) -> dict[str, Any]:
    encoded = base64.b64encode(raw)
    if len(encoded) > MAX_ENCODED_CHUNK_BYTES:
        raise BridgeError("encoded transport chunk exceeded 512,000 bytes")
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "filename": path.name,
        "raw_size_bytes": len(raw),
        "raw_sha256": sha256_bytes(raw),
        "encoded_size_bytes": len(encoded),
        "encoded_sha256": sha256_bytes(encoded),
    }


def _validate_existing_transport(
    destination: Path,
    *,
    authority: Mapping[str, Any],
    expected_uid: int | None,
) -> dict[str, Any]:
    receipt_path = destination / "transport_receipt.json"
    receipt = _verify_seal(
        _read_json(receipt_path, "existing transport receipt"),
        schema=TRANSPORT_SCHEMA,
    )
    if (
        receipt.get("source_authority_payload_sha256")
        != authority.get("payload_sha256")
        or receipt.get("source_task") != authority.get("source_task")
        or receipt.get("source_plan") != authority.get("source_plan")
        or receipt.get("raw_chunk_bytes") != RAW_CHUNK_BYTES
        or receipt.get("maximum_encoded_chunk_bytes") != MAX_ENCODED_CHUNK_BYTES
    ):
        raise BridgeError("existing transport identity drifted")
    rows = receipt.get("chunks")
    if (
        not isinstance(rows, list)
        or len(rows) != receipt.get("chunk_count")
        or not 0 < len(rows) <= MAX_CHUNK_COUNT
    ):
        raise BridgeError("existing transport chunk inventory drifted")
    digest = hashlib.sha256()
    size = 0
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, "existing transport chunk")
        expected_name = f"{index:08d}.b64"
        if row.get("filename") != expected_name:
            raise BridgeError("existing chunk order drifted")
        chunk = destination / "chunks" / expected_name
        _regular_file(
            chunk,
            "existing encoded chunk",
            maximum_bytes=MAX_ENCODED_CHUNK_BYTES,
            expected_uid=expected_uid,
        )
        encoded = chunk.read_bytes()
        if len(encoded) != row.get("encoded_size_bytes") or sha256_bytes(
            encoded
        ) != row.get("encoded_sha256"):
            raise BridgeError("existing encoded chunk drifted")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise BridgeError("existing chunk is invalid base64") from exc
        if len(raw) != row.get("raw_size_bytes") or sha256_bytes(raw) != row.get(
            "raw_sha256"
        ):
            raise BridgeError("existing raw chunk drifted")
        digest.update(raw)
        size += len(raw)
    if size != receipt.get("artifact_size_bytes") or digest.hexdigest() != receipt.get(
        "artifact_sha256"
    ):
        raise BridgeError("existing transport reconstruction drifted")
    for name, expected_sha in (
        (
            "source_authority.json",
            receipt.get("source_authority_file_sha256"),
        ),
        (
            "corrected_result.json",
            receipt.get("corrected_result_sha256"),
        ),
        (
            "source_manifest.json",
            receipt.get("source_manifest_file_sha256"),
        ),
        (
            "source_execution_receipt.json",
            receipt.get("source_execution_receipt_sha256"),
        ),
    ):
        path = destination / name
        _regular_file(
            path,
            f"existing {name}",
            maximum_bytes=MAX_JSON_BYTES,
            expected_uid=expected_uid,
        )
        if sha256_file(path) != expected_sha:
            raise BridgeError(f"existing {name} drifted")
    return {
        "schema": TRANSPORT_RESULT_SCHEMA,
        "status": "existing_authenticated",
        "transport_directory": str(destination),
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "artifact_size_bytes": size,
        "artifact_sha256": digest.hexdigest(),
        "corrected_result_sha256": receipt["corrected_result_sha256"],
        "chunk_count": len(rows),
        "payload_sha256": receipt["payload_sha256"],
        **CLASSIFICATION,
    }


def publish_terminal_transport(
    *,
    authority_path: Path,
    source_retained_root: Path,
    source_task_directory: Path,
    output_relative_root: str = OUTPUT_RELATIVE_ROOT,
    cwd: Path | None = None,
    environ: Mapping[str, str] | None = None,
    uid: int | None = None,
    hostname: str | None = None,
) -> dict[str, Any]:
    """Authenticate terminal retained truth and atomically publish chunks."""
    env = dict(os.environ if environ is None else environ)
    task_id = _positive_int(env.get("SLURM_SCHED_TASK_ID"), "bridge task ID")
    bridge_job = str(env.get("SLURM_JOB_ID") or "").strip()
    if not bridge_job.isascii() or not bridge_job.isdigit() or int(bridge_job) <= 0:
        raise BridgeError("bridge Slurm job ID is invalid")
    if str(env.get("SLURM_CPUS_PER_TASK") or "").strip() != "1":
        raise BridgeError("bridge task must use exactly one CPU")
    bridge_node = (
        socket.gethostname().split(".", 1)[0] if hostname is None else hostname
    )
    if NODE_RE.fullmatch(bridge_node) is None:
        raise BridgeError("bridge node identity is invalid")
    if uid is None and os.name == "posix":
        uid = os.getuid()

    workspace = (Path.cwd() if cwd is None else cwd).resolve(strict=True)
    authority = _verify_authority(
        _read_json(authority_path, "terminal-success authority")
    )
    task_directory = source_task_directory.resolve(strict=True)
    _verify_task_files(task_directory, authority, expected_uid=uid)
    retained_root = source_retained_root.resolve(strict=True)
    package = _authenticate_retained_package(
        retained_root,
        _mapping(
            authority["source_success_marker"],
            "terminal success marker",
        ),
        expected_uid=uid,
    )

    relative_output = _safe_relative(output_relative_root, "transport output root")
    output_root = workspace.joinpath(*relative_output.parts)
    output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    output_root = output_root.resolve(strict=True)
    if workspace == output_root or workspace not in output_root.parents:
        raise BridgeError("transport output escaped bridge workspace")
    destination = output_root / TRANSPORT_DIRECTORY_NAME
    if destination.exists():
        return _validate_existing_transport(
            destination, authority=authority, expected_uid=uid
        )
    staging = output_root / (f".{TRANSPORT_DIRECTORY_NAME}.tmp-t{task_id}")
    if staging.exists():
        raise BridgeError("transport staging path already exists")
    staging.mkdir(mode=0o700)
    (staging / "chunks").mkdir(mode=0o700)

    aedt = Path(package["aedt_path"])
    source_before = aedt.stat()
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    size = 0
    published = False
    try:
        with aedt.open("rb", buffering=0) as stream:
            while raw := stream.read(RAW_CHUNK_BYTES):
                if len(rows) >= MAX_CHUNK_COUNT:
                    raise BridgeError("AEDT transport exceeded chunk bound")
                row = _write_chunk(
                    staging / "chunks" / f"{len(rows):08d}.b64",
                    raw,
                )
                rows.append(row)
                digest.update(raw)
                size += len(raw)
                if size > MAX_AEDT_BYTES:
                    raise BridgeError("AEDT transport exceeded byte bound")
        source_after = aedt.stat()
        source_row = package["rows"]["symmetric.aedt"]
        if (
            (
                source_before.st_dev,
                source_before.st_ino,
                source_before.st_size,
                source_before.st_mtime_ns,
            )
            != (
                source_after.st_dev,
                source_after.st_ino,
                source_after.st_size,
                source_after.st_mtime_ns,
            )
            or size != source_row["size_bytes"]
            or digest.hexdigest() != source_row["sha256"]
        ):
            raise BridgeError("retained AEDT changed during transport")

        source_files = {
            "source_authority.json": authority_path,
            "corrected_result.json": package["corrected_path"],
            "source_manifest.json": package["manifest_path"],
            "source_execution_receipt.json": package["execution_path"],
        }
        copied: dict[str, dict[str, Any]] = {}
        for name, source in source_files.items():
            destination_file = staging / name
            shutil.copyfile(source, destination_file)
            source_sha = sha256_file(source)
            if sha256_file(destination_file) != source_sha:
                raise BridgeError(f"transport control copy drifted: {name}")
            copied[name] = {
                "size_bytes": destination_file.stat().st_size,
                "sha256": source_sha,
            }
        if (
            copied["source_authority.json"]["sha256"] != sha256_file(authority_path)
            or copied["corrected_result.json"]["sha256"]
            != package["rows"]["corrected_result.json"]["sha256"]
        ):
            raise BridgeError("authority/result copy binding drifted")

        receipt = sealed(
            {
                "schema": TRANSPORT_SCHEMA,
                **CLASSIFICATION,
                "transport_semantics": (
                    "terminal-success retained GPFS package; symmetric.aedt "
                    "and corrected_result.json authenticated together; not a "
                    "live/open-only snapshot"
                ),
                "source_task": authority["source_task"],
                "source_plan": authority["source_plan"],
                "source_task_sh_sha256": SOURCE_TASK_SH_SHA256,
                "source_stdout_sha256": authority["source_task_files"]["stdout"][
                    "sha256"
                ],
                "source_stderr_sha256": authority["source_task_files"]["stderr"][
                    "sha256"
                ],
                "source_success_marker_payload_sha256": authority[
                    "source_success_marker"
                ]["payload_sha256"],
                "source_authority_payload_sha256": authority["payload_sha256"],
                "source_authority_file_sha256": copied["source_authority.json"][
                    "sha256"
                ],
                "source_retained_root": str(retained_root),
                "source_manifest_payload_sha256": package["manifest"]["payload_sha256"],
                "source_manifest_file_sha256": copied["source_manifest.json"]["sha256"],
                "source_execution_receipt_sha256": copied[
                    "source_execution_receipt.json"
                ]["sha256"],
                "corrected_result_sha256": copied["corrected_result.json"]["sha256"],
                "corrected_result_size_bytes": copied["corrected_result.json"][
                    "size_bytes"
                ],
                "fixed_physics": FIXED_PHYSICS,
                "constraint_observation": package["constraint_observation"],
                "temperatures": package["temperatures"],
                "artifact_filename": "symmetric.aedt",
                "artifact_size_bytes": size,
                "artifact_sha256": digest.hexdigest(),
                "raw_chunk_bytes": RAW_CHUNK_BYTES,
                "maximum_encoded_chunk_bytes": (MAX_ENCODED_CHUNK_BYTES),
                "chunk_count": len(rows),
                "chunks": rows,
                "bridge_task": {
                    "task_id": task_id,
                    "slurm_job_id": bridge_job,
                    "node": bridge_node,
                    "slurm_cpus_per_task": 1,
                },
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        receipt_path = _atomic_json(staging / "transport_receipt.json", receipt)
        if receipt_path.stat().st_size > MAX_JSON_BYTES:
            raise BridgeError("transport receipt is outside GET bound")
        for path in staging.rglob("*"):
            if path.is_file():
                os.chmod(path, 0o400)
        if os.name == "posix":
            for path in sorted(
                (item for item in staging.rglob("*") if item.is_dir()),
                key=lambda item: len(item.parts),
                reverse=True,
            ):
                os.chmod(path, 0o500)
            os.chmod(staging, 0o500)
        os.replace(staging, destination)
        published = True
    except BaseException:
        if not published and staging.exists():
            try:
                for directory in (
                    staging / "chunks",
                    staging,
                ):
                    if directory.exists():
                        os.chmod(directory, 0o700)
                for path in staging.rglob("*"):
                    if path.is_file():
                        os.chmod(path, 0o600)
            except OSError:
                pass
            shutil.rmtree(staging)
        raise

    receipt_path = destination / "transport_receipt.json"
    return {
        "schema": TRANSPORT_RESULT_SCHEMA,
        "status": "published",
        "transport_directory": str(destination),
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "artifact_size_bytes": size,
        "artifact_sha256": digest.hexdigest(),
        "corrected_result_sha256": package["rows"]["corrected_result.json"]["sha256"],
        "chunk_count": len(rows),
        "payload_sha256": receipt["payload_sha256"],
        **CLASSIFICATION,
    }


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    metadata = _regular_file(resolved, f"evidence file {resolved.name}")
    return {
        "path": str(resolved),
        "size_bytes": int(metadata.st_size),
        "sha256": sha256_file(resolved),
    }


def _transport_manifest_rows(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    value = _verify_seal(
        manifest,
        schema="mft-corrected-thermal-minimum-retained-package-v1",
        thermal_newline=True,
    )
    raw_rows = value.get("files")
    if not isinstance(raw_rows, list) or len(raw_rows) != 5:
        raise BridgeError("materialized retained manifest inventory drifted")
    rows: dict[str, dict[str, Any]] = {}
    for raw in raw_rows:
        row = _mapping(raw, "materialized retained manifest row")
        if set(row) != {"path", "size_bytes", "sha256"}:
            raise BridgeError("materialized retained manifest row drifted")
        relative = _safe_relative(str(row["path"]), "manifest row path")
        name = relative.as_posix()
        if name in rows:
            raise BridgeError("materialized retained manifest row duplicated")
        rows[name] = {
            "path": name,
            "size_bytes": _positive_int(
                row.get("size_bytes"),
                f"manifest size {name}",
                allow_zero=True,
            ),
            "sha256": _sha(row.get("sha256"), f"manifest SHA {name}"),
        }
    if not {
        "symmetric.aedt",
        "corrected_result.json",
        "execution_receipt.json",
    }.issubset(rows):
        raise BridgeError("materialized manifest lacks required result files")
    if any(
        rows[name]["size_bytes"] <= 0
        for name in (
            "symmetric.aedt",
            "corrected_result.json",
            "execution_receipt.json",
        )
    ):
        raise BridgeError("materialized required result file is empty")
    return rows


def _validate_downloaded_transport(
    transport_directory: Path,
) -> dict[str, Any]:
    _plain_directory(transport_directory, "downloaded transport directory")
    receipt_path = transport_directory / "transport_receipt.json"
    receipt = _verify_seal(
        _read_json(receipt_path, "downloaded transport receipt"),
        schema=TRANSPORT_SCHEMA,
    )
    authority_path = transport_directory / "source_authority.json"
    corrected_path = transport_directory / "corrected_result.json"
    manifest_path = transport_directory / "source_manifest.json"
    execution_path = transport_directory / "source_execution_receipt.json"
    authority = _verify_authority(
        _read_json(authority_path, "downloaded source authority")
    )
    manifest = _read_json(manifest_path, "downloaded source manifest")
    rows = _transport_manifest_rows(manifest)
    corrected = _read_json(corrected_path, "downloaded corrected thermal result")
    execution = _read_json(execution_path, "downloaded source execution receipt")
    if (
        receipt.get("diagnostic_only") is not True
        or receipt.get("canonical") is not False
        or receipt.get("production_truth_eligible") is not False
        or receipt.get("source_task") != authority.get("source_task")
        or receipt.get("source_plan") != _plan_exact_fields()
        or receipt.get("source_authority_payload_sha256")
        != authority.get("payload_sha256")
        or receipt.get("source_authority_file_sha256") != sha256_file(authority_path)
        or receipt.get("source_manifest_payload_sha256")
        != manifest.get("payload_sha256")
        or receipt.get("source_manifest_file_sha256") != sha256_file(manifest_path)
        or receipt.get("source_execution_receipt_sha256") != sha256_file(execution_path)
        or receipt.get("corrected_result_sha256") != sha256_file(corrected_path)
        or receipt.get("corrected_result_sha256")
        != rows["corrected_result.json"]["sha256"]
        or receipt.get("corrected_result_size_bytes")
        != rows["corrected_result.json"]["size_bytes"]
        or receipt.get("fixed_physics") != FIXED_PHYSICS
        or receipt.get("constraint_observation")
        != corrected.get("constraint_observation")
        or receipt.get("temperatures") != corrected.get("temperatures")
    ):
        raise BridgeError("downloaded transport control binding drifted")
    _validate_success_task(
        _mapping(receipt.get("source_task"), "transport source task")
    )
    if (
        corrected.get("schema") != "mft-corrected-thermal-diagnostic-result-v1"
        or corrected.get("diagnostic_only") is not True
        or corrected.get("canonical") is not False
        or execution.get("schema") != "mft-corrected-thermal-checkpoint-execution-v1"
        or execution.get("status") != "diagnostic_complete"
        or execution.get("fixed_physics") != FIXED_PHYSICS
        or corrected.get("source_provenance") != execution.get("source_provenance")
        or corrected.get("executor_provenance") != execution.get("executor_provenance")
        or corrected.get("constraint_observation")
        != execution.get("constraint_observation")
        or corrected.get("temperatures") != execution.get("temperatures")
    ):
        raise BridgeError("downloaded corrected result binding drifted")

    chunk_rows = receipt.get("chunks")
    chunk_count = _positive_int(receipt.get("chunk_count"), "transport chunk count")
    artifact_size = _positive_int(
        receipt.get("artifact_size_bytes"), "transport artifact size"
    )
    artifact_sha = _sha(receipt.get("artifact_sha256"), "transport artifact SHA")
    if (
        not isinstance(chunk_rows, list)
        or len(chunk_rows) != chunk_count
        or chunk_count != math.ceil(artifact_size / RAW_CHUNK_BYTES)
        or chunk_count > MAX_CHUNK_COUNT
        or receipt.get("raw_chunk_bytes") != RAW_CHUNK_BYTES
        or receipt.get("maximum_encoded_chunk_bytes") != MAX_ENCODED_CHUNK_BYTES
        or artifact_size != rows["symmetric.aedt"]["size_bytes"]
        or artifact_sha != rows["symmetric.aedt"]["sha256"]
    ):
        raise BridgeError("downloaded transport chunk contract drifted")
    digest = hashlib.sha256()
    size = 0
    validated_chunks: list[tuple[Path, bytes]] = []
    for index, raw_row in enumerate(chunk_rows):
        row = _mapping(raw_row, "downloaded transport chunk row")
        name = f"{index:08d}.b64"
        if row.get("filename") != name:
            raise BridgeError("downloaded transport chunk order drifted")
        path = transport_directory / "chunks" / name
        metadata = _regular_file(
            path,
            f"downloaded chunk {name}",
            maximum_bytes=MAX_ENCODED_CHUNK_BYTES,
        )
        encoded = path.read_bytes()
        if metadata.st_size != _positive_int(
            row.get("encoded_size_bytes"),
            f"encoded chunk size {name}",
        ) or sha256_bytes(encoded) != _sha(
            row.get("encoded_sha256"), f"encoded chunk SHA {name}"
        ):
            raise BridgeError(f"downloaded encoded chunk drifted: {name}")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise BridgeError(f"downloaded chunk is invalid base64: {name}") from exc
        if (
            len(raw)
            != _positive_int(row.get("raw_size_bytes"), f"raw chunk size {name}")
            or len(raw) > RAW_CHUNK_BYTES
            or sha256_bytes(raw) != _sha(row.get("raw_sha256"), f"raw chunk SHA {name}")
        ):
            raise BridgeError(f"downloaded raw chunk drifted: {name}")
        digest.update(raw)
        size += len(raw)
        validated_chunks.append((path, raw))
    if size != artifact_size or digest.hexdigest() != artifact_sha:
        raise BridgeError("downloaded transport reconstruction drifted")
    return {
        "receipt": receipt,
        "receipt_path": receipt_path,
        "authority": authority,
        "authority_path": authority_path,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_rows": rows,
        "corrected": corrected,
        "corrected_path": corrected_path,
        "execution": execution,
        "execution_path": execution_path,
        "chunks": validated_chunks,
        "artifact_size_bytes": size,
        "artifact_sha256": digest.hexdigest(),
    }


def materialize_collector_event(
    *,
    transport_directory: Path,
    output_root: Path,
    source_plan_path: Path,
    source_submission_path: Path,
    observed_at_utc: str | None = None,
) -> dict[str, Any]:
    """Make a terminal-collector-compatible tree for the final package gate."""
    validate_source_plan_files(source_plan_path, source_submission_path)
    transport = _validate_downloaded_transport(transport_directory.resolve(strict=True))
    root = output_root.absolute()
    root.mkdir(parents=True, exist_ok=True)
    collection = root / "collection"
    artifacts = collection / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    symmetric = artifacts / "symmetric.aedt"
    corrected = artifacts / "corrected_result.json"
    reconstructed = symmetric.with_name(f".{symmetric.name}.partial-{os.getpid()}")
    if reconstructed.exists():
        raise BridgeError("symmetric materialization staging file exists")
    digest = hashlib.sha256()
    size = 0
    try:
        with reconstructed.open("xb") as stream:
            for _path, raw in transport["chunks"]:
                stream.write(raw)
                digest.update(raw)
                size += len(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if (
            size != transport["artifact_size_bytes"]
            or digest.hexdigest() != transport["artifact_sha256"]
        ):
            raise BridgeError("materialized symmetric AEDT drifted")
        if symmetric.exists():
            if (
                symmetric.stat().st_size != size
                or sha256_file(symmetric) != digest.hexdigest()
            ):
                raise BridgeError("existing materialized symmetric AEDT drifted")
            reconstructed.unlink()
        else:
            os.replace(reconstructed, symmetric)
    finally:
        if reconstructed.exists():
            reconstructed.unlink()

    corrected_bytes = Path(transport["corrected_path"]).read_bytes()
    if corrected.exists():
        if corrected.read_bytes() != corrected_bytes:
            raise BridgeError("existing materialized corrected result drifted")
    else:
        temporary = corrected.with_name(f".{corrected.name}.tmp-{os.getpid()}")
        with temporary.open("xb") as stream:
            stream.write(corrected_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, corrected)
    manifest_copy = collection / "manifest.json"
    manifest_bytes = Path(transport["manifest_path"]).read_bytes()
    if manifest_copy.exists():
        if manifest_copy.read_bytes() != manifest_bytes:
            raise BridgeError("existing materialized manifest drifted")
    else:
        with manifest_copy.open("xb") as stream:
            stream.write(manifest_bytes)
            stream.flush()
            os.fsync(stream.fileno())

    manifest_rows = transport["manifest_rows"]
    collected_rows = [
        copy.deepcopy(manifest_rows["symmetric.aedt"]),
        copy.deepcopy(manifest_rows["corrected_result.json"]),
    ]
    if (
        symmetric.stat().st_size != manifest_rows["symmetric.aedt"]["size_bytes"]
        or sha256_file(symmetric) != manifest_rows["symmetric.aedt"]["sha256"]
        or corrected.stat().st_size
        != manifest_rows["corrected_result.json"]["size_bytes"]
        or sha256_file(corrected) != manifest_rows["corrected_result.json"]["sha256"]
    ):
        raise BridgeError("collector materialization file record drifted")

    event: dict[str, Any] = {
        "schema": "mft-terminal-collector-event-v1",
        "observed_at_utc": observed_at_utc or datetime.now(timezone.utc).isoformat(),
        "poll_number": 1,
        "target": "thermal96324",
        "task": transport["receipt"]["source_task"],
        "contract": {
            "task_sh_size_bytes": SOURCE_TASK_SH_SIZE,
            "task_sh_sha256": SOURCE_TASK_SH_SHA256,
            "command_sha256": SOURCE_COMMAND_SHA256,
            "local_evidence": {
                "plan": _file_record(source_plan_path),
                "submission": _file_record(source_submission_path),
                "terminal_transport_receipt": _file_record(
                    Path(transport["receipt_path"])
                ),
                "terminal_success_authority": _file_record(
                    Path(transport["authority_path"])
                ),
            },
            "fixed_physics_verified": True,
            "task_identity_verified": True,
            "strict_node_verified": True,
            "terminal_success_retained_transport_verified": True,
            "open_only_snapshot_used": False,
        },
        "scheduler_state": "success",
        "state": "success_collected_diagnostic",
        "artifact_collection_attempted": True,
        "failure_or_timeout": False,
        "pending_reasons": [],
        "retention_marker": transport["authority"]["source_success_marker"],
        "manifest": transport["manifest"],
        "collected_files": collected_rows,
        "terminal_transport": {
            "schema": transport["receipt"]["schema"],
            "receipt_payload_sha256": transport["receipt"]["payload_sha256"],
            "receipt_file_sha256": sha256_file(Path(transport["receipt_path"])),
            "source_authority_payload_sha256": transport["authority"]["payload_sha256"],
            "artifact_sha256": transport["artifact_sha256"],
            "artifact_size_bytes": transport["artifact_size_bytes"],
            "corrected_result_sha256": manifest_rows["corrected_result.json"]["sha256"],
            "chunk_count": len(transport["chunks"]),
            "get_file_maximum_bytes": MAX_JSON_BYTES,
        },
        **CLASSIFICATION,
        "scientific_pass_claimed": False,
        "production_claimed": False,
    }
    event["event_sha256"] = sha256_bytes(canonical_bytes(event))
    latest = root / "latest.json"
    _atomic_json(latest, event)
    return {
        "schema": (
            "mft-corrected-thermal-terminal-transport-materialization-result-v1"
        ),
        "status": "materialized",
        "thermal_event_path": str(latest),
        "thermal_event_sha256": sha256_file(latest),
        "symmetric_aedt_path": str(symmetric),
        "symmetric_aedt_sha256": sha256_file(symmetric),
        "symmetric_aedt_size_bytes": symmetric.stat().st_size,
        "corrected_result_path": str(corrected),
        "corrected_result_sha256": sha256_file(corrected),
        "final_package_gate_compatible": True,
        **CLASSIFICATION,
    }


def build_bridge_plan(
    *,
    authority: Mapping[str, Any],
    executor_revision: str,
    publisher_sha256: str,
    orchestration_root: Path = DEFAULT_ORCHESTRATION_ROOT,
    source_plan_path: Path,
    source_submission_path: Path,
    remote_ref: str = "refs/heads/integration/mft-goal-20260726",
) -> dict[str, Any]:
    """Build a deterministic one-task submission plan without submitting it."""
    source = _verify_authority(authority)
    revision = _git_revision(executor_revision, "bridge executor revision")
    tool_sha = _sha(publisher_sha256, "bridge publisher SHA")
    if remote_ref != "refs/heads/integration/mft-goal-20260726":
        raise BridgeError("bridge remote ref drifted")
    root = orchestration_root.absolute()
    source_plan_record = _file_record(source_plan_path)
    source_submission_record = _file_record(source_submission_path)
    if (
        source_plan_record["sha256"] != SOURCE_PLAN_FILE_SHA256
        or source_submission_record["sha256"] != SOURCE_SUBMISSION_FILE_SHA256
    ):
        raise BridgeError("bridge local source evidence drifted")
    authority_payload = canonical_bytes(source) + b"\n"
    authority_file_sha = sha256_bytes(authority_payload)
    encoded_authority = base64.b64encode(authority_payload).decode("ascii")
    name = f"mft-goal-thermal-terminal-transport-t96324-j839461-{tool_sha[:12]}"
    dedupe = f"mft-al:{name}:{revision}:{source['payload_sha256'][:16]}"
    command = "\n".join(
        [
            "set -euo pipefail",
            "test \"${SLURM_CPUS_PER_TASK:-}\" = '1'",
            "case \"${SLURM_SCHED_TASK_ID:-}\" in ''|*[!0-9]*) exit 91 ;; esac",
            'toolroot="/tmp/mft-thermal-terminal-transport-${SLURM_SCHED_TASK_ID}"',
            'case "$toolroot" in '
            "/tmp/mft-thermal-terminal-transport-[0-9]*) ;; "
            "*) exit 92 ;; esac",
            'test ! -e "$toolroot"',
            'mkdir -m 700 -- "$toolroot"',
            'cleanup(){ rm -rf -- "$toolroot"; }',
            "trap cleanup EXIT HUP INT TERM",
            'git -C "$toolroot" init -q',
            'git -C "$toolroot" remote add origin '
            "'https://github.com/Schwalbe262/MFT_1MW_2026.git'",
            f"git -C \"$toolroot\" fetch -q --depth=64 origin '{remote_ref}'",
            f"git -C \"$toolroot\" checkout -q --detach '{revision}'",
            f'test "$(git -C "$toolroot" rev-parse HEAD)" = \'{revision}\'',
            f"printf '%s  %s\\n' '{tool_sha}' "
            "'tools/mft_goal_corrected_thermal_transport_bridge.py' "
            '| (cd "$toolroot" && sha256sum -c -)',
            f"printf '%s' '{encoded_authority}' | base64 -d "
            '> "$toolroot/source-authority.json"',
            'test "$(sha256sum "$toolroot/source-authority.json" '
            f"| awk '{{print $1}}')\" = '{authority_file_sha}'",
            "python "
            '"$toolroot/tools/'
            'mft_goal_corrected_thermal_transport_bridge.py" publish '
            '--authority "$toolroot/source-authority.json" '
            f"--source-retained-root '{SOURCE_RETAINED_ROOT.as_posix()}' "
            "--source-task-directory "
            "'2026-07-26/task-96324-1785056187' "
            f"--output-relative-root '{OUTPUT_RELATIVE_ROOT}'",
        ]
    )
    profile = {
        "project": "MFT_1MW_2026v1",
        "account_name": SOURCE_ACCOUNT,
        "name": name,
        "dedupe_key": dedupe,
        "cpus": 1,
        "memory_mb": 4096,
        "gpus": 0,
        "timeout_seconds": 1800,
        "priority": 100,
        "env_profile": "pyaedt2026v1",
        "aedt_backend": "standalone",
        "required_capability": "conda:pyaedt2026v1",
        "node_name": SOURCE_NODE,
        "node_name_policy": "strict",
        "remote_cwd": "__SLURM_SCHEDULER_ACCOUNT_WORKSPACE__/runs",
        "max_workers_per_node": 1,
        "scheduling_profile": "fea_bursty",
    }
    scheduler_payload = {
        **profile,
        "command": command,
    }
    single_attempt = {
        "schema": (
            "mft-corrected-thermal-terminal-transport-single-attempt-contract-v1"
        ),
        "orchestration_root": str(root),
        "authority_path": str(root / "source_success_authority.json"),
        "plan_path": str(root / "transport_submission_plan.json"),
        "attempt_ledger_path": str(root / "bridge_post_attempt.json"),
        "submission_receipt_path": str(root / "bridge_submission_receipt.json"),
        "submission_seal_path": str(root / "bridge_submission_seal.json"),
        "collection_output_root": str(root / "collected"),
        "collection_seal_path": str(root / "collection_seal.json"),
        "post_call_budget": 1,
        "attempt_consumed_before_network": True,
        "output_override_allowed": False,
        "authorization_token_sha256": sha256_bytes(
            POST_AUTHORIZATION_TOKEN.encode("utf-8")
        ),
    }
    return sealed(
        {
            "schema": PLAN_SCHEMA,
            **CLASSIFICATION,
            "source_terminal_success_required": True,
            "source_authority_payload_sha256": source["payload_sha256"],
            "source_authority_file_sha256": authority_file_sha,
            "source_task": source["source_task"],
            "source_plan": source["source_plan"],
            "source_local_evidence": {
                "plan": source_plan_record,
                "submission": source_submission_record,
            },
            "executor_revision": revision,
            "publisher_file_sha256": tool_sha,
            "remote_ref": remote_ref,
            "canonical_command": command,
            "canonical_command_sha256": sha256_bytes(command.encode("utf-8")),
            "submission_profile": profile,
            "scheduler_payload": scheduler_payload,
            "scheduler_payload_sha256": canonical_sha256(scheduler_payload),
            "single_attempt_contract": single_attempt,
            "scheduler": {
                "url": "http://127.0.0.1:8002",
                "post_endpoint": "/api/tasks",
                "maximum_post_calls": 1,
                "post_calls_performed": 0,
                "submission_performed": False,
                "project_mutation_allowed": False,
            },
            "transport": {
                "output_relative_root": OUTPUT_RELATIVE_ROOT,
                "directory_name": TRANSPORT_DIRECTORY_NAME,
                "raw_chunk_bytes": RAW_CHUNK_BYTES,
                "maximum_encoded_chunk_bytes": (MAX_ENCODED_CHUNK_BYTES),
                "maximum_aedt_bytes": MAX_AEDT_BYTES,
                "maximum_chunk_count": MAX_CHUNK_COUNT,
                "maximum_json_bytes": MAX_JSON_BYTES,
            },
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )


class GetOnlyClient:
    def __init__(self, base_url: str, *, timeout_seconds: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.get_count = 0

    def get_bytes(self, endpoint: str) -> bytes:
        self.get_count += 1
        req = request.Request(
            self.base_url + endpoint,
            method="GET",
            headers={
                "Accept": "*/*",
                "User-Agent": "mft-thermal-terminal-transport-get-only/1",
            },
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                if int(response.status) != 200:
                    raise BridgeError(f"GET {endpoint} returned HTTP {response.status}")
                return response.read()
        except (error.HTTPError, error.URLError) as exc:
            raise BridgeError(f"GET {endpoint} failed: {exc}") from exc

    def get_json(self, endpoint: str) -> dict[str, Any]:
        return _parse_json_bytes(self.get_bytes(endpoint), endpoint)

    def get_value(self, endpoint: str) -> Any:
        raw = self.get_bytes(endpoint)
        if len(raw) > MAX_TASK_OUTPUT_BYTES:
            raise BridgeError(f"GET {endpoint} exceeded its JSON bound")
        try:
            return json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise BridgeError(f"GET {endpoint} returned invalid JSON") from exc

    def task_output(self, task_id: int, stream: str) -> bytes:
        if stream not in {"stdout", "stderr"}:
            raise BridgeError("unsupported task output stream")
        query = parse.urlencode({"max_bytes": MAX_TASK_OUTPUT_BYTES})
        return self.get_bytes(f"/api/tasks/{task_id}/{stream}?{query}")

    def remote_file(
        self,
        task_id: int,
        path: str,
        *,
        base: str = "remote_cwd",
        maximum_bytes: int = MAX_JSON_BYTES,
    ) -> bytes:
        if not 0 < maximum_bytes <= MAX_JSON_BYTES:
            raise BridgeError("remote GET byte bound is invalid")
        relative = _safe_relative(path, "remote file path").as_posix()
        query = parse.urlencode(
            {
                "base": base,
                "path": relative,
                "max_bytes": maximum_bytes,
            }
        )
        return self.get_bytes(f"/api/tasks/{task_id}/remote-file?{query}")


class SchedulerMutationClient(GetOnlyClient):
    """One-POST client; callers must hold the shared campaign lock."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 30.0):
        super().__init__(base_url, timeout_seconds=timeout_seconds)
        self.post_count = 0

    def post_json_once(
        self, endpoint: str, payload: Mapping[str, Any]
    ) -> tuple[int | None, dict[str, Any] | None, str | None]:
        scheduler = importlib.import_module("regression_260707.verify.scheduler_client")
        if not scheduler.campaign_mutation_lock_is_held():
            raise BridgeError("Scheduler POST attempted outside campaign mutation lock")
        if self.post_count != 0:
            raise BridgeError("bridge Scheduler POST budget was already consumed")
        self.post_count = 1
        body = canonical_bytes(payload)
        req = request.Request(
            self.base_url + endpoint,
            data=body,
            method="POST",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "mft-thermal-terminal-transport-single-post/1",
            },
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                raw = response.read()
                parsed = _parse_json_bytes(raw, "Scheduler POST response")
                return int(response.status), parsed, None
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            return int(exc.code), None, detail
        except (OSError, UnicodeError) as exc:
            return None, None, str(exc)


def scheduler_campaign_lock() -> Any:
    scheduler = importlib.import_module("regression_260707.verify.scheduler_client")
    return scheduler.campaign_mutation_lock()


def _verify_bridge_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    plan = _verify_seal(value, schema=PLAN_SCHEMA)
    profile = _mapping(plan.get("submission_profile"), "bridge submission profile")
    payload = _mapping(plan.get("scheduler_payload"), "bridge Scheduler payload")
    transport = _mapping(plan.get("transport"), "bridge transport contract")
    scheduler = _mapping(plan.get("scheduler"), "bridge scheduler contract")
    single = _mapping(
        plan.get("single_attempt_contract"), "bridge single-attempt contract"
    )
    local = _mapping(plan.get("source_local_evidence"), "bridge source local evidence")
    root = Path(str(single.get("orchestration_root") or "")).absolute()
    expected_paths = {
        "authority_path": root / "source_success_authority.json",
        "plan_path": root / "transport_submission_plan.json",
        "attempt_ledger_path": root / "bridge_post_attempt.json",
        "submission_receipt_path": root / "bridge_submission_receipt.json",
        "submission_seal_path": root / "bridge_submission_seal.json",
        "collection_output_root": root / "collected",
        "collection_seal_path": root / "collection_seal.json",
    }
    command = str(plan.get("canonical_command") or "")
    if (
        plan.get("diagnostic_only") is not True
        or plan.get("canonical") is not False
        or plan.get("production_truth_eligible") is not False
        or plan.get("source_terminal_success_required") is not True
        or plan.get("source_task", {}).get("task_id") != SOURCE_TASK_ID
        or plan.get("source_plan") != _plan_exact_fields()
        or local.get("plan", {}).get("sha256") != SOURCE_PLAN_FILE_SHA256
        or local.get("submission", {}).get("sha256") != SOURCE_SUBMISSION_FILE_SHA256
        or plan.get("canonical_command_sha256") != sha256_bytes(command.encode("utf-8"))
        or payload != {**profile, "command": command}
        or plan.get("scheduler_payload_sha256") != canonical_sha256(payload)
        or profile.get("project") != "MFT_1MW_2026v1"
        or profile.get("account_name") != SOURCE_ACCOUNT
        or profile.get("cpus") != 1
        or profile.get("memory_mb") != 4096
        or profile.get("timeout_seconds") != 1800
        or profile.get("node_name") != SOURCE_NODE
        or profile.get("node_name_policy") != "strict"
        or profile.get("remote_cwd") != "__SLURM_SCHEDULER_ACCOUNT_WORKSPACE__/runs"
        or transport.get("output_relative_root") != OUTPUT_RELATIVE_ROOT
        or transport.get("directory_name") != TRANSPORT_DIRECTORY_NAME
        or transport.get("raw_chunk_bytes") != RAW_CHUNK_BYTES
        or transport.get("maximum_encoded_chunk_bytes") != MAX_ENCODED_CHUNK_BYTES
        or transport.get("maximum_json_bytes") != MAX_JSON_BYTES
        or scheduler.get("post_calls_performed") != 0
        or scheduler.get("submission_performed") is not False
        or single.get("schema")
        != ("mft-corrected-thermal-terminal-transport-single-attempt-contract-v1")
        or any(
            Path(str(single.get(key) or "")).absolute() != expected
            for key, expected in expected_paths.items()
        )
        or single.get("post_call_budget") != 1
        or single.get("attempt_consumed_before_network") is not True
        or single.get("output_override_allowed") is not False
        or single.get("authorization_token_sha256")
        != sha256_bytes(POST_AUTHORIZATION_TOKEN.encode("utf-8"))
    ):
        raise BridgeError("bridge submission plan drifted")
    for label in ("plan", "submission"):
        record = _mapping(local.get(label), f"bridge source {label}")
        path = Path(str(record.get("path") or ""))
        if _file_record(path) != record:
            raise BridgeError(f"bridge source {label} record drifted")
    return plan


def _rows(value: Any, label: str) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [copy.deepcopy(item) for item in value if isinstance(item, Mapping)]
    if isinstance(value, Mapping):
        for key in ("tasks", "allocations", "items", "results"):
            raw = value.get(key)
            if isinstance(raw, list):
                return [
                    copy.deepcopy(item) for item in raw if isinstance(item, Mapping)
                ]
    raise BridgeError(f"{label} response shape drifted")


def _capacity_endpoint() -> str:
    query = parse.urlencode(
        [
            ("cpus", 1),
            ("memory_mb", 4096),
            ("scheduling_profile", "fea_bursty"),
            ("aedt_backend", "standalone"),
            ("required_capability", "conda:pyaedt2026v1"),
            ("env_profile", "pyaedt2026v1"),
            ("project", "MFT_1MW_2026v1"),
            ("max_workers_per_node", 1),
            ("account_name", SOURCE_ACCOUNT),
            ("node_name", SOURCE_NODE),
        ]
    )
    return f"/api/task-capacity?{query}"


def _collision_endpoint(plan: Mapping[str, Any]) -> str:
    profile = _mapping(plan.get("submission_profile"), "bridge submission profile")
    query = parse.urlencode(
        [
            ("limit", 10000),
            ("project", "MFT_1MW_2026v1"),
            ("name_prefix", profile["name"]),
        ]
    )
    return f"/api/tasks?{query}"


def live_submission_preflight(
    *,
    client: GetOnlyClient,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Recheck source, health, exact capacity, allocation, and collisions."""
    sealed_plan = _verify_bridge_plan(plan)
    profile = _mapping(sealed_plan["submission_profile"], "bridge submission profile")
    health = client.get_json("/api/health")
    source_task = client.get_json(f"/api/tasks/{SOURCE_TASK_ID}")
    capacity = client.get_json(_capacity_endpoint())
    allocations_value = client.get_value("/api/allocations")
    collisions_value = client.get_value(_collision_endpoint(sealed_plan))
    _validate_success_task(source_task)
    if (
        health.get("ok") is not True
        or health.get("scheduler_ok") is not True
        or health.get("scheduler_thread_alive") is not True
        or health.get("scheduler_stalled") is not False
    ):
        raise SubmissionNotReady("Scheduler health is not ready")
    if (
        capacity.get("queue_state") != "ready"
        or _positive_int(
            capacity.get("ready_fit_slots"),
            "ready capacity slots",
            allow_zero=True,
        )
        < 1
        or capacity.get("memory_pressure_state") != "ok"
        or _positive_int(
            capacity.get("standalone_aedt_available"),
            "standalone capacity",
            allow_zero=True,
        )
        < 1
    ):
        raise SubmissionNotReady("exact bridge task capacity is not ready")
    capacity_allocations = [
        row
        for row in _rows(capacity, "capacity allocations")
        if row.get("account_name") == SOURCE_ACCOUNT
        and row.get("node_name") == SOURCE_NODE
        and row.get("state") == "active"
        and int(row.get("fit_slots") or 0) >= 1
        and int(row.get("free_cpus") or 0) >= 1
        and int(row.get("free_memory_mb") or 0) >= 4096
        and row.get("memory_pressure_state") == "ok"
    ]
    if not capacity_allocations:
        raise SubmissionNotReady("exact n111/r1jae262 capacity is absent")
    selected = max(
        capacity_allocations,
        key=lambda row: (
            int(row.get("fit_slots") or 0),
            int(row.get("allocation_id") or 0),
        ),
    )
    allocation_id = _positive_int(
        selected.get("allocation_id"), "selected allocation ID"
    )
    exact_allocations = [
        row
        for row in _rows(allocations_value, "allocation inventory")
        if int(row.get("id") or row.get("allocation_id") or 0) == allocation_id
    ]
    if len(exact_allocations) != 1:
        raise SubmissionNotReady("selected bridge allocation is absent or ambiguous")
    allocation = exact_allocations[0]
    if (
        allocation.get("account_name") != SOURCE_ACCOUNT
        or allocation.get("node_name") != SOURCE_NODE
        or allocation.get("state") != "active"
        or not str(allocation.get("slurm_job_id") or "").isdigit()
        or int(allocation.get("free_cpus") or 0) < 1
        or int(allocation.get("free_memory_mb") or 0) < 4096
    ):
        raise SubmissionNotReady("selected bridge allocation drifted")
    collisions = [
        row
        for row in _rows(collisions_value, "collision inventory")
        if row.get("name") == profile["name"]
        or row.get("dedupe_key") == profile["dedupe_key"]
    ]
    if collisions:
        raise BridgeError("bridge task name/dedupe collision already exists")
    return sealed(
        {
            "schema": ("mft-corrected-thermal-terminal-transport-live-preflight-v1"),
            "observed_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_task": source_task,
            "scheduler_health": health,
            "capacity_endpoint": _capacity_endpoint(),
            "capacity": capacity,
            "selected_allocation": allocation,
            "selected_allocation_id": allocation_id,
            "selected_slurm_job_id": str(allocation["slurm_job_id"]),
            "collision_endpoint": _collision_endpoint(sealed_plan),
            "collision_count": 0,
            "exact_account_node_ready": True,
            "fixed_physics_unchanged": True,
            "scheduler_mutations": 0,
        }
    )


def _validate_bridge_task(
    task: Mapping[str, Any],
    *,
    bridge_task_id: int,
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    value = _mapping(task, "bridge task readback")
    profile = _mapping(plan.get("submission_profile"), "bridge submission profile")
    exact = {
        "task_id": bridge_task_id,
        "name": profile["name"],
        "dedupe_key": profile["dedupe_key"],
        "account_name": SOURCE_ACCOUNT,
        "cpus": 1,
        "memory_mb": 4096,
        "timeout_seconds": 1800,
        "node_name": SOURCE_NODE,
    }
    for key, expected in exact.items():
        if value.get(key) != expected:
            raise BridgeError(f"bridge task readback drifted: {key}")
    status = str(value.get("status") or "")
    state = str(value.get("state") or "")
    if status == "completed" and state == "succeeded":
        if (
            value.get("exit_code") != 0
            or value.get("actual_node_name") != SOURCE_NODE
            or not str(value.get("slurm_job_id") or "").isdigit()
        ):
            raise BridgeError("successful bridge task placement drifted")
        return value, "success"
    if status in {"completed", "failed", "cancelled"} or state in {
        "failed",
        "cancelled",
        "timeout",
        "timed_out",
    }:
        raise BridgeError(f"bridge task terminal without success: {status}/{state}")
    return value, "pending"


def collect_bridge_transport_get_only(
    *,
    client: GetOnlyClient,
    bridge_task_id: int,
    bridge_plan_path: Path,
    output_root: Path,
    source_plan_path: Path,
    source_submission_path: Path,
    observed_at_utc: str | None = None,
) -> dict[str, Any]:
    """GET bounded bridge files, then emit the final-gate thermal event."""
    task_id = _positive_int(bridge_task_id, "bridge task ID")
    bridge_plan = _verify_bridge_plan(
        _read_json(bridge_plan_path, "bridge submission plan")
    )
    task, state = _validate_bridge_task(
        client.get_json(f"/api/tasks/{task_id}"),
        bridge_task_id=task_id,
        plan=bridge_plan,
    )
    if state == "pending":
        return sealed(
            {
                "schema": (
                    "mft-corrected-thermal-terminal-transport-collection-pending-v1"
                ),
                **CLASSIFICATION,
                "bridge_task_id": task_id,
                "observed_status": task.get("status"),
                "observed_state": task.get("state"),
                "scheduler_get_calls": client.get_count,
                "scheduler_mutations": 0,
                "materialized": False,
                "created_at_utc": observed_at_utc
                or datetime.now(timezone.utc).isoformat(),
            }
        )

    task_sh = client.remote_file(task_id, "task.sh", base="remote_dir")
    command = str(bridge_plan["canonical_command"]).encode("utf-8")
    for sentinel in (
        f"export SLURM_SCHED_TASK_ID={task_id}".encode(),
        b"strict node placement mismatch: expected n111",
        (f"strict allocation mismatch: expected {task['slurm_job_id']}").encode(),
        command,
    ):
        if sentinel not in task_sh:
            raise BridgeError("bridge task.sh command/placement drifted")

    remote_root = f"{OUTPUT_RELATIVE_ROOT}/{TRANSPORT_DIRECTORY_NAME}"
    receipt_bytes = client.remote_file(task_id, f"{remote_root}/transport_receipt.json")
    receipt = _verify_seal(
        _parse_json_bytes(receipt_bytes, "remote transport receipt"),
        schema=TRANSPORT_SCHEMA,
    )
    bridge_runtime = _mapping(receipt.get("bridge_task"), "transport bridge runtime")
    if (
        bridge_runtime.get("task_id") != task_id
        or bridge_runtime.get("slurm_job_id") != str(task.get("slurm_job_id"))
        or bridge_runtime.get("node") != SOURCE_NODE
        or receipt.get("source_authority_payload_sha256")
        != bridge_plan.get("source_authority_payload_sha256")
    ):
        raise BridgeError("remote transport bridge task binding drifted")
    chunk_rows = receipt.get("chunks")
    if (
        not isinstance(chunk_rows, list)
        or len(chunk_rows) != receipt.get("chunk_count")
        or not 0 < len(chunk_rows) <= MAX_CHUNK_COUNT
    ):
        raise BridgeError("remote transport chunk inventory drifted")

    root = output_root.absolute()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "downloaded_transport"
    if destination.exists():
        validated = _validate_downloaded_transport(destination)
        if validated["receipt"] != receipt:
            raise BridgeError("existing downloaded transport receipt drifted")
    else:
        staging = root / f".downloaded_transport.tmp-{os.getpid()}"
        if staging.exists():
            raise BridgeError("download staging path already exists")
        staging.mkdir()
        (staging / "chunks").mkdir()
        try:
            (staging / "transport_receipt.json").write_bytes(receipt_bytes)
            for name in (
                "source_authority.json",
                "corrected_result.json",
                "source_manifest.json",
                "source_execution_receipt.json",
            ):
                data = client.remote_file(task_id, f"{remote_root}/{name}")
                (staging / name).write_bytes(data)
            for index, raw_row in enumerate(chunk_rows):
                row = _mapping(raw_row, "remote transport chunk row")
                name = f"{index:08d}.b64"
                if (
                    row.get("filename") != name
                    or _positive_int(
                        row.get("encoded_size_bytes"),
                        f"remote encoded size {name}",
                    )
                    > MAX_ENCODED_CHUNK_BYTES
                ):
                    raise BridgeError("remote transport chunk record drifted")
                data = client.remote_file(
                    task_id,
                    f"{remote_root}/chunks/{name}",
                    maximum_bytes=MAX_ENCODED_CHUNK_BYTES,
                )
                if len(data) != row["encoded_size_bytes"] or sha256_bytes(
                    data
                ) != row.get("encoded_sha256"):
                    raise BridgeError(f"remote transport chunk GET drifted: {name}")
                (staging / "chunks" / name).write_bytes(data)
            _validate_downloaded_transport(staging)
            os.replace(staging, destination)
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise
    materialized = materialize_collector_event(
        transport_directory=destination,
        output_root=root,
        source_plan_path=source_plan_path,
        source_submission_path=source_submission_path,
        observed_at_utc=observed_at_utc,
    )
    return {
        **materialized,
        "schema": ("mft-corrected-thermal-terminal-transport-get-collection-result-v1"),
        "status": "collected_and_materialized",
        "bridge_task_id": task_id,
        "bridge_slurm_job_id": str(task["slurm_job_id"]),
        "bridge_node": SOURCE_NODE,
        "bridge_task_sh_sha256": sha256_bytes(task_sh),
        "bridge_plan_sha256": sha256_file(bridge_plan_path),
        "scheduler_get_calls": client.get_count,
        "scheduler_mutations": 0,
    }


def prepare_from_scheduler_get(
    *,
    client: GetOnlyClient,
    plan_path: Path,
    submission_path: Path,
    executor_revision: str,
    publisher_sha256: str,
    orchestration_root: Path = DEFAULT_ORCHESTRATION_ROOT,
    observed_at_utc: str | None = None,
) -> dict[str, Any]:
    """GET task96324 and return either pending evidence or a ready plan."""
    task = client.get_json(f"/api/tasks/{SOURCE_TASK_ID}")
    source_task = _validate_source_task_identity(task)
    status = str(task.get("status") or "")
    state = str(task.get("state") or "")
    if status != "completed" or state != "succeeded":
        if status in {"completed", "failed", "cancelled"} or state in {
            "failed",
            "cancelled",
            "timeout",
            "timed_out",
        }:
            return sealed(
                {
                    "schema": SOURCE_FAILURE_SCHEMA,
                    **CLASSIFICATION,
                    "transport_authorized": False,
                    "source_task": source_task,
                    "reason": ("task96324 reached terminal state without success"),
                    "scheduler_get_calls": client.get_count,
                    "scheduler_mutations": 0,
                    "scheduler_post_calls": 0,
                    "terminal_failure": True,
                    "created_at_utc": observed_at_utc
                    or datetime.now(timezone.utc).isoformat(),
                }
            )
        return sealed(
            {
                "schema": PENDING_SCHEMA,
                **CLASSIFICATION,
                "transport_authorized": False,
                "source_task_id": SOURCE_TASK_ID,
                "observed_status": status,
                "observed_state": state,
                "reason": "task96324 has not reached exact terminal success",
                "scheduler_get_calls": client.get_count,
                "scheduler_mutations": 0,
                "scientific_pass_claimed": False,
                "created_at_utc": observed_at_utc
                or datetime.now(timezone.utc).isoformat(),
            }
        )
    stdout = client.task_output(SOURCE_TASK_ID, "stdout")
    stderr = client.task_output(SOURCE_TASK_ID, "stderr")
    authority = build_terminal_success_authority(
        task=task,
        stdout=stdout,
        stderr=stderr,
        plan_path=plan_path,
        submission_path=submission_path,
        observed_at_utc=observed_at_utc,
    )
    plan = build_bridge_plan(
        authority=authority,
        executor_revision=executor_revision,
        publisher_sha256=publisher_sha256,
        orchestration_root=orchestration_root,
        source_plan_path=plan_path,
        source_submission_path=submission_path,
    )
    return {
        "schema": "mft-corrected-thermal-terminal-transport-ready-v1",
        **CLASSIFICATION,
        "transport_authorized": True,
        "authority": authority,
        "plan": plan,
        "scheduler_get_calls": client.get_count,
        "scheduler_mutations": 0,
        "submission_performed": False,
    }


def _write_ready_outputs(output_root: Path, ready: Mapping[str, Any]) -> dict[str, str]:
    if ready.get("transport_authorized") is not True:
        raise PendingSource("task96324 is still pending")
    plan = _verify_bridge_plan(_mapping(ready["plan"], "plan"))
    expected_root = Path(
        plan["single_attempt_contract"]["orchestration_root"]
    ).absolute()
    if output_root.absolute() != expected_root:
        raise BridgeError("ready output differs from sealed orchestration root")
    output_root.mkdir(parents=True, exist_ok=False)
    authority_path = output_root / "source_success_authority.json"
    plan_path = output_root / "transport_submission_plan.json"
    _exclusive_json(authority_path, _mapping(ready["authority"], "authority"))
    _exclusive_json(plan_path, plan)
    return {
        "authority_path": str(authority_path),
        "authority_sha256": sha256_file(authority_path),
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
    }


def _orchestration_paths(
    plan: Mapping[str, Any],
    *,
    requested_root: Path,
    required_root: Path = DEFAULT_ORCHESTRATION_ROOT,
) -> dict[str, Path]:
    sealed_plan = _verify_bridge_plan(plan)
    single = _mapping(
        sealed_plan["single_attempt_contract"],
        "bridge single-attempt contract",
    )
    root = Path(single["orchestration_root"]).absolute()
    if requested_root.absolute() != root or required_root.absolute() != root:
        raise BridgeError("orchestration output differs from the campaign-fixed root")
    paths = {
        key: Path(str(single[key])).absolute()
        for key in (
            "authority_path",
            "plan_path",
            "attempt_ledger_path",
            "submission_receipt_path",
            "submission_seal_path",
            "collection_output_root",
            "collection_seal_path",
        )
    }
    if any(path != root and root not in path.parents for path in paths.values()):
        raise BridgeError("sealed orchestration path escaped its fixed root")
    paths["root"] = root
    return paths


def _load_ready_output(
    root: Path,
    *,
    required_root: Path = DEFAULT_ORCHESTRATION_ROOT,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    root = root.absolute()
    if root != required_root.absolute():
        raise BridgeError("orchestration output differs from the campaign-fixed root")
    plan_path = root / "transport_submission_plan.json"
    authority_path = root / "source_success_authority.json"
    plan = _verify_bridge_plan(_read_json(plan_path, "ready bridge submission plan"))
    paths = _orchestration_paths(plan, requested_root=root, required_root=required_root)
    if (
        plan_path.absolute() != paths["plan_path"]
        or authority_path.absolute() != paths["authority_path"]
    ):
        raise BridgeError("ready bridge evidence path drifted")
    authority = _verify_authority(_read_json(authority_path, "ready source authority"))
    if authority.get("payload_sha256") != plan.get(
        "source_authority_payload_sha256"
    ) or sha256_file(authority_path) != plan.get("source_authority_file_sha256"):
        raise BridgeError("ready authority/plan binding drifted")
    return plan, authority, paths


def _validate_submitted_task_readback(
    task: Mapping[str, Any],
    *,
    task_id: int,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    value = _mapping(task, "submitted bridge task readback")
    payload = _mapping(plan["scheduler_payload"], "bridge Scheduler payload")
    expected = {
        "name": payload["name"],
        "dedupe_key": payload["dedupe_key"],
        "project": payload["project"],
        "cpus": payload["cpus"],
        "memory_mb": payload["memory_mb"],
        "timeout_seconds": payload["timeout_seconds"],
        "max_workers_per_node": payload["max_workers_per_node"],
        "aedt_backend": payload["aedt_backend"],
        "env_profile": payload["env_profile"],
        "required_capability": payload["required_capability"],
    }
    observed_id = int(value.get("task_id") or value.get("id") or 0)
    drift = {
        key: {"expected": expected_value, "actual": value.get(key)}
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    if (
        observed_id != task_id
        or drift
        or value.get("status") not in {"queued", "attaching", "running", "completed"}
        or value.get("state") in {"failed", "cancelled", "timeout", "timed_out"}
        or value.get("requested_account_name", SOURCE_ACCOUNT) != SOURCE_ACCOUNT
        or value.get("requested_node_name", SOURCE_NODE) != SOURCE_NODE
        or value.get("requested_node_name_policy", "strict") != "strict"
        or value.get("account_name", SOURCE_ACCOUNT) != SOURCE_ACCOUNT
        or value.get("node_name", SOURCE_NODE) != SOURCE_NODE
    ):
        raise BridgeError(f"durable bridge task readback drifted: {drift}")
    return value


def _exact_bridge_tasks(
    client: GetOnlyClient, plan: Mapping[str, Any]
) -> list[dict[str, Any]]:
    profile = _mapping(plan["submission_profile"], "bridge submission profile")
    return [
        row
        for row in _rows(
            client.get_value(_collision_endpoint(plan)),
            "bridge task reconciliation",
        )
        if row.get("name") == profile["name"]
        and row.get("dedupe_key") == profile["dedupe_key"]
    ]


def _write_submission_evidence(
    *,
    plan: Mapping[str, Any],
    paths: Mapping[str, Path],
    authority: Mapping[str, Any],
    attempt: Mapping[str, Any],
    initial_preflight: Mapping[str, Any] | None,
    locked_preflight: Mapping[str, Any],
    task_id: int,
    task_readback: Mapping[str, Any],
    post_status: int | None,
    post_response: Mapping[str, Any] | None,
    post_error: str | None,
    post_calls_this_invocation: int,
    reconciled: bool,
) -> dict[str, Any]:
    attempt_path = paths["attempt_ledger_path"]
    receipt_path = paths["submission_receipt_path"]
    seal_path = paths["submission_seal_path"]
    receipt = sealed(
        {
            "schema": SUBMISSION_RECEIPT_SCHEMA,
            **CLASSIFICATION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan": _file_record(paths["plan_path"]),
            "plan_payload_sha256": plan["payload_sha256"],
            "authority": _file_record(paths["authority_path"]),
            "authority_payload_sha256": authority["payload_sha256"],
            "attempt_ledger": _file_record(attempt_path),
            "attempt_ledger_payload_sha256": attempt["payload_sha256"],
            "initial_live_preflight": initial_preflight,
            "locked_pre_submit_live_preflight": locked_preflight,
            "campaign_mutation_lock_acquired": True,
            "capacity_and_collision_rechecked_inside_lock": True,
            "scheduler_payload_sha256": plan["scheduler_payload_sha256"],
            "scheduler_post_calls_total": 1,
            "scheduler_post_calls_this_invocation": (post_calls_this_invocation),
            "scheduler_post_http_status": post_status,
            "scheduler_post_response": post_response,
            "scheduler_post_error": post_error,
            "reconciled_after_consumed_attempt": reconciled,
            "task_id": task_id,
            "task_readback": copy.deepcopy(dict(task_readback)),
            "task_readback_sha256": canonical_sha256(task_readback),
            "scheduler_repository_modified": False,
            "scheduler_project_mutation_performed": False,
        }
    )
    _exclusive_json(receipt_path, receipt)
    final_seal = sealed(
        {
            "schema": SUBMISSION_SEAL_SCHEMA,
            **CLASSIFICATION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan": _file_record(paths["plan_path"]),
            "plan_payload_sha256": plan["payload_sha256"],
            "authority": _file_record(paths["authority_path"]),
            "authority_payload_sha256": authority["payload_sha256"],
            "attempt_ledger": _file_record(attempt_path),
            "attempt_ledger_payload_sha256": attempt["payload_sha256"],
            "receipt": _file_record(receipt_path),
            "receipt_payload_sha256": receipt["payload_sha256"],
            "task_id": task_id,
            "scheduler_post_calls_total": 1,
            "immutable_evidence_complete": True,
        }
    )
    _exclusive_json(seal_path, final_seal)
    return {
        "schema": ("mft-corrected-thermal-terminal-transport-submission-result-v1"),
        "status": "submitted" if not reconciled else "reconciled",
        "task_id": task_id,
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "submission_seal_path": str(seal_path),
        "submission_seal_sha256": sha256_file(seal_path),
        "scheduler_post_calls_this_invocation": post_calls_this_invocation,
        "scheduler_post_calls_total": 1,
        **CLASSIFICATION,
    }


def _existing_submission_result(
    *,
    plan: Mapping[str, Any],
    authority: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    attempt = _verify_seal(
        _read_json(paths["attempt_ledger_path"], "bridge attempt ledger"),
        schema=ATTEMPT_SCHEMA,
    )
    receipt = _verify_seal(
        _read_json(
            paths["submission_receipt_path"],
            "bridge submission receipt",
        ),
        schema=SUBMISSION_RECEIPT_SCHEMA,
    )
    final_seal = _verify_seal(
        _read_json(paths["submission_seal_path"], "bridge submission seal"),
        schema=SUBMISSION_SEAL_SCHEMA,
    )
    if (
        attempt.get("plan_payload_sha256") != plan["payload_sha256"]
        or receipt.get("plan_payload_sha256") != plan["payload_sha256"]
        or receipt.get("authority_payload_sha256") != authority["payload_sha256"]
        or receipt.get("attempt_ledger_payload_sha256") != attempt["payload_sha256"]
        or final_seal.get("receipt_payload_sha256") != receipt["payload_sha256"]
        or final_seal.get("attempt_ledger_payload_sha256") != attempt["payload_sha256"]
        or final_seal.get("task_id") != receipt.get("task_id")
        or receipt.get("scheduler_post_calls_total") != 1
    ):
        raise BridgeError("existing bridge submission evidence drifted")
    return {
        "schema": ("mft-corrected-thermal-terminal-transport-submission-result-v1"),
        "status": "already_submitted",
        "task_id": int(receipt["task_id"]),
        "receipt_path": str(paths["submission_receipt_path"]),
        "receipt_sha256": sha256_file(paths["submission_receipt_path"]),
        "submission_seal_path": str(paths["submission_seal_path"]),
        "submission_seal_sha256": sha256_file(paths["submission_seal_path"]),
        "scheduler_post_calls_this_invocation": 0,
        "scheduler_post_calls_total": 1,
        **CLASSIFICATION,
    }


def submit_ready_plan_once(
    *,
    client: GetOnlyClient,
    poster: Callable[
        [str, Mapping[str, Any]],
        tuple[int | None, dict[str, Any] | None, str | None],
    ],
    output_root: Path,
    authorize_post: str,
    lock_factory: Callable[[], Any] = scheduler_campaign_lock,
    required_root: Path = DEFAULT_ORCHESTRATION_ROOT,
) -> dict[str, Any]:
    """Consume the campaign-fixed ledger, then perform at most one POST."""
    if authorize_post != POST_AUTHORIZATION_TOKEN:
        raise BridgeError("explicit bridge Scheduler POST authorization is absent")
    plan, authority, paths = _load_ready_output(
        output_root, required_root=required_root
    )
    attempt_path = paths["attempt_ledger_path"]
    receipt_path = paths["submission_receipt_path"]
    seal_path = paths["submission_seal_path"]
    if receipt_path.exists() or seal_path.exists():
        if not (attempt_path.exists() and receipt_path.exists() and seal_path.exists()):
            raise BridgeError("bridge submission evidence is incomplete")
        return _existing_submission_result(plan=plan, authority=authority, paths=paths)
    if attempt_path.exists():
        attempt = _verify_seal(
            _read_json(attempt_path, "consumed bridge attempt"),
            schema=ATTEMPT_SCHEMA,
        )
        if (
            attempt.get("plan_payload_sha256") != plan["payload_sha256"]
            or attempt.get("scheduler_payload_sha256")
            != plan["scheduler_payload_sha256"]
        ):
            raise BridgeError("consumed bridge attempt ledger drifted")
        exact = _exact_bridge_tasks(client, plan)
        if len(exact) != 1:
            raise BridgeError("consumed bridge attempt cannot be uniquely reconciled")
        task_id = int(exact[0].get("task_id") or exact[0].get("id") or 0)
        readback = _validate_submitted_task_readback(
            client.get_json(f"/api/tasks/{task_id}"),
            task_id=task_id,
            plan=plan,
        )
        locked = _mapping(
            attempt.get("locked_pre_submit_live_preflight"),
            "attempt locked preflight",
        )
        return _write_submission_evidence(
            plan=plan,
            paths=paths,
            authority=authority,
            attempt=attempt,
            initial_preflight=None,
            locked_preflight=locked,
            task_id=task_id,
            task_readback=readback,
            post_status=None,
            post_response=None,
            post_error="reconciled after previously consumed attempt",
            post_calls_this_invocation=0,
            reconciled=True,
        )

    initial = live_submission_preflight(client=client, plan=plan)
    with lock_factory():
        locked = live_submission_preflight(client=client, plan=plan)
        attempt = sealed(
            {
                "schema": ATTEMPT_SCHEMA,
                **CLASSIFICATION,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "plan": _file_record(paths["plan_path"]),
                "plan_payload_sha256": plan["payload_sha256"],
                "authority": _file_record(paths["authority_path"]),
                "authority_payload_sha256": authority["payload_sha256"],
                "scheduler_payload_sha256": plan["scheduler_payload_sha256"],
                "locked_pre_submit_live_preflight": locked,
                "campaign_mutation_lock_acquired": True,
                "capacity_and_collision_rechecked_inside_lock": True,
                "post_call_budget": 1,
                "post_call_consumed_before_network": True,
                "automatic_retry_allowed": False,
                "authorization_token_sha256": sha256_bytes(
                    authorize_post.encode("utf-8")
                ),
            }
        )
        _exclusive_json(attempt_path, attempt)
        status, response, post_error = poster("/api/tasks", plan["scheduler_payload"])

    task_id = 0
    if isinstance(response, Mapping):
        task_id = int(response.get("task_id") or response.get("id") or 0)
    reconciled = False
    if task_id <= 0:
        exact = _exact_bridge_tasks(client, plan)
        if len(exact) != 1:
            failure = sealed(
                {
                    "schema": (
                        "mft-corrected-thermal-terminal-transport-post-failure-v1"
                    ),
                    **CLASSIFICATION,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "plan": _file_record(paths["plan_path"]),
                    "attempt_ledger": _file_record(attempt_path),
                    "attempt_ledger_payload_sha256": attempt["payload_sha256"],
                    "post_http_status": status,
                    "post_error": post_error,
                    "exact_reconciliation_count": len(exact),
                    "scheduler_post_calls_total": 1,
                    "automatic_retry_allowed": False,
                }
            )
            _exclusive_json(paths["root"] / "post_failure.json", failure)
            raise BridgeError(
                "bridge Scheduler POST failed without exact reconciliation"
            )
        task_id = int(exact[0].get("task_id") or exact[0].get("id") or 0)
        reconciled = True
    readback = _validate_submitted_task_readback(
        client.get_json(f"/api/tasks/{task_id}"),
        task_id=task_id,
        plan=plan,
    )
    return _write_submission_evidence(
        plan=plan,
        paths=paths,
        authority=authority,
        attempt=attempt,
        initial_preflight=initial,
        locked_preflight=locked,
        task_id=task_id,
        task_readback=readback,
        post_status=status,
        post_response=response,
        post_error=post_error,
        post_calls_this_invocation=1,
        reconciled=reconciled,
    )


def _collection_result_existing(
    *,
    paths: Mapping[str, Path],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    seal = _verify_seal(
        _read_json(paths["collection_seal_path"], "collection seal"),
        schema=COLLECTION_SEAL_SCHEMA,
    )
    event_record = _mapping(seal.get("thermal_event"), "sealed thermal event record")
    event_path = Path(str(event_record.get("path") or ""))
    if (
        seal.get("bridge_task_id") != receipt.get("task_id")
        or _file_record(event_path) != event_record
        or seal.get("final_package_gate_compatible") is not True
    ):
        raise BridgeError("existing bridge collection seal drifted")
    return {
        "schema": (
            "mft-corrected-thermal-terminal-transport-watch-collection-result-v1"
        ),
        "status": "already_materialized",
        "task_id": int(receipt["task_id"]),
        "thermal_event_path": str(event_path),
        "collection_seal_path": str(paths["collection_seal_path"]),
        "collection_seal_sha256": sha256_file(paths["collection_seal_path"]),
        "scheduler_get_calls_this_invocation": 0,
        "scheduler_mutations": 0,
        "final_package_gate_compatible": True,
        **CLASSIFICATION,
    }


def watch_collect_materialize_once(
    *,
    client: GetOnlyClient,
    output_root: Path,
    required_root: Path = DEFAULT_ORCHESTRATION_ROOT,
) -> dict[str, Any]:
    plan, authority, paths = _load_ready_output(
        output_root, required_root=required_root
    )
    submission = _existing_submission_result(
        plan=plan, authority=authority, paths=paths
    )
    receipt = _verify_seal(
        _read_json(
            paths["submission_receipt_path"],
            "bridge submission receipt",
        ),
        schema=SUBMISSION_RECEIPT_SCHEMA,
    )
    if paths["collection_seal_path"].exists():
        return _collection_result_existing(paths=paths, receipt=receipt)
    local = _mapping(plan["source_local_evidence"], "bridge source local evidence")
    collected = collect_bridge_transport_get_only(
        client=client,
        bridge_task_id=int(submission["task_id"]),
        bridge_plan_path=paths["plan_path"],
        output_root=paths["collection_output_root"],
        source_plan_path=Path(local["plan"]["path"]),
        source_submission_path=Path(local["submission"]["path"]),
    )
    if collected.get("materialized") is False:
        return {
            **collected,
            "status": "bridge_task_pending",
            "scheduler_mutations": 0,
        }
    event_path = Path(str(collected["thermal_event_path"]))
    symmetric_path = Path(str(collected["symmetric_aedt_path"]))
    corrected_path = Path(str(collected["corrected_result_path"]))
    collection_seal = sealed(
        {
            "schema": COLLECTION_SEAL_SCHEMA,
            **CLASSIFICATION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan": _file_record(paths["plan_path"]),
            "plan_payload_sha256": plan["payload_sha256"],
            "submission_seal": _file_record(paths["submission_seal_path"]),
            "submission_receipt_payload_sha256": receipt["payload_sha256"],
            "bridge_task_id": int(receipt["task_id"]),
            "thermal_event": _file_record(event_path),
            "symmetric_aedt": _file_record(symmetric_path),
            "corrected_result": _file_record(corrected_path),
            "final_package_gate_compatible": True,
            "latest_event_path": str(event_path.resolve(strict=True)),
            "scheduler_mutations": 0,
        }
    )
    _exclusive_json(paths["collection_seal_path"], collection_seal)
    return {
        "schema": (
            "mft-corrected-thermal-terminal-transport-watch-collection-result-v1"
        ),
        "status": "collected_and_materialized",
        "task_id": int(receipt["task_id"]),
        "thermal_event_path": str(event_path),
        "collection_seal_path": str(paths["collection_seal_path"]),
        "collection_seal_sha256": sha256_file(paths["collection_seal_path"]),
        "scheduler_get_calls_this_invocation": client.get_count,
        "scheduler_mutations": 0,
        "final_package_gate_compatible": True,
        **CLASSIFICATION,
    }


def watch_collect_materialize(
    *,
    client: GetOnlyClient,
    output_root: Path,
    watch: bool,
    interval_seconds: int = WATCH_INTERVAL_SECONDS,
    sleep_fn: Callable[[float], None] = time.sleep,
    required_root: Path = DEFAULT_ORCHESTRATION_ROOT,
) -> dict[str, Any]:
    if interval_seconds != WATCH_INTERVAL_SECONDS:
        raise BridgeError("bridge watcher interval must be exactly 60 seconds")
    while True:
        result = watch_collect_materialize_once(
            client=client,
            output_root=output_root,
            required_root=required_root,
        )
        if result.get("status") != "bridge_task_pending" or not watch:
            return result
        sleep_fn(float(interval_seconds))


def _watch_pending(
    *,
    reason: str,
    client: GetOnlyClient,
) -> dict[str, Any]:
    return sealed(
        {
            "schema": (
                "mft-corrected-thermal-terminal-transport-watch-submit-pending-v1"
            ),
            **CLASSIFICATION,
            "status": "pending",
            "reason": reason,
            "scheduler_get_calls": client.get_count,
            "scheduler_post_calls": 0,
            "scheduler_mutations": 0,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )


def watch_submit_orchestrator(
    *,
    client: GetOnlyClient,
    poster: Callable[
        [str, Mapping[str, Any]],
        tuple[int | None, dict[str, Any] | None, str | None],
    ],
    output_root: Path,
    source_plan_path: Path,
    source_submission_path: Path,
    executor_revision: str,
    publisher_sha256: str,
    authorize_post: str,
    watch: bool,
    continue_collection: bool = True,
    interval_seconds: int = WATCH_INTERVAL_SECONDS,
    sleep_fn: Callable[[float], None] = time.sleep,
    lock_factory: Callable[[], Any] = scheduler_campaign_lock,
    required_root: Path = DEFAULT_ORCHESTRATION_ROOT,
) -> dict[str, Any]:
    """Poll source success, submit once, then GET-collect without intervention."""
    root = output_root.absolute()
    if root != required_root.absolute():
        raise BridgeError("watch-submit output differs from the campaign-fixed root")
    if authorize_post != POST_AUTHORIZATION_TOKEN:
        raise BridgeError("explicit bridge Scheduler POST authorization is absent")
    if interval_seconds != WATCH_INTERVAL_SECONDS:
        raise BridgeError("bridge watcher interval must be exactly 60 seconds")
    while True:
        plan_path = root / "transport_submission_plan.json"
        if not plan_path.exists():
            if root.exists():
                failure_path = root / "source_terminal_failure.json"
                if failure_path.exists():
                    failure = _verify_seal(
                        _read_json(failure_path, "source failure"),
                        schema=SOURCE_FAILURE_SCHEMA,
                    )
                    return {
                        **failure,
                        "status": "source_terminal_failure",
                        "scheduler_post_calls": 0,
                    }
                raise BridgeError(
                    "fixed orchestration root exists without a sealed plan"
                )
            prepared = prepare_from_scheduler_get(
                client=client,
                plan_path=source_plan_path,
                submission_path=source_submission_path,
                executor_revision=executor_revision,
                publisher_sha256=publisher_sha256,
                orchestration_root=root,
            )
            if prepared.get("schema") == SOURCE_FAILURE_SCHEMA:
                root.mkdir(parents=True, exist_ok=False)
                failure_path = root / "source_terminal_failure.json"
                _exclusive_json(failure_path, prepared)
                return {
                    **prepared,
                    "status": "source_terminal_failure",
                    "source_failure_path": str(failure_path),
                    "scheduler_post_calls": 0,
                }
            if prepared.get("transport_authorized") is not True:
                if not watch:
                    return _watch_pending(
                        reason=prepared.get("reason", "source task pending"),
                        client=client,
                    )
                sleep_fn(float(interval_seconds))
                continue
            _write_ready_outputs(root, prepared)
        try:
            submission = submit_ready_plan_once(
                client=client,
                poster=poster,
                output_root=root,
                authorize_post=authorize_post,
                lock_factory=lock_factory,
                required_root=required_root,
            )
        except SubmissionNotReady as exc:
            if not watch:
                return _watch_pending(reason=str(exc), client=client)
            sleep_fn(float(interval_seconds))
            continue
        if not continue_collection:
            return submission
        collection = watch_collect_materialize(
            client=client,
            output_root=root,
            watch=watch,
            interval_seconds=interval_seconds,
            sleep_fn=sleep_fn,
            required_root=required_root,
        )
        return {
            "schema": (
                "mft-corrected-thermal-terminal-transport-"
                "autonomous-orchestration-result-v1"
            ),
            "status": collection["status"],
            "submission": submission,
            "collection": collection,
            "scheduler_post_calls_this_invocation": submission[
                "scheduler_post_calls_this_invocation"
            ],
            "scheduler_mutations": submission["scheduler_post_calls_this_invocation"],
            **CLASSIFICATION,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    publish_parser = subparsers.add_parser("publish")
    publish_parser.add_argument("--authority", required=True, type=Path)
    publish_parser.add_argument("--source-retained-root", required=True, type=Path)
    publish_parser.add_argument("--source-task-directory", required=True, type=Path)
    publish_parser.add_argument("--output-relative-root", default=OUTPUT_RELATIVE_ROOT)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--api-url", default="http://127.0.0.1:8002")
    prepare_parser.add_argument("--plan", required=True, type=Path)
    prepare_parser.add_argument("--submission", required=True, type=Path)
    prepare_parser.add_argument("--executor-revision", required=True)
    prepare_parser.add_argument("--output-root", required=True, type=Path)

    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--transport-directory", required=True, type=Path)
    materialize_parser.add_argument("--source-plan", required=True, type=Path)
    materialize_parser.add_argument("--source-submission", required=True, type=Path)
    materialize_parser.add_argument("--output-root", required=True, type=Path)

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--api-url", default="http://127.0.0.1:8002")
    collect_parser.add_argument("--bridge-task-id", required=True, type=int)
    collect_parser.add_argument("--bridge-plan", required=True, type=Path)
    collect_parser.add_argument("--source-plan", required=True, type=Path)
    collect_parser.add_argument("--source-submission", required=True, type=Path)
    collect_parser.add_argument("--output-root", required=True, type=Path)

    watch_submit_parser = subparsers.add_parser("watch-submit")
    watch_submit_parser.add_argument("--api-url", default="http://127.0.0.1:8002")
    watch_submit_parser.add_argument("--source-plan", required=True, type=Path)
    watch_submit_parser.add_argument("--source-submission", required=True, type=Path)
    watch_submit_parser.add_argument("--executor-revision", required=True)
    watch_submit_parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_ORCHESTRATION_ROOT,
    )
    watch_submit_parser.add_argument("--authorize-post", required=True)
    watch_submit_parser.add_argument(
        "--interval-seconds",
        type=int,
        default=WATCH_INTERVAL_SECONDS,
    )
    watch_submit_parser.add_argument("--once", action="store_true")
    watch_submit_parser.add_argument("--submit-only", action="store_true")

    watch_collect_parser = subparsers.add_parser("watch-collect-materialize")
    watch_collect_parser.add_argument("--api-url", default="http://127.0.0.1:8002")
    watch_collect_parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_ORCHESTRATION_ROOT,
    )
    watch_collect_parser.add_argument(
        "--interval-seconds",
        type=int,
        default=WATCH_INTERVAL_SECONDS,
    )
    watch_collect_parser.add_argument("--once", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "publish":
            result = publish_terminal_transport(
                authority_path=args.authority,
                source_retained_root=args.source_retained_root,
                source_task_directory=args.source_task_directory,
                output_relative_root=args.output_relative_root,
            )
        elif args.command == "prepare":
            publisher_sha = sha256_file(Path(__file__).resolve())
            client = GetOnlyClient(args.api_url)
            ready = prepare_from_scheduler_get(
                client=client,
                plan_path=args.plan,
                submission_path=args.submission,
                executor_revision=args.executor_revision,
                publisher_sha256=publisher_sha,
                orchestration_root=args.output_root,
            )
            if ready.get("transport_authorized") is not True:
                print(json.dumps(ready, sort_keys=True, separators=(",", ":")))
                return 3
            outputs = _write_ready_outputs(args.output_root, ready)
            result = {
                **{
                    key: value
                    for key, value in ready.items()
                    if key not in {"authority", "plan"}
                },
                **outputs,
                "authority_payload_sha256": ready["authority"]["payload_sha256"],
                "plan_payload_sha256": ready["plan"]["payload_sha256"],
                "canonical_command_sha256": ready["plan"]["canonical_command_sha256"],
            }
        elif args.command == "materialize":
            result = materialize_collector_event(
                transport_directory=args.transport_directory,
                output_root=args.output_root,
                source_plan_path=args.source_plan,
                source_submission_path=args.source_submission,
            )
        elif args.command == "collect":
            client = GetOnlyClient(args.api_url)
            result = collect_bridge_transport_get_only(
                client=client,
                bridge_task_id=args.bridge_task_id,
                bridge_plan_path=args.bridge_plan,
                output_root=args.output_root,
                source_plan_path=args.source_plan,
                source_submission_path=args.source_submission,
            )
            if result.get("materialized") is False:
                print(json.dumps(result, sort_keys=True, separators=(",", ":")))
                return 3
        elif args.command == "watch-submit":
            client = SchedulerMutationClient(args.api_url)
            result = watch_submit_orchestrator(
                client=client,
                poster=client.post_json_once,
                output_root=args.output_root,
                source_plan_path=args.source_plan,
                source_submission_path=args.source_submission,
                executor_revision=args.executor_revision,
                publisher_sha256=sha256_file(Path(__file__).resolve()),
                authorize_post=args.authorize_post,
                watch=not args.once,
                continue_collection=not args.submit_only,
                interval_seconds=args.interval_seconds,
            )
            if result.get("status") in {
                "pending",
                "bridge_task_pending",
            }:
                print(json.dumps(result, sort_keys=True, separators=(",", ":")))
                return 3
        else:
            client = GetOnlyClient(args.api_url)
            result = watch_collect_materialize(
                client=client,
                output_root=args.output_root,
                watch=not args.once,
                interval_seconds=args.interval_seconds,
            )
            if result.get("status") == "bridge_task_pending":
                print(json.dumps(result, sort_keys=True, separators=(",", ":")))
                return 3
    except (OSError, BridgeError) as exc:
        print(
            f"CORRECTED_THERMAL_TRANSPORT_ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
