#!/usr/bin/env python3
"""Prepare or submit one sealed post-deadline diagnostic Full retry.

The source of truth is failed Scheduler task 96307.  This tool fetches and
hash-verifies its task snapshot, task.sh, stdout, and stderr before extracting
the original user command from task.sh.  The command is transformed only by a
small byte-level substitution allow-list:

* use a new work-directory slug;
* use a new Scheduler dedupe key in both retention contracts;
* use a new retained-artifact path;
* update the retention marker hash implied by the new dedupe key; and
* prefix the unchanged Full solver invocation with an inner wall-time guard.

The inner guard leaves a planned success-path preservation window inside the
24-hour Scheduler timeout.  The physics JSON and Full solver argv are retained
byte-for-byte.  The result is permanently diagnostic, non-production, and
non-canonical; it cannot repair the missed 2026-07-26 18:00 KST deadline.

``plan`` performs GET requests only.  ``submit`` also defaults to GET-only and
requires ``--apply``.  An immutable local attempt ledger is created before the
single allowed POST, so an ambiguous network outcome cannot be retried by
rerunning the same plan.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
from typing import Any
from urllib import error, parse, request


DEFAULT_SCHEDULER_URL = "http://127.0.0.1:8002"
SOURCE_TASK_ID = 96307
SOURCE_TASK_NAME = "mft-goal-provisional-full-l96230-b7c30cb70b95-v1"
SOURCE_DEDUPE_KEY = (
    "mft-al:mft-goal-provisional-full-l96230-b7c30cb70b95-v1:"
    "a1e4f70cefa1af04673c73a6131bf490c0cc14b5:"
    "e6b9b9d20a832ff5c3f7ca97218737a0b8650781:62f2b846d05bbc58"
)
SOLVER_REVISION = "a1e4f70cefa1af04673c73a6131bf490c0cc14b5"
LIBRARY_REVISION = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
PARAMETER_DIGEST_PREFIX = "62f2b846d05bbc58"

NEW_TASK_NAME = "mft-goal-postdeadline-diagnostic-full-retry-t96307-v1"
NEW_DEDUPE_KEY = (
    f"mft-al:{NEW_TASK_NAME}:{SOLVER_REVISION}:{LIBRARY_REVISION}:"
    f"{PARAMETER_DIGEST_PREFIX}:postdeadline-v1"
)
NEW_RETAIN_TOKEN = "postdeadline-diagnostic-full-retry-t96307-7a7da2ab5daeb028"
NEW_RETAINED_PATH = f"goal-fea-retained/{NEW_RETAIN_TOKEN}/full.aedt"

SOURCE_WORKDIR_SLUG = (
    "mft_campaign-mft_goal_provisional_full_l96230_b7c30cb70b95_v1-"
    "sa1e4f70cefa1-le6b9b9d20a83-p62f2b846d05bbc58-t03b90b61fa5f4e8a"
)
NEW_WORKDIR_SLUG = (
    "mft_campaign-mft_goal_postdeadline_diagnostic_full_retry_t96307_v1-"
    "sa1e4f70cefa1-le6b9b9d20a83-p62f2b846d05bbc58-t7a7da2ab5daeb028"
)
SOURCE_RETAIN_TOKEN = "03b90b61fa5f4e8a"

ORIGINAL_DEADLINE_KST = "2026-07-26T18:00:00+09:00"
ORIGINAL_DEADLINE_UTC = "2026-07-26T09:00:00Z"
SCHEDULER_TIMEOUT_SECONDS = 86_400
SOLVER_TIMEOUT_SECONDS = 79_200
TIMEOUT_KILL_GRACE_SECONDS = 300
NOMINAL_POST_SOLVER_WINDOW_SECONDS = SCHEDULER_TIMEOUT_SECONDS - SOLVER_TIMEOUT_SECONDS
PLANNED_MINIMUM_RETENTION_WINDOW_SECONDS = 3_600
PLANNED_PRE_SOLVER_OVERHEAD_ALLOWANCE_SECONDS = (
    NOMINAL_POST_SOLVER_WINDOW_SECONDS - PLANNED_MINIMUM_RETENTION_WINDOW_SECONDS
)

EXPECTED_SOURCE_BLOBS = {
    "task": {
        "endpoint": f"/api/tasks/{SOURCE_TASK_ID}",
        "size_bytes": 1992,
        "sha256": ("713cd9217d6718aa38bc7fb14e88464833b1eca606aade5451a7e46efa52b241"),
    },
    "task.sh": {
        "endpoint": (
            f"/api/tasks/{SOURCE_TASK_ID}/remote-file?"
            "base=remote_dir&path=task.sh&max_bytes=1048576"
        ),
        "size_bytes": 11243,
        "sha256": ("47bf1d479ad5e0d74dcec46ed05a71cfd6151253c34d06f7b32da9dc43b8b3bf"),
    },
    "stdout": {
        "endpoint": (
            f"/api/tasks/{SOURCE_TASK_ID}/remote-file?"
            "base=remote_dir&path=stdout&max_bytes=10485760"
        ),
        "size_bytes": 0,
        "sha256": ("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    },
    "stderr": {
        "endpoint": (
            f"/api/tasks/{SOURCE_TASK_ID}/remote-file?"
            "base=remote_dir&path=stderr&max_bytes=10485760"
        ),
        "size_bytes": 0,
        "sha256": ("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    },
}

SOURCE_COMMAND_START = (
    b"source /etc/profile.d/lmod.sh 2>/dev/null || true; "
    b"module load ansys-electronics/v252"
)
EXPECTED_SOURCE_COMMAND_SIZE = 9909
EXPECTED_SOURCE_COMMAND_SHA256 = (
    "87f34b089a2c3745e313a429dc885ba0c34e8f72fb0e87228a2a2997411bb1c1"
)
EXPECTED_CANDIDATE_JSON_SIZE = 1532
EXPECTED_CANDIDATE_JSON_SHA256 = (
    "62f2b846d05bbc5880165c6086fe9897e04d0588b0b8767fceb35102e4be9a44"
)

SOURCE_SOLVER_INVOCATION = (
    b"python run_simulation_260706.py --fixed --thermal --headless --full "
    b"--params cand.json"
)
NEW_SOLVER_INVOCATION = (
    b"timeout --signal=TERM --kill-after="
    + str(TIMEOUT_KILL_GRACE_SECONDS).encode("ascii")
    + b"s "
    + str(SOLVER_TIMEOUT_SECONDS).encode("ascii")
    + b"s "
    + SOURCE_SOLVER_INVOCATION
)

SOURCE_MARKER_CONTRACT_SHA256 = (
    "8fb549b74eb3b0795084754c8300911aea921cc23443f0906e46fe77177d4d7d"
)
SOURCE_MARKER_JSON = (
    b'{"owner":"MFT_1MW_2026v1","preserve":true,"reason":"Retain goal FEA '
    b"AEDT until authenticated package collection; stage=full; dedupe_key="
    + SOURCE_DEDUPE_KEY.encode("ascii")
    + b'","schema":"slurm-scheduler-prune-protection-v1"}'
)

SOURCE_ENV_SETUP = """source /etc/profile.d/lmod.sh 2>/dev/null || true
module load ansys-electronics/v252 2>/dev/null || export ANSYSEM_ROOT252=/opt/ohpc/pub/Electronics/v252/Linux64
export FLEXLM_TIMEOUT=3000000
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then source "$HOME/miniconda3/etc/profile.d/conda.sh"; elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then source "$HOME/anaconda3/etc/profile.d/conda.sh"; elif [ -f "/home1/$USER/miniconda3/etc/profile.d/conda.sh" ]; then source "/home1/$USER/miniconda3/etc/profile.d/conda.sh"; elif [ -f "/home1/$USER/anaconda3/etc/profile.d/conda.sh" ]; then source "/home1/$USER/anaconda3/etc/profile.d/conda.sh"; fi"""
SOURCE_ENV_SETUP_SHA256 = (
    "e8c5a5fba4e1d95b8a2bf6866a66c8d882f691a3d9ea2d6d005f51b6d57e6f5a"
)

CLASSIFICATION = {
    "original_deadline_missed": True,
    "post_deadline_retry": True,
    "diagnostic_only": True,
    "production": False,
    "canonical": False,
    "production_truth_eligible": False,
    "canonical_dataset_mutation_allowed": False,
    "repairs_original_deadline": False,
}

REQUIRED_SOURCE_TASK_FIELDS = {
    "task_id": SOURCE_TASK_ID,
    "id": SOURCE_TASK_ID,
    "name": SOURCE_TASK_NAME,
    "status": "failed",
    "state": "failed",
    "exit_code": 124,
    "failure_message": "task timed out after 43200s",
    "account_name": "dhj02",
    "requested_account_name": "dhj02",
    "remote_cwd": "__SLURM_SCHEDULER_ACCOUNT_WORKSPACE__/runs",
    "required_capability": "conda:pyaedt2026v1",
    "env_profile": "pyaedt2026v1",
    "project": "MFT_1MW_2026v1",
    "scheduling_profile": "fea_bursty",
    "aedt_backend": "standalone",
    "priority": 100,
    "timeout_seconds": 43_200,
    "dedupe_key": SOURCE_DEDUPE_KEY,
    "max_workers_per_node": 0,
    "cpus": 16,
    "memory_mb": 98_304,
    "gpus": 0,
    "partition": "auto",
    "node_name": "n116",
    "requested_node_name": "n116",
    "node_name_policy": "strict",
    "strict_node_placement": True,
}

REQUIRED_PHYSICS_FIELDS = {
    "full_model": 1,
    "matrix_on": 1,
    "loss_on": 1,
    "thermal_on": 1,
    "loss_sym_on": 0,
    "thermal_symmetry": "full",
    "fan_velocity": 1.5,
    "conductor_temp_C": 80.0,
    "air_temp": 50.0,
    "plate_temp": 50.0,
    "physics_data_revision": "mft1mw-1k101-native-lamination-kf0p85-v3",
}

PLAN_SCHEMA = "mft-postdeadline-diagnostic-full-retry-plan-v1"
PAYLOAD_METADATA_SCHEMA = "mft-postdeadline-diagnostic-full-retry-task-metadata-v1"
ATTEMPT_SCHEMA = "mft-single-post-attempt-ledger-v1"
RECEIPT_SCHEMA = "mft-postdeadline-diagnostic-full-retry-receipt-v1"


class ContractError(RuntimeError):
    """Raised when a sealed source, plan, or submission contract drifts."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return _sha256(_canonical_bytes(value))


def _now_utc() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _write_json_exclusive(path: Path, value: Any) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ContractError(f"refusing to overwrite existing file: {path}") from exc


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read sealed JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"sealed JSON root is not an object: {path}")
    return value


class SchedulerClient:
    """Small urllib client with an in-process at-most-one POST guard."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._post_count = 0
        self._post_lock = threading.Lock()

    @property
    def post_count(self) -> int:
        return self._post_count

    def _request_bytes(
        self,
        method: str,
        endpoint: str,
        payload: dict[str, Any] | None = None,
    ) -> tuple[int, bytes]:
        body = None if payload is None else _canonical_bytes(payload)
        req = request.Request(
            self.base_url + endpoint,
            data=body,
            method=method,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "mft-postdeadline-full-retry/1",
            },
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                return int(response.status), response.read()
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ContractError(
                f"Scheduler {method} {endpoint} failed: HTTP {exc.code}: {detail}"
            ) from exc
        except error.URLError as exc:
            raise ContractError(
                f"Scheduler {method} {endpoint} failed: {exc.reason}"
            ) from exc

    def get_bytes(self, endpoint: str) -> bytes:
        status, body = self._request_bytes("GET", endpoint)
        if status != 200:
            raise ContractError(f"Scheduler GET {endpoint} returned HTTP {status}")
        return body

    def get_json(self, endpoint: str) -> Any:
        body = self.get_bytes(endpoint)
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise ContractError(
                f"Scheduler GET {endpoint} returned invalid JSON"
            ) from exc

    def post_json_once(self, endpoint: str, payload: dict[str, Any]) -> tuple[int, Any]:
        with self._post_lock:
            if self._post_count != 0:
                raise ContractError("at-most-one POST guard has already been consumed")
            # Count before the network call.  A timeout is an ambiguous attempt
            # and must not be retried in this process.
            self._post_count = 1
        status, body = self._request_bytes("POST", endpoint, payload)
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ContractError("Scheduler POST returned invalid JSON") from exc
        return status, parsed


@dataclass(frozen=True)
class SourceBundle:
    task: dict[str, Any]
    blobs: dict[str, bytes]
    command: bytes
    command_offset_in_task_sh: int


@dataclass(frozen=True)
class ByteSubstitution:
    name: str
    reason: str
    old: bytes
    new: bytes
    expected_count: int


def _verify_blob(label: str, data: bytes) -> None:
    expected = EXPECTED_SOURCE_BLOBS[label]
    observed_size = len(data)
    observed_sha = _sha256(data)
    if observed_size != expected["size_bytes"] or observed_sha != expected["sha256"]:
        raise ContractError(
            f"source {label} drifted: "
            f"size={observed_size} sha256={observed_sha}; "
            f"expected size={expected['size_bytes']} "
            f"sha256={expected['sha256']}"
        )


def _validate_source_task(task: dict[str, Any]) -> None:
    for key, expected in REQUIRED_SOURCE_TASK_FIELDS.items():
        if task.get(key) != expected:
            raise ContractError(
                f"source task field drifted for {key}: "
                f"{task.get(key)!r} != {expected!r}"
            )


def _extract_source_command(task_sh: bytes) -> tuple[bytes, int]:
    if task_sh.count(SOURCE_COMMAND_START) != 1:
        raise ContractError("source command start sentinel multiplicity drifted")
    offset = task_sh.index(SOURCE_COMMAND_START)
    command_with_wrapper_newline = task_sh[offset:]
    if not command_with_wrapper_newline.endswith(b"\n"):
        raise ContractError("source task.sh no longer has its wrapper newline")
    command = command_with_wrapper_newline[:-1]
    if (
        len(command) != EXPECTED_SOURCE_COMMAND_SIZE
        or _sha256(command) != EXPECTED_SOURCE_COMMAND_SHA256
    ):
        raise ContractError(
            "extracted source command identity drifted: "
            f"size={len(command)} sha256={_sha256(command)}"
        )
    return command, offset


def fetch_and_validate_source(client: SchedulerClient) -> SourceBundle:
    """Fetch the four independent source objects concurrently and seal them."""

    names = tuple(EXPECTED_SOURCE_BLOBS)

    def fetch(name: str) -> tuple[str, bytes]:
        endpoint = str(EXPECTED_SOURCE_BLOBS[name]["endpoint"])
        return name, client.get_bytes(endpoint)

    with ThreadPoolExecutor(max_workers=len(names)) as executor:
        blobs = dict(executor.map(fetch, names))
    for label in names:
        _verify_blob(label, blobs[label])
    try:
        task = json.loads(blobs["task"])
    except json.JSONDecodeError as exc:
        raise ContractError("source task snapshot is invalid JSON") from exc
    if not isinstance(task, dict):
        raise ContractError("source task snapshot root is not an object")
    _validate_source_task(task)

    env_setup_bytes = SOURCE_ENV_SETUP.encode("utf-8")
    if (
        _sha256(env_setup_bytes) != SOURCE_ENV_SETUP_SHA256
        or blobs["task.sh"].count(env_setup_bytes) != 1
    ):
        raise ContractError("source environment setup identity drifted")
    command, offset = _extract_source_command(blobs["task.sh"])
    return SourceBundle(
        task=task,
        blobs=blobs,
        command=command,
        command_offset_in_task_sh=offset,
    )


def _extract_candidate(command: bytes) -> bytes:
    matches = list(
        re.finditer(
            rb"printf '%s' '(\{\"I1_rated\".*?\})' > cand\.json",
            command,
        )
    )
    if len(matches) != 1:
        raise ContractError("Full candidate JSON multiplicity drifted")
    return matches[0].group(1)


def _validate_candidate(candidate: bytes) -> None:
    if (
        len(candidate) != EXPECTED_CANDIDATE_JSON_SIZE
        or _sha256(candidate) != EXPECTED_CANDIDATE_JSON_SHA256
    ):
        raise ContractError(
            "Full physics candidate bytes drifted: "
            f"size={len(candidate)} sha256={_sha256(candidate)}"
        )
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ContractError("Full physics candidate is invalid JSON") from exc
    for key, expected in REQUIRED_PHYSICS_FIELDS.items():
        if parsed.get(key) != expected:
            raise ContractError(
                f"Full physics contract drifted for {key}: "
                f"{parsed.get(key)!r} != {expected!r}"
            )


def _substitution_specs() -> tuple[ByteSubstitution, ...]:
    if _sha256(SOURCE_MARKER_JSON).lower() != SOURCE_MARKER_CONTRACT_SHA256:
        raise ContractError("embedded source retention marker contract drifted")
    new_marker_json = SOURCE_MARKER_JSON.replace(
        SOURCE_DEDUPE_KEY.encode("ascii"),
        NEW_DEDUPE_KEY.encode("ascii"),
    )
    if new_marker_json == SOURCE_MARKER_JSON:
        raise ContractError("new retention marker did not change")
    new_marker_sha = _sha256(new_marker_json).encode("ascii")
    return (
        ByteSubstitution(
            name="workdir_slug",
            reason="isolate the retry work directory from source task 96307",
            old=SOURCE_WORKDIR_SLUG.encode("ascii"),
            new=NEW_WORKDIR_SLUG.encode("ascii"),
            expected_count=2,
        ),
        ByteSubstitution(
            name="retention_dedupe_key",
            reason="bind both retention contracts to the new diagnostic task",
            old=SOURCE_DEDUPE_KEY.encode("ascii"),
            new=NEW_DEDUPE_KEY.encode("ascii"),
            expected_count=2,
        ),
        ByteSubstitution(
            name="retained_path_token",
            reason="write success artifacts under a new non-canonical path",
            old=SOURCE_RETAIN_TOKEN.encode("ascii"),
            new=NEW_RETAIN_TOKEN.encode("ascii"),
            expected_count=7,
        ),
        ByteSubstitution(
            name="retention_marker_contract_sha256",
            reason="update the hash implied solely by the new retention dedupe key",
            old=SOURCE_MARKER_CONTRACT_SHA256.encode("ascii"),
            new=new_marker_sha,
            expected_count=1,
        ),
        ByteSubstitution(
            name="solver_walltime_guard_prefix",
            reason=(
                "cap the solver inside the 24h Scheduler timeout and leave "
                "success-path AEDT preservation time"
            ),
            old=SOURCE_SOLVER_INVOCATION,
            new=NEW_SOLVER_INVOCATION,
            expected_count=1,
        ),
    )


def transform_source_command(
    source_command: bytes,
) -> tuple[bytes, list[dict[str, Any]], bytes]:
    """Apply and prove the exact byte-level substitution allow-list."""

    if (
        len(source_command) != EXPECTED_SOURCE_COMMAND_SIZE
        or _sha256(source_command) != EXPECTED_SOURCE_COMMAND_SHA256
    ):
        raise ContractError("source command does not match task 96307")
    candidate = _extract_candidate(source_command)
    _validate_candidate(candidate)
    if source_command.count(SOURCE_MARKER_JSON) != 1:
        raise ContractError("source retention marker JSON multiplicity drifted")

    transformed = source_command
    specs = _substitution_specs()
    audit: list[dict[str, Any]] = []
    for sequence, spec in enumerate(specs, start=1):
        observed_count = transformed.count(spec.old)
        if observed_count != spec.expected_count:
            raise ContractError(
                f"allow-listed substitution {spec.name} count drifted: "
                f"{observed_count} != {spec.expected_count}"
            )
        transformed = transformed.replace(
            spec.old,
            spec.new,
            spec.expected_count,
        )
        audit.append(
            {
                "sequence": sequence,
                "name": spec.name,
                "reason": spec.reason,
                "occurrences": spec.expected_count,
                "old_size_bytes": len(spec.old),
                "old_sha256": _sha256(spec.old),
                "old_text": spec.old.decode("ascii"),
                "new_size_bytes": len(spec.new),
                "new_sha256": _sha256(spec.new),
                "new_text": spec.new.decode("ascii"),
            }
        )

    transformed_candidate = _extract_candidate(transformed)
    if transformed_candidate != candidate:
        raise ContractError("physics candidate changed outside the allow-list")
    if transformed.count(SOURCE_SOLVER_INVOCATION) != 1:
        raise ContractError("unchanged Full solver argv is not retained exactly once")

    # Prove that reversing only the declared substitutions recreates every
    # byte of the source command.  This is stronger than checking select fields.
    restored = transformed
    for spec in reversed(specs):
        observed_count = restored.count(spec.new)
        if observed_count != spec.expected_count:
            raise ContractError(
                f"reverse substitution {spec.name} count drifted: "
                f"{observed_count} != {spec.expected_count}"
            )
        restored = restored.replace(spec.new, spec.old, spec.expected_count)
    if restored != source_command:
        raise ContractError("non-allow-listed command bytes changed")

    return transformed, audit, candidate


def _source_evidence(bundle: SourceBundle) -> dict[str, Any]:
    blobs = {
        label: {
            "endpoint": str(EXPECTED_SOURCE_BLOBS[label]["endpoint"]),
            "size_bytes": len(bundle.blobs[label]),
            "sha256": _sha256(bundle.blobs[label]),
        }
        for label in EXPECTED_SOURCE_BLOBS
    }
    return {
        "task_id": SOURCE_TASK_ID,
        "task_name": SOURCE_TASK_NAME,
        "task_status": bundle.task["status"],
        "task_exit_code": bundle.task["exit_code"],
        "task_failure_message": bundle.task["failure_message"],
        "read_only": True,
        "blobs": blobs,
        "extracted_command": {
            "task_sh_byte_offset": bundle.command_offset_in_task_sh,
            "size_bytes": len(bundle.command),
            "sha256": _sha256(bundle.command),
        },
        "environment_setup": {
            "size_bytes": len(SOURCE_ENV_SETUP.encode("utf-8")),
            "sha256": SOURCE_ENV_SETUP_SHA256,
        },
    }


def _execution_budget() -> dict[str, Any]:
    return {
        "scheduler_timeout_seconds": SCHEDULER_TIMEOUT_SECONDS,
        "solver_timeout_seconds": SOLVER_TIMEOUT_SECONDS,
        "solver_timeout_kill_grace_seconds": TIMEOUT_KILL_GRACE_SECONDS,
        "nominal_post_solver_window_seconds": (NOMINAL_POST_SOLVER_WINDOW_SECONDS),
        "planned_pre_solver_overhead_allowance_seconds": (
            PLANNED_PRE_SOLVER_OVERHEAD_ALLOWANCE_SECONDS
        ),
        "planned_minimum_success_retention_window_seconds": (
            PLANNED_MINIMUM_RETENTION_WINDOW_SECONDS
        ),
        "retention_runs_only_after_solver_success": True,
    }


def _build_deterministic_sections(
    bundle: SourceBundle,
) -> dict[str, Any]:
    transformed, substitutions, candidate = transform_source_command(bundle.command)
    evidence = _source_evidence(bundle)
    budget = _execution_budget()
    allowlist_sha = _canonical_sha256(substitutions)
    command_reuse = {
        "source_command_size_bytes": len(bundle.command),
        "source_command_sha256": _sha256(bundle.command),
        "transformed_command_size_bytes": len(transformed),
        "transformed_command_sha256": _sha256(transformed),
        "non_allowlisted_bytes_preserved": True,
        "full_solver_argv_preserved_byte_exact": True,
        "physics_candidate": {
            "size_bytes": len(candidate),
            "sha256": _sha256(candidate),
            "byte_exact_unchanged": True,
        },
        "substitution_allowlist": substitutions,
        "substitution_allowlist_sha256": allowlist_sha,
    }
    metadata = {
        "schema_version": PAYLOAD_METADATA_SCHEMA,
        **CLASSIFICATION,
        "original_deadline_kst": ORIGINAL_DEADLINE_KST,
        "original_deadline_utc": ORIGINAL_DEADLINE_UTC,
        "source_task_id": SOURCE_TASK_ID,
        "source_task_name": SOURCE_TASK_NAME,
        "source_task_status": "failed",
        "source_task_exit_code": 124,
        "source_blob_sha256": {
            key: value["sha256"] for key, value in evidence["blobs"].items()
        },
        "source_command_sha256": _sha256(bundle.command),
        "transformed_command_sha256": _sha256(transformed),
        "physics_candidate_sha256": _sha256(candidate),
        "substitution_allowlist_sha256": allowlist_sha,
        "retained_artifact_path": NEW_RETAINED_PATH,
        "scheduler_timeout_seconds": SCHEDULER_TIMEOUT_SECONDS,
        "solver_timeout_seconds": SOLVER_TIMEOUT_SECONDS,
        "planned_minimum_success_retention_window_seconds": (
            PLANNED_MINIMUM_RETENTION_WINDOW_SECONDS
        ),
    }
    payload = {
        "name": NEW_TASK_NAME,
        "remote_cwd": "__SLURM_SCHEDULER_ACCOUNT_WORKSPACE__/runs",
        "command": transformed.decode("utf-8"),
        "env_setup": SOURCE_ENV_SETUP,
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "account_name": "dhj02",
        "cpus": 16,
        "memory_mb": 98_304,
        "scheduling_profile": "fea_bursty",
        "aedt_backend": "standalone",
        "gpus": 0,
        "gpu_model": "",
        "partition": "auto",
        "node_name": "n116",
        "node_name_policy": "strict",
        "exclusive_node": False,
        "priority": 100,
        "timeout_seconds": SCHEDULER_TIMEOUT_SECONDS,
        "dedupe_key": NEW_DEDUPE_KEY,
        "max_workers_per_node": 0,
        "same_node_as_task_id": 0,
        "project": "MFT_1MW_2026v1",
        "entrypoint": "",
        "cleanup_globs": NEW_WORKDIR_SLUG,
        "payload_json": metadata,
    }
    return {
        "source": evidence,
        "command_reuse": command_reuse,
        "execution_budget": budget,
        "payload": payload,
    }


def build_plan(
    client: SchedulerClient,
    *,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    bundle = fetch_and_validate_source(client)
    sections = _build_deterministic_sections(bundle)
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "created_at_utc": created_at_utc or _now_utc(),
        "classification": dict(CLASSIFICATION),
        "original_deadline": {
            "kst": ORIGINAL_DEADLINE_KST,
            "utc": ORIGINAL_DEADLINE_UTC,
            "missed": True,
            "status": "missed_and_not_repairable_by_this_retry",
        },
        **sections,
        "submission_guard": {
            "endpoint": "/api/tasks",
            "http_method": "POST",
            "maximum_post_attempts": 1,
            "explicit_apply_required": True,
            "immutable_attempt_ledger_required": True,
            "automatic_retry_allowed": False,
            "source_revalidation_required_immediately_before_post": True,
            "scheduler_dedupe_key": NEW_DEDUPE_KEY,
        },
    }
    plan["seal"] = {
        "algorithm": "sha256",
        "canonicalization": "sorted-keys-compact-utf8-json-v1",
        "sha256": _canonical_sha256(plan),
    }
    return plan


def validate_plan_for_submission(
    plan: dict[str, Any],
    bundle: SourceBundle,
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "created_at_utc",
        "classification",
        "original_deadline",
        "source",
        "command_reuse",
        "execution_budget",
        "payload",
        "submission_guard",
        "seal",
    }
    if set(plan) != expected_keys:
        raise ContractError("plan top-level fields drifted")
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ContractError("plan schema drifted")
    seal = plan.get("seal")
    if not isinstance(seal, dict):
        raise ContractError("plan seal is missing")
    unsigned = dict(plan)
    unsigned.pop("seal")
    expected_seal = {
        "algorithm": "sha256",
        "canonicalization": "sorted-keys-compact-utf8-json-v1",
        "sha256": _canonical_sha256(unsigned),
    }
    if seal != expected_seal:
        raise ContractError("plan SHA-256 seal is invalid")
    if plan.get("classification") != CLASSIFICATION:
        raise ContractError("non-production/non-canonical classification drifted")
    if plan.get("original_deadline") != {
        "kst": ORIGINAL_DEADLINE_KST,
        "utc": ORIGINAL_DEADLINE_UTC,
        "missed": True,
        "status": "missed_and_not_repairable_by_this_retry",
    }:
        raise ContractError("original missed deadline metadata drifted")
    expected_sections = _build_deterministic_sections(bundle)
    for key, expected in expected_sections.items():
        if plan.get(key) != expected:
            raise ContractError(f"sealed plan section drifted for {key}")
    expected_guard = {
        "endpoint": "/api/tasks",
        "http_method": "POST",
        "maximum_post_attempts": 1,
        "explicit_apply_required": True,
        "immutable_attempt_ledger_required": True,
        "automatic_retry_allowed": False,
        "source_revalidation_required_immediately_before_post": True,
        "scheduler_dedupe_key": NEW_DEDUPE_KEY,
    }
    if plan.get("submission_guard") != expected_guard:
        raise ContractError("single-POST submission guard drifted")
    payload = plan["payload"]
    if not isinstance(payload, dict):
        raise ContractError("plan payload is not an object")
    return payload


def _task_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ("items", "tasks", "results"):
            items = value.get(key)
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
    raise ContractError("Scheduler task preflight returned an unexpected shape")


def submit_plan(
    plan: dict[str, Any],
    client: SchedulerClient,
    *,
    apply: bool,
    attempt_ledger: Path | None = None,
) -> dict[str, Any]:
    """Revalidate source/plan and optionally consume the sole POST attempt."""

    bundle = fetch_and_validate_source(client)
    payload = validate_plan_for_submission(plan, bundle)
    query = (
        "/api/tasks?limit=100&project="
        + parse.quote(str(payload["project"]), safe="")
        + "&name_prefix="
        + parse.quote(str(payload["name"]), safe="")
    )
    existing = _task_list(client.get_json(query))
    collisions = [
        item
        for item in existing
        if item.get("name") == payload["name"]
        or item.get("dedupe_key") == payload["dedupe_key"]
    ]
    if collisions:
        item = collisions[0]
        raise ContractError(
            "at-most-one guard found an existing retry task: "
            f"id={item.get('id')} status={item.get('status')}"
        )

    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "created_at_utc": _now_utc(),
        "plan_sha256": plan["seal"]["sha256"],
        "source_revalidated": True,
        "preflight_collision_count": 0,
        "post_attempt_limit": 1,
        "post_attempts_consumed": 0,
        "submitted": False,
        **CLASSIFICATION,
    }
    if not apply:
        receipt["status"] = "validated_no_post_apply_required"
        receipt["receipt_sha256"] = _canonical_sha256(receipt)
        return receipt
    if attempt_ledger is None:
        raise ContractError("--apply requires an immutable attempt ledger path")

    attempt = {
        "schema_version": ATTEMPT_SCHEMA,
        "created_at_utc": _now_utc(),
        "plan_sha256": plan["seal"]["sha256"],
        "task_name": payload["name"],
        "dedupe_key": payload["dedupe_key"],
        "endpoint": "/api/tasks",
        "post_attempt_consumed": True,
        "maximum_post_attempts": 1,
        "automatic_retry_allowed": False,
        **CLASSIFICATION,
    }
    # This exclusive write happens before the network call.  If the POST has
    # an ambiguous outcome, rerunning with the same ledger is fail-closed.
    _write_json_exclusive(attempt_ledger, attempt)
    status, response = client.post_json_once("/api/tasks", payload)
    receipt["post_attempts_consumed"] = client.post_count
    receipt["http_status"] = status
    receipt["scheduler_response"] = response
    if status not in {200, 201} or not isinstance(response, dict):
        raise ContractError(
            f"Scheduler rejected the single POST: HTTP {status} {response!r}"
        )
    task_id = int(response.get("id") or response.get("task_id") or 0)
    if task_id <= 0 or response.get("name") != NEW_TASK_NAME:
        raise ContractError(
            f"Scheduler receipt identity mismatch after single POST: {response!r}"
        )
    if (
        response.get("dedupe_key") not in {None, NEW_DEDUPE_KEY}
        or int(response.get("timeout_seconds") or SCHEDULER_TIMEOUT_SECONDS)
        != SCHEDULER_TIMEOUT_SECONDS
    ):
        raise ContractError(
            f"Scheduler receipt contract mismatch after single POST: {response!r}"
        )
    receipt.update(
        {
            "status": "accepted",
            "submitted": True,
            "task_id": task_id,
            "task_name": NEW_TASK_NAME,
            "dedupe_key": NEW_DEDUPE_KEY,
            "deduped": bool(response.get("deduped", False)),
        }
    )
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


def _summary(value: dict[str, Any]) -> dict[str, Any]:
    result = {
        "schema_version": value.get("schema_version"),
        "original_deadline_missed": value.get(
            "original_deadline_missed",
            value.get("classification", {}).get("original_deadline_missed"),
        ),
        "production": value.get(
            "production", value.get("classification", {}).get("production")
        ),
        "canonical": value.get(
            "canonical", value.get("classification", {}).get("canonical")
        ),
    }
    if "seal" in value:
        result.update(
            {
                "plan_sha256": value["seal"]["sha256"],
                "task_name": value["payload"]["name"],
                "timeout_seconds": value["payload"]["timeout_seconds"],
                "retained_artifact_path": value["payload"]["payload_json"][
                    "retained_artifact_path"
                ],
                "post_performed": False,
            }
        )
    else:
        result.update(
            {
                "status": value.get("status"),
                "submitted": value.get("submitted"),
                "post_attempts_consumed": value.get("post_attempts_consumed"),
                "task_id": value.get("task_id"),
            }
        )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scheduler-url",
        default=DEFAULT_SCHEDULER_URL,
        help=f"Scheduler base URL (default: {DEFAULT_SCHEDULER_URL})",
    )
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    plan_parser = subparsers.add_parser(
        "plan", help="GET, verify, and write a sealed plan; never POST"
    )
    plan_parser.add_argument("--output", type=Path, required=True)

    submit_parser = subparsers.add_parser(
        "submit",
        help="revalidate a sealed plan; POST only with --apply",
    )
    submit_parser.add_argument("--plan", type=Path, required=True)
    submit_parser.add_argument("--apply", action="store_true")
    submit_parser.add_argument(
        "--attempt-ledger",
        type=Path,
        help=(
            "exclusive at-most-one POST ledger; required by --apply "
            "(default: PLAN.post-attempt.json)"
        ),
    )
    submit_parser.add_argument(
        "--receipt",
        type=Path,
        help=(
            "exclusive submission receipt path (default: PLAN.submission-receipt.json)"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    client = SchedulerClient(args.scheduler_url)
    try:
        if args.command_name == "plan":
            plan = build_plan(client)
            _write_json_exclusive(args.output, plan)
            print(json.dumps(_summary(plan), indent=2, sort_keys=True))
            return 0

        plan_path = args.plan.resolve(strict=True)
        plan = _load_json_object(plan_path)
        attempt_path = args.attempt_ledger
        if args.apply and attempt_path is None:
            attempt_path = Path(str(plan_path) + ".post-attempt.json")
        receipt = submit_plan(
            plan,
            client,
            apply=bool(args.apply),
            attempt_ledger=attempt_path,
        )
        if args.apply:
            receipt_path = args.receipt or Path(
                str(plan_path) + ".submission-receipt.json"
            )
            _write_json_exclusive(receipt_path, receipt)
        print(json.dumps(_summary(receipt), indent=2, sort_keys=True))
        return 0
    except ContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
