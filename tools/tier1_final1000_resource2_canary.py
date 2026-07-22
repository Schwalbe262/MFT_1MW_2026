"""Prepare and evaluate one isolated 2-CPU Final1000 NSGA efficiency canary.

The command has exactly one optional remote mutation: ``submit --apply`` may
POST one additive Scheduler task.  It has no cancel, preempt, FEA, AEDT, or
publication surface.  The candidate is deliberately outside the production
controller/harvester name and dedupe namespaces.

Rendering is impossible until the pinned 4-CPU task has a completed,
promotion-eligible, GET-only terminal attestation.  Submission re-authenticates
that baseline, the bundle publication/manifest, the live remote READY, and the
complete production plus resource-canary namespaces before and after its sole
POST.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import time
from typing import Any, Mapping, Sequence
import urllib.error
import urllib.parse
import urllib.request

try:
    import tier1_final1000_phase_b_shape1_canary as shape1
    from tier1_corrected_current7_receipt import canonical_sha256
    from tier1_corrected_current7_slurm_bundle import (
        STATUS_SCHEMA,
        build_task_payload as build_current7_task_payload,
        sha256_file,
        validate_required_runtime_code,
    )
    from tier1_corrected_current7_slurm_publish import (
        DEFAULT_ACCOUNTS,
        DEFAULT_SCHEDULER_SOURCE,
        scheduler_publication_transport,
    )
    from tier1_corrected_current7_slurm_seed_runner import validate_result
    from tier1_final1000_multiseed_contract import (
        scheduler_task_identity_matches,
        scheduler_task_observation,
    )
    from tier1_final1000_multiseed_phase_b_contract import (
        INFERENCE_SAFETY,
        REMOTE_CODE_FILES,
    )
    from tier1_final1000_slurm_launch import (
        DEFAULT_MAX_WORKERS_PER_NODE,
        DEFAULT_PEAK_RSS_GATE_BYTES,
        DEFAULT_PRIORITY,
        DEFAULT_TIMEOUT_SECONDS,
        REQUIRED_SCHEDULER_FIELDS,
        build_stage_task,
        validate_stage_binding,
        validate_task,
    )
    from tier1_final1000_stage_profiles import BY_ID
except ImportError:  # pragma: no cover - repository import path
    from tools import tier1_final1000_phase_b_shape1_canary as shape1
    from tools.tier1_corrected_current7_receipt import canonical_sha256
    from tools.tier1_corrected_current7_slurm_bundle import (
        STATUS_SCHEMA,
        build_task_payload as build_current7_task_payload,
        sha256_file,
        validate_required_runtime_code,
    )
    from tools.tier1_corrected_current7_slurm_publish import (
        DEFAULT_ACCOUNTS,
        DEFAULT_SCHEDULER_SOURCE,
        scheduler_publication_transport,
    )
    from tools.tier1_corrected_current7_slurm_seed_runner import validate_result
    from tools.tier1_final1000_multiseed_contract import (
        scheduler_task_identity_matches,
        scheduler_task_observation,
    )
    from tools.tier1_final1000_multiseed_phase_b_contract import (
        INFERENCE_SAFETY,
        REMOTE_CODE_FILES,
    )
    from tools.tier1_final1000_slurm_launch import (
        DEFAULT_MAX_WORKERS_PER_NODE,
        DEFAULT_PEAK_RSS_GATE_BYTES,
        DEFAULT_PRIORITY,
        DEFAULT_TIMEOUT_SECONDS,
        REQUIRED_SCHEDULER_FIELDS,
        build_stage_task,
        validate_stage_binding,
        validate_task,
    )
    from tools.tier1_final1000_stage_profiles import BY_ID


CONFIG_SCHEMA = "mft-tier1-final1000-resource2-canary-config-v1"
PACKAGE_SCHEMA = "mft-tier1-final1000-resource2-canary-package-v1"
SUBMISSION_SCHEMA = "mft-tier1-final1000-resource2-canary-submission-v1"
TELEMETRY_SCHEMA = "mft-tier1-final1000-resource2-canary-telemetry-v1"
TERMINAL_SCHEMA = "mft-tier1-final1000-resource2-canary-terminal-v1"
REMOTE_TERMINAL_SCHEMA = "mft-tier1-final1000-resource2-canary-remote-terminal-v1"
CANARY_PAYLOAD_SCHEMA = "mft-tier1-final1000-resource2-canary-payload-v1"

ENTRY_STAGE_ID = "entry-1200-t125"
BASELINE_TASK_ID = 84_880
BASELINE_SEED = 2_257_499_998
BASELINE_PACKAGE_SHA256 = (
    "c2c2337d556ea0ef05ab534d3dd73654bc1ec6a237011b3653274eff9ca88489"
)
BASELINE_SUBMISSION_SHA256 = (
    "60766f44e0b15e143e32e5bd2e628108bc6f8695eba72a0772ed18162d29fe9c"
)
BASELINE_CPUS = 4
CANDIDATE_CPUS = 2
CANDIDATE_MEMORY_MB = 28_672
BASELINE_THEORETICAL_SLOTS = 447
CANDIDATE_THEORETICAL_SLOTS = 695
MIN_SINGLE_SEED_THROUGHPUT_RATIO = 0.75
MIN_SLOT_WEIGHTED_THROUGHPUT_RATIO = 1.10
MIN_CORE_USE_RATIO = 0.80
PRODUCTION_PREFIX = "mft-t1fg-"
PRODUCTION_DEDUPE_PREFIX = "mft-tier1-final1000:"
CANARY_PREFIX = "mft-t1r2-"
CANARY_DEDUPE_PREFIX = "mft-tier1-resource2-canary:"
PAGE_SIZE = 10_000
MAX_PAGES = 10_000
DIAGNOSTIC_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{7,127}$")
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
TELEMETRY_FILENAME = "resource2_canary_telemetry.json"
REMOTE_EVIDENCE_HARD_CAP_BYTES = 16 * 1024**2
REMOTE_EVIDENCE_MAX_SEALED_BYTES = REMOTE_EVIDENCE_HARD_CAP_BYTES - 1
REMOTE_READ_MAX_ATTEMPTS = 4
REMOTE_READ_RETRYABLE_HTTP_STATUSES = (429, 503)
REMOTE_READ_BACKOFF_SECONDS = (0.25, 0.5, 1.0)
HTTP_ERROR_BODY_LIMIT_BYTES = 64 * 1024
SCHEDULER_JSON_HARD_CAP_BYTES = 64 * 1024**2
SCHEDULER_POST_RESPONSE_HARD_CAP_BYTES = 1024**2
SUPERSEDED_V2_PACKAGE_SHA256 = (
    "47a4bff263905b9d1de9d42096c332b3866b0e0369d41962724ab62127b1ddcf"
)
PBD6_RUNTIME_SHA256 = {
    "artifacts/code/tools/tier1_corrected_current7_slurm_seed_runner.py": (
        "44fddbc60874fe3c2e14d31dc4f7b0b40cb3bc046c8006b0cccf24b323b59d36"
    ),
    "artifacts/code/tools/tier1_final1000_multiseed_contract.py": (
        "cd83500be3851884b405746fe33ab9595804a2bc50e20e7351e5d770843e6de9"
    ),
    "artifacts/code/tools/tier1_final1000_multiseed_phase_b_contract.py": (
        "1313077bfeb44dfff808b645d3873175202001c4e216065c71cc632b74595f6f"
    ),
    "artifacts/code/tools/tier1_final1000_multiseed_phase_b_runner.py": (
        "638cf11e5fb553c736f01dfd6393b3fa69a83de04b3118ec012e6c4bcf249bb0"
    ),
}


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"immutable output already exists: {path}")
    staged = path.with_name(path.name + f".part.{os.getpid()}")
    try:
        staged.write_text(
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(staged, path)
    finally:
        if staged.exists():
            staged.unlink()


def _remote_read_policy() -> dict[str, Any]:
    unsigned = {
        "schema_version": "mft-tier1-resource2-remote-read-policy-v1",
        "hard_cap_bytes": REMOTE_EVIDENCE_HARD_CAP_BYTES,
        "maximum_sealed_size_bytes": REMOTE_EVIDENCE_MAX_SEALED_BYTES,
        "expected_size_plus_one_required": True,
        "ambiguous_hard_cap_rejected": True,
        "maximum_attempts": REMOTE_READ_MAX_ATTEMPTS,
        "retryable_http_statuses": list(REMOTE_READ_RETRYABLE_HTTP_STATUSES),
        "increasing_backoff_seconds": list(REMOTE_READ_BACKOFF_SECONDS),
        "error_body_limit_bytes": HTTP_ERROR_BODY_LIMIT_BYTES,
    }
    return {**unsigned, "sha256": canonical_sha256(unsigned)}


def _scheduler_client_policy() -> dict[str, Any]:
    unsigned = {
        "schema_version": "mft-tier1-resource2-additive-client-policy-v1",
        "client_class": "Resource2AdditiveSchedulerClient",
        "allowed_name_prefix": CANARY_PREFIX,
        "allowed_dedupe_prefix": CANARY_DEDUPE_PREFIX,
        "exact_expected_task_required": True,
        "maximum_post_attempts": 1,
        "post_retry_allowed": False,
        "allocation_pin_allowed": False,
        "cancel_surface_available": False,
        "preempt_surface_available": False,
        "get_retry_maximum_attempts": REMOTE_READ_MAX_ATTEMPTS,
        "get_retryable_http_statuses": list(REMOTE_READ_RETRYABLE_HTTP_STATUSES),
        "get_increasing_backoff_seconds": list(REMOTE_READ_BACKOFF_SECONDS),
        "error_body_limit_bytes": HTTP_ERROR_BODY_LIMIT_BYTES,
        "scheduler_json_hard_cap_bytes": SCHEDULER_JSON_HARD_CAP_BYTES,
        "post_response_hard_cap_bytes": SCHEDULER_POST_RESPONSE_HARD_CAP_BYTES,
    }
    return {**unsigned, "sha256": canonical_sha256(unsigned)}


def _bounded_http_error_detail(exc: urllib.error.HTTPError) -> str:
    raw = exc.read(HTTP_ERROR_BODY_LIMIT_BYTES + 1)
    truncated = len(raw) > HTTP_ERROR_BODY_LIMIT_BYTES
    detail = raw[:HTTP_ERROR_BODY_LIMIT_BYTES].decode("utf-8", errors="replace")
    return detail + ("...[truncated]" if truncated else "")


def _remote_file_bytes_bounded(
    *,
    scheduler_url: str,
    task_id: int,
    relative_path: str,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    attempt_audit: list[dict[str, Any]] | None = None,
) -> bytes:
    if (
        not relative_path
        or relative_path.startswith("/")
        or ".." in PurePosixPath(relative_path).parts
        or isinstance(task_id, bool)
        or not isinstance(task_id, int)
        or task_id <= 0
        or isinstance(expected_size, bool)
        or (
            expected_size is not None
            and (not isinstance(expected_size, int) or expected_size < 0)
        )
        or (expected_sha256 is not None and not _is_sha256(expected_sha256))
    ):
        raise RuntimeError("unsafe or unsealed Scheduler remote evidence request")
    if expected_size is not None and expected_size > REMOTE_EVIDENCE_MAX_SEALED_BYTES:
        raise RuntimeError(
            "Scheduler sealed remote evidence reaches or exceeds the 16 MiB hard cap"
        )
    request_limit = (
        expected_size + 1
        if expected_size is not None
        else REMOTE_EVIDENCE_HARD_CAP_BYTES
    )
    query = urllib.parse.urlencode(
        {
            "path": relative_path,
            "base": "remote_cwd",
            "max_bytes": request_limit,
        }
    )
    url = scheduler_url.rstrip("/") + f"/api/tasks/{int(task_id)}/remote-file?{query}"
    attempts = 0
    for attempt in range(REMOTE_READ_MAX_ATTEMPTS):
        attempts += 1
        request = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=60.0) as response:
                payload = response.read(request_limit)
        except urllib.error.HTTPError as exc:
            detail = _bounded_http_error_detail(exc)
            if (
                exc.code in REMOTE_READ_RETRYABLE_HTTP_STATUSES
                and attempt + 1 < REMOTE_READ_MAX_ATTEMPTS
            ):
                time.sleep(REMOTE_READ_BACKOFF_SECONDS[attempt])
                continue
            raise RuntimeError(
                "bounded Scheduler remote evidence GET failed for "
                f"{relative_path} after {attempts}/{REMOTE_READ_MAX_ATTEMPTS} "
                f"attempts with HTTP {exc.code}: {detail}"
            ) from exc
        break
    else:  # pragma: no cover - bounded loop either succeeds or raises
        raise AssertionError("bounded Scheduler remote evidence retry fell through")
    if expected_size is None and len(payload) == REMOTE_EVIDENCE_HARD_CAP_BYTES:
        raise RuntimeError("Scheduler remote evidence reached an ambiguous hard bound")
    if expected_size is not None and len(payload) != expected_size:
        raise RuntimeError("Scheduler remote evidence length differs from its receipt")
    if (
        expected_sha256 is not None
        and hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise RuntimeError("Scheduler remote evidence SHA differs from its receipt")
    if attempt_audit is not None:
        attempt_audit.append(
            {
                "relative_path": relative_path,
                "expected_size": expected_size,
                "request_max_bytes": request_limit,
                "read_limit_bytes": request_limit,
                "attempt_count": attempts,
                "retry_count": attempts - 1,
            }
        )
    return payload


class Resource2AdditiveSchedulerClient:
    """GET-only inventory/detail client plus one exact resource2 POST.

    The class intentionally has no generic method selector and no allocation,
    cancellation, or preemption method.  A mutating request is possible only
    after binding the client to one immutable candidate task.
    """

    __slots__ = (
        "base_url",
        "timeout",
        "_expected_task",
        "get_attempt_count",
        "get_retry_count",
        "post_count",
    )

    def __init__(
        self,
        base_url: str,
        *,
        expected_task: Mapping[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._expected_task = (
            copy.deepcopy(dict(expected_task)) if expected_task is not None else None
        )
        self.get_attempt_count = 0
        self.get_retry_count = 0
        self.post_count = 0
        if self._expected_task is not None:
            self._validate_expected_task(self._expected_task)

    @staticmethod
    def _validate_expected_task(task: Mapping[str, Any]) -> None:
        payload = task.get("payload_json")
        seed = payload.get("seed") if isinstance(payload, Mapping) else None
        dedupe = str(task.get("dedupe_key") or "")
        if (
            set(task) != REQUIRED_SCHEDULER_FIELDS
            or "requested_allocation_id" in task
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or task.get("name") != f"{CANARY_PREFIX}entry-{seed}"
            or not dedupe.startswith(CANARY_DEDUPE_PREFIX)
            or not _is_sha256(dedupe.removeprefix(CANARY_DEDUPE_PREFIX))
        ):
            raise RuntimeError(
                "resource2 additive client candidate escaped its exact namespace"
            )

    @staticmethod
    def _decode_json(raw: bytes, *, label: str) -> Any:
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"{label} is not bounded valid JSON") from exc

    def _get_json(self, path: str, *, allow_not_found: bool = False) -> Any:
        parsed = urllib.parse.urlsplit(path)
        is_task_list = parsed.path == "/api/tasks"
        is_task_detail = (
            re.fullmatch(r"/api/tasks/[1-9][0-9]*", parsed.path) is not None
        )
        if (
            parsed.scheme
            or parsed.netloc
            or parsed.fragment
            or not (is_task_list or is_task_detail)
            or (is_task_detail and bool(parsed.query))
        ):
            raise RuntimeError("resource2 additive client refused a foreign GET path")
        attempts = 0
        for attempt in range(REMOTE_READ_MAX_ATTEMPTS):
            attempts += 1
            self.get_attempt_count += 1
            request = urllib.request.Request(self.base_url + path, method="GET")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read(SCHEDULER_JSON_HARD_CAP_BYTES)
            except urllib.error.HTTPError as exc:
                detail = _bounded_http_error_detail(exc)
                if exc.code == 404 and allow_not_found:
                    return None
                if (
                    exc.code in REMOTE_READ_RETRYABLE_HTTP_STATUSES
                    and attempt + 1 < REMOTE_READ_MAX_ATTEMPTS
                ):
                    self.get_retry_count += 1
                    time.sleep(REMOTE_READ_BACKOFF_SECONDS[attempt])
                    continue
                raise RuntimeError(
                    "resource2 Scheduler GET failed after "
                    f"{attempts}/{REMOTE_READ_MAX_ATTEMPTS} attempts with "
                    f"HTTP {exc.code}: {detail}"
                ) from exc
            if len(raw) == SCHEDULER_JSON_HARD_CAP_BYTES:
                raise RuntimeError("resource2 Scheduler GET reached an ambiguous cap")
            return self._decode_json(raw, label="resource2 Scheduler GET response")
        raise AssertionError("bounded resource2 Scheduler GET retry fell through")

    def read_inventory_path(self, path: str) -> Any:
        parsed = urllib.parse.urlsplit(path)
        if parsed.path != "/api/tasks" or not parsed.query:
            raise RuntimeError("resource2 inventory path is not a task-list GET")
        return self._get_json(path)

    def list_complete_namespace_tasks(self) -> list[dict[str, Any]]:
        return [
            *_paged_namespace(self, prefix=PRODUCTION_PREFIX),
            *_paged_namespace(self, prefix=CANARY_PREFIX),
        ]

    def get_task(self, task_id: int) -> Mapping[str, Any] | None:
        if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
            raise RuntimeError("resource2 task detail id is invalid")
        value = self._get_json(f"/api/tasks/{task_id}", allow_not_found=True)
        if value is None:
            return None
        if not isinstance(value, dict):
            raise RuntimeError("resource2 task detail response is not an object")
        return value

    def submit_task(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.post_count != 0:
            raise RuntimeError(
                "resource2 additive client already attempted its one POST"
            )
        if self._expected_task is None:
            raise RuntimeError("resource2 additive client has no sealed expected task")
        self._validate_expected_task(payload)
        if dict(payload) != self._expected_task:
            raise RuntimeError("resource2 POST differs from its exact sealed candidate")
        body = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/api/tasks",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        self.post_count += 1
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read(SCHEDULER_POST_RESPONSE_HARD_CAP_BYTES)
        except urllib.error.HTTPError as exc:
            detail = _bounded_http_error_detail(exc)
            raise RuntimeError(
                "resource2 one-shot Scheduler POST failed without retry with "
                f"HTTP {exc.code}: {detail}"
            ) from exc
        if len(raw) == SCHEDULER_POST_RESPONSE_HARD_CAP_BYTES:
            raise RuntimeError(
                "resource2 Scheduler POST response reached an ambiguous cap"
            )
        value = self._decode_json(raw, label="resource2 Scheduler POST response")
        if not isinstance(value, dict):
            raise RuntimeError("resource2 Scheduler POST response is not an object")
        return value


def _thread_environment() -> dict[str, str]:
    return {name: str(CANDIDATE_CPUS) for name in THREAD_VARIABLES}


def validate_config(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "config_sha256"}
    required = {
        "schema_version",
        "diagnostic_id",
        "stage_id",
        "seed",
        "expected_offload_plan_file_sha256",
        "expected_bundle_id",
        "expected_bundle_manifest_sha256",
        "expected_bundle_contract_sha256",
        "baseline_task_id",
        "baseline_seed",
        "baseline_package_sha256",
        "baseline_submission_receipt_sha256",
        "baseline_terminal_attestation_required",
        "baseline_cpus",
        "candidate_cpus",
        "candidate_memory_mb",
        "baseline_theoretical_slots",
        "candidate_theoretical_slots",
        "minimum_single_seed_throughput_ratio",
        "minimum_slot_weighted_throughput_ratio",
        "minimum_core_use_ratio_vs_baseline",
        "maximum_peak_rss_bytes",
        "thread_environment",
        "exact_inherited_cpuset_required",
        "finite_cgroup_ancestor_required",
        "priority",
        "scheduling_profile",
        "gpus",
        "max_workers_per_node",
        "timeout_seconds",
        "task_name_prefix",
        "dedupe_prefix",
        "production_name_prefix",
        "production_dedupe_prefix",
        "controller_harvester_isolated",
        "complete_namespace_pre_and_post_scan_required",
        "one_additive_post_maximum",
        "fea_submission_performed",
        "aedt_used",
        "config_sha256",
    }
    stage = BY_ID[ENTRY_STAGE_ID]
    seed = value.get("seed")
    if (
        set(value) != required
        or value.get("schema_version") != CONFIG_SCHEMA
        or not DIAGNOSTIC_ID_PATTERN.fullmatch(str(value.get("diagnostic_id") or ""))
        or value.get("stage_id") != ENTRY_STAGE_ID
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or not stage.seed_start <= seed < stage.seed_window_end_exclusive
        or seed == BASELINE_SEED
        or not _is_sha256(value.get("expected_offload_plan_file_sha256"))
        or not str(value.get("expected_bundle_id") or "").startswith("current7-")
        or not _is_sha256(value.get("expected_bundle_manifest_sha256"))
        or not _is_sha256(value.get("expected_bundle_contract_sha256"))
        or value.get("baseline_task_id") != BASELINE_TASK_ID
        or value.get("baseline_seed") != BASELINE_SEED
        or value.get("baseline_package_sha256") != BASELINE_PACKAGE_SHA256
        or value.get("baseline_submission_receipt_sha256") != BASELINE_SUBMISSION_SHA256
        or value.get("baseline_terminal_attestation_required") is not True
        or value.get("baseline_cpus") != BASELINE_CPUS
        or value.get("candidate_cpus") != CANDIDATE_CPUS
        or value.get("candidate_memory_mb") != CANDIDATE_MEMORY_MB
        or value.get("baseline_theoretical_slots") != BASELINE_THEORETICAL_SLOTS
        or value.get("candidate_theoretical_slots") != CANDIDATE_THEORETICAL_SLOTS
        or value.get("minimum_single_seed_throughput_ratio")
        != MIN_SINGLE_SEED_THROUGHPUT_RATIO
        or value.get("minimum_slot_weighted_throughput_ratio")
        != MIN_SLOT_WEIGHTED_THROUGHPUT_RATIO
        or value.get("minimum_core_use_ratio_vs_baseline") != MIN_CORE_USE_RATIO
        or value.get("maximum_peak_rss_bytes") != DEFAULT_PEAK_RSS_GATE_BYTES
        or value.get("thread_environment") != _thread_environment()
        or value.get("exact_inherited_cpuset_required") is not True
        or value.get("finite_cgroup_ancestor_required") is not True
        or value.get("priority") != DEFAULT_PRIORITY
        or value.get("scheduling_profile") != "standard"
        or value.get("gpus") != 0
        or value.get("max_workers_per_node") != DEFAULT_MAX_WORKERS_PER_NODE
        or value.get("timeout_seconds") != DEFAULT_TIMEOUT_SECONDS
        or value.get("task_name_prefix") != CANARY_PREFIX
        or value.get("dedupe_prefix") != CANARY_DEDUPE_PREFIX
        or value.get("production_name_prefix") != PRODUCTION_PREFIX
        or value.get("production_dedupe_prefix") != PRODUCTION_DEDUPE_PREFIX
        or value.get("controller_harvester_isolated") is not True
        or value.get("complete_namespace_pre_and_post_scan_required") is not True
        or value.get("one_additive_post_maximum") != 1
        or value.get("fea_submission_performed") is not False
        or value.get("aedt_used") is not False
        or value.get("config_sha256") != canonical_sha256(unsigned)
    ):
        raise RuntimeError("Final1000 resource2 canary config seal mismatch")
    return copy.deepcopy(dict(value))


def _validate_baseline_package_and_submission(
    *,
    config: Mapping[str, Any],
    package_path: Path,
    submission_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    package = shape1.validate_package(_read_json(package_path, "baseline package"))
    submission = shape1._validate_submission_receipt(
        _read_json(submission_path, "baseline submission receipt"), package=package
    )
    if (
        package.get("package_sha256") != config["baseline_package_sha256"]
        or submission.get("receipt_sha256")
        != config["baseline_submission_receipt_sha256"]
        or submission.get("task_id") != config["baseline_task_id"]
        or package.get("seed") != config["baseline_seed"]
        or package["parent_task"].get("cpus") != config["baseline_cpus"]
        or package["parent_task"].get("memory_mb") != CANDIDATE_MEMORY_MB
    ):
        raise RuntimeError("pinned 4-CPU baseline package/submission lineage mismatch")
    return package, submission


def validate_baseline_terminal(
    value: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    baseline_package: Mapping[str, Any],
    baseline_submission: Mapping[str, Any],
) -> dict[str, Any]:
    unsigned = {
        key: item for key, item in value.items() if key != "remote_terminal_sha256"
    }
    required = {
        "schema_version",
        "observed_at",
        "package_sha256",
        "submission_receipt_sha256",
        "task_id",
        "scheduler_status",
        "scheduler_detail_get_count",
        "scheduler_remote_file_get_count",
        "remote_files",
        "batch_manifest_sha256",
        "task_status_sha256",
        "child_receipt_sha256",
        "terminal_cpu_evidence",
        "terminal_cpu_evidence_sha256",
        "promotion_eligible",
        "shape4_submission_allowed",
        "scheduler_access",
        "scheduler_post_count",
        "scheduler_cancel_count",
        "scheduler_preempt_count",
        "remote_access",
        "remote_write_count",
        "fea_submission_performed",
        "aedt_used",
        "remote_terminal_sha256",
    }
    metrics_raw = value.get("terminal_cpu_evidence")
    if not isinstance(metrics_raw, Mapping):
        raise RuntimeError("baseline terminal CPU evidence is absent")
    metrics = shape1.validate_terminal_cpu_evidence(metrics_raw)
    per_seed = metrics.get("per_seed_average_cores")
    child = per_seed[0] if isinstance(per_seed, list) and len(per_seed) == 1 else {}
    result_record = (value.get("remote_files") or {}).get("result")
    if (
        set(value) != required
        or value.get("schema_version") != shape1.REMOTE_TERMINAL_SCHEMA
        or value.get("remote_terminal_sha256") != canonical_sha256(unsigned)
        or value.get("package_sha256") != baseline_package["package_sha256"]
        or value.get("package_sha256") != config["baseline_package_sha256"]
        or value.get("submission_receipt_sha256")
        != baseline_submission["receipt_sha256"]
        or value.get("submission_receipt_sha256")
        != config["baseline_submission_receipt_sha256"]
        or value.get("task_id") != config["baseline_task_id"]
        or value.get("scheduler_status") != "completed"
        or value.get("promotion_eligible") is not True
        or value.get("shape4_submission_allowed") is not True
        or value.get("scheduler_access") != "GET-only"
        or value.get("scheduler_post_count") != 0
        or value.get("scheduler_cancel_count") != 0
        or value.get("scheduler_preempt_count") != 0
        or value.get("remote_access") != "read-only"
        or value.get("remote_write_count") != 0
        or value.get("fea_submission_performed") is not False
        or value.get("aedt_used") is not False
        or value.get("terminal_cpu_evidence_sha256")
        != metrics["terminal_cpu_evidence_sha256"]
        or metrics.get("expected_shape") != 1
        or metrics.get("parent_requested_cpus") != config["baseline_cpus"]
        or metrics.get("logical_seed_count") != 1
        or metrics.get("promotion_eligible") is not True
        or child.get("seed") != config["baseline_seed"]
        or not isinstance(child.get("cpu_set"), list)
        or len(child["cpu_set"]) != config["baseline_cpus"]
        or len(set(child["cpu_set"])) != config["baseline_cpus"]
        or not isinstance(result_record, Mapping)
        or not _is_sha256(result_record.get("sha256"))
        or isinstance(result_record.get("size"), bool)
        or not isinstance(result_record.get("size"), int)
        or result_record.get("size") <= 0
    ):
        raise RuntimeError(
            "4-CPU baseline is not an eligible GET-authenticated terminal"
        )
    for field in (
        "wall_time_seconds",
        "process_tree_cpu_seconds",
        "average_cores_used",
    ):
        raw = child.get(field)
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
            or float(raw) <= 0
        ):
            raise RuntimeError(f"baseline terminal {field} is not finite positive")
    return copy.deepcopy(dict(value))


def _baseline_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    metrics = value["terminal_cpu_evidence"]
    child = metrics["per_seed_average_cores"][0]
    return {
        "task_id": value["task_id"],
        "seed": child["seed"],
        "package_sha256": value["package_sha256"],
        "submission_receipt_sha256": value["submission_receipt_sha256"],
        "remote_terminal_sha256": value["remote_terminal_sha256"],
        "terminal_cpu_evidence_sha256": value["terminal_cpu_evidence_sha256"],
        "requested_cpus": metrics["parent_requested_cpus"],
        "wall_time_seconds": float(child["wall_time_seconds"]),
        "process_tree_cpu_seconds": float(child["process_tree_cpu_seconds"]),
        "average_cores_used": float(child["average_cores_used"]),
        "throughput_seeds_per_second": 1.0 / float(child["wall_time_seconds"]),
        "promotion_eligible": True,
    }


def _validate_remote_read_attempts(
    value: Any, *, verified_files: Mapping[str, Any], expected_total_attempts: Any
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise RuntimeError("remote evidence attempt audit is not a list")
    by_path: dict[str, Mapping[str, Any]] = {}
    for raw in verified_files.values():
        if isinstance(raw, Mapping) and isinstance(raw.get("size"), int):
            by_path[str(raw.get("relative_path") or "")] = raw
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in value:
        if not isinstance(raw, Mapping):
            raise RuntimeError("remote evidence attempt audit contains a non-object")
        path = str(raw.get("relative_path") or "")
        expected = by_path.get(path)
        attempt_count = raw.get("attempt_count")
        retry_count = raw.get("retry_count")
        size = raw.get("expected_size")
        if (
            set(raw)
            != {
                "relative_path",
                "expected_size",
                "request_max_bytes",
                "read_limit_bytes",
                "attempt_count",
                "retry_count",
            }
            or expected is None
            or path in seen
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size != expected.get("size")
            or size > REMOTE_EVIDENCE_MAX_SEALED_BYTES
            or raw.get("request_max_bytes") != size + 1
            or raw.get("read_limit_bytes") != size + 1
            or isinstance(attempt_count, bool)
            or not isinstance(attempt_count, int)
            or not 1 <= attempt_count <= REMOTE_READ_MAX_ATTEMPTS
            or retry_count != attempt_count - 1
        ):
            raise RuntimeError("remote evidence attempt audit drifted")
        seen.add(path)
        result.append(dict(raw))
    expected_paths = {
        path for path, record in by_path.items() if record.get("sha256") is not None
    }
    if (
        seen != expected_paths
        or isinstance(expected_total_attempts, bool)
        or not isinstance(expected_total_attempts, int)
        or expected_total_attempts != sum(int(item["attempt_count"]) for item in result)
    ):
        raise RuntimeError("remote evidence attempt totals/path coverage drifted")
    return result


def _reverify_baseline_remote(
    *,
    config: Mapping[str, Any],
    baseline_package: Mapping[str, Any],
    baseline_terminal: Mapping[str, Any],
    scheduler_url: str,
    scheduler: Any | None = None,
    remote_file_reader: Any | None = None,
) -> dict[str, Any]:
    scheduler = scheduler or Resource2AdditiveSchedulerClient(scheduler_url)
    reader = remote_file_reader or _remote_file_bytes_bounded
    task_id, status = _authenticate_task(
        scheduler.get_task(BASELINE_TASK_ID),
        baseline_package["parent_task"],
        label="render baseline Scheduler detail",
    )
    if task_id != config["baseline_task_id"] or status != "completed":
        raise RuntimeError("render baseline is not one completed exact Scheduler task")
    records = baseline_terminal.get("remote_files")
    if not isinstance(records, Mapping) or not records:
        raise RuntimeError("baseline remote terminal has no remote file inventory")
    verified: dict[str, dict[str, Any]] = {}
    read_attempts: list[dict[str, Any]] = []
    get_count = 0
    for label, raw_record in sorted(records.items()):
        if not isinstance(raw_record, Mapping):
            raise RuntimeError("baseline remote file record is not an object")
        path = str(raw_record.get("relative_path") or "")
        expected_sha = raw_record.get("sha256")
        expected_size = raw_record.get("size")
        if (
            not path.startswith(f"runs/task-{BASELINE_TASK_ID}/")
            or path.startswith("/")
            or ".." in PurePosixPath(path).parts
        ):
            raise RuntimeError("baseline remote evidence path escaped its task root")
        if expected_sha is None and expected_size is None:
            verified[str(label)] = {
                "relative_path": path,
                "size": None,
                "sha256": None,
                "verified": True,
            }
            continue
        if (
            not _is_sha256(expected_sha)
            or isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
        ):
            raise RuntimeError("baseline remote file record is not sealed")
        payload = reader(
            scheduler_url=scheduler_url,
            task_id=BASELINE_TASK_ID,
            relative_path=path,
            **(
                {
                    "expected_size": expected_size,
                    "expected_sha256": expected_sha,
                    "attempt_audit": read_attempts,
                }
                if remote_file_reader is None
                else {}
            ),
        )
        get_count += 1
        if (
            len(payload) != expected_size
            or hashlib.sha256(payload).hexdigest() != expected_sha
        ):
            raise RuntimeError(f"baseline remote file changed: {label}")
        if remote_file_reader is not None:
            read_attempts.append(
                {
                    "relative_path": path,
                    "expected_size": expected_size,
                    "request_max_bytes": expected_size + 1,
                    "read_limit_bytes": expected_size + 1,
                    "attempt_count": 1,
                    "retry_count": 0,
                }
            )
        verified[str(label)] = {
            "relative_path": path,
            "size": expected_size,
            "sha256": expected_sha,
            "verified": True,
        }
    final_id, final_status = _authenticate_task(
        scheduler.get_task(BASELINE_TASK_ID),
        baseline_package["parent_task"],
        label="post-file render baseline Scheduler detail",
    )
    if final_id != task_id or final_status != status:
        raise RuntimeError("baseline Scheduler identity changed during render GET")
    unsigned = {
        "task_id": task_id,
        "scheduler_status": status,
        "package_sha256": baseline_terminal["package_sha256"],
        "submission_receipt_sha256": baseline_terminal["submission_receipt_sha256"],
        "remote_terminal_sha256": baseline_terminal["remote_terminal_sha256"],
        "scheduler_detail_get_count": 2,
        "scheduler_remote_file_get_count": get_count,
        "scheduler_remote_file_attempt_count": sum(
            int(item["attempt_count"]) for item in read_attempts
        ),
        "remote_read_policy": _remote_read_policy(),
        "remote_file_read_attempts": read_attempts,
        "verified_remote_files": verified,
        "scheduler_access": "GET-only",
        "scheduler_post_count": 0,
        "scheduler_cancel_count": 0,
        "scheduler_preempt_count": 0,
        "remote_write_count": 0,
        "reverified": True,
    }
    return {**unsigned, "sha256": canonical_sha256(unsigned)}


_INLINE_WRAPPER = r"""import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone

SCHEMA = "mft-tier1-final1000-resource2-canary-telemetry-v1"
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

def now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()

def canonical(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name("." + path.name + "." + str(os.getpid()) + ".tmp")
    try:
        staged.write_bytes(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")
        os.replace(staged, path)
    finally:
        if staged.exists():
            staged.unlink()

def read_int(path, allow_zero=False):
    raw = path.read_text(encoding="ascii").strip()
    if raw == "max":
        return None, raw
    value = int(raw)
    if value < 0 or (value == 0 and not allow_zero):
        raise RuntimeError("invalid cgroup memory value: " + str(path))
    return value, raw

def cgroup_snapshot():
    lines = Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
    matches = [line.split("::", 1)[1] for line in lines if line.startswith("0::")]
    if len(matches) != 1:
        raise RuntimeError("unique cgroup-v2 membership is unavailable")
    root = Path("/sys/fs/cgroup").resolve(strict=True)
    leaf = (root / matches[0].lstrip("/")).resolve(strict=True)
    if root != leaf and root not in leaf.parents:
        raise RuntimeError("cgroup membership escaped /sys/fs/cgroup")
    records = []
    selected = None
    current = leaf
    depth = 0
    while True:
        max_path = current / "memory.max"
        current_path = current / "memory.current"
        peak_path = current / "memory.peak"
        if max_path.is_file() and current_path.is_file() and peak_path.is_file():
            limit, raw_limit = read_int(max_path)
            current_bytes, _ = read_int(current_path, allow_zero=True)
            peak_bytes, _ = read_int(peak_path, allow_zero=True)
            record = {
                "depth_from_leaf": depth,
                "relative_path": "." if current == root else current.relative_to(root).as_posix(),
                "memory_max_raw": raw_limit,
                "memory_limit_bytes": limit,
                "memory_current_bytes": current_bytes,
                "memory_peak_bytes": peak_bytes,
            }
            records.append(record)
            if selected is None and limit is not None:
                selected = dict(record)
        if current == root:
            break
        current = current.parent
        depth += 1
    if not records or selected is None:
        raise RuntimeError("no finite cgroup-v2 memory.max ancestor exists")
    return {
        "leaf_relative_path": "." if leaf == root else leaf.relative_to(root).as_posix(),
        "leaf_memory_max_unbounded": records[0]["memory_limit_bytes"] is None,
        "ancestors": records,
        "selected_finite_ancestor": selected,
    }

payload_path = Path(sys.argv[1]).resolve(strict=True)
payload_root = Path(sys.argv[2]).resolve(strict=True)
payload_sha = sys.argv[3]
seed = int(sys.argv[4])
baseline_sha = sys.argv[5]
requested_memory_bytes = int(sys.argv[6])
rss_gate_bytes = int(sys.argv[7])
bundle = Path.cwd().resolve(strict=True)
task_id = str(os.environ.get("SLURM_SCHED_TASK_ID") or "")
if not task_id.isascii() or not task_id.isdigit() or int(task_id) <= 0:
    raise SystemExit("missing canonical SLURM_SCHED_TASK_ID")
task_root = bundle / "runs" / ("task-" + task_id)
telemetry_path = task_root / "resource2_canary_telemetry.json"
started_at = now()
failure = None
exit_code = 79
cpu_set_before = []
cpu_set_after = []
cgroup_before = None
cgroup_after = None
wall = None
cpu_seconds = None
peak_rss_bytes = None
status_record = {"relative_path": "runs/task-" + task_id + "/seed_status.json", "size": None, "sha256": None, "state": None, "terminal": None, "exit_code": None}
result_record = {"relative_path": "runs/task-" + task_id + "/seed-" + str(seed) + "/result.json", "size": None, "sha256": None}
try:
    if payload_path.name != "payload.json" or payload_root not in payload_path.parents:
        raise RuntimeError("scheduler payload escaped run root")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if canonical(payload) != payload_sha:
        raise RuntimeError("scheduler payload canonical SHA mismatch")
    cpu_set_before = sorted(os.sched_getaffinity(0))
    if len(cpu_set_before) != 2 or len(set(cpu_set_before)) != 2:
        raise RuntimeError("inherited Scheduler cpuset is not exactly two CPUs")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "2":
        raise RuntimeError("SLURM_CPUS_PER_TASK is not exactly two")
    if any(os.environ.get(name) != "2" for name in THREADS):
        raise RuntimeError("inference thread environment is not exactly two")
    cgroup_before = cgroup_snapshot()
    selected_before = cgroup_before["selected_finite_ancestor"]
    if int(selected_before["memory_limit_bytes"]) < requested_memory_bytes:
        raise RuntimeError("finite cgroup ancestor does not cover requested memory")
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    wall_started = time.monotonic()
    command = [
        sys.executable, "-u",
        str(bundle / "artifacts/code/tools/tier1_corrected_current7_slurm_seed_runner.py"),
        "--bundle-root", str(bundle),
        "--payload", str(payload_path),
        "--payload-root", str(payload_root),
        "--payload-sha256", payload_sha,
    ]
    completed = subprocess.run(command, cwd=bundle / "artifacts/code", check=False)
    wall = max(0.0, time.monotonic() - wall_started)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_seconds = max(0.0, (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime))
    peak_rss_bytes = int(after.ru_maxrss) * 1024
    cpu_set_after = sorted(os.sched_getaffinity(0))
    cgroup_after = cgroup_snapshot()
    selected_after = cgroup_after["selected_finite_ancestor"]
    if selected_after["relative_path"] != selected_before["relative_path"] or selected_after["memory_limit_bytes"] != selected_before["memory_limit_bytes"]:
        raise RuntimeError("finite cgroup memory ancestor changed during execution")
    if int(selected_after["memory_peak_bytes"]) > int(selected_after["memory_limit_bytes"]):
        raise RuntimeError("cgroup memory peak exceeds finite limit")
    status_path = task_root / "seed_status.json"
    if status_path.is_file():
        raw = status_path.read_bytes()
        status = json.loads(raw.decode("utf-8"))
        status_record = {"relative_path": status_record["relative_path"], "size": len(raw), "sha256": sha_bytes(raw), "state": status.get("state"), "terminal": status.get("terminal"), "exit_code": status.get("exit_code")}
    result_path = task_root / ("seed-" + str(seed)) / "result.json"
    if result_path.is_file():
        raw = result_path.read_bytes()
        result_record = {"relative_path": result_record["relative_path"], "size": len(raw), "sha256": sha_bytes(raw)}
    exit_code = int(completed.returncode)
    if exit_code != 0 or status_record["state"] != "completed" or status_record["terminal"] is not True or status_record["exit_code"] != 0 or result_record["sha256"] is None:
        failure = "single-seed runner did not produce one completed sealed result"
        exit_code = 78 if exit_code == 0 else exit_code
    elif cpu_set_after != cpu_set_before:
        failure = "wrapper affinity changed during execution"
        exit_code = 78
    elif peak_rss_bytes <= 0 or peak_rss_bytes > rss_gate_bytes:
        failure = "per-seed peak RSS is outside its gate"
        exit_code = 78
except Exception as exc:
    failure = type(exc).__name__ + ":" + str(exc)
    exit_code = 79

capacity = wall * 2 if wall is not None else None
average_cores = cpu_seconds / wall if cpu_seconds is not None and wall and wall > 0 else None
utilization = cpu_seconds / capacity if cpu_seconds is not None and capacity and capacity > 0 else None
unsigned = {
    "schema_version": SCHEMA,
    "started_at": started_at,
    "finished_at": now(),
    "task_id": int(task_id),
    "seed": seed,
    "payload_sha256": payload_sha,
    "baseline_remote_terminal_sha256": baseline_sha,
    "requested_cpus": 2,
    "requested_memory_bytes": requested_memory_bytes,
    "rss_gate_bytes": rss_gate_bytes,
    "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
    "thread_environment": {name: os.environ.get(name) for name in THREADS},
    "cpu_set_before": cpu_set_before,
    "cpu_set_after": cpu_set_after,
    "wall_time_seconds": wall,
    "process_tree_cpu_seconds": cpu_seconds,
    "cpu_capacity_seconds": capacity,
    "average_cores_used": average_cores,
    "cpu_utilization_fraction": utilization,
    "peak_rss_bytes": peak_rss_bytes,
    "cgroup_before": cgroup_before,
    "cgroup_after": cgroup_after,
    "seed_status": status_record,
    "result": result_record,
    "exit_code": exit_code,
    "failure": failure,
    "fea_submission_performed": False,
    "aedt_used": False,
}
value = dict(unsigned)
value["telemetry_sha256"] = canonical(unsigned)
atomic_json(telemetry_path, value)
raise SystemExit(exit_code)
"""


def _resource2_command(payload_sha256: str, baseline_sha256: str) -> str:
    if not _is_sha256(payload_sha256) or not _is_sha256(baseline_sha256):
        raise RuntimeError("resource2 command SHA input is invalid")
    exports = "\n".join(f"export {name}={CANDIDATE_CPUS}" for name in THREAD_VARIABLES)
    return "\n".join(
        [
            "set -euo pipefail",
            'export PYTHONPATH="$PWD/artifacts/python-site:$PWD/artifacts/code'
            '${PYTHONPATH:+:$PYTHONPATH}"',
            exports,
            'payload_path="${SLURM_SCHEDULER_PAYLOAD_PATH:?scheduler payload path is missing}"',
            'case "$payload_path" in /*) ;; *) payload_path="$HOME/$payload_path" ;; esac',
            'payload_path=$(realpath -e -- "$payload_path")',
            'payload_root=$(realpath -e -- "$HOME/slurm_scheduler/runs")',
            'case "$payload_path" in "$payload_root"/*/payload.json) ;; '
            '*) echo "unsafe scheduler payload path: $payload_path" >&2; exit 66 ;; esac',
            'python -u - "$payload_path" "$payload_root" '
            f"{payload_sha256} {{seed}} {baseline_sha256} "
            f"{CANDIDATE_MEMORY_MB * 1024**2} {DEFAULT_PEAK_RSS_GATE_BYTES} <<'PY'",
            _INLINE_WRAPPER,
            "PY",
        ]
    )


def _transform_source_task(
    source_task: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    source = validate_task(
        source_task, expected_stage=BY_ID[ENTRY_STAGE_ID], expected_wave="refill"
    )
    payload = copy.deepcopy(source["payload_json"])
    payload["inference_threads"] = CANDIDATE_CPUS
    payload["scheduler_cpus"] = CANDIDATE_CPUS
    payload["resource_efficiency_canary"] = {
        "schema_version": CANARY_PAYLOAD_SCHEMA,
        "diagnostic_id": config["diagnostic_id"],
        "config_sha256": config["config_sha256"],
        "source_4cpu_task_sha256": canonical_sha256(source),
        "source_4cpu_dedupe_key": source["dedupe_key"],
        "baseline_task_id": baseline["task_id"],
        "baseline_remote_terminal_sha256": baseline["remote_terminal_sha256"],
        "baseline_terminal_cpu_evidence_sha256": baseline[
            "terminal_cpu_evidence_sha256"
        ],
        "requested_cpus": CANDIDATE_CPUS,
        "requested_memory_mb": CANDIDATE_MEMORY_MB,
        "thread_environment": _thread_environment(),
        "controller_harvester_isolated": True,
        "production_eligible": False,
        "automatic_promotion_allowed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    payload_sha = canonical_sha256(payload)
    command = _resource2_command(
        payload_sha, str(baseline["remote_terminal_sha256"])
    ).replace("{seed}", str(int(config["seed"])))
    resources = {
        "cpus": CANDIDATE_CPUS,
        "memory_mb": CANDIDATE_MEMORY_MB,
        "scheduling_profile": "standard",
        "aedt_backend": "standalone",
        "gpus": 0,
        "priority": DEFAULT_PRIORITY,
        "timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
        "max_workers_per_node": DEFAULT_MAX_WORKERS_PER_NODE,
    }
    identity = {
        "namespace": "final1000-resource2-canary-v1",
        "diagnostic_id": config["diagnostic_id"],
        "baseline_remote_terminal_sha256": baseline["remote_terminal_sha256"],
        "source_4cpu_task_sha256": canonical_sha256(source),
        "payload": payload,
        "resources": resources,
    }
    return {
        **source,
        "name": f"{CANARY_PREFIX}entry-{int(config['seed'])}",
        "command": command,
        "payload_json": payload,
        **resources,
        "dedupe_key": f"{CANARY_DEDUPE_PREFIX}{canonical_sha256(identity)}",
    }


def _validate_candidate_task(
    task: Mapping[str, Any],
    *,
    source_task: Mapping[str, Any],
    config: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    if set(task) != REQUIRED_SCHEDULER_FIELDS or "requested_allocation_id" in task:
        raise RuntimeError("resource2 Scheduler envelope fields drifted")
    expected = _transform_source_task(source_task, config=config, baseline=baseline)
    if dict(task) != expected:
        raise RuntimeError("resource2 candidate differs from its sealed transform")
    payload = task["payload_json"]
    canary = payload.get("resource_efficiency_canary") or {}
    if (
        task.get("cpus") != CANDIDATE_CPUS
        or task.get("memory_mb") != CANDIDATE_MEMORY_MB
        or payload.get("inference_threads") != CANDIDATE_CPUS
        or payload.get("scheduler_cpus") != CANDIDATE_CPUS
        or not str(task.get("name") or "").startswith(CANARY_PREFIX)
        or str(task.get("name") or "").startswith(PRODUCTION_PREFIX)
        or not str(task.get("dedupe_key") or "").startswith(CANARY_DEDUPE_PREFIX)
        or str(task.get("dedupe_key") or "").startswith(PRODUCTION_DEDUPE_PREFIX)
        or canary.get("controller_harvester_isolated") is not True
        or any(
            payload.get(field) is not False
            for field in (
                "production_eligible",
                "fea_submission_approved",
                "fea_submission_performed",
                "aedt_used",
                "automatic_promotion_allowed",
            )
        )
        or any(
            token in str(task.get("command") or "").lower()
            for token in ("ansysedt", "pyaedt.desktop")
        )
    ):
        raise RuntimeError("resource2 execution/isolation seal mismatch")
    return copy.deepcopy(dict(task))


def _terminal_evidence(remote_cwd: str, seed: int) -> dict[str, Any]:
    root = PurePosixPath(remote_cwd) / "runs" / "task-{scheduler_task_id}"
    return {
        "task_root": root.as_posix(),
        "telemetry_path": (root / TELEMETRY_FILENAME).as_posix(),
        "seed_status_path": (root / "seed_status.json").as_posix(),
        "result_path": (root / f"seed-{seed}" / "result.json").as_posix(),
        "scheduler_detail_required": True,
        "exact_cpuset_json_pointers": ["/cpu_set_before", "/cpu_set_after"],
        "thread_environment_json_pointer": "/thread_environment",
        "per_seed_wall_json_pointer": "/wall_time_seconds",
        "per_seed_cpu_json_pointer": "/process_tree_cpu_seconds",
        "per_seed_rss_json_pointer": "/peak_rss_bytes",
        "finite_cgroup_ancestor_json_pointer": (
            "/cgroup_after/selected_finite_ancestor"
        ),
        "terminal_evaluator": (
            "tools/tier1_final1000_resource2_canary.py::evaluate_terminal_evidence"
        ),
    }


def _terminal_gates() -> dict[str, Any]:
    return {
        "scheduler_terminal_state": "completed",
        "seed_status_state": "completed",
        "wrapper_exit_code": 0,
        "exact_inherited_and_final_cpuset_length": CANDIDATE_CPUS,
        "slurm_cpus_per_task": str(CANDIDATE_CPUS),
        "inference_thread_environment": _thread_environment(),
        "result_inference_threads": CANDIDATE_CPUS,
        "result_scheduler_cpus": CANDIDATE_CPUS,
        "maximum_peak_rss_bytes": DEFAULT_PEAK_RSS_GATE_BYTES,
        "finite_cgroup_memory_ancestor_required": True,
        "finite_cgroup_limit_covers_requested_memory": True,
        "cgroup_peak_within_finite_limit": True,
        "minimum_single_seed_throughput_ratio_vs_4cpu": (
            MIN_SINGLE_SEED_THROUGHPUT_RATIO
        ),
        "minimum_slot_weighted_throughput_ratio_vs_4cpu": (
            MIN_SLOT_WEIGHTED_THROUGHPUT_RATIO
        ),
        "minimum_core_use_ratio_vs_4cpu": MIN_CORE_USE_RATIO,
        "baseline_theoretical_slots": BASELINE_THEORETICAL_SLOTS,
        "candidate_theoretical_slots": CANDIDATE_THEORETICAL_SLOTS,
        "automatic_promotion_performed": False,
    }


def _runtime_hard_hashes(manifest: Mapping[str, Any]) -> dict[str, str]:
    files = manifest.get("files") or {}
    observed: dict[str, str] = {}
    for path, expected in PBD6_RUNTIME_SHA256.items():
        record = files.get(path)
        if (
            not isinstance(record, Mapping)
            or record.get("sha256") != expected
            or isinstance(record.get("size"), bool)
            or not isinstance(record.get("size"), int)
            or record.get("size") <= 0
        ):
            raise RuntimeError(f"pbd6 runtime hard hash drifted: {path}")
        observed[path] = str(record["sha256"])
    return observed


def _assemble_package(
    *,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    publication: Mapping[str, Any],
    baseline_terminal: Mapping[str, Any],
    baseline_package_file_sha256: str,
    baseline_submission_file_sha256: str,
    baseline_terminal_file_sha256: str,
    baseline_remote_reverification: Mapping[str, Any],
    source_task: Mapping[str, Any],
    candidate_task: Mapping[str, Any],
    offload_plan_file_sha256: str,
    publication_receipt_file_sha256: str,
    required_runtime_code_sha256: str,
    runtime_hard_hashes: Mapping[str, str],
) -> dict[str, Any]:
    config = validate_config(config)
    baseline = _baseline_summary(baseline_terminal)
    source = validate_task(
        source_task, expected_stage=BY_ID[ENTRY_STAGE_ID], expected_wave="refill"
    )
    candidate = _validate_candidate_task(
        candidate_task,
        source_task=source,
        config=config,
        baseline=baseline,
    )
    unsigned = {
        "schema_version": PACKAGE_SCHEMA,
        "diagnostic_id": config["diagnostic_id"],
        "config": config,
        "config_sha256": config["config_sha256"],
        "stage_id": ENTRY_STAGE_ID,
        "seed": config["seed"],
        "offload_plan_file_sha256": offload_plan_file_sha256,
        "bundle_id": plan["bundle_id"],
        "bundle_manifest_sha256": plan["bundle_manifest_sha256"],
        "bundle_contract_sha256": plan["contract_sha256"],
        "remote_bundle": plan["remote_bundle"],
        "publication_receipt_file_sha256": publication_receipt_file_sha256,
        "publication_receipt_sha256": publication["receipt_sha256"],
        "ready_sha256": publication["ready_sha256"],
        "required_runtime_code_sha256": required_runtime_code_sha256,
        "pbd6_runtime_hard_hashes": dict(runtime_hard_hashes),
        "manifest_publication_receipt_binding_validated": True,
        "baseline_package_file_sha256": baseline_package_file_sha256,
        "baseline_submission_file_sha256": baseline_submission_file_sha256,
        "baseline_terminal_file_sha256": baseline_terminal_file_sha256,
        "baseline": baseline,
        "baseline_terminal_gate_passed_at_render": True,
        "baseline_remote_reverification": copy.deepcopy(
            dict(baseline_remote_reverification)
        ),
        "baseline_remote_reverified_at_render": True,
        "source_4cpu_task": source,
        "source_4cpu_task_sha256": canonical_sha256(source),
        "source_4cpu_logical_dedupe_key": source["dedupe_key"],
        "candidate_task": candidate,
        "candidate_task_sha256": canonical_sha256(candidate),
        "candidate_dedupe_key": candidate["dedupe_key"],
        "terminal_evidence": _terminal_evidence(
            str(candidate["remote_cwd"]), int(config["seed"])
        ),
        "terminal_gates": _terminal_gates(),
        "remote_read_policy": _remote_read_policy(),
        "scheduler_client_policy": _scheduler_client_policy(),
        "superseded_launch_forbidden_package_sha256": SUPERSEDED_V2_PACKAGE_SHA256,
        "superseded_launch_forbidden_reason": (
            "production-client and remote-read policies were not sufficiently bounded"
        ),
        "complete_production_and_canary_namespaces_required": True,
        "post_submit_namespace_rescan_required": True,
        "remote_ready_reread_required": True,
        "scheduler_endpoint": "POST /api/tasks",
        "one_additive_post_maximum": 1,
        "explicit_apply_required": True,
        "scheduler_write_performed": False,
        "submission_performed": False,
        "cancellation_performed": False,
        "preemption_performed": False,
        "publication_performed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "production_promotion_allowed": False,
        "automatic_promotion_performed": False,
    }
    return {**unsigned, "package_sha256": canonical_sha256(unsigned)}


def validate_package(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "package_sha256"}
    required = {
        "schema_version",
        "diagnostic_id",
        "config",
        "config_sha256",
        "stage_id",
        "seed",
        "offload_plan_file_sha256",
        "bundle_id",
        "bundle_manifest_sha256",
        "bundle_contract_sha256",
        "remote_bundle",
        "publication_receipt_file_sha256",
        "publication_receipt_sha256",
        "ready_sha256",
        "required_runtime_code_sha256",
        "pbd6_runtime_hard_hashes",
        "manifest_publication_receipt_binding_validated",
        "baseline_package_file_sha256",
        "baseline_submission_file_sha256",
        "baseline_terminal_file_sha256",
        "baseline",
        "baseline_terminal_gate_passed_at_render",
        "baseline_remote_reverification",
        "baseline_remote_reverified_at_render",
        "source_4cpu_task",
        "source_4cpu_task_sha256",
        "source_4cpu_logical_dedupe_key",
        "candidate_task",
        "candidate_task_sha256",
        "candidate_dedupe_key",
        "terminal_evidence",
        "terminal_gates",
        "remote_read_policy",
        "scheduler_client_policy",
        "superseded_launch_forbidden_package_sha256",
        "superseded_launch_forbidden_reason",
        "complete_production_and_canary_namespaces_required",
        "post_submit_namespace_rescan_required",
        "remote_ready_reread_required",
        "scheduler_endpoint",
        "one_additive_post_maximum",
        "explicit_apply_required",
        "scheduler_write_performed",
        "submission_performed",
        "cancellation_performed",
        "preemption_performed",
        "publication_performed",
        "fea_submission_performed",
        "aedt_used",
        "production_promotion_allowed",
        "automatic_promotion_performed",
        "package_sha256",
    }
    config_raw = value.get("config")
    source_raw = value.get("source_4cpu_task")
    candidate_raw = value.get("candidate_task")
    baseline = value.get("baseline")
    baseline_reverification = value.get("baseline_remote_reverification")
    if not all(
        isinstance(item, Mapping)
        for item in (
            config_raw,
            source_raw,
            candidate_raw,
            baseline,
            baseline_reverification,
        )
    ):
        raise RuntimeError("resource2 package nested object is absent")
    config = validate_config(config_raw)  # type: ignore[arg-type]
    source = validate_task(
        source_raw,
        expected_stage=BY_ID[ENTRY_STAGE_ID],
        expected_wave="refill",  # type: ignore[arg-type]
    )
    candidate = _validate_candidate_task(
        candidate_raw,  # type: ignore[arg-type]
        source_task=source,
        config=config,
        baseline=baseline,  # type: ignore[arg-type]
    )
    verified_remote_files = baseline_reverification.get("verified_remote_files")
    if (
        not isinstance(verified_remote_files, Mapping)
        or baseline_reverification.get("remote_read_policy") != _remote_read_policy()
    ):
        raise RuntimeError("resource2 baseline remote-read policy seal mismatch")
    remote_attempts = _validate_remote_read_attempts(
        baseline_reverification.get("remote_file_read_attempts"),
        verified_files=verified_remote_files,
        expected_total_attempts=baseline_reverification.get(
            "scheduler_remote_file_attempt_count"
        ),
    )
    if baseline_reverification.get("scheduler_remote_file_get_count") != len(
        remote_attempts
    ):
        raise RuntimeError("resource2 baseline remote-file GET count drifted")
    baseline_required = {
        "task_id",
        "seed",
        "package_sha256",
        "submission_receipt_sha256",
        "remote_terminal_sha256",
        "terminal_cpu_evidence_sha256",
        "requested_cpus",
        "wall_time_seconds",
        "process_tree_cpu_seconds",
        "average_cores_used",
        "throughput_seeds_per_second",
        "promotion_eligible",
    }
    if (
        set(value) != required
        or value.get("schema_version") != PACKAGE_SCHEMA
        or value.get("package_sha256") != canonical_sha256(unsigned)
        or value.get("diagnostic_id") != config["diagnostic_id"]
        or value.get("config_sha256") != config["config_sha256"]
        or value.get("stage_id") != ENTRY_STAGE_ID
        or value.get("seed") != config["seed"]
        or any(
            not _is_sha256(value.get(field))
            for field in (
                "offload_plan_file_sha256",
                "bundle_manifest_sha256",
                "bundle_contract_sha256",
                "publication_receipt_file_sha256",
                "publication_receipt_sha256",
                "ready_sha256",
                "required_runtime_code_sha256",
                "baseline_package_file_sha256",
                "baseline_submission_file_sha256",
                "baseline_terminal_file_sha256",
                "source_4cpu_task_sha256",
                "candidate_task_sha256",
            )
        )
        or value.get("pbd6_runtime_hard_hashes") != PBD6_RUNTIME_SHA256
        or value.get("manifest_publication_receipt_binding_validated") is not True
        or set(baseline) != baseline_required  # type: ignore[arg-type]
        or baseline.get("task_id") != config["baseline_task_id"]  # type: ignore[union-attr]
        or baseline.get("seed") != config["baseline_seed"]  # type: ignore[union-attr]
        or baseline.get("package_sha256") != config["baseline_package_sha256"]  # type: ignore[union-attr]
        or baseline.get("submission_receipt_sha256")  # type: ignore[union-attr]
        != config["baseline_submission_receipt_sha256"]
        or baseline.get("requested_cpus") != BASELINE_CPUS  # type: ignore[union-attr]
        or baseline.get("promotion_eligible") is not True  # type: ignore[union-attr]
        or value.get("baseline_terminal_gate_passed_at_render") is not True
        or baseline_reverification.get("task_id") != BASELINE_TASK_ID  # type: ignore[union-attr]
        or baseline_reverification.get("scheduler_status") != "completed"  # type: ignore[union-attr]
        or baseline_reverification.get("remote_terminal_sha256")  # type: ignore[union-attr]
        != baseline.get("remote_terminal_sha256")  # type: ignore[union-attr]
        or baseline_reverification.get("reverified") is not True  # type: ignore[union-attr]
        or baseline_reverification.get("scheduler_access") != "GET-only"  # type: ignore[union-attr]
        or baseline_reverification.get("scheduler_post_count") != 0  # type: ignore[union-attr]
        or baseline_reverification.get("remote_write_count") != 0  # type: ignore[union-attr]
        or baseline_reverification.get("sha256")  # type: ignore[union-attr]
        != canonical_sha256(
            {
                key: item
                for key, item in baseline_reverification.items()  # type: ignore[union-attr]
                if key != "sha256"
            }
        )
        or value.get("baseline_remote_reverified_at_render") is not True
        or value.get("source_4cpu_task_sha256") != canonical_sha256(source)
        or value.get("source_4cpu_logical_dedupe_key") != source["dedupe_key"]
        or value.get("candidate_task_sha256") != canonical_sha256(candidate)
        or value.get("candidate_dedupe_key") != candidate["dedupe_key"]
        or value.get("terminal_evidence")
        != _terminal_evidence(str(candidate["remote_cwd"]), int(config["seed"]))
        or value.get("terminal_gates") != _terminal_gates()
        or value.get("remote_read_policy") != _remote_read_policy()
        or value.get("scheduler_client_policy") != _scheduler_client_policy()
        or value.get("superseded_launch_forbidden_package_sha256")
        != SUPERSEDED_V2_PACKAGE_SHA256
        or value.get("superseded_launch_forbidden_reason")
        != "production-client and remote-read policies were not sufficiently bounded"
        or value.get("complete_production_and_canary_namespaces_required") is not True
        or value.get("post_submit_namespace_rescan_required") is not True
        or value.get("remote_ready_reread_required") is not True
        or value.get("scheduler_endpoint") != "POST /api/tasks"
        or value.get("one_additive_post_maximum") != 1
        or value.get("explicit_apply_required") is not True
        or any(
            value.get(field) is not False
            for field in (
                "scheduler_write_performed",
                "submission_performed",
                "cancellation_performed",
                "preemption_performed",
                "publication_performed",
                "fea_submission_performed",
                "aedt_used",
                "production_promotion_allowed",
                "automatic_promotion_performed",
            )
        )
    ):
        raise RuntimeError("resource2 canary package seal mismatch")
    for field in (
        "wall_time_seconds",
        "process_tree_cpu_seconds",
        "average_cores_used",
        "throughput_seeds_per_second",
    ):
        raw = baseline[field]  # type: ignore[index]
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
            or float(raw) <= 0
        ):
            raise RuntimeError(f"resource2 package baseline {field} is invalid")
    return copy.deepcopy(dict(value))


def build_package(
    *,
    config_path: Path,
    offload_plan_path: Path,
    publication_receipt_path: Path,
    baseline_package_path: Path,
    baseline_submission_path: Path,
    baseline_terminal_path: Path,
    scheduler_url: str,
    baseline_scheduler: Any | None = None,
    baseline_remote_file_reader: Any | None = None,
) -> dict[str, Any]:
    config = validate_config(_read_json(config_path, "resource2 config"))
    baseline_package, baseline_submission = _validate_baseline_package_and_submission(
        config=config,
        package_path=baseline_package_path,
        submission_path=baseline_submission_path,
    )
    baseline_terminal = validate_baseline_terminal(
        _read_json(baseline_terminal_path, "baseline remote terminal"),
        config=config,
        baseline_package=baseline_package,
        baseline_submission=baseline_submission,
    )
    baseline_remote_reverification = _reverify_baseline_remote(
        config=config,
        baseline_package=baseline_package,
        baseline_terminal=baseline_terminal,
        scheduler_url=scheduler_url,
        scheduler=baseline_scheduler,
        remote_file_reader=baseline_remote_file_reader,
    )
    plan_file_sha = sha256_file(offload_plan_path.resolve(strict=True))
    if plan_file_sha != config["expected_offload_plan_file_sha256"]:
        raise RuntimeError("resource2 offload plan file SHA mismatch")
    binding = validate_stage_binding(
        bindings_root=Path.cwd(),
        record={
            "stage_id": ENTRY_STAGE_ID,
            "offload_plan": str(offload_plan_path.resolve(strict=True)),
            "publication_receipt": str(publication_receipt_path.resolve(strict=True)),
        },
        stage=BY_ID[ENTRY_STAGE_ID],
    )
    plan = binding["plan"]
    manifest = binding["manifest"]
    publication = binding["publication"]
    if (
        plan.get("bundle_id") != config["expected_bundle_id"]
        or plan.get("bundle_manifest_sha256")
        != config["expected_bundle_manifest_sha256"]
        or plan.get("contract_sha256") != config["expected_bundle_contract_sha256"]
    ):
        raise RuntimeError("resource2 bundle identity differs from sealed config")
    runtime = validate_required_runtime_code(
        manifest,
        REMOTE_CODE_FILES,
        required_code_sha256={
            str(INFERENCE_SAFETY["required_helper_path"]): str(
                INFERENCE_SAFETY["required_helper_sha256"]
            )
        },
    )
    runtime_hashes = _runtime_hard_hashes(manifest)
    base = build_current7_task_payload(
        plan,
        manifest,
        seed=int(config["seed"]),
        priority=DEFAULT_PRIORITY,
    )
    source = build_stage_task(base, stage=BY_ID[ENTRY_STAGE_ID], wave="refill")
    baseline_summary = _baseline_summary(baseline_terminal)
    candidate = _transform_source_task(source, config=config, baseline=baseline_summary)
    value = _assemble_package(
        config=config,
        plan=plan,
        publication=publication,
        baseline_terminal=baseline_terminal,
        baseline_package_file_sha256=sha256_file(
            baseline_package_path.resolve(strict=True)
        ),
        baseline_submission_file_sha256=sha256_file(
            baseline_submission_path.resolve(strict=True)
        ),
        baseline_terminal_file_sha256=sha256_file(
            baseline_terminal_path.resolve(strict=True)
        ),
        baseline_remote_reverification=baseline_remote_reverification,
        source_task=source,
        candidate_task=candidate,
        offload_plan_file_sha256=plan_file_sha,
        publication_receipt_file_sha256=sha256_file(
            publication_receipt_path.resolve(strict=True)
        ),
        required_runtime_code_sha256=runtime["sha256"],
        runtime_hard_hashes=runtime_hashes,
    )
    return validate_package(value)


def _validate_rows_for_prefix(
    rows: Sequence[Mapping[str, Any]], *, prefix: str
) -> list[dict[str, Any]]:
    expected_dedupe = (
        PRODUCTION_DEDUPE_PREFIX
        if prefix == PRODUCTION_PREFIX
        else CANARY_DEDUPE_PREFIX
    )
    result: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    seen_dedupes: dict[str, int] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise RuntimeError("Scheduler namespace contains a non-object row")
        task_id = raw.get("id", raw.get("task_id"))
        name = str(raw.get("name") or "")
        dedupe = str(raw.get("dedupe_key") or "")
        if (
            isinstance(task_id, bool)
            or not isinstance(task_id, int)
            or task_id <= 0
            or not name.startswith(prefix)
            or not dedupe.startswith(expected_dedupe)
            or task_id in seen_ids
            or (dedupe in seen_dedupes and seen_dedupes[dedupe] != task_id)
        ):
            raise RuntimeError(f"Scheduler {prefix} namespace identity drifted")
        seen_ids.add(task_id)
        seen_dedupes[dedupe] = task_id
        result.append(dict(raw))
    return result


def _paged_namespace(scheduler: Any, *, prefix: str) -> list[dict[str, Any]]:
    request = getattr(scheduler, "read_inventory_path", None)
    if not callable(request):
        raise RuntimeError("Scheduler client lacks read-only namespace pagination")

    def read_page(*, page: int, before_id: int) -> dict[str, Any]:
        query: dict[str, Any] = {
            "compact": "false",
            "paged": "true",
            "page": page,
            "page_size": PAGE_SIZE,
            "name_prefix": prefix,
            "sort_by": "id",
            "sort_order": "desc",
        }
        if before_id:
            query["before_id"] = before_id
        payload = request("/api/tasks?" + urllib.parse.urlencode(query))
        items = payload.get("items") if isinstance(payload, dict) else None
        filters = payload.get("filters") if isinstance(payload, dict) else None
        if (
            not isinstance(payload, dict)
            or not isinstance(items, list)
            or len(items) > PAGE_SIZE
            or payload.get("page") != page
            or payload.get("page_size") != PAGE_SIZE
            or payload.get("sort_by") != "id"
            or payload.get("sort_order") != "desc"
            or not isinstance(filters, dict)
            or filters.get("name_prefix") != prefix
            or filters.get("before_id") != before_id
        ):
            raise RuntimeError(f"Scheduler returned invalid {prefix} namespace page")
        return dict(payload)

    head = read_page(page=1, before_id=0)
    head_rows = _validate_rows_for_prefix(head["items"], prefix=prefix)
    high = max((int(row.get("id", row.get("task_id"))) for row in head_rows), default=0)
    snapshot_before = high + 1 if high else 0
    first = read_page(page=1, before_id=snapshot_before)
    page_count = first.get("page_count")
    filtered_total = first.get("filtered_total")
    if (
        isinstance(page_count, bool)
        or not isinstance(page_count, int)
        or not 1 <= page_count <= MAX_PAGES
        or isinstance(filtered_total, bool)
        or not isinstance(filtered_total, int)
        or filtered_total < 0
    ):
        raise RuntimeError(f"Scheduler {prefix} namespace metadata drifted")
    rows = [dict(row) for row in first["items"]]
    for page in range(2, page_count + 1):
        payload = read_page(page=page, before_id=snapshot_before)
        if (
            payload.get("page_count") != page_count
            or payload.get("filtered_total") != filtered_total
        ):
            raise RuntimeError(f"Scheduler {prefix} snapshot changed between pages")
        rows.extend(dict(row) for row in payload["items"])
    rows = _validate_rows_for_prefix(rows, prefix=prefix)
    if len(rows) != filtered_total:
        raise RuntimeError(f"Scheduler {prefix} namespace is truncated")
    tail = read_page(page=1, before_id=0)
    tail_rows = _validate_rows_for_prefix(tail["items"], prefix=prefix)
    if high and tail_rows:
        tail_min = min(int(row.get("id", row.get("task_id"))) for row in tail_rows)
        if len(tail_rows) == PAGE_SIZE and tail_min > high:
            raise RuntimeError(f"Scheduler {prefix} tail no longer overlaps snapshot")
    new_rows = [
        row for row in tail_rows if int(row.get("id", row.get("task_id"))) > high
    ]
    return _validate_rows_for_prefix([*new_rows, *rows], prefix=prefix)


def _complete_namespaces(scheduler: Any) -> list[dict[str, Any]]:
    complete = getattr(scheduler, "list_complete_namespace_tasks", None)
    if callable(complete):
        raw = complete()
        if not isinstance(raw, list):
            raise RuntimeError("complete Scheduler namespace is not a list")
        if any(not isinstance(row, Mapping) for row in raw):
            raise RuntimeError("complete Scheduler namespace contains a non-object")
        production = _validate_rows_for_prefix(
            [
                row
                for row in raw
                if str(row.get("name") or "").startswith(PRODUCTION_PREFIX)
            ],
            prefix=PRODUCTION_PREFIX,
        )
        canary = _validate_rows_for_prefix(
            [
                row
                for row in raw
                if str(row.get("name") or "").startswith(CANARY_PREFIX)
            ],
            prefix=CANARY_PREFIX,
        )
    else:
        production = _paged_namespace(scheduler, prefix=PRODUCTION_PREFIX)
        canary = _paged_namespace(scheduler, prefix=CANARY_PREFIX)
    combined = [*production, *canary]
    ids = [int(row.get("id", row.get("task_id"))) for row in combined]
    if len(ids) != len(set(ids)):
        raise RuntimeError("production/resource2 namespace task id overlap")
    return combined


def _row_claims_seed(row: Mapping[str, Any], seed: int) -> bool:
    numbers = [
        int(value)
        for value in re.findall(r"(?<!\d)(\d{9,12})(?!\d)", str(row.get("name") or ""))
    ]
    if not numbers:
        return False
    if numbers[-1] == seed:
        return True
    return len(numbers) >= 2 and numbers[-2] <= seed <= numbers[-1]


def _authenticate_task(
    row: Mapping[str, Any] | None,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> tuple[int, str]:
    observation = scheduler_task_observation(row or {})
    if observation is None or not scheduler_task_identity_matches(row or {}, expected):
        raise RuntimeError(f"{label} changed the sealed Scheduler identity")
    return observation


def _validate_submission_receipt(
    value: Mapping[str, Any], *, package: Mapping[str, Any]
) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    required = {
        "schema_version",
        "observed_at",
        "apply",
        "package_sha256",
        "bundle_id",
        "baseline_remote_terminal_sha256",
        "baseline_detail_revalidated",
        "ready_sha256",
        "remote_ready_reread_count",
        "complete_namespace_row_count",
        "complete_namespace_max_task_id",
        "complete_namespace_sha256",
        "post_submit_namespace_row_count",
        "post_submit_namespace_max_task_id",
        "post_submit_namespace_sha256",
        "post_submit_candidate_match_count",
        "post_submit_source_match_count",
        "post_submit_seed_claim_match_count",
        "all_identities_revalidated_after_post",
        "candidate_dedupe_key",
        "source_4cpu_logical_dedupe_key",
        "both_dedupes_and_seed_absent_before_post",
        "task_id",
        "task_status",
        "submitted_count",
        "scheduler_endpoint",
        "scheduler_client_policy",
        "scheduler_get_attempt_count",
        "scheduler_get_retry_count",
        "scheduler_post_count",
        "scheduler_cancel_count",
        "scheduler_preempt_count",
        "publication_count",
        "fea_submission_performed",
        "aedt_used",
        "receipt_sha256",
    }
    apply = value.get("apply") is True
    if (
        set(value) != required
        or value.get("schema_version") != SUBMISSION_SCHEMA
        or value.get("receipt_sha256") != canonical_sha256(unsigned)
        or value.get("package_sha256") != package["package_sha256"]
        or value.get("bundle_id") != package["bundle_id"]
        or value.get("baseline_remote_terminal_sha256")
        != package["baseline"]["remote_terminal_sha256"]
        or value.get("baseline_detail_revalidated") is not True
        or value.get("candidate_dedupe_key") != package["candidate_dedupe_key"]
        or value.get("source_4cpu_logical_dedupe_key")
        != package["source_4cpu_logical_dedupe_key"]
        or value.get("both_dedupes_and_seed_absent_before_post") is not True
        or value.get("submitted_count") != int(apply)
        or value.get("scheduler_client_policy") != _scheduler_client_policy()
        or isinstance(value.get("scheduler_get_attempt_count"), bool)
        or not isinstance(value.get("scheduler_get_attempt_count"), int)
        or value.get("scheduler_get_attempt_count") < 0
        or isinstance(value.get("scheduler_get_retry_count"), bool)
        or not isinstance(value.get("scheduler_get_retry_count"), int)
        or not 0
        <= value.get("scheduler_get_retry_count")
        <= value.get("scheduler_get_attempt_count")
        or value.get("scheduler_post_count") != int(apply)
        or value.get("scheduler_cancel_count") != 0
        or value.get("scheduler_preempt_count") != 0
        or value.get("publication_count") != 0
        or value.get("fea_submission_performed") is not False
        or value.get("aedt_used") is not False
        or (apply and value.get("all_identities_revalidated_after_post") is not True)
        or (
            not apply
            and value.get("all_identities_revalidated_after_post") is not False
        )
    ):
        raise RuntimeError("resource2 submission receipt seal mismatch")
    return copy.deepcopy(dict(value))


def _submission_outcome(
    *,
    package: Mapping[str, Any],
    baseline_package: Mapping[str, Any],
    publication: Mapping[str, Any],
    transport: Any,
    scheduler: Any,
    apply: bool,
    receipt_out: Path | None,
) -> dict[str, Any]:
    package = validate_package(package)
    baseline_id, baseline_status = _authenticate_task(
        scheduler.get_task(BASELINE_TASK_ID),
        baseline_package["parent_task"],
        label="4-CPU baseline Scheduler detail",
    )
    if baseline_id != BASELINE_TASK_ID or baseline_status != "completed":
        raise RuntimeError("pinned 4-CPU baseline is no longer completed/exact")
    live = shape1._live_ready(package, publication, transport)
    rows = _complete_namespaces(scheduler)
    candidate_dedupe = str(package["candidate_dedupe_key"])
    source_dedupe = str(package["source_4cpu_logical_dedupe_key"])
    seed = int(package["seed"])
    inventory = {str(row.get("dedupe_key") or ""): row for row in rows}
    seed_rows = [row for row in rows if _row_claims_seed(row, seed)]
    existing_receipt = None
    if receipt_out is not None and receipt_out.is_file():
        existing_receipt = _validate_submission_receipt(
            _read_json(receipt_out, "resource2 submission receipt"), package=package
        )
    candidate_row = inventory.get(candidate_dedupe)
    source_row = inventory.get(source_dedupe)
    expected = package["candidate_task"]
    if existing_receipt is not None:
        if candidate_row is None or source_row is not None or len(seed_rows) != 1:
            raise RuntimeError("receipt candidate disappeared or namespace was reused")
        task_id, _status = _authenticate_task(
            candidate_row, expected, label="receipt candidate inventory"
        )
        if task_id != existing_receipt["task_id"]:
            raise RuntimeError("receipt candidate task id changed")
        return existing_receipt
    if candidate_row is not None or source_row is not None or seed_rows:
        raise RuntimeError("resource2 candidate dedupe/seed is not globally unused")
    task_id = None
    status = "absent"
    post_rows: list[dict[str, Any]] = []
    candidate_matches = 0
    source_matches = 0
    seed_matches = 0
    revalidated = False
    if apply:
        if receipt_out is None:
            raise RuntimeError("--apply requires --receipt-out")
        submitted = scheduler.submit_task(expected)
        task_id, status = _authenticate_task(
            submitted, expected, label="resource2 Scheduler POST response"
        )
        detail_id, status = _authenticate_task(
            scheduler.get_task(task_id), expected, label="resource2 Scheduler detail"
        )
        if detail_id != task_id:
            raise RuntimeError("resource2 POST/detail task id mismatch")
        post_rows = _complete_namespaces(scheduler)
        candidate_rows = [
            row for row in post_rows if row.get("dedupe_key") == candidate_dedupe
        ]
        source_rows = [
            row for row in post_rows if row.get("dedupe_key") == source_dedupe
        ]
        post_seed_rows = [row for row in post_rows if _row_claims_seed(row, seed)]
        candidate_matches = len(candidate_rows)
        source_matches = len(source_rows)
        seed_matches = len(post_seed_rows)
        if candidate_matches != 1 or source_matches != 0 or seed_matches != 1:
            raise RuntimeError("resource2 identity changed during Scheduler POST")
        post_id, _ = _authenticate_task(
            candidate_rows[0], expected, label="post-submit resource2 inventory"
        )
        if post_id != task_id:
            raise RuntimeError("resource2 POST/post-scan task id mismatch")
        revalidated = True
    projection = [
        {
            "id": row.get("id", row.get("task_id")),
            "name": row.get("name"),
            "dedupe_key": row.get("dedupe_key"),
            "status": row.get("status"),
        }
        for row in rows
    ]
    post_projection = [
        {
            "id": row.get("id", row.get("task_id")),
            "name": row.get("name"),
            "dedupe_key": row.get("dedupe_key"),
            "status": row.get("status"),
        }
        for row in post_rows
    ]
    max_id = max((int(item["id"]) for item in projection), default=0)
    post_max_id = max((int(item["id"]) for item in post_projection), default=0)
    if apply and (task_id is None or task_id <= max_id or post_max_id < task_id):
        raise RuntimeError("resource2 POST was not one new additive task")
    unsigned = {
        "schema_version": SUBMISSION_SCHEMA,
        "observed_at": _now(),
        "apply": bool(apply),
        "package_sha256": package["package_sha256"],
        "bundle_id": package["bundle_id"],
        "baseline_remote_terminal_sha256": package["baseline"][
            "remote_terminal_sha256"
        ],
        "baseline_detail_revalidated": True,
        "ready_sha256": canonical_sha256(live),
        "remote_ready_reread_count": 1,
        "complete_namespace_row_count": len(rows),
        "complete_namespace_max_task_id": max_id,
        "complete_namespace_sha256": canonical_sha256(projection),
        "post_submit_namespace_row_count": len(post_rows) if apply else None,
        "post_submit_namespace_max_task_id": post_max_id if apply else None,
        "post_submit_namespace_sha256": canonical_sha256(post_projection)
        if apply
        else None,
        "post_submit_candidate_match_count": candidate_matches,
        "post_submit_source_match_count": source_matches,
        "post_submit_seed_claim_match_count": seed_matches,
        "all_identities_revalidated_after_post": revalidated,
        "candidate_dedupe_key": candidate_dedupe,
        "source_4cpu_logical_dedupe_key": source_dedupe,
        "both_dedupes_and_seed_absent_before_post": True,
        "task_id": task_id,
        "task_status": status,
        "submitted_count": int(apply),
        "scheduler_endpoint": "POST /api/tasks",
        "scheduler_client_policy": _scheduler_client_policy(),
        "scheduler_get_attempt_count": int(getattr(scheduler, "get_attempt_count", 0)),
        "scheduler_get_retry_count": int(getattr(scheduler, "get_retry_count", 0)),
        "scheduler_post_count": int(getattr(scheduler, "post_count", 0)),
        "scheduler_cancel_count": 0,
        "scheduler_preempt_count": 0,
        "publication_count": 0,
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    value = {**unsigned, "receipt_sha256": canonical_sha256(unsigned)}
    if apply:
        if getattr(scheduler, "post_count", 0) != 1:
            raise RuntimeError("resource2 canary did not perform exactly one POST")
        _atomic_json(receipt_out, value)  # type: ignore[arg-type]
    elif getattr(scheduler, "post_count", 0) != 0:
        raise RuntimeError("resource2 dry-run performed a Scheduler POST")
    return _validate_submission_receipt(value, package=package)


def submit(
    *,
    config_path: Path,
    offload_plan_path: Path,
    publication_receipt_path: Path,
    baseline_package_path: Path,
    baseline_submission_path: Path,
    baseline_terminal_path: Path,
    package_path: Path,
    scheduler_url: str,
    accounts_path: Path,
    scheduler_source: Path,
    publication_account: str,
    apply: bool,
    receipt_out: Path | None,
    dry_run_out: Path | None,
) -> dict[str, Any]:
    if apply and dry_run_out is not None:
        raise RuntimeError("--dry-run-out cannot be combined with --apply")
    rebuilt = build_package(
        config_path=config_path,
        offload_plan_path=offload_plan_path,
        publication_receipt_path=publication_receipt_path,
        baseline_package_path=baseline_package_path,
        baseline_submission_path=baseline_submission_path,
        baseline_terminal_path=baseline_terminal_path,
        scheduler_url=scheduler_url,
    )
    package = validate_package(_read_json(package_path, "resource2 package"))
    if package != rebuilt:
        raise RuntimeError("resource2 package no longer reproduces from sources")
    config = validate_config(_read_json(config_path, "resource2 config"))
    baseline_package, _ = _validate_baseline_package_and_submission(
        config=config,
        package_path=baseline_package_path,
        submission_path=baseline_submission_path,
    )
    publication = _read_json(publication_receipt_path, "publication receipt")
    with scheduler_publication_transport(
        accounts_path=accounts_path,
        scheduler_source=scheduler_source,
        account_name=publication_account,
    ) as transport:
        value = _submission_outcome(
            package=package,
            baseline_package=baseline_package,
            publication=publication,
            transport=transport,
            scheduler=Resource2AdditiveSchedulerClient(
                scheduler_url, expected_task=package["candidate_task"]
            ),
            apply=apply,
            receipt_out=receipt_out,
        )
    if not apply and dry_run_out is not None:
        _atomic_json(dry_run_out, value)
    return value


def validate_telemetry(
    value: Mapping[str, Any], *, package: Mapping[str, Any], task_id: int
) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "telemetry_sha256"}
    required = {
        "schema_version",
        "started_at",
        "finished_at",
        "task_id",
        "seed",
        "payload_sha256",
        "baseline_remote_terminal_sha256",
        "requested_cpus",
        "requested_memory_bytes",
        "rss_gate_bytes",
        "slurm_cpus_per_task",
        "thread_environment",
        "cpu_set_before",
        "cpu_set_after",
        "wall_time_seconds",
        "process_tree_cpu_seconds",
        "cpu_capacity_seconds",
        "average_cores_used",
        "cpu_utilization_fraction",
        "peak_rss_bytes",
        "cgroup_before",
        "cgroup_after",
        "seed_status",
        "result",
        "exit_code",
        "failure",
        "fea_submission_performed",
        "aedt_used",
        "telemetry_sha256",
    }
    task = package["candidate_task"]
    payload_sha = canonical_sha256(task["payload_json"])
    before = value.get("cgroup_before")
    after = value.get("cgroup_after")
    status = value.get("seed_status")
    result = value.get("result")
    if not all(isinstance(item, Mapping) for item in (before, after, status, result)):
        raise RuntimeError("resource2 telemetry nested evidence is absent")

    def validate_cgroup_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
        if set(snapshot) != {
            "leaf_relative_path",
            "leaf_memory_max_unbounded",
            "ancestors",
            "selected_finite_ancestor",
        }:
            raise RuntimeError("resource2 cgroup snapshot fields drifted")
        ancestors = snapshot.get("ancestors")
        selected = snapshot.get("selected_finite_ancestor")
        if (
            not isinstance(snapshot.get("leaf_relative_path"), str)
            or not snapshot["leaf_relative_path"]
            or not isinstance(snapshot.get("leaf_memory_max_unbounded"), bool)
            or not isinstance(ancestors, list)
            or not ancestors
            or not isinstance(selected, Mapping)
        ):
            raise RuntimeError("resource2 cgroup snapshot is incomplete")
        normalized: list[dict[str, Any]] = []
        prior_depth = -1
        for raw in ancestors:
            if not isinstance(raw, Mapping) or set(raw) != {
                "depth_from_leaf",
                "relative_path",
                "memory_max_raw",
                "memory_limit_bytes",
                "memory_current_bytes",
                "memory_peak_bytes",
            }:
                raise RuntimeError("resource2 cgroup ancestor record drifted")
            depth = raw.get("depth_from_leaf")
            limit_value = raw.get("memory_limit_bytes")
            current_value = raw.get("memory_current_bytes")
            peak_value = raw.get("memory_peak_bytes")
            if (
                isinstance(depth, bool)
                or not isinstance(depth, int)
                or depth <= prior_depth
                or not isinstance(raw.get("relative_path"), str)
                or not raw["relative_path"]
                or (
                    limit_value is not None
                    and (
                        isinstance(limit_value, bool)
                        or not isinstance(limit_value, int)
                        or limit_value <= 0
                    )
                )
                or (limit_value is None and raw.get("memory_max_raw") != "max")
                or (
                    limit_value is not None
                    and raw.get("memory_max_raw") != str(limit_value)
                )
                or isinstance(current_value, bool)
                or not isinstance(current_value, int)
                or current_value < 0
                or isinstance(peak_value, bool)
                or not isinstance(peak_value, int)
                or peak_value < 0
            ):
                raise RuntimeError("resource2 cgroup ancestor value drifted")
            prior_depth = depth
            normalized.append(dict(raw))
        if normalized[0]["depth_from_leaf"] != 0:
            raise RuntimeError("resource2 cgroup ancestry omitted the leaf")
        finite = [item for item in normalized if item["memory_limit_bytes"] is not None]
        if (
            not finite
            or dict(selected) != finite[0]
            or snapshot["leaf_memory_max_unbounded"]
            is not (normalized[0]["memory_limit_bytes"] is None)
        ):
            raise RuntimeError("resource2 nearest finite cgroup ancestor drifted")
        return {
            "leaf_relative_path": snapshot["leaf_relative_path"],
            "leaf_memory_max_unbounded": snapshot["leaf_memory_max_unbounded"],
            "ancestors": normalized,
            "selected_finite_ancestor": dict(selected),
        }

    before = validate_cgroup_snapshot(before)  # type: ignore[arg-type]
    after = validate_cgroup_snapshot(after)  # type: ignore[arg-type]
    before_selected = before["selected_finite_ancestor"]
    after_selected = after["selected_finite_ancestor"]
    before_ancestors = before["ancestors"]
    after_ancestors = after["ancestors"]
    if not isinstance(before_selected, Mapping) or not isinstance(
        after_selected, Mapping
    ):
        raise RuntimeError("resource2 finite cgroup ancestor evidence is absent")
    if not isinstance(before_ancestors, list) or not isinstance(after_ancestors, list):
        raise RuntimeError("resource2 cgroup ancestry is absent")
    if not before_ancestors or not after_ancestors:
        raise RuntimeError("resource2 cgroup ancestry is empty")
    memory_request = CANDIDATE_MEMORY_MB * 1024**2
    limit = after_selected.get("memory_limit_bytes")
    cgroup_peak = after_selected.get("memory_peak_bytes")
    cpu_before = value.get("cpu_set_before")
    cpu_after = value.get("cpu_set_after")
    if (
        set(value) != required
        or value.get("schema_version") != TELEMETRY_SCHEMA
        or value.get("telemetry_sha256") != canonical_sha256(unsigned)
        or value.get("task_id") != task_id
        or value.get("seed") != package["seed"]
        or value.get("payload_sha256") != payload_sha
        or value.get("baseline_remote_terminal_sha256")
        != package["baseline"]["remote_terminal_sha256"]
        or value.get("requested_cpus") != CANDIDATE_CPUS
        or value.get("requested_memory_bytes") != memory_request
        or value.get("rss_gate_bytes") != DEFAULT_PEAK_RSS_GATE_BYTES
        or value.get("slurm_cpus_per_task") != str(CANDIDATE_CPUS)
        or value.get("thread_environment") != _thread_environment()
        or not isinstance(cpu_before, list)
        or not isinstance(cpu_after, list)
        or len(cpu_before) != CANDIDATE_CPUS
        or len(cpu_after) != CANDIDATE_CPUS
        or len(set(cpu_before)) != CANDIDATE_CPUS
        or any(
            isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0
            for cpu in cpu_before
        )
        or cpu_before != sorted(cpu_before)
        or cpu_after != sorted(cpu_after)
        or cpu_before != cpu_after
        or before.get("leaf_relative_path") != after.get("leaf_relative_path")  # type: ignore[union-attr]
        or not isinstance(before.get("leaf_memory_max_unbounded"), bool)  # type: ignore[union-attr]
        or before.get("leaf_memory_max_unbounded")  # type: ignore[union-attr]
        != after.get("leaf_memory_max_unbounded")  # type: ignore[union-attr]
        or not isinstance(before_ancestors[0], Mapping)
        or not isinstance(after_ancestors[0], Mapping)
        or before_ancestors[0].get("depth_from_leaf") != 0
        or after_ancestors[0].get("depth_from_leaf") != 0
        or before_selected.get("relative_path") != after_selected.get("relative_path")
        or before_selected.get("memory_limit_bytes") != limit
        or isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit < memory_request
        or isinstance(cgroup_peak, bool)
        or not isinstance(cgroup_peak, int)
        or cgroup_peak < 0
        or cgroup_peak > limit
        or status.get("state") != "completed"  # type: ignore[union-attr]
        or status.get("terminal") is not True  # type: ignore[union-attr]
        or status.get("exit_code") != 0  # type: ignore[union-attr]
        or not _is_sha256(status.get("sha256"))  # type: ignore[union-attr]
        or not _is_sha256(result.get("sha256"))  # type: ignore[union-attr]
        or set(status)  # type: ignore[arg-type]
        != {"relative_path", "size", "sha256", "state", "terminal", "exit_code"}
        or status.get("relative_path")  # type: ignore[union-attr]
        != f"runs/task-{task_id}/seed_status.json"
        or isinstance(status.get("size"), bool)  # type: ignore[union-attr]
        or not isinstance(status.get("size"), int)  # type: ignore[union-attr]
        or status.get("size") <= 0  # type: ignore[operator]
        or set(result) != {"relative_path", "size", "sha256"}  # type: ignore[arg-type]
        or result.get("relative_path")  # type: ignore[union-attr]
        != f"runs/task-{task_id}/seed-{package['seed']}/result.json"
        or isinstance(result.get("size"), bool)  # type: ignore[union-attr]
        or not isinstance(result.get("size"), int)  # type: ignore[union-attr]
        or result.get("size") <= 0  # type: ignore[operator]
        or value.get("exit_code") != 0
        or value.get("failure") is not None
        or value.get("fea_submission_performed") is not False
        or value.get("aedt_used") is not False
    ):
        raise RuntimeError("resource2 telemetry identity/safety seal mismatch")
    numeric_positive = (
        "wall_time_seconds",
        "process_tree_cpu_seconds",
        "cpu_capacity_seconds",
        "average_cores_used",
        "cpu_utilization_fraction",
        "peak_rss_bytes",
    )
    for field in numeric_positive:
        raw = value.get(field)
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
            or float(raw) <= 0
        ):
            raise RuntimeError(f"resource2 telemetry {field} is not finite positive")
    wall = float(value["wall_time_seconds"])
    cpu = float(value["process_tree_cpu_seconds"])
    capacity = float(value["cpu_capacity_seconds"])
    average = float(value["average_cores_used"])
    utilization = float(value["cpu_utilization_fraction"])
    if (
        not math.isclose(capacity, wall * CANDIDATE_CPUS, rel_tol=1e-12, abs_tol=1e-9)
        or not math.isclose(average, cpu / wall, rel_tol=1e-12, abs_tol=1e-9)
        or not math.isclose(utilization, cpu / capacity, rel_tol=1e-12, abs_tol=1e-9)
        # ``getrusage`` and monotonic clocks are sampled independently, so a
        # few hundredths of a core can appear at the terminal boundary even
        # though the inherited affinity is sealed to exactly two CPUs.
        or average > CANDIDATE_CPUS + 0.05
        or int(value["peak_rss_bytes"]) > DEFAULT_PEAK_RSS_GATE_BYTES
    ):
        raise RuntimeError("resource2 telemetry derived CPU/RSS metric drifted")
    return copy.deepcopy(dict(value))


def evaluate_terminal_evidence(
    *,
    package: Mapping[str, Any],
    manifest: Mapping[str, Any],
    scheduler_status: str,
    task_id: int,
    telemetry: Mapping[str, Any],
    seed_status: Mapping[str, Any],
    seed_status_file_sha256: str,
    seed_status_size: int,
    result: Mapping[str, Any],
    result_file_sha256: str,
    result_size: int,
) -> dict[str, Any]:
    package = validate_package(package)
    evidence = validate_telemetry(telemetry, package=package, task_id=task_id)
    payload = package["candidate_task"]["payload_json"]
    payload_sha = canonical_sha256(payload)
    if (
        scheduler_status != "completed"
        or seed_status.get("schema_version") != STATUS_SCHEMA
        or seed_status.get("state") != "completed"
        or seed_status.get("terminal") is not True
        or seed_status.get("exit_code") != 0
        or seed_status.get("task_id") != str(task_id)
        or seed_status.get("seed") != package["seed"]
        or seed_status.get("bundle_id") != package["bundle_id"]
        or seed_status.get("payload_sha256") != payload_sha
        or seed_status.get("inference_threads") != CANDIDATE_CPUS
        or seed_status.get("result_sha256") != result_file_sha256
        or not _is_sha256(seed_status_file_sha256)
        or not _is_sha256(result_file_sha256)
        or seed_status_size <= 0
        or result_size <= 0
        or evidence["seed_status"]["sha256"] != seed_status_file_sha256
        or evidence["seed_status"]["size"] != seed_status_size
        or evidence["result"]["sha256"] != result_file_sha256
        or evidence["result"]["size"] != result_size
    ):
        raise RuntimeError("resource2 terminal status/result lineage mismatch")
    # This is the direct manifest -> published receipt -> submitted payload
    # scientific validation.  Callers must obtain ``manifest`` from the same
    # validate_stage_binding used to reproduce the package.
    validate_result(result, payload=payload, manifest=manifest)
    baseline = package["baseline"]
    candidate_wall = float(evidence["wall_time_seconds"])
    candidate_cpu = float(evidence["process_tree_cpu_seconds"])
    candidate_cores = candidate_cpu / candidate_wall
    single_ratio = float(baseline["wall_time_seconds"]) / candidate_wall
    slot_ratio = single_ratio * (
        CANDIDATE_THEORETICAL_SLOTS / BASELINE_THEORETICAL_SLOTS
    )
    core_ratio = candidate_cores / float(baseline["average_cores_used"])
    cgroup_selected = evidence["cgroup_after"]["selected_finite_ancestor"]
    gate_results = {
        "scheduler_completed": scheduler_status == "completed",
        "wrapper_and_seed_completed": (
            evidence["exit_code"] == 0
            and seed_status.get("state") == "completed"
            and seed_status.get("exit_code") == 0
        ),
        "exact_two_cpu_affinity": (
            len(evidence["cpu_set_before"]) == CANDIDATE_CPUS
            and evidence["cpu_set_before"] == evidence["cpu_set_after"]
        ),
        "exact_two_thread_binding": (
            evidence["thread_environment"] == _thread_environment()
            and result.get("inference_threads") == CANDIDATE_CPUS
            and result.get("scheduler_cpus") == CANDIDATE_CPUS
        ),
        "per_seed_peak_rss_within_gate": (
            int(evidence["peak_rss_bytes"]) <= DEFAULT_PEAK_RSS_GATE_BYTES
        ),
        "finite_cgroup_ancestor_covers_request": (
            isinstance(cgroup_selected.get("memory_limit_bytes"), int)
            and cgroup_selected["memory_limit_bytes"] >= CANDIDATE_MEMORY_MB * 1024**2
            and cgroup_selected["memory_peak_bytes"]
            <= cgroup_selected["memory_limit_bytes"]
        ),
        "single_seed_throughput_retains_75pct": (
            single_ratio >= MIN_SINGLE_SEED_THROUGHPUT_RATIO
        ),
        "slot_weighted_throughput_improves_10pct": (
            slot_ratio >= MIN_SLOT_WEIGHTED_THROUGHPUT_RATIO
        ),
        "core_use_retains_80pct": core_ratio >= MIN_CORE_USE_RATIO,
        "manifest_result_validation_passed": True,
        "baseline_terminal_was_promotion_eligible": (
            baseline["promotion_eligible"] is True
        ),
    }
    unsigned = {
        "schema_version": TERMINAL_SCHEMA,
        "task_id": task_id,
        "seed": package["seed"],
        "package_sha256": package["package_sha256"],
        "bundle_manifest_sha256": package["bundle_manifest_sha256"],
        "publication_receipt_sha256": package["publication_receipt_sha256"],
        "baseline_remote_terminal_sha256": baseline["remote_terminal_sha256"],
        "scheduler_status": scheduler_status,
        "telemetry_sha256": evidence["telemetry_sha256"],
        "seed_status_file_sha256": seed_status_file_sha256,
        "result_file_sha256": result_file_sha256,
        "candidate_wall_time_seconds": candidate_wall,
        "candidate_process_tree_cpu_seconds": candidate_cpu,
        "candidate_average_cores_used": candidate_cores,
        "candidate_peak_rss_bytes": evidence["peak_rss_bytes"],
        "baseline_wall_time_seconds": baseline["wall_time_seconds"],
        "baseline_average_cores_used": baseline["average_cores_used"],
        "single_seed_throughput_ratio_vs_4cpu": single_ratio,
        "baseline_theoretical_slots": BASELINE_THEORETICAL_SLOTS,
        "candidate_theoretical_slots": CANDIDATE_THEORETICAL_SLOTS,
        "slot_weighted_throughput_ratio_vs_4cpu": slot_ratio,
        "core_use_ratio_vs_4cpu": core_ratio,
        "finite_cgroup_ancestor": copy.deepcopy(cgroup_selected),
        "gate_results": gate_results,
        "promotion_eligible": all(gate_results.values()),
        "automatic_promotion_performed": False,
        "scheduler_post_count": 0,
        "scheduler_cancel_count": 0,
        "scheduler_preempt_count": 0,
        "publication_count": 0,
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    return {**unsigned, "terminal_sha256": canonical_sha256(unsigned)}


def validate_terminal_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "terminal_sha256"}
    required = {
        "schema_version",
        "task_id",
        "seed",
        "package_sha256",
        "bundle_manifest_sha256",
        "publication_receipt_sha256",
        "baseline_remote_terminal_sha256",
        "scheduler_status",
        "telemetry_sha256",
        "seed_status_file_sha256",
        "result_file_sha256",
        "candidate_wall_time_seconds",
        "candidate_process_tree_cpu_seconds",
        "candidate_average_cores_used",
        "candidate_peak_rss_bytes",
        "baseline_wall_time_seconds",
        "baseline_average_cores_used",
        "single_seed_throughput_ratio_vs_4cpu",
        "baseline_theoretical_slots",
        "candidate_theoretical_slots",
        "slot_weighted_throughput_ratio_vs_4cpu",
        "core_use_ratio_vs_4cpu",
        "finite_cgroup_ancestor",
        "gate_results",
        "promotion_eligible",
        "automatic_promotion_performed",
        "scheduler_post_count",
        "scheduler_cancel_count",
        "scheduler_preempt_count",
        "publication_count",
        "fea_submission_performed",
        "aedt_used",
        "terminal_sha256",
    }
    if (
        set(value) != required
        or value.get("schema_version") != TERMINAL_SCHEMA
        or value.get("terminal_sha256") != canonical_sha256(unsigned)
        or not isinstance(value.get("gate_results"), dict)
        or value.get("promotion_eligible") is not all(value["gate_results"].values())
        or value.get("baseline_theoretical_slots") != BASELINE_THEORETICAL_SLOTS
        or value.get("candidate_theoretical_slots") != CANDIDATE_THEORETICAL_SLOTS
        or value.get("automatic_promotion_performed") is not False
        or value.get("scheduler_post_count") != 0
        or value.get("scheduler_cancel_count") != 0
        or value.get("scheduler_preempt_count") != 0
        or value.get("publication_count") != 0
        or value.get("fea_submission_performed") is not False
        or value.get("aedt_used") is not False
    ):
        raise RuntimeError("resource2 terminal evidence seal mismatch")
    return copy.deepcopy(dict(value))


def _manifest_from_binding(
    *, offload_plan_path: Path, publication_receipt_path: Path
) -> dict[str, Any]:
    binding = validate_stage_binding(
        bindings_root=Path.cwd(),
        record={
            "stage_id": ENTRY_STAGE_ID,
            "offload_plan": str(offload_plan_path.resolve(strict=True)),
            "publication_receipt": str(publication_receipt_path.resolve(strict=True)),
        },
        stage=BY_ID[ENTRY_STAGE_ID],
    )
    _runtime_hard_hashes(binding["manifest"])
    return dict(binding["manifest"])


def _reproduce_package(
    *,
    config_path: Path,
    offload_plan_path: Path,
    publication_receipt_path: Path,
    baseline_package_path: Path,
    baseline_submission_path: Path,
    baseline_terminal_path: Path,
    package_path: Path,
    scheduler_url: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rebuilt = build_package(
        config_path=config_path,
        offload_plan_path=offload_plan_path,
        publication_receipt_path=publication_receipt_path,
        baseline_package_path=baseline_package_path,
        baseline_submission_path=baseline_submission_path,
        baseline_terminal_path=baseline_terminal_path,
        scheduler_url=scheduler_url,
    )
    package = validate_package(_read_json(package_path, "resource2 package"))
    if package != rebuilt:
        raise RuntimeError("resource2 package does not reproduce for evaluation")
    manifest = _manifest_from_binding(
        offload_plan_path=offload_plan_path,
        publication_receipt_path=publication_receipt_path,
    )
    if (
        manifest.get("bundle_id") != package["bundle_id"]
        or sha256_file(offload_plan_path.resolve(strict=True))
        != package["offload_plan_file_sha256"]
        or sha256_file(publication_receipt_path.resolve(strict=True))
        != package["publication_receipt_file_sha256"]
    ):
        raise RuntimeError("resource2 evaluator manifest/receipt binding drifted")
    return package, manifest


def evaluate_local_terminal(
    *,
    config_path: Path,
    offload_plan_path: Path,
    publication_receipt_path: Path,
    baseline_package_path: Path,
    baseline_submission_path: Path,
    baseline_terminal_path: Path,
    package_path: Path,
    task_id: int,
    telemetry_path: Path,
    seed_status_path: Path,
    result_path: Path,
    output_path: Path,
    scheduler_url: str,
) -> dict[str, Any]:
    package, manifest = _reproduce_package(
        config_path=config_path,
        offload_plan_path=offload_plan_path,
        publication_receipt_path=publication_receipt_path,
        baseline_package_path=baseline_package_path,
        baseline_submission_path=baseline_submission_path,
        baseline_terminal_path=baseline_terminal_path,
        package_path=package_path,
        scheduler_url=scheduler_url,
    )
    status_file = seed_status_path.resolve(strict=True)
    result_file = result_path.resolve(strict=True)
    terminal = evaluate_terminal_evidence(
        package=package,
        manifest=manifest,
        scheduler_status="completed",
        task_id=task_id,
        telemetry=_read_json(telemetry_path, "resource2 telemetry"),
        seed_status=_read_json(status_file, "resource2 seed status"),
        seed_status_file_sha256=sha256_file(status_file),
        seed_status_size=status_file.stat().st_size,
        result=_read_json(result_file, "resource2 result"),
        result_file_sha256=sha256_file(result_file),
        result_size=result_file.stat().st_size,
    )
    terminal = validate_terminal_evidence(terminal)
    _atomic_json(output_path, terminal)
    return terminal


def evaluate_remote_terminal(
    *,
    config_path: Path,
    offload_plan_path: Path,
    publication_receipt_path: Path,
    baseline_package_path: Path,
    baseline_submission_path: Path,
    baseline_terminal_path: Path,
    package_path: Path,
    submission_receipt_path: Path,
    scheduler_url: str,
    task_id: int,
    output_path: Path,
) -> dict[str, Any]:
    package, manifest = _reproduce_package(
        config_path=config_path,
        offload_plan_path=offload_plan_path,
        publication_receipt_path=publication_receipt_path,
        baseline_package_path=baseline_package_path,
        baseline_submission_path=baseline_submission_path,
        baseline_terminal_path=baseline_terminal_path,
        package_path=package_path,
        scheduler_url=scheduler_url,
    )
    submission = _validate_submission_receipt(
        _read_json(submission_receipt_path, "resource2 submission receipt"),
        package=package,
    )
    if (
        submission.get("apply") is not True
        or submission.get("task_id") != task_id
        or isinstance(task_id, bool)
        or not isinstance(task_id, int)
        or task_id <= 0
    ):
        raise RuntimeError("resource2 terminal task/submission identity mismatch")
    scheduler = Resource2AdditiveSchedulerClient(scheduler_url)
    observed_id, scheduler_status = _authenticate_task(
        scheduler.get_task(task_id),
        package["candidate_task"],
        label="resource2 terminal Scheduler detail",
    )
    if observed_id != task_id or scheduler_status != "completed":
        raise RuntimeError("resource2 candidate is not one completed exact task")
    root = f"runs/task-{task_id}"
    relative = {
        "telemetry": f"{root}/{TELEMETRY_FILENAME}",
        "seed_status": f"{root}/seed_status.json",
        "result": f"{root}/seed-{int(package['seed'])}/result.json",
    }
    remote_read_attempts: list[dict[str, Any]] = []
    telemetry_raw = _remote_file_bytes_bounded(
        scheduler_url=scheduler_url,
        task_id=task_id,
        relative_path=relative["telemetry"],
        attempt_audit=remote_read_attempts,
    )
    telemetry_parsed = shape1._json_bytes_object(telemetry_raw, "telemetry")
    telemetry_sealed = validate_telemetry(
        telemetry_parsed, package=package, task_id=task_id
    )
    status_record = telemetry_sealed["seed_status"]
    result_record = telemetry_sealed["result"]
    status_raw = _remote_file_bytes_bounded(
        scheduler_url=scheduler_url,
        task_id=task_id,
        relative_path=relative["seed_status"],
        expected_size=int(status_record["size"]),
        expected_sha256=str(status_record["sha256"]),
        attempt_audit=remote_read_attempts,
    )
    result_raw = _remote_file_bytes_bounded(
        scheduler_url=scheduler_url,
        task_id=task_id,
        relative_path=relative["result"],
        expected_size=int(result_record["size"]),
        expected_sha256=str(result_record["sha256"]),
        attempt_audit=remote_read_attempts,
    )
    raw = {
        "telemetry": telemetry_raw,
        "seed_status": status_raw,
        "result": result_raw,
    }
    parsed = {
        label: shape1._json_bytes_object(payload, label)
        for label, payload in raw.items()
    }
    terminal = evaluate_terminal_evidence(
        package=package,
        manifest=manifest,
        scheduler_status=scheduler_status,
        task_id=task_id,
        telemetry=parsed["telemetry"],
        seed_status=parsed["seed_status"],
        seed_status_file_sha256=hashlib.sha256(raw["seed_status"]).hexdigest(),
        seed_status_size=len(raw["seed_status"]),
        result=parsed["result"],
        result_file_sha256=hashlib.sha256(raw["result"]).hexdigest(),
        result_size=len(raw["result"]),
    )
    terminal = validate_terminal_evidence(terminal)
    final_id, final_status = _authenticate_task(
        scheduler.get_task(task_id),
        package["candidate_task"],
        label="post-evidence resource2 Scheduler detail",
    )
    if final_id != task_id or final_status != scheduler_status:
        raise RuntimeError("resource2 Scheduler identity changed during evidence GET")
    file_records = {
        label: {
            "relative_path": relative[label],
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for label, payload in raw.items()
    }
    unsigned = {
        "schema_version": REMOTE_TERMINAL_SCHEMA,
        "observed_at": _now(),
        "package_sha256": package["package_sha256"],
        "submission_receipt_sha256": submission["receipt_sha256"],
        "task_id": task_id,
        "scheduler_status": scheduler_status,
        "scheduler_detail_get_count": 2,
        "scheduler_remote_file_get_count": len(raw),
        "scheduler_remote_file_attempt_count": sum(
            int(item["attempt_count"]) for item in remote_read_attempts
        ),
        "remote_read_policy": _remote_read_policy(),
        "remote_file_read_attempts": remote_read_attempts,
        "remote_files": file_records,
        "terminal_evidence": terminal,
        "terminal_sha256": terminal["terminal_sha256"],
        "promotion_eligible": terminal["promotion_eligible"],
        "scheduler_access": "GET-only",
        "scheduler_post_count": 0,
        "scheduler_cancel_count": 0,
        "scheduler_preempt_count": 0,
        "publication_count": 0,
        "remote_access": "read-only",
        "remote_write_count": 0,
        "automatic_promotion_performed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    value = {**unsigned, "remote_terminal_sha256": canonical_sha256(unsigned)}
    _atomic_json(output_path, value)
    return value


def attest_baseline(
    *,
    config_path: Path,
    baseline_package_path: Path,
    baseline_submission_path: Path,
    scheduler_url: str,
    output_path: Path,
) -> dict[str, Any]:
    config = validate_config(_read_json(config_path, "resource2 config"))
    package, submission = _validate_baseline_package_and_submission(
        config=config,
        package_path=baseline_package_path,
        submission_path=baseline_submission_path,
    )
    if output_path.exists():
        raise RuntimeError(f"immutable output already exists: {output_path}")
    scheduler = Resource2AdditiveSchedulerClient(scheduler_url)
    observed_id, scheduler_status = _authenticate_task(
        scheduler.get_task(BASELINE_TASK_ID),
        package["parent_task"],
        label="bounded baseline Scheduler detail",
    )
    if observed_id != BASELINE_TASK_ID or scheduler_status != "completed":
        raise RuntimeError("pinned baseline is not one completed exact task")
    seed = int(package["seed"])
    task_root = f"runs/task-{BASELINE_TASK_ID}"
    relative = {
        "batch_manifest": f"{task_root}/batch_manifest.json",
        "task_status": f"{task_root}/task_status.json",
        "child_receipt": f"{task_root}/seed-{seed}/seed_status.json",
    }
    raw = {
        label: _remote_file_bytes_bounded(
            scheduler_url=scheduler_url,
            task_id=BASELINE_TASK_ID,
            relative_path=path,
        )
        for label, path in relative.items()
    }
    manifest = shape1.validate_batch_manifest(
        shape1._json_bytes_object(raw["batch_manifest"], "batch manifest")
    )
    expected_manifest = shape1.batch_manifest_from_payload(
        package["parent_task"]["payload_json"]
    )
    if manifest != expected_manifest:
        raise RuntimeError("bounded baseline manifest differs from submitted parent")
    task_status = shape1.validate_task_status(
        shape1._json_bytes_object(raw["task_status"], "task status")
    )
    child_receipt = shape1.validate_child_receipt(
        shape1._json_bytes_object(raw["child_receipt"], "child receipt"),
        manifest=manifest,
    )
    for label in ("stdout", "stderr"):
        receipt_relative = str(child_receipt[f"{label}_relative_path"])
        relative[f"child_{label}"] = f"{task_root}/{receipt_relative}"
        stream = _remote_file_bytes_bounded(
            scheduler_url=scheduler_url,
            task_id=BASELINE_TASK_ID,
            relative_path=relative[f"child_{label}"],
            expected_size=int(child_receipt[f"{label}_size_bytes"]),
            expected_sha256=str(child_receipt[f"{label}_sha256"]),
        )
        raw[f"child_{label}"] = stream
    result_sha = child_receipt.get("result_sha256")
    result_relative = f"{task_root}/seed-{seed}/result.json"
    result_bytes = None
    if result_sha is not None:
        result_bytes = _remote_file_bytes_bounded(
            scheduler_url=scheduler_url,
            task_id=BASELINE_TASK_ID,
            relative_path=result_relative,
            expected_sha256=str(result_sha),
        )
    metrics = shape1.evaluate_terminal_cpu_evidence(
        task_status, [child_receipt], expected_shape=1
    )
    final_id, final_status = _authenticate_task(
        scheduler.get_task(BASELINE_TASK_ID),
        package["parent_task"],
        label="post-evidence bounded baseline Scheduler detail",
    )
    if final_id != BASELINE_TASK_ID or final_status != scheduler_status:
        raise RuntimeError("baseline Scheduler identity changed during bounded GET")
    records = {
        label: {
            "relative_path": relative[label],
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for label, payload in raw.items()
    }
    records["result"] = {
        "relative_path": result_relative,
        "size": len(result_bytes) if result_bytes is not None else None,
        "sha256": (
            hashlib.sha256(result_bytes).hexdigest()
            if result_bytes is not None
            else None
        ),
    }
    unsigned = {
        "schema_version": shape1.REMOTE_TERMINAL_SCHEMA,
        "observed_at": _now(),
        "package_sha256": package["package_sha256"],
        "submission_receipt_sha256": submission["receipt_sha256"],
        "task_id": BASELINE_TASK_ID,
        "scheduler_status": scheduler_status,
        "scheduler_detail_get_count": 2,
        "scheduler_remote_file_get_count": len(raw) + int(result_bytes is not None),
        "remote_files": records,
        "batch_manifest_sha256": manifest["manifest_sha256"],
        "task_status_sha256": task_status["status_sha256"],
        "child_receipt_sha256": child_receipt["receipt_sha256"],
        "terminal_cpu_evidence": metrics,
        "terminal_cpu_evidence_sha256": metrics["terminal_cpu_evidence_sha256"],
        "promotion_eligible": (
            scheduler_status == "completed" and metrics["promotion_eligible"] is True
        ),
        "shape4_submission_allowed": (
            scheduler_status == "completed" and metrics["promotion_eligible"] is True
        ),
        "scheduler_access": "GET-only",
        "scheduler_post_count": 0,
        "scheduler_cancel_count": 0,
        "scheduler_preempt_count": 0,
        "remote_access": "read-only",
        "remote_write_count": 0,
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    value = {
        **unsigned,
        "remote_terminal_sha256": canonical_sha256(unsigned),
    }
    validate_baseline_terminal(
        value,
        config=config,
        baseline_package=package,
        baseline_submission=submission,
    )
    _atomic_json(output_path, value)
    return value


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--offload-plan", type=Path, required=True)
    parser.add_argument("--publication-receipt", type=Path, required=True)
    parser.add_argument("--baseline-package", type=Path, required=True)
    parser.add_argument("--baseline-submission", type=Path, required=True)
    parser.add_argument("--baseline-terminal", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    attest = sub.add_parser(
        "attest-baseline", help="GET and seal the pinned 4-CPU terminal baseline"
    )
    attest.add_argument("--config", type=Path, required=True)
    attest.add_argument("--baseline-package", type=Path, required=True)
    attest.add_argument("--baseline-submission", type=Path, required=True)
    attest.add_argument("--scheduler-url", default="http://127.0.0.1:8002")
    attest.add_argument("--out", type=Path, required=True)
    render = sub.add_parser("render", help="render one immutable local package")
    submit_parser = sub.add_parser(
        "submit", help="dry-run by default; exactly one POST only with --apply"
    )
    evaluate = sub.add_parser(
        "evaluate", help="seal local terminal telemetry/result evidence"
    )
    evaluate_remote = sub.add_parser(
        "evaluate-remote", help="GET and seal the submitted candidate terminal"
    )
    for command in (render, submit_parser, evaluate, evaluate_remote):
        _add_source_arguments(command)
    for command in (submit_parser, evaluate, evaluate_remote):
        command.add_argument("--package", type=Path, required=True)
    render.add_argument("--out", type=Path, required=True)
    render.add_argument("--scheduler-url", default="http://127.0.0.1:8002")
    submit_parser.add_argument("--scheduler-url", default="http://127.0.0.1:8002")
    submit_parser.add_argument("--accounts", type=Path, default=DEFAULT_ACCOUNTS)
    submit_parser.add_argument(
        "--scheduler-source", type=Path, default=DEFAULT_SCHEDULER_SOURCE
    )
    submit_parser.add_argument("--publication-account", default="harry261")
    submit_parser.add_argument("--apply", action="store_true")
    submit_parser.add_argument("--receipt-out", type=Path)
    submit_parser.add_argument("--dry-run-out", type=Path)
    evaluate.add_argument("--task-id", type=int, required=True)
    evaluate.add_argument("--telemetry", type=Path, required=True)
    evaluate.add_argument("--seed-status", type=Path, required=True)
    evaluate.add_argument("--result", type=Path, required=True)
    evaluate.add_argument("--out", type=Path, required=True)
    evaluate.add_argument("--scheduler-url", default="http://127.0.0.1:8002")
    evaluate_remote.add_argument("--submission-receipt", type=Path, required=True)
    evaluate_remote.add_argument("--scheduler-url", default="http://127.0.0.1:8002")
    evaluate_remote.add_argument("--task-id", type=int, required=True)
    evaluate_remote.add_argument("--out", type=Path, required=True)
    return parser


def _source_kwargs(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "config_path": args.config,
        "offload_plan_path": args.offload_plan,
        "publication_receipt_path": args.publication_receipt,
        "baseline_package_path": args.baseline_package,
        "baseline_submission_path": args.baseline_submission,
        "baseline_terminal_path": args.baseline_terminal,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.operation == "attest-baseline":
        value = attest_baseline(
            config_path=args.config,
            baseline_package_path=args.baseline_package,
            baseline_submission_path=args.baseline_submission,
            scheduler_url=args.scheduler_url,
            output_path=args.out,
        )
    elif args.operation == "render":
        value = build_package(**_source_kwargs(args), scheduler_url=args.scheduler_url)
        _atomic_json(args.out, value)
    elif args.operation == "submit":
        value = submit(
            **_source_kwargs(args),
            package_path=args.package,
            scheduler_url=args.scheduler_url,
            accounts_path=args.accounts,
            scheduler_source=args.scheduler_source,
            publication_account=args.publication_account,
            apply=args.apply,
            receipt_out=args.receipt_out,
            dry_run_out=args.dry_run_out,
        )
    elif args.operation == "evaluate":
        value = evaluate_local_terminal(
            **_source_kwargs(args),
            package_path=args.package,
            task_id=args.task_id,
            telemetry_path=args.telemetry,
            seed_status_path=args.seed_status,
            result_path=args.result,
            output_path=args.out,
            scheduler_url=args.scheduler_url,
        )
    else:
        value = evaluate_remote_terminal(
            **_source_kwargs(args),
            package_path=args.package,
            submission_receipt_path=args.submission_receipt,
            scheduler_url=args.scheduler_url,
            task_id=args.task_id,
            output_path=args.out,
        )
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
