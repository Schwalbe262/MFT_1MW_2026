"""GET-only updater for the post-deadline MFT task cards on port 8010.

The Scheduler remains a separate service.  This module only performs exact
``GET /api/tasks/{id}`` reads and atomically merges lifecycle state into the
explicit Codex UI status artifact.  Scheduler success is never interpreted as
scientific, collection, canonical, or production success.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


DEFAULT_SCHEDULER_URL = "http://127.0.0.1:8002"
DEFAULT_STATUS_FILE = Path(
    r"C:\Users\peets\slurm_scheduler_runtime"
    r"\mft_goal_20260726\ui\codex-work-status.json"
)
DEFAULT_LOCAL_SYMMETRIC_SELECTION_STATE_FILE = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\local_symmetric_selection_watch_v1\state.json"
)
DEFAULT_INTERVAL_SECONDS = 60
MAX_RESPONSE_BYTES = 1024 * 1024
CAMPAIGN_SUBMITTED_FLOOR = 126
SYNC_KEY = "postdeadline_task_sync"
SYNC_SCHEMA = "mft-goal-postdeadline-ui-sync-v1"
PID_SCHEMA = "mft-goal-postdeadline-ui-updater-pid-v1"
LOG_SCHEMA = "mft-goal-postdeadline-ui-updater-event-v1"
STATUS_SCHEMA = "mft-codex-work-status-v1"
POSTSUCCESS_STATE_SCHEMA = "mft-goal-postdeadline-standard-postsuccess-state-v1"
THERMAL_BRIDGE_STATE_SCHEMA = "mft-corrected-thermal-terminal-transport-watch-state-v1"
STANDARD_FULL_CONTINUATION_STATE_SCHEMA = "mft-goal-standard-full-continuation-state-v1"
LOCAL_SYMMETRIC_SELECTION_STATE_SCHEMA = (
    "mft-goal-local-symmetric-selection-watch-state-v1"
)
FINAL_GATE_PENDING_SCHEMA = "mft-goal-final-solver-package-pending-v1"
FINAL_GATE_SEAL_SCHEMA = "mft-goal-final-solver-package-seal-v1"
FINAL_PACKAGE_NAME = "final_solver_package_v1"
SUBMITTED_PATTERN = re.compile(r"\bSUBMITTED\s+(\d+)(?!\d)", re.IGNORECASE)
COLLECTIONS_PATTERN = re.compile(r"\bCOLLECTIONS?\s+(\d+)(?!\d)", re.IGNORECASE)


class UpdaterError(RuntimeError):
    """Raised when the GET-only updater cannot safely merge a cycle."""


class ConcurrentStatusUpdate(UpdaterError):
    """Raised when another producer changed the status during a merge."""


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    card_id: str
    task_name: str
    model_label: str
    candidate_label: str
    requested_node: str
    cpus: int
    memory_mb: int
    timeout_seconds: int
    inner_solver_seconds: int | None = None
    requested_account: str | None = None
    max_workers_per_node: int | None = None
    expected_same_node_as_task_id: int = 0
    search_only: bool = False
    submission_receipt_sha256: str | None = None
    final_seal_sha256: str | None = None
    expected_allocation_id: int | None = None
    expected_slurm_job_id: str | None = None
    allocation_force_cancel_at_kst: str | None = None
    task_timeout_at_kst: str | None = None
    force_cancel_lead_seconds: int | None = None
    terminal_failure_override: str | None = None
    selection_superseded_by_task_id: int | None = None
    selection_failover_for_task_id: int | None = None


TASK_SPECS = (
    TaskSpec(
        task_id=96324,
        card_id="postdeadline-symmetric-retry-96324",
        task_name=(
            "mft-goal-corrected-thermal-l96230-b7c30cb70b95-postdeadline-r6-n111"
        ),
        model_label="SYMMETRIC",
        candidate_label="b7c30cb70b95",
        requested_node="n111",
        cpus=8,
        memory_mb=294912,
        timeout_seconds=45000,
        inner_solver_seconds=43200,
        requested_account="r1jae262",
        terminal_failure_override=(
            "Icepak native ThermalSetup execution error after an authenticated "
            "mesh preflight; no Fluent process or temperature result was "
            "produced. The later NaN JSON serialization error only affected "
            "the failure receipt."
        ),
    ),
    TaskSpec(
        task_id=96325,
        card_id="postdeadline-standard-retry-96325",
        task_name=("mft-goal-diag-standard-postdeadline-r1-l96231-efffb6518d4e-n107"),
        model_label="STANDARD",
        candidate_label="efffb6518d4e",
        requested_node="n107",
        cpus=8,
        memory_mb=98304,
        timeout_seconds=45300,
        search_only=True,
    ),
    TaskSpec(
        task_id=96326,
        card_id="postdeadline-full-retry-96326",
        task_name="mft-goal-postdeadline-diagnostic-full-retry-t96307-v1",
        model_label="FULL",
        candidate_label="task96307 retry",
        requested_node="n116",
        cpus=16,
        memory_mb=98304,
        timeout_seconds=86400,
        inner_solver_seconds=79200,
        requested_account="dhj02",
    ),
    TaskSpec(
        task_id=96327,
        card_id="postdeadline-standard-retry-96327",
        task_name=("mft-goal-diag-standard-postdeadline-r2-l96208-b6a83bfc7212-n109"),
        model_label="STANDARD",
        candidate_label="b6a83bfc7212",
        requested_node="n109",
        cpus=8,
        memory_mb=98304,
        timeout_seconds=45300,
        search_only=True,
    ),
    TaskSpec(
        task_id=96328,
        card_id="postdeadline-standard-official6-96328",
        task_name=(
            "mft-goal-diag-standard-postdeadline-official6-s96185-2772aed82a8c-n113"
        ),
        model_label="STANDARD OFFICIAL #6",
        candidate_label="official#6 2772aed82a8c",
        requested_node="n113",
        cpus=8,
        memory_mb=98304,
        timeout_seconds=45300,
        inner_solver_seconds=43200,
        requested_account="dw16",
        max_workers_per_node=1,
        search_only=True,
        submission_receipt_sha256=(
            "42942394a873e40181f9074f2625807224239b291bcf2f4cf2ccacde50b11edc"
        ),
        final_seal_sha256=(
            "ff034d51e8da2ce97c4cd06f44767131e6d60c01f62d65d58ac0afc9bee4abe6"
        ),
        expected_allocation_id=14620,
        expected_slurm_job_id="829579",
        allocation_force_cancel_at_kst="2026-07-27T04:07:51+09:00",
        task_timeout_at_kst="2026-07-27T08:09:47+09:00",
        force_cancel_lead_seconds=14516,
    ),
    TaskSpec(
        task_id=96329,
        card_id="postdeadline-standard-official8-96329",
        task_name=(
            "mft-goal-diag-standard-postdeadline-official8-s96141-622097dde126-n114"
        ),
        model_label="STANDARD OFFICIAL #8",
        candidate_label="official#8 622097dde126",
        requested_node="n114",
        cpus=8,
        memory_mb=98304,
        timeout_seconds=45300,
        inner_solver_seconds=43200,
        requested_account="jji0930",
        max_workers_per_node=1,
        search_only=True,
        submission_receipt_sha256=(
            "8d7e59fe51523f22e3692ef87ee529d619b4a18b5e0a5353664bef9a73500ff5"
        ),
        final_seal_sha256=(
            "5319a8a4dceb27082b91fc6badd540221298eeb9f324e2fa30e76e529313d3ec"
        ),
        selection_superseded_by_task_id=96333,
    ),
    TaskSpec(
        task_id=96330,
        card_id="postdeadline-standard-official1-96330",
        task_name=(
            "mft-goal-diag-standard-postdeadline-official1-s96009-896084a59793-n110"
        ),
        model_label="STANDARD OFFICIAL #1",
        candidate_label="official#1 896084a59793",
        requested_node="n110",
        cpus=8,
        memory_mb=98304,
        timeout_seconds=45300,
        inner_solver_seconds=43200,
        requested_account="dhj02",
        max_workers_per_node=1,
        search_only=True,
        submission_receipt_sha256=(
            "d71987f49b062132fdf90a358bcc819a2cf9fd2e74c76fecad04e22ea9329233"
        ),
        final_seal_sha256=(
            "1c511227687977cbf001a0e4a77c5060d41fcd397b74756062bc6e9feb0f9332"
        ),
    ),
    TaskSpec(
        task_id=96331,
        card_id="postdeadline-standard-official12-96331",
        task_name=(
            "mft-goal-diag-standard-postdeadline-official12-s96185-828cb282cf4f-n112"
        ),
        model_label="STANDARD OFFICIAL #12",
        candidate_label="official#12 828cb282cf4f",
        requested_node="n112",
        cpus=8,
        memory_mb=98304,
        timeout_seconds=45300,
        inner_solver_seconds=43200,
        requested_account="r1jae262",
        max_workers_per_node=1,
        search_only=True,
        submission_receipt_sha256=(
            "2e207daa33165e8921816e56beb2458d1dcd8515d6cb0b8d5a25cb0cf3cde532"
        ),
        final_seal_sha256=(
            "a743046a7a878030a538988e1c5e9270de1ef56a8e541e0130c5b55cc764937a"
        ),
    ),
    TaskSpec(
        task_id=96332,
        card_id="postdeadline-standard-official5-96332",
        task_name=(
            "mft-goal-diag-standard-postdeadline-official5-s95913-909d249ebe45-n115"
        ),
        model_label="STANDARD OFFICIAL #5",
        candidate_label="official#5 909d249ebe45",
        requested_node="n115",
        cpus=8,
        memory_mb=98304,
        timeout_seconds=45300,
        inner_solver_seconds=43200,
        requested_account="jji0930",
        max_workers_per_node=1,
        search_only=True,
        submission_receipt_sha256=(
            "4275d9e8e004f3f9f57d9483ca9e7b8a48d410778f39a80848c95c396ed2171c"
        ),
        final_seal_sha256=(
            "ed28dd4d0c4ec904d2910323bc63bee2e13afddb59cd8e4d7f1a9d3d11487477"
        ),
        selection_superseded_by_task_id=96338,
    ),
    TaskSpec(
        task_id=96333,
        card_id="postdeadline-standard-official8-failover-96333",
        task_name=(
            "mft-goal-diag-standard-postdeadline-official8-failover-"
            "s96141-622097dde126-n111"
        ),
        model_label="STANDARD OFFICIAL #8 FAILOVER",
        candidate_label="official#8 622097dde126 n111 failover",
        requested_node="n111",
        cpus=8,
        memory_mb=98304,
        timeout_seconds=45300,
        inner_solver_seconds=43200,
        requested_account="r1jae262",
        max_workers_per_node=1,
        search_only=True,
        submission_receipt_sha256=(
            "8990b3f339ce36f66d4ecf5fdbbd111889d55b25d860d00e3d98e6508101180c"
        ),
        final_seal_sha256=(
            "d70eef5add63daa0148ad05af809dd231d1400a11794995a0fb81a3a6fedd45f"
        ),
        selection_failover_for_task_id=96329,
    ),
    TaskSpec(
        task_id=96337,
        card_id="postdeadline-standard-official5-direct-analyze-96337",
        task_name=(
            "mft-goal-diag-standard-official5-direct-analyze-samenode-v1-"
            "909d249ebe45-n115"
        ),
        model_label="STANDARD OFFICIAL #5 DIRECT ANALYZE ATTEMPT",
        candidate_label="official#5 909d249ebe45 same-node direct Analyze attempt",
        requested_node="n115",
        cpus=8,
        memory_mb=98304,
        timeout_seconds=45300,
        inner_solver_seconds=43200,
        requested_account="jji0930",
        max_workers_per_node=2,
        expected_same_node_as_task_id=96332,
        search_only=True,
        submission_receipt_sha256=(
            "c14d491f558308d14c1a151a0a9e63c91997921a09960e858cc495b66652bea4"
        ),
        final_seal_sha256=(
            "fb417009a6af6ce6a734f41ffe7855f4b3d9294454d93e2066c8f06b619448a3"
        ),
        expected_allocation_id=14650,
        expected_slurm_job_id="840582",
        terminal_failure_override=(
            "Pre-EM AEDT startup failed because the standalone-core opt-in "
            "authentication digest did not match. No electromagnetic or thermal "
            "solve result was produced; this is an operational failure, not a "
            "physics infeasibility."
        ),
        selection_superseded_by_task_id=96338,
    ),
    TaskSpec(
        task_id=96338,
        card_id="postdeadline-standard-official5-direct-analyze-r1-96338",
        task_name=(
            "mft-goal-diag-standard-official5-direct-analyze-samenode-r1-v2-"
            "909d249ebe45-n115"
        ),
        model_label="STANDARD OFFICIAL #5 DIRECT ANALYZE CORRECTED",
        candidate_label="official#5 909d249ebe45 corrected same-node direct Analyze",
        requested_node="n115",
        cpus=8,
        memory_mb=98304,
        timeout_seconds=45300,
        inner_solver_seconds=43200,
        requested_account="jji0930",
        max_workers_per_node=2,
        expected_same_node_as_task_id=96332,
        search_only=True,
        submission_receipt_sha256=(
            "0731cf22e3f78f93da943cc9102b7d96a2719bfbe1ca65e782789b2d98bc30dd"
        ),
        final_seal_sha256=(
            "4e2802cd321727114926da7af8631d8f267fab37fa801d966966fdb54b75946e"
        ),
        expected_allocation_id=14650,
        expected_slurm_job_id="840582",
        selection_failover_for_task_id=96332,
    ),
)

LEGACY_STANDARD_SELECTION_TASK_IDS = (
    96325,
    96327,
    96328,
    96330,
    96331,
    96332,
    96333,
)
STANDARD_SELECTION_TASK_IDS = (
    96325,
    96327,
    96328,
    96333,
    96330,
    96331,
    96338,
)
STANDARD_SELECTION_LIFECYCLE_TASK_IDS = (
    *LEGACY_STANDARD_SELECTION_TASK_IDS,
    96337,
    96338,
)
STANDARD_SELECTION_OVERLAY_TASK_IDS = (96337, 96338)
STANDARD_SELECTION_SUPERSEDED_TASK_IDS = (96332, 96337)
FULL_REFERENCE_TASK_ID = 96326
SELECTION_POLICY_CARD_ID = "codex-symmetric-primary-selection-policy"
LOCAL_SYMMETRIC_SELECTION_CARD_ID = "codex-local-symmetric-selection"
LEGACY_CONTINUATION_CARD_ID = "codex-standard-full-continuation"

RUNNING_STATES = {"running"}
QUEUED_STATES = {
    "queued",
    "pending",
    "attaching",
    "attached",
    "assigned",
    "launching",
    "starting",
}
SUCCESS_STATES = {"succeeded", "completed", "success"}
FAILURE_STATES = {
    "failed",
    "cancelled",
    "canceled",
    "timed_out",
    "timeout",
}

TaskReader = Callable[[str, int], Mapping[str, Any]]


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "payload_sha256" in result:
        raise UpdaterError("payload is already sealed")
    result["payload_sha256"] = canonical_sha256(result)
    return result


def validate_seal(value: Any, schema: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise UpdaterError("sealed value must be an object")
    unsigned = copy.deepcopy(value)
    observed = unsigned.pop("payload_sha256", None)
    if value.get("schema_version") != schema:
        raise UpdaterError(f"{schema} schema mismatch")
    if observed != canonical_sha256(unsigned):
        raise UpdaterError(f"{schema} payload seal mismatch")
    return value


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(
    path: Path,
    value: Any,
    *,
    expected_sha256: str | None = None,
) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        if expected_sha256 is not None:
            if not target.is_file() or _file_sha256(target) != expected_sha256:
                raise ConcurrentStatusUpdate(
                    "status changed while the Scheduler snapshot was merged"
                )
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


class SingleInstanceLock:
    """Process-lifetime advisory lock released automatically after a crash."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        self.stream: Any | None = None

    def __enter__(self) -> "SingleInstanceLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        self.stream.seek(0, os.SEEK_END)
        if self.stream.tell() == 0:
            self.stream.write(b"\0")
            self.stream.flush()
            os.fsync(self.stream.fileno())
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            self.stream.close()
            self.stream = None
            raise UpdaterError(f"another updater holds the lock: {self.path}") from exc
        return self

    def __exit__(self, _kind: Any, _value: Any, _traceback: Any) -> None:
        if self.stream is None:
            return
        try:
            self.stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        finally:
            self.stream.close()
            self.stream = None


def _get_scheduler_task(
    scheduler_url: str,
    task_id: int,
    *,
    timeout_seconds: float = 10.0,
) -> Mapping[str, Any]:
    url = f"{scheduler_url.rstrip('/')}/api/tasks/{task_id}"
    request = urllib.request.Request(
        url, headers={"Accept": "application/json"}, method="GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) > MAX_RESPONSE_BYTES:
                raise UpdaterError("Scheduler GET response exceeds size bound")
            payload = response.read(MAX_RESPONSE_BYTES + 1)
    except (OSError, urllib.error.URLError) as exc:
        raise UpdaterError(f"Scheduler GET failed for task{task_id}") from exc
    if len(payload) > MAX_RESPONSE_BYTES:
        raise UpdaterError("Scheduler GET response exceeds size bound")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise UpdaterError("Scheduler GET returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise UpdaterError("Scheduler task response must be an object")
    return value


def _task_state(task: Mapping[str, Any]) -> str:
    for key in ("state", "status", "queue_state"):
        value = str(task.get(key) or "").strip().lower()
        if value:
            if value in RUNNING_STATES | QUEUED_STATES:
                return value
            if value in SUCCESS_STATES | FAILURE_STATES:
                return value
    raise UpdaterError("Scheduler task has an unsupported lifecycle state")


def _positive_or_none(value: Any, label: str) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        raise UpdaterError(f"{label} has invalid type")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise UpdaterError(f"{label} is not an integer") from exc
    if result <= 0:
        raise UpdaterError(f"{label} must be positive")
    return result


def _validate_task(spec: TaskSpec, task: Mapping[str, Any]) -> dict[str, Any]:
    identifiers = {
        int(value)
        for key in ("id", "task_id")
        if (value := task.get(key)) not in (None, "")
    }
    if identifiers != {spec.task_id}:
        raise UpdaterError(f"task{spec.task_id} identity drifted")
    if task.get("name") != spec.task_name:
        raise UpdaterError(f"task{spec.task_id} name drifted")
    expected = {
        "cpus": spec.cpus,
        "memory_mb": spec.memory_mb,
        "timeout_seconds": spec.timeout_seconds,
        "same_node_as_task_id": spec.expected_same_node_as_task_id,
    }
    if spec.max_workers_per_node is not None:
        expected["max_workers_per_node"] = spec.max_workers_per_node
    for key, value in expected.items():
        if task.get(key) != value:
            raise UpdaterError(f"task{spec.task_id} {key} drifted")
    requested_node = task.get("requested_node_name") or task.get("node_name")
    if requested_node != spec.requested_node:
        raise UpdaterError(f"task{spec.task_id} requested node drifted")
    if task.get("node_name_policy") != "strict":
        raise UpdaterError(f"task{spec.task_id} is not strict-node")
    if spec.requested_account is not None:
        accounts = {
            str(task.get(key) or "")
            for key in ("requested_account_name", "account_name")
        }
        if spec.requested_account not in accounts:
            raise UpdaterError(f"task{spec.task_id} account drifted")
    state = _task_state(task)
    if (
        spec.expected_allocation_id is not None
        and task.get("allocation_id") != spec.expected_allocation_id
    ):
        raise UpdaterError(f"task{spec.task_id} corrected allocation identity drifted")
    if (
        spec.expected_slurm_job_id is not None
        and str(task.get("slurm_job_id") or "") != spec.expected_slurm_job_id
    ):
        raise UpdaterError(f"task{spec.task_id} corrected Slurm job identity drifted")
    actual_node = str(
        task.get("actual_node_name") or task.get("allocation_node_name") or ""
    )
    if state in RUNNING_STATES and (
        actual_node != spec.requested_node
        or task.get("placement_contract_satisfied") is not True
    ):
        raise UpdaterError(f"task{spec.task_id} placement is not satisfied")
    return {
        "task_id": spec.task_id,
        "state": state,
        "allocation_id": _positive_or_none(task.get("allocation_id"), "allocation_id"),
        "slurm_job_id": str(task.get("slurm_job_id") or ""),
        "account_name": str(task.get("account_name") or ""),
        "actual_node_name": actual_node,
        "created_at": task.get("created_at"),
        "started_at": task.get("started_at"),
        "finished_at": task.get("finished_at"),
        "exit_code": task.get("exit_code"),
        "failure_message": str(task.get("failure_message") or "")[:350],
    }


def fetch_tasks(
    scheduler_url: str,
    *,
    task_reader: TaskReader | None = None,
) -> dict[int, dict[str, Any]]:
    reader = task_reader or _get_scheduler_task
    with ThreadPoolExecutor(max_workers=len(TASK_SPECS)) as executor:
        futures = {
            spec.task_id: executor.submit(reader, scheduler_url, spec.task_id)
            for spec in TASK_SPECS
        }
        raw = {task_id: future.result() for task_id, future in futures.items()}
    return {
        spec.task_id: _validate_task(spec, raw[spec.task_id]) for spec in TASK_SPECS
    }


def _category(state: str) -> str:
    if state in RUNNING_STATES:
        return "running"
    if state in QUEUED_STATES:
        return "queued"
    if state in SUCCESS_STATES:
        return "succeeded"
    if state in FAILURE_STATES:
        return "failed"
    raise UpdaterError(f"unsupported normalized state: {state}")


def _aware_timestamp(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise UpdaterError(f"{label} is not ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise UpdaterError(f"{label} must include a UTC offset")
    return parsed


def _duration_text(seconds: int) -> str:
    value = max(0, int(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s"


def _force_cancel_risk(
    spec: TaskSpec,
    *,
    category: str,
    observed_at: str,
) -> dict[str, Any] | None:
    fields = (
        spec.allocation_force_cancel_at_kst,
        spec.task_timeout_at_kst,
        spec.force_cancel_lead_seconds,
    )
    if fields == (None, None, None):
        return None
    if any(value is None for value in fields):
        raise UpdaterError(
            f"task{spec.task_id} force-cancel risk contract is incomplete"
        )
    force_at = _aware_timestamp(
        str(spec.allocation_force_cancel_at_kst),
        f"task{spec.task_id} allocation force boundary",
    )
    timeout_at = _aware_timestamp(
        str(spec.task_timeout_at_kst),
        f"task{spec.task_id} timeout boundary",
    )
    observed = _aware_timestamp(observed_at, "observed_at")
    lead_seconds = int((timeout_at - force_at).total_seconds())
    if (
        lead_seconds != spec.force_cancel_lead_seconds
        or lead_seconds <= 0
        or spec.expected_allocation_id is None
    ):
        raise UpdaterError(f"task{spec.task_id} force-cancel risk timing drifted")
    remaining_seconds = int((force_at - observed).total_seconds())
    if category in {"running", "queued"}:
        lifecycle_note = (
            f"{_duration_text(remaining_seconds)} remaining"
            if remaining_seconds > 0
            else "boundary passed; live Scheduler state remains authoritative"
        )
        active = True
    else:
        lifecycle_note = "terminal lifecycle; historical operational risk"
        active = False
    return {
        "allocation_id": spec.expected_allocation_id,
        "force_at": force_at,
        "timeout_at": timeout_at,
        "lead_seconds": lead_seconds,
        "remaining_seconds": remaining_seconds,
        "lifecycle_note": lifecycle_note,
        "active": active,
    }


def _task_card(
    spec: TaskSpec,
    task: Mapping[str, Any],
    observed_at: str,
) -> dict[str, Any]:
    category = _category(str(task["state"]))
    node = task["actual_node_name"] or spec.requested_node
    if category == "succeeded":
        stage = "TERMINAL SUCCEEDED · COLLECTION/PASS PENDING"
        progress = 100
    elif category == "failed":
        stage = f"TERMINAL {str(task['state']).upper()}"
        progress = 100
    elif category == "running":
        stage = "RUNNING"
        progress = 5
    else:
        stage = "QUEUED"
        progress = 0
    force_risk = _force_cancel_risk(
        spec,
        category=category,
        observed_at=observed_at,
    )
    risk_title = ""
    if force_risk is not None and force_risk["active"]:
        risk_title = f" · FORCE-CANCEL RISK {force_risk['force_at']:%m-%d %H:%M:%S KST}"
    title = (
        f"POST-DEADLINE {spec.model_label} · task{spec.task_id} {stage} · "
        f"{node}{risk_title}"
    )
    allocation = task["allocation_id"] or "none"
    job = task["slurm_job_id"] or "none"
    lifecycle = (
        f"Scheduler GET lifecycle={task['state']}, allocation={allocation}, "
        f"Slurm job={job}, node={node}."
    )
    if category == "succeeded":
        outcome = (
            "실행 lifecycle만 terminal success입니다. 별도 collector와 "
            "artifact 인증 전에는 collection·scientific PASS가 아닙니다."
        )
    elif category == "failed":
        reason = (
            spec.terminal_failure_override
            or task["failure_message"]
            or "Scheduler failure reason unavailable"
        )
        outcome = (
            f"Terminal failure reason: {reason}. 운영 실패는 과학적 "
            "infeasibility 또는 production 판정이 아닙니다."
        )
    else:
        outcome = "아직 solver·temperature·scientific PASS 결과가 없습니다."
    detail = (
        f"{spec.candidate_label} post-deadline diagnostic/noncanonical 작업. "
        f"{lifecycle} {outcome}"
    )
    if spec.selection_superseded_by_task_id is not None:
        detail += (
            " Effective selection lane=false: "
            f"task{spec.selection_superseded_by_task_id} supersedes this queued "
            "attempt for candidate selection; this task remains visible for "
            "authenticated Scheduler lifecycle only."
        )
    if spec.selection_failover_for_task_id is not None:
        detail += (
            " Effective selection lane=true: this strict-node failover "
            f"supersedes task{spec.selection_failover_for_task_id} for "
            "candidate selection; the superseded task remains lifecycle-visible."
        )
    if force_risk is not None:
        detail += (
            f" allocation{force_risk['allocation_id']}의 source-derived "
            f"force-cancel 경계는 "
            f"{force_risk['force_at']:%Y-%m-%d %H:%M:%S KST}이며 "
            f"task timeout "
            f"{force_risk['timeout_at']:%Y-%m-%d %H:%M:%S KST}보다 "
            f"{_duration_text(force_risk['lead_seconds'])} 빠릅니다. "
            f"{force_risk['lifecycle_note']}; 이는 operational risk이며 "
            "scientific infeasibility 판정이 아닙니다."
        )
    evidence = [
        (
            f"Scheduler GET task{spec.task_id} {str(task['state']).upper()} / "
            f"allocation{allocation} / Slurm{job} / node{node}"
        ),
        (
            f"cpus{spec.cpus} / memory{spec.memory_mb}MB / "
            f"scheduler timeout{spec.timeout_seconds}s"
        ),
        (
            "strict node placement contract / same_node_as_task_id="
            f"{spec.expected_same_node_as_task_id}"
        ),
        (
            "postdeadline=true / diagnostic_only=true / noncanonical=true"
            + (" / search_only=true" if spec.search_only else "")
        ),
        (
            "scheduler_lifecycle_only=true / collection_authenticated=false / "
            "scientific_pass_generated=false / production_claim_generated=false"
        ),
    ]
    if spec.inner_solver_seconds is not None:
        evidence.insert(2, f"inner solver budget{spec.inner_solver_seconds}s")
    if spec.submission_receipt_sha256 is not None:
        evidence.append(f"submission receipt SHA256 {spec.submission_receipt_sha256}")
    if spec.final_seal_sha256 is not None:
        evidence.append(f"final seal SHA256 {spec.final_seal_sha256}")
    if spec.requested_account is not None:
        evidence.append(
            f"requested account={spec.requested_account} / "
            f"requested node={spec.requested_node} / node policy=strict"
        )
    if spec.selection_superseded_by_task_id is not None:
        evidence.append(
            "selection_lane_effective=false / "
            f"superseded_by_task{spec.selection_superseded_by_task_id}=true / "
            "lifecycle_visibility_preserved=true"
        )
    if spec.selection_failover_for_task_id is not None:
        evidence.append(
            "selection_lane_effective=true / "
            f"failover_for_task{spec.selection_failover_for_task_id}=true / "
            "unique_effective_lane=true"
        )
    if force_risk is not None:
        evidence.extend(
            [
                (
                    f"allocation{force_risk['allocation_id']} source-derived "
                    f"force-cancel boundary "
                    f"{force_risk['force_at']:%Y-%m-%d %H:%M:%S KST}"
                ),
                (
                    f"task timeout boundary "
                    f"{force_risk['timeout_at']:%Y-%m-%d %H:%M:%S KST} / "
                    f"force boundary leads by "
                    f"{_duration_text(force_risk['lead_seconds'])} / "
                    "hard residual guarantee=false / operational risk only / "
                    "scientific infeasibility=false"
                ),
            ]
        )
    if task["failure_message"]:
        evidence.append(f"failure_message={task['failure_message']}")
    if category == "failed" and spec.terminal_failure_override:
        evidence.append(
            f"authenticated terminal root cause={spec.terminal_failure_override}"
        )
    return {
        "id": spec.card_id,
        "title": title,
        "detail": detail,
        "state": "in_progress",
        "updated_at": observed_at,
        "progress_pct": progress,
        "evidence": evidence,
    }


def _single_current(payload: Mapping[str, Any], item_id: str) -> dict[str, Any]:
    current = payload.get("current")
    if not isinstance(current, list):
        raise UpdaterError("status current group is missing")
    matches = [item for item in current if item.get("id") == item_id]
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise UpdaterError(f"status item {item_id} is not unique")
    return matches[0]


def _read_sealed_local_json(
    path: Path,
    *,
    schema: str,
    schema_field: str = "schema_version",
    canonical_ensure_ascii: bool = False,
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or resolved.stat().st_size > MAX_RESPONSE_BYTES:
        raise UpdaterError(f"automation state is unsafe: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise UpdaterError(f"automation state is invalid: {resolved}") from exc
    if not isinstance(value, dict):
        raise UpdaterError(f"automation state is not an object: {resolved}")
    unsigned = copy.deepcopy(value)
    observed = unsigned.pop("payload_sha256", None)
    expected = (
        hashlib.sha256(
            json.dumps(
                unsigned,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if canonical_ensure_ascii
        else canonical_sha256(unsigned)
    )
    if value.get(schema_field) != schema or observed != expected:
        raise UpdaterError(f"automation state seal drifted: {resolved}")
    return value


def _upsert_current_card(payload: dict[str, Any], card: Mapping[str, Any]) -> None:
    current = payload.get("current")
    if not isinstance(current, list):
        raise UpdaterError("status current group is missing")
    card_id = card.get("id")
    matches = [
        index
        for index, item in enumerate(current)
        if isinstance(item, dict) and item.get("id") == card_id
    ]
    if len(matches) > 1:
        raise UpdaterError(f"automation card {card_id} is duplicated")
    if matches:
        current[matches[0]] = copy.deepcopy(dict(card))
        return
    insertion = next(
        (
            index
            for index, item in enumerate(current)
            if isinstance(item, dict) and item.get("id") == "parallel-workstreams"
        ),
        len(current),
    )
    current.insert(insertion, copy.deepcopy(dict(card)))


def _remove_current_card(payload: dict[str, Any], card_id: str) -> None:
    current = payload.get("current")
    if not isinstance(current, list):
        raise UpdaterError("status current group is missing")
    matches = [
        index
        for index, item in enumerate(current)
        if isinstance(item, dict) and item.get("id") == card_id
    ]
    if len(matches) > 1:
        raise UpdaterError(f"automation card {card_id} is duplicated")
    if matches:
        current.pop(matches[0])


def _selection_lifecycle_overlay(
    tasks: Mapping[int, Mapping[str, Any]] | None,
) -> tuple[int, int]:
    """Return uncollected-pending and operational-failure overlay counts."""
    if tasks is None:
        return 0, 0
    pending = 0
    failures = 0
    for task_id in STANDARD_SELECTION_OVERLAY_TASK_IDS:
        if task_id not in tasks:
            raise UpdaterError(f"selection lifecycle task{task_id} is missing")
        category = _category(str(tasks[task_id]["state"]))
        if category == "failed":
            failures += 1
        else:
            # The sealed source collector does not yet authenticate overlay lanes.
            pending += 1
    return pending, failures


def _symmetric_primary_policy_card(
    tasks: Mapping[int, Mapping[str, Any]],
    observed_at: str,
) -> dict[str, Any]:
    lane_categories = {
        task_id: _category(str(tasks[task_id]["state"]))
        for task_id in STANDARD_SELECTION_TASK_IDS
    }
    terminal_count = sum(
        category in {"succeeded", "failed"} for category in lane_categories.values()
    )
    full_category = _category(str(tasks[FULL_REFERENCE_TASK_ID]["state"]))
    lane_ids = ",".join(f"task{task_id}" for task_id in STANDARD_SELECTION_TASK_IDS)
    lane_lifecycle = " / ".join(
        f"task{task_id}:{lane_categories[task_id].upper()}"
        for task_id in STANDARD_SELECTION_TASK_IDS
    )
    lifecycle = " / ".join(
        f"task{task_id}:{_category(str(tasks[task_id]['state'])).upper()}"
        for task_id in STANDARD_SELECTION_LIFECYCLE_TASK_IDS
    )
    return {
        "id": SELECTION_POLICY_CARD_ID,
        "title": (
            "DESIGN SELECTION | SYMMETRY/STANDARD PRIMARY | "
            f"TERMINAL {terminal_count}/7 | AUTO FULL OFF"
        ),
        "detail": (
            "현재 NSGA-II search solution의 후보 선택은 인증된 "
            "symmetry/Standard FEA 결과를 우선 기준으로 합니다. 후보별 "
            "Standard-to-Full 자동 연쇄는 중지되어 있습니다. 기존 Full "
            "task96326은 diagnostic reference로만 계속되며 설계 선택이나 "
            "승격 근거가 아닙니다. symmetry 결과로 한 후보를 명시적으로 "
            "선택한 뒤 최대 한 후보만 최종 Full 검증할 수 있습니다. 인증된 "
            "actual 결과 전에는 scientific/production PASS를 주장하지 않습니다."
        ),
        "state": "in_progress",
        "updated_at": observed_at,
        "progress_pct": 50 + (terminal_count * 35 // 7),
        "evidence": [
            "primary candidate-selection gate=authenticated symmetric/Standard FEA",
            f"Standard selection lanes=7 / {lane_ids}",
            f"lane lifecycle={lane_lifecycle}",
            f"selection lifecycle observations=9 / {lifecycle}",
            "automatic Standard-to-Full per candidate=false",
            (
                "effective official#8 selection lane=task96333 / "
                "task96329 superseded but lifecycle-visible"
            ),
            (
                "effective official#5 selection lane=task96338 / "
                "task96332 superseded but lifecycle-visible / "
                "task96337 pre-EM operational failure only"
            ),
            (
                f"task96326 lifecycle={full_category.upper()} / "
                "role=diagnostic reference only"
            ),
            "final explicit Full validation candidate cap=1",
            (
                "policy card is not result evidence / actual scientific PASS=0 / "
                "actual production PASS=0 / canonical promotion=false / "
                "production truth=false"
            ),
        ],
    }


def _postsuccess_card(
    path: Path,
    observed_at: str,
    tasks: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    state = _read_sealed_local_json(
        path,
        schema=POSTSUCCESS_STATE_SCHEMA,
    )
    collection_count = int(state.get("collection_count") or 0)
    pending_count = int(state.get("pending_count") or 0)
    failure_count = int(state.get("terminal_failure_count") or 0)
    authoritative_lifecycle = "lifecycle_task_ids" in state
    if authoritative_lifecycle:
        lifecycle_ids = state.get("lifecycle_task_ids")
        effective_ids = state.get("effective_task_ids")
        superseded_ids = state.get("selection_superseded_task_ids")
        lanes = state.get("lanes")
        lifecycle_collection_count = int(state.get("lifecycle_collection_count") or 0)
        watcher_pending = int(state.get("lifecycle_pending_count") or 0)
        watcher_failures = int(state.get("lifecycle_terminal_failure_count") or 0)
        if (
            not isinstance(lifecycle_ids, list)
            or not isinstance(effective_ids, list)
            or not isinstance(superseded_ids, list)
            or len(lifecycle_ids) != 9
            or len(set(lifecycle_ids)) != 9
            or set(lifecycle_ids) != set(STANDARD_SELECTION_LIFECYCLE_TASK_IDS)
            or len(effective_ids) != 7
            or len(set(effective_ids)) != 7
            or set(effective_ids) != set(STANDARD_SELECTION_TASK_IDS)
            or set(superseded_ids) != set(STANDARD_SELECTION_SUPERSEDED_TASK_IDS)
            or not isinstance(lanes, list)
            or len(lanes) != 9
            or collection_count + pending_count + failure_count != 7
            or lifecycle_collection_count + watcher_pending + watcher_failures != 9
        ):
            raise UpdaterError("post-success lifecycle contract drifted")
        lane_status: dict[int, str] = {}
        for lane in lanes:
            if not isinstance(lane, Mapping):
                raise UpdaterError("post-success lifecycle lane drifted")
            task_id = lane.get("task_id")
            status = str(lane.get("status") or "")
            if (
                isinstance(task_id, bool)
                or not isinstance(task_id, int)
                or task_id not in STANDARD_SELECTION_LIFECYCLE_TASK_IDS
                or task_id in lane_status
                or status not in {"pending", "collection_ready", "terminal_failure"}
            ):
                raise UpdaterError("post-success lifecycle lane drifted")
            lane_status[task_id] = status
        if (
            sum(status == "collection_ready" for status in lane_status.values())
            != lifecycle_collection_count
            or sum(status == "pending" for status in lane_status.values())
            != watcher_pending
            or sum(status == "terminal_failure" for status in lane_status.values())
            != watcher_failures
        ):
            raise UpdaterError("post-success lifecycle lane counts drifted")
        current_failure_ids = {
            task_id
            for task_id, status in lane_status.items()
            if status == "terminal_failure"
        }
        if tasks is not None:
            current_failure_ids.update(
                task_id
                for task_id in STANDARD_SELECTION_LIFECYCLE_TASK_IDS
                if _category(str(tasks[task_id]["state"])) == "failed"
            )
        current_collection_ids = {
            task_id
            for task_id, status in lane_status.items()
            if status == "collection_ready"
        } - current_failure_ids
        lifecycle_failures = len(current_failure_ids)
        lifecycle_pending = 9 - len(current_collection_ids) - lifecycle_failures
        lifecycle_mode = "v6-sealed-effective-plus-live-GET-lifecycle"
        watcher_snapshot = (
            f"watcher snapshot pending={watcher_pending} / "
            f"operational terminal failures={watcher_failures}"
        )
    else:
        overlay_pending, overlay_failures = _selection_lifecycle_overlay(tasks)
        lifecycle_pending = pending_count + overlay_pending
        lifecycle_failures = failure_count + overlay_failures
        lifecycle_mode = "v5-overlay-fallback"
        watcher_snapshot = "watcher snapshot=v5 effective-only"
    if (
        state.get("diagnostic_only") is not True
        or state.get("production_eligible") is not False
        or state.get("scheduler_mutation_performed") is not False
        or state.get("scientific_pass_claimed") is not False
        or state.get("production_claimed") is not False
    ):
        raise UpdaterError("post-success automation safety boundary drifted")
    card = {
        "id": "codex-standard-postsuccess-pipeline",
        "title": (
            "CODEX AUTO · STANDARD RESULT PIPELINE · "
            f"COLLECTIONS {collection_count} · PENDING {pending_count}"
        ),
        "detail": (
            "Seven symmetry/Standard collectors authenticate terminal artifacts, "
            "apply hard constraints and strict-AL admission, then prepare measured "
            "global NDS inputs. Missing measured results never create a "
            "scientific or production claim."
        ),
        "state": "in_progress",
        "updated_at": observed_at,
        "progress_pct": 70 if collection_count else 40,
        "evidence": [
            f"state={state.get('status')}",
            (
                f"collections{collection_count} / pending{pending_count} / "
                f"terminal failures{failure_count}"
            ),
            (
                "surrogate retraining="
                f"{str(bool(state.get('surrogate_retraining_performed'))).lower()}"
            ),
            (
                "production Pareto emitted="
                f"{str(bool(state.get('production_pareto_emitted'))).lower()}"
            ),
            (
                "Standard selection lanes=7 / "
                + ",".join(f"task{task_id}" for task_id in STANDARD_SELECTION_TASK_IDS)
            ),
            "Scheduler mutation=false / scientific claim=false",
            f"state SHA256 {_file_sha256(path.resolve())}",
        ],
    }
    card["title"] = (
        "CODEX AUTO | STANDARD RESULT PIPELINE | "
        f"AUTH {collection_count} | PENDING {lifecycle_pending} | ACTUAL PASS 0"
    )
    card["detail"] = (
        "Nine lifecycle attempts currently feed seven effective "
        "symmetry/Standard selection lanes. Collectors authenticate terminal "
        "artifacts, apply hard constraints and strict-AL admission, then prepare "
        "measured global NDS inputs. Missing measured results never create a "
        "scientific or production claim."
    )
    card["evidence"][1] = (
        f"authenticated={collection_count}/7 / pending={lifecycle_pending} / "
        f"operational terminal failures={lifecycle_failures}"
    )
    card["evidence"].insert(
        -2,
        f"lifecycle mode={lifecycle_mode} / attempts=9 / selection-effective=7 / "
        "task96337 failed pre-EM operationally / task96338 current",
    )
    card["evidence"].insert(-2, watcher_snapshot)
    card["evidence"].insert(-2, "actual scientific PASS=0 / actual production PASS=0")
    return card


def _local_selection_fail_safe_card(
    path: Path,
    observed_at: str,
    *,
    condition: str,
) -> dict[str, Any]:
    label = {
        "missing": "STATE MISSING",
        "unavailable": "STATE UNAVAILABLE",
        "invalid": "STATE INVALID",
    }[condition]
    return {
        "id": LOCAL_SYMMETRIC_SELECTION_CARD_ID,
        "title": (f"CODEX · SYMMETRIC FEA SELECT · {label} · AUTO FULL OFF"),
        "detail": (
            "The optional sealed local-selection state is not trusted yet. "
            "No candidate selection, local-batch, or Full continuation claim "
            "is inferred from missing or invalid bytes."
        ),
        "state": "in_progress",
        "updated_at": observed_at,
        "progress_pct": 20,
        "evidence": [
            f"state={condition}_fail_safe",
            "authenticated=unavailable / pending=unavailable / terminal=unavailable",
            "selected symmetric result=none claimed",
            "local bounded budget=max3/round × 2 rounds / prepare-only",
            "AUTO FULL OFF / Scheduler mutation=false",
            f"optional state path={path.resolve()}",
        ],
    }


def _selection_count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise UpdaterError(f"local symmetric selection {label} drifted")
    return value


def _selection_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UpdaterError(f"local symmetric selection {label} drifted")
    number = float(value)
    if not math.isfinite(number):
        raise UpdaterError(f"local symmetric selection {label} drifted")
    return number


def _v6_local_selection_lifecycle_counts(
    value: Mapping[str, Any],
    *,
    authenticated: int,
    pending: int,
    failures: int,
    tasks: Mapping[int, Mapping[str, Any]] | None,
) -> tuple[int, int, int, int]:
    lifecycle_ids = value.get("lifecycle_task_ids")
    superseded_ids = value.get("selection_superseded_task_ids")
    lanes = value.get("lanes")
    if (
        not isinstance(lifecycle_ids, list)
        or len(lifecycle_ids) != 9
        or len(set(lifecycle_ids)) != 9
        or set(lifecycle_ids) != set(STANDARD_SELECTION_LIFECYCLE_TASK_IDS)
        or not isinstance(superseded_ids, list)
        or set(superseded_ids) != set(STANDARD_SELECTION_SUPERSEDED_TASK_IDS)
        or value.get("effective_lane_count") != 7
        or value.get("lifecycle_lane_count") != 9
        or not isinstance(lanes, list)
        or len(lanes) != 9
    ):
        raise UpdaterError("local symmetric selection v6 lifecycle drifted")
    lifecycle_statuses: list[str] = []
    effective_statuses: list[str] = []
    observed_ids: list[int] = []
    status_by_task: dict[int, str] = {}
    for lane in lanes:
        if not isinstance(lane, Mapping):
            raise UpdaterError("local symmetric selection v6 lane drifted")
        task_id = lane.get("task_id")
        status = str(lane.get("effective_status") or "")
        selection_effective = lane.get("selection_effective")
        if (
            isinstance(task_id, bool)
            or not isinstance(task_id, int)
            or task_id not in STANDARD_SELECTION_LIFECYCLE_TASK_IDS
            or status not in {"pending", "collection_ready", "terminal_failure"}
            or not isinstance(selection_effective, bool)
            or selection_effective != (task_id in STANDARD_SELECTION_TASK_IDS)
        ):
            raise UpdaterError("local symmetric selection v6 lane drifted")
        observed_ids.append(task_id)
        status_by_task[task_id] = status
        lifecycle_statuses.append(status)
        if selection_effective:
            effective_statuses.append(status)
    if len(set(observed_ids)) != 9:
        raise UpdaterError("local symmetric selection v6 lanes are duplicated")
    effective_counts = {
        status: effective_statuses.count(status)
        for status in {"pending", "collection_ready", "terminal_failure"}
    }
    if (
        effective_counts["collection_ready"] != authenticated
        or effective_counts["pending"] != pending
        or effective_counts["terminal_failure"] != failures
    ):
        raise UpdaterError("local symmetric selection v6 counts drifted")
    watcher_pending = lifecycle_statuses.count("pending")
    watcher_failures = lifecycle_statuses.count("terminal_failure")
    current_failure_ids = {
        task_id
        for task_id, lane_status in status_by_task.items()
        if lane_status == "terminal_failure"
    }
    if tasks is not None:
        current_failure_ids.update(
            task_id
            for task_id in STANDARD_SELECTION_LIFECYCLE_TASK_IDS
            if _category(str(tasks[task_id]["state"])) == "failed"
        )
    current_collection_ids = {
        task_id
        for task_id, lane_status in status_by_task.items()
        if lane_status == "collection_ready"
    } - current_failure_ids
    return (
        9 - len(current_collection_ids) - len(current_failure_ids),
        len(current_failure_ids),
        watcher_pending,
        watcher_failures,
    )


def _strict_local_symmetric_selection_card(
    path: Path,
    observed_at: str,
    tasks: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    try:
        value = _read_sealed_local_json(
            path,
            schema=LOCAL_SYMMETRIC_SELECTION_STATE_SCHEMA,
            canonical_ensure_ascii=True,
        )
    except FileNotFoundError:
        return _local_selection_fail_safe_card(path, observed_at, condition="missing")
    except OSError:
        return _local_selection_fail_safe_card(
            path, observed_at, condition="unavailable"
        )
    except UpdaterError:
        return _local_selection_fail_safe_card(path, observed_at, condition="invalid")

    allowed_statuses = {
        "awaiting_symmetric_results",
        "partial_measured_waiting",
        "selected_symmetric_hard_pass",
        "local_prepare_only_batch_ready",
        "terminal_no_passing_or_small_local_correction",
        "blocked_fail_closed",
    }
    status = str(value.get("status") or "")
    authenticated = _selection_count(
        value.get("authenticated_observation_count"),
        "authenticated observation count",
    )
    pending = _selection_count(value.get("pending_count"), "pending count")
    failures = _selection_count(
        value.get("terminal_failure_count"),
        "terminal failure count",
    )
    task_ids = value.get("expected_task_ids")
    budget = value.get("finite_local_budget")
    task_ids_valid = (
        isinstance(task_ids, list)
        and len(task_ids) == 7
        and all(
            isinstance(task_id, int) and not isinstance(task_id, bool)
            for task_id in task_ids
        )
        and len(set(task_ids)) == 7
    )
    authoritative_layout = task_ids_valid and set(task_ids) == set(
        STANDARD_SELECTION_TASK_IDS
    )
    legacy_layout = (
        task_ids_valid
        and set(task_ids) == set(LEGACY_STANDARD_SELECTION_TASK_IDS)
        and "lifecycle_task_ids" not in value
    )
    if (
        status not in allowed_statuses
        or not (authoritative_layout or legacy_layout)
        or authenticated + pending + failures != 7
        or value.get("diagnostic_only") is not True
        or value.get("production_eligible") is not False
        or value.get("symmetric_model_primary") is not True
        or value.get("prepare_only") is not True
        or value.get("scheduler_methods_used") != []
        or value.get("scheduler_mutation_performed") is not False
        or value.get("scheduler_submission_performed") is not False
        or value.get("scheduler_cancel_performed") is not False
        or value.get("scheduler_restart_performed") is not False
        or value.get("automatic_full_trigger") is not False
        or value.get("automatic_full_continuation") is not False
        or value.get("full_model_started_by_watcher") is not False
        or value.get("terminal_failures_are_physics_observations") is not False
        or value.get("pending_allows_local_candidate_generation") is not False
        or not isinstance(value.get("watch_complete"), bool)
        or not isinstance(budget, Mapping)
    ):
        return _local_selection_fail_safe_card(path, observed_at, condition="invalid")

    if authoritative_layout:
        if status == "blocked_fail_closed" and "lifecycle_task_ids" not in value:
            lifecycle_pending = pending
            lifecycle_failures = failures
            lifecycle_count = 7
            lifecycle_mode = "blocked-fail-closed-effective-only"
            watcher_snapshot = "watcher snapshot=blocked effective-only"
        else:
            try:
                (
                    lifecycle_pending,
                    lifecycle_failures,
                    watcher_pending,
                    watcher_failures,
                ) = _v6_local_selection_lifecycle_counts(
                    value,
                    authenticated=authenticated,
                    pending=pending,
                    failures=failures,
                    tasks=tasks,
                )
            except UpdaterError:
                return _local_selection_fail_safe_card(
                    path, observed_at, condition="invalid"
                )
            lifecycle_count = 9
            lifecycle_mode = "v6-sealed-effective-plus-live-GET-lifecycle"
            watcher_snapshot = (
                f"watcher snapshot pending={watcher_pending} / "
                f"operational terminal failures={watcher_failures}"
            )
    else:
        overlay_pending, overlay_failures = _selection_lifecycle_overlay(tasks)
        lifecycle_pending = pending + overlay_pending
        lifecycle_failures = failures + overlay_failures
        lifecycle_count = 9
        lifecycle_mode = "v5-overlay-fallback"
        watcher_snapshot = "watcher snapshot=v5 effective-only"

    current_round = _selection_count(budget.get("current_round"), "current local round")
    max_rounds = _selection_count(budget.get("max_rounds"), "maximum local rounds")
    max_per_round = _selection_count(
        budget.get("max_candidates_per_round"),
        "maximum candidates per round",
    )
    max_total = _selection_count(
        budget.get("max_candidates_total"),
        "maximum total candidates",
    )
    prepared = _selection_count(
        budget.get("candidate_count_this_round"),
        "candidate count this round",
    )
    if (
        current_round not in {0, 1}
        or max_rounds != 2
        or max_per_round != 3
        or max_total != 6
        or prepared > max_per_round
        or (pending and prepared)
    ):
        return _local_selection_fail_safe_card(path, observed_at, condition="invalid")

    selected = value.get("selected_symmetric_result")
    selected_text = "none"
    if status == "selected_symmetric_hard_pass":
        if not isinstance(selected, Mapping):
            return _local_selection_fail_safe_card(
                path, observed_at, condition="invalid"
            )
        task_id = selected.get("task_id")
        candidate_sha = selected.get("candidate_physics_sha256")
        if (
            isinstance(task_id, bool)
            or not isinstance(task_id, int)
            or task_id not in set(task_ids)
            or not isinstance(candidate_sha, str)
            or not re.fullmatch(r"[0-9a-f]{64}", candidate_sha)
        ):
            return _local_selection_fail_safe_card(
                path, observed_at, condition="invalid"
            )
        margin = _selection_number(
            selected.get("minimum_normalized_actual_margin"),
            "selected margin",
        )
        loss = _selection_number(
            selected.get("actual_total_loss_W"),
            "selected loss",
        )
        volume = _selection_number(
            selected.get("actual_volume_L"),
            "selected volume",
        )
        selected_text = (
            f"task{task_id} {candidate_sha[:12]} / margin={margin:.6g} / "
            f"loss={loss:.3f}W / volume={volume:.3f}L"
        )
        label = f"SELECTED task{task_id}"
    elif selected is not None:
        return _local_selection_fail_safe_card(path, observed_at, condition="invalid")
    else:
        label = {
            "awaiting_symmetric_results": "WAITING",
            "partial_measured_waiting": "MEASURED, WAITING",
            "local_prepare_only_batch_ready": (f"LOCAL BATCH {prepared}/3 READY"),
            "terminal_no_passing_or_small_local_correction": ("NO ELIGIBLE LOCAL STEP"),
            "blocked_fail_closed": "FAIL-CLOSED",
        }[status]

    progress = {
        "awaiting_symmetric_results": 35,
        "partial_measured_waiting": min(85, 40 + authenticated * 6),
        "selected_symmetric_hard_pass": 100,
        "local_prepare_only_batch_ready": 90,
        "terminal_no_passing_or_small_local_correction": 100,
        "blocked_fail_closed": 25,
    }[status]
    card = {
        "id": LOCAL_SYMMETRIC_SELECTION_CARD_ID,
        "title": (
            "CODEX · SYMMETRIC FEA SELECT · "
            f"{label} · AUTH {authenticated}/7 · AUTO FULL OFF"
        ),
        "detail": (
            "Authenticated symmetric FEA is the primary selection gate. "
            "A passing result stops selection immediately; otherwise only a "
            "finite prepare-only local batch can be emitted after all current "
            "lanes are terminal."
        ),
        "state": "in_progress",
        "updated_at": observed_at,
        "progress_pct": progress,
        "evidence": [
            f"status={status}",
            (
                f"authenticated={authenticated}/7 / pending={pending} / "
                f"terminal failures={failures}"
            ),
            f"selected symmetric result={selected_text}",
            (
                f"local bounded budget=round {current_round + 1}/"
                f"{max_rounds} / prepared {prepared}/{max_per_round} / "
                f"total cap {max_total} / prepare-only"
            ),
            "AUTO FULL OFF / Scheduler mutation=false / methods=[]",
            f"state SHA256 {_file_sha256(path.resolve())}",
        ],
    }
    card["title"] = (
        f"CODEX | SYMMETRIC FEA SELECT | {label} | AUTH {authenticated}/7 | "
        f"PENDING {lifecycle_pending} | ACTUAL PASS 0 | AUTO FULL OFF"
    )
    card["evidence"][1] = (
        f"authenticated={authenticated}/7 / pending={lifecycle_pending} / "
        f"operational terminal failures={lifecycle_failures}"
    )
    card["evidence"].insert(
        2,
        f"lifecycle mode={lifecycle_mode} / observations={lifecycle_count} / "
        "selection-effective=7 / "
        "task96332 superseded by task96338 / "
        "task96337 pre-EM failure is operational only",
    )
    card["evidence"].insert(3, watcher_snapshot)
    card["evidence"].insert(-1, "actual scientific PASS=0 / actual production PASS=0")
    return card


def _local_symmetric_selection_card(
    path: Path,
    observed_at: str,
    tasks: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    try:
        return _strict_local_symmetric_selection_card(path, observed_at, tasks)
    except FileNotFoundError:
        condition = "missing"
    except OSError:
        condition = "unavailable"
    except UpdaterError:
        condition = "invalid"
    return _local_selection_fail_safe_card(path, observed_at, condition=condition)


def _thermal_bridge_card(path: Path, observed_at: str) -> dict[str, Any]:
    value = _read_sealed_local_json(
        path,
        schema=THERMAL_BRIDGE_STATE_SCHEMA,
        schema_field="schema",
    )
    state = str(value.get("watcher_state") or "").lower()
    allowed = {"armed", "running", "collected", "failed"}
    if state not in allowed:
        raise UpdaterError("thermal bridge watcher state drifted")
    pid = _positive_or_none(value.get("watcher_pid"), "thermal bridge watcher PID")
    post_count = value.get("scheduler_post_calls_total")
    if isinstance(post_count, bool) or post_count not in {0, 1}:
        raise UpdaterError("thermal bridge POST count drifted")
    tool_sha = str(value.get("tool_sha256") or "")
    source_state = str(value.get("source_task_state") or "")
    stage = str(value.get("stage") or "")
    if (
        value.get("source_task_id") != 96324
        or value.get("source_task_name") != TASK_SPECS[0].task_name
        or value.get("heartbeat_interval_seconds") != 60
        or not re.fullmatch(r"[0-9a-f]{64}", tool_sha)
        or not source_state
        or not stage
        or value.get("diagnostic_only") is not True
        or value.get("canonical") is not False
        or value.get("production_truth_eligible") is not False
        or value.get("scheduler_mutation_performed") is not False
        or value.get("scientific_pass_claimed") is not False
        or value.get("production_claimed") is not False
        or value.get("artifact_collected") is not (state == "collected")
        or (state == "failed") != bool(value.get("failure"))
    ):
        raise UpdaterError("thermal bridge safety boundary drifted")
    updated = str(value.get("updated_at_utc") or "")
    try:
        parsed = datetime.fromisoformat(updated)
    except ValueError as exc:
        raise UpdaterError("thermal bridge heartbeat timestamp drifted") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise UpdaterError("thermal bridge heartbeat timestamp lacks timezone")
    progress = {
        "armed": 25,
        "running": 55,
        "collected": 100,
        "failed": 100,
    }[state]
    return {
        "id": "codex-thermal-artifact-handoff",
        "title": (f"CODEX AUTO · THERMAL ARTIFACT HANDOFF · {state.upper()}"),
        "detail": (
            "Corrected-thermal task96324의 terminal-success artifact를 "
            "인증·전송·수집하는 자동화 상태입니다. 이 lifecycle 표시는 "
            "온도 제약이나 scientific/production PASS를 주장하지 않습니다."
        ),
        "state": "in_progress",
        "updated_at": observed_at,
        "progress_pct": progress,
        "evidence": [
            f"source task96324 state={source_state} / stage={stage}",
            (f"Scheduler POST count={post_count} / heartbeat mutation=false"),
            (
                f"artifact collected="
                f"{str(bool(value.get('artifact_collected'))).lower()}"
            ),
            "scientific claim=false / production claim=false",
            f"watcher PID={pid} / tool SHA256 {tool_sha}",
            f"heartbeat updated={updated}",
            f"state SHA256 {_file_sha256(path.resolve())}",
        ],
    }


def _standard_full_continuation_card(
    path: Path,
    observed_at: str,
) -> dict[str, Any]:
    value = _read_sealed_local_json(
        path,
        schema=STANDARD_FULL_CONTINUATION_STATE_SCHEMA,
        schema_field="schema_version",
    )
    status = str(value.get("status") or "")
    allowed = {
        "pending_standard_collections",
        "terminal_no_measured_pass",
        "measured_pass_waiting_for_strict_full_lane",
        "prepared_waiting_for_post_authorization",
        "full_submitted_pending_actual_result",
        "submission_outcome_uncertain_no_repost",
        "fail_closed_retrying_get_gates",
    }
    if status not in allowed:
        raise UpdaterError("Standard-to-Full continuation status drifted")
    attempts = value.get("scheduler_post_attempts_consumed")
    if isinstance(attempts, bool) or attempts not in {0, 1}:
        raise UpdaterError("Standard-to-Full POST counter drifted")
    full_task_id = _positive_or_none(
        value.get("full_task_id"),
        "Standard-to-Full task ID",
    )
    attempt = value.get("attempt_ledger")
    receipt = value.get("submission_receipt")
    plan = value.get("plan")
    if (
        value.get("original_deadline_missed") is not True
        or value.get("postdeadline") is not True
        or value.get("canonical") is not False
        or value.get("production_eligible") is not False
        or value.get("scientific_pass_claimed") is not False
        or value.get("full_result_available") is not False
        or value.get("full_actual_constraints_passed") is not False
        or value.get("promotion_completed") is not False
        or value.get("maximum_scheduler_posts") != 1
        or value.get("scheduler_project_mutation_performed") is not False
        or value.get("scheduler_repository_modified") is not False
        or (attempts == 0 and attempt is not None)
        or (attempts == 1 and not isinstance(attempt, Mapping))
        or (
            status == "full_submitted_pending_actual_result"
            and (
                full_task_id is None
                or not isinstance(plan, Mapping)
                or not isinstance(attempt, Mapping)
                or not isinstance(receipt, Mapping)
            )
        )
        or (status != "full_submitted_pending_actual_result" and receipt is not None)
    ):
        raise UpdaterError("Standard-to-Full safety boundary drifted")
    updated = str(value.get("observed_at_utc") or "")
    try:
        parsed = datetime.fromisoformat(updated)
    except ValueError as exc:
        raise UpdaterError("Standard-to-Full heartbeat timestamp drifted") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise UpdaterError("Standard-to-Full heartbeat timestamp lacks timezone")
    progress = {
        "pending_standard_collections": 25,
        "terminal_no_measured_pass": 100,
        "measured_pass_waiting_for_strict_full_lane": 50,
        "prepared_waiting_for_post_authorization": 65,
        "full_submitted_pending_actual_result": 80,
        "submission_outcome_uncertain_no_repost": 100,
        "fail_closed_retrying_get_gates": 35,
    }[status]
    return {
        "id": LEGACY_CONTINUATION_CARD_ID,
        "title": (f"CODEX AUTO · STANDARD→FULL CONTINUATION · {status.upper()}"),
        "detail": (
            "인증된 Standard 실측 hard-feasible rank-0 후보가 생길 때만 "
            "동일 후보의 Full 계산을 별도 strict-node lane에 최대 한 번 "
            "연결합니다. Full terminal result 전에는 scientific PASS나 "
            "promotion을 주장하지 않습니다."
        ),
        "state": "in_progress",
        "updated_at": observed_at,
        "progress_pct": progress,
        "evidence": [
            f"state={status} / Full task={full_task_id or 'none'}",
            f"Scheduler POST attempts consumed={attempts}/1",
            (
                f"plan={str(isinstance(plan, Mapping)).lower()} / "
                f"receipt={str(isinstance(receipt, Mapping)).lower()}"
            ),
            "scientific PASS=false / promotion=false / canonical=false",
            "Scheduler project mutation=false / repository mixing=false",
            f"heartbeat updated={updated}",
            f"state SHA256 {_file_sha256(path.resolve())}",
        ],
    }


def _final_gate_card(root: Path, observed_at: str) -> dict[str, Any]:
    resolved_root = root.resolve()
    seal_path = resolved_root / FINAL_PACKAGE_NAME / "package_seal.json"
    pending_path = resolved_root / "pending_manifest.json"
    if seal_path.is_file():
        value = _read_sealed_local_json(
            seal_path,
            schema=FINAL_GATE_SEAL_SCHEMA,
            schema_field="schema",
        )
        if (
            value.get("all_explicit_goal_constraints_actual_pass") is not True
            or value.get("solver_result_truth_included") is not True
        ):
            raise UpdaterError("published final package truth boundary drifted")
        title = "CODEX AUTO · FINAL AEDT PACKAGE GATE · PUBLISHED"
        detail = (
            "인증된 actual solver 결과와 고정 제약을 모두 통과해 Full 및 "
            "symmetric AEDT 패키지가 봉인됐습니다."
        )
        progress = 100
        evidence = [
            "full AEDT promoted=true / symmetric AEDT promoted=true",
            "actual solver result truth=true",
            "all explicit goal constraints actual pass=true",
            f"package seal SHA256 {_file_sha256(seal_path)}",
        ]
    elif pending_path.is_file():
        value = _read_sealed_local_json(
            pending_path,
            schema=FINAL_GATE_PENDING_SCHEMA,
            schema_field="schema",
        )
        reasons = value.get("pending_reasons")
        if (
            not isinstance(reasons, list)
            or value.get("final_package_created") is not False
            or value.get("full_aedt_promoted") is not False
            or value.get("symmetric_aedt_promoted") is not False
        ):
            raise UpdaterError("pending final package truth boundary drifted")
        title = "CODEX AUTO · FINAL AEDT PACKAGE GATE · PENDING"
        detail = (
            "actual Full/thermal 결과와 solver-produced AEDT가 모두 인증될 "
            "때까지 fail-closed 상태를 유지합니다."
        )
        progress = 60
        evidence = [
            *[f"pending: {reason}" for reason in reasons[:4]],
            "full AEDT promoted=false / symmetric AEDT promoted=false",
            "open-only fallback promotion=false",
            f"pending manifest SHA256 {_file_sha256(pending_path)}",
        ]
    else:
        raise UpdaterError("final package gate state is absent")
    return {
        "id": "codex-final-aedt-package-gate",
        "title": title,
        "detail": detail,
        "state": "in_progress",
        "updated_at": observed_at,
        "progress_pct": progress,
        "evidence": evidence,
    }


def _counter(pattern: re.Pattern[str], title: str, label: str) -> int:
    values = {int(value) for value in pattern.findall(title)}
    if len(values) != 1:
        raise UpdaterError(f"existing {label} counter is unavailable")
    return next(iter(values))


def _protected_hashes(payload: Mapping[str, Any]) -> dict[str, str]:
    completed = payload.get("completed")
    attention = payload.get("attention")
    if not isinstance(completed, list) or not isinstance(attention, list):
        raise UpdaterError("protected status groups are missing")
    pareto = [
        item
        for item in attention
        if isinstance(item, dict) and item.get("id") == "pareto-truth-boundary"
    ]
    if len(pareto) != 1:
        raise UpdaterError("Pareto truth boundary is not unique")
    return {
        "completed_sha256": canonical_sha256(completed),
        "attention_sha256": canonical_sha256(attention),
        "pareto_truth_sha256": canonical_sha256(pareto[0]),
    }


def _live_summary(
    *,
    observed_at: str,
    allocation_jobs: int,
    submitted: int,
    running: int,
    queued: int,
    collections: int,
) -> str:
    try:
        observed = datetime.fromisoformat(observed_at)
    except ValueError as exc:
        raise UpdaterError("observed_at is not ISO-8601") from exc
    if observed.tzinfo is None or observed.utcoffset() is None:
        raise UpdaterError("observed_at must include a UTC offset")
    terminal = len(TASK_SPECS) - running - queued
    if terminal < 0:
        raise UpdaterError("live task counters are inconsistent")
    summary = (
        f"{observed:%H:%M} KST · original_deadline_missed=true. "
        "인증된 scientific PASS는 없습니다. "
        f"post-deadline diagnostic 작업: running{running} · queued{queued} · "
        f"terminal{terminal}. Slurm allocation jobs{allocation_jobs} · "
        f"submitted{submitted} · collections{collections}. "
        "공식 512-seed aggregate는 physical feasible0 · production Pareto "
        "front0 · audit-only objective front22입니다. Scheduler terminal "
        "success도 collector와 artifact 인증 전에는 scientific/production "
        "PASS가 아닙니다."
    )
    return summary + " actual scientific PASS=0 / actual production PASS=0."


def _parallel_workstreams_card(
    tasks: Mapping[int, Mapping[str, Any]],
    categories: Mapping[int, str],
    *,
    observed_at: str,
    allocation_jobs: int,
    running: int,
    queued: int,
) -> dict[str, Any]:
    terminal = len(TASK_SPECS) - running - queued
    operational_risks = [
        (spec, risk)
        for spec in TASK_SPECS
        if (
            risk := _force_cancel_risk(
                spec,
                category=categories[spec.task_id],
                observed_at=observed_at,
            )
        )
        is not None
    ]
    active_nodes = list(
        dict.fromkeys(
            str(tasks[spec.task_id]["actual_node_name"] or spec.requested_node)
            for spec in TASK_SPECS
            if categories[spec.task_id] in {"running", "queued"}
        )
    )
    nodes = f"ACTIVE NODES {len(active_nodes)}" if active_nodes else "NO ACTIVE NODES"
    evidence = [
        (
            f"active allocations{allocation_jobs} / running{running} / "
            f"queued{queued} / terminal{terminal} / all managed tasks "
            "diagnostic/noncanonical / Scheduler GET only / "
            "scientific_pass_generated=false / actual_scientific_pass_count=0 / "
            "actual_production_pass_count=0 / canonical_promotion=false"
        )
    ]
    task_evidence = [
        (
            f"task{spec.task_id} {spec.model_label} "
            f"{tasks[spec.task_id]['state']} / "
            f"allocation{tasks[spec.task_id]['allocation_id'] or 'none'} / "
            f"job{tasks[spec.task_id]['slurm_job_id'] or 'none'} / "
            f"{tasks[spec.task_id]['actual_node_name'] or spec.requested_node}"
        )
        for spec in TASK_SPECS
    ]
    evidence.extend(
        " | ".join(task_evidence[index : index + 2])
        for index in range(0, len(task_evidence), 2)
    )
    evidence.extend(
        (
            f"RISK task{spec.task_id} allocation{risk['allocation_id']} "
            f"force-cancel {risk['force_at']:%Y-%m-%d %H:%M:%S KST} / "
            f"task timeout {risk['timeout_at']:%Y-%m-%d %H:%M:%S KST} / "
            f"lead {_duration_text(risk['lead_seconds'])} / "
            "hard guarantee=false"
        )
        for spec, risk in operational_risks
    )
    active_risk_title = "".join(
        (
            f" · RISK task{spec.task_id}/allocation{risk['allocation_id']} "
            f"FORCE {risk['force_at']:%m-%d %H:%M:%S KST}"
        )
        for spec, risk in operational_risks
        if risk["active"]
    )
    return {
        "id": "parallel-workstreams",
        "title": (
            f"PARALLEL TRACKS · RUNNING {running} · QUEUED {queued} · "
            f"ALLOCATION JOBS {allocation_jobs} · {nodes}{active_risk_title}"
        ),
        "detail": (
            f"Scheduler GET-authenticated lifecycle for {len(TASK_SPECS)} managed "
            "post-deadline tracks. Each task stays separate from collection, "
            "scientific PASS, canonical promotion, and production truth."
        ),
        "state": "in_progress",
        "updated_at": observed_at,
        "progress_pct": 95 if running or queued else 100,
        "evidence": evidence,
    }


def merge_status(
    payload: Mapping[str, Any],
    tasks: Mapping[int, Mapping[str, Any]],
    *,
    observed_at: str,
    postsuccess_state_file: Path | None = None,
    thermal_bridge_state_file: Path | None = None,
    local_symmetric_selection_state_file: Path | None = (
        DEFAULT_LOCAL_SYMMETRIC_SELECTION_STATE_FILE
    ),
    standard_full_continuation_state_file: Path | None = None,
    final_gate_root: Path | None = None,
) -> dict[str, Any]:
    if payload.get("schema_version") != STATUS_SCHEMA:
        raise UpdaterError("Codex status schema drifted")
    result = copy.deepcopy(dict(payload))
    result.pop(SYNC_KEY, None)
    protected_before = _protected_hashes(result)
    if standard_full_continuation_state_file is None:
        _remove_current_card(result, LEGACY_CONTINUATION_CARD_ID)
    for spec in TASK_SPECS:
        _upsert_current_card(
            result,
            _task_card(spec, tasks[spec.task_id], observed_at),
        )
    _upsert_current_card(
        result,
        _symmetric_primary_policy_card(tasks, observed_at),
    )
    if postsuccess_state_file is not None:
        _upsert_current_card(
            result,
            _postsuccess_card(postsuccess_state_file, observed_at, tasks),
        )
    if thermal_bridge_state_file is not None:
        _upsert_current_card(
            result,
            _thermal_bridge_card(thermal_bridge_state_file, observed_at),
        )
    if local_symmetric_selection_state_file is not None:
        _upsert_current_card(
            result,
            _local_symmetric_selection_card(
                local_symmetric_selection_state_file,
                observed_at,
                tasks,
            ),
        )
    else:
        _remove_current_card(result, LOCAL_SYMMETRIC_SELECTION_CARD_ID)
    if standard_full_continuation_state_file is not None:
        _upsert_current_card(
            result,
            _standard_full_continuation_card(
                standard_full_continuation_state_file,
                observed_at,
            ),
        )
    if final_gate_root is not None:
        _upsert_current_card(
            result,
            _final_gate_card(final_gate_root, observed_at),
        )

    handoff = _single_current(result, "fea-handoff")
    previous_title = str(handoff.get("title") or "")
    submitted = max(
        CAMPAIGN_SUBMITTED_FLOOR,
        _counter(SUBMITTED_PATTERN, previous_title, "submitted"),
    )
    collections = _counter(COLLECTIONS_PATTERN, previous_title, "collections")
    categories = {
        task_id: _category(str(task["state"])) for task_id, task in tasks.items()
    }
    running = sum(value == "running" for value in categories.values())
    queued = sum(value == "queued" for value in categories.values())
    active_allocations = {
        task["allocation_id"]
        for task_id, task in tasks.items()
        if categories[task_id] in {"running", "queued"}
        and task["allocation_id"] is not None
    }
    allocation_jobs = len(active_allocations)
    result["summary"] = _live_summary(
        observed_at=observed_at,
        allocation_jobs=allocation_jobs,
        submitted=submitted,
        running=running,
        queued=queued,
        collections=collections,
    )
    handoff_task_evidence = [
        (
            f"task{spec.task_id} {tasks[spec.task_id]['state']} / "
            f"allocation{tasks[spec.task_id]['allocation_id'] or 'none'} / "
            f"job{tasks[spec.task_id]['slurm_job_id'] or 'none'} / "
            f"{tasks[spec.task_id]['actual_node_name'] or spec.requested_node}"
        )
        for spec in TASK_SPECS
    ]
    handoff.update(
        {
            "title": (
                f"SLURM · ALLOCATION JOBS {allocation_jobs} · "
                f"SUBMITTED {submitted} · RUNNING {running} · "
                f"QUEUED {queued} · COLLECTIONS {collections}"
            ),
            "detail": (
                "Scheduler exact GET lifecycle 집계입니다. 각 작업의 terminal "
                "success도 별도 collector/artifact 인증 전에는 collection 또는 "
                "scientific PASS가 아닙니다. completed/attention/Pareto truth는 "
                "이 updater가 수정하지 않습니다."
            ),
            "state": "in_progress",
            "updated_at": observed_at,
            "progress_pct": 97,
            "evidence": [
                " | ".join(handoff_task_evidence[index : index + 2])
                for index in range(0, len(handoff_task_evidence), 2)
            ]
            + [
                (
                    f"active allocation jobs{allocation_jobs} / running{running} / "
                    f"queued{queued} / submitted{submitted} / "
                    f"collections{collections} preserved"
                ),
                (
                    "Scheduler GET only / scientific_pass_generated=false / "
                    "canonical_promotion=false / Scheduler project remains separate "
                    "from MFT repository"
                ),
            ],
        }
    )
    _upsert_current_card(
        result,
        _parallel_workstreams_card(
            tasks,
            categories,
            observed_at=observed_at,
            allocation_jobs=allocation_jobs,
            running=running,
            queued=queued,
        ),
    )
    result["generated_at"] = observed_at
    if _protected_hashes(result) != protected_before:
        raise UpdaterError("protected completed/attention/Pareto truth changed")
    unsigned_status = copy.deepcopy(result)
    status_payload_sha256 = canonical_sha256(unsigned_status)
    task_snapshot = [
        {
            "task_id": spec.task_id,
            "state": tasks[spec.task_id]["state"],
            "allocation_id": tasks[spec.task_id]["allocation_id"],
            "slurm_job_id": tasks[spec.task_id]["slurm_job_id"],
            "node_name": tasks[spec.task_id]["actual_node_name"],
            "exit_code": tasks[spec.task_id]["exit_code"],
            "failure_message": tasks[spec.task_id]["failure_message"],
        }
        for spec in TASK_SPECS
    ]
    result[SYNC_KEY] = _sealed(
        {
            "schema_version": SYNC_SCHEMA,
            "observed_at": observed_at,
            "scheduler_endpoint": "GET /api/tasks/{task_id}",
            "scheduler_methods_used": ["GET"],
            "scheduler_mutation_performed": False,
            "managed_task_ids": [spec.task_id for spec in TASK_SPECS],
            "submitted_total": submitted,
            "allocation_jobs_active": allocation_jobs,
            "running": running,
            "queued": queued,
            "collections_preserved": collections,
            "scientific_pass_generated": False,
            "actual_scientific_pass_count": 0,
            "actual_production_pass_count": 0,
            "collection_generated": False,
            "canonical_promotion_generated": False,
            "protected": protected_before,
            "status_without_sync_sha256": status_payload_sha256,
            "tasks": task_snapshot,
        }
    )
    return result


def validate_status_sync(payload: Mapping[str, Any]) -> dict[str, Any]:
    sync = validate_seal(payload.get(SYNC_KEY), SYNC_SCHEMA)
    unsigned_status = copy.deepcopy(dict(payload))
    unsigned_status.pop(SYNC_KEY, None)
    if sync.get("status_without_sync_sha256") != canonical_sha256(unsigned_status):
        raise UpdaterError("status payload does not match its sync seal")
    if sync.get("protected") != _protected_hashes(payload):
        raise UpdaterError("protected status hashes do not match sync seal")
    return sync


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def synchronize_once(
    *,
    status_file: Path,
    scheduler_url: str = DEFAULT_SCHEDULER_URL,
    task_reader: TaskReader | None = None,
    observed_at: str | None = None,
    postsuccess_state_file: Path | None = None,
    thermal_bridge_state_file: Path | None = None,
    local_symmetric_selection_state_file: Path | None = (
        DEFAULT_LOCAL_SYMMETRIC_SELECTION_STATE_FILE
    ),
    standard_full_continuation_state_file: Path | None = None,
    final_gate_root: Path | None = None,
) -> dict[str, Any]:
    tasks = fetch_tasks(scheduler_url, task_reader=task_reader)
    source = status_file.resolve().read_bytes()
    source_sha256 = hashlib.sha256(source).hexdigest()
    try:
        payload = json.loads(source.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise UpdaterError("status file is invalid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise UpdaterError("status file root must be an object")
    updated = merge_status(
        payload,
        tasks,
        observed_at=observed_at or _timestamp(),
        postsuccess_state_file=postsuccess_state_file,
        thermal_bridge_state_file=thermal_bridge_state_file,
        local_symmetric_selection_state_file=(local_symmetric_selection_state_file),
        standard_full_continuation_state_file=(standard_full_continuation_state_file),
        final_gate_root=final_gate_root,
    )
    validate_status_sync(updated)
    if len(_json_bytes(updated)) > 256 * 1024:
        raise UpdaterError("updated status exceeds monitor size bound")
    _atomic_json(status_file, updated, expected_sha256=source_sha256)
    return updated[SYNC_KEY]


def _append_log(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = _sealed(
        {
            "schema_version": LOG_SCHEMA,
            "recorded_at": _timestamp(),
            **dict(event),
        }
    )
    with path.open("ab") as stream:
        stream.write(
            json.dumps(
                record,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def run_updater(
    *,
    status_file: Path,
    scheduler_url: str,
    once: bool,
    interval_seconds: int,
    pid_file: Path,
    log_file: Path,
    lock_file: Path,
    task_reader: TaskReader | None = None,
    max_cycles: int | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    postsuccess_state_file: Path | None = None,
    thermal_bridge_state_file: Path | None = None,
    local_symmetric_selection_state_file: Path | None = (
        DEFAULT_LOCAL_SYMMETRIC_SELECTION_STATE_FILE
    ),
    standard_full_continuation_state_file: Path | None = None,
    final_gate_root: Path | None = None,
) -> dict[str, Any] | None:
    if interval_seconds < 1:
        raise UpdaterError("interval-seconds must be positive")
    with SingleInstanceLock(lock_file):
        pid = _sealed(
            {
                "schema_version": PID_SCHEMA,
                "pid": os.getpid(),
                "started_at": _timestamp(),
                "command": [str(Path(sys.executable).resolve()), *sys.argv],
                "status_file": str(status_file.resolve()),
                "scheduler_url": scheduler_url.rstrip("/"),
                "scheduler_methods_allowed": ["GET"],
                "scheduler_mutation_performed": False,
                "interval_seconds": interval_seconds,
                "tool_sha256": _file_sha256(Path(__file__).resolve()),
            }
        )
        _atomic_json(pid_file, pid)
        _append_log(
            log_file,
            {
                "event": "updater_started",
                "pid": os.getpid(),
                "interval_seconds": interval_seconds,
            },
        )
        cycles = 0
        while True:
            try:
                result = synchronize_once(
                    status_file=status_file,
                    scheduler_url=scheduler_url,
                    task_reader=task_reader,
                    postsuccess_state_file=postsuccess_state_file,
                    thermal_bridge_state_file=thermal_bridge_state_file,
                    local_symmetric_selection_state_file=(
                        local_symmetric_selection_state_file
                    ),
                    standard_full_continuation_state_file=(
                        standard_full_continuation_state_file
                    ),
                    final_gate_root=final_gate_root,
                )
                _append_log(
                    log_file,
                    {
                        "event": "cycle_completed",
                        "sync_payload_sha256": result["payload_sha256"],
                        "running": result["running"],
                        "queued": result["queued"],
                        "allocation_jobs_active": result["allocation_jobs_active"],
                    },
                )
            except Exception as exc:
                _append_log(
                    log_file,
                    {
                        "event": "cycle_failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                if once:
                    raise
                result = None
            cycles += 1
            if once or (max_cycles is not None and cycles >= max_cycles):
                return result
            sleeper(interval_seconds)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GET-only atomic updater for MFT post-deadline UI cards"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--scheduler-url", default=DEFAULT_SCHEDULER_URL)
    parser.add_argument("--status-file", type=Path, default=DEFAULT_STATUS_FILE)
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--lock-file", type=Path)
    parser.add_argument("--postsuccess-state-file", type=Path)
    parser.add_argument("--thermal-bridge-state-file", type=Path)
    parser.add_argument(
        "--local-symmetric-selection-state-file",
        type=Path,
        default=DEFAULT_LOCAL_SYMMETRIC_SELECTION_STATE_FILE,
    )
    parser.add_argument("--standard-full-continuation-state-file", type=Path)
    parser.add_argument("--final-gate-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    ui_root = args.status_file.resolve().parent
    result = run_updater(
        status_file=args.status_file,
        scheduler_url=args.scheduler_url,
        once=args.once,
        interval_seconds=args.interval_seconds,
        pid_file=args.pid_file or ui_root / "postdeadline-ui-updater.pid.json",
        log_file=args.log_file or ui_root / "postdeadline-ui-updater.jsonl",
        lock_file=args.lock_file or ui_root / "postdeadline-ui-updater.lock",
        postsuccess_state_file=args.postsuccess_state_file,
        thermal_bridge_state_file=args.thermal_bridge_state_file,
        local_symmetric_selection_state_file=(
            args.local_symmetric_selection_state_file
        ),
        standard_full_continuation_state_file=(
            args.standard_full_continuation_state_file
        ),
        final_gate_root=args.final_gate_root,
    )
    if args.once:
        print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
