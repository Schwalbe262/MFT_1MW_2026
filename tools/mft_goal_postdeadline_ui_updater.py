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
DEFAULT_INTERVAL_SECONDS = 60
MAX_RESPONSE_BYTES = 1024 * 1024
CAMPAIGN_SUBMITTED_FLOOR = 122
SYNC_KEY = "postdeadline_task_sync"
SYNC_SCHEMA = "mft-goal-postdeadline-ui-sync-v1"
PID_SCHEMA = "mft-goal-postdeadline-ui-updater-pid-v1"
LOG_SCHEMA = "mft-goal-postdeadline-ui-updater-event-v1"
STATUS_SCHEMA = "mft-codex-work-status-v1"
POSTSUCCESS_STATE_SCHEMA = (
    "mft-goal-postdeadline-standard-postsuccess-state-v1"
)
THERMAL_BRIDGE_STATE_SCHEMA = (
    "mft-corrected-thermal-terminal-transport-watch-state-v1"
)
STANDARD_FULL_CONTINUATION_STATE_SCHEMA = (
    "mft-goal-standard-full-continuation-state-v1"
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
    search_only: bool = False
    submission_receipt_sha256: str | None = None
    final_seal_sha256: str | None = None


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
            "mft-goal-diag-standard-postdeadline-official6-"
            "s96185-2772aed82a8c-n113"
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
    ),
    TaskSpec(
        task_id=96329,
        card_id="postdeadline-standard-official8-96329",
        task_name=(
            "mft-goal-diag-standard-postdeadline-official8-"
            "s96141-622097dde126-n114"
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
    ),
)

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
        "same_node_as_task_id": 0,
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
    title = f"POST-DEADLINE {spec.model_label} · task{spec.task_id} {stage} · {node}"
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
        reason = task["failure_message"] or "Scheduler failure reason unavailable"
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
    evidence = [
        (
            f"Scheduler GET task{spec.task_id} {str(task['state']).upper()} / "
            f"allocation{allocation} / Slurm{job} / node{node}"
        ),
        (
            f"cpus{spec.cpus} / memory{spec.memory_mb}MB / "
            f"scheduler timeout{spec.timeout_seconds}s"
        ),
        "strict node placement contract / same_node_as_task_id=0",
        (
            "postdeadline=true / diagnostic_only=true / noncanonical=true"
            + (" / search_only=true" if spec.search_only else "")
        ),
        "scheduler_lifecycle_only=true / collection_authenticated=false",
        "scientific_pass_generated=false / production_claim_generated=false",
    ]
    if spec.inner_solver_seconds is not None:
        evidence.insert(2, f"inner solver budget{spec.inner_solver_seconds}s")
    if spec.submission_receipt_sha256 is not None:
        evidence.append(
            f"submission receipt SHA256 {spec.submission_receipt_sha256}"
        )
    if spec.final_seal_sha256 is not None:
        evidence.append(f"final seal SHA256 {spec.final_seal_sha256}")
    if task["failure_message"]:
        evidence.append(f"failure_message={task['failure_message']}")
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
    if value.get(schema_field) != schema or observed != canonical_sha256(unsigned):
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
            if isinstance(item, dict)
            and item.get("id") == "parallel-workstreams"
        ),
        len(current),
    )
    current.insert(insertion, copy.deepcopy(dict(card)))


def _postsuccess_card(path: Path, observed_at: str) -> dict[str, Any]:
    state = _read_sealed_local_json(
        path,
        schema=POSTSUCCESS_STATE_SCHEMA,
    )
    collection_count = int(state.get("collection_count") or 0)
    pending_count = int(state.get("pending_count") or 0)
    failure_count = int(state.get("terminal_failure_count") or 0)
    if (
        state.get("diagnostic_only") is not True
        or state.get("production_eligible") is not False
        or state.get("scheduler_mutation_performed") is not False
        or state.get("scientific_pass_claimed") is not False
        or state.get("production_claimed") is not False
    ):
        raise UpdaterError("post-success automation safety boundary drifted")
    return {
        "id": "codex-standard-postsuccess-pipeline",
        "title": (
            "CODEX AUTO · STANDARD RESULT PIPELINE · "
            f"COLLECTIONS {collection_count} · PENDING {pending_count}"
        ),
        "detail": (
            "task96325/96327/96328/96329 collector 결과를 기다리면서 인증, "
            "hard-constraint 판정, strict-AL admission 및 measured global "
            "NDS 입력을 자동 처리합니다. 결과가 없으면 "
            "scientific/production claim을 만들지 않습니다."
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
            "Scheduler mutation=false / scientific claim=false",
            f"state SHA256 {_file_sha256(path.resolve())}",
        ],
    }


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
        "title": (
            "CODEX AUTO · THERMAL ARTIFACT HANDOFF · "
            f"{state.upper()}"
        ),
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
            (
                f"Scheduler POST count={post_count} / "
                "heartbeat mutation=false"
            ),
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
        or (
            status != "full_submitted_pending_actual_result"
            and receipt is not None
        )
    ):
        raise UpdaterError("Standard-to-Full safety boundary drifted")
    updated = str(value.get("observed_at_utc") or "")
    try:
        parsed = datetime.fromisoformat(updated)
    except ValueError as exc:
        raise UpdaterError(
            "Standard-to-Full heartbeat timestamp drifted"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise UpdaterError(
            "Standard-to-Full heartbeat timestamp lacks timezone"
        )
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
        "id": "codex-standard-full-continuation",
        "title": (
            "CODEX AUTO · STANDARD→FULL CONTINUATION · "
            f"{status.upper()}"
        ),
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
    return (
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
    active_nodes = list(
        dict.fromkeys(
            str(tasks[spec.task_id]["actual_node_name"] or spec.requested_node)
            for spec in TASK_SPECS
            if categories[spec.task_id] in {"running", "queued"}
        )
    )
    nodes = " + ".join(active_nodes) if active_nodes else "NO ACTIVE NODES"
    evidence = [
        (
            f"active allocations{allocation_jobs} / running{running} / "
            f"queued{queued} / terminal{terminal}"
        )
    ]
    evidence.extend(
        (
            f"task{spec.task_id} {spec.model_label} "
            f"{tasks[spec.task_id]['state']} / "
            f"allocation{tasks[spec.task_id]['allocation_id'] or 'none'} / "
            f"job{tasks[spec.task_id]['slurm_job_id'] or 'none'} / "
            f"{tasks[spec.task_id]['actual_node_name'] or spec.requested_node}"
        )
        for spec in TASK_SPECS
    )
    evidence.extend(
        [
            "all managed post-deadline tasks are diagnostic/noncanonical",
            "Scheduler methods used: GET only",
            "scientific_pass_generated=false / canonical_promotion=false",
        ]
    )
    return {
        "id": "parallel-workstreams",
        "title": (
            f"PARALLEL TRACKS · RUNNING {running} · QUEUED {queued} · "
            f"ALLOCATION JOBS {allocation_jobs} · {nodes}"
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
    standard_full_continuation_state_file: Path | None = None,
    final_gate_root: Path | None = None,
) -> dict[str, Any]:
    if payload.get("schema_version") != STATUS_SCHEMA:
        raise UpdaterError("Codex status schema drifted")
    result = copy.deepcopy(dict(payload))
    result.pop(SYNC_KEY, None)
    protected_before = _protected_hashes(result)
    for spec in TASK_SPECS:
        _upsert_current_card(
            result,
            _task_card(spec, tasks[spec.task_id], observed_at),
        )
    if postsuccess_state_file is not None:
        _upsert_current_card(
            result,
            _postsuccess_card(postsuccess_state_file, observed_at),
        )
    if thermal_bridge_state_file is not None:
        _upsert_current_card(
            result,
            _thermal_bridge_card(thermal_bridge_state_file, observed_at),
        )
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
                (
                    f"task{spec.task_id} {tasks[spec.task_id]['state']} / "
                    f"allocation{tasks[spec.task_id]['allocation_id'] or 'none'} / "
                    f"job{tasks[spec.task_id]['slurm_job_id'] or 'none'} / "
                    f"{tasks[spec.task_id]['actual_node_name'] or spec.requested_node}"
                )
                for spec in TASK_SPECS
            ]
            + [
                f"active allocation jobs{allocation_jobs} / running{running} / queued{queued}",
                f"submitted{submitted} / collections{collections} preserved",
                "Scheduler methods used: GET only",
                "scientific_pass_generated=false / canonical_promotion=false",
                "Scheduler project remains separate from MFT repository",
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
        standard_full_continuation_state_file=(
            standard_full_continuation_state_file
        ),
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
