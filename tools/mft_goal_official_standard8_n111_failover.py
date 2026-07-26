"""One-shot strict-n111 failover for queued official Standard candidate #8.

This is a diagnostic/search-only duplicate of the already queued strict-n114
task 96329.  It never cancels or mutates that task.  Preparation is GET-only;
submission reuses the reviewed consume-before-network transaction and has a
lifetime budget of exactly one Scheduler POST.

The failover is allowed only while all of these live facts remain true:

* task 96329 is still the exact queued strict-n114 official #8 task;
* the former n111 task 96324 is terminal and allocation 14644 is closed;
* n111 has no live allocation or active FEA task and can open a strict pool;
* the latest Scheduler node observation for n114 is fresh and ``drain``.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from tools import mft_goal_official_standard8_postdeadline as official8  # noqa: E402


SCHEDULER_URL = official8.SCHEDULER_URL
PROJECT = official8.PROJECT
ACCOUNT_NAME = "r1jae262"
NODE_NAME = "n111"

CANDIDATE_SHA256 = official8.CANDIDATE_SHA256
SOURCE_SEED = official8.SOURCE_SEED
SOURCE_TASK_ID = official8.SOURCE_TASK_ID
SOURCE_TASK_NAME = official8.SOURCE_TASK_NAME
SOURCE_TASK_DEDUPE_KEY = official8.SOURCE_TASK_DEDUPE_KEY
SOLVER_REVISION = official8.SOLVER_REVISION
LIBRARY_REVISION = official8.LIBRARY_REVISION

ORIGINAL_OFFICIAL8_TASK_ID = 96_329
ORIGINAL_OFFICIAL8_TASK_NAME = official8.TASK_NAME
ORIGINAL_OFFICIAL8_DEDUPE_KEY = (
    "mft-al:mft-goal-diag-standard-postdeadline-official8-"
    "s96141-622097dde126-n114:"
    "a1e4f70cefa1af04673c73a6131bf490c0cc14b5:"
    "e6b9b9d20a832ff5c3f7ca97218737a0b8650781:"
    "6ad4b1c5a342de6b"
)
ORIGINAL_OFFICIAL8_ACCOUNT = "jji0930"
ORIGINAL_OFFICIAL8_NODE = "n114"

FREED_TASK_ID = 96_324
FREED_ALLOCATION_ID = 14_644
FREED_SLURM_JOB_ID = "839461"

CPUS = official8.CPUS
MEMORY_MB = official8.MEMORY_MB
SOLVER_SECONDS = official8.SOLVER_SECONDS
KILL_GRACE_SECONDS = official8.KILL_GRACE_SECONDS
RETENTION_SECONDS = official8.RETENTION_SECONDS
SCHEDULER_SECONDS = official8.SCHEDULER_SECONDS
MAX_WORKERS_PER_NODE = official8.MAX_WORKERS_PER_NODE
PRIORITY = official8.PRIORITY

TASK_NAME = (
    "mft-goal-diag-standard-postdeadline-official8-failover-"
    "s96141-622097dde126-n111"
)
WORKDIR = (
    "mft_goal_diag_standard_postdeadline_official8_failover_"
    "s96141_622097dde126_n111"
)
POST_AUTHORIZATION = (
    "authorize-official8-failover-r1jae262-n111-one-post-v1"
)
OUTPUT_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\postdeadline_standard_official8_failover_622097dde126_n111_260726_v1"
)
PLAN_NAME = official8.PLAN_NAME
ATTEMPT_LEDGER_NAME = official8.ATTEMPT_LEDGER_NAME
SUBMISSION_DIRECTORY_NAME = official8.SUBMISSION_DIRECTORY_NAME

PLAN_SCHEMA = "mft-goal-official-standard8-n111-failover-plan-v1"
CANDIDATE_SCHEMA = (
    "mft-goal-official-standard8-n111-failover-selected-candidate-v1"
)
PREPARE_SCHEMA = "mft-goal-official-standard8-n111-failover-prepare-v1"
INTENT_SCHEMA = (
    "mft-goal-official-standard8-n111-failover-submit-intent-v1"
)
SUBMISSION_SCHEMA = (
    "mft-goal-official-standard8-n111-failover-submission-v1"
)
FINAL_SEAL_SCHEMA = (
    "mft-goal-official-standard8-n111-failover-final-seal-v1"
)
PREFLIGHT_SCHEMA = (
    "mft-goal-official-standard8-n111-failover-live-preflight-v1"
)
ATTEMPT_NONCE_SCHEMA = (
    "mft-goal-official-standard8-n111-failover-attempt-nonce-v1"
)
SINGLE_ATTEMPT_SCHEMA = (
    "mft-goal-official-standard8-n111-failover-single-attempt-v1"
)
COLLECTOR_INTENT_SCHEMA = (
    "mft-goal-official-standard8-n111-failover-collector-intent-v1"
)
COLLECTOR_MANIFEST_SCHEMA = (
    "mft-goal-official-standard8-n111-failover-collector-manifest-v1"
)
COLLECTOR_INTENT_NAME = "collector_compatibility_intent.json"
COLLECTOR_MANIFEST_NAME = "collector_compatibility_manifest.json"
PREPOST_ABORT_SCHEMA = (
    "mft-goal-official-standard8-n111-failover-prepost-abort-v1"
)
PREPOST_ABORT_NAME = "prepost_adapter_abort_evidence.json"

OPENING_QUEUE_REASON = official8.OPENING_QUEUE_REASON
FIXED_BOUNDARY = copy.deepcopy(official8.FIXED_BOUNDARY)
BOUNDARY_PROJECTION = copy.deepcopy(official8.BOUNDARY_PROJECTION)
SAFETY_FLAGS = copy.deepcopy(official8.SAFETY_FLAGS)
NODE_METRICS_MAX_AGE_SECONDS = 600

PostdeadlineContractError = official8.PostdeadlineContractError
JsonReader = Callable[[str, Sequence[tuple[str, Any]] | None], Any]
Poster = official8.Poster

sealed = official8.sealed
validate_seal = official8.validate_seal
read_json = official8.read_json
write_immutable_json = official8.write_immutable_json
sha256_file = official8.sha256_file
file_record = official8.file_record
payload_sha256 = official8.payload_sha256
get_json = official8.get_json


def _scheduler_time(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise PostdeadlineContractError(
            "node metrics observation timestamp is absent"
        )
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc
            )
        except ValueError as exc:
            raise PostdeadlineContractError(
                "node metrics observation timestamp is malformed"
            ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _exact_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    key: str,
    expected: Any,
    label: str,
) -> dict[str, Any]:
    exact = [dict(row) for row in rows if row.get(key) == expected]
    if len(exact) != 1:
        raise PostdeadlineContractError(
            f"{label} must occur exactly once; got {len(exact)}"
        )
    return exact[0]


def capacity_query(
    *, account_name: str = ACCOUNT_NAME, node_name: str = NODE_NAME
) -> list[tuple[str, Any]]:
    return [
        ("cpus", CPUS),
        ("memory_mb", MEMORY_MB),
        ("scheduling_profile", "fea_bursty"),
        ("aedt_backend", "standalone"),
        ("required_capability", "conda:pyaedt2026v1"),
        ("env_profile", "pyaedt2026v1"),
        ("project", PROJECT),
        ("max_workers_per_node", MAX_WORKERS_PER_NODE),
        ("account_name", account_name),
        ("node_name", node_name),
        ("node_name_policy", "strict"),
    ]


def _validate_capacity(
    capacity: Any, *, label: str
) -> dict[str, Any]:
    if (
        not isinstance(capacity, dict)
        or capacity.get("queue_state") != "opening"
        or capacity.get("queue_reason") != OPENING_QUEUE_REASON
        or int(capacity.get("fit_slots") or 0) != 0
        or int(capacity.get("ready_fit_slots") or 0) != 0
        or int(capacity.get("pending_fit_slots") or 0) != 0
        or int(capacity.get("inflight_fit_slots") or 0) != 0
        or capacity.get("memory_pressure_state") != "ok"
        or capacity.get("preferred_node_relaxed") is not False
        or int(capacity.get("standalone_aedt_available") or 0) < 1
        or capacity.get("allocations") != []
    ):
        raise PostdeadlineContractError(
            f"{label} strict opening capacity gate failed"
        )
    return copy.deepcopy(capacity)


def live_preflight(
    *,
    reader: JsonReader = get_json,
    expected_dedupe_key: str,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    """Repeat all failover GET gates without mutating either project."""

    now = observed_at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise PostdeadlineContractError("preflight time must be timezone-aware")
    now = now.astimezone(timezone.utc)
    if now <= official8.ORIGINAL_DEADLINE_UTC:
        raise PostdeadlineContractError(
            "post-deadline failover cannot precede original deadline"
        )

    health = reader("/api/health", None)
    source_task = reader(f"/api/tasks/{SOURCE_TASK_ID}", None)
    original_task = reader(
        f"/api/tasks/{ORIGINAL_OFFICIAL8_TASK_ID}", None
    )
    freed_task = reader(f"/api/tasks/{FREED_TASK_ID}", None)
    capacity_n111 = reader("/api/task-capacity", capacity_query())
    capacity_n114 = reader(
        "/api/task-capacity",
        capacity_query(
            account_name=ORIGINAL_OFFICIAL8_ACCOUNT,
            node_name=ORIGINAL_OFFICIAL8_NODE,
        ),
    )
    active = reader(
        "/api/tasks",
        [
            ("status", "queued"),
            ("status", "attaching"),
            ("status", "running"),
            ("limit", 10000),
        ],
    )
    collisions = reader(
        "/api/tasks",
        [
            ("limit", 10000),
            ("project", PROJECT),
            ("name_prefix", TASK_NAME),
        ],
    )
    allocations_raw = reader("/api/allocations", [("limit", 10000)])

    if (
        not isinstance(health, dict)
        or health.get("ok") is not True
        or health.get("scheduler_ok") is not True
        or health.get("scheduler_thread_alive") is not True
        or health.get("scheduler_stalled") is not False
    ):
        raise PostdeadlineContractError("Scheduler health gate failed")

    expected_source = {
        "task_id": SOURCE_TASK_ID,
        "name": SOURCE_TASK_NAME,
        "status": "completed",
        "exit_code": 0,
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "dedupe_key": SOURCE_TASK_DEDUPE_KEY,
    }
    source_drift = {
        key: {"expected": expected, "actual": source_task.get(key)}
        for key, expected in expected_source.items()
        if not isinstance(source_task, dict)
        or source_task.get(key) != expected
    }
    if source_drift:
        raise PostdeadlineContractError(
            f"source Scheduler task identity drifted: {source_drift}"
        )

    expected_original = {
        "task_id": ORIGINAL_OFFICIAL8_TASK_ID,
        "name": ORIGINAL_OFFICIAL8_TASK_NAME,
        "status": "queued",
        "state": "queued",
        "exit_code": None,
        "dedupe_key": ORIGINAL_OFFICIAL8_DEDUPE_KEY,
        "project": PROJECT,
        "requested_account_name": ORIGINAL_OFFICIAL8_ACCOUNT,
        "requested_node_name": ORIGINAL_OFFICIAL8_NODE,
        "requested_node_name_policy": "strict",
        "preferred_node_relaxed": False,
        "allocation_id": None,
        "assigned_allocation": None,
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "aedt_backend": "standalone",
    }
    original_drift = {
        key: {"expected": expected, "actual": original_task.get(key)}
        for key, expected in expected_original.items()
        if not isinstance(original_task, dict)
        or original_task.get(key) != expected
    }
    if original_drift:
        raise PostdeadlineContractError(
            "original official #8 task is no longer exact queued n114: "
            f"{original_drift}"
        )

    expected_freed_task = {
        "task_id": FREED_TASK_ID,
        "status": "failed",
        "state": "failed",
        "exit_code": 1,
        "allocation_id": FREED_ALLOCATION_ID,
        "assigned_allocation": FREED_ALLOCATION_ID,
        "slurm_job_id": FREED_SLURM_JOB_ID,
        "requested_account_name": ACCOUNT_NAME,
        "requested_node_name": NODE_NAME,
        "requested_node_name_policy": "strict",
        "actual_node_name": NODE_NAME,
        "allocation_node_name": NODE_NAME,
    }
    freed_drift = {
        key: {"expected": expected, "actual": freed_task.get(key)}
        for key, expected in expected_freed_task.items()
        if not isinstance(freed_task, dict)
        or freed_task.get(key) != expected
    }
    if freed_drift:
        raise PostdeadlineContractError(
            f"former n111 task identity/terminal state drifted: {freed_drift}"
        )

    n111_capacity = _validate_capacity(
        capacity_n111, label=f"{ACCOUNT_NAME}/{NODE_NAME}"
    )
    n114_capacity = _validate_capacity(
        capacity_n114,
        label=f"{ORIGINAL_OFFICIAL8_ACCOUNT}/{ORIGINAL_OFFICIAL8_NODE}",
    )

    allocation_rows = official8.reviewed._task_rows(allocations_raw)
    freed_allocation = _exact_row(
        allocation_rows,
        key="id",
        expected=FREED_ALLOCATION_ID,
        label="freed n111 allocation",
    )
    if (
        freed_allocation.get("account_name") != ACCOUNT_NAME
        or freed_allocation.get("node_name") != NODE_NAME
        or str(freed_allocation.get("slurm_job_id") or "")
        != FREED_SLURM_JOB_ID
        or freed_allocation.get("state") != "closed"
        or not freed_allocation.get("closed_at")
    ):
        raise PostdeadlineContractError(
            "allocation 14644 is not the exact closed n111 allocation"
        )
    live_allocation_states = {
        "pending",
        "warm",
        "active",
        "draining",
        "closing",
    }
    live_n111 = [
        row
        for row in allocation_rows
        if row.get("node_name") == NODE_NAME
        and str(row.get("state") or "").lower() in live_allocation_states
    ]
    live_n114 = [
        row
        for row in allocation_rows
        if row.get("node_name") == ORIGINAL_OFFICIAL8_NODE
        and str(row.get("state") or "").lower() in live_allocation_states
    ]
    if live_n111 or live_n114:
        raise PostdeadlineContractError(
            "target or drained-source node still has a live allocation"
        )
    n114_history = [
        row
        for row in allocation_rows
        if row.get("node_name") == ORIGINAL_OFFICIAL8_NODE
    ]
    if not n114_history:
        raise PostdeadlineContractError(
            "Scheduler GET contains no n114 node observation"
        )
    latest_n114 = max(n114_history, key=lambda row: int(row.get("id") or 0))
    metrics_at = _scheduler_time(
        latest_n114.get("node_metrics_observed_at")
    )
    metrics_age = (now - metrics_at).total_seconds()
    if (
        latest_n114.get("node_pestat_state") != "drain"
        or metrics_age < -30
        or metrics_age > NODE_METRICS_MAX_AGE_SECONDS
    ):
        raise PostdeadlineContractError(
            "fresh Scheduler GET does not prove current n114 drain"
        )

    active_rows = official8.reviewed._task_rows(active)
    active_n111_fea = [
        row
        for row in active_rows
        if row.get("status") in {"queued", "attaching", "running"}
        and (
            row.get("node_name") == NODE_NAME
            or row.get("requested_node_name") == NODE_NAME
        )
        and (
            row.get("aedt_backend") == "standalone"
            or row.get("scheduling_profile") == "fea_bursty"
        )
    ]
    if active_n111_fea:
        raise PostdeadlineContractError("strict n111 lane is not FEA-empty")
    collision_rows = [
        row
        for row in official8.reviewed._task_rows(collisions)
        if row.get("name") == TASK_NAME
        or row.get("dedupe_key") == expected_dedupe_key
    ]
    if collision_rows:
        raise PostdeadlineContractError(
            "fresh failover task name or dedupe already exists"
        )

    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "observed_at_utc": now.isoformat(),
        "scheduler_health": health,
        "source_task_identity": source_task,
        "original_official8_task_identity": original_task,
        "freed_n111_task_identity": freed_task,
        "freed_n111_allocation": freed_allocation,
        "capacity_query": capacity_query(),
        "capacity": n111_capacity,
        "n114_capacity_query": capacity_query(
            account_name=ORIGINAL_OFFICIAL8_ACCOUNT,
            node_name=ORIGINAL_OFFICIAL8_NODE,
        ),
        "n114_capacity": n114_capacity,
        "queue_state": "opening",
        "queue_reason": OPENING_QUEUE_REASON,
        "ready_fit_slots": 0,
        "pending_fit_slots": 0,
        "inflight_fit_slots": 0,
        "preferred_node_relaxed": False,
        "standalone_aedt_available": int(
            n111_capacity["standalone_aedt_available"]
        ),
        "active_n111_fea_count": 0,
        # Compatibility alias required by the reviewed official8 plan loader.
        "active_n114_fea_count": 0,
        "collision_count": 0,
        "exact_account_node_opening": True,
        "strict_node_required": True,
        "fixed_physics_unchanged": True,
        "original_task_96329_preserved_queued": True,
        "allocation_14644_closed": True,
        "n114_drain_observed": True,
        "n114_node_metrics_observed_at_utc": metrics_at.isoformat(),
        "n114_node_metrics_age_seconds": metrics_age,
        "scheduler_get_only": True,
        "scheduler_mutation_performed": False,
    }


def _strict_readback_reader(reader: JsonReader) -> JsonReader:
    """Permit audited preflight task GETs; fence every new task to n111."""

    preflight_task_ids = {
        SOURCE_TASK_ID,
        ORIGINAL_OFFICIAL8_TASK_ID,
        FREED_TASK_ID,
    }

    def strict(
        path: str, query: Sequence[tuple[str, Any]] | None
    ) -> Any:
        value = reader(path, query)
        prefix = "/api/tasks/"
        suffix = path[len(prefix) :] if path.startswith(prefix) else ""
        if suffix.isdigit() and int(suffix) not in preflight_task_ids:
            if (
                not isinstance(value, dict)
                or value.get("requested_account_name") != ACCOUNT_NAME
                or value.get("requested_node_name") != NODE_NAME
                or value.get("requested_node_name_policy") != "strict"
                or value.get("preferred_node_relaxed") is not False
                or (
                    value.get("node_name")
                    and value.get("node_name") != NODE_NAME
                )
                or (
                    value.get("actual_node_name")
                    and value.get("actual_node_name") != NODE_NAME
                )
                or (
                    value.get("allocation_node_name")
                    and value.get("allocation_node_name") != NODE_NAME
                )
                or value.get("node_name_policy") not in {None, "strict"}
            ):
                raise PostdeadlineContractError(
                    "durable strict n111 failover readback relaxed or drifted"
                )
        return value

    return strict


_PATCH = {
    "__file__": str(Path(__file__).resolve()),
    "SCHEDULER_URL": SCHEDULER_URL,
    "PROJECT": PROJECT,
    "ACCOUNT_NAME": ACCOUNT_NAME,
    "NODE_NAME": NODE_NAME,
    "TASK_NAME": TASK_NAME,
    "WORKDIR": WORKDIR,
    "POST_AUTHORIZATION": POST_AUTHORIZATION,
    "OUTPUT_ROOT": OUTPUT_ROOT,
    "PLAN_NAME": PLAN_NAME,
    "ATTEMPT_LEDGER_NAME": ATTEMPT_LEDGER_NAME,
    "SUBMISSION_DIRECTORY_NAME": SUBMISSION_DIRECTORY_NAME,
    "PLAN_SCHEMA": PLAN_SCHEMA,
    "CANDIDATE_SCHEMA": CANDIDATE_SCHEMA,
    "PREPARE_SCHEMA": PREPARE_SCHEMA,
    "INTENT_SCHEMA": INTENT_SCHEMA,
    "SUBMISSION_SCHEMA": SUBMISSION_SCHEMA,
    "FINAL_SEAL_SCHEMA": FINAL_SEAL_SCHEMA,
    "PREFLIGHT_SCHEMA": PREFLIGHT_SCHEMA,
    "ATTEMPT_NONCE_SCHEMA": ATTEMPT_NONCE_SCHEMA,
    "SINGLE_ATTEMPT_SCHEMA": SINGLE_ATTEMPT_SCHEMA,
    "SAFETY_FLAGS": SAFETY_FLAGS,
    "FIXED_BOUNDARY": FIXED_BOUNDARY,
    "BOUNDARY_PROJECTION": BOUNDARY_PROJECTION,
    "capacity_query": capacity_query,
    "live_preflight": live_preflight,
    "_strict_readback_reader": _strict_readback_reader,
}


@contextmanager
def _failover_contract() -> Iterator[None]:
    old = {name: getattr(official8, name) for name in _PATCH}
    try:
        for name, value in _PATCH.items():
            setattr(official8, name, value)
        yield
    finally:
        for name, value in old.items():
            setattr(official8, name, value)


def load_plan(
    plan_path: Path,
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    with _failover_contract():
        return official8.load_plan(plan_path, **kwargs)


def _collector_intent(plan_path: Path) -> Path:
    plan = validate_seal(read_json(plan_path), PLAN_SCHEMA)
    intent_path = plan_path.parent / COLLECTOR_INTENT_NAME
    value = sealed(
        {
            "schema_version": COLLECTOR_INTENT_SCHEMA,
            **SAFETY_FLAGS,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan": file_record(plan_path),
            "plan_payload_sha256": plan["payload_sha256"],
            "candidate_physics_sha256": CANDIDATE_SHA256,
            "task_name": TASK_NAME,
            "dedupe_key": plan["dedupe_key"],
            "expected_account_name": ACCOUNT_NAME,
            "expected_node_name": NODE_NAME,
            "expected_node_name_policy": "strict",
            "original_task_id": ORIGINAL_OFFICIAL8_TASK_ID,
            "original_task_must_remain_queued": True,
            "collector_adapter": (
                "tools/mft_goal_official_standard8_n111_failover_collector.py"
            ),
            "collector_manifest_path": str(
                plan_path.parent / COLLECTOR_MANIFEST_NAME
            ),
            "scheduler_get_only_collector": True,
            "collector_scheduler_mutation_allowed": False,
        }
    )
    return write_immutable_json(intent_path, value)


def prepare() -> Path:
    with _failover_contract():
        plan_path = official8.prepare()
    _collector_intent(plan_path)
    return plan_path


def seal_prepost_adapter_abort(
    *, reader: JsonReader = get_json
) -> Path:
    """Seal proof that the adapter failure happened before POST consumption."""

    root = OUTPUT_ROOT.resolve(strict=True)
    path = root / PREPOST_ABORT_NAME
    if path.exists():
        value = validate_seal(read_json(path), PREPOST_ABORT_SCHEMA)
        if (
            value.get("scheduler_post_calls_evidenced") != 0
            or value.get("post_call_budget_consumed") is not False
        ):
            raise PostdeadlineContractError(
                "existing pre-POST abort evidence drifted"
            )
        return path
    plan_path = root / PLAN_NAME
    plan = validate_seal(read_json(plan_path), PLAN_SCHEMA)
    attempt_path = root / ATTEMPT_LEDGER_NAME
    submission_path = root / SUBMISSION_DIRECTORY_NAME
    if attempt_path.exists() or submission_path.exists():
        raise PostdeadlineContractError(
            "cannot seal pre-POST abort after attempt/output creation"
        )
    inventory = reader(
        "/api/tasks",
        [
            ("limit", 10000),
            ("project", PROJECT),
            ("name_prefix", TASK_NAME),
        ],
    )
    exact = [
        row
        for row in official8.reviewed._task_rows(inventory)
        if row.get("name") == TASK_NAME
        or row.get("dedupe_key") == plan["dedupe_key"]
    ]
    if exact:
        raise PostdeadlineContractError(
            "cannot prove zero POST because failover task exists"
        )
    original_task = reader(
        f"/api/tasks/{ORIGINAL_OFFICIAL8_TASK_ID}", None
    )
    if (
        not isinstance(original_task, dict)
        or original_task.get("task_id") != ORIGINAL_OFFICIAL8_TASK_ID
        or original_task.get("status") != "queued"
        or original_task.get("dedupe_key")
        != ORIGINAL_OFFICIAL8_DEDUPE_KEY
    ):
        raise PostdeadlineContractError(
            "original official #8 task changed during pre-POST abort"
        )
    evidence = sealed(
        {
            "schema_version": PREPOST_ABORT_SCHEMA,
            **SAFETY_FLAGS,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan": file_record(plan_path),
            "plan_payload_sha256": plan["payload_sha256"],
            "task_name": TASK_NAME,
            "dedupe_key": plan["dedupe_key"],
            "failed_adapter_stage": "initial_live_preflight_before_output",
            "failure_class": "strict_readback_preflight_allowlist_bug",
            "failure_message": (
                "durable strict n114 readback relaxed or drifted"
            ),
            "attempt_ledger_path": str(attempt_path),
            "attempt_ledger_exists": False,
            "submission_output_path": str(submission_path),
            "submission_output_exists": False,
            "exact_scheduler_collision_count": 0,
            "scheduler_post_calls_evidenced": 0,
            "post_call_budget_consumed": False,
            "remaining_lifetime_post_budget": 1,
            "retry_only_after_adapter_fix_and_full_revalidation": True,
            "original_task_96329_snapshot": original_task,
            "original_task_96329_snapshot_sha256": payload_sha256(
                original_task
            ),
            "original_task_cancelled_or_modified": False,
            "scheduler_get_only": True,
            "scheduler_mutation_performed": False,
            "scheduler_repository_modified": False,
            "scheduler_project_mutation_performed": False,
        }
    )
    return write_immutable_json(path, evidence)


def _collector_manifest(final_path: Path) -> Path:
    final = validate_seal(read_json(final_path), FINAL_SEAL_SCHEMA)
    root = OUTPUT_ROOT.resolve()
    plan_path = root / PLAN_NAME
    intent_path = root / SUBMISSION_DIRECTORY_NAME / "submission_intent.json"
    attempt_path = root / ATTEMPT_LEDGER_NAME
    receipt_path = (
        root / SUBMISSION_DIRECTORY_NAME / "submission_receipt.json"
    )
    collector_intent_path = root / COLLECTOR_INTENT_NAME
    plan = validate_seal(read_json(plan_path), PLAN_SCHEMA)
    intent = validate_seal(read_json(intent_path), INTENT_SCHEMA)
    attempt = validate_seal(
        read_json(attempt_path),
        "mft-goal-official-standard-postdeadline-scheduler-post-attempt-v1",
    )
    receipt = validate_seal(read_json(receipt_path), SUBMISSION_SCHEMA)
    collector_intent = validate_seal(
        read_json(collector_intent_path), COLLECTOR_INTENT_SCHEMA
    )
    task_id = int(final["task_id"])
    task = get_json(f"/api/tasks/{task_id}", None)
    if (
        not isinstance(task, dict)
        or task.get("task_id") != task_id
        or task.get("name") != TASK_NAME
        or task.get("dedupe_key") != plan["dedupe_key"]
        or task.get("requested_account_name") != ACCOUNT_NAME
        or task.get("requested_node_name") != NODE_NAME
        or task.get("requested_node_name_policy") != "strict"
        or task.get("preferred_node_relaxed") is not False
    ):
        raise PostdeadlineContractError(
            "collector manifest task GET identity drifted"
        )
    manifest_path = root / COLLECTOR_MANIFEST_NAME
    output = root / f"authenticated_get_collection_task{task_id}_v1"
    command = [
        sys.executable,
        str(
            REPOSITORY
            / "tools"
            / "mft_goal_official_standard8_n111_failover_collector.py"
        ),
        "--manifest",
        str(manifest_path),
        "--output",
        str(output),
        "--watch",
        "--interval",
        "60",
    ]
    manifest = sealed(
        {
            "schema_version": COLLECTOR_MANIFEST_SCHEMA,
            **SAFETY_FLAGS,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_physics_sha256": CANDIDATE_SHA256,
            "task_id": task_id,
            "task_name": TASK_NAME,
            "dedupe_key": plan["dedupe_key"],
            "account_name": ACCOUNT_NAME,
            "node_name": NODE_NAME,
            "node_name_policy": "strict",
            "preferred_node_relaxed": False,
            "task_readback": task,
            "task_readback_sha256": payload_sha256(task),
            "plan": file_record(plan_path),
            "plan_payload_sha256": plan["payload_sha256"],
            "submission_intent": file_record(intent_path),
            "submission_intent_payload_sha256": intent["payload_sha256"],
            "attempt_ledger": file_record(attempt_path),
            "attempt_ledger_payload_sha256": attempt["payload_sha256"],
            "submission_receipt": file_record(receipt_path),
            "submission_receipt_payload_sha256": receipt["payload_sha256"],
            "final_seal": file_record(final_path),
            "final_seal_payload_sha256": final["payload_sha256"],
            "collector_intent": file_record(collector_intent_path),
            "collector_intent_payload_sha256": collector_intent[
                "payload_sha256"
            ],
            "collector_output": str(output),
            "collector_command_argv": command,
            "scheduler_get_only_collector": True,
            "collector_scheduler_mutation_allowed": False,
            "scheduler_repository_modified": False,
            "scheduler_project_mutation_performed": False,
            "original_task_id": ORIGINAL_OFFICIAL8_TASK_ID,
            "original_task_cancelled_or_modified": False,
            "scheduler_post_call_budget": 1,
            "scheduler_post_calls_evidenced": 1,
            "retry_allowed": False,
        }
    )
    return write_immutable_json(manifest_path, manifest)


def submit(
    *,
    authorize_post: str,
    reader: JsonReader = get_json,
    poster: Poster = official8.reviewed._post_json_once,
    observed_at: datetime | None = None,
    lock_factory: Callable[[], Any] = official8.scheduler_campaign_lock,
) -> Path:
    if authorize_post != POST_AUTHORIZATION:
        raise PostdeadlineContractError(
            "explicit one-shot strict-n111 failover authorization is absent"
        )
    with _failover_contract():
        final_path = official8.submit(
            authorize_post=authorize_post,
            reader=reader,
            poster=poster,
            observed_at=observed_at,
            lock_factory=lock_factory,
        )
    _collector_manifest(final_path)
    return final_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("inspect")
    commands.add_parser("seal-prepost-abort")
    submit_parser = commands.add_parser("submit")
    submit_parser.add_argument("--authorize-post", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    plan_path = OUTPUT_ROOT / PLAN_NAME
    if args.command == "prepare":
        plan_path = prepare()
        plan = validate_seal(read_json(plan_path), PLAN_SCHEMA)
        result = {
            "event": "prepared",
            "plan": str(plan_path),
            "plan_sha256": sha256_file(plan_path),
            "plan_payload_sha256": plan["payload_sha256"],
            "candidate_physics_sha256": CANDIDATE_SHA256,
            "task_name": TASK_NAME,
            "dedupe_key": plan["dedupe_key"],
            "account_name": ACCOUNT_NAME,
            "node_name": NODE_NAME,
            "scheduler_post_calls": 0,
        }
    elif args.command == "inspect":
        plan, _params, _profile = load_plan(plan_path)
        preflight = live_preflight(
            expected_dedupe_key=plan["dedupe_key"]
        )
        result = {
            "event": "ready",
            "plan": str(plan_path),
            "task_name": TASK_NAME,
            "dedupe_key": plan["dedupe_key"],
            "original_task_96329_preserved_queued": preflight[
                "original_task_96329_preserved_queued"
            ],
            "allocation_14644_closed": preflight[
                "allocation_14644_closed"
            ],
            "n114_drain_observed": preflight["n114_drain_observed"],
            "active_n111_fea_count": preflight[
                "active_n111_fea_count"
            ],
            "scheduler_post_calls": 0,
        }
    elif args.command == "seal-prepost-abort":
        evidence_path = seal_prepost_adapter_abort()
        evidence = validate_seal(
            read_json(evidence_path), PREPOST_ABORT_SCHEMA
        )
        result = {
            "event": "prepost_abort_sealed",
            "evidence": str(evidence_path),
            "evidence_sha256": sha256_file(evidence_path),
            "scheduler_post_calls_evidenced": evidence[
                "scheduler_post_calls_evidenced"
            ],
            "remaining_lifetime_post_budget": evidence[
                "remaining_lifetime_post_budget"
            ],
        }
    else:
        final_path = submit(authorize_post=args.authorize_post)
        final = validate_seal(read_json(final_path), FINAL_SEAL_SCHEMA)
        manifest_path = OUTPUT_ROOT / COLLECTOR_MANIFEST_NAME
        result = {
            "event": "submitted",
            "final_seal": str(final_path),
            "final_seal_sha256": sha256_file(final_path),
            "collector_manifest": str(manifest_path),
            "collector_manifest_sha256": sha256_file(manifest_path),
            "task_id": final["task_id"],
            "scheduler_post_calls": final["scheduler_post_calls"],
        }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
