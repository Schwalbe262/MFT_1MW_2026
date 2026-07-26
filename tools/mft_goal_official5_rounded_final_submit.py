"""Fail-closed one-shot submitter for the rounded candidate-5 Standard lane.

This is the only mutation-capable companion to
``mft_goal_official5_rounded_final_prepare.py``.  It reauthenticates the
sealed plan, proves that the independent ``dw16/n113`` allocation still has
enough runtime, records an immutable intent and consumed-attempt
ledger, and can then issue exactly one Scheduler POST.

An uncertain POST result is never retried.  The consumed attempt may only be
reconciled by an exact name-plus-dedupe GET.
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from tools import (  # noqa: E402
    mft_goal_official5_rounded_final_prepare as prepare_only,
)


direct = prepare_only.direct
reviewed = prepare_only.reviewed
PostdeadlineContractError = prepare_only.PostdeadlineContractError
JsonReader = direct.JsonReader
Poster = Callable[
    [str, Mapping[str, Any]],
    tuple[int | None, dict[str, Any] | None, str | None],
]

SCHEDULER_URL = prepare_only.SCHEDULER_URL
PROJECT = prepare_only.PROJECT
OUTPUT_ROOT = prepare_only.OUTPUT_ROOT
PLAN_NAME = prepare_only.PLAN_NAME
TASK_NAME = prepare_only.TASK_NAME
ACCOUNT_NAME = prepare_only.ACCOUNT_NAME
NODE_NAME = prepare_only.NODE_NAME
SAME_NODE_AS_TASK_ID = prepare_only.SAME_NODE_AS_TASK_ID
SOURCE_ALLOCATION_ID = prepare_only.SOURCE_ALLOCATION_ID
SOURCE_SLURM_JOB_ID = prepare_only.SOURCE_SLURM_JOB_ID
CPUS = prepare_only.CPUS
MEMORY_MB = prepare_only.MEMORY_MB
SCHEDULER_SECONDS = prepare_only.SCHEDULER_SECONDS
MAX_WORKERS_PER_NODE = prepare_only.MAX_WORKERS_PER_NODE
CANDIDATE_SHA256 = prepare_only.SOURCE_CANDIDATE_SHA256
SAFETY_FLAGS = copy.deepcopy(direct.SAFETY_FLAGS)

SOURCE_ANCHOR_TASK_NAME = (
    "mft-goal-diag-standard-postdeadline-official6-"
    "s96185-2772aed82a8c-n113"
)
SOURCE_ANCHOR_DEDUPE_KEY = (
    "mft-al:mft-goal-diag-standard-postdeadline-official6-"
    "s96185-2772aed82a8c-n113:"
    "a1e4f70cefa1af04673c73a6131bf490c0cc14b5:"
    "e6b9b9d20a832ff5c3f7ca97218737a0b8650781:"
    "b713a9235ad804af"
)
ALLOCATION_FORCE_END_UTC = datetime(
    2026, 7, 26, 19, 7, tzinfo=timezone.utc
)
MIN_FORCE_END_BUFFER_SECONDS = 5 * 60

POST_AUTHORIZATION = "authorize-official5-rounded-final-one-post-v1"
SUBMISSION_DIRECTORY_NAME = "submission"
INTENT_NAME = "scheduler_submission_intent.json"
ATTEMPT_NAME = "scheduler_post_attempt.json"
RECEIPT_NAME = "scheduler_submission_receipt.json"
FINAL_NAME = "rounded_submission_final.json"
AMBIGUOUS_NAME = "scheduler_post_ambiguous.json"

LIVE_GATE_SCHEMA = "mft-goal-rounded-final-live-gate-v1"
INTENT_SCHEMA = "mft-goal-rounded-final-submit-intent-v1"
ATTEMPT_SCHEMA = "mft-goal-rounded-final-submit-attempt-v1"
RECEIPT_SCHEMA = "mft-goal-rounded-final-submit-receipt-v1"
FINAL_SCHEMA = "mft-goal-rounded-final-submit-final-v1"
AMBIGUOUS_SCHEMA = "mft-goal-rounded-final-submit-ambiguous-v1"

payload_sha256 = direct.payload_sha256
sealed = direct.sealed
validate_seal = direct.validate_seal
file_record = direct.file_record
read_json = direct.read_json
write_immutable_json = direct.write_immutable_json
write_exclusive_json = direct.write_exclusive_json
get_json = direct.get_json
scheduler_campaign_lock = direct.scheduler_campaign_lock


def capacity_query() -> list[tuple[str, Any]]:
    """Return the exact same-allocation admission query."""

    return [
        ("cpus", CPUS),
        ("memory_mb", MEMORY_MB),
        ("scheduling_profile", "fea_bursty"),
        ("aedt_backend", "standalone"),
        ("required_capability", "conda:pyaedt2026v1"),
        ("env_profile", "pyaedt2026v1"),
        ("project", PROJECT),
        ("max_workers_per_node", MAX_WORKERS_PER_NODE),
        ("account_name", ACCOUNT_NAME),
        ("node_name", NODE_NAME),
        ("same_node_as_task_id", SAME_NODE_AS_TASK_ID),
        ("requested_allocation_id", SOURCE_ALLOCATION_ID),
    ]


def _task_rows(value: Any) -> list[dict[str, Any]]:
    return direct._task_rows(value)


def _capability_accounts(value: Any) -> set[str]:
    return direct._capability_accounts(value)


def _account_rows(value: Any) -> list[dict[str, Any]]:
    return direct._account_rows(value)


def _existing_allocation_account_gate(
    account: Mapping[str, Any],
) -> dict[str, Any]:
    """Gate caps for an attach that creates no additional Slurm job."""

    values = {
        key: int(account.get(key) or 0)
        for key in (
            "running",
            "pending",
            "max_running",
            "max_pending",
            "max_total",
        )
    }
    if (
        values["max_running"] < 1
        or values["max_pending"] < 1
        or values["max_total"] < 1
        or values["pending"] >= values["max_pending"]
        or values["running"] + values["pending"]
        >= values["max_total"]
    ):
        raise PostdeadlineContractError(
            "dw16 account pending/total cap gate failed"
        )
    return {
        **values,
        "running_cap_saturated": (
            values["running"] >= values["max_running"]
        ),
        "new_slurm_job_requested": False,
        "existing_allocation_attach_only": True,
        "requested_allocation_id": SOURCE_ALLOCATION_ID,
        "storage": {
            "mode": "scheduler_fail_closed_runtime_guard",
            "storage_path": str(account.get("storage_path") or ""),
            "quota_fields_unset": (
                account.get("storage_quota_gb") is None
                and account.get("storage_used_gb") is None
            ),
        },
    }


def _inventory_exact(
    reader: JsonReader,
    *,
    dedupe_key: str,
) -> list[dict[str, Any]]:
    rows = _task_rows(
        reader(
            "/api/tasks",
            [
                ("limit", 10000),
                ("project", PROJECT),
                ("name_prefix", TASK_NAME),
            ],
        )
    )
    return [
        row
        for row in rows
        if row.get("name") == TASK_NAME
        and row.get("dedupe_key") == dedupe_key
    ]


def _active_target_fea(value: Any) -> list[dict[str, Any]]:
    return [
        row
        for row in _task_rows(value)
        if row.get("status") in direct.ACTIVE_TASK_STATES
        and (
            row.get("node_name") == NODE_NAME
            or row.get("requested_node_name") == NODE_NAME
            or row.get("actual_node_name") == NODE_NAME
        )
        and (
            row.get("aedt_backend") == "standalone"
            or row.get("scheduling_profile") == "fea_bursty"
        )
    ]


def live_gate(
    *,
    expected_dedupe_key: str,
    reader: JsonReader = get_json,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    """Prove n113 can run this lane now; perform GETs only."""

    now = observed_at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise PostdeadlineContractError(
            "rounded submit gate time must be timezone-aware"
        )
    now = now.astimezone(timezone.utc)
    health = reader("/api/health", None)
    anchor = reader(f"/api/tasks/{SAME_NODE_AS_TASK_ID}", None)
    capacity = reader("/api/task-capacity", capacity_query())
    allocations = reader("/api/allocations", [("limit", 10000)])
    licenses = reader("/api/licenses", None)
    accounts = reader("/api/accounts/status/live", None)
    capabilities = reader("/api/capabilities", None)
    active = reader(
        "/api/tasks",
        [
            ("status", "queued"),
            ("status", "attaching"),
            ("status", "running"),
            ("limit", 10000),
        ],
    )
    inventory = reader(
        "/api/tasks",
        [
            ("limit", 10000),
            ("project", PROJECT),
            ("name_prefix", TASK_NAME),
        ],
    )
    if (
        not isinstance(health, Mapping)
        or health.get("ok") is not True
        or health.get("scheduler_ok") is not True
        or health.get("scheduler_thread_alive") is not True
        or health.get("scheduler_stalled") is not False
    ):
        raise PostdeadlineContractError("Scheduler health gate failed")
    expected_anchor = {
        "task_id": SAME_NODE_AS_TASK_ID,
        "name": SOURCE_ANCHOR_TASK_NAME,
        "dedupe_key": SOURCE_ANCHOR_DEDUPE_KEY,
        "project": PROJECT,
        "status": "failed",
        "requested_account_name": ACCOUNT_NAME,
        "requested_node_name": NODE_NAME,
        "account_name": ACCOUNT_NAME,
        "node_name": NODE_NAME,
        "actual_node_name": NODE_NAME,
        "allocation_node_name": NODE_NAME,
        "allocation_id": SOURCE_ALLOCATION_ID,
        "assigned_allocation": SOURCE_ALLOCATION_ID,
        "slurm_job_id": SOURCE_SLURM_JOB_ID,
        "cpus": CPUS,
        "memory_mb": MEMORY_MB,
        "aedt_backend": "standalone",
        "scheduling_profile": "fea_bursty",
    }
    anchor_drift = {
        key: {"expected": expected, "actual": anchor.get(key)}
        for key, expected in expected_anchor.items()
        if not isinstance(anchor, Mapping) or anchor.get(key) != expected
    }
    if anchor_drift:
        raise PostdeadlineContractError(
            f"task96328 n113 anchor drifted: {anchor_drift}"
        )

    account_matches = [
        row
        for row in _account_rows(accounts)
        if row.get("account_name") == ACCOUNT_NAME
    ]
    if len(account_matches) != 1:
        raise PostdeadlineContractError(
            "dw16 account status is absent or ambiguous"
        )
    account_gate = _existing_allocation_account_gate(
        account_matches[0]
    )
    if ACCOUNT_NAME not in _capability_accounts(capabilities):
        raise PostdeadlineContractError(
            "dw16 lacks conda:pyaedt2026v1 capability"
        )
    license_gate = direct.source_submit._license_gate(
        licenses, required_headroom=1
    )

    allocation_rows = [
        row
        for row in _task_rows(allocations)
        if int(row.get("id") or row.get("allocation_id") or 0)
        == SOURCE_ALLOCATION_ID
    ]
    if len(allocation_rows) != 1:
        raise PostdeadlineContractError(
            "n113 source allocation is absent or ambiguous"
        )
    allocation = allocation_rows[0]
    allocation_drift = {
        key: {"expected": expected, "actual": allocation.get(key)}
        for key, expected in {
            "account_name": ACCOUNT_NAME,
            "node_name": NODE_NAME,
            "slurm_job_id": SOURCE_SLURM_JOB_ID,
        }.items()
        if allocation.get(key) != expected
    }
    if (
        allocation_drift
        or allocation.get("state") != "active"
        or allocation.get("node_pestat_state") not in {"idle", "mix"}
        or int(allocation.get("node_memory_free_mb") or 0) < MEMORY_MB
        or int(allocation.get("node_cpu_total") or 0)
        - int(allocation.get("node_cpu_used") or 0)
        < CPUS
        or allocation.get("node_memory_pressure_state") not in {None, "ok"}
    ):
        raise PostdeadlineContractError(
            f"n113 allocation resource drifted: {allocation_drift}"
        )
    remaining_seconds = (
        ALLOCATION_FORCE_END_UTC - now
    ).total_seconds()
    if remaining_seconds < (
        SCHEDULER_SECONDS + MIN_FORCE_END_BUFFER_SECONDS
    ):
        raise PostdeadlineContractError(
            "n113 allocation lacks remaining runtime for the sealed "
            "3-hour solve plus retention"
        )

    capacity_allocations = (
        capacity.get("allocations")
        if isinstance(capacity, Mapping)
        else None
    )
    if (
        not isinstance(capacity, Mapping)
        or capacity.get("queue_state") != "ready"
        or capacity.get("queue_reason")
        != f"ready to attach to allocation {SOURCE_ALLOCATION_ID}"
        or int(capacity.get("ready_fit_slots") or 0) < 1
        or int(capacity.get("pending_fit_slots") or 0) != 0
        or int(capacity.get("inflight_fit_slots") or 0) < 1
        or capacity.get("memory_pressure_state") != "ok"
        or capacity.get("preferred_node_relaxed") is not False
        or int(capacity.get("standalone_aedt_available") or 0) < 1
        or not isinstance(capacity_allocations, list)
        or len(capacity_allocations) != 1
        or int(capacity_allocations[0].get("allocation_id") or 0)
        != SOURCE_ALLOCATION_ID
        or capacity_allocations[0].get("account_name") != ACCOUNT_NAME
        or capacity_allocations[0].get("node_name") != NODE_NAME
        or capacity_allocations[0].get("memory_pressure_state") != "ok"
        or int(capacity_allocations[0].get("fit_slots") or 0) < 1
        or int(capacity_allocations[0].get("free_cpus") or 0) < CPUS
        or int(capacity_allocations[0].get("free_memory_mb") or 0)
        < MEMORY_MB
    ):
        raise PostdeadlineContractError(
            "exact task96328 same-node capacity is not ready"
        )

    active_fea = _active_target_fea(active)
    active_ids = {
        int(row.get("task_id") or row.get("id") or 0)
        for row in active_fea
    }
    if active_ids:
        raise PostdeadlineContractError(
            f"n113 active FEA set drifted: {sorted(active_ids)}"
        )
    collisions = [
        row
        for row in _task_rows(inventory)
        if row.get("name") == TASK_NAME
        or row.get("dedupe_key") == expected_dedupe_key
    ]
    if collisions:
        raise PostdeadlineContractError(
            "rounded task name or dedupe already exists"
        )
    return sealed(
        {
            "schema_version": LIVE_GATE_SCHEMA,
            **SAFETY_FLAGS,
            "observed_at_utc": now.isoformat(),
            "task_name": TASK_NAME,
            "dedupe_key": expected_dedupe_key,
            "account_name": ACCOUNT_NAME,
            "node_name": NODE_NAME,
            "node_name_policy": "strict",
            "same_node_as_task_id": SAME_NODE_AS_TASK_ID,
            "source_allocation_id": SOURCE_ALLOCATION_ID,
            "source_slurm_job_id": SOURCE_SLURM_JOB_ID,
            "scheduler_health": copy.deepcopy(dict(health)),
            "anchor_task96328": copy.deepcopy(dict(anchor)),
            "capacity_query": capacity_query(),
            "capacity": copy.deepcopy(dict(capacity)),
            "allocation": copy.deepcopy(dict(allocation)),
            "allocation_force_end_utc": (
                ALLOCATION_FORCE_END_UTC.isoformat()
            ),
            "allocation_remaining_seconds": remaining_seconds,
            "minimum_force_end_buffer_seconds": (
                MIN_FORCE_END_BUFFER_SECONDS
            ),
            "account_gate": account_gate,
            "license_gate": license_gate,
            "active_target_node_fea_task_ids": sorted(active_ids),
            "collision_count": 0,
            "parallel_to_task96338": True,
            "scheduler_get_only": True,
            "scheduler_post_calls": 0,
        }
    )


def _authenticated_readback(
    reader: JsonReader,
    *,
    task_id: int,
    dedupe_key: str,
) -> dict[str, Any]:
    value = reader(f"/api/tasks/{task_id}", None)
    expected = {
        "task_id": task_id,
        "name": TASK_NAME,
        "dedupe_key": dedupe_key,
        "project": PROJECT,
        "requested_account_name": ACCOUNT_NAME,
        "requested_node_name": NODE_NAME,
        "requested_node_name_policy": "strict",
        "preferred_node_relaxed": False,
        "same_node_as_task_id": SAME_NODE_AS_TASK_ID,
        "requested_allocation_id": SOURCE_ALLOCATION_ID,
        "cpus": CPUS,
        "memory_mb": MEMORY_MB,
        "timeout_seconds": SCHEDULER_SECONDS,
        "max_workers_per_node": MAX_WORKERS_PER_NODE,
        "aedt_backend": "standalone",
    }
    drift = {
        key: {"expected": expected_value, "actual": value.get(key)}
        for key, expected_value in expected.items()
        if not isinstance(value, Mapping)
        or value.get(key) != expected_value
    }
    if (
        drift
        or value.get("status") not in {"queued", "attaching", "running"}
        or (
            value.get("account_name") is not None
            and value.get("account_name") != ACCOUNT_NAME
        )
        or (
            value.get("node_name") is not None
            and value.get("node_name") != NODE_NAME
        )
    ):
        raise PostdeadlineContractError(
            f"rounded Scheduler readback drifted: {drift}"
        )
    if value.get("status") in {"attaching", "running"}:
        live_expected = {
            "allocation_id": SOURCE_ALLOCATION_ID,
            "assigned_allocation": SOURCE_ALLOCATION_ID,
            "slurm_job_id": SOURCE_SLURM_JOB_ID,
            "actual_node_name": NODE_NAME,
            "allocation_node_name": NODE_NAME,
        }
        live_drift = {
            key: {"expected": expected, "actual": value.get(key)}
            for key, expected in live_expected.items()
            if value.get(key) != expected
        }
        if live_drift:
            raise PostdeadlineContractError(
                f"rounded live placement drifted: {live_drift}"
            )
    return copy.deepcopy(dict(value))


def _attempt_nonce(plan: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "plan_payload_sha256": plan["payload_sha256"],
                "scheduler_payload_sha256": plan[
                    "scheduler_payload_sha256"
                ],
                "authorization": POST_AUTHORIZATION,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _write_ambiguous(
    *,
    root: Path,
    plan: Mapping[str, Any],
    attempt_path: Path,
    http_status: int | None,
    response: Mapping[str, Any] | None,
    error: str | None,
    exact_count: int,
) -> Path:
    path = root / AMBIGUOUS_NAME
    if path.exists():
        validate_seal(read_json(path), AMBIGUOUS_SCHEMA)
        return path
    return write_immutable_json(
        path,
        sealed(
            {
                "schema_version": AMBIGUOUS_SCHEMA,
                **SAFETY_FLAGS,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "plan_payload_sha256": plan["payload_sha256"],
                "attempt_ledger": file_record(attempt_path),
                "http_status": http_status,
                "http_response": copy.deepcopy(response),
                "http_error": error,
                "exact_get_reconciliation_count": exact_count,
                "scheduler_post_calls": 1,
                "attempt_consumed": True,
                "retry_allowed": False,
                "automatic_full_trigger": False,
            }
        ),
    )


def _finish(
    *,
    root: Path,
    plan_path: Path,
    plan: Mapping[str, Any],
    intent_path: Path,
    attempt_path: Path,
    task_id: int,
    readback: Mapping[str, Any],
    http_status: int | None,
    response: Mapping[str, Any] | None,
    error: str | None,
    reconciled: bool,
) -> Path:
    receipt_value = sealed(
        {
            "schema_version": RECEIPT_SCHEMA,
            **SAFETY_FLAGS,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan": file_record(plan_path),
            "plan_payload_sha256": plan["payload_sha256"],
            "intent": file_record(intent_path),
            "attempt_ledger": file_record(attempt_path),
            "candidate_physics_sha256": CANDIDATE_SHA256,
            "solver_revision": plan["solver_revision"],
            "scheduler_payload_sha256": plan[
                "scheduler_payload_sha256"
            ],
            "scheduler_post_calls": 1,
            "scheduler_post_http_status": http_status,
            "scheduler_post_response": copy.deepcopy(response),
            "scheduler_post_error": error,
            "exact_get_reconciled": reconciled,
            "task_id": task_id,
            "task_name": TASK_NAME,
            "dedupe_key": plan["scheduler_payload"]["dedupe_key"],
            "account_name": ACCOUNT_NAME,
            "node_name": NODE_NAME,
            "same_node_as_task_id": SAME_NODE_AS_TASK_ID,
            "same_node_as_allocation_id": SOURCE_ALLOCATION_ID,
            "parallel_to_task96338": True,
            "task_readback": copy.deepcopy(dict(readback)),
            "task_readback_sha256": payload_sha256(readback),
            "scheduler_repository_modified": False,
            "automatic_full_trigger": False,
        }
    )
    receipt_path = root / RECEIPT_NAME
    if receipt_path.exists():
        receipt = validate_seal(
            read_json(receipt_path), RECEIPT_SCHEMA
        )
        if receipt.get("task_id") != task_id:
            raise PostdeadlineContractError(
                "rounded receipt task identity drifted"
            )
    else:
        receipt_path = write_immutable_json(receipt_path, receipt_value)
        receipt = receipt_value
    final_value = sealed(
        {
            "schema_version": FINAL_SCHEMA,
            **SAFETY_FLAGS,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan": file_record(plan_path),
            "intent": file_record(intent_path),
            "attempt_ledger": file_record(attempt_path),
            "submission_receipt": file_record(receipt_path),
            "submission_receipt_payload_sha256": receipt[
                "payload_sha256"
            ],
            "task_id": task_id,
            "scheduler_post_call_budget": 1,
            "scheduler_post_calls_evidenced": 1,
            "retry_allowed": False,
            "full_model_started": False,
            "immutable_evidence_complete": True,
        }
    )
    final_path = root / FINAL_NAME
    if final_path.exists():
        final = validate_seal(read_json(final_path), FINAL_SCHEMA)
        if final.get("task_id") != task_id:
            raise PostdeadlineContractError(
                "rounded final task identity drifted"
            )
        return final_path
    return write_immutable_json(final_path, final_value)


def _reconcile_consumed(
    *,
    plan_path: Path,
    plan: Mapping[str, Any],
    reader: JsonReader,
) -> Path:
    root = OUTPUT_ROOT.resolve() / SUBMISSION_DIRECTORY_NAME
    intent_path = root / INTENT_NAME
    attempt_path = OUTPUT_ROOT.resolve() / ATTEMPT_NAME
    if not intent_path.is_file() or not attempt_path.is_file():
        raise PostdeadlineContractError(
            "consumed rounded attempt lacks intent/ledger"
        )
    validate_seal(read_json(intent_path), INTENT_SCHEMA)
    validate_seal(read_json(attempt_path), ATTEMPT_SCHEMA)
    dedupe_key = str(plan["scheduler_payload"]["dedupe_key"])
    exact = _inventory_exact(reader, dedupe_key=dedupe_key)
    if len(exact) != 1:
        _write_ambiguous(
            root=root,
            plan=plan,
            attempt_path=attempt_path,
            http_status=None,
            response=None,
            error="consumed attempt exact GET is not unique",
            exact_count=len(exact),
        )
        raise PostdeadlineContractError(
            "rounded POST already consumed; exact GET is not unique"
        )
    task_id = int(exact[0].get("task_id") or exact[0].get("id") or 0)
    readback = _authenticated_readback(
        reader, task_id=task_id, dedupe_key=dedupe_key
    )
    return _finish(
        root=root,
        plan_path=plan_path,
        plan=plan,
        intent_path=intent_path,
        attempt_path=attempt_path,
        task_id=task_id,
        readback=readback,
        http_status=None,
        response=None,
        error="reconciled consumed attempt by exact GET",
        reconciled=True,
    )


def submit(
    *,
    authorize_post: str,
    plan_path: Path | None = None,
    reader: JsonReader = get_json,
    poster: Poster = reviewed._post_json_once,
    observed_at: datetime | None = None,
    lock_factory: Callable[[], Any] = scheduler_campaign_lock,
    plan_loader: Callable[[Path | None], dict[str, Any]] = (
        prepare_only.load_plan
    ),
) -> Path:
    """Consume at most one Scheduler POST for the sealed rounded lane."""

    if authorize_post != POST_AUTHORIZATION:
        raise PostdeadlineContractError(
            "explicit rounded one-shot authorization is absent"
        )
    resolved_plan = (
        plan_path or (OUTPUT_ROOT / PLAN_NAME)
    ).resolve(strict=True)
    if resolved_plan != OUTPUT_ROOT.resolve() / PLAN_NAME:
        raise PostdeadlineContractError(
            "rounded plan path escaped the fixed output root"
        )
    plan = plan_loader(resolved_plan)
    payload = plan["scheduler_payload"]
    dedupe_key = str(payload["dedupe_key"])
    root = OUTPUT_ROOT.resolve() / SUBMISSION_DIRECTORY_NAME
    attempt_path = OUTPUT_ROOT.resolve() / ATTEMPT_NAME
    final_path = root / FINAL_NAME
    if final_path.is_file():
        final = validate_seal(read_json(final_path), FINAL_SCHEMA)
        _authenticated_readback(
            reader,
            task_id=int(final["task_id"]),
            dedupe_key=dedupe_key,
        )
        return final_path
    if attempt_path.exists():
        return _reconcile_consumed(
            plan_path=resolved_plan, plan=plan, reader=reader
        )
    initial_gate = live_gate(
        expected_dedupe_key=dedupe_key,
        reader=reader,
        observed_at=observed_at,
    )
    root.mkdir(parents=True, exist_ok=False)
    intent = sealed(
        {
            "schema_version": INTENT_SCHEMA,
            **SAFETY_FLAGS,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan": file_record(resolved_plan),
            "plan_payload_sha256": plan["payload_sha256"],
            "initial_live_gate": initial_gate,
            "task_name": TASK_NAME,
            "dedupe_key": dedupe_key,
            "account_name": ACCOUNT_NAME,
            "node_name": NODE_NAME,
            "same_node_as_task_id": SAME_NODE_AS_TASK_ID,
            "same_node_as_allocation_id": SOURCE_ALLOCATION_ID,
            "parallel_to_task96338": True,
            "authorization": POST_AUTHORIZATION,
            "post_call_budget": 1,
            "attempt_ledger_must_precede_network": True,
            "ambiguous_retry_allowed": False,
            "automatic_full_trigger": False,
        }
    )
    intent_path = write_immutable_json(root / INTENT_NAME, intent)
    with lock_factory():
        if attempt_path.exists():
            raise PostdeadlineContractError(
                "rounded POST attempt was consumed concurrently"
            )
        locked_gate = live_gate(
            expected_dedupe_key=dedupe_key,
            reader=reader,
            observed_at=observed_at,
        )
        attempt = sealed(
            {
                "schema_version": ATTEMPT_SCHEMA,
                **SAFETY_FLAGS,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "plan": file_record(resolved_plan),
                "intent": file_record(intent_path),
                "initial_live_gate": initial_gate,
                "locked_pre_submit_live_gate": locked_gate,
                "attempt_nonce": _attempt_nonce(plan),
                "scheduler_payload_sha256": plan[
                    "scheduler_payload_sha256"
                ],
                "campaign_mutation_lock_acquired": True,
                "post_call_budget": 1,
                "post_call_consumed_before_network": True,
                "scheduler_post_calls_before": 0,
                "scheduler_post_calls_authorized": 1,
                "ambiguous_retry_allowed": False,
            }
        )
        attempt_path = write_exclusive_json(attempt_path, attempt)
        try:
            status, response, error = poster(
                f"{SCHEDULER_URL}/api/tasks", payload
            )
        except Exception as exc:
            status, response, error = None, None, str(exc)
    exact = _inventory_exact(reader, dedupe_key=dedupe_key)
    response_task_id = (
        response.get("task_id") or response.get("id")
        if isinstance(response, Mapping)
        else None
    )
    exact_task_id = (
        exact[0].get("task_id") or exact[0].get("id")
        if len(exact) == 1
        else None
    )
    if (
        len(exact) != 1
        or exact_task_id is None
        or (
            response_task_id is not None
            and int(response_task_id) != int(exact_task_id)
        )
    ):
        _write_ambiguous(
            root=root,
            plan=plan,
            attempt_path=attempt_path,
            http_status=status,
            response=response,
            error=error,
            exact_count=len(exact),
        )
        raise PostdeadlineContractError(
            "rounded POST consumed but exact GET failed; no retry"
        )
    task_id = int(exact_task_id)
    readback = _authenticated_readback(
        reader, task_id=task_id, dedupe_key=dedupe_key
    )
    return _finish(
        root=root,
        plan_path=resolved_plan,
        plan=plan,
        intent_path=intent_path,
        attempt_path=attempt_path,
        task_id=task_id,
        readback=readback,
        http_status=status,
        response=response,
        error=error,
        reconciled=(status != 201 or response_task_id is None),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser(
        "inspect", help="reauthenticate plan and repeat GET admission gates"
    )
    inspect_parser.add_argument(
        "--plan", type=Path, default=OUTPUT_ROOT / PLAN_NAME
    )
    submit_parser = commands.add_parser(
        "submit", help="consume the one-shot Scheduler POST budget"
    )
    submit_parser.add_argument(
        "--plan", type=Path, default=OUTPUT_ROOT / PLAN_NAME
    )
    submit_parser.add_argument("--authorize-post", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "inspect":
        plan = prepare_only.load_plan(args.plan)
        gate = live_gate(
            expected_dedupe_key=plan["scheduler_payload"]["dedupe_key"]
        )
        result = {
            "event": "rounded_final_lane_admissible",
            "plan": str(args.plan.resolve()),
            "gate": gate,
            "scheduler_post_calls": 0,
        }
    else:
        final_path = submit(
            authorize_post=args.authorize_post,
            plan_path=args.plan,
        )
        final = validate_seal(read_json(final_path), FINAL_SCHEMA)
        result = {
            "event": "rounded_final_lane_submitted",
            "final_seal": str(final_path),
            "task_id": final["task_id"],
            "scheduler_post_calls": final[
                "scheduler_post_calls_evidenced"
            ],
        }
    print(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PostdeadlineContractError as exc:
        print(
            json.dumps(
                {
                    "event": "rounded_final_lane_submit_error",
                    "error": str(exc),
                    "automatic_retry": False,
                    "automatic_full_trigger": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
