"""One-shot strict-n107 hedge for the rounded candidate-5 symmetric model.

This module is deliberately bound to one placement and one immutable physics
identity.  It re-derives the exact task96340 rounded B5 payload while changing
only the Scheduler task/retention namespace, account, target node, and
``max_workers_per_node``:

* account ``harry261`` (fresh allocation capacity)
* node ``n107`` with strict placement
* no same-node or requested-allocation attachment
* ``max_workers_per_node = 1``

Preparation and collection are GET-only.  Submission consumes its single POST
attempt before the network call and never retries an ambiguous response.
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from tools import (  # noqa: E402
    mft_goal_official5_rounded_final_prepare as rounded,
)
from tools import (  # noqa: E402
    mft_goal_official5_rounded_final_submit as rounded_submit,
)
from tools import (  # noqa: E402
    mft_goal_official_standard_postdeadline as transaction,
)


PostdeadlineContractError = rounded.PostdeadlineContractError
JsonReader = Callable[[str, Sequence[tuple[str, Any]] | None], Any]
Poster = Callable[
    [str, Mapping[str, Any]],
    tuple[int | None, dict[str, Any] | None, str | None],
]

SCHEDULER_URL = "http://127.0.0.1:8002"
PROJECT = "MFT_1MW_2026v1"
ACCOUNT_NAME = "harry261"
NODE_NAME = "n107"
CPUS = 8
MEMORY_MB = 98_304
SOLVER_SECONDS = 3 * 60 * 60
KILL_GRACE_SECONDS = 300
RETENTION_SECONDS = 1_800
SCHEDULER_SECONDS = (
    SOLVER_SECONDS + KILL_GRACE_SECONDS + RETENTION_SECONDS
)
MAX_WORKERS_PER_NODE = 1
PRIORITY = 100

CANDIDATE_SHA256 = rounded.SOURCE_CANDIDATE_SHA256
SOLVER_REVISION = "623a5345ae1b85b974cb248bcb0c8dfaadf1a867"
LIBRARY_REVISION = rounded.LIBRARY_REVISION
SOURCE_TASK_ID = 96340
SOURCE_TASK_NAME = (
    "mft-goal-final-standard-official5-rounded-r10-s4-v3-"
    "909d249ebe45-n113"
)
SOURCE_TASK_DEDUPE_KEY = (
    "mft-al:mft-goal-final-standard-official5-rounded-r10-s4-v3-"
    "909d249ebe45-n113:"
    "623a5345ae1b85b974cb248bcb0c8dfaadf1a867:"
    "e6b9b9d20a832ff5c3f7ca97218737a0b8650781:"
    "0fe5177192dce5df"
)
SOURCE_PLAN_PATH = (
    rounded.OUTPUT_ROOT / rounded.PLAN_NAME
)
SOURCE_RECEIPT_PATH = (
    rounded.OUTPUT_ROOT
    / rounded_submit.SUBMISSION_DIRECTORY_NAME
    / rounded_submit.RECEIPT_NAME
)

TASK_NAME = (
    "mft-goal-final-standard-official5-rounded-r10-s4-hedge1-v1-"
    f"{CANDIDATE_SHA256[:12]}-{NODE_NAME}"
)
WORKDIR = (
    "mft_goal_final_standard_official5_rounded_r10_s4_hedge1_v1_"
    f"{CANDIDATE_SHA256[:12]}_{NODE_NAME}"
)
OUTPUT_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\final_standard_official5_rounded_r10_s4_hedge_n107_v1"
)
PLAN_NAME = "rounded_n107_hedge_plan.json"
PREPARE_RECEIPT_NAME = "prepare_receipt.json"
ATTEMPT_LEDGER_NAME = "scheduler_post_attempt.json"
SUBMISSION_DIRECTORY_NAME = "submission"
COLLECTION_DIRECTORY_NAME = "authenticated_get_collection"
POST_AUTHORIZATION = "authorize-rounded-b5-symmetric-n107-hedge-one-post-v1"

PLAN_SCHEMA = "mft-goal-rounded-b5-n107-hedge-plan-v1"
PREPARE_SCHEMA = "mft-goal-rounded-b5-n107-hedge-prepare-v1"
PREFLIGHT_SCHEMA = "mft-goal-rounded-b5-n107-hedge-live-preflight-v1"
INTENT_SCHEMA = "mft-goal-rounded-b5-n107-hedge-submit-intent-v1"
SUBMISSION_SCHEMA = "mft-goal-rounded-b5-n107-hedge-submission-v1"
FINAL_SEAL_SCHEMA = "mft-goal-rounded-b5-n107-hedge-final-seal-v1"
COLLECTION_SCHEMA = "mft-goal-rounded-b5-n107-hedge-collection-v1"
ATTEMPT_NONCE_SCHEMA = "mft-goal-rounded-b5-n107-hedge-attempt-nonce-v1"

OPENING_QUEUE_REASON = (
    "no single ready pool has 8 free CPUs; opening demand pools"
)
ACTIVE_TASK_STATES = {"queued", "attaching", "running"}
LIVE_ALLOCATION_STATES = {
    "active",
    "draining",
    "pending",
    "submitted",
    "opening",
}
SAFETY_FLAGS = {
    "diagnostic_only": True,
    "search_only": True,
    "canonical": False,
    "noncanonical": True,
    "production_eligible": False,
    "automatic_promotion": False,
    "scientific_pass_claimed": False,
    "automatic_full_trigger": False,
}
FIXED_BOUNDARY = copy.deepcopy(rounded.FIXED_BOUNDARY)
ROUNDING_POLICY = copy.deepcopy(rounded.ROUNDING_POLICY)

canonical_bytes = transaction.canonical_bytes
payload_sha256 = transaction.payload_sha256
sealed = transaction.sealed
validate_seal = transaction.validate_seal
sha256_file = transaction.sha256_file
file_record = transaction.file_record
read_json = transaction.read_json
write_immutable_json = transaction.write_immutable_json
write_exclusive_json = transaction.write_exclusive_json
get_json = transaction.get_json
scheduler_campaign_lock = transaction.scheduler_campaign_lock


def _relative_record(root: Path, path: Path) -> dict[str, Any]:
    root = root.resolve()
    path = path.resolve(strict=True)
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise PostdeadlineContractError(
            "hedge artifact escaped its immutable root"
        ) from exc
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _contained(root: Path, value: Any, label: str) -> Path:
    if (
        not isinstance(value, Mapping)
        or not isinstance(value.get("path"), str)
        or not isinstance(value.get("sha256"), str)
        or type(value.get("size_bytes")) is not int
    ):
        raise PostdeadlineContractError(f"{label} record is malformed")
    root = root.resolve(strict=True)
    path = (root / str(value["path"])).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PostdeadlineContractError(
            f"{label} escaped the immutable root"
        ) from exc
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != value["size_bytes"]
        or sha256_file(path) != value["sha256"]
    ):
        raise PostdeadlineContractError(f"{label} bytes drifted")
    return path


def _task_rows(value: Any) -> list[dict[str, Any]]:
    return transaction._task_rows(value)


def _account_rows(value: Any) -> list[dict[str, Any]]:
    return rounded.direct._account_rows(value)


def _capability_accounts(value: Any) -> set[str]:
    return rounded.direct._capability_accounts(value)


@contextmanager
def _rounded_payload_patch() -> Iterator[None]:
    """Bind the reviewed rounded payload builder to the n107 hedge."""

    replacements = {
        "SCHEDULER_URL": SCHEDULER_URL,
        "PROJECT": PROJECT,
        "ACCOUNT_NAME": ACCOUNT_NAME,
        "NODE_NAME": NODE_NAME,
        "TASK_NAME": TASK_NAME,
        "WORKDIR": WORKDIR,
        "CPUS": CPUS,
        "MEMORY_MB": MEMORY_MB,
        "SOLVER_SECONDS": SOLVER_SECONDS,
        "KILL_GRACE_SECONDS": KILL_GRACE_SECONDS,
        "RETENTION_SECONDS": RETENTION_SECONDS,
        "SCHEDULER_SECONDS": SCHEDULER_SECONDS,
        "MAX_WORKERS_PER_NODE": MAX_WORKERS_PER_NODE,
        "PRIORITY": PRIORITY,
        "SOURCE_LINEAGE_TASK_ID": SOURCE_TASK_ID,
        "SAME_NODE_AS_TASK_ID": 0,
        "SOURCE_ALLOCATION_ID": 0,
        "SOURCE_SLURM_JOB_ID": "",
    }
    previous = {name: getattr(rounded, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(rounded, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(rounded, name, value)


@contextmanager
def _transaction_patch() -> Iterator[None]:
    """Bind the reviewed consume-before-network transaction."""

    replacements = {
        "SCHEDULER_URL": SCHEDULER_URL,
        "PROJECT": PROJECT,
        "ACCOUNT_NAME": ACCOUNT_NAME,
        "NODE_NAME": NODE_NAME,
        "TASK_NAME": TASK_NAME,
        "WORKDIR": WORKDIR,
        "CPUS": CPUS,
        "MEMORY_MB": MEMORY_MB,
        "SOLVER_SECONDS": SOLVER_SECONDS,
        "KILL_GRACE_SECONDS": KILL_GRACE_SECONDS,
        "RETENTION_SECONDS": RETENTION_SECONDS,
        "SCHEDULER_SECONDS": SCHEDULER_SECONDS,
        "MAX_WORKERS_PER_NODE": MAX_WORKERS_PER_NODE,
        "PRIORITY": PRIORITY,
        "POST_AUTHORIZATION": POST_AUTHORIZATION,
        "ATTEMPT_LEDGER_NAME": ATTEMPT_LEDGER_NAME,
        "SUBMISSION_DIRECTORY_NAME": SUBMISSION_DIRECTORY_NAME,
        "PLAN_SCHEMA": PLAN_SCHEMA,
        "CANDIDATE_SCHEMA": (
            rounded.direct.source_prepare.CANDIDATE_SCHEMA
        ),
        "INTENT_SCHEMA": INTENT_SCHEMA,
        "SUBMISSION_SCHEMA": SUBMISSION_SCHEMA,
        "FINAL_SEAL_SCHEMA": FINAL_SEAL_SCHEMA,
        "SAFETY_FLAGS": SAFETY_FLAGS,
        "live_preflight": live_preflight,
    }
    previous = {name: getattr(transaction, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(transaction, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(transaction, name, value)


def _source_receipt() -> dict[str, Any]:
    receipt = validate_seal(
        read_json(SOURCE_RECEIPT_PATH),
        rounded_submit.RECEIPT_SCHEMA,
    )
    expected = {
        "task_id": SOURCE_TASK_ID,
        "task_name": SOURCE_TASK_NAME,
        "dedupe_key": SOURCE_TASK_DEDUPE_KEY,
        "candidate_physics_sha256": CANDIDATE_SHA256,
        "solver_revision": SOLVER_REVISION,
        "node_name": "n113",
        "same_node_as_task_id": 0,
    }
    drift = {
        key: {"expected": value, "actual": receipt.get(key)}
        for key, value in expected.items()
        if receipt.get(key) != value
    }
    if drift:
        raise PostdeadlineContractError(
            f"task96340 source receipt drifted: {drift}"
        )
    return receipt


def _derive() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Reauthenticate candidate 5 and derive the exact rounded hedge."""

    source_plan = rounded.load_plan(SOURCE_PLAN_PATH)
    source_receipt = _source_receipt()
    if (
        source_plan.get("source_candidate_physics_sha256")
        != CANDIDATE_SHA256
        or source_plan.get("solver_revision") != SOLVER_REVISION
        or source_plan.get("rounding_policy") != ROUNDING_POLICY
        or source_plan.get("fixed_boundary") != FIXED_BOUNDARY
    ):
        raise PostdeadlineContractError(
            "task96340 rounded source plan identity drifted"
        )
    source = rounded.direct.authenticate_source_authority()
    authority = source.get("authority")
    selected = source.get("selected")
    if (
        not isinstance(authority, dict)
        or not isinstance(selected, dict)
        or authority.get("candidate_physics_sha256")
        != CANDIDATE_SHA256
        or authority.get("official_standard_selection_order") != 5
    ):
        raise PostdeadlineContractError(
            "candidate-5 source authority is incomplete"
        )
    params, profile = rounded.derive_rounded_params_and_profile(source)
    attestation = rounded.attest_rounded_solver_revision(SOLVER_REVISION)
    validate_seal(attestation, rounded.REVISION_SCHEMA)
    with _rounded_payload_patch():
        payload, environment, retained = rounded.derive_scheduler_payload(
            params, profile, SOLVER_REVISION
        )
        rounded.validate_rounded_payload(
            payload,
            environment,
            retained,
            params=params,
            profile=profile,
            solver_revision=SOLVER_REVISION,
        )
    if (
        "same_node_as_task_id" in payload
        or "requested_allocation_id" in payload
        or payload.get("account_name") != ACCOUNT_NAME
        or payload.get("node_name") != NODE_NAME
        or payload.get("node_name_policy") != "strict"
        or payload.get("max_workers_per_node") != 1
    ):
        raise PostdeadlineContractError(
            "n107 hedge payload is not a fresh strict allocation request"
        )
    return (
        copy.deepcopy(authority),
        copy.deepcopy(selected),
        params,
        profile,
        payload,
        {
            "environment": environment,
            "retained": retained,
            "attestation": attestation,
            "source_plan": source_plan,
            "source_receipt": source_receipt,
        },
    )


def capacity_query() -> list[tuple[str, Any]]:
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
        ("partition", "auto"),
        ("node_name", NODE_NAME),
        ("node_name_policy", "strict"),
    ]


def _get_text(
    path: str,
    query: Sequence[tuple[str, Any]] | None = None,
    *,
    timeout: float = 30.0,
) -> str:
    encoded = urlencode(list(query or []), doseq=True)
    url = f"{SCHEDULER_URL}{path}"
    if encoded:
        url = f"{url}?{encoded}"
    request = Request(url, method="GET")
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _node_snapshot() -> dict[str, Any]:
    raw = _get_text(f"/nodes/{NODE_NAME}")
    without_script = re.sub(
        r"<script[\s\S]*?</script>", " ", raw, flags=re.IGNORECASE
    )
    without_style = re.sub(
        r"<style[\s\S]*?</style>", " ", without_script, flags=re.IGNORECASE
    )
    text = html.unescape(re.sub(r"<[^>]+>", " ", without_style))
    text = re.sub(r"\s+", " ", text).strip()
    pattern = re.compile(
        r"Load \(원시 / 0~100% 환산\) "
        r"(?P<load>[0-9.]+) / (?P<load_pct>[0-9.]+)% "
        r"여유 메모리 (?P<free_gb>[0-9]+) GB "
        r"\((?P<free_pct>[0-9.]+)%\) "
        r"상태 / 관측 시각 (?P<state>[a-z]+) "
        r"(?P<observed>[0-9-]+ [0-9:]+ UTC)"
    )
    match = pattern.search(text)
    task_matches = list(
        re.finditer(
            r"이 노드에서 실행 중인 태스크 \((?P<count>[0-9]+)\)",
            text,
        )
    )
    if match is None or not task_matches:
        raise PostdeadlineContractError(
            "n107 node dashboard cannot be parsed"
        )
    observed = datetime.strptime(
        match.group("observed"), "%Y-%m-%d %H:%M:%S UTC"
    ).replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    age = (now - observed).total_seconds()
    value = {
        "node_name": NODE_NAME,
        "state": match.group("state"),
        "observed_at_utc": observed.isoformat(),
        "snapshot_age_seconds": age,
        "load": float(match.group("load")),
        "load_percent": float(match.group("load_pct")),
        "free_memory_gb": int(match.group("free_gb")),
        "free_memory_percent": float(match.group("free_pct")),
        "scheduler_fea_task_count": int(
            task_matches[-1].group("count")
        ),
        "dashboard_sha256": hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest(),
    }
    if (
        value["state"] != "idle"
        or not -30 <= age <= 180
        or value["load_percent"] > 5.0
        or value["free_memory_gb"] < 512
        or value["free_memory_percent"] < 60.0
        or value["scheduler_fea_task_count"] != 0
    ):
        raise PostdeadlineContractError(
            f"n107 is no longer a fresh FEA-empty node: {value}"
        )
    return value


def _freshness_seconds(value: Any, label: str) -> float:
    if not isinstance(value, str):
        raise PostdeadlineContractError(f"{label} timestamp is absent")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise PostdeadlineContractError(
            f"{label} timestamp lacks timezone"
        )
    return (
        datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)
    ).total_seconds()


def live_preflight(
    *,
    reader: JsonReader = get_json,
    expected_dedupe_key: str,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    """Require fresh license, account, empty-node, and opening readbacks."""

    now = observed_at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise PostdeadlineContractError(
            "hedge preflight time must be timezone-aware"
        )
    now = now.astimezone(timezone.utc)
    health = reader("/api/health", None)
    source_task = reader(f"/api/tasks/{SOURCE_TASK_ID}", None)
    capacity = reader("/api/task-capacity", capacity_query())
    licenses = reader("/api/licenses", None)
    accounts = reader("/api/accounts/status/live", None)
    capabilities = reader("/api/capabilities", None)
    allocations = reader("/api/allocations", [("limit", 10000)])
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
    node = _node_snapshot()
    if (
        not isinstance(health, Mapping)
        or health.get("ok") is not True
        or health.get("scheduler_ok") is not True
        or health.get("scheduler_thread_alive") is not True
        or health.get("scheduler_stalled") is not False
        or not -30
        <= _freshness_seconds(
            health.get("last_tick_completed_at"), "scheduler health"
        )
        <= 120
    ):
        raise PostdeadlineContractError("Scheduler health gate failed")
    expected_source = {
        "task_id": SOURCE_TASK_ID,
        "name": SOURCE_TASK_NAME,
        "dedupe_key": SOURCE_TASK_DEDUPE_KEY,
        "project": PROJECT,
        "status": "running",
        "requested_account_name": "dw16",
        "requested_node_name": "n113",
        "requested_node_name_policy": "strict",
        "allocation_id": 14620,
        "slurm_job_id": "829579",
    }
    source_drift = {
        key: {"expected": value, "actual": source_task.get(key)}
        for key, value in expected_source.items()
        if not isinstance(source_task, Mapping)
        or source_task.get(key) != value
    }
    if source_drift:
        raise PostdeadlineContractError(
            f"task96340 no longer requires a hedge: {source_drift}"
        )
    if (
        not isinstance(capacity, Mapping)
        or capacity.get("queue_state") != "opening"
        or capacity.get("queue_reason") != OPENING_QUEUE_REASON
        or int(capacity.get("fit_slots") or 0) != 0
        or int(capacity.get("ready_fit_slots") or 0) != 0
        or int(capacity.get("pending_fit_slots") or 0) != 0
        or int(capacity.get("inflight_fit_slots") or 0) != 0
        or capacity.get("memory_pressure_state") != "ok"
        or capacity.get("preferred_node_relaxed") is not False
        or capacity.get("allocations") != []
        or int(capacity.get("standalone_aedt_available") or 0) < 1
    ):
        raise PostdeadlineContractError(
            "strict harry261/n107 opening capacity gate failed"
        )
    license_gate = rounded.direct.source_submit._license_gate(
        licenses, required_headroom=1
    )
    if (
        not isinstance(licenses, Mapping)
        or licenses.get("server_up") is not True
        or not -30
        <= _freshness_seconds(
            licenses.get("checked_at"), "license snapshot"
        )
        <= 120
    ):
        raise PostdeadlineContractError("fresh license gate failed")
    account_matches = [
        row
        for row in _account_rows(accounts)
        if row.get("account_name") == ACCOUNT_NAME
    ]
    if len(account_matches) != 1:
        raise PostdeadlineContractError(
            "harry261 live account status is absent or ambiguous"
        )
    account = account_matches[0]
    account_values = {
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
        account_values["running"] >= account_values["max_running"]
        or account_values["pending"] >= account_values["max_pending"]
        or account_values["running"] + account_values["pending"]
        >= account_values["max_total"]
        or ACCOUNT_NAME not in _capability_accounts(capabilities)
    ):
        raise PostdeadlineContractError(
            f"harry261 new-allocation gate failed: {account_values}"
        )
    active_rows = _task_rows(active)
    target_fea = [
        row
        for row in active_rows
        if row.get("status") in ACTIVE_TASK_STATES
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
    live_allocations = [
        row
        for row in _task_rows(allocations)
        if row.get("node_name") == NODE_NAME
        and row.get("state") in LIVE_ALLOCATION_STATES
    ]
    if target_fea or live_allocations:
        raise PostdeadlineContractError(
            "n107 is not a new-allocation-only FEA-empty target"
        )
    collisions = [
        row
        for row in _task_rows(inventory)
        if row.get("name") == TASK_NAME
        or row.get("dedupe_key") == expected_dedupe_key
    ]
    if collisions:
        raise PostdeadlineContractError(
            "rounded n107 hedge name or dedupe already exists"
        )
    return sealed(
        {
            "schema_version": PREFLIGHT_SCHEMA,
            **SAFETY_FLAGS,
            "observed_at_utc": now.isoformat(),
            "scheduler_health": copy.deepcopy(dict(health)),
            "source_task96340": copy.deepcopy(dict(source_task)),
            "capacity_query": capacity_query(),
            "capacity": copy.deepcopy(dict(capacity)),
            "license_gate": license_gate,
            "license_checked_at": licenses["checked_at"],
            "license_admission": copy.deepcopy(
                dict(licenses.get("admission") or {})
            ),
            "account_gate": {
                **account_values,
                "new_slurm_allocation_required": True,
                "existing_allocation_attach_forbidden": True,
            },
            "node_snapshot": node,
            "active_target_fea_count": 0,
            "live_target_allocation_count": 0,
            "collision_count": 0,
            "strict_node_required": True,
            "preferred_node_relaxed": False,
            "max_workers_per_node": 1,
            "scheduler_get_only": True,
            "scheduler_post_calls": 0,
        }
    )


def _attempt_nonce(payload: Mapping[str, Any]) -> str:
    return payload_sha256(
        {
            "schema_version": ATTEMPT_NONCE_SCHEMA,
            "candidate_physics_sha256": CANDIDATE_SHA256,
            "task_name": TASK_NAME,
            "dedupe_key": payload["dedupe_key"],
            "account_name": ACCOUNT_NAME,
            "node_name": NODE_NAME,
            "node_name_policy": "strict",
            "max_workers_per_node": 1,
            "scheduler_payload_sha256": payload_sha256(payload),
        }
    )


def prepare(
    *,
    output: Path | None = None,
    reader: JsonReader = get_json,
    observed_at: datetime | None = None,
) -> Path:
    """Seal one GET-only plan for the exact n107 hedge."""

    target = (output or OUTPUT_ROOT).resolve()
    if target != OUTPUT_ROOT.resolve():
        raise PostdeadlineContractError("hedge output root is not fixed")
    if target.exists():
        raise PostdeadlineContractError(
            f"immutable hedge root already exists: {target}"
        )
    (
        authority,
        selected,
        params,
        profile,
        payload,
        derived,
    ) = _derive()
    environment = derived["environment"]
    retained = derived["retained"]
    attestation = derived["attestation"]
    preflight = live_preflight(
        reader=reader,
        expected_dedupe_key=str(payload["dedupe_key"]),
        observed_at=observed_at,
    )
    now = observed_at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise PostdeadlineContractError(
            "hedge prepare time must be timezone-aware"
        )
    attempt_path = target / ATTEMPT_LEDGER_NAME
    submission_path = target / SUBMISSION_DIRECTORY_NAME
    staging = target.with_name(
        f".{target.name}.{os.getpid()}."
        f"{next(tempfile._get_candidate_names())}.tmp"
    )
    staging.mkdir(parents=True)
    try:
        authority_path = write_immutable_json(
            staging / "source_authority.json", authority
        )
        selected_path = write_immutable_json(
            staging / "source_selected_candidate.json", selected
        )
        params_path = write_immutable_json(
            staging / "rounded_fea_params.json", params
        )
        profile_path = write_immutable_json(
            staging / "rounded_execution_profile.json", profile
        )
        attestation_path = write_immutable_json(
            staging / "solver_attestation.json", attestation
        )
        effective = copy.deepcopy(params)
        effective.update(profile["param_overrides"])
        physics_contract = {
            "candidate_physics_sha256": CANDIDATE_SHA256,
            "rounded_fea_params_sha256": payload_sha256(params),
            "effective_rounded_params_sha256": payload_sha256(effective),
            "rounding_policy": ROUNDING_POLICY,
            "fixed_boundary": FIXED_BOUNDARY,
            "full_model": effective["full_model"],
            "thermal_symmetry": effective["thermal_symmetry"],
            "turns": {
                "N1_main": effective["N1_main"],
                "N1_side": effective["N1_side"],
                "N2_main": effective["N2_main"],
                "N2_side": effective["N2_side"],
            },
        }
        plan = sealed(
            {
                "schema_version": PLAN_SCHEMA,
                **SAFETY_FLAGS,
                "created_at_utc": now.astimezone(
                    timezone.utc
                ).isoformat(),
                "campaign_id": rounded.CAMPAIGN_ID,
                "task_name": TASK_NAME,
                "workdir": WORKDIR,
                "dedupe_key": payload["dedupe_key"],
                "source_task_id": SOURCE_TASK_ID,
                "source_task_name": SOURCE_TASK_NAME,
                "source_task_dedupe_key": SOURCE_TASK_DEDUPE_KEY,
                "source_plan": file_record(SOURCE_PLAN_PATH),
                "source_submission_receipt": file_record(
                    SOURCE_RECEIPT_PATH
                ),
                "source_authority": _relative_record(
                    staging, authority_path
                ),
                "source_selected_candidate": _relative_record(
                    staging, selected_path
                ),
                "rounded_fea_params": _relative_record(
                    staging, params_path
                ),
                "rounded_execution_profile": _relative_record(
                    staging, profile_path
                ),
                "solver_attestation": _relative_record(
                    staging, attestation_path
                ),
                "solver_revision": SOLVER_REVISION,
                "library_revision": LIBRARY_REVISION,
                "physics_contract": physics_contract,
                "physics_contract_sha256": payload_sha256(
                    physics_contract
                ),
                "placement": {
                    "account_name": ACCOUNT_NAME,
                    "node_name": NODE_NAME,
                    "node_name_policy": "strict",
                    "same_node_as_task_id": 0,
                    "requested_allocation_id": 0,
                    "new_slurm_allocation_required": True,
                    "existing_allocation_attach_forbidden": True,
                    "max_workers_per_node": 1,
                    "preferred_node_relaxed_allowed": False,
                },
                "resources": {
                    "cpus": CPUS,
                    "memory_mb": MEMORY_MB,
                    "solver_seconds": SOLVER_SECONDS,
                    "kill_grace_seconds": KILL_GRACE_SECONDS,
                    "retention_seconds": RETENTION_SECONDS,
                    "scheduler_timeout_seconds": SCHEDULER_SECONDS,
                    "max_workers_per_node": 1,
                },
                "scheduler_payload": payload,
                "scheduler_payload_sha256": payload_sha256(payload),
                "submission_environment": environment,
                "submission_environment_sha256": payload_sha256(
                    environment
                ),
                "retained_aedt_bundle": retained,
                "retained_aedt_bundle_sha256": payload_sha256(retained),
                "initial_live_preflight": preflight,
                "scheduler_url": SCHEDULER_URL,
                "scheduler_project": PROJECT,
                "scheduler_post_calls": 0,
                "scheduler_submission_performed": False,
                "single_attempt_contract": {
                    "schema_version": (
                        "mft-goal-rounded-b5-n107-hedge-single-attempt-v1"
                    ),
                    "attempt_ledger_path": str(attempt_path),
                    "submission_output_path": str(submission_path),
                    "attempt_nonce": _attempt_nonce(payload),
                    "post_call_budget": 1,
                    "ledger_consumed_before_network": True,
                    "ambiguous_retry_allowed": False,
                },
            }
        )
        plan_path = write_immutable_json(staging / PLAN_NAME, plan)
        receipt = sealed(
            {
                "schema_version": PREPARE_SCHEMA,
                **SAFETY_FLAGS,
                "created_at_utc": now.astimezone(
                    timezone.utc
                ).isoformat(),
                "plan": _relative_record(staging, plan_path),
                "plan_payload_sha256": plan["payload_sha256"],
                "candidate_physics_sha256": CANDIDATE_SHA256,
                "physics_contract_sha256": plan[
                    "physics_contract_sha256"
                ],
                "scheduler_payload_sha256": plan[
                    "scheduler_payload_sha256"
                ],
                "task_name": TASK_NAME,
                "dedupe_key": payload["dedupe_key"],
                "account_name": ACCOUNT_NAME,
                "node_name": NODE_NAME,
                "node_name_policy": "strict",
                "new_slurm_allocation_required": True,
                "max_workers_per_node": 1,
                "queue_state_at_prepare": "opening",
                "scheduler_get_preflight_performed": True,
                "scheduler_post_calls": 0,
                "ready_for_explicit_submit": True,
                "submit_authorization_token": POST_AUTHORIZATION,
            }
        )
        write_immutable_json(
            staging / PREPARE_RECEIPT_NAME, receipt
        )
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target / PLAN_NAME


def load_plan(
    plan_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Reauthenticate every immutable physics and transaction input."""

    resolved = plan_path.resolve(strict=True)
    expected = OUTPUT_ROOT.resolve() / PLAN_NAME
    if resolved != expected:
        raise PostdeadlineContractError(
            "hedge plan path escaped the fixed root"
        )
    plan = validate_seal(read_json(resolved), PLAN_SCHEMA)
    root = resolved.parent
    authority_path = _contained(
        root, plan.get("source_authority"), "source authority"
    )
    selected_path = _contained(
        root,
        plan.get("source_selected_candidate"),
        "source selected candidate",
    )
    params_path = _contained(
        root, plan.get("rounded_fea_params"), "rounded params"
    )
    profile_path = _contained(
        root,
        plan.get("rounded_execution_profile"),
        "rounded profile",
    )
    attestation_path = _contained(
        root, plan.get("solver_attestation"), "solver attestation"
    )
    (
        authority,
        selected,
        params,
        profile,
        payload,
        derived,
    ) = _derive()
    environment = derived["environment"]
    retained = derived["retained"]
    attestation = derived["attestation"]
    effective = copy.deepcopy(params)
    effective.update(profile["param_overrides"])
    physics_contract = {
        "candidate_physics_sha256": CANDIDATE_SHA256,
        "rounded_fea_params_sha256": payload_sha256(params),
        "effective_rounded_params_sha256": payload_sha256(effective),
        "rounding_policy": ROUNDING_POLICY,
        "fixed_boundary": FIXED_BOUNDARY,
        "full_model": effective["full_model"],
        "thermal_symmetry": effective["thermal_symmetry"],
        "turns": {
            "N1_main": effective["N1_main"],
            "N1_side": effective["N1_side"],
            "N2_main": effective["N2_main"],
            "N2_side": effective["N2_side"],
        },
    }
    single = plan.get("single_attempt_contract")
    placement = plan.get("placement")
    if (
        read_json(authority_path) != authority
        or read_json(selected_path) != selected
        or read_json(params_path) != params
        or read_json(profile_path) != profile
        or read_json(attestation_path) != attestation
        or plan.get("source_plan") != file_record(SOURCE_PLAN_PATH)
        or plan.get("source_submission_receipt")
        != file_record(SOURCE_RECEIPT_PATH)
        or plan.get("source_task_id") != SOURCE_TASK_ID
        or plan.get("source_task_name") != SOURCE_TASK_NAME
        or plan.get("source_task_dedupe_key") != SOURCE_TASK_DEDUPE_KEY
        or plan.get("solver_revision") != SOLVER_REVISION
        or plan.get("library_revision") != LIBRARY_REVISION
        or plan.get("physics_contract") != physics_contract
        or plan.get("physics_contract_sha256")
        != payload_sha256(physics_contract)
        or plan.get("scheduler_payload") != payload
        or plan.get("scheduler_payload_sha256")
        != payload_sha256(payload)
        or plan.get("submission_environment") != environment
        or plan.get("submission_environment_sha256")
        != payload_sha256(environment)
        or plan.get("retained_aedt_bundle") != retained
        or plan.get("retained_aedt_bundle_sha256")
        != payload_sha256(retained)
        or plan.get("scheduler_post_calls") != 0
        or plan.get("scheduler_submission_performed") is not False
        or not isinstance(placement, Mapping)
        or placement.get("account_name") != ACCOUNT_NAME
        or placement.get("node_name") != NODE_NAME
        or placement.get("node_name_policy") != "strict"
        or placement.get("same_node_as_task_id") != 0
        or placement.get("requested_allocation_id") != 0
        or placement.get("new_slurm_allocation_required") is not True
        or placement.get("existing_allocation_attach_forbidden") is not True
        or placement.get("max_workers_per_node") != 1
        or not isinstance(single, Mapping)
        or Path(str(single.get("attempt_ledger_path") or "")).resolve()
        != OUTPUT_ROOT.resolve() / ATTEMPT_LEDGER_NAME
        or Path(
            str(single.get("submission_output_path") or "")
        ).resolve()
        != OUTPUT_ROOT.resolve() / SUBMISSION_DIRECTORY_NAME
        or single.get("attempt_nonce") != _attempt_nonce(payload)
        or single.get("post_call_budget") != 1
        or single.get("ledger_consumed_before_network") is not True
        or single.get("ambiguous_retry_allowed") is not False
    ):
        raise PostdeadlineContractError(
            "sealed rounded n107 hedge plan drifted"
        )
    return plan, params, profile


def inspect(plan_path: Path | None = None) -> dict[str, Any]:
    resolved = plan_path or (OUTPUT_ROOT / PLAN_NAME)
    plan, _params, _profile = load_plan(resolved)
    preflight = live_preflight(
        expected_dedupe_key=plan["dedupe_key"]
    )
    return {
        "event": "rounded_n107_hedge_admissible",
        "plan": str(resolved.resolve()),
        "plan_payload_sha256": plan["payload_sha256"],
        "candidate_physics_sha256": CANDIDATE_SHA256,
        "physics_contract_sha256": plan["physics_contract_sha256"],
        "scheduler_payload_sha256": plan[
            "scheduler_payload_sha256"
        ],
        "queue_state": preflight["capacity"]["queue_state"],
        "node_snapshot": preflight["node_snapshot"],
        "account_gate": preflight["account_gate"],
        "license_gate": preflight["license_gate"],
        "scheduler_post_calls": 0,
    }


def submit(
    *,
    authorize_post: str,
    plan_path: Path | None = None,
    reader: JsonReader = get_json,
    poster: Poster = transaction._post_json_once,
    observed_at: datetime | None = None,
    lock_factory: Callable[[], Any] = scheduler_campaign_lock,
) -> Path:
    """Consume exactly zero or one POST and never retry ambiguity."""

    if authorize_post != POST_AUTHORIZATION:
        raise PostdeadlineContractError(
            "explicit rounded n107 hedge authorization is absent"
        )
    resolved_plan = (
        plan_path or (OUTPUT_ROOT / PLAN_NAME)
    ).resolve(strict=True)
    if resolved_plan != OUTPUT_ROOT.resolve() / PLAN_NAME:
        raise PostdeadlineContractError(
            "hedge submit plan path is not fixed"
        )
    loaded = load_plan(resolved_plan)

    def loaded_plan(
        requested: Path,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        if requested.resolve(strict=True) != resolved_plan:
            raise PostdeadlineContractError(
                "transaction requested a different hedge plan"
            )
        return copy.deepcopy(loaded)

    target = OUTPUT_ROOT.resolve() / SUBMISSION_DIRECTORY_NAME
    with _transaction_patch():
        result = transaction.submit(
            plan_path=resolved_plan,
            output=target,
            authorize_post=authorize_post,
            reader=reader,
            poster=poster,
            observed_at=observed_at,
            lock_factory=lock_factory,
            plan_loader=loaded_plan,
        )
    final = validate_seal(read_json(result), FINAL_SEAL_SCHEMA)
    if int(final.get("scheduler_post_calls") or 0) != 1:
        raise PostdeadlineContractError(
            "hedge final seal does not evidence one POST"
        )
    return result


def _load_submission() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any]
]:
    plan, _params, _profile = load_plan(OUTPUT_ROOT / PLAN_NAME)
    final_path = (
        OUTPUT_ROOT
        / SUBMISSION_DIRECTORY_NAME
        / "final_seal.json"
    )
    final = validate_seal(read_json(final_path), FINAL_SEAL_SCHEMA)
    receipt_path = (
        OUTPUT_ROOT
        / SUBMISSION_DIRECTORY_NAME
        / "submission_receipt.json"
    )
    receipt = validate_seal(read_json(receipt_path), SUBMISSION_SCHEMA)
    if (
        final.get("task_id") != receipt.get("task_id")
        or receipt.get("plan_payload_sha256") != plan["payload_sha256"]
        or receipt.get("scheduler_payload_sha256")
        != plan["scheduler_payload_sha256"]
        or receipt.get("scheduler_post_calls") != 1
    ):
        raise PostdeadlineContractError(
            "hedge submission evidence drifted"
        )
    return plan, final, receipt


def _task_script_attestation(
    task_id: int,
    task: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    if task.get("status") not in {"attaching", "running"}:
        return {
            "available": False,
            "reason": "task has not attached",
            "exact_scheduler_command_present": False,
        }
    try:
        script = _get_text(
            f"/api/tasks/{task_id}/remote-file",
            [
                ("base", "remote_dir"),
                ("path", "task.sh"),
                ("max_bytes", 1_048_576),
            ],
        )
    except Exception as exc:
        return {
            "available": False,
            "reason": str(exc),
            "exact_scheduler_command_present": False,
        }
    command = str(plan["scheduler_payload"]["command"])
    markers = {
        "solver_revision": SOLVER_REVISION,
        "candidate_physics_prefix": CANDIDATE_SHA256[:12],
        # Scheduler stores the command verbatim inside task.sh.  The compact
        # JSON handed to printf therefore has ordinary quotes, not the
        # backslash-escaped representation used by Python/JSON serializers.
        "round_corner": '"round_corner":1',
        "corner_radius": '"corner_radius":10.0',
        "corner_segments": '"corner_segments":4',
        "full_model": '"full_model":0',
        "thermal_symmetry": '"thermal_symmetry":"eighth"',
        "fan_velocity": '"fan_velocity":1.5',
        "tim": '"k_ins":0.2',
        "core_pad": '"core_plate_pad_t":2.0',
        "wcp_pad": '"wcp_pad_t":2.0',
        "direct_analyze": "--symmetry-thermal-direct-analyze",
    }
    marker_presence = {
        key: value in script for key, value in markers.items()
    }
    return {
        "available": True,
        "task_script_sha256": hashlib.sha256(
            script.encode("utf-8")
        ).hexdigest(),
        "task_script_size_bytes": len(script.encode("utf-8")),
        "exact_scheduler_command_present": command in script,
        "physics_marker_presence": marker_presence,
        "all_physics_markers_present": all(marker_presence.values()),
    }


def collect_once(
    *,
    reader: JsonReader = get_json,
    output: Path | None = None,
) -> dict[str, Any]:
    """GET-authenticate task, allocation, node, and physics hashes."""

    plan, final, receipt = _load_submission()
    task_id = int(final["task_id"])
    task = reader(f"/api/tasks/{task_id}", None)
    expected = {
        "task_id": task_id,
        "name": TASK_NAME,
        "dedupe_key": plan["dedupe_key"],
        "project": PROJECT,
        "requested_account_name": ACCOUNT_NAME,
        "requested_node_name": NODE_NAME,
        "requested_node_name_policy": "strict",
        "preferred_node_relaxed": False,
        "same_node_as_task_id": 0,
        "requested_allocation_id": 0,
        "cpus": CPUS,
        "memory_mb": MEMORY_MB,
        "timeout_seconds": SCHEDULER_SECONDS,
        "max_workers_per_node": 1,
        "aedt_backend": "standalone",
    }
    drift = {
        key: {"expected": value, "actual": task.get(key)}
        for key, value in expected.items()
        if not isinstance(task, Mapping) or task.get(key) != value
    }
    if drift:
        raise PostdeadlineContractError(
            f"hedge task readback drifted: {drift}"
        )
    allocation: dict[str, Any] | None = None
    allocation_id = int(
        task.get("allocation_id")
        or task.get("assigned_allocation")
        or 0
    )
    if task.get("status") in {"attaching", "running"}:
        if (
            task.get("account_name") != ACCOUNT_NAME
            or task.get("node_name") != NODE_NAME
            or task.get("actual_node_name") != NODE_NAME
            or task.get("allocation_node_name") != NODE_NAME
            or allocation_id <= 0
            or task.get("slurm_job_id") in {None, ""}
        ):
            raise PostdeadlineContractError(
                "attached hedge task placement drifted"
            )
        rows = [
            row
            for row in _task_rows(
                reader("/api/allocations", [("limit", 10000)])
            )
            if int(row.get("id") or row.get("allocation_id") or 0)
            == allocation_id
        ]
        if len(rows) != 1:
            raise PostdeadlineContractError(
                "hedge allocation readback is absent or ambiguous"
            )
        allocation = rows[0]
        if (
            allocation.get("account_name") != ACCOUNT_NAME
            or allocation.get("node_name") != NODE_NAME
            or allocation.get("slurm_job_id")
            != task.get("slurm_job_id")
            or allocation.get("state") not in {"active", "draining"}
        ):
            raise PostdeadlineContractError(
                "hedge allocation identity drifted"
            )
    script = _task_script_attestation(task_id, task, plan)
    if script.get("available") and (
        script.get("exact_scheduler_command_present") is not True
        or script.get("all_physics_markers_present") is not True
    ):
        raise PostdeadlineContractError(
            "remote hedge task script physics attestation failed"
        )
    stdout = ""
    stderr = ""
    if task.get("remote_dir"):
        try:
            stdout = _get_text(f"/api/tasks/{task_id}/stdout")
        except Exception:
            stdout = ""
        try:
            stderr = _get_text(f"/api/tasks/{task_id}/stderr")
        except Exception:
            stderr = ""
    value = sealed(
        {
            "schema_version": COLLECTION_SCHEMA,
            **SAFETY_FLAGS,
            "collected_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "task_id": task_id,
            "task_status": task.get("status"),
            "task_readback": copy.deepcopy(dict(task)),
            "task_readback_sha256": payload_sha256(task),
            "allocation_readback": copy.deepcopy(allocation),
            "allocation_readback_sha256": (
                payload_sha256(allocation)
                if allocation is not None
                else None
            ),
            "placement_authenticated": (
                task.get("status") not in {"attaching", "running"}
                or allocation is not None
            ),
            "new_allocation_authenticated": (
                allocation_id > 0
                and allocation is not None
                and allocation.get("node_name") == NODE_NAME
                and allocation.get("account_name") == ACCOUNT_NAME
            ),
            "physics": {
                "candidate_physics_sha256": CANDIDATE_SHA256,
                "physics_contract_sha256": plan[
                    "physics_contract_sha256"
                ],
                "rounded_fea_params_sha256": plan[
                    "physics_contract"
                ]["rounded_fea_params_sha256"],
                "effective_rounded_params_sha256": plan[
                    "physics_contract"
                ]["effective_rounded_params_sha256"],
                "scheduler_payload_sha256": plan[
                    "scheduler_payload_sha256"
                ],
                "task_script_attestation": script,
            },
            "stdout_sha256": hashlib.sha256(
                stdout.encode("utf-8")
            ).hexdigest(),
            "stdout_size_bytes": len(stdout.encode("utf-8")),
            "stdout_tail": stdout[-4_096:],
            "stderr_sha256": hashlib.sha256(
                stderr.encode("utf-8")
            ).hexdigest(),
            "stderr_size_bytes": len(stderr.encode("utf-8")),
            "stderr_tail": stderr[-4_096:],
            "plan_payload_sha256": plan["payload_sha256"],
            "submission_receipt_payload_sha256": receipt[
                "payload_sha256"
            ],
            "scheduler_methods": ["GET"],
            "scheduler_post_calls": 0,
            "scheduler_mutation_performed": False,
        }
    )
    target = (
        output
        or (OUTPUT_ROOT / COLLECTION_DIRECTORY_NAME / "latest.json")
    ).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.tmp"
    )
    temporary.write_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return value


def collect_watch(*, interval_seconds: int = 30) -> int:
    if interval_seconds < 10:
        raise PostdeadlineContractError(
            "collector interval must be at least 10 seconds"
        )
    while True:
        try:
            value = collect_once()
            print(
                json.dumps(
                    {
                        "event": "rounded_n107_hedge_get_collection",
                        "task_id": value["task_id"],
                        "task_status": value["task_status"],
                        "new_allocation_authenticated": value[
                            "new_allocation_authenticated"
                        ],
                        "scheduler_post_calls": 0,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if value["task_status"] not in ACTIVE_TASK_STATES:
                return 0
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "event": (
                            "rounded_n107_hedge_get_collection_error"
                        ),
                        "error": str(exc),
                        "scheduler_post_calls": 0,
                        "automatic_retry": True,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
        time.sleep(interval_seconds)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "prepare",
        help="reauthenticate physics, GET-preflight, and seal without POST",
    )
    commands.add_parser(
        "inspect",
        help="reauthenticate the sealed plan and repeat all GET gates",
    )
    submit_parser = commands.add_parser(
        "submit",
        help="consume the one-shot Scheduler POST budget",
    )
    submit_parser.add_argument("--authorize-post", required=True)
    collect_parser = commands.add_parser(
        "collect", help="GET-authenticate the submitted hedge"
    )
    collect_parser.add_argument("--watch", action="store_true")
    collect_parser.add_argument(
        "--interval-seconds", type=int, default=30
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        plan_path = prepare()
        plan = validate_seal(read_json(plan_path), PLAN_SCHEMA)
        result = {
            "event": "rounded_n107_hedge_prepared",
            "plan": str(plan_path),
            "plan_payload_sha256": plan["payload_sha256"],
            "candidate_physics_sha256": CANDIDATE_SHA256,
            "physics_contract_sha256": plan[
                "physics_contract_sha256"
            ],
            "scheduler_payload_sha256": plan[
                "scheduler_payload_sha256"
            ],
            "scheduler_post_calls": 0,
        }
    elif args.command == "inspect":
        result = inspect()
    elif args.command == "submit":
        final_path = submit(authorize_post=args.authorize_post)
        final = validate_seal(read_json(final_path), FINAL_SEAL_SCHEMA)
        result = {
            "event": "rounded_n107_hedge_submitted",
            "final_seal": str(final_path),
            "task_id": final["task_id"],
            "scheduler_post_calls": final[
                "scheduler_post_calls"
            ],
        }
    elif args.watch:
        return collect_watch(interval_seconds=args.interval_seconds)
    else:
        value = collect_once()
        result = {
            "event": "rounded_n107_hedge_get_collection",
            "task_id": value["task_id"],
            "task_status": value["task_status"],
            "new_allocation_authenticated": value[
                "new_allocation_authenticated"
            ],
            "scheduler_post_calls": 0,
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
                    "event": "rounded_n107_hedge_error",
                    "error": str(exc),
                    "automatic_retry": False,
                    "scheduler_post_calls": 0,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
