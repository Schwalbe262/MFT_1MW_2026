"""Durably recycle only pre-2026-07-12 15:52 KST MFT workers.

One running task is cancelled per cycle.  The continuous b171 controller owns
all replacement submissions; this process only waits until the logical MFT
pool is back at exactly 250 before it can authorize another cancellation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import requests
from filelock import FileLock


HERE = Path(__file__).resolve().parent
REGRESSION = HERE.parent
VERIFY = HERE.parent / "verify"
REPO = REGRESSION.parent
for item in (HERE, REGRESSION, REPO, VERIFY):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import _adopted_refill_sha688c6f9 as durable
import scheduler_client


SOLVER = "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c"
LIBRARY = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
CAMPAIGN_PLAN_SHA256 = (
    "b24e2a9b00caa22bbec8793f4dbd99de51362fac87f9e9509358610abe9982d0"
)
CUTOFF = "2026-07-12T06:52:07+00:00"
CUTOFF_AT = datetime.fromisoformat(CUTOFF)
TARGET_ACTIVE = 250
MAX_INITIAL_CANDIDATES = 51
CAMPAIGN_PREFIX = "mft-camp-"
REVIEWED_ROLLING_PLAN_SHA256 = (
    "66c2555dcf3233fb1748c2a7143c1d0bdbe0ed0ee80f98be1f5db3fe1f83ab28"
)

ROOT = HERE / "pilot_manifests" / "rolling-recycle-prebinding-260712"
PLAN_PATH = ROOT / "plan.json"
LEDGER_PATH = ROOT / "ledger.json"
LEDGER_LOCK_PATH = ROOT / "ledger.lock"
CONTROLLER_STATE_PATH = HERE / "continuous_refill_b171c7c_state.json"
CONTROLLER_SCRIPT = "_continuous_refill_b171c7c.py"

AUTHORIZATION = {
    "schema": "mft-prebinding-rolling-authorization-v1",
    "project": scheduler_client.MFT_PROJECT,
    "solver_revision": SOLVER,
    "library_revision": LIBRARY,
    "campaign_plan_sha256": CAMPAIGN_PLAN_SHA256,
    "cutoff_started_at": CUTOFF,
    "target_active": TARGET_ACTIVE,
    "max_initial_candidates": MAX_INITIAL_CANDIDATES,
    "scope": "running MFT campaign tasks started before final scheduler restart",
    "cancel_batch_size": 1,
    "replacement_owner": "continuous_refill_b171c7c",
}

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
ACTIVE_STATUSES = {"queued", "attaching", "running"}
EXCLUSION_PHASES = {
    "cancel_preauthorized",
    "cancel_acknowledged",
    "cancel_confirmed",
    "replacement_confirmed",
}


def _canonical(value) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


AUTHORIZATION_SHA256 = _sha(AUTHORIZATION)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_time(value) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"invalid scheduler timestamp: {value!r}")
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _write_once_verified(path: Path, payload: dict) -> None:
    """Create immutable JSON without relying on RaiDrive ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True,
    ).encode("utf-8") + b"\n"
    last_error = None
    for attempt in range(1, durable.ATOMIC_ATTEMPTS + 1):
        try:
            with path.open("xb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        except (FileExistsError, OSError) as exc:
            last_error = exc
        try:
            if path.read_bytes() == encoded:
                return
        except OSError as exc:
            last_error = exc
        if attempt < durable.ATOMIC_ATTEMPTS:
            time.sleep(durable.ATOMIC_RETRY_SECONDS)
    raise RuntimeError(f"verified immutable JSON create failed for {path}: {last_error}")


def _sealed(payload: dict, field: str) -> dict:
    unsigned = dict(payload)
    stored = unsigned.pop(field, None)
    if not isinstance(stored, str) or stored != _sha(unsigned):
        raise RuntimeError(f"{field} seal mismatch")
    return payload


def _get(path: str, **params):
    response = requests.get(
        f"{scheduler_client.SCHEDULER}{path}",
        params=params or None,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _rows(payload) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("tasks", "value"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise RuntimeError("scheduler returned an invalid task inventory")


def _tasks(**params) -> list[dict]:
    return _rows(_get("/api/tasks", limit=10000, **params))


def _task(task_id: int) -> dict:
    row = _get(f"/api/tasks/{int(task_id)}")
    if not isinstance(row, dict):
        raise RuntimeError(f"task {task_id} detail is invalid")
    return row


def _task_id(row: dict) -> int:
    value = row.get("id", row.get("task_id"))
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError("scheduler task has an invalid ID")
    return value


def _is_exact_identity(row: dict) -> bool:
    name = str(row.get("name") or "")
    dedupe = str(row.get("dedupe_key") or "")
    return (
        row.get("project") == scheduler_client.MFT_PROJECT
        and name.startswith(CAMPAIGN_PREFIX)
        and f":{SOLVER}:{LIBRARY}:" in dedupe
        and dedupe.startswith(f"mft-al:{name}:")
    )


def _candidate_from_live(row: dict) -> dict:
    task_id = _task_id(row)
    checks = {
        "project": row.get("project") == scheduler_client.MFT_PROJECT,
        "running": row.get("status") == "running",
        "exact_identity": _is_exact_identity(row),
        "started": bool(row.get("started_at")),
        "attached": bool(row.get("attached_at")),
        "pre_cutoff": bool(row.get("started_at"))
            and _parse_time(row["started_at"]) < CUTOFF_AT,
        "allocation": isinstance(row.get("allocation_id"), int)
            and int(row["allocation_id"]) > 0,
        "cpus": row.get("cpus") == 4,
        "memory_mb": row.get("memory_mb") == 65536,
        "scheduling_profile": row.get("scheduling_profile") == "fea_bursty",
    }
    if not all(checks.values()):
        raise RuntimeError(f"task {task_id} is outside rolling scope: {checks}")
    return {
        "task_id": task_id,
        "name": row["name"],
        "dedupe_key": row["dedupe_key"],
        "started_at": row["started_at"],
        "attached_at": row["attached_at"],
        "allocation_id": int(row["allocation_id"]),
        "slurm_job_id": str(row.get("slurm_job_id") or ""),
        "account_name": str(row.get("account_name") or ""),
        "node_name": str(
            row.get("actual_node_name")
            or row.get("allocation_node_name")
            or row.get("node_name") or ""),
    }


def build_plan() -> dict:
    running = _tasks(
        project=scheduler_client.MFT_PROJECT,
        status="running",
        include_diagnostics="true",
    )
    candidates = []
    for row in running:
        started_at = row.get("started_at")
        if not started_at or _parse_time(started_at) >= CUTOFF_AT:
            continue
        candidates.append(_candidate_from_live(row))
    candidates.sort(key=lambda item: (item["started_at"], item["task_id"]))
    if not (1 <= len(candidates) <= MAX_INITIAL_CANDIDATES):
        raise RuntimeError(
            f"initial rolling cohort must contain 1.."
            f"{MAX_INITIAL_CANDIDATES} tasks, got {len(candidates)}")
    unsigned = {
        "schema": "mft-prebinding-rolling-plan-v1",
        "created_at": _now(),
        "authorization": AUTHORIZATION,
        "authorization_sha256": AUTHORIZATION_SHA256,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    return {**unsigned, "plan_sha256": _sha(unsigned)}


def validate_plan(payload: dict) -> dict:
    _sealed(payload, "plan_sha256")
    if payload.get("plan_sha256") != REVIEWED_ROLLING_PLAN_SHA256:
        raise RuntimeError("rolling plan is not the reviewed sealed cohort")
    if payload.get("schema") != "mft-prebinding-rolling-plan-v1":
        raise RuntimeError("rolling plan schema drifted")
    if payload.get("authorization") != AUTHORIZATION \
            or payload.get("authorization_sha256") != AUTHORIZATION_SHA256:
        raise RuntimeError("rolling plan authorization drifted")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) \
            or not (1 <= len(candidates) <= MAX_INITIAL_CANDIDATES) \
            or payload.get("candidate_count") != len(candidates):
        raise RuntimeError("rolling plan candidate count drifted")
    seen = set()
    for row in candidates:
        if not isinstance(row, dict):
            raise RuntimeError("rolling plan candidate is invalid")
        task_id = row.get("task_id")
        if (isinstance(task_id, bool) or not isinstance(task_id, int)
                or task_id <= 0 or task_id in seen):
            raise RuntimeError("rolling plan has an invalid/duplicate task ID")
        seen.add(task_id)
        if (_parse_time(row.get("started_at")) >= CUTOFF_AT
                or not str(row.get("name") or "").startswith(CAMPAIGN_PREFIX)
                or f":{SOLVER}:{LIBRARY}:" not in str(row.get("dedupe_key") or "")):
            raise RuntimeError(f"rolling plan task {task_id} identity drifted")
    return payload


def _new_ledger(plan: dict) -> dict:
    unsigned = {
        "schema": "mft-prebinding-rolling-ledger-v2",
        "state_revision": 0,
        "created_at": _now(),
        "ledger_updated_at": _now(),
        "authorization_sha256": AUTHORIZATION_SHA256,
        "plan_sha256": plan["plan_sha256"],
        "entries": [],
        "completed": False,
    }
    seal_input = dict(unsigned)
    seal_input.pop("state_revision")
    return {**unsigned, "ledger_sha256": _sha(seal_input)}


def _ledger_seal_input(payload: dict) -> dict:
    unsigned = dict(payload)
    unsigned.pop("ledger_sha256", None)
    unsigned.pop("state_revision", None)
    return unsigned


def validate_ledger(payload: dict, plan: dict) -> dict:
    if payload.get("ledger_sha256") != _sha(_ledger_seal_input(payload)):
        raise RuntimeError("ledger_sha256 seal mismatch")
    revision = payload.get("state_revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise RuntimeError("rolling ledger state revision is invalid")
    if payload.get("schema") != "mft-prebinding-rolling-ledger-v2" \
            or payload.get("authorization_sha256") != AUTHORIZATION_SHA256 \
            or payload.get("plan_sha256") != plan.get("plan_sha256") \
            or not isinstance(payload.get("entries"), list):
        raise RuntimeError("rolling ledger contract drifted")
    expected = {row["task_id"]: row for row in plan["candidates"]}
    seen_sequences = set()
    seen_task_ids = set()
    for entry in payload["entries"]:
        if not isinstance(entry, dict):
            raise RuntimeError("rolling ledger entry is invalid")
        sequence = entry.get("sequence")
        task_id = entry.get("task_id")
        if (isinstance(sequence, bool) or not isinstance(sequence, int)
                or sequence <= 0 or sequence in seen_sequences):
            raise RuntimeError("rolling ledger sequence is invalid/duplicate")
        seen_sequences.add(sequence)
        if (isinstance(task_id, bool) or not isinstance(task_id, int)
                or task_id in seen_task_ids):
            raise RuntimeError("rolling ledger task ID is invalid/duplicate")
        seen_task_ids.add(task_id)
        planned = expected.get(task_id)
        if planned is None:
            raise RuntimeError(f"rolling ledger task {task_id!r} is not planned")
        for key in ("name", "dedupe_key", "started_at", "allocation_id"):
            if entry.get(key) != planned.get(key):
                raise RuntimeError(
                    f"rolling ledger task {task_id} {key} drifted")
        if entry.get("phase") not in (
                EXCLUSION_PHASES | {"status_race", "natural_terminal"}):
            raise RuntimeError(f"rolling ledger task {task_id} phase is invalid")
        before_ids = entry.get("active_ids_before")
        if (not isinstance(before_ids, list) or len(before_ids) != TARGET_ACTIVE
                or len(set(before_ids)) != TARGET_ACTIVE
                or task_id not in before_ids
                or any(isinstance(value, bool) or not isinstance(value, int)
                       or value <= 0 for value in before_ids)):
            raise RuntimeError(
                f"rolling ledger task {task_id} pre-cancel inventory is invalid")
    if seen_sequences != set(range(1, len(payload["entries"]) + 1)):
        raise RuntimeError("rolling ledger sequences are not contiguous")
    return payload


def _save_ledger(ledger: dict) -> None:
    committed = dict(ledger)
    committed["ledger_updated_at"] = _now()
    committed["ledger_sha256"] = _sha(_ledger_seal_input(committed))
    durable._save_durable_state(
        LEDGER_PATH, committed,
        lambda value: validate_ledger(value, _read_plan()),
        transition_validator=_validate_ledger_transition,
    )


def _validate_ledger_transition(previous: dict, current: dict) -> None:
    old_entries = previous["entries"]
    new_entries = current["entries"]
    if len(new_entries) not in {len(old_entries), len(old_entries) + 1}:
        raise RuntimeError("rolling ledger entry count transition is invalid")
    if len(new_entries) == len(old_entries) + 1:
        if (new_entries[:-1] != old_entries
                or new_entries[-1].get("phase") != "cancel_preauthorized"
                or new_entries[-1].get("sequence") != len(new_entries)):
            raise RuntimeError("rolling ledger append transition is invalid")
        return
    changed = [
        index for index, (old, new) in enumerate(zip(old_entries, new_entries))
        if old != new
    ]
    if not changed:
        if previous.get("completed") is False and current.get("completed") is True:
            return
        raise RuntimeError("rolling ledger save has no substantive transition")
    if len(changed) != 1 or changed[0] != len(old_entries) - 1:
        raise RuntimeError("rolling ledger can advance only its latest entry")
    old_phase = old_entries[-1]["phase"]
    new_phase = new_entries[-1]["phase"]
    allowed = {
        "cancel_preauthorized": {
            "cancel_acknowledged", "cancel_confirmed",
            "status_race", "natural_terminal",
        },
        "cancel_acknowledged": {"cancel_confirmed", "natural_terminal"},
        "cancel_confirmed": {"replacement_confirmed"},
    }
    if new_phase not in allowed.get(old_phase, set()):
        raise RuntimeError(
            f"rolling ledger phase transition {old_phase}->{new_phase} is invalid")


def _read_plan() -> dict:
    return validate_plan(json.loads(PLAN_PATH.read_text(encoding="utf-8")))


def _ledger_generations_exist() -> bool:
    return any(LEDGER_PATH.parent.glob(f"{LEDGER_PATH.name}.gen-*.json"))


def _migrate_v1_ledger(plan: dict, legacy: dict) -> dict:
    _sealed(legacy, "ledger_sha256")
    if legacy.get("schema") != "mft-prebinding-rolling-ledger-v1":
        raise RuntimeError("unknown legacy rolling ledger schema")
    migrated = dict(legacy)
    migrated.pop("ledger_sha256", None)
    migrated["schema"] = "mft-prebinding-rolling-ledger-v2"
    migrated["state_revision"] = 0
    migrated["ledger_updated_at"] = migrated.pop("updated_at", _now())
    migrated["ledger_sha256"] = _sha(_ledger_seal_input(migrated))
    validate_ledger(migrated, plan)
    return durable._initialize_durable_state(
        LEDGER_PATH, migrated, lambda value: validate_ledger(value, plan))


def _load_ledger(plan: dict, *, create: bool) -> dict:
    if LEDGER_PATH.exists() and not _ledger_generations_exist():
        raw = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
        if raw.get("schema") == "mft-prebinding-rolling-ledger-v1":
            if not create:
                raise RuntimeError("rolling ledger requires locked v1 migration")
            return _migrate_v1_ledger(plan, raw)
    return durable._load_durable_state(
        LEDGER_PATH,
        lambda value: validate_ledger(value, plan),
        lambda: _new_ledger(plan),
        create=create,
    )


def initialize() -> tuple[dict, dict]:
    ROOT.mkdir(parents=True, exist_ok=True)
    with FileLock(str(LEDGER_LOCK_PATH), timeout=30):
        if PLAN_PATH.exists():
            plan = _read_plan()
        else:
            with scheduler_client.campaign_mutation_lock():
                snapshot = scheduler_client.live_project_submission_snapshot(
                    TARGET_ACTIVE)
                if snapshot["project_active"] != TARGET_ACTIVE:
                    raise RuntimeError(
                        f"rolling plan requires exact active250: {snapshot}")
                plan = build_plan()
                _write_once_verified(PLAN_PATH, plan)
        ledger = _load_ledger(plan, create=True)
        return plan, ledger


def load_plan_and_ledger() -> tuple[dict, dict]:
    if not PLAN_PATH.is_file() or (
            not LEDGER_PATH.is_file() and not _ledger_generations_exist()):
        raise RuntimeError("rolling plan/ledger is not initialized")
    plan = _read_plan()
    ledger = _load_ledger(plan, create=False)
    return plan, ledger


def authorized_cancelled_task_ids(inventory: list[dict]) -> set[int]:
    """IDs whose intentional cancellation can be omitted from health only.

    Running, completed, failed, unplanned, or non-preauthorized cancelled tasks
    are never excluded.
    """
    if not PLAN_PATH.exists() and not LEDGER_PATH.exists():
        return set()
    plan, ledger = load_plan_and_ledger()
    planned = {row["task_id"]: row for row in plan["candidates"]}
    authorized = {
        entry["task_id"] for entry in ledger["entries"]
        if entry.get("phase") in EXCLUSION_PHASES
    }
    excluded = set()
    for row in inventory:
        task_id = _task_id(row)
        if task_id not in authorized or row.get("status") != "cancelled":
            continue
        expected = planned[task_id]
        checks = {
            "project": row.get("project") == scheduler_client.MFT_PROJECT,
            "name": row.get("name") == expected["name"],
            "dedupe": row.get("dedupe_key") == expected["dedupe_key"],
            "started_at": row.get("started_at") == expected["started_at"],
            "pre_cutoff": _parse_time(row.get("started_at")) < CUTOFF_AT,
        }
        if not all(checks.values()):
            raise RuntimeError(
                f"authorized rolling cancellation {task_id} drifted: {checks}")
        excluded.add(task_id)
    return excluded


def _controller_processes() -> list[dict]:
    matches = []
    for proc in psutil.process_iter(["pid", "cmdline", "create_time"]):
        try:
            argv = list(proc.info.get("cmdline") or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        joined = " ".join(argv)
        if CONTROLLER_SCRIPT not in joined:
            continue
        required = (
            "--execute",
            "--authorize-concurrent-pool250",
            CAMPAIGN_PLAN_SHA256,
            "--loop",
            "60",
        )
        if all(token in argv or token in joined for token in required):
            matches.append({
                "pid": int(proc.info["pid"]),
                "cmdline": argv,
                "create_time": float(proc.info.get("create_time") or 0),
            })
    return matches


def _controller_health() -> dict:
    processes = _controller_processes()
    if len(processes) != 1:
        raise RuntimeError(
            f"expected exactly one exact controller, found {len(processes)}")
    state = json.loads(CONTROLLER_STATE_PATH.read_text(encoding="utf-8"))
    evidence = state.get("last_evidence") or {}
    updated = _parse_time(state.get("updated_at"))
    age = (datetime.now(timezone.utc) - updated).total_seconds()
    checks = {
        "fresh_state": age <= 300,
        "not_paused": evidence.get("paused") is False,
        "refill_action": evidence.get("action") == "refill_250",
        "no_pause_reasons": evidence.get("pause_reasons") == [],
    }
    if not all(checks.values()):
        raise RuntimeError(f"controller is not healthy: {checks}")
    return {"process": processes[0], "state_age_seconds": age, "checks": checks}


def _active_rows() -> list[dict]:
    return _tasks(status="queued,attaching,running", include_diagnostics="true")


def _exact_active_snapshot() -> dict:
    with scheduler_client.campaign_mutation_lock():
        return scheduler_client.live_project_submission_snapshot(TARGET_ACTIVE)


def _wait_until_ready(timeout: int, sleeper=time.sleep) -> dict:
    """Wait out ordinary refill lag; unhealthy controllers still fail closed."""
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        controller = _controller_health()
        with scheduler_client.campaign_mutation_lock():
            snapshot = scheduler_client.live_project_submission_snapshot(
                TARGET_ACTIVE)
        active = int(snapshot["project_active"])
        last = {"controller": controller, "snapshot": snapshot}
        if active == TARGET_ACTIVE:
            return last
        if active > TARGET_ACTIVE:
            raise RuntimeError(f"rolling pool exceeds exact250: {snapshot}")
        sleeper(5)
    raise TimeoutError(f"controller did not restore exact250: {last}")


def _entry_by_task(ledger: dict) -> dict[int, dict]:
    return {int(row["task_id"]): row for row in ledger["entries"]}


def _select_candidate(plan: dict, ledger: dict) -> tuple[dict | None, dict]:
    entries = _entry_by_task(ledger)
    active = _active_rows()
    by_id = {_task_id(row): row for row in active}
    allocation_active = {}
    for row in active:
        if row.get("status") not in {"attaching", "running"}:
            continue
        allocation_id = row.get("allocation_id")
        if isinstance(allocation_id, int) and allocation_id > 0:
            allocation_active[allocation_id] = allocation_active.get(allocation_id, 0) + 1

    pending = []
    natural = []
    for planned in plan["candidates"]:
        if planned["task_id"] in entries:
            continue
        live = by_id.get(planned["task_id"])
        if live is None:
            detail = _task(planned["task_id"])
            if detail.get("status") in TERMINAL_STATUSES:
                natural.append((planned, detail.get("status")))
                continue
            raise RuntimeError(
                f"planned task {planned['task_id']} disappeared from active inventory")
        if live.get("status") != "running":
            continue
        checked = _candidate_from_live(live)
        for key in ("task_id", "name", "dedupe_key", "started_at", "allocation_id"):
            if checked[key] != planned[key]:
                raise RuntimeError(
                    f"planned task {planned['task_id']} live {key} drifted")
        pending.append(planned)

    old_counts = {}
    for row in pending:
        aid = row["allocation_id"]
        old_counts[aid] = old_counts.get(aid, 0) + 1
    last_allocation = None
    if ledger["entries"]:
        last_allocation = ledger["entries"][-1].get("allocation_id")
    pending.sort(key=lambda row: (
        allocation_active.get(row["allocation_id"], 0) <= 1,
        row["allocation_id"] == last_allocation,
        -old_counts.get(row["allocation_id"], 0),
        -allocation_active.get(row["allocation_id"], 0),
        row["started_at"],
        row["task_id"],
    ))
    selected = pending[0] if pending else None
    audit = {
        "allocation_active_counts": allocation_active,
        "allocation_old_counts": old_counts,
        "remaining_running_candidates": len(pending),
        "natural_terminal_unrecorded": [
            {"task_id": row["task_id"], "status": status}
            for row, status in natural
        ],
        "last_allocation_id": last_allocation,
    }
    return selected, audit


def _append_entry(ledger: dict, entry: dict) -> dict:
    ledger = dict(ledger)
    ledger["entries"] = [*ledger["entries"], entry]
    _save_ledger(ledger)
    return _load_ledger(_read_plan(), create=False)


def _replace_entry(ledger: dict, sequence: int, **updates) -> dict:
    ledger = dict(ledger)
    rows = []
    matched = 0
    for entry in ledger["entries"]:
        if entry["sequence"] == sequence:
            entry = {**entry, **updates}
            matched += 1
        rows.append(entry)
    if matched != 1:
        raise RuntimeError(f"rolling ledger sequence {sequence} is ambiguous")
    ledger["entries"] = rows
    _save_ledger(ledger)
    return _load_ledger(_read_plan(), create=False)


def _wait_for_replacement(before_ids: set[int], timeout: int) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        controller = _controller_health()
        with scheduler_client.campaign_mutation_lock():
            snapshot = scheduler_client.live_project_submission_snapshot(TARGET_ACTIVE)
            active = _tasks(
                project=scheduler_client.MFT_PROJECT,
                status="queued,attaching,running",
                include_diagnostics="true",
            )
        replacements = [
            row for row in active
            if _task_id(row) not in before_ids and _is_exact_identity(row)
        ]
        last = {
            "snapshot": snapshot,
            "controller": controller,
            "replacement_ids": [_task_id(row) for row in replacements],
            "replacement_names": [row.get("name") for row in replacements],
        }
        if snapshot["project_active"] == TARGET_ACTIVE and replacements:
            return last
        time.sleep(5)
    raise TimeoutError(f"replacement did not restore exact250: {last}")


def execute_one(plan: dict, ledger: dict, replacement_timeout: int) -> dict:
    with FileLock(str(LEDGER_LOCK_PATH), timeout=30):
        plan, ledger = load_plan_and_ledger()
        open_entries = [
            row for row in ledger["entries"]
            if row["phase"] in {"cancel_preauthorized", "cancel_acknowledged", "cancel_confirmed"}
        ]
        if len(open_entries) > 1:
            raise RuntimeError("rolling ledger has multiple unfinished sequences")

        allocation_audit = None
        if open_entries:
            entry = open_entries[0]
            sequence = int(entry["sequence"])
            selected = next(
                row for row in plan["candidates"]
                if row["task_id"] == entry["task_id"])
            before_ids = set(entry["active_ids_before"])
            controller_before = _controller_health()
        else:
            controller_before = _controller_health()
            with scheduler_client.campaign_mutation_lock():
                snapshot_before = scheduler_client.live_project_submission_snapshot(
                    TARGET_ACTIVE)
                if snapshot_before["project_active"] < TARGET_ACTIVE:
                    return {
                        "action": "wait_refill",
                        "snapshot": snapshot_before,
                    }
                if snapshot_before["project_active"] > TARGET_ACTIVE:
                    raise RuntimeError(
                        f"rolling pool exceeds exact250: {snapshot_before}")
                before_rows = _tasks(
                    project=scheduler_client.MFT_PROJECT,
                    status="queued,attaching,running",
                    include_diagnostics="true",
                )
                before_ids = {_task_id(row) for row in before_rows}
                selected, allocation_audit = _select_candidate(plan, ledger)
                if selected is None:
                    ledger = dict(ledger)
                    ledger["completed"] = True
                    _save_ledger(ledger)
                    return {
                        "action": "complete",
                        "snapshot": snapshot_before,
                        "allocation_audit": allocation_audit,
                    }
                live = _task(selected["task_id"])
                checked = _candidate_from_live(live)
                for key in (
                        "task_id", "name", "dedupe_key", "started_at",
                        "allocation_id"):
                    if checked[key] != selected[key]:
                        raise RuntimeError(
                            f"selected task {selected['task_id']} {key} drifted")
                allocation_id = selected["allocation_id"]
                allocation_active = allocation_audit[
                    "allocation_active_counts"].get(allocation_id, 0)
                if (allocation_active < 2
                        and allocation_audit["remaining_running_candidates"] > 1):
                    raise RuntimeError(
                        f"allocation {allocation_id} would be emptied too early")
                sequence = len(ledger["entries"]) + 1
                entry = {
                    **selected,
                    "sequence": sequence,
                    "phase": "cancel_preauthorized",
                    "cancel_preauthorized_at": _now(),
                    "active_ids_before": sorted(before_ids),
                    "snapshot_before": snapshot_before,
                    "controller_before": controller_before,
                    "allocation_active_before": allocation_active,
                    "allocation_old_before": allocation_audit[
                        "allocation_old_counts"].get(allocation_id, 0),
                }
                ledger = _append_entry(ledger, entry)

        phase = next(
            row["phase"] for row in ledger["entries"]
            if row["sequence"] == sequence)
        if phase == "cancel_preauthorized":
            with scheduler_client.campaign_mutation_lock():
                live = _task(selected["task_id"])
                if live.get("status") == "running":
                    checked = _candidate_from_live(live)
                    if checked["dedupe_key"] != selected["dedupe_key"]:
                        raise RuntimeError("resumed rolling task identity drifted")
                    response = requests.post(
                        f"{scheduler_client.SCHEDULER}/api/tasks/"
                        f"{selected['task_id']}/cancel",
                        params={"expected_statuses": "running"}, timeout=60)
                    response.raise_for_status()
                    acknowledgement = response.json()
                    live = _task(selected["task_id"])
                    if acknowledgement.get("cancelled"):
                        ledger = _replace_entry(
                            ledger, sequence,
                            phase="cancel_acknowledged",
                            cancellation_acknowledgement=acknowledgement,
                            cancel_acknowledged_at=_now())
                    elif live.get("status") not in {"cancelled"}:
                        ledger = _replace_entry(
                            ledger, sequence, phase="status_race",
                            cancellation_acknowledgement=acknowledgement,
                            status_race_at=_now(),
                            task_status_after=live.get("status"))
                        return {
                            "action": "status_race",
                            "task_id": selected["task_id"],
                            "acknowledgement": acknowledgement,
                        }
                if live.get("status") in {"completed", "failed"}:
                    ledger = _replace_entry(
                        ledger, sequence, phase="natural_terminal",
                        natural_terminal_at=_now(),
                        task_status_after=live.get("status"))
                    return {
                        "action": "natural_terminal",
                        "task_id": selected["task_id"],
                        "status": live.get("status"),
                    }

        deadline = time.monotonic() + 120
        after = None
        while time.monotonic() < deadline:
            after = _task(selected["task_id"])
            if after.get("status") == "cancelled":
                break
            if after.get("status") in {"completed", "failed"}:
                ledger = _replace_entry(
                    ledger, sequence, phase="natural_terminal",
                    natural_terminal_at=_now(),
                    task_status_after=after.get("status"))
                return {
                    "action": "natural_terminal",
                    "task_id": selected["task_id"],
                    "status": after.get("status"),
                }
            time.sleep(2)
        if not after or after.get("status") != "cancelled":
            raise RuntimeError(
                f"task {selected['task_id']} cancellation did not settle: {after}")
        current = next(
            row for row in ledger["entries"] if row["sequence"] == sequence)
        if current["phase"] != "cancel_confirmed":
            ledger = _replace_entry(
                ledger, sequence,
                phase="cancel_confirmed",
                cancel_confirmed_at=_now(),
                task_status_after=after.get("status"),
            )

        replacement = _wait_for_replacement(before_ids, replacement_timeout)
        plan, ledger = load_plan_and_ledger()
        ledger = _replace_entry(
            ledger, sequence,
            phase="replacement_confirmed",
            replacement_confirmed_at=_now(),
            replacement_evidence=replacement,
        )
    return {
        "action": "recycled_one",
        "sequence": sequence,
        "task_id": selected["task_id"],
        "name": selected["name"],
        "allocation_id": selected["allocation_id"],
        "replacement": replacement,
        "remaining_running_candidates_before": (
            allocation_audit["remaining_running_candidates"]
            if allocation_audit else None),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--reviewed-plan-sha")
    parser.add_argument("--authorize-rolling-recycle51", action="store_true")
    parser.add_argument("--max-cycles", type=int, default=1)
    parser.add_argument("--replacement-timeout", type=int, default=1800)
    parser.add_argument("--cycle-delay", type=int, default=5)
    args = parser.parse_args(argv)
    if args.initialize:
        plan, ledger = initialize()
    elif PLAN_PATH.exists() and LEDGER_PATH.exists():
        plan, ledger = load_plan_and_ledger()
    else:
        plan = build_plan()
        ledger = _new_ledger(plan)

    if not args.execute:
        print(json.dumps({
            "authorization_sha256": AUTHORIZATION_SHA256,
            "plan": plan,
            "ledger": ledger,
            "mode": "initialized" if args.initialize else "audit_only",
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if (args.reviewed_plan_sha != plan["plan_sha256"]
            or not args.authorize_rolling_recycle51):
        parser.error(
            "execute requires exact --reviewed-plan-sha and "
            "--authorize-rolling-recycle51")
    if args.max_cycles < 1 or args.replacement_timeout < 60:
        parser.error("invalid rolling cycle/timeout")
    completed_cycles = 0
    while completed_cycles < args.max_cycles:
        _wait_until_ready(args.replacement_timeout)
        result = execute_one(plan, ledger, args.replacement_timeout)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        if result["action"] == "complete":
            return 0
        if result["action"] == "wait_refill":
            continue
        completed_cycles += 1
        if completed_cycles < args.max_cycles:
            time.sleep(max(0, args.cycle_delay))
        plan, ledger = load_plan_and_ledger()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
