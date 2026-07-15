"""One-shot, resumable replacement of the authorized SHA3216 250-task cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from filelock import FileLock


HERE = Path(__file__).resolve().parent
REGRESSION_ROOT = HERE.parent
REPO_ROOT = REGRESSION_ROOT.parent
VERIFY_ROOT = REGRESSION_ROOT / "verify"
for item in (HERE, REGRESSION_ROOT, VERIFY_ROOT, REPO_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import feeder
import pinned_pilot
import scheduler_client
from training.checkpoint_contract import (
    checkpoint_status_revision_identity_matches,
)


OLD_SOLVER = "3216e43a5a1a362ee2ed1aba89b642498c60d1b9"
NEW_SOLVER = "688c6f9ae8b1368d2b4424e42fc8973b3c580d24"
LIBRARY = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
SEED = 260710
CURSOR_START = 939
COUNT = 250
FIRST_SERIAL = 17112
OLD_FIRST_ID = 27149
OLD_LAST_ID = 27398
OLD_PREFIX = f"mft-camp-s{OLD_SOLVER[:7]}-l{LIBRARY[:7]}-"
NEW_PREFIX = f"mft-camp-s{NEW_SOLVER[:7]}-l{LIBRARY[:7]}-"
CPUS = 4
MEMORY_MB = 65_536
TIMEOUT_SECONDS = 14_400
ACTIVE = ("queued", "attaching", "running")
EVIDENCE_ROOT = HERE / "pilot_manifests"
STEM = f"replacement-s{NEW_SOLVER[:7]}-l{LIBRARY[:7]}-seed{SEED}-cursor{CURSOR_START}"
MANIFEST_PATH = EVIDENCE_ROOT / f"{STEM}.json"
JOURNAL_PATH = EVIDENCE_ROOT / f"{STEM}.journal.json"
SUPERSEDED_MANIFEST_SHA256 = "d1e7a97b9e9b6ac41aba05e27f2a16b1c4ee4a3959c96d1c7cd6b0fea0c8f4a2"
SUPERSEDED_MANIFEST_PATH = EVIDENCE_ROOT / (
    f"{STEM}.superseded-{SUPERSEDED_MANIFEST_SHA256[:12]}.json"
)
STRICT_STATUS_PATH = REGRESSION_ROOT / "training" / "strict_data_status.json"
DATASET_PATH = REGRESSION_ROOT / "data" / "dataset" / "train.parquet"
SCHEDULER = scheduler_client.SCHEDULER


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    staged.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    last_error = None
    for attempt in range(1, 21):
        try:
            os.replace(staged, path)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt < 20:
                time.sleep(0.25)
    raise last_error


def _profile() -> dict:
    payload = json.loads(Path(feeder.PROFILE_PATH).read_text(encoding="utf-8"))
    if payload.get("param_overrides") != scheduler_client.STANDARD_PROFILE_CONTRACT:
        raise RuntimeError("standard profile contract drifted")
    # This authorized production wave deliberately extends only the scheduler
    # wall timeout; the strict standard parameter/CLI contract remains intact.
    payload["timeout_seconds"] = TIMEOUT_SECONDS
    return payload


def _manifest_payload() -> dict:
    profile = _profile()
    cursor = CURSOR_START
    records = []
    for offset in range(COUNT):
        cursor, raw_index, params = pinned_pilot.next_valid_candidate(cursor, seed=SEED)
        # Manifest files are emitted with sort_keys=True. Canonicalize through
        # that exact representation before deriving the order-sensitive
        # scheduler parameter digest, so planned and replayed identities match.
        params = json.loads(_canonical(params))
        serial = FIRST_SERIAL + offset
        name = f"{NEW_PREFIX}{serial:05d}"
        identity = scheduler_client.verification_submission_identity(
            name, params, profile, NEW_SOLVER, LIBRARY
        )
        cw1 = float(params["cw1"])
        if not math.isfinite(cw1) or cw1 > 10.0:
            raise RuntimeError(f"candidate cw1 is outside the authorized cap: {raw_index}/{cw1}")
        if int(params["N1_main"]) > 8:
            raise RuntimeError(f"candidate N1_main exceeds 8: {raw_index}/{params['N1_main']}")
        for key in ("wcp_t", "core_plate_t"):
            if not 10.0 <= float(params[key]) <= 30.0:
                raise RuntimeError(f"candidate {key} is outside [10,30]: {raw_index}")
        if float(params["wcp_pad_t"]) != 2.0 or float(params["core_plate_pad_t"]) != 2.0:
            raise RuntimeError(f"candidate thermal pad thickness drifted: {raw_index}")
        records.append({
            "index": offset,
            "serial": serial,
            "name": name,
            "workdir": f"mft_r_t{serial % 500:03d}",
            "candidate_cursor_before": records[-1]["candidate_cursor_after"] if records else CURSOR_START,
            "candidate_cursor_after": int(cursor),
            "candidate_raw_index": int(raw_index),
            "params_sha256": pinned_pilot.candidate_digest(params),
            "parameter_digest": identity["parameter_digest"],
            "dedupe_key": identity["dedupe_key"],
            "params": params,
            "effective_params": identity["merged"],
        })
    if len({row["name"] for row in records}) != COUNT:
        raise RuntimeError("replacement names are not unique")
    if len({row["dedupe_key"] for row in records}) != COUNT:
        raise RuntimeError("replacement dedupe keys are not unique")
    if len({row["params_sha256"] for row in records}) != COUNT:
        raise RuntimeError("replacement candidate payloads are not unique")
    payload = {
        "schema_version": 2,
        "created_at": _now(),
        "authorization": "replace active SHA3216 cohort with exact 250 SHA688c6f9 tasks",
        "identity_serialization": "sorted-key manifest JSON reloaded before scheduler identity",
        "supersedes_manifest_sha256": SUPERSEDED_MANIFEST_SHA256,
        "old_solver_revision": OLD_SOLVER,
        "solver_revision": NEW_SOLVER,
        "library_revision": LIBRARY,
        "seed": SEED,
        "candidate_cursor_start": CURSOR_START,
        "candidate_cursor_end": int(cursor),
        "task_count": COUNT,
        "first_serial": FIRST_SERIAL,
        "last_serial": FIRST_SERIAL + COUNT - 1,
        "old_prefix": OLD_PREFIX,
        "task_prefix": NEW_PREFIX,
        "resources": {
            "project": scheduler_client.MFT_PROJECT,
            "cpus": CPUS,
            "memory_mb": MEMORY_MB,
            "gpus": 0,
            "timeout_seconds": TIMEOUT_SECONDS,
            "required_capability": "conda:pyaedt2026v1",
            "env_profile": "pyaedt2026v1",
            "scheduling_profile": "fea_bursty",
            "priority": 0,
            "remote_cwd": scheduler_client.GPFS_RUNS_REMOTE_CWD,
        },
        "tasks": records,
    }
    payload["manifest_sha256"] = _sha(payload)
    return payload


def _validate_manifest(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise RuntimeError("replacement manifest is invalid")
    seal = payload.get("manifest_sha256")
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    if seal != _sha(unsigned):
        raise RuntimeError("replacement manifest seal mismatch")
    expected = _manifest_payload()
    # Creation time is evidence metadata; every operational field must match.
    for candidate in (payload, expected):
        candidate.pop("created_at", None)
        candidate.pop("manifest_sha256", None)
    if payload != expected:
        raise RuntimeError("replacement manifest differs from deterministic plan")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def ensure_manifest() -> dict:
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    if MANIFEST_PATH.exists():
        existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if existing.get("manifest_sha256") == SUPERSEDED_MANIFEST_SHA256:
            if not SUPERSEDED_MANIFEST_PATH.exists():
                _atomic_json(SUPERSEDED_MANIFEST_PATH, existing)
            _atomic_json(MANIFEST_PATH, _manifest_payload())
    else:
        _atomic_json(MANIFEST_PATH, _manifest_payload())
    return _validate_manifest(json.loads(MANIFEST_PATH.read_text(encoding="utf-8")))


def _tasks(prefix: str) -> list[dict]:
    last_error = None
    for attempt in range(1, 4):
        try:
            response = requests.get(
                f"{SCHEDULER}/api/tasks",
                params={"limit": 10000, "name_prefix": prefix},
                timeout=90,
            )
            response.raise_for_status()
            payload = response.json()
            rows = payload if isinstance(payload, list) else payload.get("tasks")
            if not isinstance(rows, list):
                raise RuntimeError("scheduler returned an invalid task inventory")
            return rows
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2)
    raise RuntimeError(f"scheduler inventory remained unavailable: {last_error}")


def _state(task: dict) -> str:
    return str(task.get("status") or task.get("state") or "").strip().lower()


def _validate_old_inventory(rows: list[dict]) -> dict:
    selected = [row for row in rows if OLD_FIRST_ID <= int(row.get("id") or 0) <= OLD_LAST_ID]
    if len(selected) != COUNT or len({int(row["id"]) for row in selected}) != COUNT:
        raise RuntimeError(f"old cohort inventory is not exact 250: {len(selected)}")
    for offset, row in enumerate(sorted(selected, key=lambda item: int(item["id"]))):
        expected_id = OLD_FIRST_ID + offset
        expected_name = f"{OLD_PREFIX}{FIRST_SERIAL - COUNT + offset:05d}"
        if int(row["id"]) != expected_id or str(row.get("name")) != expected_name:
            raise RuntimeError(
                f"old cohort identity mismatch: {row.get('id')}/{row.get('name')}"
            )
    extras = [row for row in rows if int(row.get("id") or 0) not in range(OLD_FIRST_ID, OLD_LAST_ID + 1)]
    if extras:
        raise RuntimeError(f"old exact prefix contains unexpected tasks: {[row.get('id') for row in extras]}")
    active = sorted(int(row["id"]) for row in selected if _state(row) in ACTIVE)
    terminal = sorted(int(row["id"]) for row in selected if _state(row) not in ACTIVE)
    return {"active": active, "terminal": terminal}


def _strict_harvest_evidence() -> dict:
    payload = json.loads(STRICT_STATUS_PATH.read_text(encoding="utf-8"))
    if not checkpoint_status_revision_identity_matches(
        payload, OLD_SOLVER, LIBRARY
    ):
        raise RuntimeError("harvest checkpoint is not pinned to the old solver/library")
    stamp = datetime.fromisoformat(str(payload["time"]).replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        stamp = stamp.astimezone()
    age = (datetime.now(timezone.utc) - stamp.astimezone(timezone.utc)).total_seconds()
    if age > 15 * 60:
        raise RuntimeError(f"harvest checkpoint is stale by {age:.0f}s")
    stat = DATASET_PATH.stat()
    return {
        "checkpoint_time": payload["time"],
        "checkpoint_age_seconds": age,
        "raw_rows": int(payload.get("raw_rows") or 0),
        "strict_em_rows": int(payload.get("strict_em_rows") or 0),
        "strict_full_rows": int(payload.get("strict_full_rows") or 0),
        "quarantined_rows": int(payload.get("quarantined_rows") or 0),
        "reconciliation_issues": payload.get("reconciliation_issues") or [],
        "dataset": str(DATASET_PATH.resolve()),
        "dataset_size": int(stat.st_size),
        "dataset_mtime_ns": int(stat.st_mtime_ns),
    }


def _new_journal(manifest: dict, harvest: dict, old_inventory: dict) -> dict:
    return {
        "schema_version": 1,
        "created_at": _now(),
        "updated_at": _now(),
        "manifest": str(MANIFEST_PATH.resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "harvest": harvest,
        "old_inventory_before": old_inventory,
        "cancel_request_ids": [],
        "cancelled_ids": [],
        "terminal_race_ids": [],
        "old_active_after": None,
        "submissions": {},
        "audit": None,
        "completed": False,
    }


def _save_journal(journal: dict) -> None:
    journal["updated_at"] = _now()
    _atomic_json(JOURNAL_PATH, journal)


def _load_or_create_journal(manifest: dict, harvest: dict, old_inventory: dict) -> dict:
    if JOURNAL_PATH.exists():
        journal = json.loads(JOURNAL_PATH.read_text(encoding="utf-8"))
        if journal.get("manifest_sha256") != manifest["manifest_sha256"]:
            if (
                journal.get("manifest_sha256") == SUPERSEDED_MANIFEST_SHA256
                and manifest.get("supersedes_manifest_sha256") == SUPERSEDED_MANIFEST_SHA256
            ):
                prior = journal["manifest_sha256"]
                journal["manifest_sha256"] = manifest["manifest_sha256"]
                journal["manifest_migration"] = {
                    "migrated_at": _now(),
                    "from_manifest_sha256": prior,
                    "to_manifest_sha256": manifest["manifest_sha256"],
                    "reason": (
                        "correct order-sensitive scheduler dedupe identities after "
                        "sorted-key manifest reload; parameter values and task IDs unchanged"
                    ),
                    "superseded_manifest": str(SUPERSEDED_MANIFEST_PATH.resolve()),
                }
                for record in manifest["tasks"]:
                    submission = (journal.get("submissions") or {}).get(str(record["index"]))
                    if submission:
                        submission["dedupe_key"] = record["dedupe_key"]
                _save_journal(journal)
            else:
                raise RuntimeError("replacement journal belongs to another manifest")
        return journal
    journal = _new_journal(manifest, harvest, old_inventory)
    _save_journal(journal)
    return journal


def _cancel_exact(active_ids: list[int], journal: dict) -> None:
    already = set(map(int, journal.get("cancel_request_ids") or []))
    if already and already != set(active_ids):
        # A resumed attempt may see some requested tasks already terminal.
        if not set(active_ids).issubset(already):
            raise RuntimeError("old active inventory expanded after cancellation started")
        request_ids = sorted(already)
    else:
        request_ids = list(active_ids)
        journal["cancel_request_ids"] = request_ids
        _save_journal(journal)
    cancelled = set(map(int, journal.get("cancelled_ids") or []))
    # On resume, never repeat a cancellation against IDs that have already
    # become terminal. The immutable request ledger remains the initial set.
    pending = [task_id for task_id in active_ids if task_id not in cancelled]
    for start in range(0, len(pending), 100):
        batch = pending[start:start + 100]
        response = requests.post(
            f"{SCHEDULER}/api/tasks/cancel",
            params={
                "statuses": ",".join(ACTIVE),
                "task_ids": ",".join(map(str, batch)),
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        returned = payload.get("cancelled") if isinstance(payload, dict) else None
        if not isinstance(returned, list) or not set(map(int, returned)).issubset(set(batch)):
            raise RuntimeError("scheduler returned invalid explicit cancellation IDs")
        cancelled.update(map(int, returned))
        journal["cancelled_ids"] = sorted(cancelled)
        _save_journal(journal)

    deadline = time.time() + 180
    remaining = []
    current_rows = []
    while time.time() < deadline:
        current_rows = _tasks(OLD_PREFIX)
        current = _validate_old_inventory(current_rows)
        remaining = current["active"]
        if not remaining:
            break
        time.sleep(2)
    if remaining:
        raise RuntimeError(f"old cohort cancellation did not drain: {remaining}")
    # The bulk endpoint acknowledges the request before asynchronous scheduler
    # state transitions and can return an empty list. Seal outcomes from the
    # authoritative final inventory instead of treating that empty ack as a
    # natural-terminal race.
    final_by_id = {int(row["id"]): row for row in current_rows}
    actual_cancelled = sorted(
        task_id for task_id in request_ids
        if _state(final_by_id[task_id]) == "cancelled"
    )
    journal["old_active_after"] = []
    journal["cancelled_ids"] = actual_cancelled
    journal["terminal_race_ids"] = sorted(set(request_ids) - set(actual_cancelled))
    _save_journal(journal)


def _submit(manifest: dict, journal: dict) -> None:
    profile = _profile()
    submissions = journal.setdefault("submissions", {})
    for record in manifest["tasks"]:
        key = str(record["index"])
        existing = submissions.get(key)
        if existing and existing.get("task_id"):
            continue
        task_id = scheduler_client.submit_verification(
            name=record["name"],
            workdir=record["workdir"],
            params=record["params"],
            profile=profile,
            mem_mb=MEMORY_MB,
            cpus=CPUS,
            solver_revision=NEW_SOLVER,
            library_revision=LIBRARY,
        )
        if not isinstance(task_id, int) or task_id <= 0:
            raise RuntimeError(f"submission returned no durable task ID: {record['name']}")
        submissions[key] = {
            "index": record["index"],
            "name": record["name"],
            "dedupe_key": record["dedupe_key"],
            "task_id": task_id,
            "accepted_or_reconciled_at": _now(),
        }
        _save_journal(journal)


def _audit(manifest: dict, journal: dict) -> dict:
    rows = _tasks(NEW_PREFIX)
    by_name = {str(row.get("name")): row for row in rows}
    if len(rows) != COUNT or len(by_name) != COUNT:
        raise RuntimeError(f"replacement inventory count mismatch: {len(rows)}")
    ids = []
    status_counts = {}
    failures = []
    for record in manifest["tasks"]:
        row = by_name.get(record["name"])
        if row is None:
            failures.append({"name": record["name"], "reason": "missing"})
            continue
        task_id = int(row.get("id") or row.get("task_id") or 0)
        ids.append(task_id)
        checks = {
            "journal_id": task_id == int(journal["submissions"][str(record["index"])]["task_id"]),
            "project": row.get("project") == scheduler_client.MFT_PROJECT,
            "dedupe": row.get("dedupe_key") == record["dedupe_key"],
            "cpus": int(row.get("cpus") or 0) == CPUS,
            "memory_mb": int(row.get("memory_mb") or 0) == MEMORY_MB,
            "gpus": int(row.get("gpus") or 0) == 0,
            "timeout": int(row.get("timeout_seconds") or 0) == TIMEOUT_SECONDS,
            "capability": row.get("required_capability") == "conda:pyaedt2026v1",
            "env": row.get("env_profile") == "pyaedt2026v1",
            "profile": row.get("scheduling_profile") == "fea_bursty",
            "priority": int(row.get("priority") or 0) == 0,
            "remote_cwd": row.get("remote_cwd") == scheduler_client.GPFS_RUNS_REMOTE_CWD,
        }
        if not all(checks.values()):
            failures.append({"id": task_id, "name": record["name"], "checks": checks})
        state = _state(row)
        status_counts[state] = status_counts.get(state, 0) + 1
    if len(ids) != len(set(ids)) or any(task_id <= 0 for task_id in ids):
        raise RuntimeError("replacement task IDs are invalid or duplicated")
    if failures:
        raise RuntimeError(f"replacement exhaustive audit failed: {failures[:3]}")
    old_after = _validate_old_inventory(_tasks(OLD_PREFIX))
    if old_after["active"]:
        raise RuntimeError(f"old active cohort reappeared: {old_after['active']}")
    with scheduler_client.campaign_mutation_lock():
        capacity = scheduler_client.live_project_submission_snapshot(
            scheduler_client.MFT_PROJECT_MAX_ACTIVE_TASKS
        )
    return {
        "audited_at": _now(),
        "task_count": len(rows),
        "first_task_id": min(ids),
        "last_task_id": max(ids),
        "unique_task_ids": len(set(ids)),
        "unique_dedupe_keys": len({row["dedupe_key"] for row in manifest["tasks"]}),
        "status_counts": status_counts,
        "old_active_after": old_after["active"],
        "project_active": capacity["project_active"],
        "project_open_slots": capacity["project_submission_slots"],
        "manifest_sha256": manifest["manifest_sha256"],
    }


def execute() -> dict:
    manifest = ensure_manifest()
    feeder._require_deployed_revisions(NEW_SOLVER, LIBRARY)
    harvest = _strict_harvest_evidence()
    with FileLock(str(JOURNAL_PATH) + ".lock", timeout=30):
        initial_old = _validate_old_inventory(_tasks(OLD_PREFIX))
        journal = _load_or_create_journal(manifest, harvest, initial_old)
        with scheduler_client.campaign_mutation_lock():
            current_old = _validate_old_inventory(_tasks(OLD_PREFIX))
            _cancel_exact(current_old["active"], journal)
            _submit(manifest, journal)
        audit = _audit(manifest, journal)
        journal["audit"] = audit
        journal["completed"] = True
        _save_journal(journal)
        return {
            "mode": "execute",
            "manifest": str(MANIFEST_PATH.resolve()),
            "journal": str(JOURNAL_PATH.resolve()),
            "cancel_requested": len(journal["cancel_request_ids"]),
            "cancelled": len(journal["cancelled_ids"]),
            "terminal_races": len(journal["terminal_race_ids"]),
            "submitted": len(journal["submissions"]),
            "audit": audit,
        }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.execute:
            result = execute()
        else:
            manifest = ensure_manifest()
            result = {
                "mode": "plan",
                "manifest": str(MANIFEST_PATH.resolve()),
                "manifest_sha256": manifest["manifest_sha256"],
                "task_count": len(manifest["tasks"]),
                "cursor_start": manifest["candidate_cursor_start"],
                "cursor_end": manifest["candidate_cursor_end"],
                "first": {
                    key: manifest["tasks"][0][key]
                    for key in ("name", "candidate_raw_index", "params_sha256", "dedupe_key")
                },
                "last": {
                    key: manifest["tasks"][-1][key]
                    for key in ("name", "candidate_raw_index", "params_sha256", "dedupe_key")
                },
            }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({
            "mode": "execute" if args.execute else "plan",
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
