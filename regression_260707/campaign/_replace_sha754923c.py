"""Authorized exact replacement of the sealed SHA688 fleet with SHA754923c.

This thin specialization reuses the already-audited resumable replacement
engine while replacing every cohort/cursor/revision identity.  It deliberately
retains the current SHA688 manifest as the superseded evidence seal.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import requests

import _replace_sha688c6f9 as engine


OLD_SOLVER = "688c6f9ae8b1368d2b4424e42fc8973b3c580d24"
NEW_SOLVER = "754923cf1c97bc45bcd9d8c6ba60d98773a5c30a"
LIBRARY = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
CURRENT_MANIFEST_SHA256 = (
    "10b9524fd2b21368fb29b63eac3c9ab2bb5efe5b99dd5e89bbd05cf8eb9c2c57"
)
CURSOR_START = 1843
CURSOR_END = 2795
FIRST_RAW_INDEX = 1845
LAST_RAW_INDEX = 2794
COUNT = 250
FIRST_SERIAL = 17362
OLD_FIRST_ID = 27471
OLD_LAST_ID = 27720


engine.OLD_SOLVER = OLD_SOLVER
engine.NEW_SOLVER = NEW_SOLVER
engine.LIBRARY = LIBRARY
engine.SEED = 260710
engine.CURSOR_START = CURSOR_START
engine.COUNT = COUNT
engine.FIRST_SERIAL = FIRST_SERIAL
engine.OLD_FIRST_ID = OLD_FIRST_ID
engine.OLD_LAST_ID = OLD_LAST_ID
engine.OLD_PREFIX = f"mft-camp-s{OLD_SOLVER[:7]}-l{LIBRARY[:7]}-"
engine.NEW_PREFIX = f"mft-camp-s{NEW_SOLVER[:7]}-l{LIBRARY[:7]}-"
engine.STEM = (
    f"replacement-s{NEW_SOLVER[:7]}-l{LIBRARY[:7]}-"
    f"seed{engine.SEED}-cursor{CURSOR_START}"
)
engine.MANIFEST_PATH = engine.EVIDENCE_ROOT / f"{engine.STEM}.json"
engine.JOURNAL_PATH = engine.EVIDENCE_ROOT / f"{engine.STEM}.journal.json"
engine.SUPERSEDED_MANIFEST_SHA256 = CURRENT_MANIFEST_SHA256
engine.SUPERSEDED_MANIFEST_PATH = engine.EVIDENCE_ROOT / (
    "replacement-s688c6f9-le6b9b9d-seed260710-cursor939.json"
)


_next_valid_candidate = engine.pinned_pilot.next_valid_candidate


def _guarded_next_valid_candidate(cursor=0, seed=260710, max_attempts=1000):
    next_cursor, raw_index, params = _next_valid_candidate(
        cursor, seed=seed, max_attempts=max_attempts)
    primary_turns = int(params["N1_main"]) + int(params["N1_side"])
    cw1 = float(params["cw1"])
    if not math.isfinite(cw1) or cw1 > 10.0 or primary_turns > 8:
        raise RuntimeError(
            f"replacement candidate primary cap mismatch: {raw_index}/"
            f"cw1={cw1}/turns={primary_turns}"
        )
    for key in ("wcp_t", "core_plate_t"):
        value = float(params[key])
        if not math.isfinite(value) or not 10.0 <= value <= 30.0:
            raise RuntimeError(
                f"replacement candidate {key} outside [10,30]: {raw_index}/{value}"
            )
    if tuple(float(params[key]) for key in ("wcp_pad_t", "core_plate_pad_t")) != (
            2.0, 2.0):
        raise RuntimeError(f"replacement pad thickness drifted: {raw_index}")
    return next_cursor, raw_index, params


engine.pinned_pilot.next_valid_candidate = _guarded_next_valid_candidate
_base_manifest_payload = engine._manifest_payload


def _manifest_payload():
    payload = _base_manifest_payload()
    payload.pop("manifest_sha256", None)
    payload["authorization"] = (
        "replace exact active SHA688 cohort with exact 250 SHA754923c tasks"
    )
    payload["supersedes_manifest_sha256"] = CURRENT_MANIFEST_SHA256
    tasks = payload["tasks"]
    checks = {
        "cursor_end": payload["candidate_cursor_end"] == CURSOR_END,
        "first_raw": tasks[0]["candidate_raw_index"] == FIRST_RAW_INDEX,
        "last_raw": tasks[-1]["candidate_raw_index"] == LAST_RAW_INDEX,
        "last_serial": payload["last_serial"] == FIRST_SERIAL + COUNT - 1,
        "old_prefix": payload["old_prefix"] == engine.OLD_PREFIX,
        "new_prefix": payload["task_prefix"] == engine.NEW_PREFIX,
    }
    if not all(checks.values()):
        raise RuntimeError(f"SHA754 replacement deterministic seal mismatch: {checks}")
    payload["manifest_sha256"] = engine._sha(payload)
    return payload


engine._manifest_payload = _manifest_payload


def _cancel_exact_bounded(active_ids, journal):
    """Resume cancellation from authoritative state in small exact-ID chunks."""
    ledger = set(map(int, journal.get("cancel_request_ids") or []))
    if ledger:
        if not set(active_ids).issubset(ledger):
            raise RuntimeError("old active inventory expanded beyond cancellation ledger")
    else:
        ledger = set(map(int, active_ids))
        journal["cancel_request_ids"] = sorted(ledger)
        engine._save_journal(journal)
    if not ledger:
        raise RuntimeError("authorized old cancellation ledger is unexpectedly empty")

    no_progress = 0
    deadline = time.time() + 8 * 60
    final_rows = None
    while time.time() < deadline:
        final_rows = engine._tasks(engine.OLD_PREFIX)
        current = engine._validate_old_inventory(final_rows)
        remaining = sorted(set(current["active"]) & ledger)
        outside = sorted(set(current["active"]) - ledger)
        if outside:
            raise RuntimeError(
                f"old active inventory escaped cancellation ledger: {outside}")
        if not remaining:
            break
        batch = remaining[:25]
        event = {
            "attempted_at": engine._now(),
            "task_ids": batch,
            "active_before": len(remaining),
            "http_status": None,
            "acknowledged_ids": [],
            "exception": None,
        }
        try:
            response = requests.post(
                f"{engine.SCHEDULER}/api/tasks/cancel",
                params={
                    "statuses": ",".join(engine.ACTIVE),
                    "task_ids": ",".join(map(str, batch)),
                },
                timeout=60,
            )
            event["http_status"] = int(response.status_code)
            response.raise_for_status()
            payload = response.json()
            acknowledged = payload.get("cancelled") if isinstance(payload, dict) else None
            if (not isinstance(acknowledged, list)
                    or not set(map(int, acknowledged)).issubset(set(batch))):
                raise RuntimeError("scheduler returned invalid bounded cancellation IDs")
            event["acknowledged_ids"] = sorted(map(int, acknowledged))
        except requests.RequestException as exc:
            # A lost response is not a failed cancellation. Reconcile only from
            # the authoritative inventory before deciding whether to retry.
            event["exception"] = f"{type(exc).__name__}: {exc}"
        journal.setdefault("cancel_attempts", []).append(event)
        engine._save_journal(journal)

        progressed = False
        for _ in range(15):
            time.sleep(2)
            final_rows = engine._tasks(engine.OLD_PREFIX)
            after = engine._validate_old_inventory(final_rows)
            if any(task_id not in set(after["active"]) for task_id in batch):
                progressed = True
                break
        no_progress = 0 if progressed else no_progress + 1
        if no_progress >= 3:
            raise RuntimeError(
                f"bounded exact cancellation made no progress for batch {batch}")
    else:
        raise RuntimeError("old exact cohort cancellation exceeded bounded deadline")

    if final_rows is None:
        final_rows = engine._tasks(engine.OLD_PREFIX)
    final = engine._validate_old_inventory(final_rows)
    if final["active"]:
        raise RuntimeError(f"old exact cohort did not drain: {final['active']}")
    by_id = {int(row["id"]): row for row in final_rows}
    actual_cancelled = sorted(
        task_id for task_id in ledger
        if engine._state(by_id[task_id]) == "cancelled"
    )
    journal["old_active_after"] = []
    journal["cancelled_ids"] = actual_cancelled
    journal["terminal_race_ids"] = sorted(ledger - set(actual_cancelled))
    engine._save_journal(journal)


engine._cancel_exact = _cancel_exact_bounded


if __name__ == "__main__":
    raise SystemExit(engine.main())
