"""Cancel only never-started cycle-335 work to restore the MFT pool to 300.

The command is audit-only unless ``--execute`` is supplied.  Every mutation
is protected by the shared campaign lock, preceded by a sealed durable batch
record, and uses the scheduler's queued-only compare-and-set cancellation.
Running, attaching, allocated, or otherwise unverified tasks are never
eligible.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests


HERE = Path(__file__).resolve().parent
REGRESSION_ROOT = HERE.parent
VERIFY_ROOT = REGRESSION_ROOT / "verify"
for item in (HERE, REGRESSION_ROOT, VERIFY_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import _adopted_refill_sha688c6f9 as durable
import scheduler_client


SCHEDULER = scheduler_client.SCHEDULER
PROJECT = scheduler_client.MFT_PROJECT
TARGET_ACTIVE = 300
PROJECT_CAP = 300
SOLVER = "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c"
LIBRARY = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
PLAN_SHA256 = "b24e2a9b00caa22bbec8793f4dbd99de51362fac87f9e9509358610abe9982d0"
CYCLE_SERIAL = 335
CYCLE_PATH = (
    HERE / "pilot_manifests" / "continuous-refill-sb171c7c-le6b9b9d"
    / f"cycle-{CYCLE_SERIAL:06d}.json"
)
AUDIT_PATH = (
    HERE / "pilot_manifests"
    / "target-rollback-pool400-to300-20260713.json"
)
PREFIX = f"mft-camp-s{SOLVER[:7]}-l{LIBRARY[:7]}-"
ACTIVE_STATUSES = ("queued", "attaching", "running")
NULL_TEXT = (None, "")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha(value) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rows(response: requests.Response, label: str) -> list[dict]:
    response.raise_for_status()
    payload = response.json()
    rows = payload if isinstance(payload, list) else (
        payload.get("tasks") if isinstance(payload, dict) else None
    )
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError(f"scheduler returned invalid {label} rows")
    return rows


def _get(path: str, *, params=None, timeout: int = 30):
    last_error = None
    for _ in range(4):
        try:
            response = requests.get(
                f"{SCHEDULER}{path}", params=params, timeout=timeout
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
    raise RuntimeError(f"scheduler GET failed for {path}: {last_error}")


def _active_inventory() -> list[dict]:
    response = requests.get(
        f"{SCHEDULER}/api/tasks",
        params={
            "limit": 10_000,
            "project": PROJECT,
            "status": ",".join(ACTIVE_STATUSES),
        },
        timeout=30,
    )
    return _rows(response, "active project")


def _task(task_id: int) -> dict:
    payload = _get(f"/api/tasks/{int(task_id)}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"task {task_id} detail is invalid")
    return payload


def _cycle_validator(payload: dict) -> dict:
    if (
        not isinstance(payload, dict)
        or payload.get("cycle_serial") != CYCLE_SERIAL
        or payload.get("plan_sha256") != PLAN_SHA256
        or payload.get("target_active") != 400
    ):
        raise RuntimeError("cycle 335 durable identity is invalid")
    journal = payload.get("formal_journal")
    events = journal.get("events") if isinstance(journal, dict) else None
    if not isinstance(events, list):
        raise RuntimeError("cycle 335 journal is invalid")
    return payload


def _cycle_identities() -> dict[int, dict]:
    cycle = durable._authoritative_state(
        CYCLE_PATH, _cycle_validator, repair=False
    )
    if cycle is None:
        raise RuntimeError("cycle 335 durable history is missing")
    journal = cycle["formal_journal"]
    if (
        cycle.get("status") != "completed"
        or len(journal["events"]) != 320
        or journal.get("submitted_count") != 320
        or journal.get("completed") is not True
    ):
        raise RuntimeError("cycle 335 is not a complete 320-task batch")
    identities = {}
    for event in journal["events"]:
        task_id = event.get("task_id")
        if (
            isinstance(task_id, bool)
            or not isinstance(task_id, int)
            or task_id <= 0
            or task_id in identities
            or event.get("accepted_or_reconciled") is not True
            or event.get("ledger_committed") is not True
        ):
            raise RuntimeError("cycle 335 contains an uncommitted/duplicate task")
        identities[task_id] = {
            "id": task_id,
            "name": event.get("name"),
            "dedupe_key": event.get("dedupe_key"),
        }
    return identities


def _never_started_queued(row: dict, identities: dict[int, dict]) -> bool:
    task_id = row.get("id", row.get("task_id"))
    expected = identities.get(task_id)
    if expected is None:
        return False
    checks = (
        row.get("id", row.get("task_id")) == expected["id"],
        row.get("name") == expected["name"],
        row.get("dedupe_key") == expected["dedupe_key"],
        str(row.get("name") or "").startswith(PREFIX),
        f":{SOLVER}:{LIBRARY}:" in str(row.get("dedupe_key") or ""),
        row.get("project") == PROJECT,
        row.get("status") == "queued",
        row.get("attached_at") in NULL_TEXT,
        row.get("launch_started_at") in NULL_TEXT,
        row.get("started_at") in NULL_TEXT,
        row.get("finished_at") in NULL_TEXT,
        row.get("allocation_id") is None,
        row.get("assigned_allocation") is None,
        row.get("slurm_job_id") in NULL_TEXT,
        row.get("allocation_node_name") in NULL_TEXT,
        row.get("account_name") in NULL_TEXT,
        row.get("requested_account_name") in NULL_TEXT,
        row.get("exit_code") is None,
        row.get("failure_message") in NULL_TEXT,
        row.get("cpus") == 4,
        row.get("memory_mb") == 65_536,
        row.get("timeout_seconds") == 14_400,
        row.get("scheduling_profile") == "fea_bursty",
        row.get("required_capability") == "conda:pyaedt2026v1",
        row.get("env_profile") == "pyaedt2026v1",
        row.get("gpus") == 0,
    )
    return all(checks)


def _identity(row: dict) -> dict:
    return {
        key: row.get(key)
        for key in (
            "id", "name", "dedupe_key", "project", "status", "created_at",
            "attached_at", "launch_started_at", "started_at", "finished_at",
            "allocation_id", "assigned_allocation", "slurm_job_id",
            "allocation_node_name", "account_name", "requested_account_name",
            "exit_code", "failure_message", "cpus", "memory_mb",
            "timeout_seconds", "scheduling_profile", "required_capability",
            "env_profile", "gpus",
        )
    }


def _audit_validator(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise RuntimeError("rollback audit is not an object")
    durable._state_revision(payload, "pool400-to300 rollback")
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_type") != "mft_pool_target_rollback"
        or payload.get("project") != PROJECT
        or payload.get("from_target") != 400
        or payload.get("to_target") != TARGET_ACTIVE
        or payload.get("cycle_serial") != CYCLE_SERIAL
        or payload.get("plan_sha256") != PLAN_SHA256
        or payload.get("status") not in {
            "prepared", "cancelling", "completed", "failed_closed"
        }
        or not isinstance(payload.get("batches"), list)
    ):
        raise RuntimeError("rollback audit identity is invalid")
    return payload


def _save_audit(state: dict) -> None:
    durable._save_durable_state(AUDIT_PATH, state, _audit_validator)


def _initial_audit(active: list[dict], candidates: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "state_revision": 0,
        "artifact_type": "mft_pool_target_rollback",
        "project": PROJECT,
        "from_target": 400,
        "to_target": TARGET_ACTIVE,
        "project_cap": PROJECT_CAP,
        "cycle_serial": CYCLE_SERIAL,
        "plan_sha256": PLAN_SHA256,
        "created_at": _now(),
        "updated_at": _now(),
        "status": "prepared",
        "initial_active": len(active),
        "initial_statuses": dict(sorted(Counter(
            str(row.get("status") or "") for row in active
        ).items())),
        "eligible_snapshot": [_identity(row) for row in candidates],
        "eligible_snapshot_sha256": _sha([
            _identity(row) for row in candidates
        ]),
        "batches": [],
        "cancelled_ids": [],
        "cancelled_identity_sha256": None,
        "final_active": None,
        "final_statuses": None,
        "error": None,
    }


def _verify_project_contract() -> dict:
    project = _get(f"/api/projects/{PROJECT}")
    if (
        not isinstance(project, dict)
        or project.get("max_active_tasks") != PROJECT_CAP
        or len(project.get("repos") or []) != 2
        or len(project.get("entrypoints") or []) != 2
        or not str(project.get("setup") or "").strip()
        or project.get("cleanup_globs") != "*.aedtresults"
        or project.get("auto_pull") is not False
    ):
        raise RuntimeError("live MFT project cap/config contract is invalid")
    return project


def _existing_audit_history() -> bool:
    return bool(
        AUDIT_PATH.exists()
        or list(AUDIT_PATH.parent.glob(f"{AUDIT_PATH.name}.gen-*.json"))
        or durable._recovery_artifact_paths(AUDIT_PATH)
    )


def run(*, execute: bool) -> dict:
    identities = _cycle_identities()
    _verify_project_contract()
    active = _active_inventory()
    candidates = sorted(
        (row for row in active if _never_started_queued(row, identities)),
        key=lambda row: int(row["id"]),
        reverse=True,
    )
    audit = {
        "mode": "audit_only",
        "active": len(active),
        "excess": max(0, len(active) - TARGET_ACTIVE),
        "eligible_queued": len(candidates),
        "eligible_ids_desc": [int(row["id"]) for row in candidates],
        "eligible_identity_sha256": _sha([_identity(row) for row in candidates]),
        "project_cap": PROJECT_CAP,
    }
    if not execute:
        return audit
    if _existing_audit_history():
        raise RuntimeError(f"rollback audit history already exists: {AUDIT_PATH}")

    state = durable._initialize_durable_state(
        AUDIT_PATH, _initial_audit(active, candidates), _audit_validator
    )
    try:
        while True:
            active = _active_inventory()
            excess = len(active) - TARGET_ACTIVE
            if excess <= 0:
                break
            candidates = sorted(
                (row for row in active if _never_started_queued(row, identities)),
                key=lambda row: int(row["id"]),
                reverse=True,
            )
            if len(candidates) < excess:
                raise RuntimeError(
                    f"only {len(candidates)} safe queued tasks for excess {excess}"
                )
            selected = candidates[:excess]
            batch = {
                "batch": len(state["batches"]) + 1,
                "prepared_at": _now(),
                "active_before": len(active),
                "excess_before": excess,
                "requested": [_identity(row) for row in selected],
                "requested_sha256": _sha([_identity(row) for row in selected]),
                "cancelled_ids": [],
                "cancelled_readback": [],
                "skipped_ids": [],
                "verified_at": None,
            }
            state["batches"].append(batch)
            state["status"] = "cancelling"
            state["updated_at"] = _now()
            _save_audit(state)

            task_ids = [int(row["id"]) for row in selected]
            response = requests.post(
                f"{SCHEDULER}/api/tasks/cancel",
                params={
                    "task_ids": ",".join(map(str, task_ids)),
                    "statuses": "queued",
                },
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()
            cancelled = {
                int(task_id) for task_id in result.get("cancelled", [])
            }
            if int(result.get("count", -1)) != len(cancelled):
                raise RuntimeError("scheduler cancellation count is inconsistent")

            readbacks = []
            skipped = []
            for task_id in task_ids:
                row = _task(task_id)
                if task_id in cancelled:
                    expected = identities[task_id]
                    if not (
                        row.get("status") == "cancelled"
                        and row.get("name") == expected["name"]
                        and row.get("dedupe_key") == expected["dedupe_key"]
                        and row.get("attached_at") in NULL_TEXT
                        and row.get("launch_started_at") in NULL_TEXT
                        and row.get("started_at") in NULL_TEXT
                        and row.get("allocation_id") is None
                        and row.get("slurm_job_id") in NULL_TEXT
                        and row.get("exit_code") is None
                        and row.get("failure_message") in NULL_TEXT
                    ):
                        raise RuntimeError(
                            f"cancelled task {task_id} lost never-started identity"
                        )
                    readbacks.append(_identity(row))
                else:
                    if row.get("status") == "queued":
                        raise RuntimeError(
                            f"queued-only cancel unexpectedly skipped queued task {task_id}"
                        )
                    skipped.append(task_id)
            batch["cancelled_ids"] = sorted(cancelled)
            batch["cancelled_readback"] = readbacks
            batch["skipped_ids"] = sorted(skipped)
            batch["verified_at"] = _now()
            state["cancelled_ids"] = sorted({
                *state["cancelled_ids"], *cancelled,
            })
            state["updated_at"] = _now()
            _save_audit(state)

        final_active = _active_inventory()
        final_counts = Counter(
            str(row.get("status") or "") for row in final_active
        )
        cancelled_readbacks = [
            _identity(_task(task_id)) for task_id in state["cancelled_ids"]
        ]
        state["cancelled_identity_sha256"] = _sha(cancelled_readbacks)
        state["final_active"] = len(final_active)
        state["final_statuses"] = dict(sorted(final_counts.items()))
        state["status"] = "completed"
        state["updated_at"] = _now()
        _save_audit(state)
        file_sha = hashlib.sha256(AUDIT_PATH.read_bytes()).hexdigest()
        return {
            "mode": "execute",
            "artifact": str(AUDIT_PATH.resolve()),
            "artifact_sha256": file_sha,
            "state_revision": state["state_revision"],
            "cancelled_count": len(state["cancelled_ids"]),
            "cancelled_ids": state["cancelled_ids"],
            "cancelled_identity_sha256": state["cancelled_identity_sha256"],
            "final_active": state["final_active"],
            "final_statuses": state["final_statuses"],
        }
    except BaseException as exc:
        state["status"] = "failed_closed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["updated_at"] = _now()
        try:
            _save_audit(state)
        except BaseException as audit_exc:
            raise RuntimeError(
                f"rollback failed and audit save failed: original={exc}; "
                f"audit={audit_exc}"
            ) from exc
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    with scheduler_client.campaign_mutation_lock(timeout=15 * 60):
        result = run(execute=args.execute)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
