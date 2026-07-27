"""Keep the corrected-physics NSGA cohort at 100 active Scheduler tasks.

The original immutable receipts remain read-only.  A separately authenticated
and staged reserve plan provides fresh consecutive seeds.  Each cycle GETs the
original and replacement task states, computes the live active deficit, and
POSTs only that many reserve tasks.  Scheduler source is imported only for its
existing account/SSH client; this controller never edits the Scheduler project
and has no cancel, preempt, or retry endpoint.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Iterator, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPOSITORY_ROOT))

from module.mft_goal_20260726_contract import canonical_sha256  # noqa: E402
from tools import mft_goal_corrected_physics_nsga_lane as lane  # noqa: E402
from tools.tier1_final1000_multiseed_contract import (  # noqa: E402
    scheduler_task_observation,
)


LEDGER_SCHEMA = "mft-goal-corrected-physics-target100-ledger-v1"
CYCLE_SCHEMA = "mft-goal-corrected-physics-target100-cycle-v1"
STATUS_SCHEMA = "mft-goal-corrected-physics-target100-status-v1"
TARGET_ACTIVE = 100
ACTIVE = frozenset({"queued", "attaching", "running"})
TERMINAL = frozenset(
    {"completed", "succeeded", "failed", "cancelled", "timeout", "timed_out"}
)
THREAD_EXPORTS = (
    "export OMP_NUM_THREADS=1",
    "export OPENBLAS_NUM_THREADS=1",
    "export MKL_NUM_THREADS=1",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(resolved),
        "sha256": digest.hexdigest(),
        "size_bytes": resolved.stat().st_size,
    }


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "payload_sha256" in result:
        raise RuntimeError("value is already sealed")
    result["payload_sha256"] = canonical_sha256(result)
    return result


def _validate_seal(value: Mapping[str, Any], schema: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    unsigned = dict(result)
    observed = unsigned.pop("payload_sha256", None)
    if (
        result.get("schema_version") != schema
        or observed != canonical_sha256(unsigned)
    ):
        raise RuntimeError(f"{schema} seal mismatch")
    return result


def _validate_sha_seal(value: Mapping[str, Any], schema: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    unsigned = dict(result)
    observed = unsigned.pop("sha256", None)
    if (
        result.get("schema_version") != schema
        or observed != canonical_sha256(unsigned)
    ):
        raise RuntimeError(f"{schema} SHA seal mismatch")
    return result


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    staged_path = Path(staged)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                value,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged_path, path)
    finally:
        staged_path.unlink(missing_ok=True)


def _immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read_json(path) != dict(value):
            raise RuntimeError(f"immutable cycle changed: {path}")
        return
    descriptor, staged = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    staged_path = Path(staged)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                value,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(staged_path, path)
        except FileExistsError:
            if _read_json(path) != dict(value):
                raise RuntimeError(f"concurrent cycle changed: {path}")
    finally:
        staged_path.unlink(missing_ok=True)


@contextmanager
def _controller_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "controller.lock"
    descriptor: int | None = None
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield
    except FileExistsError as exc:
        raise RuntimeError("corrected target100 controller already active") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
            lock.unlink(missing_ok=True)


def _receipt_entries(paths: Iterable[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    for path in paths:
        record = _file_record(path)
        value = _validate_sha_seal(
            _read_json(path), lane.OFFLOAD_RECEIPT_SCHEMA
        )
        rows = value.get("tasks")
        if (
            value.get("apply") is not True
            or not isinstance(rows, list)
            or int(value.get("task_count", -1)) != len(rows)
            or int(value.get("scheduler_post_count", -1))
            != int(value.get("submitted_count", -2))
        ):
            raise RuntimeError("base receipt accounting mismatch")
        records.append(record)
        for row in rows:
            entries.append(
                {
                    "task_id": int(row["task_id"]),
                    "seed": int(row["seed"]),
                    "name": str(row["name"]),
                    "dedupe_key": str(row["dedupe_key"]),
                    "source_receipt_sha256": record["sha256"],
                }
            )
    if (
        len(entries) != len({entry["task_id"] for entry in entries})
        or len(entries) != len({entry["seed"] for entry in entries})
    ):
        raise RuntimeError("base receipt union is duplicated")
    return records, sorted(entries, key=lambda row: row["seed"])


def _status(task: Mapping[str, Any]) -> str:
    return str(task.get("state") or task.get("status") or "unknown").lower()


def _observe(
    *,
    scheduler: Any,
    entries: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    observed: list[dict[str, Any]] = []
    for entry in entries:
        task = scheduler.get_task(int(entry["task_id"]))
        if (
            int(task.get("id", -1)) != int(entry["task_id"])
            or str(task.get("name") or "") != str(entry["name"])
            or str(task.get("dedupe_key") or "") != str(entry["dedupe_key"])
        ):
            raise RuntimeError("Scheduler GET identity differs from receipt")
        status = _status(task)
        if status not in ACTIVE | TERMINAL:
            raise RuntimeError(f"unknown Scheduler state: {status}")
        observed.append(
            {
                **dict(entry),
                "status": status,
                "account_name": str(task.get("account_name") or ""),
                "node": str(task.get("actual_node_name") or ""),
            }
        )
    return observed


def _load_or_initialize_ledger(
    *,
    root: Path,
    reserve_plan: Path,
    receipt_records: list[dict[str, Any]],
    base_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    path = root / "ledger.json"
    plan_record = _file_record(reserve_plan)
    if path.exists():
        value = _validate_seal(_read_json(path), LEDGER_SCHEMA)
        if (
            value["reserve_plan"] != plan_record
            or value["base_receipts"] != receipt_records
            or value["base_entries"] != base_entries
            or int(value["target_active_count"]) != TARGET_ACTIVE
        ):
            raise RuntimeError("corrected target100 ledger authority drifted")
        return value
    value = _seal(
        {
            "schema_version": LEDGER_SCHEMA,
            "created_at": _now(),
            "updated_at": _now(),
            "target_active_count": TARGET_ACTIVE,
            "base_receipts": receipt_records,
            "base_entries": base_entries,
            "reserve_plan": plan_record,
            "replacement_entries": [],
            "scheduler_mutations": {
                "post_endpoint": "POST /api/tasks",
                "cancel_count": 0,
                "preempt_count": 0,
                "scheduler_project_modified": False,
            },
        }
    )
    _atomic_json(path, value)
    return value


def _save_ledger(root: Path, ledger: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = copy.deepcopy(dict(ledger))
    unsigned.pop("payload_sha256", None)
    unsigned["updated_at"] = _now()
    value = _seal(unsigned)
    _atomic_json(root / "ledger.json", value)
    return value


def _submit_one(
    *,
    offload: Any,
    scheduler: Any,
    plan: Mapping[str, Any],
    tasks: list[dict[str, Any]],
    expected: Mapping[str, Mapping[str, Any]],
    inventory: dict[str, dict[str, Any]],
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    dedupe = str(payload["dedupe_key"])
    claim_state, pending, finalized = offload._acquire_task_claim(
        plan=plan,
        tasks=tasks,
        payload=payload,
    )
    row = inventory.get(dedupe)
    source = "finalized_claim"
    if claim_state == "fresh_pending":
        inventory = offload._scheduler_inventory(
            scheduler=scheduler,
            name_prefix=offload._task_name_prefix(plan),
            expected=expected,
        )
        if dedupe in inventory:
            raise RuntimeError("unclaimed reserve task appeared before POST")
        response = scheduler.submit_task(payload)
        if response.get("deduped") is not False:
            raise RuntimeError("fresh reserve POST was unexpectedly deduplicated")
        task_id = response.get("task_id", response.get("id"))
        if isinstance(task_id, bool) or not isinstance(task_id, int):
            raise RuntimeError("reserve POST lacks task id")
        row = scheduler.get_task(task_id)
        source = "submitted"
    elif claim_state == "existing_pending":
        if row is None:
            inventory = offload._scheduler_inventory(
                scheduler=scheduler,
                name_prefix=offload._task_name_prefix(plan),
                expected=expected,
            )
            row = inventory.get(dedupe)
        if row is None:
            raise RuntimeError("pending reserve claim has no Scheduler task")
        observation = scheduler_task_observation(row)
        if observation is None:
            raise RuntimeError("pending reserve task has no identity")
        row = scheduler.get_task(observation[0])
        source = "recovered_pending"
    else:
        assert finalized is not None
        row = scheduler.get_task(int(finalized["task_id"]))
    assert row is not None
    task_id, status = offload._task_authentication(
        row, payload, label=source
    )
    if finalized is None:
        finalized = offload._finalize_task_claim(
            plan=plan,
            payload=payload,
            pending=pending,
            task_id=task_id,
        )
    result = {
        "task_id": task_id,
        "seed": int(payload["payload_json"]["seed"]),
        "name": str(payload["name"]),
        "dedupe_key": dedupe,
        "status_at_submission": status,
        "source": source,
        "scheduler_payload_sha256": canonical_sha256(payload),
        "claim_finalized_sha256": finalized["sha256"],
        "native_thread_limits": {
            "OMP_NUM_THREADS": 1,
            "OPENBLAS_NUM_THREADS": 1,
            "MKL_NUM_THREADS": 1,
        },
        "scheduler_resources": {
            "cpus": int(payload["cpus"]),
            "memory_mb": int(payload["memory_mb"]),
            "max_workers_per_node": int(payload["max_workers_per_node"]),
        },
    }
    inventory[dedupe] = dict(row)
    return result, inventory


def cycle(
    *,
    reserve_plan: Path,
    base_receipts: list[Path],
    state_root: Path,
    scheduler_url: str,
    accounts: Path,
    scheduler_source: Path,
    staging_account: str,
) -> dict[str, Any]:
    lane._configure_from_plan(reserve_plan)
    offload = lane._coordinator_offload()
    plan, _deployment, tasks, authentication = offload.authenticate_plan(
        reserve_plan
    )
    ready = offload._remote_ready_authenticated(
        plan=plan,
        accounts_path=accounts,
        scheduler_source=scheduler_source,
        staging_account=staging_account,
    )
    receipt_records, base_entries = _receipt_entries(base_receipts)
    ledger = _load_or_initialize_ledger(
        root=state_root,
        reserve_plan=reserve_plan,
        receipt_records=receipt_records,
        base_entries=base_entries,
    )
    scheduler = offload.SchedulerClient(scheduler_url)
    replacement_entries = list(ledger["replacement_entries"])
    all_entries = [*base_entries, *replacement_entries]
    observed_before = _observe(scheduler=scheduler, entries=all_entries)
    active_before = sum(row["status"] in ACTIVE for row in observed_before)
    deficit = max(0, TARGET_ACTIVE - active_before)

    expected = {
        payload["dedupe_key"]: payload
        for payload in (
            offload.scheduler_payload(
                plan=plan,
                task=task,
                priority=lane.SCHEDULER_PRIORITY,
            )
            for task in tasks
        )
    }
    for payload in expected.values():
        command = str(payload["command"])
        if (
            any(command.count(export) != 1 for export in THREAD_EXPORTS)
            or int(payload.get("cpus", -1)) != 8
            or int(payload.get("memory_mb", -1)) != 65_536
        ):
            raise RuntimeError("reserve payload thread/resource contract drifted")
    inventory = offload._scheduler_inventory(
        scheduler=scheduler,
        name_prefix=offload._task_name_prefix(plan),
        expected=expected,
    )
    submitted: list[dict[str, Any]] = []
    observed_after = observed_before
    while True:
        active_current = sum(
            row["status"] in ACTIVE for row in observed_after
        )
        current_deficit = max(0, TARGET_ACTIVE - active_current)
        if current_deficit == 0:
            break
        used_seeds = {int(row["seed"]) for row in replacement_entries}
        available = sorted(
            (
                payload
                for payload in expected.values()
                if int(payload["payload_json"]["seed"]) not in used_seeds
            ),
            key=lambda payload: int(payload["payload_json"]["seed"]),
        )
        if current_deficit > len(available):
            raise RuntimeError(
                "reserve exhausted: "
                f"deficit={current_deficit}, available={len(available)}"
            )
        for payload in available[:current_deficit]:
            row, inventory = _submit_one(
                offload=offload,
                scheduler=scheduler,
                plan=plan,
                tasks=tasks,
                expected=expected,
                inventory=inventory,
                payload=payload,
            )
            replacement_entries.append(row)
            submitted.append(row)
            ledger = dict(ledger)
            ledger["replacement_entries"] = replacement_entries
            ledger = _save_ledger(state_root, ledger)
        # Original tasks can finish while a large refill batch is being
        # submitted.  Re-observe and top up that newly opened deficit in the
        # same cycle instead of claiming the first snapshot still equals 100.
        observed_after = _observe(
            scheduler=scheduler,
            entries=[*base_entries, *replacement_entries],
        )

    active_after = sum(row["status"] in ACTIVE for row in observed_after)
    counts: dict[str, int] = {}
    for row in observed_after:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    if active_after != TARGET_ACTIVE:
        raise RuntimeError(
            f"target100 postcondition failed: active_after={active_after}"
        )
    cycle_index = len(list((state_root / "cycles").glob("cycle-*.json")))
    value = _seal(
        {
            "schema_version": CYCLE_SCHEMA,
            "cycle_index": cycle_index,
            "observed_at": _now(),
            "target_active_count": TARGET_ACTIVE,
            "active_before": active_before,
            "deficit_before_post": deficit,
            "submitted_count": len(submitted),
            "submitted": submitted,
            "active_after": active_after,
            "status_counts_after": dict(sorted(counts.items())),
            "base_task_count": len(base_entries),
            "replacement_task_count": len(replacement_entries),
            "reserve_remaining": len(expected) - len(replacement_entries),
            "authentication_sha256": authentication["sha256"],
            "reserve_ready": ready,
            "ledger_payload_sha256": ledger["payload_sha256"],
            "scheduler_get_count": scheduler.get_count,
            "scheduler_post_count": scheduler.post_count,
            "scheduler_cancel_count": 0,
            "scheduler_preempt_count": 0,
            "scheduler_project_modified": False,
        }
    )
    cycle_path = state_root / "cycles" / f"cycle-{cycle_index:08d}.json"
    _immutable_json(cycle_path, value)
    status = _seal(
        {
            "schema_version": STATUS_SCHEMA,
            "updated_at": _now(),
            "controller_pid": os.getpid(),
            "target_active_count": TARGET_ACTIVE,
            "active_count": active_after,
            "status_counts": dict(sorted(counts.items())),
            "replacement_task_count": len(replacement_entries),
            "reserve_remaining": len(expected) - len(replacement_entries),
            "last_cycle": _file_record(cycle_path),
            "watch_active": True,
            "scheduler_project_modified": False,
        }
    )
    _atomic_json(state_root / "status.json", status)
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reserve-plan", type=Path, required=True)
    parser.add_argument("--base-receipt", type=Path, action="append", required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--scheduler-url", default=lane.DEFAULT_SCHEDULER_URL)
    parser.add_argument("--accounts", type=Path, default=lane.DEFAULT_ACCOUNTS)
    parser.add_argument(
        "--scheduler-source", type=Path, default=lane.DEFAULT_SCHEDULER_SOURCE
    )
    parser.add_argument(
        "--staging-account", default=lane.DEFAULT_STAGING_ACCOUNT
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-seconds", type=float, default=15.0)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if len(args.base_receipt) < 1:
        raise RuntimeError("at least one base receipt is required")
    with _controller_lock(args.state_root):
        while True:
            value = cycle(
                reserve_plan=args.reserve_plan,
                base_receipts=args.base_receipt,
                state_root=args.state_root,
                scheduler_url=args.scheduler_url,
                accounts=args.accounts,
                scheduler_source=args.scheduler_source,
                staging_account=args.staging_account,
            )
            print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)
            if not args.watch:
                return 0
            time.sleep(max(5.0, float(args.interval_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
