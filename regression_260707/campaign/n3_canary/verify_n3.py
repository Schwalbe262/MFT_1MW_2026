"""Poll and verify one N=3 AEDT-sharing canary run."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import requests


DEFAULT_SCHEDULER_URL = "http://127.0.0.1:8000"
CANARY_BUNDLE_ID = "mft-aedt-n3canary-260714"
TERMINAL = {"completed", "failed", "cancelled", "timeout"}
ALIASES = {"canceled": "cancelled", "timed_out": "timeout", "succeeded": "completed"}


def _status(task: Mapping[str, object]) -> str:
    value = str(task.get("status") or task.get("state") or "").strip().lower()
    return ALIASES.get(value, value)


def _parse_time(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat().replace("+00:00", "Z")


def _task(session: requests.Session, scheduler_url: str, task_id: int) -> dict[str, Any]:
    response = session.get(f"{scheduler_url}/api/tasks/{task_id}", timeout=20)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"task {task_id} response is not an object")
    return payload


def _task_inventory(
    session: requests.Session, scheduler_url: str
) -> list[dict[str, Any]]:
    response = session.get(
        f"{scheduler_url}/api/tasks",
        params={"limit": 10_000, "name_prefix": f"{CANARY_BUNDLE_ID}-client-"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    rows = payload if isinstance(payload, list) else payload.get("tasks", [])
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("client task inventory is not a list of objects")
    return rows


def _client_report(task_id: int, task: Mapping[str, object]) -> dict[str, object]:
    started = _parse_time(task.get("started_at"))
    finished = _parse_time(task.get("finished_at"))
    wall_seconds = None
    if started is not None and finished is not None:
        wall_seconds = max(0.0, round((finished - started).total_seconds(), 3))
    return {
        "finished_at": _iso(finished),
        "name": str(task.get("name") or ""),
        "started_at": _iso(started),
        "status": _status(task),
        "task_id": task_id,
        "wall_time_seconds": wall_seconds,
    }


def _emit_and_return(payload: Mapping[str, object], code: int) -> int:
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
    return code


def _parse_clients(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("clients must be comma-separated integers") from exc
    if len(result) != 3 or len(set(result)) != 3 or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("--clients requires three distinct positive IDs")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, type=int)
    parser.add_argument("--clients", required=True, type=_parse_clients)
    parser.add_argument("--scheduler-url", default=DEFAULT_SCHEDULER_URL)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--timeout-seconds", type=float, default=18_000.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.host <= 0:
        raise SystemExit("--host must be positive")
    if args.poll_seconds <= 0 or args.timeout_seconds <= 0:
        raise SystemExit("poll and timeout durations must be positive")
    scheduler_url = args.scheduler_url.rstrip("/")
    deadline = time.monotonic() + args.timeout_seconds
    session = requests.Session()
    client_records: dict[int, dict[str, Any]] = {}
    host_record: dict[str, Any] = {}
    host_terminal_before_all_clients = False
    poll_count = 0
    last_error = ""
    timed_out = False

    while True:
        poll_count += 1
        try:
            client_records = {
                task_id: _task(session, scheduler_url, task_id)
                for task_id in args.clients
            }
            host_record = _task(session, scheduler_url, args.host)
            all_terminal = all(
                _status(task) in TERMINAL for task in client_records.values()
            )
            if _status(host_record) in TERMINAL and not all_terminal:
                host_terminal_before_all_clients = True
            if all_terminal:
                # One final read minimizes timestamp/status skew at the boundary.
                host_record = _task(session, scheduler_url, args.host)
                last_error = ""
                break
            last_error = ""
        except Exception as exc:
            last_error = str(exc)
        if time.monotonic() >= deadline:
            timed_out = True
            break
        time.sleep(min(args.poll_seconds, max(0.0, deadline - time.monotonic())))

    reports = [
        _client_report(task_id, client_records.get(task_id, {}))
        for task_id in args.clients
    ]
    statuses = [str(item["status"]) for item in reports]
    expected_names = {
        f"{CANARY_BUNDLE_ID}-client-{index}" for index in range(1, 4)
    }
    actual_names = {str(item["name"]) for item in reports}
    all_completed = len(reports) == 3 and all(status == "completed" for status in statuses)
    supplied_ids = set(args.clients)
    fallback_records: list[dict[str, object]] = []
    related_ids: set[int] = set()
    try:
        for row in _task_inventory(session, scheduler_url):
            name = str(row.get("name") or "")
            raw_id = row.get("task_id") or row.get("id")
            if isinstance(raw_id, bool) or not isinstance(raw_id, int):
                continue
            related_ids.add(raw_id)
            if name not in expected_names or raw_id not in supplied_ids:
                fallback_records.append(
                    {"name": name, "status": _status(row), "task_id": raw_id}
                )
    except Exception as exc:
        last_error = str(exc)
    no_client_fallback = (
        actual_names == expected_names
        and len(actual_names) == 3
        and related_ids == supplied_ids
        and not fallback_records
    )

    client_finished = [
        value
        for value in (_parse_time(task.get("finished_at")) for task in client_records.values())
        if value is not None
    ]
    last_client_finished = max(client_finished) if len(client_finished) == 3 else None
    host_status = _status(host_record)
    host_finished = _parse_time(host_record.get("finished_at"))
    timestamp_order_known = host_finished is not None and last_client_finished is not None
    if timestamp_order_known:
        # Scheduler timestamps are authoritative. Sequential GETs can otherwise
        # observe the host terminal just after a still-running client snapshot.
        host_stayed_alive = host_finished >= last_client_finished
    else:
        host_stayed_alive = (
            host_status not in TERMINAL and not host_terminal_before_all_clients
        )
    host_identity_ok = str(host_record.get("name") or "") == f"{CANARY_BUNDLE_ID}-host"
    colocation_links_ok = all(
        task.get("same_node_as_task_id") == args.host for task in client_records.values()
    )
    passed = (
        all_completed
        and host_stayed_alive
        and no_client_fallback
        and host_identity_ok
        and colocation_links_ok
        and not timed_out
        and not last_error
    )

    verdict = {
        "checks": {
            "all_3_completed": all_completed,
            "client_host_links_match": colocation_links_ok,
            "host_identity_matches": host_identity_ok,
            "host_stayed_alive_until_last_client_finished": host_stayed_alive,
            "no_client_fallback": no_client_fallback,
        },
        "clients": reports,
        "fallback_or_extra_client_tasks": fallback_records,
        "host": {
            "finished_at": _iso(host_finished),
            "name": str(host_record.get("name") or ""),
            "status": host_status,
            "task_id": args.host,
        },
        "last_client_finished_at": _iso(last_client_finished),
        "last_error": last_error or None,
        "pass": passed,
        "poll_count": poll_count,
        "timed_out": timed_out,
    }
    return _emit_and_return(verdict, 0 if passed else 1)


if __name__ == "__main__":
    sys.exit(main())
