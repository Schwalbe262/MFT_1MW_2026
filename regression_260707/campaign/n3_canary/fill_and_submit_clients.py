"""Fill discovery placeholders and submit the three N=3 canary clients."""

from __future__ import annotations

import argparse
import ipaddress
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import requests


DEFAULT_SCHEDULER_URL = "http://127.0.0.1:8000"
CANARY_BUNDLE_ID = "mft-aedt-n3canary-260714"
EXPECTED_PROJECTS = 3
DISCOVERY_PREFIX = "NODE_CANARY_DISCOVERY "
HOST_TASK_PLACEHOLDER = "{HOST_TASK_ID}"
SCHEDULER_URL_PLACEHOLDER = "{SCHEDULER_URL}"
HOST_CLONE_ROOT_PLACEHOLDER = "{HOST_CLONE_ROOT}"
DEFAULT_HOST_CLONE_ROOT = f"~/slurm_scheduler/runs/{CANARY_BUNDLE_ID}-host"


def _read_discovery(value: str | None) -> dict[str, Any]:
    if value is None or value == "-":
        text = sys.stdin.read()
    else:
        try:
            candidate = Path(value)
            text = candidate.read_text(encoding="utf-8") if candidate.is_file() else value
        except OSError:
            text = value
    for line in reversed(text.splitlines()):
        if line.startswith(DISCOVERY_PREFIX):
            text = line[len(DISCOVERY_PREFIX) :]
            break
    payload = json.loads(text.strip())
    if not isinstance(payload, dict):
        raise ValueError("discovery JSON must be an object")
    return payload


def _loopback_scheduler_url(value: object) -> str:
    url = str(value or "").strip().rstrip("/")
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("discovery scheduler_url must be an HTTP(S) origin")
    if parsed.hostname.lower() == "localhost":
        is_loopback = True
    else:
        try:
            is_loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            is_loopback = False
    if not is_loopback:
        raise ValueError("discovery scheduler_url must use a loopback address")
    return url


def _validate_discovery(payload: Mapping[str, object]) -> tuple[str, str]:
    checks = {
        "schema_version": payload.get("schema_version") == 1,
        "mode": payload.get("mode") == "scheduler_managed_node_local_canary",
        "expected_projects": payload.get("expected_projects") == EXPECTED_PROJECTS,
        "node": bool(str(payload.get("node") or "").strip()),
        "rollback_file": str(payload.get("rollback_file") or "").startswith("/tmp/"),
    }
    if not all(checks.values()):
        raise ValueError(f"discovery contract failed: {checks}")
    scheduler_url = _loopback_scheduler_url(payload.get("scheduler_url"))
    # This is persisted deterministically by the controller, not supplied by
    # the node-local discovery record. Never splice discovery-controlled text
    # into the already shell-quoted command template.
    return scheduler_url, DEFAULT_HOST_CLONE_ROOT


def _replace(value: object, replacements: Mapping[str, object]) -> object:
    if isinstance(value, dict):
        return {key: _replace(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace(item, replacements) for item in value]
    if isinstance(value, str):
        if value in replacements:
            return replacements[value]
        result = value
        for placeholder, replacement in replacements.items():
            result = result.replace(placeholder, str(replacement))
        return result
    return value


def _submit(url: str, payload: Mapping[str, object]) -> int:
    response = requests.post(url, json=dict(payload), timeout=20)
    if response.status_code not in {200, 201}:
        raise RuntimeError(
            f"HTTP {response.status_code} for {payload.get('name')}: {response.text[:1000]}"
        )
    result = response.json()
    task_id = result.get("task_id") or result.get("id")
    if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
        raise RuntimeError(f"scheduler accepted {payload.get('name')} without a task ID")
    return task_id


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-task-id", required=True, type=int)
    parser.add_argument(
        "--discovery-json",
        help="JSON text or file path; omit (or use '-') to read JSON/host stdout from stdin",
    )
    parser.add_argument("--payload-dir", type=Path, default=Path("payloads"))
    parser.add_argument("--scheduler-url", default=DEFAULT_SCHEDULER_URL)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.host_task_id <= 0:
        raise SystemExit("--host-task-id must be positive")
    discovery = _read_discovery(args.discovery_json)
    scheduler_url, clone_root = _validate_discovery(discovery)
    replacements: dict[str, object] = {
        HOST_TASK_PLACEHOLDER: args.host_task_id,
        SCHEDULER_URL_PLACEHOLDER: scheduler_url,
        HOST_CLONE_ROOT_PLACEHOLDER: clone_root,
    }
    central_url = f"{args.scheduler_url.rstrip('/')}/api/tasks"
    payloads: list[dict[str, object]] = []
    for index in range(1, EXPECTED_PROJECTS + 1):
        path = args.payload_dir / f"client_payload_{index}.json"
        template = json.loads(path.read_text(encoding="utf-8"))
        payload = _replace(template, replacements)
        if not isinstance(payload, dict):
            raise ValueError(f"client template is not an object: {path}")
        expected_name = f"{CANARY_BUNDLE_ID}-client-{index}"
        if payload.get("name") != expected_name:
            raise ValueError(f"unexpected client name in {path}: {payload.get('name')!r}")
        serialized = json.dumps(payload, sort_keys=True)
        leftovers = [key for key in replacements if key in serialized]
        if leftovers:
            raise ValueError(f"unfilled placeholders in {path}: {leftovers}")
        payloads.append(payload)

    submitted: list[int] = []
    try:
        for payload in payloads:
            submitted.append(_submit(central_url, payload))
    except Exception as exc:
        print(
            json.dumps(
                {"error": str(exc), "submitted_client_task_ids": submitted},
                separators=(",", ":"),
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "client_task_ids": submitted,
                "host_task_id": args.host_task_id,
                "node": discovery.get("node"),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
