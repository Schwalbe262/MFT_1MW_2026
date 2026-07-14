"""Submit the generated N=3 canary host payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import requests


DEFAULT_SCHEDULER_URL = "http://127.0.0.1:8000"
EXPECTED_HOST_NAME = "mft-aedt-n3canary-260714-host"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, default=Path("payloads/host_payload.json"))
    parser.add_argument("--scheduler-url", default=DEFAULT_SCHEDULER_URL)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("name") != EXPECTED_HOST_NAME:
        raise SystemExit(f"refusing to submit an unexpected host payload: {args.payload}")
    url = f"{args.scheduler_url.rstrip('/')}/api/tasks"
    response = requests.post(url, json=payload, timeout=20)
    if response.status_code not in {200, 201}:
        raise SystemExit(
            f"host submission failed: HTTP {response.status_code}: {response.text[:1000]}"
        )
    result = response.json()
    task_id = result.get("task_id") or result.get("id")
    if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
        raise SystemExit(f"scheduler accepted host without a task ID: {result!r}")
    print(
        json.dumps(
            {"host_task_id": task_id, "name": EXPECTED_HOST_NAME},
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
