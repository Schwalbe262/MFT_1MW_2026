"""Cancel only the four never-started SHA7873 recovery tasks, with a sealed audit."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


HERE = Path(__file__).resolve().parent
VERIFY = HERE.parent / "verify"
for item in (HERE, VERIFY):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import scheduler_client


IDS = (28073, 28074, 28075, 28076)
PREFIX = "mft-recovery4-s7873ddd-le6b9b9d-"
SUBMISSION_SHA256 = "8623bd6ebc9b3839cc2087e1d9084c7e574ddb10e6a4563b3c869c0d4f8f2cb6"
SUBMISSION_PATH = HERE / "pilot_manifests" / (
    "thermal-recovery4-s7873ddd-le6b9b9d.submission.json"
)
JOURNAL_PATH = HERE / "pilot_manifests" / (
    "thermal-recovery4-s7873ddd-le6b9b9d.queued-cancellation.json"
)


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic(path, payload):
    fd, staged = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, path)
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def _task(task_id):
    response = requests.get(
        f"{scheduler_client.SCHEDULER}/api/tasks/{task_id}", timeout=20,
    )
    response.raise_for_status()
    return response.json()


def main():
    if JOURNAL_PATH.exists():
        journal = json.loads(JOURNAL_PATH.read_text(encoding="utf-8"))
        unsigned = dict(journal)
        stored = unsigned.pop("journal_sha256", None)
        if stored != _sha(unsigned):
            raise RuntimeError("existing queued cancellation journal seal mismatch")
        print(json.dumps(journal, sort_keys=True))
        return 0

    submission = json.loads(SUBMISSION_PATH.read_text(encoding="utf-8"))
    unsigned = dict(submission)
    stored = unsigned.pop("submission_sha256", None)
    if stored != SUBMISSION_SHA256 or _sha(unsigned) != SUBMISSION_SHA256:
        raise RuntimeError("SHA7873 submission journal seal mismatch")
    expected = {int(row["task_id"]): row for row in submission["tasks"]}
    if tuple(expected) != IDS:
        raise RuntimeError("SHA7873 submission task IDs drifted")

    with scheduler_client.campaign_mutation_lock():
        before = []
        for task_id in IDS:
            row = _task(task_id)
            checks = {
                "name": row.get("name") == expected[task_id]["name"],
                "prefix": str(row.get("name") or "").startswith(PREFIX),
                "dedupe": row.get("dedupe_key") == expected[task_id]["dedupe_key"],
                "project": row.get("project") == scheduler_client.MFT_PROJECT,
                "queued": row.get("status") == "queued",
                "never_started": not row.get("started_at"),
                "never_attached": not row.get("attached_at"),
            }
            if not all(checks.values()):
                raise RuntimeError(f"task {task_id} is not safe queued-only cancellation: {checks}")
            before.append({
                "id": task_id, "name": row["name"], "status": row["status"],
                "created_at": row.get("created_at"), "checks": checks,
            })

        response = requests.post(
            f"{scheduler_client.SCHEDULER}/api/tasks/cancel",
            params={"statuses": "queued", "task_ids": ",".join(map(str, IDS))},
            timeout=60,
        )
        response.raise_for_status()
        acknowledgement = response.json()
        deadline = time.time() + 120
        after = []
        while time.time() < deadline:
            after = [_task(task_id) for task_id in IDS]
            if all(row.get("status") == "cancelled" for row in after):
                break
            time.sleep(2)
        if not after or not all(row.get("status") == "cancelled" for row in after):
            raise RuntimeError(
                f"queued recovery cancellation did not settle: "
                f"{[(row.get('id'), row.get('status')) for row in after]}"
            )

        journal = {
            "schema": "queued-recovery4-cancellation-v1",
            "created_at": _now(),
            "submission_journal": str(SUBMISSION_PATH.resolve()),
            "submission_sha256": SUBMISSION_SHA256,
            "authorization": "superseded before allocation by SHA b171c7c recovery4",
            "requested_ids": list(IDS),
            "before": before,
            "acknowledgement": acknowledgement,
            "after": [
                {"id": int(row["id"]), "name": row["name"], "status": row["status"]}
                for row in after
            ],
            "scheduler_mutation_count": 1,
        }
        journal["journal_sha256"] = _sha(journal)
        _atomic(JOURNAL_PATH, journal)
    print(json.dumps(journal, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
