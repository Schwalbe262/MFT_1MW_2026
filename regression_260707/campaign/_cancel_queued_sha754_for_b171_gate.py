"""Cancel only still-queued SHA754 cohort tasks so the SHA-b171 gate can start."""
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


MANIFEST_PATH = HERE / "pilot_manifests" / (
    "replacement-s754923c-le6b9b9d-seed260710-cursor1843.json"
)
MANIFEST_SHA256 = "f1490f2cda497c9475fe079fb0a04e5adb7686c6f4c99ae28a0f946a918319a8"
SUBMISSION_JOURNAL_PATH = HERE / "pilot_manifests" / (
    "replacement-s754923c-le6b9b9d-seed260710-cursor1843.journal.json"
)
RECOVERY_IDS = {28077, 28078, 28079, 28080}
OUTPUT_PATH = HERE / "pilot_manifests" / (
    "replacement-s754923c-le6b9b9d-seed260710-cursor1843.queued-gate-cancellation.json"
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


def _get(path, **params):
    response = requests.get(
        f"{scheduler_client.SCHEDULER}{path}", params=params or None, timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _task(task_id):
    return _get(f"/api/tasks/{int(task_id)}")


def _load_expected():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    unsigned = dict(manifest)
    stored = unsigned.pop("manifest_sha256", None)
    if stored != MANIFEST_SHA256 or _sha(unsigned) != MANIFEST_SHA256:
        raise RuntimeError("SHA754 manifest seal mismatch")
    if manifest.get("task_count") != 250:
        raise RuntimeError("SHA754 manifest count drifted")

    journal = json.loads(SUBMISSION_JOURNAL_PATH.read_text(encoding="utf-8"))
    if journal.get("manifest_sha256") != MANIFEST_SHA256 \
            or journal.get("audit", {}).get("manifest_sha256") != MANIFEST_SHA256 \
            or journal.get("audit", {}).get("first_task_id") != 27755 \
            or journal.get("audit", {}).get("last_task_id") != 28004 \
            or journal.get("audit", {}).get("task_count") != 250:
        raise RuntimeError("SHA754 submission journal identity drifted")
    submissions = journal.get("submissions", {})
    if len(submissions) != 250:
        raise RuntimeError("SHA754 submission journal is incomplete")

    expected = {}
    for row in submissions.values():
        task_id = int(row["task_id"])
        expected[task_id] = {
            "name": row["name"],
            "dedupe_key": row["dedupe_key"],
        }
    if sorted(expected) != list(range(27755, 28005)):
        raise RuntimeError("SHA754 submitted task ID range drifted")
    for row in manifest["tasks"]:
        task_id = 27755 + int(row["index"])
        if expected.get(task_id) != {
            "name": row["name"],
            "dedupe_key": row["dedupe_key"],
        }:
            raise RuntimeError(f"SHA754 task {task_id} ordinal identity drifted")
    manifest_identity = {
        (row["name"], row["dedupe_key"]) for row in manifest["tasks"]
    }
    if {(row["name"], row["dedupe_key"]) for row in expected.values()} \
            != manifest_identity:
        raise RuntimeError("SHA754 manifest/submission identities drifted")
    return expected


def main():
    if OUTPUT_PATH.exists():
        payload = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        unsigned = dict(payload)
        stored = unsigned.pop("journal_sha256", None)
        if stored != _sha(unsigned):
            raise RuntimeError("existing queued-cancellation journal seal mismatch")
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0

    expected = _load_expected()
    with scheduler_client.campaign_mutation_lock():
        rows = _get(
            "/api/tasks", project=scheduler_client.MFT_PROJECT, limit=5000,
        )
        if isinstance(rows, dict):
            rows = rows.get("tasks", [])
        queued = []
        for row in rows:
            task_id = int(row.get("id") or row.get("task_id") or 0)
            if task_id not in expected or row.get("status") != "queued":
                continue
            detail = _task(task_id)
            checks = {
                "name": detail.get("name") == expected[task_id]["name"],
                "dedupe": detail.get("dedupe_key") == expected[task_id]["dedupe_key"],
                "project": detail.get("project") == scheduler_client.MFT_PROJECT,
                "queued": detail.get("status") == "queued",
                "never_attached": not detail.get("attached_at"),
                "never_started": not detail.get("started_at"),
                "not_recovery": task_id not in RECOVERY_IDS,
            }
            if not all(checks.values()):
                raise RuntimeError(f"task {task_id} failed queued-only checks: {checks}")
            queued.append({
                "id": task_id,
                "name": detail["name"],
                "dedupe_key": detail["dedupe_key"],
                "checks": checks,
            })
        queued.sort(key=lambda row: row["id"])
        requested_ids = [row["id"] for row in queued]
        acknowledgement = {"cancelled": [], "count": 0}
        if requested_ids:
            response = requests.post(
                f"{scheduler_client.SCHEDULER}/api/tasks/cancel",
                params={
                    "statuses": "queued",
                    "task_ids": ",".join(map(str, requested_ids)),
                },
                timeout=60,
            )
            response.raise_for_status()
            acknowledgement = response.json()

        acknowledged = sorted(int(item) for item in acknowledgement.get("cancelled", []))
        if not set(acknowledged).issubset(set(requested_ids)):
            raise RuntimeError("scheduler acknowledged an out-of-scope cancellation")
        deadline = time.time() + 120
        after = []
        while True:
            after = [
                {
                    "id": task_id,
                    "status": _task(task_id).get("status"),
                }
                for task_id in requested_ids
            ]
            settled = all(
                row["status"] == "cancelled" if row["id"] in acknowledged
                else row["status"] != "queued"
                for row in after
            )
            if settled or time.time() >= deadline:
                break
            time.sleep(2)
        if not settled:
            raise RuntimeError(f"queued cancellations did not settle: {after}")

        recovery_after = [
            {"id": task_id, "status": _task(task_id).get("status")}
            for task_id in sorted(RECOVERY_IDS)
        ]
        if any(row["status"] == "cancelled" for row in recovery_after):
            raise RuntimeError("recovery task was unexpectedly cancelled")
        payload = {
            "schema": "sha754-queued-gate-cancellation-v1",
            "created_at": _now(),
            "manifest": str(MANIFEST_PATH.resolve()),
            "manifest_sha256": MANIFEST_SHA256,
            "authorization": "remove only never-started SHA754 queue entries ahead of SHA-b171 recovery gate",
            "requested": queued,
            "requested_ids": requested_ids,
            "acknowledgement": acknowledgement,
            "after": after,
            "recovery_after": recovery_after,
            "scheduler_mutation_count": 1 if requested_ids else 0,
        }
        payload["journal_sha256"] = _sha(payload)
        _atomic(OUTPUT_PATH, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
