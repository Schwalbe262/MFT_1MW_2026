"""Build and optionally publish reviewed no-mutation evidence for cycle 389."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

from filelock import FileLock
import requests


PROJECT_ROOT = Path(r"Y:\git\MFT_1MW_2026")
REGRESSION_ROOT = PROJECT_ROOT / "regression_260707"
RUNTIME_ROOT = REGRESSION_ROOT / "logs" / "controller_release_6a870_runtime"
LOCK_ROOT = RUNTIME_ROOT / "locks"
EVIDENCE_PATH = RUNTIME_ROOT / "cycle389_no_mutation_evidence.json"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regression_260707.campaign import _continuous_refill_b171c7c as controller
from regression_260707.verify import scheduler_client


SERIAL = 389
PREFIX = "mft-camp-sb171c7c-le6b9b9d-"
NAME_PATTERN = re.compile(rf"^{re.escape(PREFIX)}(\d+)$")


def _feeder_snapshot() -> dict:
    payload = json.loads(controller.FEEDER_STATE_PATH.read_text(encoding="utf-8"))
    return {
        key: payload[key]
        for key in (
            "state_revision", "serial", "candidate_cursor", "submitted_samples"
        )
    }


def build() -> dict:
    cycle = json.loads(controller._cycle_path(SERIAL).read_text(encoding="utf-8"))
    journal = cycle["formal_journal"]
    if not (
        cycle["status"] == "failed_closed"
        and journal.get("events") == []
        and int(journal.get("submitted_count") or 0) == 0
        and int(journal.get("planned_count") or 0) == 0
    ):
        raise RuntimeError("cycle 389 is not the observed pre-submit failure")

    before = _feeder_snapshot()
    response = requests.get(
        f"{scheduler_client.SCHEDULER}/api/tasks",
        params={"limit": 10000, "name_prefix": PREFIX},
        timeout=60,
    )
    response.raise_for_status()
    tasks = response.json()
    if not isinstance(tasks, list):
        tasks = tasks.get("tasks") if isinstance(tasks, dict) else None
    if not isinstance(tasks, list):
        raise RuntimeError("scheduler task inventory is invalid")
    rows = []
    for task in tasks:
        match = NAME_PATTERN.fullmatch(str(task.get("name") or ""))
        if match is None:
            continue
        if str(task.get("project") or "").strip() not in (
            "", scheduler_client.MFT_PROJECT
        ):
            raise RuntimeError("production identity exists in a foreign project")
        rows.append((int(match.group(1)), int(task["id"]), task["name"]))
    if not rows:
        raise RuntimeError("scheduler returned no production identities")
    max_serial = max(row[0] for row in rows)
    later_names = sorted(row[2] for row in rows if row[0] > before["serial"])
    after = _feeder_snapshot()
    artifacts = sorted(
        str(path)
        for path in controller.CYCLE_ROOT.glob("cycle-000389*.tmp*")
    )
    evidence = {
        "schema": controller.RECONCILIATION_EVIDENCE_SCHEMA,
        "cycle_serial": SERIAL,
        "action": "reconciled_no_mutation",
        "observed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "controller_stopped": True,
        "feeder_before": before,
        "feeder_after": after,
        "scheduler": {
            "matching_task_ids": [],
            "production_names_above_feeder_serial": later_names,
            "max_production_serial": max_serial,
            "inventory_rows_examined": len(tasks),
            "exact_generation_rows": len(rows),
            "project": scheduler_client.MFT_PROJECT,
        },
        "interrupted_artifacts": artifacts,
        "cycle_observation": {
            "status": cycle["status"],
            "error": cycle.get("error"),
            "events": len(journal.get("events") or []),
            "planned_count": int(journal.get("planned_count") or 0),
            "submitted_count": int(journal.get("submitted_count") or 0),
        },
    }
    evidence["evidence_sha256"] = controller._sha(evidence)
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--reviewed-sha")
    args = parser.parse_args()
    LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    with FileLock(str(LOCK_ROOT / "controller-loop.lock"), timeout=0):
        if not args.publish:
            evidence = build()
            EVIDENCE_PATH.write_text(
                json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(evidence, sort_keys=True))
            return 0
        evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
        if args.reviewed_sha != evidence.get("evidence_sha256"):
            raise RuntimeError("explicit reviewed SHA does not match evidence")
        sealed = controller._publish_reconciled_no_mutation(
            SERIAL, copy.deepcopy(evidence), args.reviewed_sha
        )
        print(json.dumps({
            "cycle_serial": SERIAL,
            "status": sealed["status"],
            "state_revision": sealed["state_revision"],
            "evidence_sha256": args.reviewed_sha,
            "submitted_count": sealed["formal_journal"]["submitted_count"],
        }, sort_keys=True))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
