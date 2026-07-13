"""Review and seal the two accepted tasks from interrupted refill cycle 467."""

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
EVIDENCE_PATH = RUNTIME_ROOT / "cycle467_committed_evidence.json"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regression_260707.campaign import _continuous_refill_b171c7c as controller
from regression_260707.campaign import feeder, pinned_pilot
from regression_260707.verify import scheduler_client


SERIAL = 467
EXPECTED_ACCEPTED = 2
EXPECTED_BEFORE_SERIAL = 19587
EXPECTED_BEFORE_SUBMITTED = 1976
EXPECTED_LAST_SERIAL = 19589
PREFIX = "mft-camp-sb171c7c-le6b9b9d-"
NAME_PATTERN = re.compile(rf"^{re.escape(PREFIX)}(\d+)$")
IDENTITY_FIELDS = ("id", "name", "dedupe_key", "project")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_cycle() -> dict:
    cycle = controller._load_cycle(controller._cycle_path(SERIAL))
    if not isinstance(cycle, dict):
        raise RuntimeError("cycle 467 is missing")
    return cycle


def _accepted_prefix(cycle: dict) -> tuple[list[dict], list[dict]]:
    journal = cycle["formal_journal"]
    events = list(journal["events"])
    accepted = [event for event in events if event.get("task_id") is not None]
    if not (
        cycle["status"] == "failed_closed"
        and journal.get("batch_commit") is True
        and len(events) == 3
        and len(accepted) == EXPECTED_ACCEPTED
        and accepted == events[:EXPECTED_ACCEPTED]
        and all(event.get("accepted_or_reconciled") is True for event in accepted)
        and all(event.get("ledger_committed") is not True for event in events)
        and all(event.get("task_id") is None for event in events[EXPECTED_ACCEPTED:])
        and not any(event.get("uncertain") is True for event in events)
    ):
        raise RuntimeError("cycle 467 is not the exact observed accepted-prefix failure")
    return events, accepted


def _scheduler_rows(accepted: list[dict]) -> list[dict]:
    rows = []
    for event in accepted:
        task_id = int(event["task_id"])
        response = requests.get(
            f"{scheduler_client.SCHEDULER}/api/tasks/{task_id}", timeout=30
        )
        response.raise_for_status()
        task = response.json()
        checks = {
            "name": task.get("name") == event["name"],
            "dedupe": task.get("dedupe_key") == event["dedupe_key"],
            "project": task.get("project") == scheduler_client.MFT_PROJECT,
        }
        if not all(checks.values()):
            raise RuntimeError(f"accepted task {task_id} identity drifted: {checks}")
        rows.append({key: task.get(key) for key in IDENTITY_FIELDS} | {
            "status": task.get("status"),
        })
    if len({row["id"] for row in rows}) != EXPECTED_ACCEPTED:
        raise RuntimeError("accepted task IDs are not unique")
    planned_unaccepted = {
        event["name"] for event in _load_cycle()["formal_journal"]["events"]
        if event.get("task_id") is None
    }
    for name in planned_unaccepted:
        response = requests.get(
            f"{scheduler_client.SCHEDULER}/api/tasks",
            params={"limit": 10, "name_prefix": name},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        tasks = payload if isinstance(payload, list) else payload.get("tasks")
        if not isinstance(tasks, list):
            raise RuntimeError("scheduler task inventory is invalid")
        if any(str(task.get("name") or "") == name for task in tasks):
            raise RuntimeError("an unaccepted planned identity exists in the scheduler")
    return rows


def _replay_target(before: dict, accepted: list[dict]) -> dict:
    generation = f"{controller.SOLVER}:{controller.LIBRARY}:seed{controller.SEED}"
    cursor = int(before["candidate_cursor"])
    used = set(before["used_params_sha256"])
    pending = set()
    final_raw = int(before["candidate_raw_index"])
    profile = json.loads(Path(feeder.PROFILE_PATH).read_text(encoding="utf-8"))
    profile["timeout_seconds"] = controller.TIMEOUT_SECONDS
    for index, event in enumerate(accepted):
        while True:
            next_cursor, raw_index, params = feeder.next_valid_candidate(
                cursor, seed=controller.SEED
            )
            controller._candidate_contract(params, f"reconcile[{index}]")
            params_sha = pinned_pilot.candidate_digest(params)
            cursor = next_cursor
            if params_sha in used or params_sha in pending:
                continue
            break
        submission_params = {key: copy.deepcopy(params[key]) for key in sorted(params)}
        identity = scheduler_client.verification_submission_identity(
            event["name"], submission_params, profile,
            controller.SOLVER, controller.LIBRARY,
        )
        checks = {
            "raw": int(event["candidate_raw_index"]) == int(raw_index),
            "params": event["params_sha256"] == params_sha,
            "dedupe": event["dedupe_key"] == identity["dedupe_key"],
        }
        if not all(checks.values()):
            raise RuntimeError(f"accepted event {index} candidate drifted: {checks}")
        pending.add(params_sha)
        final_raw = int(raw_index)

    first_serial = int(NAME_PATTERN.fullmatch(accepted[0]["name"]).group(1))
    last_serial = int(NAME_PATTERN.fullmatch(accepted[-1]["name"]).group(1))
    if (
        first_serial != int(before["serial"]) + 1
        or last_serial != EXPECTED_LAST_SERIAL
    ):
        raise RuntimeError("accepted event serial range is not contiguous")

    target = copy.deepcopy(before)
    target["serial"] = last_serial
    target["candidate_cursor"] = cursor
    target["candidate_cursors"][generation] = cursor
    target["candidate_raw_index"] = final_raw
    target["submitted_samples"] = int(before["submitted_samples"]) + len(accepted)
    target.setdefault("outstanding", []).extend(int(event["task_id"]) for event in accepted)
    expected_rows = target.setdefault("task_expected_rows", {})
    for event in accepted:
        expected_rows[str(int(event["task_id"]))] = feeder.COUNT_PER_TASK
    target["used_names"] = sorted(set(before["used_names"]) | {
        event["name"] for event in accepted
    })
    target["used_dedupe_keys"] = sorted(set(before["used_dedupe_keys"]) | {
        event["dedupe_key"] for event in accepted
    })
    target["used_params_sha256"] = sorted(set(before["used_params_sha256"]) | pending)
    return target


def build() -> dict:
    cycle = _load_cycle()
    _events, accepted = _accepted_prefix(cycle)
    before = controller._load_feeder_state(controller._static_bundle(), create=False)
    if (
        int(before["serial"]) != EXPECTED_BEFORE_SERIAL
        or int(before["submitted_samples"]) != EXPECTED_BEFORE_SUBMITTED
    ):
        raise RuntimeError("feeder is not at the exact pre-cycle state")
    rows = _scheduler_rows(accepted)
    target = _replay_target(before, accepted)
    evidence = {
        "schema": controller.RECONCILIATION_EVIDENCE_SCHEMA,
        "cycle_serial": SERIAL,
        "action": "reconciled_committed",
        "observed_at": _now(),
        "controller_stopped": True,
        "accepted_count": len(accepted),
        "unaccepted_count": 1,
        "accepted_task_ids": [int(event["task_id"]) for event in accepted],
        "scheduler_identity_sha256": controller._sha([
            {key: row[key] for key in IDENTITY_FIELDS} for row in rows
        ]),
        "scheduler_status_counts": {
            status: sum(row["status"] == status for row in rows)
            for status in sorted({str(row["status"]) for row in rows})
        },
        "feeder_before_sha256": controller._sha(before),
        "feeder_before": {
            key: before[key] for key in (
                "state_revision", "serial", "candidate_cursor",
                "candidate_raw_index", "submitted_samples",
            )
        },
        "feeder_target_sha256": controller._sha(target),
        "feeder_target": {
            key: target[key] for key in (
                "serial", "candidate_cursor", "candidate_raw_index",
                "submitted_samples",
            )
        },
        "cycle_failed_generation_revision": int(cycle["state_revision"]),
        "cycle_error": cycle.get("error"),
    }
    evidence["evidence_sha256"] = controller._sha(evidence)
    return evidence


def publish(reviewed_sha: str) -> dict:
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    unsigned = copy.deepcopy(evidence)
    stored_sha = unsigned.pop("evidence_sha256", None)
    if reviewed_sha != stored_sha or controller._sha(unsigned) != reviewed_sha:
        raise RuntimeError("explicit reviewed evidence SHA does not match")
    cycle = _load_cycle()
    _events, accepted = _accepted_prefix(cycle)
    rows = _scheduler_rows(accepted)
    identity_sha = controller._sha([
        {key: row[key] for key in IDENTITY_FIELDS} for row in rows
    ])
    if identity_sha != evidence["scheduler_identity_sha256"]:
        raise RuntimeError("scheduler accepted identity set changed after review")
    bundle = controller._static_bundle()
    before = controller._load_feeder_state(bundle, create=False)
    if controller._sha(before) != evidence["feeder_before_sha256"]:
        raise RuntimeError("feeder state changed after review")
    target = _replay_target(before, accepted)
    if controller._sha(target) != evidence["feeder_target_sha256"]:
        raise RuntimeError("replayed feeder target changed after review")

    controller._save_feeder_state(target)
    committed = controller._load_feeder_state(bundle, create=False)
    if any(committed[key] != target[key] for key in target if key != "state_revision"):
        raise RuntimeError("feeder durable readback does not match reconciled target")

    updated = copy.deepcopy(cycle)
    journal = updated["formal_journal"]
    for event in journal["events"][:EXPECTED_ACCEPTED]:
        event["ledger_committed"] = True
    journal["submitted_count"] = EXPECTED_ACCEPTED
    journal["completed"] = True
    journal["stop_reason"] = "reviewed_reconciled_committed_partial_capacity"
    updated["reconciliation"] = {
        "action": "reconciled_committed",
        "published_at": _now(),
        "evidence_sha256": reviewed_sha,
        "evidence": evidence,
    }
    controller._save_cycle(
        controller._cycle_path(SERIAL), updated, "reconciled_committed"
    )
    sealed = controller._load_cycle(controller._cycle_path(SERIAL))
    return {
        "cycle_serial": SERIAL,
        "status": sealed["status"],
        "state_revision": sealed["state_revision"],
        "submitted_count": sealed["formal_journal"]["submitted_count"],
        "feeder_serial": committed["serial"],
        "feeder_state_revision": committed["state_revision"],
        "evidence_sha256": reviewed_sha,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--reviewed-sha")
    args = parser.parse_args()
    LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    with FileLock(str(LOCK_ROOT / "controller-loop.lock"), timeout=0):
        with scheduler_client.campaign_mutation_lock(timeout=30):
            if not args.publish:
                evidence = build()
                EVIDENCE_PATH.write_text(
                    json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                print(json.dumps(evidence, sort_keys=True))
                return 0
            print(json.dumps(publish(str(args.reviewed_sha or "")), sort_keys=True))
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
