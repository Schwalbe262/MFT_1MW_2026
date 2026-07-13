"""Operate the user-authorized SHA3216 preloaded-250 campaign.

This is an untracked operational controller so the simulation solver HEAD stays
at SHA3216.  It adopts the already audited 250-task cohort, waits for local3
and a strict fleet20/90% gate, then keeps the logical MFT project at 300 active
tasks using the same candidate ledger, 64 GiB memory, and four-hour timeout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock


HERE = Path(__file__).resolve().parent
REGRESSION_ROOT = HERE.parent
REPO_ROOT = REGRESSION_ROOT.parent
VERIFY = REGRESSION_ROOT / "verify"
for path in (str(HERE), str(REGRESSION_ROOT), str(VERIFY), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import feeder
import pinned_pilot
import rapid_campaign
import scheduler_client
from training.checkpoint_contract import (
    checkpoint_status_revision_identity_matches,
)


SOLVER = "3216e43a5a1a362ee2ed1aba89b642498c60d1b9"
LIBRARY = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
SEED = 260710
PREFIX = f"mft-camp-s{SOLVER[:7]}-l{LIBRARY[:7]}-"
INITIAL_COUNT = 250
INITIAL_FIRST_ID = 27149
INITIAL_FIRST_SERIAL = 16862
TARGET_ACTIVE = 300
TARGET_STRICT_ROWS = 3000
EMERGENCY_RAW_CEILING = 12000
MEMORY_MB = 65_536
TIMEOUT_SECONDS = 14_400
FLEET_GATE_TERMINAL = 20
FLEET_GATE_VALID_RATE = 0.90
STATE_PATH = HERE / "adopted_refill_3216_state.json"
STRICT_STATUS_PATH = REGRESSION_ROOT / "training" / "strict_data_status.json"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    staged.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(staged, path)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    return _utc_now().isoformat(timespec="seconds")


def _load_profile() -> dict:
    with open(feeder.PROFILE_PATH, encoding="utf-8") as stream:
        profile = json.load(stream)
    profile["timeout_seconds"] = TIMEOUT_SECONDS
    return profile


def _submit_64gb(name, workdir, params, solver_revision, library_revision):
    return scheduler_client.submit_verification(
        name=name,
        workdir=workdir,
        params=params,
        profile=_load_profile(),
        mem_mb=MEMORY_MB,
        cpus=4,
        solver_revision=solver_revision,
        library_revision=library_revision,
    )


def _new_state() -> dict:
    return {
        "schema_version": 1,
        "solver_revision": SOLVER,
        "library_revision": LIBRARY,
        "seed": SEED,
        "initial_count": INITIAL_COUNT,
        "target_active": TARGET_ACTIVE,
        "target_strict_rows": TARGET_STRICT_ROWS,
        "adoption_sealed": False,
        "adoption_sha256": None,
        "promoted": False,
        "promoted_at": None,
        "paused": False,
        "pause_reasons": [],
        "task_outcomes": {},
        "last_strict_rows": 0,
        "last_strict_growth_at": None,
        "updated_at": None,
    }


def _load_state() -> dict:
    if not STATE_PATH.is_file():
        return _new_state()
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "solver_revision": SOLVER,
        "library_revision": LIBRARY,
        "seed": SEED,
        "initial_count": INITIAL_COUNT,
        "target_active": TARGET_ACTIVE,
        "target_strict_rows": TARGET_STRICT_ROWS,
    }
    mismatches = {
        key: (state.get(key), value)
        for key, value in expected.items()
        if state.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"adopted refill state identity mismatch: {mismatches}")
    if not isinstance(state.get("task_outcomes"), dict):
        raise RuntimeError("adopted refill task_outcomes is invalid")
    return state


def _save_state(state: dict) -> None:
    state["updated_at"] = _iso_now()
    _atomic_json(STATE_PATH, state)


def _current_generation(tasks: list[dict]) -> list[dict]:
    selected = [
        task for task in tasks
        if re.fullmatch(re.escape(PREFIX) + r"\d{5,}", str(task.get("name") or ""))
    ]
    ids = [int(task["id"]) for task in selected]
    if len(ids) != len(set(ids)):
        raise RuntimeError("current generation has duplicate task IDs")
    return selected


def _seal_initial_cohort(tasks: list[dict]) -> str:
    by_name = {str(task.get("name") or ""): task for task in tasks}
    if len(by_name) != len(tasks):
        raise RuntimeError("current generation has duplicate task names")
    profile = _load_profile()
    cursor = pinned_pilot.cursor_after_valid_candidates(10, seed=SEED)
    records = []
    for offset in range(INITIAL_COUNT):
        cursor, raw_index, params = pinned_pilot.next_valid_candidate(
            cursor, seed=SEED
        )
        serial = INITIAL_FIRST_SERIAL + offset
        name = f"{PREFIX}{serial:05d}"
        task = by_name.get(name)
        if task is None:
            raise RuntimeError(f"preloaded cohort task is missing: {name}")
        expected_id = INITIAL_FIRST_ID + offset
        task_id = int(task["id"])
        expected_dedupe = scheduler_client.verification_dedupe_key(
            name, params, profile, SOLVER, LIBRARY
        )
        checks = {
            "id": task_id == expected_id,
            "project": task.get("project") == scheduler_client.MFT_PROJECT,
            "dedupe": task.get("dedupe_key") == expected_dedupe,
            "cpus": int(task.get("cpus") or 0) == 4,
            "memory_mb": int(task.get("memory_mb") or 0) == MEMORY_MB,
            "timeout_seconds": int(task.get("timeout_seconds") or 0)
            == TIMEOUT_SECONDS,
        }
        if not all(checks.values()):
            raise RuntimeError(
                f"preloaded cohort contract mismatch for {name}: {checks}"
            )
        cw1 = float(params["cw1"])
        if not math.isfinite(cw1) or cw1 > 10.0:
            raise RuntimeError(f"preloaded cohort cw1 contract failed: {name}={cw1}")
        records.append(
            {
                "id": task_id,
                "name": name,
                "raw_index": int(raw_index),
                "dedupe_key": expected_dedupe,
            }
        )
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _revalidate_adopted_ids(tasks: list[dict]) -> None:
    by_id = {int(task["id"]): task for task in tasks}
    for offset in range(INITIAL_COUNT):
        task_id = INITIAL_FIRST_ID + offset
        name = f"{PREFIX}{INITIAL_FIRST_SERIAL + offset:05d}"
        task = by_id.get(task_id)
        if task is None or task.get("name") != name:
            raise RuntimeError(f"adopted task identity disappeared: {task_id}/{name}")
        if f":{SOLVER}:{LIBRARY}:" not in str(task.get("dedupe_key") or ""):
            raise RuntimeError(f"adopted task pin changed: {task_id}")


def _local3_passed() -> bool:
    manifest_dir = pinned_pilot.campaign_manifest_dir()
    path = manifest_dir / f"{pinned_pilot.local_gate_tag(SOLVER, LIBRARY)}.json"
    if not path.is_file():
        return False
    pinned_pilot.validate_local_gate(SOLVER, LIBRARY, manifest_dir=manifest_dir)
    return True


def _strict_rows() -> int:
    payload = json.loads(STRICT_STATUS_PATH.read_text(encoding="utf-8"))
    if not checkpoint_status_revision_identity_matches(
        payload, SOLVER, LIBRARY
    ):
        raise RuntimeError("strict status revision identity mismatch")
    timestamp = datetime.fromisoformat(str(payload["time"]).replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        timestamp = timestamp.astimezone()
    age_seconds = (_utc_now() - timestamp.astimezone(timezone.utc)).total_seconds()
    if age_seconds > 20 * 60:
        raise RuntimeError(f"strict status is stale by {age_seconds:.0f}s")
    rows = int(payload.get("strict_full_rows") or 0)
    if rows < 0:
        raise RuntimeError("strict row count is negative")
    return rows


def run_once(execute: bool) -> dict:
    feeder._require_deployed_revisions(SOLVER, LIBRARY)
    with FileLock(str(STATE_PATH) + ".lock", timeout=30):
        with scheduler_client.campaign_mutation_lock():
            state = _load_state()
            inventory = feeder.campaign_inventory()
            current = _current_generation(inventory)
            if len(current) < INITIAL_COUNT:
                raise RuntimeError(
                    f"current generation has only {len(current)}/{INITIAL_COUNT} tasks"
                )
            if not state["adoption_sealed"]:
                state["adoption_sha256"] = _seal_initial_cohort(current)
                state["adoption_sealed"] = True
            else:
                _revalidate_adopted_ids(current)

            production = rapid_campaign.inspect_production_tasks(
                inventory,
                SOLVER,
                LIBRARY,
                cached_outcomes=state.get("task_outcomes"),
            )
            state["task_outcomes"] = production["cache"]
            outcomes = production["outcomes"]
            terminal = len(outcomes)
            valid = sum(item["state"] == "valid" for item in outcomes)
            valid_rate = valid / terminal if terminal else None
            local3 = _local3_passed()
            strict_rows = _strict_rows()

            if strict_rows > int(state.get("last_strict_rows") or 0):
                state["last_strict_rows"] = strict_rows
                state["last_strict_growth_at"] = _iso_now()

            reasons = rapid_campaign._production_gate_reasons(production)
            if state.get("promoted") and not local3:
                reasons.append("local3_evidence_missing_after_promotion")
            last_growth_text = state.get("last_strict_growth_at")
            if state.get("promoted") and last_growth_text and valid:
                last_growth = datetime.fromisoformat(
                    str(last_growth_text).replace("Z", "+00:00")
                )
                if last_growth.tzinfo is None:
                    last_growth = last_growth.replace(tzinfo=timezone.utc)
                stalled_seconds = (
                    _utc_now() - last_growth.astimezone(timezone.utc)
                ).total_seconds()
                if stalled_seconds >= 90 * 60:
                    reasons.append("strict_dataset_growth_stalled_90m")
            if not state.get("promoted") and terminal >= FLEET_GATE_TERMINAL:
                if valid_rate is None or valid_rate < FLEET_GATE_VALID_RATE:
                    reasons.append(
                        f"fleet20_valid_rate_below_90pct:{valid_rate or 0.0:.3f}"
                    )
            if reasons:
                state["paused"] = True
                state["pause_reasons"] = sorted(
                    set([*state.get("pause_reasons", []), *reasons])
                )

            action = "observe"
            mutation = None
            if strict_rows >= TARGET_STRICT_ROWS:
                action = "target_reached_drain"
            elif state.get("paused"):
                action = "paused"
            elif not local3:
                action = "wait_local3"
            elif terminal < FLEET_GATE_TERMINAL:
                action = "wait_fleet20"
            elif not state.get("promoted"):
                action = "promote_300" if execute else "ready_to_promote_300"
                if execute:
                    state["promoted"] = True
                    state["promoted_at"] = _iso_now()
            else:
                action = "refill_300"

            if execute and action in ("promote_300", "refill_300"):
                feeder.MAX_STANDALONE_ACTIVE = TARGET_ACTIVE
                feeder.submit = _submit_64gb
                feeder._step_locked(
                    EMERGENCY_RAW_CEILING,
                    target=TARGET_ACTIVE,
                    buffer=0,
                    solver_revision=SOLVER,
                    library_revision=LIBRARY,
                    candidate_seed=SEED,
                )
                mutation = {"refill_target": TARGET_ACTIVE}

            if execute:
                _save_state(state)
            return {
                "time": _iso_now(),
                "mode": "execute" if execute else "read_only",
                "action": action,
                "mutation": mutation,
                "adoption_sealed": state["adoption_sealed"],
                "adoption_sha256": state["adoption_sha256"],
                "local3_passed": local3,
                "promoted": state["promoted"],
                "paused": state["paused"],
                "pause_reasons": state["pause_reasons"],
                "production_active": production["active"],
                "production_terminal": terminal,
                "production_valid": valid,
                "production_valid_rate": valid_rate,
                "strict_full_rows": strict_rows,
                "target_strict_rows": TARGET_STRICT_ROWS,
            }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--loop", type=int, default=None)
    args = parser.parse_args()
    if args.loop is not None and args.loop < 60:
        parser.error("--loop must be at least 60 seconds")
    if args.loop is not None and not args.execute:
        parser.error("--loop requires --execute")
    while True:
        try:
            print(json.dumps(run_once(args.execute), sort_keys=True), flush=True)
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "time": _iso_now(),
                        "mode": "execute" if args.execute else "read_only",
                        "action": "observation_error_no_mutation",
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if args.loop is None:
            return
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
