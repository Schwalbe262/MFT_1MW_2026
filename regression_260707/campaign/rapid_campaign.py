"""Fail-closed 3 -> 10 -> 50 -> 300 campaign promotion controller.

The default command is read-only.  ``--execute`` is required before this tool
may submit a pinned pilot stage or ask :mod:`feeder` to refill production work.
It never cancels tasks.  The controller deliberately delegates candidate
generation, submission, capacity limits, ledgers, and strict result validation
to the existing campaign modules instead of introducing a second submit path.
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
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock


HERE = Path(__file__).resolve().parent
REGRESSION_ROOT = HERE.parent
REPO_ROOT = REGRESSION_ROOT.parent
sys.path.insert(0, str(REGRESSION_ROOT))
sys.path.insert(0, str(REGRESSION_ROOT / "verify"))

import feeder
import deployment_gate
import pinned_pilot
import provisional_wave
import scheduler_client


SCHEMA_VERSION = 1
DEFAULT_SEED = 260710
# Operationally unlimited while retaining feeder.step's integer hard-cap API.
# The campaign must continue past 10k until an operator stops the controller.
DEFAULT_MAX_SAMPLES = 2_000_000_000
DEFAULT_LOOP_SECONDS = 60
CANDIDATE_AUDIT_COUNT = 300
PILOT_EARLY_VALID = 5
FLEET_GATE_TERMINAL = 20
FLEET_GATE_VALID_RATE = 0.90
RECENT_WINDOW = 30
RECENT_MIN_VALID_RATE = 0.70
REPEATED_ERROR_LIMIT = 3
DATA_STALL_SECONDS = 90 * 60
MAX_STANDARD_TIMEOUT_SECONDS = 2 * 60 * 60
ACTIVE_STATUSES = frozenset(("queued", "attaching", "running"))
TERMINAL_STATUSES = frozenset(("completed", "failed", "cancelled"))
STATE_DIR_ENV = "MFT_RAPID_CAMPAIGN_STATE_DIR"
TEMPERATURE_PATTERN = re.compile(r"^(?:T_(?:max|mean)_|Tprobe_)")

STAGE_LOCAL3 = "local3"
STAGE_PILOT10 = "pilot10"
STAGE_FLEET50 = "fleet50"
STAGE_PRODUCTION300 = "production300"
STAGE_TARGETS = {
    STAGE_LOCAL3: 3,
    STAGE_PILOT10: 10,
    STAGE_FLEET50: 50,
    STAGE_PRODUCTION300: 300,
}


def _now():
    return datetime.now(timezone.utc)


def _iso(value):
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def _parse_time(value):
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def default_state_path(solver_revision, library_revision, seed=DEFAULT_SEED):
    override = os.environ.get(STATE_DIR_ENV, "").strip()
    root = (
        Path(override).expanduser().resolve()
        if override else pinned_pilot.campaign_manifest_dir())
    name = (
        f"rapid-s{solver_revision[:12]}-l{library_revision[:12]}-"
        f"seed{int(seed)}.json")
    return root / name


def new_state(solver_revision, library_revision, seed=DEFAULT_SEED):
    return {
        "schema_version": SCHEMA_VERSION,
        "solver_revision": solver_revision,
        "library_revision": library_revision,
        "seed": int(seed),
        "stage": STAGE_LOCAL3,
        "target_active": 0,
        "paused": False,
        "pause_reasons": [],
        "candidate_audit": None,
        "last_dataset_rows": None,
        "last_dataset_growth_at": None,
        "first_production_valid_at": None,
        "last_production_valid_at": None,
        "task_outcomes": {},
        "updated_at": None,
    }


def load_state(path, solver_revision, library_revision, seed=DEFAULT_SEED):
    path = Path(path)
    if not path.is_file():
        return new_state(solver_revision, library_revision, seed)
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f"rapid campaign state is unreadable: {path}: {exc}") from exc
    expected = {
        "schema_version": SCHEMA_VERSION,
        "solver_revision": solver_revision,
        "library_revision": library_revision,
        "seed": int(seed),
    }
    mismatches = {
        key: (state.get(key), value)
        for key, value in expected.items() if state.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"rapid campaign state identity mismatch: {mismatches}")
    if state.get("stage") not in STAGE_TARGETS:
        raise RuntimeError(f"rapid campaign state has an invalid stage: {state.get('stage')!r}")
    if not isinstance(state.get("pause_reasons", []), list):
        raise RuntimeError("rapid campaign pause_reasons must be a list")
    if not isinstance(state.get("task_outcomes", {}), dict):
        raise RuntimeError("rapid campaign task_outcomes must be an object")
    return state


def save_state(state, path):
    state = dict(state)
    state["updated_at"] = _iso(_now())
    pinned_pilot._atomic_manifest(state, Path(path))


def candidate_supply_audit(
        solver_revision, library_revision, seed=DEFAULT_SEED,
        count=CANDIDATE_AUDIT_COUNT):
    """Pre-generate and authenticate the first production candidate tranche."""
    profile = provisional_wave._load_profile()
    try:
        timeout_seconds = int(profile["timeout_seconds"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("standard profile has no valid timeout_seconds") from exc
    if not 0 < timeout_seconds <= MAX_STANDARD_TIMEOUT_SECONDS:
        raise RuntimeError(
            "standard profile timeout must be at most 7200 seconds before rapid "
            f"promotion, got {timeout_seconds}")
    records = provisional_wave.build_plan(
        solver_revision, library_revision, profile, seed=seed, count=count)
    expected = provisional_wave.new_manifest(
        solver_revision, library_revision, profile, records, seed=seed)
    provisional_wave.validate_manifest(expected, expected)
    raw_indices = [record["candidate_raw_index"] for record in records]
    digests = [record["params_sha256"] for record in records]
    names = [record["name"] for record in records]
    if (len(set(raw_indices)) != count or len(set(digests)) != count
            or len(set(names)) != count):
        raise RuntimeError("prevalidated production candidates are not unique")
    payload = [provisional_wave._plan_identity(record) for record in records]
    plan_sha256 = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "count": count,
        "first_raw_index": raw_indices[0],
        "last_raw_index": raw_indices[-1],
        "plan_sha256": plan_sha256,
    }


def production_prefix(solver_revision, library_revision):
    return f"mft-camp-s{solver_revision[:7]}-l{library_revision[:7]}-"


def is_feeder_task(task, solver_revision, library_revision):
    prefix = re.escape(production_prefix(solver_revision, library_revision))
    return bool(re.fullmatch(prefix + r"\d{5,}", str(task.get("name") or "")))


def thermal_saturation_columns(result):
    """Return temperature fields at or above the Icepak trust ceiling."""
    if not isinstance(result, dict):
        return []
    saturated = []
    for key, value in result.items():
        if not TEMPERATURE_PATTERN.match(str(key)):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(numeric) and numeric >= scheduler_client.MAX_TRUSTED_TEMPERATURE_C:
            saturated.append(str(key))
    return sorted(saturated)


def invalid_result_reason(result, solver_revision, library_revision, fetch_state=None):
    saturated = thermal_saturation_columns(result)
    if saturated:
        return "thermal_saturation:" + ",".join(saturated)
    if not isinstance(result, dict):
        return f"result_{fetch_state or 'missing'}"
    solver_hash = str(result.get("git_hash") or "").strip().lower()
    if solver_hash and solver_hash != solver_revision:
        return "solver_revision_mismatch"
    library_hash = str(result.get("pyaedt_library_git_hash") or "").strip().lower()
    if library_hash and library_hash != library_revision:
        return "library_revision_mismatch"
    return f"result_{fetch_state or 'strict_invalid'}"


def _failure_message(task):
    parts = []
    for key in (
            "failure_reason", "error_message", "error", "status_reason",
            "message", "exit_code"):
        value = task.get(key)
        if value not in (None, "", []):
            parts.append(f"{key}={value}")
    return " | ".join(parts) or f"status={task.get('status', 'failed')}"


def error_fingerprint(message):
    normalized = str(message or "unknown").strip().lower()
    normalized = re.sub(r"[0-9a-f]{12,}", "<hex>", normalized)
    normalized = re.sub(r"\b\d+\b", "<n>", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _terminal_time(task, result=None):
    for value in (
            task.get("finished_at"), task.get("completed_at"),
            task.get("updated_at"),
            result.get("saved_at") if isinstance(result, dict) else None):
        parsed = _parse_time(value)
        if parsed is not None:
            return parsed
    return None


def inspect_production_tasks(
        tasks, solver_revision, library_revision, cached_outcomes=None):
    """Judge terminal feeder tasks for only the exact pinned generation."""
    cached_outcomes = dict(cached_outcomes or {})
    selected = [
        task for task in tasks
        if is_feeder_task(task, solver_revision, library_revision)
    ]
    outcomes = []
    seen_ids = set()
    for task in selected:
        task_id = task.get("id")
        if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
            raise RuntimeError("production inventory contains a task without a valid ID")
        status = str(task.get("status") or "")
        if status in ACTIVE_STATUSES:
            continue
        if status not in TERMINAL_STATUSES:
            raise RuntimeError(
                f"production task {task_id} has an unknown status: {status!r}")
        seen_ids.add(task_id)
        cached = cached_outcomes.get(str(task_id))
        if (isinstance(cached, dict)
                and cached.get("task_id") == task_id
                and cached.get("name") == task.get("name")
                and cached.get("status") == status):
            outcomes.append(dict(cached))
            continue
        outcome = {
            "task_id": task_id,
            "name": task.get("name"),
            "status": status,
            "state": "invalid",
            "reason": None,
            "error_fingerprint": None,
            "error_message": None,
            "terminal_at": None,
            "saturation_columns": [],
        }
        result = None
        if status == "completed":
            try:
                fetched = scheduler_client.fetch_result(
                    task_id,
                    expected_revision=solver_revision,
                    expected_library_revision=library_revision,
                )
            except scheduler_client.ResultFetchError as exc:
                raise RuntimeError(
                    f"production task {task_id} result is unavailable: {exc}") from exc
            result = fetched.result
            saturated = thermal_saturation_columns(result)
            outcome["saturation_columns"] = saturated
            if (fetched.state == scheduler_client.RESULT_VALID
                    and scheduler_client.is_valid_result(
                        result,
                        expected_revision=solver_revision,
                        expected_library_revision=library_revision)):
                outcome["state"] = "valid"
            else:
                outcome["reason"] = invalid_result_reason(
                    result, solver_revision, library_revision, fetched.state)
        else:
            message = _failure_message(task)
            outcome["reason"] = f"task_{status}"
            outcome["error_message"] = message
            outcome["error_fingerprint"] = error_fingerprint(message)
        terminal_at = _terminal_time(task, result)
        outcome["terminal_at"] = _iso(terminal_at) if terminal_at else None
        outcomes.append(outcome)
        cached_outcomes[str(task_id)] = dict(outcome)
    for key, cached in cached_outcomes.items():
        if not isinstance(cached, dict):
            raise RuntimeError(f"cached production outcome {key!r} is invalid")
        task_id = cached.get("task_id")
        if (isinstance(task_id, bool) or not isinstance(task_id, int)
                or task_id <= 0 or str(task_id) != str(key)
                or cached.get("status") not in TERMINAL_STATUSES
                or cached.get("state") not in ("valid", "invalid")):
            raise RuntimeError(f"cached production outcome {key!r} has invalid identity")
        if task_id in seen_ids:
            continue
        if not is_feeder_task(cached, solver_revision, library_revision):
            raise RuntimeError(
                f"cached production outcome {key!r} has the wrong generation")
        outcomes.append(dict(cached))
    outcomes.sort(key=lambda item: (
        _parse_time(item["terminal_at"]) or datetime.min.replace(tzinfo=timezone.utc),
        item["task_id"],
    ))
    active = sum(
        str(task.get("status") or "") in ACTIVE_STATUSES for task in selected)
    return {
        "tasks": selected,
        "active": active,
        "outcomes": outcomes,
        "cache": cached_outcomes,
    }


def _pilot_path(solver_revision, library_revision, stage, seed, manifest_dir=None):
    contract = pinned_pilot.PILOT_STAGE_CONTRACT[stage]
    tag = pinned_pilot.pilot_tag(
        solver_revision, library_revision, stage, seed, contract["offset"])
    root = (
        Path(manifest_dir) if manifest_dir is not None
        else pinned_pilot.campaign_manifest_dir())
    return root / f"{tag}.json"


def inspect_pilots(solver_revision, library_revision, seed, manifest_dir=None):
    stages = {}
    for stage in ("p02", "p08"):
        path = _pilot_path(
            solver_revision, library_revision, stage, seed, manifest_dir)
        if not path.is_file():
            stages[stage] = {"exists": False, "path": str(path), "outcomes": []}
            continue
        inspected = pinned_pilot.inspect_pilot_stage(
            solver_revision, library_revision, stage, seed,
            manifest_dir=manifest_dir)
        stages[stage] = {
            "exists": True,
            "path": str(inspected["path"]),
            "outcomes": inspected["outcomes"],
        }
    return stages


def _pilot_counts(pilots):
    outcomes = [
        outcome
        for stage in ("p02", "p08")
        for outcome in pilots[stage]["outcomes"]
    ]
    return {
        "valid": sum(item["state"] == "valid" for item in outcomes),
        "invalid": sum(item["state"] == "invalid" for item in outcomes),
        "pending": sum(item["state"] == "pending" for item in outcomes),
        "outcomes": outcomes,
    }


def _production_gate_reasons(production):
    outcomes = production["outcomes"]
    reasons = []
    saturated = [
        outcome for outcome in outcomes if outcome["saturation_columns"]]
    if saturated:
        reasons.append(
            "thermal_saturation_detected:" + ",".join(
                str(item["task_id"]) for item in saturated))
    revision_mismatches = [
        item for item in outcomes
        if item["reason"] in (
            "solver_revision_mismatch", "library_revision_mismatch")]
    if revision_mismatches:
        reasons.append(
            "revision_mismatch_detected:" + ",".join(
                str(item["task_id"]) for item in revision_mismatches))

    fingerprints = Counter(
        item["error_fingerprint"] for item in outcomes
        if item["error_fingerprint"])
    repeated = sorted(
        (fingerprint, count) for fingerprint, count in fingerprints.items()
        if count >= REPEATED_ERROR_LIMIT)
    for fingerprint, count in repeated:
        reasons.append(f"repeated_runtime_error:{fingerprint}:{count}")

    if len(outcomes) >= RECENT_WINDOW:
        recent = outcomes[-RECENT_WINDOW:]
        valid_rate = sum(item["state"] == "valid" for item in recent) / RECENT_WINDOW
        if valid_rate < RECENT_MIN_VALID_RATE:
            reasons.append(
                f"recent_valid_rate_below_70pct:{valid_rate:.3f}")
    return reasons


def _update_progress(state, dataset_rows, production, now):
    previous_rows = state.get("last_dataset_rows")
    if previous_rows is not None and dataset_rows < int(previous_rows):
        return [f"dataset_row_count_regressed:{previous_rows}->{dataset_rows}"]
    if previous_rows is None or dataset_rows > int(previous_rows):
        state["last_dataset_rows"] = int(dataset_rows)
        state["last_dataset_growth_at"] = _iso(now)

    valid = [
        item for item in production["outcomes"] if item["state"] == "valid"]
    if valid:
        valid_times = [
            parsed for parsed in (_parse_time(item["terminal_at"]) for item in valid)
            if parsed is not None]
        earliest = min(valid_times) if valid_times else now
        latest = max(valid_times) if valid_times else now
        if not state.get("first_production_valid_at"):
            state["first_production_valid_at"] = _iso(earliest)
        state["last_production_valid_at"] = _iso(latest)
    if not state.get("first_production_valid_at"):
        return []

    first_valid = _parse_time(state.get("first_production_valid_at"))
    last_growth = _parse_time(state.get("last_dataset_growth_at"))
    anchors = [anchor for anchor in (first_valid, last_growth) if anchor is not None]
    if anchors and (now - max(anchors)).total_seconds() >= DATA_STALL_SECONDS:
        return ["valid_dataset_growth_stalled_90m"]
    return []


def decide_campaign(state, local_passed, pilots, production, dataset_rows, now=None):
    """Apply monotonic promotion and all fail-closed health gates."""
    now = now or _now()
    pilot = _pilot_counts(pilots)
    reasons = []
    if state.get("target_active", 0) >= 50 and not local_passed:
        reasons.append("local3_evidence_missing_after_promotion")
    if state.get("target_active", 0) >= 50 and (
            not pilots["p02"]["exists"] or not pilots["p08"]["exists"]):
        reasons.append("pilot_evidence_missing_after_promotion")
    for outcome in pilot["outcomes"]:
        saturated = thermal_saturation_columns(outcome.get("result"))
        if saturated:
            reasons.append(f"pilot_thermal_saturation:{outcome['task_id']}")
        if outcome["state"] == "invalid":
            reasons.append(
                f"pilot_invalid:{outcome['task_id']}:{outcome.get('reason')}")

    reasons.extend(_production_gate_reasons(production))
    reasons.extend(_update_progress(state, dataset_rows, production, now))

    stage = STAGE_LOCAL3
    target_active = 0
    action = "run_local_gate"
    if local_passed:
        stage = STAGE_PILOT10
        action = "submit_p02" if not pilots["p02"]["exists"] else "wait_p02"
        p02 = pilots["p02"]["outcomes"]
        p02_all_valid = (
            len(p02) == pinned_pilot.PILOT_STAGE_CONTRACT["p02"]["tasks"]
            and all(item["state"] == "valid" for item in p02))
        if p02_all_valid:
            action = "submit_p08" if not pilots["p08"]["exists"] else "wait_pilot10"
        if (p02_all_valid and pilot["valid"] >= PILOT_EARLY_VALID
                and pilot["invalid"] == 0):
            stage = STAGE_FLEET50
            target_active = 50
            action = "refill_50"

    terminal = len(production["outcomes"])
    valid = sum(item["state"] == "valid" for item in production["outcomes"])
    production_valid_rate = valid / terminal if terminal else None
    previous_target = int(state.get("target_active") or 0)
    if (previous_target < 300 and target_active >= 50
            and terminal >= FLEET_GATE_TERMINAL):
        if production_valid_rate >= FLEET_GATE_VALID_RATE:
            stage = STAGE_PRODUCTION300
            target_active = 300
            action = "refill_300"
        else:
            reasons.append(
                f"fleet20_valid_rate_below_90pct:{production_valid_rate:.3f}")

    target_active = max(target_active, previous_target)
    if target_active >= 300:
        stage = STAGE_PRODUCTION300
        action = "refill_300"
    elif target_active >= 50:
        stage = STAGE_FLEET50
        action = "refill_50"

    state["stage"] = stage
    state["target_active"] = target_active
    if reasons:
        state["paused"] = True
        state["pause_reasons"] = sorted(set(
            [*state.get("pause_reasons", []), *reasons]))
    if state.get("paused"):
        action = "manual_intervention"
    return {
        "stage": stage,
        "stage_target": STAGE_TARGETS[stage],
        "target_active": target_active,
        "action": action,
        "paused": bool(state.get("paused")),
        "pause_reasons": list(state.get("pause_reasons", [])),
        "dataset_rows": int(dataset_rows),
        "pilot": {
            "valid": pilot["valid"],
            "invalid": pilot["invalid"],
            "pending": pilot["pending"],
        },
        "production": {
            "active": production["active"],
            "terminal": terminal,
            "valid": valid,
            "invalid": terminal - valid,
            "valid_rate": production_valid_rate,
        },
    }


def _validate_pinned_local_revisions(solver_revision, library_revision, library_root=None):
    solver_revision = provisional_wave._validate_sha(
        solver_revision, "solver revision")
    library_revision = provisional_wave._validate_sha(
        library_revision, "library revision")
    provisional_wave._validate_local_revisions(
        solver_revision, library_revision, library_root=library_root)
    return solver_revision, library_revision


def run_once(
        solver_revision, library_revision, seed=DEFAULT_SEED,
        max_samples=DEFAULT_MAX_SAMPLES, execute=False, clear_pause=False,
        library_root=None, state_path=None, manifest_dir=None, now=None):
    """Observe once and optionally perform exactly one safe promotion/refill step."""
    now = now or _now()
    solver_revision, library_revision = _validate_pinned_local_revisions(
        solver_revision, library_revision, library_root=library_root)
    path = Path(state_path) if state_path else default_state_path(
        solver_revision, library_revision, seed)
    if execute:
        path.parent.mkdir(parents=True, exist_ok=True)
    state_lock = (
        FileLock(str(path) + ".lock", timeout=5) if execute else nullcontext())
    with state_lock:
        state = load_state(path, solver_revision, library_revision, seed)
        if clear_pause:
            if not execute:
                raise RuntimeError("--clear-pause requires --execute")
            state["paused"] = False
            state["pause_reasons"] = []

        if execute and not state.get("candidate_audit"):
            state["candidate_audit"] = candidate_supply_audit(
                solver_revision, library_revision, seed)

        local_manifest_dir = manifest_dir
        local_path = (
            Path(local_manifest_dir)
            if local_manifest_dir is not None
            else pinned_pilot.campaign_manifest_dir()) / (
                f"{pinned_pilot.local_gate_tag(solver_revision, library_revision)}.json")
        local_passed = False
        if local_path.is_file():
            pinned_pilot.validate_local_gate(
                solver_revision, library_revision,
                manifest_dir=local_path.parent)
            local_passed = True

        pilots = inspect_pilots(
            solver_revision, library_revision, seed,
            manifest_dir=manifest_dir)
        inventory = feeder.campaign_inventory()
        production = inspect_production_tasks(
            inventory, solver_revision, library_revision,
            cached_outcomes=state.get("task_outcomes"))
        state["task_outcomes"] = production["cache"]
        dataset_rows, judged_ids = feeder.dataset_collection_snapshot()
        decision = decide_campaign(
            state, local_passed, pilots, production, dataset_rows, now=now)

        mutation = None
        if execute and not decision["paused"]:
            if decision["action"] == "submit_p02":
                result = pinned_pilot.submit_pilot_stage(
                    solver_revision, library_revision, "p02", seed=seed,
                    execute=True, manifest_dir=manifest_dir)
                mutation = {
                    "submitted": "p02",
                    "task_ids": [
                        record["task_id"] for record in result["manifest"]["tasks"]],
                }
            elif decision["action"] == "submit_p08":
                result = pinned_pilot.submit_pilot_stage(
                    solver_revision, library_revision, "p08", seed=seed,
                    execute=True, manifest_dir=manifest_dir)
                mutation = {
                    "submitted": "p08",
                    "task_ids": [
                        record["task_id"] for record in result["manifest"]["tasks"]],
                }
            elif decision["target_active"] >= 50:
                feeder.step(
                    max_samples,
                    target=decision["target_active"],
                    buffer=0,
                    solver_revision=solver_revision,
                    library_revision=library_revision,
                    candidate_seed=seed,
                )
                mutation = {
                    "refill_target": decision["target_active"],
                    "max_samples": int(max_samples),
                }
            save_state(state, path)

        return {
            "schema_version": SCHEMA_VERSION,
            "mode": "execute" if execute else "read_only",
            "solver_revision": solver_revision,
            "library_revision": library_revision,
            "seed": int(seed),
            "state_path": str(path),
            "local3_passed": local_passed,
            "candidate_audit": state.get("candidate_audit"),
            "collector_judged_tasks": len(judged_ids),
            "mutation": mutation,
            **decision,
        }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fail-closed 3 -> 10 -> 50 -> 300 MFT campaign controller")
    parser.add_argument("--solver-revision", required=True)
    parser.add_argument("--library-revision", required=True)
    parser.add_argument("--library-root")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--state-path")
    parser.add_argument("--manifest-dir")
    parser.add_argument(
        "--execute", action="store_true",
        help="allow pinned pilot submission or feeder refill; never cancels")
    parser.add_argument(
        "--clear-pause", action="store_true",
        help="explicitly clear a latched pause before re-evaluating all gates")
    parser.add_argument(
        "--loop", type=int, default=None,
        help="repeat every N seconds for continuous refill (requires --execute)")
    args = parser.parse_args(argv)
    if args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.loop is not None and args.loop <= 0:
        parser.error("--loop must be positive")
    if args.loop is not None and not args.execute:
        parser.error("--loop requires --execute")
    if args.execute:
        if not args.library_root:
            parser.error("--execute requires --library-root for remote deployment validation")
        deployment_gate.validate_deployment(
            REPO_ROOT, args.solver_revision,
            args.library_root, args.library_revision,
        )

    while True:
        try:
            result = run_once(
                args.solver_revision,
                args.library_revision,
                seed=args.seed,
                max_samples=args.max_samples,
                execute=args.execute,
                clear_pause=args.clear_pause,
                library_root=args.library_root,
                state_path=args.state_path,
                manifest_dir=args.manifest_dir,
            )
            print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        except Exception as exc:
            # No feeder or submission call can occur after an observation error.
            print(json.dumps({
                "mode": "execute" if args.execute else "read_only",
                "paused": True,
                "action": "manual_intervention",
                "pause_reasons": [f"controller_error:{exc}"],
            }, ensure_ascii=False, sort_keys=True), flush=True)
            if args.loop is None:
                raise
        if args.loop is None:
            return
        args.clear_pause = False
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
