"""Plan or cancel one isolated 300-task provisional generation.

Direct submission from this legacy entry point is intentionally disabled.  Use
rapid_campaign.py --execute so the p02/p08 gates and staged capacity controls
remain in force.  Cancellation always uses explicit task IDs from the ledger.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import requests


HERE = Path(__file__).resolve().parent
REGRESSION_ROOT = HERE.parent
REPO_ROOT = REGRESSION_ROOT.parent
sys.path.insert(0, str(REGRESSION_ROOT))
sys.path.insert(0, str(REGRESSION_ROOT / "verify"))

import al_driver
import pinned_pilot
import scheduler_client


SCHEDULER = "http://127.0.0.1:8000"
TASK_COUNT = 300
SEED = 260710
CPUS = 4
MEMORY_MB = 32768
ACTIVE_STATUSES = ("queued", "attaching", "running")
SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
PROFILE_PATH = REGRESSION_ROOT / "verify" / "profiles" / "standard.json"
STATE_DIR_ENV = "MFT_PROVISIONAL_STATE_DIR"
DIRECT_SUBMISSION_DISABLED = (
    "direct provisional 300-task submission is disabled; "
    "use rapid_campaign.py --execute"
)


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _validate_sha(value, label):
    revision = str(value or "").strip().lower()
    if not SHA_PATTERN.fullmatch(revision):
        raise ValueError(f"{label} must be a full 40-character git SHA")
    return revision


def generation_tag(solver_revision, library_revision, seed=SEED):
    return f"provisional300-s{solver_revision[:12]}-l{library_revision[:12]}-seed{seed}"


def task_prefix(solver_revision, library_revision):
    return f"mft-camp-s{solver_revision[:7]}-l{library_revision[:7]}-prov-"


def default_manifest_dir():
    override = os.environ.get(STATE_DIR_ENV, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        if not local_app_data:
            raise RuntimeError("LOCALAPPDATA is required for provisional state on Windows")
        return Path(local_app_data) / "MFT_1MW_2026" / "provisional_manifests"
    return HERE / "provisional_manifests"


def manifest_path(solver_revision, library_revision, seed=SEED, manifest_dir=None):
    root = Path(manifest_dir) if manifest_dir is not None else default_manifest_dir()
    return root / f"{generation_tag(solver_revision, library_revision, seed)}.json"


def _expected_dedupe_key(name, params, profile, solver_revision, library_revision):
    merged = dict(params)
    merged.update(profile.get("param_overrides", {}))
    payload = json.dumps(merged, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"mft-al:{name}:{solver_revision}:{library_revision}:{digest}"


def build_plan(solver_revision, library_revision, profile, seed=SEED, count=TASK_COUNT):
    prefix = task_prefix(solver_revision, library_revision)
    cursor = pinned_pilot.cursor_after_valid_candidates(
        pinned_pilot.PILOT_RESERVED_VALID_CANDIDATES, seed=seed)
    records = []
    for index in range(count):
        cursor, raw_index, params = pinned_pilot.next_valid_candidate(cursor, seed=seed)
        name = f"{prefix}{index:03d}"
        records.append({
            "index": index,
            "name": name,
            "candidate_raw_index": raw_index,
            "params_sha256": pinned_pilot.candidate_digest(
                pinned_pilot.effective_candidate(params)),
            "dedupe_key": _expected_dedupe_key(
                name, params, profile, solver_revision, library_revision),
            "params": params,
            "task_id": None,
        })
    return records


def new_manifest(solver_revision, library_revision, profile, records, seed=SEED):
    return {
        "tag": generation_tag(solver_revision, library_revision, seed),
        "solver_revision": solver_revision,
        "library_revision": library_revision,
        "seed": seed,
        "task_count": len(records),
        "task_prefix": task_prefix(solver_revision, library_revision),
        "remote_cwd": scheduler_client.GPFS_RUNS_REMOTE_CWD,
        "profile": profile,
        "created_at": _now(),
        "updated_at": _now(),
        "tasks": records,
    }


def _plan_identity(record):
    return {
        key: record.get(key)
        for key in (
            "index", "name", "candidate_raw_index", "params_sha256",
            "dedupe_key", "params",
        )
    }


def validate_manifest(manifest, expected, require_all_task_ids=False):
    for key in (
            "tag", "solver_revision", "library_revision", "seed", "task_count",
            "task_prefix", "remote_cwd", "profile"):
        if manifest.get(key) != expected.get(key):
            raise RuntimeError(f"provisional ledger identity mismatch for {key}")
    actual_tasks = manifest.get("tasks")
    expected_tasks = expected.get("tasks")
    if not isinstance(actual_tasks, list) or len(actual_tasks) != len(expected_tasks):
        raise RuntimeError("provisional ledger does not contain exactly 300 task records")
    for index, (actual, planned) in enumerate(zip(actual_tasks, expected_tasks)):
        if _plan_identity(actual) != _plan_identity(planned):
            raise RuntimeError(f"provisional ledger candidate {index} changed")
        task_id = actual.get("task_id")
        if task_id is not None and (
                isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0):
            raise RuntimeError(f"provisional ledger task {index} has an invalid task ID")
        if require_all_task_ids and task_id is None:
            raise RuntimeError(f"provisional ledger task {index} has no task ID")
    task_ids = [record.get("task_id") for record in actual_tasks if record.get("task_id")]
    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError("provisional ledger contains duplicate task IDs")
    return manifest


def _load_or_initialize_ledger(expected, path):
    if path.is_file():
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise RuntimeError(f"provisional ledger is unreadable: {path}: {exc}") from exc
        return validate_manifest(manifest, expected)
    pinned_pilot._atomic_manifest(expected, path)
    return expected


def _scheduler_json(path, params=None):
    response = requests.get(f"{SCHEDULER}{path}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def _task_inventory(prefix):
    payload = _scheduler_json(
        "/api/tasks", params={"limit": 10000, "name_prefix": prefix})
    tasks = payload if isinstance(payload, list) else payload.get("tasks")
    if not isinstance(tasks, list):
        raise RuntimeError("scheduler returned an invalid provisional task inventory")
    return [task for task in tasks if str(task.get("name") or "").startswith(prefix)]


def _validate_scheduler_task(task, expected_record):
    task_id = task.get("id")
    if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
        raise RuntimeError("scheduler task has no valid ID")
    for key in ("name", "dedupe_key"):
        if task.get(key) != expected_record[key]:
            raise RuntimeError(
                f"scheduler task {task_id} does not match provisional {key}")
    if task.get("remote_cwd") != scheduler_client.GPFS_RUNS_REMOTE_CWD:
        raise RuntimeError(f"scheduler task {task_id} is outside the runs workspace")
    return task_id


def _validate_inventory(tasks, records):
    expected_by_name = {record["name"]: record for record in records}
    seen_names = set()
    mapped = {}
    for task in tasks:
        name = task.get("name")
        if name not in expected_by_name:
            raise RuntimeError(f"unexpected task in provisional prefix: {name!r}")
        if name in seen_names:
            raise RuntimeError(f"duplicate scheduler task name in provisional prefix: {name}")
        seen_names.add(name)
        task_id = _validate_scheduler_task(task, expected_by_name[name])
        mapped[name] = task_id
    return mapped


def _validate_local_revisions(solver_revision, library_revision, library_root=None):
    if solver_revision != al_driver._current_solver_revision():
        raise RuntimeError("solver revision is not the current vetted local solver")
    previous = os.environ.get("MFT_PYAEDT_LIBRARY_ROOT")
    if library_root:
        os.environ["MFT_PYAEDT_LIBRARY_ROOT"] = str(Path(library_root).resolve())
    try:
        current_library = al_driver._current_library_revision()
    finally:
        if previous is None:
            os.environ.pop("MFT_PYAEDT_LIBRARY_ROOT", None)
        else:
            os.environ["MFT_PYAEDT_LIBRARY_ROOT"] = previous
    if library_revision != current_library:
        raise RuntimeError("library revision is not the current clean local library")


def submit_generation(
        solver_revision, library_revision, profile, path, library_root=None,
        seed=SEED, count=TASK_COUNT):
    raise RuntimeError(DIRECT_SUBMISSION_DISABLED)


def cancel_generation(manifest, expected, batch_size=100):
    validate_manifest(manifest, expected)
    records = manifest["tasks"]
    tasks = _task_inventory(manifest["task_prefix"])
    mapped = _validate_inventory(tasks, records)
    ledger_ids = {
        int(record["task_id"])
        for record in records if record.get("task_id") is not None
    }
    inventory_ids = set(mapped.values())
    if not inventory_ids.issubset(ledger_ids):
        missing = sorted(inventory_ids - ledger_ids)
        raise RuntimeError(f"scheduler tasks are absent from the exact ledger: {missing}")

    active_ids = sorted(
        int(task["id"]) for task in tasks
        if task.get("status") in ACTIVE_STATUSES)
    cancelled = []
    for start in range(0, len(active_ids), batch_size):
        batch = active_ids[start:start + batch_size]
        response = requests.post(
            f"{SCHEDULER}/api/tasks/cancel",
            params={
                "statuses": ",".join(ACTIVE_STATUSES),
                "task_ids": ",".join(str(task_id) for task_id in batch),
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        returned = payload.get("cancelled") if isinstance(payload, dict) else None
        if not isinstance(returned, list) or not set(returned).issubset(set(batch)):
            raise RuntimeError("scheduler returned invalid explicit cancellation IDs")
        cancelled.extend(int(task_id) for task_id in returned)
    return {"active": active_ids, "cancelled": sorted(set(cancelled))}


def _load_profile():
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    if profile.get("param_overrides") != scheduler_client.STANDARD_PROFILE_CONTRACT:
        raise RuntimeError("standard profile differs from scheduler client contract")
    if scheduler_client.GPFS_RUNS_REMOTE_CWD != "__SLURM_SCHEDULER_ACCOUNT_WORKSPACE__/runs":
        raise RuntimeError("provisional submissions are not pinned to the runs workspace")
    return profile


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver-revision", required=True)
    parser.add_argument("--library-revision", required=True)
    parser.add_argument("--library-root")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--manifest-dir")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--cancel", action="store_true")
    parser.add_argument(
        "--acknowledge-provisional", action="store_true",
        help="required for cancellation; direct execution is disabled")
    args = parser.parse_args(argv)

    try:
        solver_revision = _validate_sha(args.solver_revision, "solver revision")
        library_revision = _validate_sha(args.library_revision, "library revision")
    except ValueError as exc:
        parser.error(str(exc))
    if args.execute:
        parser.error(DIRECT_SUBMISSION_DISABLED)
    if args.cancel and not args.acknowledge_provisional:
        parser.error("--acknowledge-provisional is required for cancel")

    profile = _load_profile()
    records = build_plan(
        solver_revision, library_revision, profile, args.seed, TASK_COUNT)
    expected = new_manifest(
        solver_revision, library_revision, profile, records, args.seed)
    path = manifest_path(
        solver_revision, library_revision, args.seed, args.manifest_dir)

    if args.cancel:
        if not path.is_file():
            raise RuntimeError(f"exact provisional ledger is unavailable: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        result = cancel_generation(manifest, expected)
        print(json.dumps({
            "mode": "cancel", "manifest": str(path), **result,
        }, ensure_ascii=False))
        return


    print(json.dumps({
        "mode": "plan", "manifest": str(path),
        "solver_revision": solver_revision,
        "library_revision": library_revision,
        "task_prefix": expected["task_prefix"],
        "remote_cwd": expected["remote_cwd"],
        "task_count": len(records),
        "first": _plan_identity(records[0]),
        "last": _plan_identity(records[-1]),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
