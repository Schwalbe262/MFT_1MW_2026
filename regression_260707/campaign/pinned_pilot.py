"""Submit exactly 2 or 8 isolated, revision-pinned campaign pilot samples."""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import requests
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
REGRESSION_ROOT = HERE.parent
REPO_ROOT = REGRESSION_ROOT.parent
sys.path.insert(0, str(REGRESSION_ROOT))
sys.path.insert(0, str(REGRESSION_ROOT / "verify"))
sys.path.insert(0, str(REPO_ROOT))

import al_driver
import deployment_gate
import scheduler_client
from module.input_parameter_260706 import (
    KEYS,
    _SOBOL_DIMS,
    create_input_parameter,
    decode_unit_sample,
    unit_to_dims,
    validation_check,
)

SCHEDULER = "http://127.0.0.1:8000"
CPUS = 4
MEMORY_MB = 32768
CPU_HEADROOM = 0.85
ACTIVE_STATUSES = ("queued", "attaching", "running")
QUEUE_STATES = frozenset(("ready", "pending", "opening", "blocked"))
LEGACY_MFT_NAME_PREFIX = "mft-"
PILOT_STAGE_CONTRACT = {
    "p02": {"tasks": 2, "offset": 0},
    "p08": {"tasks": 8, "offset": 2},
}
LOCAL_GATE_COUNT = 3
PILOT_RESERVED_VALID_CANDIDATES = sum(
    contract["tasks"] for contract in PILOT_STAGE_CONTRACT.values())
MFT_PROJECT = scheduler_client.MFT_PROJECT
MFT_PROJECT_MAX_ACTIVE_TASKS = scheduler_client.MFT_PROJECT_MAX_ACTIVE_TASKS
PILOT_PROJECT_HARD_CAP = PILOT_RESERVED_VALID_CANDIDATES
_LOCALAPPDATA = os.environ.get("LOCALAPPDATA", "").strip()
if not _LOCALAPPDATA:
    _LOCALAPPDATA = str(Path.home() / "AppData" / "Local")
CAMPAIGN_MUTATION_LOCK_PATH = (
    Path(_LOCALAPPDATA) / "MFT_1MW_2026" / "campaign-mutation.lock")
CAMPAIGN_MUTATION_LOCK_TIMEOUT = 15 * 60


def campaign_mutation_lock():
    """Return the one host-wide lock shared by every campaign submit path."""
    return scheduler_client.campaign_mutation_lock(
        path=CAMPAIGN_MUTATION_LOCK_PATH,
        timeout=CAMPAIGN_MUTATION_LOCK_TIMEOUT,
    )


def _get_json(path, params=None):
    response = requests.get(f"{SCHEDULER}{path}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def queue_allows_demand_submission(queue_state):
    """Allow demand to enter the scheduler unless its fit contract is blocked."""
    normalized = str(queue_state or "").strip().lower()
    if normalized not in QUEUE_STATES:
        raise RuntimeError(f"scheduler returned invalid queue_state: {queue_state!r}")
    return normalized != "blocked"


def project_submission_snapshot(
        projects, project_tasks, required_hard_cap, legacy_tasks=None):
    """Return the absolute project+legacy MFT demand budget without double count."""
    if isinstance(required_hard_cap, bool):
        raise RuntimeError("MFT project hard cap must be a positive integer")
    try:
        required_hard_cap = int(required_hard_cap)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("MFT project hard cap must be a positive integer") from exc
    if required_hard_cap < 1:
        raise RuntimeError("MFT project hard cap must be a positive integer")
    if required_hard_cap > MFT_PROJECT_MAX_ACTIVE_TASKS:
        raise RuntimeError(
            f"MFT project stage cap {required_hard_cap} exceeds absolute cap "
            f"{MFT_PROJECT_MAX_ACTIVE_TASKS}")
    if not isinstance(projects, list):
        raise RuntimeError("scheduler returned an invalid project inventory")
    matches = [
        project for project in projects
        if isinstance(project, dict)
        and str(project.get("name") or "").strip() == MFT_PROJECT
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"scheduler project {MFT_PROJECT!r} is missing or ambiguous")
    try:
        project_contract = scheduler_client.validate_project_mutation_contract(
            matches[0])
    except scheduler_client.ProjectContractError as exc:
        raise RuntimeError(str(exc)) from exc
    server_cap = project_contract["max_active_tasks"]
    if not isinstance(project_tasks, list):
        raise RuntimeError("scheduler returned an invalid MFT project task inventory")
    if legacy_tasks is None:
        legacy_tasks = []
    if not isinstance(legacy_tasks, list):
        raise RuntimeError("scheduler returned an invalid legacy MFT task inventory")

    def indexed(rows, source, allowed_projects, require_prefix=False):
        inventory = {}
        for task in rows:
            if not isinstance(task, dict):
                raise RuntimeError(
                    f"scheduler returned an invalid {source} task")
            task_id = task.get("id", task.get("task_id"))
            if (isinstance(task_id, bool) or not isinstance(task_id, int)
                    or task_id <= 0):
                raise RuntimeError(
                    f"scheduler returned an invalid {source} task ID")
            if task_id in inventory:
                raise RuntimeError(
                    f"scheduler returned duplicate {source} task ID {task_id}")
            project_name = str(task.get("project") or "").strip()
            if project_name not in allowed_projects:
                raise RuntimeError(
                    f"scheduler returned {source} task {task_id} from "
                    f"unexpected project {project_name!r}")
            status = str(task.get("status") or "").strip().lower()
            if status not in ACTIVE_STATUSES:
                raise RuntimeError(
                    f"scheduler returned unexpected {source} active task status: "
                    f"{status!r}")
            if (require_prefix
                    and not str(task.get("name") or "").startswith(
                        LEGACY_MFT_NAME_PREFIX)):
                raise RuntimeError(
                    f"scheduler legacy MFT filter returned task {task_id} "
                    "outside the mft- namespace")
            inventory[task_id] = dict(task)
        return inventory

    tagged = indexed(
        project_tasks, "MFT project", {MFT_PROJECT})
    legacy_scan = indexed(
        legacy_tasks, "legacy MFT", {"", MFT_PROJECT}, require_prefix=True)
    combined = dict(tagged)
    for task_id, task in legacy_scan.items():
        if task_id in combined:
            if str(task.get("project") or "").strip() != MFT_PROJECT:
                raise RuntimeError(
                    f"scheduler task {task_id} is both project-tagged and legacy")
            continue
        combined[task_id] = task

    counts = {status: 0 for status in ACTIVE_STATUSES}
    tagged_counts = {status: 0 for status in ACTIVE_STATUSES}
    legacy_counts = {status: 0 for status in ACTIVE_STATUSES}
    for task in combined.values():
        if not isinstance(task, dict):
            raise RuntimeError("scheduler returned an invalid MFT project task")
        status = str(task.get("status") or "").strip().lower()
        counts[status] += 1
        bucket = (
            tagged_counts
            if str(task.get("project") or "").strip() == MFT_PROJECT
            else legacy_counts
        )
        bucket[status] += 1
    active = sum(counts.values())
    server_open_slots = max(0, server_cap - active)
    stage_open_slots = max(0, required_hard_cap - active)
    return {
        "project": MFT_PROJECT,
        "project_max_active_tasks": server_cap,
        "project_required_hard_cap": required_hard_cap,
        "project_counts": counts,
        "project_tagged_counts": tagged_counts,
        "legacy_counts": legacy_counts,
        "project_active": active,
        "project_tagged_active": sum(tagged_counts.values()),
        "legacy_active": sum(legacy_counts.values()),
        "project_server_open_slots": server_open_slots,
        "project_stage_open_slots": stage_open_slots,
        "project_submission_slots": min(server_open_slots, stage_open_slots),
    }


def calculate_submission_headroom(
        statuses, allocations, ready_fit_slots, queue_state="ready", queue_reason=""):
    """Return immediate-capacity telemetry plus the queued-demand admission gate."""
    global_active = sum(int(statuses.get(status, 0) or 0) for status in ACTIVE_STATUSES)
    usable = [
        allocation for allocation in allocations
        if allocation.get("state") in ("active", "warm")
        and allocation.get("resource_pool", "cpu") == "cpu"
    ]
    total_cpus = sum(max(0, int(a.get("total_cpus") or 0)) for a in usable)
    free_cpus = sum(max(0, int(a.get("free_cpus") or 0)) for a in usable)
    total_slots = math.floor(total_cpus / CPUS * CPU_HEADROOM)
    free_slots = math.floor(free_cpus / CPUS * CPU_HEADROOM)
    headroom = max(0, min(
        free_slots,
        total_slots - global_active,
        max(0, int(ready_fit_slots or 0)),
    ))
    queue_submission_allowed = queue_allows_demand_submission(queue_state)
    return {
        "global_active": global_active,
        "total_cpus": total_cpus,
        "free_cpus": free_cpus,
        "total_slots": total_slots,
        "free_slots": free_slots,
        "ready_fit_slots": int(ready_fit_slots or 0),
        "headroom": headroom,
        "queue_state": str(queue_state or "").strip().lower(),
        "queue_reason": str(queue_reason or "").strip(),
        "queue_submission_allowed": queue_submission_allowed,
        "submission_allowed": queue_submission_allowed,
    }


def capacity_snapshot(required_hard_cap=PILOT_PROJECT_HARD_CAP):
    summary = _get_json("/api/tasks/summary")
    allocations = _get_json("/api/allocations")
    projects = _get_json("/api/projects")
    project_tasks = _get_json("/api/tasks", params={
        "limit": 10000,
        "project": MFT_PROJECT,
        "status": ",".join(ACTIVE_STATUSES),
    })
    legacy_tasks = _get_json("/api/tasks", params={
        "limit": 10000,
        "name_prefix": LEGACY_MFT_NAME_PREFIX,
        "status": ",".join(ACTIVE_STATUSES),
    })
    capacity = _get_json("/api/task-capacity", params={
        "cpus": CPUS,
        "memory_mb": MEMORY_MB,
        "scheduling_profile": "fea_bursty",
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
    })
    if not isinstance(summary, dict) or not isinstance(allocations, list) \
            or not isinstance(capacity, dict):
        raise RuntimeError("scheduler returned an invalid capacity snapshot")
    snapshot = calculate_submission_headroom(
        summary.get("statuses") or {}, allocations, capacity.get("ready_fit_slots"),
        queue_state=capacity.get("queue_state"),
        queue_reason=capacity.get("queue_reason"),
    )
    snapshot.update(project_submission_snapshot(
        projects, project_tasks, required_hard_cap, legacy_tasks=legacy_tasks))
    snapshot["submission_allowed"] = bool(
        snapshot["queue_submission_allowed"]
        and snapshot["project_submission_slots"] > 0
    )
    return snapshot


def deterministic_candidate_at(index, seed=260710):
    if index < 0:
        raise ValueError("candidate index must be non-negative")
    engine = qmc.Sobol(d=len(_SOBOL_DIMS), scramble=True, seed=seed)
    if index:
        engine.fast_forward(index)
    unit = engine.random(1)[0]
    decoded = decode_unit_sample(unit_to_dims(unit), allow_space_shrink=True)
    params = {key: decoded[key] for key in KEYS}
    valid, expanded, errors = validation_check(
        create_input_parameter(params), return_errors=True)
    if not valid:
        raise RuntimeError("deterministic pilot candidate is invalid: " + " / ".join(errors))
    row = expanded.iloc[0]
    return {
        key: (row[key].item() if isinstance(row[key], np.generic) else row[key])
        for key in KEYS
    }


def next_valid_candidate(cursor=0, seed=260710, max_attempts=1000):
    for raw_index in range(cursor, cursor + max_attempts):
        try:
            return raw_index + 1, raw_index, deterministic_candidate_at(raw_index, seed)
        except RuntimeError:
            continue
    raise RuntimeError(
        f"no valid deterministic candidate in raw Sobol range {cursor}:{cursor + max_attempts}")


def cursor_after_valid_candidates(count, seed=260710):
    cursor = 0
    for _ in range(count):
        cursor, _, _ = next_valid_candidate(cursor, seed=seed)
    return cursor


def deterministic_candidates(count, offset=0, seed=260710):
    candidates = []
    cursor = 0
    valid_seen = 0
    while len(candidates) < count:
        cursor, _, candidate = next_valid_candidate(cursor, seed)
        if valid_seen >= offset:
            candidates.append(candidate)
        valid_seen += 1
    return candidates


def candidate_digest(params):
    return hashlib.sha256(json.dumps(
    params, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def effective_candidate(params):
    effective = dict(params)
    effective.update(scheduler_client.STANDARD_PROFILE_CONTRACT)
    return effective


def result_matches_candidate(result, params):
    """Require every intended input value to be echoed by the solver result."""
    normalized = {
        key: (value.item() if isinstance(value, np.generic) else value)
        for key, value in params.items()
    }
    return scheduler_client.result_matches_params(result, normalized)


def resolve_stage_contract(stage, tasks, offset=None):
    """Return the only allowed offset for a pilot stage."""
    contract = PILOT_STAGE_CONTRACT.get(stage)
    if contract is None:
        raise ValueError(f"unknown pilot stage: {stage}")
    if tasks != contract["tasks"]:
        raise ValueError(
            f"{stage} requires exactly {contract['tasks']} tasks, got {tasks}")
    resolved = contract["offset"] if offset is None else offset
    if resolved != contract["offset"]:
        raise ValueError(
            f"{stage} requires offset {contract['offset']}, got {resolved}")
    return resolved


def pilot_tag(solver_revision, library_revision, stage, seed, offset):
    """Build a collision-resistant pilot identity for names and manifests."""
    return (
        f"s{solver_revision[:7]}-l{library_revision[:7]}-"
        f"{stage}-seed{seed}-o{offset}"
    )


def campaign_manifest_dir():
    """Return one shared manifest directory across linked solver worktrees."""
    override = os.environ.get("MFT_CAMPAIGN_STATE_DIR", "").strip()
    if override:
        return Path(override).resolve()
    try:
        common_git = subprocess.check_output(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True).strip()
        common_root = Path(common_git).resolve().parent
        return common_root / "regression_260707" / "campaign" / "pilot_manifests"
    except (OSError, subprocess.SubprocessError):
        return HERE / "pilot_manifests"


def local_gate_tag(solver_revision, library_revision):
    return f"local3-s{solver_revision[:12]}-l{library_revision[:12]}"


def local_gate_profile_contract():
    contract = dict(scheduler_client.STANDARD_PROFILE_CONTRACT)
    contract["keep_project"] = 1
    return contract


def validate_local_gate(solver_revision, library_revision, manifest_dir=None):
    """Require three consecutive exact-profile local results for this source pair."""
    manifest_dir = Path(manifest_dir) if manifest_dir is not None else campaign_manifest_dir()
    tag = local_gate_tag(solver_revision, library_revision)
    path = manifest_dir / f"{tag}.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f"p02 requires a readable local3 manifest: {path}: {exc}") from exc
    expected = {
        "tag": tag,
        "solver_revision": solver_revision,
        "library_revision": library_revision,
        "sample_count": LOCAL_GATE_COUNT,
        "passed": True,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items() if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"local3 manifest identity is invalid: {mismatches}")
    results = manifest.get("results")
    if not isinstance(results, list) or len(results) != LOCAL_GATE_COUNT:
        raise RuntimeError("local3 manifest must contain exactly three results")
    projects = []
    profile = local_gate_profile_contract()
    for index, result in enumerate(results):
        if not scheduler_client.is_valid_result(
                result, expected_revision=solver_revision,
                expected_library_revision=library_revision,
                expected_profile=profile):
            raise RuntimeError(f"local3 result {index} is not strict-valid")
        if result.get("matrix_extraction_backend") != "export_rl_matrix":
            raise RuntimeError(f"local3 result {index} did not use export_rl_matrix")
        for label in ("matrix", "loss"):
            try:
                attempts = int(float(result[f"{label}_solve_attempts"]))
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(
                    f"local3 result {index} has no {label} solve-attempt telemetry") from exc
            if attempts != 1:
                raise RuntimeError(
                    f"local3 result {index} used {attempts} {label} solves instead of one")
        projects.append(str(result["project_name"]))
    if len(set(projects)) != LOCAL_GATE_COUNT:
        raise RuntimeError("local3 manifest contains duplicate project identities")
    return {"manifest": str(path), "tag": tag, "projects": projects}


def load_pilot_stage_manifest(
        solver_revision, library_revision, stage, seed=260710,
        manifest_dir=None, required_by=None):
    """Load and authenticate a submitted pilot ledger without judging results.

    This is the shared boundary used by the strict predecessor gates and by the
    rapid campaign controller.  Keeping candidate identity checks here prevents
    an early-promotion observer from accepting a hand-edited or cross-revision
    manifest.
    """
    contract = PILOT_STAGE_CONTRACT.get(stage)
    if contract is None:
        raise ValueError(f"unknown pilot stage: {stage}")
    manifest_dir = (
        Path(manifest_dir) if manifest_dir is not None else campaign_manifest_dir())
    tag = pilot_tag(
        solver_revision, library_revision, stage, seed, contract["offset"])
    path = manifest_dir / f"{tag}.json"
    consumer = required_by or "campaign controller"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(
            f"{consumer} requires a readable {stage} manifest: {path}: {exc}") from exc

    expected = {
        "tag": tag,
        "stage": stage,
        "solver_revision": solver_revision,
        "library_revision": library_revision,
        "seed": seed,
        "offset": contract["offset"],
        "task_count": contract["tasks"],
        "executed": True,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items() if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"{stage} manifest identity is invalid: {mismatches}")

    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != contract["tasks"]:
        raise RuntimeError(
            f"{stage} manifest must contain exactly {contract['tasks']} task records")
    candidates = [
        effective_candidate(candidate)
        for candidate in deterministic_candidates(
            contract["tasks"], offset=contract["offset"], seed=seed)
    ]
    task_ids = []
    task_names = []
    for index, record in enumerate(tasks):
        expected_name = f"mft-pilot-{tag}-{index:02d}"
        if (not isinstance(record, dict) or record.get("index") != index
                or record.get("name") != expected_name):
            raise RuntimeError(f"{stage} manifest task {index} has an invalid identity")
        if record.get("params_sha256") != candidate_digest(candidates[index]):
            raise RuntimeError(
                f"{stage} manifest task {index} has an invalid parameter digest")
        task_id = record.get("task_id")
        if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
            raise RuntimeError(f"{stage} manifest task {index} has no valid task ID")
        task_ids.append(task_id)
        task_names.append(expected_name)
    if len(set(task_ids)) != len(task_ids):
        raise RuntimeError(f"{stage} manifest contains duplicate task IDs")
    return {
        "path": path,
        "manifest": manifest,
        "tag": tag,
        "tasks": tasks,
        "task_ids": task_ids,
        "task_names": task_names,
        "candidates": candidates,
    }


def inspect_pilot_stage(
        solver_revision, library_revision, stage, seed=260710,
        manifest_dir=None):
    """Return strict, physical outcomes while a pilot stage is still running."""
    loaded = load_pilot_stage_manifest(
        solver_revision, library_revision, stage, seed, manifest_dir=manifest_dir)
    outcomes = []
    for index, task_id in enumerate(loaded["task_ids"]):
        status = scheduler_client.get_status(task_id)
        if status is None:
            raise RuntimeError(f"scheduler status is unavailable for {stage} task {task_id}")
        outcome = {
            "index": index,
            "task_id": task_id,
            "name": loaded["task_names"][index],
            "status": status,
            "state": "pending",
            "reason": None,
            "result": None,
        }
        if status in ("failed", "cancelled"):
            outcome.update(state="invalid", reason=f"task_{status}")
        elif status == "completed":
            try:
                fetched = scheduler_client.fetch_result(
                    task_id,
                    expected_revision=solver_revision,
                    expected_library_revision=library_revision,
                )
            except scheduler_client.ResultFetchError as exc:
                raise RuntimeError(
                    f"{stage} task {task_id} result is unavailable: {exc}") from exc
            outcome["result"] = fetched.result
            if (fetched.state == scheduler_client.RESULT_VALID
                    and scheduler_client.is_valid_result(
                        fetched.result,
                        expected_revision=solver_revision,
                        expected_library_revision=library_revision)
                    and result_matches_candidate(
                        fetched.result, loaded["candidates"][index])):
                outcome["state"] = "valid"
            else:
                outcome.update(
                    state="invalid",
                    reason=(
                        "candidate_mismatch"
                        if fetched.state == scheduler_client.RESULT_VALID
                        else f"result_{fetched.state}"
                    ),
                )
        elif status not in ACTIVE_STATUSES:
            raise RuntimeError(
                f"{stage} task {task_id} returned an unknown status: {status!r}")
        outcomes.append(outcome)
    return {**loaded, "outcomes": outcomes}


def validate_p02_predecessor(
        solver_revision, library_revision, seed, manifest_dir=None):
    """Require two completed, strict-valid p02 results before p08 executes."""
    loaded = load_pilot_stage_manifest(
        solver_revision, library_revision, "p02", seed, manifest_dir=manifest_dir,
        required_by="p08",
    )
    manifest_path = loaded["path"]
    task_ids = loaded["task_ids"]
    expected_names = loaded["task_names"]
    candidates = loaded["candidates"]

    for task_id, expected_name in zip(task_ids, expected_names):
        status = scheduler_client.get_status(task_id)
        if status != "completed":
            raise RuntimeError(
                f"p02 task {task_id} ({expected_name}) is not completed: {status!r}")
        try:
            fetched = scheduler_client.fetch_result(
                task_id,
                expected_revision=solver_revision,
                expected_library_revision=library_revision,
            )
        except scheduler_client.ResultFetchError as exc:
            raise RuntimeError(
                f"p02 task {task_id} stdout/result is unavailable: {exc}") from exc
        if (fetched.state != scheduler_client.RESULT_VALID
                or not scheduler_client.is_valid_result(
                    fetched.result,
                    expected_revision=solver_revision,
                    expected_library_revision=library_revision)):
            raise RuntimeError(
                f"p02 task {task_id} ({expected_name}) has no strict-valid result")
        candidate_index = task_ids.index(task_id)
        if not result_matches_candidate(fetched.result, candidates[candidate_index]):
            raise RuntimeError(
                f"p02 task {task_id} ({expected_name}) result inputs do not match its candidate")

    return {
        "manifest": str(manifest_path),
        "tag": loaded["tag"],
        "task_ids": task_ids,
    }


def validate_p08_completion(
        solver_revision, library_revision, seed=260710, manifest_dir=None):
    """Require all eight p08 tasks to be terminal and strict-valid before refill."""
    loaded = load_pilot_stage_manifest(
        solver_revision, library_revision, "p08", seed, manifest_dir=manifest_dir,
        required_by="feeder",
    )
    path = loaded["path"]
    tag = loaded["tag"]
    task_ids = loaded["task_ids"]
    candidates = loaded["candidates"]
    for index, task_id in enumerate(task_ids):
        status = scheduler_client.get_status(task_id)
        if status != "completed":
            raise RuntimeError(f"p08 task {task_id} is not completed: {status!r}")
        try:
            fetched = scheduler_client.fetch_result(
                task_id, expected_revision=solver_revision,
                expected_library_revision=library_revision)
        except scheduler_client.ResultFetchError as exc:
            raise RuntimeError(f"p08 task {task_id} result is unavailable: {exc}") from exc
        if (fetched.state != scheduler_client.RESULT_VALID
                or not scheduler_client.is_valid_result(
                    fetched.result, expected_revision=solver_revision,
                    expected_library_revision=library_revision)):
            raise RuntimeError(f"p08 task {task_id} has no strict-valid result")
        if not result_matches_candidate(fetched.result, candidates[index]):
            raise RuntimeError(
                f"p08 task {task_id} result inputs do not match its candidate")
    return {"manifest": str(path), "tag": tag, "task_ids": task_ids}


def _atomic_manifest(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, staged = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
        os.replace(staged, path)
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def submit_pilot_stage(
        solver_revision, library_revision, stage, seed=260710,
        execute=False, offset=None, manifest_dir=None, library_root=None):
    if execute and not scheduler_client.campaign_mutation_lock_is_held():
        with campaign_mutation_lock():
            return _submit_pilot_stage_locked(
                solver_revision, library_revision, stage, seed=seed,
                execute=execute, offset=offset, manifest_dir=manifest_dir,
                library_root=library_root,
            )
    return _submit_pilot_stage_locked(
        solver_revision, library_revision, stage, seed=seed,
        execute=execute, offset=offset, manifest_dir=manifest_dir,
        library_root=library_root,
    )


def _submit_pilot_stage_locked(
        solver_revision, library_revision, stage, seed=260710,
        execute=False, offset=None, manifest_dir=None, library_root=None):
    """Plan or submit one contract-defined pilot stage.

    All mutation remains behind ``execute=True``.  The function is intentionally
    reusable by the rapid controller so it cannot drift from the standalone
    pilot CLI's revision, capacity, candidate, and predecessor gates.
    """
    if execute and not scheduler_client.campaign_mutation_lock_is_held():
        raise RuntimeError("pilot mutation requires the campaign mutation lock")
    contract = PILOT_STAGE_CONTRACT.get(stage)
    if contract is None:
        raise ValueError(f"unknown pilot stage: {stage}")
    tasks = contract["tasks"]
    offset = resolve_stage_contract(stage, tasks, offset)
    solver_revision = str(solver_revision or "").strip().lower()
    library_revision = str(library_revision or "").strip().lower()
    if solver_revision != al_driver._current_solver_revision():
        raise RuntimeError("solver revision is not the current vetted local solver")
    if library_revision != al_driver._current_library_revision():
        raise RuntimeError("library revision is not the current clean local library")
    if execute:
        if not library_root:
            raise RuntimeError(
                "pilot execution requires a library root for deployment validation"
            )
        deployment_gate.validate_deployment(
            REPO_ROOT, solver_revision, library_root, library_revision
        )

    predecessor = None
    if execute and stage == "p02":
        if manifest_dir is None:
            predecessor = validate_local_gate(solver_revision, library_revision)
        else:
            predecessor = validate_local_gate(
                solver_revision, library_revision, manifest_dir=manifest_dir)
    elif execute and stage == "p08":
        if manifest_dir is None:
            predecessor = validate_p02_predecessor(
                solver_revision, library_revision, seed)
        else:
            predecessor = validate_p02_predecessor(
                solver_revision, library_revision, seed,
                manifest_dir=manifest_dir)

    profile_path = REGRESSION_ROOT / "verify" / "profiles" / "standard.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    candidates = deterministic_candidates(tasks, offset=offset, seed=seed)
    tag = pilot_tag(
        solver_revision, library_revision, stage, seed, offset)
    plans = []
    for index, params in enumerate(candidates):
        digest = candidate_digest(effective_candidate(params))
        name = f"mft-pilot-{tag}-{index:02d}"
        dedupe_key = scheduler_client.verification_dedupe_key(
            name, params, profile, solver_revision, library_revision)
        existing_id = (
            scheduler_client.reconcile_task_id(name, dedupe_key)
            if execute else None
        )
        record = {
            "index": index,
            "name": name,
            "params_sha256": digest,
            "task_id": existing_id,
            "recovered": bool(existing_id is not None),
        }
        plans.append((record, params))
    missing_count = sum(
        record["task_id"] is None for record, _params in plans)
    snapshot = capacity_snapshot(required_hard_cap=PILOT_PROJECT_HARD_CAP)
    if missing_count and not snapshot["queue_submission_allowed"]:
        raise RuntimeError(
            "pilot submission is blocked by scheduler capacity: "
            f"{snapshot.get('queue_reason') or snapshot.get('queue_state')}: {snapshot}"
        )
    if snapshot["project_submission_slots"] < missing_count:
        raise RuntimeError(
            f"pilot needs {missing_count} missing MFT project slots but only "
            f"{snapshot['project_submission_slots']} remain under stage cap "
            f"{PILOT_PROJECT_HARD_CAP}: {snapshot}"
        )

    manifest = {
        "tag": tag,
        "stage": stage,
        "solver_revision": solver_revision,
        "library_revision": library_revision,
        "seed": seed,
        "offset": offset,
        "task_count": tasks,
        "profile": profile,
        "capacity": snapshot,
        "missing_task_count": missing_count,
        "executed": bool(execute),
        "predecessor": predecessor,
        "tasks": [record for record, _params in plans],
    }
    suffix = ".json" if execute else ".preview.json"
    root = Path(manifest_dir) if manifest_dir is not None else campaign_manifest_dir()
    manifest_path = root / f"{tag}{suffix}"
    partial_path = root / f"{tag}.partial.json"
    if execute:
        # Keep crash recovery separate from the canonical completion gate.
        # rapid_campaign must never interpret a ledger containing None IDs as
        # a submitted pilot manifest.
        _atomic_manifest(manifest, partial_path)
    for record, params in plans:
        if execute and record["task_id"] is None:
            name = record["name"]
            record["task_id"] = scheduler_client.submit_verification(
                name=name,
                workdir=name.replace("-", "_"),
                params=params,
                profile=profile,
                mem_mb=MEMORY_MB,
                cpus=CPUS,
                solver_revision=solver_revision,
                library_revision=library_revision,
            )
            if record["task_id"] is None:
                raise RuntimeError(f"scheduler did not return a task ID for {name}")
            _atomic_manifest(manifest, partial_path)
    if execute:
        _atomic_manifest(manifest, manifest_path)
        partial_path.unlink(missing_ok=True)
    else:
        _atomic_manifest(manifest, manifest_path)
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "capacity": snapshot,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, choices=(2, 8), required=True)
    parser.add_argument("--stage", choices=("p02", "p08"), required=True)
    parser.add_argument("--offset", type=int, default=None)
    parser.add_argument("--seed", type=int, default=260710)
    parser.add_argument("--solver-revision", required=True)
    parser.add_argument("--library-revision", required=True)
    parser.add_argument("--library-root")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    try:
        offset = resolve_stage_contract(args.stage, args.tasks, args.offset)
    except ValueError as exc:
        parser.error(str(exc))
    result = submit_pilot_stage(
        args.solver_revision,
        args.library_revision,
        args.stage,
        seed=args.seed,
        execute=args.execute,
        offset=offset,
        library_root=args.library_root,
    )
    manifest = result["manifest"]
    snapshot = result["capacity"]
    manifest_path = result["manifest_path"]
    print(json.dumps({
        "manifest": str(manifest_path),
        "capacity": snapshot,
        "task_ids": [record["task_id"] for record in manifest["tasks"]],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
