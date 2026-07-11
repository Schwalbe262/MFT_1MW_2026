"""
상시 포화 피더: 캠페인 태스크(실행+대기)를 목표 수준(기본 400+버퍼 40)으로 유지.

웨이브 장벽 없이, 완료되는 만큼 새 태스크를 채워 넣어 400 병렬을 상시 유지한다.
- 태스크: --count 5 (샘플 5개 연속, 실패 재추첨 내장)
- 이름: mft-camp-c-<일련번호> (serial은 feeder_state.json에 영속)
- 총량 상한: --max-samples 도달 시 중단 (기본 12000)

사용: python feeder.py --once        # 1회 보충 (크론/수동)
      python feeder.py --loop 600   # 데몬 (600초 주기)
"""
import argparse
import copy
import json
import math
import os
import sys
import time
from dataclasses import dataclass

import requests
import pyarrow.parquet as pq
from filelock import FileLock

from pinned_pilot import (
    LEGACY_MFT_NAME_PREFIX,
    MFT_PROJECT,
    MFT_PROJECT_MAX_ACTIVE_TASKS,
    PILOT_RESERVED_VALID_CANDIDATES,
    al_driver,
    campaign_mutation_lock,
    cursor_after_valid_candidates,
    next_valid_candidate,
    project_submission_snapshot,
    queue_allows_demand_submission,
    validate_p08_completion,
)

HERE = os.path.dirname(os.path.abspath(__file__))
VERIFY_DIR = os.path.abspath(os.path.join(HERE, "..", "verify"))
if VERIFY_DIR not in sys.path:
    sys.path.insert(0, VERIFY_DIR)
import scheduler_client
import deployment_gate

STATE = os.path.join(HERE, "feeder_state.json")
SCHEDULER = "http://127.0.0.1:8000"
CAMPAIGN_PREFIX = "mft-camp-"

TARGET_ACTIVE = 50    # standalone 실행+대기 목표 (--target으로 오버라이드)
BUFFER = 0            # production 300 promotion is owned by rapid_campaign
MAX_STANDALONE_ACTIVE = 50
COUNT_PER_TASK = 1
CPUS_PER_TASK = 4
CPU_HEADROOM = 0.85
SCHEDULER_ATTEMPTS = 3
ACTIVE_TASK_STATUSES = ("queued", "attaching", "running")
PROFILE_PATH = os.path.join(HERE, "..", "verify", "profiles", "standard.json")
TRAIN_PARQUET = os.path.join(HERE, "..", "data", "dataset", "train.parquet")
COLLECT_CACHE = os.path.join(HERE, "..", "data", "dataset", "collect_cache.json")
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def _require_deployed_revisions(solver_revision, library_revision):
    library_root = os.environ.get("MFT_PYAEDT_LIBRARY_ROOT", "").strip()
    if not library_root:
        library_root = os.path.abspath(
            os.path.join(REPO_ROOT, "..", "pyaedt_library")
        )
    return deployment_gate.validate_deployment(
        REPO_ROOT, solver_revision, library_root, library_revision
    )


class SchedulerError(RuntimeError):
    pass


_RAPID_REFILL_SEAL = object()


@dataclass(frozen=True)
class _RapidRefillAuthorization:
    target: int
    max_samples: int
    solver_revision: str
    library_revision: str
    candidate_seed: int
    seal: object


def _authorize_rapid_refill(
        decision, *, max_samples, solver_revision, library_revision,
        candidate_seed, local_passed, pilots_complete):
    """Seal one refill decision only after the rapid promotion evidence passes."""
    if not scheduler_client.campaign_mutation_lock_is_held():
        raise SchedulerError("rapid refill authorization requires the mutation lock")
    if not isinstance(decision, dict) or decision.get("paused"):
        raise SchedulerError("rapid refill decision is absent or paused")
    target = decision.get("target_active")
    if isinstance(target, bool) or target not in (50, 300):
        raise SchedulerError("rapid refill target must be exactly 50 or 300")
    expected_action = f"refill_{target}"
    if decision.get("action") != expected_action:
        raise SchedulerError("rapid refill action does not match its target")
    pilot = decision.get("pilot") or {}
    if (not local_passed or not pilots_complete
            or int(pilot.get("valid") or 0) < 5
            or int(pilot.get("invalid") or 0) != 0):
        raise SchedulerError("rapid refill lacks local3/pilot promotion evidence")
    if target == 300:
        production = decision.get("production") or {}
        terminal = int(production.get("terminal") or 0)
        valid_rate = production.get("valid_rate")
        if (terminal < 20 or isinstance(valid_rate, bool)
                or not isinstance(valid_rate, (int, float))
                or not math.isfinite(float(valid_rate))
                or float(valid_rate) < 0.90):
            raise SchedulerError("rapid refill lacks fleet20/90% promotion evidence")
    return _RapidRefillAuthorization(
        target=int(target),
        max_samples=int(max_samples),
        solver_revision=str(solver_revision),
        library_revision=str(library_revision),
        candidate_seed=int(candidate_seed),
        seal=_RAPID_REFILL_SEAL,
    )


def _step_from_rapid_controller(
        max_samples, *, authorization, target, buffer=0,
        solver_revision=None, library_revision=None, candidate_seed=260710):
    """Execute a production refill using one evidence-bound authorization."""
    if not scheduler_client.campaign_mutation_lock_is_held():
        raise SchedulerError("rapid production refill requires the mutation lock")
    expected = (
        int(target), int(max_samples), str(solver_revision),
        str(library_revision), int(candidate_seed),
    )
    actual = (
        getattr(authorization, "target", None),
        getattr(authorization, "max_samples", None),
        getattr(authorization, "solver_revision", None),
        getattr(authorization, "library_revision", None),
        getattr(authorization, "candidate_seed", None),
    )
    if (not isinstance(authorization, _RapidRefillAuthorization)
            or authorization.seal is not _RAPID_REFILL_SEAL
            or actual != expected or int(buffer) != 0):
        raise SchedulerError("rapid production refill authorization is invalid")
    return _step_locked(
        max_samples, target=target, buffer=buffer,
        solver_revision=solver_revision, library_revision=library_revision,
        candidate_seed=candidate_seed, _rapid_authorization=authorization,
    )


def submit(name, workdir, params, solver_revision, library_revision):
    with open(PROFILE_PATH, encoding="utf-8") as stream:
        profile = json.load(stream)
    return scheduler_client.submit_verification(
        name=name,
        workdir=workdir,
        params=params,
        profile=profile,
        mem_mb=32768,
        cpus=CPUS_PER_TASK,
        solver_revision=solver_revision,
        library_revision=library_revision,
    )


def load_state():
    if os.path.isfile(STATE):
        return json.load(open(STATE))
    return {"serial": 0, "submitted_samples": 0}


def save_state(st):
    tmp = STATE + ".tmp"
    json.dump(st, open(tmp, "w"))
    os.replace(tmp, STATE)


def dataset_collection_snapshot():
    """Read master row count and collector judgements in one lock epoch."""
    with FileLock(TRAIN_PARQUET + ".lock", timeout=30):
        rows = (
            int(pq.ParquetFile(TRAIN_PARQUET).metadata.num_rows)
            if os.path.isfile(TRAIN_PARQUET) else 0)
        if not os.path.isfile(COLLECT_CACHE):
            return rows, set()
        try:
            with open(COLLECT_CACHE, encoding="utf-8") as stream:
                cache = json.load(stream)
            judged = {
                int(task_id)
                for key in ("harvested", "nodata")
                for task_id in cache.get(key, [])
            }
        except (OSError, ValueError, TypeError) as exc:
            raise SchedulerError(f"collector cache is unreadable: {exc}") from exc
        return rows, judged


def dataset_row_count():
    return dataset_collection_snapshot()[0]


def campaign_inventory():
    payload = _scheduler_json(
        "/api/tasks", params={
            "limit": 10000,
            "name_prefix": CAMPAIGN_PREFIX,
        })
    tasks = payload if isinstance(payload, list) else (
        payload.get("tasks") if isinstance(payload, dict) else None)
    if not isinstance(tasks, list):
        raise SchedulerError("scheduler returned an invalid campaign task inventory")
    inventory = []
    seen_ids = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise SchedulerError("scheduler returned an invalid campaign task")
        if not str(task.get("name") or "").startswith(CAMPAIGN_PREFIX):
            raise SchedulerError(
                "scheduler campaign prefix filter returned an unrelated task")
        project = str(task.get("project") or "").strip()
        if project not in ("", MFT_PROJECT):
            raise SchedulerError(
                f"campaign task belongs to unexpected project {project!r}")
        task_id = task.get("id", task.get("task_id"))
        if (isinstance(task_id, bool) or not isinstance(task_id, int)
                or task_id <= 0 or task_id in seen_ids):
            raise SchedulerError(
                "scheduler returned an invalid/duplicate campaign task ID")
        seen_ids.add(task_id)
        inventory.append(task)
    return inventory


def expected_task_rows(task):
    name = str((task or {}).get("name") or "")
    return 1 if name.startswith("mft-camp-s") else 8


def reserved_unjudged_rows(state, tasks, judged_ids):
    """Reserve outputs until the collector durably classifies every task ID."""
    by_id = {
        int(task["id"]): task for task in tasks
        if task.get("id") is not None
    }
    ledger_ids = {
        int(task_id) for task_id in state.get("outstanding", [])
        if task_id is not None
    }
    active_ids = {
        task_id for task_id, task in by_id.items()
        if task.get("status") in ACTIVE_TASK_STATUSES
    }
    inventory_unjudged = {
        task_id for task_id in by_id if task_id not in judged_ids
    }
    reserved_ids = (ledger_ids | active_ids | inventory_unjudged) - judged_ids
    recorded_rows = state.get("task_expected_rows") or {}
    return sum(
        int(recorded_rows.get(str(task_id), expected_task_rows(by_id.get(task_id))))
        for task_id in reserved_ids
    )


def _scheduler_json(path, params=None):
    last_error = None
    for attempt in range(SCHEDULER_ATTEMPTS):
        try:
            response = requests.get(f"{SCHEDULER}{path}", params=params, timeout=30)
            status_code = int(response.status_code)
            if status_code >= 400:
                raise SchedulerError(f"HTTP {status_code} from {path}")
            return response.json()
        except (requests.RequestException, ValueError, SchedulerError) as exc:
            last_error = exc
            if attempt + 1 < SCHEDULER_ATTEMPTS:
                time.sleep(0.5 * (2 ** attempt))
    raise SchedulerError(f"scheduler request failed for {path}: {last_error}")


def scheduler_snapshot(required_hard_cap):
    global_summary = _scheduler_json("/api/tasks/summary")
    allocations = _scheduler_json("/api/allocations")
    projects = _scheduler_json("/api/projects")
    project_tasks = _scheduler_json("/api/tasks", params={
        "limit": 10000,
        "project": MFT_PROJECT,
        "status": ",".join(ACTIVE_TASK_STATUSES),
    })
    legacy_tasks = _scheduler_json("/api/tasks", params={
        "limit": 10000,
        "name_prefix": LEGACY_MFT_NAME_PREFIX,
        "status": ",".join(ACTIVE_TASK_STATUSES),
    })
    capacity = _scheduler_json("/api/task-capacity", params={
        "cpus": CPUS_PER_TASK,
        "memory_mb": 32768,
        "scheduling_profile": "fea_bursty",
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
    })
    global_statuses = (
        global_summary.get("statuses") if isinstance(global_summary, dict) else None)
    if (not isinstance(global_statuses, dict)
            or not isinstance(allocations, list)
            or not isinstance(capacity, dict)):
        raise SchedulerError("scheduler returned an invalid snapshot")
    global_counts = {
        status: int(global_statuses.get(status, 0) or 0)
        for status in ACTIVE_TASK_STATUSES}
    try:
        queue_submission_allowed = queue_allows_demand_submission(
            capacity.get("queue_state"))
        project_gate = project_submission_snapshot(
            projects, project_tasks, required_hard_cap,
            legacy_tasks=legacy_tasks)
    except RuntimeError as exc:
        raise SchedulerError(str(exc)) from exc
    capacity_gate = {
        "ready_fit_slots": int(capacity.get("ready_fit_slots") or 0),
        "queue_state": str(capacity.get("queue_state") or "").strip().lower(),
        "queue_reason": str(capacity.get("queue_reason") or "").strip(),
        "queue_submission_allowed": queue_submission_allowed,
    }
    capacity_gate.update(project_gate)
    capacity_gate["submission_allowed"] = bool(
        queue_submission_allowed and project_gate["project_submission_slots"] > 0)
    return project_gate["project_counts"], global_counts, allocations, capacity_gate


def cpu_submission_headroom(status_counts, allocations, ready_fit_slots):
    """Report immediate owned-pool room; queued demand is not capped by it."""
    usable_allocations = [
        a for a in allocations
        if a.get("state") in ("active", "warm")
        and a.get("resource_pool", "cpu") == "cpu"]
    total_cpus = sum(max(0, int(a.get("total_cpus") or 0)) for a in usable_allocations)
    free_cpus = sum(max(0, int(a.get("free_cpus") or 0)) for a in usable_allocations)
    total_slots = math.floor(total_cpus / CPUS_PER_TASK * CPU_HEADROOM)
    free_slots = math.floor(free_cpus / CPUS_PER_TASK * CPU_HEADROOM)
    global_active = sum(status_counts.values())
    headroom = max(0, min(
        free_slots,
        total_slots - global_active,
        max(0, int(ready_fit_slots or 0)),
    ))
    return headroom, total_cpus, free_cpus, global_active


def step(max_samples, target=TARGET_ACTIVE, buffer=BUFFER,
         solver_revision=None, library_revision=None, candidate_seed=260710):
    requested_active = int(target) + int(buffer)
    if requested_active > MAX_STANDALONE_ACTIVE:
        raise SchedulerError(
            f"direct feeder hard cap is {MAX_STANDALONE_ACTIVE}; "
            "only rapid_campaign may authorize production promotion")
    if (requested_active > 0
            and not scheduler_client.campaign_mutation_lock_is_held()):
        with campaign_mutation_lock():
            return _step_locked(
                max_samples, target=target, buffer=buffer,
                solver_revision=solver_revision,
                library_revision=library_revision,
                candidate_seed=candidate_seed,
            )
    return _step_locked(
        max_samples, target=target, buffer=buffer,
        solver_revision=solver_revision,
        library_revision=library_revision,
        candidate_seed=candidate_seed,
    )


def _step_locked(max_samples, target=TARGET_ACTIVE, buffer=BUFFER,
                 solver_revision=None, library_revision=None,
                 candidate_seed=260710, _rapid_authorization=None):
    requested_active = int(target) + int(buffer)
    if requested_active > 0 and not scheduler_client.campaign_mutation_lock_is_held():
        raise SchedulerError("campaign refill requires the project mutation lock")
    if requested_active > MAX_STANDALONE_ACTIVE:
        if (not isinstance(_rapid_authorization, _RapidRefillAuthorization)
                or _rapid_authorization.seal is not _RAPID_REFILL_SEAL
                or _rapid_authorization.target != requested_active):
            raise SchedulerError("production refill requires rapid promotion authorization")
    st = load_state()
    committed_state = copy.deepcopy(st)
    # 로컬 장부나 다른 project가 아니라 scheduler의 MFT logical project가 source of truth다.
    hard_cap = max(1, int(target) + int(buffer))
    if hard_cap > MFT_PROJECT_MAX_ACTIVE_TASKS:
        raise SchedulerError(
            f"campaign hard cap {hard_cap} exceeds project maximum "
            f"{MFT_PROJECT_MAX_ACTIVE_TASKS}")
    campaign_counts, global_counts, allocations, capacity_gate = scheduler_snapshot(
        hard_cap)
    ready_fit_slots = capacity_gate["ready_fit_slots"]
    campaign_active = sum(campaign_counts.values())
    data_rows, judged_ids = dataset_collection_snapshot()
    if target + buffer > 0:
        tasks = campaign_inventory()
        reserved_rows = reserved_unjudged_rows(st, tasks, judged_ids)
    else:
        reserved_rows = 0
    projected_rows = data_rows + reserved_rows
    immediate_headroom, total_cpus, free_cpus, global_active = cpu_submission_headroom(
        global_counts, allocations, ready_fit_slots)
    campaign_deficit = max(0, target + buffer - campaign_active)
    deficit = (
        min(campaign_deficit, capacity_gate["project_submission_slots"])
        if capacity_gate["submission_allowed"] else 0
    )
    print(f"[feeder] campaign active {campaign_active} "
          f"(queued={campaign_counts['queued']}, attaching={campaign_counts['attaching']}, "
          f"running={campaign_counts['running']}), global active={global_active}, "
          f"CPU total/free={total_cpus}/{free_cpus}, ready_fit={ready_fit_slots}, "
          f"immediate_headroom={immediate_headroom}, "
          f"queue_state={capacity_gate['queue_state']}, "
          f"project={MFT_PROJECT} active/open="
          f"{capacity_gate['project_active']}/"
          f"{capacity_gate['project_submission_slots']}, "
          "dataset/reserved/projected="
          f"{data_rows}/{reserved_rows}/{projected_rows}")
    if data_rows >= max_samples:
        print(f"[feeder] dataset target reached ({data_rows}/{max_samples}) - no refill")
        return False
    if projected_rows >= max_samples:
        print(f"[feeder] no refill: projected dataset rows {projected_rows}/{max_samples}")
        return True
    if deficit <= 0:
        print(
            f"[feeder] no refill: campaign_deficit={campaign_deficit}, "
            f"queue_state={capacity_gate['queue_state']}, "
            f"queue_reason={capacity_gate['queue_reason'] or '-'}"
        )
        return True
    if not isinstance(solver_revision, str) or len(solver_revision) != 40:
        raise SchedulerError("a full pinned solver revision is required before refill")
    if not isinstance(library_revision, str) or len(library_revision) != 40:
        raise SchedulerError("a full pinned pyaedt_library revision is required before refill")
    candidate_generation = (
        f"{solver_revision}:{library_revision}:seed{int(candidate_seed)}")
    generation_changed = st.get("candidate_generation") != candidate_generation
    candidate_cursors = dict(st.get("candidate_cursors") or {})
    # Migrate the legacy single cursor once, then retain an independent cursor
    # for every revision/seed generation. Switching controllers cannot replay
    # candidates that an earlier generation already consumed.
    previous_generation = st.get("candidate_generation")
    if (previous_generation and previous_generation not in candidate_cursors
            and st.get("candidate_cursor") is not None):
        candidate_cursors[previous_generation] = int(st["candidate_cursor"])
    if candidate_generation in candidate_cursors:
        candidate_cursor = int(candidate_cursors[candidate_generation])
    else:
        candidate_cursor = cursor_after_valid_candidates(
            PILOT_RESERVED_VALID_CANDIDATES, seed=candidate_seed)
    if generation_changed:
        st["candidate_generation"] = candidate_generation
        st["candidate_cursor"] = candidate_cursor
        st.pop("candidate_raw_index", None)
    st["candidate_cursors"] = candidate_cursors
    n_new = min(deficit, (max_samples - projected_rows) // COUNT_PER_TASK)
    if n_new <= 0:
        return False
    ok = 0
    for _ in range(n_new):
        st["serial"] += 1
        generation = f"s{solver_revision[:7]}-l{library_revision[:7]}"
        name = f"mft-camp-{generation}-{st['serial']:05d}"
        wd = f"mft_c_t{st['serial'] % 500:03d}"  # 500개 디렉토리 풀 재사용 (클론 재활용)
        next_cursor, raw_index, params = next_valid_candidate(
            int(st.get("candidate_cursor", 0)), seed=candidate_seed)
        st["candidate_cursor"] = next_cursor
        st["candidate_cursors"][candidate_generation] = next_cursor
        st["candidate_raw_index"] = raw_index
        try:
            tid = submit(name, wd, params, solver_revision, library_revision)
        except Exception:
            st.clear()
            st.update(copy.deepcopy(committed_state))
            raise
        if tid is None:
            st.clear()
            st.update(copy.deepcopy(committed_state))
            raise SchedulerError(
                f"scheduler did not return a task ID for {name}; "
                "candidate state was not advanced")
        ok += 1
        st["submitted_samples"] += COUNT_PER_TASK
        st.setdefault("outstanding", []).append(tid)
        st.setdefault("task_expected_rows", {})[str(tid)] = COUNT_PER_TASK
        # Only a durable task ID commits the candidate/name cursor.
        save_state(st)
        committed_state = copy.deepcopy(st)
        time.sleep(0.3)
    print(f"[feeder] campaign active {campaign_active} -> +{ok}/{n_new} tasks "
          f"(누적 제출 샘플 {st['submitted_samples']})")
    return True


def publish_ready_marker(path, solver_revision, library_revision):
    """Atomically prove that this feeder completed one guarded cycle."""
    if not path:
        return
    target = os.path.abspath(path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    staged = f"{target}.{os.getpid()}.tmp"
    payload = {
        "pid": os.getpid(),
        "ready_at": time.time(),
        "solver_revision": solver_revision,
        "library_revision": library_revision,
    }
    try:
        with open(staged, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True)
        os.replace(staged, target)
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", type=int, default=None, help="반복 주기 [s]")
    ap.add_argument("--max-samples", type=int, default=12000)
    ap.add_argument("--target", type=int, default=TARGET_ACTIVE,
                    help="실행+대기 목표 (라이선스 서버 과부하 시 감속용)")
    ap.add_argument("--buffer", type=int, default=BUFFER,
                    help="목표 초과 대기 버퍼")
    ap.add_argument("--solver-revision")
    ap.add_argument("--library-revision")
    ap.add_argument("--candidate-seed", type=int, default=260710)
    ap.add_argument(
        "--ready-file",
        help="atomically written after the first successful guarded cycle",
    )
    args = ap.parse_args()

    requested_active = int(args.target) + int(args.buffer)
    if requested_active > MAX_STANDALONE_ACTIVE:
        raise SchedulerError(
            f"standalone feeder hard cap is {MAX_STANDALONE_ACTIVE}; "
            "use rapid_campaign.py for 300-task production promotion")

    if args.target + args.buffer > 0:
        if args.solver_revision != al_driver._current_solver_revision():
            raise SchedulerError("feeder solver revision is not the current vetted local solver")
        if args.library_revision != al_driver._current_library_revision():
            raise SchedulerError("feeder library revision is not the current clean local library")
        validate_p08_completion(
            args.solver_revision, args.library_revision, seed=args.candidate_seed)

    def guarded_step():
        def run_locked_step():
            if args.target + args.buffer > 0:
                _require_deployed_revisions(
                    args.solver_revision, args.library_revision
                )
            return step(
                args.max_samples, target=args.target, buffer=args.buffer,
                solver_revision=args.solver_revision,
                library_revision=args.library_revision,
                candidate_seed=args.candidate_seed,
            )

        if args.target + args.buffer > 0:
            with campaign_mutation_lock():
                return run_locked_step()
        return run_locked_step()

    if args.once or not args.loop:
        guarded_step()
        publish_ready_marker(
            args.ready_file, args.solver_revision, args.library_revision)
        return
    ready_published = False
    while True:
        try:
            keep_running = guarded_step()
            if not ready_published:
                publish_ready_marker(
                    args.ready_file, args.solver_revision, args.library_revision)
                ready_published = True
            if not keep_running:
                break
        except Exception as e:
            print(f"[feeder] step error: {e}")
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
