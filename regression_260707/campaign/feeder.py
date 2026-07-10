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
import json
import math
import os
import sys
import time

import requests
import pyarrow.parquet as pq
from filelock import FileLock

from pinned_pilot import (
    PILOT_RESERVED_VALID_CANDIDATES,
    al_driver,
    cursor_after_valid_candidates,
    next_valid_candidate,
    validate_p08_completion,
)

HERE = os.path.dirname(os.path.abspath(__file__))
VERIFY_DIR = os.path.abspath(os.path.join(HERE, "..", "verify"))
if VERIFY_DIR not in sys.path:
    sys.path.insert(0, VERIFY_DIR)
import scheduler_client

STATE = os.path.join(HERE, "feeder_state.json")
SCHEDULER = "http://127.0.0.1:8000"
CAMPAIGN_PREFIX = "mft-camp-"

TARGET_ACTIVE = 130   # 실행+대기 목표 (--target으로 오버라이드)
BUFFER = 40           # 대기 버퍼 (슬롯이 비는 순간 즉시 붙도록)
COUNT_PER_TASK = 1
CPUS_PER_TASK = 4
CPU_HEADROOM = 0.85
SCHEDULER_ATTEMPTS = 3
ACTIVE_TASK_STATUSES = ("queued", "attaching", "running")
PROFILE_PATH = os.path.join(HERE, "..", "verify", "profiles", "standard.json")
TRAIN_PARQUET = os.path.join(HERE, "..", "data", "dataset", "train.parquet")
COLLECT_CACHE = os.path.join(HERE, "..", "data", "dataset", "collect_cache.json")


class SchedulerError(RuntimeError):
    pass


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
        "/api/tasks", params={"limit": 10000, "name_prefix": CAMPAIGN_PREFIX})
    tasks = payload if isinstance(payload, list) else (
        payload.get("tasks") if isinstance(payload, dict) else None)
    if not isinstance(tasks, list):
        raise SchedulerError("scheduler returned an invalid campaign task inventory")
    return [
        task for task in tasks
        if str(task.get("name") or "").startswith(CAMPAIGN_PREFIX)
    ]


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


def scheduler_snapshot():
    campaign_summary = _scheduler_json(
        "/api/tasks/summary", params={"name_prefix": CAMPAIGN_PREFIX})
    global_summary = _scheduler_json("/api/tasks/summary")
    allocations = _scheduler_json("/api/allocations")
    capacity = _scheduler_json("/api/task-capacity", params={
        "cpus": CPUS_PER_TASK,
        "memory_mb": 32768,
        "scheduling_profile": "fea_bursty",
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
    })
    campaign_statuses = (
        campaign_summary.get("statuses") if isinstance(campaign_summary, dict) else None)
    global_statuses = (
        global_summary.get("statuses") if isinstance(global_summary, dict) else None)
    if (not isinstance(campaign_statuses, dict)
            or not isinstance(global_statuses, dict)
            or not isinstance(allocations, list)
            or not isinstance(capacity, dict)):
        raise SchedulerError("scheduler returned an invalid snapshot")
    campaign_counts = {
        status: int(campaign_statuses.get(status, 0) or 0)
        for status in ACTIVE_TASK_STATUSES}
    global_counts = {
        status: int(global_statuses.get(status, 0) or 0)
        for status in ACTIVE_TASK_STATUSES}
    return campaign_counts, global_counts, allocations, int(capacity.get("ready_fit_slots") or 0)


def cpu_submission_headroom(status_counts, allocations, ready_fit_slots):
    """Use global demand plus active/warm total and free cores."""
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
    st = load_state()
    # 로컬 장부가 아니라 스케줄러의 전체 campaign prefix를 source of truth로 사용한다.
    campaign_counts, global_counts, allocations, ready_fit_slots = scheduler_snapshot()
    campaign_active = sum(campaign_counts.values())
    data_rows, judged_ids = dataset_collection_snapshot()
    if target + buffer > 0:
        tasks = campaign_inventory()
        reserved_rows = reserved_unjudged_rows(st, tasks, judged_ids)
    else:
        reserved_rows = 0
    projected_rows = data_rows + reserved_rows
    headroom, total_cpus, free_cpus, global_active = cpu_submission_headroom(
        global_counts, allocations, ready_fit_slots)
    campaign_deficit = max(0, target + buffer - campaign_active)
    deficit = min(campaign_deficit, headroom)
    print(f"[feeder] campaign active {campaign_active} "
          f"(queued={campaign_counts['queued']}, attaching={campaign_counts['attaching']}, "
          f"running={campaign_counts['running']}), global active={global_active}, "
          f"CPU total/free={total_cpus}/{free_cpus}, ready_fit={ready_fit_slots}, "
          f"submission_headroom={headroom}, dataset/reserved/projected="
          f"{data_rows}/{reserved_rows}/{projected_rows}")
    if data_rows >= max_samples:
        print(f"[feeder] dataset target reached ({data_rows}/{max_samples}) - no refill")
        return False
    if projected_rows >= max_samples:
        print(f"[feeder] no refill: projected dataset rows {projected_rows}/{max_samples}")
        return True
    if deficit <= 0:
        print(f"[feeder] no refill: campaign_deficit={campaign_deficit}, headroom={headroom}")
        return True
    if not isinstance(solver_revision, str) or len(solver_revision) != 40:
        raise SchedulerError("a full pinned solver revision is required before refill")
    if not isinstance(library_revision, str) or len(library_revision) != 40:
        raise SchedulerError("a full pinned pyaedt_library revision is required before refill")
    candidate_generation = (
        f"{solver_revision}:{library_revision}:seed{int(candidate_seed)}")
    if st.get("candidate_generation") != candidate_generation:
        st["candidate_generation"] = candidate_generation
        st["candidate_cursor"] = cursor_after_valid_candidates(
            PILOT_RESERVED_VALID_CANDIDATES, seed=candidate_seed)
        st.pop("candidate_raw_index", None)
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
        st["candidate_raw_index"] = raw_index
        tid = submit(name, wd, params, solver_revision, library_revision)
        if tid:
            ok += 1
            st["submitted_samples"] += COUNT_PER_TASK
            st.setdefault("outstanding", []).append(tid)
            st.setdefault("task_expected_rows", {})[str(tid)] = COUNT_PER_TASK
        # 실패한 제출도 serial을 보존해 이름 재사용을 막는다.
        save_state(st)
        time.sleep(0.3)
    print(f"[feeder] campaign active {campaign_active} -> +{ok}/{n_new} tasks "
          f"(누적 제출 샘플 {st['submitted_samples']})")
    return True


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
    args = ap.parse_args()

    if args.target + args.buffer > 0:
        if args.solver_revision != al_driver._current_solver_revision():
            raise SchedulerError("feeder solver revision is not the current vetted local solver")
        if args.library_revision != al_driver._current_library_revision():
            raise SchedulerError("feeder library revision is not the current clean local library")
        validate_p08_completion(
            args.solver_revision, args.library_revision, seed=args.candidate_seed)

    if args.once or not args.loop:
        step(
            args.max_samples, target=args.target, buffer=args.buffer,
            solver_revision=args.solver_revision,
            library_revision=args.library_revision,
            candidate_seed=args.candidate_seed,
        )
        return
    while True:
        try:
            if not step(
                    args.max_samples, target=args.target, buffer=args.buffer,
                    solver_revision=args.solver_revision,
                    library_revision=args.library_revision,
                    candidate_seed=args.candidate_seed):
                break
        except Exception as e:
            print(f"[feeder] step error: {e}")
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
