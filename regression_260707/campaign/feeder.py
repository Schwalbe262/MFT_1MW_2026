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
import time

import requests

from submit_wave import submit, CAMPAIGN_SETS

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "feeder_state.json")
SCHEDULER = "http://127.0.0.1:8000"
CAMPAIGN_PREFIX = "mft-camp-c-"

TARGET_ACTIVE = 130   # 실행+대기 목표 (--target으로 오버라이드)
BUFFER = 40           # 대기 버퍼 (슬롯이 비는 순간 즉시 붙도록)
COUNT_PER_TASK = 8
CPUS_PER_TASK = 4
CPU_HEADROOM = 0.85
SCHEDULER_ATTEMPTS = 3
ACTIVE_TASK_STATUSES = ("queued", "attaching", "running")


class SchedulerError(RuntimeError):
    pass


def load_state():
    if os.path.isfile(STATE):
        return json.load(open(STATE))
    return {"serial": 0, "submitted_samples": 0}


def save_state(st):
    tmp = STATE + ".tmp"
    json.dump(st, open(tmp, "w"))
    os.replace(tmp, STATE)


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
    summary = _scheduler_json(
        "/api/tasks/summary", params={"name_prefix": CAMPAIGN_PREFIX})
    allocations = _scheduler_json("/api/allocations")
    statuses = summary.get("statuses") if isinstance(summary, dict) else None
    if not isinstance(statuses, dict) or not isinstance(allocations, list):
        raise SchedulerError("scheduler returned an invalid snapshot")
    counts = {s: int(statuses.get(s, 0) or 0) for s in ACTIVE_TASK_STATUSES}
    return counts, allocations


def cpu_task_cap(status_counts, allocations):
    """Use both total and free cores so other workloads retain headroom."""
    active_allocations = [
        a for a in allocations
        if a.get("state") == "active" and a.get("resource_pool", "cpu") == "cpu"]
    total_cpus = sum(max(0, int(a.get("total_cpus") or 0)) for a in active_allocations)
    free_cpus = sum(max(0, int(a.get("free_cpus") or 0)) for a in active_allocations)
    total_slots = math.floor(total_cpus / CPUS_PER_TASK * CPU_HEADROOM)
    free_slots = math.floor(free_cpus / CPUS_PER_TASK * CPU_HEADROOM)
    inflight = status_counts["running"] + status_counts["attaching"]
    return min(total_slots, inflight + free_slots), total_cpus, free_cpus


def step(max_samples, target=TARGET_ACTIVE, buffer=BUFFER):
    st = load_state()
    # 로컬 장부가 아니라 스케줄러의 전체 campaign prefix를 source of truth로 사용한다.
    status_counts, allocations = scheduler_snapshot()
    active = sum(status_counts.values())
    core_cap, total_cpus, free_cpus = cpu_task_cap(status_counts, allocations)
    hard_cap = min(target + buffer, core_cap)
    deficit = hard_cap - active
    print(f"[feeder] scheduler active {active} "
          f"(queued={status_counts['queued']}, attaching={status_counts['attaching']}, "
          f"running={status_counts['running']}), CPU total/free={total_cpus}/{free_cpus}, "
          f"hard_cap={hard_cap}")
    if st["submitted_samples"] >= max_samples:
        print(f"[feeder] max samples reached ({st['submitted_samples']}/{max_samples}) - no refill")
        return False
    if deficit <= 0:
        print(f"[feeder] active {active} >= hard cap {hard_cap} - no refill")
        return True
    n_new = min(deficit, (max_samples - st["submitted_samples"]) // COUNT_PER_TASK)
    if n_new <= 0:
        return False
    run_args = f"--count {COUNT_PER_TASK} --thermal --headless {CAMPAIGN_SETS}"
    ok = 0
    for _ in range(n_new):
        st["serial"] += 1
        name = f"mft-camp-c-{st['serial']:05d}"
        wd = f"mft_c_t{st['serial'] % 500:03d}"  # 500개 디렉토리 풀 재사용 (클론 재활용)
        tid = submit(name, wd, run_args)
        if tid:
            ok += 1
            st["submitted_samples"] += COUNT_PER_TASK
            st.setdefault("outstanding", []).append(tid)
        # 실패한 제출도 serial을 보존해 이름 재사용을 막는다.
        save_state(st)
        time.sleep(0.3)
    print(f"[feeder] active {active} -> +{ok}/{n_new} tasks "
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
    args = ap.parse_args()

    if args.once or not args.loop:
        step(args.max_samples, target=args.target, buffer=args.buffer)
        return
    while True:
        try:
            if not step(args.max_samples, target=args.target, buffer=args.buffer):
                break
        except Exception as e:
            print(f"[feeder] step error: {e}")
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
