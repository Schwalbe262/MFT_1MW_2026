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
import os
import time

from submit_wave import submit, CAMPAIGN_SETS
from collect_wave import list_tasks

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "feeder_state.json")

TARGET_ACTIVE = 400   # 실행+대기 목표 (--target으로 오버라이드)
BUFFER = 40           # 대기 버퍼 (슬롯이 비는 순간 즉시 붙도록)
COUNT_PER_TASK = 5


def load_state():
    if os.path.isfile(STATE):
        return json.load(open(STATE))
    return {"serial": 0, "submitted_samples": 0}


def save_state(st):
    tmp = STATE + ".tmp"
    json.dump(st, open(tmp, "w"))
    os.replace(tmp, STATE)


def step(max_samples):
    st = load_state()
    ts = list_tasks("mft-camp-")
    active = sum(1 for t in ts if t.get("status") in ("running", "attaching", "queued"))
    deficit = TARGET_ACTIVE + BUFFER - active
    if st["submitted_samples"] >= max_samples:
        print(f"[feeder] max samples reached ({st['submitted_samples']}/{max_samples}) - no refill")
        return False
    if deficit <= 0:
        print(f"[feeder] active {active} >= target - no refill")
        return True
    n_new = min(deficit, (max_samples - st["submitted_samples"] + COUNT_PER_TASK - 1) // COUNT_PER_TASK)
    run_args = f"--count {COUNT_PER_TASK} --thermal --headless {CAMPAIGN_SETS}"
    ok = 0
    for _ in range(n_new):
        st["serial"] += 1
        name = f"mft-camp-c-{st['serial']:05d}"
        wd = f"mft_c_t{st['serial'] % 500:03d}"  # 500개 디렉토리 풀 재사용 (클론 재활용)
        if submit(name, wd, run_args):
            ok += 1
            st["submitted_samples"] += COUNT_PER_TASK
        time.sleep(0.3)
    save_state(st)
    print(f"[feeder] active {active} -> +{ok} tasks (누적 제출 샘플 {st['submitted_samples']})")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", type=int, default=None, help="반복 주기 [s]")
    ap.add_argument("--max-samples", type=int, default=12000)
    ap.add_argument("--target", type=int, default=None,
                    help="실행+대기 목표 (라이선스 서버 과부하 시 감속용)")
    args = ap.parse_args()
    global TARGET_ACTIVE
    if args.target:
        TARGET_ACTIVE = args.target

    if args.once or not args.loop:
        step(args.max_samples)
        return
    while True:
        try:
            if not step(args.max_samples):
                break
        except Exception as e:
            print(f"[feeder] step error: {e}")
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
