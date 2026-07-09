"""
데이터 캠페인 웨이브 제출기.

사용:
  python submit_wave.py --tasks 400 --count 5 --wave 1          # 본 웨이브 (2000 샘플)
  python submit_wave.py --tasks 400 --count 1 --wave 0 --pilot  # 파일럿 (게이트 2)

- 태스크마다 독립 클론 디렉토리 (mft_w<wave>_t<idx>) -> NFS 락 경합 없음
- 결과는 각 디렉토리의 parquet 파트 + stdout CSV 로 회수 (collect_wave.py)
- 웨이브당 golden case 1개 동반 (드리프트 감시)
- 캠페인 설정: 대칭 loss + 1/8 thermal (기본값), matrix pe 1.0/14패스, P_target 1MW
"""
import argparse
import json
import os
import sys
import time

import requests

SCHEDULER = "http://127.0.0.1:8000"

CAMPAIGN_SETS = "--set percent_error=1.0 --set max_passes=14 --set P_target=1e6"


def submit(name, workdir, run_args, mem_mb=32768, cpus=4):
    """Project 방식 제출 (2026-07-09 전환): 스케줄러가 배포/경로/환경/git pull 전담.
    workdir 인자는 구버전 호환용으로 무시. 반환: task_id (실패 시 None)."""
    try:
        r = requests.post(f"{SCHEDULER}/api/tasks", json={
            "name": name,
            "project": "MFT_1MW_2026v1",
            "entrypoint": "run_simulation_260706.py",
            "arguments": run_args,
            "cpus": cpus, "memory_mb": mem_mb,
            "scheduling_profile": "fea_bursty",
        }, timeout=20)
        if r.status_code in (200, 201):
            return r.json().get("task_id") or r.json().get("id")
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=400)
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--wave", type=int, required=True)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--thermal", action="store_true", default=True)
    args = ap.parse_args()

    run_args = f"--count {args.count} --thermal --headless {CAMPAIGN_SETS}"
    tag = "pilot" if args.pilot else f"w{args.wave}"

    ok = 0
    for i in range(args.tasks):
        name = f"mft-camp-{tag}-{i:03d}"
        wd = f"mft_{tag}_t{i:03d}"
        if submit(name, wd, run_args):
            ok += 1
        else:
            print(f"submit FAILED: {name}", file=sys.stderr)
        if (i + 1) % 50 == 0:
            print(f"{i+1}/{args.tasks} submitted")
            time.sleep(2)  # 스케줄러 부하 완화

    # golden case 동반
    submit(f"mft-camp-{tag}-golden", f"mft_{tag}_golden", "--golden --headless")

    # 웨이브마다 스테일 스위퍼 동반 제출 (취소/킬 잔재 자동 청소 - 상시 위생)
    import subprocess
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "sweep_stale.py"), "--hours", "6"], check=False)
    print(f"done: {ok}/{args.tasks} tasks + golden + sweeper submitted (wave {args.wave})")


if __name__ == "__main__":
    main()
