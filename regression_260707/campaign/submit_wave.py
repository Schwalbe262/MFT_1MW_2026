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

BASE = ("source /etc/profile.d/lmod.sh 2>/dev/null || true; "
        "module load ansys-electronics/v252 || export ANSYSEM_ROOT252=/opt/ohpc/pub/Electronics/v252/Linux64; "
        # 라이선스 데몬 보호: 클라이언트 타임아웃 3초(기본 0.1초 - 바쁜 데몬에서 리셋 유발),
        # 시작 지터 0-240초로 체크아웃 동시 폭주 분산
        "export FLEXLM_TIMEOUT=3000000; sleep $((RANDOM % 240)); ")

CAMPAIGN_SETS = "--set percent_error=1.0 --set max_passes=14 --set P_target=1e6"

# 동시 클론 레이스 방지: 임시 디렉토리에 클론 후 원자적 rename (같은 계정에 여러 태스크 배정 시)
LIB_CLONE = ("([ -d pyaedt_library/src ] || { git clone -q --depth 1 "
             "https://github.com/Schwalbe262/pyaedt_library.git pyaedt_library.tmp.$$ "
             "&& { mv -T pyaedt_library.tmp.$$ pyaedt_library 2>/dev/null || rm -rf pyaedt_library.tmp.$$; }; }) && "
             "[ -d pyaedt_library/src ] && ")


def submit(name, workdir, run_args, mem_mb=32768, cpus=4):
    cmd = (BASE + LIB_CLONE +
           f"([ -d {workdir} ] || git clone -q --depth 1 https://github.com/Schwalbe262/MFT_1MW_2026.git {workdir}) && "
           f"cd {workdir} && git pull -q && "
           # 시작 시 이전 태스크/실패 샘플 잔재 제거 (라이선스 폭풍 시 개당 5GB+ 누적 실측)
           f"rm -rf simulation aedt_temp 2>/dev/null; "
           f"python run_simulation_260706.py {run_args}; "
           f"echo ===RESULT_CSV===; cat simulation_results_260706.csv 2>/dev/null; "
           f"echo ===FAILED_CSV===; cat failed_samples_260706.csv 2>/dev/null | head -50; "
           # 셸 레벨 클린업: 솔루션 파일(GB급, 쿼터 주범)만 삭제하고 클론(~50MB)은
           # 다음 웨이브에서 git pull로 재사용 (재클론 IO/트래픽 절약)
           f"rm -rf simulation aedt_temp *.lock 2>/dev/null; true")
    r = requests.post(f"{SCHEDULER}/tasks", data={
        "name": name, "remote_cwd": "__SLURM_SCHEDULER_ACCOUNT_WORKSPACE__",
        "command": cmd, "required_capability": "conda:pyaedt2026v1", "env_profile": "pyaedt2026v1",
        "scheduling_profile": "fea_bursty", "cpus": cpus, "memory_mb": mem_mb, "gpus": 0,
    }, allow_redirects=False, timeout=20)
    return r.status_code in (200, 303)


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
