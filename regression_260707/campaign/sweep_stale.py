"""
스테일 워크디렉토리 스위퍼 (디스크 쿼터 방어).

취소/킬된 태스크는 셸 클린업(rm -rf workdir)이 실행되지 않아 솔루션 파일이
계정 쿼터를 잠식한다 (실측: 계정당 60-70GB). 이 스크립트는 계정별로
"mtime이 오래된(기본 6시간) mft_* 디렉토리"를 지우는 청소 태스크를 제출한다.
실행 중인 태스크의 워크디렉토리는 솔버가 계속 쓰므로 mtime이 최신 -> 안전.

사용:
  python sweep_stale.py               # 6시간 기준
  python sweep_stale.py --hours 12
  python sweep_stale.py --dry-run     # 삭제 없이 목록만
"""
import argparse

import requests

SCHEDULER = "http://127.0.0.1:8000"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--accounts", type=int, default=12, help="계정 분산용 제출 개수")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    mins = int(args.hours * 60)
    action = "-print" if args.dry_run else "-print -exec rm -rf {} +"
    # 솔루션 파일(쿼터 주범)은 {hours}h+, 클론 디렉토리 통짜는 7일+ 방치분만
    # (클론은 웨이브 간 재사용 자산 - 과잉 삭제로 재클론 IO를 유발하지 않음)
    cmd = (f"echo '--- before:'; du -sh . 2>/dev/null; "
           f"find . -maxdepth 2 -type d -path './mft_*/simulation' -mmin +{mins} {action}; "
           # 공유 프로젝트 폴더: simulation 아래 개별 프로젝트(6h+)만 스윕 (폴더 자체는 유지)
           f"find MFT_1MW_2026v1/simulation -mindepth 1 -maxdepth 1 -mmin +{mins} {action} 2>/dev/null; "
           f"find . -maxdepth 1 -type d -name 'mft_*' -mtime +7 {action}; "
           f"echo '--- after:'; du -sh . 2>/dev/null; true")
    ok = 0
    for i in range(args.accounts):
        r = requests.post(f"{SCHEDULER}/tasks", data={
            "name": f"mft-sweep-{i:02d}", "remote_cwd": "__SLURM_SCHEDULER_ACCOUNT_WORKSPACE__",
            "command": cmd, "required_capability": "conda:pyaedt2026v1",
            "env_profile": "pyaedt2026v1", "scheduling_profile": "fea_bursty",
            "cpus": 1, "memory_mb": 1024, "gpus": 0, "priority": 300,
        }, allow_redirects=False, timeout=15)
        ok += r.status_code in (200, 303)
    print(f"sweep tasks: {ok}/{args.accounts} ({'dry-run' if args.dry_run else f'{args.hours}h+ stale 삭제'})")


if __name__ == "__main__":
    main()
