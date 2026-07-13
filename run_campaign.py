"""캠페인 진입점 (스케줄러 Projects 기능용).

스케줄러의 프로젝트 실행 폼에는 인자 입력이 없으므로, 캠페인 표준 인자를
여기서 고정해 run_simulation_260706.py를 실행한다. 인자를 바꾸고 싶으면
환경변수 MFT_RUN_ARGS로 통째로 오버라이드 가능.

  entrypoints 등록: run_campaign.py|pyaedt2026v1|MFT_1MW_2026
"""
import os
import shlex
import subprocess
import sys

DEFAULT_ARGS = ("--count 20 --thermal --headless "
                "--set percent_error=1.0 --set max_passes=14 --set P_target=1e6 "
                "--set n_explicit_turns=0")


def main():
    args = os.environ.get("MFT_RUN_ARGS", DEFAULT_ARGS)
    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "run_simulation_260706.py")] + shlex.split(args)
    print(f"[run_campaign] exec: {' '.join(cmd)}", flush=True)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
