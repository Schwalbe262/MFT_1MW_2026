import subprocess
import time
import getpass
from pathlib import Path
import shutil

username = getpass.getuser()
interval_seconds = 8 * 3600
itr = 10

base = Path(".")

def rm_if_exists(path: Path):
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink(missing_ok=True)

def kill_remote_sbatch_jobs_every_minute():
    """
    NAME이 remote_sbatch.sh인 slurm job을 1분마다 죽이는 함수
    """
    while True:
        try:
            print("remote_sbatch.sh slurm job을 종료 (scancel) 시도 중...")
            # squeue에서 이름이 remote_sbatch.sh인 JOBID 리스트를 가져옴
            completed = subprocess.run(
                ["squeue", "-u", username, "--name=remote_sbatch.sh", "--noheader", "--format=%A"],
                capture_output=True,
                text=True,
            )
            jobids = [
                jid.strip() for jid in completed.stdout.strip().splitlines() if jid.strip()
            ]
            if jobids:
                print(f"remote_sbatch.sh 관련 JOBID 찾음: {jobids}. 종료 시도.")
                for jid in jobids:
                    subprocess.run(["scancel", jid])
            else:
                print("remote_sbatch.sh 관련 slurm job 없음.")
        except Exception as e:
            print(f"remote_sbatch.sh 잡 종료 중 오류: {e}")
        time.sleep(60)

import threading

# remote_sbatch.sh 잡을 1분마다 종료하는 쓰레드 실행
killer_thread = threading.Thread(target=kill_remote_sbatch_jobs_every_minute, daemon=True)
killer_thread.start()

while True:
    print(f"모든 작업 종료(scancel) 실행 중... user={username}")
    subprocess.run(["scancel", "-u", username, "--signal=KILL"])

    print("기존 파일/폴더 정리 중...")
    rm_if_exists(base / "simulation")
    rm_if_exists(base / "error")
    rm_if_exists(base / "log")
    rm_if_exists(base / "simul_log")
    rm_if_exists(base / "simulation_log")
    rm_if_exists(base / "batch.log")
    rm_if_exists(base / "info.log")
    rm_if_exists(base / "log.csv")
    rm_if_exists(base / "log.txt")
    rm_if_exists(base / "run_debug.log")
    rm_if_exists(base / "simulation_num.txt")

    (base / "simul_log").mkdir(exist_ok=True)

    time.sleep(10)

    for i in range(itr):
        print(f"{i+1}번째 simulation1.sh 제출 (sbatch) 실행 중...")
        subprocess.run(["sbatch", "simulation1.sh"])
        time.sleep(5)
        subprocess.run(["squeue", "-u", username])
        time.sleep(60)

    print(f"8시간 대기 중... ({interval_seconds}초)")
    time.sleep(interval_seconds)