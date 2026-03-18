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