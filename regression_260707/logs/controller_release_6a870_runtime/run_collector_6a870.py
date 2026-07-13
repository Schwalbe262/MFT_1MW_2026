"""Single-instance Y-only strict collector loop for the B171 campaign."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

from filelock import FileLock


PROJECT_ROOT = Path(r"Y:\git\MFT_1MW_2026")
REGRESSION_ROOT = PROJECT_ROOT / "regression_260707"
RUNTIME_ROOT = REGRESSION_ROOT / "logs" / "controller_release_6a870_runtime"
LOCK_ROOT = RUNTIME_ROOT / "locks"
INTERVAL_SECONDS = 300


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    dataset = REGRESSION_ROOT / "data" / "dataset" / "train.parquet"
    if str(dataset).upper().startswith("C:\\"):
        raise RuntimeError("C: dataset is forbidden")
    command = [
        sys.executable,
        "campaign/collect_wave.py",
        "--prefix",
        "mft-camp",
        "--running-fetch-limit",
        "0",
    ]
    loop_lock = FileLock(str(LOCK_ROOT / "collector-loop.lock"), timeout=0)
    with loop_lock:
        while True:
            print(f"[collector] start {_stamp()}", flush=True)
            completed = subprocess.run(
                command,
                cwd=str(REGRESSION_ROOT),
                check=False,
            )
            print(
                f"[collector] exit={completed.returncode}; "
                f"sleep={INTERVAL_SECONDS}s {_stamp()}",
                flush=True,
            )
            time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
