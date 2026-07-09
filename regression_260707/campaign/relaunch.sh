#!/bin/bash
# Restart the non-destructive campaign feeder and hourly collector.
# Usage: bash relaunch.sh [target] [buffer]
set -u

TARGET="${1:-130}"
BUFFER="${2:-40}"
PY=~/anaconda3/envs/pyaedt2026v1/python.exe
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"

echo "=== 1. Scheduler health check"
curl -fsS -m 10 "http://127.0.0.1:8000/api/health" > /dev/null || {
  echo "Scheduler is unavailable; aborting relaunch"
  exit 1
}
echo "scheduler ok"

echo "=== 2. Stop and verify existing feeder/collector process trees"
powershell.exe -NoProfile -ExecutionPolicy Bypass \
  -File "$(cygpath -w "$PWD/manage_campaign_loops.ps1")" || exit 1

echo "=== 3. Feeder (target $TARGET, buffer $BUFFER, count 8/task)"
# feeder_state.json is intentionally retained. Resetting it forgets in-flight
# submissions and can duplicate task names or overfill the scheduler.
nohup "$PY" feeder.py --loop 600 --max-samples 12000 \
  --target "$TARGET" --buffer "$BUFFER" > feeder_relaunch.log 2>&1 &
echo "feeder pid $!"

echo "=== 4. Hourly collector and checkpoint loop"
nohup bash auto_collect_loop.sh > collect_relaunch.log 2>&1 &
echo "collector pid $!"

sleep 3
powershell.exe -NoProfile -ExecutionPolicy Bypass \
  -File "$(cygpath -w "$PWD/manage_campaign_loops.ps1")" -VerifyOnly || exit 1

echo "=== Relaunch complete. Inspect campaign/feeder_relaunch.log"
