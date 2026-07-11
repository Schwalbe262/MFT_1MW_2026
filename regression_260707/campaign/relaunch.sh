#!/bin/bash
# Restart the non-destructive campaign feeder and periodic collector.
# Usage: bash relaunch.sh [target] [buffer] [solver_revision] [library_revision]
set -u

usage() {
  echo "Usage: bash relaunch.sh [target] [buffer] [solver_40sha] [library_40sha]" >&2
  echo "       defaults: target=50, buffer=0; target+buffer must not exceed 50." >&2
  echo "       Both revisions remain required for training." >&2
}

if [ "$#" -gt 4 ]; then
  usage
  exit 2
fi

TARGET_RAW="${1:-50}"
BUFFER_RAW="${2:-0}"
if ! [[ "$TARGET_RAW" =~ ^[0-9]+$ ]]; then
  echo "target must be a non-negative integer: $TARGET_RAW" >&2
  usage
  exit 2
fi
TARGET=$((10#$TARGET_RAW))
if [ -z "$BUFFER_RAW" ]; then
  BUFFER=0
elif [[ "$BUFFER_RAW" =~ ^[0-9]+$ ]]; then
  BUFFER=$((10#$BUFFER_RAW))
else
  echo "buffer must be a non-negative integer: $BUFFER_RAW" >&2
  usage
  exit 2
fi
if [ $((TARGET + BUFFER)) -gt 50 ]; then
  echo "target+buffer must not exceed 50: $TARGET+$BUFFER" >&2
  usage
  exit 2
fi

SOLVER_REVISION="${3:-}"
LIBRARY_REVISION="${4:-}"
PY=~/anaconda3/envs/pyaedt2026v1/python.exe
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"

if [ ! -x "$PY" ]; then
  echo "Python runtime is unavailable: $PY" >&2
  exit 1
fi
for required_command in curl cygpath powershell.exe nohup sed sleep tail tr; do
  if ! command -v "$required_command" >/dev/null 2>&1; then
    echo "Required command is unavailable: $required_command" >&2
    exit 1
  fi
done

if [ -z "$SOLVER_REVISION" ] || [ -z "$LIBRARY_REVISION" ]; then
  echo "full solver and library revisions are required for feeder and checkpoint training" >&2
  usage
  exit 2
fi
if { [ -n "$SOLVER_REVISION" ] && [ -z "$LIBRARY_REVISION" ]; } || \
   { [ -z "$SOLVER_REVISION" ] && [ -n "$LIBRARY_REVISION" ]; }; then
  echo "solver and library revisions must be supplied together" >&2
  usage
  exit 2
fi

if [ -n "$SOLVER_REVISION" ]; then
  if ! [[ "$SOLVER_REVISION" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "solver revision must be a full 40-character SHA" >&2
    exit 2
  fi
  if ! [[ "$LIBRARY_REVISION" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "library revision must be a full 40-character SHA" >&2
    exit 2
  fi
  SOLVER_REVISION="$(printf '%s' "$SOLVER_REVISION" | tr '[:upper:]' '[:lower:]')"
  LIBRARY_REVISION="$(printf '%s' "$LIBRARY_REVISION" | tr '[:upper:]' '[:lower:]')"

  CURRENT_REVISIONS="$("$PY" -c '
import os
import sys
sys.path.insert(0, os.path.abspath(".."))
import al_driver
solver = al_driver._current_solver_revision()
library = al_driver._current_library_revision()
if int(sys.argv[1]) + int(sys.argv[2]) > 0:
    from campaign import pinned_pilot
    pinned_pilot.validate_p08_completion(solver, library)
print(solver)
print(library)
' "$TARGET" "$BUFFER" 2>&1)" || {
    echo "Local revision validation failed; campaign loops were not stopped:" >&2
    echo "$CURRENT_REVISIONS" >&2
    exit 1
  }
  CURRENT_SOLVER="$(printf '%s\n' "$CURRENT_REVISIONS" | sed -n '1p')"
  CURRENT_LIBRARY="$(printf '%s\n' "$CURRENT_REVISIONS" | sed -n '2p')"
  if ! [[ "$CURRENT_SOLVER" =~ ^[0-9a-f]{40}$ ]] || \
     ! [[ "$CURRENT_LIBRARY" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Local revision validation returned an invalid response: $CURRENT_REVISIONS" >&2
    exit 1
  fi
  if [ "$SOLVER_REVISION" != "$CURRENT_SOLVER" ]; then
    echo "solver revision is not the current vetted local revision: $SOLVER_REVISION != $CURRENT_SOLVER" >&2
    exit 1
  fi
  if [ "$LIBRARY_REVISION" != "$CURRENT_LIBRARY" ]; then
    echo "library revision is not the current clean local revision: $LIBRARY_REVISION != $CURRENT_LIBRARY" >&2
    exit 1
  fi
fi

echo "=== 1. Scheduler health check"
curl -fsS -m 10 "http://127.0.0.1:8000/api/health" > /dev/null || {
  echo "Scheduler is unavailable; aborting relaunch"
  exit 1
}
echo "scheduler ok"

echo "=== 2. Stop and verify existing feeder/collector process trees"
powershell.exe -NoProfile -ExecutionPolicy Bypass \
  -File "$(cygpath -w "$PWD/manage_campaign_loops.ps1")" || exit 1

echo "=== 3. Feeder (target $TARGET, buffer $BUFFER, count 1/task)"
# feeder_state.json is intentionally retained. Resetting it forgets in-flight
# submissions and can duplicate task names or overfill the scheduler.
READY_FILE="$PWD/feeder_relaunch.ready"
: > "$READY_FILE"
nohup "$PY" feeder.py --loop 600 --max-samples 12000 \
  --target "$TARGET" --buffer "$BUFFER" \
  --solver-revision "$SOLVER_REVISION" --library-revision "$LIBRARY_REVISION" \
  --ready-file "$READY_FILE" \
  > feeder_relaunch.log 2>&1 &
FEEDER_PID=$!
echo "feeder pid $FEEDER_PID"
FEEDER_READY=0
for ((attempt = 0; attempt < 90; attempt++)); do
  if [ -s "$READY_FILE" ] && kill -0 "$FEEDER_PID" 2>/dev/null; then
    FEEDER_READY=1
    break
  fi
  if ! kill -0 "$FEEDER_PID" 2>/dev/null; then
    break
  fi
  sleep 1
done
if [ "$FEEDER_READY" -ne 1 ]; then
  echo "Feeder did not complete its first guarded cycle; relaunch aborted. Log tail:" >&2
  tail -n 50 feeder_relaunch.log >&2
  powershell.exe -NoProfile -ExecutionPolicy Bypass \
    -File "$(cygpath -w "$PWD/manage_campaign_loops.ps1")" >/dev/null 2>&1 || true
  exit 1
fi
echo "feeder ready"

echo "=== 4. Periodic collector loop"
MFT_SOLVER_REVISION="$SOLVER_REVISION" \
MFT_LIBRARY_REVISION="$LIBRARY_REVISION" \
nohup bash auto_collect_loop.sh > collect_relaunch.log 2>&1 &
echo "collector pid $!"

echo "=== 5. Independent durable checkpoint loop"
MFT_SOLVER_REVISION="$SOLVER_REVISION" \
MFT_LIBRARY_REVISION="$LIBRARY_REVISION" \
nohup bash auto_checkpoint_loop.sh > checkpoint_relaunch.log 2>&1 &
echo "checkpoint pid $!"

sleep 3
powershell.exe -NoProfile -ExecutionPolicy Bypass \
  -File "$(cygpath -w "$PWD/manage_campaign_loops.ps1")" -VerifyOnly || exit 1

echo "=== Relaunch complete. Inspect campaign/feeder_relaunch.log"
