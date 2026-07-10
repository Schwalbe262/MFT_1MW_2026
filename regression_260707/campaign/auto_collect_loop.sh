#!/bin/bash
set -u
set -o pipefail

cd "$(dirname "$0")/.."
PY=~/anaconda3/envs/pyaedt2026v1/python.exe
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
COLLECT_INTERVAL_SECONDS="${MFT_COLLECT_INTERVAL_SECONDS:-600}"

case "$COLLECT_INTERVAL_SECONDS" in
  ''|*[!0-9]*)
    echo "invalid MFT_COLLECT_INTERVAL_SECONDS=$COLLECT_INTERVAL_SECONDS" >&2
    exit 2
    ;;
esac

while true; do
  printf '[collector] start %s\n' "$(date -Iseconds)"
  "$PY" campaign/collect_wave.py --prefix mft-camp 2>&1 | tail -4
  # This persistent controller counts recomputed strict-full rows, snapshots
  # one due checkpoint, trains every required model, and leaves the prior
  # registry generation active on any failure.  A failed quality gate is
  # retried on the next collector cycle without stopping data collection.
  if ! "$PY" training/checkpoint_orchestrator.py \
      --runtime-root "$PWD" --execute 2>&1 | tail -30; then
    echo "[collector] checkpoint retraining failed; state preserved for retry" >&2
  fi
  printf '[collector] sleep %ss %s\n' "$COLLECT_INTERVAL_SECONDS" "$(date -Iseconds)"
  sleep "$COLLECT_INTERVAL_SECONDS"
done
