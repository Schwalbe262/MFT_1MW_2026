#!/bin/bash
set -u

cd "$(dirname "$0")/.."
PY=~/anaconda3/envs/pyaedt2026v1/python.exe
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
LAST=0
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
  N=$("$PY" -c 'import os, pandas as pd
p = "data/dataset/train.parquet"
print(len(pd.read_parquet(p)) if os.path.isfile(p) else 0)' 2>/dev/null)
  echo "rows=$N"
  for TH in 500 1000 2000 4000 8000; do
    if [ "$N" -ge "$TH" ] && [ "$LAST" -lt "$TH" ]; then
      "$PY" training/checkpoint_train.py 2>&1 | tail -12
      LAST=$TH
    fi
  done
  printf '[collector] sleep %ss %s\n' "$COLLECT_INTERVAL_SECONDS" "$(date -Iseconds)"
  sleep "$COLLECT_INTERVAL_SECONDS"
done
