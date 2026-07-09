#!/bin/bash
set -u

cd "$(dirname "$0")/.."
PY=~/anaconda3/envs/pyaedt2026v1/python.exe
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
LAST=0

while true; do
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
  sleep 3600
done
