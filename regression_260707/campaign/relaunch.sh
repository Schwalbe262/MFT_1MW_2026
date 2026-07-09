#!/bin/bash
# 스케줄러 수술 후 원클릭 재가동 (Git Bash에서 실행)
# 사용: bash relaunch.sh [target]   (기본 130)
set -u
TARGET="${1:-130}"
PY=~/anaconda3/envs/pyaedt2026v1/python.exe
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")"

echo "=== 1. 스케줄러 헬스체크"
curl -s -m 10 "http://127.0.0.1:8000/api/tasks?limit=1" > /dev/null || { echo "스케줄러 응답 없음 - 중단"; exit 1; }
echo ok

echo "=== 2. 장부 초기화 + 피더 (target $TARGET, count20/무지터/하드캡)"
$PY - << EOF
import json
json.dump({"serial": 6000, "submitted_samples": 0, "outstanding": []},
          open("feeder_state.json", "w"))
EOF
nohup $PY feeder.py --loop 600 --max-samples 12000 --target "$TARGET" > feeder_relaunch.log 2>&1 &
echo "feeder pid $!"

echo "=== 3. 시간당 회수 + 체크포인트 루프"
nohup bash -c 'cd ..; PY=~/anaconda3/envs/pyaedt2026v1/python.exe; export PYTHONIOENCODING=utf-8; LAST=0
while true; do
  $PY campaign/collect_wave.py --prefix mft-camp 2>&1 | tail -2
  N=$($PY -c "import pandas as pd,os;p=\"data/dataset/train.parquet\";print(len(pd.read_parquet(p)) if os.path.isfile(p) else 0)" 2>/dev/null)
  echo "rows=$N"
  for TH in 500 1000 2000 4000 8000; do
    if [ "$N" -ge "$TH" ] && [ "$LAST" -lt "$TH" ]; then $PY training/checkpoint_train.py 2>&1 | tail -12; LAST=$TH; fi
  done
  sleep 3600
done' > collect_relaunch.log 2>&1 &
echo "collector pid $!"

echo "=== 완료. 확인: tail -f campaign/feeder_relaunch.log"
