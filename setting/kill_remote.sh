#!/bin/bash
set -euo pipefail

USER_NAME=$(whoami)

echo "[INFO] Target user: $USER_NAME"

# 1️⃣ 현재 내 job 리스트 가져오기
JOB_IDS=$(squeue -h -u "$USER_NAME" -o "%A")

if [ -z "$JOB_IDS" ]; then
    echo "[INFO] No jobs found"
    exit 0
fi

echo "[INFO] Found jobs:"
echo "$JOB_IDS"

# 2️⃣ sbatch를 호출한 PID 추출
PIDS=()

for j in $JOB_IDS; do
    SID=$(scontrol show job "$j" | grep "AllocNode:Sid" | sed -E 's/.*Sid=[^:]+:([0-9]+).*/\1/')
    if [[ -n "$SID" ]]; then
        PIDS+=("$SID")
    fi
done

# 중복 제거
PIDS=($(printf "%s\n" "${PIDS[@]}" | sort -u))

echo "[INFO] Found submitter PIDs:"
printf '%s\n' "${PIDS[@]}"

# 3️⃣ PID 죽이기
for pid in "${PIDS[@]}"; do
    echo "[INFO] Killing PID $pid"
    kill -9 "$pid" 2>/dev/null || true
done

# 4️⃣ 전체 job 강제 종료
echo "[INFO] Cancelling all jobs"
scancel -u "$USER_NAME"

# 5️⃣ 추가 안전장치 (루프/컨트롤러 제거)
pkill -u "$USER_NAME" -f "aedt_runs" || true
pkill -u "$USER_NAME" -f "remote_worker_payload" || true
pkill -u "$USER_NAME" -f "remote_job.sh" || true
pkill -u "$USER_NAME" -f "peets" || true

# 6️⃣ tmp 스풀 제거
echo "[INFO] Cleaning /tmp spool"
rm -rf "/tmp/$USER_NAME/aedt_runs" 2>/dev/null || true

echo "[DONE] cleanup complete"