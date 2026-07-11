#!/bin/bash
set -u
set -o pipefail

# Deliberately separate from auto_collect_loop.sh: multi-hour training must not
# delay result harvesting.  checkpoint_orchestrator's durable state and locks
# make overlapping launcher attempts defer instead of creating two writers.
cd "$(dirname "$0")/.."
PY=~/anaconda3/envs/pyaedt2026v1/python.exe
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
CHECKPOINT_INTERVAL_SECONDS="${MFT_CHECKPOINT_INTERVAL_SECONDS:-600}"

case "$CHECKPOINT_INTERVAL_SECONDS" in
  ''|*[!0-9]*)
    echo "invalid MFT_CHECKPOINT_INTERVAL_SECONDS=$CHECKPOINT_INTERVAL_SECONDS" >&2
    exit 2
    ;;
esac

PROFILE_ARGS=()
if [ -n "${MFT_QUALITY_PROFILE:-}" ]; then
  PROFILE_ARGS=(--profile "$MFT_QUALITY_PROFILE")
fi

if [ -z "${MFT_SOLVER_REVISION:-}" ] || [ -z "${MFT_LIBRARY_REVISION:-}" ]; then
  echo "MFT_SOLVER_REVISION and MFT_LIBRARY_REVISION are required" >&2
  exit 2
fi
REVISION_ARGS=(
  --solver-revision "$MFT_SOLVER_REVISION"
  --library-revision "$MFT_LIBRARY_REVISION"
)

while true; do
  printf '[checkpoint] start %s\n' "$(date -Iseconds)"
  if ! "$PY" training/checkpoint_orchestrator.py \
      --runtime-root "$PWD" --execute \
      "${REVISION_ARGS[@]}" "${PROFILE_ARGS[@]}" 2>&1 | tail -30; then
    echo "[checkpoint] attempt failed or another worker owns the durable lock" >&2
  fi
  printf '[checkpoint] sleep %ss %s\n' "$CHECKPOINT_INTERVAL_SECONDS" "$(date -Iseconds)"
  sleep "$CHECKPOINT_INTERVAL_SECONDS"
done
