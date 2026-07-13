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
THRESHOLDS="${MFT_QUALITY_THRESHOLDS:-$PWD/training/model_quality_thresholds.json}"
THRESHOLD_ARGS=(--thresholds "$THRESHOLDS")

if [ -z "${MFT_SOLVER_REVISION:-}" ] || [ -z "${MFT_LIBRARY_REVISION:-}" ]; then
  echo "MFT_SOLVER_REVISION and MFT_LIBRARY_REVISION are required" >&2
  exit 2
fi
REVISION_ARGS=(
  --solver-revision "$MFT_SOLVER_REVISION"
  --library-revision "$MFT_LIBRARY_REVISION"
)
OUTPUT_ROOT="$PWD/training"

while true; do
  # Re-evaluate the content contract every cycle.  A profile, threshold,
  # validator, or model-target change gets a fresh evidence root without
  # overwriting the immutable state for the prior contract.
  if ! CONTRACT_KEY="$("$PY" training/checkpoint_contract.py \
      "${PROFILE_ARGS[@]}" "${THRESHOLD_ARGS[@]}")"; then
    echo "[checkpoint] unable to fingerprint the training contract" >&2
    sleep "$CHECKPOINT_INTERVAL_SECONDS"
    continue
  fi
  if ! [[ "$CONTRACT_KEY" =~ ^[0-9a-f]{16}$ ]]; then
    echo "[checkpoint] invalid training contract key: $CONTRACT_KEY" >&2
    sleep "$CHECKPOINT_INTERVAL_SECONDS"
    continue
  fi
  RUN_ROOT="$OUTPUT_ROOT/checkpoint_runs/${MFT_SOLVER_REVISION}-${MFT_LIBRARY_REVISION}-c${CONTRACT_KEY}"
  printf '[checkpoint] start %s\n' "$(date -Iseconds)"
  if ! "$PY" training/checkpoint_orchestrator.py \
      --runtime-root "$PWD" \
      --output-root "$OUTPUT_ROOT" --run-root "$RUN_ROOT" --execute \
      --expected-contract-key "$CONTRACT_KEY" \
      "${REVISION_ARGS[@]}" "${PROFILE_ARGS[@]}" \
      "${THRESHOLD_ARGS[@]}" 2>&1 | tail -30; then
    echo "[checkpoint] attempt failed or another worker owns the durable lock" >&2
  fi
  printf '[checkpoint] sleep %ss %s\n' "$CHECKPOINT_INTERVAL_SECONDS" "$(date -Iseconds)"
  sleep "$CHECKPOINT_INTERVAL_SECONDS"
done
