#!/bin/bash
set -u
set -o pipefail

cd "$(dirname "$0")/.."
PY=~/anaconda3/envs/pyaedt2026v1/python.exe
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
COLLECT_INTERVAL_SECONDS="${MFT_COLLECT_INTERVAL_SECONDS:-600}"

case "${MFT_SOLVER_REVISION:-}" in
  [0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F]) ;;
  *) echo "MFT_SOLVER_REVISION must be an explicit full SHA" >&2; exit 2 ;;
esac
case "${MFT_LIBRARY_REVISION:-}" in
  [0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F]) ;;
  *) echo "MFT_LIBRARY_REVISION must be an explicit full SHA" >&2; exit 2 ;;
esac

case "$COLLECT_INTERVAL_SECONDS" in
  ''|*[!0-9]*)
    echo "invalid MFT_COLLECT_INTERVAL_SECONDS=$COLLECT_INTERVAL_SECONDS" >&2
    exit 2
    ;;
esac

while true; do
  printf '[collector] start %s\n' "$(date -Iseconds)"
  "$PY" campaign/collect_wave.py --prefix mft-camp 2>&1 | tail -4
  printf '[collector] sleep %ss %s\n' "$COLLECT_INTERVAL_SECONDS" "$(date -Iseconds)"
  sleep "$COLLECT_INTERVAL_SECONDS"
done
