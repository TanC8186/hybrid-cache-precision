#!/usr/bin/env bash
# Wait for the NIAH fixed rerun to finish, then launch the RULER-subset
# quality matrix on the same GPU. Idempotent: safe to re-run.
set -euo pipefail
cd /root/autodl-tmp/MLSys_Research

NIAH_LOG="${1:-logs/niah-fixed-20260807.log}"
RULER_ATTEMPT="${2:-ruler-subset-20260807}"
MAX_WAIT_S="${3:-21600}"
WAITED=0

while ! grep -q '\[DONE\] niah-fixed-20260807' "$NIAH_LOG" 2>/dev/null; do
  if [ "$WAITED" -ge "$MAX_WAIT_S" ]; then
    echo "timed out waiting for NIAH" >&2
    exit 2
  fi
  sleep 60
  WAITED=$((WAITED + 60))
done

if pgrep -f "run_ruler_quality.sh $RULER_ATTEMPT" >/dev/null 2>&1; then
  echo "RULER already running; nothing to do"
  exit 0
fi

mkdir -p logs
setsid nohup bash scripts/eval/run_ruler_quality.sh "$RULER_ATTEMPT" \
  > "logs/${RULER_ATTEMPT}.nohup.log" 2>&1 < /dev/null &
echo "launched RULER attempt=$RULER_ATTEMPT pid=$!"
