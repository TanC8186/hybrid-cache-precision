#!/usr/bin/env bash
# Wait for the RULER v2 (max_tokens=256) matrix, then run the Qwen3.5-9B
# NIAH rerun (core allocations) on the same GPU.
set -euo pipefail

RULER_V2_LOG="${1:-/root/autodl-tmp/MLSys_Research/logs/ruler-subset-20260807-v2-256.log}"
MAX_WAIT_S="${2:-43200}"
WAITED=0

while true; do
  if grep -q '\[DONE\] ruler-subset-20260807-v2-256' "$RULER_V2_LOG" 2>/dev/null; then
    break
  fi
  if grep -q '\[FAIL\]' "$RULER_V2_LOG" 2>/dev/null; then
    echo "RULER v2 FAILED; not launching 9B" >&2
    exit 3
  fi
  if [ "$WAITED" -ge "$MAX_WAIT_S" ]; then
    echo "timed out waiting for RULER v2" >&2
    exit 2
  fi
  sleep 60
  WAITED=$((WAITED + 60))
done

GATES_LOG="/root/autodl-tmp/MLSys_Research/logs/r5-serving-v3-gates-20260807.log"
if grep -q '\[FAIL\]' "$GATES_LOG"; then
  echo "serving gates FAILED; not launching 9B" >&2
  exit 3
fi

cd /root/autodl-tmp/MLSys_Research
if pgrep -f "run_niah_fixed_9b.sh" >/dev/null 2>&1; then
  echo "9B NIAH already running"
  exit 0
fi
mkdir -p logs
setsid nohup bash scripts/eval/run_niah_fixed_9b.sh niah-fixed-9b-20260807 256 \
  > logs/niah-fixed-9b-20260807.nohup.log 2>&1 < /dev/null &
echo "launched 9B NIAH pid=$!"
