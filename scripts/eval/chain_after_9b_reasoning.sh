#!/usr/bin/env bash
# Wait for the Qwen3.5-9B NIAH rerun, then run the 2B reasoning benchmarks.
set -euo pipefail

LOG_9B="${1:-/root/autodl-tmp/MLSys_Research/logs/niah-fixed-9b-20260807.log}"
MAX_WAIT_S="${2:-43200}"
WAITED=0

while true; do
  if grep -q '\[DONE\] niah-fixed-9b-20260807' "$LOG_9B" 2>/dev/null; then
    break
  fi
  if grep -q '\[FAIL\]' "$LOG_9B" 2>/dev/null; then
    echo "9B NIAH FAILED; not launching reasoning" >&2
    exit 3
  fi
  if [ "$WAITED" -ge "$MAX_WAIT_S" ]; then
    echo "timed out waiting for 9B NIAH" >&2
    exit 2
  fi
  sleep 60
  WAITED=$((WAITED + 60))
done

cd /root/autodl-tmp/MLSys_Research
if pgrep -f "run_reasoning_bench.sh" >/dev/null 2>&1; then
  echo "reasoning already running"
  exit 0
fi
mkdir -p logs
setsid nohup bash scripts/eval/run_reasoning_bench.sh reasoning-20260807 \
  > logs/reasoning-20260807.nohup.log 2>&1 < /dev/null &
echo "launched reasoning pid=$!"
