#!/usr/bin/env bash
# Wait for the Qwen3.5-9B NIAH rerun, then run the FWE disable-thinking
# matrix (all 5 allocations), then the 2B reasoning benchmarks.
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

FWE_LOG=/root/autodl-tmp/MLSys_Research/logs/ruler-fwe-fixed-nothink-20260807.log
echo "[RUN] fwe-nothink-all (5 allocs, 256, enable_thinking=False)" >> "$FWE_LOG"
if bash scripts/eval/run_ruler_fwe_nothink.sh ruler-fwe-fixed-nothink-20260807 256 \
    >> "$FWE_LOG" 2>&1; then
  echo "[OK] fwe-nothink-all" >> "$FWE_LOG"
else
  echo "[FAIL] fwe-nothink-all" >> "$FWE_LOG"
  exit 1
fi

if pgrep -f "run_reasoning_bench.sh" >/dev/null 2>&1; then
  echo "reasoning already running"
  exit 0
fi
mkdir -p logs
setsid nohup bash scripts/eval/run_reasoning_bench.sh reasoning-20260807 \
  > logs/reasoning-20260807.nohup.log 2>&1 < /dev/null &
echo "launched reasoning pid=$!"
