#!/usr/bin/env bash
# Wait for the R5 serving gates (MVEx+Pilot+capacity), then run the RULER
# v2 matrix with max_tokens=256 (removes <think> budget artifacts), then
# signal 9B via [DONE_RULER_V2].
set -euo pipefail

GATES_LOG="${1:-/root/autodl-tmp/MLSys_Research/logs/r5-serving-v3-gates-20260807.log}"
MAX_WAIT_S="${2:-43200}"
WAITED=0

while true; do
  if grep -q '\[DONE_GATES\]' "$GATES_LOG" 2>/dev/null; then
    break
  fi
  if grep -q '\[FAIL\]' "$GATES_LOG" 2>/dev/null; then
    echo "serving gates FAILED; not launching RULER v2" >&2
    exit 3
  fi
  if [ "$WAITED" -ge "$MAX_WAIT_S" ]; then
    echo "timed out waiting for serving gates" >&2
    exit 2
  fi
  sleep 60
  WAITED=$((WAITED + 60))
done

cd /root/autodl-tmp/MLSys_Research
if pgrep -f "run_ruler_quality.sh ruler-subset-20260807-v2-256" >/dev/null 2>&1; then
  echo "RULER v2 already running"
  exit 0
fi
mkdir -p logs

echo "[RUN] fwe-nothink (fp16/uniform/packed, 256, enable_thinking=False)" >> "$GATES_LOG"
if bash scripts/eval/run_ruler_fwe_nothink.sh ruler-fwe-fixed-nothink-20260807 256 \
    >> "$GATES_LOG" 2>&1; then
  echo "[OK] fwe-nothink" >> "$GATES_LOG"
else
  echo "[FAIL] fwe-nothink" >> "$GATES_LOG"
  exit 1
fi

setsid nohup bash scripts/eval/run_ruler_quality.sh ruler-subset-20260807-v2-256 256 \
  > logs/ruler-subset-20260807-v2-256.nohup.log 2>&1 < /dev/null &
echo "launched RULER v2 pid=$!"
