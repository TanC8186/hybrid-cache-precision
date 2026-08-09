#!/usr/bin/env bash
# Run one phase of the statebf16 int4-KV serving matrix (ARS 2026-08-09 R4).
# Usage: bash scripts/bench/run_statebf16_serving.sh <config> <phase> <attempt> [output_root]
# Run in the serving worktree (/root/autodl-tmp/MLSys_Serving_f7a79f5).
set -euo pipefail

CONFIG="${1:?config name}"
PHASE="${2:?phase (mvex|pilot|formal)}"
ATTEMPT="${3:?attempt id}"
OUT_ROOT="${4:-/root/autodl-tmp/statebf16-serving-20260809}"
cd /root/autodl-tmp/MLSys_Serving_f7a79f5
export PATH="/root/autodl-tmp/MLSys_Research/.venv/bin:/usr/bin:/bin"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
PY=/root/autodl-tmp/MLSys_Research/.venv/bin/python
RUNNER=scripts/bench/run_steady_state.py
LOG=/root/autodl-tmp/MLSys_Research/logs/statebf16-serving-20260809.log
mkdir -p /root/autodl-tmp/MLSys_Research/logs

echo "[RUN] $PHASE $ATTEMPT" >> "$LOG"
if $PY "$RUNNER" \
    --config "experiments/configs/$CONFIG" \
    --phase "$PHASE" \
    --attempt-id "$ATTEMPT" \
    --output-root "$OUT_ROOT" \
    >> "$LOG" 2>&1; then
  echo "[OK] $PHASE $ATTEMPT" >> "$LOG"
else
  echo "[FAIL] $PHASE $ATTEMPT" >> "$LOG"
  exit 1
fi
