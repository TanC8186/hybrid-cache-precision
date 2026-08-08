#!/usr/bin/env bash
# Launch the R5 TurboQuant/FP8 protocol-v3 FORMAL matrix (Random60 + ShareGPT300).
# Intended to be run AFTER the MVEx+Pilot gates pass and are reviewed.
set -euo pipefail

OUT_ROOT="${1:-/root/autodl-tmp/r5-serving-20260807}"
cd /root/autodl-tmp/MLSys_Serving_f7a79f5
export PATH="/root/autodl-tmp/MLSys_Research/.venv/bin:/usr/bin:/bin"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
PY=/root/autodl-tmp/MLSys_Research/.venv/bin/python
RUNNER=scripts/bench/run_steady_state.py
LOG=/root/autodl-tmp/MLSys_Research/logs/r5-serving-v3-formal-20260807.log
mkdir -p /root/autodl-tmp/MLSys_Research/logs

run_phase() {
  local config="$1" phase="$2" attempt="$3"
  echo "[RUN] $phase $attempt" >> "$LOG"
  if $PY "$RUNNER" \
      --config "experiments/configs/$config" \
      --phase "$phase" \
      --attempt-id "$attempt" \
      --output-root "$OUT_ROOT" \
      >> "$LOG" 2>&1; then
    echo "[OK] $phase $attempt" >> "$LOG"
  else
    echo "[FAIL] $phase $attempt" >> "$LOG"
    exit 1
  fi
}

run_phase r5_turboquant_protocol_v3_random60_formal.yaml formal r5-tq-v3-random60-formal-20260807
run_phase r5_turboquant_protocol_v3_sharegpt300_formal.yaml formal r5-tq-v3-sharegpt300-formal-20260807
echo "[DONE_FORMAL]" >> "$LOG"
