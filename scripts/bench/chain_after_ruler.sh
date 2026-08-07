#!/usr/bin/env bash
# Wait for the RULER-subset matrix to finish, then run the R5 TurboQuant/FP8
# protocol-v3 serving gates: MVEx (random60 + sharegpt300) then Pilot.
# Formal is intentionally not auto-launched; review the pilot first.
set -euo pipefail

RULER_LOG="${1:-/root/autodl-tmp/MLSys_Research/logs/ruler-subset-20260807.log}"
OUT_ROOT="${2:-/root/autodl-tmp/r5-serving-20260807}"
MAX_WAIT_S="${3:-43200}"
WAITED=0

while ! grep -q '\[DONE\] ruler-subset-20260807' "$RULER_LOG" 2>/dev/null; do
  if [ "$WAITED" -ge "$MAX_WAIT_S" ]; then
    echo "timed out waiting for RULER" >&2
    exit 2
  fi
  sleep 60
  WAITED=$((WAITED + 60))
done

cd /root/autodl-tmp/MLSys_Serving_f7a79f5
PY=/root/autodl-tmp/MLSys_Research/.venv/bin/python
RUNNER=scripts/bench/run_steady_state.py
LOG=/root/autodl-tmp/MLSys_Research/logs/r5-serving-v3-gates-20260807.log
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

run_phase r5_turboquant_protocol_v3_random60_formal.yaml mvex r5-tq-v3-random60-mvex-20260807
run_phase r5_turboquant_protocol_v3_sharegpt300_formal.yaml mvex r5-tq-v3-sharegpt300-mvex-20260807
run_phase r5_turboquant_protocol_v3_random60_formal.yaml pilot r5-tq-v3-random60-pilot-20260807
run_phase r5_turboquant_protocol_v3_sharegpt300_formal.yaml pilot r5-tq-v3-sharegpt300-pilot-20260807
echo "[DONE_GATES]" >> "$LOG"
