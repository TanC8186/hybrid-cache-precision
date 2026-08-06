#!/usr/bin/env bash
set -uo pipefail

ROOT=/root/autodl-tmp/MLSys_Serving_f7a79f5
BASE=/root/autodl-tmp/a2-serving-20260805-f7a79f5
ATTEMPT=a2-comparative-serving-random60-formal-v3-piecewise-3108650-westd-01
CONFIG="$ROOT/experiments/configs/a2_comparative_piecewise_protocol_v3_random60_formal.yaml"
SUPERVISOR="$BASE/supervisors/$ATTEMPT/slice-003"

cd "$ROOT"
printf '%s\n' "$$" > "$SUPERVISOR/supervisor.pid"
date -u +%Y-%m-%dT%H:%M:%SZ > "$SUPERVISOR/started_at.txt"
ulimit -Sn > "$SUPERVISOR/soft_nofile_before.txt"
ulimit -n 65535
ulimit -Sn > "$SUPERVISOR/soft_nofile_after.txt"
timeout --signal=TERM --kill-after=30s 3600s bash -lc \
  "ulimit -n 65535; PYTHONPATH=. /root/autodl-tmp/MLSys_Research/.venv/bin/python scripts/bench/run_steady_state.py \
    --config '$CONFIG' \
    --phase comparative_random60_formal_v3 \
    --attempt-id '$ATTEMPT' \
    --parent-attempt a2-comparative-serving-sharegpt300-pilot-v3-piecewise-fc07868-westd-01 \
    --output-root '$BASE/attempts' \
    --resume \
    --max-samples 5" \
  > "$SUPERVISOR/stdout.log" \
  2> "$SUPERVISOR/stderr.log"
ec=$?
printf '%s\n' "$ec" > "$SUPERVISOR/exit_code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$SUPERVISOR/finished_at.txt"
exit "$ec"
