#!/usr/bin/env bash
set -uo pipefail

ROOT=/root/autodl-tmp/MLSys_Serving_f7a79f5
BASE=/root/autodl-tmp/a2-serving-20260805-f7a79f5
ATTEMPT=a2-packed-serving-mvex-piecewise-f7a79f5-westd-01
CONFIG="$ROOT/experiments/configs/a2_packed_piecewise_serving_f7a79f5.yaml"
SUPERVISOR="$BASE/supervisors/$ATTEMPT"

cd "$ROOT"
printf '%s\n' "$$" > "$SUPERVISOR/supervisor.pid"
date -u +%Y-%m-%dT%H:%M:%SZ > "$SUPERVISOR/started_at.txt"
timeout --signal=TERM --kill-after=30s 1800s bash -lc \
  "PYTHONPATH=. /root/autodl-tmp/MLSys_Research/.venv/bin/python scripts/bench/run_steady_state.py \
    --config '$CONFIG' \
    --phase packed_mvex \
    --attempt-id '$ATTEMPT' \
    --parent-attempt a2-packed-serving-debug-piecewise-f7a79f5-westd-01 \
    --output-root '$BASE/attempts'" \
  > "$SUPERVISOR/stdout.log" \
  2> "$SUPERVISOR/stderr.log"
ec=$?
printf '%s\n' "$ec" > "$SUPERVISOR/exit_code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$SUPERVISOR/finished_at.txt"
exit "$ec"
