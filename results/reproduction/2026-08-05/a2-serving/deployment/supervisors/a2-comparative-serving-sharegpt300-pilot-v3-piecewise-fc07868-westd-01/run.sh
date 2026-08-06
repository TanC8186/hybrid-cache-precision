#!/usr/bin/env bash
set -uo pipefail

ROOT=/root/autodl-tmp/MLSys_Serving_f7a79f5
BASE=/root/autodl-tmp/a2-serving-20260805-f7a79f5
ATTEMPT=a2-comparative-serving-sharegpt300-pilot-v3-piecewise-fc07868-westd-01
CONFIG="$ROOT/experiments/configs/a2_comparative_piecewise_protocol_v3_sharegpt300_pilot.yaml"
SUPERVISOR="$BASE/supervisors/$ATTEMPT"

cd "$ROOT"
printf '%s\n' "$$" > "$SUPERVISOR/supervisor.pid"
date -u +%Y-%m-%dT%H:%M:%SZ > "$SUPERVISOR/started_at.txt"
ulimit -Sn > "$SUPERVISOR/soft_nofile_before.txt"
ulimit -n 65535
ulimit -Sn > "$SUPERVISOR/soft_nofile_after.txt"
timeout --signal=TERM --kill-after=30s 7200s bash -lc \
  "ulimit -n 65535; PYTHONPATH=. /root/autodl-tmp/MLSys_Research/.venv/bin/python scripts/bench/run_steady_state.py \
    --config '$CONFIG' \
    --phase comparative_sharegpt300_pilot_v3 \
    --attempt-id '$ATTEMPT' \
    --parent-attempt a2-comparative-serving-random60-pilot-v3-piecewise-fc07868-westd-01 \
    --output-root '$BASE/attempts'" \
  > "$SUPERVISOR/stdout.log" \
  2> "$SUPERVISOR/stderr.log"
ec=$?
printf '%s\n' "$ec" > "$SUPERVISOR/exit_code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$SUPERVISOR/finished_at.txt"
exit "$ec"
