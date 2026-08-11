#!/usr/bin/env bash
set +e
cd /root/autodl-tmp/MLSys_Research || exit 125
source /etc/profile.d/mlsys-data-disk.sh
timeout --signal=TERM --kill-after=30s 21600 \
  bash scripts/bench/run_capacity_phase_diagram.sh \
  formal capacity-phase-repro-r3-20260811 \
  > logs/capacity-phase-repro-r3-20260811.log 2>&1
capacity_r3_exit_code=$?
printf '%s\n' "$capacity_r3_exit_code" \
  > logs/capacity-phase-repro-r3-20260811.exit_code
exit "$capacity_r3_exit_code"
