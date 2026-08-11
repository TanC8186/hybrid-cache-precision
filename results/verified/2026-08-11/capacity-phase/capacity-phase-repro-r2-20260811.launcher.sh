#!/usr/bin/env bash
set +e
cd /root/autodl-tmp/MLSys_Research || exit 125
source /etc/profile.d/mlsys-data-disk.sh
timeout --signal=TERM --kill-after=30s 21600 bash scripts/bench/run_capacity_phase_diagram.sh formal capacity-phase-repro-r2-20260811
capacity_r2_exit_code=$?
printf '%s\n' "$capacity_r2_exit_code" > logs/capacity-phase-repro-r2-20260811.exit_code
exit "$capacity_r2_exit_code"
