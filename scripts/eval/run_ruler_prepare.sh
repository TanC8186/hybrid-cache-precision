#!/usr/bin/env bash
# Generate RULER-subset datasets (7 tasks x {4096, 8192} x 20 samples, seed 42).
set -euo pipefail
cd /root/autodl-tmp/MLSys_Research
LOGDIR="logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/ruler-prepare-20260807.log"

for task in ruler_niah_single ruler_niah_multikey ruler_niah_multivalue ruler_niah_multiquery ruler_vt ruler_cwe ruler_fwe; do
  for len in 4096 8192; do
    echo "[RUN] $task L$len" >> "$LOG"
    if .venv/bin/python scripts/eval/ruler_prepare.py \
        --task "$task" --length "$len" --num-samples 20 --save-dir data/ruler \
        >> "$LOG" 2>&1; then
      echo "[OK] $task L$len" >> "$LOG"
    else
      echo "[FAIL] $task L$len" >> "$LOG"
      exit 1
    fi
  done
done
echo "[DONE]" >> "$LOG"
