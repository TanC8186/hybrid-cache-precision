#!/usr/bin/env bash
# GSM8K state-direction 3-seed v2 protocol (ARS 2026-08-09 R1 fix).
# Rows are sampled per seed (random_state=seed) inside reasoning_bench.py, so
# the three seeds are genuine question-subset repeats while decode stays greedy.
# Cells: 2B x {fp16, fp16_statebf16, uniform_int4, uniform_int4_statebf16} x
# seeds {7,42,2026} = 12 cells.
set -euo pipefail
cd /root/autodl-tmp/MLSys_Research
export VLLM_USE_FLASHINFER_SAMPLER=0
ATTEMPT="${1:-reasoning-gsm8k-state3seed-v2-20260809}"
LOGDIR="logs"
mkdir -p "$LOGDIR"

for alloc in fp16 fp16_statebf16 uniform_int4 uniform_int4_statebf16; do
  for seed in 7 42 2026; do
    if .venv/bin/python scripts/eval/reasoning_bench.py \
        --bench gsm8k --allocation "$alloc" --seed "$seed" \
        --out-dir results/quality/reasoning \
        --attempt-id "$ATTEMPT" --disable-thinking --resume \
        >> "$LOGDIR/${ATTEMPT}.log" 2>&1; then
      echo "[OK] gsm8k $alloc seed=$seed" >> "$LOGDIR/${ATTEMPT}.log"
    else
      echo "[FAIL] gsm8k $alloc seed=$seed" >> "$LOGDIR/${ATTEMPT}.log"
      exit 1
    fi
  done
done
echo "[DONE] $ATTEMPT" >> "$LOGDIR/${ATTEMPT}.log"
