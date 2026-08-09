#!/usr/bin/env bash
# GSM8K state-direction 9-seed v2 (ARS 2026-08-09 R1 + S3 power extension).
# Seeds {7,42,2026,11,23,31,47,73,97}; 2B x 4 allocations = 36 cells.
set -euo pipefail
cd /root/autodl-tmp/MLSys_Research
export VLLM_USE_FLASHINFER_SAMPLER=0
ATTEMPT="${1:-reasoning-gsm8k-state9seed-v2-20260809}"
SEEDS="${SEEDS:-7 42 2026 11 23 31 47 73 97}"
LOGDIR="logs"
mkdir -p "$LOGDIR"

for alloc in fp16 fp16_statebf16 uniform_int4 uniform_int4_statebf16; do
  for seed in $SEEDS; do
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
