#!/usr/bin/env bash
# GSM8K 9B state-dtype matrix: fp16 (fp32 state) + fp16_statebf16, 3 seeds.
# Mirrors reasoning-gsm8k-3seed-20260808 (200 samples, greedy, no-think).
set -euo pipefail
cd /root/autodl-tmp/MLSys_Research
export VLLM_USE_FLASHINFER_SAMPLER=0
ATTEMPT="${1:-reasoning-gsm8k-3seed-9b-20260808}"
MODEL_9B="/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-9B/snapshots/master"
LOGDIR="logs"
mkdir -p "$LOGDIR"

for alloc in fp16 fp16_statebf16; do
  for seed in 7 42 2026; do
    if .venv/bin/python scripts/eval/reasoning_bench.py \
        --bench gsm8k --allocation "$alloc" --seed "$seed" \
        --model "$MODEL_9B" --out-dir results/quality/reasoning \
        --attempt-id "$ATTEMPT" --disable-thinking --resume \
        >> "$LOGDIR/${ATTEMPT}.log" 2>&1; then
      echo "[OK] $alloc seed=$seed" >> "$LOGDIR/${ATTEMPT}.log"
    else
      echo "[FAIL] $alloc seed=$seed" >> "$LOGDIR/${ATTEMPT}.log"
      exit 1
    fi
  done
done
echo "[DONE] $ATTEMPT" >> "$LOGDIR/${ATTEMPT}.log"
