#!/usr/bin/env bash
# Reasoning benchmarks: gsm8k(200) + mmlu(500) + aime25(30) x 5 allocations x 3 seeds.
set -euo pipefail
cd /root/autodl-tmp/MLSys_Research
export VLLM_USE_FLASHINFER_SAMPLER=0
ATTEMPT="${1:-reasoning-20260807}"
LOGDIR="logs"
mkdir -p "$LOGDIR"

for alloc in fp16 uniform_int4 packed_per_layer turboquant_k8v4 turboquant_4bit_nc; do
  for bench in gsm8k mmlu aime25; do
    for seed in 7 42 2026; do
      if .venv/bin/python scripts/eval/reasoning_bench.py \
          --bench "$bench" --allocation "$alloc" --seed "$seed" \
          --out-dir results/quality/reasoning \
          --attempt-id "$ATTEMPT" --resume \
          >> "$LOGDIR/${ATTEMPT}.log" 2>&1; then
        echo "[OK] $alloc $bench seed=$seed" >> "$LOGDIR/${ATTEMPT}.log"
      else
        echo "[FAIL] $alloc $bench seed=$seed" >> "$LOGDIR/${ATTEMPT}.log"
        exit 1
      fi
    done
  done
done
echo "[DONE] $ATTEMPT" >> "$LOGDIR/${ATTEMPT}.log"
