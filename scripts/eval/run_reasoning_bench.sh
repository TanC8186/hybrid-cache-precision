#!/usr/bin/env bash
# Reasoning benchmarks: gsm8k(200) + mmlu(500) + aime25(30) x 5 allocations x 1 seed.
# Greedy (temperature=0) with fixed datasets; engine seed does not change outputs.
# Main protocol: --disable-thinking. Qwen3.5 thinking traces hit the generation
# budget on a large share of cells (gsm8k 171/200 @256, mmlu 113/500 @128,
# aime25 30/30 @1024 in attempt reasoning-20260807), so "last token" extraction
# becomes a truncation artifact; attempt reasoning-20260807 is retained as
# protocol-sensitivity data and must not be used as the main protocol.
set -euo pipefail
cd /root/autodl-tmp/MLSys_Research
export VLLM_USE_FLASHINFER_SAMPLER=0
ATTEMPT="${1:-reasoning-20260807-nothink}"
LOGDIR="logs"
mkdir -p "$LOGDIR"

for alloc in fp16 uniform_int4 packed_per_layer turboquant_k8v4 turboquant_4bit_nc; do
  for bench in gsm8k mmlu aime25; do
    for seed in 7; do
      if .venv/bin/python scripts/eval/reasoning_bench.py \
          --bench "$bench" --allocation "$alloc" --seed "$seed" \
          --out-dir results/quality/reasoning \
          --attempt-id "$ATTEMPT" --disable-thinking --resume \
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
