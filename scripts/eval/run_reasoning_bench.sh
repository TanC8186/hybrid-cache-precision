#!/usr/bin/env bash
# Reasoning benchmarks: gsm8k(200) + mmlu(500) + aime25(30) x 5 allocations x 1 seed.
# Greedy (temperature=0) with fixed datasets; engine seed does not change outputs.
# Main protocol: --disable-thinking (chat template, enable_thinking=False) with
# generous budgets (gsm8k 1024 / mmlu 512 / aime25 4096) so the model can state
# a final answer; extraction prefers the last answer marker (strict final).
# Attempts reasoning-20260807 (thinking, small budgets) and
# reasoning-20260807-nothink (no-think, small budgets) are retained as
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
