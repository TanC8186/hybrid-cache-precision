#!/usr/bin/env bash
# P1 chain (ARS R7/R8): chunk ablation -> RULER dataset seeds 11/23 -> 20 cells.
# Run after GPU is free. Fail-fast on any step.
set -euo pipefail
cd /root/autodl-tmp/MLSys_Research
export VLLM_USE_FLASHINFER_SAMPLER=0
CHAIN_LOG="logs/p1-chain-20260809.log"
MODEL_2B="/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master"
MODEL_9B="/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-9B/snapshots/master"

echo "[CHAIN] start" >> "$CHAIN_LOG"
bash scripts/exp/run_chunk_ablation.sh chunk-ablation-20260809
echo "[CHAIN] chunk ablation done" >> "$CHAIN_LOG"

gen() {
  local task="$1" length="$2" seed="$3" model="$4"
  mkdir -p "data/ruler/${task}_L${length}/seed${seed}"
  .venv/bin/python scripts/eval/ruler_prepare.py \
      --task "$task" --length "$length" --random-seed "$seed" \
      --dataset-seed-dir --tokenizer-path "$model" \
      >> "$CHAIN_LOG" 2>&1
  echo "[GEN] $task L$length seed=$seed" >> "$CHAIN_LOG"
}

for seed in 11 23; do
  gen ruler_fwe 4096 "$seed" "$MODEL_2B"
  gen ruler_fwe 8192 "$seed" "$MODEL_2B"
  gen ruler_niah_multiquery 4096 "$seed" "$MODEL_9B"
  gen ruler_niah_multiquery 8192 "$seed" "$MODEL_9B"
  gen ruler_fwe 8192 "$seed" "$MODEL_9B"
done
echo "[CHAIN] ruler datasets done" >> "$CHAIN_LOG"

bash scripts/eval/run_ruler_statebf16_multi_seed.sh ruler-subset-20260809-multiseed
echo "[CHAIN] ruler cells done" >> "$CHAIN_LOG"
echo "[CHAIN] all done" >> "$CHAIN_LOG"
