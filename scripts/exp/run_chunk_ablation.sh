#!/usr/bin/env bash
# R7 harness chunk ablation (ARS 2026-08-09): chunk=1 vs 128 write-boundary
# rounding on 2B C4, 1 seed, 1 seq, fp16 KV. Cells: {auto, bfloat16} x
# {chunk1, chunk128} = 4.
set -euo pipefail
cd /root/autodl-tmp/MLSys_Research
ATTEMPT="${1:-chunk-ablation-20260809}"
LOGDIR="logs"
mkdir -p "$LOGDIR" "results/quality/chunk-ablation"

MODEL_2B="/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master"

run_cell() {
  local state="$1" chunk="$2"
  local dtype_arg
  if [ "$state" = "fp32" ]; then dtype_arg="auto"; else dtype_arg="bfloat16"; fi
  local out="results/quality/chunk-ablation/${ATTEMPT}__state${state}__chunk${chunk}__2b.csv"
  if .venv/bin/python scripts/exp/hybrid_premise.py \
      --bits 16 --seeds 42 --num-seqs 1 --max-len 2048 --chunk "$chunk" \
      --state-dtype "$dtype_arg" --corpus data/c4_slice.txt \
      --model "$MODEL_2B" --out "$out" \
      >> "$LOGDIR/${ATTEMPT}.log" 2>&1; then
    echo "[OK] $state chunk=$chunk" >> "$LOGDIR/${ATTEMPT}.log"
  else
    echo "[FAIL] $state chunk=$chunk" >> "$LOGDIR/${ATTEMPT}.log"
    exit 1
  fi
}

run_cell fp32 128
run_cell fp32 1
run_cell bf16 128
run_cell bf16 1
echo "[DONE] $ATTEMPT" >> "$LOGDIR/${ATTEMPT}.log"
