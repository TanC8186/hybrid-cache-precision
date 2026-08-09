#!/usr/bin/env bash
# Q-stacking PPL (ARS 2026-08-09 R3): uniform int4 attention KV x
# {fp32,bf16} GDN state on Qwen3.5-2B, C4/PG19, 3 seeds.
# Harness: hybrid_premise.py --bits 4 --state-dtype auto|bfloat16.
set -euo pipefail
cd /root/autodl-tmp/MLSys_Research
ATTEMPT="${1:-ppl-stacking-20260809}"
LOGDIR="logs"
mkdir -p "$LOGDIR" "results/quality/ppl-stacking"

MODEL_2B="/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master"

run_cell() {
  local corpus="$1" state="$2"
  local dtype_arg
  if [ "$state" = "fp32" ]; then
    dtype_arg="auto"
  elif [ "$state" = "bf16" ]; then
    dtype_arg="bfloat16"
  else
    echo "unknown state $state"; return 2
  fi
  local out="results/quality/ppl-stacking/${ATTEMPT}__${corpus}__state${state}__2b.csv"
  local args=(--bits 4 --seeds 7,42,2026 --num-seqs 5 --max-len 2048 --chunk 128
              --state-dtype "$dtype_arg" --corpus "data/${corpus}_slice.txt"
              --model "$MODEL_2B" --out "$out")
  if [ -f "$out" ] && [ -f "$out.seeds.csv" ]; then
    echo "[SKIP] $corpus $state (exists)" >> "$LOGDIR/${ATTEMPT}.log"
    return 0
  fi
  if .venv/bin/python scripts/exp/hybrid_premise.py "${args[@]}" \
      >> "$LOGDIR/${ATTEMPT}.log" 2>&1; then
    echo "[OK] $corpus $state" >> "$LOGDIR/${ATTEMPT}.log"
  else
    echo "[FAIL] $corpus $state" >> "$LOGDIR/${ATTEMPT}.log"
    exit 1
  fi
}

for corpus in c4 pg19; do
  run_cell "$corpus" fp32
  run_cell "$corpus" bf16
done
echo "[DONE] $ATTEMPT" >> "$LOGDIR/${ATTEMPT}.log"
