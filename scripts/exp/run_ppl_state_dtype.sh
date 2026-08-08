#!/usr/bin/env bash
# C4/PG19 PPL for GDN state dtype (fp32 baseline vs bf16 storage).
# Canonical protocol: hybrid_premise.py --bits 16 --seeds 7,42,2026
# --num-seqs 5 --max-len 2048 --chunk 128 --state-dtype <auto|bfloat16>.
# fp32 uses --state-dtype auto (no cache patch, identical to the existing
# PPL-extra fp16 baseline); bf16 uses --state-dtype bfloat16 (state cast at
# every recurrent-state write boundary, simulating vLLM --mamba-ssm-cache-dtype).
set -euo pipefail
cd /root/autodl-tmp/MLSys_Research
ATTEMPT="${1:-ppl-state-20260808}"
LOGDIR="logs"
mkdir -p "$LOGDIR" "results/quality/ppl-state-dtype"

MODEL_2B="/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master"
MODEL_9B="/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-9B/snapshots/master"

run_cell() {
  local corpus="$1" state="$2" model="$3"
  local dtype_arg
  if [ "$state" = "fp32" ]; then
    dtype_arg="auto"
  elif [ "$state" = "bf16" ]; then
    dtype_arg="bfloat16"
  else
    echo "unknown state $state"; return 2
  fi
  local tag
  if [[ "$model" == *"Qwen3.5-9B"* ]]; then tag="9b"; else tag="2b"; fi
  local out="results/quality/ppl-state-dtype/${ATTEMPT}__${corpus}__state${state}__${tag}.csv"
  local args=(--bits 16 --seeds 7,42,2026 --num-seqs 5 --max-len 2048 --chunk 128
              --state-dtype "$dtype_arg" --corpus "data/${corpus}_slice.txt"
              --model "$model" --out "$out")
  if [ -f "$out" ] && [ -f "$out.seeds.csv" ]; then
    echo "[SKIP] $corpus $state $tag (exists)" >> "$LOGDIR/${ATTEMPT}.log"
    return 0
  fi
  if .venv/bin/python scripts/exp/hybrid_premise.py "${args[@]}" \
      >> "$LOGDIR/${ATTEMPT}.log" 2>&1; then
    echo "[OK] $corpus $state $tag" >> "$LOGDIR/${ATTEMPT}.log"
  else
    echo "[FAIL] $corpus $state $tag" >> "$LOGDIR/${ATTEMPT}.log"
    exit 1
  fi
}

for model in "$MODEL_2B" "$MODEL_9B"; do
  for state in fp32 bf16; do
    run_cell c4 "$state" "$model"
    run_cell pg19 "$state" "$model"
  done
done
echo "[DONE] $ATTEMPT" >> "$LOGDIR/${ATTEMPT}.log"
