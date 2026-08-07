#!/usr/bin/env bash
# C4/PG19 PPL (extra corpora), canonical transformers protocol:
# hybrid_premise.py --seeds 7,42,2026 --num-seqs 5 --max-len 2048 --chunk 128
# fp16=--bits 16; uniform=--bits 4; packed=--bits 4 --layer-bits '{"23":16}'.
# Models: Qwen3.5-2B and Qwen3.5-9B (scale column).
set -euo pipefail
cd /root/autodl-tmp/MLSys_Research
ATTEMPT="${1:-ppl-extra-20260807}"
LOGDIR="logs"
mkdir -p "$LOGDIR" "results/quality/ppl-extra"

MODEL_2B="/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master"
MODEL_9B="/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-9B/snapshots/master"

run_cell() {
  local corpus="$1" alloc="$2" model="$3"
  local bits layer_bits
  if [ "$alloc" = "fp16" ]; then
    bits="16"; layer_bits=""
  elif [ "$alloc" = "uniform" ]; then
    bits="4"; layer_bits=""
  elif [ "$alloc" = "packed" ]; then
    bits="4"; layer_bits='{"23":16}'
  else
    echo "unknown alloc $alloc"; return 2
  fi
  local out="results/quality/ppl-extra/${ATTEMPT}__${corpus}__${alloc}__$(basename "$model").csv"
  local args=(--bits "$bits" --seeds 7,42,2026 --num-seqs 5 --max-len 2048 --chunk 128
              --corpus "data/${corpus}_slice.txt" --model "$model" --out "$out")
  if [ -n "$layer_bits" ]; then
    args+=(--layer-bits "$layer_bits")
  fi
  if .venv/bin/python scripts/exp/hybrid_premise.py "${args[@]}" \
      >> "$LOGDIR/${ATTEMPT}.log" 2>&1; then
    echo "[OK] $corpus $alloc $(basename "$model")" >> "$LOGDIR/${ATTEMPT}.log"
  else
    echo "[FAIL] $corpus $alloc $(basename "$model")" >> "$LOGDIR/${ATTEMPT}.log"
    exit 1
  fi
}

for model in "$MODEL_2B" "$MODEL_9B"; do
  for alloc in fp16 uniform packed; do
    run_cell c4 "$alloc" "$model"
    run_cell pg19 "$alloc" "$model"
  done
done
echo "[DONE] $ATTEMPT" >> "$LOGDIR/${ATTEMPT}.log"
