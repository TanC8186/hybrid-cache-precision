#!/usr/bin/env bash
# R4 quality closure: PPL (transformers canonical protocol) + NIAH (vLLM offline greedy).
# Resumable per sample; run with nohup on the 5090.
set -euo pipefail
cd /root/autodl-tmp/MLSys_Research

ATTEMPT="${1:-r4-20260806}"
LOGDIR="logs"
mkdir -p "$LOGDIR"
export VLLM_USE_FLASHINFER_SAMPLER=0
MODEL="/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master"

run_ppl() {
  local alloc="$1" bits="$2" layer_bits="${3:-}"
  local args=(--model "$MODEL" --corpus data/wikitext2_test.txt --num-seqs 5
              --max-len 2048 --seeds 7,42,2026 --bits "$bits"
              --out "results/quality/r4-ppl/${alloc}.csv")
  if [ -n "$layer_bits" ]; then
    args+=(--layer-bits "$layer_bits")
  fi
  if .venv/bin/python scripts/exp/hybrid_premise.py "${args[@]}" \
      >> "$LOGDIR/${ATTEMPT}.ppl.log" 2>&1; then
    echo "[OK] ppl $alloc" >> "$LOGDIR/${ATTEMPT}.ppl.log"
  else
    echo "[FAIL] ppl $alloc" >> "$LOGDIR/${ATTEMPT}.ppl.log"
    exit 1
  fi
}

run_ppl fp16 16
run_ppl uniform_int4 4
run_ppl packed_per_layer 4 '{"23":16}'

for alloc in fp16 uniform_int4 packed_per_layer; do
  for seed in 7 42 2026; do
    for depth in 25 50 75; do
      for len in 2048 4096; do
        if .venv/bin/python scripts/eval/kv_quality_retrieval.py \
            --allocation "$alloc" --seed "$seed" --depth-pct "$depth" --max-len "$len" \
            --num-needles 3 --out-dir results/quality/r4-niah \
            --attempt-id "${ATTEMPT}-niah" --resume \
            >> "$LOGDIR/${ATTEMPT}.niah.log" 2>&1; then
          echo "[OK] niah $alloc seed=$seed d=$depth l=$len" >> "$LOGDIR/${ATTEMPT}.niah.log"
        else
          echo "[FAIL] niah $alloc seed=$seed d=$depth l=$len" >> "$LOGDIR/${ATTEMPT}.niah.log"
          exit 1
        fi
      done
    done
  done
done

echo "[DONE] $ATTEMPT" >> "$LOGDIR/${ATTEMPT}.log"
