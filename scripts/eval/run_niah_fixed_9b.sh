#!/usr/bin/env bash
# Qwen3.5-9B NIAH rerun with max_tokens>=128 (fixes <think> truncation).
# Core paper allocations: fp16 / uniform_int4 / packed_per_layer.
set -euo pipefail
cd /root/autodl-tmp/MLSys_Research
export VLLM_USE_FLASHINFER_SAMPLER=0
ATTEMPT="${1:-niah-fixed-9b-20260807}"
MAX_TOKENS="${2:-256}"
MODEL="/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-9B/snapshots/master"
LOGDIR="logs"
mkdir -p "$LOGDIR"

for alloc in fp16 uniform_int4 packed_per_layer; do
  for seed in 7 42 2026; do
    for depth in 25 50 75; do
      for len in 2048 4096; do
        if .venv/bin/python scripts/eval/kv_quality_retrieval.py \
            --allocation "$alloc" --seed "$seed" --depth-pct "$depth" --max-len "$len" \
            --num-needles 3 --max-tokens "$MAX_TOKENS" --model "$MODEL" \
            --max-model-len 16384 \
            --out-dir results/quality/niah-fixed-9b \
            --attempt-id "$ATTEMPT" --resume \
            >> "$LOGDIR/${ATTEMPT}.log" 2>&1; then
          echo "[OK] $alloc seed=$seed d=$depth l=$len" >> "$LOGDIR/${ATTEMPT}.log"
        else
          echo "[FAIL] $alloc seed=$seed d=$depth l=$len" >> "$LOGDIR/${ATTEMPT}.log"
          exit 1
        fi
      done
    done
  done
done
echo "[DONE] $ATTEMPT" >> "$LOGDIR/${ATTEMPT}.log"
