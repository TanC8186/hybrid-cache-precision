#!/usr/bin/env bash
# NIAH rerun with max_tokens>=128 (fixes <think> truncation artifact) across all
# 5 allocations used in R4/R5: fp16, uniform_int4, packed_per_layer, TurboQuant k8v4/4bit_nc.
set -euo pipefail
cd /root/autodl-tmp/MLSys_Research
export VLLM_USE_FLASHINFER_SAMPLER=0
ATTEMPT="${1:-niah-fixed-20260807}"
MAX_TOKENS="${2:-256}"
LOGDIR="logs"
mkdir -p "$LOGDIR"

for alloc in fp16 uniform_int4 packed_per_layer turboquant_k8v4 turboquant_4bit_nc; do
  for seed in 7 42 2026; do
    for depth in 25 50 75; do
      for len in 2048 4096; do
        if .venv/bin/python scripts/eval/kv_quality_retrieval.py \
            --allocation "$alloc" --seed "$seed" --depth-pct "$depth" --max-len "$len" \
            --num-needles 3 --max-tokens "$MAX_TOKENS" \
            --out-dir results/quality/niah-fixed \
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
