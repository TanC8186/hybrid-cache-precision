#!/usr/bin/env bash
# R5 feasibility MVEx: TurboQuant dtypes on Qwen3.5-2B (engine start + greedy generation).
# Run after R4 matrix completes (GPU is shared).
set -euo pipefail
cd /root/autodl-tmp/MLSys_Research
export VLLM_USE_FLASHINFER_SAMPLER=0
LOGDIR="logs"
mkdir -p "$LOGDIR"

for alloc in turboquant_k8v4 turboquant_4bit_nc; do
  if .venv/bin/python scripts/eval/kv_quality_retrieval.py \
      --allocation "$alloc" --seed 7 --depth-pct 50 --max-len 2048 --num-needles 3 \
      --out-dir results/quality/r5-turboquant --attempt-id r5-turboquant-mvex-20260806 --resume \
      >> "$LOGDIR/r5-turboquant-mvex.log" 2>&1; then
    echo "[OK] $alloc" >> "$LOGDIR/r5-turboquant-mvex.log"
  else
    echo "[FAIL] $alloc" >> "$LOGDIR/r5-turboquant-mvex.log"
    exit 1
  fi
done
echo "[DONE] r5-turboquant-mvex" >> "$LOGDIR/r5-turboquant-mvex.log"
