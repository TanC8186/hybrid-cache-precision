#!/usr/bin/env bash
# R5 TurboQuant NIAH quality matrix (after feasibility MVEx passed).
set -euo pipefail
cd /root/autodl-tmp/MLSys_Research
export VLLM_USE_FLASHINFER_SAMPLER=0
ATTEMPT="${1:-r5-turboquant-20260806}"
LOGDIR="logs"
mkdir -p "$LOGDIR"

for alloc in turboquant_k8v4 turboquant_4bit_nc; do
  for seed in 7 42 2026; do
    for depth in 25 50 75; do
      for len in 2048 4096; do
        if .venv/bin/python scripts/eval/kv_quality_retrieval.py \
            --allocation "$alloc" --seed "$seed" --depth-pct "$depth" --max-len "$len" \
            --num-needles 3 --out-dir results/quality/r5-turboquant \
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
