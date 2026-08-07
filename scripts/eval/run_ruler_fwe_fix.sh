#!/usr/bin/env bash
# FWE rerun for the three core allocations with max_tokens=256
# (fixes the official-50-token <think> budget artifact).
set -euo pipefail
cd /root/autodl-tmp/MLSys_Research
export VLLM_USE_FLASHINFER_SAMPLER=0
ATTEMPT="${1:-ruler-fwe-fixed-20260807}"
MAX_TOKENS="${2:-256}"
LOGDIR="logs"
mkdir -p "$LOGDIR"

for alloc in fp16 uniform_int4 packed_per_layer; do
  for length in 4096 8192; do
    if .venv/bin/python scripts/eval/ruler_quality.py \
        --task ruler_fwe --length "$length" --allocation "$alloc" --seed 7 \
        --max-tokens "$MAX_TOKENS" \
        --out-dir results/quality/ruler-subset \
        --attempt-id "$ATTEMPT" --resume \
        >> "$LOGDIR/${ATTEMPT}.log" 2>&1; then
      echo "[OK] $alloc FWE L$length" >> "$LOGDIR/${ATTEMPT}.log"
    else
      echo "[FAIL] $alloc FWE L$length" >> "$LOGDIR/${ATTEMPT}.log"
      exit 1
    fi
  done
done
echo "[DONE] $ATTEMPT" >> "$LOGDIR/${ATTEMPT}.log"
