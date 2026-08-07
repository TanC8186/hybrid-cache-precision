#!/usr/bin/env bash
# FWE rerun for all five allocations with max_tokens=256 AND
# enable_thinking=False (chat-template wrapped). Removes the <think>
# budget artifact at the source.
set -euo pipefail
cd /root/autodl-tmp/MLSys_Research
export VLLM_USE_FLASHINFER_SAMPLER=0
ATTEMPT="${1:-ruler-fwe-fixed-nothink-20260807}"
MAX_TOKENS="${2:-256}"
LOGDIR="logs"
mkdir -p "$LOGDIR"

for alloc in fp16 uniform_int4 packed_per_layer turboquant_k8v4 turboquant_4bit_nc; do
  for length in 4096 8192; do
    if .venv/bin/python scripts/eval/ruler_quality.py \
        --task ruler_fwe --length "$length" --allocation "$alloc" --seed 7 \
        --max-tokens "$MAX_TOKENS" --disable-thinking \
        --out-dir results/quality/ruler-subset \
        --attempt-id "$ATTEMPT" --resume \
        >> "$LOGDIR/${ATTEMPT}.log" 2>&1; then
      echo "[OK] $alloc FWE L$length no-think" >> "$LOGDIR/${ATTEMPT}.log"
    else
      echo "[FAIL] $alloc FWE L$length no-think" >> "$LOGDIR/${ATTEMPT}.log"
      exit 1
    fi
  done
done
echo "[DONE] $ATTEMPT" >> "$LOGDIR/${ATTEMPT}.log"
