#!/usr/bin/env bash
# RULER-subset state-dtype supplement: fp16 attention KV + bf16 SSM state.
# Mirrors the fp16 cells of ruler-subset-20260807-v2-256 (7 tasks x 2 lengths
# x 1 seed, max_tokens=256, thinking=default) so the only variable is the GDN
# state storage dtype (fp32 baseline vs bf16).
set -euo pipefail
cd /root/autodl-tmp/MLSys_Research
export VLLM_USE_FLASHINFER_SAMPLER=0
ATTEMPT="${1:-ruler-subset-20260808-statebf16}"
LOGDIR="logs"
mkdir -p "$LOGDIR"

for task in ruler_niah_single ruler_niah_multikey ruler_niah_multivalue ruler_niah_multiquery ruler_vt ruler_cwe ruler_fwe; do
  for length in 4096 8192; do
    if .venv/bin/python scripts/eval/ruler_quality.py \
        --task "$task" --length "$length" --allocation fp16_statebf16 --seed 7 \
        --out-dir results/quality/ruler-subset \
        --attempt-id "$ATTEMPT" --max-tokens 256 --resume \
        >> "$LOGDIR/${ATTEMPT}.log" 2>&1; then
      echo "[OK] $task L$length fp16_statebf16 seed=7" >> "$LOGDIR/${ATTEMPT}.log"
    else
      echo "[FAIL] $task L$length fp16_statebf16 seed=7" >> "$LOGDIR/${ATTEMPT}.log"
      exit 1
    fi
  done
done
echo "[DONE] $ATTEMPT" >> "$LOGDIR/${ATTEMPT}.log"
