#!/usr/bin/env bash
# RULER-subset 9B state-dtype matrix: fp16 (fp32 state) + fp16_statebf16.
# Mirrors ruler-subset-20260807-v2-256 protocol (7 tasks x 2 lengths x seed 7,
# max_tokens=256, thinking=default). Both allocations share one attempt dir so
# the state-dtype comparison is directly readable.
set -euo pipefail
cd /root/autodl-tmp/MLSys_Research
export VLLM_USE_FLASHINFER_SAMPLER=0
ATTEMPT="${1:-ruler-subset-20260808-9b}"
MODEL_9B="/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-9B/snapshots/master"
LOGDIR="logs"
mkdir -p "$LOGDIR"

for alloc in fp16 fp16_statebf16; do
  for task in ruler_niah_single ruler_niah_multikey ruler_niah_multivalue ruler_niah_multiquery ruler_vt ruler_cwe ruler_fwe; do
    for length in 4096 8192; do
      if .venv/bin/python scripts/eval/ruler_quality.py \
          --task "$task" --length "$length" --allocation "$alloc" --seed 7 \
          --model "$MODEL_9B" --out-dir results/quality/ruler-subset \
          --attempt-id "$ATTEMPT" --max-tokens 256 --resume \
          >> "$LOGDIR/${ATTEMPT}.log" 2>&1; then
        echo "[OK] $alloc $task L$length seed=7" >> "$LOGDIR/${ATTEMPT}.log"
      else
        echo "[FAIL] $alloc $task L$length seed=7" >> "$LOGDIR/${ATTEMPT}.log"
        exit 1
      fi
    done
  done
done
echo "[DONE] $ATTEMPT" >> "$LOGDIR/${ATTEMPT}.log"
