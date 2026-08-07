#!/usr/bin/env bash
# RULER-subset quality matrix: 7 tasks x 2 lengths x 5 allocations x 3 seeds.
set -euo pipefail
cd /root/autodl-tmp/MLSys_Research
export VLLM_USE_FLASHINFER_SAMPLER=0
ATTEMPT="${1:-ruler-subset-20260807}"
LOGDIR="logs"
mkdir -p "$LOGDIR"

for alloc in fp16 uniform_int4 packed_per_layer turboquant_k8v4 turboquant_4bit_nc; do
  for task in ruler_niah_single ruler_niah_multikey ruler_niah_multivalue ruler_niah_multiquery ruler_vt ruler_cwe ruler_fwe; do
    for length in 4096 8192; do
      for seed in 7 42 2026; do
        if .venv/bin/python scripts/eval/ruler_quality.py \
            --task "$task" --length "$length" --allocation "$alloc" --seed "$seed" \
            --out-dir results/quality/ruler-subset \
            --attempt-id "$ATTEMPT" --resume \
            >> "$LOGDIR/${ATTEMPT}.log" 2>&1; then
          echo "[OK] $alloc $task L$length seed=$seed" >> "$LOGDIR/${ATTEMPT}.log"
        else
          echo "[FAIL] $alloc $task L$length seed=$seed" >> "$LOGDIR/${ATTEMPT}.log"
          exit 1
        fi
      done
    done
  done
done
echo "[DONE] $ATTEMPT" >> "$LOGDIR/${ATTEMPT}.log"
