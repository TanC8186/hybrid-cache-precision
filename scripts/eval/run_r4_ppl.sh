#!/usr/bin/env bash
# R4 quality closure: Wikitext-2 PPL for fp16 / uniform_int4 / packed_per_layer, 3 seeds.
# Resumable: skips completed_validated samples. Run with nohup on the 5090.
set -euo pipefail
cd /root/autodl-tmp/MLSys_Research

ATTEMPT="${1:-r4-ppl-20260806}"
LOGDIR="logs"
mkdir -p "$LOGDIR"

for alloc in fp16 uniform_int4 packed_per_layer; do
  for seed in 7 42 2026; do
    if .venv/bin/python scripts/eval/kv_quality_ppl.py \
        --allocation "$alloc" --seed "$seed" \
        --out-dir results/quality/r4-ppl --attempt-id "$ATTEMPT" --resume \
        >> "$LOGDIR/${ATTEMPT}.log" 2>&1; then
      echo "[OK] $alloc seed=$seed" >> "$LOGDIR/${ATTEMPT}.log"
    else
      echo "[FAIL] $alloc seed=$seed" >> "$LOGDIR/${ATTEMPT}.log"
      exit 1
    fi
  done
done

echo "[DONE] $ATTEMPT" >> "$LOGDIR/${ATTEMPT}.log"
