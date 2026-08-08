#!/usr/bin/env bash
# GSM8K state-dtype supplement: fp16 attention KV + bf16 SSM state.
# Mirrors reasoning-gsm8k-3seed-20260808 (200 samples, greedy, no-think,
# seeds 7/42/2026) so the only variable is the GDN state storage dtype.
set -euo pipefail
cd /root/autodl-tmp/MLSys_Research
export VLLM_USE_FLASHINFER_SAMPLER=0
ATTEMPT="${1:-reasoning-gsm8k-3seed-statebf16-20260808}"
LOGDIR="logs"
mkdir -p "$LOGDIR"

for seed in 7 42 2026; do
  if .venv/bin/python scripts/eval/reasoning_bench.py \
      --bench gsm8k --allocation fp16_statebf16 --seed "$seed" \
      --out-dir results/quality/reasoning \
      --attempt-id "$ATTEMPT" --disable-thinking --resume \
      >> "$LOGDIR/${ATTEMPT}.log" 2>&1; then
    echo "[OK] gsm8k fp16_statebf16 seed=$seed" >> "$LOGDIR/${ATTEMPT}.log"
  else
    echo "[FAIL] gsm8k fp16_statebf16 seed=$seed" >> "$LOGDIR/${ATTEMPT}.log"
    exit 1
  fi
done
echo "[DONE] $ATTEMPT" >> "$LOGDIR/${ATTEMPT}.log"
