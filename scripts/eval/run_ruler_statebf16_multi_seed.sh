#!/usr/bin/env bash
# R8 RULER non-zero cells x 2 extra dataset seeds (ARS 2026-08-09).
# Cells: 2B fwe L4096/L8192; 9B niah_multiquery L4096/L8192, fwe L8192.
# New dataset seeds {11,23}; engine seed fixed 7; allocations fp16 +
# fp16_statebf16. Total new runs: 20 (existing seed-42 cells reused).
set -euo pipefail
cd /root/autodl-tmp/MLSys_Research
export VLLM_USE_FLASHINFER_SAMPLER=0
ATTEMPT="${1:-ruler-subset-20260809-multiseed}"
LOGDIR="logs"
mkdir -p "$LOGDIR"

MODEL_2B="/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master"
MODEL_9B="/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-9B/snapshots/master"

run_cell() {
  local task="$1" length="$2" dseed="$3" alloc="$4" model="$5"
  if .venv/bin/python scripts/eval/ruler_quality.py \
      --task "$task" --length "$length" --allocation "$alloc" --seed 7 \
      --dataset-seed "$dseed" --model "$model" \
      --out-dir results/quality/ruler-subset \
      --attempt-id "$ATTEMPT" --max-tokens 256 --resume \
      >> "$LOGDIR/${ATTEMPT}.log" 2>&1; then
    echo "[OK] $task L$length dseed=$dseed $alloc" >> "$LOGDIR/${ATTEMPT}.log"
  else
    echo "[FAIL] $task L$length dseed=$dseed $alloc" >> "$LOGDIR/${ATTEMPT}.log"
    exit 1
  fi
}

for dseed in 11 23; do
  for alloc in fp16 fp16_statebf16; do
    run_cell ruler_fwe 4096 "$dseed" "$alloc" "$MODEL_2B"
    run_cell ruler_fwe 8192 "$dseed" "$alloc" "$MODEL_2B"
    run_cell ruler_niah_multiquery 4096 "$dseed" "$alloc" "$MODEL_9B"
    run_cell ruler_niah_multiquery 8192 "$dseed" "$alloc" "$MODEL_9B"
    run_cell ruler_fwe 8192 "$dseed" "$alloc" "$MODEL_9B"
  done
done
echo "[DONE] $ATTEMPT" >> "$LOGDIR/${ATTEMPT}.log"
