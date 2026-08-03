#!/bin/bash
# Generic vllm bench serve driver for 5090 (via ssh).
# usage: bench_driver_5090.sh <model_path> <result_dir> <seed> <log_file> <rate1> [rate2 ...]
# Runs each rate serially against http://127.0.0.1:8000, saves JSON to <result_dir>,
# appends per-rate progress + bench stdout to <log_file>.
set -u

MODEL="$1"; RESULT_DIR="$2"; SEED="$3"; LOG="$4"; shift 4

mkdir -p "$RESULT_DIR"

for R in "$@"; do
  echo "### rate=$R seed=$SEED start $(date '+%F %T')" >> "$LOG"
  /root/autodl-tmp/MLSys_Research/.venv/bin/vllm bench serve \
    --backend openai --base-url http://127.0.0.1:8000 \
    --model "$MODEL" --dataset-name random --num-prompts 400 \
    --request-rate "$R" --max-concurrency 512 --save-result \
    --result-dir "$RESULT_DIR" --random-input-len 1024 --random-output-len 128 \
    --seed "$SEED" --metadata seed="$SEED" >> "$LOG" 2>&1
  echo "### rate=$R seed=$SEED done rc=$? $(date '+%F %T')" >> "$LOG"
done
echo "### ALL DONE $(date '+%F %T')" >> "$LOG"
