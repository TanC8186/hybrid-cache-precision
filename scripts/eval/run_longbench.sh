#!/usr/bin/env bash
# LongBench v1 subset: 8 English tasks x 50 samples x 1 seed (greedy).
# 2B runs all 5 allocations; 9B runs the core 3 allocations (fp16 / uniform
# int4 / packed per-layer), consistent with the 9B NIAH scale column.
set -euo pipefail
cd /root/autodl-tmp/MLSys_Research
export VLLM_USE_FLASHINFER_SAMPLER=0
ATTEMPT="${1:-longbench-20260807}"
MODEL_2B="/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master"
MODEL_9B="/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-9B/snapshots/master"
LOGDIR="logs"
mkdir -p "$LOGDIR"

run_cell() {
  local model="$1" alloc="$2"
  for task in trec triviaqa samsum lcc repobench-p gov_report qmsum multi_news; do
    if .venv/bin/python scripts/eval/longbench_bench.py \
        --task "$task" --allocation "$alloc" --seed 7 \
        --model "$model" --max-model-len 16384 --max-samples 50 \
        --disable-thinking \
        --out-dir results/quality/longbench \
        --attempt-id "$ATTEMPT" --resume \
        >> "$LOGDIR/${ATTEMPT}.log" 2>&1; then
      echo "[OK] $alloc $task" >> "$LOGDIR/${ATTEMPT}.log"
    else
      echo "[FAIL] $alloc $task" >> "$LOGDIR/${ATTEMPT}.log"
      exit 1
    fi
  done
}

for alloc in fp16 uniform_int4 packed_per_layer turboquant_k8v4 turboquant_4bit_nc; do
  run_cell "$MODEL_2B" "$alloc"
done
for alloc in fp16 uniform_int4 packed_per_layer; do
  run_cell "$MODEL_9B" "$alloc"
done
echo "[DONE] $ATTEMPT" >> "$LOGDIR/${ATTEMPT}.log"
