#!/usr/bin/env bash
# Capacity probes for GDN state dtype under fp16 attention KV (ARS 2026-08-09
# R2 / M-2x2). Same probe protocol as the int4 capacity run
# (probe_ssm_state_dtype.py, gpu_mem 0.85, kv_cache_dtype=auto,
# kv_cache_dtype_per_layer={}): 2B x {auto,bfloat16,float16} x L4096,
# 2B x {auto,bfloat16} x L16384, 9B x {auto,bfloat16,float16} x L4096.
set -euo pipefail
cd /root/autodl-tmp/MLSys_Research
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_USE_FLASHINFER_SAMPLER=0
ATTEMPT="${1:-capacity-state-fp16kv-20260809}"
LOGDIR="logs"
mkdir -p "$LOGDIR" "results/verified/2026-08-09/capacity-state-fp16kv"

MODEL_2B="/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master"
MODEL_9B="/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-9B/snapshots/master"

run_probe() {
  local model="$1" dtype="$2" mml="$3" tag
  if [[ "$model" == *"Qwen3.5-9B"* ]]; then tag="9b"; else tag="2b"; fi
  local out="results/verified/2026-08-09/capacity-state-fp16kv/${ATTEMPT}__${tag}__${dtype}__L${mml}.json"
  if [ -f "$out" ] && [ -f "$out.sha256" ]; then
    echo "[SKIP] $tag $dtype L$mml (exists)" >> "$LOGDIR/${ATTEMPT}.log"
    return 0
  fi
  if .venv/bin/python scripts/bench/probe_ssm_state_dtype.py \
      --model "$model" --dtype "$dtype" --max-model-len "$mml" \
      --kv-cache-dtype auto --kv-cache-dtype-per-layer '{}' \
      --gpu-memory-utilization 0.85 --output "$out" \
      >> "$LOGDIR/${ATTEMPT}.log" 2>&1; then
    echo "[OK] $tag $dtype L$mml" >> "$LOGDIR/${ATTEMPT}.log"
  else
    echo "[FAIL] $tag $dtype L$mml" >> "$LOGDIR/${ATTEMPT}.log"
    exit 1
  fi
}

for dtype in auto bfloat16 float16; do
  run_probe "$MODEL_2B" "$dtype" 4096
done
for dtype in auto bfloat16; do
  run_probe "$MODEL_2B" "$dtype" 16384
done
for dtype in auto bfloat16 float16; do
  run_probe "$MODEL_9B" "$dtype" 4096
done
echo "[DONE] $ATTEMPT" >> "$LOGDIR/${ATTEMPT}.log"
