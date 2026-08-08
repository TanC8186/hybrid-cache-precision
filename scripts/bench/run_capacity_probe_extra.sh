#!/usr/bin/env bash
# M3: Qwen3.5-9B packed per-layer capacity probes (legacy/uniform/packed).
# M4: pure-attention control on Qwen2.5-7B-Instruct (fp16/int4).
# Protocol mirrors the VERIFIED A2 gate (attempt_contract_westd_03):
# inspect_kv_config.py, max_model_len=4096, gpu_memory_utilization=0.85,
# seed=42, --enforce-eager; uniform must pass --kv-cache-dtype-per-layer '{}'.
set -euo pipefail

WORKTREE="/root/autodl-tmp/MLSys_Serving_f7a79f5"
PY="/root/autodl-tmp/MLSys_Research/.venv/bin/python"
OUT_ROOT="${1:-/root/autodl-tmp/capacity-probe-extra-20260808}"
LOG="/root/autodl-tmp/MLSys_Research/logs/capacity-probe-extra-20260808.log"
mkdir -p "$OUT_ROOT" "$(dirname "$LOG")"

MODEL_9B="/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-9B/snapshots/master"
MODEL_ATTN="/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master"
PER_LAYER='{"23":"auto","3":"int4_per_token_head","7":"int4_per_token_head","11":"int4_per_token_head","15":"int4_per_token_head","19":"int4_per_token_head"}'

export PATH="/root/autodl-tmp/MLSys_Research/.venv/bin:/usr/bin:/bin"

run_probe() {
  local name="$1" model="$2" dtype="$3" per_layer="$4" extra="${5:-}"
  local out="$OUT_ROOT/$name.json"
  echo "[RUN] $name" >> "$LOG"
  if (cd "$WORKTREE" && env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
      VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
      "$PY" scripts/bench/inspect_kv_config.py \
      --model "$model" --max-model-len 4096 --gpu-memory-utilization 0.85 --seed 42 \
      --kv-cache-dtype "$dtype" --kv-cache-dtype-per-layer "$per_layer" \
      --enforce-eager $extra --output "$out" >> "$LOG" 2>&1); then
    echo "[OK] $name" >> "$LOG"
  else
    echo "[FAIL] $name" >> "$LOG"
    exit 1
  fi
}

# M3: 9B
run_probe 9b_legacy "$MODEL_9B" int4_per_token_head "$PER_LAYER"
run_probe 9b_uniform "$MODEL_9B" int4_per_token_head '{}'
run_probe 9b_packed "$MODEL_9B" int4_per_token_head "$PER_LAYER" "--enable-per-layer-page-groups --expect-packed-per-layer"

# M4: pure-attention control
run_probe attn_fp16 "$MODEL_ATTN" auto '{}'
run_probe attn_int4 "$MODEL_ATTN" int4_per_token_head '{}'

echo "[DONE] capacity-probe-extra-20260808" >> "$LOG"
