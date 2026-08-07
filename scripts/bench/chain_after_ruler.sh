#!/usr/bin/env bash
# Wait for the RULER-subset matrix to finish, then run the R5 TurboQuant/FP8
# protocol-v3 serving gates: MVEx (random60 + sharegpt300) then Pilot.
# Formal is intentionally not auto-launched; review the pilot first.
set -euo pipefail
export PATH="/root/autodl-tmp/MLSys_Research/.venv/bin:$PATH"

RULER_LOG="${1:-/root/autodl-tmp/MLSys_Research/logs/ruler-subset-20260807.log}"
OUT_ROOT="${2:-/root/autodl-tmp/r5-serving-20260807}"
MAX_WAIT_S="${3:-43200}"
WAITED=0

while true; do
  if grep -q '\[DONE\] ruler-subset-20260807' "$RULER_LOG" 2>/dev/null; then
    break
  fi
  if grep -q '\[FAIL\]' "$RULER_LOG" 2>/dev/null; then
    echo "RULER run FAILED; not launching serving gates" >&2
    exit 3
  fi
  if [ "$WAITED" -ge "$MAX_WAIT_S" ]; then
    echo "timed out waiting for RULER" >&2
    exit 2
  fi
  sleep 60
  WAITED=$((WAITED + 60))
done

cd /root/autodl-tmp/MLSys_Serving_f7a79f5
PY=/root/autodl-tmp/MLSys_Research/.venv/bin/python
RUNNER=scripts/bench/run_steady_state.py
LOG=/root/autodl-tmp/MLSys_Research/logs/r5-serving-v3-gates-20260807.log
mkdir -p /root/autodl-tmp/MLSys_Research/logs

run_phase() {
  local config="$1" phase="$2" attempt="$3" parent="${4:-}"
  echo "[RUN] $phase $attempt" >> "$LOG"
  local args=(
      --config "experiments/configs/$config"
      --phase "$phase"
      --attempt-id "$attempt"
      --output-root "$OUT_ROOT"
      --resume
  )
  if [ -n "$parent" ]; then
    args+=(--parent-attempt "$parent")
  fi
  if $PY "$RUNNER" \
      "${args[@]}" \
      >> "$LOG" 2>&1; then
    echo "[OK] $phase $attempt" >> "$LOG"
  else
    echo "[FAIL] $phase $attempt" >> "$LOG"
    exit 1
  fi
}

run_capacity_probe() {
  local alloc="$1"; shift
  local out="$OUT_ROOT/capacity/${alloc}.json"
  mkdir -p "$(dirname "$out")"
  echo "[RUN] capacity $alloc" >> "$LOG"
  if VLLM_ALLOW_INSECURE_SERIALIZATION=1 $PY scripts/bench/inspect_kv_config.py \
      --model /root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master \
      --max-model-len 4096 --gpu-memory-utilization 0.85 --seed 42 \
      --kv-cache-dtype "$1" --enforce-eager --output "$out" \
      ${EXTRA_ARGS:-} >> "$LOG" 2>&1; then
    echo "[OK] capacity $alloc" >> "$LOG"
  else
    echo "[FAIL] capacity $alloc" >> "$LOG"
    exit 1
  fi
}

run_phase r5_turboquant_protocol_v3_random60_formal.yaml mvex r5-tq-v3-random60-mvex-20260807-b r5-tq-v3-random60-mvex-20260807
run_phase r5_turboquant_protocol_v3_sharegpt300_formal.yaml mvex r5-tq-v3-sharegpt300-mvex-20260807-b
run_phase r5_turboquant_protocol_v3_random60_formal.yaml pilot r5-tq-v3-random60-pilot-20260807-b
run_phase r5_turboquant_protocol_v3_sharegpt300_formal.yaml pilot r5-tq-v3-sharegpt300-pilot-20260807-b

echo "[RUN] fwe-fix (fp16/uniform/packed, max_tokens=256)" >> "$LOG"
if bash /root/autodl-tmp/MLSys_Research/scripts/eval/run_ruler_fwe_fix.sh ruler-fwe-fixed-20260807 256 >> "$LOG" 2>&1; then
  echo "[OK] fwe-fix" >> "$LOG"
else
  echo "[FAIL] fwe-fix" >> "$LOG"
  exit 1
fi

EXTRA_ARGS="" run_capacity_probe fp16 auto
EXTRA_ARGS="" run_capacity_probe uniform_int4 int4_per_token_head
EXTRA_ARGS="--kv-cache-dtype-per-layer {\"23\":\"auto\",\"3\":\"int4_per_token_head\",\"7\":\"int4_per_token_head\",\"11\":\"int4_per_token_head\",\"15\":\"int4_per_token_head\",\"19\":\"int4_per_token_head\"} --enable-per-layer-page-groups" run_capacity_probe packed_per_layer int4_per_token_head
EXTRA_ARGS="" run_capacity_probe turboquant_k8v4 turboquant_k8v4
EXTRA_ARGS="" run_capacity_probe turboquant_4bit_nc turboquant_4bit_nc
EXTRA_ARGS="" run_capacity_probe fp8 fp8

echo "[DONE_GATES]" >> "$LOG"
