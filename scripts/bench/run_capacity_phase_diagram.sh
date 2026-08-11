#!/usr/bin/env bash
# Capacity phase-diagram sweep for joint KV/state precision budgeting.
# Usage: bash scripts/bench/run_capacity_phase_diagram.sh <mvex|pilot|formal> [attempt]
set -euo pipefail

cd /root/autodl-tmp/MLSys_Research
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_USE_FLASHINFER_SAMPLER=0

PHASE="${1:-mvex}"
ATTEMPT="${2:-capacity-phase-${PHASE}-20260811}"
OUTDIR="results/verified/2026-08-11/capacity-phase"
LOGDIR="logs"
MODEL_2B="/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master"
MODEL_9B="/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-9B/snapshots/master"

mkdir -p "$OUTDIR" "$LOGDIR"

verify_existing() {
  local out="$1" expected actual
  if [[ ! -f "$out" || ! -f "$out.sha256" ]]; then
    return 1
  fi
  expected="$(tr -d '\r\n' < "$out.sha256")"
  read -r actual _ < <(sha256sum "$out")
  if [[ "$actual" != "$expected" ]]; then
    echo "[INVALID] SHA mismatch: $out" >> "$LOGDIR/${ATTEMPT}.log"
    exit 1
  fi
  return 0
}

run_probe() {
  local model="$1" tag="$2" kv="$3" state="$4" length="$5" util="$6"
  local kv_arg util_tag out
  case "$kv" in
    fp16) kv_arg="auto" ;;
    int4) kv_arg="int4_per_token_head" ;;
    *) echo "unknown KV dtype: $kv" >&2; exit 2 ;;
  esac
  util_tag="${util/./}"
  out="$OUTDIR/${ATTEMPT}__${tag}__kv${kv}__state${state}__L${length}__u${util_tag}.json"
  if verify_existing "$out"; then
    echo "[SKIP] $tag kv=$kv state=$state L=$length util=$util" >> "$LOGDIR/${ATTEMPT}.log"
    return 0
  fi
  if .venv/bin/python scripts/bench/probe_ssm_state_dtype.py \
      --model "$model" --dtype "$state" --max-model-len "$length" \
      --kv-cache-dtype "$kv_arg" --kv-cache-dtype-per-layer '{}' \
      --gpu-memory-utilization "$util" --output "$out" \
      >> "$LOGDIR/${ATTEMPT}.log" 2>&1; then
    echo "[OK] $tag kv=$kv state=$state L=$length util=$util" >> "$LOGDIR/${ATTEMPT}.log"
  else
    echo "[FAIL] $tag kv=$kv state=$state L=$length util=$util" >> "$LOGDIR/${ATTEMPT}.log"
    exit 1
  fi
}

run_core_grid() {
  local model="$1" tag="$2" util length kv state
  shift 2
  local -a lengths=("$@")
  for util in ${UTILS}; do
    for length in "${lengths[@]}"; do
      for kv in fp16 int4; do
        for state in auto bfloat16; do
          run_probe "$model" "$tag" "$kv" "$state" "$length" "$util"
        done
      done
    done
  done
}

case "$PHASE" in
  mvex)
    run_probe "$MODEL_2B" 2b int4 auto 4096 0.85
    run_probe "$MODEL_2B" 2b int4 bfloat16 4096 0.85
    ;;
  pilot)
    UTILS="0.80"
    run_core_grid "$MODEL_2B" 2b 1024 4096 16384 32768
    run_probe "$MODEL_2B" 2b fp16 float16 4096 0.85
    run_probe "$MODEL_2B" 2b int4 float16 4096 0.85
    ;;
  formal)
    UTILS="0.70 0.80 0.90"
    run_core_grid "$MODEL_2B" 2b 1024 2048 4096 8192 16384 32768
    UTILS="0.80 0.90"
    run_core_grid "$MODEL_9B" 9b 2048 4096 16384 32768
    for model_tag in "${MODEL_2B}:2b" "${MODEL_9B}:9b"; do
      model="${model_tag%:*}"
      tag="${model_tag##*:}"
      for length in 4096 16384; do
        for kv in fp16 int4; do
          run_probe "$model" "$tag" "$kv" float16 "$length" 0.85
        done
      done
    done
    ;;
  *)
    echo "phase must be mvex, pilot, or formal" >&2
    exit 2
    ;;
esac

echo "[DONE] phase=$PHASE attempt=$ATTEMPT" >> "$LOGDIR/${ATTEMPT}.log"
