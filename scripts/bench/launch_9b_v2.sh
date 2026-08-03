#!/bin/bash
# 9B int4 per-layer server launch (standard: venv + VLLM_USE_FLASHINFER_SAMPLER=0)
# NOTE: layer 23 protected as "auto" (follows model bf16) NOT "float16" —
# bf16 model + float16 KV cache triggers flash-attn "query and key must have
# the same dtype" (RuntimeError). "auto" = don't quantize this layer.
source /root/autodl-tmp/MLSys_Research/.venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0
cd /root/autodl-tmp
nohup vllm serve /root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-9B/snapshots/master \
  --kv-cache-dtype int4_per_token_head \
  --kv-cache-dtype-per-layer '{"23":"auto","3":"int4_per_token_head","7":"int4_per_token_head","11":"int4_per_token_head","15":"int4_per_token_head","19":"int4_per_token_head","27":"int4_per_token_head","31":"int4_per_token_head"}' \
  --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.85 \
  > /root/autodl-tmp/serve9b_int4_v2.log 2>&1 &
echo "LAUNCH_PID=$!"
