"""vLLM 冒烟验证：Qwen3.5-2B 在 vLLM 中运行（fp16 / int4_per_token_head）。

用途：Phase 1 MVP —— 确认 vLLM 已内置的 KV 量化 dtype 能否直接跑混合架构模型。

用法（WSL 内）:
  python scripts/exp/vllm_smoke.py --kv-dtype float16   # 基线
  python scripts/exp/vllm_smoke.py --kv-dtype int4_per_token_head  # 量化 KV
  python scripts/exp/vllm_smoke.py --kv-dtype int4_per_token_head --kv-skip-layers 3  # 逐层（layer3 跳过量化）

env: VLLM_ATTENTION_BACKEND=TRITON_ATTN（4060 无 FA3/4），--enforce-eager
"""
from __future__ import annotations

import argparse
from pathlib import Path

MODEL = "data/modelscope_cache/models/Qwen--Qwen3.5-2B/snapshots/master"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--kv-dtype", default="float16")
    ap.add_argument("--kv-skip-layers", default="", help="逗号分隔要跳过量化的层索引")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=2048)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    kv_args = {"kv_cache_dtype": args.kv_dtype}
    if args.kv_skip_layers:
        kv_args["kv_cache_dtype_skip_layers"] = [int(x) for x in args.kv_skip_layers.split(",")]

    llm = LLM(
        model=str(Path(args.model).resolve()),
        enforce_eager=True,
        max_model_len=args.max_model_len,
        **kv_args,
    )
    out = llm.generate([args.prompt], SamplingParams(max_tokens=args.max_tokens))
    text = out[0].outputs[0].text
    print(f"[{args.kv_dtype}] 输出: {text!r}")
    print(f"[{args.kv_dtype}] 首 token 数: {len(out[0].outputs[0].token_ids)}")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
