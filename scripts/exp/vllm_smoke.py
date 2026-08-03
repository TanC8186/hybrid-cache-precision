"""vLLM 冒烟验证：Qwen3.5-2B 在 vLLM 中运行（auto / int4_per_token_head / per-layer）。

用途：Phase 1 MVP —— 确认 vLLM 已内置的 KV 量化 dtype 能否直接跑混合架构模型。

用法（5090 / WSL 内）:
  python scripts/exp/vllm_smoke.py --kv-dtype auto                     # 基线（bf16，跟随模型）
  python scripts/exp/vllm_smoke.py --kv-dtype int4_per_token_head      # 量化 KV
  python scripts/exp/vllm_smoke.py --kv-dtype int4_per_token_head --kv-skip-layers 3
  python scripts/exp/vllm_smoke.py --kv-dtype int4_per_token_head --kv-per-layer 23:float16,3:int4_per_token_head

注意: 不要用 --kv-dtype float16 —— bf16 模型 + fp16 KV cache 会让 flash-attn 报
`query and key must have the same dtype`。非量化基线用 auto（跟随模型 bf16）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

MODEL_CANDIDATES = [
    # 本地 dev / 服务器数据盘
    Path("data/modelscope_cache/models/Qwen--Qwen3.5-2B/snapshots/master"),
    Path("/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master"),
]


def resolve_model(flag: str) -> Path:
    if flag:
        return Path(flag)
    for c in MODEL_CANDIDATES:
        if c.exists():
            return c
    raise SystemExit(
        "未找到模型目录，请用 --model 指定。尝试过: "
        + ", ".join(str(c) for c in MODEL_CANDIDATES)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="模型路径（默认自动探测）")
    ap.add_argument("--kv-dtype", default="auto",
                    help="auto（基线，跟随模型 bf16）| int4_per_token_head | ...")
    ap.add_argument("--kv-skip-layers", default="", help="逗号分隔要跳过量化的层索引")
    ap.add_argument("--kv-per-layer", default="", help="per-layer dtype: '23:float16,3:int4_per_token_head' 等")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    model = resolve_model(args.model)
    kv_args = {"kv_cache_dtype": args.kv_dtype}
    if args.kv_skip_layers:
        kv_args["kv_cache_dtype_skip_layers"] = [int(x) for x in args.kv_skip_layers.split(",")]
    if args.kv_per_layer:
        kv_args["kv_cache_dtype_per_layer"] = {
            k.strip(): v.strip() for k, v in (p.split(":") for p in args.kv_per_layer.split(","))
        }
        print(f"per-layer dtype: {kv_args['kv_cache_dtype_per_layer']}")

    llm = LLM(
        model=str(model.resolve()),
        enforce_eager=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.82,  # 4060 8GB 只剩 ~6.9GB，0.92 默认会 OOM 报错
        seed=args.seed,
        **kv_args,
    )
    out = llm.generate([args.prompt], SamplingParams(max_tokens=args.max_tokens))
    text = out[0].outputs[0].text
    print(f"[{args.kv_dtype}] 输出: {text!r}")
    print(f"[{args.kv_dtype}] 首 token 数: {len(out[0].outputs[0].token_ids)}")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
