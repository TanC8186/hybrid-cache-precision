"""vLLM serving benchmark —— 量化 KV 的 serving 指标（5090 上跑）。

对每个分配（fp16 baseline / int4 均匀 / 逐层保护），测：
- 吞吐（tokens/sec, requests/sec）
- TTFT / TPOT（p50/p99）
- KV cache 容量（分配的总 blocks / 每 token 的 block 数 → 容量倍数）

用法（在 5090 实例上）:
  python scripts/exp/vllm_serving_bench.py --allocation fp16 --num-reqs 100 --max-len 4096
  python scripts/exp/vllm_serving_bench.py --allocation uniform_int4 ...
  python scripts/exp/vllm_serving_bench.py --allocation results/ablations/allocation.json ...

--allocation 取值: fp16 | uniform_int4 | <json 路径>（per-layer 分配）
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

MODEL = "data/modelscope_cache/models/Qwen--Qwen3.5-2B/snapshots/master"


def build_requests(num_reqs: int, max_len: int) -> list[str]:
    """合成请求：每个请求是一个固定长度的续写提示（简化 workload）。"""
    seed_text = (
        "The field of language model serving has grown rapidly. In modern inference, "
        "the key-value cache dominates memory for long contexts. Quantizing this cache "
    )
    # 凑到 ~max_len/2 tokens 的提示
    return [seed_text * (max_len // 64 + 1) for _ in range(num_reqs)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--allocation", required=True,
                    help="fp16 | uniform_int4 | <json path with per-layer map>")
    ap.add_argument("--num-reqs", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=64, help="每个请求生成多少 token")
    ap.add_argument("--gpu-util", type=float, default=0.90)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    # 解析分配
    kv_args = {"gpu_memory_utilization": args.gpu_util}
    if args.allocation == "fp16":
        kv_args["kv_cache_dtype"] = "float16"
    elif args.allocation == "uniform_int4":
        kv_args["kv_cache_dtype"] = "int4_per_token_head"
    else:
        alloc = json.loads(Path(args.allocation).read_text())
        kv_args["kv_cache_dtype"] = "int4_per_token_head"
        kv_args["kv_cache_dtype_per_layer"] = alloc
        print(f"per-layer allocation: {alloc}")

    llm = LLM(
        model=str(Path(args.model).resolve()),
        enforce_eager=True,
        max_model_len=args.max_len,
        **kv_args,
    )

    prompts = build_requests(args.num_reqs, args.max_len)
    params = SamplingParams(max_tokens=args.max_tokens)

    # 测量
    t0 = time.time()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    elapsed = time.time() - t0

    total_out_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    # TTFT/TPOT 从 engine 的 iteration 统计（vLLM 内置）
    ttft = getattr(llm.engine, "last_ttft", None)
    tpot = getattr(llm.engine, "last_tpot", None)

    # KV cache 容量
    kv_cfg = getattr(llm.engine, "kv_cache_config", None)
    kv_info = "n/a"
    if kv_cfg is not None:
        total_blocks = kv_cfg.num_gpu_blocks
        kv_info = f"total_gpu_blocks={total_blocks}"

    print(f"\n=== serving: {args.allocation} ===")
    print(f"throughput: {total_out_tokens / elapsed:.1f} out-tokens/sec, "
          f"{args.num_reqs / elapsed:.1f} req/sec")
    if ttft:
        print(f"TTFT: p50={ttft.get('p50', 'n/a')} p99={ttft.get('p99', 'n/a')} ms")
    if tpot:
        print(f"TPOT: p50={tpot.get('p50', 'n/a')} p99={tpot.get('p99', 'n/a')} ms")
    print(f"KV: {kv_info}")
    print("BENCH DONE")


if __name__ == "__main__":
    main()
