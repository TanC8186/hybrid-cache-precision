"""vLLM serving benchmark —— 量化 KV 的 serving 指标（5090 上跑）。

对每个分配（fp16 baseline / int4 均匀 / 逐层保护），测：
- 吞吐（out-tokens/sec, req/sec）
- TTFT / TPOT（p50/p99）
- KV cache 容量（总 GPU blocks × block_size）

用法（5090 实例上）:
  python scripts/exp/vllm_serving_bench.py --allocation fp16 --num-reqs 100 --max-len 4096
  python scripts/exp/vllm_serving_bench.py --allocation uniform_int4 ...
  python scripts/exp/vllm_serving_bench.py --allocation results/ablations/allocation.json ...

--allocation 取值: fp16 | uniform_int4 | <json 路径>（per-layer 分配）

注意:
- `fp16` baseline 用 kv_cache_dtype="auto"（跟随模型 bf16），不是 "float16"——
  bf16 模型 + float16 KV cache 会让 flash-attn 报 query/key dtype 不一致。
- 必须保持 disable_log_stats=False（默认），否则 RequestOutput.metrics 为 None，TTFT/TPOT 测不到。
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

MODEL_CANDIDATES = [
    # 本地 dev / 服务器数据盘
    Path("data/modelscope_cache/models/Qwen--Qwen3.5-2B/snapshots/master"),
    Path("/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master"),
]
DEFAULT_ALLOC = {"3": "int4_per_token_head", "7": "int4_per_token_head",
                 "11": "int4_per_token_head", "15": "int4_per_token_head",
                 "19": "int4_per_token_head", "23": "float16"}  # layer23 保护


def resolve_model() -> Path:
    for c in MODEL_CANDIDATES:
        if c.exists():
            return c
    raise SystemExit(
        "未找到模型目录，请用 --model 指定。尝试过: "
        + ", ".join(str(c) for c in MODEL_CANDIDATES)
    )


SEED_TEXT = (
    "The field of language model serving has grown rapidly. In modern inference, "
    "the key-value cache dominates memory for long contexts. Quantizing this cache "
)


def build_requests(num_reqs: int, max_len: int, seed: int = 42) -> list[str]:
    """合成请求：请求间前缀不同（避免 prefix cache 全命中），prompt ~max_len/5。

    max_model_len = max_len，prompt + 生成 ≤ max_len 必须成立；此处 prompt
    取 max_len/5（对应 input:output ≈ 1:4），--max-tokens 默认 64 远小于余量。
    """
    rng = random.Random(seed)
    vocab = ["quantization", "serving", "memory", "latency", "throughput",
             "efficient", "inference", "key-value cache", "blackwell", "batch"]
    target = max(1, max_len // 5)
    prompts = []
    for _ in range(num_reqs):
        # 每个请求从 vocab 随机抽取不同短语前缀，保证请求间前缀不同
        head = " ".join(rng.sample(vocab, 4))
        phrase = head + ". " + SEED_TEXT
        # phrase 约 40-60 token；重复到接近 target
        repeat = max(1, target // 40)
        prompts.append(phrase * repeat)
    return prompts


def percentile(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, int(p / 100 * len(sorted_vals)))
    return sorted_vals[idx]


def report(name: str, vals: list[float]) -> str:
    """p50/p99/mean 字符串；空列表返回 n/a。"""
    if not vals:
        return "n/a"
    s = sorted(vals)
    return f"p50={percentile(s, 50):.1f} p99={percentile(s, 99):.1f} mean={statistics.mean(vals):.1f} ms"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="模型路径（默认自动探测）")
    ap.add_argument("--allocation", required=True,
                    help="fp16 | uniform_int4 | <json path with per-layer map>")
    ap.add_argument("--num-reqs", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=4096, help="max_model_len")
    ap.add_argument("--max-tokens", type=int, default=64, help="每个请求生成多少 token")
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--warmup-n", type=int, default=5, help="预热请求数（不计入统计）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="", help="结果 json 路径（默认 experiments/bench_<alloc>_<ts>.json）")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    model = Path(args.model) if args.model else resolve_model()
    kv_args = {"gpu_memory_utilization": args.gpu_util}
    if args.allocation == "fp16":
        kv_args["kv_cache_dtype"] = "auto"  # 无量化 baseline，跟随模型 bf16
    elif args.allocation == "uniform_int4":
        kv_args["kv_cache_dtype"] = "int4_per_token_head"
    elif args.allocation == "default_alloc":
        kv_args["kv_cache_dtype"] = "int4_per_token_head"
        kv_args["kv_cache_dtype_per_layer"] = DEFAULT_ALLOC
    else:
        alloc = json.loads(Path(args.allocation).read_text())
        kv_args["kv_cache_dtype"] = "int4_per_token_head"
        kv_args["kv_cache_dtype_per_layer"] = alloc
        print(f"per-layer allocation: {alloc}")

    # 离线 LLM 入口默认强制 disable_log_stats=True（vllm/entrypoints/llm.py:226），
    # 会让 RequestOutput.metrics 为 None → 显式关闭，否则 TTFT/TPOT 测不到。
    llm = LLM(
        model=str(model),
        enforce_eager=True,
        max_model_len=args.max_len,
        seed=args.seed,
        disable_log_stats=False,
        **kv_args,
    )

    def run_batch(prompts: list[str]) -> tuple[float, float, list, list]:
        params = SamplingParams(max_tokens=args.max_tokens)
        t0 = time.time()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        elapsed = time.time() - t0
        total_out = sum(len(o.outputs[0].token_ids) for o in outputs)
        ttfts, tpots = [], []
        for o in outputs:
            m = getattr(o, "metrics", None)
            if m is None:
                continue
            if m.first_token_latency and m.first_token_latency > 0:
                ttfts.append(m.first_token_latency * 1000)  # s -> ms
            if m.num_generation_tokens > 1:
                itl = (m.last_token_ts - m.first_token_ts) / (m.num_generation_tokens - 1)
                tpots.append(itl * 1000)
        return elapsed, total_out, ttfts, tpots

    # 预热（丢弃首个请求：JIT/autotune/allocator 一次性开销）
    if args.warmup_n > 0:
        warm = build_requests(args.warmup_n, args.max_len, seed=args.seed)
        run_batch(warm)

    prompts = build_requests(args.num_reqs, args.max_len, seed=args.seed)
    elapsed, total_out, ttfts, tpots = run_batch(prompts)

    # KV cache 容量（profiling 后填充）
    cc = llm.llm_engine.vllm_config.cache_config
    num_blocks = getattr(cc, "num_gpu_blocks", None)
    block_size = getattr(cc, "block_size", None)
    kv_slots = (num_blocks * block_size) if num_blocks and block_size else None

    result = {
        "allocation": args.allocation,
        "kv_args": {k: v for k, v in kv_args.items() if k != "gpu_memory_utilization"},
        "num_reqs": args.num_reqs,
        "max_len": args.max_len,
        "max_tokens": args.max_tokens,
        "gpu_memory_utilization": args.gpu_util,
        "seed": args.seed,
        "throughput_out_tokens_per_sec": total_out / elapsed if elapsed else 0.0,
        "requests_per_sec": args.num_reqs / elapsed if elapsed else 0.0,
        "ttft_p50_ms": percentile(sorted(ttfts), 50),
        "ttft_p99_ms": percentile(sorted(ttfts), 99),
        "ttft_mean_ms": statistics.mean(ttfts) if ttfts else None,
        "tpot_p50_ms": percentile(sorted(tpots), 50),
        "tpot_p99_ms": percentile(sorted(tpots), 99),
        "tpot_mean_ms": statistics.mean(tpots) if tpots else None,
        "kv_cache_blocks": num_blocks,
        "kv_cache_block_size": block_size,
        "kv_cache_total_slots": kv_slots,
    }

    print(f"\n=== serving: {args.allocation} ===")
    print(f"throughput: {result['throughput_out_tokens_per_sec']:.1f} out-tokens/sec, "
          f"{result['requests_per_sec']:.1f} req/sec")
    print(f"TTFT: {report('ttft', ttfts)}")
    print(f"TPOT: {report('tpot', tpots)}")
    print(f"KV: total_slots={kv_slots} (blocks={num_blocks} × block_size={block_size})")
    if kv_slots:
        print(f"  ≈ {kv_slots / 1e6:.2f}M slots; 容量倍数需与 fp16 baseline 对比")

    out_path = Path(args.out) if args.out else Path(
        f"experiments/bench_{args.allocation.replace('/', '_')}_{int(time.time())}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"→ {out_path}")
    print("BENCH DONE")


if __name__ == "__main__":
    main()
