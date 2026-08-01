"""混合架构前提验证：Qwen3.5-2B（18 GDN + 6 GQA）的 KV 位宽容差曲线 + 字节预算排序。

验证问题（文献尽职调查后调整的贡献定位）：
1. 6 层 GQA 的 KV 量化容忍度：q4 是否真无损（llama.cpp #21385 断言）？2-bit value 是否瓶颈（TurboQuant）？
2. 位宽 {2,3,4,8} 全保留的 PPL 曲线 → 张力区在哪？
3. （phase 2）等字节预算下驱逐 vs sub-4bit 排序

用法:
  python scripts/exp/hybrid_premise.py --bits 2,4,8 --max-len 2048 --chunk 128
  python scripts/exp/hybrid_premise.py --bits 4 --evict-budget 512   # 驱逐模式
  python scripts/exp/hybrid_premise.py --smoke                       # 小规模验证机械
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from kvcache.cache.quantized_cache import QuantizedEvictingHybridCache

MODEL_PATH = Path(__file__).resolve().parents[2] / "data/modelscope_cache/models/Qwen--Qwen3.5-2B/snapshots/master"

# 本地冒烟语料（机制验证用；正式评测用 Wikitext-2）
_SMOKE_CORPUS = (
    "The quick brown fox jumps over the lazy dog. Language models predict the next word given "
    "the previous context, and their accuracy reflects how well they capture the statistical "
    "structure of natural language. In modern machine learning, transformers process sequences "
    "with self-attention, allowing every position to attend to every prior position. The key "
    "and value matrices are cached during generation to avoid recomputing them, but this cache "
    "grows with the sequence length and becomes the dominant memory cost for long contexts. "
    "Quantizing this cache trades a small loss in fidelity for a large reduction in memory, "
    "enabling longer sequences and larger batches on the same hardware. The attention layer "
    "plays a central role in this tradeoff, as its precision directly determines how faithfully "
    "the model recalls distant information. Recent hybrid architectures combine linear attention "
    "layers, which maintain a fixed-size recurrent state, with a few full attention layers that "
    "still use a growing key value cache. In such models, the memory cost is concentrated in "
    "those few full attention layers, making them the natural target for compression."
)


def attention_layer_indices(model) -> list[int]:
    cfg = model.config
    types = getattr(cfg, "layer_types", None)
    if types is None:
        return list(range(cfg.num_hidden_layers))
    return [i for i, t in enumerate(types) if t == "full_attention"]


def make_cache(bits: int, evict_budget: int | None, attn_indices: list[int], model) -> QuantizedEvictingHybridCache:
    return QuantizedEvictingHybridCache(
        attention_layer_indices=attn_indices,
        bits=bits,
        granularity="per_token",
        evict_budget=evict_budget,
        config=model.config,
    )


def chunked_ppl(
    model,
    tokenizer,
    ids: torch.Tensor,
    cache_factory,
    chunk_size: int,
    *,
    use_cache: bool = True,
) -> tuple[float, float, float]:
    """分块前向计算 PPL，同时返回量化字节与 FP16 字节（均指 6 层 GQA KV 总计）。

    ids: [1, N]。逐 chunk 处理：每个 chunk 的注意力看到此前（已压缩的）KV 历史。
    返回 (ppl, kv_quant_bytes, kv_fp16_bytes)。
    """
    device = next(model.parameters()).device
    seq_len = ids.shape[1]
    cache = cache_factory()
    total_loss = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for start in range(0, seq_len - 1, chunk_size):
            chunk = ids[:, start : start + chunk_size].to(device)
            L = chunk.shape[1]
            pos_ids = torch.arange(start, start + L, device=device).unsqueeze(0)
            outputs = model(
                input_ids=chunk,
                position_ids=pos_ids,
                past_key_values=cache if use_cache else None,
                use_cache=use_cache,
            )
            logits = outputs.logits[:, :-1]  # [1, L-1, vocab]
            targets = chunk[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum"
            )
            total_loss += loss.item()
            total_tokens += targets.numel()

    ppl = math.exp(total_loss / total_tokens)
    return ppl, cache.total_bytes, cache.total_fp16_bytes


def tokenize_corpus(tokenizer, corpus: str, max_len: int) -> torch.Tensor:
    ids = tokenizer(corpus, return_tensors="pt").input_ids
    return ids[:, :max_len]


def run_bits(
    model, tokenizer, ids, attn_indices, bits_list, chunk_size, evict_budget, out_path: Path
) -> None:
    rows = []
    for bits in bits_list:
        t0 = time.time()
        ppl, qbytes, fbytes = chunked_ppl(
            model, tokenizer, ids,
            lambda b=bits: make_cache(b, evict_budget, attn_indices, model),
            chunk_size,
        )
        rows.append((bits, evict_budget, ppl, qbytes, fbytes, time.time() - t0))
        print(f"bits={bits} evict={evict_budget}: PPL={ppl:.4f} KV_bytes={qbytes:.1f} (FP16={fbytes:.1f}) "
              f"ratio={fbytes / qbytes:.2f}x [{time.time() - t0:.0f}s]")

    # FP16 baseline（不量化）
    ppl0, _, fbytes0 = chunked_ppl(
        model, tokenizer, ids,
        lambda: make_cache(16, None, attn_indices, model),
        chunk_size,
    )
    rows.append((16, 0, ppl0, fbytes0, fbytes0, 0.0))
    print(f"bits=16 (FP16 baseline): PPL={ppl0:.4f} KV_bytes={fbytes0:.1f}")

    import csv
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bits", "evict_budget", "ppl", "kv_quant_bytes", "kv_fp16_bytes", "time_s"])
        w.writerows(rows)
    print(f"→ {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", default="2,4,8", help="逗号分隔位宽")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--evict-budget", type=int, default=None, help="驱逐保留 token 数（None=不驱逐）")
    ap.add_argument("--smoke", action="store_true", help="冒烟模式（小语料 + 短序列）")
    ap.add_argument("--out", default="results/ablations/bit_curve.csv")
    ap.add_argument("--corpus", type=str, default=None, help="评测语料文本文件；默认用内置冒烟语料")
    args = ap.parse_args()

    bits_list = [int(b) for b in args.bits.split(",")]

    print(f"加载模型: {MODEL_PATH}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH), torch_dtype=torch.float16, attn_implementation="eager"
    ).cuda()
    model.eval()

    attn_indices = attention_layer_indices(model)
    print(f"全注意力层索引: {attn_indices}（共 {model.config.num_hidden_layers} 层，DeltaNet {model.config.num_hidden_layers - len(attn_indices)} 层）")

    if args.smoke:
        corpus = _SMOKE_CORPUS * 3
        max_len = 512
        chunk = 64
    else:
        corpus = Path(args.corpus).read_text() if args.corpus else _SMOKE_CORPUS
        max_len = args.max_len
        chunk = args.chunk

    ids = tokenize_corpus(tokenizer, corpus, max_len)
    print(f"序列长度: {ids.shape[1]} tokens, chunk={chunk}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_bits(model, tokenizer, ids, attn_indices, bits_list, chunk, args.evict_budget, out)


if __name__ == "__main__":
    main()
