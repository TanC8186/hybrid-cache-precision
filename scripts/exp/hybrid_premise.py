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


def make_cache(bits: int, evict_budget: int | None, attn_indices: list[int], model, evict_window: int = 256) -> QuantizedEvictingHybridCache:
    """evict_window 必须 >= chunk_size：驱逐时保护整个当前 chunk，
    否则新 chunk 前缀（分数为 0）会被旧 token 替换，导致因果 mask 错误。
    """
    return QuantizedEvictingHybridCache(
        attention_layer_indices=attn_indices,
        bits=bits,
        granularity="per_token",
        evict_budget=evict_budget,
        evict_window=evict_window,
        config=model.config,
    )


def build_causal_mask(L: int, R: int, device) -> torch.Tensor:
    """构建 [1,1,L,R] 因果 mask。

    前 R-L 列（历史 token，可含被驱逐后的子集）全部可 attend；
    后 L 列（当前 chunk）严格因果（上三角 -inf）。
    驱逐会缩小缓存到 R < 理论长度，因此必须手动构建以匹配实际返回的 KV。
    """
    mask = torch.zeros((1, 1, L, R), device=device)
    if L > 1:
        block = torch.triu(torch.full((L, L), float("-inf"), device=device), diagonal=1)
        mask[0, 0, :, R - L :] = block
    return mask


def patch_attention_recording(model, cache, attn_indices: list[int]):
    """包装 6 层注意力的 forward，把注意力权重喂给 cache 的驱逐打分。

    仅在驱逐开启时调用。返回 restore 函数，必须在运行结束后调用（否则
    wrapper 会捕获旧 cache 污染后续运行）。
    """
    layers = model.model.layers
    originals: dict[int, object] = {}

    def make_wrapper(layer_idx: int, orig):
        def wrapped(hidden_states, **kwargs):
            out, weights = orig(hidden_states, **kwargs)
            if weights is not None:
                cache.record_scores(layer_idx, weights)
            return out, weights
        return wrapped

    for i in attn_indices:
        attn = layers[i].self_attn
        originals[i] = attn.forward
        attn.forward = make_wrapper(i, attn.forward)  # type: ignore[method-assign]

    def restore() -> None:
        for i, orig in originals.items():
            layers[i].self_attn.forward = orig  # type: ignore[method-assign]

    return restore


def chunked_ppl(
    model,
    tokenizer,
    ids: torch.Tensor,
    cache_factory,
    chunk_size: int,
    *,
    use_cache: bool = True,
    attn_indices: list[int] | None = None,
) -> tuple[float, float, float]:
    """分块前向计算 PPL，同时返回量化字节与 FP16 字节（均指 6 层 GQA KV 总计）。

    ids: [1, N]。逐 chunk 处理：每个 chunk 的注意力看到此前（已压缩的）KV 历史。
    返回 (ppl, kv_quant_bytes, kv_fp16_bytes)。
    """
    device = next(model.parameters()).device
    seq_len = ids.shape[1]
    cache = cache_factory()
    evicting = getattr(cache, "evict_budget", None) is not None
    restore_patch = patch_attention_recording(model, cache, attn_indices or []) if (evicting and attn_indices) else None
    try:
        return _chunked_ppl_inner(model, tokenizer, ids, cache, chunk_size, use_cache, evicting)
    finally:
        if restore_patch:
            restore_patch()


def _chunked_ppl_inner(model, tokenizer, ids, cache, chunk_size, use_cache, evicting) -> tuple[float, float, float]:
    device = next(model.parameters()).device
    seq_len = ids.shape[1]
    total_loss = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for start in range(0, seq_len - 1, chunk_size):
            chunk = ids[:, start : start + chunk_size].to(device)
            L = chunk.shape[1]
            pos_ids = torch.arange(start, start + L, device=device).unsqueeze(0)

            # 驱逐时：手动构建因果 mask，宽度 = 实际返回的缓存长度
            mask = None
            if evicting:
                pre = cache.attention_seq_length()
                R = min(pre + L, cache.evict_budget)
                R = max(R, L)
                mask = build_causal_mask(L, R, device)

            outputs = model(
                input_ids=chunk,
                position_ids=pos_ids,
                attention_mask=mask,
                past_key_values=cache if use_cache else None,
                use_cache=use_cache,
            )
            logits = outputs.logits[:, :-1]  # [1, L-1, vocab]
            targets = chunk[:, 1:]
            # 转 fp32 计算 loss，避免 fp16 跨 chunk 累加的舍入误差
            loss = torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum"
            )
            total_loss += loss.item()
            total_tokens += targets.numel()

    ppl = math.exp(total_loss / total_tokens)
    return ppl, cache.total_bytes, cache.total_fp16_bytes


def tokenize_corpus(tokenizer, corpus: str, max_len: int, num_seqs: int) -> list[torch.Tensor]:
    """把整个语料拼接后切成固定 max_len 的序列（保证驱逐预算真实触发）。

    Wikitext 文档常只有几十~几百 token，若按文档截断，驱逐预算（1024/1536）
    永远不触发。拼接成 token 流再切成等长序列，每个序列都是满 2048。
    """
    ids = tokenizer(corpus, return_tensors="pt").input_ids[0]  # [N]
    seqs = []
    for start in range(0, len(ids) - 1, max_len):
        chunk = ids[start : start + max_len]
        if chunk.shape[0] >= 2:
            seqs.append(chunk.unsqueeze(0))
        if len(seqs) >= num_seqs:
            break
    if not seqs:
        raise ValueError("语料太短，无法生成等长序列")
    return seqs


def run_bits(
    model, tokenizer, ids_list, attn_indices, bits_list, evict_budgets, chunk_size, out_path: Path
) -> None:
    import csv
    import math

    rows = []
    configs = [(b, e) for b in bits_list for e in (evict_budgets or [0])]

    for bits, evict in configs:
        t0 = time.time()
        total_loss, total_tokens = 0.0, 0
        qbytes = fbytes = 0.0
        eb = evict if evict else None
        for ids in ids_list:
            ppl, qb, fb = chunked_ppl(
                model, tokenizer, ids,
                lambda b=bits, e=eb: make_cache(b, e, attn_indices, model),
                chunk_size,
                attn_indices=attn_indices,
            )
            total_loss += math.log(ppl) * (ids.shape[1] - 1)  # log-prob 总和
            total_tokens += ids.shape[1] - 1
            qbytes, fbytes = qb, fb
        avg_ppl = math.exp(total_loss / total_tokens)
        rows.append((bits, evict, avg_ppl, qbytes, fbytes, time.time() - t0))
        print(f"bits={bits} evict={evict}: PPL={avg_ppl:.4f} "
              f"KV_bytes={qbytes:.1f} (FP16={fbytes:.1f}) ratio={fbytes / qbytes:.2f}x "
              f"[{time.time() - t0:.0f}s / {len(ids_list)} seqs]")

    # FP16 baseline（不量化）
    total_loss, total_tokens = 0.0, 0
    fbytes0 = 0.0
    for ids in ids_list:
        ppl0, _, fb = chunked_ppl(
            model, tokenizer, ids,
            lambda: make_cache(16, None, attn_indices, model),
            chunk_size,
            attn_indices=attn_indices,
        )
        total_loss += math.log(ppl0) * (ids.shape[1] - 1)
        total_tokens += ids.shape[1] - 1
        fbytes0 = fb
    ppl0 = math.exp(total_loss / total_tokens)
    rows.append((16, 0, ppl0, fbytes0, fbytes0, 0.0))
    print(f"bits=16 (FP16 baseline): PPL={ppl0:.4f} KV_bytes={fbytes0:.1f}")

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
    ap.add_argument("--evict-budget", type=str, default="", help="逗号分隔驱逐 token 预算列表，如 '1024,1536'；空=不驱逐")
    ap.add_argument("--smoke", action="store_true", help="冒烟模式（小语料 + 短序列）")
    ap.add_argument("--out", default="results/ablations/bit_curve.csv")
    ap.add_argument("--corpus", type=str, default=None, help="评测语料文本文件；默认用内置冒烟语料")
    ap.add_argument("--num-seqs", type=int, default=10, help="评测前 N 篇文档（多文档平均 PPL）")
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
        num_seqs = 1
    else:
        corpus = Path(args.corpus).read_text() if args.corpus else _SMOKE_CORPUS
        max_len = args.max_len
        chunk = args.chunk
        num_seqs = args.num_seqs

    ids_list = tokenize_corpus(tokenizer, corpus, max_len, num_seqs)
    print(f"评测序列: {len(ids_list)} 篇 × 最大 {max_len} tokens, chunk={chunk}")

    evict_budgets = [int(x) for x in args.evict_budget.split(",")] if args.evict_budget else []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_bits(model, tokenizer, ids_list, attn_indices, bits_list, evict_budgets, chunk, out)


if __name__ == "__main__":
    main()
