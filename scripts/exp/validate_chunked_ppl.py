"""校验 chunked-with-cache PPL 机械：标准单次前向 PPL vs 分块+FP16 cache PPL。

两者数学上必须完全一致（PPL 与是否缓存无关）。若不一致 → chunked 机械有 bug。
用法: python scripts/exp/validate_chunked_ppl.py
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

MODEL_PATH = Path(__file__).resolve().parents[2] / "data/modelscope_cache/models/Qwen--Qwen3.5-2B/snapshots/master"
CORPUS = Path(__file__).resolve().parents[2] / "data/wikitext2_test.txt"


def plain_ppl(model, tokenizer, ids: torch.Tensor) -> float:
    """标准单次前向（无 cache）：一次性跑完整序列，greedy 下 loss 严格等于 chunked。"""
    device = next(model.parameters()).device
    with torch.no_grad():
        out = model(input_ids=ids.to(device), use_cache=False)
    logits = out.logits[:, :-1].reshape(-1, out.logits.shape[-1])
    targets = ids[0, 1:]
    loss = torch.nn.functional.cross_entropy(logits, targets.to(device), reduction="sum").item()
    return math.exp(loss / targets.numel())


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH), torch_dtype=torch.float16, attn_implementation="eager"
    ).cuda()
    model.eval()

    text = CORPUS.read_text()
    ids = tokenizer(text, return_tensors="pt").input_ids[:, :2048]

    ppl_plain = plain_ppl(model, tokenizer, ids)
    print(f"标准单次前向 PPL: {ppl_plain:.4f}")

    # chunked + FP16 cache（同 harness 的 bits=16 baseline）
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
    from hybrid_premise import attention_layer_indices, chunked_ppl, make_cache

    attn = attention_layer_indices(model)
    ppl_chunked, _, _ = chunked_ppl(
        model, tokenizer, ids, lambda: make_cache(16, None, attn, model), 128, attn_indices=attn
    )
    print(f"chunked+FP16 cache PPL: {ppl_chunked:.4f}")
    diff = abs(ppl_plain - ppl_chunked)
    verdict = "PASS" if diff < 0.05 else "FAIL"  # fp16 数值噪声容差
    print(f"差异: {diff:.5f} → {verdict}")


if __name__ == "__main__":
    main()
