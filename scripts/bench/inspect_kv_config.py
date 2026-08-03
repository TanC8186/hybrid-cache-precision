"""Inspect the KV cache config for per-layer dtype allocation.

Builds an offline LLM with the per-layer args and dumps:
- cache_config num_gpu_blocks / block_size / cache_dtype
- kv_cache_config groups, per-layer specs, page sizes
- total slots
Run from server venv with a free GPU.
"""
from __future__ import annotations

import json

from vllm import LLM

MODEL = "/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master"
PER_LAYER = {
    "23": "auto",
    "3": "int4_per_token_head",
    "7": "int4_per_token_head",
    "11": "int4_per_token_head",
    "15": "int4_per_token_head",
    "19": "int4_per_token_head",
}


def main() -> None:
    llm = LLM(
        model=MODEL,
        enforce_eager=True,
        max_model_len=4096,
        seed=42,
        disable_log_stats=False,
        gpu_memory_utilization=0.85,
        kv_cache_dtype="int4_per_token_head",
        kv_cache_dtype_per_layer=PER_LAYER,
    )

    cc = llm.llm_engine.vllm_config.cache_config
    print("=== cache_config ===")
    print("cache_dtype:", cc.cache_dtype)
    print("kv_cache_dtype_per_layer:", cc.kv_cache_dtype_per_layer)
    print("num_gpu_blocks:", getattr(cc, "num_gpu_blocks", None))
    print("block_size:", getattr(cc, "block_size", None))
    print("mamba_page_size_padded:", getattr(cc, "mamba_page_size_padded", None))

    # kv_cache_config (groups & per-layer specs)
    kcc = llm.llm_engine.model_executor.kv_cache_config
    print("=== kv_cache_config ===")
    print("num_blocks:", kcc.num_blocks)
    for g in kcc.kv_cache_groups:
        print("GROUP layer_names=", g.layer_names)
        print("  spec type:", type(g.kv_cache_spec).__name__)
        print("  page_size_bytes:", getattr(g.kv_cache_spec, "page_size_bytes", None))
        print("  dtype:", getattr(g.kv_cache_spec, "dtype", None))
        print("  kv_quant_mode:", getattr(g.kv_cache_spec, "kv_quant_mode", None))

    try:
        from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity
        num_tokens, max_concurrency = get_kv_cache_capacity(
            llm.llm_engine.vllm_config, kcc
        )
        print("get_kv_cache_capacity: num_tokens=", num_tokens,
              "max_concurrency=", max_concurrency)
    except Exception as e:  # noqa
        print("capacity call failed:", e)

    total_slots = (
        kcc.num_blocks * cc.block_size if kcc.num_blocks and cc.block_size else None
    )
    print("kv_cache_total_slots = num_blocks * block_size =", total_slots)

    out = {
        "num_gpu_blocks": cc.num_gpu_blocks,
        "block_size": cc.block_size,
        "kv_cache_blocks": kcc.num_blocks,
        "kv_cache_block_size": cc.block_size,
        "kv_cache_total_slots": total_slots,
    }
    print("JSON:", json.dumps(out, indent=2))
    print("INSPECT DONE")


if __name__ == "__main__":
    main()
