"""量化/驱逐 KV cache：HybridCache 子类，只对 full_attention 层的 KV 应用量化+驱逐。

混合架构前提验证的核心组件：
- Qwen3.5-2B = 18 层 Gated DeltaNet（无增长 KV，走 recurrent state）+ 6 层 Gated Attention（有 KV）
- 本 cache 只干预 6 层 GQA 的 KV；DeltaNet 层的 conv/recurrent state 完全不动（走 super().update）

纯量化模式（first premise test）：
- 覆写 update()：对 full_attention 层的完整累计 KV 做 quantize→dequantize，返回反量化 KV 给注意力
- 记录量化后的真实字节数（bytes_used，含 scale 开销）

驱逐模式（phase 2）：
- 通过 record_scores() 喂入注意力权重，update() 时按 H2O keep_mask 驱逐再量化
"""
from __future__ import annotations

import torch

from transformers.cache_utils import (
    DYNAMIC_LAYER_TYPE_MAPPING,
    HybridCache,
    get_layer_types_and_kwargs,
)

from ..eviction import AttentionScoreAccumulator, keep_mask
from ..quantizers import KVQuantizer


class QuantizedEvictingHybridCache(HybridCache):
    """只量化/驱逐 full_attention 层的 KV，DeltaNet 层状态不动。"""

    def __init__(
        self,
        *,
        attention_layer_indices: list[int],
        bits: int = 8,
        granularity: str = "per_token",
        evict_budget: int | None = None,   # None = 不驱逐
        evict_window: int = 64,
        quantize_every: bool = True,       # 每次 update 都重量化完整 KV（保真测量）
        config=None,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
    ) -> None:
        if config is not None:
            # 从 config 构建逐层 cache（DeltaNet 用 recurrent layer，full_attention 用 KV layer）
            decoder_config = config.get_text_config(decoder=True)
            layer_types, layer_kwargs = get_layer_types_and_kwargs(decoder_config)
            layers = [DYNAMIC_LAYER_TYPE_MAPPING[lt](**layer_kwargs) for lt in layer_types]
            super().__init__(layers=layers, offloading=offloading,
                             offload_only_non_sliding=offload_only_non_sliding)
        else:
            super().__init__(layer_class_to_replicate=None, offloading=offloading,
                             offload_only_non_sliding=offload_only_non_sliding)
        self.attention_layer_indices = set(attention_layer_indices)
        self.quantizer = KVQuantizer(bits=bits, granularity=granularity)
        self.evict_budget = evict_budget
        self.evict_window = evict_window
        self.quantize_every = quantize_every
        self.bytes_per_layer: dict[int, float] = {}
        self.fp16_bytes_per_layer: dict[int, float] = {}
        self.scores: dict[int, AttentionScoreAccumulator] = {
            i: AttentionScoreAccumulator("cpu") for i in attention_layer_indices
        }

    # ---- 逐层分数（供驱逐） ----
    def record_scores(self, layer_idx: int, attn_weights: torch.Tensor) -> None:
        """在注意力计算后调用：把跨头平均的注意力权重喂给 H2O 累计器。"""
        if layer_idx not in self.attention_layer_indices:
            return
        acc = self.scores[layer_idx]
        acc.scores = acc.scores.to(attn_weights.device)
        acc.accumulate(attn_weights)

    # ---- 覆写 update ----
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx not in self.attention_layer_indices:
            return super().update(key_states, value_states, layer_idx, *args, **kwargs)

        keys, values = super().update(key_states, value_states, layer_idx, *args, **kwargs)

        # 驱逐（基于此前累计的注意力分数），发生在量化前
        if self.evict_budget is not None and keys.shape[-2] > self.evict_budget:
            acc = self.scores[layer_idx]
            mask = keep_mask(acc.scores[: keys.shape[-2]], self.evict_budget, self.evict_window)
            # keys/values: [bsz, n_kv_heads, T, head_dim] → 沿 token 维保留
            keys = keys[..., mask, :]
            values = values[..., mask, :]
            acc.after_evict(mask)

        if self.quantize_every:
            kq = self.quantizer.quantize(keys)
            vq = self.quantizer.quantize(values)
            self.bytes_per_layer[layer_idx] = kq.bytes_used() + vq.bytes_used()
            # 对照：同 KV 的 FP16 字节（= 元素数 × 2）
            self.fp16_bytes_per_layer[layer_idx] = (keys.numel() + values.numel()) * 2
            return self.quantizer.dequantize(kq), self.quantizer.dequantize(vq)

        return keys, values

    # ---- 字节记账 ----
    @property
    def total_bytes(self) -> float:
        """6 层 GQA KV 的量化字节总和。"""
        return sum(self.bytes_per_layer.values())

    @property
    def total_fp16_bytes(self) -> float:
        """对照：同 KV 的 FP16 字节总和（用于压缩率/字节预算归一化）。"""
        return sum(self.fp16_bytes_per_layer.values())

    @property
    def compression_ratio(self) -> float:
        """FP16 字节 / 量化字节（含 scale 开销的端到端口径）。"""
        if self.total_bytes <= 0:
            return 1.0
        return self.total_fp16_bytes / self.total_bytes
