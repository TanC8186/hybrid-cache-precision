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

from transformers.cache_utils import DynamicCache

from ..eviction import AttentionScoreAccumulator, keep_mask
from ..quantizers import KVQuantizer


class QuantizedEvictingHybridCache(DynamicCache):
    """只量化/驱逐 full_attention 层的 KV，DeltaNet 层状态不动。

    继承 DynamicCache（transformers 5.x 中它是 hybrid 能力的基础：逐层 cache，
    DeltaNet 层走 conv/recurrent state，full_attention 层走 KV）。覆写 update()
    仅干预 6 层 GQA 的 KV。
    """

    def __init__(
        self,
        *,
        attention_layer_indices: list[int],
        bits: int = 8,
        granularity: str = "per_token",
        evict_budget: int | None = None,   # None = 不驱逐
        evict_window: int = 64,
        quantize_every: bool = True,       # 每次 update 都重量化完整 KV（保真测量）
        layer_bits: dict[int, int] | None = None,  # 逐层位宽覆盖（per-layer 敏感度实验用）
        config=None,
        **kwargs,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self.attention_layer_indices = set(attention_layer_indices)
        self.layer_bits = layer_bits or {}
        self.quantizer = KVQuantizer(bits=bits, granularity=granularity)
        self.quantizers: dict[int, KVQuantizer] = {}
        self.evict_budget = evict_budget
        self.evict_window = evict_window
        self.quantize_every = quantize_every
        self.bytes_per_layer: dict[int, float] = {}
        self.fp16_bytes_per_layer: dict[int, float] = {}
        self.scores: dict[int, AttentionScoreAccumulator] = {
            i: AttentionScoreAccumulator("cpu") for i in attention_layer_indices
        }

    def _quantizer_for(self, layer_idx: int) -> KVQuantizer:
        """返回该层量化器（支持逐层位宽覆盖）。"""
        if layer_idx not in self.quantizers:
            b = self.layer_bits.get(layer_idx, self.quantizer.bits)
            self.quantizers[layer_idx] = KVQuantizer(bits=b, granularity=self.quantizer.granularity)
        return self.quantizers[layer_idx]

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
            # 把分数补齐到当前缓存长度（新 append 的 token 分数为 0，靠 window 保护）
            if acc.scores.shape[0] < keys.shape[-2]:
                pad = torch.zeros(
                    keys.shape[-2] - acc.scores.shape[0], device=acc.scores.device
                )
                acc.scores = torch.cat([acc.scores, pad])
            mask = keep_mask(acc.scores[: keys.shape[-2]], self.evict_budget, self.evict_window)
            # keys/values: [bsz, n_kv_heads, T, head_dim] → 沿 token 维保留
            keys = keys[..., mask, :]
            values = values[..., mask, :]
            acc.after_evict(mask)
            # 持久化驱逐到存储层：让 get_seq_length / 下次 update 看到的是驱逐后长度
            layer = self.layers[layer_idx]
            layer.keys = keys
            layer.values = values

        if self.quantize_every:
            q = self._quantizer_for(layer_idx)
            kq = q.quantize(keys)
            vq = q.quantize(values)
            self.bytes_per_layer[layer_idx] = kq.bytes_used() + vq.bytes_used()
            # 对照：同 KV 的 FP16 字节（= 元素数 × 2）
            self.fp16_bytes_per_layer[layer_idx] = (keys.numel() + values.numel()) * 2
            return q.dequantize(kq), q.dequantize(vq)

        return keys, values

    # ---- 长度查询（供驱逐时手动构建 mask） ----
    def attention_seq_length(self) -> int:
        """当前 full_attention 层 KV 的 token 数（驱逐后 = 实际保留数）。"""
        for i in sorted(self.attention_layer_indices):
            if i < len(self.layers):
                sl = self.layers[i].get_seq_length()
                if sl > 0:
                    return sl
        return 0

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
