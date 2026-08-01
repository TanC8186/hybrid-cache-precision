"""驱逐策略（baseline）：H2O / SnapKV。

- h2o.keep_mask / AttentionScoreAccumulator: Heavy-Hitter Oracle 实现
- 注意：这些方法基于 attention-mask 驱逐（非连续 KV），与"最小改动 vLLM"
  原则存在张力——实现时允许在 vendor/vllm 内做必要改动（见设计文档 §5.5）。
"""
from .h2o import AttentionScoreAccumulator, keep_mask

__all__ = ["keep_mask", "AttentionScoreAccumulator"]
