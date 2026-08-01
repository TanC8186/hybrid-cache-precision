"""量化器：我们的方法 + KIVI / KVQuant baseline。

- base.KVQuantizer: 2/4/8-bit，per-token/per-channel/per-tensor，对称/非对称
- 基准对照：
  - KIVI:     2-bit 分组量化，组大小 16
  - KVQuant:  感知校准的量化，含 per-channel 离群处理
"""
from .base import GRANULARITIES, KVQuantizer, QuantizedKV

__all__ = ["KVQuantizer", "QuantizedKV", "GRANULARITIES"]
