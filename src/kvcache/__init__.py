"""kvcache — KV cache 量化/压缩核心包（MLSys 研究）。

分层：
- cache:       与 vLLM cache engine 对接的核心集成（量化 KV tensor / 自定义注意力后端）
- quantizers:  量化器实现 + KIVI/KVQuant baseline
- eviction:    H2O/SnapKV baseline（attention-mask 驱逐）
- calibration: 校准数据采集与算法
- utils:       正确性校验（roundtrip 位精确、KV 重建 MSE）
"""

__version__ = "0.1.0"
