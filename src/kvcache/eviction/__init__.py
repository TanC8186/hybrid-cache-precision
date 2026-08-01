"""驱逐策略（baseline）：H2O / SnapKV。

- H2O:     Heavy-Hitter Oracle，按注意力分数保留重要 token
- SnapKV:  基于观察窗口的 prompt 压缩

注意：这些方法基于 attention-mask 驱逐（非连续 KV），与"最小改动 vLLM"
原则存在张力——实现时允许在 vendor/vllm 内做必要改动（见设计文档 §5.5）。
"""
