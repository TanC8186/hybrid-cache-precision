"""核心集成层：让量化 KV 在 vLLM 的真实注意力路径中执行。

这是论文的中心 claim，也是最大的技术风险点。
vLLM 默认的 FlashAttention / FlashInfer 后端不接受量化后的 K/V，因此需要：

1. 实现与 vLLM 的 PagedAttention / KV 管理器兼容的量化 KV tensor / 分配器
   （参考设计：`docs/integration.md`）
2. 或在 `vendor/vllm` 内实现自定义 attention backend / 内核

验收门槛：至少服务一个请求，量化 KV 通过 vLLM 的真实注意力路径产出正确输出。
"""

# TODO: 实现 quantized KV tensor / 自定义 backend
# 参考上游：vllm.v1.core.kv_cache_manager, vllm.attention
