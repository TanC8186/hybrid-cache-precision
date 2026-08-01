# vLLM 集成设计（核心改动说明）

> 这是论文的中心 claim 和最大技术风险。本文档记录"量化 KV 如何在 vLLM 真实注意力路径中执行"的设计与进展。

## 问题

vLLM 默认的 FlashAttention / FlashInfer 后端**不接受量化后的 K/V**。
要让量化 KV 真正跑起来（而不是单独做个量化库），必须在 vLLM 内做集成。

## 候选技术路线（按侵入度排序）

1. **自定义 attention backend**（推荐起点）
   - 在 `vendor/vllm` 内注册一个 `KVCacheQuantBackend`，走非 flash 的 attention 路径，接受量化 KV tensor
   - 最小可用：先不追求 kernel 性能，先让"量化 KV 通过真实注意力路径产出正确输出"
2. **量化 KV tensor + 分配器**
   - 在 vLLM 的 KV cache manager 层实现量化张量布局（如 KIVI 的分组布局），分配器按位数分配
   - 工作量更大，但能报真实内存节省
3. **自写 attention CUDA kernel**
   - 最高性能上限，工程门槛最高；放到中后期，作为性能优化而非起步方案

## 实施门槛

- [ ] 至少服务一个请求，量化 KV 通过 vLLM 真实注意力路径产出正确输出
- [ ] 与 FP16 KV 的输出一致（小容差）
- [ ] 冒烟测试纳入 CI 门禁（`tests/`），任何论文表格前必须通过
- [ ] 每个 `vendor/vllm` 改动记录为 `vendor/vllm-patches/<change>.diff`

## 兼容矩阵（冒烟级，不全量）

- 默认 backend（flash_attention）+ 一个替代 backend
- chunked_prefill on/off、prefix caching on/off 各冒烟一次

## 与 baseline 的关系

H2O/SnapKV（驱逐）与 KIVI（非连续分组 KV）在 `vendor/vllm` 需要更多改动。
**放宽"最小改动"原则**：允许为 baseline 做必要改动，但每个改动留 `.diff`。
