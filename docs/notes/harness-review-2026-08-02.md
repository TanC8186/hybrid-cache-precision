# Harness 对抗式代码审查记录（2026-08-02）

> 多 Agent 对抗式审查（PPL 正确性 / 驱逐数学 / 对抗性） + 人工复核。
> 目的：确认实验结果可信，或修复后重跑。

## 结论

### ✅ 纯量化路径可信（bit-tolerance 曲线）
- update() 在注意力前返回反量化的完整 KV，机制正确
- 2-bit 显著劣化证明量化真实生效（非 no-op）
- 字节记账公平（含 per-token scale；4-bit 3.9x 与 head_dim=256/n_kv_heads=2 解析一致）
- 8-bit≈FP16、4-bit 近无损、2-bit 退化 = 真实模式

### 🔴 驱逐路径曾有 critical bug（已修复 + 重跑）
1. **window(64) < chunk(128)**：新 chunk 前 64 个 token（分数 0）在同一次 update 被立即驱逐，返回 key 布局与因果 mask 错位 → 静默产出错误驱逐 PPL
   - 修复：evict_window=256 ≥ chunk_size，保护整个当前 chunk
2. **PPL 少计边界 token**：每 chunk 首 token 从不被预测，绝对 PPL 系统性抬高（各配置同受影响，相对结论不变）
   - 修复：chunk 多喂 1 个 token，全部位置计分
3. **fp16 跨 chunk loss 累加舍入**（chunked vs plain 差 0.024）
   - 修复：logits.float() 后计算 loss

### ⚠️ 需在论文披露的口径问题（非 bug）
- **驱逐分数滞后一拍**：驱逐用上一 chunk 的注意力分数（H2O 在线近似的系统偏差）
- **字节口径 = 打包后理论字节**（numel×bits/8），非 int8 真实内存（真实 8x）；vLLM kernel 阶段才真实打包
- **驱逐分数基于压缩模型**：importance 在量化+驱逐后的模型上测得（H2O 标准做法，需写明）
- **语料拼接**：Wikitext-2 拼接成 token 流切 2048，跨文档边界轻微污染（为触发驱逐预算的必要手段）

## 审查发现的时间线
- 审查确认了人工发现的 window bug（critical）
- 补充发现：边界 token 少计（minor）、分数滞后一拍（major 语义）、字节口径（minor）
- 所有 critical/major 已修复，驱逐结果已用修正版重跑
