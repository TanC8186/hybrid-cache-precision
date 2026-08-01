# 文献尽职调查记录（2026-08-02）

> 撞题检查 + 方法学习，4 视角并行检索（混合架构KV量化 / 联合驱逐×量化 / 顶会方法协议 / 备选方向）。
> 触发原因：模型选定 Qwen3.5-2B（混合架构）后，需确认 2025-2026 最新文献是否已覆盖我们的联合驱逐×量化方向。

## 判定：主攻方向泛化版撞题，需 adjust

### 泛化版直接撞题（5-6 篇）
| 论文 | 出处 | 内容 | 与我们关系 |
|---|---|---|---|
| QPruningKV | EMNLP 2025 Findings, arXiv:2412.12706 | "存更多 token 于更低精度"优于纯驱逐与纯量化两个极端；budget-equivalence 协议 1×16 vs 2×8 vs 4×4；结论"砍 token 比降精度更伤" | **字面发表我们的假设**，最致命 |
| RDKV | arXiv:2605.08317, 2026-05 | 驱逐=0-bit 量化，率失真反向注水分配 bit | 占据"联合最优"表述 |
| ARKV | arXiv:2603.08727, 2026-03 | FP16/低bit/驱逐三态字节预算分配（Qwen3/LLaMA3） | 机制同构 |
| HqeKV / MiniKV / ThinKV | ACL'26 / ACL'25 / ICLR'26 | 量×驱逐联合（分层/2-bit/推理场景） | 联合范式已被铺开 |
| KV Pareto / NeurIPS'25 Efficiency / MiKV / KVC-Q / MoE-nD | 2025-2026 | 字节级 Pareto、全量低比特>驱逐、低精度保留>驱逐、连续保真度分配、per-layer 路由 | 系统/理论层均已占 |

### 唯一幸存窄缝：混合架构专属
- 上述联合论文**全部只在标准 Transformer（LLaMA3/Qwen3/Mistral）验证**
- Qwen3.5-2B 混合模型三特性未被任何联合预算工作利用：
  1. **仅 6/24 层有 KV cache**（18 层 Gated DeltaNet 无增长 KV）
  2. **线性层吸收量化噪声**（llama.cpp Issue #21385 实测 Qwen3.5 上 q4_0 KV 完全无损）
  3. **6 个全注意力层层间异构**
- TurboQuant/RotorQuant（Qwen3.5 KV 量化）：只量化不驱逐；RecurrentBitNet：量化权重非 KV；Hypic：serving 级 PIC 缓存非 KV 量化

### 危险实证信号
- llama.cpp #21385：q4 在 Qwen3.5 上无损 → "剩多少 bit"在 q4 以上失去意义，**故事被迫下移到 sub-4bit 区**（恰是 TurboQuant 承认的 2-bit value 瓶颈区 = 机会）
- TurboQuant：混合模型 decode 需把历史全 dequant 到 fp32 → **字节预算模型必须计入 serve 层带宽成本**

### 备选方向（长上下文误差累积+在线刷新）：部分撞题 + 两条逆风
- KVarN（2026-06，华为）：误差累积机制分析 + pseudo-decode 协议（已占现象与协议）
- Runtime-Certified Bounded-Error（2026-05）：在线误差界+fallback（=一种刷新）
- PM-KVQ（ICLR'26）、RefreshKV（ACL'25）、Elastic-Cache（ICLR'26）、FreqDepthKV（2026-07）
- **逆风 1**：Stage-Replay Divergence（2026-07）证明重算不能忠实还原 live decoder 状态，动摇"刷新可恢复精度"前提
- **逆风 2**：2026 趋势（OSCAR ~2.28bit 128K 近无损）削弱"刷新必要性"
- 尚存空角：混合模型上仅 6 层 attention KV、刷新成本极低 → "混合模型专属的在线 up-refresh"仍空

## 调整后的研究贡献定位

**混合线性注意力架构（Gated DeltaNet + 仅 6 层 GQA）下，量化×驱逐联合字节预算的第一个系统研究 + 无人验证的排序结论：**
> 固定字节预算下，sub-4bit 区驱逐应优先于进一步降 bit（2-bit value 是瓶颈；q4 以上近无损）

**差异化三要素**：混合架构三特性 + sub-4bit 排序 + serve 语义的字节公平口径

## 方法协议要学（照搬）
- QPruningKV budget-equivalence 协议（格点模板）
- KVarN pseudo-decode 评测协议（误差累积真实测量）
- KIVI grouped G=128 + FP16 residual R=32-128（压缩率含 residual 端到端口径 ≈5.05x）
- RULER 13 子集 4K-128K 扫描 + LongBench 16 数据集
- 驱逐 baseline 统一 "Protection Is All You Need" 结构性保护设置（否则驱逐 F1≈0.06）
- kv_bench 两遍法（mask -inf 模拟驱逐，先于真机验证）
- 字节级公平口径（memory bytes vs accuracy Pareto），非 token-count

## 必引 baseline
QPruningKV、RDKV、ARKV、MiniKV、KV Pareto、NeurIPS'25 Efficiency-for-Reasoning、MiKV、TurboQuant、KVarN、Runtime-Certified、H2O、SnapKV、KIVI、KVQuant

## 下一步行动
1. 混合架构前置验证：Qwen3.5-2B 上 6 层 GQA 的 KV 量化容忍度（q4 是否无损？2-bit value 是否瓶颈？误差曲线测到 2-3bit）
2. 字节预算排序实验：等字节下驱逐 vs sub-4bit 的排序
3. 逐层异构：6 个注意力层哪些对 KV 量化最敏感
4. 精读 RDKV 与 ARKV 全文，确认其是否触及混合模型
