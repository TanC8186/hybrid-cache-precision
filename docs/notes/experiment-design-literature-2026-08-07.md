# 同类顶会/顶刊论文实验设计调研（2026-08-07）

> 目的：为 BandKV/MLSys 论文的“KV cache 量化/压缩 + serving”实验设计提供对标。
> 检索：网络检索 + arXiv/ACL/NeurIPS/ICML/USENIX 官方页面 + 项目 README；
> 每项标注 [VERIFIED]（已核验）/ [PARTIAL]（部分核验）/ [UNVERIFIED]（未核验）。

## 1. 代表性论文的实验设计一览

| 论文（venue） | 模型 | 质量评测 | 系统/容量评测 | 硬件 | 核验 |
|---|---|---|---|---|---|
| vLLM / PagedAttention（SOSP 2023） | OPT-13B/66B/175B、LLaMA-13B | 不损失精度 | ShareGPT + Alpaca trace；吞吐 vs HF/FasterTransformer/TGI/Orca 2–4×；KV 显存利用率 | A100 40GB | [VERIFIED]（官方博客/论文） |
| KIVI（ICML 2024） | LLaMA-2、Falcon、Mistral | MMLU 类 + long-context 基准，2-bit 误差 <1 分 | 峰值内存 −2.6×、batch 4×、端到端吞吐 2.35–3.47× | 未详列 | [VERIFIED]（作者页） |
| KVQuant（NeurIPS 2024） | LLaMA、LLaMA-2、LLaMA-3、Mistral | Wikitext-2/C4 PPL，3-bit 退化 <0.1 | 单 A100-80GB 跑 LLaMA-7B 1M context，8 卡 10M | A100-80GB | [VERIFIED]（官方摘要） |
| TurboQuant（ICLR 2026） | 论文：大模型（社区复现覆盖 LLaMA-3 等） | PPL 与基线差约 1.1%；vLLM 独立研究用 long-context retrieval + AIME25/LiveCodeBench-v6 | 压缩 3.8–6.4×；vLLM 研究：TPOT/P99 TTFT/吞吐 vs BF16/FP8 | H100（vLLM 研究 4×/2×H100） | [VERIFIED]（arXiv 摘要 + vLLM 官方博客 + GitHub） |
| QPruningKV（EMNLP 2025 Findings） | Llama-3、Mistral | LongBench + RULER + Needle-in-a-Haystack（1K–8K） | KV 预算 128–2048 token/层；量化方法 KIVI/KVQuant/LayerQuant（2/4/8-bit）；baseline PyramidKV/SnapKV/H2O/StreamingLLM | 单 GPU（FlashAttention v2） | [VERIFIED]（官方 GitHub/README） |
| RDKV（arXiv 2605.08317） | 未核验（LLaMA/Mistral/Qwen 类） | LongBench/RULER/InfiniteBench；2.48% 保留恢复 97.81% 全缓存精度，平均 +9.1% vs 最佳 baseline；128K | 解码性能 vs FlashAttention-2 | 未详列 | [PARTIAL]（摘要已核验，模型表未核验） |
| ARKV（CCGRID 2026） | LLaMA3、Qwen3 | long + short context 任务；~97% 基线精度 | KV 内存 −4×，吞吐损失最小 | 未详列 | [VERIFIED]（arXiv 摘要） |
| MiniKV（arXiv 2411.18077） | 未核验 | LongBench 等长上下文任务；86% 压缩恢复 >98.5% 精度 | 专用 CUDA kernel + FlashAttention；延迟/吞吐/内存 | 未详列 | [PARTIAL]（摘要已核验，模型未核验） |
| HqeKV（ACL 2026 Findings） | 未核验 | 同内存约束下质量优于 baseline（未在摘要列具体基准） | 混合量化+驱逐 | 未详列 | [PARTIAL]（页面已核验，细节未核验） |
| H2O（NeurIPS 2023） | OPT、LLaMA、GPT-NeoX | 广泛任务验证（长文生成/对话等） | 20% H2 保留：吞吐 vs DeepSpeed-ZI/HF-Accelerate/FlexGen 最高 29×/29×/3×（OPT-6.7B/30B）；延迟 −1.9× | 未详列 | [VERIFIED]（arXiv 摘要） |
| StreamingLLM（ICLR 2024） | Llama-2、MPT、Falcon、Pythia | PG19 拼接 400K tokens PPL | 流式推理至 4M+ tokens | 未详列 | [VERIFIED]（摘要/README） |
| SnapKV（NeurIPS 2024） | LLaMA-2/3、Mistral、Qwen2 等（16 数据集上） | 16 个长序列数据集（LongBench）性能持平；NIAH 380K 上下文 | 16K 输入：生成速度 3.6×、内存效率 8.2×；A100-80GB 380K | A100-80GB | [PARTIAL]（官方摘要已核验；模型清单来自二手笔记） |
| DistServe（OSDI 2024） | 未核验 | — | TTFT+TPOT 双 SLO 下的 goodput；prefill/decode 分离，最大化约束内速率 | 未详列 | [VERIFIED]（USENIX 页面） |

## 2. 顶会论文的“标准实验设计”模式

### 2.1 质量轴（离线）

1. **困惑度（PPL）**：Wikitext-2（KVQuant、本工作）、C4（KVQuant）、PG19（StreamingLLM）。
2. **长上下文理解**：LongBench（事实上的标准：KIVI、QPruningKV、MiniKV、RDKV、SnapKV、ARKV、HqeKV 等）；
   RULER（QPruningKV、RDKV）；InfiniteBench（RDKV）。
3. **检索/针测试**：Needle-in-a-Haystack / passkey / NIAH（QPruningKV 1K–8K、SnapKV 380K、TurboQuant 社区复现、本工作）。
4. **推理/下游**：AIME25、LiveCodeBench-v6（vLLM TurboQuant 研究）；MMLU 类（KIVI）。
5. **做法要点**：固定上下文长度或预算扫描（如 QPruningKV 128–2048 token/层）、同字节预算对比、多 seed 或确定性 greedy。

### 2.2 容量/内存轴

- 指标：KV 压缩比（2.6×–8×）、峰值内存（含权重）、同上下文下 batch 大小、单卡最大上下文长度
  （KVQuant 1M/10M、SnapKV 380K、StreamingLLM 4M）、每 token KV 字节。
- 做法要点：固定 GPU 显存预算、报告 KV 容量 tokens 与最大并发（本工作 E1/A2 即此口径）。

### 2.3 服务/性能轴

- 指标：端到端吞吐（req/s、tok/s）、TTFT/TPOT（p50/p99）、SLO goodput（DistServe 风格：
  TTFT+TPOT 双约束下的最大可持续速率）、burst 负载下 P99 TTFT、吞吐-容量 Pareto
  （vLLM TurboQuant 研究）。
- 工作负载：ShareGPT（vLLM、本工作）、Alpaca（vLLM）、真实 trace 回放（LMSYS-Chat-1M、
  agentic）、合成 Poisson（本工作 E3）、burst 负载（vLLM TurboQuant 研究）。
- 基线：BF16/FP16、FP8（vLLM 原生，现在几乎是 serving 论文标配）、KIVI/KVQuant/TurboQuant、
  H2O/StreamingLLM/SnapKV/PyramidKV。

### 2.4 可复现性

- 固定硬件/框架/commit；发布代码；报告配置生效证据；多 seed 或确定性子样；失败保留
  （本工作已具备）。vLLM TurboQuant 研究还发布了跨模型/跨硬件的 Pareto 图。

## 3. 与本项目的对照与差距

| 维度 | 顶会主流做法 | 本项目现状 | 差距/建议 |
|---|---|---|---|
| 模型规模 | ≥7B/8B，serving 论文常 30B–200B | 2B 完整矩阵、9B 仅容量 | 补 Qwen3.5-9B 的 E3/质量矩阵（9B 是家族最小 ≥7B 型号） |
| PPL 数据集 | Wikitext-2 + C4 + PG19 | Wikitext-2 | 补 C4/PG19 至少 1 个 |
| 长上下文质量 | LongBench + RULER + InfiniteBench + NIAH | 计划 LongBench；NIAH 有 32-token 截断缺陷 | 重跑 NIAH（≥128 token 或禁 thinking）；补 RULER |
| 推理/下游 | AIME25、LiveCodeBench-v6、MMLU | 无 | 至少补 1 个推理基准（AIME/GSM8K）+ MMLU 类 |
| 服务基线 | BF16 + FP8 + TurboQuant 多 variant | fp16 + int4 + packed；TurboQuant 质量已跑 | serving 矩阵加 FP8 与 TurboQuant k8v4/4bit_nc；3-bit NC 作为负面结果 |
| 服务指标 | goodput（TTFT+TPOT）、P99 TTFT（burst）、TPOT 开销、Pareto | goodput/offered、TTFT 阈值 sweep | 补 burst P99 TTFT、TPOT vs BF16 开销、容量-吞吐 Pareto |
| 上下文长度 | 64K–256K | 容量到 16K、serving 4K | 补 32K/64K 容量与质量探针（5090 单卡 2B 可行） |
| 硬件 | A100/H100 为主 | RTX 5090（消费级 Blackwell） | 定位为“低成本单卡”，并声明与 A100/H100 的差异 |

## 4. 关键外部证据（可写入论文 Related Work / baseline 讨论）

- **vLLM 官方 TurboQuant 研究（2026-05-11，Red Hat AI）** [VERIFIED]：
  LLaMA-3.3-70B、Qwen3-30B-A3B(-Thinking)、MiniMax-M2.7；long-context retrieval ≤64K/256K、
  AIME25、LiveCodeBench-v6；结论：FP8 在吞吐/容量上占优；TurboQuant 4-bit variant 帮助容量，
  **3-bit NC variant 在推理+超长上下文上明显掉精度，且延迟 +10–68%、吞吐 −40–52%**；
  burst 下 P99 TTFT：BF16 ~17s vs TQ <3.5s vs FP8 <1.5s。
  → 我们的 R5 应把 FP8 纳入 serving 对照，TurboQuant 用 k8v4/4bit_nc，3-bit 作为负面结果。
- **KVQuant** [VERIFIED]：3-bit 在 Wikitext-2/C4 PPL 退化 <0.1 → 顶会接受“PPL 为主 + 长上下文
  容量”的证据组合，与我们 E1+质量闭环保局一致。
- **DistServe（OSDI 2024）** [VERIFIED]：TTFT+TPOT 双 SLO 的 goodput 是 serving 论文主流评价
  协议，与我们的 E3 protocol-v2 一致。

## 5. 建议的下一步实验清单（按优先级）

1. NIAH 重跑（max_tokens≥128 或禁用 thinking），并补 RULER 4K/8K；
2. TurboQuant k8v4/4bit_nc + FP8 serving SLO 矩阵（protocol-v3），含 burst P99 TTFT 与
   TPOT 开销、容量-吞吐 Pareto；
3. Qwen3.5-9B 的 E3 边界 + LongBench/RULER 质量闭环；
4. 推理/下游基准（AIME 子集或 GSM8K + MMLU）；
5. 32K/64K 容量与长上下文质量探针；
6. C4/PG19 PPL 作为第二个质量数据集。

## 6. 来源清单（主要）

- vLLM/PagedAttention：https://blog.vllm.com.cn/2023/06/20/vllm.html ；SOSP 2023
- KIVI：https://zirui-ray-liu.github.io/projects/mlsys-numerics/kivi/ ；ICML 2024
- KVQuant：https://papers.nips.cc/paper/2024/hash/028fcbcf85435d39a40c4d61b42c99a4-Abstract-Conference.html
- TurboQuant：arXiv:2504.19874（ICLR 2026）；vLLM 研究：
  https://raw.githubusercontent.com/vllm-project/vllm-project.github.io/main/_posts/2026-05-11-turboquant.md
- QPruningKV：https://github.com/zhzihao/QPruningKV ；ACL：2025.findings-emnlp.429
- RDKV：arXiv:2605.08317；ARKV：arXiv:2603.08727（CCGRID 2026）
- MiniKV：arXiv:2411.18077；HqeKV：2026.findings-acl.201
- H2O：arXiv:2306.14048（NeurIPS 2023）；StreamingLLM：arXiv:2309.17453（ICLR 2024）
- SnapKV：NeurIPS 2024（https://proceedings.neurips.cc/paper_files/paper/2024/hash/28ab418242603e0f7323e54185d19bde-Abstract-Conference.html）
- DistServe：https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin
