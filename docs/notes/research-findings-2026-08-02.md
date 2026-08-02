# 研究结果汇总（2026-08-02，模型侧已确认）

> 方法定位：B（灵敏度引导的逐层异构预算分配）。集成路线：Route 1（lazy-dequantization）。

## 确认的发现（Qwen3.5-2B，18 DeltaNet + 6 GQA）

### 1. 混合模型 KV 位宽容差（Wikitext-2，2048/4096 均稳健）
| bits | PPL (2048) | vs FP16 | 压缩 |
|---|---|---|---|
| 8 | 13.63 | 无损 | 2.0x |
| 4 | 13.86 | +1.7% 近无损 | 3.9x |
| 3 | 15.87 | +16% | 5.2x |
| 2 | 21.07 | +55% | 7.5x |

### 2. 字节预算排序：驱逐 > sub-4bit
等字节下"高精度+驱逐"始终优于"低精度+全保留"（如 4bit keep1024 14.10 vs 2bit 全保留 21.07 @ ~3.3MB）。4096 下更强。

### 3. 逐层敏感度（方法基础）
layer23 最敏感（28.7%），layer3 超不敏感（-5.9%，可白送 2-bit），中四层 ~5-8%。

### 4. 灵敏度引导分配 = 方法（results/ablations/hetero_budget.csv）
- sens_guided {3:2, 中:3, 23:4}：PPL 14.63 @ 4.87MB，击败均匀 3-bit (15.87 @ 4.85MB)
- only_layer3_2bit：PPL 13.39 @ 5.92MB，完胜均匀 4-bit (13.86 @ 6.44MB)

### 5. Serving 容量（results/ablations/serving_metrics.csv，Phase 0）
固定 4GB KV 预算：4-bit = 3.94x 上下文 (1.27M tokens) @ +1.7% PPL。

## 已解决的关键问题
- 撞题：泛化版"联合驱逐×量化"已被 6+ 篇占据 → 重构为混合架构专属
- vLLM 已内置 KV 量化基础设施（int4_per_token_head、TurboQuant 3/4-bit、逐层 skip_layers）
- 网络：huggingface.co/github.com 被墙 → ModelScope/hf-mirror/ghfast 镜像
- 兼容性：vLLM 要求 torch 2.13 = WSL 环境

## 论文故事（草案）
**混合线性注意力模型（Gated DeltaNet + 少数 GQA 层）的 KV cache 逐层异构预算分配。**
1. 洞察：混合模型 4-bit 近无损，sub-4bit 区驱逐优于降 bit
2. 方法：按逐层敏感度分配位宽（敏感层保 4-bit、不敏感层 2-bit）
3. 收益：容量 3.94x @ +1.7%（4-bit），sens_guided 5.2x @ +7%
4. Serving：vLLM lazy-dequant 集成（Phase 1-3 进行中）

## 下一步
- Phase 1：vLLM MVP（int4_per_token_head + lazy-dequant）跑通一个请求
- Phase 2：2/3-bit + 逐层位宽
- Phase 3：量化方法注册 + serving benchmark（5090 headline）
