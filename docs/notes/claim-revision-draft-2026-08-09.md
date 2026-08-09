# Claim 修订草稿（ARS 2026-08-09 执行后，R5/R11/R12）

> 待 S-formal 结果回填后再并入 research-summary / paper-mainline；本文件先固化
> 已完成实验带来的措辞变化，避免旧表述继续污染草稿。

## 1. Claim 3（bf16 state 容量收益与质量）修订

旧表述：“PPL/RULER 基本持平，GSM8K 有披露回退（2B −2.7pt / 9B −0.5pt）”。

新表述（基于 seed 化 v2 协议）：

- 容量：uniform int4 KV 下 2B/9B、4K/16K 收益 +11~41%，模型误差 −3.24~−0.18%
  （int4 列 4/4 全负，同号概率 0.0625 → 保守下界表述）；
- PPL：fp16-KV 与 int4-KV 下 bf16 state 的 3-seed 配对 CI 均含 0；
- RULER：3 dataset-seed 点估计，5 个原非零格 Δ CI 均含 0（2B fwe L4096
  −3.89 [−32.97,+25.19]；9B niah_multiquery L8192 −4.17 [−8.91,+0.58] 等）；
  FWE 跨 seed 波动极大 → 只能“点估计 + 宽 CI”，不得称持平或掉分；
- GSM8K（真实随机协议，9 seeds）：2B state −1.00pt [−1.71,−0.29]（p=0.025）、
  int4 −2.72pt [−4.20,−1.24]（p=0.007）→ 回退显著；2B stacking 边际
  +1.17pt [−0.33,+2.66]（无叠加代价）；9B state +0.33pt [−0.07,+0.73]
  （无回退）。旧 head-200 协议的 −2.67pt/−0.5pt 是伪重复产物，退役。

## 2. Claim 5（serving SLO 收益）修订

- S-formal（int4 KV × {fp32,bf16} state，Random60+ShareGPT300，3 seeds）完成前：
  “ANALYZED / formal pending”，不进 Abstract；
- 完成后：按 workload × TTFT threshold 的 3-seed 边界与 paired goodput Δ 表述；
- DA 的“带宽 vs 容量”替代解释：用每 server 的 num_gpu_blocks/concurrency 与容量
  探针对照讨论，不把 TTFT 改善单独归因于容量。

## 3. Claim 1 措辞收窄

“现有工作只优化其中一维” → “现有 serving 系统未把 state 精度纳入与 KV 量化
联合的预算分配”；保留 ReplaySSM / PR#43518 边界，不宣称首次 state 压缩。

## 4. R5 容量模型定位（按 KV 列分开）

- int4 KV（headline）：4/4 全负 → “保守下界”，机制 = 离散 block 分配
  （fp32 block 2064/3287 vs bf16 block 1072/6330；9B 2048/385 vs 1040/758）；
- fp16 KV：符号混合（+2.86/+3.53/−2.83），不做“全负/保守”表述，按 signed error
  逐格报告；原因仍是 block 粒度，但方向随 KV 布局翻转；
- 模型参数 A_q/A_f/G 独立于被预测量（非 tautology），推导链见 S1。

## 5. R12 “第二维度”叙事支撑数据

- r_kv（int4/fp16）：fp32 state 2.245（2B@4K）→ bf16 state 2.675；9B 2.189→2.740；
- r_state：fp16 KV 1.07~1.15、int4 KV 1.11~1.41（L 越大越小）→ 短上下文
  state 主导、长上下文 KV 主导；
- Q-stacking PPL：int4+bf16 相对 int4+fp32 Δ CI 含 0 → 复合维度无质量叠加代价；
- 叙事模板：bf16 state 与 KV 量化是正交可复合预算维度，2×2 表 + stacking 证据
  论证复合增益，不做幅度并列。
