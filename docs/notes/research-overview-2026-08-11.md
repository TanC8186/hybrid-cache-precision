# 研究总览：混合线性注意力 Serving 的联合精度预算

生成日期：2026-08-11
范围：`paper/mlsys2026` 当前稿件、已验证实验证据、审稿修复进度与下一阶段计划。

> 本文件所有数字均来自已固化（sha256）的原子 JSON；未执行、未验证或仅
> 单 seed 的内容均明确标注，不作为结论。

---

## 1. 研究方向介绍

### 1.1 问题背景

传统 LLM serving 的内存优化建立在 attention-only Transformer 假设上：
内存大头是随序列增长、可被量化的 attention KV cache。因此现有工作集中
在 KV 位宽（KIVI、KVQuant、MiniKV、TurboQuant 等）和驱逐/混合策略
（H2O、StreamingLLM、SnapKV 等）。

混合线性注意力模型（Jamba、RecurrentGemma、Zamba、Qwen3.5）改变了这一
前提：模型由少量 GQA 注意力层和大量 Gated Delta Network（GDN）层组成。
GDN 层没有 per-token KV，但每条序列携带一个**固定大小的循环状态
（recurrent state）**。该状态与 attention KV 从同一个 GPU 内存池分配，
因此 serving 内存是二维的：

- per-token、随上下文增长、可被 KV 位宽压缩的 attention KV；
- per-sequence、大小固定、但随 state dtype 变化的 recurrent state。

vLLM 已暴露 `mamba_ssm_cache_dtype` 部署开关，但现有预算分析没有把
state dtype 当作与 KV bit-width 联合分配的第二维度。

### 1.2 核心思想

**把 recurrent-state dtype 从“被忽略的部署开关”提升为与 KV bit-width
同等重要的联合预算维度。**

研究对象：Qwen3.5-2B（18 GDN + 6 GQA）与 Qwen3.5-9B（24 GDN + 8 GQA），
vLLM（commit `55f4768`），单张 RTX 5090（32 GB），
`gpu_memory_utilization=0.85`。

### 1.3 核心模型

最大可服务 token 数（单卡、固定内存预算 \(M\)）：

\[
C(L)=\frac{L\cdot M}{A\cdot L+G}
\]

其中 \(A\) 为每 token 的 attention-KV 占用（由 KV dtype 决定），\(G\) 为
每条序列的 state 占用（由 state dtype 决定），\(C(L)/L\) 为可并发的
长度 \(L\) 序列数。

state fp32→bf16 的容量比：

\[
r_{state}(L)=\frac{A L + G_{fp32}}{A L + G_{bf16}}
\]

该比值随 \(L\) 增大而减小：短上下文时 state 主导，长上下文时 KV 主导。
参数全部来自模型配置与 vLLM page layout，无拟合参数，因此可在未探测的
上下文长度上做预测。

### 1.4 与现有工作的区别

- KV 量化/驱逐只压缩 attention 侧，GDN state 不受影响；
- 现有 state 压缩工作（ReplaySSM、FP8 checkpointing 等）不把 state
  dtype 作为 serving 分配器的预算维度；
- Quamba/MambaQuant 是 W8A8/W4A8 权重激活量化，不改变 serving 中
  per-sequence state 的 dtype 预算；
- 本工作不做新算子，而是做“预算记账 + 容量模型 + 质量/serving 证据链”。

---

## 2. 当前实验进度总览

| 模块 | 状态 | 主要证据 |
|---|---|---|
| 容量 2x2（2B/9B × 4K/16K × int4/fp16 × fp32/bf16） | 完成 | `results/verified/2026-08-09/capacity-2x2-analysis.json` |
| 容量探针确定性/block 记录 | 完成 | verified JSON + 每 attempt 配置 sha256 |
| PPL stacking（C4/PG19，3 seeds） | 完成 | `results/quality/ppl-stacking-analysis-20260809.json` |
| GSM8K 2B/9B（9 seeds） | 完成 | `results/quality/gsm8k-*-state9seed-v2-analysis-20260809.json` |
| RULER 5 格 3-seed 复测 | 完成（think 协议） | `results/quality/ruler-statebf16-multiseed-analysis-20260809.json` |
| RULER 全网格单 seed 筛选 | 完成（screen） | Appendix Table `tab:ruler-grid` |
| 逐层敏感性（18 层 × 2 语料 × 3 seeds） | 完成 | `results/quality/state-sensitivity-analysis-20260809-bonf.json` |
| Serving formal + second formal run（60 格 × 2 轮） | 完成 | `results/verified/2026-08-09/statebf16-serving-{formal,repro}-analysis.json` |
| Serving BH-FDR（每格 p/q） | 完成 | `results/quality/serving-bh/serving-bh-analysis-20260810.json` |
| Serving 方向一致性审计（E-1） | 完成（2026-08-11） | `results/quality/serving-direction/` |
| GSM8K paired Cohen's d（E-7） | 完成（2026-08-11） | `results/quality/reasoning/gsm8k-cohens-d-9seed-20260811.json` |
| 引用核验与措辞修复（M0） | 完成（2026-08-11） | `docs/notes/citation-audit-2026-08-11.md` |
| 论文重建/QA | 完成（9 页，32 条 bib） | `paper/mlsys2026/main.pdf` |
| RULER no-think 重跑 | 未做（future work） | — |
| fixed-block/fixed-bytes 机制判别 | 未做 | — |
| TP=2/4 实测 | 未做 | — |
| offloading/prefix-caching 对照 | 未做 | — |
| Mamba2 系外部模型探针 | 未做 | — |

---

## 3. 实验方法

### 3.1 通用纪律

- 每个 attempt 记录解析后的配置（`kv_cache_dtype`、
  `mamba_ssm_cache_dtype`、配置 sha256）；
- 结果以原子 JSON + `.sha256` 固化；
- 新分析先冻结 attempt 合同（sign 规则、子集定义、统计方法），再生成输出；
- 配对比较按 seed 配对；2B/9B、formal/repro 数据不合并；
- 第二次 serving run 与第一次同合同、同 seeds，因此语义是
  **run-stability，不是独立样本**；
- 历史失败轮次和旧 denominator 不混入新的正式结论。

### 3.2 容量协议

- 使用 vLLM maximum-token probe，4K/16K；
- uniform int4 KV（`int4_per_token_head`）或 fp16 KV；
- state fp32/bf16；
- 每格单次确定性运行（固定 engine build + 配置下 block allocator 确定）；
- 记录 block size、block count，用于解释离散取整残差；
- 无拟合参数：预测值直接来自模型配置与 page layout。

### 3.3 质量协议

- PPL：chunk-level write-back harness（chunk=128），C4/PG19，5 条序列、
  3 seeds；明确标注为近似（chunk=1 时 PPL 19.35→36.16，+87%），只作
  supporting evidence；
- GSM8K：200 条 seed 化题目/seed，greedy，无 CoT，9 个 dataset seeds；
  同一 seed 在所有 allocation 中使用同一子集；
- RULER：5 个由先前 KV 量化筛选预先确定的格子，3 dataset seeds；
  完整披露 `--thinking default --max-tokens 256`；no-think 重跑列为
  future work。

### 3.4 Serving 协议

- Protocol v3：piecewise constant Poisson arrivals；
- 120 s warmup；60 s 测量窗（边界确认 300 s）；
- TTFT 阈值 250–3000 ms；TPOT 200 ms；
- 可持续边界要求 goodput/offered ≥ 0.95，失败计入 denominator；
- 3 seeds；Random60（60-token）与 ShareGPT300（300-token traces）；
- 60 格/轮，formal + second formal run，两轮均原子化；
- BH-FDR 逐格 p/q；所有边界格如实报告，不指派效应。

### 3.5 逐层敏感性

- 18 个 GDN 层逐个降 bf16，其余保持 fp32；
- C4/PG19 × 3 seeds = 36 个测试；
- 多重比较使用 Bonferroni（α/36）与 BH-FDR。

---

## 4. 实验结果

### 4.1 容量

int4 KV 下 bf16 state 的实测容量提升：

| 模型 | L | 提升 |
|---|---|---:|
| 2B | 4096 | +37.6% |
| 2B | 16384 | +11.5% |
| 9B | 4096 | +40.6% |
| 9B | 16384 | +14.0% |

int4 列 4 个实测点全部低于模型预测（误差 −0.18% 至 −3.24%），因此该列
作为**保守下界**；fp16 列符号混合，只逐格报告。

复合收益：2B/4K 下 int4-over-fp16 KV 比从 fp32 state 的 2.245× 升至
bf16 state 的 2.675×。部署换算：2B/4K/int4 下 657 → 904 条并发
长度-4096 序列，约 **+247 slots**（KV bytes 不变）。

### 4.2 质量

- PPL：C4 delta −0.0029 [−0.0129, +0.0072]；PG19 +0.0065 [−0.0447,
  +0.0578]，均含零；
- GSM8K 2B：bf16 state −1.00 分 [−1.71, −0.29]，p=0.025，MDE=1.16，
  观测 power 67.5%，paired Cohen's d = −0.92；
- GSM8K 9B：+0.33 分 [−0.07, +0.73]，p=0.141，paired d = +0.54 但 CI
  含零，不解释为增益；
- int4 KV 单独损失 −2.72 分；state-bf16 叠到 int4 上 marginal +1.17 分，
  无额外回归；
- RULER 5 格 3-seed CI 均含零，FWE 区间宽 ±5–33 分，**不做 equivalence
  claim**；单 seed 全网格均值 +0.49（2B）/ −0.71（9B）分，仅作 screen。

### 4.3 Serving

- 唯一重复出现的边界差是 Random60/1000 ms，且第二轮未复现；ShareGPT
  250 ms 两轮方向相反，因此边界表不指派效应；
- 两轮均过载的 Random60 13 格全部同向为正（13/13）；
- 两轮均过载的 ShareGPT 10 格中 7 格反向，方向证据不跨 workload；
- 60 格无任何一格在 BH-FDR q<0.05 存活（最近：formal r40/500 ms
  q=0.066；second run r40/250 ms q=0.052）；
- 结论为 **workload-limited、方向性 paired-goodput 效应 + run-stability，
  不是独立复现，也不是 sustainable-SLO 改进**。

### 4.4 逐层敏感性

36 个测试中 2 个 raw interval 排除零（C4 layers 2/8，p=0.049/p=0.0036），
但均不通过 Bonferroni 或 BH-FDR。**没有逐层精度收益证据**，论文的分配
粒度是整 state 开关。

### 4.5 本轮新增纯分析

**E-1 serving 方向一致性**

- 输出：`results/quality/serving-direction/serving-direction-agreement-20260811.json`
  + contract + sha256；
- Random60 overload（两轮均 overload）13/13 同正；
- ShareGPT overload 10 格 7 反向；
- 全 60 格符号检验不显著（细胞共享 seeds，仅探索性描述）。

**E-7 GSM8K paired Cohen's d**

- 输出：`results/quality/reasoning/gsm8k-cohens-d-9seed-20260811.json`
  + contract + sha256；
- 2B state-bf16 d=−0.92；9B d=+0.54（CI 含零）；
- 正文 §5.2 已回填。

---

## 5. 声明边界

- 仅覆盖 GDN 系模型；Mamba2 系、其他硬件未测；
- 单卡 RTX 5090；TP=2/4 只有一阶推导，未实测；
- 未测 offloading、prefix caching；
- serving 机制（容量 vs 带宽 vs 页对齐）混杂，归因保持开放；
- fp16-state 质量仅 smoke-test；fp8/int8 是 future work；
- LongBench 等长文 pilot 为单 seed，明确排除在 claims 之外；
- PPL harness 是 chunk 级近似，kernel 路径结论以 GSM8K/RULER 为准。

---

## 6. 论文与审稿状态

- 标题：*Joint Precision Budgeting Across Attention KV and Recurrent State
  in Hybrid Linear-Attention Serving*；
- 状态：Major Revision（五席审稿 EIC/R1/R2/R3/DA）；
- 上一轮已落地 R1–R10 主体；本轮完成 M0 引用/措辞修复 + M1 纯分析；
- 当前 PDF：9 页、32 条 bib、无 undefined citation、无 Overfull；
- `figures/verify_figure_data.py`：229 条 ledger 全绿；
- `git diff --check` 干净。

---

## 7. 实验素材与重要文件索引

### 论文与图表

- 稿件：[main.tex](../../paper/mlsys2026/main.tex)、
  [main.bib](../../paper/mlsys2026/main.bib)、
  [main.pdf](../../paper/mlsys2026/main.pdf)
- 图表契约：[figures/figure_contract.md](../../paper/mlsys2026/figures/figure_contract.md)
- 绘图/核验：[figures/make_figures.py](../../paper/mlsys2026/figures/make_figures.py)、
  [figures/verify_figure_data.py](../../paper/mlsys2026/figures/verify_figure_data.py)
- 图 1–8：`paper/mlsys2026/figures/fig{1..8}_*.{pdf,png,svg,tiff}`

### 核心结果 JSON

- `results/verified/2026-08-09/capacity-2x2-analysis.json`
- `results/verified/2026-08-09/statebf16-serving-formal-analysis.json`
- `results/verified/2026-08-09/statebf16-serving-repro-analysis.json`
- `results/verified/2026-08-09/statebf16-formal-20260809.tar.gz`
- `results/quality/gsm8k-state9seed-v2-analysis-20260809.json`
- `results/quality/gsm8k-9b-state9seed-v2-analysis-20260809.json`
- `results/quality/ppl-stacking-analysis-20260809.json`
- `results/quality/ruler-statebf16-multiseed-analysis-20260809.json`
- `results/quality/state-sensitivity-analysis-20260809-bonf.json`
- `results/quality/serving-bh/serving-bh-analysis-20260810.json`
- `results/quality/serving-direction/serving-direction-agreement-20260811.json`
- `results/quality/reasoning/gsm8k-cohens-d-9seed-20260811.json`

### 脚本

- 容量/serving 分析：`scripts/bench/analyze_statebf16_serving.py`、
  `scripts/bench/analyze_capacity_2x2.py`、`scripts/bench/run_serving_formal.sh`
- 质量分析：`scripts/eval/analyze_gsm8k_state3seed_v2.py`、
  `scripts/eval/analyze_ruler_statebf16_multi_seed.py`、
  `scripts/eval/analyze_r4_quality.py`
- 本轮纯分析：`scripts/quality/analyze_serving_direction.py`、
  `scripts/quality/analyze_gsm8k_cohens_d.py`

### 过程文档

- 审稿报告：`docs/notes/mlsys-review-ars-2026-08-10-paper-draft.md`
- 引用审计：[citation-audit-2026-08-11.md](citation-audit-2026-08-11.md)
- 修复执行：[ars-repair-execution-2026-08-11.md](ars-repair-execution-2026-08-11.md)
- 下一阶段计划：`docs/notes/next-stage-experiment-plan-2026-08-09.md`
- 结果摘要：`docs/notes/results-digest-2026-08-09.md`
- 复现语义：`docs/notes/repro-final-2026-08-09.md`、
  `docs/notes/seed-semantics-audit-2026-08-09.md`
- 数据清单：`docs/notes/data-inventory-2026-08-04.md`

---

## 8. 下一阶段实验

| 实验 | 目标 | 优先级 |
|---|---|---|
| RULER no-think 5 格重跑 | 消除 think 截断对 FWE null 的方向不确定性 | P0 |
| fixed-block-count / fixed-bytes serving 对照 | 分离容量、带宽、页对齐机制 | P1 |
| TP=2/4 容量/分片实测 | 把一阶推导变成测量 | P1 |
| offloading / prefix-caching 对照 | 验证替代杠杆与 state-bf16 的正交性 | P2 |
| Mamba2 系外部模型探针 | 扩展 GDN-only 的一般性边界 | P2 |

所有后续实验沿用同一 guardrail：attempt 合同 → MVEx 断连验证 → 短切片
可恢复运行 → 原子 JSON + sha256 → 统计谬误自查；历史失败轮次不并入新的
正式 denominator。

---

## 9. 维护规则（防止伪影）

1. 正文每个数字必须能追溯到原子 JSON；新增数字先落盘再回填；
2. 引用元数据以 arXiv/ACL/出版社官方页面为准，修复后记录来源；
3. “第二次运行”“复现”“独立样本”等词按操作定义使用，不混用；
4. 不把 screen、pilot、单 seed 数据升级为正式结论；
5. 不删除/改写历史失败轮次；负数结果、denominator、sha256 均保留。

---

## 10. 2026-08-11 formal capacity closure update

本节覆盖并更新本文前面仍标为“未完成”的状态快照；历史计划和失败 attempt
保持不变。

| 实验 | 当前状态 | 可用结论 |
|---|---|---|
| RULER no-think 5 格 | 完成，30/30，ANALYZED | fp16 与 bf16-state 在预注册格和 3 个 dataset seed 上未观察到退化；不作 equivalence claim |
| Capacity MVEx | 完成，2/2 | measured/predicted ratio 与残差通过严格 analyzer |
| Capacity pilot | 完成，18/18 | 8/8 core pairs 严格分离；prediction residual median absolute 2.6239% |
| Capacity formal phase diagram | 完成，112/112，ANALYZED | 52/52 core pairs 为 bf16-state > fp32-state；Gate 4 复现门仍待完成 |

Formal 复核详情见
`docs/notes/capacity-phase-formal-review-2026-08-11.md`。formal 分析文件及其
sidecar 已从服务器回传并独立重算；在新 attempt 的 reproducibility comparison
通过前，不能将其状态升级为 `VERIFIED`，也不能把容量预测写成 lower bound。

本轮结果把 capacity phase diagram 从 P0 缺口推进到“完整矩阵已分析”；仍最急的
后续系统缺口是 joint precision selector/controller、fixed-block/fixed-bytes/
fixed-concurrency 机制隔离，以及 full-precision/KV-only/state-only/joint 的
完整 serving load/SLO formal。它们尚未在本轮自动启动。
