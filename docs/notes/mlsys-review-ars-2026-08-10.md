# ARS 最大强度审稿报告（2026-08-10 重审版）— 精度预算分配方向

> 模式：`academic-paper-reviewer` **full**（EIC + R1 Methodology + R2 Domain +
> R3 Perspective + DA，5/5 全齐）
> 审稿对象：`docs/paper/research-summary-2026-08-10.md`（08-09 版补跑后的重审快照）
> + 全证据库（08-08/08-09/08-10 容量、质量、serving、复现）
> 契约：`reviewer/reviewer_full/v1`（panel_size=5，D1–D5，F0–F3）

## 0. 执行边界与诚实声明

1. 单模型面板（`ARS_CROSS_MODEL` 未设置），无独立跨模型盲检；
2. Sprint Contract 的两调用隔离无法在本内联会话物理执行，契约仅作为评分判据与
   决策算术使用，不宣称完成盲审预注册；
3. DA 不打分；F0–F3 机械判定基于 4 位打分审稿人；
4. 只读审稿：除本报告与 08-10 研究汇总（作者侧准备的重审输入）外，未修改任何
   实验/论文文件；
5. 加权分数沿用 08-09 轮的会议校准口径；最终判定以契约 F0–F3 为准
   （量规文件 65–79=Minor 与契约 F2 冲突时，按契约优先级处理并在附录 C 说明）。

---

# Phase 0 — 领域分析与审稿人配置

## 1. 六维分析

| 维度 | 结果 |
|---|---|
| 主学科 | 计算机系统 / ML Systems（LLM serving 内存系统） |
| 次学科 | KV cache 量化与压缩、循环 state 精度、混合线性注意力、容量建模、serving 评测 |
| 研究范式 | 定量实证系统研究（受控对比实验） |
| 方法类型 | 系统测量 + 解析建模 + 质量/服务闭环评测 |
| 目标层级 | MLSys 2026/2027 Research Track（Q1 顶会） |
| 论文成熟度 | 证据链约 3/4 完成；论文未按新证据重写（最大风险点） |

## 2. Reviewer Configuration Cards

### Card #1 — EIC（MLSys Area Chair）
- 身份：MLSys 领域主席，LLM serving 系统软件方向；
- 关注：二维预算定位是否仍成立、serving 主张是否越界、双贡献叙事、文本-证据一致；
- 盲区：统计细节（由 R1 补）。

### Card #2 — R1（Methodology）
- 身份：serving 评测方法论 + 统计报告专家；
- 关注：seed 协议、power/MDE、多重比较、复现容差、pilot/formal 边界；
- 盲区：SSM/量化领域文献（由 R2 补）。

### Card #3 — R2（Domain）
- 身份：线性注意力/SSM cache 量化专家；
- 关注：容量模型正确性、文献边界、kernel 语义、一般性；
- 盲区：部署经济性（由 R3 补）。

### Card #4 — R3（Perspective）
- 身份：内存系统/数据中心推理基础设施研究者；
- 关注：成本/请求、替代杠杆、TP/多卡、跨架构、生产可部署性；
- 盲区：统计功效（由 R1 补）。

### Card #5 — DA（Devil's Advocate）
- 身份：对抗式研究者；只挑战不打分；
- 关注：贡献线（flag-flip 是否够）、cherry-picking、serving 过度解读、替代解释。

---

# Phase 1 — 独立审稿报告（摘要 + 分数）

## EIC Report Summary

- **Recommendation**: Major Revision | Confidence: 4 | 加权 71.2
- 核心判断：方向与框架原创性仍成立（二维预算 + 容量模型 + 诚实负面结果）；
  补跑后证据链显著完整（GSM8K seed 修复、敏感度校正、RULER 补 seed、S-formal+
  复现），但 **serving 主张只能以 PARTIALLY_REPRODUCIBLE 的窄口径表述**，
  论文正文尚未重写是本轮最大风险。
- 维度：Originality 76 / Rigor 73 / Evidence 62 (warn) / Coherence 76 / Writing 72。
- 关键发现：
  - S1：容量收益是强证据（int4 列 4 点保守下界、复合 r_kv 2.245→2.675）；
  - S2：GSM8K 9-seed 与敏感度校正让统计口径可信；
  - S3：复现 72/72 审计 + PARTIALLY_REPRODUCIBLE 判定本身是诚实性的加分项；
  - W1（CRITICAL 候选）：claim #5 若在正文/Abstract 以 headline 出现即越界；
  - W2：单架构（Qwen3.5 2B/9B）+ 单卡，一般性声明必须收窄；
  - W3：harness PPL 的 chunk 语义边界必须写进正文，否则 PPL 证据会被攻击；
  - W4：A2 与 state-bits 双贡献的叙事缝合未完成。

## R1 (Methodology) Report Summary

- **Recommendation**: Major Revision | Confidence: 4 | 加权 70.6
- 核心判断：统计体系是领域标杆级（seed 语义修复、MDE/power 预注册、多重比较校正、
  复现容差预注册），但 **GSM8K 2B state 效应显著却 power 仅 67.5%**、serving
  复现部分失败，论文必须把“显著”与“功效不足”同时披露。
- 维度：Originality 74 / Rigor 68 / Evidence 58 (warn) / Coherence 80 / Writing 82。
- 关键发现：
  - F0 算术语义全部可复算（配对 t-CI、MDE、Bonferroni/BH-FDR）；
  - F1 配置生效证据逐 cell 校验（含 int4+bf16 双重校验）；
  - W1：serving 边界以 5 req/s 网格 + 三 seed 判定，±1 步边界差异不能当结论，
    复现中 3 处边界不一致即证明其敏感性；
  - W2：ShareGPT r45 复现差异 >100%，说明过载区 mean goodput 不可作为稳定数字；
  - W3：RULER FWE 的 20 样本 × 3 dataset seed 仍宽到 ±10~30pt，只能“点估计 +
    无差异检测能力”；
  - W4：harness chunk=1 的 +87% PPL 偏移必须作为方法学限制写入。

## R2 (Domain) Report Summary

- **Recommendation**: Major Revision | Confidence: 4 | 加权 72.7
- 核心判断：容量模型与 vLLM 代码链逐行吻合，保守下界定位可辩护；文献边界正确
  （ReplaySSM/PR#43518），但文献扩展（R10）、跨架构、以及“容量→SLO”机制论证
  仍是硬缺口。
- 维度：Originality 76 / Rigor 72 / Evidence 63 (warn) / Coherence 82 / Writing 76。
- 关键发现：
  - S1：2×2 容量表 + block 粒度证据（2064/1072、544/288）支持保守下界；
  - S2：RULER 多 seed 补跑消除了单 seed 抽奖式非零差异；
  - W1：A_q/A_f 推导链未入论文（S1 待办）；
  - W2：仅 Qwen3.5，非 GDN 架构无证据；
  - W3：serving 的“容量→SLO”链条在 ShareGPT 上断裂（r45 方向翻转），
    机制叙事必须限定在 Random60 过载区；
  - W4：FWE think 截断伪影仍未完全移出主表（应统一 no-think 口径）。

## R3 (Perspective) Report Summary

- **Recommendation**: Major Revision | Confidence: 4 | 加权 71.2
- 核心判断：state 内存与短上下文/高并发场景的生产相关性真实，诚实披露是优点；
  但成本/请求、替代杠杆、TP/多卡、跨架构的缺失使“生产意义”仍停留在口号层面。
- 维度：Originality 75 / Rigor 70 / Evidence 61 (warn) / Coherence 80 / Writing 76。
- 关键发现：
  - W1：缺少成本/请求模型（capacity × concurrency 的换算）；
  - W2：无 TP 分片/多卡影响讨论；
  - W3：替代路径（H2O/SnapKV/PyramidKV、offloading、prefix caching）只有定性；
  - W4：fp8/int8 state 与跨架构未测 → “精度谱系”必须是显式边界声明；
  - W5：train-inference mismatch 假说未讨论（R16）。

## DA (Devil's Advocate) Report Summary

### Strongest Counter-Argument

即使补跑全部完成，最有力的反驳仍然是：核心贡献可被描述为“翻转 vLLM 已有
`--mamba-ssm-cache-dtype bf16` 开关 + 记账型容量模型 + 一组诚实但大多为
null/负面的质量结果”。serving 转换只有 Random60 过载区部分可复现，ShareGPT 甚至
方向翻转；RULER 无差异、harness PPL 不能证明 per-token 等价、GSM8K 有真实回退。
若论文把 serving 或“质量持平”写成 headline，这条反驳就会成立。

### Issue List

#### CRITICAL

| # | 维度 | 问题 | 位置 | Field-Norm Boundary | Evidence-Crossing Rationale |
|---|---|---|---|---|---|
| C1 | 数据-结论匹配 | 若正文/Abstract 以 serving 收益为 headline，则与 PARTIALLY_REPRODUCIBLE 直接矛盾 | §4 claim 5、Abstract | 顶会系统论文不得用未复现数字作主卖点（venue 公开评估惯例） | 复现已执行：R60 边界 1 处、SG 250ms 方向、r45 >100% 差异 |
| C2 | 数据-结论匹配 | “容量模型误差 <3.3%”若不做 KV 列限定，fp16 列 +2.86%/+3.53% 与之矛盾 | §3.1、claim 2 | 保守下界定位要求按被预测量分组报告（编辑裁决 08-09 已定） | int4 4/4 全负、fp16 符号混合，已入分析 JSON |

#### MAJOR

| # | 维度 | 问题 | 位置 |
|---|---|---|---|
| M1 | 功效披露 | GSM8K 2B state 效应 p=0.025 但 power=67.5%、MDE 1.16pt > 效应 1.00pt | §3.3 |
| M2 | 证据边界 | “PPL/RULER 持平”不得写成“无损失”；RULER 只能“无检测能力” | §3.2/§3.4 |
| M3 | 机制归因 | serving 增益未隔离带宽 vs 容量；page 对齐是替代解释 | §3.7 |
| M4 | 一般性 | 仅 Qwen3.5 2B/9B；无 Mamba2 或显式 GDN-scope | §5 |
| M5 | 证据层级 | harness PPL 必须标注 chunk 级近似，否则会被当作 kernel 语义 | §3.6 |
| M6 | 贡献线 | “flag-flip + 模型 + null 结果”是否够顶会：需要 Intro 直接回答 | §1 |

### Ignored Alternative Explanations/Paths
1. 带宽主导：state bf16 同时改变 page size 与 block 数，收益可能部分来自分配粒度；
2. 页对齐主导：fp16 KV 列误差符号翻转说明 block 取整效应可压过模型预测；
3. 硬件/租机漂移：r45 的 TTFT 双峰 vs 全 ~18.5s 可能是环境敏感而非系统效应。

### Observations (Non-Defects)
- 08-09 的三条 CRITICAL（零宽 CI、单 seed pilot、负偏差）已被实际修复；
- 复现判定采用 PARTIALLY_REPRODUCIBLE 而非强行 VERIFIED，是正确做法。

---

# Phase 2 — Editorial Synthesis

## Part 1: 共识与分歧

### Points of Agreement

- **[CONSENSUS-5] 论文正文必须按 claim whitelist（08-10 版）重写后才能投稿**：
  serving 只能 ANALYZED 窄口径、容量按 KV 列限定、GSM8K 披露 power、
  harness 标注 chunk 级、RULER 用“无检测能力”措辞。无审稿人认为当前论文正文可投。
- **[CONSENSUS-5] 容量收益是强证据**：int4 列保守下界、复合 r_kv、2B/9B 跨规模。
- **[CONSENSUS-4] GSM8K seed 协议修复被接受**：9-seed + MDE/power 预注册 +
  决策规则，2B 回退显著、9B 无回退。
- **[CONSENSUS-4] 敏感度门与 RULER 多 seed 的统计修复被接受**。
- **[CONSENSUS-3] serving 主张必须 workload × threshold 限定**（EIC/R1/R2/R3）。

### Points of Disagreement

- **决策严格度**：EIC 认为 Evidence 仍 warn 且正文未重写 → 本轮回不到 Minor；
  R1/R2/R3 同判 Major；量规文件 65–79 名义上是 Minor 带，但契约 F2
  （多个 mandatory 维度 warn）触发 `major_revision`。**裁决**：按契约优先级
  Major Revision；加权分 71 左右只说明“接近但未达到可投状态”。
- **serving 的口径**：R1 主张连 Random60 过载区也只能报“方向性”；
  R2/R3 认为可在复现区间内写点估计。**裁决**：采用折中——正文可报
  “原始+复现均值区间 + CI”，但所有边界与方向翻转格只能出现在附录或限制节。

### DA-CRITICAL 裁决（IRON RULE 4）

| # | DA-CRITICAL | 裁决 | 处置 |
|---|---|---|---|
| C1 | serving headline 越界 | **VALIDATED（条件触发）** | 当前 claim whitelist 已降级；投稿前逐字检查 Abstract/Intro/结论，违者退回 |
| C2 | 容量误差未按 KV 列限定 | **VALIDATED** | §3.1 表格已按列报告；论文正文必须同步，禁止“<3.3%”单一口径 |

### DA-MAJOR 裁决摘要

| # | 裁决 | 处置 |
|---|---|---|
| M1 | 采纳 | GSM8K 章节同时报 p、CI、MDE、power，并写明“显著但按预注册 MDE 标准功效不足” |
| M2 | 采纳 | “无损失/持平”改为“无检测能力/CI 含 0”；RULER 仅点估计 |
| M3 | 采纳 | serving 机制段补 page/block 与带宽讨论，不把收益单独归因容量 |
| M4 | 采纳 | 显式 GDN-scope；跨架构探针列为 P2 |
| M5 | 采纳 | harness 方法学限制入正文，vLLM 侧为主证据 |
| M6 | 采纳 | Intro 必须正面回答“为什么这是系统论文”（测量+模型+闭环三位一体） |

## Part 2: Editorial Decision Letter

Dear Author(s),

感谢提交重审版《Precision Budget Allocation for Hybrid Linear Attention Models
in LLM Serving》。本轮 5 席全齐。补跑质量很高：GSM8K seed 语义、敏感度多重比较、
RULER 多 seed、S-formal 独立复现均已执行并归档，诚实披露到位。

### Decision: MAJOR REVISION

契约 F2 触发（多个 mandatory 维度 warn），无 F1/F3 block；加权 70.6–72.7
（平均 71.4）。与 08-09 轮相比，**证据链从 2/3 提升到约 3/4，且所有 DA-CRITICAL
均有明确处置**；当前不可投的核心原因不再是缺实验，而是**论文正文未按新证据重写**。

### 核心裁决点

1. **容量是 headline，serving 是限定性证据**：容量收益（int4 列 +38~41%@4K、
  保守下界）可作主卖点；serving 只能按 Random60 过载区 paired goodput 增益表述，
  ShareGPT 只写 500ms+ 边界持平。
2. **统计口径必须同时披露 power**：GSM8K 2B state 效应显著但 power 67.5%，
  不得只报 p 值。
3. **harness PPL 标注 chunk 级近似**，kernel 质量以 vLLM 侧为准。
4. **一般性收窄为 GDN-based 混合架构**，跨架构与 fp8/int8 放 future work。
5. **正文重写完成后即可触发重审**（预计 P0 工作量 3–5 天，含写作）。

## Part 3: Revision Roadmap

### Required Revisions (Must Fix)

| # | 修订项 | Source | Priority | 工作量 |
|---|---|---|---|---|
| R1 | 按 08-10 claim whitelist 重写 Abstract/Intro/结论：serving=ANALYZED、容量按 KV 列、GSM8K 报 power、harness 标注、RULER 措辞 | CONSENSUS-5 / DA C1 | P0 | 1–2 天 |
| R2 | Intro 直接回答“为什么这是系统论文”（测量+模型+闭环+负面结果的价值） | DA M6 / EIC W1 | P0 | 半天 |
| R3 | §3.3 GSM8K 段落补 MDE/power 与“显著但功效不足”披露 | R1 W1 / DA M1 | P0 | 2h |
| R4 | 容量模型定位段按 KV 列拆分保守下界与 signed error | DA C2 / R2 S1 | P0 | 2h |
| R5 | serving 段：Random60 原始+复现区间、ShareGPT 500ms+ 持平、边界敏感性入限制节 | CONSENSUS-3 / EIC W1 | P0 | 半天 |
| R6 | harness 方法学限制段（chunk 消融数据）+ vLLM 主证据声明 | R1 W4 / DA M5 | P1 | 2h |
| R7 | 文献扩展：Mamba2 state 精度、FlashInfer SSU、SSM 理论、prior-art 数值直觉 | R2 W1 | P1 | 0.5–1 天 |
| R8 | A_q/A_f 推导链 + 表 5.1 fp16 列 + signed error 列 + r_state(L) 曲线 | R2 S1 / S2 | P1 | 半天 |
| R9 | RULER 主表统一 no-think（FWE），多 seed 表注明“无检测能力” | R2 W4 / DA M2 | P1 | 半天 |
| R10 | 精度谱系边界声明（fp16 smoke、fp8/int8 future work） | R3 W4 | P1 | 2h |
| R11 | 成本/请求、TP、替代杠杆、train-inference mismatch 讨论 | R3 W1/W2/W3/W5 | P1 | 1 天 |
| R12 | 双贡献（A2 + state-bits）叙事缝合与章节结构决策 | EIC W4 / S7 | P1 | 半天 |

### Suggested Revisions (Should Fix)

| # | 修订项 | Source | Priority |
|---|---|---|---|
| S1 | Mamba2 2.7B 4K 容量探针或显式 GDN-scope | R2 W2 / R3 W4 | P2 |
| S2 | 独立复现的边界判定写成正式“复现容差 + 通过/不通过”记录 | R1 W1 | P2 |
| S3 | results/MANIFEST.json 索引 | EIC Minor | P2 |
| S4 | “bit-match”术语澄清（1e-9 容差） | R1 Minor | P3 |
| S5 | 旧方向（A2）与新方向（state-bits）是否同篇的决策文档 | EIC / S7 | P2 |

### 重审触发条件

R1–R6 + R7–R12 完成后，以 08-10 claim whitelist 为准做一次 `re-review` 验证
（逐条核对修订与证据），再进入 5 席重审。

---

## 附录 A — 主编侧独立核验记录

| 核验项 | 声明值 | 原始 JSON/CSV | 结论 |
|---|---|---|---|
| 容量 int4 误差 | −2.37/−3.24/−0.18/−1.07% | capacity-2x2-analysis.json | ✅ 一致 |
| 容量 fp16 误差 | +2.86/+3.53/−2.83% | 同上 | ✅ 一致 |
| r_kv 复合 | 2.245→2.675（2B@4K） | 同上 | ✅ 一致 |
| Q-stacking PPL | CI 含 0 | ppl-stacking-analysis-20260809.json | ✅ 一致 |
| GSM8K 2B 9-seed | −1.00pt [−1.71,−0.29] | gsm8k-state9seed-v2-analysis | ✅ 一致 |
| GSM8K 9B 9-seed | +0.33pt [−0.07,+0.73] | gsm8k-9b-state9seed-v2-analysis | ✅ 一致 |
| RULER 5 格 | 全部 CI 含 0 | ruler-statebf16-multiseed-analysis | ✅ 一致 |
| 敏感度 | 2/36 校正后不显著 | state-sensitivity-analysis-…-bonf.json | ✅ 一致 |
| chunk 消融 | 19.35→36.1 | chunk-ablation/ | ✅ 一致 |
| S-formal 边界 | 四列边界表 | statebf16-serving-formal-analysis.json | ✅ 一致 |
| 复现差异 | R60 1 处边界、SG 2 处边界、r45 >100% | repro 分析 + r45 取证 | ✅ 一致 |

## 附录 B — 文本 vs 证据状态对照

| 主题 | 08-09 状态 | 08-10 状态 |
|---|---|---|
| 容量模型 | 4 点验证 | ✅ 2×2 全矩阵 + 保守下界 + block 粒度 |
| GSM8K | 零宽 CI 退化 | ✅ 9-seed 显著/无回退 + power 披露 |
| RULER | 单 seed 非零格 | ✅ 3 dataset-seed 无检测差异 |
| 敏感度 | 未校正 | ✅ Bonferroni/BH-FDR |
| harness PPL | 未查 chunk | ✅ 已量化边界并降级为辅助证据 |
| serving | formal 待补 | ✅ 72/72 + 复现 PARTIALLY_REPRODUCIBLE |
| 论文正文 | 未重写 | ❌ 仍是投稿前最大缺口 |

## 附录 C — 审稿人状态与加权汇总

| 审稿人 | 推荐 | 加权 | Orig 20% | Rigor 25% | Evidence 25% | Coherence 15% | Writing 15% | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| EIC | Major Revision | 71.2 | 76 | 73 | 62 (warn) | 76 | 72 | warn |
| R1 | Major Revision | 70.6 | 74 | 68 | 58 (warn) | 80 | 82 | warn |
| R2 | Major Revision | 72.7 | 76 | 72 | 63 (warn) | 82 | 76 | warn |
| R3 | Major Revision | 71.2 | 75 | 70 | 61 (warn) | 80 | 76 | warn |
| DA | 不打分 | — | — | — | — | — | — | C1/C2 VALIDATED |
| 均值 | Major Revision | 71.4 | 75.3 | 70.8 | 61.0 | 79.5 | 76.5 | — |

**加权分与决策说明**：量规文件将 65–79 名义映射为 Minor Revision；但契约
`failure_conditions` 优先级高于量规，F2（多个 mandatory 维度 warn）触发
`major_revision`。本报告以契约为准，并在决策信中说明。
