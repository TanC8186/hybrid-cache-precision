# ARS 最大强度对抗式审稿报告（2026-08-06）

> 模式：`academic-paper-reviewer` **full**（EIC + R1 Methodology + R2 Domain + R3 Perspective + Devil's Advocate）
> 审稿对象：`docs/paper/paper-mainline-2026-08-03.md` + `docs/paper/serving-evaluation-2026-08-03.md`
> 审稿输入：论文草稿 + 仓库可归档证据（`results/verified/2026-08-04/`、`results/reproduction/2026-08-05/`）+ 外部核验（MLSys 2026 CFP/AE、Qwen3.5 架构、KIVI/KVQuant/TurboQuant/QPruningKV/RDKV/ARKV/MiniKV/HqeKV、DeepSeek V4 packed layout、RotorQuant/llama.cpp、本地 vLLM fork 源码）
> 契约：`reviewer/reviewer_full/v1`（panel_size=5，D1–D5，F0–F3）

## 0. 执行边界与诚实声明

1. **Sprint Contract 两调用协议未物理执行**：Phase 1（paper-blind 预承诺）与 Phase 2（paper-visible）在本内联会话无法真正调用隔离；本报告将契约作为评分判据与决策算术使用，但不宣称完成盲审预注册。需要无偏盲审时应在新会话按 2×5 调用重新执行。
2. **DA 角色不打分**：按 `devils_advocate_reviewer_agent.md` 硬边界，DA 不产出维度分；F0–F3 机械判定基于 4 位打分审稿人，DA-CRITICAL 按 IRON RULE 4 单独处理。
3. **未启用跨模型**（`ARS_CROSS_MODEL` 未设置），本报告为单模型面板。
4. **只读审稿**：未修改任何论文草稿；本报告为独立文档。

---

# Phase 0 — 领域分析与审稿人配置

## 1. 六维分析

| 维度 | 结果 |
|---|---|
| 主学科 | 计算机系统 / ML Systems（LLM serving 系统） |
| 次学科 | KV cache 量化与压缩、内存管理、性能评测方法、混合线性注意力架构 |
| 研究范式 | 定量实证系统研究（受控对比实验） |
| 方法类型 | 系统实现 + 基准评测（容量探针、离线质量评测、稳态 serving benchmark） |
| 目标期刊层级 | Q1 顶会：MLSys 2026 Research Track 首选；OSDI/ASPLOS（若贡献泛化为通用 KV 布局机制）；NeurIPS 系统方向次选 |
| 论文成熟度 | 修订草稿/预提交前：结构完整但缺 References/Discussion/Limitations/A2 章节；Evaluation 落后于仓库已验证证据 |

## 2. 推荐目标会议（Top 3）

1. **MLSys 2026** —— 主题完全匹配；CFP 明确鼓励代码/数据共享与 Artifact Evaluation（badging 标准）。
2. **OSDI/ASPLOS（系统方向）** —— 若 A2 packed per-layer page group 提炼为通用 vLLM V1 KV 布局机制并做多模型/多硬件验证。
3. **NeurIPS（ML systems / benchmarks）** —— 若重心收窄为“混合架构 KV 容量稀释测量研究”。

## 3. Reviewer Configuration Cards

### Card #1 — EIC（MLSys Area Chair）

- **身份**：MLSys 2026 Research Track Area Chair，主攻 LLM serving 系统软件（调度、内存、量化）；审稿偏好：系统贡献必须可部署、数字必须可复现、headline 必须诚实。
- **关注**：① 主题契合度与读者兴趣；② 原创性；③ 投稿文本与证据一致性；④ 投稿完整性。
- **盲区**：不深挖统计细节（交 R1），可能低估 PPL 矛盾的方法学严重性。

### Card #2 — R1（Methodology）

- **身份**：系统性能评测方法论学者，专攻 serving benchmark 协议（arrival process、warmup、SLO 边界、goodput 定义）与统计报告（3-seed mean±std、CI、效应量）。
- **关注**：① 实验协议是否支撑结论；② 论文内数字与仓库协议是否一致；③ 统计报告完整性；④ 复现门禁。
- **盲区**：对 KV 量化领域文献深度有限（交 R2）。

### Card #3 — R2（Domain）

- **身份**：KV cache 量化与 vLLM V1 内核专家（KIVI/KVQuant/TurboQuant 社区 + KV manager 实现）。
- **关注**：① 文献覆盖与引用准确性；② “first” 声明强度；③ A2 packed layout 的领域新颖性；④ 质量评测是否支撑近无损。
- **盲区**：可能较少考虑生产部署经济性（交 R3）。

### Card #4 — R3（Perspective）

- **身份**：内存系统/数据中心推理基础设施研究者（内存池、slab allocator、多租户、HBM 预算管理），兼有生产 inference provider 经验。
- **关注**：① 可部署性与可上游化；② 内存布局一般性（单 GPU vs 多 GPU）；③ 吞吐/容量/SLO 部署经济性；④ 被忽略的替代路径（直接压缩 GDN state）。
- **盲区**：对 KV 量化文献细节与 vLLM 内部实现深度有限（交 R2）。

### Card #5 — DA（Devil's Advocate）

- **身份**：对抗式系统研究者，专门攻击核心主张、数据-结论一致性、新颖性底线与替代解释。
- **关注**：① headline 是否被作者自己的最新证据推翻；② 系统贡献是否只是现有机制复用；③ 质量证据是否支撑近无损；④ 更简洁的替代解释。

---

# Phase 1 — 独立审稿报告

---

## EIC Review Report

### Reviewer Identity

MLSys 2026 Research Track Area Chair（LLM serving 系统软件方向）。

### Overall Recommendation

**Reject（Resubmit Encouraged）/ 等价 Major Revision 级整改** —— 当前提交文本未达可送审质量。

### Confidence Score

4 / 5

### Summary Assessment

研究问题（混合线性注意力 LLM 的 KV 量化/压缩）切中 MLSys 读者痛点；GDN state 不可量化且与 attention KV 共享内存池的结构特殊性成立。仓库实验纪律（commit-before-run、请求守恒、3-seed、独立复现门禁、失败保留）在同类投稿中属于优秀水平，E3 protocol-v2 已 VERIFIED，A2 的 runtime/capacity/serving 证据链完整。但**论文文本没有跟上证据**：Abstract 与 Introduction 仍在报告已被作者自己弃用的“+25% SLO”；A2（真正的系统贡献）没有进入正文，反而被写成 future work；PPL 三文件矛盾未解决；无 References/Discussion/Limitations。当前稿件若原样送审，审稿人会同时抓到“被撤回数字仍作为卖点”和“无新方法”两个致命问题。结论：研究有潜力，稿件不合格。

### Strengths

1. **S1 证据纪律罕见地好**：容量/复现/SLO 全部有 SHA 审计、独立 attempt 与 quarantine 语义。
2. **S2 诚实披露传统**：Evaluation 草稿主动声明单 run、warmup 协议不一致、per-layer 反噬。
3. **S3 核心容量测量完整**：2B/9B × 4096/16384 四组容量数字齐全，机制解释与代码形状一致。
4. **S4 新的系统证据链**：A2 gate → 跨主机复现 → serving formal 108/108。
5. **S5 审稿反馈闭环**：上轮 6 条 CRITICAL 中，SLO 伪影与 GDN dtype 已在证据层面修复。

### Weaknesses

1. **W1 已撤回 headline 仍在论文中（Critical）**：Abstract/Intro 写“+25% SLO（50 vs 40 req/s）”，但仓库 VERIFIED 结论是 Random 250ms 0%、1000–3000ms +14.3%、ShareGPT −17.6%，计划文档明令“禁止使用 +25%”。文本与作者自己的最新证据直接冲突。
2. **W2 论文没有新方法（Critical）**：A2 已实现并验证，但正文 §3.4 仍写“future work”；按当前文本，方法=stock vLLM `int4_per_token_head`，上轮“无新方法”critique 在文本层面未修复。
3. **W3 质量证据不足以支撑近无损（Major）**：PPL 三文件矛盾（13.86/11.67/11.03）、无 CI、无种子信息；4-bit “near-lossless +1.7%”是全篇前提。
4. **W4 投稿不完整（Major）**：无 references.bib、无 Discussion/Limitations、无 Data-Availability/artifact 声明。
5. **W5 “first system study” 声明过强（Major）**：RotorQuant/llama.cpp 对 Qwen3.5 已有社区测量，TurboQuant 也涉及混合/长上下文模型。

### Detailed Comments

#### Journal Fit

主题契合 MLSys；但系统贡献目前只在仓库而非论文。若 A2 正式入文并完成质量闭环，匹配度很高。

#### Originality

测量部分（GDN 摊薄）有价值但属初等代数推论；A2 packed layout 是唯一可能撑起新颖性的工程贡献，但复用 vLLM 为 DeepSeek V4 已有的 `_get_packed_kv_cache_layout` 路径，且只验证 Qwen3.5 与单 GPU——原创性必须靠“通用化 + 质量闭环 + 外部 baseline”证明。

#### Significance

若成立，影响限于“混合架构下 KV 量化该怎么做、能省多少”，对纯 attention serving 无直接价值；对 Qwen3.5 类模型部署方有实际价值。

#### Structural Coherence

标题/摘要/正文与仓库最新证据三者不一致（+25%、future work、旧 E2/E3 协议），是当前最大的结构性问题。

#### Title & Abstract

标题可接受；Abstract 需整段重写以对齐 workload-specific 边界与 A2 贡献。

#### Conclusion

草稿无 Conclusion/Discussion；现有 Limitation 只写了 per-layer 反噬，且已被 A2 部分修复，描述过时。

### Questions for Authors

1. 下一版是否把 Abstract/Intro 的 +25% 全部替换为 workload-specific VERIFIED 边界？
2. A2 会作为“方法”章节正式入文，还是继续作为 future work？若是前者，如何论证它不是对 DeepSeek V4 packed 布局的浅层复用？
3. PPL 三文件（13.86/11.67/11.03）哪一个是 canonical？最终会给 3-seed mean±std + CI 吗？
4. 距离投稿还有哪些 blocking 实验（质量闭环、external baseline、9B 长上下文 serving）？

### Minor Issues

- 图表仍为 08-03 版本，部分标签过时；
- vLLM 0.26.1rc1 未能在公开渠道精确核验（可见 0.26.1rc0）；
- “≈60% of KV budget”应标注为 code-derived estimate 并给出推导。

### Dimension Scores

| 维度 | 0–100 | 描述 | 契约分 |
|---|---|---|---|
| Originality (20%) | 55 | Weak–Adequate：测量有价值，A2 未入文 | — |
| Methodological Rigor (25%) | 62 | Adequate：仓库协议强，论文报告不完整 | warn |
| Evidence Sufficiency (25%) | 55 | Weak–Adequate：serving 证据厚，质量/基线缺失 | warn |
| Argument Coherence (15%) | 45 | Weak：文本与自身最新证据矛盾 | block |
| Writing Quality (15%) | 62 | Adequate：可读、诚实，但结构过时 | warn |
| 加权平均 | 56.75 | Major Revision（0–100 映射） | — |

### Failure Condition Checks

- F1（any mandatory dimension scores 'block'）：**fired: true**（D3 = block）
- F2（majority: two or more mandatory dimensions 'warn' or worse）：**fired: true**（D1/D2/D5 warn）
- F3（any high-priority dimension scores 'block'）：**fired: false**
- F0（every mandatory dimension scores 'pass'）：**fired: false**

### Editorial Decision（EIC 单卡）

`editorial_decision=reject_or_major_revision`

### Recommendation to Peer Reviewers

请 R1 核对论文内协议与仓库 VERIFIED 协议的差异；请 R2 评估 A2 的领域新颖性与外部 baseline 缺失；请 R3 评估单 GPU/单模型/单框架的系统一般性；请 DA 攻击“+25% 仍在论文中”与“无新方法”两个死穴。

---

## Methodology Review Report (Peer Reviewer 1)

### Reviewer Identity

系统性能评测方法论学者：Poisson 到达过程、warmup、SLO 边界、3-seed 统计、请求守恒审计。

### Overall Recommendation

**Major Revision**

### Confidence Score

5 / 5

### Summary Assessment

仓库层面的实验协议在同类工作中属于顶尖：E3 protocol-v2 的 72/72 formal + 48/48 复现 + 请求守恒 + 到达窗口审计达到可复现性“黄金标准”；A2 的 gate → 跨主机复现 → 108/108 serving formal 也是规范的放行链。但**论文草稿完全没有反映这套协议**：Evaluation 草稿仍在报告单 run E2/E3、`num_warmups=0`、旧 +25% 边界；PPL 质量证据三文件矛盾且无 CI；A2 serving formal 只到 ANALYZED，独立复现门禁未过；等字节预算对比非 byte-exact（5.4% 级偏差）。方法学结论：证据基础设施 A+，论文报告 C−。

### Strengths

1. **S1 请求守恒与失败语义**：`completed + failed = expected`、failed 计入 SLO miss denominator、failed/running 残留不重跑。
2. **S2 到达窗口固定 60s + warmup 120**：修复了把过载瞬时边界当服务能力的伪影。
3. **S3 独立复现门禁**：E3 80/80 cell ≤10%、60/60 边界精确、11/11 谬误扫描，verdict `REPRODUCIBLE`。
4. **S4 诚实保留失败**：`e3-formal-c7379f0-01/02` 等 QUARANTINED attempt 不与新分母合并。
5. **S5 A2 容量 gate 用三个独立 probe**：legacy/uniform/packed 各自独立 attempt，比率判据预设明确（3.232x / 0.833）。

### Weaknesses

1. **W1 论文内协议与仓库协议不一致（Critical）**：Evaluation §1 写 `num_warmups=0`、单 run/rate；仓库已有 3-seed bench_lat3 与 protocol-v2 稳态协议。按当前文本，读者会复现到作者自己已废弃的协议。
2. **W2 PPL 质量证据不满足统计报告标准（Critical）**：同一 4-bit 配置三文件 PPL 13.86 / 11.67 / 11.03（绝对差 ~26%），无 3-seed mean±std、无 CI、无效应量；Wikitext-2 仅 5×2048 序列，“near-lossless (+1.7%)”不能由单次运行承载。
3. **W3 A2 serving formal 仅 ANALYZED（Major）**：108/108 完成且审计 PASSED，但报告明确写“Quantitative formal claims remain blocked until independent reproducibility passes”。论文引用边界时必须标注 ANALYZED 或等复现完成。
4. **W4 等字节预算比较非 byte-exact（Major）**：3.19 MB vs 3.37 MB、4.79 vs 4.85 MB 等有 5.4% 级偏差；需严格等字节或公开容差并做敏感性说明。
5. **W5 缺外部 baseline 与 9B serving 完整矩阵（Major）**：KIVI/KVQuant/TurboQuant 无可执行对照；9B 只有容量与 E2 单 run，论文却用 9B 支持跨规模结论。

### Detailed Comments

#### Research Questions & Hypotheses

研究问题清晰、可测；但论文未把问题操作化为“容量 vs 质量 vs SLO 边界”三个可分离假设。

#### Research Design

受控对比设计合理（同硬件/同模型/同 SLO 协议、3 alloc 对比）；问题在于论文文本与仓库实际执行的协议版本不一致。

#### Sampling Strategy

workload 覆盖面不足：synthetic random（1024/128）+ ShareGPT300；缺 multi-turn、>16K 长上下文 serving、多长度混合与持续多租户负载。

#### Data Collection

第一手日志 + JSON + 源码三方对照优秀；但论文未描述 provenance 结构，读者无法从论文本身判断数字来源。

#### Analysis Methods

3-seed mean±std 与 t-CI 计划正确；论文主表仍是单 run；PPL 无统计；5 个 TTFT 阈值全部报告（避免 look-elsewhere），值得肯定。

#### Results Presentation

容量表清晰；SLO 表已过时（旧 +25%）；质量表存在跨文件矛盾；图 2 单 run 数据（1574/566）与 3-seed（1671.5/2163）不一致且无误差棒。

#### Reproducibility

仓库可复现性极强；论文缺 artifact/availability 声明与 reproduce 入口（MLSys 2026 CFP/AE 明确鼓励代码/数据共享与关键结果可复现）。

#### Methodological Fallacies Detected

- 选择报告/幸存者偏差（已规避）：失败 attempt 全部保留隔离，未混入分母——优秀。
- Look-elsewhere（已处理）：5 个阈值全报告并加 CAUTION——合格。
- 口径混合（存在）：论文主表单 run、笔记 3-seed、Abstract +25%，三者不一致。

### Questions for Authors

1. 论文最终 canonical 协议是哪个版本？请给出“论文表 → 仓库 JSON → 协议版本”映射。
2. PPL 三文件的 canonical 来源与 3-seed mean±std/CI 何时补？
3. A2 serving formal 边界在独立复现完成前如何标注（ANALYZED vs VERIFIED）？
4. 等字节预算对比能否 byte-exact，或给出明确容差与敏感性分析？

### Minor Issues

- 9B E2 单 run 数据未标注“scale-check only”；
- 缺 a priori 最小可检测差异/效应量说明；
- warmup 协议在离线（5 vs 120）与 serving 间不统一，需单独成段解释。

### Dimension Scores

| 维度 | 0–100 | 描述 | 契约分 |
|---|---|---|---|
| Originality (20%) | 55 | Weak–Adequate | — |
| Methodological Rigor (25%) | 58 | Weak–Adequate：证据基础设施强，论文报告弱 | warn |
| Evidence Sufficiency (25%) | 52 | Weak–Adequate：serving 厚、质量薄、基线缺 | warn |
| Argument Coherence (15%) | 55 | Weak–Adequate：内部协议不一致 | warn |
| Writing Quality (15%) | 60 | Adequate | warn |
| 加权平均 | 55.75 | Major Revision | — |

### Failure Condition Checks

- F1：**fired: false**
- F2：**fired: true**（D1/D2/D3 warn，满足 majority）
- F3：**fired: false**
- F0：**fired: false**

### Editorial Decision（R1 单卡）

`editorial_decision=major_revision`

---

## Domain Review Report (Peer Reviewer 2)

### Reviewer Identity

KV cache 量化与 vLLM V1 内核专家（KIVI/KVQuant/TurboQuant/QPruningKV 文献 + KV manager 实现）。

### Overall Recommendation

**Major Revision**

### Confidence Score

4 / 5

### Summary Assessment

研究问题在领域内真实存在：现有 KV 量化工作几乎全部假设纯 attention 架构，Qwen3.5 这类 hybrid（18/24 GDN + 6/8 GQA）确实带来新内存账本。容量模型与代码形状一致——我在本地 vendor/vllm 源码核对 `gated_delta_net_state_shape`：temporal (16,128,128) fp32 + conv (6144,3) bf16 = 1,085,440 B/layer，与论文 18.63 MiB 推导吻合。但领域贡献目前是“测量 + 工程验证”而非“方法”：A2 复用 vLLM 为 DeepSeek V4 已存在的 packed 布局并加开关；论文没写 A2；无外部 baseline；引用大量占位/未验证条目；“first”声明过强。方向正确，系统贡献需要重新包装并补齐质量与对照证据。

### Strengths

1. **S1 混合架构特殊性抓得准**：GDN state 与 attention KV 共享内存池、不可量化、随并发增长。
2. **S2 机制解释与代码闭合**：3.88x → 2.245x 稀释公式、page 对齐解释，与源码和日志一致。
3. **S3 A2 定位正确**：把 per-layer 反噬（×0.258）转化为可部署修复（packed/uniform=0.833）。
4. **S4 诚实报告 workload 反转**：ShareGPT 下 int4 边界低于 fp16（−17.6%）被如实记录。
5. **S5 文献尽职调查有实质内容**：QPruningKV（EMNLP 2025 Findings, arXiv:2412.12706）、RDKV（arXiv:2605.08317）、ARKV（arXiv:2603.08727）撞题识别准确，外部核验全部属实。

### Weaknesses

1. **W1 “first system study” 过强（Major）**：RotorQuant 与 llama.cpp q4_0 社区测量已覆盖 Qwen3.5 的 4-bit KV 质量；TurboQuant 在 vLLM 原生支持 sub-4-bit KV。建议改为“首个在 vLLM serving 栈内对 Qwen3.5 hybrid 做端到端容量/SLO 系统研究的报告”。
2. **W2 A2 新颖性论证不足（Critical）**：packed `offset/block_stride` 机制是 vLLM 为 DeepSeek V4 已存在的路径（外部核验：PR #44454/#46252、issue #47783 确认 upstream）。论文必须论证 A2 增量（混 dtype attention group + Mamba 共存、内存核算、Mamba reshape）为何不是“翻开关 + 复用布局”，并给出通用性证据。
3. **W3 质量证据不足支撑“近无损”（Critical）**：PPL 三文件矛盾未解决；无 retrieval/long-context（LongBench/RULER）；驱逐类排序仅 PPL 支撑，对系统主张无效。
4. **W4 无外部 baseline（Major）**：KIVI/KVQuant/TurboQuant 都没有同硬件/模型/SLO 协议的可执行对照；2.245x 是相对 fp16 的稀释后比值，无法说明与 2-bit/3-bit 系统的位置。
5. **W5 引用与表述不完整（Major）**：MiniKV、HqeKV 已核验存在，但 ThinKV/KV-Pareto/MiKV 未找到可验证来源；RW 使用占位符与 [VERIFY]；Qwen3.5 架构数字外部核验属实，但“262K 上下文”仍标 [VERIFY]。

### Detailed Comments

#### Literature Review

- **Coverage**：主干覆盖较好（KIVI/KVQuant/TurboQuant/QPruningKV/RDKV/ARKV/MiniKV/HqeKV）；ThinKV/KV-Pareto/MiKV 需给出可核验来源或删除；缺混合架构 serving 侧对照（vLLM LinearAttentionBackend、TensorRT-LLM Qwen3.5）。
- **Integration quality**：按量化/联合预算/线性注意力三线组织合理，但“系统对照”是空白。
- **Research gap argument**：gap 成立，但 “first” 措辞需收窄。

#### Theoretical Framework

- **Appropriateness**：容量模型 `r_s(L) = (A_f L + G)/(A_q L + G)` 简洁有效。
- **Application depth**：用于解释 2B/9B × 4096/16384 四组数据，闭合良好。
- **Alternative frameworks**：缺少“GDN state 自身可压缩”这一直接替代路径的讨论。

#### Academic Argument Quality

- **Factual accuracy**：架构数字（层数、索引、dtype）经外部 + 源码核验基本准确；vLLM 0.26.1rc1 未精确核验。
- **Argument logic**：3.88x → 2.245x 推导严谨；“60% of budget”需注明 code-derived。
- **Terminology precision**：“uniform int4 / packed per-layer / legacy per-layer”首次出现需定义清楚。

#### Contribution to the Field

- **Incremental contribution**：可辩护的增量是“hybrid 容量稀释的端到端测量 + 把 per-layer 保护做成容量中性”，不是新量化器/新内核。
- **Positioning**：需与 DeepSeek V4 packed 布局明确区分。
- **Overclaiming**：“first system study”与“near-lossless”均有过度声明风险。

#### Missing Key References

- KIVI（ICML 2024）、KVQuant（arXiv:2401.18079）、TurboQuant（ICLR 2026）正式引用；
- QPruningKV / RDKV / ARKV 完整元数据；
- MiniKV、HqeKV（ACL 2026 Findings）正式引用；
- RotorQuant / llama.cpp q4_0 作为社区证据引用（注明非同行评审）；
- LongBench/RULER 作为长上下文质量评估规范。

### Questions for Authors

1. A2 相对 vLLM 现有 packed 路径的增量是什么？请给出改前/改后代码级说明。
2. 用哪个 retrieval/long-context benchmark 证明“容量恢复没有以质量为代价”？
3. ThinKV/KV-Pareto/MiKV 能否提供可核验 arXiv/DOI？
4. “first system study”最终收窄到什么范围？

### Minor Issues

- “262K 上下文”补核验或删除；
- 9B 的 `mamba_ssm_dtype=float32` 需与 2B 一样显式记录；
- RW 占位符（`[*KIVI*]` 等）必须全部替换。

### Dimension Scores

| 维度 | 0–100 | 描述 | 契约分 |
|---|---|---|---|
| Originality (20%) | 52 | Weak：测量 + 工程复用，A2 未入文 | — |
| Methodological Rigor (25%) | 60 | Adequate | warn |
| Evidence Sufficiency (25%) | 50 | Weak：质量与基线缺 | warn |
| Argument Coherence (15%) | 50 | Weak：与自身证据不一致 | warn |
| Writing Quality (15%) | 58 | Adequate | warn |
| 加权平均 | 54.10 | Major Revision | — |

### Failure Condition Checks

- F1：**fired: false**
- F2：**fired: true**（D1/D2/D3 warn）
- F3：**fired: false**
- F0：**fired: false**

### Editorial Decision（R2 单卡）

`editorial_decision=major_revision`

---

## Perspective Review Report (Peer Reviewer 3)

### Reviewer Identity

内存系统/数据中心推理基础设施研究者（内存池、slab allocator、多租户、HBM 预算管理），兼有生产 inference provider 经验。

### Overall Recommendation

**Major Revision**

### Confidence Score

3 / 5

### Summary Assessment

从生产系统视角，这篇论文回答了一个真实问题：混合架构里“量化 attention KV”到底能买回多少并发与上下文。容量模型和 A2 packed layout 的设计直觉（避免统一 page 导致的 group 爆炸）与操作系统 slab/内存池经验高度一致，是可借鉴的好思路。但当前版本距离“可部署的系统研究”还差三层：一是只在单卡、单模型、单框架 fork 上验证；二是没有回答“为什么不直接压缩 GDN state”——int4 最大并发下 GDN state 约占 KV 池 60%，attention KV 量化已是边际收益；三是论文没有给生产决策者需要的吞吐/成本/SLO 权衡表（ShareGPT 下 int4 边界反而 −17.6%）。结论：方向有价值，当前是“实验室系统报告”，不是“生产可部署系统论文”。

### Strengths

1. **S1 用 SLO 边界而非原始吞吐做结论**：goodput/offered ≥ 0.95 是生产运维语言。
2. **S2 workload 分开报告**：Random 与 ShareGPT 方向相反被明确披露，避免 Simpson 式误导。
3. **S3 A2 布局直觉正确**：混 dtype attention 层合并为一个 group、复用 packed offset/stride，符合“减少 group 数、避免页大小统一”的内存池设计原则。
4. **S4 容量-上下文缩放关系**：16K 下稀释减弱（3.155x/3.167x）回应长上下文场景。
5. **S5 失败与隔离语义成熟**：QUARANTINED/FAILED_RUNTIME_COLLECTION 等状态管理达到生产级审计水平。

### Weaknesses

1. **W1 系统一般性未验证（Major）**：只在 RTX 5090（sm_120）单卡、vLLM fork、Qwen3.5 上验证；无多 GPU/TP、无其他 hybrid 模型（Jamba/Zamba/RecurrentGemma/Qwen3-Next）、无上游 patch 可维护性说明；flag 默认关闭意味着默认路径零收益。
2. **W2 未讨论 GDN state 压缩这一直接替代路径（Major）**：GDN state 18.63 MiB/request、int4 最大并发时约占 KV 预算 60%；若 temporal fp32→bf16/int8 可行，论文“上限”论述会变化。审稿人一定会问：为什么不先压缩最大的那部分？
3. **W3 部署经济性权衡缺失（Major）**：容量收益（2.245x）与代价（TPOT +8%、ShareGPT SLO 边界 −17.6%、吞吐 −5.5%）需要一张净收益表。
4. **W4 长上下文主张超出证据（Major）**：论文称 hybrid KV 量化有 long-context advantage，但 serving 只测到 16K 容量，无 32K/64K/262K serving 或质量证据。
5. **W5 缺少多租户/前缀缓存视角（Minor–Major）**：A2 对 prefix-cache 命中、块对齐（block_size 16 vs 64）、动态再平衡的影响未讨论。

### Detailed Comments

#### Assumption Audit

- **Explicit assumptions**：GDN state 不可量化且固定 fp32——全篇稀释论述的前提，论文把它当常数，没有给“state 可压缩性”的下界或论证。
- **Implicit assumptions**：假设“量化 attention KV 是混合架构内存优化的正确杠杆”；60% 预算在 state 上时，杠杆排序存疑。
- **Paradigmatic assumptions**：假设单 GPU 内存池、单框架调度语义可代表生产；多 GPU 下 GDN state 的 TP 分片与通信开销会改变账本。

#### Cross-Disciplinary Connections

- **Parallel research**：OS slab 分配器、memcg 分组、HugeTLB 对齐与 vLLM block pool/page group 是同构问题。
- **Borrowing opportunities**：可借“per-object-size-class slab”语言，把 A2 表述为“混合 precision slab 而非全局统一 page”。
- **Methodological borrowing**：生产推理常用“容量 × 每 token 成本 × SLO 达标率”决策；建议补 TTFT/TPOT CDF 与成本模型。

#### Practical Impact

- **Real-world application**：对跑 Qwen3.5 系列的推理服务有意义；对纯 attention 模型无直接价值（论文应明确边界）。
- **Implementation feasibility**：fork patch + 默认关闭的开关，离上游可用有距离；需补 vLLM upstream PR 可行性评估。
- **Stakeholders**：缺“推理服务运营商”（吞吐/成本权衡）与“框架维护者”（patch 可维护性）视角。

#### Broader Implications

- **Ethical dimensions**：不显著；但“近无损未证”可能误导生产用户，建议写明“质量评估完整前勿用于生产”。
- **Social impact**：低。
- **Future directions**：GDN state 压缩、per-dtype 动态再平衡（A1）、多 GPU 布局、上游化。

### Cross-Disciplinary Reading Recommendations

- vLLM KV cache 布局 RFC（issue #42082）；
- DeepSeek V4 packed layout PRs（#44454/#46252）；
- Linux slab allocator / memcg 文档（per-type slab 与共享池类比）；
- LongBench/RULER（长上下文质量评估）；
- MLSys 2026 CFP/AE（artifact 与复现规范）。

### Questions for Authors

1. 如果 GDN temporal state 可 bf16 化，论文的 2.245x 会变成多少？请给出敏感性分析。
2. A2 布局在 prefix-cache hit 场景下的块对齐损失是多少？
3. 能否给出“每 1000 token 服务成本”或等效部署经济性指标？
4. 多 GPU（TP≥2）下 GDN state 分片与 packed 布局是否仍成立？

### Minor Issues

- “≈60% of KV budget”应给出并发区间的函数而非单点；
- 建议增加“容量 × 吞吐 × SLO 达标率 × PPL”净收益表。

### Dimension Scores

| 维度 | 0–100 | 描述 | 契约分 |
|---|---|---|---|
| Originality (20%) | 60 | Adequate：布局直觉有价值 | — |
| Methodological Rigor (25%) | 58 | Weak–Adequate | warn |
| Evidence Sufficiency (25%) | 55 | Weak–Adequate：缺多 GPU/长上下文/经济性 | warn |
| Argument Coherence (15%) | 52 | Weak–Adequate | warn |
| Writing Quality (15%) | 60 | Adequate | warn |
| 加权平均 | 57.05 | Major Revision | — |

### Failure Condition Checks

- F1：**fired: false**
- F2：**fired: true**
- F3：**fired: false**
- F0：**fired: false**

### Editorial Decision（R3 单卡）

`editorial_decision=major_revision`

---

## Devil's Advocate Review

### Strongest Counter-Argument

这篇论文目前最致命的问题不是缺数据，而是**论文说的和作者自己证明的不是一回事**。作者在 Abstract 和 Introduction 写“uniform int4 把 SLO 容量提升 25%（50 vs 40 req/s）”，但他们自己的 VERIFIED 审计报告（`results/verified/2026-08-04/e3/validation_report.md`）明确给出：Random 250ms 阈值收益 0%、500ms +4.8%、1000–3000ms +14.3%，而真实 ShareGPT trace 下 int4 的可持续边界比 fp16 **低 17.6%**（23.33 vs 28.33 req/s）。也就是说，投稿文本仍在宣传一个作者自己已经撤回的、只在过载瞬时态出现的伪影数字。任何一位认真审稿人读到 Abstract 再核对作者自述的补充材料，都会得到“数据-结论不一致”的结论——这是被拒的最快路径，也是无法用“未来补实验”挽回的第一印象。同时，“新方法”主张也站不住：正文没有 A2；而仓库里的 A2 本质上是把 vLLM 为 DeepSeek V4 已存在的 packed 布局（`UniformTypeKVCacheSpecs`、`_get_packed_kv_cache_layout`）用开关扩展到 Qwen3.5 的混 dtype GQA 层，外加 Mamba reshape 修复。如果没有质量闭环和外部 baseline，一个更简洁的解释是：作者对混合架构做了一次诚实的容量测量，其数值是架构代数而非新机制发现；其系统贡献（A2）则是已有机制的工程接线。当前稿件无法反驳这个解读。

### Issue List

#### CRITICAL

| # | 维度 | Issue Description | Location | Field-Norm Boundary | Evidence-Crossing Rationale |
|---|---|---|---|---|---|
| C1 | 数据-结论一致性 | Abstract/Intro 仍在报告已撤回的“+25% SLO”；作者自己的 VERIFIED 结论为 workload-specific（Random 0%/+4.8%/+14.3%，ShareGPT −17.6%） | paper-mainline Abstract、§1 Contributions (a) | 不依赖场域规范：内部矛盾即可判定 | 仓库 `e3/validation_report.md` 与 `next-stage-experiment-plan-2026-08-04.md` §7.3 明文禁止使用 +25% |
| C2 | 证据缺口 | “4-bit near-lossless +1.7% PPL”是全篇前提，但 PPL 三文件矛盾（13.86/11.67/11.03）、无 CI/种子；无 retrieval/long-context 验证 | Eval §6 Table 4、paper-mainline §3.2 | MLSys 2026 CFP/AE 鼓励代码/数据共享与关键结果可复现（mlsys.org 官方页面） | 同一配置三个数字绝对差 ~26%，连论文内部一致性都不满足 |
| C3 | 核心贡献缺失 | 论文正文没有 A2（唯一可能的新方法）；per-layer page group 仍被写成 future work；按当前文本方法=stock vLLM dtype | paper-mainline §3.4、Eval §8 | 不依赖场域规范 | 仓库已实现并验证 A2（gate PASSED、serving 108/108），文本与自己的工程状态矛盾 |

#### MAJOR

| # | 维度 | Issue Description | Location | Field-Norm Boundary | Evidence-Crossing Rationale |
|---|---|---|---|---|---|
| M1 | 替代路径 | GDN state 占 int4 最大并发 KV 预算 ~60%，论文把 state 当固定常数，未讨论 state 压缩/降精度 | paper-mainline §3.3 | 不依赖场域规范 | 若 state 可压缩，“attention-only 量化上限”结论会变；至少需要敏感性分析 |
| M2 | 新颖性 | A2 复用 vLLM 为 DeepSeek V4 已存在的 packed layout 机制，增量仅为开关+混 dtype 分组+Mamba reshape；通用性未证明 | 仓库 per-layer-page-group-design、vendor/vllm `_get_packed_kv_cache_layout` | 不依赖场域规范 | 外部核验：vLLM PR #44454/#46252、issue #47783 确认 packed path 已 upstream |
| M3 | 过度声明 | “first system study”被 RotorQuant/llama.cpp q4_0（Qwen3.5 社区测量）与 TurboQuant（vLLM 原生 sub-4-bit）削弱 | paper-mainline §2.4 | 不依赖场域规范 | 外部核验确认这些工作存在且覆盖 Qwen3.5/混合模型 |
| M4 | 系统证据缺口 | A2 serving formal 只有 ANALYZED，未过独立复现；论文若引用 40 vs 35 等边界必须标注状态 | results/reproduction slice reports | MLSys AE 要求关键结果可复现 | slice 报告原文写明 quantitative formal claims 在独立复现前保持 blocked |

#### MINOR

| # | 维度 | Issue Description | Location |
|---|---|---|---|
| m1 | 引用完整性 | ThinKV/KV-Pareto/MiKV 无可核验来源；`[*KIVI*]` 等占位符未替换 | RW §2 |
| m2 | 版本核验 | vLLM 0.26.1rc1 未在公开渠道核验（可见 0.26.1rc0） | Eval §1 |
| m3 | 一致性 | 图 2 单 run 数据（1574/566）与 3-seed（1671.5/2163）不一致且无误差棒 | results/figures/fig2 |

### Ignored Alternative Explanations/Paths

1. **GDN state 压缩**：18.63 MiB/request 是最大单项；temporal fp32→bf16 直接减半，论文完全未测。
2. **byte-matched 外部 baseline**：真正该比的是同字节预算下 KIVI 2-bit / KVQuant 3-bit / TurboQuant / eviction-only。
3. **原生上游路径**：与其维护 fork patch，不如把 per-layer page group 做进 vLLM upstream，否则部署方不会采用。
4. **9B 才是主场景**：2B KV 预算充裕；9B（KV 预算仅 6.5 GiB）才是容量受限的真实部署场景，却被降级为 scale-check。
5. **不做 serving 的窄贡献**：若 A2 无法及时完成质量闭环，rescope 为“hybrid 容量稀释测量 + 等字节质量排序”也是可投稿的窄贡献。

### Missing Stakeholder Perspectives

- 推理服务运营商（吞吐/成本/SLO 净收益）；
- vLLM 框架维护者（patch 上游化与回归面）；
- 多租户云用户（隔离、突发、prefix-cache 命中）；
- Qwen 模型团队（GDN state dtype 是否可改、未来架构是否保持 GDN 布局）。

### Unexamined Premise

全篇隐含前提：**GDN recurrent state 是固定、不可压缩的常数，因此唯一可优化的是 attention KV。** 论文从未给出该前提的下界论证或敏感性分析。若该前提不成立，论文的“上限”论述与“容量随上下文放大”的机制故事都会改变。

### Observations (Non-Defects)

- 仓库证据纪律（SHA、attempt、quarantine、独立分母）是最好的投稿材料之一，建议写入 Data Availability 作为卖点；
- E3 v2 的 11/11 谬误扫描说明作者对统计陷阱有意识；
- A2 serving formal 108/108 即使只是 ANALYZED，工程完成度也值得如实呈现。

---

# Phase 2 — Editorial Synthesis

## Part 1: Editorial Decision Letter

Dear Author(s),

感谢提交《Uniform 4-bit KV Cache Quantization for Hybrid Linear-Attention LLMs》（工作稿）至 MLSys 2026 模拟审稿。本报告由 4 位打分审稿人（EIC + 3 位领域审稿人）与 1 位 Devil's Advocate 独立评审后合成。我们对仓库中的实验纪律和证据基础设施给予明确肯定；但当前投稿文本无法进入送审流程。

### Decision: REJECT（Resubmit Encouraged）

契约机械判定：`editorial_decision=reject_or_major_revision`（F1 触发：至少一位审稿人在 mandatory 维度给出 block）。

### Consensus Analysis

#### Points of Agreement (Consensus)

- **[CONSENSUS-4] 论文文本必须切换到 canonical 已验证证据**：Abstract/Intro 的 +25% 与旧 E2/E3 协议必须删除，替换为 workload-specific 的 VERIFIED 边界（EIC W1、R1 W1、R2 W3 相关、R3 W4 相关）。
- **[CONSENSUS-4] A2 系统贡献必须进入正文并重新定位**：不能再写为 future work；必须论证与 DeepSeek V4 packed 布局的增量（EIC W2、R2 W2、R3 S3）。
- **[CONSENSUS-3] PPL 质量证据必须先闭环**（R1 W2、R2 W3、EIC W3；R3 未提及）：统一三文件、3-seed mean±std + CI、补 retrieval/long-context。
- **[CONSENSUS-3] A2 质量闭环（packed vs uniform 的 PPL/retrieval）是投稿前提**（R2 W3、R1 W4 相关、R3 W3；EIC 未单独提及）。

#### Points of Disagreement

- **决策严格度（Reject vs Major Revision）**：EIC 认为当前文本应拒稿重投；R1/R2/R3 建议 Major Revision。**Editor's Resolution**：按契约 F1 判定为 `reject_or_major_revision`，操作化为 **Reject（Resubmit Encouraged）**——论文含已撤回 headline 属于投稿层面不可接受的内部矛盾；但所有 blocking 项都有明确修复路径且大部分证据已在仓库中，强烈鼓励重投。
- **A2 贡献定性**：R2/R3 认为 A2 目前是“工程接线”（复用 DeepSeek V4 布局）；EIC 认为完成通用性与质量闭环即可辩护。**Editor's Resolution**：采纳 R2/R3 的怀疑立场作为必答问题，作者必须在论文中给出机制增量证明（改前/改后 diff、多模型验证）。
- **9B 角色**：R2 认为 9B 应做主场景；论文目前以 2B 为 headline。**Editor's Resolution**：保留 2B 为完整矩阵、9B 为规模验证可以，但 Discussion 必须解释“为什么 2B 的结论能外推到容量受限的 9B 部署”。

### Decision Rationale

当前稿件最严重的单一问题是**文本与作者自己的已验证证据冲突**：+25% 是仓库明令弃用的数字，却仍是 Abstract 的 headline，直接触发 DA-CRITICAL C1 与 EIC 的 D3 block。其次，PPL 质量证据（C2）与 A2 缺席（C3）分别破坏“近无损”与“新方法”两个核心主张。四位打分审稿人加权平均在 54–57 区间（Major Revision 档），但契约判定因 mandatory block 升级为 `reject_or_major_revision`。我们选择 Reject（Resubmit Encouraged）而不是温和 Major Revision，是因为投稿文本层面的矛盾必须先“重写”而非“修补”；同时，仓库证据（E3 VERIFIED、A2 gate PASSED、serving formal 108/108、完整 provenance）表明作者具备完成重投的实力。若作者在下版完成 Required Revisions 的 R1–R6，我们预期可按“Major Revision 后重审”或直接新投稿处理。

### Summary of Key Issues

1. 已撤回的 +25% 仍是论文 headline（DA-CRITICAL C1 / EIC D3 block）。
2. 唯一系统贡献 A2 未进入正文，仍为 future work（DA-CRITICAL C3）。
3. 4-bit 近无损缺乏可信统计支撑，PPL 三文件矛盾（DA-CRITICAL C2）。
4. 无外部 baseline、无质量闭环、无 references/artifact 声明（R2/R1/EIC 共识）。

---

## Part 2: Revision Roadmap

### Required Revisions (Must Fix)

| # | Revision Item | Sub-Claim(s) | Source | Priority | Estimated Effort |
|---|---|---|---|---|---|
| R1 | 全篇替换为 workload-specific VERIFIED 边界；删除 +25% 与旧 E2/E3 单 run 表格 | SC-1 | EIC/R1/R2/R3 | P1 | 2–3 天 |
| R2 | 新增 A2 方法章节（packed per-layer page group），给出机制增量论证与改前/改后 diff 摘要 | SC-2 | EIC/R2/R3 | P1 | 3–5 天 |
| R3 | 统一 PPL 三文件为 canonical 3-seed mean±std + CI；补 retrieval/long-context（LongBench/RULER） | SC-3 | R1/R2/EIC | P1 | 1–2 周（含实验） |
| R4 | A2 质量闭环：packed vs uniform 多 seed PPL + retrieval，证明容量恢复无质量回退 | SC-4 | R2/R1/R3 | P1 | 1–2 周 |
| R5 | 外部 baseline：KIVI/KVQuant/TurboQuant（或 byte-equivalent 替代）同硬件/模型/SLO 协议 | SC-5 | R2/R1 | P1 | 2–3 周 |
| R6 | 收窄 “first system study” 声明；替换全部占位引用；补齐 references.bib | SC-6/SC-11 | R2/EIC | P1 | 2–3 天 |
| R7 | A2 serving formal 边界标注 ANALYZED（或等独立复现后 VERIFIED）；9B E2 标注 scale-check | SC-8/SC-9 | R1/R3 | P1 | 1 天 |
| R8 | 新增 Discussion/Limitations/Data-Availability；包含 GDN state 压缩敏感性分析与多 GPU 局限 | SC-9/SC-10 | R3/R2/EIC | P1 | 3–5 天 |

### Suggested Revisions (Should Fix)

| # | Revision Item | Sub-Claim(s) | Source | Priority | Estimated Effort |
|---|---|---|---|---|---|
| S1 | 等字节预算对比改为 byte-exact 或声明容差+敏感性 | SC-12 | R1 | P2 | 2–3 天 |
| S2 | 增加“容量×吞吐×SLO×PPL”净收益表与部署经济性讨论 | — | R3 | P2 | 2 天 |
| S3 | 前缀缓存命中/块对齐的 A2 分析 | — | R3 | P2 | 2–3 天 |
| S4 | vLLM 版本与 Qwen3.5 262K 上下文等 [VERIFY] 项核验 | — | R2 | P2 | 1 天 |
| S5 | 图 2/图 3 换用 3-seed 数据并加误差棒 | SC-7 | R1/EIC | P2 | 1 天 |
| S6 | artifact 声明：reproduce 入口、SHA manifest、quarantine 语义（MLSys AE 规范） | SC-13 | R1/R3 | P2 | 1 天 |

### Revision Checklist (Checkable List)

#### Priority 1 — Structural Revisions（估计 4–6 周）

- [ ] R1: Abstract/Intro/Eval 主表切换到 VERIFIED workload-specific 边界；全文搜索并删除 “+25%”
- [ ] R2: A2 方法章节 + 机制增量论证 + 与 DeepSeek V4 packed 路径的边界说明
- [ ] R3: PPL canonical 统一（3-seed mean±std + CI）+ LongBench/RULER
- [ ] R4: packed vs uniform 质量闭环（PPL + retrieval）
- [ ] R5: 外部 baseline 三选一（KIVI/KVQuant/TurboQuant），同协议对比
- [ ] R6: references.bib 完整、全部占位符替换、first-claim 收窄
- [ ] R7: A2 serving formal 状态标注（ANALYZED/VERIFIED）+ 9B 标注
- [ ] R8: Discussion/Limitations/Data-Availability + GDN state 敏感性

#### Priority 2 — Content Supplementation（估计 1–2 周）

- [ ] S1: byte-exact 或容差声明
- [ ] S2: 净收益表
- [ ] S3: prefix-cache 分析
- [ ] S4: [VERIFY] 核验
- [ ] S5: 3-seed 图表
- [ ] S6: artifact/availability 声明

#### Priority 3 — Text and Formatting（估计 2–3 天）

- [ ] 全部 Minor Issues（版本号、图表标签、术语首次定义、引文格式）

### Revision Deadline

- 模拟场景建议 **6–8 周**（Major Revision 标准窗口）；R5 外部 baseline 与 R3/R4 质量闭环是长尾项。
- 若投稿节奏不允许，建议先完成 R1/R2/R3/R6/R7/R8 作为“重投骨架”，R4/R5 作为 blocking 补实验后再投。

### Response Letter Template

请使用 ARS `templates/revision_response_template.md` 的 R→A→C 格式逐条回应：每条 Required/Suggested Revision 写明作者回应、修改位置与验收标准；对不采纳的建议给出理由。DA-CRITICAL C1/C2/C3 需要单独成段回应。

---

## Part 3: Reviewer Report Summary (Appendix)

### EIC Report Summary

- Recommendation: Reject（Resubmit Encouraged）| Confidence: 4
- Key Point: 研究有潜力，但投稿文本仍在宣传已被自己撤回的 +25%，且 A2 未入文。

### Reviewer 1 (Methodology) Summary

- Recommendation: Major Revision | Confidence: 5
- Key Point: 证据基础设施 A+，论文报告 C−；PPL 矛盾与协议口径混合是主因。

### Reviewer 2 (Domain) Summary

- Recommendation: Major Revision | Confidence: 4
- Key Point: 方向真实、代码闭合，但 “first” 过强、A2 增量未证、质量与基线缺失。

### Reviewer 3 (Perspective) Summary

- Recommendation: Major Revision | Confidence: 3
- Key Point: 单卡/单模型/单框架验证不足；未回答“为何不压缩占 60% 预算的 GDN state”。

### Devil's Advocate Summary

- 定位：只挑战不打分 | CRITICAL 3 项（C1 数据-结论矛盾、C2 质量证据、C3 A2 缺席）
- Key Point: 论文当前最容易被一句话杀死：作者自己的验证报告推翻了论文的 headline。

---

## Appendix A — 文本 vs 仓库证据状态对照

| 主题 | 论文草稿（08-03） | 仓库证据（08-04/05/06） | 状态 |
|---|---|---|---|
| SLO headline | +25%（50 vs 40） | Random 0%/+4.8%/+14.3%；ShareGPT −17.6%（VERIFIED） | 文本与证据冲突 |
| E2/E3 协议 | 单 run、warmup-0 | protocol-v2 稳态、3-seed、独立复现 | 文本落后 |
| per-layer 混 dtype | 容量 ×0.258，future work | A2 packed 修复：×3.232 legacy / ×0.833 uniform | 文本落后 |
| A2 serving | 未提及 | 108/108 completed_validated（ANALYZED） | 文本落后 |
| PPL | 13.86（+1.7%） | 三文件 13.86/11.67/11.03 | 未闭环 |
| 外部 baseline | 无 | 无 | 缺失 |
| references | 占位符 | — | 缺失 |
| GDN dtype | 未显式记录 | runtime 固化 float32 | 证据已补、文本未补 |

## Appendix B — 外部核验记录（2026-08-06）

- Qwen3.5-2B/9B 混合架构（18/24 GDN + 6/8 GQA）：属实（HF/TransformerLens/社区仓库）。
- Qwen3.5 `mamba_ssm_dtype=float32`：属实（HF config 样本）。
- QPruningKV（EMNLP 2025 Findings, arXiv:2412.12706）：属实。
- RDKV（arXiv:2605.08317）、ARKV（arXiv:2603.08727）：属实。
- KIVI（ICML 2024）、KVQuant（arXiv:2401.18079）、TurboQuant（ICLR 2026, vLLM 3/4-bit 模式）：属实。
- MiniKV、HqeKV（ACL 2026 Findings）：属实；ThinKV/KV-Pareto/MiKV：未核验。
- RotorQuant / llama.cpp q4_0（Qwen3.5 社区 KV 压缩）：存在，非同行评审。
- DeepSeek V4 packed KV layout（vLLM `UniformTypeKVCacheSpecs`）：属实（PR #44454/#46252、issue #47783）。
- MLSys 2026 CFP/AE：属实，官方鼓励代码/数据共享与 Artifact Evaluation（badging）。
- vLLM `int4_per_token_head`、`enable_per_layer_page_groups`、`_get_packed_kv_cache_layout`、`gated_delta_net_state_shape`：在本地 vendor/vllm 源码确认存在；GDN state 1,085,440 B/layer 推导与源码形状一致。
- vLLM 0.26.1rc1：未精确核验（公开渠道可见 0.26.1rc0）。
