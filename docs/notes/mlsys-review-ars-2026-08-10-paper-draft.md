# ARS 最大强度审稿报告（2026-08-10 · 论文正文实体首审版）

> 模式：`academic-paper-reviewer` **full**（EIC + R1 Methodology + R2 Domain +
> R3 Perspective + DA，5/5 全齐）
> 审稿对象：`paper/mlsys2026/main.tex` + `main.pdf`（8 图、2 表、27 引文、
> 匿名双盲初稿）+ 全证据库（`results/verified/2026-08-08|09`、
> `results/quality/`、`docs/notes/repro-final-2026-08-09.md`）
> 契约：`reviewer/reviewer_full/v1`（panel_size=5，D1–D5，F0–F3）

## 0. 执行边界与诚实声明

1. 单模型面板（`ARS_CROSS_MODEL` 未设置），无独立跨模型盲检；
2. Sprint Contract 的两调用隔离无法在本内联会话物理执行，契约仅作为评分判据与
   决策算术使用，不宣称完成盲审预注册；
3. DA 不打分；F0–F3 机械判定基于 4 位打分审稿人；
4. 只读审稿：除本报告外未修改任何论文/实验文件；
5. 加权分数沿用 08-09/08-10 轮会议校准口径；量规 65–79 名义为 Minor，但契约
   F2 优先级更高，多个 mandatory 维度 warn 时触发 `major_revision`；
6. 本报告是**论文正文实体**（非 research-summary）的首次最大强度审稿。

---

# Phase 0 — 领域分析与审稿人配置

## 1. 六维分析

| 维度 | 结果 |
|---|---|
| 主学科 | 计算机系统 / ML Systems（LLM serving 内存与精度预算） |
| 次学科 | KV cache 量化、循环 state 精度、混合线性注意力、容量建模、serving 评测 |
| 研究范式 | 定量实证系统研究（受控对比 + 解析建模 + 闭环评测） |
| 方法类型 | 系统测量 + 解析容量模型 + 统计配对实验 + 独立复现 |
| 目标层级 | MLSys 2026 Research Track（Q1 顶会） |
| 论文成熟度 | Pre-submission 前夜：结构完整、语言已润色、图表齐全；仍缺若干披露与定位修正 |

## 2. Reviewer Configuration Cards

### Card #1 — EIC（MLSys Area Chair）
- 关注：二维预算定位是否仍是“配置报告”，容量 headline 是否足够撑起系统论文，
  8 图是否服务论证，正文是否兑现摘要承诺；
- 盲区：统计细节（由 R1 补）。

### Card #2 — R1（Methodology）
- 身份：serving 评测方法论 + 统计报告专家；
- 关注：serving 矩阵多重比较、RULER 的选择性复测、n=3 配对推断、power/MDE
  披露、“独立复现”的操作定义、capacity 探针的确定性声明；
- 盲区：vLLM 代码链与文献归属（由 R2 补）。

### Card #3 — R2（Domain）
- 身份：线性注意力/SSM serving 与量化专家；
- 关注：vLLM dtype 引文准确性、RULER think 协议伪影、state 精度 prior art
  完整性、容量模型推导链、GDN-scope 表述；
- 盲区：部署经济性与替代杠杆（由 R3 补）。

### Card #4 — R3（Perspective）
- 身份：数据中心推理基础设施 / 成本-性能研究者；
- 关注：容量→收益的落地换算（cost/request、请求/小时）、TP/多卡、目标部署
  场景、图密度与可读性；
- 盲区：统计校正（由 R1 补）。

### Card #5 — DA（Devil's Advocate）
- 身份：对抗式研究者；只挑战不打分；
- 关注：flag-flip 反驳是否成立、标题是否过度承诺、serving “独立复现”是否
  真独立、RULER 伪影是否让 null 结论失效。

---

# Phase 1 — 独立审稿报告（摘要 + 分数）

## EIC Report Summary

- **Recommendation**: Major Revision | Confidence: 4 | 加权 74.5
- 维度：Originality 76 / Rigor 74 / Evidence 66 / Coherence 78 / Writing 84。
- 核心判断：论文首次把 08-10 whitelist 落到实体正文，容量 headline、
  GSM8K power 披露、harness 边界、serving 窄口径均兑现；语言润色后明显
  更接近投稿态。剩余问题不再是 claim 越界，而是“贡献定位”与“系统论文
  论证厚度”。
- 关键发现：
  - S1：容量模型 + block 粒度机制（Fig 5）构成自洽的 systems 证据链；
  - S2：8 图中每张都有独立证据角色，无纯装饰图；
  - W1（MAJOR）：标题 "Allocating Precision" 承诺了“分配”，正文实际只评估
    整 state 开关 fp32→bf16，且逐层分配被证否；标题应改为
    "Joint Precision Budgeting" 或明确“分配=整 state 精度决策”；
  - W2（MAJOR）：5.5 页正文对 MLSys 10 页上限而言偏薄，intro 没有显式
    contributions 列表，serving 协议与机制归因缺乏展开，投稿易被读为
    “短实证”而非完整系统研究；
  - W3：8 图 × 5 页正文的图文密度偏高，Fig 8 与 Fig 2 信息有重叠，建议
    Fig 8 移附录（不阻塞，用户如坚持 7+ 主图可保留）。

## R1 (Methodology) Report Summary

- **Recommendation**: Major Revision | Confidence: 4 | 加权 72.4
- 维度：Originality 74 / Rigor 70 / Evidence 62 (warn) / Coherence 82 /
  Writing 82。
- 核心判断：统计体系（配对 CI、9-seed、MDE/power、Bonferroni/BH-FDR、
  复现审计）仍是领域标杆级；但 serving 矩阵与 RULER 的选择性复测存在
  未处理的多重比较风险，论文对“独立复现”一词的操作定义不充分。
- 关键发现：
  - S1：GSM8K 同时报 p/CI/MDE/power，且明写“显著但效应低于 MDE”，符合
    08-10 裁决；
  - S2：chunk 消融 + stacking 成本图（Fig 7）把 harness 边界量化进正文；
  - W1（MAJOR）：serving 配对 Δ 只报 CI 不报 p，50+ 单元无任何多重比较
    校正；论文以“过载区 r40–r50”为结论窗口，属于事后选择。需补 p 值 +
    BH 校正，或明确把全部 serving 结论降级为“方向性”；
  - W2（MAJOR）：RULER 只复测“原单 seed 非零的 5 格”，3-seed CI 是条件于
    选择的；应报告全任务×长度网格的 3-seed 结果，或至少声明选择性并给出
    未选格的数据；
  - W3（MAJOR 边界）：复现 attempt 与原始矩阵同配置、同 seeds、同输出根
    （repro-final 文档），论文称 "independent reproduction" 但未定义
    “独立”的含义（新运行？新租机？新样本？）；要求正文补一句操作定义；
  - W4（MINOR）：capacity 探针每格单次运行，未声明确定性；block 计数是
    确定性分配，建议明写“单次探针 + 确定性分配器”；
  - W5（MINOR）：RULER/PPL 的 n=3 配对 t 自由度 2，CI 对单 seed 极敏感；
    已在图内保留逐 seed 结构，建议正文补一句对 n=3 的解释。

## R2 (Domain) Report Summary

- **Recommendation**: Major Revision | Confidence: 4 | 加权 73.7
- 维度：Originality 76 / Rigor 73 / Evidence 65 (warn) / Coherence 80 /
  Writing 80。
- 核心判断：容量模型、GDN-scope、ReplaySSM/PR 边界均正确；但发现一处
  **可验证的引文错误**（state-dtype 路径应引 PR #22196，而非 #43518），
  且 08-10 轮 R9（RULER 统一 no-think 口径）在正文仍未落实。
- 关键发现：
  - S1：附录容量模型推导 + Fig 5 block 证据把“保守下界”落到机制层；
  - W1（MAJOR，事实错误）：正文 "vLLM exposes a configurable state-dtype
    path [replayssm,vllmpr43518]"，但 `mamba_ssm_cache_dtype` 由
    vLLM PR #22196（danielafrimi，"Support FP32 SSM cache"）引入；PR #43518
    是 "FP8 SSM Cache Checkpointing (FlashInfer SSU)" 的 WIP。应改引
    #22196，并按需保留 #43518 描述 FP8 checkpoint；
  - W2（MAJOR）：RULER 协议为 `--max-tokens 256 --thinking default`
    （multiseed 分析 JSON 可查），正文只写 "3 dataset seeds" 未披露 think
    模式；08-10 R2 W4 已指出 FWE think 截断伪影，主表仍默认 think——
    要么改用 no-think 数据，要么在正文与图注显式披露并给出边界；
  - W3（MAJOR 建议）：state 精度 prior art 只引 ReplaySSM/PR#43518；需核对
    并显式处置近年 Mamba/SSM 量化工作（如 Quamba arXiv:2410.13211、
    MambaQuant arXiv:2504.13785），说明它们是否量化 serving 态，若否，
    应写明“这些工作针对权重/激活，未处理逐序列 state dtype”；
  - W4（MINOR）：RecurrentGemma 基于 Griffin（De et al. 2024），建议补引
    Griffin 以给出理论谱系；DeltaNet（Yang et al./Schlag et al.）与 GDN
    的谱系同理。

## R3 (Perspective) Report Summary

- **Recommendation**: Major Revision | Confidence: 4 | 加权 71.8
- 维度：Originality 74 / Rigor 70 / Evidence 62 (warn) / Coherence 80 /
  Writing 80。
- 核心判断：容量数字真实，但“生产意义”仍停在口号层：没有 cost/request 或
  请求/小时换算，TP/多卡被一句话带过，目标部署场景（短上下文高并发）没有
  显式写出。
- 关键发现：
  - S1：诚实披露负面/未复现结果在顶会中是稀缺资产；
  - W1（MAJOR）：容量 +38–41%（4K）→ +11–14%（16K）的衰减恰好说明
    “state 主导短上下文”是唯一强场景；intro/结论应把目标场景直接写成
    “短上下文、高并发、memory-bound serving”，否则读者会拿长上下文场景
    否定论文价值；
  - W2（MAJOR）：无成本换算。capacity token 应至少换算为
    “同一 GPU 预算下并发请求数/请求成本”的一行分析（用 18.63 MiB/请求
    和 int4 KV token 数即可），R3 认为这是 10 分钟可补的 P1；
  - W3（MAJOR 建议）：TP=2/4 时 state 与 KV 如何分片未讨论（正文只写
    “未测 TP”）；补一段机制推导（state 每 rank 分片 vs KV 每 rank 分片）
    即可把限制升级为洞见；
  - W4（MINOR）：替代杠杆（offloading、prefix caching、H2O/SnapKV 类
    驱逐）只有一句话定性；建议在 related work 或 discussion 给出
    “为何这些方案与 state 精度正交/互补”的 3–5 行论证；
  - W5（MINOR）：图 8 与图 2 冗余，建议移附录。

## DA (Devil's Advocate) Report Summary

### Strongest Counter-Argument

即便论文已按 whitelist 收窄，最强的反驳仍是：**这是一篇 flag-flip 论文**——
打开 vLLM 现成的 `mamba_ssm_cache_dtype bf16` 开关，套一个记账型容量公式，
报一组“诚实但多为 null/负面”的质量结果，serving 只有 Random60 过载区部分
可复现，ShareGPT 方向翻转。若审稿人认定“系统研究”需要**新机制或新决策
面**，则容量模型只是对已有开关的重新记账。论文当前的防御是“架构推导参数
独立于探针 + 保守下界 + 闭环验证”，这在方法论上成立，但 intro 没有把
“**不跑探针即可预测容量**”这一可操作知识放到最显眼位置，防御力被削弱。

### Issue List

#### CRITICAL

| # | 维度 | 问题 | 位置 | 处置建议 |
|---|---|---|---|---|
| C1 | 标题-结论匹配 | "Allocating Precision" 承诺分配算法，正文只评估整 state 开关且逐层分配被证否 | 标题、摘要 | 改为 "Joint Precision Budgeting for ..." 或明确“分配=整 state 精度决策” |
| C2 | 证据-结论匹配 | RULER 主表使用 think-default 协议且未披露；若 think 截断改变 FWE 分数，null 结论方向未知 | §3.2、Fig 3 | 换 no-think 或正文+图注显式披露协议与伪影边界 |
| C3 | 复现语义 | "independent reproduction" 与正式矩阵同配置/同 seeds/同输出根，未定义“独立” | §3.4、摘要 | 正文补操作定义；若仅为第二次运行，应写 "a second formal run" 而非 implied 独立样本 |
| C4 | 统计-结论匹配 | serving 50+ 单元无多重比较校正，论文选取 r40–r50 作结论窗口；n=3 的 CI 排除 0 只是 p<0.05，未跨单元控制 | §3.4、Fig 4 | 补 p + BH 校正，或把全部 serving 表述降为“方向性” |

#### MAJOR

| # | 维度 | 问题 |
|---|---|---|
| M1 | 贡献线 | intro 未正面回答“operator 从本文获得什么新知识”；建议显式写出“模型可在不探针时预测容量、block 取整给出修正项” |
| M2 | 证据边界 | capacity 每格单次探针未声明确定性；分配器确定性应写入 setup |
| M3 | 一般性 | 无 Mamba2/其他 GDN 模型探针；GDN-scope 已写，但标题与摘要的“hybrid”暗示仍需在 intro 显式收窄一次 |
| M4 | 替代解释 | serving 增益未量化“带宽 vs 容量 vs 页对齐”；正文已承认，但建议补一个可检验判别（如固定 block 数下的对照）或明确放弃归因 |

### Ignored Alternative Explanations/Paths
1. state bf16 同时改变 page size 与 block 数，goodput 收益可能主要来自分配
   粒度而非字节数；
2. RULER FWE 的 think 截断会让“无检测差异”同时在两个方向失真；
3. r45 的 TTFT 双峰 vs 全 ~18.5s 是环境敏感而非系统效应（论文已披露，
   但 “replicated” 一词仍需与“同一环境两次运行”区分）。

### Observations (Non-Defects)
- 08-10 三条 CRITICAL（serving headline、容量单一口径、单 seed pilot）在
  正文已实质解决；
- 语言润色（去 “not X but Y”）有效，正文无残留 AI 二元句式；
- 8 图中无一张是纯装饰，全部可溯源到原子 JSON。

---

# Phase 2 — 编辑综合

## Part 1: 共识与分歧

### Points of Agreement

- **[CONSENSUS-4] RULER think 协议必须披露或改用 no-think**：R2 给出协议
  证据，DA 升级为 C2，R1 从选择性复测角度佐证；正文目前零披露。
- **[CONSENSUS-4] serving 多重比较与“独立复现”定义必须补**：R1 主提，
  DA C3/C4 佐证，R3 认可“方向性”表述；无审稿人反对。
- **[CONSENSUS-3] 标题 "Allocating" 过度承诺**：EIC 与 DA 独立提出，
  R2 认可贡献是记账+整开关评估。
- **[CONSENSUS-3] 容量 headline 与 Fig 5 机制证据是论文最强资产**：
  EIC/R2/R3 一致；R1 认可其确定性可再声明。
- **[CONSENSUS-3] 正文偏薄（5.5 页），需补贡献列表与目标部署场景**：
  EIC/R3 主提，R2 认可 prior-art 处置后正文可再扩。

### Points of Disagreement

- **serving 结论的表述强度**：R1/DA 要求校正或降级；R3 认为
  formal+repro 双点 + 工作负载限定已足够“方向性”；EIC 裁决：
  **正文保留 paired goodput 点估计，但补 p 值与 BH 校正（或每格
  标注 “directional, n=3”），“replicated” 改为 “reproduced in a second
  formal run”**。
- **图 8 去留**：R3/EIC 认为与图 2 冗余，建议移附录；用户明确要求主图
  7+，裁决：**保留在主图，但图注补一句 “per-seed resolution of Fig. 2”**，
  不要求移动。
- **Quamba/MambaQuant 是否必须引**：R2 要求核对并显式处置；DA 认为若
  它们不量化 serving state 则一句话排除即可；裁决：**正文加一句显式
  边界，引用核验后决定**。

## Part 2: Editorial Decision Letter

Dear Author(s),

感谢提交《Allocating Precision Across Attention KV and Recurrent State in
Hybrid Linear-Attention Serving》。本轮为论文正文实体首审，5 席全齐。总体
判断：正文已兑现 08-10 whitelist 的绝大部分承诺，容量 headline、统计
披露、负面结果与复现限定均达到投稿级；语言润色与 8 图证据链有效。

### Decision: MAJOR REVISION

契约 F2 触发（多数审稿人在 mandatory 维度 warn），加权 71.8–74.5（均值
73.1）。与 08-10 研究汇总审相比，证据-正文匹配度显著提升；当前不可投的
原因集中在四处**可执行的披露/定位修正**，而非实验缺口。

### 核心裁决点

1. **RULER think 协议（C2）**：正文与图注必须披露 `--thinking default
   --max-tokens 256`，或直接改用 no-think 数据；否则 FWE 的 null 结论
   方向不确定。
2. **vLLM 引文（R2 W1）**：state-dtype 路径改引 PR #22196；PR #43518
   仅用于 FP8 checkpointing 语境。
3. **serving 统计与复现语义（C3/C4）**：补 p 值与 BH 校正（或每格
   “directional”标注）；“independent reproduction”改为
   “second formal run（same contracts and seeds）”并给出操作定义。
4. **标题与贡献定位（C1/M1）**：标题收窄为 budgeting/accounting，intro
   显式写出“无需探针即可预测容量”的可操作知识。
5. **正文补厚度（W2/W3）**：contributions 列表、目标场景（短上下文高
   并发）、cost/请求一行换算、TP 分片一段讨论。

## Part 3: Revision Roadmap

### Required Revisions (Must Fix)

| # | 修订项 | Source | Priority | 工作量 |
|---|---|---|---|---|
| R1 | RULER 正文+图注披露 think/max-tokens，或换 no-think 数据重跑 5 格 | R2/DA C2 | P0 | 0.5–1 天 |
| R2 | vLLM dtype 引文改 PR #22196；#43518 语义纠正 | R2 W1 | P0 | 1h |
| R3 | serving 补 p 值 + BH 校正（或逐格 directional 标注）；"independent reproduction" 改 "second formal run" 并定义 | R1/DA C3/C4 | P0 | 0.5 天 |
| R4 | 标题改为 Joint Precision Budgeting/Accounting；intro 补“免探针容量预测”与 contributions 列表 | EIC/DA C1/M1 | P0 | 0.5 天 |
| R5 | 正文补目标部署场景（短上下文高并发）+ cost/请求或并发数换算一行 | R3 W1/W2 | P1 | 2h |
| R6 | TP/多卡下 state 与 KV 分片的一段讨论 | R3 W3 | P1 | 2h |
| R7 | RULER 全网格 3-seed 报告或显式声明选择性复测 | R1 W2 | P1 | 0.5 天 |
| R8 | capacity 探针确定性声明（单次探针 + 确定性分配器） | R1 W4/DA M2 | P1 | 1h |
| R9 | state 精度 prior art 显式边界：核对 Quamba/MambaQuant 后一句排除或补引 | R2 W3 | P1 | 2h |
| R10 | 机制归因段补“带宽 vs 容量 vs 页对齐”的可检验判别或明确放弃归因 | DA M4/R3 | P1 | 0.5 天 |

### Suggested Revisions (Should Fix)

| # | 修订项 | Source | Priority |
|---|---|---|---|
| S1 | 补 Griffin（De et al. 2024）与 DeltaNet 谱系引用 | R2 W4 | P2 |
| S2 | Fig 8 图注注明 “per-seed resolution of Fig. 2” | EIC/R3 | P2 |
| S3 | 替代杠杆（offloading/prefix caching/驱逐）3–5 行正交性论证 | R3 W4 | P2 |
| S4 | 附录补 block 取整公式与 G 推导的数值例子 | R2 | P2 |
| S5 | GSM8K 补 effect size（Cohen's d）或说明原始分差+CI 即效应量 | R1 | P3 |
| S6 | 投稿前补 AI 使用披露、作者贡献与 artifact 链接（MLSys 政策） | EIC | P3 |

### 重审触发条件

R1–R10 完成后以本报告逐条 re-review（重点核 RULER 协议披露、PR 引文、
serving p/BH、标题与 intro），再进入下一轮 5 席审稿。

---

## 附录 A — 主编侧独立核验记录

| 核验项 | 结论 |
|---|---|
| 正文数字 vs 原子 JSON（容量/GSM8K/RULER/serving/chunk/敏感度） | ✅ 229 项账本全部一致（figures/verify_figure_data.py） |
| RULER 协议字段 | ✅ `--max-tokens 256 --thinking default`（multiseed JSON），正文未披露 |
| vLLM dtype 引文 | ⚠️ `mamba_ssm_cache_dtype` 源自 PR #22196；#43518 为 FP8 SSM Checkpointing WIP |
| serving 分析 JSON | ✅ 仅有 mean/CI，无 p 值；50+ 单元无校正 |
| 复现定义 | ⚠️ repro-final 文档：同配置、同 seeds、同输出根的第二次 attempt |
| 编译/图表 QA | ✅ 7 页、0 Overfull、8 图编号 1–8、字号 ≥5.8pt、引用全解析 |

## 附录 B — 审稿人状态与加权汇总

| 审稿人 | 推荐 | 加权 | Orig 20% | Rigor 25% | Evidence 25% | Coherence 15% | Writing 15% | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| EIC | Major Revision | 74.5 | 76 | 74 | 66 (warn) | 78 | 84 | warn |
| R1 | Major Revision | 72.4 | 74 | 70 | 62 (warn) | 82 | 82 | warn |
| R2 | Major Revision | 73.7 | 76 | 73 | 65 (warn) | 80 | 80 | warn |
| R3 | Major Revision | 71.8 | 74 | 70 | 62 (warn) | 80 | 80 | warn |
| DA | 不打分 | — | — | — | — | — | — | C1–C4 VALIDATED |
| 均值 | Major Revision | 73.1 | 75.0 | 71.8 | 63.8 | 80.0 | 81.5 | — |

**加权分与决策说明**：量规 65–79 名义映射 Minor；契约 F2（多数审稿人 ≥2
mandatory 维度 warn）优先级更高，触发 `major_revision`。本报告以契约为准。
