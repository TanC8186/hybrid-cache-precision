# MLSys 2026 最大强度审稿（第二轮）— Editorial Decision Package

> 审稿日期：2026-08-14（contract generated 2026-08-13T17:29:15Z）
> 审稿对象：`paper/mlsys2026/main.tex`（修订版，11 页 / 5,086 词 / 32 参考文献）
> 模式：ARS `academic-paper-reviewer` full + v3.6.2 sprint contract（盲预注册 → 看文评分 → 机械仲裁）
> 审稿人：EIC + R1 Methodology + R2 Domain + R3 Perspective + Devil's Advocate
> 本报告为只读审稿产物，未修改论文正文。

---

## 0. 契约机械层（v3.6.2 Layer 2，机器校验 pin）

### 0.1 评分矩阵（N=5）

| 维度 | EIC | R1 | R2 | R3 | DA |
|---|---|---|---|---|---|
| D1 methodology_rigor (mandatory) | warn | warn | pass | warn | warn |
| D2 domain_accuracy (mandatory) | pass | pass | pass | pass | pass |
| D3 argumentative_coherence (mandatory) | pass | pass | warn | pass | warn |
| D4 cross_disciplinary_relevance (high) | warn | pass | warn | warn | pass |
| D5 writing_and_structure (normal) | warn | pass | pass | pass | pass |

### 0.2 面板级 failure-condition 评估

| 条件 | 判定 | 计算 |
|---|---|---|
| F1（any 审稿人 mandatory 维度 block） | 未触发 | 5 人中无任何 mandatory block |
| F2（majority ≥3 审稿人有 ≥2 个 mandatory warn-or-worse） | 未触发 | 仅 DA（D1+D3=2 warn）= 1/5 < 3 |
| F3（any 审稿人 high-priority 维度 block） | 未触发 | D4 最高为 warn |
| F0（all 审稿人全部 mandatory pass） | 未触发 | 每人至少 1 个 mandatory warn |

### 0.3 机械层输出（供 check_panel_synthesis.py 校验）

fired_conditions: []
editorial_decision=accept

**说明**：上两行为契约算术的机械结果——没有任何面板级 failure condition 触发，故契约动作值落入 accept 档。**这不是最终编辑决定**：按本技能 Checkpoint Rule #4（DA-CRITICAL 发现 → 决定不得为 Accept）与决策矩阵保守原则，最终决定见 §1。

---

## 1. 最终编辑决定

## Decision: Major Revision（resubmission encouraged）

### 1.1 双层决定依据

- **契约机械层 = accept**：面板级 F1–F3 全部未触发（见 §0）。
- **技能铁律层 = 不得 Accept**：Devil's Advocate 提交了 CRITICAL 发现 C1（标题与贡献 (b) 的 "joint precision budgeting" 从未被任何 joint-vs-independent 对照检验，且论文自身 stacking 证据支持两维度近似可分）。按 Checkpoint Rule #4，DA-CRITICAL 存在时决定不得为 Accept。
- **决策矩阵层**：4 名审稿人建议 Accept（附 minor 修订），DA 建议 major_revision 并给出 1 CRITICAL + 5 MAJOR；按 "split decision → 保守" 与 "outlier 理由有效且被他人部分遗漏 → 升级 Major" 两条标准，落点应为 Major Revision。
- **可行性**：全部阻断项均可通过新增对照实验或收敛标题/定位解决，不属于不可修复缺陷；修订后需要复审（re-review）。

### 1.2 DA-CRITICAL 处理块（技能强制）

**DA 的论证（C1）**：标题与贡献 (b) 主张 "joint precision budgeting"，但全文不存在 joint 与 independent/sequential 分配的任何对照实验；selector 只是对 ≤4 个候选配置的约束过滤，从未与"先选 KV dtype、再选 state dtype"的独立顺序策略或任何静态规则比较。更严重的是论文自己的 stacking 证据（bf16 state 叠加 int4 KV 无额外 PPL/GSM8K 代价）支持两维度近似可分——若两维独立，"joint" 选择器退化为两个独立开关的组合，核心贡献不成立。论文从未正面检验其自身数据提出的"jointness 是否必要"这一问题。

**其他审稿人的佐证**：R2（Domain）独立提出同一缺口——"标题级承诺（joint 选择的增益）未被直接验证"，并将 D3 打为 warn，建议"把标题承诺收敛为 joint accounting，或补一个 independent-budgeting 对照设计"；EIC 的 D1 warn 部分同源（selector 无任何对照策略）；R3 同样指出 selector 无 baseline 比较。

**EIC 仲裁评估**：缺口真实存在且位置关键（它位于标题与贡献 (b) 上，而非论文已适当限定的结论上）。按 DA 自己的四条 CRITICAL 标准衡量，其严重性处于 CRITICAL/MAJOR 边界——论文的显式结论（"verified capacity effect and scoped policy execution"）与证据范围是匹配的，被击穿的是标题级承诺而非结论本身；DA 在 D3 打分理由中也写明"可指名的是缺失实验，而非被证伪的结论"。综合判定：**成立为必须修复的阻断性缺口，按 CRITICAL 处理（决定约束）但修订路径为可执行的对照实验或定位收敛**（见路线图 R1/R7）。

**要求作者响应**（即使不同意 DA 也必须逐点回应）：(a) 说明为何 "joint" 框架对本文必要，或接受定位收敛；(b) 若坚持 joint 框架，给出 joint-vs-independent 对照的实验设计或已运行数据；(c) 对 stacking 可分性证据与 jointness 主张之间的张力给出正面处理。

### 1.3 其余 DA MAJOR 发现（决定依据的一部分，均须回应）

| DA 编号 | 内容（摘要） | 佐证 | 处理要求 |
|---|---|---|---|
| M1 | RULER 臂 30/30 sample-level 精确全等更相容于"state-dtype 开关在该臂未改变执行计算"（no-op 假说）；质量臂未报告干预生效验证，与容量臂 "records the resolved dtype" 的纪律不对称 | 无（新发现，置信高） | 补质量臂 resolved dtype / 状态张量生效证据（路线图 R2） |
| M2 | "52/52 方向一致" 是 G 减半下分母单调性的算术必然，不携带经验信息；真实内容是幅度（15.44%）与残差，且幅度由 vLLM 块粒度舍入主导 | EIC 部分佐证（"该效应接近算术必然"） | 重构 headline 框架为单调性预测 + 测量幅度（路线图 R3） |
| M3 | selector 对 4 点网格求 max 不构成优化贡献；无对照策略，"三个预算映射只验证了 plumbing" | EIC W3、R3 W4 部分佐证 | selector baseline 或定位收敛（路线图 R7） |
| M4 | 用"效应低于预注册 MDE"对已显著的结果作解释性降权属于功效误用（Hoenig & Heisey 2001）；operational rule 未按模型规模条件化，向恰好遭受 −1.0pt 显著回归的 2B 用户无条件推荐该开关 | 无（新发现，有文献锚定） | 删除 MDE 降权表述；operational rule 条件化（路线图 R4） |
| M5 | 全部系统级正面主张未通过自己的门限（0/60 BH-FDR、Gate 4 失败、TTFT 10/18、controller 单 slice），贡献重量全压在接近定义性成立的 capacity 上 | EIC W1、R3 W1 佐证 | serving 章节重新定位 + 补齐证据（路线图 S2/S3） |

---

## 2. 共识分析

### 2.1 全体一致（5/5，含 DA）

- 论文的方法学纪律与披露水平是 serving systems 领域罕见的高标准：预注册门禁、配对设计、MDE/power 报告、BH-FDR/Holm/Bonferroni 校正、run-stability 与 replication 的区分、门禁失败如实保留（183/720、79.61% 进 abstract）、零宽 CI 三处声明非等价性。数字与 `results/` 档案逐位吻合（多位审稿人独立核验）。
- 核心容量效应被双次 formal attempts 验证（52/52 方向、15.44% 中位增益、1.81% 中位绝对残差、最大 per-cell 差异 1.42%），容量模型与 vLLM GDN 路径逐项对应（R2 数值重算吻合）。
- 上一轮的三处硬伤（N(L)/T(L) 符号、lower-bound 方向、共享内存矛盾）已全部修复，且修复质量经独立核验。

### 2.2 弱点子主张清单（Step 1b，按 sub-claim 分解）

| sub_claim_id | 内容 | 非-DA 审稿人立场（agree/4） | DA 追踪 |
|---|---|---|---|
| SC-1 | serving 收益未建立（0/60 BH-FDR、Gate 4 失败、TTFT 仅 throughput 复现） | EIC 提出 + R3 提出 = 2/4 佐证 | M5 |
| SC-2 | 无第三方 baseline（state 压缩/调度器）head-to-head | EIC 提出 = 1/4 | M3 |
| SC-3 | jointness 从未被检验（joint-vs-independent 对照缺失；stacking 可分性张力） | R2 提出 = 1/4 | **C1 (CRITICAL)** |
| SC-4 | RULER 臂干预生效未验证（no-op 假说） | 0/4 | M1 |
| SC-5 | "52/52" 是算术必然，headline 框架错置 | EIC 实质佐证 = 1/4 | M2 |
| SC-6 | MDE 事后降权属功效误用 | 0/4 | M4 |
| SC-7 | 9.32 MiB 应为 9.63 MiB（conv state 恒 bf16） | R2 提出（数值验证）= 1/4 | — |
| SC-8 | M3/serving 第二 run 分析产物未入 results/，"full release" 不实 | EIC + R1 = 2/4 佐证 | — |
| SC-9 | Table 1 attempt 来源未注明；fp16/bf16 标签出入 | R1 提出 = 1/4 | — |
| SC-10 | selector 技术深度有限（过滤+argmax、cold restart、无在线自适应） | EIC + R3 = 2/4 佐证 | M3 |
| SC-11 | 质量证据分辨率低（RULER 5cell×3seed×20、GSM8K 低于 MDE、PPL 近似） | EIC 提出 = 1/4 | — |
| SC-12 | 无完整 operating curves 与部署经济学（cost/request、GPU 数换算、冷重启成本） | R3 提出 = 1/4 | M5 |
| SC-13 | 附录缺 padded block accounting 与参数表（A_f/A_q/G） | R2 提出 = 1/4 | — |
| SC-14a | FWE/NIAH-multiquery 等术语未定义 | EIC 提出 = 1/4 | — |
| SC-14b | state 张量概念（temporal vs conv、chunked scan）未铺垫 | R2 提出 = 1/4 | — |
| SC-15 | serving 不稳定的机制假设未讨论（对同协议 2B/4K 结论的外部效度） | R2 提出 = 1/4 | — |

**标签汇总**：无 [CONSENSUS-4] / [CONSENSUS-3]；佐证发现（2/4）：SC-1、SC-8、SC-10；其余为单审稿人发现（按 Confidence Score 加权入路线图）。无 SPLIT（EIC 的 D3=pass 与 R2 的 D3=warn 针对不同证据面——EIC 评"无循环论证"，R2 评"缺 head-to-head 对照"，非互相否定）。

### 2.3 分歧

无硬性分歧。一处需记录的面板内部张力：DA 的 issue list 将 C1 标为 CRITICAL，而其 D3 打分理由明确"可指名的是缺失实验，而非被证伪的结论"（warn 级）。仲裁见 §1.2——按 CRITICAL 处理决定约束，按可修复缺口处理路线图。

---

## 3. 决定理由（Decision Rationale）

本轮的机械评分在契约层落入 accept 档，但这恰是 DA-CRITICAL 铁律存在的原因：四位审稿人（包括 EIC 本人）都把注意力放在论文"已适当限定"的结论上，而 DA 攻击的是结论之上的标题与贡献框架——"Joint Precision Budgeting" 作为一个**系统贡献**从未被检验：没有任何实验把 joint 分配与逐组件独立分配放在一起比较，而论文自己的 stacking 数据（无交互代价）反而支持两维可分。这一缺口被 R2 独立佐证（D3=warn），并构成上轮审稿"身份-证据错位"问题在系统路线上的残留形态。同时 DA 的两个新发现（M1 no-op 假说、M4 功效误用）单独均可击穿相应主张：RULER 的 30/30 精确全等与"开关未生效"完全相容，MDE 修辞对已显著回归的解释性降权在统计上不成立。综上，论文的可验证贡献（容量效应 + 协议纪律）真实且值得肯定，但标题级承诺、RULER 质量证据与 operational rule 的表述需要实质修订，部分需要新增实验；修订完成前不满足 MLSys 主轨录用标准。**Major Revision，6–8 周，修订后复审。**

---

## 4. 关键问题汇总（按严重度）

1. **[DA-C1 / SC-3]** jointness 从未被检验：标题与贡献 (b) 的框架未被任何对照实验支撑，且被自身 stacking 证据削弱。
2. **[DA-M1 / SC-4]** RULER 臂干预生效未验证：30/30 精确全等与 no-op 假说相容，质量证据可能为零信息量。
3. **[DA-M2 / SC-5]** headline 框架错置：52/52 方向是算术必然，"发现"的表述必须改为单调性预测 + 测量幅度。
4. **[DA-M4 / SC-6]** MDE 功效误用 + operational rule 未按模型规模条件化（−1.0pt 显著回归被修辞性降权）。
5. **[SC-8]** 可复现承诺不实：M3 与 serving 第二 run 产物未归档，与 "full artifacts released" 矛盾。
6. **[SC-1/SC-12]** serving 证据无正面效应且无完整 operating curves/部署经济学——serving 章节需重新定位为测量方法学案例。
7. **[SC-7]** 9.32 MiB 事实性错误（应 9.63 MiB，conv state 不参与减半）。
8. **[SC-2/SC-10]** baseline 缺失与 selector 定位：无 state 压缩 head-to-head、无 selector 对照策略。

---

## 5. 修订路线图（Revision Roadmap）

### Required Revisions（Must Fix，P1）

| # | 修订项 | Sub-Claim(s) | 来源 | 优先级 | 预估工作量 |
|---|---|---|---|---|---|
| R1 | jointness 二选一：(a) 在 2B/4K slice 至少补 joint vs independent/sequential-budget 对照 + 静态规则 baseline；或 (b) 标题与贡献 (b) 收敛为 "joint accounting / characterization"，selector 定位为可执行策略层 | SC-3, SC-2 | DA-C1, R2 | P1 | (a) 2–3 周 / (b) 1 周 |
| R2 | RULER 质量臂干预生效验证：记录 resolved state dtype / 状态张量抽查，正面回应 no-op 假说；若假说成立需重跑或撤下该证据 | SC-4 | DA-M1 | P1 | 3–5 天 |
| R3 | 重构 headline：52/52 表述为模型单调性预测（G 减半下 r_state>1 必然成立），headline 改为测量幅度与残差分布；全文核对 | SC-5 | DA-M2, EIC | P1 | 2–3 天 |
| R4 | 删除对 2B GSM8K 显著回归的 MDE 事后降权表述（锚定 Hoenig & Heisey 2001）；operational rule 按模型规模/任务条件化，正面处理 2B/9B 差异 | SC-6 | DA-M4 | P1 | 1 周 |
| R5 | M3 与 serving 第二 run 的 analysis/gate4 JSON 入 results/；controller 审计标注 logical-only；修正 "full release" 表述或补全归档 | SC-8 | EIC, R1 | P1 | 1–2 天 |
| R6 | 修正 9.32→9.63 MiB 与 conv-state bf16 说明；附录补参数表（A_f/A_q/G × 2B/9B）与 padded block accounting | SC-7, SC-13 | R2 | P1 | 2 天 |
| R7 | selector 对照：静态规则（state 恒 bf16 + KV 按质量选）与 sequential-independent 策略；或将贡献 (b) 定位收敛（与 R1 联动） | SC-10 | DA-M3, EIC, R3 | P1 | 1–2 周 |

### Suggested Revisions（Should Fix，P2）

| # | 修订项 | Sub-Claim(s) | 来源 | 优先级 | 预估工作量 |
|---|---|---|---|---|---|
| S1 | 至少一个 state 压缩替代方案（ReplaySSM 式重算或 fp16-state）的容量/质量 head-to-head cell；补引 Nemotron-H、WKVQuant | SC-2 | EIC, R2 | P2 | 1–2 周 |
| S2 | 用现有 per-seed 数据画完整 operating curves（offered load → P50/P95/P99 TTFT/TPOT/goodput）；补部署经济学 worked example（cost/request 或 requests/GPU-hour）与冷重启成本 | SC-12 | R3, DA-M5 | P2 | 1 周 |
| S3 | serving 章节重新定位为测量方法学/noise-floor 案例；tab:serving caption 加 "single-cell detections, not effects" | SC-1 | EIC, R1 | P2 | 3 天 |
| S4 | Table 1 注明 attempt 来源；fp16/bf16 state 字节等价说明；15.44% 的 per-L 分解 | SC-9 | R1, EIC | P2 | 2 天 |
| S5 | 定义 FWE/NIAH-multiquery；state 张量形状与 temporal/conv state、chunked scan 半页铺垫；TP 推导移附录 | SC-14a, SC-14b | EIC, R2 | P2 | 2 天 |
| S6 | serving 不稳定机制假设讨论（engine commit / kernel 非确定性 / 协议）及其对 2B/4K 方向性证据的外部效度 | SC-15 | R2 | P2 | 2 天 |

### Priority 3 — 文本与格式（Nice to Fix）

- [ ] DA-m1：Sec 5.1 "7.0--15.4% at 4K" 指代错误（7.0% 为 16K 值）；DA-m2：Fig 1b caption "7 cells" 与 "int4 column" 表述冲突；DA-m3：abstract 补 112 = 52×2+8 构成。
- [ ] EIC minors：R2-to-R3 命名定义；vllmpr43518 是 checkpointing 非 compression；\cite{replayssm,vllmpr22196} 配对欠精确；M3 的 TTFT/TPOT Gate-4 容差补全；两个 commit 的 package 对应关系标注。
- [ ] R2 minors：tokens/block（2064/1072 等）语义定义；Fig 5 caption "halves"→"roughly halves"；ARKV/KVQuant/ReplaySSM 三处 prior art 措辞。
- [ ] R3 minors：Table 1 gap 符号约定；Fig 4 附绝对量小图；abstract 加一句操作化总结；1.42% vs 1.41% 引用核对；RTX 5090 功耗设置注明。
- [ ] R1 minors：统计报告口径在 Sec 4 开头统一约定；RULER screen 的归档指针；controller 三预算请求参数公开。

### 修订期限

**Major Revision：建议 6–8 周**；响应信逐条回复（R1–R7 为必答项，DA-CRITICAL C1 必须逐点回应），修订后进入 re-review。

---

## 6. Review Panel Provenance (#540)

单模型家族披露：本轮五席审稿人（EIC/R1/R2/R3/DA）全部运行于同一模型家族（会话继承模型），未启用 cross-model track（无 `ARS_CROSS_MODEL` 配置，且未取得将稿件上传外部提供商的用户同意）。据此披露相关误差提示：五份独立报告共享同一模型的潜在系统性偏误，panel 独立性来自协议层面的角色/上下文隔离与盲预注册，而非模型多样性（Ren et al. 2026, arXiv:2607.13104 §5.2）。审稿人对论文的声明性数字均做了本地 `results/`、`vendor/vllm` 代码或外部文献的独立交叉核验，作为对同模型偏误的实质性补偿。

---

## 7. Reviewer Report Summary（Appendix）

### EIC 摘要
- Recommendation: Accept（建议 minor）| Confidence: 3/5 | 契约决定: accept
- Key Point: 身份-证据匹配成立、披露纪律罕见；D1/D4/D5 warn 来自基线缺口、术语未定义与 provenance 缺口；机械层无条件触发。

### R1（Methodology）摘要
- Recommendation: Accept（建议 minor）| Confidence: 4/5 | 契约决定: accept
- Key Point: 方法学证据链属领域罕见高标准，数字与归档逐位吻合；唯一 warn（D1）为 M3 归档缺失与 Table 1 provenance；明确"第二 run = run-stability 非独立样本"处理正确。

### R2（Domain）摘要
- Recommendation: Accept（建议 minor）| Confidence: 4/5 | 契约决定: accept
- Key Point: 容量模型与 vLLM GDN 路径逐项吻合（重算 Table 1 全部 int4 gap 一致）；D3 warn = joint-vs-independent 对照缺失；发现 9.32 MiB 事实性错误与附录参数表缺失。

### R3（Perspective）摘要
- Recommendation: Weak Accept | Confidence: 4/5 | 契约决定: accept
- Key Point: 部署侧可直接采信数字；D1/D4 warn = 无完整 operating curve、无部署经济学、切换成本未量化；建议用现有数据补曲线与 worked example。

### Devil's Advocate 摘要
- Recommendation: Major Revision | Confidence: 未评分（只挑战）| 契约决定: major_revision（个人 F2 触发）
- Key Point: 1 CRITICAL（C1 jointness 未检验）+ 5 MAJOR（M1 no-op 假说、M2 算术必然性、M3 selector plumbing、M4 功效误用、M5 贡献重量）；最强反方论证：剩余可验证内容"接近定义性成立"，作为系统贡献不足以支撑主轨。

---

## 8. 协议轨迹（Protocol Trail）

1. **Phase 0**：领域分析 + 5 人配置卡，用户确认"最大强度"。
2. **Phase 1（盲预注册）**：5 审稿人只看合同+元数据，各自预注册 D1–D5 评分触发词；5/5 通过结构 lint（paraphrase ×5 + scoring plan ×5 + `[CONTRACT-ACKNOWLEDGED]`）。
3. **Phase 2（看文评分）**：5 审稿人读取论文 + 各自 Phase 1 + 独立核验（results/ 档案、vendor/vllm 代码、外部文献）；5/5 输出通过 lint。机器语法规范化：R1 与 DA 的 Failure-Condition 小节标题规范为精确 condition id（内容未改动，已记录）。
4. **Layer 1 自洽**：5/5 审稿人各自 scores → fired → decision 链一致（DA 个人 F2 触发 → major_revision；其余 4 人无触发 → accept）。
5. **Layer 2 面板机械仲裁**：F1/F2/F3/F0 全部未触发 → 契约动作值 accept（§0.3 为机器 pin）。
6. **Checkpoint Rule #4 覆盖**：DA-CRITICAL C1 存在 → 最终决定不得为 Accept → **Major Revision**（§1）。
7. 校验脚本运行结果：`check_sprint_contract.py` → **OK（Schema 13.1）**；`check_panel_synthesis.py` → **PANEL-SYNTHESIS: PASS（exit 0）**——5 份审稿报告解析通过、Layer 1（scores→fired→decision）逐人自洽、Layer 2（面板量词→仲裁→pin）与合成输出一致。
