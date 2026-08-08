# ARS 最大强度审稿报告（2026-08-09）— 精度预算分配方向

> 模式：`academic-paper-reviewer` **full**（EIC + R1 Methodology + R2 Domain + R3 Perspective + DA，5/5 全齐）
> 审稿对象：`docs/paper/research-summary-2026-08-09.md`（KV bits × state bits 二维精度预算方向）+ 全证据库
> 审稿输入：research-summary、claim-evidence-map、实验矩阵计划、`results/verified/2026-08-08/{capacity-state,ssm_dtype}/`、`results/verified/2026-08-09/statebf16-random60-pilot-20260809/`、`results/quality/{ppl-state-dtype,state-sensitivity,ruler-subset,reasoning}/` 全部分析 JSON、关键脚本（probe_ssm_state_dtype.py、analyze_capacity_state.py、hybrid_premise.py、run_state_sensitivity.py、analyze_statebf16_pilot.py 等）、pilot 配置（sha256 f7475580…）、vendor/vllm-patches/*.diff、CLAUDE.md 硬规则
> 契约：`reviewer/reviewer_full/v1`（panel_size=5，D1–D5，F0–F3）

## 0. 执行边界与诚实声明

1. **Sprint Contract 两调用协议未物理执行**：Phase 1（paper-blind 预承诺）与 Phase 2（paper-visible）在本内联会话无法真正调用隔离；审稿人 prompt 要求先做"预注册评估标准"再进入证据审查，契约作为评分判据与决策算术使用，不宣称完成盲审预注册。
2. **R3 (Perspective) 延迟提交**：该席位审稿人在综合完成后才提交报告（经 4 次催促）；主编已将其完整纳入本最终版综合（5/5 全齐）。F 判定基于 4 位打分审稿人（EIC/R1/R2/R3）。
3. **DA 角色不打分**：按 `devils_advocate_reviewer_agent.md` 硬边界，DA 不产出维度分；F0–F3 机械判定基于 4 位打分审稿人（EIC/R1/R2/R3），DA-CRITICAL 按 IRON RULE 4 逐条裁决。
4. **未启用跨模型**（`ARS_CROSS_MODEL` 未设置），本报告为单模型面板。
5. **只读审稿**：未修改任何论文草稿/实验文件；本报告为独立文档。

---

# Phase 0 — 领域分析与审稿人配置

## 1. 六维分析

| 维度 | 结果 |
|---|---|
| 主学科 | 计算机系统 / ML Systems（LLM serving 系统） |
| 次学科 | KV cache 量化与压缩、循环 state 精度、混合线性注意力架构、容量建模、serving 性能评测 |
| 研究范式 | 定量实证系统研究（受控对比实验） |
| 方法类型 | 系统测量 + 解析建模 + 质量/服务闭环评测 |
| 目标期刊层级 | Q1 顶会：MLSys 2026/2027 Research Track 首选；OSDI/ASPLOS（若机制部分扩展）；NeurIPS 系统方向次选 |
| 论文成熟度 | 预提交前研究总结：证据链约 2/3 完成，论文未重写 |

## 2. Reviewer Configuration Cards（5 席，R3 缺席）

| # | 角色 | 身份 | 关注 | 状态 |
|---|---|---|---|---|
| 1 | EIC (Journal-Fit) | MLSys Area Chair，LLM serving 系统软件 | 二维预算定位原创性、贡献强度、文本-证据一致性 | ✅ 已提交 |
| 2 | R1 (Methodology) | Serving 评测方法论 + 统计报告专家 | 3-seed/配对 CI、pilot vs formal、容量模型误差、复现门禁 | ✅ 已提交 |
| 3 | R2 (Domain) | 线性注意力/SSM cache 量化专家 | 文献边界、容量模型正确性、质量证据链、claim 红线 | ✅ 已提交 |
| 4 | R3 (Perspective) | 内存系统/数据中心推理基础设施研究者 | 二维预算生产意义、替代路径、经济性 | ✅ 延迟提交（已纳入） |
| 5 | DA (Devil's Advocate) | 对抗式研究者 | 最强反驳、cherry-picking、pilot 过度解读、诚实性 | ✅ 已提交 |

---

# Phase 1 — 独立审稿报告（摘要）

## EIC Report Summary

- **Recommendation**: Major Revision | Confidence: 4 | 加权 66.0
- **核心判断**：方向有顶会潜力（二维精度预算框架原创性 78 分），实验 provenance 纪律为领域标杆；但"闭环验证"标题承诺被三个阻塞缺口（M-2x2 容量、Q-stacking 质量叠加、S-formal serving）拖累，Evidence Sufficiency 判 **block (repairable)**。
- **关键发现**：S1 框架原创性成立；S2 容量模型跨规模验证质量罕见（10 个探针 JSON 全部交叉核对一致）；S3 provenance 纪律 MLSys 标杆；W1 三缺实验阻塞（CRITICAL）；W2 claim #5 仅单 seed pilot；W3 GSM8K 9B 零宽 CI；W4 harness vs kernel gap。

## R1 (Methodology) Report Summary

- **Recommendation**: Major Revision | Confidence: 4 | 加权 66.8
- **核心判断**：证据架构（四层闭环）与复现基础设施超过 MLSys 常规水准；统计层面两处可修复缺口（模型系统性负偏差未分析、敏感度门缺多重比较校正），证据链三组阻塞实验待补。
- **关键发现**：F0 统计算术全部 VERIFIED（GSM8K CI [-0.0338,-0.0195] 重算一致）；F1 配置审计全部 PASS（runtime dtype 非 CLI 回显）；F3 复现审计 PASS；W1 容量模型误差全负（P=0.0625 同号概率）未讨论根因；W2 敏感度 2/36 CI 不含 0 恰为多重比较预期假阳性率；W3 三缺实验阻塞（约 8-9h 计算量）。

## R2 (Domain) Report Summary

- **Recommendation**: Major Revision | Confidence: 4 | 加权 69.8
- **核心判断**：方向文献差异化成立（claim 1 pass），容量模型推导与 vLLM 代码链路逐行吻合，实验卫生上乘；四个 MAJOR 需投稿前解决。
- **关键发现**：F0-F3 全部 PASS（无 fatal、红线全部遵守、诚实披露完整）；W1 GSM8K 2B vs 9B 不对称回退（-2.67pt vs -0.5pt）缺机制解释；W2 文献偏窄（仅 vLLM PR 生态，缺 Mamba2/FlashInfer SSU/SSM state 理论）；W3 harness chunk 模拟 vs per-token 语义差异未量化；W4 RULER 非零差异格单 seed 归因缺统计支撑。

## R3 (Perspective) Report Summary

- **Recommendation**: Major Revision | Confidence: 4 | 加权 70.4
- **核心判断**：设计规则"省 1 字节 KV ≈ 省 L 字节 state"（S1）与 serving 边界区救援叙事（S2）有真实部署价值，诚实性基础设施超规范（S4）；但"第二维度"框架与 KV int4 的幅度不对称（+38~41% vs +124.5%）需更强论证（W1），bf16 下界未探索（W2），单架构限制一般性声明（W3）。
- **关键发现**：W1 [Major] 边际价值论证不足（1.3755 vs 2.245x）；W2 [Major] fp8/int8 未探索、fp16 state 质量未测；W3 [Major] 仅 Qwen3.5 单架构，建议跨架构探针或显式 scope 到 GDN；W4-W6 [Minor] TP 经济性、替代容量杠杆（驱逐/卸载）未对比、短/长上下文交叉点未量化。
- **额外提出**：train-inference mismatch 假说（Qwen3.5 以 fp32 state 训练，推理期降 bf16 的 GSM8K 回退可能源于训练-推理精度失配而非精度损失本身）；成本模型扩展建议（成本/请求 而非仅容量）。

## DA (Devil's Advocate) Report Summary

- **定位**：只挑战不打分 | CRITICAL 3 项 + MAJOR 6 项 + MINOR 4 项
- **最强反驳**：核心贡献可归结为"翻转 vLLM 已有 `--mamba-ssm-cache-dtype` 开关并测量"——容量模型是记账恒等式而非预测理论，闭环验证不完整（pilot 单 seed、GSM8K 真实回退、2×2 联合预算实验待补）。审稿人可能判定"测量 + 模型 + flag-flip"低于顶会贡献线。
- **CRITICAL**：C1 9B GSM8K 零宽 CI（3 seed 全相同 → CI 计算退化，n=200 二项抽样下 3 seed 完全一致的概率极低，暗示种子非独立）；C2 claim #5 仅单 seed pilot 支撑（r40 边界区收益是 3 个 rate 中唯一点）；C3 容量模型误差全负 = 结构性保守偏差未被讨论。
- **替代解释**：pilot 收益可能来自内存带宽而非容量（未隔离）；容量模型是 tautology（A_q/G 来自同一 vLLM 记账框架）；GSM8K 回退是 PPL 掩盖的真实精度效应；2/36 敏感 CI 符号一致性（均为正）弱提示真实微小效应。

---

# Phase 2 — Editorial Synthesis

## Part 1: 共识识别与分歧仲裁

### Points of Agreement (Consensus)

- **[CONSENSUS-5] 三组阻塞实验（M-2x2 容量、Q-stacking 质量叠加、S-formal serving）是投稿前提**：EIC W1 (CRITICAL)、R1 W3 (Major)、R2（warn）、R3 W1（2x2 完成后"第二维度"框架才成立）、DA（stacking/2x2 缺失）。无一审稿人认为当前证据底座可投稿。
- **[CONSENSUS-5] harness chunk 级模拟 vs kernel per-token 语义的差异必须量化**：EIC W4 (Major)、R2 W3 (Major)、R1 W5 (Minor)、DA m3。chunk=1 消融即可化解。
- **[CONSENSUS-3] fp16 下界质量缺失削弱"精度预算谱系"完整性**：R3 W2 (Major)、R2 W6 (Minor)、DA M6（fp8/int8 未探索）。fp16 state 质量 smoke（~5 min）即可补一个精度点。
- **[CONSENSUS-3] 容量模型系统性负偏差需作为偏差讨论而非仅报误差幅度**：DA C3 (CRITICAL)、R1 W1 (Major)、R2（残差识别为 page 对齐但未展开）。4 点误差全负（P=0.0625 同号）不能当随机噪声处理。
- **[CONSENSUS-3] GSM8K 证据链需修复**：DA C1（9B 零宽 CI）、EIC W3 (Major)、R2 W1（不对称缺机制）。2B 的 -2.67pt CI 计算正确，但 9B 的零宽 CI 不可信。
- **[CONSENSUS-3] claim #5（serving SLO 收益）目前只能算方向性证据**：EIC W2 (Major)、DA C2 (CRITICAL)、R2（partially supported）。S-formal 完成前不得作 headline。

### Points of Disagreement

- **决策严格度**：EIC 判 Evidence Sufficiency **block**（F1 触发候选），R1/R2/R3 均 warn（R3 Evidence 58、R2 60、R1 55）。**Editor's Resolution**：EIC 的 block 标注为 *repairable*（三缺实验 1-2 天内可补齐），其余三位无 block；且与前轮（08-06）的 REJECT 不同——本轮不存在"文本与自身已验证证据矛盾"（+25% 事件）这类投稿不可接受缺陷，全部问题为**证据链未完成 + 统计细节可修复**。裁定：**Major Revision（拒稿但鼓励修复后重投）**，不等价于 Reject。
- **"第二维度"的边际价值**：R3 W1 质疑 bf16 state（+38~41%）与 KV int4（+124.5%）幅度不对称下"并列维度"的合理性；EIC/R1/R2 认可框架原创性（72-78 分）但同样要求 stacking 证明复合增益。**Editor's Resolution**：采纳 R3——论文必须把 bf16 state 定位为"与 KV 量化正交、可复合的维度"（2x2 表 + stacking 质量即为此证据），而非与 KV 量化并列比较幅度；设计规则 ∂C/∂A vs ∂C/∂G 的互补性（短上下文 state 主导、长上下文 KV 主导）是论证核心。
- **容量模型负偏差的定性**：DA 认为可能反映"模型是记账恒等式"（tautology），R1 认为需 block 粒度解释，R2 认为残差即 page 对齐。**Editor's Resolution**：三者可统一——模型参数 A_q/G 来自架构推导 + vLLM 记账（§2.3 与 analyze_capacity_state.py 一致），非从实测 fit 出；负偏差方向（模型高估收益）恰为 vLLM 离散 block 分配所致（fp32 block 2064 vs bf16 block 1072 的 granularity 差异），**模型应重新定位为"保守下界"**，这是可辩护的实用属性而非缺陷。DA 的 tautology 指控不成立（参数独立于被预测的容量比），但需要在论文中展示 A_q/G 的架构推导链（EIC W5）。
- **9B GSM8K CI 的处理**：DA 认为零宽 CI 暗示种子非独立（可能为 bug），EIC 建议核查种子传递链。**Editor's Resolution**：零宽 CI 的直接原因是 3 个 seed 的 fp32 精度恰为 0.885、bf16 恰为 0.88——greedy 解码 + 固定 200 样本下，seed 只影响序列采样顺序而不改变最终准确度集合，故 3 seed 输出可能确定性地相同；**需核查 reasoning_bench.py 的 seed 语义**（若 seed 仅控制采样顺序而 greedy 解码输出与顺序无关，则"3 seeds"实为同一结果重复 3 次，CI 无意义，9B GSM8K 应改写为"确定性差异 -0.5pt 或补跑有真实随机性的协议"）。这是本轮最高优先级待查项之一。
- **RULER 非零格定性**：R2 认为单 seed 无法排除真实效应（建议补 2 seed），DA 认为伪影解释可信。**Editor's Resolution**：采纳 R2——对非零差异格（2B FWE L4096/L8192、9B niah_multiquery L4096/L8192、FWE L8192）补跑 2 seed 是低成本高信息量动作；补跑前 claim 3 的"RULER 基本持平"措辞保持"点估计 + 单 seed"标注。
- **敏感度门多重比较**：R1 认为需 Bonferroni 校正（alpha/36=0.0014），DA 提示 2/36 的符号一致性（均为正）弱提示真实微小效应。**Editor's Resolution**：两者可调和——Bonferroni 后 2/36 均不显著（形式化支持"噪声"结论），但论文应同时披露"两个非零 CI 均为正方向"这一观察（诚实性），并指出量级（0.0004-0.0007 PPL）远低于 seed 间标准差。

### DA-CRITICAL 逐条裁决（IRON RULE 4）

| # | DA-CRITICAL | 裁决 | 依据与处置 |
|---|---|---|---|
| C1 | 9B GSM8K 零宽 CI 不可信 | **VALIDATED** | 主编重算确认 3 seed fp32=0.885/bf16=0.88 全同、CI=[-0.005,-0.005] 退化。greedy + 固定 200 样本下 seed 独立性存疑。**必改**：核查 reasoning_bench.py seed 语义；若 seed 无真实随机性，9B GSM8K 改报"确定性 −0.5pt（3 seed 重复）"或补跑带真实随机性的协议；禁止用退化 CI 支撑显著性表述 |
| C2 | claim #5 仅单 seed pilot 支撑 | **VALIDATED** | summary.json 确认 6/6 均 seed 7。文档 §7.2 已标注"pilot 不得当 formal"，但 claim whitelist 未携带依赖标注。**必改**：claim #5 降级为 ANALYZED + "formal pending"，S-formal 完成后按 3-seed mean±CI 表述；另需回应 DA 的"带宽 vs 容量"替代解释（pilot 未隔离） |
| C3 | 容量模型误差全负 = 结构性保守偏差 | **VALIDATED（定性修正）** | 4 点误差全负属实；根因为离散 block 分配（fp32 block 2064 vs bf16 1072），非随机噪声。但**非缺陷**：模型参数独立于被预测量（非 tautology），保守下界对容量规划是实用属性。**必改**：论文报 signed error + 讨论 block 粒度机制 + 将模型表述为"保守下界"而非"无偏点估计" |

### DA-MAJOR 裁决摘要

| # | DA-MAJOR | 裁决 | 处置 |
|---|---|---|---|
| M1 | "So what?"（flag-flip 够不够顶会） | **采纳为定位风险** | 论文必须用"视角 + 预测模型 + 闭环"三位一体论证增量，且 M-2x2/Q-stacking/S-formal 补齐后闭环才成立；审稿人会问"为什么这是系统论文"——答案要写在 Intro |
| M2 | 敏感度 36 假设无校正 | **采纳** | Bonferroni/FDR 校正 + 决策规则预注册 + 符号一致性披露 |
| M3 | GSM8K 2B vs 9B 不对称（5.3x） | **采纳** | 机制假说讨论（容量冗余、attention 层补偿、规模效应）；可选 9B 简版逐层 GSM8K 探针 |
| M4 | fp16 容量数据从表 5.1 省略 | **采纳** | 表 5.1 补 fp16 列或注释（分析 JSON 已有数据，仅展示问题） |
| M5 | RULER 2B +0.49 vs 9B -0.71 方向相反 | **采纳** | 非零格补 2 seed（同 R2 W4）；结果出来前不宣称"整体持平"的强表述 |
| M6 | fp8/int8 state 未探索 | **采纳为范围声明** | Discussion 明确"state 精度谱系"边界（fp16 容量已测、质量未测；fp8/int8 为 future work），不宣称"二维预算"是完整谱系 |

## Part 2: Editorial Decision Letter

Dear Author(s),

感谢提交《Precision Budget Allocation for Hybrid Linear Attention Models in LLM Serving（工作稿）》至 MLSys 2026 模拟审稿。本报告由 3 位打分审稿人（EIC + R1 + R2）+ 1 位 Devil's Advocate 独立评审后合成（R3 缺席）。我们对研究方向的原创性定位与实验纪律给出明确肯定；当前证据底座尚不能支撑投稿，但所有阻塞项均有低成本明确修复路径。

### Decision: MAJOR REVISION

契约机械判定：无审稿人给出不可修复 block（EIC 的 Evidence Sufficiency block 标注 *repairable*，1-2 天可补齐），4 位打分审稿人加权 66.0 / 66.8 / 69.8 / 70.4（Major Revision 档，平均 68.3）。与 08-06 轮 REJECT 的关键区别：本轮不存在"文本与自身已验证证据矛盾"的投稿级缺陷；全部问题为**证据链未完成 + 统计表述可修复**。若三缺实验与统计修复在 1-2 周内完成，本方向具备 MLSys 投稿竞争力。

### Consensus Analysis

四组共识（见 Part 1）均已转化为 Revision Roadmap R1-R6。三组分歧的裁决已给出（决策严格度、负偏差定性、9B CI 处理）。

### 核心裁决点

1. **9B GSM8K CI 需立即核查 seed 语义**（DA-CRITICAL C1 VALIDATED）：这是本轮唯一可能涉及"数据产生方式"的问题——若 seed 不产生真实随机性，9B GSM8K 的"3-seed 配对"表述不成立，需按确定性结果改写。
2. **claim #5 降级**：S-formal 完成前，serving SLO 收益只能作为方向性证据（ANALYZED），不得进 Abstract。
3. **容量模型重新定位为"保守下界"**：signed error + block 粒度机制讨论。
4. **三缺实验 = 投稿红线**：M-2x2（~30 min）、Q-stacking（~45 min）、S-formal（6-7 h 挂机）完成前不投。
5. **"第二维度"叙事 = "正交可复合"而非"幅度并列"**（R3 W1）：bf16 state 与 KV 量化是互补维度（短上下文 state 主导、长上下文 KV 主导），2x2 表 + stacking 质量必须证明复合增益。

## Part 3: Revision Roadmap

### Required Revisions (Must Fix)

| # | Revision Item | Sub-Claim(s) | Source | Priority | Estimated Effort |
|---|---|---|---|---|---|
| R1 | **9B GSM8K seed 语义核查 + CI 修复**：核查 reasoning_bench.py 的 seed 是否产生真实随机性；若否，9B 改报确定性差异或补跑真实随机协议；禁止零宽 CI 作显著性表述 | SC-3 | DA-C1 / EIC W3 | P0 | 0.5-1 天 |
| R2 | **M-2x2 容量补齐**（fp16 KV × {fp32,bf16} state，2B 4K/16K + 9B 4K，6 探针）| SC-2 | EIC W1 / R1 W3 / DA | P0 | ~30 min 计算 |
| R3 | **Q-stacking 质量叠加**（uniform int4 KV × {fp32,bf16} state 的 PPL 2B C4/PG19 3-seed + GSM8K 2B 3-seed）| SC-3 | EIC W1 / R1 W3 / DA | P0 | ~45 min 计算 |
| R4 | **S-formal serving**（Random60 + ShareGPT300，3 seeds，uniform int4 × {fp32,bf16} state）| SC-5 | EIC W2 / DA C2 | P0 | 6-7 h 挂机 |
| R5 | **容量模型偏差讨论**：signed error 报告 + block 粒度机制（fp32 2064 vs bf16 1072）+ 模型定位"保守下界" | SC-2 | DA C3 / R1 W1 | P0 | 2-3 h 分析+写作 |
| R6 | **敏感度门统计修复**：Bonferroni（alpha/36）或 FDR 校正 + 决策规则预注册 + 2 个非零 CI 的符号一致性披露 | SC-4 | R1 W2 / DA M2 | P1 | 1-2 h |
| R7 | **harness chunk 消融**：chunk=1 vs 128 的 smoke PPL（2B C4，1 seed）量化语义差异 | SC-3 | R2 W3 / EIC W4 / DA m3 | P1 | ~1 h |
| R8 | **RULER 非零格补 2 seed**（5 格：2B FWE L4096/L8192；9B niah_multiquery L4096/L8192、FWE L8192）| SC-3 | R2 W4 / DA M5 | P1 | 2-3 h |
| R9 | **GSM8K 不对称机制讨论 + fp16 state 质量 smoke**（2B C4 1 seed PPL，~5 min）| SC-3 | R2 W1/W6 / DA M3 | P1 | 2 h |
| R10 | **文献扩展**：Mamba2 state 精度原始讨论、FlashInfer SSU 独立条目、SSM state 表示理论；prior art 表加"为什么 state 可压缩"数值直觉 | SC-1 | R2 W2 / EIC W6 | P1 | 0.5-1 天 |
| R11 | **claim whitelist 修订**：claim #5 降级 ANALYZED + formal pending；claim 3 加"RULER 单 seed / harness chunk 语义"限定；claim 1 限定"现有 serving 系统工作" | SC-5/SC-3 | DA C2 / EIC / R1 | P1 | 2 h |
| R12 | **"第二维度"叙事重写**：bf16 state 定位为"与 KV 量化正交可复合"，用设计规则（短上下文 state 主导、长上下文 KV 主导）+ 2x2 表 + stacking 证据论证复合增益，不做幅度并列 | SC-1 | R3 W1 | P1 | 0.5 天 |
| R13 | **fp16 state 质量 smoke**（2B C4，1 seed PPL，~5 min）+ bf16 下界讨论（GDN 循环误差累积、fp8 动态范围、kernel 支持现状）→ Discussion 明确"精度谱系"边界 | SC-3/SC-2 | R3 W2 / DA M6 / R2 W6 | P1 | 2 h |
| R14 | **一般性声明收窄或跨架构探针**：至少一个非 Qwen3.5 架构（Mamba2 2.7B 等）4K 容量探针，或明确 scope 为 GDN-based 架构 + 用 A/G 参数预测其他架构的 r_state | SC-1 | R3 W3 | P2 | 0.5-1 天 |
| R15 | **替代容量杠杆对比段**（KV 驱逐 H2O/SnapKV/PyramidKV、KV offloading、prefix caching）+ 短/长上下文交叉点分析图（由容量模型解析推导）+ TP 分片影响讨论 | SC-2 | R3 W4/W5/W6 | P2 | 0.5 天 |
| R16 | **train-inference mismatch 讨论**：Qwen3.5 以 fp32 state 训练，GSM8K 回退可能源于训练-推理精度失配；讨论或简版验证 | SC-3 | R3 Q2 | P2 | 0.5-1 天 |

### Suggested Revisions (Should Fix)

| # | Revision Item | Source | Priority |
|---|---|---|---|
| S1 | A_q 推导链（9B：16_384/3.878 的架构直接计算交叉验证）补进论文方法论节 | EIC W5 / R2 W5 | P2 |
| S2 | 表 5.1 补 fp16 列；容量表加 signed error 列；r_state(L) vs L 预测-实测曲线图 | DA M4 / R1 Minor / R2 Minor | P2 |
| S3 | PPL 补最小可检测效应（80% power 敏感性分析）或等价性检验（预设 margin） | R1 W4 | P2 |
| S4 | 数据地图补 `MANIFEST.json` / `results/verified|quality` 索引文件 | EIC Minor | P2 |
| S5 | pilot 的 worktree commit 关系澄清（attempt_contract git_commit 3267efa vs summary d39e98c/56674fd） | R1 Reproducibility 观察 | P2 |
| S6 | "bit-match" 术语澄清（容差 1e-9 而非严格位级） | R1 Minor | P3 |
| S7 | 旧方向（A2/packed）与新方向（state bits）的论文衔接策略明确：两贡献如何共处一篇 | EIC / DA 观察 | P2 |

### 修订后重审触发条件

R1-R4（P0 四项）+ R5-R6（统计修复）+ R12（叙事重写）+ R13（fp16 下界）完成，且 claim whitelist 按 R11 修订后，5 席重审。预计投入：P0 约 1.5-2 天（含 S-formal 挂机），P1 约 1-2 天，P2 视取舍（跨架构探针/替代杠杆对比可作 future work 表述）。

## 附录 C — 审稿人状态与加权汇总（最终）

| 审稿人 | 推荐 | 加权 | Originality 20% | Rigor 25% | Evidence 25% | Coherence 15% | Writing 15% | 判定 |
|---|---|---|---|---|---|---|---|---|
| EIC | Major Revision | 66.0 | 78 | 72 | 50 (**block-repairable**) | 68 | 65 | warn/block |
| R1 | Major Revision | 66.8 | 72 | 60 (warn) | 55 (warn) | 78 | 80 | warn |
| R2 | Major Revision | 69.8 | 75 | 68 (warn) | 60 (warn) | 80 | 75 | warn |
| R3 | Major Revision | 70.4 | 78 | 72 | 58 (warn) | 80 | — (not_assessed) | warn |
| DA | 不打分 | — | — | — | — | — | — | 3 CRITICAL 全 VALIDATED |
| **均值** | **Major Revision** | **68.3** | 76 | 68 | 56 | 77 | — | — |

---

## 附录 A — 主编侧独立核验记录（Editor's Own Verification）

以下数字由主编亲自重读原始 JSON 交叉核对（非转述审稿人）：

| 核验项 | 文档声明 | 原始 JSON | 结论 |
|---|---|---|---|
| 容量模型 4 点误差 | −3.24%~−0.18% | capacity-state-analysis.json rows[0-3]：−2.37/−3.24/−0.18/−1.07 | ✅ 一致 |
| fp16==bf16 容量 | 同字节 | 2B/9B @4K fp16_capacity==bf16_capacity | ✅ 一致 |
| PPL 4 格 CI 含 0 | 统计不可区分 | ppl-state-dtype-analysis：4 个 ci95 均含 0 | ✅ 一致 |
| GSM8K 2B 配对 | −2.67pt [−3.38,−1.95] | mean=−0.0267, ci95=[−0.0338,−0.0195] | ✅ 一致 |
| GSM8K 9B 配对 | −0.5pt [−0.5,−0.5] | 3 seed 全 −0.005，CI 单点 | ✅ 一致（但退化，见 DA C1） |
| RULER 2B 总体 | +0.49 | ruler-statebf16-analysis：delta_mean=0.49 | ✅ 一致 |
| Pilot r40 | 0.666→0.958，TTFT 4686→1540 | pilot-analysis rows：fp32 0.6656 / bf16 0.9583，ttft_p99 4686.3/1540.4 | ✅ 一致 |
| 敏感度 2/36 CI 不含 0 | L2/L8 C4 | L2 [3e-6,0.00078]、L8 [0.000486,0.000824]，其余 34 格全不显著 | ✅ 一致 |
| 配置 sha256 | f7475580… | experiments/configs/statebf16_random60_pilot.yaml sha256=f7475580… | ✅ 一致 |
| bf16 生效硬证据 | "Using the user-specified value" | 3 个 bf16 sample 的 contract.json + server.log 均含 | ✅ 一致 |
| 脚本存在性 | 14 个引用脚本 | 全部存在；hybrid_premise.py 含 --state-dtype 完整实现 | ✅ 一致 |
| 所有 headline 来自 5090 | 硬规则 5 | pilot environment.json 确认 RTX 5090（GPU UUID GPU-5a34486f…，R3 亦独立确认） | ✅ 一致 |

**主编侧新增观察**（审稿人未报）：9B GSM8K 的 CI 退化 + 2B/9B 差异 5.3x 组合暗示"2B 对 state 精度敏感、9B 几乎免疫"可能是规模效应（更大模型的 representational redundancy），也可能是 9B 测试集的 floor/ceiling 效应（0.885→0.88 接近能力上限）。论文需讨论这一解释空间。

## 附录 B — 文本 vs 证据状态对照

| 主题 | research-summary（08-09） | 仓库证据 | 状态 |
|---|---|---|---|
| 容量模型 | 4 点验证误差 <3.3% | 10 探针 JSON 全匹配 | ✅ 证据充分（偏差讨论待补） |
| bf16 容量收益 | +38~41% @4K / +11~14% @16K | 探针一致 | ✅ 证据充分 |
| PPL 持平 | 4 CI 含 0 | 一致 | ✅ 证据充分（MDE 报告待补） |
| RULER 持平 | 2B +0.49 / 9B −0.71 | 单 seed | ⚠️ 单 seed 归因待补 seed |
| GSM8K 回退 | 2B −2.67pt / 9B −0.5pt | 2B CI 正确；9B CI 退化 | ⚠️ 9B 需修复 |
| 逐层敏感度负面 | 无逐层收益 | 2/36 CI 不含 0 | ⚠️ 统计校正待补 |
| serving SLO | pilot 方向性 | 6/6 单 seed | ⚠️ formal 待补 |
| 2×2 联合预算 | 待补 | 缺 | ❌ 阻塞 |
| stacking 质量 | 待补 | 缺 | ❌ 阻塞 |
| fp16 state 质量 | 未测（§5.6 诚实标注） | 缺 | ⚠️ 下界谱系不完整（R3 W2/R2 W6） |
| 非 Qwen3.5 架构 | 未测 | 缺 | ⚠️ 一般性未验证（R3 W3） |
