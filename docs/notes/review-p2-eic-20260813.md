# ARS Review — Phase 2 (Paper-visible) — EIC

Phase 2 完整输出。依据：v3.6.2 contract（reviewer_full）、Phase 1 预注册 Scoring Plan（docs/notes/review-p1-eic-20260813.md，本阶段无 dissent）、论文全文 main.tex 及其引用的 8 张图、以及 E:\MLSys_Research\results\ 下的档案核验（核验方法：Grep/Read/pymupdf 读取现有 PNG 预览，未写入任何文件）。

核验摘要（evidence check）：capacity-phase 分析 JSON 的 52 对中位增益 15.44%、中位绝对残差 1.81%、残差范围 [-3.66%, +13.21%]、52/52 方向一致，与论文逐位一致；GSM8K 9-seed 的 -1.00pp [-1.71, -0.29]、p=0.02494、MDE 1.16pp、power 67.53% 一致；RULER 五 cell 零 delta 与 91.11/98.33/100.00 一致；controller 三映射（strict=full / medium=state_only / high=joint）、18,000/18,000 请求、Gate 4 PASS 一致；M4 的 537/720、183/720、max 79.61%、713/720、38/40 一致。未能从本地 results/ 核验：M3 mechanism 数字（75.39% 等）、serving 第二 run 的 r40/r45 deltas 与 Table 2 括号值、capacity R2-to-R3 Gate-4 比较 JSON——相关分析脚本/配置存在但产物未入库。

contract_role: eic

## Dimension Scores

### D1: methodology_rigor
score: warn

依据 Phase 1 触发词打分：端到端缺失不成立（serving 级测量大量存在，且论文未对未验证维度做 headline 主张）；混淆比较不成立（同硬件/同引擎/同模型、paired 设计）；可复现性非零（四个 headline 包有 sha256 档案）。触发 warn 的是：①基线上限不足——全文无任何第三方系统基线，ReplaySSM/fp8-state 等 state 压缩 prior work 被引用但从未 head-to-head 测量，selector 无任何现有调度器/策略基线，joint 增量不可与 closest prior art 归因对比；②可复现性 affordance 部分缺失——论文承诺 full release，但 M3 与 serving 第二 run 分析产物不在本地 results/，controller 复现审计为 logical-only（hash_validation_performed=false）。

### D2: domain_accuracy
score: pass

Related-work 表述与文献事实一致：KIVI/KVQuant/MiniKV/TurboQuant 均被正确描述为 attention-KV 侧工作；Quamba/MambaQuant 被正确区分（W8A8/W4A8 计算路径，不涉及 per-sequence state dtype）；ReplaySSM 与 FP8 state 路径被如实承认为 prior art，"gap" 声明被小心限定为 joint budget accounting 而非 state compression 本身；无 "first" 类可证伪声明。方法依赖的架构事实经档案核验成立：2B 为 18 GDN + 6 attention 层（a2 capacity gate JSON 的 18 个 MambaSpec group + 6 attention group）、state 18.63 MiB/seq = 18×1,085,440 B 精确吻合、int4 0.5 B/element 正确。个别 citation 配对欠精确（vllmpr43518 是 checkpointing 而非 compression），不足以触发 warn。

### D3: argumentative_coherence
score: pass

身份-证据匹配成立：论文自称 "a system study with four contributions"，证据强度与身份相符——未包装为新系统/新 kernel，headline 仅为 capacity effect + scoped policy execution，负结果保留而非转化（four-config Gate 4 失败、BH-FDR 零存活均如实披露）。四个贡献 bullet 与 Sec 4.1-4.5 一一落点。摘要/结论措辞与正文证据一致（abstract 的限制披露与正文逐条对应）。joint 论证无循环：容量模型参数来自架构与 page layout、独立于 probe 数据（Sec 3 明确声明非循环），joint 的组件消融（full/kv-only/state-only/joint）虽在 serving 上未通过稳定性 gate，但论文因此不主张 joint serving 优势——这是正确的论证收缩而非断裂。

### D4: cross_disciplinary_relevance
score: warn

触发 warn 的是 Phase 1 计划中的"关键术语无定义"：RULER 的两个 subtask 名 "FWE" 与 "NIAH-multiquery" 在 Sec 4.2 与 Fig 3 中出现且承载全部 RULER 质量主张，但全文从未展开（systems 读者无法判断 91.11 的 FWE 分数测量的是什么）；abstract 中 "no-think RULER cells" 首次出现即用，正文 Sec 4.2 才解释 thinking-disabled 的原因。双向可读性总体良好（recurrent state/内存池/TTFT/TPOT 均有服务侧桥梁，质量侧有 CI 与统计口径），推广声明受控（GDN-based hybrids 限定、TP 段落标明"expectation, not a measurement"），故为 warn 而非 block。

### D5: writing_and_structure
score: warn

触发 warn 的是"紧凑篇幅挤掉 setup/provenance 细节"：①完整 112-cell 容量矩阵未入文（仅 Table 1 的 7 行 slice），读者只能信任汇总统计；②M3 的 Gate-4 容差未给出（仅 throughput 10%，TTFT/TPOT 的容差缺失）；③Sec 4.5 的多 package 时间线（55f4768 vs e2fa285、Table 2 与第二 run 的对应关系）需对照 Sec 3 反复拼凑。signature figure 存在（Fig 1 双面板：joint budget 示意图 + measured-vs-predicted），图注基本自包含；贡献 bullet 与实验映射完整。结构性失败不成立，故为 warn。

## Failure Condition Checks

### F1
fired: false

依据：我的 mandatory 维度 D1=warn、D2=pass、D3=pass，无 block。

### F2
fired: false

依据：mandatory 维度中仅 D1 一项为 warn（或更差），未达两项。

### F3
fired: false

依据：high-priority 维度仅 D4，且为 warn 而非 block。

### F0
fired: false

依据：D1=warn，非全部 pass。

## Review Body

### Overall Recommendation

Minor（Accept with minor revisions）

### Confidence Score

3 / 5

### Summary Assessment

本论文是一篇方法论极其自律的 system study：将 hybrid linear-attention serving 的 recurrent-state dtype 提升为与 KV bit-width 并列的第二预算维度，交付容量模型、可执行 selector、配对质量地图，并如实披露 serving 不稳定性。我核验了本地 results/ 档案：112-cell 容量矩阵的 52/52 方向、15.44% 中位增益、1.81% 中位绝对残差、GSM8K 9-seed 的 p 值与 MDE/power、RULER 五 cell 零 delta、controller 三映射与 18,000 请求、M4 的 537/720 与 79.61% 等均与论文逐位一致；但 M3 与 serving 第二 run 的产物未在本地档案中。论文身份（system study）与证据匹配：无端到端收益虚标，负结果被保留而非转化，摘要的边界披露罕见地准确。主要短板：serving payoff 未建立（无 cell 通过 BH-FDR、four-config Gate 4 失败）、无第三方 state 压缩基线、selector 技术深度有限、质量证据分辨率低（RULER 零宽度区间非等价性检验）。作为 EIC，我认为该文的真实贡献是问题框架与验证协议纪律；按预注册评分标准无 failure condition 触发，我建议以 minor revision 接受：术语展开、完整矩阵入附录、serving 叙事收紧并明确其方法学定位。

### Strengths

1. 方法学纪律在 serving systems 文献中罕见：预注册 gate、seed 与 paired 设计、MDE/power 报告、BH-FDR/Holm 修正、fail-closed selector、负结果保留。四个 headline 包（capacity/GSM8K/RULER/controller/M4）的本地档案与论文数字逐位一致（EIC 独立核验）。
2. 身份-证据匹配：自称 system study，claim 集严格限定于已验证部分（capacity effect + scoped policy execution），Sec 4.5 与 Sec 5 对未验证维度（tail latency、four-config 优势、TP、equivalence）的收缩准确。
3. 容量效应验证完整：52/52 对方向一致、双 attempt 复现（最大 per-cell token 差异 1.42%）、模型残差被如实刻画为 idealized predictor 而非 bound（残差跨零，[-3.66%, +13.21%]）。
4. 选题及时且结论可操作：hybrid 模型的 fp32 state 默认配置浪费被精确量化（2B/4K int4 下 +37.6% 容量、约 247 sequence slots），"Operational rule" 段落直接转化为部署指导。
5. serving 不稳定性披露本身有方法学价值：183/720 超容差、max 79.61%、BH-FDR 零存活为社区提供了可审计的 serving 测量噪声地板证据。

### Weaknesses

1. **Serving 收益未建立（Sec 4.5、Table 2、Fig 4）。** 问题：60 cell 无 BH-FDR 存活，four-config Gate 4 失败，mechanism 仅 throughput 复现（TTFT P95 10/18，max 75.39%）。为什么：对 serving 论文，速度/SLO 是核心 payoff，而本文唯一被验证的效应是内存容量——该效应接近算术必然，serving 维度实际为 null。建议：要么大幅强化 serving 稳定性方法（更多 seeds/更长窗口/环境固化），要么将 serving 章节明确降级为"测量方法学案例"，把论文重心放在 capacity+quality+协议纪律上。
2. **基线缺口（Sec 2、Sec 4.4）。** 问题：ReplaySSM/fp8-state 等 state 压缩 prior work 被引用但从未 head-to-head 测量；selector 无任何现有调度器或简单策略（greedy）基线。为什么：joint budgeting 相对 state compression 的增量不可归因，"gap" 声明（Sec 2 末段）因此悬空。建议：至少补一个 state 压缩替代方案的容量/质量 cell；selector 与简单决策基线对比。
3. **Selector 技术深度有限（Sec 3 Eq. 2、Sec 4.4）。** 问题：约束过滤 + argmax、无在线适配、无插值、fail-closed；验证仅一个 2B/4K/Random/TP=1 slice 与三个预算。为什么：作为四大贡献之一，工程新颖度偏薄，"robust SLO-goodput lower bound" 实质是存储 CI 下界。建议：明确定位为"可执行策略层 + 验证协议"，弱化系统机制预期，或扩展跨 strata 验证。
4. **质量证据分辨率低（Sec 4.2/4.3、Sec 5）。** 问题：RULER 5 cells×3 seeds×20 samples、零宽度区间（明示非等价性检验）；GSM8K 2B 的 1.0pp 回归低于预注册 MDE（1.16pp）；PPL 为 chunk-128 近似（chunk-1 使 PPL +87%）。为什么：质量维度只能支持"未检出损失"，正面保持证据薄弱。建议：增加 cells/samples，或在摘要与结论更突出"no detected loss"的精确表述。
5. **可复现 affordance 有缺口（Sec 3 vs results/ 档案）。** 问题：M3 与 serving 第二 run 的分析 JSON 不在本地 results/；controller 复现审计 hash_validation_performed=false（logical-only）；完整 112-cell 矩阵未入文。为什么：论文承诺 full release，当前仓库仅部分兑现。建议：补齐 M3/serving 分析入库、全矩阵入附录，并注明 controller 审计为 logical-only。

### Detailed Comments

- **Journal Fit**：主题（serving 内存预算与精度权衡、hybrid 架构服务化）契合 MLSys；论文以"system study"而非"新系统"定位，MLSys 接受此类测量/方法论工作，但竞争激烈，取决于 panel 对"无 serving payoff 的 serving 论文"的接受度。
- **Originality**：双维度 joint budgeting 的问题框架与 selector 有新意；核心容量效应本身接近算术（state 字节减半→容量增加），新颖性集中于框架、验证协议与 honest-null 报告。
- **Significance**：对 Qwen3.5/GDN 类 hybrid 部署者有直接操作价值；对 attention-only 服务影响有限；2B/9B 单卡范围限制了推广性（论文已诚实限定）。
- **Structural Coherence**：强。四贡献 bullet 与 Sec 4.1-4.5 一一对应；Limitations 逐条对应正文的每处收缩；无论证断裂。
- **Title & Abstract**：标题准确但较窄（未含 selector/方法论贡献信号）。abstract 信息密度过高——塞满门限值、tolerance 与限定词，对非专业读者不友好；建议精简数字、保留"claim-证据-边界"结构。
- **Conclusion**：与证据一致；"Operational rule" 将容量结论转化为部署指导是亮点；release 承诺与当前仓库状态存在前述缺口。

### Questions for Authors

1. serving 章节占全文最大篇幅却未产生任何 verified speed 效应——是否考虑重构为 capacity+quality+方法学论文，将 serving 不稳定作为"测量地板"结果呈现？
2. M3 与 serving 第二 run 的分析产物何时进入 results/？capacity R2-to-R3 的 Gate-4 比较 JSON 在哪个 package？
3. 与 ReplaySSM/fp8-state 的直接对比是否有计划？joint 相对 state compression 的增量如何界定？
4. 15.44% 中位增益是否主要由短上下文 cell 驱动？能否给出 per-L 分解（正文目前仅 4K/16K slice）？
5. selector 的 lower confidence bound 覆盖率/校准如何保证（profile 的 CI 来自 3 seeds，样本量很小）？

### Minor Issues

1. Sec 4.2/Fig 3："FWE"、"NIAH-multiquery" 全文未展开（D4 warn 主因）。
2. Sec 1："R2-to-R3" 内部命名未定义即用。
3. Sec 2：vllmpr43518（FP8 checkpointing）被用于支撑 "state compression" prior work——checkpointing ≠ compression，建议改写为"state 持久化/精度路径"。
4. Sec 4.5：M3 的 TTFT/TPOT Gate-4 容差未给出（仅 throughput 10%）；Table 2 与 Sec 3 中两个 commit（55f4768/e2fa285）的 package 对应关系需明确标注。
5. Sec 2：\cite{replayssm,vllmpr22196} 的配对欠精确（前者是 blog，后者才是 vLLM state-dtype 路径）。
6. Fig 1(b) 标注 "7 cells" 与 Table 1 的 8 行易混淆（fp16 9B/16K 未入图）。
7. Sec 5 TP 推导段落纯分析未测量（文内已标明），建议移至附录。
8. abstract 中 "no-think RULER cells" 首次出现即用、未解释。

### Recommendation to Peer Reviewers

请各自按冻结的 Scoring Plan 独立打分，不要受本文诚实语气的影响调整触发阈值。EIC 侧机械决策为 accept（无 failure condition 触发），但 D1（serving 证据强度与基线完整性）、D4（术语可读性）是我标记的 warn 点：若 methodology 审稿人在 M3/serving 复现证据或 baseline 上给出更严评分，F2（两项 mandatory warn）将触发 major_revision。请重点核：M3 与 serving 第二 run 的档案完整性；domain 审稿人核 Qwen3.5 架构事实与 state-compression prior art 边界（尤其 ReplaySSM/fp8-state 是否已构成 joint budgeting 的先行工作）；perspective 审稿人评估"无 serving payoff 的 serving 论文"的社区价值定位。

## Editorial Decision

editorial_decision=accept
