# Phase 2 — Paper-visible Review: Peer Reviewer 3 (Perspective)

contract_role: perspective

## Dimension Scores

### D1: methodology_rigor
score: warn
依据冻结 Scoring Plan：论文有真实 serving 数据（protocol v3、Poisson 到达、warmup、goodput/offered≥0.95、3 seeds、失败计入分母）、双次 formal run、预注册 gate 与完整容差披露，且容量效应经 112-cell 确定性分配器探针两次独立验证。但扫描范围窄：单一硬件（一张 RTX 5090）、单引擎、TP=1、仅过载区 goodput delta 与边界表而无完整 operating curve。结论措辞严格未超出该范围（"we make no general serving-speed claim"），统计披露强（CI/power/MDE/FDR/种子），PPL 简化 harness 已如实标注——精确命中我预注册的 warn 触发条件（窄扫描 + 措辞未越界 + 简化 harness 已披露），未达 block（block 要求核心部署主张完全无 serving/并发扫描证据，此处并非如此；我抽查结果档案，容量矩阵与 gate 数字与论文逐项吻合）。

### D2: domain_accuracy
score: pass
对部署现实的描述属实：vLLM 的 PagedAttention 统一内存池、`mamba_ssm_cache_dtype` 配置路径、int4 KV 与 fp16 KV 的每 token 字节账目（0.5/2 bytes）、GDN state 18.63 MiB/序列的推导（18×1,085,440 bytes 计算一致）、KV 量化先验工作的定位（KIVI/KVQuant/MiniKV/ReplaySSM 描述准确）、TP 扩展明确标注为 "expectation, not a measurement" 而非冒充实测。我核验了 M4 gate 档案的 precision_log_evidence：四个 allocation 的 `kv_cache_dtype`/`mamba_ssm_cache_dtype` 均 PASS，配置真实生效，不存在静默退化。成本/尾延迟常识（batch 对 TTFT 与 TPOT 的差异化影响、P95 来源、失败计入分母）处理正确。未发现会把实践者引入错误决策的事实性错误。

### D3: argumentative_coherence
score: pass
证据链完整且每一步的推广都限定在已验证范围：机制（state 减半字节）→ 容量效应（52/52 对、15.44% median、双 attempt 复现）→ 质量（GSM8K 2B 回归如实保留、RULER 零差异标注为非等价）→ serving（拒绝 headline，披露 183/720 超差与 79.61% 最大差）。abstract 与正文措辞一致（"verified capacity effect and scoped policy execution... reporting serving instability"），收益成立条件（短上下文、memory-bound、2B/4K 切片）全文各处一致。不利证据（回归、gate 失败、运行敏感性）被正面处理而非回避。无未证明的关键跳步：论文没有从"省显存"直接推出"更多并发用户"的部署赢家叙事，selector 的 3 个预算映射也被限定在一个切片。

### D4: cross_disciplinary_relevance
score: warn
论文确实用部署语言说话（P95 TTFT/TPOT、goodput、sustainable rate boundary、failures in denominator、SLO-goodput 下界），容量模型对产能规划者可直接代入使用，这是少见的。但部署者决策所需的关键信息缺失：无 cost/request、无 requests/GPU-hour、无 GPU 数缩减换算（"247 slots" 停在并发数）；无完整 operating curve（低负载区、P50/P99、latency CDF 均缺）；切换成本未量化（cold restart，热切换与在线自适应留给 future work）；多硬件与 TP 未测。按我的预注册 warn 条件：提供了部分部署视角但缺完整曲线与切换成本讨论，且论文已在 Limitations 逐项如实声明这些 gap（Sec. 7 的 Controller scope、Tensor parallelism、Generalization）——恰为 warn，未达 block（block 要求 gap 完全空白且未承认，此处承认得相当彻底）。

### D5: writing_and_structure
score: pass
信息提取效率高：数字、单位、条件（硬件、seed、协议版本、commit）均可追溯；Table 1 有单位与方向标注，Fig. 2/3/4 带 95% CI 与样本条件；Limitations 章节真实存在且具体（8 个小节，逐项量化影响范围）；abstract/conclusion 与正文证据覆盖一致；复现路径（冻结 worktree commit、measurement contracts、per-seed 数据、artifacts 发布）可查找。可改进处（Fig. 4 只画 delta、abstract 密度过高）均为轻微问题，不构成结构障碍。

## Failure Condition Checks

### F1
fired: false

### F2
fired: false

### F3
fired: false

### F0
fired: false

## Review Body

**Overall Recommendation**: Weak Accept（接受）。作为产能规划与 SLO/cost 视角的 outsider 审稿人，我认为论文的核心主张——容量效应（52/52 对、15.44% median、模型中位绝对残差 1.81%）与 scoped 策略执行——有据可立且可审计；部署经济学与切换成本的空白被如实声明为 future work，不足以构成拒稿或大修。采纳该工作的风险低，但部署价值量化尚未完成。

**Confidence Score**: 4/5。我抽查了 results/ 下的结果档案（capacity-phase-formal-20260811.analysis.json 的 52 对/112 cells/1.81% 残差/全对偏向 bf16；m4_gate4_validation.json 的 537/720、183/720、79.61%、713/720、38/40；R2 验证的 112 cells 复现），与论文数字逐项吻合。扣一分是作为 outsider 对硬件迁移性（消费级 5090 → 数据中心卡）与 GDN 模型生态部署普及度的不确定性——论文本身也承认此边界。

**Summary Assessment**（约 220 词）：这是一篇让我作为产能规划者罕见地可以直接采信数字的系统研究。作者用冻结协议、预注册 gate 与双次 formal run 验证了核心容量效应：52 对 fp32/bf16 单元全部偏向 bf16 state、中位容量增益 15.44%，无参数模型中位绝对残差 1.81%——我抽查结果档案与论文数字逐一吻合，且日志级精度证据显示配置确实生效，不存在静默退化。质量侧诚实保留 2B GSM8K 的 1.0 pt 回归（p=0.025，低于 MDE 1.16），RULER 零差异被明确标注为观测一致而非等价。serving 侧作者主动披露不稳定（183/720 超差、最大 79.61%）并放弃速度 headline——这种纪律在系统论文里罕见，正是部署者需要的可审计性。短板在部署经济学：无 cost/request 或 GPU 数换算、无完整 operating curve、切换用 cold restart、硬件仅一张消费级 5090；作者均在 Limitations 如实声明。作为 outsider，我认为其核心主张有据可立、采纳风险低，但部署价值量化尚未完成。

**Strengths**
1. 可复现纪律达到生产审计标准：预注册 gate、冻结 worktree、双次 formal run、失败 gate 保留披露而非转化（Sec. 4、Sec. 6.5）。这是我读过的系统论文中部署者最需要的品质。
2. 诚实范围控制：serving 不稳定与 null results 被如实报告（183/720、79.61%），abstract 即声明 "reporting serving instability rather than a universal performance benefit"（abstract、Sec. 6.5）。
3. 无参数容量模型 N(L)=M/(AL+G) 简单、可迁移，由确定性分配器探针双验证（Sec. 3、Sec. 6.1）——产能规划者可直接代入自己的 M 与 L 计算收益。
4. 质量 tradeoff 以 CI/power/MDE/FDR 量化，包括 2B 回归的完整披露（Sec. 6.2、Fig. 2）。
5. 落地门槛极低：一行 vLLM 配置（`mamba_ssm_cache_dtype=bfloat16`），无重训、无 kernel 改动；selector fail-closed、不插值（Sec. 3、Sec. 6.4）——安全默认值符合生产实践。

**Weaknesses**
1. 缺部署经济学翻译（Sec. 6.1 与 Sec. 8 Operational rule 停在 "247 slots"）。为什么是弱点：部署决策最终是 TCO/GPU 预算决策，并发槽数本身不构成决策输入。替代建议：补一个 worked example——"32 GB 预算、4K 短上下文、同 SLO 下 bf16 state 使每 GPU-hour 多服务约 X 请求，折合单位成本下降约 Y%"。
2. 单一消费级硬件（Sec. 4 Platform：一张 RTX 5090）。为什么是弱点：数据中心部署在 H100/H200/B200 与多卡上；容量账目与硬件基本无关（可论证），但过载区 latency/goodput 行为与带宽强相关。替代建议：至少加一段带宽无关性论证，或补一张数据中心卡的验证。
3. 无完整 operating curve（Sec. 6.5 + Fig. 4：只有 5 档 TTFT 阈值边界表与过载区 goodput delta）。为什么是弱点：部署者做 SLO 规划需要 offered load 从低到高的 P50/P95/P99 TTFT/TPOT 全曲线，delta 图无法回答"我的流量在哪个点失守"。替代建议：per-seed P95 数据已在档案中，用现有数据画两条 dtype 的完整曲线成本应很低。
4. 切换成本未量化（Sec. 6.4 selector 用 cold restart；Limitations "Controller scope" 承认无在线自适应）。为什么是弱点：生产环境不允许为换精度配置而冷重启。替代建议：报告一次 cold restart 的端到端分钟数作为下限，并说明 vLLM 热切换所需的改动量级。
5. 混合长度请求下的碎片损失未量化（Sec. 6.1 探针为统一 L 的确定性分配器运行）。为什么是弱点：247 slots 假设全部 4K 请求；真实流量混合长度，block 碎片会侵蚀实现率。替代建议：声明该数字为上限，或用现有探针框架做一次混合长度模拟。

**Detailed Comments**

*Assumption Audit*
- Explicit（已声明）：短上下文高并发 memory-bound regime（Sec. 1）；uniform int4 KV（`int4_per_token_head`）与 fp16 KV；TP=1 单卡；Qwen3.5 GDN hybrid；vLLM 分配器行为；Poisson 到达 + 120 s warmup + goodput/offered≥0.95（Sec. 4）；TTFT 250–3000 ms / TPOT 200 ms SLO 档；utilization 0.70–0.90。
- Implicit（未声明）：(1) 确定性分配器探针 = 生产容量——无混合长度碎片、无 prefix caching/eviction 交互（Limitations 披露未测，但碎片影响未量化）；(2) bf16 state 的质量影响与 thinking 模式无关——RULER 禁用 thinking，而生产 Qwen3.5 默认开启，thinking 模式下精度路径不同（这是最值得提醒实践者的一点）；(3) 消费级卡结果代表数据中心卡——容量比率模型层面对，latency 层未验证；(4) Poisson 到达代表生产流量——实际流量有突发性与昼夜性。
- Paradigmatic（学科范式）："省内存 → 部署赢"——论文自己抵抗了这个系统领域范式（保留失败 gate、拒绝 speed headline），这是本工作最重要的方法论贡献；另一范式"benchmark ≈ 生产质量"——论文部分挑战（排除单 seed long-form 证据），但 GSM8K/RULER 仍是窄任务证据，作者亦承认。

*Cross-Disciplinary Connections*
- Parallel research：SaVoir 的 cost-quality Pareto 建模与本工作的 joint budget 是同构问题（多维预算下的前沿搜索）；Splitwise 提供推理成本构成的分析口径；KIVI/KVQuant 的 serving 曲线可作为对照基线。
- Borrowing opportunities（作者可借鉴的）：容量规划传统的 offered load → SLO 曲线方法（源自 Erlang B/queueing 传统，Google Autopilot 的 workload-driven 容量预测、USL 扩展律）——作者已有容量模型雏形，借用曲线方法即可补齐 operating curve。
- Methodological borrowing（领域可借鉴作者的）：预注册 + gate 来自临床试验传统，作者在系统评测中实践得极好；gate 失败即停船（不发布 claim）类似 A/B 实验 guardrail 惯例，值得领域推广。

*Practical Impact*
- Real-world application：短上下文高并发 memory-bound 场景（agent 式短请求、RAG 检索、客服）下，state-bf16 是接近零门槛的容量杠杆；15.44% median 直接转并发或 GPU 预算；9B 无质量回归，2B 需 per-task 验证（GSM8K -1.0 pt）。
- Implementation feasibility：一行配置、无重训、无 kernel 改动——同类工作中落地门槛最低；selector 是研究原型（cold restart + 需 calibrated profile），生产化需要热切换与在线自适应。
- Stakeholders：推理平台团队（默认 dtype 决策）、容量规划/SRE（预算公式）、模型服务商（per-task 质量验证清单）。

*Broader Implications*
伦理与社会：单位推理成本下降有能耗与普惠含义；诚实报告 null results 对领域可复现文化有示范价值。未来方向：热切换与在线自适应、fp8/int8 state 精度、TP 实测、数据中心卡验证、混合长度碎片建模。

**Cross-Disciplinary Reading Recommendations**（供作者与读者）
1. J. Dean, L. A. Barroso. The Tail at Scale. CACM, 2013.——tail latency 与 SLO 方法论的起点。
2. P. Patel et al. Splitwise: Efficient Generative LLM Inference Using Phase Splitting. ISCA 2024.——推理成本构成与 GPU 预算口径。
3. B. Sun et al. Llumnix: Dynamic Scheduling for Large Language Model Serving. OSDI 2024.——SLO 感知调度与负载动态。
4. K. Rzadca et al. Autopilot: Workload Autoscaling at Google. EuroSys 2020.——workload-driven 容量规划实践。
5. Y. Chen et al. SaVoir: Cost-Quality Pareto-Optimal LLM Serving. arXiv 2024.——与本工作最直接可对话的 cost-quality 前沿建模。

**Questions for Authors**
1. 一次 cold restart 的端到端成本（分钟数、资源）是多少？vLLM 支持热切换 state dtype 需要哪些改动？
2. 为何不画完整 operating curve（低负载区、P50/P99、latency CDF）？per-seed P95 数据已在档案中，是数据缺失还是协议限制？
3. 247 slots 在混合长度请求分布下的实现率估计是多少？现有探针框架能否做一次混合长度模拟？
4. 9B GSM8K 的 int4-KV + bf16-state 交互效应是否有更大样本的验证计划（当前 n=9 seeds 下 CI 含零）？
5. 数据中心卡（H100/H200）上，state/KV 字节占比不变（由模型决定），但带宽与过载区行为会变——你们预期 15.44% median 不变而 delta 方向可能变化，还是两者都需要重测？

**Minor Issues**
- Table 1 的 gap 列符号混合，建议统一方向并注明 "signed gap, (measured−predicted)/predicted" 之外的正负含义。
- Fig. 4 仅画 delta；建议附一张绝对 goodput/latency 小图，或明确指引读者到档案中的绝对曲线。
- Abstract 密度过高（读起来像验证清单）；建议为部署读者增加一句操作化总结（"在短上下文 memory-bound 场景，切换 state 到 bf16 约带来 15% 额外并发容量，代价是 2B 模型需 per-task 质量验证"）。
- 正文称 "maximum per-cell token difference 1.42%"，我核对的 R2 验证档案为 1.41%（0.0141）；请确认该数字引用的是 R2-to-R3 比较。
- 建议注明 RTX 5090 为消费级卡及功耗设置（stock 或受限），便于读者判断与数据中心卡的可比性。

## Editorial Decision
editorial_decision=accept
