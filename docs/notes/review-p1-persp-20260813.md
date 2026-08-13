## Contract Paraphrase

**D1 — methodology_rigor（方法论严谨性）**：从我产能规划的位置看，"严谨"的翻译是：这些数字我敢不敢放进一次真实的部署决策。这意味着实验条件必须接近真实 serving（continuous batching、并发 offered load、真实请求分布），测量流程披露到我能带着自己的环境去重建判断（config、seed、版本、扫描范围），并且报告的变异性要让我能判断尾延迟行为——只有均值没有分布的数字，对 SLO 规划几乎无用。我不做统计细节审查（那是 R1 的职责），但我负责检查研究设计是否系统性地藏掉了部署侧最关心的失效模式：单点结果冒充全曲线、理想化 microbenchmark 冒充 serving 数字、缺少第三方可验证的复现凭据。

**D2 — domain_accuracy（领域准确性）**：我会专门检查论文对"生产 serving 和硬件实际怎么工作"的描述是否属实，因为这是作者学科训练之外、最容易出错、也最容易误导实践者的部分。具体包括：KV cache 内存账目、GPU 内存层级行为、TP/多卡下的真实计算与通信成本、现有 serving 栈（vLLM/SGLang/TensorRT-LLM 等）实际做了什么、成本与尾延迟的构成常识。作者自己的机制可以对，但如果对部署现实的描述错了，论文对我这类读者就从"可参考"变成"有误导性"，所以这一维度对我同样是 mandatory。

**D3 — argumentative_coherence（论证一致性）**：作为部署侧读者，我把论文读成一个故事："这里有个机制 → 这里有证据它有效 → 因此部署者应该采纳它"。一致性的意思是每一步的证据真的支撑下一步，且收益成立的条件（负载区间、batch、序列长度、硬件、模型规模）在全文各处表述一致——不能实验只覆盖窄条件，abstract 却推广到"production serving"。我不做形式化谬误检测（那是 DA 的职责），但我会标记部署叙事断裂的地方：abstract 承诺的收益实验从未演示，或机制层面的节省在翻译到端到端 serving 指标时悄悄蒸发。

**D4 — cross_disciplinary_relevance（跨学科相关性）**：这是最贴近我身份的一个维度。论文应当对相邻领域的实践者——产能规划师、SRE、推理成本负责人——可读且可行动。这意味着用部署原生量词说话（P50/P95/P99 latency、TTFT/TPOT、goodput、cost/request、requests/GPU-hour、同等 SLO 下所需 GPU 数），而不是只用研究原生量词（精度 delta、显存比例、kernel 加速比）。"降成本""支持更多并发""更少 GPU"这类跨学科主张必须配部署侧证据，不能从省了多少显存直接推出。如果我读完还需要自己重做整份容量分析才能决定采不采纳，相关性主张就是不成立的。

**D5 — writing_and_structure（写作与结构）**：对我而言这个维度就是"提取效率"：一个部署实践者能不能不重读三遍就找到数字、适用条件和局限。图/表要 deployment-readable——latency 曲线要展示分布或分位数而不是只有均值，memory/cost 表的单位和硬件条件要可追溯，局限（冷启动、切换成本、硬件依赖）要放在无法被忽略的位置。我关心的不是排版美学，而是结构能不能让我快速走到一个 go/no-go 决策。

## Scoring Plan

### D1: methodology_rigor
- dimension_id: D1
- what_to_look_for: 是否在真实 serving 条件下测量（continuous batching、并发 offered load、端到端栈），还是只有 microbenchmark/单请求数据；是否提供完整 operating curve（offered load 从低到高 → P50/P95/P99 TTFT/TPOT、goodput、排队/超时行为）而非单一工作点；是否披露硬件（GPU 型号/显存/数量）、软件栈版本、seed 与重复次数、方差或分位数信息，使我能在自己的环境重建判断；baseline 是否与生产部署可比（同 stack、同精度预算口径、同 batch 策略）而非弱化版对照；是否说明结果对负载特征（request rate、序列长度分布、decode 长度）的敏感性。
- what_triggers_block: 核心部署主张（如"可支撑更高并发/更少 GPU/更低成本"）完全没有 operating-curve 或并发扫描证据支撑，只有单点或非 serving 条件测量；或测量条件披露不足到无法判断数字适用边界，论文却据此向部署者给出采纳建议。
- what_triggers_warn: 有 serving 数据但扫描范围窄（单一 request rate、单一硬件、单一模型规模）且结论措辞未超出该范围；或统计披露薄弱（无 seed/方差）但结论没有夸大；或使用了自定义简化 harness 但已如实标注其与生产栈的差异。

### D2: domain_accuracy
- dimension_id: D2
- what_to_look_for: 对现有 serving 栈行为、KV cache 内存账目、GPU 内存层级（预分配、碎片）、attention/linear-attention 算子在 TP/多卡下真实成本（带宽受限 vs 算力受限 regime）的描述是否与生产实践一致；对不同硬件的差异化行为是否表述准确（同一机制在 H100 与消费级 GPU、不同显存容量上可能结论不同）；成本/尾延迟构成常识是否准确（P95/P99 的来源、batch 对 TTFT 与 TPOT 的差异化影响）；术语是否按领域惯例使用；引用先验工作时对其部署相关结论的转述是否属实。
- what_triggers_block: 论文对部署现实的核心事实性描述有错误（如现有系统能力、硬件行为、成本构成的错误陈述），且这些错误直接支撑其采纳主张——会把实践者引入错误决策。
- what_triggers_warn: 部署相关描述大致正确但过时或简化（如仅按旧版 stack 行为做对比、忽略 TP 场景差异），或对引用工作的转述有轻微偏差但不影响主论证。

### D3: argumentative_coherence
- dimension_id: D3
- what_to_look_for: 机制收益（显存/计算节省）→ 端到端部署收益（TTFT/TPOT/吞吐/GPU 数）的证据链是否完整，中间是否有未证明的跳跃（如"省显存"直接推出"更多并发用户"，而未论证部署场景确实显存受限）；收益成立的条件在全文各处表述是否一致，abstract/intro 措辞与实验实际覆盖范围是否匹配；混合 linear-attention + recurrent state 方案中，精度损失、切换/迁移路径、失效 regime 是否与其收益主张按同一标准对待；不利证据（低负载下收益消失、精度退化区间）是否正面处理而非回避。
- what_triggers_block: 核心论点依赖的关键跳步完全没有证据（典型：机制层面省了资源，端到端 serving 收益从未被演示或论证，论文却以部署收益为卖点）；或全文条件表述自相矛盾，使部署者无法判断适用边界。
- what_triggers_warn: 主证据链成立但存在未论证的次级推广（如单一模型规模结果泛化到所有规模）；或不利 regime 被提及但未量化、被轻描淡写。

### D4: cross_disciplinary_relevance
- dimension_id: D4
- what_to_look_for: 是否用部署者决策语言报告结果（offered load → P95/P99 TTFT/TPOT/goodput 全曲线、cost/request、requests/GPU-hour、同等 SLO 下所需 GPU 数），而不只是研究语言（精度 delta、显存比例、kernel 加速比）；对部署者关键问题的可操作性：给定我的流量与 SLO，能否读出收益量级、适用负载条件、需要的改造工作量；切换/过渡成本是否被讨论（模型重训或转换、精度重新校准、serving engine 集成、运行时在线自适应是否可用）；跨学科主张是否有对应证据（"降成本"是否给出成本口径量化，"更少 GPU"是否有 SLO 约束下对比）；多硬件、TP/多卡扩展可行性是否有证据或如实披露缺失；冷启动/离线预处理等不可部署约束是否诚实交代。
- what_triggers_block: 论文以部署价值为核心卖点，但部署者最需要的信息——成本量化、operating curve、切换可行性——完全空白且未承认这一 gap；部署者按论文结论行动存在被误导的实际风险。
- what_triggers_warn: 提供了部分部署视角（如单条 latency/cost 对比）但缺完整曲线或切换成本讨论，且论文已如实声明范围与局限（如明确 deployment integration 留给 future work）；或离线/冷启动方案已说明其局限。

### D5: writing_and_structure
- dimension_id: D5
- what_to_look_for: 图/表是否 deployment-readable：latency 类图是否展示分布/分位数（P50/P95/P99）而非仅均值，是否有 offered load 扫描；cost/memory 表格的单位、硬件条件、batch 条件是否可追溯；关键部署信息（收益条件、硬件假设、精度 tradeoff、局限）是否结构上可快速定位；abstract 与 conclusion 措辞是否与实验覆盖一致；limitations 章节是否真实存在且具体；实验 config/环境/版本/复现路径是否可查找；术语与符号是否全文一致。
- what_triggers_block: 结构混乱到无法可靠提取部署决策所需信息（关键图表不可读、单位/条件缺失且无法从正文还原、abstract 主张与正文证据明显不符且无修正路径）。
- what_triggers_warn: 信息可提取但需跨章节拼接；个别图表缺单位/图例说明；limitations 存在但偏笼统，未量化影响范围。

[CONTRACT-ACKNOWLEDGED]
