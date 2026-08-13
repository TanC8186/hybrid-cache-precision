## Contract Paraphrase

**D1 — methodology_rigor（mandatory）**：我作为编辑把关者，对这一维度的理解是——论文的每一个 headline 数字都必须经得起审计级追问。对 serving systems 论文而言，这具体化为五件事：基线是否最新且公平（有没有挑弱基线）、测量是否端到端（TTFT/TPOT/throughput/memory 而非仅 microbenchmark）、比较是否无混淆（同硬件、同引擎、同模型，只变被测变量）、统计是否诚实（seed、方差、噪声地板）、可复现性是否落地（代码/配置/环境/seed 的可获取性）。编辑层面我最警惕的是"漂亮数字建立在混淆比较之上"——这类问题在 systems 顶会一旦被坐实，基本等于 reject 或 major revision，因此这是 mandatory 维度中我权重最高的一项。

**D2 — domain_accuracy（mandatory）**：编辑层面的理解——论文必须被正确安放在 hybrid linear-attention 服务化与精度量化的真实文献版图中。作者对 prior work 的表述（Mamba 系、GLA/RetNet 系、hybrid 架构、KV 量化、recurrent-state 相关量化工作）必须准确，术语使用必须正确，novelty 声明必须与真实 prior art 对得上。一个被坐实的 related-work 失实（例如声称自己是"首个"做某件事，而该事已有发表工作）会摧毁整篇论文的可信度——读者会带着不信任去读其它所有数字。这是编辑眼中一票否决级的硬伤：它不是笔误，而是对领域地图的失察或选择性失明。

**D3 — argumentative_coherence（mandatory）**：编辑层面的理解——论文的核心论点（"在 hybrid linear-attention serving 中对 attention KV 与 recurrent state 做 joint precision budgeting 是正确的设计"）必须从头到尾自洽：问题陈述确实是问题（而非稻草人）、方法确实解决该问题、摘要/引言/结论的每句主张都能在正文中找到对应实验。对我而言 coherence 最尖锐的形态是**论文身份与证据的匹配**：如果论文自我包装为"新系统/新技术"，但证据只支撑"配置空间评估"，或声称的端到端收益没有端到端测量支撑，那就是身份错位——既是论证断裂，也是贡献定位失败，直接决定与 MLSys 录用门槛的距离。

**D4 — cross_disciplinary_relevance（high priority）**：MLSys 本质是跨领域会场（ML × systems），这一维度在编辑层面是录用门槛的组成部分而非加分项：只被一个社区读懂的论文在本会不合格。我的理解是双向可读性与跨领域主张的实证化——systems 读者无需补修 sequence modeling 课程就能理解"recurrent state 的精度敏感性为何与 KV 不同"，ML 读者无需懂服务化细节就能理解质量-系统权衡；凡是跨领域普适性主张（如"对所有 hybrid 架构成立"）必须有实验范围支撑，不能只是修辞。

**D5 — writing_and_structure（normal priority）**：编辑层面的理解——写作与结构是"通常可修、但症状可能指向更深问题"的维度。具体信号：30 秒图形叙事（读者只看图能否得到"问题→方法→收益"闭环）、贡献清单与实验章节的一一映射、图表自包含性、会议规范。5086 词对 MLSys 属紧凑篇幅，紧凑本身不是问题；但若压缩挤掉的是 setup/provenance 等其它维度依赖的细节，写作问题就升级为方法论问题。写作单独不足以 reject（故为 normal priority），但结构混乱到无法定位贡献时，我会把它作为其它维度问题的症状一并记录。

## Scoring Plan

### D1: methodology_rigor
- dimension_id: D1
- what_to_look_for: ①端到端证据链——headline 数字是否有真实 serving 场景的 TTFT/TPOT/throughput/memory 测量，而非仅 kernel 级 microbenchmark；②基线公平性——是否包含最接近的 SOTA（KV 量化与 recurrent-state 处理各自的最新工作），基线是否同硬件/同引擎/同模型、只变被测变量；③"joint"的可分离性——与 uniform precision、per-component budgeting、仅量化 KV、仅量化 state 是否构成完整对照，使 joint 增量可归因；④统计报告——headline 是否为 3 seeds 的 mean±std、是否报告方差与噪声地板；⑤可复现性 affordance——代码/配置/seed/环境版本是否可获得、结论可否被第三方重建。
- what_triggers_block: 端到端缺失——核心收益主张（尤其延迟/吞吐类）只有 microbenchmark 支撑而无任何 serving 级测量；或混淆比较——关键对比在不同硬件/引擎/量化组合上运行，增益无法归因于被测方法本身；或可复现性为零且测量主张不可审计（无代码、无 seed、无环境，且正文不足以重建实验）。
- what_triggers_warn: 单 seed 或未报告方差；把噪声地板内的差异当作增益；基线上限不足（缺最接近的 SOTA 或明显挑弱基线）；对照矩阵缺关键臂（如缺 per-component budgeting，使 joint 增量不可归因）；可复现性 affordance 部分缺失（承诺未兑现、配置不完整）。

### D2: domain_accuracy
- dimension_id: D2
- what_to_look_for: ①related-work 准确性——对 hybrid linear-attention 各架构（Mamba-2/GLA/RetNet/Jamba 类 hybrid 等）及其服务化/量化现状的描述是否与文献事实一致，是否存在已发表的 recurrent-state 量化或混合精度工作被遗漏或弱化；②novelty 声明的真实性——"first/首个"类表述可否被我掌握的领域地图证伪；③术语正确性——recurrent state、linear attention、precision budgeting、数值格式（FP8/INT4/INT8）的使用是否准确一致；④方法所依赖的架构事实是否正确（如 hybrid 模型内存瓶颈究竟在 KV 还是 state 的断言）。
- what_triggers_block: 核心 novelty 声明被已有文献证伪——存在可指认的 prior work 已做 recurrent-state 量化或 joint precision budgeting 而论文未承认；或论文方法依赖的架构属性描述有事实性错误，导致整个设计前提不成立。
- what_triggers_warn: 相邻 prior work 存在但定位错误（被归入不相干类别，或被弱化为"未处理"而实际已处理）；"first"声明实为增量扩展；术语混用导致无法判断实际测量对象（如"recurrent state"与"KV"的指代在章节间不一致）。

### D3: argumentative_coherence
- dimension_id: D3
- what_to_look_for: ①身份-证据匹配——论文自我定位（新系统/新技术/评估研究）与证据强度是否相符：是"新系统"的证据还是"配置开关评估"的证据，这是编辑把关的第一问；②主张-证据闭环——摘要/引言/结论的每个 headline 主张能否追溯到具体实验，尤其端到端收益类主张；③问题-方法闭环——动机部分声称的问题（如 uniform 精度浪费、KV 与 state 敏感性不同）是否被方法实际解决，且该问题是否被测量证明真实存在；④joint 论证的独立性——"joint 优于各组件之和"是否有独立证据（互补性测量），而非循环论证。
- what_triggers_block: 核心机制论证失败——joint budgeting 从未在与自身组件消融（per-component、仅 KV、仅 state）的对照中胜出；或身份错位——包装为新技术/新系统，但全部证据只是参数空间的评估扫描、无可指认的技术贡献；或 headline 端到端收益主张在正文中无对应端到端实验。
- what_triggers_warn: 局部主张超出证据（如单一模型族上的结论被推广为普适结论）；摘要/结论措辞比正文证据更激进；动机命题（如敏感性差异）仅被断言而未被测量；个别贡献 bullet 在实验章节中找不到落点。

### D4: cross_disciplinary_relevance
- dimension_id: D4
- what_to_look_for: ①双向可读性——serving systems 背景读者能否仅凭本文理解 hybrid linear-attention 的 recurrent state 是什么、其精度敏感性与内存角色为何与 KV 不同；②双面实证——建模侧主张（量化下质量保持）与系统侧主张（内存/带宽/延迟收益）是否都有实验支撑，而非一头实证一头修辞；③术语首次出现即定义；④推广声明的范围控制——普适性主张是否被实验范围（模型家族数、架构多样性、工作负载）覆盖。
- what_triggers_block: 单向写作——论文实质只面向单一社区（如大量 SSM 形式化推演而无系统侧桥梁，或纯系统工程叙事但质量/建模侧主张仅在摘要出现），导致 MLSys 读者无法独立评估贡献。
- what_triggers_warn: 关键术语无定义或定义过晚；摘要/引言预设专家知识；跨领域主张（"对所有 hybrid 架构成立"）超出实验范围；评估章节只对单一子领域读者可读。

### D5: writing_and_structure
- dimension_id: D5
- what_to_look_for: ①30 秒图形叙事——是否有 signature figure 能独立回答"问题→方法→收益"，图注是否自包含（单位、基线名、关键条件）；②贡献清单与证据映射——引言贡献 bullet 是否与实验章节一一对应；③5086 词篇幅下的信息完整性——压缩是否挤掉了方法细节与实验 setup（该缺口的实质后果记入 D1）；④结构与会议规范——摘要结构、related-work 定位、表格基线标注是否符合 MLSys 惯例。
- what_triggers_block: 结构性失败导致贡献无法定位或验证——章节/图引用断裂、结果与任何方法描述均无法对应，或组织混乱到无法重建论证链（此情形通常与 D3 身份错位并存，作为症状记录）。
- what_triggers_warn: 无 signature figure 或图形无法独立解读；贡献 bullet 缺失或与实验章节无映射；紧凑篇幅挤掉 setup/provenance 细节；图注不完整、表格基线未标注版本。

[CONTRACT-ACKNOWLEDGED]
