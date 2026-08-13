## Contract Paraphrase

**D1 — methodology_rigor（方法论严谨性）**：我的理解是，D1 要求论文的测量装置本身经得起领域核对——实验设计中对"显存占用、吞吐、质量损失"的操作化定义必须对应到真实引擎的物理量（vLLM 的 KV block 布局与逐层 state 张量、dtype 开关实际作用的张量集合），数据必须来自真实运行路径而非估计值或默认值兜底，统计报告与可复现性（engine 版本、config、seed、开关生效验证）达到 MLSys 领域同行评审门槛。我作为领域审稿人不评判统计推断方法本身（那是 methodology 审稿人的职责），但我负责确认：**被统计的数字所依赖的物理量定义是否领域正确**——量纲错了，统计再漂亮也无效。

**D2 — domain_accuracy（领域准确性）**：论文对 hybrid linear-attention/SSM serving 的一切事实性陈述必须与当前领域证据一致：KV 与 recurrent state 的容量模型（block 大小、per-layer/per-head 维度、d_state/d_inner、dtype 字节数）与引擎布局相符；对 prior art 的表述准确——KV 量化线（KIVI、KVQuant、MiniKV、TurboQuant）、state 压缩线（ReplaySSM、Quamba、MambaQuant）、既有 dtype 开关路径（vLLM `mamba_ssm_cache_dtype`、FlashInfer SSU）各自做了什么、代价是什么、适用范围是什么，必须如实转述；术语谱系（linear attention / SSM / delta-rule / GDN、state vs conv state）使用正确。这一维度直接决定我对其"增量（increment）声明"是否成立的判断。

**D3 — argumentative_coherence（论证一致性）**：核心论点——joint precision budgeting 优于分治——必须内部自洽：证据（容量、质量、吞吐数字）要真正支撑"联合"的实质增益，而非只是两个独立最优的简单相加；若主张 KV 误差与 state 误差存在跨组件耦合从而 justify 联合分配，其机制证据必须与结论方向一致；全文数字与立场不得相互矛盾（总节省不能超物理上限，对"哪一组件更精度敏感"的判断前后须一致）。从领域角度我特别盯一条：**precision 是零和资源**，任何"两头都省"的叙述必须说明代价在哪一侧支付。

**D4 — cross_disciplinary_relevance（跨学科相关性）**：framing、定义与结论含义要让相邻领域读者（纯 KV 量化、纯 SSM、通用 serving 背景）能理解并采信：hybrid 特有概念（d_state、chunked scan、gating、conv state、SSU）需有最低限度铺垫；跨架构/跨领域主张（对 Griffin/RecurrentGemini/Jamba/Zamba 等不同 hybrid 家族的适用性、对训练侧或硬件协同的影响）必须有证据支撑或明确限定，不能把单一架构上的观察包装成普遍规律。

**D5 — writing_and_structure（写作与结构）**：稿件组织、论述清晰度、图表质量与 MLSys 规范符合标准。领域准确性视角下，我要求图表**可被独立核验**：显存分解图必须正确区分 attention-KV 与 state 的逐层贡献、表格必须逐组件标注 dtype 配置（只写 "int4" 而不说明作用于哪个张量，等于不可核验）、公式符号与引擎张量形状的对应关系可追踪。

## Scoring Plan

### D1: methodology_rigor
- dimension_id: D1
- what_to_look_for: (a) 内存计量是否逐项对应引擎布局：KV 部分按 block 大小 × 层数 × head 数 × dtype 计算，state 部分按 per-layer d_state × d_inner × dtype 计算，且 conv state、SSM 权重是否被误计入 cache；(b) 基线中的 dtype 开关（`kv_cache_dtype`、`mamba_ssm_cache_dtype`）是否确认真正生效——有无日志/实测内存交叉验证，而非假定生效（静默 NO-OP/fallback 是领域已知陷阱）；(c) 对比是否同引擎、同硬件、同 decode 阶段与长度谱下进行；(d) 测量来源可追踪（engine 版本、config、seed、运行 provenance）；(e) 质量评估（perplexity/长程任务）的精度设置与显存数字对应同一配置。
- what_triggers_block: 核心实验装置的物理量定义与引擎实现语义矛盾——例如 state 内存按错误张量形状计算（d_model 当 d_inner、或把 GDN delta-rule 状态按未秩约简的 n×d×d 计）、或声称的 dtype 开关在该引擎路径上根本不生效（静默退化），使全部数字建立在错误量纲或虚假配置之上。
- what_triggers_warn: 装置定义正确但存在已披露的近似（忽略小项、估算代替实测并注明）；workload/硬件覆盖窄（单一模型族或单一 GPU）且未声称为局限。

### D2: domain_accuracy
- dimension_id: D2
- what_to_look_for: (a) 容量模型与引擎实现逐项对照：KV（block 大小、层数、head 数）与 state（d_state、d_inner、层数、dtype 字节、conv state 是否计入）与实际布局一致；(b) 与既有 dtype 开关工作的增量边界：是否承认 vLLM `mamba_ssm_cache_dtype`、FlashInfer SSU、各 KV 量化方案已覆盖的部分，并精确说明自己多做了什么（novelty 边界不夸大）；(c) 精度谱系声明范围：fp8e4m3/e5m2、int8、int4 等在不同组件上的质量-显存特性是否与领域已知证据一致（如 state 对低精度普遍比 KV 更敏感的既有结论，被正确引用或提供对证实验反驳）；(d) prior art 表述准确性：上述各线工作的方法与结论如实转述；(e) 术语谱系：linear attention / SSM / delta-rule / GDN 的关系、chunked scan、state 与 conv state 的区分使用正确。
- what_triggers_block: (1) 核心公式或容量模型与引擎实现语义矛盾；(2) 增量被实质夸大——如声称"首次探索 state 精度"而既有的公开 dtype 开关未被正确界定关系，或把 prior art 的结论转述成相反意思；(3) 把 state 精度与 KV 精度的质量特性等同处理，与领域已知证据直接冲突且无对证实验。
- what_triggers_warn: 个别 prior art 表述不精确但不改变论证（简化转述、细节记错）；精度谱系覆盖不完整（如未覆盖 fp8 谱系）但已解释；对某架构家族的推断超出证据但已限定语气。

### D3: argumentative_coherence
- dimension_id: D3
- what_to_look_for: (a) "joint" 的实质性：联合预算结果是否显著区别于并优于两组件独立最优之和——有无 direct ablation（independent per-component budgeting vs joint）；(b) 跨组件误差耦合主张（若声称 hybrid 层内 KV 误差与 state 误差传播放大）是否有与其方向一致的机制证据；(c) 数字自洽：各图表之间的节省率、质量损失、吞吐能否互相印证（各部分节省相加不超物理上限）；(d) 立场一致：全文对"哪一组件更精度敏感"的判断前后一致，且预算分配结论与之一致（代价在哪一侧支付有明确交代）。
- what_triggers_block: 核心论点的证据与其主张矛盾——如论文自己的数据显示 joint 与 independent 无差异（声称的耦合效应不可见），或关键数字互相矛盾（节省超出理论总量、前后对精度敏感组件的判断反转且无解释），中心论点失去支撑。
- what_triggers_warn: 主论点成立但次级论断证据松散（hardware 外推、端到端吞吐提升归因未排除混杂因素）；joint 对比缺严格 ablation 但容量模型与质量数字的联合证据链仍能自洽支撑结论。

### D4: cross_disciplinary_relevance
- dimension_id: D4
- what_to_look_for: (a) 对相邻读者（纯 KV 量化、纯 SSM、通用 serving 背景）关键概念是否有最低限度定义（d_state、chunked scan、SSU、delta-rule 的 vk 状态、gating/conv state）；(b) 结论含义是否被翻译成相邻领域语言（如对 KV-only 读者说明 state 在 hybrid 显存中的占比与增长律）；(c) 跨架构/跨领域主张（Griffin/RecurrentGemini/Jamba/Zamba 等不同家族、不同硬件、训练侧影响）是否有证据或明确限定。
- what_triggers_block: 把单一架构/单一硬件的结论表述为跨架构普遍规律——例如仅在一个 delta-rule 或 Mamba 类 hybrid 上验证，却断言"hybrid serving 的精度分配应遵循 X"，而不同家族（state 尺寸、decay 机制、attention:recurrent 层比、conv state 有无）显然会改变预算计算，构成对相邻领域读者的实质误导。
- what_triggers_warn: 普遍性声明已有 hedge 但概念铺垫不足（相邻读者需自行补课才能判断结论边界）；跨领域含义（训练侧、硬件协同）以推测语气提出但无证据。

### D5: writing_and_structure
- dimension_id: D5
- what_to_look_for: (a) 图表可独立核验：显存分解图正确区分 attention-KV 与 state 的逐层贡献且合计与总显存一致；(b) 表格逐组件、逐 dtype 标注配置；(c) 公式符号与引擎张量形状对应可追踪；(d) 结构符合 MLSys 惯例（背景/方法/实验清晰，related work 同时覆盖 KV 量化与 state 压缩两侧）。
- what_triggers_block: 关键图表/表格与正文数字矛盾（图中 headline 数字与正文/表格不一致），或配置表格缺失组件级 dtype 标注使核心结果无法被独立解读与核验。
- what_triggers_warn: 图表密集难读、术语定义滞后、结构偏离 venue 惯例但不影响理解。

[CONTRACT-ACKNOWLEDGED]
