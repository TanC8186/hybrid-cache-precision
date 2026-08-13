# Phase 2 Domain Review (Peer Reviewer 2)

> 核验方式（read-only）：(1) 全文通读 main.tex；(2) 用 Grep/Read 核对 vendor/vllm 中
> MambaStateShapeCalculator、mamba_ssm_cache_dtype/mamba_cache_dtype 语义、qwen3_5.py 的
> GDN state 形状与 conv state 处理、CacheDType 取值；(3) 用 WebSearch 外部核验 PR #22196、
> PR #43518、ReplaySSM 博客、ARKV 元数据；(4) 用论文自身脚本参数（scripts/bench/analyze_capacity_state.py）
> 数值重算 Table 1 全部 int4 列 gap。未读取任何其他审稿人的 Phase 1/2 文件。

contract_role: domain

## Dimension Scores

### D1: methodology_rigor
score: pass

按 Phase 1 冻结的 scoring plan 打分（我仅评判物理量定义与引擎实现的对应，不评判统计推断本身）：

- **(a) 内存计量逐项对应引擎布局（核验通过）**：GDN temporal state 形状为
  `(num_v_heads=16, head_v_dim=128, head_k_dim=128)`，fp32 下 1,048,576 B/层；conv state
  `(6,144, 3)` 在两种配置下均 bf16（36,864 B/层）。合计 fp32 配置 1,085,440 B/层 =
  论文 §4.1/appendix 的数值；bf16 配置 561,152 B/层。KV 侧 A_q=3,168 B/token（2B，6 层 × 528 B）、
  9B 的 A_f=16,384、A_q=16,384/3.878 与 GQA 层数/head 数自洽。conv state 被计入 cache（未误计权重、未误用 d_model 当 d_inner）。
- **(b) dtype 开关生效有实测交叉验证**：容量增幅随 L 递减（4K +37.6% → 16K +11.5%）恰是
  state 主导短上下文的模型预言；静默 NO-OP 不可能产生该模式。Setup 声明每次 attempt 记录
  resolved dtype 与 block counts。
- **(c) 同引擎同硬件同阶段**：同一 RTX 5090；两个冻结 vLLM commit 均在 Setup 披露且
  跨 package 不混比。
- **(d) 测量来源可追踪**：commit、协议版本、utilization 网格（0.70–0.90）均有 provenance。
- **(e) 质量与显存同一配置**：GSM8K/RULER 走 vLLM kernel 路径；PPL harness 的 chunk 近似
  已披露且被明确降级为 supporting evidence。

不触发 block（核心装置与引擎语义无矛盾）；不触发 warn（近似均已披露、覆盖窄但已声明为局限）。
遗留小问题（不计分）：正文 "9.32 MiB at bf16" 与模型自身参数不一致，见 D2 与 Minor Issues。

### D2: domain_accuracy
score: pass

- **容量模型 vs 引擎实现**：G 的构成（temporal + conv）与
  `MambaStateShapeCalculator.gated_delta_net_state_shape`（mamba_utils.py:247）完全一致；
  我以 A_q=3,168、G_fp32=19,537,920、G_bf16=10,100,736 重算 Table 1 全部 int4 列 gap
  （-2.37%/-3.24%/-0.18%/-1.07%），与论文逐位吻合——模型参数确为 code-derived 且预测确用该参数。
- **增量边界**：正确承认 `mamba_ssm_cache_dtype` 是既有部署开关（PR #22196 "[Model] Mamba
  models - Support FP32 SSM cache"，标题经外部核验属实）与 FlashInfer SSU FP8 checkpointing
  （PR #43518 "WIP: FP8 SSM Cache Checkpointing" 属实，"under development" 表述准确）。不声称
  "首次探索 state 精度"；gap 限定为 joint budget accounting + validation，与 ReplaySSM
  （重算路线）、FP8 checkpointing（序列化）正确区分。
- **prior art 表述**：ReplaySSM "caching inputs instead of state" 与 Tri Dao 博客标题一致；
  TurboQuant dtypes（turboquant_k8v4 等）确存在于该 fork；Quamba/MambaQuant 的 W8A8/W4A8
  定位属实；GDN/DeltaNet/Mamba2 谱系、Jamba/RecurrentGemma/Griffin/Zamba 表述准确。
- **精度谱系声明**："fp8/int8 state 为 future work" 与本 fork `MambaDType =
  Literal["auto","float32","float16","bfloat16"]` 一致，未越界。
- **唯一事实性瑕疵**：§4.1 "9.32 MiB at bf16" 与附录 "G for bf16 is half of G_fp32"
  不精确——conv state 在两种配置下均为 bf16，仅 temporal state 减半，正确值为
  18 × 561,152 B = 9.63 MiB（减幅 48.3% 而非 50%）。Table 1 的预测本身用了正确参数
  （我已数值验证），故该瑕疵不影响任何结果，但文字与计算口径不一致，必须修正。
- 个别 prior art 措辞偏宽（ARKV 归入 quantization+eviction、KVQuant 的 "per-token-head"
  粒度描述），不改变论证 → 列 Minor Issues，不触发 warn。

### D3: argumentative_coherence
score: warn

- **主论点成立**：52/52 方向、joint 交互的实质证据（int4-over-fp16 容量比 2.2451→2.6754
  随 state dtype 变化）、657→904 slots、全部百分比我逐一重算自洽；"代价在哪一侧支付"有交代
  （2B GSM8K -1.0pt、p=0.025 如实保留）。无数字矛盾、无与证据方向相反的耦合声称 → 不触发 block。
- **触发 warn（按 Phase 1 plan 原文"joint 对比缺严格 ablation 但容量模型与质量数字的联合
  证据链仍能自洽支撑结论"）**：selector 的"联合分配优于逐组件独立最优"没有与 independent
  budgeting 的 head-to-head；serving 证据方向为正但 60 细胞无一过 BH-FDR，四配置矩阵未过
  其 frozen Gate 4。论文对此自限得当（"It is not evidence that the selected allocation
  dominates baselines across models or workloads"），但标题 "Joint Precision Budgeting"
  的承诺与已验证证据（joint accounting + 可执行 selector）之间仍有一小段缺口。

### D4: cross_disciplinary_relevance
score: warn

- **普遍性声明 hedge 良好**（不触发 block）：Limitations 明确 scope 限定 GDN 系、Mamba2
  未测、单 GPU、TP 外推标注 "expectation, not a measurement"；结论被翻译成操作语言
  （+247 slots、短上下文 state 主导）。
- **触发 warn：概念铺垫不足**——全文未定义 state 张量本身（形状、为何 per-sequence 固定、
  temporal vs conv state 之分）、chunked scan/SSU/delta-rule 机制、GDN 与 Mamba2 的技术区别
  （decay vs delta rule → 状态形状与精度敏感性的差异）。相邻领域读者（纯 KV 量化背景）需
  自行补课才能判断 "GDN-based hybrids" 泛化边界的含义。[FIELD-NORM UNVERIFIED]（"概念铺垫为
  最低要求"是我 Phase 1 的领域判断，未找到可指认的 venue 政策条文可锚定；故仅 advisory 级，
  不用于提升 severity。）

### D5: writing_and_structure
score: pass

- 图表与正文数字交叉核验一致（Table 1、Fig 1/4/5 caption 与正文吻合）；表格逐组件标注
  dtype（KV int4/fp16 × state fp32/bf16 行轴）；结构符合 MLSys 惯例，related work 双侧覆盖。
- Minor：§3 承诺 "Details of the derivation and the padded block accounting are in the
  appendix"，但附录既无 padded block accounting 也无 A_f/A_q 数值表（fp16 列参数无法从论文
  重建）；§4.1 "tokens per block"（2064/1072、544/288）语义未定义。

## Failure Condition Checks

### F1
fired: false

（依据我的 Dimension Scores：无任何 mandatory 维度 block。）

### F2
fired: false

（依据我的 Dimension Scores：mandatory 三维为 pass/pass/warn，仅 1 个 warn，未达"两个及以上"。）

### F3
fired: false

（依据我的 Dimension Scores：唯一 high-priority 维度 D4 = warn，非 block。）

### F0
fired: false

（依据我的 Dimension Scores：D3 = warn，非全部 pass。）

## Review Body

**Overall Recommendation**: Accept，建议 minor revision。按合同机械规则：我的四个 failure
condition 均未 fired。论文是一份罕见诚实的 measurement paper——其已验证的核心贡献（容量效应 +
联合核算 + 可执行 selector）在领域正确性上完全站得住；主要弱点在于标题级承诺（joint 选择的
增益）未被直接验证、以及若干口径/附录瑕疵，均可在 minor revision 内解决。

**Confidence Score**: 4/5。已独立核验：state 形状与字节数、dtype 开关语义与取值、PR #22196/#43518
标题、ReplaySSM 博客、Table 1 全部 int4 列 gap 的数值重算、全部 8 张图存在性、fig/正文数字一致性。
未能核验：Qwen3.5 HF config 原始数值（模型不在本仓库）、ARKV 内容、serving metric 的内部语义。

**Summary Assessment**（约 210 字）:
本文把 hybrid linear-attention serving 的 recurrent-state dtype 确立为与 attention-KV
bit-width 共享同一显存池的第二精度维度：容量模型 N(L)=M/(A·L+G) 将两维纳入同一分母，
并实现 fail-closed 的 joint selector。核心声明保守且被双 formal runs 验证：112-cell 矩阵
52/52 方向一致、中位增益 15.44%，模型中位绝对残差 1.81%（明确声明是 predictor 而非 bound）。
质量证据诚实分档：2B GSM8K 的 1.0pt 回归（低于预注册 MDE）被保留，no-think RULER 的零差值
被明确限定为"observed agreement"而非等价性。serving 部分如实披露四配置矩阵未过稳定性门。
我核验了其容量装置与 vLLM GDN 路径的逐项对应（state 形状、conv state dtype、dtype 开关、
PR 引用），数字全部对上；领域上唯一的实质缺口是"joint 优于 independent"未做直接 ablation
（论文已自限），以及 9.32 MiB 一处口径错误。总体：领域正确、增量边界准确、值得录用。

**Strengths**:
1. **测量诚实度达到审稿人可验证的水平**：负面结果（183/720 超容差、60 细胞无一生还 BH-FDR、
   MDE/power 显式披露、PPL harness 的 87% chunk-1 对照）原样保留而非转化为 headline；
   这是 systems 领域近年少见的报告纪律。
2. **容量模型与引擎布局逐项对应且经双重验证**：G 的 temporal+conv 构成与
   MambaStateShapeCalculator 一致；int4 列 gap 我可精确重算复现；52/52 方向、两次
   112-cell 复现（max 1.42% per-cell 差）。
3. **增量边界界定准确**：不声称发明 state 精度开关（PR #22196 存在且被引用），把贡献锁定在
   "joint accounting + 统计分辨率披露的验证"，与 ReplaySSM/FP8 checkpointing 两条既有路线正交。
4. **joint 交互的实质证据**：int4-over-fp16 容量比随 state dtype 从 2.2451 升至 2.6754、
   2B/4K 下 +247 slots——共享分母的乘性交互被测量而非假定。
5. **工程可执行性**：selector 输出 cold-restart vLLM 命令、fail-closed、三预算映射 18,000/18,000
   requests 零失败双次复现，对部署者直接可用。

**Weaknesses**:
1. **joint-vs-independent 直接 ablation 缺失（D3 warn 来源）**。问题：标题 "Joint Precision
   Budgeting" 隐含"联合分配优于分别优化"，但 selector 只验证了可执行性与复现性，未与
   逐组件独立最优做 head-to-head；serving 证据方向为正但无一过 BH-FDR。为什么重要：这是
   论文增量声明（joint accounting → 部署决策价值）的最后一环。建议：把标题级承诺收敛为
   "joint accounting"（论文 Limitations 已实质如此），或补一个 independent-budgeting 对照
   设计（哪怕在现有 2B/4K Random 切片上）。无需外部规范锚定——这是论文自身论点-证据一致性问题。
2. **概念铺垫不足（D4 warn 来源）**。问题：state 张量（形状、per-sequence 固定性、temporal
   vs conv 之分）、chunked scan/SSU、GDN 与 Mamba2 的技术区别全文未定义。为什么重要：判断
   "GDN-based hybrids" 泛化边界需要这些概念，而 §2 只给了谱系名。建议：在 §2 或 §3 加一段
   半页的 state 定义（含形状公式与 conv state 的 dtype 处理）。[FIELD-NORM UNVERIFIED]——
   我无法把"概念铺垫是 MLSys 最低要求"锚定到可指认的 venue 政策，故按 #215 纪律降级为
   advisory 并标注。
3. **§4.1 的 9.32 MiB 与模型自身参数不一致**。问题：engine 中 conv state 恒为 bf16
   （`mamba_cache_dtype` 控制，auto → 模型 dtype），仅 temporal state 随 `mamba_ssm_cache_dtype`
   减半；正确 bf16 state 为 18 × 561,152 B = 9.63 MiB（减幅 48.3%）。为什么重要：读者按
   "9.32 = 18.63/2" 会误以为整个 state 减半，且附录 "G for bf16 is half of G_fp32" 与
   Table 1 实际使用的参数矛盾（我已验证预测用的是正确参数）。建议：正文改为 9.63 MiB，
   附录补一句 conv state 不参与 dtype 切换。
4. **附录不完整**。问题：§3 承诺的 "padded block accounting" 与推导细节在附录中不存在，
   A_f/A_q 数值全文未发布（fp16 列的 gap 我因此无法独立复算）。为什么重要：容量模型是
   论文核心装置，"参数可重建"是它区别于 curve-fitting 的卖点。建议：补一张参数表
   （A_f/A_q/G_fp32/G_bf16 × 2B/9B，含 per-token-head scale 开销与 padding 口径）。
5. **serving 不稳定对结论外部效度的讨论不足**。问题：四配置矩阵 Gate 4 失败（183/720 超 10%、
   max 79.61%）被如实披露，但未讨论其机制假设（engine commit？kernel 非确定性？测量协议？）
   及其对 2B/4K Random60 directional evidence 与 controller 结论的含义。为什么重要：
   读者无法判断同协议的 2B/4K 结果是否同样处于不稳定区。建议：在 Limitations/Serving
   variability 中加一段机制性讨论。

**Detailed Comments**:

*Literature Review — Coverage*: 双侧覆盖良好：KV 量化线（KIVI/KVQuant/MiniKV/TurboQuant/
QPruningKV）、eviction 线（H2O/StreamingLLM/SnapKV/HqeKV/RDKV/ARKV）、hybrid 架构线
（Mamba/Mamba2/GDN/DeltaNet/Jamba/RecurrentGemma/Griffin/Zamba/Qwen3.5）、state 精度线
（ReplaySSM/PR22196/PR43518/FlashInfer/Quamba/MambaQuant）。

*Literature Review — Integration quality*: §2 末段的"轴"区分（eviction/offload 改变
哪些 token/层留在内存，state dtype 改变每元素字节数；W/A 量化作用于 compute 路径而非
serving allocator 预算）是全文领域功力最扎实的段落，为 novelty 边界提供了可核验的锚。

*Literature Review — Research gap*: "serving systems expose state dtype without
analyzing it as a precision dimension that multiplies with the KV bit-width decision"
准确、不越界。三处措辞瑕疵：(i) ARKV 被归入 "co-optimizes eviction with quantization"——
我外部核验到该文（arXiv 2603.08727, CCGRID 2026）标题仅涉 memory management，未见量化成分，
请作者确认或改述；(ii) "KIVI and KVQuant established 2–4-bit per-token-head KV
quantization"——KVQuant 为 per-channel(K)/per-token(V) 粒度，"per-token-head" 是 KIVI 的
设计语言，建议合并为 "2–4-bit KV quantization"；(iii) ReplaySSM 置于 "Quantizing SSM
states" 主题句下——其机制是 input caching + replay（重算换内存），非量化。

*Theoretical Framework*: Eq.(1) 是 vLLM 单 pool（KV block 与 mamba cache 同池分配）下的
正确计数；参数化与引擎一致（我以 A_q=3,168、G_fp32=19,537,920、G_bf16=10,100,736 重算
int4 列 gap 逐位吻合；9B 参数 A_q=16,384/3.878、G=26,050,560/13,467,648 同样吻合）。
r_state(L) 随 L 递减与测量方向一致。Eq.(2) fail-closed（缺失 profile 行 → infeasible、
不插值）是稳妥的系统设计。缺口：M 与 gpu_memory_utilization 的换算口径未显式化（建议一行
M = util × total − weights/activations）；附录参数表缺失（见 Weakness 4）。

*Academic Argument Quality — Factual accuracy*: 我抽查的全部数字（37.6/40.6/11.5/14.0%、
2.2451/2.6754、657/904 slots、+247、18.63/24.84 MiB、112=52×2+8、-2.37% gap）与测量/参数
一致。唯一事实性错误：9.32 MiB（应 9.63 MiB，见 Weakness 3）。该错误未进入 Table 1 预测，
不影响任何结果。

*Academic Argument Quality — Argument logic*: "joint" 的实质支撑是共享分母下的交互
（2.245→2.675），链条完整；论文未声称 KV/state 误差耦合（stacking 实验显示无可测附加
代价，且被如实报告为"no measurable additional cost"）。selector 层级的 joint 增益未验证
（Weakness 1）。无谬误、无循环论证（Eq.(2) 的参数独立于 probe 的 token 计数，§3 已说明）。

*Academic Argument Quality — Terminology*: linear attention/SSM/delta-rule/GDN 谱系使用
正确；"Mamba2-style" 与 "GDN-based" 的区分在 Limitations 中成立；"state precision" 与
"state compression" 在 §2 标题下有轻微混用（见 Literature Review iii）。

*Contribution to the Field — Incremental contribution*: 增量真实且边界诚实——不是新 dtype
开关，而是把既有开关纳入与 KV bit-width 的联合核算，并以双 formal runs + 统计分辨率披露
验证。operational rule（短上下文高并发下 +247 slots @2B/4K）对部署者直接可用；容量矩阵与
selector 可复用为 GDN 系 hybrid 的部署工具。

*Contribution to the Field — Positioning*: 与 ReplaySSM（重算路线）、FP8 checkpointing
（序列化）正交而非竞争，定位准确；对 QPruningKV（attention-only 的 token-precision 权衡）
构成自然延伸。

*Contribution to the Field — Overclaiming*: 总体克制（"prediction, not a bound"、
"observed agreement, not equivalence"）。两处可再收紧：abstract "reproducible 112-cell
allocator matrix" 的 reproducible 实为同 contracts/seeds 的 run-stability（正文已说明，
abstract 未限定）；"median gain 15.44%" 是确定性分配器测量的结果（无 CI），建议在
abstract 或 §4.1 注明其确定性来源。

*Missing Key References*:
1. **Nemotron-H**（Dey et al., 2025, arXiv:2504.03624）——Mamba2 系 production hybrid，
   vLLM 官方支持且默认 `mamba_ssm_cache_dtype=float32`（vLLM PR #39032）。论文 Limitations
   声明 "Mamba2-style architectures untested"——Nemotron-H 恰是该家族代表，引用它可将
   泛化边界落到可指认的架构上（本仓库 fork 即有 nemotron_h.py 实现，可直接对标）。
2. **WKVQuant**（Yue et al., 2024, arXiv:2407.05665）——SSM 权重/KV 量化线（Mamba1），
   与 Quamba、MambaQuant 并列时应纳入 "State precision and compression" 谱系。

**Questions for Authors**:
1. §4.1 "9.32 MiB at bf16"：conv state 在 fp32/bf16 两种配置下是否均为 bf16？若是，请将
   正文改为 9.63 MiB 并修正附录 "half of G_fp32" 的表述（Table 1 预测我核验过用的是正确参数，
   结果不受影响）。
2. §4.3 per-layer sensitivity 是否运行在 chunked harness（write-back 模拟）而非 vLLM
   kernel 路径？若是，请在小节或图注中明示——该负结果的解释边界（chunk 128 粒度）目前
   只能靠读者猜。
3. selector 是否有与"逐组件独立最优"的 head-to-head 计划？若无，是否考虑把标题承诺从
   "joint budgeting" 收敛为 "joint accounting"（正文已实质如此）？
4. 四配置矩阵 Gate 4 失败（183/720 超 10% 容差、max 79.61%）的可能机制是什么（engine
   commit、kernel 非确定性、测量协议）？这对同协议的 2B/4K Random60 directional evidence
   与 controller 复现结论有何外部效度含义？
5. TP 论证假设 GDN state 与 attention KV 均按 hidden/heads 均匀 shard；vLLM TP 下 conv
   state 的 shard 维度是 conv_dim（= 2·k_heads·d_k + v_heads·d_v），请确认该均匀性假设在
   实现中成立（或说明 conv state 尺寸可忽略）。

**Minor Issues**:
1. §4.1 "tokens per block"（int4 2B: 2064/1072；fp16: 544/288）语义未定义，且 fp32→bf16
   时该量下降（2064→1072）与"bf16 提升容量"的直觉方向相反——请说明该量是分配器的何种派生量
   （以及为什么 tokens/block 下降而总容量上升）。
2. 附录补参数表（A_f/A_q/G_fp32/G_bf16 × 2B/9B，含 int4 per-token-head scale 开销与
   padding 口径），兑现 §3 的 "padded block accounting" 承诺。
3. Fig 5 caption "bf16 halves the per-block state footprint"——实际减幅 48.3%（conv state
   不参与减半），建议改 "roughly halves"。
4. §2 三处措辞：ARKV 归类、KVQuant "per-token-head"、"Quantizing SSM states" 主题句下的
   ReplaySSM（见 Detailed Comments）。
5. Production hybrids 列表补 Nemotron-H（见 Missing Key References）。
6. Setup 节建议加一行 Qwen3.5-2B/9B config 摘要（层数、kv heads、d_k/d_v、conv kernel），
   使附录参数可独立复算。

## Editorial Decision

editorial_decision=accept
