# MLSys 论文 ARS 最大强度审稿报告

> 审稿日期：2026-08-10  
> 审稿对象：`paper/mlsys2026/main.tex` 与当前 10 页 `main.pdf`  
> 论文题目：*Joint Precision Budgeting Across Attention KV and Recurrent State in Hybrid Linear-Attention Serving*  
> 目标会议：MLSys  
> 模式：Academic Research Suite `academic-paper-reviewer/full`，EIC + R1 Methodology + R2 Domain + R3 Systems/Deployment Perspective + Devil's Advocate  
> 最终决定：**Reject — Resubmission Encouraged**  
> 审稿属性：只读审稿；本报告不修改论文正文或实验数据。

---

## 0. 审稿边界与诚实声明

1. 本轮评估以当前论文实体、现有实验结果、图表和作者已经披露的局限性为准，不把计划中的实验视为已完成。
2. 本轮没有启用外部跨模型审稿，也没有把未公开论文发送给第三方模型或 API。
3. 由于当前会话在建立审稿角色之前已经接触过论文，无法宣称实现了物理意义上的 paper-blind Phase-1 预承诺。ARS sprint contract 在本轮作为评判维度与机械决策规则使用，而非作为真正盲审的证明。
4. 数值评分用于结构化比较，不是录用概率。任何 mandatory dimension 的 blocking issue 均优先于加权平均分。
5. 论文主动报告失败、宽置信区间、无显著结果、相同 seeds 的第二次运行以及机制混杂，这是明显优点；本报告不会为了提高“故事性”删除这些负面证据。

---

## 1. 领域定位与稿件类型

| 项目 | 判定 |
|---|---|
| 主领域 | ML Systems / LLM inference serving |
| 次领域 | Hybrid linear attention、GDN/SSM recurrent state、KV-cache quantization、GPU memory allocation、SLO-driven serving |
| 研究范式 | 解析容量建模 + 确定性 allocator 测量 + 配对质量评估 + serving workload evaluation |
| 当前最准确的稿件类型 | **Hybrid serving memory characterization / measurement study** |
| 论文声称的类型 | **Joint precision budgeting system study** |
| 核心张力 | 当前证据支持“两个精度维度的联合核算与测量”，尚不支持“一个真正执行联合预算决策的系统” |
| 稿件成熟度 | 统计披露较成熟；系统机制、比较基线、因果归因与泛化不足 |

### 1.1 当前论文试图建立的主张链

1. Hybrid linear-attention 模型的 serving memory 不只有 attention KV，还包括固定的 per-sequence recurrent state。
2. KV dtype 与 state dtype 是两个可组合的精度维度。
3. 由模型配置和 vLLM page layout 可推导容量模型。
4. 将 GDN state 从 fp32 改为 bf16，可在短上下文、高并发、memory-bound 条件下增加容量。
5. 质量下降总体可控，但存在任务和模型规模差异。
6. Random60 overload region 出现跨两次相同契约运行的方向性 goodput 改善。

### 1.2 当前证据实际能够支持的边界

- 能较强支持：在指定 vLLM commit、RTX 5090、Qwen3.5-2B/9B 和已测布局上，state dtype 会显著改变 allocator 可用容量。
- 能有限支持：容量模型能够近似解释已测七个 cell，但仍受 page/block rounding 影响。
- 能支持：2B GSM8K 上 state-bf16 存在约 1.0 percentage-point 的配对回归；9B 未检测到同方向回归。
- 只能方向性支持：Random60 overload region 的 goodput delta 在相同 contracts/seeds 的第二次运行中同向出现。
- 不能支持：跨硬件、跨模型家族、跨 TP 配置的普遍规律。
- 不能支持：state-bf16 通过某一个已隔离的机制导致 serving 改善。
- 不能支持：论文已经实现了一个 SLO/quality-aware joint precision budgeting controller。

---

## 2. 编辑决定

## Reject — Resubmission Encouraged

拒稿的主要原因不是写作不清楚，也不是图没有矢量化，而是论文的系统身份与证据深度不匹配。当前工作最扎实的部分是容量核算、allocator 测量与诚实的统计边界；最薄弱的部分是：

1. 标题和贡献使用 **joint precision budgeting**，但论文没有预算优化问题、可执行选择策略、runtime controller 或新的 allocator 机制。
2. 核心数学与文字存在三处会直接损害可信度的矛盾：容量变量定义、lower-bound 方向、共享内存解释。
3. serving 主张在 60 个 cell 中没有一个通过 BH-FDR；机制归因明确未完成；现有图无法把方向性现象提升为系统因果结论。
4. 只有一张 RTX 5090、一个 Qwen3.5 架构家族、两个模型规模；没有 TP=2/4 或其他 hybrid/SSM 架构。
5. 没有足以代表真实系统竞争关系的 baseline suite，也没有 quality–capacity–SLO 的联合决策面。

这些问题要求新增系统实现和实验，而不是仅靠文字修订，因此编辑层面选择 Reject/Resubmit，而不是普通 Major Revision。

---

## 3. ARS mandatory-dimension 判定

| 维度 | 结果 | 理由 |
|---|---|---|
| D1 Methodological rigor | **Warning** | 容量 probe 对固定引擎具有确定性，但 serving 只有 `n=3` seeds/cell；RULER 分辨率很低；机制对照未执行 |
| D2 Domain and mathematical accuracy | **Block** | `C(L)` 定义冲突；预测值高于测量值却称模型为 lower bound；“do not share memory”与同一 GPU pool 矛盾 |
| D3 Argumentative coherence | **Block** | “joint budgeting”没有对应的预算决策系统；标题、贡献、图形叙事和实际证据类型不一致 |
| D4 Systems relevance and external validity | **Warning** | 问题重要，但硬件、模型家族、TP、workload 和 state precision spectrum 均较窄 |
| D5 Writing and disclosure quality | **Pass** | 失败、置信区间、MDE、power、BH-FDR、相同 seeds 第二次运行和局限性均有明确披露 |

根据 full reviewer contract，mandatory dimension 出现 block 时，最终决定不得为 Accept/Minor；考虑到修复需要新增系统与实验，落点为 Reject/Resubmit Encouraged。

---

## 4. 三个必须立即修正的硬性问题

### CRITICAL-1：容量变量定义不一致

**位置**：`paper/mlsys2026/main.tex:76–78` 与 `:172–182`。

引言贡献写成：

\[
C(L)=\frac{M}{A L+G},
\]

正文 Eq. (1) 写成：

\[
C(L)=\frac{L M}{A L+G}.
\]

前者是并发序列数，后者是总 token capacity。两者不应共用同一符号。建议统一定义：

\[
N(L)=\frac{M}{A L+G},\qquad
T(L)=L\,N(L)=\frac{L M}{A L+G}.
\]

**影响**：这是论文最核心的模型。变量定义错误会使所有 capacity、concurrency 和 ratio 图的语义受到质疑。

**验收标准**：标题、摘要、贡献、正文公式、表格、图轴、附录推导和结论均只使用统一后的 `N(L)` 与 `T(L)`，不再混用。

### CRITICAL-2：“conservative lower bound”方向错误且缺少证明

**位置**：摘要、`main.tex:93–97`、表 1 caption、`main.tex:264–272`、结论。

在 int4 cells 中，测量 ratio 低于 architecture-derived prediction。若 prediction 大于 measurement，则预测模型不是测得容量收益的 lower bound；它在这些 cell 上是偏高的理想化预测。四个 cell 同向、sign test `P=0.0625` 也不能构成数学下界或统计下界证明。

**允许的表述**：

> Across the four tested int4-KV cells, measured gains were 0.18%–3.24% below the idealized architecture-derived prediction, consistent with discrete allocator rounding.

**禁止的表述**：

- “the capacity model is a conservative lower bound”
- “all four signs justify a lower bound”
- 把已测的确定性值外推为未测上下文或硬件的保证

**验收标准**：全文删除 lower-bound 主张，改为 measured-vs-predicted residual，并明确外推范围。

### CRITICAL-3：共享内存解释自相矛盾

**位置**：`main.tex:67–70` 与 `:277–280`。

前文正确说明 attention KV 和 recurrent state 从同一 GPU memory pool 分配；后文却称二者“do not share memory”。正确解释是：它们共享总预算，但在 per-sequence footprint 中形成两个可加项 `A·L` 与 `G`。组合 ratio 可以按连续切换的比值代数分解，但不能归因于“不共享内存”。

**验收标准**：所有 schematic、公式解释和正文统一画成同一共享池内的两个 footprint component；不得画成两个独立显存池。

---

## 5. Reviewer Configuration

### EIC：MLSys Area Chair

- 专长：ML systems、inference serving、系统论文贡献定位。
- 重点：是否有新的系统抽象或机制；30 秒内图形叙事能否回答“问题、方法、收益”。
- 盲区：不深挖每个统计检验，由 R1 负责。

### R1：Serving Benchmark and Statistics Reviewer

- 专长：在线 serving benchmark、paired design、multiple comparisons、reproducibility。
- 重点：seed 语义、MDE/power、BH-FDR、SLO boundary、run-to-run stability 与独立复现的区别。

### R2：Hybrid Linear-Attention / SSM Serving Reviewer

- 专长：GDN、Mamba/SSM、KV/state quantization、vLLM/FlashInfer memory layout。
- 重点：模型和公式准确性、state compression prior art、架构泛化、贡献是否只是已有 dtype flag 的评估。

### R3：Production Inference and Capacity Planning Reviewer

- 专长：GPU capacity planning、SLO、cost efficiency、production deployment。
- 重点：结果能否支持部署决策；是否有完整 load curve、P95 tail latency、成本和多硬件证据。

### Devil's Advocate

- 任务：构造最强拒稿叙事，检查 overclaim、选择性展示和替代解释。
- 核心挑战：这是一个“existing flag + bookkeeping equation”研究，还是一个新的系统？

---

## 6. EIC 独立审稿报告

**Recommendation**：Reject — Resubmit Encouraged  
**Confidence**：4/5  
**启发式加权分**：64/100

### 6.1 优点

1. **问题真实且容易被 attention-only 假设忽略。** Hybrid 模型同时承担随 token 增长的 KV 和固定 per-sequence state，容量规划确实需要重新审视。
2. **容量 headline 具有实际意义。** 2B/4K、int4 KV 下并发序列由 657 增到 904，增加约 247 slots，属于可被部署人员理解的结果。
3. **对负面证据披露充分。** 论文没有把 RULER 宽区间、serving FDR 失败或相同 seeds 第二次运行包装为强复现。
4. **论文已经形成容量—质量—serving 三层结构。** 这为后续发展成完整系统论文提供了骨架。

### 6.2 主要问题

#### EIC-W1：系统贡献不成立（Critical）

论文没有解决一个形式化的 budget selection problem，也没有实现根据 workload、SLO 与 quality budget 自动选择 KV/state dtype 的组件。现有工作是对四个静态配置的测量，而不是“预算系统”。

**建议**：实现 joint budget selector，至少定义：输入、目标函数、约束、候选配置、选择算法、切换时机和运行时开销；否则修改标题与定位为 memory accounting/characterization。

#### EIC-W2：主结果没有真实 baseline 竞争关系（Major）

当前比较主要是 fp32/bf16 state 与 fp16/int4 KV 的内部 2×2 对照。顶会系统论文需要说明它相对现有 serving 策略带来什么新 operating point。

**建议**：加入 KV-only、state-only、joint、full precision，以及可实现的 offloading、state compression、KV compression/eviction 代表方案；统一 workload 与 SLO。

#### EIC-W3：图形篇幅分配与贡献层级相反（Major）

当前 8 张图中，大量主文面积用于 null results、harness limitation 和 per-seed diagnostics，而缺少一张系统 hero figure、完整 capacity phase diagram、Pareto frontier 和机制隔离图。

### 6.3 给作者的问题

1. “Joint precision budgeting”具体执行了什么决策，而不仅是评估四个配置？
2. 如果没有 selector/controller，为什么不将论文定位为 hybrid serving memory characterization？
3. 一个生产部署者读完论文后，如何根据 workload 与质量约束选择配置？

---

## 7. R1 Methodology 独立审稿报告

**Recommendation**：Major Revision；在 MLSys 当前轮次不建议录用  
**Confidence**：4/5  
**启发式加权分**：62/100

### 7.1 优点

1. GSM8K 使用配对 seeds，并同时报告 effect、CI、`p`、MDE、power 与 Cohen's `d`。
2. serving 把失败计入分母，披露完整 60-cell multiple-comparison family。
3. 第二次 formal run 被正确降格为 run-stability，而不是 independent replication。
4. capacity probe 对固定 build/config 的 deterministic allocator 有清楚记录，包括 resolved dtype、block count 与配置 hash。

### 7.2 主要问题

#### R1-W1：serving 推断分辨率不足（Critical for serving claim）

每个 cell 只有 3 seeds，60 个 cell 无一通过 BH-FDR；最接近的 `q` 仍为 0.052/0.066。13/13 Random60 overload cells 同向是有价值的描述性模式，但不能替代预注册的 family-level 推断。

**建议**：预先指定少量 primary endpoints，例如 P95 TTFT、P95 TPOT、goodput 和一个主 SLO；扩大独立 seeds 或独立 workload traces；把其余 grid 作为 secondary/exploratory。

#### R1-W2：RULER 无法支持等价或无损结论（Major）

五个 cell、每 cell 20 samples、3 dataset seeds，CI 可宽至数十 points；default thinking + max 256 tokens 还引入截断风险。

**建议**：完成 no-think 复测；增加样本量；预先定义 non-inferiority/equivalence margin。如果做不到，只能保留“未检测到差异，实验分辨率不足”。

#### R1-W3：容量模型验证点过少（Major）

七个 deterministic cells 可以验证实现一致性，但不足以证明模型在连续 context range、不同 GPU memory budgets 或不同 allocator layouts 上的泛化。

**建议**：增加 1K/2K/8K/32K 等上下文、不同 memory utilization、至少另一硬件或显存规格，并报告 prediction residual。

#### R1-W4：机制仍然是 confounded aggregate（Major）

state dtype 同时改变 state bytes、tokens/block 与 allocated blocks，serving delta 无法归因。

**建议**：执行 fixed-block-count、fixed-bytes、fixed-concurrency 对照，并采集 HBM bytes、kernel time、queueing 和 allocator metrics。

### 7.3 给作者的问题

1. 哪个 serving endpoint 是预先指定的 primary endpoint？
2. 第二次运行除时间不同之外，有没有独立 trace、独立 seeds、独立机器或独立 build？
3. 如何区分容量增加、内存带宽变化和 page rounding 对 goodput 的贡献？

---

## 8. R2 Domain 独立审稿报告

**Recommendation**：Reject — Resubmit Encouraged  
**Confidence**：4/5  
**启发式加权分**：60/100

### 8.1 优点

1. 正确识别了 hybrid model 与 attention-only transformer 在 serving state 上的结构差异。
2. `A·L + G` 的基本 footprint 分解具有解释性，且与 state-dtype/KV-dtype 两个配置维度相对应。
3. 论文主动区分 state precision、weight/activation quantization、KV quantization 和 eviction/offloading。
4. 对 GDN scope、Mamba2 未测试和 fp8/int8 state 未测试有明确限定。

### 8.2 主要问题

#### R2-W1：核心公式与术语错误（Critical）

`C(L)` 混用 sequence capacity 和 token capacity；lower bound 方向错误；同一 memory pool 被描述为“不共享内存”。三者均位于核心贡献链上。

#### R2-W2：novelty 仍可能只是现有配置开关的经验研究（Critical）

`mamba_ssm_cache_dtype` 已存在；论文没有新的低精度数值格式、量化算法、allocator、kernel 或 runtime policy。联合“核算”有价值，但距离联合“预算系统”仍有空缺。

**建议**：提出并实现 SLO/quality-constrained policy，或把贡献严格限定为首个 architecture-aware measurement study。

#### R2-W3：架构泛化不足（Major）

只有 Qwen3.5 GDN hybrid。Mamba2-style recurrent state、RecurrentGemma/Griffin、Jamba/Zamba 或其他 serving layout 未验证。当前不能声称 hybrid linear-attention 的普遍规律。

#### R2-W4：precision spectrum 不完整（Major）

state 只正式比较 fp32/bf16；fp16 主要是 smoke test，fp8/int8 未测。若题目强调 precision budgeting，应至少解释候选空间为什么只有一个 binary switch。

### 8.3 给作者的问题

1. 论文的新系统 artifact 是什么？模型、策略、runtime、allocator 还是 benchmark protocol？
2. 如果 `mamba_ssm_cache_dtype` 是既有能力，论文新增的机制和代码路径在哪里？
3. 为什么选择 fp32/bf16，而不是构造完整 state precision frontier？

---

## 9. R3 Systems/Deployment 独立审稿报告

**Recommendation**：Reject — Resubmit Encouraged  
**Confidence**：4/5  
**启发式加权分**：57/100

### 9.1 优点

1. 把 token capacity 转换为 657→904 concurrent sequences，比只报告内存字节更有部署意义。
2. serving protocol 同时考虑 TTFT、TPOT、goodput/offered 和 failure denominator。
3. Random60 与 ShareGPT300 暴露了 workload sensitivity，没有只选最有利 workload。

### 9.2 主要问题

#### R3-W1：缺少完整 operating curves（Critical）

当前图展示 delta 和 threshold boundary，而不是 offered load→P95 TTFT/TPOT/goodput 的完整曲线。部署者无法看到 saturation point、queueing knee 或 tail-latency trade-off。

#### R3-W2：缺少成本与资源维度（Major）

没有 cost/request、requests/GPU-hour、GPU count reduction、energy 或显存利用率图。容量增加是否转化为成本收益仍未量化。

#### R3-W3：单 GPU 证据不足（Major）

没有 A100/H100/L40S/另一消费级 GPU，也没有 TP=2/4。RTX 5090 的 allocator granularity 和 bandwidth 特征可能不代表数据中心部署。

#### R3-W4：没有 workload-aware policy（Critical）

结果已经证明 Random60 与 ShareGPT 的边界不同，但系统不会根据 workload 切换配置。这使论文错过了最自然的系统贡献。

### 9.3 给作者的问题

1. 在什么 offered-load 区间选择 joint precision 才值得承担质量风险？
2. 是否存在负载下降后恢复高精度的 runtime transition，切换成本是多少？
3. 该方法能减少多少 GPU 数量或 GPU-hour，而不是只增加理论 capacity？

---

## 10. Devil's Advocate 报告

### 10.1 最强反方论证

这项工作并没有提出新的 quantization method、state representation、allocator、kernel、scheduler 或 serving controller。它打开 vLLM 已经存在的 state dtype 配置，测量四种静态组合，并用 `A·L+G` 解释显存占用。容量提升主要来自 bf16 相对 fp32 的字节减半，这是预期结果；block rounding 使实测比理想化比值略有偏差。质量证据中，2B GSM8K 已经出现显著回归，RULER 又缺乏足够统计分辨率。serving 结果在 60 个比较中没有一个通过 BH-FDR，且 ShareGPT 方向不稳定；论文自己承认没有隔离容量、带宽和 allocator granularity。因而最强结论只是：“hybrid serving 时不要忘记 recurrent state 的字节数。”这一结论适合工程说明、measurement note 或短文，但不足以支撑题目中的 joint precision budgeting system。当前精美重画只能提高可读性，不能补足系统 novelty 与 causal evidence。

### 10.2 CRITICAL issues

1. **Flag-flip criticism 未被击破**：没有新系统机制。
2. **核心数学叙述错误**：capacity symbol、lower bound、shared-memory explanation。
3. **标题过度承诺**：budgeting 暗示有优化/分配决策，正文只有配置 sweep。
4. **serving headline 不具 family-wise statistical support**：0/60 cells 通过 BH-FDR。

### 10.3 被忽略的替代解释

- goodput 改善可能来自 block count 增加，而不是 state bandwidth 降低。
- 短请求 Random60 的结果可能来自特定 queueing regime，不能迁移到长请求。
- int4 KV 与 state-bf16 的组合优势可能只是固定 32 GB budget 下的 allocator rounding artifact。
- 2B GSM8K 回归可能意味着 state precision 对某些 reasoning path 更敏感，而现有 PPL/RULER protocol 无法检测。

### 10.4 “So what?”测试

若没有一个 selector 或 deployment rule，用户仍需要手工判断何时切换 state dtype。论文必须回答：它改变了 serving system 的什么决策，而不仅是提醒工程师多算一项内存。

---

## 11. 跨审稿人共识与分歧

### 11.1 全体共识

1. 问题真实：hybrid models 的 recurrent state 是 attention-only capacity analysis 遗漏的维度。
2. 容量 measurement 与配置记录是当前最强证据。
3. 统计披露和负面结果保留值得肯定。
4. 当前“joint budgeting”系统身份没有成立。
5. 需要完整主结果图、机制图和 deployment operating curves。
6. 需要新增实验，而不是只重画。

### 11.2 轻微分歧

- R1认为若把 serving 完全降格并重新定位，稿件可达到 Major Revision；EIC/R2/R3认为 MLSys 顶会门槛下仍需要 Reject/Resubmit。
- 编辑裁决：采用更严格结论，因为关键修改涉及新实现、多硬件/TP、baseline 与因果对照，不可能通过普通文字返修完成。

---

## 12. 当前八张图逐图裁决

| 当前图 | 当前作用 | 主要问题 | 主文裁决 | 可复用内容 |
|---|---|---|---|---|
| Fig. 1 Block granularity | tokens/block 与 allocated blocks | 诊断性强，不能承担第一张主图；没有问题和系统 insight | 移附录 | block arithmetic、原始数量 |
| Fig. 2 Precision budget | 简化架构 + measured/predicted ratio | schematic 像通用方框图；没有真正 dataflow；七点比较不是 phase diagram | 完全重构 | 共享池概念、ratio 数据 |
| Fig. 3 GSM8K deltas | 配对质量 forest plot | 可读但孤立；没有与 capacity/serving 联合形成决策图 | 合并进 Pareto/guardrail | 配对 delta 与 CI |
| Fig. 4 PPL + RULER | 辅助质量证据 | 主视觉由宽 CI/null result 主导；RULER 无 equivalence 能力 | 附录或紧凑辅助面板 | PPL delta、RULER uncertainty |
| Fig. 5 Harness boundary | 暴露 chunk approximation | 属于方法限制，不是核心贡献；放主文会削弱主线 | 移附录 | chunk=1 vs 128、stacking ablation |
| Fig. 6 Per-seed GSM8K | seed 轨迹 | 与 Fig. 3 重复同一证据 | 移附录 | 透明复现证据 |
| Fig. 7 Layer sensitivity | 单层 bf16 null result | 使用双栏面积证明“没有显著层”；信息收益低 | 移附录 | 36 tests 与校正结果 |
| Fig. 8 Serving | overload delta + boundary grid | 没有完整 load/SLO curves；0/60 FDR；grid 主要为零 | 现形式放弃 | 描述性 delta、boundary 数据 |

**总裁决**：0 张原样保留；约 3 组数据可在新图中复用；其余进入附录或补充材料。

---

## 13. 对照 MLSys 论文得到的图形证据标准

以下论文的价值在于图形承担明确审稿任务，而不只是视觉风格：

- [FlashInfer, MLSys 2025](https://proceedings.mlsys.org/paper_files/paper/2025/file/dbf02b21d77409a2db30e56866a8ab3a-Paper-Conference.pdf)：Figure 1 直接解释 compile-time specification、runtime scheduling、KV layout 与 kernel 的系统关系，后续再给 kernel 和 end-to-end performance。
- [QServe, MLSys 2025](https://proceedings.mlsys.org/paper_files/paper/2025/file/fbe2b2f74a2ece8070d8fb073717bda6-Paper-Conference.pdf)：第一张图给真实 baseline 下的吞吐/成本 headline，后续 GPU execution schematic 解释 dequantization overhead。
- [Rethinking KV Cache Compression, MLSys 2025](https://proceedings.mlsys.org/paper_files/paper/2025/file/26289c647c6828e862e271ca3c490486-Paper-Conference.pdf)：跨 prompt length、KV length、batch、TP 和 output length 展示边界与副作用。
- [MorphServe, MLSys 2026](https://proceedings.mlsys.org/paper_files/paper/2026/hash/8144a9d62e506af0fcdeac0e456b2710-Abstract-Conference.html)：把 runtime mechanism、SLO violations、P95 TTFT、generation quality 与 Pareto frontier 联系起来。
- [From Tokens to Layers, MLSys 2026](https://proceedings.mlsys.org/paper_files/paper/2026/hash/c0f460c6d63599ea870ba9db63dc96a9-Abstract-Conference.html)：用调度机制、memory traffic、TTFT–TBT Pareto 和 energy 形成闭环。

由此，本论文需要的不是更多独立小图，而是一套五层证据架构：

1. Problem/Insight/System hero figure。
2. Capacity phase diagram 与模型验证。
3. Quality–capacity/serving Pareto 或可行域。
4. End-to-end load/SLO curves 与真实 baseline。
5. Mechanism isolation 与 bottleneck breakdown。
6. 若篇幅允许，再增加 generality/scaling 图。

详细 draw.io 图形合同与提示词见：`docs/notes/mlsys-drawio-figure-spec-prompts-2026-08-10.md`。

---

## 14. 修订路线图

## P0：提交前阻断项

### P0-1 决定论文身份

二选一：

1. **系统路线（推荐）**：实现 joint precision budget selector/controller。
2. **表征路线**：改题目、摘要和贡献，只声称 architecture-aware memory accounting/characterization。

系统路线建议形式化为：

\[
\max_{q_{kv},q_{state}} \; \text{Goodput}(q_{kv},q_{state})
\]

subject to：

\[
\Delta Q_t \ge -\epsilon_t,\quad
\mathrm{P95\ TTFT}\le\tau_f,\quad
\mathrm{P95\ TPOT}\le\tau_p,\quad
\mathrm{Memory}\le M.
\]

### P0-2 修复数学与表述

- [ ] 区分 `N(L)` 并发序列数与 `T(L)` token capacity。
- [ ] 删除所有 lower-bound 说法。
- [ ] 删除“do not share memory”，改为同一 memory pool 的可加 footprint。
- [ ] 重新检查所有 compound ratio 的 baseline 和分母。

### P0-3 完成机制对照

- [ ] fixed-block-count。
- [ ] fixed-bytes。
- [ ] fixed-concurrency。
- [ ] allocator metrics、HBM bytes、kernel latency、queueing breakdown。

### P0-4 重建 serving 主实验

- [ ] 预注册 primary endpoints。
- [ ] full precision、KV-only、state-only、joint 四个配置。
- [ ] offered load→goodput/P95 TTFT/P95 TPOT 完整曲线。
- [ ] 独立 seeds/traces，区分 run stability 与 replication。
- [ ] 对 primary family 做合适的 multiplicity control。

## P1：MLSys 竞争力增强项

- [ ] 至少另一 GPU/显存规格。
- [ ] TP=2/4。
- [ ] 至少另一 hybrid/SSM architecture。
- [ ] no-think RULER 与更高样本量。
- [ ] state fp16/fp8/int8 precision frontier，或解释候选空间为什么受限。
- [ ] 可实现的 KV compression、state compression、offloading/prefix caching baselines。
- [ ] cost/request 或 requests/GPU-hour。

## P2：论文与图形重构

- [ ] 主文压缩为 5–6 张主图。
- [ ] diagnostics 和 null grids 进入 appendix。
- [ ] 全篇统一颜色和配置名称。
- [ ] 每张图只承担一个核心结论，面板之间不重复数据切片。
- [ ] 所有 stochastic aggregate 显示 `n`、CI 定义和统计 family。
- [ ] 以 SVG/PDF/PPTX/draw.io 原生矢量对象交付。

---

## 15. 重新投稿的最低验收标准

在再次按 MLSys 标准审稿前，至少应满足：

1. 核心公式和三处硬性矛盾全部修复。
2. “budgeting”有实际 selector/policy；否则标题与贡献降格。
3. 至少一个真实 baseline suite，而不是只比较内部配置。
4. serving 有完整 load/SLO curves，而不是筛选后的 delta cells。
5. 机制至少通过一组 controlled contrast 得到隔离。
6. 质量图能够定义可接受区域，而不是只说“没有检测到差异”。
7. 至少新增一种 hardware/TP/model-family 泛化证据。
8. 主图架构形成“问题→系统→主结果→机制→边界”的闭环。

在这些条件满足前，单独替换现有图不会改变 Reject/Resubmit 的结论。

