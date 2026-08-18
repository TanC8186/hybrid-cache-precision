# MLSys 2026 最大强度互盲审稿报告

审稿日期 2026-08-14

## Review setup

- **Input scope** `paper/mlsys2026/main.tex`、当前 11 页 `main.pdf`、`main.bib`、八幅论文图、`figures/figure_contract.md`、`figures/verify_figure_data.py`、相关实验与分析脚本、vLLM allocator 实现以及编译日志。
- **Assessment boundary** 这是投稿前模拟审稿，不是编辑决定，也不是作者回复。评估聚焦 originality、scientific importance、interdisciplinary readership、technical soundness 和 readability for nonspecialists，并辅以系统论文的实验设计、统计、复现、claim moderation 和 artifact 检查。
- **Shared manuscript claim summary** 论文主张 hybrid linear-attention serving 的内存应将随长度增长的 attention KV 与每序列固定 recurrent state 联合预算。稿件报告 52 个 fp32/bf16 capacity pairs 全部受益，中位容量增益 15.44%，2B/4K/int4 下从 657 增至 904 个并发序列槽，并提出 constraint-aware selector 将质量、容量和 SLO 条件映射到 full、KV-only、state-only 或 joint 配置。
- **Visible evidence base** 可见主文展示七个容量单元、质量实验、selector 的三个输出、失败的 serving stability gate、机制对照和局限性说明。工作区还提供更广的原始结果与分析脚本，但当前匿名论文包没有提供可独立访问的 artifact。
- **Missing materials affecting confidence** 完整 52-pair/112-cell 容量表、可访问的匿名 artifact、候选级 selector decision trace、实际接近 657/904 序列的 admission/decode 实验、跨模型与跨硬件验证均不在当前投稿材料中。
- **Reviewer isolation** 三位评审使用同一不可变稿件包，在三个隔离上下文中独立完成。预设强调分别是技术可靠性与证据链、原创性与系统意义、跨领域影响与可读性。三份报告在比较前均已冻结，交叉综合未回传给任何评审。
- **Frozen report hashes** Reviewer 1 `F77474E6287E1ACEF32F51CED092F6EABAD6944F29A22888FB0E69320170C232`。Reviewer 2 `BF56053EC23BEA7DE251D4B1B85D7CDABFA6C2F7C4031FF9EE3851F5CB628468`。Reviewer 3 `01492CD037C6CB79582FFDBE1A7AC09D79EEA0B482C1449A9EAA99905A4F636D`。

## Overall posture

三位评审均为 **当前不支持接收**。最可信、也最值得保留的贡献是一个诚实且有工程意义的 Qwen3.5/vLLM allocator characterization。当前没有建立与标题和四项贡献相匹配的联合精度 serving system。若保持现有系统定位，需要补齐 operational capacity、端到端 SLO 收益、selector 正确性与基线、有效质量统计和真实 artifact。另一条可信路线是显著收缩论文，将其重构为范围严格限定的 allocator capacity characterization。

## Reviewer 1

<!-- FROZEN REVIEWER 1 REPORT START -->
# Reviewer 1

## Overall assessment

本文识别了混合线性注意力服务中的一个真实资源维度。注意力 KV 随序列长度增长，GDN recurrent state 则按序列固定分配，因而 KV 位宽与 state dtype 应联合计入内存预算。稿件对负结果和实验边界的披露明显优于通常的系统论文，也正确区分了 run-to-run stability、统计显著性和等价性。

但当前技术证据尚未建立论文的中央系统论证。最可靠的结果仍是 vLLM allocator 报告的 token-block 容量变化，而不是实际可服务的并发容量。完整 52 对 headline 无法由当前投稿包复核，离散 allocator 模型没有给出可重构公式，端到端四配置矩阵未通过作者自己的稳定性门槛，selector 的实验只证明三个预设映射能够重复执行，并未证明可行性判断或最优化正确。质量侧的 GSM8K 统计单位、seed 独立性和区间构造也不足以支撑 selector 所需的 quality lower bound。

主动披露这些局限是优点，但披露不能替代缺失的证据。按当前形态，论文更接近一个有价值的 allocator characterization，而不是已经验证的联合精度服务系统。

## Who would be interested in the results, and why

MLSys、LLM serving、KV cache 管理和混合 recurrent-attention 架构研究者会关心这一结果，因为固定 per-sequence state 会改变短上下文高并发场景中的缓存经济性。部署 Qwen3.5 类模型的系统工程人员也会关心 state dtype 是否会改变可分配 cache blocks。

目前对更广泛系统读者的价值受限于一点。稿件尚未证明 allocator 中释放出的容量能转化为真实并发、稳定 SLO-goodput 或跨工作负载的部署收益。

## Major strengths

- 稿件明确区分 allocator token capacity、质量结果、serving stability 和机制归因，没有把失败的四配置矩阵包装成正面性能结论。
- 容量方向在两个确定性 attempt 中一致，正文也将连续模型正确降格为 predictor，而不是 bound。
- GSM8K 的 2B 回归被保留，RULER 的全零结果被明确描述为 observed equality，而非 equivalence。
- 随稿 `verify_figure_data.py` 可运行并打印 229 条 ledger，目视核对的主要图值与正文未发现明显抄录冲突。该结果仅说明数据重导出链能运行，不等于实验或绘图已被真正验证。
- 图表总体清晰，失败门槛、multiplicity correction 和 run-sensitive cells 均有显式披露。

## Major Concerns

### R1-M1 [experimental-design]

**Severity**  
Major

**Blocking**  
Yes

**Claim pointer**  
论文的 headline 是“verified capacity effect”，并进一步声称 2B/4K/int4 下增加约 247 个 concurrent sequence slots。

**Evidence pointer**  
`main.tex` 第 38 至 41 行，第 81 至 84 行，第 176 至 200 行，第 231 至 239 行，第 334 至 338 行，第 651 至 669 行。

**Concern**  
所谓 capacity 来自 engine maximum-token probe，每个 cell 是一次确定性 allocator run。稿件没有展示在预测的 657 和 904 个 length-4096 请求下实际完成并发 admission、prefill 和 decode，也没有报告 scheduler 的 sequence 上限、请求级 block 需求、运行时保留内存或其他会限制并发的约束。式 (1) 甚至没有整数 floor、page ceil 或 scheduler cap。

因此，数据直接建立的是 allocator-reported token ceiling。它尚未建立“可同时服务 904 个序列”，也未建立“多出 247 个实际 sequence slots”。把总 token ceiling 除以 \(L\) 不能自动得到可执行并发。

**Why it matters**  
这是论文最核心、也是唯一声称 verified 的结果。如果释放出的 allocator blocks 不能被 scheduler 接纳并在目标 SLO 下使用，中央结论只能是静态内存记账，而不是系统容量提升。selector 中的 \(N_q(L)\ge N_{\min}\) 也会继承同一未验证假设。

**Resolution test**  
给出精确的 operational capacity 定义和完整离散公式，包括 page/block ceil、sequence state blocks、`max_num_seqs`、scheduler 限制及运行时内存保留。随后在至少 headline 2B/4K cell 上实际启动接近两个预测上限的固定长度并发请求，证明 admission 无拒绝、无 OOM，并报告完成率、峰值内存、TTFT、TPOT 和 goodput。若不做该验证，则应把所有 concurrent slots 表述改为 allocator-equivalent token slots，并删除实际并发外推。

### R1-M2 [reproducibility]

**Severity**  
Major

**Blocking**  
Yes

**Claim pointer**  
全部 52 对均受益，median gain 为 15.44%，模型 median absolute residual 为 1.81%，范围为 \(-3.66\%\) 到 \(+13.21\%\)，且残差由 block rounding 解释。

**Evidence pointer**  
`main.tex` 第 38 至 41 行，第 96 至 104 行，第 231 至 239 行，第 281 至 303 行，第 312 至 327 行，第 655 至 657 行，第 671 至 694 行。`verify_figure_data.py` 第 1 至 5 行，第 25 至 37 行，第 133 至 135 行。`main.bib` 第 316 至 320 行。

**Concern**  
当前论文只展示 7 个 capacity cells，完整 52 对及 112 cells 没有随稿表格或机器可读数据。随稿 verifier 运行时也只为 Figure 1 打印这 7 个 capacity rows，并不重算 52 对 headline。

该脚本不是实际 cross-checker。它不读取 PDF 或 plotting objects，不保存预期值，没有任何 equality assertion，也不会在绘图值错误时失败。成功退出仅表明外部 JSON 能被读取并打印。其引用的原子 JSON 不在不可变审稿包中。正文声称 artifacts 可审计，但 `mlsys2026ae` 实际只指向 MLSys Artifact Evaluation 的 call page，不是论文 artifact。

附录同样没有兑现“padded block accounting details”。它未给出 \(A_f\)、\(A_q\)、\(M\)、metadata/alignment 开销、离散 page 公式或从 block size/count 到 measured token ceiling 的映射。因此，残差只能被观察为与 rounding 相容，尚不能被唯一归因于 rounding。

**Why it matters**  
52 对方向、15.44% median 和 residual range 是摘要、引言及结论反复使用的 headline。当前审稿材料既无法检查这些统计量，也无法排除 cell selection、遗漏、模型实现或绘图错误。中央证据不可审计时，“verified”与“reproducible”均过强。

**Resolution test**  
提交完整 112-cell 表和原子数据，包含每个 cell 的配置、commit、利用率、resolved dtype、block size、block count、token ceiling、pair ID 和 attempt ID。给出无拟合参数的完整离散 allocator 公式及至少一个逐项 worked example。把 verifier 改为独立重算所有 headline、表格和图值并执行 assertions，任何不一致必须返回非零退出码。提供可访问、版本固定且带 checksum 的 artifact 链接与 preregistration 记录。

### R1-M3 [claim-moderation]

**Severity**  
Major

**Blocking**  
Yes

**Claim pointer**  
本文被定位为联合精度服务系统，贡献包括 executable selector、serving consequences 和 controlled mechanism contrasts。

**Evidence pointer**  
`main.tex` 第 46 至 53 行，第 81 至 94 行，第 273 至 279 行，第 479 至 492 行，第 530 至 572 行，第 606 至 626 行，第 648 至 662 行。

**Concern**  
端到端系统证据没有建立 joint precision 的可用收益。60 个 serving cells 中没有一个通过 BH-FDR。可持续边界在两个 run 间改变方向。更广的四配置矩阵有 183/720 个 continuous-goodput comparisons 超出 10% tolerance，最大差异为 79.61%，因此失败了作者自己的 primary gate。机制实验中的 joint-minus-full throughput 在三个控制条件下均为负，且九个检验均未通过 Holm correction。

这些披露是诚实的，但它们仍然是失败或不确定证据。它们不能因为被放进 Limitations 就转化为系统贡献的支持证据。三个预算映射的重复执行也不能替代 full、KV-only、state-only、joint 四者之间稳定的端到端比较。

**Why it matters**  
一个 MLSys 系统论文需要证明额外 allocator capacity 能转化为用户可观察的部署收益，例如更高 admitted concurrency、稳定 SLO-goodput 或相同质量约束下更好的 Pareto frontier。当前证据只稳定支持静态 allocator 容量方向，尚未支持系统效用。

**Resolution test**  
在预先冻结的多 stratum 设计中，比较全部四个内部基线，至少覆盖两个模型、多个上下文长度和两种真实 workload。使用独立 arrival traces 或 seeds 重复实验，证明 effect magnitude 而不只是原始 cell 值能够复现，并在质量与 SLO 约束下展示可用并发或 goodput 改善。若无法获得稳定端到端收益，应把论文重新定位为 allocator characterization，移除 selector 和 serving system benefit 作为主要贡献的表述。

### R1-M4 [experimental-design]

**Severity**  
Major

**Blocking**  
Yes

**Claim pointer**  
selector 会过滤违反质量、内存、并发和 latency constraints 的候选，最大化 SLO-goodput lower bound，并对未测量或歧义 profile fail closed。

**Evidence pointer**  
`main.tex` 第 202 至 215 行，第 220 至 229 行，第 252 至 262 行，第 476 至 492 行，第 621 至 626 行。

**Concern**  
selector 实验只展示 strict、medium、high 三个预算分别输出 full、state-only 和 joint，然后运行被选中的配置。稿件未展示三个预算的实际数值、四个候选的 profile rows、每项 confidence bound、被拒绝原因、目标值或 argmax margin。KV-only 从未被选中。missing row、duplicate row、ambiguous row、tie 和真正 infeasible request 的行为也没有可见测试结果。

重复得到相同映射只能证明实现是可重复执行的，不能证明 feasibility predicate、lower-bound optimization 或 fail-closed 逻辑正确。被选配置满足 SLO 也不能证明它是可行集中最优的配置。

此外，稿件说明 earlier quality/serving 与后续 capacity/controller 使用不同 vLLM commits 和不同 memory-utilization protocols，却没有给出 selector 每个 profile field 的 provenance。因此无法确认一个候选的质量、容量和 latency 数据是否对应同一个可执行系统版本。

**Why it matters**  
selector 是将简单内存公式提升为系统贡献的核心。如果没有候选级 decision trace 和 oracle comparison，三个预期映射很容易由预算选择方式产生，不能证伪 selector 实现错误、profile 泄漏或跨版本不兼容。

**Resolution test**  
为每个 request 提供完整候选表，列出 dtype 映射、profile provenance、quality lower bound、TTFT/TPOT upper bounds、cache bytes、concurrency、goodput lower bound、可行性和最终排序。使用独立 holdout measurement 验证被选配置。加入可由人工 oracle 验证的 exhaustive unit tests，覆盖四种候选、边界相等、缺失、歧义、tie 和 infeasible cases。所有联合 profile 应来自同一 commit 和兼容 protocol，或给出跨版本校准验证。

### R1-M5 [statistical-rigor]

**Severity**  
Major

**Blocking**  
Yes

**Claim pointer**  
2B GSM8K 上 state-bf16 导致显著的 1.00 percentage-point 回归，\(p=0.025\)，并给出 MDE、observed power 和 paired Cohen's \(d\)。这些质量统计也被 selector 用作 lower bounds。

**Evidence pointer**  
`main.tex` 第 106 至 115 行，第 241 至 250 行，第 379 至 402 行，第 637 至 644 行。`verify_figure_data.py` 第 39 至 54 行。

**Concern**  
推理采用 greedy decoding，九个所谓 dataset seeds 的主要随机性来自如何选择 200 items。稿件没有说明九组 items 是相同、互斥还是重叠，也没有定义统计总体和独立实验单位。若相同 items 被多个 seed 重用，九个 seed means 不是九个独立样本。若为重叠抽样，普通 paired t-test 仍会低估依赖性。若每个 seed 是不同子集，则应以 item-level matched outcomes 或明确的 cluster design 分析。

稿件也没有给出 GSM8K 多个 contrasts 的 primary endpoint 和 multiplicity plan。2B state、9B state、int4、joint 和 stacking 都被检验，但只有 serving 与 per-layer family 明确报告 correction。所谓 observed power 是由已观察效应和方差反推的量，不能提供独立证据，也不能修复抽样单位问题。

**Why it matters**  
联合精度配置是否可部署取决于质量损失的可信上界，而不只是均值方向。若 1.0-point 回归的 CI 和 p-value 使用了错误的独立单位，质量 trade-off、MDE 陈述及 selector feasibility 都可能改变。

**Resolution test**  
披露每个 seed 的 item IDs、重叠结构和配对方式，并提供 item-level correctness pairs。采用适合 matched binary outcomes 和重复 item sampling 的方法，例如按 item cluster bootstrap、hierarchical model 或预先定义的 exact paired test。明确 primary contrast、multiplicity family、alpha、MDE 假设和非劣界。使用修正后的 simultaneous quality bounds 重新运行 selector。

### R1-M6 [mechanism-evidence]

**Severity**  
Major

**Blocking**  
No

**Claim pointer**  
chunk-level PPL 被用于支持 state-bf16 在 int4 KV 下没有额外 perplexity cost，以及没有 layer-wise precision benefit。

**Evidence pointer**  
`main.tex` 第 241 至 244 行，第 366 至 377 行，第 435 至 439 行，第 454 至 472 行，第 588 至 592 行。

**Concern**  
chunk 从 128 改为 1 时，C4 PPL 从约 19.35 变为 36.16，变化约 87%。这说明结果对 state write-back semantics 极度敏感。chunk-128 harness 与真实 per-token kernel path 不等价，因此其很小的 fp32/bf16 delta 不能直接支持真实 serving kernel 上的 perplexity 或 per-layer allocation 结论。

chunk ablation 本身只有 1 seed 和 1 sequence。GSM8K 与 RULER 虽走 kernel path，但任务范围有限，不能校准 perplexity harness 的偏差。

**Why it matters**  
PPL 与 per-layer sensitivity 被作为质量闭环的一部分。一个已知改变绝对 PPL 87% 的近似 harness，除非证明 treatment effect 对该近似不敏感，否则不能承担这种作用。

**Resolution test**  
在真实 kernel path 上计算 token-level log-likelihood，或使用与 per-token state write-back 等价的 harness，在多文档和多 seeds 上重做 PPL、stacking 和 per-layer sensitivity。至少应直接比较 chunk-1 与 chunk-128 的 paired treatment effects 及其不确定性。若做不到，应删除 PPL 和 per-layer 结果对 kernel-level quality 的支持性表述。

### R1-M7 [causal-vs-correlative]

**Severity**  
Major

**Blocking**  
No

**Claim pointer**  
controlled mechanism contrasts 被描述为 reproducing throughput，并据此“promote only the tested throughput direction”。

**Evidence pointer**  
`main.tex` 第 273 至 275 行，第 554 至 562 行，第 614 至 619 行。

**Concern**  
18/18 throughput cells 在两个 attempt 中相差不超过 10%，只是原始观测值的稳定性，不是 joint-minus-full treatment effect 的复现。正文只给出 formal attempt 的三个 effect means，没有给出第二个 attempt 的 effect、差值区间或 sign agreement。与此同时，九个 tests 均未通过 Holm correction。

fixed-concurrency contrast 还改变 queueing constraint，作者也承认它不是纯 dtype intervention。因此现有结果不能支持 throughput direction 的机制解释。

**Why it matters**  
原始 metric 稳定而 treatment contrast 不稳定是完全可能的。将前者解释为后者会夸大机制证据，并使读者误以为 causal channel 已经缩小到 throughput。

**Resolution test**  
分别报告两个 attempts 中三个 contrast 的 paired effect、置信区间、方向和 difference-of-differences，并对预先定义的 effect family 做 multiplicity correction。机制实验应一次只改变一个资源约束，同时保持 offered load 和 queueing policy 可比。若 effect 本身不能复现，应删除“promote throughput direction”并把结果限定为 raw metric repeatability。

## Minor Comments

### R1-m1 [writing-clarity]

**Severity**  
Minor

**Blocking**  
No

**Claim pointer**  
full、KV-only、state-only 和 joint 被反复作为候选与基线使用。

**Evidence pointer**  
`main.tex` 第 81 至 86 行，第 273 至 277 行，第 479 至 481 行。

**Issue**  
正文没有给出四个标签的正式配置映射。读者只能推测 full 是否表示 fp16 KV 加 fp32 state，以及 weight、activation 和 compute precision 是否保持一致。

**Why it matters**  
配置身份不明确会妨碍基线公平性判断和复现。

**Resolution test**  
加入一张配置表，明确每个标签的 KV dtype、state dtype、weight/activation dtype、quantizer、scale/metadata layout、engine arguments 和 commit。

### R1-m2 [figures-and-tables]

**Severity**  
Minor

**Blocking**  
No

**Claim pointer**  
Table 1 被描述为 fixed-utilization 4K/16K slice，fp16-KV gains 被写为 “7.0--15.4% at 4K”。

**Evidence pointer**  
`main.tex` 第 238 至 239 行，第 281 至 315 行。

**Issue**  
表中缺失 9B/fp16/16K cell，却没有说明原因。7.0% 对应 2B/fp16/16K，不是 4K，因此 “at 4K” 与表中数值不一致。

**Why it matters**  
该表是中央容量结果的唯一可见数值切片。无解释缺失和错误范围会降低对选择性报告的信心。

**Resolution test**  
补入 9B/fp16/16K，或以明确的 NA 和原因标注。把范围改为准确的 cell 集合描述，并自动从表数据生成正文范围。

### R1-m3 [claim-moderation]

**Severity**  
Minor

**Blocking**  
No

**Claim pointer**  
`figure_contract.md` 定义图的 core conclusion 和 evidence hierarchy。

**Evidence pointer**  
`figure_contract.md` 第 3 至 6 行，第 25 至 32 行。`main.tex` 第 40 至 45 行，第 269 至 271 行，第 320 至 322 行，第 641 至 644 行。

**Issue**  
Figure contract 仍声称质量代价不可检测、serving 为独立复现、int4 列构成保守下界。正文则报告显著 GSM8K 回归，说明 serving 两次使用相同 seeds/contracts，并明确 capacity model 不是 bound。

**Why it matters**  
同一提交包对 central claim 的定义不一致，容易造成图形、artifact 和 rebuttal 后续继续使用过期结论。

**Resolution test**  
将 contract 与冻结正文逐项同步，删除“独立复现”“保守下界”和广义“质量代价不可检测”等不再成立的表述。

### R1-m4 [statistical-rigor]

**Severity**  
Minor

**Blocking**  
No

**Claim pointer**  
selector 最大化 Goodput lower bound，serving 图报告 paired goodput change。

**Evidence pointer**  
`main.tex` 第 202 至 215 行，第 264 至 279 行，第 530 至 549 行。Paper Figure 8。

**Issue**  
Goodput 的精确定义、单位、失败请求处理、窗口边界和 confidence-bound estimator 未完整给出。Figure 8 的 y 轴也未标明单位。

**Why it matters**  
读者无法判断 0.334 等数值是 req/s、比例还是其他归一化量，也无法复算 selector objective。

**Resolution test**  
给出 goodput 的公式、单位、请求纳入规则、失败和超时处理，以及 lower CI 的计算方法。所有图表轴和表格应使用一致单位。

### R1-m5 [figures-and-tables]

**Severity**  
Minor

**Blocking**  
No

**Claim pointer**  
Paper Figure 2b 展示 measured 和 capacity-model gap，Paper Figure 4b 展示 RULER exact rerun。

**Evidence pointer**  
`main.pdf` 第 6 页和第 7 页，对应 `main.tex` 第 351 至 360 行、第 420 至 429 行。

**Issue**  
Figure 2b 的 2B/16K fp16 gap label 与 baseline/legend 区域拥挤。RULER 的 exact-rerun 注记也挤在最后一个 cell 附近。最终双栏尺寸下这些信息不易辨认。

**Why it matters**  
这些注记承载的是残差方向与“非等价性”边界，不是装饰性文字。

**Resolution test**  
移动 annotations，增加局部留白，并在最终嵌入尺寸下确保标签不与 baseline、marker 或 legend 重叠。

### R1-m6 [reproducibility]

**Severity**  
Minor

**Blocking**  
No

**Claim pointer**  
当前 PDF 被视为可提交的 MLSys 2026 编译稿。

**Evidence pointer**  
`main.tex` 第 14 行。`main.log` 第 613 至 707 行。`main.pdf` metadata 的 Subject 为 `mlsys 2025`。

**Issue**  
编译日志对 Table 1、Table 2 及 Figure 1 至 Figure 8 均报告 duplicate destination，超链接目标可能错误。源文件仍加载 `mlsys2025` style，PDF metadata 也保留 2025 标识。

**Why it matters**  
这不改变科学结果，但会损害审稿导航，也可能与最终投稿格式不一致。

**Resolution test**  
消除全部 duplicate-anchor warnings，检查每个 `\ref` 和 PDF bookmark 跳转，并确认 MLSys 2026 官方模板及 metadata 要求。

## Technical failings that need to be addressed before the case is established

阻断性技术问题为 R1-M1、R1-M2、R1-M3、R1-M4 和 R1-M5。

当前必须建立以下证据链。allocator token ceiling 需要转化为实际可服务并发。完整 52 对 headline 和离散模型需要可独立复算。四配置端到端收益需要通过预设稳定性标准。selector 需要候选级正确性验证。质量 lower bounds 需要基于明确且有效的统计单位。

## Assessment against Nature-style criteria

**Originality**  
把 KV 与 recurrent state 作为两个联合预算维度是合理且可能有用的视角。在随稿引用范围内，这一表述具有一定新意。但 state dtype 开关和 state compression 均为已有技术，核心连续公式较直接。原创性最终取决于 selector 与端到端系统收益，而这两部分当前尚未建立。

**Scientific importance**  
混合 recurrent-attention 模型正在进入实际 serving，固定 state 对短上下文并发的影响值得研究。现有证据只覆盖两个 Qwen3.5 sizes、单张 RTX 5090 和有限 workload，且真实并发未验证，因此目前证明的是较窄的 allocator effect，而不是广泛系统影响。

**Interdisciplinary readership**  
LLM serving、memory management 和混合架构研究者会直接关心。对更广读者的吸引力取决于能否证明这种预算方式改变实际 SLO、资源成本或部署 Pareto frontier。

**Technical soundness**  
容量方向本身可信且与简单内存逻辑一致，但 operational capacity、完整 headline audit、selector correctness、质量统计和端到端系统效用存在阻断缺口。当前证据不足以支持完整技术主张。

**Readability for nonspecialists**  
论文主线和限制写得较清楚，负结果边界也容易找到。但 full/KV-only/state-only/joint 的定义、离散 allocator 公式、profile provenance 和统计单位缺失，使非直接参与该实现的读者无法重构证据链。

## Recommendation posture

当前不支持接收。论文包含一个值得保留的 allocator observation，也展现了较好的负结果披露，但中央 MLSys 系统论证尚未由端到端证据和可复核 artifact 建立。需要实质性重做后再评估。

<!-- FROZEN REVIEWER 1 REPORT END -->

## Reviewer 2

<!-- FROZEN REVIEWER 2 REPORT START -->
## Reviewer 2

### Overall assessment

本报告仅依据指定的匿名主稿、当前 11 页 PDF、参考文献、八幅图、figure contract、图数据核验脚本和编译告警。论文最可信的结果是一个范围明确的 allocator 观察。将 Qwen3.5 的 GDN recurrent state 从 fp32 改为 bf16，会减少每序列状态字节数，并在若干固定配置下增加 vLLM 可分配的 token/block 数。作者对质量回归、统计分辨率不足和 serving 复现失败的披露较为透明。

但以 MLSys 系统论文标准衡量，当前工作更接近对已有 vLLM dtype 开关的系统化测量，而不是已建立的新系统技术。所谓联合预算模型主要是两个已知内存项的加法，selector 是对完整测量表执行约束过滤和 argmax，真正的端到端 serving 主门则失败。论文尚未证明其贡献在原创性、系统价值或可部署收益上超过“现有开关加简单内存算术加离线查表”的组合。当前证据不足以支撑论文所宣称的 MLSys 级系统贡献。

### Who would be interested in the results, and why

直接受众是部署 hybrid recurrent-attention 模型的推理系统工程师、vLLM 内存管理开发者，以及研究 KV/state 低精度缓存的研究者。固定 recurrent state 在短上下文、高并发场景中可能成为显著内存项，这个观察具有实际诊断价值。更广泛的系统读者是否会受益，则取决于论文能否证明该观察会转化为稳定的 SLO goodput、硬件节省或跨系统决策优势，目前尚未证明。

### Major strengths

- 论文没有把失败的四配置 serving 复现实验包装成正结果，明确保留了 183/720 个连续 goodput 比较超出 10% 容差这一事实。
- 容量公式、block 粒度测量和 allocator token ceiling 之间形成了较清楚的局部证据链。
- GSM8K 的 2B 质量回归被如实报告，RULER 的精确相等也被正确限定为 observed agreement，而不是 equivalence。
- 图表总体清晰，正结果、负结果和方法边界均有可见呈现。

### Major Concerns

#### R2-M1 [novelty-significance]

**Severity**

Major

**Blocking**

Yes

**Claim pointer**

论文将“联合分配 attention KV bit-width 和 recurrent-state dtype”定位为主要原创贡献，并声称现有系统没有把 state precision 作为与 KV bit-width 联合优化的预算维度。

**Evidence pointer**

`main.tex` 第 76 至 92 行，第 124 至 171 行，第 176 至 215 行，以及第 476 至 492 行。`main.bib` 第 213 至 249 行列出了 ReplaySSM、vLLM state-dtype PR 和 FP8 state checkpointing。

**Concern**

论文已经承认 state compression 是 prior art，vLLM 也已经公开 `mamba_ssm_cache_dtype` 开关。新增模型是 \(M/(AL+G)\)，selector 则枚举 full、KV-only、state-only 和 joint 四个候选，对已测量 profile 做约束过滤后取最大 lower-bound goodput。稿件没有说明这里新增了什么不可由现有 vLLM 配置、一个 spreadsheet 或简单部署脚本直接实现的系统机制，也没有展示核心代码路径、allocator 改动、在线策略或新算法。

相关工作只通过文字断言“尚无人联合分析”，没有给出与 ReplaySSM、现有 state-dtype 路径、FP8 state 工作及强 KV quantization 方法的逐项能力比较。论文也没有说明当前 vLLM/Qwen3.5 的默认 state dtype 或实际部署常用配置。若 bf16 已经是常见或默认状态，fp32 到 bf16 的容量增益将更接近对一个已有配置差异的测量，而不是新的部署能力。52/52 的方向性也近乎由“减小每序列字节数不会减少可分配容量”这一单调性预定，不能单独证明原创性。

**Why it matters**

原创性是本稿当前最薄弱的环节。若去掉 selector 包装，核心结果仍然是已有 dtype 开关带来的内存算术后果。论文必须证明新贡献不只是把两个已有旋钮放进同一个二维表中。

**Resolution test**

给出明确的 novelty matrix，逐项比较现有 vLLM state 开关、ReplaySSM、可执行的 state compression 路径、KV-only 策略和本文系统。说明默认配置、实现改动、决策机制及现有系统无法完成的行为。用实际基线证明本文方法在质量约束下产生更优或不同的部署决策。若不存在这样的差异，应将贡献降格为范围明确的 empirical characterization，并相应收缩标题和系统创新主张。

#### R2-M2 [technical-soundness]

**Severity**

Major

**Blocking**

Yes

**Claim pointer**

论文将 allocator capacity 解释为可部署并发能力，尤其声称 2B/4K/int4 场景增加约 247 个 concurrent sequence slots，并据此提出短上下文、高并发、内存受限场景的 operational rule。

**Evidence pointer**

`main.tex` 第 93 至 104 行，第 231 至 239 行，第 312 至 338 行，第 494 至 572 行，第 606 至 619 行，以及第 648 至 669 行。相关图为源码标签 `fig:capacity`、`fig:blocks` 和 `fig:serving`。

**Concern**

容量 probe 是单次 deterministic maximum-token allocator run。657 和 904 个并发序列是用 token ceiling 除以 4096 得出的，并未实际启动、维持或完成这些并发请求。真实 serving 还受到排队、prefill/decode 交叠、输出增长、碎片、工作区、调度和 latency SLO 的约束。

最关键的是，端到端证据没有闭合这一推断。四配置 serving 主门失败，183/720 个连续 goodput 值超出 10% 容差。60 个 serving cells 中没有一个通过 BH-FDR。可持续边界在两个 workload 中出现不一致方向。机制矩阵中的 joint-minus-full throughput 在三个控制条件下分别为负值，且九个检验均未通过 Holm 校正。稿件也没有展示 serving 实验实际达到 657 或 904 个 active sequences，或证明实验处于由 recurrent-state memory 决定的瓶颈区。

**Why it matters**

MLSys 价值不能仅由“allocator 能预留更多块”替代。若新增块不能稳定转化为 SLO 下的吞吐、goodput、可服务并发或更少硬件，15.44% 的 median capacity gain 仍主要是内存会计结果，而不是已验证的系统收益。

**Resolution test**

在明确的 memory-saturated 工作负载中，实际运行接近预测 \(N(L)\) 的并发序列，并报告成功请求数、OOM 边界、TTFT、TPOT、throughput、goodput、GPU memory occupancy 和成本。至少覆盖 full、KV-only、state-only 和 joint，使用独立运行或环境重复，并在预先定义的主指标上通过稳定性门。若端到端收益仍不稳定，应删除“247 个可用并发槽位”和 operational benefit 的暗示，只保留 allocator capacity characterization。

#### R2-M3 [experimental-design]

**Severity**

Major

**Blocking**

Yes

**Claim pointer**

论文声称实现了 constraint-aware executable selector，能够根据质量、内存、并发和 latency 约束选择最优 precision configuration，并在缺失 profile 时 fail closed。

**Evidence pointer**

`main.tex` 第 202 至 215 行，第 252 至 262 行，第 476 至 492 行，以及第 621 至 626 行。

**Concern**

selector 只在 exact measured profile rows 上运行，不插值、不外推，也不在线适配。当前验证仅覆盖 Qwen3.5-2B、4K、Random、TP=1 的一个切片和三个预算。稿件没有给出三个预算的具体数值、四个候选的完整 profile、置信界构造、被拒候选及理由、最终 objective 值，因而无法检查 argmax 是否正确。确认所选配置能够启动并满足 SLO，只证明命令执行成功，不证明 selector 优于手工设置、always-bf16、KV-only、state-only、现有 vLLM 默认策略或 exhaustive oracle。

此外，四配置 serving matrix 正是 selector 需要依赖的比较基础，但其 primary continuous-goodput gate 已失败。稿件也没有说明 profiling 数据与 confirmatory evaluation 是否严格隔离，因此无法排除选择和评估使用同一 profile stratum 所造成的乐观性。

**Why it matters**

selector 是论文列出的四项主要贡献之一，也是把简单容量观察提升为系统论文的关键部分。当前证据只支持离线查表和冷启动命令生成，不支持正确性、最优性、泛化性或实际决策价值。

**Resolution test**

公开 selector 算法、所有预算、候选 profile、置信界方法和逐候选决策轨迹。使用 held-out strata 评估，并与手工 vLLM 配置、always-bf16、KV-only、state-only、简单 greedy 和 exhaustive oracle 比较。报告 regret、SLO violation、fail-closed 频率、profiling cost 和 cold-restart overhead。至少在多个 model、context 和 workload 上证明联合 selector 带来稳定优势。

#### R2-M4 [reproducibility]

**Severity**

Major

**Blocking**

Yes

**Claim pointer**

论文将 52 对容量结果、完整 serving contracts、per-seed 数据和 reproduction artifacts 描述为已释放且可独立审计。

**Evidence pointer**

`main.tex` 第 231 至 239 行，第 269 至 279 行，第 318 至 324 行，第 564 至 572 行，以及第 671 至 675 行。`main.bib` 第 316 至 320 行。`verify_figure_data.py` 第 1 至 18 行和第 25 至 134 行。

**Concern**

当前提交包没有呈现完整 52 对容量矩阵，正文表格和 capacity figure 只显示 7 个切片。用于所谓 atomic artifacts 的引用 `mlsys2026ae` 实际指向 MLSys 2026 Call for Artifact Evaluations，而不是匿名 artifact、代码仓库或数据归档。

图数据脚本也不构成有效的独立 verifier。它将 `ROOT` 指向 `paper/mlsys2026` 之外的仓库根目录，并从外部 `results/...` 路径加载数据。脚本只重新读取并打印 229 个 ledger 条目，不读取 PDF、绘图输出或预期值，也没有 assertion 或 mismatch comparison。因此，即使脚本在当前工作区运行成功，也不能检测绘图错误，不能让只获得提交包的审稿人复现结果，更没有验证完整 52 对 headline matrix。selector 的代码、三组预算和逐候选输入同样不可见。

**Why it matters**

论文反复使用 verified、reproduced、preregistered 和 independently auditable 作为可信度支柱。若核心矩阵和实际 artifact 不在匿名提交中，这些声明无法由审稿包支持，headline median、范围和 selector 行为也无法独立核验。

**Resolution test**

提供可访问的匿名 artifact，包含完整 112-cell 和 144-cell 原始记录、全部 52 个 paired rows、所有 profile、每个 seed、环境锁文件、模型 revision、driver/CUDA/PyTorch/vLLM 信息、绘图代码和 selector 代码。将 `mlsys2026ae` 替换为真实 artifact 引用。verifier 应在隔离目录中从原始日志重算统计量，并通过明确 assertions 与论文表图中的冻结期望值比较。

#### R2-M5 [statistical-rigor]

**Severity**

Major

**Blocking**

No

**Claim pointer**

论文将 perplexity、GSM8K 和 RULER 组成 quality map，并让 selector 使用 task-level quality lower bounds 判断候选是否可行。

**Evidence pointer**

`main.tex` 第 81 至 90 行，第 106 至 115 行，第 241 至 250 行，第 363 至 472 行，第 588 至 604 行，以及第 637 至 646 行。相关图为源码标签 `fig:gsm8k`、`fig:ruler`、`fig:harness` 和 `fig:gsm8k-seeds`。

**Concern**

现有质量证据不足以支撑一个可推广的 quality-aware policy。2B state-bf16 在 GSM8K 上有显著 1.0 percentage point 回归。RULER 只有五个经先前 screen 选择的 cells，每 cell 20 samples、三个 dataset seeds，作者也承认没有 equivalence margin。PPL harness 在 chunk=1 时改变绝对 PPL 约 87%，因此不能验证真实 token-level kernel semantics。

GSM8K 的九个 dataset seeds 如何生成、200 个 item 子集是否重叠、统计独立单位是什么，稿件没有说明。多个 GSM8K 配置比较的 multiplicity plan 也不可见。对一个要求精确 quality lower bound 的 selector 而言，这些证据只能支持少数已测 task/configuration 的局部描述。

**Why it matters**

state-bf16 已经显示 task-dependent harm。若每个新任务、模型和 workload 都需要重新建立低分辨率 profile，selector 的实际适用范围和科学意义会显著收缩。

**Resolution test**

为每项任务预先定义有操作意义的 non-inferiority margin，使用足够 sample-level 数据和明确的统计单位，说明 seed 子集重叠及 multiplicity 处理。覆盖四种 precision configurations、更多推理和长上下文任务，以及至少另一个模型。将这些置信界直接对应到 selector 的具体 quality floors，并展示决策对统计不确定性的敏感性。

#### R2-M6 [claim-moderation]

**Severity**

Major

**Blocking**

No

**Claim pointer**

标题和结论将结果概括为 hybrid linear-attention serving 的联合精度预算规律和部署规则。

**Evidence pointer**

`main.tex` 第 20 行，第 68 至 94 行，第 220 至 229 行，第 600 至 635 行，以及第 648 至 669 行。

**Concern**

所有结果来自一个 Qwen3.5 GDN family 的两个尺寸、一张 RTX 5090 和单 GPU。Mamba2-style architecture、其他 hybrid family、其他 hardware、TP=2/4、prefix caching、offloading 均未测试。TP 扩展只是“一阶期望”，而离散 rounding 和通信正是可能改变结果的因素。论文对限制写得诚实，但标题和前部 framing 仍比证据范围更宽。对外部 memory-management baselines 的缺失也使目前无法判断 state switch 是否是目标部署中的最重要旋钮。

**Why it matters**

局部效果可以有工程价值，但一篇以 hybrid linear-attention serving 为对象的 MLSys 论文需要证明结论跨越单个模型实现和单个消费级 GPU，或者明确将主张限制为 Qwen3.5/vLLM/RTX 5090 characterization。

**Resolution test**

至少增加另一个 hybrid architecture、另一类 GPU 或 tensor-parallel 配置，并在同一 SLO protocol 下比较。若无法扩展实验，应在标题、摘要、贡献列表和结论中一致地限定 Qwen3.5、单 RTX 5090、offline profiling 和 allocator-capacity 范围。

### Minor Comments

#### R2-m1 [writing-clarity]

**Severity**

Minor

**Blocking**

No

**Claim pointer**

论文使用 R2-to-R3、Gate 4、M3 和 M4 等标签描述复现和验收状态。

**Evidence pointer**

`main.tex` 第 96 至 103 行，第 231 至 237 行，第 273 至 279 行，以及第 637 至 646 行。

**Issue**

这些内部 protocol 标签没有完整定义，读者看不到 Gate 1 至 Gate 3 或 M1 至 M2，也不清楚 R2、R3 对应什么。它们增加了“内部实验日志”感，并妨碍独立理解。

**Why it matters**

核心结果大量依赖这些门限，标签含义不透明会削弱可读性和可审查性。

**Resolution test**

用自包含名称替换内部标签，或给出一张表定义每个 gate、metric family、threshold、样本和通过条件。

#### R2-m2 [reproducibility]

**Severity**

Minor

**Blocking**

No

**Claim pointer**

论文比较 full、KV-only、state-only、joint，并提到额外 fp16-state controls。

**Evidence pointer**

`main.tex` 第 81 至 87 行，第 231 至 237 行，第 476 至 482 行，第 497 至 519 行，以及第 600 至 604 行。

**Issue**

四个配置没有在一个位置被精确定义为具体 KV dtype、state dtype 和其他固定参数。与此同时，setup 又出现 8 个 fp16-state controls，而主要 state 维度被描述为 fp32/bf16，容易混淆 `full` 和 `fp16 state` 的含义。

**Why it matters**

配置命名不清会影响基线公平性判断和命令级复现。

**Resolution test**

增加 configuration table，列出每个名称对应的 KV dtype、state dtype、weight/activation dtype、quantizer、默认值及完整 vLLM 参数。

#### R2-m3 [statistical-rigor]

**Severity**

Minor

**Blocking**

No

**Claim pointer**

论文同时报告 preregistered MDE 和 67.5% observed power，用于解释 GSM8K 的 1.0-point 回归。

**Evidence pointer**

`main.tex` 第 379 至 385 行和第 637 至 644 行，源码标签 `fig:gsm8k` 的图注。

**Issue**

基于观测效应计算的 post hoc observed power 通常不增加超过 p value 和 confidence interval 的信息，并可能使读者误以为“低于 MDE 但显著”存在矛盾。稿件也没有给出 power calculation 的方差假设和预注册记录。

**Why it matters**

这是核心质量风险的统计解释，应避免使用容易误导的诊断量。

**Resolution test**

保留预先计算的 MDE、效应值和 confidence interval，删除或明确区分 post hoc power。补充 prospective power 的计算假设和可核验的预注册位置。

#### R2-m4 [claim-moderation]

**Severity**

Minor

**Blocking**

No

**Claim pointer**

论文由校正后无显著 layer effect 推出 whole-state switch 是应采用的 allocation dimension。

**Evidence pointer**

`main.tex` 第 454 至 464 行，源码标签 `fig:sensitivity`。

**Issue**

36 个检验在三个 seeds 和近似 PPL harness 下未通过 Bonferroni/BH-FDR，只能说明当前实验没有检测到 layer-wise effect，不能排除有意义但低于检测能力的 heterogeneous sensitivity，也不能据此证明 whole-state granularity 最优。

**Why it matters**

该推断超出了当前 null result 的分辨率。

**Resolution test**

改写为“未检测到逐层差异，因此本文仅评估 whole-state switch”，并报告逐层分析的检测能力或可识别效应范围。

#### R2-m5 [figures-and-tables]

**Severity**

Minor

**Blocking**

No

**Claim pointer**

核心图旨在支撑 capacity、quality、sensitivity 和 serving 各节。

**Evidence pointer**

当前 `main.pdf` 第 5 至 10 页。源码中图块位于 `main.tex` 第 340 至 473 行和第 574 至 583 行。

**Issue**

多幅图被推迟到远离首次引用的位置。capacity schematic 出现在 capacity 讨论之后，GSM8K、harness、per-layer 和 serving 图散落到参考文献区域，其中 serving 图直到 PDF 第 10 页才出现。Figure 1 和 Figure 2 的源码文件名顺序也与最终编号相反。

**Why it matters**

读者必须跨越数页和参考文献寻找证据，显著增加核心论证的阅读成本。

**Resolution test**

重新安排 float 顺序和尺寸，使每幅核心图靠近首次引用。统一文件名、源码顺序和最终 Figure 编号。

#### R2-m6 [writing-clarity]

**Severity**

Minor

**Blocking**

No

**Claim pointer**

当前 PDF 被作为 MLSys 2026 匿名稿提交。

**Evidence pointer**

`main.tex` 第 14 行使用 `mlsys2025` style。`main.log` 第 504 行和第 613 至 703 行报告 empty anchor、重复 table/figure destination 及多处 underfull box。编译结果为 11 页。

**Issue**

重复 PDF destinations 会使部分 figure/table hyperlink 指向错误或被忽略。稿件还使用 2025 style 文件，且第 11 页仅有较短附录，版面利用率较低。无法仅凭审稿包判断 2026 模板和页数规则是否满足。

**Why it matters**

这不改变科学结论，但会影响提交合规性、导航和最终可读性。

**Resolution test**

使用官方 2026 模板重新编译，消除重复 destinations 和 empty anchor，检查所有链接，并确认正文、参考文献和附录的页数合规。

### Technical failings that need to be addressed before the case is established

阻断当前论文核心论证的问题为 `R2-M1`、`R2-M2`、`R2-M3` 和 `R2-M4`。作者需要证明相对已有 dtype 开关和 prior work 的真实原创性，建立 allocator capacity 到端到端 SLO 收益的证据链，验证 selector 相对简单策略的实际价值，并提供可独立审计的完整核心数据与 artifact。

### Assessment against Nature-style criteria

**Originality**

低。稿件发现了一个值得注意的预算维度，但当前方法由已有 state dtype 开关、已有 KV quantization 和直接加法内存模型组成。selector 的算法和系统新颖性尚未建立。

**Scientific importance / significance**

潜在价值集中在短上下文、高并发、state-heavy 的特定部署区间。由于端到端 serving 主门失败、allocator slots 未转化为可持续并发，当前证明的是容量会计效应，而不是广泛系统收益。

**Interdisciplinary readership**

主要吸引 LLM serving、低精度推理和 hybrid architecture 的专业读者。对更广泛系统群体的意义仍依赖跨架构、跨硬件或成本级收益，目前证据不足。

**Technical soundness**

局部 allocator 结果在稿件范围内具有可信度，负结果披露也较规范。但 selector 验证、quality profile 的统计基础、核心矩阵可审查性和端到端系统证据均不足，尚不能支撑完整系统主张。

**Readability for nonspecialists**

背景和主要公式较容易理解，图表视觉质量较好。内部 gate 标签、配置命名、密集的限定语以及核心图远离正文降低了可读性。非专业读者也很难从当前稿件区分“更多 allocator blocks”和“更高可用 serving capacity”。

### Recommendation posture

当前不支持接收，倾向拒稿。最有价值的部分是一项诚实、范围明确的 allocator characterization，但原创性和系统重要性尚未达到 MLSys 论文所需强度。要改变这一判断，论文需要展示现有 vLLM 开关组合无法直接得到的新系统能力，并给出稳定的端到端 SLO 或成本收益。否则，更可信的方向是将全文重构为严格限定于 Qwen3.5/vLLM/RTX 5090 的经验测量研究，并显著收缩 selector 和广义 serving 贡献。

<!-- FROZEN REVIEWER 2 REPORT END -->

## Reviewer 3

<!-- FROZEN REVIEWER 3 REPORT START -->
# Reviewer 3

## Overall assessment

本报告仅依据指定的匿名主稿、当前 11 页 PDF、参考文献、八张图、图契约、图数据核验脚本和编译告警形成，未读取其他审稿意见或历史分析。

稿件抓住了一个真实且容易解释的系统现象。混合线性注意力模型的服务内存不是单一 KV 项，而是随上下文增长的注意力 KV 加上每序列固定的循环状态。将状态精度纳入预算，在短上下文、高并发、内存受限场景中可能带来显著容量收益。可见的七个容量单元、简单的 \(AL+G\) 模型以及 2B/4K 下约 247 个额外序列槽，共同构成了有吸引力的核心故事。

然而，当前稿件将容量模型、联合 selector、质量地图和 serving 机制研究并列为四项贡献，真正由稿内证据充分支撑的只有容量现象的一个切片。52 个核心配对单元和完整 112-cell 矩阵不可检查，selector 的三次决策无法从论文重建，端到端 serving 证据明确未通过稳定性和多重比较门槛，原创性边界主要依靠断言。与此同时，摘要、引言、Serving 和 Limitations 被 Gate 编号、运行包术语、失败计数和统计免责声明占据，系统设计与实际 takeaway 反而退居次要位置。

我的判断是，中心想法有价值，但当前提交尚未建立与其标题和四项贡献相匹配的 MLSys 系统论文论证。

## Who would be interested in the results, and why

LLM serving、KV cache quantization、vLLM、混合线性注意力和内存分配系统的研究者与工程团队会直接关心这一结果，因为固定的每序列状态会改变 KV 量化的边际收益，现有 attention-only 预算直觉不能直接沿用。

更广泛的 MLSys 读者也可能关心这一资源模型，因为它揭示了新模型架构如何改变 serving resource economics。不过，当前证据只覆盖 Qwen3.5-2B/9B、单张 RTX 5090 和一个主要 selector 切片，且端到端性能结论未验证，因此广泛影响目前仍是潜力，而不是已建立的结果。

## Major strengths

- \(N(L)=M/(AL+G)\) 将问题压缩为一个清楚的系统直觉，并正确区分并发序列数和总 token capacity。
- 稿件没有隐藏不利结果。2B GSM8K 的 1.0 percentage-point 回退、RULER 的低分辨率、PPL harness 的近似性和 serving 稳定性失败均被明确保留。
- 容量结果给出了可操作的部署解释。2B/4K/int4 下由 657 增至 904 个并发序列槽，比仅报告倍率更容易理解。
- 图表总体使用成对差异和置信区间，RULER 图也明确写出零差异不是 equivalence test。
- `verify_figure_data.py` 的设计是从源数据重新推导图中数值，而不是复用绘图常量。当前工作树中的只读运行输出完整 ledger，未显示图值不一致。此核验不能替代可提交 artifact，但说明作者重视图数据一致性。

## Major Concerns

### R3-M1 [reproducibility, data-resource-quality]

**Severity** Major  
**Blocking** Yes  
**Claim pointer** 完整 112-cell allocator matrix 中全部 52 个 fp32/bf16 核心配对均有正向容量收益，中位增益为 15.44%，模型中位绝对残差为 1.81%。  
**Evidence pointer** `main.tex` lines 38-41, 96-104, 231-239, 281-304, 312-338, 671-675 and 684-694; `main.bib` lines 316-320; `verify_figure_data.py` lines 13-18 and 25-131.  
**Concern** 中心 headline 依赖 52 个配对单元，但论文只展示七个容量单元。完整矩阵既不在正文，也不在当前附录。表 1 甚至没有解释为何固定 4K/16K 切片缺少 9B/fp16/16K 单元。论文声称原子 artifacts 已提交并引用 `mlsys2026ae`，但该条目实际指向 MLSys 2026 Artifact Evaluation 征稿页，不是匿名 artifact。核验脚本又依赖提交包外的 `results/...` 文件，单独拿到可见提交材料时无法运行。  
**Why it matters** 15.44% 和 1.81% 是摘要、引言和结论反复使用的中心数字。读者只能看到作者选择的七个单元，无法判断完整矩阵的覆盖、分层分布、异常值或聚合方式。当前稿件因此不能在自身或可访问 artifact 中建立其最重要的实证结论。  
**Resolution test** 在论文附录或真实可访问的匿名 artifact 中提供完整 52-pair/112-cell 表、每个单元的条件和两次运行值，并给出可重算 headline 聚合量的脚本与原始输入。审稿人应能仅从提交材料重算 15.44%、1.81% 和残差范围。若不能提供，则将结论严格缩小到正文实际展示的七个单元。

### R3-M2 [experimental-design, reproducibility]

**Severity** Major  
**Blocking** Yes  
**Claim pointer** selector 将质量、内存、并发和尾延迟约束映射为可执行配置，并对 strict、medium 和 high 三个预算分别选择 full、state-only 和 joint。  
**Evidence pointer** `main.tex` lines 84-87, 202-215, 252-262 and 476-492.  
**Concern** selector 是四项主贡献之一，也是标题中 “Joint Precision Budgeting” 的系统实现，但论文没有展示三个 budget 的具体输入，没有明确定义 full、KV-only、state-only 和 joint 的 dtype 组合，没有给出各候选的质量下界、内存、并发、TTFT/TPOT 上界和 goodput 下界，也没有展示候选被拒绝或胜出的逐项原因。Goodput 及其置信下界的计算方法同样未定义。现有结果只告诉读者 selector 输出了什么，不能判断它为何输出该配置。  
**Why it matters** 一个约束过滤加 argmax 的公式本身不足以构成可评估的系统贡献。缺少决策轨迹后，读者无法检查约束方向、候选公平性、置信界构造、fail-closed 行为或选择是否由某个任意阈值决定。  
**Resolution test** 增加一个完整决策表。每个 budget 应列出全部输入约束、四个候选的 dtype、所有被比较的估计量和界、可行性判定、拒绝原因、最终目标值与选择结果。给出 goodput 和置信界的精确定义。读者应能从表中独立复现三次映射。

### R3-M3 [scientific-importance, claim-moderation]

**Severity** Major  
**Blocking** Yes  
**Claim pointer** 联合精度预算是一项有实际 serving 价值的系统方法，而不仅是 allocator capacity characterization。  
**Evidence pointer** `main.tex` lines 81-94, 115-119, 479-492, 523-572, 606-619 and 637-646.  
**Concern** 当前稳定证据停留在 allocator token ceiling。唯一重复的 sustainable-boundary 差异未在第二次运行重现，ShareGPT 的边界方向相反，60 个 serving 单元均未通过 BH-FDR，尾延迟机制结果不稳定，四配置主门控有 183/720 个比较超出容差。固定条件下的 joint-minus-full throughput 甚至为负。selector 验证又只覆盖一个 2B/4K/Random/TP=1 切片，并明确没有证明所选配置优于基线。  
**Why it matters** 更多可分配 token 或序列槽不自动等于更高 SLO goodput、更低成本或更好的用户体验。当前标题、贡献列表和系统叙事要求读者相信一个 end-to-end serving 方法，但论文自己的主要 serving 证据只支持“不稳定且任务相关”。这使系统重要性与实际证据发生断裂。  
**Resolution test** 二选一。第一种做法是在代表性 workload、模型和负载区间中，用相同预算和协议比较 full、KV-only、state-only 和 joint，并得到预先定义、可重复的用户可见结果。第二种做法是将标题、摘要和贡献重构为 hybrid allocator capacity characterization，删除尚未建立的 selector 与 serving 系统价值暗示。

### R3-M4 [originality, novelty-significance]

**Severity** Major  
**Blocking** Yes  
**Claim pointer** 论文的原创贡献是首次将已有的 state dtype 开关作为与 KV bit-width 联合分配的系统预算维度。  
**Evidence pointer** `main.tex` lines 76-79, 124-141 and 154-171, together with equations at lines 176-215.  
**Concern** 稿件承认 state compression 是 prior art，vLLM 已暴露 state-dtype switch，已有工作也研究 state compression、KV quantization 和 budget trade-off。当前对差异的说明主要是“此前没有把两者联合核算”。论文没有提供 nearest-work comparison，也没有说明 selector、allocator probe 或运行时集成中哪些机制超出了现有开关、穷举候选和加法内存公式。  
**Why it matters** 对一般 MLSys 读者而言，\(AL+G\) 是自然的容量分解，约束过滤加 argmax 也是标准控制逻辑。若没有具体的系统机制、实现差异或先前方法无法完成的任务，读者无法判断这是原创系统工作，还是对一个现有 dtype 开关的认真测量。  
**Resolution test** 增加针对最近相关工作的能力对比，至少覆盖预算变量、支持的模型、质量约束、在线或离线决策、可执行部署和验证范围。明确列出新实现机制和非平凡工程工作。若贡献实质上只是首次测量与核算，则相应降低原创性措辞并把价值定位为经验性 characterization。

### R3-M5 [writing-clarity, interdisciplinary-readership]

**Severity** Major  
**Blocking** No  
**Claim pointer** 摘要和引言应让一般 MLSys 读者迅速理解问题、方法、主结果和边界。  
**Evidence pointer** `main.tex` lines 30-53, 81-119, 231-280, 530-572 and 585-646.  
**Concern** 摘要连续引入 112-cell、52 paired cells、residual range、preregistered no-think RULER、30 exact outputs、144/144、183/720 和 stability gate。引言又重复 R2-to-R3、repair、formal run 和 frozen tolerance 等审计术语。Serving 与 Limitations 继续使用 Gate 4、M3、M4、parent、temporal 和 formal 等仅对作者实验管理流程有意义的标签。真正的 takeaway 被大量防御性限定淹没。  
**Why it matters** 严谨限定是优点，但读者不应先理解作者的内部审计体系才能理解贡献。当前结构让跨子领域读者难以区分一级结论、验证证据、失败实验和仅用于 provenance 的细节，也挤压了 selector 与系统实现的解释空间。  
**Resolution test** 重写摘要为问题、方法、一个主结果和一个边界。引言只保留语义化证据层级，将 Gate 编号、运行包名称、逐门控计数和完整失败审计移至方法附录或 artifact。陌生读者读完摘要后应能准确复述固定 state 为什么重要、做了什么、容量提高多少、质量和性能结论到哪里为止。

### R3-M6 [figures-and-tables, writing-clarity]

**Severity** Major  
**Blocking** No  
**Claim pointer** 图表应按概念、机制、质量和 serving 证据的顺序支持 Results 中的论证。  
**Evidence pointer** `main.tex` lines 312-360, 363-474 and 494-583; current `main.pdf` pages 5-10; `main.log` lines 622-708.  
**Concern** 当前 PDF 将 block granularity 编为 Figure 1 并放在第 5 页，而解释整体问题的 precision-budget 概念图成为 Figure 2 并在第 6 页才出现。质量图被推至 Limitations 和 Conclusion 附近，per-layer sensitivity 位于参考文献页，serving 图直到参考文献后段才出现。结果段首次引用顺序因而成为 Figure 2 后接 Figure 1。编译日志还报告 figure/table destination 重复，PDF 内部链接可能指向错误目标。  
**Why it matters** 读者无法在阅读相应论证时看到证据，概念图也失去建立阅读框架的作用。八张图中多张是次级诊断或负结果，却占据主文浮动体资源并把核心图推离正文。这是当前 PDF 的实际可用性问题，不只是美观偏好。  
**Resolution test** 让概念与 headline capacity 图成为第一张图并靠近首次引用。将核心质量图和 selector 决策展示放在对应 Results 小节，将 per-layer、per-seed、harness 和完整 serving audit 移至附录。重新编译后，图的编号、首次引用和页面顺序应单调一致，且日志中不存在重复 destination。

### R3-M7 [claim-moderation, consistency]

**Severity** Major  
**Blocking** No  
**Claim pointer** 可见提交材料应对质量代价和 serving 复现等级给出一致结论。  
**Evidence pointer** `figure_contract.md` lines 3-6 and 25-31; `main.tex` lines 42-45, 106-119, 269-271, 530-547 and 637-644.  
**Concern** 图契约将核心结论写成“质量代价在统计分辨率内不可检测”，但主稿报告 2B GSM8K 有显著 1.0-point 回退。图契约还称 serving goodput 增益在“独立复现”中重现，而主稿明确说明第二次运行使用相同 contracts 和 seeds，只能视为 run-to-run stability，不是 independent sample。  
**Why it matters** 这两处差异都改变读者对证据强度的理解。若这些文件随匿名提交可见，它们会造成 packet-level claim inconsistency，并使图的设计目标偏向已经被主稿否定的 headline。  
**Resolution test** 删除不属于投稿包的内部契约，或将其更新为与主稿完全一致的 task-dependent quality 和 temporal rerun 表述。全包搜索后不应再出现“质量代价不可检测”或 serving “独立复现”的无范围限定说法。

## Minor Comments

### R3-m1 [submission-format]

**Severity** Minor  
**Blocking** No  
**Claim pointer** 当前 PDF 应使用 MLSys 2026 的正式匿名模板并生成可靠的导航结构。  
**Evidence pointer** `main.tex` lines 14, 22-26 and 677; `main.pdf` metadata; `main.log` lines 504 and 613-708.  
**Issue** 稿件加载并使用 `mlsys2025` 样式和 bibliography style，PDF metadata 的 Subject 也是 `mlsys 2025`。匿名页仍生成 Anonymous Institution、Country 和 email 脚注。日志包含空 anchor 和重复的 table/figure destinations。  
**Why it matters** 即使正文页数看似在当前版式下可容纳，错误模板可能改变页限、间距和匿名要求，重复 anchor 还会破坏审稿导航。  
**Resolution test** 使用官方 MLSys 2026 模板重新编译，核对正文、参考文献和附录的计页规则，清除占位脚注、空 anchor 和所有 duplicate destination 告警。

### R3-m2 [writing-clarity]

**Severity** Minor  
**Blocking** No  
**Claim pointer** 关键概念和实验标签应能被非本项目成员独立理解。  
**Evidence pointer** `main.tex` lines 31-37, 84-87, 99, 202-215, 245-279, 406-418 and 637-646.  
**Issue** GQA、TTFT、TPOT、FWE、NIAH 和 SLO-goodput 没有在首次使用时完整定义。R2-to-R3、Gate 4、M3、M4、parent、repair、formal 和 temporal 是内部实验管理标签，而不是通用系统术语。  
**Why it matters** 这些标签正好出现在摘要、贡献列表和证据等级判断处，会阻断跨领域读者理解。  
**Resolution test** 展开通用缩写并给出一句语义定义。用 “independent temporal rerun”“capacity reproducibility criterion” 等自解释表达替代内部编号，或在一张 protocol table 中统一定义。

### R3-m3 [figures-and-tables]

**Severity** Minor  
**Blocking** No  
**Claim pointer** 表 1 是读者理解代表性 capacity slice 的主要数值入口。  
**Evidence pointer** `main.tex` lines 281-304 and 312-320.  
**Issue** 表中包含四个 int4 单元，却只有三个 fp16 单元，缺少 9B/16K/fp16，正文和 caption 均未解释。`fp32 tok` 和 `bf16 tok` 也容易被误读为每请求 token，而实际是总 token ceiling。  
**Why it matters** 一个未解释的不完整 factorial slice 会让读者怀疑选择性展示，并增加对容量单位的理解成本。  
**Resolution test** 补齐缺失单元，或明确说明其缺失原因。将列名改为 total token capacity，并同时给出或链接对应 concurrent-sequence count。

### R3-m4 [figures-and-tables, accessibility]

**Severity** Minor  
**Blocking** No  
**Claim pointer** 图中显著性、配置和单位不应依赖颜色或上下文猜测。  
**Evidence pointer** `main.tex` lines 396-428, 466-472 and 574-581; Figures 2, 4 and 6 in the supplied PDF assets.  
**Issue** GSM8K caption 依赖 “red intervals”，per-layer 图依赖 orange rings，serving 图的 paired goodput change 未在纵轴或 caption 中明确写出 req/s 单位。若灰度打印或存在色觉差异，部分编码会减弱。  
**Why it matters** 审稿 PDF 和打印件常以缩放或灰度阅读。单位和统计状态必须由形状、文字或线型冗余表达。  
**Resolution test** 为显著结果增加不同 marker 或显式符号，为运行类型使用形状和线型双编码，并在 Figure 4 纵轴与 caption 中写明 goodput delta 的单位。

### R3-m5 [statistical-rigor, writing-clarity]

**Severity** Minor  
**Blocking** No  
**Claim pointer** 2B GSM8K 的 1.0-point 回退应以不会误导一般读者的方式解释。  
**Evidence pointer** `main.tex` lines 379-390 and 637-640.  
**Issue** 文中并列报告 effect 小于 preregistered MDE、显著 \(p=0.025\) 和 67.5% observed power，却没有解释 MDE 是设计阶段概念，也没有说明 observed power 对已观察结果增加了什么信息。  
**Why it matters** 一般读者可能误以为“小于 MDE”削弱了显著结果，或把 observed power 当作另一项独立证据。  
**Resolution test** 以效应量和 95% CI 为主要解释，说明预设功效对应的设计假设。若 67.5% 是基于观察效应计算的 post hoc power，应删除或明确标记其有限解释价值。

## Technical failings that need to be addressed before the case is established

阻断当前中心论证的问题是 R3-M1、R3-M2、R3-M3 和 R3-M4。作者需要使完整容量证据可检查，使 selector 决策可重建，在稳定 end-to-end outcome 与较窄的 capacity-characterization 定位之间做出明确选择，并建立相对于已有 state dtype 和 compression 工作的具体原创性边界。

## Assessment against Nature-style criteria

**Originality**  
联合考虑 KV 和 recurrent state 的部署视角具有潜在原创性，但稿件尚未说明其非平凡性。现有开关、已有 state compression 和简单加法预算之间的差异仍主要由文字断言支撑。

**Scientific importance**  
在短上下文、高并发、状态占比较高的场景中，15.44% median capacity gain 和代表性单元的 247 个额外序列槽可能具有实际重要性。重要性目前受单模型家族、单 GPU、缺乏稳定端到端收益和不可见完整矩阵限制。

**Interdisciplinary readership**  
直接受众明确，但广泛 MLSys 读者需要更清楚地看到架构变化、资源模型、系统动作和用户结果之间的链条。当前证据审计语言多于系统解释，跨子领域可达性不足。

**Technical soundness**  
可见容量模型是连贯的，作者对质量和 serving 失败的披露也较可信。完整容量证据、selector 决策链、artifact 可访问性和端到端 serving 价值仍不足，因此完整技术主张尚未建立。

**Readability for nonspecialists**  
概念图本身较清楚，但出现过晚。摘要和主线包含过多内部 Gate、运行包和统计审计标签，图也被推入 Limitations、Conclusion 和 References。当前稿件不满足非专门读者快速判断贡献和证据强度的要求。

## Recommendation posture

当前不支持接收。更合适的姿态是大幅修改后重新评估。若作者解决四个 Blocking concern，并将论文主线收束为一个可核查的容量与决策故事，这项工作可能成为一篇有价值的 MLSys 系统研究。当前版本则同时高估了 selector 与 serving 证据的完成度，又低估了清楚呈现核心容量观察的重要性。

本 Reviewer 3 报告已在上述评审边界内冻结。

<!-- FROZEN REVIEWER 3 REPORT END -->

## Cross-review synthesis (post-review; not shown to reviewers)

### Consensus strengths

- 三位评审都认可固定 per-sequence recurrent state 会改变 hybrid model 的 cache economics，尤其在短上下文、高并发、内存受限区间。
- 三位评审都认为作者对负结果的披露较好。2B GSM8K 回归、RULER 的非等价性边界、PPL harness 近似和 serving stability failure 均没有被隐藏。
- 可见的 allocator 方向、简单的 `AL+G` 直觉和七个展示单元构成一个有价值但范围较窄的 empirical characterization。

### Consensus blocking concerns

#### S-B1 Allocator ceiling 尚未成为真实 serving capacity

**Mapped concerns** R1-M1、R1-M3、R2-M2、R3-M3。

论文的 657、904 和 `+247 concurrent sequence slots` 来自 allocator token ceiling 除以 4096。probe 最终调用 vLLM 的 KV cache capacity calculation，它建立的是 cache-pool capacity，不是 scheduler 已接纳并完成 decode 的并发请求数。稿件没有展示接近 904 个 length-4096 requests 的 admission、prefill、decode、完成率、OOM boundary 或 SLO。四配置 serving gate 又以 183/720 个连续 goodput 比较超出 10% tolerance 失败，60 个 serving cells 无一通过 BH-FDR。当前证据因此只能支持 allocator-equivalent capacity，不能支持稳定可用的并发或 end-to-end benefit。

#### S-B2 核心 headline 与容量模型不可独立复算

**Mapped concerns** R1-M2、R2-M4、R3-M1。

摘要与结论依赖完整 52 pairs、15.44% median gain、1.81% median absolute residual 和 residual range，但正文只显示七个单元。`mlsys2026ae` 实际指向 Artifact Evaluation 征稿页，不是本文 artifact。`verify_figure_data.py` 依赖论文目录之外的 `results/...`，主要打印 ledger，没有冻结 expected values 或 mismatch assertions，也不核验 PDF/rendered figures。

容量模型还有一个具体的内部不一致。`main.tex` 第 689 行称 `G_bf16` 是 `G_fp32` 的一半；实际脚本使用 2B 的 `G_fp32 = 18 x 1,085,440` 和 `G_bf16 = 18 x 561,152`。每层 1,085,440 bytes 来自 36,864 bytes 的 bf16 convolution state 加 1,048,576 bytes 的 fp32 temporal state。切换后只有 temporal state 减半，convolution state 仍为 bf16，因此 561,152 并不是 1,085,440 的一半。padded pages 又是 1,089,792 与 566,016 bytes。附录当前公式无法重构脚本实际预测，`A_q` 也没有明确包含 int4 metadata 与 alignment。

#### S-B3 Selector 的正确性、最优性与系统价值未建立

**Mapped concerns** R1-M4、R2-M3、R3-M2。

selector 只展示 2B/4K/Random/TP=1 下 strict、medium 和 high 三次映射。论文没有完整列出预算值、四个候选的 dtype 定义、profile provenance、quality lower bound、TTFT/TPOT upper bound、capacity、goodput lower bound、可行性、拒绝原因和 argmax margin。也没有与 manual、always-bf16、单轴策略、greedy 或 exhaustive oracle 比较，更没有 held-out evaluation。当前证据证明的是离线查表和命令可执行，不是 selector 正确、最优或有实际 regret/SLO 优势。

#### S-B4 原创性边界不足以支撑 MLSys 系统贡献

**Mapped concerns** R2-M1、R3-M4，并与 R1 的 overall originality assessment 一致。

vLLM 已有 state dtype switch，state compression、KV quantization 和 budget trade-off 也有 prior art。当前新意主要由 `M/(AL+G)`、四候选枚举和“此前没有联合核算”的文字断言构成。没有 nearest-work capability matrix，也没有说明新的 runtime mechanism、allocator change、online policy 或既有系统不能完成的行为。若不能展示非平凡系统能力，最可信的定位是首次系统化测量和经验 characterization，而不是新 serving system。

### Other consensus major concerns

#### S-M1 Quality map 的统计基础和推广范围不足

**Mapped concerns** R1-M5、R1-M6、R2-M5，另见 R2-m3、R3-m5。

GSM8K 的九个 seeds 各从同一 1,319-item test set 抽 200 items。1,800 次抽取只有 1,017 个 unique items，seed pair overlap 为 19 至 39，median 30。当前 analysis 将九个 seed means 当作 iid paired observations。重复 item 在 greedy decoding 下仍出现配置内不一致，fp16 为 30 个 items，state-bf16 为 27 个，因此统计独立单位与 estimand 都不清楚。当前 seed deltas 为 `-1, -3, 0, -2.5, 0, 0, -1, -1, -0.5` percentage points。一个直接按 item 汇总重复观测的诊断得到 29 gains、37 losses、66 个非零方向 items，平均差约 `-0.778 pp`。这不是替代性的正式检验，因为重复 item 的输出为何不一致本身仍需解释，但它足以证明当前 paired t-test、MDE 和 observed power 需要重构。PPL 从 chunk 128 的约 19.35 变到 chunk 1 的 36.16，也不能验证真实 per-token kernel semantics。

#### S-M2 结论外推范围过宽

**Mapped concerns** R2-M6、R3-M3，并与 R1 的 scientific importance 判断一致。

现有结果覆盖 Qwen3.5 2B/9B、单 RTX 5090、单 GPU 和有限 workload。跨 hybrid architecture、另一类 GPU、tensor parallel、prefix caching 与 offloading 均未建立。若不扩展实验，标题、摘要和结论应一致限定为 Qwen3.5/vLLM/RTX 5090 的 offline allocator characterization。

#### S-M3 证据叙事、图序与提交包一致性需要重构

**Mapped concerns** R3-M5、R3-M6、R3-M7，另见 R1-m2、R1-m3、R1-m5、R1-m6、R2-m1、R2-m5。

摘要和主线被 Gate、R2-to-R3、M3/M4、parent、repair、formal、temporal 等内部实验管理标签占据。核心概念图出现较晚，质量与 serving 图远离首次引用，部分图进入参考文献区域。`figure_contract.md` 仍称质量代价不可检测、serving 为独立复现和 int4 是 conservative lower bound，与主稿现有边界不一致。`main.tex` 第 315 行还把包含 16K cell 的 7.0% 写成 `at 4K`，Table 1 缺 9B/fp16/16K 且未解释。

### Where emphasis differs across reviewers

- Reviewer 1 对统计单位、PPL harness 和机制归因最严格，认为 GSM8K 的独立性问题直接阻断 selector 的 quality lower bounds。
- Reviewer 2 对原创性和系统意义权重最高，认为即使局部 allocator 结果成立，当前仍可能只是已有 dtype switch 的系统化测量。
- Reviewer 3 对跨领域可读性与证据编排权重最高，将摘要的审计术语密度和整篇图序列为 Major 但非 Blocking。
- Reviewer 3 的冻结报告把 `mlsys2025` style 当成可能错误模板。事后核查 MLSys 2026 官方 CFP 表明 2026 明确使用 2025 style，因此保留冻结原文，但不把 style 文件名本身列为最终缺陷。真正的提交风险是主文页限与附录上传规则。

### Minor revision checklist

- 精确定义 full、KV-only、state-only、joint 和 fp16-state controls，并列出完整 engine arguments。R1-m1、R2-m2。
- 修正 `7.0--15.4% at 4K`，补齐或解释 9B/fp16/16K，并将 `fp32 tok`、`bf16 tok` 改为 total token capacity。R1-m2、R3-m3。
- 定义 goodput 的公式、单位、失败请求处理、观察窗口和 confidence-bound estimator。R1-m4、R3-m4。
- 删除或谨慎解释 post hoc observed power，以 effect size 和 CI 为主。R2-m3、R3-m5。
- 不要由校正后不显著推出 whole-state granularity 最优。R2-m4。
- 重排 floats，使首次引用、编号和页面顺序一致，改善拥挤 annotation 与灰度可访问性。R1-m5、R2-m5、R3-m4。
- 消除 empty anchor 和 table/figure duplicate destination，检查所有 PDF hyperlinks。R1-m6、R2-m6、R3-m1。
- 展开 GQA、TTFT、TPOT、FWE、NIAH 等缩写，用语义名称替代内部 Gate 与 run package 标签。R2-m1、R3-m2。

### Submission-format audit

- MLSys 2026 官方 CFP 指定使用 2025 style，因此 `mlsys2025.sty` 本身不是错误。
- main paper 限 10 页，references 不计入；appendix 必须作为 separate upload。当前 `main.pdf` 第 11 页包含附录，提交前需拆分并核对主文页数。
- 当前 PDF 可编译且字体已嵌入。
- `main.log` 报告 Table 1、Table 2、Figure 1 至 Figure 8 的 duplicate destinations，并有 empty anchor。这会影响 PDF 导航，应修复。
- 官方规则核查来源 `https://mlsys.org/Conferences/2026/CallForPapers`，核查日期 2026-08-14。

### Broad-interest / significance readout

直接受众明确，包括 LLM serving、vLLM、hybrid recurrent-attention、KV/state quantization 和内存管理研究者。固定 state 改变短上下文 cache economics 是一个真实、易解释的观察。广泛 MLSys 影响仍是潜力，因为论文没有证明更多 allocator blocks 会转化为稳定 SLO goodput、成本下降或跨架构决策优势。

### Most important issues to resolve before a strong case is established

1. 决定论文身份。若是系统论文，就用真实 memory-saturated workload 建立 allocator ceiling 到 admission、decode、SLO goodput 和成本的完整链条。若做不到，就将标题和贡献收缩为 allocator characterization。
2. 修正并完整公开离散 capacity model。解释 conv state、temporal state、page padding、metadata、alignment、scheduler cap 和 integer rounding，并让论文公式与脚本逐字节一致。
3. 提供真正匿名且可独立运行的 artifact。纳入完整矩阵、原始 cell、环境、selector、绘图和 assertion-based verifier。
4. 将 selector 变成可证伪的系统组件。公开全部候选 decision trace，与 oracle 和简单基线比较，并在 held-out strata 上报告 regret、SLO violations 与 profiling overhead。
5. 重做 quality inference。使用 item-level matched outcomes 或适合重复抽样的 cluster/hierarchical 方法，预先定义 non-inferiority margin 与 multiplicity family，再重新生成 selector quality bounds。
6. 建立原创性差异。用 capability matrix 和实现证据说明相对已有 vLLM dtype switch、state compression 和 KV-only methods 的非平凡增量，否则主动降格为 empirical characterization。

## Risk / unsupported claims

- **Unsupported operational claim** `657 -> 904 concurrent sequences` 和 `+247 slots` 目前只是 allocator-equivalent extrapolation，不是实跑并发。
- **Unsupported system-value claim** 当前 serving 主门失败，不能声称 joint precision 已带来稳定 end-to-end goodput 或 SLO benefit。
- **Unsupported selector claim** 三个预期映射不能证明 correctness、optimality、fail-closed completeness 或泛化。
- **Unsupported novelty claim** 尚未证明超出现有 state dtype switch、简单内存 accounting 和 offline lookup 的新系统能力。
- **Unsupported reproducibility claim** `mlsys2026ae` 不是本文 artifact，当前 verifier 也不是 assertion-based independent reproduction。
- **Internally inconsistent model claim** `G_bf16 = G_fp32/2` 与实际脚本和 state layout 不一致，因此 published prediction 不能从附录重构。
- **Statistically unsupported quality bound** 九个 GSM8K seed means 不是清楚的 iid units，重复 item sampling、multiplicity 与 estimand 需重建。
- **Unsupported mechanism claim** raw throughput cells 的 run-to-run stability 不等于 joint-minus-full treatment effect 被复现，九个机制检验也未通过 Holm correction。
- **Unsupported generality claim** 当前证据不能推广到 hybrid linear-attention serving 整体、其他 GPU、tensor parallel 或其他架构。
- **Packet-level inconsistency** `figure_contract.md` 中 undetectable quality cost、independent serving reproduction 和 conservative lower bound 等说法已被主稿自己的结果否定或降级。

## Final reviewer-mode verdict

这不是对中心想法的否定。固定 recurrent state 改变 hybrid serving 的内存经济性，值得发表和传播。问题在于当前稿件用 allocator characterization 承担了系统论文、selector、质量策略和 end-to-end serving 四层主张，而后三层证据没有闭合。

按当前版本，三位评审的一致姿态是 **Reject / not ready**。最短的可信改稿路线是把论文收束为可完全审计的 Qwen3.5/vLLM allocator characterization，修正模型与统计，并删除未建立的系统收益。若坚持 Joint Precision Budgeting 作为系统论文，则需要新增实质性端到端实验、selector baselines 与完整 artifact，不能仅靠措辞调整解决。
