# Phase 2 输出 — Peer Reviewer 1（Methodology）

## 预注册与评分依据声明

本评审按 Phase 1 冻结的 Scoring Plan（`review-p1-meth-20260813.md`）执行，无维度异议，故省略 Scoring Plan Dissent 节。所有数字主张均与 `results/` 归档交叉核验（只读）。

contract_role: methodology

## Dimension Scores

### D1: methodology_rigor
score: warn

### D2: domain_accuracy
score: pass

### D3: argumentative_coherence
score: pass

### D4: cross_disciplinary_relevance
score: pass

### D5: writing_and_structure
score: pass

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

### Overall Recommendation

Accept（建议 minor revision）。论文的方法学证据链在 serving benchmark 领域属罕见的高标准：预注册门禁、配对设计、MDE/power、多重比较校正、run-stability 与 independent replication 的区分、门禁失败的如实披露，均执行到位且可独立核验。唯一 warn（D1）来自可修复的 provenance 缺陷，不危及已核验的 headline capacity 效应。

### Confidence Score

4

### Summary Assessment

方法学上是 serving benchmark 领域少见的严谨投稿。预注册端到端贯穿：各实验包冻结 contract 与 Gate 3/4 容差；GSM8K 预注册 MDE 1.16pp 并报告 67.5% 观察功效；per-layer 36 测试、serving 60-cell、mechanism 9 测试分别以 Bonferroni/BH-FDR、BH-FDR、Holm 校正，无未校正主张进入 headline；同 seed 配对设计贯穿质量与 serving 端点；run-stability 与 independent replication 明确区分（第二 formal run 同 contract/seed，仅作 run-stability 证据）；零宽 CI 三次声明不构成等价性检验；M4 门禁失败（537/720）如实披露为 serving instability 而非转化为性能结论。我独立核验并与 results/ 归档全部吻合：capacity 52/52 方向、15.44% 中位增益、残差统计（1.81%，−3.66~+13.21%）、Gate 4 复现最大 cell 差异 1.42%、GSM8K/RULER/controller 数字、60-cell q 值（0.066/0.052）、M4 门禁（537/720、79.610%）。两点保留：M3 mechanism 数字在归档中缺失，与结论中"full artifacts released"的声明矛盾；Table 1 未注明 attempt 来源，且 formal attempt 的减半 state 单元在归档中解析为 float16 而非 bf16。两者均不危及已核验的 capacity 效应。

### Strengths

1. **门禁式预注册且失败结果如实保留**（Sec. 4 各 package、Sec. 5.5）：四配置矩阵 144/144 完成但 Gate 4 失败（537/720 在 10% 容差内、183/720 越界、最大 79.61%），论文将其作为 serving instability 披露而非转化为性能 headline——这是本领域极少见的做法，且三个数字与我核验的 `gate4-r3/validation_report.md` 完全一致。
2. **配对设计与 seed 语义清晰**（Sec. 4 Quality/Serving、Sec. 5.2）：GSM8K 9 个配对数据集 seed、RULER 3 个配对 seed、serving 3 seed 同 cell 配对 delta；RULER 固定 engine seed 的理由（关闭 thinking 避免截断）有说明；"第二 run 使用相同 contract 与 seed → run-stability 而非 independent sample"（Sec. 5.5）明确区分了测量重复与独立复制的统计含义，没有把测量重复当独立样本。
3. **统计分辨率诚实**（Sec. 5.2、5.3、5.5）：GSM8K 低于预注册 MDE 的显著回归（1.00 点 < MDE 1.16 点）如实标注并同时报告功效 67.5%；per-layer 36 测试原始 p<0.05 的两个点在 Bonferroni/BH-FDR 后不成立，明确"无 per-layer 收益证据"；60-cell BH-FDR 全部未通过，报告最近 q 值（0.066/0.052）而非选择性引用；零宽 CI 在 Abstract、Sec. 5.2 RULER、Fig. 3 caption 三处声明"不构成等价性检验"。
4. **数字可核验性**：headline capacity 数字与归档逐一吻合——formal analysis JSON 给出 52/52 方向、中位增益 15.4400%、中位绝对残差 1.811%、残差范围 −3.661~+13.213%；Gate 4 报告最大 per-cell token 差异 1.415962%（论文 1.42%）；GSM8K MDE/power（0.0116/0.6753、0.0241/0.8828）、RULER 五格准确值与 30 个复现值、controller 复现（29.552/38.905）、serving r40/r45 delta 与 q 值全部与 results/ 归档一致。
5. **主张边界严格**（Abstract、Sec. 6、Sec. 7）：摘要本身即列出"verified capacity effect + scoped policy execution + disclosed instability"的边界；LongBench 单 seed pilot 证据被明确排除出 claim；TP 扩展标注为"an expectation, not a measurement"。

### Weaknesses

1. **M3 mechanism 隔离实验的数字在归档中缺失**（Sec. 5.5 Mechanism 段）。问题：论文报告 18/18（throughput）、10/18（TTFT P95）、17/18（TPOT P95）、最大差异 75.39%/12.13%、joint-minus-full −0.760/−0.815/−2.492 req/s、9 测试 Holm 校正不通过，但 `results/reproduction/2026-08-13/` 下只有 m4 与 ruler 两个目录，全库检索不到 mechanism 的 analysis/gate4 产物（仅存在未跟踪的 `configs/experiments/m3_mechanism_isolation_2b.yaml` 与 `scripts/analyze/analyze_m3_mechanism_isolation.py`）。为什么是问题：Sec. 7 宣称"We release the full measurement contracts, per-seed data, and reproduction artifacts"，mechanism 结果无法被第三方核验，违背论文自身的可审计承诺。建议：将 M3 的 gate4/analysis JSON 与 per-cell 数据归档进 results/ 并在正文给出路径；若实验仍在进行，正文应如实标注归档状态而非以完成态陈述。
2. **Table 1（tab:capacity）未注明数据来自哪一次 attempt，且 state dtype 标签存在不一致**（Sec. 5.1 与 Table 1 caption）。问题：formal attempt 与 clean R3 的 per-cell token 有最大 1.42% 差异，导致残差统计不同（formal：median |res| 1.81%、范围 −3.66~+13.21%；clean R3：2.24%、−3.55~+12.80）——论文引用的残差统计来自 formal attempt，但未说明；此外 formal attempt 的 u0.85 减半 state 单元在归档中 `resolved_mamba_ssm_cache_dtype=float16`（非 bfloat16），与论文"fp32/bf16"表述标签不一致（字节数相同，容量等价，但读者无法从论文看出这一点）。为什么是问题：在"四个一致"（代码/配置/日志/结果）的核验纪律下，表格来源不可唯一追溯即为 provenance 缺陷。建议：在 caption 注明 slice 的 attempt 来源与 state dtype 命名约定，或统一采用 clean chain 的数值并同步更新残差统计。
3. **Serving boundary 表（tab:serving）是单 cell 检测、无 CI，容易被误读为效应**（Sec. 5.5 首段）。问题：正文承认"single boundary cells are run-sensitive and are not claimed as effects"，但表格本身呈现为配置×阈值矩阵，没有不确定性信息，读者可能把 35 vs 40 的差异当效应。为什么是问题：这正是论文自己在别处防范的"表格与文字分离导致误读"模式。建议：在表 caption 中重复"boundaries are single-cell detections, not effects"一句。
4. **Mechanism 小节只有正文数字、无表格/图、无 CI**（Sec. 5.5 Mechanism 段）。问题：与全文其他小节（每图带 n、seed、CI）的统计报告标准不一致，18 cell 的复现通过率与最大差异只能从散落句子中拼出。为什么是问题：D5 意义上该节信息完整但呈现方式使核验成本高。建议：补一个紧凑小表（3 metric × 3 contrast 的通过率、max diff、均值差与 CI），与 weakness 1 的归档一并解决。

### Detailed Comments

**Research Questions & Hypotheses.** 研究问题（state precision 作为第二预算维度）可检验、被操作化为冻结的 gate 与容差（Sec. 3、4），primary endpoint（capacity 效应）预注册且全程未被替换；RULER 五格的筛选来自此前的 KV 量化 screen，论文声明"selection predates the state-dtype grid"（Sec. 5.2），方向正确，但应给出该 screen 的归档指针以便审稿人验证非 post hoc。

**Research Design.** 同硬件（单 RTX 5090）、同 seed 配对、冻结 worktree commit（55f4768 / e2fa285）的设计与主张匹配；capacity 用确定性 allocator probe，其无随机性、故无需 p 值/CI 的论证在 Gate 4 报告中明确陈述，正确；mechanism 固定 concurrency 对比改变排队约束、非纯 dtype 干预这一自我批评（Sec. 6 Mechanism attribution）显示了设计自觉。

**Sampling Strategy.** Seed 语义清楚：GSM8K 9 配对数据集 seed（唯一做正式推断的端点，有 MDE/power）、PPL 与 RULER 3 配对 seed、serving 3 seed。n=3 的 serving 端点只做方向性陈述（Sec. 5.5："directional evidence"），符合小样本推断边界。RULER n=3 的精确一致被明确定性为描述性而非等价性。抽样强度与所下结论的强度相称。

**Data Collection.** Provenance 链完整：per-cell JSON + sha256 sidecar、contract 含 config hash、resolved dtype 记录（capacity 单元含 `resolved_mamba_ssm_cache_dtype`）、M4 有 precision-log 检查；失败计入分母（SLO goodput）；formal 矩阵零失败请求。PPL chunk 级 harness 是近似（Sec. 6 Harness boundary，Fig. 7），已披露并降级为 supporting evidence——处理正确。

**Analysis Methods.** 配对 t 区间（GSM8K、serving delta）、Cohen's d_z、Bonferroni/BH-FDR（36 测试）、BH-FDR（60 cell）、Holm（9 测试）选择恰当且全部披露；无未校正的"显著"结论进入 headline；MDE/power 预注册于 GSM8K；零宽 CI 未误用。唯一方法学注脚：r40 的配对 t 检验（p=0.030/0.004）在 60-cell BH-FDR 失败后仍被呈现为"directional evidence"——论文已明确其证据权重来自 cross-run 同 cell 稳定性而非显著性，处理可接受。

**Results Presentation.** 摘要-正文-结论的数字一致性经逐项核验成立（15.44%、1.81%、−3.66~+13.21%、1.42%、1.0 点、183/720、537/720 等均与归档吻合且文内自洽）；2B/4K 的 2.2451x→2.6754x、657→904 并发序列、18.63 MiB state 等衍生数字我复算无误。保留项：Table 1 的 attempt 来源未注明（见 Weakness 2）；图表 caption 均带 n/seed/CI 信息，但本评审环境无法渲染 vector PDF，坐标轴完整性请作者自查。

**Reproducibility.** 属全稿最强部分：冻结 commit、contract、Gate 3/4 验证报告、sha256 sidecar、独立复现链（capacity R2→R3、RULER temporal attempt、controller gate4、M4 两 attempt）齐备；60-cell p/q 全量归档（`serving-direction-agreement-20260811.json`）。缺口即 Weakness 1：M3 mechanism 归档缺失，使"full artifacts released"（Sec. 7）对 mechanism 一节当前不成立。

**Methodological Fallacies Detected.** Gate 4 报告内置结构化 fallacy scan（Simpson's、look-elsewhere、garden of forking paths、survivorship、correlation≠causation 等），论文实践本身是反谬误的。未发现致命谬误；三类边界情形：RULER 筛选的选择过程依赖论文外的 screen 归档（需补指针）；Conclusion 的 "Operational rule" 将 2B/4K 数字推广为部署建议，但已用"target regime"限定，可接受；boundary 表无 CI 的呈现（Weakness 3）可能诱发误读。无 endpoint switching、无 strawman、无选择性呈现有利配置的证据。

### Questions for Authors

1. Table 1 的 4K/16K 切片取自哪一次 attempt（formal 还是 clean R3）？两次 attempt 的 per-cell token 最大差 1.42%，残差统计随之不同（formal 1.81% vs clean R3 2.24%）。论文引用的残差统计与 formal attempt 吻合，请注明来源，并考虑统一采用 clean chain 数值。
2. formal attempt 的 u0.85 减半 state 单元在归档中 `resolved_mamba_ssm_cache_dtype=float16`，而论文全程表述为 bf16。formal attempt 的减半臂是否实际以 float16 运行（与 bf16 字节等价，容量相同）？若是，请在正文明确"float16 与 bf16 state 在容量上字节等价"这一点，以消除标签与归档的出入。
3. M3 mechanism 隔离实验的 per-cell 数据与 Gate 4/analysis 产物归档在哪里？我在 `results/` 下未能定位；这与 Sec. 7 的完整发布声明不符。
4. RULER 五格的前置 KV 量化 screen 是否有归档产物可引用？请提供路径，以便验证"selection predates the state-dtype grid"。
5. Random60 overload 区 13/13 cell 两 run 同向（与 ShareGPT 7/10 反向形成对照），若未来想将其升级为 confirmatory endpoint，是否会先预注册再跑新 run？

### Minor Issues

- tab:serving 的 caption 建议加一行"boundaries are single-cell detections, not effects"（正文已有，图/表处重复以防误读）。
- Sec. 5.5 Mechanism 段建议补小表（3 metric × 3 contrast 的复现通过率、max diff），并给出归档路径。
- Sec. 4 Capacity probes 提到"0.70–0.90 in the capacity grid (plus 0.85 controls)"，建议说明 0.85 control 的用途（Table 1 的部署解释切片即来自 u0.85），使表格与 grid 的关系显式化。
- Sec. 5.4 Controller 建议给出三个冻结 budget 的具体请求参数（或指向归档 contract），使 strict→full / medium→state_only / high→joint 的映射可被读者独立判断合理性。
- Fig. 1 caption 称"(b) … (7 cells; gap labels are signed percent error)"，与 Table 1 的 7 行一致，但建议在 caption 中注明该图的 gap 来自哪次 attempt（与 Weakness 2 同源）。
- Sec. 5.2 GSM8K 与 Sec. 5.5 Serving 中 p 值与 CI 的报告格式（显著性判定 vs 方向性证据）建议在 Sec. 4 开头加一句统一的统计报告约定说明，减少相邻领域读者混淆。

## Editorial Decision

editorial_decision=accept
