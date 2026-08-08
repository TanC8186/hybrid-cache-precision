# Evaluation 章节补跑文本框架（2026-08-07，数字已回填并核对哈希）

> 数字来源：`results/quality/niah-fixed-analysis.json`、
> `results/quality/ruler-subset-analysis-v2-256.json`、
> `results/quality/reasoning-nothink-v2-analysis.json`、
> `results/quality/longbench-analysis-20260807.json`。
> 以下数字均由对应 analyzer 产出，原始 cell 哈希已核对；serving Formal 与
> C4/PG19 PPL 未跑，相关小节以显式状态标注，不设占位数字。

## 1. Retrieval（NIAH rerun，max_tokens=256）

使用与 R4 完全相同的 18-cell 网格（3 depths × 2 lengths × 3 seeds × 3 needles），
生成上限从 32 提升到 256 token，并逐样本记录 `hit_final`（`</think>` 之后的命中）。
在 `max_tokens=256` 下，`<think>` 截断伪影基本消失：fp16 0.9630，uniform int4 0.9815，
packed per-layer 0.9815，TurboQuant k8v4 0.9630，TurboQuant 4-bit NC 0.9444。
配对差（vs fp16，18-cell t-CI）全部包含 0：uniform +0.0185 [−0.0206, +0.0576]；
packed +0.0185 [−0.0206, +0.0576]；k8v4 0.0000 [−0.0804, +0.0804]；
4bit_nc −0.0185 [−0.0875, +0.0505]。严格最终答案命中（hit_final）排序与 R4/R5
一致（packed 0.9259 > fp16/uniform 0.9074 > 4bit_nc 0.8889 > k8v4 0.8519）。
状态：ANALYZED（90/90 完成，哈希全匹配；独立复现未跑）。

## 2. Long-context（RULER subset，noise haystack）

使用官方 RULER 生成器（commit `c3f5e3b4`）与官方 `string_match_all` 评分，
任务覆盖 NIAH-single/multikey/multivalue/multiquery、variable tracking、common/freq words
extraction，长度 4096/8192，每 cell 20 samples、greedy、单 seed。第一轮使用官方
`tokens_to_generate`（v1，70/70 完成）：NIAH single 全部 100；packed 在 CWE 4K 点估计
最高（98.5）；TurboQuant 在 CWE 4K 明显回落（k8v4 65.5 / 4bit_nc 67.0 vs fp16 97.5）。
**FWE 的 0 分与 VT/multiquery 部分 miss 是 `<think>` 消耗官方短预算的协议伪影**
（已逐样本抽查），v1 保留为协议敏感性数据；第二版以 max_tokens=256 重跑全部 70
cells（attempt `ruler-subset-20260807-v2-256`）作为主报告协议，70/70 完成。
v2 要点：NIAH single 全部 100；multikey/multivalue 4K 中 TurboQuant 低 5–6.25 分；
multiquery 8K packed 最高（88.75，fp16 86.25，k8v4 80）；VT 4K TurboQuant 低 1–2 分、
8K 基本持平；**CWE 4K TurboQuant 掉分在 256 token 下仍存在**
（k8v4 69.0 / 4bit_nc 70.0 vs fp16 97.5，−28.5/−27.5），确认为真实质量退化而非截断伪影。

**FWE 专项重跑（fp16/uniform/packed，max_tokens=256，6/6 完成）**：4K 得分
15.0 / 28.33 / 20.0，8K 得分 41.67 / 61.67 / 55.0。逐样本审计表明 miss 仍以
`<think>` 未输出答案为主，量化列“更高”来自思考文本顺带命中目标词，属协议伪影，
不构成质量结论；FWE 建议从主表格中排除或仅作披露性附注。

**FWE 禁用 thinking 版（enable_thinking=False，6/6 完成）**：4K 93.33 / 88.33 /
88.33，8K 100.0 / 100.0 / 100.0（fp16 / uniform / packed）。禁用思考后 FWE 可解读：
8K 完全持平，4K 量化点估计低 5 分；作为 FWE 主报告协议。

## 3. Serving（TurboQuant/FP8，protocol-v3）

状态：**Formal 未跑**（需人工放行，约 8h）。计划契约：protocol-v3（PIECEWISE、
Random60/ShareGPT300、warmup 120、TTFT {250..3000}ms、TPOT 200ms、
goodput/offered ≥ 0.95、3 seeds），比较 fp16 / int4 / packed per-layer /
TurboQuant k8v4 / TurboQuant 4-bit NC / FP8，并报告 P99 TTFT/TPOT 与容量探针。
当前仅有门禁阶段 pilot 边界（ANALYZED，未独立复现）：

| workload | TTFT | fp16 | int4 | packed per-layer |
|---|---:|---:|---:|---:|
| Random60 | 250 ms | 30 | NONE* | 30 |
| Random60 | 500 ms | 35 | 35 | 35 |
| Random60 | 1000+ ms | 35 | 40 | 40 |
| ShareGPT300 | 250 ms | 45 | 35 | 40 |
| ShareGPT300 | 500+ ms | 45 | 40 | 40 |

*NONE = 测试网格内无 3-seed 全可持续点。TurboQuant/FP8 正式边界、P99 表与容量探针
在 Formal 完成后回填；当前不设占位数字。

## 4. Reasoning（2B 子集）

GSM8K（前 200）、MMLU（all/test 前 500）、AIME2025（全 30），greedy、单 seed。
主协议：chat template `enable_thinking=False` + 大预算（1024/512/4096）+ 最终答案
抽取（最后一个 answer/result 标记后的候选；无标记退回全文本最后候选并记录）。
（thinking 模式与小预算版本因截断伪影保留为敏感性数据，禁用作主协议。）

| bench | n | fp16 | uniform int4 | packed per-layer | TQ k8v4 | TQ 4-bit NC |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K | 200 | 0.760 | 0.695 | 0.680 | 0.675 | 0.685 |
| MMLU | 500 | 0.588 | 0.586 | 0.596 | 0.600 | 0.612 |
| AIME2025 | 30 | 0.167 | 0.100 | 0.167 | 0.067 | 0.100 |

结论边界：GSM8K 量化列点估计低 fp16 6.5–8.5pt（单 seed、无 CI，需多 seed 确认）；
MMLU 量化列与 fp16 持平或略高（capped 99–118/500 各分配对称，严格最终答案口径
仍持平）；AIME2025 在 4096 预算下仍有 20–22/30 未完成，各列均近地板，仅报告
"各列无显著优势"。证据：`results/quality/reasoning-nothink-v2-analysis.json`。

## 5. LongBench v1 子集（2B 全 5 分配 + 9B 核心 3 分配）

LongBench 8 个英文任务（TREC/TriviaQA/SAMSum/LCC/RepoBench-P/GovReport/QMSum/
MultiNews），每任务前 50 样本、greedy、单 seed、no-think；官方 v1 prompt 与
指标（GitHub commit `4c4b985bcf`）；数据为 v1 parquet 镜像（原 JSONL revision
已下线，见 data/MANIFEST.yaml）。分数表见
`results/quality/longbench-analysis-20260807.json`。

要点：2B 上 QA 与摘要量化列与 fp16 持平（多数 ±1pt 内），TREC 的 uniform/packed
点估计低 6pt；9B 上 QA/摘要持平，代码任务量化列点估计一致略低（LCC −1.8/−2.6，
RepoBench-P −2.0/−5.2）。单 seed、无 CI，全部按点估计披露，禁止写成显著退化；
2B 代码任务接近地板，不用于结论。截断（中段截断至 15.8K token）各分配完全对称。
证据：`results/quality/longbench-analysis-20260807.json`。

## 6. PPL 扩展语料（C4/PG19，2B + 9B）

与 Wikitext-2 相同协议（3 seeds × 5×2048，`hybrid_premise.py`），语料为固定切片
（`data/c4_slice.txt`、`data/pg19_slice.txt`，sha 见 `data/MANIFEST.yaml`）：

| 模型 | corpus | fp16 | uniform | packed |
|---|---:|---:|---:|---:|
| 2B | C4 | 17.58 | 17.87（+0.29 [+0.21, +0.37]） | 17.78（+0.20 [+0.14, +0.25]） |
| 2B | PG19 | 27.18 | 27.62（+0.44 [+0.29, +0.59]） | 27.42（+0.25 [+0.10, +0.40]） |
| 9B | C4 | 12.73 | 12.87（+0.14 [+0.08, +0.19]） | 12.87（+0.14 [+0.11, +0.17]） |
| 9B | PG19 | 18.00 | 18.22（+0.22 [+0.12, +0.32]） | 18.21（+0.21 [+0.12, +0.30]） |

Δ 均为 3-seed 配对 95% t-CI。结论：第二、第三语料上量化列相对 fp16 增加约
1.1–1.6%（2B）与约 1.1%（9B），packed 点估计普遍略优于 uniform；与 Wikitext-2
结论一致，质量门禁通过。证据：`results/quality/ppl-extra-analysis-20260807.json`。
