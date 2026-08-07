# Evaluation 章节补跑文本框架（2026-08-07，数字待回填）

> 数字来源：`results/quality/niah-fixed-analysis.json`、
> `results/quality/ruler-subset-analysis.json`、`results/quality/reasoning-analysis.json`、
> `results/quality/r5-serving-analysis.json`。回填前必须运行对应 analyzer 并核对哈希。

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
cells（attempt `ruler-subset-20260807-v2-256`）作为主报告协议，待回填。

**FWE 专项重跑（fp16/uniform/packed，max_tokens=256，6/6 完成）**：4K 得分
15.0 / 28.33 / 20.0，8K 得分 41.67 / 61.67 / 55.0。逐样本审计表明 miss 仍以
`<think>` 未输出答案为主，量化列“更高”来自思考文本顺带命中目标词，属协议伪影，
不构成质量结论；FWE 建议从主表格中排除或仅作披露性附注。

**FWE 禁用 thinking 版（enable_thinking=False，6/6 完成）**：4K 93.33 / 88.33 /
88.33，8K 100.0 / 100.0 / 100.0（fp16 / uniform / packed）。禁用思考后 FWE 可解读：
8K 完全持平，4K 量化点估计低 5 分；作为 FWE 主报告协议。

## 3. Serving（TurboQuant/FP8，protocol-v3）

（待回填）在 A2 protocol-v3 相同契约下（PIECEWISE、Random60/ShareGPT300、warmup 120、
TTFT {250..3000}ms、TPOT 200ms、goodput/offered ≥ 0.95、3 seeds）比较
fp16 / int4 / packed per-layer / TurboQuant k8v4 / TurboQuant 4-bit NC / FP8 的可持续边界，
并报告 P99 TTFT/TPOT 与容量探针（capacity tokens）。边界表：待回填。

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
