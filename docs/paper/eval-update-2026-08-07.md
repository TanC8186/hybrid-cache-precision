# Evaluation 章节补跑文本框架（2026-08-07，数字待回填）

> 数字来源：`results/quality/niah-fixed-analysis.json`、
> `results/quality/ruler-subset-analysis.json`、`results/quality/reasoning-analysis.json`、
> `results/quality/r5-serving-analysis.json`。回填前必须运行对应 analyzer 并核对哈希。

## 1. Retrieval（NIAH rerun，max_tokens=256）

（待回填）使用与 R4 完全相同的 18-cell 网格（3 depths × 2 lengths × 3 seeds × 3 needles），
生成上限从 32 提升到 256 token，并逐样本记录 `hit_final`（`</think>` 之后的命中）。
在 `max_tokens=256` 下，`<think>` 截断伪影消失：fp16 平均 accuracy（待回填），
uniform int4（待回填），packed per-layer（待回填），TurboQuant k8v4（待回填），
TurboQuant 4-bit NC（待回填）。配对差（vs fp16，18-cell t-CI）：待回填。

## 2. Long-context（RULER subset，noise haystack）

（待回填）使用官方 RULER 生成器（commit `c3f5e3b4`）与官方 `string_match_all` 评分，
任务覆盖 NIAH-single/multikey/multivalue/multiquery、variable tracking、common/freq words
extraction，长度 4096/8192，每 cell 20 samples、greedy、单 seed。
表格：待回填。

## 3. Serving（TurboQuant/FP8，protocol-v3）

（待回填）在 A2 protocol-v3 相同契约下（PIECEWISE、Random60/ShareGPT300、warmup 120、
TTFT {250..3000}ms、TPOT 200ms、goodput/offered ≥ 0.95、3 seeds）比较
fp16 / int4 / packed per-layer / TurboQuant k8v4 / TurboQuant 4-bit NC / FP8 的可持续边界，
并报告 P99 TTFT/TPOT 与容量探针（capacity tokens）。边界表：待回填。

## 4. Reasoning（2B 子集）

（待回填）GSM8K（前 200）、MMLU（all/test 前 500）、AIME2025（全 30），greedy；
抽取规则与子集规模如实披露。准确率：待回填。
