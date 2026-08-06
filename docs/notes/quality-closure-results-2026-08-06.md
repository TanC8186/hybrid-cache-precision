# R4 质量闭环结果（2026-08-06，进行中）

> 状态：**完成**。PPL 3-seed + NIAH 54/54 样本，0 失败（含 MVEx 去重）。
> 证据：`results/quality/r4-ppl/*.csv.seeds.csv`（PPL）、`results/quality/r4-niah/`（NIAH）。

## 1. PPL（Wikitext-2，3 seeds，transformers canonical 协议）

入口：`scripts/exp/hybrid_premise.py --seeds 7,42,2026 --num-seqs 5 --max-len 2048`。
与 `byte_budget_3seed.log` 同协议；fp16 均值 11.4827 与 canonical 11.4832 一致。

| 分配 | 每 seed PPL (7/42/2026) | mean ± SD |
|---|---:|---:|
| fp16 | 9.7969 / 12.9118 / 11.7393 | 11.4827 ± 1.5732 |
| uniform int4 | 9.9869 / 13.1308 / 11.9256 | 11.6811 ± 1.5861 |
| packed per-layer (L23 bf16) | 9.9311 / 13.0048 / 11.8595 | 11.5985 ± 1.5534 |

配对 Δ（vs fp16，95% t-CI，df=2）：

| 对比 | Δ | 相对变化 |
|---|---:|---:|
| uniform int4 vs fp16 | +0.1984 [+0.1541, +0.2428] | +1.73% |
| packed per-layer vs fp16 | +0.1158 [+0.0637, +0.1679] | +1.01% |
| **packed per-layer vs uniform int4** | **−0.0826 [−0.1767, +0.0115]** | **−0.71%** |

结论：**packed per-layer 的容量恢复没有以 PPL 质量回退为代价**——相对 uniform int4 反而略优
（L23 保护生效），配对 CI 上界仅 +0.0115（≈+0.1%）。

## 2. NIAH（vLLM 离线贪婪，seed 化）

契约：depths {25,50,75} × lengths {2048,4096} × 3 needles × 3 seeds × 3 allocs = 57 样本。
配置生效由 engine `vllm_config.cache_config` + KV group 结构校验（已通过 MVEx）。

| 分配 | 平均准确率（待补） |
|---|---:|
| fp16 | 0.9074 |
| uniform int4 | 0.9074 |
| packed per-layer | **0.9259** |

配对差（18 cells，95% t-CI）：

| 对比 | Δ accuracy | CI |
|---|---:|---:|
| uniform int4 vs fp16 | 0.0000 | [−0.0985, +0.0985] |
| packed per-layer vs fp16 | +0.0185 | [−0.0505, +0.0875] |
| packed per-layer vs uniform int4 | +0.0185 | [−0.0505, +0.0875] |

结论：**packed per-layer 在 PPL 与 NIAH 上均不劣于 uniform int4（点估计更优）**，
容量恢复（0.833× uniform）没有质量回退。证据状态：ANALYZED（尚未做独立复现）。

## 3. 论文使用规则

- PPL 与 NIAH 数字已进入 Eval §6 / mainline §4（3-seed 配对 CI，标注 ANALYZED）；
- 仍缺：LongBench 子集（追加项）、A2 serving 独立复现（与质量闭环分开）。
