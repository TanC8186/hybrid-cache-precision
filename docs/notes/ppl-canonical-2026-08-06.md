# PPL Canonical 决策（2026-08-06）— 解决三文件矛盾

> 背景：ARS 审稿（2026-08-06）CRITICAL C2 指出同一 4-bit 配置出现 13.86 / 11.67 / 11.03 三个 PPL，
> 无 CI/种子信息。本文档给出 canonical 判定与配对统计，并已同步到论文草稿（Eval §6）。

## 1. 三个文件的真实关系（证据：experiments/*.log）

| 文件 | PPL(4-bit full) | 协议 | 日志 | 状态 |
|---|---|---|---|---|
| `results/ablations/byte_budget_ordering.csv` | 13.86295 | 单 seed 确定性，5×2048，chunk=128（最终 harness） | `byte_budget_final.log` | **保留**：仅作为逐层敏感度/异构预算的确定性锚点（all-8bit 13.63、all-2bit 21.07 同协议） |
| `results/ablations/byte_budget_3seed.csv` | **11.6749 ± 1.6145** | **3 seed {7,42,2026} × 5×2048，chunk=128（最终 harness）** | `byte_budget_3seed.log` | **canonical**：论文 headline PPL 唯一来源 |
| `results/ablations/headline_3seed.csv` | 11.0252 ± 2.6897 | 3 seed × **3 条序列**（序列数更少、std 更大） | `seed3.log` | **退役**：不得用于论文 claim |

三者不是“同一配置跑出三个数”的矛盾，而是三份协议/抽样不同的产物：13.86 是单 seed 确定性协议，
11.67 是最终 harness 的 3-seed 均值，11.03 是 3 序列/seed 的较弱协议。

## 2. Canonical 配对统计（3-seed，逐 seed 配对 FP16）

数据来自 `byte_budget_3seed.log`（seed 42/2026/7）：

| 配置 | PPL mean ± SD | vs FP16 配对 Δ（95% t-CI, df=2） | 相对变化 |
|---|---:|---:|---:|
| 8-bit | 11.4828 ± 1.5695 | −0.0004 [−0.0110, +0.0102] | ≈0%（无损） |
| 4-bit | 11.6749 ± 1.6145 | +0.1917 [+0.0877, +0.2957] | **+1.7% [0.8%, 2.6%]** |
| 3-bit | 13.5317 ± 1.8726 | +2.0485 [+1.1491, +2.9479] | +17.8% [10.0%, 25.7%] |
| 2-bit | 19.0004 ± 3.1922 | +7.5171 [+3.4955, +11.5388] | +65.5% [30.4%, 100.5%] |
| FP16 | 11.4832 ± 1.5736 | baseline | — |

等字节配对（逐 seed）：

| 对比 | Δ（95% CI） |
|---|---:|
| 4-bit keep1024 vs 2-bit full（≈3.2 MB，字节差 5.38%） | −7.1530 [−10.9651, −3.3408] |
| 4-bit keep1536 vs 3-bit full（≈4.8 MB，字节差 1.25%） | −1.8214 [−2.6504, −0.9924] |

## 3. 论文使用规则

1. Headline PPL 一律用 `byte_budget_3seed.csv`（mean ± SD + 配对 95% CI）。
2. 逐层敏感度/异构预算表保留确定性单协议（`layer_sensitivity.csv`、`hetero_budget.csv`），
   必须标注 “deterministic single-protocol, 3-seed replication pending”。
3. `headline_3seed.csv`（3 序列协议）不再引用。
4. 等字节对比非 byte-exact（≤5.4%），论文中显式声明容差。

## 4. 待办

- [ ] 在 5090 上按最终 harness 重跑逐层敏感度/异构预算 3-seed（R4 质量闭环的一部分）。
- [ ] 补充 packed vs uniform 的 3-seed PPL + retrieval/long-context。
