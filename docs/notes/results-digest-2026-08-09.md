# 2026-08-09 P0 补跑结果消化（ARS 审稿修订执行中）

> 状态：P0-1/P0-2/P0-3/P0-5/P0-6 已完成并提交；P0-4（S-formal）门禁进行中。
> 所有数字来自 `results/verified/2026-08-09/` 与 `results/quality/` 的原子 JSON + sha256。

## 1. M-2×2 容量（P0-2 / R2，已完成）

fp16-KV 探针 8/8（2B×{auto,bf16,fp16}@4K、2B×{auto,bf16}@16K、9B×{auto,bf16,fp16}@4K），
全部 sha 匹配、resolved dtype 正确、fp16 state 容量==bf16。

| KV | 模型 | L | fp32 tokens | bf16 tokens | r_state 实测 | 预测 | 误差 |
|---|---|---:|---:|---:|---:|---:|---:|
| fp16 | 2B | 4096 | 1,199,383 | 1,384,448 | 1.1543 | 1.1222 | +2.86% |
| fp16 | 2B | 16384 | 1,552,143 | 1,661,337 | 1.0704 | 1.0339 | +3.53% |
| fp16 | 9B | 4096 | 144,104 | 161,899 | 1.1235 | 1.1562 | −2.83% |
| int4 | 2B | 4096 | 2,692,710 | 3,703,954 | 1.3755 | 1.4089 | −2.37% |
| int4 | 2B | 16384 | 4,895,837 | 5,458,458 | 1.1149 | 1.1522 | −3.24% |
| int4 | 9B | 4096 | 315,392 | 443,538 | 1.4063 | 1.4089 | −0.18% |
| int4 | 9B | 16384 | 573,440 | 653,635 | 1.1398 | 1.1522 | −1.07% |

关键结论：

- r_kv 在 bf16 state 下上升（2B@4K：2.245→2.675；9B@4K：2.189→2.740），
  “KV 量化 × state 量化”复合收益方向成立；
- **误差符号按 KV 列分开**：int4 列 4/4 全负（−0.18~−3.24%，同号概率 0.0625），
  “保守下界”表述只适用于 int4 headline；fp16 列符号混合（+2.86/+3.53/−2.83），
  与 block 粒度差异一致（fp16 KV block 544/288 vs int4 2064/1072）。

## 2. Q-stacking PPL（P0-3 / R3，已完成）

2B × C4/PG19 × {fp32,bf16} state × uniform int4 KV，3-seed 配对：

| corpus | int4+fp32 | int4+bf16 | Δ [95% CI] |
|---|---:|---:|---:|
| C4 | 17.8730 | 17.8701 | −0.0029 [−0.0129, +0.0072] |
| PG19 | 27.6210 | 27.6276 | +0.0065 [−0.0447, +0.0578] |

CI 均含 0；与 fp16-KV 下的 state 代价（−0.0003/+0.0004）同量级 →
叠加不引入可测 state 精度代价。fp32 格与 08-07 ppl-extra 数字完全一致（复现）。

## 3. fp16 state 质量 smoke（P0-3/R13，已完成）

2B C4，1 seed × 1 seq：fp32 19.3480 / bf16 19.3498 / fp16 19.3479，
最大差 0.0018 PPL，可忽略；fp16 作为谱系新增精度点成立。

## 4. GSM8K seed 语义修复（P0-1 / R1，已完成 2B 12/12 + 9B 6/6）

新协议：GSM8K test 按 seed 无放回抽 200 题，greedy 解码；同 seed 跨分配共享题目集。
旧 head-200 协议保留为 legacy，不合并。

| 模型 | 分配 | per-seed acc | mean±SD | Δ vs fp16 [95% CI] |
|---|---|---|---:|---:|
| 2B | fp16 | 0.705/0.695/0.705 | 0.7017±0.0058 | — |
| 2B | fp16_statebf16 | 0.695/0.665/0.705 | 0.6883±0.0208 | −1.33pt [−5.13,+2.46] |
| 2B | uniform_int4 | 0.645/0.700/0.695 | 0.6800±0.0304 | −2.17pt [−10.62,+6.29] |
| 2B | int4+bf16 | 0.645/0.700/0.705 | 0.6833±0.0333 | −1.83pt [−10.82,+7.15] |
| 9B | fp16 | 0.850/0.860/0.885 | 0.8650±0.0180 | — |
| 9B | fp16_statebf16 | 0.845/0.860/0.890 | 0.8650±0.0229 | 0.00pt [−1.24,+1.24] |

关键结论（诚实修正）：

- 9B 零宽 CI 根因确认：旧协议 seed 无真实随机性；新协议下 9B state 效应 = 0；
- 2B 点估计仍为负（state −1.3pt、int4 −2.2pt），但 3-seed CI 全含 0 →
  旧“CI 不含 0”的显著性结论是固定题集伪重复产物；如需显著性需增加 seed 数（S3 MDE）；
- int4+bf16 相对 int4 的叠加边际 +0.33pt [−1.1,+1.77]，无可测叠加代价；
- `uniform_int4_statebf16` config effect 已逐 cell 验证（int4 KV + bf16 state）。

## 5. 敏感度多重比较校正（P0-6 / R6，已完成）

`state-sensitivity-analysis-20260809-bonf.json`：2/36 原始 CI 不含 0（C4 L2
p=0.0493、L8 p=0.0036，均正），Bonferroni（α/36=0.001389）与 BH-FDR 后均不显著；
量级 0.0004–0.0007 PPL 远低于 seed 间波动。结论保持“噪声级 + 符号一致性观察”。

## 6. S-formal serving（P0-4 / R4，进行中）

- MVEx：Random60 2/2 + ShareGPT300 2/2，全部 completed_validated、到达窗口偏差
  ≤0.01%、int4_statebf16 日志含 `int4_per_token_head` + `Using the user-specified
  value` + `CUDAGraphMode.PIECEWISE`；
- Pilot：Random60 6/6 完成；ShareGPT300 6/6 进行中；
- Formal：72 样本待 pilot 门禁通过后挂机。

## 证据路径

- `results/verified/2026-08-09/capacity-state-fp16kv/` + `capacity-2x2-analysis.json`
- `results/quality/ppl-stacking/` + `ppl-stacking-analysis-20260809.json`
- `results/quality/ppl-state-smoke-fp16/`
- `results/quality/reasoning/reasoning-{gsm8k,9b-state3seed}-v2-20260809/` + 分析 JSON
- `results/quality/state-sensitivity-analysis-20260809-bonf.json`
- 远端 `/root/autodl-tmp/statebf16-serving-20260809/`（S-formal）
