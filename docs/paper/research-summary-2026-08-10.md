# 研究汇总（2026-08-10 重审版）：混合线性注意力模型服务内存的精度预算分配

> 本文档是 2026-08-09 版研究汇总的重审快照：并入 08-09/08-10 全部补跑与复现结果，
> 按证据边界收窄 claim。所有数字可溯源到 `results/verified/2026-08-08|09` 与
> `results/quality/` 下的原子 JSON/CSV + sha256；未入库内容一律不写。

## 1. 研究问题与定位

Qwen3.5（Gated DeltaNet + GQA 混合架构）服务内存包含两类开销：

- 注意力 KV cache：随上下文 L 线性增长，可经 KV 量化（fp16→int4）压缩；
- GDN 循环 state：每序列固定大小（2B 18.63 MiB、9B 24.84 MiB，代码推导），
  此前被视为 KV 方案的固定稀释剂。

主张：把 **state 精度** 作为与 **KV 位数** 联合分配的第二个预算维度，给出容量模型
`C(L;A,G)=M/(A·L+G)` 并做容量 × 质量 × serving 闭环。定位红线：不主张“首次 state
压缩/量化”（ReplaySSM、vLLM PR#43518 存在），只主张“现有 serving 系统未把 state
精度纳入与 KV 量化联合的预算分配，并以系统化证据闭环验证”。

## 2. 方法

- 容量：`probe_ssm_state_dtype.py`，`gpu_memory_utilization=0.85`，`max_model_len`
  4K/16K；逐探针校验 resolved `mamba_ssm_cache_dtype`、容量 token、sha。
- 质量（harness）：`hybrid_premise.py --state-dtype`（写回边界 cast，chunk=128），
  3-seed 配对 CI；**已披露 harness 不能等同 kernel 逐 token 语义**（chunk 消融见 §5.6）。
- 质量（kernel）：vLLM 离线 greedy（RULER/GSM8K），`config_effect` 逐 cell 校验。
- serving：protocol-v3（PIECEWISE、warmup 120、60s/300s 窗、TTFT 250–3000ms、
  TPOT 200ms、goodput/offered ≥0.95、失败计入分母、3 seeds）。
- 统计：3–9 seeds 配对 t-CI；GSM8K 预注册 9-seed 功效（MDE 公式）与决策规则；
  敏感度门 Bonferroni（α/36）与 BH-FDR。

## 3. 实验结果

### 3.1 容量模型（2×2，已完成 8/8 fp16-KV + 10/10 int4-KV 探针）

| KV | 模型 | L | fp32 state tokens | bf16 state tokens | r_state 实测 | 预测 | 误差 |
|---|---|---:|---:|---:|---:|---:|---:|
| int4 | 2B | 4096 | 2,692,710 | 3,703,954 | 1.3755 | 1.4089 | −2.37% |
| int4 | 2B | 16384 | 4,895,837 | 5,458,458 | 1.1149 | 1.1522 | −3.24% |
| int4 | 9B | 4096 | 315,392 | 443,538 | 1.4063 | 1.4089 | −0.18% |
| int4 | 9B | 16384 | 573,440 | 653,635 | 1.1398 | 1.1522 | −1.07% |
| fp16 | 2B | 4096 | 1,199,383 | 1,384,448 | 1.1543 | 1.1222 | +2.86% |
| fp16 | 2B | 16384 | 1,552,143 | 1,661,337 | 1.0704 | 1.0339 | +3.53% |
| fp16 | 9B | 4096 | 144,104 | 161,899 | 1.1235 | 1.1562 | −2.83% |

- fp16 state 容量 == bf16 state（同字节数，2B/9B @4K 均确认）。
- r_kv（int4/fp16）：fp32 state 下 2.2451（2B@4K）/3.1542（16K）/2.1886（9B@4K）；
  bf16 state 下 2.6754/3.2856/2.7396 → KV × state 复合收益成立。
- block 粒度证据：int4 fp32 2064/3287 vs bf16 1072/6330（2B）；fp16 544/3221 vs
  288/6084（2B）。int4 列 4/4 误差全负（P=0.0625）→ 模型定位为**保守下界**；
  fp16 列符号混合，按格报告。

### 3.2 Q-stacking PPL（2B，int4 KV × {fp32,bf16} state，3-seed）

| corpus | int4+fp32 | int4+bf16 | Δ [95% CI] |
|---|---:|---:|---:|
| C4 | 17.8730 | 17.8701 | −0.0029 [−0.0129, +0.0072] |
| PG19 | 27.6210 | 27.6276 | +0.0065 [−0.0447, +0.0578] |

两格 CI 均含 0；与 fp16-KV 下 state 代价同量级 → **叠加不引入可测 state 精度代价**。

### 3.3 GSM8K（vLLM kernel 路径，seed 化 200 题子采样）

| 模型 | 对比 | Δ [95% CI] | p | MDE(80%) | power | 判定 |
|---|---|---:|---:|---:|---:|---|
| 2B | statebf16 vs fp16 | −1.00pt [−1.71,−0.29] | 0.0249 | 1.16pt | 67.5% | 显著 |
| 2B | uniform int4 vs fp16 | −2.72pt [−4.20,−1.24] | 0.0069 | 2.41pt | 88.3% | 显著 |
| 2B | int4+bf16 vs fp16 | −1.56pt [−3.53,+0.42] | 0.1615 | 3.22pt | 27.5% | 不显著 |
| 2B | stacking（int4+bf16 vs int4） | +1.17pt [−0.33,+2.66] | 0.165 | 2.44pt | 27.1% | 不显著 |
| 9B | statebf16 vs fp16 | +0.33pt [−0.07,+0.73] | 0.141 | 0.65pt | 30.2% | 不显著 |

旧 head-200 协议数字（2B −2.67pt、9B −0.5pt）确认是伪重复产物，已退役。

### 3.4 RULER（3 个 dataset seed，engine seed 固定 7）

| 格 | fp16 mean | bf16 mean | Δ [95% CI] |
|---|---:|---:|---:|
| 2B fwe L4096 | 29.44 | 25.55 | −3.89 [−32.97,+25.19] |
| 2B fwe L8192 | 35.00 | 36.67 | +1.66 [−5.51,+8.83] |
| 9B niah_multiquery L4096 | 87.92 | 88.75 | +0.83 [−3.91,+5.58] |
| 9B niah_multiquery L8192 | 72.92 | 68.75 | −4.17 [−8.91,+0.58] |
| 9B fwe L8192 | 53.89 | 54.44 | +0.55 [−11.39,+12.50] |

全部 CI 含 0；原单 seed 非零差异为抽奖/think 截断噪声。RULER 只按点估计 + 宽 CI。

### 3.5 逐层敏感度（决策门）

2B，C4+PG19，3-seed，20 配置，cast 审计 + 参考逐位复现通过；2/36 原始 CI 不含 0
（C4 L2 p=0.049、L8 p=0.0036），Bonferroni/BH-FDR 后均不显著 → 无逐层收益空间。

### 3.6 harness chunk 消融（必须披露的边界）

2B C4，1 seed×1 seq：fp32 chunk128=19.3480 vs chunk1=36.1643（+87%）；bf16
19.3498 vs 36.1347。**harness 的 chunk 级写回舍入不等于 kernel 逐 token 语义**，
PPL 只能作辅助证据；kernel 路径质量以 vLLM GSM8K/RULER 为准。

### 3.7 serving（int4 KV × {fp32,bf16} state，protocol-v3）

3-seed 可持续边界（最大 rate，req/s）：

| workload | allocation | 250ms | 500ms | 1000ms | 2000/3000ms |
|---|---|---:|---:|---:|---:|
| Random60 | int4 | 30 | 35 | 35 | 40 |
| Random60 | int4_statebf16 | 30 | 35 | **40** | 40 |
| ShareGPT300 | int4 | 40 | 40 | 40 | 40 |
| ShareGPT300 | int4_statebf16 | **35** | 40 | 40 | 40 |

配对 goodput Δ（bf16−fp32，CI 不含 0 的要点）：Random60 r40 250ms +0.334
[+0.078,+0.589]、500ms +0.215 [+0.154,+0.276]；r45 2000/3000ms +0.324/+0.367；
r50 2000/3000ms +0.041/+0.072。ShareGPT300 r40 250ms **−0.002 [−0.0026,−0.0015]**。

**独立复现（72/72 审计通过）判定 PARTIALLY_REPRODUCIBLE**：
- Random60 过载区 paired goodput 增益复现（量级一致）；
- Random60 int4@1000ms 边界 35→40 不复现（复现中 int4=40）；
- ShareGPT 250ms 边界方向翻转；**ShareGPT r45 int4 原始 mean goodput 0.67–0.73
  vs 复现 0.14–0.23**（TTFT p99 262/822/16191ms vs ≈18.5s + 160–264 请求失败），
  属过载边界环境/调度波动，非审计伪影；
- claim #5 维持 `ANALYZED`，只按 workload × threshold 限定表述。

## 4. Claim whitelist（收窄版）

1. 混合模型服务内存是 KV 位数 × state 位数的二维精度预算；现有 serving 系统未把
   state 精度纳入联合分配（不主张首次 state 压缩）。
2. 容量模型 `r_state(L)` 在 int4 KV 下为保守下界（误差 −3.24%~−0.18%），
   fp16 KV 列按 signed error 逐格报告；A_q/G 为架构推导参数（非实测拟合）。
3. bf16 state：int4 KV 下 4K 容量 +38~41%、16K +11~14%；PPL 无可测差异；
   GSM8K 2B 回退显著（−1.0pt，CI 不含 0）、9B 无回退；RULER 无可检测差异。
4. 逐层 state 精度无质量收益（统计校正后负面结论）。
5. serving：Random60 过载区 paired goodput 增益为方向性证据（ANALYZED；
   独立复现中重现），ShareGPT 500ms 及以上边界持平；不得作普适 SLO claim。
6. packed per-layer page groups 解决 KV 维度混合精度部署（已有，VERIFIED 边界）。

## 5. 已知限制

- 模型/硬件：Qwen3.5-2B/9B、RTX 5090 单卡；跨架构（Mamba2 等）未测，一般性声明
  收窄为 GDN-based 混合架构。
- harness PPL 为 chunk 级写回舍入近似（§3.6）；关键质量结论以 vLLM kernel 路径为准。
- serving 收益 workload/threshold 依赖，过载边界存在运行间波动（§3.7）。
- fp16 state 质量仅 smoke（1 seed），fp8/int8 state 未测（future work）。

## 6. 证据索引

- `results/verified/2026-08-08/capacity-state/` + `capacity-state-analysis.json`
- `results/verified/2026-08-09/capacity-state-fp16kv/` + `capacity-2x2-analysis.json`
- `results/quality/ppl-stacking/` + `ppl-stacking-analysis-20260809.json`
- `results/quality/ppl-state-smoke-fp16/`
- `results/quality/reasoning/reasoning-gsm8k-{state,9b-state}9seed-v2-20260809/` + 分析 JSON
- `results/quality/ruler-subset/ruler-subset-20260809-multiseed-{2b,9b}/` + 分析 JSON
- `results/quality/state-sensitivity-analysis-20260809-bonf.json`
- `results/quality/chunk-ablation/`
- `results/verified/2026-08-09/statebf16-serving-formal-analysis.json`（原始）
- `results/verified/2026-08-09/statebf16-serving-repro-analysis.json`（复现）
- `results/verified/2026-08-09/statebf16-formal-20260809.tar.gz`（原始 1.0GB 归档）
- 审计/口径：`docs/notes/repro-final-2026-08-09.md`、`docs/notes/claim-revision-draft-2026-08-09.md`、
  `docs/notes/results-digest-2026-08-09.md`
