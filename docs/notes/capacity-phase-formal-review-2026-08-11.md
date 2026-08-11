# Capacity Phase Formal 复核记录

## Material Passport

- Origin Skill: experiment-skill
- Origin Mode: validate
- Origin Date: 2026-08-11
- Verification Status: ANALYZED
- Version Label: capacity_phase_formal_review_v1

## 复核对象

- Attempt: `capacity-phase-formal-20260811`
- Phase: `formal`
- Host: RTX 5090 (32607 MiB), driver `580.105.08`
- Frozen contract: `results/verified/2026-08-11/capacity-phase/capacity-phase-formal-20260811.contract.json`
- Contract SHA-256: `e21566781209029ec641240d04cb9c8e1bb4c1cbb91569c5e68466ac19179a4e`
- Analysis SHA-256: `1b5bc54d599529e1261dceda5238fd6bdafa5cec8263106df085dbc11c37a789`
- Runner log: `results/verified/2026-08-11/capacity-phase/capacity-phase-formal-20260811.run.log`
- Runner log SHA-256: `758cceddfeb4f38ca90fe1c245a0486e99caf16980031aad31cd15f1cf2215ec` (1,796,571 bytes; raw logs remain Git-ignored)

Frozen command:

```text
timeout --signal=TERM --kill-after=30s 21600 bash scripts/bench/run_capacity_phase_diagram.sh formal capacity-phase-formal-20260811
```

The local code hashes for the runner, probe, runtime inspector, and analyzer match
the hashes recorded in the frozen contract. The remote runtime snapshot has no
top-level Git repository; the contract's file hashes are therefore the authoritative
provenance for the run.

## 完整性与门禁

| 检查 | 结果 | 证据 |
|---|---|---|
| 预注册矩阵 | PASS | 2B core 72 + 9B core 32 + float16 controls 8 = 112 |
| 原始 JSON | PASS | 112/112 文件存在，文件名集合与 contract 完全一致 |
| SHA sidecar | PASS | 112/112 sidecar 与本地内容匹配 |
| Schema/配置 | PASS | analyzer 严格校验通过；seed=42、空 per-layer override、KV/state 参数均匹配 |
| resolved state dtype | PASS | auto→float32: 52；bfloat16→bfloat16: 52；float16→float16: 8 |
| 数值域 | PASS | 每格 tokens、max concurrency、GPU blocks、elapsed 均有限且为正 |
| generation workload | PASS | 112/112 为 `null`，与 capacity-only protocol 一致 |
| core 配对 | PASS | 52/52 对满足 `bf16_state_tokens > fp32_state_tokens` |
| 运行日志 | PASS | 112 个 `[OK]`、1 个 `[DONE]`；无 `[FAIL]`、`[SKIP]`、`[RETRY]`、OOM、Traceback 或 Killed |
| analyzer 独立重算 | PASS | 本地重跑 analyzer 的 SHA 与远端分析文件完全相同 |

`tokens` 与 `max_concurrency * max_model_len` 的最大绝对差为 0.9474 token，
符合整数 token 截断/取整，不构成数据错误。

Gate 0--3（合同、MVEx、pilot、formal 完整性）已满足本轮检查条件。Gate 4
仍未完成：尚未以新 attempt ID 做适用的 formal reproducibility re-run，因此本
结果不能升级为 `VERIFIED`。

## 统计性描述

Formal capacity probe 是冻结 build/config 下的确定性 allocator 测量，每个 cell
只有一次观测。因此下面是描述性配对结果，不计算伪重复的 p 值、置信区间或
等效性结论。

| 分组 | n | bf16 相对 fp32 最小增益 | 中位增益 | 最大增益 |
|---|---:|---:|---:|---:|
| 2B + fp16 KV | 18 | 3.3238% | 13.8032% | 34.9420% |
| 2B + int4 KV | 18 | 7.6031% | 30.0425% | 93.2759% |
| 9B + fp16 KV | 8 | 3.3661% | 9.3033% | 23.5911% |
| 9B + int4 KV | 8 | 6.9914% | 26.8518% | 58.0486% |
| 全部 core pairs | 52 | 3.3238% | 15.4400% | 93.2759% |

容量模型残差（measured/predicted - 1）为：

- median absolute residual: `1.8114%`
- mean absolute residual: `2.6612%`
- range: `-3.6614%` to `+13.2134%`
- 最大正残差：2B/int4/L=1024/u=0.80 (`+13.2134%`)
- 最大负残差：2B/int4/L=16384/u=0.70 (`-3.6614%`)

这些残差是 idealized architecture-derived model 的误差描述，反映 page/block
离散分配及实现开销；它们不是 lower bound、置信界或跨硬件预测保证。尤其不能
把 formal 的 `+13.2134%` 误写成模型“保守下界”。

Float16 controls 的用途是确认第三种 state dtype 的解析与容量前沿记录；其
utilization/context 与 core pairs 不完全匹配，不应被当作额外的质量或公平基线。

## 11 类统计谬误自查

覆盖：`11/11`。

| 类型 | 状态 | 说明 |
|---|---|---|
| Simpson's paradox | NOTE | 结果按 model/KV/length/util 分层；52 对方向一致，没有观察到聚合方向反转。 |
| Ecological fallacy | NOTE | 分析单位是配置 cell，不从群体平均推断个体行为。 |
| Berkson's paradox | NOTE | 使用预注册完整矩阵，没有按结果筛选样本。 |
| Collider bias | NOTE | 没有把由处理和结果共同决定的变量作为控制量。 |
| Base-rate neglect | NOTE | 不涉及诊断概率、PPV/NPV 或敏感度/特异度。 |
| Regression to the mean | NOTE | 不是按极端值筛选的 pre/post 随机测量。 |
| Survivorship bias | NOTE | 没有 silent exclusion；112 个请求 cell 全部完成。 |
| Look-elsewhere effect | CAUTION | 52 个配对与残差属于预注册矩阵，但没有把多格结果包装成 p 值显著性；任何 headline 必须标明探索范围。 |
| Garden of forking paths | NOTE | contract、矩阵和 analyzer 在运行前冻结，分析脚本不以结果选择模型。 |
| Correlation != causation | CAUTION | capacity 对照支持冻结环境内的配置差异描述，不支持 serving goodput 机制或普遍因果归因。 |
| Reverse causality | NOTE | 没有时间序列方向性推断；该项对 allocator cell 不适用。 |

## 允许的结论边界

可以使用的表述：

> 在冻结的 vLLM build、RTX 5090、Qwen3.5-2B/9B 和预注册 context/utilization
> 矩阵上，state=bfloat16（resolved 为 bf16）在 52 个 KV/state 配对中均测得
> 高于 state=auto（resolved 为 fp32）的 allocator token capacity；收益随 KV
> dtype、上下文长度和 page rounding 改变。

必须避免的表述：

- “capacity model is a conservative lower bound”；
- “已证明跨硬件、跨模型或跨 TP 普遍成立”；
- “formal capacity 直接证明 serving goodput 或质量改善”；
- “formal 已独立复现/已 VERIFIED”。

## 后续 Gate 4

需要另建一个新的 reproducibility attempt（不得复用本 attempt ID），冻结同一
contract 与环境并记录差异；比较结构、完整矩阵、解析状态和主要容量比值。复现门
通过后才可把证据状态从 `ANALYZED` 升级为 `VERIFIED`。在此之前，本报告和分析
JSON 只作为可追溯的实验进度，不作为论文 quantitative headline 的最终验证凭证。
