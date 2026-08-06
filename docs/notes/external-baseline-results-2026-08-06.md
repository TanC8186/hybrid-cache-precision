# R5 外部 Baseline 结果（2026-08-06）

> 状态：TurboQuant 可行性 MVEx **PASSED**；NIAH 质量矩阵 **36/36 完成，0 失败**（ANALYZED）。
> Serving SLO 矩阵（protocol-v3）待执行。证据：`results/quality/r5-turboquant/`、
> `results/quality/r5-analysis.json`。

## 1. 可行性

本 fork 的 dtype 校验器原生支持 `turboquant_k8v4` / `turboquant_4bit_nc` /
`turboquant_3bit_nc`。`turboquant_k8v4` 与 `turboquant_4bit_nc` 均在 Qwen3.5-2B 上完成
引擎启动 + 贪婪生成（NIAH acc=1.00，seed7/d50/L2048），config effect 校验通过。

## 2. NIAH 质量矩阵（与 R4 同一 18-cell 网格：3 depths × 2 lengths × 3 seeds × 3 needles）

| 分配 | 平均准确率 | 相对 fp16 配对 Δ（95% t-CI，df=17） |
|---|---:|---:|
| fp16（R4 基线） | 0.9074 | — |
| uniform int4（R4） | 0.9074 | 0.0000 [−0.0985, +0.0985] |
| packed per-layer（R4） | 0.9259 | +0.0185 [−0.0505, +0.0875] |
| **turboquant_k8v4** | 0.8519 | −0.0556 [−0.1408, +0.0297] |
| **turboquant_4bit_nc** | 0.8889 | −0.0185 [−0.0875, +0.0505] |

结论：TurboQuant 两个 dtype 在 NIAH 上与 fp16 无显著差异（CI 均含 0），点估计略低；
packed per-layer 点估计最高。外部 baseline 质量证据成立（ANALYZED）。

## 3. 论文使用规则

- TurboQuant 行必须标注 dtype（`k8v4` / `4bit_nc`）、协议与状态（ANALYZED）；
- Serving SLO 矩阵完成前，不得声称“TurboQuant 同协议 serving 对照”；
- KIVI/KVQuant 仍无同栈 serving 实现，只能作为 transformers 路径质量附录（待做或明确不适用）。

## 4. 待办

- [ ] TurboQuant × Random/ShareGPT × 3 seeds serving SLO 矩阵（protocol-v3，数小时）；
- [ ] TurboQuant 容量探针（server 日志 capacity tokens）与吞吐对比；
- [ ] 若 serving 矩阵通过，升级为 VERIFIED 级外部 baseline（需独立复现门禁）。
