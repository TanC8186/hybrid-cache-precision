# R5 — 外部 Baseline 实验方案（2026-08-06）

> 目的：回应 ARS 审稿 R5 —— 在相同硬件/模型/SLO 协议下补齐可执行外部对照，
> 明确论文容量/SLO 数字相对“fp16 serving”与相对“SOTA KV 量化系统”的位置。

## 1. Baseline 候选（按可执行性排序）

| 候选 | 形态 | 可执行性 | 备注 |
|---|---|---|---|
| TurboQuant（k8v4 / 4bit_nc / 3bit_nc） | vLLM 原生 KV dtype（本 fork 校验器已列出 `turboquant_k8v4`、`turboquant_4bit_nc`、`turboquant_3bit_nc`） | **待可行性 MVEx**：引擎启动+生成通过即进入 serving 矩阵；失败则降级 int8 同栈对照 | ICLR 2026；最接近“同栈系统对照” |
| int8/fp8 + 驱逐（byte-equivalent） | vLLM 原生 dtype + 自定义驱逐 | 高（自研已有 transformers 路径） | 作为“同栈字节等价”对照，论文可明确标注非外部系统 |
| KIVI / KVQuant | 研究型 kernel，无官方 vLLM serving 集成 | 低：仅 transformers 路径 PPL/质量对照，需明确标注协议差异 | 用于质量表对照，不进入 serving SLO 表 |
| MiniKV / HqeKV / ARKV | 论文代码库，集成成本高 | 低 | 若时间允许再评估；不作为第一轮 blocking |

## 2. 契约

- 硬件/模型：RTX 5090，Qwen3.5-2B；冻结根代码与 vLLM commit。
- Serving 协议：与 protocol-v2/protocol-v3 完全一致（60s 窗口、warmup 120、3 seeds、
  goodput/offered ≥ 0.95、TTFT {250..3000}ms、TPOT 200ms、Random + ShareGPT300）。
- 质量协议：与 `byte_budget_3seed` 同协议（Wikitext-2 3-seed）+ NIAH/LongBench 子集。
- 成功判据：至少一个外部系统（TurboQuant 优先）在同一 serving 协议下完成 3-seed 矩阵；
  其余候选给出明确的可执行性报告与“为何未进入 serving 表”的说明。

## 3. 分阶段放行

1. **可行性 MVEx（2026-08-06 已确认 dtype 字符串存在）**：确认
   `kv_cache_dtype=turboquant_k8v4 / turboquant_4bit_nc` 对 Qwen3.5-2B 引擎启动+贪婪生成可用；
   记录日志中的生效证明；失败则降级为 `int8_per_token_head`/fp8 同栈对照并报告。
2. **Pilot**：TurboQuant × Random/ShareGPT × rates {30,40} × seed 7，12 样本；
   请求守恒、到达窗口、schema 全部通过才放行。
3. **Formal**：3 alloc（fp16/int4/TurboQuant）或（fp16/int4/byte-equivalent）×
   Random + ShareGPT × 3 seeds；切片执行 + `--resume`。
4. **Reproducibility**：新 attempt 复跑边界；10% 容差 + 边界精确。

## 4. 论文使用规则

- 所有外部 baseline 行必须带“同协议/同硬件/同模型”标注；协议不同的仅进质量附录。
- 若 TurboQuant 不可执行，论文明确写“external baselines not yet available”，
  不得用 transformers 路径数字充当 serving 对照。
- 字节等价对照（int4 vs int8+eviction 等）需报告实际字节差容差（≤5%）。

## 5. 归档

- `experiments/` 原始产物 + `results/` 聚合与 provenance；baseline 的 SHA/commit 固定；
- 论文表：容量/SLO 边界对比表 + 质量对比表 + 可执行性附录。
