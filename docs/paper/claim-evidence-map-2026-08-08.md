# Claim–Evidence Map（声明-证据映射）

> 用途：投稿前逐条核对"论文写了什么 ↔ 仓库有什么证据"。每条声明必须能点击到
> 原子结果；标注 ANALYZED 的不得作 headline，标注 VERIFIED 的可作。

| # | 声明（Claim） | 证据（Evidence） | 状态 | 边界 |
|---|---|---|---|---|
| C1 | int4 端到端容量 2.245×@4K、3.155×@16K（2B） | `results/verified/2026-08-04/a2/` + 服务器启动日志 | VERIFIED | 单 GPU 0.85 util |
| C2 | int4 端到端容量 2.19×@4K（9B） | 服务器日志（NIAH 9B 会话） | VERIFIED | 单 run 口径 |
| C3 | attention KV 机制压缩 3.88× | 容量模型 + probe | VERIFIED | 与 C1 分开报告 |
| C4 | GDN state 18.63 MiB/请求、≈60% KV 预算 | `MambaStateShapeCalculator` 代码推导 + 容量核对 | ANALYZED | 60% 为标注估计 |
| C5 | SLO 边界：Random 0%/+4.8%/+14.3%，ShareGPT −17.6% | `results/verified/2026-08-04/e3/validation_report.md` | VERIFIED | n=3、5 req/s 网格、CAUTION |
| C6 | packed per-layer 恢复 0.258×→0.833×（3.232× vs legacy） | `a2_capacity_gate_c7379f0_v2.json` + westd-03 复现 | VERIFIED | 2B @4K |
| C7 | packed ShareGPT 边界 ≥ uniform（40 vs 35 @250ms） | R5 gates（pilot） | ANALYZED | 未独立复现 |
| C8 | 4-bit 近无损：Wikitext-2 PPL +1.73%（uniform）、+1.01%（packed） | `results/quality/r4-ppl/*.seeds.csv` | ANALYZED | 3-seed CI；无独立复现 |
| C9 | 等字节：4-bit+驱逐优于 <4-bit 全保留（11.85 vs 19.00） | `byte_budget_3seed.csv` | ANALYZED | 3-seed 配对 CI |
| C10 | NIAH 2B/9B 量化不劣（CI 含 0） | `niah-fixed-analysis.json` / `niah-fixed-9b-analysis.json` | ANALYZED | 90/90、54/54、哈希全匹配 |
| C11 | RULER v2：CWE 4K TQ 掉分真实 | `ruler-subset-analysis-v2-256.json` | ANALYZED | 20 samples、单 seed |
| C12 | Reasoning：MMLU 持平、GSM8K 点估计低 6.5–8.5pt | `reasoning-nothink-v2-analysis.json` | ANALYZED | 单 seed，禁止显著性表述 |
| C13 | LongBench：QA/摘要持平；9B 代码点估计偏低 | `longbench-analysis-20260807.json` | ANALYZED | 50 samples、单 seed |
| C14 | C4/PG19 PPL（第二/三语料，2B+9B） | `results/quality/ppl-extra/` + `ppl-extra-analysis-20260807.json` | DONE / ANALYZED | 3-seed CI；无独立复现 |
| C15 | Serving protocol-v3 六列正式边界 | — | PENDING | Formal 未跑 |
| C16 | A2 在 9B 恢复容量（3.230×/0.832×）、纯 attention 无稀释（3.765× vs 混合 2.245×） | `results/verified/2026-08-08/capacity-probe-extra/` | DONE / ANALYZED | 单主机探针；与 2B VERIFIED 比例一致 |
| C17 | KIVI/KVQuant 同协议对照 | — | PENDING | B1 |

## 写作红线

1. ANALYZED 数字只进正文表格时标注 "point estimates, single-seed" 或附
   "independent reproduction pending"；只有 VERIFIED 可作 headline。
2. 禁止使用被撤回的 "+25% SLO"；Random/ShareGPT 分开报告。
3. A2 的措辞必须是 "extend/adapt vLLM's existing packed layout"，不是 "propose a
   novel packed layout"。
4. 单 seed 的负向点估计（GSM8K、TREC、9B 代码）写 "point-estimate decline,
   multi-seed confirmation pending"，不写显著退化。
5. AIME25 只报告 "budget-limited, near-floor, no cross-allocation signal"。
