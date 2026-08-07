# Experiment Matrix Plan — MLSys Submission

> 状态：2026-08-08。本文是投稿面实验的唯一权威矩阵：每个实验给出研究问题、
> 协议、状态、关键结果、证据路径与放行门禁；"待补"项只允许以显式 PENDING
> 状态存在，禁止用占位数字。所有数字必须可溯源到本文"数据溯源"节列出的
> 原子 JSON/CSV + sha256。

## 1. 覆盖矩阵（Coverage Matrix）

| 实验 | 分配 | 模型 | seeds | 样本 | 状态 |
|---|---|---|---|---|---|
| E1 容量探针（A2 gate） | legacy/uniform/packed | 2B | — | 3 probes | **DONE / VERIFIED** |
| E1b 端到端容量比 | fp16→int4 | 2B/9B | — | 4096/16384 | **DONE / VERIFIED** |
| E2 Serving gates（protocol-v3） | 6 allocs | 2B | 3 | MVEx 6 + Pilot 18 + 容量 6 | **DONE / ANALYZED** |
| E3 SLO（protocol-v2 formal） | fp16/int4 | 2B | 3 | 72 + 48 复现 | **DONE / VERIFIED** |
| E4 SLO（protocol-v3 formal） | 6 allocs | 2B | 3 | Random60 45 + ShareGPT300 63 | **PENDING（需人工放行）** |
| Q1 PPL Wikitext-2 | fp16/uniform/packed | 2B | 3 | 5×2048 | **DONE / ANALYZED** |
| Q2 PPL C4/PG19 | fp16/uniform/packed | 2B/9B | 3 | 5×2048 | **2/12 DONE（续跑中）** |
| Q3 NIAH 2B | 5 allocs | 2B | 3 | 90 cells | **DONE / ANALYZED** |
| Q3b NIAH 9B | fp16/uniform/packed | 9B | 3 | 54 cells | **DONE / ANALYZED** |
| Q4 RULER v2 | 5 allocs | 2B | 1 | 70 cells | **DONE / ANALYZED** |
| Q5 FWE no-think | 5 allocs | 2B | 1 | 10 cells | **DONE / ANALYZED** |
| Q6 Reasoning（no-think v2） | 5 allocs | 2B | 1 | 15 cells | **DONE / ANALYZED** |
| Q7 LongBench v1 | 5 allocs（2B）+ 3 allocs（9B） | 2B/9B | 1 | 64 cells | **DONE / ANALYZED** |
| M1 A2 机制门禁 | — | 2B | — | 8/8 检查 | **DONE / VERIFIED** |
| M2 A2 独立复现 | — | 2B | — | 4/4 probes | **DONE / VERIFIED** |
| M3 9B packed 容量探针 | legacy/uniform/packed | 9B | — | 3 probes | **PENDING** |
| M4 纯注意力对照 | fp16/int4 | Qwen2.5-7B | — | 容量 + 质量切片 | **PENDING** |
| B1 外部 baseline（KIVI/KVQuant） | — | 2B | 3 | 待定 | **PENDING（需评估）** |
| B2 GSM8K 多 seed | 5 allocs | 2B | 3 | 3×200 | **PENDING** |
| B3 LongBench 多 seed | 关键任务 | 2B/9B | 3 | 3×50×任务 | **PENDING（可选）** |
| B4 32K/64K 探针 | fp16/int4/packed | 2B | — | 容量+检索 | **PENDING（可选）** |

分配缩写：uniform = `uniform_int4`；packed = `packed_per_layer`；k8v4/4bit_nc =
`turboquant_k8v4` / `turboquant_4bit_nc`；fp8 = `fp8`（serving 列）。

## 2. 系统实验（E）

### E1 容量探针（A2 gate）— DONE / VERIFIED

- 研究问题：逐层混合精度在 vLLM V1 统一页 KV 管理下的容量塌缩有多大？packed
  per-layer 能否恢复？
- 协议：`gpu_memory_utilization=0.85`、`max_model_len=4096`、Qwen3.5-2B；
  三个独立 probe（legacy/uniform/packed），不合并历史数据。
- 结果（capacity tokens / max concurrency）：

| 配置 | tokens | concurrency | vs uniform |
|---|---:|---:|---:|
| legacy 逐层（L23 bf16） | 705,604 | 172.3 | ×0.258 |
| uniform int4 | 2,736,947 | 668.2 | 1.000 |
| packed per-layer | 2,280,448 | 556.8 | **0.833**（3.232× vs legacy） |

- 门禁：packed/legacy ≥ 3×（实测 3.232×）；packed/uniform ∈ [0.80, 0.92]
  （实测 0.833）；`a2_capacity_gate_c7379f0_v2.json` = PASSED。
- 证据：`results/verified/2026-08-04/a2/`；复现 `results/reproduction/2026-08-05/a2/`
  （westd-03，REPRODUCIBLE）。

### E1b 端到端容量比 — DONE / VERIFIED

- 结果：int4 vs fp16 端到端容量比 2.245×@4096、3.155×@16384（2B）、2.19×@4096
  （9B）；机制层 attention KV 3.88×；GDN 每请求 18.63 MiB（代码推导，≈60% KV 预算）。
- 论文口径：机制层与系统层两个标尺同时报告；GDN 占比为标注过的代码推导估计。

### E2 Serving gates（protocol-v3）— DONE / ANALYZED

- 门禁链：MVEx 6/6 → Pilot 18/18 → 容量探针 6/6 → `DONE_GATES`；fp8 首轮
  缺 ninja 失败已隔离修复。
- Pilot 边界（3 seeds、goodput/offered ≥ 0.95）：

| workload | TTFT | fp16 | int4 | packed |
|---|---:|---:|---:|---:|
| Random60 | 250 ms | 30 | NONE* | 30 |
| Random60 | 500 ms | 35 | 35 | 35 |
| Random60 | 1000+ ms | 35 | 40 | 40 |
| ShareGPT300 | 250 ms | 45 | 35 | 40 |
| ShareGPT300 | 500+ ms | 45 | 40 | 40 |

*NONE = 测试网格内无 3-seed 全可持续点。状态：ANALYZED，未独立复现，
不得作 headline。

### E3 SLO（protocol-v2 formal）— DONE / VERIFIED

- 72/72 formal + 48/48 独立复现；160,200 请求、0 失败；请求守恒审计通过。
- 边界：Random 250 ms 0%、500 ms +4.8%、1000–3000 ms +14.3%；ShareGPT 250–3000 ms
  **−17.6%**；TPOT 全矩阵不 binding。
- 口径：禁止 workload-general 表述；Random/ShareGPT 分开报告。

### E4 SLO（protocol-v3 formal）— PENDING

- 目标：六列（fp16/int4/packed/k8v4/4bit_nc/fp8）× Random60/ShareGPT300 × 3 seeds；
  PIECEWISE、warmup 120、TTFT {250..3000}、TPOT 200、goodput ≥ 0.95。
- 成本：约 8h；放行门禁：人工确认后 `bash scripts/exp/run_serving_formal.sh`（待创建）。
- 产出：`analyze_r5_serving.py` 边界表 + P99 TTFT/TPOT + 容量探针。

## 3. 质量实验（Q）

### Q1 PPL Wikitext-2 — DONE / ANALYZED

| 分配 | mean ± SD | Δ vs fp16 [95% t-CI] |
|---|---:|---:|
| fp16 | 11.4827 ± 1.5732 | — |
| uniform int4 | 11.6811 ± 1.5861 | +0.1984 [+0.1541, +0.2428]（+1.73%） |
| packed per-layer | 11.5985 ± 1.5534 | +0.1158 [+0.0637, +0.1679]（+1.01%） |
| packed vs uniform | — | −0.0826 [−0.1767, +0.0115] |

- 证据：`results/quality/r4-ppl/*.csv.seeds.csv`；corpus sha
  `f7c3d825...`；协议 `hybrid_premise.py --seeds 7,42,2026 --num-seqs 5 --max-len 2048`。

### Q2 PPL C4/PG19 — 2/12 DONE（续跑中）

- 协议与 Q1 相同；语料 `data/c4_slice.txt`（sha `7ee17255...`）、
  `data/pg19_slice.txt`（sha `c898ba29...`）。
- 已完成：c4 fp16 2B（17.58 ± 1.06）、pg19 fp16 2B（29.84/28.79/22.90）。
- 剩余 10 cells：2B uniform/packed ×2 + 9B fp16/uniform/packed ×2。
- 续跑命令：`bash scripts/exp/run_ppl_extra.sh ppl-extra-20260807`（已有 skip 逻辑）。
- 注意：`hybrid_premise.py` 对 bits=16 会输出 6 行/3 seeds（FP16 baseline 重复），
  分析器已按 (bits, seed) 去重（`analyze_ppl_extra.py`）。

### Q3 NIAH — DONE / ANALYZED

- 2B（90/90，max_tokens=256）：fp16 0.9630 / uniform 0.9815 / packed 0.9815 /
  k8v4 0.9630 / 4bit_nc 0.9444；配对 CI 全部含 0；`hit_final` 排序
  packed 0.9259 > fp16/uniform 0.9074 > 4bit_nc 0.8889 > k8v4 0.8519。
- 9B（54/54）：fp16 1.0000 / uniform 0.9815 / packed 0.9815；CI 含 0；
  `hit_last_section` 与 `hit` 零分歧。
- 证据：`results/quality/niah-fixed-analysis.json`、`niah-fixed-9b-analysis.json`。

### Q4 RULER v2 — DONE / ANALYZED

- 70/70（官方生成器 `c3f5e3b4`、`string_match_all`、20 samples/task/length、
  greedy 单 seed、max_tokens=256）。
- 要点：NIAH single 全 100；CWE 4K TurboQuant 掉分真实（k8v4 69.0 / 4bit_nc 70.0
  vs fp16 97.5）；multiquery 8K packed 88.75 最高；v1 短预算版保留为敏感性数据。
- 证据：`results/quality/ruler-subset-analysis-v2-256.json`。

### Q5 FWE no-think — DONE / ANALYZED

- 10/10（全 5 分配，max_tokens=256，chat template `enable_thinking=False`）：
  4K fp16 93.33 / uniform 88.33 / packed 88.33 / k8v4 93.33 / 4bit_nc 96.67；
  8K 全部 100。

### Q6 Reasoning（no-think v2）— DONE / ANALYZED

| bench | n | fp16 | uniform | packed | k8v4 | 4bit_nc |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K | 200 | 0.760 | 0.695 | 0.680 | 0.675 | 0.685 |
| MMLU | 500 | 0.588 | 0.586 | 0.596 | 0.600 | 0.612 |
| AIME25 | 30 | 0.167 | 0.100 | 0.167 | 0.067 | 0.100 |

- 协议：no-think + 大预算（1024/512/4096）+ 最终答案抽取；单 seed；
  thinking/小预算版本为敏感性数据。
- 结论边界：GSM8K 量化列点估计低 fp16 6.5–8.5pt（需多 seed，见 B2）；
  MMLU 持平；AIME 预算受限、近地板。

### Q7 LongBench v1 — DONE / ANALYZED

- 8 任务 × 50 样本；官方 v1 prompt/指标（commit `4c4b985bcf`）；2B 全 5 分配 +
  9B 核心 3 分配；64/64。
- 要点：QA/摘要量化列与 fp16 ±1.5pt 内持平；TREC 2B uniform/packed −6pt 点估计；
  9B 代码任务量化列一致偏低（LCC −1.8/−2.6，RepoBench −2.0/−5.2）；
  2B 代码任务近地板；截断各分配对称。
- 证据：`results/quality/longbench-analysis-20260807.json`。

## 4. 机制与通用性实验（M）

| ID | 实验 | 状态 | 目的 | 对应审稿要求 |
|---|---|---|---|---|
| M1 | A2 运行时门禁（8/8 检查） | DONE/VERIFIED | 单 backing、混 dtype、GDN dtype、真实生成 | W2 机制正确性 |
| M2 | A2 独立主机复现 | DONE/VERIFIED | 冻结代码+SHA 部署，比例复现 | 可复现性 |
| M3 | 9B packed 容量探针 | PENDING | packed 恢复在 9B 成立（现只有 0.258 塌缩） | W2 通用性 |
| M4 | 纯注意力对照（Qwen2.5-7B） | PENDING | 3.88×→2.245× 稀释是 hybrid 性质 | W2/W3 一般性 |
| M5 | A2 上游化 patch（diff+单测） | PENDING（工程） | 与 DSV4 packed 路径的边界证明 | W2 机制增量 |

## 5. 待补实验计划（Priority-ordered）

| 优先级 | 实验 | 成本 | 命令/入口 | 完成门禁 |
|---|---|---|---|---|
| P0 | Q2 续跑（10 cells） | 1.5–2h | `run_ppl_extra.sh ppl-extra-20260807` | 12/12 + 分析 + sha |
| P0 | E4 Serving Formal | ~8h | 人工放行 → `run_serving_formal.sh` | 108/108 + ANALYZED→（复现后 VERIFIED） |
| P0 | M3 9B packed 容量探针 | 0.5–1h | 复用 A2 probe 脚本 | 3/3 probes + ratio gate |
| P0 | M4 纯注意力对照 | 1–2h | 同容量协议 @ Qwen2.5-7B | 稀释对比表 |
| P1 | B2 GSM8K 3-seed | 1–1.5h | `run_reasoning_bench.sh --seeds 7,42,2026`（改造） | 配对 CI |
| P1 | B1 外部 baseline | 2–5 天（评估后） | KIVI/KVQuant 同协议实现 | 同硬件同协议对照表 |
| P2 | B3 LongBench 多 seed | 2–3h | `longbench_bench.py` 加 seeds | 关键任务 CI |
| P2 | B4 32K/64K 探针 | 2–4h | 容量+检索探针 | 长上下文优势扩展 |

## 6. 声明-证据一致性（Claims Gate）

**允许的声明（现有证据支撑）：**
- int4 端到端容量比 2.245×@4K / 3.155×@16K（2B）、2.19×@4K（9B），机制 3.88×；
- GDN state 稀释：18.63 MiB/请求（代码推导），≈60% KV 预算（标注为估计）；
- SLO 边界 workload-dependent：Random 0%/+4.8%/+14.3%，ShareGPT −17.6%（VERIFIED）；
- packed per-layer：0.258×→0.833×（3.232× vs legacy），独立复现 VERIFIED；
- 质量持平（点估计口径）：PPL/NIAH/RULER/FWE/MMLU/LongBench；
- 等字节排序：4-bit+驱逐优于 <4-bit 全保留（3-seed CI）。

**禁止/暂缓的声明：**
- 任何 "+25% SLO" 或 workload-general 容量收益；
- "packed 布局是我们提出的新机制"（应写"扩展 vLLM 既有 packed 路径"）；
- 单 seed 实验的显著差异结论（GSM8K 负 delta、TREC −6pt、9B 代码 delta）；
- AIME25 的跨分配比较（预算受限、近地板）；
- Serving formal 前的 ANALYZED 边界作为 headline；
- "first system study" 类绝对化表述（需收窄范围）。

## 7. 数据溯源清单

| 证据类别 | 路径 |
|---|---|
| 质量原子结果 | `results/quality/`（NIAH/RULER/FWE/reasoning/LongBench/ppl-extra） |
| 质量分析 | `results/quality/*-analysis*.json` |
| 容量/门禁 | `results/verified/2026-08-04/a2/` |
| 独立复现 | `results/reproduction/2026-08-05/a2/` |
| Serving gates | 服务器 `/root/autodl-tmp/r5-serving-20260807/` |
| E3 formal | `results/verified/2026-08-04/e3/` |
| 语料清单 | `data/MANIFEST.yaml` + `data/*_manifest.json` |

## 8. 审稿要求映射（ARS Required Revisions）

| 要求 | 状态 |
|---|---|
| R1 全篇替换为 workload-specific VERIFIED 边界、删除 +25% | **DONE**（commit `0cf6039`） |
| R2 A2 方法章节 + 机制增量论证 | DONE（§4 已入文）；M3/M5 补证据 |
| R3 PPL 三文件统一 + retrieval/long-context | DONE（Q1/Q3/Q4/Q7） |
| R4 外部 baseline | **PENDING（B1）** |
| R5 收窄 first-claim、删占位 | DONE（占位清理）；first-claim 措辞待终稿复核 |
| R6 references.bib 完整 | DONE（无占位扫描通过） |
