# 研究汇总：混合线性注意力模型服务内存的精度预算分配（KV 位数 × state 位数）

> 文档日期：2026-08-09。本文汇总研究方向、方法、实验矩阵、进度、已有结果与待补实验，作为论文写作与后续实验的统一参照。所有数字均来自已入库 artifact（结果文件 + sha256 + commit），未入库内容一律标注“待补/未测”。

---

## 1. 研究方向

### 1.1 核心问题

以 Qwen3.5（Gated DeltaNet + GQA 混合架构）为代表的混合线性注意力模型，在服务时有两类内存开销：

1. **注意力 KV cache**：随上下文长度 L 线性增长，可由 KV 量化（fp16 → int4 等）压缩；
2. **GDN 循环 state**：每序列固定大小，与 L 无关，此前被视为“任何 KV 方案都无法量化”的固定稀释剂。

二者共享同一 GPU KV 内存池。我们的研究方向是：**把 state 精度作为与 KV 位数并列的第二个精度预算维度，建立统一的容量模型，并用容量 × 质量 × serving 的闭环验证支撑系统级主张**。

### 1.2 目标定位

论文贡献定位（区别于“新 kernel/新机制”）：

- **视角**：混合模型服务内存是“KV 位数 × state 位数”的二维精度预算；
- **模型**：`C(L; A, G) = M/(A·L + G)`，A = 每 token 注意力 KV 字节（fp16/int4），G = 每序列 state 字节（fp32/bf16），跨 2B/9B、4K/16K 验证；
- **闭环**：容量实测 + PPL/RULER/GSM8K 质量 + 逐层敏感度负面结果 + serving 边界实验；
- **已有机制**：packed per-layer page groups（A2）作为 KV 维度混合精度的部署机制。

### 1.3 与现有工作的边界（prior art，均已完成文献核验）

| 工作 | 内容 | 与本工作的差异 |
|---|---|---|
| ReplaySSM（Tri Dao 2026-06；vLLM issue #47572、PR #47576/#48792/#49847） | 缓存输入 (d,k,g) 代替写回 state，优化解码带宽，支持 Mamba2 + GDN | 优化写回/带宽，不是存储精度预算；本 fork 已内置 `use_replayssm` 配置 |
| vLLM PR #43518（WIP） | FlashInfer checkpointing SSU，Mamba2 SSM state 支持 FP8/Int8/Int16 | 单点 kernel/cache 能力，无 GDN 系统研究、无容量模型、无质量闭环 |
| vLLM PR #51052（Kimi-K3） | 混合 KDA conv+ssm state 的 MoRIIO KV 传输（1P1D 分离式） | 跨机传输，不涉及精度/容量预算 |
| Quamba/Quamba2/MambaQuant | SSM 权重/激活 PTQ | 模型级量化，不是服务期 cache 精度分配 |
| vLLM issue #37121 | 混合 Mamba/Attention 模型 KV 内存估计偏大 | 与我们的容量建模直接相关，可作为 related work |

**红线**：不能主张“首次做 state 压缩/state 量化”，只能主张“state 精度作为与 KV 位数联合分配的预算维度 + 预测模型 + 闭环验证”。

---

## 2. 研究方法与思路

### 2.1 方法学

1. **证据闭环**：每个主张必须有 容量探针（vLLM 实测）→ 容量模型（解析预测）→ 质量（PPL/RULER/GSM8K）→ serving（SLO 边界） 四层中的对应证据；
2. **决策门**：重大投入前先跑低成本判别实验（如逐层敏感度扫描），数据不支持就放弃，避免无效实现；
3. **防伪影**：实验内置审计（cast 计数/dtype 记录、required log substrings、参考值逐位复现、sha256 原子写、fail-closed analyzer）；
4. **诚实披露**：所有回退、伪影（thinking 截断）、协议差异、未测项都明确标注，不把 pilot 当 formal、不把 harness 模拟当 kernel 语义。

### 2.2 关键技术机制

- **vLLM 侧（真实 serving 语义）**：fork 原生支持 `--mamba-ssm-cache-dtype`（`MambaDType = {auto, float32, float16, bfloat16}`）。Qwen3.5 的 HF 配置 `mamba_ssm_dtype=float32`，显式传 bf16 会触发 warning `Using the user-specified value`（作为 serving 日志硬校验）。链路：EngineArgs → CacheConfig → `mamba_utils.py` 的 `temporal_state_dtype` → GDN cache spec dtypes `[conv, temporal]` → 缓存张量分配。
- **GDN state 构成（2B，每层每 4096-token block）**：temporal `(16,128,128)` fp32 = 1,048,576 B + conv `(3,6144)` bf16 = 36,864 B = 1,085,440 B；2B ×18 = 19,537,920 B（18.63 MiB/请求）；9B ×24 = 26,050,560 B。bf16 只减半 temporal：每层 561,152 B，2B G_bf16 = 10,100,736 B，9B = 13,467,648 B。
- **harness 侧（PPL 敏感度，transformers）**：`hybrid_premise.py` 新增 `--state-dtype` 与逐层 `layer_ids`/`audit`，在 recurrent state 每次写回缓存边界 cast 到目标 dtype（chunk 粒度模拟，不是 kernel 逐 token 语义）。

### 2.3 容量模型

论文 §3.3 原模型（KV 量化，A 变化、G 固定）：

$$r_s(L) = \frac{A_f L + G}{A_q L + G}, \qquad r_s(\infty) = A_f/A_q = 3.878$$

state 精度扩展（A 固定、G 可量化）：

$$r_{\text{state}}(L) = \frac{A_q L + G_{\text{fp32}}}{A_q L + G_{\text{bf16}}}$$

参数（论文口径）：2B `A_q=3,168`、`G=19,537,920`；9B `A_q=16,384/3.878≈4,224.86`、`G=26,050,560`。

**由模型可导出的设计规则**：`C = M/(A·L + G)`，则 `∂C/∂A = −ML/(A·L+G)²`、`∂C/∂G = −M/(A·L+G)²`，即“省 1 字节 KV ≈ 省 L 字节 state”。state 精度收益集中在短上下文/高并发，KV 量化收益随 L 增长，两维度互补。

---

## 3. 实验矩阵设计

### 3.1 总览

| 矩阵 | 变量 | 配置 | 协议 | 状态 |
|---|---|---|---|---|
| M-capacity-state（容量探针） | state dtype ∈ {fp32, bf16, fp16} | uniform int4 KV；2B/9B；4K/16K | probe_ssm_state_dtype.py，gpu_mem 0.85 | ✅ 10/10 |
| M-capacity-legacy（早期探针） | state dtype ∈ {fp32, bf16} | per-layer legacy（无 packed）；2B；4K | 同上 | ✅ 2/2 |
| Q-PPL-state | state dtype ∈ {fp32, bf16} | fp16 KV；2B/9B；C4/PG19；3 seeds | hybrid_premise，5×2048，chunk128 | ✅ 8/8 |
| Q-RULER-2B | state dtype ∈ {fp32, bf16} | fp16 KV；7 tasks × 2 lengths；seed7；max_tokens=256 | vLLM 离线，thinking=default | ✅ 14/14 |
| Q-GSM8K-2B | state dtype ∈ {fp32, bf16} | fp16 KV；200 samples；3 seeds | vLLM 离线，no-think | ✅ 3/3 |
| Q-RULER-9B | state dtype ∈ {fp32, bf16} | fp16 KV；同上 9B | 同上 | ✅ 28/28（两状态） |
| Q-GSM8K-9B | state dtype ∈ {fp32, bf16} | fp16 KV；3 seeds；9B | 同上 | ✅ 6/6 |
| S-sensitivity（决策门） | 单层 bf16 vs 全局 vs fp32 | 2B；C4+PG19；3 seeds；20 配置 | hybrid_premise + 逐层 audit | ✅ 2/2 语料 |
| S-pilot（serving pilot） | state dtype ∈ {fp32, bf16} | Random60；seed7；rates 30/40/50；60s 窗 | protocol-v3 steady-state | ✅ 6/6 |
| S-formal（serving formal） | state dtype ∈ {fp32, bf16} | Random60 + ShareGPT300；3 seeds | protocol-v3 | ⏳ 待补 |
| M-capacity-2x2 补齐 | KV ∈ {fp16, int4} × state ∈ {fp32, bf16} | 2B/9B；4K/16K | 同探针协议 | ⏳ 待补 |
| Q-stacking | KV int4 × state bf16 叠加 | PPL（2B C4/PG19 3 seeds）+ GSM8K（3 seeds） | 同上 | ⏳ 待补 |

### 3.2 关键协议说明

- **容量探针**：`kv_cache_dtype=int4_per_token_head`（uniform，`per_layer={}`）或 fp16（auto），`gpu_memory_utilization=0.85`，`max_model_len` 4K/16K；报告 resolved `mamba_ssm_cache_dtype`、capacity tokens、max concurrency、总分配字节。
- **PPL**：5 序列 × 2048 tokens、chunk 128、3 seeds（7/42/2026）、注意力 KV fp16；fp32 基线与 bf16_all 必须逐位复现已入库矩阵（ref diff = 0）。
- **RULER**：官方 `string_match_all`，7 tasks × 4096/8192，seed 7，max_tokens 256（v2-256 口径，thinking=default；FWE 头条走 nothink 协议，本矩阵仅作同协议参考）。
- **GSM8K**：前 200 条、greedy、no-think、3 seeds；提取偏好最后 answer 标记。
- **serving**：protocol-v3，失败计为 SLO miss，TTFT 阈值 250/500/1000/2000/3000 ms、TPOT 200 ms、可持续 goodput ≥0.95；Random60 60s 窗 / ShareGPT300 300s 窗；120 预热请求。

---

## 4. 实验进度

### 4.1 已完成（全部入库）

| 内容 | commit | 证据路径 |
|---|---|---|
| 可行性检查（文献 + 服务器开关） | 258ba82 | docs/notes/state-compression-feasibility-2026-08-08.md |
| 早期 per-layer 容量探针（2B，+37.5%） | 258ba82 | results/verified/2026-08-08/ssm_dtype/ |
| PPL 8 格矩阵（2B/9B × fp32/bf16 × C4/PG19） | 903a38d | results/quality/ppl-state-dtype/ + analysis |
| vLLM 侧质量 2B（RULER 14 + GSM8K 3） | eb86037 | results/quality/ruler-subset/…、reasoning/… |
| vLLM 侧质量 9B（RULER 28 + GSM8K 6） | da404c0 | 同上 9b attempt |
| 容量模型验证（uniform int4 10 探针） | 19fa395 | results/verified/2026-08-08/capacity-state/ + analysis |
| 逐层敏感度决策门 | 1e37fcd | results/quality/state-sensitivity/ + analysis |
| serving pilot（Random60 seed7，6/6） | d39e98c + 56674fd | results/verified/2026-08-09/…pilot…/ + analysis |

### 4.2 论文原有已完成（摘要，供定位）

- E1 容量：uniform int4 2.245×@4K（1,203,106→2,701,721）、3.155×@16K（1,556,961→4,910,731）；9B 2.19×（150,062→328,499）；纯注意力控制 3.765×（M4）。
- E4 serving formal：108/108（Random60 45 + ShareGPT300 63；TQ k8v4/4bit_nc/fp8）。
- 质量：C4/PG19 12/12（量化 Δ+1.1~1.6% @2B / ~1.1% @9B）；LongBench 64/64；RULER v2 矩阵；GSM8K 3-seed 量化回退（uniform −6.2pt [−6.9,−5.5] 等，CI 不含 0）；KIVI 风格外部基线（Δ≈0）。
- 机制：packed per-layer page groups（A2）修复 legacy 混合精度容量塌缩。

### 4.3 进行中 / 未开始

- 2×2 容量矩阵补齐（fp16 KV × fp32/bf16 state）；
- 质量叠加（uniform int4 KV × bf16 state 的 PPL/GSM8K）；
- serving formal（3-seed，Random60 + ShareGPT300）；
- 论文叙事重写（旧“state 不可量化”表述必须删除）。

---

## 5. 已有实验结果

### 5.1 容量（vLLM 实测，uniform int4）

| 模型 | L | fp32 (tokens) | bf16 (tokens) | 实测比 | 模型预测 | 误差 |
|---|---|---:|---:|---:|---:|---:|
| 2B | 4096 | 2,692,710 | 3,703,954 | 1.3755 | 1.4089 | −2.37% |
| 2B | 16384 | 4,895,837 | 5,458,458 | 1.1149 | 1.1522 | −3.24% |
| 9B | 4096 | 315,392 | 443,538 | 1.4063 | 1.4089 | −0.18% |
| 9B | 16384 | 573,440 | 653,635 | 1.1398 | 1.1522 | −1.07% |

- fp16 与 bf16 容量完全一致（同字节数）；早期 per-layer legacy 2B@4K 亦为 +37.5%（694,272→954,855）。
- 结论：**2B/9B、4K/16K 四点验证，模型误差 −3.24%~−0.18%**；收益随 L 衰减（4K +38~41%，16K +11~14%）。

### 5.2 PPL（harness 级写回舍入，3 seeds 配对）

| 模型 | 语料 | fp32 | bf16 | Δ | 95% CI |
|---|---|---|---:|---:|---|
| 2B | C4 | 17.5800 | 17.5797 | −0.0003 | [−0.0016, +0.0010] |
| 2B | PG19 | 27.1783 | 27.1787 | +0.0004 | [−0.0033, +0.0041] |
| 9B | C4 | 12.7287 | 12.7289 | +0.0002 | [−0.0022, +0.0026] |
| 9B | PG19 | 18.0016 | 18.0022 | +0.0006 | [−0.0025, +0.0037] |

四个 CI 均含 0：**bf16 state 在 PPL 上与 fp32 统计不可区分**。

### 5.3 RULER / GSM8K（vLLM 真实 kernel 路径）

| 模型 | 指标 | fp32 | bf16 | Δ |
|---|---|---|---:|---:|
| 2B | RULER 总体（14 格） | 87.83 | 88.33 | +0.49 |
| 2B | GSM8K 配对（3 seeds） | 0.755–0.76 | 0.73×3 | −2.67pt，CI [−3.38, −1.95] |
| 9B | RULER 总体（14 格） | 95.83 | 95.12 | −0.71 |
| 9B | GSM8K 配对（3 seeds） | 0.885×3 | 0.88×3 | −0.5pt，CI [−0.5, −0.5] |

- RULER 非零格均与 think 截断伪影/单 seed 抽奖相关（如 2B FWE L4096 +8.33：20/20 未闭合 `<think>` 且 256 token 用满；9B niah_multiquery L8192 −6.25：13–15/20 截断；9B FWE L8192 −5.0：1 样本翻转）。
- GSM8K 回退真实、稳定，论文必须披露。

### 5.4 逐层敏感度（决策门）

- 2B，C4+PG19，3 seeds，20 配置；fp32/bf16_all 与已入库矩阵**逐位一致**（ref diff = 0）；cast 审计全部通过（bf16_all 18 层全 cast、单层配置只 cast 目标层）。
- 所有单层 |Δ| ≤ 0.00074（C4）/ 0.001274（PG19）；仅 2/36 个 CI 恰好不含 0（C4 L2 +0.00039、L8 +0.00066），PG19 不显著，量级 ~0.002–0.004%，为多重比较噪声。
- **结论：无逐层 state 精度收益空间，不做逐层 state dtype 机制；全局 bf16 即可**。

### 5.5 serving pilot（Random60，seed 7，60s 窗）

| rate | 配置 | TTFT p99 (ms) | TPOT p99 (ms) | goodput@1s | goodput@3s | 可持续 |
|---|---|---:|---:|---:|---:|---|
| 30 | fp32 | 222.6 | 21.4 | 0.983 | 0.983 | 全部阈值 |
| 30 | bf16 | 233.2 | 18.8 | 0.983 | 0.983 | 全部阈值 |
| 40 | fp32 | 4,686.3 | 48.7 | 0.348 | 0.666 | 否（任何阈值） |
| 40 | bf16 | 1,540.4 | 45.9 | 0.781 | 0.958 | TTFT≥2s |
| 50 | fp32 | 20,987.5 | 48.9 | 0.064 | 0.142 | 否 |
| 50 | bf16 | 16,166.2 | 45.9 | 0.082 | 0.177 | 否 |

- bf16 服务器日志含 `Using the user-specified value` + `CUDAGraphMode.PIECEWISE`（配置生效硬证据）。
- 解读：r40 边界区，bf16 把 goodput 从 0.67 推到 0.958（≥0.95），TTFT p99 4.7s→1.5s，方向与 +37.6% 容量一致。**pilot 为方向性证据，论文 claim 需 formal**。

### 5.6 未测 / 明确未做

- fp16 state 的质量（容量与 bf16 相同，质量未测）；
- 9B serving formal；
- 16K 质量（RULER/GSM8K）；
- uniform int4 KV × bf16 state 的叠加质量与 serving；
- 2×2 联合容量表中的 fp16 KV 列（同协议探针）。

---

## 6. 仍需补全的实验

### 6.1 必做（论文闭环）

| 任务 | 内容 | 预估耗时 |
|---|---|---|
| M-2x2 容量补齐 | fp16 KV × {fp32, bf16} state：2B 4K/16K + 9B 4K，共 6 探针 | ~30 min |
| Q-stacking PPL | uniform int4 KV × {fp32, bf16} state，2B C4/PG19，3 seeds（4 cells） | ~15 min |
| Q-stacking GSM8K | uniform int4 KV × {fp32, bf16} state，2B，3 seeds | ~30 min |
| S-formal | Random60 + ShareGPT300，3 seeds，uniform int4 × {fp32, bf16} state | ~6–7 h（可挂机） |
| 联合分析 | 2×2 容量/质量帕累托表 + 设计规则验证 | ~1 h |

### 6.2 可选

- 9B 的 stacking 质量（PPL/GSM8K）；
- 9B serving formal；
- fp16 state 质量抽样；
- 16K 下 GSM8K/RULER；
- 联合容量模型工具脚本（给定预算输出最优 KV/state 分配）。

---

## 7. 论文定位与可写 claim

### 7.1 主张（claim whitelist，均需证据）

1. 混合模型服务内存是“KV 位数 × state 位数”的二维精度预算，现有工作只优化其中一维；
2. `C(L; A, G) = M/(A·L + G)` 统一容量模型，state 可量化时 `r_state(L)` 在 2B/9B、4K/16K 误差 <3.3%；
3. bf16 state 在 4K（uniform int4）带来 +38~41% 容量、16K +11~14%；PPL/RULER 基本持平，GSM8K 有披露回退（2B −2.7pt / 9B −0.5pt）；
4. 逐层 state 精度无质量收益（负面结果），全局 state 精度即可；
5. 容量收益在 serving 边界区转化为 SLO 收益（pilot 方向性证据，formal 待补）；
6. packed per-layer page groups 解决 KV 维度的混合精度部署（已有）。

### 7.2 红线与诚实披露

- 不得写“首次 state 压缩/量化”（ReplaySSM、PR #43518 存在）；
- 删除论文旧表述“state 不可量化/固定稀释剂”，改为“state 是第二个精度预算维度”；
- 容量数字必须标注 L 与 KV 配置；pilot 不得当 formal；harness 敏感度注明 chunk 级舍入；
- GSM8K 回退、FWE think 伪影、9B 单 seed 抽奖必须如实披露；
- 不出现占位符与未验证数字。

---

## 8. 复现与证据索引

### 8.1 关键脚本

- 容量探针：`scripts/bench/probe_ssm_state_dtype.py`、`scripts/bench/run_capacity_state_probes.sh`
- 容量模型分析：`scripts/bench/analyze_capacity_state.py`
- PPL/敏感度：`scripts/exp/hybrid_premise.py`（`--state-dtype`、逐层 `layer_ids`/`audit`）、`scripts/exp/run_state_sensitivity.py`、`scripts/exp/run_ppl_state_dtype.sh`
- vLLM 质量：`scripts/eval/kv_quality_retrieval.py` / `ruler_quality.py` / `reasoning_bench.py`（`fp16_statebf16` allocation）、`run_ruler_statebf16.sh`、`run_gsm8k_statebf16.sh`、9B 版本
- 分析器：`analyze_ppl_state_dtype.py`、`analyze_ruler_statebf16.py`、`analyze_gsm8k_statebf16.py`、`analyze_state_sensitivity.py`、`analyze_statebf16_pilot.py`
- serving：`experiments/configs/statebf16_random60_pilot.yaml`（sha256 f7475580…）、serving worktree `run_steady_state.py`

### 8.2 关键结果目录

- `results/verified/2026-08-08/ssm_dtype/`、`.../capacity-state/`
- `results/quality/ppl-state-dtype/`、`.../ruler-subset/`、`.../reasoning/`、`.../state-sensitivity/`
- `results/verified/2026-08-09/statebf16-random60-pilot-20260809/`

### 8.3 相关 commit（本方向）

`258ba82`（可行性+早期探针）→ `903a38d`（PPL 8 格）→ `eb86037`（2B 质量）→ `da404c0`（9B 质量）→ `19fa395`（容量模型验证）→ `1e37fcd`（敏感度决策门）→ `d39e98c`/`56674fd`（serving pilot）。

---

## 9. 下一步计划

1. 今天：M-2x2 容量补齐 + Q-stacking PPL/GSM8K（约 1.5~2 h）；
2. 今晚挂机：S-formal（Random60 → ShareGPT300，约 6~7 h）；
3. 明天起：联合帕累托分析 + 论文叙事重写（2~3 天）；
4. 提交前：claim-evidence-map 更新、旧表述清理、全库数字核对。
