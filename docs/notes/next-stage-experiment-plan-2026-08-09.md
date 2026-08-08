# 下一阶段实验补全计划（ARS 审稿 2026-08-09 版）

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-09
- Verification Status: UNVERIFIED（本文件是计划，未声明任何实验已执行）
- Version Label: code_plan_v1

> 输入：`docs/notes/mlsys-review-ars-2026-08-09.md`（5/5 全席、Major Revision、
> 平均加权 68.3）、`docs/paper/research-summary-2026-08-09.md`、
> `docs/paper/experiment-matrix-plan-2026-08-08.md`、`docs/paper/claim-evidence-map-2026-08-08.md`、
> `results/verified/2026-08-08/{ssm_dtype,capacity-state}/`、
> `results/verified/2026-08-09/statebf16-random60-pilot-20260809/`、
> `results/quality/{ppl-state-dtype,state-sensitivity,ruler-subset,reasoning}/` 全部分析 JSON，
> 以及 `scripts/bench/probe_ssm_state_dtype.py`、`scripts/bench/run_capacity_state_probes.sh`、
> `scripts/bench/analyze_capacity_state.py`、`scripts/exp/hybrid_premise.py`、
> `scripts/exp/run_state_sensitivity.py`、`scripts/eval/reasoning_bench.py`、
> `scripts/eval/kv_quality_retrieval.py`、`scripts/eval/ruler_quality.py`、
> `scripts/eval/ruler_prepare.py`、`scripts/bench/run_steady_state.py` 等入口。

---

## 0. 目的与状态

本计划把 2026-08-09 ARS 最大强度模拟审稿的全部修订项（R1–R16、S1–S7）映射到仓库
现有证据与脚本，给出**下一步实验补全的优先级、协议、入口、产物与门禁**。计划本身
不产生实验数字；每完成一步后按仓库纪律原子化归档（sha256 + commit）并回填状态。

相对旧文档的关系：

- 本计划**扩展并部分取代** `docs/notes/next-stage-experiment-plan-2026-08-04.md`
  （该文件面向 A2/E3 旧方向，其 A2/E3 部分已完成；state 精度方向以本计划为准）。
- 本计划是 `docs/paper/experiment-matrix-plan-2026-08-08.md` 的 08-09 增量，
  不改写已审稿的 08-08 矩阵文件（保留审稿输入快照）。
- 旧协议产物（固定 head-200 GSM8K、单 dataset-seed RULER、pilot 单 seed）全部保留，
  不合并进新分母；协议变更后新 attempt 独立成档。

---

## 1. 审稿结论摘要与投稿红线

审稿结论：**Major Revision（可修复）**。无“文本与已验证证据矛盾”类缺陷，全部问题为
证据链未完成 + 统计表述可修复。加权 66.0 / 66.8 / 69.8 / 70.4，平均 68.3；
EIC 的 Evidence Sufficiency 判 block（repairable）。DA 三条 CRITICAL 全部 VALIDATED：

| DA-CRITICAL | 内容 | 裁决后必改 |
|---|---|---|
| C1 | 9B GSM8K 零宽 CI（3 seed 全同） | 核查 seed 语义；改为真实随机协议或确定性表述 |
| C2 | claim #5 仅单 seed pilot | S-formal 完成前降级 ANALYZED，不得进 Abstract |
| C3 | 容量模型 4 点误差全负 | signed error + block 粒度机制 + “保守下界”定位 |

投稿红线（审稿一致结论）：

1. **M-2x2 容量、Q-stacking 质量、S-formal serving 三组实验完成前不投**（约 8–9 h 计算量）。
2. 9B GSM8K 禁止用退化 CI 支撑显著性表述。
3. claim #5 在 S-formal 前只能写 “ANALYZED / formal pending”。
4. “第二维度”叙事 = “与 KV 量化正交可复合”，不是“幅度并列”（2×2 表 + stacking 证据）。
5. 容量模型定位为“保守下界”（signed error + block 粒度机制讨论），不是无偏点估计。

---

## 2. 全量映射表（R1–R16 + S1–S7）

| # | 修订项 | 现状证据（已核实） | 缺口 | 处置 | 优先级 |
|---|---|---|---|---|---|
| R1 | 9B GSM8K seed 语义核查 + CI 修复 | `reasoning_bench.py` 固定 head-200 + greedy(temp=0) + engine seed；9B 三 seed 全同 | seed 不产生真实随机性，配对 CI 语义不成立 | P0-1：核查记录 + 改为 seed 化题目子采样协议重跑 | **P0** |
| R2 | M-2x2 容量补齐 | int4 KV 10/10 探针（含 fp16/bf16 state）；`capacity-state-analysis.json` 4 点误差 −2.37/−3.24/−0.18/−1.07% | fp16 KV × {fp32,bf16} state 的 2B 4K/16K + 9B 4K 未测 | P0-2：fp16-KV 6 探针 + 2×2 分析 | **P0** |
| R3 | Q-stacking 质量叠加 | harness `--bits 4` 可用；无 `uniform_int4_statebf16` 分配 | int4 KV × bf16 state 的 PPL/GSM8K 未测 | P0-3：harness PPL 4 格 + vLLM GSM8K 新分配 6 格（新 seed 协议） | **P0** |
| R4 | S-formal serving | E4 六列 108/108 完成（旧方向）；statebf16 pilot 6/6 单 seed（fp16 KV） | int4 KV × {fp32,bf16} state 的 Random60+ShareGPT300 3-seed 未测 | P0-4：MVEx→Pilot→Formal 72 样本，挂机 6–7 h | **P0** |
| R5 | 容量模型偏差讨论 | `gap_pct` 已含符号；fp32 block 2064/3287 vs bf16 block 1072/6330（探针 JSON 已核实） | 未写成“保守下界”、未披露 4/4 全负（P=0.0625） | P0-5：分析器补 block 粒度字段 + 论文定位 | **P0** |
| R6 | 敏感度门统计修复 | 2/36 CI 不含 0（C4 L2/L8，均正），量级 0.0004–0.0007 | 无多重比较校正、无决策规则预注册 | P0-6：Bonferroni(α/36≈0.001389)/BH-FDR + 符号一致性披露 | **P0** |
| R7 | harness chunk 消融 | 协议 chunk=128；无 chunk=1 对照 | chunk 级写回舍入 vs per-token 语义未量化 | P1-1：chunk=1 vs 128 smoke（2B C4，1 seed） | P1 |
| R8 | RULER 非零格补 seed | 数据 `ruler_prepare.py --random-seed 42` 单 dataset seed；engine seed 7 | 5 个非零格（2B fwe L4096/L8192；9B niah_multiquery L4096/L8192、fwe L8192）单 seed | P1-2：生成 dataset seed {11,23} 并重跑两分配（20 新 cell） | P1 |
| R9 | GSM8K 不对称机制讨论 | 2B −2.67pt [−3.38,−1.95]；9B −0.5pt（3 seed 全同） | 无机制假说讨论（容量冗余/attention 补偿/规模/floor-ceiling） | P1-3：分析 + 写作（与 R1 新协议结果联动） | P1 |
| R10 | 文献扩展 | `state-compression-feasibility-2026-08-08.md` 已列 ReplaySSM/PR#43518/PR#51052/issue#37121 | 缺 Mamba2 state 精度原始讨论、FlashInfer SSU 独立条目、SSM state 表示理论、prior-art 数值直觉 | P1-4：逐条核验元数据 + 补 references.bib + prior-art 表 | P1 |
| R11 | claim whitelist 修订 | research-summary §7.1 六条 claim | claim #5 无依赖标注；claim 3 无单 seed/chunk 限定；claim 1 措辞过宽 | P1-5：按裁决修订 claim 表 + claim-evidence-map | P1 |
| R12 | “第二维度”叙事重写 | 容量模型设计规则 ∂C/∂A=−ML/(…)²、∂C/∂G=−M/(…)² 已推导 | 正文仍是“并列维度”口吻；缺 2×2/stacking 证据支撑 | P1-5：论文 §1/§3.3 重写（P0 实验后填表） | P1 |
| R13 | fp16 state 质量 smoke + bf16 下界讨论 | fp16 state 容量=bf16（10 探针已测）；质量未测 | 缺 fp16 PPL smoke；缺精度谱系边界声明 | P1-3：2B C4 1 seed smoke（~5 min）+ Discussion | P1 |
| R14 | 一般性声明收窄或跨架构探针 | 仅 Qwen3.5-2B/9B；M4 纯注意力 Qwen2.5-7B 对照已有 | 无非 Qwen3.5 混合/SSM 架构探针 | P2-1：Mamba2-2.7B 4K 探针（gate：模型+内核可用）或 GDN-scope 声明 | P2 |
| R15 | 替代容量杠杆对比 | 已有 byte-budget 驱逐数据 + E3/E4 serving 表 | 无 H2O/SnapKV/PyramidKV/offloading/prefix caching 对照段、无交叉点图、无 TP 讨论 | P2-2：解析推导交叉点 L* + 写作段（不新跑 serving） | P2 |
| R16 | train-inference mismatch 讨论 | GSM8K 回退真实（PPL 无感、GSM8K 有感） | 无训练-推理精度失配假说讨论 | P2-3：Discussion + 可选最小验证 | P2 |
| S1 | A_q 推导链入文 | analyzer 参数 2B A_q=3168、9B A_q=16384/3.878 | 方法论节缺推导链与交叉验证 | P2-4：§3.3 方法学补推导 | P2 |
| S2 | fp16 列 / signed error 列 / r_state(L) 曲线 | 分析 JSON 已有数据 | 表 5.1 缺 fp16 列；容量表缺 signed error；缺曲线图 | P2-4：P0-2/P0-5 产物回填论文 | P2 |
| S3 | PPL 最小可检测效应 | PPL 8 格 CI 全含 0 | 无 80% power MDE/等价性 margin 预注册 | P2-4：统计附录补 MDE | P2 |
| S4 | 数据地图 MANIFEST.json / 索引 | data/MANIFEST.yaml 已有 | results 缺索引文件 | P2-5：生成 `results/MANIFEST.json` | P2 |
| S5 | pilot worktree commit 澄清 | attempt_contract git_commit 3267efa vs summary d39e98c/56674fd | 文档未解释两仓库 commit 关系 | P2-5：provenance 注记 | P2 |
| S6 | “bit-match”术语澄清 | 容差 1e-9 的逐位比较 | 术语易误读为严格位级 | P2-5：全文措辞统一 | P3 |
| S7 | 旧方向与新方向衔接 | A2 packed 机制已完成 VERIFIED；state-bits 为新 headline | 两贡献如何共处一篇未定 | P2-5：章节结构决策（同篇两节 or 拆分） | P2 |

---

## 3. 通用实验契约（所有新实验必须遵守）

- 环境：RTX 5090 32GB（`remote_5090`，sm_120）；论文 headline 只出自该环境；
  运行前 `env_check.sh` 校验 driver/CUDA/vLLM wheel 与锁定版本一致。
- 代码：commit-before-run；serving 相关改动进 serving worktree
  （`/root/autodl-tmp/MLSys_Serving_f7a79f5`，截至 08-07 为 3267efa），
  研究脚本进本仓库；每次运行前 git 树 clean（`require_clean_git` 按协议）。
- 产物：原子 JSON + `.sha256` + contract/status sidecar；失败 attempt 保留隔离，
  不进入正式分母；`--resume` 只跳过 `completed_validated`。
- 配置生效：每 cell 必须留硬证据（日志子串 / `config_effect` / resolved dtype），
  缺证据即 fail-closed，不采信“应该生效”。
- 统计：headline = 3 seeds 配对 mean±CI；多重比较预先声明校正规则；
  pilot 不得当 formal；harness chunk 级写回舍入不得冒充 kernel per-token 语义。
- 监控：`scripts/monitor_progress.sh` + 链式脚本；长任务用 nohup/Start-Process 挂机，
  超时与失败由 runner fail-closed。

---

## 4. P0 必做（投稿红线，预计 2–2.5 天含挂机）

### P0-1：R1 — GSM8K seed 语义核查 + 真实随机协议重跑

**现状（本轮已核实）**：`reasoning_bench.py` 用 `df.head(200)` 固定题目集，
`temperature=0.0` greedy，seed 只传给 vLLM engine（采样器 RNG）。因此 9B 三 seed
结果全同（0.885/0.88）是确定性结果重复 3 次，不是随机重复；2B fp16 三 seed
0.760/0.755/0.755 的微小差异来自 attempt 间代码/引擎漂移而非 seed 效应。

**行动**：

1. 写核查结论到 `docs/notes/seed-semantics-audit-2026-08-09.md`（引用
   `reasoning_bench.py` 代码行、9B/2B 分析 JSON）。
2. 修改 `scripts/eval/reasoning_bench.py`：
   - `load_rows` 增加 seed 参数；GSM8K 由 `df.head(200)` 改为
     `df.sample(n=max_samples, random_state=seed)`（无放回），并记录
     `sampled_indices`（`df.index` 复位后保存）。
   - 保持 greedy（temp=0）；engine seed 保留作 provenance。
   - record 增加 `seed_semantics: "rows sampled by random_state=seed; decode deterministic"`。
3. 新运行器 `scripts/eval/run_gsm8k_state3seed_v2.sh`：
   - 2B：allocations `fp16, fp16_statebf16, uniform_int4, uniform_int4_statebf16` ×
     seeds {7,42,2026} × 200 samples = **12 cells**；
     attempt `reasoning-gsm8k-state3seed-v2-20260809`。
   - 9B（同日或次日）：`fp16, fp16_statebf16` × 3 seeds = **6 cells**；
     attempt `reasoning-gsm8k-9b-state3seed-v2-20260809`。
4. 分析：`analyze_gsm8k_statebf16.py` 与 `analyze_reasoning_gsm8k_3seed.py` 复用，
   输出配对 mean±CI；新 attempt 单独成档，旧 head-200 attempt 保留但不合并。

**产物**：`results/quality/reasoning/reasoning-gsm8k-state3seed-v2-20260809/*.json(.sha256)`、
同 9B attempt、`gsm8k-state3seed-v2-analysis-20260809.json`、audit 笔记。

**成功判据**：12/12（2B）与 6/6（9B）全部 `completed_validated`；每个 cell 记录
`sampled_indices` 且长度 200；分析输出含 per-seed 准确率与配对 CI；9B CI 不再退化
（若仍退化则按确定性结果改写 claim，并在论文标注协议）。

**依赖**：`kv_quality_retrieval.py` 新增 `uniform_int4_statebf16`（见 P0-3）先于或
与 P0-1 同步完成。

**耗时**：代码 1–2 h；2B 12 cells ≈ 40–60 min；9B 6 cells ≈ 40–60 min。

---

### P0-2：R2 — M-2×2 容量补齐（fp16 KV 列）

**行动**：

1. 新运行器 `scripts/bench/run_capacity_state_probes_fp16kv.sh`（复用
   `probe_ssm_state_dtype.py --kv-cache-dtype auto --kv-cache-dtype-per-layer '{}'
   --gpu-memory-utilization 0.85`）：
   - 2B × {auto, bfloat16} × L4096/L16384（4 探针）
   - 9B × {auto, bfloat16} × L4096（2 探针）
   - 可选：9B × L16384（2 探针，补齐对称矩阵）
   - 输出 `results/verified/2026-08-09/capacity-state-fp16kv/`，
     attempt `capacity-state-fp16kv-20260809`。
2. 分析器扩展（新增 `scripts/bench/analyze_capacity_2x2.py`）：
   - 合并 int4 与 fp16 两个 KV 列的 2×2 容量表（tokens + 比例）；
   - 每 cell 输出 measured/predicted ratio 与 signed gap（fp16 列用 A_f，
     int4 列用 A_q；A_q 推导链见 S1）；
   - 输出 block_size / num_gpu_blocks / mamba_page_size_padded（用于 R5）；
   - fail-closed：缺 cell 即退出。

**输入/输出**：

| 输入 | 路径 |
|---|---|
| 探针脚本 | `scripts/bench/probe_ssm_state_dtype.py` |
| int4 探针 | `results/verified/2026-08-08/capacity-state/` |
| 新 fp16 探针 | `results/verified/2026-08-09/capacity-state-fp16kv/` |
| 输出 | `results/verified/2026-08-09/capacity-2x2-analysis.json` |

**成功判据**：6/6（或 8/8）探针 exit 0 + resolved dtype 与参数一致 + sha 匹配；
预测误差按预声明 ±5% 容差评估（现有 int4 误差 ≤3.24%）；2×2 表完整。

**预期**：fp16 KV 下 r_state 更接近 1（模型预测参考：2B@4K ≈1.122、2B@16K ≈1.034、
9B@4K ≈1.156），与“state 收益在 KV 已量化时更显著”的互补叙事一致。

**耗时**：探针 ~30–40 min；分析 ~1 h。

---

### P0-3：R3 — Q-stacking 质量叠加

**A. harness PPL（4 格）**

1. 新运行器 `scripts/exp/run_ppl_stacking.sh`：
   `hybrid_premise.py --bits 4 --seeds 7,42,2026 --num-seqs 5 --max-len 2048
   --chunk 128 --state-dtype auto|bfloat16 --corpus data/{c4,pg19}_slice.txt
   --model <Qwen3.5-2B>`
2. 输出 `results/quality/ppl-stacking/ppl-stacking-20260809__<corpus>__state<fp32|bf16>__2b.csv(.seeds.csv)`。
3. 新分析器 `analyze_ppl_stacking.py`：配对 CI（int4+bf16 vs int4+fp32；
   再 vs 已有 fp16-KV state 矩阵，量化 KV 量化 × state 量化的叠加成本）。

**B. vLLM GSM8K（6 格，真实 kernel 路径）**

1. `scripts/eval/kv_quality_retrieval.py` 新增 allocation `uniform_int4_statebf16`：
   `kwargs` = `kv_cache_dtype="int4_per_token_head"` +
   `mamba_ssm_cache_dtype="bfloat16"`；`verify_config_effect` 同时校验
   `cache_dtype=="int4_per_token_head"`、`mamba_ssm_cache_dtype=="bfloat16"`、
   无 per_layer、无 A2 flag。
2. `reasoning_bench.py` ALLOCATIONS 加入 `uniform_int4_statebf16`。
3. 用 P0-1 的新 seed 协议跑 2B：`uniform_int4` 与 `uniform_int4_statebf16` ×
   seeds {7,42,2026} = **6 cells**（与 P0-1 的 fp16 两列合并成 4×3 配对表）。
4. 分析：`analyze_gsm8k_stacking.py` 输出 int4_statebf16 vs int4（叠加边际回退）
   与 vs fp16（总回退）的配对 CI。

**成功判据**：每 cell `config_effect.ok`；PPL 4 格 + GSM8K 6 格全部
`completed_validated`；分析含 CI；harness 侧注明 chunk 级写回舍入，vLLM 侧为
kernel 路径。

**耗时**：PPL ~15–20 min；GSM8K ~40–60 min；代码 ~1–1.5 h。

---

### P0-4：R4 — S-formal serving（int4 KV × {fp32,bf16} state）

**行动**（serving worktree 内完成）：

1. 新配置：
   - `experiments/configs/statebf16_int4_random60_formal.yaml`
   - `experiments/configs/statebf16_int4_sharegpt300_formal.yaml`
   - allocations：
     - `int4`：`--kv-cache-dtype int4_per_token_head` + PIECEWISE；
       required substrings：`int4_per_token_head`、`CUDAGraphMode.PIECEWISE`
     - `int4_statebf16`：追加 `--mamba-ssm-cache-dtype bfloat16`；
       required substrings：`int4_per_token_head`、`Using the user-specified value`、
       `CUDAGraphMode.PIECEWISE`
   - 其余参数与 A2 formal 配置逐项一致（60s/300s 窗、warmup 120、TTFT
     {250..3000}、TPOT 200、goodput≥0.95、`require_clean_git: true`）。
   - phases：mvex（2 alloc × random r30 + sharegpt r20 × seed7 = 4）、
     pilot（2 alloc × random r30/40/50 + sharegpt r30/40/50 × seed7 = 12）、
     formal（2 alloc × random [30..50] 5 rate + sharegpt [20..50] 7 rate ×
     seeds {7,42,2026} = 30 + 42 = **72 samples**）。
2. 新运行器 `scripts/bench/run_statebf16_serving_formal.sh`（镜像
   `run_serving_formal.sh`，输出根 `/root/autodl-tmp/statebf16-serving-20260809`）；
   attempt：`statebf16-v3-{random60,sharegpt300}-{mvex,pilot,formal}-20260809`。
3. 放行链：MVEx 4/4 → Pilot 12/12 → **人工审阅** → Formal 72/72（`--resume` 分片）。
4. 新分析器 `scripts/bench/analyze_statebf16_serving.py`：
   - 边界表（workload × TTFT threshold × allocation）；
   - 每 rate/threshold 的 paired goodput Δ（bf16−fp32）mean±CI（3 seeds）；
   - 失败按协议计入分母；arrival-window 偏差 ≤10%；
   - 每 allocation 从 server 启动日志/contract 记录 num_gpu_blocks 与
     concurrency，供“容量 vs 带宽”归因讨论（DA C2 替代解释）。
5. 产出 `results/verified/2026-08-09/statebf16-serving-formal-analysis.json`
   （原始产物在服务器输出根，按现有回传流程拉回 + 哈希核对）。

**成功判据**：72/72 completed_validated、0 进程失败；日志硬证据齐全；
分析输出边界表与 paired CI；状态先标 `ANALYZED`，独立复现门禁后才可升级
`VERIFIED`；claim #5 仅在升级后进 Abstract。

**耗时**：配置/代码 ~2 h；MVEx+pilot ~30 min；formal 挂机 6–7 h；分析 1–2 h。

---

### P0-5：R5 — 容量模型偏差分析（无新计算）

**行动**：

1. `analyze_capacity_2x2.py`（或单独脚本）输出：
   - signed gap（已有 `gap_pct`）；
   - 每 cell 的 block_size / num_gpu_blocks / mamba_page_size_padded；
   - 4 点误差全负的符号检验披露（同号概率 P=0.0625）；
   - 残差来源：fp32 state 时 attention block=2064、bf16 时 block=1072，
     页对齐/整数 block 取整导致模型高估收益 → “保守下界”。
2. 论文/research-summary 表述改为：“容量模型是保守下界（预测收益略高于实测，
   误差 −3.24%~−0.18%），来源为 vLLM 离散 block 分配；参数独立于被预测量
   （非 tautology），A_q/G 推导链见 S1”。

**耗时**：2–3 h（含写作）。

---

### P0-6：R6 — 敏感度门统计修复（无新计算）

**行动**：

1. `scripts/eval/analyze_state_sensitivity.py` 增加：
   - `--n-tests 36`、`--alpha 0.05`；
   - Bonferroni 阈值 α/36 ≈ 0.001389 与 BH-FDR 两列；
   - “决策规则预注册”写进 analyzer 头注释与
     `docs/notes/state-sensitivity-stats-2026-08-09.md`。
2. 输出披露：2/36 个 CI 不含 0（C4 L2/L8）在 Bonferroni 后均不显著；
   两者符号均为正、量级 0.0004–0.0007 PPL，远低于 seed 间标准差；
   结论仍为“噪声级 + 符号一致性观察”。

**耗时**：1–2 h。

---

## 5. P1 必做（投稿前建议，预计 1.5–2 天）

### P1-1：R7 — harness chunk 消融

入口：`hybrid_premise.py --bits 16 --state-dtype auto|bfloat16 --chunk 1|128
--seeds 42 --num-seqs 1 --max-len 2048 --corpus data/c4_slice.txt --model 2B`，
输出 `results/quality/chunk-ablation/chunk-ablation-20260809.csv`。

成功判据（预声明）：chunk=1 与 chunk=128 的 PPL 差 ≤ 0.01 PPL（或相对 ≤0.05%，
以先达到者为准）；若超出，则论文必须量化披露“chunk 级写回舍入与 per-token 语义
的偏差”。chunk=1 更接近 vLLM 每次写入即舍入的存储语义，但仍不等同 kernel 内部
逐 token 数学（需在论文写明）。

耗时：30–60 min（1 seq；若太慢降为 512 token smoke 并记录）。

---

### P1-2：R8 — RULER 非零格补 dataset seed

**行动**：

1. `ruler_prepare.py` 支持 dataset-seed 子目录（默认 seed 42 保持原路径；
   新增 `seed11/`、`seed23/` 变体，manifest 记录 random_seed + sha256）。
2. 生成 5 个格 × 2 个新 dataset seed：2B `fwe` L4096/L8192；
   9B `niah_multiquery` L4096/L8192、`fwe` L8192。
3. `ruler_quality.py` 增加 `--dataset-seed`，engine seed 固定 7；
   跑 fp16 与 fp16_statebf16 两分配 × 新 dataset seed = **20 新 cells**
   （+ 复用现有 seed-42 的 10 cells 参与分析）。
4. 新分析器 `analyze_ruler_statebf16_multi_seed.py`：每格 3 dataset seed 的
   mean 与配对 CI；报告 think 截断计数（`<think>` 未闭合）作为伪影诊断。

**成功判据**：数据 manifest 完整、sha 匹配；30 cells 全部 completed_validated；
claim 3 的 “RULER 基本持平” 在补 seed 后才可去“单 seed”限定。

**耗时**：数据生成 10–20 min；运行 2–3 h。

---

### P1-3：R9 + R13 — GSM8K 机制讨论 + fp16 state 质量 smoke

**fp16 smoke**：`hybrid_premise.py --bits 16 --state-dtype float16 --seeds 42
--num-seqs 1 --max-len 2048 --chunk 128 --corpus data/c4_slice.txt --model 2B`
（~5 min）；若 PPL 与 fp32/bf16 同量级（Δ ≤ 0.01），可选补 vLLM 侧
`fp16_statefp16` 1-seed GSM8K smoke（10–15 min，验证 kernel 路径）。

**讨论（写作）**：

- 2B vs 9B 不对称（−2.67pt vs −0.5pt）：容量冗余、attention 层补偿、规模效应、
  9B floor/ceiling（0.885→0.88 接近能力上限）四个假说逐条讨论；
- train-inference mismatch（R16）作为候选解释，标注“需微调实验才能分离”；
- bf16 下界：GDN 循环误差累积、fp8 动态范围、kernel 支持现状（MambaDType
  fp16/bf16；FlashInfer SSU PR#43518 面向 Mamba2 FP8/Int8/Int16）；
  “精度谱系”边界声明：fp16 容量已测、质量 smoke；fp8/int8 为 future work。

**耗时**：smoke 10–20 min；讨论写作 2 h。

---

### P1-4：R10 — 文献扩展（核验 + 入库）

目标条目（逐条官方元数据核验，禁编造）：

- Mamba2（Dao & Gu 2024）state 表示/精度讨论；
- GDN（Qwen3.5 混合架构）原始论文/技术报告；
- ReplaySSM 博客 + vLLM PR #47576 / #48792 / #49847；
- FlashInfer SSU checkpointing（vLLM PR #43518）独立条目；
- vLLM issue #37121（混合模型 KV 内存估计偏大）；
- 1–2 篇 SSM state 表示/压缩理论（如 Mamba state space 理论类）；
- 已有 TurboQuant/FP8、KIVI/KVQuant、DistServe 等条目复核。

产出：`docs/paper/references.bib` 增量 + prior-art 表补“为什么 state 可压缩”
数值直觉（G 固定、1 字节 KV ≈ L 字节 state）。

**耗时**：0.5–1 天。

---

### P1-5：R11 + R12 — claim whitelist 与叙事重写

- claim #5 → “ANALYZED；formal pending”；S-formal + 复现后升级；
- claim 3 → “RULER 点估计（补 seed 后去单 seed 限定）+ harness chunk 语义限定”；
- claim 1 → “现有 serving 系统工作”限定（“现有工作只优化其中一维”改为
  “现有 serving 系统未将 state 精度纳入联合预算”）；
- §1/§3.3 叙事：bf16 state = 与 KV 量化正交可复合的维度；用
  ∂C/∂A 与 ∂C/∂G 设计规则（短上下文 state 主导、长上下文 KV 主导）+ 2×2 表 +
  stacking 证据论证复合增益；删除“幅度并列”口吻。

**耗时**：0.5 天（2×2/stacking 数字就绪后回填）。

---

## 6. P2 可选（可作 future work 表述）

| 项 | 内容 | 放行 gate | 备注 |
|---|---|---|---|
| P2-1（R14） | Mamba2-2.7B @4K 容量探针（`--mamba-ssm-cache-dtype bf16`，A=0 预测 r_state≈G_fp32/G_bf16≈1.87） | 模型下载 + vLLM kernel 支持检查；失败则选 GDN-scope 声明 + A/G 参数预测其他架构 | 0.5–1 天 |
| P2-2（R15） | 替代杠杆对比段：H2O/SnapKV/PyramidKV 驱逐、KV offloading、prefix caching；r_state(L) vs r_kv(L) 交叉点 L* 解析图；TP 分片影响 | 无需新 serving；复用已有 byte-budget/E3/E4 数据 + 文献 | 0.5 天 |
| P2-3（R16） | train-inference mismatch Discussion；可选最小验证（记录 Qwen3.5 训练 dtype 证据） | 文献 + 现有 GSM8K 数据 | 0.5–1 天 |
| P2-4（S1–S3） | A_q/A_f 推导链入文；表 5.1 fp16 列 + signed error 列 + r_state(L) 曲线；PPL MDE/等价性附录 | P0-2/P0-5 产物 | 0.5 天 |
| P2-5（S4–S7） | `results/MANIFEST.json`；pilot commit 关系注记；bit-match 术语；新旧方向章节结构决策 | 无 | 半天分散 |

---

## 7. 执行顺序与时间线（单卡 5090 串行，短任务优先）

| 阶段 | 内容 | GPU 占用 | 预计 |
|---|---|---|---|
| 准备（不占 GPU） | R1 audit 文档、reasoning_bench/kv_quality_retrieval 新分配、M-2×2 runner、analyzer、serving 配置与 runner 代码 | 0 | 2–4 h |
| 阶段 A（短任务） | M-2×2 探针（~40 min）→ PPL stacking（~20 min）→ fp16 smoke（~10 min）→ GSM8K v2 2B（~60 min） | 连续 ~2.5 h | 同日 |
| 阶段 B（门禁） | S-formal MVEx 4/4 → Pilot 12/12 → 人工审阅 | ~30–45 min | 同日 |
| 阶段 C（挂机） | S-formal Random60 → ShareGPT300（`--resume` 分片） | 6–7 h | 当晚 |
| 阶段 D | 9B GSM8K v2（6 cells，~60 min）、RULER dataset seed 生成+20 cells（~2–3 h）、chunk 消融（~40 min） | 阶段 C 后串行 | 次日 |
| 阶段 E（分析） | capacity-2×2 分析、serving formal 分析、R5/R6 统计、P1-5 写作、文献核验 | 0 | 1–2 天 |
| 重审 gate | claim whitelist 更新 → research-summary/矩阵回填 → 5 席 ARS 重审 | 0 | 半天 |

---

## 8. 完成门禁与重审触发条件

重审前置（审稿原文条件）：

- R1–R4（P0 四项）完成且归档；
- R5–R6（统计修复）完成；
- R12（叙事重写）+ R13（fp16 下界）完成；
- claim whitelist 按 R11 修订；
- 论文表格数字全部可溯源到新 attempt JSON + sha256。

状态升级规则：

- S-formal 完成后：`ANALYZED`；独立复现（新 attempt + 边界/均值容差）后
  claim #5 才可 `VERIFIED`；
- RULER 补 seed 后：claim 3 去掉“单 seed”限定，但仍按点估计 + CI 表述；
- GSM8K v2 后：9B 若 3 seed 仍全同（题目子采样下不应全同），按确定性结果改写。

---

## 9. 风险与回退

| 风险 | 影响 | 回退 |
|---|---|---|
| `int4_statebf16` 组合在 vLLM 启动失败/无硬证据 | 阻塞 R3/R4 | 先跑单 cell MVEx；查 kernel/page group 兼容；必要时记录 issue 并以 fp16-KV formal + int4 容量探针作为次优证据，向审稿说明 |
| GSM8K 题目子采样后 CI 变宽、−2.67pt 不再显著 | claim 3 弱化 | 如实报告新 CI；保持“点估计回退 + 有限样本”表述；可增补 5 seeds 提高功效（S3 MDE 联动） |
| RULER dataset seed 生成时间长/与官方 generator 版本漂移 | P1-2 延迟 | 冻结 vendor commit c3f5e3b4；每数据集 sha+manifest；必要时只补 2B 两格 + 9B 两格 |
| S-formal 高负载样本失败 | 分母变化 | 按 protocol count_as_slo_miss 计入；失败样本不重跑同 attempt，新 attempt 关联 |
| 租机/驱动漂移 | 数字不可比 | 每次租机先 env_check + 跨期数值校验；headline 只留单租期一致批次 |
| 时间不足 | P2 未完成 | P2 项显式转 future work 表述，不占 claim |

---

## 10. 完成后需更新的文档

- `docs/paper/research-summary-2026-08-09.md`：实验进度、5.1/5.6、6.1、7.1、9；
- `docs/paper/claim-evidence-map-2026-08-08.md`：新增 state-bits claim 行；
- `docs/paper/experiment-matrix-plan-2026-08-08.md`：仅追加“2026-08-09 增量”指针，
  不改写已审稿内容；
- `docs/notes/results-digest-2026-08-07.md`（或新建 08-09 digest）：新 attempt 汇总；
- `docs/notes/state-compression-feasibility-2026-08-08.md`：追加 M-2×2/stacking/formal
  结果小节；
- `docs/paper/paper-mainline-2026-08-03.md`：§1/§3.3/§5/§7 叙事与表格回填。

