# MLSys 项目总览（2026-08-04）

> 研究方向、实验方法、实验进度、审稿人意见的完整汇总。数据地图：`docs/notes/data-inventory-2026-08-04.md`；审稿全文：`docs/notes/mlsys-review-2026-08-04.md`；论文素材：`docs/paper/`。

---

## 一、研究方向

### 1.1 目标与基座
- **论文目标**：MLSys 顶会，LLM serving 的 **KV cache 量化/压缩**
- **基座**：vLLM fork（`vendor/vllm` git submodule，commit e2fa285）
- **模型**：Qwen3.5-2B（**18 层 Gated DeltaNet 线性注意力 + 6 层 GQA**，full attention @ L{3,7,11,15,19,23}）；Qwen3.5-9B（**24 GDN + 8 GQA**，full @ L{3,7,11,15,19,23,27,31}）—— 混合架构（linear-attention + full-attention）

### 1.2 定位演变（文献尽职调查后重构）
1. 原"量化×驱逐联合内存预算"**撞题**：QPruningKV / RDKV / ARKV / HqeKV / MiniKV 等 6+ 篇已覆盖泛化版
2. **重构后**：混合架构专属的 KV 量化系统研究 —— 混合架构的 recurrent state 不可量化且与 attention KV 共享内存池，这是纯 attention 方法忽视的结构特殊性
3. **主线（2026-08-03 定）**：**uniform int4** KV cache + 三大发现
4. **stretch goal（设计就绪）**：per-layer 混 dtype 高效 page 管理（A2 方案）

### 1.3 核心发现/贡献（候选）
| # | 发现 | 证据强度 |
|---|---|---|
| 1 | **uniform int4 容量**：2B 端到端 2.245x @4096 / **3.155x @16384**；9B 2.19x @4096 / **3.167x @16384**（3.88x 纯 attention 被 GDN state 摊薄）| ✅ 第一手日志，双模型复现 |
| 2 | **GDN state 摊薄机制**：18.63 MiB/request（1,085,440 B/layer × 18，temporal fp32 + conv bf16）不可量化，随并发线性占用 KV 池预算 → 把 attention 3.88x 摊薄到端到端 ~2.2x；**容量随上下文放大**（长上下文并发降 → 摊薄减弱）| ✅ 源码 + 日志 + capacity model 闭合 |
| 3 | **等字节预算排序**：sub-4bit 区"高精度+驱逐" > "低精度+全保留"（PPL 14.10 vs 21.07 @~3.3MB）| ✅ PPL 实测 |
| 4 | **逐层敏感度异构**：layer23 +28.7% vs layer3 -5.9% → 灵敏度引导分配在 PPL 侧有效 | ✅ PPL 实测 |
| 5 | **Limitation**：per-layer 混 dtype 在 vLLM V1 下触发 page 统一（容量 ×0.258，低于 fp16）；A2 独立 page group 可解决（设计就绪）| ✅ 实测 + 源码定位 |

---

## 二、实验方法

### 2.1 硬件与环境
- **本地**：RTX 4060 8GB（WSL2，dev-only，禁混入 headline）
- **5090 服务器**：RTX 5090 32GB（sm_120），AutoDL，vLLM 0.26.1rc1 预编译 wheel + per-layer patch，venv `/root/autodl-tmp/MLSys_Research/.venv`
- 网络：HF 被墙 → ModelScope / hf-mirror

### 2.2 评估框架
| 评估 | 方法 | 工具 |
|---|---|---|
| **质量（离线）** | Wikitext-2，5×2048-token，bits{2,3,4,8}×keep{2048..512} + FP16，PPL；等字节预算对比 | transformers / `scripts/exp/vllm_serving_bench.py` |
| **E1 容量** | server 启动日志 `GPU KV cache size` / `Maximum concurrency`（gpu_util 0.85，max-len {4096,16384}）| `vllm serve` |
| **E2 吞吐-延迟** | `vllm bench serve --dataset-name random`（in1024/out128，400 req），Poisson，rate [1..75]，10 点 × 2 alloc，3-seed（seed 7/42/2026）| `run_bench.sh` / `bench_driver_5090.sh` |
| **E3 SLO** | TTFT p99 < 2000ms 且 TPOT p99 < 200ms（`configs/bench/throughput.yaml`）下的最大 request-rate | `mean_std_e2.py` |
| **真实 trace** | ShareGPT_Vicuna_unfiltered（94K 条，hf-mirror），rate 8/16 | `vllm bench serve --dataset-name sharegpt` |

### 2.3 量化方案
- **uniform int4**：`kv_cache_dtype=int4_per_token_head`（per-token scale，528 B/token/layer vs fp16 2048 B）
- **per-layer**（已评估，serving 反噬）：`kv_cache_dtype_per_layer`（layer23 保护），当前 vLLM V1 下 ×0.258
- **协议铁律**：3-seed mean±std、warmup-120、commit-before-run、日志验证配置真实生效（CLAUDE.md 第 7 条）

---

## 三、实验进度

### 3.1 ✅ 已完成
| 实验 | 结果 |
|---|---|
| 离线 PPL 排序 | 8-bit 无损（13.63）、4-bit +1.7%、3-bit +16%、2-bit +55%；**驱逐 > 降 bit**（14.10 vs 21.07）|
| 逐层敏感度 | layer23 +28.7%、layer3 -5.9%；sens_guided PPL 14.63@4.87MB 击败均匀 3-bit 15.87@4.85MB |
| E1 容量（2B）| fp16 1,203,106 / int4 2,701,721 tokens = **2.2456x @4096**；**3.155x @16384**（4,910,731/1,556,961）|
| E1 容量（9B）| **2.19x @4096**（328,499/150,062）；**3.167x @16384**（597,271/188,650）|
| E2/E3（2B 3-seed）| 饱和 int4 37.76 vs fp16 35.94 req/s；**SLO 下 int4 50 vs fp16 40 req/s（+25%）**；吞吐 -6~8%、TPOT p50 +8-10%；TPOT p99 拖尾 = Triton JIT 冷编译（可消除）|
| E2（9B 单 run）| KV 预算 6.5GiB（权重 19.3GB）→ 饱和 fp16 8.3 / int4 9.3 req/s（+12%）；SLO 下都到 rate 8 |
| ShareGPT trace | int4 vs fp16：吞吐 -3%、TPOT +8%（rate 8/16，低负载）|
| **per-layer 反噬** | 真 per-layer（L23 保护）容量 ×0.258（2B 696,456 / 9B 84,787），**低于 fp16**；根因 = vLLM V1 page 统一 |
| **bug 修复** | `--kv-cache-dtype-per-layer` 曾静默 NO-OP（arg_utils 漏传）→ 已修 + 铁律第 7 条 |
| 论文素材 | 五章草稿（Abstract/Intro/RW/Method/Evaluation）+ 3 图（uniform int4 标签）|
| 代码审查 | 0 CRITICAL/3 HIGH/4 MEDIUM 全闭环；数据地图 `data-inventory-2026-08-04.md` |
| **审稿（2026-08-04）** | 见 §四 |

### 3.2 🔄 进行中 / 待办（按优先级）
1. **[审稿 blocking #1] 修 E3**：steady-state 协议（可持续 SLO 率）+ threshold sweep {250..3000}ms + ShareGPT 高负载（rate 30-50）
2. **[审稿 blocking #2] 建立系统贡献（路线 A/B，需决策）**：A = 实现 A2 packed per-dtype layout（~1 周，per-layer 容量中性 → 正面贡献）；B = rescope 窄测量 claim + TurboQuant/KVQuant/KIVI baseline
3. **[审稿] 统一 3-seed canonical**：表/图全从 bench_lat3 重生成 + 误差棒；解决 PPL 三文件矛盾（13.86/11.67/11.03）
4. **[审稿] GDN dtype 确认**：从 config/log 记录 mamba_ssm_cache_dtype → PROVENANCE.md
5. **[审稿] 完成手稿**：references.bib、sharpening 'first' claim、9B 16384 补进、Discussion/Limitations 节
6. **int2/int3 Triton 内核**（完整方法 sub-4bit 早层）
7. **A2 工程实施**（若走路线 A）

---

## 四、审稿人意见（2026-08-04 最大强度对抗审稿）

### 4.1 决策与评分：**REJECT**（6/4/4/5.5）
| 审稿人 | 维度 | 评分 |
|---|---|---|
| data-audit | 数据一致性 | 6/10 |
| method | 实验方法/统计 | 4/10 |
| novelty | 新颖性/意义 | 4/10 |
| clarity | 写作/完整性 | 5.5/10 |

### 4.2 一致确认的强项
- **无造假、可复现**：全部 headline 数字（容量/SLO/per-layer/capacity model/GDN state）从第一手日志 + JSON + vLLM 源码独立复现
- 62 个 JSON 全 completed=400 failed=0；3-seed 真实；诚实披露 = 亮点

### 4.3 CRITICAL FAILURES（blocking，6 条）
1. **SLO +25% 是伪影**：SLO 边界点过载（offered > goodput）；阈值换 250/500ms 收益归零
2. **无新方法**：stock vLLM dtype，novelty 不达 MLSys bar（唯一 bespoke 是 NO-OP bug fix）
3. **Headline 混口径**：mainline 用 single-run +5.2%，headline note 用 3-seed +5.1%
4. **GDN dtype 假设未记录**（fp32 temporal state 无 config 存档）
5. **Quality 单 seed 无不确定度**（PPL 三文件矛盾，差 26%）
6. **投稿不完整**（无 references.bib、'first' claim 过强、9B 16384 缺失）

### 4.4 修复路径
- **[blocking] 修 E3** → 可持续 SLO 率 claim（诚实 ~+5% 饱和差距）
- **[blocking] 建立真系统贡献**：路线 A（实现 A2 packed layout，per-layer 变贡献）或 路线 B（rescope + baseline）
- 统一 3-seed canonical、PPL CI + retrieval eval（RULER/LongBench）、确认 GDN dtype、完成手稿（references.bib + 9B 16384 + 修数字）

### 4.5 决策点（等待用户）
**路线 A**（A2 工程，~1 周）→ per-layer 从 limitation 变贡献，novelty 达标，可能 weak-accept；**路线 B**（rescope 实证测量研究）→ 诚实完整但 novelty 弱，borderline。零成本修正（口径统一/PPL/GDN dtype/数字）两条路线都必须做。

---

## 五、关键文档索引
| 文档 | 内容 |
|---|---|
| `docs/paper/paper-mainline-2026-08-03.md` | 论文主线四章草稿 |
| `docs/paper/serving-evaluation-2026-08-03.md` | Evaluation 章节草稿 |
| `docs/notes/serving-3seed-9b-2026-08-03.md` | **headline 数据**（3-seed + 9B）|
| `docs/notes/mlsys-review-2026-08-04.md` | **审稿报告**（全文）|
| `docs/notes/data-inventory-2026-08-04.md` | 数据地图（审稿导航）|
| `docs/notes/per-layer-page-group-design-2026-08-03.md` | A2 设计（386 行）|
| `docs/notes/serving-latency-throughput-2026-08-03.md` | E2/E3 主矩阵 |
| `docs/notes/lit-due-diligence-2026-08-02.md` | 文献调查 |
| `docs/notes/byte-budget-ordering-2026-08-02.md` | 等字节预算排序 |
| `results/ablations/bench_lat/` | 全部结果 JSON + 日志 |
| `scripts/bench/` | 全部编排/分析脚本 |
| `vendor/vllm-patches/per-layer-kv-dtype.diff` | vLLM fork patch |
