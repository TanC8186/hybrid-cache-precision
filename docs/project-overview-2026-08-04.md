# MLSys 项目总览（2026-08-04）

> 研究方向、实验方法、实验进度、审稿人意见的完整汇总。数据地图：`docs/notes/data-inventory-2026-08-04.md`；审稿全文：`docs/notes/mlsys-review-2026-08-04.md`；本轮 VERIFIED/PASSED 证据：`results/verified/2026-08-04/`。

---

## 一、研究方向

### 1.1 目标与基座
- **论文目标**：MLSys 顶会，LLM serving 的 **KV cache 量化/压缩**
- **基座**：vLLM fork（`vendor/vllm` git submodule，当前 commit `55f47685`）
- **模型**：Qwen3.5-2B（**18 层 Gated DeltaNet 线性注意力 + 6 层 GQA**，full attention @ L{3,7,11,15,19,23}）；Qwen3.5-9B（**24 GDN + 8 GQA**，full @ L{3,7,11,15,19,23,27,31}）—— 混合架构（linear-attention + full-attention）

### 1.2 定位演变（文献尽职调查后重构）
1. 原"量化×驱逐联合内存预算"**撞题**：QPruningKV / RDKV / ARKV / HqeKV / MiniKV 等 6+ 篇已覆盖泛化版
2. **重构后**：混合架构专属的 KV 量化系统研究 —— 混合架构的 recurrent state 不可量化且与 attention KV 共享内存池，这是纯 attention 方法忽视的结构特殊性
3. **主线（2026-08-03 定）**：**uniform int4** KV cache + 三大发现
4. **路线 A 已实现（2026-08-04）**：packed per-layer 混 dtype page group 通过运行时与容量 gate；独立复现和 serving/质量 formal 尚待完成

### 1.3 核心发现/贡献（候选）
| # | 发现 | 证据强度 |
|---|---|---|
| 1 | **uniform int4 容量**：2B 端到端 2.245x @4096 / **3.155x @16384**；9B 2.19x @4096 / **3.167x @16384**（3.88x 纯 attention 被 GDN state 摊薄）| ✅ 第一手日志，双模型复现 |
| 2 | **GDN state 摊薄机制**：18.63 MiB/request（1,085,440 B/layer × 18，temporal fp32 + conv bf16）不可量化，随并发线性占用 KV 池预算 → 把 attention 3.88x 摊薄到端到端 ~2.2x；**容量随上下文放大**（长上下文并发降 → 摊薄减弱）| ✅ 源码 + 日志 + capacity model 闭合 |
| 3 | **等字节预算排序**：sub-4bit 区"高精度+驱逐" > "低精度+全保留"（PPL 14.10 vs 21.07 @~3.3MB）| ✅ PPL 实测 |
| 4 | **逐层敏感度异构**：layer23 +28.7% vs layer3 -5.9% → 灵敏度引导分配在 PPL 侧有效 | ✅ PPL 实测 |
| 5 | **A2 packed per-layer page group**：L23 bf16 + 其余 5 个 GQA int4 合并为单一 mixed-precision group；容量 2,280,448 tokens，为旧逐层布局 **3.232x**、uniform int4 的 **0.833x** | ✅ 运行时/容量 gate PASSED；独立复现与 serving formal 待补 |

---

## 二、实验方法

### 2.1 硬件与环境
- **本地**：RTX 4060 8GB（WSL2，dev-only，禁混入 headline）
- **5090 服务器**：RTX 5090 32GB（sm_120），AutoDL，vLLM 0.26.1rc1 预编译 wheel + packed per-layer patch，venv `/root/autodl-tmp/MLSys_Research/.venv`
- **冻结代码**：A2 gate 根 commit `c7379f0`；E3 protocol-v2 根 commit `d1d52c4`；两者 vLLM commit 均为 `55f47685`
- 网络：HF 被墙 → ModelScope / hf-mirror

### 2.2 评估框架
| 评估 | 方法 | 工具 |
|---|---|---|
| **质量（离线）** | Wikitext-2，5×2048-token，bits{2,3,4,8}×keep{2048..512} + FP16，PPL；等字节预算对比 | transformers / `scripts/exp/vllm_serving_bench.py` |
| **E1 容量** | server 启动日志 `GPU KV cache size` / `Maximum concurrency`（gpu_util 0.85，max-len {4096,16384}）| `vllm serve` |
| **E2 吞吐-延迟** | `vllm bench serve --dataset-name random`（in1024/out128，400 req），Poisson，rate [1..75]，10 点 × 2 alloc，3-seed（seed 7/42/2026）| `run_bench.sh` / `bench_driver_5090.sh` |
| **E3 稳态 SLO** | 60s seed 化 Poisson 到达、warmup 120、`goodput/offered >= 0.95`；TTFT {250,500,1000,2000,3000}ms、TPOT 200ms | `run_steady_state.py` / `steady_state.py` |
| **真实 trace** | ShareGPT_Vicuna_unfiltered（94K 条，hf-mirror），rates 20--50，`ignore_eos` 配对输出长度 | `vllm bench serve --dataset-name sharegpt` |
| **复现验证** | 独立 attempt、10% 对称相对差、边界精确比较、11/11 谬误扫描 | `verify_e3_reproducibility.py` |

### 2.3 量化方案
- **uniform int4**：`kv_cache_dtype=int4_per_token_head`（per-token scale，528 B/token/layer vs fp16 2048 B）
- **legacy per-layer**：`kv_cache_dtype_per_layer`（layer23 保护）在旧统一-page 路径下容量塌缩
- **packed per-layer**：`--enable-per-layer-page-groups` 将 5×int4 + 1×bf16 attention page 打包到单一 backing storage
- **协议铁律**：3-seed mean±std/t-CI、warmup-120、commit-before-run、请求守恒、配置生效证明、失败 attempt 独立保留

---

## 三、实验进度

### 3.1 ✅ 已完成
| 实验 | 结果 |
|---|---|
| 离线 PPL 排序 | 8-bit 无损（13.63）、4-bit +1.7%、3-bit +16%、2-bit +55%；**驱逐 > 降 bit**（14.10 vs 21.07）|
| 逐层敏感度 | layer23 +28.7%、layer3 -5.9%；sens_guided PPL 14.63@4.87MB 击败均匀 3-bit 15.87@4.85MB |
| E1 容量（2B）| fp16 1,203,106 / int4 2,701,721 tokens = **2.2456x @4096**；**3.155x @16384**（4,910,731/1,556,961）|
| E1 容量（9B）| **2.19x @4096**（328,499/150,062）；**3.167x @16384**（597,271/188,650）|
| E3 protocol-v2 formal | `e3-v2-formal-d1d52c4-01`：72/72 samples，160,200/160,200 requests，0 failures；到达窗口比 0.999661--1.000427 |
| E3 VERIFIED 复现 | 48 个独立复现/上邻点 samples，97,200 requests，0 failures；80/80 cell 在 10% 内（最大 4.993%），60/60 边界精确复现，verdict `REPRODUCIBLE` |
| E3 workload 结论 | Random：250ms 无增益、500ms +4.8%、1000--3000ms +14.3%；ShareGPT：int4 23.33 vs fp16 28.33 req/s（**-17.6%**，方向反转） |
| E2（9B 单 run）| KV 预算 6.5GiB（权重 19.3GB）→ 饱和 fp16 8.3 / int4 9.3 req/s（+12%）；SLO 下都到 rate 8 |
| ShareGPT trace | int4 vs fp16：吞吐 -3%、TPOT +8%（rate 8/16，低负载）|
| **per-layer 反噬** | 真 per-layer（L23 保护）容量 ×0.258（2B 696,456 / 9B 84,787），**低于 fp16**；根因 = vLLM V1 page 统一 |
| **A2 packed gate** | runtime 8/8 checks；packed 2,280,448 tokens，旧布局 705,604，uniform 2,736,947；packed/legacy **3.232x**，packed/uniform **0.833** |
| **A2 replacement-host 复现** | `westd-02` 4/4 attempts、10/10 结构检查通过；新主机 legacy/uniform/packed 为 706,560 / 2,740,224 / 2,283,520（较原值 +0.120%--0.135%）；比例门禁复现，但绝对 token 精确门禁失败，verdict `PARTIALLY_REPRODUCIBLE`，状态仍为 `PASSED_NOT_VERIFIED` |
| **A2 protocol-v2 确认** | 全新 `westd-03` 4/4 attempts 精确重复 `westd-02`；容量差异 <=0.1353%（阈值 1%）、比例差异 <=0.0150%（阈值 0.1%），10/10 结构与 7/7 协议检查通过；runtime/capacity 子范围 `VERIFIED`，整体仍待 serving/quality |
| **GDN dtype 证据** | A2 runtime/config 固化 `mamba_ssm_cache_dtype=float32`，worker tensor 同时记录 bf16 conv state 与 fp32 temporal state |
| **bug 修复** | `--kv-cache-dtype-per-layer` 曾静默 NO-OP（arg_utils 漏传）→ 已修 + 铁律第 7 条 |
| 论文素材 | 五章草稿（Abstract/Intro/RW/Method/Evaluation）+ 3 图（uniform int4 标签）|
| 代码审查 | 0 CRITICAL/3 HIGH/4 MEDIUM 全闭环；数据地图 `data-inventory-2026-08-04.md` |
| **审稿（2026-08-04）** | 见 §四 |

### 3.2 🔄 进行中 / 待办（按优先级）
1. **[A2] serving/质量 formal**：fp16、uniform int4、packed L23-protected × Random/ShareGPT；补多 seed PPL 与 retrieval/long-context
2. **[baseline] 外部系统对比**：KIVI/KVQuant/TurboQuant 或可执行等价 baseline，同硬件/模型/SLO 协议
3. **[审稿] 统一 canonical**：表/图切换到 E3 VERIFIED 数据；删除旧 +25%；解决 PPL 三文件矛盾（13.86/11.67/11.03）
4. **[审稿] 完成手稿**：references.bib、收窄 'first' claim、9B 16384、Discussion/Limitations
5. **int2/int3 Triton 内核**：仅在上述 blocking 项完成后扩展

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

### 4.3 CRITICAL FAILURES 当前状态
1. **SLO +25% 伪影：已修复**。protocol-v2 数据已 VERIFIED；旧 +25% 不得再使用。
2. **无新方法：部分修复**。A2 packed layout 已实现并通过 gate；protocol-v2
   已将 runtime/capacity 子范围升级为 `VERIFIED`，但 serving formal、质量
   闭环与外部 baseline 仍缺。
3. **Headline 混口径：数据源已确定，文稿待改**。统一使用本轮 VERIFIED E3。
4. **GDN dtype 未记录：已修复**。A2 runtime 固化 `float32` temporal state。
5. **Quality 单 seed/PPL 矛盾：未修复**。
6. **投稿不完整：未修复**。

### 4.4 修复路径
- **E3 已闭环**：只报告 workload-specific 可持续边界，禁止 workload-general 增益 claim。
- **A2 runtime/capacity 已完成 scoped verification**：跨主机有
  0.12%--0.14% 环境漂移，protocol-v2 新 suite 已确认容量与比例容差。
  下一步是 serving/质量 formal 和外部 baseline。
- **剩余必做**：统一 3-seed canonical、PPL CI + retrieval、references.bib、9B 16384、Discussion/Limitations。

### 4.5 当前投稿就绪度
路线 A 已执行，但项目仍未达到投稿就绪：E3 blocking 已解除；A2
runtime/capacity 子范围已 `VERIFIED`，但端到端 serving、quality 与外部
baseline 仍缺；手稿完整性问题仍为 blocking。当前应继续研究，
不应将原 REJECT 评估直接改判为 accept/weak-accept。

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
| `docs/notes/next-stage-experiment-plan-2026-08-04.md` | A2/E3 冻结合同、执行结果与下一轮计划 |
| `docs/notes/serving-latency-throughput-2026-08-03.md` | E2/E3 主矩阵 |
| `docs/notes/lit-due-diligence-2026-08-02.md` | 文献调查 |
| `docs/notes/byte-budget-ordering-2026-08-02.md` | 等字节预算排序 |
| `results/ablations/bench_lat/` | 全部结果 JSON + 日志 |
| `results/verified/2026-08-04/` | A2 PASSED 与 E3 VERIFIED 的本地哈希证据 |
| `scripts/bench/` | 全部编排/分析脚本 |
| `scripts/analyze/verify_e3_reproducibility.py` | E3 SHA/分母/到达窗口/复现验证 |
| `vendor/vllm-patches/per-layer-kv-dtype.diff` | vLLM fork patch |
