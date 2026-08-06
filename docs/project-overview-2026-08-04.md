# MLSys 项目总览（2026-08-06 更新）

> 研究方向、实验方法、实验进度、审稿人意见的完整汇总（最新：2026-08-06，含 A2 serving formal 108/108 结果）。
> 数据地图：`docs/notes/data-inventory-2026-08-04.md`；审稿全文：`docs/notes/mlsys-review-2026-08-04.md`；审稿后实验计划/冻结合同：`docs/notes/next-stage-experiment-plan-2026-08-04.md`；VERIFIED 证据：`results/verified/2026-08-04/`。

---

## 一、研究方向

### 1.1 目标与基座
- **论文目标**：MLSys 顶会，LLM serving 的 **KV cache 量化/压缩**
- **基座**：vLLM fork（`vendor/vllm` git submodule，vLLM commit `55f47685`）
- **模型**：Qwen3.5-2B（**18 层 Gated DeltaNet 线性注意力 + 6 层 GQA**，full attention @ L{3,7,11,15,19,23}）；Qwen3.5-9B（**24 GDN + 8 GQA**，full @ L{3,7,11,15,19,23,27,31}）—— 混合架构（linear-attention + full-attention）

### 1.2 定位演变（文献尽职调查后重构）
1. 原"量化×驱逐联合内存预算"**撞题**：QPruningKV / RDKV / ARKV / HqeKV / MiniKV 等 6+ 篇已覆盖泛化版
2. **重构后**：混合架构专属的 KV 量化系统研究 —— 混合架构的 recurrent state 不可量化且与 attention KV 共享内存池，这是纯 attention 方法忽视的结构特殊性
3. **主线**：**uniform int4** KV cache + GDN 摊薄机制 + 容量随上下文放大
4. **路线 A（2026-08-04/05 实现）**：**A2 packed per-layer page group**（混 dtype 独立 page 布局），使 per-layer 保护从容量反噬（×0.258）变为容量中性（0.833 uniform）——审稿"无新方法"critique 的系统贡献

### 1.3 核心发现/贡献（候选）
| # | 发现 | 证据强度 |
|---|---|---|
| 1 | **uniform int4 容量**：2B 端到端 2.245x @4096 / **3.155x @16384**；9B 2.19x @4096 / **3.167x @16384**（3.88x 纯 attention 被 GDN state 摊薄）| ✅ 第一手日志，双模型复现 |
| 2 | **GDN state 摊薄机制**：18.63 MiB/request（1,085,440 B/layer × 18，temporal fp32 + conv bf16，dtype 已固化确认）不可量化，随并发线性占用 KV 池预算 → 3.88x 摊薄到端到端 ~2.2x；**容量随上下文放大**（长上下文并发降 → 摊薄减弱）| ✅ 源码 + 日志 + capacity model 闭合 |
| 3 | **等字节预算排序**：sub-4bit 区"高精度+驱逐" > "低精度+全保留"（PPL 14.10 vs 21.07 @~3.3MB）| ✅ PPL 实测 |
| 4 | **逐层敏感度异构**：layer23 +28.7% vs layer3 -5.9% → 灵敏度引导分配在 PPL 侧有效 | ✅ PPL 实测 |
| 5 | **A2 packed per-layer page group**：L23 bf16 + 其余 5 个 GQA int4 合并单一 mixed-precision group；容量 2,280,448 tokens（legacy 3.232x / uniform 0.833）；**serving formal 108/108 完成**：ShareGPT 下 packed SLO 边界 ≥ uniform int4（250ms 阈值 40 vs 35）| ✅ runtime/capacity VERIFIED + serving formal 全部 completed_validated（审计 PASSED）；质量闭环与 external baseline 待补 |

---

## 二、实验方法

### 2.1 硬件与环境
- **本地**：RTX 4060 8GB（WSL2，dev-only，禁混入 headline）
- **5090 服务器**：RTX 5090 32GB（sm_120），AutoDL。**当前实例 `connect.westd.seetacloud.com:43022`（2026-08-06 起，免密已配：key `seetacloud_rtx5090`，`ssh 5090`）**；数据盘 `/root/autodl-tmp`（模型缓存 + venv + overlay 部署的实验代码）
- **冻结代码**：A2 gate 根 commit `c7379f0`；E3 protocol-v2 根 `d1d52c4`；A2 serving protocol-v3 根 `3108650`；vLLM commit 均 `55f47685`
- 网络：HF 被墙 → ModelScope / hf-mirror

### 2.2 评估框架
| 评估 | 方法 | 工具 |
|---|---|---|
| **质量（离线）** | Wikitext-2，5×2048-token，bits{2,3,4,8}×keep{2048..512} + FP16，PPL；等字节预算对比 | transformers / `scripts/exp/vllm_serving_bench.py` |
| **E1 容量** | server 启动日志 `GPU KV cache size` / `Maximum concurrency`（gpu_util 0.85，max-len {4096,16384}）| `vllm serve` |
| **E2 吞吐-延迟** | `vllm bench serve --dataset-name random`（in1024/out128，400 req），Poisson，rate [1..75]，10 点 × 2 alloc，3-seed（7/42/2026）| `run_bench.sh` / `bench_driver_5090.sh` |
| **E3 稳态 SLO** | 60s seed 化 Poisson 到达、warmup 120、`goodput/offered >= 0.95`；TTFT {250,500,1000,2000,3000}ms、TPOT 200ms | `run_steady_state.py` / `steady_state.py` |
| **A2 comparative serving (v3)** | **3 alloc（fp16 / int4 / packed_per_layer）× workloads（Random60 rates 30-50 × ShareGPT300 rates 20-50）× 3 seeds**；300s 窗口（ShareGPT）、ignore_eos 配对 | `experiments/configs/a2_comparative_piecewise_*`（3108650 冻结）|
| **真实 trace** | ShareGPT_Vicuna_unfiltered（94K 条，hf-mirror），`ignore_eos` 配对输出长度 | `vllm bench serve --dataset-name sharegpt` |
| **复现验证** | 独立 attempt、10% 对称相对差、边界精确比较、11/11 谬误扫描 | `verify_e3_reproducibility.py`、`verify_a2_*.py` |

### 2.3 量化方案
- **uniform int4**：`kv_cache_dtype=int4_per_token_head`（per-token scale，528 B/token/layer vs fp16 2048 B）
- **legacy per-layer**：`kv_cache_dtype_per_layer`（layer23 保护）在旧统一-page 路径下容量塌缩（×0.258）
- **packed per-layer（A2）**：`--enable-per-layer-page-groups` 将 5×int4 + 1×bf16 attention page 打包到单一 backing storage（复用 `_get_packed_kv_cache_layout`，DeepSeek V4 布局）
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
| E3 VERIFIED 复现 | 48 个独立复现/上邻点 samples，97,200 requests，0 failures；80/80 cell ≤10%（max 4.993%），60/60 边界精确，verdict `REPRODUCIBLE` |
| **E3 workload 结论（VERIFIED）** | Random：250ms **0% 增益**、500ms +4.8%、1000--3000ms **+14.3%**；ShareGPT：int4 23.33 vs fp16 28.33 req/s（**-17.6%**）。**旧 +25% 与"普遍提高 SLO 容量"禁止使用** |
| **A2 packed gate** | runtime 8/8 checks；packed 2,280,448 / legacy 705,604 / uniform 2,736,947 tokens；packed/legacy **3.232x**、packed/uniform **0.833** → gate `PASSED` |
| **A2 跨主机复现** | `westd-02`（+0.12--0.14% 环境漂移，`PARTIALLY_REPRODUCIBLE`）+ `westd-03`（4/4 attempts 精确重复，容量差 ≤0.1353%、比例差 ≤0.0150%，`REPRODUCIBLE`）→ **runtime/capacity 子范围 VERIFIED** |
| **A2 serving formal（protocol-v3）** | **108/108 samples completed_validated**：ShareGPT300 **63/63** + Random60 **45/45**；slice 审计全部 PASSED（`RANDOM_FORMAL_SLICE_009_PASSED` 覆盖 45、`SHAREGPT_FORMAL_SLICE_013_PASSED` 覆盖 63）|
| **GDN dtype 证据** | A2 runtime/config 固化 `mamba_ssm_cache_dtype=float32`（temporal fp32 + conv bf16）|
| per-layer 反噬（背景）| 真 per-layer（L23 保护）容量 ×0.258（2B 696,456 / 9B 84,787），低于 fp16 —— A2 修复的问题 |
| bug 修复 | `--kv-cache-dtype-per-layer` NO-OP（arg_utils 漏传）→ 已修 + 铁律第 7 条 |
| 论文素材 | 五章草稿（Abstract/Intro/RW/Method/Evaluation）+ 3 图（uniform int4 标签）|
| 审稿（2026-08-04）| REJECT（6/4/4/5.5），6 critical —— 见 §四 |

### 3.1b 🎯 A2 serving formal 结果（2026-08-06 汇总，108/108 样本 × 5 TTFT 阈值）
**最大可持续 rate**（3 seeds 全部 goodput/offered ≥ 0.95 = Y）：

**ShareGPT300**（rates 20-50）：
| TTFT 阈值 | fp16 | int4 | **packed_per_layer** |
|---|---|---|---|
| 250ms | 45 | 35 | **40** |
| 500ms | 45 | 40 | **40** |
| 1000-3000ms | 45 | 40 | **40** |

**Random60**（rates 30-50）：
| TTFT 阈值 | fp16 | int4 | **packed_per_layer** |
|---|---|---|---|
| 250ms | 30 | NONE | 30 |
| 500ms | 35 | 35 | 35 |
| 1000ms | 35 | 35 | 35 |
| 2000-3000ms | 35 | **40** | **40** |

**解读**：
1. **A2 packed 核心价值验证**：ShareGPT 下 packed **始终 ≥ int4**（250ms 阈值 40 vs 35）——L23 保护的**质量收益转化为 SLO 边界优势**；Random 下与 int4 持平（宽松阈值 +14.3% vs fp16 复现）。容量恢复到 uniform 0.833 的同时 serving 边界不降反升
2. **诚实披露**：ShareGPT 下 fp16 仍最高（45）——量化 TPOT 开销（+8%）在真实流量下压低边界（与 E3 v2 方向一致）；Random 250ms 严格阈值下 int4 甚至无 3-seed 全可持续点
3. **论文口径**：Random 与 ShareGPT 必须分开报告；A2 的价值主张 = 容量恢复（3.232x legacy）+ ShareGPT 下优于 uniform int4 的 SLO 边界

### 3.2 🔄 进行中 / 待办（按优先级）
1. **[A2] 质量闭环**：packed vs uniform 多 seed PPL + retrieval/long-context（验证容量恢复未以质量回退为代价）
2. **[baseline] 外部系统对比**：KIVI/KVQuant/TurboQuant 或可执行等价 baseline，同硬件/模型/SLO 协议
3. **[审稿] 统一 canonical**：表/图切换到 VERIFIED E3 + A2 serving formal 数据；删除旧 +25%；解决 PPL 三文件矛盾（13.86/11.67/11.03）
4. **[审稿] 完成手稿**：references.bib、收窄 'first' claim、9B 16384 补进、Discussion/Limitations、A2 系统贡献章节
5. **int2/int3 Triton 内核**：仅在上述 blocking 项完成后扩展

---

## 四、审稿人意见（2026-08-04 最大强度对抗审稿）

### 4.1 决策与评分：**REJECT**（6/4/4/5.5）
data-audit 6/10 · method 4/10 · novelty 4/10 · clarity 5.5/10

### 4.2 一致确认的强项
- 无造假、可复现：全部 headline 数字从第一手日志 + JSON + vLLM 源码独立复现；诚实披露 = 亮点

### 4.3 CRITICAL FAILURES 当前状态
1. **SLO +25% 伪影：已修复**。protocol-v2 VERIFIED；**A2 serving formal（108/108）进一步给出 3-alloc 边界**（Random +14.3% / ShareGPT 方向反转均分 workload 报告）
2. **无新方法：已实质修复**。A2 packed per-layer page group 实现 + gate PASSED + 跨主机 REPRODUCIBLE + **serving formal 108/108 完成**（packed 在 ShareGPT 下 SLO 边界 ≥ uniform int4）；剩余：质量闭环 + external baseline
3. **Headline 混口径：数据源已确定，文稿待改**（统一 VERIFIED E3 + A2 formal）
4. **GDN dtype 未记录：已修复**（A2 runtime 固化 float32）
5. **Quality 单 seed/PPL 矛盾：未修复**
6. **投稿不完整：未修复**

### 4.4 修复路径
- **E3/A2 serving 已闭环**：只报告 workload-specific 可持续边界；Random/ShareGPT 分开
- **A2 runtime/capacity VERIFIED + serving formal 完成**：下一步质量闭环 + external baseline
- **剩余必做**：3-seed canonical 统一、PPL CI + retrieval、references.bib、9B 16384、Discussion/Limitations

### 4.5 当前投稿就绪度
**接近可投但未就绪**：E3 blocking 已解除；A2 系统贡献的 runtime/capacity/serving 证据链完整（gate → 跨主机复现 → 108/108 serving formal → SLO 边界优于 uniform）；仍缺 quality 闭环、external baseline、手稿完整性（PPL 矛盾、references、A2 章节）。完成质量闭环 + 手稿后可按 weak-accept 方向投稿。

---

## 五、关键文档索引
| 文档 | 内容 |
|---|---|
| `docs/paper/paper-mainline-2026-08-03.md` | 论文主线四章草稿 |
| `docs/paper/serving-evaluation-2026-08-03.md` | Evaluation 章节草稿 |
| `docs/notes/next-stage-experiment-plan-2026-08-04.md` | **A2/E3 冻结合同、执行结果、下一轮计划**（A2 serving formal 契约）|
| `docs/notes/mlsys-review-2026-08-04.md` | 审稿报告（全文）|
| `docs/notes/data-inventory-2026-08-04.md` | 数据地图（审稿导航）|
| `docs/notes/serving-3seed-9b-2026-08-03.md` | 2B 3-seed + 9B 数据 |
| `docs/notes/per-layer-page-group-design-2026-08-03.md` | A2 设计（386 行）|
| `results/verified/2026-08-04/` | A2 gate PASSED + E3 VERIFIED 哈希证据 |
| `results/reproduction/2026-08-05/` | 跨主机复现 + A2 serving formal slice 审计报告 |
| `scripts/analyze/verify_a2_*.py` | A2 gate/复现/serving formal 审计器（含 protocol-v3）|
| `scripts/bench/` | serving 编排/分析脚本 |
| `vendor/vllm-patches/per-layer-kv-dtype.diff` | vLLM fork patch |
