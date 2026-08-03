# Serving 吞吐-延迟曲线与 SLO 分析（2026-08-03）— uniform int4 vs fp16（论文级证据）

> 数据：`results/ablations/bench_lat/{int4,fp16}/`，Qwen3.5-2B serving（RTX 5090，gpu_util=0.85，max-len 4096，`vllm serve` + `vllm bench serve --dataset-name random`，400 req/rate 点，Poisson 到达）。
> 性质声明：**E2（吞吐-延迟矩阵）+ E3（SLO 下容量）正式分析**，论文 supplement 级证据；数字全部由脚本独立重算（`/tmp/g_analysis/parse_all.py`），非手抄。
>
> ⚠️ **2026-08-03 深夜修正（主线 re-base = uniform int4）**：下文原标 "int4 per-layer（layer23 fp16 保护）" 的矩阵**实为 uniform int4**。原因：`vendor/vllm/.../engine/arg_utils.py` 构造 `CacheConfig` 漏传 `kv_cache_dtype_per_layer` → CLI 能解析但 override 永不生效（NO-OP），**所有 "per-layer" 运行静默回落到 uniform int4**。修复后（`vendor/vllm-patches/per-layer-kv-dtype.diff`，服务器 site-packages 已同步）真 per-layer 在 vLLM V1 下容量反噬（×0.258，甚至低于 fp16），详见下方「修正说明」。**本文全部 E2/E3 矩阵、SLO +25%、GDN 摊薄、16384 3.155x 数字 = uniform int4 真实值，主线有效。**

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-03
- Verification Status: VALIDATED（独立重算 20 个 JSON，与 Agent F 报告逐一核对，无出入）
- Version Label: paper_evidence_v1
- Upstream Dependencies: Agent F E2/E3 矩阵（commit `fe1f6a6`，20 个 JSON）；设计文档 `docs/notes/serving-eval-paper-level-2026-08-03.md`；单点基线 `docs/notes/serving-benchmark-2026-08-03.md`

---

## 0. 2026-08-03 深夜修正（主线 re-base：uniform int4 + 真 per-layer 容量反噬）

**背景（重大发现，已由 PROVENANCE.md 更新归档）**：

- **(a) 原 "per-layer" 数据实为 uniform int4（bug）**：`vendor/vllm/vllm/engine/arg_utils.py` 构造 `CacheConfig` 漏传 `kv_cache_dtype_per_layer`，`attention.py:319` override 永不触发 → CLI 能解析但静默无效。**本文 §2/§3 全部矩阵（含 `server_pl.log` 的 2,701,721）都是 uniform int4 假象。** 已修复（本地 `8ef67f4` + 服务器 site-packages 同步验证；patch 更新 158 行）。
- **(b) 二次 bug（保护 dtype）**：per-layer dict 用 `"float16"` 保护层在 bf16 模型必崩（flash-attn query/key dtype 不匹配）→ 正确保护 dtype = `"auto"`（跟随模型 bf16；page 尺寸与 fp16 等价）。
- **(c) 真 per-layer 容量反噬（决定性）**：修复后真 per-layer 在 vLLM V1 下容量 **×0.258**：
  - 2B：696,456 tokens（170.0x conc）vs uniform 2,701,721（659.6x）→ **×0.258**；**低于 fp16 baseline 1,203,106**
  - 9B：84,787（20.7x）vs uniform 328,499（80.2x）→ **×0.258**；**低于 fp16 150,062**
  - 机制：V1 KV manager 要求同池 uniform page size，混 dtype 触发 `unify_kv_cache_spec_page_size` 统一到最大 page（bf16=4×int4）→ int4 block_size 2064→8256 → 物理内存 4× + KV group 数暴增 → max_concurrency 暴跌。2B/9B 衰减一致 = 机制确定。
- **(d) 主线数字不变（uniform int4 全部有效）**：容量 2.245x @4096 / 3.155x @16384 / 2.19x(9B)、吞吐 -6~8%、TPOT +8-10%、SLO +25%、ShareGPT -3% —— 即本笔记 §2/§3 的数字，全部是 uniform int4 真实值，论文主线可直接使用。

**数据真实性标注**：下方 §2.1 表格标题原为 "int4 per-layer（layer23 fp16 保护）"，**实际运行的是 uniform int4**（per-layer NO-OP）。该表格数字真实、可用于 uniform int4 主线；只是标签需要改正。

**真 per-layer 3-seed 补充证据**（`results/ablations/serving_bench_20260803/bench_default_alloc_perlayer_4096_{7,42,2026}.json`，warm-120，eager，gpu_util=0.90，L23=`auto`）：

| seed | 吞吐 (out-tok/s) | TTFT p50 (ms) | TPOT p50 (ms) | TPOT p99 (ms) |
|---|---|---|---|---|
| 7 | 1898.2 | 709.7 | 37.7 | 47.3 |
| 42 | 1745.6 | 793.9 | 40.7 | 49.8 |
| 2026 | 1860.7 | 704.6 | 37.7 | 91.5 |
| **mean** | **1834.8** | ≈736 | **38.7** | 47–92 |

对照旧 uniform 假象（`bench_default_alloc_4096_*`，L23=`float16` NO-OP）：吞吐 1999.6（**-8.2%**）、TPOT p50 34.0（**+13.8%**，比 uniform int4 的 +8-10% 更差）、TTFT p50 ≈741（基本持平）。**TPOT p99 反而改善**（47–92 vs 88–159），因 bf16 保护层绕开 int4 Triton JIT（见 `tpot-tail-latency-2026-08-03.md`）。

---

## 1. 验证结论（对抗式复核，Agent F 报告 vs JSON 重算）

用 `/e/anaconda3/python`（matplotlib 3.10.0 / pandas 2.2.3 / numpy 2.1.3）重新解析 20 个 `openai-*.json`，独立重算后与 Agent F 报告核对：

| 项目 | Agent F 报告 | JSON 重算 | 核对 |
|---|---|---|---|
| int4 最大 SLO 满足率 | 50 req/s | 50 req/s（R=50 TTFT p99 1574.1 ✓；R=75 4036.7 ✗） | ✓ 一致 |
| fp16 最大 SLO 满足率 | 40 req/s | 40 req/s（R=40 TTFT p99 566.3 ✓；R=50 2081.3 ✗） | ✓ 一致 |
| int4 饱和 goodput 平台 | ~38.1 req/s | 37.46（R=50）→ 38.14（R=75） | ✓ 一致 |
| fp16 饱和 goodput 平台 | ~36.3 req/s | 35.72（R=50）→ 36.26（R=75） | ✓ 一致 |
| 容量比（实测） | 2.245x | 2,701,721 / 1,203,106 = **2.2456x** | ✓ 一致 |
| R=1 TTFT p99 | fp16 116 / int4 125 ms | fp16 115.9 / int4 124.8 ms | ✓ 一致 |
| R=50 TTFT p99 | int4 1574 / fp16 2081 ms | 1574.1 / 2081.3 ms | ✓ 一致 |

**核对结论：Agent F 报告的 E2/E3 数字全部与 JSON 原始数据一致，无任何出入。**

---

## 2. E2 完整矩阵（每分配 10 个 rate 点）

负载：`random` 数据集，input_len=1024 / output_len=128 固定（log 确认），400 请求，Poisson（burstiness=1），`--max-concurrency 512`，单 run/rate（无 3-seed）。

### 2.1 int4 uniform（实际配置；原标 "per-layer layer23 fp16 保护" 为 NO-OP bug，见 §0）

| rate (offered) | req/s (goodput) | out-tok/s | TTFT mean (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | peak conc |
|---|---|---|---|---|---|---|---|
| 1 | 0.9985 | 127.8 | 72.2 | 124.8 | 3.80 | 4.82 | 6 |
| 4 | 3.976 | 508.9 | 76.0 | 129.4 | 4.68 | 6.07 | 15 |
| 8 | 7.904 | 1011.7 | 81.6 | 130.9 | 5.50 | 7.39 | 26 |
| 12 | 11.775 | 1507.2 | 87.8 | 132.8 | 6.50 | 8.38 | 37 |
| 16 | 15.593 | 1995.9 | 95.4 | 148.0 | 7.12 | 9.31 | 49 |
| 20 | 19.368 | 2479.1 | 105.5 | 174.1 | 8.11 | 11.53 | 65 |
| 30 | 28.489 | 3646.6 | 137.9 | 205.7 | 12.33 | 16.10 | 97 |
| 40 | 35.711 | 4571.0 | 251.9 | 533.9 | 28.44 | 32.84 | 219 |
| 50 | 37.457 | 4794.5 | 667.3 | 1574.1 | 38.63 | 44.96 | 335 |
| 75 | 38.139 | 4881.8 | 1819.8 | 4036.7 | 40.97 | 45.12 | 400 |

### 2.2 fp16 baseline

| rate (offered) | req/s (goodput) | out-tok/s | TTFT mean (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | peak conc |
|---|---|---|---|---|---|---|---|
| 1 | 0.9987 | 127.8 | 70.2 | 115.9 | 3.52 | 4.71 | 6 |
| 4 | 3.977 | 509.0 | 69.9 | 119.3 | 4.41 | 5.81 | 15 |
| 8 | 7.907 | 1012.1 | 75.6 | 125.1 | 5.02 | 7.04 | 26 |
| 12 | 11.785 | 1508.4 | 78.0 | 117.2 | 5.83 | 7.79 | 37 |
| 16 | 15.616 | 1998.9 | 87.7 | 141.8 | 6.55 | 8.38 | 42 |
| 20 | 19.405 | 2483.8 | 98.7 | 173.3 | 7.19 | 9.44 | 61 |
| 30 | 28.559 | 3655.5 | 132.1 | 209.2 | 11.61 | 15.86 | 94 |
| 40 | 35.106 | 4493.6 | 252.3 | 566.3 | 27.88 | 38.57 | 236 |
| 50 | 35.723 | 4572.5 | 801.7 | 2081.3 | 41.34 | 48.97 | 346 |
| 75 | 36.258 | 4641.0 | 1966.5 | 4560.0 | 42.86 | 49.28 | 400 |

**无失败请求**（所有点 `failed=0`，`completed=400`）；R=75 时 peak conc=400 = 全部请求在途，两分配均达饱和。

---

## 3. E3 SLO 分析

SLO 定义（`configs/bench/throughput.yaml`）：**TTFT p99 < 2000 ms 且 TPOT p99 < 200 ms**。

| 分配 | 最大 SLO 满足率 | 越过 SLO 的第一个点 | TPOT p99 全程 | 结论 |
|---|---|---|---|---|
| int4 uniform（原标 per-layer，见 §0） | **50 req/s**（R=50 TTFT p99 1574 ✓） | R=75（4037 ms，2.0x） | 4.8–45.1 ms（远 < 200） | 约束由 TTFT 绑定 |
| fp16 | **40 req/s**（R=40 TTFT p99 566 ✓） | R=50（2081 ms，1.04x） | 4.7–49.3 ms（远 < 200） | 约束由 TTFT 绑定 |

- **TPOT 全程 < 200 ms**（最大 p99 = 49.3 ms @ fp16 R=75），**约束实际由 TTFT 绑定**；TPOT 项对 SLO 判定无贡献。
- int4 的 SLO 容量 = 50/40 = **+25%**（承载更多 offered load 而满足相同 SLO）。
- 越界幅度：fp16 在 R=50 即越过（1.04x），int4 到 R=75 才越过（2.02x）——int4 把 SLO 断崖推迟了一个 rate 档 + 余量更大。

---

## 4. GDN 摊薄量化证据（容量双口径）

**关键问题：为什么"纯 attention 3.88x"在 server 端只实测 2.245x？**——Qwen3.5-2B 的 18 层 Gated DeltaNet 的 recurrent state **不可量化**，摊薄了注意力 KV 的压缩收益。

### 4.1 纯 attention 口径（离线模式，3.88x）

| 分配 | B/token/layer（6 层 GQA，2 KV head × 256 head_dim） | KV slots | 倍数 |
|---|---|---|---|
| fp16 | 2 × 256 × 2B × 2(K+V) = **2048 B** | 1,917,056（离线）/ 1,203,106（server） | 1.00x |
| int4 per_token_head（含 per-token scale） | **≈ 528 B** | 7,434,528（离线） | **3.878x**（2048/528） |

- fp16 2048 B/token/layer 由架构参数直接推导（本笔记复核）；int4 ≈528 B（含 per-token scale 开销）→ 2048/528 = 3.88x，与离线 bench `serving-benchmark-2026-08-03.md` 的 3.878x 完全一致。

### 4.2 GDN state 不可量化部分（本笔记从 vLLM 代码复核）

`MambaStateShapeCalculator.gated_delta_net_state_shape`（`vendor/vllm/.../mamba/mamba_utils.py`）：
- temporal state：`(num_v_heads=16, head_v_dim=128, head_k_dim=128)` = 262,144 元素/层
- conv state：`(conv_dim=6144, conv_kernel-1=3)` = 18,432 元素/层
- dtype：temporal=fp32（`mamba_ssm_dtype=float32`），conv=bf16 → **1,048,576 + 36,864 = 1,085,440 B/layer**
- **GDN state = 1,085,440 B/layer × 18 层 = 19,537,920 B ≈ 18.63 MiB/request**，不可量化，与 Agent F 报告一致。

### 4.3 摊薄机制

- server 端 fp16 总 KV 容量 1,203,106 tokens、int4 2,701,721 tokens → **实测 2.2456x**。
- 纯 attention 3.88x 与实测 2.245x 的差距来自 GDN state 占用同一显存池：GDN state 随**并发序列数**线性增长（每请求 18.63 MiB）。**理论满并发（659.6）下 ≈ 60% KV cache 预算**（18.63 MiB × 659.6 = 12.29 GiB / 20.08 GiB）；**观测峰值并发（R=75 达 400）下 ≈ 36%**（7.45 GiB / 20.08 GiB）。该固定预算把注意力 KV 的 3.88x 摊薄到 2.245x。
- **诚实表述**：论文两个口径都报——"attention KV 压缩 3.88x"（机制层）+ "端到端 server 容量 2.245x"（系统层，含 GDN state 摊薄）。这也提示混合架构下"只量化 6 层 GQA"的收益上限受 18 层线性注意力的 per-request state 约束。

---

## 5. 关键结论

1. **SLO 容量 +25%**：int4 uniform 在 TTFT p99 < 2000 ms 下承载 50 req/s vs fp16 40 req/s（`configs/bench/throughput.yaml` 定义的 SLO）。
2. **饱和 goodput 略优**：int4 平台 ~38.1 vs fp16 ~36.3 req/s（+5%）；out-tok/s 饱和值 4882 vs 4641（+5.2%）。int4 容量优势在饱和点转化为更高承载。
3. **低负载 fp16 延迟略优（诚实报告）**：R=1 时 fp16 TTFT p99 115.9 vs int4 124.8 ms（+7.7%）、TPOT p50 3.52 vs 3.80 ms（+8.0%）——int4 每步 lazy-dequant 开销在低负载可见，与 `serving-benchmark-2026-08-03.md` 的 "TPOT p50 +8~10%"一致。
4. **高负载 TPOT 反转**：R≥50 时 int4 TPOT p50 反而低于 fp16（R=75：40.97 vs 42.86 ms），可能来自更大 cache 减少排队/抢占，但为单 run 观测，不单独下结论。
5. **TPOT 不是约束**：两分配 TPOT p99 全程 ≤ 49.3 ms << 200 ms，SLO 判定完全由 TTFT 绑定。

> **注意（主线）**：以上 1–5 的 "int4" 均指 **uniform int4**（per-layer NO-OP bug 使原矩阵即 uniform 配置，见 §0）。真 per-layer 的容量/吞吐/TPOT 均更差，见 §0(c) 与论文草稿 §8。

---

## 6. 诚实性声明 / 局限

- **主线 re-base（bug 披露）**：本文矩阵原标 "per-layer" 实为 uniform int4（`arg_utils.py` 漏传 `kv_cache_dtype_per_layer`，NO-OP）。**数字本身真实且为 uniform int4 主线**，但标签必须改正；原 per-layer 主张撤回，改以论文草稿 §8 的 limitation 框架表述。真 per-layer 容量 ×0.258、低于 fp16，另见 §0(c)。
- **模型**：Qwen3.5-2B（非 7B）。headline 需 **Qwen3.5-9B** 在 `remote_5090` 复验（Qwen3.5 家族无 7B；9B 容量探针已测 2.19x @4096）。
- **负载**：`vllm bench serve --dataset-name random` 合成负载（input_len=1024 固定 / output_len=128 固定），**非 ShareGPT 真实 trace**；变长/真实流量下的 SLO 边界可能不同。
- **单 GPU 单 run/rate**：无 3-seed mean±std，跨点对比带调度噪声；低负载 int4 延迟劣势与高负载 TPOT 反转均为单 run 观测，方向可信、精确值待补 run。
- **协议**：`num_warmups=0`（log 确认）；`disable_log_stats=False` 已由 server 端配置保证 metrics 完整（`docs/notes/serving-benchmark-2026-08-03.md` 记录的坑）。
- **容量 provenance**：server 启动日志已归档（`results/ablations/bench_lat/logs/`，commit `525020f`，含 PROVENANCE.md 索引）；注意 int4 侧 `server_pl.log` 为矩阵后同配置**补录**（原 13:33 server 日志被覆盖，见 PROVENANCE.md H1 修正）。GDN 占 KV 预算：理论满并发 659.6 下 ≈60%，观测峰值并发 400 下 ≈36%（推导见 4.3）。
- 本笔记所有矩阵数字由 `/tmp/g_analysis/parse_all.py` 直接读 JSON 重算；容量/state 数字来源已在 4 节标注推导。

---

## 7. 相关文件

- 结果：`E:\MLSys_Research\results\ablations\bench_lat\int4\`（10 JSON + rate_*.log）、`...\fp16\`（10 JSON + rate_*.log）
- 图：`E:\MLSys_Research\results\figures\fig1_latency_throughput.png`、`fig2_slo_ttft.png`、`fig3_tpot.png`（PNG dpi 150）
- 分析脚本：`/tmp/g_analysis/parse_all.py`、`/tmp/g_analysis/make_figs.py`（/e/anaconda3/python，matplotlib 3.10.0；临时目录，未入库）
- 上游：commit `fe1f6a6`（20 JSON）；设计 `docs/notes/serving-eval-paper-level-2026-08-03.md`；基线 `docs/notes/serving-benchmark-2026-08-03.md`

---

## 8. Fallacy Scan（validate mode，11/11 覆盖）

| Fallacy | Severity | 说明 | 建议 |
|---|---|---|---|
| 1. Simpson's Paradox | N/A | 无分组变量；int4/fp16 为同一批 rate 点 | — |
| 2. Ecological Fallacy | N/A | 无聚合推断（逐 rate 点报告） | — |
| 3. Berkson's Paradox | N/A | 无过滤抽样 | — |
| 4. Collider Bias | N/A | 无控制变量调整 | — |
| 5. Base Rate Neglect | N/A | 无条件概率 | — |
| 6. Regression to the Mean | **NOTE** | 单 run/rate，R=75 peak conc=400 的饱和平台可能包含排队噪声；但趋势单调 | 补 ≥3 seed 或更长窗口 |
| 7. Survivorship Bias | N/A | 无 dropout（failed=0） | — |
| 8. Look-Elsewhere Effect | **NOTE** | 多指标（TTFT/TPOT × p50/p99 × 10 rate × 2 分配）未做多重比较校正 | headline 固定 primary endpoint = TTFT p99 @ SLO |
| 9. Garden of Forking Paths | **NOTE** | 容量数字未含 server 日志存档（provenance 缺口）；rate 点 10 个为预设扫描 | 补存档 server 启动日志 |
| 10. Correlation != Causation | **CAUTION** | 分配是唯一 IV（受控对比），因果归因合理；但 2B 合成负载下结论不自动泛化 | 措辞限定 "Qwen3.5-2B / random 负载 / 5090 单卡" |
| 11. Reverse Causality | N/A | 分配先于测量 | — |

**Overall Confidence: VALIDATED（数据层）** —— 矩阵数字与 Agent F 报告完全一致；结论方向稳健，精确 headline 数值待 7B + 3-seed + 真实 trace 补齐。
