# 2B E2/E3 3-seed mean±std + 9B E2 规模验证（2026-08-03）

> 数据：`results/ablations/bench_lat/bench_lat3/{fp16,int4}/`（2B，8 rate × 3 seed × 2 alloc = 48 JSON）+ `results/ablations/bench_lat/bench9b_4096/{fp16,int4}/`（9B，7 rate × 2 alloc = 14 JSON）。分析脚本 `scripts/bench/mean_std_e2.py` / `summarize_e2.py`。commit `30e78e7`。
> 性质：论文 headline 级证据（2B 3-seed mean±std）；9B 为规模验证（单 run/rate）。

## 1. 2B E2/E3 3-seed mean ± std（SLO 结论确认）

负载同 E2/E3 主矩阵（random, in1024/out128, 400 req, gpu_util 0.85, max-len 4096, server 模式）。seed {7,42,2026}。

### fp16（24 files）
| rate | req/s | out-tok/s | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) |
|---|---|---|---|---|---|
| 1 | 1.00 | 127.9 | 106.8 ± 3.7 | 3.52 | 4.63 |
| 4 | 3.98 | 509.5 | 114.4 ± 3.1 | 4.38 | 5.51 |
| 8 | 7.92 | 1013.5 | 128.5 ± 1.4 | 5.10 | 6.65 |
| 16 | 15.65 | 2003.2 | 150.0 ± 5.3 | 6.54 | 8.60 |
| 30 | 28.71 | 3675.2 | 215.7 ± 8.7 | 11.31 | 14.91 |
| 40 | 35.05 | 4486.2 | 537.5 ± 47.6 | 29.04 | 41.38 |
| **50** | 35.66 | 4564.7 | **2163.0 ± 16.8 ✗** | 40.56 | 49.11 |
| 75 | 35.94 | 4600.4 | 4680.5 ✗ | 43.44 | 49.52 |

### int4 uniform（24 files）
| rate | req/s | out-tok/s | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) |
|---|---|---|---|---|---|
| 1 | 1.00 | 127.8 | 121.7 ± 1.3 | 3.75 | 4.91 |
| 4 | 3.98 | 509.4 | 130.9 ± 5.2 | 4.65 | 6.03 |
| 8 | 7.91 | 1012.7 | 138.6 ± 2.6 | 5.59 | 7.39 |
| 16 | 15.63 | 2000.7 | 174.2 ± 9.2 | 7.29 | 9.76 |
| 30 | 28.56 | 3655.4 | 234.5 ± 7.4 | 13.38 | 17.74 |
| 40 | 35.67 | 4565.8 | 422.4 ± 46.0 | 28.45 | 37.03 |
| **50** | 37.41 | 4788.0 | **1671.5 ± 53.1 ✓** | 39.93 | 45.31 |
| 75 | 37.76 | 4833.0 | 4176.1 ✗ | 41.37 | 45.44 |

### E3 SLO 结论（TTFT p99 < 2000ms，mean）
- **int4 最大满足率 = 50 req/s**（R=50 1671.5±53 ✓；R=75 4176 ✗）
- **fp16 最大满足率 = 40 req/s**（R=40 537.5 ✓；R=50 2163 ✗）
- **int4 = +25%**（与单点 seed=0 结论一致，现为 3-seed mean±std 确认）
- 吞吐分散度极低（rel_std < 1%）；SLO 边界（40/50）TTFT p99 std ~50ms（跨 seed 稳定）
- 饱和 goodput：int4 37.8 vs fp16 35.9 req/s（+5%）；out-tok/s 4833 vs 4600（+5.1%）

## 2. 9B 规模验证（E2，单 run/rate）

**环境**：9B 权重 19.3GB bf16 → KV 预算仅 **6.5 GiB**（gpu_util 0.85），fp16 @4096 容量 150,062 tokens / 36.6x，int4 328,499 / 80.2x（容量 **2.19x**）。

| rate | fp16 req/s | fp16 out-tok/s | fp16 TTFT p99 | int4 req/s | int4 out-tok/s | int4 TTFT p99 |
|---|---|---|---|---|---|---|
| 1 | 1.00 | 127.5 | 197 | 1.00 | 127.5 | 216 |
| 4 | 3.93 | 503.3 | 342 | 3.93 | 503.2 | 328 |
| 8 | 7.70 | 986.0 | 920 | 7.70 | 985.7 | 692 |
| 16 | **8.33** | 1065.6 | 20928 | **9.30** | 1190.0 | 16015 |
| 30 | 8.29 | 1060.7 | 32793 | 9.30 | 1190.5 | 27677 |
| 50 | 8.30 | 1061.8 | 38020 | 9.31 | 1191.3 | 32965 |
| 75 | 8.29 | 1060.9 | 40689 | 9.32 | 1193.0 | 35552 |

### 9B 解读（诚实）
- **9B 单卡 serving 受 KV 预算严重限制**：rate 8→16 即撞并发上限（fp16 ~8.3 / int4 ~9.3 req/s 饱和），TTFT 秒级爆炸——这是 32GB 单卡 + 19.3GB 权重的硬件现实，**不是量化引入的问题**。
- **int4 相对优势跨规模保持**：饱和 goodput +12%（9.3 vs 8.3）、容量 2.19x、同 rate 下 TTFT 更低（rate8: 692 vs 920）。
- **E3 @9B**：TTFT p99<2000ms 下两者都到 rate 8（fp16 920 / int4 692 均 OK，rate16 爆）；int4 的 TTFT 余量更大（-25%）。
- **论文定位**：2B = 主 headline（KV 预算充足，SLO +25% 完整验证）；9B = 规模验证（机制跨 2B→9B 稳健：GDN 摊薄、int4 容量/goodput 优势均复现，但绝对 serving 能力受小 KV 池约束——适合写进 discussion 的"硬件预算 vs 模型规模"权衡）。

## 3. Material Passport
- Origin Mode: validate（2B 3-seed）/ scale-check（9B）
- Version: paper_evidence_v2（2B 从单点升级为 3-seed mean±std）
- Upstream: commit `30e78e7`；分析脚本 `scripts/bench/{mean_std_e2,summarize_e2}.py`
- Verification: 数字由脚本从 JSON 直接计算，非手抄；9B server 配置（uniform int4，无 per-layer）由启动日志确认

## 4. 局限
- 9B 单 run/rate（无 3-seed）；9B @16384 容量未测（被停止）
- 9B 的 TTFT 爆炸区间（rate 16+）只有单点，饱和平台可信但过渡区需补点
- 2B 的 8 点 × 3 seed 覆盖 SLO 边界（40/50/75），低于 16 的点（1/4/8/16/30）也有 3-seed，完整 mean±std 就绪
