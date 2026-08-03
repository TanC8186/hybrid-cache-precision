# Serving Benchmark 汇总（2026-08-03）— KV 量化的 serving 指标

> 数据：`results/ablations/serving_bench_20260803/`，Qwen3.5-2B serving，100 req @ 4096/8192 ctx，`enforce_eager=True`，vLLM 0.26.1rc1（5090 实例）。
> 性质声明：**单点 workload（合成请求，100 req）的 serving 指标扫描，非正式 headline 扫描矩阵**。论文 headline 需待 5090 全矩阵 + 7B 验证后，由 `scripts/analyze` 产出。

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-03T13:00:00+08:00
- Verification Status: ANALYZED（统计解读完成；未做 reproducibility re-run）
- Version Label: validation_v1
- Upstream Dependencies: exp 脚本 `scripts/exp/vllm_serving_bench.py`（commit f544b0d）；离群调查见 `docs/notes/tpot-tail-latency-2026-08-03.md`

---

## 1. 数据来源

12 个 JSON（4 个 default_alloc 文件已被 warmup 版就地覆盖，无 warmup 原版已不存在）：

| 分配 | 4096（3-seed: 7/42/2026） | 8192（seed 42） |
|---|---|---|
| fp16（`kv_cache_dtype=auto`，bf16） | `bench_fp16_4096_{7,42,2026}.json` | `bench_fp16_8192_42.json` |
| uniform_int4（全层 int4_per_token_head） | `bench_uniform_int4_4096_{7,42,2026}.json` | `bench_uniform_int4_8192_42.json` |
| default_alloc（per-layer：L3-19 int4 + L23 fp16 保护，**warmup 重跑版**） | `bench_default_alloc_4096_{7,42,2026}.json` | `bench_default_alloc_8192_42.json` |

每 JSON 字段：allocation / num_reqs=100 / max_len / seed / throughput_out_tokens_per_sec / requests_per_sec / ttft_p50,p99,mean_ms / tpot_p50,p99,mean_ms / kv_cache_total_slots, blocks, block_size。合成请求按 seed 生成、请求间前缀不同（避免 prefix-cache 全命中），input:output ≈ 1:4，`--max-tokens 64`。

---

## 2. 汇总表

### 2.1 @4096，3-seed（7/42/2026）mean ± std

| 分配 | 吞吐 (out-tok/s) | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | KV slots | 容量倍数 |
|---|---|---|---|---|---|---|---|
| fp16 | **2135.3 ± 25.8** | 635.1 ± 7.7 | 1138.6 ± 17.8 | **31.3 ± 0.2** | 145.5 ± 9.7 | 1,917,056 | 1.00x |
| uniform_int4 | 1956.7 ± 48.2 | 687.3 ± 11.8 | 1186.2 ± 10.6 | 34.8 ± 1.4 | 159.6 ± 0.4 | 7,434,528 | **3.88x** |
| default_alloc | 1999.6 ± 34.3 | 741.3 ± 66.7 | 1205.9 ± 19.0 | 34.0 ± 0.8 | 124.7 ± 35.7 | 7,434,528 | **3.88x** |

相对 fp16（@4096，mean）：

| 分配 | 吞吐 | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 |
|---|---|---|---|---|---|
| default_alloc | **-6.4%**（逐 seed -5.0 ~ -8.1%） | +16.7% | +5.9% | **+8.6%**（逐 seed +6.8 ~ +11.6%） | -14.3%（方差大） |
| uniform_int4 | -8.4% | +8.2% | +4.2% | +10.9% | +9.7% |

### 2.2 @8192，seed 42（单点）

| 分配 | 吞吐 (out-tok/s) | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | KV slots |
|---|---|---|---|---|---|---|
| fp16 | 1456.0 | 1267.8 | 2278.0 | 41.0 | 155.4 | 1,917,056 |
| uniform_int4 | 1411.3 (-3.1%) | 1280.5 | 2433.9 | 43.1 | 161.5 | 7,434,528 |
| default_alloc | 1422.2 (-2.3%) | 1306.5 | 2459.4 | 42.7 | 161.5 | 7,434,528 |

@8192 的 KV 容量倍数仍为 **3.878x**（cache 按 `gpu_memory_utilization=0.9` 预分配，与 max_len 无关）。

---

## 3. 3-seed 分散度

| 分配 | 吞吐 rel_std | (max-min)/mean | TPOT p50 rel_std | TPOT p99 rel_std |
|---|---|---|---|---|
| fp16 | 1.21% | 2.4% | 0.73% | 6.7% |
| uniform_int4 | 2.47% | 4.4% | 3.99% | 0.27% |
| default_alloc | 1.71% | 3.4% | 2.37% | **28.6%**（seed42=159 vs seed2026=88） |

- 吞吐分散度低（rel_std ≤ 2.5%）：**默认分配相对 fp16 的 -6~8% 吞吐代价是稳定、可复现的方向**。
- TPOT p50 分散度低（≤ 4%）：**+8~10% 的每步开销增量真实**。
- TPOT p99 在 default_alloc 下分散度高（28.6% rel_std），且均值甚至低于 fp16 —— 说明 p99 主要受调度/尾部噪声支配，不是稳定的每步代价，不能拿单一 seed 的 p99 下结论。

---

## 4. 协议说明（warmup 差异 + 离群修复）

1. **矩阵（Agent A）无显式 `--warmup-n`**：按脚本默认 `warmup_n=5` 跑（见 `vllm_serving_bench.py`，L100）。warmup-5 只编译 3D-decode 特化（batch≤64），**batch 跨 64 的 2D-decode Triton 特化在 measured batch 内才编译** → 服务期冷编译 stall 可能落入统计窗口。
2. **default_alloc 3-seed 用 `--warmup-n 120` 重跑（Agent D）**：warmup 并发 120 > 64，触发全部 decode 特化编译，排除冷启动尖峰。`bench_default_alloc_4096_{7,42,2026}.json` 与 `bench_default_alloc_8192_42.json` **均为 warmup 版**。
3. **离群实例**：default_alloc @4096 seed42 无 warmup 时吞吐 1575.9（离群，明显低于其它 2 seed）；warmup 后回到 **1963.4**。根因与 `docs/notes/tpot-tail-latency-2026-08-03.md` 的服务期 Triton JIT 冷编译一致（`_attn_packed` / `reduce_segments`）。
4. **Protocol 选择**：headline 对比应统一 warmup 协议；fp16/uniform_int4 目前只有无 warmup（warmup_n=5）版，与 default_alloc 的 120-warmup 版**协议不完全对齐**。因分散度小（fp16 1.2%、uniform_int4 ~2.5-4.5%），可作近似；建议补 uniform_int4 的 warmup 3-seed 作为正式对照。

> **Provenance 缺口**：`vllm_serving_bench.py` 的 JSON 输出**未记录 `warmup_n`**，文件层面无法区分 warmup 5/120。无 warmup 的 seed42=1575.9 原版文件已被覆盖，仅存在于编排日志。后续需在 JSON 加 `warmup_n` 字段并归档无 warmup 版（R1 修复点）。

---

## 5. 核心结论

1. **KV 容量 3.878x**：int4（uniform / default_alloc）总 slots 7,434,528 vs fp16 1,917,056；block_size 544→2064、blocks 3524→3602。容量优势与上下文长度无关（预分配），4096 与 8192 倍数一致。
2. **吞吐代价 ~-6~8%**：default_alloc @4096 3-seed mean 比 fp16 低 6.4%（逐 seed -5.0~-8.1%）；uniform_int4 低 8.4%。@8192 单点代价更小（-2.3% / -3.1%），但仅 seed42。
3. **TPOT p50 +8-10%**：default_alloc +8.6%、uniform_int4 +10.9% —— 这是 int4 的真实每步开销（每层 2× RHT 变换 + Triton 注意力核慢于 flash-attn），"卖容量"路线需诚实报告。
4. **TPOT p99 与 fp16 相当**：default_alloc p99 mean 124.7 甚至低于 fp16（145.5），但分散度高；冷启动特例（~894ms JIT stall）已被 warmup 版排除，不再出现在本表。

**附：TTFT 观察（CAUTION）**：default_alloc TTFT p50 较 fp16 +16.7%（741 vs 635 ms），但 3-seed rel_std 9.0%（seed42=817.9），单点噪声大；若为真，可能是 int4 首 token 去量化路径或 profile 差异。n=3 不足以定论，待 warmup 统一后复测。

---

## 6. 诚实性声明 / 局限

- 单点 workload：合成请求、100 req/run、固定 input:output ≈ 1:4、单模型（2B）、单 GPU（5090）。
- 3 分配 × 2 上下文中，只有 @4096 有 3-seed；@8192 单 seed，不做 mean±std 结论。
- 协议不完全统一（warmup 5 vs 120），跨分配绝对数值对比带协议噪声；方向性结论（容量 3.88x、吞吐 -6~8%、TPOT p50 +8-10%）稳健，精确数值待补跑。
- 本 notes 所有数字由 Python 直接读 JSON 计算（mean±std），非手算；对应结果文件未移动。

---

## 7. 待补实验

1. **uniform_int4 的 warmup 3-seed**（`--warmup-n 120`）→ 与 default_alloc 同协议，正式对照；并给 fp16 补 warmup 版以消除协议偏移。
2. **更长上下文（16384）验证容量优势**：当前 100 req @ 4096/8192 不触及容量上限；16384 下 fp16 应先 OOM，int4 可继续 serve —— 这是"容量"叙事的关键实验。
3. **7B 模型验证**（最终 headline 环境）：2B 上 -6~8% 吞吐代价 + 3.88x 容量是否在 7B 复现；检查 TPOT p50 +8-10% 的每步开销是否随模型放大。
4. **p99 机制验证**：`VLLM_BATCH_INVARIANT=1` 隔离 decode 2D/3D 切换；`jit_monitor_verbose=True` 归因剩余 Triton JIT（见 tpot-tail-latency note 的验证方法）。
5. **JSON 输出加 `warmup_n` 字段** + 归档无 warmup 原版（provenance 修复）。

---

## 8. Fallacy Scan（validate mode，11/11 覆盖）

| Fallacy | Severity | 说明 | 建议 |
|---|---|---|---|
| 1. Simpson's Paradox | N/A | 无分组变量；每 cell 为单一 config | — |
| 2. Ecological Fallacy | N/A | 无聚合推断 | — |
| 3. Berkson's Paradox | N/A | 无过滤抽样 | — |
| 4. Collider Bias | N/A | 无控制变量调整 | — |
| 5. Base Rate Neglect | N/A | 无条件概率 | — |
| 6. Regression to the Mean | **NOTE** | seed42 离群 "修复"（1575.9→1963.4）归因 warmup，但重测本身可能混合 re-measure 噪声；修复机制已有日志证据，方向可信 | 用统一 warmup 协议 + ≥3 seed 复核 |
| 7. Survivorship Bias | N/A | 无 dropout | — |
| 8. Look-Elsewhere Effect | **NOTE** | 多指标（吞吐/TTFT/TPOT × p50/p99 × 2 上下文）未做多重比较校正；探索性 | headline 前固定 primary endpoint，避免 cherry-pick |
| 9. Garden of Forking Paths | **CAUTION** | 协议跨分配不一致（warmup 5 vs 120）+ 离群事后处理（丢弃无 warmup 版）+ JSON 未记 warmup_n | 统一协议、公开离群处理、记录 warmup_n |
| 10. Correlation != Causation | **CAUTION** | 分配是唯一 IV（受控对比），config 内因果归因合理；但跨 workload 泛化受限 | 措辞限定 "在此 workload 中" |
| 11. Reverse Causality | N/A | 分配先于测量 | — |

**Overall Confidence: CAUTION** —— 方向性结论稳健，精确数值受单点 workload + 协议不完全统一限制。

---

## 9. 相关文件

- 结果：`E:\MLSys_Research\results\ablations\serving_bench_20260803\`（12 个 JSON，未移动）
- 基准脚本：`E:\MLSys_Research\scripts\exp\vllm_serving_bench.py`（commit f544b0d；warmup 默认 5，JSON 未记 warmup_n）
- 离群根因调查：`E:\MLSys_Research\docs\notes\tpot-tail-latency-2026-08-03.md`
- 设计文档：`E:\MLSys_Research\docs\superpowers\specs\2026-08-01-mlsys-experiment-framework-design.md`
