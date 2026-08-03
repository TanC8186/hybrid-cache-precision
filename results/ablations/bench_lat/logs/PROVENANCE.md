# Server 启动日志 Provenance 索引（2026-08-03）

> 本目录归档 5090 服务器全部运行日志。每份日志含启动配置（`kv_cache_dtype*`、`gpu_memory_utilization`）、`GPU KV cache size`、`Maximum concurrency`、`Available KV cache memory`，是 E1/E2/E3 容量的第一手证据。
> 归档 commit：见 git log（本目录随 `bench_lat/` 一起提交）。

## E2/E3 吞吐-延迟矩阵使用的 server（gpu_util=0.85, max-len 4096, Qwen3.5-2B, RTX 5090）

| log | 分配 | 启动时刻 | KV cache (tokens) | Max concurrency | KV 内存 |
|---|---|---|---|---|---|
| `server_pl.log` | int4 per-layer (L23 fp16) | 08-03 14:31（**矩阵后补录**）| **2,701,721** | **659.60x** | 20.08 GiB |
| `server_fp16.log` | fp16 (`kv_cache_dtype=auto`) | 08-03 14:08（矩阵前启动，**第一手**）| **1,203,106** | **293.73x** | 20.12 GiB |

**容量比 = 2,701,721 / 1,203,106 = 2.2456x**（E3 与 notes 引用的数字）

> ⚠️ **Provenance 修正（code-review-2026-08-03, H1）**：int4 矩阵实际运行于 13:47–14:05，原 server（13:33 启动，pid 8790）日志被 14:31 补录 server 的 `>` 重定向**覆盖丢失**。`server_pl.log` 实为矩阵后**同配置补录**（2,701,721 / 659.60x）。同配置下 KV cache size 为确定性数值、数字可信，但严格第一手证据有损；fp16 侧 `server_fp16.log`（14:08）早于矩阵（14:16–14:29），为第一手。

int4 server 启动配置（`server_pl.log` non-default args）：
`kv_cache_dtype=int4_per_token_head, kv_cache_dtype_per_layer={23:float16, 3:float16→int4, 7,11,15,19:int4_per_token_head}, gpu_memory_utilization=0.85, max_model_len=4096, enforce_eager=False, enable_chunked_prefill=True`

> 注意：E2/E3 server 是 CUDA graph 模式（enforce_eager=False），与离线 benchmark（enforce_eager=True）的每步延迟口径不同，不可混用。

## 早期 E1 / 离线矩阵 server（11:44-11:47，KV 内存 21.94 GiB）

| log | 分配 | KV cache (tokens) | Max concurrency | KV 内存 |
|---|---|---|---|---|
| `bench_pl.log` | int4 per-layer | 2,950,758 | 720.40x | 21.94 GiB |
| `bench_fp16.log` | fp16 | 1,312,209 | 320.36x | 21.94 GiB |

- 21.94 vs 20.08 GiB 差异：不同启动时段的 CUDA graph profiling 估算与其他分配波动；早期 run 的 KV 内存更高 → 容量比 2.95M/1.31M = 2.248x（仍 ≈2.25x，稳健）。
- 离线 3.88x 矩阵（`bench_{fp16,uniform_int4,default_alloc}_*.log`）各 run 的 KV cache size 见各文件 `GPU KV cache size` 行（slot 口径 1,917,056 vs 7,434,528）。

## 补跑实验（Agent 5，14:50–15:26）

| log | 分配 | max-len | KV cache (tokens) | Max concurrency | KV 内存 | attn block_size |
|---|---|---|---|---|---|---|
| `server_fp16_16384.log` | fp16 | 16384 | 1,556,961 | 95.03x | 20.12 GiB | 544 |
| `server_int4_16384.log` | int4 per-layer | 16384 | **4,910,731** | 299.73x | 20.08 GiB | 2064 |
| `server_fp16_4096.log` | fp16（ShareGPT 对比） | 4096 | 1,203,106 | 293.73x | 20.12 GiB | 544 |
| `server_int4_4096.log` | int4 per-layer（ShareGPT 对比） | 4096 | 2,701,721 | 659.60x | 20.08 GiB | 2064 |
| `bench_warm_uniform_int4_4096_7/42/2026.log` | int4 uniform（warm-120 3-seed） | 4096 | 2,701,721 | 659.60x | 20.08 GiB | 2064 |

**容量随上下文放大（2026-08-03 关键发现）**：
- @4096：int4/fp16 = 2,701,721/1,203,106 = **2.245x**；@16384：4,910,731/1,556,961 = **3.155x**
- **机制**（日志 `interface.py:911`）：vLLM 强制 attention block_size 使 attention page size ≥ mamba page size（1,085,440 B）→ fp16 block 544 tokens / int4 block 2064 tokens（各自 ~1.09 MB page）。GDN per-seq state（18.63 MiB）与 attention KV 共占同一 KV 池预算；长上下文并发上限降低（int4 659.6→299.7）→ GDN state 总量下降 → 摊薄减弱 → int4 容量优势向纯 attention 的 3.88x 回归
- **诚实标注**：fp16 @4096→@16384 容量变化（1.203M→1.557M）涉 vLLM 内部 block 池分配细节，未逐项建模；两个 max-len 下数字均为第一手日志

## 补跑实验（Agent 5，14:50–15:26）

| log | 分配 | max-len | KV cache (tokens) | Max concurrency | KV 内存 | attn block_size |
|---|---|---|---|---|---|---|
| `server_fp16_16384.log` | fp16 | 16384 | 1,556,961 | 95.03x | 20.12 GiB | 544 |
| `server_int4_16384.log` | int4 per-layer | 16384 | **4,910,731** | 299.73x | 20.08 GiB | 2064 |
| `server_fp16_4096.log` | fp16（ShareGPT 对比） | 4096 | 1,203,106 | 293.73x | 20.12 GiB | 544 |
| `server_int4_4096.log` | int4 per-layer（ShareGPT 对比） | 4096 | 2,701,721 | 659.60x | 20.08 GiB | 2064 |
| `bench_warm_uniform_int4_4096_7/42/2026.log` | int4 uniform（warm-120 3-seed） | 4096 | 2,701,721 | 659.60x | 20.08 GiB | 2064 |

**容量随上下文放大（2026-08-03 关键发现）**：
- @4096：int4/fp16 = 2,701,721/1,203,106 = **2.245x**；@16384：4,910,731/1,556,961 = **3.155x**
- **机制**（日志 `interface.py:911`）：vLLM 强制 attention block_size 使 attention page size ≥ mamba page size（1,085,440 B）→ fp16 block 544 tokens / int4 block 2064 tokens（各自 ~1.09 MB page）。GDN per-seq state（18.63 MiB）与 attention KV 共占同一 KV 池预算；长上下文并发上限降低（int4 659.6→299.7）→ GDN state 总量下降 → 摊薄减弱 → int4 容量优势向纯 attention 的 3.88x 回归
- **诚实标注**：fp16 @4096→@16384 容量变化（1.203M→1.557M）涉 vLLM 内部 block 池分配细节，未逐项建模；两个 max-len 下数字均为第一手日志

## 修复复验 + 真正 per-layer 数据（Agent，20:44–21:15）

**背景**：`arg_utils.py` 构造 `CacheConfig` 漏传 `kv_cache_dtype_per_layer`（本地 8ef67f4 修复，diff 见 `vendor/vllm-patches/per-layer-kv-dtype.diff`）。本时段把这一行同步到服务器 site-packages（`arg_utils.py.bak` 已备份），per-layer 从此真正生效。**之前所有 per-layer 运行（包括上面 `server_pl.log` 的 2,701,721）都是 uniform int4 假象。**

**二次 bug 暴露**：per-layer dict 用 `"23":"float16"` 保护层在 bf16 模型上触发 flash-attn `query and key must have the same dtype`（RuntimeError）——bf16 query + fp16 KV 不匹配，且 flash_attn 后端无 dtype cast。正确保护 dtype = `"auto"`（跟随模型 bf16，语义=不量化该层，与 fp16 baseline 的 `kv_cache_dtype=auto` 一致）。fp16/bf16 保护层 page 均为 int4 的 4 倍，容量行为相同。

| log | 模型 | 分配 | KV cache (tokens) | Max concurrency | KV 内存 | attn block_size |
|---|---|---|---|---|---|---|
| `serve9b_int4_v2.log` | 9B | int4 + L23 `auto`（真 per-layer） | **84,787** | 20.70x | 6.47 GiB | 2048 |
| `serve2b_pl.log` | 2B | int4 + L23 `auto`（真 per-layer） | **696,456** | 170.03x | 20.08 GiB | 2064 |

- 启动日志均确认 `per-layer kv_cache_dtype override` 出现（9B L23→auto；2B L23→auto），修复生效；两 server 均完成 completion smoke。
- **对照（修复前 uniform 假象）**：9B uniform int4（`serve9b_int4.log`）= 328,499 / 80.20x；2B uniform int4（`server_pl.log`）= 2,701,721 / 659.60x。
- **容量比**：真 per-layer / uniform = 84,787/328,499 = 0.258（9B）；696,456/2,701,721 = 0.258（2B）。≈ **3.87x 下降**，且 2B 真 per-layer（696,456）已**低于 fp16 baseline（1,203,106）**。

**机制（关键发现）**：vLLM V1 的 KV cache manager 要求同池 uniform page size（`kv_cache_utils.get_uniform_page_size` 断言）。per-layer 混 dtype 触发 `unify_kv_cache_spec_page_size`：把全部分层**统一到最大 page**（bf16 层 = 4×int4 page），int4 层 block_size 由 2064 膨胀到 8256。结果（1）全部 attention 层按 bf16 page 记账（物理内存 4×）；（2）block_size 不同 → 层无法共 group → KV group 数暴增 → 每请求 block 开销倍增 → max_concurrency 大幅下降。因此**当前 fork 的 per-layer 保护在容量上是 4× 代价，比 uniform int4 差，甚至比 fp16 baseline 差**。注意：bench JSON 的 `kv_cache_total_slots`（num_blocks×`cc.block_size`）在此口径下**失真**（仍按 2064 计，未反映 4× page），容量对比应以 server 日志 `GPU KV cache size`（=max_concurrency×max_len）为准。

**2B default_alloc 真正 per-layer 3-seed bench（warm-120，offline LLM eager，gpu_util=0.90，L23=`auto`）**：
JSON 见 `results/ablations/serving_bench_20260803/bench_default_alloc_perlayer_4096_{7,42,2026}.json`。对照旧 uniform 假象 `bench_default_alloc_4096_*`（L23=`float16` NO-OP）：

| seed | 吞吐 (out-tok/s) | TTFT p50 (ms) | TPOT p50 (ms) | blocks/slots |
|---|---|---|---|---|
| 42 真 per-layer | 1745.6 | 793.9 | 40.7 | 5573 / 11,502,672 |
| 7 真 per-layer | 1898.2 | 709.7 | 37.7 | 5573 / 11,502,672 |
| 2026 真 per-layer | 1860.7 | 704.6 | 37.7 | 5573 / 11,502,672 |
| 42 旧（uniform 假象） | 1963.4 | 817.9 | 34.9 | 3602 / 7,434,528 |
| 7 旧（uniform 假象） | 2031.5 | 710.1 | 33.9 | 3602 / 7,434,528 |
| 2026 旧（uniform 假象） | 2003.8 | 695.8 | 33.3 | 3602 / 7,434,528 |

- 真 per-layer 均值：吞吐 ≈1834.8（vs 1999.6，-8.2%）；TPOT p50 ≈38.7（vs 34.0，+13.8%）；TTFT p50 基本持平（≈736 vs ≈741）。
- TPOT p99 反而大幅改善（真 per-layer 47–92 vs 旧 88–159）——旧 uniform 假象的 p99 尾巴来自 int4 内核 serving-time Triton JIT（见 b5a871d），per-layer 的 bf16 层绕开该 JIT。
- **诚实标注**：本次 per-layer dict 用 `"auto"` 代替设计稿的 `"float16"`（见上二次 bug），两者 page 尺寸/容量行为等价，但该替换使"保护层全精度"语义 = bf16 而非 fp16；如需 fp16 保护，需模型以 fp16 加载或 flash_attn 加 dtype cast。

## 9B @16384 容量（2026-08-04，gpu_util 0.85）

| log | 分配 | max-len | KV cache (tokens) | Max concurrency | KV 内存 | attn block_size |
|---|---|---|---|---|---|---|
| `serve9b_fp16_16384.log` | fp16 | 16384 | 188,650 | 11.51x | 6.51 GiB | 528 |
| `serve9b_int4_16384.log` | int4 uniform | 16384 | **597,271** | 36.45x | 6.47 GiB | 2048 |

**容量比 = 597,271 / 188,650 = 3.167x**（@16384）。对照 9B @4096 的 2.19x（328,499/150,062）——**9B 同样复现"容量随上下文放大"**（与 2B 的 2.245x→3.155x 一致），GDN 摊薄随长上下文并发降低而减弱，机制跨模型稳健。fp16 @16384 单请求（16384 tokens）可装（11.51x 并发）。

## 其余日志

- `bench_warm_*.log`：default_alloc/fp16 warmup-120 重跑（seed 42）的 server 日志
- `smoke_{bf16,fp16,int4,pl}.log` + `smoke_pl2.log`：smoke 测试
- `setup_5090.log`：环境搭建
- `dl_qwen25_7b.log`：Qwen2.5-7B-Instruct 下载日志（headline 预研，ModelScope，PID 31219）
- `exp_fill_20260803.log`：Agent 5 补跑实验编排日志（A/B/C/D 全部命令）
