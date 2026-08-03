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

## 其余日志

- `bench_warm_*.log`：default_alloc/fp16 warmup-120 重跑（seed 42）的 server 日志
- `smoke_{bf16,fp16,int4,pl}.log` + `smoke_pl2.log`：smoke 测试
- `setup_5090.log`：环境搭建
- `dl_qwen25_7b.log`：Qwen2.5-7B-Instruct 下载日志（headline 预研，ModelScope，PID 31219）
- `exp_fill_20260803.log`：Agent 5 补跑实验编排日志（A/B/C/D 全部命令）
