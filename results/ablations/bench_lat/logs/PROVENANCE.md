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

## 其余日志

- `bench_warm_*.log`：default_alloc/fp16 warmup-120 重跑（seed 42）的 server 日志
- `smoke_{bf16,fp16,int4,pl}.log` + `smoke_pl2.log`：smoke 测试
- `setup_5090.log`：环境搭建
