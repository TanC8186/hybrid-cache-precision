# TPOT p99 拖尾调查（2026-08-03，5090 实测）

> 环境：Qwen3.5-2B serving, 100 req @ 4096 ctx, `enforce_eager=True`, vLLM 0.26.1rc1。
> 现象：fp16 TPOT p50=31ms p99=154ms；per-layer int4 TPOT p50=44ms **p99=894ms**（p99/p50 ≈ 20x）。

## 结论先行

**p99 拖尾主因不是 int4 量化的每步固有成本，而是服务期 Triton JIT 冷编译 int4 注意力核**（`_attn_packed` + `reduce_segments`）触发的 ~900ms 尖峰：

1. **int4 强制换用 TRITON_ATTN 后端**（fp16 走 FLASH_ATTN，flash-attn 是预编译 CUDA 库，无 Triton JIT）。日志证据：fp16 run `Using FLASH_ATTN`，int4 run `Using TRITON_ATTN out of potential backends: ['TRITON_ATTN']`。
2. **引擎的 profile / kernel_warmup / 多模态 warmup 都不预热 Triton 注意力核** → int4 注意力核在首个 `llm.generate` 才编译。`TritonAttentionImpl.forward` 在 `attn_metadata is None` 时直接返回（profile 跳过）；`kernel_warmup` 只跑 GDN 线性注意力核 + block-table 核；FlashInfer warmup 被 TRITON_ATTN 后端门控跳过。
3. **decode 2D/3D 路径随 batch 跨阈值 64 切换**（`seq_threshold_3D = 128 // num_kv_heads = 64`）：`num_seqs ≤ 64` 走 3D 核（含 `reduce_segments`），`>64` 走 2D 核。两者是不同的 Triton 特化。warmup-5（batch=5）只编译 3D-decode；**2D-decode 特化在 measured batch 中 decode batch 首次 >64 时才编译** → ~900ms stall 落在 measured batch 内。
4. **bench 的 TPOT 是 per-request 平均 inter-token latency**：单次 ~900ms stall 命中"生成 token 少"的请求最后 1-2 个 token 时，`(last-first)/(n-1) ≈ 900ms` 被放大到 p99。数值自洽（5s run 内 p99 请求必为短生成）。

**jit_monitor 告警（决定性证据，bench_pl.log 11:47:43-46）**：
int4 run 比 fp16 run 多 `_attn_packed` 和 `reduce_segments` 两条 JIT 告警；两者共享 mrope/causal_conv1d/GDN/layernorm 4 条模型级 JIT。

## 对论文的意义

- **p50（31→44ms）和 mean（32→52ms）是 int4 的真实每步开销**（每层 2× RHT 变换 + Triton 注意力核慢于 flash-attn）——"卖容量"路线要诚实报告的权衡。
- **p99 拖尾是可消除的 warmup 缺口，不是 int4 固有代价**。论文应报告"预热干净后"的 TPOT，并注明该 warmup 缺口。

## 根因假设（排序）

| # | 假设 | 证据强度 |
|---|---|---|
| H1 | 服务期 Triton JIT 冷编译 `_attn_packed`/`reduce_segments` | **高**（jit_monitor 告警 + 后端差异 + warmup 代码路径） |
| H2 | decode 2D/3D 随 batch 跨 64 切换，2D 特化在 measured batch 编译 | 中高（代码逻辑 + reduce_segments 告警 + 并发数学） |
| H3 | 每步系统性开销（2× RHT + Triton 慢于 flash-attn）→ p50 成因 | 高（代码逐算子 + p50 31→44ms） |
| H4 | TPOT 平均指标放大单次 stall 给短生成请求 → p99 | 中（数值自洽） |

## 验证方法

1. **H1 直接验证**：`--warmup-n 120`（warmup 并发 120 > 64 → 全特化编译）。预期 p99 从 893 掉到 ~150-250ms。
2. **H2 隔离**：`VLLM_BATCH_INVARIANT=1` 重跑，预期不再有 `reduce_segments` 告警、p99 下降。
3. **H1 精确归因**：`jit_monitor_verbose=True` 枚举 `_attn_packed` 编译的特化数与首次出现批次。
4. **H3 纯净 p50**：无 JIT 环境下重测 p50，得到 int4 真实每步开销；对比 `uniform_int4` vs `default_alloc` 隔离 layer23 fp16 影响。
5. **H4 请求级明细**：改 bench 落盘每 request 的 metrics，画 TPOT 直方图/时间序列。

## 可优化方向（论文/工程）

- 服务启动后、基准前，用 dummy decode（batch >64 与 ≤64 各一次）显式预热 `_attn_packed` 与 `reduce_segments`
- `VLLM_BATCH_INVARIANT=1` 固定 2D 路径（需评估大 batch 性能损失）
- 把注意力核并入 `qwen_triton_warmup`

## 相关文件

- 服务器日志：`/root/autodl-tmp/bench_pl.log`、`/root/autodl-tmp/bench_fp16.log`
- 服务器代码：`site-packages/vllm/v1/attention/ops/int4_per_token_head.py`、`.../backends/triton_attn.py`、`.../model_executor/warmup/kernel_warmup.py`
- 本地参考：`vendor/vllm/vllm/v1/attention/ops/int4_per_token_head.py`、`vendor/vllm/vllm/v1/attention/ops/triton_unified_attention.py`
