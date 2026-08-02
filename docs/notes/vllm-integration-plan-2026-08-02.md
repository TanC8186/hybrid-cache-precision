# vLLM 集成方案（2026-08-02，基于多 Agent 架构测绘）

> 目标：在 vLLM 实现量化 KV cache（2/3/4-bit 逐层），读路径反量化回 fp16 喂现有 attention
> （lazy-dequant，Route 1）。卖容量（更长上下文/更大 batch）。

## 重大发现：vLLM 已内置 KV 量化基础设施

- `kv_cache_dtype` 已支持：fp8、fp8_per_token_head、int4_per_token_head、int8_per_token_head、
  nvfp4、turboquant_k8v4、turboquant_4bit_nc、turboquant_3bit_nc 等（vllm/config/cache.py:19-36）
- **TurboQuant 3/4-bit KV backend 已存在**（vllm/v1/attention/backends/turboquant_attn.py）
- **逐层机制已存在**：`kv_cache_dtype_skip_layers`（vllm/config/cache.py:116-118）
- 架构：`cache_dtype 字符串 → KVQuantMode → KVCacheSpec`（vllm/v1/kv_cache_interface.py）
- 写路径已"存量化"（reshape_and_cache kernel 内 CopyWithScaleOp，cache_kernels.cu:241-252）
- 读路径现有两种：kernel 内 fused dequant（FA backend）或独立 op
  `gather_and_maybe_dequant_cache`（cache_kernels.cu:1088-1195）——正是 lazy-dequant 语义

## 兼容性确认
- vLLM 要求 torch==2.13.0，WSL venv 是 torch 2.13.0+cu130 ✓
- 支持 sm_89（4060）与 sm_120（5090）✓

## RTX 5090（Blackwell / sm_120）配置（2026-08-02 确认）
| 规格 | 值 |
|---|---|
| 架构 | Blackwell, GB202, **compute capability sm_120 (12.0)** |
| CUDA cores | 21,760（170 SMs） |
| 显存 | 32GB GDDR7, 512-bit |
| **带宽** | **1,792 GB/s**（比 4060 的 ~256GB/s 高 7×） |
| L2 | 96MB |
| Tensor Core | 5th-gen, FP4/FP8 支持 |
| 接口 | PCIe 5.0 x16 |
| 发布 | 2025-01-30，CUDA 12.x/13.x 编译器支持 sm_120 |

**对集成的意义**：
- 5090 上 vLLM 应全量构建（`TORCH_CUDA_ARCH_LIST="12.0"`），避免 precompiled wheel 的 sm_120 缺失
- 高带宽 → lazy-dequant 的带宽代价小，容量收益（长上下文/大 batch）是主卖点
- FP4/FP8 Tensor Core 可用，但我们的方法是逐层 2/3/4-bit，仍需扩展
- 7B 模型 + 长上下文可跑（32GB）

## 集成计划（架构师推荐顺序）

### Phase 0（先做）：transformers 参考实现（fallback 路径）
纯 torch per-token-head 2/3/4-bit + fp16 物化 + F.scaled_dot_product_attention，
用 4060 dev + 5090 headline 验证算法。vLLM 集成只复刻同一后端做 serving 声明。
- 复用现有 harness（hybrid_premise.py）
- 加 serving 指标（TTFT/TPOT/吞吐/内存）

### Phase 1（MVP）：int4_per_token_head 跑通一个请求
- S1.1 读路径：TritonAttentionImpl.forward 加 lazy dequant（物化 fp16）
- S1.2 物化 op：仿 gather_and_maybe_dequant_cache 写 dequant-gather
- S1.3 验证：Qwen3.5-2B + --kv-cache-dtype=int4_per_token_head --enforce-eager
  + VLLM_ATTENTION_BACKEND=TRITON_ATTN（4060 无 FA3/4，用 Triton）

### Phase 2：2/3-bit + 逐层位宽
- S2.1 扩 CacheDType（int2/int3_per_token_head）+ KVQuantMode + page_size_bytes
- S2.2 写路径：triton reshape_and_cache 内核加 2/3-bit 支持
- S2.3 逐层：kv_cache_dtype_per_layer: dict[layer_idx→dtype]，6 层各成 group
- S2.4 物化 op 支持任意 bits

### Phase 3：量化方法注册 + 基准
- 注册 kv cache method + serving benchmark

## 风险
1. dtype 字符串判定（is_quantized_kv_cache 靠 endswith('per_token_head')）需同步
2. group 同构断言：逐层不同 bit 必须各成 group（Qwen3.5 6 层 = 6 group，可接受）
3. 混合精度触发 cache zeroing 开销
4. cache_dtype 参与 CUDA graph hash，新增 dtype 需加入 hash
5. lazy-dequant 物化 fp16 有带宽成本，需实测 vs kernel 内 dequant
6. 2/3-bit 需要新量化/反量化 kernel（无现成，参考 TurboQuant）

## 下一步建议
Phase 0（transformers 参考实现 + serving 指标）先行——最快验证方法，为 vLLM 集成去风险。
