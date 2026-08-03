# 论文级 Serving 评估设计（2026-08-03）

> 目标：把 serving 证据从"单点粗测"提升到 MLSys 顶会标准：真实负载 + SLO 约束 + 吞吐-延迟曲线 + 容量证据。
> 前提已解锁（2026-08-03 确认）：**vLLM server 模式 + per-layer KV dtype 工作**（`vllm serve --kv-cache-dtype int4_per_token_head --kv-cache-dtype-per-layer '{"23":"float16",...}'`，CLI 期望 JSON）。修复了 CLI 的 `help` 参数冲突 bug（patch 更新至 150 行）。

## 评估环境
- 5090（sm_120），Qwen3.5-2B，vLLM 0.26.1rc1（预编译 wheel + patch）
- OpenAI-compatible server（vllm serve）+ 压测客户端（vllm bench serve）

## 核心问题（审稿人会问）
固定 GPU 预算（gpu_memory_utilization=0.85）下，per-layer int4 KV 相比 fp16：
1. **容量**：KV cache 容量提升多少 → 支撑多少并发/多长上下文？
2. **效率**：真实负载下吞吐 × 延迟（TTFT/TPOT p50/p99）的权衡？
3. **SLO**：满足 SLO（TTFT p99 < 2000ms, TPOT p99 < 200ms，来自 configs/bench/throughput.yaml）的最大负载？

## 实验设计

### E1. 容量-并发（从启动日志 + 实测）
- 起 fp16 与 int4-per-layer 两个 server（同 gpu_util=0.85），记录 `GPU KV cache size` 与 `Maximum concurrency`（server 启动日志已有此数字）
- 对比 → 容量倍数（预期 ~3.88x）
- 补充：扫 max-model-len（8192/16384）看 fp16 何时 OOM / int4 继续（容量优势随上下文放大）

### E2. 吞吐-延迟曲线（系统论文标准）
- 负载：`vllm bench serve --dataset-name random`（变长请求：input 512-2048 / output 128-256 token）
- 扫描 request-rate（Poisson）：如 1, 2, 4, 8, 16, 32 req/s
- 分配：fp16 vs int4-per-layer
- 指标：吞吐（req/s, tok/s）× TTFT/TPOT p50/p99
- 产出：吞吐-延迟曲线（每分配一条）

### E3. SLO 下容量
- 从 E2 曲线读：TTFT p99 < 2000ms 且 TPOT p99 < 200ms 的最大 request-rate
- fp16 vs int4 对比 → "SLO 下谁承载更多负载"

## 诚实性（写入最终 notes）
- 负载用 vLLM random 数据集（变长合成），**非 ShareGPT 真实 trace**；若时间允许补 ShareGPT（HF 被墙，需 hf-mirror/ModelScope）
- 模型 2B（非 7B），headline 需 7B 复验
- 协议统一：每 server 冷启动后跑（或统一预热），记录 warmup 与 seed

## 工具链命令
```bash
# server（int4 per-layer 示例）
vllm serve <model> --kv-cache-dtype int4_per_token_head \
  --kv-cache-dtype-per-layer '{"23":"float16","3":"int4_per_token_head","7":"int4_per_token_head","11":"int4_per_token_head","15":"int4_per_token_head","19":"int4_per_token_head"}' \
  --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.85

# 压测
vllm bench serve --backend openai --base-url http://127.0.0.1:8000/v1 \
  --model <model> --dataset-name random --num-prompts 500 \
  --request-rate <rate> --result-dir <dir> --max-num-seqs <conc>
```

## 分工（teammates）
- Agent E：工具链确认（bench serve 参数）+ 跑通验证点 + E1 容量实验
- Agent F：E2 吞吐-延迟矩阵 + E3 SLO 分析（依赖 E 验证工具后）
- Agent G：最终分析 + 画图 + 归档（依赖 F）

## 相关
- 单点粗测基线：docs/notes/serving-benchmark-2026-08-03.md（3.88x 容量, -6~8% 吞吐）
- 拖尾调查：docs/notes/tpot-tail-latency-2026-08-03.md
