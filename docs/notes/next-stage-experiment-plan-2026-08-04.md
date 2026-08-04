# 下一阶段实验补全计划（2026-08-04）

## 1. 目标与证据边界

本阶段只处理审稿中的两个 blocking 问题：

1. **A2 系统贡献**：验证 packed per-layer KV page group 能消除混合精度逐层配置的容量塌缩，并保持 Qwen3.5 混合注意力模型可正确生成。
2. **E3 方法修复**：用固定 60 秒到达窗口、120 请求预热和 `goodput / offered >= 0.95` 重测可持续 SLO，而不是把过载后的瞬时边界当成服务能力。

失败、超时、到达窗口漂移、分母不完整或配置未生效的尝试均保留，但不得进入论文定量证据。只有完成复现门禁的结果才能标记为 `VERIFIED`。

## 2. A2 实验契约

### 2.1 假设

- H-A2-1：Qwen3.5-2B 的 5 个 int4 GQA 层与 1 个 bf16 GQA 层被合并为一个 `UniformTypeKVCacheSpecs` group。
- H-A2-2：GDN/Mamba state 仍保持独立语义，但所有 cache view 共享一个 packed backing storage。
- H-A2-3：`mamba_ssm_cache_dtype` 的实际运行值为 `float32`。
- H-A2-4：L23 保护配置的容量恢复到 uniform int4 的 0.80--0.92，且相对旧 per-layer collapse 至少提升 3 倍。
- H-A2-5：至少一个真实生成请求完成且输出非空。

### 2.2 Gate 1：运行时 MVEx

入口：

```bash
python scripts/bench/inspect_kv_config.py \
  --enable-per-layer-page-groups \
  --enforce-eager \
  --generate \
  --expect-packed-per-layer \
  --output <attempt>/a2_runtime.json
```

必须归档：

- 根仓库 commit、vLLM commit、wheel overlay 前后哈希；
- 有效 cache config、Mamba shapes/dtypes；
- group、per-layer page size、tensor offset/stride；
- worker 实际 tensor shape/stride/storage pointer；
- token capacity、max concurrency 和单请求生成结果。

任一检查失败即停止，不进入 E3 pilot。

## 3. E3 稳态实验契约

### 3.1 固定因素

- 硬件：RTX 5090 32 GB；
- 模型：Qwen3.5-2B；
- `max_model_len=4096`，`gpu_memory_utilization=0.85`；
- allocations：bf16/auto 与 uniform int4；
- seeds：`{7, 42, 2026}`；
- 预热：120 requests；
- 到达窗口：60 s；
- 到达过程：vLLM seed 化 Poisson，到达时间归一化到 `num_prompts / request_rate = 60 s`；
- 客户端不设置 `max_concurrency`，防止客户端 semaphore 改写 offered arrival；
- TTFT 阈值：`{250, 500, 1000, 2000, 3000}` ms；
- TPOT 阈值：200 ms；
- 可持续判据：逐请求同时满足 TTFT/TPOT SLO 的 goodput 除以 offered rate 不低于 0.95。

### 3.2 Workload

- Random：input 1024、output 128，rates `{30,35,40,45,50}` req/s。
- ShareGPT：真实 prompt 与 completion 长度分布，强制 `ignore_eos` 以保证两种 allocation 的目标输出长度配对一致；rates `{20,25,30,35,40,45,50}` req/s，其中 30--50 回答审稿人的高负载问题。

### 3.3 分阶段放行

1. **MVEx**：int4/random/rate30/seed7，验证完整命令、详细 JSON、到达跨度、阈值重算和哈希链。
2. **Pilot**：单 seed 7，fp16/int4 × random/ShareGPT × rates `{30,40,50}`，共 12 个样本。任一 silent exclusion、到达窗口漂移或 schema 问题均阻断正式矩阵。
3. **Formal**：3 seeds，random 30--50 与 ShareGPT 20--50，共 72 个样本。按短切片执行，使用同一 attempt ID 与 `--resume` 继续未启动样本。
4. **Reproducibility**：新 attempt ID，记录 parent attempt，复跑边界附近的配对样本。正式结果与复跑在预声明容差内一致后才升级为 `VERIFIED`。

## 4. 可恢复与失败语义

- 每个 `{allocation, workload, rate, seed}` 是独立不可变 sample。
- sample 运行前写 contract；结果只在进程退出 0 且校验通过后原子发布。
- `--resume` 只跳过 `completed_validated` 样本并继续尚未启动的样本。
- 已出现 `failed` 或 `running` 残留的 sample 不在原 attempt 内重跑；重试必须新建 attempt，并通过 `--parent-attempt` 关联失败原因。
- `--max-samples` 只控制本次切片大小，不改变冻结的完整实验矩阵。

## 5. 分析与论文使用

- 单元为 seed 配对样本，不把请求当独立重复。
- 每个 cell 报 mean、sample std 和 small-sample t 95% CI。
- 对每个 TTFT 阈值单独求可持续 offered-rate 边界。
- 明示边界低于测试范围或位于最高测试点的 censoring，禁止静默删除无边界 seed。
- 报告 int4 相对 fp16 的 paired goodput difference。
- Pilot 只用于发现协议缺陷，不进入 formal denominator。
- 在独立复跑完成前，正式结果最高标记为 `ANALYZED`，不得直接写成已验证论文数字。
