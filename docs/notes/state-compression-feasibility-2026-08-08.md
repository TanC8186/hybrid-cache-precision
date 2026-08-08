# 循环状态压缩方向可行性检查（2026-08-08）

目的：评估“GDN 循环状态（temporal state）推理期压缩”作为论文原创创新点的可行性。两个检查：

1. 文献检查：推理服务期循环状态压缩是否已有人做过；
2. 服务器检查：当前 vLLM fork 是否支持 GDN state dtype 开关，能否立即做 fp32→bf16 容量/质量实验。

## 检查 1：文献结论——不是空白赛道，但“系统研究”角度未撞车

推理期状态内存优化是 2026 年的活跃方向，以下工作均已核验（标题/摘要级）：

| 工作 | 内容 | 与我们的关系 |
|---|---|---|
| ReplaySSM（Tri Dao 博客 2026-06-14；vLLM issue #47572、PR #47576/#48792/#49847） | 缓存输入 (d,k,g) 代替写回 state；标准解码 ~1.48×，推测解码 1.87–2.5×；支持 Mamba-2 与 GDN | **不能主张首次做“推理期 state 优化”**；本 fork 已内置 `use_replayssm` 配置项 |
| vLLM PR #43518（WIP） | FlashInfer checkpointing SSU，SSM state 支持 FP8/Int8/Int16 | 已有 state **量化**支持，但面向 Mamba2 且是单点 kernel/缓存能力，无系统精度-容量-质量研究 |
| vLLM PR #51052（Kimi-K3） | 混合 KDA conv+ssm 循环 state 的 MoRIIO KV 传输（1P1D 分离式） | 关注 state 的跨机传输，不涉及精度/容量预算 |
| Quamba/Quamba2/MambaQuant 等 | SSM 权重/激活 PTQ | 模型级量化，不是服务期 cache 精度预算分配 |
| vLLM issue #37121 | 混合 Mamba/Attention 模型 KV cache 内存估计偏大（Qwen3.5） | 与本项目容量建模直接相关，可作为 related work 引用 |

**结论与可主张的坑**：

- 不能写“首次提出 state 压缩/state 量化”——会被 ReplaySSM 和 PR #43518 反驳。
- 可主张的是**把 state 精度作为与 KV 精度联合分配的预算维度**做系统研究：
  - state 位数 × KV 位数 × 上下文长度的联合分配；
  - 由量化 state 驱动的端到端容量模型（把已有 `R(L)=(A_f·L+G)/(A_q·L+G)` 推广到 G 可量化）；
  - 精度-内存-质量全链路闭环（state 精度扫描 + 容量 + PPL/检索 + serving 端到端）。

## 检查 2：服务器实测——开关原生可用，fp32→bf16 立即可跑

### 代码证据

- `vllm/config/cache.py:45`：`MambaDType = Literal["auto", "float32", "float16", "bfloat16"]`。
- CLI 参数：`--mamba-ssm-cache-dtype`（`engine/arg_utils.py:1238`）。
- `model_executor/models/config.py:811`：Qwen3.5 在 auto 时读 HF 配置 `mamba_ssm_dtype`（本项目模型为 float32）；用户显式传值会覆盖并打 warning（实测日志确认）。
- `model_executor/layers/mamba/mamba_utils.py:103-106`：`temporal_state_dtype` 按 `mamba_ssm_cache_dtype` 落地；GDN 层分配 conv state 与 temporal state 均受此控制（conv 由 `mamba_cache_dtype` 控制）。

### 实测数据（Qwen3.5-2B，同一配置）

协议：`probe_ssm_state_dtype.py`，Qwen3.5-2B，`kv_cache_dtype=int4_per_token_head` + 逐层 per-layer（层 3/7/11/15/19 int4、层 23 auto），`gpu_memory_utilization=0.85`，`max_model_len=4096`，greedy 生成冒烟 16 tokens。

| 配置 | resolved mamba_ssm_cache_dtype | GDN cache dtypes | 容量 (tokens) | max concurrency | 生成冒烟 |
|---|---|---|---|---|---|
| auto（默认） | float32 | [bf16 conv, **fp32 state**] | 694,272 | 169.5× | 16 tokens 正常 |
| `--mamba-ssm-cache-dtype bfloat16` | bfloat16 | [bf16 conv, **bf16 state**] | 954,855 | 233.1× | 16 tokens 正常 |

- 容量提升：954,855 / 694,272 = **+37.5%**（并发提升同比例 +37.5%）。
- 机制：temporal state 每 4096-token block 由 fp32 1 MiB → bf16 0.5 MiB（18 个 GDN 层）；同时 mamba page size 变小（attention block 2064 → 1072）改变页面对齐，净容量收益为实测 +37.5%。
- 冒烟输出：两配置 greedy 前 16 tokens 完全一致（"KV-cache capacity matters because it directly determines the maximum…"），未出现立即质量崩坏。
- 注意：`logical_nbytes` 是共享 backing storage 上的虚拟视图字节（单层看似 5.3 GB），**不能**用作物理占用；真实总缓存分配为 ~20.02 GiB（`kv_cache_config.tensors[].size`）。

### 可行性结论

- **容量实验立即可做**：现有 serving/PPL harness 只需在启动参数里加 `--mamba-ssm-cache-dtype bfloat16`（或 LLM kwargs `mamba_ssm_cache_dtype="bfloat16"`）。
- **质量实验同样可做**：同一 harness 跑 C4/PG19 PPL 与现有 RULER/GSM8K 脚本即可；本次只做了 16-token greedy 冒烟，尚未做正式质量对比。
- 未测项：float16（MambaDType 允许但未验证 kernel）、9B 模型、ReplaySSM 与 state 压缩的组合、逐层 state 精度分配（当前开关是全局的，逐层需 fork 内扩展）。

## 补充：C4/PG19 PPL 质量对比（2B/9B × fp32/bf16，2026-08-08 完成）

在 transformers 研究 harness（`hybrid_premise.py`）中加入 `--state-dtype`：在 recurrent state 每次写入缓存边界 cast 到目标 dtype（bf16 = 模拟 vLLM `--mamba-ssm-cache-dtype bfloat16` 的存储精度；fp32 = 原协议不动）。

协议：`--bits 16 --seeds 7,42,2026 --num-seqs 5 --max-len 2048 --chunk 128`，Qwen3.5-2B/9B × C4/PG19，注意力 KV 保持 fp16（隔离 state dtype 单变量）。

| 模型 | 语料 | fp32 PPL | bf16 PPL | Δ (bf16−fp32) | 95% CI（配对） |
|---|---|---|---|---|---|
| 2B | C4 | 17.5800 | 17.5797 | −0.0003 | [−0.0016, +0.0010] |
| 2B | PG19 | 27.1783 | 27.1787 | +0.0004 | [−0.0033, +0.0041] |
| 9B | C4 | 12.7287 | 12.7289 | +0.0002 | [−0.0022, +0.0026] |
| 9B | PG19 | 18.0016 | 18.0022 | +0.0006 | [−0.0025, +0.0037] |

逐 seed 差异为 10^-4~10^-3 级（确认 bf16 路径真实生效，非空跑）；四个 cell 的 95% CI 均包含 0，即 **bf16 state 在 C4/PG19 PPL 上与 fp32 统计不可区分**。结合容量探针 +37.5%，"state 精度作为容量预算维度"的可行性得到直接支撑。

注意边界：该 harness 是 chunk 边界（128 token）一次的写回舍入，不是 vLLM kernel 内部逐 token 的 bf16 计算；论文引用时需按此措辞，正式 serving 侧数值仍建议用 vLLM 侧质量实验（RULER/GSM8K）复核。

## 证据文件

- `scripts/bench/probe_ssm_state_dtype.py`（探针脚本）
- `results/verified/2026-08-08/ssm_dtype/2b_auto.json` + `.sha256`
- `results/verified/2026-08-08/ssm_dtype/2b_bf16.json` + `.sha256`
- `scripts/exp/hybrid_premise.py`（新增 `--state-dtype`）
- `scripts/exp/run_ppl_state_dtype.sh`（8 格 runner）
- `scripts/eval/analyze_ppl_state_dtype.py`（配对 CI 分析）
- `results/quality/ppl-state-dtype/ppl-state-20260808__*`（8 格 CSV + seeds CSV）
- `results/quality/ppl-state-dtype-analysis-20260808.json`
- 文献：https://tridao.me/blog/2026/replayssm/ ；https://github.com/vllm-project/vllm/pull/43518 ；https://github.com/vllm-project/vllm/pull/51052 ；https://github.com/vllm-project/vllm/issues/37121
