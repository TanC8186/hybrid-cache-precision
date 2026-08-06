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

## 6. Protocol v2 修订（2026-08-04）

两次 formal attempt 均因 ShareGPT rate 20 中单个 HTTP 连接断开而 fail-closed：
`e3-formal-c7379f0-01` 为 1199/1200，`e3-formal-c7379f0-02` 为 1199/1200。
两份 attempt 保留并隔离，不与后续分母合并。

v2 在不改变模型、allocation、seed、到达过程、请求数和 SLO 阈值的前提下修订运行基础设施：

- 显式设置 `VLLM_HTTP_TIMEOUT_KEEP_ALIVE=75`，高于 benchmark 客户端固定的 60 秒连接复用窗口。
- 请求级失败必须满足 `completed + failed = expected`，且 detailed 数组长度、`errors` 非空条数与 `failed` 完全一致。
- 已记账的失败请求作为 SLO miss 保留在 offered denominator 中；其 `ttft=0`/`tpot=0` 占位值不得被误算为好请求。
- benchmark 非零退出、超时、结果缺失、未记账请求、字段长度不一致或错误计数不一致仍使 sample 失败。
- 协议变更后重新执行故障定向 MVEx 和 pilot；只有新 gate 通过后才能启动新的完整 formal attempt。

## 7. 执行结果与证据状态

### 7.1 A2 packed per-layer gate

- 代码状态：根仓库 `c7379f0c68a67a4eeb838573fdfe5560c1a42bd9`，vLLM
  `55f47685a553ad8d776c464c59785399a98c7185`。
- 运行时 MVEx：`a2-mvex-c7379f0-03`，所有 8 项检查通过；6 个 full-attention
  层形成一个 `UniformTypeKVCacheSpecs` 混精度 group，5 个 int4 层和 L23 bf16
  层均使用 packed layout。
- worker 证据：每个 worker 只有一个 backing storage；GDN temporal state 为
  `float32`，conv state 为 bf16；真实生成返回 16 个非空 token。
- 容量 gate 使用三个独立 probe，不合并历史数据：

| 配置 | Attempt | Capacity tokens | Max concurrency |
|---|---|---:|---:|
| 旧逐层布局 | `a2-capacity-legacy-c7379f0-01` | 705,604 | 172.27 |
| uniform int4 | `a2-capacity-uniform-c7379f0-01` | 2,736,947 | 668.20 |
| packed L23-protected | `a2-capacity-packed-c7379f0-01` | 2,280,448 | 556.75 |

最终 gate `a2_capacity_gate_c7379f0_v2.json` 为 `PASSED`：

- packed / legacy = **3.232x**，超过预设 3x；
- packed / uniform = **0.833**，落在预设 `[0.80, 0.92]`；
- 三个 probe 均 exit 0，报告 SHA 全部匹配。

旧的 gate v1 因 legacy 配置判定谓词错误被保留为失败分析，v2 通过
`a2_capacity_gate_c7379f0_v1_diagnosis.json` 显式纠正，不覆盖旧文件。A2
当前状态为工程与容量门禁 `PASSED`；在新 attempt 下独立复跑以及完成
packed serving/质量验证前，不标记为 `VERIFIED`。

#### 2026-08-05 replacement-host independent reproduction

新租用的 RTX 5090（`connect.westd.seetacloud.com:43022`）复用了原数据盘的
模型缓存和 venv，但重新部署了冻结的根代码 `c7379f0` 与 vLLM 运行时文件
`55f47685`。部署后 7 个 overlay 文件逐一 SHA 匹配，模型 `config.json` SHA
仍为 `ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4`。

- `a2-repro-suite-c7379f0-westd-01` 因冻结命令未设置
  `VLLM_ALLOW_INSECURE_SERIALIZATION=1`，在 `collective_rpc` 传输本地函数时
  失败；该 suite 保留为 `FAILED_RUNTIME_COLLECTION`，0 个有效输出，其余
  3 个 probe 未启动。
- 新 suite `a2-repro-suite-c7379f0-westd-02` 显式记录上述 transport-only
  环境变更，4/4 attempts 均 exit 0，JSON/SHA 全部匹配。
- runtime 与 packed probe 均通过 8/8 检查；真实生成返回 16 tokens；legacy
  为 24 个独立 group，uniform 为 4 个 group，packed 为
  `UniformTypeKVCacheSpecs + 2x MambaSpec`；GDN temporal state 仍为
  `float32`。

| 配置 | 原始 tokens | 新主机 tokens | 相对变化 |
|---|---:|---:|---:|
| legacy | 705,604 | 706,560 | +0.135% |
| uniform int4 | 2,736,947 | 2,740,224 | +0.120% |
| packed | 2,280,448 | 2,283,520 | +0.135% |

新主机的三个 allocation 均多获得少量整数 cache blocks，导致冻结合同要求的
“绝对 token 精确一致”失败；但 packed/legacy 为 `3.231884x`，
packed/uniform 为 `0.833333`，与原比例的对称相对差分别仅 `0.0008%` 和
`0.0150%`，原 ratio gate 全部复现。

因此冻结合同 verdict 为 `PARTIALLY_REPRODUCIBLE`；按通用 10% 环境敏感容差
为 `REPRODUCIBLE`，但不得据此回填旧合同。A2 状态保持
`PASSED_NOT_VERIFIED`。本地证据位于
`results/reproduction/2026-08-05/a2/`，审计器为
`scripts/analyze/verify_a2_reproduction.py`。

#### Protocol-v2 confirmatory reproduction

在 `westd-02` 仅作为环境敏感性发现数据、不得进入确认判定的前提下，冻结
`a2-repro-v2-suite-c7379f0-westd-03`：

- 三项容量相对原值的对称相对差不超过 1%；
- packed/legacy 与 packed/uniform 比例相对原值的对称相对差不超过
  0.1%；
- runtime/packed 结构检查、真实生成、group 语义与 GDN dtype 全部通过。

`westd-03` 4/4 attempts 均 exit 0，三项容量精确重复 `westd-02`：
706,560 / 2,740,224 / 2,283,520 tokens。容量最大对称相对差为
`0.1353%`，比例最大对称相对差为 `0.0150%`；10/10 结构检查与 7/7
protocol-v2 检查全部通过，verdict 为 `REPRODUCIBLE`。

由链接式报告将 **A2 runtime/capacity mechanism and capacity ratios**
子范围升级为 `VERIFIED`；`westd-02` 的 `ANALYZED` 报告仍保留且未覆盖。
由于 packed serving SLO 与质量评估尚未完成，A2 整体状态仍为
`PASSED_NOT_VERIFIED_SERVING_QUALITY_PENDING`。确认审计器为
`scripts/analyze/verify_a2_protocol_v2.py`。

### 7.2 E3 protocol-v2 formal 与复现

放行链完整通过：

| 阶段 | Attempt | 分母 | 失败 |
|---|---|---:|---:|
| MVEx | `e3-v2-mvex-d1d52c4-01` | 4/4 samples | 0 |
| Pilot | `e3-v2-pilot-d1d52c4-01` | 12/12 samples | 0 |
| Formal | `e3-v2-formal-d1d52c4-01` | 72/72 samples；160,200/160,200 requests | 0 |
| Random reproduction | `e3-v2-repro-random-d1d52c4-02` | 18/18；43,200/43,200 | 0 |
| ShareGPT reproduction | `e3-v2-repro-sharegpt-d1d52c4-02` | 24/24；39,600/39,600 | 0 |
| ShareGPT rate-40 upper neighbor | `e3-v2-repro-sharegpt-upper-d1d52c4-01` | 6/6；14,400/14,400 | 0 |

Formal 的到达窗口比为 `0.999661--1.000427`。复现审计逐文件计算 SHA，
核验 contract/result/analysis sidecar、请求守恒、错误计数、到达窗口和
SLO 重算：

- 80/80 个重叠 cell mean goodput 在 10% 对称相对差容差内，最大差异
  **4.993%**；
- 60/60 个 `{allocation, workload, seed, threshold}` 边界精确复现；
- ShareGPT rate 40 的 6 个样本在 5 个 TTFT 阈值下均不可持续，消除
  复现计划的上界删失；
- 11/11 统计谬误类别已扫描。

最终 verdict 为 `REPRODUCIBLE`，E3 scope 状态通过链接式报告升级为
`VERIFIED`；原始 `aggregate.json` 和 `validation.md` 仍保留
`ANALYZED`，未被覆盖。

### 7.3 可用于论文的 E3 结论边界

| Workload | TTFT threshold | FP16 boundary | INT4 boundary | Relative change |
|---|---:|---:|---:|---:|
| Random | 250 ms | 35.00 | 35.00 | 0.0% |
| Random | 500 ms | 35.00 | 36.67 | +4.8% |
| Random | 1000/2000/3000 ms | 35.00 | 40.00 | +14.3% |
| ShareGPT | 250--3000 ms | 28.33 | 23.33 | -17.6% |

因此禁止继续使用“int4 普遍提高 SLO 容量”或旧的“+25%”表述。Random
与 ShareGPT 必须分开报告；总体置信度为 `CAUTION`，原因是 `n=3`、
5 req/s 离散网格以及 ShareGPT 的宽置信区间。

### 7.4 保留但排除的失败证据

- `e3-formal-c7379f0-01` 与 `e3-formal-c7379f0-02` 始终为
  `QUARANTINED`，不得进入 protocol-v2 分母。
- `e3-v2-repro-suite-d1d52c4-01` 为 `FAILED_PRECOMPUTE`：supervisor
  未创建 slice 目录，0 sample、未创建 scientific attempt、未启动 GPU。
- 成功 suite `e3-v2-repro-suite-d1d52c4-02` 为 42/42、exit 0。
- `a2-repro-suite-c7379f0-westd-01` 为
  `FAILED_RUNTIME_COLLECTION`：模型已加载，但 `collective_rpc` transport
  配置缺失，0 个有效 JSON；不得与 `westd-02` 合并。

## 8. 下一轮实验

1. **A2 serving pilot/formal**：比较 fp16、uniform int4、packed L23-protected
   的 Random 与 ShareGPT 稳态边界，沿用 protocol v2 和独立分母。
2. **A2 质量闭环**：对 uniform int4 与 packed L23-protected 做多 seed PPL
   和 retrieval/long-context 评估，验证容量恢复没有以质量回退为代价。
3. **系统 baseline**：补齐可执行的 KIVI/KVQuant/TurboQuant 或明确可比替代，
   把 A2 与已有量化系统放在同一硬件、模型、SLO 协议下比较。
4. **论文口径统一**：删除旧 +25% SLO headline；图表改用本次 VERIFIED E3
   数据，并完成 PPL 矛盾、references、9B 长上下文和 Limitations。
