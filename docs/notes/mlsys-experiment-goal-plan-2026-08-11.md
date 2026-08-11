# MLSys 剩余实验持续目标计划

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-11
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1

## 1. 目标与当前基线

目标是在不混用失败 attempt、不静默缩小分母、不覆盖既有证据的前提下，完成
4 个必须实验包（其中 Capacity 已完成）和 4 个当前活动增强实验包，并对每个正式
结果完成完整性、统计与复现审查。第二 GPU / TP=2/4 包保持 deferred，不计入当前
完成条件。

当前基线：

- RULER no-think 5-cell 已完成并通过协议审查；
- Capacity clean R2/R3 均已完成 112/112 cell；R2-to-R3 Gate 4 比较为
  `REPRODUCIBLE`，证据状态为 `VERIFIED`；
- Capacity 验证链已封存在 Git commit `628099f`；
- Selector/controller Gate 0 实现已冻结，并在 clean revision `6ad20b4` 上通过
  32/32 focused tests、122/122 full tests、Ruff lint/format 和正式 dry-run；
- Gate 0 dry-run 只验证选择、证据哈希、部署映射和 runner plan，不替代真实
  calibration、MVEx、formal 或 Gate 4；
- 论文工作区的未暂存改动不纳入实验提交。

## 2. 全局运行合同

### 2.1 数据与磁盘

- 所有工作区、模型、下载、编译缓存、临时文件、日志和原始结果均放在
  `/root/autodl-tmp`；系统盘只保留小型配置文件和启动脚本。
- 固定使用 `/etc/profile.d/mlsys-data-disk.sh`，将 pip、Hugging Face、
  ModelScope、Torch、Triton、CUDA、vLLM 和 `TMPDIR` 指向数据盘。
- 每次正式运行前记录 `df -hT`。系统盘可用空间低于 8 GiB 或数据盘可用空间
  低于 10 GiB 时，不启动新 cell，先审计并只清理可再生成缓存或已核实重复副本。
- 模型、合同、正式 attempt、原始 JSON、sidecar、日志归档不得作为常规清理对象。

### 2.2 证据完整性

- 每个 retry 使用新 attempt ID，并记录 parent attempt、失败原因和环境差异。
- 正式矩阵在运行前冻结；不得因中途结果改变 seeds、traces、loads、SLO 或分母。
- 每个 cell 必须有原始结构化结果、SHA-256 sidecar、解析后的实际配置和退出状态。
- OOM、timeout、NaN、缺 cell、配置未生效或哈希不匹配均 fail closed；失败样本不得
  静默排除。
- pilot 与 formal 分开归档，pilot 不进入 formal 分母。
- 定量论文证据必须通过 Gate 4 后才能从 `ANALYZED` 升级为 `VERIFIED`。

### 2.3 监控与 Git

- GPU 运行默认每 30 秒检查进程、日志增长、GPU 显存和磁盘；至少每 60 秒形成
  一次可回报状态。
- 单 cell 或单一长 kernel 的日志静默只作为 advisory，不自动终止。
- 只有冻结合同中的 hard timeout、用户明确停止或即时完整性/磁盘风险可终止进程。
- 每个实验包在合同冻结、formal 完成审查、Gate 4 完成后分别建立 Git 检查点。
- 提交只包含该实验包的代码、合同、结果、审查和索引，不混入论文的并行改动。

## 3. 执行顺序与依赖

| 顺序 | 包 | 类型 | 依赖 | 完成定义 |
|---:|---|---|---|---|
| 1 | Capacity formal 独立复现 | 必须 | 已完成 formal | 112/112 + 复现比较 + Gate 4 |
| 2 | Selector/controller 端到端 | 必须 | capacity phase evidence | 策略可执行且优于/不劣于预注册 baselines |
| 3 | 机制隔离 | 必须 | serving runner | fixed-block、fixed-bytes、fixed-concurrency 全部完成 |
| 4 | 四配置 serving formal | 必须 | selector 与机制审计 | 4 配置、独立 seeds/traces、完整 load/SLO 曲线 |
| 5 | TP=2/4 | 增强 | 多 GPU 节点能力 | TP=1/2/4 同协议容量与 serving 比较 |
| 6 | 第二 hybrid/SSM 架构 | 增强 | 可用模型与内核 | 非 Qwen3.5 架构 probe + serving 子矩阵 |
| 7 | State 精度前沿 | 增强 | dtype 内核 capability gate | fp16/fp8/int8 的容量、质量、serving 前沿 |
| 8 | 真实系统 baselines | 增强 | baseline 实现与正确性测试 | 压缩、卸载、prefix caching 可执行对照 |
| 9 | 成本效率 | 增强 | serving formal + 硬件价格口径 | cost/request 与 requests/GPU-hour，含敏感性范围 |

第二 GPU 包按 2026-08-11 用户指令标为 `DEFERRED`，不属于当前持续目标的完成
条件，也不在当前资源上启动。

必须包按 1 到 4 串行放行。增强包可在不争抢同一 GPU、不改变正式合同的条件下
准备实现，但正式结果仍逐包审查和封存。

## 4. 各实验包预注册摘要

### M1. Capacity formal Gate 4 复现

- Objective: 验证 112-cell capacity phase matrix 在新 attempt、新服务器实例上的
  结构、dtype 解析、方向和容量数值可复现性。
- Parent: `capacity-phase-formal-20260811`。
- New attempt: `capacity-phase-repro-20260811`。
- Matrix: 与 parent 完全相同，2B core 72、9B core 32、float16 controls 8。
- Primary metrics: `max_num_batched_tokens`、`max_concurrency`、52 个 bf16/fp32
  配对增益方向、分组中位增益。
- Determinism class: environment-sensitive allocator benchmark。
- Gate 4 tolerance: 文件集合、配置解析、dtype 和分母必须精确一致；52/52 增益方向
  必须一致；逐 cell token capacity 对称相对差不超过 2%；分组中位增益差不超过
  2 percentage points。运行时间字段不比较。
- Working directory: `/root/autodl-tmp/MLSys_Research`。
- Frozen command:

  ```bash
  timeout --signal=TERM --kill-after=30s 21600 bash scripts/bench/run_capacity_phase_diagram.sh formal capacity-phase-repro-20260811
  ```

- Expected outputs: 112 JSON、112 SHA sidecar、contract、run log、analysis、
  reproducibility validation report。
- Success: 112/112 完成、无 silent exclusion、分析器 fail-closed 通过、上述 tolerance
  全部满足。

### M2. Executable selector/controller

- Objective: 检验 joint precision selector 是否能根据 model、context、memory 和 SLO
  约束选择可执行配置，并在真实 runner 中兑现预测。
- Treatments: full precision、KV-only、state-only、joint、selector-selected。
- Workload strata: 2B/9B、短/中/长 context、至少三个预注册 memory/SLO budgets。
- Primary metrics: feasible decision rate、prediction error、SLO attainment、goodput、
  capacity utilization；unit 是独立 workload trace/seed。
- Baselines: 固定 full precision、最佳单杠杆 oracle、预注册静态 joint 配置。
- Gate: dry-run 决策必须映射到实际 runner 参数；无效配置 fail closed；selector 不得
  读取正式结果后调参。

### M3. Mechanism isolation

- Objective: 分离容量、传输/计算带宽、page/block rounding 与 queueing 对 serving
  收益的贡献。
- Contrasts: fixed-block-count、fixed-bytes、fixed-concurrency，另加默认配置。
- Metrics: allocator/HBM bytes、blocks/pages、kernel time、prefill/decode time、queue
  time、TTFT、TPOT、goodput、SLO attainment。
- Gate: 每个 contrast 只改变目标约束，其余 workload、trace、seed、duration 一致；
  profiler 开销单独校准，不能与无 profiler 的吞吐直接混合。

### M4. Four-configuration serving formal

- Configurations: full precision、KV-only、state-only、joint。
- Design: 独立 seeds 与 traces；load 从低负载覆盖到饱和后区间；Random 与 ShareGPT
  均保留；完整 TTFT/TPOT/goodput SLO 曲线。
- Primary endpoint: paired goodput under predeclared SLO；secondary endpoints 为
  request throughput、TTFT/TPOT quantiles、capacity 和失败率。
- Statistics: trace/seed 为单位做 paired effect、95% CI 和多重比较校正；不得把
  单请求或时间采样点当作独立重复。

### E1-E5. 竞争力增强包（E1 deferred）

- E1 TP=2/4 (`DEFERRED`): 同一多卡节点上比较 TP=1/2/4；记录 state/KV 每 rank 分片、通信、
  显存和端到端 SLO。
- E2 architecture: 优先选择内核支持的 Mamba2 或另一 hybrid/SSM 模型；若 capability
  gate 失败，保留失败证据并改用预注册备选架构，不用理论推导冒充测量。
- E3 precision frontier: fp16、fp8、int8 必须是真实 kernel dtype；仅做序列化或
  理论字节换算的配置不得进入 formal。
- E4 baselines: 至少包含一种真实 KV compression/eviction、一种 offloading 和
  prefix caching；每个 baseline 先做输出正确性与配置生效检查。
- E5 cost: 以 measured goodput 和公开/固定的 GPU-hour 价格口径计算；同时报告
  requests/GPU-hour，价格敏感性不改变系统测量。

## 5. 统一统计与终审

- 所有验证覆盖 11/11 统计谬误；报告缺失样本、失败率、效应量、CI 和校正规则。
- 确定性 allocator 结果不制造伪 p 值；serving 以 trace/seed 为重复单位。
- 多模型、多长度、多 SLO 的探索结果与预注册 primary endpoint 分开标识。
- 终审核对 claim-evidence map：跨 GPU、TP、架构、精度或 baseline 的结论不得超出
  实测矩阵。
- 最终交付包含结果索引、环境快照、合同/哈希、失败 attempt 清单、验证报告和 Git
  commits；只有 4 个必须包（含已完成的 Capacity）和 4 个当前活动增强包各自达到
  其完成定义后，持续目标才可完成。第二 GPU 包保持 deferred，除非用户后续重新
  纳入范围。

## 6. 执行状态更新（2026-08-11 18:32 +08:00）

| 包 | 当前门 | 状态 | 证据 |
|---|---|---|---|
| M1 Capacity | Gate 4 | `VERIFIED` | commit `628099f`，clean R2/R3 112/112 |
| M2 Selector/controller | Gate 0 dry-run | `PASS / UNVERIFIED` | commit `6ad20b4`；attempt `joint-precision-gate0-dryrun-20260811` |
| M3 Mechanism isolation | 计划 | `NOT_STARTED` | 依赖真实 serving runner |
| M4 Four-configuration serving | 计划 | `NOT_STARTED` | 依赖 M2/M3 门禁 |
| E2 Second architecture | 计划 | `NOT_STARTED` | 单 GPU 范围 |
| E3 State precision frontier | capacity controls | `PARTIAL` | float16 control 已有；fp8/int8 capability/quality/serving 尚缺 |
| E4 Real baselines | 计划 | `NOT_STARTED` | 依赖 baseline capability/correctness gate |
| E5 Cost efficiency | 计划 | `NOT_STARTED` | 依赖 M4 measured goodput |

M2 的下一放行条件是用真实四配置 calibration 结果构建非 fixture、哈希可验证的
schema-v2 profile。`TEST_FIXTURE` profile 继续被代码强制禁止用于真实 GPU 执行。
