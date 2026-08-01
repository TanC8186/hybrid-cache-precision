# MLSys 实验框架设计文档

- **日期**: 2026-08-01
- **状态**: 已批准（用户认可修订后框架）
- **项目**: `E:\MLSys_Research`

## 1. 背景与目标

目标是发表一篇 **MLSys 顶会论文**，研究方向为 **LLM serving 的 KV cache 量化/压缩**，基于 vLLM 实现。本框架的目标是支撑一篇**可复现、能过 MLSys 审稿**的论文所需的全部实验证据。

研究者的画像：ML 训练/模型侧背景，AI 辅助开发，可探索陌生领域。要求"好落地、好发顶会"。

## 2. 硬件与双环境工作流

| 环境 | 硬件 | 用途 | 状态 |
|---|---|---|---|
| 本地开发 | RTX 4060 Laptop 8GB (sm_89, Ada) | 开发、调试、1-3B 小模型实验 | **必须经 WSL2 或 Docker 运行 vLLM**（Windows 原生不支持） |
| 按需租用 | RTX 5090 32GB (sm_120, Blackwell) | 最终 7B 实验、全部论文 headline | 权威运行环境 |

**双环境硬规则**：
- 所有论文 headline 数字**只来自 remote_5090/7B 环境**；本地 4060 仅作 dev，标记 `env=dev`，**禁止混入 results/**。
- Ada 与 Blackwell 的舍入/累加/TF32 数值不同，跨 GPU 结果不可混用。提交 results 前必须跑跨 GPU 数值校验（quant-dequant roundtrip + 单 seed PPL）。
- 实验矩阵锁定单一模型家族（3B/7B 作为独立实验，绝不求平均），保证 num_kv_heads/布局一致。

## 3. 设计原则

1. **config 驱动一切**：每个实验由 YAML 唯一定义，运行固化 config 快照，可追溯。
2. **provenance 优先**：数字必须能追溯到"哪个 config + 哪份代码 + 哪个环境 + 哪些数据"。
3. **src 与 vendor 分离**：自研算法在 `src/kvcache`，vLLM fork 是 submodule，改动留 `.diff` 可 rebase。
4. **原始产物与聚合分析分离**：`experiments/`(gitignored 原始运行) 与 `results/`(入库聚合) 分离。
5. **确定性产出**：论文表格/图只由 `scripts/analyze` 产生，notebook 仅探索。

## 4. 目录结构

```
MLSys_Research/
├── CLAUDE.md                    # AI 协作契约（见 §5.9）
├── README.md                    # 项目总览 + 快速开始
├── Makefile                     # 命令入口（wsl2/docker/setup/run/bench/eval/analyze/archive/reproduce/check）
├── pyproject.toml               # 核心包 kvcache；Python 3.10-3.12（不是宿主 3.13）
├── uv.lock / requirements.lock  # 锁定依赖（torch/vllm/flash-attn 兼容）
├── Dockerfile                   # 锁定 vLLM 镜像 digest（5090 权威运行环境）
├── reproduce.sh                 # 审稿人一键复现入口
├── env_check.sh                 # 租机环境自检（driver/CUDA/VRAM/残留进程/vLLM self-test）
├── scripts/
│   ├── env/                     # WSL2/Docker-first 环境搭建；环境探针（nvidia-smi→每个 run 快照）
│   ├── run.sh                   # 唯一运行入口，每次产出 provenance bundle
│   ├── run.py --sweep           # 网格消融 → results/ablations/*.csv
│   ├── bench/                   # --mode=memory、kv_mem.py、serving 压测
│   ├── eval/                    # perplexity/longbench/niah 封装
│   ├── build_vllm.sh / archive.sh / smoke_remote.sh / fetch_data.sh / gen_synthetic_retrieval.py
│   └── analyze/                 # 唯一允许产出论文图表的地方
├── configs/
│   ├── env/{local_4060,remote_5090}.yaml   # 完整版本锁定（含 sm_89/sm_120）
│   ├── datasets/*.yaml          # 数据卡（url/split/sha256/tokenizer/license/预处理）
│   ├── models/*.yaml            # 本地小模型 vs 远端 7B
│   ├── quantization/*.yaml      # 含 typed 校准块（dataset/samples/seq_len/seed/算法/采样序）
│   ├── experiments/*.yaml       # 强制 seed + 引擎默认值 + 预热/测量协议 + 后端开关
│   ├── bench/{throughput,latency}.yaml      # batch×context 扫描、请求分布、并发、SLO
│   └── eval/*.yaml              # 环境路由 + 每 GPU 上下文长度预算
├── src/kvcache/                 # 自研核心包（pip install -e）
│   ├── cache/                   # ★ 核心集成：量化 KV tensor / 自定义注意力后端
│   ├── quantizers/              # 我们的量化器 + KIVI/KVQuant baselines
│   ├── eviction/                # H2O/SnapKV baselines
│   ├── calibration.py           # typed、seed 化、记录采样索引
│   ├── utils/                   # roundtrip 位精确、KV 重建 MSE 校验
├── vendor/                      # git submodule
│   ├── vllm/                    # vLLM fork（固定 SHA）
│   └── vllm-patches/*.diff      # 每个 fork 改动留档，可 rebase 审计
├── data/                        # 数据集与 trace（大文件 gitignored，哈希入 MANIFEST）
│   └── MANIFEST.yaml            # 每文件 sha256 + 来源 URL + license + 预处理说明
├── experiments/                 # (gitignored) 每次 run：logs/metrics(jsonl)/checkpoints + provenance bundle
├── results/
│   ├── _provenance.jsonl        # run_id → config_hash/code_commit/vllm_sha/env_hash/data_hash/seeds
│   ├── ablations/*.csv          # sweep 输出
│   ├── tables/ · figures/
├── tests/                       # pytest：roundtrip 容差、config schema 校验、baseline 复现对照、跨 GPU 数值、单表冒烟
├── notebooks/                   # 仅探索，永不产出提交图表
├── docs/
│   ├── environment.md           # WSL2/Docker-first、WDDM/CUDA-13 注意事项、逐架构内核编译、租机 driver 要求
│   ├── experiment_matrix.md     # 正式 schema：指标+单位、SLO、环境路由、GPU 数值政策
│   ├── integration.md           # vLLM 集成设计（核心改动说明）
│   └── notes/                   # 研究/文献笔记
└── paper/                       # main.tex / figures/ / references.bib
```

## 5. 关键决策详解

### 5.1 config 驱动 + 运行固化
每次运行把**解析后的有效配置**（合并 env 默认值与覆盖后，resolved.yaml + sha256）固化为 provenance bundle。配置文件用 schema 校验（pydantic/Hydra 结构化配置级别即可，不做重型插件框架），typo 快速失败。

### 5.2 Provenance 体系
`scripts/run.sh` 是唯一运行入口，每次运行向 `experiments/<name>/` 产出：
- `resolved.yaml` + sha256
- `git rev-parse HEAD`、`git submodule status`（vLLM SHA）、dirty-tree 标记
- 环境探针（nvidia-smi、driver、CUDA、device capability、torch/vLLM 版本、pip freeze）
- seed 列表、校准采样索引

`results/_provenance.jsonl` 记录 run_id → config_hash/code_commit/vllm_sha/env_hash/data_hash/seeds。headline 运行结束后 `archive.sh` 归档原始 runs（zip+hash，指针+校验和写入 results/）。

### 5.3 环境锁定
- `uv.lock`（或 requirements.lock）锁定 torch/vllm/flash-attn 兼容版本。
- `configs/env/{local_4060,remote_5090}.yaml` 完整版本锁定，标注 sm_89/sm_120。
- `Dockerfile` 引用固定 vLLM 镜像 digest（`vllm/vllm-openai:<tag>@sha256:...`）作为 5090 权威运行环境。
- `scripts/build_vllm.sh` 按固定 submodule SHA 构建并记录 wheel sha256。
- `reproduce.sh` = 顶层一键复现（clone → 构建锁定 submodule → 下载并校验数据 → 跑一个 canonical 实验 → 对照存储的期望指标区间）。

### 5.4 vLLM 集成（最大技术风险）
vLLM 默认 FlashAttention/FlashInfer 后端不接受量化 K/V。`src/kvcache/cache/` 需实现量化 KV tensor / 自定义注意力后端（或非 flash backend）。所有 fork 改动记录为 `vendor/vllm-patches/*.diff`。上线前必须通过"至少服务一个请求"的冒烟测试。**放宽"最小改动"原则**——这是论文核心 claim，允许侵入式改动。

### 5.5 Baseline 策略
H2O/SnapKV（attention-mask 驱逐）→ `src/kvcache/eviction`；KIVI/KVQuant → `src/kvcache/quantizers`。统一评测代码 + 相同 seed + 等内存预算对比，公共结果 schema 以 (baseline, config, seed) 为键。提交 results 前跑 baseline 已知答案测试（复现数值对照发表值）。

### 5.6 内存测量
`scripts/bench --mode=memory`：torch.cuda.memory_snapshot / max_memory_allocated + nvidia-smi 采样，输出 KV-vs-weights 分解、memory-vs-seqlen 曲线。`scripts/bench/kv_mem.py` 读 vLLM allocator 的 num_blocks×block_size 得 memory-vs-bit-width 曲线。**所有内存证据只从 5090/7B 出**。

### 5.7 Serving 压测
`configs/bench/{throughput,latency}.yaml` 编码扫描（batch 大小、上下文长度、输入/输出长度分布、shared-prefix 比例、prefill chunk、并发、Poisson 到达）与 SLO 定义。标准协议：预热请求、测量窗口、稳态检查、每 seed 重复 N 次。报告 TTFT/TPOT/吞吐的 mean±std，保存逐请求原始 metrics(jsonl)，不只聚合值。

### 5.8 seed 与统计约定
每个实验 config 强制 `seed` 字段，seed 所有 RNG（数据采样、校准选择）。Headline 数字 = 3 seeds 的 greedy-decode mean±std。`scripts/analyze/aggregate` 折叠重复次数为 mean/std/p50/p99。

### 5.9 AI 协作治理（写入 CLAUDE.md）
- 除 vendor/ 内不修改 vendor/vllm
- 不手改 experiments/
- 只经 Makefile/scripts 入口运行
- 每次 AI 改动在运行前完成 git commit
- 运行前 clean-tree 门禁（或把 dirty diff 记入 provenance）
- 每次运行快照 git status

### 5.10 数据溯源
`data/MANIFEST.yaml`：每文件 sha256、来源 URL、license、预处理说明。`scripts/fetch_data.sh` 下载固定版本。HF 数据集固定 revision + split 哈希。固定预处理管线（tokenizer、截断、prompt 模板、请求长度分布、seed 化子采样）编码进 `configs/datasets/*.yaml`。

### 5.11 质量评测
`vendor/eval` 以 submodule 固定 lm-eval-harness 与 LongBench commit。`scripts/eval/{perplexity,longbench,niah}.py` 从 configs 读取并写 metrics(jsonl)。`scripts/gen_synthetic_retrieval.py` 生成 seed 化 NIAH 数据。长上下文评测路由到 5090/7B，断言每 GPU 上下文预算，**禁止把 1-3B 长上下文结果当作 7B claim**。

### 5.12 校准规范
`configs/quantization/*.yaml` 内 typed 校准块（dataset、split、num_samples、seq_len、seed、算法 MinMax/percentile/MSE、采样序），双环境共用一份，dev 与 final 断言参数一致。scale/zero/clip 校准产物有 schema、随 run 快照存储、提供 loader/verifier，作为 artifact 交付物。

## 6. 实施阶段（9 步顺序）

1. **修复本地开发环境**：WSL2/Docker-first、Python 3.10-3.12 —— 一切的前提
2. **核心集成**：`src/kvcache/cache` + 单请求冒烟
3. **Provenance 体系**：`scripts/run.sh` + `results/_provenance.jsonl`
4. **环境锁定**：lockfile + Dockerfile + reproduce.sh
5. **Baseline harness + 内存测量**
6. **Eval + bench harness**
7. **Seed/统计 + 数据清单**
8. **校准规范 + 测试**
9. **消融编排 + notebook 约定**

## 7. 明确不做（过度设计裁剪）

- DVC 式数据版本系统（registry + archive 足够）
- 每环境独立 lockfile 生态（一份 lockfile + 锁定镜像覆盖）
- 重型 Hydra/pydantic 插件化配置框架（typed loader + schema 校验足够）
- 多后端/多 feature-flag 全矩阵（默认后端 + 一个替代后端的冒烟测试即可）
- jupytext/papermill notebook 工具链（"仅 scripts/analyze 产出图表"约定足够）
- 独立 CI pipeline（Makefile 再生成 target 足够）
- 早期投入 per-channel/outlier 诊断分析（推迟到主表格成型）

## 8. 主要风险

| 风险 | 应对 |
|---|---|
| vLLM 集成需要侵入式内核/后端改动（最大风险） | AI 辅助 + 尽早冒烟；每个改动留 .diff 可回退 |
| 双环境数值不一致（Ada vs Blackwell） | headline 只出 5090；跨 GPU 数值校验门禁 |
| 长上下文评测超本地预算 | 明确路由到 5090；per-GPU 预算断言 |
| 实验规模膨胀 | 明确"只做审稿人需要 + 支撑 claim"的证据，控制实验矩阵 |
