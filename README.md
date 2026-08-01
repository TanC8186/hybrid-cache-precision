# MLSys Research

面向 **MLSys 顶会**的 LLM serving **KV cache 量化/压缩**研究项目，基于 vLLM。

> 设计文档：`docs/superpowers/specs/2026-08-01-mlsys-experiment-framework-design.md`

## 工作流

| 环境 | 硬件 | 用途 | 规则 |
|---|---|---|---|
| 本地开发 | RTX 4060 Laptop 8GB (sm_89) | 开发 / 冒烟 / 1-3B 小模型 | 必须经 WSL2 或 Docker 运行 vLLM；结果 **dev-only** |
| 按需租用 | RTX 5090 32GB (sm_120) | 最终 7B 实验 | **唯一**允许产出论文 headline 数字的环境 |

## 快速开始

```bash
# 1. 本地开发环境（WSL2 内执行，见 docs/environment.md）
bash scripts/env/setup_wsl2.sh

# 2. 初始化 vLLM fork
git submodule update --init vendor/vllm

# 3. 安装核心包
make setup-local

# 4. 运行实验（示例）
make run EXP=template
```

## 目录结构

```
configs/        ← 一切实验由 YAML 唯一定义（env / datasets / models / quantization / experiments / bench / eval）
src/kvcache/    ← 自研核心包：cache(集成) / quantizers / eviction / calibration / utils
vendor/         ← vLLM fork (submodule) + eval harness + vllm-patches/*.diff
scripts/        ← run / bench / eval / analyze / env / build_vllm / archive 等入口
experiments/    ← 运行原始产物（gitignored，每次 run 带 provenance bundle）
results/        ← 聚合分析 + _provenance.jsonl（入库）
data/           ← 数据集与 trace（溯源见 MANIFEST.yaml）
docs/           ← 环境说明 / 实验矩阵 / 集成设计 / 笔记
tests/          ← 正确性门禁（roundtrip / config schema / baseline 复现）
notebooks/      ← 仅探索，不产出提交图表
paper/          ← 论文（图表只由 scripts/analyze 产生）
```

## 铁律

- 只经 `Makefile` / `scripts/run.sh` 入口运行
- 运行前 git 树 clean（commit-before-run）
- 论文 headline 只来自 5090/7B 环境
- 每个实验 config 强制 `seed`
- 见 `CLAUDE.md` 的完整协作契约
