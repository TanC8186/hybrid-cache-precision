# CLAUDE.md — 项目协作契约

## 项目
- **目标**：MLSys 顶会论文，LLM serving 的 KV cache 量化/压缩
- **基座**：vLLM（`vendor/vllm` fork，git submodule）
- **工作流**：本地 RTX 4060 8G（WSL2/Docker 内开发，1-3B 模型）+ 租用 RTX 5090 32G（最终实验，7B）
- **设计文档**：`docs/superpowers/specs/2026-08-01-mlsys-experiment-framework-design.md`

## 可复现性硬规则（违反 = 数字不可信）

1. **commit-before-run**：任何运行前，所有代码改动必须已 git commit。
2. **只经入口运行**：通过 `Makefile` / `scripts/run.sh` 启动实验，不手敲 vLLM 命令行。
3. **不手改 `experiments/`**：该目录是运行产物（gitignored），任何手工改动视为无效。
4. **vendor/vllm 改动必须留档**：记录到 `vendor/vllm-patches/*.diff`，便于 rebase 审计。
5. **headline 只从 5090**：论文数字只来自 `remote_5090` 环境；本地 4060 是 dev，禁止混入 `results/`。
6. **seed 强制**：每个实验 config 必须带 `seed`；headline = 3 seeds 的 mean±std。

## AI 助手行为约定

- 每次改动结束 → git commit（原子提交，信息清晰）。
- 运行实验前检查 git 树是否 clean；不 clean 则先提交或丢弃。
- 论文表格/图只允许由 `scripts/analyze` 产生；notebook 仅探索。
- 修改 `vendor/vllm` 前先说明改动意图，改动后留 `.diff`。
- 对不熟悉的接口先查 `docs/` 与已有代码，不臆造 API。
- 涉及硬件/评测的声明以 `configs/env/*.yaml` 与运行 provenance 为准，不写进散落注释。

## 常用命令

```bash
make setup-local      # 初始化本地开发环境（WSL2 内）
make run EXP=<name>   # 运行实验（provenance 固化）
make sweep            # 网格消融
make bench            # serving benchmark
make eval             # 质量评测
make analyze          # 汇总 → results/（唯一论文图表来源）
make archive          # 归档 headline 原始运行
make reproduce        # 一键复现（供审稿人）
make check            # 环境自检
```
