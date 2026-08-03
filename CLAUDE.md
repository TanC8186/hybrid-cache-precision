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
7. **频繁审查，防幻觉/假象（铁律）**：对任何配置生效、数值结论，必须用证据二次确认，不轻信"应该生效"——CLI 参数可能静默 NO-OP / fallback / 缓存残留（实例：`--kv-cache-dtype-per-layer` 曾静默退化为 uniform int4，per-layer 数据长期是假象）。
   - **运行前/后都要验证配置真正生效**：grep 启动日志里的 override / 实际 `kv_cache_dtype` / KV block 结构，确认不是假象再采信。
   - **对每个数值先质疑来源**：真实测量 vs 默认值 / 兜底路径 / 旧缓存；来源不明 = 数字不可用。
   - **频繁自审**：每完成一步改动或运行，回读产物交叉核对（代码、配置、日志、结果 JSON 四方一致）；发现不一致立即停下重查，绝不带着错误前提继续堆叠实验。

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
