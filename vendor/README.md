# vendor — 锁定的上游依赖（git submodule）

| 路径 | 内容 | 固定方式 |
|---|---|---|
| `vendor/vllm` | vLLM 工作副本 | 浅克隆（`--depth 1 --filter=blob:none`），provenance 记录 commit SHA |
| `vendor/eval` | lm-eval-harness / LongBench 评测脚本 | submodule 固定 commit（待建） |
| `vendor/vllm-patches/` | 我们对 vLLM 的每个改动，`.diff` 留档 | 入库 |

> **网络注意**：huggingface.co 与 github.com 在此环境被墙。
> - 模型/数据集经 ModelScope 或 hf-mirror.com
> - vLLM 经 ghfast.top 镜像克隆：`git clone --depth 1 --filter=blob:none https://ghfast.top/https://github.com/vllm-project/vllm vendor/vllm`

## 初始化

```bash
git clone --depth 1 --filter=blob:none https://ghfast.top/https://github.com/vllm-project/vllm vendor/vllm
```

## 修改 vLLM 的纪律

1. 在 `vendor/vllm` 内提交改动（不能只停在工作区）
2. 生成 diff 留档：`git diff main > ../vllm-patches/<描述>.diff`（或按 commit 记录）
3. 记录到 `docs/integration.md` 的进展清单

> 放宽"最小改动"原则是允许的——论文的中心 claim（量化 KV 走真实注意力路径）本身就需要侵入式改动。关键是**每个改动可审计、可回退**。

## 版本说明

- vLLM 无原生 Windows 支持；本地开发在 WSL2/Docker 内进行
- torch/vLLM 按 CUDA 架构（sm_89 vs sm_120）编译，**二进制不互通**，跨机器要重建
- 工作副本是浅克隆；需完整历史做 rebase 时，用官方 remote 加深（网络允许时）
