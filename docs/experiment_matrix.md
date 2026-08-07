# 实验矩阵（正式 schema）

> **2026-08-08**：投稿面实验的权威矩阵与状态见
> [docs/paper/experiment-matrix-plan-2026-08-08.md](paper/experiment-matrix-plan-2026-08-08.md)；
> 声明-证据映射见
> [docs/paper/claim-evidence-map-2026-08-08.md](paper/claim-evidence-map-2026-08-08.md)。
> 本文件保留为字段/schema 规范。

> 目标：论文里出现的**每一个数字**都能追溯到：config + 代码 commit + 环境 + 数据 + seed。
> 每条实验结果按下面的 schema 记录；缺字段的数字不允许进论文。

## 每条实验记录的字段

```yaml
experiment_id: quality_7b_2bitkv_longbench   # 与 configs/experiments/<name>.yaml 一致
config_hash: <sha256 of resolved.yaml>
env_id: remote_5090                           # headline 只允许 final 环境
code_commit: <git HEAD>
vllm_sha: <submodule SHA>
data_hash: <data/MANIFEST.yaml 对应条目>
seeds: [42, 2026, 7]
metrics:                                      # 见下方"指标与单位"
  - name: ppl
    value: 8.42
    unit: null
    mean_over_seeds: true
```

## 指标与单位（必须显式标注）

| 指标 | 定义 | 单位 |
|---|---|---|
| PPL | 困惑度 | 无 |
| TTFT | 首 token 延迟（p50/p99） | ms |
| TPOT / ITL | 每 token 延迟（p50/p99） | ms |
| throughput | 吞吐 | tokens/s（注明并发与 batch） |
| memory | KV cache 占用 | GB（注明 context length） |
| max_context_under_budget | 固定 GPU 预算下最长上下文 | tokens |
| 检索准确率 | NIAH 各 (depth, length) | % |

## 引擎默认值（显式化，不靠代码默认值）

`max_model_len` / `max_num_seqs` / `gpu_memory_utilization` / `page_size` / `chunked_prefill` / `enable_prefix_caching` / `attention_backend` —— 全部写进 `configs/experiments/*.yaml`。

## 环境路由政策

- 论文 headline **只**来自 `remote_5090`；`local_4060` 标记 `env=dev`，**禁止**混入 `results/`
- 长上下文评测路由到 5090，`context_budget_guard` 断言每 GPU 预算
- 禁止把 1-3B（4060）长上下文结果当作 7B claim

## SLO 阈值

见 `configs/bench/*.yaml`：`ttft_p99_ms` / `tpot_p99_ms`。容量卖点 = 相同 SLO 下能容纳的并发请求数 / 上下文长度。

## GPU 数值政策

跨 GPU 数字不可混用（Ada vs Blackwell 数值不同）。提交 results 前跑跨 GPU 校验。
