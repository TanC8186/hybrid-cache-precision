# RULER no-think 5 格重跑 — 实验合同（2026-08-11）

对应审稿意见：R1/R7/C2（RULER think 截断伪影）。

## 目的

论文当前 RULER 5 格使用 `--thinking default --max-tokens 256`，FWE 分数可能
受 think 截断影响。本实验用 `enable_thinking=False`（no-think）重跑同一批
5 格，得到无截断伪影的 paired delta，判断 null 结论的方向是否改变。

## 格子定义（与论文 Fig. 3 一致）

| # | Model | Task | Length |
|---:|---|---|---:|
| 1 | 2B | ruler_fwe | 4096 |
| 2 | 2B | ruler_fwe | 8192 |
| 3 | 9B | ruler_niah_multiquery | 4096 |
| 4 | 9B | ruler_niah_multiquery | 8192 |
| 5 | 9B | ruler_fwe | 8192 |

每个格子：allocations `fp16` / `fp16_statebf16`，dataset seeds
`{42, 11, 23}`，engine seed `7`。

总运行数：5 格 × 2 allocations × 3 dataset seeds = **30 cells**。

## 协议

- 引擎：vLLM offline greedy，`ruler_quality.py`；
- `--max-tokens 256`；
- `--disable-thinking`（chat template `enable_thinking=False`）；
- `--max-model-len 16384`，`--gpu-memory-utilization 0.85`；
- 输出：每 cell 原子 JSON + `.sha256`，attempt 目录可 `--resume`；
- 配置 effect 校验失败即 fail-closed（`verify_config_effect`）。

## Attempt ID 与路径

- 2B：`ruler-statebf16-nothink-20260811-2b`
- 9B：`ruler-statebf16-nothink-20260811-9b`
- 输出：`results/quality/ruler-subset/<attempt>/`
- 日志：`logs/<attempt>.log`

## 执行顺序（远程 GPU）

```bash
cd /root/autodl-tmp/MLSys_Research
bash scripts/eval/run_ruler_statebf16_nothink_5cell.sh
```

## 分析决策规则

1. 只使用本 attempt 下 `status=completed_validated`、
   `thinking=disabled`、`max_tokens=256` 的 cell；
2. 不合并 2026-08-07 no-think 旧 attempt 或 think 数据到 no-think
   denominator；
3. 每个格子计算 fp16/bf16 三 seed 均值、paired delta、95% CI；
4. 输出 think vs no-think 对照：delta 符号、大小、CI 重叠；
5. 若 no-think delta 与 think delta 符号/量级发生实质变化，论文必须按
   no-think 重新措辞；若仍落在宽 CI 内，保留 null 结论并补一句
   “no-think corroborates the wide-interval null”。

## 统计自审

- 5 格 × 3 seeds，无多重比较校正，全部逐格报告；
- no-think 是协议修正，不是新筛选窗口；
- 若个别 cell 失败，保留失败记录，不允许用旧 attempt 补齐。

## 交付物

- `results/quality/ruler-nothink-5cell-analysis-20260811.json` + `.sha256`
- `...contract.json` + `.sha256`
- 正文/图注更新建议（视分析结果而定）
