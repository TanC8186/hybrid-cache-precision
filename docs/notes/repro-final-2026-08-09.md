# S-formal 独立复现最终审计（2026-08-09）

## 执行

- 新 attempt：`statebf16-v3-random60-repro-20260809`（30 样本）与
  `statebf16-v3-sharegpt300-repro-20260809`（42 样本），与原始矩阵同配置、
  同 seeds、同输出根（`statebf16-serving-repro-20260809`）；
- 完整性：R60 30/30、SG 42/42 全部 `completed_validated`；sidecar 哈希、
  请求守恒、到达窗口（≤0.01%）、SLO 键、日志硬证据全部通过，0 进程失败。

## 对比结果

### Random60（PARTIALLY_REPRODUCIBLE）

- 边界 1/10 不一致：int4@1000ms 原始 35、复现 40（bf16 列两轮一致=40）；
- 重叠 mean goodput：48/50 ≤10%；2 格超差（int4 r40 250/500ms，均属过载低均值格）；
- 配对 Δ（bf16−fp32）：显著格方向与量级复现——r40 250ms +0.334→+0.304、
  500ms +0.215→+0.138；r45 2000/3000ms +0.324/+0.367→+0.338/+0.372；
  r50 2000/3000ms +0.041/+0.072→+0.041/+0.072；微小效应格 2 处 CI 符号翻转。

### ShareGPT300（关键格 NOT REPRODUCIBLE）

- 边界 2/10 不一致（250ms：int4 40→35、int4_statebf16 35→40，方向翻转）；
- 重叠 mean goodput：11/70 超差；其中 **int4 r45 全部阈值相对差 >100%**
  （原始 0.67–0.73 vs 复现 0.14–0.23）；
- r45 取证：原始 int4 三 seed TTFT p99 = 262/822/16191ms（双峰，0 失败）；
  复现三 seed ≈ 18.5s 且出现 160/260/264 请求失败——过载边界的环境/调度波动，
  非审计伪影；
- 配对 Δ 方向在 r45 翻转（原始 bf16 差于 int4，复现 bf16 优于 int4）。

## 判定

**PARTIALLY_REPRODUCIBLE**（非 VERIFIED）：

- Random60 过载区（r40 250/500ms、r45、r50 2000/3000ms）的 paired goodput 增益
  在独立复现中重现，可作为点估计 + 宽 CI 披露；
- “Random60 1000ms 边界 35→40” 与 ShareGPT 250ms 边界方向均**不复现**，
  论文不得写为稳定 headline；
- ShareGPT r45 复现不一致（量级与方向均翻转）→ 该区不可作任何 serving 收益表述；
- claim #5 维持 `ANALYZED`，仅在上述 workload × threshold 限定内进正文，
  不进 Abstract 的 headline。

## 建议的论文口径

1. Random60：报 r40 250/500ms 与 r45/r50 高阈值处的 paired goodput 增益
  （原始 + 复现均值区间），并披露边界敏感性；
2. ShareGPT300：只报“500ms 及以上边界持平（两轮一致 40）”，不报 250ms 差异；
3. 明确说明 serving 收益是 workload-dependent 且过载边界存在运行间波动。
