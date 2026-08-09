# GSM8K 9-seed 功效预注册（ARS R1/S3 + nature-statistics）

## 目的

seed 化题目子采样后，3-seed 配对 CI 全部含 0（2B state −1.33pt [−5.13,+2.46]、
9B 0.00pt [−1.24,+1.24]）。为判断“回退是否存在”并满足 S3 的最小可检测效应
（MDE）报告要求，将 seed 数扩到 9。

## 预注册（2026-08-09，运行前）

- 分配：2B × {fp16, fp16_statebf16, uniform_int4, uniform_int4_statebf16}，
  9B × {fp16, fp16_statebf16}；
- seeds：{7, 42, 2026, 11, 23, 31, 47, 73, 97}（每 seed 独立无放回抽 200 题，
  同 seed 跨分配共享题目集；greedy 解码）；
- 统计：配对差值 mean ± 95% t-CI（df = n−1）；精确 p 值；
- 多重比较：state、int4、stacking 三组对比为预注册主对比，不做 Bonferroni
  （限定数量的计划内比较）；敏感性分析另行披露；
- 决策规则：仅当配对 CI 不含 0 时写“回退显著”；否则写“点估计 + 宽 CI”，
  禁止“趋势/可能回退”式表述；
- MDE：MDE = (t_{1−α/2,df} + t_{0.8,df}) × sd/√n，α=0.05 双侧、power=0.80；
- 报告：每分配输出 p、CI、MDE、observed-power；若 |Δ| < MDE，明确写
  “功效不足以检测该量级差异”，不因 p 值大小改变结论措辞。

## 运行与审查

- 2B 36 cells + 9B 18 cells，由 S-formal 完成后自动接力执行；
- 每 cell 校验：`sampled_indices` 长度 200、`seed_semantics`、`config_effect`、
  sha sidecar；
- 分析器输出 `paired_t` 字段（p、CI、MDE、power），全部落
  `results/quality/*-state9seed-v2-analysis-20260809.json`。

## 已知边界

- GSM8K 题目子集是唯一随机源（greedy 解码确定性），n 的独立单元是 seed 子集，
  不是单题；
- 2B state 效应点估计 −1.33pt 低于 n=9 时典型 MDE（若 sd≈0.016，MDE≈1.7pt），
  显著性不保证；此结论如实写入论文。
