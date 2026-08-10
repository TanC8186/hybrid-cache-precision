# MLSys 2026 论文 Figure Contract（nature-figure 契约）

Core conclusion: 把循环 state 精度纳入与 KV 位数的联合内存预算，可在 int4 KV 下
把容量增益从 2.245x 提高到 2.675x（2B@4K），且质量代价在统计分辨率内不可检测；
serving 过载区的 paired goodput 增益在独立复现中重现，但边界对 workload/threshold
敏感，只作限定性证据。

Figure archetype: quantitative grid（图 2/3/4）+ schematic-led composite（图 1）
Target journal/output: MLSys 2026，双栏 10pt，正文 10 页；图宽单栏 3.4 in / 全宽 7.0 in
Backend: Python（matplotlib/seaborn，已保存偏好，全部绘图/预览/导出均 Python）

Final size: 图1 全宽 7.0 in；图2/图3/图4 单栏或全宽按内容定；字体 >= 5pt@final

Panel map:
  fig1a: 混合架构内存组成示意（KV 随 L 增长 vs state 固定；KV 位数 x state 位数）
  fig1b: r_state 2x2 网格（int4/fp16 KV x 2B/9B x 4K/16K），measured + predicted
  fig2: 质量森林图（GSM8K 2B/9B、stacking、int4，9-seed 配对 Δ [95% CI]）
  fig3: PPL stacking + RULER（3-seed 配对 Δ [95% CI]）
  fig4: serving Random60 过载区 paired goodput Δ（formal + repro 双点）
  fig5: block 粒度证据（tokens/block 与 block 数，fp32 vs bf16 state）
  fig6: 逐层 state 敏感度（18 GDN 层 x C4/PG19，Bonferroni/BH-FDR 后无显著）
  fig7: harness chunk 消融（chunk1 vs 128）+ stacking 成本（fp16 vs int4 KV）
  fig8: GSM8K 9-seed 逐 seed 配对折线（2B/9B）

Evidence hierarchy:
  hero evidence: 图1b int4 列保守下界（4/4 全负，P=0.0625）+ 复合 r_kv 2.245→2.675
  mechanism: 图5 block 粒度（离散 block 取整解释残差）
  validation evidence: 图2/图3 质量闭环（GSM8K/RULER/PPL）
  negative-result: 图6 逐层敏感度校正后无显著
  robustness: 图4 独立复现区间；图8 逐 seed 配对稳定性；boundaries 表放正文
  method-boundary: 图7a chunk 级 harness 近似；图7b stacking 无叠加成本
Statistics needed: n=3/9 seeds；paired 95% CI；MDE/power 标注（正文）；RULER 宽 CI
Source data needed: results/verified/2026-08-09/capacity-2x2-analysis.json；
  results/quality/gsm8k-{state9seed,9b-state9seed}-v2-analysis-20260809.json；
  results/quality/ppl-stacking-analysis-20260809.json；
  results/quality/ruler-statebf16-multiseed-analysis-20260809.json；
  results/verified/2026-08-09/statebf16-serving-{formal,repro}-analysis.json；
  results/quality/state-sensitivity-analysis-20260809-bonf.json；
  results/quality/chunk-ablation/*.csv
Image-integrity notes: 无显微/照片；无裁剪/调色；数值面板全部由脚本从原子 JSON 读取
Reviewer risk: CI 宽、RULER 无检测能力、serving 仅方向性 + 复现限定；图内不出现
  未复现的边界结论；所有误差线定义统一为 paired 95% CI
