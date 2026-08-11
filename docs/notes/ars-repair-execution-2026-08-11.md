# ARS 修复补全计划与执行状态（2026-08-11）

范围：`paper/mlsys2026` 五席审稿（EIC/R1/R2/R3/DA）意见的 M0 措辞+引用修复、
M1 纯分析，以及后续实验里程碑。

## 一、本轮已完成（M0 措辞与引用）

### 1. 引用修复（nature-ref-verifier / arXiv API 核验）

| Bib key | 问题 | 修复 |
|---|---|---|
| `turboquant` | 标题/venue 错误（写成 KV 压缩论文） | 改为 *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*，ICLR 2026，arXiv:2504.19874 |
| `jamba` | 作者列表与 arXiv 官方 22 人不符 | 整体替换为官方 22 人列表 |
| `recurrentgemma` | `de Frietas` 拼写错误 | 改为 `de Freitas, Nando` |
| `streamingllm` | 误作 arXiv 2023 article | 改为 ICLR 2024 inproceedings |
| `mamba` | 误作 arXiv 2023 article | 改为 COLM 2024 inproceedings |
| 新增 `griffin` | 谱系引用缺失 | arXiv:2402.19427 |
| 新增 `deltanet` | 谱系引用缺失 | NeurIPS 2024（不是 ICML） |

审计报告：[citation-audit-2026-08-11.md](citation-audit-2026-08-11.md)

### 2. 措辞修复（W-1…W-7、W-9、W-10）

- W-1：TurboQuant 正文描述与真实论文一致，并补“两者都未把 recurrent-state dtype
  当作可分配维度”。
- W-4（S3）：related work 补 eviction/offloading 与 state-dtype 正交性的 3 行论证。
- W-6：正文明确“第二次 formal run = run-stability，不是独立样本”，覆盖摘要、§1、
  §4、§5.4、§6、结论。
- W-7：Tensor parallelism 补 P=2/2B/int4/L=4096 的一阶数值示例。
- W-9：补 Griffin/DeltaNet 谱系句。
- W-10：Fig. 8 图注补 “per-seed resolution of Fig. 2”。
- 结论新增 operator 决策规则：memory-bound 目标场景、2B/4K/int4 约 +247 slots、
  KV 主导时重新评估。
- 删除/改写 “replicated measurements”“reproduced aggregate delta” 等易被误读为
  独立复现的措辞。

## 二、本轮已完成（M1 纯分析，不跑新实验）

### E-1 serving 方向一致性

- 输入：formal/repro 各 60 格 paired deltas + BH-FDR JSON。
- 输出：
  - `results/quality/serving-direction/serving-direction-agreement-20260811.json`
  - `...contract.json` + 两个 `.sha256`
- 关键结果：
  - Random60 overload（两轮均 overload）13/13 格在两轮中同为正方向；
  - ShareGPT overload 10 格中 7 格反向，方向证据不跨 workload；
  - 全网格符号检验不显著（cell 间共享 seeds，仅作探索性描述）。
- 正文 §5.4 已回填一句 workload-limited 方向审计。

### E-7 GSM8K paired Cohen's d

- 输入：2B/9B 各 9-seed 原子 JSON。
- 输出：
  - `results/quality/reasoning/gsm8k-cohens-d-9seed-20260811.json`
  - `...contract.json` + 两个 `.sha256`
- 关键结果：2B state-bf16 paired d = −0.92（n=9）；9B d = +0.54 但 CI 含零，
  不解释为增益。
- 正文 §5.2 已回填。

## 三、后续实验里程碑（本轮不跑，已列入计划）

| 实验 | 对应审稿意见 | 目的 |
|---|---|---|
| RULER no-think 5 格重跑 | R1/R7/C2 | 消除 think 截断对 FWE null 结论的方向不确定性 |
| fixed-block-count / fixed-bytes serving 对照 | R10/DA M4 | 分离容量、带宽、页对齐三个机制通道 |
| TP=2/4 容量/分片实测 | R6/W3 | 验证 TP 一阶推导不是纯预期 |
| offloading / prefix-caching 对照 | R3/S3 | 明确替代杠杆与 state-bf16 的交互 |
| 外部模型（Mamba2 系）容量探针 | DA M3 | 扩展 GDN-only 的一般性边界 |

## 四、自审门

- 11 类统计谬误自查已写入两个 analysis JSON 的 `self_review` 字段。
- 每个新数字均从原子 JSON 计算，正文回填前先落盘 + sha256。
- 未修改任何历史 formal/repro 数据文件。
- `verify_figure_data.py`：LEDGER_ENTRIES=229，全部通过。
- LaTeX 重建：9 页，32 条 bib，无 undefined citation，`git diff --check` 干净。

## 五、涉及文件

- 修改：`paper/mlsys2026/main.bib`、`paper/mlsys2026/main.tex`、`main.pdf`
- 新增：`docs/notes/citation-audit-2026-08-11.md`
- 新增脚本：`scripts/quality/analyze_serving_direction.py`、
  `scripts/quality/analyze_gsm8k_cohens_d.py`
- 新增结果：`results/quality/serving-direction/`、
  `results/quality/reasoning/gsm8k-cohens-d-9seed-20260811.*`
