# GSM8K seed 语义核查（ARS 2026-08-09 R1 / DA-C1）

## 结论

`scripts/eval/reasoning_bench.py` 的旧协议中，`seed` 只传给 vLLM engine
（`LLM(seed=args.seed)`），GSM8K 数据集固定取 test 前 200 行，解码为
`temperature=0.0` greedy。因此：

- 不同 seed 不改变题目集合，也不改变 greedy 解码输出 → “3 seeds”是同一确定性
  结果重复 3 次，配对 CI 无随机重复语义；
- 9B 三 seed 全同（fp32=0.885、bf16=0.88）是预期结果，不是 bug；
- 2B fp16 三 seed（0.760/0.755/0.755）的差异来自 attempt 间代码/引擎漂移
  （seed=7 cell 从早期 attempt 复制），不是 seed 效应。

## 代码证据

- `reasoning_bench.py`：`df = pd.read_parquet(...); df.head(max_samples)`；
  `SamplingParams(max_tokens=..., temperature=0.0)`；`LLM(**kwargs)` 中
  `kwargs["seed"] = args.seed`。
- 分析 JSON：`gsm8k-9b-statebf16-analysis-20260808.json` 三行 delta 均为
  −0.005，CI 退化为单点。

## 修复（新协议 v2）

- GSM8K 改为 `df.sample(n=200, random_state=seed)`（无放回），同 seed 在所有
  allocation 之间共享同一题目子集 → 配对比较成立；
- 解码保持 greedy（temp=0），engine seed 仅作 provenance；
- 每个 cell 记录 `sampled_indices` 与 `seed_semantics`，分析器 fail-closed；
- 旧 head-200 attempt 保留为 legacy，不并入新分母。

## 表述规则

- 新协议完成前，9B GSM8K 只写“确定性差异 −0.5pt（旧协议 3 seed 重复）”，
  禁止用退化 CI 支撑显著性；
- 新协议完成后按真实 3-seed 配对 mean±CI 表述。
