# R4 — A2 质量闭环实验方案（2026-08-06）

> 目的：回应 ARS 审稿 R4 —— 证明 packed per-layer page group 的容量恢复（0.833× uniform）
> 没有以质量回退为代价；同时补逐层敏感度/异构预算的 3-seed 统计（R3 遗留）。
> 执行环境：RTX 5090（sm_120）`ssh 5090`；冻结根代码与 vLLM commit；本地 4060 结果仅 dev。

## 1. 目标与假设

- H-Q1：packed per-layer 的 Wikitext-2 PPL 不劣于 uniform int4（配对 Δ 95% CI 上界 < 预声明容差）。
- H-Q2：packed per-layer 在 retrieval/long-context（NIAH 4K/8K/16K + LongBench 子集）不劣于 uniform int4。
- H-Q3：逐层敏感度/异构预算排序在 3-seed 最终 harness 下保持（L23 最敏感、驱逐 > 降 bit）。

## 2. 冻结因素

- 模型：Qwen3.5-2B（ModelScope 路径，config hash 与 A2 相同）。
- 分配：fp16（`kv_cache_dtype=auto`）、uniform int4（`int4_per_token_head`）、
  packed per-layer（`int4_per_token_head` + `--kv-cache-dtype-per-layer {"23":"auto",...}` +
  `--enable-per-layer-page-groups`）。
- 种子：{7, 42, 2026}。
- **PPL 协议（transformers 路径，与 `byte_budget_3seed.log` 完全一致）**：
  Wikitext-2 test、5×2048、chunk=128、seed 化起始位置；
  入口 `scripts/exp/hybrid_premise.py --seeds 7,42,2026 --num-seqs 5 --max-len 2048
  --corpus data/wikitext2_test.txt`，fp16=`--bits 16`、uniform=`--bits 4`、
  packed=`--bits 4 --layer-bits '{"23":16}'`；每 seed 输出一份 PPL（`.seeds.csv`），
  用配对 t-CI 比较（`scripts/eval/analyze_r4_quality.py`）。
- **Retrieval 协议（vLLM 离线贪婪，入口 `scripts/eval/kv_quality_retrieval.py`）**：
  NIAH seed 化合成上下文，depths {25,50,75} × lengths {2048,4096} × 3 needles/样本；
  配置生效用 engine `vllm_config.cache_config` + KV group 结构校验；
  精确匹配 needle code 计 accuracy。LongBench 子集在 NIAH 通过后追加。
- 成功判据（预声明）：PPL 配对 Δ 的 95% CI 上界 ≤ +2% 相对 FP16；retrieval 准确率差 ≤ 1 分；
  敏感度排序方向不变。

## 3. 分阶段放行（沿用 resumable-runner 语义）

1. **MVEx**：packed/seed7/单任务单样本，验证命令、config 生效（日志含
   `enable_per_layer_page_groups` + `CUDAGraphMode`/eager 选择）、PPL 与 retrieval schema、
   哈希链；注入断连复测。
2. **Pilot**：3 alloc × 3 seeds × Wikitext（3 个 PPL 进程）+ NIAH 单 (depth,length)（9 样本）；
   任一 schema 漂移或静默排除即阻断。
3. **Formal**：3 alloc × 3 seeds × {Wikitext, NIAH×3, LongBench×2}；短切片 + `--resume`，
   每样本独立不可变 attempt。
4. **Reproducibility**：新 attempt，复跑边界样本；10% 对称相对差 / 边界精确比较。

## 4. 产出与归档

- `experiments/` 原始产物（gitignored）+ `results/` 聚合与 `_provenance.jsonl`；
- 论文表：packed vs uniform vs fp16 的 PPL 配对 Δ [CI]、retrieval 分数表；
- 敏感度/异构预算 3-seed 表（替换 `layer_sensitivity.csv`/`hetero_budget.csv` 单跑声明）。

## 5. 失败语义

- 失败/超时/分母不完整/配置未生效的 attempt 保留隔离，不进入正式分母；
- A2 serving 边界在独立复现前保持 ANALYZED，质量闭环完成不自动升级 serving 状态。
