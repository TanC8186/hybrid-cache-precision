# 2026-08-07 补跑实验运行状态（服务器已开机）

> 服务器：`connect.westd.seetacloud.com:43022`，RTX 5090 32GB，免密 SSH 已生效。
> 目标：执行文献调研清单（`docs/notes/experiment-design-literature-2026-08-07.md`）
> 的前 4 项，全部证据原子化 + sha256 + 失败 fail-closed + 结果及时回传 git。

## 1. GPU 串行流水线（单卡，链式守护已挂载）

| 顺序 | 实验 | Attempt ID | 规模 | 状态 |
|---|---|---|---|---|
| 1 | NIAH 重跑（max_tokens=256，修复 `<think>` 32-token 截断伪影） | `niah-fixed-20260807` | 5 alloc × 3 seeds × 3 depths × 2 lengths = 90 cells | **运行中**（12:00 约 20/90） |
| 2 | RULER 子集（官方生成器/评分，noise haystack；greedy 固定数据 → 单 seed） | `ruler-subset-20260807` | 7 tasks × 2 lengths × 5 alloc × 1 seed = 70 cells | 数据已生成（14/14 校验通过），等 NIAH 完成自动启动 |
| 3 | TurboQuant k8v4 / 4bit_nc + FP8 serving protocol-v3 门禁 | `r5-tq-v3-*-mvex/pilot-20260807` | MVEx 3+3 → Pilot 9+9 | 配置已提交 `MLSys_Serving_f7a79f5`（3267efa）并 dry-run 通过，等 RULER 完成 |
| 4 | Qwen3.5-9B NIAH 重跑（fp16/int4/packed） | `niah-fixed-9b-20260807` | 3 alloc × 3 seeds × 3 depths × 2 lengths = 54 cells | 脚本就绪，等 serving 门禁完成 |
| 5 | 推理基准（2B；greedy 固定数据 → 单 seed） | `reasoning-20260807` | gsm8k 200 + mmlu 500 + aime25 30，× 5 alloc × 1 seed | 数据已下载，等 9B 完成 |

链式守护：`scripts/eval/chain_after_niah.sh` → `scripts/bench/chain_after_ruler.sh`
（MVEx+Pilot+6-dtype 容量探针）→ `scripts/eval/chain_after_serving_gates_9b.sh` → `scripts/eval/chain_after_9b_reasoning.sh`。
任一上游出现 `[FAIL]` 即 fail-fast，不启动下游。

## 2. 本轮新增/修复内容

- `scripts/eval/kv_quality_retrieval.py`：新增 `--max-tokens`（默认 256，resume 校验 max_tokens），
  每条 record 增加 `prompt_tokens`/`output_tokens`、`hit_think`/`hit_final` 诊断。
- `scripts/eval/run_niah_fixed.sh`：5 alloc × 90 cells，resumable。
- `scripts/eval/run_niah_fixed_9b.sh`：9B 核心 3 alloc × 54 cells。
- `vendor/ruler/`：官方 RULER 生成器/评分（commit `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`，
  Apache-2.0），`ruler_subset.yaml` 定义 noise-haystack 变体（官方任务脚本 + 官方模板）。
  `scripts/eval/ruler_prepare.py` 生成 7 task × {4096,8192} × 20 samples（seed 42），
  `scripts/eval/ruler_quality.py` 用官方 `string_match_all` 评分。
- `scripts/eval/reasoning_bench.py`：gsm8k（openai/gsm8k test 前 200）、
  mmlu（cais/mmlu all/test 前 500）、aime25（opencompass/AIME2025 全 30），
  确定性抽取：gsm8k=最后一个数字、mmlu=最后一个 A-D、aime25=最后一个整数。
- serving 配置：`configs/experiments/r5_turboquant_protocol_v3_{random60,sharegpt300}_formal.yaml`，
  与 A2 protocol-v3 formal 完全一致（PIECEWISE、60s/300s 窗口、warmup 120、
  TTFT {250..3000}、TPOT 200、goodput≥0.95、3 seeds），allocations 为
  turboquant_k8v4 / turboquant_4bit_nc / fp8。
- 容量探针：`inspect_kv_config.py` 对 fp16 / uniform_int4 / packed_per_layer /
  turboquant_k8v4 / turboquant_4bit_nc / fp8 各启动一次（eager，max-len 4096），
  输出 capacity tokens + max concurrency，落在 `$OUT_ROOT/capacity/`。

## 3. 数据与溯源

- RULER 数据：`data/ruler/<task>_L<len>/validation.jsonl` + `.sha256` + `manifest.json`
  （服务器工作区；数据本身 gitignored，回传时只回传结果与 manifest）。
- 推理数据：`/root/autodl-tmp/caches/datasets/{gsm8k,mmlu,aime2025}`（ModelScope 快照）。
- vLLM fork：commit `55f47685`；serving 根代码 worktree `3108650` + 本轮配置 commit `3267efa`。

## 4. 待办/人工门禁

- [ ] RULER 完成 → serving MVEx+Pilot 自动跑；Pilot 通过后**人工审阅**再启动 Formal
  （`--phase formal`，Random60 45 cells + ShareGPT300 63 cells）。
- [ ] 每阶段结果回传本地（`results/quality/`）+ 哈希核对 + git 提交。
- [ ] 9B 与推理完成后更新 `results/quality/*-analysis.json` 与论文 Eval 章节。
- [ ] C4/PG19 PPL、32K/64K 探针、LongBench 子集（数据未下载）仍待后续安排。
