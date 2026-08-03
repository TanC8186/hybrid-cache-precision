# 数据地图 / 归档索引（2026-08-04）— 审稿人导航

> 全部实验数据、日志、脚本、notes、论文草稿的位置一览。每条 headline 数字可追溯到第一手 JSON/日志。
> 服务器 5090 研究文件已全量下载并归档到本索引所列位置。

## 1. 论文素材（docs/）
| 文件 | 内容 |
|---|---|
| `docs/paper/paper-mainline-2026-08-03.md` | 论文主线草稿：Abstract/Intro/Related Work/Method（四章，`52b77c3`）|
| `docs/paper/serving-evaluation-2026-08-03.md` | Evaluation 章节草稿（uniform int4 主线 + §8 per-layer limitation，`4be09b6`）|
| `docs/notes/serving-3seed-9b-2026-08-03.md` | **2B E2/E3 3-seed mean±std + 9B E2/16384**（headline 数据，`30e78e7`/`f442dea`/`4621bf1`）|
| `docs/notes/serving-latency-throughput-2026-08-03.md` | E2/E3 主矩阵 + §0 深夜修正（uniform re-base）|
| `docs/notes/serving-benchmark-2026-08-03.md` | 离线 3.88x 容量 + 3-seed（serving 单点基线）|
| `docs/notes/tpot-tail-latency-2026-08-03.md` | TPOT p99 拖尾根因（Triton JIT 冷编译）|
| `docs/notes/byte-budget-ordering-2026-08-02.md` | 等字节预算"驱逐>降bit"排序（PPL）|
| `docs/notes/per-layer-page-group-design-2026-08-03.md` | A2 独立 page group 设计（stretch goal）|
| `docs/notes/code-review-2026-08-03.md` | 代码审查报告（3 HIGH/4 MEDIUM 全修复）|
| `docs/notes/serving-eval-paper-level-2026-08-03.md` | E1/E2/E3 评估设计 |

## 2. 结果数据（results/ablations/）
| 目录/文件 | 内容 |
|---|---|
| `serving_bench_20260803/` | 离线 LLM benchmark：fp16/uniform_int4/default_alloc × {4096,8192} × seed{7,42,2026} + **bench_default_alloc_perlayer_*.json（真 per-layer 3-seed）**|
| `bench_lat/{int4,fp16}/` | E2/E3 主矩阵 20 JSON（10 rate × 2 alloc，单 seed=0）|
| `bench_lat/bench_lat3/{int4,fp16}/` | **2B E2/E3 3-seed 48 JSON**（8 rate × 3 seed × 2 alloc）|
| `bench_lat/bench9b_4096/{int4,fp16}/` | **9B E2 14 JSON**（7 rate × 2 alloc）|
| `bench_lat/fp16_16384/, int4_16384/` | 2B @16384 吞吐对比（rate 8 单点）|
| `bench_lat/sharegpt/{int4,fp16}/` | ShareGPT 真实 trace 对比（rate 8/16）|
| `bench_lat/JSON_PROVENANCE.md` | 20 JSON 的 seed/warmup/config 提取表 |
| `bench_lat/logs/` | **55 个 server 日志** + PROVENANCE.md 索引 + bench_logs/（2B 3-seed + 9B 原始日志 14 个）|
| `allocation.json` / `allocation_2b_perlayer.json` | 2B 分配（default_alloc / 真 per-layer）|
| `byte_budget_ordering.csv` | 等字节预算 PPL 排序 |

## 3. 脚本（scripts/）
| 目录 | 内容 |
|---|---|
| `scripts/bench/` | serving 编排/分析：run_bench.sh、bench_driver_5090.sh、bench_seed_loop.sh、launch_2b_pl.sh、launch_9b_v2.sh、extract_metrics.py、mean_std_e2.py、summarize_e2.py、make_figs.py、parse_all.py、gen_json_provenance.py、inspect_kv_config.py、patch_per_layer_fix.py |
| `scripts/exp/` | 实验入口：vllm_serving_bench.py、vllm_smoke.py、gen_allocation.py 等 |
| `scripts/env/` | 环境搭建：install_torch_aliyun.sh、install_vllm_wheel.sh、dl_model_5090.sh |

## 4. 配置（configs/）
`configs/bench/throughput.yaml`（SLO 定义）、`configs/env/{local_4060,remote_5090}.yaml`、`configs/datasets/sharegpt_trace.yaml`、`configs/quantization/*` 等。

## 5. 图（results/figures/）
`fig1_latency_throughput.png`（吞吐-延迟）、`fig2_slo_ttft.png`（SLO）、`fig3_tpot.png`（TPOT p50/p99）—— 均 uniform int4 标签（`a374213`）。

## 6. vLLM fork（vendor/）
- `vendor/vllm/`：git submodule（per-layer KV dtype 扩展）
- `vendor/vllm-patches/per-layer-kv-dtype.diff`：158 行 patch（含 NO-OP 修复，`8ef67f4`）

## 7. Headline 数字 → 第一手来源
| 数字 | 来源 |
|---|---|
| 容量 2.245x @4096 / 3.155x @16384（2B）| logs/PROVENANCE.md（server_pl + server_fp16 + 16384 logs）|
| 容量 2.19x @4096 / 3.167x @16384（9B）| logs/PROVENANCE.md（serve9b_* logs）|
| SLO +25%（int4 50 vs fp16 40 req/s）| bench_lat3/ 3-seed mean±std（mean_std_e2.py）|
| 吞吐 -6~8%、TPOT +8-10% | bench_lat3/ + serving_bench_20260803/ |
| GDN state 18.63 MiB/req | vLLM mamba_utils.py + logs 推导 |
| per-layer ×0.258 反噬 | serve2b_pl.log / serve9b_int4_v2.log（真 per-layer 容量）|
| 9B 饱和 fp16 8.3 / int4 9.3 req/s | bench9b_4096/ |

## 8. 服务器归档（scripts/bench/_server_extra/）
`mlsys_v3.tar.gz`：服务器项目副本打包（configs 与本地一致，本地为超集，仅归档参考）。
