# 2026-08-07 补跑结果消化（模板/进度表）

> 本文件按阶段填写；每阶段完成并下载、哈希核对、git 提交后更新对应小节并删除
> “待填写”占位。所有数字必须可溯源到 `results/quality/` 下原子 JSON + sha256。

## 1. NIAH 重跑（max_tokens=256，attempt `niah-fixed-20260807`）

状态：**运行中**（90 cells = 5 alloc × 3 seeds × 3 depths × 2 lengths）。
判定：每 cell 3 needles；主指标 accuracy（code 出现在 answer），
诊断指标 hit_final（`</think>` 后出现）/hit_think；paired Δ vs fp16（18-cell t-CI）。

| 分配 | cell 均值 | 总体 needle | hit_final | Δ vs fp16 [95% CI] |
|---|---:|---:|---:|---:|
| fp16 | 待填写 | 待填写 | 待填写 | — |
| uniform int4 | 待填写 | 待填写 | 待填写 | 待填写 |
| packed per-layer | 待填写 | 待填写 | 待填写 | 待填写 |
| turboquant_k8v4 | 待填写 | 待填写 | 待填写 | 待填写 |
| turboquant_4bit_nc | 待填写 | 待填写 | 待填写 | 待填写 |

结论（待填写）：32-token `<think>` 截断伪影是否消除；packed/TurboQuant 相对 fp16 是否无显著退化。

## 2. RULER 子集（attempt `ruler-subset-20260807`）

状态：**排队中**（70 cells = 7 tasks × 2 lengths × 5 alloc × 1 seed；官方生成器
`c3f5e3b4`，noise haystack，20 samples/task/length，官方 `string_match_all`）。

| task | L | fp16 | uniform | packed | k8v4 | 4bit_nc |
|---|---:|---:|---:|---:|---:|---:|
| niah_single | 4096 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| niah_multikey | 4096 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| niah_multivalue | 4096 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| niah_multiquery | 4096 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| vt | 4096 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| cwe | 4096 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| fwe | 4096 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| （同上 8192） | 8192 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |

结论（待填写）：packed 是否在长上下文检索/抽取任务上与 fp16/uniform 持平；
TurboQuant 4-bit NC 是否如 vLLM 研究提示在推理任务上有可见回落。

## 3. TurboQuant/FP8 serving（protocol-v3）

状态：**排队中**（门禁 MVEx+Pilot → 人工审阅 → Formal）。
Attempts：`r5-tq-v3-{random60,sharegpt300}-{mvex,pilot,formal}-20260807`。
输出根：`/root/autodl-tmp/r5-serving-20260807`。

边界表（可持续 goodput/offered ≥ 0.95，3 seeds 全部满足）：

| workload | TTFT | fp16（A2） | int4（A2） | packed（A2） | k8v4 | 4bit_nc | fp8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random60 | 250 | 30 | NONE | 30 | 待填写 | 待填写 | 待填写 |
| Random60 | 500 | 35 | 35 | 35 | 待填写 | 待填写 | 待填写 |
| Random60 | 1000+ | 35 | 40 | 40 | 待填写 | 待填写 | 待填写 |
| ShareGPT300 | 250 | 45 | 35 | 40 | 待填写 | 待填写 | 待填写 |
| ShareGPT300 | 500+ | 45 | 40 | 40 | 待填写 | 待填写 | 待填写 |

附：P99 TTFT/TPOT（每个 cell 的 `reported_ttft_p99_ms`/`reported_tpot_p99_ms`）、
TPOT 相对 fp16 开销、容量-吞吐 Pareto 点。

## 4. Qwen3.5-9B NIAH 重跑（attempt `niah-fixed-9b-20260807`）

状态：**排队中**（54 cells = 3 alloc × 3 seeds × 3 depths × 2 lengths）。

## 5. 推理基准（attempt `reasoning-20260807`）

状态：**排队中**（15 cells = 3 bench × 5 alloc × 1 seed）。
子集：gsm8k test 前 200、mmlu all/test 前 500、aime25 全 30；
抽取：gsm8k=最后一个数字、mmlu=最后一个 A-D、aime25=最后一个整数。
