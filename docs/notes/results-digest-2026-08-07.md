# 2026-08-07 补跑结果消化（模板/进度表）

> 本文件按阶段填写；每阶段完成并下载、哈希核对、git 提交后更新对应小节并删除
> “待填写”占位。所有数字必须可溯源到 `results/quality/` 下原子 JSON + sha256。

## 1. NIAH 重跑（max_tokens=256，attempt `niah-fixed-20260807`）

状态：**完成**（90/90 cells，0 失败，90 个 JSON 哈希全部匹配）。
判定：每 cell 3 needles；主指标 accuracy（code 出现在 answer），
诊断指标 hit_final（`</think>` 后出现）/hit_think；paired Δ vs fp16（18-cell t-CI）。

| 分配 | cell 均值 | 总体 needle | hit_final | Δ vs fp16 [95% CI] |
|---|---:|---:|---:|---:|
| fp16 | 0.9630 ± 0.1078 | 0.9630 | 0.9074 | — |
| uniform int4 | 0.9815 ± 0.0786 | 0.9815 | 0.9074 | +0.0185 [−0.0206, +0.0576] |
| packed per-layer | 0.9815 ± 0.0786 | 0.9815 | 0.9259 | +0.0185 [−0.0206, +0.0576] |
| turboquant_k8v4 | 0.9630 ± 0.1078 | 0.9630 | 0.8519 | 0.0000 [−0.0804, +0.0804] |
| turboquant_4bit_nc | 0.9444 ± 0.1278 | 0.9444 | 0.8889 | −0.0185 [−0.0875, +0.0505] |

结论：max_tokens=256 后 `<think>` 截断伪影基本消除（带 `<think>` 标签的 needle 从 28
降到 5/6/5/10/11，且这些案例也多数已完整输出答案）；所有分配相对 fp16 的配对
95% CI 均包含 0；packed 点估计最优，TurboQuant 4bit_nc 点估计最低但无统计显著差异。
`hit_final`（严格最终答案）保持 R4/R5 的排序：packed 0.9259 > fp16/uniform 0.9074 >
4bit_nc 0.8889 > k8v4 0.8519——与旧协议数值一致，说明旧协议虽截断但配对比较方向仍成立。

## 2. RULER 子集（attempt `ruler-subset-20260807`）

状态：**v1（官方 tokens_to_generate）完成 70/70**；v2（max_tokens=256）排在 serving
门禁之后（attempt `ruler-subset-20260807-v2-256`）。
协议：官方生成器 `c3f5e3b4`，noise haystack，20 samples/task/length，
官方 `string_match_all`，单 seed（greedy）。

| task | L | fp16 | uniform | packed | k8v4 | 4bit_nc |
|---|---:|---:|---:|---:|---:|---:|
| niah_single | 4096 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| niah_single | 8192 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| niah_multikey | 4096 | 100.0 | 100.0 | 100.0 | 95.0 | 95.0 |
| niah_multikey | 8192 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| niah_multivalue | 4096 | 100.0 | 100.0 | 100.0 | 93.75 | 93.75 |
| niah_multivalue | 8192 | 98.75 | 98.75 | 98.75 | 100.0 | 98.75 |
| niah_multiquery | 4096 | 100.0 | 95.0 | 100.0 | 98.75 | 98.75 |
| niah_multiquery | 8192 | 85.0 | 80.0 | 85.0 | 80.0 | 85.0 |
| vt | 4096 | 91.0 | 89.0 | 89.0 | 91.0 | 89.0 |
| vt | 8192 | 96.0 | 99.0 | 99.0 | 95.0 | 96.0 |
| cwe | 4096 | 97.5 | 97.0 | 98.5 | 65.5 | 67.0 |
| cwe | 8192 | 97.5 | 97.5 | 98.5 | 97.5 | 97.0 |
| fwe | 4096 | 0.0 | 0.0 | 0.0 | 31.67 | 56.67 |
| fwe | 8192 | 0.0 | 5.0 | 5.0 | 61.67 | 71.67 |

**方法学披露（必须写入论文）**：v1 使用官方 RULER 的 `tokens_to_generate`
（NIAH 128 / VT 30 / CWE 120 / FWE 50）。抽查显示 fp16 的 FWE=0 全部是
`<think>` 推理消耗完预算、未输出答案的**协议伪影**（TurboQuant 空 think 后直接输出
答案所以分数反而高）；VT 部分 miss 也是 think 截断与真实漏词混合；multiquery 8K
的 3/20 miss 同为 think 截断。**v1 只作为协议敏感性数据，不得用于“TurboQuant
优于 fp16”类结论**。已安排 v2（max_tokens=256，70 cells）在 serving 门禁后补跑。

结论（v1，待 v2 确认）：packed 在 NIAH 系列与 fp16/uniform 持平且 CWE 点估计最高；
TurboQuant 在 CWE 4K 出现明显掉分（65.5/67.0 vs 97.5），与 vLLM TurboQuant 研究的
“推理任务精度回落”方向一致，但其中含 think 截断成分，需 v2 定量。

**FWE 三方法重跑（max_tokens=256，attempt `ruler-fwe-fixed-20260807`，6/6 完成）**：

| 分配 | FWE 4K | FWE 8K |
|---|---:|---:|
| fp16 | 15.0 | 41.67 |
| uniform int4 | 28.33 | 61.67 |
| packed per-layer | 20.0 | 55.0 |

**解读（必须如实写进论文）**：即使预算提升到 256 token，逐样本抽查显示 miss 仍几乎
全部是 `<think>` 思考未在预算内输出答案（fp16 4K 20 样本中 19 个如此）；uniform 的部分
“命中”来自思考文本顺带写出目标词，并非最终答案正确。因此：
（a）FWE 的绝对分数仍是“思考型模型 + 固定预算”的协议伪影；
（b）量化列高于 fp16 的排序**不是质量信号**，禁止用于任何结论；
（c）若需可解读的 FWE，后续应使用更大预算（如 1024）或禁用 thinking 后重跑。

**FWE 禁用 thinking 重跑（`enable_thinking=False` 经 Qwen3.5 chat template 包装，
max_tokens=256，attempt `ruler-fwe-fixed-nothink-20260807`，6/6 完成）**：

| 分配 | FWE 4K | FWE 8K |
|---|---:|---:|
| fp16 | **93.33** | 100.0 |
| uniform int4 | 88.33 | 100.0 |
| packed per-layer | 88.33 | 100.0 |

解读：禁用 thinking 后 FWE 变为可解读的抽取任务；量化两列与 fp16 在 8K 持平，
4K 点估计低 5 分（88.33 vs 93.33，20 样本，单 seed）；无显著退化结论需更多样本
支撑，但方向上 fp16 ≥ 量化，符合预期。此版本作为 FWE 主报告协议（v1/v2-256
保留为协议敏感性数据）。

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
