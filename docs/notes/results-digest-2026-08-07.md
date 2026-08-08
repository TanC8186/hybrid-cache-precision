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

**最终答案指标（post-hoc，`hit_last_section`）**：取生成文本中最后一个 `Answer:`
之后的内容判定 code（无 `Answer:` 时用全文）。对 2B 全部 270 条 needle 与 9B 全部
54 条 needle，`hit_last_section` 与 `hit`（任意位置）**零分歧**；9B 的重复输出
（先答 code 再重复 Question/Answer）不影响判定，最后一次 Answer 块仍包含正确 code。
结果：`results/quality/niah-fixed-final-answer.json`、
`results/quality/niah-fixed-9b-final-answer.json`。

## 2. RULER 子集（attempt `ruler-subset-20260807`）

状态：**v1（官方 tokens_to_generate）完成 70/70**；**v2（max_tokens=256）完成 70/70**
（attempt `ruler-subset-20260807-v2-256`，0 失败，70 个 JSON 哈希全匹配）。
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

**v2（max_tokens=256）要点**：NIAH single 100 全列；multikey/multivalue 4K 仅 TQ
−5/−6.25；multiquery 8K packed 最高（88.75，fp16 86.25，k8v4 80）；VT 4K TQ
−1..−5、8K 基本持平；**CWE 4K TQ 掉分在 256 token 下仍存在（k8v4 69.0 / 4bit_nc
70.0 vs fp16 97.5，−28.5/−27.5）→ 该掉分是真实质量退化而非截断伪影**，与 vLLM
TurboQuant 研究一致；FWE 仍为 think 伪影（主协议用 no-think 版）。

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
max_tokens=256，attempt `ruler-fwe-fixed-nothink-20260807`，10/10 完成，全 5 分配）**：

| 分配 | FWE 4K | FWE 8K |
|---|---:|---:|
| fp16 | **93.33** | 100.0 |
| uniform int4 | 88.33 | 100.0 |
| packed per-layer | 88.33 | 100.0 |
| turboquant_k8v4 | 93.33 | 100.0 |
| turboquant_4bit_nc | 96.67 | 100.0 |

解读：禁用 thinking 后 FWE 变为可解读的抽取任务；量化两列与 fp16 在 8K 持平，
4K 点估计 fp16/k8v4 93.33、4bit_nc 96.67、uniform/packed 88.33（20 样本，单 seed）；
无显著退化结论需更多样本支撑。此版本作为 FWE 主报告协议（v1/v2-256 保留为
协议敏感性数据）。

## 3. TurboQuant/FP8 serving（protocol-v3）

状态：**排队中**（门禁 MVEx+Pilot → 人工审阅 → Formal）。
Attempts：`r5-tq-v3-{random60,sharegpt300}-{mvex,pilot,formal}-20260807`。
输出根：`/root/autodl-tmp/r5-serving-20260807`。

边界表（可持续 goodput/offered ≥ 0.95，3 seeds 全部满足）：

| workload | TTFT | fp16（A2） | int4（A2） | packed（A2） | k8v4 | 4bit_nc | fp8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random60 | 250 | 30 | NONE | 30 | NONE | NONE | 30 |
| Random60 | 500 | 35 | 35 | 35 | 35 | 35 | 35 |
| Random60 | 1000 | 35 | 40 | 40 | 40 | 35 | 40 |
| Random60 | 2000/3000 | 35 | 40 | 40 | 40 | 40 | 40 |
| ShareGPT300 | 250 | 45 | 35 | 40 | 45 | 40 | 45 |
| ShareGPT300 | 500+ | 45 | 40 | 40 | 45 | 40 | 45 |

附：P99 TTFT/TPOT（每个 cell 的 `reported_ttft_p99_ms`/`reported_tpot_p99_ms`）、
TPOT 相对 fp16 开销、容量-吞吐 Pareto 点。

**E4 Formal 补跑（2026-08-08，108/108 完成）**：Random60 45 + ShareGPT300 63
全部 `completed_validated`、0 进程失败；高负载样本（ShareGPT r45–r50 的
TQ/FP8）按协议 `count_as_slo_miss` 计入失败（fp8 r50 681–891、4bit_nc r50
1171–1407、k8v4 r50 465–747），到达窗口偏差 ≤0.03%，TTFT/TPOT 重算与报告
一致 → 属真实过载测量而非伪影。上表六列合并（A2 三列来自门禁 formal，
TQ/FP8 三列来自本次 Formal）；整体仍为 ANALYZED（独立复现未跑）。
分析：`results/quality/r5-serving-formal-analysis-20260808.json`；
原始 attempt：服务器 `/root/autodl-tmp/r5-serving-20260807/`。
**过程修复**：`analyze_r5_serving.py` 的 `verify_sidecar` 误哈希 `.sha256`
自身（必然 mismatch）→ 改为哈希被保护文件；status.json 按 runner 设计无
sidecar → 改为校验其内嵌 analysis/result sha。

## 4. Qwen3.5-9B NIAH 重跑（attempt `niah-fixed-9b-20260807`）

状态：**完成 54/54，0 失败**（54 个 JSON 哈希全匹配）。

| 分配 | cell 均值 | 总体 needle | hit_final | Δ vs fp16 [95% CI] |
|---|---:|---:|---:|---:|
| fp16 | 1.0000 ± 0.0000 | 1.0000 | 0.8889 | — |
| uniform int4 | 0.9815 ± 0.0786 | 0.9815 | 0.8519 | −0.0185 [−0.0576, +0.0206] |
| packed per-layer | 0.9815 ± 0.0786 | 0.9815 | 0.8333 | −0.0185 [−0.0576, +0.0206] |

结论：9B 上量化两列相对 fp16 的配对 95% CI 均包含 0；`hit_last_section` 与 `hit`
零分歧（重复输出不影响判定）。分析：`results/quality/niah-fixed-9b-analysis.json`。

## 5. 推理基准（attempt `reasoning-20260807`）

状态：**主协议完成 15/15**（attempt `reasoning-20260807-nothink-v2`；0 失败，
30 个 JSON 哈希全部匹配）。另保留两个协议敏感性 attempt，**禁用作主协议**：

- `reasoning-20260807`（thinking 模式 + 小预算 256/128/1024）：4/15 cells 完成
  （fp16 三格 + uniform gsm8k），gsm8k 171–173/200、mmlu 113/500、aime25 30/30
  撞 token 上限 → "最后候选"抽取为截断伪影。
- `reasoning-20260807-nothink`（no-think + 小预算 256/128/1024）：15/15 完成，
  但 gsm8k 95/200、mmlu 462/500、aime25 29/30 仍撞上限，伪影未消除。

主协议：no-think（chat template `enable_thinking=False`）+ 大预算
（gsm8k 1024 / mmlu 512 / aime25 4096）+ 最终答案抽取（取最后一个
`answer`/`result` 标记后的候选；无标记退回全文本最后候选并在 cell 内记录
`extraction_source`）。单 seed（greedy，seed=7），无配对 CI（n=1）。

| bench | n | fp16 | uniform | packed | k8v4 | 4bit_nc |
|---|---:|---:|---:|---:|---:|---:|
| gsm8k | 200 | 0.7600 | 0.6950 | 0.6800 | 0.6750 | 0.6850 |
| mmlu | 500 | 0.5880 | 0.5860 | 0.5960 | 0.6000 | 0.6120 |
| aime25 | 30 | 0.1667 | 0.1000 | 0.1667 | 0.0667 | 0.1000 |

截断与覆盖：gsm8k capped 4–7/200（≤3.5%）；mmlu capped 99–118/500
（约 20–24%，各分配对称）；aime25 capped 20–22/30（4096 预算仍不足，2B 模型
长推导/循环）。最终答案标记覆盖：gsm8k ~58–62%、mmlu ~78–80%、aime25 ~37–50%。

结论边界：
- **gsm8k**：量化列点估计低于 fp16 6.5–8.5pt（packed −8.0pt），单 seed 无 CI，
  方向一致但不足以作统计结论，论文只能写"点估计回退，需多 seed 确认"。
- **mmlu**：量化列与 fp16 持平或略高（packed +0.8pt，4bit_nc +2.4pt）；严格最终
  答案口径仍持平（fp16 0.524，packed 0.512，4bit_nc 0.546）。
- **aime25**：2B 模型在 4096 预算下仍有 2/3 无法完成，各列均近地板，Δ 仅 1–3 题，
  无跨分配信号；论文应披露为预算受限，或仅报告"各列均无显著优势"。

分析：`results/quality/reasoning-nothink-v2-analysis.json`；
原始 cell：`results/quality/reasoning/reasoning-20260807-nothink-v2/`。

## 6. LongBench v1 子集（attempt `longbench-20260807`）

状态：**完成 64/64**（2B 全 5 分配 × 8 任务 + 9B 核心 3 分配 × 8 任务；0 失败，
64 个 JSON 哈希全部匹配）。首次运行曾因 resume 校验缺 model 字段把 9B 当 2B
跳过，已修复（文件名带模型标签 + 校验 model/max_model_len），9B 重跑完成。

协议：官方 LongBench v1 prompt/指标（THUDM/LongBench@`4c4b985bcf`），数据为
Xnhyacinth/LongBench parquet 镜像（原 JSONL revision 已下线）；每任务前 50 样本、
greedy、seed 7、no-think、`max_model_len=16384`（超限中段截断）。指标：
TREC=classification、TriviaQA=qa_f1、SAMSum/GovReport/QMSum/MultiNews=ROUGE-L F、
LCC/RepoBench-P=code_sim（fuzzywuzzy）。

**2B 分数表（每任务 n=50）**

| task | fp16 | uniform | packed | k8v4 | 4bit_nc |
|---|---:|---:|---:|---:|---:|
| TREC | 72.00 | 66.00 | 66.00 | 72.00 | 70.00 |
| TriviaQA | 84.53 | 82.87 | 82.87 | 84.93 | 84.53 |
| SAMSum | 32.36 | 33.39 | 31.10 | 30.25 | 31.62 |
| LCC | 2.40 | 3.36 | 3.06 | 2.62 | 2.06 |
| RepoBench-P | 0.88 | 1.10 | 1.20 | 0.44 | 0.64 |
| GovReport | 31.52 | 31.33 | 31.28 | 31.39 | 30.84 |
| QMSum | 22.30 | 21.66 | 21.21 | 21.54 | 22.25 |
| MultiNews | 23.47 | 23.01 | 23.22 | 23.90 | 22.93 |

**9B 分数表（每任务 n=50；仅核心 3 分配）**

| task | fp16 | uniform | packed |
|---|---:|---:|---:|
| TREC | 72.00 | 72.00 | 74.00 |
| TriviaQA | 88.20 | 89.20 | 87.53 |
| SAMSum | 38.43 | 38.11 | 37.69 |
| LCC | 40.56 | 38.72 | 37.96 |
| RepoBench-P | 37.40 | 35.36 | 32.16 |
| GovReport | 32.10 | 32.12 | 31.82 |
| QMSum | 22.54 | 21.94 | 21.67 |
| MultiNews | 23.34 | 22.53 | 22.46 |

结论边界（单 seed、无 CI，全部为点估计）：
- **2B QA/摘要**：量化列与 fp16 基本持平（多数任务 ±1pt 内）；TREC 上
  uniform/packed 点估计低 fp16 6pt，与 k8v4 的 72 不齐，需多 seed 确认；
  LCC/RepoBench-P 对 2B 接近地板（<4），不可解读。
- **9B**：QA/摘要持平；代码任务量化列点估计一致略低（LCC −1.8/−2.6，
  RepoBench-P −2.0/−5.2），方向一致但单 seed 不可作统计结论。
- 截断（prompt 超 15.8K token 中段截断）各分配完全对称：TriviaQA 12/50、
  SAMSum 5/50、RepoBench-P 15/50、GovReport 10/50、QMSum 23/50、其余 0。

分析：`results/quality/longbench-analysis-20260807.json`；
原始 cell：`results/quality/longbench/longbench-20260807/`。

**LongBench 输出预算审计（2026-08-08，防伪影核查）**：3200 样本中 1268
（39.6%）撞 max_new_tokens，但 **0 个样本含 `<think>` 块**（no-think wrapper
生效，无"思考吃光预算"伪影）。撞限按任务分布：LCC 94%、RepoBench-P 99%、
GovReport 77%、MultiNews 42%，TREC/TriviaQA/SAMSum/QMSum ≈0%；各分配间
撞限率对称（fp16 41% / uniform 41% / packed 40% / k8v4 36% / 4bit_nc 36%），
相对比较不受影响。输出预算与官方 LongBench 协议一致（64/512 等），截断属
协议内行为；抽样确认 GovReport 截断为正常摘要截尾而非思考文本。

## 7. C4/PG19 PPL（attempt `ppl-extra-20260807`）

状态：**完成 12/12**（2B + 9B × fp16/uniform/packed × c4/pg19；0 失败；
24 个 CSV 哈希已核对并生成 `.sha256` 边车）。协议与 Q1 相同
（`hybrid_premise.py --seeds 7,42,2026 --num-seqs 5 --max-len 2048 --chunk 128`）。

| 模型 | corpus | fp16 | uniform | Δ uniform [95% CI] | packed | Δ packed [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| 2B | c4 | 17.5800 | 17.8730 | +0.2930 [+0.2127, +0.3732] | 17.7761 | +0.1961 [+0.1373, +0.2548] |
| 2B | pg19 | 27.1783 | 27.6210 | +0.4428 [+0.2949, +0.5907] | 27.4244 | +0.2462 [+0.0962, +0.3961] |
| 9B | c4 | 12.7287 | 12.8678 | +0.1391 [+0.0847, +0.1935] | 12.8677 | +0.1391 [+0.1051, +0.1730] |
| 9B | pg19 | 18.0016 | 18.2226 | +0.2210 [+0.1207, +0.3213] | 18.2080 | +0.2064 [+0.1172, +0.2955] |

结论：与 Wikitext-2 一致——uniform/packed 相对 fp16 增加约 1.1–1.6%（2B）与约
1.1%（9B），3-seed 配对 CI 不含 0；packed 点估计普遍略优于 uniform
（2B c4 −0.097、pg19 −0.197；9B pg19 −0.015、c4 持平），与"L23 保护无质量回退
代价"的故事自洽。语料：c4 sha `7ee17255...`、pg19 sha `c898ba29...`。
分析：`results/quality/ppl-extra-analysis-20260807.json`；
原始 CSV：`results/quality/ppl-extra/`。

## 8. 容量探针扩展（M3 9B + M4 纯注意力对照）

协议与 VERIFIED A2 gate 完全一致（`inspect_kv_config.py`，max_model_len=4096、
gpu_memory_utilization=0.85、seed=42、`--enforce-eager`；uniform 显式传
`--kv-cache-dtype-per-layer '{}'`）。探针 JSON + sha256：
`results/verified/2026-08-08/capacity-probe-extra/`。

| 模型 | 配置 | tokens | max concurrency | 比例 |
|---|---|---:|---:|---:|
| Qwen3.5-9B | legacy per-layer | 89,088 | 21.75 | — |
| Qwen3.5-9B | uniform int4 | 345,702 | 84.4 | — |
| Qwen3.5-9B | packed per-layer | 287,744 | 70.25 | packed/legacy **3.230**；packed/uniform **0.832** |
| Qwen2.5-7B（纯 attention） | fp16 | 204,512 | 49.93 | — |
| Qwen2.5-7B（纯 attention） | int4 | 769,968 | 187.98 | int4/fp16 **3.765** |

结论：
- A2 在 9B 上比例与 2B VERIFIED gate 几乎逐位一致（3.230 vs 3.232；0.832 vs
  0.833）→ 机制跨规模成立；
- 纯注意力模型 int4/fp16 = 3.765×，接近机制层 3.88×，而混合 2B 端到端只有
  2.245× → GDN state 稀释是混合架构性质，不是模型个例。

过程记录：9B 首轮探针因未传 `--enforce-eager` 触发 ninja 缺失失败（已补装）；
纯注意力探针因 `inspect_kv_config.py` 访问混合架构专属属性
`_kernel_block_sizes` 失败（已改为 `getattr` 防御式读取，从研究仓运行）。

## 9. GSM8K 3-seed（attempt `reasoning-gsm8k-3seed-20260808`）

状态：**完成 15/15**（5 allocs × seeds {7,42,2026}；seed 7 从
`reasoning-20260807-nothink-v2` 哈希复制，seed 42/2026 新跑；0 失败，15 个 JSON
哈希全部匹配）。协议：no-think、greedy、max_tokens=1024、200 samples/seed。

| 分配 | seed 7/42/2026 | mean ± SD | Δ vs fp16 [95% t-CI] |
|---|---:|---:|---:|
| fp16 | 0.760 / 0.755 / 0.755 | 0.7567 ± 0.0029 | — |
| uniform int4 | 0.695 / 0.695 / 0.695 | 0.6950 ± 0.0000 | −0.0617 [−0.0688, −0.0545] |
| packed per-layer | 0.680 / 0.690 / 0.690 | 0.6867 ± 0.0058 | −0.0700 [−0.0915, −0.0485] |
| turboquant_k8v4 | 0.675 / 0.690 / 0.675 | 0.6800 ± 0.0087 | −0.0767 [−0.1025, −0.0508] |
| turboquant_4bit_nc | 0.685 / 0.685 / 0.685 | 0.6850 ± 0.0000 | −0.0717 [−0.0788, −0.0645] |

结论（必须如实写入论文）：**GSM8K 上所有量化分配相对 fp16 的回退是真实的**，
Δ −6.2 至 −7.7pt，3-seed 配对 95% CI 全部不含 0。该负结果与 MMLU/LongBench/
NIAH/PPL 的持平结论并存，论文的"质量门禁"应表述为
"多数任务持平、GSM8K 存在一致但量级有限（约 −6~−8pt）的回退"，
禁止再写全局"近无损"。注意：uniform/4bit_nc 三 seed 完全一致，fp16/packed/k8v4
seed 间波动 ≤0.01（greedy 确定性为主，记录为方法学观察）。

分析：`results/quality/reasoning-gsm8k-3seed-analysis.json`；
原始 cell：`results/quality/reasoning/reasoning-gsm8k-3seed-20260808/`。

## 10. 外部 baseline：KIVI 风格 4-bit KV 量化 PPL（attempt `ppl-external-20260808`）

实现：transformers 5.x HQQ backend 的 KIVI 风格量化（K 逐通道 group32 4-bit、
V 逐 token 4-bit、128-token fp16 residual 窗口），子类化 DynamicCache 只覆写
attention 层 update；Qwen3.5-2B、协议与 canonical 相同（seeds 7/42/2026、
5×2048、chunk 128）。fp16 为同 harness 参考。6/6 cells 完成。

| corpus | fp16 mean | kivi4 mean | Δ [95% t-CI] |
|---|---:|---:|---:|
| wikitext2 | 11.3413 | 11.3452 | +0.0039 [−0.0463, +0.0542] |
| c4 | 17.4645 | 17.4551 | −0.0094 [−0.0654, +0.0466] |
| pg19 | 26.7010 | 26.6955 | −0.0055 [−0.0514, +0.0403] |

结论：**KIVI 风格 4-bit 在三语料上与同 harness fp16 的 Δ 全部 ≈0（CI 含 0）**，
即该外部方法在 PPL 上近无损；对照本工作 uniform int4（同语料相对自身 fp16
约 +1.1–1.7%），外部方法 PPL 略优（预期：逐通道 K + residual 窗口更精确）。
论文须如实报告：质量维度外部方法不劣于本工作；本工作卖点为系统层容量/SLO 与
packed 机制。注意 harness 差异（canonical fp16 11.48 vs 本 harness 11.34，
约 1%），跨 harness 比较以各自的 fp16 为基准。

分析：`results/quality/ppl-external-analysis-20260808.json`；
原始 cell：`results/quality/ppl-external/ppl-external-20260808/`。
