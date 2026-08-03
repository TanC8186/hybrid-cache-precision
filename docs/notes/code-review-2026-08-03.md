# Serving 实验代码审查报告（2026-08-03）— audit

> 审查范围：`scripts/exp/vllm_serving_bench.py`、`scripts/exp/gen_allocation.py`、commits `fe1f6a6`/`55174d3`/`525020f`、`results/ablations/bench_lat/{int4,fp16}/*.json`（20 个）、离线 12 JSON、归档 server 日志、notes 数字一致性。
> 模式：**只读审查**。除本报告外未修改任何文件。
> 审查人执行约束：5090 服务器 **ssh 连接被拒**（`connect.bjb2.seetacloud.com:40473` refused，AutoDL 实例已下线/改端口），无法直接读取 `/root/autodl-tmp/extract_metrics.py`、`/root/autodl-tmp/run_bench.sh` 及服务器侧 `scripts/exp/` 副本。服务器侧结论基于归档日志（`results/ablations/bench_lat/logs/*`）与 `rate_*.log` 重构推断，专项结论见 §4。

---

## Material Passport

- Origin Skill: code-review（自定义 audit 任务）
- Origin Mode: audit
- Origin Date: 2026-08-03
- Verification Status: **AUDITED**（只读审查；服务器侧脚本因实例下线未能直接核验，已标注 "unverified"）
- Version Label: audit_v1
- Upstream Dependencies: `fe1f6a6`（20 JSON）、`55174d3`（notes+fig）、`525020f`（server 日志归档）、`serving-benchmark-2026-08-03.md`、`vllm-5090-runbook-2026-08-02.md`、`vendor/vllm-patches/per-layer-kv-dtype.diff`
- 审查基线 commit：`525020f`（工作树 clean）

---

## 1. 发现清单

| # | Severity | 文件 / 行 | 问题描述 | 建议修复 |
|---|---|---|---|---|
| H1 | **HIGH** | `results/ablations/bench_lat/logs/PROVENANCE.md:10`；`server_pl.log`；`docs/notes/serving-latency-throughput-2026-08-03.md:133` | **int4 E2/E3 矩阵的 server 启动日志未归档，且归档日志被误标**。int4 矩阵 rate 点运行于 13:47–14:05（JSON 文件名时间戳），而归档的 `server_pl.log` 启动于 **14:30:58**（`HTTP server started` 14:31:56）——该实例在矩阵结束后才启动，**不可能服务过矩阵**。PROVENANCE.md 把它标为 "E2/E3 吞吐-延迟矩阵使用的 server"，这是**错误标签**。真正服务 int4 矩阵的 server（~13:44 启动）启动日志从未入库。容量 2,701,721 tokens 来自同配置的**二次重启**（很可能是发现 provenance 缺口后补录），数值大概率一致（同 config 内存确定性），但**非矩阵第一手证据**。fp16 侧 `server_fp16.log`（14:07 启动，矩阵 14:16–14:29）是**第一手、无此问题**。 | PROVENANCE.md 明确标注 `server_pl.log` 为"同配置补录，非矩阵实例"；若 5090 仍可达，重跑 int4 矩阵并归档其真实 server 日志；否则在 notes 容量口径处降级为"第二手（同配置复现）"。 |
| H2 | **HIGH** | 服务器 `/root/autodl-tmp/run_bench.sh`、`/root/autodl-tmp/extract_metrics.py`、`/tmp/g_analysis/{parse_all.py,make_figs.py}` | **E2/E3 编排/解析/画图脚本从未入库**。本地仓库无 `run_bench.sh`/`extract_metrics.py`；`parse_all.py`/`make_figs.py` 在 `/tmp/g_analysis`（notes §7 明示"临时目录，未入库"）。服务器现已不可达 → **产生论文级证据的确切脚本链不可复现**。当前只能靠归档的 `rate_*.log` Namespace 转储 + server 日志反推命令。违反 CLAUDE.md 可复现性契约（commit-before-run 的代码侧）。 | 将编排/解析脚本移入 `scripts/` 或 `results/ablations/bench_lat/logs/` 并 commit；补一个 manifest 记录每个 rate 点的完整 CLI。 |
| H3 | **HIGH** | `results/ablations/bench_lat/{int4,fp16}/*.json`（全部 20 个）；`results/ablations/serving_bench_20260803/*.json`（12 个） | **JSON provenance 字段缺口（warmup_n 缺口为真，且不止 warmup_n）**。两类 JSON 均未记录：`warmup_n`/`num_warmups`、`seed`（E2/E3 实际 seed=0，仅存在于 `rate_*.log`）、server 配置（`kv_cache_dtype_per_layer`/`gpu_memory_utilization`/`max_model_len`）、commit hash、vLLM 版本。离线 12 JSON 有 `seed`/`max_len`/`gpu_memory_utilization` 但无 `warmup_n`/commit；E2/E3 20 JSON 连 seed/协议都没有。离线 bench 的后果已被 note §4 承认：warmup-5 与 warmup-120 版无法从文件区分，无 warmup 的 seed42=1575.9 离群原版已覆盖丢失。 | `vllm_serving_bench.py` 输出加 `warmup_n`、`commit_hash`、`model`、`vllm_version`、协议 flag；E2/E3 侧在 `vllm bench serve --metadata`（或 sidecar manifest）记录 `num_warmups`/`seed`/server 参数。 |
| M1 | **MEDIUM** | Makefile `bench` target；`scripts/run.sh`；归档 server 日志 | **规则 2 违反：serving bench 未走入口**。E2/E3 用裸 `vllm serve` + `vllm bench serve` 命令行（design doc §工具链命令 即如此），离线 bench 用 `python scripts/exp/vllm_serving_bench.py` 直跑——均未走 `make run`/`scripts/run.sh`，未生成 run.sh 的 provenance bundle（`git_commit`/`env_probe.txt`/`seeds.txt`）。Makefile `bench` target 指向 `scripts.bench.*` 模块，但该模块不存在对应实现（`scripts/bench/` 只有 `memory.py`/`kv_mem.py`）。 | 为 serving bench 建立 `scripts/run.sh`/Makefile 入口（或至少 wrap 进 `scripts/bench/throughput.py`），让每次跑自动固化 commit/env/seed。 |
| M2 | **MEDIUM** | `docs/notes/serving-latency-throughput-2026-08-03.md:112`（§4.3） | **"GDN state ≈ 60% KV 预算"数字自相矛盾**。按 notes 自身口径：GDN state = 19,537,920 B/请求，int4 server KV 内存 20.08 GiB。在**观测到的满并发 400**（R=75）下：400 × 18.63 MiB / 20.08 GiB = **≈36%**，不是 60%；60% 只对应**理论最大并发 659.6x**（659.6 × 18.63 MiB / 20.08 GiB ≈ 59.8%）。note 文字"在满并发（R=75 观测到 400 并发）下 ≈60%"把"观测 400"与"理论 659.6"两个口径混在一起，机制解释的量化基础是错的。 | 修正为"在理论最大并发 659.6x 下 GDN ≈60%；观测峰值 400 并发下实际 ≈36%"，或直接改用稀释比 2.2456/3.878 ≈ 58% 表述。 |
| M3 | **MEDIUM** | `docs/notes/serving-latency-throughput-2026-08-03.md:37,131`（§2/§6） | **E2/E3 单 run/rate、无 3-seed（规则 6 偏离）**。每 rate 点仅 1 run、`seed=0`，无 mean±std。notes 诚实披露（"无 3-seed，跨点对比带调度噪声"），但作为"论文 supplement 级证据"（§性质声明）的 E2/E3 矩阵，这偏离 CLAUDE.md 的 headline=3-seed 硬规则。TPOT 反转、SLO 边界（50 vs 40 req/s）等关键结论均为单 run。 | headline 前补 ≥3 seed；至少给 int4/fp16 各补 2 次 R=40/50/75 复测确认 SLO 边界。 |
| M4 | **MEDIUM** | `configs/env/remote_5090.yaml:10-12`；`docs/notes/vllm-5090-runbook-2026-08-02.md:86` | **权威环境 config 过期**。`remote_5090.yaml` 写 CUDA 12.x / Docker / vLLM digest / 7B，实际是 CUDA 13 / 预编译 wheel / Qwen3.5-2B。CLAUDE.md 规定"硬件/评测声明以 `configs/env/*.yaml` 与运行 provenance 为准"，该文件与事实不符 → 任何人按它复现都会踩坑。runbook 待做项里已自认。 | 更新为实际环境（CUDA 13、wheel 安装、2B 模型、`VLLM_USE_FLASHINFER_SAMPLER=0`）。 |
| L1 | LOW | `docs/notes/serving-latency-throughput-2026-08-03.md:23-28,61`（§1/§2.2） | **notes 表格 3 个 cell 与 JSON 有 ≤0.1 舍入出入**（与"无出入"声明不符）：fp16 R=4 TTFT p99 notes 119.3 vs JSON 119.250→119.2；fp16 R=75 TTFT mean notes 1966.5 vs JSON 1966.445→1966.4；int4 R=20 TPOT p99 notes 11.53 vs JSON 11.525→11.52。§1 交叉核验表 R=1 用整值 116/125 vs 矩阵表 115.9/124.8。不影响任何结论，但说明表格部分为转录而非逐 cell 重算。 | 以 `parse_all.py` 直接生成表格（保留原始精度），或把 round 规则写进脚本。 |
| L2 | LOW | `scripts/exp/vllm_serving_bench.py:55-73` | `build_requests` 的 "prompt ≈ max_len/5" 是粗估计：`target//40` 假设每个 phrase 恰 ~40 token，但 phrase 由 4 个随机 vocab 词 + 固定 SEED_TEXT 组成，token 数随 tokenizer 浮动，实际 input 长度 ~800–1200 token（max_len=4096 时），input:output 并非严格 1:4。不影响离线 bench 结论（都远小于 max_len），但 docstring 声称的精确比例不成立。 | docstring 改为"约 max_len/5（近似）"；或在 JSON 记实际 token 数（`total_input_tokens`）。 |
| L3 | LOW | `scripts/exp/gen_allocation.py:33-42` | 敏感度公式 `(ppl-p8)/(p2-p8)` 边界未保护：`p2==p8` 时除零；且 `layer3_2bit` ppl(13.195) < `all_8bit` ppl(13.633) → **负敏感度**（2-bit 量化单层反而 ppl 更低，数据异常），负值会翻转"越高越保护"的语义。该脚本未被 E2/E3 使用（E2/E3 用硬编码 `DEFAULT_ALLOC`），但作为分配生成工具存在正确性隐患。 | 除零保护 + 负敏感度 clamp 到 0 + 提示数据异常。 |
| L4 | LOW | `vendor/vllm-patches/per-layer-kv-dtype.diff`；`docs/notes/vllm-5090-runbook-2026-08-02.md:80-82` | **int2/int3 dtype 已注册但无内核**。patch 加了 `int2/int3_per_token_head` 到 `CacheDType`/`KVQuantMode`/dtype 映射，但 Triton 内核分支未实现（runbook 自认待做）。`--kv-cache-dtype int2_per_token_head` 能通过配置校验、在 kernel launch 时才崩——潜伏陷阱，审稿人跑会误导。 | 未实现内核前，从公开 CLI 隐藏 int2/int3（或在校验时报"未实现"）。 |
| N1 | NOTE | `results/ablations/bench_lat/int4/rate_*.log`（line 3 WARNING） | **E2/E3 请求非 greedy 采样**。`vllm bench serve` 未设 `--temperature=0`（warning 明示默认由 server 决定），生成使用随机采样 → 同 seed 复跑输出 token 不同，TTFT/TPOT 带采样噪声。`total_output_tokens=51200` 精确 = 400×128 说明无提前 EOS，指标未受影响，但复现时注意。 | 复现命令加 `--temperature=0`（或记录采样配置）。 |
| N2 | NOTE | 本报告 §4.3 | **服务器与本地脚本 drift 无法直接核验**：5090 下线，`/root/autodl-tmp/MLSys_Research/scripts/exp/` 与本地 diff 不可行。基于归档证据推断：E2/E3 走的是 vLLM 自带 `vllm bench serve`（与本地 `vllm_serving_bench.py` 是不同工具），离线 12 JSON 字段与本地脚本输出结构完全吻合 → 离线路径与本地无 drift；编排脚本本身未入库（见 H2）。 | 恢复实例后补 diff 核验；或接受推断结论。 |
| N3 | NOTE | `configs/bench/throughput.yaml` 与设计 | E2/E3 实际负载（`random` 固定 input=1024/output=128）偏离 design doc（变长 512–2048/128–256）与 `throughput.yaml`（`input_output_ratio:0.2`、`concurrency:16`）——设计/配置与执行不一致，notes 已诚实披露固定长度。SLO 定义（TTFT p99<2000/TPOT p99<200）本身一致。 | 执行前更新 config 与设计一致，或注明偏离。 |

---

## 2. 数字一致性核查（notes 表格 vs JSON 重算）

**全部 20 个 E2/E3 JSON 逐字段重算**（`request_throughput`/`output_throughput`/`mean_ttft`/`p99_ttft`/`median_tpot`/`p99_tpot`/`max_concurrent_requests`）与 `serving-latency-throughput-2026-08-03.md` §2.1/§2.2 对比：

- **19/20 rate 点完全一致**；唯一出入为 L1 的 3 个 ≤0.1ms cell（舍入）。
- SLO 判定复核：int4 R=50 TTFT p99=1574.1<2000 ✓ / R=75=4036.7>2000 ✗ → 最大 SLO 50 req/s；fp16 R=40=566.3 ✓ / R=50=2081.3 ✗ → 40 req/s；50/40=+25% ✓。
- 容量复核：`server_pl.log` 2,701,721 / `server_fp16.log` 1,203,106 = 2.2456x ✓（但见 H1，int4 侧非矩阵第一手）。
- 离线 12 JSON 3-seed mean±std 全部与 `serving-benchmark-2026-08-03.md` §2 一致（含 default_alloc TPOT p99 124.7±35.7、seed42=159.3/seed2026=88.0、rel_std 28.6%）。

**结论：数据层数字可信；provenance 层有 H1–H3 缺口。**

---

## 3. Provenance 缺口总结 + 修复优先级

| 缺口 | 影响面 | 优先级 |
|---|---|---|
| 离线 bench JSON 无 `warmup_n`（12 文件，warmup-5/120 无法区分） | 已造成 seed42 离群原版不可恢复 | **P0** |
| E2/E3 JSON 无 `num_warmups`/`seed`/server 配置/commit（20 文件） | 协议仅存在于 `rate_*.log`，日志一旦丢即不可复原 | **P0** |
| `run_bench.sh`/`extract_metrics.py`/`/tmp/g_analysis/*` 未入库 | 论文级证据的确切脚本链不可复现 | **P0** |
| int4 矩阵 server 日志未归档 + `server_pl.log` 误标 | 容量第一手证据链断裂 | **P1** |
| `remote_5090.yaml` 与实际环境不符 | 复现/审稿人按 config 走会踩坑 | **P1** |
| notes 3 处舍入 cell + §1 整值 vs 矩阵表不一致 | 轻微，影响"无出入"声明可信度 | **P2** |

---

## 4. 专项问题回答

### 4.1 JSON 的 warmup_n 字段缺口是否真实存在？
**是，真实存在，且是双重缺口。**
- 离线 bench：`results/ablations/serving_bench_20260803/*.json`（12 个，由 `vllm_serving_bench.py` 产出）**均无 `warmup_n`**（已逐一核验 keys）。`serving-benchmark-2026-08-03.md` §4 所述"无法区分 warmup 5/120"为真；无 warmup 的 seed42 离群原版确已被 warmup-120 版就地覆盖。
- E2/E3：`results/ablations/bench_lat/{int4,fp16}/*.json`（20 个，由 `vllm bench serve` 产出）**无 `num_warmups` 字段**。实际协议 `num_warmups=0`（`rate_*.log` Namespace 确认），故 E2/E3 侧 warmup 协议本身统一（0），**风险低于离线 bench**，但该信息仅存于日志、不在 JSON。

### 4.2 extract_metrics.py 是否引入了新 bug？
**无法直接核验（服务器下线，ssh 拒绝连接），归档数据无支持 bug 的证据。**
- E2/E3 的 20 个 JSON 是 `vllm bench serve` 的原生 `save_result` 输出（字段集与 vLLM 内置 benchmark 一致），不是 extract_metrics.py 转录的。因此即便它存在解析逻辑，指标源头是 vLLM 官方工具。
- 我独立从 20 JSON 重算的每个数字与 notes/`rate_*.log` 尾部 `Serving Benchmark Result` 表格完全一致 → 未见数值提取错误。
- 未归档的 `run_bench.sh`/`extract_metrics.py` 本身是 H2 缺口；若需确认其中有无 bug，须等实例恢复后读取。**当前结论：无证据表明引入新 bug，但不可证伪。**

### 4.3 服务器脚本与本地是否 drift？
**E2/E3 路径与本地是"不同工具"，非 drift；离线路径无 drift 证据；编排脚本未入库。**
- E2/E3 用 vLLM 自带 `vllm serve` + `vllm bench serve`，**不是**本地 `scripts/exp/vllm_serving_bench.py` → 不存在"同一脚本服务器版 vs 本地版"的 drift，而是**工具链本身与仓库脚本体系分离**。
- 离线 12 JSON 的字段（`allocation`/`kv_args`/`num_reqs`/`max_len`/`max_tokens`/`gpu_memory_utilization`/`seed`/kv 字段）与本地 `vllm_serving_bench.py`（commit `5dcb4a5`）输出结构完全吻合，且本地工作树 clean、脚本最后改动（11:49）早于离线跑（12:00–12:31）→ 离线路径与本地无 drift。
- 服务器 `/root/autodl-tmp/MLSys_Research/scripts/exp/` 副本无法访问，无法逐文件 diff；`run_bench.sh`/`extract_metrics.py` 在本地仓库不存在（H2）。

---

## 5. 审查统计

- **CRITICAL：0**
- **HIGH：3**（H1 server 日志误标/缺失、H2 编排脚本未入库、H3 JSON provenance 字段缺口）
- **MEDIUM：4**（M1 入口规则违反、M2 60% 数字矛盾、M3 无 3-seed、M4 env config 过期）
- **LOW/NOTE：7**（L1–L4 + N1–N3）

**总体判断**：数字层（矩阵/表格/统计）可信度良好，20 个 rate 点重算仅 3 个 ≤0.1ms 舍入出入；**风险集中在 provenance 层**——E2/E3 的编排脚本、int4 矩阵 server 日志、以及所有 JSON 的 warmup/seed/server 配置均未在文件内固化，且服务器实例已下线，恢复成本随实例销毁不可逆上升。建议优先按 §3 P0 顺序补救。
