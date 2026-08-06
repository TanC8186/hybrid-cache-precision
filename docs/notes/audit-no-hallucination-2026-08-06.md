# R4/R5 实验幻觉审计报告（2026-08-06，服务器已关机，纯本地）

> 审计范围：`scripts/eval/kv_quality_retrieval.py`、`analyze_r4_quality.py`、
> `analyze_r5_turboquant.py`、`scripts/exp/hybrid_premise.py`（改动）、runner 脚本、
> `results/quality/` 全部产物、论文/笔记中的结论表述。
> 方法：脚本逐行审读 + 92 个 JSON 哈希核对 + 独立重算全部数字 + 样本级 case 检查 +
> 结论逐条溯源。

## 0. 结论摘要

- **未发现数字幻觉**：论文/笔记中的每个数字都能追溯到原始 JSON/CSV，独立重算完全一致。
- **发现 1 个方法学披露缺口（必须修正）**：NIAH “accuracy” 是 32-token 上限 + Qwen3.5
  thinking 输出的协议伪影——全部 28 个 miss 都是 `<think>` 推理被截断，0 个真实检索失败。
- **发现 1 个证据表述偏差（已修正）**：packed 样本的 JSON `config_effect` 缺少 KV group
  结构细节（内省路径失效），真正的 group 证据在引擎日志；文档需改为引用日志标记。
- 若干脚本健壮性/溯源小问题（不影响本次数字，已记录并部分修复）。
- 证据状态：R4/R5 全部保持 **ANALYZED**，无 VERIFIED 冒充；“0 失败”“54/54”“36/36”
  均经日志核实。

## 1. 脚本审计

### 1.1 `kv_quality_retrieval.py`

- **config-effect 回退（弱验证）**：packed 分支在拿不到 KV group 结构时仍可 `ok=True`
  （只验 dtype/per-layer/flag）。本次 18/18 packed 样本都走了该回退；group 结构由引擎日志
  证明（见 §3）。建议后续把 `group_structure_verified` 显式写入 JSON。
- **hit 判定**：`code in answer.upper().replace(" ","")` 合理；代码 8 位、32 字符字母表，
  误命中概率可忽略；已抽查答案确认。
- **rng 确定性**：`random.Random(seed*1000003 + depth*31 + len*17)` 保证同一 cell 跨分配
  的 prompt/code 完全一致（已验证 fp16/uniform/packed 同 cell codes 相同、answers 相同）。
- **长度口径**：`max_len` 是**填充词数**而非 token 数；depth 按词位置计算；JSON 未记录
  实际 token 长度。属于近似口径，需披露（见 §3）。
- **生成上限**：`max_tokens=32, temperature=0`。Qwen3.5 默认输出 `<think>` 推理，32 token
  会被推理消耗殆尽 → 这是 miss 的唯一原因（见 §3）。

### 1.2 `analyze_r4_quality.py` / `analyze_r5_turboquant.py`

- PPL 读取已按 `bits` 过滤（此前“last wins”bug 已修复）；NIAH 按 cell 配对。
- R4 analyzer 对 NIAH 缺少 18-cell 强制保护（交集会静默缩小）；R5 analyzer 有 18-cell
  保护。本次数据均 18/18，数字无影响。建议给 R4 analyzer 补同样保护。
- t 临界值表覆盖 df≤30，n=3（PPL）与 n=18（NIAH）正确。

### 1.3 `hybrid_premise.py`（PPL 路径）

- 已确认 `KVQuantizer(bits=16)` 是 **FP16 直通**（不量化，字节按 fp16 计）；
  `layer_bits={23:16}` + base bits=4 的语义正确（L23 保护、其余 int4）。
- 这是 transformers 自定义缓存**模拟**，不是 vLLM kernel 路径；与 canonical 的
  `byte_budget_3seed.log` 同协议但 corpus 快照不同（fp16 均值 11.4827 vs 11.4832，
  差异 <0.01%）。
- `fp16.csv.seeds.csv` 存在重复行（bits=16 被计算两次，值相同）——无害但冗余。
- PPL 产物未记录 corpus sha（本次语料 sha256 = `f7c3d825fe137ae727909932428eec25ff1b05e685b713eefc4efb289bfd49d0`）。

### 1.4 Runner 脚本

- `set -euo pipefail`、`--resume`、失败即停、[OK]/[DONE] 标记齐全；日志无 FAIL。

## 2. 数据审计

- 92 个服务器 JSON 的 SHA-256 与 sidecar 全部匹配（0 mismatch）；本地仅多 2 个分析 JSON。
- 样本完整性：R4 NIAH 54/54（3 alloc × 18 cell，unique=18/alloc）、R5 36/36（2 alloc ×
  18）、R5 MVEx 2/2；无重复 cell、无缺失。
- 独立重算（不调用 analyzer）与已提交分析完全一致：

| 指标 | 独立重算 | r4/r5-analysis.json |
|---|---:|---:|
| fp16 PPL mean | 11.4827 | 11.4827 |
| uniform PPL mean | 11.6811 | 11.6811 |
| packed PPL mean | 11.5985 | 11.5985 |
| packed vs uniform PPL Δ | −0.0826 [−0.1767, +0.0115] | 同 |
| fp16/uniform/packed NIAH | 0.9074 / 0.9074 / 0.9259 | 同 |
| turboquant k8v4 / 4bit_nc NIAH | 0.8519 / 0.8889 | 同 |
| k8v4 vs fp16 Δ | −0.0556 [−0.1408, +0.0297] | 同 |

- 一致性抽查：R5 MVEx 与矩阵同 cell（seed7/d50/L2048）accuracy 均为 1.00；
  PPL MVEx packed seed7 = 9.9311 = 矩阵 packed seed7。
- 日志：R4 niah [OK] 54、R5 [OK] 36、[DONE] 标记存在、[FAIL] 计数 0。

## 3. 方法学审计（重点披露）

1. **NIAH 绝对准确率是协议伪影**：全部 28 个 miss（fp16 5、uniform 5、packed 4、
   k8v4 8、4bit_nc 6）的答案都以 `<think>` 开始，在 32 token 内未到达答案；
   **0 个是真正检索失败**。因此：
   - 0.85–0.93 的 accuracy 不能解读为“检索能力差异”；
   - 配对比较仍然公平（同 prompt、同 code、同生成条件）；
   - 后续应把 `max_tokens` 提到 ≥128 或加 “Answer directly” 指令重跑（P1）。
2. **length/depth 为词数近似**，非 token；论文若报告 NIAH 需注明。
3. **config-effect 证据**：packed 样本 JSON 缺 group 结构（内省路径
   `core_engine.model_executor...` 在离线引擎不可用）；引擎日志
   `logs/r4-20260806.niah.log` 有 17/18 条 “Using packed per-layer page groups for
   6 full-attention layers and 18 Mamba layers in 2 Mamba groups”（第 18 个样本即
   seed7/d50/L2048 为前台 MVEx，日志在会话记录中；其 JSON config_effect 通过
   dtype/per-layer/flag 校验）。R5 TurboQuant 36/36 均有
   `kv_cache_dtype=turboquant_*` 引擎日志标记。文档表述已从“JSON+group 校验”改为
   “引擎日志验证”。
4. **PPL 路径**：transformers 自定义缓存模拟（per-token 粒度）而非 vLLM kernel；
   canonical 一致性成立（差异 <0.01%），但 corpus 快照不同，需记录 sha。

## 4. 结论溯源

- 论文 Eval §6 Table 4b、mainline §4.4、外部 baseline 文档中的数字与 r4/r5-analysis
  完全一致（逐项 grep 核对）。
- 状态标签：R4/R5 全部为 ANALYZED，无 VERIFIED 冒充；论文明确“serving 独立复现未过”。
- 唯一需要修正的表述：NIAV accuracy 绝对值与“检索验证”措辞（已修正为带披露）。

## 5. 修正与建议

**已修正（本审计提交）**：
- 文档/论文补充 NIAH 32-token/thinking 截断披露与词数长度口径；
- config-effect 证据改为引用引擎日志；
- 补记 corpus sha（PPL）与 17/18 日志标记说明。

**建议下次运行前（P1）**：
- NIAH `max_tokens` ≥128 或禁用 thinking 后重跑，得到可解读的检索准确率；
- retrieval JSON 记录 `context_words`/实际 token 长度与 `group_structure_verified`；
- analyzer 对 NIAH 强制 18-cell 完整性检查；
- PPL 产物记录 corpus sha 与 hybrid_premise commit。
