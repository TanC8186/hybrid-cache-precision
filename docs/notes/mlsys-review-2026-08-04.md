# MLSys 对抗式审稿报告（2026-08-04，最大强度）

> 4 维度对抗审稿（data-audit / method / novelty / clarity）+ Area Chair 汇总。**决策：REJECT**。
> 完整审稿 JSON 在 workflow journal：`subagents/workflows/wf_95cc7495-0e7/journal.jsonl`。

## 1. 评分
| 审稿人 | 维度 | 评分 |
|---|---|---|
| data-audit | 数据一致性 | 6/10 |
| method | 实验方法/统计 | 4/10 |
| novelty | 新颖性/意义 | 4/10 |
| clarity | 写作/完整性 | 5.5/10 |
| **AC** | 汇总 | **REJECT** |

## 2. 一致确认的强项（所有审稿人独立验证通过）
- **无造假、可复现**：容量 2.2456x@4K / 3.155x@16K / 2.19x@9B / 3.167x@16384、per-layer ×0.258、capacity model r_s=2.149/3.091、offline TPOT +8.0%/-5.5%、byte-budget PPL，全部从 server 日志 + 原始 JSON + vLLM fork 源码独立复现
- 62 个 JSON 全 completed=400 failed=0；3-seed 是真实 3 次 run（seed 7/42/2026）
- 诚实披露（bug re-basing、E2/E3 单 run 性质、per-layer limitation）= 亮点
- GDN state 数学（1,085,440 B/layer × 18 = 18.63 MiB）从代码形状核对成立

## 3. CRITICAL FAILURES（AC 列 6 条，blocking）
1. **SLO +25% 是 transient/overloaded 伪影 + 单阈值 artifact**：两个 SLO 边界点 offered > goodput（过载态）；阈值从 2000ms 换到 250/500ms 时收益消失（都 cap 在 rate 30 = 0% 收益）。+25% 不是可持续 SLO 容量 claim
2. **Novelty 低于 MLSys bar：无新方法**：用 stock vLLM dtype（int4_per_token_head），无新 quantizer/kernel/scheduler/memory manager；唯一 bespoke 工作（per-layer patch）是 NO-OP bug fix。**唯一出路**：实现 packed per-dtype layout（A2）把 ×0.258 limitation 变成正面贡献，或 rescope + 加压缩 baseline（TurboQuant/KVQuant/KIVI）
3. **Headline 混口径**：mainline 用 single-run +5.2%（38.14/36.26），headline note 用 3-seed +5.1%（37.76/35.94）——无 canonical 来源
4. **GDN dtype 假设未记录**：整个稀释模型依赖 temporal fp32（1,085,440 B/layer），但 config/log 无 mamba_ssm_cache_dtype 记录；若 bf16 则减半 → r_s(4096)=2.62 与实测 2.2456 矛盾（间接支持 fp32 但需直接确认）
5. **Quality 单 seed 无不确定度**：PPL 三个文件互相矛盾（13.86 / 11.67 / 11.03，同一 4-bit 配置绝对 PPL 差 ~26%）；byte-budget 对非 byte-exact（5.4% 差）
6. **投稿不完整**：无 references.bib；RW 全是 [*KIVI*] 占位 + [VERIFY]/community 引用；'first system study' claim 过强（TurboQuant/RotorQuant 有先例）；9B @16384 结果（3.167x）未进论文

## 4. 其他 major（按审稿人）
- **method**：E3 边界点过载（fp16@40 offered 40 vs goodput 35.05）；ShareGPT 只跑 rate 8/16（低负载），与高负载 SLO claim 方向相反，'matches synthetic' 混淆低负载代价与高负载收益；9B 的 SLO 收益消失（两者都 rate 8）
- **novelty**：'GDN 稀释'是线性注意力架构的初等代数推论，非机制发现；2.245x 是稀释后比值 vs fp16，弱于纯 attention 3.88x 且无 KIVI/TurboQuant/KVQuant 对照；equal-byte 排序是 QPruningKV 已发结论的 re-verification；per-layer 失败是 vLLM V1 实现 artifact 非混合架构性质
- **data-audit**：饱和 goodput 单/3-seed 不一致；PPL 三文件；offline TPOT 'warmup120 3-seed' 标签但 fp16 侧是 warmup-5；GDN dtype 无记录；小数值不符（+13.8% vs 重算 +13.6%；0.2581 vs 0.2584；表格舍入；'−6~8%/+8~10%' 范围不含自身端点）
- **clarity**：无 bibliography（critical）；Evaluation 草稿自相矛盾（§7 说单 run 但 3-seed 已完成）；图用 single-run 数据无误差棒（fig2 标注 1574/566 vs 3-seed 1671.5/2163）；陈旧指示（per-layer relabel 已做仍标 pending）；9B 16384 缺失

## 5. AC prioritized fixes（可执行修复清单）
1. **[blocking] 修 E3 方法**：offered-rate 边界 → steady-state 协议（goodput=offered 的可持续 SLO 率 / goodput-under-SLO）；加 threshold sweep {250,500,1000,2000,3000}ms；ShareGPT 跑 rate 30-50 记 seed；Abstract 改为诚实的可持续差距（~+5% 饱和 goodput，3-seed 37.76 vs 35.94）除非 int4@50 在更长窗口存活
2. **[blocking] 建立真系统贡献**：实现 packed per-dtype page layout（A2）使 per-layer 保护容量中性，把 ×0.258 变成正面贡献；或 rescope 为窄测量 claim + 加 equal-byte 压缩 baseline（TurboQuant int4 / KVQuant 3-bit / KIVI）
3. **统一 3-seed 为 canonical**：表/图全从 bench_lat3 重生成 + mean±std 误差棒；解决 PPL 三文件不一致；清 Evaluation 草稿陈旧声明
4. **加固 quality**：PPL 3-seed CI；equal-byte 对 byte-exact 或注明容差；加 retrieval eval（RULER/LongBench）验证驱逐排序（PPL-only 对驱逐 claim 无效）
5. **确认 GDN dtype**：从 config/log 记录 mamba_ssm_cache_dtype → PROVENANCE.md；论文正文显式声明 fp32 假设（含 fp32-vs-bf16 敏感性）
6. **完成手稿**：references.bib（verified peer-reviewed 替换全部占位/[VERIFY]/community）；sharpening 'first' claim；9B @16384（3.167x）补进；Abstract 声明 E3 仅 2B（9B 单 run）；修数字舍入 + 范围端点；加 Discussion/Limitations/Data-Availability 节

## 6. 审稿人问题（供作者回复）
- 阈值 250/500/1000/3000ms 下 +25% 是否还在？（已答：250 和 500 下 0%）
- 为何用 fp16 而非 byte-matched baseline？纯 attention 3.88x 的稀释是混合架构固有还是可消除？
- ShareGPT 真实 trace 在高负载（SLO 边界）下结论是否反转？
- per-layer 失败的 vLLM V1 实现 artifact，能否用 packed layout 消除 → 这决定贡献定性
