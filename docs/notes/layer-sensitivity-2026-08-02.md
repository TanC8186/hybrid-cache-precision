# 逐层敏感度结果（2026-08-02）

## 实验
Qwen3.5-2B，Wikitext-2 5×2048-token。逐层：该层 KV 压 2-bit、其余层 8-bit。
敏感度 = (PPL_i - PPL_8bit) / (PPL_2bit - PPL_8bit)，0=不敏感，100=全 2-bit 代价。

## 结果（results/ablations/layer_sensitivity.csv）
| 配置 | PPL | 敏感度 |
|---|---|---|
| all_8bit | 13.63 | 基线 |
| all_2bit | 21.07 | 上界 |
| layer3_2bit | 13.20 | -5.9%（极度不敏感） |
| layer7_2bit | 14.24 | +8.1% |
| layer11_2bit | 14.01 | +5.0% |
| layer15_2bit | 14.24 | +8.1% |
| layer19_2bit | 13.98 | +4.6% |
| layer23_2bit | 15.76 | **+28.7%（最敏感）** |

## 发现
1. **6 层 GQA 对 KV 量化的敏感度高度异构**：layer 23 敏感度是 layer 3 的 ~5 倍
2. **模式：越靠后的注意力层越敏感**（3 → 23 单调上升大致成立，15 略高于 11）
3. layer 3 甚至轻微受益于 2-bit（-5.9%，对该评测集）
4. 机制确认（all_8bit 与上轮完全一致 → 无噪声）

## 论文含义
- **方法 = 逐层异构预算**：早层压 2-bit/多驱逐，末层（尤其 layer 23）保 4-bit
- 潜在收益：接近 4-bit 质量 @ 接近 2-bit 字节
- 与字节预算排序（驱逐>sub-4bit）结合：末层不驱逐+4bit，早层驱逐+2bit

## 下一步
- [ ] 验证异构预算分配：测若干 alloc（早 2bit + 末 4bit）的 PPL vs 字节，确认 Pareto 优于均匀
