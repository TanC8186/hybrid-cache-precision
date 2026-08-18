# MLSys 论文 AI draw.io 矢量科研图规格与逐图提示词

> 日期：2026-08-10  
> 目标：为 hybrid linear-attention serving 论文重建一套 MLSys 级主图与附录图  
> 绘图方式：AI 驱动的 draw.io 客户端  
> 强制格式：draw.io 原生矢量对象；主交付 `.drawio`，并导出 editable SVG 与 vector PDF  
> 关联审稿报告：`docs/notes/mlsys-ars-max-intensity-review-2026-08-10.md`

---

## 0. 使用说明

本文不是“配色建议清单”，而是逐张图的科学合同。每张图都包含：

1. 图的类型与论文角色。
2. 它必须回答的审稿问题。
3. 核心结论与证据层级。
4. 面板结构、视觉层级、元素、连线和文字。
5. 当前已有数据与需要补实验的数据。
6. 可直接复制给 AI draw.io 客户端的详细提示词。
7. 生成后的验收清单。

### 0.1 最重要的数据完整性规则

AI draw.io 客户端必须遵守：

- 不得编造实验点、置信区间、`p` 值、`q` 值、硬件、模型、baseline 或吞吐结果。
- 文档明确给出数值时，只使用这些数值。
- 文档写明 `DATA REQUIRED` 时，保留带浅灰斜线的占位框，并在框内写 `DATA REQUIRED — DO NOT INFER`。
- 未测配置必须写 `Not evaluated`，不得填 0；0 会被误读为“测得无效果”。
- 理论预测使用虚线或空心 marker；实测使用实线/实心 marker；二者不得混淆。
- “Future system component”必须用虚线边框，并明确写 `Not implemented in current artifact`。
- 统计结果必须在图或 caption 中给出 `n`、95% CI 定义和 multiple-comparison family。
- 相同 contracts/seeds 的第二次 formal run 只能写 `run-stability`, 不能写 `independent replication`。
- 0/60 serving cells 通过 BH-FDR 必须保留，不能因为影响视觉故事而删除。

### 0.2 当前证据与目标图的关系

| 图 | 当前能否诚实完成 | 说明 |
|---|---|---|
| Fig. 1 Problem–Insight–Workflow | **可以，当前证据版** | 若画成 runtime selector，必须等实现后再启用对应版本 |
| Fig. 2 Capacity phase diagram | **大部分可以** | 连续曲线是理论预测；实测点只有当前七个 cell |
| Fig. 3 Quality–capacity decision map | **可以做 2B/4K 核心版** | 不得把 PPL、GSM8K、RULER混成一个不透明总分 |
| Fig. 4 End-to-end load/SLO curves | **需要读取完整 serving artifacts** | 没有原始值时只能画 layout，不得猜曲线 |
| Fig. 5 Mechanism isolation | **被新实验阻断** | 只能画实验设计图，不能画结果图 |
| Fig. 6 Generality/scaling | **被新实验阻断** | 当前只能画 scope matrix，不能伪造跨硬件结果 |

---

## 1. 全篇统一视觉系统

### 1.1 画布与导出

- 双栏主图宽度：7.0–7.2 inch。
- 单栏图宽度：3.35–3.5 inch。
- draw.io 工作画布可用 1800×900 或 1800×1000 logical units，导出时保持比例。
- 页面背景纯白 `#FFFFFF`。
- 所有形状、箭头、文字、marker、误差线必须为原生 vector object。
- 不嵌入 PNG/JPEG，不把文字转成图片。
- SVG 中保留 `<text>`，PDF 保留 selectable text。
- 每个 panel 单独 group；整张 figure 再做总 group。

### 1.2 字体层级

- 字体：Arial；若不可用，Helvetica 或 Liberation Sans。
- panel letter：9–10 pt，bold，例如 `a`、`b`、`c`。
- panel title：8–9 pt，semibold。
- axis title：7.5–8 pt。
- tick、legend、annotation：6.5–7.5 pt。
- 最终导出后任何字符不得小于 5 pt。
- 不使用全大写长标题；不用艺术字体；不用粗重黑体占满画面。

### 1.3 全篇配置颜色

以下语义在所有图中必须固定：

| 配置/语义 | 颜色 | 用法 |
|---|---|---|
| Full precision / fp16 KV + fp32 state | `#4B5563` | 深灰 baseline |
| KV-only / int4 KV + fp32 state | `#2563EB` | 蓝色 |
| State-only / fp16 KV + bf16 state | `#0F766E` | 青绿色 |
| Joint / int4 KV + bf16 state | `#E57A1F` | 橙色，主方法/主 operating point |
| Architecture prediction | `#FFFFFF` fill + `#374151` stroke | 空心 diamond 或虚线 |
| Positive verified direction | `#2E9E44` | 只用于提升箭头或通过条件 |
| Regression/SLO violation | `#C43C39` | 只用于下降、失败、违规 |
| Uncertainty/individual seeds | `#B8C0CC` | 细线、浅灰 |
| Not evaluated | `#E5E7EB` + diagonal hatch | 未测，不是 0 |

### 1.4 线条和形状

- 主轮廓 1.0–1.2 pt；辅助轮廓 0.6–0.8 pt。
- 数据主线 1.5–2.0 pt；individual seed 0.5–0.7 pt。
- 箭头使用简洁实心三角箭头；避免粗大流程箭头。
- 圆角矩形半径轻微，禁止“PPT 卡片”式大圆角和阴影。
- 禁止渐变、玻璃效果、3D、发光、阴影、装饰 icon、GPU 照片。
- panel 之间依靠留白与对齐区分，不使用粗边框包围每个 panel。

### 1.5 坐标与图例

- 普通定量图保留左、下坐标轴；去掉上、右边框。
- 默认不画网格；需要对齐时只用非常浅的水平参考线 `#E5E7EB`。
- 相同配置优先 direct label；图例只保留一份共享 legend。
- 0 reference line 使用 `#9CA3AF`、0.8 pt、短虚线。
- CI 必须有端帽；同类 panel 使用同一种 CI 定义。
- 不截断会改变结论的坐标轴；若不从 0 开始，必须有清楚刻度和 caption 说明。

### 1.6 可复制的全局 master prompt

```text
Create an editable, publication-grade MLSys scientific figure in draw.io using native vector shapes only. Use a white background, Arial typography, no shadows, no gradients, no 3D, no decorative icons, and no raster images. The final figure must remain legible at 7.1-inch two-column width. Use small bold lowercase panel labels, restrained colors, direct labels where possible, and a single shared legend. Keep the following colors fixed throughout the paper: full precision = #4B5563, KV-only int4 = #2563EB, state-only bf16 = #0F766E, joint int4+bf16 = #E57A1F, prediction = white-filled dark-gray diamond or dashed line, positive direction = #2E9E44, regression or SLO violation = #C43C39, individual seeds/uncertainty = #B8C0CC, not evaluated = #E5E7EB with diagonal hatch. All text, markers, axes, error bars, arrows, and shapes must be editable vector objects. Never invent data. Use only values explicitly supplied in the prompt. Render missing experiments as a hatched box labeled “DATA REQUIRED — DO NOT INFER”. Render unimplemented components with a dashed border and the label “Not implemented in current artifact”.
```

---

# Main Figure 1 — Hybrid Memory Problem, Core Insight, and Workflow

## 2. 图类型与科学合同

- **图类型**：`schematic-led composite`，即 schematic 主导的非对称多面板图。
- **论文角色**：Problem/Insight/Hero Figure；应成为读者进入论文后第一张看到的图。
- **核心结论**：Hybrid linear-attention serving 的每个 sequence 同时消耗随上下文增长的 attention KV `A·L` 和固定 recurrent state `G`；因此 KV dtype 与 state dtype 必须在同一 GPU memory budget 中联合核算。
- **审稿问题**：为什么 attention-only 的 KV budget 思维不足？你的系统或测量流程新增了什么？headline payoff 是什么？
- **证据层级**：结构事实 > 正确容量公式 > 已测 2B/4K headline；不要把宽 CI 或限制塞进 Fig. 1。
- **建议尺寸**：7.1×3.6 inch，逻辑画布 1800×900。

## 2.1 推荐布局

使用三列非对称布局：

- 左列 42%：panel `a`，hybrid layer stack 与两类 state。
- 中列 31%：panel `b`，共享 GPU pool、公式和 context scaling。
- 右列 27%：panel `c`，当前 measurement workflow 与 headline result。

### Panel a：Hybrid architecture creates two memory components

1. 顶部画一条横向 layer stack，不要画全部 24/32 层，以 repetition glyph 表示。
2. GDN layer 使用青绿色小矩形，标注 `GDN / recurrent state`。
3. GQA attention layer 使用蓝色小矩形，标注 `GQA / KV cache`。
4. 下方从 attention layer 引出多个 token-shaped small blocks，标签：`per-token KV, grows with context L`。
5. 从 GDN layer 引出一个固定大小 state matrix glyph，标签：`per-sequence recurrent state, fixed with L`。
6. 在 layer stack 下用小字说明：`Qwen3.5-2B: 18 GDN + 6 GQA`；`Qwen3.5-9B: 24 GDN + 8 GQA`。
7. 不画两个独立 GPU pool；两个箭头都必须指向 panel b 的同一 shared pool。

### Panel b：One shared memory budget

1. 画一个简洁 GPU memory outline，标题 `Shared GPU memory budget M`。
2. 内部横向堆叠两段：
   - 蓝色段 `Attention KV = A·L`，旁边用小箭头说明 `grows with L`。
   - 青绿色段 `Recurrent state = G`，说明 `fixed per sequence`。
3. 下方放两条公式，公式必须正确：
   - `Concurrent sequences: N(L) = M / (A·L + G)`
   - `Total token capacity: T(L) = L·N(L)`
4. 画一个从短 context 到长 context 的三步 mini-sequence：
   - short L：`G` 占比大；
   - medium L：两者接近；
   - long L：`A·L` 占主导。
5. 在 panel 下方写一句 insight：`State precision matters most in short-context, high-concurrency serving.`

### Panel c：当前证据版 workflow + headline

1. 画一个从左到右的三步流程：
   - `Model config + vLLM page layout`
   - `Architecture-derived footprint model`
   - `Deterministic capacity probe + quality/SLO evaluation`
2. 三个框使用细边、无阴影；箭头只表示测量流程，不要画成 runtime controller。
3. 下方放一个重点 headline：
   - 小标题：`Qwen3.5-2B, L = 4096, int4 KV, RTX 5090`
   - 大数字：`657 → 904 concurrent sequences`
   - 次级文字：`+247 sequence slots`
   - 底部：`int4/fp16 KV capacity ratio: 2.245× → 2.675× when state changes fp32 → bf16`
4. 使用橙色突出 joint config；baseline 用灰色。
5. 不出现 `conservative lower bound`。

## 2.2 当前证据版 draw.io 提示词

```text
Using the global MLSys vector style, create a three-panel asymmetric hero figure titled only through its panel content, with no large figure-wide banner.

Panel a, occupying about 42% of the width, explains the hybrid model structure. Draw a horizontal sequence of repeated layer blocks: teal GDN blocks and blue GQA attention blocks, using ellipsis/repetition rather than drawing every layer. Add small labels “Qwen3.5-2B: 18 GDN + 6 GQA” and “Qwen3.5-9B: 24 GDN + 8 GQA”. From GQA blocks, route a blue connector to token-shaped KV blocks labeled “per-token KV, grows with context L”. From GDN blocks, route a teal connector to a fixed matrix-shaped state labeled “per-sequence recurrent state, fixed with L”. Both connectors must continue toward the same shared memory pool in panel b. Do not depict separate memory pools.

Panel b, occupying about 31%, shows a single GPU outline labeled “Shared GPU memory budget M”. Inside, create a stacked horizontal allocation with blue “Attention KV = A·L” and teal “Recurrent state = G”. Add a small three-state context-length progression showing that G dominates at short L and A·L dominates at long L. Beneath the pool, typeset exactly: “Concurrent sequences: N(L) = M / (A·L + G)” and “Total token capacity: T(L) = L·N(L)”. Add the direct insight label: “State precision matters most in short-context, high-concurrency serving.”

Panel c, occupying about 27%, shows the current evidence workflow, not a runtime controller. Use three compact boxes connected left-to-right: “Model config + vLLM page layout” → “Architecture-derived footprint model” → “Deterministic capacity probe + quality/SLO evaluation”. Below, create the hero numeric callout: “Qwen3.5-2B, L = 4096, int4 KV, RTX 5090”; “657 → 904 concurrent sequences”; “+247 sequence slots”; and “int4/fp16 KV capacity ratio: 2.245× → 2.675× when state changes fp32 → bf16”. Use gray for the baseline and orange for the joint configuration. Do not use the phrase lower bound. Keep all objects editable and ensure the visual reading order is architecture → shared budget → measured payoff.
```

## 2.3 目标系统版替换模块（实现 selector 后才可使用）

当前 panel c 可以在系统完成后替换为：

```text
Replace panel c only after the runtime selector is implemented and evaluated. Draw a system box labeled “Joint precision budget selector”. Inputs on the left: “GPU budget M”, “context/workload distribution”, “quality budget ε”, “TTFT/TPOT SLO”. Inside the selector, show a small constrained optimization glyph. Outputs on the right: four candidate configurations — full precision, state-only, KV-only, joint — with the selected joint operating point outlined in orange. Below, route the chosen KV dtype and state dtype into the vLLM allocator. Include a small runtime feedback loop from measured memory pressure and SLO telemetry back to the selector only if such feedback exists in the implementation. Do not use this version if the selector is not implemented.
```

## 2.4 验收清单

- [ ] `N(L)` 与 `T(L)` 正确定义。
- [ ] KV 和 state 明确共享同一个 pool。
- [ ] 没有 lower-bound 说法。
- [ ] 当前版没有伪造 runtime selector。
- [ ] 读图顺序在 5 秒内清楚：hybrid state → shared budget → 657→904。

---

# Main Figure 2 — Capacity Phase Diagram and Model Validation

## 3. 图类型与科学合同

- **图类型**：`quantitative grid`，以一个大 hero line chart + 两个支持 panel 构成。
- **论文角色**：核心科学结果图。
- **核心结论**：state-bf16 的容量收益随 context length 增长而下降；architecture-derived model 描述总体趋势，但 discrete allocator rounding 产生带符号 residual。
- **审稿问题**：模型是否预测了未测 context？实测与预测一致到什么程度？偏差是否系统性？
- **建议尺寸**：7.1×4.0 inch，逻辑画布 1800×1000。

## 3.1 已验证数据

| KV | Model | L | fp32 tokens | bf16 tokens | measured ratio | predicted ratio | residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| int4 | 2B | 4096 | 2,692,710 | 3,703,954 | 1.375549 | 1.408941 | -2.37% |
| int4 | 2B | 16384 | 4,895,837 | 5,458,458 | 1.114918 | 1.152251 | -3.24% |
| int4 | 9B | 4096 | 315,392 | 443,538 | 1.406307 | 1.408843 | -0.18% |
| int4 | 9B | 16384 | 573,440 | 653,635 | 1.139849 | 1.152177 | -1.07% |
| fp16 | 2B | 4096 | 1,199,383 | 1,384,448 | 1.154300 | 1.122205 | +2.86% |
| fp16 | 2B | 16384 | 1,552,143 | 1,661,337 | 1.070350 | 1.033855 | +3.53% |
| fp16 | 9B | 4096 | 144,104 | 161,899 | 1.123487 | 1.156208 | -2.83% |

理论连续曲线必须由修正后的公式和真实 `A`、`G` 参数生成。若 draw.io AI 没有获得 curve points，只保留曲线占位并写 `PREDICTED CURVE POINTS REQUIRED`，不得凭视觉插值。

## 3.2 推荐布局

- panel `a`：左侧 62%，hero line chart。
- panel `b`：右上 38%，actual token capacity grouped dot/bar chart。
- panel `c`：右下 38%，signed residual lollipop。

### Panel a：State gain across context length

- x-axis：`Context length L (tokens)`，建议 log2 刻度 1K、2K、4K、8K、16K、32K。
- y-axis：`Capacity ratio, state bf16 / fp32`。
- y=1 画浅灰虚线。
- 理论预测：虚线；实测点：实心 marker。
- 2B 与 9B 用 marker shape 区分，KV dtype 用线色/线型区分；不要创造八种颜色。
- 在 4K 与 16K 实测点旁只标 headline ratios，避免每点塞完整数据。
- panel 内加一个小 annotation：`state contribution decays as A·L dominates`。

### Panel b：Measured total token capacity

- 使用 horizontal paired dot plot，不用密集柱状图。
- 每一行一个 `(KV, model, L)` cell。
- 灰点为 fp32 state，配置对应色点为 bf16 state；细线连接。
- x-axis 使用 total token capacity，允许科学计数法。
- 只画七个已测 cell。

### Panel c：Allocator residual

- x-axis：`(measured − predicted) / predicted (%)`。
- 中间为 0 线。
- int4 cells 使用蓝色/橙色 family；fp16 cells 使用灰/青 family。
- 明确同时存在正负 residual；标题使用 `Discrete allocator residual`。
- 不写 lower bound。

## 3.3 draw.io 提示词

```text
Create a three-panel quantitative MLSys figure using only the supplied seven measured cells. Panel a is the hero panel and occupies 62% of the width. Draw a clean line chart with x-axis “Context length L (tokens)” on log2-style positions 1K, 2K, 4K, 8K, 16K, 32K and y-axis “Capacity ratio, state bf16 / fp32”. Add a light dashed y=1 reference. The architecture-derived predictions must be dashed lines or white-filled dark-outline markers; measured values must be solid markers. Distinguish 2B and 9B by marker shape, not by adding many colors. Use the global configuration palette. Plot measured points only at available L values: int4 2B = 1.375549 at 4K and 1.114918 at 16K; int4 9B = 1.406307 at 4K and 1.139849 at 16K; fp16 2B = 1.154300 at 4K and 1.070350 at 16K; fp16 9B = 1.123487 at 4K only. Predicted values at the same points are 1.408941, 1.152251, 1.408843, 1.152177, 1.122205, 1.033855, and 1.156208 in the same order. Do not invent unmeasured points. If continuous predicted curve coordinates are not supplied, create a dashed placeholder labeled “PREDICTED CURVE POINTS REQUIRED” rather than guessing.

Panel b occupies the upper-right region. Create a horizontal paired-dot chart titled “Measured total token capacity”. Use seven rows corresponding to the supplied cells. Place fp32-state capacity as a dark-gray dot and bf16-state capacity as the appropriate colored dot, connected by a thin neutral line. Use exact totals: 2,692,710→3,703,954; 4,895,837→5,458,458; 315,392→443,538; 573,440→653,635; 1,199,383→1,384,448; 1,552,143→1,661,337; 144,104→161,899. Label rows compactly as “int4 · 2B · 4K”, etc.

Panel c occupies the lower-right region. Create a signed lollipop plot titled “Discrete allocator residual” with x-axis “(measured − predicted) / predicted (%)” and a vertical zero line. Plot exactly: -2.37, -3.24, -0.18, -1.07, +2.86, +3.53, -2.83 percent. Directly label the cell names. Add the small conclusion “Residual signs depend on page/block layout.” Do not call the model a lower bound. Use restrained annotations and a single shared legend for state precision, KV dtype, model size, prediction, and measurement.
```

## 3.4 验收清单

- [ ] 只有七个实测 cell。
- [ ] 预测与实测编码清楚不同。
- [ ] residual 同时展示正负号。
- [ ] 图中没有用实线连接不存在的实测 context。
- [ ] Fig. 2 不重复 Fig. 1 的共享池 schematic。

---

# Main Figure 3 — Quality–Capacity Decision Map

## 4. 图类型与科学合同

- **图类型**：`asymmetric mixed-modality figure`，Pareto scatter + 2×2 decision matrix + secondary quality strip。
- **论文角色**：回答“容量收益是否值得质量代价”。
- **核心结论**：在 2B/4K 的四种配置中，joint allocation 提供最高容量 ratio，但质量影响是 task-dependent，不能称为 universally lossless。
- **审稿问题**：每个配置位于什么 quality–capacity operating point？质量约束下哪个点可行？
- **建议尺寸**：7.1×3.8 inch。

## 4.1 可用于 hero panel 的 2B/4K 数据

以 `fp16 KV + fp32 state` 为 baseline：

| Configuration | Capacity ratio vs baseline | GSM8K delta (pp) | 95% CI |
|---|---:|---:|---|
| fp16 KV + fp32 state | 1.0000 | 0.00 | baseline |
| fp16 KV + bf16 state | 1.1543 | -1.00 | [-1.71, -0.29] |
| int4 KV + fp32 state | 2.2451 | -2.72 | [-4.20, -1.24] |
| int4 KV + bf16 state | 2.6754 | -1.56 | `USE VERIFIED SOURCE CI — DO NOT APPROXIMATE` |

补充质量证据：

- 9B state-bf16 GSM8K：`+0.33 pp [-0.07, +0.73]`, `p=0.141`。
- 2B state-bf16 GSM8K：`-1.00 pp [-1.71, -0.29]`, `p=0.025`, MDE 1.16 pp, observed power 67.5%。
- int4 KV：`-2.72 pp [-4.20, -1.24]`, `p=0.007`, power 88.3%。
- state-bf16 marginal under int4：`+1.17 pp [-0.33, +2.66]`。
- PPL marginal under int4：C4 `-0.0029`，PG19 `+0.0065`，均不支持强 equivalence claim。
- RULER 五个复测点估计值：`-3.89, +1.66, +0.83, -4.17, +0.55 pp`；CI 必须从 verified source 获取，当前只知道全部跨 0 且很宽。

## 4.2 推荐布局

- panel `a` 左侧 55%：2B/4K Pareto-style scatter。
- panel `b` 右上 45%：2×2 configuration matrix。
- panel `c` 右下 45%：task-specific quality evidence strip。

### Panel a：Capacity versus GSM8K change

- x-axis：`GSM8K accuracy change (percentage points; higher is better)`。
- y-axis：`Token-capacity ratio vs fp16 KV + fp32 state`。
- 四个点使用全篇配置颜色。
- x error bars 使用 95% paired CI；joint CI 未提供时必须占位。
- 不要自动画 Pareto frontier，除非根据 quality constraint 明确定义 dominance。
- 可用浅灰竖带表示示例 quality budget `Δ accuracy ≥ -1 pp`，但必须标成 `example constraint`，不能声称预注册，除非实验协议确实定义。

### Panel b：2×2 decision matrix

- columns：KV precision `fp16`、`int4`。
- rows：state precision `fp32`、`bf16`。
- 每个 cell 上半部显示 capacity ratio，下半部显示 GSM8K delta。
- joint cell 用橙色边框突出，但不能贴 `best`；可写 `highest measured capacity`。

### Panel c：Secondary quality evidence

- 使用三个紧凑 horizontal forest mini-panels：GSM8K、PPL、RULER。
- 每个 mini-panel 统一 0 reference，但轴单位独立，不能共用数值轴。
- RULER 写 `wide CIs; no equivalence claim`。
- PPL 写 `chunk-level approximation`。

## 4.3 draw.io 提示词

```text
Create an asymmetric three-panel decision figure. Panel a occupies 55% of the width and is the visual hero. Draw a scatter plot with x-axis “GSM8K accuracy change (percentage points; higher is better)” and y-axis “Token-capacity ratio vs fp16 KV + fp32 state”. Plot four configurations with fixed colors: full precision gray at (0.00, 1.0000); state-only teal at (-1.00, 1.1543) with horizontal 95% CI [-1.71, -0.29]; KV-only blue at (-2.72, 2.2451) with CI [-4.20, -1.24]; joint orange at (-1.56, 2.6754). For the joint point, do not invent an interval: add a small hatched horizontal placeholder labeled “VERIFIED JOINT CI REQUIRED”. Use direct labels beside points. Do not automatically connect points or call any point Pareto-optimal without a stated constraint. Optionally add a very pale vertical band labeled “example quality constraint, not pre-registered” at Δ accuracy ≥ -1 pp.

Panel b in the upper-right is a 2×2 matrix. Columns are “KV fp16” and “KV int4”; rows are “State fp32” and “State bf16”. In each cell, display capacity ratio on the first line and GSM8K delta on the second line: 1.0000 / 0.00 pp; 2.2451 / -2.72 pp; 1.1543 / -1.00 pp; 2.6754 / -1.56 pp. Use the global colors and outline the joint cell in orange with the neutral label “highest measured capacity”, not “best”.

Panel c in the lower-right is a task-specific evidence strip with three compact sections, each with its own x scale. GSM8K: show 2B state -1.00 [-1.71, -0.29], 9B state +0.33 [-0.07, +0.73], int4 KV -2.72 [-4.20, -1.24], and marginal state under int4 +1.17 [-0.33, +2.66]. PPL: show C4 -0.0029 and PG19 +0.0065 and label “chunk-level approximation”. RULER: show point estimates -3.89, +1.66, +0.83, -4.17, +0.55 only if their verified confidence intervals are also supplied; otherwise use a hatched placeholder. Add the annotation “wide CIs; no equivalence claim”. Never average GSM8K, PPL, and RULER into one score.
```

## 4.4 验收清单

- [ ] 所有 metric 使用独立、明确单位。
- [ ] 不存在不透明 aggregate quality score。
- [ ] joint CI 未知时没有猜测。
- [ ] `highest measured capacity`没有被写成`best overall`。
- [ ] 2B 显著回归与 RULER 低分辨率都保留。

---

# Main Figure 4 — End-to-End Serving Load and SLO Curves

## 5. 图类型与科学合同

- **图类型**：`quantitative grid`，按 workload 分行、metric 分列。
- **论文角色**：端到端系统主结果。
- **核心结论**：不同 precision allocation 在 offered load 增长时形成不同 saturation/SLO operating regions；收益必须通过完整曲线和 workload sensitivity 展示。
- **审稿问题**：何时出现 queueing knee？joint 配置是否扩大 sustainable region？是否跨 workload 稳定？
- **状态**：需要从完整 serving artifacts 提取每个 rate/seed/config 的原始指标。没有这些值时禁止生成曲线。
- **建议尺寸**：7.1×4.4 inch。

## 5.1 必须使用的数据字段

- workload：Random60、ShareGPT300。
- configuration：full precision、state-only、KV-only、joint。若某配置未运行，写 `Not evaluated`。
- offered rate。
- goodput 或 goodput/offered。
- P95 TTFT。
- P95 TPOT/TBT。
- failure count 与 denominator。
- seed 与 run id。
- 95% uncertainty。
- sustainable condition：`goodput/offered ≥ 0.95`、`TPOT ≤ 200 ms`、指定 TTFT threshold。

## 5.2 推荐布局

两行三列：

| | Column 1 | Column 2 | Column 3 |
|---|---|---|---|
| Row 1 | Random60 goodput | Random60 P95 TTFT | Random60 P95 TPOT |
| Row 2 | ShareGPT300 goodput | ShareGPT300 P95 TTFT | ShareGPT300 P95 TPOT |

附加一个窄 panel `g` 放在右侧或底部：sustainable-rate boundary summary + run-to-run stability disclosure。

### 曲线规则

- x-axis 统一 `Offered request rate (req/s)`。
- 每个 config 一条线，颜色固定。
- seed aggregate 使用点 + 95% CI band；若只有 3 seeds，caption 必须写明。
- horizontal SLO line：TTFT threshold、TPOT 200 ms、goodput/offered 0.95。
- 不只截取 overload region。
- formal 与 second run 不宜用两套颜色；使用 circle/diamond marker 区分 run。

## 5.3 当前可确定的 boundary 信息

| Workload | Config | 250 ms | 500 ms | 1000 ms | 2000 ms | 3000 ms |
|---|---|---:|---:|---:|---:|---:|
| Random60 | int4 | 30 | 35 | 35 (second: 40) | 40 | 40 |
| Random60 | int4+bf16 | 30 | 35 | 40 | 40 | 40 |
| ShareGPT | int4 | 40 (second: 35) | 40 | 40 | 40 | 40 |
| ShareGPT | int4+bf16 | 35 (second: 40) | 40 | 40 | 40 | 40 |

已披露 selected deltas：

- Random60 r40/250 ms：formal `+0.334 [0.078, 0.589]`，second `+0.304 [0.250, 0.359]`。
- Random60 r40/500 ms：formal `+0.215 [0.154, 0.276]`，second `+0.138 [0.082, 0.195]`。
- Random60 r45/3000 ms：formal `+0.367 [0.308, 0.426]`，second `+0.372 [0.310, 0.435]`。
- 60 cells 中没有一个 BH-FDR `q<0.05`。
- 13/13 Random60 cells 在两次运行中均为正；7/10 ShareGPT overload cells 符号相反。

这些摘要不能替代完整曲线。

## 5.4 draw.io 提示词

```text
Create a two-row by three-column MLSys serving-results grid plus one compact boundary-summary panel. Do not draw any metric curve unless the prompt is supplied with the exact per-rate aggregate and uncertainty. If the data table is missing, create the axes, labels, SLO reference lines, legend, and a hatched central placeholder reading “DATA REQUIRED — DO NOT INFER”.

Rows are Random60 and ShareGPT300. Columns are goodput or goodput/offered, P95 TTFT, and P95 TPOT. Use a shared x-axis label “Offered request rate (req/s)”. Use the global configuration colors for full precision, state-only, KV-only, and joint. Use circles for the formal run and diamonds for the second run, while keeping colors tied only to configurations. Show 95% uncertainty consistently. Add horizontal SLO lines: goodput/offered = 0.95, TPOT = 200 ms, and the applicable TTFT threshold. Show the entire offered-load range, including the pre-saturation region, queueing knee, and overload region. Never show only selected positive cells.

In the summary panel, draw a compact threshold-by-workload boundary table using exact current values: Random60 int4 = 30,35,35(40),40,40; Random60 int4+bf16 = 30,35,40,40,40; ShareGPT int4 = 40(35),40,40,40,40; ShareGPT int4+bf16 = 35(40),40,40,40,40 for TTFT thresholds 250,500,1000,2000,3000 ms. Use parentheses for the second run. Add three factual annotations: “0/60 cells survive BH-FDR q<0.05”; “Random60 overload: 13/13 cells positive in both runs”; “ShareGPT overload: 7/10 cells flip sign”. Label the second run “run-stability, same contracts and seeds”, never “independent replication”.

If a small selected-delta inset is included, use exactly: r40/250 +0.334 [0.078,0.589] vs +0.304 [0.250,0.359]; r40/500 +0.215 [0.154,0.276] vs +0.138 [0.082,0.195]; r45/3000 +0.367 [0.308,0.426] vs +0.372 [0.310,0.435]. Label this inset “descriptive selected cells” and do not make it the hero panel.
```

## 5.5 验收清单

- [ ] 没有 raw data 就没有假曲线。
- [ ] full operating range 可见，不只展示 overload。
- [ ] SLO 条件直接画出。
- [ ] 0/60 FDR 与 workload sign instability 未被隐藏。
- [ ] run marker 与 config color 语义不冲突。

---

# Main Figure 5 — Mechanism Isolation and Bottleneck Breakdown

## 6. 图类型与科学合同

- **图类型**：`schematic-led composite`，因果路径图 + controlled contrasts + breakdown。
- **论文角色**：解释 serving gain 的来源，击破“只是相关性”的审稿意见。
- **核心结论目标**：分离 state bytes、block geometry、memory bandwidth 和 queueing 对容量/serving 的贡献。
- **当前状态**：fixed-block-count 与 fixed-bytes 尚未执行。当前只能画成实验计划，不能作为已完成结果图。
- **建议尺寸**：7.1×3.8 inch。

## 6.1 推荐布局

- panel `a` 左侧 38%：causal pathway schematic。
- panel `b` 中间 27%：controlled experiment matrix。
- panel `c` 右上 35%：resource breakdown bars，`DATA REQUIRED`。
- panel `d` 右下 35%：causal contribution summary，`DATA REQUIRED`。

### Panel a：可能机制路径

画两条并行路径：

1. **Capacity/allocator path**：
   `state fp32 → bf16` → `state bytes ↓` → `tokens per block / allocated blocks change` → `concurrency ↑` → `queueing pressure changes` → `goodput/TTFT`。
2. **Bandwidth/compute path**：
   `state fp32 → bf16` → `state read/write traffic ↓` → `kernel or memory time changes` → `TPOT/goodput`。

所有未验证的因果箭头使用虚线，标题写 `Candidate mechanisms`。

### Panel b：控制实验矩阵

| Contrast | State bytes | Block count | Block geometry | Concurrency | Purpose |
|---|---|---|---|---|---|
| Natural dtype switch | changes | changes | changes | changes | aggregate effect |
| Fixed-block-count | changes | fixed | may change | controlled | isolate capacity allocation |
| Fixed-bytes | fixed | may change | changes | controlled | isolate geometry/rounding |
| Fixed-concurrency | changes | may change | may change | fixed | isolate per-request execution |

### Panel c/d：结果位置

- HBM read/write bytes/token。
- state kernel latency。
- allocator blocks。
- queue wait。
- P95 TTFT/TPOT。
- goodput。

当前全部标 `DATA REQUIRED`。

## 6.2 draw.io 提示词

```text
Create a mechanism-isolation figure, but explicitly render it as a planned experiment until controlled results exist. Panel a is a candidate causal graph. Start with “state fp32 → bf16” and split into two dashed-arrow paths. Capacity/allocator path: “state bytes ↓” → “tokens per block / allocated blocks change” → “concurrency ↑” → “queueing pressure changes” → “goodput / TTFT”. Bandwidth/compute path: “state read/write traffic ↓” → “kernel or memory time changes” → “TPOT / goodput”. Label the panel “Candidate mechanisms — not yet isolated”. Use dashed arrows for every unverified causal relation.

Panel b is a controlled-contrast matrix with four rows: Natural dtype switch, Fixed-block-count, Fixed-bytes, Fixed-concurrency. Columns: State bytes, Block count, Block geometry, Concurrency, Purpose. Fill the matrix exactly as follows: Natural switch = changes/changes/changes/changes/aggregate effect; Fixed-block-count = changes/fixed/may change/controlled/isolate capacity allocation; Fixed-bytes = fixed/may change/changes/controlled/isolate geometry and rounding; Fixed-concurrency = changes/may change/may change/fixed/isolate per-request execution.

Panel c contains aligned horizontal breakdown bars for HBM read bytes/token, HBM write bytes/token, state-kernel latency, allocator blocks, queue wait, and end-to-end latency. Panel d contains a compact contribution summary linking each controlled contrast to capacity ratio and serving metrics. Because these experiments have not been run, fill panels c and d with light-gray diagonal hatch and the exact label “DATA REQUIRED — DO NOT INFER”. Do not draw fabricated bars. Keep this figure out of the manuscript results section until the placeholders can be replaced by verified measurements.
```

## 6.3 验收清单

- [ ] 未验证因果箭头全部是虚线。
- [ ] 未完成实验清楚标为 DATA REQUIRED。
- [ ] 不把 block rounding、bandwidth 和 queueing 合并成一个“mechanism”标签。
- [ ] 完成实验后，每条结论都能对应一个 controlled contrast。

---

# Main Figure 6 — Generality, Scaling, and Scope Boundary

## 7. 图类型与科学合同

- **图类型**：`quantitative grid` 或 compact heatmap matrix。
- **论文角色**：回答规律是否超出单卡 RTX 5090 + Qwen3.5。
- **核心结论目标**：architecture-derived model 与 joint precision benefit 在 model size、context、hardware、TP 和 workload 上的适用范围。
- **当前状态**：只有 Qwen3.5-2B/9B、RTX 5090、TP=1。当前只能画 scope matrix；新增数据前不能画 scaling result。
- **建议尺寸**：7.1×3.8 inch。

## 7.1 推荐布局

- panel `a`：evidence coverage matrix。
- panel `b`：capacity gain vs context，按 hardware/TP small multiples。
- panel `c`：model prediction error heatmap。
- panel `d`：quality/SLO feasibility summary。

### Panel a：Evidence coverage matrix

Rows 可包含：

- Qwen3.5-2B / RTX 5090 / TP1。
- Qwen3.5-9B / RTX 5090 / TP1。
- Other hybrid model / hardware / TP：未测。

Columns：capacity、GSM8K、RULER、serving、mechanism isolation。

已测 cell 用对应颜色；未测 cell 用 `Not evaluated` hatch；低分辨率 cell 用空心或浅色并标 `low resolution`。

### Panel b–d

新增实验前全部保留 `DATA REQUIRED`。不能用理论 TP scaling 代替实测。

## 7.2 draw.io 提示词

```text
Create a generality-and-scope figure that distinguishes measured evidence from untested transfer. Panel a is an evidence-coverage matrix. Rows include “Qwen3.5-2B · RTX 5090 · TP1” and “Qwen3.5-9B · RTX 5090 · TP1”, followed by planned rows for another hybrid/SSM model, another GPU class, TP2, and TP4. Columns are Capacity, GSM8K, RULER, Serving, Mechanism isolation. Mark only actually measured cells as measured. Mark RULER cells as “measured, low resolution”. Mark mechanism isolation as “not evaluated”. Use gray diagonal hatch and the words “Not evaluated” for every untested cell; never use zero.

Panel b is reserved for capacity ratio versus context with small multiples by hardware and tensor parallelism. Panel c is reserved for model prediction residual heatmaps across model, context, and hardware. Panel d is reserved for quality/SLO feasibility. Until new experiments exist, use hatched placeholders labeled “DATA REQUIRED — DO NOT INFER”. Add the factual boundary note: “Current evidence: Qwen3.5-2B/9B, one RTX 5090, TP=1.” Add another note: “TP scaling is an analytical expectation, not a measurement.” Do not imply cross-architecture generality.
```

## 7.3 验收清单

- [ ] 未测不等于 0。
- [ ] TP analytical expectation 与 measurement 明确分开。
- [ ] 当前证据边界在图内可见。
- [ ] 新增数据后使用一致 metric 和 normalization。

---

# Appendix Figure A1 — GSM8K Paired-Seed Trajectories

## 8. 图类型

- `paired trajectory / spaghetti plot with emphasized mean`。
- 作用：展示配对 seed 结构和 between-seed variability，不承担主结论。
- 单栏或双栏均可；推荐 7.1×2.6 inch 两面板。

## 8.1 draw.io 提示词

```text
Create a two-panel appendix paired-trajectory figure. Panel a shows Qwen3.5-2B across four ordered configurations: fp16 KV + fp32 state, fp16 KV + bf16 state, int4 KV + fp32 state, int4 KV + bf16 state. Panel b shows Qwen3.5-9B across the verified configurations only. Draw each of the nine paired dataset seeds as a thin light-gray line connecting the same seed across configurations. Draw the seed mean as a thicker colored line with circular markers. Use exact source data for every seed; if seed-level values are not supplied, do not infer trajectories and use a DATA REQUIRED placeholder. Label “200 items per seed; 9 paired dataset seeds; greedy; no chain-of-thought”. Do not add significance stars; the paired CI summary belongs in the main quality figure.
```

---

# Appendix Figure A2 — Perplexity Harness Boundary and Stacking Ablation

## 9. 图类型

- `methods diagnostic + ablation`。
- 作用：量化 harness approximation，不在主文占据 hero 位置。

## 9.1 已知数值

- C4 chunk=128 PPL：19.35。
- C4 chunk=1 PPL：36.16，约 +87%。
- marginal state-bf16 PPL under int4：C4 -0.0029，PG19 +0.0065。

## 9.2 draw.io 提示词

```text
Create a compact two-panel appendix diagnostic. Panel a is a paired two-condition comparison titled “Chunk-level state write-back changes the PPL scale”. Plot C4 PPL 19.35 for chunk=128 and 36.16 for chunk=1, annotate “+87%”. Use neutral gray and blue, not red, because this is a harness difference rather than a model regression. Add “1 seed, 1 sequence” directly below the panel title. Panel b is a horizontal dot plot titled “Marginal state-bf16 PPL under int4 KV”. Plot C4 -0.0029 and PG19 +0.0065 against a zero reference. Add “3 paired seeds” and “chunk-level approximation; not kernel-equivalent”. Use editable vector objects and no decorative bars.
```

---

# Appendix Figure A3 — RULER Full Evidence and Resolution Boundary

## 10. 图类型

- `forest plot + evidence-coverage matrix`。
- 作用：完整披露五个复测 cell 与 single-seed full grid，防止选择性展示。

## 10.1 draw.io 提示词

```text
Create a two-part RULER appendix figure. Panel a is a forest plot for the five screen-selected cells: 2B FWE 4K, 2B FWE 8K, 9B NIAH-multiquery 4K, 9B NIAH-multiquery 8K, 9B FWE 8K. Use verified point estimates -3.89, +1.66, +0.83, -4.17, +0.55 percentage points and draw the exact verified 95% confidence intervals from the source data. If the intervals are not supplied, do not estimate them. Add the direct annotation “All intervals include zero; no equivalence claim; 20 samples per cell; 3 dataset seeds; default thinking; max 256 tokens.” Panel b is the complete 7-task × 2-length × 2-model single-seed grid. Use a diverging blue-to-red scale centered at zero, print cell values, and mark the five multi-seed retested cells with a thin outline. Add the text “single-seed screen; descriptive only”. Missing cells must be hatched, not zero.
```

---

# Appendix Figure A4 — Per-Layer State Sensitivity

## 11. 图类型

- `compact forest plot or heatmap`。
- 作用：说明没有 layer-wise signal 通过 multiple-comparison correction，从而支持 whole-state switch 作为当前配置粒度。

## 11.1 draw.io 提示词

```text
Create a compact two-panel appendix forest plot for per-layer state sensitivity on Qwen3.5-2B. Panel a is C4 and panel b is PG19. Rows are the 18 GDN layers actually tested. Plot each per-layer PPL delta for switching only that layer to bf16, with exact 95% paired confidence intervals from 3 seeds. Add a vertical zero line. Use light-gray intervals and dark-gray markers. Outline C4 layers 2 and 8 with a thin orange ring and label “raw p<0.05 only”. Above both panels, add the factual statement “No per-layer effect survives Bonferroni or BH-FDR across 36 tests.” Do not color raw p<0.05 points red or imply discovery. Keep the panel in the appendix.
```

---

# Appendix Figure A5 — Complete Serving Cell Audit

## 12. 图类型

- `matrix heatmap + formal-versus-second-run agreement scatter`。
- 作用：完整披露所有 60 cells、方向、`p/q` 与 run stability。

## 12.1 draw.io 提示词

```text
Create a two-panel serving-audit appendix figure from the complete 60-cell verified table. Panel a is a workload × offered-rate × TTFT-threshold matrix of paired goodput delta, using a diverging color scale centered at zero. Print each delta in the cell and add a small symbol only for cells meeting the specified raw criterion; do not use significance stars. Add a companion annotation or small second line for BH q values. Explicitly state “0/60 cells with BH-FDR q<0.05”. Panel b is an agreement scatter with formal-run delta on the x-axis and second-run delta on the y-axis. Add x=0, y=0, and y=x reference lines. Use different marker shapes for Random60 and ShareGPT300. Highlight the 13 Random60 overload cells that are positive in both runs without changing their numerical values. Add “same contracts and seeds: run-stability, not independent replication”. Use exact source values only; if the 60-cell table is not supplied, keep the data regions as DATA REQUIRED placeholders.
```

---

# Appendix Figure A6 — Allocator Block Granularity

## 13. 图类型

- `paired dumbbell diagnostics`。
- 作用：展示 state bytes 如何改变 tokens/block 和 allocated blocks，作为容量 residual 的机制支持。

## 13.1 draw.io 提示词

```text
Create a two-panel appendix dumbbell figure. Panel a is “Tokens per GPU block”; panel b is “Allocated GPU blocks”. Use one row for each of the seven verified capacity cells. For each row, place fp32-state and bf16-state dots connected by a thin line. Use dark gray for fp32 state and the configuration-consistent color for bf16 state. Directly label representative exact values such as int4 2B tokens per block 2064→1072 and fp16 2B 544→288 only where verified. Populate all other positions from the allocator records; never infer values from visual spacing. Add the conclusion “Discrete page/block arithmetic explains signed model residuals.” Keep axes aligned, remove decorative framing, and place a single shared legend.
```

---

## 14. 不应该继续保留的图形形式

以下形式即使是矢量图，也不符合本稿当前的证据需求：

- 用一排等大方框代替系统 dataflow。
- 为每个数据表单独做一张图。
- 用大面积 forest plot 展示大量 null results，并占据主文双栏。
- 把同一 GSM8K 数据同时画成 delta forest 和 seed trajectory 两张主图。
- 只画 positive overload cells，不画完整 load curve。
- 用 radar chart 汇总不可比的 PPL、accuracy、latency、capacity。
- 用红绿热图把 `Not evaluated` 显示成 0。
- 用卡片、阴影、渐变和图标制造“科技感”。
- 在没有 controller 的情况下画一个看起来已部署的 closed-loop runtime system。
- 在没有 controlled contrast 的情况下把 block count 或 bandwidth 标成已证实原因。

---

## 15. AI draw.io 客户端分步工作流

建议每张图不要一次性要求 AI 完成所有细节，而采用以下顺序：

### Step 1：只生成布局骨架

提示 AI：

```text
Generate only the panel layout, panel labels, bounding regions, shared legend location, and alignment guides. Do not add data or decorative styling yet. Keep all panels grouped separately.
```

检查：hero panel 面积是否最大、阅读顺序是否自然、是否有重复 panel。

### Step 2：生成 schematic 元素

提示 AI 只增加系统框、内存组件、箭头和公式；先不画定量数据。

### Step 3：填入 verified data

逐个 panel 提供明确表格，不让 AI 从正文摘要猜数据。完成一个 panel 后锁定 group。

### Step 4：增加 uncertainty 与 disclosure

单独提示加入 CI、`n`、zero line、SLO line、FDR disclosure 和 `Not evaluated`。

### Step 5：统一视觉语义

检查相同配置在所有图中的颜色、marker 和名称是否完全一致。

### Step 6：导出与最终尺寸 QA

- 导出 `.drawio`。
- 导出 SVG，确认文字仍为 text object。
- 导出 PDF，确认字体可选中、没有 rasterized panel。
- 将图放入论文 7.1-inch 或 3.45-inch 实际宽度检查。
- 检查所有文字 ≥5 pt、CI 不与标签碰撞、legend 不遮挡数据。
- 黑白打印检查 configuration 仍能通过 marker/line style 区分。

---

## 16. 每张图最终提交前的科学 QA

对每张图逐项回答：

1. 用一句话写出该图的唯一核心结论。
2. 遮住任意 panel 后，论证是否真的缺一环？若没有，删除该 panel。
3. 理论预测与实测是否编码不同？
4. 未测值是否明确写 `Not evaluated`？
5. 是否展示了所有与结论相关的 seeds、workloads 或 configurations？
6. 是否存在选择 positive cells、隐藏 null results 或改变 denominator？
7. `n`、CI、SLO 和 multiple-comparison family 是否可见？
8. baseline 是否明确？ratio 的分母是否一致？
9. 图中是否出现论文正文没有实现的模块？
10. 该图是否帮助击破“existing flag + bookkeeping equation”的反方叙事？

若第 9 项为“是”或第 4–8 项无法回答，图不得进入投稿版本。

