# per-layer 混 dtype 独立 page group 设计（2026-08-03）

> 任务：调研 vLLM V1 KV cache manager 的 page 分配机制，评估"支持混 dtype 独立 page group"
> 的可行性，产出设计文档。**只调研和设计，不写实现代码。**
>
> 相关代码：`vendor/vllm`（submodule fork），per-layer dtype patch 见
> `vendor/vllm-patches/per-layer-kv-dtype.diff`。
> 本设计只针对 vLLM V1（`vllm/v1/**`），不涉及 V0 路径。

---

## 0. 结论速览

- **根因**：vLLM V1 的 KV cache 配置层要求"共享 block pool 的各 group 物理 page 大小统一"。
  混 dtype（L23 bf16 + 其余 int4）让 `unify_kv_cache_spec_page_size`
  （`vllm/v1/core/kv_cache_utils.py:1070`）把 int4 层 block_size 从 16 放大到 64
  （page 2064→8256 B），随后 `_get_kv_cache_groups_uniform_page_size` 把每层拆成独立 group
  （Qwen3.5-2B 有 6 个 GQA 层 → 6 个 GQA group + GDN mamba groups），
  容量从 uniform int4 的 2,701,721 塌到 696,456（×0.258，低于 fp16 的 1,203,106）。
- **方案对比**：
  - **方案 A（独立 page group）**：每 dtype 一组、各组用自己的 page_size/num_blocks。可恢复混 dtype
    理论容量（Qwen3.5-2B 下 ≈ uniform int4 的 0.83–0.91，约 2.2–2.5M tokens，推导见 §1.3）。推荐。
    - 子方案 A2（packed slab 复用现有 `offset/block_stride` 机制）：改动集中在
      `kv_cache_utils.py` + 一处 worker reshape，**~1 周**。
    - 子方案 A1（per-group block pool）：更彻底但改动面大，**≥2 周**。
  - **方案 B（page 对齐 + byte 分摊）**：不可行/无增益。`block_size` 放大本身就是"byte 分摊"，
    已由 `unify_kv_cache_spec_page_size` 实现；单独使用只会重复当前 collapse。
  - **方案 C（skip_layers 替代 per-layer）**：**同样触发 page 统一**。`platforms/interface.py`
    `_align_heterogeneous_kv_block_size`（line 654）会同样把 int4 主层 block_size 顶到 64、
    把 skip 层 `page_size_padded` 填到共享 page（8256）→ 容量与 per-layer 相同，**不是修复**。
- **推荐**：方案 A2（先做 packed 化，复用 `_get_packed_kv_cache_layout`），把 GQA 层合并成一个
  `UniformTypeKVCacheSpecs` group（混 dtype、per-layer page），GDN mamba 层保持独立 group；
  风险集中在内存核算与 Mamba reshape，需重点验证。A1 作为后续架构方向。

---

## 1. 机制分析：为什么混 dtype 触发 page 统一？

### 1.1 数据结构速览（先建立心智模型）

- `KVCacheSpec`（`vllm/v1/kv_cache_interface.py:115`）：每层 KV cache 格式描述。
  `block_size`=每 block token 数；`page_size_bytes`=每 block 物理字节。
  `FullAttentionSpec.real_page_size_bytes`（line 344）= `block_size × num_kv_heads × last_dim × dtype_size`，
  其中 int4 的 `last_dim = head_size/2 + head_size_v/2`（打包 2×int4/byte），并外加 per-token-head scales。
- `KVCacheGroupSpec`（line 973）：一组共享同一 block table 的 layer + 合并后的 spec。
- `KVCacheConfig`（line 988）：`num_blocks`（唯一全局数）+ `kv_cache_tensors`（每层/共享 tensor 布局）
  + `kv_cache_groups`。
- `BlockPool`（`vllm/v1/core/block_pool.py:143`）：**全局唯一** block 池，`num_gpu_blocks` 个
  `KVCacheBlock`，一个 free 链表、一个 prefix-cache hash 表（`BlockHashWithGroupId` 已含 group_id）。
- `KVCacheCoordinator`（`vllm/v1/core/kv_cache_coordinator.py:60`）：创建**一个** BlockPool，
  每个 group 一个 `SingleTypeKVCacheManager`，全部 manager 从**同一池**取 block。

### 1.2 触发链（Qwen3.5 混 dtype 的实际路径）

Qwen3.5 是混合架构：GDN 线性注意力层 → `MambaSpec`（经 `LinearAttentionBackend`，
见 `vllm/model_executor/layers/mamba/abstract.py:63`）；GQA 全注意力层 → `FullAttentionSpec`
（经 `attention.py get_kv_cache_spec`，line 638）。per-layer patch 让 L23 的 `kv_quant_mode=NONE`、
其余 GQA 层 `INT4_PER_TOKEN_HEAD`，因此 **GQA 层之间 page_size_bytes 不同**。

`get_kv_cache_groups`（`kv_cache_utils.py:1781`）判定路径：

1. `is_kv_cache_spec_uniform`（line 912）→ False：`FullAttentionSpec.merge` 断言所有
   `AttentionSpec` 字段相等，`kv_quant_mode` 不同 → 抛 AssertionError。
2. `UniformTypeKVCacheSpecs.from_specs`（line 889）→ None：`is_uniform_with_collection`
   要求**所有** spec（含 MambaSpec）都是 `FullAttentionSpec` 实例 → MambaSpec 使 False。
   ⚠️ 关键点：如果模型全是 FullAttention（无 GDN/mamba），混 dtype 会直接走
   `UniformTypeKVCacheSpecs` 单组路径（见 §2.2 的"隐藏的后门"），**根本不会塌**。
   塌方只发生在 GDN+mixed-dtype 同时出现时。
3. 进入混合路径：`unify_kv_cache_spec_page_size`（line 1070）把**所有** spec 的
   `page_size_bytes` 统一到最大（bf16 page）。对 int4 GQA 层，因为
   `max_page % int4_page == 0`，走 `replace(spec, block_size=int4_block_size × 4)`
   （line 1113-1116），即 **int4 层 block_size 16→64、page 2064→8256**。
   MambaSpec 则被 `page_size_padded = max_page`（line 1108）。
4. `_get_kv_cache_groups_uniform_page_size`（line 1140）按**精确 spec** 分组：
   5 个 int4 GQA（同 spec）一组 bucket、1 个 bf16 GQA 一组 bucket。`min_num_layers=1` →
   `group_size=1` → 每个 GQA 层一个独立 group；Mamba GDN 再拆若干 group。
5. `get_kv_cache_config_from_groups`（line 1361）general case 要求
   `get_uniform_page_size`（line 1013，**断言所有 group page 相等**）→ 满足（已统一到 8256），
   但 `group_size=1`、`num_blocks = available // 8256`，tensor 布局退化，容量崩塌。

### 1.3 为什么容量塌到 0.258（而非理论 ~0.9）

- **page 统一本身不损失 int4 密度**：int4 block_size 16→64 后，per-token 字节不变
  （2064/16 = 8256/64 = 129 B）。理想混 dtype 的容量损失只来自"1 个 GQA 层从 int4 变 bf16"：
  GQA 部分 per-token 从 `6×i` 变 `5×i + b`，其中 `b ≈ 2.2456×i`（int4/fp16 实测容量比），
  即 `(5+2.2456)/6 = 1.2076`；计入 GDN mamba 层字节（混 dtype 不改变它们）后，整体 ratio
  落在 `[1.10, 1.21]`，即**修复后容量 ≈ uniform int4 的 0.83–0.91（约 2.2–2.5M @2B）**，
  仍是 fp16 的约 1.9–2.1×。
- **真正的杀伤来自 group 数爆炸 + 布局退化**：
  1. 每层独立 group（`group_size=1`）后，general case 为每层分配独立的
     `page_size × num_blocks` tensor，物理分配 ≈ `层数 × available`（过承诺）；
     即使不 OOM，`num_blocks` 也被 `available // max_page` 卡小。
  2. `scheduler_block_size = LCM(64,16) = 64`（`resolve_kv_cache_block_sizes`，line 659），
     prefix-cache 命中/分块对齐粒度从 16 变 64，短请求容量浪费加剧。
  3. `get_max_concurrency_for_kv_cache_config`（line 937）把每个 group 的
     `cdiv(max_mem, page_size)` 相加，25+ 个 group 使 `num_blocks_per_request` 虚高。
- 实测 2B/9B 衰减一致 ×0.258 → 机制确定（非个别配置噪声）。

### 1.4 结论

V1 的多 group 架构**本身支持不同 block_size**（`MultiGroupBlockTable`、
`HybridKVCacheCoordinator`、`BlockHashListWithBlockSize` 都为此而设），
但配置层 `get_kv_cache_config_from_groups` 和内存核算 `_max_memory_usage_bytes_from_groups`
强依赖"所有 group 共享统一 page_size"。**改造目标 = 打破这一假设。**

---

## 2. 方案 A：独立 page group

### 2.1 目标布局（Qwen3.5-2B 为例）

```
kv_cache_groups:
  group 0: UniformTypeKVCacheSpecs { 5× int4 GQA + 1× bf16 GQA(L23) }
           block_size=16（各层相同），per-layer page: int4=2064, bf16=8256
  group 1..k: MambaSpec（GDN 层，独立 block_size/page）
```

GQA 全部层共享**一个** block table（block ID 全局一致，指向每层自己 page 大小的槽位），
这与现有 `UniformTypeKVCacheSpecs` 单组路径完全一致，只是 Mamba 组并存。

### 2.2 实现要点（按组件拆解）

#### (a) 分组：`get_kv_cache_groups`（kv_cache_utils.py:1781）

新增分支：在 `UniformTypeKVCacheSpecs.from_specs` 之后、进入混合路径之前，
**把同 `uniform_type_base_spec` 的注意力层（FullAttentionSpec 一族）单独抽出，
不参与 `unify_kv_cache_spec_page_size`**：

```python
full_attn_specs = {k: v for k, v in kv_cache_spec.items()
                   if isinstance(v, FullAttentionSpec)}   # 含 int4/bf16
if len(full_attn_specs) > 0 and len(full_attn_specs) < len(kv_cache_spec):
    uniform = UniformTypeKVCacheSpecs.from_specs(full_attn_specs)  # 同 block_size + 同类型 → 成功
    groups = [KVCacheGroupSpec(list(uniform.kv_cache_specs.keys()), uniform)]
    # 剩余 Mamba/HiddenState 层走原有 unify + _get_kv_cache_groups_uniform_page_size
```

要点：
- `UniformTypeKVCacheSpecs.from_specs` 只要求同 `block_size` + 同类型，**不要求同 page_size**
  （`is_uniform_type`，line 874）。int4/bf16 GQA 均 block_size=16 且都是 FullAttentionSpec → 成功。
- 需先确认 `_promote_local_kv_cache_specs`/`disable_hybrid_kv_cache_manager` 的交互
  （`get_kv_cache_groups` 顶部已有，line 1794）。
- 为降低回归风险，可用新配置开关
  `--enable-per-layer-page-groups`（或复用 `kv_cache_dtype_per_layer` 非空即启用），
  默认关闭时行为与现网一致。

#### (b) 配置生成：`get_kv_cache_config_from_groups`（line 1361）

现状 general case 要求统一 page。改为：**当各 group page_size 不统一且是纯 attention+mamba 时，
走 packed 布局**：

- 扩展 `_use_packed_kv_cache_config`（line 1308）：从 `is_dsv4`/`enable_cross_layers_blocks`
  扩展到"存在 page_size 不统一"的情况（或新开关）。
- `_get_packed_kv_cache_layout`（line 1283）**已支持非 UniformType group**（Mamba 用
  `spec.page_size_bytes`），且已支持 UniformType 的 per-layer page。block_stride =
  max(各 group 层字节和)。无需大改。
- `_get_kv_cache_config_packed`（line 1330）：`num_blocks = available // block_stride`，
  每层一个 `KVCacheTensor(size=total, offset=byte_offset, block_stride=block_stride)`。
  这是**单个共享 block 池**，`kv_cache_config.num_blocks` 仍是全局一个数 → 调度/coordinator 零改动。

#### (c) worker 侧 reshape：`gpu_model_runner.py::_reshape_kv_cache_tensors`（line 7356）

- **attention 层**：`_reshape_attention_kv_cache`（`worker/gpu/attn_utils.py:211`）已支持
  `packing=(offset, block_stride)`（line 225-233），无改动。
- **Mamba 层**：当前 Mamba 分支（line 7447-7458）按 `raw_tensor[:num_blocks*page_size]` 切
  **偏移 0 连续页**，packed 下必须改为 strided view：
  ```python
  if packing is not None:
      offset, blk_stride = packing
      raw = raw_tensor.view(-1, blk_stride)[:, offset:offset+page_size_bytes].reshape(-1)
  else:
      raw = raw_tensor
  kv_caches[layer_name] = raw[:num_blocks*page_size_bytes].view(num_blocks, 1, 1, page_size_bytes)
  ```
- `_allocate_kv_cache_tensors`（line 7304）的 packed 分支只分配一块 backing 并 alias，已支持。

#### (d) 内存核算：`_max_memory_usage_bytes_from_groups`（line 1890）

现状只有"全 UniformType（DSV4）"特例；混 GQA-UniformType + Mamba 会落到 general case 的
`get_uniform_page_size` **断言失败**。需新增 packed 分支：
`total = block_stride × num_blocks`（或各 group `cdiv(max_mem, page_size) × block_stride` 求和），
与 `_get_packed_kv_cache_config` 保持一致，否则 startup `_check_enough_kv_cache_memory`
（line 2186）与运行时 admission 对不上。
`_pool_bytes_per_block`（line 972）已含 packed 分支，`num_gpu_blocks_override` 路径兼容。

#### (e) 调度/coordinator：零改动（A2 的核心红利）

- `BlockPool` 仍全局一个、`num_blocks` 仍全局一个 → `allocate_slots` 的 free-block 检查、
  `KVCacheCoordinator.get_num_blocks_to_allocate` 求和、`find_longest_cache_hit`
  （按 group 的 `BlockHashWithGroupId` 查）全部不变。
- `MultiGroupBlockTable` 已按 group 支持不同 block_size；CUDA graph 的 block table tensor
  按 group 构建（`_get_block_table`，gpu_model_runner.py:2325），group 数少（GQA 合并为 1）
  对 graph 友好。
- `needs_kv_cache_zeroing`（`has_mixed_precision_kv_cache`，kv_cache_interface.py:1011）已对
  混精度返回 True → worker 对新块清零，packed 下继续成立。

### 2.3 改动量评估

| 文件 | 改动 |
|---|---|
| `vllm/v1/core/kv_cache_utils.py` | 分组分支 (~30 行)、`_use_packed` 扩展 (~5)、`_max_memory_usage_bytes_from_groups` packed 分支 (~20) |
| `vllm/v1/worker/gpu_model_runner.py` | Mamba reshape packing 处理 (~10) |
| `vllm/config/cache.py` + `arg_utils.py` | 可选新开关（若不复用 per_layer 判空） |
| 测试 | `tests/v1/core/test_kv_cache_utils.py` 新增 ~2 用例 |

- 若由熟悉该代码库的工程师实施：**纯改动 ~1 周**（含单元测试）；加 smoke + 容量回归 **1-1.5 周**。
- 主要风险点（§2.4）集中在内存核算一致性和 Mamba/attention 布局混合，测试要覆盖。

### 2.4 风险

1. **内存核算漂移**：`_max_memory_usage_bytes_from_groups` 与 packed 布局不一致会导致
   startup 通过但运行时 OOM/死锁（项目历史有 #39734 前科）。必须用同一 `block_stride`。
2. **`_update_hybrid_attention_mamba_layout`（gpu_model_runner.py:7499）**：packed strided view
   与 blocks-first 归一化的 `as_strided_` 交互需要验证（attention 与 mamba 并存时触发）。
3. **回归面**：`_use_packed_kv_cache_config` 一旦放开会影响所有多 group 模型。建议用开关
   限定，先只对 `kv_cache_dtype_per_layer` 生效。
4. **per-token-head scales 的独立 tensor**：int4 的 scales 由 attention backend 管理
   （`unpadded_page_size_bytes` 已预算），packed 按 byte offset 摆放不冲突，但需在
   `_reshape_attention_kv_cache` 确认 page_bytes 计算含 scales。

---

## 3. 方案 B：page size 对齐但 byte 分摊

**定义**：保持所有 group 统一 page_size，但让 int4 层在同样 page 里塞更多 token
（即"byte 分摊"）。

**结论：不可行/无增益——这就是 `unify_kv_cache_spec_page_size` 已经做的事。**

- 统一 page 时 int4 层 `block_size` 放大到 64，per-token 字节保持 129 B，**分摊已经发生**；
  问题从来不在 per-token 密度，而在 §1.3 的 group 数爆炸 + 布局退化 + LCM 对齐。
- 任何"保持统一 page + 改分摊"的变体都无法绕开 `get_kv_cache_config_from_groups` 对
  `group_size`/`page_size` 的假设；要让 int4 层恢复 block_size=16，就必须让 int4 层与
  bf16 层 **page_size 不同** → 就变成了方案 A。
- 唯一沾边的改进：若坚持统一 page，可把 int4 层 block_size 放大的同时**避免 group 拆分**
  （把 int4 层合并成 1 组 + bf16 1 组），按 `group_size=2` 的 general-case 布局粗算容量约为
  uniform int4 的 0.45（远好于 0.258，但远低于 A 的 0.83–0.91），
  且需要改 `_get_kv_cache_groups_uniform_page_size` 的分组策略——收益低、复杂度与 A 相当。
- **不作为候选。**

---

## 4. 方案 C：放弃 per-layer，只做 uniform + skip_layers 保护

**问题：skip_layers 是否同样触发统一 page？——是，已从代码确认。**

`platforms/interface.py::_align_heterogeneous_kv_block_size`（line 654，Phase 3，
在 `update_block_size_for_backend` 里无条件执行）：

1. `primary_page` = int4 per-token page；`padded_pages` 含 skip 层 bf16 per-token page（line 709）。
2. `largest_padded_page`（bf16）> `primary_page`（int4，约 4×）。
3. `primary_block_size = block_alignment × cdiv(required_page, block_alignment × primary_page)`
   ≈ 16 × 4 = **64**（line 744-748）→ int4 主层 block_size 从 16 顶到 64。
4. `skip_page_size_padded = cache_config.block_size × primary_page` = **8256**（line 756-758），
   skip 层 `page_size_padded=8256`。

随后 `get_kv_cache_spec`（attention.py:662）读 `skip_page_size_padded` 生成 bf16 层 spec，
`unify_kv_cache_spec_page_size` 因 page 已统一而保持平凡。**最终效果与 per-layer patch 完全一致**
（int4 层 block_size 64 / page 8256，容量同样塌到 0.258）。

**结论：方案 C 不是修复，是同一个问题的另一种触发方式。** 它的唯一价值是"不新增代码"，
但论文方法（末层保护）的容量收益拿不到。若想走 C，仍需做方案 A 才能恢复容量；
C 可作为 A 上线前的**对照组**（确认 C 容量 ≈ 现 per-layer 容量，机制自洽）。

---

## 5. 推荐

**推荐方案 A2（packed slab + 单共享 pool）作为落地路径。**

理由（对比表）：

| 维度 | A2（packed） | A1（per-group pool） | C（skip_layers） |
|---|---|---|---|
| 预期容量（2B，L23 保护） | ~2.2–2.5M（uniform int4 的 0.83–0.91） | 同 A2 | 0.696M（×0.258） |
| 改动文件 | 2-3 | 6-8 | 0 |
| 调度/coordinator 改动 | 无 | 有（admission/prefix-cache/zeroing） | 无 |
| 风险 | 内存核算一致 + Mamba reshape | 高（多池 admission/驱逐） | 低但无效 |
| 工时 | ~1 周 | ≥2 周 | 0（不推荐） |

- **为什么先 A2**：V1 的 `KVCacheTensor(offset, block_stride)` + `_get_packed_kv_cache_layout`
  就是为"多组不同 page 密集摆放"而生的（DeepSeek V4 已用），只是目前被
  `_use_packed_kv_cache_config` 限定在 `is_dsv4`/cross-layers。放开它 + 修 Mamba reshape
  + 对齐内存核算，即可在不碰调度器的情况下拿到容量收益。
- **A1 何时再上**：当需要跨 dtype **动态再平衡**（int4 层空闲块给 bf16 层用）或
  消除 packed 的块内 padding 浪费时，A1（per-group pool）是正确架构。论文当前只需要
  static capacity，A2 足够。
- **落地顺序**：
  1. 开关 + 分组分支（GQA 合并 UniformType，Mamba 独立）；
  2. packed 布局放开 + Mamba reshape 修 offset；
  3. 内存核算 packed 分支 + startup 校验；
  4. 单测 + 容量对比 + smoke。

---

## 6. 验证方法

### 6.1 单元测试（`vendor/vllm/tests/v1/core/`）

- **`test_kv_cache_utils.py`**：
  - 新增：构造 5×int4 FullAttentionSpec + 1×bf16 + 若干 MambaSpec 的 `kv_cache_spec`，
    断言 `get_kv_cache_groups` 输出 1 个 `UniformTypeKVCacheSpecs`(GQA) + N 个 Mamba group，
    且**不调用 `unify_kv_cache_spec_page_size` 改变 GQA 的 block_size**。
  - 新增：`get_kv_cache_config_from_groups` 对该 groups 走 packed 分支，断言
    `block_stride == sum(page_size)`、`num_blocks == available // block_stride`、
    每层 `KVCacheTensor.offset` 正确、`_max_memory_usage_bytes_from_groups` 与
    `block_stride × num_blocks` 一致。
  - 参考既有：`test_get_max_concurrency_packed_kv_cache_config`（line 1541）、
    `test_mixed_precision_kv_cache_with_uniform_type_specs`（line 2072）、
    `test_contiguous_kv_packing.py`。
- **`test_contiguous_kv_packing.py`**：新增"混 dtype UniformType + Mamba"用例，断言
  group 间 offset 不相交、每层 num_blocks 一致、Mamba 层能按 offset 取页。

### 6.2 容量对比（决定性指标）

对 Qwen3.5-2B（本地 4060 dev）与 9B（5090 最终）跑：

```
配置 A: uniform int4                        → 期望 2,701,721（基线）
配置 B: uniform fp16/bf16                    → 期望 1,203,106（基线）
配置 C: 现 per-layer（L23 bf16 保护）         → 期望 696,456（复现塌方）
配置 D: 方案 A 后 per-layer（L23 bf16 保护）  → 期望 ≈ 2.2–2.5M（uniform int4 的 0.83–0.91）
```

- 断言：`D/C ≥ 3.0×`、`D/B ≥ 1.8×`、`D/A ∈ [0.80, 0.92]`（容差 ±5%）。
- 读取 `get_kv_cache_configs` 日志的 "GPU KV cache size" 或 `cache_config.kv_cache_size_tokens`
  （`kv_cache_utils.py:2231`），统一 `gpu_memory_utilization` 与 `max_model_len`。
- 跨 2B/9B 衰减一致（现状 ×0.258），修后应同为 ~0.83–0.91。

### 6.3 smoke（正确性）

- 单请求 + 多请求（含 prefix-cache hit）：`vllm serve` 小模型，断言生成一致、不 OOM。
- `needs_kv_cache_zeroing` 生效（混精度）→ 新块清零路径正常。
- CUDA graph 开/关各跑一次（packed strided view 与 graph 兼容性）。
- `num_gpu_blocks_override` 路径（`_pool_bytes_per_block`）冒烟。
- 现有 hybrid 回归：Qwen3.5 不开 per-layer（纯 uniform）时，行为与改造前逐位一致。

### 6.4 回归保护

- 跑 `tests/v1/core/test_kv_cache_utils.py`、`tests/v1/core/test_contiguous_kv_packing.py`、
  `tests/v1/core/test_kv_cache_manager.py`（如存在）。
- 跑 1-2 个既有 hybrid 模型单测（mamba+attention）确保 `_use_packed` 开关默认关闭无回归。

---

## 7. 关键文件/函数索引（供实现工程师对照）

| 组件 | 文件:行 |
|---|---|
| `unify_kv_cache_spec_page_size` | `vllm/v1/core/kv_cache_utils.py:1070` |
| `get_uniform_page_size`（断言单 page） | 同文件:1013 |
| `get_kv_cache_groups`（入口） | 同文件:1781 |
| `_get_kv_cache_groups_uniform_page_size` | 同文件:1140 |
| `get_kv_cache_config_from_groups`（general case） | 同文件:1361 |
| `_get_packed_kv_cache_layout` / `_get_kv_cache_config_packed` | 同文件:1283 / 1330 |
| `_use_packed_kv_cache_config` | 同文件:1308 |
| `_max_memory_usage_bytes_from_groups` | 同文件:1890 |
| `get_max_concurrency_for_kv_cache_config` | 同文件:937 |
| `resolve_kv_cache_block_sizes`（scheduler/hash block size） | 同文件:626 |
| `BlockPool`（单池） | `vllm/v1/core/block_pool.py:143` |
| `KVCacheCoordinator.__init__`（建单池） | `vllm/v1/core/kv_cache_coordinator.py:90` |
| `UniformTypeKVCacheSpecs`（per-layer page） | `vllm/v1/kv_cache_interface.py:852` |
| `FullAttentionSpec.real_page_size_bytes` | 同文件:344 |
| `KVCacheConfig.has_mixed_precision_kv_cache` | 同文件:1011 |
| per-layer dtype override 消费者 | `vllm/model_executor/layers/attention/attention.py:317` |
| `Attention.get_kv_cache_spec` | 同文件:638 |
| skip_layers 对齐（方案 C 依据） | `vllm/platforms/interface.py:654` |
| `_allocate_kv_cache_tensors` | `vllm/v1/worker/gpu_model_runner.py:7304` |
| `_reshape_kv_cache_tensors`（Mamba 分支待改） | 同文件:7356 / 7447 |
| `_reshape_attention_kv_cache`（packed 已支持） | `vllm/v1/worker/gpu/attn_utils.py:211` |
| `MultiGroupBlockTable` | `vllm/v1/worker/block_table.py:20` |

---

## 8. 未决问题（实现前需确认）

1. Qwen3.5 的 GDN 层在 `disable_hybrid_kv_cache_manager` 下是否会被
   `_promote_local_kv_cache_specs` 改型，影响新分组分支。
2. packed 布局下 `_update_hybrid_attention_mamba_layout` 对 strided view 调 `as_strided_`
   的正确性（attention/mamba 并存时）。
3. per-token-head scales 是否需要在 `_reshape_attention_kv_cache` 的 page_bytes 计算中
   显式含入（`unpadded_page_size_bytes` 已预算，但 reshape 用 `kv_cache_shape[1:]*dtype`）。
4. `num_gpu_blocks_override` 场景下 packed 分支的 `_pool_bytes_per_block` 是否要与
   新 `_max_memory_usage_bytes_from_groups` 完全一致。
5. 是否引入独立开关（`--enable-per-layer-page-groups`）以避免默认路径回归。
