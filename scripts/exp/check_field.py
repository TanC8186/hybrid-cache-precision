"""聚焦测试：kv_cache_dtype_per_layer 字段是否被 vLLM config 系统注册。"""
import dataclasses
import sys

import vllm.config.cache as c

# 1) 字段在源码中？
src = open(c.__file__).read()
print(f"[1] field in source: {'kv_cache_dtype_per_layer' in src}")

# 2) dataclasses.fields 是否列出它？
fnames = [f.name for f in dataclasses.fields(c.CacheConfig)]
print(f"[2] in dataclass fields: {'kv_cache_dtype_per_layer' in fnames}")
print(f"    all cache fields count: {len(fnames)}")

# 3) 实例化并读写
cfg = c.CacheConfig()
print(f"[3] default value: {cfg.kv_cache_dtype_per_layer}")
cfg.kv_cache_dtype_per_layer = {"3": "int2_per_token_head", "23": "int4_per_token_head"}
print(f"    after set: {cfg.kv_cache_dtype_per_layer}")

# 4) get_field 能否拾取（CLI 注册用）
from vllm.config.utils import get_field
try:
    f = get_field(c.CacheConfig, "kv_cache_dtype_per_layer")
    print(f"[4] get_field ok: {f}")
except ValueError as e:
    print(f"[4] get_field FAIL: {e}")

# 5) 用 skip_layers 对照（它应该正常）
print(f"[5] skip_layers in fields: {'kv_cache_dtype_skip_layers' in fnames}")
print("DONE")
