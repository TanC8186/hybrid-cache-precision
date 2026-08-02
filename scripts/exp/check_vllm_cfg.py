"""校验 vLLM 配置改动（per-layer dtype + int2/int3）导入正确。"""
import sys

import vllm.config.cache as c
import vllm.utils.torch_utils as t
import vllm.v1.kv_cache_interface as k

out = []
out.append(f"CacheDType has int2: {'int2_per_token_head' in str(c.CacheDType)}")
out.append(f"CacheDType has int3: {'int3_per_token_head' in str(c.CacheDType)}")
out.append(f"CacheConfig.per_layer field: {hasattr(c.CacheConfig, 'kv_cache_dtype_per_layer')}")
out.append(f"KVQuantMode int2/int3: {int(k.KVQuantMode.INT2_PER_TOKEN_HEAD)}/{int(k.KVQuantMode.INT3_PER_TOKEN_HEAD)}")
out.append(f"get_kv_quant_mode int2 -> {k.get_kv_quant_mode('int2_per_token_head')}")
out.append(f"int3 is_per_token_head: {k.KVQuantMode.INT3_PER_TOKEN_HEAD.is_per_token_head}")
out.append(f"torch dtype int2: {t.kv_cache_dtype_str_to_dtype('int2_per_token_head', None)}")
out.append(f"torch dtype int3: {t.kv_cache_dtype_str_to_dtype('int3_per_token_head', None)}")
out.append(f"is_quantized int3: {t.is_quantized_kv_cache('int3_per_token_head')}")

print("\n".join(out))
print("ALL OK")
