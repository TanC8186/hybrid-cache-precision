#!/usr/bin/env bash
# 从 wheels.vllm.ai 装预编译 vLLM wheel（绕开构建）；依赖走阿里云 PyPI
set -uo pipefail
source /root/autodl-tmp/MLSys_Research/.venv/bin/activate
cd /root/autodl-tmp/MLSys_Research

COMMIT=e2fa28594f7baad142a426b0b6a2cfe2c79201c7
# 从 metadata 拿精确的 x86_64 wheel 文件名（版本含 +，不能用 %2B）
META_URL="https://wheels.vllm.ai/$COMMIT/cu130/vllm/metadata.json"
python - "$META_URL" > /root/vllm_wheel.log 2>&1 <<'PY'
import json, sys, urllib.request
url = sys.argv[1]
data = json.load(urllib.request.urlopen(url, timeout=20))
for w in data:
    if w.get("package_name") == "vllm" and "x86_64" in w.get("platform_tag", ""):
        print("WHEEL_URL=" + w.get("url", w.get("path", "")))
        break
PY
# url 可能是相对路径，拼全
WHEEL_URL=$(grep -oP 'WHEEL_URL=.*' /root/vllm_wheel.log | head -1 | sed 's/WHEEL_URL=//')
echo "WHEEL_URL=$WHEEL_URL" >> /root/vllm_wheel.log

# 如果相对路径，拼前缀
case "$WHEEL_URL" in
  http*) FULL="$WHEEL_URL" ;;
  /*) FULL="https://wheels.vllm.ai$WHEEL_URL" ;;
  ../*) FULL="https://wheels.vllm.ai/$COMMIT/cu130/vllm/$WHEEL_URL" ;;
esac
echo "FULL=$FULL" >> /root/vllm_wheel.log

python -m pip install "$FULL" --index-url "http://mirrors.aliyun.com/pypi/simple" >> /root/vllm_wheel.log 2>&1
echo "wheel install exit=$?" >> /root/vllm_wheel.log

# 覆盖 per-layer patch
SP=$(python -c "import site; print(site.getsitepackages()[0])")
for f in config/cache.py engine/arg_utils.py model_executor/layers/attention/attention.py utils/torch_utils.py v1/kv_cache_interface.py; do
  cp "vendor/vllm/vllm/$f" "$SP/vllm/$f" 2>>/root/vllm_wheel.log && echo "patched $f" >> /root/vllm_wheel.log || echo "PATCH FAIL $f" >> /root/vllm_wheel.log
done

python -c "import vllm; print('vllm', vllm.__version__)" >> /root/vllm_wheel.log 2>&1
python -c "from vllm.config.cache import CacheConfig; print('per_layer:', hasattr(CacheConfig, 'kv_cache_dtype_per_layer'))" >> /root/vllm_wheel.log 2>&1
tail -8 /root/vllm_wheel.log
