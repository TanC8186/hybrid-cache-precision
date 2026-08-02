#!/usr/bin/env bash
# 下载 Qwen3.5-2B 到数据盘（ModelScope，国内快）
set -uo pipefail
source /root/autodl-tmp/MLSys_Research/.venv/bin/activate
export MODELSCOPE_CACHE=/root/autodl-tmp/caches/modelscope
cd /root/autodl-tmp/MLSys_Research
python - <<'PY' > /root/dl_model.log 2>&1
from modelscope import snapshot_download
p = snapshot_download("Qwen/Qwen3.5-2B", cache_dir="/root/autodl-tmp/caches/modelscope")
print("MODEL_DIR=", p)
PY
echo "dl exit=$?" >> /root/dl_model.log
tail -3 /root/dl_model.log
