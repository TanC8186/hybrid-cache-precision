#!/usr/bin/env bash
# 服务器诊断：确认环境状态 + 尝试装 pip
set -uo pipefail
LOG=/root/diag.log
{
echo "=== 1. 系统 ==="
cat /etc/os-release 2>/dev/null | grep -E "PRETTY_NAME|VERSION_ID"
echo "=== 2. python ==="
/root/MLSys_Research/.venv/bin/python --version 2>&1
command -v python3 && python3 --version 2>&1
ls /usr/bin/python3* 2>/dev/null
echo "=== 3. uv 缓存 ==="
du -sh /root/.cache/uv 2>/dev/null || echo "no uv cache"
echo "=== 4. pip wheel ==="
ls -la /tmp/pip.whl 2>/dev/null
echo "=== 5. 尝试装 pip（下载 pip wheel 到文件，校验 zip）==="
SP=$(/root/MLSys_Research/.venv/bin/python -c "import site; print(site.getsitepackages()[0])")
echo "site-packages=$SP"
timeout 30 /root/MLSys_Research/.venv/bin/python - <<PYEOF
import json, urllib.request, zipfile
# 用 tsinghua 镜像拿 pip wheel（国内快）
try:
    d = json.load(urllib.request.urlopen("https://pypi.tuna.tsinghua.edu.cn/pypi/pip/json", timeout=15))
except Exception as e:
    d = json.load(urllib.request.urlopen("https://pypi.org/pypi/pip/json", timeout=15))
url = [u["url"] for u in d["urls"] if u["filename"].endswith("py3-none-any.whl")][0]
fname = url.split("/")[-1]
print("pip", d["info"]["version"], "url", url[:80])
import subprocess
subprocess.run(["curl", "-sL", url, "-o", "/tmp/pip.whl"], timeout=30)
print("downloaded size:", __import__("os").path.getsize("/tmp/pip.whl"))
# 校验 + 解压
z = zipfile.ZipFile("/tmp/pip.whl")
z.testzip()
z.extractall("$SP")
print("pip extracted OK")
PYEOF
echo "=== 6. pip version ==="
/root/MLSys_Research/.venv/bin/python -m pip --version 2>&1 | head -1
echo "=== DONE ==="
} > "$LOG" 2>&1
echo "log written to $LOG"
