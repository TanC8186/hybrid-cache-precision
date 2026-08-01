#!/usr/bin/env bash
# 本地开发环境搭建（WSL2/Docker-first）。
# vLLM 不支持 Windows 原生运行；宿主 Python 3.13 也不在支持范围。必须在此环境内开发。
#
# 前提：WSL2 Ubuntu 已安装（Windows 侧 `wsl -l -v` 可见），NVIDIA 驱动已装。
# 启动方式（Windows PowerShell / 本会话）：
#   wsl -d Ubuntu -- bash -lc "cd /mnt/e/MLSys_Research && bash scripts/env/setup_wsl2.sh"
#
# 注意：在 /mnt/e（NTFS）上构建 vLLM 较慢。可选：把仓库软链到 Linux 文件系统
#   （ln -s /mnt/e/MLSys_Research ~/mlsys）以提升性能，代码仍留在 Windows 侧。
set -euo pipefail

if [ -f /proc/version ] && grep -qi microsoft /proc/version; then
  echo "运行在 WSL2 内，继续。"
else
  echo "ERROR: 必须在 WSL2 (Ubuntu) 内运行。" >&2
  exit 1
fi

# 1) Python 3.10-3.12（vLLM 支持范围；Ubuntu 24.04 自带 3.12，优先用）
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY=""
for c in python3.12 python3.11 python3.10; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "==> 未找到 3.10-3.12，安装 python3.12 ..."
  sudo apt-get update
  sudo apt-get install -y python3.12 python3.12-venv python3.12-dev build-essential
  PY=python3.12
fi
echo "==> 使用 $PY ($("$PY" --version))"
command -v "$PY" >/dev/null || { echo "ERROR: $PY 不存在"; exit 1; }

# 2) 虚拟环境 + uv（Ubuntu 默认缺 python3.x-venv，需先装）
if ! "$PY" -m venv .venv 2>/tmp/venv_err.log; then
  echo "==> venv 创建失败，安装 ${PY}-venv ..."
  sudo apt-get update
  sudo apt-get install -y "${PY}-venv"
  rm -rf .venv
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip uv

# 3) 锁定依赖（lockfile 生成后启用）
if [ -f requirements.lock ]; then
  uv pip install -r requirements.lock
else
  uv pip install -e ".[dev]"
fi

# 4) vLLM fork 构建（固定 submodule SHA）
if [ -d vendor/vllm ]; then
  bash scripts/build_vllm.sh
else
  echo "WARN: vendor/vllm 未初始化，先运行: git submodule update --init --recursive vendor/vllm"
fi

# 5) 正确性门禁
pytest -q tests/ || echo "WARN: 测试未通过，检查后重跑"

echo
echo "本地开发环境就绪。铁律：4060 结果仅供 dev，禁止进入 results/。"
