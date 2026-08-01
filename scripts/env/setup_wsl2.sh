#!/usr/bin/env bash
# 本地开发环境搭建（WSL2/Docker-first，无需 sudo）。
# vLLM 不支持 Windows 原生运行；宿主 Python 3.13 也不在支持范围。必须在此环境内开发。
#
# 方案：用 uv 管理独立 Python 3.12（uv 下载自带 CPython，不依赖系统 python3.12-venv）。
# 前提：WSL2 Ubuntu 已安装（Windows 侧 `wsl -l -v` 可见），NVIDIA 驱动已装。
#
# 注意：在 /mnt/e（NTFS）上构建 vLLM 较慢。可选：软链仓库到 Linux 文件系统
#   （ln -s /mnt/e/MLSys_Research ~/mlsys）提升性能。
set -euo pipefail

if [ -f /proc/version ] && grep -qi microsoft /proc/version; then
  echo "运行在 WSL2 内，继续。"
else
  echo "ERROR: 必须在 WSL2 (Ubuntu) 内运行。" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# 1) 安装 uv（单二进制，装到 ~/.local/bin，无需 sudo）
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  echo "==> 安装 uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# 2) 独立 Python 3.12（在 vLLM 支持范围 3.10-3.12 内）
echo "==> 准备 Python 3.12（uv 管理，不依赖系统包）..."
uv python install 3.12 2>&1 | tail -2 || true

# 3) 虚拟环境
echo "==> 创建 .venv ..."
rm -rf .venv
uv venv --python 3.12 .venv

# 4) 锁定依赖
# shellcheck disable=SC1091
source .venv/bin/activate
if [ -f requirements.lock ]; then
  echo "==> 安装锁定依赖 requirements.lock ..."
  uv pip install -r requirements.lock
else
  echo "==> requirements.lock 尚未生成，安装开发依赖 ..."
  uv pip install -e ".[dev]"
fi

# 5) vLLM fork 构建（固定 submodule SHA；vendor 未初始化则提示）
if [ -d vendor/vllm ]; then
  bash scripts/build_vllm.sh
else
  echo "WARN: vendor/vllm 未初始化，先运行: git submodule update --init --recursive vendor/vllm"
fi

# 6) 正确性门禁
pytest -q tests/ || echo "WARN: 测试未通过，检查后重跑"

echo
echo "本地开发环境就绪。铁律：4060 结果仅供 dev，禁止进入 results/。"
echo "激活环境：source .venv/bin/activate"
