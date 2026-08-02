#!/usr/bin/env bash
# 5090 环境一键搭建（RTX 5090, sm_120, CUDA 13）—— pip 版本
# 经验总结：uv 在此服务器死锁 → 全用 pip；github 被墙 → 镜像/阿里云；flashinfer-cubin 仅 github → 跳过用 Triton
#
# 用法（在 5090 上，项目已就位）: bash scripts/env/setup_5090.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "=== 前提检查 ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "WARN: 无 GPU（无卡模式，部分步骤可能受影响）"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
echo "CUDA_HOME=$CUDA_HOME"

# ---------- 1) venv (Python 3.12) + pip ----------
echo "=== 创建 venv + pip ==="
if [ ! -x "$HOME/.local/bin/uv" ]; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export PATH="$HOME/.local/bin:$PATH"
if [ ! -d .venv ]; then
  uv venv --python 3.12 .venv
fi
source .venv/bin/activate
# 确保 pip 可用（uv venv 无 pip → ensurepip 引导）
python -m pip --version >/dev/null 2>&1 || python -m ensurepip --upgrade

# ---------- 2) torch 2.13.0+cu130 ----------
echo "=== 安装 torch 2.13.0+cu130 ==="
python -m pip install "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -2
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# ---------- 3) git 镜像（github → ghproxy）----------
echo "=== git 镜像配置 ==="
git config --global url."https://ghproxy.com/https://github.com/".insteadOf "https://github.com/" 2>/dev/null || true

# ---------- 4) vLLM fork + patch ----------
echo "=== vLLM fork + per-layer patch ==="
VLLM="vendor/vllm"
VLLM_COMMIT="e2fa28594f7baad142a426b0b6a2cfe2c79201c7"
if [ ! -d "$VLLM" ]; then
  git clone --filter=blob:none --no-checkout https://ghfast.top/https://github.com/vllm-project/vllm "$VLLM" 2>/dev/null \
    || git clone --filter=blob:none --no-checkout https://github.com/vllm-project/vllm "$VLLM"
fi
git -C "$VLLM" checkout "$VLLM_COMMIT" 2>/dev/null \
  || { git -C "$VLLM" fetch origin "$VLLM_COMMIT" && git -C "$VLLM" checkout FETCH_HEAD; }
# unshallow（让 git merge-base 可用）
git -C "$VLLM" fetch --unshallow > /dev/null 2>&1 || true
git -C "$VLLM" config remote.origin.promisor false 2>/dev/null || true
if grep -q "kv_cache_dtype_per_layer" "$VLLM/vllm/config/cache.py"; then
  echo "patch 已应用，跳过"
else
  git -C "$VLLM" apply --ignore-whitespace --ignore-space-change "$ROOT/vendor/vllm-patches/per-layer-kv-dtype.diff"
fi

# ---------- 5) 构建依赖 + 运行时依赖 ----------
echo "=== 安装构建 + 运行时依赖 ==="
python -m pip install packaging numpy "setuptools>=77" "setuptools-scm>=8" setuptools-rust cmake ninja 2>&1 | tail -1
# flashinfer 从阿里云装（github 被墙）；flashinfer-cubin 仅 github → 跳过，用 Triton backend
if ! python -c "import flashinfer" 2>/dev/null; then
  echo "==> 从阿里云装 flashinfer ..."
  cd /tmp
  FI_URL=$(curl -sL "http://mirrors.aliyun.com/pypi/simple/flashinfer-python/" | grep -oE 'href="[^"]*flashinfer_python-[0-9.]+[^"]*\.whl"' | sed 's/href="//;s/"$//' | head -1)
  echo "flashinfer URL: $FI_URL"
  curl -sL -o /tmp/fi.whl "http://mirrors.aliyun.com$FI_URL" 2>/dev/null
  # 修正文件名
  FN=$(basename "$FI_URL" | cut -d'#' -f1)
  mv /tmp/fi.whl "/tmp/$FN" 2>/dev/null
  python -m pip install "/tmp/$FN" 2>&1 | tail -1
  cd "$ROOT"
fi
# 其余运行时依赖（去掉 flashinfer 行，避免 github）
cd "$VLLM"
sed '/flashinfer/d; s|^-r common.txt|-r /root/MLSys_Research/vendor/vllm/requirements/common.txt|' requirements/cuda.txt > /tmp/req_filt.txt
python -m pip install -r /tmp/req_filt.txt 2>&1 | tail -1
cd "$ROOT"

# ---------- 6) 构建 vLLM（precompiled, sm_120）----------
echo "=== 构建 vLLM (sm_120, precompiled) ==="
cd "$VLLM"
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_COMMIT="$VLLM_COMMIT"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.8.4.dev
export VLLM_VERSION_OVERRIDE=0.8.4.dev
export TORCH_CUDA_ARCH_LIST=12.0
export CUDA_HOME
export GIT_TERMINAL_PROMPT=0
python -m pip install -e . --no-build-isolation --no-deps 2>&1 | tail -3
cd "$ROOT"

# ---------- 7) 项目包 + 验证 ----------
echo "=== 安装项目包 + 验证 ==="
python -m pip install -e ".[dev]" 2>&1 | tail -1
python scripts/exp/check_vllm_cfg.py 2>&1 | tail -3 || echo "check_vllm_cfg 失败"
echo "=== 5090 环境就绪 ==="
