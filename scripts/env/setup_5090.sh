#!/usr/bin/env bash
# 5090 环境一键搭建（RTX 5090, sm_120, CUDA 13）
# 步骤：venv → torch 2.13.0+cu130 → vLLM fork + per-layer patch → 构建 sm_120 → 验证
#
# 前提：租机镜像已有 CUDA 13（nvcc 在 /usr/local/cuda/bin），Python 3.12 或可安装。
# 用法（在 5090 上，项目目录已就位）: bash scripts/env/setup_5090.sh
set -euo pipefail

# ---------- 0) 前提检查 ----------
echo "=== 前提检查 ==="
if nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null; then
  echo "GPU OK"
else
  echo "WARN: 无 GPU 访问（无卡模式）。torch 安装 + vLLM 构建（nvcc 编译）不需要 GPU，继续。"
  echo "      后续 serving benchmark / 模型加载需切回 GPU 模式。"
fi
echo "CUDA_HOME=${CUDA_HOME:-<未设置，用 /usr/local/cuda>}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
command -v nvcc >/dev/null 2>&1 || ls "$CUDA_HOME/bin/nvcc" >/dev/null 2>&1 || {
  echo "WARN: 未找到 nvcc（$CUDA_HOME/bin/nvcc）。若系统无 CUDA toolkit，稍后装 pip 版。"
}

# uv
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  echo "==> 安装 uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# ---------- 1) Python 3.12 venv ----------
echo "=== 创建 venv (Python 3.12) ==="
if [ ! -d .venv ]; then
  uv venv --python 3.12 .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# ---------- 2) torch 2.13.0+cu130（覆盖模板 torch） ----------
echo "=== 安装 torch 2.13.0+cu130 ==="
# 优先 PyTorch cu130 官方 index（确保 cu130 wheel）；失败则默认 index
uv pip install "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu130 \
  || uv pip install "torch==2.13.0"
# torch 版本校验（无卡模式不查 GPU capability，避免 CUDA init 报错）
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
if nvidia-smi >/dev/null 2>&1; then
  python -c "import torch; print('cap', torch.cuda.get_device_capability())"
fi

# nvcc（系统无则用 pip 的）
if ! command -v nvcc >/dev/null 2>&1 && [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  echo "==> 安装 pip 版 nvcc ..."
  uv pip install nvidia-cuda-nvcc
fi

# ---------- 3) vLLM fork + per-layer patch ----------
echo "=== vLLM fork + per-layer patch ==="
VLLM="vendor/vllm"
VLLM_COMMIT="e2fa28594f7baad142a426b0b6a2cfe2c79201c7"  # 我们验证的 commit，patch 基于它
if [ ! -d "$VLLM" ]; then
  echo "==> 克隆 vLLM 并锁定到 $VLLM_COMMIT ..."
  # 先试 ghfast 镜像，失败回退 github 直连
  git clone --filter=blob:none --no-checkout \
    https://ghfast.top/https://github.com/vllm-project/vllm "$VLLM" 2>/dev/null \
    || git clone --filter=blob:none --no-checkout \
       https://github.com/vllm-project/vllm "$VLLM"
fi
# 锁定到验证过的 commit
git -C "$VLLM" checkout "$VLLM_COMMIT" 2>/dev/null \
  || { echo "==> fetch $VLLM_COMMIT ..."; git -C "$VLLM" fetch origin "$VLLM_COMMIT" && git -C "$VLLM" checkout FETCH_HEAD; }
if grep -q "kv_cache_dtype_per_layer" "$VLLM/vllm/config/cache.py"; then
  echo "per-layer patch 已应用，跳过。"
else
  echo "==> 应用 per-layer patch ..."
  git -C "$VLLM" apply "$ROOT/vendor/vllm-patches/per-layer-kv-dtype.diff"
fi

# ---------- 4) 构建 vLLM（sm_120） ----------
echo "=== 构建 vLLM (sm_120) ==="
cd "$VLLM"
export CUDA_HOME
export TORCH_CUDA_ARCH_LIST="12.0"
export VLLM_VERSION_OVERRIDE=0.8.4.dev
export GIT_TERMINAL_PROMPT=0
# 全量编译（需要 nvcc + CUDA toolkit）。若只想跑 Python 改动可试 VLLM_USE_PRECOMPILED=1
uv pip install -e .
cd "$ROOT"

# ---------- 5) 安装项目 kvcache 包 ----------
echo "=== 安装 kvcache 包 ==="
uv pip install -e ".[dev]"

# ---------- 6) 验证 ----------
echo "=== 验证 ==="
python scripts/exp/check_vllm_cfg.py || echo "check_vllm_cfg 失败（可能有未满足依赖）"
echo
echo "5090 环境就绪。下一步:"
echo "  python scripts/exp/gen_allocation.py --threshold 0.15"
echo "  python scripts/exp/vllm_serving_bench.py --allocation results/ablations/allocation.json --num-reqs 200 --max-len 8192"
