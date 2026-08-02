#!/usr/bin/env bash
# 在服务器上用 pip 构建 vLLM（后台）。用法: bash launch_vllm_build.sh
set -uo pipefail
cd /root/MLSys_Research/vendor/vllm || exit 1
source /root/MLSys_Research/.venv/bin/activate || exit 1
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=12.0
export VLLM_VERSION_OVERRIDE=0.8.4.dev
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.8.4.dev
export GIT_TERMINAL_PROMPT=0
# 用预编译内核（跳过源码构建 + github 外部依赖）。若需源码构建（2/3-bit 内核）再去掉。
export VLLM_USE_PRECOMPILED=1
# 直接指定预编译 wheel 的 commit（绕开坏掉的 git merge-base，避免卡 metadata）
export VLLM_PRECOMPILED_WHEEL_COMMIT=e2fa28594f7baad142a426b0b6a2cfe2c79201c7

# --no-build-isolation：用 venv 已装的构建依赖；--no-deps：依赖已手动装好（含跳过的 flashinfer）
nohup python -m pip install -e . --no-build-isolation --no-deps > /root/pip_vllm.log 2>&1 &
echo "BUILD_LAUNCHED pid=$!"
sleep 3
tail -5 /root/pip_vllm.log 2>/dev/null
echo "---"
ps aux | grep 'pip install' | grep -v grep | head -1 | cut -c1-70
