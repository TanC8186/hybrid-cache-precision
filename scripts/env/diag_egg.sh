#!/usr/bin/env bash
set -uo pipefail
cd /root/MLSys_Research/vendor/vllm
source /root/MLSys_Research/.venv/bin/activate
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_COMMIT=e2fa28594f7baad142a426b0b6a2cfe2c79201c7
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.8.4.dev
export VLLM_VERSION_OVERRIDE=0.8.4.dev
export TORCH_CUDA_ARCH_LIST=12.0
export CUDA_HOME=/usr/local/cuda
export GIT_TERMINAL_PROMPT=0
export PYTHONFAULTHANDLER=1
timeout 40 python setup.py egg_info > /root/egg.log 2>&1
echo "egg_info exit=$?" >> /root/egg.log
