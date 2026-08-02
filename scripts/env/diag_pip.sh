#!/usr/bin/env bash
# 前台跑 pip install -v 50s，输出到 /root/fg.log 供诊断
set -uo pipefail
cd /root/MLSys_Research/vendor/vllm
source /root/MLSys_Research/.venv/bin/activate
export VLLM_USE_PRECOMPILED=1
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.8.4.dev
export VLLM_VERSION_OVERRIDE=0.8.4.dev
export TORCH_CUDA_ARCH_LIST=12.0
export CUDA_HOME=/usr/local/cuda
export GIT_TERMINAL_PROMPT=0

timeout 50 python -m pip install -e . --no-build-isolation -v > /root/fg.log 2>&1
echo "fg done exit=$?" >> /root/fg.log
