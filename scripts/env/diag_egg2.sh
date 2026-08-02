#!/usr/bin/env bash
# 不带 VLLM_USE_PRECOMPILED 测 egg_info
set -uo pipefail
cd /root/MLSys_Research/vendor/vllm
source /root/MLSys_Research/.venv/bin/activate
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.8.4.dev
export VLLM_VERSION_OVERRIDE=0.8.4.dev
export TORCH_CUDA_ARCH_LIST=12.0
export CUDA_HOME=/usr/local/cuda
export GIT_TERMINAL_PROMPT=0
timeout 40 python setup.py egg_info > /root/egg2.log 2>&1
echo "egg_info_no_precompiled exit=$?" >> /root/egg2.log
