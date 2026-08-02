#!/usr/bin/env bash
# torch 2.13.0+cu130：wheel 从阿里云 pytorch 扁平目录（find-links），依赖从阿里云 PyPI
set -uo pipefail
source /root/autodl-tmp/MLSys_Research/.venv/bin/activate
python -m pip install 'torch==2.13.0' \
  --find-links 'https://mirrors.aliyun.com/pytorch-wheels/cu130/' \
  --index-url 'http://mirrors.aliyun.com/pypi/simple' \
  > /root/torch_install.log 2>&1
echo "torch_install exit=$?" >> /root/torch_install.log
python -c "import torch; print('torch OK', torch.__version__, torch.version.cuda)" >> /root/torch_install.log 2>&1
