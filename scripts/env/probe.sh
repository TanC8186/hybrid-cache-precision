#!/usr/bin/env bash
# 环境探针：输出硬件/驱动/软件版本，写入每个 run 的 provenance（env_probe.txt）
set -uo pipefail

echo "=== date ==="
date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo "n/a"

echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>&1 || echo "no nvidia-smi"

echo "=== torch ==="
python -c "import torch; print('version', torch.__version__); print('cuda', torch.version.cuda); print('capability', torch.cuda.get_device_capability())" 2>&1 || echo "no torch"

echo "=== device ==="
python -c "import torch; print(torch.cuda.get_device_name(0))" 2>&1 || echo "n/a"

echo "=== vllm ==="
python -c "import vllm; print('vllm', vllm.__version__)" 2>&1 || echo "no vllm"

echo "=== git head ==="
git rev-parse HEAD 2>&1 || echo "n/a"

echo "=== pip freeze (top) ==="
pip freeze 2>/dev/null | head -80 || echo "no pip"
