#!/usr/bin/env bash
# 环境自检：本地或租机运行最终实验前的会话校验
# 租机注意：实例可能带残留进程 / 旧 driver / 脏 VRAM，driver 跨租期漂移常见。
set -uo pipefail

echo "=== GPU 识别 ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>&1 || { echo "ERROR: 无 nvidia-smi"; exit 1; }

echo
echo "=== 残留进程检查（应无占用显存的老进程）==="
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv 2>&1 || true

echo
echo "=== Python / vLLM ==="
python --version 2>&1
python -c "import vllm; print('vllm', vllm.__version__)" 2>&1 || { echo "ERROR: vLLM 未安装或不可导入"; exit 1; }
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" 2>&1 || true

echo
echo "=== 与锁定版本对比（configs/env/）==="
echo "TODO: 解析 configs/env/remote_5090.yaml 的 driver/CUDA/torch 版本并断言匹配"

echo
echo "=== vLLM 自测 ==="
echo "TODO: 起一个最小 serving self-test（一个请求）确认引擎可运行"

echo
echo "环境自检完成（TODO 项实现后成为正式门禁）。"
