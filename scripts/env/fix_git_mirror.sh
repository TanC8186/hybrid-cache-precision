#!/usr/bin/env bash
# 设置 git 全局 URL 重写（github → ghfast 镜像），供 vLLM cmake 外部项目克隆用。
set -uo pipefail

# 清理卡住的构建进程
pkill -f 'pip install -e' 2>/dev/null
pkill -f 'git clone' 2>/dev/null
pkill -f 'cmake' 2>/dev/null
pkill -f 'setup_5090' 2>/dev/null
sleep 2

# 设置 URL 重写（github.com → ghfast.top 镜像）
git config --global url."https://ghfast.top/https://github.com/".insteadOf "https://github.com/"

echo "=== git rewrite 配置 ==="
git config --global --get-regexp 'url\..*insteadof'

echo "=== 测试 ghfast 对 vllm-flash-attn 的 clone 能力 ==="
timeout 30 git ls-remote https://github.com/vllm-project/vllm-flash-attn 2>&1 | head -3

echo "=== done ==="
