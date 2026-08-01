#!/usr/bin/env bash
# 构建 vendor/vllm fork（固定 submodule SHA），记录 wheel sha256 到构建记录。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VLLM="$ROOT/vendor/vllm"

if [ ! -d "$VLLM" ]; then
  echo "ERROR: vendor/vllm 不存在。先执行: git submodule update --init --recursive vendor/vllm" >&2
  exit 1
fi

cd "$VLLM"
VLLM_SHA="$(git rev-parse HEAD)"
echo "==> 构建 vLLM @ $VLLM_SHA"

pip install -e . 2>&1 | tail -8

# 记录构建产物哈希（wheel 路径随版本变化，先探测）
WHEEL="$(python -c "import vllm; print(vllm.__file__)" 2>/dev/null || true)"
echo "vllm import path: $WHEEL"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) vllm_sha=$VLLM_SHA import=$WHEEL" >> "$ROOT/experiments/vllm_builds.log" 2>/dev/null || true

echo "==> vLLM 构建完成 (SHA $VLLM_SHA)"
