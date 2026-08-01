#!/usr/bin/env bash
# 下载固定版本数据并校验哈希（溯源见 data/MANIFEST.yaml）。
# 数据文件 gitignored，但 sha256 记录在 MANIFEST —— 审稿人可据此验证。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT/data"

mkdir -p "$DATA_DIR"

echo "==> TODO: 按 configs/datasets/*.yaml 下载并固定 revision 的数据"
echo "    - HF 数据集用固定 revision：datasets.load_dataset(..., revision='<fixed>')"
echo "    - 下载后更新 data/MANIFEST.yaml 的 sha256"
echo
echo "当前 data/ 内容："
find "$DATA_DIR" -type f | head -20 || true
echo
echo "提醒：serving trace（如 ShareGPT）常带重分发限制，确认 license 后再决定是否随 artifact 提供。"
