#!/usr/bin/env bash
# 归档 headline 原始运行：zip+hash，指针+校验和写入 results/
# 用法: ./scripts/archive.sh <experiment_name>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXP_NAME="${1:?用法: ./scripts/archive.sh <experiment_name>}"
RUN_DIR="$ROOT/experiments/$EXP_NAME"

[ -d "$RUN_DIR" ] || { echo "ERROR: 运行不存在: $RUN_DIR"; exit 1; }

TARBALL="$ROOT/experiments/${EXP_NAME}.tar.gz"
tar -czf "$TARBALL" -C "$ROOT/experiments" "$EXP_NAME"
SHA="$(sha256sum "$TARBALL" | cut -d' ' -f1)"
SIZE="$(du -h "$TARBALL" | cut -f1)"

mkdir -p "$ROOT/results"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $EXP_NAME sha256=$SHA size=$SIZE" >> "$ROOT/results/_archive_index.txt"

echo "归档完成: $TARBALL"
echo "  sha256 = $SHA"
echo "  已登记到 results/_archive_index.txt"
echo "提示：对 headline 运行，建议 git tag + 上传到持久存储（Zenodo/HF），并把链接写入 results/_provenance.jsonl"
