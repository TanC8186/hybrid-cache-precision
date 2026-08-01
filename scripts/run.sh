#!/usr/bin/env bash
# 唯一实验运行入口：启动实验并固化 provenance bundle
#
# 用法: ./scripts/run.sh <experiment_name>
#   <experiment_name> 对应 configs/experiments/<name>.yaml
#
# 每次运行向 experiments/<name>/ 产出：
#   resolved.yaml + .sha256    解析后的有效配置
#   git_commit / vllm_submodule_status / git_dirty_stat   代码状态
#   env_probe.txt              环境探针（nvidia-smi / torch / vllm / pip freeze）
#   seeds.txt                  seed 清单
set -euo pipefail

EXP_NAME="${1:?用法: ./scripts/run.sh <experiment_name>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="$ROOT/configs/experiments/$EXP_NAME.yaml"
RUN_DIR="$ROOT/experiments/$EXP_NAME"

[ -f "$CFG" ] || { echo "ERROR: 配置不存在: $CFG"; exit 1; }

# --- 0) clean-tree 门禁（commit-before-run 硬规则）---
DIRTY="$(git -C "$ROOT" status --porcelain)"
if [ -n "$DIRTY" ]; then
  if [ "${STRICT:-1}" = "1" ]; then
    echo "ERROR: git 树不干净。先 commit 或丢弃未提交改动再运行。" >&2
    git -C "$ROOT" status --short >&2
    exit 1
  fi
  echo "WARN: git 树不干净（非 strict 模式），dirty diff 将记入 provenance。"
fi

mkdir -p "$RUN_DIR"

# --- 1) 固化解析后的有效配置 ---
# TODO: 接入配置加载器（解析 env/model/quantization 合并后生成 resolved.yaml）
# 目前先复制源配置，加载器就绪后替换。
cp "$CFG" "$RUN_DIR/resolved.yaml"
sha256sum "$RUN_DIR/resolved.yaml" > "$RUN_DIR/resolved.yaml.sha256"

# --- 2) 固化代码状态 ---
git -C "$ROOT" rev-parse HEAD > "$RUN_DIR/git_commit"
git -C "$ROOT" submodule status > "$RUN_DIR/vllm_submodule_status" 2>/dev/null || true
if [ -n "$DIRTY" ]; then
  git -C "$ROOT" diff --stat > "$RUN_DIR/git_dirty_stat" || true
fi
git -C "$ROOT" status --short > "$RUN_DIR/git_status" 2>/dev/null || true

# --- 3) 固化环境状态 ---
bash "$ROOT/scripts/env/probe.sh" > "$RUN_DIR/env_probe.txt" 2>&1 || true

# --- 4) 固化 seed 清单 ---
grep -E "^\s*(seed|seeds):" "$CFG" > "$RUN_DIR/seeds.txt" || true

echo "=== Provenance bundle 已写入 $RUN_DIR ==="
echo "  - config:    $(cat "$RUN_DIR/resolved.yaml.sha256" | cut -d' ' -f1)"
echo "  - git:       $(cat "$RUN_DIR/git_commit")"
echo "  - seeds:     $(cat "$RUN_DIR/seeds.txt | tr '\n' ' ')"

echo
echo "TODO: 在此启动实际实验命令（bench / eval / memory），并确保输出写入 $RUN_DIR/"
