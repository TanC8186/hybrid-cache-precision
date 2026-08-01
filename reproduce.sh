#!/usr/bin/env bash
# 一键复现入口（供 MLSys Artifact Evaluation）
# 流程：clone → 构建锁定 submodule → 下载并校验数据 → 跑一个 canonical 实验 → 对照期望指标区间
#
# 用法（审稿人侧）: ./reproduce.sh <experiment_name>
set -euo pipefail

EXP_NAME="${1:-template}"
ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "==> [1/5] 校验 git 状态（应提交了当前实验相关代码）"
git -C "$ROOT" status --short | head -20

echo "==> [2/5] 构建锁定的 vLLM submodule"
git -C "$ROOT" submodule update --init --recursive vendor/vllm
bash "$ROOT/scripts/build_vllm.sh"

echo "==> [3/5] 下载并校验数据（溯源见 data/MANIFEST.yaml）"
bash "$ROOT/scripts/fetch_data.sh"

echo "==> [4/5] 运行 canonical 实验（固化 provenance）"
bash "$ROOT/scripts/run.sh" "$EXP_NAME"

echo "==> [5/5] 对照存储的期望指标区间"
echo "TODO: 从 results/_provenance.jsonl 读取该实验的期望范围，校验本次输出是否落在区间内"

echo
echo "复现完成。若与论文数字不一致，检查 experiments/$EXP_NAME/env_probe.txt 与 git_commit。"
