#!/usr/bin/env bash
# 远端 7B 冒烟：正式大规模运行前，验证精确的 7B 配置能在 5090 上跑通。
# 在租机实例上执行；输出记录到 experiments/<name>/smoke/ 供 provenance。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> 环境自检"
bash "$ROOT/env_check.sh"

echo
echo "==> 7B 配置冒烟（TODO 实现）"
echo "  - 加载 configs/models/remote_7b.yaml + configs/experiments/template.yaml"
echo "  - 步骤: forward 一次 + 至少服务一个请求 + 跑单 seed PPL"
echo "  - 输出写到 experiments/<name>/smoke/"

echo
echo "冒烟通过后才能投入正式实验（防止在 7B 全量跑完才发现配置错误）。"
