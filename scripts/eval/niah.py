"""Needle-in-a-haystack 检索评测：深度 × 长度的头部长上下文结果。

数据由 scripts/gen_synthetic_retrieval.py 生成（seed 化），评测读取 data/niah/。
"""
from __future__ import annotations

import argparse


def run_niah(experiment: str) -> None:
    raise NotImplementedError(
        "TODO: 加载 data/niah 合成数据，测各 (depth, length) 检索准确率 → metrics(jsonl)。"
        "未实现：NIAH 由 scripts/eval/run_niah_fixed.sh 采集、"
        "scripts/eval/analyze_niah_fixed.py 分析。"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="NIAH retrieval eval")
    parser.add_argument("--config", default="configs/eval/quality.yaml")
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    run_niah(args.experiment)


if __name__ == "__main__":
    main()
