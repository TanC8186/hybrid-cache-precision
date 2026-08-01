"""LongBench 下游任务评测。长上下文任务必须路由到 5090/7B，禁止用 4060/3B 数字顶替。

vendor/eval 以 submodule 固定 LongBench 评测脚本与数据 revision。
"""
from __future__ import annotations

import argparse


def run_longbench(experiment: str) -> None:
    raise NotImplementedError("TODO: 跑 LongBench 子集（sampling/summarization 等）→ metrics(jsonl)")


def main() -> None:
    parser = argparse.ArgumentParser(description="LongBench eval")
    parser.add_argument("--config", default="configs/eval/quality.yaml")
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    run_longbench(args.experiment)


if __name__ == "__main__":
    main()
