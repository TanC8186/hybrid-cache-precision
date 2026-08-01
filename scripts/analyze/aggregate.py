"""聚合 metrics(jsonl) → results/tables/ 与 results/figures/。

唯一允许产出论文图表的地方（CLAUDE.md 铁律）：
- 只读取 results/_provenance.jsonl 登记过的 run 集（防止混入不同 commit / env 的数字）
- 折叠 num_seeds 重复 → mean/std；serving 指标加 p50/p99
- 输出表 csv / latex，图 pdf/svg
- notebook 仅探索，永不为提交图表来源

用法: python -m scripts.analyze.aggregate [--runs run1,run2,...]
"""
from __future__ import annotations

import argparse


def aggregate(runs: list[str] | None = None) -> None:
    raise NotImplementedError(
        "TODO: 读取 experiments/<run>/metrics.jsonl（含 env_id 校验），"
        "折叠 seeds → mean/std/p50/p99 → 写 results/tables|figures"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="aggregate metrics into results/")
    parser.add_argument("--runs", type=str, help="逗号分隔的 run 列表（默认：provenance 登记的全部）")
    args = parser.parse_args()
    runs = args.runs.split(",") if args.runs else None
    aggregate(runs)


if __name__ == "__main__":
    main()
