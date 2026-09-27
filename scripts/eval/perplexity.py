"""困惑度评测：PPL vs 上下文长度（Wikitext / PG19），量化方法的核心质量指标。

本模块的 harness 封装未实现。论文使用的是 chunk-level C4/PG19 harness：
scripts/exp/hybrid_premise.py 采集，scripts/eval/analyze_ppl_stacking.py 等分析；
论文报告值为 chunk=128、三个 dataset seed。
"""
from __future__ import annotations

import argparse


def run_perplexity(experiment: str) -> None:
    raise NotImplementedError(
        "TODO: 调 lm-eval-harness 跑 wikitext/pg19，按 config 路由到 remote_5090，"
        "输出 metrics(jsonl)。未实现：论文 PPL 由 scripts/exp/hybrid_premise.py "
        "采集、scripts/eval/analyze_ppl_stacking.py 等分析。"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="perplexity eval")
    parser.add_argument("--config", default="configs/eval/quality.yaml")
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    run_perplexity(args.experiment)


if __name__ == "__main__":
    main()
