"""困惑度评测：PPL vs 上下文长度（Wikitext / PG19），量化方法的核心质量指标。

评测 harness 来自 vendor/eval（submodule 固定版本），本模块只做封装与路由：
- 读取 configs/eval/quality.yaml 的 env_routing
- 断言上下文长度在 configs/env/*.yaml 的预算内（context_budget_guard）
- 写 metrics(jsonl)：每 (model, quantization, seq_len, seed) 一行
"""
from __future__ import annotations

import argparse


def run_perplexity(experiment: str) -> None:
    raise NotImplementedError(
        "TODO: 调 lm-eval-harness（vendor/eval）跑 wikitext/pg19，"
        "按 config 路由到 remote_5090，输出 metrics(jsonl)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="perplexity eval")
    parser.add_argument("--config", default="configs/eval/quality.yaml")
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    run_perplexity(args.experiment)


if __name__ == "__main__":
    main()
