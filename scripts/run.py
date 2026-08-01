"""实验运行/网格消融编排器。

用法:
    python scripts/run.py --run configs/experiments/xxx.yaml        # 单实验
    python scripts/run.py --sweep --config 'configs/quantization/*.yaml'  # 网格消融

网格消融输出 results/ablations/<sweep_name>.csv，固定 schema：
    granularity, bits, seed, ppl, memory_gb, throughput, ttft_p99, tpot_p99, ...

论文的粒度×位数 Pareto 权衡曲线从该 csv 绘制（scripts/analyze）。
"""
from __future__ import annotations

import argparse


def run_experiment(cfg_path: str) -> None:
    """运行单个实验：先做 schema 校验，再调用 scripts/run.sh，最后汇总。"""
    raise NotImplementedError("TODO: 接入配置加载器 + scripts/run.sh")


def run_sweep(config_glob: str) -> None:
    """对 config 网格逐项运行并汇总为 ablations csv。"""
    raise NotImplementedError("TODO: 遍历网格（granularity × bits × seed），调用 run_experiment")


def main() -> None:
    parser = argparse.ArgumentParser(description="run / sweep experiments")
    parser.add_argument("--run", type=str, help="单实验 config 路径")
    parser.add_argument("--sweep", action="store_true", help="网格消融模式")
    parser.add_argument("--config", default="configs/quantization/*.yaml", help="网格 glob")
    args = parser.parse_args()

    if args.run:
        run_experiment(args.run)
    elif args.sweep:
        run_sweep(args.config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
