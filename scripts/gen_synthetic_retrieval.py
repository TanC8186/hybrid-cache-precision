"""生成 seed 化的 NIAH / passkey 检索数据（StreamingLLM / SnapKV / GEAR 都报的头部结果）。

输出写入 data/ 并哈希进 data/MANIFEST.yaml。
深度、长度、针文本全部可配置且 seed 化，保证可复现。
"""
from __future__ import annotations

import argparse
from pathlib import Path


def generate(
    output_dir: Path,
    *,
    depths: list[int],
    lengths: list[int],
    seed: int,
) -> None:
    raise NotImplementedError(
        "TODO: 生成 NIAH 数据——每个 (depth, length) 合成一段文档，"
        "插入 needle 文本，记录检索 query 与答案，输出 jsonl + 更新 MANIFEST"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="generate seeded NIAH retrieval data")
    parser.add_argument("--output", default="data/niah", help="输出目录")
    parser.add_argument("--depths", type=int, nargs="+", default=[0, 25, 50, 75, 100])
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192, 16384, 32768])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate(Path(args.output), depths=args.depths, lengths=args.lengths, seed=args.seed)
    print("NIAH 数据已生成，记得更新 data/MANIFEST.yaml 的 sha256")


if __name__ == "__main__":
    main()
