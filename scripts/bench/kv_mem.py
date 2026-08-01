"""读取 vLLM 分配器的实际 KV 占用，产出 memory-vs-bit-width 曲线。

通过 vLLM cache engine 的 num_blocks × block_size（或 v1 KV cache 管理器）
在固定 workload 后读取实际分配块数，避免只靠配置算术。
"""
from __future__ import annotations

import argparse


def kv_memory_curve() -> None:
    raise NotImplementedError(
        "TODO: 在 vendor/vllm 内加最小 sideband hook（或读 KV cache 管理器状态）"
        "→ 对不同 bit-width 记录实际 KV 字节 → 输出到 results/"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="KV memory vs bit-width curve")
    parser.add_argument("--config", default="configs/quantization/*.yaml")
    args = parser.parse_args()
    kv_memory_curve()


if __name__ == "__main__":
    main()
