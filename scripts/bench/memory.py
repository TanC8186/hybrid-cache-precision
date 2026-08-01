"""内存测量：KV cache 节省的实证（而非算术推算）。

方法：
- torch.cuda.memory_snapshot / max_memory_allocated + nvidia-smi 采样
- 输出 KV-vs-weights 分解、memory-vs-seqlen 曲线、memory-vs-bits 曲线

铁律：所有内存证据只从 5090/7B 环境出（8G 4060 上 KV 占比太小，不可信）。
"""
from __future__ import annotations

import argparse


def measure_memory(experiment: str) -> None:
    raise NotImplementedError(
        "TODO: 起 vLLM 引擎（量化 KV）→ 固定 workload → 读 allocator num_blocks×block_size"
        "→ 计算 KV 字节与 weights 分解 → 输出曲线到 results/（仅 remote_5090）"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="measure KV cache memory")
    parser.add_argument("--config", default="configs/bench/throughput.yaml")
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    measure_memory(args.experiment)


if __name__ == "__main__":
    main()
